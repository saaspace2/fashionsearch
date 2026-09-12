"""
FashionSearch — upload a photo, find the garment.

Three tabs:

    Search     upload an image, call the serving endpoint, rank the catalogue
    Dashboard  which model version is live, and how well it scores
    Drift      whether incoming data still resembles what was evaluated

Run locally:
    pip install -r app/requirements.txt
    export DATABRICKS_HOST=... DATABRICKS_TOKEN=... WAREHOUSE_ID=...
    streamlit run app/app.py
"""

import io

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

import databricks_client as db

st.set_page_config(page_title="FashionSearch", page_icon="🔍", layout="wide")

CATALOG = db.CATALOG


# ----------------------------------------------------------------- loading

@st.cache_data(ttl=600, show_spinner="Loading the catalogue…")
def catalogue() -> pd.DataFrame:
    """
    Every product with its embedding.

    Cached for ten minutes. A few thousand 128-float vectors is a couple of
    megabytes — small enough to hold in memory, and fetching it per search
    would make every query slow for no reason.
    """
    df = db.sql(f"""
        SELECT e.product_id, e.category, e.embedding, p.image_path
        FROM {CATALOG}.silver.product_embeddings e
        JOIN {CATALOG}.bronze.products p USING (product_id)
    """)
    if df.empty:
        return df
    df["embedding"] = df["embedding"].apply(
        lambda v: np.asarray(v if isinstance(v, list) else eval(v), dtype=np.float32))
    return df


@st.cache_data(ttl=300)
def latest_metrics() -> pd.DataFrame:
    return db.sql(f"""
        SELECT slice_dim, slice_value, n_queries,
               round(recall_at_20, 3) AS recall_at_20,
               round(ndcg_at_20, 3)   AS ndcg_at_20,
               round(mrr, 3)          AS mrr,
               model_version, evaluated_at, gate_passed
        FROM {CATALOG}.gold.retrieval_metrics
        WHERE evaluated_at = (SELECT max(evaluated_at)
                              FROM {CATALOG}.gold.retrieval_metrics)
        ORDER BY slice_dim, n_queries DESC
    """)


@st.cache_data(ttl=300)
def metric_history() -> pd.DataFrame:
    return db.sql(f"""
        SELECT evaluated_at, model_version, n_queries,
               round(recall_at_20, 4) AS recall_at_20,
               round(ndcg_at_20, 4)   AS ndcg_at_20,
               gate_passed
        FROM {CATALOG}.gold.retrieval_metrics
        WHERE slice_dim = 'overall'
        ORDER BY evaluated_at
    """)


@st.cache_data(ttl=300)
def serving_history() -> pd.DataFrame:
    return db.sql(f"""
        SELECT checked_at, model_name, alias, mode, endpoint_name,
               embedding_dim, p95_ms, note
        FROM {CATALOG}.monitoring.serving_checks
        ORDER BY checked_at DESC LIMIT 10
    """)


@st.cache_data(ttl=600)
def thumbnail(path: str, size: int = 150):
    if not path or path.startswith("kaggle://"):
        return None
    raw = db.read_volume_file(path)
    if not raw:
        return None
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        img.thumbnail((size, size))
        return img
    except Exception:
        return None


# ------------------------------------------------------------------ search

def rank_against_catalogue(query_vector, category, top_k, restrict):
    cat = catalogue()
    if cat.empty:
        return pd.DataFrame()

    vectors = np.stack(cat.embedding.values)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    query_vector = query_vector / np.linalg.norm(query_vector)

    mask = np.ones(len(cat), dtype=bool)
    if restrict and category and category in set(cat.category):
        mask = (cat.category == category).values

    scores = vectors[mask] @ query_vector
    order = np.argsort(-scores)[:top_k]
    hits = cat[mask].iloc[order].copy()
    hits["score"] = scores[order]
    return hits


def search(image_bytes: bytes, top_k: int, restrict: bool):
    """One result set per garment the detector found."""
    items = db.detect_and_embed(image_bytes)
    for item in items:
        item["hits"] = rank_against_catalogue(
            item["embedding"], item.get("detected_category"), top_k, restrict)
    return items


def crop_preview(image_bytes: bytes, box):
    if not box or any(v is None for v in box):
        return None
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return img.crop(tuple(float(v) for v in box))
    except Exception:
        return None


# -------------------------------------------------------------------- tabs

st.title("FashionSearch")

tab_search, tab_dashboard, tab_drift = st.tabs(["Search", "Dashboard", "Drift"])

with tab_search:
    left, right = st.columns([1, 3])

    with left:
        uploaded = st.file_uploader("Upload an outfit photo",
                                    type=["jpg", "jpeg", "png", "webp"])
        top_k = st.slider("Results", 4, 24, 12, step=4)
        restrict = st.checkbox(
            "Restrict to the detected category", value=True,
            help="Searching only jackets rather than everything makes results "
                 "both faster and better — the system cannot return a bag when "
                 "you meant a jacket.")
        if uploaded:
            st.image(uploaded, caption="Your photo", use_container_width=True)

    with right:
        if not uploaded:
            st.info("Upload a photo to search the catalogue.")
        else:
            try:
                with st.spinner("Calling the serving endpoint…"):
                    items = search(uploaded.getvalue(), top_k, restrict)
            except Exception as exc:
                st.error(f"Search failed: {type(exc).__name__}: {exc}")
                st.caption("Check the endpoint is running, and that "
                           "WAREHOUSE_ID is set for the catalogue query.")
                st.stop()

            if not items:
                st.warning("The endpoint returned nothing.")
                st.stop()

            if len(items) == 1 and items[0].get("used_whole_image"):
                # Worth surfacing: on the eval set this happened to 31% of
                # queries, and those score far worse than detected ones.
                st.warning(
                    "No garment detected — the whole image was embedded. "
                    "Results are usually weaker. A tighter crop helps.")
            else:
                found = [i.get("detected_category") or "?" for i in items]
                st.success(f"Found {len(items)} garment"
                           f"{'s' if len(items) > 1 else ''}: {', '.join(found)}")

            st.caption("Scores run 1.0 (identical) to 0.0 (unrelated). "
                       "Above roughly 0.75 is usually the same garment.")

            for n, item in enumerate(items):
                label = item.get("detected_category") or "whole image"
                score = item.get("detector_score")
                area = item.get("area_frac")

                header = f"**{label}**"
                if score:
                    header += f" · confidence {score:.2f}"
                if area:
                    header += f" · fills {area:.0%} of the frame"
                st.markdown(f"### {n + 1}. {header}")

                preview_col, results_col = st.columns([1, 4])
                with preview_col:
                    crop = crop_preview(uploaded.getvalue(), item.get("box"))
                    if crop:
                        st.image(crop, caption="what was searched",
                                 use_container_width=True)

                with results_col:
                    hits = item.get("hits")
                    if hits is None or hits.empty:
                        st.warning("Nothing matched in that category.")
                        continue
                    for start in range(0, len(hits), 4):
                        row = st.columns(4)
                        for col, (_, hit) in zip(row,
                                                 hits.iloc[start:start + 4].iterrows()):
                            with col:
                                img = thumbnail(hit.image_path)
                                if img:
                                    st.image(img, use_container_width=True)
                                else:
                                    st.caption("(image not available)")
                                st.caption(f"**{hit.score:.3f}** · {hit.category}")
                                st.caption(hit.product_id)
                st.divider()

with tab_dashboard:
    name, state = db.endpoint_state()
    metrics = latest_metrics()

    if metrics.empty:
        st.warning("No evaluations yet. Run the pipeline first.")
    else:
        overall = metrics[metrics.slice_dim == "overall"].iloc[0]
        cards = st.columns(5)
        cards[0].metric("Model version", f"v{overall.model_version}")
        cards[1].metric("Recall@20", overall.recall_at_20)
        cards[2].metric("NDCG@20", overall.ndcg_at_20)
        cards[3].metric("MRR", overall.mrr)
        cards[4].metric("Gate", "passed" if str(overall.gate_passed).lower() == "true"
                        else "blocked")

        st.caption(f"Endpoint **{name}** — "
                   f"{'ready' if state == 'READY' else state or 'not deployed'} · "
                   f"evaluated {overall.evaluated_at}")

        st.subheader("Per slice")
        st.caption(
            "The headline figure above is a mean over queries, so the biggest "
            "categories dominate it. Problems show up here, not there.")
        slices = metrics[metrics.slice_dim != "overall"]
        st.dataframe(slices.drop(columns=["model_version", "evaluated_at",
                                          "gate_passed"]),
                     use_container_width=True, hide_index=True)

        weak = slices[pd.to_numeric(slices.recall_at_20, errors="coerce")
                      < float(overall.recall_at_20) * 0.85]
        if not weak.empty:
            st.warning("Slices well below the overall average — look here first:")
            st.dataframe(weak[["slice_dim", "slice_value", "n_queries",
                               "recall_at_20"]],
                         use_container_width=True, hide_index=True)

        history = metric_history()
        if len(history) >= 2:
            st.subheader("Trend")
            chart = history.copy()
            chart["evaluated_at"] = pd.to_datetime(chart["evaluated_at"])
            st.line_chart(chart.set_index("evaluated_at")
                          [["recall_at_20", "ndcg_at_20"]])

        with st.expander("Serving history"):
            st.dataframe(serving_history(), use_container_width=True,
                         hide_index=True)

with tab_drift:
    st.caption(
        "Quality metrics describe a **fixed** eval set. Drift describes whether "
        "incoming data still resembles it. The second can move while the first "
        "looks perfect — and it moves first.")

    try:
        snapshots = db.sql(f"""
            SELECT captured_at, model_version, payload
            FROM {CATALOG}.monitoring.drift_snapshots
            ORDER BY captured_at DESC LIMIT 2
        """)
    except Exception as exc:
        st.error(f"Could not read drift snapshots: {exc}")
        snapshots = pd.DataFrame()

    if len(snapshots) < 2:
        st.info("Drift needs two snapshots to compare. Run notebook 11 after "
                "the next pipeline run.")
    else:
        import json
        import sys
        sys.path.insert(0, "src")
        from fashionsearch import drift as drift_lib

        current = json.loads(snapshots.iloc[0].payload)
        previous = json.loads(snapshots.iloc[1].payload)

        rows = drift_lib.summarise(previous, current)
        psi_size = drift_lib.population_stability_index(
            previous.get("size_band_counts", {}),
            current.get("size_band_counts", {}))
        rows.append({"signal": "query_size_bands", "metric": "PSI",
                     "value": round(psi_size, 4),
                     "verdict": drift_lib.psi_verdict(psi_size),
                     "hint": "how large the detected garment is in the photo"})

        table = pd.DataFrame(rows)
        cards = st.columns(min(len(table), 4))
        for col, (_, r) in zip(cards, table.iterrows()):
            col.metric(r["signal"].replace("_", " "), r["value"], r["verdict"])

        st.dataframe(table, use_container_width=True, hide_index=True)

        moved = table[table.verdict != "stable"]
        if moved.empty:
            st.success("No signal moved beyond its noise band.")
        else:
            st.warning("Signals that moved — worth re-running the evaluation "
                       "and comparing:")
            for _, r in moved.iterrows():
                st.write(f"**{r['signal']}** — {r['verdict']}. {r['hint']}")

        with st.expander("How to read this"):
            st.markdown("""
| If this moves | It probably means |
|---|---|
| Recall@20 falls, drift stable | The model got worse. Compare versions on the Dashboard tab. |
| Drift moves, Recall@20 stable | The inputs changed but the eval set did not — your eval set is going stale. |
| Both move | A genuine distribution shift. Retrain on newer data. |
| Category mix PSI high | New product types, or a supplier added or dropped. |
| Fallback rate up | Photos got harder, or the detector is degrading. |

The second row is the one people miss. A frozen eval set cannot tell you it has
become unrepresentative — it keeps reporting the same comfortable number while
production diverges from it.
""")
