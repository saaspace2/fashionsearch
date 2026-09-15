"""
Everything the app needs from Databricks, in one place.

Three different services, deliberately kept behind one module so the UI never
has to know which is which:

    SQL warehouse     reading Delta tables — metrics, drift, the catalogue
    Serving endpoint  turning an uploaded photo into an embedding
    Volumes           fetching product images to display

Credentials come from Streamlit secrets when running as a Databricks App, and
from the environment otherwise, so the same code runs both ways.
"""

from __future__ import annotations

import base64
import os
from functools import lru_cache

import numpy as np
import pandas as pd


def _setting(name: str, default: str = "") -> str:
    """Streamlit secrets first, environment second."""
    try:
        import streamlit as st
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return os.environ.get(name, default)


CATALOG = _setting("CATALOG", "fashion_dev")
ENDPOINT = _setting("ENDPOINT", "fashion-encoder-endpoint")
WAREHOUSE_ID = _setting("WAREHOUSE_ID")


@lru_cache(maxsize=1)
def workspace():
    from databricks.sdk import WorkspaceClient

    host = _setting("DATABRICKS_HOST")
    token = _setting("DATABRICKS_TOKEN")
    if host and token:
        return WorkspaceClient(host=host.rstrip("/"), token=token)
    # Running as a Databricks App: credentials are injected.
    return WorkspaceClient()


# --------------------------------------------------------------------- SQL

def sql(query: str) -> pd.DataFrame:
    """
    Run a query and return a DataFrame.

    Uses the Statement Execution API rather than a SQL driver, so the app needs
    only the SDK — no ODBC, no extra binary to install on whatever machine this
    runs on.
    """
    from databricks.sdk.service.sql import StatementState

    if not WAREHOUSE_ID:
        raise RuntimeError(
            "WAREHOUSE_ID is not set. Find it under SQL Warehouses in Databricks "
            "— the ID is in the connection details, or in the URL after "
            "/warehouses/.")

    w = workspace()
    resp = w.statement_execution.execute_statement(
        warehouse_id=WAREHOUSE_ID, statement=query, wait_timeout="50s")

    while resp.status.state in (StatementState.PENDING, StatementState.RUNNING):
        resp = w.statement_execution.get_statement(resp.statement_id)

    if resp.status.state != StatementState.SUCCEEDED:
        raise RuntimeError(f"query failed: {resp.status.error}")

    if not resp.manifest or not resp.manifest.schema.columns:
        return pd.DataFrame()

    columns = [c.name for c in resp.manifest.schema.columns]
    rows = (resp.result.data_array or []) if resp.result else []
    return pd.DataFrame(rows, columns=columns)


# ---------------------------------------------------------------- endpoint

def detect_and_embed(image_bytes: bytes) -> list:
    """
    Send one image to the endpoint. Get back every garment it found.

    Returns a list of dicts, best first:
        {embedding, detected_category, detector_score, area_frac,
         box, used_whole_image}

    A list rather than one item because an outfit photo contains several things
    a user might be searching for. Returning only the largest would mean someone
    who photographed a full outfit could search for the jacket but never the
    trousers.

    Three response shapes are handled, because the endpoint may be running any
    of three model versions and the UI should not care:
      * multi-item combined model  — one row per garment
      * single-item combined model — one row per image
      * encoder only               — a bare vector, no detection
    """
    w = workspace()
    payload = [{"image": base64.b64encode(image_bytes).decode()}]
    response = w.serving_endpoints.query(name=ENDPOINT, dataframe_records=payload)
    predictions = response.predictions

    if not predictions:
        return []

    items = []
    for p in predictions:
        if isinstance(p, dict) and "embedding" in p:
            items.append({
                "embedding": np.asarray(p["embedding"], dtype=np.float32),
                "detected_category": p.get("detected_category"),
                "detector_score": p.get("detector_score"),
                "area_frac": p.get("area_frac"),
                "box": [p.get("x1"), p.get("y1"), p.get("x2"), p.get("y2")]
                       if p.get("x2") is not None else None,
                "used_whole_image": p.get("used_whole_image", False),
            })
        else:
            vector = list(p.values()) if isinstance(p, dict) else list(p)
            items.append({
                "embedding": np.asarray(vector, dtype=np.float32),
                "detected_category": None, "detector_score": None,
                "area_frac": None, "box": None, "used_whole_image": True,
            })
    return items


def embed(image_bytes: bytes) -> dict:
    """The first detected garment only. Kept for callers that want one."""
    items = detect_and_embed(image_bytes)
    return items[0] if items else {}


def endpoint_state() -> tuple:
    """(name, state) — or (name, None) when it does not exist."""
    try:
        e = workspace().serving_endpoints.get(ENDPOINT)
        return ENDPOINT, str(e.state.ready) if e.state else "unknown"
    except Exception:
        return ENDPOINT, None


# ------------------------------------------------------------------ logging

def _sql_str(value) -> str:
    """Quote a value for SQL, or NULL. Single quotes doubled to escape them."""
    if value is None:
        return "NULL"
    return "'" + str(value).replace("'", "''") + "'"


def _sql_array(values) -> str:
    if not values:
        return "array()"
    return "array(" + ", ".join(_sql_str(v) for v in values) + ")"


def log_search_event(image_bytes: bytes, items: list, latency_ms: float,
                     session_id: str, model_version: str = "") -> str | None:
    """
    Record one search: the photo, what was detected, and what was returned.

    WHY THIS MATTERS MORE THAN IT LOOKS
    -----------------------------------
    Every query somebody runs here is a free example of what production
    actually looks like. The evaluation set is built from dataset anchors —
    pre-cropped single garments, 976 of 1000 of them 'large'. Real uploads are
    full-outfit photos where a jacket fills 15% of the frame.

    Those are precisely the queries the eval set lacks, and until now the app
    discarded them the moment the next photo was uploaded.

    Returns the event_id so a later click can be attached to it, or None if
    logging is not configured — a failure here must never break a search.
    """
    import uuid
    from datetime import datetime, timezone

    if not WAREHOUSE_ID:
        return None

    event_id = uuid.uuid4().hex
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    remote = f"/Volumes/{CATALOG}/raw/query_images/live/{stamp}/{event_id}.jpg"

    try:
        w = workspace()
        w.files.upload(remote, io.BytesIO(image_bytes), overwrite=True)
    except Exception as exc:
        print(f"could not store the query image: {exc}")
        remote = None

    first = items[0] if items else {}
    results = []
    for item in items:
        hits = item.get("hits")
        if hits is not None and not hits.empty:
            results.extend(str(p) for p in hits.product_id.tolist())

    fallback = "whole_image" if first.get("used_whole_image") else None

    try:
        sql(f"""
            INSERT INTO {CATALOG}.bronze.search_events (
                event_id, event_ts, session_id, query_image, selected_category,
                model_version, index_version, results_shown, clicked,
                added_to_cart, latency_ms, n_results, reformulated,
                query_condition, fallback_fired)
            VALUES (
                {_sql_str(event_id)}, current_timestamp(), {_sql_str(session_id)},
                {_sql_str(remote)}, {_sql_str(first.get("detected_category"))},
                {_sql_str(model_version)}, NULL,
                {_sql_array(results)}, array(), array(),
                {float(latency_ms)}, {len(results)}, false,
                'live_upload', {_sql_str(fallback)})
        """)
        return event_id
    except Exception as exc:
        # Never let logging break a search. A missing row is a small loss; a
        # failed search in front of a user is not.
        print(f"could not log the search event: {exc}")
        return None


def log_click(event_id: str, product_id: str) -> None:
    """
    Record that somebody clicked a result.

    This is what closes the feedback loop. A clicked result is a correctly
    labelled positive pair, produced by a human who had every incentive to get
    it right — and the results shown above it and passed over are hard
    negatives, worth far more than random ones the model solved weeks ago.
    """
    if not event_id or not WAREHOUSE_ID:
        return
    try:
        sql(f"""
            UPDATE {CATALOG}.bronze.search_events
            SET clicked = array_union(clicked, array({_sql_str(product_id)}))
            WHERE event_id = {_sql_str(event_id)}
        """)
    except Exception as exc:
        print(f"could not log the click: {exc}")


# ----------------------------------------------------------------- volumes

def read_volume_file(path: str) -> bytes | None:
    try:
        response = workspace().files.download(path)
        return response.contents.read()
    except Exception:
        return None
