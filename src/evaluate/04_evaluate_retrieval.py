"""
Retrieval evaluation and promotion gate.

THIS IS THE FILE THE ORIGINAL PROJECT HAS NO EQUIVALENT OF, and it is the
difference between a demo and a product.

The original's "Experiments" section shows nine query images and their results.
They look good. But nine examples cannot answer any of the questions that decide
whether a change is worth shipping:

    Is the new encoder better, or does it just fail differently?
    Does it work as well on bags as on shoes?  (It does not.)
    If we truncate 512 dims to 64 and halve serving cost, what does it cost us?
    Did last week's retrain make small, occluded items worse?

Every one of those is answerable with a labelled eval set and three metrics.

The metrics
-----------
Recall@k  Of all the correct products for this query, what fraction appeared in
          the top k? Recall@20 means "if the user scrolls one screen, is it
          there?" This is the primary metric for a search product.

MRR       1 / (rank of the first correct result), averaged. First place scores
          1.0, fifth scores 0.2. Rewards getting one thing exactly right.

NDCG@k    Position-weighted, and handles graded relevance (an exact match should
          outrank a merely-similar item). The most complete of the three.

All three are reported PER SLICE, never only in aggregate — see the note in
apply_gates() for why that matters more than it sounds.
"""

import argparse
import math
from dataclasses import dataclass, asdict

import mlflow
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from pyspark.sql import SparkSession, functions as F


SLICE_DIMENSIONS = ["category", "query_condition", "size_band", "occlusion"]

# Rare and hard. An aggregate metric will always quietly trade these away,
# because there is not enough of them to move an average. Protected explicitly.
PROTECTED_SLICES = [
    ("size_band", "small"),
    ("occlusion", "heavy"),
    ("category", "bag"),
    ("category", "hat"),
]


@dataclass
class GateResult:
    name: str
    passed: bool
    observed: float
    threshold: float
    detail: str = ""


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--catalog", required=True)
    p.add_argument("--encoder-model", required=True)
    p.add_argument("--champion-alias", default="production")
    p.add_argument("--candidate-alias", default="candidate")
    p.add_argument("--k-values", default="1,5,10,20,50")
    p.add_argument("--max-recall-drop", type=float, default=0.01)
    p.add_argument("--max-recall-drop-protected", type=float, default=0.005)
    p.add_argument("--min-slice-queries", type=int, default=50)
    p.add_argument("--eval-dims", default="64,512",
                   help="Also report metrics at truncated embedding sizes")
    return p.parse_args()


# --- metrics ---------------------------------------------------------------

def recall_at_k(ranked_ids, relevant_ids, k):
    """Fraction of the relevant items that appear in the top k."""
    if not relevant_ids:
        return float("nan")
    hits = len(set(ranked_ids[:k]) & set(relevant_ids))
    return hits / len(relevant_ids)


def reciprocal_rank(ranked_ids, relevant_ids):
    """1 / rank of the first correct result. 0 if none appear."""
    for i, pid in enumerate(ranked_ids, start=1):
        if pid in relevant_ids:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked_ids, relevance_map, k):
    """
    Normalised Discounted Cumulative Gain.

    Each result contributes its relevance grade, discounted by log2 of its
    position, so a correct answer at rank 1 is worth much more than at rank 10.
    Divided by the best achievable score so the result is always 0 to 1 and
    comparable across queries with different numbers of correct answers.
    """
    dcg = sum(relevance_map.get(pid, 0.0) / math.log2(i + 1)
              for i, pid in enumerate(ranked_ids[:k], start=1))
    ideal = sorted(relevance_map.values(), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 1) for i, g in enumerate(ideal, start=1))
    return dcg / idcg if idcg > 0 else float("nan")


# --- running the search ----------------------------------------------------

def embed_images(model_uri, paths, dim=None, batch=64):
    import torch
    from PIL import Image
    import torchvision.transforms as T

    model = mlflow.pytorch.load_model(model_uri).cuda().eval()
    tf = T.Compose([T.Resize((224, 224)), T.ToTensor(),
                    T.Normalize([0.5] * 3, [0.5] * 3)])

    out = []
    for start in range(0, len(paths), batch):
        imgs = [tf(Image.open(p).convert("RGB")) for p in paths[start:start + batch]]
        x = torch.stack(imgs).cuda()
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            emb = model(x).float().cpu().numpy()
        out.append(emb)

    embs = np.concatenate(out)
    if dim:
        # Matryoshka truncation, then renormalise so cosine still behaves.
        embs = embs[:, :dim]
        embs /= np.linalg.norm(embs, axis=1, keepdims=True)
    return embs


def run_retrieval(query_emb, catalog_emb, catalog_ids, catalog_cats,
                  query_cats, top_k=50):
    """
    Exact search over the eval catalogue. Deliberately brute-force: the eval set
    is small, and using the approximate index here would mix up two different
    questions — "is the model better?" and "is the index tuned well?". Measure
    those separately.

    The category pre-filter mirrors production, where the user has told us which
    garment they tapped.
    """
    results = []
    for i in range(len(query_emb)):
        mask = catalog_cats == query_cats[i]
        sims = catalog_emb[mask] @ query_emb[i]
        ids = catalog_ids[mask]
        order = np.argsort(-sims)[:top_k]
        results.append(list(ids[order]))
    return results


def compute_slice_metrics(eval_df, ranked, k_values, tag):
    rows = []
    eval_df = eval_df.copy()
    eval_df["_ranked"] = ranked

    for dim in SLICE_DIMENSIONS:
        if dim not in eval_df.columns:
            continue
        for value, group in eval_df.groupby(dim):
            rec = {"slice_dim": dim, "slice_value": str(value),
                   "n_queries": len(group), "embedding_dim": tag}
            for k in k_values:
                rec[f"recall_at_{k}"] = float(np.nanmean([
                    recall_at_k(r["_ranked"], r["relevant_ids"], k)
                    for _, r in group.iterrows()]))
                rec[f"ndcg_at_{k}"] = float(np.nanmean([
                    ndcg_at_k(r["_ranked"], r["relevance_map"], k)
                    for _, r in group.iterrows()]))
            rec["mrr"] = float(np.mean([
                reciprocal_rank(r["_ranked"], r["relevant_ids"])
                for _, r in group.iterrows()]))
            rows.append(rec)

    # Overall row, reported alongside the slices but never used alone as a gate.
    overall = {"slice_dim": "overall", "slice_value": "all",
               "n_queries": len(eval_df), "embedding_dim": tag}
    for k in k_values:
        overall[f"recall_at_{k}"] = float(np.nanmean([
            recall_at_k(r["_ranked"], r["relevant_ids"], k)
            for _, r in eval_df.iterrows()]))
        overall[f"ndcg_at_{k}"] = float(np.nanmean([
            ndcg_at_k(r["_ranked"], r["relevance_map"], k)
            for _, r in eval_df.iterrows()]))
    overall["mrr"] = float(np.mean([
        reciprocal_rank(r["_ranked"], r["relevant_ids"])
        for _, r in eval_df.iterrows()]))
    rows.append(overall)

    return pd.DataFrame(rows)


# --- the gate --------------------------------------------------------------

def apply_gates(cand, champ, args) -> list:
    """
    Why per-slice and not aggregate:

    A model can gain 1.5 points of overall NDCG while losing 8 points of Recall@20
    on bags. Bags are maybe 6% of eval queries, so the loss barely moves the
    average, and an aggregate gate approves it. Then bag search is broken in
    production and nobody knows why the numbers looked fine.

    Rarity is exactly why rare slices need protecting, not a reason to ignore them.
    """
    results = []
    full = cand[cand.embedding_dim == "512"]
    champ_full = champ[champ.embedding_dim == "512"]

    # Gate 1 — overall NDCG@20 at least matches the champion.
    c = float(full[full.slice_dim == "overall"].ndcg_at_20.iloc[0])
    ch = float(champ_full[champ_full.slice_dim == "overall"].ndcg_at_20.iloc[0]) \
        if not champ_full.empty else 0.0
    results.append(GateResult("overall_ndcg20_not_worse", c >= ch, c, ch,
                              "candidate vs champion NDCG@20"))

    # Gate 2 — no Recall@20 regression on ANY slice.
    worst, worst_name = 0.0, ""
    for row in full.itertuples():
        if row.slice_dim == "overall" or row.n_queries < args.min_slice_queries:
            continue
        m = champ_full[(champ_full.slice_dim == row.slice_dim)
                       & (champ_full.slice_value == row.slice_value)]
        if m.empty:
            continue
        drop = float(m.iloc[0].recall_at_20) - float(row.recall_at_20)
        if drop > worst:
            worst, worst_name = drop, f"{row.slice_dim}={row.slice_value}"
    results.append(GateResult("recall20_no_slice_regression",
                              worst <= args.max_recall_drop, worst,
                              args.max_recall_drop,
                              f"worst slice: {worst_name}" if worst_name else "none"))

    # Gate 3 — protected slices, tighter tolerance.
    pworst, pname = 0.0, ""
    for dim, val in PROTECTED_SLICES:
        cr = full[(full.slice_dim == dim) & (full.slice_value == val)]
        mr = champ_full[(champ_full.slice_dim == dim) & (champ_full.slice_value == val)]
        if cr.empty or mr.empty:
            continue
        drop = float(mr.iloc[0].recall_at_20) - float(cr.iloc[0].recall_at_20)
        if drop > pworst:
            pworst, pname = drop, f"{dim}={val}"
    results.append(GateResult("protected_slices_no_regression",
                              pworst <= args.max_recall_drop_protected, pworst,
                              args.max_recall_drop_protected,
                              f"worst protected: {pname}" if pname else "none"))

    # Gate 4 — the cheap 64-dim recall stage must still work. If truncation
    # collapses, the whole two-stage retrieval design falls apart in production
    # even though the 512-dim numbers look fine.
    short = cand[cand.embedding_dim == "64"]
    if not short.empty:
        r = float(short[short.slice_dim == "overall"].recall_at_50.iloc[0])
        full_r = float(full[full.slice_dim == "overall"].recall_at_50.iloc[0])
        ratio = r / full_r if full_r > 0 else 0.0
        results.append(GateResult("matryoshka_64d_recall_retained",
                                  ratio >= 0.95, ratio, 0.95,
                                  "64-dim Recall@50 as a fraction of 512-dim"))

    # Gate 5 — did every protected slice actually have enough queries to measure?
    # Silence is not a pass.
    missing = [f"{d}={v}" for d, v in PROTECTED_SLICES
               if full[(full.slice_dim == d) & (full.slice_value == v)
                       & (full.n_queries >= args.min_slice_queries)].empty]
    results.append(GateResult("protected_slice_coverage", len(missing) == 0,
                              len(PROTECTED_SLICES) - len(missing),
                              len(PROTECTED_SLICES),
                              f"too few queries for: {', '.join(missing)}"
                              if missing else "all covered"))
    return results


def main():
    args = parse_args()
    spark = SparkSession.builder.getOrCreate()
    mlflow.set_registry_uri("databricks-uc")
    client = MlflowClient()
    cat = args.catalog
    k_values = [int(k) for k in args.k_values.split(",")]

    latest = client.get_registered_model(args.encoder_model).latest_versions[0]
    cand_uri = f"models:/{args.encoder_model}/{latest.version}"
    try:
        client.get_model_version_by_alias(args.encoder_model, args.champion_alias)
        champ_uri = f"models:/{args.encoder_model}@{args.champion_alias}"
    except Exception:
        champ_uri = None
        print("no champion yet — bootstrap mode, gates compare against zero")

    # The frozen eval set: real queries with human relevance judgments.
    eval_df = spark.table(f"{cat}.gold.eval_queries").toPandas()
    catalogue = (spark.table(f"{cat}.bronze.products")
                 .select("product_id", "image_path", "category").toPandas())

    with mlflow.start_run(run_name=f"retrieval-gate-v{latest.version}") as run:
        cand_frames, champ_frames = [], []

        for dim in [int(d) for d in args.eval_dims.split(",")]:
            cat_emb = embed_images(cand_uri, catalogue.image_path.tolist(), dim)
            q_emb = embed_images(cand_uri, eval_df.query_image.tolist(), dim)
            ranked = run_retrieval(q_emb, cat_emb,
                                   catalogue.product_id.values,
                                   catalogue.category.values,
                                   eval_df.category.values,
                                   top_k=max(k_values))
            cand_frames.append(compute_slice_metrics(eval_df, ranked, k_values, str(dim)))

            if champ_uri:
                cat_emb_c = embed_images(champ_uri, catalogue.image_path.tolist(), dim)
                q_emb_c = embed_images(champ_uri, eval_df.query_image.tolist(), dim)
                ranked_c = run_retrieval(q_emb_c, cat_emb_c,
                                         catalogue.product_id.values,
                                         catalogue.category.values,
                                         eval_df.category.values,
                                         top_k=max(k_values))
                champ_frames.append(
                    compute_slice_metrics(eval_df, ranked_c, k_values, str(dim)))

        cand = pd.concat(cand_frames, ignore_index=True)
        champ = pd.concat(champ_frames, ignore_index=True) if champ_frames \
            else cand.assign(**{f"recall_at_{k}": 0.0 for k in k_values},
                             ndcg_at_20=0.0)

        gates = apply_gates(cand, champ, args)
        passed = all(g.passed for g in gates)

        # Persist per-slice metrics. This table is the record you compare against
        # next month, and the thing you show when someone asks "is it better?".
        (spark.createDataFrame(cand)
             .withColumn("model_name", F.lit(args.encoder_model))
             .withColumn("model_version", F.lit(str(latest.version)))
             .withColumn("evaluated_at", F.current_timestamp())
             .withColumn("gate_passed", F.lit(passed))
             .write.mode("append").saveAsTable(f"{cat}.gold.retrieval_metrics"))

        for g in gates:
            mlflow.log_metric(f"gate.{g.name}", g.observed)
            mlflow.set_tag(f"gate.{g.name}", "PASS" if g.passed else "FAIL")
        mlflow.log_dict({"gates": [asdict(g) for g in gates]}, "gate_report.json")
        mlflow.log_table(cand, "slice_metrics.json")

        print("\n" + "=" * 76)
        for g in gates:
            print(f"  [{'PASS' if g.passed else 'FAIL'}] {g.name:<34} "
                  f"{g.observed:>9.4f} vs {g.threshold:<9.4f} {g.detail}")
        print("=" * 76 + "\n")

        if not passed:
            failed = [g.name for g in gates if not g.passed]
            client.set_model_version_tag(args.encoder_model, latest.version,
                                         "gate_status", "FAILED")
            raise SystemExit(
                f"Promotion blocked. Failed: {', '.join(failed)}. Per-slice detail "
                f"in {cat}.gold.retrieval_metrics (model_version={latest.version})."
            )

        client.set_registered_model_alias(args.encoder_model,
                                          args.candidate_alias, latest.version)
        client.set_model_version_tag(args.encoder_model, latest.version,
                                     "gate_status", "PASSED")
        print(f"{args.encoder_model} v{latest.version} tagged @{args.candidate_alias}.")
        print("Promotion to @shadow and @production stays a human decision.")


if __name__ == "__main__":
    main()
