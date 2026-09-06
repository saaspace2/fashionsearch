# Databricks notebook source
# MAGIC %md
# MAGIC # 06 — Evaluate and gate
# MAGIC
# MAGIC The centre of the whole pipeline. Runs real retrieval over the frozen eval
# MAGIC set, computes Recall@k / MRR / NDCG@k **per slice**, and decides whether
# MAGIC the candidate earns `@candidate`.
# MAGIC
# MAGIC The original project's equivalent is nine screenshots that look good. This
# MAGIC produces a number you can compare next month.

# COMMAND ----------
import sys, pathlib
sys.path.insert(0, str(pathlib.Path.cwd().parent / "src"))
from fashionsearch.config import load_config, table, ensure_experiment
from fashionsearch.metrics import evaluate_query
from fashionsearch.promotion import evaluate_gate

cfg = load_config()

import json
import numpy as np, pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
from pyspark.sql import functions as F

mlflow.set_registry_uri("databricks-uc")
ensure_experiment(f"/Shared/{cfg.project.name}/evaluation")
client = MlflowClient()
K_VALUES = list(cfg.gate.k_values)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Retrieval
# MAGIC
# MAGIC Brute-force on purpose. The eval catalogue is small, and using the
# MAGIC approximate index here would mix up two different questions: "is the model
# MAGIC better?" and "is the index tuned well?". Measure those separately or you
# MAGIC will not know which one changed.
# MAGIC
# MAGIC The category pre-filter mirrors production, where the user has told us
# MAGIC which garment they tapped.

# COMMAND ----------
cat_emb = spark.table(table(cfg, "silver", "product_embeddings")) \
    .select("product_id", "category", "embedding").toPandas()
q_emb = spark.table(table(cfg, "silver", "query_embeddings")) \
    .select("post_id", "embedding").toPandas()
eval_set = spark.table(table(cfg, "gold", "eval_queries")).toPandas()

E = np.stack(cat_emb.embedding.values).astype(np.float32)
E /= np.linalg.norm(E, axis=1, keepdims=True)
ids = cat_emb.product_id.values
cats = cat_emb.category.values

q_lookup = {r.post_id: np.asarray(r.embedding, dtype=np.float32)
            for r in q_emb.itertuples()}

results = []
for row in eval_set.itertuples():
    q = q_lookup.get(row.post_id)
    if q is None:
        continue
    q = q / np.linalg.norm(q)
    mask = cats == row.category
    if not mask.any():
        results.append((row.post_id, []))
        continue
    sims = E[mask] @ q
    order = np.argsort(-sims)[:max(K_VALUES)]
    results.append((row.post_id, list(ids[mask][order])))

ranked = dict(results)
print(f"ran retrieval for {len(ranked)} queries")

# COMMAND ----------
# MAGIC %md ## Per-slice metrics

# COMMAND ----------
SLICE_DIMS = ["category", "query_condition", "size_band", "occlusion"]

per_query = []
for row in eval_set.itertuples():
    r = ranked.get(row.post_id)
    if r is None:
        continue
    m = evaluate_query(r, row.relevant_ids, dict(row.relevance_map), K_VALUES)
    m["post_id"] = row.post_id
    for d in SLICE_DIMS:
        m[d] = getattr(row, d, "unknown")
    per_query.append(m)

pq = pd.DataFrame(per_query)
metric_cols = [c for c in pq.columns if c.startswith(("recall_at_", "ndcg_at_")) or c == "mrr"]

rows = []
for dim in SLICE_DIMS:
    for value, group in pq.groupby(dim):
        rec = {"slice_dim": dim, "slice_value": str(value), "n_queries": len(group)}
        rec.update({c: float(group[c].mean()) for c in metric_cols})
        rows.append(rec)

overall = {"slice_dim": "overall", "slice_value": "all", "n_queries": len(pq)}
overall.update({c: float(pq[c].mean()) for c in metric_cols})
rows.append(overall)

slice_metrics = pd.DataFrame(rows)
display(spark.createDataFrame(slice_metrics))

# COMMAND ----------
# MAGIC %md ## The gate

# COMMAND ----------
version = client.get_registered_model(cfg.registry.encoder_model).latest_versions[0]

# Champion baseline: the last recorded run for whichever version currently holds
# @production. On a first run there is none, so every slice compares to zero.
champ_rows = []
try:
    champ = client.get_model_version_by_alias(cfg.registry.encoder_model,
                                              cfg.registry.aliases.champion)
    hist = spark.table(table(cfg, "gold", "retrieval_metrics")) \
        .filter(F.col("model_version") == str(champ.version))
    if hist.count():
        latest_run = hist.agg(F.max("evaluated_at")).first()[0]
        champ_rows = hist.filter(F.col("evaluated_at") == latest_run).toPandas() \
            .to_dict("records")
except Exception as e:
    print("no champion baseline yet:", e)

report = evaluate_gate(slice_metrics.to_dict("records"), champ_rows, dict(cfg.gate))
print(report.render())

# COMMAND ----------
with mlflow.start_run(run_name=f"gate-encoder-v{version.version}") as run:
    mlflow.log_metrics({f"overall_{c}": overall[c] for c in metric_cols})
    for v in report.verdicts:
        mlflow.log_metric(f"gate.{v.name}", v.observed)
        mlflow.set_tag(f"gate.{v.name}", "PASS" if v.passed else "FAIL")
    mlflow.log_dict(report.as_dict(), "gate_report.json")
    mlflow.log_table(slice_metrics, "slice_metrics.json")

(spark.createDataFrame(slice_metrics)
 .withColumn("model_name", F.lit(cfg.registry.encoder_model))
 .withColumn("model_version", F.lit(str(version.version)))
 .withColumn("evaluated_at", F.current_timestamp())
 .withColumn("gate_passed", F.lit(report.passed))
 .write.mode("append").saveAsTable(table(cfg, "gold", "retrieval_metrics")))

# COMMAND ----------
if not report.passed:
    client.set_model_version_tag(cfg.registry.encoder_model, version.version,
                                 "gate_status", "FAILED")
    raise Exception(
        f"Promotion blocked: {', '.join(report.failures)}. Per-slice detail in "
        f"{table(cfg, 'gold', 'retrieval_metrics')} "
        f"(model_version={version.version}).")

client.set_registered_model_alias(cfg.registry.encoder_model,
                                  cfg.registry.aliases.candidate, version.version)
client.set_model_version_tag(cfg.registry.encoder_model, version.version,
                             "gate_status", "PASSED")
print(f"v{version.version} tagged @{cfg.registry.aliases.candidate}. "
      f"Promotion to @shadow and @production stays a human decision.")
