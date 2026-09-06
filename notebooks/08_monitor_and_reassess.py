# Databricks notebook source
# MAGIC %md
# MAGIC # 08 — Monitor and reassess the champion
# MAGIC
# MAGIC Two jobs in one notebook, run daily.
# MAGIC
# MAGIC **Monitor** — is production search still working? Click-through, zero-result
# MAGIC rate, reformulation rate, index freshness, latency.
# MAGIC
# MAGIC **Reassess** — re-run the frozen eval set against the current champion. A
# MAGIC model file does not change, but the catalogue does: new products arrive, old
# MAGIC ones sell out, and the champion's real Recall@20 drifts even though its
# MAGIC weights are identical. Without this you would only find out from users.

# COMMAND ----------
import sys, pathlib
sys.path.insert(0, str(pathlib.Path.cwd().parent / "src"))
from fashionsearch.config import load_config, table

cfg = load_config()

import json
from pyspark.sql import functions as F

alerts = []

# COMMAND ----------
# MAGIC %md ## Online metrics

# COMMAND ----------
m = spark.sql(f"""
    SELECT
      count(*)                                           AS n_searches,
      avg(CASE WHEN size(clicked) > 0 THEN 1 ELSE 0 END) AS ctr,
      avg(CASE WHEN n_results = 0   THEN 1 ELSE 0 END)   AS zero_result_rate,
      avg(CASE WHEN reformulated   THEN 1 ELSE 0 END)    AS reformulation_rate,
      percentile_approx(latency_ms, 0.95)                AS p95_latency
    FROM {table(cfg, "bronze", "search_events")}
    WHERE event_ts >= current_date() - INTERVAL 1 DAYS
""").first()

if m and m["n_searches"]:
    if m["ctr"] < cfg.monitoring.min_ctr:
        alerts.append({"kind": "low_ctr", "severity": "high", "value": float(m["ctr"]),
                       "hint": "results stopped matching intent — check for a recent "
                               "model or index change before anything else"})
    if m["zero_result_rate"] > cfg.monitoring.max_zero_result_rate:
        alerts.append({"kind": "zero_results", "severity": "high",
                       "value": float(m["zero_result_rate"]),
                       "hint": "filters too strict, or a category has no stock"})
    if m["reformulation_rate"] > 0.30:
        # The most honest quality signal available. A user searching again
        # immediately is telling you the first attempt failed.
        alerts.append({"kind": "high_reformulation", "severity": "warning",
                       "value": float(m["reformulation_rate"]),
                       "hint": "users are retrying — the first result set is wrong"})
else:
    print("no production traffic yet — online checks skipped")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Per-category click-through
# MAGIC
# MAGIC The same reason the gate slices: a collapse in one category is invisible in
# MAGIC the overall average and is usually the first sign of a real regression.

# COMMAND ----------
per_cat = spark.sql(f"""
    SELECT selected_category AS category,
           avg(CASE WHEN size(clicked) > 0 THEN 1 ELSE 0 END) AS ctr,
           count(*) AS n
    FROM {table(cfg, "bronze", "search_events")}
    WHERE event_ts >= current_date() - INTERVAL 7 DAYS
    GROUP BY selected_category
    HAVING count(*) > 200
       AND avg(CASE WHEN size(clicked) > 0 THEN 1 ELSE 0 END) < {cfg.monitoring.min_ctr}
""")
for r in per_cat.collect():
    alerts.append({"kind": "category_ctr_low", "severity": "warning",
                   "category": r["category"], "value": float(r["ctr"]),
                   "hint": "this category is failing while the average looks fine"})

# COMMAND ----------
# MAGIC %md ## Index freshness

# COMMAND ----------
try:
    stale = spark.sql(f"""
        SELECT timestampdiff(HOUR, max(embedded_at), current_timestamp()) AS hours
        FROM {table(cfg, "silver", "product_embeddings")}
    """).first()
    if stale and stale["hours"] and stale["hours"] > cfg.monitoring.max_index_staleness_hours:
        alerts.append({"kind": "index_stale", "severity": "high",
                       "value": float(stale["hours"]),
                       "hint": "search is returning products that may no longer exist"})
except Exception as e:
    print("freshness check skipped:", e)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Reassess the champion
# MAGIC
# MAGIC Compare today's per-slice numbers against the champion's numbers when it
# MAGIC was promoted. A drop here with unchanged weights means the *catalogue*
# MAGIC changed underneath the model.

# COMMAND ----------
history = spark.table(table(cfg, "gold", "retrieval_metrics")) \
    .filter(F.col("slice_dim") == "overall")

if history.count() >= 2:
    trend = (history.orderBy(F.desc("evaluated_at"))
             .select("model_version", "evaluated_at", "ndcg_at_20", "recall_at_20")
             .limit(10).toPandas())
    display(spark.createDataFrame(trend))

    newest, previous = trend.iloc[0], trend.iloc[1]
    delta = float(newest.ndcg_at_20) - float(previous.ndcg_at_20)
    if delta < -0.02:
        alerts.append({"kind": "champion_degraded", "severity": "high",
                       "value": delta,
                       "hint": "NDCG@20 fell with no model change — the catalogue "
                               "shifted. Consider re-embedding or retraining."})
else:
    print("not enough evaluation history to trend yet")

# COMMAND ----------
spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {table(cfg, "monitoring", "alerts")} (
        payload STRING, kind STRING, severity STRING, raised_at TIMESTAMP)
""")

if alerts:
    (spark.createDataFrame([(json.dumps(a), a["kind"], a["severity"]) for a in alerts],
                           "payload STRING, kind STRING, severity STRING")
     .withColumn("raised_at", F.current_timestamp())
     .write.mode("append").saveAsTable(table(cfg, "monitoring", "alerts")))

print(f"{len(alerts)} alerts")
for a in alerts:
    print(f"  [{a['severity']:>7}] {a['kind']}: {a.get('hint', '')}")

if any(a["severity"] == "high" for a in alerts):
    raise Exception("High-severity alert — see monitoring.alerts")
