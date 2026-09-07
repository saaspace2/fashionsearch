# Databricks notebook source
# MAGIC %md
# MAGIC # 11 — Metrics and drift dashboard
# MAGIC
# MAGIC One page answering three questions:
# MAGIC
# MAGIC 1. **How good is the model?** Recall@k, MRR, NDCG@k, per slice.
# MAGIC 2. **Is it getting better or worse?** The same numbers over time.
# MAGIC 3. **Has the data drifted?** Whether the inputs still look like they used to.
# MAGIC
# MAGIC Safe to run any time; it writes only a drift snapshot.

# COMMAND ----------
import sys, pathlib
sys.path.insert(0, str(pathlib.Path.cwd().parent / "src"))
from fashionsearch.config import load_config, table
from fashionsearch import drift

cfg = load_config()

import json
import numpy as np
import pandas as pd
from pyspark.sql import functions as F

def card(title, value, sub="", colour="#7F77DD"):
    return (f"<div style='flex:1;min-width:150px;border:1px solid #E3E1DA;"
            f"border-left:4px solid {colour};border-radius:8px;padding:12px 14px;"
            f"font-family:-apple-system,Helvetica,Arial'>"
            f"<div style='font-size:11px;color:#5F5E5A;text-transform:uppercase;"
            f"letter-spacing:.04em'>{title}</div>"
            f"<div style='font-size:26px;font-weight:600;margin:4px 0'>{value}</div>"
            f"<div style='font-size:11px;color:#5F5E5A'>{sub}</div></div>")

# COMMAND ----------
# MAGIC %md ## 1 · Headline quality

# COMMAND ----------
metrics = spark.table(table(cfg, "gold", "retrieval_metrics"))
latest_ts = metrics.agg(F.max("evaluated_at")).first()[0]

if latest_ts is None:
    dbutils.notebook.exit("No evaluations yet. Run notebook 06 first.")

latest = metrics.filter(F.col("evaluated_at") == latest_ts)
overall = latest.filter(F.col("slice_dim") == "overall").first()
version = overall["model_version"]

cards = [
    card("Recall@20", f"{overall['recall_at_20']:.3f}",
         "correct answer on the first screen", "#639922"),
    card("NDCG@20", f"{overall['ndcg_at_20']:.3f}", "ranking quality", "#7F77DD"),
    card("MRR", f"{overall['mrr']:.3f}", "is the top result right", "#7F77DD"),
    card("Recall@1", f"{overall['recall_at_1']:.3f}", "strictest measure", "#888780"),
    card("Eval queries", f"{overall['n_queries']}", "sample size", "#888780"),
    card("Gate", "PASSED" if overall["gate_passed"] else "BLOCKED",
         f"model v{version}", "#639922" if overall["gate_passed"] else "#E24B4A"),
]
displayHTML("<div style='display:flex;gap:10px;flex-wrap:wrap'>" + "".join(cards) + "</div>")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 2 · Per slice
# MAGIC
# MAGIC The headline number above is an average, and an average hides its worst
# MAGIC behaviour in whichever slice is smallest. This table is where problems
# MAGIC actually show up.

# COMMAND ----------
slices = (latest.filter(F.col("slice_dim") != "overall")
          .select("slice_dim", "slice_value", "n_queries",
                  F.round("recall_at_20", 3).alias("recall@20"),
                  F.round("ndcg_at_20", 3).alias("ndcg@20"),
                  F.round("mrr", 3).alias("mrr"))
          .orderBy("slice_dim", F.desc("n_queries")))
display(slices)

floor = float(overall["recall_at_20"]) * 0.85
weak = slices.filter((F.col("recall@20") < floor) &
                     (F.col("n_queries") >= int(cfg.gate.min_queries_per_slice)))
if weak.count():
    print("Slices well below the overall average — look here first:")
    display(weak)
else:
    print("No slice is dramatically below the average.")

# COMMAND ----------
# MAGIC %md ## 3 · Trend over time

# COMMAND ----------
trend = (metrics.filter(F.col("slice_dim") == "overall")
         .select("evaluated_at", "model_version", "n_queries",
                 F.round("recall_at_20", 4).alias("recall_at_20"),
                 F.round("ndcg_at_20", 4).alias("ndcg_at_20"),
                 F.round("mrr", 4).alias("mrr"), "gate_passed")
         .orderBy(F.desc("evaluated_at")))
display(trend.limit(20))

hist = trend.orderBy("evaluated_at").toPandas()
if len(hist) >= 2:
    d = float(hist.iloc[-1].ndcg_at_20) - float(hist.iloc[-2].ndcg_at_20)
    verdict = ("improved" if d > 0.005 else "declined" if d < -0.005 else "held steady")
    print(f"NDCG@20 {verdict} since the previous evaluation: {d:+.4f}")
else:
    print("Only one evaluation so far — a trend needs at least two.")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 4 · Data drift
# MAGIC
# MAGIC Quality metrics tell you how the model performed on a **fixed** eval set.
# MAGIC Drift tells you whether the incoming data still resembles that set. The
# MAGIC second can move while the first looks perfect, and it moves first.
# MAGIC
# MAGIC Each run records a snapshot. Drift is the comparison between the newest
# MAGIC snapshot and the one before it.

# COMMAND ----------
spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {table(cfg, "monitoring", "drift_snapshots")} (
        captured_at TIMESTAMP, model_version STRING, payload STRING)
""")

# --- build the current snapshot -------------------------------------------
cat_counts = {r["category"]: r["n"] for r in spark.sql(f"""
    SELECT category, count(*) AS n FROM {table(cfg, "bronze", "products")}
    GROUP BY category""").collect()}

q = spark.table(table(cfg, "gold", "eval_queries"))
size_counts = {r["size_band"]: r["n"] for r in
               q.groupBy("size_band").agg(F.count("*").alias("n")).collect()}

# The fallback rate is the clearest single signal here: it is the share of
# queries where the detector found nothing and we embedded the whole image.
fallback_rate = float(size_counts.get("large", 0)) / max(q.count(), 1)

emb = (spark.table(table(cfg, "silver", "query_embeddings"))
       .select("embedding").limit(2000).toPandas())
centroid = (np.stack(emb.embedding.values).astype(np.float32).mean(axis=0).tolist()
            if len(emb) else [])

snapshot = {
    "category_counts": cat_counts,
    "size_band_counts": size_counts,
    "fallback_rate": round(fallback_rate, 4),
    "centroid": centroid,
    "n_products": int(spark.table(table(cfg, "bronze", "products")).count()),
    "n_eval_queries": int(q.count()),
}

# COMMAND ----------
prev_row = spark.sql(f"""
    SELECT payload FROM {table(cfg, "monitoring", "drift_snapshots")}
    ORDER BY captured_at DESC LIMIT 1""").first()

if prev_row is None:
    print("First snapshot recorded. Drift needs two, so run this again after the")
    print("next pipeline run to see comparisons.")
    rows = []
else:
    previous = json.loads(prev_row["payload"])
    rows = drift.summarise(previous, snapshot)

    if previous.get("centroid") and snapshot["centroid"]:
        shift = drift.centroid_shift([previous["centroid"]], [snapshot["centroid"]])
        rows.append({
            "signal": "query_embedding_centroid", "metric": "cosine distance",
            "value": round(shift, 5),
            "verdict": "stable" if shift < 0.05 else "significant shift",
            "hint": "the average visual character of incoming queries has moved",
        })

    psi_size = drift.population_stability_index(
        previous.get("size_band_counts", {}), snapshot["size_band_counts"])
    rows.append({
        "signal": "query_size_bands", "metric": "PSI", "value": round(psi_size, 4),
        "verdict": drift.psi_verdict(psi_size),
        "hint": "how large the detected garment is within the photo",
    })

# COMMAND ----------
if rows:
    df = pd.DataFrame(rows)
    display(spark.createDataFrame(df))

    colours = {"stable": "#639922", "moderate shift": "#BA7517",
               "significant shift": "#E24B4A"}
    displayHTML("<div style='display:flex;gap:10px;flex-wrap:wrap'>" + "".join(
        card(r["signal"].replace("_", " "), f"{r['value']}",
             f"{r['metric']} · {r['verdict']}", colours.get(r["verdict"], "#888780"))
        for _, r in df.iterrows()) + "</div>")

    moved = df[df.verdict != "stable"]
    if len(moved):
        print("\nSignals that moved:")
        for _, r in moved.iterrows():
            print(f"  {r['signal']}: {r['verdict']} — {r['hint']}")
        print("\nDrift is not proof that quality dropped. It is a reason to re-run")
        print("the evaluation and compare, which section 3 above will then show.")
    else:
        print("\nNo signal moved beyond its noise band.")

# COMMAND ----------
(spark.createDataFrame([(str(version), json.dumps(snapshot))],
                       "model_version STRING, payload STRING")
 .withColumn("captured_at", F.current_timestamp())
 .write.mode("append").saveAsTable(table(cfg, "monitoring", "drift_snapshots")))
print("snapshot recorded")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 5 · How to read this page
# MAGIC
# MAGIC | If this moves | It probably means |
# MAGIC |---|---|
# MAGIC | Recall@20 falls, drift stable | The model got worse. Compare versions in section 3. |
# MAGIC | Drift moves, Recall@20 stable | The inputs changed but the eval set did not. Your eval set is going stale. |
# MAGIC | Both move | Genuine distribution shift. Retrain on newer data. |
# MAGIC | Category mix PSI high | New product types, or a supplier added or dropped. |
# MAGIC | Detector fallback rate up | Photos got harder, or the detector is degrading. |
# MAGIC | Centroid shift high | Incoming photos look visually different — new crop format or photography style. |
# MAGIC
# MAGIC The second row is the one people miss. A frozen eval set cannot tell you it
# MAGIC has become unrepresentative — it will keep reporting the same comfortable
# MAGIC number while production diverges from it. That is what drift monitoring is
# MAGIC for, and it is why both halves of this page matter.
