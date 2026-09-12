# Databricks notebook source
# MAGIC %md
# MAGIC # 08 — Monitor and reassess
# MAGIC
# MAGIC Four checks, run daily. Three of them work from day one; only the fourth
# MAGIC needs production traffic.
# MAGIC
# MAGIC | Check | Needs |
# MAGIC |---|---|
# MAGIC | Catalogue coverage | nothing — runs immediately |
# MAGIC | Index freshness | nothing |
# MAGIC | Champion trend | two or more evaluations |
# MAGIC | Online search quality | real users |
# MAGIC
# MAGIC The last one is skipped with a note rather than treated as a failure. An
# MAGIC empty `search_events` table on a system nobody uses yet is not an alert;
# MAGIC reporting it as one would teach you to ignore this notebook.

# COMMAND ----------
# MAGIC %md
# MAGIC Dependencies come from the job's `environments:` block.

# COMMAND ----------
import sys, pathlib
sys.path.insert(0, str(pathlib.Path.cwd().parent / "src"))
from fashionsearch.config import load_config, table

cfg = load_config()

import json
from pyspark.sql import functions as F

alerts = []
skipped = []


def alert(kind, severity, value, hint, **extra):
    alerts.append({"kind": kind, "severity": severity, "value": value,
                   "hint": hint, **extra})

# COMMAND ----------
# MAGIC %md
# MAGIC ## 1 · Catalogue coverage
# MAGIC
# MAGIC Every product must have an embedding. A product in the catalogue but not
# MAGIC in the index is invisible to search, and nothing else in the system will
# MAGIC tell you — the searches that should have returned it simply do not.

# COMMAND ----------
n_products = spark.table(table(cfg, "bronze", "products")).count()
n_embedded = spark.table(table(cfg, "silver", "product_embeddings")).count()
coverage = n_embedded / n_products if n_products else 0.0

print(f"  products in catalogue : {n_products}")
print(f"  products embedded     : {n_embedded}")
print(f"  coverage              : {coverage:.1%}")

if coverage < 0.99:
    alert("incomplete_index", "high", round(coverage, 4),
          f"{n_products - n_embedded} products are unsearchable. Re-run notebook 05.")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 2 · Index freshness
# MAGIC
# MAGIC A fashion catalogue turns over constantly. A stale index returns products
# MAGIC that are sold out or delisted, which users experience simply as the search
# MAGIC being broken.

# COMMAND ----------
fresh = spark.sql(f"""
    SELECT max(embedded_at) AS last_embedded,
           timestampdiff(HOUR, max(embedded_at), current_timestamp()) AS hours
    FROM {table(cfg, "silver", "product_embeddings")}
""").first()

max_hours = float(cfg.monitoring.max_index_staleness_hours)
if fresh and fresh["hours"] is not None:
    print(f"  last embedded : {fresh['last_embedded']}")
    print(f"  age           : {fresh['hours']} h  (limit {max_hours:.0f} h)")
    if fresh["hours"] > max_hours:
        # Graded rather than binary. Slightly over the limit is worth noting;
        # four times over means the pipeline has genuinely stopped running, and
        # only that deserves to fail the job and wake somebody.
        severity = "high" if fresh["hours"] > max_hours * 4 else "warning"
        alert("index_stale", severity, float(fresh["hours"]),
              f"embeddings are {fresh['hours']:.0f}h old (limit {max_hours:.0f}h). "
              f"Search may be returning products that no longer exist. "
              f"Re-run the pipeline to refresh them.")
else:
    alert("no_index", "high", 0.0, "no embeddings at all — run notebook 05")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 3 · Champion trend
# MAGIC
# MAGIC The model file does not change, but the catalogue does. Re-scoring the
# MAGIC frozen eval set catches drift that the weights alone would never reveal:
# MAGIC new products arrive, old ones sell out, and real Recall@20 moves even
# MAGIC though nothing was retrained.

# COMMAND ----------
history = (spark.table(table(cfg, "gold", "retrieval_metrics"))
           .filter(F.col("slice_dim") == "overall")
           .select("model_version", "evaluated_at", "n_queries",
                   "recall_at_20", "ndcg_at_20", "mrr", "gate_passed")
           .orderBy(F.desc("evaluated_at")))

n_evals = history.count()
if n_evals:
    display(history.limit(10))

if n_evals >= 2:
    rows = history.limit(2).collect()
    newest, previous = rows[0], rows[1]
    delta = float(newest["ndcg_at_20"]) - float(previous["ndcg_at_20"])
    print(f"  NDCG@20 change since last evaluation: {delta:+.4f}")
    if delta < -0.02:
        alert("champion_degraded", "high", round(delta, 4),
              "quality fell with no model change — the catalogue shifted under it")
elif n_evals == 1:
    only = history.first()
    print(f"  one evaluation so far: NDCG@20 {only['ndcg_at_20']:.4f} "
          f"over {only['n_queries']} queries")
    skipped.append("champion trend — needs a second evaluation to compare against")
else:
    skipped.append("champion trend — no evaluations yet, run notebook 06")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 4 · Online search quality
# MAGIC
# MAGIC Offline metrics predict; online metrics decide. A model that gains two
# MAGIC points of NDCG offline and loses click-through in an A/B test has not
# MAGIC improved, whatever the eval set says.

# COMMAND ----------
n_events = spark.table(table(cfg, "bronze", "search_events")).count()

if n_events == 0:
    skipped.append("online metrics — no production traffic yet")
    print("  no search events recorded. Nothing is wrong; nobody has searched.")
    print("  Once an application writes to bronze.search_events, this section")
    print("  starts reporting click-through, zero-result and reformulation rates.")
else:
    m = spark.sql(f"""
        SELECT count(*)                                           AS n,
               avg(CASE WHEN size(clicked) > 0 THEN 1 ELSE 0 END) AS ctr,
               avg(CASE WHEN n_results = 0    THEN 1 ELSE 0 END)  AS zero_rate,
               avg(CASE WHEN reformulated    THEN 1 ELSE 0 END)   AS reform_rate,
               percentile_approx(latency_ms, 0.95)                AS p95
        FROM {table(cfg, "bronze", "search_events")}
        WHERE event_ts >= current_date() - INTERVAL 1 DAYS
    """).first()

    if m and m["n"]:
        print(f"  searches (24h)      : {m['n']}")
        print(f"  click-through       : {m['ctr']:.1%}")
        print(f"  zero-result rate    : {m['zero_rate']:.1%}")
        print(f"  reformulation rate  : {m['reform_rate']:.1%}")
        if m["ctr"] < float(cfg.monitoring.min_ctr):
            alert("low_ctr", "high", float(m["ctr"]),
                  "results stopped matching intent — check for a recent model change")
        if m["zero_rate"] > float(cfg.monitoring.max_zero_result_rate):
            alert("zero_results", "high", float(m["zero_rate"]),
                  "filters too strict, or a category has no stock")
        if m["reform_rate"] > 0.30:
            # The most honest quality signal there is: a user searching again
            # immediately is telling you the first attempt failed.
            alert("high_reformulation", "warning", float(m["reform_rate"]),
                  "users are retrying — the first result set is wrong")

# COMMAND ----------
# MAGIC %md ## Summary

# COMMAND ----------
spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {table(cfg, "monitoring", "alerts")} (
        payload STRING, kind STRING, severity STRING, raised_at TIMESTAMP)
""")

if alerts:
    (spark.createDataFrame(
        [(json.dumps(a), a["kind"], a["severity"]) for a in alerts],
        "payload STRING, kind STRING, severity STRING")
     .withColumn("raised_at", F.current_timestamp())
     .write.mode("append").saveAsTable(table(cfg, "monitoring", "alerts")))

print(f"\n{len(alerts)} alert(s), {len(skipped)} check(s) skipped\n")
for a in alerts:
    print(f"  [{a['severity']:>7}] {a['kind']}: {a['hint']}")
for s in skipped:
    print(f"  [ skipped] {s}")

if not alerts and not skipped:
    print("  everything healthy")

# COMMAND ----------
# A high-severity alert fails the job on purpose. A log line is something nobody
# reads; a red job is something somebody notices. Warnings and skips do not fail.
high = [a for a in alerts if a["severity"] == "high"]
if high:
    raise Exception(
        f"{len(high)} high-severity alert(s): {', '.join(a['kind'] for a in high)}. "
        f"See {table(cfg, 'monitoring', 'alerts')}.")
