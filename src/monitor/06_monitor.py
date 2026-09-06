"""
Production monitoring and the feedback harvest.

Offline metrics predict; online metrics decide. A model that gains two points of
NDCG offline and loses click-through in an A/B test has not improved, whatever
the eval set says.

Three modes:

  online   Search quality as users experience it: click-through, zero-result
           rate, reformulation rate, index freshness, latency.

  drift    Are incoming query images distributed like the training data? Seasonal
           shifts, a new photo convention, a new market.

  harvest  Turn logs into training data. Clicked results become positives;
           shown-but-skipped results become hard negatives. This is the highest
           quality training data available and it costs nothing.
"""

import argparse
import json

from pyspark.sql import SparkSession, functions as F


def check_online(spark, args):
    cat = args.catalog
    alerts = []

    m = spark.sql(f"""
        SELECT
          count(*)                                              AS n_searches,
          avg(CASE WHEN size(clicked) > 0 THEN 1 ELSE 0 END)    AS ctr,
          avg(CASE WHEN n_results = 0    THEN 1 ELSE 0 END)     AS zero_result_rate,
          avg(CASE WHEN reformulated    THEN 1 ELSE 0 END)      AS reformulation_rate,
          percentile_approx(latency_ms, 0.95)                   AS p95_latency
        FROM {cat}.bronze.search_events
        WHERE event_ts >= current_date() - INTERVAL 1 DAYS
    """).first()

    if m and m["n_searches"]:
        if m["ctr"] < args.min_ctr:
            alerts.append({"kind": "low_ctr", "severity": "high",
                           "value": float(m["ctr"]),
                           "hint": "results are not matching intent — check a recent "
                                   "model or index change first"})
        if m["zero_result_rate"] > args.max_zero_result_rate:
            alerts.append({"kind": "zero_results", "severity": "high",
                           "value": float(m["zero_result_rate"]),
                           "hint": "filters may be too strict, or a category has no "
                                   "in-stock inventory"})
        if m["reformulation_rate"] > args.max_reformulation_rate:
            # The most honest quality signal there is. A user searching again
            # immediately is telling you the first search failed.
            alerts.append({"kind": "high_reformulation", "severity": "warning",
                           "value": float(m["reformulation_rate"]),
                           "hint": "users are retrying — the first result set is wrong"})
        if m["p95_latency"] > args.max_p95_latency_ms:
            alerts.append({"kind": "latency", "severity": "high",
                           "value": float(m["p95_latency"]),
                           "hint": "check recall_k and index replica count"})

    # Per-category click-through. A collapse in one category is invisible in the
    # overall number and is usually the first sign of a real regression.
    per_cat = spark.sql(f"""
        SELECT selected_category AS category,
               avg(CASE WHEN size(clicked) > 0 THEN 1 ELSE 0 END) AS ctr,
               count(*) AS n
        FROM {cat}.bronze.search_events
        WHERE event_ts >= current_date() - INTERVAL 7 DAYS
        GROUP BY selected_category
        HAVING count(*) > 200 AND
               avg(CASE WHEN size(clicked) > 0 THEN 1 ELSE 0 END) < {args.min_ctr}
    """)
    for r in per_cat.collect():
        alerts.append({"kind": "category_ctr_low", "severity": "warning",
                       "category": r["category"], "value": float(r["ctr"]),
                       "hint": "this category is failing while the average looks fine"})

    stale = spark.sql(f"""
        SELECT max(synced_at) AS last_sync,
               timestampdiff(HOUR, max(synced_at), current_timestamp()) AS hours
        FROM {cat}.monitoring.index_syncs
    """).first()
    if stale and stale["hours"] and stale["hours"] > args.max_index_staleness_hours:
        alerts.append({"kind": "index_stale", "severity": "high",
                       "value": float(stale["hours"]),
                       "hint": "search is returning products that may no longer exist"})

    write_alerts(spark, cat, alerts)


def check_drift(spark, args):
    cat = args.catalog
    alerts = []
    d = spark.sql(f"""
        WITH recent AS (
            SELECT query_condition, count(*) AS n
            FROM {cat}.bronze.search_events
            WHERE event_ts >= current_date() - INTERVAL 7 DAYS
            GROUP BY query_condition),
        base AS (
            SELECT query_condition, count(*) AS n
            FROM {cat}.bronze.search_events
            WHERE event_ts BETWEEN current_date() - INTERVAL 90 DAYS
                               AND current_date() - INTERVAL 7 DAYS
            GROUP BY query_condition)
        SELECT r.query_condition,
               r.n / (SELECT sum(n) FROM recent) AS p_now,
               b.n / (SELECT sum(n) FROM base)   AS p_base
        FROM recent r JOIN base b USING (query_condition)
    """)
    for r in d.collect():
        shift = abs(float(r["p_now"]) - float(r["p_base"]))
        if shift > args.js_divergence_threshold:
            alerts.append({"kind": "query_distribution_drift", "severity": "warning",
                           "condition": r["query_condition"], "value": shift,
                           "hint": "queries no longer look like the training data"})
    write_alerts(spark, cat, alerts)


def harvest(spark, args):
    """Clicks in, training pairs out."""
    cat = args.catalog

    # Positives: the user clicked it, so it was right.
    spark.sql(f"""
        CREATE OR REPLACE TABLE {cat}.silver.click_pairs AS
        SELECT
            concat('click_', event_id, '_', clicked_id)       AS pair_id,
            clicked_id                                        AS product_id,
            event_id                                          AS post_id,
            query_image                                       AS anchor_path,
            p.image_path                                      AS positive_path,
            e.selected_category                               AS category,
            1.0                                               AS score,
            0.0                                               AS area_frac
        FROM {cat}.bronze.search_events e
        LATERAL VIEW explode(e.clicked) c AS clicked_id
        JOIN {cat}.bronze.products p ON p.product_id = c.clicked_id
        WHERE e.query_image IS NOT NULL
          AND e.event_ts >= current_date() - INTERVAL 30 DAYS
        LIMIT {args.max_pairs_per_run}
    """)

    # Hard negatives: shown near the top and deliberately not clicked. These sit
    # exactly on the model's decision boundary, which is where learning happens.
    # Random negatives were solved weeks ago; these are the ones still wrong.
    spark.sql(f"""
        CREATE OR REPLACE TABLE {cat}.silver.hard_negatives AS
        SELECT
            e.event_id,
            e.query_image                                     AS anchor_path,
            shown_id                                          AS negative_product_id,
            pos                                               AS shown_rank
        FROM {cat}.bronze.search_events e
        LATERAL VIEW posexplode(e.results_shown) r AS pos, shown_id
        WHERE size(e.clicked) > 0
          AND NOT array_contains(e.clicked, shown_id)
          AND pos < 5
          AND e.query_image IS NOT NULL
          AND e.event_ts >= current_date() - INTERVAL 30 DAYS
    """)

    for t in ["click_pairs", "hard_negatives"]:
        print(f"  {cat}.silver.{t}: {spark.table(f'{cat}.silver.{t}').count()} rows")


def write_alerts(spark, cat, alerts):
    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {cat}.monitoring.alerts (
            payload STRING, kind STRING, severity STRING, raised_at TIMESTAMP)
    """)
    if alerts:
        (spark.createDataFrame(
            [(json.dumps(a), a["kind"], a["severity"]) for a in alerts],
            "payload STRING, kind STRING, severity STRING")
         .withColumn("raised_at", F.current_timestamp())
         .write.mode("append").saveAsTable(f"{cat}.monitoring.alerts"))

    print(f"{len(alerts)} alerts")
    for a in alerts:
        print(f"  [{a['severity']:>7}] {a['kind']}: {a.get('hint','')}")
    if any(a["severity"] == "high" for a in alerts):
        raise SystemExit("High-severity alert — see monitoring.alerts")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--catalog", required=True)
    p.add_argument("--mode", choices=["online", "drift", "harvest"], required=True)
    p.add_argument("--min-ctr", type=float, default=0.18)
    p.add_argument("--max-zero-result-rate", type=float, default=0.04)
    p.add_argument("--max-reformulation-rate", type=float, default=0.30)
    p.add_argument("--max-index-staleness-hours", type=float, default=6)
    p.add_argument("--max-p95-latency-ms", type=float, default=250)
    p.add_argument("--js-divergence-threshold", type=float, default=0.15)
    p.add_argument("--max-pairs-per-run", type=int, default=50000)
    args = p.parse_args()

    spark = SparkSession.builder.getOrCreate()
    {"online": check_online, "drift": check_drift, "harvest": harvest}[args.mode](spark, args)


if __name__ == "__main__":
    main()
