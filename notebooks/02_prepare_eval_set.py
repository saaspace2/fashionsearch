# Databricks notebook source
# MAGIC %md
# MAGIC # 02 — Prepare the evaluation set
# MAGIC
# MAGIC This is the notebook the original project has no equivalent of, and it is
# MAGIC the one everything downstream depends on.
# MAGIC
# MAGIC It builds a **frozen** set of queries with known-correct answers, plus the
# MAGIC slice labels the promotion gate scores against. Without it you can compare
# MAGIC two models only by looking at pictures.

# COMMAND ----------
import sys, pathlib
sys.path.insert(0, str(pathlib.Path.cwd().parent / "src"))
from fashionsearch.config import load_config, table

cfg = load_config()
from pyspark.sql import functions as F, Window

# COMMAND ----------
# MAGIC %md
# MAGIC ## Split by PRODUCT, never by row
# MAGIC
# MAGIC Several photos can show the same product. Split rows at random and some of
# MAGIC a product's photos land in train while others land in eval — the model has
# MAGIC then seen the answer, your score comes out roughly ten points too high, and
# MAGIC nothing tells you until production disagrees with the dashboard.
# MAGIC
# MAGIC Hashing the product id makes the assignment deterministic, so it survives
# MAGIC re-runs and new data arriving.

# COMMAND ----------
posts = (spark.table(table(cfg, "bronze", "posts"))
         .filter(F.col("license_ok"))
         .select("post_id",
                 F.col("image_path").alias("query_image"),
                 F.explode("linked_products").alias("product_id")))

products = (spark.table(table(cfg, "bronze", "products"))
            .select("product_id", "category", "image_path"))

pairs = posts.join(products, "product_id")

bucket = F.abs(F.hash("product_id") % F.lit(1000))
eval_cut = int(cfg.data.eval_queries / max(products.count(), 1) * 1000)
pairs = pairs.withColumn(
    "split", F.when(bucket < F.lit(max(eval_cut, 50)), "eval").otherwise("train"))

# COMMAND ----------
# MAGIC %md
# MAGIC ## Slice labels
# MAGIC
# MAGIC Recorded now, cheaply, because the gate in notebook 06 scores every metric
# MAGIC per slice. `size_band` and `occlusion` come from the detector in notebook
# MAGIC 05; here we set up the columns and derive what we can.

# COMMAND ----------
eval_set = (pairs.filter(F.col("split") == "eval")
    .groupBy("post_id", "query_image", "category")
    .agg(F.collect_set("product_id").alias("relevant_ids"))
    # Graded relevance: exact linked product scores 1.0. If you later add
    # "same style, different colour" judgments, give them 0.5 here and NDCG
    # will use the grades automatically.
    .withColumn("relevance_map",
                F.map_from_arrays(F.col("relevant_ids"),
                                  F.transform(F.col("relevant_ids"), lambda _: F.lit(1.0))))
    .withColumn("query_condition", F.lit("styled_post"))
    .withColumn("size_band", F.lit("unknown"))     # filled in by notebook 05
    .withColumn("occlusion", F.lit("unknown"))
    .withColumn("frozen_at", F.current_timestamp()))

eval_set.write.mode("overwrite").option("overwriteSchema", "true") \
    .saveAsTable(table(cfg, "gold", "eval_queries"))

pairs.filter(F.col("split") == "train").write.mode("overwrite") \
    .option("overwriteSchema", "true").saveAsTable(table(cfg, "gold", "train_pairs"))

# COMMAND ----------
# MAGIC %md ## Integrity check — fail loudly, not silently

# COMMAND ----------
overlap = spark.sql(f"""
    SELECT count(*) AS n FROM (
      SELECT product_id FROM {table(cfg, "gold", "train_pairs")}
      INTERSECT
      SELECT explode(relevant_ids) AS product_id FROM {table(cfg, "gold", "eval_queries")})
""").first()["n"]

assert overlap == 0, (
    f"{overlap} products appear in BOTH train and eval. Every metric produced "
    f"downstream would be inflated and meaningless. Fix the split before continuing.")

n_eval = spark.table(table(cfg, "gold", "eval_queries")).count()
n_train = spark.table(table(cfg, "gold", "train_pairs")).count()
print(f"split OK — {n_train} train pairs, {n_eval} eval queries")

# COMMAND ----------
display(spark.sql(f"""
    SELECT category, count(*) AS eval_queries
    FROM {table(cfg, "gold", "eval_queries")}
    GROUP BY category ORDER BY eval_queries DESC
"""))


# COMMAND ----------
# MAGIC %md
# MAGIC ## Will the promotion gate be able to measure its protected slices?
# MAGIC
# MAGIC The gate blocks on `protected_slice_coverage` when a protected slice has
# MAGIC too few queries to measure. Finding that out here costs seconds; finding
# MAGIC out in notebook 06 costs the whole pipeline run.

# COMMAND ----------
min_q = int(cfg.gate.min_queries_per_slice)
eval_df = spark.table(table(cfg, "gold", "eval_queries"))

print(f"eval queries: {eval_df.count()}   gate floor: {min_q} per protected slice\n")

problems = []
for dim, value in [tuple(p) for p in cfg.gate.protected_slices]:
    if dim not in eval_df.columns:
        print(f"  {dim}={value:<10} column not present yet (set by notebook 05)")
        continue
    n = eval_df.filter(F.col(dim) == value).count()
    ok = n >= min_q
    print(f"  {dim}={value:<10} {n:>4} queries  [{'OK' if ok else 'TOO FEW'}]")
    if not ok:
        problems.append(f"{dim}={value} ({n})")

if problems:
    print(f"\nThese will block the gate: {', '.join(problems)}")
    print("Options, in order of preference:")
    print("  1. Raise data.sample_size in config.yaml — more data, more of everything")
    print("  2. Lower gate.min_queries_per_slice — honest only if you accept that a")
    print("     recall estimate from a handful of queries is noise")
    print("  3. Edit gate.protected_slices to match what this dataset contains")
    print("\nDo NOT simply delete the hard slices. They are protected because they")
    print("are the categories an average would hide.")
