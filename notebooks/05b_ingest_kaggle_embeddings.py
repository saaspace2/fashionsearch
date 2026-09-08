# Databricks notebook source
# MAGIC %md
# MAGIC # 05b — Ingest embeddings computed on Kaggle
# MAGIC
# MAGIC **This replaces notebook 05 when Kaggle has done the GPU work.**
# MAGIC
# MAGIC Notebook 05 embeds 10,000 products on serverless CPU, which takes around
# MAGIC 45 minutes. The same work on a Kaggle T4 takes two or three. So the
# MAGIC embedding moved there, and this notebook does what is left: read a
# MAGIC Parquet file and append to Delta.
# MAGIC
# MAGIC That is the entire point. **No torch, no transformers, no image bytes,
# MAGIC no model loading happens here.** Those are exactly the imports that make
# MAGIC serverless slow and memory-hungry, and none of them are needed to read a
# MAGIC table of numbers.
# MAGIC
# MAGIC | Where | Does what | Roughly |
# MAGIC |---|---|---|
# MAGIC | Kaggle T4 | fine-tune, detect, embed | 3 min |
# MAGIC | Databricks | read Parquet, append Delta | 20 s |
# MAGIC
# MAGIC If the inbox is empty this notebook exits cleanly, so the pipeline can
# MAGIC keep using notebook 05 until you switch over.

# COMMAND ----------
import sys, pathlib
sys.path.insert(0, str(pathlib.Path.cwd().parent / "src"))
from fashionsearch.config import load_config, table

cfg = load_config()

import json, os
from pyspark.sql import functions as F

INBOX = f"/Volumes/{cfg.catalog.name}/silver/kaggle_inbox"
print(f"inbox: {INBOX}")

# COMMAND ----------
# MAGIC %md ## 1 · What has Kaggle left for us?

# COMMAND ----------
spark.sql(f"CREATE VOLUME IF NOT EXISTS {cfg.catalog.name}.silver.kaggle_inbox")

if not os.path.isdir(INBOX):
    dbutils.notebook.exit("inbox does not exist yet — nothing from Kaggle")

files = sorted(os.listdir(INBOX))
manifests = [f for f in files if f.startswith("manifest_") and f.endswith(".json")]

if not manifests:
    dbutils.notebook.exit(
        "no Kaggle output found. Run the Kaggle kernel first — see kaggle/README.md")

# One run per timestamp. Take the newest; older ones stay for the audit trail.
latest = sorted(manifests)[-1]
stamp = latest.replace("manifest_", "").replace(".json", "")
print(f"newest Kaggle run: {stamp}")

with open(os.path.join(INBOX, latest)) as fh:
    manifest = json.load(fh)
print(json.dumps(manifest, indent=2))

# COMMAND ----------
# MAGIC %md
# MAGIC ## 2 · Check it before trusting it
# MAGIC
# MAGIC Kaggle is outside this pipeline's control. A kernel could have been run
# MAGIC with the wrong model, half-finished, or against a different dataset. Four
# MAGIC checks, all cheap, all worth more than the seconds they cost.

# COMMAND ----------
problems = []

expected_dim = int(cfg.pretrained.encoder.embedding_dim)
if manifest.get("embedding_dim") != expected_dim:
    problems.append(f"embedding_dim is {manifest.get('embedding_dim')}, "
                    f"config expects {expected_dim}")

if manifest.get("encoder_repo") != cfg.pretrained.encoder.hf_repo:
    problems.append(f"encoder was {manifest.get('encoder_repo')}, "
                    f"config says {cfg.pretrained.encoder.hf_repo}")

if manifest.get("device") != "cuda":
    # Not fatal, but it means the kernel ran without an accelerator and took
    # far longer than it needed to.
    print(f"NOTE: Kaggle ran on {manifest.get('device')}, not a GPU. "
          f"Tick Accelerator = GPU T4 next time.")

if manifest.get("n_products", 0) < int(cfg.data.sample_size) * 0.9:
    problems.append(f"only {manifest.get('n_products')} products, "
                    f"config asks for {cfg.data.sample_size}")

if problems:
    raise SystemExit("Kaggle output does not match this pipeline:\n  "
                     + "\n  ".join(problems)
                     + "\n\nRe-run the Kaggle kernel after aligning config.yaml.")

print("manifest checks passed")
print(f"detector found nothing on {manifest.get('fallback_rate', 0):.1%} of queries")

# COMMAND ----------
# MAGIC %md ## 3 · Load the embeddings

# COMMAND ----------
products = spark.read.parquet(f"{INBOX}/products_{stamp}.parquet")
queries = spark.read.parquet(f"{INBOX}/queries_{stamp}.parquet")
print(f"products: {products.count()}   queries: {queries.count()}")

source = f"kaggle:{stamp}"

(products
 .withColumn("embedded_at", F.current_timestamp())
 .withColumn("encoder_alias", F.lit(cfg.registry.aliases.champion))
 .withColumn("encoder_uri", F.lit(source))
 .withColumn("in_stock", F.lit(True))
 .withColumn("region", F.lit("KR"))
 .withColumn("brand", F.lit(None).cast("string"))
 .withColumn("price", F.lit(None).cast("double"))
 .write.mode("overwrite").option("overwriteSchema", "true")
 .saveAsTable(table(cfg, "silver", "product_embeddings")))

(queries.select("post_id", "embedding", "size_band", "occlusion")
 .withColumn("encoder_uri", F.lit(source))
 .write.mode("overwrite").option("overwriteSchema", "true")
 .saveAsTable(table(cfg, "silver", "query_embeddings")))

print("embeddings loaded")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 4 · Rebuild the catalogue and eval set to match
# MAGIC
# MAGIC The Kaggle run defines which products exist and which queries were
# MAGIC scored. Rebuilding bronze and gold from it keeps everything consistent —
# MAGIC otherwise notebook 06 would evaluate against a catalogue that no longer
# MAGIC matches the vectors.

# COMMAND ----------
(products.select("product_id", "category")
 .withColumn("image_path", F.concat(F.lit(f"kaggle://{stamp}/"), F.col("product_id")))
 .withColumn("in_stock", F.lit(True))
 .withColumn("region", F.lit("KR"))
 .withColumn("brand", F.lit(None).cast("string"))
 .withColumn("title", F.lit(None).cast("string"))
 .withColumn("price", F.lit(None).cast("double"))
 .withColumn("currency", F.lit("KRW"))
 .withColumn("source", F.lit(cfg.data.pairs.hf_dataset))
 .withColumn("license_ok", F.lit(True))
 .withColumn("ingested_at", F.current_timestamp())
 .withColumn("updated_at", F.current_timestamp())
 .write.mode("overwrite").option("overwriteSchema", "true")
 .saveAsTable(table(cfg, "bronze", "products")))

(queries
 .withColumn("query_image", F.concat(F.lit(f"kaggle://{stamp}/"), F.col("post_id")))
 .withColumn("relevant_ids", F.array(F.col("product_id")))
 .withColumn("relevance_map", F.map_from_arrays(
     F.array(F.col("product_id")), F.array(F.lit(1.0))))
 .withColumn("query_condition", F.lit("styled_post"))
 .withColumn("frozen_at", F.current_timestamp())
 .select("post_id", "query_image", "category", "relevant_ids", "relevance_map",
         "query_condition", "size_band", "occlusion", "frozen_at")
 .write.mode("overwrite").option("overwriteSchema", "true")
 .saveAsTable(table(cfg, "gold", "eval_queries")))

print("bronze.products and gold.eval_queries rebuilt from the Kaggle run")

# COMMAND ----------
# MAGIC %md ## 5 · Slice coverage, before the gate spends a run finding out

# COMMAND ----------
min_q = int(cfg.gate.min_queries_per_slice)
ev = spark.table(table(cfg, "gold", "eval_queries"))
total = ev.count()

print(f"eval queries: {total}   gate floor: {min_q}\n")
for dim, value in [tuple(p) for p in cfg.gate.protected_slices]:
    n = ev.filter(F.col(dim) == value).count()
    share = n / total if total else 0
    flag = "OK" if n >= min_q else "TOO FEW"
    line = f"  {dim}={value:<10} {n:>5} ({share:5.1%})  [{flag}]"
    if n < min_q and share > 0:
        line += f"  → needs eval_queries ≈ {int(min_q / share * 1.15)}"
    print(line)

display(spark.sql(f"""
    SELECT category, count(*) AS n FROM {table(cfg, "gold", "eval_queries")}
    GROUP BY category ORDER BY n DESC
"""))
