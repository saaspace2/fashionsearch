# Databricks notebook source
# MAGIC %md
# MAGIC # 05 — Batch inference
# MAGIC
# MAGIC Embeds the catalogue and the eval queries, and records the `size_band`
# MAGIC slice label that lets notebook 06 score small items separately.
# MAGIC
# MAGIC ## Why this runs on the driver, not across executors
# MAGIC
# MAGIC An earlier version used `mapInPandas` so Spark could parallelise. It failed:
# MAGIC `mlflow.pyfunc.load_model()` inside an executor downloads artifacts relative
# MAGIC to the working directory, which on serverless is the read-only workspace.
# MAGIC
# MAGIC That is fixable with a writable `dst_path`, but the shape was wrong anyway.
# MAGIC Distributing means every executor downloads a 348 MB model before doing any
# MAGIC work — for 2,000 images the download dominates. A plain driver loop loads
# MAGIC each model once and is both simpler and faster at this size.
# MAGIC
# MAGIC **This does not scale to two million products.** At that point you want
# MAGIC `mapInPandas` with `dst_path="/local_disk0/model"` and a GPU. The trade is
# MAGIC deliberate and worth revisiting when `sample_size` grows past ~50,000.

# COMMAND ----------
# MAGIC %md
# MAGIC Dependencies come from the job's `environments:` block in
# MAGIC `resources/jobs_pipeline.yml`. Deliberately no `%pip install` here.

# COMMAND ----------
import sys, pathlib
sys.path.insert(0, str(pathlib.Path.cwd().parent / "src"))
from fashionsearch.config import load_config, table
from fashionsearch import registry

cfg = load_config()

import base64, io, time
import numpy as np
import pandas as pd
import mlflow
from PIL import Image
from pyspark.sql import functions as F, types as T

CHAMPION = cfg.registry.aliases.champion

# Work out whether there is anything to do BEFORE loading anything.
#
# When notebook 05b has already ingested Kaggle's embeddings there is no work
# here, and loading two models to discover that wastes minutes — or fails
# outright on a workspace that cannot read model artifacts from a notebook.
_n_products = spark.table(table(cfg, "bronze", "products")).count()
_n_embedded = (spark.table(table(cfg, "silver", "product_embeddings")).count()
               if spark.catalog.tableExists(table(cfg, "silver", "product_embeddings"))
               else 0)
_n_queries = spark.table(table(cfg, "gold", "eval_queries")).count()
_n_q_embedded = (spark.table(table(cfg, "silver", "query_embeddings")).count()
                 if spark.catalog.tableExists(table(cfg, "silver", "query_embeddings"))
                 else 0)

print(f"products {_n_embedded}/{_n_products} embedded, "
      f"queries {_n_q_embedded}/{_n_queries} embedded")

if _n_embedded >= _n_products and _n_q_embedded >= _n_queries and _n_products:
    dbutils.notebook.exit(
        "nothing to embed — Kaggle already supplied everything (see notebook 05b)")

# Only now, and via the loader that falls back to the volume when this
# workspace cannot read registered model artifacts.
from fashionsearch import local_models

encoder = local_models.load(cfg, "encoder", CHAMPION)
detector = local_models.load(cfg, "detector", CHAMPION)
ENCODER = registry.resolve(cfg, cfg.registry.encoder_model, CHAMPION)
print("both models loaded")

# Serverless gives you several cores; torch does not always use them by default.
try:
    import torch, os
    torch.set_num_threads(os.cpu_count() or 4)
    print(f"torch threads: {torch.get_num_threads()}")
except Exception as exc:
    print(f"could not set thread count: {exc}")

# COMMAND ----------
# MAGIC %md ## Helpers

# COMMAND ----------
def b64_of(path: str) -> str:
    with open(path, "rb") as fh:
        return base64.b64encode(fh.read()).decode()


def b64_of_image(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=92)
    return base64.b64encode(buf.getvalue()).decode()


def embed_batch(b64_list):
    """Encoder returns a DataFrame of shape (n, 128)."""
    out = encoder.predict(pd.DataFrame({"image": b64_list}))
    return np.asarray(out, dtype=np.float32)


def progress(done, total, started, label):
    """Long steps look hung without this. Print every 200 items."""
    if done % 200 and done != total:
        return
    elapsed = time.time() - started
    rate = done / elapsed if elapsed else 0
    remaining = (total - done) / rate if rate else 0
    print(f"  {label}: {done}/{total}  {rate:.1f}/s  ~{remaining/60:.1f} min left")

# COMMAND ----------
# MAGIC %md ## Embed the catalogue

# COMMAND ----------
# The expensive part of this notebook is embedding, and most runs change
# nothing about it. Re-embed only when the encoder version changed, or for
# products that have never been embedded.
#
# ENCODER is a runs:/<run_id>/encoder URI, so it changes whenever notebook 03
# logs a new model. Using it as the cache key means a new encoder correctly
# invalidates every stored vector — which it must, since embeddings from two
# different encoders are not comparable.
all_products = spark.table(table(cfg, "bronze", "products")).select(
    "product_id", "image_path")

existing = None
if spark.catalog.tableExists(table(cfg, "silver", "product_embeddings")):
    existing = spark.table(table(cfg, "silver", "product_embeddings"))
    if "encoder_uri" in existing.columns:
        existing = existing.filter(F.col("encoder_uri") == ENCODER)
    else:
        existing = None      # written before we tracked this; treat as stale

if existing is not None and existing.count():
    todo = all_products.join(existing.select("product_id"), "product_id", "left_anti")
    print(f"{existing.count()} products already embedded with this encoder")
else:
    todo = all_products
    print("no reusable embeddings — encoder changed, or this is the first run")

products = todo.toPandas()
print(f"embedding {len(products)} products")

if len(products) == 0:
    print("nothing to do — every product is already embedded with this encoder")

BATCH = 48   # larger batches amortise the per-call overhead on CPU
ids, vecs = [], []
started = time.time()

for start in range(0, len(products), BATCH):
    chunk = products.iloc[start:start + BATCH]
    payload, kept = [], []
    for r in chunk.itertuples():
        try:
            payload.append(b64_of(r.image_path))
            kept.append(r.product_id)
        except Exception:
            continue          # a missing file should not kill the run
    if not payload:
        continue
    emb = embed_batch(payload)
    ids.extend(kept)
    vecs.extend(emb.tolist())
    progress(len(ids), len(products), started, "catalogue")

print(f"embedded {len(ids)} products in {(time.time()-started)/60:.1f} min")

# COMMAND ----------
if ids:
    emb_df = spark.createDataFrame(
        pd.DataFrame({"product_id": ids, "embedding": vecs}))

    new_rows = (emb_df
        .join(spark.table(table(cfg, "bronze", "products"))
              .select("product_id", "category", "in_stock", "region", "brand", "price"),
              "product_id")
        .withColumn("embedded_at", F.current_timestamp())
        .withColumn("encoder_alias", F.lit(CHAMPION))
        .withColumn("encoder_uri", F.lit(ENCODER)))

    if existing is not None and existing.count():
        # Add to what is already there.
        new_rows.write.mode("append").saveAsTable(
            table(cfg, "silver", "product_embeddings"))
    else:
        # Encoder changed: the old vectors are not comparable to the new ones,
        # so replace rather than mix them.
        new_rows.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(
            table(cfg, "silver", "product_embeddings"))

print("silver.product_embeddings:",
      spark.table(table(cfg, "silver", "product_embeddings")).count())

# COMMAND ----------
# MAGIC %md
# MAGIC ## Embed the eval queries, and label their slices
# MAGIC
# MAGIC `size_band` comes from how much of the frame the detected garment fills. A
# MAGIC bag covering 3% of a photo has very few pixels to identify it by, and that
# MAGIC is exactly the case an aggregate metric hides.
# MAGIC
# MAGIC When the detector finds nothing we embed the whole image. Production does
# MAGIC the same, so the eval set measures what a user would actually get rather
# MAGIC than a flattering ideal.

# COMMAND ----------
all_queries = spark.table(table(cfg, "gold", "eval_queries")).select(
    "post_id", "query_image")

q_existing = None
if spark.catalog.tableExists(table(cfg, "silver", "query_embeddings")):
    qe = spark.table(table(cfg, "silver", "query_embeddings"))
    if "encoder_uri" in qe.columns:
        q_existing = qe.filter(F.col("encoder_uri") == ENCODER)

if q_existing is not None and q_existing.count():
    q_todo = all_queries.join(q_existing.select("post_id"), "post_id", "left_anti")
    print(f"{q_existing.count()} queries already processed with this encoder")
else:
    q_todo = all_queries

queries = q_todo.toPandas()
print(f"processing {len(queries)} eval queries")

rows = []
started = time.time()
fell_back = 0

for i, r in enumerate(queries.itertuples(), start=1):
    try:
        img = Image.open(r.query_image).convert("RGB")
    except Exception:
        continue

    boxes = detector.predict(pd.DataFrame({"image": [b64_of(r.query_image)]}))

    if len(boxes):
        # Biggest confident thing wins — the item the photo is about, not a
        # shoe in the corner.
        boxes = boxes.assign(rank=boxes.area_frac * boxes.score)
        best = boxes.loc[boxes["rank"].idxmax()]
        area = float(best.area_frac)
        crop = img.crop((float(best.x1), float(best.y1),
                         float(best.x2), float(best.y2)))
    else:
        area, crop = 1.0, img
        fell_back += 1

    size_band = "small" if area < 0.08 else "medium" if area < 0.30 else "large"
    occlusion = "heavy" if area < 0.05 else "none"

    emb = embed_batch([b64_of_image(crop)])[0]
    rows.append((r.post_id, emb.tolist(), size_band, occlusion))
    progress(i, len(queries), started, "queries")

print(f"processed {len(rows)} queries in {(time.time()-started)/60:.1f} min")
print(f"detector found nothing on {fell_back} of them "
      f"({fell_back/max(len(rows),1):.0%}) — those used the whole image")

# COMMAND ----------
Q_SCHEMA = T.StructType([
    T.StructField("post_id", T.StringType()),
    T.StructField("embedding", T.ArrayType(T.FloatType())),
    T.StructField("size_band", T.StringType()),
    T.StructField("occlusion", T.StringType()),
])

if rows:
    q_new = (spark.createDataFrame(rows, schema=Q_SCHEMA)
             .withColumn("encoder_uri", F.lit(ENCODER)))
    if q_existing is not None and q_existing.count():
        q_new.write.mode("append").saveAsTable(
            table(cfg, "silver", "query_embeddings"))
    else:
        q_new.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(
            table(cfg, "silver", "query_embeddings"))

# Fold the derived slice labels back into the frozen eval set.
spark.sql(f"""
    MERGE INTO {table(cfg, "gold", "eval_queries")} AS t
    USING {table(cfg, "silver", "query_embeddings")} AS s
    ON t.post_id = s.post_id
    WHEN MATCHED THEN UPDATE SET
        t.size_band = s.size_band,
        t.occlusion = s.occlusion
""")

display(spark.sql(f"""
    SELECT size_band, occlusion, count(*) AS n
    FROM {table(cfg, "gold", "eval_queries")}
    GROUP BY size_band, occlusion ORDER BY n DESC
"""))
