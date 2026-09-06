# Databricks notebook source
# MAGIC %md
# MAGIC # 05 — Batch inference
# MAGIC
# MAGIC Embeds the whole catalogue with the registered encoder and writes the
# MAGIC vectors to Delta. Also runs the detector over the eval query images so
# MAGIC that notebook 06 can label each query with a `size_band` and `occlusion`
# MAGIC slice — those labels are what turn one number into a diagnosis.
# MAGIC
# MAGIC On CPU this is the slow step. Roughly 8–15 images/second, so 2000 products
# MAGIC takes a few minutes. Scale `sample_size` in `config.yaml` accordingly.

# COMMAND ----------
# MAGIC %pip install -q torch torchvision transformers pillow timm
# MAGIC %restart_python

# COMMAND ----------
import sys, pathlib
sys.path.insert(0, str(pathlib.Path.cwd().parent / "src"))
from fashionsearch.config import load_config, table

cfg = load_config()

import mlflow, base64, io
import numpy as np, pandas as pd
from pyspark.sql import functions as F, types as T

mlflow.set_registry_uri("databricks-uc")
ENCODER = f"models:/{cfg.registry.encoder_model}@{cfg.registry.aliases.champion}"

# COMMAND ----------
# MAGIC %md ## Embed the catalogue

# COMMAND ----------
EMB_SCHEMA = T.StructType([
    T.StructField("product_id", T.StringType()),
    T.StructField("embedding", T.ArrayType(T.FloatType())),
])


def make_embedder(model_uri, batch_size=32):
    def embed(iterator):
        model = mlflow.pyfunc.load_model(model_uri)
        for pdf in iterator:
            ids, vecs = [], []
            for start in range(0, len(pdf), batch_size):
                chunk = pdf.iloc[start:start + batch_size]
                b64, kept = [], []
                for r in chunk.itertuples():
                    try:
                        with open(r.image_path, "rb") as fh:
                            b64.append(base64.b64encode(fh.read()).decode())
                        kept.append(r.product_id)
                    except Exception:
                        continue
                if not b64:
                    continue
                out = model.predict(pd.DataFrame({"image": b64}))
                ids.extend(kept)
                vecs.extend(np.asarray(out, dtype=np.float32).tolist())
            yield pd.DataFrame({"product_id": ids, "embedding": vecs})
    return embed


products = spark.table(table(cfg, "bronze", "products")).select("product_id", "image_path")
n = products.count()
print(f"embedding {n} products")

embeddings = (products.repartition(max(1, n // 500))
              .mapInPandas(make_embedder(ENCODER), schema=EMB_SCHEMA))

(embeddings
 .join(spark.table(table(cfg, "bronze", "products"))
       .select("product_id", "category", "in_stock", "region", "brand", "price"),
       "product_id")
 .withColumn("embedded_at", F.current_timestamp())
 .withColumn("encoder_alias", F.lit(cfg.registry.aliases.champion))
 .write.mode("overwrite").option("overwriteSchema", "true")
 .saveAsTable(table(cfg, "silver", "product_embeddings")))

print("done:", spark.table(table(cfg, "silver", "product_embeddings")).count())

# COMMAND ----------
# MAGIC %md
# MAGIC ## Embed the eval queries, and label their slices
# MAGIC
# MAGIC `size_band` comes from how much of the frame the detected garment fills.
# MAGIC Small items are the hard case — a bag occupying 3% of a photo has very few
# MAGIC pixels to identify it by — and they are exactly what an aggregate metric
# MAGIC hides.

# COMMAND ----------
Q_SCHEMA = T.StructType([
    T.StructField("post_id", T.StringType()),
    T.StructField("embedding", T.ArrayType(T.FloatType())),
    T.StructField("size_band", T.StringType()),
    T.StructField("occlusion", T.StringType()),
])


def make_query_processor(encoder_uri, detector_name, alias):
    def process(iterator):
        import torch
        from PIL import Image
        from transformers import AutoImageProcessor, AutoModelForObjectDetection

        enc = mlflow.pyfunc.load_model(encoder_uri)
        det_uri = f"models:/{detector_name}@{alias}"
        det = mlflow.transformers.load_model(det_uri, return_type="components")
        model, processor = det["model"], det["image_processor"]
        model.eval()

        for pdf in iterator:
            rows = []
            for r in pdf.itertuples():
                try:
                    img = Image.open(r.query_image).convert("RGB")
                except Exception:
                    continue
                W, H = img.size

                inputs = processor(images=[img], return_tensors="pt")
                with torch.no_grad():
                    out = model(**inputs)
                res = processor.post_process_object_detection(
                    out, threshold=0.35, target_sizes=torch.tensor([[H, W]]))[0]

                if len(res["boxes"]):
                    best = int(res["scores"].argmax())
                    x1, y1, x2, y2 = [float(v) for v in res["boxes"][best]]
                    area = ((x2 - x1) * (y2 - y1)) / (W * H)
                    crop = img.crop((x1, y1, x2, y2))
                else:
                    # No detection: fall back to the whole image. This is the
                    # same fallback the serving path uses, so the eval set
                    # measures what production actually does.
                    area, crop = 1.0, img

                size_band = ("small" if area < 0.08
                             else "medium" if area < 0.30 else "large")
                occlusion = "heavy" if area < 0.05 else "none"

                buf = io.BytesIO()
                crop.save(buf, format="JPEG", quality=92)
                emb = enc.predict(pd.DataFrame(
                    {"image": [base64.b64encode(buf.getvalue()).decode()]}))
                rows.append((r.post_id,
                             np.asarray(emb, dtype=np.float32)[0].tolist(),
                             size_band, occlusion))
            yield pd.DataFrame(rows, columns=[f.name for f in Q_SCHEMA.fields])
    return process


queries = spark.table(table(cfg, "gold", "eval_queries")).select("post_id", "query_image")
q_out = (queries.repartition(8)
         .mapInPandas(make_query_processor(ENCODER, cfg.registry.detector_model,
                                           cfg.registry.aliases.champion),
                      schema=Q_SCHEMA))

(q_out.write.mode("overwrite").option("overwriteSchema", "true")
 .saveAsTable(table(cfg, "silver", "query_embeddings")))

# COMMAND ----------
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
