# Databricks notebook source
# MAGIC %md
# MAGIC # 09 — Try a search (see it working)
# MAGIC
# MAGIC Notebook 06 tells you Recall@20 is 0.83. This one shows you *what that
# MAGIC actually looks like*.
# MAGIC
# MAGIC It picks real queries from the frozen eval set, runs the full pipeline —
# MAGIC detect, crop, encode, retrieve — and displays the query photo next to the
# MAGIC top results, with the correct answer outlined in green.
# MAGIC
# MAGIC You need both of these before this notebook works:
# MAGIC
# MAGIC | Prerequisite | Created by |
# MAGIC |---|---|
# MAGIC | `silver.product_embeddings` | notebook 05 |
# MAGIC | `gold.eval_queries` | notebook 02 |
# MAGIC
# MAGIC Nothing here writes to any table. It is safe to run repeatedly.

# COMMAND ----------
# MAGIC %pip install -q torch torchvision transformers pillow timm
# MAGIC %restart_python

# COMMAND ----------
import sys, pathlib
sys.path.insert(0, str(pathlib.Path.cwd().parent / "src"))
from fashionsearch.config import load_config, table

cfg = load_config()

dbutils.widgets.text("n_queries", "6", "How many queries to show")
dbutils.widgets.dropdown("category", "any", ["any", "top", "bottom", "outer",
                                             "dress", "shoes", "bag", "hat"],
                         "Restrict to one category")
dbutils.widgets.text("top_k", "8", "Results per query")

N = int(dbutils.widgets.get("n_queries"))
CATEGORY = dbutils.widgets.get("category")
TOP_K = int(dbutils.widgets.get("top_k"))

# COMMAND ----------
import base64, io
import numpy as np
import pandas as pd
from pyspark.sql import functions as F

# --- guard rails: fail with a useful message rather than a stack trace -------
for schema, name, made_by in [("silver", "product_embeddings", "notebook 05"),
                              ("gold", "eval_queries", "notebook 02")]:
    if not spark.catalog.tableExists(table(cfg, schema, name)):
        raise SystemExit(
            f"{table(cfg, schema, name)} does not exist yet. Run {made_by} first, "
            f"or run the whole pipeline:  databricks bundle run fashion_pipeline -t dev")

# COMMAND ----------
# MAGIC %md ## Load the catalogue vectors and pick some queries

# COMMAND ----------
cat_df = (spark.table(table(cfg, "silver", "product_embeddings"))
          .join(spark.table(table(cfg, "bronze", "products"))
                .select("product_id", "image_path"), "product_id")
          .select("product_id", "category", "embedding", "image_path")
          .toPandas())

E = np.stack(cat_df.embedding.values).astype(np.float32)
E /= np.linalg.norm(E, axis=1, keepdims=True)
ids = cat_df.product_id.values
cats = cat_df.category.values
paths = dict(zip(cat_df.product_id, cat_df.image_path))

print(f"catalogue: {len(cat_df)} products across {len(set(cats))} categories")

queries = spark.table(table(cfg, "gold", "eval_queries"))
if CATEGORY != "any":
    queries = queries.filter(F.col("category") == CATEGORY)
q_pdf = queries.orderBy(F.rand(seed=7)).limit(N).toPandas()

if q_pdf.empty:
    raise SystemExit(f"No eval queries for category '{CATEGORY}'. Try 'any'.")
print(f"showing {len(q_pdf)} queries")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Run the real pipeline on each query
# MAGIC
# MAGIC Detector finds the garment, the crop is encoded, and the vector is compared
# MAGIC against the catalogue — restricted to the same category, exactly as the
# MAGIC serving path does. If detection finds nothing we fall back to the whole
# MAGIC image, which is also what production does, so what you see here is what a
# MAGIC user would get.

# COMMAND ----------
import mlflow
from PIL import Image

mlflow.set_registry_uri("databricks-uc")
ALIAS = cfg.registry.aliases.champion

# Both models are pyfunc: image bytes in, a DataFrame out. No transformers here.
encoder = mlflow.pyfunc.load_model(f"models:/{cfg.registry.encoder_model}@{ALIAS}")
detector = mlflow.pyfunc.load_model(f"models:/{cfg.registry.detector_model}@{ALIAS}")


def detect(image_path):
    with open(image_path, "rb") as fh:
        b64 = base64.b64encode(fh.read()).decode()
    return detector.predict(pd.DataFrame({"image": [b64]}))


def search(image_path, category, top_k=8):
    """One full search. Returns (ranked ids, scores, the crop, whether we fell back)."""
    img = Image.open(image_path).convert("RGB")
    boxes = detect(image_path)

    fell_back = True
    crop = img
    if len(boxes):
        # Same rule the serving path uses: biggest confident thing wins.
        boxes = boxes.assign(rank=boxes.area_frac * boxes.score)
        best = boxes.loc[boxes["rank"].idxmax()]
        crop = img.crop((float(best.x1), float(best.y1),
                         float(best.x2), float(best.y2)))
        fell_back = False

    buf = io.BytesIO()
    crop.save(buf, format="JPEG", quality=92)
    emb = encoder.predict(pd.DataFrame(
        {"image": [base64.b64encode(buf.getvalue()).decode()]}))
    q = np.asarray(emb, dtype=np.float32)[0]
    q /= np.linalg.norm(q)

    mask = cats == category
    if not mask.any():
        return [], [], crop, fell_back
    sims = E[mask] @ q
    order = np.argsort(-sims)[:top_k]
    return list(ids[mask][order]), list(sims[order]), crop, fell_back

# COMMAND ----------
# MAGIC %md ## The results

# COMMAND ----------
def thumb(path, size=110):
    """Base64 thumbnail for inline HTML display."""
    try:
        im = Image.open(path).convert("RGB")
        im.thumbnail((size, size))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=85)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return ""


def pil_thumb(im, size=110):
    im = im.copy()
    im.thumbnail((size, size))
    buf = io.BytesIO()
    im.convert("RGB").save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


html = ["""
<style>
  .fs-row{display:flex;align-items:flex-start;gap:14px;padding:14px 10px;
          border-bottom:1px solid #E3E1DA;font-family:-apple-system,Helvetica,Arial}
  .fs-q{flex:0 0 auto;text-align:center}
  .fs-q img{border:2px solid #7F77DD;border-radius:6px}
  .fs-res{display:flex;gap:8px;flex-wrap:wrap}
  .fs-hit img{border:3px solid #639922;border-radius:6px}
  .fs-miss img{border:1px solid #D3D1C7;border-radius:6px}
  .fs-cap{font-size:11px;color:#5F5E5A;margin-top:3px}
  .fs-meta{font-size:12px;color:#2C2C2A;margin-bottom:6px}
  .fs-bad{color:#B3261E;font-weight:600}
  .fs-good{color:#3B6D11;font-weight:600}
</style>
<h3 style="font-family:Helvetica">Query on the left (purple), results ranked left to right.
Green outline = a correct answer.</h3>
"""]

found_at, n_shown, n_hit = [], 0, 0

for row in q_pdf.itertuples():
    relevant = set(row.relevant_ids)
    ranked, scores, crop, fell_back = search(row.query_image, row.category, TOP_K)
    n_shown += 1

    rank = next((i + 1 for i, p in enumerate(ranked) if p in relevant), None)
    if rank:
        n_hit += 1
        found_at.append(rank)
        verdict = f'<span class="fs-good">correct answer at rank {rank}</span>'
    else:
        verdict = f'<span class="fs-bad">correct answer not in top {TOP_K}</span>'

    note = " · detector found nothing, used the whole image" if fell_back else ""
    html.append(f'<div class="fs-row"><div class="fs-q">'
                f'<img src="{thumb(row.query_image, 130)}"/>'
                f'<div class="fs-cap">query<br/>{row.category}</div></div>')
    html.append(f'<div><div class="fs-meta">{verdict}{note}</div>'
                f'<div class="fs-res">')
    for pid, s in zip(ranked, scores):
        cls = "fs-hit" if pid in relevant else "fs-miss"
        html.append(f'<div class="{cls}"><img src="{thumb(paths.get(pid, ""))}"/>'
                    f'<div class="fs-cap">{s:.3f}</div></div>')
    html.append("</div></div></div>")

displayHTML("".join(html))

# COMMAND ----------
# MAGIC %md
# MAGIC ## How to read this
# MAGIC
# MAGIC **Similarity scores** run from 1.0 (identical) down to 0.0 (unrelated). In
# MAGIC practice anything above about 0.75 is usually the same garment or a very
# MAGIC close variant. If your top result sits at 0.4, the catalogue probably does
# MAGIC not contain the item at all.
# MAGIC
# MAGIC **A green box at rank 3 is a success**, not a failure. Users scroll. That
# MAGIC is exactly why Recall@20 is the headline metric rather than Recall@1.
# MAGIC
# MAGIC **Look at the misses.** If the wrong results are visually similar — same
# MAGIC colour, same cut — the encoder is working and the catalogue simply has near
# MAGIC duplicates. If they look nothing alike, something is wrong: check that the
# MAGIC crop is landing on the right garment.

# COMMAND ----------
if n_shown:
    hit_rate = n_hit / n_shown
    print(f"Correct answer inside the top {TOP_K}: {n_hit} of {n_shown} "
          f"({hit_rate:.0%})")
    if found_at:
        print(f"Median rank when found: {int(np.median(found_at))}")
    print()
    if hit_rate >= 0.7:
        print("Working as expected. Compare against gold.retrieval_metrics for the "
              "full picture — this sample is far too small to draw conclusions from.")
    else:
        print("Lower than expected. Before blaming the model, check:")
        print("  1. Did notebook 05 embed the whole catalogue? "
              f"({len(cat_df)} products loaded)")
        print("  2. Are the crops landing on the right garment? Raise n_queries "
              "and look at the pictures.")
        print("  3. Is the catalogue big enough to contain the answers? With a "
              "small sample_size the correct product may simply be absent.")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Search with your own image
# MAGIC
# MAGIC Upload any outfit photo to the query volume and point this at it. This is
# MAGIC the closest thing to being a real user.

# COMMAND ----------
dbutils.widgets.text("my_image", "", "Path to your own image")
MY = dbutils.widgets.get("my_image").strip()

if MY:
    boxes = detect(MY)
    print("Detected:")
    for r in boxes.itertuples():
        print(f"  {r.label:>8}  {r.score:.2f}  (covers {r.area_frac:.1%} of the photo)")

    blocks = []
    for cname in boxes.label.unique():
        ranked, scores, crop, _ = search(MY, cname, TOP_K)
        if not ranked:
            continue
        blocks.append(f'<div class="fs-row"><div class="fs-q">'
                      f'<img src="{pil_thumb(crop, 130)}"/>'
                      f'<div class="fs-cap">{cname}</div></div><div>'
                      f'<div class="fs-res">')
        for pid, s in zip(ranked, scores):
            blocks.append(f'<div class="fs-miss"><img src="{thumb(paths.get(pid,""))}"/>'
                          f'<div class="fs-cap">{s:.3f}</div></div>')
        blocks.append("</div></div></div>")
    displayHTML("".join(html[:1] + blocks))
else:
    print("Set the 'my_image' widget to a path such as")
    print(f"  {cfg.catalog.volumes.query_images}/my_photo.jpg")
    print("Upload via Catalog -> fashion_dev -> raw -> query_images -> Upload.")
