# Databricks notebook source
# MAGIC %md
# MAGIC # 10 — Run a search through the serving endpoint
# MAGIC
# MAGIC This is the real thing: an image goes over HTTP to a deployed model, an
# MAGIC embedding comes back, and the catalogue is searched with it. Exactly what
# MAGIC an application would do.
# MAGIC
# MAGIC ## What it needs
# MAGIC
# MAGIC A running endpoint, which notebook 07 creates. That needs the models to be
# MAGIC in **Unity Catalog** — Model Serving cannot serve a `runs:/` URI.
# MAGIC
# MAGIC If UC registration is still refused on your workspace, this notebook says
# MAGIC so plainly and falls back to calling the model in-process. The search logic
# MAGIC below is identical either way; only the transport differs.

# COMMAND ----------
# MAGIC %md
# MAGIC Dependencies come from the job's `environments:` block. If you open this
# MAGIC notebook by hand instead, run this first:
# MAGIC
# MAGIC     %pip install -q torch torchvision transformers timm pillow
# MAGIC     %restart_python

# COMMAND ----------
import sys, pathlib
sys.path.insert(0, str(pathlib.Path.cwd().parent / "src"))
from fashionsearch.config import load_config, table
from fashionsearch import registry

cfg = load_config()

import base64, io, json, time
import numpy as np
import pandas as pd
import mlflow
from PIL import Image
from pyspark.sql import functions as F

dbutils.widgets.text("n_queries", "5", "How many searches to run")
dbutils.widgets.text("top_k", "8", "Results per search")
N = int(dbutils.widgets.get("n_queries"))
TOP_K = int(dbutils.widgets.get("top_k"))

ENDPOINT = cfg.serving.endpoint_name

# COMMAND ----------
# MAGIC %md ## Is the endpoint up?

# COMMAND ----------
from databricks.sdk import WorkspaceClient

USE_ENDPOINT = False
try:
    w = WorkspaceClient()
    names = [e.name for e in w.serving_endpoints.list()]
    if ENDPOINT in names:
        state = w.serving_endpoints.get(ENDPOINT).state
        print(f"endpoint '{ENDPOINT}' exists — state: {state}")
        USE_ENDPOINT = True
    else:
        print(f"endpoint '{ENDPOINT}' not found.")
        print(f"existing endpoints: {names or 'none'}")
        print("\nRun notebook 07 to create it. If 07 reports that Unity Catalog")
        print("registration is unavailable, Model Serving cannot be used on this")
        print("workspace and the fallback below is your option.")
except Exception as exc:
    print(f"Model Serving unavailable: {type(exc).__name__}: {exc}")

print(f"\nmode: {'HTTP endpoint' if USE_ENDPOINT else 'in-process fallback'}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## One function, two transports
# MAGIC
# MAGIC `embed()` hides whether the model is behind HTTP or loaded locally. That is
# MAGIC deliberate — everything downstream is then identical, so a result you see
# MAGIC here is the result the application would get.

# COMMAND ----------
if not USE_ENDPOINT:
    ALIAS = cfg.registry.aliases.champion
    from fashionsearch import local_models
    _encoder = local_models.load(cfg, "encoder", ALIAS)
    _detector = local_models.load(cfg, "detector", ALIAS)


def b64_of(path):
    with open(path, "rb") as fh:
        return base64.b64encode(fh.read()).decode()


def b64_of_image(img):
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=92)
    return base64.b64encode(buf.getvalue()).decode()


def embed(b64: str) -> np.ndarray:
    """Returns a 128-number unit vector, via HTTP or locally."""
    if USE_ENDPOINT:
        resp = w.serving_endpoints.query(
            name=ENDPOINT, dataframe_records=[{"image": b64}])
        return np.asarray(resp.predictions[0], dtype=np.float32)
    out = _encoder.predict(pd.DataFrame({"image": [b64]}))
    return np.asarray(out, dtype=np.float32)[0]


def detect(b64: str) -> pd.DataFrame:
    """The detector is not served — it runs locally in both modes.

    Only the encoder needs an endpoint: a real application detects garments
    client-side or in its own service, then asks for one embedding per crop.
    """
    return _detector.predict(pd.DataFrame({"image": [b64]})) if not USE_ENDPOINT \
        else _detector_fallback(b64)


def _detector_fallback(b64):
    global _detector
    if "_detector" not in globals():
        _detector = mlflow.pyfunc.load_model(registry.resolve(
            cfg, cfg.registry.detector_model, cfg.registry.aliases.champion))
    return _detector.predict(pd.DataFrame({"image": [b64]}))

# COMMAND ----------
# MAGIC %md ## Load the catalogue vectors

# COMMAND ----------
cat_df = (spark.table(table(cfg, "silver", "product_embeddings"))
          .join(spark.table(table(cfg, "bronze", "products"))
                .select("product_id", "image_path"), "product_id")
          .select("product_id", "category", "embedding", "image_path").toPandas())

E = np.stack(cat_df.embedding.values).astype(np.float32)
E /= np.linalg.norm(E, axis=1, keepdims=True)
ids, cats = cat_df.product_id.values, cat_df.category.values
paths = dict(zip(cat_df.product_id, cat_df.image_path))
print(f"catalogue: {len(cat_df)} products, {len(set(cats))} categories")

# COMMAND ----------
# MAGIC %md ## Search

# COMMAND ----------
def search(image_path, top_k=TOP_K):
    """The full request: detect, crop, embed, retrieve. Returns a JSON-shaped dict."""
    t0 = time.perf_counter()
    img = Image.open(image_path).convert("RGB")
    raw = b64_of(image_path)

    boxes = detect(raw)
    if len(boxes):
        boxes = boxes.assign(rank=boxes.area_frac * boxes.score)
        best = boxes.loc[boxes["rank"].idxmax()]
        crop = img.crop((float(best.x1), float(best.y1),
                         float(best.x2), float(best.y2)))
        category, fell_back = str(best.label), False
    else:
        crop, category, fell_back = img, None, True

    q = embed(b64_of_image(crop))
    q = q / np.linalg.norm(q)

    mask = cats == category if category in set(cats) else np.ones(len(cats), bool)
    sims = E[mask] @ q
    order = np.argsort(-sims)[:top_k]

    return {
        "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        "detected_category": category,
        "used_whole_image": fell_back,
        "results": [{"product_id": str(p), "score": round(float(s), 4)}
                    for p, s in zip(ids[mask][order], sims[order])],
        "_crop": crop,
    }

# COMMAND ----------
# Only rows with a real image on disk. Most rows carry a kaggle://
# marker because Kaggle ships a sample of pictures, not all 10,000 —
# opening a marker raises FileNotFoundError.
queries = (spark.table(table(cfg, "gold", "eval_queries"))
           .filter(~F.col("query_image").startswith("kaggle://"))
           .orderBy(F.rand(seed=11)).limit(N).toPandas())
if queries.empty:
    dbutils.notebook.exit(
        "no eval queries have a viewable image yet — re-run the Kaggle job so "
        "it uploads sample_images/, then run this again")

def thumb(path, size=110):
    try:
        im = Image.open(path).convert("RGB"); im.thumbnail((size, size))
        buf = io.BytesIO(); im.save(buf, format="JPEG", quality=85)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return ""

def pil_thumb(im, size=110):
    im = im.copy(); im.thumbnail((size, size))
    buf = io.BytesIO(); im.convert("RGB").save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

html = ["""<style>
 .r{display:flex;gap:14px;padding:14px 6px;border-bottom:1px solid #E3E1DA;
    font-family:-apple-system,Helvetica,Arial}
 .q img{border:2px solid #7F77DD;border-radius:6px}
 .hit img{border:3px solid #639922;border-radius:6px}
 .miss img{border:1px solid #D3D1C7;border-radius:6px}
 .c{font-size:11px;color:#5F5E5A;margin-top:3px;text-align:center}
 .m{font-size:12px;margin-bottom:6px}
</style>"""]

latencies, hits = [], 0
for row in queries.itertuples():
    r = search(row.query_image)
    latencies.append(r["latency_ms"])
    relevant = set(row.relevant_ids)
    rank = next((i + 1 for i, x in enumerate(r["results"])
                 if x["product_id"] in relevant), None)
    if rank:
        hits += 1

    verdict = (f"<b style='color:#3B6D11'>correct at rank {rank}</b>" if rank
               else f"<b style='color:#B3261E'>not in top {TOP_K}</b>")
    note = " · detector found nothing" if r["used_whole_image"] else ""
    html.append(f"<div class='r'><div class='q'><img src='{pil_thumb(r['_crop'],130)}'/>"
                f"<div class='c'>{r['detected_category'] or 'unknown'}</div></div><div>"
                f"<div class='m'>{verdict}{note} · {r['latency_ms']} ms</div><div style='display:flex;gap:8px'>")
    for x in r["results"]:
        cls = "hit" if x["product_id"] in relevant else "miss"
        html.append(f"<div class='{cls}'><img src='{thumb(paths.get(x['product_id'],''))}'/>"
                    f"<div class='c'>{x['score']:.3f}</div></div>")
    html.append("</div></div></div>")

displayHTML("".join(html))

# COMMAND ----------
latencies.sort()
print(f"mode          : {'HTTP endpoint' if USE_ENDPOINT else 'in-process'}")
print(f"searches      : {len(latencies)}")
print(f"correct in top {TOP_K}: {hits}/{len(latencies)}")
print(f"latency p50   : {latencies[len(latencies)//2]:.0f} ms")
print(f"latency p95   : {latencies[int(0.95*len(latencies))-1]:.0f} ms")
if not USE_ENDPOINT:
    print("\nIn-process timing excludes the network hop, so real endpoint latency")
    print("will be higher. Treat this as a floor.")

# COMMAND ----------
# MAGIC %md ## The raw response, as an application would receive it

# COMMAND ----------
sample = search(queries.iloc[0].query_image)
sample.pop("_crop")
print(json.dumps(sample, indent=2))
