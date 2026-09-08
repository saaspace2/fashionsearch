# Databricks notebook source
# MAGIC %md
# MAGIC # 01 — Ingest data
# MAGIC
# MAGIC Downloads the original project author's published datasets from Hugging
# MAGIC Face and lands them in Unity Catalog. Databricks has outbound internet,
# MAGIC so this works even though you cannot ship the data in a repo.
# MAGIC
# MAGIC | Dataset | What it is |
# MAGIC |---|---|
# MAGIC | `yainage90/fashion-object-detection` | Images with garment boxes, 7 categories |
# MAGIC | `yainage90/onthelook-fashion-anchor-positive-images` | Anchor–positive pairs for the encoder |
# MAGIC
# MAGIC **Sampling is deliberate.** `config.yaml` defaults to 2000 images. On Free
# MAGIC Edition there is no GPU, and embedding 290k images on CPU takes days. Two
# MAGIC thousand is enough to produce real Recall@k numbers. Raise it when you
# MAGIC have GPU compute.

# COMMAND ----------
# MAGIC %md
# MAGIC Dependencies come from the job's `environments:` block in
# MAGIC `resources/jobs_pipeline.yml`. Deliberately no `%pip install` here:
# MAGIC installing again inside the notebook makes serverless build and cache a
# MAGIC per-session environment archive, and a missing archive fails the run with
# MAGIC `ENVIRONMENT_DOWNLOAD_USER_ERROR.NOT_FOUND`.
# MAGIC
# MAGIC To run this notebook interactively instead, install them by hand first.

# COMMAND ----------
import sys, pathlib
sys.path.insert(0, str(pathlib.Path.cwd().parent / "src"))

from fashionsearch.config import load_config, table
from pyspark.sql import functions as F

cfg = load_config()
cat = cfg.catalog.name

dbutils.widgets.text("sample_size", str(cfg.data.sample_size))
SAMPLE = int(dbutils.widgets.get("sample_size"))
print(f"Sampling {SAMPLE} items")

# COMMAND ----------
# MAGIC %md ## Download the anchor–positive pairs

# COMMAND ----------
dbutils.widgets.dropdown("force_refresh", "false", ["false", "true"],
                         "Re-download even if data is present")
FORCE = dbutils.widgets.get("force_refresh") == "true"

# Downloading 10,000 images takes minutes and produces the same rows every time.
# Skip when the data is already there and matches what config asks for.
already = 0
if spark.catalog.tableExists(table(cfg, "bronze", "products")):
    already = spark.table(table(cfg, "bronze", "products")).filter(
        F.col("source") == cfg.data.pairs.hf_dataset).count()

if already >= SAMPLE and not FORCE:
    print(f"{already} products already ingested from {cfg.data.pairs.hf_dataset} "
          f"(config asks for {SAMPLE}). Skipping the download.")
    print("Set the force_refresh widget to true to re-ingest anyway.")
    dbutils.notebook.exit("skipped — data already present")

if already:
    print(f"{already} products present but {SAMPLE} requested — re-ingesting.")

import io, os, hashlib
from datasets import load_dataset

pairs = load_dataset(cfg.data.pairs.hf_dataset, split=f"{cfg.data.pairs.split}[:{SAMPLE}]")
print(pairs)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Work out which columns are which
# MAGIC
# MAGIC Hugging Face datasets do not agree on column names, and this one is not
# MAGIC documented. Rather than hardcode a guess — which is exactly how the first
# MAGIC version of this notebook broke — we inspect the schema and identify the
# MAGIC columns by type and name.
# MAGIC
# MAGIC The mapping is printed below. **Check it before continuing.** If it picked
# MAGIC wrong, set the three widgets at the bottom of this cell by hand.

# COMMAND ----------
from datasets import Image as HFImage

features = pairs.features
print("Schema:")
for name, feat in features.items():
    print(f"  {name:<24} {feat}")

image_cols = [n for n, f in features.items() if isinstance(f, HFImage)]
other_cols = [n for n in features if n not in image_cols]
print(f"\nImage columns: {image_cols}")
print(f"Other columns: {other_cols}")

if len(image_cols) < 2:
    raise SystemExit(
        f"Expected two image columns (anchor and positive), found {image_cols}. "
        f"Inspect the schema above and set the widgets below by hand.")

# Anchor = the messy crop from a styled post. Positive = the clean shop thumbnail.
ANCHOR_HINTS = ("anchor", "query", "crop", "post", "street", "source", "wild")
POSITIVE_HINTS = ("positive", "target", "product", "thumb", "item", "shop", "catalog")


def pick(cols, hints, default):
    for hint in hints:
        for c in cols:
            if hint in c.lower():
                return c
    return default


anchor_col = pick(image_cols, ANCHOR_HINTS, image_cols[0])
positive_col = pick([c for c in image_cols if c != anchor_col],
                    POSITIVE_HINTS, image_cols[1] if image_cols[1] != anchor_col else image_cols[0])
category_col = pick(other_cols, ("category", "class", "label", "cat", "type"), None)

# Widgets let you override if the guesses are wrong — no code edit needed.
dbutils.widgets.text("anchor_col", anchor_col, "Anchor column")
dbutils.widgets.text("positive_col", positive_col, "Positive column")
dbutils.widgets.text("category_col", category_col or "", "Category column (optional)")

anchor_col = dbutils.widgets.get("anchor_col").strip()
positive_col = dbutils.widgets.get("positive_col").strip()
category_col = dbutils.widgets.get("category_col").strip() or None

print(f"\n  anchor   -> {anchor_col}")
print(f"  positive -> {positive_col}")
print(f"  category -> {category_col or 'MISSING — all rows will be labelled unknown'}")

assert anchor_col in features, f"'{anchor_col}' is not a column"
assert positive_col in features, f"'{positive_col}' is not a column"
assert anchor_col != positive_col, "anchor and positive must be different columns"

if category_col is None:
    print("\nWARNING: no category column found. Retrieval filters by category, so "
          "without it every query searches the whole catalogue and both Recall@k "
          "and the per-category slices will be meaningless. Check the schema above.")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Write images to the volume, metadata to Delta
# MAGIC
# MAGIC Images go to a UC Volume; Delta holds one row per item with a pointer.
# MAGIC Storing image bytes in Delta works but makes every query drag megabytes
# MAGIC around — pointers keep the tables small enough to scan freely.

# COMMAND ----------
# Hugging Face stores categories as ClassLabel integers. Writing str(0) gives
# you a catalogue of categories called "0".."6", which looks fine until the gate
# tries to score `category = 'bag'` and finds nothing. Resolve the names here.
_cat_feature = features.get(category_col) if category_col else None
_cat_names = getattr(_cat_feature, "names", None)

if _cat_names:
    print(f"category names from the dataset: {_cat_names}")
else:
    print("category column is not a ClassLabel; values will be used as-is")


def category_of(item):
    if not category_col:
        return "unknown"
    raw = item[category_col]
    if _cat_names is not None and isinstance(raw, int):
        return _cat_names[raw]
    return str(raw)


prod_dir = cfg.catalog.volumes.product_images
query_dir = cfg.catalog.volumes.query_images
os.makedirs(prod_dir, exist_ok=True)
os.makedirs(query_dir, exist_ok=True)

product_rows, post_rows = [], []
skipped = 0

for i, item in enumerate(pairs):
    pid = f"p{i:07d}"
    try:
        positive_img = item[positive_col]
        anchor_img = item[anchor_col]
        if positive_img is None or anchor_img is None:
            skipped += 1
            continue

        # positive = the clean catalogue thumbnail
        pos_path = os.path.join(prod_dir, f"{pid}.jpg")
        positive_img.convert("RGB").save(pos_path, quality=92)

        # anchor = the messy in-the-wild crop
        anc_path = os.path.join(query_dir, f"{pid}_anchor.jpg")
        anchor_img.convert("RGB").save(anc_path, quality=92)
    except Exception as e:
        skipped += 1
        if skipped <= 3:
            print(f"  skipped row {i}: {e}")
        continue

    category = category_of(item)

    product_rows.append((pid, pos_path, category, None, None, None, "KRW",
                         True, "KR", cfg.data.pairs.hf_dataset, True))
    post_rows.append((f"post_{pid}", anc_path, [pid],
                      cfg.data.pairs.hf_dataset, True))

print(f"wrote {len(product_rows)} products and {len(post_rows)} posts"
      + (f" ({skipped} rows skipped)" if skipped else ""))
assert product_rows, "Nothing was written. Check the column mapping above."

# COMMAND ----------
from pyspark.sql import functions as F

products = (spark.createDataFrame(
    product_rows,
    "product_id string, image_path string, category string, brand string, "
    "title string, price double, currency string, in_stock boolean, "
    "region string, source string, license_ok boolean")
    .withColumn("ingested_at", F.current_timestamp())
    .withColumn("updated_at", F.current_timestamp()))

products.write.mode("overwrite").option("overwriteSchema", "true") \
    .saveAsTable(table(cfg, "bronze", "products"))

posts = (spark.createDataFrame(
    post_rows,
    "post_id string, image_path string, linked_products array<string>, "
    "source string, license_ok boolean")
    .withColumn("captured_at", F.current_timestamp())
    .withColumn("ingested_at", F.current_timestamp()))

posts.write.mode("overwrite").option("overwriteSchema", "true") \
    .saveAsTable(table(cfg, "bronze", "posts"))

# COMMAND ----------
cat_counts = spark.sql(f"""
    SELECT category, count(*) AS n
    FROM {table(cfg, "bronze", "products")}
    GROUP BY category ORDER BY n DESC
""")
display(cat_counts)

numeric = [r["category"] for r in cat_counts.collect() if r["category"].isdigit()]
if numeric:
    print(f"\nWARNING: categories look numeric ({numeric[:5]}). The promotion gate "
          f"scores slices by name (bag, hat, ...), so those slices will never "
          f"match and notebook 06 will block on protected_slice_coverage.")

# COMMAND ----------
# MAGIC %md
# MAGIC ### Licensing note
# MAGIC
# MAGIC These pairs were crawled from commercial Korean fashion platforms. Fine
# MAGIC for learning; genuinely risky commercially, since most platforms' terms
# MAGIC prohibit it and a model carries its data's provenance permanently.
# MAGIC
# MAGIC That is what `license_ok` is for — set it honestly, and
# MAGIC `03_build_pairs` will filter on it.
