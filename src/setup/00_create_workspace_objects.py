"""
One-time workspace setup: create the catalog, schemas and volumes that every
other job assumes already exist.

Run this FIRST. Safe to run repeatedly — everything is IF NOT EXISTS.
"""

import argparse
from pyspark.sql import SparkSession

SCHEMAS = ["raw", "bronze", "silver", "gold", "ml", "monitoring"]

VOLUMES = [
    ("raw",    "product_images", "Catalogue thumbnails"),
    ("raw",    "post_images",    "Styled / user-generated outfit photos"),
    ("raw",    "query_images",   "Real production queries, sampled"),
    ("ml",     "artifacts",      "Exported models and index snapshots"),
    # Where GitHub Actions drops what the Kaggle GPU run produced. Notebook 05b
    # also creates it, but 05b runs long after the upload — so it has to exist
    # here, before anything tries to write to it.
    ("silver", "kaggle_inbox",   "Embeddings and manifests from the Kaggle run"),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--catalog", default="fashion_dev")
    args = p.parse_args()

    spark = SparkSession.builder.getOrCreate()
    cat = args.catalog

    print(f"Creating catalog {cat} ...")
    spark.sql(f"CREATE CATALOG IF NOT EXISTS {cat} "
              f"COMMENT 'FashionSearch visual search project'")

    for s in SCHEMAS:
        print(f"  schema  {cat}.{s}")
        spark.sql(f"CREATE SCHEMA IF NOT EXISTS {cat}.{s}")

    for schema, name, comment in VOLUMES:
        print(f"  volume  {cat}.{schema}.{name}")
        spark.sql(f"CREATE VOLUME IF NOT EXISTS {cat}.{schema}.{name} "
                  f"COMMENT '{comment}'")

    print("\nDone. Next: upload product thumbnails to "
          f"/Volumes/{cat}/raw/product_images/ and run fashion_data_pipeline.")


if __name__ == "__main__":
    main()
