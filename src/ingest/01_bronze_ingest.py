"""
Bronze ingest: register catalogue products and styled posts.

Two sources, kept separate because they behave differently:

  products — the shop catalogue. Clean studio thumbnails, one per item, with
             structured metadata (category, price, stock, brand). This is what
             gets embedded into the search index.

  posts    — styled or user-generated photos, each linked to one or more
             products. These are messy: multiple garments, odd angles, poor
             light, partial occlusion. They are where the ANCHOR crops come
             from, and the messiness is the point — it is what the encoder has
             to learn to see through.

Also creates the search_events table that production logging writes into. It is
created empty here so that the monitoring and feedback jobs have something to
read against from day one rather than failing on a missing table.
"""

import argparse

from pyspark.sql import SparkSession, functions as F


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--catalog", required=True)
    args = p.parse_args()

    spark = SparkSession.builder.getOrCreate()
    cat = args.catalog

    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {cat}.bronze.products (
            product_id     STRING  COMMENT 'Stable catalogue identifier',
            image_path     STRING  COMMENT 'UC Volume path to the thumbnail',
            category       STRING  COMMENT 'top|bottom|outer|dress|shoes|bag|hat',
            brand          STRING,
            title          STRING,
            price          DOUBLE,
            currency       STRING,
            in_stock       BOOLEAN,
            region         STRING,
            source         STRING  COMMENT 'Where this listing came from',
            license_ok     BOOLEAN COMMENT 'Cleared for model training use',
            ingested_at    TIMESTAMP,
            updated_at     TIMESTAMP
        ) USING DELTA
        TBLPROPERTIES (delta.enableChangeDataFeed = true)
    """)

    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {cat}.bronze.posts (
            post_id        STRING,
            image_path     STRING,
            linked_products ARRAY<STRING> COMMENT 'product_ids this post references',
            source         STRING,
            captured_at    TIMESTAMP,
            license_ok     BOOLEAN,
            ingested_at    TIMESTAMP
        ) USING DELTA
    """)

    # Production search logs. Every row is one query; the results shown and
    # which of them were clicked are what close the feedback loop.
    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {cat}.bronze.search_events (
            event_id       STRING,
            event_ts       TIMESTAMP,
            session_id     STRING,
            query_image    STRING  COMMENT 'Volume path, only if the user consented',
            selected_box   STRUCT<x1: DOUBLE, y1: DOUBLE, x2: DOUBLE, y2: DOUBLE>,
            selected_category STRING,
            model_version  STRING,
            index_version  STRING,
            results_shown  ARRAY<STRING> COMMENT 'product_ids, in rank order',
            clicked        ARRAY<STRING> COMMENT 'product_ids clicked',
            added_to_cart  ARRAY<STRING>,
            latency_ms     DOUBLE,
            n_results      INT,
            reformulated   BOOLEAN COMMENT 'User searched again within the session',
            query_condition STRING COMMENT 'catalogue_shot|styled_post|phone_photo|screenshot'
        ) USING DELTA
        PARTITIONED BY (DATE(event_ts))
    """)

    counts = {t: spark.table(f"{cat}.bronze.{t}").count()
              for t in ["products", "posts", "search_events"]}
    for t, n in counts.items():
        print(f"  bronze.{t}: {n} rows")

    if counts["products"] == 0:
        print("\nNo products yet. Load the catalogue by writing rows into "
              f"{cat}.bronze.products with image_path pointing at "
              f"/Volumes/{cat}/raw/product_images/...")


if __name__ == "__main__":
    main()
