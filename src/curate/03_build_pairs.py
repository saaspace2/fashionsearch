"""
Assemble anchor-positive training pairs, and split them correctly.

anchor   = a crop detected in a styled post (messy, real-world)
positive = the clean catalogue thumbnail of the same product

The original project matched these by category agreement, which is the right
idea. What is added here is everything that stops the dataset degrading:
deduplication, a product-level split, and click feedback from production.

READ THIS BEFORE CHANGING THE SPLIT LOGIC
-----------------------------------------
The split must be by PRODUCT, not by row. If product X has eight crops and you
split rows at random, some of X's crops land in train and some in eval. The
model has then literally seen the eval answer during training, your Recall@20
comes out ten points too high, and you will not discover this until production
disagrees with your dashboard.

This is the single most common way visual-search evaluations end up wrong. It is
invisible in the code unless you look for it, which is why it is in a comment
block this large.
"""

import argparse

from pyspark.sql import SparkSession, functions as F, Window


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--catalog", required=True)
    p.add_argument("--dedup-threshold", type=float, default=0.97)
    p.add_argument("--eval-fraction", type=float, default=0.08)
    p.add_argument("--split-key", default="product_id")
    p.add_argument("--min-score", type=float, default=0.45)
    p.add_argument("--include-click-feedback", default="true")
    args = p.parse_args()

    spark = SparkSession.builder.getOrCreate()
    cat = args.catalog

    posts = (spark.table(f"{cat}.bronze.posts")
             .select("post_id", F.explode("linked_products").alias("product_id")))
    products = (spark.table(f"{cat}.bronze.products")
                .filter(F.col("license_ok") == True)
                .select("product_id",
                        F.col("image_path").alias("positive_path"),
                        F.col("category").alias("product_category")))
    detections = (spark.table(f"{cat}.silver.detections")
                  .filter(F.col("score") >= args.min_score))

    # Match a detection to a linked product when their categories agree.
    pairs = (detections
             .join(posts, "post_id")
             .join(products, "product_id")
             .filter(F.col("category") == F.col("product_category"))
             .select(
                 F.col("detection_id").alias("pair_id"),
                 "product_id", "post_id",
                 F.col("crop_path").alias("anchor_path"),
                 "positive_path",
                 F.col("category").alias("category"),
                 "score", "area_frac",
             ))

    # A post that links three products and detects three garments of three
    # different categories is unambiguous. One that detects two "top" boxes and
    # links two tops is not — we cannot tell which crop goes with which product,
    # and a wrong pairing is worse than no pairing. Drop those.
    ambiguous = Window.partitionBy("post_id", "category")
    pairs = (pairs
             .withColumn("_n", F.count("*").over(ambiguous))
             .filter(F.col("_n") == 1)
             .drop("_n"))

    # Click feedback from production: a clicked-and-purchased result is a real
    # positive drawn from the true query distribution, for free.
    if args.include_click_feedback.lower() == "true" and \
            spark.catalog.tableExists(f"{cat}.silver.click_pairs"):
        clicks = spark.table(f"{cat}.silver.click_pairs").select(pairs.columns)
        pairs = pairs.unionByName(clicks)
        print("  folded in click-derived pairs")

    # Deduplicate: crawled catalogues list the same product under several IDs.
    # Near-duplicates split across train and eval inflate the score exactly like
    # a bad split does.
    if spark.catalog.tableExists(f"{cat}.silver.product_embeddings"):
        dupes = spark.sql(f"""
            SELECT b.product_id AS drop_id
            FROM {cat}.silver.product_embeddings a
            JOIN {cat}.silver.product_embeddings b
              ON a.product_id < b.product_id
             AND a.category = b.category
            WHERE aggregate(zip_with(a.embedding, b.embedding, (x, y) -> x * y),
                            CAST(0.0 AS DOUBLE), (acc, v) -> acc + v)
                  >= {args.dedup_threshold}
        """).distinct()
        before = pairs.count()
        pairs = pairs.join(dupes, pairs.product_id == dupes.drop_id, "left_anti")
        print(f"  dedup removed {before - pairs.count()} pairs")

    # Deterministic split BY PRODUCT. hash() on product_id means the same product
    # always lands on the same side, across reruns and across new data arriving.
    bucket = F.abs(F.hash(F.col(args.split_key)) % F.lit(1000))
    pairs = pairs.withColumn(
        "split",
        F.when(bucket < F.lit(int(args.eval_fraction * 1000)), "eval").otherwise("train"),
    )

    (pairs.withColumn("built_at", F.current_timestamp())
          .write.mode("overwrite").option("overwriteSchema", "true")
          .saveAsTable(f"{cat}.gold.train_pairs"))

    stats = (spark.table(f"{cat}.gold.train_pairs")
             .groupBy("split", "category").count()
             .orderBy("split", "category").collect())
    for r in stats:
        print(f"  {r['split']:>5} {r['category']:>8}: {r['count']}")

    overlap = spark.sql(f"""
        SELECT count(*) AS n FROM (
            SELECT product_id FROM {cat}.gold.train_pairs WHERE split = 'train'
            INTERSECT
            SELECT product_id FROM {cat}.gold.train_pairs WHERE split = 'eval')
    """).first()["n"]
    # If this is ever non-zero the split is broken and every metric downstream
    # is meaningless. Fail loudly rather than quietly reporting a great score.
    assert overlap == 0, f"{overlap} products appear in BOTH splits — split is leaking"
    print("  split integrity check passed")


if __name__ == "__main__":
    main()
