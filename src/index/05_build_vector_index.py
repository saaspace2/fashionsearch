"""
Embed the catalogue and keep the Vector Search index in sync.

The original searches by computing similarity against every one of ~15,000
thumbnails. That is brute-force search: correct, simple, and completely
impossible at two million products.

An approximate nearest neighbour index solves this by pre-building a graph of
which vectors are near which others, so a query walks toward its neighbourhood
instead of comparing against everything. It gives up a fraction of a percent of
accuracy for several orders of magnitude of speed.

Two things people underestimate:

  Metadata filters do real work. Filtering to category='outer' AND in_stock=true
  before the vector search shrinks the pool by an order of magnitude, which makes
  results both faster AND better — the index is no longer allowed to return a
  perfect match that is sold out.

  Index freshness is a metric, not an assumption. Fashion catalogues turn over
  constantly. A four-day-stale index returns dead products, which users
  experience as the search being broken.
"""

import argparse

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession, functions as F, types as T


EMB_SCHEMA = T.StructType([
    T.StructField("product_id", T.StringType()),
    T.StructField("embedding", T.ArrayType(T.FloatType())),
])


def make_embedder(model_uri, batch_size=64):

    def embed(iterator):
        import mlflow
        import torch
        import torchvision.transforms as T_
        from PIL import Image

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = mlflow.pytorch.load_model(model_uri).to(device).eval()
        tf = T_.Compose([T_.Resize((224, 224)), T_.ToTensor(),
                         T_.Normalize([0.5] * 3, [0.5] * 3)])

        for pdf in iterator:
            ids, vecs = [], []
            for start in range(0, len(pdf), batch_size):
                chunk = pdf.iloc[start:start + batch_size]
                imgs, kept = [], []
                for r in chunk.itertuples():
                    try:
                        imgs.append(tf(Image.open(r.image_path).convert("RGB")))
                        kept.append(r.product_id)
                    except Exception:
                        continue
                if not imgs:
                    continue
                x = torch.stack(imgs).to(device)
                with torch.no_grad(), torch.amp.autocast(device, dtype=torch.bfloat16):
                    emb = model(x).float().cpu().numpy()
                ids.extend(kept)
                vecs.extend(emb.tolist())
            yield pd.DataFrame({"product_id": ids, "embedding": vecs})

    return embed


def embed_catalogue(spark, args):
    cat = args.catalog
    products = (spark.table(f"{cat}.bronze.products")
                .select("product_id", "image_path", "category", "in_stock",
                        "region", "brand", "price"))

    if args.only_new.lower() == "true" and \
            spark.catalog.tableExists(f"{cat}.silver.product_embeddings"):
        done = spark.table(f"{cat}.silver.product_embeddings").select("product_id")
        products = products.join(done, "product_id", "left_anti")

    if products.isEmpty():
        print("no new products to embed")
        return

    n = products.count()
    print(f"embedding {n} products ...")

    embs = (products.select("product_id", "image_path")
            .repartition(max(1, n // 2000))
            .mapInPandas(make_embedder(f"models:/{args.encoder_model}@{args.alias}"),
                         schema=EMB_SCHEMA))

    out = (embs.join(products.drop("image_path"), "product_id")
               .withColumn("embedded_at", F.current_timestamp())
               .withColumn("encoder_alias", F.lit(args.alias)))

    (out.write.mode("append").saveAsTable(f"{cat}.silver.product_embeddings"))

    # Change Data Feed is required for a Delta Sync vector index.
    spark.sql(f"""
        ALTER TABLE {cat}.silver.product_embeddings
        SET TBLPROPERTIES (delta.enableChangeDataFeed = true)
    """)
    print(f"embedded {n} products")


def sync_index(spark, args):
    from databricks.vector_search.client import VectorSearchClient

    cat = args.catalog
    vsc = VectorSearchClient(disable_notice=True)
    source = f"{cat}.silver.product_embeddings"

    try:
        vsc.create_delta_sync_index(
            endpoint_name=args.vs_endpoint,
            index_name=args.vs_index,
            source_table_name=source,
            pipeline_type="TRIGGERED",
            primary_key="product_id",
            embedding_dimension=int(args.embedding_dim),
            embedding_vector_column="embedding",
            # These become filterable at query time. Without them you cannot
            # constrain a search to in-stock items in the user's region, and the
            # index will cheerfully return things nobody can buy.
            columns_to_sync=["product_id", "embedding", "category",
                             "in_stock", "region", "brand", "price"],
        )
        print(f"created index {args.vs_index}")
    except Exception as e:
        if "already exists" not in str(e).lower():
            raise
        vsc.get_index(args.vs_endpoint, args.vs_index).sync()
        print(f"synced index {args.vs_index}")

    # Record freshness so the monitoring job can alert on staleness rather than
    # everyone assuming the index is current.
    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {cat}.monitoring.index_syncs (
            index_name STRING, synced_at TIMESTAMP, n_vectors BIGINT)
    """)
    n = spark.table(source).count()
    spark.sql(f"""
        INSERT INTO {cat}.monitoring.index_syncs
        VALUES ('{args.vs_index}', current_timestamp(), {n})
    """)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--catalog", required=True)
    p.add_argument("--mode", choices=["embed", "sync"], required=True)
    p.add_argument("--encoder-model", default=None)
    p.add_argument("--alias", default="production")
    p.add_argument("--only-new", default="true")
    p.add_argument("--vs-endpoint", default="fashion-vs-endpoint")
    p.add_argument("--vs-index", default=None)
    p.add_argument("--embedding-dim", default="512")
    args = p.parse_args()

    spark = SparkSession.builder.getOrCreate()
    if args.mode == "embed":
        embed_catalogue(spark, args)
    else:
        sync_index(spark, args)


if __name__ == "__main__":
    main()
