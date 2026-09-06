"""
Run the detector over every post image and record each detected garment.

This is the first half of pair construction. The output is one row per detected
item, with the crop written to a volume so the encoder can read it directly
without re-running detection every epoch.

Two filters matter more than they look:

  score_threshold      — a low-confidence box is usually a garment-shaped shadow
  min_box_area_frac    — a 30x40 pixel crop upscaled to 224x224 is a blur. It
                         teaches the encoder that blurs are valid garments, which
                         degrades every subsequent search.
"""

import argparse
import hashlib
import os

import pandas as pd
from pyspark.sql import SparkSession, functions as F, types as T


DET_SCHEMA = T.StructType([
    T.StructField("detection_id", T.StringType()),
    T.StructField("post_id", T.StringType()),
    T.StructField("category", T.StringType()),
    T.StructField("score", T.DoubleType()),
    T.StructField("x1", T.DoubleType()), T.StructField("y1", T.DoubleType()),
    T.StructField("x2", T.DoubleType()), T.StructField("y2", T.DoubleType()),
    T.StructField("area_frac", T.DoubleType()),
    T.StructField("crop_path", T.StringType()),
])


def make_detector(model_uri, catalog, threshold, min_area):

    def detect(iterator):
        import mlflow
        import torch
        from PIL import Image
        from transformers import AutoImageProcessor

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = mlflow.pytorch.load_model(model_uri).to(device).eval()
        processor = AutoImageProcessor.from_pretrained(model.config._name_or_path)
        id2label = model.config.id2label

        crop_dir = f"/Volumes/{catalog}/raw/post_images/_crops"
        os.makedirs(crop_dir, exist_ok=True)

        for pdf in iterator:
            rows = []
            for r in pdf.itertuples():
                try:
                    img = Image.open(r.image_path).convert("RGB")
                except Exception:
                    continue
                W, H = img.size

                inputs = processor(images=[img], return_tensors="pt").to(device)
                with torch.no_grad():
                    out = model(**inputs)
                results = processor.post_process_object_detection(
                    out, threshold=threshold,
                    target_sizes=torch.tensor([[H, W]]),
                )[0]

                for score, label, box in zip(results["scores"],
                                             results["labels"],
                                             results["boxes"]):
                    x1, y1, x2, y2 = [float(v) for v in box]
                    area_frac = ((x2 - x1) * (y2 - y1)) / float(W * H)
                    if area_frac < min_area:
                        continue

                    det_id = hashlib.sha256(
                        f"{r.post_id}|{label.item()}|{x1:.1f}|{y1:.1f}".encode()
                    ).hexdigest()[:24]

                    crop_path = os.path.join(crop_dir, f"{det_id}.webp")
                    img.crop((x1, y1, x2, y2)).save(crop_path, quality=92)

                    rows.append({
                        "detection_id": det_id,
                        "post_id": r.post_id,
                        "category": id2label[label.item()],
                        "score": float(score),
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "area_frac": area_frac,
                        "crop_path": crop_path,
                    })

            yield pd.DataFrame(rows, columns=[f.name for f in DET_SCHEMA.fields])

    return detect


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--catalog", required=True)
    p.add_argument("--detector-model", required=True)
    p.add_argument("--score-threshold", type=float, default=0.40)
    p.add_argument("--min-box-area-frac", type=float, default=0.008)
    args = p.parse_args()

    spark = SparkSession.builder.getOrCreate()
    cat = args.catalog

    posts = spark.table(f"{cat}.bronze.posts").filter(F.col("license_ok") == True)
    if spark.catalog.tableExists(f"{cat}.silver.detections"):
        done = spark.table(f"{cat}.silver.detections").select("post_id").distinct()
        posts = posts.join(done, "post_id", "left_anti")

    if posts.isEmpty():
        print("no new posts to process")
        return

    dets = (posts.select("post_id", "image_path")
            .repartition(64)
            .mapInPandas(make_detector(f"models:/{args.detector_model}@production",
                                       cat, args.score_threshold,
                                       args.min_box_area_frac),
                         schema=DET_SCHEMA))

    (dets.withColumn("detected_at", F.current_timestamp())
         .write.mode("append").saveAsTable(f"{cat}.silver.detections"))

    summary = (spark.table(f"{cat}.silver.detections")
               .groupBy("category").count().orderBy(F.desc("count")).collect())
    for r in summary:
        print(f"  {r['category']:>8}: {r['count']}")


if __name__ == "__main__":
    main()
