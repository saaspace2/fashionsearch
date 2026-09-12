"""
Register one model that does the whole search: photo in, ranked garment out.

WHY ONE MODEL AND NOT TWO ENDPOINTS
-----------------------------------
Two models are involved. Serving them separately would mean the application
calls the detector, reads the boxes, crops the image itself, and calls the
encoder — two network hops and cropping logic duplicated in every client.

Every client would then have to agree on which box to pick, and the moment two
of them disagree the embeddings stop being comparable in a way that is very
hard to see from the outside.

So the pair is served as one artifact. The application sends a photo and gets
boxes plus an embedding back in a single call.

INDEPENDENT VERSIONS, ONE DEPLOYABLE
------------------------------------
Both models stay registered separately in Unity Catalog, keeping their own
version histories, lineage and gate results:

    fashion_detector  v1 v2 v3 ...
    fashion_encoder   v1 v2 v3 ...
            │
            └──► fashion_search v7   pins detector v2 + encoder v5

The combined version is tagged with both component versions, so the endpoint's
lineage says exactly what it contains. The cost is that changing either model
means re-logging this artifact — a job step, not manual work.

That trade is worth stating plainly: you give up independent redeployment of
the two halves, and you get a client that does not need to know there are two.
"""

import argparse
import base64
import io
import json
import os


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--detector-dir", required=True,
                   help="Directory holding the detector's from_pretrained files")
    p.add_argument("--encoder-dir", required=True,
                   help="Directory holding config.json, preprocessor_config.json, state_dict.pt")
    p.add_argument("--manifest", required=True)
    p.add_argument("--model-name", required=True)
    p.add_argument("--detector-version", default="")
    p.add_argument("--encoder-version", default="")
    p.add_argument("--champion-alias", default="production")
    p.add_argument("--score-threshold", type=float, default=0.35)
    return p.parse_args()


def build_wrapper(mlflow, score_threshold):
    import pandas as pd

    class SearchModel(mlflow.pyfunc.PythonModel):
        """
        Input : DataFrame with a base64 'image' column
        Output: DataFrame, one row per image, with
                  detected_category  what the detector picked
                  detector_score     its confidence
                  area_frac          how much of the frame the garment fills
                  used_whole_image   True when detection found nothing
                  embedding          128 floats, unit length
        """

        def load_context(self, context):
            import os as _os
            import torch
            import torch.nn as nn
            import torch.nn.functional as F
            import torchvision.transforms as T
            from PIL import Image
            from transformers import (AutoImageProcessor,
                                      AutoModelForObjectDetection,
                                      SwinConfig, SwinModel)

            self.torch, self.Image = torch, Image

            det_dir = context.artifacts["detector"]
            self.det = AutoModelForObjectDetection.from_pretrained(det_dir).eval()
            self.det_proc = AutoImageProcessor.from_pretrained(det_dir)
            self.id2label = self.det.config.id2label

            enc_dir = context.artifacts["encoder"]

            # Defined here so unpickling never has to find the class by name —
            # a module-level class pickles by reference as __main__.X, which
            # exists nowhere else.
            class ImageEncoder(nn.Module):
                def __init__(self, cfg, dim):
                    super().__init__()
                    self.swin = SwinModel(config=cfg)
                    self.embedding_layer = nn.Linear(cfg.hidden_size, dim)

                def forward(self, pixel_values):
                    feats = self.swin(pixel_values).pooler_output
                    return F.normalize(self.embedding_layer(feats), p=2, dim=1)

            cfg = SwinConfig.from_pretrained(enc_dir)
            proc = AutoImageProcessor.from_pretrained(enc_dir)
            state = torch.load(_os.path.join(enc_dir, "state_dict.pt"),
                               map_location="cpu")
            dim = state["embedding_layer.weight"].shape[0]

            enc = ImageEncoder(cfg, dim)
            enc.load_state_dict(state)
            enc.eval()
            self.enc = enc
            self.dim = dim

            size = cfg.image_size
            self.tf = T.Compose([
                T.Resize((size, size)), T.ToTensor(),
                T.Normalize(mean=proc.image_mean, std=proc.image_std)])

            self.threshold = score_threshold

        def predict(self, context, model_input, params=None):
            """
            One row per DETECTED GARMENT, not one row per image.

            An outfit photo contains several things a user might be searching
            for. Returning only the largest, most confident one — which an
            earlier version did — means someone who photographed a full outfit
            can search for the jacket but never the trousers.

            So every box above the threshold gets its own crop, its own
            embedding and its own row, tagged with which input image it came
            from. The caller decides which to act on; the model does not decide
            for them.

            When nothing is detected at all, one row is returned for the whole
            image. That keeps the contract stable — there is always at least one
            row per input — and matches what the evaluation set measures.
            """
            rows = []
            for idx, b64 in enumerate(model_input["image"]):
                img = self.Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
                W, H = img.size

                inputs = self.det_proc(images=[img], return_tensors="pt")
                with self.torch.no_grad():
                    out = self.det(**inputs)
                res = self.det_proc.post_process_object_detection(
                    out, threshold=self.threshold,
                    target_sizes=self.torch.tensor([[H, W]]))[0]

                found = []
                for score, label, box in zip(res["scores"], res["labels"], res["boxes"]):
                    x1, y1, x2, y2 = [float(v) for v in box]
                    area = ((x2 - x1) * (y2 - y1)) / (W * H)
                    found.append({
                        "category": self.id2label[int(label)],
                        "score": float(score),
                        "area": area,
                        "crop": img.crop((x1, y1, x2, y2)),
                        "box": [x1, y1, x2, y2],
                    })

                if not found:
                    # Screenshots, flat-lays and tight crops defeat detection
                    # routinely. A worse answer beats no answer, and the eval
                    # set goes through this same path so the numbers are honest.
                    found = [{"category": None, "score": 0.0, "area": 1.0,
                              "crop": img, "box": [0.0, 0.0, float(W), float(H)]}]
                    fell_back = True
                else:
                    fell_back = False
                    # Biggest and most confident first, so a caller that only
                    # wants one gets the same answer as before.
                    found.sort(key=lambda f: f["area"] * f["score"], reverse=True)

                tensors = [self.tf(f["crop"]) for f in found]
                with self.torch.no_grad():
                    embeddings = self.enc(self.torch.stack(tensors)).cpu().numpy()

                for rank, (f, emb) in enumerate(zip(found, embeddings)):
                    rows.append({
                        "row": idx,
                        "item_index": rank,
                        "detected_category": f["category"],
                        "detector_score": f["score"],
                        "area_frac": f["area"],
                        "x1": f["box"][0], "y1": f["box"][1],
                        "x2": f["box"][2], "y2": f["box"][3],
                        "used_whole_image": fell_back,
                        "embedding": emb.tolist(),
                    })
            return pd.DataFrame(rows)

    return SearchModel


def main():
    args = parse_args()
    for var in ("DATABRICKS_HOST", "DATABRICKS_TOKEN"):
        if not os.environ.get(var):
            raise SystemExit(f"{var} is not set")

    with open(args.manifest) as fh:
        manifest = json.load(fh)

    import mlflow
    import numpy as np
    import pandas as pd
    from mlflow.models import infer_signature
    from mlflow.tracking import MlflowClient
    from PIL import Image

    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")

    catalog = args.model_name.split(".")[0]
    try:
        from databricks.sdk import WorkspaceClient
        WorkspaceClient().workspace.mkdirs(f"/Shared/fashionsearch/{catalog}")
    except Exception as exc:
        print(f"could not create the experiment folder: {exc}")
    mlflow.set_experiment(f"/Shared/fashionsearch/{catalog}/search")

    Model = build_wrapper(mlflow, args.score_threshold)

    buf = io.BytesIO()
    Image.new("RGB", (640, 480), (120, 130, 140)).save(buf, format="JPEG")
    example = pd.DataFrame({"image": [base64.b64encode(buf.getvalue()).decode()]})
    dim = int(manifest.get("embedding_dim", 128))
    out_example = pd.DataFrame([{
        "row": 0, "item_index": 0, "detected_category": "outer",
        "detector_score": 0.9, "area_frac": 0.3,
        "x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0,
        "used_whole_image": False, "embedding": [0.0] * dim}])

    with mlflow.start_run(run_name=f"search-{manifest.get('created_at','')}") as run:
        mlflow.log_params({
            "detector_repo": manifest.get("detector_repo"),
            "encoder_repo": manifest.get("encoder_repo"),
            "detector_version": args.detector_version or "unregistered",
            "encoder_version": args.encoder_version or "unregistered",
            "score_threshold": args.score_threshold,
            "embedding_dim": dim,
            "finetuned": manifest.get("finetuned"),
        })
        mlflow.pyfunc.log_model(
            artifact_path="search",
            python_model=Model(),
            artifacts={"detector": args.detector_dir, "encoder": args.encoder_dir},
            signature=infer_signature(example, out_example),
            input_example=example,
            pip_requirements=["torch", "torchvision", "transformers<5",
                              "pillow", "timm"],
        )
        model_uri = f"runs:/{run.info.run_id}/search"
    print(f"logged {model_uri}")

    result = mlflow.register_model(model_uri=model_uri, name=args.model_name)
    client = MlflowClient()

    # Tag which component versions this contains. Without it the endpoint's
    # lineage says "some detector and some encoder", which is useless the first
    # time you need to explain a regression.
    for key, value in [("detector_version", args.detector_version),
                       ("encoder_version", args.encoder_version),
                       ("gate_status", "not_evaluated")]:
        if value:
            client.set_model_version_tag(args.model_name, result.version, key, value)
    print(f"registered {args.model_name} version {result.version}")

    # Load it back before aliasing. register_model creates the version record
    # first and uploads artifacts after, so a failed upload leaves a version
    # that resolves but cannot be downloaded — which then fails three notebooks
    # later with an unrelated-looking 400.
    print("verifying it loads and both halves work ...")
    reloaded = mlflow.pyfunc.load_model(f"models:/{args.model_name}/{result.version}")
    out = reloaded.predict(example)
    assert "embedding" in out.columns, "no embedding column"
    assert "item_index" in out.columns, "no item_index — multi-item output missing"
    assert len(out) >= 1, "a model must return at least one row per image"
    assert len(out["embedding"].iloc[0]) == dim, "wrong embedding size"
    norm = float(np.linalg.norm(np.asarray(out["embedding"].iloc[0], dtype=np.float32)))
    assert abs(norm - 1.0) < 1e-2, f"embedding is not unit length: {norm}"
    print(f"  verified: {dim} numbers, unit length, detector ran")

    try:
        mlflow.pyfunc.load_model(f"models:/{args.model_name}@{args.champion_alias}")
        print(f"champion loads — left alone. The gate decides promotion.")
    except Exception:
        client.set_registered_model_alias(args.model_name, args.champion_alias,
                                          result.version)
        print(f"pointed @{args.champion_alias} at v{result.version}")

    with open("registered_search.json", "w") as fh:
        json.dump({"model": args.model_name, "version": int(result.version)}, fh)


if __name__ == "__main__":
    main()
