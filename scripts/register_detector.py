"""
Register the detector in Unity Catalog, as a pyfunc.

Deliberately NOT mlflow.transformers: that flavour imports FlaxPreTrainedModel,
which newer transformers removed along with Flax support, so it fails with an
ImportError on any recent runtime. Pinning transformers would work today and
break later.

A pyfunc wrapper also puts the threshold and box filtering inside the artifact,
so every caller behaves identically rather than each re-implementing
post-processing slightly differently.

The detector is registered separately from the encoder so each keeps its own
version history and gate results, even though they are served together as one
combined model.
"""

import argparse
import base64
import io
import json
import os


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--artifact-dir", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--model-name", required=True)
    p.add_argument("--champion-alias", default="production")
    p.add_argument("--score-threshold", type=float, default=0.35)
    return p.parse_args()


def build_wrapper(mlflow, threshold):
    import pandas as pd

    class DetectorWrapper(mlflow.pyfunc.PythonModel):
        def load_context(self, context):
            import torch
            from PIL import Image
            from transformers import AutoImageProcessor, AutoModelForObjectDetection
            self.torch, self.Image = torch, Image
            d = context.artifacts["model"]
            self.proc = AutoImageProcessor.from_pretrained(d)
            self.model = AutoModelForObjectDetection.from_pretrained(d).eval()
            self.id2label = self.model.config.id2label
            self.threshold = threshold

        def predict(self, context, model_input, params=None):
            rows = []
            for idx, b64 in enumerate(model_input["image"]):
                img = self.Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
                W, H = img.size
                inputs = self.proc(images=[img], return_tensors="pt")
                with self.torch.no_grad():
                    out = self.model(**inputs)
                res = self.proc.post_process_object_detection(
                    out, threshold=self.threshold,
                    target_sizes=self.torch.tensor([[H, W]]))[0]
                for score, label, box in zip(res["scores"], res["labels"], res["boxes"]):
                    x1, y1, x2, y2 = [float(v) for v in box]
                    rows.append({
                        "row": idx, "label": self.id2label[int(label)],
                        "score": float(score),
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "area_frac": ((x2 - x1) * (y2 - y1)) / (W * H),
                    })
            cols = ["row", "label", "score", "x1", "y1", "x2", "y2", "area_frac"]
            return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)

    return DetectorWrapper


def main():
    args = parse_args()
    for var in ("DATABRICKS_HOST", "DATABRICKS_TOKEN"):
        if not os.environ.get(var):
            raise SystemExit(f"{var} is not set")

    with open(args.manifest) as fh:
        manifest = json.load(fh)

    import mlflow
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
    except Exception:
        pass
    mlflow.set_experiment(f"/Shared/fashionsearch/{catalog}/kaggle")

    Wrapper = build_wrapper(mlflow, args.score_threshold)

    buf = io.BytesIO()
    Image.new("RGB", (640, 480), (120, 130, 140)).save(buf, format="JPEG")
    example = pd.DataFrame({"image": [base64.b64encode(buf.getvalue()).decode()]})
    out_example = pd.DataFrame([{"row": 0, "label": "outer", "score": 0.9,
                                 "x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0,
                                 "area_frac": 0.1}])

    with mlflow.start_run(run_name=f"detector-{manifest.get('created_at','')}") as run:
        mlflow.log_params({"source": manifest.get("detector_repo"),
                           "score_threshold": args.score_threshold,
                           "registered_from": "github_actions"})
        mlflow.pyfunc.log_model(
            artifact_path="detector", python_model=Wrapper(),
            artifacts={"model": args.artifact_dir},
            signature=infer_signature(example, out_example),
            input_example=example,
            pip_requirements=["torch", "torchvision", "transformers<5",
                              "pillow", "timm"])
        model_uri = f"runs:/{run.info.run_id}/detector"

    result = mlflow.register_model(model_uri=model_uri, name=args.model_name)
    client = MlflowClient()
    client.set_model_version_tag(args.model_name, result.version,
                                 "gate_status", "not_evaluated")
    print(f"registered {args.model_name} version {result.version}")

    # Load it back. register_model creates the version record before uploading
    # artifacts, so a failed upload leaves a version that resolves but cannot
    # be downloaded — which surfaces much later as an unrelated-looking 400.
    mlflow.pyfunc.load_model(f"models:/{args.model_name}/{result.version}")
    print("  verified: loads in a fresh process")

    try:
        mlflow.pyfunc.load_model(f"models:/{args.model_name}@{args.champion_alias}")
        print("champion loads — left alone")
    except Exception:
        client.set_registered_model_alias(args.model_name, args.champion_alias,
                                          result.version)
        print(f"pointed @{args.champion_alias} at v{result.version}")

    with open("registered_detector.json", "w") as fh:
        json.dump({"model": args.model_name, "version": int(result.version)}, fh)


if __name__ == "__main__":
    main()
