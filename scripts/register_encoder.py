"""
Register the encoder in Unity Catalog, from outside Databricks.

WHY FROM OUTSIDE
----------------
Registering from inside a Databricks serverless notebook uploads artifacts using
the cluster's assumed IAM role, which some workspace tiers explicitly deny:

    S3UploadFailedError: AccessDenied ... explicit deny in a resource-based policy

Registering over the REST API with a personal access token uses a different
upload path and is not subject to that deny. A GitHub Actions runner is just as
much "outside" as a Kaggle kernel, so this does the job without needing any
credentials on Kaggle at all.

INPUT
-----
An artifact directory produced by the Kaggle run, containing:
    config.json               the Swin architecture
    preprocessor_config.json  resize and normalisation
    state_dict.pt             the weights

Weights, never a pickled module. `torch.save(model, path)` records the class by
reference as `__main__.ImageEncoder`, which exists nowhere else — the model then
loads fine where it was saved and fails everywhere else with AttributeError.
"""

import argparse
import base64
import io
import json
import os
import sys


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--artifact-dir", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--model-name", required=True)
    p.add_argument("--champion-alias", default="production")
    return p.parse_args()


class EncoderWrapper:
    """Placeholder so the real class can be defined after mlflow is imported."""


def build_wrapper(mlflow):
    import pandas as pd

    class _EncoderWrapper(mlflow.pyfunc.PythonModel):
        def load_context(self, context):
            import os as _os
            import torch
            import torch.nn as nn
            import torch.nn.functional as F
            import torchvision.transforms as T
            from PIL import Image
            from transformers import AutoImageProcessor, SwinConfig, SwinModel

            self.torch, self.Image = torch, Image
            d = context.artifacts["model"]

            # Defined here, not at module scope, so unpickling never has to
            # find it by name.
            class ImageEncoder(nn.Module):
                def __init__(self, cfg, dim):
                    super().__init__()
                    self.swin = SwinModel(config=cfg)
                    self.embedding_layer = nn.Linear(cfg.hidden_size, dim)

                def forward(self, pixel_values):
                    feats = self.swin(pixel_values).pooler_output
                    return F.normalize(self.embedding_layer(feats), p=2, dim=1)

            cfg = SwinConfig.from_pretrained(d)
            proc = AutoImageProcessor.from_pretrained(d)
            state = torch.load(_os.path.join(d, "state_dict.pt"), map_location="cpu")

            # Read the dimension off the weights rather than passing it
            # separately — one fewer value that can disagree.
            dim = state["embedding_layer.weight"].shape[0]

            model = ImageEncoder(cfg, dim)
            model.load_state_dict(state)
            model.eval()
            self.model = model

            size = cfg.image_size
            self.tf = T.Compose([
                T.Resize((size, size)), T.ToTensor(),
                T.Normalize(mean=proc.image_mean, std=proc.image_std)])

        def predict(self, context, model_input, params=None):
            tensors = []
            for b64 in model_input["image"]:
                img = self.Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
                tensors.append(self.tf(img))
            with self.torch.no_grad():
                emb = self.model(self.torch.stack(tensors)).cpu().numpy()
            return pd.DataFrame(emb)

    return _EncoderWrapper


def main():
    args = parse_args()

    for var in ("DATABRICKS_HOST", "DATABRICKS_TOKEN"):
        if not os.environ.get(var):
            raise SystemExit(f"{var} is not set")

    with open(args.manifest) as fh:
        manifest = json.load(fh)
    print(json.dumps(manifest, indent=2))

    for required in ("config.json", "preprocessor_config.json", "state_dict.pt"):
        path = os.path.join(args.artifact_dir, required)
        if not os.path.exists(path):
            raise SystemExit(
                f"{required} missing from {args.artifact_dir}.\n"
                f"The Kaggle run should have written it. Check the kernel output.")

    import mlflow
    import numpy as np
    import pandas as pd
    from mlflow.models import infer_signature
    from mlflow.tracking import MlflowClient
    from PIL import Image

    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")

    catalog = args.model_name.split(".")[0]
    experiment = f"/Shared/fashionsearch/{catalog}/kaggle"
    try:
        from databricks.sdk import WorkspaceClient
        # MLflow creates experiments but not the folders above them.
        WorkspaceClient().workspace.mkdirs(f"/Shared/fashionsearch/{catalog}")
    except Exception as exc:
        print(f"could not create the experiment folder: {exc}")
    mlflow.set_experiment(experiment)

    Wrapper = build_wrapper(mlflow)

    buf = io.BytesIO()
    Image.new("RGB", (224, 224), (128, 128, 128)).save(buf, format="JPEG")
    example = pd.DataFrame({"image": [base64.b64encode(buf.getvalue()).decode()]})
    dim = int(manifest.get("embedding_dim", 128))
    out_example = pd.DataFrame(np.zeros((1, dim), dtype=np.float32))

    with mlflow.start_run(run_name=f"kaggle-{manifest.get('created_at','')}") as run:
        mlflow.log_params({
            "source": manifest.get("encoder_repo"),
            "finetuned": manifest.get("finetuned"),
            "finetune_epochs": manifest.get("finetune_epochs", 0),
            "n_products": manifest.get("n_products"),
            "trained_on": "kaggle",
            "registered_from": "github_actions",
        })
        mlflow.pyfunc.log_model(
            artifact_path="encoder",
            python_model=Wrapper(),
            artifacts={"model": args.artifact_dir},
            signature=infer_signature(example, out_example),
            input_example=example,
            pip_requirements=["torch", "torchvision", "transformers", "pillow"],
        )
        model_uri = f"runs:/{run.info.run_id}/encoder"
    print(f"logged {model_uri}")

    result = mlflow.register_model(model_uri=model_uri, name=args.model_name)
    client = MlflowClient()
    client.set_model_version_tag(args.model_name, result.version, "trained_on", "kaggle")
    client.set_model_version_tag(args.model_name, result.version,
                                 "gate_status", "not_evaluated")
    print(f"registered {args.model_name} version {result.version} in Unity Catalog")

    # Bootstrap the champion only if there is none. After that the Databricks
    # gate decides promotion — a model does not get to promote itself past
    # evaluation just because it is newer.
    try:
        current = client.get_model_version_by_alias(args.model_name, args.champion_alias)
        print(f"champion is already v{current.version} — left alone. "
              f"The gate decides whether v{result.version} replaces it.")
    except Exception:
        client.set_registered_model_alias(args.model_name, args.champion_alias,
                                          result.version)
        print(f"no champion existed — bootstrapped @{args.champion_alias} "
              f"= v{result.version}")

    with open("registered.json", "w") as fh:
        json.dump({"model": args.model_name, "version": int(result.version)}, fh)


if __name__ == "__main__":
    main()
