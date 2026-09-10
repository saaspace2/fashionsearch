"""
Load the models from a Unity Catalog Volume instead of the model registry.

WHY THIS EXISTS
---------------
This workspace cannot read model artifacts from a notebook:

    MlflowException ... An error occurred (400) when calling the HeadObject
    operation: Bad Request

That is the mirror of the write problem. Models registered from outside
Databricks — GitHub Actions, in this pipeline — have their artifacts written
through the REST API, and the serverless cluster's assumed role is denied on
that storage in both directions.

But UC Volumes read perfectly well. Notebook 05b reads Parquet from
kaggle_inbox without trouble. Only the model-artifact path is blocked.

So GitHub uploads the raw model files into a volume alongside the embeddings,
and this module rebuilds the models from there.

WHAT THIS IS AND IS NOT
-----------------------
Unity Catalog remains the registry: versions, aliases, lineage, the gate's
record of what was promoted when. None of that changes.

The volume copy is a readable duplicate, for the one workspace tier that cannot
fetch the registered artifacts. On a workspace where models:/ works, these
functions use it and never touch the volume.

The registry stays the source of truth. This is a reading route, not a second
registry.
"""

from __future__ import annotations

import os


def artifact_dir(cfg, which: str) -> str:
    """Where GitHub Actions puts the raw model files."""
    return f"/Volumes/{cfg.catalog.name}/silver/kaggle_inbox/{which}_artifact"


def _build_encoder(directory: str):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision.transforms as T
    from transformers import AutoImageProcessor, SwinConfig, SwinModel

    class ImageEncoder(nn.Module):
        def __init__(self, cfg_, dim):
            super().__init__()
            self.swin = SwinModel(config=cfg_)
            self.embedding_layer = nn.Linear(cfg_.hidden_size, dim)

        def forward(self, pixel_values):
            feats = self.swin(pixel_values).pooler_output
            # L2 normalise so cosine similarity is a plain dot product.
            return F.normalize(self.embedding_layer(feats), p=2, dim=1)

    swin_cfg = SwinConfig.from_pretrained(directory)
    processor = AutoImageProcessor.from_pretrained(directory)
    state = torch.load(os.path.join(directory, "state_dict.pt"), map_location="cpu")

    # Read the size off the weights rather than passing it separately — one
    # fewer value that can disagree with the checkpoint.
    dim = state["embedding_layer.weight"].shape[0]

    model = ImageEncoder(swin_cfg, dim)
    model.load_state_dict(state)
    model.eval()

    size = swin_cfg.image_size
    transform = T.Compose([
        T.Resize((size, size)), T.ToTensor(),
        T.Normalize(mean=processor.image_mean, std=processor.image_std)])

    return model, transform, dim


class VolumeEncoder:
    """Same interface as the pyfunc model: base64 images in, DataFrame out."""

    def __init__(self, directory: str):
        import torch
        self.torch = torch
        self.model, self.tf, self.dim = _build_encoder(directory)

    def predict(self, model_input):
        import base64
        import io
        import pandas as pd
        from PIL import Image

        tensors = []
        for b64 in model_input["image"]:
            img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
            tensors.append(self.tf(img))
        with self.torch.no_grad():
            emb = self.model(self.torch.stack(tensors)).cpu().numpy()
        return pd.DataFrame(emb)


class VolumeDetector:
    """Same interface as the pyfunc detector: base64 images in, one row per box."""

    def __init__(self, directory: str, threshold: float = 0.35):
        import torch
        from transformers import AutoImageProcessor, AutoModelForObjectDetection

        self.torch = torch
        self.proc = AutoImageProcessor.from_pretrained(directory)
        self.model = AutoModelForObjectDetection.from_pretrained(directory).eval()
        self.id2label = self.model.config.id2label
        self.threshold = threshold

    def predict(self, model_input):
        import base64
        import io
        import pandas as pd
        from PIL import Image

        rows = []
        for idx, b64 in enumerate(model_input["image"]):
            img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
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


def load(cfg, which: str, alias: str | None = None):
    """
    Get a usable model, registry first, volume second.

    The registry is tried first so that a workspace where models:/ works keeps
    full lineage on every load. The volume is the fallback for the tier that
    cannot fetch registered artifacts.

    Raises with both failures named if neither route works — otherwise you get
    a bare 400 and no indication that a second route was even attempted.
    """
    import mlflow

    from fashionsearch import registry

    alias = alias or cfg.registry.aliases.champion
    model_name = (cfg.registry.encoder_model if which == "encoder"
                  else cfg.registry.detector_model)

    registry_error = None
    try:
        uri = registry.resolve(cfg, model_name, alias)
        model = mlflow.pyfunc.load_model(uri)
        print(f"  loaded {which} from the registry: {uri}")
        return model
    except Exception as exc:
        registry_error = f"{type(exc).__name__}: {str(exc)[:200]}"
        print(f"  registry load failed for {which} — falling back to the volume")

    directory = artifact_dir(cfg, which)
    if not os.path.isdir(directory):
        raise SystemExit(
            f"Cannot load the {which}.\n\n"
            f"  registry: {registry_error}\n"
            f"  volume  : {directory} does not exist\n\n"
            f"GitHub Actions uploads the model files there after each Kaggle "
            f"run. If the directory is missing, that step has not run yet on "
            f"this workspace — push to trigger it.")

    model = VolumeEncoder(directory) if which == "encoder" else VolumeDetector(directory)
    print(f"  loaded {which} from the volume: {directory}")
    return model
