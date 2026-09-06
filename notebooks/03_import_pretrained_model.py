# Databricks notebook source
# MAGIC %md
# MAGIC # 03 — Import the pretrained models
# MAGIC
# MAGIC **No training happens here.** This notebook downloads the two published
# MAGIC checkpoints from Hugging Face, wraps them so MLflow can serve them, and
# MAGIC logs them as MLflow models.
# MAGIC
# MAGIC That is the whole point of this pipeline shape: the models already exist,
# MAGIC and what is missing is everything around them — versioning, evaluation,
# MAGIC promotion, serving, monitoring.
# MAGIC
# MAGIC | Checkpoint | Role |
# MAGIC |---|---|
# MAGIC | `yainage90/fashion-object-detection` | finds garments, 7 categories |
# MAGIC | `yainage90/fashion-image-feature-extractor` | crop → 128-number embedding |

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
from fashionsearch.config import load_config, ensure_experiment
from fashionsearch.compat import log_model

cfg = load_config()

import mlflow
mlflow.set_registry_uri("databricks-uc")
ensure_experiment(f"/Shared/{cfg.project.name}/import")

# COMMAND ----------
# MAGIC %md ## The encoder
# MAGIC
# MAGIC The published encoder is a Swin backbone plus a 128-dimension projection,
# MAGIC L2-normalised. The class definition has to match the author's exactly or
# MAGIC the weights will not load.

# COMMAND ----------
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoImageProcessor, SwinModel, SwinConfig
from huggingface_hub import PyTorchModelHubMixin

CKPT = cfg.pretrained.encoder.hf_repo
encoder_config = SwinConfig.from_pretrained(CKPT)
image_processor = AutoImageProcessor.from_pretrained(CKPT)


class ImageEncoder(nn.Module, PyTorchModelHubMixin):
    def __init__(self):
        super().__init__()
        self.swin = SwinModel(config=encoder_config)
        self.embedding_layer = nn.Linear(encoder_config.hidden_size, 128)

    def forward(self, image_tensor):
        features = self.swin(image_tensor).pooler_output
        embeddings = self.embedding_layer(features)
        # L2 normalise so cosine similarity is a plain dot product — which is
        # exactly what the vector index computes.
        return F.normalize(embeddings, p=2, dim=1)


encoder = ImageEncoder().from_pretrained(CKPT).eval()
print(f"loaded {CKPT}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Wrap for MLflow
# MAGIC
# MAGIC Two things matter here, and the second one cost a pipeline run.
# MAGIC
# MAGIC **Preprocessing lives inside the wrapper.** Resize and normalisation must
# MAGIC happen identically at every call site, or two callers produce embeddings
# MAGIC that are not comparable — very hard to diagnose from outside.
# MAGIC
# MAGIC **Save weights, not the object.** `torch.save(model, path)` uses plain
# MAGIC pickle, which records the class *by reference* as `__main__.ImageEncoder`.
# MAGIC Load it in another session and you get
# MAGIC
# MAGIC     AttributeError: Can't get attribute 'ImageEncoder' on <module '__main__'>
# MAGIC
# MAGIC because that name only existed in the notebook that saved it. So we save a
# MAGIC `state_dict` plus the Swin config and the processor config, and rebuild the
# MAGIC architecture inside `load_context` where the class definition is local.
# MAGIC Everything needed is in the artifact directory — no network at load time.

# COMMAND ----------
import base64, io
import numpy as np
import pandas as pd


class EncoderWrapper(mlflow.pyfunc.PythonModel):

    def load_context(self, context):
        import os
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        import torchvision.transforms as T
        from PIL import Image
        from transformers import AutoImageProcessor, SwinConfig, SwinModel

        self.torch = torch
        self.Image = Image
        d = context.artifacts["model"]

        # Defined here, not at module scope, so unpickling never has to find it.
        class ImageEncoder(nn.Module):
            def __init__(self, swin_config, embedding_dim):
                super().__init__()
                self.swin = SwinModel(config=swin_config)
                self.embedding_layer = nn.Linear(swin_config.hidden_size, embedding_dim)

            def forward(self, pixel_values):
                feats = self.swin(pixel_values).pooler_output
                emb = self.embedding_layer(feats)
                # L2 normalise so cosine similarity is a plain dot product,
                # which is what the retrieval step computes.
                return F.normalize(emb, p=2, dim=1)

        swin_config = SwinConfig.from_pretrained(d)
        processor = AutoImageProcessor.from_pretrained(d)
        state = torch.load(os.path.join(d, "state_dict.pt"), map_location="cpu")

        # Read the embedding size off the weights rather than passing it
        # separately — one fewer thing that can disagree with the checkpoint.
        embedding_dim = state["embedding_layer.weight"].shape[0]

        model = ImageEncoder(swin_config, embedding_dim)
        model.load_state_dict(state)
        model.eval()
        self.model = model
        self.embedding_dim = embedding_dim

        size = swin_config.image_size
        self.tf = T.Compose([
            T.Resize((size, size)),
            T.ToTensor(),
            T.Normalize(mean=processor.image_mean, std=processor.image_std),
        ])

    def predict(self, context, model_input, params=None):
        """model_input: DataFrame with a base64-encoded 'image' column."""
        tensors = []
        for b64 in model_input["image"]:
            img = self.Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
            tensors.append(self.tf(img))
        batch = self.torch.stack(tensors)
        with self.torch.no_grad():
            emb = self.model(batch).cpu().numpy()
        return pd.DataFrame(emb)

# COMMAND ----------
import os
import tempfile
from PIL import Image
from mlflow.models import infer_signature

# A self-contained artifact directory: architecture config, preprocessing
# config, and weights. No reference to any class in this notebook.
enc_dir = tempfile.mkdtemp()
encoder_config.save_pretrained(enc_dir)          # config.json
image_processor.save_pretrained(enc_dir)         # preprocessor_config.json
torch.save(encoder.state_dict(), os.path.join(enc_dir, "state_dict.pt"))
print("encoder artifacts:", sorted(os.listdir(enc_dir)))

buf = io.BytesIO()
Image.new("RGB", (224, 224), (128, 128, 128)).save(buf, format="JPEG")
example = pd.DataFrame({"image": [base64.b64encode(buf.getvalue()).decode()]})
output_example = pd.DataFrame(
    np.zeros((1, cfg.pretrained.encoder.embedding_dim), dtype=np.float32))

with mlflow.start_run(run_name="import-encoder") as run:
    mlflow.log_params({
        "source": CKPT,
        "embedding_dim": cfg.pretrained.encoder.embedding_dim,
        "image_size": encoder_config.image_size,
        "trained_by": "yainage90 (imported, not trained here)",
        "serialisation": "state_dict (not a pickled module)",
    })
    encoder_uri = log_model(
        mlflow.pyfunc, "encoder",
        python_model=EncoderWrapper(),
        artifacts={"model": enc_dir},
        signature=infer_signature(example, output_example),
        input_example=example,
        pip_requirements=["torch", "torchvision", "transformers", "pillow"],
    )
    print("logged", encoder_uri)

# COMMAND ----------
# MAGIC %md
# MAGIC ### Verify it round-trips before moving on
# MAGIC
# MAGIC Loading in a fresh process is the only way to catch a serialisation bug
# MAGIC like the one above. Doing it here costs seconds; discovering it in
# MAGIC notebook 05 costs a whole pipeline run.

# COMMAND ----------
reloaded = mlflow.pyfunc.load_model(encoder_uri)
check = reloaded.predict(example)
assert check.shape[1] == cfg.pretrained.encoder.embedding_dim, (
    f"expected {cfg.pretrained.encoder.embedding_dim} dimensions, got {check.shape[1]}")
norm = float(np.linalg.norm(np.asarray(check, dtype=np.float32)[0]))
assert abs(norm - 1.0) < 1e-3, f"embeddings should be unit length, got {norm:.4f}"
print(f"round-trip OK: {check.shape[1]} dims, norm {norm:.4f}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## The detector
# MAGIC
# MAGIC Logged as a **pyfunc**, not with `mlflow.transformers`.
# MAGIC
# MAGIC That flavour imports `FlaxPreTrainedModel`, which newer versions of
# MAGIC transformers removed along with Flax support, so it fails with an
# MAGIC ImportError on any recent runtime. Pinning transformers to an old version
# MAGIC would work today and break later.
# MAGIC
# MAGIC A pyfunc wrapper avoids the flavour entirely and has a second benefit: the
# MAGIC box filtering, thresholding and area calculation live inside the model
# MAGIC artifact, so every caller gets identical behaviour instead of each one
# MAGIC re-implementing post-processing slightly differently.

# COMMAND ----------
from transformers import AutoModelForObjectDetection

DET_CKPT = cfg.pretrained.detector.hf_repo

# Load explicitly rather than letting a failure here surface later as a
# confusing NameError on a variable that was never assigned.
try:
    detector = AutoModelForObjectDetection.from_pretrained(DET_CKPT).eval()
    det_processor = AutoImageProcessor.from_pretrained(DET_CKPT)
except ImportError as exc:
    raise SystemExit(
        f"Could not load {DET_CKPT}: {exc}\n\n"
        f"This detector needs 'timm' — conditional DETR builds its ResNet-50 "
        f"backbone through TimmBackbone. Add timm to the %pip line at the top of "
        f"this notebook AND to the 'torch' environment in "
        f"resources/jobs_pipeline.yml, then re-run.") from exc

print("categories:", detector.config.id2label)

# COMMAND ----------
DET_THRESHOLD = 0.35


class DetectorWrapper(mlflow.pyfunc.PythonModel):
    """Image bytes in, one row per detected garment out."""

    def load_context(self, context):
        import torch
        from PIL import Image
        from transformers import AutoImageProcessor, AutoModelForObjectDetection
        self.torch = torch
        self.Image = Image
        self.processor = AutoImageProcessor.from_pretrained(context.artifacts["model"])
        self.model = AutoModelForObjectDetection.from_pretrained(
            context.artifacts["model"]).eval()
        self.id2label = self.model.config.id2label

    def predict(self, context, model_input, params=None):
        rows = []
        for idx, b64 in enumerate(model_input["image"]):
            img = self.Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
            W, H = img.size
            inputs = self.processor(images=[img], return_tensors="pt")
            with self.torch.no_grad():
                out = self.model(**inputs)
            res = self.processor.post_process_object_detection(
                out, threshold=DET_THRESHOLD,
                target_sizes=self.torch.tensor([[H, W]]))[0]

            for score, label, box in zip(res["scores"], res["labels"], res["boxes"]):
                x1, y1, x2, y2 = [float(v) for v in box]
                rows.append({
                    "row": idx,
                    "label": self.id2label[int(label)],
                    "score": float(score),
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    # Area fraction feeds the size_band eval slice: a small box
                    # is a distant or partly hidden garment, which is the hard case.
                    "area_frac": ((x2 - x1) * (y2 - y1)) / (W * H),
                })
        cols = ["row", "label", "score", "x1", "y1", "x2", "y2", "area_frac"]
        return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)

# COMMAND ----------
det_dir = tempfile.mkdtemp()
detector.save_pretrained(det_dir)
det_processor.save_pretrained(det_dir)

det_out_example = pd.DataFrame([{
    "row": 0, "label": "outer", "score": 0.9,
    "x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0, "area_frac": 0.1}])

with mlflow.start_run(run_name="import-detector") as run:
    mlflow.log_params({
        "source": DET_CKPT,
        "n_classes": len(detector.config.id2label),
        "score_threshold": DET_THRESHOLD,
        "flavor": "pyfunc (not mlflow.transformers)",
    })
    detector_uri = log_model(
        mlflow.pyfunc, "detector",
        python_model=DetectorWrapper(),
        artifacts={"model": det_dir},
        signature=infer_signature(example, det_out_example),
        input_example=example,
        pip_requirements=["torch", "torchvision", "transformers", "pillow", "timm"],
    )
    print("logged", detector_uri)

# COMMAND ----------
dbutils.jobs.taskValues.set("encoder_uri", encoder_uri)
dbutils.jobs.taskValues.set("detector_uri", detector_uri)
print("Next: 04_register_model")
