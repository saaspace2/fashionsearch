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
# MAGIC %pip install -q transformers torch torchvision huggingface_hub
# MAGIC %restart_python

# COMMAND ----------
import sys, pathlib
sys.path.insert(0, str(pathlib.Path.cwd().parent / "src"))
from fashionsearch.config import load_config

cfg = load_config()

import mlflow
mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(f"/Shared/{cfg.project.name}/import")

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
# MAGIC A `pyfunc` wrapper means the serving endpoint takes image bytes and
# MAGIC returns embeddings, rather than exposing raw tensors. Anything that has to
# MAGIC happen identically at training and serving time — here, the resize and
# MAGIC normalisation — belongs inside the wrapper, not in the caller. Otherwise
# MAGIC two callers preprocess slightly differently and the embeddings stop being
# MAGIC comparable, which is very hard to debug from the outside.

# COMMAND ----------
import base64, io
import numpy as np
import pandas as pd


class EncoderWrapper(mlflow.pyfunc.PythonModel):

    def load_context(self, context):
        import torch, torchvision.transforms as T
        from PIL import Image
        self.torch = torch
        self.Image = Image
        self.model = torch.load(context.artifacts["model"], weights_only=False)
        self.model.eval()
        size = encoder_config.image_size
        self.tf = T.Compose([
            T.Resize((size, size)),
            T.ToTensor(),
            T.Normalize(mean=image_processor.image_mean, std=image_processor.image_std),
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
import tempfile, os
from mlflow.models import infer_signature

tmp = tempfile.mkdtemp()
model_path = os.path.join(tmp, "encoder.pt")
torch.save(encoder, model_path)

# A tiny real image as the input example, so the logged signature is honest.
from PIL import Image
buf = io.BytesIO()
Image.new("RGB", (224, 224), (128, 128, 128)).save(buf, format="JPEG")
example = pd.DataFrame({"image": [base64.b64encode(buf.getvalue()).decode()]})
output_example = pd.DataFrame(np.zeros((1, cfg.pretrained.encoder.embedding_dim),
                                       dtype=np.float32))

with mlflow.start_run(run_name="import-encoder") as run:
    mlflow.log_params({
        "source": CKPT,
        "embedding_dim": cfg.pretrained.encoder.embedding_dim,
        "image_size": encoder_config.image_size,
        "trained_by": "yainage90 (imported, not trained here)",
    })
    info = mlflow.pyfunc.log_model(
        name="encoder",
        python_model=EncoderWrapper(),
        artifacts={"model": model_path},
        signature=infer_signature(example, output_example),
        input_example=example,
        pip_requirements=["torch", "torchvision", "transformers", "pillow"],
    )
    encoder_uri = info.model_uri
    print("logged", encoder_uri)

# COMMAND ----------
# MAGIC %md ## The detector

# COMMAND ----------
from transformers import AutoModelForObjectDetection

DET_CKPT = cfg.pretrained.detector.hf_repo
detector = AutoModelForObjectDetection.from_pretrained(DET_CKPT).eval()
det_processor = AutoImageProcessor.from_pretrained(DET_CKPT)
print("categories:", detector.config.id2label)

with mlflow.start_run(run_name="import-detector") as run:
    mlflow.log_params({"source": DET_CKPT,
                       "n_classes": len(detector.config.id2label)})
    det_info = mlflow.transformers.log_model(
        transformers_model={"model": detector, "image_processor": det_processor},
        name="detector",
        task="object-detection",
    )
    detector_uri = det_info.model_uri
    print("logged", detector_uri)

# COMMAND ----------
dbutils.jobs.taskValues.set("encoder_uri", encoder_uri)
dbutils.jobs.taskValues.set("detector_uri", detector_uri)
print("Next: 04_register_model")
