# Databricks notebook source
# MAGIC %md
# MAGIC # 04 — Register in Unity Catalog
# MAGIC
# MAGIC Takes the models logged in notebook 03 and registers them as versioned UC
# MAGIC models. **No alias is set here.** A freshly registered version is just a
# MAGIC numbered artifact; only the gate in notebook 06 may tag `@candidate`, and
# MAGIC only a human moves `@shadow` → `@production`.
# MAGIC
# MAGIC That separation is the whole point. If registration also promoted, there
# MAGIC would be nothing standing between "it ran" and "users see it".

# COMMAND ----------
import sys, pathlib
sys.path.insert(0, str(pathlib.Path.cwd().parent / "src"))
from fashionsearch.config import load_config

cfg = load_config()

import mlflow
from mlflow.tracking import MlflowClient
mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()

# COMMAND ----------
try:
    encoder_uri = dbutils.jobs.taskValues.get("03_import_pretrained_model", "encoder_uri")
    detector_uri = dbutils.jobs.taskValues.get("03_import_pretrained_model", "detector_uri")
except Exception:
    dbutils.widgets.text("encoder_uri", "")
    dbutils.widgets.text("detector_uri", "")
    encoder_uri = dbutils.widgets.get("encoder_uri")
    detector_uri = dbutils.widgets.get("detector_uri")

print(encoder_uri, detector_uri, sep="\n")

# COMMAND ----------
def register(uri: str, name: str, description: str):
    version = mlflow.register_model(model_uri=uri, name=name)
    client.update_registered_model(name=name, description=description)
    client.set_model_version_tag(name, version.version, "source", "huggingface")
    client.set_model_version_tag(name, version.version, "gate_status", "not_evaluated")
    print(f"{name} -> version {version.version}")
    return version

enc = register(
    encoder_uri, cfg.registry.encoder_model,
    "Fashion image encoder. Swin backbone + 128-d projection, L2-normalised. "
    "Imported from yainage90/fashion-image-feature-extractor, not trained here.")

det = register(
    detector_uri, cfg.registry.detector_model,
    "Fashion object detector, 7 categories. Imported from "
    "yainage90/fashion-object-detection.")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Bootstrap only: the first version becomes the champion
# MAGIC
# MAGIC With no champion there is nothing to compare against, so the gate has no
# MAGIC baseline. On a completely empty registry we set `@production` once, so
# MAGIC that every subsequent version faces a real comparison.

# COMMAND ----------
for model_name, version in [(cfg.registry.encoder_model, enc),
                            (cfg.registry.detector_model, det)]:
    try:
        current = client.get_model_version_by_alias(
            model_name, cfg.registry.aliases.champion)
        print(f"{model_name}: champion is already v{current.version} — leaving it")
    except Exception:
        client.set_registered_model_alias(
            model_name, cfg.registry.aliases.champion, version.version)
        print(f"{model_name}: bootstrapped @{cfg.registry.aliases.champion} "
              f"= v{version.version}")

# COMMAND ----------
dbutils.jobs.taskValues.set("encoder_version", enc.version)
dbutils.jobs.taskValues.set("detector_version", det.version)
