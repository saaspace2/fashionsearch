# Databricks notebook source
# MAGIC %md
# MAGIC # 04 — Register the models
# MAGIC
# MAGIC Registers what notebook 03 logged, so later notebooks can ask for
# MAGIC "the current champion" rather than hardcoding a run id.
# MAGIC
# MAGIC **No alias is set by the gate's rules here.** A freshly registered version
# MAGIC is just a numbered artifact. Only the gate in notebook 06 may tag
# MAGIC `@candidate`, and only a human moves `@shadow` → `@production`.
# MAGIC
# MAGIC ## If Unity Catalog registration fails
# MAGIC
# MAGIC It will, on Databricks Free Edition. Registering to UC copies artifacts
# MAGIC into UC managed storage, which needs dedicated (single-user) compute, and
# MAGIC Free Edition is serverless-only. You get an S3 AccessDenied.
# MAGIC
# MAGIC The pipeline does not stop. `fashionsearch.registry` falls back to a Delta
# MAGIC pointer table holding name, version, alias and run URI. You lose UC
# MAGIC governance; you keep versions, aliases and the audit trail the gate needs.
# MAGIC The notebook prints which mode it used.

# COMMAND ----------
import sys, pathlib
sys.path.insert(0, str(pathlib.Path.cwd().parent / "src"))
from fashionsearch.config import load_config
from fashionsearch import registry

cfg = load_config()

# COMMAND ----------
try:
    encoder_uri = dbutils.jobs.taskValues.get("03_import_pretrained_model", "encoder_uri")
    detector_uri = dbutils.jobs.taskValues.get("03_import_pretrained_model", "detector_uri")
except Exception:
    dbutils.widgets.text("encoder_uri", "")
    dbutils.widgets.text("detector_uri", "")
    encoder_uri = dbutils.widgets.get("encoder_uri")
    detector_uri = dbutils.widgets.get("detector_uri")

print("encoder :", encoder_uri)
print("detector:", detector_uri)
assert encoder_uri and detector_uri, "Run notebook 03 first — it produces both URIs."

# COMMAND ----------
enc = registry.register(
    cfg, encoder_uri, cfg.registry.encoder_model,
    "Fashion image encoder. Swin backbone + 128-d projection, L2-normalised. "
    "Imported from yainage90/fashion-image-feature-extractor, not trained here.")

det = registry.register(
    cfg, detector_uri, cfg.registry.detector_model,
    "Fashion object detector, 7 categories, logged as pyfunc. Imported from "
    "yainage90/fashion-object-detection.")

print(f"\nregistration mode: {enc['mode']}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Bootstrap the champion
# MAGIC
# MAGIC With no champion there is nothing for the gate to compare against, so on a
# MAGIC completely empty registry the first version becomes champion. Every version
# MAGIC after that faces a real comparison.

# COMMAND ----------
CHAMPION = cfg.registry.aliases.champion

for name, result in [(cfg.registry.encoder_model, enc),
                     (cfg.registry.detector_model, det)]:
    try:
        existing = registry.resolve(cfg, name, CHAMPION)
        print(f"{name}: champion already set ({existing}) — leaving it")
    except SystemExit:
        registry.set_alias(cfg, name, CHAMPION, result["version"])
        print(f"{name}: bootstrapped @{CHAMPION} = v{result['version']}")

# COMMAND ----------
if enc["mode"] == "fallback":
    display(spark.table(f"{cfg.catalog.name}.ml.model_pointers"))

dbutils.jobs.taskValues.set("encoder_version", enc["version"])
dbutils.jobs.taskValues.set("detector_version", det["version"])
