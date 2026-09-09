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
# MAGIC ## Champion selection
# MAGIC
# MAGIC Normally the gate in notebook 06 decides promotion and this notebook leaves
# MAGIC the champion alone. Two exceptions:
# MAGIC
# MAGIC 1. **No champion yet.** Nothing to compare against, so the first version
# MAGIC    becomes champion and every version after it faces a real comparison.
# MAGIC 2. **The champion cannot be loaded.** That is not a quality judgement, it is
# MAGIC    a broken artifact — a bad serialisation, a deleted run, an expired
# MAGIC    upload. Leaving it in place blocks the whole pipeline on a model nobody
# MAGIC    can use, and no amount of evaluation will fix it.
# MAGIC
# MAGIC Case 2 is why this exists. An earlier version of notebook 03 saved the
# MAGIC encoder with `torch.save(model, path)`, which pickles the class by
# MAGIC reference. Every later notebook failed with
# MAGIC `AttributeError: Can't get attribute 'ImageEncoder'`, and re-running 03 did
# MAGIC not help because the alias still pointed at the broken version.

# COMMAND ----------
import mlflow

CHAMPION = cfg.registry.aliases.champion


def champion_status(name: str):
    """
    Returns (state, detail). state is 'missing', 'broken', 'unverifiable' or 'ok'.

    The distinction between 'broken' and 'unverifiable' matters. A
    ModuleNotFoundError means THIS notebook lacks torch — it says nothing about
    the model. Treating that as "broken" made an earlier version repoint the
    champion on every single run, quietly climbing to v6.
    """
    try:
        uri = registry.resolve(cfg, name, CHAMPION)
    except SystemExit:
        return "missing", "no champion alias set"

    try:
        mlflow.pyfunc.load_model(uri)
        return "ok", uri
    except ModuleNotFoundError as exc:
        return "unverifiable", (
            f"cannot check {uri} from this environment: {exc}. "
            f"Give task 04 the 'torch' environment in jobs_pipeline.yml.")
    except Exception as exc:
        # A 400 on HeadObject is this workspace being unable to READ artifacts
        # written from outside — not a broken model. Repointing the alias would
        # not help, because every version has the same problem.
        if "400" in str(exc) or "HeadObject" in str(exc):
            return "unverifiable", (
                f"cannot read {uri} from a notebook on this workspace "
                f"(HeadObject 400). The artifacts are fine; this tier cannot "
                f"fetch externally-registered model files.")
        return "broken", f"{uri} does not load: {type(exc).__name__}: {exc}"


for name, result in [(cfg.registry.encoder_model, enc),
                     (cfg.registry.detector_model, det)]:
    state, detail = champion_status(name)

    if state == "ok":
        print(f"{name}: champion loads fine ({detail}) — leaving it alone.")
        print("  Promotion of the new version is the gate's decision, not this notebook's.")
    elif state == "unverifiable":
        # Leave the champion alone. Repointing on the strength of a check we
        # could not run would be worse than not checking.
        print(f"{name}: champion left in place, could not verify it.")
        print(f"  {detail}")
    elif state == "missing":
        registry.set_alias(cfg, name, CHAMPION, result["version"])
        print(f"{name}: no champion existed — bootstrapped @{CHAMPION} "
              f"= v{result['version']}")
    else:
        print(f"{name}: EXISTING CHAMPION IS BROKEN")
        print(f"  {detail}")
        registry.set_alias(cfg, name, CHAMPION, result["version"])
        registry.set_tag(cfg, name, result["version"], "promoted_reason",
                         "previous champion could not be loaded")
        print(f"  repointed @{CHAMPION} to v{result['version']} (this run's import)")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Can this workspace read the model?
# MAGIC
# MAGIC Reported, not enforced — and the distinction matters.
# MAGIC
# MAGIC When models are registered from outside Databricks (GitHub Actions, in
# MAGIC this pipeline), their artifacts are written through the REST API. Some
# MAGIC workspace tiers then cannot read them back from a serverless notebook:
# MAGIC
# MAGIC     400 Bad Request when calling the HeadObject operation
# MAGIC
# MAGIC That is the mirror of the write problem — the cluster's assumed role is
# MAGIC denied on this storage in both directions. The same artifacts download
# MAGIC perfectly from a GitHub runner.
# MAGIC
# MAGIC **The pipeline does not need Databricks to load the models.** Embeddings
# MAGIC are computed on Kaggle and ingested by 05b as Parquet. Evaluation in 06
# MAGIC works on vectors, not models. Serving loads the model in the serving
# MAGIC container, not here.
# MAGIC
# MAGIC So a failure here is worth knowing about and must not stop the run.
# MAGIC Notebooks 09 and 10 are the ones that genuinely need a local load, and
# MAGIC they say so themselves if it fails.

# COMMAND ----------
import mlflow

readable = {}
for name in [cfg.registry.encoder_model, cfg.registry.detector_model]:
    uri = registry.resolve(cfg, name, CHAMPION)
    try:
        mlflow.pyfunc.load_model(uri)
        readable[name] = True
        print(f"  [OK]      {name} @{CHAMPION} loads here")
    except Exception as exc:
        readable[name] = False
        print(f"  [CANNOT]  {name} @{CHAMPION} does not load in this notebook")
        print(f"            {type(exc).__name__}: {str(exc)[:160]}")

if all(readable.values()):
    print("\nBoth models load. Notebook 05 can embed locally if it needs to.")
else:
    print("\nThis workspace cannot read the model artifacts from a notebook.")
    print("Expected when models are registered from outside Databricks.")
    print("")
    print("Unaffected: 05b (reads Parquet), 06 (scores vectors), 07 (the serving")
    print("            container loads the model, not this notebook).")
    print("Affected:   05 local embedding, and notebooks 09 and 10.")
    print("            With Kaggle supplying embeddings, none of those are on")
    print("            the pipeline's critical path.")

spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {cfg.catalog.name}.monitoring.model_readability (
        checked_at TIMESTAMP, model_name STRING, alias STRING, readable BOOLEAN)
""")
from pyspark.sql import functions as _F
(spark.createDataFrame([(n, CHAMPION, bool(v)) for n, v in readable.items()],
                       "model_name STRING, alias STRING, readable BOOLEAN")
 .withColumn("checked_at", _F.current_timestamp())
 .write.mode("append").saveAsTable(f"{cfg.catalog.name}.monitoring.model_readability"))

# COMMAND ----------
if enc["mode"] == "fallback":
    display(spark.table(f"{cfg.catalog.name}.ml.model_pointers"))

dbutils.jobs.taskValues.set("encoder_version", enc["version"])
dbutils.jobs.taskValues.set("detector_version", det["version"])
