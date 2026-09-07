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
# MAGIC %md ## Verify what downstream notebooks will actually get

# COMMAND ----------
ok = True
for name in [cfg.registry.encoder_model, cfg.registry.detector_model]:
    uri = registry.resolve(cfg, name, CHAMPION)
    try:
        mlflow.pyfunc.load_model(uri)
        print(f"  [PASS] {name} @{CHAMPION} loads: {uri}")
    except ModuleNotFoundError as exc:
        ok = False
        print(f"  [SKIP] {name}: cannot verify here ({exc})")
    except Exception as exc:
        raise SystemExit(
            f"{name} @{CHAMPION} resolves to {uri} but does not load: "
            f"{type(exc).__name__}: {exc}\n\n"
            f"Notebook 05 would fail on exactly this. Re-run notebook 03 to log a "
            f"fresh version, then clear the alias so 04 bootstraps from it:\n"
            f"    DELETE FROM {cfg.catalog.name}.ml.model_pointers")

print("\nBoth champions load. Notebook 05 will use exactly these." if ok
      else "\nVerification skipped — task 04 needs the 'torch' environment to run it.")

# COMMAND ----------
if enc["mode"] == "fallback":
    display(spark.table(f"{cfg.catalog.name}.ml.model_pointers"))

dbutils.jobs.taskValues.set("encoder_version", enc["version"])
dbutils.jobs.taskValues.set("detector_version", det["version"])
