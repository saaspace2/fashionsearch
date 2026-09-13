# Databricks notebook source
# MAGIC %md
# MAGIC # 12 — Promote a model
# MAGIC
# MAGIC The one step that is deliberately not automated.
# MAGIC
# MAGIC ## Why you are here
# MAGIC
# MAGIC `@production` is set once, on the first version registered, purely so the
# MAGIC pipeline has something to point at. After that **every new version
# MAGIC registers and waits.**
# MAGIC
# MAGIC The gate in notebook 06 can prove a model is not worse than the champion
# MAGIC on any measured slice. It cannot know whether the business is ready for
# MAGIC it, or whether today is a sensible day to change how search behaves. So
# MAGIC it tags `@candidate` and stops.
# MAGIC
# MAGIC That means a registry full of green versions with the alias still on v1
# MAGIC is the system working, not a bug — but it does mean somebody has to come
# MAGIC here and decide.
# MAGIC
# MAGIC ## What this notebook does
# MAGIC
# MAGIC Shows every version with its gate result, loads the one you name to check
# MAGIC it actually works, then moves the alias. Every move is recorded in Unity
# MAGIC Catalog with your name and a timestamp.

# COMMAND ----------
import sys, pathlib
sys.path.insert(0, str(pathlib.Path.cwd().parent / "src"))
from fashionsearch.config import load_config, table

cfg = load_config()

import mlflow
from mlflow.tracking import MlflowClient
from pyspark.sql import functions as F

mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()

dbutils.widgets.dropdown("model", "search",
                         ["search", "encoder", "detector"], "Which model")
dbutils.widgets.text("version", "", "Version to promote (blank = just look)")
dbutils.widgets.dropdown("alias", "production",
                         ["candidate", "shadow", "production"], "Alias to move")

WHICH = dbutils.widgets.get("model")
MODEL = {"search": cfg.registry.get("search_model", cfg.registry.encoder_model),
         "encoder": cfg.registry.encoder_model,
         "detector": cfg.registry.detector_model}[WHICH]
VERSION = dbutils.widgets.get("version").strip()
ALIAS = dbutils.widgets.get("alias")

print(f"model: {MODEL}")

# COMMAND ----------
# MAGIC %md ## What is registered, and what holds which alias

# COMMAND ----------
versions = client.search_model_versions(f"name='{MODEL}'")
rows = []
for v in sorted(versions, key=lambda x: int(x.version), reverse=True):
    tags = v.tags or {}
    rows.append((int(v.version),
                 ", ".join(v.aliases) if v.aliases else "",
                 tags.get("gate_status", ""),
                 tags.get("detector_version", ""),
                 tags.get("encoder_version", ""),
                 tags.get("trained_on", ""),
                 v.creation_timestamp))

display(spark.createDataFrame(rows,
    "version INT, aliases STRING, gate_status STRING, detector_version STRING, "
    "encoder_version STRING, trained_on STRING, created_at LONG"))

current = None
try:
    current = client.get_model_version_by_alias(MODEL, ALIAS)
    print(f"@{ALIAS} currently points at v{current.version}")
except Exception:
    print(f"@{ALIAS} is not set on any version")

newest = max(int(v.version) for v in versions) if versions else None
if newest and current and int(current.version) != newest:
    print(f"\\nNewest registered version is v{newest}, but @{ALIAS} is on "
          f"v{current.version}.")
    print("That is expected — nothing promotes itself. Set the 'version' widget "
          "to promote.")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Check it works before pointing traffic at it
# MAGIC
# MAGIC Loading the model here is best-effort. On workspace tiers that cannot
# MAGIC read model artifacts from a notebook this fails with HeadObject 400 — the
# MAGIC serving container is unaffected, so that is reported rather than treated
# MAGIC as a reason to refuse.

# COMMAND ----------
if not VERSION:
    dbutils.notebook.exit("no version given — nothing changed")

try:
    mv = client.get_model_version(MODEL, VERSION)
except Exception as exc:
    raise SystemExit(f"v{VERSION} does not exist for {MODEL}: {exc}")

gate = (mv.tags or {}).get("gate_status", "unknown")
print(f"v{VERSION}  gate_status = {gate}")

if gate == "FAILED":
    raise SystemExit(
        f"v{VERSION} FAILED the promotion gate. Look at why before overriding:\\n"
        f"  SELECT * FROM {table(cfg, 'gold', 'retrieval_metrics')}\\n"
        f"  WHERE model_version = '{VERSION}'\\n\\n"
        f"If you still want it, clear the tag first — deliberately, so the "
        f"override is a decision rather than an accident.")

try:
    mlflow.pyfunc.load_model(f"models:/{MODEL}/{VERSION}")
    print("  loads cleanly in this notebook")
except Exception as exc:
    if "HeadObject" in str(exc) or "400" in str(exc):
        print("  cannot read the artifacts from this notebook — expected on this")
        print("  workspace tier. The serving container reads them fine.")
    else:
        raise SystemExit(f"v{VERSION} does not load: {type(exc).__name__}: {exc}")

# COMMAND ----------
# MAGIC %md ## Move the alias

# COMMAND ----------
previous = current.version if current else None
client.set_registered_model_alias(MODEL, ALIAS, VERSION)
print(f"@{ALIAS}: v{previous or '(none)'} → v{VERSION}")

spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {cfg.catalog.name}.monitoring.promotions (
        promoted_at TIMESTAMP, model_name STRING, alias STRING,
        from_version STRING, to_version STRING, gate_status STRING,
        promoted_by STRING)
""")
me = spark.sql("SELECT current_user() AS u").first()["u"]
(spark.createDataFrame([(MODEL, ALIAS, str(previous or ""), str(VERSION), gate, me)],
    "model_name STRING, alias STRING, from_version STRING, to_version STRING, "
    "gate_status STRING, promoted_by STRING")
 .withColumn("promoted_at", F.current_timestamp())
 .write.mode("append").saveAsTable(f"{cfg.catalog.name}.monitoring.promotions"))

print(f"recorded: {me}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Next
# MAGIC
# MAGIC Moving the alias does not update a running endpoint by itself. Run
# MAGIC notebook **07** — or the whole pipeline — so Model Serving picks up the
# MAGIC new version.
# MAGIC
# MAGIC To undo, run this notebook again with the previous version number. That
# MAGIC is the whole rollback procedure: seconds, no rebuild, no redeploy.

# COMMAND ----------
display(spark.sql(f"""
    SELECT promoted_at, model_name, alias, from_version, to_version, promoted_by
    FROM {cfg.catalog.name}.monitoring.promotions
    ORDER BY promoted_at DESC LIMIT 10
"""))
