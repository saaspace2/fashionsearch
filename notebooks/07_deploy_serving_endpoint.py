# Databricks notebook source
# MAGIC %md
# MAGIC # 07 — Serving
# MAGIC
# MAGIC Two modes, chosen automatically.
# MAGIC
# MAGIC **Real endpoint.** If the models are in Unity Catalog and Model Serving is
# MAGIC available, this deploys a real-time endpoint. A candidate goes in at 0%
# MAGIC traffic alongside the champion — shadow mode — so you get a comparison on
# MAGIC live inputs at no risk.
# MAGIC
# MAGIC **Local validation.** On Free Edition neither is available. Rather than
# MAGIC exit, this validates the *serving contract* on the driver: the model loads
# MAGIC from its registered URI, accepts the documented input, returns the
# MAGIC documented output, and answers within the latency budget.
# MAGIC
# MAGIC That second mode is worth more than it sounds. Most serving failures are
# MAGIC not the endpoint — they are a model that cannot load in a fresh process, or
# MAGIC returns a different shape than its signature promises. Those are caught
# MAGIC here either way.

# COMMAND ----------
# MAGIC %md
# MAGIC Dependencies come from the job's `environments:` block.

# COMMAND ----------
# MAGIC %md
# MAGIC ### Where is this running?
# MAGIC
# MAGIC Inside the job, dependencies come from the `torch` environment in
# MAGIC `resources/jobs_pipeline.yml` and there is nothing to install.
# MAGIC
# MAGIC Opened by hand in the workspace, that environment does not apply and
# MAGIC loading the model fails with a bare `ModuleNotFoundError: No module named
# MAGIC 'torch'`, which does not explain itself. Hence this check.

# COMMAND ----------
_missing = []
for _mod in ("torch", "torchvision", "transformers"):
    try:
        __import__(_mod)
    except ImportError:
        _missing.append(_mod)

if _missing:
    raise SystemExit(
        f"Missing: {', '.join(_missing)}.\n\n"
        f"You are running this notebook interactively, so the job's 'torch' "
        f"environment does not apply.\n\n"
        f"Either run it through the job:\n"
        f"    databricks bundle run fashion_pipeline -t dev\n"
        f"or re-run the 07_serving_check task from the Databricks Jobs UI.\n\n"
        f"To keep working here instead, add this as the first cell and re-run:\n"
        f"    %pip install -q torch torchvision transformers timm\n"
        f"    %restart_python")

print("dependencies present")

# COMMAND ----------
import sys, pathlib
sys.path.insert(0, str(pathlib.Path.cwd().parent / "src"))
from fashionsearch.config import load_config, table
from fashionsearch import registry

cfg = load_config()

import base64, io, json, time
import numpy as np
import pandas as pd
import mlflow
from PIL import Image
from pyspark.sql import functions as F

dbutils.widgets.dropdown("alias", "production", ["candidate", "shadow", "production"])
ALIAS = dbutils.widgets.get("alias")

# COMMAND ----------
# MAGIC %md ## Can we deploy a real endpoint?

# COMMAND ----------
def serving_available() -> tuple:
    """Returns (bool, reason). Both conditions must hold."""
    try:
        from mlflow.tracking import MlflowClient
        mlflow.set_registry_uri("databricks-uc")
        MlflowClient().get_model_version_by_alias(cfg.registry.encoder_model, ALIAS)
    except Exception as exc:
        return False, f"model not in Unity Catalog ({type(exc).__name__})"

    try:
        from databricks.sdk import WorkspaceClient
        list(WorkspaceClient().serving_endpoints.list())
        return True, "Unity Catalog and Model Serving both available"
    except Exception as exc:
        return False, f"Model Serving unavailable ({type(exc).__name__})"


CAN_SERVE, reason = serving_available()
print(f"real endpoint: {'yes' if CAN_SERVE else 'no'} — {reason}")
if not CAN_SERVE:
    print("Falling back to local contract validation. This is expected on Free Edition.")

# COMMAND ----------
# MAGIC %md ## Validate the serving contract
# MAGIC
# MAGIC Runs in both modes. A model that fails here would fail behind an endpoint too.

# COMMAND ----------
uri = registry.resolve(cfg, cfg.registry.encoder_model, ALIAS)
model = mlflow.pyfunc.load_model(uri)
print(f"loaded {uri}")

# Use a real catalogue image, not a blank square — a blank image can mask
# preprocessing bugs that only show up on actual content.
sample = spark.table(table(cfg, "bronze", "products")).select("image_path").limit(1).first()
with open(sample["image_path"], "rb") as fh:
    payload = pd.DataFrame({"image": [base64.b64encode(fh.read()).decode()]})

out = model.predict(payload)
vec = np.asarray(out, dtype=np.float32)[0]
dim = vec.shape[0]
norm = float(np.linalg.norm(vec))

expected = int(cfg.pretrained.encoder.embedding_dim)
checks = [
    ("output_dimensions", dim == expected, dim, expected),
    ("unit_normalised", abs(norm - 1.0) < 1e-3, round(norm, 5), 1.0),
]
for name, ok, got, want in checks:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:<20} got {got}, expected {want}")

assert all(c[1] for c in checks), "the model does not honour its own signature"

# COMMAND ----------
# MAGIC %md
# MAGIC ## Latency
# MAGIC
# MAGIC Measured on the driver, one image per call, which is what a serving request
# MAGIC looks like. This is **not** the same as endpoint latency — no network hop,
# MAGIC no queuing, no cold start — so treat it as a floor rather than a forecast.
# MAGIC A model too slow here will certainly be too slow behind an endpoint.

# COMMAND ----------
N_WARMUP, N_RUNS = 3, 30

for _ in range(N_WARMUP):
    model.predict(payload)

timings = []
for _ in range(N_RUNS):
    t0 = time.perf_counter()
    model.predict(payload)
    timings.append((time.perf_counter() - t0) * 1000)

timings.sort()
p50 = timings[len(timings) // 2]
p95 = timings[int(0.95 * len(timings)) - 1]
budget = float(cfg.serving.max_p95_latency_ms)

print(f"  p50 {p50:7.1f} ms")
print(f"  p95 {p95:7.1f} ms   (budget {budget:.0f} ms)")
within = p95 <= budget
print(f"  [{'PASS' if within else 'FAIL'}] within budget")

if not within:
    print("\n  On serverless CPU this is expected — the encoder is a Swin transformer")
    print("  and CPU inference is slow. A GPU endpoint changes the picture entirely.")
    print("  Recorded rather than raised, because CPU latency is not evidence about")
    print("  the model's quality.")

# COMMAND ----------
# MAGIC %md ## Record the result

# COMMAND ----------
spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {table(cfg, "monitoring", "serving_checks")} (
        checked_at TIMESTAMP, model_name STRING, model_uri STRING, alias STRING,
        mode STRING, embedding_dim INT, unit_normalised BOOLEAN,
        p50_ms DOUBLE, p95_ms DOUBLE, budget_ms DOUBLE, within_budget BOOLEAN,
        endpoint_name STRING, note STRING)
""")

note = reason
endpoint_name = None

# COMMAND ----------
# MAGIC %md ## Deploy, if we can

# COMMAND ----------
if CAN_SERVE:
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service.serving import (
        EndpointCoreConfigInput, ServedEntityInput, TrafficConfig, Route)
    from mlflow.tracking import MlflowClient

    w = WorkspaceClient()
    client = MlflowClient()
    version = client.get_model_version_by_alias(cfg.registry.encoder_model, ALIAS)

    entities = [ServedEntityInput(
        entity_name=cfg.registry.encoder_model,
        entity_version=version.version,
        name=f"encoder-v{version.version}",
        workload_size=cfg.serving.workload_size,
        scale_to_zero_enabled=cfg.serving.scale_to_zero)]

    # A candidate enters at 0% traffic. Both models are loaded and callable, but
    # users only ever see the champion. That is a free comparison on real inputs.
    traffic = 100 if ALIAS == "production" else 0
    routes = [Route(served_model_name=f"encoder-v{version.version}",
                    traffic_percentage=traffic)]

    if traffic == 0:
        try:
            champ = client.get_model_version_by_alias(
                cfg.registry.encoder_model, cfg.registry.aliases.champion)
            entities.append(ServedEntityInput(
                entity_name=cfg.registry.encoder_model,
                entity_version=champ.version,
                name=f"encoder-v{champ.version}",
                workload_size=cfg.serving.workload_size,
                scale_to_zero_enabled=cfg.serving.scale_to_zero))
            routes.append(Route(served_model_name=f"encoder-v{champ.version}",
                                traffic_percentage=100))
        except Exception:
            routes[0].traffic_percentage = 100   # nothing to shadow against

    config = EndpointCoreConfigInput(
        served_entities=entities, traffic_config=TrafficConfig(routes=routes))
    endpoint_name = cfg.serving.endpoint_name

    if endpoint_name in [e.name for e in w.serving_endpoints.list()]:
        w.serving_endpoints.update_config_and_wait(name=endpoint_name, **config.as_dict())
        print(f"updated endpoint {endpoint_name}")
    else:
        w.serving_endpoints.create_and_wait(name=endpoint_name, config=config)
        print(f"created endpoint {endpoint_name}")
    note = f"deployed at {traffic}% traffic"
else:
    print("Skipping deployment. The contract and latency checks above still ran,")
    print("and they are what would have caught a broken model anyway.")

# COMMAND ----------
row = [(cfg.registry.encoder_model, uri, ALIAS,
        "endpoint" if CAN_SERVE else "local_validation",
        int(dim), bool(abs(norm - 1.0) < 1e-3),
        float(p50), float(p95), budget, bool(within),
        endpoint_name, note)]

(spark.createDataFrame(row,
    "model_name STRING, model_uri STRING, alias STRING, mode STRING, "
    "embedding_dim INT, unit_normalised BOOLEAN, p50_ms DOUBLE, p95_ms DOUBLE, "
    "budget_ms DOUBLE, within_budget BOOLEAN, endpoint_name STRING, note STRING")
 .withColumn("checked_at", F.current_timestamp())
 .write.mode("append").saveAsTable(table(cfg, "monitoring", "serving_checks")))

display(spark.table(table(cfg, "monitoring", "serving_checks"))
        .orderBy(F.desc("checked_at")).limit(5))
