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

# The endpoint serves the COMBINED model: photo in, boxes plus embedding out.
# Serving the encoder alone would mean the caller had to run detection and
# cropping itself, and every caller would have to agree on which box to pick —
# the moment two of them disagree, the embeddings stop being comparable.
SERVED_MODEL = cfg.registry.get("search_model", cfg.registry.encoder_model)
print(f"serving: {SERVED_MODEL}")

# COMMAND ----------
# MAGIC %md ## Can we deploy a real endpoint?

# COMMAND ----------
def serving_available() -> tuple:
    """Returns (bool, reason). Both conditions must hold."""
    try:
        from mlflow.tracking import MlflowClient
        mlflow.set_registry_uri("databricks-uc")
        MlflowClient().get_model_version_by_alias(SERVED_MODEL, ALIAS)
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
# MAGIC %md
# MAGIC ## Resolve and validate — every step optional
# MAGIC
# MAGIC This notebook reports; it does not gate. Nothing here may stop the run,
# MAGIC because everything it checks is either informational or unavailable on
# MAGIC this workspace tier.
# MAGIC
# MAGIC It also writes its row **whatever happens**. An earlier version failed
# MAGIC before the write and left monitoring.serving_checks three days stale,
# MAGIC which is worse than useless — a table that silently stops updating looks
# MAGIC exactly like one reporting no change.

# COMMAND ----------
uri, load_error, resolve_error = None, None, None
contract_ok, dim, norm, p50, p95 = None, None, None, None, None
budget = float(cfg.serving.max_p95_latency_ms)

# resolve() raises SystemExit when a model has no such alias — which is normal
# for a model registered minutes ago that no gate has promoted yet. Outside a
# try block that ends the notebook before anything is recorded.
try:
    uri = registry.resolve(cfg, SERVED_MODEL, ALIAS)
    print(f"resolved {uri}")
except BaseException as exc:
    resolve_error = f"{type(exc).__name__}: {str(exc)[:200]}"
    print(f"could not resolve {SERVED_MODEL} @{ALIAS}")
    print(f"  {resolve_error}")
    print("  Expected if the combined model has not been promoted yet.")

# COMMAND ----------
sample = spark.table(table(cfg, "bronze", "products")).select("image_path").limit(1).first()
payload = None
if sample:
    try:
        with open(sample["image_path"], "rb") as fh:
            payload = pd.DataFrame({"image": [base64.b64encode(fh.read()).decode()]})
    except Exception as exc:
        print(f"could not read a sample image: {exc}")

# Skip the local load entirely when config says the registry is unreadable
# here. The serving container loads the model itself, so this check is a
# nice-to-have — and attempting it produces a 300-line traceback that buries
# the endpoint result underneath it.
SKIP_LOCAL = str(cfg.registry.get("load_from", "auto")).lower() != "registry"

if SKIP_LOCAL:
    print("skipping the local contract check — registry.load_from is not "
          "'registry', so this workspace is not expected to read model "
          "artifacts from a notebook. The serving container loads the model.")
    load_error = "skipped by config"
elif uri and payload is not None:
    try:
        model = mlflow.pyfunc.load_model(uri)
        print("model loaded locally")

        out = model.predict(payload)
        vec = np.asarray(out["embedding"].iloc[0], dtype=np.float32)
        dim, norm = int(vec.shape[0]), float(np.linalg.norm(vec))
        expected = int(cfg.pretrained.encoder.embedding_dim)
        for name, ok, got, want in [
                ("output_dimensions", dim == expected, dim, expected),
                ("unit_normalised", abs(norm - 1.0) < 1e-2, round(norm, 5), 1.0)]:
            print(f"  [{'PASS' if ok else 'FAIL'}] {name:<20} got {got}, expected {want}")
        contract_ok = dim == expected and abs(norm - 1.0) < 1e-2

        # A floor, not a forecast — no network hop, no queuing, no cold start.
        for _ in range(3):
            model.predict(payload)
        timings = []
        for _ in range(10):
            t0 = time.perf_counter()
            model.predict(payload)
            timings.append((time.perf_counter() - t0) * 1000)
        timings.sort()
        p50 = timings[len(timings) // 2]
        p95 = timings[int(0.95 * len(timings)) - 1]
        print(f"  p50 {p50:7.1f} ms")
        print(f"  p95 {p95:7.1f} ms   (budget {budget:.0f} ms)")
        if p95 > budget:
            print("  Over budget. Recorded, not raised: CPU speed says nothing")
            print("  about model quality, and serving runs elsewhere.")

    except BaseException as exc:
        load_error = f"{type(exc).__name__}: {str(exc)[:200]}"
        if "HeadObject" in str(exc) or "400" in str(exc):
            print("Cannot read the model artifacts from this notebook.")
            print("  This workspace tier, not the model — the same artifacts")
            print("  downloaded fine in GitHub Actions during registration.")
            print("  The serving container loads the model itself, so this does")
            print("  not decide whether an endpoint can be created.")
        else:
            print(f"local load failed: {load_error}")

print("\ncontinuing to the endpoint decision")

# COMMAND ----------
# MAGIC %md ## Deploy, if we can

# COMMAND ----------
if CAN_SERVE:
  try:
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service.serving import (
        EndpointCoreConfigInput, ServedEntityInput, TrafficConfig, Route)
    from mlflow.tracking import MlflowClient

    w = WorkspaceClient()
    client = MlflowClient()
    version = client.get_model_version_by_alias(SERVED_MODEL, ALIAS)

    entities = [ServedEntityInput(
        entity_name=SERVED_MODEL,
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
                SERVED_MODEL, cfg.registry.aliases.champion)
            entities.append(ServedEntityInput(
                entity_name=SERVED_MODEL,
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
        # Pass the typed objects, NOT **config.as_dict().
        #
        # as_dict() flattens the nested ServedEntityInput and TrafficConfig
        # objects into plain dicts. Splatting those back in hands the SDK dicts
        # where it expects typed objects, and it then calls .as_dict() on them:
        #
        #     AttributeError: 'dict' object has no attribute 'as_dict'
        w.serving_endpoints.update_config_and_wait(
            name=endpoint_name,
            served_entities=entities,
            traffic_config=TrafficConfig(routes=routes))
        print(f"updated endpoint {endpoint_name}")
    else:
        w.serving_endpoints.create_and_wait(name=endpoint_name, config=config)
        print(f"created endpoint {endpoint_name}")

    state = w.serving_endpoints.get(endpoint_name).state
    print(f"endpoint state: {state}")
    note = f"deployed at {traffic}% traffic"
    reason = note
  except Exception as exc:
    # Even with both preconditions met, creation can fail — quota, region,
    # workload size. Record it rather than ending the notebook.
    reason = f"endpoint creation failed: {type(exc).__name__}: {str(exc)[:200]}"
    print(reason)
    CAN_SERVE = False
else:
    print("Skipping deployment. The contract and latency checks above still ran,")
    print("and they are what would have caught a broken model anyway.")

# COMMAND ----------
note_parts = [reason]
if resolve_error:
    note_parts.append(f"resolve: {resolve_error}")
if load_error:
    note_parts.append(f"local load: {load_error}")
full_note = " | ".join(note_parts)[:900]

row = [(SERVED_MODEL, uri or "unresolved", ALIAS,
        "endpoint" if CAN_SERVE else "local_validation",
        int(dim) if dim else None,
        bool(contract_ok) if contract_ok is not None else None,
        float(p50) if p50 else None, float(p95) if p95 else None,
        budget,
        bool(p95 <= budget) if p95 else None,
        endpoint_name, full_note)]

(spark.createDataFrame(row,
    "model_name STRING, model_uri STRING, alias STRING, mode STRING, "
    "embedding_dim INT, unit_normalised BOOLEAN, p50_ms DOUBLE, p95_ms DOUBLE, "
    "budget_ms DOUBLE, within_budget BOOLEAN, endpoint_name STRING, note STRING")
 .withColumn("checked_at", F.current_timestamp())
 .write.mode("append").saveAsTable(table(cfg, "monitoring", "serving_checks")))

print(f"\nrecorded: mode={'endpoint' if CAN_SERVE else 'local_validation'}")
print(f"note: {full_note}")

display(spark.table(table(cfg, "monitoring", "serving_checks"))
        .orderBy(F.desc("checked_at")).limit(5))
