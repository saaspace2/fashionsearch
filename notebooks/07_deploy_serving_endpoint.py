# Databricks notebook source
# MAGIC %md
# MAGIC # 07 — Serving endpoint
# MAGIC
# MAGIC Puts the encoder behind a real-time endpoint so an application can send an
# MAGIC image and get an embedding back.
# MAGIC
# MAGIC Note what is served and what is not: only the **query** side runs online.
# MAGIC Catalogue embeddings are computed in batch by notebook 05 and stored, so
# MAGIC the endpoint handles one image per request, not two million.
# MAGIC
# MAGIC Not available on Free Edition — skip to notebook 08 if you are there.

# COMMAND ----------
import sys, pathlib
sys.path.insert(0, str(pathlib.Path.cwd().parent / "src"))
from fashionsearch.config import load_config

cfg = load_config()

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput, ServedEntityInput, TrafficConfig, Route,
)
from mlflow.tracking import MlflowClient
import mlflow

mlflow.set_registry_uri("databricks-uc")
w = WorkspaceClient()
client = MlflowClient()

# COMMAND ----------
# MAGIC %md
# MAGIC ## Requirements check
# MAGIC
# MAGIC This notebook needs two things Free Edition does not provide: models
# MAGIC registered in Unity Catalog, and Databricks Model Serving. It is not part
# MAGIC of `fashion_pipeline` for that reason.
# MAGIC
# MAGIC If you are on Free Edition, skip it. The pipeline is complete without it:
# MAGIC notebook 05 embeds the catalogue in batch and notebook 09 runs searches
# MAGIC interactively, which is what you need to see the system working. A serving
# MAGIC endpoint only matters once a real application is calling it.

# COMMAND ----------
from fashionsearch import registry

try:
    from mlflow.tracking import MlflowClient as _C
    mlflow.set_registry_uri("databricks-uc")
    _C().get_registered_model(cfg.registry.encoder_model)
except Exception as exc:
    raise SystemExit(
        f"{cfg.registry.encoder_model} is not in Unity Catalog: {exc}\n\n"
        f"Notebook 04 fell back to the pointer table, which means UC model "
        f"registration is unavailable on this workspace. Model Serving requires "
        f"UC, so this notebook cannot run here.\n\n"
        f"Use notebook 09 instead — it runs real searches and shows the results.")

# COMMAND ----------
dbutils.widgets.dropdown("alias", "candidate", ["candidate", "shadow", "production"])
ALIAS = dbutils.widgets.get("alias")

version = client.get_model_version_by_alias(cfg.registry.encoder_model, ALIAS)
print(f"serving {cfg.registry.encoder_model} v{version.version} (@{ALIAS})")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Shadow mode
# MAGIC
# MAGIC When deploying a `@candidate`, send it **0% of traffic** and keep the
# MAGIC champion at 100%. Both are loaded, both can be called explicitly, but
# MAGIC users see only the champion. That gives you a comparison on real inputs at
# MAGIC zero risk, which offline metrics cannot provide.

# COMMAND ----------
entities = [ServedEntityInput(
    entity_name=cfg.registry.encoder_model,
    entity_version=version.version,
    name=f"encoder-v{version.version}",
    workload_size=cfg.serving.workload_size,
    scale_to_zero_enabled=cfg.serving.scale_to_zero,
)]
routes = [Route(served_model_name=f"encoder-v{version.version}",
                traffic_percentage=100 if ALIAS == "production" else 0)]

if ALIAS != "production":
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
        routes[0].traffic_percentage = 100     # nothing to shadow against

config = EndpointCoreConfigInput(served_entities=entities,
                                 traffic_config=TrafficConfig(routes=routes))

# COMMAND ----------
name = cfg.serving.endpoint_name
existing = [e.name for e in w.serving_endpoints.list()]

if name in existing:
    w.serving_endpoints.update_config_and_wait(name=name, **config.as_dict())
    print(f"updated {name}")
else:
    w.serving_endpoints.create_and_wait(name=name, config=config)
    print(f"created {name}")

# COMMAND ----------
# MAGIC %md ## Smoke test — and check the latency budget

# COMMAND ----------
import base64, io, time
from PIL import Image

buf = io.BytesIO()
Image.new("RGB", (224, 224), (200, 120, 90)).save(buf, format="JPEG")
payload = {"dataframe_records": [{"image": base64.b64encode(buf.getvalue()).decode()}]}

latencies = []
for _ in range(10):
    t0 = time.time()
    resp = w.serving_endpoints.query(name=name, dataframe_records=payload["dataframe_records"])
    latencies.append((time.time() - t0) * 1000)

p95 = sorted(latencies)[int(0.95 * len(latencies)) - 1]
print(f"embedding dim: {len(resp.predictions[0])}")
print(f"p95 latency: {p95:.0f} ms (budget {cfg.serving.max_p95_latency_ms} ms)")

assert p95 <= cfg.serving.max_p95_latency_ms, (
    f"p95 {p95:.0f} ms exceeds the budget. A more accurate model that is too "
    f"slow is not deployable — treat this as a gate failure, not a warning.")
