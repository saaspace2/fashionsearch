"""
Model registration that degrades gracefully when Unity Catalog is unavailable.

WHY THIS EXISTS
---------------
Registering a model to Unity Catalog copies its artifacts into UC managed
storage. Databricks requires *dedicated* (single-user) access mode for that, and
Databricks Free Edition is serverless-only, so the copy is refused:

    S3UploadFailedError: AccessDenied ... explicit deny in a resource-based policy

The historical fallback — the workspace model registry — was disabled in April
2024 for new accounts whose default catalog lives in Unity Catalog. So on Free
Edition both registries are closed.

WHAT WE DO INSTEAD
------------------
The models themselves are fine: notebook 03 logged them to the MLflow experiment
and those artifacts uploaded without trouble. Only the *copy into UC* fails. So
when UC registration is unavailable we keep a small Delta table of pointers —
name, version, alias, run URI — and resolve models through that.

You lose UC governance: cross-workspace discovery, UC lineage, UC permissions.
You keep everything the pipeline actually depends on: versions, aliases, an
audit trail of which model was champion when, and a single place to look it up.

On a workspace with dedicated compute, UC is used and this table is never
touched. Nothing in the notebooks has to know which mode is active.
"""

from __future__ import annotations

from typing import Optional

POINTER_TABLE = "model_pointers"


def _table(cfg) -> str:
    return f"{cfg.catalog.name}.ml.{POINTER_TABLE}"


def _spark():
    from pyspark.sql import SparkSession
    return SparkSession.builder.getOrCreate()


def _ensure_table(cfg) -> None:
    _spark().sql(f"""
        CREATE TABLE IF NOT EXISTS {_table(cfg)} (
            model_name  STRING  COMMENT 'Logical name, e.g. fashion_dev.ml.fashion_encoder',
            version     BIGINT,
            uri         STRING  COMMENT 'runs:/<run_id>/<artifact> — loadable directly',
            alias       STRING  COMMENT 'candidate | shadow | production | NULL',
            description STRING,
            updated_at  TIMESTAMP
        ) USING DELTA
        COMMENT 'Fallback model registry, used when Unity Catalog model registration is unavailable.'
    """)


# --------------------------------------------------------------------- register

def register(cfg, model_uri: str, name: str, description: str = "") -> dict:
    """
    Register a logged model. Returns {"mode", "version", "uri"}.

    Tries Unity Catalog first. Falls back to the pointer table on any failure,
    printing why — a silent fallback would hide a fixable permissions problem.
    """
    import mlflow
    from mlflow.tracking import MlflowClient

    try:
        mlflow.set_registry_uri("databricks-uc")
        mv = mlflow.register_model(model_uri=model_uri, name=name)
        client = MlflowClient()
        if description:
            client.update_registered_model(name=name, description=description)
        client.set_model_version_tag(name, mv.version, "source", "huggingface")
        client.set_model_version_tag(name, mv.version, "gate_status", "not_evaluated")
        print(f"  [UC] {name} -> version {mv.version}")
        return {"mode": "uc", "version": int(mv.version),
                "uri": f"models:/{name}/{mv.version}"}

    except Exception as exc:
        print(f"  Unity Catalog registration failed for {name}:\n    {exc}\n")
        print("  Falling back to the pointer table. This is expected on Databricks")
        print("  Free Edition, where UC model registration needs dedicated compute.")
        return _register_fallback(cfg, model_uri, name, description)


def _register_fallback(cfg, model_uri: str, name: str, description: str) -> dict:
    from pyspark.sql import functions as F

    _ensure_table(cfg)
    spark = _spark()

    current = spark.sql(
        f"SELECT max(version) AS v FROM {_table(cfg)} WHERE model_name = '{name}'"
    ).first()["v"]
    version = int(current or 0) + 1

    row = spark.createDataFrame(
        [(name, version, model_uri, None, description)],
        "model_name STRING, version BIGINT, uri STRING, alias STRING, description STRING",
    ).withColumn("updated_at", F.current_timestamp())
    row.write.mode("append").saveAsTable(_table(cfg))

    print(f"  [pointer table] {name} -> version {version}")
    return {"mode": "fallback", "version": version, "uri": model_uri}


# ------------------------------------------------------------------- aliases

def set_alias(cfg, name: str, alias: str, version: int) -> None:
    """Point an alias at a version, in whichever registry is in use."""
    try:
        import mlflow
        from mlflow.tracking import MlflowClient

        mlflow.set_registry_uri("databricks-uc")
        MlflowClient().set_registered_model_alias(name, alias, str(version))
        print(f"  [UC] {name} @{alias} = v{version}")
        return
    except Exception:
        pass

    _ensure_table(cfg)
    # One row holds each alias, so moving it is a clear before/after in the
    # table history — which is the audit trail the gate depends on.
    _spark().sql(f"""
        UPDATE {_table(cfg)} SET alias = NULL
        WHERE model_name = '{name}' AND alias = '{alias}'
    """)
    _spark().sql(f"""
        UPDATE {_table(cfg)} SET alias = '{alias}', updated_at = current_timestamp()
        WHERE model_name = '{name}' AND version = {version}
    """)
    print(f"  [pointer table] {name} @{alias} = v{version}")


def set_tag(cfg, name: str, version: int, key: str, value: str) -> None:
    """Best-effort tag. Silently ignored in fallback mode — tags are not load-bearing."""
    try:
        import mlflow
        from mlflow.tracking import MlflowClient

        mlflow.set_registry_uri("databricks-uc")
        MlflowClient().set_model_version_tag(name, str(version), key, value)
    except Exception:
        pass


# ------------------------------------------------------------------- resolve

def resolve(cfg, name: str, alias: str) -> str:
    """
    Return a URI that mlflow.pyfunc.load_model() can actually load.

    UC first, then the pointer table. Raises with a useful message if neither
    has the alias, because 'model not found' three notebooks later is much
    harder to diagnose than failing here.
    """
    try:
        import mlflow
        from mlflow.tracking import MlflowClient

        mlflow.set_registry_uri("databricks-uc")
        MlflowClient().get_model_version_by_alias(name, alias)
        return f"models:/{name}@{alias}"
    except Exception:
        pass

    _ensure_table(cfg)
    row = _spark().sql(f"""
        SELECT uri, version FROM {_table(cfg)}
        WHERE model_name = '{name}' AND alias = '{alias}'
        ORDER BY version DESC LIMIT 1
    """).first()

    if row is None:
        raise SystemExit(
            f"No model found for {name} @{alias}.\n"
            f"Neither Unity Catalog nor {_table(cfg)} has it. Run notebooks 03 and "
            f"04 first — 03 logs the models, 04 registers them and sets the alias.")

    print(f"  resolved {name}@{alias} -> v{row['version']} ({row['uri']})")
    return row["uri"]


def latest_version(cfg, name: str) -> int:
    """Highest version number, from whichever registry is in use."""
    try:
        import mlflow
        from mlflow.tracking import MlflowClient

        mlflow.set_registry_uri("databricks-uc")
        versions = MlflowClient().get_registered_model(name).latest_versions
        if versions:
            return int(versions[0].version)
    except Exception:
        pass

    _ensure_table(cfg)
    row = _spark().sql(
        f"SELECT max(version) AS v FROM {_table(cfg)} WHERE model_name = '{name}'"
    ).first()
    if row is None or row["v"] is None:
        raise SystemExit(f"No versions registered for {name}. Run notebook 04 first.")
    return int(row["v"])
