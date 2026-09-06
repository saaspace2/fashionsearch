"""
Small shims for MLflow API differences.

The Databricks serverless environment does not necessarily ship the same MLflow
version you develop against, and MLflow 3 renamed things. Rather than pin a
version and fight the runtime, detect what is actually there.
"""

from __future__ import annotations

import inspect
from typing import Any


def log_model(flavor, artifact_name: str, **kwargs) -> str:
    """
    Log a model and return its URI, on either MLflow 2 or MLflow 3.

    MLflow 2:  log_model(artifact_path="encoder", ...)
    MLflow 3:  log_model(name="encoder", ...)

    Passing the wrong one raises
        TypeError: log_model() got an unexpected keyword argument 'name'

    We inspect the signature first and fall back to try/except for versions that
    hide the parameter behind **kwargs.

    The returned URI is built from the active run rather than read off the result
    object, because `ModelInfo.model_uri` did not exist in earlier MLflow 2
    releases and this needs to work on whatever the runtime happens to have.
    """
    import mlflow

    try:
        params = inspect.signature(flavor.log_model).parameters
        key = "name" if "name" in params else "artifact_path"
    except (TypeError, ValueError):
        key = "artifact_path"

    try:
        flavor.log_model(**{key: artifact_name}, **kwargs)
    except TypeError as exc:
        if "unexpected keyword argument" not in str(exc):
            raise
        other = "artifact_path" if key == "name" else "name"
        flavor.log_model(**{other: artifact_name}, **kwargs)

    run = mlflow.active_run()
    if run is None:
        raise RuntimeError("log_model must be called inside an active MLflow run")
    return f"runs:/{run.info.run_id}/{artifact_name}"
