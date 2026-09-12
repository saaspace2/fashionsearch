"""
Everything the app needs from Databricks, in one place.

Three different services, deliberately kept behind one module so the UI never
has to know which is which:

    SQL warehouse     reading Delta tables — metrics, drift, the catalogue
    Serving endpoint  turning an uploaded photo into an embedding
    Volumes           fetching product images to display

Credentials come from Streamlit secrets when running as a Databricks App, and
from the environment otherwise, so the same code runs both ways.
"""

from __future__ import annotations

import base64
import os
from functools import lru_cache

import numpy as np
import pandas as pd


def _setting(name: str, default: str = "") -> str:
    """Streamlit secrets first, environment second."""
    try:
        import streamlit as st
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return os.environ.get(name, default)


CATALOG = _setting("CATALOG", "fashion_dev")
ENDPOINT = _setting("ENDPOINT", "fashion-encoder-endpoint")
WAREHOUSE_ID = _setting("WAREHOUSE_ID")


@lru_cache(maxsize=1)
def workspace():
    from databricks.sdk import WorkspaceClient

    host = _setting("DATABRICKS_HOST")
    token = _setting("DATABRICKS_TOKEN")
    if host and token:
        return WorkspaceClient(host=host.rstrip("/"), token=token)
    # Running as a Databricks App: credentials are injected.
    return WorkspaceClient()


# --------------------------------------------------------------------- SQL

def sql(query: str) -> pd.DataFrame:
    """
    Run a query and return a DataFrame.

    Uses the Statement Execution API rather than a SQL driver, so the app needs
    only the SDK — no ODBC, no extra binary to install on whatever machine this
    runs on.
    """
    from databricks.sdk.service.sql import StatementState

    if not WAREHOUSE_ID:
        raise RuntimeError(
            "WAREHOUSE_ID is not set. Find it under SQL Warehouses in Databricks "
            "— the ID is in the connection details, or in the URL after "
            "/warehouses/.")

    w = workspace()
    resp = w.statement_execution.execute_statement(
        warehouse_id=WAREHOUSE_ID, statement=query, wait_timeout="50s")

    while resp.status.state in (StatementState.PENDING, StatementState.RUNNING):
        resp = w.statement_execution.get_statement(resp.statement_id)

    if resp.status.state != StatementState.SUCCEEDED:
        raise RuntimeError(f"query failed: {resp.status.error}")

    if not resp.manifest or not resp.manifest.schema.columns:
        return pd.DataFrame()

    columns = [c.name for c in resp.manifest.schema.columns]
    rows = (resp.result.data_array or []) if resp.result else []
    return pd.DataFrame(rows, columns=columns)


# ---------------------------------------------------------------- endpoint

def detect_and_embed(image_bytes: bytes) -> list:
    """
    Send one image to the endpoint. Get back every garment it found.

    Returns a list of dicts, best first:
        {embedding, detected_category, detector_score, area_frac,
         box, used_whole_image}

    A list rather than one item because an outfit photo contains several things
    a user might be searching for. Returning only the largest would mean someone
    who photographed a full outfit could search for the jacket but never the
    trousers.

    Three response shapes are handled, because the endpoint may be running any
    of three model versions and the UI should not care:
      * multi-item combined model  — one row per garment
      * single-item combined model — one row per image
      * encoder only               — a bare vector, no detection
    """
    w = workspace()
    payload = [{"image": base64.b64encode(image_bytes).decode()}]
    response = w.serving_endpoints.query(name=ENDPOINT, dataframe_records=payload)
    predictions = response.predictions

    if not predictions:
        return []

    items = []
    for p in predictions:
        if isinstance(p, dict) and "embedding" in p:
            items.append({
                "embedding": np.asarray(p["embedding"], dtype=np.float32),
                "detected_category": p.get("detected_category"),
                "detector_score": p.get("detector_score"),
                "area_frac": p.get("area_frac"),
                "box": [p.get("x1"), p.get("y1"), p.get("x2"), p.get("y2")]
                       if p.get("x2") is not None else None,
                "used_whole_image": p.get("used_whole_image", False),
            })
        else:
            vector = list(p.values()) if isinstance(p, dict) else list(p)
            items.append({
                "embedding": np.asarray(vector, dtype=np.float32),
                "detected_category": None, "detector_score": None,
                "area_frac": None, "box": None, "used_whole_image": True,
            })
    return items


def embed(image_bytes: bytes) -> dict:
    """The first detected garment only. Kept for callers that want one."""
    items = detect_and_embed(image_bytes)
    return items[0] if items else {}


def endpoint_state() -> tuple:
    """(name, state) — or (name, None) when it does not exist."""
    try:
        e = workspace().serving_endpoints.get(ENDPOINT)
        return ENDPOINT, str(e.state.ready) if e.state else "unknown"
    except Exception:
        return ENDPOINT, None


# ----------------------------------------------------------------- volumes

def read_volume_file(path: str) -> bytes | None:
    try:
        response = workspace().files.download(path)
        return response.contents.read()
    except Exception:
        return None
