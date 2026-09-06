"""
Load config.yaml once and expose it as a dotted-access object.

Every notebook starts with `cfg = load_config()`. Nothing anywhere else in the
codebase should contain a catalog name, a model name or a threshold — if you
find one, it belongs here instead. That is what makes the same code run against
dev and prod without edits.
"""

from __future__ import annotations

import pathlib
from typing import Any


class Cfg(dict):
    """A dict that also supports attribute access, recursively."""

    def __getattr__(self, key: str) -> Any:
        try:
            value = self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc
        return Cfg(value) if isinstance(value, dict) else value


def _find_config(start: pathlib.Path | None = None) -> pathlib.Path:
    """
    Walk upward looking for config.yaml.

    Notebooks run from different working directories depending on whether they
    were launched by a job, from the workspace UI, or locally, so a relative
    path is not reliable.
    """
    here = (start or pathlib.Path(__file__)).resolve()
    for parent in [here, *here.parents]:
        candidate = parent / "config.yaml"
        if candidate.exists():
            return candidate
    raise FileNotFoundError("config.yaml not found in any parent directory")


def load_config(path: str | None = None, overrides: dict | None = None) -> Cfg:
    import yaml

    p = pathlib.Path(path) if path else _find_config()
    cfg = Cfg(yaml.safe_load(p.read_text()))

    # Allow a job parameter to swap catalogs without editing the file, e.g.
    # overrides={"catalog": {"name": "fashion_prod"}}
    if overrides:
        _deep_update(cfg, overrides)
    return cfg


def _deep_update(base: dict, extra: dict) -> None:
    for k, v in extra.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v


def table(cfg: Cfg, schema: str, name: str) -> str:
    """Fully-qualified table name: catalog.schema.name"""
    return f"{cfg.catalog.name}.{schema}.{name}"


def ensure_experiment(name: str) -> str:
    """
    Point MLflow at an experiment, creating the workspace folders above it first.

    MLflow will happily create an experiment, but it will NOT create the
    directories in its path. So `/Shared/fashionsearch/import` fails with

        RestException: NOT_FOUND: Parent directory does not exist: /Shared/fashionsearch

    on any workspace where nobody has made that folder yet — which is every fresh
    workspace. Creating the parent explicitly fixes it.

    If /Shared turns out not to be writable (permissions vary between workspace
    tiers), this falls back to a flat name under the caller's own home folder,
    which always exists. Returns the path actually used.
    """
    import mlflow

    try:
        from databricks.sdk import WorkspaceClient

        w = WorkspaceClient()
        parent = name.rsplit("/", 1)[0]
        try:
            w.workspace.mkdirs(parent)
        except Exception as exc:
            me = w.current_user.me().user_name
            flat = name.strip("/").replace("/", "_")
            name = f"/Users/{me}/{flat}"
            print(f"  could not create {parent} ({exc}); using {name} instead")
    except Exception as exc:
        print(f"  workspace client unavailable ({exc}); trying {name} as-is")

    mlflow.set_experiment(name)
    print(f"  MLflow experiment: {name}")
    return name
