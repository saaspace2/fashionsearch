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
