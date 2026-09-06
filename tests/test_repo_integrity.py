"""
Fast checks that run in CI before anything is deployed.

These exist to catch the mistakes that are cheap to make and expensive to find
in a half-deployed workspace: a stray tab in a YAML file, a typo that stops a
Python module importing, a job pointing at a script that was renamed.

They need no Databricks connection and finish in under a second.
"""

import ast
import pathlib

import pytest
import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent

PY_FILES = sorted(ROOT.glob("src/**/*.py"))
YAML_FILES = sorted(ROOT.glob("resources/*.yml")) + [ROOT / "databricks.yml"]


@pytest.mark.parametrize("path", PY_FILES, ids=lambda p: p.name)
def test_python_parses(path):
    """Every source file must be syntactically valid Python."""
    ast.parse(path.read_text(), filename=str(path))


@pytest.mark.parametrize("path", YAML_FILES, ids=lambda p: p.name)
def test_yaml_parses(path):
    """Every bundle file must be valid YAML."""
    assert yaml.safe_load(path.read_text()) is not None


def test_every_task_points_at_a_real_file():
    """
    A job task referencing a script that does not exist deploys cleanly and then
    fails at runtime, often hours later. Catch it here instead.
    """
    missing = []
    for yml in ROOT.glob("resources/*.yml"):
        doc = yaml.safe_load(yml.read_text()) or {}
        for job in (doc.get("resources", {}).get("jobs", {}) or {}).values():
            for task in job.get("tasks", []) or []:
                spec = task.get("spark_python_task") or task.get("python_wheel_task")
                if not spec or "python_file" not in spec:
                    continue
                # Paths in resources/*.yml are relative to that file.
                target = (yml.parent / spec["python_file"]).resolve()
                if not target.exists():
                    missing.append(f"{yml.name}:{task['task_key']} -> {spec['python_file']}")
    assert not missing, "tasks point at files that do not exist:\n  " + "\n  ".join(missing)


def test_no_hardcoded_secrets():
    """A crude but effective guard against committing a token."""
    suspicious = ("dapi", "AKIA", "-----BEGIN")
    hits = []
    for path in PY_FILES + list(ROOT.glob("resources/*.yml")):
        text = path.read_text()
        for marker in suspicious:
            if marker in text:
                hits.append(f"{path.name} contains '{marker}'")
    assert not hits, "possible secret committed:\n  " + "\n  ".join(hits)


def test_every_task_has_compute():
    """
    Every task must specify where it runs.

    Omitting compute only works for notebook tasks. For a spark_python_task,
    python_wheel_task or dbt task on serverless, environment_key is REQUIRED.
    Getting this wrong fails at 'bundle validate' with:

        Task "<name>" requires a cluster or an environment to run.

    which is cheap to catch here and annoying to catch by hand.
    """
    compute_fields = {"job_cluster_key", "environment_key",
                      "existing_cluster_id", "new_cluster"}
    problems = []
    for yml in ROOT.glob("resources/*.yml"):
        doc = yaml.safe_load(yml.read_text()) or {}
        for jname, job in (doc.get("resources", {}).get("jobs", {}) or {}).items():
            declared = {e["environment_key"]
                        for e in (job.get("environments") or [])}
            declared |= {c["job_cluster_key"]
                         for c in (job.get("job_clusters") or [])}
            for task in job.get("tasks", []) or []:
                if not (compute_fields & set(task)):
                    problems.append(f"{jname}.{task['task_key']}: no compute specified")
                    continue
                key = task.get("environment_key") or task.get("job_cluster_key")
                if key and key not in declared:
                    problems.append(
                        f"{jname}.{task['task_key']}: references '{key}' "
                        f"which the job never declares")
    assert not problems, "compute configuration errors:\n  " + "\n  ".join(problems)
