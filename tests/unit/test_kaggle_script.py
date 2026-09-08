"""
Tests for the Kaggle script.

This file exists because of a specific bug that cost several failed runs.

`run_on_kaggle.py` is executed two ways: as a script from a terminal, and as a
cell inside a Kaggle notebook. A Kaggle notebook is a Jupyter kernel, and
sys.argv there is the kernel's own launch flags:

    ['.../ipykernel_launcher.py', '-f', '.../kernel-abc123.json']

argparse.parse_args() rejected those, printed usage and called sys.exit(2) — on
the first line of main(), before any work. The kernel died in seconds with
nothing in the log but ERROR, which looks like almost anything.

Nothing in CI executed the notebook path, so nothing caught it. These tests do.
"""

import ast
import pathlib
import sys
import types

import pytest

SCRIPT = pathlib.Path(__file__).resolve().parents[2] / "kaggle" / "run_on_kaggle.py"


@pytest.fixture
def module():
    """Load the script without running main()."""
    src = SCRIPT.read_text()
    ns = {"__name__": "run_on_kaggle_undertest"}
    exec(compile(src, str(SCRIPT), "exec"), ns)
    return ns


@pytest.fixture
def pretend_notebook(monkeypatch):
    """Make in_notebook() return True, as it does on Kaggle."""
    fake = types.ModuleType("IPython")
    fake.get_ipython = lambda: object()
    monkeypatch.setitem(sys.modules, "IPython", fake)
    monkeypatch.setattr(
        sys, "argv",
        ["/opt/conda/lib/python3.11/site-packages/ipykernel_launcher.py",
         "-f", "/root/.local/share/jupyter/runtime/kernel-abc123.json"])


def test_script_is_valid_python():
    ast.parse(SCRIPT.read_text())


class TestArgumentParsing:
    def test_survives_a_jupyter_kernel_argv(self, module, pretend_notebook):
        """The regression this file was written for."""
        args = module["parse_args"]()
        assert args.catalog
        assert args.sample_size > 0

    def test_command_line_flags_still_work(self, module, monkeypatch):
        monkeypatch.setattr(sys, "argv",
                            ["run_on_kaggle.py", "--sample_size", "500"])
        assert module["parse_args"]().sample_size == 500

    def test_unknown_flags_are_ignored_not_fatal(self, module, monkeypatch):
        # A stray flag should not kill a run that is about to do 20 minutes of
        # GPU work.
        monkeypatch.setattr(sys, "argv",
                            ["run_on_kaggle.py", "--not-a-real-flag", "x"])
        assert module["parse_args"]().catalog


class TestCredentialsAreOptional:
    """
    Missing credentials are the NORMAL path: Kaggle holds none by design, and
    GitHub Actions does the Databricks work. Raising here would kill a run that
    had already finished its GPU work — the one outcome worth avoiding.
    """

    def test_get_secret_returns_none_rather_than_raising(self, module, monkeypatch):
        monkeypatch.delenv("DATABRICKS_HOST", raising=False)
        assert module["get_secret"]("DATABRICKS_HOST") is None

    def test_get_secret_reads_the_environment(self, module, monkeypatch):
        monkeypatch.setenv("DATABRICKS_HOST", "https://example.databricks.com")
        assert module["get_secret"]("DATABRICKS_HOST") == "https://example.databricks.com"

    def test_upload_declines_quietly_without_credentials(self, module, monkeypatch):
        monkeypatch.delenv("DATABRICKS_HOST", raising=False)
        monkeypatch.delenv("DATABRICKS_TOKEN", raising=False)
        assert module["push_to_databricks"]("/tmp/x", "fashion_dev", "x") is False


# Aborting is correct only when continuing would produce results that look
# fine and are wrong. Both of these qualify:
#
#   "image columns"  — the dataset schema is not what the code expects
#   "did not load"   — the encoder is at random init, so every embedding would
#                      be meaningless while looking perfectly normal
#
# Aborting over a missing credential or an odd argv does NOT qualify: those end
# a run that has already spent twenty minutes of GPU time, for something the
# design deliberately does without.
ALLOWED_ABORTS = ("image columns", "did not load")


def test_sys_exit_only_where_continuing_would_mislead():
    source = SCRIPT.read_text()
    tree = ast.parse(source)
    offenders = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Raise) and isinstance(node.exc, ast.Call):
            if getattr(node.exc.func, "id", "") != "SystemExit":
                continue
            text = ast.get_source_segment(source, node) or ""
            if not any(reason in text for reason in ALLOWED_ABORTS):
                offenders.append(text[:100])
    assert not offenders, (
        "SystemExit raised where continuing would have been fine:\n  "
        + "\n  ".join(offenders)
        + f"\n\nOnly these justify aborting: {ALLOWED_ABORTS}")


def test_encoder_load_is_verified():
    """
    The weight-load check must stay.

    transformers 5.0 renamed the Swin attention layers, and from_pretrained does
    not raise on a mismatch — it warns and leaves those layers at random
    initialisation. The result runs, returns unit-length vectors, and encodes
    nothing. One Kaggle run produced exactly that before this check existed.
    """
    source = SCRIPT.read_text()
    assert "random initialisation" in source or "random init" in source, \
        "build_encoder no longer verifies that the checkpoint actually loaded"
