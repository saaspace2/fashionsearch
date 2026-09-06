"""
Unit tests for the promotion gate.

This is the most important test file in the repo. The gate decides whether a
model reaches users, and its failure mode is silent approval — a bug here does
not raise, it just says PASS.

The scenario that matters most is `test_blocks_rare_slice_regression`: a model
that improves overall while collapsing on a rare category. An aggregate gate
approves it. This one must not.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))

from fashionsearch.promotion import evaluate_gate

GATE = {
    "max_recall20_drop_any_slice": 0.01,
    "max_recall20_drop_protected": 0.005,
    "min_queries_per_slice": 25,
    "protected_slices": [["category", "bag"], ["category", "hat"]],
}


def row(dim, value, n=100, recall=0.80, ndcg=0.75):
    return {"slice_dim": dim, "slice_value": value, "n_queries": n,
            "recall_at_20": recall, "ndcg_at_20": ndcg}


def baseline(**over):
    rows = [row("overall", "all"), row("category", "top"),
            row("category", "bag"), row("category", "hat")]
    for r in rows:
        key = f"{r['slice_dim']}:{r['slice_value']}"
        if key in over:
            r.update(over[key])
    return rows


def test_identical_model_passes():
    report = evaluate_gate(baseline(), baseline(), GATE)
    assert report.passed, report.render()


def test_clear_improvement_passes():
    better = baseline()
    for r in better:
        r["recall_at_20"] += 0.03
        r["ndcg_at_20"] += 0.03
    assert evaluate_gate(better, baseline(), GATE).passed


def test_blocks_overall_regression():
    worse = baseline()
    for r in worse:
        if r["slice_dim"] == "overall":
            r["ndcg_at_20"] = 0.70
    report = evaluate_gate(worse, baseline(), GATE)
    assert not report.passed
    assert "overall_ndcg20_not_worse" in report.failures


def test_blocks_rare_slice_regression():
    """
    The scenario this whole file exists for.

    Overall NDCG improves by 2 points. Bag Recall@20 collapses by 8. An
    aggregate gate ships this and bag search is broken in production.
    """
    candidate = baseline(**{
        "overall:all": {"ndcg_at_20": 0.77},      # better overall
        "category:bag": {"recall_at_20": 0.72},   # 8 points worse
    })
    report = evaluate_gate(candidate, baseline(), GATE)
    assert not report.passed, "a rare-slice collapse must block promotion"
    assert "protected_slices_no_regression" in report.failures


def test_tiny_slices_do_not_trigger_false_alarms():
    """A 3-query slice is noise; it must not block on statistical wobble."""
    candidate = baseline()
    candidate.append(row("category", "scarf", n=3, recall=0.20))
    champion = baseline()
    champion.append(row("category", "scarf", n=3, recall=0.90))
    report = evaluate_gate(candidate, champion, GATE)
    assert "recall20_no_slice_regression" not in report.failures


def test_missing_protected_slice_is_not_a_free_pass():
    """Silence is not a pass. No hat queries means we cannot claim hats are fine."""
    candidate = [r for r in baseline() if r["slice_value"] != "hat"]
    report = evaluate_gate(candidate, baseline(), GATE)
    assert not report.passed
    assert "protected_slice_coverage" in report.failures


def test_report_renders_without_error():
    assert "VERDICT" in evaluate_gate(baseline(), baseline(), GATE).render()
