"""
The promotion gate.

Pure logic, no I/O — you hand it two DataFrames of per-slice metrics and it
returns verdicts. That makes it unit testable, which matters more here than
anywhere else in the codebase: this is the code that decides whether a model
reaches users, and a bug in it fails silently by approving things.

Design principle, stated once because everything below follows from it:

    An aggregate metric hides its worst behaviour in whichever slice is rarest,
    and the rarest slice is usually the hardest one.

A model can gain 1.5 points of overall NDCG while losing 8 points of Recall@20
on bags. Bags are ~6% of queries, so the average barely moves. An aggregate gate
approves that model and bag search is quietly broken in production.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field


@dataclass
class Verdict:
    name: str
    passed: bool
    observed: float
    threshold: float
    detail: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class GateReport:
    verdicts: list = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(v.passed for v in self.verdicts)

    @property
    def failures(self) -> list:
        return [v.name for v in self.verdicts if not v.passed]

    def as_dict(self) -> dict:
        return {"passed": self.passed,
                "verdicts": [v.as_dict() for v in self.verdicts]}

    def render(self) -> str:
        lines = ["=" * 78]
        for v in self.verdicts:
            mark = "PASS" if v.passed else "FAIL"
            lines.append(f"  [{mark}] {v.name:<34} {v.observed:>9.4f} "
                         f"vs {v.threshold:<9.4f} {v.detail}")
        lines.append("=" * 78)
        lines.append("VERDICT: " + ("PROMOTE to @candidate" if self.passed
                                    else f"BLOCKED — {', '.join(self.failures)}"))
        return "\n".join(lines)


def _lookup(rows, dim, value):
    for r in rows:
        if r["slice_dim"] == dim and r["slice_value"] == value:
            return r
    return None


def evaluate_gate(candidate: list, champion: list, gate_cfg: dict) -> GateReport:
    """
    candidate / champion: list of dicts, one per slice, each with at least
      slice_dim, slice_value, n_queries, recall_at_20, ndcg_at_20
    """
    report = GateReport()
    protected = [tuple(p) for p in gate_cfg["protected_slices"]]
    min_n = gate_cfg["min_queries_per_slice"]

    # 1 — overall primary metric must not fall below the champion.
    c_overall = _lookup(candidate, "overall", "all")
    m_overall = _lookup(champion, "overall", "all")
    c_val = float(c_overall["ndcg_at_20"]) if c_overall else 0.0
    m_val = float(m_overall["ndcg_at_20"]) if m_overall else 0.0
    report.verdicts.append(Verdict(
        "overall_ndcg20_not_worse",
        c_val >= m_val - 1e-9, c_val, m_val,
        "candidate vs champion"))

    # 2 — no Recall@20 regression on ANY slice with enough queries to measure.
    worst, worst_name = 0.0, ""
    for row in candidate:
        if row["slice_dim"] == "overall" or row["n_queries"] < min_n:
            continue
        match = _lookup(champion, row["slice_dim"], row["slice_value"])
        if not match:
            continue
        drop = float(match["recall_at_20"]) - float(row["recall_at_20"])
        if drop > worst:
            worst, worst_name = drop, f"{row['slice_dim']}={row['slice_value']}"
    report.verdicts.append(Verdict(
        "recall20_no_slice_regression",
        worst <= gate_cfg["max_recall20_drop_any_slice"],
        worst, gate_cfg["max_recall20_drop_any_slice"],
        f"worst: {worst_name}" if worst_name else "no regression"))

    # 3 — protected slices, tighter tolerance. Rarity is the reason these need
    #     protecting, not a reason to let them slide.
    p_worst, p_name = 0.0, ""
    for dim, value in protected:
        c = _lookup(candidate, dim, value)
        m = _lookup(champion, dim, value)
        if not c or not m:
            continue
        drop = float(m["recall_at_20"]) - float(c["recall_at_20"])
        if drop > p_worst:
            p_worst, p_name = drop, f"{dim}={value}"
    report.verdicts.append(Verdict(
        "protected_slices_no_regression",
        p_worst <= gate_cfg["max_recall20_drop_protected"],
        p_worst, gate_cfg["max_recall20_drop_protected"],
        f"worst: {p_name}" if p_name else "no regression"))

    # 4 — coverage. Silence is not a pass: a model must not clear a protected
    #     slice merely because there were nine queries in it.
    missing = [f"{d}={v}" for d, v in protected
               if not (_lookup(candidate, d, v) or {}).get("n_queries", 0) >= min_n]
    report.verdicts.append(Verdict(
        "protected_slice_coverage",
        len(missing) == 0, len(protected) - len(missing), len(protected),
        f"too few queries: {', '.join(missing)}" if missing else "all covered"))

    return report
