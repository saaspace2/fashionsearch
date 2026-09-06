"""
Retrieval metrics: Recall@k, MRR, NDCG@k, computed per slice.

Kept as plain functions with no Spark or MLflow dependency so they can be unit
tested in milliseconds. tests/unit/test_metrics.py exercises the edge cases that
actually bite: empty result lists, queries with no correct answer, and ties.
"""

from __future__ import annotations

import math
from typing import Iterable, Mapping, Sequence


def recall_at_k(ranked: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Fraction of the correct answers that appear in the top k."""
    relevant = set(relevant)
    if not relevant:
        return float("nan")
    return len(set(ranked[:k]) & relevant) / len(relevant)


def reciprocal_rank(ranked: Sequence[str], relevant: Iterable[str]) -> float:
    """1 / position of the first correct result. 0.0 if none appear at all."""
    relevant = set(relevant)
    for i, item in enumerate(ranked, start=1):
        if item in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: Sequence[str], relevance: Mapping[str, float], k: int) -> float:
    """
    Normalised Discounted Cumulative Gain.

    Each result contributes its relevance grade divided by log2 of its position,
    so rank 1 is worth much more than rank 10. Dividing by the best achievable
    score keeps the result between 0 and 1 and comparable across queries that
    have different numbers of correct answers.
    """
    dcg = sum(relevance.get(item, 0.0) / math.log2(i + 1)
              for i, item in enumerate(ranked[:k], start=1))
    ideal = sorted(relevance.values(), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 1) for i, g in enumerate(ideal, start=1))
    return dcg / idcg if idcg > 0 else float("nan")


def evaluate_query(ranked, relevant, relevance, k_values) -> dict:
    out = {f"recall_at_{k}": recall_at_k(ranked, relevant, k) for k in k_values}
    out.update({f"ndcg_at_{k}": ndcg_at_k(ranked, relevance, k) for k in k_values})
    out["mrr"] = reciprocal_rank(ranked, relevant)
    return out
