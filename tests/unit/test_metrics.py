"""
Unit tests for the retrieval metrics.

These cover the cases that actually cause silent wrongness in production
evaluation code: empty results, no correct answer, and the boundary between
"found at rank k" and "found at rank k+1".
"""

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))

from fashionsearch.metrics import ndcg_at_k, recall_at_k, reciprocal_rank


class TestRecallAtK:
    def test_all_relevant_in_top_k(self):
        assert recall_at_k(["a", "b", "c"], ["a", "b"], k=3) == 1.0

    def test_half_found(self):
        assert recall_at_k(["a", "x", "y"], ["a", "b"], k=3) == 0.5

    def test_k_truncates(self):
        # 'b' sits at rank 3, so Recall@2 must not count it.
        assert recall_at_k(["a", "x", "b"], ["a", "b"], k=2) == 0.5

    def test_no_relevant_items_is_nan_not_zero(self):
        # Zero would silently drag slice averages down. NaN is excluded from
        # means instead, which is the honest behaviour.
        assert math.isnan(recall_at_k(["a"], [], k=5))

    def test_empty_results(self):
        assert recall_at_k([], ["a"], k=5) == 0.0


class TestReciprocalRank:
    def test_first_position(self):
        assert reciprocal_rank(["a", "b"], ["a"]) == 1.0

    def test_third_position(self):
        assert reciprocal_rank(["x", "y", "a"], ["a"]) == 1 / 3

    def test_absent(self):
        assert reciprocal_rank(["x", "y"], ["a"]) == 0.0


class TestNDCG:
    def test_perfect_ranking_is_one(self):
        rel = {"a": 1.0, "b": 1.0}
        assert ndcg_at_k(["a", "b", "x"], rel, k=3) == 1.0

    def test_order_matters(self):
        rel = {"a": 1.0}
        good = ndcg_at_k(["a", "x", "y"], rel, k=3)
        bad = ndcg_at_k(["x", "y", "a"], rel, k=3)
        assert good > bad

    def test_graded_relevance_prefers_exact(self):
        # An exact match at rank 1 should beat a partial match at rank 1.
        rel = {"exact": 1.0, "similar": 0.5}
        assert (ndcg_at_k(["exact", "similar"], rel, k=2)
                > ndcg_at_k(["similar", "exact"], rel, k=2))

    def test_no_relevant_is_nan(self):
        assert math.isnan(ndcg_at_k(["a"], {}, k=5))
