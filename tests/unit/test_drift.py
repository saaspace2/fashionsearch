"""
Unit tests for drift detection.

The failure mode to guard against here is a drift metric that reports "stable"
no matter what. It never crashes, so nothing tells you it is broken — you just
stop hearing about problems, which feels exactly like not having any.
"""

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))

from fashionsearch.drift import (
    centroid_shift, jensen_shannon_distance, population_stability_index,
    psi_verdict, summarise,
)


class TestPSI:
    def test_identical_distributions_score_zero(self):
        d = {"top": 50, "bag": 30, "hat": 20}
        assert population_stability_index(d, d) < 1e-9

    def test_proportions_not_counts(self):
        """Twice as much traffic, same mix, is not drift."""
        small = {"top": 50, "bag": 50}
        large = {"top": 500, "bag": 500}
        assert population_stability_index(small, large) < 1e-9

    def test_small_shift_stays_under_the_noise_floor(self):
        before = {"top": 50, "bag": 30, "hat": 20}
        after = {"top": 52, "bag": 29, "hat": 19}
        assert population_stability_index(before, after) < 0.10

    def test_large_shift_is_flagged(self):
        before = {"top": 80, "bag": 15, "hat": 5}
        after = {"top": 20, "bag": 30, "hat": 50}
        assert population_stability_index(before, after) > 0.25

    def test_new_category_does_not_divide_by_zero(self):
        """A category that never appeared before is the case you most want
        reported, so it must not crash."""
        before = {"top": 100}
        after = {"top": 50, "scarf": 50}
        psi = population_stability_index(before, after)
        assert math.isfinite(psi) and psi > 0.25

    def test_disappearing_category_is_also_drift(self):
        psi = population_stability_index({"top": 50, "hat": 50}, {"top": 100})
        assert math.isfinite(psi) and psi > 0.25

    def test_empty_inputs_do_not_crash(self):
        assert population_stability_index({}, {}) == 0.0


class TestVerdict:
    def test_bands(self):
        assert psi_verdict(0.02) == "stable"
        assert psi_verdict(0.15) == "moderate shift"
        assert psi_verdict(0.40) == "significant shift"


class TestJensenShannon:
    def test_identical_is_zero(self):
        h = [1, 5, 20, 40, 20, 5, 1]
        assert jensen_shannon_distance(h, h) < 1e-9

    def test_disjoint_is_near_one(self):
        assert jensen_shannon_distance([10, 10, 0, 0], [0, 0, 10, 10]) > 0.9

    def test_symmetric(self):
        a, b = [10, 5, 1], [1, 5, 10]
        assert abs(jensen_shannon_distance(a, b)
                   - jensen_shannon_distance(b, a)) < 1e-9

    def test_confidence_dropping_is_detected(self):
        """Detector confidence sliding downward is the lens-degradation signal."""
        healthy = [0, 0, 2, 8, 30, 60]     # mostly high confidence
        degraded = [20, 40, 25, 10, 4, 1]  # mostly low
        assert jensen_shannon_distance(healthy, degraded) > 0.20


class TestCentroidShift:
    def test_same_embeddings_no_shift(self):
        e = [[1.0, 0.0], [0.0, 1.0]]
        assert centroid_shift(e, e) < 1e-6

    def test_rotated_embeddings_shift(self):
        assert centroid_shift([[1.0, 0.0]], [[0.0, 1.0]]) > 0.9

    def test_empty_is_nan_not_zero(self):
        # Zero would read as "no drift", which is a lie about a measurement
        # that was never taken.
        assert math.isnan(centroid_shift([], [[1.0, 0.0]]))


class TestSummarise:
    def test_stable_snapshots_produce_stable_verdicts(self):
        snap = {"category_counts": {"top": 50, "bag": 50},
                "confidence_hist": [1, 5, 20, 40],
                "fallback_rate": 0.05, "mean_confidence": 0.72}
        rows = summarise(snap, snap)
        assert rows and all(r["verdict"] == "stable" for r in rows)

    def test_missing_signals_are_skipped_not_faked(self):
        rows = summarise({}, {"fallback_rate": 0.05})
        assert rows == []
