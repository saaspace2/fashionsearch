"""
Drift detection.

The question this answers: has the world changed underneath a model that has
not changed itself?

That happens constantly in fashion. New season, new styles, a new social
platform whose crop format differs, a supplier added, a category discontinued.
The model file is byte-identical and its real-world quality has moved.

WHAT WE MEASURE, AND WHY NOT ACCURACY
-------------------------------------
The tempting answer is "watch accuracy". You usually cannot: knowing whether a
search result was right needs a human, and you get that for a tiny fraction of
traffic at best.

So drift detection watches the INPUTS and the model's own behaviour, both of
which are free and available immediately:

  * category mix        — are we seeing the same kinds of garment as before?
  * detector confidence — a lens smudge, a new photo style, a harder catalogue
  * fallback rate       — how often detection finds nothing at all
  * embedding geometry  — are the vectors landing where they used to?

None of these prove quality dropped. All of them are early warnings that
something moved, which is what you want from monitoring. Quality is confirmed
afterwards, by re-running the frozen eval set.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np

# Population Stability Index bands. These are the conventional cut-offs from
# credit risk, where PSI has been used for decades, and they transfer well.
PSI_NO_SHIFT = 0.10
PSI_MODERATE = 0.25


def population_stability_index(expected: Mapping[str, float],
                               actual: Mapping[str, float],
                               epsilon: float = 1e-6) -> float:
    """
    How much a categorical distribution has moved.

        PSI = sum over categories of (a - e) * ln(a / e)

    where a and e are proportions. Zero means identical. Under 0.10 is noise;
    0.10 to 0.25 is a moderate shift worth a look; above 0.25 is a real change.

    Symmetric in a useful way: a category appearing that never appeared before
    contributes as much as one disappearing. Both matter.

    epsilon guards the logarithm — a category present now but absent before
    would otherwise divide by zero, and that is precisely the case you most
    want reported rather than crashed on.
    """
    keys = set(expected) | set(actual)
    if not keys:
        return 0.0

    e_total = sum(expected.values()) or 1.0
    a_total = sum(actual.values()) or 1.0

    psi = 0.0
    for k in keys:
        e = max(expected.get(k, 0.0) / e_total, epsilon)
        a = max(actual.get(k, 0.0) / a_total, epsilon)
        psi += (a - e) * math.log(a / e)
    return float(psi)


def psi_verdict(psi: float) -> str:
    if psi < PSI_NO_SHIFT:
        return "stable"
    if psi < PSI_MODERATE:
        return "moderate shift"
    return "significant shift"


def jensen_shannon_distance(p: Sequence[float], q: Sequence[float]) -> float:
    """
    Distance between two numeric distributions, 0 (identical) to 1 (disjoint).

    Used for continuous things — confidence scores, similarity scores — where
    PSI's category buckets do not apply. Unlike KL divergence it is symmetric
    and always finite, which matters when one side has values the other never
    produced.
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = p / (p.sum() or 1.0)
    q = q / (q.sum() or 1.0)
    m = 0.5 * (p + q)

    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / np.maximum(b[mask], 1e-12))))

    return float(math.sqrt(max(0.5 * kl(p, m) + 0.5 * kl(q, m), 0.0)))


def histogram(values: Sequence[float], bins: int = 20,
              lo: float = 0.0, hi: float = 1.0) -> list:
    """Fixed-range histogram, so two runs are always directly comparable."""
    counts, _ = np.histogram(np.asarray(values, dtype=np.float64),
                             bins=bins, range=(lo, hi))
    return counts.tolist()


def centroid_shift(before: np.ndarray, after: np.ndarray) -> float:
    """
    Cosine distance between the average embedding of two sets of images.

    Embeddings are unit length, so their mean points in the "average direction"
    of the collection. If that direction moves, the visual character of what is
    being embedded has moved — new product photography, a different mix of
    garments, a change in how images are cropped.

    0.0 means no movement. Above roughly 0.05 is worth investigating.
    """
    if len(before) == 0 or len(after) == 0:
        return float("nan")
    b = np.asarray(before, dtype=np.float32).mean(axis=0)
    a = np.asarray(after, dtype=np.float32).mean(axis=0)
    bn, an = np.linalg.norm(b), np.linalg.norm(a)
    if bn == 0 or an == 0:
        return float("nan")
    return float(1.0 - np.dot(b / bn, a / an))


def summarise(previous: dict, current: dict) -> list:
    """
    Compare two monitoring snapshots and return one row per signal.

    Each row: signal, metric, value, verdict, hint. Designed to be written
    straight to a Delta table and rendered on a dashboard.
    """
    rows = []

    if previous.get("category_counts") and current.get("category_counts"):
        psi = population_stability_index(previous["category_counts"],
                                         current["category_counts"])
        rows.append({
            "signal": "category_mix", "metric": "PSI", "value": round(psi, 4),
            "verdict": psi_verdict(psi),
            "hint": "the mix of garment types being searched has changed",
        })

    if previous.get("confidence_hist") and current.get("confidence_hist"):
        js = jensen_shannon_distance(previous["confidence_hist"],
                                     current["confidence_hist"])
        rows.append({
            "signal": "detector_confidence", "metric": "JS distance",
            "value": round(js, 4),
            "verdict": "stable" if js < 0.10 else
                       "moderate shift" if js < 0.20 else "significant shift",
            "hint": "how sure the detector is; falls when photos get harder "
                    "or a lens degrades",
        })

    for key, label, warn, hint in [
        ("fallback_rate", "detector_fallback_rate", 0.10,
         "share of queries where no garment was detected at all"),
        ("mean_confidence", "mean_confidence", 0.05,
         "average detector confidence across all queries"),
    ]:
        if key in previous and key in current:
            delta = float(current[key]) - float(previous[key])
            rows.append({
                "signal": label, "metric": "absolute change",
                "value": round(delta, 4),
                "verdict": "stable" if abs(delta) < warn else "significant shift",
                "hint": hint,
            })

    return rows
