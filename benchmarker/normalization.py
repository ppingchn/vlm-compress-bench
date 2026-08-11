"""
Normalization module for VLM compression benchmarking.

This module converts raw metric scores into comparable, direction-agnostic
values for dashboard visualization. The core concepts are:

**Retention**
    How much capability the compressed model retained relative to the
    baseline, expressed as a percentage.  100 % means identical performance;
    values below 100 % indicate degradation.

**Delta**
    The signed difference between compressed and baseline scores, oriented
    so that *positive always means improvement* regardless of whether the
    underlying metric is higher-is-better or lower-is-better.

**Normalized delta**
    A delta rescaled to [0, 1], where 0.5 represents no change from
    baseline, values above 0.5 indicate improvement, and values below 0.5
    indicate degradation.

The only lower-is-better metric in the suite is **CHAIR-S** (hallucination).
All other metrics are higher-is-better.
"""

from __future__ import annotations

from typing import Dict, Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Metric keys where a *lower* score is *better*.
INVERTED_METRICS: set[str] = {"chair_s"}

#: The single primary metric used for each group in radar-chart comparisons.
PRIMARY_METRICS: Dict[str, str] = {
    "vqa": "accuracy",
    "captioning": "cider",
    "visual_reasoning": "accuracy",
    "ocr_document": "anls",
    "hallucination": "chair_s",
    "spatial_awareness": "accuracy",
}

#: Keys inside each metric group that are metadata, not numeric scores.
_NON_SCORE_KEYS: set[str] = {"dataset", "num_samples"}


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def is_inverted(metric_name: str) -> bool:
    """Return ``True`` if *metric_name* is a lower-is-better metric."""
    return metric_name in INVERTED_METRICS


def compute_retention(
    baseline_score: float,
    compressed_score: float,
    metric_name: str,
) -> float:
    """Compute retention percentage for a single metric.

    For **higher-is-better** metrics::

        retention = (compressed / baseline) × 100

    For **lower-is-better** metrics (CHAIR-S)::

        retention = (baseline / compressed) × 100

    In both cases, 100 % means the compressed model matches the baseline
    exactly, and values above 100 % indicate the compressed model is
    *better* than baseline.

    Args:
        baseline_score: The baseline model's score for this metric.
        compressed_score: The compressed model's score for this metric.
        metric_name: Name of the metric (used to determine direction).

    Returns:
        Retention as a percentage, clamped to [0, 200].
    """
    if is_inverted(metric_name):
        # Lower is better → good if compressed > baseline in denominator
        if compressed_score == 0.0:
            # Compressed model produced zero hallucinations — perfect
            return 200.0 if baseline_score > 0 else 100.0
        if baseline_score < 0.0 or compressed_score < 0.0:
            return 0.0
        retention = (baseline_score / compressed_score) * 100.0
    else:
        # Higher is better
        if baseline_score == 0.0:
            return 100.0 if compressed_score == 0.0 else 200.0
        if baseline_score < 0.0 or compressed_score < 0.0:
            return 0.0
        retention = (compressed_score / baseline_score) * 100.0

    return max(0.0, min(200.0, retention))


def compute_delta(
    baseline_score: float,
    compressed_score: float,
    metric_name: str,
) -> float:
    """Compute the raw, direction-adjusted delta for a metric.

    The result is oriented so that **positive always means improvement**.

    For higher-is-better::

        delta = compressed − baseline

    For lower-is-better (CHAIR-S)::

        delta = baseline − compressed

    Args:
        baseline_score: Baseline value.
        compressed_score: Compressed value.
        metric_name: Metric name (used to determine direction).

    Returns:
        The signed delta (positive = improvement).
    """
    if is_inverted(metric_name):
        return baseline_score - compressed_score
    return compressed_score - baseline_score


def normalize_delta(
    delta: float,
    baseline_score: float,
    metric_name: str,  # noqa: ARG001 — reserved for future per-metric logic
) -> float:
    """Normalize a delta to the [0, 1] range.

    The baseline score is used as the reference scale:

    * **0.5** — no change from baseline
    * **> 0.5** — improvement
    * **< 0.5** — degradation

    The mapping is ``normalized = 0.5 + delta / (2 * |baseline|)``, clamped
    to [0, 1].

    Args:
        delta: Direction-adjusted delta (from :func:`compute_delta`).
        baseline_score: Absolute baseline score used as denominator.
        metric_name: Metric name (reserved for future per-metric scaling).

    Returns:
        Normalized value in [0, 1].
    """
    abs_base = abs(baseline_score)
    if abs_base == 0.0:
        # Cannot normalise against a zero baseline; treat as neutral.
        return 0.5

    normalized = 0.5 + delta / (2.0 * abs_base)
    return max(0.0, min(1.0, normalized))


# ---------------------------------------------------------------------------
# Batch / group-level helpers
# ---------------------------------------------------------------------------

def compute_group_retentions(
    baseline_metrics: dict,
    compressed_metrics: dict,
) -> Dict[str, float]:
    """Compute retention for the **primary metric** of each group.

    This is the data source for the dashboard radar chart.

    Args:
        baseline_metrics: The ``"metrics"`` dict from the baseline snapshot.
        compressed_metrics: The ``"metrics"`` dict from the compressed snapshot.

    Returns:
        A dict mapping group name → retention percentage, e.g.
        ``{"vqa": 95.2, "captioning": 88.7, …}``.
    """
    retentions: Dict[str, float] = {}

    for group_name, primary_key in PRIMARY_METRICS.items():
        b_group = baseline_metrics.get(group_name, {})
        c_group = compressed_metrics.get(group_name, {})

        b_val = b_group.get(primary_key)
        c_val = c_group.get(primary_key)

        if b_val is None or c_val is None:
            continue

        retentions[group_name] = compute_retention(b_val, c_val, primary_key)

    return retentions


def compute_all_deltas(
    baseline_metrics: dict,
    compressed_metrics: dict,
) -> Dict[str, Dict[str, float]]:
    """Compute deltas for **all** numeric metrics across all groups.

    Non-score keys (``dataset``, ``num_samples``) and ``None`` values are
    skipped automatically.

    Args:
        baseline_metrics: The ``"metrics"`` dict from the baseline snapshot.
        compressed_metrics: The ``"metrics"`` dict from the compressed snapshot.

    Returns:
        Nested dict ``{group_name: {metric_name: delta}}``.
        Positive deltas always indicate improvement.
    """
    all_deltas: Dict[str, Dict[str, float]] = {}

    for group_name in baseline_metrics:
        b_group = baseline_metrics[group_name]
        c_group = compressed_metrics.get(group_name, {})
        group_deltas: Dict[str, float] = {}

        for key, b_val in b_group.items():
            if key in _NON_SCORE_KEYS:
                continue

            c_val = c_group.get(key)
            if b_val is None or c_val is None:
                continue

            group_deltas[key] = compute_delta(b_val, c_val, key)

        if group_deltas:
            all_deltas[group_name] = group_deltas

    return all_deltas
