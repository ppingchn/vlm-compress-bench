"""
VLM Compression Benchmarking Toolkit.

A toolkit for evaluating the impact of compression on Visual Language Models.
Provides snapshot-based evaluation, persistence, comparison, normalization
utilities, and a full suite of metric group evaluators. Dashboard
visualization is provided via Plotly Dash (Phase 3).

Quick start::

    from benchmarker import full_evaluation, save_snapshot

    snap = full_evaluation(
        model_name="LLaVA-1.5-7B",
        model=model,
        processor=processor,
        label="Baseline (FP16)",
        is_baseline=True,
    )
    save_snapshot(snap, "snapshots/baseline.json")
"""

__version__ = "0.2.0"

# Snapshot I/O
from .snapshot import (  # noqa: E402
    snapshot,
    save_snapshot,
    load_snapshot,
    load_snapshots,
    compare_snapshots,
)

# Schema & validation
from .schema import (  # noqa: E402
    SNAPSHOT_SCHEMA,
    validate_snapshot,
)

# Normalization
from .normalization import (  # noqa: E402
    compute_retention,
    compute_group_retentions,
    compute_all_deltas,
)

# Evaluator (Phase 2)
from .evaluator import (  # noqa: E402
    evaluate_model,
    full_evaluation,
    ALL_GROUPS,
)

__all__ = [
    # Version
    "__version__",
    # Snapshot lifecycle
    "snapshot",
    "save_snapshot",
    "load_snapshot",
    "load_snapshots",
    "compare_snapshots",
    # Schema
    "SNAPSHOT_SCHEMA",
    "validate_snapshot",
    # Normalization
    "compute_retention",
    "compute_group_retentions",
    "compute_all_deltas",
    # Evaluator
    "evaluate_model",
    "full_evaluation",
    "ALL_GROUPS",
]
