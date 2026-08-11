"""
VLM Compression Benchmarking Toolkit.

A toolkit for evaluating the impact of compression on Visual Language Models.
Provides snapshot-based evaluation, persistence, comparison, and normalization
utilities. Dashboard visualization is provided via Plotly Dash (Phase 3).

Quick start::

    from benchmarker import snapshot, save_snapshot, load_snapshot

    baseline_snap = snapshot(
        model_name="LLaVA-1.5-7B",
        label="Baseline (FP16)",
        model_info={...},
        general_efficiency={...},
        metrics={...},
        is_baseline=True,
    )
    save_snapshot(baseline_snap, "snapshots/baseline.json")
"""

__version__ = "0.1.0"

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
]
