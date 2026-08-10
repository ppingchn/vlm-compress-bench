"""
Snapshot I/O module for VLM compression benchmarking.

This module handles the full lifecycle of evaluation snapshots:
- Creating snapshots from model evaluation results
- Saving snapshots to JSON files (persistence layer)
- Loading snapshots from JSON files
- Comparing compressed snapshots against a baseline
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .schema import validate_snapshot

logger = logging.getLogger(__name__)

# Metrics where a *lower* score means *better* performance.
# Used by compare_snapshots to invert the delta direction.
_INVERTED_METRICS = {"chair_s"}

# Keys inside each metric group that are NOT numeric scores
# (they are metadata and should be skipped during delta computation).
_NON_SCORE_KEYS = {"dataset", "num_samples"}


# ---------------------------------------------------------------------------
# Snapshot creation
# ---------------------------------------------------------------------------

def snapshot(
    model_name: str,
    label: str,
    model_info: dict,
    general_efficiency: dict,
    metrics: dict,
    evaluation_config: Optional[dict] = None,
    notes: Optional[str] = None,
    is_baseline: bool = False,
) -> Dict[str, Any]:
    """Create a complete snapshot dictionary from evaluation results.

    A UUID and ISO-8601 timestamp are generated automatically. The snapshot
    is validated against the schema before being returned.

    Args:
        model_name: Name of the evaluated model (e.g. ``"LLaVA-1.5-7B"``).
        label: Human-readable label (e.g. ``"Baseline (FP16)"``).
        model_info: Dictionary with model metadata. Must include all fields
            required by the schema **except** ``model_name`` and
            ``is_baseline``, which are injected automatically.
        general_efficiency: Efficiency metrics dict (model_size_mb, etc.).
        metrics: Nested dict of the six metric groups
            (vqa, captioning, visual_reasoning, ocr_document,
            hallucination, spatial_awareness).
        evaluation_config: Optional evaluation configuration dict.
            Defaults to ``{"device": "cuda", "batch_size": 1,
            "max_new_tokens": 128, "seed": None}``.
        notes: Optional freeform notes about the snapshot.
        is_baseline: Whether this snapshot represents the baseline model.

    Returns:
        A validated snapshot dictionary ready for saving or dashboard use.

    Raises:
        jsonschema.ValidationError: If the assembled snapshot is invalid.
    """
    # Inject convenience fields into model_info
    model_info = {**model_info, "model_name": model_name, "is_baseline": is_baseline}

    if evaluation_config is None:
        evaluation_config = {
            "device": "cuda",
            "batch_size": 1,
            "max_new_tokens": 128,
            "seed": None,
        }

    snap: Dict[str, Any] = {
        "schema_version": "1.0",
        "snapshot_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_info": model_info,
        "general_efficiency": general_efficiency,
        "metrics": metrics,
        "evaluation_config": evaluation_config,
        "label": label,
        "notes": notes,
    }

    validate_snapshot(snap)
    return snap


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_snapshot(snapshot_data: dict, filepath: str) -> str:
    """Save a snapshot to a JSON file.

    Parent directories are created automatically if they don't exist.
    The snapshot is validated against the schema before writing.

    Args:
        snapshot_data: A validated snapshot dictionary.
        filepath: Destination file path (relative or absolute).

    Returns:
        The absolute file path of the saved snapshot.

    Raises:
        jsonschema.ValidationError: If *snapshot_data* is invalid.
    """
    validate_snapshot(snapshot_data)

    path = Path(filepath).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(snapshot_data, f, indent=2, ensure_ascii=False)

    logger.info("Snapshot saved to %s", path)
    return str(path)


def load_snapshot(filepath: str) -> dict:
    """Load a single snapshot from a JSON file.

    The loaded data is validated against the schema.

    Args:
        filepath: Path to the snapshot JSON file.

    Returns:
        The validated snapshot dictionary.

    Raises:
        FileNotFoundError: If *filepath* does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
        jsonschema.ValidationError: If the JSON does not match the schema.
    """
    path = Path(filepath).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Snapshot file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    validate_snapshot(data)
    logger.info("Snapshot loaded from %s", path)
    return data


def load_snapshots(directory: str) -> List[dict]:
    """Load all JSON snapshot files from a directory.

    Invalid files are skipped with a warning instead of raising.

    Args:
        directory: Path to a directory containing ``*.json`` files.

    Returns:
        A list of validated snapshot dictionaries.
    """
    dir_path = Path(directory).resolve()
    if not dir_path.is_dir():
        raise NotADirectoryError(f"Not a directory: {dir_path}")

    snapshots: List[dict] = []
    for json_file in sorted(dir_path.glob("*.json")):
        try:
            snap = load_snapshot(str(json_file))
            snapshots.append(snap)
        except Exception as exc:
            logger.warning("Skipping %s: %s", json_file.name, exc)

    logger.info("Loaded %d snapshot(s) from %s", len(snapshots), dir_path)
    return snapshots


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def _compute_metric_deltas(
    baseline_group: dict,
    compressed_group: dict,
    group_name: str,
) -> Dict[str, Optional[float]]:
    """Compute per-metric deltas between two metric group dicts.

    For higher-is-better metrics: ``delta = compressed - baseline``
    For lower-is-better metrics (CHAIR-S): ``delta = baseline - compressed``

    In both cases, a **positive** delta means the compressed model is better.
    """
    deltas: Dict[str, Optional[float]] = {}

    for key in baseline_group:
        if key in _NON_SCORE_KEYS:
            continue

        b_val = baseline_group.get(key)
        c_val = compressed_group.get(key)

        if b_val is None or c_val is None:
            deltas[key] = None
            continue

        if key in _INVERTED_METRICS:
            # Lower is better → positive delta when compressed < baseline
            deltas[key] = b_val - c_val
        else:
            deltas[key] = c_val - b_val

    return deltas


def compare_snapshots(baseline: dict, compressed: dict) -> Dict[str, Any]:
    """Compare a compressed snapshot against a baseline.

    Args:
        baseline: The baseline (reference) snapshot dictionary.
        compressed: The compressed model snapshot dictionary.

    Returns:
        A comparison dictionary containing:

        - ``baseline_info`` / ``compressed_info``: model metadata for both.
        - ``general_efficiency``: side-by-side efficiency comparison with
          deltas and the compression ratio.
        - ``metric_deltas``: per-group, per-metric deltas. Positive values
          always indicate improvement regardless of metric direction.
    """
    # --- Model info ---
    result: Dict[str, Any] = {
        "baseline_info": baseline["model_info"],
        "compressed_info": compressed["model_info"],
    }

    # --- General efficiency ---
    b_eff = baseline["general_efficiency"]
    c_eff = compressed["general_efficiency"]
    eff_comparison: Dict[str, Any] = {}
    for key in b_eff:
        eff_comparison[key] = {
            "baseline": b_eff[key],
            "compressed": c_eff[key],
            "delta": c_eff[key] - b_eff[key],
        }
    result["general_efficiency"] = eff_comparison

    # --- Metric deltas ---
    b_metrics = baseline["metrics"]
    c_metrics = compressed["metrics"]
    metric_deltas: Dict[str, Dict[str, Optional[float]]] = {}

    for group_name in b_metrics:
        metric_deltas[group_name] = _compute_metric_deltas(
            b_metrics[group_name],
            c_metrics[group_name],
            group_name,
        )

    result["metric_deltas"] = metric_deltas
    return result
