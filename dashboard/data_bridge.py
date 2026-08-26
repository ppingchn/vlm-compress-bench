"""Data bridge between benchmarker APIs and Dash dashboard.

Provides cached data access, DataFrame conversions, and Dash-friendly
data transformations.
"""
import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd

# Add project root to path so we can import benchmarker
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from benchmarker.snapshot import load_snapshot, load_snapshots, compare_snapshots
from benchmarker.normalization import (
    compute_group_retentions,
    compute_all_deltas,
    compute_retention,
    compute_delta,
    normalize_delta,
    PRIMARY_METRICS,
    INVERTED_METRICS,
)
from benchmarker.schema import validate_snapshot

# Default directories
BASELINES_DIR = _PROJECT_ROOT / 'baselines'
SNAPSHOTS_DIR = _PROJECT_ROOT / 'snapshots'

# Non-score keys to skip
_NON_SCORE_KEYS = {'dataset', 'num_samples'}


class SnapshotStore:
    """Manages loading and caching of snapshots for the dashboard."""

    def __init__(self):
        self._snapshots: Dict[str, dict] = {}  # snapshot_id -> snapshot
        self._baseline_id: Optional[str] = None

    def load_defaults(self) -> None:
        """Load all snapshots from baselines/ and snapshots/ directories."""
        for directory in [BASELINES_DIR, SNAPSHOTS_DIR]:
            if directory.is_dir():
                for snap in load_snapshots(str(directory)):
                    sid = snap['snapshot_id']
                    self._snapshots[sid] = snap
                    if snap['model_info'].get('is_baseline'):
                        self._baseline_id = sid

    def add_snapshot(self, snap: dict) -> str:
        """Add a snapshot to the store. Returns snapshot_id."""
        validate_snapshot(snap)
        sid = snap['snapshot_id']
        self._snapshots[sid] = snap
        if snap['model_info'].get('is_baseline'):
            self._baseline_id = sid
        return sid

    @property
    def baseline(self) -> Optional[dict]:
        if self._baseline_id:
            return self._snapshots.get(self._baseline_id)
        return None

    @property
    def all_snapshots(self) -> List[dict]:
        return list(self._snapshots.values())

    @property
    def compressed_snapshots(self) -> List[dict]:
        return [s for s in self._snapshots.values() if not s['model_info'].get('is_baseline')]

    def get_snapshot(self, snapshot_id: str) -> Optional[dict]:
        return self._snapshots.get(snapshot_id)

    def get_dropdown_options(self) -> List[dict]:
        """Return options for Dash dropdown components."""
        return [
            {'label': s['label'], 'value': s['snapshot_id']}
            for s in self._snapshots.values()
        ]

    def get_baseline_options(self) -> List[dict]:
        return [
            {'label': s['label'], 'value': s['snapshot_id']}
            for s in self._snapshots.values()
            if s['model_info'].get('is_baseline')
        ]

    def get_compressed_options(self) -> List[dict]:
        return [
            {'label': s['label'], 'value': s['snapshot_id']}
            for s in self._snapshots.values()
            if not s['model_info'].get('is_baseline')
        ]

    def compare(self, baseline_id: str, compressed_id: str) -> dict:
        baseline = self._snapshots[baseline_id]
        compressed = self._snapshots[compressed_id]
        return compare_snapshots(baseline, compressed)

    def get_retentions(self, baseline_id: str, compressed_id: str) -> Dict[str, float]:
        baseline = self._snapshots[baseline_id]
        compressed = self._snapshots[compressed_id]
        return compute_group_retentions(baseline['metrics'], compressed['metrics'])

    def get_all_deltas(self, baseline_id: str, compressed_id: str) -> Dict[str, Dict[str, float]]:
        baseline = self._snapshots[baseline_id]
        compressed = self._snapshots[compressed_id]
        return compute_all_deltas(baseline['metrics'], compressed['metrics'])

    def get_metrics_dataframe(self, snapshot_id: str, group: str) -> pd.DataFrame:
        """Get a DataFrame of metrics for a specific group."""
        snap = self._snapshots[snapshot_id]
        group_data = snap['metrics'].get(group, {})
        rows = []
        for key, val in group_data.items():
            if key in _NON_SCORE_KEYS or val is None:
                continue
            rows.append({'metric': key, 'value': val})
        return pd.DataFrame(rows)

    def get_comparison_dataframe(
        self, baseline_id: str, compressed_id: str, group: str
    ) -> pd.DataFrame:
        """Get a comparison DataFrame for a metric group."""
        baseline = self._snapshots[baseline_id]
        compressed = self._snapshots[compressed_id]
        b_group = baseline['metrics'].get(group, {})
        c_group = compressed['metrics'].get(group, {})

        rows = []
        for key in b_group:
            if key in _NON_SCORE_KEYS:
                continue
            b_val = b_group.get(key)
            c_val = c_group.get(key)
            if b_val is None or c_val is None:
                continue
            delta = compute_delta(b_val, c_val, key)
            retention = compute_retention(b_val, c_val, key)
            norm = normalize_delta(delta, b_val, key)
            rows.append({
                'metric': key,
                'baseline': b_val,
                'compressed': c_val,
                'delta': delta,
                'retention_pct': retention,
                'normalized': norm,
                'is_inverted': key in INVERTED_METRICS,
            })
        return pd.DataFrame(rows)


# Singleton store
store = SnapshotStore()
