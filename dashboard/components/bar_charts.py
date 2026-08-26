"""Bar chart components for metric comparisons."""
import plotly.graph_objects as go
from typing import Dict, List
import pandas as pd

from dashboard.styles import COLORS, CHART_LAYOUT, METRIC_GROUP_LABELS, METRIC_LABELS, MODEL_COLORS
from benchmarker.normalization import INVERTED_METRICS


def create_group_comparison_bars(
    comparison_df: pd.DataFrame,
    group_name: str,
    baseline_label: str,
    compressed_label: str,
) -> go.Figure:
    """Create a grouped horizontal bar chart for sub-metrics in a group.

    Args:
        comparison_df: DataFrame from SnapshotStore.get_comparison_dataframe().
            Columns: metric, baseline, compressed, delta, retention_pct,
            normalized, is_inverted.
        group_name: Name of the metric group.
        baseline_label: Label for the baseline model.
        compressed_label: Label for the compressed model.
    """
    fig = go.Figure()

    metrics = comparison_df['metric'].tolist()
    display_metrics = [METRIC_LABELS.get(m, m) for m in metrics]

    baseline_vals = comparison_df['baseline'].tolist()
    compressed_vals = comparison_df['compressed'].tolist()
    deltas = comparison_df['delta'].tolist()

    fig.add_trace(go.Bar(
        y=display_metrics,
        x=baseline_vals,
        name=baseline_label,
        orientation='h',
        marker_color=MODEL_COLORS[0],
        hovertemplate='%{y}: %{x:.2f}<extra>' + baseline_label + '</extra>',
    ))

    fig.add_trace(go.Bar(
        y=display_metrics,
        x=compressed_vals,
        name=compressed_label,
        orientation='h',
        marker_color=MODEL_COLORS[1],
        hovertemplate='%{y}: %{x:.2f}<extra>' + compressed_label + '</extra>',
    ))

    # Add annotations for delta
    for i, (b, c, d, m) in enumerate(zip(baseline_vals, compressed_vals, deltas, metrics)):
        if b is None or c is None or pd.isna(b) or pd.isna(c):
            continue

        is_inverted = m in INVERTED_METRICS
        # Delta from compute_delta is already direction-adjusted (positive = better)
        is_better = d > 0

        color = COLORS['improvement'] if is_better else (
            COLORS['degradation'] if d != 0 else COLORS['text_muted']
        )
        sign = '+' if d > 0 else ''
        text = f"{sign}{d:.2f}"

        max_val = max(abs(b), abs(c))
        fig.add_annotation(
            y=display_metrics[i],
            x=max_val + (0.05 * max_val if max_val != 0 else 0.5),
            text=text,
            showarrow=False,
            font=dict(color=color, size=11, family='Inter, system-ui, sans-serif'),
            xanchor='left',
        )

    fig.update_layout(
        **CHART_LAYOUT,
        barmode='group',
        title=dict(
            text=f'{METRIC_GROUP_LABELS.get(group_name, group_name)} — Metric Comparison',
            font=dict(size=14),
        ),
        yaxis=dict(autorange="reversed"),
        xaxis=dict(gridcolor=COLORS['grid_color']),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig


def create_overview_bars(
    retentions_list: List[Dict[str, float]],
    labels_list: List[str],
) -> go.Figure:
    """Create vertical bar chart for group retentions across models.

    Args:
        retentions_list: List of dicts mapping group_name -> retention_pct.
        labels_list: List of model labels.
    """
    fig = go.Figure()

    categories = list(METRIC_GROUP_LABELS.keys())
    display_names = [METRIC_GROUP_LABELS.get(c, c) for c in categories]

    for i, (retentions, label) in enumerate(zip(retentions_list, labels_list)):
        vals = [retentions.get(cat, 0) for cat in categories]

        fig.add_trace(go.Bar(
            x=display_names,
            y=vals,
            name=label,
            marker_color=MODEL_COLORS[i % len(MODEL_COLORS)],
            hovertemplate='%{x}: %{y:.1f}%<extra>' + label + '</extra>',
        ))

    # Add 100% reference line
    fig.add_hline(
        y=100,
        line_dash="dash",
        line_color=COLORS['text_muted'],
        annotation_text="100% (Baseline)",
        annotation_position="top right",
        annotation_font_color=COLORS['text_muted'],
    )

    # Compute sensible y-axis range
    all_vals = []
    for r in retentions_list:
        all_vals.extend(r.values())
    y_max = max(all_vals) if all_vals else 120

    fig.update_layout(
        **CHART_LAYOUT,
        barmode='group',
        title=dict(text='Capability Retention Overview', font=dict(size=14)),
        yaxis=dict(
            title='Retention (%)',
            range=[0, min(y_max + 15, 150)],
            gridcolor=COLORS['grid_color'],
        ),
        xaxis=dict(gridcolor=COLORS['grid_color']),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig
