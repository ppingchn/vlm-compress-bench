"""Overview page — at-a-glance summary of all loaded snapshots."""
import dash_bootstrap_components as dbc
from dash import html, dcc
from dashboard.data_bridge import store
from dashboard.components.cards import create_stat_card
from dashboard.components.radar import create_radar_chart
from dashboard.components.bar_charts import create_overview_bars
from dashboard.styles import COLORS


def layout():
    snapshots = store.all_snapshots
    if not snapshots:
        return dbc.Container([
            html.H3("No snapshots loaded.", className="text-white mt-4")
        ], fluid=True)

    baseline = store.baseline
    compressed = store.compressed_snapshots

    num_models = len(snapshots)
    baseline_name = baseline['label'] if baseline else "None"

    best_ratio = "N/A"
    avg_retention = "N/A"
    if compressed and baseline:
        # Get compression ratios from general_efficiency (correct path)
        ratios = [
            snap['general_efficiency'].get('compression_ratio', 0)
            for snap in compressed
        ]
        valid_ratios = [r for r in ratios if r is not None and r > 0]
        if valid_ratios:
            best_ratio = f"{max(valid_ratios):.1f}×"

        all_rets = []
        for c in compressed:
            rets = store.get_retentions(baseline['snapshot_id'], c['snapshot_id'])
            all_rets.extend(rets.values())
        if all_rets:
            # Retentions are already percentages (e.g. 95.2)
            avg_retention = f"{sum(all_rets) / len(all_rets):.1f}%"

    cards_row = dbc.Row([
        dbc.Col(create_stat_card(
            "Models Loaded", str(num_models),
            icon="bi-collection", color=COLORS['accent_blue'],
        ), md=3, sm=6, className="mb-3"),
        dbc.Col(create_stat_card(
            "Baseline Model", baseline_name,
            icon="bi-star", color=COLORS['accent_purple'],
        ), md=3, sm=6, className="mb-3"),
        dbc.Col(create_stat_card(
            "Best Compression", best_ratio,
            icon="bi-arrows-angle-contract", color=COLORS['accent_cyan'],
        ), md=3, sm=6, className="mb-3"),
        dbc.Col(create_stat_card(
            "Avg Retention", avg_retention,
            icon="bi-speedometer", color=COLORS['improvement'],
        ), md=3, sm=6, className="mb-3"),
    ], className="mt-4")

    # Build radar data — baseline is always 100% on all axes
    labels = []
    retentions_list = []

    if baseline and compressed:
        # Use one compressed model to discover the group keys
        sample_rets = store.get_retentions(
            baseline['snapshot_id'], compressed[0]['snapshot_id']
        )
        # Baseline = 100% on every axis
        labels.append(baseline['label'])
        retentions_list.append({group: 100.0 for group in sample_rets})

        for c in compressed:
            labels.append(c['label'])
            retentions_list.append(
                store.get_retentions(baseline['snapshot_id'], c['snapshot_id'])
            )

    radar_chart = html.Div()
    overview_bars = html.Div()
    if labels and retentions_list:
        try:
            radar_fig = create_radar_chart(
                retentions_list, labels, title="Capability Retention"
            )
            radar_chart = dcc.Graph(
                figure=radar_fig, config={'responsive': True}
            )
        except Exception:
            pass

        try:
            bars_fig = create_overview_bars(retentions_list, labels)
            overview_bars = dcc.Graph(
                figure=bars_fig, config={'responsive': True}
            )
        except Exception:
            pass

    return dbc.Container([
        cards_row,
        dbc.Row([
            dbc.Col(radar_chart, lg=6, md=12, className="mb-4"),
            dbc.Col(overview_bars, lg=6, md=12, className="mb-4"),
        ])
    ], fluid=True)
