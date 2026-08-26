"""Comparison page — side-by-side baseline vs. compressed model analysis."""
import dash_bootstrap_components as dbc
from dash import html, dcc, callback, Input, Output
from dashboard.data_bridge import store
from dashboard.components.efficiency import create_efficiency_panel
from dashboard.components.radar import create_radar_chart
from dashboard.components.bar_charts import create_group_comparison_bars
from dashboard.styles import METRIC_GROUP_LABELS, COLORS


def layout():
    baseline_options = store.get_baseline_options()
    compressed_options = store.get_compressed_options()

    if not baseline_options or not compressed_options:
        return dbc.Container(
            html.H3(
                "Insufficient snapshots for comparison. "
                "Please load at least one baseline and one compressed snapshot.",
                className="text-white mt-4",
            ),
            fluid=True,
        )

    default_baseline = baseline_options[0]['value']
    default_compressed = compressed_options[0]['value']

    group_options = [
        {'label': v, 'value': k}
        for k, v in METRIC_GROUP_LABELS.items()
    ]

    controls = dbc.Row([
        dbc.Col([
            html.Label("Baseline Model", className="text-white mb-1"),
            dcc.Dropdown(
                id='compare-baseline-dropdown',
                options=baseline_options,
                value=default_baseline,
                clearable=False,
                className="text-dark",
            ),
        ], md=6),
        dbc.Col([
            html.Label("Compressed Model", className="text-white mb-1"),
            dcc.Dropdown(
                id='compare-compressed-dropdown',
                options=compressed_options,
                value=default_compressed,
                clearable=False,
                className="text-dark",
            ),
        ], md=6),
    ], className="mt-4 mb-4")

    return dbc.Container([
        controls,
        dbc.Row([
            dbc.Col(id='compare-radar-container', lg=6, md=12, className="mb-4"),
            dbc.Col(id='compare-efficiency-container', lg=6, md=12, className="mb-4"),
        ]),
        dbc.Row([
            dbc.Col([
                html.Label("Metric Group", className="text-white mb-1"),
                dcc.Dropdown(
                    id='compare-group-dropdown',
                    options=group_options,
                    value=list(METRIC_GROUP_LABELS.keys())[0],
                    clearable=False,
                    className="text-dark",
                ),
            ], md=6),
        ], className="mb-4"),
        dbc.Row(dbc.Col(id='compare-bar-container')),
    ], fluid=True)


@callback(
    Output('compare-radar-container', 'children'),
    Output('compare-efficiency-container', 'children'),
    Output('compare-bar-container', 'children'),
    Input('compare-baseline-dropdown', 'value'),
    Input('compare-compressed-dropdown', 'value'),
    Input('compare-group-dropdown', 'value'),
)
def update_compare_charts(baseline_id, compressed_id, group_id):
    if not baseline_id or not compressed_id:
        return html.Div(), html.Div(), html.Div()

    baseline_snap = store.get_snapshot(baseline_id)
    compressed_snap = store.get_snapshot(compressed_id)

    if not baseline_snap or not compressed_snap:
        return html.Div(), html.Div(), html.Div()

    # --- Efficiency panel ---
    efficiency_panel = create_efficiency_panel(baseline_snap, compressed_snap)

    # --- Radar chart ---
    rets = store.get_retentions(baseline_id, compressed_id)
    baseline_rets = {k: 100.0 for k in rets}  # Baseline is always 100%
    retentions_list = [baseline_rets, rets]
    labels = [baseline_snap['label'], compressed_snap['label']]

    radar_fig = create_radar_chart(
        retentions_list, labels, title="Capability Retention"
    )
    radar_chart = dcc.Graph(figure=radar_fig, config={'responsive': True})

    # --- Bar chart for selected metric group ---
    df = store.get_comparison_dataframe(baseline_id, compressed_id, group_id)
    if df is not None and not df.empty:
        bar_fig = create_group_comparison_bars(
            df, group_id,
            baseline_snap['label'],
            compressed_snap['label'],
        )
        bar_chart = dcc.Graph(figure=bar_fig, config={'responsive': True})
    else:
        bar_chart = html.Div(
            "No data for this group.",
            className="text-white",
            style={'padding': '20px'},
        )

    return radar_chart, efficiency_panel, bar_chart
