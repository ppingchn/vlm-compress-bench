import dash_bootstrap_components as dbc
from dash import html, dcc, callback, Input, Output
from dashboard.data_bridge import store
from dashboard.styles import METRIC_GROUP_LABELS

def layout():
    baseline_options = store.get_baseline_options()
    compressed_options = store.get_compressed_options()
    
    if not baseline_options or not compressed_options:
        return dbc.Container(html.H3("Insufficient snapshots.", className="text-white mt-4"), fluid=True)
        
    default_baseline = baseline_options[0]['value'] if baseline_options else None
    default_compressed = compressed_options[0]['value'] if compressed_options else None
    
    controls = dbc.Row([
        dbc.Col([
            html.Label("Metric Group", className="text-white"),
            dcc.Dropdown(id='deep-dive-group-dropdown', options=[{'label': v, 'value': k} for k, v in METRIC_GROUP_LABELS.items()], value=list(METRIC_GROUP_LABELS.keys())[0], clearable=False, className="text-dark")
        ], md=4),
        dbc.Col([
            html.Label("Baseline Model", className="text-white"),
            dcc.Dropdown(id='deep-dive-baseline-dropdown', options=baseline_options, value=default_baseline, clearable=False, className="text-dark")
        ], md=4),
        dbc.Col([
            html.Label("Compressed Model", className="text-white"),
            dcc.Dropdown(id='deep-dive-compressed-dropdown', options=compressed_options, value=default_compressed, clearable=False, className="text-dark")
        ], md=4)
    ], className="mt-4 mb-4")
    
    return dbc.Container([
        controls,
        html.Div(id='deep-dive-header', className="mb-4 text-white"),
        dbc.Row(dbc.Col(id='deep-dive-table-container'))
    ], fluid=True)

@callback(
    Output('deep-dive-header', 'children'),
    Output('deep-dive-table-container', 'children'),
    Input('deep-dive-group-dropdown', 'value'),
    Input('deep-dive-baseline-dropdown', 'value'),
    Input('deep-dive-compressed-dropdown', 'value')
)
def update_deep_dive(group_id, baseline_id, compressed_id):
    if not group_id or not baseline_id or not compressed_id:
        return "", html.Div()
        
    df = store.get_comparison_dataframe(baseline_id, compressed_id, group_id)
    if df is None or df.empty:
        return html.H4(f"No data for {METRIC_GROUP_LABELS.get(group_id, group_id)}"), html.Div()
        
    header = html.H4(f"Detailed Comparison: {METRIC_GROUP_LABELS.get(group_id, group_id)}")
    table = dbc.Table.from_dataframe(df, striped=True, bordered=True, hover=True, color="dark")
    
    return header, table
