"""VLM-Compress-Bench Dashboard Application.

Launch with: python -m dashboard.app
"""
import sys
from pathlib import Path

# Ensure project root is on path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output, callback

from dashboard.styles import DBC_THEME, COLORS
from dashboard.components.navbar import create_navbar
from dashboard.data_bridge import store

# Import page layouts
from dashboard.pages import overview, comparison, deep_dive, manager

# Initialize Dash app
app = dash.Dash(
    __name__,
    external_stylesheets=[
        DBC_THEME,
        'https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.0/font/bootstrap-icons.css',
        'https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap',
    ],
    suppress_callback_exceptions=True,
    title='VLM-Compress-Bench',
    update_title='Loading...',
)

server = app.server  # For deployment

# App layout
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    create_navbar(),
    html.Div(id='page-content', style={'minHeight': '80vh'}),
], style={
    'backgroundColor': COLORS['background'],
    'minHeight': '100vh',
    'fontFamily': 'Inter, system-ui, sans-serif',
})

# Page routing
@callback(Output('page-content', 'children'), Input('url', 'pathname'))
def display_page(pathname):
    if pathname == '/compare':
        return comparison.layout()
    elif pathname == '/metrics':
        return deep_dive.layout()
    elif pathname == '/snapshots':
        return manager.layout()
    else:
        return overview.layout()

# Load data on startup
store.load_defaults()

if __name__ == '__main__':
    app.run(debug=True, port=8050)
