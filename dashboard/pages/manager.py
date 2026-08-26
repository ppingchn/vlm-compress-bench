import dash_bootstrap_components as dbc
from dash import html, dcc, callback, Input, Output, State
from dashboard.data_bridge import store
from dashboard.components.snapshot_table import create_snapshot_table

def layout():
    upload_area = dcc.Upload(
        id='manager-upload',
        children=html.Div(['Drag and Drop or ', html.A('Select Files')]),
        style={
            'width': '100%', 'height': '60px', 'lineHeight': '60px',
            'borderWidth': '1px', 'borderStyle': 'dashed',
            'borderRadius': '5px', 'textAlign': 'center', 'margin': '10px',
            'color': 'white'
        },
        multiple=False
    )
    
    return dbc.Container([
        html.H3("Snapshot Manager", className="text-white mt-4 mb-4"),
        upload_area,
        html.Div(id='manager-upload-feedback'),
        html.Div(id='manager-snapshot-table', children=create_snapshot_table(store.all_snapshots))
    ], fluid=True)

@callback(
    Output('manager-upload-feedback', 'children'),
    Output('manager-snapshot-table', 'children'),
    Input('manager-upload', 'contents'),
    State('manager-upload', 'filename'),
    prevent_initial_call=True,
)
def handle_upload(contents, filename):
    import base64, json
    if contents is None:
        return '', create_snapshot_table(store.all_snapshots)
    
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        snap = json.loads(decoded)
        store.add_snapshot(snap)
        feedback = dbc.Alert(f'✓ Successfully loaded {filename}', color='success', dismissable=True)
    except Exception as e:
        feedback = dbc.Alert(f'✗ Error: {str(e)}', color='danger', dismissable=True)
    
    return feedback, create_snapshot_table(store.all_snapshots)
