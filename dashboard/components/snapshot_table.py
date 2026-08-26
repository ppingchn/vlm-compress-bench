"""Snapshot table component."""
import dash_bootstrap_components as dbc
from dash import html
from datetime import datetime
from typing import List, Dict, Any

from dashboard.styles import COLORS

def create_snapshot_table(snapshots_list: List[Dict[str, Any]]) -> dbc.Table:
    """Create a table displaying snapshot information.
    
    Args:
        snapshots_list: List of snapshot dictionaries.
        
    Returns:
        dbc.Table component.
    """
    if not snapshots_list:
        return html.Div("No snapshots found.", style={'color': COLORS.get('text_muted', '#6c757d')})
        
    # Table header
    header = [
        html.Thead(html.Tr([
            html.Th("Label"),
            html.Th("Model Name"),
            html.Th("Precision"),
            html.Th("Method"),
            html.Th("Ratio"),
            html.Th("Timestamp"),
            html.Th("ID"),
        ]))
    ]
    
    # Table rows
    rows = []
    for snap in snapshots_list:
        info = snap.get('model_info', {})
        eff = snap.get('general_efficiency', {})
        
        is_baseline = info.get('is_baseline', False)
        label_text = snap.get('label', 'Unnamed')
        if is_baseline:
            label_col = html.Td(html.Strong(label_text))
        else:
            label_col = html.Td(label_text)
            
        # Parse timestamp
        ts_raw = snap.get('timestamp', '')
        try:
            ts_dt = datetime.fromisoformat(ts_raw)
            ts_str = ts_dt.strftime('%Y-%m-%d %H:%M')
        except (ValueError, TypeError):
            ts_str = ts_raw
            
        method = info.get('compression_method') or 'None'
        ratio = f"{eff.get('compression_ratio', 1.0):.1f}x"
        short_id = snap.get('snapshot_id', '')[:8]
        
        row = html.Tr([
            label_col,
            html.Td(info.get('model_name', '')),
            html.Td(info.get('precision', '')),
            html.Td(method),
            html.Td(ratio),
            html.Td(ts_str),
            html.Td(html.Code(short_id)),
        ])
        rows.append(row)
        
    table_body = [html.Tbody(rows)]
    
    return dbc.Table(
        header + table_body,
        striped=True,
        bordered=True,
        hover=True,
        dark=True,
        responsive=True,
        className='mt-3',
        style={'backgroundColor': COLORS['card_bg']}
    )
