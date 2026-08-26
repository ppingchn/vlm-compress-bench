"""Efficiency metrics comparison panel."""
import dash_bootstrap_components as dbc
from dash import html
from dashboard.components.cards import create_efficiency_card

def create_efficiency_panel(baseline_snap: dict, compressed_snap: dict):
    """Create a panel comparing efficiency metrics between two snapshots."""
    b_eff = baseline_snap['general_efficiency']
    c_eff = compressed_snap['general_efficiency']
    
    metrics = [
        ('Model Size', 'model_size_mb', 'MB', 'lower'),
        ('GPU Memory', 'gpu_memory_mb', 'MB', 'lower'),
        ('Latency', 'inference_latency_ms', 'ms', 'lower'),
        ('Throughput', 'throughput_tokens_per_sec', 'tok/s', 'higher'),
        ('Compression Ratio', 'compression_ratio', '×', 'higher'),
        ('Parameters', 'parameter_count', '', 'lower'),
    ]
    
    cards = []
    for label, key, unit, better in metrics:
        b_val = b_eff.get(key, 0)
        c_val = c_eff.get(key, 0)
        if key == 'parameter_count':
            # Format as billions
            b_display = b_val / 1e9 if b_val else 0
            c_display = c_val / 1e9 if c_val else 0
            unit = 'B'
        else:
            b_display = b_val
            c_display = c_val
        
        cards.append(
            dbc.Col(
                create_efficiency_card(label, b_display, c_display, unit, better),
                md=4, sm=6, xs=12, className='mb-3',
            )
        )
    
    return dbc.Row(cards)
