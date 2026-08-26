"""Summary statistic cards for the dashboard."""
import dash_bootstrap_components as dbc
from dash import html
from dashboard.styles import COLORS, CARD_STYLE

def create_stat_card(title: str, value: str, subtitle: str = '', icon: str = '', color: str = None):
    """Create a single stat card.
    
    Args:
        title: Card title (e.g. 'Models Loaded')
        value: Main display value (e.g. '3')
        subtitle: Secondary text
        icon: Bootstrap icon class (e.g. 'bi-cpu')
        color: Accent color for the icon/value
    """
    if color is None:
        color = COLORS.get('accent_blue', '#0d6efd')
    
    return dbc.Card(
        dbc.CardBody([
            html.Div([
                html.I(className=f'{icon} me-2', style={'fontSize': '1.5rem', 'color': color}) if icon else None,
                html.Span(title, style={'color': COLORS['text_secondary'], 'fontSize': '0.85rem', 'textTransform': 'uppercase', 'letterSpacing': '0.5px'}),
            ], className='d-flex align-items-center mb-2'),
            html.H3(value, style={'color': color, 'fontWeight': '700', 'marginBottom': '4px'}),
            html.Small(subtitle, style={'color': COLORS.get('text_muted', '#6c757d')}) if subtitle else None,
        ]),
        style={**CARD_STYLE},
        className='h-100',
    )

def create_efficiency_card(label: str, baseline_val, compressed_val, unit: str = '', better: str = 'lower'):
    """Create a card comparing an efficiency metric between baseline and compressed.
    
    Args:
        label: Metric name
        baseline_val: Baseline value
        compressed_val: Compressed value  
        unit: Unit suffix (e.g. 'MB', 'ms')
        better: 'lower' or 'higher' - which direction is better
    """
    if better == 'lower':
        is_better = compressed_val < baseline_val
    else:
        is_better = compressed_val > baseline_val
    
    color = COLORS['improvement'] if is_better else COLORS['degradation']
    arrow = '↓' if compressed_val < baseline_val else '↑'
    
    if baseline_val != 0:
        pct_change = ((compressed_val - baseline_val) / baseline_val) * 100
    else:
        pct_change = 0
    
    return dbc.Card(
        dbc.CardBody([
            html.Div(label, style={'color': COLORS['text_secondary'], 'fontSize': '0.8rem', 'textTransform': 'uppercase', 'letterSpacing': '0.5px', 'marginBottom': '8px'}),
            html.Div([
                html.Div([
                    html.Div('Baseline', style={'color': COLORS.get('text_muted', '#6c757d'), 'fontSize': '0.75rem'}),
                    html.Div(f'{baseline_val:,.1f} {unit}', style={'color': COLORS['text_primary'], 'fontWeight': '600'}),
                ], style={'flex': '1'}),
                html.Div([
                    html.Div('→', style={'color': COLORS.get('text_muted', '#6c757d'), 'fontSize': '1.2rem', 'padding': '0 8px'}),
                ], className='d-flex align-items-center'),
                html.Div([
                    html.Div('Compressed', style={'color': COLORS.get('text_muted', '#6c757d'), 'fontSize': '0.75rem'}),
                    html.Div(f'{compressed_val:,.1f} {unit}', style={'color': color, 'fontWeight': '600'}),
                ], style={'flex': '1'}),
            ], className='d-flex'),
            html.Div(
                f'{arrow} {abs(pct_change):.1f}%',
                style={'color': color, 'fontSize': '0.85rem', 'fontWeight': '600', 'marginTop': '8px'},
            ),
        ]),
        style={**CARD_STYLE},
        className='h-100',
    )
