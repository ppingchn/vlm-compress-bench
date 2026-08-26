"""Radar chart for holistic model comparison."""
import plotly.graph_objects as go
from typing import Dict, List, Optional
from dashboard.styles import COLORS, CHART_LAYOUT, METRIC_GROUP_LABELS, MODEL_COLORS

def create_radar_chart(
    retentions_list: List[Dict[str, float]],
    labels: List[str],
    title: str = 'Capability Retention (%)',
) -> go.Figure:
    """Create a radar chart comparing retention across metric groups.
    
    Args:
        retentions_list: List of dicts, each mapping group_name -> retention_pct.
            First entry is typically the baseline (always 100%).
        labels: Display labels for each model trace.
        title: Chart title.
    
    Returns:
        A plotly Figure.
    """
    # Define the axes order
    categories = list(METRIC_GROUP_LABELS.keys())
    display_names = [METRIC_GROUP_LABELS[c] for c in categories]
    
    fig = go.Figure()
    
    for i, (retentions, label) in enumerate(zip(retentions_list, labels)):
        r_values = [retentions.get(cat, 0) for cat in categories]
        # Close the polygon
        r_values_closed = r_values + [r_values[0]]
        theta_closed = display_names + [display_names[0]]
        
        color = MODEL_COLORS[i % len(MODEL_COLORS)]
        
        fig.add_trace(go.Scatterpolar(
            r=r_values_closed,
            theta=theta_closed,
            name=label,
            line=dict(color=color, width=2.5),
            fill='toself',
            fillcolor=f'rgba({_hex_to_rgb(color)}, 0.1)',
            hovertemplate='%{theta}: %{r:.1f}%<extra>' + label + '</extra>',
        ))
    
    fig.update_layout(
        **CHART_LAYOUT,
        title=dict(text=title, font=dict(size=16)),
        polar=dict(
            bgcolor=COLORS['chart_bg'],
            radialaxis=dict(
                visible=True,
                range=[0, 120],
                ticksuffix='%',
                gridcolor=COLORS['grid_color'],
                linecolor=COLORS['grid_color'],
                tickfont=dict(size=10, color=COLORS.get('text_muted', '#6c757d')),
            ),
            angularaxis=dict(
                gridcolor=COLORS['grid_color'],
                linecolor=COLORS['grid_color'],
                tickfont=dict(size=12, color=COLORS['text_secondary']),
            ),
        ),
        showlegend=True,
        legend=dict(x=1.1, y=1),
    )
    
    return fig


def _hex_to_rgb(hex_color: str) -> str:
    """Convert '#RRGGBB' to 'R, G, B' string for rgba()."""
    hex_color = hex_color.lstrip('#')
    r, g, b = int(hex_color[:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f'{r}, {g}, {b}'
