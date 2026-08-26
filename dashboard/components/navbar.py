"""Top navigation bar for the VLM-Compress-Bench dashboard."""
import dash_bootstrap_components as dbc
from dash import html
from dashboard.styles import COLORS

def create_navbar():
    """Create the top navigation bar."""
    return dbc.Navbar(
        dbc.Container([
            dbc.NavbarBrand(
                [
                    html.I(className='bi bi-gpu-card me-2'),
                    'VLM-Compress-Bench',
                ],
                href='/',
                className='fw-bold',
                style={'fontSize': '1.2rem'},
            ),
            dbc.NavbarToggler(id='navbar-toggler'),
            dbc.Collapse(
                dbc.Nav([
                    dbc.NavItem(dbc.NavLink('Overview', href='/', active='exact')),
                    dbc.NavItem(dbc.NavLink('Compare', href='/compare', active='exact')),
                    dbc.NavItem(dbc.NavLink('Deep Dive', href='/metrics', active='exact')),
                    dbc.NavItem(dbc.NavLink('Snapshots', href='/snapshots', active='exact')),
                ], navbar=True),
                id='navbar-collapse',
                navbar=True,
            ),
        ], fluid=True),
        color='dark',
        dark=True,
        className='mb-4',
        style={'borderBottom': f"1px solid {COLORS['card_border']}"},
    )
