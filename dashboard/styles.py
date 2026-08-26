"""Centralized theme and style configuration for the VLM-Compress-Bench dashboard."""
import dash_bootstrap_components as dbc

# Bootstrap theme
DBC_THEME = dbc.themes.DARKLY

# Color palette
COLORS = {
    'background': '#0f1117',
    'card_bg': '#1a1d26',
    'card_border': '#2d3140',
    'text_primary': '#e8eaed',
    'text_secondary': '#9aa0a6',
    'text_muted': '#5f6368',
    'accent_blue': '#8ab4f8',
    'accent_purple': '#c58af9',
    'accent_cyan': '#78d9ec',
    'baseline': '#8ab4f8',        # Blue for baseline
    'compressed_1': '#f28b82',    # Coral for first compressed
    'compressed_2': '#81c995',    # Green for second compressed
    'compressed_3': '#fcad70',    # Orange for third compressed
    'improvement': '#81c995',     # Green
    'degradation': '#f28b82',     # Red/Coral
    'neutral': '#c58af9',         # Purple
    'chart_bg': '#1a1d26',
    'grid_color': '#2d3140',
}

# Model color assignment (order matters)
MODEL_COLORS = [
    COLORS['baseline'],
    COLORS['compressed_1'],
    COLORS['compressed_2'],
    COLORS['compressed_3'],
]

# Chart layout template
CHART_TEMPLATE = 'plotly_dark'

CHART_LAYOUT = dict(
    paper_bgcolor=COLORS['chart_bg'],
    plot_bgcolor=COLORS['chart_bg'],
    font=dict(family='Inter, system-ui, sans-serif', color=COLORS['text_primary']),
    margin=dict(l=40, r=40, t=50, b=40),
    legend=dict(
        bgcolor='rgba(0,0,0,0)',
        font=dict(size=12),
    ),
)

# Card style
CARD_STYLE = {
    'backgroundColor': COLORS['card_bg'],
    'border': f"1px solid {COLORS['card_border']}",
    'borderRadius': '12px',
    'padding': '20px',
}

# Metric group display names
METRIC_GROUP_LABELS = {
    'vqa': 'VQA',
    'captioning': 'Image Captioning',
    'visual_reasoning': 'Visual Reasoning',
    'ocr_document': 'OCR / Document',
    'hallucination': 'Hallucination',
    'spatial_awareness': 'Spatial Awareness',
}

# Metric display names (for sub-metrics)
METRIC_LABELS = {
    'accuracy': 'Accuracy',
    'yes_no_accuracy': 'Yes/No Accuracy',
    'number_accuracy': 'Number Accuracy',
    'other_accuracy': 'Other Accuracy',
    'cider': 'CIDEr',
    'bleu4': 'BLEU-4',
    'meteor': 'METEOR',
    'rouge_l': 'ROUGE-L',
    'consistency': 'Consistency',
    'validity': 'Validity',
    'plausibility': 'Plausibility',
    'anls': 'ANLS',
    'exact_match_accuracy': 'Exact Match',
    'partial_match_rate': 'Partial Match',
    'chair_s': 'CHAIR-S ↓',
    'chair_i': 'CHAIR-I ↓',
    'coverage': 'Coverage',
    'above_below_accuracy': 'Above/Below',
    'left_right_accuracy': 'Left/Right',
    'near_far_accuracy': 'Near/Far',
    'inside_outside_accuracy': 'Inside/Outside',
    'gap_above_random': 'Gap Above Random',
}
