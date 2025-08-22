"""
Konfiguration für Bullish Divergence Analyzer
"""

import os
import tempfile
from pathlib import Path

# Flask App Konfiguration
APP_CONFIG = {
    'UPLOAD_FOLDER': tempfile.gettempdir(),
    'MAX_CONTENT_LENGTH': 100 * 1024 * 1024,  # 100 MB
    'SECRET_KEY': 'bullish-divergence-analyzer-key'
}

# Analyse Parameter Defaults
DEFAULT_PARAMS = {
    'window': 5,
    'candle_tolerance': 0.1,
    'macd_tolerance': 3.25
}

# Varianten Presets
VARIANT_PRESETS = {
    'standard': {
        'name': 'Standard',
        'window': 5,
        'candleTol': 0.1,
        'macdTol': 3.25
    },
    'conservative': {
        'name': 'Konservativ',
        'window': 7,
        'candleTol': 0.05,
        'macdTol': 2.0
    },
    'aggressive': {
        'name': 'Aggressiv',
        'window': 3,
        'candleTol': 0.2,
        'macdTol': 5.0
    }
}

# Farben für Varianten
VARIANT_COLORS = [
    '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4',
    '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F'
]

# Chart Konfiguration
CHART_CONFIG = {
    'height': 600,
    'background_color': '#0a0a0a',
    'paper_color': '#1a1a1a',
    'grid_color': 'rgba(255,255,255,0.1)',
    'font_color': '#FFFFFF',
    'ema_colors': {
        20: '#FFD700',
        50: '#00FFFF',
        100: '#FF00FF',
        200: '#9370DB'
    }
}