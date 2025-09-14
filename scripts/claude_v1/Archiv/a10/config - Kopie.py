"""
Konfiguration und Konstanten für Bullish Divergence Analyzer
"""

import os
import tempfile
from pathlib import Path

# Server Konfiguration (nur für app.run())
SERVER_CONFIG = {
    'host': '0.0.0.0',
    'port': 5000,
    'debug': False
}

# Flask App Konfiguration (separate von Server-Config)
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

# Farben für Varianten (kräftige Farben)
VARIANT_COLORS = [
    '#FF0000',  # Knallrot
    '#00FF00',  # Knallgrün  
    '#0080FF',  # Hellblau
    '#FFD700',  # Gold
    '#FF00FF',  # Magenta
    '#00FFFF',  # Cyan
    '#FFA500',  # Orange
    '#FF1493',  # Deep Pink
    '#7FFF00',  # Chartreuse
    '#9370DB'   # Medium Purple
]

# Chart Konfiguration
CHART_CONFIG = {
    'height': 800,
    'background_color': '#0a0a0a',
    'paper_color': '#1a1a1a',
    'grid_color': 'rgba(255,255,255,0.1)',
    'font_color': '#FFFFFF',
    'ema_colors': {
        20: '#FFD700',   # Gold
        50: '#00FFFF',   # Cyan
        100: '#FF00FF',  # Magenta
        200: '#9370DB'   # Purple
    }
}

# Export Konfiguration
EXPORT_CONFIG = {
    'formats': ['csv', 'json', 'excel'],
    'export_dir': Path('exports'),
    'include_stats': True
}

# Pfade
PATHS = {
    'static': Path('static'),
    'exports': Path('exports'),
    'configs': Path('configs'),
    'uploads': Path(tempfile.gettempdir()) / 'bullish_analyzer_uploads'
}

# Erstelle notwendige Verzeichnisse
for path in PATHS.values():
    try:
        path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Warnung: Konnte Verzeichnis {path} nicht erstellen: {e}")