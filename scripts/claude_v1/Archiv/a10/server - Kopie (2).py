"""
Flask Server für Bullish Divergence Analyzer
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import json
from datetime import datetime
from pathlib import Path
import os
import tempfile

from config import SERVER_CONFIG, APP_CONFIG, PATHS, VARIANT_PRESETS, EXPORT_CONFIG
from analysis_engine import AnalysisEngine

# Flask App
app = Flask(__name__, static_folder='static')
CORS(app)

# Flask App Konfiguration (getrennt von Server-Config)
app.config.update(APP_CONFIG)

# Analysis Engine
engine = AnalysisEngine()

# Session Storage (in production würde man Redis/DB verwenden)
sessions = {}

@app.route('/')
def index():
    """Serve HTML"""
    return send_from_directory('static', 'index.html')

@app.route('/static/<path:path>')
def serve_static(path):
    """Serve static files"""
    return send_from_directory('static', path)

@app.route('/api/health')
def health():
    """Health check"""
    return jsonify({
        'status': 'ok',
        'modules': engine.check_modules(),
        'version': '2.0.0'
    })

@app.route('/upload', methods=['POST'])
def upload_file():
    """Datei Upload und Validierung"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'Keine Datei'}), 400
        
        file = request.files['file']
        if not file.filename:
            return jsonify({'error': 'Leere Datei'}), 400
        
        # Speichern
        filename = file.filename
        upload_dir = PATHS['uploads']
        upload_dir.mkdir(exist_ok=True)
        filepath = upload_dir / f"{datetime.now().timestamp()}_{filename}"
        file.save(str(filepath))
        
        # Laden und validieren
        df = engine.load_data(str(filepath))
        if df is None:
            return jsonify({'error': 'Ungültige Datendatei'}), 400
        
        # Session erstellen
        session_id = str(hash(f"{filename}_{datetime.now()}"))
        sessions[session_id] = {
            'filepath': str(filepath),
            'filename': filename,
            'rows': len(df),
            'columns': list(df.columns),
            'date_range': [
                str(df['date'].min()) if 'date' in df else None,
                str(df['date'].max()) if 'date' in df else None
            ]
        }
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'info': sessions[session_id]
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/analyze', methods=['POST'])
def analyze():
    """Hauptanalyse Endpoint"""
    try:
        data = request.json
        session_id = data.get('session_id')
        variants = data.get('variants', [])
        date_range = data.get('date_range', None)
        
        if session_id not in sessions:
            return jsonify({'error': 'Session abgelaufen'}), 404
        
        # Analyse durchführen
        filepath = sessions[session_id]['filepath']
        results = engine.analyze(filepath, variants, date_range)
        
        if results is None:
            return jsonify({'error': 'Analyse fehlgeschlagen'}), 500
        
        return jsonify({
            'success': True,
            **results
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/export', methods=['POST'])
def export_results():
    """Export der Analyseergebnisse"""
    try:
        data = request.json
        format_type = data.get('format', 'csv')
        results = data.get('results')
        variants = data.get('variants')
        
        if format_type not in EXPORT_CONFIG['formats']:
            return jsonify({'error': f'Format {format_type} nicht unterstützt'}), 400
        
        # Export erstellen
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"divergence_analysis_{timestamp}.{format_type}"
        export_dir = PATHS['exports']
        export_dir.mkdir(exist_ok=True)
        filepath = export_dir / filename
        
        if format_type == 'json':
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump({'results': results, 'variants': variants}, f, indent=2, ensure_ascii=False)
        
        elif format_type == 'csv':
            # Flatten results für CSV
            rows = []
            for variant_id, data in results.items():
                variant = next((v for v in variants if v['id'] == variant_id), {})
                for div_type in ['classic', 'hidden']:
                    for div in data.get(div_type, []):
                        rows.append({
                            'variant': variant.get('name', ''),
                            'type': div_type,
                            'date': div.get('date'),
                            'low': div.get('low'),
                            'rsi': div.get('rsi'),
                            'macd': div.get('macd')
                        })
            
            df = pd.DataFrame(rows)
            df.to_csv(filepath, index=False, encoding='utf-8')
        
        return jsonify({
            'success': True,
            'filename': filename,
            'path': str(filepath)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/presets')
def get_presets():
    """Liefert vordefinierte Varianten"""
    return jsonify(VARIANT_PRESETS)

@app.route('/api/save_config', methods=['POST'])
def save_config():
    """Speichert Varianten-Konfiguration"""
    try:
        data = request.json
        name = data.get('name', f'config_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        variants = data.get('variants', [])
        
        config_dir = PATHS['configs']
        config_dir.mkdir(exist_ok=True)
        filepath = config_dir / f"{name}.json"
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(variants, f, indent=2, ensure_ascii=False)
        
        return jsonify({'success': True, 'filename': name})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/load_config/<name>')
def load_config(name):
    """Lädt gespeicherte Varianten-Konfiguration"""
    try:
        filepath = PATHS['configs'] / f"{name}.json"
        if not filepath.exists():
            return jsonify({'error': 'Konfiguration nicht gefunden'}), 404
        
        with open(filepath, 'r', encoding='utf-8') as f:
            variants = json.load(f)
        
        return jsonify({'success': True, 'variants': variants})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/list_configs')
def list_configs():
    """Listet verfügbare Konfigurationen"""
    try:
        config_dir = PATHS['configs']
        config_dir.mkdir(exist_ok=True)
        configs = [f.stem for f in config_dir.glob('*.json')]
        return jsonify(configs)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def start_server():
    """Server starten"""
    print(f"📍 Server läuft auf: http://localhost:{SERVER_CONFIG['port']}")
    print("   Zum Beenden: Strg+C")
    
    # Nur gültige Flask app.run() Parameter verwenden
    run_config = {
        'host': SERVER_CONFIG['host'],
        'port': SERVER_CONFIG['port'],
        'debug': SERVER_CONFIG['debug']
    }
    
    app.run(**run_config)