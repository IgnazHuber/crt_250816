"""
Flask Server für Bullish Divergence Analyzer - KORRIGIERT
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import json
from datetime import datetime
from pathlib import Path
import os
import tempfile
import io

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
        'version': '2.0.0',
        'export_path': str(PATHS['exports'])  # ✅ Export-Pfad in Health-Check
    })

@app.route('/upload', methods=['POST'])
def upload_file():
    """🔧 KORRIGIERTER Datei Upload - Parquet + CSV robuster"""
    try:
        if 'file' not in request.files:
            print("❌ Keine Datei in request.files")
            return jsonify({'error': 'Keine Datei'}), 400
        
        file = request.files['file']
        if not file.filename:
            print("❌ Leerer Dateiname")
            return jsonify({'error': 'Leere Datei'}), 400
        
        filename = file.filename
        print(f"📤 Upload: {filename}")
        
        # NEU: Prüfe Content-Type und Dateigröße
        content_type = file.content_type
        print(f"📡 Content-Type: {content_type}")
        file_stream = io.BytesIO()
        file.save(file_stream)
        file_size = file_stream.getbuffer().nbytes
        print(f"📏 Dateigröße: {file_size / 1024 / 1024:.2f} MB")
        file_stream.seek(0)
        
        # NEU: Prüfe, ob der Stream leer ist
        if file_size == 0:
            print("❌ Stream ist leer!")
            return jsonify({'error': 'Datei ist leer oder nicht korrekt übertragen'}), 400
        
        # NEU: Logge Python- und Pandas-Version für Debugging
        import sys
        import pandas as pd
        import pyarrow
        print(f"🛠 Python-Version: {sys.version}")
        print(f"🛠 pandas-Version: {pd.__version__}")
        print(f"🛠 pyarrow-Version: {pyarrow.__version__}")
        
        # Laden aus Memory-Stream
        try:
            if filename.lower().endswith('.parquet'):
                print("📊 Lade Parquet aus Memory...")
                try:
                    df = pd.read_parquet(file_stream, engine='pyarrow')
                except ImportError:
                    print("⚠️ pyarrow nicht installiert, versuche fastparquet...")
                    df = pd.read_parquet(file_stream, engine='fastparquet')
                except Exception as fallback_error:
                    raise Exception(f"Parquet-Engine-Fehler: {fallback_error}. Installiere pyarrow oder fastparquet!")
            elif filename.lower().endswith('.csv'):
                print("📈 Lade CSV aus Memory...")
                file_stream.seek(0)
                df = pd.read_csv(file_stream)
            else:
                print(f"❌ Unbekanntes Format: {filename}")
                return jsonify({'error': f'Unbekanntes Format: {filename}'}), 400
                
        except Exception as load_error:
            print(f"❌ Lade-Fehler: {load_error}")
            import traceback
            traceback.print_exc()
            return jsonify({
                'error': f'Datei konnte nicht geladen werden: {str(load_error)}',
                'hint': 'Überprüfe, ob die Parquet-Datei korrekt übertragen wird (Content-Type, Dateigröße). '
                        'Teste die Datei lokal mit pd.read_parquet. Benötigte Spalten: date, open, high, low, close.'
            }), 400
        
        # NEU: Logge Spalten und erste Zeilen für Debugging
        print(f"📋 Spalten: {list(df.columns)}")
        print(f"📝 Erste Zeilen:\n{df.head().to_string()}")
        
        # Validierung
        required = ['date', 'open', 'high', 'low', 'close']
        missing = [col for col in required if col not in df.columns]
        
        if missing:
            available_cols = list(df.columns)
            print(f"❌ Fehlende Spalten: {missing}")
            return jsonify({
                'error': f'Fehlende Spalten: {missing}',
                'available_columns': available_cols,
                'help': 'Benötigt: date, open, high, low, close'
            }), 400
        
        # Backup-Speicherung
        upload_dir = PATHS['uploads']
        upload_dir.mkdir(exist_ok=True)
        filepath = upload_dir / f"{datetime.now().timestamp()}_{filename}"
        
        if len(df) < 1000000:
            if filename.lower().endswith('.parquet'):
                df.to_parquet(str(filepath), index=False)
            else:
                df.to_csv(str(filepath), index=False)
            file_path_str = str(filepath)
        else:
            file_path_str = f"memory://{filename}"
        
        # Session erstellen
        session_id = str(hash(f"{filename}_{datetime.now()}"))
        sessions[session_id] = {
            'filepath': file_path_str,
            'filename': filename,
            'rows': len(df),
            'columns': list(df.columns),
            'data': df if file_path_str.startswith('memory://') else None,
            'date_range': [
                str(df['date'].min()) if 'date' in df else None,
                str(df['date'].max()) if 'date' in df else None
            ]
        }
        
        print(f"✅ Upload erfolgreich: {len(df)} Zeilen, Session: {session_id}")
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'info': {
                'filename': filename,
                'rows': len(df),
                'columns': list(df.columns),
                'date_range': sessions[session_id]['date_range'],
                'memory_only': file_path_str.startswith('memory://')
            }
        })
        
    except Exception as e:
        print(f"❌ Upload-Fehler: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Upload fehlgeschlagen: {str(e)}'}), 500

@app.route('/api/upload', methods=['POST'])
def api_upload_file():
    """🔧 Weiterleitung für falsche Frontend-URL /api/upload"""
    print("📤 Weiterleitung von /api/upload zu /upload")
    return upload_file()

@app.route('/analyze', methods=['POST'])
def analyze():
    """🔧 KORRIGIERTE Hauptanalyse - Memory-Support"""
    try:
        data = request.json
        session_id = data.get('session_id')
        variants = data.get('variants', [])
        date_range = data.get('date_range', None)
        
        if session_id not in sessions:
            return jsonify({'error': 'Session abgelaufen'}), 404
        
        session_data = sessions[session_id]
        
        # ✅ MEMORY vs FILE Loading
        if session_data['data'] is not None:
            # Aus Memory laden (große Dateien)
            print("📊 Verwende Memory-Cache...")
            df = session_data['data']
        else:
            # Aus Datei laden
            print("📁 Lade aus Datei...")
            df = engine.load_data(session_data['filepath'])
            if df is None:
                return jsonify({'error': 'Daten konnten nicht geladen werden'}), 500
        
        # ✅ DIREKTE DataFrame-Analyse statt Dateipfad
        results = engine.analyze_dataframe(df, variants, date_range)
        
        if results is None:
            return jsonify({'error': 'Analyse fehlgeschlagen'}), 500
        
        return jsonify({
            'success': True,
            **results
        })
        
    except Exception as e:
        print(f"❌ Analyse-Fehler: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """🔧 Weiterleitung für falsche Frontend-URL /api/analyze"""
    print("📤 Weiterleitung von /api/analyze zu /analyze")
    return analyze()

@app.route('/export', methods=['POST'])
def export_results():
    """🔧 KORRIGIERTER Export mit absoluten Pfaden"""
    try:
        data = request.json
        format_type = data.get('format', 'csv')
        results = data.get('results')
        variants = data.get('variants')
        
        if format_type not in EXPORT_CONFIG['formats']:
            return jsonify({'error': f'Format {format_type} nicht unterstützt'}), 400
        
        # ✅ ABSOLUTER Export-Pfad verwenden
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"divergence_analysis_{timestamp}.{format_type}"
        export_dir = PATHS['exports']
        export_dir.mkdir(parents=True, exist_ok=True)  # Sicherstellen dass Pfad existiert
        filepath = export_dir / filename
        
        print(f"📤 Export nach: {filepath}")
        
        if format_type == 'json':
            with open(filepath, 'w', encoding='utf-8') as f:
                export_data = {
                    'results': results, 
                    'variants': variants,
                    'export_timestamp': timestamp,
                    'export_path': str(filepath)
                }
                json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        elif format_type == 'csv':
            # Flatten results für CSV
            rows = []
            for variant_id, data_result in results.items():
                variant = next((v for v in variants if v['id'] == variant_id), {})
                for div_type in ['classic', 'hidden']:
                    for div in data_result.get(div_type, []):
                        rows.append({
                            'variant': variant.get('name', ''),
                            'type': div_type,
                            'date': div.get('date'),
                            'low': div.get('low'),
                            'high': div.get('high'),
                            'close': div.get('close'),
                            'rsi': div.get('rsi'),
                            'macd': div.get('macd'),
                            'strength_score': div.get('validation', {}).get('strength_score', 0)
                        })
            
            if rows:
                df = pd.DataFrame(rows)
                df.to_csv(filepath, index=False, encoding='utf-8')
            else:
                # Leere CSV mit Headern
                empty_df = pd.DataFrame(columns=['variant', 'type', 'date', 'low', 'high', 'close', 'rsi', 'macd', 'strength_score'])
                empty_df.to_csv(filepath, index=False, encoding='utf-8')
        
        print(f"✅ Export erfolgreich: {filepath}")
        
        return jsonify({
            'success': True,
            'filename': filename,
            'path': str(filepath),
            'absolute_path': str(filepath.absolute())
        })
        
    except Exception as e:
        print(f"❌ Export-Fehler: {e}")
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
    print(f"🚀 Server läuft auf: http://localhost:{SERVER_CONFIG['port']}")
    print(f"📁 Export-Pfad: {PATHS['exports']}")
    print("   Zum Beenden: Strg+C")
    
    # Nur gültige Flask app.run() Parameter verwenden
    run_config = {
        'host': SERVER_CONFIG['host'],
        'port': SERVER_CONFIG['port'],
        'debug': SERVER_CONFIG['debug']
    }
    
    app.run(**run_config)
