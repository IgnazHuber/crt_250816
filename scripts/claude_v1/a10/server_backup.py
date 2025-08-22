from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
import os
import logging
import json
from analysis_engine import AnalysisEngine

class SafeJSONEncoder(json.JSONEncoder):
    def encode(self, obj):
        if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
            return 'null'
        return super().encode(obj)

app = Flask(__name__, static_folder='static')
app.json_encoder = SafeJSONEncoder
CORS(app)

# Logging konfigurieren
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialisiere Analysis Engine mit den erforderlichen Modulen
required_modules = [
    "Initialize_RSI_EMA_MACD",
    "Local_Maximas_Minimas",
    "CBullDivg_Analysis_vectorized",
    "DivergenceArrows"
]
engine = AnalysisEngine(required_modules)

@app.route('/api/health', methods=['GET'])
def health_check():
    try:
        missing_modules = [mod for mod, loaded in engine.modules if loaded is None]
        if not missing_modules:
            return jsonify({"status": "ok", "message": "Alle Module geladen"})
        else:
            return jsonify({
                "status": "error",
                "message": f"Fehlende Module: {', '.join(missing_modules)}"
            })
    except Exception as e:
        logger.error(f"Health Check Fehler: {e}")
        return jsonify({"status": "error", "message": f"Health Check Fehler: {e}"})

@app.route('/', methods=['GET'])
def index():
    try:
        return send_from_directory(app.static_folder, 'index.html')
    except Exception as e:
        logger.error(f"Fehler beim Bereitstellen von index.html: {e}")
        return jsonify({"status": "error", "message": f"Fehler beim Laden der Seite: {e}"}), 500

@app.route('/static/<path:path>')
def serve_static(path):
    try:
        return send_from_directory(app.static_folder, path)
    except Exception as e:
        logger.error(f"Fehler beim Bereitstellen statischer Datei {path}: {e}")
        return jsonify({"status": "error", "message": f"Datei nicht gefunden: {path}"}), 404

@app.route('/api/upload', methods=['POST'])
def upload_file():
    try:
        file = request.files.get('file')
        if not file:
            return jsonify({"success": False, "error": "Keine Datei hochgeladen"})
        
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in ['.csv', '.parquet']:
            return jsonify({"success": False, "error": "Ungültiger Dateityp"})
        
        if file_ext == '.csv':
            df = pd.read_csv(file)
        else:
            df = pd.read_parquet(file)
        
        session_id = engine.load_data(df)
        if session_id:
            return jsonify({
                "success": True,
                "session_id": session_id,
                "info": {"rows": len(df), "columns": list(df.columns)}
            })
        else:
            return jsonify({"success": False, "error": "Fehler beim Laden der Daten"})
    except Exception as e:
        logger.error(f"Upload Fehler: {e}")
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        variants = data.get('variants', [])
        date_range = data.get('date_range')
        
        if not session_id or not variants:
            return jsonify({"success": False, "error": "Session-ID oder Varianten fehlen"})
        
        logger.info(f"Starting analysis for session {session_id} with {len(variants)} variants")
        logger.info(f"Received variants: {variants}")
        result = engine.analyze_dataframe(session_id, variants, date_range)
        
        # Ensure the result is JSON-safe
        if isinstance(result, dict):
            logger.info("Analysis completed successfully")
            return jsonify(result)
        else:
            logger.error(f"Analysis returned invalid result type: {type(result)}")
            return jsonify({"success": False, "error": "Analysis returned invalid result"})
            
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {e}")
        return jsonify({"success": False, "error": "Invalid JSON in request"})
    except Exception as e:
        logger.error(f"Analyse Fehler: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/export', methods=['POST'])
def export():
    try:
        data = request.get_json()
        format_type = data.get('format')
        results = data.get('results')
        variants = data.get('variants')
        
        if format_type == 'csv':
            output_path = f"exports/results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
            os.makedirs('exports', exist_ok=True)
            df = pd.DataFrame()
            for variant_id, result in results.items():
                for div_type in ['classic', 'hidden']:
                    for div in result.get(div_type, []):
                        df = pd.concat([df, pd.DataFrame([{
                            'variant': variants[int(variant_id) - 1]['name'],
                            'div_type': div_type,
                            'date': div['date'],
                            'low': div['low'],
                            'rsi': div['rsi'],
                            'macd': div['macd'],
                            'strength': div['validation']['strength_score']
                        }])], ignore_index=True)
            df.to_csv(output_path, index=False)
            return jsonify({"success": True, "csv_path": output_path})
        elif format_type == 'json':
            output_path = f"exports/results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"
            os.makedirs('exports', exist_ok=True)
            with open(output_path, 'w') as f:
                import json
                json.dump({"results": results, "variants": variants}, f)
            return jsonify({"success": True, "json_path": output_path})
        else:
            return jsonify({"success": False, "error": "Ungültiges Format"})
    except Exception as e:
        logger.error(f"Export Fehler: {e}")
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/save_config', methods=['POST'])
def save_config():
    try:
        data = request.get_json()
        name = data.get('name')
        variants = data.get('variants')
        filename = f"configs/{name}.json"
        os.makedirs('configs', exist_ok=True)
        with open(filename, 'w') as f:
            import json
            json.dump({"variants": variants}, f)
        return jsonify({"success": True, "filename": filename})
    except Exception as e:
        logger.error(f"Config Save Fehler: {e}")
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/list_configs', methods=['GET'])
def list_configs():
    try:
        configs = [f for f in os.listdir('configs') if f.endswith('.json')]
        return jsonify(configs)
    except Exception as e:
        logger.error(f"List Configs Fehler: {e}")
        return jsonify([])

@app.route('/api/load_config/<name>', methods=['GET'])
def load_config(name):
    try:
        filename = f"configs/{name}.json"
        with open(filename, 'r') as f:
            import json
            data = json.load(f)
        return jsonify({"success": True, "variants": data['variants']})
    except Exception as e:
        logger.error(f"Config Load Fehler: {e}")
        return jsonify({"success": False, "error": str(e)})

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('exports', exist_ok=True)
    os.makedirs('configs', exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5000)