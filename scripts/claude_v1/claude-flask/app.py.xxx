"""
Flask API Server für Bullish Divergence Analyzer
Verbindet HTML-Frontend mit Python-Analysemodulen
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
import json
import os
import tempfile
from pathlib import Path
import logging

# Import der existierenden Module (unverändert)
try:
    from Initialize_RSI_EMA_MACD import Initialize_RSI_EMA_MACD
    from Local_Maximas_Minimas import Local_Max_Min
    from CBullDivg_Analysis_vectorized import CBullDivg_analysis
except ImportError as e:
    print(f"Fehler beim Import: {e}")
    print("Stelle sicher, dass alle Module im gleichen Verzeichnis sind!")

# Flask App initialisieren
app = Flask(__name__, static_folder='static')
CORS(app)  # Für Cross-Origin Requests vom Browser

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Temporärer Speicher für hochgeladene Dateien
UPLOAD_FOLDER = tempfile.gettempdir()
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100 MB max

@app.route('/')
def index():
    """Serve HTML Interface"""
    return send_from_directory('static', 'index.html')

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """
    Datei-Upload Endpoint
    Akzeptiert CSV oder Parquet Dateien
    """
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'Keine Datei gefunden'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'Keine Datei ausgewählt'}), 400
        
        # Datei temporär speichern
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Datei laden
        if filename.endswith('.csv'):
            df = pd.read_csv(filepath)
        elif filename.endswith('.parquet'):
            df = pd.read_parquet(filepath)
        else:
            return jsonify({'error': 'Nur CSV und Parquet unterstützt'}), 400
        
        # Basis-Validierung
        required_cols = ['date', 'open', 'high', 'low', 'close']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            return jsonify({'error': f'Fehlende Spalten: {missing}'}), 400
        
        # Speichere DataFrame-Info in Session (vereinfacht)
        session_id = str(hash(filename))
        df.to_parquet(f"{UPLOAD_FOLDER}/session_{session_id}.parquet")
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'rows': len(df),
            'columns': list(df.columns),
            'preview': df.head(5).to_dict('records')
        })
        
    except Exception as e:
        logger.error(f"Upload-Fehler: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/analyze', methods=['POST'])
def analyze():
    """
    Hauptanalyse-Endpoint
    Führt die komplette Divergenz-Analyse durch
    """
    try:
        data = request.json
        session_id = data.get('session_id')
        variants = data.get('variants', [])
        
        if not session_id:
            return jsonify({'error': 'Keine Session ID'}), 400
        
        # Lade gespeicherte Daten
        df_path = f"{UPLOAD_FOLDER}/session_{session_id}.parquet"
        if not os.path.exists(df_path):
            return jsonify({'error': 'Session abgelaufen'}), 404
        
        df = pd.read_parquet(df_path)
        
        # Schritt 1: Technische Indikatoren berechnen
        logger.info("Berechne Indikatoren...")
        df = Initialize_RSI_EMA_MACD(df)
        
        # Schritt 2: Lokale Extrema finden
        logger.info("Finde lokale Extrema...")
        Local_Max_Min(df)
        
        # Schritt 3: Für jede Variante Divergenzen analysieren
        results = {}
        for variant in variants:
            logger.info(f"Analysiere Variante: {variant['name']}")
            
            # Kopie für diese Variante
            df_variant = df.copy()
            
            # Divergenz-Analyse mit Varianten-Parametern
            CBullDivg_analysis(
                df_variant,
                variant['window'],
                variant['candleTol'],
                variant['macdTol']
            )
            
            # Ergebnisse sammeln
            classic_divs = []
            hidden_divs = []
            
            for i in range(len(df_variant)):
                if df_variant.iloc[i].get('CBullD_gen', 0) == 1:
                    classic_divs.append({
                        'index': i,
                        'date': str(df_variant.iloc[i]['date']),
                        'low': float(df_variant.iloc[i]['low']),
                        'rsi': float(df_variant.iloc[i].get('RSI', 0)),
                        'macd': float(df_variant.iloc[i].get('macd_histogram', 0))
                    })
                
                if df_variant.iloc[i].get('CBullD_neg_MACD', 0) == 1:
                    hidden_divs.append({
                        'index': i,
                        'date': str(df_variant.iloc[i]['date']),
                        'low': float(df_variant.iloc[i]['low']),
                        'rsi': float(df_variant.iloc[i].get('RSI', 0)),
                        'macd': float(df_variant.iloc[i].get('macd_histogram', 0))
                    })
            
            results[variant['id']] = {
                'classic': classic_divs,
                'hidden': hidden_divs,
                'total': len(classic_divs) + len(hidden_divs)
            }
        
        # Basis-Daten für Chart vorbereiten
        chart_data = {
            'dates': df['date'].astype(str).tolist(),
            'open': df['open'].tolist(),
            'high': df['high'].tolist(),
            'low': df['low'].tolist(),
            'close': df['close'].tolist(),
            'rsi': df.get('RSI', pd.Series()).fillna(0).tolist(),
            'macd_histogram': df.get('macd_histogram', pd.Series()).fillna(0).tolist(),
            'volume': df.get('volume', pd.Series(np.random.randint(100000, 1000000, len(df)))).tolist()
        }
        
        return jsonify({
            'success': True,
            'chartData': chart_data,
            'results': results
        })
        
    except Exception as e:
        logger.error(f"Analyse-Fehler: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health():
    """Health Check Endpoint"""
    return jsonify({'status': 'ok', 'modules': {
        'Initialize_RSI_EMA_MACD': 'Initialize_RSI_EMA_MACD' in globals(),
        'Local_Max_Min': 'Local_Max_Min' in globals(),
        'CBullDivg_analysis': 'CBullDivg_analysis' in globals()
    }})

if __name__ == '__main__':
    # Erstelle static Ordner falls nicht vorhanden
    os.makedirs('static', exist_ok=True)
    
    print("\n" + "="*50)
    print("🚀 Bullish Divergence Analyzer Server")
    print("="*50)
    print("Server läuft auf: http://localhost:5000")
    print("Öffne diese URL im Browser!")
    print("="*50 + "\n")
    
    app.run(debug=True, port=5000)