#!/usr/bin/env python3
"""
Bullish Divergence Analyzer Server - Enhanced with debugging
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
import os
import tempfile
import webbrowser
import threading
import time
import sys
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import optimized modules
try:
    from data_processor_v2 import DataProcessorV2
    logger.info("✅ data_processor_v2.py gefunden - Fast loading aktiviert")
    fast_loader = DataProcessorV2()
except ImportError:
    logger.warning("⚠️ data_processor_v2.py nicht gefunden - Fallback auf Pandas")
    fast_loader = None

try:
    from fast_analyzer import analyze_ultra_fast
    logger.info("✅ fast_analyzer.py gefunden - Ultra-fast analysis aktiviert")
    ultra_fast_analysis = True
except ImportError:
    logger.warning("⚠️ fast_analyzer.py nicht gefunden - Using standard analysis")
    ultra_fast_analysis = False

# Import checks for required modules
logger.info("🔍 Prüfe Module...")
modules_ok = True

try:
    from Initialize_RSI_EMA_MACD import Initialize_RSI_EMA_MACD
    logger.info("✅ Initialize_RSI_EMA_MACD.py gefunden")
except ImportError:
    logger.error("❌ Initialize_RSI_EMA_MACD.py FEHLT!")
    modules_ok = False
    Initialize_RSI_EMA_MACD = None

try:
    from Local_Maximas_Minimas import Local_Max_Min
    logger.info("✅ Local_Maximas_Minimas.py gefunden")
except ImportError:
    logger.error("❌ Local_Maximas_Minimas.py FEHLT!")
    modules_ok = False
    Local_Max_Min = None

try:
    from CBullDivg_Analysis_vectorized import CBullDivg_analysis
    logger.info("✅ CBullDivg_Analysis_vectorized.py gefunden")
except ImportError:
    logger.error("❌ CBullDivg_Analysis_vectorized.py FEHLT!")
    modules_ok = False
    CBullDivg_analysis = None

if not modules_ok:
    logger.warning("\n⚠️ WICHTIG: Kopiere die fehlenden Python-Module in diesen Ordner!")
    response = input("Trotzdem starten? (j/n): ")
    if response.lower() != 'j':
        sys.exit(1)

app = Flask(__name__, static_folder='static')
CORS(app)
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()

# Session storage
sessions = {}

@app.route('/')
def index():
    logger.info("📄 Serving index.html")
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/static/<path:path>')
def serve_static(path):
    logger.info(f"📄 Serving static file: {path}")
    return send_from_directory(app.static_folder, path)

@app.route('/api/health', methods=['GET'])
def health_check():
    logger.info("🔍 Health check requested")
    missing = []
    if not Initialize_RSI_EMA_MACD:
        missing.append("Initialize_RSI_EMA_MACD")
    if not Local_Max_Min:
        missing.append("Local_Maximas_Minimas")
    if not CBullDivg_analysis:
        missing.append("CBullDivg_Analysis_vectorized")
    
    if missing:
        logger.error(f"❌ Missing modules: {', '.join(missing)}")
        return jsonify({"status": "error", "message": f"Fehlende Module: {', '.join(missing)}"})
    logger.info("✅ All modules loaded")
    return jsonify({"status": "ok", "message": "Alle Module geladen"})

@app.route('/api/upload', methods=['POST'])
def upload_file():
    logger.info("📤 File upload requested")
    try:
        file = request.files.get('file')
        if not file:
            logger.error("❌ No file uploaded")
            return jsonify({"success": False, "error": "Keine Datei hochgeladen"})

        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in ['.csv', '.parquet']:
            logger.error(f"❌ Invalid file type: {file_ext}")
            return jsonify({"success": False, "error": "Ungültiger Dateityp"})

        logger.info(f"📂 Processing file: {file.filename} (using fast loader)")
        
        # Save uploaded file temporarily
        temp_path = os.path.join(tempfile.gettempdir(), f"upload_{file.filename}")
        file.save(temp_path)
        
        # Use optimized data processor if available
        if fast_loader:
            try:
                df = fast_loader.load_for_analysis(temp_path)
                logger.info(f"🚀 Fast loading successful: {len(df):,} rows")
            except Exception as e:
                logger.warning(f"⚠️ Fast loading failed: {e}. Using Pandas fallback")
                if file_ext == '.csv':
                    df = pd.read_csv(temp_path)
                else:
                    df = pd.read_parquet(temp_path)
        else:
            # Fallback to standard Pandas
            if file_ext == '.csv':
                df = pd.read_csv(temp_path)
            else:
                df = pd.read_parquet(temp_path)
        
        # Clean up temp file
        try:
            os.remove(temp_path)
        except:
            pass

        # Validate required columns
        required = ['date', 'open', 'high', 'low', 'close']
        missing = [col for col in required if col not in df.columns]
        if missing:
            logger.error(f"❌ Missing columns: {missing}")
            return jsonify({"success": False, "error": f"Fehlende Spalten: {missing}. Lade OHLC-Daten, keine Zusammenfassungen."})

        # Ensure numeric data
        for col in ['open', 'high', 'low', 'close']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        if df[['open', 'high', 'low', 'close']].isna().any().any():
            logger.error("❌ Invalid numeric data in OHLC columns")
            return jsonify({"success": False, "error": "Ungültige numerische Daten in OHLC-Spalten"})

        session_id = str(abs(hash(file.filename + str(len(df)))))
        sessions[session_id] = df
        logger.info(f"✅ File uploaded, session_id: {session_id}, rows: {len(df)}")

        return jsonify({
            "success": True,
            "session_id": session_id,
            "info": {"rows": len(df), "columns": list(df.columns)}
        })
    except Exception as e:
        logger.error(f"❌ Upload error: {e}")
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/analyze', methods=['POST'])
def analyze():
    logger.info("🚀 Analyze request received")
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        variants = data.get('variants', [])

        logger.info(f"📊 Processing session_id: {session_id}, variants: {len(variants)}")
        if not session_id or session_id not in sessions:
            logger.error("❌ Invalid session")
            return jsonify({"success": False, "error": "Ungültige Session"})

        if not variants:
            logger.error("❌ No variants provided")
            return jsonify({"success": False, "error": "Keine Varianten angegeben"})

        # Get data
        df = sessions[session_id].copy()
        logger.info(f"📈 Data loaded: {len(df)} rows")

        # Use ultra-fast analysis if available
        if ultra_fast_analysis:
            logger.info("🚀 Using ultra-fast analyzer...")
            try:
                result = analyze_ultra_fast(df, variants)
                logger.info(f"✅ Ultra-fast analysis completed: {len(result['chartData']['dates'])} chart points")
                return jsonify(result)
            except Exception as e:
                logger.warning(f"⚠️ Ultra-fast analysis failed: {e}. Falling back to standard analysis")
        
        # Fallback to standard analysis
        logger.info("📊 Using standard analysis pipeline...")
        if not Initialize_RSI_EMA_MACD:
            logger.error("❌ Initialize_RSI_EMA_MACD not available")
            return jsonify({"success": False, "error": "Initialize_RSI_EMA_MACD module missing"})
        df = Initialize_RSI_EMA_MACD(df)
        if df is None:
            logger.error("❌ Initialize_RSI_EMA_MACD returned None")
            return jsonify({"success": False, "error": "Initialize_RSI_EMA_MACD failed"})

        if not Local_Max_Min:
            logger.error("❌ Local_Maximas_Minimas not available")
            return jsonify({"success": False, "error": "Local_Maximas_Minimas module missing"})
        Local_Max_Min(df)

        # Standard analysis (simplified for fallback only)
        results = {}
        for variant in variants:
            logger.info(f"   Standard analysis: {variant['name']}")
            df_var = df.copy()
            if not CBullDivg_analysis:
                logger.error("❌ CBullDivg_Analysis_vectorized not available")
                return jsonify({"success": False, "error": "CBullDivg_Analysis_vectorized module missing"})
            CBullDivg_analysis(df_var, variant['window'], variant['candleTol'], variant['macdTol'])

            # Extract divergences (simplified)
            classic = []
            hidden = []
            for i in range(len(df_var)):
                row = df_var.iloc[i]
                if row.get('CBullD_gen', 0) == 1:
                    classic.append({
                        'date': str(row['date']), 'low': float(row['low']),
                        'rsi': float(row.get('RSI', 0)), 'macd': float(row.get('macd_histogram', 0)),
                        'type': 'classic', 'strength': 0.5, 'div_id': len(classic) + 1, 'window': variant['window']
                    })
                if row.get('CBullD_neg_MACD', 0) == 1:
                    hidden.append({
                        'date': str(row['date']), 'low': float(row['low']),
                        'rsi': float(row.get('RSI', 0)), 'macd': float(row.get('macd_histogram', 0)),
                        'type': 'hidden', 'strength': 0.5, 'div_id': len(hidden) + 1, 'window': variant['window']
                    })

            results[variant['id']] = {'classic': classic, 'hidden': hidden, 'total': len(classic) + len(hidden)}

        # Chart data (sampled for performance)
        sample_size = min(5000, len(df))
        df_sampled = df.tail(sample_size)
        
        chart_data = {
            'dates': df_sampled['date'].astype(str).tolist(),
            'open': df_sampled['open'].fillna(0).tolist(),
            'high': df_sampled['high'].fillna(0).tolist(),
            'low': df_sampled['low'].fillna(0).tolist(),
            'close': df_sampled['close'].fillna(0).tolist(),
            'rsi': df_sampled.get('RSI', pd.Series()).fillna(0).tolist(),
            'macd_histogram': df_sampled.get('macd_histogram', pd.Series()).fillna(0).tolist(),
            'ema20': df_sampled.get('EMA_20', pd.Series()).fillna(0).tolist() if 'EMA_20' in df_sampled else None,
            'ema50': df_sampled.get('EMA_50', pd.Series()).fillna(0).tolist() if 'EMA_50' in df_sampled else None,
            'ema100': df_sampled.get('EMA_100', pd.Series()).fillna(0).tolist() if 'EMA_100' in df_sampled else None,
            'ema200': df_sampled.get('EMA_200', pd.Series()).fillna(0).tolist() if 'EMA_200' in df_sampled else None
        }

        logger.info(f"✅ Standard analysis completed: {len(chart_data['dates'])} chart points")
        return jsonify({'success': True, 'chartData': chart_data, 'results': results})
    except Exception as e:
        logger.error(f"❌ Analysis error: {e}")
        return jsonify({"success": False, "error": str(e)})

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs('static', exist_ok=True)
    print("\n" + "="*60)
    print("🚀 BULLISH DIVERGENCE ANALYZER - Enhanced")
    print("="*60)
    print("✨ Features:")
    print("   • Vergleich zur Basis-Variante")
    print("   • Y-Achsen Zoom")
    print("   • Schwarzer Hintergrund")
    print("   • EMA 20, 50, 100, 200")
    print("   • Kräftige Marker-Farben")
    print("="*60)
    print("📍 Server läuft auf: http://localhost:5000")
    print("   Browser öffnet automatisch in 3 Sekunden...")
    print("   Zum Beenden: Strg+C")
    print("="*60 + "\n")

    def open_browser():
        time.sleep(3)
        webbrowser.open('http://localhost:5000')

    threading.Thread(target=open_browser, daemon=True).start()
    app.run(debug=True, host='0.0.0.0', port=5000)