#!/usr/bin/env python3
"""
WORKING BULLISH DIVERGENCE ANALYZER - Copied from all_in_one_analyzer.py
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
import os
import tempfile
import sys

# ========== IMPORT MODULES ==========
try:
    from Initialize_RSI_EMA_MACD import Initialize_RSI_EMA_MACD
    print("✅ Initialize_RSI_EMA_MACD loaded")
except ImportError:
    print("❌ Initialize_RSI_EMA_MACD MISSING!")
    Initialize_RSI_EMA_MACD = None

try:
    from Local_Maximas_Minimas import Local_Max_Min
    print("✅ Local_Maximas_Minimas loaded")
except ImportError:
    print("❌ Local_Maximas_Minimas MISSING!")
    Local_Max_Min = None

try:
    from CBullDivg_Analysis_vectorized import CBullDivg_analysis
    print("✅ CBullDivg_Analysis_vectorized loaded")
except ImportError:
    print("❌ CBullDivg_Analysis_vectorized MISSING!")
    CBullDivg_analysis = None

app = Flask(__name__, static_folder='static')
CORS(app)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Simple session storage
sessions = {}

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory(app.static_folder, path)

@app.route('/api/health', methods=['GET'])
def health_check():
    missing = []
    if not Initialize_RSI_EMA_MACD: missing.append("Initialize_RSI_EMA_MACD")
    if not Local_Max_Min: missing.append("Local_Maximas_Minimas") 
    if not CBullDivg_analysis: missing.append("CBullDivg_Analysis_vectorized")
    
    if missing:
        return jsonify({"status": "error", "message": f"Missing modules: {', '.join(missing)}"})
    return jsonify({"status": "ok", "message": "All modules loaded"})

@app.route('/api/upload', methods=['POST'])
def upload_file():
    try:
        file = request.files.get('file')
        if not file:
            return jsonify({"success": False, "error": "No file uploaded"})
        
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in ['.csv', '.parquet']:
            return jsonify({"success": False, "error": "Invalid file type"})
        
        if file_ext == '.csv':
            df = pd.read_csv(file)
        else:
            df = pd.read_parquet(file)
        
        # Validate required columns
        required = ['date', 'open', 'high', 'low', 'close']
        missing = [col for col in required if col not in df.columns]
        if missing:
            return jsonify({"success": False, "error": f"Missing columns: {missing}. Load OHLC data, not summary files."})
        
        session_id = str(abs(hash(file.filename + str(len(df)))))
        sessions[session_id] = df
        
        return jsonify({
            "success": True,
            "session_id": session_id,
            "info": {"rows": len(df), "columns": list(df.columns)}
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        variants = data.get('variants', [])
        
        if not session_id or session_id not in sessions:
            return jsonify({"success": False, "error": "Invalid session"})
        
        if not variants:
            return jsonify({"success": False, "error": "No variants provided"})
        
        # Get data
        df = sessions[session_id].copy()
        
        # EXACT COPY FROM WORKING VERSION
        print("📊 Starting analysis...")
        if Initialize_RSI_EMA_MACD:
            df = Initialize_RSI_EMA_MACD(df)
        if Local_Max_Min:
            Local_Max_Min(df)
        
        results = {}
        for variant in variants:
            print(f"   Analyzing: {variant['name']}")
            df_var = df.copy()
            if CBullDivg_analysis:
                CBullDivg_analysis(df_var, variant['window'], variant['candleTol'], variant['macdTol'])
            
            classic = []
            hidden = []
            for i in range(len(df_var)):
                row = df_var.iloc[i]
                if row.get('CBullD_gen', 0) == 1:
                    classic.append({
                        'date': str(row['date']),
                        'low': float(row['low']),
                        'rsi': float(row.get('RSI', 0)),
                        'macd': float(row.get('macd_histogram', 0))
                    })
                if row.get('CBullD_neg_MACD', 0) == 1:
                    hidden.append({
                        'date': str(row['date']),
                        'low': float(row['low']),
                        'rsi': float(row.get('RSI', 0)),
                        'macd': float(row.get('macd_histogram', 0))
                    })
            
            results[variant['id']] = {
                'classic': classic,
                'hidden': hidden,
                'total': len(classic) + len(hidden)
            }
        
        # EXACT CHART DATA FROM WORKING VERSION
        chart_data = {
            'dates': df['date'].astype(str).tolist(),
            'open': df['open'].tolist(),
            'high': df['high'].tolist(),
            'low': df['low'].tolist(),
            'close': df['close'].tolist(),
            'rsi': df.get('RSI', pd.Series()).fillna(0).tolist(),
            'macd_histogram': df.get('macd_histogram', pd.Series()).fillna(0).tolist(),
            'ema20': df.get('EMA_20', pd.Series()).fillna(0).tolist() if 'EMA_20' in df else None,
            'ema50': df.get('EMA_50', pd.Series()).fillna(0).tolist() if 'EMA_50' in df else None,
            'ema100': df.get('EMA_100', pd.Series()).fillna(0).tolist() if 'EMA_100' in df else None,
            'ema200': df.get('EMA_200', pd.Series()).fillna(0).tolist() if 'EMA_200' in df else None
        }
        
        return jsonify({
            'success': True,
            'chartData': chart_data,
            'results': results
        })
    except Exception as e:
        print(f"Analysis error: {e}")
        return jsonify({"success": False, "error": str(e)})

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    print("🚀 WORKING SERVER STARTING...")
    print("📍 Server: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)