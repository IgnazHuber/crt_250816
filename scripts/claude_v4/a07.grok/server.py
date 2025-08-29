#!/usr/bin/env python3
"""
Enhanced Bullish Divergence Analyzer Server - With Full Analysis Support
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

# Import checks for required modules
logger.info("🔍 Checking modules...")
modules_ok = True

try:
    from Initialize_RSI_EMA_MACD import Initialize_RSI_EMA_MACD
    logger.info("✅ Initialize_RSI_EMA_MACD.py found")
except ImportError:
    logger.error("❌ Initialize_RSI_EMA_MACD.py MISSING!")
    modules_ok = False
    Initialize_RSI_EMA_MACD = None

try:
    from Local_Maximas_Minimas import Local_Max_Min
    logger.info("✅ Local_Maximas_Minimas.py found")
except ImportError:
    logger.error("❌ Local_Maximas_Minimas.py MISSING!")
    modules_ok = False
    Local_Max_Min = None

try:
    from CBullDivg_Analysis_vectorized import CBullDivg_analysis
    logger.info("✅ CBullDivg_Analysis_vectorized.py found")
except ImportError:
    logger.error("❌ CBullDivg_Analysis_vectorized.py MISSING!")
    modules_ok = False
    CBullDivg_analysis = None

try:
    from CBullDivg_x2_analysis_vectorized import BullDivg_x2_analysis
    logger.info("✅ CBullDivg_x2_analysis_vectorized.py found")
except ImportError:
    logger.error("❌ CBullDivg_x2_analysis_vectorized.py MISSING!")
    modules_ok = False
    BullDivg_x2_analysis = None

try:
    from HBearDivg_analysis_vectorized import HBearDivg_analysis
    logger.info("✅ HBearDivg_analysis_vectorized.py found")
except ImportError:
    logger.error("❌ HBearDivg_analysis_vectorized.py MISSING!")
    modules_ok = False
    HBearDivg_analysis = None

try:
    from HBullDivg_analysis_vectorized import HBullDivg_analysis
    logger.info("✅ HBullDivg_analysis_vectorized.py found")
except ImportError:
    logger.error("❌ HBullDivg_analysis_vectorized.py MISSING!")
    modules_ok = False
    HBullDivg_analysis = None

if not modules_ok:
    logger.warning("\n⚠️ IMPORTANT: Copy missing Python modules to this folder!")

app = Flask(__name__, static_folder='.')
CORS(app)
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()

# Session storage
sessions = {}

@app.route('/')
def index():
    logger.info("📄 Serving index.html")
    return send_from_directory('.', 'index.html')

@app.route('/static/<path:path>')
def serve_static(path):
    logger.info(f"📄 Serving static file: {path}")
    return send_from_directory('.', path)

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
    if not BullDivg_x2_analysis:
        missing.append("CBullDivg_x2_analysis_vectorized")
    if not HBearDivg_analysis:
        missing.append("HBearDivg_analysis_vectorized")
    if not HBullDivg_analysis:
        missing.append("HBullDivg_analysis_vectorized")
    
    if missing:
        logger.error(f"❌ Missing modules: {', '.join(missing)}")
        return jsonify({"status": "error", "message": f"Missing modules: {', '.join(missing)}"})
    logger.info("✅ All modules loaded")
    return jsonify({"status": "ok", "message": "All modules loaded"})

@app.route('/api/upload', methods=['POST'])
def upload_file():
    logger.info("📤 File upload requested")
    try:
        file = request.files.get('file')
        if not file:
            logger.error("❌ No file uploaded")
            return jsonify({"success": False, "error": "No file uploaded"})

        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in ['.csv', '.parquet']:
            logger.error(f"❌ Invalid file type: {file_ext}")
            return jsonify({"success": False, "error": "Invalid file type"})

        logger.info(f"📂 Processing file: {file.filename}")
        
        # Save uploaded file temporarily
        temp_path = os.path.join(tempfile.gettempdir(), f"upload_{file.filename}")
        file.save(temp_path)
        
        # Load with pandas
        if file_ext == '.csv':
            df = pd.read_csv(temp_path, low_memory=False)
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
            return jsonify({"success": False, "error": f"Missing columns: {missing}. Load OHLC data, not summaries."})

        # Ensure numeric data
        for col in ['open', 'high', 'low', 'close']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        if df[['open', 'high', 'low', 'close']].isna().any().any():
            logger.error("❌ Invalid numeric data in OHLC columns")
            return jsonify({"success": False, "error": "Invalid numeric data in OHLC columns"})

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
            return jsonify({"success": False, "error": "Invalid session"})

        if not variants:
            logger.error("❌ No variants provided")
            return jsonify({"success": False, "error": "No variants provided"})

        # Get data
        df = sessions[session_id].copy()
        logger.info(f"📈 Data loaded: {len(df)} rows")

        # Initialize indicators
        logger.info("📊 Initializing indicators...")
        if not Initialize_RSI_EMA_MACD:
            logger.error("❌ Initialize_RSI_EMA_MACD not available")
            return jsonify({"success": False, "error": "Initialize_RSI_EMA_MACD module missing"})
        Initialize_RSI_EMA_MACD(df)
        
        if not Local_Max_Min:
            logger.error("❌ Local_Maximas_Minimas not available")
            return jsonify({"success": False, "error": "Local_Maximas_Minimas module missing"})
        Local_Max_Min(df)

        # Identify standard variant (first or name='Standard')
        standard_variant = next((v for v in variants if v['name'].lower() == 'standard'), variants[0] if variants else None)
        standard_id = standard_variant['id'] if standard_variant else None
        standard_dates = set()

        results = {}
        div_counter = 1  # Global counter for divergence numbers
        
        for variant in variants:
            logger.info(f"   Analyzing: {variant['name']}")
            df_var = df.copy()
            
            # Run all 4 analysis types based on variant settings
            variant_results = {
                'CBullDivg_classic': [],
                'CBullDivg_hidden': [],
                'CBullDivg_x2_classic': [],
                'CBullDivg_x2_hidden': [],
                'HBearDivg_classic': [],
                'HBearDivg_hidden': [],
                'HBullDivg_classic': [],
                'HBullDivg_hidden': []
            }
            
            # CBullDivg Analysis
            if CBullDivg_analysis:
                CBullDivg_analysis(df_var, variant['window'], variant['candleTol'], variant['macdTol'])
                
                for i in range(len(df_var)):
                    row = df_var.iloc[i]
                    
                    # CBullDivg Classic
                    if row.get('CBullD_gen', 0) == 1:
                        date_str = str(row['date'])
                        div_data = {
                            'date': date_str,
                            'low': float(row['low']) if not pd.isna(row['low']) else 0,
                            'rsi': float(row.get('RSI', 0)) if not pd.isna(row.get('RSI', 0)) else 0,
                            'macd': float(row.get('macd_histogram', 0)) if not pd.isna(row.get('macd_histogram', 0)) else 0,
                            'type': 'classic',
                            'analysis': 'CBullDivg',
                            'strength': min(1.0, abs(float(row.get('macd_histogram', 0))) / variant['macdTol']) if variant['macdTol'] > 0 else 0.5,
                            'is_new': date_str not in standard_dates if variant['id'] != standard_id else False,
                            'div_id': div_counter,
                            'window': variant['window']
                        }
                        variant_results['CBullDivg_classic'].append(div_data)
                        div_counter += 1
                    
                    # CBullDivg Hidden
                    if row.get('CBullD_neg_MACD', 0) == 1:
                        date_str = str(row['date'])
                        div_data = {
                            'date': date_str,
                            'low': float(row['low']) if not pd.isna(row['low']) else 0,
                            'rsi': float(row.get('RSI', 0)) if not pd.isna(row.get('RSI', 0)) else 0,
                            'macd': float(row.get('macd_histogram', 0)) if not pd.isna(row.get('macd_histogram', 0)) else 0,
                            'type': 'hidden',
                            'analysis': 'CBullDivg',
                            'strength': min(1.0, abs(float(row.get('macd_histogram', 0))) / variant['macdTol']) if variant['macdTol'] > 0 else 0.5,
                            'is_new': date_str not in standard_dates if variant['id'] != standard_id else False,
                            'div_id': div_counter,
                            'window': variant['window']
                        }
                        variant_results['CBullDivg_hidden'].append(div_data)
                        div_counter += 1
            
            # CBullDivg_x2 Analysis 
            if BullDivg_x2_analysis:
                df_var_x2 = df_var.copy()
                BullDivg_x2_analysis(df_var_x2, variant['window'], variant['candleTol'], variant['macdTol'])
                
                for i in range(len(df_var_x2)):
                    row = df_var_x2.iloc[i]
                    if row.get('CBullD_x2_gen', 0) == 1:
                        date_str = str(row['date'])
                        div_data = {
                            'date': date_str,
                            'low': float(row['low']) if not pd.isna(row['low']) else 0,
                            'rsi': float(row.get('RSI', 0)) if not pd.isna(row.get('RSI', 0)) else 0,
                            'macd': float(row.get('macd_histogram', 0)) if not pd.isna(row.get('macd_histogram', 0)) else 0,
                            'type': 'classic',
                            'analysis': 'CBullDivg_x2',
                            'strength': min(1.0, abs(float(row.get('macd_histogram', 0))) / variant['macdTol']) if variant['macdTol'] > 0 else 0.5,
                            'is_new': date_str not in standard_dates if variant['id'] != standard_id else False,
                            'div_id': div_counter,
                            'window': variant['window']
                        }
                        variant_results['CBullDivg_x2_classic'].append(div_data)
                        div_counter += 1
            
            # HBearDivg Analysis
            if HBearDivg_analysis:
                df_var_hbear = df_var.copy()
                HBearDivg_analysis(df_var_hbear, variant['window'], variant['candleTol'], variant['macdTol'])
                
                for i in range(len(df_var_hbear)):
                    row = df_var_hbear.iloc[i]
                    if row.get('HBearD_gen', 0) == 1:
                        date_str = str(row['date'])
                        div_data = {
                            'date': date_str,
                            'low': float(row['high']) if not pd.isna(row['high']) else 0,  # Use high for bearish
                            'rsi': float(row.get('RSI', 0)) if not pd.isna(row.get('RSI', 0)) else 0,
                            'macd': float(row.get('macd_histogram', 0)) if not pd.isna(row.get('macd_histogram', 0)) else 0,
                            'type': 'classic',
                            'analysis': 'HBearDivg',
                            'strength': min(1.0, abs(float(row.get('macd_histogram', 0))) / variant['macdTol']) if variant['macdTol'] > 0 else 0.5,
                            'is_new': date_str not in standard_dates if variant['id'] != standard_id else False,
                            'div_id': div_counter,
                            'window': variant['window']
                        }
                        variant_results['HBearDivg_classic'].append(div_data)
                        div_counter += 1
            
            # HBullDivg Analysis
            if HBullDivg_analysis:
                df_var_hbull = df_var.copy()
                HBullDivg_analysis(df_var_hbull, variant['window'], variant['candleTol'], variant['macdTol'])
                
                for i in range(len(df_var_hbull)):
                    row = df_var_hbull.iloc[i]
                    
                    # HBullDivg Classic
                    if row.get('HBullD_gen', 0) == 1:
                        date_str = str(row['date'])
                        div_data = {
                            'date': date_str,
                            'low': float(row['low']) if not pd.isna(row['low']) else 0,
                            'rsi': float(row.get('RSI', 0)) if not pd.isna(row.get('RSI', 0)) else 0,
                            'macd': float(row.get('macd_histogram', 0)) if not pd.isna(row.get('macd_histogram', 0)) else 0,
                            'type': 'classic',
                            'analysis': 'HBullDivg',
                            'strength': min(1.0, abs(float(row.get('macd_histogram', 0))) / variant['macdTol']) if variant['macdTol'] > 0 else 0.5,
                            'is_new': date_str not in standard_dates if variant['id'] != standard_id else False,
                            'div_id': div_counter,
                            'window': variant['window']
                        }
                        variant_results['HBullDivg_classic'].append(div_data)
                        div_counter += 1
                    
                    # HBullDivg Hidden
                    if row.get('HBullD_neg_MACD', 0) == 1:
                        date_str = str(row['date'])
                        div_data = {
                            'date': date_str,
                            'low': float(row['low']) if not pd.isna(row['low']) else 0,
                            'rsi': float(row.get('RSI', 0)) if not pd.isna(row.get('RSI', 0)) else 0,
                            'macd': float(row.get('macd_histogram', 0)) if not pd.isna(row.get('macd_histogram', 0)) else 0,
                            'type': 'hidden',
                            'analysis': 'HBullDivg',
                            'strength': min(1.0, abs(float(row.get('macd_histogram', 0))) / variant['macdTol']) if variant['macdTol'] > 0 else 0.5,
                            'is_new': date_str not in standard_dates if variant['id'] != standard_id else False,
                            'div_id': div_counter,
                            'window': variant['window']
                        }
                        variant_results['HBullDivg_hidden'].append(div_data)
                        div_counter += 1

            # Store standard variant dates for comparison
            if variant['id'] == standard_id:
                all_divs = []
                for key, divs in variant_results.items():
                    all_divs.extend([d['date'] for d in divs])
                standard_dates = set(all_divs)

            # Group results by classic/hidden for frontend compatibility
            classic = []
            hidden = []
            
            for analysis_type in ['CBullDivg', 'CBullDivg_x2', 'HBearDivg', 'HBullDivg']:
                classic.extend(variant_results.get(f'{analysis_type}_classic', []))
                hidden.extend(variant_results.get(f'{analysis_type}_hidden', []))
            
            results[variant['id']] = {
                'classic': classic,
                'hidden': hidden,
                'total': len(classic) + len(hidden),
                'detailed': variant_results  # Keep detailed breakdown for new features
            }

        # Prepare chart data with NaN handling
        chart_data = {
            'dates': df['date'].astype(str).tolist(),
            'open': df['open'].fillna(0).tolist(),
            'high': df['high'].fillna(0).tolist(),
            'low': df['low'].fillna(0).tolist(),
            'close': df['close'].fillna(0).tolist(),
            'rsi': df.get('RSI', pd.Series()).fillna(0).tolist(),
            'macd_histogram': df.get('macd_histogram', pd.Series()).fillna(0).tolist(),
            'ema20': df.get('EMA_20', pd.Series()).fillna(0).tolist() if 'EMA_20' in df else None,
            'ema50': df.get('EMA_50', pd.Series()).fillna(0).tolist() if 'EMA_50' in df else None,
            'ema100': df.get('EMA_100', pd.Series()).fillna(0).tolist() if 'EMA_100' in df else None,
            'ema200': df.get('EMA_200', pd.Series()).fillna(0).tolist() if 'EMA_200' in df else None
        }

        # Validate chart data
        if not chart_data['dates'] or not chart_data['open']:
            logger.error("❌ Empty chart data")
            return jsonify({"success": False, "error": "Empty chart data"})

        logger.info(f"✅ Analysis completed, returning {len(chart_data['dates'])} data points")
        return jsonify({
            'success': True,
            'chartData': chart_data,
            'results': results
        })
    except Exception as e:
        logger.error(f"❌ Analysis error: {e}")
        return jsonify({"success": False, "error": str(e)})

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    print("\n" + "="*60)
    print("ENHANCED DIVERGENCE ANALYZER SERVER")
    print("="*60)
    print("Features:")
    print("   • All 4 analysis types (CBullDivg, CBullDivg_x2, HBearDivg, HBullDivg)")
    print("   • CSV and Parquet file support")
    print("   • Interactive web interface with zoom/pan")
    print("   • Logical marker grouping and toggling")
    print("   • Transparent yellow/blue comparison circles")
    print("   • Advanced statistics and export")
    print("="*60)
    print("Server running on: http://localhost:5000")
    print("   Browser opens automatically in 3 seconds...")
    print("   To stop: Ctrl+C")
    print("="*60 + "\n")

    def open_browser():
        time.sleep(3)
        webbrowser.open('http://localhost:5000')

    threading.Thread(target=open_browser, daemon=True).start()
    app.run(debug=True, host='0.0.0.0', port=5000)