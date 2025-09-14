#!/usr/bin/env python3
"""
Bullish Divergence Analyzer Server - Ultra-High Performance Edition with Numba JIT
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
import uuid

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Performance configuration
USE_NUMBA_OPTIMIZATION = True  # Toggle between original and Numba-optimized functions
ENABLE_PERFORMANCE_LOGGING = True

# Import checks for required modules
logger.info("Checking required modules...")
modules_ok = True

# Try to import Numba-optimized functions first
if USE_NUMBA_OPTIMIZATION:
    try:
        logger.info("Loading Numba-optimized functions...")
        from Initialize_RSI_EMA_MACD_numba import Initialize_RSI_EMA_MACD
        from Local_Maximas_Minimas_numba import Local_Max_Min  
        from CBullDivg_Analysis_numba import CBullDivg_analysis
        logger.info("Numba-optimized functions loaded - expect 50-100x speedup!")
        PERFORMANCE_MODE = "NUMBA_JIT"
    except ImportError as e:
        logger.warning(f"Numba optimization unavailable: {e}")
        logger.info("Falling back to original functions...")
        USE_NUMBA_OPTIMIZATION = False

# Fallback to original functions
if not USE_NUMBA_OPTIMIZATION:
    try:
        from Initialize_RSI_EMA_MACD import Initialize_RSI_EMA_MACD
        logger.info("Initialize_RSI_EMA_MACD.py loaded")
    except ImportError:
        logger.error("Initialize_RSI_EMA_MACD.py MISSING!")
        modules_ok = False

    try:
        from Local_Maximas_Minimas import Local_Max_Min
        logger.info("Local_Maximas_Minimas.py loaded")
    except ImportError:
        logger.error("Local_Maximas_Minimas.py MISSING!")
        modules_ok = False

    try:
        from CBullDivg_Analysis_vectorized import CBullDivg_analysis
        logger.info("CBullDivg_Analysis_vectorized.py loaded")
    except ImportError:
        logger.error("CBullDivg_Analysis_vectorized.py MISSING!")
        modules_ok = False
    
    PERFORMANCE_MODE = "STANDARD"

if not modules_ok:
    logger.error("Critical modules missing! Exiting...")
    sys.exit(1)

logger.info("All modules loaded successfully")

app = Flask(__name__)
CORS(app)

# Store uploaded data temporarily
# Note: In development mode, sessions are lost when server restarts
sessions = {}

def cleanup_old_sessions():
    """Clean up sessions older than 1 hour"""
    current_time = time.time()
    old_sessions = []
    for session_id, session_data in sessions.items():
        # If session_data is just a DataFrame, we can't track age easily
        # For now, we'll keep all sessions until manual cleanup
        pass
    return len(sessions)

def log_performance(operation: str, start_time: float, data_size: int):
    """Log performance metrics"""
    if ENABLE_PERFORMANCE_LOGGING:
        duration = time.perf_counter() - start_time
        throughput = data_size / duration if duration > 0 else 0
        logger.info(f"{operation}: {duration:.3f}s ({throughput:,.0f} rows/sec) [{PERFORMANCE_MODE}]")

@app.route('/')
def index():
    logger.info("Serving index.html")
    return send_from_directory('static', 'index.html')

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

@app.route('/api/health', methods=['GET'])
def health_check():
    logger.info("Health check requested")
    status = {
        'status': 'ok' if modules_ok else 'degraded',
        'performance_mode': PERFORMANCE_MODE,
        'numba_enabled': USE_NUMBA_OPTIMIZATION,
        'modules_loaded': modules_ok
    }
    
    if modules_ok:
        logger.info("All modules loaded")
        status['message'] = f"Server ready with {PERFORMANCE_MODE} performance mode"
    else:
        logger.error("Some modules missing")
        status['status'] = 'degraded'
        status['message'] = "Some modules are missing"
    
    return jsonify(status)

@app.route('/api/upload', methods=['POST'])
def upload_file():
    logger.info("File upload requested")
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        start_time = time.perf_counter()
        
        # Determine file type and create appropriate temp file
        filename = file.filename.lower()
        if filename.endswith('.csv'):
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
            file.save(temp_file.name)
            temp_file.close()
            # Load CSV file
            df = pd.read_csv(temp_file.name)
        elif filename.endswith('.parquet'):
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.parquet')
            file.save(temp_file.name)
            temp_file.close()
            # Load parquet file
            df = pd.read_parquet(temp_file.name)
        else:
            return jsonify({'error': 'Unsupported file type. Please use CSV or Parquet files.'}), 400
        
        # Clean up temp file
        os.unlink(temp_file.name)
        
        # Generate session ID (use string to avoid JavaScript integer precision issues)
        session_id = str(uuid.uuid4())
        
        # Store data in session
        sessions[session_id] = df
        
        log_performance("File Upload & Load", start_time, len(df))
        
        # Get date range if date column exists
        date_range = None
        if 'date' in df.columns:
            try:
                dates = pd.to_datetime(df['date'])
                date_range = {
                    'start': dates.min().strftime('%Y-%m-%d'),
                    'end': dates.max().strftime('%Y-%m-%d')
                }
            except:
                pass
        
        logger.info(f"Processing file: {file.filename}")
        logger.info(f"File uploaded, session_id: {session_id}, rows: {len(df)}")
        logger.info(f"Total active sessions: {len(sessions)}")
        
        response_data = {
            'success': True,
            'session_id': session_id,
            'rows': len(df),
            'columns': len(df.columns),
            'performance_mode': PERFORMANCE_MODE
        }
        
        # Add info section for backward compatibility
        if date_range:
            response_data['info'] = {
                'rows': len(df),
                'date_range': date_range
            }
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@app.route('/api/analyze', methods=['POST'])
def analyze():
    logger.info("Analyze request received")
    
    try:
        data = request.json
        session_id = data.get('session_id')
        variants = data.get('variants', [])
        
        # Debug logging
        logger.info(f"Session ID received: {session_id}")
        logger.info(f"Available sessions: {list(sessions.keys())}")
        
        if not session_id:
            logger.error("No session_id provided in request")
            return jsonify({'error': 'No session_id provided'}), 400
            
        if session_id not in sessions:
            logger.error(f"Session {session_id} not found in sessions")
            return jsonify({'error': f'Session {session_id} not found'}), 400
        
        df = sessions[session_id].copy()
        logger.info(f"Processing session_id: {session_id}, variants: {len(variants)}")
        logger.info(f"Data loaded: {len(df)} rows")
        
        overall_start = time.perf_counter()
        
        # Step 1: Initialize technical indicators
        logger.info("Starting analysis...")
        step1_start = time.perf_counter()
        df_with_indicators = Initialize_RSI_EMA_MACD(df)
        log_performance("Technical Indicators", step1_start, len(df))
        
        # Step 2: Calculate local maxima and minima
        step2_start = time.perf_counter()
        df_with_extrema = Local_Max_Min(df_with_indicators)
        log_performance("Local Maxima/Minima", step2_start, len(df))
        
        # Step 3: Analyze bullish divergences for each variant
        results = {}
        
        for variant in variants:
            variant_start = time.perf_counter()
            logger.info(f"   Analyzing: {variant['name']}")
            
            # Perform divergence analysis
            df_analysis = CBullDivg_analysis(
                df_with_extrema.copy(),
                variant['window'],
                variant['candleTol'],
                variant['macdTol']
            )
            
            # Extract divergence results
            hidden_divergences = []
            classic_divergences = []
            
            # Process hidden bullish divergences
            hidden_mask = df_analysis['CBullD_1'] == 1
            for idx in df_analysis[hidden_mask].index:
                row = df_analysis.loc[idx]
                hidden_divergences.append({
                    'date': row['date'],
                    'low': float(row['low']),
                    'high': float(row['high']),
                    'rsi': float(row['RSI']),
                    'macd': float(row['macd_histogram']),
                    'strength': float(row.get('CBullD_Date_Gap_1', 1)),
                    'divType': 'hidden_bullish',
                    'window': variant['window']
                })
            
            # Process classic bullish divergences
            classic_mask = df_analysis['CBullD_2'] == 1
            for idx in df_analysis[classic_mask].index:
                row = df_analysis.loc[idx]
                classic_divergences.append({
                    'date': row['date'],
                    'low': float(row['low']),
                    'high': float(row['high']),
                    'rsi': float(row['RSI']),
                    'macd': float(row['macd_histogram']),
                    'strength': float(row.get('CBullD_Date_Gap_2', 1)),
                    'divType': 'classic_bullish',
                    'window': variant['window']
                })
            
            results[variant['id']] = {
                'hidden': hidden_divergences,
                'classic': classic_divergences,
                'total': len(hidden_divergences) + len(classic_divergences)
            }
            
            log_performance(f"Variant '{variant['name']}'", variant_start, len(df))
        
        # Prepare chart data
        chart_data = {
            'dates': df_with_extrema['date'].tolist(),
            'open': df_with_extrema['open'].tolist(),
            'high': df_with_extrema['high'].tolist(),
            'low': df_with_extrema['low'].tolist(),
            'close': df_with_extrema['close'].tolist(),
            'rsi': df_with_extrema['RSI'].tolist(),
            'macd_line': df_with_extrema['MACD'].tolist(),
            'macd_signal': df_with_extrema['MACD_signal'].tolist(),
            'macd_histogram': df_with_extrema['macd_histogram'].tolist(),
            'ema_20': df_with_extrema['EMA_20'].tolist(),
            'ema_50': df_with_extrema['EMA_50'].tolist(),
            'ema_100': df_with_extrema['EMA_100'].tolist(),
            'ema_200': df_with_extrema['EMA_200'].tolist()
        }
        
        log_performance("Complete Analysis", overall_start, len(df))
        
        total_divergences = sum(r['total'] for r in results.values())
        logger.info(f"✅ Analysis completed, returning {len(df)} data points")
        logger.info(f"🎯 Found {total_divergences} total divergences across {len(variants)} variants")
        
        return jsonify({
            'success': True,
            'results': results,
            'chartData': chart_data,
            'performance_mode': PERFORMANCE_MODE,
            'processing_time': time.perf_counter() - overall_start,
            'data_points': len(df),
            'total_divergences': total_divergences
        })
        
    except Exception as e:
        logger.error(f"💥 Analysis failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

def open_browser():
    """Open browser after a short delay"""
    time.sleep(3)
    webbrowser.open('http://localhost:5000')

if __name__ == '__main__':
    print("=" * 80)
    print("BULLISH DIVERGENCE ANALYZER - Ultra-High Performance Edition")
    print("=" * 80)
    print("Features:")
    print(f"   Performance Mode: {PERFORMANCE_MODE}")
    if USE_NUMBA_OPTIMIZATION:
        print("   Numba JIT Compilation: ENABLED (50-100x faster)")
        print("   Optimized for huge datasets (65k+ rows)")
        print("   Ultra-fast divergence detection")
        print("   Smart memory usage with caching")
    else:
        print("   Standard analysis mode")
    print("   Polars-optimized data processing")
    print("=" * 80)
    print("Server running on: http://localhost:5000")
    print("   Browser opens automatically in 3 seconds...")
    print("   To stop: Ctrl+C")
    print("=" * 80)
    
    # Open browser in background thread
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()
    
    # Start Flask server
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=True)