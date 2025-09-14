#!/usr/bin/env python3
"""
BULLISH DIVERGENCE ANALYZER - All-in-One Version
Kompletter Server + Interface in einer Datei!

VERWENDUNG:
1. Speichere diese Datei als 'bullish_analyzer.py'
2. Stelle sicher, dass deine 3 Module im gleichen Ordner sind
3. Starte mit: python bullish_analyzer.py
4. Öffne Browser: http://localhost:5000
"""

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import pandas as pd
import numpy as np
import os
import tempfile
import webbrowser
import threading
import time
import sys

# ========== IMPORT CHECK ==========
print("🔍 Prüfe Module...")
modules_ok = True

try:
    from Initialize_RSI_EMA_MACD import Initialize_RSI_EMA_MACD
    print("✅ Initialize_RSI_EMA_MACD.py gefunden")
except ImportError:
    print("❌ Initialize_RSI_EMA_MACD.py FEHLT!")
    modules_ok = False

try:
    from Local_Maximas_Minimas import Local_Max_Min
    print("✅ Local_Maximas_Minimas.py gefunden")
except ImportError:
    print("❌ Local_Maximas_Minimas.py FEHLT!")
    modules_ok = False

try:
    from CBullDivg_Analysis_vectorized import CBullDivg_analysis
    print("✅ CBullDivg_Analysis_vectorized.py gefunden")
except ImportError:
    print("❌ CBullDivg_Analysis_vectorized.py FEHLT!")
    modules_ok = False

if not modules_ok:
    print("\n⚠️  WICHTIG: Kopiere die fehlenden Python-Module in diesen Ordner!")
    response = input("Trotzdem starten? (j/n): ")
    if response.lower() != 'j':
        sys.exit(1)

# ========== HTML TEMPLATE ==========
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="de">
<head>
    <meta charset="UTF-8">
    <title>Bullish Divergence Analyzer</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.27.0/plotly.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Segoe UI', sans-serif; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white; padding: 20px;
        }
        .container { max-width: 1400px; margin: 0 auto; }
        h1 { text-align: center; margin-bottom: 30px; }
        .panel {
            background: rgba(255,255,255,0.1); backdrop-filter: blur(10px);
            border-radius: 15px; padding: 20px; margin-bottom: 20px;
        }
        .btn {
            padding: 10px 20px; border: none; border-radius: 8px;
            cursor: pointer; font-weight: bold; margin: 5px;
            transition: transform 0.2s;
        }
        .btn:hover { transform: translateY(-2px); }
        .btn-primary { background: linear-gradient(45deg, #667eea, #764ba2); color: white; }
        .btn-success { background: linear-gradient(45deg, #4CAF50, #45a049); color: white; }
        .btn-danger { background: #ff4444; color: white; }
        input[type="file"] { display: none; }
        .file-label {
            display: inline-block; padding: 10px 20px;
            background: linear-gradient(45deg, #4CAF50, #45a049);
            color: white; border-radius: 8px; cursor: pointer;
        }
        .param-grid {
            display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px; margin: 20px 0;
        }
        .param-group {
            background: rgba(255,255,255,0.05); padding: 10px;
            border-radius: 8px;
        }
        .param-group label { display: block; margin-bottom: 5px; font-size: 0.9em; }
        .param-group input {
            width: 100%; padding: 5px; border: 1px solid rgba(255,255,255,0.2);
            background: rgba(255,255,255,0.1); color: white; border-radius: 4px;
        }
        .variant-item {
            background: rgba(255,255,255,0.1); padding: 10px;
            border-radius: 8px; margin: 5px 0;
            display: flex; justify-content: space-between; align-items: center;
        }
        #chart { height: 600px; background: rgba(0,0,0,0.3); border-radius: 10px; }
        .message { padding: 15px; border-radius: 8px; margin: 10px 0; }
        .success { background: rgba(76,175,80,0.2); border: 1px solid #4CAF50; }
        .error { background: rgba(244,67,54,0.2); border: 1px solid #f44336; }
        .loading { background: rgba(33,150,243,0.2); border: 1px solid #2196F3; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔍 Bullish Divergence Parameter Analyzer</h1>
        
        <div class="panel">
            <h3>📁 Daten laden</h3>
            <label for="fileInput" class="file-label">CSV/Parquet auswählen</label>
            <input type="file" id="fileInput" accept=".csv,.parquet">
            <span id="fileName" style="margin-left: 15px;"></span>
            <div id="uploadMsg"></div>
        </div>
        
        <div class="panel">
            <h3>⚙️ Parameter einstellen</h3>
            <div class="param-grid">
                <div class="param-group">
                    <label>Window Size</label>
                    <input type="number" id="window" value="5" min="1" max="20">
                </div>
                <div class="param-group">
                    <label>Candle Tolerance (%)</label>
                    <input type="number" id="candleTol" value="0.1" step="0.01">
                </div>
                <div class="param-group">
                    <label>MACD Tolerance (%)</label>
                    <input type="number" id="macdTol" value="3.25" step="0.01">
                </div>
                <div class="param-group">
                    <label>Varianten-Name</label>
                    <input type="text" id="variantName" placeholder="z.B. Standard">
                </div>
            </div>
            <button class="btn btn-success" onclick="addVariant()">➕ Variante hinzufügen</button>
            <button class="btn btn-primary" onclick="runAnalysis()">🚀 Analyse starten</button>
            
            <div id="variants" style="margin-top: 20px;">
                <h4>Varianten:</h4>
                <div id="variantsList"></div>
            </div>
        </div>
        
        <div class="panel" id="chartPanel" style="display: none;">
            <h3>📊 Ergebnisse</h3>
            <div id="chart"></div>
            <div id="stats" style="margin-top: 20px;"></div>
        </div>
    </div>
    
    <script>
        let sessionId = null;
        let variants = [];
        const colors = ['#FF0000','#00FF00','#0080FF','#FFD700','#FF00FF'];
        
        // File Upload
        document.getElementById('fileInput').addEventListener('change', async (e) => {
            const file = e.target.files[0];
            if (!file) return;
            
            document.getElementById('fileName').textContent = file.name;
            const formData = new FormData();
            formData.append('file', file);
            
            showMessage('uploadMsg', 'Lade Datei...', 'loading');
            
            try {
                const response = await fetch('/upload', {
                    method: 'POST',
                    body: formData
                });
                const data = await response.json();
                
                if (data.success) {
                    sessionId = data.session_id;
                    showMessage('uploadMsg', `✅ ${data.rows} Zeilen geladen`, 'success');
                } else {
                    showMessage('uploadMsg', data.error, 'error');
                }
            } catch (error) {
                showMessage('uploadMsg', 'Upload fehlgeschlagen', 'error');
            }
        });
        
        function addVariant() {
            const variant = {
                id: variants.length,
                name: document.getElementById('variantName').value || `Variante ${variants.length + 1}`,
                window: parseInt(document.getElementById('window').value),
                candleTol: parseFloat(document.getElementById('candleTol').value),
                macdTol: parseFloat(document.getElementById('macdTol').value),
                color: colors[variants.length % colors.length]
            };
            variants.push(variant);
            updateVariantsList();
        }
        
        function updateVariantsList() {
            const html = variants.map((v, i) => `
                <div class="variant-item" style="border-left: 4px solid ${v.color};">
                    <span><strong>${v.name}</strong> - W:${v.window} C:${v.candleTol}% M:${v.macdTol}%</span>
                    <button class="btn btn-danger" onclick="removeVariant(${i})">✕</button>
                </div>
            `).join('');
            document.getElementById('variantsList').innerHTML = html;
        }
        
        function removeVariant(index) {
            variants.splice(index, 1);
            updateVariantsList();
        }
        
        async function runAnalysis() {
            if (!sessionId) {
                alert('Bitte erst eine Datei laden!');
                return;
            }
            if (variants.length === 0) {
                alert('Bitte mindestens eine Variante hinzufügen!');
                return;
            }
            
            showMessage('uploadMsg', 'Analyse läuft...', 'loading');
            
            try {
                const response = await fetch('/analyze', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({session_id: sessionId, variants: variants})
                });
                const data = await response.json();
                
                if (data.success) {
                    showChart(data);
                    showMessage('uploadMsg', '✅ Analyse abgeschlossen', 'success');
                } else {
                    showMessage('uploadMsg', data.error, 'error');
                }
            } catch (error) {
                showMessage('uploadMsg', 'Analyse fehlgeschlagen', 'error');
            }
        }
        
        function showChart(data) {
            document.getElementById('chartPanel').style.display = 'block';
            
            const traces = [
                // Candlestick
                {
                    x: data.chartData.dates,
                    open: data.chartData.open,
                    high: data.chartData.high,
                    low: data.chartData.low,
                    close: data.chartData.close,
                    type: 'candlestick',
                    name: 'Price',
                    yaxis: 'y'
                },
                // RSI
                {
                    x: data.chartData.dates,
                    y: data.chartData.rsi,
                    type: 'scatter',
                    name: 'RSI',
                    yaxis: 'y2',
                    line: {color: '#FFA500'}
                },
                // MACD
                {
                    x: data.chartData.dates,
                    y: data.chartData.macd_histogram,
                    type: 'bar',
                    name: 'MACD',
                    yaxis: 'y3',
                    marker: {color: data.chartData.macd_histogram.map(v => v > 0 ? '#00FF00' : '#FF0000')}
                }
            ];
            
            // Divergenz-Marker
            variants.forEach(v => {
                const res = data.results[v.id];
                if (res && res.classic.length > 0) {
                    traces.push({
                        x: res.classic.map(d => d.date),
                        y: res.classic.map(d => d.low),
                        type: 'scatter',
                        mode: 'markers',
                        name: v.name + ' Classic',
                        marker: {size: 12, color: v.color, symbol: 'triangle-up'},
                        yaxis: 'y'
                    });
                }
            });
            
            const layout = {
                title: 'Bullish Divergence Analysis',
                grid: {rows: 3, columns: 1, pattern: 'independent'},
                xaxis: {anchor: 'y3'},
                yaxis: {domain: [0.55, 1], title: 'Price'},
                yaxis2: {domain: [0.28, 0.52], title: 'RSI'},
                yaxis3: {domain: [0, 0.25], title: 'MACD'},
                showlegend: true,
                height: 600
            };
            
            Plotly.newPlot('chart', traces, layout);
            
            // Stats
            let stats = '<h4>Statistiken:</h4>';
            variants.forEach(v => {
                const res = data.results[v.id];
                if (res) {
                    stats += `<p><strong style="color:${v.color}">${v.name}:</strong> 
                             Classic: ${res.classic.length}, Hidden: ${res.hidden.length}</p>`;
                }
            });
            document.getElementById('stats').innerHTML = stats;
        }
        
        function showMessage(id, msg, type) {
            document.getElementById(id).innerHTML = `<div class="message ${type}">${msg}</div>`;
        }
        
        // Auto-add first variant
        window.onload = () => {
            document.getElementById('variantName').value = 'Standard';
            addVariant();
        };
    </script>
</body>
</html>
'''

# ========== FLASK SERVER ==========
app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        file = request.files['file']
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Load data
        if filename.endswith('.csv'):
            df = pd.read_csv(filepath)
        elif filename.endswith('.parquet'):
            df = pd.read_parquet(filepath)
        else:
            return jsonify({'error': 'Nur CSV/Parquet unterstützt'}), 400
        
        # Save for session
        session_id = str(abs(hash(filename)))
        df.to_parquet(f"{app.config['UPLOAD_FOLDER']}/session_{session_id}.parquet")
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'rows': len(df)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.json
        session_id = data['session_id']
        variants = data['variants']
        
        # Load data
        df = pd.read_parquet(f"{app.config['UPLOAD_FOLDER']}/session_{session_id}.parquet")
        
        # Run analysis
        print("📊 Starte Analyse...")
        df = Initialize_RSI_EMA_MACD(df)
        Local_Max_Min(df)
        
        results = {}
        for variant in variants:
            df_var = df.copy()
            CBullDivg_analysis(df_var, variant['window'], variant['candleTol'], variant['macdTol'])
            
            classic = []
            hidden = []
            for i in range(len(df_var)):
                if df_var.iloc[i].get('CBullD_gen', 0) == 1:
                    classic.append({
                        'date': str(df_var.iloc[i]['date']),
                        'low': float(df_var.iloc[i]['low'])
                    })
                if df_var.iloc[i].get('CBullD_neg_MACD', 0) == 1:
                    hidden.append({
                        'date': str(df_var.iloc[i]['date']),
                        'low': float(df_var.iloc[i]['low'])
                    })
            
            results[variant['id']] = {
                'classic': classic,
                'hidden': hidden
            }
        
        # Prepare chart data
        chart_data = {
            'dates': df['date'].astype(str).tolist(),
            'open': df['open'].tolist(),
            'high': df['high'].tolist(),
            'low': df['low'].tolist(),
            'close': df['close'].tolist(),
            'rsi': df.get('RSI', pd.Series()).fillna(0).tolist(),
            'macd_histogram': df.get('macd_histogram', pd.Series()).fillna(0).tolist()
        }
        
        return jsonify({
            'success': True,
            'chartData': chart_data,
            'results': results
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ========== AUTO-START ==========
if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 BULLISH DIVERGENCE ANALYZER")
    print("="*60)
    print("📍 Server läuft auf: http://localhost:5000")
    print("   Browser öffnet automatisch in 3 Sekunden...")
    print("   Zum Beenden: Strg+C")
    print("="*60 + "\n")
    
    # Auto-open browser
    def open_browser():
        time.sleep(3)
        webbrowser.open('http://localhost:5000')
    
    threading.Thread(target=open_browser, daemon=True).start()
    
    # Start server
    app.run(debug=False, port=5000)