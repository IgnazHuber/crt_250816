#!/usr/bin/env python3
"""
BULLISH DIVERGENCE ANALYZER - Enhanced Version
Mit Vergleichsmarkern, EMAs, Y-Zoom und schwarzem Design

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
            background: #000000;
            color: white; 
            padding: 20px;
        }
        .container { max-width: 1600px; margin: 0 auto; }
        h1 { 
            text-align: center; 
            margin-bottom: 30px; 
            font-size: 2.5em;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
            background: linear-gradient(135deg, #667eea, #764ba2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .panel {
            background: rgba(255,255,255,0.05); 
            backdrop-filter: blur(10px);
            border-radius: 15px; 
            padding: 20px; 
            margin-bottom: 20px;
            border: 1px solid rgba(255,255,255,0.1);
        }
        .btn {
            padding: 10px 20px; 
            border: none; 
            border-radius: 8px;
            cursor: pointer; 
            font-weight: bold; 
            margin: 5px;
            transition: all 0.3s;
        }
        .btn:hover { 
            transform: translateY(-2px); 
            box-shadow: 0 6px 20px rgba(102,126,234,0.4);
        }
        .btn-primary { 
            background: linear-gradient(45deg, #667eea, #764ba2); 
            color: white; 
        }
        .btn-success { 
            background: linear-gradient(45deg, #4CAF50, #45a049); 
            color: white; 
        }
        .btn-danger { 
            background: #ff4444; 
            color: white; 
            padding: 5px 10px;
            font-size: 0.9em;
        }
        input[type="file"] { display: none; }
        .file-label {
            display: inline-block; 
            padding: 12px 24px;
            background: linear-gradient(45deg, #4CAF50, #45a049);
            color: white; 
            border-radius: 8px; 
            cursor: pointer;
            font-weight: bold;
            transition: all 0.3s;
            box-shadow: 0 4px 15px rgba(76,175,80,0.3);
        }
        .file-label:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(76,175,80,0.4);
        }
        .param-grid {
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px; 
            margin: 20px 0;
        }
        .param-group {
            background: rgba(255,255,255,0.05); 
            padding: 15px;
            border-radius: 10px;
            border: 1px solid rgba(255,255,255,0.1);
        }
        .param-group h3 {
            margin-bottom: 10px;
            color: #a0d0ff;
            font-size: 1.1em;
        }
        .param-group label { 
            display: block; 
            margin-bottom: 5px; 
            font-size: 0.9em; 
            color: #ddd;
        }
        .param-group input {
            width: 100%; 
            padding: 5px; 
            border: 1px solid rgba(255,255,255,0.2);
            background: rgba(255,255,255,0.1); 
            color: white; 
            border-radius: 5px;
            text-align: center;
        }
        .variant-item {
            background: rgba(255,255,255,0.1); 
            padding: 10px;
            border-radius: 8px; 
            margin: 5px 0;
            display: flex; 
            justify-content: space-between; 
            align-items: center;
            border-left: 4px solid;
        }
        #chart { 
            height: 800px; 
            background: rgba(0,0,0,0.8); 
            border-radius: 10px; 
            border: 1px solid rgba(255,255,255,0.1);
        }
        .message { 
            padding: 15px; 
            border-radius: 8px; 
            margin: 10px 0; 
        }
        .success { 
            background: rgba(0,255,0,0.1); 
            border: 1px solid rgba(0,255,0,0.3); 
            color: #6bff6b;
        }
        .error { 
            background: rgba(255,0,0,0.1); 
            border: 1px solid rgba(255,0,0,0.3); 
            color: #ff6b6b;
        }
        .loading { 
            background: rgba(33,150,243,0.1); 
            border: 1px solid rgba(33,150,243,0.3);
            color: #a0d0ff;
        }
        .stats-container {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        .stats-card {
            background: rgba(255,255,255,0.05);
            padding: 20px;
            border-radius: 10px;
            border: 1px solid rgba(255,255,255,0.1);
        }
        .stats-card h3 {
            color: #a0d0ff;
            margin-bottom: 15px;
        }
        .variant-toggle {
            margin-right: 10px;
        }
        .legend-info {
            background: rgba(255,255,255,0.05);
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 20px;
            border: 1px solid rgba(255,255,255,0.1);
        }
        .legend-item {
            display: inline-flex;
            align-items: center;
            margin-right: 20px;
            margin-bottom: 5px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔍 Bullish Divergence Parameter Analyzer</h1>
        
        <div class="panel">
            <h3>📁 Daten laden</h3>
            <label for="fileInput" class="file-label">📁 CSV/Parquet auswählen</label>
            <input type="file" id="fileInput" accept=".csv,.parquet">
            <span id="fileName" style="margin-left: 15px; color: #a0d0ff;"></span>
            <div id="uploadMsg"></div>
        </div>
        
        <div class="panel">
            <h3>⚙️ Parameter einstellen</h3>
            <div class="param-grid">
                <div class="param-group">
                    <h3>Window Parameter</h3>
                    <label>Window Size:</label>
                    <input type="number" id="window" value="5" min="1" max="20">
                </div>
                <div class="param-group">
                    <h3>Toleranzen</h3>
                    <label>Candle Tol (%):</label>
                    <input type="number" id="candleTol" value="0.1" step="0.01">
                    <label style="margin-top: 10px;">MACD Tol (%):</label>
                    <input type="number" id="macdTol" value="3.25" step="0.01">
                </div>
                <div class="param-group">
                    <h3>Variantenname</h3>
                    <input type="text" id="variantName" placeholder="z.B. Standard">
                </div>
            </div>
            <button class="btn btn-success" onclick="addVariant()">➕ Variante hinzufügen</button>
            <button class="btn btn-primary" onclick="runAnalysis()">🚀 Analyse starten</button>
            
            <div style="margin-top: 20px;">
                <h4>Parametervarianten:</h4>
                <div id="variantsList"></div>
            </div>
        </div>
        
        <div class="panel" id="chartPanel" style="display: none;">
            <h3>📊 Ergebnisse</h3>
            
            <div class="legend-info">
                <h4>Legende:</h4>
                <div class="legend-item">
                    <span style="font-size: 20px;">▲</span>
                    <span style="margin-left: 5px;">Classic Divergence</span>
                </div>
                <div class="legend-item">
                    <span style="font-size: 20px;">◆</span>
                    <span style="margin-left: 5px;">Hidden Divergence</span>
                </div>
                <div class="legend-item">
                    <span style="font-size: 20px;">⬆️</span>
                    <span style="margin-left: 5px;">Zusätzlich zur Basis</span>
                </div>
            </div>
            
            <div id="chart"></div>
            <div class="stats-container" id="statsContainer"></div>
        </div>
    </div>
    
    <script>
        let sessionId = null;
        let variants = [];
        // Sehr kräftige, unterscheidbare Farben
        const colors = [
            '#FF0000', // Knallrot
            '#00FF00', // Knallgrün  
            '#0080FF', // Hellblau
            '#FFD700', // Gold
            '#FF00FF', // Magenta
            '#00FFFF', // Cyan
            '#FFA500', // Orange
            '#FF1493', // Deep Pink
            '#7FFF00', // Chartreuse
            '#9370DB'  // Medium Purple
        ];
        
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
                color: colors[variants.length % colors.length],
                visible: true
            };
            variants.push(variant);
            updateVariantsList();
            document.getElementById('variantName').value = '';
        }
        
        function updateVariantsList() {
            const html = variants.map((v, i) => `
                <div class="variant-item" style="border-left-color: ${v.color};">
                    <div>
                        <input type="checkbox" class="variant-toggle" 
                               id="toggle_${v.id}" 
                               ${v.visible ? 'checked' : ''}
                               onchange="toggleVariant(${v.id})">
                        <label for="toggle_${v.id}">
                            <strong>${v.name}</strong>${i === 0 ? ' (BASIS)' : ''} - 
                            W:${v.window} C:${v.candleTol}% M:${v.macdTol}%
                        </label>
                    </div>
                    <button class="btn btn-danger" onclick="removeVariant(${i})">✕</button>
                </div>
            `).join('');
            document.getElementById('variantsList').innerHTML = html;
        }
        
        function removeVariant(index) {
            variants.splice(index, 1);
            updateVariantsList();
        }
        
        function toggleVariant(id) {
            const variant = variants.find(v => v.id === id);
            if (variant) {
                variant.visible = !variant.visible;
                // Chart wird nach Analyse neu gezeichnet
            }
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
            
            showMessage('uploadMsg', 'Analyse läuft mit Python-Modulen...', 'loading');
            
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
            
            const traces = [];
            
            // Candlestick
            traces.push({
                x: data.chartData.dates,
                open: data.chartData.open,
                high: data.chartData.high,
                low: data.chartData.low,
                close: data.chartData.close,
                type: 'candlestick',
                name: 'Price',
                xaxis: 'x',
                yaxis: 'y',
                increasing: {line: {color: '#00FF00'}},
                decreasing: {line: {color: '#FF0000'}}
            });
            
            // EMAs hinzufügen
            if (data.chartData.ema20) {
                traces.push({
                    x: data.chartData.dates,
                    y: data.chartData.ema20,
                    type: 'scatter',
                    mode: 'lines',
                    name: 'EMA 20',
                    line: {color: '#FFD700', width: 1},
                    xaxis: 'x',
                    yaxis: 'y'
                });
            }
            if (data.chartData.ema50) {
                traces.push({
                    x: data.chartData.dates,
                    y: data.chartData.ema50,
                    type: 'scatter',
                    mode: 'lines',
                    name: 'EMA 50',
                    line: {color: '#00FFFF', width: 1},
                    xaxis: 'x',
                    yaxis: 'y'
                });
            }
            if (data.chartData.ema100) {
                traces.push({
                    x: data.chartData.dates,
                    y: data.chartData.ema100,
                    type: 'scatter',
                    mode: 'lines',
                    name: 'EMA 100',
                    line: {color: '#FF00FF', width: 1},
                    xaxis: 'x',
                    yaxis: 'y'
                });
            }
            if (data.chartData.ema200) {
                traces.push({
                    x: data.chartData.dates,
                    y: data.chartData.ema200,
                    type: 'scatter',
                    mode: 'lines',
                    name: 'EMA 200',
                    line: {color: '#9370DB', width: 1},
                    xaxis: 'x',
                    yaxis: 'y'
                });
            }
            
            // RSI mit Überkauft/Überverkauft-Linien
            traces.push({
                x: data.chartData.dates,
                y: data.chartData.rsi,
                type: 'scatter',
                mode: 'lines',
                name: 'RSI',
                yaxis: 'y2',
                line: {color: '#FFA500', width: 2}
            });
            
            // RSI Referenzlinien
            traces.push({
                x: data.chartData.dates,
                y: new Array(data.chartData.dates.length).fill(70),
                type: 'scatter',
                mode: 'lines',
                name: 'Overbought',
                line: {color: '#FF0000', width: 1, dash: 'dash'},
                yaxis: 'y2',
                showlegend: false
            });
            
            traces.push({
                x: data.chartData.dates,
                y: new Array(data.chartData.dates.length).fill(30),
                type: 'scatter',
                mode: 'lines',
                name: 'Oversold',
                line: {color: '#00FF00', width: 1, dash: 'dash'},
                yaxis: 'y2',
                showlegend: false
            });
            
            // MACD Histogram
            traces.push({
                x: data.chartData.dates,
                y: data.chartData.macd_histogram,
                type: 'bar',
                name: 'MACD Histogram',
                yaxis: 'y3',
                marker: {color: data.chartData.macd_histogram.map(v => v > 0 ? '#00FF00' : '#FF0000')}
            });
            
            // Basis-Divergenzen für Vergleich (erste Variante)
            let baseDivergences = null;
            if (variants.length > 0 && data.results[variants[0].id]) {
                baseDivergences = [
                    ...data.results[variants[0].id].classic.map(d => ({...d, type: 'classic'})),
                    ...data.results[variants[0].id].hidden.map(d => ({...d, type: 'hidden'}))
                ];
            }
            
            // Divergenz-Marker für jede Variante
            variants.forEach((v, idx) => {
                if (!v.visible) return;
                
                const res = data.results[v.id];
                if (!res) return;
                
                // Finde zusätzliche Divergenzen im Vergleich zur Basis
                let additionalClassic = [];
                let additionalHidden = [];
                
                if (idx > 0 && baseDivergences) {
                    // Vergleiche mit Basis
                    res.classic.forEach(div => {
                        const isNew = !baseDivergences.some(bd => 
                            bd.date === div.date && bd.type === 'classic'
                        );
                        if (isNew) additionalClassic.push(div);
                    });
                    
                    res.hidden.forEach(div => {
                        const isNew = !baseDivergences.some(bd => 
                            bd.date === div.date && bd.type === 'hidden'
                        );
                        if (isNew) additionalHidden.push(div);
                    });
                }
                
                // Classic Divergences - große Dreiecke
                if (res.classic.length > 0) {
                    traces.push({
                        x: res.classic.map(d => d.date),
                        y: res.classic.map(d => d.low),
                        type: 'scatter',
                        mode: 'markers',
                        name: v.name + ' Classic',
                        marker: {
                            size: 20,
                            color: v.color,
                            symbol: 'triangle-up',
                            line: {color: '#FFFFFF', width: 3},
                            opacity: 1
                        },
                        text: res.classic.map(d => 
                            `${v.name}<br>Classic<br>RSI: ${d.rsi?.toFixed(2) || 'N/A'}<br>MACD: ${d.macd?.toFixed(4) || 'N/A'}`
                        ),
                        hovertemplate: '%{text}<extra></extra>',
                        yaxis: 'y'
                    });
                }
                
                // Hidden Divergences - große Diamanten
                if (res.hidden.length > 0) {
                    traces.push({
                        x: res.hidden.map(d => d.date),
                        y: res.hidden.map(d => d.low),
                        type: 'scatter',
                        mode: 'markers',
                        name: v.name + ' Hidden',
                        marker: {
                            size: 18,
                            color: v.color,
                            symbol: 'diamond',
                            line: {color: '#FFFFFF', width: 3},
                            opacity: 1
                        },
                        text: res.hidden.map(d => 
                            `${v.name}<br>Hidden<br>RSI: ${d.rsi?.toFixed(2) || 'N/A'}<br>MACD: ${d.macd?.toFixed(4) || 'N/A'}`
                        ),
                        hovertemplate: '%{text}<extra></extra>',
                        yaxis: 'y'
                    });
                }
                
                // Zusätzliche Marker (Pfeile) für neue Divergenzen
                if (idx > 0 && additionalClassic.length > 0) {
                    traces.push({
                        x: additionalClassic.map(d => d.date),
                        y: additionalClassic.map(d => d.low * 0.98), // Etwas unter dem Low
                        type: 'scatter',
                        mode: 'markers+text',
                        name: v.name + ' NEU',
                        marker: {
                            size: 25,
                            color: v.color,
                            symbol: 'arrow-up',
                            line: {color: '#FFFF00', width: 3}
                        },
                        text: '⬆',
                        textposition: 'bottom center',
                        textfont: {size: 20, color: v.color},
                        showlegend: false,
                        yaxis: 'y'
                    });
                }
                
                if (idx > 0 && additionalHidden.length > 0) {
                    traces.push({
                        x: additionalHidden.map(d => d.date),
                        y: additionalHidden.map(d => d.low * 0.98),
                        type: 'scatter',
                        mode: 'markers+text',
                        name: v.name + ' NEU Hidden',
                        marker: {
                            size: 23,
                            color: v.color,
                            symbol: 'arrow-up',
                            line: {color: '#FFFF00', width: 3}
                        },
                        text: '⬆',
                        textposition: 'bottom center',
                        textfont: {size: 18, color: v.color},
                        showlegend: false,
                        yaxis: 'y'
                    });
                }
            });
            
            const layout = {
                title: {
                    text: 'Bullish Divergence Analysis',
                    font: {size: 24, color: '#FFFFFF'}
                },
                dragmode: 'zoom',
                showlegend: true,
                legend: {
                    orientation: 'h',
                    y: 1.15,
                    x: 0,
                    bgcolor: 'rgba(0,0,0,0.5)',
                    font: {color: '#FFFFFF'}
                },
                xaxis: {
                    rangeslider: {visible: false},
                    domain: [0, 1],
                    anchor: 'y3',
                    gridcolor: 'rgba(255,255,255,0.1)',
                    zerolinecolor: 'rgba(255,255,255,0.2)'
                },
                yaxis: {
                    title: 'Price',
                    domain: [0.55, 1],
                    gridcolor: 'rgba(255,255,255,0.1)',
                    titlefont: {color: '#FFFFFF'},
                    tickfont: {color: '#FFFFFF'},
                    fixedrange: false  // Y-Achse zoombar
                },
                yaxis2: {
                    title: 'RSI',
                    domain: [0.28, 0.52],
                    anchor: 'x',
                    gridcolor: 'rgba(255,255,255,0.1)',
                    titlefont: {color: '#FFFFFF'},
                    tickfont: {color: '#FFFFFF'},
                    fixedrange: false  // Y-Achse zoombar
                },
                yaxis3: {
                    title: 'MACD Histogram',
                    domain: [0, 0.25],
                    anchor: 'x',
                    gridcolor: 'rgba(255,255,255,0.1)',
                    titlefont: {color: '#FFFFFF'},
                    tickfont: {color: '#FFFFFF'},
                    fixedrange: false  // Y-Achse zoombar
                },
                plot_bgcolor: '#0a0a0a',
                paper_bgcolor: '#1a1a1a',
                font: {color: '#FFFFFF'},
                hovermode: 'x unified'
            };
            
            const config = {
                responsive: true,
                displayModeBar: true,
                displaylogo: false,
                modeBarButtonsToRemove: ['select2d', 'lasso2d']
            };
            
            Plotly.newPlot('chart', traces, layout, config);
            
            // Stats aktualisieren
            updateStats(data.results);
        }
        
        function updateStats(results) {
            const container = document.getElementById('statsContainer');
            let html = '';
            
            // Vergleichstabelle
            html += '<div class="stats-card" style="grid-column: span 2;">';
            html += '<h3>📊 Parametervergleich</h3>';
            html += '<table style="width: 100%; border-collapse: collapse;">';
            html += `
                <tr style="border-bottom: 2px solid rgba(255,255,255,0.2);">
                    <th style="text-align: left; padding: 10px;">Variante</th>
                    <th style="text-align: center; padding: 10px;">Classic</th>
                    <th style="text-align: center; padding: 10px;">Hidden</th>
                    <th style="text-align: center; padding: 10px;">Gesamt</th>
                    <th style="text-align: center; padding: 10px;">Zusätzlich</th>
                </tr>
            `;
            
            const baseTotal = variants.length > 0 && results[variants[0].id] ? 
                results[variants[0].id].total : 0;
            
            variants.forEach((v, idx) => {
                const res = results[v.id];
                if (!res) return;
                
                const additional = idx === 0 ? 0 : res.total - baseTotal;
                const additionalText = additional > 0 ? `+${additional}` : 
                                      additional < 0 ? `${additional}` : '0';
                const additionalColor = additional > 0 ? '#00FF00' : 
                                       additional < 0 ? '#FF0000' : '#FFFFFF';
                
                html += `
                    <tr style="border-bottom: 1px solid rgba(255,255,255,0.1);">
                        <td style="padding: 10px;">
                            <span style="background: ${v.color}; width: 15px; height: 15px; 
                                        display: inline-block; margin-right: 8px; 
                                        border: 2px solid white;"></span>
                            ${v.name}${idx === 0 ? ' (BASIS)' : ''}
                        </td>
                        <td style="text-align: center; padding: 10px; color: #00FF00;">
                            ${res.classic.length}
                        </td>
                        <td style="text-align: center; padding: 10px; color: #00FFFF;">
                            ${res.hidden.length}
                        </td>
                        <td style="text-align: center; padding: 10px; font-weight: bold; color: #FFD700;">
                            ${res.total}
                        </td>
                        <td style="text-align: center; padding: 10px; font-weight: bold; color: ${additionalColor};">
                            ${additionalText}
                        </td>
                    </tr>
                `;
            });
            
            html += '</table></div>';
            
            // Beste Variante
            let bestVariant = null;
            let maxTotal = 0;
            variants.forEach(v => {
                const total = results[v.id]?.total || 0;
                if (total > maxTotal) {
                    maxTotal = total;
                    bestVariant = v;
                }
            });
            
            if (bestVariant) {
                html += '<div class="stats-card">';
                html += '<h3>🏆 Beste Variante</h3>';
                html += `<p><strong style="color: ${bestVariant.color};">${bestVariant.name}</strong></p>`;
                html += `<p>Divergenzen: ${maxTotal}</p>`;
                html += `<p>Window: ${bestVariant.window}</p>`;
                html += `<p>Candle Tol: ${bestVariant.candleTol}%</p>`;
                html += `<p>MACD Tol: ${bestVariant.macdTol}%</p>`;
                html += '</div>';
            }
            
            container.innerHTML = html;
        }
        
        function showMessage(id, msg, type) {
            document.getElementById(id).innerHTML = `<div class="message ${type}">${msg}</div>`;
            if (type === 'success') {
                setTimeout(() => document.getElementById(id).innerHTML = '', 3000);
            }
        }
        
        // Auto-add default variants
        window.onload = () => {
            // Standard (Basis)
            document.getElementById('variantName').value = 'Standard';
            document.getElementById('window').value = 5;
            document.getElementById('candleTol').value = 0.1;
            document.getElementById('macdTol').value = 3.25;
            addVariant();
            
            // Konservativ
            document.getElementById('variantName').value = 'Konservativ';
            document.getElementById('window').value = 7;
            document.getElementById('candleTol').value = 0.05;
            document.getElementById('macdTol').value = 2.0;
            addVariant();
            
            // Aggressiv
            document.getElementById('variantName').value = 'Aggressiv';
            document.getElementById('window').value = 3;
            document.getElementById('candleTol').value = 0.2;
            document.getElementById('macdTol').value = 5.0;
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
        
        # Validierung
        required = ['date', 'open', 'high', 'low', 'close']
        missing = [col for col in required if col not in df.columns]
        if missing:
            return jsonify({'error': f'Fehlende Spalten: {missing}'}), 400
        
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
        print("📊 Starte Analyse mit Python-Modulen...")
        df = Initialize_RSI_EMA_MACD(df)
        Local_Max_Min(df)
        
        results = {}
        for variant in variants:
            print(f"   Analysiere: {variant['name']}")
            df_var = df.copy()
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
        
        # Prepare chart data with EMAs
        chart_data = {
            'dates': df['date'].astype(str).tolist(),
            'open': df['open'].tolist(),
            'high': df['high'].tolist(),
            'low': df['low'].tolist(),
            'close': df['close'].tolist(),
            'rsi': df.get('RSI', pd.Series()).fillna(0).tolist(),
            'macd_histogram': df.get('macd_histogram', pd.Series()).fillna(0).tolist(),
            # EMAs hinzufügen
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
        return jsonify({'error': str(e)}), 500

# ========== AUTO-START ==========
if __name__ == '__main__':
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
    
    # Auto-open browser
    def open_browser():
        time.sleep(3)
        webbrowser.open('http://localhost:5000')
    
    threading.Thread(target=open_browser, daemon=True).start()
    
    # Start server
    app.run(debug=False, port=5000)