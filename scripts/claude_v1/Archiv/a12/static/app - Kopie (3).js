// Bullish Divergence Analyzer - Frontend JavaScript

let sessionId = null;
let variants = [];
let currentResults = null;
let chartInstance = null;
let arrowTraces = {}; // Speichert Pfeil-Traces für jede Variante

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

// Server Status Check
async function checkServerStatus() {
    try {
        const response = await fetch('/api/health');
        const data = await response.json();
        
        if (data.status === 'ok') {
            document.getElementById('serverStatus').innerHTML = '✅ Server online';
            document.getElementById('serverStatus').className = 'status-indicator status-online';
        } else {
            document.getElementById('serverStatus').innerHTML = '❌ Module fehlen';
            document.getElementById('serverStatus').className = 'status-indicator status-offline';
        }
    } catch (error) {
        document.getElementById('serverStatus').innerHTML = '❌ Server offline';
        document.getElementById('serverStatus').className = 'status-indicator status-offline';
    }
}

// File Upload Handler
document.addEventListener('DOMContentLoaded', function() {
    const fileInput = document.getElementById('fileInput');
    const fileName = document.getElementById('fileName');
    const uploadMsg = document.getElementById('uploadMsg');
    const clearData = document.getElementById('clearData');
    
    fileInput.addEventListener('change', async (e) => {
        const file = e.target.files[0];
        if (!file) return;
        
        fileName.textContent = file.name;
        clearData.style.display = 'inline-block';
        
        const formData = new FormData();
        formData.append('file', file);
        
        showMessage('uploadMsg', 'Lade Datei...', 'loading');
        
        try {
            const response = await fetch('/api/upload', {
                method: 'POST',
                body: formData
            });
            const data = await response.json();
            
            if (data.success) {
                sessionId = data.session_id;
                showMessage('uploadMsg', `✅ ${data.info.rows} Zeilen geladen`, 'success');
                document.getElementById('dataStatus').innerHTML = `📊 ${data.info.rows} Zeilen geladen`;
            } else {
                showMessage('uploadMsg', data.error, 'error');
            }
        } catch (error) {
            showMessage('uploadMsg', 'Upload fehlgeschlagen', 'error');
        }
    });
    
    clearData.addEventListener('click', () => {
        fileInput.value = '';
        fileName.textContent = '';
        clearData.style.display = 'none';
        sessionId = null;
        uploadMsg.innerHTML = '';
        document.getElementById('dataStatus').innerHTML = '';
        document.getElementById('resultsPanel').style.display = 'none';
        document.getElementById('validationPanel').style.display = 'none';
        currentResults = null;
        chartInstance = null;
        arrowTraces = {};
    });
    
    // Initial server check
    checkServerStatus();
});

// Preset Loading
function loadPreset(presetName) {
    const presets = {
        'standard': { window: 5, candleTol: 0.1, macdTol: 3.25, name: 'Standard' },
        'conservative': { window: 7, candleTol: 0.05, macdTol: 2.0, name: 'Konservativ' },
        'aggressive': { window: 3, candleTol: 0.2, macdTol: 5.0, name: 'Aggressiv' }
    };
    
    const preset = presets[presetName];
    if (preset) {
        document.getElementById('window').value = preset.window;
        document.getElementById('candleTol').value = preset.candleTol;
        document.getElementById('macdTol').value = preset.macdTol;
        document.getElementById('variantName').value = preset.name;
    }
}

// Variant Management
function addVariant() {
    const name = document.getElementById('variantName').value || `Variante ${variants.length + 1}`;
    const window = parseInt(document.getElementById('window').value);
    const candleTol = parseFloat(document.getElementById('candleTol').value);
    const macdTol = parseFloat(document.getElementById('macdTol').value);
    
    if (isNaN(window) || isNaN(candleTol) || isNaN(macdTol)) {
        alert('Bitte alle Parameter korrekt eingeben!');
        return;
    }
    
    const variant = {
        id: variants.length,
        name: name,
        window: window,
        candleTol: candleTol,
        macdTol: macdTol,
        color: colors[variants.length % colors.length],
        visible: true
    };
    
    variants.push(variant);
    updateVariantsList();
    
    // Clear inputs
    document.getElementById('variantName').value = '';
}

function updateVariantsList() {
    const container = document.getElementById('variantsList');
    if (!container) return;
    
    const html = variants.map((v, i) => `
        <div class="variant-item" style="border-left-color: ${v.color};">
            <div class="variant-controls">
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
    
    container.innerHTML = html;
}

function removeVariant(index) {
    variants.splice(index, 1);
    // Update IDs
    variants.forEach((v, i) => v.id = i);
    updateVariantsList();
    
    // Chart neu zeichnen falls vorhanden
    if (currentResults) {
        showChart(currentResults);
    }
}

function toggleVariant(id) {
    const variant = variants.find(v => v.id === id);
    if (variant) {
        variant.visible = !variant.visible;
        
        // Chart-Sichtbarkeit updaten falls Chart bereits vorhanden
        if (chartInstance && currentResults) {
            updateChartVisibility();
        }
    }
}

function updateChartVisibility() {
    if (!chartInstance) return;
    
    const updates = {};
    
    variants.forEach((variant, idx) => {
        // Hauptmarker (Classic/Hidden) ein/ausblenden
        const classicTraceName = variant.name + ' Classic';
        const hiddenTraceName = variant.name + ' Hidden';
        
        chartInstance.data.forEach((trace, traceIdx) => {
            if (trace.name === classicTraceName || trace.name === hiddenTraceName) {
                updates[`visible[${traceIdx}]`] = variant.visible;
            }
        });
        
        // Pfeile ein/ausblenden (nur für Nicht-Basis-Varianten)
        if (idx > 0 && arrowTraces[variant.id]) {
            arrowTraces[variant.id].forEach(arrowTraceIdx => {
                updates[`visible[${arrowTraceIdx}]`] = variant.visible;
            });
        }
    });
    
    if (Object.keys(updates).length > 0) {
        Plotly.restyle('mainChart', updates);
    }
}

// Analysis
async function runAnalysis() {
    const analyzeBtn = document.getElementById('analyzeBtn');
    
    if (!sessionId) {
        alert('Bitte erst eine Datei laden!');
        return;
    }
    if (variants.length === 0) {
        alert('Bitte mindestens eine Variante hinzufügen!');
        return;
    }
    
    analyzeBtn.disabled = true;
    analyzeBtn.innerHTML = '⏳ Analysiere...';
    showMessage('messageContainer', 'Analyse läuft mit Python-Modulen...', 'loading');
    
    try {
        const response = await fetch('/api/analyze', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                session_id: sessionId, 
                variants: variants
            })
        });
        const data = await response.json();
        
        if (data.success) {
            currentResults = data;
            showChart(data);
            showMessage('messageContainer', '✅ Analyse abgeschlossen', 'success');
        } else {
            showMessage('messageContainer', data.error || 'Analyse fehlgeschlagen', 'error');
        }
    } catch (error) {
        showMessage('messageContainer', 'Analyse fehlgeschlagen: ' + error.message, 'error');
    } finally {
        analyzeBtn.disabled = false;
        analyzeBtn.innerHTML = '🚀 Analyse starten';
    }
}

// Chart Display
function showChart(data) {
    document.getElementById('resultsPanel').style.display = 'block';
    
    const traces = [];
    let traceIndex = 0;
    arrowTraces = {}; // Reset arrow traces
    
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
    traceIndex++;
    
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
        traceIndex++;
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
        traceIndex++;
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
        traceIndex++;
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
        traceIndex++;
    }
    
    // RSI mit Überkauft/Überverkauft-Linien
    if (data.chartData.rsi) {
        traces.push({
            x: data.chartData.dates,
            y: data.chartData.rsi,
            type: 'scatter',
            mode: 'lines',
            name: 'RSI',
            yaxis: 'y2',
            line: {color: '#FFA500', width: 2}
        });
        traceIndex++;
        
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
        traceIndex++;
        
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
        traceIndex++;
    }
    
    // MACD Histogram
    if (data.chartData.macd_histogram) {
        traces.push({
            x: data.chartData.dates,
            y: data.chartData.macd_histogram,
            type: 'bar',
            name: 'MACD Histogram',
            yaxis: 'y3',
            marker: {color: data.chartData.macd_histogram.map(v => v > 0 ? '#00FF00' : '#FF0000')}
        });
        traceIndex++;
    }
    
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
        const res = data.results[v.id];
        if (!res) return;
        
        // Array für Pfeil-Traces dieser Variante
        arrowTraces[v.id] = [];
        
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
                    `${v.name}<br>Classic Divergence<br>` +
                    `Stärke: ${d.validation?.strength_score || 0}%<br>` +
                    `RSI: ${d.rsi?.toFixed(2) || 'N/A'}<br>` +
                    `MACD: ${d.macd?.toFixed(4) || 'N/A'}<br>` +
                    `Klick für Details`
                ),
                hovertemplate: '%{text}<extra></extra>',
                yaxis: 'y',
                visible: v.visible,
                customdata: res.classic.map(d => ({
                    variant: v,
                    divergence: d,
                    type: 'classic'
                }))
            });
            traceIndex++;
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
                    `${v.name}<br>Hidden Divergence<br>` +
                    `Stärke: ${d.validation?.strength_score || 0}%<br>` +
                    `RSI: ${d.rsi?.toFixed(2) || 'N/A'}<br>` +
                    `MACD: ${d.macd?.toFixed(4) || 'N/A'}<br>` +
                    `Klick für Details`
                ),
                hovertemplate: '%{text}<extra></extra>',
                yaxis: 'y',
                visible: v.visible,
                customdata: res.hidden.map(d => ({
                    variant: v,
                    divergence: d,
                    type: 'hidden'
                }))
            });
            traceIndex++;
        }
        
        // Zusätzliche Marker (Pfeile) für neue Divergenzen
        if (idx > 0 && additionalClassic.length > 0) {
            traces.push({
                x: additionalClassic.map(d => d.date),
                y: additionalClassic.map(d => d.low * 0.98), // Etwas unter dem Low
                type: 'scatter',
                mode: 'markers+text',
                name: v.name + ' NEU Classic',
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
                yaxis: 'y',
                visible: v.visible
            });
            arrowTraces[v.id].push(traceIndex);
            traceIndex++;
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
                yaxis: 'y',
                visible: v.visible
            });
            arrowTraces[v.id].push(traceIndex);
            traceIndex++;
        }
    });
    
    const layout = {
        title: {
            text: 'Bullish Divergence Analysis - Klick auf Marker für Validierung',
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
    
    Plotly.newPlot('mainChart', traces, layout, config);
    
    // Chart-Instanz speichern und Event-Handler hinzufügen
    chartInstance = document.getElementById('mainChart');
    
    // Click-Event für Marker-Validierung hinzufügen
    chartInstance.on('plotly_click', function(eventData) {
        console.log('Chart clicked:', eventData); // Debug
        
        if (eventData.points && eventData.points.length > 0) {
            const point = eventData.points[0];
            console.log('Point data:', point); // Debug
            
            if (point.customdata) {
                console.log('Custom data found:', point.customdata); // Debug
                showValidationDetails(point.customdata);
            } else {
                // Fallback: Versuche Divergenz anhand Trace-Name und Index zu finden
                const traceName = point.data.name;
                const pointIndex = point.pointIndex;
                
                console.log('Fallback search:', traceName, pointIndex); // Debug
                
                if (traceName && (traceName.includes('Classic') || traceName.includes('Hidden'))) {
                    findAndShowValidation(traceName, pointIndex, point);
                }
            }
        }
    });
    
    // Plotly-Events debuggen
    chartInstance.on('plotly_hover', function(eventData) {
        console.log('Hover event:', eventData);
    });
    
    // Stats aktualisieren
    updateStats(data.results);
}

// Fallback-Funktion für Validierung
function findAndShowValidation(traceName, pointIndex, point) {
    try {
        // Extrahiere Varianten-Name aus Trace-Name
        const variantName = traceName.replace(' Classic', '').replace(' Hidden', '');
        const isClassic = traceName.includes('Classic');
        
        // Finde die entsprechende Variante
        const variant = variants.find(v => v.name === variantName);
        if (!variant) {
            console.log('Variant not found:', variantName);
            return;
        }
        
        // Finde die entsprechende Divergenz in den Ergebnissen
        const results = currentResults.results[variant.id];
        if (!results) {
            console.log('Results not found for variant:', variant.id);
            return;
        }
        
        const divergences = isClassic ? results.classic : results.hidden;
        if (pointIndex < divergences.length) {
            const divergence = divergences[pointIndex];
            
            showValidationDetails({
                variant: variant,
                divergence: divergence,
                type: isClassic ? 'classic' : 'hidden'
            });
        }
    } catch (error) {
        console.error('Fallback validation failed:', error);
        alert('Validierung fehlgeschlagen. Siehe Console für Details.');
    }
}

// Validation Display
function showValidationDetails(data) {
    console.log('showValidationDetails called with:', data); // Debug
    
    const panel = document.getElementById('validationPanel');
    const content = document.getElementById('validationContent');
    
    if (!panel || !content) {
        console.error('Validation panel elements not found!');
        alert('Validierungs-Panel nicht gefunden! Überprüfe HTML.');
        return;
    }
    
    panel.style.display = 'block';
    panel.scrollIntoView({ behavior: 'smooth' });
    
    const divergence = data.divergence;
    const variant = data.variant;
    const validation = divergence.validation || {};
    
    console.log('Validation data:', validation); // Debug
    
    // Header
    let html = `
        <div class="validation-header">
            <h3 style="color: ${variant.color};">
                ${variant.name} - ${data.type.toUpperCase()} DIVERGENCE
            </h3>
            <div class="validation-meta">
                <span class="validation-date">📅 ${divergence.date.split('T')[0]}</span>
                <span class="validation-strength" style="color: ${getStrengthColor(validation.strength_score || 50)};">
                    💪 Stärke: ${validation.strength_score || 50}%
                </span>
            </div>
        </div>
    `;
    
    // Debug-Informationen
    html += `
        <div class="validation-explanation">
            <h4>🐛 Debug-Info:</h4>
            <pre>Typ: ${data.type}
Variante: ${variant.name} (${variant.window}/${variant.candleTol}/${variant.macdTol})
Datum: ${divergence.date}
Preis: ${divergence.low}
RSI: ${divergence.rsi}
MACD: ${divergence.macd}
Validation verfügbar: ${validation ? 'Ja' : 'Nein'}</pre>
        </div>
    `;
    
    // Explanation
    if (validation.divergence_explanation) {
        html += `
            <div class="validation-explanation">
                <h4>🔍 Divergenz-Erklärung:</h4>
                <pre>${validation.divergence_explanation}</pre>
            </div>
        `;
    } else {
        // Fallback-Erklärung
        const typeText = data.type === 'classic' ? 'Classic (Preis vs. RSI)' : 'Hidden (Preis vs. MACD)';
        html += `
            <div class="validation-explanation">
                <h4>🔍 ${typeText} Divergenz erkannt:</h4>
                <pre>Divergenz-Typ: ${typeText}
Zeitpunkt: ${divergence.date.split('T')[0]}
Preis-Level: ${divergence.low.toFixed(4)}
RSI-Wert: ${divergence.rsi.toFixed(2)}
MACD-Wert: ${divergence.macd.toFixed(4)}

${data.type === 'classic' ? 
  'CLASSIC: Preis macht tieferes Tief, RSI macht höheres Tief (nachlassender Verkaufsdruck)' :
  'HIDDEN: MACD zeigt Momentum-Stärke (unterliegende Bullishness)'
}

Hinweis: Detaillierte Validierung erfordert lokale Minima-Daten.</pre>
            </div>
        `;
    }
    
    // Key Metrics
    html += `
        <div class="validation-metrics">
            <h4>📊 Schlüsselwerte:</h4>
            <div class="metrics-grid">
                <div class="metric-item">
                    <label>Preis (Low):</label>
                    <span>${divergence.low.toFixed(4)}</span>
                </div>
                <div class="metric-item">
                    <label>RSI:</label>
                    <span style="color: #FFA500">${divergence.rsi.toFixed(2)}</span>
                </div>
                <div class="metric-item">
                    <label>MACD Histogram:</label>
                    <span style="color: ${divergence.macd > 0 ? '#00FF00' : '#FF0000'}">${divergence.macd.toFixed(4)}</span>
                </div>
                <div class="metric-item">
                    <label>Varianten-Parameter:</label>
                    <span>W:${variant.window} C:${variant.candleTol}% M:${variant.macdTol}%</span>
                </div>
            </div>
        </div>
    `;
    
    // Context Data Table (falls verfügbar)
    if (validation.context_data && validation.context_data.length > 0) {
        html += `
            <div class="validation-context">
                <h4>🕘 Kontext-Daten (${validation.context_data.length} Kerzen):</h4>
                <div class="context-table-container">
                    <table class="context-table">
                        <thead>
                            <tr>
                                <th>Index</th>
                                <th>Datum</th>
                                <th>Low</th>
                                <th>RSI</th>
                                <th>MACD</th>
                                <th>P-Min</th>
                                <th>R-Min</th>
                                <th>M-Min</th>
                            </tr>
                        </thead>
                        <tbody>
        `;
        
        validation.context_data.forEach(point => {
            const isSignal = point.is_signal;
            const rowClass = isSignal ? 'signal-row' : '';
            const bgColor = isSignal ? 'background: rgba(255,255,0,0.1);' : '';
            
            html += `
                <tr class="${rowClass}" style="${bgColor}">
                    <td>${point.relative_index}${isSignal ? ' 🎯' : ''}</td>
                    <td>${point.date.split('T')[0]}</td>
                    <td>${point.low.toFixed(4)}</td>
                    <td style="color: #FFA500">${point.rsi.toFixed(2)}</td>
                    <td style="color: ${point.macd > 0 ? '#00FF00' : '#FF0000'}">${point.macd.toFixed(4)}</td>
                    <td>${point.is_price_min ? '✅' : ''}</td>
                    <td>${point.is_rsi_min ? '✅' : ''}</td>
                    <td>${point.is_macd_min ? '✅' : ''}</td>
                </tr>
            `;
        });
        
        html += `
                        </tbody>
                    </table>
                </div>
            </div>
        `;
    }
    
    // Local Minima Summary (falls verfügbar)
    if (validation.local_minima && validation.local_minima.length > 0) {
        html += `
            <div class="validation-minima">
                <h4>📍 Lokale Minima (${validation.local_minima.length}):</h4>
                <div class="minima-list">
        `;
        
        validation.local_minima.forEach((minimum, index) => {
            html += `
                <div class="minimum-item">
                    <strong>${minimum.date.split('T')[0]}</strong>: 
                    Low=${minimum.low.toFixed(4)}, 
                    RSI=${minimum.rsi.toFixed(2)}, 
                    MACD=${minimum.macd.toFixed(4)}
                    ${minimum.is_price_min ? ' 📉' : ''}
                    ${minimum.is_rsi_min ? ' 📊' : ''}
                    ${minimum.is_macd_min ? ' 📈' : ''}
                </div>
            `;
        });
        
        html += `
                </div>
            </div>
        `;
    }
    
    // Debug Information
    if (validation.error) {
        html += `
            <div class="validation-error">
                <h4>⚠️ Fehler:</h4>
                <p>${validation.error}</p>
            </div>
        `;
    }
    
    content.innerHTML = html;
}

function getStrengthColor(strength) {
    if (strength >= 80) return '#00FF00';
    if (strength >= 60) return '#FFD700';
    if (strength >= 40) return '#FFA500';
    return '#FF4444';
}

function closeValidation() {
    document.getElementById('validationPanel').style.display = 'none';
}
function updateStats(results) {
    const container = document.getElementById('statsContainer');
    if (!container) return;
    
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

// Export Functions
async function exportResults(format) {
    if (!sessionId || variants.length === 0) {
        alert('Keine Daten zum Exportieren!');
        return;
    }
    
    try {
        const response = await fetch('/api/export', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                format: format,
                results: {}, // Would need to store results from analysis
                variants: variants
            })
        });
        
        const data = await response.json();
        if (data.success) {
            showMessage('messageContainer', `✅ Export erfolgreich: ${data.filename}`, 'success');
        } else {
            showMessage('messageContainer', data.error, 'error');
        }
    } catch (error) {
        showMessage('messageContainer', 'Export fehlgeschlagen', 'error');
    }
}

// Config Management
async function saveConfig() {
    if (variants.length === 0) {
        alert('Keine Varianten zum Speichern!');
        return;
    }
    
    const name = prompt('Name für die Konfiguration:', `config_${new Date().toISOString().slice(0,10)}`);
    if (!name) return;
    
    try {
        const response = await fetch('/api/save_config', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                name: name,
                variants: variants
            })
        });
        
        const data = await response.json();
        if (data.success) {
            showMessage('messageContainer', `✅ Konfiguration gespeichert: ${data.filename}`, 'success');
        } else {
            showMessage('messageContainer', data.error, 'error');
        }
    } catch (error) {
        showMessage('messageContainer', 'Speichern fehlgeschlagen', 'error');
    }
}

async function loadConfig() {
    try {
        const response = await fetch('/api/list_configs');
        const configs = await response.json();
        
        if (configs.length === 0) {
            alert('Keine gespeicherten Konfigurationen gefunden!');
            return;
        }
        
        const name = prompt('Konfiguration laden:\n' + configs.join('\n'), configs[0]);
        if (!name) return;
        
        const loadResponse = await fetch(`/api/load_config/${name}`);
        const data = await loadResponse.json();
        
        if (data.success) {
            variants = data.variants;
            updateVariantsList();
            showMessage('messageContainer', `✅ Konfiguration geladen: ${name}`, 'success');
        } else {
            showMessage('messageContainer', data.error, 'error');
        }
    } catch (error) {
        showMessage('messageContainer', 'Laden fehlgeschlagen', 'error');
    }
}

// Utility Functions
function showMessage(containerId, msg, type) {
    const container = document.getElementById(containerId);
    if (!container) return;
    
    container.innerHTML = `<div class="message ${type}">${msg}</div>`;
    if (type === 'success') {
        setTimeout(() => container.innerHTML = '', 3000);
    }
}

function toggleFullscreen() {
    const chart = document.getElementById('mainChart');
    if (chart.requestFullscreen) {
        chart.requestFullscreen();
    }
}

// Auto-load default variants on page load
window.addEventListener('load', () => {
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
});