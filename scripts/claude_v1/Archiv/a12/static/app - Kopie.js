// Bullish Divergence Analyzer - Frontend JavaScript

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
}

function toggleVariant(id) {
    const variant = variants.find(v => v.id === id);
    if (variant) {
        variant.visible = !variant.visible;
        // Chart would be redrawn after analysis
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
    
    Plotly.newPlot('mainChart', traces, layout, config);
    
    // Stats aktualisieren
    updateStats(data.results);
}

// Statistics
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