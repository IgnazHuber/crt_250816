/**
 * Bullish Divergence Analyzer - Frontend JavaScript
 */

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

// Plotly Layout
const layout = {
    xaxis: {title: 'Datum', type: 'date', autorange: true},
    yaxis: {title: 'Preis', domain: [0.35, 1], autorange: true, fixedrange: false},
    yaxis2: {title: 'RSI', domain: [0.15, 0.3], autorange: true, fixedrange: false},
    yaxis3: {title: 'MACD', domain: [0, 0.1], autorange: true, fixedrange: false},
    margin: {t: 50, b: 50, l: 50, r: 50},
    showlegend: true,
    legend: {x: 0, y: 1.1, orientation: 'h'},
    paper_bgcolor: '#1a1a1a',
    plot_bgcolor: '#1a1a1a',
    font: {color: '#ffffff'},
    annotations: []
};

const config = {
    responsive: true,
    displayModeBar: true,
    scrollZoom: true,
    modeBarButtonsToAdd: ['zoom2d', 'pan2d', 'zoomIn2d', 'zoomOut2d', 'autoScale2d', 'resetScale2d']
};

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
        console.error('Health Check Fehler:', error);
    }
}

// File Upload Handler
document.addEventListener('DOMContentLoaded', function() {
    const fileInput = document.getElementById('fileInput');
    const fileName = document.getElementById('fileName');
    const uploadMsg = document.getElementById('uploadMsg');
    const clearData = document.getElementById('clearData');
    const analyzeBtn = document.getElementById('analyzeBtn');
    
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
                console.log('Upload erfolgreich:', data);
            } else {
                showMessage('uploadMsg', data.error === 'Ungültiger Dateityp' ? 
                    '❌ Nur CSV- und Parquet-Dateien sind erlaubt' : `❌ ${data.error}`, 'error');
                console.error('Upload Fehler:', data.error);
            }
        } catch (error) {
            showMessage('uploadMsg', 'Upload fehlgeschlagen: ' + error.message, 'error');
            console.error('Upload Fehler:', error);
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
    
    analyzeBtn.addEventListener('click', async () => {
        if (!sessionId) {
            showMessage('messageContainer', '❌ Keine Daten geladen. Bitte lade zuerst eine Datei hoch.', 'error');
            return;
        }
        if (variants.length === 0) {
            showMessage('messageContainer', '❌ Keine Varianten definiert. Bitte füge mindestens eine Variante hinzu.', 'error');
            return;
        }
        
        showMessage('messageContainer', '🔍 Analyse wird durchgeführt...', 'loading');
        
        try {
            const response = await fetch('/api/analyze', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    session_id: sessionId,
                    variants: variants,
                    date_range: null
                })
            });
            const data = await response.json();
            
            console.log('Analyse Response:', data);
            
            if (data.success) {
                currentResults = data;
                showMessage('messageContainer', '✅ Analyse abgeschlossen', 'success');
                document.getElementById('resultsPanel').style.display = 'block';
                document.getElementById('validationPanel').style.display = 'block';
                updateCharts(data.chartData, data.results, data.performance, variants);
            } else {
                showMessage('messageContainer', `❌ Analyse fehlgeschlagen: ${data.error}`, 'error');
                console.error('Analyse Fehler:', data.error);
            }
        } catch (error) {
            showMessage('messageContainer', 'Analyse fehlgeschlagen: ' + error.message, 'error');
            console.error('Analyse Fehler:', error);
        }
    });
    
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
        document.getElementById('variantName').value = preset.name;
        document.getElementById('window').value = preset.window;
        document.getElementById('candleTol').value = preset.candleTol;
        document.getElementById('macdTol').value = preset.macdTol;
        addVariant();
    }
}

// Variante hinzufügen
function addVariant() {
    const variantName = document.getElementById('variantName').value;
    const window = parseInt(document.getElementById('window').value);
    const candleTol = parseFloat(document.getElementById('candleTol').value);
    const macdTol = parseFloat(document.getElementById('macdTol').value);
    
    if (!variantName || isNaN(window) || isNaN(candleTol) || isNaN(macdTol)) {
        showMessage('messageContainer', '❌ Bitte alle Felder korrekt ausfüllen', 'error');
        return;
    }
    
    const variant = {
        id: variants.length + 1,
        name: variantName,
        window: window,
        candleTol: candleTol,
        macdTol: macdTol,
        calculate_performance: true
    };
    
    variants.push(variant);
    updateVariantsList();
    showMessage('messageContainer', `✅ Variante ${variantName} hinzugefügt`, 'success');
}

// Variantenliste aktualisieren
function updateVariantsList() {
    const container = document.getElementById('variantsList');
    let html = '';
    
    variants.forEach(variant => {
        html += `<div class="variant-item">
            <span>${variant.name} (Window: ${variant.window}, Candle: ${variant.candleTol}, MACD: ${variant.macdTol})</span>
            <button class="btn btn-danger" onclick="removeVariant(${variant.id})">🗑️</button>
        </div>`;
    });
    
    container.innerHTML = html;
}

// Variante entfernen
function removeVariant(id) {
    variants = variants.filter(v => v.id !== id);
    updateVariantsList();
    showMessage('messageContainer', '✅ Variante entfernt', 'success');
}

// Chart aktualisieren
function updateCharts(chartData, results, performance, variants) {
    const traces = [];
    
    // Candlestick-Trace
    traces.push({
        x: chartData.dates,
        open: chartData.open,
        high: chartData.high,
        low: chartData.low,
        close: chartData.close,
        type: 'candlestick',
        name: 'Preis',
        yaxis: 'y'
    });
    
    // EMA-Traces
    ['ema12', 'ema20', 'ema26', 'ema50', 'ema100', 'ema200'].forEach((ema, index) => {
        if (chartData[ema]) {
            traces.push({
                x: chartData.dates,
                y: chartData[ema],
                type: 'scatter',
                mode: 'lines',
                name: ema.toUpperCase(),
                line: {color: colors[index % colors.length]},
                yaxis: 'y'
            });
        }
    });
    
    // RSI-Trace
    if (chartData.rsi) {
        traces.push({
            x: chartData.dates,
            y: chartData.rsi,
            type: 'scatter',
            mode: 'lines',
            name: 'RSI',
            line: {color: colors[4]},
            yaxis: 'y2'
        });
    }
    
    // MACD-Trace
    if (chartData.macd_histogram) {
        traces.push({
            x: chartData.dates,
            y: chartData.macd_histogram,
            type: 'bar',
            name: 'MACD Histogram',
            marker: {color: colors[5]},
            yaxis: 'y3'
        });
    }
    
    // Divergenz-Marker und Vergleich zur Basis
    const baseVariant = variants.find(v => v.name === 'Standard');
    const baseVariantId = baseVariant ? baseVariant.id : null;
    const baseDates = baseVariantId ? new Set([
        ...results[baseVariantId].classic.map(div => div.date),
        ...results[baseVariantId].hidden.map(div => div.date)
    ]) : new Set();
    
    variants.forEach((variant, index) => {
        const result = results[variant.id];
        console.log(`Divergenzen für ${variant.name}:`, result);
        ['classic', 'hidden'].forEach(divType => {
            if (result[divType]) {
                result[divType].forEach(div => {
                    const isBaseOnly = variant.name === 'Standard';
                    const isUniqueToVariant = !isBaseOnly && !baseDates.has(div.date);
                    const markerColor = isUniqueToVariant ? colors[(index + 2) % colors.length] : colors[index % colors.length];
                    const markerSymbol = divType === 'classic' ? 'triangle-up' : 'diamond';
                    const shortExplanation = [
                        `Art: ${divType === 'classic' ? 'Klassische' : 'Versteckte'} Divergenz`,
                        `Grund: ${divType === 'classic' ? 'Niedrigerer Tiefpunkt im Preis, höherer in RSI/MACD' : 'Höherer Tiefpunkt im Preis, niedrigerer in RSI/MACD'}`,
                        `Stärke: ${div.validation.strength_score.toFixed(2)}`
                    ];
                    const detailedExplanation = [
                        `Divergenz: ${divType === 'classic' ? 'Klassische' : 'Versteckte'} Bullish Divergenz`,
                        `Datum: ${div.date}`,
                        `Preis-Tiefpunkt: ${div.low.toFixed(2)}`,
                        `RSI: ${div.rsi.toFixed(2)}`,
                        `MACD: ${div.macd.toFixed(2)}`,
                        `Validierung: ${div.validation.status} (${div.validation.message})`,
                        `Stärke: ${div.validation.strength_score.toFixed(2)}`,
                        `Niedrigerer Tiefpunkt: ${div.lower_low.toFixed(2)} (Datum: ${div.lower_low_date})`,
                        `Höherer Tiefpunkt: ${div.higher_low.toFixed(2)} (Datum: ${div.higher_low_date})`,
                        `Datumsabstand: ${div.date_gap.toFixed(2)} Tage`
                    ];
                    traces.push({
                        x: [div.date],
                        y: [div.low],
                        type: 'scatter',
                        mode: 'markers',
                        name: `${variant.name} ${divType}${isUniqueToVariant ? ' (Unique)' : ''}`,
                        marker: {
                            symbol: markerSymbol,
                            size: 12,
                            color: markerColor,
                            line: {width: 2, color: '#ffffff'}
                        },
                        yaxis: 'y',
                        hoverinfo: 'text',
                        text: [shortExplanation.join('<br>')],
                        customdata: [{
                            short: shortExplanation,
                            detailed: detailedExplanation,
                            variant: variant.name,
                            divType: divType,
                            isUnique: isUniqueToVariant
                        }]
                    });
                });
            }
        });
    });
    
    // Layout mit Annotationen
    const updatedLayout = { ...layout, annotations: [] };
    variants.forEach(variant => {
        const result = results[variant.id];
        if (result?.annotations) {
            updatedLayout.annotations.push(...(result.annotations.candlestick || []));
            updatedLayout.annotations.push(...(result.annotations.rsi || []));
            updatedLayout.annotations.push(...(result.annotations.macd || []));
        }
    });
    
    // Plotly-Chart rendern
    Plotly.newPlot('mainChart', traces, updatedLayout, config).then(() => {
        chartInstance = document.getElementById('mainChart');
        console.log('Chart gerendert mit', traces.length, 'Traces');
        
        // Klick-Handler für Marker
        chartInstance.on('plotly_click', (data) => {
            const point = data.points[0];
            if (point.customdata && point.customdata[0]) {
                const { short, detailed, variant, divType, isUnique } = point.customdata[0];
                const infoText = `
                    <b>${variant} - ${divType.charAt(0).toUpperCase() + divType.slice(1)} Divergenz${isUnique ? ' (Unique)' : ''}</b><br>
                    <b>Kurz:</b><br>${short.join('<br>')}<br>
                    <b>Details:</b><br>${detailed.join('<br>')}
                `;
                console.log('Marker geklickt:', { short, detailed, variant, divType, isUnique });
                
                // Anzeige links oben im Chart
                Plotly.relayout('mainChart', {
                    annotations: updatedLayout.annotations.concat([{
                        x: point.x,
                        y: point.y,
                        xref: 'x',
                        yref: 'y',
                        text: infoText,
                        showarrow: false,
                        font: {size: 12, color: '#ffffff'},
                        bgcolor: '#333333',
                        bordercolor: '#ffffff',
                        borderwidth: 1,
                        xanchor: 'left',
                        yanchor: 'top'
                    }])
                });
            }
        });
    });
    
    // Ergebnisse anzeigen
    const statsContainer = document.getElementById('statsContainer');
    const validationContainer = document.getElementById('validationContainer');
    let statsHtml = '';
    let validationHtml = '';
    
    variants.forEach(variant => {
        const result = results[variant.id];
        const perf = performance[variant.id] || {classic: 0, hidden: 0};
        statsHtml += `<div class="stats-item">
            <h4>${variant.name}</h4>
            <p>Classic Divergences: ${result.classic?.length || 0}</p>
            <p>Hidden Divergences: ${result.hidden?.length || 0}</p>
            <p>Total: ${(result.classic?.length || 0) + (result.hidden?.length || 0)}</p>
            <p>Performance Classic: ${perf.classic.toFixed(2)}%</p>
            <p>Performance Hidden: ${perf.hidden.toFixed(2)}%</p>
        </div>`;
        
        ['classic', 'hidden'].forEach(divType => {
            if (result[divType]) {
                result[divType].forEach(div => {
                    validationHtml += `<div class="validation-item">
                        <h5>${variant.name} - ${divType.charAt(0).toUpperCase() + divType.slice(1)}</h5>
                        <p>Date: ${div.date}</p>
                        <p>Low: ${div.low.toFixed(2)}</p>
                        <p>RSI: ${div.rsi.toFixed(2)}</p>
                        <p>MACD: ${div.macd.toFixed(2)}</p>
                        <p>Strength: ${div.validation?.strength_score?.toFixed(2) || 'N/A'}</p>
                        <p>Validation: ${div.validation?.status || 'N/A'} (${div.validation?.message || 'N/A'})</p>
                    </div>`;
                });
            }
        });
    });
    
    statsContainer.innerHTML = statsHtml;
    validationContainer.innerHTML = validationHtml;
}

// Export Functions
async function exportResults(format) {
    if (!sessionId || !currentResults || variants.length === 0) {
        showMessage('messageContainer', '❌ Keine Daten zum Exportieren!', 'error');
        return;
    }
    
    try {
        const response = await fetch('/api/export', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                format: format,
                results: currentResults,
                variants: variants
            })
        });
        
        const data = await response.json();
        if (data.success) {
            showMessage('messageContainer', `✅ Export erfolgreich: ${data.csv_path || data.json_path}`, 'success');
        } else {
            showMessage('messageContainer', `❌ Export fehlgeschlagen: ${data.error}`, 'error');
            console.error('Export Fehler:', data.error);
        }
    } catch (error) {
        showMessage('messageContainer', 'Export fehlgeschlagen: ' + error.message, 'error');
        console.error('Export Fehler:', error);
    }
}

// Config Management
async function saveConfig() {
    if (variants.length === 0) {
        showMessage('messageContainer', '❌ Keine Varianten zum Speichern!', 'error');
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
            showMessage('messageContainer', `❌ Speichern fehlgeschlagen: ${data.error}`, 'error');
            console.error('Config Save Fehler:', data.error);
        }
    } catch (error) {
        showMessage('messageContainer', 'Speichern fehlgeschlagen: ' + error.message, 'error');
        console.error('Config Save Fehler:', error);
    }
}

async function loadConfig() {
    try {
        const response = await fetch('/api/list_configs');
        const configs = await response.json();
        
        if (configs.length === 0) {
            showMessage('messageContainer', '❌ Keine gespeicherten Konfigurationen gefunden!', 'error');
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
            showMessage('messageContainer', `❌ Laden fehlgeschlagen: ${data.error}`, 'error');
            console.error('Config Load Fehler:', data.error);
        }
    } catch (error) {
        showMessage('messageContainer', 'Laden fehlgeschlagen: ' + error.message, 'error');
        console.error('Config Load Fehler:', error);
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