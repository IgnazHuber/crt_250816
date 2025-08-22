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

// Plotly Layout
const layout = {
    xaxis: {title: 'Datum', type: 'date'},
    yaxis: {title: 'Preis', domain: [0.35, 1]},
    yaxis2: {title: 'RSI', domain: [0.15, 0.3]},
    yaxis3: {title: 'MACD', domain: [0, 0.1]},
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
    displayModeBar: true
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
    console.log('Updating chart visibility for variants:', variants);
    
    const updates = {};
    
    variants.forEach((variant, idx) => {
        const classicTraceName = variant.name + ' Classic';
        const hiddenTraceName = variant.name + ' Hidden';
        const newClassicTraceName = variant.name + ' NEU Classic';
        const newHiddenTraceName = variant.name + ' NEU Hidden';
        const validationClassicTraceName = variant.name + ' Classic Validation';
        const validationHiddenTraceName = variant.name + ' Hidden Validation';
        const extremaStrengthName = 'Extrema Strength';
        
        chartInstance.data.forEach((trace, traceIdx) => {
            if (
                trace.name === classicTraceName ||
                trace.name === hiddenTraceName ||
                trace.name === newClassicTraceName ||
                trace.name === newHiddenTraceName ||
                trace.name === validationClassicTraceName ||
                trace.name === validationHiddenTraceName ||
                trace.name === extremaStrengthName
            ) {
                updates[`visible[${traceIdx}]`] = variant.visible;
            }
        });
    });
    
    console.log('Visibility updates:', updates);
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
    console.log('showChart called with data:', data);
    
    document.getElementById('resultsPanel').style.display = 'block';
    
    const traces = [];
    let traceIndex = 0;
    arrowTraces = {}; // Reset arrow traces
    
    // Candlestick
    console.log('Adding Candlestick trace');
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
    ['ema20', 'ema50', 'ema100', 'ema200'].forEach(ema => {
        if (data.chartData[ema]) {
            console.log(`Adding ${ema} trace`);
            traces.push({
                x: data.chartData.dates,
                y: data.chartData[ema],
                type: 'scatter',
                mode: 'lines',
                name: ema.toUpperCase(),
                line: {color: colors[traceIndex % colors.length], width: 1},
                xaxis: 'x',
                yaxis: 'y'
            });
            traceIndex++;
        }
    });
    
    // RSI mit Überkauft/Überverkauft-Linien
    if (data.chartData.rsi) {
        console.log('Adding RSI trace');
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
        console.log('Adding MACD Histogram trace');
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
        console.log('Base Divergences:', baseDivergences.length);
    }
    
    // Divergenz-Marker und Pfeile
    variants.forEach((v, idx) => {
        const res = data.results[v.id];
        if (!res) {
            console.warn(`No results for variant ${v.name}`);
            return;
        }
        
        console.log(`Processing variant ${v.name}:`, res);
        arrowTraces[v.id] = [];
        
        let additionalClassic = [];
        let additionalHidden = [];
        
        if (idx > 0 && baseDivergences) {
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
            console.log(`Additional Classic: ${additionalClassic.length}, Hidden: ${additionalHidden.length}`);
        }
        
        // Classic Divergences
        if (res.classic && res.classic.length > 0) {
            console.log(`Adding Classic Divergence markers for ${v.name}`);
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
            
            // Validierungsmarker
            traces.push({
                x: res.classic.map(d => d.date),
                y: res.classic.map(d => d.low * (d.validation_result?.status === 'success' ? 0.97 : 0.96)),
                type: 'scatter',
                mode: 'markers',
                name: v.name + ' Classic Validation',
                marker: {
                    size: 15,
                    color: res.classic.map(d => d.validation_result?.status === 'success' ? '#00FF00' : '#FF0000'),
                    symbol: res.classic.map(d => d.validation_result?.status === 'success' ? 'triangle-up' : 'cross'),
                    line: {color: '#FFFFFF', width: 2},
                    opacity: 1
                },
                showlegend: false,
                yaxis: 'y',
                visible: v.visible
            });
            traceIndex++;
        }
        
        // Hidden Divergences
        if (res.hidden && res.hidden.length > 0) {
            console.log(`Adding Hidden Divergence markers for ${v.name}`);
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
            
            // Validierungsmarker
            traces.push({
                x: res.hidden.map(d => d.date),
                y: res.hidden.map(d => d.low * (d.validation_result?.status === 'success' ? 0.97 : 0.96)),
                type: 'scatter',
                mode: 'markers',
                name: v.name + ' Hidden Validation',
                marker: {
                    size: 15,
                    color: res.hidden.map(d => d.validation_result?.status === 'success' ? '#00FF00' : '#FF0000'),
                    symbol: res.hidden.map(d => d.validation_result?.status === 'success' ? 'diamond' : 'cross'),
                    line: {color: '#FFFFFF', width: 2},
                    opacity: 1
                },
                showlegend: false,
                yaxis: 'y',
                visible: v.visible
            });
            traceIndex++;
        }
        
        // Zusätzliche Marker (Pfeile)
        if (idx > 0 && additionalClassic.length > 0) {
            console.log(`Adding additional Classic arrows for ${v.name}`);
            traces.push({
                x: additionalClassic.map(d => d.date),
                y: additionalClassic.map(d => d.low * 0.98),
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
            console.log(`Adding additional Hidden arrows for ${v.name}`);
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
        
        // RSI und MACD Annotationen
        if (res.annotations && typeof res.annotations === 'object') {
            console.log(`Adding annotations for ${v.name}:`, res.annotations);
            const candlestickAnnotations = Array.isArray(res.annotations.candlestick) ? res.annotations.candlestick : [];
            const rsiAnnotations = Array.isArray(res.annotations.rsi) ? res.annotations.rsi : [];
            const macdAnnotations = Array.isArray(res.annotations.macd) ? res.annotations.macd : [];
            
            layout.annotations.push(...candlestickAnnotations);
            layout.annotations.push(...rsiAnnotations);
            layout.annotations.push(...macdAnnotations);
            
            // RSI-Marker als Scatter-Trace
            if (rsiAnnotations.length > 0) {
                traces.push({
                    x: rsiAnnotations.map(a => a.x),
                    y: rsiAnnotations.map(a => a.y),
                    type: 'scatter',
                    mode: 'markers',
                    name: v.name + ' RSI Markers',
                    marker: {
                        size: 12,
                        color: v.color,
                        symbol: 'circle',
                        line: {color: '#FFFFFF', width: 1}
                    },
                    showlegend: false,
                    yaxis: 'y2',
                    visible: v.visible
                });
                arrowTraces[v.id].push(traceIndex);
                traceIndex++;
            }
            
            // MACD-Marker als Scatter-Trace
            if (macdAnnotations.length > 0) {
                traces.push({
                    x: macdAnnotations.map(a => a.x),
                    y: macdAnnotations.map(a => a.y),
                    type: 'scatter',
                    mode: 'markers',
                    name: v.name + ' MACD Markers',
                    marker: {
                        size: 12,
                        color: v.color,
                        symbol: 'circle',
                        line: {color: '#FFFFFF', width: 1}
                    },
                    showlegend: false,
                    yaxis: 'y3',
                    visible: v.visible
                });
                arrowTraces[v.id].push(traceIndex);
                traceIndex++;
            }
        } else {
            console.warn(`Keine oder ungültige Annotationsdaten für Variante ${v.name}:`, res.annotations);
        }
    });
    
    // Extrema-Stärke
    if (data.chartData.extrema_strength) {
        console.log('Adding Extrema Strength trace');
        const extremaIndices = data.chartData.extrema_strength
            .map((strength, idx) => strength > 0 ? idx : -1)
            .filter(idx => idx >= 0);
        
        traces.push({
            x: extremaIndices.map(idx => data.chartData.dates[idx]),
            y: extremaIndices.map(idx => data.chartData.low[idx] * 0.95),
            type: 'scatter',
            mode: 'text',
            text: extremaIndices.map(idx => `S:${data.chartData.extrema_strength[idx].toFixed(1)}`),
            textposition: 'bottom center',
            textfont: {size: 12, color: '#FFFFFF'},
            name: 'Extrema Strength',
            showlegend: false,
            yaxis: 'y',
            visible: true
        });
        traceIndex++;
    }
    
    console.log('Final traces:', traces);
    console.log('Final layout:', layout);
    
    Plotly.newPlot('mainChart', traces, layout, config);
    
    chartInstance = document.getElementById('mainChart');
    chartInstance.on('plotly_click', function(eventData) {
        console.log('Chart clicked:', eventData);
        if (eventData.points && eventData.points.length > 0) {
            const point = eventData.points[0];
            if (point.customdata) {
                showValidationDetails(point.customdata);
            } else if (point.data.name && (point.data.name.includes('Classic') || point.data.name.includes('Hidden'))) {
                findAndShowValidation(point.data.name, point.pointIndex, point);
            } else {
                console.warn('No divergence info for point:', point);
            }
        }
    });
    
    updateStats(data.results);
}

function findAndShowValidation(traceName, pointIndex, point) {
    console.log(`Finding validation for trace: ${traceName}, index: ${pointIndex}`);
    
    const variantName = traceName.split(' ')[0];
    const type = traceName.includes('Classic') ? 'classic' : 'hidden';
    const variant = variants.find(v => v.name === variantName);
    
    if (!variant || !currentResults) {
        console.warn('Variant or results not found');
        return;
    }
    
    const divergences = currentResults.results[variant.id][type];
    const divergence = divergences[pointIndex];
    
    if (divergence) {
        showValidationDetails({
            variant: variant,
            divergence: divergence,
            type: type
        });
    } else {
        console.warn('Divergence not found for point:', point);
    }
}

// Validation Details
function showValidationDetails(data) {
    console.log('showValidationDetails called with:', data);
    
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
    
    const typeText = data.type === 'classic' ? 'Classic' : 'Hidden';
    
    let html = `
        <div class="validation-header">
            <h3 style="color: ${variant.color};">
                ${variant.name} - ${typeText.toUpperCase()} DIVERGENCE
            </h3>
            <div class="validation-meta">
                <span class="validation-date">📅 ${divergence.date.split('T')[0]}</span>
                <span class="validation-strength" style="color: ${getStrengthColor(validation.strength_score || 50)};">
                    💪 Stärke: ${validation.strength_score || 50}%
                </span>
            </div>
        </div>
    `;
    
    // Kurze Erklärung
    html += `
        <div class="validation-short">
            <h4>🔍 Kurzübersicht:</h4>
            <ul>
                <li>Typ: ${data.type === 'classic' ? 'Classic (Preis vs. RSI)' : 'Hidden (Preis vs. MACD)'}</li>
                <li>Signal: ${data.type === 'classic' ? 'Tieferes Preis-Tief, höheres RSI-Tief' : 'Momentum-Stärke im MACD'}</li>
                <li>Stärke: ${validation.strength_score || 50}% (${validation.data_quality || 'unbekannt'})</li>
            </ul>
        </div>
    `;
    
    // Ausführliche Erklärung
    if (validation.divergence_explanation) {
        html += `
            <div class="validation-explanation">
                <h4>📜 Detaillierte Erklärung:</h4>
                <pre>${validation.divergence_explanation}</pre>
            </div>
        `;
    } else {
        html += `
            <div class="validation-explanation">
                <h4>📜 Detaillierte Erklärung:</h4>
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
    
    // Validierungsergebnis
    if (divergence.validation_result) {
        html += `
            <div class="validation-performance">
                <h4>📈 Performance-Validierung:</h4>
                <p>Status: <span style="color: ${divergence.validation_result.status === 'success' ? '#00FF00' : '#FF0000'}">
                    ${divergence.validation_result.status === 'success' ? '✅ Erfolgreich' : '❌ Fehlgeschlagen'}
                </span></p>
                <p>${divergence.validation_result.message}</p>
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
    if (!sessionId || !currentResults || variants.length === 0) {
        alert('Keine Daten zum Exportieren!');
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
            showMessage('messageContainer', data.error, 'error');
        }
    } catch (error) {
        showMessage('messageContainer', 'Export fehlgeschlagen: ' + error.message, 'error');
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
        showMessage('messageContainer', 'Speichern fehlgeschlagen: ' + error.message, 'error');
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
        showMessage('messageContainer', 'Laden fehlgeschlagen: ' + error.message, 'error');
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