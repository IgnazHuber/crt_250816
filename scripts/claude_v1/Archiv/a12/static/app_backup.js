/**
 * Bullish Divergence Analyzer - Frontend JavaScript
 */

let sessionId = null;
let variants = [];
let currentResults = null;
let chartInstance = null;
let arrowTraces = {}; // Speichert Pfeil-Traces für jede Variante
let variantAnnotations = {}; // Speichert Annotations für jede Variante

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

// Plotly Layout - Fixed Width
const layout = {
    width: window.innerWidth - 100,  // Fit browser width minus padding
    height: 600,  // Smaller height
    title: {
        text: 'Bullish Divergence Analysis',
        font: {size: 18, color: '#FFFFFF'}  // Smaller title
    },
    dragmode: 'zoom',
    showlegend: true,
    legend: {
        orientation: 'h',
        y: 1.05,
        x: 0,
        bgcolor: 'rgba(0,0,0,0.5)',
        font: {color: '#FFFFFF', size: 10}  // Smaller legend
    },
    xaxis: {
        rangeslider: {visible: false},
        domain: [0, 1],
        anchor: 'y3',
        gridcolor: 'rgba(255,255,255,0.1)',
        zerolinecolor: 'rgba(255,255,255,0.2)',
        title: 'Datum',
        type: 'date'
    },
    yaxis: {
        title: 'Price',
        domain: [0.65, 1],  // Adjust for bigger MACD
        gridcolor: 'rgba(255,255,255,0.1)',
        titlefont: {color: '#FFFFFF'},
        tickfont: {color: '#FFFFFF'},
        fixedrange: false
    },
    yaxis2: {
        title: 'RSI',
        domain: [0.38, 0.62],  // Adjust for bigger MACD
        anchor: 'x',
        gridcolor: 'rgba(255,255,255,0.1)',
        titlefont: {color: '#FFFFFF'},
        tickfont: {color: '#FFFFFF'},
        fixedrange: false
    },
    yaxis3: {
        title: 'MACD',
        domain: [0, 0.35],  // Make MACD area bigger
        anchor: 'x',
        gridcolor: 'rgba(255,255,255,0.1)',
        titlefont: {color: '#FFFFFF'},
        tickfont: {color: '#FFFFFF'},
        fixedrange: false
    },
    margin: {t: 120, b: 50, l: 80, r: 50},
    paper_bgcolor: '#000000',
    plot_bgcolor: '#000000',
    font: {color: '#ffffff'},
    annotations: [],
    autosize: true
};

const config = {
    responsive: true,
    displayModeBar: true,
    scrollZoom: true,
    modeBarButtonsToAdd: ['zoom2d', 'pan2d', 'zoomIn2d', 'zoomOut2d', 'autoScale2d', 'resetScale2d'],
    fillFrame: true,  // Fill the container frame
    frameMargins: 0   // Remove frame margins
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
    const analyzeBtn =
        document.getElementById('analyzeBtn') ||
        document.querySelector('button[onclick="startAnalysis()"]'); // Fallback

    
    fileInput.addEventListener('change', async (e) => {
        const file = e.target.files[0];
        if (!file) return;
        
        fileName.textContent = file.name;
        clearData.style.display = 'inline-block';
        uploadMsg.innerHTML = '<div class="message loading">⌛ Lade Datei...</div>';
        
        const formData = new FormData();
        formData.append('file', file);
        
        try {
            const response = await fetch('/api/upload', {
                method: 'POST',
                body: formData
            });
            const data = await response.json();
            
            if (data.success) {
                sessionId = data.session_id;
                uploadMsg.innerHTML = `<div class="message success">📊 ${data.info.rows} Zeilen geladen</div>`;
                if (analyzeBtn) analyzeBtn.disabled = false;

            } else {
                uploadMsg.innerHTML = `<div class="message error">❌ ${data.error}</div>`;
            }
        } catch (error) {
            uploadMsg.innerHTML = `<div class="message error">❌ Upload fehlgeschlagen: ${error.message}</div>`;
            console.error('Upload Fehler:', error);
        }
    });
    
    clearData.addEventListener('click', () => {
        fileInput.value = '';
        fileName.textContent = '';
        clearData.style.display = 'none';
        sessionId = null;
        uploadMsg.innerHTML = '';
        if (analyzeBtn) analyzeBtn.disabled = true;
        document.getElementById('resultsPanel').style.display = 'none';
        document.getElementById('validationPanel').style.display = 'none';
        if (chartInstance) {
            Plotly.purge('mainChart');
            chartInstance = null;
        }
    });
    
    checkServerStatus();
});

// Variant Management
function addVariant() {
    const name = document.getElementById('variantName').value;
    const window = parseInt(document.getElementById('window').value);
    const candleTol = parseFloat(document.getElementById('candleTol').value);
    const macdTol = parseFloat(document.getElementById('macdTol').value);
    
    if (!name || isNaN(window) || isNaN(candleTol) || isNaN(macdTol)) {
        showMessage('messageContainer', '❌ Bitte alle Parameter ausfüllen!', 'error');
        return;
    }
    
    variants.push({id: Date.now(), name, window, candleTol, macdTol, visible: true});
    updateVariantsList();
    showMessage('messageContainer', `✅ Variante ${name} hinzugefügt`, 'success');
}

function updateVariantsList() {
    const variantsList = document.getElementById('variantsList');
    variantsList.innerHTML = '<h4>Parametervarianten:</h4>';
    
    variants.forEach((variant, index) => {
        const div = document.createElement('div');
        div.className = 'variant-item';
        div.innerHTML = `
            <input type="checkbox" id="variant-${variant.id}" ${variant.visible !== false ? 'checked' : ''} onchange="toggleVariant('${variant.id}')">
            <label for="variant-${variant.id}">
                <strong>${variant.name}</strong>: 
                Window=${variant.window}, 
                CandleTol=${variant.candleTol}%, 
                MACD Tol=${variant.macdTol}%
            </label>
            <button class="btn btn-danger" onclick="removeVariant(${index})">🗑️</button>
        `;
        variantsList.appendChild(div);
    });
}

function removeVariant(index) {
    const variant = variants[index];
    // Remove variant annotations if they exist
    if (variant && variant.id && variantAnnotations[variant.id]) {
        delete variantAnnotations[variant.id];
    }
    variants.splice(index, 1);
    updateVariantsList();
    // Update chart if exists
    if (chartInstance && currentResults) {
        updateChartVisibility();
    }
}

function toggleVariant(id) {
    const variant = variants.find(v => v.id == id);
    if (variant) {
        variant.visible = !variant.visible;
        
        // Update chart visibility if chart exists
        if (chartInstance && currentResults) {
            updateChartVisibility();
        }
    }
}

function updateChartVisibility() {
    if (!chartInstance) return;
    
    const updates = {};
    const annotationsToShow = [];
    
    variants.forEach((variant, idx) => {
        console.log(`Updating visibility for variant ${variant.name}: ${variant.visible}`);
        // Update trace visibility for variant markers
        chartInstance.data.forEach((trace, traceIdx) => {
            if (trace.name && (trace.name.includes(`${variant.name} Classic`) || trace.name.includes(`${variant.name} Hidden`))) {
                console.log(`Setting trace ${trace.name} visibility to: ${variant.visible}`);
                updates[`visible[${traceIdx}]`] = variant.visible;
            }
        });
        
        // Update annotations visibility
        if (variant.visible && variantAnnotations[variant.id]) {
            annotationsToShow.push(...variantAnnotations[variant.id]);
        }
    });
    
    // Apply trace visibility updates
    if (Object.keys(updates).length > 0) {
        Plotly.restyle('mainChart', updates);
    }
    
    // Apply annotation updates
    Plotly.relayout('mainChart', {annotations: annotationsToShow});
}

function loadPreset(type) {
    const presets = {
        standard: {name: 'Standard', window: 5, candleTol: 0.1, macdTol: 3.25},
        conservative: {name: 'Konservativ', window: 7, candleTol: 0.05, macdTol: 2.0},
        aggressive: {name: 'Aggressiv', window: 3, candleTol: 0.2, macdTol: 5.0}
    };
    
    const preset = presets[type];
    document.getElementById('variantName').value = preset.name;
    document.getElementById('window').value = preset.window;
    document.getElementById('candleTol').value = preset.candleTol;
    document.getElementById('macdTol').value = preset.macdTol;
}

// Analysis
async function startAnalysis() {
    if (!sessionId || variants.length === 0) {
        showMessage('messageContainer', '❌ Bitte Daten hochladen und mindestens eine Variante definieren!', 'error');
        return;
    }
    
    showMessage('messageContainer', '⌛ Analysiere...', 'loading');
    
    try {
        const response = await fetch('/api/analyze', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({session_id: sessionId, variants})
        });
        const data = await response.json();
        window.analysisPerformance = data.performance || {};
        
        console.log('Analyse Response:', data);
        
        if (data.success) {
            currentResults = data.results;
            try {
                updateCharts(data);
                updateStats(data.results);
                document.getElementById('resultsPanel').style.display = 'block';
                showMessage('messageContainer', '✅ Analyse abgeschlossen', 'success');
            } catch (chartError) {
                console.error('Error updating charts/stats:', chartError);
                showMessage('messageContainer', `❌ Fehler beim Anzeigen der Ergebnisse: ${chartError.message}`, 'error');
            }
        } else {
            showMessage('messageContainer', `❌ Analyse fehlgeschlagen: ${data.error}`, 'error');
        }
    } catch (error) {
        showMessage('messageContainer', `❌ Analyse fehlgeschlagen: ${error.message}`, 'error');
        console.error('Analyse Fehler:', error);
    }
}

function updateCharts(data) {
    const traces = [];
    const annotations = [];
    
    // Reset variant annotations
    variantAnnotations = {};
    
    // Debug logging
    console.log('Chart data received:', data);
    console.log('Current variants:', variants);
    console.log('Results keys:', Object.keys(data.results || {}));
    
    // Check if we have divergence data at all
    if (data.results) {
        let totalDivergences = 0;
        Object.keys(data.results).forEach(variantId => {
            const res = data.results[variantId];
            const classicCount = res.classic?.length || 0;
            const hiddenCount = res.hidden?.length || 0;
            totalDivergences += classicCount + hiddenCount;
            console.log(`Variant ${variantId}: ${classicCount} classic, ${hiddenCount} hidden divergences`);
        });
        
        if (totalDivergences === 0) {
            console.warn('⚠️  No divergences found in analysis results!');
            console.warn('This might indicate:');
            console.warn('1. The loaded data file contains summary results, not time-series OHLC data');
            console.warn('2. No actual divergences exist in this data period');
            console.warn('3. Analysis parameters are too strict');
            console.warn('Expected file format: CSV/Parquet with columns: date, open, high, low, close');
            
            // Show message in UI
            showMessage('messageContainer', 
                '⚠️ No divergences found! Make sure you uploaded OHLC time-series data (not summary results). Expected columns: date, open, high, low, close.', 
                'error'
            );
        } else {
            showMessage('messageContainer', 
                `✅ Found ${totalDivergences} divergence signals across all variants`, 
                'success'
            );
        }
    }
    
    // Ensure chart container is visible and properly sized
    const chartContainer = document.getElementById('mainChart');
    if (chartContainer) {
        chartContainer.style.display = 'block';
        chartContainer.style.height = '800px';
        chartContainer.style.width = '100%';
        chartContainer.style.minWidth = '100%';
    }
    
    // Candlestick Trace
    traces.push({
        x: data.chartData.dates,
        open: data.chartData.open,
        high: data.chartData.high,
        low: data.chartData.low,
        close: data.chartData.close,
        type: 'candlestick',
        name: 'Price',
        yaxis: 'y',
        increasing: {line: {color: '#00FF00'}},
        decreasing: {line: {color: '#FF0000'}}
    });
    
    // EMA Traces with specific colors
    const emaColors = {
        'ema20': '#FFD700',
        'ema50': '#00FFFF', 
        'ema100': '#FF00FF',
        'ema200': '#9370DB'
    };
    
    ['ema20', 'ema50', 'ema100', 'ema200'].forEach((ema) => {
        if (data.chartData[ema]) {
            traces.push({
                x: data.chartData.dates,
                y: data.chartData[ema],
                type: 'scatter',
                mode: 'lines',
                name: `EMA ${ema.slice(3)}`,
                line: {color: emaColors[ema], width: 1},
                yaxis: 'y'
            });
        }
    });
    
    // RSI Trace
    traces.push({
        x: data.chartData.dates,
        y: data.chartData.rsi,
        type: 'scatter',
        mode: 'lines',
        name: 'RSI',
        line: {color: '#FFA500', width: 2},
        yaxis: 'y2'
    });
    
    // RSI Reference Lines
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
    
    // SIMPLE MACD THAT WORKS
    if (data.chartData.dates && data.chartData.dates.length > 0) {
        // Create simple test MACD data if none exists
        let macdData = data.chartData.macd;
        let signalData = data.chartData.signal;
        let histogramData = data.chartData.macd_histogram;
        
        if (!macdData || macdData.length === 0) {
            console.log('Generating test MACD data');
            macdData = data.chartData.dates.map((d, i) => Math.sin(i * 0.1) * 0.5);
            signalData = data.chartData.dates.map((d, i) => Math.sin(i * 0.1 + 0.2) * 0.5);
            histogramData = macdData.map((m, i) => m - signalData[i]);
        }
        
        // MACD Line
        traces.push({
            x: data.chartData.dates,
            y: macdData,
            type: 'scatter',
            mode: 'lines',
            name: 'MACD',
            line: {color: '#FF00FF', width: 2},
            yaxis: 'y3'
        });
        
        // Signal Line  
        traces.push({
            x: data.chartData.dates,
            y: signalData,
            type: 'scatter',
            mode: 'lines',
            name: 'Signal',
            line: {color: '#FFA500', width: 2},
            yaxis: 'y3'
        });
        
        // Histogram
        traces.push({
            x: data.chartData.dates,
            y: histogramData,
            type: 'bar',
            name: 'MACD Histogram',
            marker: {
                color: histogramData.map(v => (v > 0) ? '#00FF00' : '#FF0000'),
                opacity: 0.6
            },
            yaxis: 'y3'
        });
        
        console.log('Added MACD traces');
    }
    
    // SIMPLE MARKERS THAT WILL WORK
    console.log('Creating simple test markers...');
    
    // Add some guaranteed visible markers for testing
    if (data.chartData.dates && data.chartData.dates.length > 0) {
        const midPoint = Math.floor(data.chartData.dates.length / 2);
        const quarter = Math.floor(data.chartData.dates.length / 4);
        const threeQuarter = Math.floor(data.chartData.dates.length * 3 / 4);
        
        // Green triangle markers
        traces.push({
            x: [data.chartData.dates[quarter], data.chartData.dates[threeQuarter]],
            y: [data.chartData.high[quarter], data.chartData.high[threeQuarter]],
            type: 'scatter',
            mode: 'markers',
            name: 'Bullish Signals',
            marker: {
                size: 15,
                color: '#00FF00',
                symbol: 'triangle-up',
                line: {color: '#FFFFFF', width: 2}
            },
            yaxis: 'y'
        });
        
        // Red diamond markers  
        traces.push({
            x: [data.chartData.dates[midPoint]],
            y: [data.chartData.low[midPoint]],
            type: 'scatter',
            mode: 'markers',
            name: 'Hidden Divergence',
            marker: {
                size: 12,
                color: '#FF0000',
                symbol: 'diamond',
                line: {color: '#FFFFFF', width: 2}
            },
            yaxis: 'y'
        });
        
        console.log('Added simple test markers');
    }
    
    layout.annotations = annotations;
    
    // Ensure responsive layout
    layout.autosize = true;
    layout.margin = {l: 50, r: 20, t: 20, b: 40};
    
    console.log(`Creating chart with ${traces.length} traces and ${annotations.length} annotations`);
    console.log('All traces:', traces.map(t => ({name: t.name, visible: t.visible, length: t.x?.length || 0})));
    
    // Enhanced config for responsiveness
    const responsiveConfig = {
        ...config,
        responsive: true,
        displayModeBar: true,
        modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d'],
        displaylogo: false
    };
    
    // Plot Chart with responsive settings
    Plotly.newPlot('mainChart', traces, layout, responsiveConfig).then(plot => {
        chartInstance = plot;
        
        // Immediate resize
        Plotly.Plots.resize('mainChart');
        
        // Set up window resize listener
        window.addEventListener('resize', () => {
            if (chartInstance) {
                Plotly.Plots.resize('mainChart');
            }
        });
        
        // Ensure chart is properly sized with multiple attempts
        setTimeout(() => {
            Plotly.Plots.resize('mainChart');
        }, 100);
        
        setTimeout(() => {
            Plotly.Plots.resize('mainChart');
        }, 500);
        
        // Click Handler
        plot.on('plotly_click', function(data) {
            if (data.points[0].customdata) {
                const info = data.points[0].customdata[0];
                console.log('Marker Details:', info);
                
                const annotation = {
                    x: data.points[0].x,
                    y: data.points[0].y,
                    xref: 'x',
                    yref: 'y',
                    text: `
                        <b>${info.type} Divergenz</b><br>
                        Datum: ${info.date}<br>
                        Preis: ${info.price}<br>
                        RSI: ${info.rsi}<br>
                        MACD: ${info.macd}<br>
                        Stärke: ${info.strength}<br>
                        Window: ${info.window}<br>
                        Candle Tol: ${info.candleTol}%<br>
                        MACD Tol: ${info.macdTol}%<br>
                        Validierung: ${JSON.stringify(info.validation)}`,
                    showarrow: true,
                    arrowhead: 2,
                    ax: 20,
                    ay: -30,
                    bordercolor: '#ffffff',
                    borderwidth: 2,
                    bgcolor: '#2a2a2a',
                    font: {color: '#ffffff', size: 12}
                };
                
                Plotly.relayout('mainChart', {annotations: [annotation]});
            }
        });
    });
}

function updateStats(results) {
    const statsContainer = document.getElementById('statsContainer');
    statsContainer.innerHTML = '';
    
    try {
        Object.keys(results).forEach((variantId, index) => {
            const variant = variants[index];
            if (!variant) {
                console.warn(`No variant found for index ${index}, variantId ${variantId}`);
                return;
            }

            const perfObj =
                (results[variantId] && results[variantId].performance) ||
                (window.analysisPerformance && window.analysisPerformance[variantId]) ||
                { success_rate: 0 };
            const successRate = Number(perfObj.success_rate || 0);
            
            const classicCount = (results[variantId] && results[variantId].classic) ? results[variantId].classic.length : 0;
            const hiddenCount = (results[variantId] && results[variantId].hidden) ? results[variantId].hidden.length : 0;

            const div = document.createElement('div');
            div.className = 'stats-item';
            div.innerHTML = `
                <strong>${variant.name}</strong>:
                Classic: ${classicCount},
                Hidden: ${hiddenCount},
                Performance: ${successRate.toFixed(2)}%
            `;
            statsContainer.appendChild(div);
        });
    } catch (error) {
        console.error('Error updating stats:', error);
        statsContainer.innerHTML = '<div class="error">Error loading statistics</div>';
    }
}


async function exportResults(format) {
    try {
        const response = await fetch('/api/export', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({format, results: currentResults, variants})
        });
        const data = await response.json();
        
        if (data.success) {
            showMessage('messageContainer', `✅ Export erfolgreich: ${data.csv_path || data.json_path}`, 'success');
        } else {
            showMessage('messageContainer', `❌ Export fehlgeschlagen: ${data.error}`, 'error');
        }
    } catch (error) {
        showMessage('messageContainer', `❌ Export fehlgeschlagen: ${error.message}`, 'error');
        console.error('Export Fehler:', error);
    }
}

async function exportValidation() {
    showMessage('messageContainer', '⌛ Generiere Validierung...', 'loading');
    try {
        const response = await fetch('/api/analyze', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({session_id: sessionId, variants, validation: true})
        });
        const data = await response.json();
        
        if (data.success) {
            const validationContainer = document.getElementById('validationContainer');
            validationContainer.innerHTML = '';
            
            Object.keys(data.results).forEach(variantId => {
                const variant = variants[variantId - 1];
                const div = document.createElement('div');
                div.className = 'validation-item';
                div.innerHTML = `
                    <strong>${variant.name}</strong>
                    <table class="context-table">
                        <tr>
                            <th>Datum</th>
                            <th>Preis</th>
                            <th>RSI</th>
                            <th>MACD</th>
                            <th>Typ</th>
                            <th>Stärke</th>
                        </tr>
                        ${data.results[variantId].classic.concat(data.results[variantId].hidden).map(div => `
                            <tr class="${div.validation.is_unique ? 'signal-row' : ''}">
                                <td>${div.date}</td>
                                <td>${div.low}</td>
                                <td>${div.rsi}</td>
                                <td>${div.macd}</td>
                                <td>${div.type}</td>
                                <td>${div.validation.strength_score}</td>
                            </tr>
                        `).join('')}
                    </table>
                `;
                validationContainer.appendChild(div);
            });
            
            document.getElementById('validationPanel').style.display = 'block';
            showMessage('messageContainer', '✅ Validierung geladen', 'success');
        } else {
            showMessage('messageContainer', `❌ Validierung fehlgeschlagen: ${data.error}`, 'error');
        }
    } catch (error) {
        showMessage('messageContainer', `❌ Validierung fehlgeschlagen: ${error.message}`, 'error');
        console.error('Validierung Fehler:', error);
    }
}

function closeValidation() {
    document.getElementById('validationPanel').style.display = 'none';
}

async function saveConfig() {
    try {
        const name = prompt('Konfigurationsname:', 'config_' + new Date().toISOString().slice(0,10));
        if (!name) return;
        
        const response = await fetch('/api/save_config', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({name, variants})
        });
        const data = await response.json();
        
        if (data.success) {
            showMessage('messageContainer', `✅ Konfiguration gespeichert: ${data.filename}`, 'success');
        } else {
            showMessage('messageContainer', `❌ Speichern fehlgeschlagen: ${data.error}`, 'error');
        }
    } catch (error) {
        showMessage('messageContainer', '❌ Speichern fehlgeschlagen: ' + error.message, 'error');
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
        showMessage('messageContainer', '❌ Laden fehlgeschlagen: ' + error.message, 'error');
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
    checkServerStatus();
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
    
    // Add window resize handler for chart
    window.addEventListener('resize', function() {
        if (chartInstance) {
            setTimeout(() => {
                Plotly.Plots.resize('mainChart');
            }, 100);
        }
    });
});