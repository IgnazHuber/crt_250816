// WORKING APP.JS - COPIED FROM all_in_one_analyzer.py
let sessionId = null;
let variants = [];
let variantIdCounter = 1;

// Color palette
const colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F'];

// Health check
async function checkHealth() {
    try {
        const response = await fetch('/api/health');
        const data = await response.json();
        
        const statusDiv = document.getElementById('serverStatus');
        if (data.status === 'ok') {
            statusDiv.innerHTML = '✅ Server online';
            statusDiv.className = 'status-indicator status-online';
        } else {
            statusDiv.innerHTML = '❌ Modules missing';
            statusDiv.className = 'status-indicator status-offline';
        }
    } catch (error) {
        const statusDiv = document.getElementById('serverStatus');
        statusDiv.innerHTML = '❌ Server offline';
        statusDiv.className = 'status-indicator status-offline';
    }
}

// File upload
document.addEventListener('DOMContentLoaded', function() {
    const fileInput = document.getElementById('fileInput');
    const fileName = document.getElementById('fileName');
    const clearData = document.getElementById('clearData');
    const analyzeBtn = document.getElementById('analyzeBtn');
    
    checkHealth();
    
    fileInput.addEventListener('change', async (e) => {
        const file = e.target.files[0];
        if (!file) return;
        
        fileName.textContent = file.name;
        clearData.style.display = 'inline-block';
        showMessage('uploadMsg', '⌛ Loading file...', 'loading');
        
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
                showMessage('uploadMsg', `📊 ${data.info.rows} rows loaded`, 'success');
                if (analyzeBtn) analyzeBtn.disabled = false;
            } else {
                showMessage('uploadMsg', `❌ ${data.error}`, 'error');
            }
        } catch (error) {
            showMessage('uploadMsg', `❌ ${error.message}`, 'error');
        }
    });
    
    clearData.addEventListener('click', () => {
        fileInput.value = '';
        fileName.textContent = '';
        clearData.style.display = 'none';
        document.getElementById('uploadMsg').innerHTML = '';
        if (analyzeBtn) analyzeBtn.disabled = true;
        sessionId = null;
    });
});

// Variant management
function addVariant() {
    const name = document.getElementById('variantName').value;
    const window = parseInt(document.getElementById('window').value);
    const candleTol = parseFloat(document.getElementById('candleTol').value);
    const macdTol = parseFloat(document.getElementById('macdTol').value);
    
    if (!name || isNaN(window) || isNaN(candleTol) || isNaN(macdTol)) {
        showMessage('messageContainer', '❌ Please fill all parameters!', 'error');
        return;
    }
    
    const variant = {
        id: variantIdCounter++,
        name,
        window,
        candleTol,
        macdTol,
        visible: true,
        color: colors[(variants.length) % colors.length]
    };
    
    variants.push(variant);
    updateVariantsList();
    showMessage('messageContainer', `✅ Variant ${name} added`, 'success');
}

function updateVariantsList() {
    const variantsList = document.getElementById('variantsList');
    variantsList.innerHTML = '<h4>Parameter Variants:</h4>';
    
    variants.forEach((variant, index) => {
        const div = document.createElement('div');
        div.className = 'variant-item';
        div.style.borderLeftColor = variant.color;
        div.innerHTML = `
            <span>
                <strong>${variant.name}</strong>: 
                Window=${variant.window}, 
                CandleTol=${variant.candleTol}%, 
                MACD Tol=${variant.macdTol}%
            </span>
            <button class="btn btn-danger" onclick="removeVariant(${index})">🗑️</button>
        `;
        variantsList.appendChild(div);
    });
}

function removeVariant(index) {
    variants.splice(index, 1);
    updateVariantsList();
}

// Analysis
async function startAnalysis() {
    if (!sessionId || variants.length === 0) {
        showMessage('messageContainer', '❌ Upload file and add variants first!', 'error');
        return;
    }
    
    showMessage('messageContainer', '⌛ Analysis running...', 'loading');
    
    try {
        const response = await fetch('/api/analyze', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({session_id: sessionId, variants: variants})
        });
        const data = await response.json();
        
        if (data.success) {
            showChart(data);
            showMessage('messageContainer', '✅ Analysis complete', 'success');
        } else {
            showMessage('messageContainer', `❌ Analysis failed: ${data.error}`, 'error');
        }
    } catch (error) {
        showMessage('messageContainer', `❌ Analysis failed: ${error.message}`, 'error');
    }
}

// Chart creation - EXACT COPY FROM WORKING VERSION
function showChart(data) {
    document.getElementById('resultsPanel').style.display = 'block';
    
    const traces = [];
    
    // Candlestick - EXACT COPY
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
    
    // EMAs - EXACT COPY
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
    
    // RSI - EXACT COPY
    traces.push({
        x: data.chartData.dates,
        y: data.chartData.rsi,
        type: 'scatter',
        mode: 'lines',
        name: 'RSI',
        yaxis: 'y2',
        line: {color: '#FFA500', width: 2}
    });
    
    // RSI Reference lines - EXACT COPY
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
    
    // MACD Histogram - EXACT COPY
    traces.push({
        x: data.chartData.dates,
        y: data.chartData.macd_histogram,
        type: 'bar',
        name: 'MACD Histogram',
        yaxis: 'y3',
        marker: {color: data.chartData.macd_histogram.map(v => v > 0 ? '#00FF00' : '#FF0000')}
    });
    
    // Divergence markers - EXACT COPY
    variants.forEach((v, idx) => {
        if (!v.visible) return;
        
        const res = data.results[v.id];
        if (!res) return;
        
        // Classic Divergences
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
        
        // Hidden Divergences
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
    });
    
    // Layout - EXACT COPY FROM WORKING VERSION
    const layout = {
        title: {
            text: 'Bullish Divergence Analysis',
            font: {size: 20, color: '#FFFFFF'}
        },
        dragmode: 'zoom',
        showlegend: true,
        legend: {
            orientation: 'h',
            y: 1.05,
            x: 0,
            bgcolor: 'rgba(0,0,0,0.5)',
            font: {color: '#FFFFFF', size: 10}
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
            fixedrange: false
        },
        yaxis2: {
            title: 'RSI',
            domain: [0.28, 0.52],
            anchor: 'x',
            gridcolor: 'rgba(255,255,255,0.1)',
            titlefont: {color: '#FFFFFF'},
            tickfont: {color: '#FFFFFF'},
            fixedrange: false
        },
        yaxis3: {
            title: 'MACD Histogram',
            domain: [0, 0.25],
            anchor: 'x',
            gridcolor: 'rgba(255,255,255,0.1)',
            titlefont: {color: '#FFFFFF'},
            tickfont: {color: '#FFFFFF'},
            fixedrange: false
        },
        plot_bgcolor: '#0a0a0a',
        paper_bgcolor: '#1a1a1a',
        font: {color: '#FFFFFF'},
        hovermode: 'x unified'
    };
    
    // Config - EXACT COPY
    const config = {
        responsive: true,
        displayModeBar: true,
        displaylogo: false,
        modeBarButtonsToRemove: ['select2d', 'lasso2d']
    };
    
    Plotly.newPlot('mainChart', traces, layout, config);
    
    // Show stats
    updateStats(data.results);
}

function updateStats(results) {
    const container = document.getElementById('statsContainer');
    let html = '<div class="stats-card"><h3>📊 Results</h3><table style="width: 100%; border-collapse: collapse;">';
    html += '<tr style="border-bottom: 2px solid rgba(255,255,255,0.2);"><th style="text-align: left; padding: 10px;">Variant</th><th style="text-align: center; padding: 10px;">Classic</th><th style="text-align: center; padding: 10px;">Hidden</th><th style="text-align: center; padding: 10px;">Total</th></tr>';
    
    variants.forEach(v => {
        const res = results[v.id] || {classic: [], hidden: [], total: 0};
        html += `<tr style="border-bottom: 1px solid rgba(255,255,255,0.1);">
            <td style="padding: 8px; color: ${v.color}; font-weight: bold;">${v.name}</td>
            <td style="text-align: center; padding: 8px;">${res.classic.length}</td>
            <td style="text-align: center; padding: 8px;">${res.hidden.length}</td>
            <td style="text-align: center; padding: 8px; font-weight: bold;">${res.total}</td>
        </tr>`;
    });
    
    html += '</table></div>';
    container.innerHTML = html;
}

// Utility functions
function showMessage(containerId, message, type) {
    const container = document.getElementById(containerId);
    container.innerHTML = `<div class="message ${type}">${message}</div>`;
    if (type === 'success') {
        setTimeout(() => container.innerHTML = '', 3000);
    }
}

// Presets
function loadPreset(type) {
    document.getElementById('variantName').value = type.charAt(0).toUpperCase() + type.slice(1);
    
    switch(type) {
        case 'standard':
            document.getElementById('window').value = 5;
            document.getElementById('candleTol').value = 0.1;
            document.getElementById('macdTol').value = 3.25;
            break;
        case 'conservative':
            document.getElementById('window').value = 7;
            document.getElementById('candleTol').value = 0.05;
            document.getElementById('macdTol').value = 2.0;
            break;
        case 'aggressive':
            document.getElementById('window').value = 3;
            document.getElementById('candleTol').value = 0.2;
            document.getElementById('macdTol').value = 5.0;
            break;
    }
}