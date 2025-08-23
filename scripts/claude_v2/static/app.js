let sessionId = null;
let variants = [];
let variantIdCounter = 1;

// Color palette for variants
const presetColors = {
    'basis': '#90EE90',
    'v1': '#ADD8E6',
    'v2': '#FFA500',
    'v3': '#FF69B4',
    'v4': '#FFFF00'
};
const colors = ['#90EE90', '#ADD8E6', '#FFA500', '#FF69B4', '#FFFF00', '#FF6B6B', '#4ECDC4', '#45B7D1'];

async function checkHealth() {
    console.log("Checking server health...");
    try {
        const response = await fetch('/api/health', { method: 'GET' });
        const data = await response.json();
        const statusDiv = document.getElementById('serverStatus');
        if (data.status === 'ok') {
            statusDiv.innerHTML = 'Server online';
            statusDiv.className = 'status-indicator status-online';
        } else {
            statusDiv.innerHTML = 'Modules missing';
            statusDiv.className = 'status-indicator status-offline';
        }
    } catch (error) {
        console.error("Health check failed:", error.message);
        document.getElementById('serverStatus').innerHTML = 'Server offline';
        document.getElementById('serverStatus').className = 'status-indicator status-offline';
    }
}

document.addEventListener('DOMContentLoaded', function() {
    console.log("DOM loaded, initializing event listeners");
    checkHealth();
    
    const fileInput = document.getElementById('fileInput');
    const fileName = document.getElementById('fileName');
    const clearData = document.getElementById('clearData');
    const analyzeBtn = document.getElementById('analyzeBtn');
    
    if (!fileInput || !fileName || !clearData || !analyzeBtn) {
        console.error("DOM elements missing:", { fileInput, fileName, clearData, analyzeBtn });
        showMessage('uploadMsg', 'UI initialization failed', 'error');
        return;
    }
    
    fileInput.addEventListener('change', async (e) => {
        const file = e.target.files[0];
        if (!file) return;
        
        console.log("File selected:", file.name);
        fileName.textContent = file.name;
        clearData.style.display = 'inline-block';
        showMessage('uploadMsg', 'Loading file...', 'loading');
        
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
                showMessage('uploadMsg', `${data.info.rows} rows loaded`, 'success');
                analyzeBtn.disabled = false;
                console.log("File uploaded successfully, session:", sessionId);
            } else {
                showMessage('uploadMsg', `Upload failed: ${data.error}`, 'error');
            }
        } catch (error) {
            console.error("Upload error:", error);
            showMessage('uploadMsg', `Upload error: ${error.message}`, 'error');
        }
    });
    
    clearData.addEventListener('click', () => {
        fileInput.value = '';
        fileName.textContent = '';
        clearData.style.display = 'none';
        document.getElementById('uploadMsg').innerHTML = '';
        analyzeBtn.disabled = true;
        sessionId = null;
        console.log("Data cleared");
    });
    
    analyzeBtn.addEventListener('click', analyze);
});

function addVariant() {
    const name = document.getElementById('variantName').value;
    const window = parseInt(document.getElementById('window').value);
    const candleTol = parseFloat(document.getElementById('candleTol').value);
    const macdTol = parseFloat(document.getElementById('macdTol').value);
    
    if (!name || isNaN(window) || isNaN(candleTol) || isNaN(macdTol)) {
        showMessage('messageContainer', 'Please fill all parameters!', 'error');
        return;
    }
    
    const variant = {
        id: variantIdCounter++,
        name: name,
        window: window,
        candleTol: candleTol,
        macdTol: macdTol,
        color: presetColors[name.toLowerCase()] || colors[variants.length % colors.length],
        showClassic: true,
        showHidden: true
    };
    
    variants.push(variant);
    updateVariantsList();
    showMessage('messageContainer', `Variant ${name} added`, 'success');
    
    // Clear inputs
    document.getElementById('variantName').value = '';
    
    console.log("Added variant:", variant);
}

function updateVariantsList() {
    const container = document.getElementById('variants');
    container.innerHTML = '';
    
    variants.forEach((variant, index) => {
        const div = document.createElement('div');
        div.className = 'variant';
        div.innerHTML = `
            <span style="color: ${variant.color}; font-weight: bold;">${variant.name}</span>
            <span>W:${variant.window} C:${variant.candleTol}% M:${variant.macdTol}%</span>
            <button class="btn btn-danger" onclick="removeVariant(${index})">Remove</button>
        `;
        container.appendChild(div);
    });
}

function removeVariant(index) {
    variants.splice(index, 1);
    updateVariantsList();
    console.log("Removed variant at index:", index);
}

async function analyze() {
    if (!sessionId || variants.length === 0) {
        showMessage('messageContainer', 'Upload file and add variants first!', 'error');
        return;
    }
    
    console.log("Starting analysis with", variants.length, "variants");
    showMessage('messageContainer', 'Analysis running...', 'loading');
    
    try {
        const response = await fetch('/api/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                session_id: sessionId,
                variants: variants
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            console.log("Analysis completed:", data);
            plotChart(data.chartData, data.results);
            showMessage('messageContainer', 'Analysis complete', 'success');
        } else {
            showMessage('messageContainer', `Analysis failed: ${data.error}`, 'error');
        }
    } catch (error) {
        console.error("Analysis error:", error);
        showMessage('messageContainer', `Analysis error: ${error.message}`, 'error');
    }
}

function plotChart(chartData, results) {
    console.log("Plotting chart with data:", { 
        dates_length: chartData.dates?.length, 
        open_length: chartData.open?.length,
        ema20_available: !!chartData.ema20
    });
    
    const traces = [
        // Candlestick
        {
            x: chartData.dates,
            open: chartData.open,
            high: chartData.high,
            low: chartData.low,
            close: chartData.close,
            type: 'candlestick',
            name: 'OHLC',
            xaxis: 'x',
            yaxis: 'y',
            increasing: { line: { color: '#00FF00' } },
            decreasing: { line: { color: '#FF0000' } },
            showlegend: true
        },
        // RSI
        {
            x: chartData.dates,
            y: chartData.rsi,
            type: 'scatter',
            name: 'RSI',
            xaxis: 'x',
            yaxis: 'y2',
            line: { color: '#FFA500' },
            legendgroup: 'indicators',
            showlegend: true
        },
        // MACD Histogram
        {
            x: chartData.dates,
            y: chartData.macd_histogram,
            type: 'bar',
            name: 'MACD Histogram',
            xaxis: 'x',
            yaxis: 'y3',
            marker: { color: chartData.macd_histogram?.map(v => v >= 0 ? '#00FF00' : '#FF0000') },
            legendgroup: 'indicators',
            showlegend: true
        }
    ];

    // Add EMAs
    if (chartData.ema20) {
        traces.push({
            x: chartData.dates,
            y: chartData.ema20,
            type: 'scatter',
            name: 'EMA 20',
            xaxis: 'x',
            yaxis: 'y',
            line: { color: '#FFD700', width: 1 },
            legendgroup: 'emas',
            showlegend: true
        });
    }
    if (chartData.ema50) {
        traces.push({
            x: chartData.dates,
            y: chartData.ema50,
            type: 'scatter',
            name: 'EMA 50',
            xaxis: 'x',
            yaxis: 'y',
            line: { color: '#00FFFF', width: 1 },
            legendgroup: 'emas',
            showlegend: true
        });
    }
    if (chartData.ema100) {
        traces.push({
            x: chartData.dates,
            y: chartData.ema100,
            type: 'scatter',
            name: 'EMA 100',
            xaxis: 'x',
            yaxis: 'y',
            line: { color: '#FF00FF', width: 1 },
            legendgroup: 'emas',
            showlegend: true
        });
    }
    if (chartData.ema200) {
        traces.push({
            x: chartData.dates,
            y: chartData.ema200,
            type: 'scatter',
            name: 'EMA 200',
            xaxis: 'x',
            yaxis: 'y',
            line: { color: '#9370DB', width: 1 },
            legendgroup: 'emas',
            showlegend: true
        });
    }

    // Add divergence markers for each variant
    variants.forEach(variant => {
        const res = results[variant.id];
        if (!res) return;
        
        const legendgroup = `variant_${variant.id}`;
        
        // Classic markers
        if (variant.showClassic && res.classic.length > 0) {
            traces.push({
                x: res.classic.map(d => d.date),
                y: res.classic.map(d => d.low),
                mode: 'markers',
                type: 'scatter',
                name: `${variant.name} Classic`,
                marker: {
                    size: 12,
                    color: variant.color,
                    symbol: 'triangle-up',
                    line: { color: '#FFFFFF', width: 2 }
                },
                xaxis: 'x',
                yaxis: 'y',
                legendgroup: legendgroup,
                showlegend: true
            });
            traces.push({
                x: res.classic.map(d => d.date),
                y: res.classic.map(d => d.rsi),
                mode: 'markers',
                type: 'scatter',
                name: `${variant.name} RSI`,
                marker: {
                    size: 8,
                    color: variant.color,
                    symbol: 'circle'
                },
                xaxis: 'x',
                yaxis: 'y2',
                legendgroup: legendgroup,
                showlegend: false
            });
            traces.push({
                x: res.classic.map(d => d.date),
                y: res.classic.map(d => d.macd),
                mode: 'markers',
                type: 'scatter',
                name: `${variant.name} MACD`,
                marker: {
                    size: 8,
                    color: variant.color,
                    symbol: 'square'
                },
                xaxis: 'x',
                yaxis: 'y3',
                legendgroup: legendgroup,
                showlegend: false
            });
        }
        
        // Hidden markers
        if (variant.showHidden && res.hidden.length > 0) {
            traces.push({
                x: res.hidden.map(d => d.date),
                y: res.hidden.map(d => d.low),
                mode: 'markers',
                type: 'scatter',
                name: `${variant.name} Hidden`,
                marker: {
                    size: 10,
                    color: variant.color,
                    symbol: 'diamond',
                    line: { color: '#FFFFFF', width: 2 }
                },
                xaxis: 'x',
                yaxis: 'y',
                legendgroup: legendgroup,
                showlegend: true
            });
            traces.push({
                x: res.hidden.map(d => d.date),
                y: res.hidden.map(d => d.rsi),
                mode: 'markers',
                type: 'scatter',
                name: `${variant.name} Hidden RSI`,
                marker: {
                    size: 6,
                    color: variant.color,
                    symbol: 'diamond'
                },
                xaxis: 'x',
                yaxis: 'y2',
                legendgroup: legendgroup,
                showlegend: false
            });
            traces.push({
                x: res.hidden.map(d => d.date),
                y: res.hidden.map(d => d.macd),
                mode: 'markers',
                type: 'scatter',
                name: `${variant.name} Hidden MACD`,
                marker: {
                    size: 6,
                    color: variant.color,
                    symbol: 'diamond'
                },
                xaxis: 'x',
                yaxis: 'y3',
                legendgroup: legendgroup,
                showlegend: false
            });
        }
    });

    const layout = {
        title: {
            text: 'Bullish Divergence Analysis',
            font: { size: 20, color: '#FFFFFF' }
        },
        xaxis: {
            rangeslider: { visible: false },
            domain: [0, 1],
            anchor: 'y3',
            gridcolor: 'rgba(255,255,255,0.1)',
            zerolinecolor: 'rgba(255,255,255,0.2)',
            titlefont: { color: '#FFFFFF' },
            tickfont: { color: '#FFFFFF' }
        },
        yaxis: {
            title: 'Price',
            domain: [0.55, 1],
            gridcolor: 'rgba(255,255,255,0.1)',
            titlefont: { color: '#FFFFFF' },
            tickfont: { color: '#FFFFFF' },
            fixedrange: false
        },
        yaxis2: {
            title: 'RSI',
            domain: [0.28, 0.52],
            anchor: 'x',
            gridcolor: 'rgba(255,255,255,0.1)',
            titlefont: { color: '#FFFFFF' },
            tickfont: { color: '#FFFFFF' },
            fixedrange: false
        },
        yaxis3: {
            title: 'MACD Histogram',
            domain: [0, 0.25],
            anchor: 'x',
            gridcolor: 'rgba(255,255,255,0.1)',
            titlefont: { color: '#FFFFFF' },
            tickfont: { color: '#FFFFFF' },
            fixedrange: false
        },
        plot_bgcolor: '#0a0a0a',
        paper_bgcolor: '#1a1a1a',
        font: { color: '#FFFFFF' },
        showlegend: true,
        legend: {
            orientation: 'h',
            y: 1.02,
            x: 0,
            bgcolor: 'rgba(0,0,0,0.5)',
            font: { color: '#FFFFFF', size: 10 }
        },
        hovermode: 'x unified',
        height: Math.max(600, window.innerHeight - 200)
    };

    const config = {
        responsive: true,
        displayModeBar: true,
        displaylogo: false,
        modeBarButtonsToRemove: ['select2d', 'lasso2d']
    };

    try {
        Plotly.newPlot('mainChart', traces, layout, config);
        console.log("Chart plotted successfully");
        updateStats(results);
    } catch (error) {
        console.error("Plotly error:", error.message);
        showMessage('messageContainer', 'Chart plotting error: ' + error.message, 'error');
    }
}

function updateStats(results) {
    console.log("Updating stats with results:", results);
    const container = document.getElementById('statsContainer');
    let html = '<div class="stats-card"><h3>Results</h3><table style="width: 100%; border-collapse: collapse;">';
    html += '<tr style="border-bottom: 2px solid rgba(255,255,255,0.2);"><th style="text-align: left; padding: 5px; font-size: 12px;">Variant</th><th style="text-align: center; padding: 5px; font-size: 12px;">Classic</th><th style="text-align: center; padding: 5px; font-size: 12px;">Hidden</th><th style="text-align: center; padding: 5px; font-size: 12px;">Total</th></tr>';

    variants.forEach(v => {
        const res = results[v.id] || { classic: [], hidden: [], total: 0 };
        html += `<tr style="border-bottom: 1px solid rgba(255,255,255,0.1);">
            <td style="padding: 5px; color: ${v.color}; font-weight: bold; font-size: 12px;">${v.name.charAt(0).toUpperCase() + v.name.slice(1)}</td>
            <td style="text-align: center; padding: 5px; font-size: 12px;">${res.classic.length}</td>
            <td style="text-align: center; padding: 5px; font-size: 12px;">${res.hidden.length}</td>
            <td style="text-align: center; padding: 5px; font-weight: bold; font-size: 12px;">${res.total}</td>
        </tr>`;
    });

    html += '</table></div>';
    container.innerHTML = html;
}

function showMessage(containerId, message, type) {
    const container = document.getElementById(containerId);
    container.innerHTML = `<div class="message ${type}" style="font-size: 12px;">${message}</div>`;
    if (type === 'success') {
        setTimeout(() => container.innerHTML = '', 3000);
    }
}

function loadPreset(type) {
    console.log("Loading preset:", type);
    document.getElementById('variantName').value = type.charAt(0).toUpperCase() + type.slice(1);
    switch(type) {
        case 'standard':
            document.getElementById('window').value = 5;
            document.getElementById('candleTol').value = 0.1;
            document.getElementById('macdTol').value = 3.25;
            break;
    }
}