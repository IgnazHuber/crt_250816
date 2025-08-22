let sessionId = null;
let variants = [];
let variantIdCounter = 1;

// Color palette for variants - Basis: hellgrün, V1: hellblau, V2: orange, V3: pink, V4: gelb
const presetColors = {
    'basis': '#90EE90',
    'v1': '#ADD8E6',
    'v2': '#FFA500',
    'v3': '#FF69B4',
    'v4': '#FFFF00'
};
const colors = ['#90EE90', '#ADD8E6', '#FFA500', '#FF69B4', '#FFFF00', '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F'];

async function checkHealth() {
    console.log("🔍 Checking server health...");
    try {
        const response = await fetch('/api/health', { method: 'GET' });
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
        console.error("❌ Health check failed:", error.message);
        document.getElementById('serverStatus').innerHTML = '❌ Server offline';
        document.getElementById('serverStatus').className = 'status-indicator status-offline';
    }
}

document.addEventListener('DOMContentLoaded', function() {
    console.log("✅ DOM loaded, initializing event listeners");
    checkHealth();
    
    const fileInput = document.getElementById('fileInput');
    const fileName = document.getElementById('fileName');
    const clearData = document.getElementById('clearData');
    const analyzeBtn = document.getElementById('analyzeBtn');
    
    if (!fileInput || !fileName || !clearData || !analyzeBtn) {
        console.error("❌ DOM elements missing:", { fileInput, fileName, clearData, analyzeBtn });
        showMessage('uploadMsg', '❌ UI initialization failed', 'error');
        return;
    }
    
    fileInput.addEventListener('change', async (e) => {
        console.log("📂 File input changed");
        const file = e.target.files[0];
        if (!file) {
            console.log("❌ No file selected");
            showMessage('uploadMsg', '❌ Keine Datei ausgewählt', 'error');
            return;
        }
        
        fileName.textContent = file.name;
        clearData.style.display = 'inline-block';
        showMessage('uploadMsg', '⌛ Lade Datei...', 'loading');
        
        const formData = new FormData();
        formData.append('file', file);
        
        try {
            console.log("📤 Uploading file:", file.name);
            const response = await fetch('/api/upload', {
                method: 'POST',
                body: formData
            });
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            const data = await response.json();
            
            console.log("📥 Upload response:", data);
            if (data.success) {
                sessionId = data.session_id;
                showMessage('uploadMsg', `📊 ${data.info.rows} Zeilen geladen`, 'success');
                console.log("✅ File uploaded, session_id:", sessionId);
                analyzeBtn.disabled = false;
            } else {
                showMessage('uploadMsg', `❌ ${data.error}`, 'error');
                console.error("❌ Upload error:", data.error);
            }
        } catch (error) {
            showMessage('uploadMsg', `❌ Upload fehlgeschlagen: ${error.message}`, 'error');
            console.error("❌ Upload failed:", error.message);
        }
    });
    
    clearData.addEventListener('click', () => {
        console.log("🗑️ Clearing data");
        fileInput.value = '';
        fileName.textContent = '';
        clearData.style.display = 'none';
        document.getElementById('uploadMsg').innerHTML = '';
        analyzeBtn.disabled = true;
        sessionId = null;
    });

    analyzeBtn.addEventListener('click', () => {
        console.log("🚀 Analyze button clicked");
        console.time("analyze");
        analyze();
    });
});

function addVariant() {
    console.log("➕ Adding variant");
    let name = document.getElementById('variantName').value.toLowerCase();
    if (!name) name = 'basis';
    const window = parseInt(document.getElementById('window').value);
    const candleTol = parseFloat(document.getElementById('candleTol').value);
    const macdTol = parseFloat(document.getElementById('macdTol').value);
    
    if (isNaN(window) || isNaN(candleTol) || isNaN(macdTol)) {
        showMessage('messageContainer', '❌ Bitte alle Felder ausfüllen', 'error');
        console.error("❌ Invalid variant inputs:", { name, window, candleTol, macdTol });
        return;
    }

    const variant = {
        id: variantIdCounter++,
        name: name,
        window: window,
        candleTol: candleTol,
        macdTol: macdTol,
        color: presetColors[name] || colors[(variantIdCounter - 1) % colors.length],
        showClassic: true,
        showHidden: true
    };
    variants.push(variant);
    
    const variantList = document.getElementById('variants');
    const variantDiv = document.createElement('div');
    variantDiv.className = 'variant';
    variantDiv.id = `variant-${variant.id}`;
    variantDiv.innerHTML = `
        <span style="color: ${variant.color}">${variant.name.charAt(0).toUpperCase() + variant.name.slice(1)}</span> 
        (Window: ${window}, Candle: ${candleTol}%, MACD: ${macdTol}%)
        <label><input type="checkbox" class="show-classic" data-id="${variant.id}" checked> Classic</label>
        <label><input type="checkbox" class="show-hidden" data-id="${variant.id}" checked> Hidden</label>
        <button class="btn btn-danger" onclick="removeVariant(${variant.id})">🗑️</button>
    `;
    variantList.appendChild(variantDiv);
    showMessage('messageContainer', '✅ Variante hinzugefügt', 'success');
    console.log("✅ Variant added:", { id: variant.id, name: variant.name, color: variant.color, window, candleTol, macdTol });

    document.querySelector(`#variant-${variant.id} .show-classic`).addEventListener('change', (e) => {
        const variantId = parseInt(e.target.dataset.id);
        const variant = variants.find(v => v.id === variantId);
        variant.showClassic = e.target.checked;
        console.log(`🔄 Updated showClassic for variant ${variantId}: ${variant.showClassic}`);
        if (sessionId) analyze();
    });
    document.querySelector(`#variant-${variant.id} .show-hidden`).addEventListener('change', (e) => {
        const variantId = parseInt(e.target.dataset.id);
        const variant = variants.find(v => v.id === variantId);
        variant.showHidden = e.target.checked;
        console.log(`🔄 Updated showHidden for variant ${variantId}: ${variant.showHidden}`);
        if (sessionId) analyze();
    });
}

function removeVariant(id) {
    console.log("🗑️ Removing variant:", id);
    variants = variants.filter(v => v.id !== id);
    const variantList = document.getElementById('variants');
    variantList.innerHTML = '';
    variants.forEach(v => {
        const variantDiv = document.createElement('div');
        variantDiv.className = 'variant';
        variantDiv.id = `variant-${v.id}`;
        variantDiv.innerHTML = `
            <span style="color: ${v.color}">${v.name.charAt(0).toUpperCase() + v.name.slice(1)}</span> 
            (Window: ${v.window}, Candle: ${v.candleTol}%, MACD: ${v.macdTol}%)
            <label><input type="checkbox" class="show-classic" data-id="${v.id}" ${v.showClassic ? 'checked' : ''}> Classic</label>
            <label><input type="checkbox" class="show-hidden" data-id="${v.id}" ${v.showHidden ? 'checked' : ''}> Hidden</label>
            <button class="btn btn-danger" onclick="removeVariant(${v.id})">🗑️</button>
        `;
        variantList.appendChild(variantDiv);
        document.querySelector(`#variant-${v.id} .show-classic`).addEventListener('change', (e) => {
            const variantId = parseInt(e.target.dataset.id);
            const variant = variants.find(v => v.id === variantId);
            variant.showClassic = e.target.checked;
            console.log(`🔄 Updated showClassic for variant ${variantId}: ${variant.showClassic}`);
            if (sessionId) analyze();
        });
        document.querySelector(`#variant-${v.id} .show-hidden`).addEventListener('change', (e) => {
            const variantId = parseInt(e.target.dataset.id);
            const variant = variants.find(v => v.id === variantId);
            variant.showHidden = e.target.checked;
            console.log(`🔄 Updated showHidden for variant ${variantId}: ${variant.showHidden}`);
            if (sessionId) analyze();
        });
    });
}

async function analyze() {
    console.log("📊 Starting analysis...");
    if (!sessionId) {
        showMessage('messageContainer', '❌ Bitte erst eine Datei hochladen', 'error');
        console.error("❌ No sessionId available");
        console.timeEnd("analyze");
        return;
    }
    if (variants.length === 0) {
        showMessage('messageContainer', '❌ Bitte mindestens eine Variante hinzufügen', 'error');
        console.error("❌ No variants defined");
        console.timeEnd("analyze");
        return;
    }

    showMessage('messageContainer', '⌛ Analysiere...', 'loading');
    
    try {
        console.log("📤 Sending analyze request with session_id:", sessionId, "variants:", variants);
        const response = await fetch('/api/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_id: sessionId, variants: variants })
        });
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }
        const data = await response.json();
        
        console.log("📥 Received response:", data);
        if (data.success) {
            showMessage('messageContainer', '✅ Analyse abgeschlossen', 'success');
            plotChart(data.chartData, data.results);
        } else {
            showMessage('messageContainer', `❌ ${data.error}`, 'error');
            console.error("❌ Analysis error:", data.error);
        }
    } catch (error) {
        showMessage('messageContainer', `❌ Analyse fehlgeschlagen: ${error.message}`, 'error');
        console.error("❌ Analyze request failed:", error.message);
    }
    console.timeEnd("analyze");
}

function plotChart(chartData, results) {
    console.log("📈 Plotting chart with data:", { dates_length: chartData.dates.length, first_date: chartData.dates[0], last_date: chartData.dates[chartData.dates.length - 1] });

    // Preserve current zoom and range settings
    const chartDiv = document.getElementById('mainChart');
    let preservedLayout = {};
    if (chartDiv.data) {
        const currentLayout = chartDiv.layout;
        preservedLayout = {
            'xaxis.range': currentLayout.xaxis?.range,
            'yaxis.range': currentLayout.yaxis?.range,
            'yaxis2.range': currentLayout.yaxis2?.range,
            'yaxis3.range': currentLayout.yaxis3?.range
        };
    }

    const traces = [
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
            legendgroup: 'main',
            showlegend: true
        },
        {
            x: chartData.dates,
            y: chartData.rsi,
            type: 'scatter',
            name: 'RSI',
            xaxis: 'x',
            yaxis: 'y2',
            line: { color: '#00FFFF', width: 1.0 },
            legendgroup: 'indicators',
            showlegend: true
        },
        {
            x: chartData.dates,
            y: chartData.macd_histogram,
            type: 'bar',
            name: 'MACD Histogram',
            xaxis: 'x',
            yaxis: 'y3',
            marker: { color: chartData.macd_histogram.map(v => v >= 0 ? '#00FF00' : '#FF0000') },
            legendgroup: 'indicators',
            showlegend: true
        }
    ];

    if (chartData.ema20) {
        traces.push({
            x: chartData.dates,
            y: chartData.ema20,
            type: 'scatter',
            name: 'EMA 20',
            xaxis: 'x',
            yaxis: 'y',
            line: { color: '#FFFF00', dash: 'dot', width: 1.0 },
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
            line: { color: '#FFA500', dash: 'dot', width: 1.0 },
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
            line: { color: '#FF4040', dash: 'dot', width: 1.0 },
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
            line: { color: '#FF0000', dash: 'dot', width: 1.0 },
            legendgroup: 'emas',
            showlegend: true
        });
    }

    // Calculate non-overlapping text positions
    const getTextPosition = (divs, yKey, yFactor) => {
        const positions = [];
        divs.forEach((div, i) => {
            let offset = 0;
            const baseY = div[yKey] * yFactor;
            for (let j = 0; j < i; j++) {
                const prevDiv = divs[j];
                const prevY = prevDiv[yKey] * yFactor;
                if (Math.abs(chartData.dates.indexOf(div.date) - chartData.dates.indexOf(prevDiv.date)) < 5 &&
                    Math.abs(baseY - prevY) < 0.05 * Math.abs(baseY)) {
                    offset += 0.02;
                }
            }
            positions.push({ y: baseY * (1 - offset), position: offset > 0 ? 'bottom center' : 'top center' });
        });
        return positions;
    };

    // Identify Basis variant and log results
    const basisVariant = variants.find(v => v.name.toLowerCase() === 'basis');
    const basisResults = basisVariant ? results[basisVariant.id] || { classic: [], hidden: [] } : { classic: [], hidden: [] };
    console.log("🔍 Basis variant:", { id: basisVariant?.id, name: basisVariant?.name, color: basisVariant?.color }, "Basis results:", basisResults);

    variants.forEach(variant => {
        const res = results[variant.id] || { classic: [], hidden: [] };
        const legendgroup = `variant_${variant.id}`;
        console.log(`🔍 Processing variant ${variant.name}:`, { id: variant.id, color: variant.color, classic: res.classic.length, hidden: res.hidden.length });
        console.log(`🔍 Variant ${variant.name} dates check:`, {
            classic_dates: res.classic.map(d => d.date),
            hidden_dates: res.hidden.map(d => d.date),
            in_chartData: res.classic.concat(res.hidden).map(d => chartData.dates.includes(d.date))
        });
        
        // Classic markers
        if (variant.showClassic && res.classic.length > 0) {
            const textPositions = getTextPosition(res.classic, 'low', 0.99);
            traces.push({
                x: res.classic.map(d => d.date),
                y: res.classic.map(d => d.low),
                mode: 'markers',
                name: `${variant.name} Classic`,
                marker: { symbol: 'triangle-up', size: 10, color: variant.color, line: { color: 'black', width: 1 } },
                hoverinfo: 'text',
                hovertext: res.classic.map(d => `Divergenz ${d.div_id}: ${d.type}, Stärke: ${d.strength.toFixed(2)}`),
                xaxis: 'x',
                yaxis: 'y',
                legendgroup: legendgroup,
                showlegend: true
            });
            traces.push({
                x: res.classic.map(d => d.date),
                y: res.classic.map(d => d.rsi),
                mode: 'markers',
                name: `${variant.name} Classic RSI`,
                marker: { symbol: 'triangle-up', size: 10, color: variant.color, line: { color: 'black', width: 1 } },
                hoverinfo: 'text',
                hovertext: res.classic.map(d => `Divergenz ${d.div_id}: ${d.type}, Stärke: ${d.strength.toFixed(2)}`),
                xaxis: 'x',
                yaxis: 'y2',
                legendgroup: legendgroup,
                showlegend: false
            });
            traces.push({
                x: res.classic.map(d => d.date),
                y: res.classic.map(d => d.macd),
                mode: 'markers',
                name: `${variant.name} Classic MACD`,
                marker: { symbol: 'triangle-up', size: 10, color: variant.color, line: { color: 'black', width: 1 } },
                hoverinfo: 'text',
                hovertext: res.classic.map(d => `Divergenz ${d.div_id}: ${d.type}, Stärke: ${d.strength.toFixed(2)}`),
                xaxis: 'x',
                yaxis: 'y3',
                legendgroup: legendgroup,
                showlegend: false
            });

            // Numbers for divergences
            traces.push({
                x: res.classic.map(d => d.date),
                y: textPositions.map(p => p.y),
                mode: 'text',
                text: res.classic.map(d => d.div_id.toString()),
                textposition: textPositions.map(p => p.position),
                textfont: { color: variant.color, size: 6 },
                hoverinfo: 'text',
                hovertext: res.classic.map(d => `Divergenz ${d.div_id}: ${d.type}, Stärke: ${d.strength.toFixed(2)}`),
                xaxis: 'x',
                yaxis: 'y',
                legendgroup: legendgroup,
                showlegend: false
            });
        }

        // Hidden markers
        if (variant.showHidden && res.hidden.length > 0) {
            const textPositions = getTextPosition(res.hidden, 'low', 0.99);
            traces.push({
                x: res.hidden.map(d => d.date),
                y: res.hidden.map(d => d.low),
                mode: 'markers',
                name: `${variant.name} Hidden`,
                marker: { symbol: 'diamond', size: 10, color: variant.color, line: { color: 'black', width: 1 } },
                hoverinfo: 'text',
                hovertext: res.hidden.map(d => `Divergenz ${d.div_id}: ${d.type}, Stärke: ${d.strength.toFixed(2)}`),
                xaxis: 'x',
                yaxis: 'y',
                legendgroup: legendgroup,
                showlegend: true
            });
            traces.push({
                x: res.hidden.map(d => d.date),
                y: res.hidden.map(d => d.rsi),
                mode: 'markers',
                name: `${variant.name} Hidden RSI`,
                marker: { symbol: 'diamond', size: 10, color: variant.color, line: { color: 'black', width: 1 } },
                hoverinfo: 'text',
                hovertext: res.hidden.map(d => `Divergenz ${d.div_id}: ${d.type}, Stärke: ${d.strength.toFixed(2)}`),
                xaxis: 'x',
                yaxis: 'y2',
                legendgroup: legendgroup,
                showlegend: false
            });
            traces.push({
                x: res.hidden.map(d => d.date),
                y: res.hidden.map(d => d.macd),
                mode: 'markers',
                name: `${variant.name} Hidden MACD`,
                marker: { symbol: 'diamond', size: 10, color: variant.color, line: { color: 'black', width: 1 } },
                hoverinfo: 'text',
                hovertext: res.hidden.map(d => `Divergenz ${d.div_id}: ${d.type}, Stärke: ${d.strength.toFixed(2)}`),
                xaxis: 'x',
                yaxis: 'y3',
                legendgroup: legendgroup,
                showlegend: false
            });

            // Numbers for hidden
            traces.push({
                x: res.hidden.map(d => d.date),
                y: textPositions.map(p => p.y),
                mode: 'text',
                text: res.hidden.map(d => d.div_id.toString()),
                textposition: textPositions.map(p => p.position),
                textfont: { color: variant.color, size: 6 },
                hoverinfo: 'text',
                hovertext: res.hidden.map(d => `Divergenz ${d.div_id}: ${d.type}, Stärke: ${d.strength.toFixed(2)}`),
                xaxis: 'x',
                yaxis: 'y',
                legendgroup: legendgroup,
                showlegend: false
            });
        }

        // Additional and missing markers compared to Basis
        if (basisVariant && variant.id !== basisVariant.id) {
            const basisClassic = basisResults.classic.map(d => d.date);
            const basisHidden = basisResults.hidden.map(d => d.date);
            const currentClassic = variant.showClassic ? res.classic.map(d => d.date) : [];
            const currentHidden = variant.showHidden ? res.hidden.map(d => d.date) : [];

            // Additional markers (in current but not in Basis)
            const additionalClassic = res.classic.filter(d => !basisClassic.includes(d.date));
            const additionalHidden = res.hidden.filter(d => !basisHidden.includes(d.date));
            console.log(`🔍 Additional markers for ${variant.name}:`, { 
                classic: additionalClassic.map(d => ({ div_id: d.div_id, date: d.date, type: d.type, strength: d.strength, y: d.low * 0.92 })),
                hidden: additionalHidden.map(d => ({ div_id: d.div_id, date: d.date, type: d.type, strength: d.strength, y: d.low * 0.92 }))
            });
            if (additionalClassic.length > 0 || additionalHidden.length > 0) {
                traces.push({
                    x: [...additionalClassic.map(d => d.date), ...additionalHidden.map(d => d.date)],
                    y: [...additionalClassic.map(d => d.low * 0.92), ...additionalHidden.map(d => d.low * 0.92)],
                    mode: 'markers+lines',
                    name: `${variant.name} Additional`,
                    marker: { symbol: 'arrow-up', size: 8, color: variant.color, line: { color: '#FFFF00', width: 1.0 } },
                    line: { color: variant.color, width: 1 },
                    hoverinfo: 'text',
                    hovertext: [...additionalClassic.map(d => `Neue Divergenz ${d.div_id}: ${d.type}, Stärke: ${d.strength.toFixed(2)}, Preis: ${d.low.toFixed(2)}, y: ${(d.low * 0.92).toFixed(2)}`), 
                               ...additionalHidden.map(d => `Neue Divergenz ${d.div_id}: ${d.type}, Stärke: ${d.strength.toFixed(2)}, Preis: ${d.low.toFixed(2)}, y: ${(d.low * 0.92).toFixed(2)}`)],
                    xaxis: 'x',
                    yaxis: 'y',
                    legendgroup: legendgroup,
                    showlegend: true
                });
            }

            // Missing markers (in Basis but not in current)
            const missingClassic = basisResults.classic.filter(d => !currentClassic.includes(d.date));
            const missingHidden = basisResults.hidden.filter(d => !currentHidden.includes(d.date));
            console.log(`🔍 Missing markers for ${variant.name}:`, { 
                classic: missingClassic.map(d => ({ div_id: d.div_id, date: d.date, type: d.type, strength: d.strength, y: d.low * 0.85 })),
                hidden: missingHidden.map(d => ({ div_id: d.div_id, date: d.date, type: d.type, strength: d.strength, y: d.low * 0.85 }))
            });
            if (missingClassic.length > 0 || missingHidden.length > 0) {
                traces.push({
                    x: [...missingClassic.map(d => d.date), ...missingHidden.map(d => d.date)],
                    y: [...missingClassic.map(d => d.low * 0.85), ...missingHidden.map(d => d.low * 0.85)],
                    mode: 'markers+lines',
                    name: `${variant.name} Missing`,
                    marker: { symbol: 'arrow-down', size: 8, color: variant.color, line: { color: '#FFFF00', width: 1.0 } },
                    line: { color: variant.color, width: 1, dash: 'dot' },
                    hoverinfo: 'text',
                    hovertext: [...missingClassic.map(d => `Fehlende Divergenz ${d.div_id}: ${d.type}, Stärke: ${d.strength.toFixed(2)}, Preis: ${d.low.toFixed(2)}, y: ${(d.low * 0.85).toFixed(2)}`), 
                               ...missingHidden.map(d => `Fehlende Divergenz ${d.div_id}: ${d.type}, Stärke: ${d.strength.toFixed(2)}, Preis: ${d.low.toFixed(2)}, y: ${(d.low * 0.85).toFixed(2)}`)],
                    xaxis: 'x',
                    yaxis: 'y',
                    legendgroup: legendgroup,
                    showlegend: true
                });
            }
        }

        // Divergence arrows
        const allDivs = [...(variant.showClassic ? res.classic : []), ...(variant.showHidden ? res.hidden : [])];
        allDivs.forEach(div => {
            console.log(`🔍 Processing divergence for ${variant.name}:`, { div_id: div.div_id, date: div.date, type: div.type, strength: div.strength, low: div.low });
            const window = div.window;
            const idx = chartData.dates.indexOf(div.date);
            if (idx === -1) {
                console.warn(`⚠️ Date ${div.date} not found in chartData.dates`);
                return;
            }
            const startIdx = Math.max(0, idx - window);
            const x_start = chartData.dates[startIdx];
            const x_end = div.date;

            const rsi_start = chartData.rsi[startIdx] || 0;
            const rsi_end = div.rsi;
            const idx_diff = idx - startIdx;
            const rsi_gradient = idx_diff > 0 ? Math.abs((rsi_end - rsi_start) / idx_diff) : 0;
            const max_gradient = 10;
            div.strength = Math.min(1.0, rsi_gradient / max_gradient);

            traces.push({
                x: [x_start, x_end],
                y: [chartData.low[startIdx], div.low],
                mode: 'lines+markers+text',
                name: `${variant.name} Div Arrow`,
                line: { color: variant.color, width: 2 },
                marker: { symbol: 'triangle-right-open', size: 8, color: variant.color },
                text: [null, div.type.charAt(0).toUpperCase() + div.type.slice(1)],
                textposition: 'bottom right',
                textfont: { color: variant.color, size: 6 },
                hoverinfo: 'text',
                hovertext: `Divergenz Pfeil ${div.div_id}: ${div.type}, Stärke: ${div.strength.toFixed(2)}`,
                xaxis: 'x',
                yaxis: 'y',
                legendgroup: legendgroup,
                showlegend: false
            });

            traces.push({
                x: [x_start, x_end],
                y: [rsi_start, rsi_end],
                mode: 'lines+markers+text',
                name: `${variant.name} RSI Arrow`,
                line: { color: variant.color, width: 2 },
                marker: { symbol: 'triangle-right-open', size: 8, color: variant.color },
                text: [null, div.type.charAt(0).toUpperCase() + div.type.slice(1)],
                textposition: 'bottom right',
                textfont: { color: variant.color, size: 6 },
                hoverinfo: 'text',
                hovertext: `Divergenz Pfeil ${div.div_id}: ${div.type}, Stärke: ${div.strength.toFixed(2)}`,
                xaxis: 'x',
                yaxis: 'y2',
                legendgroup: legendgroup,
                showlegend: false
            });

            traces.push({
                x: [x_start, x_end],
                y: [chartData.macd_histogram[startIdx] || 0, div.macd],
                mode: 'lines+markers+text',
                name: `${variant.name} MACD Arrow`,
                line: { color: variant.color, width: 2 },
                marker: { symbol: 'triangle-right-open', size: 8, color: variant.color },
                text: [null, div.type.charAt(0).toUpperCase() + div.type.slice(1)],
                textposition: 'bottom right',
                textfont: { color: variant.color, size: 6 },
                hoverinfo: 'text',
                hovertext: `Divergenz Pfeil ${div.div_id}: ${div.type}, Stärke: ${div.strength.toFixed(2)}`,
                xaxis: 'x',
                yaxis: 'y3',
                legendgroup: legendgroup,
                showlegend: false
            });
        });
    });

    // Results table annotation
    let resultsText = 'Results:\n';
    variants.forEach(v => {
        const res = results[v.id] || { classic: [], hidden: [], total: 0 };
        resultsText += `${v.name.charAt(0).toUpperCase() + v.name.slice(1)}: Classic ${res.classic.length}, Hidden ${res.hidden.length}\n`;
    });

    const layout = {
        grid: { rows: 3, columns: 1, pattern: 'independent' },
        xaxis: {
            title: 'Date',
            gridcolor: 'rgba(255,255,255,0.1)',
            titlefont: { color: '#FFFFFF', size: 6 },
            tickfont: { color: '#FFFFFF', size: 6 },
            autorange: true,
            range: preservedLayout['xaxis.range'] || undefined
        },
        yaxis: {
            title: 'Price',
            domain: [0.4, 1],
            gridcolor: 'rgba(255,255,255,0.1)',
            titlefont: { color: '#FFFFFF', size: 6 },
            tickfont: { color: '#FFFFFF', size: 6 },
            fixedrange: false,
            range: preservedLayout['yaxis.range'] || undefined
        },
        yaxis2: {
            title: 'RSI',
            domain: [0.25, 0.4],
            anchor: 'x',
            gridcolor: 'rgba(255,255,255,0.1)',
            titlefont: { color: '#FFFFFF', size: 6 },
            tickfont: { color: '#FFFFFF', size: 6 },
            fixedrange: false,
            range: preservedLayout['yaxis2.range'] || undefined
        },
        yaxis3: {
            title: 'MACD Histogram',
            domain: [0, 0.25],
            anchor: 'x',
            gridcolor: 'rgba(255,255,255,0.1)',
            titlefont: { color: '#FFFFFF', size: 6 },
            tickfont: { color: '#FFFFFF', size: 6 },
            fixedrange: false,
            range: preservedLayout['yaxis3.range'] || undefined
        },
        plot_bgcolor: '#0a0a0a',
        paper_bgcolor: '#1a1a1a',
        font: { color: '#FFFFFF', size: 6 },
        hovermode: 'x unified',
        hoverlabel: { bgcolor: 'rgba(0,0,0,0)', font: { size: 6 } },
        legend: {
            x: 0,
            xanchor: 'left',
            y: 1.05,
            yanchor: 'bottom',
            orientation: 'h',
            font: { size: 5 },
            bgcolor: 'rgba(0,0,0,0.5)',
            bordercolor: 'rgba(255,255,255,0.1)',
            borderwidth: 1,
            itemsizing: 'constant',
            itemwidth: 50,
            tracegroupgap: 0,
            itemspacing: 0.5,
            width: document.getElementById('mainChart').offsetWidth || 800,
            tracewidth: 1
        },
        annotations: [{
            text: resultsText,
            x: 0,
            xref: 'paper',
            y: 1,
            yref: 'paper',
            showarrow: false,
            font: { color: '#FFFFFF', size: 6 },
            bgcolor: 'rgba(0,0,0,0.7)',
            bordercolor: 'rgba(255,255,255,0.3)',
            borderwidth: 1,
            padding: 3
        }],
        height: window.innerHeight - 150
    };

    const config = {
        responsive: true,
        displayModeBar: true,
        displaylogo: false,
        modeBarButtonsToRemove: ['select2d', 'lasso2d']
    };

    try {
        // Use Plotly.react to preserve data and traces
        if (chartDiv.data) {
            Plotly.react('mainChart', traces, layout, config);
            Plotly.relayout('mainChart', preservedLayout);
        } else {
            Plotly.newPlot('mainChart', traces, layout, config);
        }
        console.log("✅ Chart plotted successfully");
        updateStats(results);
    } catch (error) {
        console.error("❌ Plotly error:", error.message);
        showMessage('messageContainer', '❌ Fehler beim Plotten des Charts: ' + error.message, 'error');
    }
}

function updateStats(results) {
    console.log("📊 Updating stats with results:", results);
    const container = document.getElementById('statsContainer');
    let html = '<div class="stats-card"><h3 style="font-size: 12px;">📊 Results</h3><table style="width: 100%; border-collapse: collapse;">';
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
    console.log("📋 Loading preset:", type);
    document.getElementById('variantName').value = type.charAt(0).toUpperCase() + type.slice(1);
    switch(type) {
        case 'standard':
            document.getElementById('window').value = 5;
            document.getElementById('candleTol').value = 0.1;
            document.getElementById('macdTol').value = 3.25;
            break;
    }
}