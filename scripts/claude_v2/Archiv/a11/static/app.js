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
        showHidden: true,
        showBullish: true,
        showBearish: true,
        showNames: true,
        showNumbers: true,
        showCircles: true
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
            <div style="margin-left: 10px;">
                <label><input type="checkbox" id="classic_${index}" ${variant.showClassic ? 'checked' : ''} onchange="toggleVariantType(${index}, 'classic')"> Classic</label>
                <label><input type="checkbox" id="hidden_${index}" ${variant.showHidden ? 'checked' : ''} onchange="toggleVariantType(${index}, 'hidden')"> Hidden</label>
                <label><input type="checkbox" id="bullish_${index}" ${variant.showBullish ? 'checked' : ''} onchange="toggleVariantType(${index}, 'bullish')"> <span style="color: #00FF00;">Bullish</span></label>
                <label><input type="checkbox" id="bearish_${index}" ${variant.showBearish ? 'checked' : ''} onchange="toggleVariantType(${index}, 'bearish')"> <span style="color: #FF0000;">Bearish</span></label>
                <label><input type="checkbox" id="names_${index}" ${variant.showNames ? 'checked' : ''} onchange="toggleVariantType(${index}, 'names')"> Names</label>
                <label><input type="checkbox" id="numbers_${index}" ${variant.showNumbers ? 'checked' : ''} onchange="toggleVariantType(${index}, 'numbers')"> Numbers</label>
                <label><input type="checkbox" id="circles_${index}" ${variant.showCircles ? 'checked' : ''} onchange="toggleVariantType(${index}, 'circles')"> Circles</label>
            </div>
            <button class="btn btn-danger" onclick="removeVariant(${index})" style="margin-left: 10px;">Remove</button>
        `;
        container.appendChild(div);
    });
}

function toggleVariantType(index, type) {
    if (index < variants.length) {
        switch(type) {
            case 'classic':
                variants[index].showClassic = document.getElementById(`classic_${index}`).checked;
                break;
            case 'hidden':
                variants[index].showHidden = document.getElementById(`hidden_${index}`).checked;
                break;
            case 'bullish':
                variants[index].showBullish = document.getElementById(`bullish_${index}`).checked;
                break;
            case 'bearish':
                variants[index].showBearish = document.getElementById(`bearish_${index}`).checked;
                break;
            case 'names':
                variants[index].showNames = document.getElementById(`names_${index}`).checked;
                // Update visibility of name traces without replotting
                if (window.currentResults && sessionId) {
                    updateTextTraceVisibility('names', index);
                    return;
                }
                break;
            case 'numbers':
                variants[index].showNumbers = document.getElementById(`numbers_${index}`).checked;
                // Update visibility of number traces without replotting
                if (window.currentResults && sessionId) {
                    updateTextTraceVisibility('numbers', index);
                    return;
                }
                break;
            case 'circles':
                variants[index].showCircles = document.getElementById(`circles_${index}`).checked;
                // Update visibility of circle traces without replotting
                if (window.currentResults && sessionId) {
                    updateTextTraceVisibility('circles', index);
                    return;
                }
                break;
        }
        
        // Update chart visibility without replotting
        if (window.currentResults && sessionId) {
            updateChartVisibility();
        }
    }
}

function updateTextTraceVisibility(type, variantIndex) {
    const chartDiv = document.getElementById('mainChart');
    if (!chartDiv || !chartDiv.data) return;
    
    const variant = variants[variantIndex];
    const variantName = variant.name;
    const isVisible = variant[type === 'names' ? 'showNames' : type === 'numbers' ? 'showNumbers' : 'showCircles'];
    
    const visibilityUpdate = {};
    chartDiv.data.forEach((trace, traceIndex) => {
        const traceName = trace.name || '';
        const legendGroup = trace.legendgroup || '';
        
        // Check if this trace belongs to this variant and type
        let matchesType = false;
        if (type === 'names' && (traceName.includes('Label'))) {
            // Names: trace name contains variant name
            matchesType = traceName.includes(variantName);
        } else if (type === 'numbers' && (traceName.includes('Number'))) {
            // Numbers: trace name contains variant name
            matchesType = traceName.includes(variantName);
        } else if (type === 'circles' && (traceName.includes('New in Variant') || traceName.includes('Missing in Variant'))) {
            // For circles: Use legendgroup to determine ownership
            if (traceName.includes('New in Variant')) {
                // Yellow circles belong to the variant that creates them (check legendgroup)
                matchesType = legendGroup.includes(variantName);
            } else if (traceName.includes('Missing in Variant')) {
                // Blue circles always belong to first variant (they show markers missing compared to first variant)
                matchesType = variantIndex === 0;
            }
        }
        
        if (matchesType) {
            visibilityUpdate[traceIndex] = isVisible;
        }
    });
    
    // Apply visibility updates only to affected traces
    if (Object.keys(visibilityUpdate).length > 0) {
        const traceIndices = Object.keys(visibilityUpdate).map(Number);
        const visibilityValues = traceIndices.map(i => visibilityUpdate[i]);
        Plotly.restyle(chartDiv, { visible: visibilityValues }, traceIndices);
    }
}

function updateChartVisibility() {
    if (!window.currentResults) return;
    
    // Instead of replotting, update visibility of traces
    const chartDiv = document.getElementById('mainChart');
    if (chartDiv && chartDiv.data) {
        const visibilityUpdates = [];
        const traceNames = chartDiv.data.map(trace => trace.name);
        
        // Determine which traces should be visible
        traceNames.forEach((name, index) => {
            let shouldShow = true;
            
            // Check if this trace belongs to a variant with specific settings
            for (let variant of variants) {
                if (name.includes(variant.name)) {
                    if (name.includes('Classic') && !variant.showClassic) shouldShow = false;
                    if (name.includes('Hidden') && !variant.showHidden) shouldShow = false;
                    if (name.includes('Bullish') && !variant.showBullish) shouldShow = false;
                    if (name.includes('Bearish') && !variant.showBearish) shouldShow = false;
                    break;
                }
            }
            
            visibilityUpdates.push(shouldShow ? true : 'legendonly');
        });
        
        // Update trace visibility without replotting
        Plotly.restyle(chartDiv, { visible: visibilityUpdates });
        return;
    }
    
    // Fallback to replotting if restyle fails
    plotChart(window.currentChartData, window.currentResults);
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
            // Show results panel
            document.getElementById('resultsPanel').style.display = 'block';
            // Store current data for toggle functionality
            window.currentChartData = data.chartData;
            window.currentResults = data.results;
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
            legendgroup: 'essential',
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
            line: { color: '#FFA500', width: 1 },
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
            line: { color: '#FFD700', width: 0.5 },
            legendgroup: 'essential',
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
            line: { color: '#00FFFF', width: 0.5 },
            legendgroup: 'essential',
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
            line: { color: '#FF00FF', width: 0.5 },
            legendgroup: 'essential',
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
            line: { color: '#9370DB', width: 0.5 },
            legendgroup: 'essential',
            showlegend: true
        });
    }

    // Add divergence markers and arrows for each variant
    let globalDivCount = 0;
    const divergenceDetails = [];
    const legendEntries = new Set(); // Track legend entries to avoid duplicates
    let firstVariantDivergences = null; // Track first variant divergences for comparison
    
    variants.forEach((variant, variantIndex) => {
        const res = results[variant.id];
        if (!res) return;
        
        const legendgroup = `variant_${variant.id}`;
        const bullishColor = '#00FF00';  // Green for bullish
        const bearishColor = '#FF0000';  // Red for bearish
        
        // Process all divergences (classic + hidden) together for numbering
        const allDivergences = [
            ...res.classic.map(d => ({...d, divType: 'classic'})),
            ...res.hidden.map(d => ({...d, divType: 'hidden'}))
        ].sort((a, b) => new Date(a.date) - new Date(b.date));
        
        // Store first variant divergences for comparison with full details
        if (variantIndex === 0) {
            firstVariantDivergences = allDivergences.map(d => ({
                date: d.date,
                divType: d.divType,
                type: d.type,
                low: d.low,
                rsi: d.rsi,
                macd: d.macd
            }));
        }
        
        allDivergences.forEach(div => {
            globalDivCount++;
            div.globalId = globalDivCount;
            
            // Determine if bullish or bearish
            const isBullish = div.type === 'classic' ? true : div.macd > 0;
            const divName = `${div.divType === 'classic' ? 'class' : 'hid'}.${isBullish ? 'bull' : 'bear'}.`;
            
            // Check if this divergence should be shown based on toggle settings
            const showType = (div.divType === 'classic' && variant.showClassic) || 
                           (div.divType === 'hidden' && variant.showHidden);
            const showDirection = (isBullish && variant.showBullish) || 
                                (!isBullish && variant.showBearish);
            
            if (!showType || !showDirection) {
                return; // Skip this divergence
            }
            
            // Store details for table
            divergenceDetails.push({
                id: globalDivCount,
                variant: variant.name,
                type: divName,
                date: div.date,
                price: div.low,
                rsi: div.rsi,
                macd: div.macd,
                strength: div.strength,
                trigger: `${div.divType} divergence detected with window=${div.window}, RSI=${div.rsi.toFixed(2)}, MACD=${div.macd.toFixed(4)}`
            });
            
            // Track this legend entry
            const legendKey = `${variant.name}_${divName}`;
            const shouldShowLegend = !legendEntries.has(legendKey);
            if (shouldShowLegend) {
                legendEntries.add(legendKey);
            }
            
            // Add main marker without text
            traces.push({
                x: [div.date],
                y: [div.low],
                mode: 'markers',
                type: 'scatter',
                name: `${variant.name} ${divName}`,
                marker: {
                    size: 14,
                    color: variant.color,
                    symbol: div.divType === 'classic' ? 'triangle-up' : 'diamond',
                    line: { color: '#808080', width: 0.5 }
                },
                xaxis: 'x',
                yaxis: 'y',
                legendgroup: `${variant.name}_${div.divType}_${isBullish ? 'bullish' : 'bearish'}`,
                showlegend: shouldShowLegend,
                hovertemplate: `<b>${divName}</b><br>` +
                              `Date: %{x}<br>` +
                              `Price: %{y:.4f}<br>` +
                              `RSI: ${div.rsi.toFixed(2)}<br>` +
                              `MACD: ${div.macd.toFixed(4)}<br>` +
                              `Strength: ${div.strength.toFixed(2)}<br>` +
                              `<extra></extra>`
            });
            
            // Add yellow highlighting circle for new markers (only for non-first variants)
            if (div.is_new === true && variantIndex > 0) {
                // Yellow circle for new markers not in first variant
                traces.push({
                    x: [div.date],
                    y: [div.low],
                    mode: 'markers',
                    type: 'scatter',
                    name: 'New in Variant',
                    marker: {
                        size: 24,
                        color: 'rgba(255, 255, 0, 0.3)',
                        symbol: 'circle',
                        line: { color: '#FFFF00', width: 0.5 }
                    },
                    xaxis: 'x',
                    yaxis: 'y',
                    legendgroup: `${variant.name}_${div.divType}_${isBullish ? 'bullish' : 'bearish'}`,
                    showlegend: false,
                    hoverinfo: 'skip'
                });
            }
            
            // Add marker number positioned to avoid overlap with names (if enabled)
            if (variant.showNumbers) {
                // Numbers on the right side, names will be on the left
                const numberPositions = ['bottom right', 'middle right', 'top right', 'bottom right', 'middle right', 'top right'];
                const numberPosition = numberPositions[(globalDivCount - 1) % 6];
                
                traces.push({
                x: [div.date],
                y: [div.low],
                mode: 'text',
                type: 'scatter',
                name: `${variant.name} ${divName} Number`,
                text: [globalDivCount],
                textposition: numberPosition,
                textfont: { color: variant.color, size: 10, family: 'Arial Bold' },
                xaxis: 'x',
                yaxis: 'y',
                legendgroup: `${variant.name}_${div.divType}_${isBullish ? 'bullish' : 'bearish'}`,
                showlegend: false,
                hoverinfo: 'skip'
            });
            }
            
            // Add RSI marker with same shape/size as candlestick
            traces.push({
                x: [div.date],
                y: [div.rsi],
                mode: 'markers',
                type: 'scatter',
                name: `${variant.name} ${divName} RSI`,
                marker: {
                    size: 14,
                    color: variant.color,
                    symbol: div.divType === 'classic' ? 'triangle-up' : 'diamond',
                    line: { color: '#808080', width: 0.5 }
                },
                xaxis: 'x',
                yaxis: 'y2',
                legendgroup: `${variant.name}_${div.divType}_${isBullish ? 'bullish' : 'bearish'}`,
                showlegend: false
            });
            
            // Add RSI highlighting circle for variant differences
            if (div.is_new === true && variantIndex > 0) {
                traces.push({
                    x: [div.date],
                    y: [div.rsi],
                    mode: 'markers',
                    type: 'scatter',
                    name: 'New in Variant RSI',
                    marker: {
                        size: 24,
                        color: 'rgba(255, 255, 0, 0.3)',
                        symbol: 'circle',
                        line: { color: '#FFFF00', width: 0.5 }
                    },
                    xaxis: 'x',
                    yaxis: 'y2',
                    legendgroup: `${variant.name}_${div.divType}_${isBullish ? 'bullish' : 'bearish'}`,
                    showlegend: false,
                    hoverinfo: 'skip'
                });
            }
            
            // Add RSI number with staggering
            const rsiRange = Math.max(...chartData.rsi.filter(x => x)) - Math.min(...chartData.rsi.filter(x => x));
            const rsiOffset = rsiRange * 0.008 * (globalDivCount % 3);
            traces.push({
                x: [div.date],
                y: [div.rsi - 3 - rsiOffset],
                mode: 'text',
                type: 'scatter',
                name: `${variant.name} ${divName} RSI Number`,
                text: [globalDivCount],
                textposition: 'middle center',
                textfont: { color: variant.color, size: 10, family: 'Arial Bold' },
                xaxis: 'x',
                yaxis: 'y2',
                legendgroup: `${variant.name}_${div.divType}_${isBullish ? 'bullish' : 'bearish'}`,
                showlegend: false,
                hoverinfo: 'skip'
            });
            
            // Add RSI spray line
            traces.push({
                x: [div.date, div.date],
                y: [div.rsi, div.rsi - 2 - rsiOffset],
                mode: 'lines',
                type: 'scatter',
                name: `${variant.name} ${divName} RSI Line`,
                line: { color: variant.color, width: 0.5, dash: 'dot' },
                xaxis: 'x',
                yaxis: 'y2',
                legendgroup: `${variant.name}_${div.divType}_${isBullish ? 'bullish' : 'bearish'}`,
                showlegend: false,
                hoverinfo: 'skip'
            });
            
            // Add MACD marker with same shape/size as candlestick
            traces.push({
                x: [div.date],
                y: [div.macd],
                mode: 'markers',
                type: 'scatter',
                name: `${variant.name} ${divName} MACD`,
                marker: {
                    size: 14,
                    color: variant.color,
                    symbol: div.divType === 'classic' ? 'triangle-up' : 'diamond',
                    line: { color: '#808080', width: 0.5 }
                },
                xaxis: 'x',
                yaxis: 'y3',
                legendgroup: `${variant.name}_${div.divType}_${isBullish ? 'bullish' : 'bearish'}`,
                showlegend: false
            });
            
            // Add MACD highlighting circle for variant differences
            if (div.is_new === true && variantIndex > 0) {
                traces.push({
                    x: [div.date],
                    y: [div.macd],
                    mode: 'markers',
                    type: 'scatter',
                    name: 'New in Variant MACD',
                    marker: {
                        size: 24,
                        color: 'rgba(255, 255, 0, 0.3)',
                        symbol: 'circle',
                        line: { color: '#FFFF00', width: 0.5 }
                    },
                    xaxis: 'x',
                    yaxis: 'y3',
                    legendgroup: `${variant.name}_${div.divType}_${isBullish ? 'bullish' : 'bearish'}`,
                    showlegend: false,
                    hoverinfo: 'skip'
                });
            }
            
            // Add MACD number with staggering
            const macdRange = Math.max(...chartData.macd_histogram.filter(x => x)) - Math.min(...chartData.macd_histogram.filter(x => x));
            const macdOffset = macdRange * 0.008 * (globalDivCount % 3);
            const macdNumberY = div.macd < 0 ? div.macd + 0.002 + macdOffset : div.macd - 0.002 - macdOffset;
            traces.push({
                x: [div.date],
                y: [macdNumberY],
                mode: 'text',
                type: 'scatter',
                name: `${variant.name} ${divName} MACD Number`,
                text: [globalDivCount],
                textposition: 'middle center',
                textfont: { color: variant.color, size: 10, family: 'Arial Bold' },
                xaxis: 'x',
                yaxis: 'y3',
                legendgroup: `${variant.name}_${div.divType}_${isBullish ? 'bullish' : 'bearish'}`,
                showlegend: false,
                hoverinfo: 'skip'
            });
            
            // Add MACD spray line
            traces.push({
                x: [div.date, div.date],
                y: [div.macd, macdNumberY],
                mode: 'lines',
                type: 'scatter',
                name: `${variant.name} ${divName} MACD Line`,
                line: { color: variant.color, width: 0.5, dash: 'dot' },
                xaxis: 'x',
                yaxis: 'y3',
                legendgroup: `${variant.name}_${div.divType}_${isBullish ? 'bullish' : 'bearish'}`,
                showlegend: false,
                hoverinfo: 'skip'
            });
            
            // Add divergence arrows and names
            const dateIndex = chartData.dates.indexOf(div.date);
            if (dateIndex > variant.window) {
                // Create trend lines showing divergence pattern
                const startIndex = Math.max(0, dateIndex - variant.window);
                const endIndex = dateIndex;
                
                // Create curved arrows for Price plot
                const midIndex = Math.floor((startIndex + endIndex) / 2);
                const priceStart = chartData.low[startIndex];
                const priceMid = (chartData.low[midIndex] + chartData.low[startIndex] + div.low) / 3;
                const curvature = (priceStart - div.low) * 0.5; // Increased curvature
                
                traces.push({
                    x: [chartData.dates[startIndex], chartData.dates[midIndex], chartData.dates[endIndex]],
                    y: [priceStart, priceMid + curvature, div.low],
                    mode: 'lines',
                    type: 'scatter',
                    name: `${variant.name} ${divName} Trend`,
                    line: { color: variant.color, width: 0.5, dash: 'dash', shape: 'spline' },
                    xaxis: 'x',
                    yaxis: 'y',
                    legendgroup: `${variant.name}_${div.divType}_${isBullish ? 'bullish' : 'bearish'}`,
                    showlegend: false,
                    hoverinfo: 'skip'
                });
                
                // Add divergence name annotation to Price plot - positioned to avoid overlap with numbers (if enabled)
                if (variant.showNames) {
                    // Names on the left side, numbers are on the right
                    const namePositions = ['bottom left', 'middle left', 'top left', 'bottom left', 'middle left', 'top left'];
                    const namePosition = namePositions[(globalDivCount - 1) % 6];
                    
                    traces.push({
                    x: [div.date],
                    y: [div.low],
                    mode: 'text',
                    type: 'scatter',
                        name: `${variant.name} ${divName} Label`,
                        text: [divName],
                        textposition: namePosition,
                        textfont: { color: variant.color, size: 10, family: 'Arial Bold' },
                        xaxis: 'x',
                        yaxis: 'y',
                        legendgroup: `${variant.name}_${div.divType}_${isBullish ? 'bullish' : 'bearish'}`,
                        showlegend: false,
                        hoverinfo: 'skip'
                    });
                }
                
                // Create curved arrows for RSI plot
                const rsiStart = chartData.rsi[startIndex];
                const rsiMid = (chartData.rsi[midIndex] + rsiStart + div.rsi) / 3;
                const rsiCurvature = (rsiStart - div.rsi) * 0.5;
                
                traces.push({
                    x: [chartData.dates[startIndex], chartData.dates[midIndex], chartData.dates[endIndex]],
                    y: [rsiStart, rsiMid + rsiCurvature, div.rsi],
                    mode: 'lines',
                    type: 'scatter',
                    name: `${variant.name} ${divName} RSI Trend`,
                    line: { color: variant.color, width: 0.5, dash: 'dash', shape: 'spline' },
                    xaxis: 'x',
                    yaxis: 'y2',
                    legendgroup: `${variant.name}_${div.divType}_${isBullish ? 'bullish' : 'bearish'}`,
                    showlegend: false,
                    hoverinfo: 'skip'
                });
                
                // Add divergence name annotation to RSI plot
                traces.push({
                    x: [div.date],
                    y: [div.rsi - 6 - rsiOffset],
                    mode: 'text',
                    type: 'scatter',
                    name: `${variant.name} ${divName} RSI Label`,
                    text: [divName],
                    textposition: 'bottom center',
                    textfont: { color: variant.color, size: 10, family: 'Arial Bold' },
                    xaxis: 'x',
                    yaxis: 'y2',
                    legendgroup: `${variant.name}_${div.divType}_${isBullish ? 'bullish' : 'bearish'}`,
                    showlegend: false,
                    hoverinfo: 'skip'
                });
                
                // Create curved arrows for MACD plot
                const macdStart = chartData.macd_histogram[startIndex];
                const macdMid = (chartData.macd_histogram[midIndex] + macdStart + div.macd) / 3;
                const macdCurvature = (macdStart - div.macd) * 0.5;
                
                traces.push({
                    x: [chartData.dates[startIndex], chartData.dates[midIndex], chartData.dates[endIndex]],
                    y: [macdStart, macdMid + macdCurvature, div.macd],
                    mode: 'lines',
                    type: 'scatter',
                    name: `${variant.name} ${divName} MACD Trend`,
                    line: { color: variant.color, width: 0.5, dash: 'dash', shape: 'spline' },
                    xaxis: 'x',
                    yaxis: 'y3',
                    legendgroup: `${variant.name}_${div.divType}_${isBullish ? 'bullish' : 'bearish'}`,
                    showlegend: false,
                    hoverinfo: 'skip'
                });
                
                // Add divergence name annotation to MACD plot
                traces.push({
                    x: [div.date],
                    y: [div.macd < 0 ? div.macd + 0.004 + macdOffset : div.macd - 0.004 - macdOffset],
                    mode: 'text',
                    type: 'scatter',
                    name: `${variant.name} ${divName} MACD Label`,
                    text: [divName],
                    textposition: 'bottom center',
                    textfont: { color: variant.color, size: 10, family: 'Arial Bold' },
                    xaxis: 'x',
                    yaxis: 'y3',
                    legendgroup: `${variant.name}_${div.divType}_${isBullish ? 'bullish' : 'bearish'}`,
                    showlegend: false,
                    hoverinfo: 'skip'
                });
            }
        });
        
        // Add blue highlighting for missing markers (exist in first variant but not current)
        if (variantIndex > 0 && firstVariantDivergences) {
            const currentVariantDates = allDivergences.map(d => d.date);
            const missingDivergences = firstVariantDivergences.filter(firstDiv => !currentVariantDates.includes(firstDiv.date));
            
            missingDivergences.forEach(missingDiv => {
                // Find the closest price and indicators for this date
                const dateIndex = chartData.dates.indexOf(missingDiv.date);
                if (dateIndex >= 0) {
                    const price = chartData.low[dateIndex];
                    const rsi = chartData.rsi[dateIndex];
                    const macd = chartData.macd_histogram[dateIndex];
                    
                    // Blue circles belong to the first variant (which induced these markers)
                    // Determine bullish/bearish based on divergence logic
                    const isBullish = missingDiv.divType === 'classic' ? true : macd > 0;
                    const firstVariantName = variants[0].name;
                    const legendGroup = `${firstVariantName}_${missingDiv.divType}_${isBullish ? 'bullish' : 'bearish'}`;
                    
                    // Add blue circle to candlestick plot
                    traces.push({
                        x: [missingDiv.date],
                        y: [price],
                        mode: 'markers',
                        type: 'scatter',
                        name: 'Missing in Variant',
                        marker: {
                            size: 24,
                            color: 'rgba(0, 0, 255, 0.3)',
                            symbol: 'circle',
                            line: { color: '#0000FF', width: 0.5 }
                        },
                        xaxis: 'x',
                        yaxis: 'y',
                        legendgroup: legendGroup,
                        showlegend: false,
                        hoverinfo: 'skip'
                    });
                    
                    // Add blue circle to RSI plot
                    traces.push({
                        x: [missingDiv.date],
                        y: [rsi],
                        mode: 'markers',
                        type: 'scatter',
                        name: 'Missing in Variant RSI',
                        marker: {
                            size: 24,
                            color: 'rgba(0, 0, 255, 0.3)',
                            symbol: 'circle',
                            line: { color: '#0000FF', width: 0.5 }
                        },
                        xaxis: 'x',
                        yaxis: 'y2',
                        legendgroup: legendGroup,
                        showlegend: false,
                        hoverinfo: 'skip'
                    });
                    
                    // Add blue circle to MACD plot
                    traces.push({
                        x: [missingDiv.date],
                        y: [macd],
                        mode: 'markers',
                        type: 'scatter',
                        name: 'Missing in Variant MACD',
                        marker: {
                            size: 24,
                            color: 'rgba(0, 0, 255, 0.3)',
                            symbol: 'circle',
                            line: { color: '#0000FF', width: 0.5 }
                        },
                        xaxis: 'x',
                        yaxis: 'y3',
                        legendgroup: legendGroup,
                        showlegend: false,
                        hoverinfo: 'skip'
                    });
                }
            });
        }
    });
    
    // Store divergence details globally for table display
    window.divergenceDetails = divergenceDetails;

    // Create results summary for text box with per-variant breakdown
    let resultsSummary = `<b>Divergence Summary</b><br>`;
    resultsSummary += `<b>Total: ${globalDivCount}</b><br><br>`;
    
    variants.forEach(variant => {
        const variantDivs = divergenceDetails.filter(d => d.variant === variant.name);
        if (variantDivs.length > 0) {
            const bullish = variantDivs.filter(d => d.type.includes('Bullish')).length;
            const bearish = variantDivs.filter(d => d.type.includes('Bearish')).length;
            const classic = variantDivs.filter(d => d.type.includes('Classic')).length;
            const hidden = variantDivs.filter(d => d.type.includes('Hidden')).length;
            
            resultsSummary += `<b style="color: ${variant.color};">${variant.name}: ${variantDivs.length}</b><br>`;
            resultsSummary += `  <span style="color: #00FF00;">▲${bullish}</span> `;
            resultsSummary += `<span style="color: #FF0000;">▼${bearish}</span> `;
            resultsSummary += `C:${classic} H:${hidden}<br>`;
        }
    });

    const layout = {
        xaxis: {
            rangeslider: { visible: false },
            domain: [0, 1],
            anchor: 'y3',
            gridcolor: 'rgba(255,255,255,0.1)',
            zerolinecolor: 'rgba(255,255,255,0.2)',
            titlefont: { color: '#FFFFFF', size: 8 },
            tickfont: { color: '#FFFFFF', size: 8 }
        },
        yaxis: {
            title: 'Price',
            domain: [0.35, 1],
            gridcolor: 'rgba(255,255,255,0.1)',
            titlefont: { color: '#FFFFFF', size: 8 },
            tickfont: { color: '#FFFFFF', size: 8 },
            fixedrange: false
        },
        yaxis2: {
            title: 'RSI',
            domain: [0.18, 0.32],
            anchor: 'x',
            gridcolor: 'rgba(255,255,255,0.1)',
            titlefont: { color: '#FFFFFF', size: 8 },
            tickfont: { color: '#FFFFFF', size: 8 },
            fixedrange: false
        },
        yaxis3: {
            title: 'MACD Histogram',
            domain: [0, 0.15],
            anchor: 'x',
            gridcolor: 'rgba(255,255,255,0.1)',
            titlefont: { color: '#FFFFFF', size: 8 },
            tickfont: { color: '#FFFFFF', size: 8 },
            fixedrange: false
        },
        plot_bgcolor: '#0a0a0a',
        paper_bgcolor: '#1a1a1a',
        font: { color: '#FFFFFF' },
        showlegend: true,
        legend: {
            orientation: 'h',
            y: 1.15,
            x: 0,
            bgcolor: 'rgba(0,0,0,0.7)',
            font: { color: '#FFFFFF', size: 8 },
            bordercolor: 'rgba(255,255,255,0.2)',
            borderwidth: 1,
            itemsizing: 'constant',
            itemwidth: 30
        },
        hovermode: 'x unified',
        height: Math.max(600, window.innerHeight - 200),
        annotations: [
            {
                text: resultsSummary,
                showarrow: false,
                xref: 'paper',
                yref: 'paper',
                x: 0.02,
                y: 0.85, // Moved down to avoid legend overlap
                xanchor: 'left',
                yanchor: 'top',
                bgcolor: 'rgba(0,0,0,0.8)',
                bordercolor: 'rgba(255,255,255,0.2)',
                borderwidth: 1,
                font: { color: '#FFFFFF', size: 10 }
            }
        ]
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
    
    // Create two-row layout: summary on top, detailed list below
    let html = '<div style="display: flex; flex-direction: column; gap: 20px;">';
    
    // Summary table
    html += '<div class="stats-card"><h3>📊 Summary Results</h3><table style="width: 100%; border-collapse: collapse;">';
    html += '<tr style="border-bottom: 2px solid rgba(255,255,255,0.2);"><th style="text-align: left; padding: 8px; font-size: 13px;">Variant</th><th style="text-align: center; padding: 8px; font-size: 13px;">Classic</th><th style="text-align: center; padding: 8px; font-size: 13px;">Hidden</th><th style="text-align: center; padding: 8px; font-size: 13px;">Total</th></tr>';

    variants.forEach(v => {
        const res = results[v.id] || { classic: [], hidden: [], total: 0 };
        html += `<tr style="border-bottom: 1px solid rgba(255,255,255,0.1);">
            <td style="padding: 8px; color: ${v.color}; font-weight: bold; font-size: 13px;">${v.name.charAt(0).toUpperCase() + v.name.slice(1)}</td>
            <td style="text-align: center; padding: 8px; font-size: 13px;">${res.classic.length}</td>
            <td style="text-align: center; padding: 8px; font-size: 13px;">${res.hidden.length}</td>
            <td style="text-align: center; padding: 8px; font-weight: bold; font-size: 13px;">${res.total}</td>
        </tr>`;
    });

    html += '</table></div>';
    
    // Detailed divergence table below summary
    if (window.divergenceDetails && window.divergenceDetails.length > 0) {
        html += '<div class="stats-card"><h3>🔍 Detailed Divergence List</h3>';
        html += '<table style="width: 100%; border-collapse: collapse; font-size: 11px;">';
        html += '<tr style="border-bottom: 2px solid rgba(255,255,255,0.2);">';
        html += '<th style="text-align: center; padding: 6px;">#</th>';
        html += '<th style="text-align: left; padding: 6px;">Variant</th>';
        html += '<th style="text-align: left; padding: 6px;">Type</th>';
        html += '<th style="text-align: left; padding: 6px;">Date</th>';
        html += '<th style="text-align: right; padding: 6px;">Price</th>';
        html += '<th style="text-align: right; padding: 6px;">RSI</th>';
        html += '<th style="text-align: right; padding: 6px;">MACD</th>';
        html += '<th style="text-align: left; padding: 6px;">Trigger Reason</th>';
        html += '</tr>';
        
        window.divergenceDetails.forEach(div => {
            const colorStyle = div.type.includes('Bullish') ? 'color: #00FF00;' : 'color: #FF0000;';
            html += `<tr style="border-bottom: 1px solid rgba(255,255,255,0.05); cursor: pointer; transition: background-color 0.2s;" 
                         onmouseover="this.style.backgroundColor='rgba(255,255,255,0.1)'" 
                         onmouseout="this.style.backgroundColor='transparent'"
                         onclick="jumpToMarker('${div.date}', ${div.price})">
                <td style="text-align: center; padding: 6px; font-weight: bold; ${colorStyle}">${div.id}</td>
                <td style="padding: 6px;">${div.variant}</td>
                <td style="padding: 6px; ${colorStyle} font-weight: bold;">${div.type}</td>
                <td style="padding: 6px;">${div.date}</td>
                <td style="text-align: right; padding: 6px;">${div.price.toFixed(4)}</td>
                <td style="text-align: right; padding: 6px;">${div.rsi.toFixed(2)}</td>
                <td style="text-align: right; padding: 6px;">${div.macd.toFixed(4)}</td>
                <td style="padding: 6px; font-size: 10px; color: #aaa;">${div.trigger}</td>
            </tr>`;
        });
        
        html += '</table></div>';
    }
    
    html += '</div>'; // Close the flex container
    container.innerHTML = html;
}

function jumpToMarker(targetDate, targetPrice) {
    const plotDiv = document.getElementById('mainChart');
    if (!plotDiv) return;
    
    // Get current layout to preserve zoom and other settings
    const currentLayout = plotDiv.layout;
    if (!currentLayout) return;
    
    // Get current axis ranges to calculate center position
    const currentXRange = currentLayout.xaxis.range;
    const currentYRange = currentLayout.yaxis.range;
    
    if (!currentXRange || !currentYRange) return;
    
    // Calculate the range width to maintain zoom level
    const xRangeWidth = new Date(currentXRange[1]) - new Date(currentXRange[0]);
    const yRangeHeight = currentYRange[1] - currentYRange[0];
    
    // Center the view on the target date and price
    const targetDateObj = new Date(targetDate);
    const newXStart = new Date(targetDateObj.getTime() - xRangeWidth / 2);
    const newXEnd = new Date(targetDateObj.getTime() + xRangeWidth / 2);
    const newYStart = targetPrice - yRangeHeight / 2;
    const newYEnd = targetPrice + yRangeHeight / 2;
    
    // Update layout to center on the marker while preserving zoom
    const update = {
        'xaxis.range': [newXStart.toISOString(), newXEnd.toISOString()],
        'yaxis.range': [newYStart, newYEnd]
    };
    
    // Use Plotly.relayout to update without changing other plot characteristics
    Plotly.relayout(plotDiv, update);
    
    // Show a brief highlight animation at the target point
    showMarkerHighlight(targetDate, targetPrice);
}

function showMarkerHighlight(targetDate, targetPrice) {
    const plotDiv = document.getElementById('mainChart');
    if (!plotDiv) return;
    
    // Add a temporary highlight circle that fades out
    const highlightTrace = {
        x: [targetDate],
        y: [targetPrice],
        mode: 'markers',
        type: 'scatter',
        name: 'Highlight',
        marker: {
            size: 35,
            color: 'rgba(255, 255, 255, 0.8)',
            symbol: 'circle',
            line: { color: '#FFFFFF', width: 0.5 }
        },
        xaxis: 'x',
        yaxis: 'y',
        showlegend: false,
        hoverinfo: 'skip'
    };
    
    // Add the highlight
    Plotly.addTraces(plotDiv, [highlightTrace]);
    
    // Remove it after animation
    setTimeout(() => {
        const traceIndex = plotDiv.data.length - 1;
        Plotly.deleteTraces(plotDiv, [traceIndex]);
    }, 1500);
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
        case 'conservative':
            document.getElementById('window').value = 7;
            document.getElementById('candleTol').value = 0.05;
            document.getElementById('macdTol').value = 2.0;
            break;
        case 'aggressive':
            document.getElementById('window').value = 3;
            document.getElementById('candleTol').value = 0.15;
            document.getElementById('macdTol').value = 5.0;
            break;
    }
}

function saveConfig() {
    const config = {
        variants: variants,
        sessionId: sessionId
    };
    const blob = new Blob([JSON.stringify(config, null, 2)], {type: 'application/json'});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'bullish_config.json';
    a.click();
    URL.revokeObjectURL(url);
    showMessage('messageContainer', 'Configuration saved', 'success');
}

function loadConfig() {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = '.json';
    input.onchange = function(event) {
        const file = event.target.files[0];
        if (!file) return;
        
        const reader = new FileReader();
        reader.onload = function(e) {
            try {
                const config = JSON.parse(e.target.result);
                variants = config.variants || [];
                variantIdCounter = Math.max(...variants.map(v => v.id), 0) + 1;
                updateVariantsList();
                showMessage('messageContainer', 'Configuration loaded', 'success');
            } catch (error) {
                showMessage('messageContainer', 'Invalid configuration file', 'error');
            }
        };
        reader.readAsText(file);
    };
    input.click();
}

function exportResults(format) {
    showMessage('messageContainer', `Export to ${format.toUpperCase()} - Feature coming soon`, 'loading');
}

function toggleFullscreen() {
    const chartDiv = document.getElementById('mainChart');
    if (!document.fullscreenElement) {
        chartDiv.requestFullscreen().catch(err => {
            showMessage('messageContainer', 'Fullscreen not supported', 'error');
        });
    } else {
        document.exitFullscreen();
    }
}

function closeValidation() {
    document.getElementById('validationPanel').style.display = 'none';
}