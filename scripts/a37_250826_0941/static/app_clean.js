let sessionId = null;
let variants = [];
let variantIdCounter = 1;

// Caching system for indicators
const indicatorCache = new Map();
const maxCacheSize = 50; // Limit cache entries

function getCacheKey(chartData, variant) {
    return `${JSON.stringify([chartData.dates?.length, chartData.open?.[0], chartData.close?.slice(-1)[0]])}_${variant.window}_${variant.candleTol}_${variant.macdTol}`;
}

function cacheIndicators(key, data) {
    if (indicatorCache.size >= maxCacheSize) {
        const firstKey = indicatorCache.keys().next().value;
        indicatorCache.delete(firstKey);
    }
    indicatorCache.set(key, { data, timestamp: Date.now() });
}

function getCachedIndicators(key) {
    const cached = indicatorCache.get(key);
    if (cached && (Date.now() - cached.timestamp) < 300000) { // 5 min cache
        return cached.data;
    }
    return null;
}

// Color palette for variants
const presetColors = {
    'basis': '#90EE90',
    'v1': '#ADD8E6',
    'v2': '#FFA500',
    'v3': '#FF69B4',
    'v4': '#FFFF00'
};
const colors = ['#90EE90', '#ADD8E6', '#FFA500', '#FF69B4', '#FFFF00', '#FF6B6B', '#4ECDC4', '#45B7D1', 
                '#9370DB', '#00CED1', '#32CD32', '#FF1493', '#1E90FF', '#FF4500', '#DA70D6', '#00FF7F',
                '#FFD700', '#DC143C', '#00BFFF', '#ADFF2F', '#FF6347', '#40E0D0', '#EE82EE', '#98FB98'];

function cycleVariantColor(variantIndex) {
    const variant = variants[variantIndex];
    if (!variant) return;
    
    const currentIndex = colors.indexOf(variant.color);
    const nextIndex = (currentIndex + 1) % colors.length;
    variant.color = colors[nextIndex];
    
    updateVariantsList();
    if (window.currentResults && sessionId) {
        plotChart(window.currentResults.chartData, window.currentResults.results);
    }
}

function calculateConfidence(div, variant) {
    let confidence = div.strength; // Base from server strength
    
    // RSI factor: extreme RSI values (oversold/overbought) increase confidence
    const rsiExtreme = Math.min(div.rsi, 100 - div.rsi) / 30; // 0-1 scale
    confidence += rsiExtreme * 0.3;
    
    // MACD magnitude factor
    const macdStrength = Math.abs(div.macd) / variant.macdTol;
    confidence += Math.min(macdStrength, 1) * 0.2;
    
    // Window size factor: larger windows more reliable
    const windowFactor = Math.min(variant.window / 20, 1) * 0.1;
    confidence += windowFactor;
    
    return Math.min(confidence, 1); // Cap at 1.0
}

function getConfidenceLevel(confidence) {
    if (confidence >= 0.8) return 'High';
    if (confidence >= 0.6) return 'Medium';
    if (confidence >= 0.4) return 'Low';
    return 'Very Low';
}

function calculateSuccessRate(div, chartData, lookAheadDays = 30) {
    const dateIndex = chartData.dates.indexOf(div.date);
    if (dateIndex === -1 || dateIndex >= chartData.dates.length - 1) {
        return { success: null, performance: null, reason: 'No future data' };
    }
    
    const currentPrice = chartData.close[dateIndex];
    const endIndex = Math.min(dateIndex + lookAheadDays, chartData.dates.length - 1);
    const futurePrice = chartData.close[endIndex];
    
    if (!currentPrice || !futurePrice) {
        return { success: null, performance: null, reason: 'Missing price data' };
    }
    
    const performance = ((futurePrice - currentPrice) / currentPrice) * 100;
    const isBullish = div.divType === 'classic'; // Assume classic = bullish, hidden = bearish
    
    // For bullish signals: success if price went up
    // For bearish signals: success if price went down  
    const success = isBullish ? performance > 0 : performance < 0;
    
    return {
        success: success,
        performance: Math.abs(performance),
        direction: performance > 0 ? '+' : '-',
        actualPerformance: performance,
        lookAheadDays: endIndex - dateIndex
    };
}

function addSuccessRateToVariant(variant, index) {
    if (!variant.showSuccessRate) {
        variant.showSuccessRate = true;
    } else {
        variant.showSuccessRate = false;
    }
    
    updateVariantsList();
    if (window.currentResults && sessionId) {
        plotChart(window.currentResults.chartData, window.currentResults.results);
    }
}

function simulateBullishTradingStrategy(chartData, allBullishSignals, startingCapital = 10000, stopLossPercent = 5, takeProfitPercent = 25, positionSizePercent = 100, startDate = null, endDate = null) {
    const capitalHistory = [];
    const tradeHistory = [];
    let currentCapital = startingCapital;
    let cashBalance = startingCapital; // Track cash separately for position sizing
    let position = null; // { entryDate, entryPrice, quantity, highestClose, stopLoss, cashUsed }
    
    // Create capital tracking for each date
    for (let i = 0; i < chartData.dates.length; i++) {
        const currentDate = chartData.dates[i];
        const currentOpen = chartData.open[i];
        const currentHigh = chartData.high[i];
        const currentLow = chartData.low[i];
        const currentClose = chartData.close[i];
        
        // Skip dates outside trading range
        const dateStr = new Date(currentDate).toISOString().split('T')[0];
        if (startDate && dateStr < startDate) {
            capitalHistory.push({
                date: currentDate,
                capital: currentCapital,
                inPosition: !!position
            });
            continue;
        }
        if (endDate && dateStr > endDate) {
            capitalHistory.push({
                date: currentDate,
                capital: currentCapital,
                inPosition: !!position
            });
            continue;
        }
        
        // Check if we should enter a position (bullish signal from previous day)
        if (!position) {
            const signalYesterday = allBullishSignals.find(signal => {
                const signalIndex = chartData.dates.indexOf(signal.date);
                return signalIndex === i - 1; // Signal was yesterday, enter today
            });
            
            if (signalYesterday && cashBalance > 0) {
                // Enter position: buy at opening price with position sizing
                const tradingCapital = cashBalance * (positionSizePercent / 100);
                const quantity = tradingCapital / currentOpen;
                cashBalance -= tradingCapital; // Remove cash used for position
                position = {
                    entryDate: currentDate,
                    entryPrice: currentOpen,
                    quantity: quantity,
                    highestClose: currentOpen,
                    stopLoss: currentOpen * (1 - stopLossPercent / 100), // Initial stop loss
                    cashUsed: tradingCapital
                };
                
                tradeHistory.push({
                    type: 'BUY',
                    date: currentDate,
                    price: currentOpen,
                    quantity: quantity,
                    capital: currentCapital,
                    signal: signalYesterday
                });
            }
        }
        
        // If we have a position, manage it
        if (position) {
            // Update highest close and trailing stop loss
            if (currentClose > position.highestClose) {
                position.highestClose = currentClose;
                position.stopLoss = currentClose * (1 - stopLossPercent / 100); // Trailing stop from highest close
            }
            
            // Check exit conditions
            let shouldExit = false;
            let exitReason = '';
            let exitPrice = currentClose;
            
            // Take profit: user-defined percentage
            if (currentHigh >= position.entryPrice * (1 + takeProfitPercent / 100)) {
                shouldExit = true;
                exitReason = `Take Profit (+${takeProfitPercent}%)`;
                exitPrice = position.entryPrice * (1 + takeProfitPercent / 100);
            }
            // Stop loss: hit during the day
            else if (currentLow <= position.stopLoss) {
                shouldExit = true;
                exitReason = `Stop Loss (-${stopLossPercent}% from highest)`;
                exitPrice = position.stopLoss;
            }
            
            if (shouldExit) {
                // Exit position
                const totalValue = position.quantity * exitPrice;
                cashBalance += totalValue; // Return proceeds to cash
                currentCapital = cashBalance; // Update total capital
                
                const profitLoss = ((exitPrice - position.entryPrice) / position.entryPrice) * 100;
                
                tradeHistory.push({
                    type: 'SELL',
                    date: currentDate,
                    price: exitPrice,
                    quantity: position.quantity,
                    capital: currentCapital,
                    reason: exitReason,
                    profitLoss: profitLoss,
                    holdDays: chartData.dates.indexOf(currentDate) - chartData.dates.indexOf(position.entryDate)
                });
                
                position = null;
            } else {
                // Update current value for display (cash + position value)
                currentCapital = cashBalance + (position.quantity * currentClose);
            }
        }
        
        // Record capital for this date
        capitalHistory.push({
            date: currentDate,
            capital: currentCapital,
            inPosition: !!position
        });
    }
    
    return {
        capitalHistory,
        tradeHistory,
        finalCapital: currentCapital,
        totalReturn: ((currentCapital - startingCapital) / startingCapital) * 100,
        totalTrades: tradeHistory.filter(t => t.type === 'SELL').length
    };
}

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
    
    // Drag & Drop support
    const container = document.querySelector('.container');
    
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        container.addEventListener(eventName, preventDefaults, false);
    });
    
    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }
    
    ['dragenter', 'dragover'].forEach(eventName => {
        container.addEventListener(eventName, () => container.style.backgroundColor = 'rgba(100, 255, 218, 0.1)', false);
    });
    
    ['dragleave', 'drop'].forEach(eventName => {
        container.addEventListener(eventName, () => container.style.backgroundColor = '', false);
    });
    
    container.addEventListener('drop', handleDrop, false);
    
    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        
        if (files.length > 0) {
            const file = files[0];
            const fileExt = file.name.split('.').pop().toLowerCase();
            
            if (['csv', 'parquet'].includes(fileExt)) {
                document.getElementById('fileInput').files = files;
                document.getElementById('fileInput').dispatchEvent(new Event('change'));
                showMessage('messageContainer', `Dropped file: ${file.name}`, 'success');
            } else {
                showMessage('messageContainer', 'Only CSV/Parquet files supported', 'error');
            }
        }
    }
    
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
                
                // Update data timeframe display
                if (data.info.date_range) {
                    const startDate = data.info.date_range.start;
                    const endDate = data.info.date_range.end;
                    document.getElementById('dataTimeFrame').textContent = `${startDate} to ${endDate}`;
                    
                    // Set default trading dates to data range
                    document.getElementById('tradingStartDate').value = startDate;
                    document.getElementById('tradingEndDate').value = endDate;
                } else {
                    document.getElementById('dataTimeFrame').textContent = 'Range not available';
                }
                
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
    
    // Trading checkbox toggle to show/hide date range
    const tradingCheckbox = document.getElementById('enableTrading');
    const tradingDateRange = document.getElementById('tradingDateRange');
    
    tradingCheckbox.addEventListener('change', (e) => {
        if (e.target.checked) {
            tradingDateRange.style.display = 'flex';
        } else {
            tradingDateRange.style.display = 'none';
        }
    });
});

function addVariant() {
    const name = document.getElementById('variantName').value;
    const window = parseInt(document.getElementById('window').value);
    const candleTol = parseFloat(document.getElementById('candleTol').value);
    const macdTol = parseFloat(document.getElementById('macdTol').value);
    
    // Get trading parameters for this variant
    const stopLoss = parseFloat(document.getElementById('stopLoss').value) || 5;
    const takeProfit = parseFloat(document.getElementById('takeProfit').value) || 25;
    const positionSize = parseFloat(document.getElementById('positionSize').value) || 100;
    const startDate = document.getElementById('tradingStartDate').value || null;
    const endDate = document.getElementById('tradingEndDate').value || null;
    
    // Get plot styling parameters for this variant
    const textSize = parseFloat(document.getElementById('textSize').value) || 10;
    const lineWidth = parseFloat(document.getElementById('lineWidth').value) || 1.5;
    const numberPosition = parseFloat(document.getElementById('numberPosition').value) || 4;
    const namePosition = parseFloat(document.getElementById('namePosition').value) || 6;
    const successPosition = parseFloat(document.getElementById('successPosition').value) || 8;
    
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
        // Trading parameters stored per variant
        stopLoss: stopLoss,
        takeProfit: takeProfit,
        positionSize: positionSize,
        startDate: startDate,
        endDate: endDate,
        // Plot styling parameters stored per variant
        textSize: textSize,
        lineWidth: lineWidth,
        numberPosition: numberPosition,
        namePosition: namePosition,
        successPosition: successPosition,
        color: presetColors[name.toLowerCase()] || colors[variants.length % colors.length],
        showClassic: true,
        showHidden: true,
        showBullish: true,
        showBearish: true,
        showNames: true,
        showNumbers: true,
        showCircles: true,
        showSuccessRate: false,
        showRSINames: true,
        showRSINumbers: true,
        showMACDNames: true,
        showMACDNumbers: true
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
            <span style="display: inline-flex; align-items: center; gap: 8px;">
                <div onclick="cycleVariantColor(${index})" style="width: 24px; height: 24px; background-color: ${variant.color}; border: 3px solid #ffffff; border-radius: 50%; cursor: pointer; box-shadow: 0 0 6px rgba(0,0,0,0.5), inset 0 0 6px rgba(255,255,255,0.3); transition: all 0.2s;" title="🎨 Click to change color" onmouseover="this.style.transform='scale(1.2)'" onmouseout="this.style.transform='scale(1)'"></div>
                <span style="color: ${variant.color}; font-weight: bold; text-shadow: 1px 1px 2px rgba(0,0,0,0.5);">${variant.name}</span>
            </span>
            <span>W:${variant.window} C:${variant.candleTol}% M:${variant.macdTol}% | SL:${variant.stopLoss || 5}% TP:${variant.takeProfit || 25}% Risk:${variant.positionSize || 100}% | SD:${variant.startDate || '-'} ED:${variant.endDate || '-'} | TS:${variant.textSize || 10} LW:${variant.lineWidth || 1.5}</span>
            <div style="margin-left: 10px;">
                <label><input type="checkbox" id="classic_${index}" ${variant.showClassic ? 'checked' : ''} onchange="toggleVariantType(${index}, 'classic')"> Classic</label>
                <label><input type="checkbox" id="hidden_${index}" ${variant.showHidden ? 'checked' : ''} onchange="toggleVariantType(${index}, 'hidden')"> Hidden</label>
                <label><input type="checkbox" id="bullish_${index}" ${variant.showBullish ? 'checked' : ''} onchange="toggleVariantType(${index}, 'bullish')"> <span style="color: #00FF00;">Bullish</span></label>
                <label><input type="checkbox" id="bearish_${index}" ${variant.showBearish ? 'checked' : ''} onchange="toggleVariantType(${index}, 'bearish')"> <span style="color: #FF0000;">Bearish</span></label>
                <label><input type="checkbox" id="names_${index}" ${variant.showNames ? 'checked' : ''} onchange="toggleVariantType(${index}, 'names')"> Names</label>
                <label><input type="checkbox" id="numbers_${index}" ${variant.showNumbers ? 'checked' : ''} onchange="toggleVariantType(${index}, 'numbers')"> Numbers</label>
                <label><input type="checkbox" id="rsi_names_${index}" ${variant.showRSINames ? 'checked' : ''} onchange="toggleVariantType(${index}, 'rsi_names')"> RSI Names</label>
                <label><input type="checkbox" id="rsi_numbers_${index}" ${variant.showRSINumbers ? 'checked' : ''} onchange="toggleVariantType(${index}, 'rsi_numbers')"> RSI Numbers</label>
                <label><input type="checkbox" id="macd_names_${index}" ${variant.showMACDNames ? 'checked' : ''} onchange="toggleVariantType(${index}, 'macd_names')"> MACD Names</label>
                <label><input type="checkbox" id="macd_numbers_${index}" ${variant.showMACDNumbers ? 'checked' : ''} onchange="toggleVariantType(${index}, 'macd_numbers')"> MACD Numbers</label>
                <label><input type="checkbox" id="circles_${index}" ${variant.showCircles ? 'checked' : ''} onchange="toggleVariantType(${index}, 'circles')"> Circles</label>
                <label><input type="checkbox" id="success_${index}" ${variant.showSuccessRate ? 'checked' : ''} onchange="toggleVariantType(${index}, 'success')"> Success %</label>
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
            case 'success':
                variants[index].showSuccessRate = document.getElementById(`success_${index}`).checked;
                // Update visibility of success rate traces without replotting
                if (window.currentResults && sessionId) {
                    updateTextTraceVisibility('success', index);
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
                // Blue circles show what's missing from other variants compared to first variant
                // They should be controlled by the first variant's circles checkbox only
                matchesType = (variantIndex === 0);
            }
        } else if (type === 'success' && traceName.includes('Success Rate')) {
            // Success rate traces belong to the variant that creates them
            matchesType = traceName.includes(variantName);
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
    const removedVariant = variants.splice(index, 1)[0];
    
    // Memory cleanup: remove cached data for this variant
    if (removedVariant) {
        const keysToRemove = [];
        for (const [key, value] of indicatorCache.entries()) {
            if (key.includes(`_${removedVariant.window}_${removedVariant.candleTol}_${removedVariant.macdTol}`)) {
                keysToRemove.push(key);
            }
        }
        keysToRemove.forEach(key => indicatorCache.delete(key));
        
        // Clear any variant-specific stored data
        removedVariant.data = null;
        removedVariant.results = null;
    }
    
    updateVariantsList();
    console.log("Removed variant at index:", index, "Cleaned cache entries:", keysToRemove?.length || 0);
}

async function analyze() {
    if (!sessionId || variants.length === 0) {
        showMessage('messageContainer', 'Upload file and add variants first!', 'error');
        return;
    }
    
    // Performance: Prevent duplicate calls
    if (analyze._running) {
        console.log("Analysis already running, skipping duplicate");
        return;
    }
    analyze._running = true;
    
    console.log("Starting analysis with", variants.length, "variants");
    showMessage('messageContainer', 'Analysis running...', 'loading');
    
    try {
        // Performance: Check cache first
        const cacheKey = `${sessionId}_${JSON.stringify(variants.map(v => [v.window, v.candleTol, v.macdTol]))}`;
        const cached = getCachedIndicators(cacheKey);
        
        if (cached) {
            console.log("Using cached results");
            window.currentChartData = cached.chartData;
            window.currentResults = cached.results;
            
            // Non-blocking render
            requestAnimationFrame(() => {
                plotChart(cached.chartData, cached.results);
                document.getElementById('resultsPanel').style.display = 'block';
                showMessage('messageContainer', 'Analysis complete (cached)', 'success');
            });
            return;
        }

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
            
            // Cache results for next time
            cacheIndicators(cacheKey, data);
            
            // Store current data for toggle functionality
            window.currentChartData = data.chartData;
            window.currentResults = data.results;
            
            // Performance: Non-blocking render
            requestAnimationFrame(() => {
                plotChart(data.chartData, data.results);
                document.getElementById('resultsPanel').style.display = 'block';
                showMessage('messageContainer', 'Analysis complete', 'success');
            });
        } else {
            showMessage('messageContainer', `Analysis failed: ${data.error}`, 'error');
        }
    } catch (error) {
        console.error("Analysis error:", error);
        showMessage('messageContainer', `Analysis error: ${error.message}`, 'error');
    } finally {
        analyze._running = false;
    }
}

function sampleChartData(chartData, maxPoints = 5000) {
    if (!chartData.dates || chartData.dates.length <= maxPoints) {
        return chartData;
    }
    
    const ratio = Math.ceil(chartData.dates.length / maxPoints);
    const sampledData = {
        dates: [],
        open: [],
        high: [],
        low: [],
        close: [],
        rsi: [],
        macd_histogram: []
    };
    
    // Smart sampling: keep every nth point but preserve extremes
    for (let i = 0; i < chartData.dates.length; i += ratio) {
        const endIndex = Math.min(i + ratio - 1, chartData.dates.length - 1);
        
        // Find min/max in this segment for OHLC preservation
        let segmentHigh = -Infinity, segmentLow = Infinity;
        let highIdx = i, lowIdx = i;
        
        for (let j = i; j <= endIndex; j++) {
            if (chartData.high[j] > segmentHigh) {
                segmentHigh = chartData.high[j];
                highIdx = j;
            }
            if (chartData.low[j] < segmentLow) {
                segmentLow = chartData.low[j];
                lowIdx = j;
            }
        }
        
        // Use the segment end point, but preserve extremes
        const useIndex = endIndex;
        sampledData.dates.push(chartData.dates[useIndex]);
        sampledData.open.push(chartData.open[i]); // Start of segment
        sampledData.high.push(segmentHigh); // Highest in segment
        sampledData.low.push(segmentLow); // Lowest in segment
        sampledData.close.push(chartData.close[useIndex]); // End of segment
        sampledData.rsi.push(chartData.rsi[useIndex]);
        sampledData.macd_histogram.push(chartData.macd_histogram[useIndex]);
    }
    
    // Copy EMA data if available
    ['ema20', 'ema50', 'ema100', 'ema200'].forEach(ema => {
        if (chartData[ema]) {
            sampledData[ema] = [];
            for (let i = 0; i < chartData.dates.length; i += ratio) {
                const useIndex = Math.min(i + ratio - 1, chartData.dates.length - 1);
                sampledData[ema].push(chartData[ema][useIndex]);
            }
        }
    });
    
    console.log(`Sampled data from ${chartData.dates.length} to ${sampledData.dates.length} points`);
    return sampledData;
}

function plotChart(chartData, results) {
    console.log("🎯 PLOTCHART START - Data:", !!chartData, "Results:", !!results);
    
    try {
        // Apply intelligent sampling for large datasets
        console.log("📊 Starting data sampling...");
        const sampledData = sampleChartData(chartData);
        console.log("✅ Data sampling completed");
        
        console.log("Plotting chart with data:", { 
            original_length: chartData.dates?.length,
            sampled_length: sampledData.dates?.length, 
            ema20_available: !!sampledData.ema20
        });
        
        const traces = [
            // Candlestick
            {
                x: sampledData.dates,
                open: sampledData.open,
                high: sampledData.high,
                low: sampledData.low,
                close: sampledData.close,
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
                x: sampledData.dates,
                y: sampledData.rsi,
                type: 'scatter',
                name: 'RSI',
                xaxis: 'x',
                yaxis: 'y2',
                line: { color: '#FFA500', width: 1.5 },
                showlegend: true
            },
            // MACD Histogram
            {
                x: sampledData.dates,
                y: sampledData.macd_histogram,
                type: 'bar',
                name: 'MACD Histogram',
                xaxis: 'x',
                yaxis: 'y3',
                marker: { color: sampledData.macd_histogram?.map(v => v >= 0 ? '#00FF00' : '#FF0000') },
                showlegend: true
            }
        ];

        // Add EMAs
        if (sampledData.ema20) {
            traces.push({
                x: sampledData.dates,
                y: sampledData.ema20,
                type: 'scatter',
                name: 'EMA 20',
                xaxis: 'x',
                yaxis: 'y',
                line: { color: '#FFD700', width: 1.5 },
                showlegend: true
            });
        }
        if (sampledData.ema50) {
            traces.push({
                x: sampledData.dates,
                y: sampledData.ema50,
                type: 'scatter',
                name: 'EMA 50',
                xaxis: 'x',
                yaxis: 'y',
                line: { color: '#00FFFF', width: 1.5 },
                showlegend: true
            });
        }
        if (sampledData.ema100) {
            traces.push({
                x: sampledData.dates,
                y: sampledData.ema100,
                type: 'scatter',
                name: 'EMA 100',
                xaxis: 'x',
                yaxis: 'y',
                line: { color: '#FF00FF', width: 1.5 },
                showlegend: true
            });
        }
        if (sampledData.ema200) {
            traces.push({
                x: sampledData.dates,
                y: sampledData.ema200,
                type: 'scatter',
                name: 'EMA 200',
                xaxis: 'x',
                yaxis: 'y',
                line: { color: '#9370DB', width: 1.5 },
                showlegend: true
            });
        }

        // Add simple divergence markers for each variant
        console.log("🔍 Adding divergence traces for", variants.length, "variants");
        variants.forEach((variant, variantIndex) => {
            console.log(`📈 Processing variant ${variantIndex}: ${variant.name}`);
            const res = results[variant.id];
            if (!res) return;
            
            // Add classic divergence markers
            if (res.classic && res.classic.length > 0) {
                traces.push({
                    x: res.classic.map(d => d.date),
                    y: res.classic.map(d => d.low),
                    type: 'scatter',
                    mode: 'markers',
                    name: `${variant.name} Classic`,
                    marker: {
                        symbol: 'triangle-up',
                        size: 12,
                        color: variant.color
                    },
                    xaxis: 'x',
                    yaxis: 'y',
                    showlegend: true
                });
            }
            
            // Add hidden divergence markers
            if (res.hidden && res.hidden.length > 0) {
                traces.push({
                    x: res.hidden.map(d => d.date),
                    y: res.hidden.map(d => d.low),
                    type: 'scatter',
                    mode: 'markers',
                    name: `${variant.name} Hidden`,
                    marker: {
                        symbol: 'diamond',
                        size: 12,
                        color: variant.color
                    },
                    xaxis: 'x',
                    yaxis: 'y',
                    showlegend: true
                });
            }
        });

        // Basic layout for three subplots
        const layout = {
            title: {
                text: 'Bullish Divergence Analysis',
                font: { color: '#FFFFFF', size: 16 }
            },
            plot_bgcolor: 'rgba(15, 15, 30, 0.8)',
            paper_bgcolor: 'rgba(0, 0, 0, 0)',
            font: { color: '#FFFFFF' },
            xaxis: {
                domain: [0, 1],
                anchor: 'y3',
                showgrid: true,
                gridcolor: 'rgba(255,255,255,0.1)'
            },
            yaxis: {
                domain: [0.7, 1],
                title: 'Price',
                showgrid: true,
                gridcolor: 'rgba(255,255,255,0.1)'
            },
            yaxis2: {
                domain: [0.35, 0.65],
                title: 'RSI',
                showgrid: true,
                gridcolor: 'rgba(255,255,255,0.1)'
            },
            yaxis3: {
                domain: [0, 0.3],
                title: 'MACD',
                showgrid: true,
                gridcolor: 'rgba(255,255,255,0.1)'
            }
        };

        const config = {
            responsive: true,
            displayModeBar: true,
            displaylogo: false
        };

        // Create the plot
        console.log("🚀 Creating Plotly chart with", traces.length, "traces");
        Plotly.newPlot('mainChart', traces, layout, config);
        console.log("✅ Plotly chart created successfully");
        updateStats(results);

    } catch (error) {
        console.error("❌ PlotChart Error:", error);
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
    
    variants.forEach(variant => {
        const res = results[variant.id];
        if (res) {
            const classicCount = res.classic ? res.classic.length : 0;
            const hiddenCount = res.hidden ? res.hidden.length : 0;
            const totalCount = classicCount + hiddenCount;
            
            html += `<tr style="border-bottom: 1px solid rgba(255,255,255,0.1);">
                <td style="padding: 8px; color: ${variant.color}; font-weight: bold;">${variant.name}</td>
                <td style="text-align: center; padding: 8px;">${classicCount}</td>
                <td style="text-align: center; padding: 8px;">${hiddenCount}</td>
                <td style="text-align: center; padding: 8px; font-weight: bold;">${totalCount}</td>
            </tr>`;
        }
    });
    
    html += '</table></div></div>';
    container.innerHTML = html;
}

function jumpToMarker(targetDate, targetPrice) {
    const plotDiv = document.getElementById('mainChart');
    if (!plotDiv) return;
    
    try {
        Plotly.relayout('mainChart', {
            'xaxis.range': [new Date(Date.parse(targetDate) - 7*24*60*60*1000), new Date(Date.parse(targetDate) + 7*24*60*60*1000)],
            'yaxis.range': [targetPrice * 0.95, targetPrice * 1.05]
        });
        showMarkerHighlight(targetDate, targetPrice);
    } catch (error) {
        console.error('Error jumping to marker:', error);
    }
}

function showMarkerHighlight(targetDate, targetPrice) {
    setTimeout(() => {
        try {
            const update = {
                'annotations[0]': {
                    x: targetDate,
                    y: targetPrice,
                    xref: 'x',
                    yref: 'y',
                    text: '🎯',
                    showarrow: false,
                    font: { size: 20, color: 'yellow' }
                }
            };
            Plotly.relayout('mainChart', update);
            
            setTimeout(() => {
                Plotly.relayout('mainChart', {'annotations': []});
            }, 3000);
        } catch (error) {
            console.error('Error showing marker highlight:', error);
        }
    }, 100);
}

function showMessage(containerId, message, type) {
    const container = document.getElementById(containerId);
    const msg = document.createElement('div');
    msg.className = `message ${type}`;
    msg.textContent = message;
    container.appendChild(msg);
    
    setTimeout(() => {
        if (msg.parentNode) {
            msg.parentNode.removeChild(msg);
        }
    }, 5000);
}

// End of core functions