// indicator-calculator.js - Technische Indikatoren (basierend auf Initialize_RSI_EMA_MACD.py)
class IndicatorCalculator {
    constructor() {
        this.RSI_PERIOD = 14;
        this.EMA_SPANS = [12, 20, 26, 50, 100, 200];
        this.MACD_SIGNAL_PERIOD = 9;
    }
    
    calculateAllIndicators(data) {
        if (!data || data.length < this.RSI_PERIOD + 1) {
            throw new Error(`Mindestens ${this.RSI_PERIOD + 1} Datenpunkte erforderlich`);
        }
        
        const closes = data.map(d => d.close);
        
        // RSI berechnen
        const rsiData = this.calculateRSI(closes, this.RSI_PERIOD);
        
        // EMAs berechnen
        const emas = {};
        for (const span of this.EMA_SPANS) {
            emas[`EMA_${span}`] = this.calculateEMA(closes, span);
        }
        
        // MACD berechnen
        const macdData = this.calculateMACD(closes);
        
        // Daten zusammenführen
        const processedData = data.map((row, i) => ({
            ...row,
            ...rsiData[i],
            ...Object.keys(emas).reduce((acc, key) => {
                acc[key] = emas[key][i];
                return acc;
            }, {}),
            macd: macdData.macd[i],
            signal: macdData.signal[i],
            macd_histogram: macdData.histogram[i]
        }));
        
        return processedData;
    }
    
    calculateRSI(prices, period = 14) {
        const n = prices.length;
        const rsiData = [];
        
        // Price changes
        const priceChanges = [];
        const gains = [];
        const losses = [];
        
        for (let i = 0; i < n; i++) {
            if (i === 0) {
                priceChanges.push(0);
                gains.push(0);
                losses.push(0);
            } else {
                const change = prices[i] - prices[i - 1];
                priceChanges.push(change);
                gains.push(change > 0 ? change : 0);
                losses.push(change < 0 ? -change : 0);
            }
        }
        
        // Calculate average gains and losses
        const avgGains = [];
        const avgLosses = [];
        const rsValues = [];
        const rsiValues = [];
        
        for (let i = 0; i < n; i++) {
            if (i < period) {
                avgGains.push(null);
                avgLosses.push(null);
                rsValues.push(null);
                rsiValues.push(null);
            } else if (i === period) {
                // Initial average (SMA)
                const initialAvgGain = gains.slice(1, period + 1).reduce((a, b) => a + b, 0) / period;
                const initialAvgLoss = losses.slice(1, period + 1).reduce((a, b) => a + b, 0) / period;
                
                avgGains.push(initialAvgGain);
                avgLosses.push(initialAvgLoss);
                
                const rs = initialAvgLoss === 0 ? 100 : initialAvgGain / initialAvgLoss;
                rsValues.push(rs);
                rsiValues.push(100 - (100 / (1 + rs)));
            } else {
                // EMA calculation
                const alpha = 1.0 / period;
                const avgGain = alpha * gains[i] + (1 - alpha) * avgGains[i - 1];
                const avgLoss = alpha * losses[i] + (1 - alpha) * avgLosses[i - 1];
                
                avgGains.push(avgGain);
                avgLosses.push(avgLoss);
                
                const rs = avgLoss === 0 ? 100 : avgGain / avgLoss;
                rsValues.push(rs);
                rsiValues.push(100 - (100 / (1 + rs)));
            }
            
            rsiData.push({
                price_change: priceChanges[i],
                gain: gains[i],
                loss: losses[i],
                avg_gain: avgGains[i],
                avg_loss: avgLosses[i],
                RS: rsValues[i],
                RSI: rsiValues[i]
            });
        }
        
        return rsiData;
    }
    
    calculateEMA(prices, period) {
        const ema = [];
        const multiplier = 2 / (period + 1);
        
        // Initial SMA
        let sum = 0;
        for (let i = 0; i < prices.length; i++) {
            if (i < period - 1) {
                sum += prices[i];
                ema.push(null);
            } else if (i === period - 1) {
                sum += prices[i];
                ema.push(sum / period);
            } else {
                const emaValue = (prices[i] - ema[i - 1]) * multiplier + ema[i - 1];
                ema.push(emaValue);
            }
        }
        
        return ema;
    }
    
    calculateMACD(prices) {
        const ema12 = this.calculateEMA(prices, 12);
        const ema26 = this.calculateEMA(prices, 26);
        
        const macd = [];
        const signal = [];
        const histogram = [];
        
        // MACD line
        for (let i = 0; i < prices.length; i++) {
            if (ema12[i] !== null && ema26[i] !== null) {
                macd.push(ema12[i] - ema26[i]);
            } else {
                macd.push(null);
            }
        }
        
        // Signal line (9-period EMA of MACD)
        const validMacd = macd.filter(v => v !== null);
        const signalLine = this.calculateEMA(validMacd, 9);
        
        let signalIndex = 0;
        for (let i = 0; i < macd.length; i++) {
            if (macd[i] !== null) {
                const sig = signalLine[signalIndex++];
                signal.push(sig);
                if (sig !== null) {
                    histogram.push(macd[i] - sig);
                } else {
                    histogram.push(null);
                }
            } else {
                signal.push(null);
                histogram.push(null);
            }
        }
        
        return { macd, signal, histogram };
    }
    
    // Lokale Maxima/Minima finden (aus Local_Maximas_Minimas.py)
    findLocalExtrema(data, window = 5) {
        const n = data.length;
        const processedData = [...data];
        
        // Arrays für lokale Extrema
        const extremaFields = {
            LM_High_window_1_CS: [],
            LM_High_window_2_CS: [],
            LM_Low_window_1_CS: [],
            LM_Low_window_2_CS: [],
            LM_High_window_1_MACD: [],
            LM_High_window_2_MACD: [],
            LM_Low_window_1_MACD: [],
            LM_Low_window_2_MACD: []
        };
        
        // Initialisieren
        for (let i = 0; i < n; i++) {
            Object.keys(extremaFields).forEach(key => {
                extremaFields[key][i] = 0;
            });
        }
        
        // Lokale Maxima/Minima für Candlesticks (window_1 = 5)
        for (let i = window; i < n - window; i++) {
            const high = data[i].high;
            const low = data[i].low;
            const macdHist = data[i].macd_histogram;
            
            // High window_1
            let isMax = true;
            for (let j = i - window; j < i; j++) {
                if (data[j].high >= high) isMax = false;
            }
            for (let j = i + 1; j <= i + window && j < n; j++) {
                if (data[j].high > high) isMax = false;
            }
            if (isMax) extremaFields.LM_High_window_1_CS[i] = high;
            
            // Low window_1
            let isMin = true;
            for (let j = i - window; j < i; j++) {
                if (data[j].low <= low) isMin = false;
            }
            for (let j = i + 1; j <= i + window && j < n; j++) {
                if (data[j].low < low) isMin = false;
            }
            if (isMin) extremaFields.LM_Low_window_1_CS[i] = low;
            
            // MACD-basierte Extrema
            if (macdHist !== null && macdHist !== undefined) {
                let isMacdMax = true;
                let isMacdMin = true;
                
                for (let j = i - window; j < i; j++) {
                    if (data[j].macd_histogram >= macdHist) isMacdMax = false;
                    if (data[j].macd_histogram <= macdHist) isMacdMin = false;
                }
                for (let j = i + 1; j <= i + window && j