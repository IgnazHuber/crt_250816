"""
Enhanced Divergence System - Erweiterte Divergenz-Erkennung
Implementiert Hidden Bullish und Bearish Divergenzen
"""

import pandas as pd
import numpy as np
import warnings
import logging

warnings.filterwarnings("ignore", category=FutureWarning)
logger = logging.getLogger(__name__)

def Enhanced_Divergence_Analysis(ohlc: pd.DataFrame, window: int, 
                                candle_tol: float, macd_tol: float,
                                enable_hidden: bool = True,
                                enable_bearish: bool = True) -> pd.DataFrame:
    """
    Erweiterte Divergenz-Analyse mit Hidden Bullish und Bearish Divergenzen
    
    Args:
        ohlc: DataFrame mit OHLC-Daten und technischen Indikatoren
        window: Fenster-Größe für Extrema-Suche
        candle_tol: Toleranz für Candlestick-Preise (%)
        macd_tol: Toleranz für MACD-Werte (%)
        enable_hidden: Aktiviert Hidden Bullish Divergenzen
        enable_bearish: Aktiviert Bearish Divergenzen
        
    Returns:
        DataFrame mit zusätzlichen Divergenz-Spalten
    """
    logger.info("Starte erweiterte Divergenz-Analyse...")
    
    # Konvertiere notwendige Spalten zu numerisch
    numeric_columns = [
        'LM_Low_window_2_CS', 'LM_Low_window_1_CS',
        'LM_Low_window_2_MACD', 'LM_Low_window_1_MACD',
        'LM_High_window_2_CS', 'LM_High_window_1_CS',  # Für Bearish
        'LM_High_window_2_MACD', 'LM_High_window_1_MACD',  # Für Bearish
        'RSI', 'macd_histogram'
    ]
    
    for col in numeric_columns:
        if col in ohlc.columns:
            ohlc[col] = pd.to_numeric(ohlc[col], errors='coerce')
    
    # Parameter
    window_1 = window
    window_2 = 1
    RSI_tol = 2
    
    # Basis-Daten extrahieren
    n = len(ohlc)
    dates = pd.to_datetime(ohlc['date']).to_numpy()
    CL = ohlc["low"].to_numpy()
    CH = ohlc["high"].to_numpy()
    RSI = ohlc['RSI'].to_numpy()
    macd_histogram = ohlc['macd_histogram'].to_numpy()
    
    # ================== HIDDEN BULLISH DIVERGENZEN ==================
    
    if enable_hidden and 'LM_Low_window_2_CS' in ohlc.columns:
        logger.info("Analysiere Hidden Bullish Divergenzen...")
        
        # Hidden Bullish: Höhere Lows im Preis, niedrigere Lows in Indikatoren
        LM_Low_window_2_CS = ohlc['LM_Low_window_2_CS'].to_numpy()
        LM_Low_window_1_CS = ohlc['LM_Low_window_1_CS'].to_numpy()
        
        # Initialize Hidden Bullish Arrays
        HBullD = np.zeros(n)
        HBullD_Lower_Low = np.zeros(n)
        HBullD_Higher_Low = np.zeros(n)
        HBullD_Lower_Low_RSI = np.zeros(n)
        HBullD_Higher_Low_RSI = np.zeros(n)
        HBullD_Lower_Low_MACD = np.zeros(n)
        HBullD_Higher_Low_MACD = np.zeros(n)
        HBullD_Lower_Low_date = np.zeros(n, dtype=object)
        HBullD_Higher_Low_date = np.zeros(n, dtype=object)
        HBullD_Date_Gap = np.zeros(n)
        
        # Finde Hidden Bullish Divergenzen
        valid_indices = np.where(LM_Low_window_2_CS > 0)[0]
        valid_indices = valid_indices[(valid_indices >= window_1) & (valid_indices < n - window_2)]
        
        for i in valid_indices:
            # Finde letzten non-zero LM_Low_window_1_CS vor Index i
            prev_nonzero = np.where(LM_Low_window_1_CS[:i] != 0)[0]
            if len(prev_nonzero) == 0:
                continue
            
            j = prev_nonzero[-1]
            
            # Hidden Bullish Bedingungen
            current_low = LM_Low_window_2_CS[i]
            previous_low = LM_Low_window_1_CS[j]
            
            current_rsi = RSI[i]
            previous_rsi = RSI[j]
            
            current_macd = macd_histogram[i]
            previous_macd = macd_histogram[j]
            
            # Hidden Bullish: Preis macht höhere Lows, aber Indikatoren niedrigere Lows
            price_condition = current_low > previous_low  # Höhere Lows im Preis
            rsi_condition = current_rsi < previous_rsi    # Niedrigere Lows im RSI
            macd_condition = current_macd < previous_macd # Niedrigere Lows im MACD
            
            # Toleranz-Checks
            price_diff_percent = abs(100 * (current_low - previous_low) / previous_low)
            rsi_diff = abs(current_rsi - previous_rsi)
            macd_diff_percent = abs(100 * (current_macd - previous_macd) / previous_macd) if previous_macd != 0 else 0
            
            # Angepasste Bedingungen für Hidden Divergenzen
            if (price_condition and rsi_condition and macd_condition and
                price_diff_percent > candle_tol and  # Signifikanter Preisunterschied
                rsi_diff > RSI_tol and              # Signifikanter RSI-Unterschied
                macd_diff_percent > macd_tol):      # Signifikanter MACD-Unterschied
                
                HBullD[i] = 1
                HBullD_Lower_Low[i] = previous_low
                HBullD_Higher_Low[i] = current_low
                HBullD_Lower_Low_RSI[i] = previous_rsi
                HBullD_Higher_Low_RSI[i] = current_rsi
                HBullD_Lower_Low_MACD[i] = previous_macd
                HBullD_Higher_Low_MACD[i] = current_macd
                HBullD_Lower_Low_date[i] = dates[j]
                HBullD_Higher_Low_date[i] = dates[i]
                HBullD_Date_Gap[i] = i - j
        
        # Speichere Hidden Bullish Ergebnisse
        ohlc['HBullD'] = HBullD
        ohlc['HBullD_Lower_Low'] = HBullD_Lower_Low
        ohlc['HBullD_Higher_Low'] = HBullD_Higher_Low
        ohlc['HBullD_Lower_Low_RSI'] = HBullD_Lower_Low_RSI
        ohlc['HBullD_Higher_Low_RSI'] = HBullD_Higher_Low_RSI
        ohlc['HBullD_Lower_Low_MACD'] = HBullD_Lower_Low_MACD
        ohlc['HBullD_Higher_Low_MACD'] = HBullD_Higher_Low_MACD
        ohlc['HBullD_Lower_Low_date'] = HBullD_Lower_Low_date
        ohlc['HBullD_Higher_Low_date'] = HBullD_Higher_Low_date
        ohlc['HBullD_Date_Gap'] = HBullD_Date_Gap
        
        hidden_count = (HBullD == 1).sum()
        logger.info(f"✅ Hidden Bullish Divergenzen gefunden: {hidden_count}")
    
    # ================== BEARISH DIVERGENZEN ==================
    
    if enable_bearish and 'LM_High_window_2_CS' in ohlc.columns:
        logger.info("Analysiere Bearish Divergenzen...")
        
        LM_High_window_2_CS = ohlc['LM_High_window_2_CS'].to_numpy()
        LM_High_window_1_CS = ohlc['LM_High_window_1_CS'].to_numpy()
        
        # Initialize Bearish Arrays
        BearD = np.zeros(n)
        BearD_Higher_High = np.zeros(n)
        BearD_Lower_High = np.zeros(n)
        BearD_Higher_High_RSI = np.zeros(n)
        BearD_Lower_High_RSI = np.zeros(n)
        BearD_Higher_High_MACD = np.zeros(n)
        BearD_Lower_High_MACD = np.zeros(n)
        BearD_Higher_High_date = np.zeros(n, dtype=object)
        BearD_Lower_High_date = np.zeros(n, dtype=object)
        BearD_Date_Gap = np.zeros(n)
        
        # Finde Classic Bearish Divergenzen
        valid_indices = np.where(LM_High_window_2_CS > 0)[0]
        valid_indices = valid_indices[(valid_indices >= window_1) & (valid_indices < n - window_2)]
        
        for i in valid_indices:
            # Finde letzten non-zero LM_High_window_1_CS vor Index i
            prev_nonzero = np.where(LM_High_window_1_CS[:i] != 0)[0]
            if len(prev_nonzero) == 0:
                continue
            
            j = prev_nonzero[-1]
            
            # Bearish Bedingungen
            current_high = LM_High_window_2_CS[i]
            previous_high = LM_High_window_1_CS[j]
            
            current_rsi = RSI[i]
            previous_rsi = RSI[j]
            
            current_macd = macd_histogram[i]
            previous_macd = macd_histogram[j]
            
            # Classic Bearish: Preis macht höhere Highs, aber Indikatoren niedrigere Highs
            price_condition = current_high > previous_high  # Höhere Highs im Preis
            rsi_condition = current_rsi < previous_rsi      # Niedrigere Highs im RSI
            macd_condition = current_macd < previous_macd   # Niedrigere Highs im MACD
            
            # Toleranz-Checks
            price_diff_percent = abs(100 * (current_high - previous_high) / previous_high)
            rsi_diff = abs(current_rsi - previous_rsi)
            
            # Zusätzliche RSI-Bedingungen für Bearish (typisch bei hohen RSI-Werten)
            rsi_condition_extended = (
                (rsi_condition) or
                (current_rsi > 60 and abs(rsi_diff) < 4 * RSI_tol)  # Bei hohen RSI-Werten
            )
            
            if (price_condition and rsi_condition_extended and macd_condition and
                price_diff_percent > candle_tol):
                
                BearD[i] = 1
                BearD_Higher_High[i] = current_high
                BearD_Lower_High[i] = previous_high
                BearD_Higher_High_RSI[i] = current_rsi
                BearD_Lower_High_RSI[i] = previous_rsi
                BearD_Higher_High_MACD[i] = current_macd
                BearD_Lower_High_MACD[i] = previous_macd
                BearD_Higher_High_date[i] = dates[i]
                BearD_Lower_High_date[i] = dates[j]
                BearD_Date_Gap[i] = i - j
        
        # Speichere Bearish Ergebnisse
        ohlc['BearD'] = BearD
        ohlc['BearD_Higher_High'] = BearD_Higher_High
        ohlc['BearD_Lower_High'] = BearD_Lower_High
        ohlc['BearD_Higher_High_RSI'] = BearD_Higher_High_RSI
        ohlc['BearD_Lower_High_RSI'] = BearD_Lower_High_RSI
        ohlc['BearD_Higher_High_MACD'] = BearD_Higher_High_MACD
        ohlc['BearD_Lower_High_MACD'] = BearD_Lower_High_MACD
        ohlc['BearD_Higher_High_date'] = BearD_Higher_High_date
        ohlc['BearD_Lower_High_date'] = BearD_Lower_High_date
        ohlc['BearD_Date_Gap'] = BearD_Date_Gap
        
        bearish_count = (BearD == 1).sum()
        logger.info(f"✅ Bearish Divergenzen gefunden: {bearish_count}")
        
        
        # ================== HIDDEN BEARISH DIVERGENZEN ==================
        
        # Hidden Bearish: Niedrigere Highs im Preis, höhere Highs in Indikatoren
        HBearD = np.zeros(n)
        HBearD_Higher_High = np.zeros(n)
        HBearD_Lower_High = np.zeros(n)
        HBearD_Higher_High_RSI = np.zeros(n)
        HBearD_Lower_High_RSI = np.zeros(n)
        HBearD_Higher_High_MACD = np.zeros(n)
        HBearD_Lower_High_MACD = np.zeros(n)
        HBearD_Higher_High_date = np.zeros(n, dtype=object)
        HBearD_Lower_High_date = np.zeros(n, dtype=object)
        HBearD_Date_Gap = np.zeros(n)
        HBearD_Higher