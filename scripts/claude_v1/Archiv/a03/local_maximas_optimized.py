"""
Optimierte Version der Local_Maximas_Minimas.py mit Vektorisierung
Performance-Verbesserung durch NumPy-basierte Operationen statt Loops
"""

import pandas as pd
import numpy as np
import warnings
from scipy.signal import argrelextrema, find_peaks
import logging

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)

logger = logging.getLogger(__name__)

def find_local_extrema_vectorized(data: np.ndarray, window_size: int, find_maxima: bool = True) -> np.ndarray:
    """
    Vektorisierte Suche nach lokalen Extrema
    
    Args:
        data: 1D NumPy Array der Daten
        window_size: Fenstergröße für Extrema-Suche
        find_maxima: True für Maxima, False für Minima
        
    Returns:
        Array mit Extrema-Werten (0 für keine Extrema)
    """
    n = len(data)
    extrema = np.zeros(n)
    
    if n < 2 * window_size + 1:
        return extrema
    
    # Verwende scipy.signal für effiziente Peak-Detection
    if find_maxima:
        # Finde Maxima
        peaks, _ = find_peaks(data, distance=window_size)
        
        # Validiere Peaks mit symmetrischem Fenster
        for peak in peaks:
            start_idx = max(0, peak - window_size)
            end_idx = min(n, peak + window_size + 1)
            
            # Prüfe ob Peak wirklich höchster Wert im Fenster ist
            window_data = data[start_idx:end_idx]
            peak_in_window = peak - start_idx
            
            if peak_in_window < len(window_data) and data[peak] == np.max(window_data):
                # Zusätzliche Bedingung: links strikt größer, rechts größer-gleich
                left_ok = peak == start_idx or all(data[peak] > data[j] for j in range(start_idx, peak))
                right_ok = peak == end_idx - 1 or all(data[peak] >= data[j] for j in range(peak + 1, end_idx))
                
                if left_ok and right_ok:
                    extrema[peak] = data[peak]
    else:
        # Finde Minima (invertiere Daten)
        inverted_data = -data
        peaks, _ = find_peaks(inverted_data, distance=window_size)
        
        # Validiere Minima
        for peak in peaks:
            start_idx = max(0, peak - window_size)
            end_idx = min(n, peak + window_size + 1)
            
            window_data = data[start_idx:end_idx]
            peak_in_window = peak - start_idx
            
            if peak_in_window < len(window_data) and data[peak] == np.min(window_data):
                # Zusätzliche Bedingung: links strikt kleiner, rechts kleiner-gleich
                left_ok = peak == start_idx or all(data[peak] < data[j] for j in range(start_idx, peak))
                right_ok = peak == end_idx - 1 or all(data[peak] <= data[j] for j in range(peak + 1, end_idx))
                
                if left_ok and right_ok:
                    extrema[peak] = data[peak]
    
    return extrema

def find_level_n_extrema(level_n_minus_1: np.ndarray, find_maxima: bool = True) -> np.ndarray:
    """
    Findet Level-N Extrema basierend auf Level-(N-1) Extrema
    
    Args:
        level_n_minus_1: Array mit Level-(N-1) Extrema
        find_maxima: True für Maxima, False für Minima
        
    Returns:
        Array mit Level-N Extrema
    """
    n = len(level_n_minus_1)
    level_n = np.zeros(n)
    
    # Finde alle non-zero Indizes
    non_zero_indices = np.where(level_n_minus_1 != 0)[0]
    
    if len(non_zero_indices) < 3:
        return level_n
    
    # Vektorisierte Vergleiche
    for i, idx in enumerate(non_zero_indices):
        current_val = level_n_minus_1[idx]
        is_extremum = True
        
        # Vergleiche mit vorherigem Wert
        if i > 0:
            prev_val = level_n_minus_1[non_zero_indices[i - 1]]
            if find_maxima:
                if current_val <= prev_val:
                    is_extremum = False
            else:
                if current_val >= prev_val:
                    is_extremum = False
        
        # Vergleiche mit nächstem Wert
        if i < len(non_zero_indices) - 1:
            next_val = level_n_minus_1[non_zero_indices[i + 1]]
            if find_maxima:
                if current_val <= next_val:
                    is_extremum = False
            else:
                if current_val >= next_val:
                    is_extremum = False
        
        if is_extremum:
            level_n[idx] = current_val
    
    return level_n

def Local_Max_Min_Optimized(ohlc: pd.DataFrame) -> pd.DataFrame:
    """
    Optimierte Version der Local_Max_Min Funktion mit Vektorisierung
    
    Args:
        ohlc: DataFrame mit OHLC-Daten und technischen Indikatoren
        
    Returns:
        DataFrame mit zusätzlichen Extrema-Spalten
    """
    logger.info("Starte optimierte Extrema-Berechnung...")
    
    # Parameter
    window_1 = 5  # Slow moving Window
    window_2 = 1  # Fast moving Window
    
    # Konvertiere zu NumPy für bessere Performance
    n = len(ohlc)
    low_prices = ohlc["low"].to_numpy()
    high_prices = ohlc["high"].to_numpy()
    macd_histogram = ohlc["macd_histogram"].to_numpy()
    
    logger.debug(f"Verarbeite {n} Datenpunkte...")
    
    # ================== CANDLESTICK EXTREMA ==================
    
    # Lokale Maxima - Candlestick (beide Fenstergrößen)
    logger.debug("Berechne Candlestick Maxima...")
    lm_high_window_1_cs = find_local_extrema_vectorized(high_prices, window_1, find_maxima=True)
    lm_high_window_2_cs = find_local_extrema_vectorized(high_prices, window_2, find_maxima=True)
    
    # Lokale Minima - Candlestick (beide Fenstergrößen)
    logger.debug("Berechne Candlestick Minima...")
    lm_low_window_1_cs = find_local_extrema_vectorized(low_prices, window_1, find_maxima=False)
    lm_low_window_2_cs = find_local_extrema_vectorized(low_prices, window_2, find_maxima=False)
    
    # ================== MACD EXTREMA ==================
    
    # Lokale Maxima - MACD (beide Fenstergrößen)
    logger.debug("Berechne MACD Maxima...")
    macd_high_window_1 = find_local_extrema_vectorized(macd_histogram, window_1, find_maxima=True)
    macd_high_window_2 = find_local_extrema_vectorized(macd_histogram, window_2, find_maxima=True)
    
    # Für MACD Maxima verwenden wir die High-Preise an den MACD Peak-Positionen
    lm_high_window_1_macd = np.where(macd_high_window_1 != 0, high_prices, 0)
    lm_high_window_2_macd = np.where(macd_high_window_2 != 0, high_prices, 0)
    
    # Lokale Minima - MACD (beide Fenstergrößen)
    logger.debug("Berechne MACD Minima...")
    macd_low_window_1 = find_local_extrema_vectorized(macd_histogram, window_1, find_maxima=False)
    macd_low_window_2 = find_local_extrema_vectorized(macd_histogram, window_2, find_maxima=False)
    
    # Für MACD Minima verwenden wir die Low-Preise an den MACD Trough-Positionen
    lm_low_window_1_macd = np.where(macd_low_window_1 != 0, low_prices, 0)
    lm_low_window_2_macd = np.where(macd_low_window_2 != 0, low_prices, 0)
    
    # ================== LEVEL EXTREMA ==================
    
    # Level 1 Extrema (basierend auf window_1 Extrema)
    logger.debug("Berechne Level 1 Extrema...")
    level_1_high_window_1_cs = find_level_n_extrema(lm_high_window_1_cs, find_maxima=True)
    level_1_low_window_1_cs = find_level_n_extrema(lm_low_window_1_cs, find_maxima=False)
    
    # Level 2 Extrema (basierend auf Level 1 Extrema)
    logger.debug("Berechne Level 2 Extrema...")
    level_2_high_window_1_cs = find_level_n_extrema(level_1_high_window_1_cs, find_maxima=True)
    level_2_low_window_1_cs = find_level_n_extrema(level_1_low_window_1_cs, find_maxima=False)
    
    # ================== ERGEBNISSE SPEICHERN ==================
    
    logger.debug("Speichere Ergebnisse...")
    
    # Candlestick Extrema
    ohlc['LM_High_window_1_CS'] = lm_high_window_1_cs
    ohlc['LM_High_window_2_CS'] = lm_high_window_2_cs
    ohlc['LM_Low_window_1_CS'] = lm_low_window_1_cs
    ohlc['LM_Low_window_2_CS'] = lm_low_window_2_cs
    
    # MACD Extrema
    ohlc['LM_High_window_1_MACD'] = lm_high_window_1_macd
    ohlc['LM_High_window_2_MACD'] = lm_high_window_2_macd
    ohlc['LM_Low_window_1_MACD'] = lm_low_window_1_macd
    ohlc['LM_Low_window_2_MACD'] = lm_low_window_2_macd
    
    # Level Extrema
    ohlc['Level_1_High_window_1_CS'] = level_1_high_window_1_cs
    ohlc['Level_1_Low_window_1_CS'] = level_1_low_window_1_cs
    ohlc['Level_2_High_window_1_CS'] = level_2_high_window_1_cs
    ohlc['Level_2_Low_window_1_CS'] = level_2_low_window_1_cs
    
    # Statistiken
    stats = {
        'candlestick_highs_w1': np.count_nonzero(lm_high_window_1_cs),
        'candlestick_lows_w1': np.count_nonzero(lm_low_window_1_cs),
        'candlestick_highs_w2': np.count_nonzero(lm_high_window_2_cs),
        'candlestick_lows_w2': np.count_nonzero(lm_low_window_2_cs),
        'macd_highs_w1': np.count_nonzero(lm_high_window_1_macd),
        'macd_lows_w1': np.count_nonzero(lm_low_window_1_macd),
        'level_1_highs': np.count_nonzero(level_1_high_window_1_cs),
        'level_1_lows': np.count_nonzero(level_1_low_window_1_cs),
        'level_2_highs': np.count_nonzero(level_2_high_window_1_cs),
        'level_2_lows': np.count_nonzero(level_2_low_window_1_cs)
    }
    
    logger.info(f"Extrema-Berechnung abgeschlossen. Statistiken: {stats}")
    
    return ohlc

# Legacy-Kompatibilität: Alias für die ursprüngliche Funktion
def Local_Max_Min(ohlc: pd.DataFrame) -> pd.DataFrame:
    """
    Legacy-Wrapper für die optimierte Funktion
    """
    return Local_Max_Min_Optimized(ohlc)
