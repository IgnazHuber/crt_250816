import pandas as pd
import numpy as np
import warnings
from numba import njit

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)

@njit(cache=True)
def find_local_maxima_minima_jit(
    high_prices: np.ndarray,
    low_prices: np.ndarray, 
    macd_histogram: np.ndarray,
    window_1: int,
    window_2: int
) -> tuple:
    n = len(high_prices)
    LM_High_window_1_CS = np.zeros(n, dtype=np.float64)
    LM_High_window_2_CS = np.zeros(n, dtype=np.float64)
    LM_Low_window_1_CS = np.zeros(n, dtype=np.float64)
    LM_Low_window_2_CS = np.zeros(n, dtype=np.float64)
    LM_High_window_1_MACD = np.zeros(n, dtype=np.float64)
    LM_High_window_2_MACD = np.zeros(n, dtype=np.float64)
    LM_Low_window_1_MACD = np.zeros(n, dtype=np.float64)
    LM_Low_window_2_MACD = np.zeros(n, dtype=np.float64)
    Level_1_High_window_1_CS = np.zeros(n, dtype=np.float64)
    Level_1_Low_window_1_CS = np.zeros(n, dtype=np.float64)
    Level_2_High_window_1_CS = np.zeros(n, dtype=np.float64)
    Level_2_Low_window_1_CS = np.zeros(n, dtype=np.float64)
    
    # Candlestick local maxima (window_1)
    for i in range(window_1, n - window_1):
        is_max = True
        for j in range(i - window_1, i):
            if high_prices[i] <= high_prices[j]:
                is_max = False
                break
        if is_max:
            for j in range(i + 1, i + window_1 + 1):
                if high_prices[i] < high_prices[j]:
                    is_max = False
                    break
        if is_max:
            LM_High_window_1_CS[i] = high_prices[i]
    
    # Candlestick local maxima (window_2)
    for i in range(window_2, n - window_2):
        is_max = True
        for j in range(i - window_2, i):
            if high_prices[i] <= high_prices[j]:
                is_max = False
                break
        if is_max:
            for j in range(i + 1, i + window_2 + 1):
                if high_prices[i] < high_prices[j]:
                    is_max = False
                    break
        if is_max:
            LM_High_window_2_CS[i] = high_prices[i]
    
    # Candlestick local minima (window_1)
    for i in range(window_1, n - window_1):
        is_min = True
        for j in range(i - window_1, i):
            if low_prices[i] >= low_prices[j]:
                is_min = False
                break
        if is_min:
            for j in range(i + 1, i + window_1 + 1):
                if low_prices[i] > low_prices[j]:
                    is_min = False
                    break
        if is_min:
            LM_Low_window_1_CS[i] = low_prices[i]
    
    # Candlestick local minima (window_2)
    for i in range(window_2, n - window_2):
        is_min = True
        for j in range(i - window_2, i):
            if low_prices[i] >= low_prices[j]:
                is_min = False
                break
        if is_min:
            for j in range(i + 1, i + window_2 + 1):
                if low_prices[i] > low_prices[j]:
                    is_min = False
                    break
        if is_min:
            LM_Low_window_2_CS[i] = low_prices[i]
    
    # MACD local maxima (window_1)
    for i in range(window_1, n - window_1):
        is_max = True
        for j in range(i - window_1, i):
            if macd_histogram[i] <= macd_histogram[j]:
                is_max = False
                break
        if is_max:
            for j in range(i + 1, i + window_1 + 1):
                if macd_histogram[i] < macd_histogram[j]:
                    is_max = False
                    break
        if is_max:
            LM_High_window_1_MACD[i] = macd_histogram[i]
    
    # MACD local maxima (window_2)
    for i in range(window_2, n - window_2):
        is_max = True
        for j in range(i - window_2, i):
            if macd_histogram[i] <= macd_histogram[j]:
                is_max = False
                break
        if is_max:
            for j in range(i + 1, i + window_2 + 1):
                if macd_histogram[i] < macd_histogram[j]:
                    is_max = False
                    break
        if is_max:
            LM_High_window_2_MACD[i] = macd_histogram[i]
    
    # MACD local minima (window_1)
    for i in range(window_1, n - window_1):
        is_min = True
        for j in range(i - window_1, i):
            if macd_histogram[i] >= macd_histogram[j]:
                is_min = False
                break
        if is_min:
            for j in range(i + 1, i + window_1 + 1):
                if macd_histogram[i] > macd_histogram[j]:
                    is_min = False
                    break
        if is_min:
            LM_Low_window_1_MACD[i] = low_prices[i]
    
    # MACD local minima (window_2)
    for i in range(window_2, n - window_2):
        is_min = True
        for j in range(i - window_2, i):
            if macd_histogram[i] >= macd_histogram[j]:
                is_min = False
                break
        if is_min:
            for j in range(i + 1, i + window_2 + 1):
                if macd_histogram[i] > macd_histogram[j]:
                    is_min = False
                    break
        if is_min:
            LM_Low_window_2_MACD[i] = low_prices[i]
    
    # Level 1 Highs
    for i in range(window_1, n):
        prev_high_found = False
        for j in range(i - 1, -1, -1):
            if LM_High_window_1_CS[j] != 0:
                if LM_High_window_1_CS[i] != 0 and LM_High_window_1_CS[i] > LM_High_window_1_CS[j]:
                    Level_1_High_window_1_CS[i] = LM_High_window_1_CS[i]
                prev_high_found = True
                break
    
    # Level 1 Lows
    for i in range(window_1, n):
        prev_low_found = False
        for j in range(i - 1, -1, -1):
            if LM_Low_window_1_CS[j] != 0:
                if LM_Low_window_1_CS[i] != 0 and LM_Low_window_1_CS[i] < LM_Low_window_1_CS[j]:
                    Level_1_Low_window_1_CS[i] = LM_Low_window_1_CS[i]
                prev_low_found = True
                break
    
    # Level 2 Highs
    non_zero_high_indices = [i for i, v in enumerate(Level_1_High_window_1_CS) if v != 0]
    for idx, i in enumerate(non_zero_high_indices):
        current_val = Level_1_High_window_1_CS[i]
        is_peak = True
        if idx > 0 and current_val <= Level_1_High_window_1_CS[non_zero_high_indices[idx - 1]]:
            is_peak = False
        if idx < len(non_zero_high_indices) - 1 and current_val <= Level_1_High_window_1_CS[non_zero_high_indices[idx + 1]]:
            is_peak = False
        Level_2_High_window_1_CS[i] = current_val if is_peak else 0
    
    # Level 2 Lows
    non_zero_low_indices = [i for i, v in enumerate(Level_1_Low_window_1_CS) if v != 0]
    for idx, i in enumerate(non_zero_low_indices):
        current_val = Level_1_Low_window_1_CS[i]
        is_trough = True
        if idx > 0 and current_val >= Level_1_Low_window_1_CS[non_zero_low_indices[idx - 1]]:
            is_trough = False
        if idx < len(non_zero_low_indices) - 1 and current_val >= Level_1_Low_window_1_CS[non_zero_low_indices[idx + 1]]:
            is_trough = False
        Level_2_Low_window_1_CS[i] = current_val if is_trough else 0
    
    return (
        LM_High_window_1_CS, LM_High_window_2_CS, LM_Low_window_1_CS, LM_Low_window_2_CS,
        LM_High_window_1_MACD, LM_High_window_2_MACD, LM_Low_window_1_MACD, LM_Low_window_2_MACD,
        Level_1_High_window_1_CS, Level_1_Low_window_1_CS,
        Level_2_High_window_1_CS, Level_2_Low_window_1_CS
    )

def Local_Max_Min(ohlc: pd.DataFrame) -> pd.DataFrame:
    high_prices = ohlc["high"].to_numpy(dtype=np.float64)
    low_prices = ohlc["low"].to_numpy(dtype=np.float64)
    macd_histogram = ohlc["macd_histogram"].to_numpy(dtype=np.float64)
    window_1 = 5
    window_2 = 1
    results = find_local_maxima_minima_jit(high_prices, low_prices, macd_histogram, window_1, window_2)
    (LM_High_window_1_CS, LM_High_window_2_CS, LM_Low_window_1_CS, LM_Low_window_2_CS,
     LM_High_window_1_MACD, LM_High_window_2_MACD, LM_Low_window_1_MACD, LM_Low_window_2_MACD,
     Level_1_High_window_1_CS, Level_1_Low_window_1_CS,
     Level_2_High_window_1_CS, Level_2_Low_window_1_CS) = results
    ohlc['LM_High_window_1_CS'] = LM_High_window_1_CS
    ohlc['LM_High_window_2_CS'] = LM_High_window_2_CS
    ohlc['LM_Low_window_1_CS'] = LM_Low_window_1_CS
    ohlc['LM_Low_window_2_CS'] = LM_Low_window_2_CS
    ohlc['LM_High_window_1_MACD'] = LM_High_window_1_MACD
    ohlc['LM_High_window_2_MACD'] = LM_High_window_2_MACD
    ohlc['LM_Low_window_1_MACD'] = LM_Low_window_1_MACD
    ohlc['LM_Low_window_2_MACD'] = LM_Low_window_2_MACD
    ohlc['Level_1_High_window_1_CS'] = Level_1_High_window_1_CS
    ohlc['Level_1_Low_window_1_CS'] = Level_1_Low_window_1_CS
    ohlc['Level_2_High_window_1_CS'] = Level_2_High_window_1_CS
    ohlc['Level_2_Low_window_1_CS'] = Level_2_Low_window_1_CS
    return ohlc