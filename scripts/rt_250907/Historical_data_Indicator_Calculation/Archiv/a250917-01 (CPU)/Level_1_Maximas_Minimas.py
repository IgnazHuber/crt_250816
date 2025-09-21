import pandas as pd
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def Level_1_Max_Min(ohlc: pd.DataFrame) -> pd.Series:
    # Extract columns to lists for faster repeated access
    CL = ohlc["low"].tolist()
    CH = ohlc["high"].tolist()
    macd = ohlc["macd_histogram"].tolist()

    window_1 = 5    #  Slow moving Window
    window_2 = 1    #  Fast moving Window
    n = len(CL)

    # Initializing for local maximas in the Data based on Candlestick
    LM_High_window_1_CS = [0.0] * n
    LM_High_window_2_CS = [0.0] * n

    # Initializing for local minimas in the Data based on Candlestick
    LM_Low_window_1_CS = [0.0] * n
    LM_Low_window_2_CS = [0.0] * n

    # Initializing for local maximas in the Data based on MACD Highs
    LM_High_window_1_MACD = [0.0] * n
    LM_High_window_2_MACD = [0.0] * n

    # Initializing for local minimas in the Data based on MACD Highs
    LM_Low_window_1_MACD = [0.0] * n
    LM_Low_window_2_MACD = [0.0] * n

    # Initializing for Level 1 Highs based on Candlestick (comparing non-zero LM_High_window_1_CS)
    Level_1_High_window_1_CS = [0.0] * n
    Level_1_Low_window_1_CS = [0.0] * n

#---------------------------------------------------------------------------------------------------------
    # Locating Candlestick local maximas in the Data based on Candlestick for window-size=4 (slow moving)
    for i in range(window_1, n - window_1):
        if all(CH[i] > CH[j] for j in range(i - window_1, i)) and all(CH[i] >= CH[j] for j in range(i + 1, i + window_1 + 1)): # Searching for maximas on a symmetric window
            LM_High_window_1_CS[i] = CH[i]
        else:
            LM_High_window_1_CS[i] = 0
    ohlc['LM_High_window_1_CS'] = LM_High_window_1_CS

    # Locating Candlestick local maximas for fixed window-size=1 (fast moving)
    for i in range(window_2, n - window_2):
        if all(CH[i] > CH[j] for j in range(i - window_2, i)) and all(CH[i] >= CH[j] for j in range(i + 1, i + window_2 + 1)): # Searching for maximas on a symmetric window
            LM_High_window_2_CS[i] = CH[i]
        else:
            LM_High_window_2_CS[i] = 0
    ohlc['LM_High_window_2_CS'] = LM_High_window_2_CS

    # ---------------------------------------------------------------------------------------------------------
    # Identifying Level 1 Highs by comparing non-zero LM_High_window_1_CS values to previous and next non-zero values
    non_zero_indices = [i for i, v in enumerate(LM_High_window_1_CS) if v != 0]
    for idx, i in enumerate(non_zero_indices):
        current_val = LM_High_window_1_CS[i]
        is_peak = True
        # Check previous non-zero value
        if idx > 0:
            prev_val = LM_High_window_1_CS[non_zero_indices[idx - 1]]
            if current_val <= prev_val:
                is_peak = False
        # Check next non-zero value
        if idx < len(non_zero_indices) - 1:
            next_val = LM_High_window_1_CS[non_zero_indices[idx + 1]]
            if current_val <= next_val:
                is_peak = False
        Level_1_High_window_1_CS[i] = current_val if is_peak else 0
    ohlc['Level_1_High_window_1_CS'] = Level_1_High_window_1_CS

# ---------------------------------------------------------------------------------------------------------
    # Locating Candlestick local minimas in the Data based on Candlestick for window-size=4 (slow moving)
    for i in range(window_1, n - window_1):
        if all(CL[i] < CL[j] for j in range(i - window_1, i)) and all(CL[i] <= CL[j] for j in range(i + 1, i + window_1 + 1)): # Searching for minimas on a symmetric window
            LM_Low_window_1_CS[i] = CL[i]
        else:
            LM_Low_window_1_CS[i] = 0
    ohlc['LM_Low_window_1_CS'] = LM_Low_window_1_CS

    # Locating Candlestick local minimas in the Data based on Candlestick for window-size=1 (fast moving)
    for i in range(window_2, n - window_2):
        if all(CL[i] < CL[j] for j in range(i - window_2, i)) and all(CL[i] <= CL[j] for j in range(i + 1, i + window_2 + 1)): # Searching for minimas on a symmetric window
            LM_Low_window_2_CS[i] = CL[i]
        else:
            LM_Low_window_2_CS[i] = 0
    ohlc['LM_Low_window_2_CS'] = LM_Low_window_2_CS

#----------------------------------------------------------------------------
    # Identifying Level 1 Lows by comparing non-zero LM_High_window_1_CS values to previous and next non-zero values
    non_zero_indices = [i for i, v in enumerate(LM_Low_window_1_CS) if v != 0]
    for idx, i in enumerate(non_zero_indices):
        current_val = LM_Low_window_1_CS[i]
        is_trough = True
        if idx > 0:
            prev_val = LM_Low_window_1_CS[non_zero_indices[idx - 1]]
            if current_val >= prev_val:
                is_trough = False
        if idx < len(non_zero_indices) - 1:
            next_val = LM_Low_window_1_CS[non_zero_indices[idx + 1]]
            if current_val >= next_val:
                is_trough = False
        Level_1_Low_window_1_CS[i] = current_val if is_trough else 0
    ohlc['Level_1_Low_window_1_CS'] = Level_1_Low_window_1_CS

#----------------------------------------------------------------------------------------------------------

    # Locating Candlestick local maximas in the Data based on MACD for window-size=4 (slow moving)
    for i in range(window_1, n - window_1):
        if all(macd[i] > macd[j] for j in range(i - window_1, i)) and all(macd[i] >= macd[j] for j in range(i + 1,i + window_1 + 1)):  # Searching for maximas on a symmetric window
            LM_High_window_1_MACD[i] = CH[i]
        else:
            LM_High_window_1_MACD[i] = 0
    ohlc['LM_High_window_1_MACD'] = LM_High_window_1_MACD

    # Locating Candlestick local maximas in the Data based on MACD for window-size=1 (fast moving)
    for i in range(window_2, n - window_2):
        if all(macd[i] > macd[j] for j in range(i - window_2, i)) and all(macd[i] >= macd[j] for j in range(i + 1,i + window_2 + 1)):  # Searching for maximas on a symmetric window
            LM_High_window_2_MACD[i] = CH[i]
        else:
            LM_High_window_2_MACD[i] = 0
    ohlc['LM_High_window_2_MACD'] = LM_High_window_2_MACD

#----------------------------------------------------------------------------------------------------------

    # Locating Candlestick local minimas in the Data based on MACD for window-size=4 (slow moving)
    for i in range(window_1, n - window_1):
        if all(macd[i] < macd[j] for j in range(i - window_1, i)) and all(macd[i] <= macd[j] for j in range(i + 1,i + window_1 + 1)):  # Searching for minimas on a symmetric window
            LM_Low_window_1_MACD[i] = CL[i]
        else:
            LM_Low_window_1_MACD[i] = 0
    ohlc['LM_Low_window_1_MACD'] = LM_Low_window_1_MACD

    # Locating Candlestick local minimas in the Data based on MACD for window-size=1 (fast moving)
    for i in range(window_2, n - window_2):
        if all(macd[i] < macd[j] for j in range(i - window_2, i)) and all(macd[i] <= macd[j] for j in range(i + 1,i + window_2 + 1)):  # Searching for minimas on a symmetric window
            LM_Low_window_2_MACD[i] = CL[i]
        else:
            LM_Low_window_2_MACD[i] = 0
    ohlc['LM_Low_window_2_MACD'] = LM_Low_window_2_MACD

    # ---------------------------------------------------------------------------------------------------------
    # Level 2 Highs
    Level_2_High_window_1_CS = [0.0] * n
    non_zero_high_indices = [i for i, v in enumerate(Level_1_High_window_1_CS) if v != 0]
    for idx, i in enumerate(non_zero_high_indices):
        current_val = Level_1_High_window_1_CS[i]
        is_peak = True
        if idx > 0 and current_val <= Level_1_High_window_1_CS[non_zero_high_indices[idx - 1]]:
            is_peak = False
        if idx < len(non_zero_high_indices) - 1 and current_val <= Level_1_High_window_1_CS[non_zero_high_indices[idx + 1]]:
            is_peak = False
        Level_2_High_window_1_CS[i] = current_val if is_peak else 0
    ohlc['Level_2_High_window_1_CS'] = Level_2_High_window_1_CS

    # Level 2 Lows
    Level_2_Low_window_1_CS = [0.0] * n
    non_zero_low_indices = [i for i, v in enumerate(Level_1_Low_window_1_CS) if v != 0]
    for idx, i in enumerate(non_zero_low_indices):
        current_val = Level_1_Low_window_1_CS[i]
        is_trough = True
        if idx > 0 and current_val >= Level_1_Low_window_1_CS[non_zero_low_indices[idx - 1]]:
            is_trough = False
        if idx < len(non_zero_low_indices) - 1 and current_val >= Level_1_Low_window_1_CS[non_zero_low_indices[idx + 1]]:
            is_trough = False
        Level_2_Low_window_1_CS[i] = current_val if is_trough else 0
    ohlc['Level_2_Low_window_1_CS'] = Level_2_Low_window_1_CS


#----------------------------------------------------------------------------------------------------------