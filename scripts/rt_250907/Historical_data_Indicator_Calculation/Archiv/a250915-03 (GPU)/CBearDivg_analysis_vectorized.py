import pandas as pd
import numpy as np
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def CBearDivg_analysis(ohlc: pd.DataFrame, Candle_Tol: float, MACD_tol: float) -> pd.DataFrame:
    # Convert necessary columns to numeric, coercing errors
    columns_to_convert = [
        'LM_High_window_2_CS', 'LM_High_window_1_CS',
        'LM_High_window_2_MACD', 'LM_High_window_1_MACD',
        'RSI', 'macd_histogram'
    ]
    for col in columns_to_convert:
        ohlc[col] = pd.to_numeric(ohlc[col], errors='coerce')

    # Parameters
    window_1 = 5
    window_2 = 1
    RSI_tol = 0.5     # Absolute Difference in RSI Values

    # Extract data to NumPy arrays
    CH = ohlc["high"].to_numpy()
    n = len(CH)
    dates = pd.to_datetime(ohlc['date'], utc=True).to_numpy()
    LM_High_window_2_CS = ohlc['LM_High_window_2_CS'].to_numpy()
    LM_High_window_1_CS = ohlc['LM_High_window_1_CS'].to_numpy()
    LM_High_window_2_MACD = ohlc['LM_High_window_2_MACD'].to_numpy()
    LM_High_window_1_MACD = ohlc['LM_High_window_1_MACD'].to_numpy()
    RSI_arr = ohlc['RSI'].to_numpy()
    macd_histogram_arr = ohlc['macd_histogram'].to_numpy()

    # Initialize output arrays for CBearD_1 (Classic Bearish Divergence - Candlestick Highs)
    CBearD_1 = np.zeros(n)
    CBearD_Lower_High_1 = np.zeros(n)
    CBearD_Higher_High_1 = np.zeros(n)
    CBearD_Lower_High_RSI_1 = np.zeros(n)
    CBearD_Higher_High_RSI_1 = np.zeros(n)
    CBearD_Lower_High_MACD_1 = np.zeros(n)
    CBearD_Higher_High_MACD_1 = np.zeros(n)
    CBearD_Lower_High_date_1 = np.empty(n, dtype=object)
    CBearD_Higher_High_date_1 = np.empty(n, dtype=object)
    CBearD_Date_Gap_1 = np.zeros(n)

    # Initialize output arrays for CBearD_2 (Classic Bearish Divergence - MACD)
    CBearD_2 = np.zeros(n)
    CBearD_Lower_High_2 = np.zeros(n)
    CBearD_Higher_High_2 = np.zeros(n)
    CBearD_Lower_High_RSI_2 = np.zeros(n)
    CBearD_Higher_High_RSI_2 = np.zeros(n)
    CBearD_Lower_High_MACD_2 = np.zeros(n)
    CBearD_Higher_High_MACD_2 = np.zeros(n)
    CBearD_Lower_High_date_2 = np.empty(n, dtype=object)
    CBearD_Higher_High_date_2 = np.empty(n, dtype=object)
    CBearD_Date_Gap_2 = np.zeros(n)

    # Initialize output arrays for CBearD_3 (Classic Bearish Divergence - Positive MACD Maximas)
    CBearD_pos_MACD = np.zeros(n)
    CBearD_Lower_High_pos_MACD = np.zeros(n)
    CBearD_Higher_High_pos_MACD = np.zeros(n)
    CBearD_Lower_High_RSI_pos_MACD = np.zeros(n)
    CBearD_Higher_High_RSI_pos_MACD = np.zeros(n)
    CBearD_Lower_High_MACD_pos_MACD = np.zeros(n)
    CBearD_Higher_High_MACD_pos_MACD = np.zeros(n)
    CBearD_Lower_High_date_pos_MACD = np.empty(n, dtype=object)
    CBearD_Higher_High_date_pos_MACD = np.empty(n, dtype=object)
    CBearD_Date_Gap_pos_MACD = np.zeros(n)

    # Helper function to find the last non-zero indices before each index
    def get_last_nonzero_indices(indices, lm_high_window, extra_condition=None):
        last_nonzero = np.full(len(indices), -1, dtype=int)
        for i, idx in enumerate(indices):
            if extra_condition is None:
                nonzero_before = np.where(lm_high_window[:idx] != 0)[0]
            else:
                nonzero_before = np.where((lm_high_window[:idx] != 0) & extra_condition[:idx])[0]
            if len(nonzero_before) > 0:
                last_nonzero[i] = nonzero_before[-1]
        return last_nonzero

    # Process CBearD_1: Classic Bearish Divergence based on Candlestick Highs
    valid_indices_1 = np.where(LM_High_window_2_CS > 0)[0]
    valid_indices_1 = valid_indices_1[(valid_indices_1 >= window_1) & (valid_indices_1 < n - window_2)]
    last_nonzero_indices_1 = get_last_nonzero_indices(valid_indices_1, LM_High_window_1_CS)
    valid_mask_1 = last_nonzero_indices_1 != -1
    valid_indices_1 = valid_indices_1[valid_mask_1]
    last_nonzero_indices_1 = last_nonzero_indices_1[valid_mask_1]

    if len(valid_indices_1) > 0:
        LM_High_window_2_high = LM_High_window_2_CS[valid_indices_1]
        LM_High_window_1_high = LM_High_window_1_CS[last_nonzero_indices_1]
        LM_High_window_2_rsi = RSI_arr[valid_indices_1]
        LM_High_window_1_rsi = RSI_arr[last_nonzero_indices_1]
        LM_High_window_2_macd = macd_histogram_arr[valid_indices_1]
        LM_High_window_1_macd = macd_histogram_arr[last_nonzero_indices_1]
        LM_High_window_2_date = dates[valid_indices_1]
        LM_High_window_1_date = dates[last_nonzero_indices_1]

        price_diff = LM_High_window_2_high - LM_High_window_1_high
        price_diff_percent = np.abs(100 * price_diff / LM_High_window_1_high)
        rsi_diff = LM_High_window_2_rsi - LM_High_window_1_rsi
        macd_diff = LM_High_window_2_macd - LM_High_window_1_macd
        macd_sign_check = LM_High_window_2_macd * LM_High_window_1_macd

        price_condition = (price_diff > 0) | (price_diff_percent < Candle_Tol)
        rsi_condition = (
            (rsi_diff < 0) |
            (np.abs(rsi_diff) < RSI_tol) |
            ((macd_sign_check > 0) & (np.abs(rsi_diff) < 2 * RSI_tol)) |
            ((LM_High_window_2_rsi > 70) & (np.abs(rsi_diff) < 2 * RSI_tol))
        )
        macd_condition = macd_diff < 0

        bearish_divergence_1 = price_condition & rsi_condition & macd_condition

        CBearD_1[valid_indices_1[bearish_divergence_1]] = 1
        CBearD_Lower_High_1[valid_indices_1[bearish_divergence_1]] = LM_High_window_1_high[bearish_divergence_1]
        CBearD_Higher_High_1[valid_indices_1[bearish_divergence_1]] = LM_High_window_2_high[bearish_divergence_1]
        CBearD_Lower_High_RSI_1[valid_indices_1[bearish_divergence_1]] = LM_High_window_1_rsi[bearish_divergence_1]
        CBearD_Higher_High_RSI_1[valid_indices_1[bearish_divergence_1]] = LM_High_window_2_rsi[bearish_divergence_1]
        CBearD_Lower_High_MACD_1[valid_indices_1[bearish_divergence_1]] = LM_High_window_1_macd[bearish_divergence_1]
        CBearD_Higher_High_MACD_1[valid_indices_1[bearish_divergence_1]] = LM_High_window_2_macd[bearish_divergence_1]
        CBearD_Lower_High_date_1[valid_indices_1[bearish_divergence_1]] = LM_High_window_1_date[bearish_divergence_1]
        CBearD_Higher_High_date_1[valid_indices_1[bearish_divergence_1]] = LM_High_window_2_date[bearish_divergence_1]
        CBearD_Date_Gap_1[valid_indices_1[bearish_divergence_1]] = valid_indices_1[bearish_divergence_1] - last_nonzero_indices_1[bearish_divergence_1]

    # Process CBearD_2: Classic Bearish Divergence based on MACD
    valid_indices_2 = np.where(LM_High_window_2_MACD > 0)[0]
    valid_indices_2 = valid_indices_2[(valid_indices_2 >= window_1) & (valid_indices_2 < n - window_2)]
    last_nonzero_indices_2 = get_last_nonzero_indices(valid_indices_2, LM_High_window_1_MACD)
    valid_mask_2 = last_nonzero_indices_2 != -1
    valid_indices_2 = valid_indices_2[valid_mask_2]
    last_nonzero_indices_2 = last_nonzero_indices_2[valid_mask_2]

    if len(valid_indices_2) > 0:
        LM_High_window_2_high = LM_High_window_2_MACD[valid_indices_2]
        LM_High_window_1_high = LM_High_window_1_MACD[last_nonzero_indices_2]
        LM_High_window_2_rsi = RSI_arr[valid_indices_2]
        LM_High_window_1_rsi = RSI_arr[last_nonzero_indices_2]
        LM_High_window_2_macd = macd_histogram_arr[valid_indices_2]
        LM_High_window_1_macd = macd_histogram_arr[last_nonzero_indices_2]
        LM_High_window_2_date = dates[valid_indices_2]
        LM_High_window_1_date = dates[last_nonzero_indices_2]

        price_diff = LM_High_window_2_high - LM_High_window_1_high
        price_diff_percent = np.abs(100 * price_diff / LM_High_window_1_high)
        rsi_diff = LM_High_window_2_rsi - LM_High_window_1_rsi
        macd_diff = LM_High_window_2_macd - LM_High_window_1_macd

        price_condition = (price_diff > 0) | (price_diff_percent < Candle_Tol)
        rsi_condition = (rsi_diff < 0) | (np.abs(rsi_diff) < RSI_tol)
        macd_condition = (LM_High_window_2_macd > 0) & (LM_High_window_1_macd > 0) & (macd_diff < 0)

        bearish_divergence_2 = price_condition & rsi_condition & macd_condition

        CBearD_2[valid_indices_2[bearish_divergence_2]] = 1
        CBearD_Lower_High_2[valid_indices_2[bearish_divergence_2]] = LM_High_window_1_high[bearish_divergence_2]
        CBearD_Higher_High_2[valid_indices_2[bearish_divergence_2]] = LM_High_window_2_high[bearish_divergence_2]
        CBearD_Lower_High_RSI_2[valid_indices_2[bearish_divergence_2]] = LM_High_window_1_rsi[bearish_divergence_2]
        CBearD_Higher_High_RSI_2[valid_indices_2[bearish_divergence_2]] = LM_High_window_2_rsi[bearish_divergence_2]
        CBearD_Lower_High_MACD_2[valid_indices_2[bearish_divergence_2]] = LM_High_window_1_macd[bearish_divergence_2]
        CBearD_Higher_High_MACD_2[valid_indices_2[bearish_divergence_2]] = LM_High_window_2_macd[bearish_divergence_2]
        CBearD_Lower_High_date_2[valid_indices_2[bearish_divergence_2]] = LM_High_window_1_date[bearish_divergence_2]
        CBearD_Higher_High_date_2[valid_indices_2[bearish_divergence_2]] = LM_High_window_2_date[bearish_divergence_2]
        CBearD_Date_Gap_2[valid_indices_2[bearish_divergence_2]] = valid_indices_2[bearish_divergence_2] - last_nonzero_indices_2[bearish_divergence_2]

    # Process CBearD_pos_MACD: Classic Bearish Divergence based on Positive MACD Maximas
    valid_indices_3 = np.where((LM_High_window_2_MACD > 0) & (macd_histogram_arr > 0))[0]
    valid_indices_3 = valid_indices_3[(valid_indices_3 >= window_1) & (valid_indices_3 < n - window_2)]
    last_nonzero_indices_3 = get_last_nonzero_indices(valid_indices_3, LM_High_window_1_MACD, macd_histogram_arr > 0)
    valid_mask_3 = last_nonzero_indices_3 != -1
    valid_indices_3 = valid_indices_3[valid_mask_3]
    last_nonzero_indices_3 = last_nonzero_indices_3[valid_mask_3]

    if len(valid_indices_3) > 0:
        LM_High_window_2_high = LM_High_window_2_MACD[valid_indices_3]
        LM_High_window_1_high = LM_High_window_1_MACD[last_nonzero_indices_3]
        LM_High_window_2_rsi = RSI_arr[valid_indices_3]
        LM_High_window_1_rsi = RSI_arr[last_nonzero_indices_3]
        LM_High_window_2_macd = macd_histogram_arr[valid_indices_3]
        LM_High_window_1_macd = macd_histogram_arr[last_nonzero_indices_3]
        LM_High_window_2_date = dates[valid_indices_3]
        LM_High_window_1_date = dates[last_nonzero_indices_3]

        price_diff = LM_High_window_2_high - LM_High_window_1_high
        price_diff_percent = np.abs(100 * price_diff / LM_High_window_1_high)
        rsi_diff = LM_High_window_2_rsi - LM_High_window_1_rsi
        macd_diff = LM_High_window_2_macd - LM_High_window_1_macd
        macd_diff_percent = np.abs(100 * macd_diff / (0.000001 + LM_High_window_1_macd))

        price_condition = (price_diff > 0) | (price_diff_percent < Candle_Tol)
        rsi_condition = (rsi_diff < 0) | (np.abs(rsi_diff) < 2 * RSI_tol)
        macd_condition = (macd_diff < 0) | (macd_diff_percent < MACD_tol)

        bearish_divergence_3 = price_condition & rsi_condition & macd_condition

        CBearD_pos_MACD[valid_indices_3[bearish_divergence_3]] = 1
        CBearD_Lower_High_pos_MACD[valid_indices_3[bearish_divergence_3]] = LM_High_window_1_high[bearish_divergence_3]
        CBearD_Higher_High_pos_MACD[valid_indices_3[bearish_divergence_3]] = LM_High_window_2_high[bearish_divergence_3]
        CBearD_Lower_High_RSI_pos_MACD[valid_indices_3[bearish_divergence_3]] = LM_High_window_1_rsi[bearish_divergence_3]
        CBearD_Higher_High_RSI_pos_MACD[valid_indices_3[bearish_divergence_3]] = LM_High_window_2_rsi[bearish_divergence_3]
        CBearD_Lower_High_MACD_pos_MACD[valid_indices_3[bearish_divergence_3]] = LM_High_window_1_macd[bearish_divergence_3]
        CBearD_Higher_High_MACD_pos_MACD[valid_indices_3[bearish_divergence_3]] = LM_High_window_2_macd[bearish_divergence_3]
        CBearD_Lower_High_date_pos_MACD[valid_indices_3[bearish_divergence_3]] = LM_High_window_1_date[bearish_divergence_3]
        CBearD_Higher_High_date_pos_MACD[valid_indices_3[bearish_divergence_3]] = LM_High_window_2_date[bearish_divergence_3]
        CBearD_Date_Gap_pos_MACD[valid_indices_3[bearish_divergence_3]] = valid_indices_3[bearish_divergence_3] - last_nonzero_indices_3[bearish_divergence_3]

    # Combine CBearD_1 and CBearD_2 into CBearD_gen
    CBearD_gen = np.zeros(n)
    CBearD_Lower_High_gen = np.zeros(n)
    CBearD_Higher_High_gen = np.zeros(n)
    CBearD_Lower_High_RSI_gen = np.zeros(n)
    CBearD_Higher_High_RSI_gen = np.zeros(n)
    CBearD_Lower_High_MACD_gen = np.zeros(n)
    CBearD_Higher_High_MACD_gen = np.zeros(n)
    CBearD_Lower_High_date_gen = np.empty(n, dtype=object)
    CBearD_Higher_High_date_gen = np.empty(n, dtype=object)
    CBearD_Date_Gap_gen = np.zeros(n)

    indices = np.arange(window_1, n - window_2)
    condition_1 = CBearD_1[indices] == 1
    condition_2 = CBearD_2[indices] == 1
    combined_condition_gen = condition_1 | condition_2
    valid_indices_gen = indices[combined_condition_gen]

    for idx in valid_indices_gen:
        if CBearD_1[idx] == 1 or (CBearD_1[idx] == 1 and CBearD_2[idx] == 1):
            CBearD_gen[idx] = 1
            CBearD_Lower_High_gen[idx] = CBearD_Lower_High_1[idx]
            CBearD_Higher_High_gen[idx] = CBearD_Higher_High_1[idx]
            CBearD_Lower_High_RSI_gen[idx] = CBearD_Lower_High_RSI_1[idx]
            CBearD_Higher_High_RSI_gen[idx] = CBearD_Higher_High_RSI_1[idx]
            CBearD_Lower_High_MACD_gen[idx] = CBearD_Lower_High_MACD_1[idx]
            CBearD_Higher_High_MACD_gen[idx] = CBearD_Higher_High_MACD_1[idx]
            CBearD_Lower_High_date_gen[idx] = CBearD_Lower_High_date_1[idx]
            CBearD_Higher_High_date_gen[idx] = CBearD_Higher_High_date_1[idx]
            CBearD_Date_Gap_gen[idx] = CBearD_Date_Gap_1[idx]
        elif CBearD_2[idx] == 1:
            CBearD_gen[idx] = 1
            CBearD_Lower_High_gen[idx] = CBearD_Lower_High_2[idx]
            CBearD_Higher_High_gen[idx] = CBearD_Higher_High_2[idx]
            CBearD_Lower_High_RSI_gen[idx] = CBearD_Lower_High_RSI_2[idx]
            CBearD_Higher_High_RSI_gen[idx] = CBearD_Higher_High_RSI_2[idx]
            CBearD_Lower_High_MACD_gen[idx] = CBearD_Lower_High_MACD_2[idx]
            CBearD_Higher_High_MACD_gen[idx] = CBearD_Higher_High_MACD_2[idx]
            CBearD_Lower_High_date_gen[idx] = CBearD_Lower_High_date_2[idx]
            CBearD_Higher_High_date_gen[idx] = CBearD_Higher_High_date_2[idx]
            CBearD_Date_Gap_gen[idx] = CBearD_Date_Gap_2[idx]

    # Assign results to DataFrame
    ohlc['CBearD_gen'] = CBearD_gen
    ohlc['CBearD_Lower_High_gen'] = CBearD_Lower_High_gen
    ohlc['CBearD_Higher_High_gen'] = CBearD_Higher_High_gen
    ohlc['CBearD_Lower_High_RSI_gen'] = CBearD_Lower_High_RSI_gen
    ohlc['CBearD_Higher_High_RSI_gen'] = CBearD_Higher_High_RSI_gen
    ohlc['CBearD_Lower_High_MACD_gen'] = CBearD_Lower_High_MACD_gen
    ohlc['CBearD_Higher_High_MACD_gen'] = CBearD_Higher_High_MACD_gen
    ohlc['CBearD_Lower_High_date_gen'] = CBearD_Lower_High_date_gen
    ohlc['CBearD_Higher_High_date_gen'] = CBearD_Higher_High_date_gen
    ohlc['CBearD_Date_Gap_gen'] = CBearD_Date_Gap_gen

    ohlc['CBearD_pos_MACD'] = CBearD_pos_MACD
    ohlc['CBearD_Lower_High_pos_MACD'] = CBearD_Lower_High_pos_MACD
    ohlc['CBearD_Higher_High_pos_MACD'] = CBearD_Higher_High_pos_MACD
    ohlc['CBearD_Lower_High_RSI_pos_MACD'] = CBearD_Lower_High_RSI_pos_MACD
    ohlc['CBearD_Higher_High_RSI_pos_MACD'] = CBearD_Higher_High_RSI_pos_MACD
    ohlc['CBearD_Lower_High_MACD_pos_MACD'] = CBearD_Lower_High_MACD_pos_MACD
    ohlc['CBearD_Higher_High_MACD_pos_MACD'] = CBearD_Higher_High_MACD_pos_MACD
    ohlc['CBearD_Lower_High_date_pos_MACD'] = CBearD_Lower_High_date_pos_MACD
    ohlc['CBearD_Higher_High_date_pos_MACD'] = CBearD_Higher_High_date_pos_MACD
    ohlc['CBearD_Date_Gap_pos_MACD'] = CBearD_Date_Gap_pos_MACD

    return ohlc[['CBearD_gen', 'CBearD_pos_MACD']]