import pandas as pd
import numpy as np
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def HBearDivg_analysis(ohlc: pd.DataFrame, window: int, Candle_Tol: float, MACD_tol: float) -> pd.DataFrame:
    # Convert necessary columns to numeric, coercing errors
    columns_to_convert = [
        'LM_High_window_2_CS', 'LM_High_window_1_CS',
        'LM_High_window_2_MACD', 'LM_High_window_1_MACD',
        'RSI', 'macd_histogram'
    ]
    for col in columns_to_convert:
        ohlc[col] = pd.to_numeric(ohlc[col], errors='coerce')

    # Parameters
    window_1 = window
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
    macd_hist = ohlc['macd_histogram'].to_numpy()

    # Initialize output arrays for HBearD_1 (Hidden Bearish Divergence - Candlestick Highs)
    HBearD_1 = np.zeros(n)
    HBearD_Lower_High_1 = np.zeros(n)
    HBearD_Higher_High_1 = np.zeros(n)
    HBearD_Lower_High_RSI_1 = np.zeros(n)
    HBearD_Higher_High_RSI_1 = np.zeros(n)
    HBearD_Lower_High_MACD_1 = np.zeros(n)
    HBearD_Higher_High_MACD_1 = np.zeros(n)
    HBearD_Lower_High_date_1 = np.empty(n, dtype=object)
    HBearD_Higher_High_date_1 = np.empty(n, dtype=object)
    HBearD_Date_Gap_1 = np.zeros(n)

    # Initialize output arrays for HBearD_2 (Hidden Bearish Divergence - MACD)
    HBearD_2 = np.zeros(n)
    HBearD_Lower_High_2 = np.zeros(n)
    HBearD_Higher_High_2 = np.zeros(n)
    HBearD_Lower_High_RSI_2 = np.zeros(n)
    HBearD_Higher_High_RSI_2 = np.zeros(n)
    HBearD_Lower_High_MACD_2 = np.zeros(n)
    HBearD_Higher_High_MACD_2 = np.zeros(n)
    HBearD_Lower_High_date_2 = np.empty(n, dtype=object)
    HBearD_Higher_High_date_2 = np.empty(n, dtype=object)
    HBearD_Date_Gap_2 = np.zeros(n)

    # Initialize output arrays for HBearD_pos_MACD (Hidden Bearish Divergence - Positive MACD Maximas)
    HBearD_pos_MACD = np.zeros(n)
    HBearD_Lower_High_pos_MACD = np.zeros(n)
    HBearD_Higher_High_pos_MACD = np.zeros(n)
    HBearD_Lower_High_RSI_pos_MACD = np.zeros(n)
    HBearD_Higher_High_RSI_pos_MACD = np.zeros(n)
    HBearD_Lower_High_MACD_pos_MACD = np.zeros(n)
    HBearD_Higher_High_MACD_pos_MACD = np.zeros(n)
    HBearD_Lower_High_date_pos_MACD = np.empty(n, dtype=object)
    HBearD_Higher_High_date_pos_MACD = np.empty(n, dtype=object)
    HBearD_Date_Gap_pos_MACD = np.zeros(n)

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

    # Process HBearD_1: Hidden Bearish Divergence based on Candlestick Highs
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
        LM_High_window_2_macd = macd_hist[valid_indices_1]
        LM_High_window_1_macd = macd_hist[last_nonzero_indices_1]
        LM_High_window_2_date = dates[valid_indices_1]
        LM_High_window_1_date = dates[last_nonzero_indices_1]

        price_diff = LM_High_window_2_high - LM_High_window_1_high
        price_diff_percent = np.abs(100 * price_diff / LM_High_window_1_high)
        rsi_diff = LM_High_window_2_rsi - LM_High_window_1_rsi
        macd_diff = LM_High_window_2_macd - LM_High_window_1_macd
        macd_sign_check = LM_High_window_2_macd * LM_High_window_1_macd

        price_condition = (price_diff < 0) | (price_diff_percent < Candle_Tol)
        rsi_condition = (
            (rsi_diff > 0) |
            (np.abs(rsi_diff) < RSI_tol) |
            ((macd_sign_check > 0) & (np.abs(rsi_diff) < 2 * RSI_tol))
        )
        macd_condition = macd_diff > 0

        bearish_divergence_1 = price_condition & rsi_condition & macd_condition

        HBearD_1[valid_indices_1[bearish_divergence_1]] = 1
        HBearD_Higher_High_1[valid_indices_1[bearish_divergence_1]] = LM_High_window_1_high[bearish_divergence_1]
        HBearD_Lower_High_1[valid_indices_1[bearish_divergence_1]] = LM_High_window_2_high[bearish_divergence_1]
        HBearD_Higher_High_RSI_1[valid_indices_1[bearish_divergence_1]] = LM_High_window_1_rsi[bearish_divergence_1]
        HBearD_Lower_High_RSI_1[valid_indices_1[bearish_divergence_1]] = LM_High_window_2_rsi[bearish_divergence_1]
        HBearD_Higher_High_MACD_1[valid_indices_1[bearish_divergence_1]] = LM_High_window_1_macd[bearish_divergence_1]
        HBearD_Lower_High_MACD_1[valid_indices_1[bearish_divergence_1]] = LM_High_window_2_macd[bearish_divergence_1]
        HBearD_Higher_High_date_1[valid_indices_1[bearish_divergence_1]] = LM_High_window_1_date[bearish_divergence_1]
        HBearD_Lower_High_date_1[valid_indices_1[bearish_divergence_1]] = LM_High_window_2_date[bearish_divergence_1]
        HBearD_Date_Gap_1[valid_indices_1[bearish_divergence_1]] = valid_indices_1[bearish_divergence_1] - last_nonzero_indices_1[bearish_divergence_1]

    # Process HBearD_2: Hidden Bearish Divergence based on MACD
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
        LM_High_window_2_macd = macd_hist[valid_indices_2]
        LM_High_window_1_macd = macd_hist[last_nonzero_indices_2]
        LM_High_window_2_date = dates[valid_indices_2]
        LM_High_window_1_date = dates[last_nonzero_indices_2]

        price_diff = LM_High_window_2_high - LM_High_window_1_high
        price_diff_percent = np.abs(100 * price_diff / LM_High_window_1_high)
        rsi_diff = LM_High_window_2_rsi - LM_High_window_1_rsi
        macd_diff = LM_High_window_2_macd - LM_High_window_1_macd

        price_condition = (price_diff < 0) | (price_diff_percent < Candle_Tol)
        rsi_condition = (rsi_diff > 0) | (np.abs(rsi_diff) < RSI_tol)
        macd_condition = (
            ((LM_High_window_2_macd < 0) & (LM_High_window_1_macd < 0) & (macd_diff > 0)) |
            ((LM_High_window_2_macd > 0) & (LM_High_window_1_macd > 0) & (macd_diff > 0))
        )

        bearish_divergence_2 = price_condition & rsi_condition & macd_condition

        HBearD_2[valid_indices_2[bearish_divergence_2]] = 1
        HBearD_Higher_High_2[valid_indices_2[bearish_divergence_2]] = LM_High_window_1_high[bearish_divergence_2]
        HBearD_Lower_High_2[valid_indices_2[bearish_divergence_2]] = LM_High_window_2_high[bearish_divergence_2]
        HBearD_Higher_High_RSI_2[valid_indices_2[bearish_divergence_2]] = LM_High_window_1_rsi[bearish_divergence_2]
        HBearD_Lower_High_RSI_2[valid_indices_2[bearish_divergence_2]] = LM_High_window_2_rsi[bearish_divergence_2]
        HBearD_Higher_High_MACD_2[valid_indices_2[bearish_divergence_2]] = LM_High_window_1_macd[bearish_divergence_2]
        HBearD_Lower_High_MACD_2[valid_indices_2[bearish_divergence_2]] = LM_High_window_2_macd[bearish_divergence_2]
        HBearD_Higher_High_date_2[valid_indices_2[bearish_divergence_2]] = LM_High_window_1_date[bearish_divergence_2]
        HBearD_Lower_High_date_2[valid_indices_2[bearish_divergence_2]] = LM_High_window_2_date[bearish_divergence_2]
        HBearD_Date_Gap_2[valid_indices_2[bearish_divergence_2]] = valid_indices_2[bearish_divergence_2] - last_nonzero_indices_2[bearish_divergence_2]

    # Process HBearD_pos_MACD: Hidden Bearish Divergence based on Positive MACD Maximas
    valid_indices_3 = np.where((LM_High_window_2_MACD > 0) & (macd_hist > 0))[0]
    valid_indices_3 = valid_indices_3[(valid_indices_3 >= window_1) & (valid_indices_3 < n - window_2)]
    last_nonzero_indices_3 = get_last_nonzero_indices(valid_indices_3, LM_High_window_1_MACD, macd_hist > 0)
    valid_mask_3 = last_nonzero_indices_3 != -1
    valid_indices_3 = valid_indices_3[valid_mask_3]
    last_nonzero_indices_3 = last_nonzero_indices_3[valid_mask_3]

    if len(valid_indices_3) > 0:
        LM_High_window_2_high = LM_High_window_2_MACD[valid_indices_3]
        LM_High_window_1_high = LM_High_window_1_MACD[last_nonzero_indices_3]
        LM_High_window_2_rsi = RSI_arr[valid_indices_3]
        LM_High_window_1_rsi = RSI_arr[last_nonzero_indices_3]
        LM_High_window_2_macd = macd_hist[valid_indices_3]
        LM_High_window_1_macd = macd_hist[last_nonzero_indices_3]
        LM_High_window_2_date = dates[valid_indices_3]
        LM_High_window_1_date = dates[last_nonzero_indices_3]

        price_diff = LM_High_window_2_high - LM_High_window_1_high
        price_diff_percent = np.abs(100 * price_diff / LM_High_window_1_high)
        rsi_diff = LM_High_window_2_rsi - LM_High_window_1_rsi
        macd_diff = LM_High_window_2_macd - LM_High_window_1_macd
        macd_diff_percent = np.abs(100 * macd_diff / (0.000001 + LM_High_window_1_macd))

        price_condition = (price_diff < 0) | (price_diff_percent < Candle_Tol)
        rsi_condition = (rsi_diff > 0) | (np.abs(rsi_diff) < 2 * RSI_tol)
        macd_condition = (macd_diff > 0) | (macd_diff_percent < MACD_tol)

        bearish_divergence_3 = price_condition & rsi_condition & macd_condition

        HBearD_pos_MACD[valid_indices_3[bearish_divergence_3]] = 1
        HBearD_Higher_High_pos_MACD[valid_indices_3[bearish_divergence_3]] = LM_High_window_1_high[bearish_divergence_3]
        HBearD_Lower_High_pos_MACD[valid_indices_3[bearish_divergence_3]] = LM_High_window_2_high[bearish_divergence_3]
        HBearD_Higher_High_RSI_pos_MACD[valid_indices_3[bearish_divergence_3]] = LM_High_window_1_rsi[bearish_divergence_3]
        HBearD_Lower_High_RSI_pos_MACD[valid_indices_3[bearish_divergence_3]] = LM_High_window_2_rsi[bearish_divergence_3]
        HBearD_Higher_High_MACD_pos_MACD[valid_indices_3[bearish_divergence_3]] = LM_High_window_1_macd[bearish_divergence_3]
        HBearD_Lower_High_MACD_pos_MACD[valid_indices_3[bearish_divergence_3]] = LM_High_window_2_macd[bearish_divergence_3]
        HBearD_Higher_High_date_pos_MACD[valid_indices_3[bearish_divergence_3]] = LM_High_window_1_date[bearish_divergence_3]
        HBearD_Lower_High_date_pos_MACD[valid_indices_3[bearish_divergence_3]] = LM_High_window_2_date[bearish_divergence_3]
        HBearD_Date_Gap_pos_MACD[valid_indices_3[bearish_divergence_3]] = valid_indices_3[bearish_divergence_3] - last_nonzero_indices_3[bearish_divergence_3]

    # Combine HBearD_1 and HBearD_2 into HBearD_gen
    HBearD_gen = np.zeros(n)
    HBearD_Lower_High_gen = np.zeros(n)
    HBearD_Higher_High_gen = np.zeros(n)
    HBearD_Lower_High_RSI_gen = np.zeros(n)
    HBearD_Higher_High_RSI_gen = np.zeros(n)
    HBearD_Lower_High_MACD_gen = np.zeros(n)
    HBearD_Higher_High_MACD_gen = np.zeros(n)
    HBearD_Lower_High_date_gen = np.empty(n, dtype=object)
    HBearD_Higher_High_date_gen = np.empty(n, dtype=object)
    HBearD_Date_Gap_gen = np.zeros(n)

    indices = np.arange(window_1, n - window_2)
    condition_1 = HBearD_1[indices] == 1
    condition_2 = HBearD_2[indices] == 1
    combined_condition_gen = condition_1 | condition_2
    valid_indices_gen = indices[combined_condition_gen]

    for idx in valid_indices_gen:
        if HBearD_1[idx] == 1 or (HBearD_1[idx] == 1 and HBearD_2[idx] == 1):
            HBearD_gen[idx] = 1
            HBearD_Lower_High_gen[idx] = HBearD_Lower_High_1[idx]
            HBearD_Higher_High_gen[idx] = HBearD_Higher_High_1[idx]
            HBearD_Lower_High_RSI_gen[idx] = HBearD_Lower_High_RSI_1[idx]
            HBearD_Higher_High_RSI_gen[idx] = HBearD_Higher_High_RSI_1[idx]
            HBearD_Lower_High_MACD_gen[idx] = HBearD_Lower_High_MACD_1[idx]
            HBearD_Higher_High_MACD_gen[idx] = HBearD_Higher_High_MACD_1[idx]
            HBearD_Lower_High_date_gen[idx] = HBearD_Lower_High_date_1[idx]
            HBearD_Higher_High_date_gen[idx] = HBearD_Higher_High_date_1[idx]
            HBearD_Date_Gap_gen[idx] = HBearD_Date_Gap_1[idx]
        elif HBearD_2[idx] == 1:
            HBearD_gen[idx] = 1
            HBearD_Lower_High_gen[idx] = HBearD_Lower_High_2[idx]
            HBearD_Higher_High_gen[idx] = HBearD_Higher_High_2[idx]
            HBearD_Lower_High_RSI_gen[idx] = HBearD_Lower_High_RSI_2[idx]
            HBearD_Higher_High_RSI_gen[idx] = HBearD_Higher_High_RSI_2[idx]
            HBearD_Lower_High_MACD_gen[idx] = HBearD_Lower_High_MACD_2[idx]
            HBearD_Higher_High_MACD_gen[idx] = HBearD_Higher_High_MACD_2[idx]
            HBearD_Lower_High_date_gen[idx] = HBearD_Lower_High_date_2[idx]
            HBearD_Higher_High_date_gen[idx] = HBearD_Higher_High_date_2[idx]
            HBearD_Date_Gap_gen[idx] = HBearD_Date_Gap_2[idx]

    # Assign results to DataFrame
    ohlc['HBearD_gen'] = HBearD_gen
    ohlc['HBearD_Lower_High_gen'] = HBearD_Lower_High_gen
    ohlc['HBearD_Higher_High_gen'] = HBearD_Higher_High_gen
    ohlc['HBearD_Lower_High_RSI_gen'] = HBearD_Lower_High_RSI_gen
    ohlc['HBearD_Higher_High_RSI_gen'] = HBearD_Higher_High_RSI_gen
    ohlc['HBearD_Lower_High_MACD_gen'] = HBearD_Lower_High_MACD_gen
    ohlc['HBearD_Higher_High_MACD_gen'] = HBearD_Higher_High_MACD_gen
    ohlc['HBearD_Lower_High_date_gen'] = HBearD_Lower_High_date_gen
    ohlc['HBearD_Higher_High_date_gen'] = HBearD_Higher_High_date_gen
    ohlc['HBearD_Date_Gap_gen'] = HBearD_Date_Gap_gen

    ohlc['HBearD_pos_MACD'] = HBearD_pos_MACD
    ohlc['HBearD_Lower_High_pos_MACD'] = HBearD_Lower_High_pos_MACD
    ohlc['HBearD_Higher_High_pos_MACD'] = HBearD_Higher_High_pos_MACD
    ohlc['HBearD_Lower_High_RSI_pos_MACD'] = HBearD_Lower_High_RSI_pos_MACD
    ohlc['HBearD_Higher_High_RSI_pos_MACD'] = HBearD_Higher_High_RSI_pos_MACD
    ohlc['HBearD_Lower_High_MACD_pos_MACD'] = HBearD_Lower_High_MACD_pos_MACD
    ohlc['HBearD_Higher_High_MACD_pos_MACD'] = HBearD_Higher_High_MACD_pos_MACD
    ohlc['HBearD_Lower_High_date_pos_MACD'] = HBearD_Lower_High_date_pos_MACD
    ohlc['HBearD_Higher_High_date_pos_MACD'] = HBearD_Higher_High_date_pos_MACD
    ohlc['HBearD_Date_Gap_pos_MACD'] = HBearD_Date_Gap_pos_MACD

    return ohlc[['HBearD_gen', 'HBearD_pos_MACD']]