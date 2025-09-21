import pandas as pd
import numpy as np
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def CBullDivg_x2_analysis(ohlc: pd.DataFrame, Candle_Tol: float, MACD_tol: float) -> pd.DataFrame:
    # Convert necessary columns to numeric, coercing errors
    columns_to_convert = [
        'LM_Low_window_2_CS', 'LM_Low_window_1_CS',
        'LM_Low_window_2_MACD', 'LM_Low_window_1_MACD',
        'RSI', 'macd_histogram'
    ]
    for col in columns_to_convert:
        ohlc[col] = pd.to_numeric(ohlc[col], errors='coerce')

    # Parameters
    window_1 = 5
    window_2 = 1
    RSI_tol = 2     # Absolute Difference in RSI Values

    # Extract data to NumPy arrays
    CL = ohlc["low"].to_numpy()
    n = len(CL)
    dates = pd.to_datetime(ohlc['date'], utc=True).to_numpy()
    LM_Low_window_2_CS = ohlc['LM_Low_window_2_CS'].to_numpy()
    LM_Low_window_1_CS = ohlc['LM_Low_window_1_CS'].to_numpy()
    LM_Low_window_2_MACD = ohlc['LM_Low_window_2_MACD'].to_numpy()
    LM_Low_window_1_MACD = ohlc['LM_Low_window_1_MACD'].to_numpy()
    RSI = ohlc['RSI'].to_numpy()
    macd_histogram = ohlc['macd_histogram'].to_numpy()

    # Initialize output arrays for CBullD_1 (based on Candlestick Lows)
    CBullD_1 = np.zeros(n)
    CBullD_Lower_Low_1 = np.zeros(n)
    CBullD_Higher_Low_1 = np.zeros(n)
    CBullD_Lower_Low_RSI_1 = np.zeros(n)
    CBullD_Higher_Low_RSI_1 = np.zeros(n)
    CBullD_Lower_Low_MACD_1 = np.zeros(n)
    CBullD_Higher_Low_MACD_1 = np.zeros(n)
    CBullD_Lower_Low_date_1 = np.empty(n, dtype=object)
    CBullD_Higher_Low_date_1 = np.empty(n, dtype=object)

    CBullD_Higher_Low_1_2 = np.zeros(n)
    CBullD_Higher_Low_RSI_1_2 = np.zeros(n)
    CBullD_Higher_Low_MACD_1_2 = np.zeros(n)
    CBullD_Higher_Low_date_1_2 = np.empty(n, dtype=object)

    # Initialize output arrays for CBullD_2 (based on MACD)
    CBullD_2 = np.zeros(n)
    CBullD_Lower_Low_2 = np.zeros(n)
    CBullD_Higher_Low_2 = np.zeros(n)
    CBullD_Lower_Low_RSI_2 = np.zeros(n)
    CBullD_Higher_Low_RSI_2 = np.zeros(n)
    CBullD_Lower_Low_MACD_2 = np.zeros(n)
    CBullD_Higher_Low_MACD_2 = np.zeros(n)
    CBullD_Lower_Low_date_2 = np.empty(n, dtype=object)
    CBullD_Higher_Low_date_2 = np.empty(n, dtype=object)

    CBullD_Higher_Low_2_2 = np.zeros(n)
    CBullD_Higher_Low_RSI_2_2 = np.zeros(n)
    CBullD_Higher_Low_MACD_2_2 = np.zeros(n)
    CBullD_Higher_Low_date_2_2 = np.empty(n, dtype=object)

    # Helper function to find the last two non-zero indices before each index
    def get_last_two_nonzero_indices(indices, lm_low_window, extra_condition=None):
        last_nonzero = np.full(len(indices), -1, dtype=int)
        second_last_nonzero = np.full(len(indices), -1, dtype=int)
        for j, idx in enumerate(indices):
            if extra_condition is None:
                nonzero_before = np.where(lm_low_window[:idx] != 0)[0]
            else:
                nonzero_before = np.where((lm_low_window[:idx] != 0) & extra_condition[:idx])[0]
            if len(nonzero_before) >= 2:
                last_nonzero[j] = nonzero_before[-1]
                second_last_nonzero[j] = nonzero_before[-2]
        return last_nonzero, second_last_nonzero

    # Process CBullD_1: based on Candlestick Lows
    valid_indices_1 = np.where(LM_Low_window_2_CS > 0)[0]
    valid_indices_1 = valid_indices_1[(valid_indices_1 >= window_1) & (valid_indices_1 < n - window_2)]
    last_nonzero_1, second_last_1 = get_last_two_nonzero_indices(valid_indices_1, LM_Low_window_1_CS)
    valid_mask_1 = second_last_1 != -1
    valid_indices_1 = valid_indices_1[valid_mask_1]
    last_nonzero_1 = last_nonzero_1[valid_mask_1]
    second_last_1 = second_last_1[valid_mask_1]

    spacing_mask_1 = (valid_indices_1 - last_nonzero_1) > window_1
    valid_indices_1 = valid_indices_1[spacing_mask_1]
    last_nonzero_1 = last_nonzero_1[spacing_mask_1]
    second_last_1 = second_last_1[spacing_mask_1]

    if len(valid_indices_1) > 0:
        # Point 3 (current lower low)
        point3_low = LM_Low_window_2_CS[valid_indices_1]
        point3_rsi = RSI[valid_indices_1]
        point3_macd = macd_histogram[valid_indices_1]
        point3_date = dates[valid_indices_1]

        # Point 2 (higher low)
        point2_low = LM_Low_window_1_CS[last_nonzero_1]
        point2_rsi = RSI[last_nonzero_1]
        point2_macd = macd_histogram[last_nonzero_1]
        point2_date = dates[last_nonzero_1]

        # Point 1 (higher low x2)
        point1_low = LM_Low_window_1_CS[second_last_1]
        point1_rsi = RSI[second_last_1]
        point1_macd = macd_histogram[second_last_1]
        point1_date = dates[second_last_1]

        # Conditions between point1 and point2
        price_diff_12 = point2_low - point1_low
        rsi_diff_12 = point2_rsi - point1_rsi
        macd_diff_12 = point2_macd - point1_macd
        prev_div_1 = (price_diff_12 < 0) & (rsi_diff_12 > 0) & (macd_diff_12 > 0)

        # Conditions between point2 and point3
        price_diff_23 = point3_low - point2_low
        price_diff_percent_23 = np.abs(100 * price_diff_23 / point2_low)
        rsi_diff_23 = point3_rsi - point2_rsi
        macd_diff_23 = point3_macd - point2_macd
        macd_sign_check_23 = point3_macd * point2_macd

        price_condition_23 = (price_diff_23 < 0) | (price_diff_percent_23 < Candle_Tol)
        rsi_condition_23 = (
            (rsi_diff_23 > 0) |
            (np.abs(rsi_diff_23) < RSI_tol) |
            ((macd_sign_check_23 > 0) & (np.abs(rsi_diff_23) < 4 * RSI_tol)) |
            ((point3_rsi < 40) & (np.abs(rsi_diff_23) < 4 * RSI_tol))
        )
        macd_condition_23 = macd_diff_23 > 0

        curr_div_1 = price_condition_23 & rsi_condition_23 & macd_condition_23

        bullish_divergence_1 = prev_div_1 & curr_div_1

        valid_bd_1 = valid_indices_1[bullish_divergence_1]

        CBullD_1[valid_bd_1] = 1
        CBullD_Lower_Low_1[valid_bd_1] = point3_low[bullish_divergence_1]
        CBullD_Higher_Low_1[valid_bd_1] = point2_low[bullish_divergence_1]
        CBullD_Lower_Low_RSI_1[valid_bd_1] = point3_rsi[bullish_divergence_1]
        CBullD_Higher_Low_RSI_1[valid_bd_1] = point2_rsi[bullish_divergence_1]
        CBullD_Lower_Low_MACD_1[valid_bd_1] = point3_macd[bullish_divergence_1]
        CBullD_Higher_Low_MACD_1[valid_bd_1] = point2_macd[bullish_divergence_1]
        CBullD_Lower_Low_date_1[valid_bd_1] = point3_date[bullish_divergence_1]
        CBullD_Higher_Low_date_1[valid_bd_1] = point2_date[bullish_divergence_1]

        CBullD_Higher_Low_1_2[valid_bd_1] = point1_low[bullish_divergence_1]
        CBullD_Higher_Low_RSI_1_2[valid_bd_1] = point1_rsi[bullish_divergence_1]
        CBullD_Higher_Low_MACD_1_2[valid_bd_1] = point1_macd[bullish_divergence_1]
        CBullD_Higher_Low_date_1_2[valid_bd_1] = point1_date[bullish_divergence_1]

    # Process CBullD_2: based on MACD
    valid_indices_2 = np.where(LM_Low_window_2_MACD > 0)[0]
    valid_indices_2 = valid_indices_2[(valid_indices_2 >= window_1) & (valid_indices_2 < n - window_2)]
    last_nonzero_2, second_last_2 = get_last_two_nonzero_indices(valid_indices_2, LM_Low_window_1_MACD)
    valid_mask_2 = second_last_2 != -1
    valid_indices_2 = valid_indices_2[valid_mask_2]
    last_nonzero_2 = last_nonzero_2[valid_mask_2]
    second_last_2 = second_last_2[valid_mask_2]

    spacing_mask_2 = (valid_indices_2 - last_nonzero_2) > window_1
    valid_indices_2 = valid_indices_2[spacing_mask_2]
    last_nonzero_2 = last_nonzero_2[spacing_mask_2]
    second_last_2 = second_last_2[spacing_mask_2]

    if len(valid_indices_2) > 0:
        # Point 3 (current lower low)
        point3_low = LM_Low_window_2_MACD[valid_indices_2]
        point3_rsi = RSI[valid_indices_2]
        point3_macd = macd_histogram[valid_indices_2]
        point3_date = dates[valid_indices_2]

        # Point 2 (higher low)
        point2_low = LM_Low_window_1_MACD[last_nonzero_2]
        point2_rsi = RSI[last_nonzero_2]
        point2_macd = macd_histogram[last_nonzero_2]
        point2_date = dates[last_nonzero_2]

        # Point 1 (higher low x2)
        point1_low = LM_Low_window_1_MACD[second_last_2]
        point1_rsi = RSI[second_last_2]
        point1_macd = macd_histogram[second_last_2]
        point1_date = dates[second_last_2]

        # Conditions between point1 and point2
        price_diff_12 = point2_low - point1_low
        rsi_diff_12 = point2_rsi - point1_rsi
        macd_diff_12 = point2_macd - point1_macd
        prev_div_2 = (price_diff_12 < 0) & (rsi_diff_12 > 0) & (macd_diff_12 > 0)

        # Conditions between point2 and point3
        price_diff_23 = point3_low - point2_low
        price_diff_percent_23 = np.abs(100 * price_diff_23 / point2_low)
        rsi_diff_23 = point3_rsi - point2_rsi
        macd_diff_23 = point3_macd - point2_macd
        macd_diff_percent_23 = np.abs(100 * macd_diff_23 / (0.000001+point2_macd))

        price_condition_23 = (price_diff_23 < 0) | (price_diff_percent_23 < Candle_Tol)
        rsi_condition_23 = (rsi_diff_23 > 0) | (np.abs(rsi_diff_23) < RSI_tol)
        macd_condition_23 = (
            (point3_macd < 0) & (point2_macd < 0) &
            ((macd_diff_23 > 0) | (macd_diff_percent_23 < MACD_tol))
        )

        curr_div_2 = price_condition_23 & rsi_condition_23 & macd_condition_23

        bullish_divergence_2 = prev_div_2 & curr_div_2

        valid_bd_2 = valid_indices_2[bullish_divergence_2]

        CBullD_2[valid_bd_2] = 1
        CBullD_Lower_Low_2[valid_bd_2] = point3_low[bullish_divergence_2]
        CBullD_Higher_Low_2[valid_bd_2] = point2_low[bullish_divergence_2]
        CBullD_Lower_Low_RSI_2[valid_bd_2] = point3_rsi[bullish_divergence_2]
        CBullD_Higher_Low_RSI_2[valid_bd_2] = point2_rsi[bullish_divergence_2]
        CBullD_Lower_Low_MACD_2[valid_bd_2] = point3_macd[bullish_divergence_2]
        CBullD_Higher_Low_MACD_2[valid_bd_2] = point2_macd[bullish_divergence_2]
        CBullD_Lower_Low_date_2[valid_bd_2] = point3_date[bullish_divergence_2]
        CBullD_Higher_Low_date_2[valid_bd_2] = point2_date[bullish_divergence_2]

        CBullD_Higher_Low_2_2[valid_bd_2] = point1_low[bullish_divergence_2]
        CBullD_Higher_Low_RSI_2_2[valid_bd_2] = point1_rsi[bullish_divergence_2]
        CBullD_Higher_Low_MACD_2_2[valid_bd_2] = point1_macd[bullish_divergence_2]
        CBullD_Higher_Low_date_2_2[valid_bd_2] = point1_date[bullish_divergence_2]

    # Combine CBullD_1 and CBullD_2 into CBullD_x2
    CBullD_x2 = np.zeros(n)
    CBullD_x2_Lower_Low = np.zeros(n)
    CBullD_x2_Higher_Low = np.zeros(n)
    CBullD_x2_Lower_Low_RSI = np.zeros(n)
    CBullD_x2_Higher_Low_RSI = np.zeros(n)
    CBullD_x2_Lower_Low_MACD = np.zeros(n)
    CBullD_x2_Higher_Low_MACD = np.zeros(n)
    CBullD_x2_Lower_Low_date = np.empty(n, dtype=object)
    CBullD_x2_Higher_Low_date = np.empty(n, dtype=object)

    CBullD_x2_Higher_Low_x2 = np.zeros(n)
    CBullD_x2_Higher_Low_RSI_x2 = np.zeros(n)
    CBullD_x2_Higher_Low_MACD_x2 = np.zeros(n)
    CBullD_x2_Higher_Low_date_x2 = np.empty(n, dtype=object)

    indices = np.arange(window_1, n - window_2)
    condition_1 = CBullD_1[indices] == 1
    condition_2 = CBullD_2[indices] == 1
    combined_condition = condition_1 | condition_2
    valid_indices = indices[combined_condition]

    for idx in valid_indices:
        if CBullD_1[idx] == 1 or (CBullD_1[idx] == 1 and CBullD_2[idx] == 1):
            CBullD_x2[idx] = 1
            CBullD_x2_Lower_Low[idx] = CBullD_Lower_Low_1[idx]
            CBullD_x2_Higher_Low[idx] = CBullD_Higher_Low_1[idx]
            CBullD_x2_Lower_Low_RSI[idx] = CBullD_Lower_Low_RSI_1[idx]
            CBullD_x2_Higher_Low_RSI[idx] = CBullD_Higher_Low_RSI_1[idx]
            CBullD_x2_Lower_Low_MACD[idx] = CBullD_Lower_Low_MACD_1[idx]
            CBullD_x2_Higher_Low_MACD[idx] = CBullD_Higher_Low_MACD_1[idx]
            CBullD_x2_Lower_Low_date[idx] = CBullD_Lower_Low_date_1[idx]
            CBullD_x2_Higher_Low_date[idx] = CBullD_Higher_Low_date_1[idx]

            CBullD_x2_Higher_Low_x2[idx] = CBullD_Higher_Low_1_2[idx]
            CBullD_x2_Higher_Low_RSI_x2[idx] = CBullD_Higher_Low_RSI_1_2[idx]
            CBullD_x2_Higher_Low_MACD_x2[idx] = CBullD_Higher_Low_MACD_1_2[idx]
            CBullD_x2_Higher_Low_date_x2[idx] = CBullD_Higher_Low_date_1_2[idx]
        elif CBullD_2[idx] == 1:
            CBullD_x2[idx] = 1
            CBullD_x2_Lower_Low[idx] = CBullD_Lower_Low_2[idx]
            CBullD_x2_Higher_Low[idx] = CBullD_Higher_Low_2[idx]
            CBullD_x2_Lower_Low_RSI[idx] = CBullD_Lower_Low_RSI_2[idx]
            CBullD_x2_Higher_Low_RSI[idx] = CBullD_Higher_Low_RSI_2[idx]
            CBullD_x2_Lower_Low_MACD[idx] = CBullD_Lower_Low_MACD_2[idx]
            CBullD_x2_Higher_Low_MACD[idx] = CBullD_Higher_Low_MACD_2[idx]
            CBullD_x2_Lower_Low_date[idx] = CBullD_Lower_Low_date_2[idx]
            CBullD_x2_Higher_Low_date[idx] = CBullD_Higher_Low_date_2[idx]

            CBullD_x2_Higher_Low_x2[idx] = CBullD_Higher_Low_2_2[idx]
            CBullD_x2_Higher_Low_RSI_x2[idx] = CBullD_Higher_Low_RSI_2_2[idx]
            CBullD_x2_Higher_Low_MACD_x2[idx] = CBullD_Higher_Low_MACD_2_2[idx]
            CBullD_x2_Higher_Low_date_x2[idx] = CBullD_Higher_Low_date_2_2[idx]

    # Assign results to DataFrame
    ohlc['CBullD_x2'] = CBullD_x2
    ohlc['CBullD_x2_Lower_Low'] = CBullD_x2_Lower_Low
    ohlc['CBullD_x2_Higher_Low'] = CBullD_x2_Higher_Low
    ohlc['CBullD_x2_Lower_Low_RSI'] = CBullD_x2_Lower_Low_RSI
    ohlc['CBullD_x2_Higher_Low_RSI'] = CBullD_x2_Higher_Low_RSI
    ohlc['CBullD_x2_Lower_Low_MACD'] = CBullD_x2_Lower_Low_MACD
    ohlc['CBullD_x2_Higher_Low_MACD'] = CBullD_x2_Higher_Low_MACD
    ohlc['CBullD_x2_Lower_Low_date'] = CBullD_x2_Lower_Low_date
    ohlc['CBullD_x2_Higher_Low_date'] = CBullD_x2_Higher_Low_date

    ohlc['CBullD_x2_Higher_Low_x2'] = CBullD_x2_Higher_Low_x2
    ohlc['CBullD_x2_Higher_Low_RSI_x2'] = CBullD_x2_Higher_Low_RSI_x2
    ohlc['CBullD_x2_Higher_Low_MACD_x2'] = CBullD_x2_Higher_Low_MACD_x2
    ohlc['CBullD_x2_Higher_Low_date_x2'] = CBullD_x2_Higher_Low_date_x2

    return ohlc[['CBullD_x2']]