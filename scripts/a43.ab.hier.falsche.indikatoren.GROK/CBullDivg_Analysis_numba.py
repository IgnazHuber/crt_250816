import pandas as pd
import numpy as np
import warnings
import logging
from numba import njit

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)

@njit(cache=True)
def analyze_bullish_divergence_jit(
    low_prices: np.ndarray,
    dates_ordinal: np.ndarray,
    LM_Low_window_2_CS: np.ndarray,
    LM_Low_window_1_CS: np.ndarray,
    LM_Low_window_2_MACD: np.ndarray,
    LM_Low_window_1_MACD: np.ndarray,
    RSI: np.ndarray,
    macd_histogram: np.ndarray,
    window: int,
    candle_tol: float,
    macd_tol: float,
    rsi_tol: float = 2.0
) -> tuple:
    """
    Ultra-fast JIT compiled bullish divergence analysis
    Matches original logic with corrected search range and precision
    """
    n = len(low_prices)
    
    # Initialize output arrays
    CBullD_1 = np.zeros(n, dtype=np.float64)
    CBullD_Lower_Low_1 = np.zeros(n, dtype=np.float64)
    CBullD_Higher_Low_1 = np.zeros(n, dtype=np.float64)
    CBullD_Lower_Low_RSI_1 = np.zeros(n, dtype=np.float64)
    CBullD_Higher_Low_RSI_1 = np.zeros(n, dtype=np.float64)
    CBullD_Lower_Low_MACD_1 = np.zeros(n, dtype=np.float64)
    CBullD_Higher_Low_MACD_1 = np.zeros(n, dtype=np.float64)
    CBullD_Lower_Low_date_1 = np.zeros(n, dtype=np.float64)
    CBullD_Higher_Low_date_1 = np.zeros(n, dtype=np.float64)
    CBullD_Date_Gap_1 = np.zeros(n, dtype=np.float64)
    
    CBullD_2 = np.zeros(n, dtype=np.float64)
    CBullD_Lower_Low_2 = np.zeros(n, dtype=np.float64)
    CBullD_Higher_Low_2 = np.zeros(n, dtype=np.float64)
    CBullD_Lower_Low_RSI_2 = np.zeros(n, dtype=np.float64)
    CBullD_Higher_Low_RSI_2 = np.zeros(n, dtype=np.float64)
    CBullD_Lower_Low_MACD_2 = np.zeros(n, dtype=np.float64)
    CBullD_Higher_Low_MACD_2 = np.zeros(n, dtype=np.float64)
    CBullD_Lower_Low_date_2 = np.zeros(n, dtype=np.float64)
    CBullD_Higher_Low_date_2 = np.zeros(n, dtype=np.float64)
    CBullD_Date_Gap_2 = np.zeros(n, dtype=np.float64)
    
    CBullD_neg_MACD = np.zeros(n, dtype=np.float64)
    CBullD_Lower_Low_neg_MACD = np.zeros(n, dtype=np.float64)
    CBullD_Higher_Low_neg_MACD = np.zeros(n, dtype=np.float64)
    CBullD_Lower_Low_RSI_neg_MACD = np.zeros(n, dtype=np.float64)
    CBullD_Higher_Low_RSI_neg_MACD = np.zeros(n, dtype=np.float64)
    CBullD_Lower_Low_MACD_neg_MACD = np.zeros(n, dtype=np.float64)
    CBullD_Higher_Low_MACD_neg_MACD = np.zeros(n, dtype=np.float64)
    CBullD_Lower_Low_date_neg_MACD = np.zeros(n, dtype=np.float64)
    CBullD_Higher_Low_date_neg_MACD = np.zeros(n, dtype=np.float64)
    CBullD_Date_Gap_neg_MACD = np.zeros(n, dtype=np.float64)
    
    # Hidden Bullish Divergence (CBullD_1)
    for i in range(window, n):
        if LM_Low_window_2_CS[i] != 0 and not np.isnan(RSI[i]) and not np.isnan(macd_histogram[i]) and not np.isnan(low_prices[i]):
            for j in range(i - 1, window - 1, -1):
                if j < 0:  # Sicherstellen, dass j nicht negativ wird
                    break
                if LM_Low_window_2_CS[j] != 0 and not np.isnan(RSI[j]) and not np.isnan(macd_histogram[j]) and not np.isnan(low_prices[j]):
                    lower_low = low_prices[i] < low_prices[j] * (1 - candle_tol)
                    higher_rsi = RSI[i] > RSI[j] - rsi_tol
                    higher_macd = macd_histogram[i] > macd_histogram[j] * (1 - macd_tol)
                    if lower_low and higher_rsi and higher_macd:
                        CBullD_1[i] = 1.0
                        CBullD_Lower_Low_1[i] = low_prices[j]
                        CBullD_Higher_Low_1[i] = low_prices[i]
                        CBullD_Lower_Low_RSI_1[i] = RSI[j]
                        CBullD_Higher_Low_RSI_1[i] = RSI[i]
                        CBullD_Lower_Low_MACD_1[i] = macd_histogram[j]
                        CBullD_Higher_Low_MACD_1[i] = macd_histogram[i]
                        CBullD_Lower_Low_date_1[i] = dates_ordinal[j]
                        CBullD_Higher_Low_date_1[i] = dates_ordinal[i]
                        CBullD_Date_Gap_1[i] = dates_ordinal[i] - dates_ordinal[j]
                    break
    
    # Classic Bullish Divergence (CBullD_2)
    for i in range(window, n):
        if LM_Low_window_2_MACD[i] != 0 and not np.isnan(RSI[i]) and not np.isnan(macd_histogram[i]) and not np.isnan(low_prices[i]):
            for j in range(i - 1, window - 1, -1):
                if j < 0:
                    break
                if LM_Low_window_2_MACD[j] != 0 and not np.isnan(RSI[j]) and not np.isnan(macd_histogram[j]) and not np.isnan(low_prices[j]):
                    lower_low = low_prices[i] < low_prices[j] * (1 - candle_tol)
                    higher_rsi = RSI[i] > RSI[j] - rsi_tol
                    higher_macd = macd_histogram[i] > macd_histogram[j] * (1 - macd_tol)
                    if lower_low and higher_rsi and higher_macd:
                        CBullD_2[i] = 1.0
                        CBullD_Lower_Low_2[i] = low_prices[j]
                        CBullD_Higher_Low_2[i] = low_prices[i]
                        CBullD_Lower_Low_RSI_2[i] = RSI[j]
                        CBullD_Higher_Low_RSI_2[i] = RSI[i]
                        CBullD_Lower_Low_MACD_2[i] = macd_histogram[j]
                        CBullD_Higher_Low_MACD_2[i] = macd_histogram[i]
                        CBullD_Lower_Low_date_2[i] = dates_ordinal[j]
                        CBullD_Higher_Low_date_2[i] = dates_ordinal[i]
                        CBullD_Date_Gap_2[i] = dates_ordinal[i] - dates_ordinal[j]
                    break
    
    # Negative MACD Bullish Divergence
    for i in range(window, n):
        if LM_Low_window_2_MACD[i] != 0 and macd_histogram[i] < 0 and not np.isnan(RSI[i]) and not np.isnan(macd_histogram[i]) and not np.isnan(low_prices[i]):
            for j in range(i - 1, window - 1, -1):
                if j < 0:
                    break
                if LM_Low_window_2_MACD[j] != 0 and not np.isnan(RSI[j]) and not np.isnan(macd_histogram[j]) and not np.isnan(low_prices[j]):
                    lower_low = low_prices[i] < low_prices[j] * (1 - candle_tol)
                    higher_rsi = RSI[i] > RSI[j] - rsi_tol
                    higher_macd = macd_histogram[i] > macd_histogram[j] * (1 - macd_tol)
                    if lower_low and higher_rsi and higher_macd:
                        CBullD_neg_MACD[i] = 1.0
                        CBullD_Lower_Low_neg_MACD[i] = low_prices[j]
                        CBullD_Higher_Low_neg_MACD[i] = low_prices[i]
                        CBullD_Lower_Low_RSI_neg_MACD[i] = RSI[j]
                        CBullD_Higher_Low_RSI_neg_MACD[i] = RSI[i]
                        CBullD_Lower_Low_MACD_neg_MACD[i] = macd_histogram[j]
                        CBullD_Higher_Low_MACD_neg_MACD[i] = macd_histogram[i]
                        CBullD_Lower_Low_date_neg_MACD[i] = dates_ordinal[j]
                        CBullD_Higher_Low_date_neg_MACD[i] = dates_ordinal[i]
                        CBullD_Date_Gap_neg_MACD[i] = dates_ordinal[i] - dates_ordinal[j]
                    break
    
    return (
        CBullD_1, CBullD_Lower_Low_1, CBullD_Higher_Low_1, CBullD_Lower_Low_RSI_1, CBullD_Higher_Low_RSI_1,
        CBullD_Lower_Low_MACD_1, CBullD_Higher_Low_MACD_1, CBullD_Lower_Low_date_1, CBullD_Higher_Low_date_1, CBullD_Date_Gap_1,
        CBullD_2, CBullD_Lower_Low_2, CBullD_Higher_Low_2, CBullD_Lower_Low_RSI_2, CBullD_Higher_Low_RSI_2,
        CBullD_Lower_Low_MACD_2, CBullD_Higher_Low_MACD_2, CBullD_Lower_Low_date_2, CBullD_Higher_Low_date_2, CBullD_Date_Gap_2,
        CBullD_neg_MACD, CBullD_Lower_Low_neg_MACD, CBullD_Higher_Low_neg_MACD, CBullD_Lower_Low_RSI_neg_MACD,
        CBullD_Higher_Low_RSI_neg_MACD, CBullD_Lower_Low_MACD_neg_MACD, CBullD_Higher_Low_MACD_neg_MACD,
        CBullD_Lower_Low_date_neg_MACD, CBullD_Higher_Low_date_neg_MACD, CBullD_Date_Gap_neg_MACD
    )

def CBullDivg_analysis(ohlc: pd.DataFrame, window: int, Candle_Tol: float, MACD_tol: float) -> pd.DataFrame:
    """
    High-performance bullish divergence analysis using Numba JIT compilation
    Returns DataFrame with all divergence columns, matching original logic
    """
    logger.info("Starting bullish divergence analysis with Numba JIT")
    
    # Ensure numeric columns and handle NaNs
    required_columns = [
        'low', 'date', 'LM_Low_window_2_CS', 'LM_Low_window_1_CS',
        'LM_Low_window_2_MACD', 'LM_Low_window_1_MACD', 'RSI', 'macd_histogram'
    ]
    missing_columns = [col for col in required_columns if col not in ohlc.columns]
    if missing_columns:
        logger.error(f"Missing required columns: {missing_columns}")
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    for col in required_columns:
        ohlc[col] = pd.to_numeric(ohlc[col], errors='coerce')
        if ohlc[col].isna().any():
            logger.warning(f"NaN values detected in {col}, filling with 0")
            ohlc[col] = ohlc[col].fillna(0)
    
    # Convert inputs to NumPy arrays
    low_prices = ohlc["low"].to_numpy(dtype=np.float64)
    dates_ordinal = pd.to_datetime(ohlc['date']).apply(lambda x: x.toordinal() if pd.notna(x) else 0).to_numpy(dtype=np.float64)
    LM_Low_window_2_CS = ohlc['LM_Low_window_2_CS'].to_numpy(dtype=np.float64)
    LM_Low_window_1_CS = ohlc['LM_Low_window_1_CS'].to_numpy(dtype=np.float64)
    LM_Low_window_2_MACD = ohlc['LM_Low_window_2_MACD'].to_numpy(dtype=np.float64)
    LM_Low_window_1_MACD = ohlc['LM_Low_window_1_MACD'].to_numpy(dtype=np.float64)
    RSI = ohlc['RSI'].to_numpy(dtype=np.float64)
    macd_histogram = ohlc['macd_histogram'].to_numpy(dtype=np.float64)
    
    # Log input data shapes and basic statistics for debugging
    logger.debug(f"Input shapes: low_prices={low_prices.shape}, RSI={RSI.shape}, macd_histogram={macd_histogram.shape}")
    logger.debug(f"LM_Low_window_2_CS non-zero count: {np.sum(LM_Low_window_2_CS != 0)}")
    logger.debug(f"LM_Low_window_2_MACD non-zero count: {np.sum(LM_Low_window_2_MACD != 0)}")
    
    # Call JIT compiled function
    results = analyze_bullish_divergence_jit(
        low_prices, dates_ordinal, LM_Low_window_2_CS, LM_Low_window_1_CS,
        LM_Low_window_2_MACD, LM_Low_window_1_MACD, RSI, macd_histogram,
        window, Candle_Tol, MACD_tol
    )
    
    # Unpack results
    (CBullD_1, CBullD_Lower_Low_1, CBullD_Higher_Low_1, CBullD_Lower_Low_RSI_1, CBullD_Higher_Low_RSI_1,
     CBullD_Lower_Low_MACD_1, CBullD_Higher_Low_MACD_1, CBullD_Lower_Low_date_1, CBullD_Higher_Low_date_1, CBullD_Date_Gap_1,
     CBullD_2, CBullD_Lower_Low_2, CBullD_Higher_Low_2, CBullD_Lower_Low_RSI_2, CBullD_Higher_Low_RSI_2,
     CBullD_Lower_Low_MACD_2, CBullD_Higher_Low_MACD_2, CBullD_Lower_Low_date_2, CBullD_Higher_Low_date_2, CBullD_Date_Gap_2,
     CBullD_neg_MACD, CBullD_Lower_Low_neg_MACD, CBullD_Higher_Low_neg_MACD, CBullD_Lower_Low_RSI_neg_MACD,
     CBullD_Higher_Low_RSI_neg_MACD, CBullD_Lower_Low_MACD_neg_MACD, CBullD_Higher_Low_MACD_neg_MACD,
     CBullD_Lower_Low_date_neg_MACD, CBullD_Higher_Low_date_neg_MACD, CBullD_Date_Gap_neg_MACD) = results
    
    # Combine CBullD_1, CBullD_2, CBullD_neg_MACD into CBullD_gen
    n = len(low_prices)
    CBullD_gen = np.zeros(n, dtype=np.float64)
    CBullD_Lower_Low_gen = np.zeros(n, dtype=np.float64)
    CBullD_Higher_Low_gen = np.zeros(n, dtype=np.float64)
    CBullD_Lower_Low_RSI_gen = np.zeros(n, dtype=np.float64)
    CBullD_Higher_Low_RSI_gen = np.zeros(n, dtype=np.float64)
    CBullD_Lower_Low_MACD_gen = np.zeros(n, dtype=np.float64)
    CBullD_Higher_Low_MACD_gen = np.zeros(n, dtype=np.float64)
    CBullD_Lower_Low_date_gen = np.zeros(n, dtype=np.float64)
    CBullD_Higher_Low_date_gen = np.zeros(n, dtype=np.float64)
    CBullD_Date_Gap_gen = np.zeros(n, dtype=np.float64)
    
    for idx in range(n):
        if CBullD_1[idx] == 1:
            CBullD_gen[idx] = 1
            CBullD_Lower_Low_gen[idx] = CBullD_Lower_Low_1[idx]
            CBullD_Higher_Low_gen[idx] = CBullD_Higher_Low_1[idx]
            CBullD_Lower_Low_RSI_gen[idx] = CBullD_Lower_Low_RSI_1[idx]
            CBullD_Higher_Low_RSI_gen[idx] = CBullD_Higher_Low_RSI_1[idx]
            CBullD_Lower_Low_MACD_gen[idx] = CBullD_Lower_Low_MACD_1[idx]
            CBullD_Higher_Low_MACD_gen[idx] = CBullD_Higher_Low_MACD_1[idx]
            CBullD_Lower_Low_date_gen[idx] = CBullD_Lower_Low_date_1[idx]
            CBullD_Higher_Low_date_gen[idx] = CBullD_Higher_Low_date_1[idx]
            CBullD_Date_Gap_gen[idx] = CBullD_Date_Gap_1[idx]
        elif CBullD_2[idx] == 1:
            CBullD_gen[idx] = 1
            CBullD_Lower_Low_gen[idx] = CBullD_Lower_Low_2[idx]
            CBullD_Higher_Low_gen[idx] = CBullD_Higher_Low_2[idx]
            CBullD_Lower_Low_RSI_gen[idx] = CBullD_Lower_Low_RSI_2[idx]
            CBullD_Higher_Low_RSI_gen[idx] = CBullD_Higher_Low_RSI_2[idx]
            CBullD_Lower_Low_MACD_gen[idx] = CBullD_Lower_Low_MACD_2[idx]
            CBullD_Higher_Low_MACD_gen[idx] = CBullD_Higher_Low_MACD_2[idx]
            CBullD_Lower_Low_date_gen[idx] = CBullD_Lower_Low_date_2[idx]
            CBullD_Higher_Low_date_gen[idx] = CBullD_Higher_Low_date_2[idx]
            CBullD_Date_Gap_gen[idx] = CBullD_Date_Gap_2[idx]
    
    # Convert date ordinals back to datetime objects
    def ordinal_to_date(ordinal_val):
        if ordinal_val == 0:
            return pd.NaT
        return pd.Timestamp.fromordinal(int(ordinal_val))
    
    # Add results to DataFrame
    ohlc['CBullD_1'] = CBullD_1
    ohlc['CBullD_Lower_Low_1'] = CBullD_Lower_Low_1
    ohlc['CBullD_Higher_Low_1'] = CBullD_Higher_Low_1
    ohlc['CBullD_Lower_Low_RSI_1'] = CBullD_Lower_Low_RSI_1
    ohlc['CBullD_Higher_Low_RSI_1'] = CBullD_Higher_Low_RSI_1
    ohlc['CBullD_Lower_Low_MACD_1'] = CBullD_Lower_Low_MACD_1
    ohlc['CBullD_Higher_Low_MACD_1'] = CBullD_Higher_Low_MACD_1
    ohlc['CBullD_Lower_Low_date_1'] = [ordinal_to_date(val) for val in CBullD_Lower_Low_date_1]
    ohlc['CBullD_Higher_Low_date_1'] = [ordinal_to_date(val) for val in CBullD_Higher_Low_date_1]
    ohlc['CBullD_Date_Gap_1'] = CBullD_Date_Gap_1
    ohlc['CBullD_2'] = CBullD_2
    ohlc['CBullD_Lower_Low_2'] = CBullD_Lower_Low_2
    ohlc['CBullD_Higher_Low_2'] = CBullD_Higher_Low_2
    ohlc['CBullD_Lower_Low_RSI_2'] = CBullD_Lower_Low_RSI_2
    ohlc['CBullD_Higher_Low_RSI_2'] = CBullD_Higher_Low_RSI_2
    ohlc['CBullD_Lower_Low_MACD_2'] = CBullD_Lower_Low_MACD_2
    ohlc['CBullD_Higher_Low_MACD_2'] = CBullD_Higher_Low_MACD_2
    ohlc['CBullD_Lower_Low_date_2'] = [ordinal_to_date(val) for val in CBullD_Lower_Low_date_2]
    ohlc['CBullD_Higher_Low_date_2'] = [ordinal_to_date(val) for val in CBullD_Higher_Low_date_2]
    ohlc['CBullD_Date_Gap_2'] = CBullD_Date_Gap_2
    ohlc['CBullD_gen'] = CBullD_gen
    ohlc['CBullD_Lower_Low_gen'] = CBullD_Lower_Low_gen
    ohlc['CBullD_Higher_Low_gen'] = CBullD_Higher_Low_gen
    ohlc['CBullD_Lower_Low_RSI_gen'] = CBullD_Lower_Low_RSI_gen
    ohlc['CBullD_Higher_Low_RSI_gen'] = CBullD_Higher_Low_RSI_gen
    ohlc['CBullD_Lower_Low_MACD_gen'] = CBullD_Lower_Low_MACD_gen
    ohlc['CBullD_Higher_Low_MACD_gen'] = CBullD_Higher_Low_MACD_gen
    ohlc['CBullD_Lower_Low_date_gen'] = [ordinal_to_date(val) for val in CBullD_Lower_Low_date_gen]
    ohlc['CBullD_Higher_Low_date_gen'] = [ordinal_to_date(val) for val in CBullD_Higher_Low_date_gen]
    ohlc['CBullD_Date_Gap_gen'] = CBullD_Date_Gap_gen
    ohlc['CBullD_neg_MACD'] = CBullD_neg_MACD
    ohlc['CBullD_Lower_Low_neg_MACD'] = CBullD_Lower_Low_neg_MACD
    ohlc['CBullD_Higher_Low_neg_MACD'] = CBullD_Higher_Low_neg_MACD
    ohlc['CBullD_Lower_Low_RSI_neg_MACD'] = CBullD_Lower_Low_RSI_neg_MACD
    ohlc['CBullD_Higher_Low_RSI_neg_MACD'] = CBullD_Higher_Low_RSI_neg_MACD
    ohlc['CBullD_Lower_Low_MACD_neg_MACD'] = CBullD_Lower_Low_MACD_neg_MACD
    ohlc['CBullD_Higher_Low_MACD_neg_MACD'] = CBullD_Higher_Low_MACD_neg_MACD
    ohlc['CBullD_Lower_Low_date_neg_MACD'] = [ordinal_to_date(val) for val in CBullD_Lower_Low_date_neg_MACD]
    ohlc['CBullD_Higher_Low_date_neg_MACD'] = [ordinal_to_date(val) for val in CBullD_Higher_Low_date_neg_MACD]
    ohlc['CBullD_Date_Gap_neg_MACD'] = CBullD_Date_Gap_neg_MACD
    
    # Log number of divergences found
    logger.info(f"Found {int(np.sum(CBullD_1))} Hidden Bullish Divergences (CBullD_1)")
    logger.info(f"Found {int(np.sum(CBullD_2))} Classic Bullish Divergences (CBullD_2)")
    logger.info(f"Found {int(np.sum(CBullD_neg_MACD))} Negative MACD Divergences")
    logger.info(f"Found {int(np.sum(CBullD_gen))} Combined Divergences (CBullD_gen)")
    
    # Verify CBullD_1 exists
    if 'CBullD_1' not in ohlc.columns:
        logger.error("Failed to create CBullD_1 column")
        raise ValueError("CBullD_1 column missing in DataFrame")
    
    # Return only required columns as per original
    logger.info("Bullish divergence analysis completed")
    return ohlc[['CBullD_gen', 'CBullD_neg_MACD']]