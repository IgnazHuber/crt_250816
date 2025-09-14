import pandas as pd
import numpy as np
from typing import Optional
import logging
from numba import njit

# Set up logging
logger = logging.getLogger(__name__)

# Constants
RSI_PERIOD = 14
EMA_SPANS = [12, 20, 26, 50, 100, 200]
MACD_SIGNAL_PERIOD = 9
STORAGE_PRECISION = {
    'price': 4,
    'rsi_components': 6,
    'rsi_final': 2,
    'ema': 4,
    'macd': 4
}

@njit(cache=True)
def calculate_rsi_jit(
    prices: np.ndarray,
    period: int,
    price_precision: int,
    rsi_components_precision: int,
    rsi_final_precision: int
) -> tuple:
    n = len(prices)
    changes = np.zeros(n, dtype=np.float64)
    gains = np.zeros(n, dtype=np.float64)
    losses = np.zeros(n, dtype=np.float64)
    avg_gains = np.zeros(n, dtype=np.float64)
    avg_losses = np.zeros(n, dtype=np.float64)
    rs_values = np.zeros(n, dtype=np.float64)
    rsi_values = np.zeros(n, dtype=np.float64)
    
    # Calculate price changes
    for i in range(1, n):
        changes[i] = np.round(prices[i] - prices[i-1], price_precision)
        if changes[i] > 0:
            gains[i] = changes[i]
        else:
            losses[i] = -changes[i]
    
    # Calculate initial averages
    if n > period:
        initial_avg_gain = 0.0
        initial_avg_loss = 0.0
        for i in range(1, period + 1):
            initial_avg_gain += gains[i]
            initial_avg_loss += losses[i]
        initial_avg_gain = np.round(initial_avg_gain / period, rsi_components_precision)
        initial_avg_loss = np.round(initial_avg_loss / period, rsi_components_precision)
        avg_gains[period] = initial_avg_gain
        avg_losses[period] = initial_avg_loss
        
        # Wilder's smoothing
        for i in range(period + 1, n):
            avg_gains[i] = np.round(((avg_gains[i-1] * (period - 1)) + gains[i]) / period, rsi_components_precision)
            avg_losses[i] = np.round(((avg_losses[i-1] * (period - 1)) + losses[i]) / period, rsi_components_precision)
            if avg_losses[i] != 0:
                rs_values[i] = np.round(avg_gains[i] / avg_losses[i], rsi_components_precision)
                rsi_values[i] = np.round(100.0 - (100.0 / (1.0 + rs_values[i])), rsi_final_precision)
            else:
                rsi_values[i] = 100.0
    
    return changes, gains, losses, avg_gains, avg_losses, rs_values, rsi_values

@njit(cache=True)
def calculate_ema_jit(prices: np.ndarray, span: int, ema_precision: int) -> np.ndarray:
    n = len(prices)
    ema = np.zeros(n, dtype=np.float64)
    if n == 0:
        return ema
    alpha = 2.0 / (span + 1.0)
    ema[0] = prices[0]
    for i in range(1, n):
        ema[i] = np.round((prices[i] * alpha) + (ema[i-1] * (1.0 - alpha)), ema_precision)
    return ema

@njit(cache=True)
def calculate_macd_jit(
    prices: np.ndarray,
    fast_period: int,
    slow_period: int,
    signal_period: int,
    macd_precision: int,
    ema_precision: int
) -> tuple:
    n = len(prices)
    macd_line = np.zeros(n, dtype=np.float64)
    signal_line = np.zeros(n, dtype=np.float64)
    histogram = np.zeros(n, dtype=np.float64)
    if n == 0:
        return macd_line, signal_line, histogram
    ema_fast = calculate_ema_jit(prices, fast_period, ema_precision)
    ema_slow = calculate_ema_jit(prices, slow_period, ema_precision)
    for i in range(n):
        macd_line[i] = np.round(ema_fast[i] - ema_slow[i], macd_precision)
    signal_line = calculate_ema_jit(macd_line, signal_period, ema_precision)
    for i in range(n):
        histogram[i] = np.round(macd_line[i] - signal_line[i], macd_precision)
    return macd_line, signal_line, histogram

def Initialize_RSI_EMA_MACD(df_input: pd.DataFrame) -> pd.DataFrame:
    logger.info("Initializing technical indicators with Numba JIT")
    close_prices = df_input['close'].to_numpy(dtype=np.float64)
    
    # RSI
    changes, gains, losses, avg_gains, avg_losses, rs_values, rsi_values = calculate_rsi_jit(
        close_prices,
        RSI_PERIOD,
        STORAGE_PRECISION['price'],
        STORAGE_PRECISION['rsi_components'],
        STORAGE_PRECISION['rsi_final']
    )
    
    # EMAs
    ema_results = {}
    for span in EMA_SPANS:
        ema_results[f'EMA_{span}'] = calculate_ema_jit(close_prices, span, STORAGE_PRECISION['ema'])
    
    # MACD
    macd_line, signal_line, histogram = calculate_macd_jit(
        close_prices,
        fast_period=12,
        slow_period=26,
        signal_period=MACD_SIGNAL_PERIOD,
        macd_precision=STORAGE_PRECISION['macd'],
        ema_precision=STORAGE_PRECISION['ema']
    )
    
    # Assign to DataFrame
    df_input['price_change'] = changes
    df_input['gain'] = gains
    df_input['loss'] = losses
    df_input['avg_gain'] = avg_gains
    df_input['avg_loss'] = avg_losses
    df_input['RS'] = rs_values
    df_input['RSI'] = rsi_values
    df_input['macd'] = macd_line
    df_input['signal'] = signal_line
    df_input['macd_histogram'] = histogram
    for col_name, values in ema_results.items():
        df_input[col_name] = values
    
    logger.info("Technical indicators initialized successfully")
    return df_input