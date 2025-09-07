import logging
from decimal import ROUND_HALF_UP, Decimal, getcontext
from typing import Optional

import numpy as np
import pandas as pd

# Set up logging
logger = logging.getLogger(__name__)

# Constants for better maintainability
RSI_PERIOD = 14
EMA_SPANS = [12, 20, 26, 50, 100, 200]
MACD_SIGNAL_PERIOD = 9

# PRECISION CONTROL: Set high precision for calculations to minimize accumulation errors
getcontext().prec = 28  # High precision for decimal calculations
CALCULATION_PRECISION = np.float64  # Use float64 for all calculations
STORAGE_PRECISION = {
    "price": 4,  # 4 decimal places for prices and price changes
    "rsi_components": 6,  # 6 decimal places for RSI intermediate values
    "rsi_final": 2,  # 2 decimal places for final RSI
    "ema": 4,  # 4 decimal places for EMAs
    "macd": 4,  # 4 decimal places for MACD values
}


def precise_round(value: float, decimals: int) -> float:
    """
    High-precision rounding to minimize accumulation errors
    Uses Decimal for precise rounding operations
    """
    if np.isnan(value) or np.isinf(value):
        return value

    # Convert to Decimal for precise rounding
    decimal_value = Decimal(str(value))
    rounded_decimal = decimal_value.quantize(
        Decimal("0.1") ** decimals, rounding=ROUND_HALF_UP
    )
    return float(rounded_decimal)


def calculate_rsi_vectorized(
    prices: np.ndarray, period: int = RSI_PERIOD
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    High-precision vectorized RSI calculation with controlled accumulation errors.

    Args:
        prices: Array of closing prices
        period: RSI period (default 14)

    Returns:
        Tuple of (avg_gains, avg_losses, rs_values, rsi_values)
    """
    n = len(prices)

    # PRECISION: Use float64 for all calculations to minimize errors
    prices = prices.astype(CALCULATION_PRECISION)

    # Calculate price changes
    price_changes = np.diff(prices, prepend=prices[0])

    # Calculate gains and losses vectorized
    gains = np.where(price_changes > 0, price_changes, 0.0)
    losses = np.where(price_changes < 0, -price_changes, 0.0)

    # Initialize arrays with high precision
    avg_gains = np.zeros(n, dtype=CALCULATION_PRECISION)
    avg_losses = np.zeros(n, dtype=CALCULATION_PRECISION)

    if n <= period:
        return avg_gains, avg_losses, np.zeros(n), np.zeros(n)

    # Calculate initial averages (SMA for first period) - high precision
    initial_avg_gain = np.mean(gains[1 : period + 1].astype(CALCULATION_PRECISION))
    initial_avg_loss = np.mean(losses[1 : period + 1].astype(CALCULATION_PRECISION))

    avg_gains[period] = precise_round(
        float(initial_avg_gain), STORAGE_PRECISION["rsi_components"]
    )
    avg_losses[period] = precise_round(
        float(initial_avg_loss), STORAGE_PRECISION["rsi_components"]
    )

    # PRECISION: Use high-precision alpha and controlled accumulation
    alpha = CALCULATION_PRECISION(1.0) / CALCULATION_PRECISION(period)
    one_minus_alpha = CALCULATION_PRECISION(1.0) - alpha

    for i in range(period + 1, n):
        # High-precision EMA calculation with controlled rounding
        raw_avg_gain = alpha * gains[i] + one_minus_alpha * avg_gains[i - 1]
        raw_avg_loss = alpha * losses[i] + one_minus_alpha * avg_losses[i - 1]

        # Apply precision control to prevent accumulation errors
        avg_gains[i] = precise_round(raw_avg_gain, STORAGE_PRECISION["rsi_components"])
        avg_losses[i] = precise_round(raw_avg_loss, STORAGE_PRECISION["rsi_components"])

    # Calculate RS and RSI with precision control
    rs_values = np.divide(
        avg_gains, avg_losses, out=np.zeros_like(avg_gains), where=avg_losses != 0
    )

    # Apply precision rounding to RS values
    rs_values = np.array(
        [precise_round(rs, STORAGE_PRECISION["rsi_components"]) for rs in rs_values]
    )

    # Calculate RSI with high precision
    rsi_values = np.where(avg_losses != 0, 100.0 - (100.0 / (1.0 + rs_values)), 100.0)

    # Apply final precision to RSI
    rsi_values = np.array(
        [precise_round(rsi, STORAGE_PRECISION["rsi_final"]) for rsi in rsi_values]
    )

    return avg_gains, avg_losses, rs_values, rsi_values


def Initialize_RSI_EMA_MACD(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Optimized initialization of RSI, EMA, and MACD indicators with vectorized operations.

    Args:
        df: DataFrame with OHLCV data (must contain 'close' column)

    Returns:
        DataFrame with additional technical indicator columns, or None if error
    """
    try:
        if df.empty or "close" not in df.columns:
            logger.error("Invalid DataFrame: empty or missing 'close' column")
            return None

        if len(df) < RSI_PERIOD + 1:
            logger.warning(
                f"Insufficient data: need at least {RSI_PERIOD + 1} rows for RSI calculation"
            )
            return df

        logger.info(f"Initializing technical indicators for {len(df)} rows")

        # Convert to numpy for faster operations with high precision
        close_prices = df["close"].values.astype(CALCULATION_PRECISION)

        # PRECISION: High-precision RSI calculation with controlled accumulation
        avg_gains, avg_losses, rs_values, rsi_values = calculate_rsi_vectorized(
            close_prices, RSI_PERIOD
        )

        # Calculate price changes with precision control
        price_changes = np.diff(close_prices, prepend=close_prices[0])
        gains = np.where(price_changes > 0, price_changes, 0.0)
        losses = np.where(price_changes < 0, -price_changes, 0.0)

        # Apply precision rounding to intermediate values
        price_changes_rounded = np.array(
            [precise_round(pc, STORAGE_PRECISION["price"]) for pc in price_changes]
        )
        gains_rounded = np.array(
            [precise_round(g, STORAGE_PRECISION["price"]) for g in gains]
        )
        losses_rounded = np.array(
            [precise_round(l, STORAGE_PRECISION["price"]) for l in losses]
        )

        # Efficiently add RSI columns in bulk with controlled precision
        rsi_columns = {
            "price_change": price_changes_rounded,
            "gain": gains_rounded,
            "loss": losses_rounded,
            "avg_gain": avg_gains,  # Already precision-controlled
            "avg_loss": avg_losses,  # Already precision-controlled
            "RS": rs_values,  # Already precision-controlled
            "RSI": rsi_values,  # Already precision-controlled
        }

        for col_name, values in rsi_columns.items():
            df[col_name] = values

        # PRECISION: High-precision EMA calculation with accumulation control
        logger.debug("Calculating EMAs with precision control...")
        ema_data = {}

        for span in EMA_SPANS:
            ema_series = (
                df["close"]
                .astype(CALCULATION_PRECISION)
                .ewm(span=span, adjust=False)
                .mean()
            )
            ema_rounded = np.array(
                [precise_round(ema, STORAGE_PRECISION["ema"]) for ema in ema_series]
            )
            ema_data[f"EMA_{span}"] = ema_rounded

        for col_name, values in ema_data.items():
            df[col_name] = values

        # PRECISION: High-precision MACD calculation
        logger.debug("Calculating MACD with precision control...")
        macd_raw = ema_data["EMA_12"] - ema_data["EMA_26"]
        macd_precise = np.array(
            [precise_round(m, STORAGE_PRECISION["macd"]) for m in macd_raw]
        )

        df["macd"] = macd_precise

        # Signal line calculation with precision control
        signal_series = (
            pd.Series(macd_precise).ewm(span=MACD_SIGNAL_PERIOD, adjust=False).mean()
        )
        signal_precise = np.array(
            [precise_round(s, STORAGE_PRECISION["macd"]) for s in signal_series]
        )

        # MACD histogram with precision control
        histogram_raw = macd_precise - signal_precise
        histogram_precise = np.array(
            [precise_round(h, STORAGE_PRECISION["macd"]) for h in histogram_raw]
        )

        macd_data = {"signal": signal_precise, "macd_histogram": histogram_precise}

        for col_name, values in macd_data.items():
            df[col_name] = values

        logger.info("Technical indicators initialized successfully")
        return df

    except Exception as e:
        logger.error(f"Error initializing technical indicators: {e}")
        return None
