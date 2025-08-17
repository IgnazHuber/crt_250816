import numpy as np
import pandas as pd

def make_synthetic_series(n=220, seed=1):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")
    base = np.linspace(100, 80, n) + rng.normal(0, 0.2, n)  # leichter Abwärtstrend
    low = base - 0.5 + rng.normal(0, 0.1, n)
    high = base + 0.5 + rng.normal(0, 0.1, n)
    open_ = base + rng.normal(0, 0.05, n)
    close = base + rng.normal(0, 0.05, n)
    vol = np.full(n, 1000.0)

    # Erzeuge 3 bullishe Divergenzen garantiert (Preis LL, RSI HL)
    rsi = np.clip(50 + rng.normal(0, 2, n), 20, 80)
    pivots = [60, 120, 180]
    for p in pivots:
        low[p-3]  = low[p-3] + 0.2
        low[p]    = low[p-3] - 0.3  # Lower Low
        rsi[p-3]  = 30.0
        rsi[p]    = 40.0            # Higher Low
    # MACD: HL an denselben Stellen (weniger negativ)
    macd = np.clip(-1.5 + 0.002*np.arange(n), -2.0, 1.0)
    for p in pivots:
        macd[p-3] = -1.2
        macd[p]   = -0.8

    signal = pd.Series(macd).rolling(9, min_periods=1).mean().to_numpy()
    hist = macd - signal

    df = pd.DataFrame({
        "timestamp": ts, "open": open_, "high": high, "low": low, "close": close, "volume": vol,
        "RSI": rsi, "macd": macd, "signal": signal, "macd_histogram": hist
    })
    return df, pivots
