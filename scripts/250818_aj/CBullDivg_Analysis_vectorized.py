# scripts/aj/CBullDivg_Analysis_vectorized.py

import pandas as pd
import numpy as np


def _ensure_date_column(df: pd.DataFrame) -> pd.DataFrame:
    """Sorgt dafür, dass eine Spalte 'date' existiert."""
    if "date" not in df.columns:
        if "timestamp" in df.columns:
            df["date"] = pd.to_datetime(df["timestamp"])
        else:
            raise KeyError("Weder 'date' noch 'timestamp' gefunden!")
    return df


def _validate_inputs(df: pd.DataFrame):
    """Checke auf notwendige Spalten."""
    needed = [
        "close", "macd",
        "LM_High_window_1_CS", "LM_Low_window_1_CS",
        "LM_High_window_2_CS", "LM_Low_window_2_CS",
    ]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise KeyError(f"Fehlende Spalten: {missing}")


def _find_bullish_divergences(df: pd.DataFrame, lookback: int, alpha: float, gamma: float):
    """Finde Bullish Divergenzen (Higher Low im Indikator, Lower Low im Preis)."""
    n = len(df)
    res = pd.DataFrame(index=df.index)
    res["CBullD_gen"] = 0.0
    res["CBullD_Higher_Low_date_gen"] = pd.NaT
    res["CBullD_Lower_Low_date_gen"] = pd.NaT

    lows = df.index[df["LM_Low_window_2_CS"] == 1]
    for i1 in range(len(lows) - 1):
        for i2 in range(i1 + 1, len(lows)):
            idx1, idx2 = lows[i1], lows[i2]
            if (idx2 - idx1) < lookback:
                continue
            # Preis tiefer, MACD höher → bullish divergence
            if df.loc[idx2, "close"] < df.loc[idx1, "close"] and df.loc[idx2, "macd"] > df.loc[idx1, "macd"]:
                if abs(df.loc[idx2, "macd"] - df.loc[idx1, "macd"]) > alpha:
                    res.loc[idx2, "CBullD_gen"] = 1.0
                    res.loc[idx2, "CBullD_Higher_Low_date_gen"] = df.loc[idx1, "date"]
                    res.loc[idx2, "CBullD_Lower_Low_date_gen"] = df.loc[idx2, "date"]
    return res


def _find_bearish_divergences(df: pd.DataFrame, lookback: int, alpha: float, gamma: float):
    """Finde Bearish Divergenzen (Lower High im Indikator, Higher High im Preis)."""
    n = len(df)
    res = pd.DataFrame(index=df.index)
    res["CBearD_gen"] = 0.0
    res["CBearD_Higher_High_date_gen"] = pd.NaT
    res["CBearD_Lower_High_date_gen"] = pd.NaT

    highs = df.index[df["LM_High_window_2_CS"] == 1]
    for i1 in range(len(highs) - 1):
        for i2 in range(i1 + 1, len(highs)):
            idx1, idx2 = highs[i1], highs[i2]
            if (idx2 - idx1) < lookback:
                continue
            # Preis höher, MACD tiefer → bearish divergence
            if df.loc[idx2, "close"] > df.loc[idx1, "close"] and df.loc[idx2, "macd"] < df.loc[idx1, "macd"]:
                if abs(df.loc[idx2, "macd"] - df.loc[idx1, "macd"]) > alpha:
                    res.loc[idx2, "CBearD_gen"] = 1.0
                    res.loc[idx2, "CBearD_Higher_High_date_gen"] = df.loc[idx1, "date"]
                    res.loc[idx2, "CBearD_Lower_High_date_gen"] = df.loc[idx2, "date"]
    return res


def _find_negative_macd_divergences(df: pd.DataFrame, lookback: int, alpha: float):
    """Negative Divergenzen basierend auf MACD."""
    res = pd.DataFrame(index=df.index)
    res["CBullD_neg_MACD"] = 0.0
    res["CBullD_Higher_Low_date_neg_MACD"] = pd.NaT
    res["CBullD_Lower_Low_date_neg_MACD"] = pd.NaT

    lows = df.index[df["LM_Low_window_2_CS"] == 1]
    for i1 in range(len(lows) - 1):
        for i2 in range(i1 + 1, len(lows)):
            idx1, idx2 = lows[i1], lows[i2]
            if (idx2 - idx1) < lookback:
                continue
            if df.loc[idx2, "macd"] > df.loc[idx1, "macd"] and (df.loc[idx2, "macd"] - df.loc[idx1, "macd"]) > alpha:
                res.loc[idx2, "CBullD_neg_MACD"] = 1.0
                res.loc[idx2, "CBullD_Higher_Low_date_neg_MACD"] = df.loc[idx1, "date"]
                res.loc[idx2, "CBullD_Lower_Low_date_neg_MACD"] = df.loc[idx2, "date"]
    return res


def CBullDivg_analysis(ohlc: pd.DataFrame, lookback: int, alpha: float, gamma: float) -> pd.DataFrame:
    """
    Führt die Analyse für Bullish & Bearish Divergenzen durch.
    Erwartet LM_* Spalten und MACD in ohlc.
    """
    ohlc = _ensure_date_column(ohlc.copy())
    _validate_inputs(ohlc)

    # numerische Spalten casten
    for col in ["close", "macd"]:
        ohlc[col] = pd.to_numeric(ohlc[col], errors="coerce")

    # Ergebnisse sammeln
    res_bull = _find_bullish_divergences(ohlc, lookback, alpha, gamma)
    res_bear = _find_bearish_divergences(ohlc, lookback, alpha, gamma)
    res_neg = _find_negative_macd_divergences(ohlc, lookback, alpha)

    res = pd.concat([res_bull, res_bear, res_neg], axis=1)

    return res
