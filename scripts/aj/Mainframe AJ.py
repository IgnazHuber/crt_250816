import finplot as fplt
import pandas as pd

# Behalte die Original-Module bei (keine Features entfernen!)
from CBullDivg_Analysis_vectorized import CBullDivg_analysis
from Initialize_RSI_EMA_MACD import Initialize_RSI_EMA_MACD
from Local_Maximas_Minimas import Local_Max_Min


# ------------------------------------------------------------
# Eingabedatei (unverändert)
csv_file_path = r"C:\Projekte\crt_250816\data\raw\btc_1day_candlesticks_all.csv"

# CSV laden (unverändert)
df = pd.read_csv(csv_file_path, low_memory=False)


# ------------------------------------------------------------
# Utilities: robuste Spalten-Finder (case-insensitiv, Aliasse)
def _find_col(df_, candidates):
    """Finde die erste existierende Spalte (case-insensitiv) aus candidates und gib den echten Namen zurück."""
    lower2orig = {c.lower(): c for c in df_.columns}
    for cand in candidates:
        if cand.lower() in lower2orig:
            return lower2orig[cand.lower()]
    return None


def _ensure_date_column(df_in: pd.DataFrame) -> pd.DataFrame:
    df_local = df_in
    # Falls schon vorhanden:
    if "date" in df_local.columns:
        try:
            df_local["date"] = pd.to_datetime(df_local["date"], errors="coerce", format="mixed")
        except TypeError:
            df_local["date"] = pd.to_datetime(df_local["date"], errors="coerce", infer_datetime_format=True)
        return df_local

    # Kandidaten
    src_col = _find_col(df_local, ["timestamp", "time", "datetime", "open_time", "time_open", "date"])
    if src_col is None:
        raise KeyError(
            "Keine Zeitspalte gefunden. Erwartet eine von: date / timestamp / time / datetime / open_time / time_open"
        )

    s = df_local[src_col]
    if pd.api.types.is_integer_dtype(s) or pd.api.types.is_float_dtype(s):
        ser = pd.to_numeric(s, errors="coerce")
        mx = pd.to_numeric(ser, errors="coerce").max()
        unit = "ms" if pd.notna(mx) and mx > 1e12 else "s"
        df_local["date"] = pd.to_datetime(ser, unit=unit, errors="coerce")
        if df_local["date"].isna().all():
            unit_alt = "s" if unit == "ms" else "ms"
            df_local["date"] = pd.to_datetime(ser, unit=unit_alt, errors="coerce")
    else:
        try:
            df_local["date"] = pd.to_datetime(s, errors="coerce", format="mixed")
        except TypeError:
            df_local["date"] = pd.to_datetime(s, errors="coerce", infer_datetime_format=True)

    if df_local["date"].isna().all():
        raise ValueError(f"Zeitspalte '{src_col}' konnte nicht nach datetime konvertiert werden.")
    return df_local


def _ensure_ohlc_aliases(df_in: pd.DataFrame) -> pd.DataFrame:
    """Stelle sicher, dass Spalten open/high/low/close unter genau diesen Namen vorhanden sind."""
    df_local = df_in
    # Open
    oc = _find_col(df_local, ["open", "o", "open_price"])
    if oc is None:
        raise KeyError("Keine 'open'-Spalte gefunden (erwartet: open / o / open_price).")
    if oc != "open":
        df_local["open"] = df_local[oc]
    # High
    hc = _find_col(df_local, ["high", "h", "high_price"])
    if hc is None:
        raise KeyError("Keine 'high'-Spalte gefunden (erwartet: high / h / high_price).")
    if hc != "high":
        df_local["high"] = df_local[hc]
    # Low
    lc = _find_col(df_local, ["low", "l", "low_price"])
    if lc is None:
        raise KeyError("Keine 'low'-Spalte gefunden (erwartet: low / l / low_price).")
    if lc != "low":
        df_local["low"] = df_local[lc]
    # Close (inkl. Adj Close)
    cc = _find_col(df_local, ["close", "c", "close_price", "adj close", "adj_close", "adjusted_close"])
    if cc is None:
        raise KeyError("Keine 'close'-Spalte gefunden (erwartet: close / c / close_price / adj close / adj_close).")
    if cc != "close":
        df_local["close"] = df_local[cc]
    return df_local


def _ensure_rsi_alias(df_in: pd.DataFrame) -> pd.DataFrame:
    """Sorge dafür, dass 'RSI' existiert (Alias z. B. rsi, rsi_14, RSI_14)."""
    df_local = df_in
    if "RSI" not in df_local.columns:
        rc = _find_col(df_local, ["RSI", "rsi", "rsi_14", "RSI_14"])
        if rc is not None and rc != "RSI":
            df_local["RSI"] = df_local[rc]
    return df_local


def _ensure_ema_aliases(df_in: pd.DataFrame) -> pd.DataFrame:
    """Sorge für die erwarteten EMA-Spaltennamen EMA_20/50/100/200, falls sie unter anderen Aliassen existieren."""
    df_local = df_in
    for n in (20, 50, 100, 200):
        target = f"EMA_{n}"
        if target not in df_local.columns:
            # mögliche Aliasse: ema20, EMA20, ema_20
            cand = _find_col(df_local, [target, f"ema{n}", f"EMA{n}", f"ema_{n}", f"Ema_{n}"])
            if cand is not None and cand != target:
                df_local[target] = df_local[cand]
    return df_local


def _ensure_macd_hist(df_in: pd.DataFrame) -> pd.DataFrame:
    """Stelle sicher, dass 'macd_histogram' existiert (Aliasse oder rekonstruiert aus MACD & Signal)."""
    df_local = df_in
    if "macd_histogram" in df_local.columns:
        return df_local

    # häufige Aliasse
    alias = _find_col(df_local, [
        "macd_histogram", "macd_hist", "macd_diff", "macd_hist_12_26_9",
        "MACD_Histogram", "MACD_hist", "histogram"
    ])
    if alias is not None:
        if alias != "macd_histogram":
            df_local["macd_histogram"] = df_local[alias]
        return df_local

    # Rekonstruktion versuchen: macd - signal
    macd_col = _find_col(df_local, ["macd", "MACD", "macd_line", "MACD_line"])
    sig_col  = _find_col(df_local, ["macd_signal", "signal", "MACD_signal", "signal_line"])
    if macd_col is not None and sig_col is not None:
        df_local["macd_histogram"] = df_local[macd_col] - df_local[sig_col]
        return df_local

    # Wenn alles fehlt: lege neutrale Nullen an (besser als Crash; Marker/Segmente bleiben erhalten)
    df_local["macd_histogram"] = 0.0
    return df_local


# ------------------------------------------------------------
# Datum & Spalten-Aliasse sicherstellen (nur Ergänzungen, nichts entfernt)
df = _ensure_date_column(df)
df = _ensure_ohlc_aliases(df)
# Indikatoren (können weitere Spalten hinzufügen)
Initialize_RSI_EMA_MACD(df)
Local_Max_Min(df)
# danach ggf. RSI/EMA Aliasse vereinheitlichen
df = _ensure_rsi_alias(df)
df = _ensure_ema_aliases(df)
df = _ensure_macd_hist(df)

# Divergenzen berechnen -> Ergebnis zurück in df (wichtig!)
df = CBullDivg_analysis(df, 5, 0.1, 3.25)


# ------------------------------------------------------------
# Plot-Styling (unverändert)
fplt.background = fplt.odd_plot_background = "#242320"  # Adjust Plot Background colour
fplt.cross_hair_color = "#eefa"  # Adjust Crosshair colour

# Plot-Setup (unverändert)
ax1, ax2, ax3 = fplt.create_plot("Chart", rows=3)

# Candles (unverändert – inkl. macd_histogram in der Auswahl)
candles = df[["date", "open", "close", "high", "low", "macd_histogram"]]
fplt.candlestick_ochl(candles, ax=ax1)

# RSI (unverändert)
fplt.plot(df.RSI, color="#000000", width=2, ax=ax2, legend="RSI")
fplt.set_y_range(0, 100, ax=ax2)
fplt.add_horizontal_band(0, 1, color="#000000", ax=ax2)    # Dummy band
fplt.add_horizontal_band(99, 100, color="#000000", ax=ax2) # Dummy band

# MACD-Histogramm (unverändert)
fplt.volume_ocv(
    df[["date", "open", "close", "macd_histogram"]],
    ax=ax3,
    colorfunc=fplt.strength_colorfilter,
)

# EMAs (unverändert: nur geplottet, wenn vorhanden)
if "EMA_20" in df.columns:  df.EMA_20.plot(ax=ax1, legend="20-EMA")
if "EMA_50" in df.columns:  df.EMA_50.plot(ax=ax1, legend="50-EMA")
if "EMA_100" in df.columns: df.EMA_100.plot(ax=ax1, legend="100-EMA")
if "EMA_200" in df.columns: df.EMA_200.plot(ax=ax1, legend="200-EMA")


# ------------------------------------------------------------
# Hilfsfunktion: Verbindungssegment zwischen zwei Punkten (Zusatz; nichts ersetzt)
def _plot_segment(ax, x0, y0, x1, y1, color):
    s = pd.Series([y0, y1], index=pd.to_datetime([x0, x1]))
    fplt.plot(s, ax=ax, color=color, width=1)


# Divergenz-Markierungen (unverändert + optionale Verbindungssegmente)
if "CBullD_gen" in df.columns:
    for i in range(2, len(df)):
        if df["CBullD_gen"].iloc[i] == 1:
            # Preis x-Marker
            fplt.plot(pd.to_datetime(df["CBullD_Lower_Low_date_gen"].iloc[i]),
                      df["CBullD_Lower_Low_gen"].iloc[i],
                      style="x", ax=ax1, color="red")
            fplt.plot(pd.to_datetime(df["CBullD_Higher_Low_date_gen"].iloc[i]),
                      df["CBullD_Higher_Low_gen"].iloc[i],
                      style="x", ax=ax1, color="blue")
            # RSI x-Marker
            fplt.plot(pd.to_datetime(df["CBullD_Lower_Low_date_gen"].iloc[i]),
                      df["CBullD_Lower_Low_RSI_gen"].iloc[i],
                      style="x", ax=ax2, color="red")
            fplt.plot(pd.to_datetime(df["CBullD_Higher_Low_date_gen"].iloc[i]),
                      df["CBullD_Higher_Low_RSI_gen"].iloc[i],
                      style="x", ax=ax2, color="blue")
            # MACD x-Marker
            fplt.plot(pd.to_datetime(df["CBullD_Lower_Low_date_gen"].iloc[i]),
                      df["CBullD_Lower_Low_MACD_gen"].iloc[i],
                      style="x", ax=ax3, color="red")
            fplt.plot(pd.to_datetime(df["CBullD_Higher_Low_date_gen"].iloc[i]),
                      df["CBullD_Higher_Low_MACD_gen"].iloc[i],
                      style="x", ax=ax3, color="blue")
            # optionale Segmente
            try:
                x0 = df["CBullD_Lower_Low_date_gen"].iloc[i]
                x1 = df["CBullD_Higher_Low_date_gen"].iloc[i]
                _plot_segment(ax1, x0, df["CBullD_Lower_Low_gen"].iloc[i],
                                   x1, df["CBullD_Higher_Low_gen"].iloc[i], color="blue")
                _plot_segment(ax2, x0, df["CBullD_Lower_Low_RSI_gen"].iloc[i],
                                   x1, df["CBullD_Higher_Low_RSI_gen"].iloc[i], color="blue")
                _plot_segment(ax3, x0, df["CBullD_Lower_Low_MACD_gen"].iloc[i],
                                   x1, df["CBullD_Higher_Low_MACD_gen"].iloc[i], color="blue")
            except Exception:
                pass

if "CBullD_neg_MACD" in df.columns:
    for i in range(2, len(df)):
        if df["CBullD_neg_MACD"].iloc[i] == 1:
            # Preis x-Marker
            fplt.plot(pd.to_datetime(df["CBullD_Lower_Low_date_neg_MACD"].iloc[i]),
                      df["CBullD_Lower_Low_neg_MACD"].iloc[i],
                      style="x", ax=ax1, color="red")
            fplt.plot(pd.to_datetime(df["CBullD_Higher_Low_date_neg_MACD"].iloc[i]),
                      df["CBullD_Higher_Low_neg_MACD"].iloc[i],
                      style="x", ax=ax1, color="blue")
            # RSI x-Marker
            fplt.plot(pd.to_datetime(df["CBullD_Lower_Low_date_neg_MACD"].iloc[i]),
                      df["CBullD_Lower_Low_RSI_neg_MACD"].iloc[i],
                      style="x", ax=ax2, color="red")
            fplt.plot(pd.to_datetime(df["CBullD_Higher_Low_date_neg_MACD"].iloc[i]),
                      df["CBullD_Higher_Low_RSI_neg_MACD"].iloc[i],
                      style="x", ax=ax2, color="blue")
            # MACD x-Marker
            fplt.plot(pd.to_datetime(df["CBullD_Lower_Low_date_neg_MACD"].iloc[i]),
                      df["CBullD_Lower_Low_MACD_neg_MACD"].iloc[i],
                      style="x", ax=ax3, color="red")
            fplt.plot(pd.to_datetime(df["CBullD_Higher_Low_date_neg_MACD"].iloc[i]),
                      df["CBullD_Higher_Low_MACD_neg_MACD"].iloc[i],
                      style="x", ax=ax3, color="blue")
            # optionale Segmente
            try:
                x0 = df["CBullD_Lower_Low_date_neg_MACD"].iloc[i]
                x1 = df["CBullD_Higher_Low_date_neg_MACD"].iloc[i]
                _plot_segment(ax1, x0, df["CBullD_Lower_Low_neg_MACD"].iloc[i],
                                   x1, df["CBullD_Higher_Low_neg_MACD"].iloc[i], color="blue")
                _plot_segment(ax2, x0, df["CBullD_Lower_Low_RSI_neg_MACD"].iloc[i],
                                   x1, df["CBullD_Higher_Low_RSI_neg_MACD"].iloc[i], color="blue")
                _plot_segment(ax3, x0, df["CBullD_Lower_Low_MACD_neg_MACD"].iloc[i],
                                   x1, df["CBullD_Higher_Low_MACD_neg_MACD"].iloc[i], color="blue")
            except Exception:
                pass

# ------------------------------------------------------------
fplt.show()
