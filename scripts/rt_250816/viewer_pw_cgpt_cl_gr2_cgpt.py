# viewer_plotly_white_optimized_polars.py
# Hochoptimierte Version mit:
# - Polars für schnelle Datenverarbeitung
# - Lazy Evaluation für effiziente Feature-Berechnung
# - Vectorized operations für Geschwindigkeit
# - Optimierte Button-Darstellung (X/□ Symbole, keine Überlappung)
# - Vollständige X-Achsen-Synchronisation
# - Kein Überlappen der Subplots
# - Fehlerbehandlung für Polars-kompatible Module

import json
import os
import sys
from tkinter import Tk, filedialog

import numpy as np
import pandas as pd  # Nur für Plotly-Kompatibilität und Legacy-Module
import plotly.graph_objects as go
import polars as pl
from extrema_local import detect_local_extrema
from plotly.subplots import make_subplots

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

# Bestehende Module
from Initialize_RSI_EMA_MACD import Initialize_RSI_EMA_MACD

try:
    from Local_Maximas_Minimas_fast import Local_Max_Min
except ImportError:
    from Local_Maximas_Minimas import Local_Max_Min

from CBullDivg_Analysis_vectorized import CBullDivg_analysis
from patterns_candles import (build_pattern_boxes, build_pattern_traces,
                              detect_patterns)
from plot_divergences import (build_rsi_divergence_traces, detect_divergences,
                              div_shapes_from_table, rsi_traces_from_table)

DATA_DIR_DEFAULT = os.path.join(THIS_DIR, "data")
STATE_FILE = os.path.join(THIS_DIR, ".last_dir.json")


# ---------------- Performance-Helpers ----------------
def _load_last_dir(default_dir: str) -> str:
    try:
        if os.path.isfile(STATE_FILE):
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                j = json.load(f)
                d = j.get("last_dir")
                if d and os.path.isdir(d):
                    return d
    except Exception:
        pass
    return default_dir


def _save_last_dir(d: str):
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump({"last_dir": d}, f)
    except Exception:
        pass


def select_files() -> list:
    root = Tk()
    root.withdraw()
    init_dir = _load_last_dir(DATA_DIR_DEFAULT)
    files = filedialog.askopenfilenames(
        title="Datei(en) öffnen (CSV oder Parquet)",
        initialdir=init_dir,
        filetypes=[("CSV/Parquet", "*.csv *.parquet"), ("Alle Dateien", "*.*")],
    )
    root.destroy()
    files = list(files)
    if files:
        _save_last_dir(os.path.dirname(files[0]))
    return files


def _ann_to_dict_list(anns):
    if not anns:
        return []
    out = []
    for a in anns:
        if isinstance(a, dict):
            out.append(a)
        else:
            try:
                out.append(
                    a.to_plotly_json() if hasattr(a, "to_plotly_json") else dict(a)
                )
            except Exception:
                pass
    return out


def _median_step_vectorized(dts: pl.Series) -> pd.Timedelta:
    if len(dts) < 2:
        return pd.Timedelta(minutes=1)
    diffs = np.diff(np.sort(dts.to_numpy())).astype("timedelta64[ns]")
    diffs = diffs[diffs > np.timedelta64(0)]
    if len(diffs) == 0:
        return pd.Timedelta(minutes=1)
    med = np.median(diffs)
    return pd.Timedelta(med)


def _ensure_full_features_optimized(df: pl.DataFrame) -> pl.DataFrame:
    required_features = {
        "rsi_ema_macd": [
            "RSI",
            "macd",
            "macd_signal",
            "macd_histogram",
            "EMA_20",
            "EMA_50",
            "EMA_100",
            "EMA_200",
        ],
        "extrema": [
            "LM_Low_window_1_CS",
            "LM_Low_window_2_CS",
            "LM_Low_window_1_MACD",
            "LM_Low_window_2_MACD",
        ],
        "bullish_div": ["CBullD_gen", "CBullD_neg_MACD"],
        "patterns": [
            "bullish_engulfing",
            "bearish_engulfing",
            "hammer",
            "shooting_star",
            "dragonfly_doji",
            "gravestone_doji",
            "morning_star",
            "evening_star",
            "tweezer_top",
            "tweezer_bottom",
        ],
        "local_extrema": ["local_max", "local_min"],
    }

    missing_features = []
    df_cols = df.columns

    if not all(c in df_cols for c in required_features["rsi_ema_macd"]):
        missing_features.append("rsi_ema_macd")

    if not all(c in df_cols for c in required_features["extrema"]):
        missing_features.append("extrema")

    if not all(c in df_cols for c in required_features["bullish_div"]):
        missing_features.append("bullish_div")

    if not any(c in df_cols for c in required_features["patterns"]):
        missing_features.append("patterns")

    if not all(c in df_cols for c in required_features["local_extrema"]):
        missing_features.append("local_extrema")

    # LazyFrame für Berechnungen
    lazy_df = df.lazy()

    if "rsi_ema_macd" in missing_features:
        try:
            # Fallback: Konvertiere zu Pandas, falls Initialize_RSI_EMA_MACD nicht Polars-kompatibel ist
            pandas_df = lazy_df.collect().to_pandas()
            pandas_df = Initialize_RSI_EMA_MACD(pandas_df)
            if pandas_df is None or pandas_df.empty:
                raise ValueError(
                    "Initialize_RSI_EMA_MACD returned None or empty DataFrame"
                )
            lazy_df = pl.from_pandas(pandas_df).lazy()
        except Exception as e:
            print(f"[ERROR] Fehler bei RSI/EMA/MACD-Berechnung: {e}")
            # Fallback: Manuelle MACD-Berechnung
            lazy_df = (
                lazy_df.with_columns(
                    [
                        pl.col("close").ewm_mean(span=12, adjust=False).alias("ema12"),
                        pl.col("close").ewm_mean(span=26, adjust=False).alias("ema26"),
                    ]
                )
                .with_columns(
                    [
                        (pl.col("ema12") - pl.col("ema26")).alias("macd"),
                        pl.col("macd")
                        .ewm_mean(span=9, adjust=False)
                        .alias("macd_signal"),
                    ]
                )
                .with_columns(
                    [(pl.col("macd") - pl.col("macd_signal")).alias("macd_histogram")]
                )
                .with_columns(
                    [
                        pl.col("close")
                        .pct_change()
                        .rolling_apply(
                            lambda x: 100 * (1 - x[-1] / x[0]) if len(x) > 0 else 0,
                            window_size=14,
                        )
                        .alias("RSI"),
                        pl.col("close").ewm_mean(span=20, adjust=False).alias("EMA_20"),
                        pl.col("close").ewm_mean(span=50, adjust=False).alias("EMA_50"),
                        pl.col("close")
                        .ewm_mean(span=100, adjust=False)
                        .alias("EMA_100"),
                        pl.col("close")
                        .ewm_mean(span=200, adjust=False)
                        .alias("EMA_200"),
                    ]
                )
            )

    if "extrema" in missing_features:
        try:
            lazy_df = Local_Max_Min(lazy_df)
        except Exception as e:
            print(f"[ERROR] Fehler bei Extrema-Berechnung: {e}")
            # Fallback: Skippen, um Plot fortzusetzen
            pass

    if "bullish_div" in missing_features:
        try:
            lazy_df = CBullDivg_analysis(
                lazy_df, window=5, Candle_Tol=0.1, MACD_tol=3.25
            )
        except Exception as e:
            print(f"[ERROR] Fehler bei Divergenz-Analyse: {e}")
            pass

    if "patterns" in missing_features:
        try:
            patterns_df = detect_patterns(lazy_df.collect().to_pandas())
            patterns_df = pl.from_pandas(patterns_df)
            lazy_df = lazy_df.join(patterns_df.lazy(), on="date", how="left")
        except Exception as e:
            print(f"[ERROR] Fehler bei Pattern-Erkennung: {e}")
            pass

    if "local_extrema" in missing_features:
        try:
            # Konvertiere LazyFrame zu DataFrame für detect_local_extrema
            df_temp = lazy_df.collect()
            df_temp = detect_local_extrema(
                df_temp,
                price_col="close",
                order=5,
                display_threshold=5,
                price_mode="oc",
            )
            lazy_df = df_temp.lazy()
        except Exception as e:
            print(f"[ERROR] Fehler bei lokalen Extrema: {e}")
            pass

    df = lazy_df.collect()

    if "RSI" in df.columns:
        df = df.with_columns(pl.col("RSI").cast(pl.Float64).clip(0, 100))

    if (
        "macd" in df.columns
        and "macd_histogram" in df.columns
        and "macd_signal" not in df.columns
    ):
        df = df.with_columns(
            pl.col("macd").ewm_mean(span=9, adjust=False).alias("macd_signal")
        )

    if df.is_empty():
        raise ValueError("DataFrame ist leer nach Feature-Berechnung")

    return df


# ---------------- Optimized Divergence Markers ----------------
def _add_divergence_markers_vectorized(
    fig, df: pl.DataFrame, sel_mask, kind_label, row_map
):
    if not sel_mask.any():
        return

    ids_idx = np.where(sel_mask.to_numpy())[0]
    ids = np.arange(1, len(ids_idx) + 1)
    step = _median_step_vectorized(df.filter(sel_mask)["date"])
    x_shift_low = -0.15 * step
    x_shift_high = 0.15 * step

    def add_pair_vectorized(
        row, col, xLd, yL, xHd, yH, label_low, label_high, y_offset_frac=0.004
    ):
        mask_data = df.filter(sel_mask)
        x_low = mask_data[xLd].cast(pl.Datetime) + x_shift_low
        x_high = mask_data[xHd].cast(pl.Datetime) + x_shift_high
        y_low = mask_data[yL].cast(pl.Float64).to_numpy()
        y_high = mask_data[yH].cast(pl.Float64).to_numpy()

        y_range = np.concatenate([y_low, y_high])
        span = np.nanmax(y_range) - np.nanmin(y_range) if len(y_range) > 0 else 1.0
        span = span if span > 0 else 1.0

        y_low_lbl = y_low - y_offset_frac * span
        y_high_lbl = y_high + y_offset_frac * span

        common_data = np.column_stack([ids, np.full(len(ids), kind_label)])
        id_strings = [str(i) for i in ids]

        fig.add_trace(
            go.Scatter(
                x=x_low.to_pandas(),
                y=y_low_lbl,
                mode="markers+text",
                text=id_strings,
                textposition="bottom center",
                textfont=dict(size=10),
                marker_symbol="x",
                marker_size=10,
                marker_color="red",
                customdata=common_data,
                hovertemplate=(
                    f"ID %{{customdata[0]}} · {label_low}<br>"
                    f"Typ: %{{customdata[1]}}<br>"
                    "Datum: %{x|%Y-%m-%d %H:%M}<br>"
                    "Wert: %{y:.6f}<extra></extra>"
                ),
                name=label_low,
                showlegend=False,
            ),
            row=row,
            col=col,
        )

        fig.add_trace(
            go.Scatter(
                x=x_high.to_pandas(),
                y=y_high_lbl,
                mode="markers+text",
                text=id_strings,
                textposition="top center",
                textfont=dict(size=10),
                marker_symbol="x",
                marker_size=10,
                marker_color="blue",
                customdata=common_data,
                hovertemplate=(
                    f"ID %{{customdata[0]}} · {label_high}<br>"
                    f"Typ: %{{customdata[1]}}<br>"
                    "Datum: %{x|%Y-%m-%d %H:%M}<br>"
                    "Wert: %{y:.6f}<extra></extra>"
                ),
                name=label_high,
                showlegend=False,
            ),
            row=row,
            col=col,
        )

    for key in ["price", "rsi", "macd"]:
        if key in row_map:
            r, c, yL, yH, xLd, xHd = row_map[key]
            labels = {
                "price": ("Preis Low (x)", "Preis High (x)"),
                "rsi": ("RSI Low (x)", "RSI High (x)"),
                "macd": ("MACD Low (x)", "MACD High (x)"),
            }
            add_pair_vectorized(r, c, xLd, yL, xHd, yH, labels[key][0], labels[key][1])


# ---------------- Optimized Plot Function ----------------
def _plot_one_file_optimized(path: str):
    if path.endswith(".csv"):
        precomp_paths = [
            os.path.splitext(path)[0] + ".precomp.enriched.parquet",
            os.path.splitext(path)[0] + ".precomp.parquet",
        ]
        for p in precomp_paths:
            if os.path.isfile(p):
                path = p
                break

    if path.endswith(".parquet"):
        df = pl.read_parquet(path)
    else:
        df = pl.read_csv(path, infer_schema_length=10000)

    date_cols = ["date", "timestamp"]
    for col in date_cols:
        if col in df.columns:
            df = df.with_columns(pl.col(col).cast(pl.Datetime).alias("date"))
            break
    else:
        df = df.with_columns(pl.col(df.columns[0]).cast(pl.Datetime).alias("date"))

    try:
        df = _ensure_full_features_optimized(df)
    except Exception as e:
        print(f"[ERROR] Fehler bei Feature-Berechnung: {e}")
        return

    if df.is_empty():
        print("[ERROR] DataFrame ist leer nach Feature-Berechnung")
        return

    t = df["date"]
    n_points = len(df)

    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.55, 0.15, 0.15, 0.15],
        specs=[[{"type": "xy"}], [{"type": "xy"}], [{"type": "xy"}], [{"type": "xy"}]],
    )

    sample_every = max(1, n_points // 1000)
    hover_indices = slice(None, None, sample_every)

    # Hover-Text mit Polars-kompatibler Datumsformatierung und numerischer Handhabung
    date_series = df["date"][hover_indices].cast(pl.Datetime)
    date_str = date_series.dt.strftime("%Y-%m-%d")
    open_vals = (
        df["open"][hover_indices]
        .cast(pl.Float64, strict=False)
        .fill_null(np.nan)
        .to_numpy()
    )
    high_vals = (
        df["high"][hover_indices]
        .cast(pl.Float64, strict=False)
        .fill_null(np.nan)
        .to_numpy()
    )
    low_vals = (
        df["low"][hover_indices]
        .cast(pl.Float64, strict=False)
        .fill_null(np.nan)
        .to_numpy()
    )
    close_vals = (
        df["close"][hover_indices]
        .cast(pl.Float64, strict=False)
        .fill_null(np.nan)
        .to_numpy()
    )

    # Formatierung mit separater Behandlung von NaN
    ht = [
        (
            f"Datum: {d}<br>Open: {o:.6f}<br>High: {h:.6f}<br>Low: {l:.6f}<br>Close: {c:.6f}"
            if not (np.isnan(o) or np.isnan(h) or np.isnan(l) or np.isnan(c))
            else f"Datum: {d}<br>Open: N/A<br>High: N/A<br>Low: N/A<br>Close: N/A"
        )
        for d, o, h, l, c in zip(date_str, open_vals, high_vals, low_vals, close_vals)
    ]

    fig.add_trace(
        go.Candlestick(
            x=t.to_pandas(),
            open=df["open"].to_numpy(),
            high=df["high"].to_numpy(),
            low=df["low"].to_numpy(),
            close=df["close"].to_numpy(),
            increasing=dict(line=dict(color="green", width=1.2), fillcolor="green"),
            decreasing=dict(line=dict(color="red", width=1.2), fillcolor="red"),
            name="OHLC",
            opacity=0.95,
            hovertext=ht if len(ht) > 1000 else ht,
            hoverinfo="text",
        ),
        row=1,
        col=1,
    )

    extrema_start = len(fig.data)

    for extrema_type, config in [
        (
            "local_max",
            {
                "id_col": "max_id",
                "strength_col": "max_strength",
                "symbol": "diamond",
                "position": "top center",
                "prefix": "ma",
                "name": "Lokales Maximum",
            },
        ),
        (
            "local_min",
            {
                "id_col": "min_id",
                "strength_col": "min_strength",
                "symbol": "diamond-open",
                "position": "bottom center",
                "prefix": "mi",
                "name": "Lokales Minimum",
            },
        ),
    ]:
        if extrema_type in df.columns and df[extrema_type].any():
            mask = df[extrema_type]
            extrema_data = df.filter(mask)

            ids = extrema_data[config["id_col"]].cast(pl.Int32)
            strengths = extrema_data[config["strength_col"]].cast(pl.Int32)
            texts = [f"{config['prefix']}{i:03d}/{s}" for i, s in zip(ids, strengths)]

            fig.add_trace(
                go.Scatter(
                    x=extrema_data["date"].to_pandas(),
                    y=extrema_data["close"].to_numpy(),
                    mode="markers+text",
                    marker_symbol=config["symbol"],
                    marker_size=10,
                    marker_color="blue",
                    marker_line_width=2 if "open" in config["symbol"] else 0,
                    text=texts,
                    textposition=config["position"],
                    name=config["name"],
                    hovertemplate=f"{config['name']}<br>ID: %{{text}}<br>Datum: %{{x|%Y-%m-%d %H:%M}}<br>Kurs: %{{y:.6f}}<extra></extra>",
                ),
                row=1,
                col=1,
            )

    extrema_trace_indices = list(range(extrema_start, len(fig.data)))

    ema_configs = [
        ("EMA_20", "EMA 20", dict(width=1.5, color="blue")),
        ("EMA_50", "EMA 50", dict(width=1.5, color="orange")),
        ("EMA_100", "EMA 100", dict(width=1.5, color="purple")),
        ("EMA_200", "EMA 200", dict(width=1.5, color="brown")),
    ]

    for ema, name, line_config in ema_configs:
        if ema in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=t.to_pandas(),
                    y=df[ema].to_numpy(),
                    mode="lines",
                    name=name,
                    line=line_config,
                    hovertemplate=f"{name}: %{{y:.6f}}<extra></extra>",
                ),
                row=1,
                col=1,
            )

    if "macd_histogram" in df.columns:
        colors = [
            "green" if x >= 0 else "red" for x in df["macd_histogram"].fill_null(0)
        ]
        fig.add_trace(
            go.Bar(
                x=t.to_pandas(),
                y=df["macd_histogram"].to_numpy(),
                name="MACD Hist",
                marker_color=colors,
                hovertemplate="MACD Hist: %{y:.6f}<extra></extra>",
            ),
            row=2,
            col=1,
        )

    if "macd" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=t.to_pandas(),
                y=df["macd"].to_numpy(),
                mode="lines",
                name="MACD Line",
                line=dict(width=1.5, color="blue"),
                hovertemplate="MACD: %{y:.6f}<extra></extra>",
            ),
            row=2,
            col=1,
        )

    if "macd_signal" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=t.to_pandas(),
                y=df["macd_signal"].to_numpy(),
                mode="lines",
                name="MACD Signal",
                line=dict(width=1.5, color="orange"),
                hovertemplate="Signal: %{y:.6f}<extra></extra>",
            ),
            row=2,
            col=1,
        )

    if "RSI" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=t.to_pandas(),
                y=df["RSI"].to_numpy(),
                mode="lines",
                name="RSI",
                line=dict(width=2, color="purple"),
                hovertemplate="RSI: %{y:.2f}<extra></extra>",
            ),
            row=3,
            col=1,
        )

        for level, name in [(30, "RSI 30"), (70, "RSI 70")]:
            fig.add_trace(
                go.Scatter(
                    x=[t[0].to_pandas(), t[-1].to_pandas()],
                    y=[level, level],
                    mode="lines",
                    name=name,
                    line=dict(width=1, dash="dot", color="gray"),
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=3,
                col=1,
            )

    pattern_start = len(fig.data)
    pats_cols = [
        "bullish_engulfing",
        "bearish_engulfing",
        "hammer",
        "shooting_star",
        "dragonfly_doji",
        "gravestone_doji",
        "morning_star",
        "evening_star",
        "tweezer_top",
        "tweezer_bottom",
    ]

    formation_shapes, formation_traces, formation_annotations = [], [], []

    if any(col in df.columns for col in pats_cols):
        pattern_mask = df[pats_cols].eq(1).any(axis=1)
        if pattern_mask.any():
            shapes_boxes, ann_boxes = build_pattern_boxes(
                df.to_pandas(), df[pats_cols].to_pandas(), opacity=0.18
            )
            traces_poly, ann_poly = build_pattern_traces(
                df.to_pandas(), df[pats_cols].to_pandas(), opacity=0.10
            )

            formation_shapes = list(shapes_boxes)
            formation_traces = list(traces_poly)
            formation_annotations = list(ann_boxes) + list(ann_poly)

            for tr in formation_traces:
                fig.add_trace(tr, row=1, col=1)

    pattern_end = len(fig.data)
    pattern_trace_indices = list(range(pattern_start, pattern_end))

    if "CBullD_gen" in df.columns and "CBullD_neg_MACD" in df.columns:
        row_maps = {
            "gen": {
                "price": (
                    1,
                    1,
                    "CBullD_Lower_Low_gen",
                    "CBullD_Higher_Low_gen",
                    "CBullD_Lower_Low_date_gen",
                    "CBullD_Higher_Low_date_gen",
                ),
                "rsi": (
                    3,
                    1,
                    "CBullD_Lower_Low_RSI_gen",
                    "CBullD_Higher_Low_RSI_gen",
                    "CBullD_Lower_Low_date_gen",
                    "CBullD_Higher_Low_date_gen",
                ),
                "macd": (
                    2,
                    1,
                    "CBullD_Lower_Low_MACD_gen",
                    "CBullD_Higher_Low_MACD_gen",
                    "CBullD_Lower_Low_date_gen",
                    "CBullD_Higher_Low_date_gen",
                ),
            },
            "neg": {
                "price": (
                    1,
                    1,
                    "CBullD_Lower_Low_neg_MACD",
                    "CBullD_Higher_Low_neg_MACD",
                    "CBullD_Lower_Low_date_neg_MACD",
                    "CBullD_Higher_Low_date_neg_MACD",
                ),
                "rsi": (
                    3,
                    1,
                    "CBullD_Lower_Low_RSI_neg_MACD",
                    "CBullD_Higher_Low_RSI_neg_MACD",
                    "CBullD_Lower_Low_date_neg_MACD",
                    "CBullD_Higher_Low_date_neg_MACD",
                ),
                "macd": (
                    2,
                    1,
                    "CBullD_Lower_Low_MACD_neg_MACD",
                    "CBullD_Higher_Low_MACD_neg_MACD",
                    "CBullD_Lower_Low_date_neg_MACD",
                    "CBullD_Higher_Low_date_neg_MACD",
                ),
            },
        }

        sel_gen = df["CBullD_gen"] == 1
        sel_neg = df["CBullD_neg_MACD"] == 1

        _add_divergence_markers_vectorized(
            fig, df, sel_gen, "Bullish Divergence (gen)", row_maps["gen"]
        )
        _add_divergence_markers_vectorized(
            fig, df, sel_neg, "Bullish Divergence (neg MACD)", row_maps["neg"]
        )

    def _derive_base_optimized(p):
        base = os.path.splitext(p)[0]
        suffixes = (".precomp.enriched", ".precomp.divergences_rsi", ".precomp")
        for suf in suffixes:
            if base.endswith(suf):
                base = base[: -len(suf)]
                break
        return base

    base = _derive_base_optimized(path)
    div_candidates = [
        f"{base}.precomp.divergences_rsi.parquet",
        f"{os.path.splitext(path)[0].replace('.precomp.enriched', '.precomp.divergences_rsi')}.parquet",
        f"{os.path.splitext(path)[0].replace('.precomp', '')}.precomp.divergences_rsi.parquet",
    ]

    div_tbl = None
    for cand in div_candidates:
        if cand and os.path.isfile(cand):
            try:
                div_tbl = pl.read_parquet(cand)
                break
            except Exception:
                continue

    if div_tbl is not None and len(div_tbl) > 0:
        div_res = div_shapes_from_table(
            df.to_pandas(), div_tbl.to_pandas(), curvature=0.12, arrow_size_frac=0.06
        )
        divergence_shapes = list(div_res.get("shapes", []))
        divergence_annotations = list(div_res.get("annotations", []))
        div_traces = rsi_traces_from_table(div_tbl.to_pandas())
    else:
        live = detect_divergences(
            df.to_pandas(),
            price_col="close",
            indicator_col="RSI",
            lookback=20,
            curvature=0.12,
        )
        divergence_shapes = list(live.get("shapes", []))
        divergence_annotations = list(live.get("annotations", []))
        div_traces, _ = build_rsi_divergence_traces(
            df.to_pandas(), lookback=20, price_col="close", indicator_col="RSI"
        )

    diver_trace_indices = []
    for tr in div_traces:
        fig.add_trace(tr, row=4, col=1)
        diver_trace_indices.append(len(fig.data) - 1)

    fname = os.path.basename(path)

    for shape in formation_shapes + divergence_shapes:
        if "yref" in shape and shape["yref"] == "paper":
            shape["y0"] = max(0, min(shape["y0"], 1))
            shape["y1"] = max(0, min(shape["y1"], 1))

    layout_updates = {
        "template": "plotly_white",
        "title": {
            "text": f"Classic Bullish Divergence — {fname}",
            "x": 0.5,
            "y": 0.98,
            "xanchor": "center",
            "yanchor": "top",
            "pad": {"t": 6, "b": 4},
        },
        "margin": {"l": 60, "r": 60, "t": 200, "b": 60},
        "legend": {
            "orientation": "h",
            "yanchor": "bottom",
            "y": 0.82,
            "xanchor": "center",
            "x": 0.5,
        },
        "hovermode": "x unified",
        "dragmode": "zoom",
        "uirevision": "cbulldiv_viewer_optimized",
        "hoverdistance": 15,
        "spikedistance": 15,
        "annotations": _ann_to_dict_list(
            formation_annotations + divergence_annotations
        ),
        "shapes": formation_shapes + divergence_shapes,
    }

    fig.update_layout(**layout_updates)

    x_range = [t[0].to_pandas(), t[-1].to_pandas()]
    for row in [1, 2, 3, 4]:
        fig.update_xaxes(
            row=row,
            col=1,
            matches="x",
            anchor="x",
            rangeslider_visible=False,
            range=x_range,
            showspikes=True,
            spikemode="across",
            spikesnap="cursor",
            spikethickness=1,
        )

    fig.update_yaxes(showspikes=True, spikethickness=1)

    yaxis_configs = [
        (1, "Preis", {"fixedrange": False, "rangemode": "normal"}),
        (2, "MACD", {"fixedrange": False, "rangemode": "normal"}),
        (3, "RSI", {"fixedrange": False, "rangemode": "normal", "range": [0, 100]}),
        (
            4,
            "RSI Divergenzen",
            {"fixedrange": False, "rangemode": "normal", "range": [0, 100]},
        ),
    ]

    for row, title, config in yaxis_configs:
        fig.update_yaxes(title_text=title, row=row, col=1, **config)

    n = len(fig.data)
    base_visible = [True] * n

    visibility_masks = {
        "pattern_off": [i not in pattern_trace_indices for i in range(n)],
        "div_off": [i not in diver_trace_indices for i in range(n)],
        "extrema_off": [i not in extrema_trace_indices for i in range(n)],
    }

    shape_combinations = {
        "all": formation_shapes + divergence_shapes,
        "form_only": formation_shapes,
        "div_only": divergence_shapes,
    }

    annotation_combinations = {
        "all": formation_annotations + divergence_annotations,
        "form_only": formation_annotations,
        "div_only": divergence_annotations,
    }

    button_style = {
        "font": {"size": 11, "color": "#333"},
        "bgcolor": "rgba(255,255,255,0.95)",
        "bordercolor": "#ccc",
        "borderwidth": 1,
        "active": 0,
        "direction": "down",
        "x": 0.98,
        "xanchor": "right",
        "yanchor": "top",
        "pad": {"r": 8, "t": 4, "l": 4, "b": 4},
    }

    button_configs = [
        {
            "name": "formations",
            "y": 0.98,
            "buttons": [
                {
                    "label": "☑ Formationen",
                    "visible": base_visible,
                    "shapes": "all",
                    "annotations": "all",
                },
                {
                    "label": "☐ Formationen",
                    "visible": visibility_masks["pattern_off"],
                    "shapes": "div_only",
                    "annotations": "div_only",
                },
            ],
        },
        {
            "name": "divergences",
            "y": 0.90,
            "buttons": [
                {
                    "label": "☑ Divergenzen",
                    "visible": base_visible,
                    "shapes": "all",
                    "annotations": "all",
                },
                {
                    "label": "☐ Divergenzen",
                    "visible": visibility_masks["div_off"],
                    "shapes": "form_only",
                    "annotations": "form_only",
                },
            ],
        },
        {
            "name": "extrema",
            "y": 0.82,
            "buttons": [
                {"label": "☑ Extrema", "visible": base_visible},
                {"label": "☐ Extrema", "visible": visibility_masks["extrema_off"]},
            ],
        },
        {
            "name": "form_names",
            "y": 0.74,
            "buttons": [
                {"label": "☑ Formationsnamen", "annotations": "all"},
                {"label": "☐ Formationsnamen", "annotations": "div_only"},
            ],
        },
        {
            "name": "hover",
            "y": 0.66,
            "buttons": [
                {"label": "☑ Hover", "hover_mode": "x unified", "hover_enabled": True},
                {"label": "☐ Hover", "hover_mode": False, "hover_enabled": False},
            ],
        },
    ]

    hover_states = {
        "on": {
            "hoverinfo": [getattr(tr, "hoverinfo", None) for tr in fig.data],
            "hovertemplate": [getattr(tr, "hovertemplate", None) for tr in fig.data],
        },
        "off": {"hoverinfo": ["skip"] * n, "hovertemplate": [None] * n},
    }

    update_menus = []

    for config in button_configs:
        menu = dict(button_style)
        menu["type"] = "buttons"
        menu["y"] = config["y"]
        menu["buttons"] = []

        for btn in config["buttons"]:
            args = []

            if "visible" in btn:
                trace_updates = {"visible": btn["visible"]}
                args.append(trace_updates)

            layout_updates = {}

            if "shapes" in btn:
                layout_updates["shapes"] = shape_combinations[btn["shapes"]]

            if "annotations" in btn:
                layout_updates["annotations"] = annotation_combinations[
                    btn["annotations"]
                ]

            if "hover_mode" in btn:
                layout_updates["hovermode"] = btn["hover_mode"]
                if "hover_enabled" in btn:
                    if btn["hover_enabled"]:
                        trace_updates = hover_states["on"]
                    else:
                        trace_updates = hover_states["off"]
                    args = [trace_updates, layout_updates]

            if layout_updates and len(args) == 0:
                args = [layout_updates]
            elif layout_updates and len(args) == 1:
                args.append(layout_updates)

            method = (
                "restyle"
                if len(args) == 1 and "visible" in btn
                else (
                    "relayout"
                    if "annotations" in btn and "visible" not in btn
                    else "update"
                )
            )

            menu["buttons"].append(
                {
                    "label": btn["label"],
                    "method": method,
                    "args": args if args else [{}],
                }
            )

        update_menus.append(menu)

    fig.update_layout(updatemenus=update_menus)

    config = {
        "scrollZoom": True,
        "displaylogo": False,
        "toImageButtonOptions": {
            "scale": 2,
            "filename": f"cbulldiv_plot_{fname}",
            "format": "png",
        },
        "modeBarButtonsToRemove": ["lasso2d", "select2d"],
        "displayModeBar": True,
        "responsive": True,
    }

    fig.show(config=config)


# ---------------- Optimized Main Function ----------------
def main():
    try:
        paths = select_files()
        if not paths:
            print("[INFO] Keine Datei gewählt.")
            return

        print(f"[INFO] Verarbeite {len(paths)} Datei(en)...")

        for i, path in enumerate(paths, 1):
            print(f"[INFO] Plot {i}/{len(paths)}: {os.path.basename(path)}")
            try:
                _plot_one_file_optimized(path)
            except Exception as e:
                print(f"[ERROR] Fehler bei Datei {path}: {e}")
                continue

        print("[INFO] Alle Plots erstellt.")

    except KeyboardInterrupt:
        print("\n[INFO] Abgebrochen durch Benutzer.")
    except Exception as e:
        print(f"[ERROR] Unerwarteter Fehler: {e}")


if __name__ == "__main__":
    main()
