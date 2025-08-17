# -*- coding: utf-8 -*-
"""
sensitivity_plotly.py – α/γ-Sensitivitäts-Overlay (Plotly, ohne Qt)

Features:
- 3 Panels: Candles(+EMA 20/50/100/200), RSI, MACD(+Hist)
- Farbe = Signaltyp (blau=gen, rot=neg_MACD), Form = Param-Set (25 eindeutige Marker)
- Große Marker mit schwarzer Outline
- Legende:
  • Ein Eintrag pro (α,γ) → toggelt alle Spuren dieses Sets
  • Eine Gruppe für Referenz: REF – Brute Force → toggelt alle REF-Marker
- Heatmap (Trefferzahlen) + CSV/HTML-Export
- Validierung (optional): Brute-Force-Referenz, TP/FP/FN, Precision/Recall/F1, CSV
- XLSX-Export (--save-xlsx): counts, Pivot-Heatmaps (formatiert), optional validation
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Modulpfade
THIS_FILE = Path(__file__).resolve()
RT_DIR = THIS_FILE.parent
PROJECT_ROOT = RT_DIR.parents[1]
AJ_DIR = PROJECT_ROOT / "scripts" / "aj"
if str(AJ_DIR) not in sys.path:
    sys.path.insert(0, str(AJ_DIR))

# Analyse-Module
from CBullDivg_Analysis_vectorized import CBullDivg_analysis
from Initialize_RSI_EMA_MACD import Initialize_RSI_EMA_MACD
from Local_Maximas_Minimas import Local_Max_Min

# Optional: polars für CSV->Parquet
try:
    import polars as pl
except ImportError:
    pl = None


# ---------------- Loader ----------------
def ensure_parquet_from_csv(csv_path: Path) -> Path:
    pq_path = csv_path.with_suffix(".parquet")
    if (not pq_path.exists()) or (csv_path.stat().st_mtime > pq_path.stat().st_mtime):
        if pl is None:
            raise RuntimeError("Polars benötigt für CSV→Parquet (`pip install polars`).")
        pl.read_csv(str(csv_path)).write_parquet(str(pq_path))
    return pq_path


def load_market_data(path: Path, engine: str = "polars") -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        path = ensure_parquet_from_csv(path)

    if engine == "polars":
        if pl is None:
            raise RuntimeError("Polars nicht installiert.")
        lf = pl.scan_parquet(str(path))
        cols = lf.collect_schema().keys()
        ts_col = "timestamp" if "timestamp" in cols else ("date" if "date" in cols else None)
        if ts_col is None:
            raise RuntimeError(f"Keine Zeitspalte (timestamp/date). Vorhanden: {cols}")
        return lf.sort(ts_col).collect().to_pandas()
    else:
        df = pd.read_parquet(str(path))
        if "timestamp" in df.columns:
            return df.sort_values("timestamp")
        elif "date" in df.columns:
            _ts = pd.to_datetime(df["date"], utc=True, errors="coerce")
            return df.assign(_ts=_ts).sort_values("_ts").drop(columns=["_ts"])
        return df


# ---------------- CLI ----------------
def parse_args():
    p = argparse.ArgumentParser("CBullDivg Sensitivität (Overlay + Heatmap + Validation + XLSX)")
    p.add_argument("--input", type=str, required=True)
    p.add_argument("--engine", choices=["polars", "pandas"], default="polars")
    p.add_argument("--lookback", type=int, default=5)

    # Parameter: Listen ODER Linspace (start:end:steps) – Default: 5×5
    p.add_argument("--alphas", type=str, help='Liste α, z.B. "0.06,0.10,0.14,0.18,0.22"')
    p.add_argument("--gammas", type=str, help='Liste γ, z.B. "2.0,2.75,3.5,4.25,5.0"')
    p.add_argument("--alpha-lin", type=str, help='α-Linspace "0.04:0.24:5"')
    p.add_argument("--gamma-lin", type=str, help='γ-Linspace "2.0:5.0:5"')

    p.add_argument("--title", type=str, default="α/γ-Sensitivitätsanalyse")
    p.add_argument("--save-csv", action="store_true")
    p.add_argument("--save-html", action="store_true")
    p.add_argument("--save-xlsx", action="store_true")

    # Validation / Brute-Force
    p.add_argument("--validate", action="store_true", help="Brute-Force-Referenz + Metriken ausgeben")
    p.add_argument("--match-tol-bars", type=int, default=2, help="Toleranz für Event-Matching in Bars")

    return p.parse_args()


def parse_list_floats(s: str):
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def parse_lin(s: str):
    # "start:end:steps"
    a, b, n = s.split(":")
    a, b, n = float(a), float(b), int(n)
    if n < 2:
        return [a]
    xs = np.linspace(a, b, n)
    return [float(f"{x:.10g}") for x in xs]


# -------------- Plot Helpers --------------
def _ts_series(df: pd.DataFrame) -> pd.Series:
    if "timestamp" in df.columns:
        return pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    return pd.to_datetime(df["date"], utc=True, errors="coerce")


def make_base_figure(df: pd.DataFrame, title: str) -> go.Figure:
    ts = _ts_series(df)

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.55, 0.22, 0.23], vertical_spacing=0.03,
        specs=[[{"type": "xy"}], [{"type": "xy"}], [{"type": "xy"}]],
        subplot_titles=(title, "RSI", "MACD")
    )

    # Candles
    fig.add_trace(go.Candlestick(
        x=ts, open=df["open"], high=df["high"], low=df["low"], close=df["close"], name="OHLC"
    ), row=1, col=1)

    # EMAs
    for ema_col, label in [("EMA_20", "EMA 20"), ("EMA_50", "EMA 50"),
                           ("EMA_100", "EMA 100"), ("EMA_200", "EMA 200")]:
        if ema_col in df.columns and not pd.Series(df[ema_col]).isna().all():
            fig.add_trace(go.Scatter(x=ts, y=df[ema_col], mode="lines", name=label, opacity=0.75), row=1, col=1)

    # RSI
    if "RSI" in df.columns and not pd.Series(df["RSI"]).isna().all():
        fig.add_trace(go.Scatter(x=ts, y=df["RSI"], mode="lines", name="RSI"), row=2, col=1)
        fig.add_hrect(y0=70, y1=70, line=dict(dash="dot"), fillcolor="rgba(0,0,0,0)", row=2, col=1)
        fig.add_hrect(y0=30, y1=30, line=dict(dash="dot"), fillcolor="rgba(0,0,0,0)", row=2, col=1)

    # MACD + Signal + Histogramm
    if "macd_histogram" in df.columns and not pd.Series(df["macd_histogram"]).isna().all():
        fig.add_trace(go.Bar(x=ts, y=df["macd_histogram"], name="MACD-Hist"), row=3, col=1)
    if "macd" in df.columns and not pd.Series(df["macd"]).isna().all():
        fig.add_trace(go.Scatter(x=ts, y=df["macd"], mode="lines", name="MACD"), row=3, col=1)
    if "signal" in df.columns and not pd.Series(df["signal"]).isna().all():
        fig.add_trace(go.Scatter(x=ts, y=df["signal"], mode="lines", name="Signal"), row=3, col=1)

    fig.update_layout(
        title=title,
        xaxis_rangeslider_visible=False,
        height=900,
        legend_title="(α, γ) / REF",
        legend=dict(groupclick="togglegroup", tracegroupgap=10)
    )
    return fig


def _coerce_datetime(s: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(s, utc=True, errors="coerce")
    except Exception:
        return pd.Series(pd.NaT, index=s.index)


def _safe_series(df: pd.DataFrame, name: str) -> pd.Series:
    return df[name] if name in df.columns else pd.Series([np.nan]*len(df), index=df.index)


# 25 eindeutige Markerformen (für 5x5 Grid)
PARAM_SYMBOLS = [
    "circle", "square", "diamond", "triangle-up", "triangle-down",
    "triangle-left", "triangle-right", "x", "cross", "star",
    "hexagon", "hexagon2", "pentagon", "hourglass", "bowtie",
    "circle-x", "square-x", "diamond-x", "circle-cross", "square-cross",
    "triangle-ne", "triangle-se", "triangle-sw", "triangle-nw", "y-up"
]


def add_markers_for_run(fig: go.Figure, res: pd.DataFrame, base_df: pd.DataFrame,
                        alpha: float, gamma: float, symbol_for_set: str):
    """
    Farbe = Signaltyp: gen→blau, neg→rot. Form = symbol_for_set (einzigartig je Param-Set).
    Große Marker + schwarze Outline. Legenden-Gruppenkopf zeigt dieselbe Form.
    """
    ts_base = _ts_series(base_df)
    group_key = f"a{alpha}g{gamma}"
    group_title = f"α={alpha:g}, γ={gamma:g}"

    # Gruppenkopf in Legende (zeigt Form)
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(color="#cccccc", symbol=symbol_for_set, size=12, line=dict(color="black", width=1.6)),
        name=group_title,
        legendgroup=group_key, legendgrouptitle_text=group_title,
        showlegend=True
    ), row=1, col=1)

    def _plot_points(prefix: str, typ: str, y_fallback_col: str):
        flag_col = f"CBullD_{typ}"
        date_col = f"CBullD_{prefix}_date_{typ}"
        y_col = f"CBullD_{prefix}_{typ}"
        rsi_col = f"CBullD_{prefix}_RSI_{typ}"
        macd_col = f"CBullD_{prefix}_MACD_{typ}"

        if flag_col not in res.columns or res[flag_col].sum() <= 0:
            return

        mask_flag = res[flag_col].astype(float) == 1.0

        x = _coerce_datetime(res[date_col]) if date_col in res.columns else pd.Series(pd.NaT, index=res.index)
        if x.isna().all():
            x = ts_base
        x = x[mask_flag]

        y = res[y_col] if y_col in res.columns and not _safe_series(res, y_col).isna().all() \
            else _safe_series(base_df, y_fallback_col)
        y = y[mask_flag]
        valid = x.notna() & y.notna()
        x, y = x[valid], y[valid]
        if len(x) == 0:
            return

        # Farbe nach Typ
        color_kurs = "#1f77b4" if typ == "gen" else "#d62728"  # blau / rot
        linecol = "black"

        # Kurs-Paneel
        fig.add_trace(go.Scatter(
            x=x, y=y, mode="markers",
            marker=dict(color=color_kurs, symbol=symbol_for_set, size=14, line=dict(color=linecol, width=1.8)),
            name=("gen – " if typ == "gen" else "neg – ") + prefix.replace("_", " "),
            legendgroup=group_key, showlegend=False
        ), row=1, col=1)

        # RSI
        yrsi = _safe_series(res, rsi_col)
        if not yrsi.isna().all():
            yrsi = yrsi.loc[x.index]
            fig.add_trace(go.Scatter(
                x=x, y=yrsi, mode="markers",
                marker=dict(color=color_kurs, symbol=symbol_for_set, size=12, line=dict(color=linecol, width=1.6)),
                name=("gen RSI " if typ == "gen" else "neg RSI ") + prefix.replace("_", " "),
                legendgroup=group_key, showlegend=False
            ), row=2, col=1)

        # MACD
        ymacd = _safe_series(res, macd_col)
        if not ymacd.isna().all():
            ymacd = ymacd.loc[x.index]
            fig.add_trace(go.Scatter(
                x=x, y=ymacd, mode="markers",
                marker=dict(color=color_kurs, symbol=symbol_for_set, size=12, line=dict(color=linecol, width=1.6)),
                name=("gen MACD " if typ == "gen" else "neg MACD ") + prefix.replace("_", " "),
                legendgroup=group_key, showlegend=False
            ), row=3, col=1)

    for typ in ["gen", "neg_MACD"]:
        for prefix in ["Lower_Low", "Higher_Low"]:
            _plot_points(prefix, typ, "low")


# ---------- Referenz (Brute-Force) + Overlay ----------
def ref_bullish_divergences(df: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """LowerLow im Preis + HigherLow im RSI in 2..lookback Bars (naive Referenz)."""
    ts = _ts_series(df)
    low = df["low"].to_numpy()
    rsi = df["RSI"].to_numpy() if "RSI" in df.columns else np.full(len(df), np.nan)

    is_min = np.zeros(len(df), dtype=bool)
    is_min[1:-1] = (low[1:-1] < low[:-2]) & (low[1:-1] <= low[2:])

    events = []
    idxs = np.where(is_min)[0]
    for k in range(1, len(idxs)):
        i1, i2 = idxs[k-1], idxs[k]
        d = i2 - i1
        if 2 <= d <= max(2, lookback):
            if (low[i2] < low[i1]) and np.isfinite(rsi[i1]) and np.isfinite(rsi[i2]) and (rsi[i2] > rsi[i1]):
                events.append({"ts": ts.iloc[i2], "y_price": low[i2], "y_osci": rsi[i2]})
    return pd.DataFrame(events)


def ref_neg_macd_divergences(df: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """LowerLow Preis + HigherLow im MACD (weniger negativ) im Fenster 2..lookback."""
    ts = _ts_series(df)
    low = df["low"].to_numpy()
    macd = df["macd"].to_numpy() if "macd" in df.columns else np.full(len(df), np.nan)

    is_min = np.zeros(len(df), dtype=bool)
    is_min[1:-1] = (low[1:-1] < low[:-2]) & (low[1:-1] <= low[2:])

    events = []
    idxs = np.where(is_min)[0]
    for k in range(1, len(idxs)):
        i1, i2 = idxs[k-1], idxs[k]
        d = i2 - i1
        if 2 <= d <= max(2, lookback):
            # <-- FIX: Klammer korrigiert
            if (low[i2] < low[i1]) and np.isfinite(macd[i1]) and np.isfinite(macd[i2]) and (macd[i2] > macd[i1]):
                events.append({"ts": ts.iloc[i2], "y_price": low[i2], "y_osci": macd[i2]})
    return pd.DataFrame(events)


def add_reference_overlay(fig: go.Figure, df: pd.DataFrame, lookback: int):
    """Zeichnet Referenz-Events als eigene Legendengruppe 'REF – Brute Force' in alle Panels."""
    ref_gen = ref_bullish_divergences(df, lookback)
    ref_neg = ref_neg_macd_divergences(df, lookback)

    def _add(ts, y, color, name, row):
        if len(ts) == 0:
            return
        fig.add_trace(go.Scatter(
            x=ts, y=y, mode="markers",
            marker=dict(
                color="#ffffff",  # Weißer Fill
                symbol="star",    # Einprägsam
                size=13,
                line=dict(color=color, width=2.2)  # Kontur in Typfarbe
            ),
            name=name, legendgroup="REF", showlegend=True
        ), row=row, col=1)

    # Kurs
    _add(ref_gen["ts"], ref_gen["y_price"], "#1f77b4", "REF gen (Kurs)", 1)
    _add(ref_neg["ts"], ref_neg["y_price"], "#d62728", "REF neg (Kurs)", 1)
    # RSI/MACD
    _add(ref_gen["ts"], ref_gen["y_osci"], "#1f77b4", "REF gen (RSI)", 2)
    _add(ref_neg["ts"], ref_neg["y_osci"], "#d62728", "REF neg (MACD)", 3)

    # Gruppenkopf zum Gruppentoggle
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(color="#ffffff", symbol="star", size=12, line=dict(color="#000000", width=1.6)),
        name="REF – Brute Force", legendgroup="REF", legendgrouptitle_text="REF – Brute Force", showlegend=True
    ), row=1, col=1)

    return ref_gen, ref_neg


# -------------- Heatmap --------------
def build_heatmap_fig(df_counts: pd.DataFrame, title: str) -> go.Figure:
    pv_gen = df_counts.pivot(index="alpha", columns="gamma", values="n_div_gen").sort_index().sort_index(axis=1)
    pv_neg = df_counts.pivot(index="alpha", columns="gamma", values="n_div_neg_macd").sort_index().sort_index(axis=1)

    fig = make_subplots(
        rows=1, cols=2, subplot_titles=("Bullish gen", "neg MACD"),
        horizontal_spacing=0.12, specs=[[{"type": "heatmap"}, {"type": "heatmap"}]]
    )

    fig.add_trace(go.Heatmap(
        z=pv_gen.values, x=[str(c) for c in pv_gen.columns], y=[str(i) for i in pv_gen.index],
        coloraxis="coloraxis", hovertemplate="α=%{y}<br>γ=%{x}<br>gen=%{z}<extra></extra>"
    ), row=1, col=1)

    fig.add_trace(go.Heatmap(
        z=pv_neg.values, x=[str(c) for c in pv_neg.columns], y=[str(i) for i in pv_neg.index],
        coloraxis="coloraxis", hovertemplate="α=%{y}<br>γ=%{x}<br>neg=%{z}<extra></extra>"
    ), row=1, col=2)

    fig.update_layout(
        title=title + " – Treffer-Heatmap",
        coloraxis_colorscale="Viridis",
        xaxis_title="γ", xaxis2_title="γ",
        yaxis_title="α", yaxis2_title="α",
        height=520
    )

    # Zahlen overlay
    for (pv, r, c) in [(pv_gen, 1, 1), (pv_neg, 1, 2)]:
        for i, a in enumerate(pv.index):
            for j, g in enumerate(pv.columns):
                val = pv.iloc[i, j]
                txt = "0" if pd.isna(val) else str(int(val))
                fig.add_annotation(x=str(g), y=str(a), text=txt, showarrow=False, row=r, col=c,
                                   font=dict(color="white", size=10))
    return fig


# -------------- Validierung --------------
def extract_algo_events(res: pd.DataFrame, typ: str) -> pd.Series:
    n = len(res)
    if typ == "gen":
        a = _coerce_datetime(res.get("CBullD_Higher_Low_date_gen", pd.Series([pd.NaT]*n)))
        b = _coerce_datetime(res.get("CBullD_Lower_Low_date_gen", pd.Series([pd.NaT]*n)))
        flag = res.get("CBullD_gen", pd.Series([0]*n)).astype(float) == 1.0
    else:
        a = _coerce_datetime(res.get("CBullD_Higher_Low_date_neg_MACD", pd.Series([pd.NaT]*n)))
        b = _coerce_datetime(res.get("CBullD_Lower_Low_date_neg_MACD", pd.Series([pd.NaT]*n)))
        flag = res.get("CBullD_neg_MACD", pd.Series([0]*n)).astype(float) == 1.0
    ts = a.where(a.notna(), b)
    return ts[flag & ts.notna()]


def match_events(ts_algo: pd.Series, ts_ref: pd.Series, tol_bars: int, base_ts: pd.Series) -> tuple[int,int,int]:
    idx_map = pd.Series(range(len(base_ts)), index=base_ts.values)
    idx_algo = ts_algo.map(lambda t: idx_map.get(t, np.nan)).dropna().astype(int).tolist()
    idx_ref  = ts_ref.map(lambda t: idx_map.get(t, np.nan)).dropna().astype(int).tolist()

    used_ref = set()
    tp = 0
    for ia in idx_algo:
        for ir in idx_ref:
            if ir in used_ref:
                continue
            if abs(ia - ir) <= tol_bars:
                used_ref.add(ir)
                tp += 1
                break
    fp = len(idx_algo) - tp
    fn = len(idx_ref) - tp
    return tp, fp, fn


# -------------- XLSX --------------
def save_xlsx(out_dir: Path, stamp: str, counts: pd.DataFrame,
              val_df: pd.DataFrame | None,
              alphas: list[float], gammas: list[float]):
    xlsx_path = out_dir / f"sensitivity_{stamp}.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as writer:
        counts.to_excel(writer, sheet_name="counts", index=False)

        # Pivots
        pv_gen = counts.pivot(index="alpha", columns="gamma", values="n_div_gen").reindex(index=alphas, columns=gammas)
        pv_neg = counts.pivot(index="alpha", columns="gamma", values="n_div_neg_macd").reindex(index=alphas, columns=gammas)
        pv_gen.to_excel(writer, sheet_name="pivot_gen")
        pv_neg.to_excel(writer, sheet_name="pivot_neg")

        # Heatmap-Formatierung (bedingte Formatierung)
        wb = writer.book
        fmt = wb.add_format({"num_format": "0", "align": "center"})
        cmap = {"type": "3_color_scale", "min_color": "#ffffff", "mid_color": "#9ecae1", "max_color": "#08519c"}

        for sheet, pv in [("pivot_gen", pv_gen), ("pivot_neg", pv_neg)]:
            ws = writer.sheets[sheet]
            start_row, start_col = 1, 1  # B2
            end_row = start_row + len(pv.index) - 1
            end_col = start_col + len(pv.columns) - 1
            if end_row >= start_row and end_col >= start_col:
                ws.set_column(start_col, end_col, 10, fmt)
                ws.conditional_format(start_row, start_col, end_row, end_col, cmap)

        if val_df is not None and not val_df.empty:
            val_df.to_excel(writer, sheet_name="validation", index=False)

    print(f"[OK] XLSX gespeichert: {xlsx_path}")


# ---------------- Main ----------------
def main():
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        print("[ERROR] Datei nicht gefunden:", input_path)
        sys.exit(1)

    # Daten
    df = load_market_data(input_path, args.engine)
    print(f"[OK] Daten geladen: {len(df)} Zeilen")

    # Indikatoren + Extrema einmalig
    df = Initialize_RSI_EMA_MACD(df)
    if df is None:
        print("[ERROR] Initialize_RSI_EMA_MACD lieferte None.")
        sys.exit(1)
    Local_Max_Min(df)
    print("[OK] Indikatoren + LM_* erstellt.")

    # Overlay-Basis
    fig_overlay = make_base_figure(df, args.title)

    # Parameterlisten / -linspace (Default: 5×5)
    if args.alphas:
        alphas = parse_list_floats(args.alphas)
    elif args.alpha_lin:
        alphas = parse_lin(args.alpha_lin)
    else:
        alphas = parse_lin("0.04:0.24:5")

    if args.gammas:
        gammas = parse_list_floats(args.gammas)
    elif args.gamma_lin:
        gammas = parse_lin(args.gamma_lin)
    else:
        gammas = parse_lin("2.0:5.0:5")

    # 25 Shapes sicherstellen
    if len(alphas)*len(gammas) > len(PARAM_SYMBOLS):
        print("[WARN] Mehr als 25 Param-Kombinationen – Formen werden recycelt.")
    sym_idx = 0

    # Zähler
    rows = []

    # Läufe
    for a in alphas:
        for g in gammas:
            symbol_for_set = PARAM_SYMBOLS[sym_idx % len(PARAM_SYMBOLS)]
            sym_idx += 1

            res = CBullDivg_analysis(df.copy(), args.lookback, a, g)
            add_markers_for_run(fig_overlay, res, df, a, g, symbol_for_set)

            n_gen = int(res["CBullD_gen"].sum()) if "CBullD_gen" in res.columns else 0
            n_neg = int(res["CBullD_neg_MACD"].sum()) if "CBullD_neg_MACD" in res.columns else 0
            rows.append({"alpha": a, "gamma": g, "lookback": args.lookback,
                         "n_div_gen": n_gen, "n_div_neg_macd": n_neg})
            print(f"[OK] α={a}, γ={g} → gen={n_gen}, neg={n_neg}")

    df_counts = pd.DataFrame(rows)

    # Referenz-Overlay (eigene Legendengruppe „REF – Brute Force“)
    ref_gen_df, ref_neg_df = add_reference_overlay(fig_overlay, df, args.lookback)

    # Heatmap
    fig_heatmap = build_heatmap_fig(df_counts, args.title)

    # Ausgaben
    out_dir = PROJECT_ROOT / "results" / "sensitivity_overlay"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    if args.save_csv:
        csv_path = out_dir / f"overlay_counts_{stamp}.csv"
        df_counts.to_csv(csv_path, index=False)
        print(f"[OK] CSV gespeichert: {csv_path}")

    if args.save_html:
        overlay_html = out_dir / f"overlay_plot_{stamp}.html"
        heatmap_html = out_dir / f"overlay_heatmap_{stamp}.html"
        fig_overlay.write_html(str(overlay_html))
        fig_heatmap.write_html(str(heatmap_html))
        print(f"[OK] HTML gespeichert: {overlay_html}")
        print(f"[OK] HTML gespeichert: {heatmap_html}")

    # Validierung (optional)
    val_df = None
    if args.validate:
        base_ts = _ts_series(df)
        ts_ref_gen = ref_gen_df["ts"] if not ref_gen_df.empty else pd.Series(dtype="datetime64[ns, UTC]")
        ts_ref_neg = ref_neg_df["ts"] if not ref_neg_df.empty else pd.Series(dtype="datetime64[ns, UTC]")

        val_rows = []
        for _, r in df_counts.iterrows():
            a, g = float(r["alpha"]), float(r["gamma"])
            res = CBullDivg_analysis(df.copy(), args.lookback, a, g)

            ts_algo_gen = extract_algo_events(res, "gen")
            ts_algo_neg = extract_algo_events(res, "neg_MACD")

            tp_g, fp_g, fn_g = match_events(ts_algo_gen, ts_ref_gen, args.match_tol_bars, base_ts)
            tp_n, fp_n, fn_n = match_events(ts_algo_neg, ts_ref_neg, args.match_tol_bars, base_ts)

            prec_g = tp_g / (tp_g + fp_g) if (tp_g + fp_g) > 0 else 0.0
            rec_g  = tp_g / (tp_g + fn_g) if (tp_g + fn_g) > 0 else 0.0
            f1_g   = 2*prec_g*rec_g / (prec_g + rec_g) if (prec_g + rec_g) > 0 else 0.0

            prec_n = tp_n / (tp_n + fp_n) if (tp_n + fp_n) > 0 else 0.0
            rec_n  = tp_n / (tp_n + fn_n) if (tp_n + fn_n) > 0 else 0.0
            f1_n   = 2*prec_n*rec_n / (prec_n + rec_n) if (prec_n + rec_n) > 0 else 0.0

            val_rows.append({
                "alpha": a, "gamma": g, "lookback": args.lookback,
                "tp_gen": tp_g, "fp_gen": fp_g, "fn_gen": fn_g, "precision_gen": prec_g, "recall_gen": rec_g, "f1_gen": f1_g,
                "tp_neg": tp_n, "fp_neg": fp_n, "fn_neg": fn_n, "precision_neg": prec_n, "recall_neg": rec_n, "f1_neg": f1_n,
                "ref_gen": len(ts_ref_gen), "ref_neg": len(ts_ref_neg),
            })

        val_df = pd.DataFrame(val_rows)
        val_csv = out_dir / f"validation_{stamp}.csv"
        val_df.to_csv(val_csv, index=False)
        print(f"[OK] Validierungsreport: {val_csv}")

    # XLSX
    if args.save_xlsx:
        save_xlsx(out_dir, stamp, df_counts, val_df, alphas, gammas)

    # Anzeigen
    fig_overlay.show()
    fig_heatmap.show()


if __name__ == "__main__":
    main()
