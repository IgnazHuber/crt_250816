# -*- coding: utf-8 -*-
"""
sensitivity_plotly.py – α/γ-Sensitivitäts-Overlay (Plotly, ohne Qt)

Neu:
- --no-variants / --no-ref: initiale Sichtbarkeit
- --master-toggle: Updatemenü mit Buttons (Alle an/aus, Nur Varianten, Nur Referenz)
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- optional: Tkinter-Explorer ---
try:
    import tkinter as tk
    from tkinter import filedialog
    TK_OK = True
except Exception:
    TK_OK = False

THIS_FILE = Path(__file__).resolve()
RT_DIR = THIS_FILE.parent
PROJECT_ROOT = RT_DIR.parents[1]
AJ_DIR = PROJECT_ROOT / "scripts" / "aj"
if str(AJ_DIR) not in sys.path:
    sys.path.insert(0, str(AJ_DIR))

from CBullDivg_Analysis_vectorized import CBullDivg_analysis
from Initialize_RSI_EMA_MACD import Initialize_RSI_EMA_MACD
from Local_Maximas_Minimas import Local_Max_Min

try:
    import polars as pl
except ImportError:
    pl = None

UI_DIR = PROJECT_ROOT / "data" / ".ui_state"
UI_DIR.mkdir(parents=True, exist_ok=True)
LAST_DIR_FILE = UI_DIR / "last_dir_sensitivity.txt"

PARAM_SYMBOLS = [
    "circle", "square", "diamond", "triangle-up", "triangle-down",
    "triangle-left", "triangle-right", "x", "cross", "star",
    "hexagon", "hexagon2", "pentagon", "hourglass", "bowtie",
    "circle-x", "square-x", "diamond-x", "circle-cross", "square-cross",
    "triangle-ne", "triangle-se", "triangle-sw", "triangle-nw", "y-up"
]

# ---------------- Utils ----------------
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
    df = pd.read_parquet(str(path))
    if "timestamp" in df.columns:
        return df.sort_values("timestamp")
    if "date" in df.columns:
        _ts = pd.to_datetime(df["date"], utc=True, errors="coerce")
        return df.assign(_ts=_ts).sort_values("_ts").drop(columns=["_ts"])
    return df

def get_last_dir() -> Path:
    if LAST_DIR_FILE.exists():
        try:
            p = Path(LAST_DIR_FILE.read_text(encoding="utf-8").strip())
            if p.exists():
                return p
        except Exception:
            pass
    return (PROJECT_ROOT / "data" / "raw")

def set_last_dir(p: Path) -> None:
    try:
        LAST_DIR_FILE.write_text(str(p), encoding="utf-8")
    except Exception:
        pass

def pick_files_via_explorer() -> list[Path]:
    if not TK_OK:
        print("[ERROR] Tkinter nicht verfügbar; bitte --input verwenden.")
        return []
    root = tk.Tk(); root.withdraw()
    initdir = str(get_last_dir())
    paths = filedialog.askopenfilenames(
        title="Dateien wählen (Parquet/CSV)",
        initialdir=initdir,
        filetypes=[("Parquet", "*.parquet"), ("CSV", "*.csv"), ("Alle Dateien", "*.*")]
    )
    root.destroy()
    paths = [Path(p) for p in paths]
    if paths:
        set_last_dir(paths[0].parent)
    return paths

def parse_list_floats(s: str):
    return [float(x.strip()) for x in s.split(",") if x.strip()]

def parse_lin(s: str):
    a, b, n = s.split(":")
    a, b, n = float(a), float(b), int(n)
    if n < 2:
        return [a]
    xs = np.linspace(a, b, n)
    return [float(f"{x:.10g}") for x in xs]

def _ts_series(df: pd.DataFrame) -> pd.Series:
    if "timestamp" in df.columns:
        return pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    return pd.to_datetime(df["date"], utc=True, errors="coerce")

def _coerce_datetime(s: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(s, utc=True, errors="coerce")
    except Exception:
        return pd.Series(pd.NaT, index=s.index)

def _safe_series(df: pd.DataFrame, name: str) -> pd.Series:
    return df[name] if name in df.columns else pd.Series([np.nan]*len(df), index=df.index)

# ---------------- CLI ----------------
def parse_args():
    p = argparse.ArgumentParser("CBullDivg Sensitivität (Overlay + Heatmap + Validation + XLSX)")
    p.add_argument("--input", type=str, help="Pfad zu einer Datei (CSV/Parquet). Bei --pick oder leer → Explorer.")
    p.add_argument("--pick", action="store_true", help="Explorer öffnen und 1..n Dateien auswählen.")
    p.add_argument("--engine", choices=["polars", "pandas"], default="polars")
    p.add_argument("--lookback", type=int, default=5)
    p.add_argument("--alphas", type=str)
    p.add_argument("--gammas", type=str)
    p.add_argument("--alpha-lin", type=str, help='z.B. "0.04:0.24:5"')
    p.add_argument("--gamma-lin", type=str, help='z.B. "2.0:5.0:5"')
    p.add_argument("--title", type=str, default="α/γ-Sensitivitätsanalyse")
    p.add_argument("--save-csv", action="store_true")
    p.add_argument("--save-html", action="store_true")
    p.add_argument("--save-xlsx", action="store_true")
    p.add_argument("--validate", action="store_true")
    p.add_argument("--match-tol-bars", type=int, default=2)
    # Sichtbarkeit / Master-Toggle
    p.add_argument("--no-variants", action="store_true", help="Varianten initial ausblenden")
    p.add_argument("--no-ref", action="store_true", help="Referenz initial ausblenden")
    p.add_argument("--master-toggle", action="store_true", default=True, help="Master-Toggle-Buttons einblenden")
    return p.parse_args()

# ---------------- Plot-Grundaufbau ----------------
def make_base_figure(df: pd.DataFrame, title: str) -> go.Figure:
    ts = _ts_series(df)
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.55, 0.22, 0.23], vertical_spacing=0.03,
        specs=[[{"type": "xy"}], [{"type": "xy"}], [{"type": "xy"}]],
        subplot_titles=(title, "RSI", "MACD")
    )
    fig.add_trace(go.Candlestick(
        x=ts, open=df["open"], high=df["high"], low=df["low"], close=df["close"], name="OHLC"
    ), row=1, col=1)
    for ema_col, label in [("EMA_20", "EMA 20"), ("EMA_50", "EMA 50"),
                           ("EMA_100", "EMA 100"), ("EMA_200", "EMA 200")]:
        if ema_col in df.columns and not pd.Series(df[ema_col]).isna().all():
            fig.add_trace(go.Scatter(x=ts, y=df[ema_col], mode="lines", name=label, opacity=0.75), row=1, col=1)
    if "RSI" in df.columns and not pd.Series(df["RSI"]).isna().all():
        fig.add_trace(go.Scatter(x=ts, y=df["RSI"], mode="lines", name="RSI"), row=2, col=1)
        fig.add_hrect(y0=70, y1=70, line=dict(dash="dot"), fillcolor="rgba(0,0,0,0)", row=2, col=1)
        fig.add_hrect(y0=30, y1=30, line=dict(dash="dot"), fillcolor="rgba(0,0,0,0)", row=2, col=1)
    if "macd_histogram" in df.columns and not pd.Series(df["macd_histogram"]).isna().all():
        fig.add_trace(go.Bar(x=_ts_series(df), y=df["macd_histogram"], name="MACD-Hist"), row=3, col=1)
    if "macd" in df.columns and not pd.Series(df["macd"]).isna().all():
        fig.add_trace(go.Scatter(x=_ts_series(df), y=df["macd"], mode="lines", name="MACD"), row=3, col=1)
    if "signal" in df.columns and not pd.Series(df["signal"]).isna().all():
        fig.add_trace(go.Scatter(x=_ts_series(df), y=df["signal"], mode="lines", name="Signal"), row=3, col=1)
    fig.update_layout(
        title=title,
        xaxis_rangeslider_visible=False,
        height=900,
        legend_title="(α, γ) / REF",
        legend=dict(groupclick="togglegroup", tracegroupgap=10)
    )
    return fig

# ---------------- Marker-Overlay ----------------
def add_markers_for_run(fig: go.Figure, res: pd.DataFrame, base_df: pd.DataFrame,
                        alpha: float, gamma: float, symbol_for_set: str, initial_visible=True):
    ts_base = _ts_series(base_df)
    group_key = f"a{alpha}g{gamma}"
    group_title = f"α={alpha:g}, γ={gamma:g}"
    initial_vis = True if initial_visible else "legendonly"

    # Gruppenkopf
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(color="#cccccc", symbol=symbol_for_set, size=12, line=dict(color="black", width=1.6)),
        name=group_title, legendgroup=group_key, legendgrouptitle_text=group_title,
        showlegend=True, visible=initial_vis
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
        if x.isna().all(): x = ts_base
        x = x[mask_flag]
        y = res[y_col] if y_col in res.columns and not _safe_series(res, y_col).isna().all() else _safe_series(base_df, y_fallback_col)
        y = y[mask_flag]
        valid = x.notna() & y.notna()
        x, y = x[valid], y[valid]
        if len(x) == 0: return
        color_kurs = "#1f77b4" if typ == "gen" else "#d62728"
        fig.add_trace(go.Scatter(
            x=x, y=y, mode="markers",
            marker=dict(color=color_kurs, symbol=symbol_for_set, size=14, line=dict(color="black", width=1.8)),
            name=("gen – " if typ == "gen" else "neg – ") + prefix.replace("_", " "),
            legendgroup=group_key, showlegend=False, visible=initial_vis
        ), row=1, col=1)
        yrsi = _safe_series(res, rsi_col)
        if not yrsi.isna().all():
            yrsi = yrsi.loc[x.index]
            fig.add_trace(go.Scatter(
                x=x, y=yrsi, mode="markers",
                marker=dict(color=color_kurs, symbol=symbol_for_set, size=12, line=dict(color="black", width=1.6)),
                name=("gen RSI " if typ == "gen" else "neg RSI ") + prefix.replace("_", " "),
                legendgroup=group_key, showlegend=False, visible=initial_vis
            ), row=2, col=1)
        ymacd = _safe_series(res, macd_col)
        if not ymacd.isna().all():
            ymacd = ymacd.loc[x.index]
            fig.add_trace(go.Scatter(
                x=x, y=ymacd, mode="markers",
                marker=dict(color=color_kurs, symbol=symbol_for_set, size=12, line=dict(color="black", width=1.6)),
                name=("gen MACD " if typ == "gen" else "neg MACD ") + prefix.replace("_", " "),
                legendgroup=group_key, showlegend=False, visible=initial_vis
            ), row=3, col=1)

    for typ in ["gen", "neg_MACD"]:
        for prefix in ["Lower_Low", "Higher_Low"]:
            _plot_points(prefix, typ, "low")

# ---------- Referenz (Brute-Force) + Overlay ----------
def ref_bullish_divergences(df: pd.DataFrame, lookback: int) -> pd.DataFrame:
    ts = _ts_series(df); low = df["low"].to_numpy()
    rsi = df["RSI"].to_numpy() if "RSI" in df.columns else np.full(len(df), np.nan)
    is_min = np.zeros(len(df), dtype=bool); is_min[1:-1] = (low[1:-1] < low[:-2]) & (low[1:-1] <= low[2:])
    events = []; idxs = np.where(is_min)[0]
    for k in range(1, len(idxs)):
        i1, i2 = idxs[k-1], idxs[k]; d = i2 - i1
        if 2 <= d <= max(2, lookback):
            if (low[i2] < low[i1]) and np.isfinite(rsi[i1]) and np.isfinite(rsi[i2]) and (rsi[i2] > rsi[i1]):
                events.append({"ts": ts.iloc[i2], "y_price": low[i2], "y_osci": rsi[i2]})
    return pd.DataFrame(events)

def ref_neg_macd_divergences(df: pd.DataFrame, lookback: int) -> pd.DataFrame:
    ts = _ts_series(df); low = df["low"].to_numpy()
    macd = df["macd"].to_numpy() if "macd" in df.columns else np.full(len(df), np.nan)
    is_min = np.zeros(len(df), dtype=bool); is_min[1:-1] = (low[1:-1] < low[:-2]) & (low[1:-1] <= low[2:])
    events = []; idxs = np.where(is_min)[0]
    for k in range(1, len(idxs)):
        i1, i2 = idxs[k-1], idxs[k]; d = i2 - i1
        if 2 <= d <= max(2, lookback):
            if (low[i2] < low[i1]) and np.isfinite(macd[i1]) and np.isfinite(macd[i2]) and (macd[i2] > macd[i1]):
                events.append({"ts": ts.iloc[i2], "y_price": low[i2], "y_osci": macd[i2]})
    return pd.DataFrame(events)

def add_reference_overlay(fig: go.Figure, df: pd.DataFrame, lookback: int, initial_visible=True):
    ref_gen = ref_bullish_divergences(df, lookback)
    ref_neg = ref_neg_macd_divergences(df, lookback)
    vis = True if initial_visible else "legendonly"
    def _add(ts, y, color, name, row):
        if len(ts) == 0: return
        fig.add_trace(go.Scatter(
            x=ts, y=y, mode="markers",
            marker=dict(color="#ffffff", symbol="star", size=13, line=dict(color=color, width=2.2)),
            name=name, legendgroup="REF", showlegend=True, visible=vis
        ), row=row, col=1)
    _add(ref_gen["ts"], ref_gen["y_price"], "#1f77b4", "REF gen (Kurs)", 1)
    _add(ref_neg["ts"], ref_neg["y_price"], "#d62728", "REF neg (Kurs)", 1)
    _add(ref_gen["ts"], ref_gen["y_osci"], "#1f77b4", "REF gen (RSI)", 2)
    _add(ref_neg["ts"], ref_neg["y_osci"], "#d62728", "REF neg (MACD)", 3)
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(color="#ffffff", symbol="star", size=12, line=dict(color="#000000", width=1.6)),
        name="REF – Brute Force", legendgroup="REF", legendgrouptitle_text="REF – Brute Force",
        showlegend=True, visible=vis
    ), row=1, col=1)
    return ref_gen, ref_neg

# -------------- Heatmap --------------
def build_heatmap_fig(df_counts: pd.DataFrame, title: str) -> go.Figure:
    pv_gen = df_counts.pivot(index="alpha", columns="gamma", values="n_div_gen").sort_index().sort_index(axis=1)
    pv_neg = df_counts.pivot(index="alpha", columns="gamma", values="n_div_neg_macd").sort_index().sort_index(axis=1)
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Bullish gen", "neg MACD"),
                        horizontal_spacing=0.12, specs=[[{"type": "heatmap"}, {"type": "heatmap"}]])
    fig.add_trace(go.Heatmap(
        z=pv_gen.values, x=[str(c) for c in pv_gen.columns], y=[str(i) for i in pv_gen.index],
        coloraxis="coloraxis", hovertemplate="α=%{y}<br>γ=%{x}<br>gen=%{z}<extra></extra>"
    ), row=1, col=1)
    fig.add_trace(go.Heatmap(
        z=pv_neg.values, x=[str(c) for c in pv_neg.columns], y=[str(i) for i in pv_neg.index],
        coloraxis="coloraxis", hovertemplate="α=%{y}<br>γ=%{x}<br>neg=%{z}<extra></extra>"
    ), row=1, col=2)
    fig.update_layout(title=title + " – Treffer-Heatmap",
                      coloraxis_colorscale="Viridis",
                      xaxis_title="γ", xaxis2_title="γ",
                      yaxis_title="α", yaxis2_title="α", height=520)
    for (pv, r, c) in [(pv_gen, 1, 1), (pv_neg, 1, 2)]:
        for i, a in enumerate(pv.index):
            for j, g in enumerate(pv.columns):
                val = pv.iloc[i, j]; txt = "0" if pd.isna(val) else str(int(val))
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
    used_ref = set(); tp = 0
    for ia in idx_algo:
        for ir in idx_ref:
            if ir in used_ref: continue
            if abs(ia - ir) <= tol_bars:
                used_ref.add(ir); tp += 1; break
    fp = len(idx_algo) - tp; fn = len(idx_ref) - tp
    return tp, fp, fn

# -------------- XLSX --------------
def save_xlsx(out_dir: Path, stamp: str, counts: pd.DataFrame,
              val_df: pd.DataFrame | None,
              alphas: list[float], gammas: list[float]):
    xlsx_path = out_dir / f"sensitivity_{stamp}.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as writer:
        counts.to_excel(writer, sheet_name="counts", index=False)
        pv_gen = counts.pivot(index="alpha", columns="gamma", values="n_div_gen").reindex(index=alphas, columns=gammas)
        pv_neg = counts.pivot(index="alpha", columns="gamma", values="n_div_neg_macd").reindex(index=alphas, columns=gammas)
        pv_gen.to_excel(writer, sheet_name="pivot_gen"); pv_neg.to_excel(writer, sheet_name="pivot_neg")
        wb = writer.book; fmt = wb.add_format({"num_format": "0", "align": "center"})
        cmap = {"type": "3_color_scale", "min_color": "#ffffff", "mid_color": "#9ecae1", "max_color": "#08519c"}
        for sheet, pv in [("pivot_gen", pv_gen), ("pivot_neg", pv_neg)]:
            ws = writer.sheets[sheet]; r0, c0 = 1, 1
            r1 = r0 + len(pv.index) - 1; c1 = c0 + len(pv.columns) - 1
            if r1 >= r0 and c1 >= c0:
                ws.set_column(c0, c1, 10, fmt); ws.conditional_format(r0, c0, r1, c1, cmap)
        if val_df is not None and not val_df.empty:
            val_df.to_excel(writer, sheet_name="validation", index=False)
    print(f"[OK] XLSX gespeichert: {xlsx_path}")

# -------------- Master-Toggle (Buttons) --------------
def add_master_toggle_buttons(fig: go.Figure):
    n = len(fig.data)
    # Indizes klassifizieren
    idx_variants = [i for i, tr in enumerate(fig.data) if getattr(tr, "legendgroup", None) not in (None, "REF")]
    idx_ref      = [i for i, tr in enumerate(fig.data) if getattr(tr, "legendgroup", None) == "REF"]
    # Hilfsfunktion Sichtbarkeitsvektor
    def vis_vec(all_on=False, all_off=False, only_variants=False, only_ref=False):
        v = [False]*n
        if all_on:
            v = [True]*n
        elif all_off:
            v = [False]*n
        elif only_variants:
            for i in idx_variants: v[i] = True
        elif only_ref:
            for i in idx_ref: v[i] = True
        else:
            v = [True]*n
        return v
    fig.update_layout(
        updatemenus=[dict(
            type="buttons", direction="right", x=0.5, xanchor="center", y=1.13, yanchor="top",
            buttons=[
                dict(label="Alle an", method="update", args=[{"visible": vis_vec(all_on=True)}]),
                dict(label="Alle aus", method="update", args=[{"visible": vis_vec(all_off=True)}]),
                dict(label="Nur Varianten", method="update", args=[{"visible": vis_vec(only_variants=True)}]),
                dict(label="Nur Referenz", method="update", args=[{"visible": vis_vec(only_ref=True)}]),
            ]
        )]
    )

# ---------------- Per-File Pipeline ----------------
def run_for_file(input_path: Path, args, alphas: list[float], gammas: list[float]):
    if not input_path.exists():
        print("[ERROR] Datei nicht gefunden:", input_path); return
    df = load_market_data(input_path, args.engine)
    print(f"[OK] {input_path.name}: Daten geladen ({len(df)} Zeilen)")
    df = Initialize_RSI_EMA_MACD(df); 
    if df is None: print("[ERROR] Initialize_RSI_EMA_MACD lieferte None."); return
    Local_Max_Min(df); print("[OK] Indikatoren + LM_* erstellt.")
    title = f"{args.title} – {input_path.stem}"
    fig_overlay = make_base_figure(df, title)
    if len(alphas)*len(gammas) > len(PARAM_SYMBOLS):
        print("[WARN] >25 Kombinationen – Formen werden recycelt.")
    sym_idx = 0; rows = []
    for a in alphas:
        for g in gammas:
            symbol_for_set = PARAM_SYMBOLS[sym_idx % len(PARAM_SYMBOLS)]; sym_idx += 1
            res = CBullDivg_analysis(df.copy(), args.lookback, a, g)
            add_markers_for_run(fig_overlay, res, df, a, g, symbol_for_set, initial_visible=not args.no_variants)
            n_gen = int(res["CBullD_gen"].sum()) if "CBullD_gen" in res.columns else 0
            n_neg = int(res["CBullD_neg_MACD"].sum()) if "CBullD_neg_MACD" in res.columns else 0
            rows.append({"alpha": a, "gamma": g, "lookback": args.lookback,
                         "n_div_gen": n_gen, "n_div_neg_macd": n_neg})
    df_counts = pd.DataFrame(rows)
    ref_gen_df, ref_neg_df = add_reference_overlay(fig_overlay, df, args.lookback, initial_visible=not args.no_ref)
    if args.master_toggle:
        add_master_toggle_buttons(fig_overlay)
    fig_heatmap = build_heatmap_fig(df_counts, title)

    out_dir = PROJECT_ROOT / "results" / "sensitivity_overlay"; out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S"); base = f"{input_path.stem}_{stamp}"
    if args.save_csv:
        (out_dir / f"overlay_counts_{base}.csv").write_text(df_counts.to_csv(index=False), encoding="utf-8")
    if args.save_html:
        fig_overlay.write_html(str(out_dir / f"overlay_plot_{base}.html"))
        fig_heatmap.write_html(str(out_dir / f"overlay_heatmap_{base}.html"))
    val_df = None
    if args.validate:
        base_ts = _ts_series(df)
        ts_ref_gen = ref_gen_df["ts"] if not ref_gen_df.empty else pd.Series(dtype="datetime64[ns, UTC]")
        ts_ref_neg = ref_neg_df["ts"] if not ref_neg_df.empty else pd.Series(dtype="datetime64[ns, UTC]")
        vals = []
        for a in alphas:
            for g in gammas:
                res = CBullDivg_analysis(df.copy(), args.lookback, a, g)
                ts_algo_gen = extract_algo_events(res, "gen")
                ts_algo_neg = extract_algo_events(res, "neg_MACD")
                tp_g, fp_g, fn_g = match_events(ts_algo_gen, ts_ref_gen, args.match_tol_bars, base_ts)
                tp_n, fp_n, fn_n = match_events(ts_algo_neg, ts_ref_neg, args.match_tol_bars, base_ts)
                prec_g = tp_g/(tp_g+fp_g) if (tp_g+fp_g)>0 else 0.0
                rec_g  = tp_g/(tp_g+fn_g) if (tp_g+fn_g)>0 else 0.0
                f1_g   = 2*prec_g*rec_g/(prec_g+rec_g) if (prec_g+rec_g)>0 else 0.0
                prec_n = tp_n/(tp_n+fp_n) if (tp_n+fp_n)>0 else 0.0
                rec_n  = tp_n/(tp_n+fn_n) if (tp_n+fn_n)>0 else 0.0
                f1_n   = 2*prec_n*rec_n/(prec_n+rec_n) if (prec_n+rec_n)>0 else 0.0
                vals.append({"file": input_path.name, "alpha": a, "gamma": g, "lookback": args.lookback,
                             "tp_gen": tp_g, "fp_gen": fp_g, "fn_gen": fn_g, "precision_gen": prec_g, "recall_gen": rec_g, "f1_gen": f1_g,
                             "tp_neg": tp_n, "fp_neg": fp_n, "fn_neg": fn_n, "precision_neg": prec_n, "recall_neg": rec_n, "f1_neg": f1_n,
                             "ref_gen": len(ts_ref_gen), "ref_neg": len(ts_ref_neg)})
        val_df = pd.DataFrame(vals)
        val_df.to_csv(out_dir / f"validation_{base}.csv", index=False)
    if args.save_xlsx:
        save_xlsx(out_dir, base, df_counts, val_df, alphas, gammas)
    fig_overlay.show(); fig_heatmap.show()

# ---------------- Main ----------------
def main():
    args = parse_args()
    if args.alphas: alphas = parse_list_floats(args.alphas)
    elif args.alpha_lin: alphas = parse_lin(args.alpha_lin)
    else: alphas = parse_lin("0.04:0.24:5")
    if args.gammas: gammas = parse_list_floats(args.gammas)
    elif args.gamma_lin: gammas = parse_lin(args.gamma_lin)
    else: gammas = parse_lin("2.0:5.0:5")
    if args.pick or not args.input:
        paths = pick_files_via_explorer()
        if not paths: print("[INFO] Keine Dateien gewählt. Abbruch."); return
    else:
        paths = [Path(args.input)]
    for p in paths:
        run_for_file(p, args, alphas, gammas)

if __name__ == "__main__":
    main()
