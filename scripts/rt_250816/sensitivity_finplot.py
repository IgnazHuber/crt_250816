import os

# Qt/HighDPI-Workarounds: vor finplot/PyQt import setzen
os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "0"
os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "0"

# -*- coding: utf-8 -*-
"""
sensitivity_finplot.py – Visuelle Sensitivität von CBullDivg_analysis über (alpha, gamma)
Darstellung wie in Mainframe.py mit finplot:
- Candles, EMAs, RSI, MACD-Hist
- Marker pro (alpha, gamma): Farbe je Kombination, Symbol je Signaltyp (x=gen, o=neg)

CLI:
  --input <file>                  CSV/Parquet
  --engine polars|pandas          Loader (Analyse auf Pandas)
  --lookback 5                    Fixer Lookback (nur alpha,gamma variieren)
  --alphas 0.08,0.1,0.12          Liste α
  --gammas 3.0,3.25,3.5           Liste γ
  --title "BTCUSD 1D"             Plot-Titel
  --save-csv                      Zähler je Kombination als CSV speichern
"""

import argparse
import sys
from pathlib import Path

# Pfade/Imports
THIS_FILE = Path(__file__).resolve()
RT_DIR = THIS_FILE.parent
PROJECT_ROOT = RT_DIR.parents[1]
AJ_DIR = PROJECT_ROOT / "scripts" / "aj"
if str(AJ_DIR) not in sys.path:
    sys.path.insert(0, str(AJ_DIR))

# Externe Libs
import numpy as np
import pandas as pd

try:
    import polars as pl
except ImportError:
    pl = None

import finplot as fplt
# Analyse-Module aus aj
from CBullDivg_Analysis_vectorized import CBullDivg_analysis
from Initialize_RSI_EMA_MACD import Initialize_RSI_EMA_MACD
from Local_Maximas_Minimas import Local_Max_Min


# ----------- Loader (liefert Pandas) -----------
def ensure_parquet_from_csv(csv_path: Path) -> Path:
    parquet_path = csv_path.with_suffix(".parquet")
    if (not parquet_path.exists()) or (
        csv_path.stat().st_mtime > parquet_path.stat().st_mtime
    ):
        if pl is None:
            raise RuntimeError(
                "Polars nicht installiert (für CSV->Parquet erforderlich)."
            )
        df_pl = pl.read_csv(str(csv_path))
        df_pl.write_parquet(str(parquet_path))
    return parquet_path


def load_market_data(path: Path, engine: str = "polars") -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        path = ensure_parquet_from_csv(path)

    if engine == "polars":
        if pl is None:
            raise RuntimeError("Polars nicht installiert.")
        lf = pl.scan_parquet(str(path))
        colnames = lf.collect_schema().keys()
        ts_col = (
            "timestamp"
            if "timestamp" in colnames
            else ("date" if "date" in colnames else None)
        )
        if ts_col is None:
            raise RuntimeError(f"Keine Zeitspalte gefunden. Vorhanden: {colnames}")
        df = lf.sort(ts_col).collect().to_pandas()
        return df
    else:
        df = pd.read_parquet(str(path))
        # nur sortieren, nicht umbenennen
        if "timestamp" in df.columns:
            df = df.sort_values("timestamp")
        elif "date" in df.columns:
            _ts = pd.to_datetime(df["date"], utc=True, errors="coerce")
            df = df.assign(_ts=_ts).sort_values("_ts").drop(columns=["_ts"])
        return df


# ----------- CLI -----------
def parse_args():
    p = argparse.ArgumentParser("CBullDivg Sensitivitäts-Viewer (finplot)")
    p.add_argument(
        "--input", type=str, required=True, help="Pfad zur Datei (CSV/Parquet)"
    )
    p.add_argument("--engine", choices=["polars", "pandas"], default="polars")
    p.add_argument(
        "--lookback", type=int, default=5, help="Fixer Lookback (nur α,γ variieren)"
    )
    p.add_argument(
        "--alphas",
        type=str,
        required=True,
        help="Kommagetrennte Liste, z.B. 0.08,0.1,0.12",
    )
    p.add_argument(
        "--gammas",
        type=str,
        required=True,
        help="Kommagetrennte Liste, z.B. 3.0,3.25,3.5",
    )
    p.add_argument(
        "--title", type=str, default="Divergenz-Sensitivität", help="Plot-Titel"
    )
    p.add_argument(
        "--save-csv", action="store_true", help="Zähler je Kombination als CSV sichern"
    )
    return p.parse_args()


def parse_list_floats(s: str):
    return [float(x.strip()) for x in s.split(",") if x.strip()]


# ----------- Plot Helper -----------
def plot_base_panels(df: pd.DataFrame, title: str):
    fplt.background = fplt.odd_plot_background = "#242320"
    fplt.cross_hair_color = "#eefa"

    ax1, ax2, ax3 = fplt.create_plot(title, rows=3)

    # Datumsspalte für finplot
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], format="mixed", utc=True)
    elif "timestamp" in df.columns:
        df["date"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

    # Candles (OCHL)
    candles = df[["date", "open", "close", "high", "low"]]
    fplt.candlestick_ochl(candles, ax=ax1)

    # RSI
    if "RSI" in df.columns and not df["RSI"].isna().all():
        fplt.plot(df["RSI"], color="#AAAAAA", width=2, ax=ax2, legend="RSI")
        fplt.set_y_range(0, 100, ax=ax2)
        fplt.add_horizontal_line(70, ax=ax2, color="#777777")
        fplt.add_horizontal_line(30, ax=ax2, color="#777777")
    else:
        # robust: ax2 bleibt leer, aber das Layout steht
        pass

    # MACD-Hist (als „Volumen“-Balken)
    if "macd_histogram" in df.columns and not df["macd_histogram"].isna().all():
        fplt.volume_ocv(
            df[["date", "open", "close", "macd_histogram"]],
            ax=ax3,
            colorfunc=fplt.strength_colorfilter,
        )

    # EMAs (falls vorhanden) – Bugfix: nicht getattr(df, name), sondern df[name]
    for name in ["EMA_20", "EMA_50", "EMA_100", "EMA_200"]:
        if name in df.columns and not df[name].isna().all():
            fplt.plot(df[name], ax=ax1, legend=name)

    return ax1, ax2, ax3


def plot_markers_for_run(
    df_run: pd.DataFrame, ax1, ax2, ax3, color: str, alpha_val: float, gamma_val: float
):
    """Zeichnet Marker für einen Run: 'x' = gen (bullish), 'o' = neg_MACD; Farbe kodiert (α,γ)."""

    label_gen = f"a={alpha_val:g} g={gamma_val:g} gen"
    label_neg = f"a={alpha_val:g} g={gamma_val:g} neg"

    n = len(df_run)

    # Bullish Divergence (gen)
    if "CBullD_gen" in df_run.columns and df_run["CBullD_gen"].sum() > 0:
        for i in range(2, n):
            if df_run["CBullD_gen"].iat[i] == 1:
                t_lo = pd.to_datetime(
                    df_run.get("CBullD_Lower_Low_date_gen", pd.NaT)
                ).iat[i]
                t_hi = pd.to_datetime(
                    df_run.get("CBullD_Higher_Low_date_gen", pd.NaT)
                ).iat[i]
                y_lo = df_run.get("CBullD_Lower_Low_gen", pd.Series([np.nan] * n)).iat[
                    i
                ]
                y_hi = df_run.get("CBullD_Higher_Low_gen", pd.Series([np.nan] * n)).iat[
                    i
                ]

                fplt.plot(t_lo, y_lo, style="x", ax=ax1, color=color, legend=label_gen)
                fplt.plot(t_hi, y_hi, style="x", ax=ax1, color=color)

                # RSI-Paneel (falls vorhanden)
                if (
                    "CBullD_Lower_Low_RSI_gen" in df_run.columns
                    and "CBullD_Higher_Low_RSI_gen" in df_run.columns
                ):
                    rsi_lo = df_run["CBullD_Lower_Low_RSI_gen"].iat[i]
                    rsi_hi = df_run["CBullD_Higher_Low_RSI_gen"].iat[i]
                    fplt.plot(t_lo, rsi_lo, style="x", ax=ax2, color=color)
                    fplt.plot(t_hi, rsi_hi, style="x", ax=ax2, color=color)

                # MACD-Paneel (falls vorhanden)
                if (
                    "CBullD_Lower_Low_MACD_gen" in df_run.columns
                    and "CBullD_Higher_Low_MACD_gen" in df_run.columns
                ):
                    macd_lo = df_run["CBullD_Lower_Low_MACD_gen"].iat[i]
                    macd_hi = df_run["CBullD_Higher_Low_MACD_gen"].iat[i]
                    fplt.plot(t_lo, macd_lo, style="x", ax=ax3, color=color)
                    fplt.plot(t_hi, macd_hi, style="x", ax=ax3, color=color)

    # Negative MACD-Divergenz (neg)
    if "CBullD_neg_MACD" in df_run.columns and df_run["CBullD_neg_MACD"].sum() > 0:
        for i in range(2, n):
            if df_run["CBullD_neg_MACD"].iat[i] == 1:
                t_lo = pd.to_datetime(
                    df_run.get("CBullD_Lower_Low_date_neg_MACD", pd.NaT)
                ).iat[i]
                t_hi = pd.to_datetime(
                    df_run.get("CBullD_Higher_Low_date_neg_MACD", pd.NaT)
                ).iat[i]
                y_lo = df_run.get(
                    "CBullD_Lower_Low_neg_MACD", pd.Series([np.nan] * n)
                ).iat[i]
                y_hi = df_run.get(
                    "CBullD_Higher_Low_neg_MACD", pd.Series([np.nan] * n)
                ).iat[i]

                fplt.plot(t_lo, y_lo, style="o", ax=ax1, color=color, legend=label_neg)
                fplt.plot(t_hi, y_hi, style="o", ax=ax1, color=color)

                # RSI-Paneel (falls vorhanden)
                if (
                    "CBullD_Lower_Low_RSI_neg_MACD" in df_run.columns
                    and "CBullD_Higher_Low_RSI_neg_MACD" in df_run.columns
                ):
                    rsi_lo = df_run["CBullD_Lower_Low_RSI_neg_MACD"].iat[i]
                    rsi_hi = df_run["CBullD_Higher_Low_RSI_neg_MACD"].iat[i]
                    fplt.plot(t_lo, rsi_lo, style="o", ax=ax2, color=color)
                    fplt.plot(t_hi, rsi_hi, style="o", ax=ax2, color=color)

                # MACD-Paneel (falls vorhanden)
                if (
                    "CBullD_Lower_Low_MACD_neg_MACD" in df_run.columns
                    and "CBullD_Higher_Low_MACD_neg_MACD" in df_run.columns
                ):
                    macd_lo = df_run["CBullD_Lower_Low_MACD_neg_MACD"].iat[i]
                    macd_hi = df_run["CBullD_Higher_Low_MACD_neg_MACD"].iat[i]
                    fplt.plot(t_lo, macd_lo, style="o", ax=ax3, color=color)
                    fplt.plot(t_hi, macd_hi, style="o", ax=ax3, color=color)


def main():
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        print("[ERROR] Datei nicht gefunden:", input_path)
        sys.exit(1)

    # Laden
    df = load_market_data(input_path, args.engine)
    print(f"[OK] Daten geladen: {len(df)} Zeilen")

    # Indikatoren + Extrema (einmalig!)
    df = Initialize_RSI_EMA_MACD(df)
    if df is None:
        print("[ERROR] Initialize_RSI_EMA_MACD lieferte None.")
        sys.exit(1)
    Local_Max_Min(df)
    print("[OK] Indikatoren + LM_* erstellt.")

    # Basisplots
    ax1, ax2, ax3 = plot_base_panels(df, args.title)

    # Parameterlisten
    alphas = parse_list_floats(args.alphas)
    gammas = parse_list_floats(args.gammas)

    # Farbzyklus pro Kombination (Karte: (alpha,gamma) -> Farbe)
    palette = [
        "#e6194B",
        "#3cb44b",
        "#0082c8",
        "#f58231",
        "#911eb4",
        "#46f0f0",
        "#f032e6",
        "#d2f53c",
        "#fabed4",
        "#008080",
        "#f0e442",
        "#a65628",
        "#4daf4a",
        "#377eb8",
        "#984ea3",
    ]
    colors = {}
    idx = 0

    # Zähl-Output (optional)
    rows = []

    # Läufe (nur alpha & gamma variieren; lookback fix)
    for a in alphas:
        for g in gammas:
            color = colors.setdefault((a, g), palette[idx % len(palette)])
            idx += 1

            res = CBullDivg_analysis(df.copy(), args.lookback, a, g)
            # Marker zeichnen
            plot_markers_for_run(
                res, ax1, ax2, ax3, color=color, alpha_val=a, gamma_val=g
            )

            n_gen = int(res["CBullD_gen"].sum()) if "CBullD_gen" in res.columns else -1
            n_neg = (
                int(res["CBullD_neg_MACD"].sum())
                if "CBullD_neg_MACD" in res.columns
                else -1
            )
            rows.append(
                {
                    "alpha": a,
                    "gamma": g,
                    "lookback": args.lookback,
                    "n_div_gen": n_gen,
                    "n_div_neg_macd": n_neg,
                }
            )

            print(f"[OK] a={a}, g={g} → gen={n_gen}, neg={n_neg}")

    # Optional: CSV der Zähler
    if args.save_csv:
        out_dir = PROJECT_ROOT / "results" / "sensitivity_overlay"
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"overlay_counts_{stamp}.csv"
        pd.DataFrame(rows).to_csv(out_path, index=False)
        print(f"[OK] Overlay-Zähler gespeichert: {out_path}")

    # Anzeige
    fplt.show()


if __name__ == "__main__":
    main()
