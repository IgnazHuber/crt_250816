# -*- coding: utf-8 -*-
"""
Mainframe_RT.py – Erweiterte Analyseumgebung mit Sensitivitätsanalyse

Funktionen:
- Datei-Explorer (CSV/Parquet) mit persistenter "LastDir"
- CSV → Parquet Konvertierung (nur bei Bedarf)
- Indikatorinitialisierung (RSI/EMA/MACD)
- Lokale Extrema (LM_* Spalten)
- Analyse via CBullDivg_analysis
- Optionale Sensitivitätsstudie (CSV/XLSX-Export)
"""

import argparse
import json
import sys
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog
from typing import Optional

try:
    import polars as pl
except ImportError:
    pl = None

try:
    import pandas as pd
except ImportError:
    pd = None

# Projektpfade
THIS_FILE = Path(__file__).resolve()
AJ_DIR = THIS_FILE.parent
PROJECT_ROOT = AJ_DIR.parents[1]
CONFIG_DIR = PROJECT_ROOT / "config"
CONFIG_DIR.mkdir(parents=True, exist_ok=True)
RUNTIME_JSON = CONFIG_DIR / "runtime.json"


def _load_last_dir() -> str:
    if RUNTIME_JSON.exists():
        try:
            data = json.loads(RUNTIME_JSON.read_text(encoding="utf-8"))
            last_dir = data.get("last_dir")
            if last_dir and Path(last_dir).exists():
                return last_dir
        except Exception:
            pass
    return str(PROJECT_ROOT / "data" / "raw")


def _save_last_dir(path: str):
    try:
        payload = {"last_dir": str(path)}
        RUNTIME_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception:
        pass


def select_input_file() -> Optional[Path]:
    initdir = _load_last_dir()
    root = tk.Tk()
    root.withdraw()
    filetypes = [("Daten", "*.parquet *.csv"), ("Alle Dateien", "*.*")]
    path = filedialog.askopenfilename(
        parent=root,
        title="Wähle Rohdaten (Parquet/CSV)",
        initialdir=initdir,
        filetypes=filetypes,
    )
    root.destroy()
    if path:
        p = Path(path)
        _save_last_dir(str(p.parent))
        return p
    return None


def ensure_parquet_from_csv(csv_path: Path) -> Path:
    parquet_path = csv_path.with_suffix(".parquet")
    if (
        not parquet_path.exists()
        or csv_path.stat().st_mtime > parquet_path.stat().st_mtime
    ):
        if pl is None:
            raise RuntimeError("Polars nicht installiert.")
        print(f"[INFO] Konvertiere {csv_path.name} → {parquet_path.name}")
        df_pl = pl.read_csv(str(csv_path))
        df_pl.write_parquet(str(parquet_path))
    else:
        print(f"[INFO] Parquet-Datei vorhanden: {parquet_path.name}")
    return parquet_path


def load_market_data(path: Path, engine: str = "polars") -> "pd.DataFrame":
    """Lädt Daten und liefert **Pandas**-DataFrame (Analysemodule erwarten Pandas)."""
    if path.suffix.lower() == ".csv":
        # CSV optional zu Parquet heben (schneller beim nächsten Lauf), aber wir liefern am Ende Pandas
        path = ensure_parquet_from_csv(path)

    if engine == "polars":
        if pl is None:
            raise RuntimeError("Polars nicht installiert.")
        lf = pl.scan_parquet(str(path))
        # Schema abfragen ohne Warnung
        colnames = lf.collect_schema().keys()
        ts_col = (
            "timestamp"
            if "timestamp" in colnames
            else ("date" if "date" in colnames else None)
        )
        if ts_col is None:
            raise RuntimeError(
                f"Keine Zeitspalte gefunden (erwartet: 'timestamp' oder 'date'). Vorhanden: {colnames}"
            )

        lf = lf.sort(ts_col)
        df_pl = lf.collect()
        df = df_pl.to_pandas()  # >>> Pandas für Analyse
        return df

    else:
        if pd is None:
            raise RuntimeError("Pandas nicht installiert.")
        df = pd.read_parquet(str(path))
        # Für Sortierung (Änderung an 'date' ist nicht nötig, CBullDivg_analysis parsed selbst)
        sort_key = None
        if "timestamp" in df.columns:
            sort_key = "timestamp"
        elif "date" in df.columns:
            # temporärer Sortierschlüssel ohne die 'date'-Spalte zu verändern
            _ts = pd.to_datetime(df["date"], utc=True, errors="coerce")
            df = df.assign(_ts=_ts).sort_values("_ts").drop(columns=["_ts"])
        return df


def _import_CBullDivg():
    try:
        mod = __import__(
            "CBullDivg_Analysis_vectorized", fromlist=["CBullDivg_analysis"]
        )
        return getattr(mod, "CBullDivg_analysis", None)
    except Exception as e:
        print(f"[ERROR] Import fehlgeschlagen: {e}", file=sys.stderr)
        return None


def parse_args():
    p = argparse.ArgumentParser("Mainframe Divergenz Analyse")
    p.add_argument("--input", type=str, help="Pfad zur Datei (CSV/Parquet)")
    p.add_argument("--engine", choices=["polars", "pandas"], default="polars")
    p.add_argument("--lookback", type=int, default=5)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--gamma", type=float, default=3.25)
    p.add_argument(
        "--sensitivity", action="store_true", help="Starte Sensitivitätsanalyse"
    )
    return p.parse_args()


def run_sensitivity(df: "pd.DataFrame", CBullDivg_analysis, out_dir: Path):
    lookback_range = [3, 5, 7]
    alpha_range = [0.05, 0.1, 0.2]
    gamma_range = [2.5, 3.0, 3.5]

    rows = len(lookback_range) * len(alpha_range) * len(gamma_range)
    print(f"[INFO] Starte Sensitivitätsanalyse über {rows} Kombinationen...")

    results = []
    for lb in lookback_range:
        for alpha in alpha_range:
            for gamma in gamma_range:
                try:
                    t0 = time.perf_counter()
                    res = CBullDivg_analysis(df.copy(), lb, alpha, gamma)
                    dt = time.perf_counter() - t0

                    # Divergenzzähler korrekt ermitteln
                    n_gen = (
                        int(res["CBullD_gen"].sum())
                        if "CBullD_gen" in res.columns
                        else -1
                    )
                    n_neg = (
                        int(res["CBullD_neg_MACD"].sum())
                        if "CBullD_neg_MACD" in res.columns
                        else -1
                    )

                    results.append(
                        {
                            "lookback": lb,
                            "alpha": alpha,
                            "gamma": gamma,
                            "n_div_gen": n_gen,
                            "n_div_neg_macd": n_neg,
                            "runtime_sec": round(dt, 4),
                        }
                    )
                    print(
                        f"[✓] lb={lb}, alpha={alpha}, gamma={gamma} → gen={n_gen}, neg_macd={n_neg} in {dt:.2f}s"
                    )

                except Exception as e:
                    results.append(
                        {
                            "lookback": lb,
                            "alpha": alpha,
                            "gamma": gamma,
                            "n_div_gen": -1,
                            "n_div_neg_macd": -1,
                            "runtime_sec": -1,
                            "error": str(e),
                        }
                    )
                    print(f"[✗] Fehler bei lb={lb}, alpha={alpha}, gamma={gamma} → {e}")

    if pd is None:
        raise RuntimeError("Für den Export wird pandas benötigt.")

    df_result = pd.DataFrame(results)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / f"sensitivity_results_{timestamp}.csv"
    xlsx_path = out_dir / f"sensitivity_results_{timestamp}.xlsx"

    df_result.to_csv(csv_path, index=False)
    df_result.to_excel(xlsx_path, index=False)

    print(f"[OK] Ergebnisse gespeichert:\n  - {csv_path}\n  - {xlsx_path}")


def main():
    args = parse_args()
    print("[INFO] Starte Analyse mit Parametern:", vars(args))

    input_path = Path(args.input) if args.input else select_input_file()
    if not input_path or not input_path.exists():
        print("[ERROR] Datei nicht gefunden.")
        sys.exit(1)

    t0 = time.perf_counter()
    df = load_market_data(input_path, args.engine)
    print(f"[OK] Daten geladen ({len(df)} Zeilen) in {time.perf_counter()-t0:.2f}s")

    # 1) Indikatoren berechnen (nutzt deine gelieferte Funktion/Signaturen)
    try:
        from Initialize_RSI_EMA_MACD import Initialize_RSI_EMA_MACD

        df = Initialize_RSI_EMA_MACD(df)
        if df is None:
            raise RuntimeError("Initialize_RSI_EMA_MACD(...) lieferte None.")
        print("[OK] Indikatoren berechnet (RSI/EMA/MACD).")
    except Exception as e:
        print(f"[ERROR] Fehler bei Indikatorberechnung: {e}")
        sys.exit(1)

    # 2) Lokale Extrema berechnen (in-place; erzeugt LM_* Spalten)
    try:
        from Local_Maximas_Minimas import Local_Max_Min

        Local_Max_Min(df)  # in-place
        print("[OK] Lokale Extrema (LM_*) berechnet.")
    except Exception as e:
        print(f"[ERROR] Fehler bei lokalen Extrema: {e}")
        sys.exit(1)

    # 3) Analysefunktion laden & ausführen
    CBullDivg_analysis = _import_CBullDivg()
    if CBullDivg_analysis is None:
        print("[ERROR] Analysefunktion nicht gefunden.")
        sys.exit(1)

    try:
        if args.sensitivity:
            output_dir = PROJECT_ROOT / "results" / "sensitivity"
            run_sensitivity(df, CBullDivg_analysis, output_dir)
        else:
            print("[INFO] Starte CBullDivg_analysis(...)")
            res = CBullDivg_analysis(df.copy(), args.lookback, args.alpha, args.gamma)
            # Ergebnis sauber mergen für Export/Weiterverarbeitung
            out_df = df.copy()
            for col in res.columns:
                out_df[col] = res[col].values

            n_gen = int(out_df.get("CBullD_gen", pd.Series([0] * len(out_df))).sum())
            n_neg = int(
                out_df.get("CBullD_neg_MACD", pd.Series([0] * len(out_df))).sum()
            )
            print(f"[OK] Analyse abgeschlossen: gen={n_gen}, neg_macd={n_neg}")

            timestamp = time.strftime("%Y%m%d_%H%M%S")
            outdir = PROJECT_ROOT / "results" / "divergences"
            outdir.mkdir(parents=True, exist_ok=True)
            out_df.to_csv(outdir / f"divergences_{timestamp}.csv", index=False)
            out_df.to_excel(outdir / f"divergences_{timestamp}.xlsx", index=False)
            print(
                f"[OK] Ergebnis gespeichert: {outdir / f'divergences_{timestamp}.csv'}"
            )
    except Exception as e:
        print(f"[ERROR] Analysefehler: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
