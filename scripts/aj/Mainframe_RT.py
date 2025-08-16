# -*- coding: utf-8 -*-
"""
Mainframe_RT.py – Erweiterte Analyseumgebung

Funktionen:
- Datei-Explorer (CSV/Parquet) mit persistenter "LastDir"
- CSV → Parquet Konvertierung (nur wenn nötig)
- Parametrisierung der bisherigen Analyse-Konstanten
- Engine-Wahl: Polars (default) oder Pandas
- Integration: CBullDivg_analysis aus CBullDivg_Analysis_vectorized.py
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

# Projektverzeichnis & Konfigpfad
THIS_FILE = Path(__file__).resolve()
AJ_DIR = THIS_FILE.parent
PROJECT_ROOT = AJ_DIR.parents[1]
CONFIG_DIR = PROJECT_ROOT / "config"
CONFIG_DIR.mkdir(parents=True, exist_ok=True)
RUNTIME_JSON = CONFIG_DIR / "runtime.json"

# Scripts\aj zum Pfad hinzufügen
if str(AJ_DIR) not in sys.path:
    sys.path.insert(0, str(AJ_DIR))


def _load_last_dir() -> Optional[str]:
    if RUNTIME_JSON.exists():
        try:
            data = json.loads(RUNTIME_JSON.read_text(encoding="utf-8"))
            last_dir = data.get("last_dir")
            if last_dir and Path(last_dir).exists():
                return last_dir
        except Exception:
            pass
    return None


def _save_last_dir(path: str) -> None:
    try:
        payload = {"last_dir": str(path)}
        RUNTIME_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception:
        pass


def select_input_file() -> Optional[Path]:
    initdir = _load_last_dir() or str(PROJECT_ROOT)
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
        df = pl.read_csv(str(csv_path))
        df.write_parquet(str(parquet_path))
    else:
        print(f"[INFO] Parquet-Datei vorhanden: {parquet_path.name}")
    return parquet_path


def load_market_data(path: Path, engine: str = "polars"):
    if path.suffix.lower() == ".csv":
        path = ensure_parquet_from_csv(path)
    if engine == "polars":
        if pl is None:
            raise RuntimeError("Polars nicht installiert.")
        lf = pl.scan_parquet(str(path))
        lf = lf.with_columns(pl.col("timestamp").cast(pl.Datetime)).sort("timestamp")
        return lf.collect()
    else:
        if pd is None:
            raise RuntimeError("Pandas nicht installiert.")
        df = pd.read_parquet(str(path))
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.sort_values("timestamp")
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
    return p.parse_args()


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

    CBullDivg_analysis = _import_CBullDivg()
    if CBullDivg_analysis is None:
        print("[ERROR] Analysefunktion nicht gefunden.")
        sys.exit(1)

    try:
        print("[INFO] Starte CBullDivg_analysis(...)")
        result = CBullDivg_analysis(df, args.lookback, args.alpha, args.gamma)
        print("[OK] Analyse abgeschlossen.")
    except Exception as e:
        print(f"[ERROR] Analysefehler: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
