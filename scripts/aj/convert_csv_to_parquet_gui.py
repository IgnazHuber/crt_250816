# -*- coding: utf-8 -*-
"""
convert_csv_to_parquet_gui.py – Wähle mehrere CSVs und konvertiere zu Parquet
"""

import json
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import polars as pl
except ImportError:
    pl = None

# Speicherort für zuletzt verwendetes Verzeichnis
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"
CONFIG_DIR.mkdir(parents=True, exist_ok=True)
RUNTIME_JSON = CONFIG_DIR / "runtime_converter.json"


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


def convert_csv_file(csv_path: Path, engine: str = "polars") -> Path:
    parquet_path = csv_path.with_suffix(".parquet")

    if (
        parquet_path.exists()
        and csv_path.stat().st_mtime <= parquet_path.stat().st_mtime
    ):
        print(f"[SKIP] {csv_path.name} (bereits aktuell)")
        return parquet_path

    print(f"[INFO] Konvertiere: {csv_path.name}")
    t0 = time.perf_counter()

    if engine == "polars":
        if pl is None:
            raise ImportError("polars ist nicht installiert.")
        df = pl.read_csv(str(csv_path))
        df.write_parquet(str(parquet_path))
        n_rows, n_cols = df.shape
    else:
        if pd is None:
            raise ImportError("pandas ist nicht installiert.")
        df = pd.read_csv(str(csv_path))
        df.to_parquet(str(parquet_path), index=False, engine="pyarrow")
        n_rows, n_cols = df.shape

    dt = time.perf_counter() - t0
    print(
        f"[OK] {csv_path.name} → {parquet_path.name} ({n_rows}x{n_cols}) in {dt:.2f}s"
    )
    return parquet_path


def select_and_convert_csv_files(engine: str = "polars"):
    initial_dir = _load_last_dir()

    root = tk.Tk()
    root.withdraw()
    file_paths = filedialog.askopenfilenames(
        title="Wähle eine oder mehrere CSV-Dateien",
        filetypes=[("CSV Dateien", "*.csv")],
        initialdir=initial_dir,
    )
    root.destroy()

    if not file_paths:
        print("[ABBRUCH] Keine Datei gewählt.")
        return

    print(f"[INFO] {len(file_paths)} Datei(en) gewählt.")
    last_dir = Path(file_paths[0]).parent
    _save_last_dir(str(last_dir))

    for path_str in file_paths:
        csv_path = Path(path_str)
        try:
            convert_csv_file(csv_path, engine)
        except Exception as e:
            print(f"[ERROR] Fehler bei {csv_path.name}: {e}")

    messagebox.showinfo(
        "Konvertierung abgeschlossen", "Alle Dateien wurden verarbeitet."
    )


if __name__ == "__main__":
    select_and_convert_csv_files(engine="polars")
