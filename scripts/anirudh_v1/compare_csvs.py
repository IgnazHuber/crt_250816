#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_csvs.py — Präziser CSV-Vergleich mit Pandas

Funktionen:
- Exakter Zellenvergleich oder numerisch mit Toleranzen (rtol/atol)
- Optionaler Abgleich über Schlüsselspalten (Key-Join)
- Form-/Schema-Checks (Shape, Spalten, Dtypes)
- Ignoriere Spaltenreihenfolge (optional)
- Whitespace-Trim für Strings (optional)
- NA-Gleichheit (NaN==NaN) steuerbar
- Ausführlicher Unterschiedsreport (Konsole) + optional Excel (XLSX)

Beispiele:
    python compare_csvs.py a.csv b.csv
    python compare_csvs.py a.csv b.csv --float-rtol 1e-5 --float-atol 1e-8
    python compare_csvs.py a.csv b.csv --key-cols id,timestamp --report-xlsx diff.xlsx
    python compare_csvs.py a.csv b.csv --ignore-column-order --trim

Neu:
- Positionsargumente optional; ohne Args werden zwei Fallback-Pfade verwendet.
"""

from __future__ import annotations
import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


# -------------------------- Fallbacks (deine Pfade) --------------------------

FALLBACK_A = Path(r"c://Projekte/Anirudh/ETH/output_4hour_parquet/backtest_results_ETH_4hour_100perc_with_brokerage_Basis_412s.csv")
FALLBACK_B = Path(r"c://Projekte/Anirudh/ETH/output_4hour_parquet/backtest_results_ETH_4hour_100perc_with_brokerage_v03_pf4_w8_88s.csv")


# -------------------------- Utility --------------------------

def sniff_delimiter(path: Path, default: str = ",") -> str:
    """Ermittelt Trennzeichen robust aus den ersten ~8 KB, sonst default."""
    try:
        with path.open("r", newline="", encoding="utf-8") as f:
            sample = f.read(8192)
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        return dialect.delimiter
    except Exception:
        return default


def read_csv_smart(
    path: Path,
    sep: Optional[str],
    index_cols: Optional[List[str]],
    trim: bool,
) -> pd.DataFrame:
    if sep is None:
        sep = sniff_delimiter(path)

    df = pd.read_csv(
        path,
        sep=sep,
        engine="python",
        quoting=csv.QUOTE_MINIMAL,
    )

    # Index setzen (optional)
    if index_cols:
        missing = [c for c in index_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Indexspalten fehlen in {path.name}: {missing}")
        df = df.set_index(index_cols)

    # String-Trim (optional)
    if trim:
        obj_cols = df.select_dtypes(include=["object", "string"]).columns
        if len(obj_cols) > 0:
            df[obj_cols] = df[obj_cols].apply(lambda s: s.astype("string").str.strip())

    return df


def align_columns(
    a: pd.DataFrame, b: pd.DataFrame, ignore_column_order: bool
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], List[str], List[str]]:
    cols_a = list(a.columns)
    cols_b = list(b.columns)
    only_a = [c for c in cols_a if c not in cols_b]
    only_b = [c for c in cols_b if c not in cols_a]
    common = [c for c in cols_a if c in cols_b]

    if ignore_column_order:
        # Auf gemeinsame Spalten reduzieren und gleiche Reihenfolge (alphabetisch)
        common_sorted = sorted(common)
        a2 = a[common_sorted].copy()
        b2 = b[common_sorted].copy()
        return a2, b2, only_a, only_b, common_sorted
    else:
        # Gleiche Spaltenreihenfolge erzwingen gemäß A; B entsprechend anordnen
        if only_a or only_b:
            pass
        intersect = [c for c in cols_a if c in cols_b]
        a2 = a[intersect + only_a].copy()  # only_a am Ende (für Bericht)
        b2 = b[intersect + only_b].copy()
        return a2, b2, only_a, only_b, intersect


def comparable_dtypes(a: pd.Series, b: pd.Series) -> bool:
    # Erlaubt z. B. int64 vs float64 bei numerischem Vergleich
    if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
        return True
    return a.dtype == b.dtype


@dataclass
class DiffSummary:
    equal: bool
    reason: List[str]


# -------------------------- Vergleichslogik --------------------------

def compare_dataframes(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    *,
    float_rtol: float,
    float_atol: float,
    na_equal: bool,
) -> Tuple[bool, pd.DataFrame]:
    """
    Zellenweiser Vergleich:
    - Numerische Spalten via np.isclose(rtol/atol)
    - Nicht-numerische Spalten via exakt Gleich
    - NA-Gleichheit steuerbar
    Rückgabe:
        (alles_gleich, unterschiede_dataframe)
        unterschiede_dataframe: MultiIndex (Zeile, Spalte) mit Werten (left, right)
    """
    # Index-Ausrichtung
    if not df1.index.equals(df2.index):
        # Vereinheitlichen (inner Join auf Index)
        common_idx = df1.index.intersection(df2.index)
        df1c = df1.loc[common_idx]
        df2c = df2.loc[common_idx]
    else:
        df1c, df2c = df1, df2

    # Spalten-Ausrichtung (Schnittmenge)
    common_cols = [c for c in df1c.columns if c in df2c.columns]
    df1c = df1c[common_cols]
    df2c = df2c[common_cols]

    # Vergleich pro Spalte
    diffs_mask = pd.DataFrame(False, index=df1c.index, columns=common_cols)

    for col in common_cols:
        s1 = df1c[col]
        s2 = df2c[col]

        if not comparable_dtypes(s1, s2):
            # Sanfte Konvertierung für numerische Strings
            if pd.api.types.is_numeric_dtype(s1) and pd.api.types.is_object_dtype(s2):
                s2 = pd.to_numeric(s2, errors="coerce")
            elif pd.api.types.is_numeric_dtype(s2) and pd.api.types.is_object_dtype(s1):
                s1 = pd.to_numeric(s1, errors="coerce")

        if pd.api.types.is_numeric_dtype(s1) and pd.api.types.is_numeric_dtype(s2):
            left = s1.astype("float64")
            right = s2.astype("float64")
            both_na = left.isna() & right.isna()
            close = np.isclose(left, right, rtol=float_rtol, atol=float_atol, equal_nan=na_equal)
            mask = ~(close | (both_na if na_equal else False))
        else:
            if na_equal:
                eq = (s1 == s2) | (s1.isna() & s2.isna())
            else:
                eq = (s1 == s2)
            mask = ~eq

        diffs_mask[col] = mask

    if not diffs_mask.values.any():
        return True, pd.DataFrame()

    # Unterschiede aufbereiten
    where = np.where(diffs_mask.values)
    rows = df1c.index.to_numpy()[where[0]]
    cols = diffs_mask.columns.to_numpy()[where[1]]

    out = pd.DataFrame({"index": rows, "column": cols})
    out["left"] = [df1c.at[idx, col] for idx, col in zip(rows, cols)]
    out["right"] = [df2c.at[idx, col] for idx, col in zip(rows, cols)]
    out = out.set_index(["index", "column"])

    return False, out


def key_based_alignment(
    a: pd.DataFrame, b: pd.DataFrame, key_cols: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Liefert (A∩B aligned, B∩A aligned, only_in_A, only_in_B) basierend auf Schlüsselspalten."""
    for k in key_cols:
        if k not in a.columns or k not in b.columns:
            missing = [k for k in key_cols if k not in a.columns or k not in b.columns]
            raise ValueError(f"Key-Spalten fehlen: {missing}")

    a_keyed = a.set_index(key_cols)
    b_keyed = b.set_index(key_cols)

    common_idx = a_keyed.index.intersection(b_keyed.index)
    only_a_idx = a_keyed.index.difference(b_keyed.index)
    only_b_idx = b_keyed.index.difference(a_keyed.index)

    a_common = a_keyed.loc[common_idx]
    b_common = b_keyed.loc[common_idx]
    only_a = a_keyed.loc[only_a_idx]
    only_b = b_keyed.loc[only_b_idx]
    return a_common, b_common, only_a, only_b


# -------------------------- Reporting --------------------------

def schema_report(a: pd.DataFrame, b: pd.DataFrame) -> List[str]:
    msgs = []
    if a.shape != b.shape:
        msgs.append(f"- Unterschiedliche Form: A{a.shape} vs B{b.shape}")

    cols_a = list(a.columns)
    cols_b = list(b.columns)
    only_a = [c for c in cols_a if c not in cols_b]
    only_b = [c for c in cols_b if c not in cols_a]
    if only_a:
        msgs.append(f"- Spalten nur in A: {only_a}")
    if only_b:
        msgs.append(f"- Spalten nur in B: {only_b}")

    common = [c for c in cols_a if c in cols_b]
    dtype_diff = []
    for c in common:
        if a[c].dtype != b[c].dtype:
            dtype_diff.append((c, str(a[c].dtype), str(b[c].dtype)))
    if dtype_diff:
        msgs.append("- Unterschiedliche dtypes (Spalte, A, B): " + str(dtype_diff))

    return msgs


def print_summary(
    *, equal: bool, reasons: List[str], n_cell_diffs: int, n_row_diffs: int, only_a_rows: int, only_b_rows: int
) -> None:
    print("\n================ Vergleichs-Zusammenfassung ================")
    print(f"Identisch: {'JA' if equal else 'NEIN'}")
    if reasons:
        for r in reasons:
            print(r)
    if n_cell_diffs >= 0:
        print(f"- Zell-Differenzen: {n_cell_diffs}")
    if n_row_diffs >= 0:
        print(f"- Betroffene Zeilen (mit mindestens 1 Abweichung): {n_row_diffs}")
    if only_a_rows >= 0 or only_b_rows >= 0:
        print(f"- Nur in A (Key-basiert): {only_a_rows}")
        print(f"- Nur in B (Key-basiert): {only_b_rows}")
    print("============================================================\n")


def write_excel_report(
    path: Path,
    *,
    cell_diffs: Optional[pd.DataFrame],
    only_a: Optional[pd.DataFrame],
    only_b: Optional[pd.DataFrame],
    schema_msgs: List[str],
) -> None:
    with pd.ExcelWriter(path, engine="xlsxwriter") as xw:
        # Schema
        pd.DataFrame({"Schema-Differenzen": schema_msgs or ["(keine)"]}).to_excel(xw, sheet_name="schema", index=False)

        # Zellen-Diffs
        if cell_diffs is not None and not cell_diffs.empty:
            cd = cell_diffs.reset_index().rename(columns={"index": "row_index", "column": "column"})
            cd.to_excel(xw, sheet_name="cell_diffs", index=False)

        # Nur in A/B
        if only_a is not None and not only_a.empty:
            only_a.reset_index().to_excel(xw, sheet_name="only_in_A", index=False)
        if only_b is not None and not only_b.empty:
            only_b.reset_index().to_excel(xw, sheet_name="only_in_B", index=False)


# -------------------------- Main/CLI --------------------------

def _normalize_path(p: Optional[Path]) -> Path:
    """Normalisiert Pfade robust; nutzt Fallback, wenn None."""
    if p is None:
        raise ValueError("Pfad ist None")
    # Path normalisiert gemischte Slashes automatisch
    return Path(str(p))


def main():
    ap = argparse.ArgumentParser(description="CSV/Tabellenvergleich mit Toleranzen und Key-Abgleich.")
    # WICHTIG: Positionsargumente optional; Fallbacks als Defaults
    ap.add_argument("csv_a", nargs="?", type=Path, default=FALLBACK_A, help="Pfad CSV A (optional; sonst Fallback)")
    ap.add_argument("csv_b", nargs="?", type=Path, default=FALLBACK_B, help="Pfad CSV B (optional; sonst Fallback)")
    ap.add_argument("--sep-a", type=str, default=None, help="Trennzeichen A (auto, wenn leer)")
    ap.add_argument("--sep-b", type=str, default=None, help="Trennzeichen B (auto, wenn leer)")
    ap.add_argument("--index-cols", type=str, default=None, help="Indexspalten, kommasepariert (optional)")
    ap.add_argument("--key-cols", type=str, default=None, help="Schlüsselspalten für Zeilenabgleich (optional)")
    ap.add_argument("--ignore-column-order", action="store_true", help="Spaltenreihenfolge ignorieren")
    ap.add_argument("--trim", action="store_true", help="Whitespace an Stringspalten trimmen")
    ap.add_argument("--float-rtol", type=float, default=1e-12, help="Rel. Toleranz für float")
    ap.add_argument("--float-atol", type=float, default=0.0, help="Abs. Toleranz für float")
    ap.add_argument("--na-equal", action="store_true", help="Behandle NaN==NaN als gleich")
    ap.add_argument("--report-xlsx", type=Path, default=None, help="Pfad für Excel-Report (optional)")
    args = ap.parse_args()

    idx_cols = args.index_cols.split(",") if args.index_cols else None
    key_cols = args.key_cols.split(",") if args.key_cols else None

    # Pfade normalisieren + Existenz prüfen (vorher klar kommunizieren)
    csv_a = _normalize_path(args.csv_a)
    csv_b = _normalize_path(args.csv_b)
    print(f"[INFO] Datei A: {csv_a}")
    print(f"[INFO] Datei B: {csv_b}")
    if not csv_a.exists():
        print(f"[WARN] Datei A nicht gefunden: {csv_a}")
    if not csv_b.exists():
        print(f"[WARN] Datei B nicht gefunden: {csv_b}")

    dfA = read_csv_smart(csv_a, args.sep_a, idx_cols, trim=args.trim)
    dfB = read_csv_smart(csv_b, args.sep_b, idx_cols, trim=args.trim)

    # Optional: Key-basierte Ausrichtung
    only_a = only_b = None
    if key_cols:
        a_al, b_al, only_a, only_b = key_based_alignment(dfA.reset_index(), dfB.reset_index(), key_cols)
        # Nach Key-Ausrichtung: Index = Key, gleiche Spaltenmengen angleichen
        dfA2, dfB2, onlyAcols, onlyBcols, _ = align_columns(a_al, b_al, args.ignore_column_order)
        schema_msgs = schema_report(dfA2, dfB2)
        all_equal, cell_diffs = compare_dataframes(
            dfA2, dfB2, float_rtol=args.float_rtol, float_atol=args.float_atol, na_equal=args.na_equal
        )
        # Kennzahlen
        n_cell = 0 if cell_diffs is None or cell_diffs.empty else cell_diffs.shape[0]
        n_row = 0 if n_cell == 0 else cell_diffs.reset_index()["index"].nunique()
        print_summary(
            equal=all_equal and not onlyAcols and not onlyBcols and (only_a.empty if only_a is not None else True)
                  and (only_b.empty if only_b is not None else True),
            reasons=(schema_msgs + ([f"- Spalten nur in A: {onlyAcols}"] if onlyAcols else [])
                     + ([f"- Spalten nur in B: {onlyBcols}"] if onlyBcols else [])),
            n_cell_diffs=n_cell,
            n_row_diffs=n_row,
            only_a_rows=(0 if only_a is None else len(only_a)),
            only_b_rows=(0 if only_b is None else len(only_b)),
        )
    else:
        # Spalten angleichen (optional)
        dfA2, dfB2, onlyAcols, onlyBcols, _ = align_columns(dfA, dfB, args.ignore_column_order)
        schema_msgs = schema_report(dfA2, dfB2)
        all_equal, cell_diffs = compare_dataframes(
            dfA2, dfB2, float_rtol=args.float_rtol, float_atol=args.float_atol, na_equal=args.na_equal
        )
        n_cell = 0 if cell_diffs is None or cell_diffs.empty else cell_diffs.shape[0]
        n_row = 0 if n_cell == 0 else cell_diffs.reset_index()["index"].nunique()
        print_summary(
            equal=all_equal and not onlyAcols and not onlyBcols,
            reasons=(schema_msgs + ([f"- Spalten nur in A: {onlyAcols}"] if onlyAcols else [])
                     + ([f"- Spalten nur in B: {onlyBcols}"] if onlyBcols else [])),
            n_cell_diffs=n_cell,
            n_row_diffs=n_row,
            only_a_rows=-1,
            only_b_rows=-1,
        )
        only_a = only_b = None

    # Kurzdetails bei Abweichungen ausgeben
    if cell_diffs is not None and not cell_diffs.empty:
        print("Beispiel-Differenzen (Top 20):")
        print(cell_diffs.head(20))

    # Excel-Report schreiben (optional)
    if args.report_xlsx:
        write_excel_report(
            args.report_xlsx,
            cell_diffs=cell_diffs,
            only_a=only_a,
            only_b=only_b,
            schema_msgs=schema_msgs,
        )
        print(f"[OK] Excel-Report geschrieben: {args.report_xlsx}")


if __name__ == "__main__":
    pd.set_option("display.max_rows", 200)
    pd.set_option("display.max_columns", 200)
    main()
