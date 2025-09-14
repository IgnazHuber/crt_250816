# -*- coding: utf-8 -*-
# file: run_doe_grid.py
"""
DOE-Runner (Flags, robust, CSV nach C:\\Projekte\\crt_250816\\results\\)

- Ranges: a..b..s oder a:b:s; Singletons ok (a==b → [a])
- --verbose zeigt je Kombination die tatsächlich gezählten Werte
- Fallback: zählt aus 'details' DataFrame, falls Rückgabedict 0 meldet
"""

from __future__ import annotations
import os, argparse, itertools, importlib
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

try:
    import pandas as pd
except Exception:
    pd = None


def _ensure_outdir(path: str) -> str:
    path = os.path.abspath(path); os.makedirs(path, exist_ok=True); return path

def _parse_range(spec: str, want_int: bool) -> List[float | int]:
    sep = ".." if ".." in spec else ":"
    parts = spec.split(sep)
    if len(parts) != 3:
        raise ValueError(f"Ungültige Range: {spec} (Format a..b..s)")
    a_s, b_s, s_s = parts
    if want_int:
        a, b = int(float(a_s)), int(float(b_s))
        if a == b:
            return [a]
        s = int(float(s_s))
        if s <= 0:
            raise ValueError(f"Schrittweite muss > 0 sein (erhalten: {s})")
        if a > b:
            a, b = b, a
        return list(range(a, b + 1, s))
    else:
        a, b = Decimal(a_s), Decimal(b_s)
        if a == b:
            return [float(a)]
        s = Decimal(s_s)
        if s <= 0:
            raise ValueError(f"Schrittweite muss > 0 sein (erhalten: {s})")
        if a > b:
            a, b = b, a
        out: List[float] = []
        cur = a
        for _ in range(500000):
            out.append(float(cur))
            cur = (cur + s).quantize(Decimal("0.0000001"))
            if cur > b + Decimal("1e-12"):
                break
        return out

def _import_main():
    mod = importlib.import_module("Mainframe_RT_DOE")
    if not hasattr(mod, "analyze"):
        raise AttributeError("Mainframe_RT_DOE.analyze nicht gefunden.")
    return mod

def _extract_from_details(details) -> Tuple[int, int]:
    c = n = 0
    if pd is not None and isinstance(details, pd.DataFrame):
        if "CBullD_gen" in details.columns:
            c = int(details["CBullD_gen"].astype("float64").fillna(0).gt(0.0).sum())
        if "CBullD_neg_MACD" in details.columns:
            n = int(details["CBullD_neg_MACD"].astype("float64").fillna(0).gt(0.0).sum())
    return c, n

def _call(mod, w: int, ct: float, mt: float, input_path: Optional[str], verbose: bool) -> Tuple[int, int]:
    res = mod.analyze(window=w, candle_tolerance=ct, macd_tolerance=mt, input_path=input_path, enable_plot=False)
    c = int(res.get("classic_count", 0)) if isinstance(res, dict) else 0
    n = int(res.get("neg_macd_count", 0)) if isinstance(res, dict) else 0
    if (c == 0 and n == 0) and isinstance(res, dict) and ("details" in res):
        c2, n2 = _extract_from_details(res["details"])
        if c2 or n2:
            c, n = c2, n2
    if verbose:
        print(f"    -> Counts: Classic={c}, NegMACD={n}")
    return c, n

def _fmt_table(rows: List[Dict[str, Any]]) -> str:
    headers = ["window", "candle_tolerance", "macd_tolerance", "classic_count", "neg_macd_count"]
    widths = {h: max(len(h), *(len(str(r[h])) for r in rows)) for h in headers}
    sep = "-+-".join("-" * widths[h] for h in headers)
    out = [" | ".join(h.rjust(widths[h]) for h in headers), sep]
    for r in rows:
        out.append(" | ".join(str(r[h]).rjust(widths[h]) for h in headers))
    return "\n".join(out)

def main():
    ap = argparse.ArgumentParser(description="DOE – (window, candle_tolerance, macd_tolerance)")
    ap.add_argument("--input", "-i", type=str, default=None)
    ap.add_argument("--out", "-o", type=str, default=r"C:\Projekte\crt_250816\results")
    ap.add_argument("--w", type=str, required=True, help="z.B. 5..5..1")
    ap.add_argument("--ct", type=str, required=True, help="z.B. 0.1..0.1..0.01")
    ap.add_argument("--mt", type=str, required=True, help="z.B. 1.25..5.25..2.0")
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()

    out_dir = _ensure_outdir(args.out)
    windows = _parse_range(args.w, want_int=True)
    candles = _parse_range(args.ct, want_int=False)
    macds   = _parse_range(args.mt, want_int=False)

    print("===================================")
    print("Versuchsplan")
    print("===================================")
    print(f"- window:           {windows}")
    print(f"- candle_tolerance: {candles}")
    print(f"- macd_tolerance:   {macds}")
    total = len(windows) * len(candles) * len(macds)
    print(f"- Kombinationen:    {total}\n")

    mod = _import_main()

    rows: List[Dict[str, Any]] = []
    i = 0
    for w, ct, mt in itertools.product(windows, candles, macds):
        i += 1
        print(f"[{i}/{total}] window={w}, candle_tol={ct}, macd_tol={mt} ...")
        try:
            c, n = _call(mod, int(w), float(ct), float(mt), args.input, args.verbose)
        except Exception as e:
            print(f"    -> FEHLER: {e}")
            c = n = 0
        rows.append({
            "window": int(w),
            "candle_tolerance": round(float(ct), 6),
            "macd_tolerance": round(float(mt), 6),
            "classic_count": int(c),
            "neg_macd_count": int(n),
        })

    print("\n===================")
    print("DOE-Ergebnistabelle")
    print("===================")
    print(_fmt_table(rows))

    out_csv = os.path.join(out_dir, "doe_results.csv")
    try:
        if pd is not None:
            pd.DataFrame(rows).to_csv(out_csv, index=False)
        else:
            with open(out_csv, "w", encoding="utf-8") as f:
                f.write("window,candle_tolerance,macd_tolerance,classic_count,neg_macd_count\n")
                for r in rows:
                    f.write(f"{r['window']},{r['candle_tolerance']},{r['macd_tolerance']},{r['classic_count']},{r['neg_macd_count']}\n")
        print(f"\n[Info] CSV gespeichert: {out_csv}")
    except Exception as e:
        print(f"[Warn] CSV nicht gespeichert: {e}")

    # Summen + Top
    total_c = sum(r["classic_count"] for r in rows)
    total_n = sum(r["neg_macd_count"] for r in rows)
    print("\nSummen:")
    print(f"- Classic Divergences:       {total_c}")
    print(f"- Negative MACD Divergences: {total_n}")

    ranked = sorted(rows, key=lambda r: (r["classic_count"] + r["neg_macd_count"], r["classic_count"]), reverse=True)[:min(5, len(rows))]
    print("\nTop-Kombinationen:")
    for j, r in enumerate(ranked, 1):
        tot = r["classic_count"] + r["neg_macd_count"]
        print(f" {j:>2}. w={r['window']} ct={r['candle_tolerance']} mt={r['macd_tolerance']}  -> C={r['classic_count']}, N={r['neg_macd_count']} (Tot={tot})")


if __name__ == "__main__":
    main()
