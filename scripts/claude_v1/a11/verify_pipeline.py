# -*- coding: utf-8 -*-
# file: verify_pipeline.py
"""
Einmalige Mini-Diagnose für eine Kombination.
Erzeugt Snapshot unter C:\\Projekte\\crt_250816\\results\\_debug\\snapshot.csv
und druckt Spalten- & Werteprüfungen in die Konsole.
"""

import os
import pandas as pd
import Mainframe_RT_DOE as mf

OUT_DIR = r"C:\Projekte\crt_250816\results\_debug"
os.makedirs(OUT_DIR, exist_ok=True)

# ---- HIER ggf. anpassen ----
INPUT = r"C:\Projekte\crt_250816\data\processed\btc_1day_candlesticks_all.parquet"
WINDOW = 5
CT = 0.5
MT = 2.0
# ----------------------------

res = mf.analyze(window=WINDOW, candle_tolerance=CT, macd_tolerance=MT,
                 input_path=INPUT, enable_plot=False, diagnose=True)

df = res.get("details")
if isinstance(df, pd.DataFrame):
    # nur die interessantesten Spalten exportieren
    keep = [c for c in df.columns if c.lower() in
            {"date","cbulld_gen","cbulld_neg_macd","cbulld_lower_low_date_gen","cbulld_higher_low_date_gen",
             "cbulld_lower_low_gen","cbulld_higher_low_gen","rsi","macd_histogram",
             "lm_low_window_1_cs","lm_low_window_2_cs","lm_low_window_1_macd","lm_low_window_2_macd"}]
    # falls Spaltennamen anders sind: heuristisch sammeln
    if not keep:
        for c in df.columns:
            cl = c.lower()
            if ("cbulld" in cl) or ("neg" in cl and "macd" in cl) or ("rsi" in cl) or ("macd" in cl and "hist" in cl) or ("lm_" in cl):
                keep.append(c)
    snap = df[keep].copy() if keep else df.copy()
    out_csv = os.path.join(OUT_DIR, "snapshot.csv")
    snap.to_csv(out_csv, index=False)
    print(f"[Snapshot] geschrieben: {out_csv}")

# kurze Anzeige
print("Classic:", res.get("classic_count"), "NegMACD:", res.get("neg_macd_count"))
if isinstance(df, pd.DataFrame):
    for probe in ("CBullD_gen","CBullD_neg_MACD"):
        for c in df.columns:
            if c.lower() == probe.lower():
                print(f"value_counts({c}):")
                print(df[c].value_counts(dropna=False).head(10))
