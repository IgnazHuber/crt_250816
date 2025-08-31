import os
import pandas as pd
import numpy as np

def run_portfolio_allocation(paths, out_xlsx: str, out_html: str):
    """Scaffold: load multiple CSVs, compute a simple return correlation matrix.
    Assumes columns contain 'date' and 'close'. Writes minimal outputs.
    """
    series = {}
    for p in paths:
        try:
            df = pd.read_csv(p)
            if 'date' in df.columns and 'close' in df.columns:
                s = pd.to_datetime(df['date'])
                r = pd.Series(df['close']).pct_change().rename(os.path.basename(p))
                r.index = s
                series[os.path.basename(p)] = r
        except Exception:
            continue
    if not series:
        mat = pd.DataFrame({'Note':['No compatible files loaded']})
    else:
        mat = pd.DataFrame(series).dropna(how='all').corr()
    os.makedirs(os.path.dirname(out_xlsx) or '.', exist_ok=True)
    try:
        with pd.ExcelWriter(out_xlsx, engine='openpyxl') as w:
            mat.to_excel(w, sheet_name='Correlation')
    except Exception:
        mat.to_csv(out_xlsx.replace('.xlsx', '.csv'))
    with open(out_html, 'w', encoding='utf-8') as f:
        f.write("<html><body><h3>Portfolio Layer (Scaffold)</h3><pre>" + mat.to_string() + "</pre></body></html>")

