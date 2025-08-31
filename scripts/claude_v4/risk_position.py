import os
import pandas as pd

def run_risk_management(df: pd.DataFrame, markers: pd.DataFrame, out_xlsx: str, out_html: str, asset_label: str):
    """Scaffold: Risk-of-Ruin and Kelly range placeholder.
    Later consume trade stats from backtest to compute metrics.
    """
    info = pd.DataFrame({
        'Asset': [asset_label],
        'Rows': [len(df)],
        'Markers': [len(markers) if markers is not None else 0],
        'Kelly_Fraction': [None],
        'Risk_of_Ruin_%': [None],
        'Note': ['Scaffold – compute from trade-level stats later']
    })
    os.makedirs(os.path.dirname(out_xlsx) or '.', exist_ok=True)
    try:
        with pd.ExcelWriter(out_xlsx, engine='openpyxl') as w:
            info.to_excel(w, sheet_name='Info', index=False)
    except Exception:
        info.to_csv(out_xlsx.replace('.xlsx', '.csv'), index=False)
    with open(out_html, 'w', encoding='utf-8') as f:
        f.write(f"<html><body><h3>Risk Management (Scaffold)</h3><p>Asset: {asset_label}</p></body></html>")

