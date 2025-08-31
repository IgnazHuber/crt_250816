import os
import pandas as pd

def run_regime_conditioning(df: pd.DataFrame, out_xlsx: str, out_html: str, asset_label: str):
    """Scaffold: Regime conditioning placeholder.
    Writes a basic volatility proxy and TODOs.
    """
    dfx = df.copy()
    if 'close' in dfx.columns:
        dfx['ret'] = dfx['close'].pct_change()
        dfx['vol_30'] = (dfx['ret'].rolling(30).std() * (30 ** 0.5)).fillna(0)
    info = dfx[['date', 'vol_30']].tail(5) if 'date' in dfx.columns and 'vol_30' in dfx.columns else pd.DataFrame()
    os.makedirs(os.path.dirname(out_xlsx) or '.', exist_ok=True)
    try:
        with pd.ExcelWriter(out_xlsx, engine='openpyxl') as w:
            (info if not info.empty else pd.DataFrame({'Note':['Scaffold – compute regimes later']})).to_excel(w, sheet_name='Preview', index=False)
    except Exception:
        (info if not info.empty else pd.DataFrame({'Note':['Scaffold – compute regimes later']})).to_csv(out_xlsx.replace('.xlsx', '.csv'), index=False)
    with open(out_html, 'w', encoding='utf-8') as f:
        f.write(f"<html><body><h3>Regime Conditioning (Scaffold)</h3><p>Asset: {asset_label}</p></body></html>")

