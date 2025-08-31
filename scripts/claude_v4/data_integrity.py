import os
import pandas as pd

def run_data_quality_audit(df: pd.DataFrame, out_xlsx: str, out_html: str, asset_label: str):
    """Scaffold: simple gap/missing audit.
    Computes basic timestamp gaps and writes a small report.
    """
    report = []
    if 'date' in df.columns:
        dts = pd.to_datetime(df['date'])
        gaps = dts.diff().dropna()
        report.append({'Metric':'Rows','Value':len(df)})
        if not gaps.empty:
            report.append({'Metric':'MinGap','Value':gaps.min()})
            report.append({'Metric':'MedianGap','Value':gaps.median()})
            report.append({'Metric':'MaxGap','Value':gaps.max()})
    else:
        report.append({'Metric':'Rows','Value':len(df)})
        report.append({'Metric':'Note','Value':'No date column detected'})
    rep_df = pd.DataFrame(report)
    os.makedirs(os.path.dirname(out_xlsx) or '.', exist_ok=True)
    try:
        with pd.ExcelWriter(out_xlsx, engine='openpyxl') as w:
            rep_df.to_excel(w, sheet_name='Audit', index=False)
    except Exception:
        rep_df.to_csv(out_xlsx.replace('.xlsx', '.csv'), index=False)
    with open(out_html, 'w', encoding='utf-8') as f:
        f.write(f"<html><body><h3>Data Quality Audit (Scaffold)</h3><p>Asset: {asset_label}</p></body></html>")

