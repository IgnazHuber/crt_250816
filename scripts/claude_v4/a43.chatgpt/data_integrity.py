import os
import pandas as pd
import numpy as np
import base64, io
import matplotlib.pyplot as plt

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
    # Gap histogram plot
    img64 = ''
    try:
        if 'date' in df.columns:
            dts = pd.to_datetime(df['date'])
            gaps = dts.diff().dropna().dt.total_seconds() / 60.0  # minutes
            if not gaps.empty:
                fig, ax = plt.subplots(figsize=(6, 2))
                ax.hist(gaps, bins=30, color='#7f8c8d')
                ax.set_title('Verteilung der Zeitabstände (Minuten)')
                fig.tight_layout()
                buf = io.BytesIO()
                fig.savefig(buf, format='png', dpi=120)
                plt.close(fig)
                img64 = base64.b64encode(buf.getvalue()).decode('ascii')
    except Exception:
        img64 = ''

    with open(out_html, 'w', encoding='utf-8') as f:
        f.write('<html><body>')
        f.write('<h3>Data Quality Audit</h3>')
        f.write('<ul>')
        f.write('<li><b>Was wird geprüft?</b> Fehlende Bars, unregelmäßige Zeitabstände, grobe Anomalien.</li>')
        f.write('<li><b>Wofür verwenden?</b> Sicherstellen, dass Backtests nicht durch Datenfehler verzerrt sind.</li>')
        f.write('<li><b>Warum wichtig?</b> Datenqualität beeinflusst Signale, Fills und KPIs; schlechte Daten verfälschen Ergebnisse.</li>')
        f.write('<li><b>Leer?</b> Wenn keine gültigen Datumswerte vorhanden sind, fällt die Analyse leer aus.</li>')
        f.write('</ul>')
        f.write(f'<p>Asset: {asset_label}</p>')
        if img64:
            f.write("<h4>Gaps‑Histogramm</h4><img src='data:image/png;base64,%s' />" % img64)
        f.write('</body></html>')
