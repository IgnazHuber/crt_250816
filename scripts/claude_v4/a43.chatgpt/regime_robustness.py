import os
import pandas as pd
import base64, io
import matplotlib.pyplot as plt

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
    # Simple vol chart
    img64 = ''
    try:
        if 'date' in dfx.columns and 'vol_30' in dfx.columns and not dfx['vol_30'].empty:
            fig, ax = plt.subplots(figsize=(6, 2))
            ax.plot(pd.to_datetime(dfx['date']), dfx['vol_30'], color='#e67e22', lw=1.2)
            ax.set_title('30‑Tage Volatilität (ungefähr)')
            ax.tick_params(axis='x', labelrotation=30)
            fig.tight_layout()
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=120)
            plt.close(fig)
            img64 = base64.b64encode(buf.getvalue()).decode('ascii')
    except Exception:
        img64 = ''

    with open(out_html, 'w', encoding='utf-8') as f:
        f.write('<html><body>')
        f.write('<h3>Regime Conditioning</h3>')
        f.write('<ul>')
        f.write('<li><b>Was ist ein Regime?</b> Markt‑Zustand (z. B. Trend/Seitwärts, hohe/niedrige Volatilität), der die Wirksamkeit einer Strategie beeinflusst.</li>')
        f.write('<li><b>Wofür verwenden?</b> Strategie‑Leistung nach Regimen segmentieren; ggf. Filtern/Gewichten nach aktuellem Regime.</li>')
        f.write('<li><b>Warum wichtig?</b> Viele Strategien funktionieren nur in bestimmten Phasen; Regime‑Bewusstsein erhöht Robustheit.</li>')
        f.write('<li><b>Leer?</b> Wenn keine Daten/Spalten vorhanden, bleibt der Plot leer. Stellen Sie sicher, dass close/date vorhanden sind.</li>')
        f.write('</ul>')
        f.write(f'<p>Asset: {asset_label}</p>')
        if img64:
            f.write("<h4>Volatilität</h4><img src='data:image/png;base64,%s' />" % img64)
        f.write('</body></html>')
