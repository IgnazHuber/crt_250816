import os
import pandas as pd
import numpy as np
import base64, io
import matplotlib.pyplot as plt
from Backtest_Divergences import BacktestParams, backtest as run_backtest

def run_risk_management(df: pd.DataFrame, markers: pd.DataFrame, out_xlsx: str, out_html: str, asset_label: str):
    """Scaffold: Risk-of-Ruin and Kelly range placeholder.
    Later consume trade stats from backtest to compute metrics.
    """
    # Compute simple diagnostics via a default backtest if markers provided
    try:
        res = run_backtest(df.copy(), (markers or pd.DataFrame()).copy(), BacktestParams())
        trades = res.get('trades', pd.DataFrame())
        if trades is not None and not trades.empty:
            wins = trades[trades['PnL_$'] > 0]['PnL_$']
            losses = trades[trades['PnL_$'] <= 0]['PnL_$']
            w = float((wins.size) / max(1, trades.shape[0]))
            avg_win = float(wins.mean()) if wins.size else 0.0
            avg_loss = abs(float(losses.mean())) if losses.size else 0.0
            R = (avg_win / avg_loss) if avg_loss > 0 else float('nan')
            kelly = w - (1 - w) / R if R and R > 0 else float('nan')
            # probability of N consecutive losses as proxy for ruin risk
            N = 10
            ruin_proxy = ((1 - w) ** N) * 100.0
        else:
            w = float('nan'); avg_win = float('nan'); avg_loss = float('nan'); R = float('nan'); kelly = float('nan'); ruin_proxy = float('nan')
    except Exception:
        w = float('nan'); avg_win = float('nan'); avg_loss = float('nan'); R = float('nan'); kelly = float('nan'); ruin_proxy = float('nan')

    info = pd.DataFrame({
        'Asset': [asset_label],
        'Rows': [len(df)],
        'Markers': [len(markers) if markers is not None else 0],
        'Win_Rate_%': [round(w*100.0,2) if w==w else None],
        'Avg_Win_$': [round(avg_win,2) if avg_win==avg_win else None],
        'Avg_Loss_$': [round(avg_loss,2) if avg_loss==avg_loss else None],
        'Payoff_R': [round(R,3) if R==R else None],
        'Kelly_Fraction': [round(kelly,3) if kelly==kelly else None],
        'Ruin_Proxy_%(10_losses)': [round(ruin_proxy,2) if ruin_proxy==ruin_proxy else None],
        'Beschreibung_DE': ['Risk‑of‑Ruin & Kelly (vereinfachte Schätzungen): aus Trade‑Kennzahlen abgeleitet; Ruin‑Proxy = P(10 Verlusttrades in Folge).']
    })
    os.makedirs(os.path.dirname(out_xlsx) or '.', exist_ok=True)
    try:
        with pd.ExcelWriter(out_xlsx, engine='openpyxl') as w:
            info.to_excel(w, sheet_name='Info', index=False)
    except Exception:
        info.to_csv(out_xlsx.replace('.xlsx', '.csv'), index=False)
    # Equity curve for context (default params)
    img64 = ''
    try:
        res = run_backtest(df.copy(), (markers or pd.DataFrame()).copy(), BacktestParams())
        eq = res.get('equity', pd.DataFrame())
        if eq is not None and not eq.empty:
            fig, ax = plt.subplots(figsize=(6, 2))
            ax.plot(pd.to_datetime(eq['Date']), eq['Equity'], color='#1f77b4', lw=1.2)
            ax.set_title('Equity über Zeit')
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
        f.write('<h3>Risk Management</h3>')
        f.write('<ul>')
        f.write('<li><b>Voraussetzungen:</b> Marker‑CSV in results/ (aus Optionen a–e → Marker exportieren oder DOE f); im Zweifel Zeitspanne einschränken.</li>')
        f.write('<li><b>Was wird analysiert?</b> Risiko‑Kennzahlen wie Risk‑of‑Ruin und Kelly‑Range (Platzhalter); später aus Trade‑Verteilung berechnet.</li>')
        f.write('<li><b>Wofür verwenden?</b> Ermitteln, welche Positionsgröße dauerhaft überlebensfähig ist und Ruinwahrscheinlichkeit klein hält.</li>')
        f.write('<li><b>Warum wichtig?</b> Eine gute Strategie kann mit falscher Positionsgröße scheitern; Risikosteuerung stabilisiert Ergebnisse.</li>')
        f.write('<li><b>Leer?</b> Wenn keine Trades/Marker vorliegen, bleiben Kennzahlen leer. Bitte zuerst Marker erzeugen und Backtest durchführen.</li>')
        f.write('<li><b>Hinweis zu Schätzungen:</b> Kelly und Ruin‑Proxy sind vereinfachte Indikatoren (Annahmen: stationäre Verteilung, unabhängige Trades); konservativ interpretieren.</li>')
        f.write('</ul>')
        f.write(f'<p>Asset: {asset_label}</p>')
        try:
            f.write('<h4>Schätzwerte</h4>')
            f.write(info.to_html(index=False))
        except Exception:
            pass
        if img64:
            f.write("<h4>Equity über Zeit</h4><img src='data:image/png;base64,%s' />" % img64)
        f.write('</body></html>')
