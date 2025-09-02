import os
import pandas as pd
import numpy as np
import base64, io
import matplotlib.pyplot as plt
from Backtest_Divergences import BacktestParams, backtest as run_backtest

def run_risk_management(df: pd.DataFrame, markers: pd.DataFrame, out_xlsx: str, out_html: str, asset_label: str, params: BacktestParams | None = None):
    """Scaffold: Risk-of-Ruin and Kelly range placeholder.
    Later consume trade stats from backtest to compute metrics.
    """
    # Compute simple diagnostics via a default backtest if markers provided
    res = {}
    try:
        bt_params = params if params is not None else BacktestParams()
        res = run_backtest(df.copy(), (markers or pd.DataFrame()).copy(), bt_params)
        trades = res.get('trades', pd.DataFrame())
        summary = res.get('summary', pd.DataFrame())
        if trades is not None and not trades.empty:
            wins = trades[trades['PnL_$'] > 0]['PnL_$']
            losses = trades[trades['PnL_$'] <= 0]['PnL_$']
            w = float((wins.size) / max(1, trades.shape[0]))
            avg_win = float(wins.mean()) if wins.size else 0.0
            avg_loss = abs(float(losses.mean())) if losses.size else 0.0
        else:
            # Fallback to summary-based estimates when no trade rows available
            if summary is not None and not summary.empty:
                s = summary.iloc[0]
                wr = s.get('Win_Rate_%')
                w = float(wr) / 100.0 if wr is not None and wr == wr else float('nan')
                avg_win = float(s.get('Avg_Win_$', float('nan')))
                avg_loss = abs(float(s.get('Avg_Loss_$', float('nan'))))
            else:
                w = float('nan'); avg_win = float('nan'); avg_loss = float('nan')
        R = (avg_win / avg_loss) if isinstance(avg_loss,(int,float)) and avg_loss > 0 else float('nan')
        kelly = w - (1 - w) / R if isinstance(R,(int,float)) and R > 0 and isinstance(w,(int,float)) and 0 <= w <= 1 else float('nan')
        # probability of N consecutive losses as proxy for ruin risk
        N = 10
        ruin_proxy = ((1 - w) ** N) * 100.0 if isinstance(w,(int,float)) else float('nan')
    except Exception:
        trades = pd.DataFrame(); summary = pd.DataFrame()
        w = float('nan'); avg_win = float('nan'); avg_loss = float('nan'); R = float('nan'); kelly = float('nan'); ruin_proxy = float('nan')

    # Trades count for reference
    try:
        trades_n = int(trades.shape[0]) if trades is not None else (int(summary.iloc[0].get('Trades', 0)) if summary is not None and not summary.empty else 0)
    except Exception:
        trades_n = 0

    # Defer building/writing info until after potential preview recomputation
    # Equity curve for context + trade overlays
    img64 = ''
    preview_note = ''
    try:
        eq = res.get('equity', pd.DataFrame()) if isinstance(res, dict) else pd.DataFrame()
        trs = res.get('trades', pd.DataFrame()) if isinstance(res, dict) else pd.DataFrame()
        # If no trades/equity, run a preview with default params to visualize behavior similar to option l
        if (eq is None or eq.empty) or (trs is None or trs.empty):
            try:
                res_preview = run_backtest(df.copy(), (markers or pd.DataFrame()).copy(), BacktestParams())
                eq_preview = res_preview.get('equity', pd.DataFrame())
                tr_preview = res_preview.get('trades', pd.DataFrame())
                if eq_preview is not None and not eq_preview.empty:
                    eq = eq_preview
                    trs = tr_preview
                    preview_note = '<p><i>Hinweis:</i> Keine Trades mit gewählten Parametern → Vorschau mit Standardparametern gezeigt.</p>'
            except Exception:
                pass
        if eq is not None and not eq.empty:
            fig, ax = plt.subplots(figsize=(6, 2))
            ax.plot(pd.to_datetime(eq['Date']), eq['Equity'], color='#1f77b4', lw=1.2, label='Equity')
            # Overlay trade exits on equity (green=win, red=loss)
            try:
                if trs is not None and not trs.empty:
                    eq_dt = pd.to_datetime(eq['Date']).reset_index(drop=True)
                    eq_eq = eq['Equity'].reset_index(drop=True)
                    xs_win, ys_win, xs_loss, ys_loss = [], [], [], []
                    for _, r in trs.iterrows():
                        t = pd.to_datetime(r.get('Exit_Date'))
                        if pd.isna(t):
                            continue
                        pos = int(np.searchsorted(eq_dt.values, t.to_datetime64(), side='right')) - 1
                        pos = max(0, min(pos, len(eq_dt) - 1))
                        yv = float(eq_eq.iloc[pos])
                        if float(r.get('PnL_$', 0.0)) >= 0:
                            xs_win.append(eq_dt.iloc[pos]); ys_win.append(yv)
                        else:
                            xs_loss.append(eq_dt.iloc[pos]); ys_loss.append(yv)
                    if xs_win:
                        ax.scatter(xs_win, ys_win, marker='^', color='#2ecc71', s=18, label='Win exits')
                    if xs_loss:
                        ax.scatter(xs_loss, ys_loss, marker='v', color='#e74c3c', s=18, label='Loss exits')
            except Exception:
                pass
            ax.set_title('Equity über Zeit (mit Trade‑Markierungen)')
            ax.legend(loc='best', fontsize=8)
            ax.tick_params(axis='x', labelrotation=30)
            fig.tight_layout()
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=120)
            plt.close(fig)
            img64 = base64.b64encode(buf.getvalue()).decode('ascii')
    except Exception:
        img64 = ''

    # Recompute metrics if the primary run had no trades but preview produced trades
    try:
        if (trades_n == 0 or not (w == w)) and 'trs' in locals() and trs is not None and not trs.empty:
            wins = trs[trs['PnL_$'] > 0]['PnL_$']
            losses = trs[trs['PnL_$'] <= 0]['PnL_$']
            w = float((wins.size) / max(1, trs.shape[0]))
            avg_win = float(wins.mean()) if wins.size else 0.0
            avg_loss = abs(float(losses.mean())) if losses.size else 0.0
            R = (avg_win / avg_loss) if avg_loss > 0 else float('nan')
            kelly = w - (1 - w) / R if R and R > 0 else float('nan')
            N = 10
            ruin_proxy = ((1 - w) ** N) * 100.0
            trades_n = int(trs.shape[0])
    except Exception:
        pass

    # Build final Info DataFrame and write XLSX/CSV
    info = pd.DataFrame({
        'Asset': [asset_label],
        'Rows': [len(df)],
        'Markers': [len(markers) if markers is not None else 0],
        'Trades': [trades_n],
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
            f.write('<h4>Begriffe</h4><ul>')
            f.write('<li><b>Trades:</b> Anzahl abgeschlossener Trades in diesem Lauf.</li>')
            f.write('<li><b>Win_Rate_%:</b> Anteil gewinnender Trades.</li>')
            f.write('<li><b>Payoff_R:</b> Verhältnis durchschnittlicher Gewinn zu durchschnittlichem Verlust (|Avg_Loss|).</li>')
            f.write('<li><b>Kelly_Fraction:</b> Theoretisch optimale Einsatzquote (vereinfachte Formel), nur als Orientierung.</li>')
            f.write('<li><b>Ruin_Proxy_%(10_losses):</b> Wahrscheinlichkeit von 10 Verlusttrades in Folge (aus Win‑Rate abgeleitet).</li>')
            f.write('</ul>')
        except Exception:
            pass
        if preview_note:
            f.write(preview_note)
        if img64:
            f.write("<h4>Equity über Zeit</h4><img src='data:image/png;base64,%s' />" % img64)
        else:
            f.write('<p><i>Keine Equity‑Vorschau verfügbar (keine Trades und keine Vorschau erzeugt).</i></p>')
        f.write('</body></html>')
