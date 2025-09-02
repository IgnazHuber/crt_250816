import os
import pandas as pd
import numpy as np
from Backtest_Divergences import BacktestParams, backtest as run_backtest
import base64, io
import matplotlib.pyplot as plt

def _apply_latency(df: pd.DataFrame, markers: pd.DataFrame, latency_bars: int) -> pd.DataFrame:
    if latency_bars <= 0:
        return markers
    dates = pd.to_datetime(df['date']).reset_index(drop=True)
    m = markers.copy()
    m['Date'] = pd.to_datetime(m['Date'], utc=True, errors='coerce')
    # Map marker time to index of the next bar, then add latency
    idx_pos = np.searchsorted(dates.values, m['Date'].values, side='left')
    idx_pos = idx_pos + latency_bars
    idx_pos = np.clip(idx_pos, 0, len(dates) - 1)
    m['Date'] = dates.iloc[idx_pos].values
    return m

def _filter_by_entry_touch(df: pd.DataFrame, markers: pd.DataFrame, model: str, tick_pct: float) -> pd.DataFrame:
    """Require an intrabar touch on the entry bar for a simple limit/stop approximation.
    - model='limit': require favorable move of at least tick_pct from entry open
      long: low <= open*(1 - tick_pct); short: high >= open*(1 + tick_pct)
    - model='stop': require adverse move of at least tick_pct from entry open
      long: high >= open*(1 + tick_pct); short: low <= open*(1 - tick_pct)
    Entry bar = first bar strictly after marker Date.
    """
    if model not in ('limit', 'stop') or tick_pct <= 0:
        return markers
    dts = pd.to_datetime(df['date']).reset_index(drop=True)
    opens = df['open'].reset_index(drop=True)
    highs = df['high'].reset_index(drop=True) if 'high' in df.columns else opens
    lows = df['low'].reset_index(drop=True) if 'low' in df.columns else opens

    def dir_from_type(t: str) -> int:
        t = str(t or '')
        if 'Bear' in t or 'bear' in t:
            return -1
        return 1

    kept = []
    for _, r in markers.iterrows():
        md = pd.to_datetime(r.get('Date'), utc=True, errors='coerce')
        if pd.isna(md):
            continue
        i = int(np.searchsorted(dts.values, md.to_datetime64(), side='right'))
        if i >= len(dts):
            continue
        o = float(opens.iloc[i])
        h = float(highs.iloc[i])
        l = float(lows.iloc[i])
        d = dir_from_type(r.get('Type'))
        ok = True
        if model == 'limit':
            if d > 0:  # long
                ok = (l <= o * (1 - tick_pct))
            else:      # short
                ok = (h >= o * (1 + tick_pct))
        elif model == 'stop':
            if d > 0:  # long
                ok = (h >= o * (1 + tick_pct))
            else:      # short
                ok = (l <= o * (1 - tick_pct))
        if ok:
            kept.append(r)
    return pd.DataFrame(kept, columns=markers.columns)


def run_execution_simulator(df: pd.DataFrame, markers: pd.DataFrame, out_xlsx: str, out_html: str, asset_label: str):
    """Execution Simulator (lightweight):
    - Reuses the backtest engine
    - Adds optional latency (bars) via EXEC_LATENCY_BARS (default 0)
    - Allows overriding slippage via EXEC_SLIPPAGE_PCT (default use BT_SLIPPAGE_PCT)
    """
    try:
        latency = int(os.getenv('EXEC_LATENCY_BARS', '0'))
    except Exception:
        latency = 0
    # Prepare params (inherit default fee/slippage envs used elsewhere)
    try:
        fee_def = float(os.getenv('BT_FEE_PCT', '0.0'))
    except Exception:
        fee_def = 0.0
    try:
        slip_bt = float(os.getenv('BT_SLIPPAGE_PCT', '0.0'))
    except Exception:
        slip_bt = 0.0
    try:
        slip_exec = float(os.getenv('EXEC_SLIPPAGE_PCT', str(slip_bt)))
    except Exception:
        slip_exec = slip_bt

    params = BacktestParams(fee_pct=fee_def, slippage_pct=slip_exec)
    mk = _apply_latency(df, markers, latency)
    # Simple order model
    order_model = os.getenv('EXEC_ORDER_MODEL', 'market').strip().lower()
    try:
        tick_pct = float(os.getenv('EXEC_TICK_PCT', '0.001'))  # 0.1% default
    except Exception:
        tick_pct = 0.001
    if order_model in ('limit', 'stop'):
        before = len(mk)
        mk = _filter_by_entry_touch(df, mk, order_model, tick_pct)
        filtered = before - len(mk)
    else:
        filtered = 0
    results = run_backtest(df.copy(), mk.copy(), params)
    os.makedirs(os.path.dirname(out_xlsx) or '.', exist_ok=True)
    try:
        from Backtest_Divergences import export_backtest_xlsx
        export_backtest_xlsx(results, out_xlsx)
        # Attempt to append a small Info sheet with DE description
        try:
            import openpyxl
            wb = openpyxl.load_workbook(out_xlsx)
            ws = wb.create_sheet('Exec_Info')
            ws.append(['Beschreibung_DE', 'Order_Model', 'Latency_Bars', 'Tick_Pct', 'Slippage_%'])
            ws.append(['Einfache Ausführungssimulation: Latenz und Limit/Stop-Einstieg (Bar-Touch) als Approximation.', order_model, latency, tick_pct, slip_exec])
            wb.save(out_xlsx)
        except Exception:
            pass
    except Exception:
        # Fallback: write summary only
        summ = results.get('summary', pd.DataFrame())
        if summ is None or summ.empty:
            summ = pd.DataFrame([{'Info': 'No trades or summary'}])
        with pd.ExcelWriter(out_xlsx, engine='openpyxl') as w:
            summ.to_excel(w, sheet_name='Summary', index=False)
            pd.DataFrame({'Beschreibung_DE':['Einfache Ausführungssimulation: Latenz und Limit/Stop-Einstieg (Bar-Touch) als Approximation.']}).to_excel(w, sheet_name='Info', index=False)
    # Build equity chart with trade markers
    img64 = ''
    try:
        eq = results.get('equity', pd.DataFrame())
        tr = results.get('trades', pd.DataFrame())
        if eq is not None and not eq.empty:
            fig, ax = plt.subplots(figsize=(6, 2))
            ax.plot(pd.to_datetime(eq['Date']), eq['Equity'], color='#1f77b4', lw=1.2, label='Equity')
            try:
                if tr is not None and not tr.empty:
                    eq_dt = pd.to_datetime(eq['Date']).reset_index(drop=True)
                    eq_eq = eq['Equity'].reset_index(drop=True)
                    xs_win, ys_win, xs_loss, ys_loss = [], [], [], []
                    for _, r in tr.iterrows():
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

    with open(out_html, 'w', encoding='utf-8') as f:
        f.write("<html><body>")
        f.write(f"<h3>Execution Simulator</h3>")
        f.write('<ul>')
        f.write('<li><b>Voraussetzungen:</b> Marker‑CSV in results/ (aus Optionen a–e → Marker exportieren oder DOE f); Zeitspanne ggf. vorab einschränken.</li>')
        f.write('<li><b>Latency bars:</b> Anzahl Bars Verzögerung zwischen Signal und Orderplatzierung (größer = spätere Einstiege).</li>')
        f.write('<li><b>Slippage%:</b> Adverser Zuschlag auf Fill‑Preis (Ein-/Ausstieg).</li>')
        f.write('<li><b>Order model:</b> ') 
        # Bold the selected model and list options
        om_disp = f"<b>{order_model}</b>" if order_model in ('market','limit','stop') else order_model
        f.write(om_disp + ' — Modelle: ')
        f.write('<b>market</b>: sofort am nächsten Open; ')
        f.write('limit: Fill nur bei intrabar günstigerem Tick (Touch) als Open±tick_pct; ')
        f.write('stop: Fill nur bei intrabar ungünstigerem Tick (Touch) als Open±tick_pct.')
        f.write('</li>')
        f.write('<li><b>Wofür verwenden?</b> Abschätzen realistischer Ausführung: geringere Füllquote, geänderte PnL/Trades im Vergleich zur idealisierten Ausführung.</li>')
        f.write('<li><b>Warum wichtig?</b> Zeigt Robustheit der Strategie unter realistischeren Marktbedingungen (Slippage/Fees/Latenz).</li>')
        f.write('</ul>')
        f.write(f"<p>Asset: {asset_label}</p>")
        f.write(f"<p>Latency bars: {latency} | Slippage%: {slip_exec} | Order model: {om_disp}")
        if order_model in ('limit','stop'):
            f.write(f" (tick_pct={tick_pct:.4f}) | Filtered markers: {filtered}")
        f.write("</p>")
        if img64:
            f.write("<h4>Equity über Zeit</h4><img src='data:image/png;base64,%s' />" % img64)
        try:
            summ = results.get('summary')
            if summ is not None and not summ.empty:
                f.write(summ.to_html(index=False))
        except Exception:
            pass
        f.write("</body></html>")
    # Return details for console summary
    try:
        return {
            'summary': results.get('summary'),
            'filtered': filtered,
            'order_model': order_model,
            'latency': latency,
            'tick_pct': tick_pct,
            'slippage': slip_exec,
        }
    except Exception:
        return {}
