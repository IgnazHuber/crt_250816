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
    # Build equity chart
    img64 = ''
    try:
        eq = results.get('equity', pd.DataFrame())
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
        f.write("<html><body>")
        f.write(f"<h3>Execution Simulator</h3>")
        f.write('<ul>')
        f.write('<li><b>Was wird simuliert?</b> Latenz (Einstieg nach n Bars) und einfache Limit/Stop‑Einstiegslogik: Einstiegsbar muss den Preis intrabar berühren (Touch), sonst keine Füllung.</li>')
        f.write('<li><b>Wofür verwenden?</b> Abschätzen realistischer Ausführung: geringere Füllquote, geänderte PnL/Trades im Vergleich zur idealisierten Ausführung.</li>')
        f.write('<li><b>Warum wichtig?</b> Zeigt Robustheit der Strategie unter realistischeren Marktbedingungen (Slippage/Fees/Lattenz).</li>')
        f.write('</ul>')
        f.write(f"<p>Asset: {asset_label}</p>")
        f.write(f"<p>Latency bars: {latency} | Slippage%: {slip_exec} | Order model: {order_model}")
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
