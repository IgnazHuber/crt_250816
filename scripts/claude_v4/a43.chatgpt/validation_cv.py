import os
from datetime import datetime
import numpy as np
import pandas as pd

from Backtest_Divergences import BacktestParams, backtest as run_backtest
import base64, io
import matplotlib.pyplot as plt


def _fold_ranges(dates: pd.Series, n_splits: int, embargo_frac: float):
    idx = np.linspace(0, len(dates), n_splits + 1, dtype=int)
    ranges = []
    for k in range(n_splits):
        lo, hi = idx[k], idx[k + 1]
        # apply embargo around fold boundaries relative to full index length
        e = max(1, int((hi - lo) * embargo_frac))
        start_i = lo + e
        end_i = max(start_i + 1, hi - e)
        if start_i < end_i:
            ranges.append((dates.iloc[start_i], dates.iloc[end_i - 1]))
    return ranges


def _safe_kpi_row(summ: pd.DataFrame, fold_id: int, start, end, asset_label: str):
    row = {'Fold': fold_id, 'Start': start, 'End': end, 'Asset': asset_label}
    if summ is None or summ.empty:
        return {**row, 'Trades': 0, 'Total_PnL': 0.0, 'PF': np.nan, 'MaxDD_%': np.nan, 'Final_Equity': np.nan}
    s = summ.iloc[0]
    return {
        **row,
        'Trades': float(s.get('Total_Trades', np.nan)),
        'Total_PnL': float(s.get('Total_PnL', np.nan)),
        'PF': float(s.get('Profit_Factor', np.nan)),
        'MaxDD_%': float(s.get('MaxDD_%', np.nan)),
        'Final_Equity': float(s.get('Final_Equity', np.nan)),
    }


def run_purged_wfcv(df: pd.DataFrame, markers: pd.DataFrame, out_xlsx: str, out_html: str, asset_label: str):
    """Purged Walk-Forward CV on markers using the existing backtest core.
    - Splits controlled by env: WFCV_SPLITS (default 5), WFCV_EMBARGO_PCT (default 0.02)
    - Exports XLSX with per-fold KPIs and simple HTML summary.
    """
    if df is None or df.empty or 'date' not in df.columns:
        raise ValueError('Dataframe must have a date column')
    if markers is None or markers.empty or 'Date' not in markers.columns:
        raise ValueError('Markers must include a Date column')

    n_splits = int(os.getenv('WFCV_SPLITS', '5'))
    embargo = float(os.getenv('WFCV_EMBARGO_PCT', '0.02'))

    dates = pd.to_datetime(df['date']).reset_index(drop=True)
    ranges = _fold_ranges(dates, n_splits, embargo)

    rows = []
    fold_details = []

    # Backtest params (respect env defaults used elsewhere)
    try:
        fee_def = float(os.getenv('BT_FEE_PCT', '0.0'))
    except Exception:
        fee_def = 0.0
    try:
        slip_def = float(os.getenv('BT_SLIPPAGE_PCT', '0.0'))
    except Exception:
        slip_def = 0.0
    params = BacktestParams(fee_pct=fee_def, slippage_pct=slip_def)

    markers_local = markers.copy()
    markers_local['Date'] = pd.to_datetime(markers_local['Date'], utc=True, errors='coerce')
    df_local = df.copy()
    df_local['date'] = pd.to_datetime(df_local['date'], utc=True, errors='coerce')

    for i, (start, end) in enumerate(ranges, 1):
        # slice DF for fold
        mask_df = (df_local['date'] >= start) & (df_local['date'] <= end)
        fold_df = df_local.loc[mask_df].reset_index(drop=True)
        # align markers to fold
        mk = markers_local[(markers_local['Date'] >= start) & (markers_local['Date'] <= end)].copy()
        if mk.empty or fold_df.empty:
            rows.append(_safe_kpi_row(pd.DataFrame(), i, start, end, asset_label))
            continue
        res = run_backtest(fold_df.copy(), mk.copy(), params)
        summ = res.get('summary')
        rows.append(_safe_kpi_row(summ, i, start, end, asset_label))

        # keep minimal details path for HTML
        try:
            trades = res.get('trades', pd.DataFrame())
            fold_details.append({'Fold': i, 'Trades': int(trades.shape[0]) if trades is not None else 0})
        except Exception:
            pass

    df_folds = pd.DataFrame(rows)
    # Aggregate and robust score
    agg = {}
    if not df_folds.empty:
        for col in ['Trades', 'Total_PnL', 'PF', 'MaxDD_%', 'Final_Equity']:
            try:
                agg[f'{col}_mean'] = float(np.nanmean(df_folds[col].astype(float)))
                agg[f'{col}_std'] = float(np.nanstd(df_folds[col].astype(float), ddof=1)) if df_folds[col].notna().sum() > 1 else 0.0
            except Exception:
                pass
        lam = float(os.getenv('WFCV_LAMBDA', '1.0'))
        try:
            agg['Robust_Score'] = float(agg.get('Total_PnL_mean', np.nan)) - lam * float(agg.get('Total_PnL_std', 0.0))
        except Exception:
            pass

    os.makedirs(os.path.dirname(out_xlsx) or '.', exist_ok=True)
    try:
        with pd.ExcelWriter(out_xlsx, engine='openpyxl') as w:
            # Info
            info = pd.DataFrame({
                'Asset': [asset_label],
                'Rows': [len(df_local)],
                'Markers': [len(markers_local)],
                'Splits': [n_splits],
                'Embargo%': [embargo * 100.0],
                'Generated': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
                'Beschreibung_DE': ['Purged Walk-Forward CV: Zeitreihe in Folds mit Embargo; je Fold separater Backtest. Aggregation (μ/σ, Robust_Score) bewertet Robustheit.']
            })
            info.to_excel(w, sheet_name='Info', index=False)
            df_folds.to_excel(w, sheet_name='WFCV_Folds', index=False)
            if agg:
                pd.DataFrame([agg]).to_excel(w, sheet_name='Aggregates', index=False)
    except Exception:
        # Fallback CSVs plus kurze DE-Beschreibung
        base = out_xlsx.replace('.xlsx', '')
        df_folds.to_csv(base + '_folds.csv', index=False)
        pd.DataFrame({'Beschreibung_DE': ['Purged WFCV: Folds mit Embargo; Aggregation μ/σ und Robust_Score.']}).to_csv(base + '_info.csv', index=False)

    # HTML summary
    # Build an overall equity curve (for context)
    try:
        res_full = run_backtest(df_local.copy(), markers_local.copy(), params)
        eq = res_full.get('equity', pd.DataFrame())
        img64 = ''
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
        f.write('<html><head><style>table{border-collapse:collapse}td,th{border:1px solid #555;padding:4px 6px} .num{text-align:right}</style></head><body>')
        f.write('<h3>Purged Walk-Forward CV (WFCV)</h3>')
        f.write('<ul>')
        f.write('<li><b>Was ist ein Fold?</b> Ein zeitlicher Abschnitt (Teil der Historie), der separat validiert wird. Zwischen Folds wird ein Embargo (Sicherheitsabstand) gelassen, um Informationsleckage zu vermeiden.</li>')
        f.write('<li><b>Wofür verwenden?</b> Robustheitsprüfung: Stabilität der Kennzahlen über Folds; Erkennen von Überanpassung.</li>')
        f.write('<li><b>Warum wichtig?</b> Ergebnisse sind weniger verzerrt durch einmalige Marktphasen; Robust_Score (μ−λ·σ) bevorzugt stabile PnL‑Verteilung.</li>')
        f.write('</ul>')
        f.write('<p>Asset: %s | Splits: %d | Embargo: %.1f%%</p>' % (asset_label, n_splits, embargo*100))
        if img64:
            f.write("<h4>Equity über Zeit</h4><img src='data:image/png;base64,%s' />" % img64)
        f.write(f'<p>Asset: {asset_label} | Splits: {n_splits} | Embargo: {embargo*100:.1f}%</p>')
        if not df_folds.empty:
            # Format numeric columns
            df_html = df_folds.copy()
            for c in ['Trades']:
                if c in df_html.columns:
                    df_html[c] = df_html[c].map(lambda v: f"{v:.0f}" if pd.notna(v) else "")
            for c in ['Total_PnL','PF','MaxDD_%','Final_Equity']:
                if c in df_html.columns:
                    df_html[c] = df_html[c].map(lambda v: f"{v:.2f}" if pd.notna(v) else "")
            f.write(df_html.to_html(index=False, classes='tbl', border=0))
            if agg:
                f.write('<h4>Aggregates</h4>')
                agg_fmt = {}
                for k,v in agg.items():
                    if isinstance(v,(int,float)):
                        agg_fmt[k] = f"{v:.2f}"
                    else:
                        agg_fmt[k] = v
                f.write(pd.DataFrame([agg_fmt]).to_html(index=False, classes='tbl', border=0))
            # Simple PnL bar chart (per Fold)
            try:
                mx = float(np.nanmax(np.abs(df_folds['Total_PnL'].astype(float)))) or 1.0
                f.write('<h4>Fold PnL (Balken)</h4><div>')
                for _, r in df_folds.iterrows():
                    val = float(r.get('Total_PnL', 0.0))
                    w = int(300 * abs(val) / mx)
                    color = '#2ecc71' if val >= 0 else '#e74c3c'
                    f.write(f"<div>Fold {int(r['Fold'])}: <span style='display:inline-block;background:{color};height:10px;width:{w}px'></span> {val:.2f}</div>")
                f.write('</div>')
            except Exception:
                pass
        else:
            f.write('<p>No folds generated.</p>')
        f.write('</body></html>')

    # Console summary
    try:
        print("\n=== WFCV Folds ===")
        if not df_folds.empty:
            for _, r in df_folds.iterrows():
                print(f"Fold {int(r['Fold'])}: {r['Start']} → {r['End']} | Trades={int(r.get('Trades',0))} | PnL={r.get('Total_PnL',np.nan):.2f} | PF={r.get('PF',np.nan):.2f} | MaxDD%={r.get('MaxDD_%',np.nan):.2f}")
        if agg:
            print("=== Aggregates ===")
            rs = agg.get('Robust_Score', None)
            rs_str = f" | RobustScore={rs:.2f}" if rs is not None else ""
            print(f"PnL μ/σ = {agg.get('Total_PnL_mean',np.nan):.2f}/{agg.get('Total_PnL_std',0.0):.2f}{rs_str}")
    except Exception:
        pass

    # Also return data for callers that want to print
    return df_folds, agg
