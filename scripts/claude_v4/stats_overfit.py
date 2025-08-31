import os
from datetime import datetime
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from Backtest_Divergences import BacktestParams, backtest as run_backtest


def _fold_ranges(dates: pd.Series, n_splits: int, embargo_frac: float):
    idx = np.linspace(0, len(dates), n_splits + 1, dtype=int)
    ranges = []
    for k in range(n_splits):
        lo, hi = idx[k], idx[k + 1]
        e = max(1, int((hi - lo) * embargo_frac))
        start_i = lo + e
        end_i = max(start_i + 1, hi - e)
        if start_i < end_i:
            ranges.append((dates.iloc[start_i], dates.iloc[end_i - 1]))
    return ranges


def run_pbo_deflated_sharpe(df: pd.DataFrame, markers: pd.DataFrame, out_xlsx: str, out_html: str, asset_label: str):
    """PBO/Deflated Sharpe scaffold with explanatory outputs.
    Methodology (surrogate):
    - Split time into K purged folds. For each fold k:
      • IS = all data outside fold k; OOS = fold k only
      • Backtest both IS and OOS with the same markers filtered to respective spans
      • Record KPI (Total_PnL, PF, Trades)
    - PBO surrogate: share of folds where OOS Total_PnL is below the median of IS Total_PnL across folds.
      (True PBO requires multiple competing models; here we approximate across folds.)
    - Deflated Sharpe surrogate: mean/std of OOS Total_PnL across folds (as a coarse proxy for risk‑adjusted performance).
    Limits: This is an approximation; for rigorous PBO/DSR you need multiple model paths and return distributions.
    """
    if df is None or df.empty or 'date' not in df.columns:
        raise ValueError('Dataframe must have a date column')
    if markers is None or markers.empty or 'Date' not in markers.columns:
        raise ValueError('Markers must include a Date column')

    n_splits = int(os.getenv('PBO_SPLITS', os.getenv('WFCV_SPLITS', '5')))
    embargo = float(os.getenv('PBO_EMBARGO_PCT', os.getenv('WFCV_EMBARGO_PCT', '0.02')))

    dates = pd.to_datetime(df['date']).reset_index(drop=True)
    ranges = _fold_ranges(dates, n_splits, embargo)

    # Backtest params (respect env)
    fee_def = float(os.getenv('BT_FEE_PCT', '0.0') or 0.0)
    slip_def = float(os.getenv('BT_SLIPPAGE_PCT', '0.0') or 0.0)
    params = BacktestParams(fee_pct=fee_def, slippage_pct=slip_def)

    markers_local = markers.copy()
    markers_local['Date'] = pd.to_datetime(markers_local['Date'], utc=True, errors='coerce')
    df_local = df.copy()
    df_local['date'] = pd.to_datetime(df_local['date'], utc=True, errors='coerce')

    rows = []
    is_pnls = []
    oos_pnls = []
    for i, (start, end) in enumerate(ranges, 1):
        # OOS fold
        oos_df = df_local[(df_local['date'] >= start) & (df_local['date'] <= end)].copy()
        oos_mk = markers_local[(markers_local['Date'] >= start) & (markers_local['Date'] <= end)].copy()
        # IS complement
        is_df = df_local[(df_local['date'] < start) | (df_local['date'] > end)].copy()
        is_mk = markers_local[(markers_local['Date'] < start) | (markers_local['Date'] > end)].copy()

        def _kpis(dfx, mkx):
            if dfx.empty or mkx.empty:
                return {'Total_PnL': 0.0, 'PF': np.nan, 'Trades': 0}
            res = run_backtest(dfx.copy(), mkx.copy(), params)
            summ = res.get('summary', pd.DataFrame())
            if summ is None or summ.empty:
                return {'Total_PnL': 0.0, 'PF': np.nan, 'Trades': 0}
            s = summ.iloc[0]
            return {
                'Total_PnL': float(s.get('Total_PnL', 0.0)),
                'PF': float(s.get('Profit_Factor', np.nan)),
                'Trades': int(s.get('Total_Trades', 0)),
            }

        is_kpi = _kpis(is_df, is_mk)
        oos_kpi = _kpis(oos_df, oos_mk)
        is_pnls.append(is_kpi['Total_PnL'])
        oos_pnls.append(oos_kpi['Total_PnL'])
        rows.append({
            'Fold': i, 'Start': start, 'End': end,
            'IS_Trades': is_kpi['Trades'], 'IS_Total_PnL': is_kpi['Total_PnL'], 'IS_PF': is_kpi['PF'],
            'OOS_Trades': oos_kpi['Trades'], 'OOS_Total_PnL': oos_kpi['Total_PnL'], 'OOS_PF': oos_kpi['PF'],
        })

    df_folds = pd.DataFrame(rows)
    # Surrogates
    pbo_surrogate = None
    corr_spear = None
    ds_sharpe = None
    if len(is_pnls) >= 2 and len(oos_pnls) >= 2:
        try:
            med_is = float(np.median(is_pnls))
            pbo_surrogate = float(np.mean([1.0 if v < med_is else 0.0 for v in oos_pnls]))
        except Exception:
            pass
        try:
            corr_spear = float(spearmanr(is_pnls, oos_pnls, nan_policy='omit').correlation)
        except Exception:
            pass
        try:
            mu = float(np.mean(oos_pnls))
            sigma = float(np.std(oos_pnls, ddof=1)) if len(oos_pnls) > 1 else 0.0
            ds_sharpe = mu / sigma if sigma > 0 else np.nan
        except Exception:
            pass

    os.makedirs(os.path.dirname(out_xlsx) or '.', exist_ok=True)
    try:
        with pd.ExcelWriter(out_xlsx, engine='openpyxl') as w:
            info = pd.DataFrame({
                'Asset': [asset_label],
                'Rows': [len(df_local)],
                'Markers': [len(markers_local)],
                'Splits': [n_splits],
                'Embargo%': [embargo * 100.0],
                'Generated': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
            })
            info.to_excel(w, sheet_name='Info', index=False)
            df_folds.to_excel(w, sheet_name='IS_OOS_KPIs', index=False)
            summ = pd.DataFrame([{
                'PBO_surrogate': pbo_surrogate,
                'Spearman_IS_OOS': corr_spear,
                'Sharpe_surrogate_OOS': ds_sharpe,
            }])
            summ.to_excel(w, sheet_name='Summary', index=False)
    except Exception:
        df_folds.to_csv(out_xlsx.replace('.xlsx', '_is_oos.csv'), index=False)
        pd.DataFrame([{'PBO_surrogate': pbo_surrogate, 'Spearman_IS_OOS': corr_spear, 'Sharpe_surrogate_OOS': ds_sharpe}]).to_csv(out_xlsx.replace('.xlsx', '_summary.csv'), index=False)

    # HTML with explanation
    with open(out_html, 'w', encoding='utf-8') as f:
        f.write('<html><head><style>table{border-collapse:collapse}td,th{border:1px solid #555;padding:4px 6px} .num{text-align:right}</style></head><body>')
        f.write('<h3>PBO / Deflated Sharpe (Surrogate)</h3>')
        f.write(f'<p>Asset: {asset_label} | Splits: {n_splits} | Embargo: {embargo*100:.1f}%</p>')
        f.write('<p><b>Method (surrogate):</b> For each purged fold k, we backtest In‑Sample (IS) on all data except fold k and Out‑of‑Sample (OOS) on fold k. '
                'We record Total_PnL and PF for IS and OOS. PBO surrogate is the share of folds where OOS PnL is below the median of IS PnL. '
                'We also report Spearman rank correlation of IS vs OOS PnL and a simple OOS Sharpe surrogate μ/σ across folds. '
                'True PBO/Deflated Sharpe require multiple competing models and return distributions; treat these as diagnostic proxies.</p>')
        if not df_folds.empty:
            f.write('<h4>Per‑Fold IS/OOS KPIs</h4>')
            df_fmt = df_folds.copy()
            for c in ['IS_Total_PnL','OOS_Total_PnL','IS_PF','OOS_PF']:
                if c in df_fmt.columns:
                    df_fmt[c] = df_fmt[c].map(lambda v: f"{v:.2f}" if pd.notna(v) else "")
            f.write(df_fmt.to_html(index=False, border=0))
        f.write('<h4>Summary</h4>')
        f.write('<ul>')
        f.write(f"<li>PBO surrogate: {pbo_surrogate if pbo_surrogate is not None else 'n/a'}</li>")
        f.write(f"<li>Spearman(IS,OOS): {corr_spear if corr_spear is not None else 'n/a'}</li>")
        f.write(f"<li>Sharpe surrogate (OOS μ/σ): {f'{ds_sharpe:.2f}' if ds_sharpe is not None and not np.isnan(ds_sharpe) else 'n/a'}</li>")
        f.write('</ul>')
        f.write('</body></html>')
