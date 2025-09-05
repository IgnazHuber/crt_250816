import os
import pandas as pd
import numpy as np


def _build_markers_df(df: pd.DataFrame, analysis_results: dict) -> pd.DataFrame:
    """Extract markers from divergence columns if present.
    Mirrors the logic used by the main marker exporter without importing it.
    """
    markers = []

    def _append(t, dt):
        markers.append({'Type': t, 'Date': pd.to_datetime(dt, utc=True, errors='coerce')})

    for i in range(len(df)):
        dt = df['date'].iloc[i]
        row = df.iloc[i]
        if analysis_results.get('CBullDivg', False):
            if 'CBullD_gen' in df.columns and pd.notna(row.get('CBullD_gen')) and row.get('CBullD_gen') == 1:
                _append('CBullDivg_Classic', dt)
            if 'CBullD_neg_MACD' in df.columns and pd.notna(row.get('CBullD_neg_MACD')) and row.get('CBullD_neg_MACD') == 1:
                _append('CBullDivg_Hidden', dt)
        if analysis_results.get('CBullDivg_x2', False):
            if 'CBullD_x2_gen' in df.columns and pd.notna(row.get('CBullD_x2_gen')) and row.get('CBullD_x2_gen') == 1:
                _append('CBullDivg_x2_Classic', dt)
        if analysis_results.get('HBearDivg', False):
            if 'HBearD_gen' in df.columns and pd.notna(row.get('HBearD_gen')) and row.get('HBearD_gen') == 1:
                _append('HBearDivg_Classic', dt)
        if analysis_results.get('HBullDivg', False):
            if 'HBullD_gen' in df.columns and pd.notna(row.get('HBullD_gen')) and row.get('HBullD_gen') == 1:
                _append('HBullDivg_Classic', dt)
            if 'HBullD_neg_MACD' in df.columns and pd.notna(row.get('HBullD_neg_MACD')) and row.get('HBullD_neg_MACD') == 1:
                _append('HBullDivg_Hidden', dt)

    mk = pd.DataFrame(markers)
    if not mk.empty:
        mk['Date'] = pd.to_datetime(mk['Date'], utc=True, errors='coerce')
        mk = mk.dropna(subset=['Date'])
    return mk


def _heikin_ashi(df_ochl: pd.DataFrame) -> pd.DataFrame:
    """Compute Heikin-Ashi candles from a DataFrame with columns open, high, low, close.
    Returns a new DataFrame with the same columns (open, high, low, close).
    """
    o = df_ochl['open'].astype(float).values
    h = df_ochl['high'].astype(float).values
    l = df_ochl['low'].astype(float).values
    c = df_ochl['close'].astype(float).values
    ha_c = (o + h + l + c) / 4.0
    ha_o = np.empty_like(ha_c)
    ha_o[0] = (o[0] + c[0]) / 2.0
    for i in range(1, len(ha_o)):
        ha_o[i] = (ha_o[i-1] + ha_c[i-1]) / 2.0
    ha_h = np.maximum.reduce([h, ha_o, ha_c])
    ha_l = np.minimum.reduce([l, ha_o, ha_c])
    out = pd.DataFrame({'open': ha_o, 'high': ha_h, 'low': ha_l, 'close': ha_c}, index=df_ochl.index)
    return out


def show_fast_finplot(df: pd.DataFrame,
                      analysis_results: dict,
                      asset_label: str,
                      candle_style: str = 'ochl',
                      vol_up: str = '#2ecc71',
                      vol_down: str = '#e74c3c') -> None:
    """Fast local viewer using finplot (opens a window)."""
    try:
        import finplot as fplt
    except Exception:
        print("[fastviz_finplot] finplot not installed; pip install finplot PyQt5")
        return

    dfx = df.copy()
    if 'date' not in dfx.columns:
        raise ValueError("DataFrame must contain 'date' column")
    dfx['date'] = pd.to_datetime(dfx['date'], utc=True, errors='coerce')
    dfx = dfx.dropna(subset=['date']).sort_values('date').reset_index(drop=True)
    try:
        dfx.index = pd.to_datetime(dfx['date']).dt.tz_convert(None)
    except Exception:
        dfx.index = pd.to_datetime(dfx['date']).dt.tz_localize(None)

    # Main price axis
    ax_price = fplt.create_plot(f'{asset_label} — Fast Viewer (finplot)')

    # Candles or line (use datetime index consistently for all plots)
    if {'open', 'close', 'high', 'low'}.issubset(dfx.columns):
        # Choose candle style: 'ochl' (default) or 'ha' (Heikin-Ashi)
        style = (candle_style or 'ochl').strip().lower()
        base = dfx[['open', 'high', 'low', 'close']]
        if style == 'ha':
            base = _heikin_ashi(base)
        candles = pd.DataFrame({'open': base['open'], 'close': base['close'], 'high': base['high'], 'low': base['low']}, index=dfx.index)
        fplt.candlestick_ochl(candles, ax=ax_price)
    elif 'close' in dfx.columns:
        fplt.plot(dfx['close'], ax=ax_price, color='#1f77b4', legend='Close')

    # Volume subplot (finplot auto-colors by open/close); use datetime index
    if 'volume' in dfx.columns:
        ax_vol = fplt.create_plot('Volume', rows=1)
        vol = pd.DataFrame({'open': dfx['open'] if 'open' in dfx.columns else dfx['close'], 'close': dfx['close'], 'volume': dfx['volume']}, index=dfx.index)
        try:
            fplt.volume_ocv(vol[['open','close','volume']], ax=ax_vol)
        except TypeError:
            fplt.volume_ocv(vol[['open','close','volume']], ax=ax_vol)

    # RSI subplot
    rsi_col = 'RSI' if 'RSI' in dfx.columns else ('rsi' if 'rsi' in dfx.columns else None)
    if rsi_col is not None:
        ax_rsi = fplt.create_plot('RSI', rows=1)
        fplt.plot(dfx[rsi_col], ax=ax_rsi, color='#2ca02c', legend='RSI')
        # Markers on RSI
        try:
            mk = _build_markers_df(dfx.reset_index(drop=True).assign(date=pd.to_datetime(dfx.index)).copy(), analysis_results)
            if not mk.empty:
                style_map = {
                    'CBullDivg_Classic': '^',
                    'CBullDivg_Hidden': 'd',
                    'CBullDivg_x2_Classic': 'x',
                    'HBullDivg_Classic': 's',
                    'HBearDivg_Classic': 'v',
                    'HBullDivg_Hidden': 'o',
                }
                for t in sorted(mk['Type'].dropna().unique()):
                    sub = mk[mk['Type'] == t]
                    ts_sub = pd.to_datetime(sub['Date']).dt.tz_convert(None)
                    idx_sub = np.searchsorted(dfx.index.values, ts_sub.values, side='right') - 1
                    idx_sub = np.clip(idx_sub, 0, len(dfx) - 1)
                    ys = dfx[rsi_col].iloc[idx_sub]
                    color = '#e74c3c' if ('Bear' in t or 'HBear' in t) else '#2ecc71'
                    style = style_map.get(t, '^')
                    fplt.plot(pd.Series(ys.values, index=ts_sub), ax=ax_rsi, color=color, style=style, width=4, legend=str(t)+' RSI')
        except Exception:
            pass

    # MACD subplot (hist preferred; else MACD or MACD - signal)
    macd_series = None; macd_title = 'MACD'
    if 'MACD_histogram' in dfx.columns:
        macd_series = dfx['MACD_histogram']; macd_title = 'MACD Histogram'
    elif {'MACD','MACD_signal'}.issubset(dfx.columns):
        macd_series = dfx['MACD'] - dfx['MACD_signal']; macd_title = 'MACD Histogram'
    elif 'MACD' in dfx.columns:
        macd_series = dfx['MACD']; macd_title = 'MACD'
    if macd_series is not None:
        ax_macd = fplt.create_plot(macd_title, rows=1)
        fplt.plot(macd_series, ax=ax_macd, color='#7f8c8d', legend=macd_title)
        # Markers on MACD
        try:
            mk = _build_markers_df(dfx.reset_index(drop=True).assign(date=pd.to_datetime(dfx.index)).copy(), analysis_results)
            if not mk.empty:
                style_map = {
                    'CBullDivg_Classic': '^',
                    'CBullDivg_Hidden': 'd',
                    'CBullDivg_x2_Classic': 'x',
                    'HBullDivg_Classic': 's',
                    'HBearDivg_Classic': 'v',
                    'HBullDivg_Hidden': 'o',
                }
                for t in sorted(mk['Type'].dropna().unique()):
                    sub = mk[mk['Type'] == t]
                    ts_sub = pd.to_datetime(sub['Date']).dt.tz_convert(None)
                    idx_sub = np.searchsorted(dfx.index.values, ts_sub.values, side='right') - 1
                    idx_sub = np.clip(idx_sub, 0, len(dfx) - 1)
                    ys = macd_series.iloc[idx_sub]
                    color = '#e74c3c' if ('Bear' in t or 'HBear' in t) else '#2ecc71'
                    style = style_map.get(t, '^')
                    fplt.plot(pd.Series(ys.values, index=ts_sub), ax=ax_macd, color=color, style=style, width=4, legend=str(t)+' MACD')
        except Exception:
            pass

    # Markers on price
    mk = _build_markers_df(dfx.reset_index(drop=True).assign(date=pd.to_datetime(dfx.index)).copy(), analysis_results)
    if not mk.empty and 'close' in dfx.columns:
        for t in sorted(mk['Type'].dropna().unique()):
            sub = mk[mk['Type'] == t]
            ts_sub = pd.to_datetime(sub['Date']).dt.tz_convert(None)
            idx_sub = np.searchsorted(dfx.index.values, ts_sub.values, side='right') - 1
            idx_sub = np.clip(idx_sub, 0, len(dfx) - 1)
            ys_sub = dfx['close'].iloc[idx_sub]
            color = '#e74c3c' if ('Bear' in t or 'HBear' in t) else '#2ecc71'
            style_map = {
                'CBullDivg_Classic': '^',
                'CBullDivg_Hidden': 'd',
                'CBullDivg_x2_Classic': 'x',
                'HBullDivg_Classic': 's',
                'HBearDivg_Classic': 'v',
                'HBullDivg_Hidden': 'o',
            }
            style = style_map.get(t, '^')
            s = pd.Series(ys_sub.values, index=ts_sub)
            fplt.plot(s, ax=ax_price, color=color, style=style, width=5, legend=str(t))

    # Show window (blocks until closed)
    # Toolbar hint line & show
    try:
        # add a small hint overlay
        try:
            y_hint = float(dfx['close'].max()) if 'close' in dfx.columns else 0
            fplt.add_text((dfx.index[0], y_hint), 'Hints: drag=pan, wheel=zoom, right-drag=zoom-x, ctrl+drag=snap', color='#7f8c8d', ax=ax_price)
        except Exception:
            pass
        fplt.show()
    except Exception as e:
        print(f"[fastviz_finplot] show failed: {e}")
