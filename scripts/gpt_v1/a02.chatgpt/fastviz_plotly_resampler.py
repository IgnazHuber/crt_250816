import os
from datetime import datetime

import numpy as np
import pandas as pd


def _build_markers_df(df: pd.DataFrame,
                      analysis_results: dict,
                      candle_percent: float,
                      macd_percent: float,
                      asset_tag: str | None = None) -> pd.DataFrame:
    """Lightweight recreation of marker extraction to avoid importing Mainframe.
    Uses the same column names the divergence engines produce.
    """
    markers = []

    def _append(t, dt):
        markers.append({
            'Type': t,
            'Date': pd.to_datetime(dt, utc=True, errors='coerce'),
            'Candle_Percent': candle_percent,
            'MACD_Percent': macd_percent,
            'Asset_Tag': asset_tag or ''
        })

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


def show_fast_plot(df: pd.DataFrame,
                   analysis_results: dict,
                   candle_percent: float,
                   macd_percent: float,
                   asset_label: str,
                   asset_tag: str | None = None,
                   out_html: str | None = None) -> str | None:
    """Fast interactive viewer using plotly-resampler.
    Builds resampled plots for Close/Volume/RSI/MACD and overlays markers.
    Returns the HTML path if written, else None.
    """
    try:
        from plotly_resampler import FigureResampler
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception:
        print("[fastviz_plotly_resampler] plotly-resampler not installed; skipping fast view.")
        return None

    dfx = df.copy()
    if 'date' not in dfx.columns:
        raise ValueError("DataFrame must contain 'date' column")
    dfx['date'] = pd.to_datetime(dfx['date'], utc=True, errors='coerce')
    dfx = dfx.dropna(subset=['date']).sort_values('date').reset_index(drop=True)

    # Derive additional panels if possible (ATR%, OBV, VWAP)
    try:
        if {'high','low','close'}.issubset(dfx.columns):
            prev_close = dfx['close'].shift(1)
            tr = (dfx['high'] - dfx['low']).abs()
            tr = np.maximum(tr, (dfx['high'] - prev_close).abs())
            tr = np.maximum(tr, (dfx['low'] - prev_close).abs())
            dfx['ATR_14'] = pd.Series(tr).rolling(14, min_periods=1).mean()
            with np.errstate(divide='ignore', invalid='ignore'):
                dfx['ATR_Pct'] = (dfx['ATR_14'] / dfx['close']) * 100.0
        if 'volume' in dfx.columns and 'close' in dfx.columns:
            delta = dfx['close'].diff().fillna(0)
            obv = (np.sign(delta).replace(0, 0) * dfx['volume']).cumsum()
            dfx['OBV'] = obv
        if {'high','low','close','volume'}.issubset(dfx.columns):
            typical = (dfx['high'] + dfx['low'] + dfx['close']) / 3.0
            cum_pv = (typical * dfx['volume']).cumsum()
            cum_v = dfx['volume'].cumsum().replace(0, np.nan)
            dfx['VWAP'] = cum_pv / cum_v
    except Exception:
        pass

    # Compute RSI/MACD panels if present; otherwise skip gracefully
    cols = set(dfx.columns)
    has_rsi = 'RSI' in cols or 'rsi' in cols
    rsi_col = 'RSI' if 'RSI' in cols else ('rsi' if 'rsi' in cols else None)
    # Resolve MACD series: prefer histogram; else compute from MACD - signal; else use MACD line
    macd_hist_col = 'MACD_histogram' if 'MACD_histogram' in cols else ('macd_histogram' if 'macd_histogram' in cols else None)
    macd_line_col = 'MACD' if 'MACD' in cols else ('macd' if 'macd' in cols else None)
    macd_signal_col = 'MACD_signal' if 'MACD_signal' in cols else ('macd_signal' if 'macd_signal' in cols else None)
    if macd_hist_col is not None:
        has_macd = True
        macd_mode = 'hist'
        macd_series = dfx[macd_hist_col].copy()
    elif macd_line_col is not None and macd_signal_col is not None:
        has_macd = True
        macd_mode = 'hist'
        macd_series = (dfx[macd_line_col] - dfx[macd_signal_col]).copy()
    elif macd_line_col is not None:
        has_macd = True
        macd_mode = 'line'
        macd_series = dfx[macd_line_col].copy()
    else:
        has_macd = False
        macd_mode = None
        macd_series = pd.Series(dtype=float)
    has_volume = 'volume' in cols

    # Layout similar to existing: Price, RSI, MACD, Volume
    rows = 3 + (1 if has_volume else 0) + (1 if 'ATR_Pct' in dfx.columns else 0) + (1 if 'OBV' in dfx.columns else 0) + (1 if 'VWAP' in dfx.columns else 0)
    specs = [[{"secondary_y": False}] for _ in range(rows)]
    fig = FigureResampler(make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.02, specs=specs))

    # Row index helper
    r_price, r_rsi, r_macd = 1, 2, 3
    next_row = 4
    r_vol = next_row if has_volume else None
    next_row = next_row + (1 if has_volume else 0)
    r_atr = next_row if 'ATR_Pct' in dfx.columns else None
    next_row = next_row + (1 if r_atr else 0)
    r_obv = next_row if 'OBV' in dfx.columns else None
    next_row = next_row + (1 if r_obv else 0)
    r_vwap = next_row if 'VWAP' in dfx.columns else None

    # Price (line for speed) + VWAP overlay on same row when available
    if 'close' in dfx.columns:
        fig.add_trace(go.Scattergl(name='Close', line=dict(color='#1f77b4', width=1.2)), hf_x=dfx['date'], hf_y=dfx['close'], row=r_price, col=1)
        if 'VWAP' in dfx.columns:
            fig.add_trace(go.Scattergl(name='VWAP', line=dict(color='#9b59b6', width=1)), hf_x=dfx['date'], hf_y=dfx['VWAP'], row=r_price, col=1)
    elif 'price' in dfx.columns:
        fig.add_trace(go.Scattergl(name='Price', line=dict(color='#1f77b4', width=1)), hf_x=dfx['date'], hf_y=dfx['price'], row=r_price, col=1)

    # RSI
    if rsi_col is not None:
        fig.add_trace(go.Scattergl(name='RSI', line=dict(color='#2ca02c', width=1.1)), hf_x=dfx['date'], hf_y=dfx[rsi_col], row=r_rsi, col=1)
        # RSI bands 70/30
        try:
            fig.add_trace(go.Scattergl(name='RSI 70', line=dict(color='#bdc3c7', width=0.8, dash='dot')), hf_x=dfx['date'], hf_y=pd.Series(70.0, index=dfx.index), row=r_rsi, col=1)
            fig.add_trace(go.Scattergl(name='RSI 30', line=dict(color='#bdc3c7', width=0.8, dash='dot')), hf_x=dfx['date'], hf_y=pd.Series(30.0, index=dfx.index), row=r_rsi, col=1)
        except Exception:
            pass

    # MACD histogram
    if has_macd:
        if macd_mode == 'hist':
            macd = macd_series.fillna(0.0)
            pos = macd.clip(lower=0)
            neg = macd.clip(upper=0)
            fig.add_trace(go.Bar(name='MACD +', marker_color='#27ae60'), hf_x=dfx['date'], hf_y=pos, row=r_macd, col=1)
            fig.add_trace(go.Bar(name='MACD -', marker_color='#c0392b'), hf_x=dfx['date'], hf_y=neg, row=r_macd, col=1)
        else:
            fig.add_trace(go.Scattergl(name='MACD', line=dict(color='#7f8c8d', width=1)), hf_x=dfx['date'], hf_y=macd_series, row=r_macd, col=1)

    # Volume
    if has_volume and r_vol is not None:
        # Color volume by up/down
        try:
            up = (dfx['close'] >= dfx['close'].shift(1)).fillna(True)
            v_up = dfx['volume'].where(up, other=0)
            v_dn = dfx['volume'].where(~up, other=0)
            fig.add_trace(go.Bar(name='Volume +', marker_color='#2ecc71'), hf_x=dfx['date'], hf_y=v_up, row=r_vol, col=1)
            fig.add_trace(go.Bar(name='Volume -', marker_color='#e74c3c'), hf_x=dfx['date'], hf_y=v_dn, row=r_vol, col=1)
        except Exception:
            fig.add_trace(go.Bar(name='Volume', marker_color='#95a5a6'), hf_x=dfx['date'], hf_y=dfx['volume'], row=r_vol, col=1)

    # ATR %
    if r_atr is not None:
        fig.add_trace(go.Scattergl(name='ATR %', line=dict(color='#34495e', width=1)), hf_x=dfx['date'], hf_y=dfx['ATR_Pct'], row=r_atr, col=1)
    # OBV
    if r_obv is not None:
        fig.add_trace(go.Scattergl(name='OBV', line=dict(color='#16a085', width=1)), hf_x=dfx['date'], hf_y=dfx['OBV'], row=r_obv, col=1)

    # Markers (overlay on price, RSI, MACD) grouped into 4 divergences
    mk = _build_markers_df(dfx, analysis_results, candle_percent, macd_percent, asset_tag)
    if not mk.empty:
        # Map raw Types to 4 analysis groups
        def _group(t: str) -> str:
            tl = str(t)
            if 'x2' in tl:
                return 'CBullDivg_x2'
            if 'HBear' in tl:
                return 'HBearDivg'
            if 'HBull' in tl:
                return 'HBullDivg'
            return 'CBullDivg'
        sym_map = {
            'CBullDivg': 'triangle-up',
            'CBullDivg_x2': 'star',
            'HBullDivg': 'square',
            'HBearDivg': 'triangle-down',
        }
        dt_series = pd.to_datetime(dfx['date']).values
        mk['Group'] = mk['Type'].map(_group)
        for g, grp in mk.groupby('Group'):
            xs = pd.to_datetime(grp['Date']).dt.tz_convert(None).values
            idx = np.searchsorted(dt_series, xs, side='right') - 1
            idx = np.clip(idx, 0, len(dfx) - 1)
            # Price Y
            ys_p = dfx['close'].iloc[idx].values if 'close' in dfx.columns else np.full(len(idx), np.nan)
            # Color by direction: HBear red, others green
            col = '#e74c3c' if g == 'HBearDivg' else '#2ecc71'
            sym = sym_map.get(g, 'circle')
            # Price markers
            fig.add_trace(go.Scattergl(name=g, mode='markers', marker=dict(size=11, symbol=sym, color=col, opacity=0.9)), hf_x=xs, hf_y=ys_p, row=r_price, col=1)
            # RSI markers
            if rsi_col is not None:
                ys_r = dfx[rsi_col].iloc[idx].values
                fig.add_trace(go.Scattergl(name=g+' RSI', mode='markers', marker=dict(size=10, symbol=sym, color=col, opacity=0.85)), hf_x=xs, hf_y=ys_r, row=r_rsi, col=1)
            # MACD markers
            if has_macd:
                ys_m = macd_series.iloc[idx].values
                fig.add_trace(go.Scattergl(name=g+' MACD', mode='markers', marker=dict(size=10, symbol=sym, color=col, opacity=0.85)), hf_x=xs, hf_y=ys_m, row=r_macd, col=1)

    fig.update_layout(
        template='plotly_white',
        title=dict(text=f"{asset_label} — Fast Viewer (plotly-resampler)", x=0.01),
        showlegend=True,
        height=max(900, 220 * rows),
        margin=dict(l=60, r=20, t=60, b=40),
    )
    try:
        fig.update_xaxes(showgrid=True, gridcolor='#ecf0f1')
        fig.update_yaxes(showgrid=True, gridcolor='#ecf0f1')
    except Exception:
        pass

    # Compact legend note (marker mapping)
    try:
        legend_txt = (
            "Markers: "
            "CBullDivg=△ (green), "
            "CBullDivg_x2=★ (green), "
            "HBullDivg=■ (green), "
            "HBearDivg=▽ (red)"
        )
        fig.add_annotation(
            xref='paper', yref='paper', x=0.01, y=0.99, xanchor='left', yanchor='top',
            text=legend_txt,
            font=dict(size=12, color='#2c3e50'),
            align='left', showarrow=False,
            bordercolor='#bdc3c7', borderwidth=1, borderpad=4,
            bgcolor='rgba(255,255,255,0.8)'
        )
    except Exception:
        pass

    os.makedirs('results', exist_ok=True)
    if out_html is None:
        out_html = os.path.join('results', f"fastviz_plotly_resampler_{(asset_tag or 'asset')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
    try:
        fig.write_html(out_html, include_plotlyjs='cdn')
        print(f"Fast viz (plotly-resampler) saved: {out_html}")
        return out_html
    except Exception as e:
        print(f"[fastviz_plotly_resampler] Failed to write HTML: {e}")
        return None
