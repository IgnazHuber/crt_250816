import os
from datetime import datetime

import numpy as np
import pandas as pd


def show_fast_hvplot(df: pd.DataFrame,
                     asset_label: str,
                     asset_tag: str | None = None,
                     out_html: str | None = None,
                     use_datashade: bool = True) -> str | None:
    """Fast interactive viewer using HoloViz hvPlot + Datashader.
    Focus on scalable line/volume rendering; exports a Bokeh-based HTML.
    """
    try:
        import holoviews as hv
        import hvplot.pandas  # noqa: F401 - registers accessor
        hv.extension('bokeh')
    except Exception:
        print("[fastviz_hvplot] hvPlot/HoloViews not installed; skipping fast view.")
        return None

    dfx = df.copy()
    if 'date' not in dfx.columns:
        raise ValueError("DataFrame must contain 'date' column")
    dfx['date'] = pd.to_datetime(dfx['date'], utc=True, errors='coerce')
    # Bokeh expects tz-naive datetimes
    try:
        dfx['date'] = pd.to_datetime(dfx['date']).dt.tz_convert(None)
    except Exception:
        try:
            dfx['date'] = pd.to_datetime(dfx['date']).dt.tz_localize(None)
        except Exception:
            pass
    dfx = dfx.dropna(subset=['date']).sort_values('date').reset_index(drop=True)

    # Core lines with datashade where beneficial
    plots = []

    # Derive additional features (ATR_Pct, OBV, VWAP)
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
    # Detect datashader availability
    try:
        import datashader  # noqa: F401
        _has_ds = True
    except Exception:
        _has_ds = False
    use_ds = _has_ds and bool(use_datashade)

    if 'close' in dfx.columns:
        p_close = dfx.hvplot(x='date', y='close', title=f"{asset_label} — Close{' (datashaded)' if use_ds else ''}", datashade=use_ds, width=1000, height=300)
        # VWAP overlay if present
        try:
            if 'VWAP' in dfx.columns:
                p_close = (p_close * dfx.hvplot(x='date', y='VWAP', width=1000, height=300, color='#9b59b6')).opts(legend_position='top_left')
        except Exception:
            pass
        plots.append(p_close)
    if 'RSI' in dfx.columns:
        p_rsi = dfx.hvplot(x='date', y='RSI', title='RSI', width=1000, height=250)
        plots.append(p_rsi)
    # MACD series resolution: histogram preferred; else MACD - signal; else MACD line
    macd_series = None
    if 'MACD_histogram' in dfx.columns:
        macd_series = dfx[['date','MACD_histogram']].rename(columns={'MACD_histogram':'macd_v'})
        p_macd = macd_series.hvplot.step(x='date', y='macd_v', title='MACD Histogram', width=1000, height=250, color='#7f8c8d')
        plots.append(p_macd)
    else:
        if {'MACD','MACD_signal'}.issubset(dfx.columns):
            tmp = dfx[['date','MACD','MACD_signal']].copy()
            tmp['macd_v'] = tmp['MACD'] - tmp['MACD_signal']
            macd_series = tmp[['date','macd_v']]
            p_macd = macd_series.hvplot.step(x='date', y='macd_v', title='MACD Histogram', width=1000, height=250, color='#7f8c8d')
            plots.append(p_macd)
        elif 'MACD' in dfx.columns:
            macd_series = dfx[['date','MACD']].rename(columns={'MACD':'macd_v'})
            p_macd = macd_series.hvplot(x='date', y='macd_v', title='MACD', width=1000, height=250, color='#7f8c8d')
            plots.append(p_macd)
    if 'volume' in dfx.columns:
        # Color volume by up/down via two series
        try:
            up = (dfx['close'] >= dfx['close'].shift(1)).fillna(True)
            v_up = dfx[['date','volume']].copy(); v_up.loc[~up, 'volume'] = 0
            v_dn = dfx[['date','volume']].copy(); v_dn.loc[up, 'volume'] = 0
            p_vol = v_up.hvplot.step(x='date', y='volume', title='Volume', width=1000, height=250, color='#2ecc71') * \
                    v_dn.hvplot.step(x='date', y='volume', width=1000, height=250, color='#e74c3c')
        except Exception:
            p_vol = dfx.hvplot.step(x='date', y='volume', title='Volume', width=1000, height=250)
        plots.append(p_vol)
    if 'ATR_Pct' in dfx.columns:
        p_atr = dfx.hvplot(x='date', y='ATR_Pct', title='ATR %', width=1000, height=250, color='#34495e')
        plots.append(p_atr)
    if 'OBV' in dfx.columns:
        p_obv = dfx.hvplot(x='date', y='OBV', title='OBV', width=1000, height=250, color='#16a085')
        plots.append(p_obv)

    if not plots:
        print("[fastviz_hvplot] No recognized columns to plot (close/RSI/MACD_histogram/volume).")
        return None

    # Markers as overlay points on price/RSI/MACD (if possible)
    try:
        # Build markers similar to main logic
        mk = []
        cols = set(dfx.columns)
        def _add(dt, t): mk.append({'Date': dt, 'Type': t})
        if 'CBullD_gen' in cols:
            for dt in dfx.loc[dfx['CBullD_gen'] == 1, 'date']:
                _add(dt, 'CBullDivg_Classic')
        if 'CBullD_neg_MACD' in cols:
            for dt in dfx.loc[dfx['CBullD_neg_MACD'] == 1, 'date']:
                _add(dt, 'CBullDivg_Hidden')
        if 'CBullD_x2_gen' in cols:
            for dt in dfx.loc[dfx['CBullD_x2_gen'] == 1, 'date']:
                _add(dt, 'CBullDivg_x2_Classic')
        if 'HBearD_gen' in cols:
            for dt in dfx.loc[dfx['HBearD_gen'] == 1, 'date']:
                _add(dt, 'HBearDivg_Classic')
        if 'HBullD_gen' in cols:
            for dt in dfx.loc[dfx['HBullD_gen'] == 1, 'date']:
                _add(dt, 'HBullDivg_Classic')
        if 'HBullD_neg_MACD' in cols:
            for dt in dfx.loc[dfx['HBullD_neg_MACD'] == 1, 'date']:
                _add(dt, 'HBullDivg_Hidden')
        if mk and 'close' in dfx.columns:
            mk_df = pd.DataFrame(mk)
            mk_df = mk_df.merge(dfx[['date','close','RSI'] if 'RSI' in dfx.columns else ['date','close']], left_on='Date', right_on='date', how='left')
            if macd_series is not None:
                mk_df = mk_df.merge(macd_series, left_on='Date', right_on='date', how='left', suffixes=('','_m'))
            # Group to 4 divergences and map markers, larger size
            def _group(t: str) -> str:
                tl = str(t)
                if 'x2' in tl:
                    return 'CBullDivg_x2'
                if 'HBear' in tl:
                    return 'HBearDivg'
                if 'HBull' in tl:
                    return 'HBullDivg'
                return 'CBullDivg'
            mk_df['Group'] = mk_df['Type'].map(_group)
            sym_map = {
                'CBullDivg': 'triangle_up',
                'CBullDivg_x2': 'asterisk',
                'HBullDivg': 'square',
                'HBearDivg': 'triangle_down',
            }
            def color_for_group(g):
                return '#e74c3c' if g == 'HBearDivg' else '#2ecc71'
            # overlay on Price
            for g, sub in mk_df.groupby('Group'):
                pts = sub.hvplot.scatter(x='Date', y='close', color=color_for_group(g), marker=sym_map.get(g, 'circle'), size=10, alpha=0.85)
                plots[0] = plots[0] * pts
            # overlay on RSI if present
            if 'RSI' in dfx.columns:
                # Find RSI plot index (after price): search by title matching 'RSI'
                try:
                    rsi_idx = next(i for i,p in enumerate(plots) if hasattr(p, 'opts') and 'RSI' in str(p.opts.get('title')))
                except Exception:
                    rsi_idx = 1 if len(plots) > 1 else None
                if rsi_idx is not None:
                    for g, sub in mk_df.dropna(subset=['RSI']).groupby('Group') if 'RSI' in mk_df.columns else []:
                        pts = sub.hvplot.scatter(x='Date', y='RSI', color=color_for_group(g), marker=sym_map.get(g, 'circle'), size=9, alpha=0.8)
                        plots[rsi_idx] = plots[rsi_idx] * pts
            # overlay on MACD if present
            if macd_series is not None:
                try:
                    macd_idx = next(i for i,p in enumerate(plots) if hasattr(p, 'opts') and ('MACD' in str(p.opts.get('title'))))
                except Exception:
                    macd_idx = None
                if macd_idx is not None and 'macd_v' in mk_df.columns:
                    for g, sub in mk_df.dropna(subset=['macd_v']).groupby('Group'):
                        pts = sub.hvplot.scatter(x='Date', y='macd_v', color=color_for_group(g), marker=sym_map.get(g, 'circle'), size=9, alpha=0.8)
                        plots[macd_idx] = plots[macd_idx] * pts
    except Exception:
        pass

    # EMA overlays on price (detect EMA_* columns)
    try:
        ema_cols = [c for c in dfx.columns if c.upper().startswith('EMA_')]
        if ema_cols:
            overlay = None
            for c in sorted(ema_cols, key=lambda s: (len(s), s)):
                curve = dfx.hvplot(x='date', y=c, width=1000, height=300)
                overlay = curve if overlay is None else overlay * curve
            if overlay is not None:
                plots[0] = plots[0] * overlay.opts(alpha=0.9)
    except Exception:
        pass

    # Compact legend note (marker mapping) as on-chart text
    try:
        import holoviews as hv
        min_dt = pd.to_datetime(dfx['date']).min()
        y_top = float(dfx['close'].max()) if 'close' in dfx.columns else 1.0
        note = 'Markers: CBullDivg=△ (green), CBullDivg_x2=★ (green), HBullDivg=■ (green), HBearDivg=▽ (red)'
        text = hv.Text(min_dt, y_top, note).opts(text_color='#2c3e50', bgcolor='white', alpha=0.8, fontsize=10)
        plots[0] = plots[0] * text
    except Exception:
        pass

    layout = hv.Layout(plots).cols(1)

    os.makedirs('results', exist_ok=True)
    if out_html is None:
        out_html = os.path.join('results', f"fastviz_hvplot_{(asset_tag or 'asset')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
    try:
        hv.save(layout, out_html, resources='cdn')
        print(f"Fast viz (hvplot) saved: {out_html}")
        return out_html
    except Exception as e:
        print(f"[fastviz_hvplot] Failed to write HTML: {e}")
        return None
