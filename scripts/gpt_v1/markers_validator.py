import os
import io
import base64
import glob
import subprocess
from datetime import datetime
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd


try:
    import openpyxl  # noqa: F401
    _HAS_XLSX = True
except Exception:
    _HAS_XLSX = False


# Fixed divergence groups for reporting (exactly 4)
GROUPS = ['CBullDivg', 'CBullDivg_x2', 'HBullDivg', 'HBearDivg']


def _ps_open_dialog(multiselect: bool, title: str, filter_str: str, initial_dir: str | None = None) -> List[str] | str:
    """Windows Forms OpenFileDialog via PowerShell. Returns list[str] for multiselect, else str or '' on cancel."""
    try:
        init_dir = initial_dir or os.getcwd()
        ps = rf'''
Add-Type -AssemblyName System.Windows.Forms
$dlg = New-Object System.Windows.Forms.OpenFileDialog
$dlg.Filter = "{filter_str}"
$dlg.Title = "{title}"
$dlg.InitialDirectory = "{init_dir}"
$dlg.Multiselect = {'$true' if multiselect else '$false'}
if ($dlg.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) {{
  {'Write-Output ($dlg.FileNames -join "|")' if multiselect else 'Write-Output $dlg.FileName'}
}}
'''
        result = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps],
                                capture_output=True, text=True, timeout=60)
        if result.returncode == 0 and result.stdout:
            s = result.stdout.strip()
            if multiselect:
                return [p for p in s.split('|') if p]
            return s
    except Exception:
        pass
    return [] if multiselect else ''


def _console_select_files(patterns: List[str], allow_custom: bool = True) -> List[str]:
    print("\nAvailable files:")
    candidates: List[str] = []
    for pat in patterns:
        try:
            candidates.extend([p for p in glob.glob(pat)])
        except Exception:
            continue
    candidates = sorted(dict.fromkeys(candidates))
    for i, p in enumerate(candidates, 1):
        print(f"{i}: {p}")
    print(f"0: Cancel")
    if allow_custom:
        print(f"{len(candidates)+1}: Enter custom path")
    sel: List[str] = []
    while True:
        try:
            s = input("Select (comma-separated indexes): ").strip()
        except (EOFError, KeyboardInterrupt):
            return []
        if not s:
            return []
        if s == '0':
            return []
        if allow_custom and s == str(len(candidates)+1):
            try:
                p = input("Enter full path: ").strip()
                return [p] if p else []
            except (EOFError, KeyboardInterrupt):
                return []
        try:
            idxs = [int(x) for x in s.split(',')]
            for k in idxs:
                if 1 <= k <= len(candidates):
                    sel.append(candidates[k-1])
            return sel
        except Exception:
            print("Invalid input. Try again.")


def _direction_from_type(t: str) -> int:
    tl = str(t or '').lower()
    if 'hbear' in tl or 'bear' in tl:
        return -1
    return 1


def _align_entry_index(dates: pd.DatetimeIndex, signal_ts: pd.Timestamp) -> int:
    pos = dates.searchsorted(signal_ts, side='right')
    return int(pos) if pos < len(dates) else -1


def _validate_marker(df: pd.DataFrame,
                     entry_i: int,
                     direction: int,
                     lookahead: int,
                     hit_pct: float,
                     stop_pct: float) -> Dict[str, Any]:
    """Evaluate a single marker from entry index forward."""
    n = len(df)
    hi = min(n-1, entry_i + lookahead)
    entry_open = float(df['open'].iloc[entry_i])
    # levels
    if direction > 0:
        hit_level = entry_open * (1.0 + hit_pct/100.0)
        stop_level = entry_open * (1.0 - stop_pct/100.0)
    else:
        hit_level = entry_open * (1.0 - hit_pct/100.0)  # favorable downward
        stop_level = entry_open * (1.0 + stop_pct/100.0)

    mfe = -1e18
    mae = 1e18
    event = 'none'
    event_bar = None
    event_price = None

    # conservative ordering: stop before hit on the same bar
    for j in range(entry_i, hi+1):
        h = float(df['high'].iloc[j])
        l = float(df['low'].iloc[j])
        if direction > 0:
            # update excursions
            mfe = max(mfe, (h/entry_open) - 1.0)
            mae = min(mae, (l/entry_open) - 1.0)
            if l <= stop_level:
                event = 'stop'; event_bar = j; event_price = stop_level; break
            if h >= hit_level:
                event = 'hit'; event_bar = j; event_price = hit_level; break
        else:
            mfe = max(mfe, (entry_open/l) - 1.0 if l > 0 else 0.0)
            mae = min(mae, 1.0 - (entry_open/h) if h > 0 else 0.0)
            if h >= stop_level:
                event = 'stop'; event_bar = j; event_price = stop_level; break
            if l <= hit_level:
                event = 'hit'; event_bar = j; event_price = hit_level; break

    bars_to_event = (event_bar - entry_i) if event_bar is not None else None
    return {
        'Entry_Open': entry_open,
        'Hit_Level': hit_level,
        'Stop_Level': stop_level,
        'Event': event,
        'Event_Bars': int(bars_to_event) if bars_to_event is not None else None,
        'Event_Price': float(event_price) if event_price is not None else None,
        'MFE_%': float(mfe*100.0 if mfe!=-1e18 else np.nan),
        'MAE_%': float(mae*100.0 if mae!=1e18 else np.nan),
    }


def _normalize_ohlc_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols_lower = {c.lower(): c for c in df.columns}

    def pick(alias_list: List[str]):
        for a in alias_list:
            if a in df.columns:
                return a
            al = a.lower()
            if al in cols_lower:
                return cols_lower[al]
        return None

    # date column
    if 'date' not in df.columns:
        date_col = pick(['date', 'Date', 'timestamp', 'Timestamp', 'time', 'Time', 'datetime', 'Datetime'])
        if date_col is not None:
            df['date'] = df[date_col]

    alias = {
        'open':   ['open','Open','o','open_price','OpenPrice','price_open'],
        'high':   ['high','High','h','high_price','HighPrice','price_high'],
        'low':    ['low','Low','l','low_price','LowPrice','price_low'],
        'close':  ['close','Close','c','close_price','ClosePrice','price_close','price','Price'],
        'volume': ['volume','Volume','vol','Vol','base_volume','quote_volume','Volume_USD']
    }
    for k, al in alias.items():
        if k not in df.columns:
            col = pick(al)
            if col is not None:
                df[k] = df[col]

    # Last resort fallback
    if 'close' in df.columns:
        for k in ('open','high','low'):
            if k not in df.columns:
                df[k] = df['close']

    # Ensure numeric
    for k in ('open','high','low','close','volume'):
        if k in df.columns:
            try:
                df[k] = pd.to_numeric(df[k], errors='coerce')
            except Exception:
                pass
    return df


def validate_markers(ohlc_path: str, marker_paths: List[str], lookahead: int, hit_pct: float, stop_pct: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Load OHLC
    if ohlc_path.lower().endswith('.parquet'):
        df = pd.read_parquet(ohlc_path)
    else:
        df = pd.read_csv(ohlc_path, low_memory=False)
    df = _normalize_ohlc_columns(df)
    df['date'] = pd.to_datetime(df['date'], utc=True, errors='coerce')
    df = df.dropna(subset=['date']).sort_values('date').reset_index(drop=True)
    # Ensure OHLC exist; fallback to any available price column
    required = ['open','high','low','close']
    if any(c not in df.columns for c in required):
        # try to pick a generic price column and synthesize
        price_col = None
        for cand in ['close','Close','price','Price','Adj Close','adj_close','c']:
            if cand in df.columns:
                price_col = cand; break
        if price_col is not None:
            for k in required:
                if k not in df.columns:
                    df[k] = pd.to_numeric(df[price_col], errors='coerce')
        # if still missing, raise with diagnostics
        missing = [k for k in required if k not in df.columns]
        if missing:
            raise ValueError(f"OHLC missing columns {missing}; available: {list(df.columns)}")
    dates = pd.DatetimeIndex(df['date'])

    # Load markers
    rows = []
    for p in marker_paths:
        try:
            mk = pd.read_csv(p)
            if 'Date' not in mk.columns or 'Type' not in mk.columns:
                print(f"Skipping {p}: no Date/Type columns")
                continue
            mk['Date'] = pd.to_datetime(mk['Date'], utc=True, errors='coerce')
            mk = mk.dropna(subset=['Date'])
            mk['SourceCSV'] = os.path.basename(p)
            sel_cols = ['Date','Type','SourceCSV']
            if 'Candle_Percent' in mk.columns:
                sel_cols.append('Candle_Percent')
            if 'MACD_Percent' in mk.columns:
                sel_cols.append('MACD_Percent')
            rows.append(mk[sel_cols])
        except Exception as e:
            print(f"Failed to read markers {p}: {e}")
    if not rows:
        return pd.DataFrame(), pd.DataFrame()
    markers = pd.concat(rows, ignore_index=True).sort_values('Date').reset_index(drop=True)

    # Evaluate
    out = []
    for _, r in markers.iterrows():
        ts = pd.to_datetime(r['Date'], utc=True, errors='coerce')
        if pd.isna(ts):
            continue
        ei = _align_entry_index(dates, ts)
        if ei < 0:
            continue
        direction = _direction_from_type(r['Type'])
        res = _validate_marker(df, ei, direction, lookahead, hit_pct, stop_pct)
        out.append({
            'Marker_Date': ts,
            'Entry_Date': df['date'].iloc[ei],
            'Type': r['Type'],
            'Direction': 'Long' if direction>0 else 'Short',
            'Candle_Percent': r.get('Candle_Percent', np.nan),
            'MACD_Percent': r.get('MACD_Percent', np.nan),
            'SourceCSV': r.get('SourceCSV',''),
            'Lookahead_Bars': lookahead,
            'Hit_Threshold_%': hit_pct,
            'Stop_Threshold_%': stop_pct,
            **res,
            'Hit': True if res['Event']=='hit' else False,
        })
    per_marker = pd.DataFrame(out)
    # Map detailed Type (Classic/Hidden) to 4 analysis groups for reporting
    if not per_marker.empty:
        def _group_name(t: str) -> str:
            tl = str(t or '')
            if 'x2' in tl:
                return 'CBullDivg_x2'
            if 'HBear' in tl:
                return 'HBearDivg'
            if 'HBull' in tl:
                return 'HBullDivg'
            return 'CBullDivg'
        per_marker['Group'] = per_marker['Type'].map(_group_name)
    # Summary by Type
    if per_marker.empty:
        return per_marker, pd.DataFrame()
    per_marker['Group'] = per_marker['Group'].fillna('Unknown').astype(str)
    g = per_marker.groupby('Group')
    summary = pd.DataFrame({
        'Markers': g.size(),
        'Hit_Rate_%': g['Hit'].mean()*100.0,
        'Avg_Bars_to_Event': g['Event_Bars'].mean(),
        'MFE_%_mean': g['MFE_%'].mean(),
        'MAE_%_mean': g['MAE_%'].mean(),
    }).reset_index().rename(columns={'Group':'Type'}).sort_values('Hit_Rate_%', ascending=False)
    return per_marker, summary


def _prompt_config() -> Tuple[int, float, float]:
    def _ask_int(prompt_text: str, default: int) -> int:
        try:
            s = input(f"{prompt_text} (default {default}): ").strip()
            return int(s) if s != '' else default
        except Exception:
            return default
    def _ask_float(prompt_text: str, default: float) -> float:
        try:
            s = input(f"{prompt_text} (default {default}): ").strip()
            return float(s) if s != '' else default
        except Exception:
            return default
    print("\nValidation parameters:")
    lookahead = _ask_int('Lookahead bars', 20)
    hit_pct = _ask_float('Hit threshold %', 1.5)
    stop_pct = _ask_float('Stop threshold % (opposite move)', 1.5)
    return lookahead, hit_pct, stop_pct


def _explanation_df(per_marker: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    def add(sec, txt):
        rows.append({'Abschnitt': sec, 'Erläuterung': txt})
    # Randbedingungen
    try:
        lk = per_marker['Lookahead_Bars'].dropna().astype(int)
        hp = per_marker['Hit_Threshold_%'].dropna().astype(float)
        sp = per_marker['Stop_Threshold_%'].dropna().astype(float)
        def _fmt_range(s):
            if s.empty:
                return 'n/a'
            a, b = (int(s.min()) if s.dtype.kind in 'iu' else float(s.min())), (int(s.max()) if s.dtype.kind in 'iu' else float(s.max()))
            return f"{a}" if a == b else f"{a} – {b}"
        add('Randbedingungen', f"Lookahead (Bars): {_fmt_range(lk)}; Hit-Schwelle (%): {_fmt_range(hp)}; Stop-Schwelle (%): {_fmt_range(sp)}")
        add('Randbedingungen', 'Lookahead: Zeitraum (in Bars), in dem Hit/Stop gezählt werden — außerhalb findet keine Wertung statt.')
        add('Randbedingungen', 'Hit-/Stop-Schwellen: prozentual relativ zum Entry-Kurs (Open der nächsten Bar).')
        add('Randbedingungen', 'Konservative Reihenfolge: Wenn Hit und Stop in derselben Bar erreichbar sind, zählt Stop zuerst.')
    except Exception:
        add('Randbedingungen', 'Lookahead/Hit/Stop: n/a')
    # Gesamtergebnis in Prozent
    try:
        cnt = per_marker.shape[0]
        hits = int((per_marker['Event'].str.lower()=='hit').sum())
        stops = int((per_marker['Event'].str.lower()=='stop').sum())
        nones = cnt - hits - stops
        add('Gesamt', f"Marker gesamt: {cnt}; Hit: {hits} ({(100*hits/max(1,cnt)):.1f}%), Stop: {stops} ({(100*stops/max(1,cnt)):.1f}%), None: {nones} ({(100*nones/max(1,cnt)):.1f}%).")
    except Exception:
        pass
    # Allgemein (DE)
    add('Überblick', 'Jeder Marker wird ab dem Eröffnungskurs der nächsten Bar über N Lookahead‑Bars ausgewertet.')
    add('Überblick', 'Long: pro Bar zuerst Stop (Low ≤ Entry·(1−Stop%)), dann Hit (High ≥ Entry·(1+Hit%)).')
    add('Überblick', 'Short: pro Bar zuerst Stop (High ≥ Entry·(1+Stop%)), dann Hit (Low ≤ Entry·(1−Hit%)).')
    add('Überblick', 'Konservative Reihenfolge: Stop wird vor Hit geprüft, falls beides in derselben Bar möglich wäre.')
    # Diagramme
    add('MFE vs. MAE', 'MFE = maximale günstige Bewegung (%), MAE = maximale ungünstige Bewegung (%) relativ zum Entry. Oben‑links = stark/risikoarm; Ursprung = schwach; unten‑rechts = riskant/Whipsaw.')
    add('Trefferquote', 'Anteil (%) der Marker pro Typ, die den Hit‑Schwellenwert vor dem Stop innerhalb des Lookahead erreicht haben (höher = besser).')
    add('Bars bis Ereignis', 'Histogramme pro Typ über die Anzahl Bars bis zum Ereignis (Hit/Stop). Linkslastig = schnelle Bestätigung; breite Verteilung = hohe Streuung.')
    add('Ereignisverteilung', 'Anteil/Anzahl von Hit/Stop/None über alle Marker; schneller Überblick über Erfolgs‑ vs. Misserfolgsrate und offene Fälle.')
    # Legende
    add('Legende', 'CBullDivg_Classic △ (grün), CBullDivg_Hidden ◆ (grün), CBullDivg_x2_Classic ✱ (grün), HBullDivg_Classic ■ (grün), HBearDivg_Classic ▽ (rot), HBullDivg_Hidden × (grün).')
    # Glossar (Begriffe)
    add('Glossar', 'Klassische Divergenz: Preis macht ein tieferes Tief (Long) bzw. höheres Hoch (Short), der Indikator bestätigt das Gegenteil (höheres Tief / tieferes Hoch). Trendwendesignal.')
    add('Glossar', 'Verborgene (Hidden) Divergenz: Preis macht ein höheres Tief (Long) bzw. tieferes Hoch (Short), der Indikator bestätigt das Gegenteil. Trendsignalkontinuität (Fortsetzung).')
    add('Glossar', 'CBull (Classic Bullish): bullische klassische Divergenz (Long‑Bias).')
    add('Glossar', 'HBull (Hidden Bullish): bullische verborgene Divergenz (Long‑Bias, Fortsetzung).')
    add('Glossar', 'HBear (Hidden Bearish): bärische verborgene Divergenz (Short‑Bias, Fortsetzung).')
    add('Glossar', 'x2: verstärkte/erweiterte Variante der klassischen bullischen Divergenz (strengere Kriterien / zusätzliche Bestätigung).')
    add('Glossar', 'Richtung/Entry: Long für bullische Typen, Short für bärische. Entry = Eröffnungskurs der ersten Bar nach Markerzeit.')
    return pd.DataFrame(rows)


def _tz_naive_df(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    try:
        from pandas.api.types import is_datetime64_any_dtype
    except Exception:
        is_datetime64_any_dtype = lambda s: False
    for c in df2.columns:
        try:
            if is_datetime64_any_dtype(df2[c]):
                # try tz_convert then tz_localize
                try:
                    df2[c] = pd.to_datetime(df2[c]).dt.tz_convert(None)
                except Exception:
                    try:
                        df2[c] = pd.to_datetime(df2[c]).dt.tz_localize(None)
                    except Exception:
                        pass
        except Exception:
            pass
    return df2


def _write_outputs(per_marker: pd.DataFrame, summary: pd.DataFrame, asset_tag: str | None = None):
    os.makedirs('results', exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    base = f"{asset_tag + '_' if asset_tag else ''}markers_validation_{ts}"
    csv_path = os.path.join('results', base + '_per_marker.csv')
    xlsx_path = os.path.join('results', base + '_summary.xlsx')
    try:
        per_marker.to_csv(csv_path, index=False)
        print(f"Per-marker results: {csv_path}")
    except Exception:
        print("Failed to write per-marker CSV")
    if not _HAS_XLSX:
        try:
            summary.to_csv(xlsx_path.replace('.xlsx','.csv'), index=False)
            print(f"Summary CSV written: {xlsx_path.replace('.xlsx','.csv')}")
        except Exception:
            pass
        return
    try:
        with pd.ExcelWriter(xlsx_path, engine='openpyxl') as w:
            # Info
            info = pd.DataFrame([{ 'Generated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'Rows_Validated': int(per_marker.shape[0]) }])
            info.to_excel(w, sheet_name='Info', index=False)
            pm = per_marker if not per_marker.empty else pd.DataFrame({'Note':['No markers evaluated']})
            pm = _tz_naive_df(pm)
            pm.to_excel(w, sheet_name='Per_Marker', index=False)
            (summary if not summary.empty else pd.DataFrame({'Note':['No summary (no markers / no types)']})).to_excel(w, sheet_name='Summary_By_Type', index=False)
            # Event distribution by Type sheet
            try:
                types = per_marker['Type'].fillna('Unknown').astype(str).unique()
                rows = []
                for t in types:
                    d = per_marker[per_marker['Type']==t]['Event'].fillna('none').str.lower()
                    h = int((d=='hit').sum()); s = int((d=='stop').sum()); n = int((d=='none').sum()); tot = h+s+n
                    rows.append({'Type': t, 'Hit': h, 'Stop': s, 'None': n, 'Total': tot, 'Hit_Rate_%': (100*h/max(1,tot))})
                df_ed = pd.DataFrame(rows).sort_values('Hit_Rate_%', ascending=False)
                df_ed.to_excel(w, sheet_name='Event_By_Type', index=False)
            except Exception:
                pass
            # Quartal/Monat Pivots als Tabellen (Heatmap-Grundlagen)
            try:
                pm = per_marker.copy()
                pm['Entry_Date'] = pd.to_datetime(pm['Entry_Date'])
                pm = pm.dropna(subset=['Entry_Date'])
                pm['HitInt'] = (pm['Event'].str.lower()=='hit').astype(int)
                # Verwende die fixen Gruppen
                pm['Group'] = pm['Group'].astype(str)
                # Quartal
                pm['Q'] = pm['Entry_Date'].dt.to_period('Q').astype(str)
                gq = pm.groupby(['Group','Q']).agg(Markers=('Group','count'), Hits=('HitInt','sum'), First=('Entry_Date','min'), Last=('Entry_Date','max')).reset_index().rename(columns={'Group':'Type'})
                if not gq.empty:
                    gq['Days'] = (pd.to_datetime(gq['Last']) - pd.to_datetime(gq['First'])).dt.days.replace(0, np.nan).fillna(90)
                    gq['Density_per_100d'] = 100.0 * gq['Markers'] / gq['Days']
                    gq['Hit_Rate_%'] = 100.0 * gq['Hits'] / gq['Markers'].replace(0, np.nan)
                    gq.pivot(index='Type', columns='Q', values='Density_per_100d').reindex(index=GROUPS).to_excel(w, sheet_name='Density_Quarter')
                    gq.pivot(index='Type', columns='Q', values='Hit_Rate_%').reindex(index=GROUPS).to_excel(w, sheet_name='HitRate_Quarter')
                # Jahr
                pm['Year'] = pm['Entry_Date'].dt.to_period('Y').astype(str)
                gy = pm.groupby(['Group','Year']).agg(Markers=('Group','count'), Hits=('HitInt','sum')).reset_index().rename(columns={'Group':'Type'})
                if not gy.empty:
                    gy['Hit_Rate_%'] = 100.0 * gy['Hits'] / gy['Markers'].replace(0, np.nan)
                    gy.pivot(index='Type', columns='Year', values='Markers').reindex(index=GROUPS).to_excel(w, sheet_name='Count_Year')
                    gy.pivot(index='Type', columns='Year', values='Hit_Rate_%').reindex(index=GROUPS).to_excel(w, sheet_name='HitRate_Year')
                # Monat
                pm['M'] = pm['Entry_Date'].dt.to_period('M').astype(str)
                gm = pm.groupby(['Group','M']).agg(Markers=('Group','count'), Hits=('HitInt','sum')).reset_index().rename(columns={'Group':'Type'})
                if not gm.empty:
                    gm['Hit_Rate_%'] = 100.0 * gm['Hits'] / gm['Markers'].replace(0, np.nan)
                    gm.pivot(index='Type', columns='M', values='Markers').reindex(index=GROUPS).to_excel(w, sheet_name='Count_Month')
                    gm.pivot(index='Type', columns='M', values='Hit_Rate_%').reindex(index=GROUPS).to_excel(w, sheet_name='HitRate_Month')
                    # Timeseries: Monatliche Trefferquote als Zeitreihe
                    ts = gm.pivot(index='M', columns='Type', values='Hit_Rate_%').reindex(columns=GROUPS).sort_index()
                    ts.to_excel(w, sheet_name='HitRate_Month_TS')
            except Exception:
                pass
            _explanation_df(per_marker, summary).to_excel(w, sheet_name='Erklaerung', index=False)
        print(f"Summary workbook: {xlsx_path}")
    except Exception as e:
        print(f"Failed to write XLSX: {e}")


def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    return base64.b64encode(buf.getvalue()).decode('ascii')


def _make_dashboard(per_marker: pd.DataFrame, summary: pd.DataFrame, out_html: str):
    import matplotlib.pyplot as plt
    sections: List[str] = []
    # Randbedingungen (aus Per_Marker ableiten)
    try:
        lk = per_marker['Lookahead_Bars'].dropna().astype(int)
        hp = per_marker['Hit_Threshold_%'].dropna().astype(float)
        sp = per_marker['Stop_Threshold_%'].dropna().astype(float)
        def _fmt_range(s):
            if s.empty:
                return 'n/a'
            a, b = s.min(), s.max()
            return f"{a}" if a == b else f"{a} – {b}"
        cond_html = (
            "<ul>"
            f"<li><b>Lookahead (Bars):</b> {_fmt_range(lk)}</li>"
            f"<li><b>Hit‑Schwelle (%):</b> {_fmt_range(hp)}</li>"
            f"<li><b>Stop‑Schwelle (%):</b> {_fmt_range(sp)}</li>"
            "</ul>"
        )
    except Exception:
        cond_html = ""
    # Abschnitt: MFE vs. MAE (pro Typ) — zuerst
    try:
        fig, ax = plt.subplots(figsize=(6,4))
        # shape mapping consistent with legend
        shape_map = {
            'CBullDivg_Classic': '^',
            'CBullDivg_Hidden': 'D',
            'CBullDivg_x2_Classic': '*',
            'HBullDivg_Classic': 's',
            'HBearDivg_Classic': 'v',
            'HBullDivg_Hidden': 'x',
        }
        # always iterate fixed groups
        for t in GROUPS:
            d = per_marker[per_marker['Group']==t]
            color = '#e74c3c' if ('HBear' in t or 'Bear' in t) else '#2ecc71'
            # choose a generic marker for group
            g_marker = {'CBullDivg':'^','CBullDivg_x2':'*','HBullDivg':'s','HBearDivg':'v'}.get(t, 'o')
            if not d.empty:
                ax.scatter(d['MAE_%'], d['MFE_%'], s=28, alpha=0.7, label=t, marker=g_marker, c=color, edgecolors='none')
            else:
                ax.scatter([], [], label=t, marker=g_marker, c=color)
        ax.axvline(0,color='#bdc3c7',lw=0.8); ax.axhline(0,color='#bdc3c7',lw=0.8)
        ax.set_xlabel('MAE % (ungünstig)'); ax.set_ylabel('MFE % (günstig)'); ax.set_title('MFE vs. MAE')
        ax.legend(fontsize=7, ncol=2)
        img3 = _fig_to_base64(fig); plt.close(fig)
        sections.append(f"<h4>MFE vs. MAE</h4><img src='data:image/png;base64,{img3}'/>")
    except Exception:
        pass
    # Abschnitt: Trefferquote nach Typ
    try:
        s = summary.set_index('Type') if not summary.empty else pd.DataFrame(index=GROUPS, columns=['Hit_Rate_%']).fillna(0.0)
        fig, ax = plt.subplots(figsize=(6, 3))
        vals = [float(s['Hit_Rate_%'].get(g, 0.0)) for g in GROUPS]
        colors = ['#2ecc71','#2ecc71','#2ecc71','#e74c3c']
        ax.bar(GROUPS, vals, color=colors)
        ax.set_title('Trefferquote nach Typ (%)'); ax.set_ylabel('%'); ax.set_xticklabels(s['Type'], rotation=30, ha='right')
        ax.set_xticklabels(GROUPS, rotation=30, ha='right')
        img1 = _fig_to_base64(fig); plt.close(fig)
        sections.append(f"<h4>Trefferquote nach Typ</h4><img src='data:image/png;base64,{img1}'/>")

        # Monats‑Trefferquote (Linienplot) je Typ (immer 4 Gruppen)
        try:
            pm_ts = per_marker.copy()
            pm_ts['Entry_Date'] = pd.to_datetime(pm_ts['Entry_Date'])
            pm_ts = pm_ts.dropna(subset=['Entry_Date'])
            pm_ts['M'] = pm_ts['Entry_Date'].dt.to_period('M').astype(str)
            pm_ts['HitInt'] = (pm_ts['Event'].str.lower()=='hit').astype(int)
            gm = pm_ts.groupby(['M','Group']).agg(Markers=('Group','count'), Hits=('HitInt','sum')).reset_index()
            gm['Hit_Rate_%'] = 100.0 * gm['Hits'] / gm['Markers'].replace(0, np.nan)
            # Pivot to Month × Group
            pv = gm.pivot(index='M', columns='Group', values='Hit_Rate_%')
            if pv is not None and not pv.empty:
                pv = pv.reindex(columns=GROUPS)
                # Plot lines
                import matplotlib.pyplot as plt
                fig2, ax2 = plt.subplots(figsize=(7, 3.2))
                for g in GROUPS:
                    if g in pv.columns:
                        ax2.plot(pd.to_datetime(pv.index, errors='coerce'), pv[g], lw=1.2, label=g)
                    else:
                        ax2.plot([], [], label=g)
                ax2.set_title('Monatliche Trefferquote (%) je Typ')
                ax2.set_ylabel('%'); ax2.legend(fontsize=7, ncol=2)
                ax2.tick_params(axis='x', labelrotation=30)
                img_mt = _fig_to_base64(fig2); plt.close(fig2)
                sections.append(f"<h4>Monatliche Trefferquote (%) je Typ</h4><img src='data:image/png;base64,{img_mt}'/>")
        except Exception:
            pass
    except Exception:
        pass
    # Abschnitt: Bars bis Ereignis (je Typ, kleine Multiples)
    try:
        types = GROUPS
        fig, axes = plt.subplots(len(types), 1, figsize=(6, 1.8*len(types)), sharex=True)
        if len(types) == 1:
            axes = [axes]
        for ax, t in zip(axes, types):
            x = per_marker.loc[per_marker['Group']==t, 'Event_Bars'].dropna().astype(int)
            ax.hist(x, bins=20, color='#7f8c8d'); ax.set_title(str(t)); ax.set_ylabel('Anzahl')
        axes[-1].set_xlabel('Bars bis Ereignis')
        img2 = _fig_to_base64(fig); plt.close(fig)
        sections.append(f"<h4>Bars bis Ereignis (je Typ)</h4><img src='data:image/png;base64,{img2}'/>")
    except Exception:
        pass
    # Abschnitt: Ereignisverteilung (Hit/Stop/None)
    try:
        fig, ax = plt.subplots(figsize=(6,3))
        counts = per_marker['Event'].fillna('none').str.lower().value_counts()
        labels = ['hit','stop','none']
        vals = [counts.get('hit',0), counts.get('stop',0), counts.get('none',0)]
        colors = ['#2ecc71','#e74c3c','#7f8c8d']
        ax.bar(labels, vals, color=colors)
        ax.set_title('Ereignisverteilung (Hit/Stop/None)'); ax.set_ylabel('Anzahl'); ax.set_xlabel('Ereignis')
        img4 = _fig_to_base64(fig); plt.close(fig)
        sections.append(f"<h4>Ereignisverteilung</h4><img src='data:image/png;base64,{img4}'/>")
    except Exception:
        pass
    # Abschnitt: Ereignisverteilung pro Typ (gestapelt)
    try:
        types = GROUPS
        if types:
            fig, ax = plt.subplots(figsize=(max(6, 0.6*len(types)+2), 3.2))
            type_order = types
            def counts_for(t):
                d = per_marker[per_marker['Group']==t]['Event'].fillna('none').str.lower()
                return d.value_counts()
            hit = [counts_for(t).get('hit',0) for t in type_order]
            stop = [counts_for(t).get('stop',0) for t in type_order]
            none = [counts_for(t).get('none',0) for t in type_order]
            width = 0.6
            p1 = ax.bar(type_order, hit, width, color='#2ecc71', label='Hit')
            p2 = ax.bar(type_order, stop, width, bottom=hit, color='#e74c3c', label='Stop')
            p3 = ax.bar(type_order, none, width, bottom=[h+s for h,s in zip(hit,stop)], color='#7f8c8d', label='None')
            ax.set_title('Ereignisverteilung pro Typ (gestapelt, Anzahl)'); ax.legend()
            ax.tick_params(axis='x', rotation=30, labelsize=9)
            img5 = _fig_to_base64(fig); plt.close(fig)
            sections.append(f"<h4>Ereignisverteilung pro Typ (gestapelt, Anzahl)</h4><img src='data:image/png;base64,{img5}'/>")
    except Exception:
        pass
    # Abschnitt: Ereignisverteilung pro Typ (gestapelt, %)
    try:
        types = list(per_marker['Type'].dropna().astype(str).unique())
        if types:
            fig, ax = plt.subplots(figsize=(max(6, 0.6*len(types)+2), 3.2))
            type_order = types
            def perc_for(t):
                d = per_marker[per_marker['Type']==t]['Event'].fillna('none').str.lower()
                total = max(1, len(d))
                return (
                    100.0*d.value_counts().get('hit',0)/total,
                    100.0*d.value_counts().get('stop',0)/total,
                    100.0*d.value_counts().get('none',0)/total,
                )
            hitp, stopp, nonep = [], [], []
            for t in type_order:
                h,s,n = perc_for(t)
                hitp.append(h); stopp.append(s); nonep.append(n)
            width = 0.6
            ax.bar(type_order, hitp, width, color='#2ecc71', label='Hit')
            ax.bar(type_order, stopp, width, bottom=hitp, color='#e74c3c', label='Stop')
            ax.bar(type_order, nonep, width, bottom=[h+s for h,s in zip(hitp, stopp)], color='#7f8c8d', label='None')
            ax.set_title('Ereignisverteilung pro Typ (gestapelt, %)'); ax.set_ylabel('%'); ax.legend()
            ax.tick_params(axis='x', rotation=30, labelsize=9)
            img6 = _fig_to_base64(fig); plt.close(fig)
            sections.append(f"<h4>Ereignisverteilung pro Typ (gestapelt, %)</h4><img src='data:image/png;base64,{img6}'/>")
    except Exception:
        pass

    # Heatmaps mit fixierter Quartals-/Monatsachse (leere Zellen transparent) und Annotationen
    try:
        pm2 = per_marker.copy()
        pm2['Entry_Date'] = pd.to_datetime(pm2['Entry_Date'])
        pm2 = pm2.dropna(subset=['Entry_Date'])
        if not pm2.empty:
            pm2['Q'] = pm2['Entry_Date'].dt.to_period('Q').astype(str)
            pm2['M'] = pm2['Entry_Date'].dt.to_period('M').astype(str)
            pm2['HitInt'] = (pm2['Event'].str.lower()=='hit').astype(int)
            gq2 = pm2.groupby(['Type','Q']).agg(Markers=('Type','count'), Hits=('HitInt','sum'), First=('Entry_Date','min'), Last=('Entry_Date','max')).reset_index()
            if not gq2.empty:
                # Baue vollständige Quartalsachse zwischen min und max
                q_min = pd.Period(min(gq2['Q']), freq='Q')
                q_max = pd.Period(max(gq2['Q']), freq='Q')
                all_q = [str(pd.Period(q_min) + i) for i in range((q_max - q_min).n + 1)]
                # Dichte/Hit-Rate berechnen
                gq2['Days'] = (pd.to_datetime(gq2['Last']) - pd.to_datetime(gq2['First'])).dt.days.replace(0, np.nan).fillna(90)
                gq2['Density_per_100d'] = 100.0 * gq2['Markers'] / gq2['Days']
                gq2['Hit_Rate_%'] = 100.0 * gq2['Hits'] / gq2['Markers'].replace(0, np.nan)
                p_den = gq2.pivot(index='Type', columns='Q', values='Density_per_100d').reindex(index=GROUPS, columns=all_q)
                p_hr = gq2.pivot(index='Type', columns='Q', values='Hit_Rate_%').reindex(index=GROUPS, columns=all_q)
                # Typ-Reihenfolge nach Median-Trefferquote, sonst nach Median-Dichte
                type_order = None
                if p_hr is not None and p_hr.notna().any().any():
                    type_order = list(p_hr.median(axis=1).sort_values(ascending=False).index)
                elif p_den is not None and p_den.notna().any().any():
                    type_order = list(p_den.median(axis=1).sort_values(ascending=False).index)
                if type_order:
                    p_den = p_den.reindex(index=type_order)
                    p_hr = p_hr.reindex(index=type_order)
                import matplotlib
                import matplotlib.pyplot as plt
                # Helper to draw annotated heatmap
                def draw_heatmap(pivot: pd.DataFrame, title: str, value_format: str = None):
                    if pivot is None or not (pivot.notna().any().any()):
                        return None
                    fig, ax = plt.subplots(figsize=(max(6, 0.5*pivot.shape[1]+2), max(3, 0.3*pivot.shape[0]+2)))
                    cmap = matplotlib.cm.get_cmap('viridis').copy()
                    cmap.set_bad(alpha=0.0)
                    data = np.ma.masked_invalid(pivot.values.astype(float))
                    im = ax.imshow(data, aspect='auto', cmap=cmap)
                    ax.set_title(title)
                    ax.set_yticks(range(len(pivot.index))); ax.set_yticklabels(pivot.index)
                    ax.set_xticks(range(len(pivot.columns))); ax.set_xticklabels(pivot.columns, rotation=45, ha='right')
                    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    # Annotate values
                    try:
                        for i in range(pivot.shape[0]):
                            for j in range(pivot.shape[1]):
                                val = pivot.iat[i,j]
                                if pd.notna(val):
                                    txt = (value_format.format(val) if value_format else f"{val:.1f}")
                                    ax.text(j, i, txt, ha='center', va='center', fontsize=7, color='white')
                    except Exception:
                        pass
                    img_b64 = _fig_to_base64(fig); plt.close(fig)
                    return img_b64

                img_h1 = draw_heatmap(p_den, 'Heatmaps – Dichte (Marker/100 Tage) je Quartal', value_format="{:.1f}")
                if img_h1:
                    sections.append(f"<h4>Heatmaps</h4><h5>Dichte (Marker/100 Tage) je Quartal</h5><img src='data:image/png;base64,{img_h1}'/>")
                img_h2 = draw_heatmap(p_hr, 'Heatmaps – Trefferquote (%) je Quartal', value_format="{:.0f}%")
                if img_h2:
                    sections.append(f"<h5>Trefferquote (%) je Quartal</h5><img src='data:image/png;base64,{img_h2}'/>")

            # Monats‑Heatmaps (fixierte Monatsachse)
            gm2 = pm2.groupby(['Type','M']).agg(Markers=('Type','count'), Hits=('HitInt','sum')).reset_index()
            if not gm2.empty:
                m_min = pd.Period(min(gm2['M']), freq='M')
                m_max = pd.Period(max(gm2['M']), freq='M')
                all_m = [str(pd.Period(m_min) + i) for i in range((m_max - m_min).n + 1)]
                gm2['Hit_Rate_%'] = 100.0 * gm2['Hits'] / gm2['Markers'].replace(0, np.nan)
                pm_count = gm2.pivot(index='Type', columns='M', values='Markers').reindex(index=GROUPS, columns=all_m)
                pm_hr = gm2.pivot(index='Type', columns='M', values='Hit_Rate_%').reindex(index=GROUPS, columns=all_m)
                # sort types as above
                type_order2 = None
                if pm_hr is not None and pm_hr.notna().any().any():
                    type_order2 = list(pm_hr.median(axis=1).sort_values(ascending=False).index)
                elif pm_count is not None and pm_count.notna().any().any():
                    type_order2 = list(pm_count.median(axis=1).sort_values(ascending=False).index)
                if type_order2:
                    pm_count = pm_count.reindex(index=type_order2)
                    pm_hr = pm_hr.reindex(index=type_order2)
                import matplotlib
                import matplotlib.pyplot as plt
                img_m1 = draw_heatmap(pm_count, 'Heatmaps – Anzahl Marker je Monat', value_format="{:.0f}")
                if img_m1:
                    sections.append(f"<h5>Anzahl Marker je Monat</h5><img src='data:image/png;base64,{img_m1}'/>")
                img_m2 = draw_heatmap(pm_hr, 'Heatmaps – Trefferquote (%) je Monat', value_format="{:.0f}%")
                if img_m2:
                    sections.append(f"<h5>Trefferquote (%) je Monat</h5><img src='data:image/png;base64,{img_m2}'/>")
    except Exception:
        pass

    html = "<html><head><style>body{font-family:sans-serif} img{max-width:100%} .note{font-size:12px;color:#555} ul{margin-top:4px}</style></head><body>"
    html += "<h3>Validierung der Marker — Dashboard</h3>"
    # Legend block (DE)
    html += "<div class='note'><b>Legende:</b> CBullDivg_Classic △ (grün), CBullDivg_Hidden ◆ (grün), CBullDivg_x2_Classic ✱ (grün), HBullDivg_Classic ■ (grün), HBearDivg_Classic ▽ (rot), HBullDivg_Hidden × (grün).</div>"
    # Randbedingungen
    if cond_html:
        html += "<h4>Randbedingungen</h4>" + cond_html
    # Explanations bullets (DE)
    html += "<h4>So lesen Sie die Diagramme</h4><ul>"
    html += "<li><b>MFE vs. MAE:</b> MFE = maximale günstige Bewegung (%), MAE = maximale ungünstige Bewegung (%). Punkte oben‑links (hohe MFE, geringe |MAE|) deuten auf starke/risikoarme Signale hin; Nähe zum Ursprung = schwach; unten‑rechts = riskant/Whipsaw.</li>"
    html += "<li><b>Trefferquote nach Typ:</b> Anteil der Marker pro Typ, die innerhalb des Lookahead den Hit‑Schwellenwert vor dem Stop erreicht haben (höher = besser).</li>"
    html += "<li><b>Bars bis Ereignis:</b> Verteilung, wie schnell das Ereignis (Hit/Stop) eintritt; kleinere Werte = schnellere Bestätigung.</li>"
    html += "<li><b>Ereignisverteilung (gesamt & je Typ):</b> Verteilung der Ereignisse insgesamt sowie gestapelt je Typ — sowohl als Anzahl als auch in %.</li>"
    html += "<li><b>Auswertungslogik:</b> Long: pro Bar zuerst Stop, dann Hit; Short: zuerst Stop, dann Hit (konservative Reihenfolge).</li>"
    html += "</ul>"
    # Glossar
    html += "<h4>Glossar</h4><ul>"
    html += "<li><b>Klassische Divergenz:</b> Preis macht tieferes Tief (Long) bzw. höheres Hoch (Short), der Indikator bestätigt das Gegenteil → Trendwende.</li>"
    html += "<li><b>Verborgene (Hidden) Divergenz:</b> Preis macht höheres Tief (Long) bzw. tieferes Hoch (Short), Indikator das Gegenteil → Trendfortsetzung.</li>"
    html += "<li><b>CBull / HBull / HBear:</b> bullische/bärische (klassisch/hidden) Divergenz.</li>"
    html += "<li><b>x2:</b> verstärkte Variante der klassischen bullischen Divergenz (strengere Kriterien).</li>"
    html += "<li><b>Richtung/Entry:</b> Long für bullische, Short für bärische Typen; Entry = Open der ersten Bar nach Marker.</li>"
    html += "</ul>"
    html += "".join(sections)
    html += "</body></html>"
    os.makedirs(os.path.dirname(out_html) or '.', exist_ok=True)
    with open(out_html, 'w', encoding='utf-8') as f:
        f.write(html)


def _make_per_marker_pngs(df: pd.DataFrame, per_marker: pd.DataFrame, out_dir: str, max_images: int = 100, window: int = 40):
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)
    count = 0
    for _, r in per_marker.iterrows():
        if count >= max_images:
            break
        try:
            entry_dt = pd.to_datetime(r['Entry_Date'])
            # find local window
            idx = int(np.searchsorted(pd.to_datetime(df['date']).values, entry_dt.to_datetime64()))
            lo = max(0, idx - window)
            hi = min(len(df)-1, idx + window)
            sub = df.iloc[lo:hi+1].copy()
            fig, ax = plt.subplots(figsize=(6,2))
            ax.plot(pd.to_datetime(sub['date']), sub['close'], color='#2c3e50', lw=1.1)
            ax.axvline(entry_dt, color='#2980b9', lw=1, linestyle='--', label='Entry')
            # plot hit/stop levels
            if pd.notna(r.get('Hit_Level')):
                ax.axhline(float(r['Hit_Level']), color='#27ae60', lw=0.8, linestyle=':')
            if pd.notna(r.get('Stop_Level')):
                ax.axhline(float(r['Stop_Level']), color='#c0392b', lw=0.8, linestyle=':')
            # annotate event
            ev = str(r.get('Event'))
            ax.set_title(f"{r.get('Type')} | {ev} in {r.get('Event_Bars')} bars")
            ax.tick_params(axis='x', labelrotation=30)
            fig.tight_layout()
            fname = os.path.join(out_dir, f"marker_{count:04d}.png")
            fig.savefig(fname, dpi=120)
            plt.close(fig)
            count += 1
        except Exception:
            continue


def main():
    print("Marker Validation — select OHLC file and one/more marker CSVs.")
    # Load last selection if present
    last_cfg_path = os.path.join('results', 'markers_validator_last.json')
    last_ohlc = ''
    last_mks: List[str] = []
    try:
        import json
        if os.path.isfile(last_cfg_path):
            with open(last_cfg_path, 'r', encoding='utf-8') as f:
                cfg = json.load(f)
                last_ohlc = cfg.get('last_ohlc','')
                last_mks = cfg.get('last_markers', []) or []
    except Exception:
        pass

    # OHLC file
    # Prefer last path; else common raw data folder if exists
    default_raw = os.path.join(os.getcwd(), 'data', 'raw')
    init_dir = os.path.dirname(last_ohlc) if last_ohlc else (default_raw if os.path.isdir(default_raw) else None)
    ohlc = _ps_open_dialog(False, 'Select OHLC CSV/Parquet', 'CSV/Parquet (*.csv;*.parquet)|*.csv;*.parquet|CSV (*.csv)|*.csv|Parquet (*.parquet)|*.parquet', initial_dir=init_dir)
    if not ohlc:
        try:
            ohlc = _console_select_files(['*.csv','*.parquet','data/*.csv','data/*.parquet','../**/*.csv','../**/*.parquet'], allow_custom=True)
            ohlc = ohlc[0] if ohlc else ''
        except Exception:
            ohlc = ''
    if not ohlc:
        print('No OHLC selected. Exiting.')
        return
    # If user accidentally picked a markers CSV as OHLC, auto-move it to markers list
    mks_seed: List[str] = []
    try:
        if ohlc.lower().endswith('.csv'):
            probe = pd.read_csv(ohlc, nrows=5)
        else:
            probe = pd.read_parquet(ohlc)
        # heuristics: markers have 'Type' and 'Date', OHLC have open/high/low/close
        cols = set(probe.columns)
        if {'Type','Date'}.issubset(cols) and not ({'open','high','low','close'} & set([c.lower() for c in cols])):
            mks_seed.append(ohlc)
            print('Detected a marker CSV selected as OHLC; using it as marker and asking for OHLC again...')
            ohlc = _ps_open_dialog(False, 'Select OHLC CSV/Parquet (actual OHLC)', 'CSV/Parquet (*.csv;*.parquet)|*.csv;*.parquet|CSV (*.csv)|*.csv|Parquet (*.parquet)|*.parquet', initial_dir=init_dir)
            if not ohlc:
                print('No OHLC selected. Exiting.')
                return
    except Exception:
        pass

    # Marker files
    init_dir_mk = os.path.dirname(last_mks[0]) if last_mks else None
    mks = _ps_open_dialog(True, 'Select marker CSV files', 'CSV (*.csv)|*.csv|All files (*.*)|*.*', initial_dir=init_dir_mk)
    if not mks:
        try:
            mks = _console_select_files(['results/*.csv','*.csv'], allow_custom=True)
        except Exception:
            mks = []
    if mks_seed:
        # Prepend detected mistaken OHLC selection as markers
        mks = list(dict.fromkeys(mks_seed + (mks or [])))
    if not mks:
        print('No marker files selected. Exiting.')
        return
    lookahead, hit_pct, stop_pct = _prompt_config()
    try:
        per_marker, summary = validate_markers(ohlc, mks, lookahead, hit_pct, stop_pct)
        if per_marker is None or per_marker.empty:
            print('No markers validated (empty result).')
            return
        # derive asset tag from OHLC filename
        base = os.path.splitext(os.path.basename(ohlc))[0]
        parts = base.split('_')
        tag = f"{parts[0]}_{parts[1]}" if len(parts)>=2 else parts[0]
        _write_outputs(per_marker, summary, asset_tag=tag)
        # remember last selection
        try:
            import json
            os.makedirs('results', exist_ok=True)
            with open(last_cfg_path, 'w', encoding='utf-8') as f:
                json.dump({'last_ohlc': ohlc, 'last_markers': mks}, f)
        except Exception:
            pass
        # Dashboard + per-marker images
        try:
            dash_path = os.path.join('results', f"{tag}_markers_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}_dashboard.html")
            _make_dashboard(per_marker, summary, dash_path)
            print(f"Dashboard: {dash_path}")
        except Exception as e:
            print(f"Dashboard generation failed: {e}")
        # PNG snapshots disabled by default (can be enabled later)
    except Exception as e:
        print(f"Validation failed: {e}")


if __name__ == '__main__':
    main()
