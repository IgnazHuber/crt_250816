import os
import glob
from itertools import product
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd


def _parse_band_list(env_name: str, default: str) -> List[Tuple[float, float]]:
    """
    Parse semicolon-separated list of bands "min-max" from env var.
    Example: "40-70;45-75" -> [(40.0,70.0),(45.0,75.0)]
    """
    raw = os.getenv(env_name, default)
    out: List[Tuple[float, float]] = []
    for part in str(raw).split(';'):
        part = part.strip()
        if not part:
            continue
        try:
            lo, hi = part.split('-')
            out.append((float(lo), float(hi)))
        except Exception:
            continue
    return out


def _parse_cb_sets(env_name: str, default: str) -> List[Tuple[float, float, float]]:
    """
    Parse semicolon-separated CB RSI sets "hl_min/ll_min/max" from env var.
    Example: "30/15/55;35/20/60" -> [(30.0,15.0,55.0),(35.0,20.0,60.0)]
    """
    raw = os.getenv(env_name, default)
    out: List[Tuple[float, float, float]] = []
    for part in str(raw).split(';'):
        part = part.strip()
        if not part:
            continue
        try:
            a, b, c = part.split('/')
            out.append((float(a), float(b), float(c)))
        except Exception:
            continue
    return out


def _load_parquet_snapshot(folder_path: str) -> List[Dict[str, Any]]:
    """
    Load last 2 rows from each parquet (for signal state), plus LM_Low and dates.
    Mirrors the structure used in the backtest scripts but keeps it minimal for DOE counts.
    """
    req_cols = [
        'date', 'open', 'close', 'EMA_50', 'EMA_200', 'HBullD_gen', 'HBullD_Lower_Low_RSI_gen',
        'HBullD_Higher_Low_RSI_gen', 'HBullD_Higher_Low_gen', 'HBullD_neg_MACD',
        'HBullD_Lower_Low_RSI_neg_MACD', 'HBullD_Higher_Low_RSI_neg_MACD', 'HBullD_Higher_Low_neg_MACD',
        'CBullD_gen', 'CBullD_neg_MACD', 'CBullD_Higher_Low_RSI_gen', 'CBullD_Lower_Low_RSI_gen',
        'CBullD_Lower_Low_gen', 'CBullD_Higher_Low_RSI_neg_MACD', 'CBullD_Lower_Low_RSI_neg_MACD',
        'CBullD_Lower_Low_neg_MACD', 'CBullD_x2', 'CBullD_x2_Lower_Low', 'LM_Low_window_1_CS',
        'HBullD_Lower_Low_gen', 'HBullD_Lower_Low_neg_MACD', 'CBullD_Higher_Low_gen',
        'CBullD_Higher_Low_neg_MACD', 'CBullD_Date_Gap_gen', 'CBullD_Date_Gap_neg_MACD',
        'HBullD_Date_Gap_gen', 'HBullD_Date_Gap_neg_MACD'
    ]
    files = sorted(glob.glob(os.path.join(folder_path, '*.parquet')))
    out: List[Dict[str, Any]] = []
    for fp in files:
        try:
            df = pd.read_parquet(fp, columns=req_cols).tail(100)
            if len(df) < 2:
                continue
            last_row = df.iloc[-1]
            prev_row = df.iloc[-2]
            non_zero_df = df[df['LM_Low_window_1_CS'] != 0][['LM_Low_window_1_CS', 'date']].dropna()
            lm_low, lm_date = (non_zero_df['LM_Low_window_1_CS'].iloc[-1], non_zero_df['date'].iloc[-1]) if not non_zero_df.empty else (0, 0)
            out.append({
                'date': last_row['date'],
                'open': last_row['open'],
                'close': last_row['close'],
                'prev_close': prev_row['close'],
                'ema50_prev': prev_row['EMA_50'],
                'ema200_prev': prev_row['EMA_200'],
                'hb_gen': prev_row['HBullD_gen'],
                'hb_ll_rsi_gen': prev_row['HBullD_Lower_Low_RSI_gen'],
                'hb_hl_rsi_gen': prev_row['HBullD_Higher_Low_RSI_gen'],
                'hb_hl_gen': prev_row['HBullD_Higher_Low_gen'],
                'hb_neg_macd': prev_row['HBullD_neg_MACD'],
                'hb_ll_rsi_neg': prev_row['HBullD_Lower_Low_RSI_neg_MACD'],
                'hb_hl_rsi_neg': prev_row['HBullD_Higher_Low_RSI_neg_MACD'],
                'hb_hl_neg': prev_row['HBullD_Higher_Low_neg_MACD'],
                'cb_gen': prev_row['CBullD_gen'],
                'cb_neg_macd': prev_row['CBullD_neg_MACD'],
                'cb_hl_rsi_gen': prev_row['CBullD_Higher_Low_RSI_gen'],
                'cb_ll_rsi_gen': prev_row['CBullD_Lower_Low_RSI_gen'],
                'cb_ll_gen': prev_row['CBullD_Lower_Low_gen'],
                'cb_hl_rsi_neg': prev_row['CBullD_Higher_Low_RSI_neg_MACD'],
                'cb_ll_rsi_neg': prev_row['CBullD_Lower_Low_RSI_neg_MACD'],
                'cb_ll_neg': prev_row['CBullD_Lower_Low_neg_MACD'],
                'cb_x2': prev_row['CBullD_x2'],
                'cb_x2_ll': prev_row['CBullD_x2_Lower_Low'],
                'hb_ll_gen': prev_row['HBullD_Lower_Low_gen'],
                'hb_ll_neg': prev_row['HBullD_Lower_Low_neg_MACD'],
                'cb_hl_gen': prev_row['CBullD_Higher_Low_gen'],
                'cb_hl_neg': prev_row['CBullD_Higher_Low_neg_MACD'],
                'cb_date_gap_gen': prev_row['CBullD_Date_Gap_gen'],
                'cb_date_gap_neg': prev_row['CBullD_Date_Gap_neg_MACD'],
                'hb_date_gap_gen': prev_row['HBullD_Date_Gap_gen'],
                'hb_date_gap_neg': prev_row['HBullD_Date_Gap_neg_MACD'],
            })
        except Exception:
            continue
    return out


def _signals_counts(data: List[Dict[str, Any]],
                    hb_band: Tuple[float, float],
                    cb_set: Tuple[float, float, float]) -> Dict[str, Any]:
    """
    Compute buy signal counts for each divergence type given Item-1/2 thresholds.
    Item 1 (HB band): (hb_min, hb_max)
    Item 2 (CB bands): (cb_hl_min, cb_ll_min, cb_max)
    Returns counts and simple breakdowns; no PnL simulation here (fast DOE stage).
    """
    if not data:
        return {
            'count_all': 0,
            'count_hb_gen': 0,
            'count_hb_neg': 0,
            'count_cb_gen': 0,
            'count_cb_neg': 0,
            'count_cb_x2': 0,
        }

    hb_min, hb_max = hb_band
    cb_hl_min, cb_ll_min, cb_max = cb_set

    arr = lambda k: np.array([d[k] for d in data], dtype=float)
    ema50_prev = arr('ema50_prev')
    ema200_prev = arr('ema200_prev')
    closes = arr('close')

    hb_gen = arr('hb_gen')
    hb_ll_rsi_gen = arr('hb_ll_rsi_gen')
    hb_hl_rsi_gen = arr('hb_hl_rsi_gen')
    hb_hl_gen = arr('hb_hl_gen')
    hb_neg_macd = arr('hb_neg_macd')
    hb_ll_rsi_neg = arr('hb_ll_rsi_neg')
    hb_hl_rsi_neg = arr('hb_hl_rsi_neg')
    hb_hl_neg = arr('hb_hl_neg')

    cb_gen = arr('cb_gen')
    cb_neg_macd = arr('cb_neg_macd')
    cb_hl_rsi_gen = arr('cb_hl_rsi_gen')
    cb_ll_rsi_gen = arr('cb_ll_rsi_gen')
    cb_ll_gen = arr('cb_ll_gen')
    cb_hl_rsi_neg = arr('cb_hl_rsi_neg')
    cb_ll_rsi_neg = arr('cb_ll_rsi_neg')
    cb_ll_neg = arr('cb_ll_neg')
    cb_x2 = arr('cb_x2')
    cb_x2_ll = arr('cb_x2_ll')

    cond1 = (hb_gen == 1) & (hb_ll_rsi_gen < hb_max) & (hb_ll_rsi_gen > hb_min) & (hb_hl_rsi_gen < hb_max) & (hb_hl_rsi_gen > hb_min) & (ema50_prev > ema200_prev) & (closes > hb_hl_gen)
    cond2 = (hb_neg_macd == 1) & (hb_ll_rsi_neg < hb_max) & (hb_ll_rsi_neg > hb_min) & (hb_hl_rsi_neg < hb_max) & (hb_hl_rsi_neg > hb_min) & (ema50_prev > ema200_prev) & (closes > hb_hl_neg)
    cond3 = (((cb_gen == 1) & (cb_neg_macd == 1)) | (cb_gen == 1)) & (cb_hl_rsi_gen < cb_max) & (cb_hl_rsi_gen > cb_hl_min) & (cb_ll_rsi_gen < cb_max) & (cb_ll_rsi_gen > cb_ll_min) & (closes > cb_ll_gen)
    cond4 = (cb_neg_macd == 1) & (cb_hl_rsi_neg < cb_max) & (cb_hl_rsi_neg > cb_hl_min) & (cb_ll_rsi_neg < cb_max) & (cb_ll_rsi_neg > cb_ll_min) & (closes > cb_ll_neg)
    cond5 = (cb_gen == 1) & (cb_hl_rsi_gen < cb_max) & (cb_hl_rsi_gen > cb_hl_min) & (cb_ll_rsi_gen < cb_max) & (cb_ll_rsi_gen > cb_ll_min) & (closes > cb_ll_gen)
    cond6 = (cb_x2 == 1) & (closes > cb_x2_ll)

    counts = {
        'count_hb_gen': int(np.sum(cond1)),
        'count_hb_neg': int(np.sum(cond2)),
        'count_cb_gen': int(np.sum(cond3) + np.sum(cond5)),
        'count_cb_neg': int(np.sum(cond4)),
        'count_cb_x2': int(np.sum(cond6)),
    }
    counts['count_all'] = int(sum(counts.values()))
    return counts


def run_items12_full_factorial(folder_path: str,
                                hb_bands: List[Tuple[float, float]],
                                cb_sets: List[Tuple[float, float, float]]) -> pd.DataFrame:
    data = _load_parquet_snapshot(folder_path)
    rows: List[Dict[str, Any]] = []
    for (hb_min, hb_max), (cb_hl_min, cb_ll_min, cb_max) in product(hb_bands, cb_sets):
        c = _signals_counts(data, (hb_min, hb_max), (cb_hl_min, cb_ll_min, cb_max))
        c.update({
            'HB_Min': hb_min,
            'HB_Max': hb_max,
            'CB_HL_Min': cb_hl_min,
            'CB_LL_Min': cb_ll_min,
            'CB_Max': cb_max,
        })
        rows.append(c)
    df = pd.DataFrame(rows)
    return df.sort_values('count_all', ascending=False)


def main():
    folder = os.getenv('DOE_DATA_DIR', '').strip()
    if not folder:
        raise SystemExit('Set DOE_DATA_DIR to the folder with *.parquet files')

    # Defaults derived from current scripts (PDF items 1 & 2 assumed):
    # Item 1 (HB RSI band): 40–70
    # Item 2 (CB RSI bands): HL 30–55, LL 15–55
    hb_bands = _parse_band_list('DOE_HB_BANDS', '40-70;45-70;40-75')
    cb_sets = _parse_cb_sets('DOE_CB_SETS', '30/15/55;35/20/55;30/20/60')

    df = run_items12_full_factorial(folder, hb_bands, cb_sets)
    os.makedirs('results', exist_ok=True)
    out_csv = os.path.join('results', 'doe_items12_summary.csv')
    df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")

    # Quick pivot examples (top-level totals by HB vs CB choices)
    try:
        df['HB_Band'] = df.apply(lambda r: f"{r['HB_Min']}-{r['HB_Max']}", axis=1)
        df['CB_Set'] = df.apply(lambda r: f"{r['CB_HL_Min']}/{r['CB_LL_Min']}/{r['CB_Max']}", axis=1)
        pv = df.pivot_table(index='HB_Band', columns='CB_Set', values='count_all', aggfunc='sum', fill_value=0)
        out_pv = os.path.join('results', 'doe_items12_pivot.csv')
        pv.to_csv(out_pv)
        print(f"Saved: {out_pv}")
    except Exception:
        pass


if __name__ == '__main__':
    main()

