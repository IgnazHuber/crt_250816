import pandas as pd
import os
import numpy as np
from multiprocessing import Pool, cpu_count
import glob
import argparse
import sys
import json
import importlib
import argparse
import sys
try:
    import tkinter as _tk
    from tkinter import filedialog as _filedialog
except Exception:
    _tk = None
    _filedialog = None

# Set pandas display options
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

Risk_percentage = 1.00
Brokerage = 0.1  # in %
Brokerage_buy = 1 + (Brokerage / 100)
Brokerage_sell = 1 - (Brokerage / 100)

# Function to process a single data file (parquet or csv)
def process_parquet_file(file_path):
    return process_market_file(file_path)

def _read_csv_last_n_rows(path: str, usecols: list[str], n: int = 100):
    from collections import deque
    import io as _io
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            header = f.readline()
            tail_lines = deque(f, maxlen=max(1, n))
        buf = header + ''.join(tail_lines)
        return pd.read_csv(_io.StringIO(buf), usecols=usecols)
    except Exception:
        # Fallback: read whole file (may be slow on very large files)
        return pd.read_csv(path, usecols=usecols)

def process_market_file(file_path):
    try:
        required_columns = [
            'date', 'open', 'close', 'EMA_50', 'EMA_200', 'HBullD_gen', 'HBullD_Lower_Low_RSI_gen',
            'HBullD_Higher_Low_RSI_gen', 'HBullD_Higher_Low_gen', 'HBullD_neg_MACD',
            'HBullD_Lower_Low_RSI_neg_MACD', 'HBullD_Higher_Low_RSI_neg_MACD', 'HBullD_Higher_Low_neg_MACD',
            'CBullD_gen', 'CBullD_neg_MACD', 'CBullD_Higher_Low_RSI_gen', 'CBullD_Lower_Low_RSI_gen',
            'CBullD_Lower_Low_gen', 'CBullD_Higher_Low_RSI_neg_MACD', 'CBullD_Lower_Low_RSI_neg_MACD',
            'CBullD_Lower_Low_neg_MACD', 'CBullD_x2', 'CBullD_x2_Lower_Low', 'LM_Low_window_1_CS',
            'HBullD_Lower_Low_gen', 'HBullD_Lower_Low_neg_MACD', 'CBullD_Higher_Low_gen',
            'CBullD_Higher_Low_neg_MACD', 'CBullD_Date_Gap_gen', 'CBullD_Date_Gap_neg_MACD',
            'HBullD_Date_Gap_gen', 'HBullD_Date_Gap_neg_MACD', 'EMA_20'
        ]
        # Robust read: parquet or csv; if EMA_20 missing, read without and synthesize later
        ext = os.path.splitext(file_path)[1].lower()
        has_ema20 = True
        if ext == '.parquet':
            try:
                df = pd.read_parquet(file_path, columns=required_columns).tail(100)
                has_ema20 = True
            except Exception:
                fallback_cols = [c for c in required_columns if c != 'EMA_20']
                df = pd.read_parquet(file_path, columns=fallback_cols).tail(100)
                has_ema20 = False
        elif ext == '.csv':
            try:
                df = _read_csv_last_n_rows(file_path, usecols=required_columns, n=100)
                has_ema20 = True
            except Exception:
                fallback_cols = [c for c in required_columns if c != 'EMA_20']
                df = _read_csv_last_n_rows(file_path, usecols=fallback_cols, n=100)
                has_ema20 = False
        else:
            return None
        if len(df) < 2:
            return None
        last_row = df.iloc[-1]
        prev_row = df.iloc[-2]
        non_zero_df = df[df['LM_Low_window_1_CS'] != 0][['LM_Low_window_1_CS', 'date']].dropna()
        lm_low, lm_date = (non_zero_df['LM_Low_window_1_CS'].iloc[-1], non_zero_df['date'].iloc[-1]) if not non_zero_df.empty else (0, 0)
        out = {
            'date': last_row['date'],
            'open': last_row['open'],
            'close': last_row['close'],
            'prev_close': prev_row['close'],
            'ema50_prev': prev_row['EMA_50'],
            'ema200_prev': prev_row['EMA_200'],
            'ema20_prev': (prev_row['EMA_20'] if has_ema20 else prev_row['EMA_50']),
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
            'lm_low': lm_low,
            'lm_date': lm_date,
            'hb_ll_gen': prev_row['HBullD_Lower_Low_gen'],
            'hb_ll_neg': prev_row['HBullD_Lower_Low_neg_MACD'],
            'cb_hl_gen': prev_row['CBullD_Higher_Low_gen'],
            'cb_hl_neg': prev_row['CBullD_Higher_Low_neg_MACD'],
            'cb_date_gap_gen': prev_row['CBullD_Date_Gap_gen'],
            'cb_date_gap_neg': prev_row['CBullD_Date_Gap_neg_MACD'],
            'hb_date_gap_gen': prev_row['HBullD_Date_Gap_gen'],
            'hb_date_gap_neg': prev_row['HBullD_Date_Gap_neg_MACD']
        }
        return out
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Parallel processing of Parquet files
def parallel_load_files(file_paths, procs: int | None = None):
    # Allow external control to avoid oversubscription under DOE
    env_override = os.environ.get('DOE_LOAD_PROCS')
    try:
        env_override_val = int(env_override) if env_override is not None else None
    except Exception:
        env_override_val = None
    n_proc = procs if isinstance(procs, int) and procs > 0 else (env_override_val if isinstance(env_override_val, int) and env_override_val > 0 else cpu_count())
    with Pool(processes=max(1, n_proc)) as pool:
        results = pool.map(process_market_file, file_paths)
    return [r for r in results if r is not None]

# Stoploss trigger check
def Stoploss_Trigger_Check(close, stoploss):
    return 1 if close < stoploss else 0

if __name__ == '__main__':
    # Output CSV name
    output_combined_file = 'backtest_results_BTC_4hour_100perc_with_brokerage_orig.csv'

    # Allow selecting a parquet file (or a directory containing parquet files) via Explorer
    # and remember the last-used directory next to this script.
    last_dir_file = os.path.join(os.path.dirname(__file__), '.last_parquet_dir.txt')
    initial_dir = None
    if os.path.exists(last_dir_file):
        try:
            with open(last_dir_file, 'r', encoding='utf-8') as f:
                saved = f.read().strip()
                if saved:
                    initial_dir = saved
        except Exception:
            initial_dir = None

    # CLI overrides for headless usage
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--folder', '-f', dest='input_folder', default=None,
                        help='Folder containing parquet files')
    parser.add_argument('--file', '-F', dest='input_file', default=None,
                        help='Single parquet file to process')
    parser.add_argument('--doe-module', dest='doe_module', default=None,
                        help='Python module path that provides compute_signals(context, spec)')
    parser.add_argument('--doe-spec', dest='doe_spec', default=None,
                        help='JSON string or path to JSON file with DOE spec')
    parser.add_argument('--output', '-o', dest='output_csv', default=None,
                        help='Optional output CSV path')
    parser.add_argument('--load-procs', type=int, default=None,
                        help='Parallel processes for loading parquet files (overrides env DOE_LOAD_PROCS)')
    args, _ = parser.parse_known_args(sys.argv[1:])

    folder_path = None
    parquet_files = []

    # 1) Explicit file
    if args.input_file:
        input_file = os.path.expanduser(args.input_file)
        if not (os.path.isfile(input_file) and os.path.splitext(input_file)[1].lower() in ('.parquet', '.csv')):
            raise FileNotFoundError(f"--file not found or not supported (.parquet|.csv): {input_file}")
        folder_path = os.path.dirname(input_file)
        parquet_files = [input_file]

    # 2) Explicit folder
    elif args.input_folder:
        input_folder = os.path.expanduser(args.input_folder)
        if not os.path.isdir(input_folder):
            raise FileNotFoundError(f"--folder not found: {input_folder}")
        folder_path = input_folder
        parquet_files = sorted(glob.glob(os.path.join(folder_path, '*.parquet'))) + \
                         sorted(glob.glob(os.path.join(folder_path, '*.csv')))
        # If none at top-level, search recursively inside provided folder
        if not parquet_files:
            for root_dir, _subdirs, _files in os.walk(folder_path):
                files = sorted(glob.glob(os.path.join(root_dir, '*.parquet'))) + \
                        sorted(glob.glob(os.path.join(root_dir, '*.csv')))
                if files:
                    parquet_files = files
                    folder_path = root_dir
                    break

    # 3) Tkinter folder dialog
    else:
        if _filedialog is not None and _tk is not None:
            try:
                root = _tk.Tk()
                root.withdraw()
                folder_path = _filedialog.askdirectory(
                    title='Select a folder containing parquet files',
                    initialdir=initial_dir or os.getcwd(),
                )
                root.destroy()
            except Exception:
                folder_path = None

        # 4) Non-dialog fallback cascade (robust for headless environments)
        candidate_dirs = []
        if folder_path:
            candidate_dirs.append(folder_path)
        if initial_dir and initial_dir not in candidate_dirs:
            candidate_dirs.append(initial_dir)
        cwd_dir = os.getcwd()
        if cwd_dir not in candidate_dirs:
            candidate_dirs.append(cwd_dir)

        resolved_folder = None
        # Try direct candidates first
        for d in candidate_dirs:
            files = sorted(glob.glob(os.path.join(d, '*.parquet'))) + \
                    sorted(glob.glob(os.path.join(d, '*.csv')))
            if files:
                resolved_folder = d
                parquet_files = files
                break

        # If still nothing, search recursively under CWD for the first directory with parquet/csv
        if not parquet_files:
            for root_dir, _subdirs, _files in os.walk(cwd_dir):
                files = sorted(glob.glob(os.path.join(root_dir, '*.parquet'))) + \
                        sorted(glob.glob(os.path.join(root_dir, '*.csv')))
                if files:
                    resolved_folder = root_dir
                    parquet_files = files
                    break

        if not folder_path:
            folder_path = resolved_folder

    if not parquet_files:
        # Interactive console prompt as a last resort
        try:
            user_input = input('Enter a folder with .parquet files, or a single .parquet file path (leave blank to abort): ').strip()
        except Exception:
            user_input = ''
        if user_input:
            candidate = os.path.expanduser(user_input)
            if os.path.isfile(candidate) and os.path.splitext(candidate)[1].lower() in ('.parquet', '.csv'):
                folder_path = os.path.dirname(candidate)
                parquet_files = [candidate]
            elif os.path.isdir(candidate):
                folder_path = candidate
                parquet_files = sorted(glob.glob(os.path.join(folder_path, '*.parquet'))) + \
                                 sorted(glob.glob(os.path.join(folder_path, '*.csv')))
        if not parquet_files:
            raise ValueError('No parquet files found. Provide --folder <dir> or --file <path>, or ensure .parquet files are available.')

    # Informative log about input selection
    try:
        print(f"Using {len(parquet_files)} parquet file(s) from folder: {folder_path}")
    except Exception:
        pass

    # Persist last-used directory for next run
    try:
        last_dir_to_save = folder_path if folder_path else None
        if last_dir_to_save:
            with open(last_dir_file, 'w', encoding='utf-8') as f:
                f.write(last_dir_to_save)
    except Exception:
        pass

    num_files = len(parquet_files)

    # Load data in parallel
    data = parallel_load_files(parquet_files, procs=getattr(args, 'load_procs', None))
    if not data:
        raise ValueError("No valid data loaded from Parquet files")
    # Use actual loaded data length in case some files failed to parse
    num_files = len(data)

    # Preallocate arrays
    dates = np.array([d['date'] for d in data], dtype=object)
    opens = np.array([d['open'] for d in data], dtype=float)
    closes = np.array([d['close'] for d in data], dtype=float)
    prev_closes = np.array([d['prev_close'] for d in data], dtype=float)
    lm_lows = np.array([d['lm_low'] for d in data], dtype=float)
    lm_dates = np.array([d['lm_date'] for d in data], dtype=object)
    ema50_prev = np.array([d['ema50_prev'] for d in data], dtype=float)
    ema200_prev = np.array([d['ema200_prev'] for d in data], dtype=float)
    ema20_prev = np.array([d['ema20_prev'] for d in data], dtype=float)
    hb_gen = np.array([d['hb_gen'] for d in data], dtype=float)
    hb_ll_rsi_gen = np.array([d['hb_ll_rsi_gen'] for d in data], dtype=float)
    hb_hl_rsi_gen = np.array([d['hb_hl_rsi_gen'] for d in data], dtype=float)
    hb_hl_gen = np.array([d['hb_hl_gen'] for d in data], dtype=float)
    hb_neg_macd = np.array([d['hb_neg_macd'] for d in data], dtype=float)
    hb_ll_rsi_neg = np.array([d['hb_ll_rsi_neg'] for d in data], dtype=float)
    hb_hl_rsi_neg = np.array([d['hb_hl_rsi_neg'] for d in data], dtype=float)
    hb_hl_neg = np.array([d['hb_hl_neg'] for d in data], dtype=float)
    cb_gen = np.array([d['cb_gen'] for d in data], dtype=float)
    cb_neg_macd = np.array([d['cb_neg_macd'] for d in data], dtype=float)
    cb_hl_rsi_gen = np.array([d['cb_hl_rsi_gen'] for d in data], dtype=float)
    cb_ll_rsi_gen = np.array([d['cb_ll_rsi_gen'] for d in data], dtype=float)
    cb_ll_gen = np.array([d['cb_ll_gen'] for d in data], dtype=float)
    cb_hl_rsi_neg = np.array([d['cb_hl_rsi_neg'] for d in data], dtype=float)
    cb_ll_rsi_neg = np.array([d['cb_ll_rsi_neg'] for d in data], dtype=float)
    cb_ll_neg = np.array([d['cb_ll_neg'] for d in data], dtype=float)
    cb_x2 = np.array([d['cb_x2'] for d in data], dtype=float)
    cb_x2_ll = np.array([d['cb_x2_ll'] for d in data], dtype=float)
    hb_ll_gen = np.array([d['hb_ll_gen'] for d in data], dtype=float)
    hb_ll_neg = np.array([d['hb_ll_neg'] for d in data], dtype=float)
    cb_hl_gen = np.array([d['cb_hl_gen'] for d in data], dtype=float)
    cb_hl_neg = np.array([d['cb_hl_neg'] for d in data], dtype=float)
    cb_date_gap_gen = np.array([d['cb_date_gap_gen'] for d in data], dtype=float)
    cb_date_gap_neg = np.array([d['cb_date_gap_neg'] for d in data], dtype=float)
    hb_date_gap_gen = np.array([d['hb_date_gap_gen'] for d in data], dtype=float)
    hb_date_gap_neg = np.array([d['hb_date_gap_neg'] for d in data], dtype=float)

    # Vectorized buy signal and initial stoploss calculation (default)
    def compute_default_signals():
        cond1 = (hb_gen == 1) & (hb_ll_rsi_gen < 70) & (hb_ll_rsi_gen > 40) & (hb_hl_rsi_gen < 70) & (hb_hl_rsi_gen > 40) & (ema50_prev > ema200_prev) & (closes > hb_hl_gen)
        cond2 = (hb_neg_macd == 1) & (hb_ll_rsi_neg < 70) & (hb_ll_rsi_neg > 40) & (hb_hl_rsi_neg < 70) & (hb_hl_rsi_neg > 40) & (ema50_prev > ema200_prev) & (closes > hb_hl_neg)
        cond3 = (((cb_gen == 1) & (cb_neg_macd == 1)) | (cb_gen == 1)) & (cb_hl_rsi_gen < 55) & (cb_hl_rsi_gen > 30) & (cb_ll_rsi_gen < 55) & (cb_ll_rsi_gen > 15) & (closes > cb_ll_gen)
        cond4 = (cb_neg_macd == 1) & (cb_hl_rsi_neg < 55) & (cb_hl_rsi_neg > 30) & (cb_ll_rsi_neg < 55) & (cb_ll_rsi_neg > 15) & (closes > cb_ll_neg)
        cond5 = (cb_gen == 1) & (cb_hl_rsi_gen < 55) & (cb_hl_rsi_gen > 30) & (cb_ll_rsi_gen < 55) & (cb_ll_rsi_gen > 15) & (closes > cb_ll_gen)
        cond6 = (cb_x2 == 1) & (closes > cb_x2_ll)
        conds_local = [cond1, cond2, cond3, cond4, cond5, cond6]
        stop_choices_local = [hb_hl_gen, hb_hl_neg, cb_ll_gen, cb_ll_neg, cb_ll_gen, cb_x2_ll]
        bs = np.select(conds_local, [1] * len(conds_local), default=0)
        isl = np.select(conds_local, stop_choices_local, default=0)
        with np.errstate(divide='ignore', invalid='ignore'):
            denom1 = hb_ll_gen * hb_date_gap_gen
            denom2 = hb_ll_neg * hb_date_gap_neg
            denom3 = cb_hl_gen * cb_date_gap_gen
            denom4 = cb_hl_neg * cb_date_gap_neg
            d1 = 10000 * np.divide(np.abs(hb_ll_gen - hb_hl_gen), denom1, out=np.zeros_like(hb_ll_gen), where=(denom1 != 0))
            d2 = 10000 * np.divide(np.abs(hb_ll_neg - hb_hl_neg), denom2, out=np.zeros_like(hb_ll_neg), where=(denom2 != 0))
            d3 = 10000 * np.divide(np.abs(cb_ll_gen - cb_hl_gen), denom3, out=np.zeros_like(cb_hl_gen), where=(denom3 != 0))
            d4 = 10000 * np.divide(np.abs(cb_ll_neg - cb_hl_neg), denom4, out=np.zeros_like(cb_hl_neg), where=(denom4 != 0))
        d5 = d3
        d6 = np.zeros_like(d1)
        choices = [d1, d2, d3, d4, d5, d6]
        ds = np.select(conds_local, choices, default=0.0)
        ds = np.nan_to_num(ds, nan=0.0, posinf=0.0, neginf=0.0)
        return bs, isl, ds

    # DOE hook
    buy_signals_pre = None
    initial_stoploss_pre = None
    divg_slope_pre = None
    if getattr(args, 'doe_module', None):
        # Build context for DOE strategy
        context = {
            'dates': dates,
            'opens': opens,
            'closes': closes,
            'prev_closes': prev_closes,
            'ema20_prev': ema20_prev,
            'ema50_prev': ema50_prev,
            'ema200_prev': ema200_prev,
            'hb_gen': hb_gen,
            'hb_neg_macd': hb_neg_macd,
            'hb_ll_rsi_gen': hb_ll_rsi_gen,
            'hb_hl_rsi_gen': hb_hl_rsi_gen,
            'hb_ll_rsi_neg': hb_ll_rsi_neg,
            'hb_hl_rsi_neg': hb_hl_rsi_neg,
            'hb_hl_gen': hb_hl_gen,
            'hb_hl_neg': hb_hl_neg,
            'hb_ll_gen': hb_ll_gen,
            'hb_ll_neg': hb_ll_neg,
            'cb_gen': cb_gen,
            'cb_neg_macd': cb_neg_macd,
            'cb_ll_rsi_gen': cb_ll_rsi_gen,
            'cb_hl_rsi_gen': cb_hl_rsi_gen,
            'cb_ll_rsi_neg': cb_ll_rsi_neg,
            'cb_hl_rsi_neg': cb_hl_rsi_neg,
            'cb_ll_gen': cb_ll_gen,
            'cb_ll_neg': cb_ll_neg,
            'cb_hl_gen': cb_hl_gen,
            'cb_hl_neg': cb_hl_neg,
            'cb_x2': cb_x2,
            'cb_x2_ll': cb_x2_ll,
            'cb_date_gap_gen': cb_date_gap_gen,
            'cb_date_gap_neg': cb_date_gap_neg,
            'hb_date_gap_gen': hb_date_gap_gen,
            'hb_date_gap_neg': hb_date_gap_neg,
        }
        spec_raw = getattr(args, 'doe_spec', None)
        spec = None
        if spec_raw:
            try:
                if os.path.isfile(spec_raw):
                    with open(spec_raw, 'r', encoding='utf-8') as f:
                        spec = json.load(f)
                else:
                    spec = json.loads(spec_raw)
            except Exception:
                spec = None
        try:
            mod = importlib.import_module(args.doe_module)
            buy_signals_pre, initial_stoploss_pre, divg_slope_pre = mod.compute_signals(context, spec)
        except Exception as e:
            print(f"DOE module failed ({args.doe_module}): {e}. Falling back to default signals.")
            buy_signals_pre, initial_stoploss_pre, divg_slope_pre = compute_default_signals()
    else:
        buy_signals_pre, initial_stoploss_pre, divg_slope_pre = compute_default_signals()

    # Initialize arrays
    Buy_Signal = np.zeros(num_files, dtype=int)
    Buy_Signal_date = np.zeros(num_files, dtype=object)
    Actual_Buy = np.zeros(num_files, dtype=int)
    First_buy_date = np.zeros(num_files, dtype=object)
    Stoploss = np.zeros(num_files, dtype=float)
    Stoploss_Trigger = np.zeros(num_files, dtype=int)
    Stoploss_Trigger_date = np.zeros(num_files, dtype=object)
    Actual_Sell = np.zeros(num_files, dtype=int)
    Current_Capital_Value = np.zeros(num_files, dtype=float)
    Available_Capital_for_trade = np.zeros(num_files, dtype=float)
    Buy_Quantity = np.zeros(num_files, dtype=float)
    Total_Buy_Quantity = np.zeros(num_files, dtype=float)
    loss_per_unit = np.zeros(num_files, dtype=float)
    serial_date = dates
    LM_Low_window_1_CS_last = lm_lows
    LM_Low_window_1_CS_last_date = lm_dates
    Divg_Slope = divg_slope_pre
    Trade_Profit = np.zeros(num_files, dtype=float)
    Trade_Loss = np.zeros(num_files, dtype=float)
    Trade_Brokerage_Paid = np.zeros(num_files, dtype=float)

    # Initialize trade accumulators
    current_total_cost = 0.0
    current_brokerage_paid = 0.0

    # Sequential trade execution loop
    for i in range(num_files):
        # Initialization
        if i == 0:
            Current_Capital_Value[i] = 10000
            Available_Capital_for_trade[i] = Current_Capital_Value[i]

        # Trade Execution
        if i > 0:
            if Buy_Signal[i-1] == 1 and Stoploss_Trigger[i-1] != 1:
                loss_per_unit[i] = (Brokerage_buy * prev_closes[i]) - (Brokerage_sell * Stoploss[i-1])
                Buy_Quantity[i] = (Risk_percentage * Available_Capital_for_trade[i-1]) / loss_per_unit[i] if loss_per_unit[i] > 0 else 0
                if Buy_Quantity[i] * opens[i] * Brokerage_buy > Available_Capital_for_trade[i-1]:
                    Buy_Quantity[i] = Available_Capital_for_trade[i-1] / (opens[i] * Brokerage_buy)
                Buy_Quantity[i] = max(Buy_Quantity[i], 0)
                Available_Capital_for_trade[i] = Available_Capital_for_trade[i-1] - (Buy_Quantity[i] * opens[i] * Brokerage_buy)
                Total_Buy_Quantity[i] = Total_Buy_Quantity[i-1] + Buy_Quantity[i]
                if Buy_Quantity[i] > 0:
                    Actual_Buy[i] = 1
                    buy_cost = Buy_Quantity[i] * opens[i] * Brokerage_buy
                    brokerage_on_this_buy = Buy_Quantity[i] * opens[i] * (Brokerage / 100)
                    current_total_cost += buy_cost
                    current_brokerage_paid += brokerage_on_this_buy
                    if Total_Buy_Quantity[i-1] == 0:
                        First_buy_date[i] = dates[i]
            elif Stoploss_Trigger[i-1] == 1 and Buy_Signal[i-1] != 1 and Total_Buy_Quantity[i-1] > 0:
                Sold_Quantity = Total_Buy_Quantity[i-1]
                proceeds = Sold_Quantity * opens[i] * Brokerage_sell
                brokerage_on_sell = Sold_Quantity * opens[i] * (Brokerage / 100)
                current_brokerage_paid += brokerage_on_sell
                pnl = proceeds - current_total_cost
                Trade_Profit[i] = max(pnl, 0)
                Trade_Loss[i] = abs(min(pnl, 0))
                Trade_Brokerage_Paid[i] = current_brokerage_paid
                Available_Capital_for_trade[i] = Available_Capital_for_trade[i-1] + proceeds
                Stoploss[i] = 0
                Actual_Sell[i] = 1
                # Reset accumulators
                current_total_cost = 0.0
                current_brokerage_paid = 0.0
            else:
                Available_Capital_for_trade[i] = Available_Capital_for_trade[i-1]
                Total_Buy_Quantity[i] = Total_Buy_Quantity[i-1]

        # Signal Monitoring
        Buy_Signal[i] = buy_signals_pre[i]
        if Buy_Signal[i] == 1:
            Buy_Signal_date[i] = dates[i]
            Stoploss[i] = initial_stoploss_pre[i]
        elif Total_Buy_Quantity[i] == 0:
            Stoploss[i] = 0
        else:
            Stoploss[i] = Stoploss[i-1] if i > 0 else 0

        Stoploss_Trigger[i] = Stoploss_Trigger_Check(closes[i], Stoploss[i])
        if Stoploss_Trigger[i] == 1 and Total_Buy_Quantity[i] > 0:
            Stoploss_Trigger_date[i] = dates[i]

        # Overall Parameter Monitoring
        if First_buy_date[i] != 0:
            First_buy_date[i] = First_buy_date[i]
        elif Actual_Sell[i] == 1:
            First_buy_date[i] = 0
        else:
            First_buy_date[i] = First_buy_date[i-1] if i > 0 else 0

        if Buy_Signal[i] != 1 and Total_Buy_Quantity[i] > 0:
            Stoploss[i] = Stoploss[i-1] if i > 0 else 0
        if Buy_Signal[i] == 1 and Actual_Sell[i] != 1 and (Total_Buy_Quantity[i-1] if i > 0 else 0) > 0:
            Stoploss[i] = min(Stoploss[i], Stoploss[i-1] if i > 0 else Stoploss[i])
        if Buy_Signal[i] != 1 and Actual_Sell[i] != 1 and (Total_Buy_Quantity[i-1] if i > 0 else 0) > 0:
            if LM_Low_window_1_CS_last_date[i] > First_buy_date[i] and LM_Low_window_1_CS_last[i] > Stoploss[i-1]:
                Stoploss[i] = LM_Low_window_1_CS_last[i]

        Current_Capital_Value[i] = Available_Capital_for_trade[i] + (Total_Buy_Quantity[i] * closes[i])

    # Combine into a DataFrame
    df = pd.DataFrame({
        'date': serial_date,
        'Buy_Signal': Buy_Signal,
        'Buy_Signal_Date': Buy_Signal_date,
        'Stoploss_Trigger': Stoploss_Trigger,
        'Stoploss_Trigger_Date': Stoploss_Trigger_date,
        'Actual_Buy': Actual_Buy,
        'Buy_Quantity': Buy_Quantity,
        'Total_Buy_Quantity': Total_Buy_Quantity,
        'Actual_Sell': Actual_Sell,
        'Available_Capital_for_trade': Available_Capital_for_trade,
        'Current_Capital_Value': Current_Capital_Value,
        'Stoploss': Stoploss,
        'LM_Low_window_1_CS_last': LM_Low_window_1_CS_last,
        'LM_Low_window_1_CS_last_date': LM_Low_window_1_CS_last_date,
        'First_buy_date': First_buy_date,
        'Divg_Slope': Divg_Slope,
        'Trade_Profit': Trade_Profit,
        'Trade_Loss': Trade_Loss,
        'Trade_Brokerage_Paid': Trade_Brokerage_Paid
    })

    # Save to CSV
    output_path = args.output_csv if getattr(args, 'output_csv', None) else output_combined_file
    df.to_csv(output_path, index=False)

    print(serial_date[0])
    print(Current_Capital_Value[-1])
    print(serial_date[-1])
