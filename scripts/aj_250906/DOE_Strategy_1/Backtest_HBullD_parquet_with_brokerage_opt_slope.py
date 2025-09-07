import pandas as pd
import os
import numpy as np
from multiprocessing import Pool, cpu_count
import glob

# Set pandas display options
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

Risk_percentage = 1.00
Brokerage = 0.1  # in %
Brokerage_buy = 1 + (Brokerage / 100)
Brokerage_sell = 1 - (Brokerage / 100)

# Function to process a single Parquet file
def process_parquet_file(file_path):
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
            'HBullD_Date_Gap_gen', 'HBullD_Date_Gap_neg_MACD'
        ]
        df = pd.read_parquet(file_path, columns=required_columns).tail(100)
        if len(df) < 2:
            return None
        last_row = df.iloc[-1]
        prev_row = df.iloc[-2]
        non_zero_df = df[df['LM_Low_window_1_CS'] != 0][['LM_Low_window_1_CS', 'date']].dropna()
        lm_low, lm_date = (non_zero_df['LM_Low_window_1_CS'].iloc[-1], non_zero_df['date'].iloc[-1]) if not non_zero_df.empty else (0, 0)
        return {
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
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Parallel processing of Parquet files
def parallel_load_files(file_paths):
    with Pool(processes=cpu_count()) as pool:
        results = pool.map(process_parquet_file, file_paths)
    return [r for r in results if r is not None]

# Stoploss trigger check
def Stoploss_Trigger_Check(close, stoploss):
    return 1 if close < stoploss else 0

if __name__ == '__main__':
    # Example usage
    folder_path = r'C:\Anirudh\Python\IBKR\Incremental\ETH\output_4hour_parquet'
    output_combined_file = 'backtest_results_ETH_4hour_100perc_with_brokerage_orig.csv'

    # Get sorted list of Parquet files
    parquet_files = sorted(glob.glob(os.path.join(folder_path, '*.parquet')))
    num_files = len(parquet_files)

    # Load data in parallel
    data = parallel_load_files(parquet_files)
    if not data:
        raise ValueError("No valid data loaded from Parquet files")

    # Preallocate arrays
    dates = np.array([d['date'] for d in data], dtype=object)
    opens = np.array([d['open'] for d in data], dtype=float)
    closes = np.array([d['close'] for d in data], dtype=float)
    prev_closes = np.array([d['prev_close'] for d in data], dtype=float)
    lm_lows = np.array([d['lm_low'] for d in data], dtype=float)
    lm_dates = np.array([d['lm_date'] for d in data], dtype=object)
    ema50_prev = np.array([d['ema50_prev'] for d in data], dtype=float)
    ema200_prev = np.array([d['ema200_prev'] for d in data], dtype=float)
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

    # Vectorized buy signal and initial stoploss calculation
    cond1 = (hb_gen == 1) & (hb_ll_rsi_gen < 70) & (hb_ll_rsi_gen > 40) & (hb_hl_rsi_gen < 70) & (hb_hl_rsi_gen > 40) & (ema50_prev > ema200_prev) & (closes > hb_hl_gen)
    cond2 = (hb_neg_macd == 1) & (hb_ll_rsi_neg < 70) & (hb_ll_rsi_neg > 40) & (hb_hl_rsi_neg < 70) & (hb_hl_rsi_neg > 40) & (ema50_prev > ema200_prev) & (closes > hb_hl_neg)
    cond3 = (((cb_gen == 1) & (cb_neg_macd == 1)) | (cb_gen == 1)) & (cb_hl_rsi_gen < 55) & (cb_hl_rsi_gen > 30) & (cb_ll_rsi_gen < 55) & (cb_ll_rsi_gen > 15) & (closes > cb_ll_gen)
    cond4 = (cb_neg_macd == 1) & (cb_hl_rsi_neg < 55) & (cb_hl_rsi_neg > 30) & (cb_ll_rsi_neg < 55) & (cb_ll_rsi_neg > 15) & (closes > cb_ll_neg)
    cond5 = (cb_gen == 1) & (cb_hl_rsi_gen < 55) & (cb_hl_rsi_gen > 30) & (cb_ll_rsi_gen < 55) & (cb_ll_rsi_gen > 15) & (closes > cb_ll_gen)
    cond6 = (cb_x2 == 1) & (closes > cb_x2_ll)
    conds = [cond1, cond2, cond3, cond4, cond5, cond6]
    stop_choices = [hb_hl_gen, hb_hl_neg, cb_ll_gen, cb_ll_neg, cb_ll_gen, cb_x2_ll]
    buy_signals_pre = np.select(conds, [1] * len(conds), default=0)
    initial_stoploss_pre = np.select(conds, stop_choices, default=0)

    # Vectorized Divg_Slope calculation
    divg1 = 10000 * (np.abs(hb_ll_gen - hb_hl_gen) / hb_ll_gen) / hb_date_gap_gen
    divg2 = 10000 * (np.abs(hb_ll_neg - hb_hl_neg) / hb_ll_neg) / hb_date_gap_neg
    divg3 = 10000 * (np.abs(cb_ll_gen - cb_hl_gen) / cb_hl_gen) / cb_date_gap_gen
    divg4 = 10000 * (np.abs(cb_ll_neg - cb_hl_neg) / cb_hl_neg) / cb_date_gap_neg
    divg5 = divg3
    divg6 = np.zeros_like(divg1)
    divg_choices = [divg1, divg2, divg3, divg4, divg5, divg6]
    divg_slope_pre = np.select(conds, divg_choices, default=0.0)
    divg_slope_pre = np.nan_to_num(divg_slope_pre, nan=0.0, posinf=0.0, neginf=0.0)

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
    df.to_csv(output_combined_file, index=False)

    print(serial_date[0])
    print(Current_Capital_Value[-1])
    print(serial_date[-1])