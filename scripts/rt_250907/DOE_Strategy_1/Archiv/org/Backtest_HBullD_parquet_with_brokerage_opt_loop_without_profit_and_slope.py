import pandas as pd
import os
import numpy as np
from datetime import datetime
from multiprocessing import Pool, cpu_count
import glob
from Initialize_RSI_EMA_MACD import Initialize_RSI_EMA_MACD

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

def Buy_Signal_Check(d, hb_upper, hb_lower, cb_upper, cb_hl_lower, cb_ll_lower):
    if (d['hb_gen'] == 1) and (d['hb_ll_rsi_gen'] < hb_upper) and (d['hb_ll_rsi_gen'] > hb_lower) and (d['hb_hl_rsi_gen'] < hb_upper) and (d['hb_hl_rsi_gen'] > hb_lower) and (d['ema50_prev'] > d['ema200_prev']):
        return 1, d['hb_hl_gen']
    elif (d['hb_neg_macd'] == 1) and (d['hb_ll_rsi_neg'] < hb_upper) and (d['hb_ll_rsi_neg'] > hb_lower) and (d['hb_hl_rsi_neg'] < hb_upper) and (d['hb_hl_rsi_neg'] > hb_lower) and (d['ema50_prev'] > d['ema200_prev']):
        return 1, d['hb_hl_neg']
    elif (d['cb_gen'] == 1) and (d['cb_neg_macd'] == 1) and (d['cb_hl_rsi_gen'] < cb_upper) and (d['cb_hl_rsi_gen'] > cb_hl_lower) and (d['cb_ll_rsi_gen'] < cb_upper) and (d['cb_ll_rsi_gen'] > cb_ll_lower):
        return 1, d['cb_ll_gen']
    elif (d['cb_neg_macd'] == 1) and (d['cb_hl_rsi_neg'] < cb_upper) and (d['cb_hl_rsi_neg'] > cb_hl_lower) and (d['cb_ll_rsi_neg'] < cb_upper) and (d['cb_ll_rsi_neg'] > cb_ll_lower):
        return 1, d['cb_ll_neg']
    elif (d['cb_gen'] == 1) and (d['cb_hl_rsi_gen'] < cb_upper) and (d['cb_hl_rsi_gen'] > cb_hl_lower) and (d['cb_ll_rsi_gen'] < cb_upper) and (d['cb_ll_rsi_gen'] > cb_ll_lower):
        return 1, d['cb_ll_gen']
    elif (d['cb_x2'] == 1):
        return 1, d['cb_x2_ll']
    else:
        return 0, 0

if __name__ == '__main__':
    # Example usage
    folder_path = r'C:\Anirudh\Python\IBKR\Incremental\KOTAKBANK\output_4hour_parquet'
    output_combined_file = 'backtest_results_KOTAK_4hour_100perc_with_brokerage.csv'
    optimization_output = 'optimization_results.csv'

    # Get sorted list of Parquet files
    parquet_files = sorted(glob.glob(os.path.join(folder_path, '*.parquet')))
    num_files = len(parquet_files)

    # Load data in parallel
    data = parallel_load_files(parquet_files)
    if not data:
        raise ValueError("No valid data loaded from Parquet files")

    # Optimization loop
    results = []
    for hb_upper in range(60, 81, 5):
        for hb_lower in range(30, 51, 5):
            if hb_lower >= hb_upper:
                continue
            for cb_upper in range(45, 66, 5):
                for cb_hl_lower in range(20, 41, 5):
                    if cb_hl_lower >= cb_upper:
                        continue
                    for cb_ll_lower in range(5, 26, 5):
                        if cb_ll_lower >= cb_upper:
                            continue

                        # Initialize lists for each combination
                        Buy_Signal = [0] * num_files
                        Buy_Signal_date = [0] * num_files
                        Actual_Buy = [0] * num_files
                        First_buy_date = [0] * num_files
                        Stoploss = [0.0] * num_files
                        Stoploss_Trigger = [0] * num_files
                        Stoploss_Trigger_date = [0] * num_files
                        Actual_Sell = [0] * num_files
                        Current_Capital_Value = [0.0] * num_files
                        Available_Capital_for_trade = [0.0] * num_files
                        Buy_Quantity = [0.0] * num_files
                        Total_Buy_Quantity = [0.0] * num_files
                        loss_per_unit = [0.0] * num_files
                        serial_date = [0] * num_files
                        LM_Low_window_1_CS_last = [0.0] * num_files
                        LM_Low_window_1_CS_last_date = [0] * num_files

                        # Sequential trade execution loop
                        for i in range(num_files):
                            d = data[i]
                            serial_date[i] = d['date']

                            # Initialization
                            if i == 0:
                                Current_Capital_Value[i] = 10000
                                Available_Capital_for_trade[i] = Current_Capital_Value[i]

                            # Trade Execution
                            if i > 0:
                                if Buy_Signal[i-1] == 1 and Stoploss_Trigger[i-1] != 1:
                                    loss_per_unit[i] = (Brokerage_buy * d['prev_close']) - (Brokerage_sell * Stoploss[i-1])
                                    if loss_per_unit[i] > 0:
                                        Buy_Quantity[i] = (Risk_percentage * Available_Capital_for_trade[i-1]) / loss_per_unit[i]
                                    else:
                                        Buy_Quantity[i] = 0
                                    if Buy_Quantity[i] * d['open'] * Brokerage_buy > Available_Capital_for_trade[i-1]:
                                        Buy_Quantity[i] = Available_Capital_for_trade[i-1] / (d['open'] * Brokerage_buy)
                                    Buy_Quantity[i] = max(Buy_Quantity[i], 0)
                                    Available_Capital_for_trade[i] = Available_Capital_for_trade[i-1] - (Buy_Quantity[i] * d['open'] * Brokerage_buy)
                                    Total_Buy_Quantity[i] = Total_Buy_Quantity[i-1] + Buy_Quantity[i]
                                    if Buy_Quantity[i] > 0:
                                        Actual_Buy[i] = 1
                                        if Total_Buy_Quantity[i-1] == 0:
                                            First_buy_date[i] = d['date']
                                elif Stoploss_Trigger[i-1] == 1 and Buy_Signal[i-1] != 1 and Total_Buy_Quantity[i-1] > 0:
                                    Sold_Quantity = Total_Buy_Quantity[i-1]
                                    Available_Capital_for_trade[i] = Available_Capital_for_trade[i-1] + (Sold_Quantity * d['open'] * Brokerage_sell)
                                    Stoploss[i] = 0
                                    Actual_Sell[i] = 1
                                else:
                                    Available_Capital_for_trade[i] = Available_Capital_for_trade[i-1]
                                    Total_Buy_Quantity[i] = Total_Buy_Quantity[i-1]

                            # Signal Monitoring
                            Buy_Signal[i], Stoploss[i] = Buy_Signal_Check(d, hb_upper, hb_lower, cb_upper, cb_hl_lower, cb_ll_lower)
                            if Buy_Signal[i] == 1:
                                Buy_Signal_date[i] = d['date']
                            else:
                                Stoploss[i] = Stoploss[i-1] if i > 0 else 0.0

                            Stoploss_Trigger[i] = Stoploss_Trigger_Check(d['close'], Stoploss[i])
                            if Stoploss_Trigger[i] == 1 and Total_Buy_Quantity[i] > 0:
                                Stoploss_Trigger_date[i] = d['date']

                            # Overall Parameter Monitoring
                            if First_buy_date[i] != 0:
                                First_buy_date[i] = First_buy_date[i]
                            elif Actual_Sell[i] == 1:
                                First_buy_date[i] = 0
                            else:
                                First_buy_date[i] = First_buy_date[i-1] if i > 0 else 0

                            LM_Low_window_1_CS_last[i] = d['lm_low']
                            LM_Low_window_1_CS_last_date[i] = d['lm_date']

                            if Buy_Signal[i] != 1:
                                Stoploss[i] = Stoploss[i-1] if i > 0 else 0.0
                            if Buy_Signal[i] == 1 and Actual_Sell[i] != 1 and (Total_Buy_Quantity[i-1] if i > 0 else 0) > 0:
                                Stoploss[i] = min(Stoploss[i], Stoploss[i-1] if i > 0 else Stoploss[i])
                            if Buy_Signal[i] != 1 and Actual_Sell[i] != 1 and (Total_Buy_Quantity[i-1] if i > 0 else 0) > 0:
                                if LM_Low_window_1_CS_last_date[i] > First_buy_date[i] and LM_Low_window_1_CS_last[i] > Stoploss[i-1]:
                                    Stoploss[i] = LM_Low_window_1_CS_last[i]

                            Current_Capital_Value[i] = Available_Capital_for_trade[i] + (Total_Buy_Quantity[i] * d['close'])

                        # Print results for this parameter set
                        print(f"Parameters: hb_upper={hb_upper}, hb_lower={hb_lower}, cb_upper={cb_upper}, cb_hl_lower={cb_hl_lower}, cb_ll_lower={cb_ll_lower}")
                        print(serial_date[0])
                        print(Current_Capital_Value[-1])
                        print(serial_date[-1])
                        print("---")

                        # Collect results
                        results.append({
                            'hb_upper': hb_upper,
                            'hb_lower': hb_lower,
                            'cb_upper': cb_upper,
                            'cb_hl_lower': cb_hl_lower,
                            'cb_ll_lower': cb_ll_lower,
                            'Final_Capital': Current_Capital_Value[-1]
                        })

    # Convert results to DataFrame and find the optimal combination
    results_df = pd.DataFrame(results)
    optimal = results_df.loc[results_df['Final_Capital'].idxmax()]

    # Save results to CSV
    results_df.to_csv(optimization_output, index=False)

    # Print the optimal combination
    print(f"Optimal parameters: hb_upper={optimal['hb_upper']}, hb_lower={optimal['hb_lower']}, cb_upper={optimal['cb_upper']}, cb_hl_lower={optimal['cb_hl_lower']}, cb_ll_lower={optimal['cb_ll_lower']}, Final Capital: {optimal['Final_Capital']}")