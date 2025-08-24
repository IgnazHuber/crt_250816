import pandas as pd
import os
import finplot as fplt
import glob
from datetime import datetime
import numpy as np
import time  # <-- neu: für präzise Laufzeitmessung
from Initialize_RSI_EMA_MACD import Initialize_RSI_EMA_MACD


# Set pandas display options to show all rows and columns
pd.set_option('display.max_rows', None)  # Show all rows
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.width', None)  # Adjust width to fit all columns

Risk_percentage = 1.00 # Percentage of total capital in case stoploss is hit
Brokerage = 0.1 # in %

Brokerage_buy = 1 + (Brokerage/100)
Brokerage_sell = 1 - (Brokerage/100)

# Example usage
folder_path = r'c:\\Projekte\\Anirudh\\ETH\\output_4hour_parquet'
#output_combined_file = 'backtest_results_ETH_4hour_100perc_with_brokerage.csv'
output_combined_file = 'c:\\Projekte\\Anirudh\\ETH\\output_4hour_parquet\\backtest_results_ETH_4hour_100perc_with_brokerage.csv'
#input_file = r'C:\Anirudh\Python\IBKR\Final_Version\ETH\eth_4hour_candlesticks_all.csv'
input_file = r'c:\\Projekte\Anirudh\\ETH\\output_4hour_parquet_10\\eth_4hour_candlesticks_all.csv'


# ===== Timing-Helfer =====
class Timings:
    def __init__(self):
        self.sections = {}
        self.per_file = {}   # dateiname -> dict(section -> sekunden)

    def add(self, key, seconds):
        self.sections[key] = self.sections.get(key, 0.0) + seconds

    def add_file(self, fname, key, seconds):
        d = self.per_file.setdefault(fname, {})
        d[key] = d.get(key, 0.0) + seconds

    def fmt(self, s):
        return f"{s:,.3f}s"

    def print_summary(self):
        print("\n================ Laufzeit-Übersicht (aggregiert) ================")
        total = 0.0
        for k, v in sorted(self.sections.items(), key=lambda x: -x[1]):
            print(f"{k:<30s}: {self.fmt(v)}")
            total += v
        print(f"{'-'*30}: {'-'*10}")
        print(f"{'Summe gemessener Abschnitte':<30s}: {self.fmt(total)}")
        print("=================================================================\n")

    def print_per_file(self, limit=10):
        # Optional: erste N Dateien detailliert ausgeben
        if not self.per_file:
            return
        print("================ Laufzeit je Datei (Top/erste N) ================")
        count = 0
        for fname, secs in self.per_file.items():
            print(f"[{fname}]")
            subtotal = 0.0
            for k, v in sorted(secs.items(), key=lambda x: -x[1]):
                print(f"  {k:<26s}: {self.fmt(v)}")
                subtotal += v
            print(f"  {'-'*26}: {'-'*10}")
            print(f"  {'Subtotal':<26s}: {self.fmt(subtotal)}\n")
            count += 1
            if count >= limit:
                break
        if len(self.per_file) > limit:
            print(f"... ({len(self.per_file)-limit} weitere Dateien nicht angezeigt)")
        print("=================================================================\n")


timings = Timings()
t_total_start = time.perf_counter()

# Get a list of all Parquet files in the folder and sort them
t0 = time.perf_counter()
parquet_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.parquet')])
timings.add("Dateiliste erstellen/sortieren", time.perf_counter() - t0)

# Get last non-zero value in LM_Low_window_1_CS and its date
def get_last_nonzero_lm_low(df):
    """
    Return the last non-zero value in LM_Low_window_1_CS and its date, or None if no non-zero values exist.
    """
    non_zero_df = df[df['LM_Low_window_1_CS'] != 0][['LM_Low_window_1_CS', 'date']].dropna()
    if non_zero_df.empty:
        return 0, 0
    return non_zero_df['LM_Low_window_1_CS'].iloc[-1], non_zero_df['date'].iloc[-1]

def Buy_Signal_Check(df):
    if df['HBullD_gen'].iloc[-2] == 1 and 70 > df['HBullD_Lower_Low_RSI_gen'].iloc[-2] > 40 and 70 > df['HBullD_Higher_Low_RSI_gen'].iloc[-2] > 40 and df['EMA_50'].iloc[-2] > df['EMA_200'].iloc[-2]:
        Buy_Signal = 1
        Stoploss = df['HBullD_Higher_Low_gen'].iloc[-2]
    elif df['HBullD_neg_MACD'].iloc[-2] == 1 and 70 > df['HBullD_Lower_Low_RSI_neg_MACD'].iloc[-2] > 40 and 70 > df['HBullD_Higher_Low_RSI_neg_MACD'].iloc[-2] > 40 and df['EMA_50'].iloc[-2] > df['EMA_200'].iloc[-2]:
        Buy_Signal = 1
        Stoploss = df['HBullD_Higher_Low_neg_MACD'].iloc[-2]
    elif ((df['CBullD_gen'].iloc[-2] == 1 and df['CBullD_neg_MACD'].iloc[-2] == 1) or df['CBullD_gen'].iloc[-2] == 1) and 55 > df['CBullD_Higher_Low_RSI_gen'].iloc[-2] > 30 and 55 > df['CBullD_Lower_Low_RSI_gen'].iloc[-2] > 15:
        Buy_Signal = 1
        Stoploss = df['CBullD_Lower_Low_gen'].iloc[-2]
    elif df['CBullD_neg_MACD'].iloc[-2] == 1 and 55 > df['CBullD_Higher_Low_RSI_neg_MACD'].iloc[-2] > 30 and 55 > df['CBullD_Lower_Low_RSI_neg_MACD'].iloc[-2] > 15:
        Buy_Signal = 1
        Stoploss = df['CBullD_Lower_Low_neg_MACD'].iloc[-2]
    elif df['CBullD_gen'].iloc[-2] == 1 and 55 > df['CBullD_Higher_Low_RSI_gen'].iloc[-2] > 30 and 55 > df['CBullD_Lower_Low_RSI_gen'].iloc[-2] > 15:
        Buy_Signal = 1
        Stoploss = df['CBullD_Lower_Low_gen'].iloc[-2]
    elif df['CBullD_x2'].iloc[-2] == 1:
        Buy_Signal = 1
        Stoploss = df['CBullD_x2_Lower_Low'].iloc[-2]
    else:
        Buy_Signal = 0
        Stoploss = 0
    return Buy_Signal, Stoploss

def Stoploss_Trigger_Check(df, Stoploss):
    if df['close'].iloc[-1] < Stoploss:
        Stoploss_Trigger = 1
    else:
        Stoploss_Trigger = 0
    return Stoploss_Trigger

# Initialize arrays
t_init_start = time.perf_counter()
Buy_Signal = [0] * len(parquet_files)
Buy_Signal_date = [0] * len(parquet_files)
Actual_Buy = [0] * len(parquet_files)
First_buy_date = [0] * len(parquet_files)
Stoploss = [0] * len(parquet_files)
Stoploss_Trigger = [0] * len(parquet_files)
Stoploss_Trigger_date = [0] * len(parquet_files)
Last_LM_window_1_CS = [0] * len(parquet_files)
Last_LM_window_1_CS_date = [0] * len(parquet_files)
Actual_Sell = [0] * len(parquet_files)
Current_Capital_Value = [0] * len(parquet_files)
Available_Capital_for_trade = [0] * len(parquet_files)
Buy_Quantity = [0] * len(parquet_files)
Total_Buy_Quantity = [0] * len(parquet_files)
loss_per_unit = [0] * len(parquet_files)
serial_date = [0] * len(parquet_files)
LM_Low_window_1_CS_last = [0] * len(parquet_files)
LM_Low_window_1_CS_last_date = [0] * len(parquet_files)
timings.add("Arrays initialisieren", time.perf_counter() - t_init_start)

# Loop through each Parquet file
t_loop_total = 0.0
for i, file_name in enumerate(parquet_files):
    file_t0 = time.perf_counter()
    # Construct full file path
    file_path = os.path.join(folder_path, file_name)

    # Read the Parquet file
    t_read_start = time.perf_counter()
    df = pd.read_parquet(file_path).tail(100)
    t_read = time.perf_counter() - t_read_start
    timings.add("Parquet lesen (gesamt)", t_read)
    timings.add_file(file_name, "Parquet lesen", t_read)

    serial_date[i] = df['date'].iloc[-1]

    # Initialization
    if i == 0:
        Current_Capital_Value[i] = 10000
        Available_Capital_for_trade[i] = Current_Capital_Value[i]

    # Trade Execution
    t_trade_start = time.perf_counter()
    if i > 0:
        if Buy_Signal[i-1] == 1 and Stoploss_Trigger[i-1] != 1:
            loss_per_unit[i] = ((Brokerage_buy*df['close'].iloc[-2]) - (Brokerage_sell*Stoploss[i-1]))
            if loss_per_unit[i] > 0:
                Buy_Quantity[i] = (Risk_percentage * Available_Capital_for_trade[i - 1]) / loss_per_unit[i]
            else:
                Buy_Quantity[i] = 0

            if (Buy_Quantity[i] * df['open'].iloc[-1] * Brokerage_buy) > Available_Capital_for_trade[i-1]:
                Buy_Quantity[i] = Available_Capital_for_trade[i-1] / (df['open'].iloc[-1] * Brokerage_buy)

            if Buy_Quantity[i] < 0:
                Buy_Quantity[i] = 0

            Available_Capital_for_trade[i] = Available_Capital_for_trade[i-1] - (Buy_Quantity[i] * df['open'].iloc[-1] * Brokerage_buy)
            Total_Buy_Quantity[i] = Total_Buy_Quantity[i-1] + Buy_Quantity[i]

            if Buy_Quantity[i] > 0:
                Actual_Buy[i] = 1
                # print('Actual Buying on,', df['date'].iloc[-1])
                if Total_Buy_Quantity[i-1] == 0:
                    First_buy_date[i] = df['date'].iloc[-1]

        if Stoploss_Trigger[i-1] == 1 and Buy_Signal[i-1] != 1 and Total_Buy_Quantity[i-1] > 0:
            Sold_Quantity = Total_Buy_Quantity[i-1]
            Available_Capital_for_trade[i] = Available_Capital_for_trade[i-1] + (Sold_Quantity * df['open'].iloc[-1] * Brokerage_sell)
            Stoploss[i] = 0
            Actual_Sell[i] = 1
            # print('Actual Selling on,', df['date'].iloc[-1])

        if Actual_Buy[i] != 1 and Actual_Sell[i] != 1:
            Available_Capital_for_trade[i] = Available_Capital_for_trade[i - 1]
            Total_Buy_Quantity[i] = Total_Buy_Quantity[i-1]
    t_trade = time.perf_counter() - t_trade_start
    timings.add("Trade-Execution (gesamt)", t_trade)
    timings.add_file(file_name, "Trade-Execution", t_trade)

    # Signal Monitoring
    t_signal_start = time.perf_counter()
    Buy_Signal[i], Stoploss[i] = Buy_Signal_Check(df)
    if Buy_Signal[i] == 1:
        Buy_Signal_date[i] = df['date'].iloc[-1]
    else:
        Stoploss[i] = Stoploss[i - 1]

    Stoploss_Trigger[i] = Stoploss_Trigger_Check(df, Stoploss[i])
    if Stoploss_Trigger[i] == 1 and Total_Buy_Quantity[i] > 0:
        Stoploss_Trigger_date[i] = df['date'].iloc[-1]
    t_signal = time.perf_counter() - t_signal_start
    timings.add("Signal-Monitoring (gesamt)", t_signal)
    timings.add_file(file_name, "Signal-Monitoring", t_signal)

    # Overall Parameter Monitoring
    t_param_start = time.perf_counter()
    if First_buy_date[i] != 0:
        First_buy_date[i] = First_buy_date[i]
    elif Actual_Sell[i] == 1:
        First_buy_date[i] = 0
    else:
        First_buy_date[i] = First_buy_date[i-1]

    LM_Low_window_1_CS_last[i], LM_Low_window_1_CS_last_date[i] = get_last_nonzero_lm_low(df)

    if Buy_Signal[i] != 1:
        Stoploss[i] = Stoploss[i-1]
    if Buy_Signal[i] == 1 and Actual_Sell[i] != 1 and Total_Buy_Quantity[i-1] > 0:
        Stoploss[i] = min(Stoploss[i], Stoploss[i - 1])
    if Buy_Signal[i] != 1 and Actual_Sell[i] != 1 and Total_Buy_Quantity[i-1] > 0:
        if LM_Low_window_1_CS_last_date[i] > First_buy_date[i] and LM_Low_window_1_CS_last[i] > Stoploss[i - 1]:
            Stoploss[i] = LM_Low_window_1_CS_last[i]

    Current_Capital_Value[i] = Available_Capital_for_trade[i] + (Total_Buy_Quantity[i] * df['close'].iloc[-1])
    t_param = time.perf_counter() - t_param_start
    timings.add("Parameter-Update/Stoploss-Logik (gesamt)", t_param)
    timings.add_file(file_name, "Parameter-Update/Stoploss-Logik", t_param)

    timings.add_file(file_name, "Datei-Gesamtdurchlauf", time.perf_counter() - file_t0)

# Combine into a DataFrame
t_df_start = time.perf_counter()
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
    'First_buy_date': First_buy_date
})
timings.add("DataFrame zusammenbauen", time.perf_counter() - t_df_start)

# Save to CSV
t_csv_start = time.perf_counter()
df.to_csv(output_combined_file, index=False)
timings.add("CSV schreiben", time.perf_counter() - t_csv_start)

print(serial_date[0])
print(Current_Capital_Value[len(df)-1])
print(serial_date[len(df)-1])

# # Read the last Parquet file for plotting
# df_1 = pd.read_csv(input_file)
# Initialize_RSI_EMA_MACD(df_1)
#
#
# # Read the backtest results CSV
# df_2 = pd.read_csv(output_combined_file)
#
# fplt.background = fplt.odd_plot_background = '#242320'  # Adjust Plot Background colour
# fplt.cross_hair_color = '#eefa'  # Adjust Crosshair colour
#
# # Plotting Chart
# ax1, ax2, ax3 = fplt.create_plot('Chart', rows=3)
# df_1['date'] = pd.to_datetime(df_1['date'], format='mixed')
# candles = df_1[['date', 'open', 'close', 'high', 'low', 'macd_histogram']]
# fplt.candlestick_ochl(candles, ax=ax1)
#
# # Plotting RSI
# fplt.plot(df_1.RSI, color='#000000', width=2, ax=ax2, legend='RSI')
# fplt.set_y_range(0, 100, ax=ax2)
# fplt.add_horizontal_band(0, 1, color='#000000', ax=ax2)
# fplt.add_horizontal_band(99, 100, color='#000000', ax=ax2)
#
# # Plotting the MACD
# fplt.volume_ocv(df_1[['date', 'open', 'close', 'macd_histogram']], ax=ax3, colorfunc=fplt.strength_colorfilter)
#
# # Plotting EMAs
# df_1.EMA_20.plot(ax=ax1, legend='20-EMA')
# df_1.EMA_50.plot(ax=ax1, legend='50-EMA')
# df_1.EMA_100.plot(ax=ax1, legend='100-EMA')
# df_1.EMA_200.plot(ax=ax1, legend='200-EMA')
#
# # Plot buy/sell signals
# for i in range(0, len(df_2)):
#     if df_2['Actual_Buy'][i] == 1:
#         fplt.plot(pd.to_datetime(df_2['date'][i]), df_1['open'][i+200], style='o', ax=ax1, color='yellow')
#     if df_2['Actual_Sell'][i] == 1:
#         fplt.plot(pd.to_datetime(df_2['date'][i]), df_1['open'][i+200], style='o', ax=ax1, color='white')
#
# fplt.show()

# ===== Gesamtlaufzeit ausgeben =====
t_total_end = time.perf_counter()
total_seconds = t_total_end - t_total_start
print(f"\nGesamtlaufzeit: {total_seconds:,.3f}s")

# Detaillierte Übersicht
timings.print_summary()
# Optional: je Datei (hier: erste 10)
timings.print_per_file(limit=10)
