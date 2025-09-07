import pandas as pd
import os
import finplot as fplt
import glob
from datetime import datetime
import numpy as np
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
folder_path = r'c:\\Projekte\\Anirudh\\ETH\\output_4hour_parquet_10'
#output_combined_file = 'backtest_results_ETH_4hour_100perc_with_brokerage.csv'
output_combined_file = 'c:\\Projekte\\Anirudh\\ETH\\output_4hour_parquet_10\\backtest_results_ETH_4hour_100perc_with_brokerage_10.csv'
#input_file = r'C:\Anirudh\Python\IBKR\Final_Version\ETH\eth_4hour_candlesticks_all_10.csv'
input_file = r'c:\\Projekte\Anirudh\\ETH\\output_4hour_parquet_10\\eth_4hour_candlesticks_all.csv'


# Get a list of all Parquet files in the folder and sort them
parquet_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.parquet')])

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

# Loop through each Parquet file
for i, file_name in enumerate(parquet_files):
    # Construct full file path
    file_path = os.path.join(folder_path, file_name)

    # Read the Parquet file
    df = pd.read_parquet(file_path).tail(100)

    serial_date[i] = df['date'].iloc[-1]

    # Initialization
    if i == 0:
        Current_Capital_Value[i] = 10000
        Available_Capital_for_trade[i] = Current_Capital_Value[i]

    # Trade Execution
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

    # Signal Monitoring
    Buy_Signal[i], Stoploss[i] = Buy_Signal_Check(df)
    if Buy_Signal[i] == 1:
        Buy_Signal_date[i] = df['date'].iloc[-1]
    else:
        Stoploss[i] = Stoploss[i - 1]

    Stoploss_Trigger[i] = Stoploss_Trigger_Check(df, Stoploss[i])
    if Stoploss_Trigger[i] == 1 and Total_Buy_Quantity[i] > 0:
        Stoploss_Trigger_date[i] = df['date'].iloc[-1]

    # Overall Parameter Monitoring
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
    'First_buy_date': First_buy_date
})

# Save to CSV
df.to_csv(output_combined_file, index=False)

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