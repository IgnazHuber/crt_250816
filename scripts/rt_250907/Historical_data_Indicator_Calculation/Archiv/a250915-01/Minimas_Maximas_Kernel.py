import numpy as np
import pandas as pd
import warnings
from Kernel import Kernel

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def Min_Max_Pivots(ohlc: pd.DataFrame, percentage_range)-> pd.Series:

    date_1 = pd.to_datetime(ohlc["date"]).dt.strftime('%Y-%m-%d %H:%M:%S') # read candlestick timestamp (series class)
    date_2 = pd.to_datetime(ohlc["date"]).dt.strftime('%Y-%m-%d %H:%M:%S')  # read candlestick timestamp (series class)

    LM_High = ohlc['LM_High_window_1_CS']

    LM_Low = ohlc['LM_Low_window_1_CS']

#------------------------------------------------------------------------------------------------------
    combined_array = list(zip(date_1, LM_Low))      # Combine date and LM_Low into a list of tuples
    filtered_data = [(date, low) for date, low in combined_array if low != 0]        # Filter out rows where LM_Low is zero
    if filtered_data:   # Handle the case where all values are zero
        date_1, LM_Low_nzd = zip(*filtered_data)
        LM_Low_nzd = np.array(LM_Low_nzd)
    else:
        date_1 = []  # Empty list for dates
        LM_Low_nzd = np.array([0])  # Default to [0] if all values were zero

# ------------------------------------------------------------------------------------------------------
    combined_array = list(zip(date_2, LM_High))     # Combine date and LM_High into a list of tuples
    filtered_data = [(date, high) for date, high in combined_array if high != 0]      # Filter out rows where LM_High is zero
    if filtered_data:       # Handle the case where all values are zero
        date_2, LM_High_nzd = zip(*filtered_data)
        LM_High_nzd = np.array(LM_High_nzd)
    else:
        date_2 = []  # Empty list for dates
        LM_High_nzd = np.array([0])  # Default to [0] if all values were zero
# ------------------------------------------------------------------------------------------------------
    # Combine the data for Local highs and lows in a single array------------------------------------
    combi_LM_High_Low = np.concatenate((LM_Low_nzd, LM_High_nzd))
    combi_LM_High_Low = np.sort(combi_LM_High_Low)

    Kernel_kde= Kernel(combi_LM_High_Low)
    Kernel_kde = np.array(list(Kernel_kde.values()))

    #-----To remove levels which are within 0.5 % of each other (% high of one w.r.t. % low of another)------------
    percentage_low = (1-percentage_range/100)*Kernel_kde
    percentage_high = (1 + percentage_range / 100) * Kernel_kde

    # Compare high values of a row with low values of the next row
    comparison_result = np.zeros(len(Kernel_kde) - 1, dtype=int)
    for i in range(len(Kernel_kde) - 1):
        if percentage_high[i] == 0:     # Handle the case where High Array is empty/zeros
            percentage_high[i] = 0.00001
            percent_difference = abs((percentage_low[i + 1] - percentage_high[i]) / percentage_high[i])
        else:
            percent_difference = abs((percentage_low[i + 1] - percentage_high[i]) / percentage_high[i])
        if percent_difference < 0.01:      # Thersold is 1.0 % between %high-low
            comparison_result[i] = 0
        else:
            comparison_result[i] = 1

    for i in range(len(Kernel_kde) - 1):
        Kernel_kde[i] = Kernel_kde[i]*comparison_result[i]

    Kernel_kde = Kernel_kde[Kernel_kde != 0]

    return pd.Series(Kernel_kde)