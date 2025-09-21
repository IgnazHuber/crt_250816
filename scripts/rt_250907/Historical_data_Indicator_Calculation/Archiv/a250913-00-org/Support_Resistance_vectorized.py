import warnings
import numpy as np
import pandas as pd
from Minimas_Maximas_Kernel import Min_Max_Pivots

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def initialize_support_columns(df, num_supports=16):
    """Initialize support columns with NaN in a vectorized manner."""
    support_cols = {f'Support_{i}': np.nan for i in range(1, num_supports + 1)}
    return df.assign(**support_cols)

def filter_data_by_date(df, current_date, lookback_years):
    """Filter DataFrame to include rows within lookback_years from current_date."""
    cutoff_date = current_date - pd.DateOffset(years=lookback_years)
    return df[df['date'] >= cutoff_date].copy()

def assign_support_values(df, support_values, last_row_idx, num_supports=16):
    """Assign support values to the last row in a vectorized manner."""
    # Ensure support_values has exactly num_supports values, pad with 0 if needed
    support_values = support_values[:num_supports] + [0] * (num_supports - len(support_values))
    # Assign to support columns for the last row
    df.loc[last_row_idx, [f'Support_{i}' for i in range(1, num_supports + 1)]] = support_values
    return df

def calculate_support_levels(df, lookback_years, pivot_threshold):
    """
    Calculate support levels for the entire DataFrame, filtered by lookback_years from the last row's date.

    Parameters:
    - df: pandas DataFrame with columns ['date', 'open', 'high', 'low', 'close']
    - lookback_years: Number of years to look back from the last row's date
    - pivot_threshold: Threshold for Min_Max_Pivots function

    Returns:
    - DataFrame with additional Support_1 to Support_16 columns
    """
    # Ensure date column is in datetime format and sort by date
    df = df.copy()  # Avoid modifying the input DataFrame
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    # Initialize support columns in a vectorized way
    df = initialize_support_columns(df)

    # Get the last row's index and date
    last_row_idx = len(df) - 1
    current_date = df.loc[last_row_idx, 'date']

    # Filter data for the lookback period
    df_step = filter_data_by_date(df, current_date, lookback_years)

    # Check if df_step has enough data
    if len(df_step) < 10:  # Arbitrary minimum to avoid errors in Min_Max_Pivots
        # Assign 0 to all support columns for the last row
        df.loc[last_row_idx, [f'Support_{i}' for i in range(1, 17)]] = 0
        return df

    # Calculate Min-Max Pivots for the filtered subset
    support_values = Min_Max_Pivots(df_step, pivot_threshold)

    # Assign support values to the last row
    df = assign_support_values(df, list(support_values), last_row_idx)

    return df