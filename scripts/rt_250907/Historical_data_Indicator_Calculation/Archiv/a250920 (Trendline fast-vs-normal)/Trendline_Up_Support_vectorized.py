import pandas as pd
import numpy as np
import warnings
from numba import jit, prange
from itertools import combinations

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)

@jit(nopython=True, parallel=True)
def is_invalid_trendline_vectorized(opens, highs, lows, closes, slope, y_intercept, n):
    """
    Vectorized check for invalid trendlines using NumPy operations.
    Returns True if the trendline is invalid.
    """
    x = np.arange(n)
    trendline_y = slope * x + y_intercept

    # Check for intersections with candle bodies
    body_intersections = (
        ((opens < trendline_y) & (closes > trendline_y)) |
        ((opens > trendline_y) & (closes < trendline_y))
    )

    # Check for intersections with lower wicks
    lower_wick_intersections = (closes > trendline_y) & (lows < trendline_y)

    # Check for candles entirely below the trendline
    below_trendline = (
        (highs < trendline_y) & (lows < trendline_y) &
        (opens < trendline_y) & (closes < trendline_y)
    )

    return (
        lower_wick_intersections.sum() > 1 or
        body_intersections.sum() > 0 or
        below_trendline.any()
    )

def calc_TL_Up_Support(df, min_gap, adjacent_candles, exclude_end_points):
    """
    Calculate upward trendlines for the entire DataFrame up to the last row and add their coordinates and dates to the last row.
    If no trendlines are found, populate with dummy values (0 for prices, first/last dates for dates).
    Uses LM_Low_window_2_CS for local lows.

    Parameters:
    - df: pandas DataFrame with columns ['date', 'open', 'high', 'low', 'close', 'LM_Low_window_2_CS']
    - min_gap: Minimum gap between candles for trendline points
    - adjacent_candles: Number of adjacent candles to check for clustering
    - exclude_end_points: Number of data points to exclude from the end for the second low

    Returns:
    - DataFrame with additional columns for up to 7 trendlines:
      ['TL_U_Support_{i}_Start_Date', 'TL_U_Support_{i}_End_Date', 'TL_U_Support_{i}_Start_Price', 'TL_U_Support_{i}_End_Price']
    """
    # Make a copy and ensure date is datetime, sort by date, and reset index
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'], utc=True).dt.tz_localize(None)
    df = df.sort_values('date').reset_index(drop=True)

    # Initialize trendline columns with dummy values
    first_date = df['date'].iloc[0]
    last_date = df['date'].iloc[-1]
    for i in range(1, 8):
        df[f'TL_U_Support_{i}_Start_Date'] = pd.NaT  # Use NaT for datetime columns
        df[f'TL_U_Support_{i}_End_Date'] = pd.NaT
        df[f'TL_U_Support_{i}_Start_Price'] = 0
        df[f'TL_U_Support_{i}_End_Price'] = 0

    # Set dummy values in the last row
    last_row_idx = len(df) - 1
    for i in range(1, 8):
        df.loc[last_row_idx, f'TL_U_Support_{i}_Start_Date'] = first_date
        df.loc[last_row_idx, f'TL_U_Support_{i}_End_Date'] = last_date

    # Cache DataFrame columns as NumPy arrays for faster access
    dates = df['date'].values
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    lm_lows = df['LM_Low_window_2_CS'].values
    n = len(df)

    # Identify indices of local lows
    local_low_indices = np.where(lm_lows != 0)[0]

    # Generate all possible pairs of local lows with minimum gap
    pairs = np.array(list(combinations(local_low_indices, 2)))
    valid_pairs = pairs[pairs[:, 1] - pairs[:, 0] >= min_gap]

    # Further filter pairs where second low is not lower (upward trendlines)
    valid_pairs = valid_pairs[lows[valid_pairs[:, 1]] >= lows[valid_pairs[:, 0]]]

    # Exclude pairs where the second point is within exclude_end_points from the end
    valid_pairs = valid_pairs[valid_pairs[:, 1] < n - exclude_end_points]

    # Calculate slopes and y-intercepts for all valid pairs
    start_indices = valid_pairs[:, 0]
    end_indices = valid_pairs[:, 1]
    slopes = (lows[end_indices] - lows[start_indices]) / (end_indices - start_indices)
    y_intercepts = lows[start_indices] - slopes * start_indices

    # Vectorized validation of trendlines
    trendlines = []
    for i in prange(len(valid_pairs)):
        start_idx, end_idx = valid_pairs[i]
        slope = slopes[i]
        y_intercept = y_intercepts[i]
        if not is_invalid_trendline_vectorized(opens, highs, lows, closes, slope, y_intercept, n):
            trendlines.append((start_idx, lows[start_idx], end_idx, lows[end_idx], dates[start_idx], dates[end_idx]))

    # Filter trendlines with adjacent lows
    filtered_trendlines = []
    for start_idx, start_low, end_idx, end_low, start_date, end_date in trendlines:
        is_adjacent = False
        adjacent_trendlines = []

        # Check for adjacency with existing filtered trendlines
        for existing_start_idx, existing_start_low, existing_end_idx, existing_end_low, existing_start_date, existing_end_date in filtered_trendlines:
            if (abs(start_idx - existing_start_idx) <= adjacent_candles or
                    abs(end_idx - existing_end_idx) <= adjacent_candles):
                is_adjacent = True
                adjacent_trendlines.append(
                    (existing_start_idx, existing_start_low, existing_end_idx, existing_end_low, existing_start_date, existing_end_date)
                )

        if is_adjacent:
            # Replace if current trendline has lower lows
            if all(start_low <= atl[1] and end_low <= atl[3] for atl in adjacent_trendlines):
                filtered_trendlines = [
                    tl for tl in filtered_trendlines if tl not in adjacent_trendlines
                ]
                filtered_trendlines.append((start_idx, start_low, end_idx, end_low, start_date, end_date))
        else:
            filtered_trendlines.append((start_idx, start_low, end_idx, end_low, start_date, end_date))

    # Sort trendlines by start date (most recent first) and take up to 7
    filtered_trendlines.sort(key=lambda x: x[4], reverse=True)  # Sort by start_date
    filtered_trendlines = filtered_trendlines[:7]

    # Assign trendline data to the last row
    last_row_idx = n - 1
    if filtered_trendlines:
        for idx, (start_idx, start_low, end_idx, end_low, start_date, end_date) in enumerate(filtered_trendlines, 1):
            # Store as timezone-naive datetime objects
            df.loc[last_row_idx, f'TL_U_Support_{idx}_Start_Date'] = pd.Timestamp(start_date).tz_localize(None)
            df.loc[last_row_idx, f'TL_U_Support_{idx}_End_Date'] = pd.Timestamp(end_date).tz_localize(None)
            df.loc[last_row_idx, f'TL_U_Support_{idx}_Start_Price'] = start_low
            df.loc[last_row_idx, f'TL_U_Support_{idx}_End_Price'] = end_low

    return df