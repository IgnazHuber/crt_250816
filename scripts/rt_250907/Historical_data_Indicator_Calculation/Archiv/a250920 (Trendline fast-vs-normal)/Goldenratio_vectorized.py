import numpy as np
import pandas as pd

def calculate_golden_ratios(df):
    # Convert columns to NumPy arrays for efficiency
    highs = df['high'].values
    lows = df['low'].values
    dates = df['date'].values
    maxima_mask = df['LM_High_window_2_CS'].values > 0
    minima_mask = df['LM_Low_window_1_CS'].values > 0
    n = len(df)
    last_idx = n - 1

    # ---------- HELPER FUNCTION: FILTER MINIMA ----------
    def filter_minimas(start_idx, end_idx):
        minima_indices = np.where(minima_mask[start_idx:end_idx + 1])[0] + start_idx
        if len(minima_indices) == 0:
            return np.array([])

        sorted_indices = minima_indices[np.argsort(dates[minima_indices])]
        filtered_indices = []
        previous_low = None

        for idx in sorted_indices[::-1]:  # Process from latest to earliest
            current_low = lows[idx]
            if previous_low is None or current_low < previous_low:
                filtered_indices.append(idx)
                previous_low = current_low

        return np.array(filtered_indices)[np.argsort(filtered_indices)]

    # ---------- REPLICATE ORIGINAL LOGIC ----------
    reference_high = None
    current_retracements_50 = []
    current_retracements_618 = []
    previous_maxima_idx = None
    previous_highest_high = None
    processed_maxima = set()

    for end_idx in range(2, last_idx + 1):
        # Identify new maxima up to end_idx
        sub_indices = np.arange(end_idx + 1)
        new_maxima_mask = maxima_mask[:end_idx + 1] & ~np.isin(sub_indices, list(processed_maxima))
        new_maxima_indices = np.where(new_maxima_mask)[0]

        for max_idx in new_maxima_indices:
            processed_maxima.add(max_idx)
            current_highest_high = highs[max_idx]

            # Determine minima range and retracement initialization
            if reference_high is None or current_highest_high >= reference_high:
                minima_start_idx = 0
                new_retracements_50 = []
                new_retracements_618 = []
                reference_high = current_highest_high
            else:
                minima_start_idx = previous_maxima_idx if previous_maxima_idx is not None else 0
                new_retracements_50 = current_retracements_50.copy()
                new_retracements_618 = current_retracements_618.copy()

            # Filter minima up to max_idx
            filtered_minima = filter_minimas(minima_start_idx, max_idx)

            # Calculate retracements for each minimum
            for min_idx in filtered_minima:
                if min_idx >= max_idx:
                    continue
                previous_min = lows[min_idx]
                r50 = previous_min + (1 - 0.5) * (current_highest_high - previous_min)
                r618 = previous_min + (1 - 0.618) * (current_highest_high - previous_min)
                if r50 not in new_retracements_50:
                    new_retracements_50.append(r50)
                if r618 not in new_retracements_618:
                    new_retracements_618.append(r618)

            # Sort retracements in descending order
            new_retracements_50.sort(reverse=True)
            new_retracements_618.sort(reverse=True)

            # Update state
            current_retracements_50 = new_retracements_50
            current_retracements_618 = new_retracements_618
            previous_maxima_idx = max_idx
            previous_highest_high = current_highest_high

        # Check for higher highs after the last maxima
        if previous_maxima_idx is not None and end_idx > previous_maxima_idx:
            current_high = highs[end_idx]
            if (previous_highest_high is not None and
                    current_high > previous_highest_high and
                    (reference_high is None or current_high > reference_high)):
                current_highest_idx = end_idx
                current_highest_high = current_high
                reference_high = current_highest_high

                # Use all minima up to current_highest_idx
                filtered_minima = filter_minimas(0, current_highest_idx)

                # Reset retracements
                current_retracements_50 = []
                current_retracements_618 = []

                # Calculate new retracements
                for min_idx in filtered_minima:
                    if min_idx >= current_highest_idx:
                        continue
                    previous_min = lows[min_idx]
                    r50 = previous_min + (1 - 0.5) * (current_highest_high - previous_min)
                    r618 = previous_min + (1 - 0.618) * (current_highest_high - previous_min)
                    if r50 not in current_retracements_50:
                        current_retracements_50.append(r50)
                    if r618 not in current_retracements_618:
                        current_retracements_618.append(r618)

                # Sort retracements
                current_retracements_50.sort(reverse=True)
                current_retracements_618.sort(reverse=True)

                previous_highest_high = current_highest_high

        # Validate retracements for the current candle
        if current_retracements_50:
            current_low = lows[end_idx]
            valid_retracements_50 = []
            valid_retracements_618 = []
            for r50, r618 in zip(current_retracements_50, current_retracements_618):
                if r50 <= current_low:
                    valid_retracements_50.append(r50)
                    valid_retracements_618.append(r618)

            # Update current retracements
            current_retracements_50 = valid_retracements_50
            current_retracements_618 = valid_retracements_618

            # Reset reference_high if all retracements are violated
            if not current_retracements_50 and reference_high is not None:
                reference_high = None

    # ---------- ADD RETRACEMENTS TO DATAFRAME ----------
    for i in range(1, 26):
        df[f'Fib_50_{i}'] = 0.0
        df[f'Fib_618_{i}'] = 0.0

    # Assign retracements to the last row
    for i, (r50, r618) in enumerate(zip(current_retracements_50[:25], current_retracements_618[:25]), 1):
        df.at[last_idx, f'Fib_50_{i}'] = r50
        df.at[last_idx, f'Fib_618_{i}'] = r618

    return df