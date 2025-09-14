#!/usr/bin/env python3
"""Debug RSI calculation differences between Numba and original implementations"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Create simple test data
np.random.seed(42)
n = 50
dates = pd.date_range('2023-01-01', periods=n, freq='h')  # Using 'h' instead of deprecated 'H'
prices = 50000 + np.cumsum(np.random.randn(n) * 10)

test_df = pd.DataFrame({
    'date': dates,
    'open': prices,
    'high': prices + np.abs(np.random.randn(n) * 20),
    'low': prices - np.abs(np.random.randn(n) * 20),
    'close': prices + np.random.randn(n) * 5
})

print("Testing RSI calculation with 50 data points...")

# Test Numba version
print("\n=== NUMBA VERSION ===")
from Initialize_RSI_EMA_MACD_numba import Initialize_RSI_EMA_MACD as numba_func
df_numba = numba_func(test_df.copy())
print("RSI values (first 25):")
print(df_numba['RSI'].head(25).values)
print(f"Non-zero RSI values: {(df_numba['RSI'] != 0).sum()}")
print(f"RSI range: {df_numba['RSI'].min():.2f} to {df_numba['RSI'].max():.2f}")

# Test original version
print("\n=== ORIGINAL VERSION ===")
sys.path.insert(0, str(Path('../claude_v1').resolve()))
from Initialize_RSI_EMA_MACD import Initialize_RSI_EMA_MACD as original_func
df_original = original_func(test_df.copy())
print("RSI values (first 25):")
print(df_original['RSI'].head(25).values)
print(f"Non-zero RSI values: {(df_original['RSI'] != 0).sum()}")
print(f"RSI range: {df_original['RSI'].min():.2f} to {df_original['RSI'].max():.2f}")

# Compare specific values
print("\n=== COMPARISON ===")
print("Index | Numba RSI | Original RSI | Difference")
print("------|-----------|--------------|----------")
for i in range(20, 25):  # Look at indices 20-24
    numba_val = df_numba['RSI'].iloc[i]
    orig_val = df_original['RSI'].iloc[i]
    diff = abs(numba_val - orig_val)
    print(f"{i:5d} | {numba_val:9.3f} | {orig_val:12.3f} | {diff:10.6f}")

# Check intermediate values too
print("\n=== INTERMEDIATE VALUES COMPARISON ===")
print("avg_gain, avg_loss values at index 20:")
print(f"Numba   - avg_gain: {df_numba['avg_gain'].iloc[20]:.6f}, avg_loss: {df_numba['avg_loss'].iloc[20]:.6f}")
print(f"Original - avg_gain: {df_original['avg_gain'].iloc[20]:.6f}, avg_loss: {df_original['avg_loss'].iloc[20]:.6f}")