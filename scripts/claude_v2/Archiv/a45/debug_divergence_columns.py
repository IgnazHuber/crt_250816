#!/usr/bin/env python3
"""Debug what columns are created by each divergence analysis implementation"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add path for accessing modules
sys.path.insert(0, str(Path('Archiv/a27_last_good_variant_inkl_trading').resolve()))

# Create test data
from Initialize_RSI_EMA_MACD import Initialize_RSI_EMA_MACD
from Local_Maximas_Minimas import Local_Max_Min

np.random.seed(42)
n = 100
dates = pd.date_range('2023-01-01', periods=n, freq='h')
prices = 50000 + np.cumsum(np.random.randn(n) * 10)

test_df = pd.DataFrame({
    'date': dates,
    'open': prices,
    'high': prices + np.abs(np.random.randn(n) * 20),
    'low': prices - np.abs(np.random.randn(n) * 20),
    'close': prices + np.random.randn(n) * 5
})

# Prepare data
df_with_indicators = Initialize_RSI_EMA_MACD(test_df)
df_prepared = Local_Max_Min(df_with_indicators)

print("Prepared data columns:")
divergence_prep_cols = [col for col in df_prepared.columns if any(x in col for x in ['LM_Low', 'RSI', 'macd'])]
print(f"Relevant columns: {divergence_prep_cols}")

# Test original implementation
print("\n=== ORIGINAL IMPLEMENTATION ===")
from CBullDivg_Analysis_vectorized import CBullDivg_analysis
df_orig = CBullDivg_analysis(df_prepared.copy(), 14, 2.0, 0.001)
orig_div_cols = [col for col in df_orig.columns if 'CBullD' in col]
print(f"Divergence columns created: {orig_div_cols}")

for col in orig_div_cols:
    if not col.endswith(('_Low', '_RSI', '_MACD', '_date', '_Gap')):
        count = (df_orig[col] == 1).sum()
        print(f"  {col}: {count} divergences")

# Test Numba implementation  
print("\n=== NUMBA IMPLEMENTATION ===")
from CBullDivg_Analysis_numba import CBullDivg_analysis
df_numba = CBullDivg_analysis(df_prepared.copy(), 14, 2.0, 0.001)
numba_div_cols = [col for col in df_numba.columns if 'CBullD' in col]
print(f"Divergence columns created: {numba_div_cols}")

for col in numba_div_cols:
    if not col.endswith(('_Low', '_RSI', '_MACD', '_date', '_Gap')):
        count = (df_numba[col] == 1).sum()
        print(f"  {col}: {count} divergences")

# Check for data differences
print("\n=== DATA COMPARISON ===")
print(f"LM_Low_window_2_CS non-zero count: {(df_prepared['LM_Low_window_2_CS'] != 0).sum()}")
print(f"LM_Low_window_1_CS non-zero count: {(df_prepared['LM_Low_window_1_CS'] != 0).sum()}")
print(f"LM_Low_window_2_MACD non-zero count: {(df_prepared['LM_Low_window_2_MACD'] != 0).sum()}")
print(f"LM_Low_window_1_MACD non-zero count: {(df_prepared['LM_Low_window_1_MACD'] != 0).sum()}")
print(f"MACD histogram negative count: {(df_prepared['macd_histogram'] < 0).sum()}")
print(f"RSI valid count: {(~df_prepared['RSI'].isna()).sum()}")