#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Simple test script to debug marker visibility issue

print("=== DEBUGGING MARKER VISIBILITY ===")
print()

# Check if modules exist
modules_to_check = [
    "Initialize_RSI_EMA_MACD.py",
    "Local_Maximas_Minimas.py", 
    "CBullDivg_Analysis_vectorized.py",
    "DivergenceArrows.py"
]

print("1. Checking module files:")
for module_file in modules_to_check:
    exists = os.path.exists(module_file)
    print(f"   {module_file}: {'OK' if exists else 'MISSING'}")

print()

# Check if we can import basic dependencies
print("2. Testing imports:")
try:
    import pandas as pd
    print("   pandas: OK")
except ImportError:
    print("   pandas: MISSING")

try:
    import numpy as np
    print("   numpy: OK")
except ImportError:
    print("   numpy: MISSING")

print()

# Check sample data
print("3. Checking sample data:")
data_files = [
    "../../results/doe_results.csv"
]

for data_file in data_files:
    if os.path.exists(data_file):
        try:
            # Try loading as CSV
            df = pd.read_csv(data_file)
            print(f"   {data_file}: OK ({len(df)} rows, {len(df.columns)} columns)")
            print(f"   Sample columns: {list(df.columns)[:10]}")
            
            # Check for required columns
            required_cols = ['date', 'open', 'high', 'low', 'close']
            missing = [col for col in required_cols if col not in df.columns]
            if missing:
                print(f"   WARNING: Missing required columns: {missing}")
            else:
                print("   All required OHLC columns present")
                
            # Check for divergence columns that indicate analysis has been run
            divergence_cols = [col for col in df.columns if 'CBullD' in col]
            if divergence_cols:
                print(f"   Found {len(divergence_cols)} divergence columns")
                # Check if any divergences exist
                for col in ['CBullD_gen', 'CBullD_neg_MACD']:
                    if col in df.columns:
                        count = (df[col] == 1).sum() if col in df.columns else 0
                        print(f"     {col}: {count} signals")
            else:
                print("   No divergence analysis columns found - analysis may be needed")
                
        except Exception as e:
            print(f"   {data_file}: ERROR loading - {e}")
    else:
        print(f"   {data_file}: File not found")

print()
print("=== DEBUG COMPLETE ===")