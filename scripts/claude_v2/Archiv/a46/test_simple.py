#!/usr/bin/env python3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Create simple test data
dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(50)]
close_prices = 50000 + np.random.normal(0, 100, 50)

df = pd.DataFrame({
    'date': dates,
    'open': close_prices,
    'high': close_prices + 100,
    'low': close_prices - 100,
    'close': close_prices,
    'volume': [1000] * 50
})

print("Step 1: Test data created")

try:
    from Initialize_RSI_EMA_MACD import Initialize_RSI_EMA_MACD
    df1 = Initialize_RSI_EMA_MACD(df)
    print(f"Step 2: Indicators added, columns: {len(df1.columns)}")
    
    # Check for MACD column
    if 'macd_histogram' in df1.columns:
        print("macd_histogram column exists")
    else:
        print("ERROR: macd_histogram column missing")
        print("Available columns:", list(df1.columns))
    
    from Local_Maximas_Minimas import Local_Max_Min  
    df2 = Local_Max_Min(df1)
    print(f"Step 3: Extrema added, columns: {len(df2.columns)}")
    
    from CBullDivg_Analysis_vectorized import CBullDivg_analysis
    df3 = CBullDivg_analysis(df2, 5, 0.1, 3.25)
    print("Step 4: Divergence analysis completed")
    print("SUCCESS: Pipeline works!")
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()