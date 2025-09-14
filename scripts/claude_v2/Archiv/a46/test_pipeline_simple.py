#!/usr/bin/env python3
"""
Simple test of the analysis pipeline to debug the issue
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Create simple test data
def create_simple_test_data():
    dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(100)]
    np.random.seed(42)
    
    close_prices = 50000 + np.cumsum(np.random.normal(0, 100, 100))
    
    df = pd.DataFrame({
        'date': dates,
        'open': close_prices + np.random.normal(0, 10, 100),
        'high': close_prices + np.abs(np.random.normal(0, 20, 100)),
        'low': close_prices - np.abs(np.random.normal(0, 20, 100)),
        'close': close_prices,
        'volume': np.random.uniform(1000, 2000, 100)
    })
    
    return df

def test_pipeline():
    print("Testing analysis pipeline...")
    
    df = create_simple_test_data()
    print(f"1. Created test data: {len(df)} rows")
    print(f"   Columns: {list(df.columns)}")
    
    try:
        from Initialize_RSI_EMA_MACD import Initialize_RSI_EMA_MACD
        df_step1 = Initialize_RSI_EMA_MACD(df)
        
        if df_step1 is None:
            print("❌ Step 1 returned None")
            return
            
        print(f"2. After indicators: {len(df_step1.columns)} columns")
        print(f"   New columns: {[col for col in df_step1.columns if col not in df.columns]}")
        
        # Check if required columns exist
        required_cols = ['RSI', 'macd_histogram']
        for col in required_cols:
            if col in df_step1.columns:
                print(f"   ✅ {col}: {df_step1[col].dropna().shape[0]} valid values")
            else:
                print(f"   ❌ {col}: MISSING")
        
        from Local_Maximas_Minimas import Local_Max_Min
        df_step2 = Local_Max_Min(df_step1)
        
        if df_step2 is None:
            print("❌ Step 2 returned None")
            return
            
        print(f"3. After extrema: {len(df_step2.columns)} columns")
        
        # Check MACD columns
        macd_cols = [col for col in df_step2.columns if 'MACD' in col]
        print(f"   MACD columns: {macd_cols}")
        
        from CBullDivg_Analysis_vectorized import CBullDivg_analysis
        df_step3 = CBullDivg_analysis(df_step2, 5, 0.1, 3.25)
        
        if df_step3 is None:
            print("❌ Step 3 returned None")
            return
            
        print(f"4. After divergence analysis: {len(df_step3.columns)} columns")
        
        # Count markers
        marker_cols = [col for col in df_step3.columns if 'CBullD' in col and col.endswith(('_1', '_2', '_gen'))]
        for col in marker_cols:
            count = (df_step3[col] == 1).sum()
            print(f"   {col}: {count} markers")
            
        print("✅ Pipeline completed successfully!")
        
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_pipeline()