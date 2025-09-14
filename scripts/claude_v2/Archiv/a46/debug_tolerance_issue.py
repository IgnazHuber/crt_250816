#!/usr/bin/env python3
"""
Debug why 0.5% tolerance works but higher tolerances fail
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def create_test_data():
    """Create the same test data as used in web server test"""
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(hours=i) for i in range(200)]
    
    np.random.seed(42)  # Same seed as web test
    
    # Base price trend (decline then recovery - perfect for divergences)
    base_prices = np.concatenate([
        np.linspace(50000, 48000, 100),  # Decline phase
        np.linspace(48000, 49500, 100)   # Recovery phase
    ])
    
    volatility = np.random.normal(0, 50, 200)
    close_prices = base_prices + volatility
    
    high_prices = close_prices + np.abs(np.random.normal(0, 20, 200))
    low_prices = close_prices - np.abs(np.random.normal(0, 20, 200))
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = close_prices[0]
    
    volume = np.random.uniform(800, 1200, 200)
    
    df = pd.DataFrame({
        'date': dates,
        'open': open_prices,
        'high': high_prices, 
        'low': low_prices,
        'close': close_prices,
        'volume': volume
    })
    
    return df

def test_tolerance_values():
    """Test different tolerance values to understand the issue"""
    print("=" * 80)
    print("DEBUGGING TOLERANCE ISSUE")
    print("=" * 80)
    
    try:
        from Initialize_RSI_EMA_MACD_numba import Initialize_RSI_EMA_MACD
        from Local_Maximas_Minimas_numba import Local_Max_Min
        from CBullDivg_Analysis_numba import CBullDivg_analysis
        
        # Create test data
        df = create_test_data()
        print(f"Created {len(df)} data points")
        
        # Run pipeline
        df_step1 = Initialize_RSI_EMA_MACD(df.copy())
        print(f"After indicators: {len(df_step1.columns)} columns")
        
        df_step2 = Local_Max_Min(df_step1.copy())
        print(f"After extrema: {len(df_step2.columns)} columns")
        
        # Test different tolerance values
        tolerances = [0.5, 1.0, 2.0, 4.0, 8.0]
        
        print(f"\nTesting tolerance values:")
        for candle_tol in tolerances:
            df_result = CBullDivg_analysis(df_step2.copy(), 20, candle_tol, 0.001)
            
            # Count markers
            cbull_1_count = (df_result['CBullD_1'] == 1).sum()
            cbull_2_count = (df_result['CBullD_2'] == 1).sum()
            total_markers = cbull_1_count + cbull_2_count
            
            print(f"   Candle% {candle_tol:4.1f}: {total_markers:2d} markers (CBullD_1: {cbull_1_count}, CBullD_2: {cbull_2_count})")
            
            # Debug first few successful conditions for 0.5% vs 1.0%
            if candle_tol in [0.5, 1.0] and total_markers > 0:
                print(f"      First marker indices: {df_result[df_result['CBullD_1'] == 1].index.tolist()[:3]}")
                print(f"      First marker indices: {df_result[df_result['CBullD_2'] == 1].index.tolist()[:3]}")
        
        # Test boundary conditions around 0.5% to find the exact threshold
        print(f"\nDetailed threshold analysis:")
        fine_tolerances = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.5]
        for candle_tol in fine_tolerances:
            df_result = CBullDivg_analysis(df_step2.copy(), 20, candle_tol, 0.001)
            cbull_1_count = (df_result['CBullD_1'] == 1).sum()
            cbull_2_count = (df_result['CBullD_2'] == 1).sum()
            total_markers = cbull_1_count + cbull_2_count
            print(f"   Candle% {candle_tol:4.1f}: {total_markers:2d} markers")
            
        # Examine actual price differences in the data to understand threshold
        print(f"\nAnalyzing price movements in the data:")
        for i in range(20, min(50, len(df_step2))):
            if df_step2.loc[i, 'LM_Low_window_2_CS'] != 0:
                for j in range(i-20, i):
                    if df_step2.loc[j, 'LM_Low_window_1_CS'] != 0:
                        price_diff = df_step2.loc[i, 'LM_Low_window_2_CS'] - df_step2.loc[j, 'LM_Low_window_1_CS']
                        price_diff_percent = abs(100 * price_diff / df_step2.loc[j, 'LM_Low_window_1_CS'])
                        print(f"   i={i}, j={j}: price_diff_percent={price_diff_percent:.2f}%")
                        break
                        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_tolerance_values()