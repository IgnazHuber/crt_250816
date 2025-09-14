#!/usr/bin/env python3
"""
Compare server execution vs local execution to find the difference
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def create_test_data():
    """Create the same test data as used in web server test"""
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(hours=i) for i in range(200)]
    
    np.random.seed(42)  # Same seed as web test
    
    base_prices = np.concatenate([
        np.linspace(50000, 48000, 100),
        np.linspace(48000, 49500, 100)
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

def simulate_server_execution():
    """Simulate exact server execution path"""
    print("=" * 80)
    print("SIMULATING SERVER EXECUTION PATH")
    print("=" * 80)
    
    try:
        from Initialize_RSI_EMA_MACD_numba import Initialize_RSI_EMA_MACD
        from Local_Maximas_Minimas_numba import Local_Max_Min
        from CBullDivg_Analysis_numba import CBullDivg_analysis
        
        # Create test data
        df = create_test_data()
        print(f"Created {len(df)} data points")
        
        # Step 1: Initialize technical indicators (same as server)
        print("\nStep 1: Technical indicators...")
        df_with_indicators = Initialize_RSI_EMA_MACD(df)
        print(f"  Result: {len(df_with_indicators.columns)} columns")
        
        # Step 2: Calculate local maxima and minima (same as server) 
        print("\nStep 2: Local extrema...")
        df_with_extrema = Local_Max_Min(df_with_indicators)
        print(f"  Result: {len(df_with_extrema.columns)} columns")
        
        # Step 3: Simulate server variant processing 
        print("\nStep 3: Variant analysis (simulating server loop)...")
        
        # Test the exact same parameters as the server test
        variants = [
            {'window': 20, 'candleTol': 0.5, 'macdTol': 0.001},
            {'window': 20, 'candleTol': 1.0, 'macdTol': 0.001},
            {'window': 20, 'candleTol': 2.0, 'macdTol': 0.001},
        ]
        
        for i, variant in enumerate(variants):
            print(f"\n  Variant {i+1}: Candle% = {variant['candleTol']}")
            
            # Copy dataframe (same as server)
            df_analysis = CBullDivg_analysis(
                df_with_extrema.copy(),
                variant['window'],
                variant['candleTol'],
                variant['macdTol']
            )
            
            # Count markers (same logic as server)
            hidden_mask = df_analysis['CBullD_1'] == 1
            classic_mask = df_analysis['CBullD_2'] == 1
            
            hidden_count = hidden_mask.sum()
            classic_count = classic_mask.sum()
            total_count = hidden_count + classic_count
            
            print(f"    Hidden divergences: {hidden_count}")
            print(f"    Classic divergences: {classic_count}")
            print(f"    Total markers: {total_count}")
            
            # Debug: Show first few markers
            if total_count > 0:
                hidden_indices = df_analysis[hidden_mask].index.tolist()[:3]
                classic_indices = df_analysis[classic_mask].index.tolist()[:3]
                print(f"    First hidden indices: {hidden_indices}")
                print(f"    First classic indices: {classic_indices}")
        
        print(f"\n" + "=" * 80)
        print("COMPARISON WITH EXPECTED RESULTS")
        print("=" * 80)
        print("Expected (from local debug): 23 markers at ALL tolerance levels")
        print("Server reports: 10 markers at 0.5%, 0 markers at 1.0%+")
        print("This test shows: [results above]")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    simulate_server_execution()