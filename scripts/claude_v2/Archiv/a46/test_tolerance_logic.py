#!/usr/bin/env python3
"""Test tolerance logic behavior - increasing tolerance should find MORE markers"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add path for accessing modules
sys.path.insert(0, str(Path('Archiv/a27_last_good_variant_inkl_trading').resolve()))

def test_tolerance_behavior():
    """Test that increasing Candle% finds MORE markers (correct behavior)"""
    print("=" * 80)
    print("TESTING TOLERANCE LOGIC - CANDLE% BEHAVIOR")
    print("=" * 80)
    
    # Create test data with sufficient indicators and extrema
    from Initialize_RSI_EMA_MACD import Initialize_RSI_EMA_MACD
    from Local_Maximas_Minimas import Local_Max_Min
    
    np.random.seed(42)  # For reproducible results
    n = 200
    dates = pd.date_range('2023-01-01', periods=n, freq='h')
    prices = 50000 + np.cumsum(np.random.randn(n) * 10)
    
    test_df = pd.DataFrame({
        'date': dates,
        'open': prices,
        'high': prices + np.abs(np.random.randn(n) * 20),
        'low': prices - np.abs(np.random.randn(n) * 20),
        'close': prices + np.random.randn(n) * 5
    })
    
    # Prepare data with indicators and extrema
    df_with_indicators = Initialize_RSI_EMA_MACD(test_df)
    df_prepared = Local_Max_Min(df_with_indicators)
    
    print(f"Test data: {len(df_prepared)} rows")
    print(f"Available columns: {list(df_prepared.columns)}")
    
    # Test parameters
    window = 14
    macd_tol = 0.001
    
    # Test with different Candle% tolerances
    candle_tolerances = [0.5, 1.0, 2.0, 4.0, 8.0]  # Increasing tolerance
    
    print("\\nTesting Original Archived Implementation:")
    print("Candle% | CBullD_gen | CBullD_neg_MACD | Total")
    print("--------|------------|-----------------|------")
    
    original_results = []
    
    for candle_tol in candle_tolerances:
        from CBullDivg_Analysis_vectorized import CBullDivg_analysis
        df_result = CBullDivg_analysis(df_prepared.copy(), window, candle_tol, macd_tol)
        
        # Check what columns are available and use appropriate names
        gen_count = (df_result['CBullD_1'] == 1).sum() if 'CBullD_1' in df_result.columns else 0
        neg_macd_count = (df_result['CBullD_neg_MACD'] == 1).sum() if 'CBullD_neg_MACD' in df_result.columns else 0
        total_count = gen_count + neg_macd_count
        
        print(f"{candle_tol:7.1f} | {gen_count:10d} | {neg_macd_count:15d} | {total_count:5d}")
        original_results.append((candle_tol, gen_count, neg_macd_count, total_count))
    
    print("\\nTesting Current Numba Implementation:")
    print("Candle% | CBullD_gen | CBullD_neg_MACD | Total")
    print("--------|------------|-----------------|------")
    
    numba_results = []
    
    for candle_tol in candle_tolerances:
        from CBullDivg_Analysis_numba import CBullDivg_analysis
        df_result = CBullDivg_analysis(df_prepared.copy(), window, candle_tol, macd_tol)
        
        # Check what columns are available and use appropriate names
        gen_count = (df_result['CBullD_gen'] == 1).sum() if 'CBullD_gen' in df_result.columns else 0
        neg_macd_count = (df_result['CBullD_neg_MACD'] == 1).sum() if 'CBullD_neg_MACD' in df_result.columns else 0
        total_count = gen_count + neg_macd_count
        
        print(f"{candle_tol:7.1f} | {gen_count:10d} | {neg_macd_count:15d} | {total_count:5d}")
        numba_results.append((candle_tol, gen_count, neg_macd_count, total_count))
    
    # Analyze behavior
    print("\\n" + "=" * 80)
    print("BEHAVIOR ANALYSIS")
    print("=" * 80)
    
    print("\\nOriginal Implementation:")
    for i in range(1, len(original_results)):
        prev_total = original_results[i-1][3]
        curr_total = original_results[i][3]
        change = curr_total - prev_total
        direction = "INCREASE" if change > 0 else "DECREASE" if change < 0 else "SAME"
        print(f"  {original_results[i-1][0]:.1f}% -> {original_results[i][0]:.1f}%: {prev_total} -> {curr_total} ({direction})")
    
    print("\\nNumba Implementation:")
    for i in range(1, len(numba_results)):
        prev_total = numba_results[i-1][3]
        curr_total = numba_results[i][3]
        change = curr_total - prev_total
        direction = "INCREASE" if change > 0 else "DECREASE" if change < 0 else "SAME"
        print(f"  {numba_results[i-1][0]:.1f}% -> {numba_results[i][0]:.1f}%: {prev_total} -> {curr_total} ({direction})")
    
    # Check if behavior is correct (should generally increase with tolerance)
    print("\\n" + "=" * 80)
    print("RESULT:")
    
    original_increasing = all(original_results[i][3] >= original_results[i-1][3] for i in range(1, len(original_results)))
    numba_increasing = all(numba_results[i][3] >= numba_results[i-1][3] for i in range(1, len(numba_results)))
    
    print(f"Original: {'CORRECT' if original_increasing else 'INCORRECT'} (markers should increase/stay same with higher tolerance)")
    print(f"Numba:    {'CORRECT' if numba_increasing else 'INCORRECT'} (markers should increase/stay same with higher tolerance)")
    
    if numba_increasing:
        print("\\nSUCCESS: Tolerance logic fixed! Higher Candle% now finds more markers.")
    else:
        print("\\nFAILED: Tolerance logic still incorrect. Higher Candle% should find more markers.")
    
    return numba_increasing

if __name__ == "__main__":
    success = test_tolerance_behavior()
    sys.exit(0 if success else 1)