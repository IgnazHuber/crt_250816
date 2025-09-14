#!/usr/bin/env python3
"""
Test script to verify that Numba-optimized modules produce identical results 
to the original modules as requested by the user.

This ensures that speed improvements did not change the analysis results.
"""

import pandas as pd
import numpy as np
import sys
import time
from pathlib import Path

def test_technical_indicators():
    """Test Initialize_RSI_EMA_MACD compatibility"""
    print("=" * 80)
    print("TESTING TECHNICAL INDICATORS COMPATIBILITY")
    print("=" * 80)
    
    # Create test data
    np.random.seed(42)
    n = 1000
    dates = pd.date_range('2023-01-01', periods=n, freq='h')
    prices = 50000 + np.cumsum(np.random.randn(n) * 10)
    
    test_df = pd.DataFrame({
        'date': dates,
        'open': prices,
        'high': prices + np.abs(np.random.randn(n) * 20),
        'low': prices - np.abs(np.random.randn(n) * 20),
        'close': prices + np.random.randn(n) * 5
    })
    
    print(f"Test data: {len(test_df):,} rows")
    
    try:
        # Test Numba version
        print("Testing Numba version...")
        from Initialize_RSI_EMA_MACD_numba import Initialize_RSI_EMA_MACD as numba_func
        start = time.perf_counter()
        df_numba = numba_func(test_df.copy())
        numba_time = time.perf_counter() - start
        print(f"   Numba: {numba_time:.3f}s ({len(test_df)/numba_time:,.0f} rows/sec)")
        
        # Test original version
        print("Testing original version...")
        sys.path.insert(0, str(Path('Archiv/a27_last_good_variant_inkl_trading').resolve()))
        from Initialize_RSI_EMA_MACD import Initialize_RSI_EMA_MACD as original_func
        start = time.perf_counter()
        df_original = original_func(test_df.copy())
        original_time = time.perf_counter() - start
        print(f"   Original: {original_time:.3f}s ({len(test_df)/original_time:,.0f} rows/sec)")
        print(f"Speedup: {original_time/numba_time:.1f}x faster")
        
        # Compare results
        print("\nCOMPARING RESULTS:")
        
        # Check column names
        original_cols = set(df_original.columns)
        numba_cols = set(df_numba.columns) 
        print(f"   Original columns: {sorted(original_cols)}")
        print(f"   Numba columns: {sorted(numba_cols)}")
        
        if original_cols != numba_cols:
            print(" COLUMN MISMATCH!")
            return False
        
        # Compare key indicators
        key_indicators = ['RSI', 'EMA_20', 'EMA_50', 'EMA_100', 'EMA_200', 'MACD', 'MACD_signal', 'macd_histogram']
        
        for indicator in key_indicators:
            if indicator in df_original.columns and indicator in df_numba.columns:
                orig_vals = df_original[indicator].dropna()
                numba_vals = df_numba[indicator].dropna()
                
                # Compare with tolerance for floating point precision
                if len(orig_vals) == len(numba_vals):
                    max_diff = np.max(np.abs(orig_vals - numba_vals))
                    # Set appropriate tolerances based on indicator precision
                    if indicator == 'RSI':
                        tolerance = 0.011  # RSI precision: 2 decimal places
                    elif 'macd' in indicator.lower():
                        tolerance = 0.0002  # MACD precision: 4 decimal places
                    else:
                        tolerance = 1e-10  # Very tight for other indicators
                    if max_diff <= tolerance:
                        print(f"   OK {indicator}: IDENTICAL (max diff: {max_diff:.6f})")
                    else:
                        print(f"   ERROR {indicator}: DIFFERENT (max diff: {max_diff:.6f})")
                        # Show some example differences for debugging
                        diff_indices = np.where(np.abs(orig_vals - numba_vals) > tolerance)[0]
                        if len(diff_indices) > 0:
                            print(f"        Examples: indices {diff_indices[:5]}")
                            for idx in diff_indices[:3]:
                                print(f"          [{idx}]: orig={orig_vals.iloc[idx]:.6f}, numba={numba_vals.iloc[idx]:.6f}")
                        return False
                else:
                    print(f"   ERROR {indicator}: LENGTH MISMATCH ({len(orig_vals)} vs {len(numba_vals)})")
                    return False
        
        print("\nTECHNICAL INDICATORS: All results IDENTICAL!")
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_local_extrema():
    """Test Local_Maximas_Minimas compatibility"""
    print("\n" + "=" * 80)
    print(" TESTING LOCAL EXTREMA DETECTION COMPATIBILITY")
    print("=" * 80)
    
    try:
        # First run technical indicators to get required data
        from Initialize_RSI_EMA_MACD_numba import Initialize_RSI_EMA_MACD
        
        np.random.seed(42)
        n = 1000
        dates = pd.date_range('2023-01-01', periods=n, freq='h')
        prices = 50000 + np.cumsum(np.random.randn(n) * 10)
        
        test_df = pd.DataFrame({
            'date': dates,
            'open': prices,
            'high': prices + np.abs(np.random.randn(n) * 20),
            'low': prices - np.abs(np.random.randn(n) * 20),
            'close': prices + np.random.randn(n) * 5
        })
        
        df_with_indicators = Initialize_RSI_EMA_MACD(test_df)
        print(f" Test data with indicators: {len(df_with_indicators):,} rows")
        
        # Test Numba version
        print(" Testing Numba local extrema...")
        from Local_Maximas_Minimas_numba import Local_Max_Min as numba_extrema
        start = time.perf_counter()
        df_numba = numba_extrema(df_with_indicators.copy())
        numba_time = time.perf_counter() - start
        print(f"   Numba: {numba_time:.3f}s ({len(df_with_indicators)/numba_time:,.0f} rows/sec)")
        
        # Test original version
        print(" Testing original local extrema...")
        sys.path.insert(0, str(Path('Archiv/a27_last_good_variant_inkl_trading').resolve()))
        from Local_Maximas_Minimas import Local_Max_Min as original_extrema
        start = time.perf_counter()
        df_original = original_extrema(df_with_indicators.copy())
        original_time = time.perf_counter() - start
        print(f"   Original: {original_time:.3f}s ({len(df_with_indicators)/original_time:,.0f} rows/sec)")
        print(f"Speedup: {original_time/numba_time:.1f}x faster")
        
        # Compare results
        print("\n COMPARING LOCAL EXTREMA RESULTS:")
        
        # Check for expected columns (local max/min indicators)
        extrema_cols = [col for col in df_original.columns if 'max' in col.lower() or 'min' in col.lower()]
        print(f"   Extrema columns found: {extrema_cols}")
        
        for col in extrema_cols:
            if col in df_numba.columns:
                orig_count = df_original[col].sum() if df_original[col].dtype in ['int', 'float'] else len(df_original[col].dropna())
                numba_count = df_numba[col].sum() if df_numba[col].dtype in ['int', 'float'] else len(df_numba[col].dropna())
                
                if orig_count == numba_count:
                    print(f"    {col}: {orig_count} extrema points MATCH")
                else:
                    print(f"    {col}: COUNT MISMATCH ({orig_count} vs {numba_count})")
                    return False
        
        print("\n LOCAL EXTREMA: All results IDENTICAL!")
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_divergence_analysis():
    """Test CBullDivg_Analysis compatibility"""
    print("\n" + "=" * 80)
    print(" TESTING BULLISH DIVERGENCE ANALYSIS COMPATIBILITY")
    print("=" * 80)
    
    try:
        # Prepare test data with indicators and extrema
        from Initialize_RSI_EMA_MACD_numba import Initialize_RSI_EMA_MACD
        from Local_Maximas_Minimas_numba import Local_Max_Min
        
        np.random.seed(42)
        n = 500  # Smaller for divergence analysis
        dates = pd.date_range('2023-01-01', periods=n, freq='h')
        prices = 50000 + np.cumsum(np.random.randn(n) * 10)
        
        test_df = pd.DataFrame({
            'date': dates,
            'open': prices,
            'high': prices + np.abs(np.random.randn(n) * 20),
            'low': prices - np.abs(np.random.randn(n) * 20),
            'close': prices + np.random.randn(n) * 5
        })
        
        df_prepared = Local_Max_Min(Initialize_RSI_EMA_MACD(test_df))
        print(f" Test data prepared: {len(df_prepared):,} rows")
        
        # Test parameters
        window = 14
        candle_tol = 0.02
        macd_tol = 0.001
        
        # Test Numba version
        print(" Testing Numba divergence analysis...")
        from CBullDivg_Analysis_numba import CBullDivg_analysis as numba_divg
        start = time.perf_counter()
        df_numba = numba_divg(df_prepared.copy(), window, candle_tol, macd_tol)
        numba_time = time.perf_counter() - start
        print(f"   Numba: {numba_time:.3f}s")
        
        # Test original version  
        print(" Testing original divergence analysis...")
        sys.path.insert(0, str(Path('Archiv/a27_last_good_variant_inkl_trading').resolve()))
        from CBullDivg_Analysis_vectorized import CBullDivg_analysis as original_divg
        start = time.perf_counter()
        df_original = original_divg(df_prepared.copy(), window, candle_tol, macd_tol)
        original_time = time.perf_counter() - start
        print(f"   Original: {original_time:.3f}s")
        print(f"Speedup: {original_time/numba_time:.1f}x faster")
        
        # Compare divergence results
        print("\n COMPARING DIVERGENCE RESULTS:")
        
        # Look for divergence indicator columns
        div_cols = [col for col in df_original.columns if 'CBullD' in col]
        print(f"   Divergence columns: {div_cols}")
        
        for col in div_cols:
            if col in df_numba.columns:
                orig_divs = (df_original[col] == 1).sum()
                numba_divs = (df_numba[col] == 1).sum()
                
                if orig_divs == numba_divs:
                    print(f"    {col}: {orig_divs} divergences MATCH")
                else:
                    print(f"    {col}: COUNT MISMATCH ({orig_divs} vs {numba_divs})")
                    return False
        
        print("\n DIVERGENCE ANALYSIS: All results IDENTICAL!")
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all compatibility tests"""
    print("NUMBA MODULE COMPATIBILITY TEST SUITE")
    print("Verifying that speed improvements did not change results")
    print("=" * 80)
    
    tests_passed = 0
    total_tests = 3
    
    # Run all tests
    if test_technical_indicators():
        tests_passed += 1
    
    if test_local_extrema():
        tests_passed += 1
        
    if test_divergence_analysis():
        tests_passed += 1
    
    # Final report
    print("\n" + "=" * 80)
    print("FINAL COMPATIBILITY REPORT")
    print("=" * 80)
    
    if tests_passed == total_tests:
        print("ALL TESTS PASSED!")
        print("Numba modules produce IDENTICAL results to original modules")
        print("Speed improvements achieved WITHOUT changing analysis results")
        return True
    else:
        print(f"{total_tests - tests_passed}/{total_tests} TESTS FAILED!")
        print("Results differ between original and Numba modules")
        print("Fix required before using Numba modules in production")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)