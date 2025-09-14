#!/usr/bin/env python3
"""
Debug the analysis pipeline step by step to find where it's failing
"""

import pandas as pd
import numpy as np
import sys
from datetime import datetime, timedelta

def create_test_data():
    """Create the same test data as the web server test"""
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

def test_step1_data_loading():
    """Step 1: Test data creation"""
    print("=" * 80)
    print("STEP 1: DATA LOADING")
    print("=" * 80)
    
    try:
        df = create_test_data()
        print(f"✅ Created DataFrame with {len(df)} rows")
        print(f"   Columns: {list(df.columns)}")
        print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"   Price range: {df['close'].min():.2f} to {df['close'].max():.2f}")
        print(f"   Data types: {dict(df.dtypes)}")
        
        # Check for any NaN values
        nan_counts = df.isnull().sum()
        if nan_counts.sum() > 0:
            print(f"⚠️  NaN values found: {dict(nan_counts[nan_counts > 0])}")
        else:
            print("✅ No NaN values in data")
            
        return df
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_step2_technical_indicators(df):
    """Step 2: Test technical indicators calculation"""
    print("\\n" + "=" * 80)
    print("STEP 2: TECHNICAL INDICATORS")
    print("=" * 80)
    
    try:
        from Initialize_RSI_EMA_MACD_numba import Initialize_RSI_EMA_MACD
        print("✅ Imported Initialize_RSI_EMA_MACD_numba")
        
        df_with_indicators = Initialize_RSI_EMA_MACD(df.copy())
        
        if df_with_indicators is None:
            print("❌ Technical indicators returned None")
            return None
            
        print(f"✅ Technical indicators calculated")
        print(f"   Input columns: {len(df.columns)}")
        print(f"   Output columns: {len(df_with_indicators.columns)}")
        
        # Check key indicators
        key_indicators = ['RSI', 'EMA_20', 'EMA_50', 'macd', 'macd_histogram']
        for indicator in key_indicators:
            if indicator in df_with_indicators.columns:
                values = df_with_indicators[indicator].dropna()
                print(f"   {indicator}: {len(values)} valid values, range: {values.min():.3f} to {values.max():.3f}")
            else:
                print(f"   ❌ {indicator}: MISSING")
                
        return df_with_indicators
    except Exception as e:
        print(f"❌ Technical indicators failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_step3_local_extrema(df_with_indicators):
    """Step 3: Test local extrema detection"""
    print("\\n" + "=" * 80)
    print("STEP 3: LOCAL EXTREMA DETECTION")
    print("=" * 80)
    
    try:
        from Local_Maximas_Minimas_numba import Local_Max_Min
        print("✅ Imported Local_Maximas_Minimas_numba")
        
        df_with_extrema = Local_Max_Min(df_with_indicators.copy())
        
        if df_with_extrema is None:
            print("❌ Local extrema detection returned None")
            return None
            
        print(f"✅ Local extrema detection completed")
        
        # Check extrema columns
        extrema_columns = [col for col in df_with_extrema.columns if 'LM_' in col]
        print(f"   Found {len(extrema_columns)} extrema columns")
        
        for col in extrema_columns:
            non_zero_count = (df_with_extrema[col] != 0).sum()
            print(f"   {col}: {non_zero_count} extrema points")
            
        total_extrema = sum((df_with_extrema[col] != 0).sum() for col in extrema_columns)
        print(f"   Total extrema points: {total_extrema}")
        
        if total_extrema == 0:
            print("❌ No local extrema found - this will prevent divergence detection")
        
        return df_with_extrema
    except Exception as e:
        print(f"❌ Local extrema detection failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_step4_divergence_analysis(df_prepared):
    """Step 4: Test divergence analysis"""
    print("\\n" + "=" * 80)
    print("STEP 4: DIVERGENCE ANALYSIS")
    print("=" * 80)
    
    try:
        from CBullDivg_Analysis_numba import CBullDivg_analysis
        print("✅ Imported CBullDivg_Analysis_numba")
        
        # Test with different tolerance values
        test_params = [
            (20, 0.5, 0.001),
            (20, 2.0, 0.001),
            (20, 8.0, 0.001),
        ]
        
        for window, candle_tol, macd_tol in test_params:
            print(f"\\n   Testing: window={window}, candle_tol={candle_tol}%, macd_tol={macd_tol}")
            
            df_result = CBullDivg_analysis(df_prepared.copy(), window, candle_tol, macd_tol)
            
            if df_result is None:
                print(f"     ❌ Divergence analysis returned None")
                continue
                
            # Count divergence markers
            divergence_cols = [col for col in df_result.columns if 'CBullD' in col and not col.endswith(('_Low', '_RSI', '_MACD', '_date', '_Gap'))]
            
            total_markers = 0
            for col in divergence_cols:
                if col in df_result.columns:
                    markers = (df_result[col] == 1).sum()
                    total_markers += markers
                    if markers > 0:
                        print(f"     ✅ {col}: {markers} markers")
                    else:
                        print(f"     🔶 {col}: {markers} markers")
            
            print(f"     Total markers: {total_markers}")
            
            if total_markers > 0:
                print(f"     ✅ SUCCESS: Found {total_markers} divergence markers!")
                return df_result
        
        print("❌ No divergence markers found with any parameters")
        return df_result if 'df_result' in locals() else None
        
    except Exception as e:
        print(f"❌ Divergence analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_step5_data_requirements():
    """Step 5: Check data requirements for divergence detection"""
    print("\\n" + "=" * 80)
    print("STEP 5: DATA REQUIREMENTS CHECK")
    print("=" * 80)
    
    print("Minimum requirements for bullish divergence detection:")
    print("1. Sufficient price movement (highs and lows)")
    print("2. Valid RSI values (not all same)")
    print("3. Valid MACD histogram values")
    print("4. Local extrema points detected")
    print("5. Proper time series length (>= analysis window)")
    
    df = create_test_data()
    print(f"\\nData characteristics:")
    print(f"   Length: {len(df)} points")
    print(f"   Price volatility: {df['close'].std():.2f}")
    print(f"   Price trend: {df['close'].iloc[-1] - df['close'].iloc[0]:.2f}")
    
    # Test with minimal analysis to see what happens
    try:
        from Initialize_RSI_EMA_MACD_numba import Initialize_RSI_EMA_MACD
        from Local_Maximas_Minimas_numba import Local_Max_Min
        from CBullDivg_Analysis_numba import CBullDivg_analysis
        
        print("\\n   Running minimal pipeline...")
        df_step1 = Initialize_RSI_EMA_MACD(df.copy())
        print(f"   After indicators: {len(df_step1.columns) if df_step1 is not None else 'None'} columns")
        
        if df_step1 is not None:
            df_step2 = Local_Max_Min(df_step1.copy())
            print(f"   After extrema: {len(df_step2.columns) if df_step2 is not None else 'None'} columns")
            
            if df_step2 is not None:
                df_step3 = CBullDivg_analysis(df_step2.copy(), 10, 5.0, 0.01)  # Very permissive parameters
                
                if df_step3 is not None:
                    div_cols = [col for col in df_step3.columns if 'CBullD' in col and not col.endswith(('_Low', '_RSI', '_MACD', '_date', '_Gap'))]
                    total = sum((df_step3[col] == 1).sum() for col in div_cols)
                    print(f"   After divergence (permissive): {total} markers found")
                    
                    if total > 0:
                        print("   ✅ Pipeline CAN find markers with permissive parameters")
                    else:
                        print("   ❌ Pipeline finds NO markers even with permissive parameters")
        
    except Exception as e:
        print(f"   ❌ Minimal pipeline failed: {e}")

def main():
    """Run complete analysis pipeline debug"""
    print("🔍 ANALYSIS PIPELINE DIAGNOSTIC")
    print("This will test each step of the analysis to find where it fails")
    
    # Step 1: Data loading
    df = test_step1_data_loading()
    if df is None:
        return False
    
    # Step 2: Technical indicators
    df_with_indicators = test_step2_technical_indicators(df)
    if df_with_indicators is None:
        return False
    
    # Step 3: Local extrema
    df_with_extrema = test_step3_local_extrema(df_with_indicators)
    if df_with_extrema is None:
        return False
    
    # Step 4: Divergence analysis
    df_result = test_step4_divergence_analysis(df_with_extrema)
    
    # Step 5: Data requirements
    test_step5_data_requirements()
    
    print("\\n" + "=" * 80)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 80)
    
    if df_result is None:
        print("❌ Analysis pipeline broken - check error messages above")
        return False
    
    # Final marker count
    div_cols = [col for col in df_result.columns if 'CBullD' in col and not col.endswith(('_Low', '_RSI', '_MACD', '_date', '_Gap'))]
    total_markers = sum((df_result[col] == 1).sum() for col in div_cols if col in df_result.columns)
    
    if total_markers > 0:
        print(f"✅ SUCCESS: Pipeline working - found {total_markers} markers")
        print("   The web server issue might be:")
        print("   • Different parameters being used")
        print("   • Different data format")
        print("   • API response formatting")
        return True
    else:
        print("❌ FAILURE: Pipeline runs but finds no markers")
        print("   Possible issues:")
        print("   • Data doesn't have sufficient volatility for divergences")
        print("   • Analysis parameters too strict")
        print("   • Logic errors in divergence detection")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)