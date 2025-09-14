#!/usr/bin/env python3
"""
Performance Benchmark Script
Compare original vs Numba-optimized functions with your actual data
"""

import pandas as pd
import numpy as np
import time
import logging
import argparse
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def benchmark_with_actual_data(data_file: str):
    """Benchmark using your actual trading data"""
    try:
        logger.info(f"📊 Loading data from: {data_file}")
        
        # Load your actual data
        if data_file.endswith('.parquet'):
            df = pd.read_parquet(data_file)
        elif data_file.endswith('.csv'):
            df = pd.read_csv(data_file)
        else:
            raise ValueError("Unsupported file format. Use .parquet or .csv")
        
        logger.info(f"📈 Dataset: {len(df):,} rows, {len(df.columns)} columns")
        
        # Ensure required columns exist
        required_cols = ['date', 'open', 'high', 'low', 'close']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        return df
        
    except Exception as e:
        logger.error(f"❌ Failed to load data: {e}")
        return None

def run_benchmark(df: pd.DataFrame):
    """Run performance comparison between original and Numba functions"""
    
    print("\n" + "=" * 80)
    print("🚀 PERFORMANCE BENCHMARK - ORIGINAL vs NUMBA")
    print("=" * 80)
    print(f"📊 Dataset: {len(df):,} rows")
    print("=" * 80)
    
    # Test parameters for analysis
    test_params = {
        'window': 5,
        'candle_tol': 0.1,
        'macd_tol': 3.25
    }
    
    results = {}
    
    # Test 1: Technical Indicators
    print("🔬 Testing Technical Indicators (RSI, EMA, MACD)...")
    
    try:
        # Original version
        from Initialize_RSI_EMA_MACD import Initialize_RSI_EMA_MACD as original_indicators
        start_time = time.perf_counter()
        df_orig_ind = original_indicators(df.copy())
        original_indicators_time = time.perf_counter() - start_time
        print(f"   📊 Original: {original_indicators_time:.3f}s")
        
        try:
            # Numba version
            from Initialize_RSI_EMA_MACD_numba import Initialize_RSI_EMA_MACD as numba_indicators
            start_time = time.perf_counter()
            df_numba_ind = numba_indicators(df.copy())
            numba_indicators_time = time.perf_counter() - start_time
            print(f"   ⚡ Numba: {numba_indicators_time:.3f}s")
            
            speedup = original_indicators_time / numba_indicators_time
            print(f"   🚀 Speedup: {speedup:.1f}x ({(speedup-1)*100:.1f}% faster)")
            
            results['technical_indicators'] = {
                'original_time': original_indicators_time,
                'numba_time': numba_indicators_time,
                'speedup': speedup
            }
            
        except ImportError:
            print("   ⚠️  Numba version not available")
            df_numba_ind = df_orig_ind
            
    except Exception as e:
        print(f"   ❌ Technical indicators test failed: {e}")
        return results
    
    # Test 2: Local Maxima/Minima Detection
    print("\n🔬 Testing Local Maxima/Minima Detection...")
    
    try:
        # Original version
        from Local_Maximas_Minimas import Local_Max_Min as original_maxmin
        start_time = time.perf_counter()
        df_orig_maxmin = original_maxmin(df_orig_ind.copy())
        original_maxmin_time = time.perf_counter() - start_time
        print(f"   📊 Original: {original_maxmin_time:.3f}s")
        
        try:
            # Numba version
            from Local_Maximas_Minimas_numba import Local_Max_Min as numba_maxmin
            start_time = time.perf_counter()
            df_numba_maxmin = numba_maxmin(df_numba_ind.copy())
            numba_maxmin_time = time.perf_counter() - start_time
            print(f"   ⚡ Numba: {numba_maxmin_time:.3f}s")
            
            speedup = original_maxmin_time / numba_maxmin_time
            print(f"   🚀 Speedup: {speedup:.1f}x ({(speedup-1)*100:.1f}% faster)")
            
            results['maxima_minima'] = {
                'original_time': original_maxmin_time,
                'numba_time': numba_maxmin_time,
                'speedup': speedup
            }
            
        except ImportError:
            print("   ⚠️  Numba version not available")
            df_numba_maxmin = df_orig_maxmin
            
    except Exception as e:
        print(f"   ❌ Maxima/minima test failed: {e}")
        return results
    
    # Test 3: Divergence Analysis
    print("\n🔬 Testing Bullish Divergence Analysis...")
    
    try:
        # Original version
        from CBullDivg_Analysis_vectorized import CBullDivg_analysis as original_divergence
        start_time = time.perf_counter()
        df_orig_div = original_divergence(df_orig_maxmin.copy(), **test_params)
        original_divergence_time = time.perf_counter() - start_time
        print(f"   📊 Original: {original_divergence_time:.3f}s")
        
        try:
            # Numba version
            from CBullDivg_Analysis_numba import CBullDivg_analysis as numba_divergence
            start_time = time.perf_counter()
            df_numba_div = numba_divergence(df_numba_maxmin.copy(), **test_params)
            numba_divergence_time = time.perf_counter() - start_time
            print(f"   ⚡ Numba: {numba_divergence_time:.3f}s")
            
            speedup = original_divergence_time / numba_divergence_time
            print(f"   🚀 Speedup: {speedup:.1f}x ({(speedup-1)*100:.1f}% faster)")
            
            results['divergence_analysis'] = {
                'original_time': original_divergence_time,
                'numba_time': numba_divergence_time,
                'speedup': speedup
            }
            
        except ImportError:
            print("   ⚠️  Numba version not available")
            
    except Exception as e:
        print(f"   ❌ Divergence analysis test failed: {e}")
        return results
    
    # Calculate overall results
    if results:
        total_original = sum(r['original_time'] for r in results.values())
        total_numba = sum(r['numba_time'] for r in results.values() if 'numba_time' in r)
        
        if total_numba > 0:
            overall_speedup = total_original / total_numba
            time_saved = total_original - total_numba
            
            print("\n" + "=" * 80)
            print("📋 OVERALL PERFORMANCE RESULTS")
            print("=" * 80)
            print(f"📊 Dataset Size: {len(df):,} rows")
            print(f"⏱️  Original Total Time: {total_original:.3f}s")
            print(f"⚡ Numba Total Time: {total_numba:.3f}s")
            print(f"💾 Time Saved: {time_saved:.3f}s")
            print(f"🚀 Overall Speedup: {overall_speedup:.1f}x")
            print(f"📈 Performance Gain: +{(overall_speedup-1)*100:.1f}%")
            print("=" * 80)
            
            # Extrapolate for larger datasets
            if len(df) < 50000:
                large_dataset_time_saved = time_saved * (50000 / len(df))
                print(f"💡 Estimated time saved for 50k rows: {large_dataset_time_saved:.1f}s")
                
            if len(df) < 100000:
                huge_dataset_time_saved = time_saved * (100000 / len(df))
                print(f"💡 Estimated time saved for 100k rows: {huge_dataset_time_saved:.1f}s")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Benchmark Numba performance optimizations')
    parser.add_argument('--data', type=str, help='Path to your data file (.parquet or .csv)')
    parser.add_argument('--test-size', type=int, default=5000, help='Size for generated test data')
    
    args = parser.parse_args()
    
    if args.data:
        # Use actual data file
        df = benchmark_with_actual_data(args.data)
        if df is None:
            return
    else:
        # Generate test data
        logger.info(f"📊 Generating test data: {args.test_size:,} rows")
        np.random.seed(42)
        
        dates = pd.date_range(start='2020-01-01', periods=args.test_size, freq='D')
        base_price = 50000  # BTC-like prices
        price_walk = np.cumsum(np.random.normal(0, 0.02, args.test_size)) + base_price
        
        df = pd.DataFrame({
            'date': dates.strftime('%Y-%m-%d'),
            'open': price_walk + np.random.normal(0, 500, args.test_size),
            'high': price_walk + np.abs(np.random.normal(500, 300, args.test_size)),
            'low': price_walk - np.abs(np.random.normal(500, 300, args.test_size)),
            'close': price_walk + np.random.normal(0, 300, args.test_size),
            'volume': np.random.randint(1000000, 50000000, args.test_size)
        })
    
    # Run the benchmark
    results = run_benchmark(df)
    
    if not results:
        print("❌ No benchmark results available")
        return
    
    print("\n🎯 To use the high-performance version:")
    print("   python server_numba.py")
    print("\n💡 Or install Numba if not available:")
    print("   python install_numba.py")

if __name__ == "__main__":
    main()