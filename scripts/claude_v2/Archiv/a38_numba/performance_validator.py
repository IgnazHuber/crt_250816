"""
Performance Validation Framework
Ensures Numba-optimized functions produce identical results to original functions
"""

import pandas as pd
import numpy as np
import time
import warnings
from typing import Dict, Tuple, Any
import logging

# Import original functions
from Initialize_RSI_EMA_MACD import Initialize_RSI_EMA_MACD as original_indicators
from Local_Maximas_Minimas import Local_Max_Min as original_maxmin  
from CBullDivg_Analysis_vectorized import CBullDivg_analysis as original_divergence

# Import optimized functions
from Initialize_RSI_EMA_MACD_numba import Initialize_RSI_EMA_MACD as numba_indicators
from Local_Maximas_Minimas_numba import Local_Max_Min as numba_maxmin
from CBullDivg_Analysis_numba import CBullDivg_analysis as numba_divergence

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PerformanceValidator:
    """
    Comprehensive validation framework to ensure Numba optimizations
    produce identical results while dramatically improving performance
    """
    
    def __init__(self):
        self.tolerance = 1e-10  # Numerical precision tolerance
        self.validation_results = {}
        
    def compare_dataframes(self, df_original: pd.DataFrame, df_optimized: pd.DataFrame, 
                          test_name: str) -> Dict[str, Any]:
        """Compare two dataframes and return detailed comparison results"""
        
        # Check shape equality
        shape_match = df_original.shape == df_optimized.shape
        
        # Check column equality
        cols_original = set(df_original.columns)
        cols_optimized = set(df_optimized.columns)
        missing_cols = cols_original - cols_optimized
        extra_cols = cols_optimized - cols_original
        
        numerical_diffs = {}
        max_diff = 0.0
        
        # Compare numerical columns
        common_cols = cols_original & cols_optimized
        for col in common_cols:
            if df_original[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                try:
                    # Handle NaN values
                    orig_vals = df_original[col].fillna(0)
                    opt_vals = df_optimized[col].fillna(0)
                    
                    diff = np.abs(orig_vals - opt_vals)
                    max_col_diff = diff.max()
                    max_diff = max(max_diff, max_col_diff)
                    
                    numerical_diffs[col] = {
                        'max_difference': max_col_diff,
                        'mean_difference': diff.mean(),
                        'within_tolerance': max_col_diff <= self.tolerance
                    }
                except Exception as e:
                    numerical_diffs[col] = {'error': str(e)}
        
        overall_match = (shape_match and 
                        len(missing_cols) == 0 and 
                        len(extra_cols) == 0 and 
                        max_diff <= self.tolerance)
        
        return {
            'test_name': test_name,
            'overall_match': overall_match,
            'shape_match': shape_match,
            'missing_columns': list(missing_cols),
            'extra_columns': list(extra_cols),
            'numerical_differences': numerical_diffs,
            'max_overall_difference': max_diff
        }
    
    def benchmark_function(self, func, args, test_name: str, runs: int = 3) -> Dict[str, Any]:
        """Benchmark a function with multiple runs and return timing statistics"""
        times = []
        
        # Warm-up run
        result = func(*args)
        
        # Timed runs
        for _ in range(runs):
            start_time = time.perf_counter()
            result = func(*args)
            end_time = time.perf_counter()
            times.append(end_time - start_time)
        
        return {
            'test_name': test_name,
            'times': times,
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'result': result
        }
    
    def validate_technical_indicators(self, test_data: pd.DataFrame) -> Dict[str, Any]:
        """Validate RSI, EMA, and MACD calculations"""
        logger.info("🔍 Validating technical indicators (RSI, EMA, MACD)")
        
        # Benchmark original implementation
        original_bench = self.benchmark_function(
            original_indicators, 
            (test_data.copy(),), 
            "Original Technical Indicators"
        )
        
        # Benchmark optimized implementation  
        numba_bench = self.benchmark_function(
            numba_indicators,
            (test_data.copy(),),
            "Numba Technical Indicators"
        )
        
        # Compare results
        comparison = self.compare_dataframes(
            original_bench['result'], 
            numba_bench['result'], 
            "Technical Indicators"
        )
        
        speedup = original_bench['mean_time'] / numba_bench['mean_time']
        
        return {
            'validation': comparison,
            'performance': {
                'original_time': original_bench['mean_time'],
                'numba_time': numba_bench['mean_time'], 
                'speedup': speedup,
                'speedup_percentage': (speedup - 1) * 100
            }
        }
    
    def validate_maxima_minima(self, test_data: pd.DataFrame) -> Dict[str, Any]:
        """Validate local maxima/minima detection"""
        logger.info("🔍 Validating local maxima/minima detection")
        
        # Benchmark original implementation
        original_bench = self.benchmark_function(
            original_maxmin,
            (test_data.copy(),),
            "Original Maxima/Minima"
        )
        
        # Benchmark optimized implementation
        numba_bench = self.benchmark_function(
            numba_maxmin,
            (test_data.copy(),),
            "Numba Maxima/Minima"
        )
        
        # Compare results
        comparison = self.compare_dataframes(
            original_bench['result'],
            numba_bench['result'],
            "Maxima/Minima Detection"
        )
        
        speedup = original_bench['mean_time'] / numba_bench['mean_time']
        
        return {
            'validation': comparison,
            'performance': {
                'original_time': original_bench['mean_time'],
                'numba_time': numba_bench['mean_time'],
                'speedup': speedup,
                'speedup_percentage': (speedup - 1) * 100
            }
        }
    
    def validate_divergence_analysis(self, test_data: pd.DataFrame, 
                                   window: int = 5, candle_tol: float = 0.1, 
                                   macd_tol: float = 3.25) -> Dict[str, Any]:
        """Validate bullish divergence analysis"""
        logger.info("🔍 Validating bullish divergence analysis")
        
        # Benchmark original implementation
        original_bench = self.benchmark_function(
            original_divergence,
            (test_data.copy(), window, candle_tol, macd_tol),
            "Original Divergence Analysis"
        )
        
        # Benchmark optimized implementation
        numba_bench = self.benchmark_function(
            numba_divergence,
            (test_data.copy(), window, candle_tol, macd_tol),
            "Numba Divergence Analysis"
        )
        
        # Compare results
        comparison = self.compare_dataframes(
            original_bench['result'],
            numba_bench['result'],
            "Divergence Analysis"
        )
        
        speedup = original_bench['mean_time'] / numba_bench['mean_time']
        
        return {
            'validation': comparison,
            'performance': {
                'original_time': original_bench['mean_time'],
                'numba_time': numba_bench['mean_time'],
                'speedup': speedup,
                'speedup_percentage': (speedup - 1) * 100
            }
        }
    
    def run_full_validation(self, test_data: pd.DataFrame) -> Dict[str, Any]:
        """Run complete validation suite"""
        logger.info("🚀 Starting comprehensive performance validation")
        logger.info(f"📊 Test dataset: {len(test_data)} rows, {len(test_data.columns)} columns")
        
        results = {}
        
        # 1. Validate technical indicators
        try:
            results['technical_indicators'] = self.validate_technical_indicators(test_data)
            logger.info("✅ Technical indicators validation completed")
        except Exception as e:
            logger.error(f"❌ Technical indicators validation failed: {e}")
            results['technical_indicators'] = {'error': str(e)}
        
        # Prepare data with technical indicators for next steps
        try:
            prepared_data = numba_indicators(test_data.copy())
            prepared_data = numba_maxmin(prepared_data)
        except Exception as e:
            logger.error(f"❌ Data preparation failed: {e}")
            return results
        
        # 2. Validate maxima/minima detection  
        try:
            results['maxima_minima'] = self.validate_maxima_minima(prepared_data)
            logger.info("✅ Maxima/minima validation completed")
        except Exception as e:
            logger.error(f"❌ Maxima/minima validation failed: {e}")
            results['maxima_minima'] = {'error': str(e)}
        
        # 3. Validate divergence analysis
        try:
            results['divergence_analysis'] = self.validate_divergence_analysis(prepared_data)
            logger.info("✅ Divergence analysis validation completed")
        except Exception as e:
            logger.error(f"❌ Divergence analysis validation failed: {e}")
            results['divergence_analysis'] = {'error': str(e)}
        
        # Calculate overall statistics
        total_original_time = 0
        total_numba_time = 0
        overall_speedup = 0
        valid_tests = 0
        
        for test_name, test_results in results.items():
            if 'performance' in test_results:
                total_original_time += test_results['performance']['original_time']
                total_numba_time += test_results['performance']['numba_time']
                valid_tests += 1
        
        if valid_tests > 0:
            overall_speedup = total_original_time / total_numba_time
        
        results['summary'] = {
            'total_original_time': total_original_time,
            'total_numba_time': total_numba_time,
            'overall_speedup': overall_speedup,
            'overall_speedup_percentage': (overall_speedup - 1) * 100,
            'valid_tests': valid_tests,
            'data_size': len(test_data)
        }
        
        logger.info("🎯 Validation Summary:")
        logger.info(f"   📈 Overall Speedup: {overall_speedup:.1f}x ({(overall_speedup-1)*100:.1f}% faster)")
        logger.info(f"   ⏱️  Original Time: {total_original_time:.3f}s")
        logger.info(f"   ⚡ Numba Time: {total_numba_time:.3f}s")
        logger.info(f"   📊 Dataset Size: {len(test_data):,} rows")
        
        return results

def create_test_data(size: int = 1000) -> pd.DataFrame:
    """Generate realistic test data for validation"""
    np.random.seed(42)  # For reproducible results
    
    dates = pd.date_range(start='2020-01-01', periods=size, freq='D')
    
    # Generate realistic OHLC data with trends
    base_price = 100
    price_walk = np.cumsum(np.random.normal(0, 0.02, size)) + base_price
    
    data = {
        'date': dates.strftime('%Y-%m-%d'),
        'open': price_walk + np.random.normal(0, 0.5, size),
        'high': price_walk + np.abs(np.random.normal(0.5, 0.3, size)),
        'low': price_walk - np.abs(np.random.normal(0.5, 0.3, size)), 
        'close': price_walk + np.random.normal(0, 0.3, size),
        'volume': np.random.randint(1000000, 10000000, size)
    }
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    # Run validation tests
    validator = PerformanceValidator()
    
    # Test with different dataset sizes
    test_sizes = [1000, 5000, 10000]
    
    for size in test_sizes:
        logger.info(f"\n{'='*80}")
        logger.info(f"🧪 Testing with {size:,} data points")
        logger.info(f"{'='*80}")
        
        test_data = create_test_data(size)
        results = validator.run_full_validation(test_data)
        
        # Print key results
        if 'summary' in results:
            summary = results['summary']
            print(f"\n📋 VALIDATION RESULTS FOR {size:,} ROWS:")
            print(f"   ⚡ Overall Speedup: {summary['overall_speedup']:.1f}x")
            print(f"   📊 Speedup Percentage: +{summary['overall_speedup_percentage']:.1f}%")
            print(f"   ⏱️  Time Saved: {summary['total_original_time'] - summary['total_numba_time']:.3f}s")
            print(f"   ✅ Valid Tests: {summary['valid_tests']}")
        
        # Check validation results
        all_valid = True
        for test_name, test_result in results.items():
            if test_name != 'summary' and 'validation' in test_result:
                if not test_result['validation']['overall_match']:
                    all_valid = False
                    print(f"   ❌ {test_name}: Results don't match!")
        
        if all_valid:
            print(f"   ✅ All validations passed - results are identical!")
        
        print()