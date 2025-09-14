#!/usr/bin/env python3
"""
Data Processor V2.0.0 - High-Performance Data Loading Module

Optimized data processing using Polars for speed, with seamless Pandas compatibility.
Provides intelligent memory management and performance monitoring for large datasets.

Author: Claude Code
Version: 2.0.0
"""

import polars as pl
import pandas as pd
import numpy as np
import os
import logging
import time
import psutil
from typing import List, Iterator, Union, Optional, Tuple, Dict, Any
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """Monitor and log performance metrics"""
    
    def __init__(self):
        self.start_time: float = 0
        self.start_memory: float = 0
        
    def start(self) -> None:
        """Start monitoring"""
        self.start_time = time.time()
        self.start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
    def stop_and_log(self, operation: str, rows: int = 0) -> Dict[str, float]:
        """Stop monitoring and log results"""
        elapsed = time.time() - self.start_time
        current_memory = psutil.Process().memory_info().rss / 1024 / 1024
        memory_delta = current_memory - self.start_memory
        
        metrics = {
            'elapsed_seconds': elapsed,
            'memory_delta_mb': memory_delta,
            'current_memory_mb': current_memory,
            'rows_processed': rows
        }
        
        logger.info(f"📊 {operation}: {elapsed:.2f}s, Memory: {memory_delta:+.1f}MB, "
                   f"Rows: {rows:,}")
        return metrics

class DataProcessorV2:
    """
    High-performance data processor with Polars backend and Pandas compatibility.
    
    Features:
    - Lazy evaluation for memory efficiency
    - Intelligent column selection
    - Data type optimization
    - Batch processing for large files
    - Performance monitoring
    - 100% Pandas compatibility via adapter pattern
    """
    
    REQUIRED_OHLC_COLS = ['date', 'open', 'high', 'low', 'close']
    OPTIONAL_COLS = ['volume', 'timestamp']
    
    def __init__(self):
        self.monitor = PerformanceMonitor()
        
    def load_data_optimized(self, filepath: str, 
                          required_cols: Optional[List[str]] = None) -> pl.LazyFrame:
        """
        Load data with Polars lazy evaluation for optimal performance.
        
        Args:
            filepath: Path to parquet/csv file
            required_cols: Columns to load (default: OHLC + volume)
            
        Returns:
            Polars LazyFrame for memory-efficient processing
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If required columns missing
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
            
        self.monitor.start()
        
        # Default to OHLC columns if not specified
        if required_cols is None:
            required_cols = self.REQUIRED_OHLC_COLS.copy()
            
        try:
            file_ext = Path(filepath).suffix.lower()
            
            if file_ext == '.parquet':
                # Use scan_parquet for lazy loading
                lazy_df = pl.scan_parquet(filepath)
            elif file_ext == '.csv':
                # Use scan_csv for lazy loading
                lazy_df = pl.scan_csv(filepath, try_parse_dates=True)
            else:
                raise ValueError(f"Unsupported file format: {file_ext}")
                
            # Validate columns exist
            available_cols = lazy_df.columns
            missing_cols = [col for col in self.REQUIRED_OHLC_COLS if col not in available_cols]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
                
            # Select only required columns for memory efficiency
            cols_to_select = [col for col in required_cols if col in available_cols]
            lazy_df = lazy_df.select(cols_to_select)
            
            # Basic data type optimization at scan level
            lazy_df = lazy_df.with_columns([
                pl.col('date').cast(pl.Datetime, strict=False),
                pl.col(['open', 'high', 'low', 'close']).cast(pl.Float64, strict=False)
            ])
            
            logger.info(f"✅ Loaded {filepath} with {len(cols_to_select)} columns (lazy)")
            return lazy_df
            
        except Exception as e:
            logger.error(f"❌ Polars loading failed: {e}")
            raise
            
    def convert_to_pandas_minimal(self, lazy_df: pl.LazyFrame, 
                                required_cols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Convert Polars LazyFrame to Pandas DataFrame with minimal memory usage.
        
        Args:
            lazy_df: Polars LazyFrame
            required_cols: Specific columns needed (reduces memory)
            
        Returns:
            Optimized Pandas DataFrame
        """
        self.monitor.start()
        
        try:
            # Select only required columns if specified
            if required_cols:
                available_cols = [col for col in required_cols if col in lazy_df.columns]
                lazy_df = lazy_df.select(available_cols)
                
            # Collect with streaming for large datasets
            polars_df = lazy_df.collect(streaming=True)
            
            # Convert to pandas
            pandas_df = polars_df.to_pandas(use_pyarrow_extension_array=True)
            
            # Optimize data types
            pandas_df = self.optimize_dtypes(pandas_df)
            
            metrics = self.monitor.stop_and_log("Polars->Pandas conversion", len(pandas_df))
            return pandas_df
            
        except Exception as e:
            logger.warning(f"⚠️ Polars conversion failed: {e}. Falling back to direct Pandas load")
            return self._fallback_pandas_load(lazy_df)
            
    def optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize DataFrame data types for memory efficiency.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with optimized dtypes
        """
        original_memory = df.memory_usage(deep=True).sum() / 1024 / 1024
        
        # Optimize numeric columns
        for col in df.columns:
            if col in ['open', 'high', 'low', 'close', 'volume']:
                if df[col].dtype in ['int64', 'float64']:
                    # Check if float32 precision is sufficient
                    if df[col].dtype == 'float64':
                        if (df[col] == df[col].astype('float32')).all():
                            df[col] = df[col].astype('float32')
                    elif df[col].dtype == 'int64':
                        if df[col].min() >= 0 and df[col].max() <= 4294967295:
                            df[col] = df[col].astype('uint32')
                        elif df[col].min() >= -2147483648 and df[col].max() <= 2147483647:
                            df[col] = df[col].astype('int32')
                            
        # Ensure date column is datetime
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            
        new_memory = df.memory_usage(deep=True).sum() / 1024 / 1024
        saved_mb = original_memory - new_memory
        
        if saved_mb > 0.1:
            logger.info(f"🔧 Memory optimization: -{saved_mb:.1f}MB ({saved_mb/original_memory*100:.1f}%)")
            
        return df
        
    def process_in_chunks(self, filepath: str, 
                         chunk_size: int = 100000) -> Iterator[pd.DataFrame]:
        """
        Process large files in memory-efficient chunks.
        
        Args:
            filepath: Path to file
            chunk_size: Rows per chunk
            
        Yields:
            DataFrame chunks
        """
        file_size_mb = os.path.getsize(filepath) / 1024 / 1024
        
        if file_size_mb < 500:  # Small file, load normally
            lazy_df = self.load_data_optimized(filepath)
            yield self.convert_to_pandas_minimal(lazy_df)
            return
            
        logger.info(f"📦 Processing {file_size_mb:.1f}MB file in {chunk_size:,} row chunks")
        
        try:
            lazy_df = self.load_data_optimized(filepath)
            
            # Process in chunks using lazy evaluation
            offset = 0
            while True:
                chunk_df = lazy_df.slice(offset, chunk_size).collect(streaming=True)
                
                if chunk_df.is_empty():
                    break
                    
                pandas_chunk = chunk_df.to_pandas()
                pandas_chunk = self.optimize_dtypes(pandas_chunk)
                
                yield pandas_chunk
                offset += chunk_size
                
        except Exception as e:
            logger.error(f"❌ Chunk processing failed: {e}")
            raise
            
    def load_for_analysis(self, filepath: str) -> pd.DataFrame:
        """
        Optimized loading specifically for analysis modules.
        Maintains 100% compatibility with existing Pandas-based modules.
        
        Args:
            filepath: Data file path
            
        Returns:
            Analysis-ready Pandas DataFrame
        """
        self.monitor.start()
        
        try:
            # Load with Polars for speed
            lazy_df = self.load_data_optimized(filepath, self.REQUIRED_OHLC_COLS)
            
            # Convert to Pandas for module compatibility
            df = self.convert_to_pandas_minimal(lazy_df, self.REQUIRED_OHLC_COLS)
            
            # Final validation for analysis modules
            if df.empty:
                raise ValueError("Empty dataset after loading")
                
            if df[['open', 'high', 'low', 'close']].isnull().any().any():
                logger.warning("⚠️ Found NaN values in OHLC data, filling forward")
                df[['open', 'high', 'low', 'close']] = df[['open', 'high', 'low', 'close']].fillna(method='ffill')
                
            metrics = self.monitor.stop_and_log("Complete load_for_analysis", len(df))
            return df
            
        except Exception as e:
            logger.warning(f"⚠️ Optimized loading failed: {e}. Using fallback Pandas")
            return self._fallback_pandas_load(filepath)
            
    def _fallback_pandas_load(self, filepath: Union[str, pl.LazyFrame]) -> pd.DataFrame:
        """Fallback to pure Pandas loading if Polars fails"""
        self.monitor.start()
        
        if isinstance(filepath, str):
            file_ext = Path(filepath).suffix.lower()
            if file_ext == '.parquet':
                df = pd.read_parquet(filepath)
            elif file_ext == '.csv':
                df = pd.read_csv(filepath, parse_dates=['date'])
            else:
                raise ValueError(f"Unsupported fallback format: {file_ext}")
        else:
            # Convert LazyFrame via collect
            df = filepath.collect().to_pandas()
            
        df = self.optimize_dtypes(df)
        self.monitor.stop_and_log("Fallback Pandas load", len(df))
        return df

# Convenience functions for backward compatibility
def load_data_optimized(filepath: str, required_cols: Optional[List[str]] = None) -> pl.LazyFrame:
    """Convenience function for optimized data loading"""
    processor = DataProcessorV2()
    return processor.load_data_optimized(filepath, required_cols)

def load_for_analysis(filepath: str) -> pd.DataFrame:
    """Convenience function for analysis-ready DataFrame loading"""
    processor = DataProcessorV2()
    return processor.load_for_analysis(filepath)