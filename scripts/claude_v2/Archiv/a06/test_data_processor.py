#!/usr/bin/env python3
"""
Test script for data_processor_v2.py
"""

from data_processor_v2 import DataProcessorV2, load_for_analysis
import os

def test_processor():
    processor = DataProcessorV2()
    
    # Find a test file
    test_files = [
        "../claude_v1/uploads/20250820_171344_btc_1day_candlesticks_all.parquet",
        "../claude_v1/uploads/20250820_171806_btc_1day_candlesticks_all.parquet"
    ]
    
    test_file = None
    for file in test_files:
        if os.path.exists(file):
            test_file = file
            break
    
    if not test_file:
        print("No test file found. Place a parquet/csv file to test.")
        return
    
    print(f"Testing with: {test_file}")
    
    # Test optimized loading
    try:
        df = processor.load_for_analysis(test_file)
        print(f"SUCCESS: Loaded {len(df):,} rows, {len(df.columns)} columns")
        print(f"Columns: {list(df.columns)}")
        print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    test_processor()