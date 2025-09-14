#!/usr/bin/env python3
"""
Test web server tolerance behavior with curl commands - no external dependencies needed
"""

import subprocess
import json
import tempfile
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def create_realistic_test_data():
    """Create realistic market data with clear divergence patterns"""
    
    # Create 200 data points over several days
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(hours=i) for i in range(200)]
    
    # Generate realistic price movement with trend and volatility
    np.random.seed(42)  # For reproducible results
    
    # Base price trend (starts at 50000, declines to 48000, then recovers)
    base_prices = np.concatenate([
        np.linspace(50000, 48000, 100),  # Decline phase
        np.linspace(48000, 49500, 100)   # Recovery phase
    ])
    
    # Add realistic volatility
    volatility = np.random.normal(0, 50, 200)
    close_prices = base_prices + volatility
    
    # Generate OHLC from close prices
    high_prices = close_prices + np.abs(np.random.normal(0, 20, 200))
    low_prices = close_prices - np.abs(np.random.normal(0, 20, 200))
    open_prices = np.roll(close_prices, 1)  # Previous close becomes next open
    open_prices[0] = close_prices[0]
    
    # Generate realistic volume
    volume = np.random.uniform(800, 1200, 200)
    
    # Create DataFrame
    df = pd.DataFrame({
        'date': dates,
        'open': open_prices,
        'high': high_prices, 
        'low': low_prices,
        'close': close_prices,
        'volume': volume
    })
    
    return df

def run_curl_command(url, method="GET", json_data=None, files=None):
    """Run curl command and return response"""
    cmd = ["curl", "-s"]
    
    if method == "POST":
        cmd.append("-X")
        cmd.append("POST")
        
        if json_data:
            cmd.extend(["-H", "Content-Type: application/json"])
            cmd.extend(["-d", json.dumps(json_data)])
        elif files:
            for key, (filename, content) in files.items():
                # Write content to temp file
                with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                    f.write(content)
                    temp_path = f.name
                
                cmd.extend(["-F", f"{key}=@{temp_path}"])
    
    cmd.append(url)
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        # Clean up temp files
        if files:
            for key, (filename, content) in files.items():
                try:
                    os.unlink(temp_path)
                except:
                    pass
        
        if result.returncode == 0:
            return json.loads(result.stdout)
        else:
            print(f"Curl error: {result.stderr}")
            return None
    except Exception as e:
        print(f"Command failed: {e}")
        return None

def test_server_tolerance_behavior(server_url="http://localhost:5000"):
    """Test server tolerance behavior using curl commands"""
    
    print("=" * 80)
    print("TESTING WEB SERVER TOLERANCE BEHAVIOR WITH CURL")
    print("=" * 80)
    
    # Create realistic test data
    print("Generating realistic market data...")
    test_df = create_realistic_test_data()
    csv_content = test_df.to_csv(index=False)
    print(f"Created {len(test_df)} data points")
    
    # Check server health
    health_response = run_curl_command(f"{server_url}/api/health")
    if not health_response or health_response.get('status') != 'ok':
        print(f"Server not available at {server_url}")
        return False
        
    print(f"Server is running: {health_response.get('message')}")
    
    # Test different Candle% values
    candle_tolerances = [0.5, 1.0, 2.0, 4.0, 8.0]
    results = []
    
    print(f"\nTesting tolerance behavior with {len(candle_tolerances)} different Candle% values...")
    
    for i, candle_tol in enumerate(candle_tolerances):
        print(f"\nTest {i+1}/{len(candle_tolerances)}: Candle% = {candle_tol}%")
        
        # Upload data
        files = {'file': ('test_data.csv', csv_content)}
        upload_response = run_curl_command(f"{server_url}/api/upload", "POST", files=files)
        
        if not upload_response or not upload_response.get('success'):
            print(f"   Upload failed: {upload_response}")
            continue
            
        session_id = upload_response.get('session_id')
        print(f"   Session ID: {session_id}")
        
        # Run analysis with proper variants structure
        analysis_data = {
            'session_id': session_id,
            'variants': [
                {
                    'id': 'test_variant',
                    'name': f'Test Candle {candle_tol}%',
                    'window': 20,
                    'candleTol': candle_tol,
                    'macdTol': 0.001,
                    'enabled': True
                }
            ]
        }
        
        analysis_response = run_curl_command(f"{server_url}/api/analyze", "POST", json_data=analysis_data)
        
        if not analysis_response or not analysis_response.get('success'):
            print(f"   Analysis failed: {analysis_response}")
            continue
            
        # Count markers from server response format
        analysis_results = analysis_response.get('results', {})
        total_markers = 0
        variant_counts = {}
        
        for variant_id, variant_data in analysis_results.items():
            hidden_count = len(variant_data.get('hidden', []))
            classic_count = len(variant_data.get('classic', []))
            variant_total = variant_data.get('total', hidden_count + classic_count)
            
            variant_counts[variant_id] = {
                'hidden': hidden_count,
                'classic': classic_count,
                'total': variant_total
            }
            total_markers += variant_total
            print(f"     {variant_id}: {variant_total} markers (hidden: {hidden_count}, classic: {classic_count})")
            
        print(f"   Total markers found: {total_markers}")
        results.append((candle_tol, total_markers, variant_counts))
    
    # Analyze tolerance behavior
    print("\n" + "=" * 80)
    print("DETAILED TOLERANCE BEHAVIOR ANALYSIS")
    print("=" * 80)
    
    if len(results) < 2:
        print("Not enough results to analyze tolerance behavior")
        return False
    
    print("\nCandle% | Total | Details")
    print("--------|-------|--------------------------------------------------")
    for candle_tol, total, variants in results:
        details = ", ".join([f"{k}: {v['total']}" for k, v in variants.items() if v['total'] > 0])
        if not details:
            details = "No markers found"
        print(f"{candle_tol:7.1f} | {total:5d} | {details}")
    
    # Check for correct tolerance behavior
    print("\nTolerance Behavior Check:")
    behavior_correct = True
    
    for i in range(1, len(results)):
        prev_tol, prev_total, _ = results[i-1]
        curr_tol, curr_total, _ = results[i]
        
        change = curr_total - prev_total
        
        if change < 0:
            print(f"WRONG: {prev_tol}% -> {curr_tol}% = {prev_total} -> {curr_total} (DECREASED by {-change})")
            behavior_correct = False
        elif change > 0:
            print(f"CORRECT: {prev_tol}% -> {curr_tol}% = {prev_total} -> {curr_total} (INCREASED by {change})")
        else:
            print(f"SAME: {prev_tol}% -> {curr_tol}% = {prev_total} -> {curr_total} (no change)")
    
    # Final assessment
    print("\n" + "=" * 80)
    print("FINAL ASSESSMENT")
    print("=" * 80)
    
    total_markers_found = sum(total for _, total, _ in results)
    
    if total_markers_found == 0:
        print("CRITICAL: NO MARKERS FOUND AT ALL")
        return False
    elif not behavior_correct:
        print("TOLERANCE LOGIC IS STILL WRONG!")
        print("Higher Candle% should find MORE markers, not fewer")
        return False
    else:
        print("SUCCESS: TOLERANCE BEHAVIOR IS CORRECT!")
        print(f"Found {total_markers_found} total markers across all tests")
        print("Higher Candle% correctly finds same or more markers")
        return True

if __name__ == "__main__":
    print("COMPREHENSIVE WEB SERVER TOLERANCE TEST")
    
    # Test the server
    success = test_server_tolerance_behavior()
    
    if not success:
        print("\nTROUBLESHOOTING:")
        print("1. Make sure server_numba.py is running")
        print("2. Check server console for error messages")
        print("3. Verify correct modules are loaded")
    
    sys.exit(0 if success else 1)