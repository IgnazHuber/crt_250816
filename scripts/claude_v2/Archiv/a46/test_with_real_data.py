#!/usr/bin/env python3
"""
Test web server tolerance behavior with realistic market data that should produce divergences
"""

import sys
import requests
import json
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

def test_server_with_real_data(server_url="http://localhost:5000"):
    """Test server with realistic data that should produce divergences"""
    
    print("=" * 80)
    print("TESTING WEB SERVER WITH REALISTIC MARKET DATA")
    print("=" * 80)
    
    # Create realistic test data
    print("Generating realistic market data with divergence patterns...")
    test_df = create_realistic_test_data()
    
    # Convert to CSV
    csv_content = test_df.to_csv(index=False)
    print(f"Created {len(test_df)} data points spanning {(test_df['date'].max() - test_df['date'].min()).days} days")
    
    # Check server health
    try:
        response = requests.get(f"{server_url}/api/health", timeout=5)
        if response.status_code != 200:
            print(f"❌ Server health check failed: {response.status_code}")
            return False
        print(f"✅ Server is healthy at {server_url}")
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")
        return False
    
    # Test different Candle% values with realistic data
    candle_tolerances = [0.5, 1.0, 2.0, 4.0, 8.0]
    results = []
    
    print(f"\\nTesting tolerance behavior with {len(candle_tolerances)} different Candle% values...")
    
    for i, candle_tol in enumerate(candle_tolerances):
        print(f"\\nTest {i+1}/{len(candle_tolerances)}: Candle% = {candle_tol}%")
        
        try:
            # Upload realistic data
            files = {'file': ('realistic_data.csv', csv_content, 'text/csv')}
            upload_response = requests.post(f"{server_url}/api/upload", files=files, timeout=15)
            
            if upload_response.status_code != 200:
                print(f"   ❌ Upload failed: {upload_response.status_code}")
                continue
                
            upload_data = upload_response.json()
            if not upload_data.get('success'):
                print(f"   ❌ Upload error: {upload_data.get('error')}")
                continue
                
            session_id = upload_data.get('session_id')
            
            # Run analysis with proper variants (like the web frontend)
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
            
            analysis_response = requests.post(f"{server_url}/api/analyze", 
                                            json=analysis_data, 
                                            timeout=45)
            
            if analysis_response.status_code != 200:
                print(f"   ❌ Analysis failed: {analysis_response.status_code}")
                print(f"   Response: {analysis_response.text[:200]}...")
                continue
                
            analysis_result = analysis_response.json()
            if not analysis_result.get('success'):
                print(f"   ❌ Analysis error: {analysis_result.get('error')}")
                continue
                
            # Count markers from server response format
            results = analysis_result.get('results', {})
            total_markers = 0
            variant_counts = {}
            
            for variant_id, variant_data in results.items():
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
                
            print(f"   ✅ Total markers found: {total_markers}")
            results.append((candle_tol, total_markers, variant_counts))
            
        except Exception as e:
            print(f"   ❌ Request failed: {e}")
            continue
    
    # Analyze tolerance behavior
    print("\\n" + "=" * 80)
    print("DETAILED TOLERANCE BEHAVIOR ANALYSIS")
    print("=" * 80)
    
    if len(results) < 2:
        print("❌ Not enough results to analyze tolerance behavior")
        return False
    
    print("\\nCandle% | Total | Details")
    print("--------|-------|--------------------------------------------------")
    for candle_tol, total, variants in results:
        details = ", ".join([f"{k}:{v}" for k, v in variants.items() if v > 0])
        if not details:
            details = "No markers found"
        print(f"{candle_tol:7.1f} | {total:5d} | {details}")
    
    # Check for correct tolerance behavior
    print("\\nTolerance Behavior Check:")
    behavior_correct = True
    increasing_trend = True
    
    for i in range(1, len(results)):
        prev_tol, prev_total, _ = results[i-1]
        curr_tol, curr_total, _ = results[i]
        
        change = curr_total - prev_total
        
        if change < 0:
            print(f"❌ WRONG: {prev_tol}% → {curr_tol}% = {prev_total} → {curr_total} (DECREASED by {-change})")
            behavior_correct = False
            increasing_trend = False
        elif change > 0:
            print(f"✅ CORRECT: {prev_tol}% → {curr_tol}% = {prev_total} → {curr_total} (INCREASED by {change})")
        else:
            print(f"🔶 SAME: {prev_tol}% → {curr_tol}% = {prev_total} → {curr_total} (no change)")
    
    # Final assessment
    print("\\n" + "=" * 80)
    print("FINAL ASSESSMENT")
    print("=" * 80)
    
    total_markers_found = sum(total for _, total, _ in results)
    
    if total_markers_found == 0:
        print("❌ CRITICAL: NO MARKERS FOUND AT ALL")
        print("   This suggests a problem with:")
        print("   • Data processing pipeline")
        print("   • Technical indicator calculations") 
        print("   • Local extrema detection")
        print("   • Divergence analysis logic")
        print("   • Server module imports")
        return False
    elif not behavior_correct:
        print("❌ TOLERANCE LOGIC IS STILL WRONG!")
        print("   Higher Candle% should find MORE markers, not fewer")
        print("   The server is using broken divergence analysis code")
        return False
    else:
        print("✅ SUCCESS: TOLERANCE BEHAVIOR IS CORRECT!")
        print(f"   Found {total_markers_found} total markers across all tests")
        if increasing_trend:
            print("   Higher Candle% correctly finds more markers")
        else:
            print("   Higher Candle% correctly finds same or more markers")
        return True

if __name__ == "__main__":
    print("🧪 COMPREHENSIVE WEB SERVER TOLERANCE TEST")
    
    # Test the server
    success = test_server_with_real_data()
    
    if not success:
        print("\\n🔧 NEXT STEPS:")
        print("1. If NO markers found:")
        print("   - Check server console for error messages")
        print("   - Verify all analysis modules are working")
        print("   - Test with different data or parameters")
        print("2. If WRONG tolerance behavior:")
        print("   - Server is using old/broken modules")
        print("   - Clear Python cache and restart server")
        print("   - Verify correct modules are being imported")
        print("3. Check which server you're running:")
        print("   - claude_v2/server_numba.py (recommended)")
        print("   - claude_v1/server.py (may have issues)")
    
    sys.exit(0 if success else 1)