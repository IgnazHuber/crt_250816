#!/usr/bin/env python3
"""
Quick test to verify which modules the web server is actually using
and test the tolerance behavior directly
"""

import sys
import os
import requests
import json
import time
from pathlib import Path

def test_server_endpoint(server_url="http://localhost:5000"):
    """Test the actual web server endpoint with different Candle% values"""
    print("=" * 80)
    print("TESTING WEB SERVER TOLERANCE BEHAVIOR")
    print("=" * 80)
    
    # Test data - simple CSV content
    test_csv_content = """date,open,high,low,close,volume
2023-01-01 00:00:00,50000,50100,49900,50050,1000
2023-01-01 01:00:00,50050,50150,49950,50100,1000
2023-01-01 02:00:00,50100,50200,50000,50150,1000
2023-01-01 03:00:00,50150,50250,50050,50200,1000
2023-01-01 04:00:00,50200,50300,50100,50250,1000
2023-01-01 05:00:00,50250,50350,50150,50300,1000
2023-01-01 06:00:00,50300,50250,50150,50200,1000
2023-01-01 07:00:00,50200,50150,50050,50100,1000
2023-01-01 08:00:00,50100,50050,49950,50000,1000
2023-01-01 09:00:00,50000,49950,49850,49900,1000
2023-01-01 10:00:00,49900,49850,49750,49800,1000
2023-01-01 11:00:00,49800,49900,49700,49850,1000
2023-01-01 12:00:00,49850,49950,49750,49900,1000
2023-01-01 13:00:00,49900,50000,49800,49950,1000
2023-01-01 14:00:00,49950,50050,49850,50000,1000
2023-01-01 15:00:00,50000,50100,49900,50050,1000
2023-01-01 16:00:00,50050,50150,49950,50100,1000
2023-01-01 17:00:00,50100,50200,50000,50150,1000
2023-01-01 18:00:00,50150,50250,50050,50200,1000
2023-01-01 19:00:00,50200,50300,50100,50250,1000
"""

    # Check if server is running
    try:
        response = requests.get(f"{server_url}/api/health", timeout=5)
        if response.status_code == 200:
            print(f"✅ Server is running at {server_url}")
            health_data = response.json()
            print(f"   Status: {health_data.get('status')}")
            print(f"   Message: {health_data.get('message')}")
        else:
            print(f"❌ Server health check failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot connect to server at {server_url}")
        print(f"   Error: {e}")
        print("   Make sure the server is running with: python server.py")
        return False

    print("\\nTesting tolerance behavior with different Candle% values...")
    
    # Test different Candle% values
    candle_tolerances = [0.5, 1.0, 2.0, 4.0, 8.0]
    results = []
    
    for candle_tol in candle_tolerances:
        print(f"\\nTesting Candle% = {candle_tol}%...")
        
        try:
            # Upload test data
            files = {'file': ('test_data.csv', test_csv_content, 'text/csv')}
            upload_response = requests.post(f"{server_url}/api/upload", files=files, timeout=10)
            
            if upload_response.status_code != 200:
                print(f"❌ Upload failed: {upload_response.status_code}")
                continue
                
            upload_data = upload_response.json()
            if not upload_data.get('success'):
                print(f"❌ Upload error: {upload_data.get('error')}")
                continue
                
            session_id = upload_data.get('session_id')
            print(f"   Session ID: {session_id}")
            
            # Run analysis
            analysis_data = {
                'session_id': session_id,
                'window': 14,
                'candle_tol': candle_tol,
                'macd_tol': 0.001
            }
            
            analysis_response = requests.post(f"{server_url}/api/analyze", 
                                            json=analysis_data, 
                                            timeout=30)
            
            if analysis_response.status_code != 200:
                print(f"❌ Analysis failed: {analysis_response.status_code}")
                continue
                
            analysis_result = analysis_response.json()
            if not analysis_result.get('success'):
                print(f"❌ Analysis error: {analysis_result.get('error')}")
                continue
                
            # Count markers
            markers = analysis_result.get('markers', {})
            total_markers = 0
            
            for variant_name, variant_markers in markers.items():
                count = len(variant_markers)
                total_markers += count
                print(f"   {variant_name}: {count} markers")
                
            print(f"   Total markers: {total_markers}")
            results.append((candle_tol, total_markers))
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Request failed: {e}")
            continue
    
    # Analyze results
    print("\\n" + "=" * 80)
    print("TOLERANCE BEHAVIOR ANALYSIS")
    print("=" * 80)
    
    if len(results) < 2:
        print("❌ Not enough results to analyze behavior")
        return False
    
    print("\\nCandle% | Total Markers")
    print("--------|-------------")
    for candle_tol, total in results:
        print(f"{candle_tol:7.1f} | {total:12d}")
    
    print("\\nBehavior Analysis:")
    behavior_correct = True
    
    for i in range(1, len(results)):
        prev_tol, prev_total = results[i-1]
        curr_tol, curr_total = results[i]
        
        if curr_total < prev_total:
            print(f"❌ WRONG: {prev_tol}% -> {curr_tol}% = {prev_total} -> {curr_total} (DECREASED!)")
            behavior_correct = False
        elif curr_total > prev_total:
            print(f"✅ CORRECT: {prev_tol}% -> {curr_tol}% = {prev_total} -> {curr_total} (INCREASED)")
        else:
            print(f"✅ OK: {prev_tol}% -> {curr_tol}% = {prev_total} -> {curr_total} (SAME)")
    
    print("\\n" + "=" * 80)
    if behavior_correct:
        print("✅ SUCCESS: Web server tolerance behavior is CORRECT!")
        print("   Higher Candle% finds same or more markers (as expected)")
    else:
        print("❌ FAILURE: Web server tolerance behavior is WRONG!")
        print("   Higher Candle% should find same or more markers")
        print("   The server may be using an old/broken module")
    
    return behavior_correct

def find_running_servers():
    """Try to find which server is running"""
    possible_ports = [5000, 5001, 5002, 8000, 8080]
    found_servers = []
    
    print("\\nScanning for running servers...")
    
    for port in possible_ports:
        try:
            response = requests.get(f"http://localhost:{port}/api/health", timeout=2)
            if response.status_code == 200:
                found_servers.append(f"http://localhost:{port}")
                print(f"✅ Found server at localhost:{port}")
        except:
            pass
    
    if not found_servers:
        print("❌ No servers found running")
        print("\\nTo start the server, run one of these:")
        print("   cd C:\\Projekte\\crt_250816\\scripts\\claude_v2 && python server_numba.py")
        print("   cd C:\\Projekte\\crt_250816\\scripts\\claude_v1 && python server.py")
    
    return found_servers

if __name__ == "__main__":
    print("🔍 WEB SERVER TOLERANCE VERIFICATION")
    
    # Find running servers
    servers = find_running_servers()
    
    if not servers:
        print("\\nPlease start a server and try again.")
        sys.exit(1)
    
    # Test the first found server
    server_url = servers[0]
    success = test_server_endpoint(server_url)
    
    if not success:
        print("\\n🔧 TROUBLESHOOTING STEPS:")
        print("1. Make sure you're running the correct server:")
        print("   - For fixed version: python server_numba.py (in claude_v2)")
        print("   - Check server console for error messages")
        print("2. Clear browser cache and reload the page")
        print("3. Restart the server to ensure new code is loaded")
        print("4. Verify the server is importing the correct modules")
    
    sys.exit(0 if success else 1)