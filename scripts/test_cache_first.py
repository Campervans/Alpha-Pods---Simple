#!/usr/bin/env python3
"""
Test Cache-First Loading
Verifies that cached tickers load quickly without network requests.
"""

import sys
import time
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.market_data.downloader import download_single_ticker

def test_cache_performance():
    """Test that cached tickers load much faster than downloads."""
    print("Testing Cache-First Loading Performance")
    print("=" * 50)
    
    # Test 1: Load cached ticker (should be very fast)
    print("\n🔬 Test 1: Load Cached Ticker")
    print("Ticker: AAPL (should exist in cache)")
    
    start_time = time.time()
    try:
        df = download_single_ticker('AAPL', '2023-01-01', '2024-12-31', use_cache=True)
        load_time = time.time() - start_time
        
        print(f"✅ Success!")
        print(f"   Load time: {load_time:.3f} seconds")
        print(f"   Data points: {len(df):,}")
        print(f"   Date range: {df.index[0].date()} to {df.index[-1].date()}")
        
        # Cache load should be very fast (< 0.5 seconds)
        if load_time < 0.5:
            print(f"   🚀 FAST! (Cache hit)")
        else:
            print(f"   🐌 Slow (likely downloaded from internet)")
            
    except Exception as e:
        print(f"❌ Failed: {e}")
    
    # Test 2: Load another cached ticker
    print("\n🔬 Test 2: Load Another Cached Ticker")
    print("Ticker: MSFT (should exist in cache)")
    
    start_time = time.time()
    try:
        df = download_single_ticker('MSFT', '2023-01-01', '2024-12-31', use_cache=True)
        load_time = time.time() - start_time
        
        print(f"✅ Success!")
        print(f"   Load time: {load_time:.3f} seconds")
        print(f"   Data points: {len(df):,}")
        
        if load_time < 0.5:
            print(f"   🚀 FAST! (Cache hit)")
        else:
            print(f"   🐌 Slow (likely downloaded)")
            
    except Exception as e:
        print(f"❌ Failed: {e}")
    
    # Test 3: Test cache bypass (force download)
    print("\n🔬 Test 3: Force Download (Cache Bypass)")
    print("Ticker: AAPL with cache disabled")
    
    start_time = time.time()
    try:
        df = download_single_ticker('AAPL', '2024-12-01', '2024-12-31', use_cache=False)
        load_time = time.time() - start_time
        
        print(f"✅ Success!")
        print(f"   Load time: {load_time:.3f} seconds")
        print(f"   Data points: {len(df):,}")
        
        if load_time > 1.0:
            print(f"   🌐 SLOW (Downloaded from internet)")
        else:
            print(f"   ⚡ Fast (unexpected - might be network cache)")
            
    except Exception as e:
        print(f"❌ Failed: {e}")
    
    # Test 4: Test non-existent ticker (should fail gracefully)
    print("\n🔬 Test 4: Non-existent Ticker")
    print("Ticker: FAKECORP (should not exist)")
    
    start_time = time.time()
    try:
        df = download_single_ticker('FAKECORP', '2024-01-01', '2024-12-31', use_cache=True)
        load_time = time.time() - start_time
        
        if df.empty:
            print(f"✅ Correctly returned empty DataFrame")
            print(f"   Load time: {load_time:.3f} seconds")
        else:
            print(f"⚠️  Unexpected: Got data for fake ticker")
            
    except Exception as e:
        print(f"❌ Failed: {e}")

def test_cache_coverage():
    """Test cache coverage for different date ranges."""
    print("\n\n📊 Testing Cache Coverage")
    print("=" * 50)
    
    test_cases = [
        ('AAPL', '2024-01-01', '2024-12-31', "Recent year"),
        ('AAPL', '2020-01-01', '2024-12-31', "Multi-year"),
        ('AAPL', '2023-06-01', '2023-06-30', "One month"),
        ('GOOGL', '2024-01-01', '2024-12-31', "Different ticker"),
    ]
    
    for ticker, start, end, description in test_cases:
        print(f"\n🔍 {description}: {ticker} ({start} to {end})")
        
        start_time = time.time()
        try:
            df = download_single_ticker(ticker, start, end, use_cache=True)
            load_time = time.time() - start_time
            
            if not df.empty:
                print(f"   ✅ Got {len(df):,} data points in {load_time:.3f}s")
                if load_time < 0.5:
                    print(f"   📁 Loaded from cache")
                else:
                    print(f"   🌐 Downloaded from internet")
            else:
                print(f"   ❌ No data returned")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")

def main():
    """Run all tests."""
    print("🧪 Cache-First Loading Test Suite")
    print("Testing cache performance and functionality")
    
    # Test basic performance
    test_cache_performance()
    
    # Test different scenarios
    test_cache_coverage()
    
    print("\n" + "=" * 50)
    print("✅ Cache-First Loading Tests Complete!")
    print("\nExpected Results:")
    print("- Cached tickers should load in < 0.5 seconds")
    print("- Downloads should take > 1.0 seconds")
    print("- Cache messages should appear in output")
    print("- Non-existent tickers should fail gracefully")

if __name__ == "__main__":
    main() 