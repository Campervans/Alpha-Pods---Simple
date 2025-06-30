#!/usr/bin/env python3
"""
Test script to verify proxy implementation in yfinance downloads.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.market_data.downloader import download_single_ticker, get_random_proxy, test_proxy
import pandas as pd
import requests

def test_proxy_connectivity():
    """Test if proxies are working."""
    print("Testing proxy connectivity...")
    print("=" * 60)
    
    # Test a few random proxies
    for i in range(3):
        proxy_dict = get_random_proxy()
        proxy_url = proxy_dict['http']
        port = proxy_url.split(':')[-1]
        
        print(f"\nTesting proxy on port {port}...")
        
        # Test proxy
        is_working = test_proxy(proxy_dict)
        
        if is_working:
            # Get IP info
            try:
                response = requests.get('https://ip.decodo.com/json', 
                                      proxies=proxy_dict, 
                                      timeout=10)
                ip_info = response.json()
                print(f"✓ Proxy working! IP: {ip_info.get('ip', 'Unknown')}")
                print(f"  Location: {ip_info.get('country', 'Unknown')}, {ip_info.get('city', 'Unknown')}")
            except Exception as e:
                print(f"✓ Proxy working but couldn't get IP details: {e}")
        else:
            print(f"✗ Proxy not working")


def test_ticker_download():
    """Test downloading ticker data with proxy."""
    print("\n" + "=" * 60)
    print("Testing ticker downloads with proxy...")
    print("=" * 60)
    
    test_tickers = ['AAPL', 'MSFT', 'GOOGL']
    start_date = '2024-01-01'
    end_date = '2024-12-31'
    
    for ticker in test_tickers:
        print(f"\nDownloading {ticker} from {start_date} to {end_date}...")
        
        # Download with proxy retry logic
        data = download_single_ticker(ticker, start_date, end_date)
        
        if not data.empty:
            print(f"✓ Successfully downloaded {len(data)} days of data")
            print(f"  Date range: {data.index[0].date()} to {data.index[-1].date()}")
            print(f"  Latest close: ${data['Close'].iloc[-1]:.2f}")
        else:
            print(f"✗ Failed to download data")


def test_no_proxy_fallback():
    """Test that fallback to no proxy works."""
    print("\n" + "=" * 60)
    print("Testing fallback to no-proxy mode...")
    print("=" * 60)
    
    # Use a ticker that should work
    ticker = 'SPY'
    start_date = '2024-11-01'
    end_date = '2024-12-31'
    
    print(f"\nDownloading {ticker} (should eventually fallback to no proxy)...")
    
    data = download_single_ticker(ticker, start_date, end_date)
    
    if not data.empty:
        print(f"✓ Successfully downloaded {len(data)} days of data")
        print(f"  This may have used a proxy or fallen back to no proxy")
    else:
        print(f"✗ Failed to download data even with fallback")


def main():
    """Run all proxy tests."""
    print("PROXY IMPLEMENTATION TEST")
    print("=" * 60)
    print("Testing proxy support for yfinance downloads\n")
    
    # Test proxy connectivity
    test_proxy_connectivity()
    
    # Test ticker downloads
    test_ticker_download()
    
    # Test no-proxy fallback
    test_no_proxy_fallback()
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main() 