"""
Debug script to test yfinance data download functionality.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import yfinance as yf
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def test_single_download(ticker="AAPL", start="2023-01-01", end="2024-12-31"):
    """Test downloading data for a single ticker."""
    print(f"\n{'='*50}")
    print(f"Testing download for {ticker}")
    print(f"Date range: {start} to {end}")
    print(f"{'='*50}")
    
    try:
        # Method 1: Using yf.download
        print("\nMethod 1: Using yf.download()...")
        data = yf.download(ticker, start=start, end=end, progress=False)
        print(f"Downloaded {len(data)} rows")
        if not data.empty:
            print(f"Columns: {list(data.columns)}")
            print(f"First date: {data.index[0]}")
            print(f"Last date: {data.index[-1]}")
            print("\nFirst few rows:")
            print(data.head())
        else:
            print("No data returned!")
            
        # Method 2: Using Ticker object
        print("\n\nMethod 2: Using yf.Ticker()...")
        ticker_obj = yf.Ticker(ticker)
        hist = ticker_obj.history(start=start, end=end)
        print(f"Downloaded {len(hist)} rows")
        if not hist.empty:
            print(f"Columns: {list(hist.columns)}")
            print(f"First date: {hist.index[0]}")
            print(f"Last date: {hist.index[-1]}")
            print("\nFirst few rows:")
            print(hist.head())
        else:
            print("No data returned!")
            
        # Check ticker info
        print("\n\nTicker Info:")
        info = ticker_obj.info
        if info:
            print(f"Company: {info.get('longName', 'N/A')}")
            print(f"Exchange: {info.get('exchange', 'N/A')}")
            print(f"Currency: {info.get('currency', 'N/A')}")
        else:
            print("No info available!")
            
    except Exception as e:
        print(f"\nError occurred: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

def test_multiple_downloads():
    """Test downloading multiple tickers."""
    print(f"\n\n{'='*50}")
    print("Testing multiple ticker download")
    print(f"{'='*50}")
    
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    
    try:
        # Method 1: Download all at once
        print("\nMethod 1: Downloading all at once...")
        data = yf.download(tickers, start="2023-01-01", end="2024-12-31", 
                          group_by='ticker', progress=False)
        print(f"Downloaded data shape: {data.shape}")
        if not data.empty:
            print(f"Tickers in data: {list(data.columns.levels[0]) if hasattr(data.columns, 'levels') else 'Single level'}")
        
        # Method 2: Download one by one
        print("\nMethod 2: Downloading one by one...")
        results = {}
        for ticker in tickers:
            try:
                ticker_data = yf.download(ticker, start="2023-01-01", end="2024-12-31", 
                                        progress=False)
                if not ticker_data.empty:
                    results[ticker] = len(ticker_data)
                    print(f"  {ticker}: {len(ticker_data)} rows")
                else:
                    print(f"  {ticker}: No data")
            except Exception as e:
                print(f"  {ticker}: Error - {e}")
                
    except Exception as e:
        print(f"\nError occurred: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

def check_yfinance_version():
    """Check yfinance version and configuration."""
    print(f"\n{'='*50}")
    print("yfinance Configuration")
    print(f"{'='*50}")
    
    print(f"yfinance version: {yf.__version__}")
    print(f"Python version: {sys.version}")
    
    # Test if we can access Yahoo Finance
    print("\nTesting Yahoo Finance connectivity...")
    try:
        ticker = yf.Ticker("AAPL")
        info = ticker.info
        if info:
            print("✓ Successfully connected to Yahoo Finance")
        else:
            print("✗ Connected but no data returned")
    except Exception as e:
        print(f"✗ Connection failed: {e}")

if __name__ == "__main__":
    print("YFINANCE DEBUG SCRIPT")
    print("=" * 50)
    
    # Check version and connectivity
    check_yfinance_version()
    
    # Test single ticker download
    test_single_download("AAPL", "2023-01-01", "2024-12-31")
    
    # Test multiple ticker download
    test_multiple_downloads()
    
    print("\n\nDebug script completed!")
    
    # Leave debug info commented for reference
    """
    # Additional debug commands to try if issues persist:
    
    # 1. Test with different date ranges
    # test_single_download("AAPL", "2024-01-01", "2024-06-30")
    
    # 2. Test with auto_adjust=False
    # data = yf.download("AAPL", start="2024-01-01", end="2024-06-30", auto_adjust=False)
    
    # 3. Test with different intervals
    # data = yf.download("AAPL", period="1mo", interval="1d")
    
    # 4. Check for rate limiting
    # import time
    # for ticker in ["AAPL", "MSFT", "GOOGL"]:
    #     data = yf.download(ticker, period="1mo")
    #     print(f"{ticker}: {len(data)} rows")
    #     time.sleep(1)  # Add delay between requests
    """ 