#!/usr/bin/env python3
"""Debug script to test downloading problem tickers."""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.market_data.downloader import download_single_ticker
import pandas as pd

# Problem tickers from the error messages
problem_tickers = ['AAPL', 'MSFT', 'AMZN', 'BRK-B', 'UNH']

print("Testing download for problem tickers...")
print("="*60)

for ticker in problem_tickers:
    print(f"\nTesting {ticker}...")
    try:
        df = download_single_ticker(ticker, '2024-01-01', '2024-12-31')
        if not df.empty:
            print(f"✓ {ticker}: Success! Got {len(df)} rows")
            print(f"  Columns: {list(df.columns)}")
            print(f"  Date range: {df.index[0]} to {df.index[-1]}")
        else:
            print(f"✗ {ticker}: Got empty DataFrame")
    except Exception as e:
        print(f"✗ {ticker}: Failed - {type(e).__name__}: {e}")

print("\n" + "="*60)
print("Debug test complete.") 