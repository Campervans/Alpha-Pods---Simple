#!/usr/bin/env python3
"""
Debug Cache Data
Examines the actual contents of cache files to understand data quality issues.
"""

import sys
import os
from pathlib import Path
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.cache_utils import find_cache_files, load_and_validate_cache

def examine_cache_file(filepath: str, ticker: str):
    """Examine a single cache file in detail."""
    print(f"\n{'='*60}")
    print(f"EXAMINING: {os.path.basename(filepath)}")
    print(f"{'='*60}")
    
    try:
        # Load raw file
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        
        print(f"📊 Basic Info:")
        print(f"   Shape: {df.shape}")
        print(f"   Columns: {list(df.columns)}")
        print(f"   Index type: {type(df.index)}")
        print(f"   Date range: {df.index[0]} to {df.index[-1]}")
        print(f"   Memory usage: {df.memory_usage(deep=True).sum() / 1024:.1f} KB")
        
        print(f"\n📈 Data Quality:")
        print(f"   Missing values: {df.isnull().sum().sum()}")
        print(f"   Duplicate dates: {df.index.duplicated().sum()}")
        
        if 'Close' in df.columns:
            close_prices = df['Close']
            print(f"   Price range: ${close_prices.min():.2f} - ${close_prices.max():.2f}")
            print(f"   Zero/negative prices: {(close_prices <= 0).sum()}")
            print(f"   Price changes >50%: {(close_prices.pct_change().abs() > 0.5).sum()}")
        
        if 'Volume' in df.columns:
            volumes = df['Volume']
            print(f"   Volume range: {volumes.min():,.0f} - {volumes.max():,.0f}")
            print(f"   Zero volume days: {(volumes == 0).sum()}")
            print(f"   Negative volumes: {(volumes < 0).sum()}")
        
        print(f"\n📅 Date Analysis:")
        date_diffs = df.index.to_series().diff().dt.days.dropna()
        print(f"   Typical gap: {date_diffs.mode().iloc[0] if len(date_diffs.mode()) > 0 else 'N/A'} days")
        print(f"   Max gap: {date_diffs.max()} days")
        print(f"   Weekend gaps: {(date_diffs == 3).sum()}")  # Fri to Mon
        print(f"   Large gaps (>7 days): {(date_diffs > 7).sum()}")
        
        print(f"\n🔍 Sample Data (First 5 rows):")
        print(df.head())
        
        print(f"\n🔍 Sample Data (Last 5 rows):")
        print(df.tail())
        
        # Test validation
        print(f"\n✅ Validation Test:")
        is_valid = load_and_validate_cache(filepath, ticker) is not None
        print(f"   Passes validation: {is_valid}")
        
        return df
        
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return None

def main():
    """Main function to debug cache data."""
    print("🔍 Cache Data Debugger")
    print("Examining cache files for data quality issues")
    
    # Test a few representative tickers
    test_tickers = ['AAPL', 'MSFT', 'GOOGL']
    
    for ticker in test_tickers:
        print(f"\n\n🎯 ANALYZING {ticker} CACHE FILES")
        print("="*70)
        
        cache_files = find_cache_files(ticker, "data/raw")
        
        if not cache_files:
            print(f"❌ No cache files found for {ticker}")
            continue
            
        print(f"Found {len(cache_files)} cache files for {ticker}")
        
        # Examine the first (oldest) and last (newest) files
        files_to_check = [cache_files[0]]  # First file
        if len(cache_files) > 1:
            files_to_check.append(cache_files[-1])  # Last file
        
        for file_info in files_to_check:
            examine_cache_file(file_info['filepath'], ticker)
    
    print(f"\n\n🎯 TESTING CACHE LOADING FOR BACKTEST")
    print("="*70)
    
    # Test loading data as the backtest would
    from src.market_data.downloader import download_single_ticker
    
    test_cases = [
        ('AAPL', '2024-01-01', '2024-12-31'),
        ('MSFT', '2023-01-01', '2024-12-31'),
        ('GOOGL', '2020-01-01', '2024-12-31'),
    ]
    
    for ticker, start, end in test_cases:
        print(f"\n📥 Loading {ticker} ({start} to {end}):")
        try:
            df = download_single_ticker(ticker, start, end, use_cache=True)
            if not df.empty:
                print(f"   ✅ Success: {len(df)} rows")
                print(f"   Columns: {list(df.columns)}")
                print(f"   Date range: {df.index[0].date()} to {df.index[-1].date()}")
                
                # Check if this data would work for backtesting
                required_cols = ['Close', 'Volume']
                has_required = all(col in df.columns for col in required_cols)
                print(f"   Has required columns: {has_required}")
                
                if has_required:
                    print(f"   Price range: ${df['Close'].min():.2f} - ${df['Close'].max():.2f}")
                    print(f"   Volume range: {df['Volume'].min():,.0f} - {df['Volume'].max():,.0f}")
                
            else:
                print(f"   ❌ Failed: Empty DataFrame returned")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")

if __name__ == "__main__":
    main() 