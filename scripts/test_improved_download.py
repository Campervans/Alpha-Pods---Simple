"""
Test script for improved data download with rate limiting and caching.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.market_data.downloader import download_universe, create_sp100_list
from src.utils.schemas import PriceData
import pandas as pd
from datetime import datetime, timedelta

def test_download_with_cache():
    """Test downloading data with caching enabled."""
    print("=" * 60)
    print("TESTING IMPROVED DATA DOWNLOAD")
    print("=" * 60)
    
    # Test with a small set of tickers first
    test_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    
    # Date range - last 2 years
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
    
    print(f"\nTest 1: Downloading {len(test_tickers)} tickers")
    print(f"Date range: {start_date} to {end_date}")
    print(f"Cache directory: data/raw")
    
    try:
        # First download - will fetch from Yahoo Finance
        print("\n--- First Download (from Yahoo Finance) ---")
        start_time = datetime.now()
        
        price_data = download_universe(
            tickers=test_tickers,
            start=start_date,
            end=end_date,
            min_data_points=100,
            max_workers=3,  # Reduced to avoid rate limits
            use_cache=True,
            cache_dir="data/raw"
        )
        
        download_time = (datetime.now() - start_time).total_seconds()
        print(f"\nFirst download completed in {download_time:.2f} seconds")
        print(f"Downloaded {price_data.n_assets} assets with {len(price_data.dates)} days of data")
        
        # Second download - should use cache
        print("\n--- Second Download (from cache) ---")
        start_time = datetime.now()
        
        price_data_cached = download_universe(
            tickers=test_tickers,
            start=start_date,
            end=end_date,
            min_data_points=100,
            max_workers=3,
            use_cache=True,
            cache_dir="data/raw"
        )
        
        cache_time = (datetime.now() - start_time).total_seconds()
        print(f"\nCache load completed in {cache_time:.2f} seconds")
        print(f"Speed improvement: {download_time/cache_time:.1f}x faster")
        
        # Verify data integrity
        print("\n--- Data Verification ---")
        print(f"Assets match: {set(price_data.tickers) == set(price_data_cached.tickers)}")
        print(f"Dates match: {len(price_data.dates) == len(price_data_cached.dates)}")
        
        # Show sample data
        print("\n--- Sample Data ---")
        print(f"Tickers: {price_data.tickers}")
        print(f"Date range: {price_data.start_date.date()} to {price_data.end_date.date()}")
        print(f"\nFirst few prices:")
        print(price_data.prices.head())
        
        # Check cache directory
        print("\n--- Cache Files ---")
        cache_dir = "data/raw"
        if os.path.exists(cache_dir):
            files = [f for f in os.listdir(cache_dir) if f.endswith('.csv')]
            print(f"Found {len(files)} cached files:")
            for f in sorted(files)[:5]:  # Show first 5
                print(f"  - {f}")
            if len(files) > 5:
                print(f"  ... and {len(files) - 5} more")
        
    except Exception as e:
        print(f"\nError during test: {e}")
        import traceback
        traceback.print_exc()
        
def test_rate_limit_handling():
    """Test rate limit handling with larger ticker list."""
    print("\n\n" + "="*60)
    print("TESTING RATE LIMIT HANDLING")
    print("="*60)
    
    # Get more tickers
    sp100_tickers = create_sp100_list()[:20]  # Test with 20 tickers
    
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    
    print(f"\nTest 2: Downloading {len(sp100_tickers)} tickers with rate limit protection")
    print(f"Date range: {start_date} to {end_date}")
    
    try:
        start_time = datetime.now()
        
        price_data = download_universe(
            tickers=sp100_tickers,
            start=start_date,
            end=end_date,
            min_data_points=100,
            max_workers=3,  # Low to avoid rate limits
            use_cache=True,
            cache_dir="data/raw"
        )
        
        download_time = (datetime.now() - start_time).total_seconds()
        print(f"\nDownload completed in {download_time:.2f} seconds")
        print(f"Successfully downloaded {price_data.n_assets} out of {len(sp100_tickers)} tickers")
        print(f"Average time per ticker: {download_time/len(sp100_tickers):.2f} seconds")
        
    except Exception as e:
        print(f"\nError during test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("IMPROVED DATA DOWNLOAD TEST")
    print("Features: Rate limiting, exponential backoff, caching")
    print()
    
    # Create data directory if it doesn't exist
    os.makedirs("data/raw", exist_ok=True)
    
    # Run tests
    test_download_with_cache()
    test_rate_limit_handling()
    
    print("\n" + "="*60)
    print("TESTS COMPLETED!")
    print("="*60)
    
    # Debug info - leave commented
    """
    # To clear cache for testing:
    # import shutil
    # if os.path.exists("data/raw"):
    #     shutil.rmtree("data/raw")
    #     os.makedirs("data/raw")
    
    # To test with specific problem tickers:
    # problem_tickers = ['BRK-B', 'BF-B']  # Tickers with special characters
    # data = download_universe(problem_tickers, start_date, end_date)
    """ 