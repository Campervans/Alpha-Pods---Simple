# market data downloader using yfinance

import yfinance as yf
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from datetime import datetime
import random
import requests

from ..utils.schemas import PriceData

# PROXY CONFIGURATION - CREDENTIALS ARE INTENTIONALLY HARDCODED, DO NOT CHANGE
PROXY_USERNAME = 'sp7lr99xhd'
PROXY_PASSWORD = '7Xtywa2k3o0oxoViLX'

# Load proxy ports from CSV - now supports 100 proxies instead of hardcoded 30
from ..utils.proxy_utils import load_proxies_from_csv
PROXY_PORTS = load_proxies_from_csv()


def get_random_proxy():
    """Get a random proxy from the list."""
    port = random.choice(PROXY_PORTS)
    proxy = f"http://{PROXY_USERNAME}:{PROXY_PASSWORD}@dc.decodo.com:{port}"
    return {
        'http': proxy,
        'https': proxy
    }


def test_proxy(proxy_dict):
    """Test if a proxy is working."""
    try:
        response = requests.get('https://ip.decodo.com/json', 
                              proxies=proxy_dict, 
                              timeout=5)
        return response.status_code == 200
    except:
        return False


def download_single_ticker(ticker: str, start: str, end: str, 
                          retry_count: int = 3, delay: float = 1.0,
                          use_cache: bool = True, cache_dir: str = "data/raw") -> pd.DataFrame:
    # download one ticker with retries and rate limit handling
    
    # STEP 2: Smart cache check with validation
    if use_cache:
        # No longer need complex validation, pickle format preserves data types
        cached_df = load_ticker_data_from_pickle(ticker, cache_dir)
        
        if cached_df is not None:
            # Check if the cached data covers the requested date range
            start_date = pd.to_datetime(start)
            end_date = pd.to_datetime(end)
            
            # Ensure index is a DatetimeIndex before proceeding
            if isinstance(cached_df.index, pd.DatetimeIndex):
                # More flexible date range check - allow for market holidays at start
                # Check if cache covers the requested period (with some tolerance for holidays)
                cache_start = cached_df.index[0]
                cache_end = cached_df.index[-1]
                
                # Allow up to 5 days difference at start (for holidays/weekends)
                start_ok = cache_start <= start_date + pd.Timedelta(days=5)
                end_ok = cache_end >= end_date - pd.Timedelta(days=5)
                
                if not cached_df.empty and start_ok and end_ok:
                    # Filter to the exact requested date range
                    mask = (cached_df.index >= start_date) & (cached_df.index <= end_date)
                    filtered_data = cached_df.loc[mask]
                    
                    if not filtered_data.empty:
                        print(f"✓ {ticker} loaded from pickle cache ({len(filtered_data)} days)")
                        return filtered_data
            else:
                print(f"⚠️  {ticker}: Cached data has an invalid index type, re-downloading.")
    
    # If cache miss or invalid, proceed with download
    # Save original proxy environment variables
    original_http_proxy = os.environ.get('HTTP_PROXY', '')
    original_https_proxy = os.environ.get('HTTPS_PROXY', '')
    
    # Try with different proxies first, then fallback to no proxy
    proxy_attempts = min(5, len(PROXY_PORTS))  # Try up to 5 different proxies
    
    for proxy_attempt in range(proxy_attempts + 1):  # +1 for no-proxy attempt
        if proxy_attempt < proxy_attempts:
            # Try with a proxy
            proxy_dict = get_random_proxy()
            proxy_url = proxy_dict['http']
            proxy_info = f"proxy port {proxy_url.split(':')[-1]}"
            
            # Set environment variables for proxy
            os.environ['HTTP_PROXY'] = proxy_url
            os.environ['HTTPS_PROXY'] = proxy_url
            os.environ['http_proxy'] = proxy_url  # Some systems use lowercase
            os.environ['https_proxy'] = proxy_url
        else:
            # Last attempt without proxy - clear proxy env vars
            os.environ['HTTP_PROXY'] = ''
            os.environ['HTTPS_PROXY'] = ''
            os.environ['http_proxy'] = ''
            os.environ['https_proxy'] = ''
            proxy_info = "no proxy"
        
        for attempt in range(retry_count):
            try:
                # quiet down yfinance
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    
                # Add delay between requests to avoid rate limiting
                if attempt > 0 or proxy_attempt > 0:
                    time.sleep(delay * (attempt + 1))  # Exponential backoff
                    
                # Let yfinance handle the session internally
                try:
                    df = yf.download(
                        ticker, 
                        start=start, 
                        end=end, 
                        auto_adjust=True,
                        progress=False
                    )
                except Exception as e:
                    import traceback
                    print(f"\n{'='*60}")
                    print(f"ERROR downloading {ticker}")
                    print(f"Error type: {type(e).__name__}")
                    print(f"Error message: {e}")
                    print(f"Full traceback:")
                    traceback.print_exc()
                    print(f"{'='*60}\n")
                    raise
                
                # Debug print to see what we got (commented out for production)
                # if hasattr(df, 'shape'):
                #     print(f"Downloaded {ticker}: shape={df.shape}, empty={df.empty}")
                #     if hasattr(df, 'columns'):
                #         print(f"Columns type: {type(df.columns)}, values: {list(df.columns)[:5] if len(df.columns) > 0 else 'empty'}")
                
                if df.empty:
                    if attempt < retry_count - 1:
                        time.sleep(delay)
                        continue
                    # Try next proxy
                    break
                
                # Handle multi-level columns (when ticker is in column names)
                if isinstance(df.columns, pd.MultiIndex):
                    # print(f"{ticker} has MultiIndex columns: levels={df.columns.levels}")
                    # More robust handling
                    if len(df.columns.levels) > 1 and ticker in df.columns.get_level_values(1):
                        # If ticker is in the second level, select it
                        df = df.xs(ticker, level=1, axis=1)
                    elif len(set(df.columns.get_level_values(1))) == 1:
                        # If only one ticker in second level, just drop it
                        df.columns = df.columns.droplevel(1)
                    else:
                        # Try to extract just the price/volume data
                        df.columns = df.columns.droplevel(1)
                
                # get close & volume
                required_cols = ['Close', 'Volume']
                if all(col in df.columns for col in required_cols):
                    result = df[required_cols].copy()
                    
                    # clean up
                    result = result.dropna()
                    
                    # vol should be positive
                    result['Volume'] = result['Volume'].clip(lower=0)
                    
                    # prices too
                    if (result['Close'] <= 0).any():
                        print(f"X {ticker} has weird prices")  # happens sometimes with bad data
                        result = result[result['Close'] > 0]
                    
                    # Success! Log which method worked
                    if proxy_attempt < proxy_attempts:
                        print(f"✓ {ticker} downloaded successfully using {proxy_info}")
                    
                    # Restore original proxy settings before returning
                    os.environ['HTTP_PROXY'] = original_http_proxy
                    os.environ['HTTPS_PROXY'] = original_https_proxy
                    os.environ['http_proxy'] = original_http_proxy
                    os.environ['https_proxy'] = original_https_proxy
                    
                    return result
                else:
                    if attempt < retry_count - 1:
                        time.sleep(delay)
                        continue
                    # Try next proxy
                    break
                    
            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg or "Too Many Requests" in error_msg:
                    # Rate limit error - wait longer
                    wait_time = delay * (2 ** (attempt + 1))  # Exponential backoff
                    print(f"Rate limit hit for {ticker} using {proxy_info}, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                elif attempt < retry_count - 1:
                    # Other error - normal retry
                    time.sleep(delay)
                    continue
                else:
                    # Last attempt for this proxy failed, try next proxy
                    if proxy_attempt < proxy_attempts:
                        print(f"Failed {ticker} with {proxy_info}: {e}, trying different proxy...")
                    else:
                        print(f"Failed {ticker} with all methods: {e}")
                    break
    
    # Restore original proxy settings before returning
    os.environ['HTTP_PROXY'] = original_http_proxy
    os.environ['HTTPS_PROXY'] = original_https_proxy
    os.environ['http_proxy'] = original_http_proxy
    os.environ['https_proxy'] = original_https_proxy
    
    return pd.DataFrame()


def download_multiple_tickers(tickers: List[str], start: str, end: str, 
                            max_workers: int = 5, progress_bar: bool = True,
                            batch_delay: float = 0.5) -> Dict[str, pd.DataFrame]:
    """
    Download multiple tickers using yfinance's thread-safe batch downloading.
    This avoids the race condition issues with yfinance's shared global dictionary.
    """
    results = {}
    
    # Process in smaller batches to avoid rate limiting
    batch_size = 20  # Can use larger batches since yfinance handles threading internally
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        
        if progress_bar:
            print(f"Processing batch {i//batch_size + 1}/{(len(tickers) + batch_size - 1)//batch_size}: {len(batch)} tickers")
        
        try:
            # Use yfinance's built-in batch download which is thread-safe
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                # Download all tickers in batch using yfinance's internal threading
                batch_data = yf.download(
                    batch,  # Pass list of tickers
                    start=start,
                    end=end,
                    auto_adjust=True,
                    progress=progress_bar,
                    group_by='ticker'  # Important: group by ticker to get separate columns
                )
            
            # Process the batch results
            if not batch_data.empty:
                # Handle single ticker case (no MultiIndex)
                if len(batch) == 1:
                    ticker = batch[0]
                    if not batch_data.empty and 'Close' in batch_data.columns and 'Volume' in batch_data.columns:
                        result = batch_data[['Close', 'Volume']].copy()
                        result = result.dropna()
                        result['Volume'] = result['Volume'].clip(lower=0)
                        if (result['Close'] > 0).all():
                            results[ticker] = result
                            if progress_bar:
                                print(f"✓ {ticker}: {len(result)} days")
                        else:
                            print(f"✗ {ticker}: Invalid price data")
                    else:
                        print(f"✗ {ticker}: Missing required columns")
                
                # Handle multiple tickers case (MultiIndex columns)
                elif isinstance(batch_data.columns, pd.MultiIndex):
                    for ticker in batch:
                        try:
                            # Extract data for this ticker
                            ticker_data = batch_data[ticker] if ticker in batch_data.columns.get_level_values(0) else None
                            
                            if ticker_data is not None and not ticker_data.empty:
                                if 'Close' in ticker_data.columns and 'Volume' in ticker_data.columns:
                                    result = ticker_data[['Close', 'Volume']].copy()
                                    result = result.dropna()
                                    result['Volume'] = result['Volume'].clip(lower=0)
                                    if len(result) > 0 and (result['Close'] > 0).all():
                                        results[ticker] = result
                                        if progress_bar:
                                            print(f"✓ {ticker}: {len(result)} days")
                                    else:
                                        print(f"✗ {ticker}: Invalid price data")
                                else:
                                    print(f"✗ {ticker}: Missing required columns")
                            else:
                                print(f"✗ {ticker}: No data returned")
                        except Exception as e:
                            print(f"✗ {ticker}: Error processing - {e}")
                
                else:
                    print(f"Unexpected data structure for batch: {batch}")
            
            else:
                print(f"No data returned for batch: {batch}")
                
        except Exception as e:
            print(f"Error downloading batch {batch}: {e}")
            # Fallback to individual downloads for this batch
            print(f"Falling back to individual downloads for batch...")
            for ticker in batch:
                try:
                    data = download_single_ticker(ticker, start, end, use_cache=True, cache_dir="data/raw")
                    if not data.empty:
                        results[ticker] = data
                        if progress_bar:
                            print(f"✓ {ticker}: {len(data)} days (fallback)")
                    else:
                        print(f"✗ {ticker}: No data (fallback)")
                except Exception as fallback_e:
                    print(f"✗ {ticker}: Error in fallback - {fallback_e}")
        
        # Delay between batches to avoid rate limiting
        if i + batch_size < len(tickers):
            time.sleep(batch_delay)
    
    return results


def align_data_by_dates(data_dict: Dict[str, pd.DataFrame], 
                       min_data_points: int = 100) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Align price and volume data across tickers by common dates.
    
    Args:
        data_dict: Dictionary mapping ticker to DataFrame
        min_data_points: Minimum number of data points required per ticker
        
    Returns:
        Tuple of (price_df, volume_df) with aligned dates
        
    Example:
        >>> data_dict = {'AAPL': df1, 'MSFT': df2}
        >>> prices, volumes = align_data_by_dates(data_dict, min_data_points=100)
    """
    if not data_dict:
        return pd.DataFrame(), pd.DataFrame()
    
    # Filter tickers with sufficient data
    valid_tickers = [
        ticker for ticker, df in data_dict.items()
        if len(df) >= min_data_points
    ]
    
    if not valid_tickers:
        print(f"No tickers have at least {min_data_points} data points")
        return pd.DataFrame(), pd.DataFrame()
    
    print(f"Using {len(valid_tickers)} tickers with sufficient data")
    
    # Find common date range
    start_dates = [data_dict[ticker].index.min() for ticker in valid_tickers]
    end_dates = [data_dict[ticker].index.max() for ticker in valid_tickers]
    
    common_start = max(start_dates)
    common_end = min(end_dates)
    
    print(f"Common date range: {common_start.date()} to {common_end.date()}")
    
    # Create aligned DataFrames
    price_data = {}
    volume_data = {}
    
    for ticker in valid_tickers:
        df = data_dict[ticker]
        # Filter to common date range
        df_filtered = df[(df.index >= common_start) & (df.index <= common_end)]
        
        if len(df_filtered) >= min_data_points:
            price_data[ticker] = df_filtered['Close']
            volume_data[ticker] = df_filtered['Volume']
    
    # Convert to DataFrames and align by index
    price_df = pd.DataFrame(price_data)
    volume_df = pd.DataFrame(volume_data)
    
    # Forward fill missing values (holidays, etc.)
    price_df = price_df.ffill()
    volume_df = volume_df.fillna(0)  # Zero volume on missing days
    
    # Drop any remaining rows with missing data
    price_df = price_df.dropna()
    volume_df = volume_df.loc[price_df.index]
    
    return price_df, volume_df


def download_universe(tickers: List[str], start: str, end: str,
                     min_data_points: int = 100, max_workers: int = 5,
                     use_cache: bool = True, cache_dir: str = "data/raw") -> PriceData:
    """
    Download and validate price data for multiple tickers.
    
    Args:
        tickers: List of ticker symbols
        start: Start date in 'YYYY-MM-DD' format  
        end: End date in 'YYYY-MM-DD' format
        min_data_points: Minimum number of data points required per ticker
        max_workers: Maximum number of parallel downloads
        use_cache: Whether to use cached data if available
        cache_dir: Directory to save/load cached data
        
    Returns:
        PriceData object with validated and aligned data
        
    Example:
        >>> tickers = ['AAPL', 'MSFT', 'GOOGL']
        >>> price_data = download_universe(tickers, '2020-01-01', '2020-12-31')
    """
    print(f"Downloading data for {len(tickers)} tickers from {start} to {end}")
    
    data_dict = {}
    tickers_to_download = []
    
    # Check cache first if enabled
    if use_cache:
        print(f"Checking cache in {cache_dir}...")
        for ticker in tickers:
            cached_data = load_ticker_data_from_pickle(ticker, cache_dir)
            if cached_data is not None:
                # Check if cached data covers the requested period (with tolerance for holidays)
                start_date = pd.to_datetime(start)
                end_date = pd.to_datetime(end)
                cache_start = cached_data.index[0]
                cache_end = cached_data.index[-1]
                
                # Allow up to 5 days difference at start (for holidays/weekends)
                start_ok = cache_start <= start_date + pd.Timedelta(days=5)
                end_ok = cache_end >= end_date - pd.Timedelta(days=5)
                
                if start_ok and end_ok:
                    # Filter to requested date range
                    mask = (cached_data.index >= start) & (cached_data.index <= end)
                    filtered_data = cached_data[mask]
                    if len(filtered_data) >= min_data_points:
                        data_dict[ticker] = filtered_data
                        print(f"✓ Loaded {ticker} from cache")
                        continue
                    
            tickers_to_download.append(ticker)
    else:
        tickers_to_download = tickers
    
    # Download missing tickers
    if tickers_to_download:
        print(f"Downloading {len(tickers_to_download)} tickers from Yahoo Finance...")
        new_data = download_multiple_tickers(tickers_to_download, start, end, max_workers)
        
        # Save to cache and add to results
        for ticker, data in new_data.items():
            if use_cache:
                save_ticker_data_to_pickle(ticker, data, cache_dir)
                print(f"✓ Saved {ticker} to cache")
            data_dict[ticker] = data
    
    if not data_dict:
        raise ValueError("No data was downloaded successfully")
    
    print(f"Successfully loaded data for {len(data_dict)} out of {len(tickers)} tickers")
    
    # Align data by common dates
    price_df, volume_df = align_data_by_dates(data_dict, min_data_points)
    
    if price_df.empty:
        raise ValueError("No tickers have sufficient aligned data")
    
    print(f"Final universe: {len(price_df.columns)} tickers, {len(price_df)} days")
    
    # Create PriceData object
    return PriceData(
        tickers=list(price_df.columns),
        dates=price_df.index,
        prices=price_df,
        volumes=volume_df
    )


def download_benchmark_data(benchmark_tickers: List[str], start: str, end: str) -> Dict[str, pd.Series]:
    """
    Download benchmark data for comparison.
    
    Args:
        benchmark_tickers: List of benchmark ticker symbols (e.g., ['SPY', 'IWV'])
        start: Start date in 'YYYY-MM-DD' format
        end: End date in 'YYYY-MM-DD' format
        
    Returns:
        Dictionary mapping benchmark name to price series
        
    Example:
        >>> benchmarks = download_benchmark_data(['SPY', 'IWV'], '2020-01-01', '2020-12-31')
    """
    print(f"Downloading benchmark data for {benchmark_tickers}")
    
    benchmark_data = {}
    
    for ticker in benchmark_tickers:
        data = download_single_ticker(ticker, start, end, use_cache=True, cache_dir="data/raw")
        if not data.empty:
            benchmark_data[ticker] = data['Close']
            print(f"Downloaded {len(data)} days for {ticker}")
        else:
            print(f"Failed to download data for benchmark {ticker}")
    
    return benchmark_data


def save_price_data(price_data: PriceData, filepath: str):
    """
    Save PriceData object to disk.
    
    Args:
        price_data: PriceData object to save
        filepath: Path to save the data (should end with .pkl)
        
    Example:
        >>> save_price_data(price_data, 'data/processed/price_data.pkl')
    """
    import pickle
    
    data_to_save = {
        'tickers': price_data.tickers,
        'dates': price_data.dates,
        'prices': price_data.prices,
        'volumes': price_data.volumes
    }
    
    with open(filepath, 'wb') as f:
        pickle.dump(data_to_save, f)
    
    print(f"Saved price data to {filepath}")


def load_price_data(filepath: str) -> PriceData:
    """
    Load PriceData object from disk.
    
    Args:
        filepath: Path to the saved data file
        
    Returns:
        PriceData object
        
    Example:
        >>> price_data = load_price_data('data/processed/price_data.pkl')
    """
    import pickle
    
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    return PriceData(
        tickers=data['tickers'],
        dates=data['dates'],
        prices=data['prices'],
        volumes=data['volumes']
    )


def get_sp500_tickers() -> List[str]:
    """
    Get current S&P 500 ticker list from Wikipedia.
    
    Returns:
        List of S&P 500 ticker symbols
        
    Example:
        >>> sp500_tickers = get_sp500_tickers()
        >>> len(sp500_tickers)  # Should be around 500
    """
    try:
        # Read S&P 500 list from Wikipedia
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        tables = pd.read_html(url)
        sp500_table = tables[0]
        
        # Extract ticker symbols and clean them
        tickers = sp500_table['Symbol'].tolist()
        
        # Replace any special characters that might cause issues
        tickers = [ticker.replace('.', '-') for ticker in tickers]
        
        print(f"Retrieved {len(tickers)} S&P 500 tickers")
        return tickers
        
    except Exception as e:
        print(f"Error retrieving S&P 500 tickers: {e}")
        # Fallback to a static list of major stocks
        return [
            'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'TSLA', 'META', 'NVDA', 'BRK-B',
            'UNH', 'JNJ', 'JPM', 'V', 'PG', 'XOM', 'HD', 'CVX', 'MA', 'PFE',
            'BAC', 'ABBV', 'KO', 'AVGO', 'LLY', 'PEP', 'TMO', 'COST', 'WMT',
            'DIS', 'ABT', 'DHR', 'VZ', 'ACN', 'CMCSA', 'NKE', 'TXN', 'LIN',
            'NEE', 'ORCL', 'ADBE', 'CRM', 'MDT', 'PM', 'BMY', 'T', 'HON',
            'QCOM', 'LOW', 'UPS', 'AMD', 'C', 'SPGI', 'RTX', 'INTU', 'CAT'
        ]


def create_sp100_list() -> List[str]:
    """
    Create a representative S&P 100 list since it's not directly available.
    
    Returns:
        List of approximately 100 large-cap ticker symbols
        
    Example:
        >>> sp100_tickers = create_sp100_list()
        >>> len(sp100_tickers)  # Should be around 100
    """
    # Top 100 stocks by market cap (approximate S&P 100)
    sp100_tickers = [
        'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'TSLA', 'META', 'NVDA', 'BRK-B',
        'UNH', 'JNJ', 'JPM', 'V', 'PG', 'XOM', 'HD', 'CVX', 'MA', 'PFE',
        'BAC', 'ABBV', 'KO', 'AVGO', 'LLY', 'PEP', 'TMO', 'COST', 'WMT',
        'DIS', 'ABT', 'DHR', 'VZ', 'ACN', 'CMCSA', 'NKE', 'TXN', 'LIN',
        'NEE', 'ORCL', 'ADBE', 'CRM', 'MDT', 'PM', 'BMY', 'T', 'HON',
        'QCOM', 'LOW', 'UPS', 'AMD', 'C', 'SPGI', 'RTX', 'INTU', 'CAT',
        'AMGN', 'DE', 'ISRG', 'BKNG', 'GS', 'TGT', 'MO', 'AXP', 'SYK',
        'LRCX', 'BLK', 'GILD', 'MDLZ', 'ADI', 'ADP', 'SBUX', 'MMM', 'TJX',
        'VRTX', 'CVS', 'ZTS', 'CHTR', 'MU', 'FIS', 'PYPL', 'SO', 'CI',
        'NOW', 'REGN', 'SHW', 'DUK', 'BSX', 'CB', 'EQIX', 'ICE',
        'CL', 'CSX', 'WM', 'MMC', 'EL', 'GD', 'KLAC', 'APH', 'USB', 'PGR',
        'AON', 'CME', 'MCO', 'FDX', 'NSC', 'ITW', 'SLB', 'HUM', 'GE', 'EMR'
    ]
    
    print(f"Created S&P 100 list with {len(sp100_tickers)} tickers")
    return sp100_tickers


def create_sp100_since_2010() -> List[str]:
    """
    Create a list of the 60 largest stocks that have been in the S&P 100 since 2010-01-01, plus Tesla.
    
    This is a curated list based on historical S&P 100 membership and current market capitalization.
    
    Returns:
        List of 61 ticker symbols (60 largest S&P 100 stocks since 2010 + TSLA)
    """
    # These are stocks that have been consistently in the S&P 100 since 2010
    # Ordered approximately by current market cap
    sp100_since_2010 = [
        'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'NVDA', 'BRK-B',  # Mega caps
        'UNH', 'JNJ', 'JPM', 'V', 'PG', 'XOM', 'HD', 'CVX', 'MA', 'PFE',  # Large caps
        'BAC', 'ABBV', 'KO', 'LLY', 'PEP', 'TMO', 'COST', 'WMT',
        'DIS', 'ABT', 'VZ', 'ACN', 'CMCSA', 'NKE', 'TXN', 'LIN',
        'ORCL', 'ADBE', 'CRM', 'MDT', 'PM', 'BMY', 'T', 'HON',
        'QCOM', 'LOW', 'UPS', 'AMD', 'C', 'RTX', 'INTU', 'CAT',
        'AMGN', 'DE', 'GS', 'MO', 'AXP', 'BLK', 'GILD', 'MDLZ',
        'MMM', 'CVS', 'SO', 'DUK'  # 60 stocks total
    ]
    
    # Add Tesla (not in S&P 100 until later but requested to be included)
    if 'TSLA' not in sp100_since_2010:
        sp100_since_2010.append('TSLA')
    
    print(f"Created custom S&P 100 universe with {len(sp100_since_2010)} tickers (60 largest since 2010 + TSLA)")
    return sp100_since_2010


def save_ticker_data_to_pickle(ticker: str, data: pd.DataFrame, output_dir: str = "data/raw"):
    """Save individual ticker data to a pickle file."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create filename with ticker and a simple .pkl extension
    # We don't need date ranges in the name anymore, we'll find the right file by ticker
    filename = f"{ticker}.pkl"
    filepath = os.path.join(output_dir, filename)
    
    # Save to pickle
    data.to_pickle(filepath)
    return filepath


def load_ticker_data_from_pickle(ticker: str, output_dir: str = "data/raw") -> Optional[pd.DataFrame]:
    """Load ticker data from a pickle file if it exists."""
    filename = f"{ticker}.pkl"
    filepath = os.path.join(output_dir, filename)
    
    if os.path.exists(filepath):
        try:
            df = pd.read_pickle(filepath)
            return df
        except Exception as e:
            print(f"Error loading pickle cache file {filepath}: {e}")
            # Attempt to delete corrupted file so it can be re-downloaded
            try:
                os.remove(filepath)
                print(f"Removed corrupted cache file: {filepath}")
            except OSError as del_e:
                print(f"Error removing corrupted cache file: {del_e}")
    return None 