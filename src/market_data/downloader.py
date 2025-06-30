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
PROXY_PORTS = [
    10001, 10002, 10003, 10004, 10005, 10006, 10007, 10008, 10009, 10010,
    10011, 10012, 10013, 10014, 10015, 10016, 10017, 10018, 10019, 10020,
    10021, 10022, 10023, 10024, 10025, 10026, 10027, 10028, 10029, 10030
]


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
                          retry_count: int = 3, delay: float = 1.0) -> pd.DataFrame:
    # download one ticker with retries and rate limit handling
    
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
                df = yf.download(
                    ticker, 
                    start=start, 
                    end=end, 
                    auto_adjust=True,
                    progress=False
                )
                
                if df.empty:
                    if attempt < retry_count - 1:
                        time.sleep(delay)
                        continue
                    # Try next proxy
                    break
                
                # Handle multi-level columns (when ticker is in column names)
                if isinstance(df.columns, pd.MultiIndex):
                    # Extract just the price/volume data, dropping the ticker level
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
    # download multiple tickers with rate limit protection
    # Reduced max_workers from 10 to 5 to avoid rate limits
    results = {}
    
    # Process in smaller batches to avoid rate limiting
    batch_size = 10
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        
        # parallel downloads within batch
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # submit all
            future_to_ticker = {
                executor.submit(download_single_ticker, ticker, start, end): ticker
                for ticker in batch
            }
            
            # process results
            if progress_bar:
                iterator = tqdm(as_completed(future_to_ticker), 
                              total=len(batch), 
                              desc=f"Batch {i//batch_size + 1}/{(len(tickers) + batch_size - 1)//batch_size}")
            else:
                iterator = as_completed(future_to_ticker)
            
            for future in iterator:
                ticker = future_to_ticker[future]
                try:
                    data = future.result()
                    if not data.empty:
                        results[ticker] = data
                    else:
                        print(f"No data for {ticker}")
                except Exception as e:
                    print(f"Error: {ticker}: {e}")
        
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
            cached_data = load_ticker_data_from_csv(ticker, cache_dir)
            if cached_data is not None:
                # Check if cached data covers the requested period
                if (cached_data.index[0] <= pd.to_datetime(start) and 
                    cached_data.index[-1] >= pd.to_datetime(end)):
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
                save_ticker_data_to_csv(ticker, data, cache_dir)
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
        data = download_single_ticker(ticker, start, end)
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
        'NOW', 'REGN', 'SHW', 'DUK', 'BSX', 'CB', 'ATVI', 'EQIX', 'ICE',
        'CL', 'CSX', 'WM', 'MMC', 'EL', 'GD', 'KLAC', 'APH', 'USB', 'PGR',
        'AON', 'CME', 'MCO', 'FDX', 'NSC', 'ITW', 'SLB', 'HUM', 'GE', 'EMR'
    ]
    
    print(f"Created S&P 100 list with {len(sp100_tickers)} tickers")
    return sp100_tickers


def save_ticker_data_to_csv(ticker: str, data: pd.DataFrame, output_dir: str = "data/raw"):
    """Save individual ticker data to CSV file."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create filename with ticker and date range
    start_date = data.index[0].strftime('%Y%m%d')
    end_date = data.index[-1].strftime('%Y%m%d')
    filename = f"{ticker}_{start_date}_{end_date}.csv"
    filepath = os.path.join(output_dir, filename)
    
    # Save to CSV
    data.to_csv(filepath)
    return filepath


def load_ticker_data_from_csv(ticker: str, output_dir: str = "data/raw") -> Optional[pd.DataFrame]:
    """Load ticker data from CSV if it exists."""
    # Look for files matching the ticker pattern
    if os.path.exists(output_dir):
        for filename in os.listdir(output_dir):
            if filename.startswith(f"{ticker}_") and filename.endswith(".csv"):
                filepath = os.path.join(output_dir, filename)
                try:
                    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
                    return df
                except Exception as e:
                    print(f"Error loading {filepath}: {e}")
    return None 