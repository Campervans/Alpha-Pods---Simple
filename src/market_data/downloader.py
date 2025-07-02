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

# PROXY CONFIG - creds are hardcoded, don't change
PROXY_USERNAME = 'sp7lr99xhd'
PROXY_PASSWORD = '7Xtywa2k3o0oxoViLX'

# load proxy ports from csv
from ..utils.proxy_utils import load_proxies_from_csv
PROXY_PORTS = load_proxies_from_csv()


def get_random_proxy():
    """get a random proxy."""
    port = random.choice(PROXY_PORTS)
    proxy = f"http://{PROXY_USERNAME}:{PROXY_PASSWORD}@dc.decodo.com:{port}"
    return {
        'http': proxy,
        'https': proxy
    }


def test_proxy(proxy_dict):
    """test if a proxy works."""
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
    # download one ticker with retries & rate limit handling
    
    # smart cache check
    if use_cache:
        # pickle format preserves data types, no complex validation needed
        cached_df = load_ticker_data_from_pickle(ticker, cache_dir)
        
        if cached_df is not None:
            # check if cached data covers requested date range
            start_date = pd.to_datetime(start)
            end_date = pd.to_datetime(end)
            
            # make sure index is a DatetimeIndex
            if isinstance(cached_df.index, pd.DatetimeIndex):
                # allow for holidays at start
                cache_start = cached_df.index[0]
                cache_end = cached_df.index[-1]
                
                # allow up to 5 days diff at start
                start_ok = cache_start <= start_date + pd.Timedelta(days=5)
                end_ok = cache_end >= end_date - pd.Timedelta(days=5)
                
                if not cached_df.empty and start_ok and end_ok:
                    # filter to exact date range
                    mask = (cached_df.index >= start_date) & (cached_df.index <= end_date)
                    filtered_data = cached_df.loc[mask]
                    
                    if not filtered_data.empty:
                        print(f"✓ {ticker} loaded from pickle cache ({len(filtered_data)} days)")
                        return filtered_data
            else:
                print(f"⚠️  {ticker}: cached data has invalid index type, re-downloading.")
    
    # if cache miss, proceed with download
    # save original proxy env vars
    original_http_proxy = os.environ.get('HTTP_PROXY', '')
    original_https_proxy = os.environ.get('HTTPS_PROXY', '')
    
    # try with different proxies, then fallback to no proxy
    proxy_attempts = min(5, len(PROXY_PORTS))
    
    for proxy_attempt in range(proxy_attempts + 1):  # +1 for no-proxy attempt
        if proxy_attempt < proxy_attempts:
            # try with a proxy
            proxy_dict = get_random_proxy()
            proxy_url = proxy_dict['http']
            proxy_info = f"proxy port {proxy_url.split(':')[-1]}"
            
            # set env vars for proxy
            os.environ['HTTP_PROXY'] = proxy_url
            os.environ['HTTPS_PROXY'] = proxy_url
            os.environ['http_proxy'] = proxy_url
            os.environ['https_proxy'] = proxy_url
        else:
            # last attempt without proxy
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
                    
                # delay between requests to avoid rate limiting
                if attempt > 0 or proxy_attempt > 0:
                    time.sleep(delay * (attempt + 1))
                    
                # let yfinance handle session
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
                
                if df.empty:
                    if attempt < retry_count - 1:
                        time.sleep(delay)
                        continue
                    # try next proxy
                    break
                
                # handle multi-level columns
                if isinstance(df.columns, pd.MultiIndex):
                    if len(df.columns.levels) > 1 and ticker in df.columns.get_level_values(1):
                        # if ticker is in second level, select it
                        df = df.xs(ticker, level=1, axis=1)
                    elif len(set(df.columns.get_level_values(1))) == 1:
                        # if only one ticker in second level, drop it
                        df.columns = df.columns.droplevel(1)
                    else:
                        # try to extract just price/volume data
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
                        print(f"X {ticker} has weird prices")
                        result = result[result['Close'] > 0]
                    
                    # success!
                    if proxy_attempt < proxy_attempts:
                        print(f"✓ {ticker} downloaded successfully using {proxy_info}")
                    
                    # restore original proxy settings
                    os.environ['HTTP_PROXY'] = original_http_proxy
                    os.environ['HTTPS_PROXY'] = original_https_proxy
                    os.environ['http_proxy'] = original_http_proxy
                    os.environ['https_proxy'] = original_https_proxy
                    
                    return result
                else:
                    if attempt < retry_count - 1:
                        time.sleep(delay)
                        continue
                    # try next proxy
                    break
                    
            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg or "Too Many Requests" in error_msg:
                    # rate limit error - wait longer
                    wait_time = delay * (2 ** (attempt + 1))
                    print(f"Rate limit hit for {ticker} using {proxy_info}, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                elif attempt < retry_count - 1:
                    # other error - normal retry
                    time.sleep(delay)
                    continue
                else:
                    # last attempt for this proxy failed
                    if proxy_attempt < proxy_attempts:
                        print(f"Failed {ticker} with {proxy_info}: {e}, trying different proxy...")
                    else:
                        print(f"Failed {ticker} with all methods: {e}")
                    break
    
    # restore original proxy settings
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
    avoids race condition issues with yfinance's shared global dict.
    """
    results = {}
    
    # process in smaller batches to avoid rate limiting
    batch_size = 20
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        
        if progress_bar:
            print(f"Processing batch {i//batch_size + 1}/{(len(tickers) + batch_size - 1)//batch_size}: {len(batch)} tickers")
        
        try:
            # use yfinance's built-in batch download
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                # download all tickers in batch
                batch_data = yf.download(
                    batch,
                    start=start,
                    end=end,
                    auto_adjust=True,
                    progress=progress_bar,
                    group_by='ticker'
                )
            
            # process batch results
            if not batch_data.empty:
                # handle single ticker case
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
                
                # handle multiple tickers case
                elif isinstance(batch_data.columns, pd.MultiIndex):
                    for ticker in batch:
                        try:
                            # extract data for this ticker
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
            # fallback to individual downloads for this batch
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
        
        # delay between batches to avoid rate limiting
        if i + batch_size < len(tickers):
            time.sleep(batch_delay)
    
    return results


def align_data_by_dates(data_dict: Dict[str, pd.DataFrame], 
                       min_data_points: int = 100) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Align price and volume data across tickers by common dates.
    
    Args:
        data_dict: dict mapping ticker to DataFrame
        min_data_points: min data points required per ticker
        
    Returns:
        (price_df, volume_df) with aligned dates
        
    Example:
        >>> data_dict = {'AAPL': df1, 'MSFT': df2}
        >>> prices, volumes = align_data_by_dates(data_dict, min_data_points=100)
    """
    if not data_dict:
        return pd.DataFrame(), pd.DataFrame()
    
    # filter tickers with enough data
    valid_tickers = [
        ticker for ticker, df in data_dict.items()
        if len(df) >= min_data_points
    ]
    
    if not valid_tickers:
        print(f"No tickers have at least {min_data_points} data points")
        return pd.DataFrame(), pd.DataFrame()
    
    print(f"Using {len(valid_tickers)} tickers with sufficient data")
    
    # find common date range
    start_dates = [data_dict[ticker].index.min() for ticker in valid_tickers]
    end_dates = [data_dict[ticker].index.max() for ticker in valid_tickers]
    
    common_start = max(start_dates)
    common_end = min(end_dates)
    
    print(f"Common date range: {common_start.date()} to {common_end.date()}")
    
    # create aligned DataFrames
    price_data = {}
    volume_data = {}
    
    for ticker in valid_tickers:
        df = data_dict[ticker]
        # filter to common date range
        df_filtered = df[(df.index >= common_start) & (df.index <= common_end)]
        
        if len(df_filtered) >= min_data_points:
            price_data[ticker] = df_filtered['Close']
            volume_data[ticker] = df_filtered['Volume']
    
    # convert to DataFrames and align by index
    price_df = pd.DataFrame(price_data)
    volume_df = pd.DataFrame(volume_data)
    
    # ffill missing values (holidays, etc.)
    price_df = price_df.ffill()
    volume_df = volume_df.fillna(0)
    
    # drop any remaining rows with missing data
    price_df = price_df.dropna()
    volume_df = volume_df.loc[price_df.index]
    
    return price_df, volume_df


def download_universe(tickers: List[str], start: str, end: str,
                     min_data_points: int = 100, max_workers: int = 5,
                     use_cache: bool = True, cache_dir: str = "data/raw") -> PriceData:
    """
    Download and validate price data for multiple tickers.
    
    Args:
        tickers: list of ticker symbols
        start: start date in 'YYYY-MM-DD'
        end: end date in 'YYYY-MM-DD'
        min_data_points: min data points required per ticker
        max_workers: max number of parallel downloads
        use_cache: whether to use cached data
        cache_dir: directory to save/load cached data
        
    Returns:
        PriceData object with validated and aligned data
        
    Example:
        >>> tickers = ['AAPL', 'MSFT', 'GOOGL']
        >>> price_data = download_universe(tickers, '2020-01-01', '2020-12-31')
    """
    print(f"Downloading data for {len(tickers)} tickers from {start} to {end}")
    
    data_dict = {}
    tickers_to_download = []
    
    # check cache first if enabled
    if use_cache:
        print(f"Checking cache in {cache_dir}...")
        for ticker in tickers:
            cached_data = load_ticker_data_from_pickle(ticker, cache_dir)
            if cached_data is not None:
                # check if cached data covers the requested period
                start_date = pd.to_datetime(start)
                end_date = pd.to_datetime(end)
                cache_start = cached_data.index[0]
                cache_end = cached_data.index[-1]
                
                # allow up to 5 days diff at start
                start_ok = cache_start <= start_date + pd.Timedelta(days=5)
                end_ok = cache_end >= end_date - pd.Timedelta(days=5)
                
                if start_ok and end_ok:
                    # filter to requested date range
                    mask = (cached_data.index >= start) & (cached_data.index <= end)
                    filtered_data = cached_data[mask]
                    if len(filtered_data) >= min_data_points:
                        data_dict[ticker] = filtered_data
                        print(f"✓ Loaded {ticker} from cache")
                        continue
                    
            tickers_to_download.append(ticker)
    else:
        tickers_to_download = tickers
    
    # download missing tickers
    if tickers_to_download:
        print(f"Downloading {len(tickers_to_download)} tickers from Yahoo Finance...")
        new_data = download_multiple_tickers(tickers_to_download, start, end, max_workers)
        
        # save to cache and add to results
        for ticker, data in new_data.items():
            if use_cache:
                save_ticker_data_to_pickle(ticker, data, cache_dir)
                print(f"✓ Saved {ticker} to cache")
            data_dict[ticker] = data
    
    if not data_dict:
        raise ValueError("No data was downloaded successfully")
    
    print(f"Successfully loaded data for {len(data_dict)} out of {len(tickers)} tickers")
    
    # align data by common dates
    price_df, volume_df = align_data_by_dates(data_dict, min_data_points)
    
    if price_df.empty:
        raise ValueError("No tickers have sufficient aligned data")
    
    print(f"Final universe: {len(price_df.columns)} tickers, {len(price_df)} days")
    
    # create PriceData object
    return PriceData(
        tickers=list(price_df.columns),
        dates=price_df.index,
        prices=price_df,
        volumes=volume_df
    )


def download_benchmark_data(benchmark_tickers: List[str], start: str, end: str) -> Dict[str, pd.Series]:
    """
    Download benchmark data.
    
    Args:
        benchmark_tickers: list of benchmark symbols (e.g., ['SPY'])
        start: start date in 'YYYY-MM-DD'
        end: end date in 'YYYY-MM-DD'
        
    Returns:
        dict mapping benchmark name to price series
        
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
        filepath: path to save data (should end with .pkl)
        
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
        filepath: path to saved data file
        
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
    Get current S&P 500 tickers from Wikipedia.
    
    Returns:
        list of S&P 500 ticker symbols
        
    Example:
        >>> sp500_tickers = get_sp500_tickers()
    """
    try:
        # read S&P 500 list from Wikipedia
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        tables = pd.read_html(url)
        sp500_table = tables[0]
        
        # extract and clean tickers
        tickers = sp500_table['Symbol'].tolist()
        
        # replace special characters
        tickers = [ticker.replace('.', '-') for ticker in tickers]
        
        print(f"Retrieved {len(tickers)} S&P 500 tickers")
        return tickers
        
    except Exception as e:
        print(f"Error retrieving S&P 500 tickers: {e}")
        # fallback to a static list
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
    Create a representative S&P 100 list.
    
    Returns:
        list of ~100 large-cap ticker symbols
        
    Example:
        >>> sp100_tickers = create_sp100_list()
    """
    # top 100 stocks by market cap (approx S&P 100)
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
    Create list of 60 largest stocks in S&P 100 since 2010, plus Tesla.
    
    This is a curated list.
    
    Returns:
        list of 61 ticker symbols
    """
    # stocks consistently in S&P 100 since 2010
    sp100_since_2010 = [
        'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'NVDA', 'BRK-B',  # mega caps
        'UNH', 'JNJ', 'JPM', 'V', 'PG', 'XOM', 'HD', 'CVX', 'MA', 'PFE',  # large caps
        'BAC', 'ABBV', 'KO', 'LLY', 'PEP', 'TMO', 'COST', 'WMT',
        'DIS', 'ABT', 'VZ', 'ACN', 'CMCSA', 'NKE', 'TXN', 'LIN',
        'ORCL', 'ADBE', 'CRM', 'MDT', 'PM', 'BMY', 'T', 'HON',
        'QCOM', 'LOW', 'UPS', 'AMD', 'C', 'RTX', 'INTU', 'CAT',
        'AMGN', 'DE', 'GS', 'MO', 'AXP', 'BLK', 'GILD', 'MDLZ',
        'MMM', 'CVS', 'SO'
    ]
    
    # add Tesla
    if 'TSLA' not in sp100_since_2010:
        sp100_since_2010.append('TSLA')
    
    print(f"Created custom S&P 100 universe with {len(sp100_since_2010)} tickers (60 largest since 2010 + TSLA)")
    return sp100_since_2010


def save_ticker_data_to_pickle(ticker: str, data: pd.DataFrame, output_dir: str = "data/raw"):
    """save individual ticker data to a pickle file."""
    os.makedirs(output_dir, exist_ok=True)
    
    # create filename with ticker and .pkl extension
    filename = f"{ticker}.pkl"
    filepath = os.path.join(output_dir, filename)
    
    # save to pickle
    data.to_pickle(filepath)
    return filepath


def load_ticker_data_from_pickle(ticker: str, output_dir: str = "data/raw") -> Optional[pd.DataFrame]:
    """load ticker data from a pickle file if it exists."""
    filename = f"{ticker}.pkl"
    filepath = os.path.join(output_dir, filename)
    
    if os.path.exists(filepath):
        try:
            df = pd.read_pickle(filepath)
            return df
        except Exception as e:
            print(f"Error loading pickle cache file {filepath}: {e}")
            # try to delete corrupted file so it can be re-downloaded
            try:
                os.remove(filepath)
                print(f"Removed corrupted cache file: {filepath}")
            except OSError as del_e:
                print(f"Error removing corrupted cache file: {del_e}")
    return None 