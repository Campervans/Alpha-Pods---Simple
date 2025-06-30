# market data downloader using yfinance

import yfinance as yf
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..utils.schemas import PriceData


def download_single_ticker(ticker: str, start: str, end: str, 
                          retry_count: int = 3, delay: float = 1.0) -> pd.DataFrame:
    # download one ticker with retries
    for attempt in range(retry_count):
        try:
            # quiet down yfinance
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                # download w/ auto-adjust
                df = yf.download(
                    ticker, 
                    start=start, 
                    end=end, 
                    auto_adjust=True,
                    progress=False,
                    show_errors=False
                )
                
                if df.empty:
                    if attempt < retry_count - 1:
                        time.sleep(delay)
                        continue
                    return pd.DataFrame()
                
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
                    
                    return result
                else:
                    if attempt < retry_count - 1:
                        time.sleep(delay)
                        continue
                    return pd.DataFrame()
                    
        except Exception as e:
            if attempt < retry_count - 1:
                # print(f"Try {attempt + 1} failed for {ticker}")
                time.sleep(delay)
                continue
            else:
                print(f"Failed {ticker}: {e}")
                return pd.DataFrame()
    
    return pd.DataFrame()


def download_multiple_tickers(tickers: List[str], start: str, end: str, 
                            max_workers: int = 10, progress_bar: bool = True) -> Dict[str, pd.DataFrame]:
    # download multiple tickers in parallel
    results = {}
    
    # parallel downloads ftw
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # submit all
        future_to_ticker = {
            executor.submit(download_single_ticker, ticker, start, end): ticker
            for ticker in tickers
        }
        
        # process results
        if progress_bar:
            iterator = tqdm(as_completed(future_to_ticker), 
                          total=len(tickers), 
                          desc="Downloading")
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
    price_df = price_df.fillna(method='ffill')
    volume_df = volume_df.fillna(0)  # Zero volume on missing days
    
    # Drop any remaining rows with missing data
    price_df = price_df.dropna()
    volume_df = volume_df.loc[price_df.index]
    
    return price_df, volume_df


def download_universe(tickers: List[str], start: str, end: str,
                     min_data_points: int = 100, max_workers: int = 10) -> PriceData:
    """
    Download and validate price data for multiple tickers.
    
    Args:
        tickers: List of ticker symbols
        start: Start date in 'YYYY-MM-DD' format  
        end: End date in 'YYYY-MM-DD' format
        min_data_points: Minimum number of data points required per ticker
        max_workers: Maximum number of parallel downloads
        
    Returns:
        PriceData object with validated and aligned data
        
    Example:
        >>> tickers = ['AAPL', 'MSFT', 'GOOGL']
        >>> price_data = download_universe(tickers, '2020-01-01', '2020-12-31')
    """
    print(f"Downloading data for {len(tickers)} tickers from {start} to {end}")
    
    # Download all ticker data
    data_dict = download_multiple_tickers(tickers, start, end, max_workers)
    
    if not data_dict:
        raise ValueError("No data was downloaded successfully")
    
    print(f"Successfully downloaded data for {len(data_dict)} out of {len(tickers)} tickers")
    
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