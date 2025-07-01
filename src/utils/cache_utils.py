"""
Cache utilities for stock data management.

This module provides functions to parse, validate, and manage cached stock data files.
"""

import os
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

def parse_cache_filename(filename: str) -> Optional[Dict]:
    """
    Parse cache filename to extract ticker and date range.
    
    Expected format: TICKER_YYYYMMDD_YYYYMMDD.csv
    Example: AAPL_20090701_20241230.csv
    
    Args:
        filename: Cache filename to parse
        
    Returns:
        dict with keys: ticker, start_date, end_date, filename, start_raw, end_raw
        or None if parsing fails
        
    Example:
        >>> info = parse_cache_filename("AAPL_20090701_20241230.csv")
        >>> info['ticker']
        'AAPL'
        >>> info['start_date']
        '2009-07-01'
    """
    try:
        if not filename.endswith('.csv'):
            return None
            
        # Remove .csv extension
        name_part = filename[:-4]
        parts = name_part.split('_')
        
        if len(parts) < 3:
            return None
            
        ticker = parts[0]
        start_date = parts[1]
        end_date = parts[2]
        
        # Validate date format (YYYYMMDD)
        if len(start_date) != 8 or len(end_date) != 8:
            return None
            
        # Try parsing dates to ensure they're valid
        try:
            start_dt = datetime.strptime(start_date, '%Y%m%d')
            end_dt = datetime.strptime(end_date, '%Y%m%d')
        except ValueError:
            return None
            
        # Convert to readable format
        start_formatted = start_dt.strftime('%Y-%m-%d')
        end_formatted = end_dt.strftime('%Y-%m-%d')
        
        return {
            'ticker': ticker,
            'start_date': start_formatted,
            'end_date': end_formatted,
            'filename': filename,
            'start_raw': start_date,
            'end_raw': end_date,
            'start_dt': start_dt,
            'end_dt': end_dt
        }
        
    except Exception as e:
        print(f"Warning: Could not parse filename '{filename}': {e}")
        return None

def find_cache_files(ticker: str, cache_dir: str) -> List[Dict]:
    """
    Find all cache files for a specific ticker.
    
    Args:
        ticker: Ticker symbol to search for
        cache_dir: Directory to search in
        
    Returns:
        List of file info dictionaries, sorted by start date
        
    Example:
        >>> files = find_cache_files('AAPL', 'data/raw')
        >>> len(files)
        9
    """
    if not os.path.exists(cache_dir):
        return []
    
    cache_files = []
    
    for filename in os.listdir(cache_dir):
        if filename.startswith(f"{ticker}_") and filename.endswith(".csv"):
            parsed = parse_cache_filename(filename)
            if parsed:
                # Add full path
                parsed['filepath'] = os.path.join(cache_dir, filename)
                cache_files.append(parsed)
    
    # Sort by start date
    cache_files.sort(key=lambda x: x['start_dt'])
    
    return cache_files

def get_best_cache_file(ticker: str, start: str, end: str, cache_dir: str) -> Optional[Dict]:
    """
    Find the best cache file that covers the requested date range.
    
    Args:
        ticker: Ticker symbol
        start: Start date in 'YYYY-MM-DD' format
        end: End date in 'YYYY-MM-DD' format
        cache_dir: Directory to search in
        
    Returns:
        File info dict for best matching cache file, or None if no suitable file
        
    Example:
        >>> best = get_best_cache_file('AAPL', '2024-01-01', '2024-12-31', 'data/raw')
        >>> best['filename']
        'AAPL_20240101_20241230.csv'
    """
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)
    
    cache_files = find_cache_files(ticker, cache_dir)
    
    if not cache_files:
        return None
    
    # Strategy 1: Find exact or covering file
    for file_info in cache_files:
        if (file_info['start_dt'] <= start_dt and file_info['end_dt'] >= end_dt):
            return file_info
    
    # Strategy 2: Find file with maximum overlap
    best_file = None
    best_overlap = 0
    
    for file_info in cache_files:
        # Calculate overlap
        overlap_start = max(file_info['start_dt'], start_dt)
        overlap_end = min(file_info['end_dt'], end_dt)
        
        if overlap_start <= overlap_end:
            overlap_days = (overlap_end - overlap_start).days
            if overlap_days > best_overlap:
                best_overlap = overlap_days
                best_file = file_info
    
    return best_file

def validate_date_coverage(df: pd.DataFrame, start: str, end: str, 
                          min_coverage: float = 0.8) -> bool:
    """
    Check if DataFrame covers requested dates adequately.
    
    Args:
        df: DataFrame with DatetimeIndex
        start: Start date in 'YYYY-MM-DD' format
        end: End date in 'YYYY-MM-DD' format
        min_coverage: Minimum fraction of days that must be covered (0.8 = 80%)
        
    Returns:
        True if coverage is sufficient
        
    Example:
        >>> df = pd.DataFrame(index=pd.date_range('2024-01-01', '2024-12-31'))
        >>> validate_date_coverage(df, '2024-01-01', '2024-12-31')
        True
    """
    if df.empty:
        return False
    
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)
    
    # Filter to requested range
    mask = (df.index >= start_dt) & (df.index <= end_dt)
    filtered_df = df[mask]
    
    if len(filtered_df) == 0:
        return False
    
    # Calculate expected trading days (rough estimate)
    total_days = (end_dt - start_dt).days + 1
    expected_trading_days = total_days * 0.7  # Approximately 70% are trading days
    
    # Check if we have sufficient coverage
    actual_days = len(filtered_df)
    coverage = actual_days / expected_trading_days
    
    return coverage >= min_coverage

def get_missing_date_ranges(df: pd.DataFrame, start: str, end: str) -> List[Tuple[str, str]]:
    """
    Identify gaps in data coverage.
    
    Args:
        df: DataFrame with DatetimeIndex
        start: Start date in 'YYYY-MM-DD' format
        end: End date in 'YYYY-MM-DD' format
        
    Returns:
        List of (start_date, end_date) tuples for missing ranges
        
    Example:
        >>> gaps = get_missing_date_ranges(df, '2024-01-01', '2024-12-31')
        >>> len(gaps)
        0  # No gaps
    """
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)
    
    if df.empty:
        return [(start, end)]
    
    # Get data in requested range
    mask = (df.index >= start_dt) & (df.index <= end_dt)
    filtered_df = df[mask]
    
    if len(filtered_df) == 0:
        return [(start, end)]
    
    # Find the first and last dates we have
    actual_start = filtered_df.index.min()
    actual_end = filtered_df.index.max()
    
    gaps = []
    
    # Gap before our data
    if actual_start > start_dt:
        gap_end = (actual_start - timedelta(days=1)).strftime('%Y-%m-%d')
        gaps.append((start, gap_end))
    
    # Gap after our data
    if actual_end < end_dt:
        gap_start = (actual_end + timedelta(days=1)).strftime('%Y-%m-%d')
        gaps.append((gap_start, end))
    
    return gaps

def get_cache_stats(cache_dir: str = "data/raw") -> Dict:
    """
    Get comprehensive statistics about the cache directory.
    
    Args:
        cache_dir: Directory to analyze
        
    Returns:
        Dictionary with cache statistics
        
    Example:
        >>> stats = get_cache_stats()
        >>> stats['total_files']
        386
    """
    if not os.path.exists(cache_dir):
        return {
            'total_files': 0,
            'unique_tickers': 0,
            'total_size_mb': 0,
            'oldest_date': None,
            'newest_date': None,
            'tickers_with_multiple_files': 0
        }
    
    all_files = []
    ticker_counts = {}
    total_size = 0
    
    for filename in os.listdir(cache_dir):
        if filename.endswith('.csv'):
            parsed = parse_cache_filename(filename)
            if parsed:
                all_files.append(parsed)
                
                # Count files per ticker
                ticker = parsed['ticker']
                ticker_counts[ticker] = ticker_counts.get(ticker, 0) + 1
                
                # Calculate file size
                filepath = os.path.join(cache_dir, filename)
                try:
                    size = os.path.getsize(filepath)
                    total_size += size
                except:
                    pass
    
    if not all_files:
        return {
            'total_files': 0,
            'unique_tickers': 0,
            'total_size_mb': 0,
            'oldest_date': None,
            'newest_date': None,
            'tickers_with_multiple_files': 0
        }
    
    # Find date range
    oldest_date = min(f['start_dt'] for f in all_files)
    newest_date = max(f['end_dt'] for f in all_files)
    
    # Count tickers with multiple files
    multiple_files = sum(1 for count in ticker_counts.values() if count > 1)
    
    return {
        'total_files': len(all_files),
        'unique_tickers': len(ticker_counts),
        'total_size_mb': total_size / (1024 * 1024),
        'oldest_date': oldest_date.strftime('%Y-%m-%d'),
        'newest_date': newest_date.strftime('%Y-%m-%d'),
        'tickers_with_multiple_files': multiple_files,
        'ticker_file_counts': ticker_counts
    }

def print_cache_summary(stats: Dict):
    """
    Print a nicely formatted summary of cache statistics.
    
    Args:
        stats: Dictionary from get_cache_stats()
        
    Example:
        >>> stats = get_cache_stats()
        >>> print_cache_summary(stats)
    """
    print("📁 Cache Directory Summary")
    print("=" * 40)
    print(f"Total files: {stats['total_files']:,}")
    print(f"Unique tickers: {stats['unique_tickers']:,}")
    print(f"Total size: {stats['total_size_mb']:.1f} MB")
    print(f"Date range: {stats['oldest_date']} to {stats['newest_date']}")
    print(f"Tickers with multiple files: {stats['tickers_with_multiple_files']:,}")
    
    if stats['tickers_with_multiple_files'] > 0:
        print(f"\n⚠️ {stats['tickers_with_multiple_files']} tickers have multiple cache files")
        print("This may cause inefficient cache usage. Consider cache cleanup.")
    else:
        print("\n✅ All tickers have single cache files")

def is_cache_stale(filepath: str, max_age_days: int = 1) -> bool:
    """
    Check if a cache file is stale (too old).
    
    Args:
        filepath: Path to cache file
        max_age_days: Maximum age in days before considering stale
        
    Returns:
        True if file is stale
        
    Example:
        >>> is_cache_stale('data/raw/AAPL_20240101_20241230.csv', max_age_days=1)
        False
    """
    try:
        # Check file modification time
        mod_time = os.path.getmtime(filepath)
        mod_datetime = datetime.fromtimestamp(mod_time)
        age_days = (datetime.now() - mod_datetime).days
        
        return age_days > max_age_days
        
    except Exception:
        return True  # Consider missing/unreadable files as stale

def validate_cached_data(df: pd.DataFrame, ticker: str) -> bool:
    """
    Validate that cached data has the correct format for backtesting.
    
    Args:
        df: DataFrame loaded from cache
        ticker: Ticker symbol for error messages
        
    Returns:
        True if data is valid for backtesting
        
    Example:
        >>> df = pd.read_csv('data/raw/AAPL_20240101_20241230.csv', index_col=0, parse_dates=True)
        >>> validate_cached_data(df, 'AAPL')
        True
    """
    if df.empty:
        print(f"❌ {ticker}: Cache file is empty")
        return False
    
    # Check required columns
    required_columns = ['Close', 'Volume']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"❌ {ticker}: Missing required columns: {missing_columns}")
        print(f"   Available columns: {list(df.columns)}")
        return False
    
    # Check for valid price data
    if (df['Close'] <= 0).any():
        invalid_count = (df['Close'] <= 0).sum()
        print(f"❌ {ticker}: {invalid_count} non-positive prices found")
        return False
    
    # Check for missing data
    if df['Close'].isnull().any():
        missing_count = df['Close'].isnull().sum()
        print(f"❌ {ticker}: {missing_count} missing price values")
        return False
    
    # Check for missing data in volume
    if df['Volume'].isnull().any():
        missing_vol = df['Volume'].isnull().sum()
        print(f"❌ {ticker}: {missing_vol} missing volume values")
        return False

    # Ensure data types are numeric
    if not pd.api.types.is_numeric_dtype(df['Close']):
        print(f"❌ {ticker}: 'Close' column is not numeric (dtype={df['Close'].dtype})")
        return False
    if not pd.api.types.is_numeric_dtype(df['Volume']):
        print(f"❌ {ticker}: 'Volume' column is not numeric (dtype={df['Volume'].dtype})")
        return False

    # Check index is sorted ascending
    if not df.index.is_monotonic_increasing:
        print(f"❌ {ticker}: Index is not sorted in ascending order")
        return False

    # Ensure index is timezone-naive (yfinance sometimes keeps tz)
    if df.index.tz is not None:
        print(f"❌ {ticker}: DatetimeIndex has timezone info ({df.index.tz}), expected naive")
        return False

    # Ensure at least 50 data points for meaningful analysis
    if len(df) < 50:
        print(f"❌ {ticker}: Only {len(df)} data points, insufficient history")
        return False
    
    # Check date index
    if not isinstance(df.index, pd.DatetimeIndex):
        print(f"❌ {ticker}: Index is not DatetimeIndex, got {type(df.index)}")
        return False
    
    # Check for duplicate dates
    if df.index.duplicated().any():
        duplicate_count = df.index.duplicated().sum()
        print(f"❌ {ticker}: {duplicate_count} duplicate dates found")
        return False
    
    # Check data continuity (allow for weekends/holidays)
    date_gaps = df.index.to_series().diff().dt.days
    large_gaps = date_gaps[date_gaps > 7]  # More than a week gap
    if len(large_gaps) > 2:  # Allow a couple of holiday periods
        print(f"⚠️  {ticker}: {len(large_gaps)} large gaps in data (>7 days)")
        # Don't fail validation for this, just warn
    
    return True

def load_and_validate_cache(filepath: str, ticker: str) -> Optional[pd.DataFrame]:
    """
    Load cache file and validate its contents.
    
    Args:
        filepath: Path to cache file
        ticker: Ticker symbol
        
    Returns:
        Valid DataFrame or None if validation fails
        
    Example:
        >>> df = load_and_validate_cache('data/raw/AAPL_20240101_20241230.csv', 'AAPL')
        >>> df is not None
        True
    """
    try:
        # Load the file
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        
        # Validate the data
        if validate_cached_data(df, ticker):
            return df
        else:
            print(f"🗑️  Cache validation failed for {ticker}, will download fresh data")
            return None
            
    except Exception as e:
        print(f"❌ Error loading cache file {filepath}: {e}")
        return None 