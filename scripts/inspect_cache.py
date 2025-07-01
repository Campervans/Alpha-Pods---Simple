#!/usr/bin/env python3
"""
Cache Inspector Script
Analyzes what stock data is cached in data/raw directory.
"""

import os
import sys
from pathlib import Path
import pandas as pd
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

def parse_cache_filename(filename: str) -> dict:
    """
    Parse cache filename to extract ticker and date range.
    
    Expected format: TICKER_YYYYMMDD_YYYYMMDD.csv
    Example: AAPL_20090701_20241230.csv
    
    Returns:
        dict with keys: ticker, start_date, end_date, filename
        or None if parsing fails
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
            
        # Convert to readable format
        start_formatted = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:8]}"
        end_formatted = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:8]}"
        
        return {
            'ticker': ticker,
            'start_date': start_formatted,
            'end_date': end_formatted,
            'filename': filename,
            'start_raw': start_date,
            'end_raw': end_date
        }
        
    except Exception as e:
        print(f"Warning: Could not parse filename '{filename}': {e}")
        return None

def get_file_info(filepath: str) -> dict:
    """Get file size and modification time."""
    try:
        stat = os.stat(filepath)
        return {
            'size_mb': stat.st_size / (1024 * 1024),
            'modified': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M')
        }
    except Exception:
        return {'size_mb': 0, 'modified': 'Unknown'}

def count_data_points(filepath: str) -> int:
    """Count rows in CSV file."""
    try:
        df = pd.read_csv(filepath)
        return len(df)
    except Exception:
        return 0

def inspect_cache_directory(cache_dir: str = "data/raw") -> list:
    """
    Analyze all CSV files in the cache directory.
    
    Returns:
        List of dictionaries with cache file information
    """
    if not os.path.exists(cache_dir):
        print(f"Cache directory '{cache_dir}' does not exist!")
        return []
    
    cache_files = []
    csv_files = [f for f in os.listdir(cache_dir) if f.endswith('.csv')]
    
    print(f"Found {len(csv_files)} CSV files in {cache_dir}")
    
    for filename in csv_files:
        filepath = os.path.join(cache_dir, filename)
        parsed = parse_cache_filename(filename)
        
        if parsed:
            file_info = get_file_info(filepath)
            data_points = count_data_points(filepath)
            
            cache_info = {
                **parsed,
                **file_info,
                'data_points': data_points,
                'filepath': filepath
            }
            cache_files.append(cache_info)
        else:
            print(f"Skipping unparseable file: {filename}")
    
    return cache_files

def print_cache_summary(cache_files: list):
    """Print a nicely formatted summary of cached data."""
    if not cache_files:
        print("No cache files found!")
        return
    
    print("\n" + "="*80)
    print("CACHE SUMMARY")
    print("="*80)
    
    # Sort by ticker
    cache_files.sort(key=lambda x: x['ticker'])
    
    # Group by ticker
    tickers = {}
    for file_info in cache_files:
        ticker = file_info['ticker']
        if ticker not in tickers:
            tickers[ticker] = []
        tickers[ticker].append(file_info)
    
    print(f"{'Ticker':<8} {'Date Range':<25} {'Days':<8} {'Size':<8} {'Modified':<16}")
    print("-" * 80)
    
    total_size = 0
    total_files = 0
    unique_tickers = set()
    
    for ticker, files in tickers.items():
        unique_tickers.add(ticker)
        for file_info in files:
            date_range = f"{file_info['start_date']} to {file_info['end_date']}"
            size_str = f"{file_info['size_mb']:.1f}MB"
            days_str = f"{file_info['data_points']:,}"
            
            print(f"{ticker:<8} {date_range:<25} {days_str:<8} {size_str:<8} {file_info['modified']:<16}")
            
            total_size += file_info['size_mb']
            total_files += 1
    
    print("-" * 80)
    print(f"TOTALS: {len(unique_tickers)} unique tickers, {total_files} files, {total_size:.1f}MB")
    
    # Find date coverage
    if cache_files:
        all_starts = [f['start_raw'] for f in cache_files]
        all_ends = [f['end_raw'] for f in cache_files]
        earliest = min(all_starts)
        latest = max(all_ends)
        
        earliest_formatted = f"{earliest[:4]}-{earliest[4:6]}-{earliest[6:8]}"
        latest_formatted = f"{latest[:4]}-{latest[4:6]}-{latest[6:8]}"
        
        print(f"DATE RANGE: {earliest_formatted} to {latest_formatted}")
    
    print("="*80)

def main():
    """Main function to run cache inspection."""
    print("Cache Inspector - Analyzing data/raw directory")
    print("-" * 50)
    
    cache_files = inspect_cache_directory("data/raw")
    print_cache_summary(cache_files)
    
    # Additional analysis
    if cache_files:
        print("\nDETAILED ANALYSIS:")
        
        # Find duplicates (same ticker, overlapping dates)
        tickers = {}
        for file_info in cache_files:
            ticker = file_info['ticker']
            if ticker not in tickers:
                tickers[ticker] = []
            tickers[ticker].append(file_info)
        
        duplicates = {k: v for k, v in tickers.items() if len(v) > 1}
        if duplicates:
            print(f"⚠️  Found {len(duplicates)} tickers with multiple cache files:")
            for ticker, files in duplicates.items():
                print(f"   {ticker}: {len(files)} files")
        else:
            print("✅ No duplicate ticker files found")
        
        # Find recent vs old data
        recent_files = [f for f in cache_files if '2024' in f['end_date']]
        old_files = [f for f in cache_files if '2024' not in f['end_date']]
        
        print(f"📊 Data freshness:")
        print(f"   Recent (includes 2024): {len(recent_files)} files")
        print(f"   Older data: {len(old_files)} files")
        
        if old_files:
            print("⚠️  Old cache files that may need updating:")
            for f in old_files[:5]:  # Show first 5
                print(f"   {f['ticker']}: ends {f['end_date']}")
            if len(old_files) > 5:
                print(f"   ... and {len(old_files) - 5} more")

if __name__ == "__main__":
    main() 