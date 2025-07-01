# Rate Limit Improvements for yfinance Data Download

## Overview
Based on research using Perplexity MCP, we've implemented several improvements to handle Yahoo Finance rate limits and optimize data downloads.

## Key Improvements

### 1. Rate Limit Handling
- **Exponential Backoff**: When a 429 error is detected, the system waits with exponentially increasing delays
- **Reduced Concurrent Workers**: Decreased from 10 to 5 concurrent downloads to avoid triggering rate limits
- **Batch Processing**: Downloads are processed in batches of 10 tickers with delays between batches

### 2. Caching System
- **Local CSV Cache**: Downloaded data is automatically saved to `data/raw/` directory
- **Smart Cache Loading**: Before downloading, the system checks if cached data exists and covers the requested date range
- **File Naming Convention**: `{TICKER}_{START_DATE}_{END_DATE}.csv` (e.g., `AAPL_20230703_20250627.csv`)

### 3. Error Handling
- **Specific 429 Detection**: Checks for "429" or "Too Many Requests" in error messages
- **Retry Logic**: 3 attempts per ticker with increasing delays
- **Multi-level Column Handling**: Properly handles yfinance's multi-level column format

## Usage Example

```python
from src.market_data.downloader import download_universe

# Download with caching enabled (default)
price_data = download_universe(
    tickers=['AAPL', 'MSFT', 'GOOGL'],
    start='2023-01-01',
    end='2024-12-31',
    max_workers=3,  # Reduced to avoid rate limits
    use_cache=True,  # Enable caching
    cache_dir='data/raw'  # Cache directory
)

# Second call will load from cache (much faster)
price_data = download_universe(
    tickers=['AAPL', 'MSFT', 'GOOGL'],
    start='2023-01-01',
    end='2024-12-31'
)
```

## Performance Results
From our tests:
- First download (5 tickers): ~2.10 seconds
- Subsequent loads from cache: ~0.05 seconds (40x faster)
- Successfully downloaded 20 tickers without rate limit errors
- Average time per ticker: ~0.20 seconds with rate limit protection

## Best Practices
1. **Use Caching**: Always enable caching for repeated downloads
2. **Limit Concurrent Workers**: Keep max_workers at 3-5
3. **Batch Large Downloads**: Process large ticker lists in smaller batches
4. **Monitor Rate Limits**: Watch for 429 errors and adjust delays if needed

## Debug Commands (Left Commented)
```python
# Clear cache for testing
# import shutil
# if os.path.exists("data/raw"):
#     shutil.rmtree("data/raw")
#     os.makedirs("data/raw")

# Test with problem tickers
# problem_tickers = ['BRK-B', 'BF-B']  # Special characters
# data = download_universe(problem_tickers, start_date, end_date)

# Check cache contents
# import os
# files = os.listdir("data/raw")
# print(f"Cached files: {len(files)}")
``` 