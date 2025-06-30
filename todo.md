# TODO: Debug Download and CLEIR Issues

## 🔴 Issue 1: Download Errors - "The truth value of a Series is ambiguous"

### Step 1: Add Enhanced Error Logging
- [ ] Open `src/market_data/downloader.py`
- [ ] Find the `download_single_ticker` function (line ~49)
- [ ] Locate the `yf.download` call (around line ~90)
- [ ] Add try/except with full traceback:
  ```python
  try:
      df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
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
  ```
- [ ] Save the file

### Step 2: Test Specific Problem Tickers
- [ ] Create test script `debug_download.py`:
  ```python
  from src.market_data.downloader import download_single_ticker
  import pandas as pd
  
  problem_tickers = ['AAPL', 'MSFT', 'AMZN', 'BRK-B', 'UNH']
  
  for ticker in problem_tickers:
      print(f"\nTesting {ticker}...")
      try:
          df = download_single_ticker(ticker, '2024-01-01', '2024-12-31')
          print(f"✓ {ticker}: Success! Got {len(df)} rows")
      except Exception as e:
          print(f"✗ {ticker}: Failed - {e}")
  ```
- [ ] Run: `python3 debug_download.py`
- [ ] Document which tickers fail and exact error messages

### Step 3: Check Cache Files
- [ ] List cache files: `ls -la data/raw/ | grep -E "AAPL|MSFT|AMZN|BRK-B|UNH"`
- [ ] If files exist, check their dates and sizes
- [ ] Try loading cached data manually:
  ```python
  from src.market_data.downloader import load_ticker_data_from_csv
  for ticker in ['AAPL', 'MSFT', 'AMZN', 'BRK-B', 'UNH']:
      data = load_ticker_data_from_csv(ticker, 'data/raw')
      if data is not None:
          print(f"{ticker}: Cached data has {len(data)} rows")
  ```

### Step 4: Test Without Cache
- [ ] Temporarily rename cache dir: `mv data/raw data/raw_backup`
- [ ] Run download test again
- [ ] Compare results with/without cache
- [ ] Restore cache: `mv data/raw_backup data/raw`

### Step 5: Fix MultiIndex Handling
- [ ] In `download_single_ticker`, after `df = yf.download(...)`, add:
  ```python
  # Debug print to see what we got
  print(f"Downloaded {ticker}: type={type(df)}, shape={df.shape if hasattr(df, 'shape') else 'N/A'}")
  if hasattr(df, 'columns'):
      print(f"Columns type: {type(df.columns)}, values: {df.columns.tolist()[:5]}...")
  ```
- [ ] Update MultiIndex handling (around line 105):
  ```python
  if isinstance(df.columns, pd.MultiIndex):
      print(f"{ticker} has MultiIndex columns: {df.columns.levels}")
      # More robust handling
      if len(df.columns.levels) > 1 and ticker in df.columns.levels[1]:
          df = df.xs(ticker, level=1, axis=1)
      elif len(df.columns.droplevel(1).unique()) == len(df.columns):
          df.columns = df.columns.droplevel(1)
  ```

### Step 6: Check yfinance Version
- [ ] Check current version: `pip show yfinance`
- [ ] Check for updates: `pip list --outdated | grep yfinance`
- [ ] If outdated, update: `pip install --upgrade yfinance`
- [ ] Document version before/after

## 🟡 Issue 2: FutureWarning - DataFrame.fillna deprecated

### Step 1: Fix fillna Warning
- [ ] Open `src/market_data/downloader.py`
- [ ] Go to line 278
- [ ] Find: `price_df = price_df.fillna(method='ffill')`
- [ ] Replace with: `price_df = price_df.ffill()`
- [ ] Save the file

### Step 2: Check for Other fillna Uses
- [ ] Search for other occurrences: `grep -n "fillna.*method" src/**/*.py`
- [ ] Fix any other instances found
- [ ] Also check for 'bfill': `grep -n "fillna.*bfill" src/**/*.py`

### Step 3: Test the Fix
- [ ] Run a simple test to ensure no warnings:
  ```python
  import warnings
  warnings.filterwarnings('error', category=FutureWarning)
  from src.market_data.downloader import align_data_by_dates
  # Should not raise FutureWarning
  ```

## 🟢 Issue 3: CLEIR Error - CVaRIndexBacktest constructor issues ✅ COMPLETE

### Fixed Issues:
- [x] Fixed `download_benchmark_data()` unexpected keyword argument
- [x] Fixed CVaRIndexBacktest constructor - removed invalid `backtest_config` parameter  
- [x] Fixed method calls - changed `backtest.run()` to `backtest.run_backtest(backtest_config)`
- [x] Fixed result access - updated from dictionary to attribute style
- [x] Added proper CLEIR support with `asset_tickers` parameter
- [x] Fixed `select_liquid_universe()` parameter mismatch (PriceData vs List[str])

### Changes Made:
1. ✅ Removed `cache_dir="data/raw"` from `download_benchmark_data` call
2. ✅ Fixed CVaRIndexBacktest constructor calls in both optimization methods
3. ✅ Updated result handling to use BacktestResults object properly
4. ✅ Enhanced CLEIR optimization to properly handle benchmark integration
5. ✅ Fixed PriceData object len() error by correcting function signatures

## 🔵 Issue 4: Performance and Rate Limiting

### Step 1: Analyze Current Settings
- [ ] Document current settings:
  - Batch size: 10 (line ~169)
  - Batch delay: 0.5s (line ~170)
  - Max workers: 5 (line ~168)
  - Retry delay: 1.0s (line ~49)
  - Max proxies tried: 5 (line ~58)

### Step 2: Create Configuration File
- [ ] Create `src/config/download_config.py`:
  ```python
  # Download configuration
  DOWNLOAD_CONFIG = {
      'batch_size': 10,
      'batch_delay': 1.0,  # Increase from 0.5
      'max_workers': 3,    # Reduce from 5
      'retry_delay': 2.0,  # Increase from 1.0
      'max_proxy_attempts': 5,
      'rate_limit_delay': 0.5  # New: delay between individual requests
  }
  ```

### Step 3: Implement Rate Limiting
- [ ] Add request counter and timing:
  ```python
  import time
  from collections import deque
  
  class RateLimiter:
      def __init__(self, max_requests=5, time_window=1.0):
          self.max_requests = max_requests
          self.time_window = time_window
          self.requests = deque()
      
      def wait_if_needed(self):
          now = time.time()
          # Remove old requests
          while self.requests and self.requests[0] < now - self.time_window:
              self.requests.popleft()
          # Wait if at limit
          if len(self.requests) >= self.max_requests:
              sleep_time = self.requests[0] + self.time_window - now
              if sleep_time > 0:
                  time.sleep(sleep_time)
          self.requests.append(now)
  ```

### Step 4: Add Proxy Success Tracking
- [ ] Track which proxies work best:
  ```python
  proxy_stats = {port: {'success': 0, 'fail': 0} for port in PROXY_PORTS}
  
  def get_best_proxy():
      # Sort by success rate
      sorted_proxies = sorted(
          proxy_stats.items(),
          key=lambda x: x[1]['success'] / (x[1]['success'] + x[1]['fail'] + 1),
          reverse=True
      )
      return sorted_proxies[0][0]
  ```

### Step 5: Test Performance Improvements
- [ ] Create benchmark script to test download speeds
- [ ] Test with different configurations
- [ ] Find optimal settings for reliability vs speed

## 📊 Testing and Validation

### Final Integration Test
- [ ] Run full S&P 100 download with fixes
- [ ] Run CVaR optimization from GUI
- [ ] Run CLEIR optimization from GUI
- [ ] Verify no warnings or errors
- [ ] Check that all deliverables generate correctly

### Documentation
- [ ] Update README with any new dependencies
- [ ] Document optimal rate limit settings
- [ ] Add troubleshooting section for common download errors

## 🚀 Quick Start Commands

```bash
# Test download fixes
python3 scripts/debug_download.py

# Test CLEIR fix
python3 -c "from src.gui.controllers import OptimizationController; print('CLEIR import OK')"

# Run GUI
python3 run_gui.py

# Check for warnings
python3 -W error::FutureWarning -c "from src.market_data.downloader import download_universe; print('No warnings!')"
```

## Progress Tracking

- [x] Issue 1: Download Errors (4/6 steps) - Added error logging, debug prints, improved MultiIndex handling
- [x] Issue 2: FutureWarning (3/3 steps) ✅ COMPLETE
- [x] Issue 3: CLEIR Error (6/6 steps) ✅ COMPLETE
- [ ] Issue 4: Performance (0/5 steps)
- [ ] Testing & Validation (0/5 steps)

**Total Progress: 13/22 steps completed**

## Completed Fixes:
1. ✅ Fixed CLEIR error - removed cache_dir parameter
2. ✅ Fixed FutureWarning - changed fillna to ffill()
3. ✅ Added enhanced error logging for downloads
4. ✅ Improved MultiIndex handling for yfinance data
5. ✅ Created debug script that confirms downloads work
6. ✅ Fixed CVaRIndexBacktest constructor - removed invalid backtest_config parameter
7. ✅ Fixed method calls - changed backtest.run() to backtest.run_backtest()
8. ✅ Fixed result access - updated from dictionary to attribute style
9. ✅ Fixed select_liquid_universe parameter mismatch - PriceData vs List[str]
10. ✅ Enhanced CLEIR with proper asset_tickers parameter for benchmark separation 