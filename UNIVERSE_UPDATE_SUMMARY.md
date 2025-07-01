# Universe Update Summary

## Changes Made

### 1. New Universe Function
- Added `create_sp100_since_2010()` function in `src/market_data/downloader.py`
- This function returns 61 tickers:
  - 60 largest stocks that have been in the S&P 100 since 2010-01-01
  - Plus Tesla (TSLA) as specifically requested

### 2. Updated Files
- **src/market_data/downloader.py**: Added new universe function
- **src/market_data/universe.py**: Updated imports to include new function
- **src/gui/controllers.py**: 
  - Updated imports
  - Replaced `create_sp100_list()` calls with `create_sp100_since_2010()`
  - Removed `get_sp100_tickers()` method, using `get_universe_list()` instead
- **src/gui/app.py**: Updated to use `get_universe_list()` method
- **scripts/run_baseline_backtest.py**: Updated to use new universe
- **scripts/run_cleir_backtest.py**: Updated to use new universe
- **scripts/generate_final_results.py**: Updated to use new universe

### 3. Universe Composition
The new universe includes these 61 tickers:

**Mega Caps (7):**
AAPL, MSFT, AMZN, GOOGL, META, NVDA, BRK-B

**Large Caps (53):**
UNH, JNJ, JPM, V, PG, XOM, HD, CVX, MA, PFE, BAC, ABBV, KO, LLY, PEP, TMO, COST, WMT, DIS, ABT, VZ, ACN, CMCSA, NKE, TXN, LIN, ORCL, ADBE, CRM, MDT, PM, BMY, T, HON, QCOM, LOW, UPS, AMD, C, RTX, INTU, CAT, AMGN, DE, GS, MO, AXP, BLK, GILD, MDLZ, MMM, CVS, SO

**Special Addition (1):**
TSLA (Tesla - added per user request)

### 4. Selection Criteria
- All stocks (except TSLA) have been consistently in the S&P 100 index since January 1, 2010
- Selected the 60 largest by current market capitalization
- Tesla added as the 61st stock per specific user request

### 5. Usage
The new universe will be automatically used when running:
- The GUI application (`python3 run_gui.py`)
- Baseline backtests
- CLEIR optimization
- Final results generation

The universe selection feature (if enabled) will further filter these 61 stocks based on liquidity metrics. 