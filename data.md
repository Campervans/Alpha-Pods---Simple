# Data Integration and Caching Overhaul Task List

## Overview
Complete audit and overhaul of yfinance data integration, caching system, and data flow to CLEIR/CVaR optimizations. Focus on data consistency, error handling, and concurrent downloads with proxy support.

## Current Issues Identified
- Connection failures to yfinance (DNS/connection errors)
- Data validation errors ("Index should start at 100.0")
- Ticker misalignment between downloaded data and optimization requirements
- Thread safety issues with yfinance
- Inconsistent date ranges across different data sources
- Missing data handling for holidays/weekends

## Phase 1: Data Specification and Requirements

### 1.1 Define Strict Data Requirements
- [ ] Document exact date range: 2010-01-01 to 2024-12-31
- [ ] Define expected data structure from yfinance
  - [ ] Column names and types (Open, High, Low, Close, Volume, Adj Close)
  - [ ] Index type (DatetimeIndex with timezone handling)
  - [ ] NaN/missing data handling policy
- [ ] Document business day requirements (exclude weekends/holidays)
- [ ] Define minimum data requirements per ticker (e.g., min 252 trading days)

### 1.2 yfinance Data Characteristics
- [ ] Research yfinance data gaps and limitations
  - [ ] Holiday handling
  - [ ] Corporate actions (splits, dividends)
  - [ ] Delisted tickers
  - [ ] Data availability by ticker (IPO dates)
- [ ] Document yfinance rate limits and best practices
- [ ] Identify reliable proxy services for concurrent downloads

## Phase 2: Caching System Overhaul

### 2.1 Cache Structure Design
- [ ] Design cache metadata structure
  ```python
  {
    "ticker": "AAPL",
    "download_date": "2024-12-30",
    "start_date": "2010-01-01",
    "end_date": "2024-12-31",
    "data_points": 3773,
    "missing_dates": [],
    "data_hash": "...",
    "yfinance_version": "0.2.x"
  }
  ```
- [ ] Implement cache validation logic
- [ ] Create cache expiry policy (when to re-download)

### 2.2 Pickle File Management
- [ ] Standardize pickle file naming: `{TICKER}_{START}_{END}.pkl`
- [ ] Implement atomic file writes (write to temp, then rename)
- [ ] Add cache integrity checks (file corruption detection)
- [ ] Create cache cleanup utility for old/invalid files

### 2.3 Cache Loading Logic
- [ ] Implement strict date range validation on load
- [ ] Handle partial data scenarios
- [ ] Create fallback mechanism for cache misses
- [ ] Add logging for all cache operations

## Phase 3: Concurrent Download System

### 3.1 Process-Based Concurrency Design
- [ ] Replace ThreadPoolExecutor with ProcessPoolExecutor
- [ ] Design proxy rotation system
  ```python
  proxies = [
    {"http": "http://proxy1:port", "https": "https://proxy1:port"},
    {"http": "http://proxy2:port", "https": "https://proxy2:port"},
    # ...
  ]
  ```
- [ ] Implement download queue with proxy assignment
- [ ] Add retry logic with exponential backoff

### 3.2 Download Function Refactor
- [ ] Create single ticker download function (process-safe)
- [ ] Implement batch download coordinator
- [ ] Add progress tracking across processes
- [ ] Handle download failures gracefully

### 3.3 Error Handling
- [ ] Categorize error types (connection, data, validation)
- [ ] Implement specific retry strategies per error type
- [ ] Create fallback data sources (if available)
- [ ] Log all errors with context

## Phase 4: Data Validation Pipeline

### 4.1 Post-Download Validation
- [ ] Verify date range completeness
- [ ] Check for data anomalies (e.g., zero prices, extreme outliers)
- [ ] Validate column presence and types
- [ ] Ensure adjusted close price continuity

### 4.2 Pre-Optimization Validation
- [ ] Align all ticker data to common date range
- [ ] Handle missing data consistently
  - [ ] Forward fill for prices
  - [ ] Zero fill for volumes
- [ ] Verify minimum data requirements per ticker
- [ ] Create data quality report

### 4.3 Data Alignment for Optimizations
- [ ] Ensure all tickers have identical date indices
- [ ] Handle ticker universe changes over time
- [ ] Create consistent returns calculation
- [ ] Validate benchmark data alignment

## Phase 5: Integration with CLEIR/CVaR

### 5.1 Data Interface Specification
- [ ] Define PriceData schema requirements
- [ ] Create data transformation pipeline
- [ ] Ensure proper returns calculation
- [ ] Handle corporate actions in returns

### 5.2 Optimization Data Requirements
- [ ] Document CLEIR solver data expectations
- [ ] Document CVaR solver data expectations
- [ ] Create data validation for optimization input
- [ ] Handle edge cases (single ticker, short history)

### 5.3 Backtesting Data Flow
- [ ] Ensure consistent index calculation (starts at 100.0)
- [ ] Handle rebalancing period data slicing
- [ ] Validate transaction cost calculations
- [ ] Create performance attribution data

## Phase 6: GUI Integration Updates

### 6.1 Controller Refactoring
- [ ] Update download progress display
- [ ] Add cache status indicators
- [ ] Show data quality metrics
- [ ] Handle download failures gracefully in UI

### 6.2 Error Messaging
- [ ] Create user-friendly error messages
- [ ] Add data diagnostics panel
- [ ] Show cache hit/miss statistics
- [ ] Provide manual refresh options

## Phase 7: Testing and Validation

### 7.1 Unit Tests
- [ ] Test cache save/load functions
- [ ] Test data validation functions
- [ ] Test download retry logic
- [ ] Test proxy rotation

### 7.2 Integration Tests
- [ ] Test full download pipeline
- [ ] Test optimization with various data scenarios
- [ ] Test GUI workflows
- [ ] Test error recovery

### 7.3 Performance Tests
- [ ] Benchmark concurrent downloads
- [ ] Measure cache performance
- [ ] Profile memory usage
- [ ] Optimize bottlenecks

## Phase 8: Code Cleanup and Documentation

### 8.1 Remove Deprecated Code
- [ ] Remove old CSV-based functions
- [ ] Clean up unused imports
- [ ] Remove debug code
- [ ] Consolidate duplicate logic

### 8.2 Refactor for Clarity
- [ ] Extract constants to configuration
- [ ] Create clear module boundaries
- [ ] Improve function/variable naming
- [ ] Add type hints throughout

### 8.3 Documentation
- [ ] Document data flow architecture
- [ ] Create troubleshooting guide
- [ ] Document proxy configuration
- [ ] Add code examples

## Implementation Priority

1. **Critical** (Do First):
   - Fix date range consistency (2010-01-01 to 2024-12-31)
   - Implement process-based concurrent downloads
   - Fix data alignment issues causing optimization errors

2. **High Priority**:
   - Implement robust error handling
   - Add cache validation
   - Create data quality checks

3. **Medium Priority**:
   - Refactor code for clarity
   - Add comprehensive logging
   - Improve GUI feedback

4. **Low Priority**:
   - Performance optimizations
   - Extended documentation
   - Additional testing

## Success Criteria
- [ ] All tickers download successfully with retries
- [ ] Data consistently spans 2010-01-01 to 2024-12-31
- [ ] No "Index should start at 100.0" errors
- [ ] Concurrent downloads work with multiple proxies
- [ ] Cache hit rate > 90% for subsequent runs
- [ ] Clear error messages for any failures
- [ ] GUI remains responsive during downloads 