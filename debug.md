# Granular Debugging Plan: Caching vs. Live Data Discrepancy

## 🎯 Objective
Pinpoint and resolve the exact data discrepancy between yfinance live downloads and CSV cache loads that causes the `Index should start at 100.0` optimization error.

## 🧐 Hypotheses
The root cause is likely a subtle difference in the DataFrame structure or data types when loading from CSV vs. using the in-memory object from `yfinance`.

1.  **Data Type Mismatch:** CSVs don't preserve `float64` or `int64` types perfectly. They might be read back as `object` or a different float precision, which downstream calculations might not handle correctly.
2.  **DatetimeIndex Corruption:** The `DatetimeIndex` is the most likely culprit. Information like timezone (`tz`), frequency (`freq`), or slight start/end date mismatches due to market hours vs. full days can be lost in CSV conversion.
3.  **Data Alignment Mismatch:** The `align_data_by_dates` function might behave differently when aligning a set of perfectly-in-sync downloaded DataFrames versus a set of individually-loaded CSVs with potentially misaligned start/end dates (e.g., one starts on a Monday, another on a Tuesday).
4.  **Inadequate Cache File Selection:** The `get_best_cache_file` logic might be selecting a file that *mostly* covers the required date range but misses the exact first day needed by the backtest, causing an initialization error.
5.  **Data Format Change:** The use of `auto_adjust=True` in `yfinance` handles splits and dividends. This complex adjustment may not be perfectly represented in a simple CSV, leading to different starting prices on a given day.

## 🛠️ Debugging & Analysis Plan

### Phase 1: Isolate & Compare (The "Smoking Gun" Phase)
The goal here is to get a direct, side-by-side comparison of a live DataFrame and a cached DataFrame.

**Step 1.1: Create an Analysis Script (`scripts/compare_cache_vs_live.py`)**
- **Action:** Create a new Python script.
- **Functionality:**
    1.  **Define a Test Case:** Use a single ticker (`'AAPL'`) and a fixed date range (`'2024-01-01'` to `'2024-06-01'`).
    2.  **Live Download:** Use `yf.download()` to get the data directly. Store this in a `df_live` variable.
    3.  **Cache Cycle:**
        - Use the same `yf.download()` call to create a fresh CSV cache file.
        - Immediately load that *exact* file back into a `df_cached` variable.
    4.  **Direct Comparison:**
        - **Equality Check:** `df_live.equals(df_cached)`. This will almost certainly be `False` and is our starting point.
        - **DataFrame Info:** Print `df_live.info()` and `df_cached.info()`. Compare `Dtype` and `memory usage` for each column.
        - **Index Comparison:** `df_live.index.equals(df_cached.index)`. Check and print `df_live.index.dtype`, `df_cached.index.dtype`, and timezone info.
        - **Head-to-Head:** Print the first 5 rows (`.head(5)`) of both DataFrames. Are the values identical? Pay close attention to the number of decimal places.
        - **Column-by-Column:** Use `pd.testing.assert_series_equal(df_live['Close'], df_cached['Close'])` inside a `try...except` block to see which column fails the equality test first.

### Phase 2: Audit the Data Pipeline
If the direct comparison doesn't reveal an obvious flaw, we need to look at how data flows through the system.

**Step 2.1: Instrument `download_universe`**
- **Action:** Add detailed logging to `src/market_data/downloader.py` inside the `download_universe` function.
- **Logging Points:**
    - Before the cache check loop, log the tickers requested.
    - Inside the loop, log whether each ticker resulted in a "cache hit" or "cache miss."
    - After all data is collected (from cache and downloads), but *before* `align_data_by_dates` is called, log the `data_dict`. Specifically, for each ticker's DataFrame, log its `shape`, `start_date`, `end_date`, and `dtypes`.
    - After `align_data_by_dates` is called, log the shape and date range of the final `price_df`.
- **Goal:** See if the collection of individual DataFrames from cache is different from the collection of freshly downloaded ones.

### Phase 3: Test a More Robust Caching Format
CSVs are convenient but notoriously poor for scientific data persistence. A likely fix is to use a format that preserves DataFrame metadata.

**Step 3.1: Research Alternatives (`web_search`)**
- **Action:** Use the `web_search` tool to search for "python pandas save dataframe with metadata" or "yfinance cache implementation".
- **Goal:** Confirm best practices. Common alternatives are Pickle (`.pkl`), Parquet (`.parquet`), and Feather (`.feather`). Pickle is simplest and already used in `save_price_data`.

**Step 3.2: Implement Pickle-based Caching**
- **Action:** Modify `save_ticker_data_to_csv` and `load_ticker_data_from_csv` to use `pd.to_pickle()` and `pd.read_pickle()`.
- **Details:**
    - Rename the functions to `save_ticker_data_to_pickle` etc.
    - Change filenames to end in `.pkl`.
    - Purge the old `data/raw/*.csv` files to ensure the new system is used.
- **Rationale:** Pickle stores the Python object directly, preserving all dtypes, index properties, and other metadata, which is a very strong candidate for fixing this issue.

### Phase 4: Propose the Final Fix
- **Action:** Based on the findings from the previous phases, implement the most robust solution. This will likely be the switch to Pickle caching.
- **Validation:** Rerun the GUI and the failing backtest. Confirm that the "Index should start at 100.0" error is gone.
- **Cleanup:** Remove the debugging script and any extra logging.

---
This plan provides a structured way to identify the problem, test a solution, and implement it robustly. I will start with **Phase 1, Step 1.1**. 