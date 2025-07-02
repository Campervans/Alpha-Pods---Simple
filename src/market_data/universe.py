"""
Universe selection functionality.

This module handles the selection of investment universe based on liquidity
and other criteria from a larger set of available securities.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from ..utils.schemas import PriceData, UniverseConfig
from .downloader import download_universe, create_sp100_list, create_sp100_since_2010


def calculate_liquidity_scores(price_data: PriceData, config: UniverseConfig) -> pd.Series:
    """
    Calculate average dollar volume for each ticker.
    
    Args:
        price_data: PriceData object containing prices and volumes
        config: UniverseConfig with parameters for liquidity calculation
        
    Returns:
        Series with liquidity scores for each ticker
        
    Example:
        >>> scores = calculate_liquidity_scores(price_data, config)
        >>> scores.sort_values(ascending=False).head()
    """
    if price_data.volumes is None:
        raise ValueError("Volume data is required for liquidity calculation")
    
    # Calculate dollar volume (price * volume)
    dollar_volumes = price_data.prices * price_data.volumes
    
    # Calculate rolling average over lookback period
    if config.metric == "dollar_volume":
        liquidity_metric = dollar_volumes
    elif config.metric == "volume":
        liquidity_metric = price_data.volumes
    else:
        raise ValueError(f"Unsupported liquidity metric: {config.metric}")
    
    # Use the last N days for calculation
    if len(liquidity_metric) >= config.lookback_days:
        recent_data = liquidity_metric.tail(config.lookback_days)
    else:
        recent_data = liquidity_metric
    
    # Calculate mean liquidity score for each ticker
    liquidity_scores = recent_data.mean()
    
    # Handle any NaN values
    liquidity_scores = liquidity_scores.fillna(0)
    
    return liquidity_scores


def apply_universe_filters(price_data: PriceData, config: UniverseConfig) -> Tuple[List[str], pd.DataFrame]:
    """
    Apply universe selection filters to identify valid tickers.
    
    Args:
        price_data: PriceData object containing prices and volumes
        config: UniverseConfig with filtering parameters
        
    Returns:
        Tuple of (valid_tickers, filter_results_df)
        
    Example:
        >>> valid_tickers, results = apply_universe_filters(price_data, config)
    """
    results = []
    
    for ticker in price_data.tickers:
        ticker_prices = price_data.prices[ticker]
        ticker_volumes = price_data.volumes[ticker] if price_data.volumes is not None else None
        
        # Initialize filter results
        filters = {
            'ticker': ticker,
            'min_price_filter': True,
            'min_trading_days_filter': True,
            'valid_data_filter': True,
            'final_price': ticker_prices.iloc[-1] if len(ticker_prices) > 0 else 0,
            'trading_days': len(ticker_prices.dropna()),
            'avg_volume': ticker_volumes.mean() if ticker_volumes is not None else 0
        }
        
        # Filter 1: Minimum price filter
        if filters['final_price'] < config.min_price:
            filters['min_price_filter'] = False
        
        # Filter 2: Minimum trading days filter
        recent_data = ticker_prices.tail(config.lookback_days)
        valid_days = len(recent_data.dropna())
        if valid_days < config.min_trading_days:
            filters['min_trading_days_filter'] = False
        
        # Filter 3: Valid data filter (no zeros, reasonable values)
        if (ticker_prices <= 0).any() or ticker_prices.isnull().any():
            filters['valid_data_filter'] = False
        
        filters['passes_all_filters'] = all([
            filters['min_price_filter'],
            filters['min_trading_days_filter'], 
            filters['valid_data_filter']
        ])
        
        results.append(filters)
    
    # Convert to DataFrame for analysis
    results_df = pd.DataFrame(results)
    
    # Get list of valid tickers
    valid_tickers = results_df[results_df['passes_all_filters']]['ticker'].tolist()
    
    print(f"Universe filtering results:")
    print(f"  Starting tickers: {len(price_data.tickers)}")
    print(f"  Passed min price filter (${config.min_price}): {results_df['min_price_filter'].sum()}")
    print(f"  Passed min trading days filter ({config.min_trading_days}): {results_df['min_trading_days_filter'].sum()}")
    print(f"  Passed valid data filter: {results_df['valid_data_filter'].sum()}")
    print(f"  Final valid tickers: {len(valid_tickers)}")
    
    return valid_tickers, results_df


def select_liquid_universe(sp100_tickers: List[str], config: UniverseConfig,
                          start_date: str = "2023-01-01", end_date: str = "2024-12-31") -> List[str]:
    """
    Select top N most liquid stocks from S&P 100.
    
    Args:
        sp100_tickers: List of S&P 100 ticker symbols
        config: UniverseConfig with selection parameters
        start_date: Start date for liquidity calculation
        end_date: End date for liquidity calculation
        
    Returns:
        List of selected ticker symbols
        
    Example:
        >>> tickers = create_sp100_list()
        >>> config = UniverseConfig(n_stocks=60)
        >>> universe = select_liquid_universe(tickers, config)
    """
    print(f"Selecting {config.n_stocks} most liquid stocks from {len(sp100_tickers)} candidates")
    
    # Download recent data for liquidity analysis
    try:
        price_data = download_universe(
            sp100_tickers, 
            start_date, 
            end_date,
            min_data_points=config.min_trading_days
        )
    except Exception as e:
        print(f"Error downloading data for universe selection: {e}")
        # Fallback to first N tickers if download fails
        print(f"Using first {config.n_stocks} tickers as fallback")
        return sp100_tickers[:config.n_stocks]
    
    # Apply universe filters
    valid_tickers, filter_results = apply_universe_filters(price_data, config)
    
    if len(valid_tickers) < config.n_stocks:
        print(f"Warning: Only {len(valid_tickers)} tickers passed filters, "
              f"requested {config.n_stocks}")
        return valid_tickers
    
    # Calculate liquidity scores for valid tickers
    valid_price_data = PriceData(
        tickers=valid_tickers,
        dates=price_data.dates,
        prices=price_data.prices[valid_tickers],
        volumes=price_data.volumes[valid_tickers] if price_data.volumes is not None else None
    )
    
    liquidity_scores = calculate_liquidity_scores(valid_price_data, config)
    
    # Select top N most liquid tickers
    top_liquid_tickers = liquidity_scores.nlargest(config.n_stocks).index.tolist()
    
    print(f"Selected universe statistics:")
    print(f"  Top liquidity score: ${liquidity_scores.max():,.0f}")
    print(f"  Bottom liquidity score: ${liquidity_scores.nsmallest(config.n_stocks).iloc[-1]:,.0f}")
    print(f"  Median liquidity score: ${liquidity_scores.median():,.0f}")
    
    return top_liquid_tickers


def create_equal_weight_universe(tickers: List[str]) -> np.ndarray:
    """
    Create equal weight portfolio from universe.
    
    Args:
        tickers: List of ticker symbols
        
    Returns:
        Array of equal weights
        
    Example:
        >>> weights = create_equal_weight_universe(['AAPL', 'MSFT', 'GOOGL'])
        >>> weights  # [0.333, 0.333, 0.333]
    """
    n_assets = len(tickers)
    if n_assets == 0:
        return np.array([])
    
    return np.ones(n_assets) / n_assets


def create_market_cap_weights(price_data: PriceData, shares_outstanding: Optional[Dict[str, float]] = None) -> np.ndarray:
    """
    Create market cap weighted portfolio (if shares data is available).
    
    Args:
        price_data: PriceData object
        shares_outstanding: Optional dictionary mapping ticker to shares outstanding
        
    Returns:
        Array of market cap weights (or equal weights if shares data unavailable)
        
    Example:
        >>> shares = {'AAPL': 16e9, 'MSFT': 7.4e9}
        >>> weights = create_market_cap_weights(price_data, shares)
    """
    if shares_outstanding is None:
        print("No shares outstanding data provided, using equal weights")
        return create_equal_weight_universe(price_data.tickers)
    
    # Calculate market caps using latest prices
    latest_prices = price_data.prices.iloc[-1]
    market_caps = {}
    
    for ticker in price_data.tickers:
        if ticker in shares_outstanding:
            market_caps[ticker] = latest_prices[ticker] * shares_outstanding[ticker]
        else:
            print(f"No shares data for {ticker}, using equal weight proxy")
            market_caps[ticker] = latest_prices[ticker] * 1e9  # Proxy value
    
    # Convert to weights
    total_market_cap = sum(market_caps.values())
    weights = np.array([market_caps[ticker] / total_market_cap for ticker in price_data.tickers])
    
    return weights


def analyze_universe_characteristics(price_data: PriceData, config: UniverseConfig) -> pd.DataFrame:
    """
    Analyze characteristics of the selected universe.
    
    Args:
        price_data: PriceData object for the universe
        config: UniverseConfig used for selection
        
    Returns:
        DataFrame with universe characteristics
        
    Example:
        >>> analysis = analyze_universe_characteristics(price_data, config)
        >>> print(analysis)
    """
    analysis_data = []
    
    for ticker in price_data.tickers:
        ticker_prices = price_data.prices[ticker]
        ticker_volumes = price_data.volumes[ticker] if price_data.volumes is not None else None
        
        # Calculate returns
        returns = ticker_prices.pct_change().dropna()
        
        # Calculate statistics
        stats = {
            'ticker': ticker,
            'start_price': ticker_prices.iloc[0],
            'end_price': ticker_prices.iloc[-1],
            'total_return': (ticker_prices.iloc[-1] / ticker_prices.iloc[0]) - 1,
            'annualized_return': ((ticker_prices.iloc[-1] / ticker_prices.iloc[0]) ** (252 / len(ticker_prices))) - 1,
            'annualized_volatility': returns.std() * np.sqrt(252),
            'avg_daily_volume': ticker_volumes.mean() if ticker_volumes is not None else 0,
            'avg_dollar_volume': (ticker_prices * ticker_volumes).mean() if ticker_volumes is not None else 0,
            'max_drawdown': calculate_max_drawdown_simple(ticker_prices),
            'days_of_data': len(ticker_prices)
        }
        
        analysis_data.append(stats)
    
    analysis_df = pd.DataFrame(analysis_data)
    
    # Add summary statistics
    print(f"\nUniverse Analysis Summary:")
    print(f"Number of assets: {len(analysis_df)}")
    print(f"Average annualized return: {analysis_df['annualized_return'].mean():.2%}")
    print(f"Average annualized volatility: {analysis_df['annualized_volatility'].mean():.2%}")
    print(f"Average dollar volume: ${analysis_df['avg_dollar_volume'].mean():,.0f}")
    print(f"Date range: {price_data.start_date.date()} to {price_data.end_date.date()}")
    
    return analysis_df


def calculate_max_drawdown_simple(prices: pd.Series) -> float:
    """
    Simple maximum drawdown calculation for individual assets.
    
    Args:
        prices: Series of asset prices
        
    Returns:
        Maximum drawdown as a positive percentage
    """
    if len(prices) == 0:
        return 0.0
    
    # Calculate running maximum
    running_max = prices.expanding().max()
    
    # Calculate drawdown
    drawdown = (prices - running_max) / running_max
    
    return abs(drawdown.min())


def save_universe_selection_results(tickers: List[str], 
                                   filter_results: pd.DataFrame,
                                   liquidity_scores: pd.Series,
                                   filepath: str):
    """
    Save universe selection results to Excel file.
    
    Args:
        tickers: Selected ticker symbols
        filter_results: DataFrame with filter results
        liquidity_scores: Series with liquidity scores
        filepath: Path to save Excel file
        
    Example:
        >>> save_universe_selection_results(tickers, results, scores, 'universe_selection.xlsx')
    """
    with pd.ExcelWriter(filepath, engine='xlsxwriter') as writer:
        # Sheet 1: Selected universe
        selected_df = pd.DataFrame({'ticker': tickers})
        selected_df['rank'] = range(1, len(tickers) + 1)
        if len(liquidity_scores) > 0:
            selected_df['liquidity_score'] = [liquidity_scores.get(ticker, 0) for ticker in tickers]
        selected_df.to_excel(writer, sheet_name='Selected_Universe', index=False)
        
        # Sheet 2: Filter results
        filter_results.to_excel(writer, sheet_name='Filter_Results', index=False)
        
        # Sheet 3: All liquidity scores
        if len(liquidity_scores) > 0:
            liquidity_df = liquidity_scores.reset_index()
            liquidity_df.columns = ['ticker', 'liquidity_score']
            liquidity_df = liquidity_df.sort_values('liquidity_score', ascending=False)
            liquidity_df['rank'] = range(1, len(liquidity_df) + 1)
            liquidity_df.to_excel(writer, sheet_name='Liquidity_Scores', index=False)
    
    print(f"Universe selection results saved to {filepath}")


def load_universe_from_file(filepath: str, sheet_name: str = 'Selected_Universe') -> List[str]:
    """
    Load universe selection from Excel file.
    
    Args:
        filepath: Path to Excel file
        sheet_name: Name of the sheet containing tickers
        
    Returns:
        List of ticker symbols
        
    Example:
        >>> tickers = load_universe_from_file('universe_selection.xlsx')
    """
    try:
        df = pd.read_excel(filepath, sheet_name=sheet_name)
        tickers = df['ticker'].tolist()
        print(f"Loaded {len(tickers)} tickers from {filepath}")
        return tickers
    except Exception as e:
        print(f"Error loading universe from file: {e}")
        return []


def validate_universe_selection(tickers: List[str], 
                               price_data: PriceData,
                               min_correlation: float = 0.3,
                               max_correlation: float = 0.9) -> Dict[str, any]:
    """
    Validate the selected universe for diversification and other criteria.
    
    Args:
        tickers: Selected ticker symbols
        price_data: PriceData object
        min_correlation: Minimum acceptable average correlation
        max_correlation: Maximum acceptable average correlation
        
    Returns:
        Dictionary with validation results
        
    Example:
        >>> validation = validate_universe_selection(tickers, price_data)
    """
    # Calculate correlation matrix
    returns = price_data.get_returns()
    correlation_matrix = returns.corr()
    
    # Calculate average pairwise correlation
    n_assets = len(tickers)
    if n_assets <= 1:
        avg_correlation = 0.0
    else:
        # Get upper triangle of correlation matrix (excluding diagonal)
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)
        correlations = correlation_matrix.values[mask]
        avg_correlation = np.mean(correlations)
    
    # Check concentration (max weight if equal weighted)
    max_weight = 1.0 / n_assets if n_assets > 0 else 0.0
    
    validation_results = {
        'n_assets': n_assets,
        'avg_correlation': avg_correlation,
        'max_weight_equal': max_weight,
        'correlation_warning': avg_correlation > max_correlation or avg_correlation < min_correlation,
        'concentration_warning': max_weight > 0.1,  # Warning if any asset > 10%
        'sufficient_diversification': n_assets >= 30,
        'date_range_days': len(price_data.dates),
        'validation_passed': True
    }
    
    # Overall validation
    if (validation_results['correlation_warning'] or 
        validation_results['concentration_warning'] or 
        not validation_results['sufficient_diversification']):
        validation_results['validation_passed'] = False
    
    print(f"\nUniverse Validation Results:")
    print(f"  Number of assets: {validation_results['n_assets']}")
    print(f"  Average correlation: {validation_results['avg_correlation']:.3f}")
    print(f"  Max weight (equal): {validation_results['max_weight_equal']:.1%}")
    print(f"  Validation passed: {validation_results['validation_passed']}")
    
    return validation_results


def get_ml_universe() -> List[str]:
    """Get the 60-stock universe for ML-enhanced strategies.
    
    These are the 60 most liquid stocks from S&P 100 that have
    consistent data availability from 2010-2024. This universe
    is used for both Task A and Task B to ensure consistency.
    
    Returns:
        List of 60 ticker symbols
        
    Note:
        This list was curated based on liquidity and data availability
        analysis performed in the baseline implementation.
    """
    # Remember: Keep it simple - this is the same universe used across the project
    return [
        'AAPL', 'ABBV', 'ACN', 'ADBE', 'ADI', 'ADP', 'AMGN', 'AMZN', 'APH', 'AVGO',
        'AXP', 'BAC', 'BKNG', 'BLK', 'BRK-B', 'CAT', 'CMCSA', 'COST', 'CRM', 'CVX',
        'DIS', 'EMR', 'GE', 'GILD', 'GOOGL', 'HD', 'HUM', 'JNJ', 'JPM', 'KLAC',
        'KO', 'LIN', 'LLY', 'LRCX', 'MA', 'MDLZ', 'META', 'MO', 'MSFT', 'NEE',
        'NKE', 'NVDA', 'ORCL', 'PEP', 'PFE', 'PG', 'RTX', 'SBUX', 'SLB', 'SPGI',
        'SYK', 'TMO', 'TSLA', 'TXN', 'UNH', 'USB', 'V', 'VZ', 'WMT', 'XOM'
    ] 