"""Baseline CLEIR strategy runner for comparison.

This module provides utilities to run the baseline CLEIR strategy
matching the same universe and configuration as ML-Enhanced CLEIR.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List
from datetime import datetime
import os
import pickle

from src.optimization.cleir_solver import solve_cleir
from src.utils.schemas import OptimizationConfig, PriceData
from src.market_data.universe import get_ml_universe
from src.utils.core import calculate_turnover, calculate_transaction_costs


def run_baseline_cleir(start_date: str = "2020-01-01", 
                      end_date: str = "2024-12-31",
                      transaction_cost_bps: float = 10.0) -> Dict[str, Any]:
    """Run baseline CLEIR strategy and return results dict.
    
    Args:
        start_date: Backtest start date
        end_date: Backtest end date
        transaction_cost_bps: Transaction cost in basis points
        
    Returns:
        Results dictionary matching ML-CLEIR format with:
            - daily_values: Portfolio value series
            - returns: Daily returns
            - portfolio_weights: Weight history
            - turnover_history: List of turnovers
            - transaction_costs_history: List of costs
            - total_return, annual_return, volatility, etc.
    """
    print(f"[blue]Running baseline CLEIR from {start_date} to {end_date}[/blue]")
    
    # 1. Load price data
    price_data = _load_price_data(start_date, end_date)
    universe_tickers = get_ml_universe()
    
    # Filter to available tickers
    available_tickers = [t for t in universe_tickers if t in price_data.tickers]
    print(f"[green]Using {len(available_tickers)} stocks from universe[/green]")
    
    # 2. Get quarterly rebalance dates
    rebalance_dates = _get_quarterly_dates(start_date, end_date)
    print(f"[blue]Rebalancing on {len(rebalance_dates)} dates[/blue]")
    
    # 3. Initialize tracking
    daily_values = [100.0]  # Start at 100
    portfolio_dates = [pd.Timestamp(start_date)]
    portfolio_weights = {}
    turnover_history = []
    transaction_costs_history = []
    
    # Get returns data
    returns_df = price_data.get_returns()
    asset_returns = returns_df[available_tickers]
    
    # Check for benchmark
    benchmark_ticker = 'SPY'
    if benchmark_ticker in returns_df.columns:
        benchmark_returns = returns_df[benchmark_ticker]
    else:
        # Use equal-weight as proxy
        benchmark_returns = asset_returns.mean(axis=1)
        print("[yellow]Using equal-weight proxy for benchmark[/yellow]")
    
    # Configuration
    config = OptimizationConfig(
        confidence_level=0.95,
        sparsity_bound=1.2,
        benchmark_ticker=benchmark_ticker,
        lookback_days=252,
        max_weight=0.05,  # 5% max weight
        min_weight=0.0
    )
    
    # 4. Run backtest
    current_weights = np.zeros(len(available_tickers))
    prev_rebal_date = pd.Timestamp(start_date)
    
    for i, rebal_date in enumerate(rebalance_dates):
        print(f"\r[blue]Processing rebalance {i+1}/{len(rebalance_dates)}[/blue]", end='')
        
        # Calculate performance since last rebalance
        if i > 0:
            period_dates = returns_df.loc[prev_rebal_date:rebal_date].index[1:]
            for date in period_dates:
                if date <= rebal_date:
                    day_return = np.dot(asset_returns.loc[date].values, current_weights)
                    daily_values.append(daily_values[-1] * (1 + day_return))
                    portfolio_dates.append(date)
        
        # Get historical data for optimization
        hist_end = rebal_date
        hist_start = hist_end - pd.Timedelta(days=config.lookback_days * 1.5)
        
        hist_mask = (returns_df.index >= hist_start) & (returns_df.index < hist_end)
        hist_asset_returns = asset_returns.loc[hist_mask].values
        hist_benchmark_returns = benchmark_returns.loc[hist_mask].values
        
        if len(hist_asset_returns) < 50:
            print(f"\n[yellow]Skipping {rebal_date}: insufficient history[/yellow]")
            continue
        
        # Optimize
        try:
            new_weights, info = solve_cleir(
                asset_returns=hist_asset_returns,
                benchmark_returns=hist_benchmark_returns,
                config=config,
                verbose=False
            )
            
            # Calculate turnover and costs
            if i > 0:
                turnover = calculate_turnover(current_weights, new_weights)
                transaction_cost = calculate_transaction_costs(turnover, transaction_cost_bps)
            else:
                # Initial investment
                turnover = 1.0
                transaction_cost = calculate_transaction_costs(turnover, transaction_cost_bps)
            
            turnover_history.append(turnover)
            transaction_costs_history.append(transaction_cost)
            
            # Apply transaction costs
            if len(daily_values) > 0:
                daily_values[-1] *= (1 - transaction_cost)
            
            # Store weights
            weight_dict = {ticker: weight for ticker, weight in zip(available_tickers, new_weights)}
            portfolio_weights[rebal_date] = weight_dict
            current_weights = new_weights.copy()
            
        except Exception as e:
            print(f"\n[red]Optimization failed on {rebal_date}: {e}[/red]")
            continue
        
        prev_rebal_date = rebal_date
    
    # Final period performance
    if rebal_date < pd.Timestamp(end_date):
        final_dates = returns_df.loc[rebal_date:end_date].index[1:]
        for date in final_dates:
            day_return = np.dot(asset_returns.loc[date].values, current_weights)
            daily_values.append(daily_values[-1] * (1 + day_return))
            portfolio_dates.append(date)
    
    print("\n[green]✓ Baseline CLEIR backtest complete[/green]")
    
    # 5. Create results dictionary
    daily_values_series = pd.Series(daily_values, index=portfolio_dates[:len(daily_values)])
    returns_series = daily_values_series.pct_change().dropna()
    
    results = {
        'daily_values': daily_values_series,
        'returns': returns_series,
        'portfolio_weights': portfolio_weights,
        'turnover_history': turnover_history,
        'transaction_costs_history': transaction_costs_history,
        'avg_turnover': np.mean(turnover_history) if turnover_history else 0.0,
        'total_transaction_costs': sum(transaction_costs_history) if transaction_costs_history else 0.0,
        'config': config,
        'universe_size': len(available_tickers),
        'rebalance_dates': rebalance_dates
    }
    
    # Calculate performance metrics
    results['total_return'] = (daily_values_series.iloc[-1] / daily_values_series.iloc[0]) - 1
    results['annual_return'] = (1 + results['total_return']) ** (252 / len(returns_series)) - 1
    results['volatility'] = returns_series.std() * np.sqrt(252)
    results['sharpe_ratio'] = results['annual_return'] / results['volatility'] if results['volatility'] > 0 else 0
    
    # Max drawdown
    cumulative = daily_values_series / daily_values_series.iloc[0]
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    results['max_drawdown'] = drawdown.min()
    
    return results


def _load_price_data(start_date: str, end_date: str) -> PriceData:
    """Load price data from cache."""
    data_path = 'data/processed/price_data.pkl'
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Price data not found at {data_path}. "
            "Please run data download first."
        )
    
    with open(data_path, 'rb') as f:
        data_dict = pickle.load(f)
    
    # Convert to DataFrame
    if isinstance(data_dict, dict) and 'prices' in data_dict:
        price_df = pd.DataFrame(
            data_dict['prices'],
            index=pd.to_datetime(data_dict['dates']),
            columns=data_dict['tickers']
        )
        
        # Create PriceData object
        mask = (price_df.index >= start_date) & (price_df.index <= end_date)
        filtered_prices = price_df.loc[mask]
        
        return PriceData(
            tickers=list(filtered_prices.columns),
            prices=filtered_prices,
            dates=filtered_prices.index
        )
    else:
        raise ValueError("Unexpected price data format")


def _get_quarterly_dates(start_date: str, end_date: str) -> List[pd.Timestamp]:
    """Get quarterly rebalance dates."""
    dates = pd.date_range(start=start_date, end=end_date, freq='Q')
    return dates.tolist() 