import pandas as pd
import numpy as np
from datetime import timedelta
from typing import Dict, List, Tuple, Optional
import pickle
import os

from src.models.walk_forward import SimpleWalkForward
from src.optimization.cleir_solver import solve_cleir
from src.utils.schemas import OptimizationConfig
from src.utils.core import calculate_turnover, calculate_transaction_costs
from src.market_data.universe import get_ml_universe  # Centralized universe
from src.market_data.downloader import download_universe, download_benchmark_data

class AlphaEnhancedBacktest:
    """ML-enhanced CLEIR backtest"""
    
    def __init__(self, optimization_config: Optional[OptimizationConfig] = None, 
                 top_k: int = 60,  # match Task A universe size
                 transaction_cost_bps: float = 10.0):
        """init ML-enhanced CLEIR backtest.
        
        Note: using all 60 stocks in universe to match Task A
        and get better diversification.
        """
        self.config = optimization_config or self._get_default_config()
        self.trainer = SimpleWalkForward()
        self.top_k = top_k
        self.ml_predictions = None
        self.benchmark_ticker = 'SPY'
        self.transaction_cost_bps = transaction_cost_bps
        
    def _get_default_config(self) -> OptimizationConfig:
        """Default CLEIR config."""
        return OptimizationConfig(
            confidence_level=0.95,
            sparsity_bound=1.2,  # L1 norm for sparsity
            benchmark_ticker='SPY',
            lookback_days=252,
            max_weight=0.05,  # 5% max per stock
            min_weight=0.0
        )
        
    def run(self, start_date: str = '2020-01-01', end_date: str = '2024-12-31') -> Dict:
        """run the ML-enhanced backtest."""
        # make train/test split super clear
        train_start = '2014-01-01'
        train_end = '2019-12-31'
        
        print(f"\n🚀 Starting ML-Enhanced CLEIR Backtest")
        print(f"📊 ML Training Period: {train_start} to {train_end}")
        print(f"📈 Out-of-Sample Test: {start_date} to {end_date}")
        print(f"Top K selection: {self.top_k} stocks")
        print(f"Transaction costs: {self.transaction_cost_bps} bps")
        
        # 1. load data w/ fixed training period (2014-2019)
        # always load from 2014 for training, no matter start_date
        universe_data, returns_data, benchmark_returns = self._load_data(
            train_start, end_date
        )
        
        # 2. get quarterly rebalance dates
        rebalance_dates = self._get_rebalance_dates(start_date, end_date)
        print(f"Rebalance dates: {len(rebalance_dates)}")
        
        # 3. train models and get alpha predictions
        print("\n📊 Training ML models...")
        self.ml_predictions = self.trainer.train_predict_for_all_assets(universe_data, rebalance_dates)
        
        # 4. run backtest w/ ML-selected universe
        print("\n💼 Running portfolio optimization...")
        portfolio_weights = {}
        selected_universes = {}
        turnover_history = []
        transaction_costs_history = []
        
        # init previous weights (start with empty portfolio)
        prev_weights = {}
        
        for i, date in enumerate(rebalance_dates):
            print(f"\nRebalancing {i+1}/{len(rebalance_dates)}: {date.date()}")
            
            if date not in self.ml_predictions or self.ml_predictions[date].empty:
                print(f"  ⚠️  No predictions for {date.date()}")
                continue
                
            # select top K stocks based on alpha scores
            alpha_scores = self.ml_predictions[date]
            selected_tickers = alpha_scores.nlargest(self.top_k).index.tolist()
            selected_universes[date] = selected_tickers
            
            print(f"  Selected {len(selected_tickers)} stocks based on alpha")
            
            # get historical returns for optimization
            hist_end = date - timedelta(days=1)
            hist_start = hist_end - timedelta(days=self.config.lookback_days)
            
            # filter returns for selected universe
            mask = (returns_data.index >= hist_start) & (returns_data.index <= hist_end)
            hist_returns = returns_data.loc[mask, selected_tickers]
            hist_benchmark = benchmark_returns.loc[mask]
            
            if len(hist_returns) < 50:
                print(f"  ⚠️  Not enough data: only {len(hist_returns)} days")
                continue
            
            # run CLEIR on selected universe
            try:
                # convert DataFrames to numpy arrays
                asset_returns_np = hist_returns.values
                benchmark_returns_np = hist_benchmark.values
                
                weights, info = solve_cleir(
                    asset_returns=asset_returns_np,
                    benchmark_returns=benchmark_returns_np,
                    config=self.config
                )
                
                # store weights with ticker mapping
                weight_dict = {ticker: weight for ticker, weight in zip(selected_tickers, weights)}
                
                # calc turnover and transaction costs
                if i > 0 and prev_weights:
                    # create weight vectors for all tickers (including zeros)
                    all_tickers = list(set(prev_weights.keys()) | set(weight_dict.keys()))
                    old_weights_vec = np.array([prev_weights.get(t, 0.0) for t in all_tickers])
                    new_weights_vec = np.array([weight_dict.get(t, 0.0) for t in all_tickers])
                    
                    # calc turnover
                    turnover = calculate_turnover(old_weights_vec, new_weights_vec)
                    transaction_cost = calculate_transaction_costs(turnover, self.transaction_cost_bps)
                    
                    turnover_history.append(turnover)
                    transaction_costs_history.append(transaction_cost)
                    
                    print(f"  Turnover: {turnover:.1%}, Transaction cost: {transaction_cost:.3%}")
                else:
                    # first rebalance, initial purchase
                    turnover = 1.0  # 100% turnover from cash
                    transaction_cost = calculate_transaction_costs(turnover, self.transaction_cost_bps)
                    turnover_history.append(turnover)
                    transaction_costs_history.append(transaction_cost)
                
                portfolio_weights[date] = weight_dict
                prev_weights = weight_dict.copy()
                
                # show top holdings
                top_holdings = sorted(weight_dict.items(), key=lambda x: x[1], reverse=True)[:5]
                print(f"  Top holdings: {', '.join([f'{t}: {w:.1%}' for t, w in top_holdings])}")
                
            except Exception as e:
                print(f"  ❌ Optimization failed: {str(e)}")
                continue
        
        # 5. calculate portfolio performance
        print("\n📈 Calculating portfolio performance...")
        results = self._calculate_performance(
            portfolio_weights, returns_data, rebalance_dates, start_date, end_date,
            transaction_costs_history
        )
        
        # add ML-specific info
        results['selected_universes'] = selected_universes
        results['ml_predictions'] = self.ml_predictions
        results['turnover_history'] = turnover_history
        results['transaction_costs_history'] = transaction_costs_history
        results['avg_turnover'] = np.mean(turnover_history) if turnover_history else 0.0
        results['total_transaction_costs'] = np.sum(transaction_costs_history) if transaction_costs_history else 0.0
        
        # add train/test metadata
        results['train_period'] = (train_start, train_end)
        results['test_period'] = (start_date, end_date)
        results['model_features'] = 8  # TODO: this is hardcoded, should be dynamic
        results['universe_size'] = self.top_k
        
        print(f"\n💰 Transaction Cost Summary:")
        print(f"Average turnover: {results['avg_turnover']:.1%}")
        print(f"Total transaction costs: {results['total_transaction_costs']:.2%}")
        
        return results
    
    def _load_data(self, start_date: str, end_date: str) -> Tuple[Dict, pd.DataFrame, pd.Series]:
        """Load price data."""
        print("Loading market data...")
        
        # get universe tickers
        universe_tickers = get_ml_universe()
        
        # download price data, use cache if available
        price_data = download_universe(
            universe_tickers,
            start_date,
            end_date,
            min_data_points=252,
            use_cache=True,
            cache_dir="data/raw"
        )
        
        # download benchmark data
        if self.benchmark_ticker:
            benchmark_data = download_benchmark_data(
                [self.benchmark_ticker],
                start_date,
                end_date
            )
            
            # add benchmark to price data
            if self.benchmark_ticker in benchmark_data:
                benchmark_prices = benchmark_data[self.benchmark_ticker]
                # align benchmark with price_data dates
                aligned_benchmark = benchmark_prices.reindex(price_data.dates).ffill()
                price_data.prices[self.benchmark_ticker] = aligned_benchmark
                if price_data.volumes is not None:
                    price_data.volumes[self.benchmark_ticker] = pd.Series(1e6, index=price_data.dates)
        
        # create universe data dict for trainer
        universe_data = {}
        for ticker in universe_tickers:
            if ticker in price_data.prices.columns:
                ticker_df = pd.DataFrame({
                    'close': price_data.prices[ticker],
                    'volume': price_data.volumes[ticker] if price_data.volumes is not None else pd.Series(1e6, index=price_data.dates)
                })
                universe_data[ticker] = ticker_df
        
        # calculate returns
        returns_data = price_data.prices.pct_change().dropna()
        asset_returns = returns_data[universe_tickers]
        
        # handle benchmark returns
        if self.benchmark_ticker and self.benchmark_ticker in returns_data.columns:
            benchmark_returns = returns_data[self.benchmark_ticker]
        else:
            # use equal-weighted portfolio as benchmark
            benchmark_returns = asset_returns.mean(axis=1)
            print("Using equal-weighted universe as benchmark")
        
        print(f"Loaded data for {len(universe_data)} stocks from {price_data.dates[0]} to {price_data.dates[-1]}")
        return universe_data, asset_returns, benchmark_returns
    
    def _get_rebalance_dates(self, start_date: str, end_date: str) -> List[pd.Timestamp]:
        """Get quarterly rebalance dates."""
        dates = pd.date_range(start=start_date, end=end_date, freq='QE')
        return dates.tolist()
    
    def _calculate_performance(self, portfolio_weights: Dict, returns_data: pd.DataFrame,
                              rebalance_dates: List, start_date: str, end_date: str,
                              transaction_costs_history: List) -> Dict:
        """Calculate portfolio performance."""
        # create daily portfolio values
        daily_values = pd.Series(index=returns_data.loc[start_date:end_date].index, dtype=float)
        daily_values.iloc[0] = 100.0
        
        # track current weights
        current_weights = {}
        transaction_cost_idx = 0
        
        for i in range(1, len(daily_values)):
            date = daily_values.index[i]
            prev_date = daily_values.index[i-1]
            
            # check if we need to rebalance
            if date in portfolio_weights:
                current_weights = portfolio_weights[date]
                
                # apply transaction cost on rebalance date
                if transaction_cost_idx < len(transaction_costs_history):
                    # apply cost to previous day's value
                    daily_values.iloc[i-1] *= (1 - transaction_costs_history[transaction_cost_idx])
                    transaction_cost_idx += 1
            
            # calc portfolio return
            if current_weights:
                day_returns = returns_data.loc[date]
                portfolio_return = sum(
                    weight * day_returns.get(ticker, 0) 
                    for ticker, weight in current_weights.items()
                )
                daily_values.iloc[i] = daily_values.iloc[i-1] * (1 + portfolio_return)
            else:
                daily_values.iloc[i] = daily_values.iloc[i-1]
        
        # calc metrics
        returns = daily_values.pct_change().dropna()
        
        results = {
            'daily_values': daily_values,
            'returns': returns,
            'total_return': (daily_values.iloc[-1] / daily_values.iloc[0]) - 1,
            'annual_return': ((daily_values.iloc[-1] / daily_values.iloc[0]) ** (252 / len(daily_values))) - 1,
            'volatility': returns.std() * np.sqrt(252),
            'sharpe_ratio': (returns.mean() * 252) / (returns.std() * np.sqrt(252)),
            'max_drawdown': (daily_values / daily_values.expanding().max() - 1).min(),
            'portfolio_weights': portfolio_weights,
            'transaction_costs_history': transaction_costs_history
        }
        
        print(f"\n📊 Performance Summary:")
        print(f"Total Return: {results['total_return']:.2%}")
        print(f"Annual Return: {results['annual_return']:.2%}")
        print(f"Volatility: {results['volatility']:.2%}")
        print(f"Sharpe Ratio: {results['sharpe_ratio']:.3f}")
        print(f"Max Drawdown: {results['max_drawdown']:.2%}")
        
        return results