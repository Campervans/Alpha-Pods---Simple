import pandas as pd
import numpy as np
from datetime import timedelta
from typing import Dict, List, Tuple, Optional
import pickle
import os

from src.models.walk_forward import SimpleWalkForward
from src.optimization.cleir_solver import solve_cleir
from src.utils.schemas import OptimizationConfig

# Define top 60 universe (available large cap stocks from S&P 100)
# These are the 60 stocks available in our data
TOP_60_UNIVERSE = [
    'AAPL', 'ABBV', 'ACN', 'ADBE', 'ADI', 'ADP', 'AMGN', 'AMZN', 'APH', 'AVGO',
    'AXP', 'BAC', 'BKNG', 'BLK', 'BRK-B', 'CAT', 'CMCSA', 'COST', 'CRM', 'CVX',
    'DIS', 'EMR', 'GE', 'GILD', 'GOOGL', 'HD', 'HUM', 'JNJ', 'JPM', 'KLAC',
    'KO', 'LIN', 'LLY', 'LRCX', 'MA', 'MDLZ', 'META', 'MO', 'MSFT', 'NEE',
    'NKE', 'NVDA', 'ORCL', 'PEP', 'PFE', 'PG', 'RTX', 'SBUX', 'SLB', 'SPGI',
    'SYK', 'TMO', 'TSLA', 'TXN', 'UNH', 'USB', 'V', 'VZ', 'WMT', 'XOM'
]

class AlphaEnhancedBacktest:
    """ML-enhanced CLEIR backtest using an alpha overlay."""
    
    def __init__(self, optimization_config: Optional[OptimizationConfig] = None, top_k: int = 30):
        self.config = optimization_config or self._get_default_config()
        self.trainer = SimpleWalkForward()
        self.top_k = top_k
        self.ml_predictions = None
        self.benchmark_ticker = 'SPY'
        
    def _get_default_config(self) -> OptimizationConfig:
        """Default CLEIR configuration."""
        return OptimizationConfig(
            confidence_level=0.95,
            sparsity_bound=1.2,  # L1 norm constraint for sparsity
            benchmark_ticker='SPY',
            lookback_days=252,
            max_weight=0.10,
            min_weight=0.0
        )
        
    def run(self, start_date: str = '2020-01-01', end_date: str = '2024-12-31') -> Dict:
        """Run the ML-enhanced backtest."""
        print(f"\n🚀 Starting ML-Enhanced CLEIR Backtest")
        print(f"Period: {start_date} to {end_date}")
        print(f"Top K selection: {self.top_k} stocks")
        
        # 1. Load data with fixed training period (2014-2019)
        # Always load from 2014 for training, regardless of start_date
        train_start_date = '2014-01-01'
        universe_data, returns_data, benchmark_returns = self._load_data(
            train_start_date, end_date
        )
        
        # 2. Get quarterly rebalance dates
        rebalance_dates = self._get_rebalance_dates(start_date, end_date)
        print(f"Rebalance dates: {len(rebalance_dates)}")
        
        # 3. Train models and get alpha predictions
        print("\n📊 Training ML models...")
        self.ml_predictions = self.trainer.train_predict_for_all_assets(universe_data, rebalance_dates)
        
        # 4. Run backtest with ML-selected universe
        print("\n💼 Running portfolio optimization...")
        portfolio_weights = {}
        selected_universes = {}
        
        for i, date in enumerate(rebalance_dates):
            print(f"\nRebalancing {i+1}/{len(rebalance_dates)}: {date.date()}")
            
            if date not in self.ml_predictions or self.ml_predictions[date].empty:
                print(f"  ⚠️  No predictions available for {date.date()}")
                continue
                
            # Select top K stocks based on alpha scores
            alpha_scores = self.ml_predictions[date]
            selected_tickers = alpha_scores.nlargest(self.top_k).index.tolist()
            selected_universes[date] = selected_tickers
            
            print(f"  Selected {len(selected_tickers)} stocks based on alpha")
            
            # Get historical returns for optimization
            hist_end = date - timedelta(days=1)
            hist_start = hist_end - timedelta(days=self.config.lookback_days)
            
            # Filter returns for selected universe
            mask = (returns_data.index >= hist_start) & (returns_data.index <= hist_end)
            hist_returns = returns_data.loc[mask, selected_tickers]
            hist_benchmark = benchmark_returns.loc[mask]
            
            if len(hist_returns) < 50:
                print(f"  ⚠️  Insufficient data: only {len(hist_returns)} days")
                continue
            
            # Run CLEIR optimization on selected universe
            try:
                # Convert DataFrames to numpy arrays
                asset_returns_np = hist_returns.values
                benchmark_returns_np = hist_benchmark.values
                
                weights, info = solve_cleir(
                    asset_returns=asset_returns_np,
                    benchmark_returns=benchmark_returns_np,
                    config=self.config
                )
                
                # Store weights with full ticker mapping
                weight_dict = {ticker: weight for ticker, weight in zip(selected_tickers, weights)}
                portfolio_weights[date] = weight_dict
                
                # Show top holdings
                top_holdings = sorted(weight_dict.items(), key=lambda x: x[1], reverse=True)[:5]
                print(f"  Top holdings: {', '.join([f'{t}: {w:.1%}' for t, w in top_holdings])}")
                
            except Exception as e:
                print(f"  ❌ Optimization failed: {str(e)}")
                continue
        
        # 5. Calculate portfolio performance
        print("\n📈 Calculating portfolio performance...")
        results = self._calculate_performance(
            portfolio_weights, returns_data, rebalance_dates, start_date, end_date
        )
        
        # Add ML-specific information
        results['selected_universes'] = selected_universes
        results['ml_predictions'] = self.ml_predictions
        
        return results
    
    def _load_data(self, start_date: str, end_date: str) -> Tuple[Dict, pd.DataFrame, pd.Series]:
        """Load price data for universe and benchmark."""
        print("Loading market data...")
        
        # Load from processed data
        data_path = 'data/processed/price_data.pkl'
        if os.path.exists(data_path):
            with open(data_path, 'rb') as f:
                data_dict = pickle.load(f)
                
            # Convert dictionary to DataFrame
            if isinstance(data_dict, dict) and 'prices' in data_dict:
                # Create DataFrame from dictionary format
                price_df = pd.DataFrame(
                    data_dict['prices'],
                    index=pd.to_datetime(data_dict['dates']),
                    columns=data_dict['tickers']
                )
            else:
                # Assume it's already a DataFrame
                price_df = data_dict
                
            # Get universe tickers (top 60) that exist in the data
            available_tickers = price_df.columns.tolist()
            universe_tickers = [t for t in TOP_60_UNIVERSE if t in available_tickers]
            
            # Check if benchmark exists
            if self.benchmark_ticker not in available_tickers:
                print(f"Warning: Benchmark {self.benchmark_ticker} not found in data")
                # Use equal-weighted portfolio of universe as proxy
                self.benchmark_ticker = None
            
            # Filter date range
            mask = (price_df.index >= start_date) & (price_df.index <= end_date)
            
            if self.benchmark_ticker and self.benchmark_ticker not in universe_tickers:
                price_data = price_df.loc[mask, universe_tickers + [self.benchmark_ticker]]
            else:
                price_data = price_df.loc[mask, universe_tickers]
            
            # Create universe data dict for trainer
            universe_data = {}
            for ticker in universe_tickers:
                ticker_df = pd.DataFrame({
                    'close': price_data[ticker],
                    'volume': np.random.randint(1000000, 5000000, len(price_data))  # Dummy volume
                })
                universe_data[ticker] = ticker_df
            
            # Calculate returns
            returns_data = price_data.pct_change().dropna()
            asset_returns = returns_data[universe_tickers]
            
            # Handle benchmark returns
            if self.benchmark_ticker and self.benchmark_ticker in returns_data.columns:
                benchmark_returns = returns_data[self.benchmark_ticker]
            else:
                # Use equal-weighted portfolio as benchmark
                benchmark_returns = asset_returns.mean(axis=1)
                print("Using equal-weighted universe as benchmark")
            
            print(f"Loaded data for {len(universe_tickers)} stocks")
            return universe_data, asset_returns, benchmark_returns
            
        else:
            raise FileNotFoundError(f"Price data not found at {data_path}")
    
    def _get_rebalance_dates(self, start_date: str, end_date: str) -> List[pd.Timestamp]:
        """Get quarterly rebalance dates."""
        dates = pd.date_range(start=start_date, end=end_date, freq='Q')
        return dates.tolist()
    
    def _calculate_performance(self, portfolio_weights: Dict, returns_data: pd.DataFrame,
                              rebalance_dates: List, start_date: str, end_date: str) -> Dict:
        """Calculate portfolio performance metrics."""
        # Create daily portfolio values
        daily_values = pd.Series(index=returns_data.loc[start_date:end_date].index, dtype=float)
        daily_values.iloc[0] = 100.0
        
        # Track current weights
        current_weights = {}
        
        for i in range(1, len(daily_values)):
            date = daily_values.index[i]
            prev_date = daily_values.index[i-1]
            
            # Check if we need to rebalance
            if date in portfolio_weights:
                current_weights = portfolio_weights[date]
            
            # Calculate portfolio return
            if current_weights:
                day_returns = returns_data.loc[date]
                portfolio_return = sum(
                    weight * day_returns.get(ticker, 0) 
                    for ticker, weight in current_weights.items()
                )
                daily_values.iloc[i] = daily_values.iloc[i-1] * (1 + portfolio_return)
            else:
                daily_values.iloc[i] = daily_values.iloc[i-1]
        
        # Calculate metrics
        returns = daily_values.pct_change().dropna()
        
        results = {
            'daily_values': daily_values,
            'returns': returns,
            'total_return': (daily_values.iloc[-1] / daily_values.iloc[0]) - 1,
            'annual_return': ((daily_values.iloc[-1] / daily_values.iloc[0]) ** (252 / len(daily_values))) - 1,
            'volatility': returns.std() * np.sqrt(252),
            'sharpe_ratio': (returns.mean() * 252) / (returns.std() * np.sqrt(252)),
            'max_drawdown': (daily_values / daily_values.expanding().max() - 1).min(),
            'portfolio_weights': portfolio_weights
        }
        
        print(f"\n📊 Performance Summary:")
        print(f"Total Return: {results['total_return']:.2%}")
        print(f"Annual Return: {results['annual_return']:.2%}")
        print(f"Volatility: {results['volatility']:.2%}")
        print(f"Sharpe Ratio: {results['sharpe_ratio']:.3f}")
        print(f"Max Drawdown: {results['max_drawdown']:.2%}")
        
        return results