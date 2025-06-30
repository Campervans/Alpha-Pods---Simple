"""Business logic controllers for GUI operations."""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.market_data.downloader import (
    create_sp100_list, 
    download_universe, 
    download_benchmark_data,
    save_ticker_data_to_csv,
    load_ticker_data_from_csv
)
from src.market_data.universe import select_liquid_universe
from src.backtesting.engine import CVaRIndexBacktest
from src.utils.schemas import UniverseConfig, OptimizationConfig, BacktestConfig
from src.optimization.cvar_solver import solve_cvar
from src.optimization.cleir_solver import solve_cleir


class DataController:
    """Handle data management operations."""
    
    def __init__(self, cache_dir: str = "data/raw"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def get_cached_tickers(self) -> List[str]:
        """Get list of tickers in cache."""
        if not os.path.exists(self.cache_dir):
            return []
        
        tickers = set()
        for filename in os.listdir(self.cache_dir):
            if filename.endswith('.csv'):
                ticker = filename.split('_')[0]
                tickers.add(ticker)
        
        return sorted(list(tickers))
    
    def download_data(self, tickers: List[str], start: str, end: str) -> Dict[str, Any]:
        """Download price data for tickers."""
        try:
            price_data = download_universe(
                tickers, start, end, 
                min_data_points=100,
                use_cache=True,
                cache_dir=self.cache_dir
            )
            return {
                'success': True,
                'n_assets': price_data.n_assets,
                'start_date': price_data.start_date.strftime('%Y-%m-%d'),
                'end_date': price_data.end_date.strftime('%Y-%m-%d'),
                'n_periods': price_data.n_periods
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def clear_cache(self) -> bool:
        """Clear all cached data."""
        try:
            import shutil
            if os.path.exists(self.cache_dir):
                shutil.rmtree(self.cache_dir)
                os.makedirs(self.cache_dir)
            return True
        except Exception:
            return False
    
    def get_sp100_tickers(self) -> List[str]:
        """Get S&P 100 ticker list."""
        return create_sp100_list()


class OptimizationController:
    """Handle optimization operations."""
    
    def __init__(self):
        self.last_result = None
    
    def run_cvar_optimization(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run CVaR optimization with given config."""
        try:
            from src.utils.core import annualize_return, calculate_sharpe_ratio, calculate_max_drawdown
            
            # Create configs
            universe_config = UniverseConfig(
                n_stocks=config.get('n_stocks', 60),
                lookback_days=config.get('universe_lookback', 126),
                min_price=config.get('min_price', 5.0)
            )
            
            optimization_config = OptimizationConfig(
                confidence_level=config.get('confidence_level', 0.95),
                lookback_days=config.get('optimization_lookback', 252),
                max_weight=config.get('max_weight', 0.05),
                min_weight=0.0,
                solver="SCS"
            )
            
            backtest_config = BacktestConfig(
                start_date=config.get('start_date', '2010-01-01'),
                end_date=config.get('end_date', '2024-12-31'),
                rebalance_frequency=config.get('rebalance_freq', 'quarterly'),
                transaction_cost_bps=config.get('transaction_cost', 10.0),
                initial_capital=100.0
            )
            
            # Get S&P 100 tickers
            tickers = create_sp100_list()
            
            # Download price data
            price_data = download_universe(
                tickers, 
                backtest_config.start_date, 
                backtest_config.end_date,
                min_data_points=252,
                use_cache=True,
                cache_dir="data/raw"
            )
            
            # Select liquid universe
            liquid_universe = select_liquid_universe(
                price_data, 
                universe_config, 
                backtest_config.start_date
            )
            
            # Create and run backtest
            backtest = CVaRIndexBacktest(
                price_data=liquid_universe,
                optimization_config=optimization_config,
                backtest_config=backtest_config
            )
            
            results = backtest.run()
            
            # Calculate metrics
            total_return = (results['index_values'][-1] / 100.0) - 1.0
            annual_return = annualize_return(total_return, len(results['returns']), 252)
            sharpe_ratio = calculate_sharpe_ratio(results['returns'], 0.0, 252)
            max_dd = calculate_max_drawdown(results['returns'])
            
            # Save results
            results_df = pd.DataFrame({
                'Date': results['dates'],
                'Index_Value': results['index_values']
            })
            results_df.to_csv('results/cvar_index_gui.csv', index=False)
            
            return {
                'success': True,
                'annual_return': annual_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_dd,
                'final_value': results['index_values'][-1],
                'total_return': total_return
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def run_cleir_optimization(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run CLEIR optimization with given config."""
        try:
            from src.utils.core import annualize_return, calculate_sharpe_ratio, calculate_max_drawdown
            
            # Create configs
            universe_config = UniverseConfig(
                n_stocks=config.get('n_stocks', 60),
                lookback_days=config.get('universe_lookback', 126),
                min_price=config.get('min_price', 5.0)
            )
            
            optimization_config = OptimizationConfig(
                confidence_level=config.get('confidence_level', 0.95),
                lookback_days=config.get('optimization_lookback', 252),
                max_weight=config.get('max_weight', 0.05),
                min_weight=0.0,
                solver="CLARABEL",
                sparsity_bound=config.get('sparsity_bound', 1.2),
                benchmark_ticker=config.get('benchmark_ticker', 'SPY')
            )
            
            backtest_config = BacktestConfig(
                start_date=config.get('start_date', '2010-01-01'),
                end_date=config.get('end_date', '2024-12-31'),
                rebalance_frequency=config.get('rebalance_freq', 'quarterly'),
                transaction_cost_bps=config.get('transaction_cost', 10.0),
                initial_capital=100.0
            )
            
            # Get S&P 100 tickers
            tickers = create_sp100_list()
            
            # Download price data
            price_data = download_universe(
                tickers, 
                backtest_config.start_date, 
                backtest_config.end_date,
                min_data_points=252,
                use_cache=True,
                cache_dir="data/raw"
            )
            
            # Download benchmark data if needed
            if optimization_config.benchmark_ticker:
                benchmark_data = download_benchmark_data(
                    [optimization_config.benchmark_ticker],
                    backtest_config.start_date,
                    backtest_config.end_date
                )
                
                # Add benchmark to price data
                if optimization_config.benchmark_ticker in benchmark_data:
                    benchmark_prices = benchmark_data[optimization_config.benchmark_ticker]
                    price_data.prices[optimization_config.benchmark_ticker] = benchmark_prices.reindex(price_data.dates)
                    price_data.volumes[optimization_config.benchmark_ticker] = pd.Series(1e6, index=price_data.dates)
                    price_data.tickers = price_data.tickers + [optimization_config.benchmark_ticker]
            
            # Select liquid universe
            liquid_universe = select_liquid_universe(
                price_data, 
                universe_config, 
                backtest_config.start_date
            )
            
            # Create and run backtest
            backtest = CVaRIndexBacktest(
                price_data=liquid_universe,
                optimization_config=optimization_config,
                backtest_config=backtest_config
            )
            
            results = backtest.run()
            
            # Calculate metrics
            total_return = (results['index_values'][-1] / 100.0) - 1.0
            annual_return = annualize_return(total_return, len(results['returns']), 252)
            sharpe_ratio = calculate_sharpe_ratio(results['returns'], 0.0, 252)
            max_dd = calculate_max_drawdown(results['returns'])
            
            # Save results
            results_df = pd.DataFrame({
                'Date': results['dates'],
                'Index_Value': results['index_values']
            })
            results_df.to_csv('results/cleir_index_gui.csv', index=False)
            
            return {
                'success': True,
                'annual_return': annual_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_dd,
                'final_value': results['index_values'][-1],
                'total_return': total_return
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}


class ResultsController:
    """Handle results viewing and export."""
    
    def __init__(self, results_dir: str = "results"):
        self.results_dir = results_dir
    
    def get_available_results(self) -> List[str]:
        """Get list of available result files."""
        if not os.path.exists(self.results_dir):
            return []
        
        return [f for f in os.listdir(self.results_dir) 
                if f.endswith('.csv') or f.endswith('.png')]
    
    def load_performance_summary(self) -> Optional[pd.DataFrame]:
        """Load performance summary if exists."""
        path = os.path.join(self.results_dir, 'performance_summary.csv')
        if os.path.exists(path):
            return pd.read_csv(path)
        return None
    
    def generate_deliverables(self) -> Dict[str, bool]:
        """Generate all Task A deliverables."""
        status = {
            'daily_values': False,
            'metrics_table': False,
            'comparison_plot': False
        }
        
        # Check if files exist
        files = {
            'daily_values': 'daily_index_values.csv',
            'metrics_table': 'performance_summary.csv',
            'comparison_plot': 'index_performance_analysis.png'
        }
        
        for key, filename in files.items():
            path = os.path.join(self.results_dir, filename)
            status[key] = os.path.exists(path)
        
        return status 