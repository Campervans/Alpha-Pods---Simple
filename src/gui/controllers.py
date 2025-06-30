"""Business logic controllers for GUI operations."""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
from rich.console import Console

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.market_data.downloader import (
    create_sp100_list, 
    download_universe, 
    download_benchmark_data,
    save_ticker_data_to_pickle,
    load_ticker_data_from_pickle,
    create_sp100_since_2010,
    download_single_ticker
)
from src.market_data.universe import select_liquid_universe, apply_universe_filters, calculate_liquidity_scores
from src.backtesting.engine import CVaRIndexBacktest
from src.utils.schemas import UniverseConfig, OptimizationConfig, BacktestConfig, PriceData
from src.optimization.cvar_solver import solve_cvar
from src.optimization.cleir_solver import solve_cleir

console = Console()

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
            if filename.endswith('.pkl') and not filename.endswith('_meta.pkl'):
                ticker = filename.replace('.pkl', '')
                tickers.add(ticker)
        
        return sorted(list(tickers))
    
    def download_data(self, tickers: List[str], start: str, end: str) -> Dict[str, Any]:
        """Download price data for tickers."""
        try:
            # Import for detailed error tracking
            from ..market_data.downloader import download_multiple_tickers, align_data_by_dates
            
            # Download with detailed tracking
            print(f"Downloading {len(tickers)} tickers...")
            raw_data = download_multiple_tickers(tickers, start, end, max_workers=5, progress_bar=True)
            
            # Track success/failure
            successful_tickers = list(raw_data.keys())
            failed_tickers = [t for t in tickers if t not in successful_tickers]
            
            if not raw_data:
                return {'success': False, 'error': 'No data downloaded successfully'}
            
            # Align data
            price_df, volume_df = align_data_by_dates(raw_data, min_data_points=100)
            
            if price_df.empty:
                return {'success': False, 'error': 'No tickers have sufficient aligned data'}
            
            # Create result with detailed information
            result = {
                'success': True,
                'n_assets': len(price_df.columns),
                'start_date': price_df.index[0].strftime('%Y-%m-%d'),
                'end_date': price_df.index[-1].strftime('%Y-%m-%d'),
                'n_periods': len(price_df),
                'successful_tickers': successful_tickers,
                'failed_tickers': failed_tickers
            }
            
            # Add warning if some tickers failed
            if failed_tickers:
                result['warning'] = f"{len(failed_tickers)} tickers failed: {', '.join(failed_tickers[:5])}{'...' if len(failed_tickers) > 5 else ''}"
            
            return result
            
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
    
    def get_universe_list(self) -> List[str]:
        """Get the list of tickers in the universe."""
        return create_sp100_since_2010()


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
            tickers = create_sp100_since_2010()
            
            # Use universe selection if enabled
            if config.get('universe_selection_enabled', False):
                universe_config = UniverseConfig(
                    n_stocks=config.get('universe_size', 60),
                    lookback_days=252,
                    min_trading_days=200,
                    min_price=5.0,
                    metric="dollar_volume"
                )
                
                # Get recent data for universe selection
                end_date = config.get('end_date', '2024-12-31')
                start_date = (pd.to_datetime(end_date) - pd.Timedelta(days=365)).strftime('%Y-%m-%d')
                
                tickers = select_liquid_universe(
                    tickers,
                    universe_config,
                    start_date,
                    end_date
                )
            
            # Download price data
            price_data = download_universe(
                tickers, 
                backtest_config.start_date, 
                backtest_config.end_date,
                min_data_points=252,
                use_cache=True,
                cache_dir="data/raw"
            )
            
            # Apply universe filtering directly on the downloaded data
            # instead of calling select_liquid_universe which downloads again
            
            # Apply filters to get valid tickers
            valid_tickers, filter_results = apply_universe_filters(price_data, universe_config)
            
            if len(valid_tickers) < universe_config.n_stocks:
                console.print(f"[orange]Warning: Only {len(valid_tickers)} tickers passed filters, "
                      f"requested {universe_config.n_stocks}[/orange]")
                selected_tickers = valid_tickers
            else:
                # Calculate liquidity scores for valid tickers
                valid_price_data = PriceData(
                    tickers=valid_tickers,
                    dates=price_data.dates,
                    prices=price_data.prices[valid_tickers],
                    volumes=price_data.volumes[valid_tickers] if price_data.volumes is not None else None
                )
                
                liquidity_scores = calculate_liquidity_scores(valid_price_data, universe_config)
                
                # Select top N most liquid tickers
                selected_tickers = liquidity_scores.nlargest(universe_config.n_stocks).index.tolist()
                
                # Clean output - just show selection completed
                console.print(f"[dim]Selected {len(selected_tickers)} most liquid stocks[/dim]")
            
            # Filter price_data to only include selected tickers
            liquid_universe = PriceData(
                tickers=selected_tickers,
                dates=price_data.dates,
                prices=price_data.prices[selected_tickers],
                volumes=price_data.volumes[selected_tickers] if price_data.volumes is not None else None
            )
            
            # Create and run backtest
            backtest = CVaRIndexBacktest(
                price_data=liquid_universe,
                optimization_config=optimization_config,
                show_optimization_progress=True  # Show progress in GUI
            )
            
            results = backtest.run_backtest(backtest_config)
            
            # Calculate metrics
            total_return = (results.index_values.iloc[-1] / 100.0) - 1.0
            annual_return = annualize_return(total_return, len(results.returns), 252)
            sharpe_ratio = calculate_sharpe_ratio(results.returns, 0.0, 252)
            max_dd = calculate_max_drawdown(results.returns)
            
            # Save results with proper precision
            from ..utils.precision import round_index_values
            results_df = pd.DataFrame({
                'Date': results.index_values.index,
                'Index_Value': round_index_values(results.index_values.values)
            })
            results_df.to_csv('results/cvar_index_gui.csv', index=False)
            
            return {
                'success': True,
                'annual_return': annual_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_dd,
                'final_value': results.index_values.iloc[-1],
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
            tickers = create_sp100_since_2010()
            
            # Use universe selection if enabled
            if config.get('universe_selection_enabled', False):
                universe_config = UniverseConfig(
                    n_stocks=config.get('universe_size', 60),
                    lookback_days=252,
                    min_trading_days=200,
                    min_price=5.0,
                    metric="dollar_volume"
                )
                
                # Get recent data for universe selection
                end_date = config.get('end_date', '2024-12-31')
                start_date = (pd.to_datetime(end_date) - pd.Timedelta(days=365)).strftime('%Y-%m-%d')
                
                tickers = select_liquid_universe(
                    tickers,
                    universe_config,
                    start_date,
                    end_date
                )
            
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
                
                # Add benchmark to price data if available
                if optimization_config.benchmark_ticker in benchmark_data:
                    benchmark_prices = benchmark_data[optimization_config.benchmark_ticker]
                    # Align benchmark with price_data dates
                    aligned_benchmark = benchmark_prices.reindex(price_data.dates).ffill()
                    price_data.prices[optimization_config.benchmark_ticker] = aligned_benchmark
                    if price_data.volumes is not None:
                        price_data.volumes[optimization_config.benchmark_ticker] = pd.Series(1e6, index=price_data.dates)
                    price_data.tickers = price_data.tickers + [optimization_config.benchmark_ticker]
            
            # Apply universe filtering directly on the downloaded data
            # Apply filters to get valid tickers (exclude benchmark from filtering)
            asset_tickers_only = [t for t in price_data.tickers if t != optimization_config.benchmark_ticker]
            asset_price_data = PriceData(
                tickers=asset_tickers_only,
                dates=price_data.dates,
                prices=price_data.prices[asset_tickers_only],
                volumes=price_data.volumes[asset_tickers_only] if price_data.volumes is not None else None
            )
            
            valid_tickers, filter_results = apply_universe_filters(asset_price_data, universe_config)
            
            if len(valid_tickers) < universe_config.n_stocks:
                console.print(f"[orange]Warning: Only {len(valid_tickers)} tickers passed filters, "
                      f"requested {universe_config.n_stocks}[/orange]")
                selected_tickers = valid_tickers
            else:
                # Calculate liquidity scores for valid tickers
                valid_price_data = PriceData(
                    tickers=valid_tickers,
                    dates=price_data.dates,
                    prices=price_data.prices[valid_tickers],
                    volumes=price_data.volumes[valid_tickers] if price_data.volumes is not None else None
                )
                
                liquidity_scores = calculate_liquidity_scores(valid_price_data, universe_config)
                
                # Select top N most liquid tickers
                selected_tickers = liquidity_scores.nlargest(universe_config.n_stocks).index.tolist()
                
                # Clean output - just show selection completed
                console.print(f"[dim]Selected {len(selected_tickers)} most liquid stocks[/dim]")
            
            # Create universe with selected assets + benchmark
            if optimization_config.benchmark_ticker and optimization_config.benchmark_ticker in price_data.tickers:
                all_tickers = selected_tickers + [optimization_config.benchmark_ticker]
                liquid_universe = PriceData(
                    tickers=all_tickers,
                    dates=price_data.dates,
                    prices=price_data.prices[all_tickers],
                    volumes=price_data.volumes[all_tickers] if price_data.volumes is not None else None
                )
                asset_tickers = selected_tickers  # Only the assets, not the benchmark
            else:
                liquid_universe = PriceData(
                    tickers=selected_tickers,
                    dates=price_data.dates,
                    prices=price_data.prices[selected_tickers],
                    volumes=price_data.volumes[selected_tickers] if price_data.volumes is not None else None
                )
                asset_tickers = selected_tickers
            
            # Create and run backtest
            backtest = CVaRIndexBacktest(
                price_data=liquid_universe,
                optimization_config=optimization_config,
                asset_tickers=asset_tickers,
                show_optimization_progress=True  # Show progress in GUI
            )
            
            results = backtest.run_backtest(backtest_config)
            
            # Calculate metrics
            total_return = (results.index_values.iloc[-1] / 100.0) - 1.0
            annual_return = annualize_return(total_return, len(results.returns), 252)
            sharpe_ratio = calculate_sharpe_ratio(results.returns, 0.0, 252)
            max_dd = calculate_max_drawdown(results.returns)
            
            # Save results with proper precision
            from ..utils.precision import round_index_values
            results_df = pd.DataFrame({
                'Date': results.index_values.index,
                'Index_Value': round_index_values(results.index_values.values)
            })
            results_df.to_csv('results/cleir_index_gui.csv', index=False)
            
            # Create visualization with SPY comparison
            from ..gui.visualization import plot_index_comparison
            comparison_stats = plot_index_comparison(
                'results/cleir_index_gui.csv',
                optimization_config.benchmark_ticker or 'SPY'
            )
            
            return {
                'success': True,
                'annual_return': annual_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_dd,
                'final_value': results.index_values.iloc[-1],
                'total_return': total_return,
                'comparison_stats': comparison_stats
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