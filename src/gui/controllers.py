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
            
            # Generate performance plot
            self._generate_performance_plot('cvar')
            
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
            
            # Generate performance plot
            self._generate_performance_plot('cleir')
            
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
    
    def _generate_performance_plot(self, index_type: str):
        """Generate performance comparison plot for the given index type."""
        try:
            if index_type == 'cvar':
                # For CVaR, use the simple plot from scripts
                from scripts.generate_performance_comparison_plots import load_index_data, create_performance_plot
                
                # Load data
                cvar_df, _ = load_index_data()
                
                if cvar_df is not None:
                    create_performance_plot(cvar_df, 'CVaR', 'results/cvar_index_performance_analysis.png')
                    console.print(f"[dim]✓ Generated cvar_index_performance_analysis.png[/dim]")
                else:
                    console.print(f"[yellow]Warning: No CVaR data available to plot[/yellow]")
            else:
                # For CLEIR, we need to create the full comparison plot with SPY and equal-weighted
                import pandas as pd
                import matplotlib.pyplot as plt
                import matplotlib
                matplotlib.use('Agg')
                from ..market_data.downloader import download_benchmark_data, create_sp100_since_2010, download_universe
                from ..market_data.universe import apply_universe_filters, calculate_liquidity_scores
                from ..utils.schemas import UniverseConfig, PriceData as PriceDataSchema
                from ..utils.core import annualize_return, calculate_sharpe_ratio, calculate_max_drawdown
                import numpy as np
                
                # Load CLEIR data
                cleir_df = pd.read_csv('results/cleir_index_gui.csv')
                cleir_df['Date'] = pd.to_datetime(cleir_df['Date'])
                # Remove duplicates by keeping last value for each date
                cleir_df = cleir_df.drop_duplicates(subset='Date', keep='last')
                cleir_df.set_index('Date', inplace=True)
                
                # Get date range
                start_date = cleir_df.index[0].strftime('%Y-%m-%d')
                end_date = cleir_df.index[-1].strftime('%Y-%m-%d')
                
                # Download SPY data
                spy_data = download_benchmark_data(['SPY'], start_date, end_date)
                spy_prices = spy_data['SPY']
                
                # Calculate equal-weighted index
                # Use same universe config as the optimization
                universe_config = UniverseConfig(
                    n_stocks=60,
                    lookback_days=252,
                    min_trading_days=200,
                    min_price=5.0,
                    metric="dollar_volume"
                )
                
                # Get S&P 100 tickers
                sp100_tickers = create_sp100_since_2010()
                
                # Download price data
                price_data = download_universe(
                    sp100_tickers, 
                    start_date, 
                    end_date,
                    min_data_points=252,
                    use_cache=True,
                    cache_dir="data/raw"
                )
                
                # Apply universe filters
                valid_tickers, _ = apply_universe_filters(price_data, universe_config)
                
                if len(valid_tickers) >= universe_config.n_stocks:
                    # Calculate liquidity scores
                    valid_price_data_obj = PriceDataSchema(
                        tickers=valid_tickers,
                        dates=price_data.dates,
                        prices=price_data.prices[valid_tickers],
                        volumes=price_data.volumes[valid_tickers] if price_data.volumes is not None else None
                    )
                    
                    liquidity_scores = calculate_liquidity_scores(valid_price_data_obj, universe_config)
                    
                    # Select top N most liquid tickers
                    selected_tickers = liquidity_scores.nlargest(universe_config.n_stocks).index.tolist()
                    
                    # Calculate equal-weighted returns
                    selected_prices = price_data.prices[selected_tickers]
                    stock_returns = selected_prices.pct_change()
                    equal_weight_portfolio_returns = stock_returns.mean(axis=1)
                    
                    # Build equal-weighted index
                    equal_weight_index = pd.Series(index=selected_prices.index, dtype=float)
                    equal_weight_index.iloc[0] = 100.0
                    
                    for i in range(1, len(equal_weight_index)):
                        equal_weight_index.iloc[i] = equal_weight_index.iloc[i-1] * (1 + equal_weight_portfolio_returns.iloc[i])
                
                # Create the plot with all three lines
                plt.figure(figsize=(12, 7))
                
                # Align all data to common dates
                common_dates = cleir_df.index.intersection(spy_prices.index)
                if 'equal_weight_index' in locals():
                    common_dates = common_dates.intersection(equal_weight_index.index)
                
                cleir_aligned = cleir_df.loc[common_dates, 'Index_Value']
                spy_aligned = spy_prices.loc[common_dates]
                
                # CLEIR data is already an index starting at 100, so don't normalize it
                # SPY needs to be normalized to match CLEIR's base of 100
                spy_normalized = (spy_aligned / spy_aligned.iloc[0]) * 100.0
                
                # Plot CLEIR (already normalized as an index)
                plt.plot(common_dates, cleir_aligned, label='CLEIR Index', linewidth=2, color='green')
                
                # Plot SPY
                plt.plot(common_dates, spy_normalized, label='S&P 500 (SPY)', linewidth=2, color='blue', alpha=0.7)
                
                # Plot equal-weighted if available
                if 'equal_weight_index' in locals():
                    equal_weight_aligned = equal_weight_index.loc[common_dates]
                    # Equal weight index is already normalized to base 100
                    plt.plot(common_dates, equal_weight_aligned, label='Equal-Weighted', linewidth=2, color='orange', alpha=0.7)
                
                # Formatting
                plt.title('CLEIR Index Performance Comparison', fontsize=16, fontweight='bold')
                plt.xlabel('Date', fontsize=12)
                plt.ylabel('Index Value (Base = 100)', fontsize=12)
                plt.legend(loc='upper left', fontsize=11)
                plt.grid(True, alpha=0.3)
                
                # Format x-axis
                plt.gcf().autofmt_xdate()
                
                # Add statistics box
                cleir_returns = cleir_aligned.pct_change().dropna()
                total_return = (cleir_aligned.iloc[-1] / cleir_aligned.iloc[0]) - 1
                annual_return = annualize_return(total_return, len(cleir_returns), 252)
                volatility = cleir_returns.std() * np.sqrt(252)
                sharpe = calculate_sharpe_ratio(cleir_returns, 0.0, 252)
                
                stats_text = 'CLEIR Statistics:\n'
                stats_text += f'Total Return: {total_return:.1%}\n'
                stats_text += f'Annual Return: {annual_return:.1%}\n'
                stats_text += f'Annual Volatility: {volatility:.1%}\n'
                stats_text += f'Sharpe Ratio: {sharpe:.2f}'
                
                plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
                         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                         fontsize=10)
                
                # Save the plot
                plt.tight_layout()
                plt.savefig('results/cleir_index_performance_analysis.png', dpi=150, bbox_inches='tight')
                plt.close()
                
                console.print(f"[dim]✓ Generated cleir_index_performance_analysis.png with SPY and Equal-Weight benchmarks[/dim]")
                
        except Exception as e:
            console.print(f"[yellow]Warning: Could not generate performance plot: {str(e)}[/yellow]")


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


class ComparisonController:
    """Controller for comparing ML-Enhanced CLEIR, Baseline CLEIR, and SPY."""
    
    def __init__(self):
        self.console = Console()
        
    def run_comparison(self, 
                      start_date: str = "2018-01-01",
                      end_date: str = "2023-12-31",
                      force_refresh: bool = False) -> Dict[str, Any]:
        """Run full comparison of all strategies.
        
        Args:
            start_date: Start date for comparison
            end_date: End date for comparison  
            force_refresh: Force recalculation even if cached
            
        Returns:
            Dictionary with all results
        """
        from ..utils.results_cache import load_results, save_results, load_spy_benchmark
        from ..utils.cleir_runner import run_baseline_cleir
        from ..analysis.metrics import summarise_results, calculate_relative_metrics
        from .visualization import plot_equity_curves, render_metrics_table
        
        self.console.print("\n[bold cyan]Strategy Comparison Dashboard[/bold cyan]")
        self.console.print(f"Period: {start_date} to {end_date}\n")
        
        results = {}
        
        # 1. Load ML-Enhanced CLEIR results
        self.console.print("[yellow]Loading ML-Enhanced CLEIR results...[/yellow]")
        ml_results = load_results("ml_enhanced_cleir")
        
        if ml_results is None:
            self.console.print("[red]No ML-Enhanced CLEIR results found. Run ML optimization first.[/red]")
            return {}
        
        results['ML-Enhanced CLEIR'] = ml_results
        
        # 2. Load or generate Baseline CLEIR
        self.console.print("[yellow]Loading Baseline CLEIR results...[/yellow]")
        baseline_key = f"baseline_cleir_{start_date}_{end_date}"
        
        if force_refresh:
            baseline_results = None
        else:
            baseline_results = load_results(baseline_key)
        
        if baseline_results is None:
            self.console.print("[yellow]Generating Baseline CLEIR results...[/yellow]")
            baseline_results = run_baseline_cleir(start_date, end_date)
            save_results(baseline_results, baseline_key)
        
        results['Baseline CLEIR'] = baseline_results
        
        # 3. Load SPY benchmark
        self.console.print("[yellow]Loading SPY benchmark data...[/yellow]")
        spy_results = load_spy_benchmark(start_date, end_date)
        results['SPY Benchmark'] = spy_results
        
        # 4. Prepare equity curves
        curves = {}
        for name, result in results.items():
            if 'index_values' in result:
                curves[name] = result['index_values']
        
        # 5. Plot equity curves
        plot_equity_curves(curves, title="Strategy Performance Comparison")
        
        # 6. Prepare metrics
        metrics = {}
        for name, result in results.items():
            if 'metrics' in result:
                metrics[name] = result['metrics']
        
        # 7. Display metrics table
        render_metrics_table(metrics, title="Performance Metrics Comparison")
        
        # 8. Calculate relative metrics
        self.console.print("\n[bold cyan]Relative Performance Analysis[/bold cyan]")
        
        # ML vs Baseline
        if 'ML-Enhanced CLEIR' in results and 'Baseline CLEIR' in results:
            ml_vs_baseline = calculate_relative_metrics(
                results['ML-Enhanced CLEIR'],
                results['Baseline CLEIR']
            )
            
            self.console.print("\n[green]ML-Enhanced vs Baseline CLEIR:[/green]")
            for metric, value in ml_vs_baseline.items():
                if 'ratio' in metric:
                    self.console.print(f"  {metric}: {value:.3f}")
                else:
                    self.console.print(f"  {metric}: {value:+.2%}")
        
        # ML vs SPY
        if 'ML-Enhanced CLEIR' in results and 'SPY Benchmark' in results:
            ml_vs_spy = calculate_relative_metrics(
                results['ML-Enhanced CLEIR'],
                results['SPY Benchmark']
            )
            
            self.console.print("\n[green]ML-Enhanced vs SPY:[/green]")
            for metric, value in ml_vs_spy.items():
                if 'ratio' in metric:
                    self.console.print(f"  {metric}: {value:.3f}")
                else:
                    self.console.print(f"  {metric}: {value:+.2%}")
        
        # 9. Summary statistics
        self.console.print("\n[bold cyan]Summary Statistics[/bold cyan]")
        
        for name, result in results.items():
            summary = summarise_results(result)
            self.console.print(f"\n[yellow]{name}:[/yellow]")
            self.console.print(f"  Period: {summary['start_date']} to {summary['end_date']}")
            self.console.print(f"  Trading Days: {summary['n_days']}")
            self.console.print(f"  Total Return: {summary['total_return']:.2%}")
            self.console.print(f"  Annual Return: {summary['annual_return']:.2%}")
            self.console.print(f"  Sharpe Ratio: {summary['sharpe_ratio']:.3f}")
            self.console.print(f"  Max Drawdown: {summary['max_drawdown']:.2%}")
        
        return results 