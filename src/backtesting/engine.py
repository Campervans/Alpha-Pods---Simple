# backtesting engine for cvar index

import numpy as np
import pandas as pd
from typing import List, Dict, Optional
import time
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, TaskProgressColumn
from rich.console import Console

from src.utils.schemas import (
    PriceData,
    OptimizationConfig,
    BacktestConfig,
    BacktestResults,
    RebalanceEvent,
)
from src.optimization.cvar_solver import solve_cvar
from .rebalancer import (
    calculate_drift_adjusted_weights, 
    create_rebalance_event,
    get_rebalancing_dates
)
from src.utils.precision import round_index_values, round_returns, round_weights

console = Console()

class CVaRIndexBacktest:
    # runs the backtest
    
    def __init__(self, price_data: PriceData, optimization_config: OptimizationConfig, 
                 asset_tickers: Optional[List[str]] = None, show_optimization_progress: bool = False):
        self.price_data = price_data
        self.optimization_config = optimization_config
        self.returns = price_data.get_returns(method='simple')
        self.rebalance_events: List[RebalanceEvent] = []
        self.show_optimization_progress = show_optimization_progress
        
        # for CLEIR: separate assets from benchmark
        if asset_tickers is not None:
            self.asset_tickers = asset_tickers
            self.is_cleir = optimization_config.sparsity_bound is not None
        else:
            # legacy mode: all tickers are assets
            self.asset_tickers = price_data.tickers
            self.is_cleir = False
        
        console.print(f"[green]Backtester ready: {len(self.asset_tickers)} assets[/green]")
        if self.is_cleir and optimization_config.benchmark_ticker:
            console.print(f"[yellow]CLEIR mode: tracking {optimization_config.benchmark_ticker}[/yellow]")
        console.print(f"[blue]Period: {price_data.start_date.date()} to {price_data.end_date.date()}[/blue]")
    
    def run_backtest(self, config: BacktestConfig) -> BacktestResults:
        # main backtest loop
        console.print(f"\n[bold cyan]Running backtest: {config.start_date} to {config.end_date}[/bold cyan]")
        
        # filter data
        backtest_data = self._filter_data_to_period(config.start_date, config.end_date)
        backtest_returns = backtest_data.get_returns(method='simple')
        
        # get rebal dates
        rebalance_dates = get_rebalancing_dates(
            backtest_data.dates, 
            config.rebalance_frequency
        )
        
        console.print(f"[green]{len(rebalance_dates)} rebalances to do[/green]")
        
        # init portfolio
        index_values = [100.0]  # always start at 100
        n_assets = len(self.asset_tickers)
        current_weights = np.ones(n_assets) / n_assets  # equal weight start
        weights_history = []
        
        portfolio_dates = [backtest_data.start_date]
        
        # main loop
        prev_rebal_date = None
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            backtest_task = progress.add_task(
                "[bold blue]Processing rebalances...", 
                total=len(rebalance_dates)
            )
            
            for i, rebal_date in enumerate(rebalance_dates):
                progress.update(
                    backtest_task, 
                    description=f"[bold blue]Rebalancing {i+1}/{len(rebalance_dates)}: {rebal_date.date()}"
                )
                
                # calc perf since last rebal
                if prev_rebal_date is not None:
                    asset_returns = backtest_returns[self.asset_tickers]
                    # get the last known index value to ensure continuity
                    last_known_value = index_values[-1] if index_values else 100.0
                    period_performance = self._calculate_period_performance(
                        current_weights, asset_returns, prev_rebal_date, rebal_date,
                        last_index_value=last_known_value
                    )
                    
                    # update index vals, skip first val (previous period's close)
                    period_dates, period_values = period_performance
                    if len(period_values) > 1:
                        index_values.extend(period_values[1:])
                        portfolio_dates.extend(period_dates[1:])
                
                # drift adjusted weights (important for accurate turnover!)
                if prev_rebal_date is not None:
                    asset_returns = backtest_returns[self.asset_tickers]
                    old_weights = calculate_drift_adjusted_weights(
                        current_weights, asset_returns, prev_rebal_date, rebal_date
                    )
                else:
                    old_weights = current_weights.copy()
                
                # optimize
                new_weights, optimization_info = self._optimize_weights(
                    backtest_returns, rebal_date
                )
                
                # rebal event
                hist_returns = self._get_optimization_returns(backtest_returns, rebal_date)
                rebalance_event = create_rebalance_event(
                    date=rebal_date,
                    weights_old=old_weights,
                    weights_new=new_weights,
                    returns_used=hist_returns,
                    cost_per_side_bps=config.transaction_cost_bps,
                    optimization_time=optimization_info.get('solve_time', 0),
                    solver_status=optimization_info.get('status', 'UNKNOWN')
                )
                
                self.rebalance_events.append(rebalance_event)
                
                # apply costs to portfolio value at rebal date
                # first rebal still incurs costs to setup weights
                if len(index_values) > 0:
                    index_values[-1] *= (1 - rebalance_event.transaction_cost)
                
                # update weights
                current_weights = new_weights.copy()
                
                # record weights
                weights_record = {'date': rebal_date}
                for j, ticker in enumerate(self.asset_tickers):
                    weights_record[ticker] = new_weights[j]
                weights_history.append(weights_record)
                
                prev_rebal_date = rebal_date
                progress.update(backtest_task, advance=1)
        
        # final period
        if rebalance_dates[-1] < backtest_data.end_date:
            asset_returns = backtest_returns[self.asset_tickers]
            final_performance = self._calculate_period_performance(
                current_weights, asset_returns, rebalance_dates[-1], backtest_data.end_date
            )
            period_dates, period_values = final_performance
            index_values.extend(period_values[1:])
            portfolio_dates.extend(period_dates[1:])
        
        # create results w proper precision
        
        index_series = pd.Series(index_values, index=portfolio_dates)
        index_series = round_index_values(index_series)
        
        # make sure index starts at exactly 100
        if len(index_series) > 0:
            index_series.iloc[0] = 100.0
        
        returns_series = index_series.pct_change().dropna()
        returns_series = round_returns(returns_series)
        
        weights_df = pd.DataFrame(weights_history)
        weights_df.set_index('date', inplace=True)
        
        # round all weight columns
        for col in weights_df.columns:
            if col != 'date':
                weights_df[col] = round_weights(weights_df[col])
        
        results = BacktestResults(
            index_values=index_series,
            returns=returns_series,
            weights_history=weights_df,
            rebalance_events=self.rebalance_events,
            config=config
        )
        
        console.print(f"\n[bold green]Backtest complete! 🎯[/bold green]")
        console.print(f"[cyan]Return: {results.total_return:.2%}[/cyan]")
        console.print(f"[cyan]Annual: {results.annual_return:.2%}[/cyan]")
        console.print(f"[cyan]Vol: {results.annual_volatility:.2%}[/cyan]")
        console.print(f"[cyan]Sharpe: {results.sharpe_ratio:.3f}[/cyan]")
        console.print(f"[cyan]Max DD: {results.max_drawdown:.2%}[/cyan]")
        console.print(f"[cyan]Costs: {results.total_transaction_costs:.2%}[/cyan]")
        
        return results
    
    def _filter_data_to_period(self, start_date: str, end_date: str) -> PriceData:
        # filter to backtest period
        return self.price_data.slice_dates(start_date, end_date)
    
    def _optimize_weights(self, returns: pd.DataFrame, 
                         rebal_date: pd.Timestamp) -> tuple[np.ndarray, Dict]:
        # optimize weights for rebal date
        asset_returns = returns[self.asset_tickers]
        hist_asset_returns = self._get_optimization_returns(asset_returns, rebal_date)
        
        n_assets = len(self.asset_tickers)
        
        if len(hist_asset_returns) < 50:
            console.print(f"[orange]⚠️ Only {len(hist_asset_returns)} obs - using equal weights[/orange]")
            equal_weights = np.ones(n_assets) / n_assets
            return equal_weights, {'status': 'INSUFFICIENT_DATA'}
        
        # get benchmark returns if in CLEIR mode
        hist_benchmark_returns = None
        if self.is_cleir and self.optimization_config.benchmark_ticker:
            if self.optimization_config.benchmark_ticker in returns.columns:
                benchmark_returns = returns[self.optimization_config.benchmark_ticker]
                hist_benchmark_returns = self._get_optimization_returns(
                    benchmark_returns.to_frame(), rebal_date
                ).flatten()
            else:
                console.print(f"[orange]Warning: Benchmark {self.optimization_config.benchmark_ticker} not found[/orange]")
        
        # run optimization
        try:
            start_time = time.time()
            
            if self.is_cleir and hist_benchmark_returns is not None:
                # import CLEIR solver
                from ..optimization.cleir_solver import solve_cleir
                optimal_weights, solver_info = solve_cleir(
                    hist_asset_returns, 
                    hist_benchmark_returns,
                    self.optimization_config,
                    verbose=self.show_optimization_progress
                )
            else:
                # standard CVaR
                optimal_weights, solver_info = solve_cvar(
                    hist_asset_returns, 
                    self.optimization_config,
                    verbose=self.show_optimization_progress
                )
            
            solve_time = time.time() - start_time
            solver_info['solve_time'] = solve_time
            
            return optimal_weights, solver_info
            
        except Exception as e:
            console.print(f"[red]Optimization failed: {e}[/red]")
            # fallback
            equal_weights = np.ones(n_assets) / n_assets
            return equal_weights, {'status': 'OPTIMIZATION_ERROR', 'error': str(e)}
    
    def _get_optimization_returns(self, returns: pd.DataFrame, 
                                 rebal_date: pd.Timestamp) -> np.ndarray:
        # get historical returns for optimization
        try:
            end_loc = returns.index.get_loc(rebal_date)
        except KeyError:
            # date not found - use closest before
            end_loc = returns.index.searchsorted(rebal_date) - 1
            end_loc = max(0, end_loc)
        
        lookback_days = self.optimization_config.lookback_days
        start_loc = max(0, end_loc - lookback_days)
        
        hist_returns = returns.iloc[start_loc:end_loc].values
        
        return hist_returns
    
    def _calculate_period_performance(self, weights: np.ndarray,
                                     returns: pd.DataFrame,
                                     start_date: pd.Timestamp,
                                     end_date: pd.Timestamp,
                                     last_index_value: float = 100.0) -> tuple[List, List]:
        # calculate portfolio performance over a period
        try:
            period_returns = returns.loc[start_date:end_date]
        except KeyError:
            return [start_date, end_date], [last_index_value, last_index_value]
        
        if len(period_returns) == 0:
            return [start_date, end_date], [last_index_value, last_index_value]
        
        # daily portfolio returns
        portfolio_returns = np.dot(period_returns.values, weights)
        
        # cumulative
        cumulative_values = np.cumprod(1 + portfolio_returns)
        
        # prepend starting value, scaled by last known index value
        all_values = np.concatenate([[last_index_value], last_index_value * cumulative_values])
        all_dates = [start_date] + period_returns.index.tolist()
        
        return all_dates, all_values.tolist()
    
    def get_rebalancing_summary(self) -> pd.DataFrame:
        # get rebalancing summary as a dataframe
        if not self.rebalance_events:
            return pd.DataFrame()
        
        summary_data = []
        for event in self.rebalance_events:
            summary_data.append({
                'date': event.date,
                'turnover': event.turnover,
                'transaction_cost': event.transaction_cost,
                'optimization_time': event.optimization_time,
                'solver_status': event.solver_status,
                'largest_increase': event.largest_increase,
                'largest_decrease': event.largest_decrease
            })
        
        return pd.DataFrame(summary_data)
    
    def analyze_turnover_pattern(self) -> Dict:
        # analyze turnover stats
        if not self.rebalance_events:
            return {}
        
        turnovers = [event.turnover for event in self.rebalance_events]
        costs = [event.transaction_cost for event in self.rebalance_events]
        
        return {
            'mean_turnover': np.mean(turnovers),
            'median_turnover': np.median(turnovers),
            'std_turnover': np.std(turnovers),
            'max_turnover': np.max(turnovers),
            'min_turnover': np.min(turnovers),
            'total_transaction_costs': np.sum(costs),
            'mean_transaction_cost': np.mean(costs)
        }

# TODO: add more analytics here later...
