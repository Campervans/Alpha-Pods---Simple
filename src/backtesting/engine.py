# backtesting engine for cvar index

import numpy as np
import pandas as pd
from typing import List, Dict, Optional
import time

from ..utils.schemas import PriceData, OptimizationConfig, BacktestConfig, BacktestResults, RebalanceEvent
from ..optimization.cvar_solver import solve_cvar
from .rebalancer import (
    calculate_drift_adjusted_weights, 
    create_rebalance_event,
    get_rebalancing_dates,
    apply_transaction_costs_to_returns
)


class CVaRIndexBacktest:
    # runs the backtest simulation
    
    def __init__(self, price_data: PriceData, optimization_config: OptimizationConfig):
        self.price_data = price_data
        self.optimization_config = optimization_config
        self.returns = price_data.get_returns(method='simple')
        self.rebalance_events: List[RebalanceEvent] = []
        
        print(f"Backtester ready: {price_data.n_assets} assets")
        print(f"Period: {price_data.start_date.date()} to {price_data.end_date.date()}")
    
    def run_backtest(self, config: BacktestConfig) -> BacktestResults:
        # main backtest loop
        print(f"\nRunning backtest: {config.start_date} to {config.end_date}")
        
        # filter data
        backtest_data = self._filter_data_to_period(config.start_date, config.end_date)
        backtest_returns = backtest_data.get_returns(method='simple')
        
        # get rebal dates
        rebalance_dates = get_rebalancing_dates(
            backtest_data.dates, 
            config.rebalance_frequency
        )
        
        print(f"{len(rebalance_dates)} rebalances to do")
        
        # init portfolio
        index_values = [config.initial_capital]
        current_weights = np.ones(backtest_data.n_assets) / backtest_data.n_assets  # equal weight start
        weights_history = []
        
        portfolio_dates = [backtest_data.start_date]
        
        # main loop
        prev_rebal_date = None
        
        for i, rebal_date in enumerate(rebalance_dates):
            print(f"\nRebal {i+1}/{len(rebalance_dates)}: {rebal_date.date()}")
            
            # calc perf since last rebal
            if prev_rebal_date is not None:
                period_performance = self._calculate_period_performance(
                    current_weights, backtest_returns, prev_rebal_date, rebal_date
                )
                
                # update index vals
                period_dates, period_values = period_performance
                index_values.extend(period_values[1:])  # skip first
                portfolio_dates.extend(period_dates[1:])
            
            # drift adjusted weights (important for accurate turnover calcs!)
            if prev_rebal_date is not None:
                old_weights = calculate_drift_adjusted_weights(
                    current_weights, backtest_returns, prev_rebal_date, rebal_date
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
            
            # apply costs
            if len(index_values) > 1:
                current_return = (index_values[-1] / index_values[-2]) - 1
                net_return = apply_transaction_costs_to_returns(
                    current_return, rebalance_event.transaction_cost
                )
                index_values[-1] = index_values[-2] * (1 + net_return)
            
            # update weights
            current_weights = new_weights.copy()
            
            # record weights
            weights_record = {'date': rebal_date}
            for j, ticker in enumerate(backtest_data.tickers):
                weights_record[ticker] = new_weights[j]
            weights_history.append(weights_record)
            
            prev_rebal_date = rebal_date
        
        # final period
        if rebalance_dates[-1] < backtest_data.end_date:
            final_performance = self._calculate_period_performance(
                current_weights, backtest_returns, rebalance_dates[-1], backtest_data.end_date
            )
            period_dates, period_values = final_performance
            index_values.extend(period_values[1:])
            portfolio_dates.extend(period_dates[1:])
        
        # create results
        index_series = pd.Series(index_values, index=portfolio_dates)
        returns_series = index_series.pct_change().dropna()
        
        weights_df = pd.DataFrame(weights_history)
        weights_df.set_index('date', inplace=True)
        
        results = BacktestResults(
            index_values=index_series,
            returns=returns_series,
            weights_history=weights_df,
            rebalance_events=self.rebalance_events,
            config=config
        )
        
        print(f"\nBacktest done! 🎯")
        print(f"Return: {results.total_return:.2%}")
        print(f"Annual: {results.annual_return:.2%}")
        print(f"Vol: {results.annual_volatility:.2%}")
        print(f"Sharpe: {results.sharpe_ratio:.3f}")
        print(f"Max DD: {results.max_drawdown:.2%}")
        print(f"Costs: {results.total_transaction_costs:.2%}")
        
        return results
    
    def _filter_data_to_period(self, start_date: str, end_date: str) -> PriceData:
        # filter to backtest period
        return self.price_data.slice_dates(start_date, end_date)
    
    def _optimize_weights(self, returns: pd.DataFrame, 
                         rebal_date: pd.Timestamp) -> tuple[np.ndarray, Dict]:
        # optimize weights for rebal date
        hist_returns = self._get_optimization_returns(returns, rebal_date)
        
        if len(hist_returns) < 50:
            print(f"⚠️ Only {len(hist_returns)} obs - using equal weights")
            n_assets = len(returns.columns)
            equal_weights = np.ones(n_assets) / n_assets
            return equal_weights, {'status': 'INSUFFICIENT_DATA'}
        
        # run cvar opt
        try:
            start_time = time.time()
            optimal_weights, solver_info = solve_cvar(hist_returns, self.optimization_config)
            solve_time = time.time() - start_time
            solver_info['solve_time'] = solve_time
            
            return optimal_weights, solver_info
            
        except Exception as e:
            print(f"Opt failed: {e}")
            # fallback
            n_assets = len(returns.columns)
            equal_weights = np.ones(n_assets) / n_assets
            return equal_weights, {'status': 'OPTIMIZATION_ERROR', 'error': str(e)}
    
    def _get_optimization_returns(self, returns: pd.DataFrame, 
                                 rebal_date: pd.Timestamp) -> np.ndarray:
        # get historical returns for opt
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
                                     end_date: pd.Timestamp) -> tuple[List, List]:
        # calc portfolio perf over period
        try:
            period_returns = returns.loc[start_date:end_date]
        except KeyError:
            return [start_date, end_date], [100.0, 100.0]
        
        if len(period_returns) == 0:
            return [start_date, end_date], [100.0, 100.0]
        
        # daily port returns
        portfolio_returns = np.dot(period_returns.values, weights)
        
        # cumulative
        cumulative_values = np.cumprod(1 + portfolio_returns)
        
        # prepend starting val
        all_values = np.concatenate([[1.0], cumulative_values])
        all_dates = [start_date] + period_returns.index.tolist()
        
        return all_dates, all_values.tolist()
    
    def get_rebalancing_summary(self) -> pd.DataFrame:
        # get rebal summary df
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

# might add more analytics here later...
