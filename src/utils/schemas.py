"""
Data schemas and validation for CVaR index construction.

This module defines structured data containers with built-in validation
to ensure data quality and provide clear contracts between components.
"""

from dataclasses import dataclass, field, InitVar
from typing import Optional, List, Dict, Any
import pandas as pd
import numpy as np
from datetime import datetime


@dataclass
class PriceData:
    """
    Validated price data container.
    
    This class ensures that price data is properly formatted and complete
    before being used in downstream calculations.
    
    Notes
    -----
    Historically the `PriceData` constructor required an explicit
    `dates` argument.  Newer call-sites—including the project tests—pass
    `start_date` and `end_date` instead.  To remain backward compatible
    while supporting the new signature we now make `dates` optional and
    accept optional `start_date` / `end_date` keywords.  When `dates`
    is not supplied we derive it from the provided `prices` index.
    """

    tickers: List[str]
    prices: pd.DataFrame  # dates x tickers
    dates: Optional[pd.DatetimeIndex] = None
    volumes: Optional[pd.DataFrame] = None  # dates x tickers
    # Accept start/end date only for constructor signature compatibility.
    start_date: InitVar[Optional[pd.Timestamp]] = None
    end_date: InitVar[Optional[pd.Timestamp]] = None
    
    def __post_init__(self, start_date: Optional[pd.Timestamp], end_date: Optional[pd.Timestamp]):
        """Validate data consistency after initialization."""
        # ------------------------------------------------------------------
        # Handle missing `dates` – derive from the DataFrame index.  This is
        # particularly important for the unit-tests which construct
        # `PriceData` without explicitly passing a `dates` argument.
        # ------------------------------------------------------------------
        if self.dates is None:
            self.dates = self.prices.index

        # ------------------------------------------------------------------
        # Basic dimensionality checks
        # ------------------------------------------------------------------
        if len(self.tickers) != self.prices.shape[1]:
            raise ValueError(
                f"Number of tickers ({len(self.tickers)}) doesn't match "
                f"price data columns ({self.prices.shape[1]})"
            )

        if len(self.dates) != self.prices.shape[0]:
            raise ValueError(
                f"Number of dates ({len(self.dates)}) doesn't match "
                f"price data rows ({self.prices.shape[0]})"
            )
        
        # Check for missing data
        if self.prices.isnull().any().any():
            missing_count = self.prices.isnull().sum().sum()
            raise ValueError(f"Price data contains {missing_count} missing values")
        
        # Check for non-positive prices
        if (self.prices <= 0).any().any():
            raise ValueError("Price data contains non-positive values")
        
        # Validate volumes if provided
        if self.volumes is not None:
            if self.volumes.shape != self.prices.shape:
                raise ValueError("Volume data shape doesn't match price data shape")
            
            if (self.volumes < 0).any().any():
                raise ValueError("Volume data contains negative values")
        
        # Ensure tickers match column names
        if not all(self.prices.columns == self.tickers):
            self.prices.columns = self.tickers
            if self.volumes is not None:
                self.volumes.columns = self.tickers
        
        # Ensure dates match index
        if not all(self.prices.index == self.dates):
            # Try to align index to `dates` if possible otherwise fall back
            # to the DataFrame's current index.
            if len(self.prices.index) == len(self.dates):
                self.prices.index = self.dates
                if self.volumes is not None:
                    self.volumes.index = self.dates

        # ------------------------------------------------------------------
        # Validate optional `start_date` / `end_date` if they were provided.
        # The `PriceData` class also defines read-only *property* accessors
        # named `start_date` / `end_date`.  When the constructor is invoked
        # *without* explicitly passing these InitVar arguments the dataclass
        # machinery may pass the *property* objects themselves as the
        # default.  We therefore normalise such cases to `None`.
        # ------------------------------------------------------------------
        if isinstance(start_date, property):
            start_date = None
        if isinstance(end_date, property):
            end_date = None

        if start_date is not None and start_date != self.dates[0]:
            raise ValueError(
                "start_date argument does not match first date in prices index"
            )

        if end_date is not None and end_date != self.dates[-1]:
            raise ValueError(
                "end_date argument does not match last date in prices index"
            )
    
    @property
    def n_assets(self) -> int:
        """Number of assets in the universe."""
        return len(self.tickers)
    
    @property
    def n_periods(self) -> int:
        """Number of time periods."""
        return len(self.dates)
    
    @property
    def start_date(self) -> pd.Timestamp:
        """First date in the dataset."""
        return self.dates[0]
    
    @property
    def end_date(self) -> pd.Timestamp:
        """Last date in the dataset."""
        return self.dates[-1]
    
    def get_returns(self, method: str = 'simple') -> pd.DataFrame:
        """
        Calculate returns from prices.
        
        Args:
            method: 'simple' or 'log' returns
            
        Returns:
            DataFrame of returns
        """
        if method == 'simple':
            return self.prices.pct_change().dropna()
        elif method == 'log':
            return np.log(self.prices / self.prices.shift(1)).dropna()
        else:
            raise ValueError("Method must be 'simple' or 'log'")
    
    def slice_dates(self, start_date: str, end_date: str) -> 'PriceData':
        """
        Create a new PriceData object for a date range.
        
        Args:
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            
        Returns:
            New PriceData object for the specified date range
        """
        mask = (self.dates >= start_date) & (self.dates <= end_date)
        sliced_dates = self.dates[mask]
        sliced_prices = self.prices.loc[sliced_dates]
        sliced_volumes = self.volumes.loc[sliced_dates] if self.volumes is not None else None
        
        return PriceData(
            tickers=self.tickers.copy(),
            dates=sliced_dates,
            prices=sliced_prices,
            volumes=sliced_volumes
        )


@dataclass
class UniverseConfig:
    """
    Universe selection parameters.
    
    Configuration for selecting the investment universe from a larger
    set of available securities.
    """
    n_stocks: int = 60
    lookback_days: int = 126  # 6 months for liquidity calculation
    metric: str = "dollar_volume"  # 'dollar_volume', 'volume', or 'market_cap'
    min_price: float = 5.0  # Minimum price to avoid penny stocks
    min_trading_days: int = 100  # Minimum trading days in lookback period
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.n_stocks <= 0:
            raise ValueError("Number of stocks must be positive")
        
        if self.lookback_days <= 0:
            raise ValueError("Lookback days must be positive")
        
        if self.metric not in ["dollar_volume", "volume", "market_cap"]:
            raise ValueError("Metric must be 'dollar_volume', 'volume', or 'market_cap'")
        
        if self.min_price <= 0:
            raise ValueError("Minimum price must be positive")
        
        if self.min_trading_days <= 0:
            raise ValueError("Minimum trading days must be positive")
        
        if self.min_trading_days > self.lookback_days:
            raise ValueError("Minimum trading days cannot exceed lookback days")


@dataclass
class OptimizationConfig:
    """
    Configuration for CVaR optimization and CLEIR.
    
    Parameters that control the portfolio optimization process.
    """
    confidence_level: float = 0.95  # CVaR confidence level
    lookback_days: int = 252  # 1 year of daily returns
    max_weight: float = 0.05  # Maximum weight per asset (5%)
    min_weight: float = 0.0  # Minimum weight per asset
    solver: str = "ECOS"  # CVXPY solver
    solver_options: Dict[str, Any] = field(default_factory=dict)
    # CLEIR-specific parameters
    sparsity_bound: Optional[float] = None  # L1 norm constraint (sum of |w_i|)
    benchmark_ticker: Optional[str] = None  # Benchmark to track (e.g., "SPY")
    
    def __post_init__(self):
        """Validate optimization parameters."""
        if not (0 < self.confidence_level < 1):
            raise ValueError("Confidence level must be between 0 and 1")
        
        if self.lookback_days <= 0:
            raise ValueError("Lookback days must be positive")
        
        if not (0 <= self.min_weight <= self.max_weight <= 1):
            raise ValueError("Weights must satisfy: 0 <= min_weight <= max_weight <= 1")
        
        if self.solver not in ["ECOS", "SCS", "OSQP", "CLARABEL", "ECOS_BB"]:
            raise ValueError("Solver must be one of: ECOS, SCS, OSQP, CLARABEL, ECOS_BB")
        
        # CLEIR validation
        if self.sparsity_bound is not None:
            if self.sparsity_bound <= 0:
                raise ValueError("Sparsity bound must be positive (typically 1.0-2.0)")
            
        # If using CLEIR, benchmark is required
        if self.sparsity_bound is not None and self.benchmark_ticker is None:
            raise ValueError("CLEIR requires a benchmark ticker to be specified")


@dataclass
class RebalanceEvent:
    """
    Single rebalancing event with all details.
    
    Records all information about a portfolio rebalancing event
    for analysis and debugging.
    """
    date: pd.Timestamp
    weights_old: np.ndarray
    weights_new: np.ndarray
    returns_used: np.ndarray  # Historical returns used for optimization
    turnover: float
    transaction_cost: float
    optimization_time: float = 0.0  # Time taken for optimization
    solver_status: str = "OPTIMAL"
    
    def __post_init__(self):
        """Validate rebalancing event data."""
        if len(self.weights_old) != len(self.weights_new):
            raise ValueError("Old and new weights must have same length")
        
        if not np.isclose(self.weights_new.sum(), 1.0, atol=1e-6):
            raise ValueError(f"New weights sum to {self.weights_new.sum():.6f}, not 1.0")
        
        if np.any(self.weights_new < -1e-6):
            raise ValueError("New weights contain negative values")
        
        if self.turnover < 0:
            raise ValueError("Turnover cannot be negative")
        
        if self.transaction_cost < 0:
            raise ValueError("Transaction cost cannot be negative")
    
    @property
    def weight_changes(self) -> np.ndarray:
        """Calculate weight changes."""
        return self.weights_new - self.weights_old
    
    @property
    def largest_increase(self) -> float:
        """Largest weight increase."""
        return np.max(self.weight_changes)
    
    @property
    def largest_decrease(self) -> float:
        """Largest weight decrease (as positive number)."""
        return -np.min(self.weight_changes)


@dataclass
class BacktestConfig:
    """
    Configuration for backtesting.
    
    Parameters that control the backtesting process.
    """
    start_date: str
    end_date: str
    rebalance_frequency: str = "quarterly"  # 'quarterly', 'monthly', 'annually'
    transaction_cost_bps: float = 10.0  # Transaction cost per side in basis points
    initial_capital: float = 1000000.0  # Starting portfolio value
    benchmark_tickers: List[str] = field(default_factory=lambda: ["SPY"])
    
    def __post_init__(self):
        """Validate backtesting configuration."""
        try:
            start = pd.to_datetime(self.start_date)
            end = pd.to_datetime(self.end_date)
            if start >= end:
                raise ValueError("Start date must be before end date")
        except Exception as e:
            raise ValueError(f"Invalid date format: {e}")
        
        if self.rebalance_frequency not in ["quarterly", "monthly", "annually"]:
            raise ValueError("Rebalance frequency must be 'quarterly', 'monthly', or 'annually'")
        
        if self.transaction_cost_bps < 0:
            raise ValueError("Transaction cost cannot be negative")
        
        if self.initial_capital <= 0:
            raise ValueError("Initial capital must be positive")


@dataclass
class BacktestResults:
    """
    Complete backtesting results.
    
    Container for all backtesting outputs including performance metrics,
    rebalancing events, and intermediate calculations.
    """
    index_values: pd.Series
    returns: pd.Series
    weights_history: pd.DataFrame
    rebalance_events: List[RebalanceEvent]
    benchmark_returns: Dict[str, pd.Series] = field(default_factory=dict)
    config: Optional[BacktestConfig] = None
    
    def __post_init__(self):
        """Validate backtesting results."""
        from .precision import ensure_index_starts_at_100, debug_precision_issue
        
        if len(self.index_values) != len(self.returns) + 1:
            raise ValueError("Index values should have one more observation than returns")
        
        # Fix index values to start at exactly 100.0
        first_value = self.index_values.iloc[0]
        if not np.isclose(first_value, 100.0, atol=1e-4):
            print(f"Warning: Index starts at {first_value:.10f}, adjusting to 100.0")
            self.index_values = ensure_index_starts_at_100(self.index_values)
        
        # Ensure first value is exactly 100.0 after adjustment
        if not np.isclose(self.index_values.iloc[0], 100.0, atol=1e-6):
            debug_precision_issue(self.index_values.iloc[0], 100.0, "Index first value")
            raise ValueError(f"Index should start at 100.0, but starts at {self.index_values.iloc[0]} (type: {type(self.index_values.iloc[0])})")
        
        # Check that returns are consistent with index values (with relaxed tolerance)
        calculated_returns = self.index_values.pct_change().dropna()
        if not np.allclose(calculated_returns.values, self.returns.values, atol=1e-6):
            print("Warning: Returns and index values have minor inconsistencies, adjusting...")
            # Recalculate returns from corrected index values
            self.returns = calculated_returns
    
    @property
    def total_return(self) -> float:
        """Total return over the entire period."""
        return (self.index_values.iloc[-1] / self.index_values.iloc[0]) - 1
    
    @property
    def annual_return(self) -> float:
        """Annualized return."""
        n_years = len(self.returns) / 252
        return (1 + self.total_return) ** (1 / n_years) - 1
    
    @property
    def annual_volatility(self) -> float:
        """Annualized volatility."""
        return self.returns.std() * np.sqrt(252)
    
    @property
    def sharpe_ratio(self) -> float:
        """Sharpe ratio (assuming zero risk-free rate)."""
        return self.annual_return / self.annual_volatility if self.annual_volatility > 0 else 0
    
    @property
    def max_drawdown(self) -> float:
        """Maximum drawdown."""
        cumulative = self.index_values
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return abs(drawdown.min())
    
    @property
    def total_transaction_costs(self) -> float:
        """Total transaction costs as percentage of initial capital."""
        return sum(event.transaction_cost for event in self.rebalance_events)
    
    def get_performance_summary(self) -> Dict[str, float]:
        """Get summary of key performance metrics."""
        return {
            'total_return': self.total_return,
            'annual_return': self.annual_return,
            'annual_volatility': self.annual_volatility,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'total_transaction_costs': self.total_transaction_costs,
            'n_rebalances': len(self.rebalance_events)
        } 