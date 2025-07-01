# CVaR Index - Baseline Implementation with CLEIR

A comprehensive implementation of both standard CVaR optimization and CVaR-LASSO Enhanced Index Replication (CLEIR) for equity portfolio construction with quarterly rebalancing, transaction costs, and comprehensive backtesting framework.

## Project Overview

This project implements two portfolio optimization approaches:

1. **Standard CVaR Optimization**: Minimizes the 95% Conditional Value at Risk of portfolio returns
2. **CLEIR (CVaR-LASSO Enhanced Index Replication)**: Minimizes the CVaR of tracking error relative to a benchmark index while enforcing sparsity through an L1 norm constraint

Both approaches operate on 60 liquid S&P-100 stocks for the period 2010-2024 with quarterly rebalancing and 10 basis points per side transaction costs.

### Key Features

- **Dual Optimization Modes**: 
  - Standard CVaR: Minimizes 95% Conditional Value at Risk of portfolio returns
  - CLEIR: Minimizes CVaR of tracking error with L1 sparsity constraint
- **Quarterly Rebalancing**: Systematic rebalancing at quarter-end dates
- **Transaction Costs**: Realistic 10 bps/side transaction cost modeling
- **Comprehensive Backtesting**: Full simulation with drift adjustment and cost application
- **Multiple Benchmarks**: Comparison with equal-weight and market-cap benchmarks
- **Robust Data Handling**: Automated data download, cleaning, and validation
- **Performance Analytics**: Extensive risk and return metrics with reporting
- **Sparse Portfolios**: CLEIR mode creates concentrated portfolios with fewer holdings

## CLEIR Methodology

The CVaR-LASSO Enhanced Index Replication (CLEIR) approach solves the following optimization problem:

```
minimize    CVaR_α(Y_t - Σᵢ wᵢ Rᵢₜ)
subject to  Σᵢ |wᵢ| ≤ s           (L1 sparsity constraint)
            Σᵢ wᵢ = 1             (budget constraint)
            wᵢ ≥ 0                (long-only constraint, optional)
```

Where:
- `Y_t` is the benchmark return at time t
- `Rᵢₜ` is the return of asset i at time t
- `wᵢ` are the portfolio weights
- `α` is the confidence level (e.g., 0.95)
- `s` is the sparsity bound controlling portfolio concentration

This formulation creates portfolios that:
1. Track the benchmark closely in normal conditions
2. Minimize downside tracking error during market stress
3. Use fewer assets due to the L1 constraint, reducing transaction costs

## Architecture

The project follows a modular architecture with clear separation of concerns:

```
src/
├── utils/                  # Core utilities and data schemas
│   ├── core.py            # Foundational functions (returns, turnover, etc.)
│   └── schemas.py         # Data validation and structured containers
├── market_data/           # Data acquisition and universe selection  
│   ├── downloader.py      # Multi-threaded data downloading
│   └── universe.py        # Liquidity-based universe selection
├── optimization/          # Portfolio optimization components
│   ├── risk_models.py     # Risk metric calculations
│   ├── cvar_solver.py     # Standard CVaR optimization using CVXPY
│   └── cleir_solver.py    # CLEIR optimization (CVaR with L1 constraint)
└── backtesting/          # Backtesting engine and analytics
    ├── engine.py         # Main backtesting orchestration
    ├── rebalancer.py     # Rebalancing logic and transaction costs
    └── metrics.py        # Performance analytics and reporting
```

### Design Principles

1. **Single Responsibility**: Each function and class has one clear purpose
2. **Testability First**: Small, pure functions that are easy to test
3. **Clear Interfaces**: Type hints and dataclasses for explicit contracts
4. **Progressive Complexity**: Simple utilities → Complex orchestration
5. **Fail Fast**: Validation at data entry points
6. **Reproducibility**: Deterministic results with comprehensive logging

## Installation

### Prerequisites

- Python 3.11 or higher
- Conda (recommended) or pip

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Alpha-Pods---Simple
   ```

2. **Create conda environment**
   ```bash
   conda env create -f environment.yml
   conda activate cvar-index
   ```

3. **Verify installation**
   ```bash
   python -c "import cvxpy, yfinance, pandas; print('All dependencies installed')"
   ```

## Quick Start

### Run Complete Backtest

```bash
python scripts/run_baseline_backtest.py
```

This will:
1. Select 60 most liquid stocks from S&P 100
2. Download historical price data (2009-2024)
3. Run CVaR optimization with quarterly rebalancing
4. Compare against equal-weight and market benchmarks
5. Generate comprehensive performance reports
6. Save all results to `results/` directory

### Expected Output

```
CVaR INDEX BASELINE BACKTEST
============================================================
STEP 1: UNIVERSE SELECTION
Selected 60 stocks for investment universe
Top 10 selected stocks: ['AAPL', 'MSFT', 'GOOGL', ...]

STEP 2: DATA DOWNLOAD  
Successfully downloaded data for 60 assets
Date range: 2009-07-01 to 2024-12-31

STEP 4: CVAR INDEX BACKTEST
Rebalancing 1/59: 2010-03-31
...
CVaR Index Performance Summary:
annual_return            :   8.45%
annual_volatility        :  15.23%
sharpe_ratio             :   0.555
max_drawdown             :  12.34%
total_transaction_costs  :   0.89%
```

## Configuration

### Universe Selection

```python
universe_config = UniverseConfig(
    n_stocks=60,                    # Number of stocks to select
    lookback_days=126,              # Liquidity calculation period
    metric="dollar_volume",         # Selection metric
    min_price=5.0,                  # Minimum price filter
    min_trading_days=100            # Minimum data requirement
)
```

### CVaR Optimization

```python
optimization_config = OptimizationConfig(
    confidence_level=0.95,          # CVaR confidence level
    lookback_days=252,              # Optimization window
    max_weight=0.05,                # Maximum weight per asset (5%)
    min_weight=0.0,                 # Long-only constraint
    solver="ECOS"                   # CVXPY solver
)
```

### Backtesting

```python
backtest_config = BacktestConfig(
    start_date="2010-01-01",
    end_date="2024-12-31", 
    rebalance_frequency="quarterly",
    transaction_cost_bps=10.0,      # 10 bps per side
    initial_capital=1000000.0,      # $1M starting capital
    benchmark_tickers=["SPY", "IWV"]
)
```

## Usage Examples

### Custom Universe Analysis

```python
from src.market_data.universe import select_liquid_universe, create_sp100_list
from src.utils.schemas import UniverseConfig

# Create custom universe
tickers = create_sp100_list()
config = UniverseConfig(n_stocks=30, min_price=10.0)
universe = select_liquid_universe(tickers, config)
print(f"Selected {len(universe)} stocks")
```

### Run Optimization Only

```python
from src.optimization.cvar_solver import solve_cvar
from src.utils.schemas import OptimizationConfig
import numpy as np

# Sample returns data
returns = np.random.normal(0.001, 0.02, (252, 60))  # 1 year, 60 assets
config = OptimizationConfig(confidence_level=0.95)

weights, info = solve_cvar(returns, config)
print(f"Optimization status: {info['status']}")
print(f"Max weight: {weights.max():.1%}")
```

### Custom Performance Analysis

```python
from src.backtesting.metrics import calculate_metrics, create_performance_report

# Calculate metrics for any return series
metrics = calculate_metrics(returns_series)
print(f"Sharpe ratio: {metrics['sharpe_ratio']:.3f}")
print(f"Max drawdown: {metrics['max_drawdown']:.2%}")
print(f"CVaR 95%: {metrics['cvar_95']:.2%}")
```

## Output Files

After running the backtest, the following files are generated in `results/`:

- **`baseline_cvar_index.csv`**: Daily index values starting at 100
- **`performance_metrics.xlsx`**: Comprehensive performance comparison
  - Performance_Metrics: Key metrics vs benchmarks
  - Monthly_Returns: Monthly return breakdown by year
- **`rebalancing_events.csv`**: Details of each rebalancing event
- **`weights_history.csv`**: Portfolio weights at each rebalancing date
- **`performance_comparison.png`**: Visual performance charts

## Testing

### Run Unit Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_core.py -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html
```

### Integration Testing

```bash
# Test data pipeline
python tests/integration/test_data_pipeline.py

# Test end-to-end backtest
python tests/integration/test_full_backtest.py
```

## Key Performance Metrics

The system calculates comprehensive performance metrics:

### Return Metrics
- Total Return, Annualized Return
- Risk-adjusted returns (Sharpe, Sortino, Calmar ratios)
- Tracking Error, Information Ratio, Alpha, Beta

### Risk Metrics  
- Volatility (annualized)
- Maximum Drawdown
- Value at Risk (VaR) 95%, 99%
- Conditional Value at Risk (CVaR) 95%, 99%
- Skewness, Kurtosis

### Portfolio Metrics
- Turnover, Transaction Costs
- Hit Rate, Win/Loss Ratio
- Risk Contribution Analysis

## Troubleshooting

### Common Issues

1. **Solver Failures**
   ```
   Warning: All CVaR solvers failed. Returning equal weights.
   ```
   **Solution**: Increase solver tolerances or use fallback solver (SCS)

2. **Data Download Errors**
   ```
   Error downloading data: No data was downloaded successfully
   ```
   **Solution**: Check internet connection, try smaller date ranges

3. **Insufficient Data**
   ```
   Warning: Only 45 observations for optimization
   ```
   **Solution**: Increase lookback period or use longer data history

### Performance Optimization

1. **Faster Downloads**: Adjust `max_workers` in download functions
2. **Solver Speed**: Use ECOS for fastest solving, SCS for robustness
3. **Memory Usage**: Process data in chunks for very large universes

## ML-Enhanced CLEIR with Alpha Overlay

The project includes an ML-enhanced version of CLEIR that uses machine learning to predict alpha and select the most promising stocks before optimization.

### Running ML-Enhanced Backtest

```bash
python scripts/run_simple_ml_backtest.py
```

This will:
1. Train Ridge regression models using 7 technical features
2. Predict quarterly alphas for all stocks
3. Select top 30 stocks based on predicted alpha
4. Run CLEIR optimization on the selected universe
5. Generate comprehensive visualizations including SHAP analysis

### Interpreting ML Alphas with SHAP

The ML models are explained using SHAP (SHapley Additive exPlanations), providing insights into:

- **Global Feature Importance**: Which features drive predictions across all stocks
- **Feature Distributions**: How feature values affect predictions (beeswarm plot)
- **Feature Dependencies**: Non-linear relationships between features and predictions
- **Model Transparency**: Exact contribution of each feature to each prediction

#### SHAP Visualizations

The system generates `ml_shap_analysis.png` containing:

1. **Feature Importance Bar Chart**: Mean absolute SHAP values showing overall feature impact
2. **Beeswarm Plot**: Distribution of SHAP values for each feature across all predictions
3. **Dependence Plots**: Detailed view of top features' impact on predictions

#### ML Predictions Diagnostics

The `ml_predictions_analysis.png` provides comprehensive diagnostics:

1. **Predicted vs Realized Returns**: Scatter plot with regression line and R²
2. **Information Coefficient (IC)**: Rolling correlation between predictions and outcomes
3. **Prediction Distribution**: Inter-quartile range showing prediction spread over time
4. **Rank Stability**: Autocorrelation of prediction ranks across quarters

### ML Features

The model uses 7 technical features:

- **Momentum**: 1-month, 3-month, and 6-month returns
- **Volatility**: 1-month and 3-month rolling volatility
- **Volume**: Ratio of current to 21-day average volume
- **RSI**: 14-day Relative Strength Index

## Advanced Features

### Custom Risk Models

```python
from src.optimization.risk_models import estimate_covariance_matrix

# Use shrinkage estimator
cov_matrix = estimate_covariance_matrix(returns_df, method="shrinkage")

# Exponential weighting
cov_matrix = estimate_covariance_matrix(returns_df, method="exponential")
```

### Alternative Optimizations

```python
from src.optimization.cvar_solver import solve_minimum_variance

# Minimum variance portfolio
weights, info = solve_minimum_variance(returns, config)
```

### Custom Rebalancing Frequencies

```python
# Monthly rebalancing
config.rebalance_frequency = "monthly"

# Annual rebalancing  
config.rebalance_frequency = "annually"
```

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/new-feature`)
3. Add tests for new functionality
4. Ensure all tests pass (`python -m pytest`)
5. Submit pull request

### Code Style

- Follow PEP 8 guidelines
- Use type hints for function signatures
- Add docstrings with examples
- Write unit tests for new functions

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **CVXPY** for convex optimization
- **yfinance** for financial data
- **pandas** for data manipulation
- **numpy** for numerical computing

## References

1. Rockafellar, R.T. and Uryasev, S. (2002). Conditional value-at-risk for general loss distributions. Journal of Banking & Finance, 26(7), 1443-1471.

2. Pflug, G.C. (2000). Some remarks on the value-at-risk and the conditional value-at-risk. In Probabilistic constrained optimization (pp. 272-281).

3. Krokhmal, P., Palmquist, J., & Uryasev, S. (2002). Portfolio optimization with conditional value-at-risk objective and constraints. Journal of Risk, 4, 43-68.
