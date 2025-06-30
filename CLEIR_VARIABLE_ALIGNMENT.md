# CLEIR Implementation - Variable Alignment with Paper

## Overview
This document explains how our CLEIR (CVaR-LASSO Enhanced Index Replication) implementation aligns with the variables defined in the research paper.

## Variable Mapping

### Model Variables (Paper → Our Implementation)

1. **Y_t** (Benchmark index return at time t)
   - **Our Implementation**: `benchmark_returns` in `cleir_solver.py`
   - **Location**: Passed as parameter to `solve_cleir()` function
   - **Data Source**: Downloaded via `download_benchmark_data()` for SPY or other benchmark

2. **R_it** (Return of i-th candidate stock at time t)
   - **Our Implementation**: `asset_returns` in `cleir_solver.py`
   - **Location**: Passed as parameter to `solve_cleir()` function
   - **Data Source**: Downloaded via `download_universe()` for selected stocks

3. **w_i** (Weight of i-th stock in portfolio)
   - **Our Implementation**: `w` variable in `create_cleir_problem()`
   - **Type**: `cp.Variable(n_assets)`
   - **Constraints**: 
     - Paper allows negative weights (short selling)
     - Our implementation supports both long-only (`min_weight=0`) and long-short (`min_weight<0`)

4. **p** (Total number of candidate stocks)
   - **Our Implementation**: `n_assets` in the code
   - **Derived from**: `asset_returns.shape[1]`
   - **Configurable via**: `UniverseConfig.n_stocks`

5. **CVaR_α** (Conditional Value-at-Risk at confidence level α)
   - **Our Implementation**: Computed using `config.confidence_level` (α)
   - **Formula**: `zeta + 1/(n*(1-alpha)) * sum(z)`
   - **Location**: `cvar_objective` in `create_cleir_problem()`

6. **s** (LASSO penalty constant)
   - **Our Implementation**: `config.sparsity_bound`
   - **Default**: 1.5 (as specified in paper)
   - **Constraint**: `sum(|w_i|) <= s`

### Computational Variables (Paper → Our Implementation)

1. **ζ (zeta)** (VaR threshold)
   - **Our Implementation**: `zeta` variable
   - **Type**: `cp.Variable(1)`
   - **Purpose**: Represents the Value-at-Risk level

2. **z_t** (Slack variables for CVaR)
   - **Our Implementation**: `z` variable
   - **Type**: `cp.Variable(n_periods)`
   - **Constraint**: `z >= tracking_error - zeta` and `z >= 0`
   - **Purpose**: Captures tracking error exceeding VaR

3. **u_i** (Dummy variables for LASSO linearization)
   - **Our Implementation**: `u` variable
   - **Type**: `cp.Variable(n_assets)`
   - **Constraints**: `u >= w` and `u >= -w`
   - **Purpose**: Converts absolute value constraint |w_i| into linear constraints

## Key Implementation Details

### Tracking Error Definition
```python
# Paper: Y_t - sum(w_i * R_it)
# Our implementation:
tracking_error = benchmark_returns - portfolio_returns
```
We minimize CVaR of tracking error, which captures the risk of underperforming the benchmark.

### CVaR Formulation
```python
# Paper: CVaR_α = ζ + 1/(T*(1-α)) * sum(z_t)
# Our implementation:
cvar_objective = zeta + (1.0 / (n_periods * (1 - alpha))) * cp.sum(z)
```

### LASSO Constraint Linearization
```python
# Paper: sum(|w_i|) <= s
# Our implementation (linearized):
u >= w
u >= -w
cp.sum(u) <= config.sparsity_bound
```

### Complete Constraint Set
1. **CVaR constraints**: 
   - `z >= tracking_error - zeta`
   - `z >= 0`

2. **LASSO linearization**:
   - `u >= w`
   - `u >= -w`
   - `sum(u) <= s`

3. **Budget constraint**:
   - `sum(w) = 1`

4. **Weight bounds** (optional):
   - `w >= min_weight` (0 for long-only, negative for long-short)
   - `w <= max_weight` (if specified)

## Usage Example

```python
# Configure CLEIR with paper's parameters
optimization_config = OptimizationConfig(
    confidence_level=0.95,      # α = 0.95
    lookback_days=252,          # T = 252 trading days
    sparsity_bound=1.5,         # s = 1.5 as in paper
    min_weight=0.0,             # Long-only (or -1.0 for long-short)
    max_weight=1.0,             # No upper bound per asset
    benchmark_ticker="SPY"      # Y_t benchmark
)

# Solve CLEIR
weights, info = solve_cleir(
    asset_returns,      # R_it matrix
    benchmark_returns,  # Y_t vector
    optimization_config
)
```

## Differences from Paper

1. **Long-only constraint**: While the paper allows short selling (negative w_i), our default implementation uses long-only constraints for practical reasons. This can be changed by setting `min_weight < 0`.

2. **Solver choice**: We use CVXPY with various solver backends (ECOS, SCS, etc.) rather than a specific LP solver.

3. **Transaction costs**: Our backtest implementation includes transaction costs, which weren't explicitly modeled in the paper's optimization.

## Validation

The implementation correctly:
- Minimizes CVaR of tracking error relative to benchmark
- Enforces sparsity through LASSO constraint
- Maintains budget constraint (weights sum to 1)
- Supports both long-only and long-short portfolios
- Uses exact linearization of absolute value constraints 