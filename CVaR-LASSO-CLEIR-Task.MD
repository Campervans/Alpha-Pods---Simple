# CVaR-LASSO-CLEIR Migration Tasks

This task list converts the current **CVaR-only** optimizer into the full **CVaR-LASSO Enhanced Index Replication (CLEIR)** formulation described below.

> **CLEIR Linear Program**  
> Minimise CVaR of **tracking error** subject to:  
> • absolute-weight (L1) sparsity bound \(\sum_i |w_i| \le s\)  
> • budget \(\sum_i w_i = 1\).

## 0. Preparation
- [x] Create new Git branch `feat/cleir` for isolated development.
- [x] Confirm `cvxpy>=1.4` installed (supports LP & mixed-norms efficiently).

## 1. Configuration Schema
1.1 `src/utils/schemas.py`
- [x] Add `sparsity_bound: float` (alias *s*).  
- [x] Deprecate/replace `lasso_lambda`; CLEIR uses a **constraint**, not penalty.
- [x] Add `benchmark_ticker: str` **required**.
- [x] Validate `sparsity_bound > 0` (typically 1.0-2.0 for long-short portfolios).

## 2. Data Plumbing
2.1 Backtester (`src/backtesting/engine.py`)
- [x] Accept separate lists: `asset_tickers` vs `benchmark_ticker`.
- [x] Slice benchmark returns series and pass to optimizer each rebalance.

## 3. Optimizer Refactor
3.1 `src/optimization/cvar_solver.py`
- [x] Rename to `cleir_solver.py` (keep old wrapper for backward compat).
- [x] Function signature: `solve_cleir(asset_returns: np.ndarray, benchmark_returns: np.ndarray, config: OptimizationConfig)`.

3.2 Build Linear Program
- [x] Variables: `w (p)`, `zeta (1)`, `z (n)`, `u (p)`.
- [x] Objective: `zeta + (1/(n*(1-alpha))) * sum(z)`.
- [x] Constraints:
  - [x] `z >= benchmark_returns - asset_returns @ w - zeta`.
  - [x] `z >= 0`.
  - [x] `u >=  w` and `u >= -w`.
  - [x] `sum(u) <= sparsity_bound`.
  - [x] `sum(w) == 1`.
  - [x] Optional long-only: `w >= 0` if `min_weight==0` else use existing bounds.
- [x] Solver selection: default `ECOS_BB` (handles LP with abs). Fallback to `SCS`.

## 4. Integration
- [x] Replace calls to `solve_cvar` with `solve_cleir` when `benchmark_ticker` provided.
- [x] Ensure turnover & transaction-cost code unchanged.

## 5. Tests
5.1 Unit Tests
- [x] New test file `tests/optimization/test_cleir_solver.py`.
  - [x] Generate toy data (n=250 days, p=10 assets + benchmark).
  - [x] Assert solution size: `|w|_1 <= s + 1e-6`.
  - [x] Assert budget sum ≈ 1.
  - [x] CVaR(tracking_err) lower than equal-weight baseline.

5.2 Integration Test
- [ ] Backtest 2019-2023 quarterly; assert fewer than 80% of assets used each rebalance.

## 6. Documentation & Examples
- [x] Update `README.md` to include CLEIR methodology.
- [ ] Provide example notebook `examples/cleir_demo.ipynb` showing optimization on sample data.

## 7. Verification Checklist
- [ ] Unit & integration tests pass.
- [ ] Lint & type checks succeed (`ruff`, `mypy`).
- [ ] Run `scripts/run_baseline_backtest.py` (updated) → no errors.
- [ ] Inspect `results/performance_summary.csv` — confirm `cvar_95_tracking_error` ≈ target and outperformance similar to paper.

---
**End of task list** 