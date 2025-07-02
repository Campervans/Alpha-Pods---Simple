"""
CVaR-LASSO Enhanced Index Replication (CLEIR) solver.

implements CLEIR optimization as a linear program:
- minimize CVaR of tracking error
- subject to L1 norm constraint on weights (sparsity)
- subject to budget constraint (weights sum to 1)
"""

import cvxpy as cp
import numpy as np
from typing import Tuple, Dict, Any, Optional
import time
import warnings
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.console import Console

from ..utils.schemas import OptimizationConfig

console = Console()


def create_cleir_problem(
    asset_returns: np.ndarray,
    benchmark_returns: np.ndarray,
    config: OptimizationConfig
) -> Tuple[cp.Problem, cp.Variable, cp.Variable, cp.Variable, cp.Variable]:
    """
    Create the CLEIR linear programming problem.
    
    Args:
        asset_returns: historical asset returns (n_periods x n_assets)
        benchmark_returns: historical benchmark returns (n_periods,)
        config: optimization config with sparsity_bound
        
    Returns:
        (problem, weights, zeta, z, u)
    """
    n_periods, n_assets = asset_returns.shape
    
    if len(benchmark_returns) != n_periods:
        raise ValueError(
            f"Benchmark returns length ({len(benchmark_returns)}) must match "
            f"number of periods ({n_periods})"
        )
    
    # decision variables
    w = cp.Variable(n_assets, name="weights")  # portfolio weights
    zeta = cp.Variable(1, name="zeta")  # VaR threshold
    z = cp.Variable(n_periods, name="z")  # auxiliary variables for CVaR
    u = cp.Variable(n_assets, name="u")  # auxiliary variables for L1 norm
    
    # tracking error: portfolio return - benchmark return
    # we minimize downside risk, so we look at negative tracking error
    portfolio_returns = asset_returns @ w
    # use negative tracking error so CVaR captures underperformance
    tracking_error = benchmark_returns - portfolio_returns
    
    # CVaR objective
    alpha = config.confidence_level
    cvar_objective = zeta + (1.0 / (n_periods * (1 - alpha))) * cp.sum(z)
    
    # constraints
    constraints = [
        # CVaR constraints
        z >= tracking_error - zeta,
        z >= 0,
        
        # L1 norm linearization: u_i >= |w_i|
        u >= w,
        u >= -w,
        
        # sparsity constraint: sum(|w_i|) <= s
        cp.sum(u) <= config.sparsity_bound,
        
        # budget constraint: sum(w_i) = 1
        cp.sum(w) == 1.0,
    ]
    
    # add weight bounds if specified
    # always enforce non-negativity for long-only
    constraints.append(w >= config.min_weight)
    if config.max_weight < 1:
        constraints.append(w <= config.max_weight)
    
    # create optimization problem
    objective = cp.Minimize(cvar_objective)
    problem = cp.Problem(objective, constraints)
    
    return problem, w, zeta, z, u


def solve_cleir(
    asset_returns: np.ndarray,
    benchmark_returns: np.ndarray,
    config: OptimizationConfig,
    verbose: bool = False
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Solve the CLEIR optimization problem.
    
    Args:
        asset_returns: historical asset returns (n_periods x n_assets)
        benchmark_returns: historical benchmark returns (n_periods,)
        config: optimization config
        verbose: whether to print solver output
        
    Returns:
        (optimal_weights, solver_info)
    """
    if asset_returns.shape[0] < 10:
        raise ValueError("Need at least 10 observations")
    
    if asset_returns.shape[1] == 0:
        raise ValueError("No assets to optimize")
    
    if config.sparsity_bound is None:
        raise ValueError("CLEIR requires sparsity_bound")
    
    # create the optimization problem
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
        disable=not verbose
    ) as progress:
        setup_task = progress.add_task("[cyan]Setting up CLEIR problem...", total=1)
        problem, w_var, zeta_var, z_var, u_var = create_cleir_problem(
            asset_returns, benchmark_returns, config
        )
        progress.update(setup_task, completed=1)
    
    # solver info tracking
    solver_info = {
        'status': 'UNKNOWN',
        'solver_used': None,
        'solve_time': 0.0,
        'objective_value': np.inf,
        'n_iterations': 0,
        'solver_stats': {},
        'sparsity': 0,  # number of non-zero weights
        'l1_norm': 0.0,  # actual L1 norm of solution
    }
    
    # try different solvers
    solvers_to_try = ['ECOS_BB', 'ECOS', 'SCS', 'CLARABEL']
    if config.solver in solvers_to_try:
        # put user's choice first
        solvers_to_try = [config.solver] + [s for s in solvers_to_try if s != config.solver]
    
    start_time = time.time()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
        disable=not verbose
    ) as progress:
        solver_task = progress.add_task(
            f"[yellow]Trying {len(solvers_to_try)} solvers...", 
            total=len(solvers_to_try)
        )
        
        for i, solver_name in enumerate(solvers_to_try):
            try:
                progress.update(
                    solver_task, 
                    description=f"[yellow]Trying {solver_name} solver..."
                )
                
                # solver-specific options
                solver_options = config.solver_options.copy()
                
                if solver_name == 'ECOS_BB':
                    # ECOS_BB is good for mixed-integer problems
                    solver_options.setdefault('mi_max_iters', 1000)
                    solver_options.setdefault('mi_abs_eps', 1e-6)
                    solver_options.setdefault('mi_rel_eps', 1e-4)
                elif solver_name == 'ECOS':
                    solver_options.setdefault('max_iters', 200)
                    solver_options.setdefault('abstol', 1e-7)
                    solver_options.setdefault('reltol', 1e-6)
                elif solver_name == 'SCS':
                    solver_options.setdefault('max_iters', 5000)
                    solver_options.setdefault('eps', 1e-5)
                    solver_options.setdefault('normalize', True)
                elif solver_name == 'CLARABEL':
                    solver_options.setdefault('max_iter', 200)
                    solver_options.setdefault('tol_feas', 1e-7)
                    solver_options.setdefault('tol_gap_abs', 1e-7)
                
                # solve the problem
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    
                    if hasattr(cp, solver_name):
                        solver = getattr(cp, solver_name)
                    else:
                        progress.update(solver_task, advance=1)
                        continue
                    
                    problem.solve(
                        solver=solver,
                        verbose=False,
                        **solver_options
                    )
                
                # check if solution is valid
                if problem.status in ['optimal', 'optimal_inaccurate']:
                    optimal_weights = w_var.value
                    
                    if optimal_weights is None:
                        progress.update(solver_task, advance=1)
                        continue
                    
                    # apply precision management to weights
                    from ..utils.precision import normalize_weights
                    optimal_weights = normalize_weights(optimal_weights)
                    
                    # check for negative weights
                    if np.any(optimal_weights < 0):
                        console.print(f"[orange]WARNING: Negative weights detected![/orange]")
                        console.print(f"Min weight: {np.min(optimal_weights)}")
                        console.print(f"Negative weights at indices: {np.where(optimal_weights < 0)[0]}")
                        # force to zero and renormalize
                        optimal_weights = np.maximum(optimal_weights, 0.0)
                        optimal_weights = normalize_weights(optimal_weights)
                    
                    # calculate sparsity metrics
                    sparsity_threshold = 1e-6
                    n_nonzero = np.sum(np.abs(optimal_weights) > sparsity_threshold)
                    l1_norm = np.sum(np.abs(optimal_weights))
                    
                    # update solver info
                    solver_info.update({
                        'status': problem.status,
                        'solver_used': solver_name,
                        'solve_time': time.time() - start_time,
                        'objective_value': problem.value,
                        'n_iterations': problem.solver_stats.num_iters if hasattr(problem.solver_stats, 'num_iters') else 0,
                        'solver_stats': problem.solver_stats.__dict__ if hasattr(problem.solver_stats, '__dict__') else {},
                        'sparsity': n_nonzero,
                        'l1_norm': l1_norm,
                    })
                    
                    progress.update(solver_task, completed=len(solvers_to_try))
                    
                    if verbose:
                        console.print(f"[green]✓ {solver_name} succeeded![/green]")
                        console.print(f"  Objective: {problem.value:.6f}")
                        console.print(f"  Sparsity: {n_nonzero}/{asset_returns.shape[1]} assets")
                        console.print(f"  L1 norm: {l1_norm:.4f} (bound: {config.sparsity_bound})")
                    
                    return optimal_weights, solver_info
                
                else:
                    progress.update(solver_task, advance=1)
            
            except Exception as e:
                progress.update(solver_task, advance=1)
                continue
    
    # all solvers failed
    solver_info['status'] = 'FAILED'
    solver_info['solve_time'] = time.time() - start_time
    
    if verbose:
        console.print("[red]All solvers failed! Using equal weights as fallback.[/red]")
    
    # return equal weights as fallback
    from ..utils.precision import normalize_weights
    n_assets = asset_returns.shape[1]
    equal_weights = normalize_weights(np.ones(n_assets) / n_assets)
    
    return equal_weights, solver_info


# b/c wrapper
def solve_cvar_with_tracking(
    asset_returns: np.ndarray,
    benchmark_returns: np.ndarray,
    config: OptimizationConfig,
    verbose: bool = False
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Solve CVaR optimization with tracking error (wrapper for CLEIR).
    
    compatibility wrapper that calls CLEIR with a high sparsity bound
    to effectively remove the sparsity constraint.
    """
    # create a modified config with high sparsity bound
    import copy
    cleir_config = copy.deepcopy(config)
    cleir_config.sparsity_bound = 2.0
    
    return solve_cleir(asset_returns, benchmark_returns, cleir_config, verbose) 