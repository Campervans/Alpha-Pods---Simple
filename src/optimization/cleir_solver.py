"""
CVaR-LASSO Enhanced Index Replication (CLEIR) solver.

This module implements the CLEIR optimization problem as a linear program:
- Minimize CVaR of tracking error (benchmark return - portfolio return)
- Subject to L1 norm constraint on weights (sparsity)
- Subject to budget constraint (weights sum to 1)
"""

import cvxpy as cp
import numpy as np
from typing import Tuple, Dict, Any, Optional
import time
import warnings

from ..utils.schemas import OptimizationConfig


def create_cleir_problem(
    asset_returns: np.ndarray,
    benchmark_returns: np.ndarray,
    config: OptimizationConfig
) -> Tuple[cp.Problem, cp.Variable, cp.Variable, cp.Variable, cp.Variable]:
    """
    Create the CLEIR linear programming problem.
    
    Args:
        asset_returns: Historical asset returns (n_periods x n_assets)
        benchmark_returns: Historical benchmark returns (n_periods,)
        config: Optimization configuration with sparsity_bound
        
    Returns:
        Tuple of (problem, weights, zeta, z, u)
    """
    n_periods, n_assets = asset_returns.shape
    
    if len(benchmark_returns) != n_periods:
        raise ValueError(
            f"Benchmark returns length ({len(benchmark_returns)}) must match "
            f"number of periods ({n_periods})"
        )
    
    # Decision variables
    w = cp.Variable(n_assets, name="weights")  # Portfolio weights
    zeta = cp.Variable(1, name="zeta")  # VaR threshold
    z = cp.Variable(n_periods, name="z")  # Auxiliary variables for CVaR
    u = cp.Variable(n_assets, name="u")  # Auxiliary variables for L1 norm
    
    # Tracking error: portfolio return - benchmark return
    # We want to minimize downside risk, so we look at negative tracking error
    portfolio_returns = asset_returns @ w
    # Use negative tracking error so CVaR captures underperformance
    tracking_error = benchmark_returns - portfolio_returns
    
    # CVaR objective: zeta + 1/(n*(1-alpha)) * sum(z)
    alpha = config.confidence_level
    cvar_objective = zeta + (1.0 / (n_periods * (1 - alpha))) * cp.sum(z)
    
    # Constraints
    constraints = [
        # CVaR constraints
        z >= tracking_error - zeta,  # z_t >= Y_t - sum(w_i*R_it) - zeta
        z >= 0,
        
        # L1 norm linearization: u_i >= |w_i|
        u >= w,
        u >= -w,
        
        # Sparsity constraint: sum(|w_i|) <= s
        cp.sum(u) <= config.sparsity_bound,
        
        # Budget constraint: sum(w_i) = 1
        cp.sum(w) == 1.0,
    ]
    
    # Add weight bounds if specified
    # Always enforce non-negativity for long-only portfolios
    constraints.append(w >= config.min_weight)
    if config.max_weight < 1:
        constraints.append(w <= config.max_weight)
    
    # Create optimization problem
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
        asset_returns: Historical asset returns (n_periods x n_assets)
        benchmark_returns: Historical benchmark returns (n_periods,)
        config: Optimization configuration
        verbose: Whether to print solver output
        
    Returns:
        Tuple of (optimal_weights, solver_info)
    """
    if asset_returns.shape[0] < 10:
        raise ValueError("Need at least 10 observations for optimization")
    
    if asset_returns.shape[1] == 0:
        raise ValueError("No assets to optimize")
    
    if config.sparsity_bound is None:
        raise ValueError("CLEIR requires sparsity_bound to be set")
    
    # Create the optimization problem
    problem, w_var, zeta_var, z_var, u_var = create_cleir_problem(
        asset_returns, benchmark_returns, config
    )
    
    # Solver info tracking
    solver_info = {
        'status': 'UNKNOWN',
        'solver_used': None,
        'solve_time': 0.0,
        'objective_value': np.inf,
        'n_iterations': 0,
        'solver_stats': {},
        'sparsity': 0,  # Number of non-zero weights
        'l1_norm': 0.0,  # Actual L1 norm of solution
    }
    
    # Try different solvers in order of preference
    solvers_to_try = ['ECOS_BB', 'ECOS', 'SCS', 'CLARABEL']
    if config.solver in solvers_to_try:
        # Put user's choice first
        solvers_to_try = [config.solver] + [s for s in solvers_to_try if s != config.solver]
    
    start_time = time.time()
    
    for solver_name in solvers_to_try:
        try:
            if verbose:
                print(f"Trying {solver_name} solver...")
            
            # Solver-specific options
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
            
            # Solve the problem
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                if hasattr(cp, solver_name):
                    solver = getattr(cp, solver_name)
                else:
                    if verbose:
                        print(f"Solver {solver_name} not available")
                    continue
                
                problem.solve(
                    solver=solver,
                    verbose=verbose,
                    **solver_options
                )
            
            # Check if solution is valid
            if problem.status in ['optimal', 'optimal_inaccurate']:
                optimal_weights = w_var.value
                
                if optimal_weights is None:
                    if verbose:
                        print(f"{solver_name} returned None weights")
                    continue
                
                # Normalize weights to ensure they sum to 1
                weight_sum = np.sum(optimal_weights)
                if abs(weight_sum - 1.0) > 1e-6:
                    if verbose:
                        print(f"Normalizing weights (sum was {weight_sum:.6f})")
                    optimal_weights = optimal_weights / weight_sum
                
                # Clean up numerical noise - set very small weights to zero
                optimal_weights[np.abs(optimal_weights) < 1e-6] = 0.0
                
                # Check for negative weights after cleanup
                if np.any(optimal_weights < 0):
                    print(f"WARNING: Negative weights detected after optimization!")
                    print(f"Min weight: {np.min(optimal_weights)}")
                    print(f"Negative weights at indices: {np.where(optimal_weights < 0)[0]}")
                    # Force to zero
                    optimal_weights = np.maximum(optimal_weights, 0.0)
                    # Renormalize
                    optimal_weights = optimal_weights / np.sum(optimal_weights)
                
                # Calculate sparsity metrics
                sparsity_threshold = 1e-6
                n_nonzero = np.sum(np.abs(optimal_weights) > sparsity_threshold)
                l1_norm = np.sum(np.abs(optimal_weights))
                
                # Update solver info
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
                
                if verbose:
                    print(f"✓ {solver_name} succeeded!")
                    print(f"  Objective: {problem.value:.6f}")
                    print(f"  Sparsity: {n_nonzero}/{asset_returns.shape[1]} assets")
                    print(f"  L1 norm: {l1_norm:.4f} (bound: {config.sparsity_bound})")
                
                return optimal_weights, solver_info
            
            else:
                if verbose:
                    print(f"{solver_name} failed with status: {problem.status}")
        
        except Exception as e:
            if verbose:
                print(f"{solver_name} error: {str(e)}")
            continue
    
    # All solvers failed
    solver_info['status'] = 'FAILED'
    solver_info['solve_time'] = time.time() - start_time
    
    # Return equal weights as fallback
    n_assets = asset_returns.shape[1]
    equal_weights = np.ones(n_assets) / n_assets
    
    return equal_weights, solver_info


# Backward compatibility wrapper
def solve_cvar_with_tracking(
    asset_returns: np.ndarray,
    benchmark_returns: np.ndarray,
    config: OptimizationConfig,
    verbose: bool = False
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Solve CVaR optimization with tracking error (wrapper for CLEIR).
    
    This is a compatibility wrapper that calls CLEIR with a high sparsity bound
    to effectively remove the sparsity constraint.
    """
    # Create a modified config with high sparsity bound
    import copy
    cleir_config = copy.deepcopy(config)
    cleir_config.sparsity_bound = 2.0  # High enough to not bind
    
    return solve_cleir(asset_returns, benchmark_returns, cleir_config, verbose) 