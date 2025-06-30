# CVaR solver w/ cvxpy

import cvxpy as cp
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, Any
import time
import warnings

from ..utils.schemas import OptimizationConfig
from .risk_models import calculate_portfolio_returns, calculate_historical_cvar


def create_cvar_problem(returns: np.ndarray, 
                       config: OptimizationConfig) -> Tuple[cp.Problem, cp.Variable, cp.Variable, cp.Variable]:
    # setup cvar optimization problem
    # minimize cvar subject to portfolio constraints
    n_periods, n_assets = returns.shape
    
    # vars
    weights = cp.Variable(n_assets, name="weights")
    alpha = cp.Variable(1, name="alpha")  # VaR threshold
    z = cp.Variable(n_periods, name="z")  # aux vars for cvar
    
    # portfolio returns
    portfolio_returns = returns @ weights
    
    # cvar formula
    confidence = config.confidence_level
    cvar_term = alpha + (1.0 / (n_periods * (1 - confidence))) * cp.sum(z)
    
    # constraints
    constraints = [
        # portfolio stuff
        cp.sum(weights) == 1.0,                    # fully invested
        weights >= config.min_weight,              # long only
        weights <= config.max_weight,              # max weight
        
        # cvar constraints
        z >= 0,                                    
        z >= -portfolio_returns - alpha           # main cvar constraint
    ]
    
    # minimize cvar
    objective = cp.Minimize(cvar_term)
    
    problem = cp.Problem(objective, constraints)
    
    return problem, weights, alpha, z


def solve_cvar(returns: np.ndarray, 
               config: OptimizationConfig,
               verbose: bool = False) -> Tuple[np.ndarray, Dict[str, Any]]:
    # solve cvar problem, try multiple solvers if needed
    if returns.shape[0] < 10:
        raise ValueError("Need at least 10 observations for optimization")
    
    if returns.shape[1] == 0:
        raise ValueError("No assets to optimize")
    
    # create problem
    problem, weights_var, alpha_var, z_var = create_cvar_problem(returns, config)
    
    # track solver info
    solver_info = {
        'status': 'UNKNOWN',
        'solver_used': None,
        'solve_time': 0.0,
        'objective_value': np.inf,
        'n_iterations': 0,
        'solver_stats': {}
    }
    
    # solvers to try (in order)
    solvers_to_try = [config.solver, 'ECOS', 'SCS', 'OSQP', 'CLARABEL']
    
    # dedupe
    solvers_to_try = list(dict.fromkeys(solvers_to_try))
    
    start_time = time.time()
    
    for solver_name in solvers_to_try:
        try:
            if verbose:
                print(f"Trying {solver_name}...")
            
            # solver options
            solver_options = config.solver_options.copy()
            
            if solver_name == 'ECOS':
                solver_options.setdefault('max_iters', 100)
                solver_options.setdefault('abstol', 1e-7)
                solver_options.setdefault('reltol', 1e-7)
            elif solver_name == 'SCS':
                solver_options.setdefault('max_iters', 5000)
                solver_options.setdefault('eps', 1e-5)
            elif solver_name == 'OSQP':
                solver_options.setdefault('max_iter', 4000)
                solver_options.setdefault('eps_abs', 1e-6)
                solver_options.setdefault('eps_rel', 1e-6)
            # clarabel usually works well with defaults
            
            # solve it
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # shh
                
                problem.solve(
                    solver=getattr(cp, solver_name),
                    verbose=verbose,
                    **solver_options
                )
            
            # check if worked
            if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                solve_time = time.time() - start_time
                
                optimal_weights = weights_var.value
                
                if optimal_weights is None:
                    continue
                
                # validate
                if not _validate_solution(optimal_weights, config):
                    if verbose:
                        print(f"Bad solution from {solver_name}")
                    continue
                
                # update info
                solver_info.update({
                    'status': problem.status,
                    'solver_used': solver_name,
                    'solve_time': solve_time,
                    'objective_value': problem.value,
                    'solver_stats': problem.solver_stats if hasattr(problem, 'solver_stats') else {}
                })
                
                if verbose:
                    print(f"Solver {solver_name} worked!")
                    print(f"Obj: {problem.value:.6f}")
                    print(f"Time: {solve_time:.3f}s")
                
                return optimal_weights, solver_info
            
            else:
                if verbose:
                    print(f"{solver_name} failed: {problem.status}")
                
        except Exception as e:
            if verbose:
                print(f"Error w/ {solver_name}: {e}")
            continue
    
    # all failed - use equal weights
    print("X All solvers failed! Using equal weights...")
    n_assets = returns.shape[1]
    fallback_weights = np.ones(n_assets) / n_assets
    
    solver_info.update({
        'status': 'FALLBACK_EQUAL_WEIGHTS',
        'solver_used': 'FALLBACK',
        'solve_time': time.time() - start_time
    })
    
    return fallback_weights, solver_info


def _validate_solution(weights: np.ndarray, config: OptimizationConfig) -> bool:
    # check if weights are valid
    if weights is None:
        return False
    
    # no nans or infs
    if not np.isfinite(weights).all():
        return False
    
    # sum to 1?
    if not np.isclose(weights.sum(), 1.0, atol=1e-6):
        return False
    
    # bounds ok?
    if np.any(weights < config.min_weight - 1e-6):
        return False
    
    if np.any(weights > config.max_weight + 1e-6):
        return False
    
    return True


def calculate_realized_cvar(weights: np.ndarray, 
                          returns: np.ndarray,
                          confidence: float = 0.95) -> float:
    # calc realized cvar for a portfolio
    portfolio_returns = calculate_portfolio_returns(weights, returns)
    return calculate_historical_cvar(portfolio_returns, confidence)

# TODO: refactor methods 
