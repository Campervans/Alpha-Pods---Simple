# CVaR solver w/ cvxpy

import cvxpy as cp
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, Any
import time
import warnings
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.console import Console

from ..utils.schemas import OptimizationConfig
from .risk_models import calculate_portfolio_returns, calculate_historical_cvar

console = Console()


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
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
        disable=not verbose
    ) as progress:
        setup_task = progress.add_task("[cyan]Setting up CVaR problem...", total=1)
        problem, weights_var, alpha_var, z_var = create_cvar_problem(returns, config)
        progress.update(setup_task, completed=1)
    
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
                    
                    if hasattr(cp, solver_name):
                        solver = getattr(cp, solver_name)
                    else:
                        progress.update(solver_task, advance=1)
                        continue
                    
                    problem.solve(
                        solver=solver,
                        verbose=False,  # Suppress solver output when using rich
                        **solver_options
                    )
                
                # check if worked
                if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                    solve_time = time.time() - start_time
                    
                    optimal_weights = weights_var.value
                    
                    if optimal_weights is None:
                        progress.update(solver_task, advance=1)
                        continue
                    
                    # validate
                    if not _validate_solution(optimal_weights, config):
                        progress.update(solver_task, advance=1)
                        continue
                    
                    # Apply precision management to weights
                    from ..utils.precision import normalize_weights
                    optimal_weights = normalize_weights(optimal_weights)
                    
                    # update info
                    solver_info.update({
                        'status': problem.status,
                        'solver_used': solver_name,
                        'solve_time': solve_time,
                        'objective_value': problem.value,
                        'solver_stats': problem.solver_stats if hasattr(problem, 'solver_stats') else {}
                    })
                    
                    progress.update(solver_task, completed=len(solvers_to_try))
                    
                    if verbose:
                        console.print(f"[green]✓ {solver_name} succeeded![/green]")
                        console.print(f"  Objective: {problem.value:.6f}")
                        console.print(f"  Time: {solve_time:.3f}s")
                    
                    return optimal_weights, solver_info
                
                else:
                    progress.update(solver_task, advance=1)
                    
            except Exception as e:
                progress.update(solver_task, advance=1)
                continue
    
    # all failed - use equal weights
    if verbose:
        console.print("[red]All solvers failed! Using equal weights as fallback.[/red]")
    
    from ..utils.precision import normalize_weights
    n_assets = returns.shape[1]
    fallback_weights = normalize_weights(np.ones(n_assets) / n_assets)
    
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
