# optimization module

from .cvar_solver import solve_cvar, create_cvar_problem
from .cleir_solver import solve_cleir, create_cleir_problem, solve_cvar_with_tracking
from .risk_models import (
    calculate_portfolio_returns,
    calculate_historical_cvar,
    calculate_historical_var,
    calculate_expected_shortfall,
    estimate_covariance_matrix
)

# Lazy import for MLCLEIROptimizer to avoid circular imports
def _get_ml_cleir_optimizer():
    """Lazy loader for MLCLEIROptimizer."""
    from .ml_cleir_optimizer import MLCLEIROptimizer
    return MLCLEIROptimizer

# Make it available as an attribute that returns the actual class when accessed
class _LazyMLCLEIROptimizer:
    def __new__(cls, *args, **kwargs):
        # When instantiated, return an instance of the actual class
        actual_class = _get_ml_cleir_optimizer()
        return actual_class(*args, **kwargs)
    
    def __getattr__(self, name):
        # For accessing class attributes/methods
        actual_class = _get_ml_cleir_optimizer()
        return getattr(actual_class, name)

# This will act as the class itself
MLCLEIROptimizer = _LazyMLCLEIROptimizer

__all__ = [
    'solve_cvar',
    'create_cvar_problem',
    'solve_cleir',
    'create_cleir_problem',
    'solve_cvar_with_tracking',
    'calculate_portfolio_returns',
    'calculate_historical_cvar',
    'calculate_historical_var',
    'calculate_expected_shortfall',
    'estimate_covariance_matrix',
    'MLCLEIROptimizer',
]
