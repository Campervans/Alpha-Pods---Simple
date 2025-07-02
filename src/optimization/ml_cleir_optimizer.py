"""
ML CLEIR Optimizer (Alpha Overlay)

This module provides an optimization-namespace alias for the
Alpha-enhanced CLEIR backtest (alpha overlay) so that users can import a
clean, descriptive class from ``src.optimization`` just like the other
solvers.

It simply subclasses ``AlphaEnhancedBacktest`` that lives in
``src.backtesting.alpha_engine`` and forwards all functionality without
modification. This keeps backward compatibility while exposing the new
optimizer in the expected package location.
"""

from typing import Optional, Dict

from src.backtesting.alpha_engine import AlphaEnhancedBacktest
from src.utils.schemas import OptimizationConfig


class MLCLEIROptimizer(AlphaEnhancedBacktest):
    """ML-enhanced CLEIR optimizer (alpha overlay).

    Parameters
    ----------
    optimization_config : Optional[OptimizationConfig], default None
        Custom optimization configuration. If *None*, the default
        configuration from ``AlphaEnhancedBacktest`` is used.
    top_k : int, default 60
        Number of top-ranked stocks (by predicted alpha) to include in
        each quarterly rebalance. Changed from 30 to use full universe.
    """

    def __init__(self,
                 optimization_config: Optional[OptimizationConfig] = None,
                 top_k: int = 60):
        super().__init__(optimization_config=optimization_config, top_k=top_k)

    # Explicitly expose run() just for type checkers / clarity (delegates)
    def run(self, *args, **kwargs) -> Dict:
        """Run the ML-enhanced CLEIR backtest.

        Parameters are forwarded directly to
        ``AlphaEnhancedBacktest.run``.
        """
        return super().run(*args, **kwargs) 