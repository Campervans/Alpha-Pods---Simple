# ML CLEIR Optimizer (Alpha Overlay)
from typing import Optional, Dict
# from src.backtesting.alpha_engine import AlphaEnhance
from src.backtesting.alpha_engine import AlphaEnhancedBacktest
from src.utils.schemas import OptimizationConfig


class MLCLEIROptimizer(AlphaEnhancedBacktest):
    """
    Paras:
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
        """
        Paras forwarded to
        AlphaEnhancedBacktest.run
        """
        return super().run(*args, **kwargs) 