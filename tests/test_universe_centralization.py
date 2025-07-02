"""Test that universe is properly centralized."""

def test_ml_universe_size():
    """Test that ML universe has exactly 60 stocks."""
    from src.market_data.universe import get_ml_universe
    
    universe = get_ml_universe()
    assert len(universe) == 60, f"Expected 60 stocks, got {len(universe)}"
    
    # Check all are strings
    assert all(isinstance(ticker, str) for ticker in universe)
    
    # Check no duplicates
    assert len(set(universe)) == len(universe), "Universe contains duplicates"
    
    print("✅ ML universe has 60 unique stocks")


def test_universe_import():
    """Test that alpha_engine can import universe function."""
    try:
        from src.backtesting.alpha_engine import AlphaEnhancedBacktest
        from src.market_data.universe import get_ml_universe
        
        # Should be able to create instance
        backtest = AlphaEnhancedBacktest()
        assert backtest is not None
        
        print("✅ Universe import works correctly")
    except ImportError as e:
        raise AssertionError(f"Import failed: {e}")


def test_default_top_k_is_60():
    """Test that default top_k parameter is now 60."""
    from src.backtesting.alpha_engine import AlphaEnhancedBacktest
    from src.optimization.ml_cleir_optimizer import MLCLEIROptimizer
    
    # Test AlphaEnhancedBacktest
    backtest = AlphaEnhancedBacktest()
    assert backtest.top_k == 60, f"Expected top_k=60, got {backtest.top_k}"
    
    # Test MLCLEIROptimizer
    optimizer = MLCLEIROptimizer()
    assert optimizer.top_k == 60, f"Expected top_k=60, got {optimizer.top_k}"
    
    print("✅ Default top_k is 60 for both classes")


if __name__ == "__main__":
    test_ml_universe_size()
    test_universe_import()
    test_default_top_k_is_60()
    print("\n✅ All universe centralization tests passed!") 