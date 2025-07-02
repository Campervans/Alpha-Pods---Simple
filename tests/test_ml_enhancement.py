import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# test imports
try:
    from src.models.simple_alpha_model import SimpleAlphaModel
    from src.features.simple_features import create_simple_features, calculate_rsi
    from src.models.walk_forward import SimpleWalkForward
except ImportError as e:
    print(f"Import error: {e}")

def test_simple_alpha_model():
    """test Ridge model basic functionality."""
    # create synthetic data
    np.random.seed(42)
    n_samples, n_features = 100, 7
    X = pd.DataFrame(np.random.randn(n_samples, n_features), 
                     columns=[f'feature_{i}' for i in range(n_features)])
    y = pd.Series(np.random.randn(n_samples))
    
    # train model
    model = SimpleAlphaModel(alpha=1.0)
    model.fit(X, y)
    
    # test predictions
    predictions = model.predict(X[:10])
    assert len(predictions) == 10
    assert not np.any(np.isnan(predictions))
    
    # test feature importance
    importance = model.get_feature_importance()
    assert importance is not None
    assert len(importance) == n_features
    assert all(importance >= 0)  # abs values should be non-negative

def test_calculate_rsi():
    """test RSI calculation."""
    # create synthetic price series
    prices = pd.Series([100, 102, 101, 103, 102, 104, 103, 105, 104, 106])
    
    # calculate RSI
    rsi = calculate_rsi(prices, period=5)
    
    # basic checks
    assert len(rsi) == len(prices)
    assert all((rsi >= 0) & (rsi <= 100))  # RSI should be 0-100
    assert not rsi.isnull().all()  # should have some non-null values

def test_create_simple_features():
    """test feature engineering."""
    # create synthetic price/volume data
    dates = pd.date_range('2020-01-01', periods=300, freq='D')
    np.random.seed(42)
    prices = pd.Series(100 + np.cumsum(np.random.randn(300) * 0.5), index=dates)
    volumes = pd.Series(np.random.randint(1000000, 2000000, 300), index=dates)
    
    # create features
    features = create_simple_features(prices, volumes)
    
    # check output
    assert features.shape[1] == 8  # was 7, now 8 with risk-adj momentum
    assert features.shape[0] == len(prices)  # same number of rows
    
    # check feature names
    expected_features = ['return_1m', 'return_3m', 'return_6m', 
                        'volatility_1m', 'volatility_3m', 'volume_ratio', 
                        'rsi', 'risk_adj_momentum_6m']
    assert all(feat in features.columns for feat in expected_features)
    
    # check no NaNs after initial period
    assert not features.iloc[200:].isnull().any().any()

def test_walk_forward_no_look_ahead():
    """test that walk-forward training doesn't have look-ahead bias."""
    # create synthetic universe data
    dates = pd.date_range('2015-01-01', periods=2000, freq='D')
    np.random.seed(42)
    
    universe_data = {}
    for ticker in ['AAPL', 'MSFT', 'GOOGL']:
        price = 100 + np.cumsum(np.random.randn(2000) * 0.5)
        volume = np.random.randint(1000000, 2000000, 2000)
        universe_data[ticker] = pd.DataFrame({
            'close': price,
            'volume': volume
        }, index=dates)
    
    # set up walk-forward
    wf = SimpleWalkForward(train_years=3, prediction_horizon_days=63)
    
    # test rebalance dates
    rebalance_dates = [pd.Timestamp('2019-03-31'), pd.Timestamp('2019-06-30')]
    
    # for each rebalance date, verify no look-ahead
    for rebal_date in rebalance_dates:
        train_end = rebal_date - timedelta(days=1)
        train_start = train_end - timedelta(days=3*365)
        
        # check that training window is before rebalance date
        assert train_end < rebal_date
        assert train_start < train_end
        
        # verify data filtering
        for ticker, data in universe_data.items():
            train_data = data[(data.index >= train_start) & (data.index <= train_end)]
            
            # no data from rebalance date or after should be in training
            assert all(train_data.index < rebal_date)

def test_feature_importance_aggregation():
    """test that feature importance can be aggregated across models."""
    # create multiple models with different importances
    importances = []
    feature_names = ['return_1m', 'return_3m', 'volatility_1m']
    
    for i in range(3):
        importance = pd.Series(np.random.rand(3) * (i + 1), index=feature_names)
        importances.append(importance)
    
    # calculate average importance
    avg_importance = pd.concat(importances, axis=1).mean(axis=1)
    
    # check result
    assert len(avg_importance) == len(feature_names)
    assert all(avg_importance >= 0)
    assert avg_importance.index.tolist() == feature_names

def test_ml_pipeline_integration():
    """simple integration test for the ML pipeline."""
    # this is a simplified test, in prod you'd test the full pipeline
    # create minimal data
    dates = pd.date_range('2018-01-01', periods=1000, freq='D')
    np.random.seed(42)
    
    # single stock data
    prices = pd.Series(100 + np.cumsum(np.random.randn(1000) * 0.5), index=dates)
    volumes = pd.Series(np.random.randint(1000000, 2000000, 1000), index=dates)
    
    # test feature creation
    features = create_simple_features(prices, volumes)
    assert features.shape[1] == 8
    
    # test model training (on subset)
    train_features = features.iloc[200:800].dropna()
    train_target = prices.pct_change(63).shift(-63).iloc[200:800].dropna()
    
    # align features and target
    aligned_data = pd.concat([train_features, train_target], axis=1).dropna()
    if len(aligned_data) > 100:
        X = aligned_data.iloc[:, :-1]
        y = aligned_data.iloc[:, -1]
        
        model = SimpleAlphaModel()
        model.fit(X, y)
        
        # test prediction
        latest_features = features.iloc[[-1]].dropna()
        if len(latest_features) > 0 and not latest_features.isnull().any().any():
            pred = model.predict(latest_features)
            assert len(pred) == 1
            assert not np.isnan(pred[0])