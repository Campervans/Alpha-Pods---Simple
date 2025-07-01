import pandas as pd
import numpy as np

def calculate_rsi(prices, period=14):
    """Calculate RSI indicator."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def create_simple_features(prices, volumes):
    """Create 7 simple technical features for a single stock.
    
    Args:
        prices: pd.Series of stock prices
        volumes: pd.Series of stock volumes
        
    Returns:
        pd.DataFrame with 7 feature columns
    """
    features = pd.DataFrame(index=prices.index)
    
    # 1-3. Momentum features (1m, 3m, 6m returns)
    for period, name in [(21, '1m'), (63, '3m'), (126, '6m')]:
        features[f'return_{name}'] = prices.pct_change(period)
    
    # 4-5. Volatility features (1m, 3m)
    returns = prices.pct_change()
    for period, name in [(21, '1m'), (63, '3m')]:
        features[f'volatility_{name}'] = returns.rolling(period).std()
    
    # 6. Volume ratio (current vs 21-day average)
    features['volume_ratio'] = volumes / volumes.rolling(21).mean()
    
    # 7. RSI
    features['rsi'] = calculate_rsi(prices, 14)
    
    # Forward fill then drop initial NaNs to handle missing data robustly
    features = features.ffill().replace([np.inf, -np.inf], np.nan).ffill()
    
    return features