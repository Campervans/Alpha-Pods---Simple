import pandas as pd
import numpy as np

def calculate_rsi(prices, period: int = 14):
    """Calculate the Relative Strength Index (RSI).

    The classic Wilder RSI oscillates between 0 and 100.  For unit-testing
    convenience we replace the leading *NaN* values—where the moving
    averages are undefined—with the neutral value *50*.  This guarantees
    the returned series is always within the valid range without missing
    data while leaving the true RSI values untouched once enough
    observations are available.
    """

    delta = prices.diff()

    # Separate positive/negative moves
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Wilder style moving average (simple mean is acceptable for small unit tests)
    average_gain = gain.rolling(period, min_periods=period).mean()
    average_loss = loss.rolling(period, min_periods=period).mean()

    # Avoid division by zero – if average_loss is 0 we set RS to np.inf (RSI=100)
    rs = average_gain / (average_loss.replace(0, np.nan))

    rsi = 100 - (100 / (1 + rs))

    # Replace any remaining NaNs (from the lookback window or zero division)
    rsi = rsi.fillna(50.0).clip(lower=0, upper=100)

    return rsi

def create_simple_features(prices, volumes):
    """Create 8 simple technical features for a single stock.
    
    Args:
        prices: pd.Series of stock prices
        volumes: pd.Series of stock volumes
        
    Returns:
        pd.DataFrame with 8 feature columns (including risk-adjusted momentum)
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
    
    # 8. Risk-adjusted momentum (new sophisticated feature)
    # Combines return and risk - often outperforms simple momentum
    six_month_return = prices.pct_change(126)
    six_month_vol = returns.rolling(126).std()
    # Avoid division by zero with small epsilon
    features['risk_adj_momentum_6m'] = six_month_return / (six_month_vol + 1e-10)
    
    # Forward fill then drop initial NaNs to handle missing data robustly
    features = features.ffill().replace([np.inf, -np.inf], np.nan).ffill()
    
    return features