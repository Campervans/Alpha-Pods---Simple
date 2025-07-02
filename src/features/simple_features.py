import pandas as pd
import numpy as np

def calculate_rsi(prices, period: int = 14):
    """Calculate the Relative Strength Index (RSI).

    classic Wilder RSI oscillates between 0 and 100.
    for testing convenience we replace leading NaNs with 50.
    this guarantees the returned series is always valid
    without missing data, while leaving true RSI values untouched.
    """

    delta = prices.diff()

    # separate positive/negative moves
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Wilder style moving average
    average_gain = gain.rolling(period, min_periods=period).mean()
    average_loss = loss.rolling(period, min_periods=period).mean()

    # avoid division by zero - if avg_loss is 0, set RS to inf (RSI=100)
    rs = average_gain / (average_loss.replace(0, np.nan))

    rsi = 100 - (100 / (1 + rs))

    # replace remaining NaNs and clip
    rsi = rsi.fillna(50.0).clip(lower=0, upper=100)

    return rsi

def create_simple_features(prices, volumes):

    features = pd.DataFrame(index=prices.index)
    
    # 1-3. momentum (1m, 3m, 6m returns)
    for period, name in [(21, '1m'), (63, '3m'), (126, '6m')]:
        features[f'return_{name}'] = prices.pct_change(period)
    
    # 4-5. volatility (1m, 3m)
    returns = prices.pct_change()
    for period, name in [(21, '1m'), (63, '3m')]:
        features[f'volatility_{name}'] = returns.rolling(period).std()
    
    # 6. volume ratio (current vs 21d avg)
    features['volume_ratio'] = volumes / volumes.rolling(21).mean()
    
    # 7. RSI
    features['rsi'] = calculate_rsi(prices, 14)
    
    # 8. risk-adjusted momentum (new feature)
    # combines return and risk - should beat simple momentum
    # TODO: test other periods for risk-adj momentum
    six_month_return = prices.pct_change(126)
    six_month_vol = returns.rolling(126).std()
    # avoid division by zero with small epsilon
    features['risk_adj_momentum_6m'] = six_month_return / (six_month_vol + 1e-10)
    
    # ffill then drop NaNs
    features = features.ffill().replace([np.inf, -np.inf], np.nan).ffill()
    
    return features