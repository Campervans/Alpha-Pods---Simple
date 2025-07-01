# Method Note: ML-Enhanced CLEIR Portfolio

## Methodology Overview

This project enhances the baseline CVaR index with a simple but robust machine learning alpha overlay. The core idea is to use a model to identify a high-conviction subset of stocks, which then becomes the investable universe for the existing CVaR optimization logic. This approach isolates the alpha signal from the risk management, maintaining the benefits of the original CLEIR framework while improving stock selection.

## Implementation Details

1. **Feature Engineering**: Seven standard technical indicators (momentum, volatility, volume, and RSI) are calculated for each stock using its historical price and volume data. These features capture well-known market patterns:
   - 3 momentum features: 1-month, 3-month, and 6-month returns
   - 2 volatility features: 1-month and 3-month rolling standard deviations
   - 1 volume feature: current volume relative to 21-day average
   - 1 mean-reversion indicator: 14-day RSI

2. **Model Training**: A Ridge regression model is trained for each stock to predict its subsequent 3-month return based on the engineered features. To prevent look-ahead bias, we employ a strict walk-forward methodology with a 3-year rolling training window. A new model is trained at each quarterly rebalance date using only historical data available up to that point.

3. **Portfolio Construction**: At each rebalance, we use the trained models to predict returns for all stocks in the universe. The **top 30 stocks** with the highest predicted returns are selected. The standard CVaR optimization is then performed on this smaller, pre-selected universe, allowing the risk model to focus on the most promising opportunities.

## Key Design Choices

- **Simplicity**: We chose Ridge regression for its stability and inherent regularization, which prevents overfitting. All parameters are fixed to avoid data snooping.
- **Robustness**: The walk-forward training ensures that all predictions are truly out-of-sample. Using a per-stock model accounts for different asset characteristics.
- **Interpretability**: The linear nature of the Ridge model allows for straightforward feature importance analysis via its coefficients.

## Conclusion

This ML enhancement successfully improves the baseline strategy's risk-adjusted returns by focusing the portfolio on assets with strong predictive signals, demonstrating how a simple, well-grounded ML overlay can add significant value without introducing unnecessary complexity.