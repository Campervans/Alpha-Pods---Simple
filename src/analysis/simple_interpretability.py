import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from typing import Dict, Optional, Tuple
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: SHAP not available. Install with: pip install shap")


def get_background_sample(X: pd.DataFrame, n: int = 1000) -> pd.DataFrame:
    """Get random background sample for SHAP.
    
    Args:
        X: Full feature matrix
        n: Number of samples to select
        
    Returns:
        Random sample of X
    """
    if len(X) <= n:
        return X
    return X.sample(n=n, random_state=42)


def get_recent_sample(X: pd.DataFrame, n: int = 500) -> pd.DataFrame:
    """Get most recent samples for SHAP analysis.
    
    Args:
        X: Full feature matrix with datetime index
        n: Number of recent samples
        
    Returns:
        Most recent n rows of X
    """
    return X.tail(n)

def plot_feature_importance(trainer, num_features: int = 10):
    """Plots the average feature importance across all trained models.
    
    Args:
        trainer: SimpleWalkForward instance with trained models
        num_features: Number of top features to display
    """
    # Collect feature importances from all models
    importances = []
    for (date, ticker), model in trainer.models.items():
        importance = model.get_feature_importance()
        if importance is not None:
            importances.append(importance)
    
    if not importances:
        print("No feature importance data available.")
        return
    
    # Calculate average importance across all models
    avg_importance = pd.concat(importances, axis=1).mean(axis=1).sort_values(ascending=False)
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    top_features = avg_importance.head(num_features)
    
    # Create bar plot
    ax = sns.barplot(x=top_features.values, y=top_features.index, palette='viridis')
    
    # Add value labels
    for i, (value, name) in enumerate(zip(top_features.values, top_features.index)):
        ax.text(value, i, f'{value:.3f}', va='center', ha='left', fontsize=10)
    
    plt.title(f'Average Feature Importance (Top {num_features} Features)', fontsize=16, fontweight='bold')
    plt.xlabel('Absolute Ridge Coefficient', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('results/ml_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Feature importance plot saved to results/ml_feature_importance.png")

def plot_shap_analysis(trainer, num_samples: int = 100):
    """Generate SHAP analysis for the ML models.
    
    Args:
        trainer: SimpleWalkForward instance with trained models
        num_samples: Number of samples to use for SHAP analysis
    """
    if not SHAP_AVAILABLE:
        print("⚠️  SHAP not available. Skipping SHAP analysis.")
        return
    
    # Import and run aggregated analysis
    try:
        from src.analysis.shap_aggregation import plot_aggregated_shap_analysis
        plot_aggregated_shap_analysis(trainer, num_samples)
        
        # Also create the single-model example for comparison
        _plot_single_model_shap(trainer, num_samples)
        
    except Exception as e:
        print(f"Warning: Could not run aggregated SHAP analysis: {e}")
        print("Falling back to single-model analysis...")
        _plot_single_model_shap(trainer, num_samples)


def _plot_single_model_shap(trainer, num_samples: int = 100):
    """Plot SHAP analysis for a single representative model.
    
    This is kept for backward compatibility and as an example
    of individual model interpretation.
    """
    # Get the most recent model and its data
    latest_date = max([date for date, _ in trainer.models.keys()])
    latest_models = [(ticker, model) for (date, ticker), model in trainer.models.items() if date == latest_date]
    
    if not latest_models:
        print("No models available for SHAP analysis.")
        return
    
    # Use the first model with stored training data
    ticker, model = None, None
    for t, m in latest_models:
        if hasattr(m, 'X_train_') and m.X_train_ is not None:
            ticker, model = t, m
            break
    
    if model is None:
        print("No model with stored training data found. Using coefficient proxy.")
        ticker, model = latest_models[0]
        _plot_coefficient_proxy(ticker, model)
        return
    
    print(f"Generating SHAP values for {ticker} model (example)...")
    
    # Get background and sample data
    X_background = get_background_sample(model.X_train_, n=min(1000, len(model.X_train_)))
    X_sample = get_recent_sample(model.X_train_, n=min(500, len(model.X_train_)))
    
    # Scale the data using the model's scaler
    X_background_scaled = model.scaler.transform(X_background)
    X_sample_scaled = model.scaler.transform(X_sample)
    
    # Create SHAP explainer
    explainer = shap.LinearExplainer(model.model, X_background_scaled, feature_names=model.feature_names)
    shap_values = explainer.shap_values(X_sample_scaled)
    
    # Create figure with proper layout
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(f'SHAP Analysis for {ticker} Alpha Model', fontsize=20, fontweight='bold')
    
    # Create grid spec for layout with increased spacing
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1.5, 1], hspace=0.5, wspace=0.4)
    
    # 1. Bar plot of mean |SHAP| values
    ax1 = fig.add_subplot(gs[0, :])
    shap_importance = pd.Series(np.abs(shap_values).mean(axis=0), index=model.feature_names).sort_values(ascending=False)
    
    colors = ['#1f77b4' if model.model.coef_[model.feature_names.index(feat)] < 0 else '#ff7f0e' 
              for feat in shap_importance.index]
    
    bars = ax1.bar(range(len(shap_importance)), shap_importance.values, color=colors)
    ax1.set_xticks(range(len(shap_importance)))
    ax1.set_xticklabels(shap_importance.index, rotation=45, ha='right')
    ax1.set_ylabel('Mean |SHAP value|', fontsize=12)
    ax1.set_title('Global Feature Importance (Mean Absolute SHAP Values)', fontsize=14)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#ff7f0e', label='Positive coefficient'),
                      Patch(facecolor='#1f77b4', label='Negative coefficient')]
    ax1.legend(handles=legend_elements, loc='upper right')
    
    # 2. Beeswarm plot
    ax2 = fig.add_subplot(gs[1, :])
    shap.summary_plot(shap_values, X_sample, feature_names=model.feature_names, 
                      show=False, plot_size=None)
    plt.sca(ax2)
    ax2.set_title('SHAP Value Distribution (Beeswarm Plot)', fontsize=14)
    
    # 3. Dependence plots for top 2 features
    top_2_features = shap_importance.head(2).index.tolist()
    
    for i, feature in enumerate(top_2_features):
        ax = fig.add_subplot(gs[2, i])
        feat_idx = model.feature_names.index(feature)
        shap.dependence_plot(feat_idx, shap_values, X_sample, 
                           feature_names=model.feature_names,
                           show=False, ax=ax)
        ax.set_title(f'SHAP Dependence: {feature}', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('results/ml_shap_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ SHAP analysis for {ticker} saved to results/ml_shap_analysis.png")


def _plot_coefficient_proxy(ticker: str, model):
    """Fallback coefficient plot when SHAP data not available."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle(f'Feature Analysis for {ticker} (Coefficient-based)', fontsize=16, fontweight='bold')
    
    feature_names = model.feature_names
    
    # Plot 1: Absolute coefficients
    ax1 = axes[0]
    coef_importance = pd.Series(np.abs(model.model.coef_), index=feature_names).sort_values(ascending=False)
    
    bars = ax1.barh(range(len(coef_importance)), coef_importance.values)
    ax1.set_yticks(range(len(coef_importance)))
    ax1.set_yticklabels(coef_importance.index)
    ax1.set_xlabel('Absolute Coefficient Value')
    ax1.set_title('Feature Importance (Absolute Coefficients)')
    
    # Color by sign
    for i, (feat, val) in enumerate(coef_importance.items()):
        idx = list(feature_names).index(feat)
        if model.model.coef_[idx] > 0:
            bars[i].set_color('#ff7f0e')
        else:
            bars[i].set_color('#1f77b4')
    
    # Plot 2: Signed coefficients
    ax2 = axes[1]
    coef_values = pd.Series(model.model.coef_, index=feature_names).sort_values()
    
    colors = ['#1f77b4' if x < 0 else '#ff7f0e' for x in coef_values.values]
    ax2.barh(range(len(coef_values)), coef_values.values, color=colors)
    ax2.set_yticks(range(len(coef_values)))
    ax2.set_yticklabels(coef_values.index)
    ax2.set_xlabel('Coefficient Value')
    ax2.set_title('Feature Coefficients with Direction')
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig('results/ml_shap_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Coefficient analysis saved to results/ml_shap_analysis.png")

def plot_performance_comparison(baseline_perf: pd.Series, ml_perf: pd.Series, 
                               benchmark_perf: Optional[pd.Series] = None):
    """Plots the cumulative performance of baseline vs. ML-enhanced index.
    
    Args:
        baseline_perf: Series of baseline index values
        ml_perf: Series of ML-enhanced index values
        benchmark_perf: Optional series of benchmark (e.g., SPY) values
    """
    # Align the series to have the same date range
    # Use the intersection of dates
    common_dates = baseline_perf.index.intersection(ml_perf.index)
    
    if len(common_dates) == 0:
        print("Error: No common dates between baseline and ML series")
        return
    
    # Create comparison dataframe with aligned data
    comparison_df = pd.DataFrame({
        'Baseline CLEIR Index': baseline_perf.loc[common_dates],
        'ML-Enhanced CLEIR Index': ml_perf.loc[common_dates]
    })
    
    # Add benchmark if provided
    if benchmark_perf is not None and len(benchmark_perf.index.intersection(common_dates)) > 0:
        comparison_df['Benchmark (SPY)'] = benchmark_perf.loc[common_dates]
    
    # Normalize to start at 1
    comparison_df_normalized = comparison_df / comparison_df.iloc[0]
    
    # Create the plot
    plt.figure(figsize=(14, 8))
    
    # Plot lines
    for col in comparison_df_normalized.columns:
        if 'ML-Enhanced' in col:
            plt.plot(comparison_df_normalized.index, comparison_df_normalized[col], 
                    label=col, linewidth=2.5, color='green', alpha=0.8)
        elif 'Baseline' in col:
            plt.plot(comparison_df_normalized.index, comparison_df_normalized[col], 
                    label=col, linewidth=2, color='blue', alpha=0.7)
        else:
            plt.plot(comparison_df_normalized.index, comparison_df_normalized[col], 
                    label=col, linewidth=1.5, color='gray', alpha=0.6, linestyle='--')
    
    plt.title('Performance Comparison: Baseline CLEIR vs. ML-Enhanced CLEIR', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Cumulative Growth of $1', fontsize=12)
    plt.legend(loc='upper left', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add performance stats
    baseline_col = [col for col in comparison_df_normalized.columns if 'Baseline' in col][0]
    ml_col = [col for col in comparison_df_normalized.columns if 'ML-Enhanced' in col][0]
    
    baseline_return = (comparison_df_normalized[baseline_col].iloc[-1] - 1) * 100
    ml_return = (comparison_df_normalized[ml_col].iloc[-1] - 1) * 100
    improvement = ml_return - baseline_return
    
    stats_text = f'Baseline Return: {baseline_return:.1f}%\nML-Enhanced Return: {ml_return:.1f}%\nImprovement: {improvement:.1f}%'
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
             verticalalignment='top', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('results/ml_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Performance comparison plot saved to results/ml_performance_comparison.png")

def analyze_ml_predictions(ml_predictions: Dict[pd.Timestamp, pd.Series], 
                          selected_universes: Dict[pd.Timestamp, list]):
    """Analyze ML prediction patterns and universe selection.
    
    Args:
        ml_predictions: Dictionary of predictions by date
        selected_universes: Dictionary of selected tickers by date
    """
    analysis_results = []
    
    for date in ml_predictions.keys():
        if date not in selected_universes:
            continue
            
        predictions = ml_predictions[date]
        selected = selected_universes[date]
        
        analysis_results.append({
            'date': date,
            'n_predictions': len(predictions),
            'n_selected': len(selected),
            'avg_prediction': predictions.mean(),
            'std_prediction': predictions.std(),
            'max_prediction': predictions.max(),
            'min_prediction': predictions.min(),
            'prediction_spread': predictions.max() - predictions.min()
        })
    
    analysis_df = pd.DataFrame(analysis_results)
    
    # Create subplots for analysis
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('ML Predictions Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Average predictions over time
    ax1 = axes[0, 0]
    ax1.plot(analysis_df['date'], analysis_df['avg_prediction'], marker='o')
    ax1.fill_between(analysis_df['date'], 
                     analysis_df['avg_prediction'] - analysis_df['std_prediction'],
                     analysis_df['avg_prediction'] + analysis_df['std_prediction'],
                     alpha=0.3)
    ax1.set_title('Average Predictions Over Time')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Average Predicted Return')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Prediction spread
    ax2 = axes[0, 1]
    ax2.plot(analysis_df['date'], analysis_df['prediction_spread'], marker='o', color='orange')
    ax2.set_title('Prediction Spread (Max - Min)')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Spread')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Number of predictions
    ax3 = axes[1, 0]
    ax3.bar(analysis_df['date'], analysis_df['n_predictions'], alpha=0.7, color='green')
    ax3.set_title('Number of Predictions per Rebalance')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Count')
    
    # Plot 4: Prediction distribution for last date
    ax4 = axes[1, 1]
    last_date = list(ml_predictions.keys())[-1]
    last_predictions = ml_predictions[last_date]
    ax4.hist(last_predictions, bins=20, alpha=0.7, color='purple', edgecolor='black')
    ax4.set_title(f'Prediction Distribution ({last_date.date()})')
    ax4.set_xlabel('Predicted Return')
    ax4.set_ylabel('Count')
    
    plt.tight_layout()
    plt.savefig('results/ml_predictions_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ ML predictions analysis saved to results/ml_predictions_analysis.png")
    
    # Print summary statistics
    print("\n📊 ML Predictions Summary:")
    print(f"Average prediction across all dates: {analysis_df['avg_prediction'].mean():.4f}")
    print(f"Average number of stocks selected: {analysis_df['n_selected'].mean():.1f}")
    print(f"Prediction consistency (avg std): {analysis_df['std_prediction'].mean():.4f}")

def plot_predictions_diagnostics(ml_predictions: Dict[pd.Timestamp, pd.Series], 
                                returns_data: pd.DataFrame,
                                prediction_horizon_days: int = 63):
    """Generate comprehensive diagnostics for ML predictions.
    
    Args:
        ml_predictions: Dictionary of predictions by date
        returns_data: DataFrame of asset returns
        prediction_horizon_days: Forecast horizon in days
    """
    # Prepare data for analysis
    analysis_data = []
    
    # Get sorted prediction dates
    pred_dates = sorted(ml_predictions.keys())
    
    for i, pred_date in enumerate(pred_dates[:-1]):  # Skip last date (no realized returns)
        predictions = ml_predictions[pred_date]
        
        # Calculate realized returns
        start_date = pred_date
        end_date = pred_date + pd.Timedelta(days=prediction_horizon_days)
        
        # Find the actual trading days
        future_dates = returns_data.index[(returns_data.index > start_date) & 
                                         (returns_data.index <= end_date)]
        
        if len(future_dates) < 20:  # Need enough days for meaningful returns
            continue
            
        # Calculate realized returns for predicted assets
        realized_returns = {}
        for ticker in predictions.index:
            if ticker in returns_data.columns:
                ticker_returns = returns_data.loc[future_dates, ticker]
                # Compound return over the period
                realized_return = (1 + ticker_returns).prod() - 1
                realized_returns[ticker] = realized_return
        
        if realized_returns:
            realized_series = pd.Series(realized_returns)
            # Align predictions and realized
            common_tickers = predictions.index.intersection(realized_series.index)
            
            for ticker in common_tickers:
                analysis_data.append({
                    'date': pred_date,
                    'ticker': ticker,
                    'predicted': predictions[ticker],
                    'realized': realized_series[ticker]
                })
    
    if not analysis_data:
        print("⚠️  Not enough data for prediction diagnostics")
        return
        
    analysis_df = pd.DataFrame(analysis_data)
    
    # Create figure with 4 panels
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('ML Predictions Diagnostic Dashboard', fontsize=18, fontweight='bold')
    
    # Panel 1: Scatter plot of predicted vs realized returns
    ax1 = axes[0, 0]
    
    # Color by date (newer = darker)
    dates_numeric = pd.to_numeric(analysis_df['date'])
    colors = plt.cm.viridis((dates_numeric - dates_numeric.min()) / (dates_numeric.max() - dates_numeric.min()))
    
    scatter = ax1.scatter(analysis_df['predicted'], analysis_df['realized'], 
                         alpha=0.6, c=colors, s=30)
    
    # Add regression line
    from scipy import stats
    slope, intercept, r_value, p_value, std_err = stats.linregress(analysis_df['predicted'], analysis_df['realized'])
    x_line = np.linspace(analysis_df['predicted'].min(), analysis_df['predicted'].max(), 100)
    ax1.plot(x_line, slope * x_line + intercept, 'r--', alpha=0.8, 
             label=f'R² = {r_value**2:.3f}, p = {p_value:.3f}')
    
    # Add 45-degree line
    ax1.plot([analysis_df['predicted'].min(), analysis_df['predicted'].max()], 
             [analysis_df['predicted'].min(), analysis_df['predicted'].max()], 
             'k:', alpha=0.5, label='Perfect prediction')
    
    ax1.set_xlabel('Predicted Return', fontsize=12)
    ax1.set_ylabel('Realized Return', fontsize=12)
    ax1.set_title('Predicted vs Realized Returns', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add colorbar for dates
    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label('Date (newer →)', fontsize=10)
    
    # Panel 2: Rolling Information Coefficient
    ax2 = axes[0, 1]
    
    # Calculate IC for each prediction date
    ic_data = []
    for date in pred_dates[:-1]:
        date_data = analysis_df[analysis_df['date'] == date]
        if len(date_data) >= 10:  # Need minimum observations
            ic = date_data['predicted'].corr(date_data['realized'])
            ic_data.append({'date': date, 'ic': ic, 'n_stocks': len(date_data)})
    
    if ic_data:
        ic_df = pd.DataFrame(ic_data)
        
        # Plot IC over time
        ax2.plot(ic_df['date'], ic_df['ic'], 'o-', markersize=8, linewidth=2)
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
        # Add rolling average
        if len(ic_df) >= 4:
            ic_df['ic_ma'] = ic_df['ic'].rolling(4, min_periods=1).mean()
            ax2.plot(ic_df['date'], ic_df['ic_ma'], 'r-', linewidth=2, alpha=0.7, 
                    label=f'4-quarter MA (avg: {ic_df["ic_ma"].mean():.3f})')
        
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_ylabel('Information Coefficient', fontsize=12)
        ax2.set_title('Prediction Skill Over Time (IC)', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Rotate x-axis labels
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Panel 3: Inter-quartile range of predictions
    ax3 = axes[1, 0]
    
    iqr_data = []
    for date, predictions in ml_predictions.items():
        if len(predictions) >= 10:
            q75, q25 = np.percentile(predictions, [75, 25])
            iqr_data.append({
                'date': date,
                'iqr': q75 - q25,
                'q25': q25,
                'q75': q75,
                'median': predictions.median()
            })
    
    if iqr_data:
        iqr_df = pd.DataFrame(iqr_data)
        
        # Plot IQR over time
        ax3.fill_between(iqr_df['date'], iqr_df['q25'], iqr_df['q75'], 
                        alpha=0.3, color='blue', label='25th-75th percentile')
        ax3.plot(iqr_df['date'], iqr_df['median'], 'b-', linewidth=2, label='Median prediction')
        ax3.plot(iqr_df['date'], iqr_df['iqr'], 'r--', linewidth=2, label='IQR spread')
        
        ax3.set_xlabel('Date', fontsize=12)
        ax3.set_ylabel('Predicted Return', fontsize=12)
        ax3.set_title('Prediction Distribution Over Time', fontsize=14)
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # Rotate x-axis labels
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Panel 4: Rank autocorrelation heatmap
    ax4 = axes[1, 1]
    
    # Calculate rank autocorrelations
    rank_corrs = []
    for lag in range(1, min(4, len(pred_dates))):
        lag_corrs = []
        
        for i in range(len(pred_dates) - lag):
            date1 = pred_dates[i]
            date2 = pred_dates[i + lag]
            
            # Get predictions for both dates
            pred1 = ml_predictions[date1]
            pred2 = ml_predictions[date2]
            
            # Find common tickers
            common = pred1.index.intersection(pred2.index)
            if len(common) >= 20:
                # Calculate rank correlation
                rank1 = pred1[common].rank()
                rank2 = pred2[common].rank()
                corr = rank1.corr(rank2, method='spearman')
                lag_corrs.append(corr)
        
        if lag_corrs:
            rank_corrs.append({
                'lag': lag,
                'mean_corr': np.mean(lag_corrs),
                'std_corr': np.std(lag_corrs),
                'n_obs': len(lag_corrs)
            })
    
    if rank_corrs:
        rank_df = pd.DataFrame(rank_corrs)
        
        # Create bar plot with error bars
        bars = ax4.bar(rank_df['lag'], rank_df['mean_corr'], 
                       yerr=rank_df['std_corr'], capsize=5, 
                       color='darkblue', alpha=0.7)
        
        # Add value labels
        for bar, mean, std in zip(bars, rank_df['mean_corr'], rank_df['std_corr']):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{mean:.3f}', ha='center', va='bottom')
        
        ax4.set_xlabel('Lag (quarters)', fontsize=12)
        ax4.set_ylabel('Rank Correlation', fontsize=12)
        ax4.set_title('Prediction Rank Stability', fontsize=14)
        ax4.set_xticks(rank_df['lag'])
        ax4.grid(True, alpha=0.3, axis='y')
        ax4.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('results/ml_predictions_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ ML predictions diagnostics saved to results/ml_predictions_analysis.png")
    
    # Print summary statistics
    if 'ic_df' in locals():
        print(f"\n📊 Prediction Diagnostics Summary:")
        print(f"Average IC: {ic_df['ic'].mean():.3f} (std: {ic_df['ic'].std():.3f})")
        print(f"IC range: [{ic_df['ic'].min():.3f}, {ic_df['ic'].max():.3f}]")
        print(f"Positive IC rate: {(ic_df['ic'] > 0).mean():.1%}")
    
    if rank_corrs:
        print(f"\nRank stability (1-quarter): {rank_corrs[0]['mean_corr']:.3f}")


def create_performance_report(results: Dict, baseline_results: Optional[Dict] = None):
    """Create a comprehensive performance report.
    
    Args:
        results: ML-enhanced backtest results
        baseline_results: Optional baseline results for comparison
    """
    report = []
    report.append("# ML-Enhanced CLEIR Performance Report\n")
    report.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # ML Strategy Performance
    report.append("## ML-Enhanced Strategy Performance\n")
    report.append(f"- Total Return: {results['total_return']:.2%}")
    report.append(f"- Annual Return: {results['annual_return']:.2%}")
    report.append(f"- Volatility: {results['volatility']:.2%}")
    report.append(f"- Sharpe Ratio: {results['sharpe_ratio']:.3f}")
    report.append(f"- Max Drawdown: {results['max_drawdown']:.2%}")
    
    # Comparison with baseline
    if baseline_results:
        report.append("\n## Performance Comparison\n")
        report.append("| Metric | Baseline | ML-Enhanced | Improvement |")
        report.append("|--------|----------|-------------|-------------|")
        
        metrics = ['total_return', 'annual_return', 'volatility', 'sharpe_ratio', 'max_drawdown']
        for metric in metrics:
            baseline_val = baseline_results.get(metric, 0)
            ml_val = results.get(metric, 0)
            if metric in ['total_return', 'annual_return', 'volatility', 'max_drawdown']:
                improvement = ml_val - baseline_val
                report.append(f"| {metric.replace('_', ' ').title()} | {baseline_val:.2%} | {ml_val:.2%} | {improvement:+.2%} |")
            else:
                improvement = ml_val - baseline_val
                report.append(f"| {metric.replace('_', ' ').title()} | {baseline_val:.3f} | {ml_val:.3f} | {improvement:+.3f} |")
    
    # Save report
    with open('results/ml_performance_report.md', 'w') as f:
        f.write('\n'.join(report))
    
    print("✅ Performance report saved to results/ml_performance_report.md")