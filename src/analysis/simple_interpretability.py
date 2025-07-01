import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from typing import Dict, Optional
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: SHAP not available. Install with: pip install shap")

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
    
    # Get the most recent model and its data
    latest_date = max([date for date, _ in trainer.models.keys()])
    latest_models = [(ticker, model) for (date, ticker), model in trainer.models.items() if date == latest_date]
    
    if not latest_models:
        print("No models available for SHAP analysis.")
        return
    
    # Use the first model for demonstration
    ticker, model = latest_models[0]
    
    # Get feature data for this model
    from src.features.simple_features import create_simple_features
    
    # We need to recreate the feature data for SHAP
    # This is a simplified version - in production you'd store this during training
    print(f"Generating SHAP values for {ticker} model...")
    
    # Create a figure with subplots
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle(f'SHAP Analysis for {ticker} (Latest Model)', fontsize=16, fontweight='bold')
    
    # For Ridge regression, we can calculate exact SHAP values
    # Get feature names from the model
    if hasattr(model, 'feature_names') and model.feature_names is not None:
        feature_names = model.feature_names
    elif hasattr(model.model, 'feature_names_in_'):
        feature_names = model.model.feature_names_in_
    else:
        # Default feature names based on simple_features.py
        feature_names = ['return_1m', 'return_3m', 'return_6m', 
                        'volatility_1m', 'volatility_3m', 'volume_ratio', 'rsi']
    
    # Plot 1: Feature importance based on absolute SHAP values (coefficients for linear model)
    ax1 = axes[0]
    coef_importance = pd.Series(np.abs(model.model.coef_), index=feature_names).sort_values(ascending=False)
    top_features = coef_importance.head(10)
    
    bars = ax1.barh(range(len(top_features)), top_features.values)
    ax1.set_yticks(range(len(top_features)))
    ax1.set_yticklabels(top_features.index)
    ax1.set_xlabel('Absolute Coefficient Value (Proxy for SHAP Importance)')
    ax1.set_title('Top 10 Most Important Features')
    
    # Color bars based on positive/negative impact
    for i, (feat, val) in enumerate(top_features.items()):
        idx = list(feature_names).index(feat)
        if model.model.coef_[idx] > 0:
            bars[i].set_color('red')
        else:
            bars[i].set_color('blue')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='red', label='Positive Impact'),
                      Patch(facecolor='blue', label='Negative Impact')]
    ax1.legend(handles=legend_elements, loc='lower right')
    
    # Plot 2: Coefficient values with direction
    ax2 = axes[1]
    coef_values = pd.Series(model.model.coef_, index=feature_names).sort_values(ascending=True)
    extreme_features = pd.concat([coef_values.head(5), coef_values.tail(5)])
    
    colors = ['blue' if x < 0 else 'red' for x in extreme_features.values]
    ax2.barh(range(len(extreme_features)), extreme_features.values, color=colors)
    ax2.set_yticks(range(len(extreme_features)))
    ax2.set_yticklabels(extreme_features.index)
    ax2.set_xlabel('Coefficient Value')
    ax2.set_title('Top 5 Positive and Negative Impact Features')
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig('results/ml_shap_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ SHAP analysis saved to results/ml_shap_analysis.png")

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