import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from typing import Dict, Optional

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

def plot_performance_comparison(baseline_perf: pd.Series, ml_perf: pd.Series, 
                               benchmark_perf: Optional[pd.Series] = None):
    """Plots the cumulative performance of baseline vs. ML-enhanced index.
    
    Args:
        baseline_perf: Series of baseline index values
        ml_perf: Series of ML-enhanced index values
        benchmark_perf: Optional series of benchmark (e.g., SPY) values
    """
    # Create comparison dataframe
    comparison_df = pd.DataFrame({
        'Baseline CVaR Index': baseline_perf,
        'ML-Enhanced Index': ml_perf
    })
    
    # Add benchmark if provided
    if benchmark_perf is not None:
        comparison_df['Benchmark (SPY)'] = benchmark_perf
    
    # Normalize to start at 1
    comparison_df_normalized = comparison_df / comparison_df.iloc[0]
    
    # Create the plot
    plt.figure(figsize=(14, 8))
    
    # Plot lines
    for col in comparison_df_normalized.columns:
        if col == 'ML-Enhanced Index':
            plt.plot(comparison_df_normalized.index, comparison_df_normalized[col], 
                    label=col, linewidth=2.5, color='green', alpha=0.8)
        elif col == 'Baseline CVaR Index':
            plt.plot(comparison_df_normalized.index, comparison_df_normalized[col], 
                    label=col, linewidth=2, color='blue', alpha=0.7)
        else:
            plt.plot(comparison_df_normalized.index, comparison_df_normalized[col], 
                    label=col, linewidth=1.5, color='gray', alpha=0.6, linestyle='--')
    
    plt.title('Performance Comparison: Baseline CVaR vs. ML-Enhanced', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Cumulative Growth of $1', fontsize=12)
    plt.legend(loc='upper left', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add performance stats
    baseline_return = (comparison_df_normalized['Baseline CVaR Index'].iloc[-1] - 1) * 100
    ml_return = (comparison_df_normalized['ML-Enhanced Index'].iloc[-1] - 1) * 100
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