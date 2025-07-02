import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import os
import json
from datetime import datetime

from src.backtesting.alpha_engine import AlphaEnhancedBacktest
from src.analysis.simple_interpretability import (
    plot_feature_importance, 
    plot_performance_comparison,
    analyze_ml_predictions,
    create_performance_report,
    plot_shap_analysis,
    plot_predictions_diagnostics
)

def calculate_performance_metrics(daily_values: pd.Series) -> dict:
    """Calculate standard performance metrics."""
    returns = daily_values.pct_change().dropna()
    
    metrics = {
        'total_return': (daily_values.iloc[-1] / daily_values.iloc[0]) - 1,
        'annual_return': ((daily_values.iloc[-1] / daily_values.iloc[0]) ** (252 / len(daily_values))) - 1,
        'volatility': returns.std() * np.sqrt(252),
        'sharpe_ratio': (returns.mean() * 252) / (returns.std() * np.sqrt(252)) if returns.std() > 0 else 0,
        'max_drawdown': (daily_values / daily_values.expanding().max() - 1).min()
    }
    
    return metrics

def main():
    """Main execution function."""
    print("=" * 60)
    print("🚀 Running Simple ML-Enhanced CLEIR Backtest")
    print("=" * 60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. Run ML-enhanced backtest
    print("\n📊 Starting ML-enhanced backtest...")
    backtest = AlphaEnhancedBacktest(top_k=30)  # Select top 30 stocks
    
    try:
        ml_results = backtest.run(start_date='2020-01-01', end_date='2024-12-31')
    except Exception as e:
        print(f"❌ Error running backtest: {str(e)}")
        print("Please ensure data is available in data/processed/price_data.pkl")
        return
    
    # Extract daily values
    ml_daily_values = ml_results['daily_values']
    
    # Save ML results
    print("\n💾 Saving ML results...")
    ml_daily_values.to_csv('results/ml_enhanced_index.csv')
    print("✅ ML index values saved to results/ml_enhanced_index.csv")
    
    # 2. Load baseline for comparison (if available)
    baseline_results = None
    baseline = None
    
    baseline_files = ['results/cleir_index_gui.csv', 'results/cvar_index_gui.csv']
    for baseline_file in baseline_files:
        if os.path.exists(baseline_file):
            try:
                baseline = pd.read_csv(baseline_file, index_col=0, parse_dates=True).squeeze()
                print(f"✅ Loaded baseline from {baseline_file}")
                baseline_results = calculate_performance_metrics(baseline)
                break
            except Exception as e:
                print(f"⚠️  Error loading {baseline_file}: {str(e)}")
    
    if baseline is None:
        print("⚠️  No baseline results found. Skipping comparison.")
    
    # 3. Calculate and print performance metrics
    ml_metrics = ml_results.copy()
    
    # Save metrics to JSON for GUI consumption
    metrics_for_json = {
        'ml_metrics': {
            'total_return': float(ml_metrics['total_return']),
            'annual_return': float(ml_metrics['annual_return']),
            'volatility': float(ml_metrics['volatility']),
            'sharpe_ratio': float(ml_metrics['sharpe_ratio']),
            'max_drawdown': float(ml_metrics['max_drawdown']),
            'avg_turnover': float(ml_metrics.get('avg_turnover', 0.0)),
            'total_transaction_costs': float(ml_metrics.get('total_transaction_costs', 0.0))
        }
    }
    
    with open('results/ml_metrics.json', 'w') as f:
        json.dump(metrics_for_json, f, indent=2)
    
    print("\n" + "=" * 60)
    print("📈 ML-Enhanced Performance Summary")
    print("=" * 60)
    print(f"Total Return:    {ml_metrics['total_return']:>10.2%}")
    print(f"Annual Return:   {ml_metrics['annual_return']:>10.2%}")
    print(f"Volatility:      {ml_metrics['volatility']:>10.2%}")
    print(f"Sharpe Ratio:    {ml_metrics['sharpe_ratio']:>10.3f}")
    print(f"Max Drawdown:    {ml_metrics['max_drawdown']:>10.2%}")
    print(f"Avg Turnover:    {ml_metrics.get('avg_turnover', 0):>10.2%}")
    print(f"Total TC:        {ml_metrics.get('total_transaction_costs', 0):>10.2%}")
    
    if baseline_results is not None:
        print("\n" + "=" * 60)
        print("📊 Baseline Performance Summary")
        print("=" * 60)
        print(f"Total Return:    {baseline_results['total_return']:>10.2%}")
        print(f"Annual Return:   {baseline_results['annual_return']:>10.2%}")
        print(f"Volatility:      {baseline_results['volatility']:>10.2%}")
        print(f"Sharpe Ratio:    {baseline_results['sharpe_ratio']:>10.3f}")
        print(f"Max Drawdown:    {baseline_results['max_drawdown']:>10.2%}")
        
        print("\n" + "=" * 60)
        print("🎯 Performance Improvement")
        print("=" * 60)
        sharpe_improvement = ((ml_metrics['sharpe_ratio'] / baseline_results['sharpe_ratio']) - 1) * 100
        print(f"Sharpe Ratio Improvement: {sharpe_improvement:>8.1f}%")
        return_improvement = ml_metrics['total_return'] - baseline_results['total_return']
        print(f"Total Return Improvement: {return_improvement:>8.2%}")
    
    # 4. Generate visualizations
    print("\n📊 Generating visualizations...")
    
    # Feature importance plot
    if hasattr(backtest, 'trainer') and backtest.trainer.models:
        plot_feature_importance(backtest.trainer)
        
        # SHAP analysis
        plot_shap_analysis(backtest.trainer)
    
    # Performance comparison plot
    if baseline is not None:
        plot_performance_comparison(baseline, ml_daily_values)
    
    # ML predictions analysis
    if 'ml_predictions' in ml_results and 'selected_universes' in ml_results:
        analyze_ml_predictions(ml_results['ml_predictions'], ml_results['selected_universes'])
        
        # New comprehensive diagnostics
        # Load returns data for diagnostics
        try:
            import pickle
            data_path = 'data/processed/price_data.pkl'
            if os.path.exists(data_path):
                with open(data_path, 'rb') as f:
                    data_dict = pickle.load(f)
                
                # Convert to DataFrame if needed
                if isinstance(data_dict, dict) and 'prices' in data_dict:
                    price_df = pd.DataFrame(
                        data_dict['prices'],
                        index=pd.to_datetime(data_dict['dates']),
                        columns=data_dict['tickers']
                    )
                else:
                    price_df = data_dict
                
                returns_data = price_df.pct_change().dropna()
                plot_predictions_diagnostics(ml_results['ml_predictions'], returns_data)
        except Exception as e:
            print(f"⚠️  Could not generate prediction diagnostics: {str(e)}")
    
    # 5. Generate performance report
    print("\n📄 Generating performance report...")
    create_performance_report(ml_metrics, baseline_results)
    
    # 6. Save detailed results
    print("\n💾 Saving detailed results...")
    
    # Save portfolio weights history
    if 'portfolio_weights' in ml_results:
        weights_df = pd.DataFrame(ml_results['portfolio_weights']).T.fillna(0)
        weights_df.to_csv('results/ml_portfolio_weights.csv')
        print("✅ Portfolio weights saved to results/ml_portfolio_weights.csv")
    
    # Save selected universes
    if 'selected_universes' in ml_results:
        with open('results/ml_selected_universes.txt', 'w') as f:
            for date, tickers in ml_results['selected_universes'].items():
                f.write(f"{date.date()}: {', '.join(tickers[:10])}...\n")
        print("✅ Selected universes saved to results/ml_selected_universes.txt")
    
    print("\n" + "=" * 60)
    print("✅ ML-Enhanced CLEIR Backtest Complete!")
    print("=" * 60)
    print("\nGenerated files:")
    print("  📊 results/ml_enhanced_index.csv - Daily index values")
    print("  📊 results/ml_feature_importance.png - Feature importance plot")
    if baseline is not None:
        print("  📊 results/ml_performance_comparison.png - Performance comparison")
    print("  📊 results/ml_predictions_analysis.png - ML predictions analysis")
    print("  📄 results/ml_performance_report.md - Detailed report")
    print("  📊 results/ml_portfolio_weights.csv - Portfolio weights history")
    print("  📄 results/ml_selected_universes.txt - Selected stocks by date")
    
    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()