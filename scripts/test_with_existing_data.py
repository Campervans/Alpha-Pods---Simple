"""
Test script using existing daily index values data.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def analyze_existing_data():
    """Analyze the existing daily index values data."""
    print("=" * 60)
    print("ANALYZING EXISTING CVAR INDEX DATA")
    print("=" * 60)
    
    # Load the data
    data_path = '/Users/james/Alpha-Pods---Simple/results/daily_index_values.csv'
    df = pd.read_csv(data_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    
    print(f"\nData loaded successfully!")
    print(f"Date range: {df.index[0].date()} to {df.index[-1].date()}")
    print(f"Total trading days: {len(df)}")
    
    # Calculate performance metrics
    print("\n" + "="*50)
    print("PERFORMANCE METRICS")
    print("="*50)
    
    # Total return
    total_return = df['Cumulative_Return'].iloc[-1]
    print(f"Total Return: {total_return:.2%}")
    
    # Annualized return
    years = (df.index[-1] - df.index[0]).days / 365.25
    annual_return = (1 + total_return) ** (1/years) - 1
    print(f"Annualized Return: {annual_return:.2%}")
    
    # Volatility
    daily_returns = df['Daily_Return']
    annual_volatility = daily_returns.std() * np.sqrt(252)
    print(f"Annual Volatility: {annual_volatility:.2%}")
    
    # Sharpe ratio
    sharpe_ratio = annual_return / annual_volatility
    print(f"Sharpe Ratio: {sharpe_ratio:.3f}")
    
    # Maximum drawdown
    cumulative = (1 + daily_returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    print(f"Maximum Drawdown: {max_drawdown:.2%}")
    
    # CVaR 95%
    var_95 = daily_returns.quantile(0.05)
    cvar_95 = daily_returns[daily_returns <= var_95].mean()
    print(f"Daily VaR 95%: {var_95:.2%}")
    print(f"Daily CVaR 95%: {cvar_95:.2%}")
    
    # Create visualizations
    print("\n" + "="*50)
    print("CREATING VISUALIZATIONS")
    print("="*50)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('CVaR Index Performance Analysis', fontsize=16)
    
    # 1. Index value over time
    ax1 = axes[0, 0]
    ax1.plot(df.index, df['Index_Value'], linewidth=1.5, color='navy')
    ax1.set_title('CVaR Index Value Over Time')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Index Value')
    ax1.grid(True, alpha=0.3)
    
    # 2. Daily returns distribution
    ax2 = axes[0, 1]
    daily_returns.hist(bins=100, ax=ax2, color='darkgreen', alpha=0.7, edgecolor='black')
    ax2.axvline(var_95, color='red', linestyle='--', label=f'VaR 95%: {var_95:.2%}')
    ax2.axvline(cvar_95, color='darkred', linestyle='--', label=f'CVaR 95%: {cvar_95:.2%}')
    ax2.set_title('Daily Returns Distribution')
    ax2.set_xlabel('Daily Return')
    ax2.set_ylabel('Frequency')
    ax2.legend()
    
    # 3. Rolling volatility
    ax3 = axes[1, 0]
    rolling_vol = daily_returns.rolling(252).std() * np.sqrt(252)
    ax3.plot(rolling_vol.index, rolling_vol, linewidth=1.5, color='orange')
    ax3.set_title('Rolling 1-Year Volatility')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Annual Volatility')
    ax3.grid(True, alpha=0.3)
    
    # 4. Drawdown chart
    ax4 = axes[1, 1]
    ax4.fill_between(drawdown.index, drawdown, 0, color='red', alpha=0.3)
    ax4.plot(drawdown.index, drawdown, linewidth=1, color='darkred')
    ax4.set_title('Drawdown Over Time')
    ax4.set_xlabel('Date')
    ax4.set_ylabel('Drawdown')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    output_path = 'results/index_performance_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")
    
    # Annual returns table
    print("\n" + "="*50)
    print("ANNUAL RETURNS")
    print("="*50)
    
    # Calculate annual returns
    df['Year'] = df.index.year
    annual_returns = df.groupby('Year')['Daily_Return'].apply(lambda x: (1 + x).prod() - 1)
    
    for year, ret in annual_returns.items():
        print(f"{year}: {ret:7.2%}")
    
    # Recent performance
    print("\n" + "="*50)
    print("RECENT PERFORMANCE (Last 30 days)")
    print("="*50)
    
    recent_data = df.iloc[-30:]
    recent_return = (recent_data['Index_Value'].iloc[-1] / recent_data['Index_Value'].iloc[0]) - 1
    recent_vol = recent_data['Daily_Return'].std() * np.sqrt(252)
    
    print(f"30-day Return: {recent_return:.2%}")
    print(f"30-day Volatility (annualized): {recent_vol:.2%}")
    
    return df

def test_risk_metrics(df):
    """Test various risk metric calculations."""
    print("\n" + "="*60)
    print("TESTING RISK METRICS")
    print("="*60)
    
    daily_returns = df['Daily_Return']
    
    # Test different confidence levels
    confidence_levels = [0.90, 0.95, 0.99]
    
    print("\nValue at Risk (VaR) and Conditional Value at Risk (CVaR):")
    print("-" * 50)
    
    for conf in confidence_levels:
        alpha = 1 - conf
        var = daily_returns.quantile(alpha)
        cvar = daily_returns[daily_returns <= var].mean()
        
        print(f"\nConfidence Level: {conf:.0%}")
        print(f"  Daily VaR: {var:.3%}")
        print(f"  Daily CVaR: {cvar:.3%}")
        print(f"  Annual VaR: {var * np.sqrt(252):.2%}")
        print(f"  Annual CVaR: {cvar * np.sqrt(252):.2%}")
    
    # Tail risk analysis
    print("\n" + "-"*50)
    print("TAIL RISK ANALYSIS")
    print("-" * 50)
    
    # Worst days
    worst_days = daily_returns.nsmallest(10)
    print("\nWorst 10 Daily Returns:")
    for date, ret in worst_days.items():
        print(f"  {date.date()}: {ret:.2%}")
    
    # Best days
    best_days = daily_returns.nlargest(10)
    print("\nBest 10 Daily Returns:")
    for date, ret in best_days.items():
        print(f"  {date.date()}: {ret:.2%}")
    
    # Skewness and kurtosis
    from scipy import stats
    skewness = stats.skew(daily_returns.dropna())
    kurtosis = stats.kurtosis(daily_returns.dropna())
    
    print(f"\nSkewness: {skewness:.3f}")
    print(f"Kurtosis: {kurtosis:.3f}")
    
    # Debug info - leave commented
    """
    # Additional debug calculations
    print("\nDEBUG INFO:")
    print(f"First index value: {df['Index_Value'].iloc[0]}")
    print(f"Last index value: {df['Index_Value'].iloc[-1]}")
    print(f"Total days: {len(df)}")
    print(f"Days with positive returns: {(daily_returns > 0).sum()}")
    print(f"Days with negative returns: {(daily_returns < 0).sum()}")
    print(f"Days with zero returns: {(daily_returns == 0).sum()}")
    """

if __name__ == "__main__":
    print("CVaR INDEX DATA ANALYSIS")
    print("Using existing data from: /Users/james/Alpha-Pods---Simple/results/daily_index_values.csv")
    print()
    
    try:
        # Analyze the data
        df = analyze_existing_data()
        
        # Test risk metrics
        test_risk_metrics(df)
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETED SUCCESSFULLY!")
        print("="*60)
        
    except Exception as e:
        print(f"\nError during analysis: {e}")
        import traceback
        traceback.print_exc()
    
    # Leave debug commands commented
    """
    # Debug: Check if matplotlib backend is working
    # import matplotlib
    # print(f"Matplotlib backend: {matplotlib.get_backend()}")
    
    # Debug: Save a simple test plot
    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.plot([1, 2, 3], [1, 4, 9])
    # plt.savefig('test_plot.png')
    # print("Test plot saved")
    """ 