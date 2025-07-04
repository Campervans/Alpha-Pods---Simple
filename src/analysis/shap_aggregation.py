"""Aggregated SHAP analysis across all models."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


def plot_aggregated_shap_analysis(trainer, num_samples: int = 100):
    """Generate aggregated SHAP analysis across all models.
    
    Gives a big picture view of feature importance, not just for one model.
    
    Args:
        trainer: SimpleWalkForward instance with trained models.
        num_samples: how many samples to use per model for SHAP.
    """
    if not SHAP_AVAILABLE:
        print("⚠️  SHAP not installed. Skipping SHAP analysis.")
        return
    
    # get the most recent models
    latest_date = max([date for date, _ in trainer.models.keys()])
    latest_models = [(ticker, model) for (date, ticker), model in trainer.models.items() 
                     if date == latest_date and hasattr(model, 'X_train_') and model.X_train_ is not None]
    
    if not latest_models:
        print("No models with training data found.")
        return
    
    print(f"Aggregating SHAP values for {len(latest_models)} models...")
    
    # collect SHAP values
    all_shap_values = []
    all_feature_values = []
    ticker_shap_importance = {}
    
    for ticker, model in latest_models:
        try:
            # smaller background for speed
            n_background = min(500, len(model.X_train_))
            n_sample = min(100, len(model.X_train_))
            
            X_background = model.X_train_.sample(n=n_background, random_state=42)
            X_sample = model.X_train_.tail(n_sample)
            
            # scale data
            X_background_scaled = model.scaler.transform(X_background)
            X_sample_scaled = model.scaler.transform(X_sample)
            
            # create SHAP explainer
            # TODO: could probably use a different explainer for non-linear models
            explainer = shap.LinearExplainer(model.model, X_background_scaled, 
                                            feature_names=model.feature_names)
            shap_values = explainer.shap_values(X_sample_scaled)
            
            # store values
            all_shap_values.append(shap_values)
            all_feature_values.append(X_sample)
            
            # calc importance for this ticker
            ticker_shap_importance[ticker] = pd.Series(
                np.abs(shap_values).mean(axis=0), 
                index=model.feature_names
            )
            
        except Exception as e:
            print(f"  Warning: SHAP failed for {ticker}: {e}")
    
    if not all_shap_values:
        print("No SHAP values could be computed.")
        return
    
    # put all shap values together
    combined_shap = np.vstack(all_shap_values)
    combined_features = pd.concat(all_feature_values, ignore_index=True)
    
    # calc global feature importance
    feature_names = latest_models[0][1].feature_names
    global_shap_importance = pd.Series(
        np.abs(combined_shap).mean(axis=0),
        index=feature_names
    ).sort_values(ascending=False)
    
    # make a big plot
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle('Aggregated SHAP Analysis', fontsize=22, fontweight='bold')
    
    # Create grid with increased spacing
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1.2, 1], hspace=0.5, wspace=0.4)
    
    # 1. global feature importance
    ax1 = fig.add_subplot(gs[0, :2])
    colors = sns.color_palette("husl", len(global_shap_importance))
    bars = ax1.bar(range(len(global_shap_importance)), global_shap_importance.values, color=colors)
    ax1.set_xticks(range(len(global_shap_importance)))
    ax1.set_xticklabels(global_shap_importance.index, rotation=45, ha='right')
    ax1.set_ylabel('Mean |SHAP value|', fontsize=12)
    ax1.set_title(f'Global Feature Importance (avg over {len(latest_models)} models)', fontsize=14)
    
    # add value labels to bars
    for bar, val in zip(bars, global_shap_importance.values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 2. feature importance variance
    ax2 = fig.add_subplot(gs[0, 2])
    
    # calc variance of importance across tickers
    importance_df = pd.DataFrame(ticker_shap_importance).T
    importance_std = importance_df.std().sort_values(ascending=False)
    
    ax2.barh(range(len(importance_std)), importance_std.values, color='coral')
    ax2.set_yticks(range(len(importance_std)))
    ax2.set_yticklabels(importance_std.index)
    ax2.set_xlabel('Std Dev of Importance', fontsize=10)
    ax2.set_title('Feature Importance Variability\n(high variance = varies by stock)', fontsize=12)
    
    # 3. aggregated beeswarm plot
    ax3 = fig.add_subplot(gs[1, :])
    
    # sample for viz, otherwise too many points
    sample_idx = np.random.choice(len(combined_shap), min(1000, len(combined_shap)), replace=False)
    shap.summary_plot(combined_shap[sample_idx], combined_features.iloc[sample_idx], 
                      feature_names=feature_names, show=False, plot_size=None)
    plt.sca(ax3)
    ax3.set_title(f'SHAP Value Distribution ({len(combined_shap)} predictions)', fontsize=14)
    
    # 4. top stocks by top feature
    ax4 = fig.add_subplot(gs[2, 0])
    
    # which stocks rely most on the top feature?
    top_feature = global_shap_importance.index[0]
    feature_reliance = {ticker: imp[top_feature] for ticker, imp in ticker_shap_importance.items()}
    top_reliant = pd.Series(feature_reliance).nlargest(10)
    
    ax4.barh(range(len(top_reliant)), top_reliant.values, color='lightblue')
    ax4.set_yticks(range(len(top_reliant)))
    ax4.set_yticklabels(top_reliant.index)
    ax4.set_xlabel(f'Importance of {top_feature}', fontsize=10)
    ax4.set_title(f'Top 10 Stocks Relying on\n{top_feature}', fontsize=12)
    
    # 5. feature correlation heatmap
    ax5 = fig.add_subplot(gs[2, 1])
    
    # correlation between feature importances
    if importance_df.shape[0] >= 5:  # need enough stocks for this
        corr_matrix = importance_df.corr()
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                    square=True, ax=ax5, cbar_kws={'shrink': 0.8})
        ax5.set_title('Feature Importance Correlations', fontsize=12)
    
    # 6. model count by feature significance
    ax6 = fig.add_subplot(gs[2, 2])
    
    # how many models have this feature in their top 3?
    feature_counts = {}
    for ticker, imp in ticker_shap_importance.items():
        top_3 = imp.nlargest(3).index
        for feat in top_3:
            feature_counts[feat] = feature_counts.get(feat, 0) + 1
    
    count_series = pd.Series(feature_counts).sort_values(ascending=True)
    ax6.barh(range(len(count_series)), count_series.values, color='lightgreen')
    ax6.set_yticks(range(len(count_series)))
    ax6.set_yticklabels(count_series.index)
    ax6.set_xlabel('# Models (in Top 3)', fontsize=10)
    ax6.set_title('Feature Prevalence', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('results/ml_shap_analysis_aggregated.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # also save individual model analysis
    _save_individual_shap_summary(ticker_shap_importance)
    
    print("✅ Aggregated SHAP analysis saved to results/ml_shap_analysis_aggregated.png")
    print(f"✅ Analyzed {len(latest_models)} models, {len(combined_shap)} total predictions")


def _save_individual_shap_summary(ticker_shap_importance: Dict[str, pd.Series]):
    """Save a text summary of individual model SHAP importances."""
    
    # to DataFrame
    importance_df = pd.DataFrame(ticker_shap_importance).T
    
    # save top features per stock
    summary_lines = ["# Individual Model SHAP Feature Importance\n\n"]
    summary_lines.append(f"Analysis of {len(importance_df)} models\n\n")
    
    # overall top features
    summary_lines.append("## Top Features Globally\n")
    global_avg = importance_df.mean().sort_values(ascending=False)
    for i, (feat, val) in enumerate(global_avg.items(), 1):
        summary_lines.append(f"{i}. {feat}: {val:.4f}\n")
    
    # stocks with weird patterns
    summary_lines.append("\n## Stocks with Unique Feature Preferences\n")
    
    for ticker in importance_df.index[:10]:  # just show 10 examples
        stock_imp = importance_df.loc[ticker].sort_values(ascending=False)
        top_feat = stock_imp.index[0]
        
        # is this stock's top feature different from the global top?
        if top_feat != global_avg.index[0]:
            summary_lines.append(f"\n**{ticker}**: Prefers {top_feat} ({stock_imp[top_feat]:.3f})")
            summary_lines.append(f" vs global top {global_avg.index[0]}\n")
    
    with open('results/ml_shap_individual_summary.txt', 'w') as f:
        f.writelines(summary_lines)
    
    print("✅ Individual SHAP summary saved to results/ml_shap_individual_summary.txt") 