# SHAP & ML-Predictions Visualisation Fix Plan

> Goal: Replace the coefficient-proxy chart with true SHAP explanations and overhaul `ml_predictions_analysis.png` to emphasise predictive power.

---

## Phase 1 – Preparation

1. **Add SHAP to environment**  
   • Update `pyproject.toml` (or `environment.yml`) to include `shap>=0.44`.  
   • `pip install shap` locally and verify import.
2. **Data accessibility**  
   • Ensure training feature matrices (`X_train`) are cached or reproducible so SHAP explainers can access background data easily.  
   • Quick win: store the final `X_train` for each model inside `SimpleAlphaModel` after fitting (`self.X_train_`).

## Phase 2 – SHAP module refactor

3. **Refactor `plot_shap_analysis()`**  
   a. Replace coefficient logic with:
   ```python
   explainer = shap.LinearExplainer(model.model, X_background, feature_names=model.feature_names)
   shap_values = explainer.shap_values(X_sample)
   ```
   b. Build one figure (1600×1200):
      • Top: bar chart of mean |SHAP|.  
      • Middle: beeswarm.  
      • Bottom-left/right: dependence plots for top-2 features.  
      • Optional: add waterfall for last prediction.
   c. Save to `results/ml_shap_analysis.png` (overwrite).

4. **Background & sample selection utilities**  
   • `get_background_sample(X, n=1000)` – uniform random rows.  
   • `get_recent_sample(X, n=500)` – most recent rows before the rebalance.

## Phase 3 – Prediction diagnostics overhaul

5. **Create `plot_predictions_diagnostics()`** (new function in `simple_interpretability.py`):
   • Inputs: `ml_predictions`, `returns_data`, `prediction_horizon_days` (default 63).  
   • Panels:
     1. Scatter α vs realised return (colour-by-date).  
     2. Rolling Information Coefficient (Pearson corr, 4-quarter window).  
     3. Inter-quartile range of α over time.  
     4. Rank-autocorrelation heatmap (lag 1-3 quarters).
   • Save to `results/ml_predictions_analysis.png`.

6. **Optional factor back-test**  
   • Build a quick long-short decile portfolio inside diagnostics; plot cumulative factor return.

## Phase 4 – Integration

7. **Update `run_simple_ml_backtest.py` (or equivalent runner)**  
   • After back-test finishes, call the new SHAP and diagnostics functions.
8. **Document usage**  
   • Add README section: *"Interpreting ML Alphas with SHAP"* explaining outputs.

## Phase 5 – Validation

9. **Unit tests**  
   • Add tests to `tests/analysis/` verifying that `plot_shap_analysis` runs without error and outputs a file.  
   • Mock a small Ridge model & dataset to keep tests lightweight.
10. **Visual sanity check**  
    • Manually inspect the new PNGs; confirm that bars, beeswarm, dependence plots render.

---

**Estimated effort:** ~½ day for coding + ½ day for validation/documentation.

**Success criteria**  
• `ml_shap_analysis.png` shows genuine SHAP explanations.  
• `ml_predictions_analysis.png` reports predictive skill metrics (scatter, IC, spread, rank stability). 