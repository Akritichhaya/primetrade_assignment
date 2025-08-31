ML approach:
- Target: 1 if closed_pnl > 0 else 0 (trade win).
- Features: sentiment (Fear/Greed), side (Buy/Sell), coin, leverage, size, execution price, start position (when present).
- Models: Logistic Regression (interpretable baseline) and Random Forest (nonlinear baseline).
- Metrics: Accuracy and ROC-AUC on a 25% holdout set.