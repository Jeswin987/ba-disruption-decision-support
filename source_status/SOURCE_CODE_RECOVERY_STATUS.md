# v0.3 source-code recovery status

## Recovered exactly
- Frozen v0.3 specification: synthetic_spec_v03_frozen.md.
- Saved v0.3 data files.
- Saved v0.3 result tables and operational HGB queue.
- Follow-up Logistic refit output and README documenting: 10 operational features, class_weight="balanced", event-level split with test_size=0.25 and random_state=42, per-event floor(n*capacity) with minimum 1, stable descending sort.

## Not recovered
- Original v0.3 synthetic data-generation Python source. The frozen specification documents distributions/mechanisms but the generator source/formula is not present in the saved artifacts.
- Original v0.3 model-training Python source for Logistic Regression, Random Forest, and HistGradientBoostingClassifier.
- Exact HGB hyperparameters beyond the model family and feature list.
- Exact Random Forest hyperparameters and exact class-weight/resampling configuration.
- Original calibration-training Python source for v0.3 HGB/RF/Logistic. The saved calibration tables/results exist.
- Original simple-rule implementation source; only aggregate results/documentation remain.
- Row-level held-out test predictions for Logistic Regression and Random Forest.
- Row-level held-out test predictions for the simple rule.
- Bootstrap code/source for HGB/simple paired confidence intervals.
- Oracle/latent-probability generator source or saved latent probabilities.

## Important
No missing source above has been silently reconstructed in this package. Where the project record gives a documented configuration, that documentation is included, but it is not represented as the original source code.
