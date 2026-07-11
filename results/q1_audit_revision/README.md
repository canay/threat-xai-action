# Q1 Audit Revision Outputs

This directory contains lightweight revision checks added after the manuscript audit.

- `simple_context_baselines.csv`: majority and context-majority baselines computed from the controlled processed dataset.
- `xgb_fixed_model_summary.csv`: metrics from one fitted XGBoost model per split and feature setting.
- `xgb_fixed_model_bootstrap_ci.csv`: conditional intervals obtained by resampling each fitted model's fixed test predictions without refitting inside the bootstrap.
- `xgb_fixed_model_per_class_recall_ci.csv`: class-conditional recall intervals from the same fixed-prediction resampling scope.
- `q1_audit_revision_metadata.json`: seed, labels, and uncertainty-scope notes.
- `classweight_sensitivity.csv`: selected-model sensitivity to inverse-frequency sample weights.
- `sourceport_nearduplicate_ablation.csv`: grouped-split sensitivity after removing source port from the grouping signature.
- `forward_chaining_chronological.csv`: expanding-window chronological folds.
- `repeated_seed_results.csv` and `repeated_seed_summary.csv`: ten-seed selected-model checks.
- `category_code_order_sensitivity.csv`, `category_code_order_sensitivity_summary.csv`, and `category_code_order_sensitivity_metadata.json`: five category-code-order permutations.
- `full_feature_lookup_baseline.csv` and `full_feature_lookup_baseline_metadata.json`: aggregate-only exact-signature lookup baselines, including the full 20-feature comparator and reproduction controls.

Bootstrap iterations do not refit models. The script can also reconstruct the same fixed-prediction scope from aggregate confusion cells in `--confusion-only` mode, which does not require event-level data.
All other files in this directory are aggregate outputs; rerunning their scripts requires the controlled processed dataset described in the repository README.
