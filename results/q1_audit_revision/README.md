# Q1 Audit Revision Outputs

This directory contains lightweight revision checks added after the manuscript audit.

- `simple_context_baselines.csv`: majority and context-majority baselines computed from the controlled processed dataset.
- `xgb_fixed_model_summary.csv`: fixed-model XGBoost metrics reconstructed from saved confusion-cell counts.
- `xgb_fixed_model_bootstrap_ci.csv`: bootstrap intervals over saved true/predicted confusion-cell counts.
- `xgb_fixed_model_per_class_recall_ci.csv`: class-conditional recall intervals from the same aggregate source.
- `q1_audit_revision_metadata.json`: seed, labels, and uncertainty-scope notes.

The bootstrap outputs do not refit models and do not require event-level data when run in `--confusion-only` mode.
