# Result Traceability Manifest

This manifest maps the manuscript's reported evidence to public scripts and aggregate outputs. Full reruns require the controlled processed dataset at `data/processed/threat_five_class.csv`; public verification can inspect code, schema, commands, saved aggregate artifacts, and the 1,000-row anonymized smoke-test sample without access to the full event-level data.

Controlled processed file identity for authorized reruns:

- Expected path: `data/processed/threat_five_class.csv`
- Event rows: 177,156
- Header rows: 1
- Size: 44,929,133 bytes
- SHA-256: `1BE9896A996DD58A582D94319180405A8559AB5193EF692BD8D2D2D614693724`

Public anonymized smoke-test sample:

- File: `data/processed/public_anonymized_sample_1000.csv`
- Audit: `data/processed/public_anonymized_sample_1000.audit.json`
- Rows: 1,000
- SHA-256: `52A00A06A8D9E7FB2AD7055B4504809451F1F22123078536D2FD42866CA91A08`
- Boundary: schema and parser smoke testing only; not used for reported manuscript metrics.

| Manuscript evidence | Script or command entry point | Public aggregate outputs |
| --- | --- | --- |
| Core holdout benchmark | `code/03_benchmark_baseline.py --dataset threat --feature-set core --tag baseline` | `results/baseline_benchmark_threat_core_baseline.csv`, `results/baseline_benchmark_threat_core_baseline.json` |
| No-threat-descriptors holdout | `code/03_benchmark_baseline.py --exclude-features "Threat/Content Name,Threat/Content Type,Severity" --tag no_threat_descriptors` | `results/baseline_benchmark_threat_core_no_threat_descriptors.csv`, `results/baseline_benchmark_threat_core_no_threat_descriptors.json` |
| Core and no-threat cross-validation | `code/03_benchmark_baseline.py --cv --tag *_top_cv` | `results/baseline_benchmark_threat_core_top_cv_cv_summary.csv`, `results/baseline_benchmark_threat_core_no_threat_descriptors_top_cv_cv_summary.csv` |
| Model-independent policy-action entropy | `code/10_policy_action_entropy_audit.py` | `results/policy_action_entropy_audit/policy_action_entropy_summary.csv`, `results/policy_action_entropy_audit/policy_action_rule_purity.csv` |
| Selected-model stratified and chronological validation | `code/05_q1_validation_extensions.py` | `results/extensions/q1_validation_summary.csv`, `results/extensions/q1_validation_confusions.csv`, `results/extensions/q1_validation_metadata.json` |
| Simple train-partition context baselines | `code/11_q1_audit_revision_checks.py` | `results/q1_audit_revision/simple_context_baselines.csv` |
| Confusion-cell bootstrap and chronological per-class recall | `code/11_q1_audit_revision_checks.py --confusions results/extensions/q1_validation_confusions.csv` | `results/q1_audit_revision/xgb_fixed_model_macro_f1_ci.csv`, `results/q1_audit_revision/xgb_fixed_model_per_class_recall_ci.csv` |
| Category-code-order sensitivity | `code/17_category_code_order_sensitivity.py --data data/processed/threat_five_class.csv` | `results/q1_audit_revision/category_code_order_sensitivity.csv`, `results/q1_audit_revision/category_code_order_sensitivity_summary.csv`, `results/q1_audit_revision/category_code_order_sensitivity_metadata.json` |
| Feature-group ablation and diagnostic referral | `code/06_strengthening_validation.py` | `results/strengthening/` |
| Duplicate-aware grouped split robustness | `code/07_duplicate_group_robustness.py` | `results/duplicate_group_robustness/duplicate_group_robustness_metadata.json` and associated CSV outputs |
| Held-out policy-context robustness | `code/08_policy_context_robustness.py` | `results/policy_context_robustness/policy_context_heldout_aggregate.csv`, `results/policy_context_robustness/policy_context_heldout_results.csv`, `results/policy_context_robustness/policy_context_heldout_confusions.csv` |
| Operational review queue and unseen-context audit | `code/09_operational_review_and_context_audit.py --n-estimators 220` | `results/operational_review_context_audit/operational_review_queue_summary.csv`, `results/operational_review_context_audit/context_holdout_aggregate.csv`, `results/operational_review_context_audit/README.md` |
| SHAP and LIME explanation artifacts | `code/04_xai_explanations.py` | `results/xai/` artifacts and manuscript figure files derived from them |
| Public sample smoke test | `code/03_benchmark_baseline.py --dataset threat --feature-set core --models "Decision Tree" --sample-per-class 20 --data-override data/processed/public_anonymized_sample_1000.csv --tag public_sample_smoke` | Local smoke-test output only; not part of reported evidence |

Event-level review queues and the full controlled processed CSV are intentionally excluded from the public package unless a separate institutional authorization approves release. The included 1,000-row public sample is strongly anonymized and intended only for smoke testing.
