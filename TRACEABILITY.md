# Result Traceability Manifest

This manifest maps the manuscript's reported evidence to public scripts and aggregate outputs. Full reruns require the controlled processed dataset at `data/processed/threat_five_class.csv`; public verification can inspect code, schema, commands, saved aggregate artifacts, and the 1,000-row anonymized smoke-test sample without access to the full event-level data.

Controlled processed file identity for authorized reruns:

- Expected path: `data/processed/threat_five_class.csv`
- Event rows: 177,156
- Header rows: 1
- Size: 44,929,133 bytes
- SHA-256: `1BE9896A996DD58A582D94319180405A8559AB5193EF692BD8D2D2D614693724`

Canonical model-fitting environment and run:

- Run ID: `2026-07-13_codex_vps_selected_model_canonicalization`
- Platform: Oracle Cloud Ampere Linux/aarch64, 4 OCPU, 24 GB RAM, Python 3.12.3
- Exact lock: `requirements-lock-primary-linux-aarch64.txt`
- Acceptance gate: current `code/05` reproduced the original VPS XGBoost holdouts exactly (core 0.998373 macro-F1, 2 errors; no-threat-descriptors 0.990091 macro-F1, 15 errors) before the dependent evidence battery was accepted.
- Public UCI input SHA-256: `0B42E7EB9A4D7C314F65810C447EBF7F09D5E3E106EC14D36950A9EBAC61E9C0`; `code/16` was rerun in the same locked environment and its aggregate JSON is included below.

Public anonymized smoke-test sample:

- File: `data/processed/public_anonymized_sample_1000.csv`
- Audit: `data/processed/public_anonymized_sample_1000.audit.json`
- Rows: 1,000
- SHA-256: `52A00A06A8D9E7FB2AD7055B4504809451F1F22123078536D2FD42866CA91A08`
- Boundary: schema and parser smoke testing only; not used for reported manuscript metrics.

| Manuscript evidence | Script or command entry point | Public aggregate outputs |
| --- | --- | --- |
| Raw-to-processed cohort flow and target mapping | `code/02_build_threat_dataset.py --input <controlled-raw.csv> --output <controlled-output.csv> --manifest <aggregate-manifest.json>` | `data/processed/threat_dataset_processing_manifest.json` |
| Core holdout benchmark | `code/03_benchmark_baseline.py --dataset threat --feature-set core --tag baseline` | `results/baseline_benchmark_threat_core_baseline.csv`, `results/baseline_benchmark_threat_core_baseline.json` |
| No-threat-descriptors holdout | `code/03_benchmark_baseline.py --exclude-features "Threat/Content Name,Threat/Content Type,Severity" --tag no_threat_descriptors` | `results/baseline_benchmark_threat_core_no_threat_descriptors.csv`, `results/baseline_benchmark_threat_core_no_threat_descriptors.json` |
| Core and no-threat cross-validation | `code/03_benchmark_baseline.py --cv --tag *_top_cv` | `results/baseline_benchmark_threat_core_top_cv_cv_summary.csv`, `results/baseline_benchmark_threat_core_no_threat_descriptors_top_cv_cv_summary.csv` |
| Policy-included upper-bound comparison | `code/03_benchmark_baseline.py --dataset threat --feature-set with_policy --models "Decision Tree,Random Forest,Extra Trees,XGBoost,LightGBM,CatBoost" --tag with_policy` | `results/baseline_benchmark_threat_with_policy_with_policy.csv`, `results/baseline_benchmark_threat_with_policy_with_policy.json` |
| Model-independent policy-action entropy | `code/10_policy_action_entropy_audit.py` | `results/policy_action_entropy_audit/policy_action_entropy_summary.csv`, `results/policy_action_entropy_audit/policy_action_rule_purity.csv` |
| Selected-model stratified and chronological validation | `code/05_q1_validation_extensions.py` | `results/extensions/q1_validation_summary.csv`, `results/extensions/q1_validation_confusions.csv`, `results/extensions/q1_validation_metadata.json` |
| Simple train-partition context baselines | `code/11_q1_audit_revision_checks.py` | `results/q1_audit_revision/simple_context_baselines.csv` |
| Fixed-model test-record bootstrap and chronological per-class recall | `code/11_q1_audit_revision_checks.py --data data/processed/threat_five_class.csv --fit-xgb --bootstrap-iters 1000` | `results/q1_audit_revision/xgb_fixed_model_summary.csv`, `results/q1_audit_revision/xgb_fixed_model_bootstrap_ci.csv`, `results/q1_audit_revision/xgb_fixed_model_per_class_recall_ci.csv`, `results/q1_audit_revision/q1_audit_revision_metadata.json` |
| Ten-seed selected-model robustness | `code/12_repeated_seed_robustness.py` | `results/q1_audit_revision/repeated_seed_results.csv`, `results/q1_audit_revision/repeated_seed_summary.csv` |
| Class-weight sensitivity | `code/13_classweight_sensitivity.py` | `results/q1_audit_revision/classweight_sensitivity.csv` |
| Source-port-excluded near-duplicate grouping | `code/14_sourceport_nearduplicate_ablation.py --data data/processed/threat_five_class.csv --outdir results/q1_audit_revision` | `results/q1_audit_revision/sourceport_nearduplicate_ablation.csv` |
| Rolling-origin chronological stress | `code/15_forward_chaining_chronological.py --data data/processed/threat_five_class.csv` | `results/q1_audit_revision/forward_chaining_chronological.csv` |
| Public UCI schema-transferable instantiation | Download the CC BY 4.0 `log2.csv` from the official [UCI Internet Firewall Data page](https://archive.ics.uci.edu/dataset/542/internet+firewall+data) (DOI: [10.24432/C5131M](https://doi.org/10.24432/C5131M)), verify SHA-256 `0B42E7EB9A4D7C314F65810C447EBF7F09D5E3E106EC14D36950A9EBAC61E9C0`, then run `code/16_uci_leaf_instantiation.py --data data/uci_internet_firewall/log2.csv --outdir results/uci_leaf_instantiation --seed 42` | `results/uci_leaf_instantiation/uci_leaf_results.json` |
| Category-code-order sensitivity | `code/17_category_code_order_sensitivity.py --data data/processed/threat_five_class.csv` | `results/q1_audit_revision/category_code_order_sensitivity.csv`, `results/q1_audit_revision/category_code_order_sensitivity_summary.csv`, `results/q1_audit_revision/category_code_order_sensitivity_metadata.json` |
| Full-feature exact-signature lookup baseline | `code/18_full_feature_lookup_baseline.py --data data/processed/threat_five_class.csv` | `results/q1_audit_revision/full_feature_lookup_baseline.csv`, `results/q1_audit_revision/full_feature_lookup_baseline_metadata.json` |
| Class-distribution, combined holdout/CV, and feature-group manuscript figures | `code/19_render_manuscript_figures.py` with the aggregate processing manifest and canonical benchmark/strengthening CSV files; Lato Regular axis-label typography is hash-recorded | `results/manuscript_figures/fig_class_distribution.png`, `results/manuscript_figures/fig_results_ablation_cv_combined.png`, `results/manuscript_figures/fig_feature_group_validation.png`, `results/manuscript_figures/manuscript_figure_render_metadata.json` |
| Methodology workflow and release-boundary figure | `code/20_render_methodology_workflow.py --processing-manifest data/processed/threat_dataset_processing_manifest.json` | `results/manuscript_figures/fig_methodology_workflow.png`, `results/manuscript_figures/methodology_workflow_render_metadata.json` |
| Feature-group ablation and diagnostic referral | `code/06_strengthening_validation.py --seed 42 --n-estimators 300 --max-depth 6 --device cpu` | `results/strengthening/strengthening_summary.csv`, `strengthening_confusions.csv`, `confidence_deciles.csv`, `selective_prediction.csv`, `strengthening_metadata.json` |
| Duplicate-aware grouped split robustness | `code/07_duplicate_group_robustness.py --seed 42 --splits 5 --test-size 0.2 --n-estimators 300 --max-depth 6 --device cpu` | `results/duplicate_group_robustness/duplicate_signature_summary.csv`, `duplicate_group_split_results.csv`, `duplicate_group_split_summary.csv`, `duplicate_group_robustness_metadata.json` |
| Held-out policy-context robustness and manuscript figure | `code/08_policy_context_robustness.py --seed 42 --min-rule-rows 50 --n-estimators 300 --max-depth 6 --device cpu`; aggregate-only figure rendered by `code/21_render_heldout_context_audit.py` | `results/policy_context_robustness/policy_context_rule_summary.csv`, `policy_context_heldout_aggregate.csv`, `policy_context_heldout_results.csv`, `policy_context_heldout_confusions.csv`, `policy_context_robustness_metadata.json`, `fig_heldout_policy_context_audit.png`, and its render metadata |
| Operational review queue and unseen-context audit | `code/09_operational_review_and_context_audit.py --seed 42 --n-estimators 300 --max-depth 6 --max-contexts-per-column 6` | Aggregate-only `results/operational_review_context_audit/operational_review_model_summary.csv`, `operational_review_queue_summary.csv`, `context_holdout_results.csv`, `context_holdout_confusions.csv`, `context_holdout_aggregate.csv`, and metadata; public per-context values use deterministic aliases and no event-level queue is included |
| SHAP and LIME explanation artifacts | `code/04_xai_explanations.py`; canonical locked environment, with Lato Regular axis-title and Inter Regular non-bold internal-text typography hash-recorded | `results/xai/xai_generation_summary.json`, `results/xai/xai_shap_global_importance.csv`, and the three manuscript-facing explanation figures; row-level LIME tables remain controlled |
| Public sample smoke test | `code/03_benchmark_baseline.py --dataset threat --feature-set core --models "Decision Tree" --sample-per-class 20 --data-override data/processed/public_anonymized_sample_1000.csv --tag public_sample_smoke` | Local smoke-test output only; not part of reported evidence |

Event-level review queues and the full controlled processed CSV are intentionally excluded from the public package unless a separate institutional authorization approves release. The included 1,000-row public sample is strongly anonymized and intended only for smoke testing.
