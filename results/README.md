# Aggregate Result Boundary

This directory contains only aggregate outputs and manuscript-facing figures used to report or verify the study. The table, figure, and validation mappings are defined in `../TRACEABILITY.md`.

- Root-level benchmark files contain the canonical holdout and cross-validation summaries plus the policy-included upper-bound comparison.
- `extensions/`, `strengthening/`, and `q1_audit_revision/` contain the selected-model temporal, calibration, feature-group, interval, sensitivity, and baseline checks.
- `duplicate_group_robustness/`, `policy_context_robustness/`, `operational_review_context_audit/`, and `policy_action_entropy_audit/` contain aggregate audit evidence only; the policy-context folder also contains its deterministic manuscript figure rendered from the aggregate CSV.
- `xai/` contains aggregate SHAP evidence and manuscript-facing explanation figures, including one deidentified illustrative LIME panel; no row-level LIME table is released.
- `manuscript_figures/` contains deterministic manuscript figures rendered from the saved aggregate evidence.
- `uci_leaf_instantiation/` contains the public-dataset schema-transfer check.

Raw exports, the full processed event-level dataset, event-level review queues, local paths, credentials, private policy identifiers, and intermediate diagnostic plots are intentionally excluded. The observation date is disclosed by author decision; the protected boundary is institutional and operational context, not the calendar date.
