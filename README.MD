# Policy-Conditioned Firewall Enforcement Audit Reproducibility Package

Public reproducibility artifact for the manuscript:

**LEAF: A Leakage-Aware, Explainable Audit Framework for Policy-Conditioned Firewall Enforcement**

Short protocol/artifact name:

`FIRE-AUDIT`

Repository short name:

`threat-xai-action`

GitHub URL:

`https://github.com/canay/threat-xai-action`

## Related Study

This repository supports FIRE-AUDIT, a paper on five-class operational firewall enforcement audit from real enterprise Palo Alto threat logs. The study uses supervised machine-learning models as comparable measurement instruments in a leakage-aware evaluation protocol, testing whether high performance persists after direct policy and threat descriptor proxies are bounded.

Target classes:

- Allow
- Deny
- Drop
- Reset-Both
- Reset-Server

## Authors And Contact

Author order:

1. **Özkan Canay**, corresponding author
2. **Cem Özkurt**
3. **Fatmir Garri**

Özkan Canay

- Department of Information Systems and Technologies
- Faculty of Computer and Information Sciences
- Sakarya University
- Email: canay@sakarya.edu.tr
- ORCID: 0000-0001-7539-6001

Cem Özkurt

- Department of Data Science and Analytics
- Faculty of Computer and Information Sciences
- Sakarya University
- Email: cemozkurt@sakarya.edu.tr
- ORCID: 0000-0002-1251-7715

Fatmir Garri

- Department of Information Technology
- Sakarya University of Applied Sciences
- Email: fatmirgarri@subu.edu.tr
- ORCID: 0009-0005-6762-6846

GitHub:

https://github.com/canay

This repository is the author-managed code-and-output replication package for the associated manuscript. A Zenodo/DOI decision is deferred until journal submission.

## Repository Contents

- `code/`: the controlled-data conversion entry point and all manuscript-used evaluation, explanation, validation, and figure-rendering scripts (`02`--`20`)
- `scripts/create_public_sample.py`: the single retained helper for generating the privacy-audited public sample
- `data/processed/`: controlled-data access and schema notes, the aggregate processing manifest, and a 1,000-row anonymized public smoke-test sample with audit
- `results/`: CSV/JSON aggregate evaluation outputs, reproducible explanation artifacts, and additional validation outputs
- `requirements.txt`: portable minimum compatible dependencies
- `requirements-lock.txt`: exact direct dependencies from the submission-facing Windows x86-64 CPU rerun
- `requirements-lock-primary-linux-aarch64.txt`: complete platform-labelled resolution from the original Linux/aarch64 run
- `SEED_MANIFEST.json`: explicit random seeds and resampling scope
- `CITATION.cff`: citation metadata for this artifact
- `MANIFEST.md`: release contract and public/controlled-content checklist
- `TRACEABILITY.md`: manuscript evidence to script/output mapping, including the controlled processed-file checksum for authorized reruns

Raw enterprise firewall exports are **not** redistributed in this repository. The full event-level processed dataset is also not included unless explicit institutional authorization confirms that redistribution is permitted after privacy and security review. A 1,000-row anonymized sample is included only for schema inspection and smoke testing; it is not used for the reported manuscript metrics.

## Data

Controlled processed dataset path:

- `data/processed/threat_five_class.csv`

The file above is the expected local path for authorized full reproduction, but the full file is not included in this public artifact. The data were derived from institutional firewall logs and may expose organization-specific or security-sensitive operational context even after processing. The calendar date, 14 May 2026, is intentionally disclosed with author approval; organization identity, topology, policy names, raw exports, and event-level records remain protected.

Aggregate cohort flow:

- Raw threat-log rows: 186,271
- Excluded detection-only `alert` rows: 9,115
- Mapped five-class rows: 177,156
- Observation window: 6 h 30 min 26 s on a high-volume weekday
- Aggregate audit: `data/processed/threat_dataset_processing_manifest.json`

Rows:

- 177,156 labeled threat-log records

Controlled processed file identity for authorized reruns:

- Event rows: 177,156
- Header rows: 1
- Size: 44,929,133 bytes
- SHA-256: `1BE9896A996DD58A582D94319180405A8559AB5193EF692BD8D2D2D614693724`

Class counts:

- Drop: 160,788
- Deny: 10,187
- Reset-Both: 2,903
- Allow: 2,502
- Reset-Server: 776

Public anonymized smoke-test sample:

- File: `data/processed/public_anonymized_sample_1000.csv`
- Audit: `data/processed/public_anonymized_sample_1000.audit.json`
- Rows: 1,000
- SHA-256: `52A00A06A8D9E7FB2AD7055B4504809451F1F22123078536D2FD42866CA91A08`
- Class counts: Drop 731, Deny 93, Reset-Both 62, Allow 61, Reset-Server 53

This sample preserves the 27-column processed-file schema and all five target classes, but it does not preserve original timestamps, raw firewall actions, policy names, zones, interfaces, ports, applications, threat names, countries, source row identifiers, original row order, or other event-level values. Except for `target`, event-level values are redacted or mapped within each column to stable generic tokens such as `application_001`, `source_zone_001`, and `rule_context_001`; sampled rows are shuffled before synthetic sequential timestamps on 2026-01-01 are assigned. The sample is suitable for checking file format, parser behavior, and lightweight script execution only. It does not reproduce the reported metrics.

Privacy note:

The package excludes raw enterprise log exports and excludes the full event-level processed CSV by default. Non-missing `Rule` values in the controlled processed file were converted to stable anonymous rule-context identifiers during internal reproduction, and the public smoke-test sample applies stronger column-wise tokenization before release. Public release of the full processed file should occur only after the relevant institutional authority approves the exact file and any required data-use agreement.

Detailed event-level review-queue records are excluded from the public artifact. `code/09_operational_review_and_context_audit.py` writes only aggregate and aliased per-context outputs by default. Authorized users may request an event-level queue only with `--private-records-out`; the script refuses any such destination inside this public repository.

Public schema aids:

- `data/processed/schema.csv` documents the expected columns, modeling role, and public-release note for each field.
- `data/processed/public_anonymized_sample_1000.csv` is a real-row-derived but strongly anonymized sample for schema inspection and smoke testing only.
- `data/processed/public_anonymized_sample_1000.audit.json` records the sample seed, source checksum, output checksum, class counts, and anonymization rules.
- `data/processed/threat_dataset_processing_manifest.json` records aggregate target-construction counts and the controlled processed-file checksum.
- `TRACEABILITY.md` maps manuscript tables and validation claims to generating scripts and aggregate output files.

## Main Evidence Snapshot

Core holdout benchmark:

- Leading two-error profile: LightGBM and CatBoost
- Downstream reference: XGBoost, with three reset-boundary errors
- Accuracy: 0.999944
- Balanced accuracy: 0.999312
- Macro-F1: 0.998373
- Weighted-F1: 0.999944

Core five-fold cross-validation macro-F1:

- LightGBM: 0.996674 +/- 0.001951
- CatBoost: 0.996630 +/- 0.001930
- XGBoost: 0.996409 +/- 0.002665
- Extra Trees: 0.995018 +/- 0.002547

Leakage-aware ablation without direct threat descriptors:

- Removed fields: `Threat/Content Name`, `Threat/Content Type`, `Severity`
- LightGBM holdout macro-F1: 0.993126
- XGBoost holdout macro-F1: 0.990588

No-threat-descriptors cross-validation:

- XGBoost CV macro-F1 mean: 0.986297
- LightGBM CV macro-F1 mean: 0.985312
- Extra Trees CV macro-F1 mean: 0.981459
- CatBoost CV macro-F1 mean: 0.976976

Additional XGBoost validation checks:

- Minimal-context stratified holdout macro-F1: 0.982613
- Core chronological holdout macro-F1: 0.866385
- No-threat-descriptors chronological holdout macro-F1: 0.833761
- Core chronological holdout ECE, 10 bins: 0.004823
- SHAP top-10 feature overlap across repeated stratified samples: 1.000000
- The canonical SHAP rerun reproduces the selected XGBoost holdout at 0.997554 macro-F1 with three errors.
- A 64-record diagnostic reconstructs raw margins within 0.00000644 maximum absolute error and probabilities within 0.00000012.
- The public XAI release includes aggregate SHAP evidence and three deidentified manuscript figures; no row-level LIME table is released.

Model-independent policy-action entropy audit:

- Outputs are in `results/policy_action_entropy_audit/`.
- Rule-context dominant-action fraction, record-weighted: 0.992758.
- Rule-context normalized entropy, record-weighted: 0.017669.
- Rule + threat type + direction dominant-action fraction, record-weighted: 0.997217.
- Low-purity rule-context rows at a 0.80 dominant-action threshold: 1,248.
- Rule values are anonymized as `rule_context_*` in the output artifacts by default.

Strengthening checks:

- Feature-group ablation results are in `results/strengthening/`.
- Chronological core macro-F1 in the feature-group audit: 0.866385.
- Chronological no-threat-descriptors macro-F1 in the feature-group audit: 0.833761.
- Chronological no-application-context macro-F1: 0.969289.
- Confidence-based referral on chronological core retains 97.51% of records with 10 retained-set errors and 0.981524 macro-F1.
- These temporal results are within-day forward-shift stress tests, not long-term temporal validation.

Duplicate-aware grouped-split stress test:

- Exact feature signatures are grouped so the same structured feature vector cannot appear in both train and test partitions.
- Core feature signatures: 47,047 unique signatures; 151,361 duplicate rows (85.44%).
- No-threat-descriptors grouped split macro-F1: 0.983254 +/- 0.004835.
- Minimal-context grouped split macro-F1: 0.977105 +/- 0.007005.
- Core grouped split macro-F1: 0.959371 +/- 0.084590, with higher variance caused by one more difficult grouped partition.

Held-out policy-context stress test:

- `Rule` is used only as a held-out grouping variable and is not included as a predictor.
- Among seven named rule contexts with at least 50 records, the core model reached 0.808266 support-weighted accuracy and 0.654679 support-weighted observed-class macro-F1.
- The no-threat-descriptors and minimal-context settings reached 0.376787 and 0.371638 support-weighted observed-class macro-F1, respectively.
- These results are a stress audit for local policy-context sensitivity, not evidence of cross-organization generalization.

Operational review and unseen-context audit:

- Outputs are in `results/operational_review_context_audit/`.
- The saved operational review aggregate is generated with the 300-tree selected-model configuration (`--n-estimators 300`), so its model-log disagreement counts match the selected-model validation table.
- Aggregate queue summaries and aliased per-context results are public; detailed event-level queue records are not distributed.
- In the stratified core setting, a review queue triggered by reset-related records or the lowest 1% confidence records routed 939 of 35,432 test records (2.65%) to review and contained all 3 model-log disagreements.
- In the no-threat-descriptors setting, the same review rule routed 903 records (2.55%) and contained all 14 model-log disagreements.
- A broader rule using model-log disagreement, lowest 5% confidence, or reset-related records routed 1,802 core records (5.09%) and 2,019 no-threat-descriptors records (5.70%).
- The exploratory unseen-context holdout audit shows that performance can deteriorate sharply when entire high-support categorical values are absent from training. Treat this as a deployment-boundary stress test, not as independent external validation.

Q1 audit revision checks:

- Outputs are in `results/q1_audit_revision/`.
- Train-majority macro-F1 is 0.190312 under stratified holdout and 0.181301 under chronological holdout.
- A non-rule, non-direct-descriptor app-direction-zone context lookup reaches 0.953460 stratified macro-F1 but falls to 0.748216 under chronological ordering.
- A richer minimal operational context lookup reaches 0.976840 stratified macro-F1 and 0.620468 chronological macro-F1.
- A fair 20-feature exact-signature lookup reaches only 0.511682 stratified macro-F1 and 0.225351 chronological macro-F1 because unmatched high-cardinality signatures fall back to the training-majority action; the learned model's performance is therefore not reproduced by exact-signature memorization.
- Test-record bootstrap intervals from saved fixed-model XGBoost outputs give chronological core macro-F1 0.866385 with 95% interval 0.853189--0.879027.
- The corresponding chronological no-threat-descriptors macro-F1 is 0.833761 with a 95% interval of 0.819881--0.848151.

Round-9 revision experiments:

- Outputs are in `results/q1_audit_revision/`. All use the 300-tree, seed-42 selected-model pipeline. The submission-facing benchmark and selected-model checks were regenerated under one locked Windows x86-64 CPU environment so that the reported XGBoost rows share the same preprocessing, library, split, and model contract.
- `classweight_sensitivity.csv`: adding inverse-frequency sample weights to XGBoost changes stratified macro-F1 by at most 0.0008 (core 0.9976 to 0.9984; no-threat-descriptors 0.9906 to 0.9907) and slightly improves rare-class balanced accuracy, so the unweighted selection does not understate minority performance.
- `sourceport_nearduplicate_ablation.csv`: `Source Port` is high-cardinality (10,345 distinct); grouping records by their signature excluding `Source Port` (which collapses 14,678 signature groups and raises the duplicate-row fraction from 0.854 to 0.923) still yields a median core macro-F1 of 0.9932 (range 0.924--0.997), so the reconstruction evidence does not collapse when burst-level source-port variation is removed from grouping.
- `forward_chaining_chronological.csv`: expanding-window chronological folds (train earliest 60/70/80/90 percent, test the following 10 percent) show that within-day temporal performance is cutpoint-dependent rather than a single value, consistent with the class-mix shift documented for the 80/20 cut.
- `category_code_order_sensitivity_summary.csv`: under canonical most-frequent categorical imputation and fixed transformed-feature order, five category-map permutations change selected-model macro-F1 by at most 0.0008 in the core setting and 0.0020 without threat descriptors.

## Experimental Environment

The original benchmark queue ran on an Oracle Cloud Ampere Linux aarch64 virtual private server with 4 OCPU, 24 GB RAM, and Python 3.12.3. The submission-facing benchmark, timestamp-safe validations, and explanation artifacts were regenerated on a Windows x86-64 CPU workstation with Python 3.12.12 and no specialized accelerator. Inference latency, throughput, and live deployment performance were not evaluated.

The exact direct dependencies for the submission-facing Python 3.12.12 Windows x86-64 CPU rerun are recorded in `requirements-lock.txt`, and the full resolved Python 3.12.3 Linux/aarch64 environment from the original queue is preserved in `requirements-lock-primary-linux-aarch64.txt`. `requirements.txt` remains the portable minimum-version installer. Round 15 canonical reruns additionally record their exact local environment and output hashes in the experiment manifests.

## Reproducing With Authorized Processed Data

The scripts below require the controlled processed dataset at `data/processed/threat_five_class.csv`. That file is not included in this public artifact by default. Use these commands only after placing an institutionally approved copy of the processed dataset at that path.

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

For exact inspection, use the direct Windows rerun lock for the submission-facing outputs and the platform-labelled Linux lock only for the preserved original queue; neither file is a cross-platform installer.

Build the controlled five-class dataset and write an aggregate manifest (all controlled paths must remain outside this public repository):

```bash
python code/02_build_threat_dataset.py --input <controlled-raw.csv> --output <controlled-output.csv> --manifest <aggregate-manifest.json>
```

Smoke-test the schema and parser on the public anonymized sample:

```bash
python code/03_benchmark_baseline.py --dataset threat --feature-set core --models "Decision Tree" --sample-per-class 20 --data-override data/processed/public_anonymized_sample_1000.csv --tag public_sample_smoke --output-dir results/public_sample_smoke
```

The command above is the supported public-sample smoke test. Other scripts require the controlled processed dataset because the public sample deliberately tokenizes sensitive numeric values; do not rename or substitute the public sample to regenerate manuscript metrics or explanation artifacts.

Run the core holdout benchmark:

```bash
python code/03_benchmark_baseline.py --dataset threat --feature-set core --models "Decision Tree,Random Forest,Extra Trees,XGBoost,LightGBM,CatBoost" --tag baseline --output-dir results
```

Run the no-threat-descriptors ablation:

```bash
python code/03_benchmark_baseline.py --dataset threat --feature-set core --models "Decision Tree,Random Forest,Extra Trees,XGBoost,LightGBM,CatBoost" --exclude-features "Threat/Content Name,Threat/Content Type,Severity" --tag no_threat_descriptors --output-dir results
```

Run the no-threat-descriptors top-model CV:

```bash
python code/03_benchmark_baseline.py --dataset threat --feature-set core --models "Extra Trees,XGBoost,LightGBM,CatBoost" --exclude-features "Threat/Content Name,Threat/Content Type,Severity" --cv --tag no_threat_descriptors_top_cv --output-dir results
```

Run the matching core top-model CV:

```bash
python code/03_benchmark_baseline.py --dataset threat --feature-set core --models "Extra Trees,XGBoost,LightGBM,CatBoost" --cv --tag top_cv --output-dir results
```

Run the policy-included upper-bound comparison:

```bash
python code/03_benchmark_baseline.py --dataset threat --feature-set with_policy --models "Decision Tree,Random Forest,Extra Trees,XGBoost,LightGBM,CatBoost" --tag with_policy --output-dir results
```

Generate the SHAP and LIME explanation artifacts used for the manuscript figures:

```bash
python code/04_xai_explanations.py --data data/processed/threat_five_class.csv --outdir results/xai
```

Run the additional validation checks:

```bash
python code/05_q1_validation_extensions.py --data data/processed/threat_five_class.csv --outdir results/extensions
```

Run the strengthening checks:

```bash
python code/06_strengthening_validation.py --data data/processed/threat_five_class.csv --outdir results/strengthening
```

If XGBoost is installed with CUDA support:

```bash
python code/06_strengthening_validation.py --data data/processed/threat_five_class.csv --outdir results/strengthening --device cuda
```

Run duplicate-aware grouped split robustness:

```bash
python code/07_duplicate_group_robustness.py --data data/processed/threat_five_class.csv --outdir results/duplicate_group_robustness
```

Run policy-context held-out robustness:

```bash
python code/08_policy_context_robustness.py --data data/processed/threat_five_class.csv --outdir results/policy_context_robustness
```

Run operational review-queue and unseen-context audit:

```bash
python code/09_operational_review_and_context_audit.py --data data/processed/threat_five_class.csv --outdir results/operational_review_context_audit --n-estimators 300
```

Run model-independent policy-action entropy audit:

```bash
python code/10_policy_action_entropy_audit.py --data data/processed/threat_five_class.csv --outdir results/policy_action_entropy_audit
```

Run Q1 audit revision baselines and bootstrap intervals:

```bash
python code/11_q1_audit_revision_checks.py --data data/processed/threat_five_class.csv --outdir results/q1_audit_revision --confusions results/extensions/q1_validation_confusions.csv --fit-xgb --bootstrap-iters 1000
```

If the controlled processed dataset is not available and only aggregate confusion-cell CI outputs are needed:

```bash
python code/11_q1_audit_revision_checks.py --outdir results/q1_audit_revision --confusions results/extensions/q1_validation_confusions.csv --confusion-only
```

Run the additional robustness and audit experiments:

```bash
python code/12_repeated_seed_robustness.py --data data/processed/threat_five_class.csv
python code/13_classweight_sensitivity.py --data data/processed/threat_five_class.csv
python code/14_sourceport_nearduplicate_ablation.py --data data/processed/threat_five_class.csv
python code/15_forward_chaining_chronological.py --data data/processed/threat_five_class.csv --outdir results/q1_audit_revision
python code/16_uci_leaf_instantiation.py --data data/uci_internet_firewall/log2.csv --outdir results/uci_leaf_instantiation --seed 42
python code/17_category_code_order_sensitivity.py --data data/processed/threat_five_class.csv --outdir results/q1_audit_revision
python code/18_full_feature_lookup_baseline.py --data data/processed/threat_five_class.csv --outdir results/q1_audit_revision
```

Render the manuscript-facing descriptive, holdout/CV, and feature-group figures from the saved aggregate evidence:

```bash
python code/19_render_manuscript_figures.py --processing-manifest data/processed/threat_dataset_processing_manifest.json --core-holdout results/baseline_benchmark_threat_core_baseline.csv --no-threat-holdout results/baseline_benchmark_threat_core_no_threat_descriptors.csv --core-cv results/baseline_benchmark_threat_core_top_cv_cv_summary.csv --no-threat-cv results/baseline_benchmark_threat_core_no_threat_descriptors_top_cv_cv_summary.csv --strengthening results/strengthening/strengthening_summary.csv --outdir results/manuscript_figures
```

Render the manuscript methodology workflow from the aggregate cohort manifest:

```bash
python code/20_render_methodology_workflow.py --processing-manifest data/processed/threat_dataset_processing_manifest.json --outdir results/manuscript_figures
```

The workflow renderer prefers the open-source Source Sans 3 Regular/Semibold faces when available, records the resolved font-file hashes, and applies a content-aware 600-dpi PNG crop with a 12-pixel safety margin. Open Sans, Lato, and DejaVu Sans are deterministic fallback families.

## Submission And Public Release Checklist

Before archival public release:

- Confirm that the GitHub URL matches the public repository.
- Fill the final commit and release tag fields in `MANIFEST.md`.
- Defer the optional repository DOI decision until the submission-stage Zenodo decision.
- Confirm that `TRACEABILITY.md` still maps the manuscript tables and figures to the final public outputs.
- Confirm that the controlled processed-file checksum is unchanged or update the checksum in `README.MD`, `MANIFEST.md`, `TRACEABILITY.md`, and `data/processed/README.md`.
- Confirm that `data/processed/public_anonymized_sample_1000.csv` matches its audit checksum and contains only anonymized sample values.
- Confirm that `data/processed/` does not contain the full event-level firewall CSV unless institutional public-release authorization has been obtained.
- Confirm that no raw enterprise logs, manuscript drafts, private review notes, or local credentials are included.
- Check whether the publication venue requires a data availability statement or code availability statement.

## License

This project's code and aggregate output artifacts are released under the MIT License (see `LICENSE`). The public anonymized 1,000-row sample is included as a limited reproducibility aid for schema inspection and smoke testing, not as a general-purpose benchmark dataset. The raw and full event-level processed enterprise firewall data are not covered by this license and remain controlled-access institutional material, as described in the Data availability section.
