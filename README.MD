# Policy-Conditioned Firewall Enforcement Audit Reproducibility Package

Public reproducibility artifact for the manuscript:

**Auditing Policy-Conditioned Firewall Enforcement: A Leakage-Aware and Explainable Evaluation Protocol for Enterprise Threat Logs**

Short protocol/artifact name:

**FIRE-AUDIT: Leakage-Aware Firewall-Action Reconstruction and Explainable Audit for Enterprise Network Logs**

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

This repository is the author-managed code-and-output replication package for the submitted manuscript. A repository DOI can be added after an archival release is created.

## Repository Contents

- `code/`: Python evaluation, post-hoc explanation, and additional validation code adapted for this standalone reproducibility package
- `scripts/`: original paper-level helper scripts from the research workspace
- `data/processed/`: controlled-data access note, schema note, a 1,000-row anonymized public smoke-test sample, its privacy audit, and a synthetic schema example
- `results/`: CSV/JSON aggregate evaluation outputs, timing logs, reproducible explanation artifacts, and additional validation outputs
- `requirements.txt`: Python dependencies used in the experiments
- `CITATION.cff`: citation metadata for this artifact
- `MANIFEST.md`: release contract and public/controlled-content checklist
- `TRACEABILITY.md`: manuscript evidence to script/output mapping, including the controlled processed-file checksum for authorized reruns

Raw enterprise firewall exports are **not** redistributed in this repository. The full event-level processed dataset is also not included unless explicit institutional authorization confirms that redistribution is permitted after privacy and security review. A 1,000-row anonymized sample is included only for schema inspection and smoke testing; it is not used for the reported manuscript metrics.

## Data

Controlled processed dataset path:

- `data/processed/threat_five_class.csv`

The file above is the expected local path for authorized full reproduction, but the full file is not included in this public artifact. The data were derived from institutional firewall logs and may expose privacy-sensitive, organization-specific, or security-sensitive operational context even after processing.

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

Detailed event-level review-queue records are also excluded from the public artifact. The operational review script can regenerate them locally for authorized users with approved access to the controlled processed dataset, while the repository retains aggregate queue summaries for inspection.

Public schema aids:

- `data/processed/schema.csv` documents the expected columns, modeling role, and public-release note for each field.
- `data/processed/public_anonymized_sample_1000.csv` is a real-row-derived but strongly anonymized sample for schema inspection and smoke testing only.
- `data/processed/public_anonymized_sample_1000.audit.json` records the sample seed, source checksum, output checksum, class counts, and anonymization rules.
- `data/processed/synthetic_schema_example.csv` is a small fake schema-compatible example. It is not used for any reported result and is not a substitute for the controlled processed dataset.
- `TRACEABILITY.md` maps manuscript tables and validation claims to generating scripts and aggregate output files.

## Main Evidence Snapshot

Core holdout baseline:

- Top models: LightGBM, XGBoost, CatBoost
- Accuracy: 0.999944
- Balanced accuracy: 0.999312
- Macro-F1: 0.998373
- Weighted-F1: 0.999944

Leakage-aware ablation without direct threat descriptors:

- Removed fields: `Threat/Content Name`, `Threat/Content Type`, `Severity`
- LightGBM holdout macro-F1: 0.993126
- XGBoost holdout macro-F1: 0.990091

No-threat-descriptors cross-validation:

- XGBoost CV macro-F1 mean: 0.986870
- LightGBM CV macro-F1 mean: 0.985312
- Extra Trees CV macro-F1 mean: 0.981459

Additional XGBoost validation checks:

- Minimal-context stratified holdout macro-F1: 0.982613
- Core chronological holdout macro-F1: 0.866385
- No-threat-descriptors chronological holdout macro-F1: 0.838867
- Core chronological holdout ECE, 10 bins: 0.004559
- SHAP top-10 feature overlap across repeated stratified samples: 1.000000
- CatBoost CV macro-F1 mean: 0.976976

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
- Chronological no-threat-descriptors macro-F1 in the feature-group audit: 0.838867.
- Chronological no-application-context macro-F1: 0.959462.
- Confidence-based referral on chronological core retains 97.65% of records with 9 retained-set errors and 0.984085 macro-F1.
- These temporal results are within-day forward-shift stress tests, not long-term temporal validation.

Duplicate-aware grouped split robustness:

- Exact feature signatures are grouped so the same structured feature vector cannot appear in both train and test partitions.
- Core feature signatures: 47,047 unique signatures; 151,361 duplicate rows (85.44%).
- No-threat-descriptors grouped split macro-F1: 0.983254 +/- 0.004835.
- Minimal-context grouped split macro-F1: 0.977105 +/- 0.007005.
- Core grouped split macro-F1: 0.959371 +/- 0.084590, with higher variance caused by one more difficult grouped partition.

Policy-context held-out robustness:

- `Rule` is used only as a held-out grouping variable and is not included as a predictor.
- Among seven named rule contexts with at least 50 records, the core model reached 0.808266 support-weighted accuracy and 0.654679 support-weighted observed-class macro-F1.
- The no-threat-descriptors and minimal-context settings reached 0.376787 and 0.371638 support-weighted observed-class macro-F1, respectively.
- These results are a stress audit for local policy-context sensitivity, not evidence of cross-organization generalization.

Operational review and unseen-context audit:

- Outputs are in `results/operational_review_context_audit/`.
- The saved operational review aggregate is generated with the 300-tree selected-model configuration (`--n-estimators 300`), so its model-log disagreement counts match the selected-model validation table.
- Aggregate queue summaries are public; detailed event-level queue records are not distributed by default.
- In the stratified core setting, a review queue triggered by reset-related records or the lowest 1% confidence records routed 939 of 35,432 test records (2.65%) to review and contained all 3 model-log disagreements.
- In the no-threat-descriptors setting, the same review rule routed 903 records (2.55%) and contained all 14 model-log disagreements.
- A broader rule using model-log disagreement, lowest 5% confidence, or reset-related records routed 1,802 core records (5.09%) and 2,019 no-threat-descriptors records (5.70%).
- The exploratory unseen-context holdout audit shows that performance can deteriorate sharply when entire high-support categorical values are absent from training. Treat this as a deployment-boundary stress test, not as independent external validation.

Q1 audit revision checks:

- Outputs are in `results/q1_audit_revision/`.
- Train-majority macro-F1 is 0.190312 under stratified holdout and 0.181301 under chronological holdout.
- A non-rule, non-direct-descriptor app-direction-zone context lookup reaches 0.953460 stratified macro-F1 but falls to 0.748216 under chronological ordering.
- A richer minimal operational context lookup reaches 0.976840 stratified macro-F1 and 0.620468 chronological macro-F1.
- Confusion-cell bootstrap intervals from saved fixed-model XGBoost outputs give chronological core macro-F1 0.866385 with 95% interval 0.853020--0.878218.
- The same interval for chronological no-threat-descriptors macro-F1 is 0.824702--0.851778.

Round-9 revision experiments:

- Outputs are in `results/q1_audit_revision/`. All use the 300-tree, seed-42 selected-model pipeline. The stratified selected-model results reproduced identically on a second architecture (Windows x86-64, Python 3.11.15, scikit-learn 1.9.0, XGBoost 3.2.0) as on the primary aarch64 environment.
- `classweight_sensitivity.csv`: adding inverse-frequency sample weights to XGBoost changes stratified macro-F1 by at most 0.0008 (core 0.9976 to 0.9984; no-threat-descriptors 0.9906 to 0.9907) and slightly improves rare-class balanced accuracy, so the unweighted selection does not understate minority performance.
- `sourceport_nearduplicate_ablation.csv`: `Source Port` is high-cardinality (10,345 distinct); grouping records by their signature excluding `Source Port` (which collapses 14,678 signature groups and raises the duplicate-row fraction from 0.854 to 0.923) still yields a median core macro-F1 of 0.9932 (range 0.924--0.997), so the reconstruction does not depend on a burst-level source-port pseudo-identifier.
- `forward_chaining_chronological.csv`: expanding-window chronological folds (train earliest 60/70/80/90 percent, test the following 10 percent) show that within-day temporal performance is cutpoint-dependent rather than a single value, consistent with the class-mix shift documented for the 80/20 cut.

## Experimental Environment

Final timing-sensitive experiments were conducted on:

- Oracle Cloud Ampere virtual private server
- 4 OCPU
- 24 GB RAM
- Python 3.12.3
- Linux aarch64

The same VPS environment was used for the final benchmark, ablation, and cross-validation results in `results/`.

## Reproducing With Authorized Processed Data

The scripts below require the controlled processed dataset at `data/processed/threat_five_class.csv`. That file is not included in this public artifact by default. Use these commands only after placing an institutionally approved copy of the processed dataset at that path.

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Smoke-test the schema and parser on the public anonymized sample:

```bash
python code/03_benchmark_baseline.py --dataset threat --feature-set core --models "Decision Tree" --sample-per-class 20 --data-override data/processed/public_anonymized_sample_1000.csv --tag public_sample_smoke --output-dir results/public_sample_smoke
```

If a script does not expose a data-override option, copy the public sample to a temporary working path named `data/processed/threat_five_class.csv` only for local smoke testing. Do not use the public sample to regenerate reported manuscript metrics.

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
python code/11_q1_audit_revision_checks.py --data data/processed/threat_five_class.csv --outdir results/q1_audit_revision --confusions results/extensions/q1_validation_confusions.csv
```

If the controlled processed dataset is not available and only aggregate confusion-cell CI outputs are needed:

```bash
python code/11_q1_audit_revision_checks.py --outdir results/q1_audit_revision --confusions results/extensions/q1_validation_confusions.csv --confusion-only
```

Run the round-9 revision experiments (class-weight sensitivity, source-port near-duplicate ablation, forward-chaining chronological):

```bash
python code/13_classweight_sensitivity.py --data data/processed/threat_five_class.csv
python code/14_sourceport_nearduplicate_ablation.py
python code/15_forward_chaining_chronological.py
```

## Submission And Public Release Checklist

Before archival public release:

- Confirm that the GitHub URL matches the public repository.
- Fill final commit, release tag, and optional DOI fields in `MANIFEST.md`.
- Add a repository DOI if an archival release is created.
- Confirm that `TRACEABILITY.md` still maps the manuscript tables and figures to the final public outputs.
- Confirm that the controlled processed-file checksum is unchanged or update the checksum in `README.MD`, `MANIFEST.md`, `TRACEABILITY.md`, and `data/processed/README.md`.
- Confirm that `data/processed/public_anonymized_sample_1000.csv` matches its audit checksum and contains only anonymized sample values.
- Confirm that `data/processed/` does not contain the full event-level firewall CSV unless institutional public-release authorization has been obtained.
- Confirm that no raw enterprise logs, manuscript drafts, private review notes, or local credentials are included.
- Check whether the target journal requires a data availability statement, code availability statement, or repository DOI.
- Consider archiving the final GitHub release on Zenodo and adding the DOI here.

## License

This project's code and aggregate output artifacts are released under the MIT License (see `LICENSE`). The public anonymized 1,000-row sample is included as a limited reproducibility aid for schema inspection and smoke testing, not as a general-purpose benchmark dataset. The raw and full event-level processed enterprise firewall data are not covered by this license and remain controlled-access institutional material, as described in the Data availability section.

