# Policy-Conditioned Firewall Action Prediction Reproducibility Package

Public reproducibility artifact for the manuscript:

**Leakage-Aware Explainable Machine Learning for Policy-Conditioned Firewall Action Prediction from Enterprise Threat Logs**

Repository short name:

`threat-xai-action`

GitHub URL:

`https://github.com/canay/threat-xai-action`

## Related Study

This repository supports a paper on five-class operational firewall action prediction from real enterprise Palo Alto threat logs. The study compares supervised machine-learning models and uses leakage-aware ablation to test whether high performance persists after direct threat descriptor fields are removed.

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

Muhammed Emin ONAY

- Department of Mechatronics Engineering
- Faculty of Technology
- Sakarya University of Applied Sciences
- Email: 22010404086@subu.edu.tr
- ORCID: 0009-0000-0454-4986

GitHub:

https://github.com/canay

Before public release, confirm the contact policy, journal name, DOI if available, and publication status.

## Repository Contents

- `code/`: Python benchmark, post-hoc explanation, and additional validation code adapted for this standalone reproducibility package
- `scripts/`: original paper-level helper scripts from the research workspace
- `data/processed/`: placeholder and access note for the controlled processed dataset path
- `results/`: CSV/JSON benchmark outputs, timing logs, reproducible explanation artifacts, and additional validation outputs
- `requirements.txt`: Python dependencies used in the experiments
- `CITATION.cff`: citation metadata for this artifact

Raw enterprise firewall exports are **not** redistributed in this repository. The event-level processed dataset is also not included in this public repository candidate unless explicit institutional authorization confirms that redistribution is permitted after privacy and security review.

## Data

Controlled processed dataset path:

- `data/processed/threat_five_class.csv`

The file above is the expected local path for authorized reproduction, but it is not included in this public repository candidate. The data were derived from institutional firewall logs and may expose privacy-sensitive, organization-specific, or security-sensitive operational context even after processing.

Rows:

- 177,156 labeled threat-log records

Class counts:

- Drop: 160,788
- Deny: 10,187
- Reset-Both: 2,903
- Allow: 2,502
- Reset-Server: 776

Privacy note:

The package excludes raw enterprise log exports and excludes the event-level processed CSV by default. Non-missing `Rule` values in the controlled processed file were converted to stable anonymous rule-context identifiers during internal reproduction, but that anonymization alone is not treated as sufficient authorization for public redistribution. Public data release should occur only after the relevant institutional authority approves the exact file and any required data-use agreement.

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

Strengthening checks:

- Feature-group ablation results are in `results/strengthening/`.
- Chronological core macro-F1 in the feature-group audit: 0.896234.
- Chronological no-threat-descriptors macro-F1 in the feature-group audit: 0.834877.
- Chronological no-application-context macro-F1: 0.959462.
- Confidence-based referral on chronological core retains 97.37% of records with 9 retained-set errors and 0.982071 macro-F1.
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
- In the stratified core setting, a review queue triggered by reset-related records or the lowest 1% confidence records routed 936 of 35,432 test records (2.64%) to review and contained all 3 model-log disagreements.
- In the no-threat-descriptors setting, the same review rule routed 891 records (2.51%) and contained all 13 model-log disagreements.
- A broader rule using model-log disagreement, lowest 5% confidence, or reset-related records routed 1,779 core records (5.02%) and 2,028 no-threat-descriptors records (5.72%).
- The exploratory unseen-context holdout audit shows that performance can deteriorate sharply when entire high-support categorical values are absent from training. Treat this as a deployment-boundary stress test, not as independent external validation.

## Experimental Environment

Final timing-sensitive experiments were conducted on:

- Oracle Cloud Ampere virtual private server
- 4 OCPU
- 24 GB RAM
- Python 3.12.3
- Linux aarch64

The same VPS environment was used for the final benchmark, ablation, and cross-validation results in `results/`.

## Reproducing With Authorized Processed Data

The scripts below require the controlled processed dataset at `data/processed/threat_five_class.csv`. That file is not included in the public repository candidate by default. Use these commands only after placing an institutionally approved copy of the processed dataset at that path.

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

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
python code/09_operational_review_and_context_audit.py --data data/processed/threat_five_class.csv --outdir results/operational_review_context_audit
```

## Submission And Public Release Checklist

Before linking this repository in a manuscript submission:

- Confirm that the GitHub URL matches the public repository.
- Update the target journal, submission status, DOI if available, and citation text.
- Confirm that `data/processed/` does not contain event-level firewall CSV files unless institutional public-release authorization has been obtained.
- Confirm that no raw enterprise logs, manuscript drafts, private review notes, or local credentials are included.
- Check whether the target journal requires a data availability statement, code availability statement, or repository DOI.
- Consider archiving the final GitHub release on Zenodo and adding the DOI here.

## License

No license has been assigned yet. Choose an appropriate code and data license before public release.
