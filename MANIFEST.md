# Replication Package Manifest

Artifact short name: `threat-xai-action`

Repository: `https://github.com/canay/threat-xai-action`

Manuscript: "LEAF: A Leakage-Aware, Explainable Audit Framework for Policy-Conditioned Firewall Enforcement"

## Public Contents

- `code/`: the controlled-data conversion entry point and every benchmark, validation, explainability, policy-context, review-queue, entropy, revision, and manuscript-figure script used by the manuscript (`02`--`20`).
- `scripts/create_public_sample.py`: the only retained helper script; it creates the privacy-audited public smoke-test sample from an authorized local copy.
- `results/`: aggregate CSV/JSON outputs and generated explanation artifacts used for manuscript reporting.
- `data/processed/README.md`: controlled-data access note.
- `data/processed/schema.csv`: column-level schema and release notes.
- `data/processed/public_anonymized_sample_1000.csv`: 1,000-row anonymized public smoke-test sample, not used for reported metrics.
- `data/processed/public_anonymized_sample_1000.audit.json`: sample generation, checksum, class-count, and anonymization audit record.
- `data/processed/threat_dataset_processing_manifest.json`: aggregate raw-to-processed step counts, class counts, observation scope, and processed-file checksum.
- `TRACEABILITY.md`: table/figure-to-command/output mapping and controlled processed-file checksum for authorized reruns.
- `SEED_MANIFEST.json`: explicit random seeds and resampling scope.
- `requirements.txt`: portable minimum compatible versions.
- `requirements-lock-primary-linux-aarch64.txt`: exact complete resolution for the canonical Linux/aarch64 VPS evidence.
- `requirements-lock.txt`: platform-labelled Windows x86-64 diagnostic environment retained for provenance only.
- `CITATION.cff` and `LICENSE`: citation metadata and software/artifact license.

## Controlled Or Excluded Contents

- Raw enterprise firewall exports are not included.
- The full event-level processed file `data/processed/threat_five_class.csv` is not included.
- Event-level review-queue records are not included. `code/09` writes aggregate outputs by default and refuses to place an optional private queue inside this repository.
- Manuscript drafts, private review notes, credentials, and local environment files are not part of the public package.

Controlled processed file identity for authorized reruns:

- Expected path: `data/processed/threat_five_class.csv`
- Event rows: 177,156
- Header rows: 1
- Size: 44,929,133 bytes
- SHA-256: `1BE9896A996DD58A582D94319180405A8559AB5193EF692BD8D2D2D614693724`

## Reproduction Contract

The public package supports inspection of code, command recipes, aggregate outputs, validation summaries, traceability notes, explanation artifacts, and a 1,000-row anonymized smoke-test sample. A full rerun requires an institutionally approved copy of `data/processed/threat_five_class.csv` placed at the expected path. The public sample is suitable for schema and parser checks only and must not be used to reproduce the manuscript metrics. The calendar date, 14 May 2026, is intentionally disclosed with author approval; organization identity, topology, policy names, raw exports, full event-level data, and event-level review queues remain excluded.

The public benchmark CSV/JSON artifacts preserve the original Linux/aarch64 VPS evidence. The registered dependent evidence bundle `2026-07-13_codex_vps_selected_model_canonicalization` first reproduced the XGBoost core and no-threat-descriptors holdouts exactly, then regenerated the model-dependent validation, explanation, grouped-split, policy-context, review-queue, seed, bootstrap, encoding-sensitivity, and public UCI schema-transfer aggregates under the same locked environment. Public policy-context outputs replace enterprise rule names with deterministic support-ranked aliases. The Windows rerun is retained only as historical diagnostic provenance.

Before public archival release:

1. Confirm that no raw or full event-level controlled firewall files are present.
2. Run the command recipes in `README.MD` against an authorized processed dataset.
3. Confirm that aggregate outputs in `results/` match the manuscript tables and figures listed in `TRACEABILITY.md`.
4. Confirm that `data/processed/public_anonymized_sample_1000.csv` matches its audit checksum and contains only anonymized sample values.
5. Record the final Git commit and release tag in this manifest; a repository DOI is intentionally deferred until the submission-stage Zenodo decision.
6. Archive only the approved public package.

## Release Identifiers

- Final public commit: resolve with `git rev-parse 'jnca-submission-r0.9^{commit}'`
- Release tag: `jnca-submission-r0.9`
- Preserved prior tags: `jnca-submission-r0`, `jnca-submission-r0.1`, `jnca-submission-r0.2`, `jnca-submission-r0.3`, `jnca-submission-r0.4`, `jnca-submission-r0.5`, `jnca-submission-r0.6`, `jnca-submission-r0.7`, and `jnca-submission-r0.8`
- Repository DOI: `DEFERRED-PENDING-SUBMISSION-DECISION`
