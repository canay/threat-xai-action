# Replication Package Manifest

Artifact short name: `threat-xai-action`

Repository: `https://github.com/canay/threat-xai-action`

Manuscript: "LEAF: A Leakage-Aware, Explainable Audit Framework for Policy-Conditioned Firewall Enforcement"

## Public Contents

- `code/`: benchmark, validation, explainability, policy-context, review-queue, entropy, and Q1 audit revision scripts.
- `scripts/`: paper-level helper scripts retained from the research workspace.
- `results/`: aggregate CSV/JSON outputs and generated explanation artifacts used for manuscript reporting.
- `data/processed/README.md`: controlled-data access note.
- `data/processed/schema.csv`: column-level schema and release notes.
- `data/processed/public_anonymized_sample_1000.csv`: 1,000-row anonymized public smoke-test sample, not used for reported metrics.
- `data/processed/public_anonymized_sample_1000.audit.json`: sample generation, checksum, class-count, and anonymization audit record.
- `data/processed/synthetic_schema_example.csv`: small fake schema-compatible example, not used for reported results.
- `scripts/create_public_sample.py`: reproducible helper used to create the public anonymized smoke-test sample from an authorized local copy.
- `TRACEABILITY.md`: table/figure-to-command/output mapping and controlled processed-file checksum for authorized reruns.
- `requirements.txt`: Python dependency specification.
- `CITATION.cff` and `LICENSE`: citation metadata and software/artifact license.

## Controlled Or Excluded Contents

- Raw enterprise firewall exports are not included.
- The full event-level processed file `data/processed/threat_five_class.csv` is not included.
- Event-level review-queue records are not included by default.
- Manuscript drafts, private review notes, credentials, and local environment files are not part of the public package.

Controlled processed file identity for authorized reruns:

- Expected path: `data/processed/threat_five_class.csv`
- Event rows: 177,156
- Header rows: 1
- Size: 44,929,133 bytes
- SHA-256: `1BE9896A996DD58A582D94319180405A8559AB5193EF692BD8D2D2D614693724`

## Reproduction Contract

The public package supports inspection of code, command recipes, aggregate outputs, validation summaries, traceability notes, explanation artifacts, and a 1,000-row anonymized smoke-test sample. A full rerun requires an institutionally approved copy of `data/processed/threat_five_class.csv` placed at the expected path. The public sample is suitable for schema and parser checks only and must not be used to reproduce the manuscript metrics.

Before public archival release:

1. Confirm that no raw or full event-level controlled firewall files are present.
2. Run the command recipes in `README.MD` against an authorized processed dataset.
3. Confirm that aggregate outputs in `results/` match the manuscript tables and figures listed in `TRACEABILITY.md`.
4. Confirm that `data/processed/public_anonymized_sample_1000.csv` matches its audit checksum and contains only anonymized sample values.
5. Record the final Git commit, release tag, and optional repository DOI in this manifest.
6. Archive only the approved public package.

## Release Identifiers

- Final public commit: `TO-BE-FILLED-AT-RELEASE`
- Release tag: `TO-BE-FILLED-AT-RELEASE`
- Repository DOI: `TO-BE-FILLED-IF-ARCHIVED`
