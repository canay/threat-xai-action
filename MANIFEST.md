# Replication Package Manifest

Artifact short name: `threat-xai-action`

Repository: `https://github.com/canay/threat-xai-action`

Manuscript: "Auditing Policy-Conditioned Firewall Enforcement: A Leakage-Aware and Explainable Evaluation Protocol for Enterprise Threat Logs"

## Public Contents

- `code/`: benchmark, validation, explainability, policy-context, review-queue, entropy, and Q1 audit revision scripts.
- `scripts/`: paper-level helper scripts retained from the research workspace.
- `results/`: aggregate CSV/JSON outputs and generated explanation artifacts used for manuscript reporting.
- `data/processed/README.md`: controlled-data access note.
- `data/processed/schema.csv`: column-level schema and release notes.
- `data/processed/synthetic_schema_example.csv`: small fake schema-compatible example, not used for reported results.
- `requirements.txt`: Python dependency specification.
- `CITATION.cff` and `LICENSE`: citation metadata and software/artifact license.

## Controlled Or Excluded Contents

- Raw enterprise firewall exports are not included.
- The event-level processed file `data/processed/threat_five_class.csv` is not included.
- Event-level review-queue records are not included by default.
- Manuscript drafts, private review notes, credentials, and local environment files are not part of the public package.

## Reproduction Contract

The public package supports inspection of code, command recipes, aggregate outputs, validation summaries, and explanation artifacts. A full rerun requires an institutionally approved copy of `data/processed/threat_five_class.csv` placed at the expected path.

Before public archival release:

1. Confirm that no raw or event-level controlled firewall files are present.
2. Run the command recipes in `README.MD` against an authorized processed dataset.
3. Confirm that aggregate outputs in `results/` match the manuscript tables and figures.
4. Record the final Git commit, release tag, and optional repository DOI in this manifest.
5. Archive only the approved public package.

## Release Identifiers

- Final public commit: `TO-BE-FILLED-AT-RELEASE`
- Release tag: `TO-BE-FILLED-AT-RELEASE`
- Repository DOI: `TO-BE-FILLED-IF-ARCHIVED`
