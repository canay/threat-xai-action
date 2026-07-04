# Processed Dataset Access

The full event-level processed dataset used in the manuscript is not included in this public artifact. This directory includes a 1,000-row anonymized public smoke-test sample for schema inspection and lightweight script checks.

The original data were derived from institutional firewall logs and may contain privacy-sensitive, organization-specific, or security-sensitive operational context even after processing. The raw firewall exports are not redistributed. The processed event-level CSV should be shared only if the relevant institutional authority confirms that public redistribution is permitted after privacy and security review.

For local reproduction by authorized researchers, place the approved processed file at:

`data/processed/threat_five_class.csv`

The scripts in this package expect that path when rerunning the experiments.

Controlled processed file identity for authorized reruns:

- Event rows: 177,156
- Header rows: 1
- Size: 44,929,133 bytes
- SHA-256: `1BE9896A996DD58A582D94319180405A8559AB5193EF692BD8D2D2D614693724`

Public anonymized smoke-test sample:

- File: `public_anonymized_sample_1000.csv`
- Audit: `public_anonymized_sample_1000.audit.json`
- Rows: 1,000
- SHA-256: `52A00A06A8D9E7FB2AD7055B4504809451F1F22123078536D2FD42866CA91A08`
- Class counts: Drop 731, Deny 93, Reset-Both 62, Allow 61, Reset-Server 53

This sample preserves the 27-column processed-file schema and all five target labels. It is strongly anonymized: `raw_action` is redacted, timestamps are synthetic, and event-level categorical/numeric values are mapped within each column to generic tokens such as `application_001`, `source_zone_001`, and `rule_context_001`. No source row identifier, original timestamp, or original row order is released.

The public sample is not used for the reported manuscript results and will not reproduce the paper's metrics. It is provided only so readers, reviewers, and editors can inspect the expected file shape and smoke-test the public code without access to the controlled dataset.

Public helper files in this directory:

- `schema.csv`: expected columns and release notes.
- `public_anonymized_sample_1000.csv`: anonymized smoke-test sample for parser and workflow checks only.
- `public_anonymized_sample_1000.audit.json`: sample generation and anonymization audit.
- `synthetic_schema_example.csv`: fake schema-compatible rows for inspection only.

The synthetic example is not used for the reported manuscript results.
