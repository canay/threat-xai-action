# Processed Dataset Access

The event-level processed dataset used in the manuscript is not included in this public artifact.

The original data were derived from institutional firewall logs and may contain privacy-sensitive, organization-specific, or security-sensitive operational context even after processing. The raw firewall exports are not redistributed. The processed event-level CSV should be shared only if the relevant institutional authority confirms that public redistribution is permitted after privacy and security review.

For local reproduction by authorized researchers, place the approved processed file at:

`data/processed/threat_five_class.csv`

The scripts in this package expect that path when rerunning the experiments.

Public helper files in this directory:

- `schema.csv`: expected columns and release notes.
- `synthetic_schema_example.csv`: fake schema-compatible rows for inspection only.

The synthetic example is not used for the reported manuscript results.
