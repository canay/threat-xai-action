# Operational Review Context Audit Outputs

This directory contains aggregate summaries for the operational review-queue and unseen-context audits.

The detailed event-level review queue is intentionally not included. The script's public default is aggregate-only, and public context values are deterministic aliases rather than raw operational values. An authorized rerun writes event-level records only when `--private-records-out <path>` is supplied; that path must resolve outside the public repository or the script stops.
