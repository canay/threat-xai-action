# Operational Review Context Audit Outputs

This directory contains aggregate summaries for the operational review-queue and unseen-context audits.

The detailed event-level review queue file (`operational_review_queue_records.csv`) is intentionally not included in the public artifact. It can expose institutional security posture and event-specific firewall context even after processing. Authorized reruns of `code/09_operational_review_and_context_audit.py` will regenerate that file locally when an approved controlled-access copy of `data/processed/threat_five_class.csv` is available.
