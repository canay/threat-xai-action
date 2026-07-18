from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path


DEFAULT_COUNTS = {
    "Drop": 731,
    "Block": 93,
    "Reset-Both": 62,
    "Allow": 61,
    "Reset-Server": 53,
}

TIMESTAMP_COLUMNS = {"Receive Time", "Generate Time", "High Res Timestamp"}
REDACTED_CONSTANTS = {
    "raw_action": "redacted_action",
    "Type": "redacted_type",
}

PREFIX_BY_COLUMN = {
    "Threat/Content Type": "threat_type",
    "Application": "application",
    "Source Zone": "source_zone",
    "Destination Zone": "destination_zone",
    "Inbound Interface": "inbound_interface",
    "Outbound Interface": "outbound_interface",
    "IP Protocol": "ip_protocol",
    "Source Port": "source_port",
    "Destination Port": "destination_port",
    "Source Country": "source_country",
    "Destination Country": "destination_country",
    "Threat/Content Name": "threat_name",
    "Category": "category",
    "Severity": "severity",
    "Direction": "direction",
    "Subcategory of app": "app_subcategory",
    "Category of app": "app_category",
    "Technology of app": "app_technology",
    "Risk of app": "app_risk",
    "SaaS of app": "app_saas",
    "Rule": "rule_context",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def normalize_missing(value: str | None) -> str:
    if value is None:
        return "missing"
    stripped = value.strip()
    if not stripped or stripped.lower() in {"nan", "none", "null"}:
        return "missing"
    return stripped


def make_value_maps(rows: list[dict[str, str]], fieldnames: list[str]) -> dict[str, dict[str, str]]:
    maps: dict[str, dict[str, str]] = {}
    for column in fieldnames:
        if column in {"target"} or column in TIMESTAMP_COLUMNS or column in REDACTED_CONSTANTS:
            continue
        prefix = PREFIX_BY_COLUMN.get(column, column.lower().replace("/", "_").replace(" ", "_"))
        values = sorted({normalize_missing(row.get(column, "")) for row in rows})
        column_map: dict[str, str] = {}
        counter = 1
        for value in values:
            if value == "missing":
                column_map[value] = f"{prefix}_missing"
            else:
                column_map[value] = f"{prefix}_{counter:03d}"
                counter += 1
        maps[column] = column_map
    return maps


def anonymize_rows(rows: list[dict[str, str]], fieldnames: list[str]) -> list[dict[str, str]]:
    value_maps = make_value_maps(rows, fieldnames)
    anonymized: list[dict[str, str]] = []
    base = datetime(2026, 1, 1, 0, 0, 0)
    for i, row in enumerate(rows):
        output: dict[str, str] = {}
        synthetic_time = base + timedelta(seconds=i)
        for column in fieldnames:
            if column == "target":
                output[column] = row[column]
            elif column in TIMESTAMP_COLUMNS:
                output[column] = synthetic_time.strftime("%Y-%m-%d %H:%M:%S")
            elif column in REDACTED_CONSTANTS:
                output[column] = REDACTED_CONSTANTS[column]
            else:
                value = normalize_missing(row.get(column, ""))
                output[column] = value_maps[column][value]
        anonymized.append(output)
    return anonymized


def load_rows_by_target(path: Path) -> tuple[list[str], dict[str, list[tuple[int, dict[str, str]]]], Counter]:
    by_target: dict[str, list[tuple[int, dict[str, str]]]] = defaultdict(list)
    counts: Counter = Counter()
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise SystemExit(f"No header found in {path}")
        fieldnames = list(reader.fieldnames)
        for index, row in enumerate(reader):
            target = row.get("target", "")
            counts[target] += 1
            by_target[target].append((index, row))
    return fieldnames, by_target, counts


def sample_rows(
    by_target: dict[str, list[tuple[int, dict[str, str]]]],
    requested_counts: dict[str, int],
    seed: int,
) -> list[tuple[int, dict[str, str]]]:
    rng = random.Random(seed)
    sampled: list[tuple[int, dict[str, str]]] = []
    for target, n in requested_counts.items():
        available = by_target.get(target, [])
        if len(available) < n:
            raise SystemExit(f"Requested {n} rows for {target}, but only {len(available)} are available")
        sampled.extend(rng.sample(available, n))
    rng.shuffle(sampled)
    return sampled


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_audit(
    path: Path,
    source: Path,
    output: Path,
    fieldnames: list[str],
    source_counts: Counter,
    sample_counts: Counter,
    seed: int,
) -> None:
    audit = {
        "purpose": "public smoke-test sample; not used for reported manuscript metrics",
        "source_identifier": source.name,
        "source_sha256": sha256_file(source),
        "output_identifier": output.name,
        "output_sha256": sha256_file(output),
        "seed": seed,
        "rows": sum(sample_counts.values()),
        "header_columns": fieldnames,
        "source_class_counts": dict(source_counts),
        "sample_class_counts": dict(sample_counts),
        "anonymization": {
            "target": "kept to preserve five-class script behavior",
            "raw_action": "replaced with a constant redacted token",
            "timestamps": "replaced with synthetic sequential timestamps on 2026-01-01",
            "event_fields": "mapped within each column to stable generic codes such as application_001",
            "rule": "mapped to generic rule_context codes; raw policy names are not released",
            "row_identifiers": "no source row id or original order identifier is released; sampled rows are shuffled before synthetic timestamps are assigned",
        },
        "release_boundary": (
            "This sample is suitable for schema inspection and smoke tests only. "
            "It does not reproduce the reported manuscript metrics."
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a public anonymized smoke-test sample.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260701)
    args = parser.parse_args()

    fieldnames, by_target, source_counts = load_rows_by_target(args.input)
    sampled = sample_rows(by_target, DEFAULT_COUNTS, args.seed)
    rows = [row for _, row in sampled]
    anonymized = anonymize_rows(rows, fieldnames)
    write_csv(args.output, fieldnames, anonymized)
    sample_counts = Counter(row["target"] for row in anonymized)
    write_audit(args.audit, args.input, args.output, fieldnames, source_counts, sample_counts, args.seed)

    print(json.dumps({
        "output": str(args.output),
        "audit": str(args.audit),
        "rows": len(anonymized),
        "sha256": sha256_file(args.output),
        "sample_class_counts": dict(sample_counts),
    }, indent=2))


if __name__ == "__main__":
    main()
