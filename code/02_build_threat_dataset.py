"""Build the controlled five-class threat-log dataset used by LEAF.

The script records aggregate step counts and checksums but never copies the raw
or full processed event-level data into the public repository. Supply explicit
paths that are outside the repository for controlled input and output files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


AUDIT_COLUMNS = ["Receive Time", "Generate Time", "High Res Timestamp", "Type"]
FEATURES = [
    "Threat/Content Type", "Application", "Source Zone", "Destination Zone",
    "Inbound Interface", "Outbound Interface", "IP Protocol", "Source Port",
    "Destination Port", "Source Country", "Destination Country",
    "Threat/Content Name", "Category", "Severity", "Direction",
    "Subcategory of app", "Category of app", "Technology of app", "Risk of app",
    "SaaS of app", "Rule",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Controlled raw threat-log CSV.")
    parser.add_argument("--output", type=Path, required=True, help="Controlled processed CSV.")
    parser.add_argument("--manifest", type=Path, required=True, help="Aggregate JSON manifest path.")
    parser.add_argument("--chunksize", type=int, default=200_000)
    return parser.parse_args()


def normalize(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip().lower()
    return text or None


def map_action(value: object) -> str | None:
    action = normalize(value)
    if action == "allow":
        return "Allow"
    if action == "block":
        return "Block"
    if action in {"drop", "drop-packet", "random-drop"}:
        return "Drop"
    if action == "reset-both":
        return "Reset-Both"
    if action == "reset-server":
        return "Reset-Server"
    return None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def main() -> None:
    args = parse_args()
    repository_root = Path(__file__).resolve().parents[1]
    for controlled_path in (args.input.resolve(), args.output.resolve()):
        try:
            controlled_path.relative_to(repository_root)
        except ValueError:
            pass
        else:
            raise ValueError("Controlled raw and full processed data paths must be outside the public repository")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    usecols = AUDIT_COLUMNS + ["Action"] + FEATURES
    first = True
    rows_in = 0
    rows_out = 0
    excluded_by_normalized_action: dict[str, int] = {}
    class_counts: dict[str, int] = {}

    for chunk in pd.read_csv(
        args.input,
        usecols=lambda column: column in usecols,
        chunksize=args.chunksize,
        low_memory=False,
    ):
        rows_in += len(chunk)
        for column in chunk.columns:
            if pd.api.types.is_object_dtype(chunk[column]) or pd.api.types.is_string_dtype(chunk[column]):
                chunk[column] = chunk[column].astype("string").str.strip().replace({"": pd.NA, "<NA>": pd.NA})
        chunk["target"] = chunk["Action"].map(map_action)
        excluded = chunk.loc[chunk["target"].isna(), "Action"].map(normalize).fillna("__MISSING__")
        for action, count in excluded.value_counts().items():
            excluded_by_normalized_action[str(action)] = excluded_by_normalized_action.get(str(action), 0) + int(count)
        output = chunk.loc[chunk["target"].notna()].copy()
        output["raw_action"] = output["Action"]
        ordered = ["target", "raw_action"] + AUDIT_COLUMNS + FEATURES
        output = output[[column for column in ordered if column in output.columns]]
        rows_out += len(output)
        for label, count in output["target"].value_counts().items():
            class_counts[str(label)] = class_counts.get(str(label), 0) + int(count)
        output.to_csv(args.output, index=False, mode="w" if first else "a", header=first, encoding="utf-8")
        first = False

    manifest = {
        "source_scope": "controlled raw threat-log CSV; not redistributed",
        "rows_in": rows_in,
        "normalization": "strip surrounding whitespace and lowercase action for mapping",
        "excluded_by_normalized_action": dict(sorted(excluded_by_normalized_action.items())),
        "rows_out": rows_out,
        "class_counts": dict(sorted(class_counts.items())),
        "row_level_deduplication": False,
        "additional_row_exclusions": False,
        "processed_sha256": sha256(args.output),
    }
    args.manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
