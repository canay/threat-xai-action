"""Audit whether raw session identifiers can support a grouped evaluation.

The output contains aggregate coverage and repetition counts only. No session
identifier, timestamp, address, policy name, or event-level record is written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-data",
        type=Path,
        default=Path("data/restricted_source_data/raw/threat.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "github-threat-xai-action/results/reviewer_revision_sensitivity/"
            "session_identifier_coverage_audit.json"
        ),
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(
        args.raw_data,
        usecols=["Action", "Session ID"],
        low_memory=False,
    )
    normalized_action = df["Action"].astype("string").str.strip().str.lower()
    eligible = normalized_action.ne("alert")
    session = pd.to_numeric(df.loc[eligible, "Session ID"], errors="coerce")
    valid = session.notna() & session.ne(0)
    counts = session.loc[valid].value_counts()
    repeated = counts.loc[counts.gt(1)]

    eligible_rows = int(eligible.sum())
    valid_rows = int(valid.sum())
    output = {
        "analysis": "session_identifier_coverage_audit",
        "aggregate_output_only": True,
        "raw_data_sha256": sha256(args.raw_data),
        "raw_rows": int(len(df)),
        "eligible_non_alert_rows": eligible_rows,
        "zero_or_missing_session_rows": int((~valid).sum()),
        "valid_nonzero_session_rows": valid_rows,
        "valid_nonzero_session_fraction": float(valid_rows / eligible_rows),
        "unique_valid_session_ids": int(counts.size),
        "session_ids_with_multiple_rows": int(repeated.size),
        "rows_in_repeated_session_ids": int(repeated.sum()),
        "maximum_rows_per_valid_session_id": int(counts.max()),
        "interpretation": (
            "The identifier covers too little of the processed benchmark to define "
            "a full session-grouped split without arbitrary pseudo-groups for "
            "zero or missing identifiers."
        ),
        "runtime": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "platform": platform.platform(),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
