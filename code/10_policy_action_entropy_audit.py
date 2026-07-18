"""Model-independent policy-action entropy audit for firewall enforcement logs.

The classification experiments treat models as measurement instruments. This
script adds a non-model audit layer: it quantifies how deterministic the
observed firewall action is inside anonymized policy contexts and selected
context refinements. Low entropy supports the policy-conditioned interpretation
of the target, while low-purity contexts identify places where a simple
policy-path lookup is not sufficient and analyst review remains relevant.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


MISSING_RULE = "__MISSING_RULE__"
LABEL_ORDER = ["Allow", "Block", "Drop", "Reset-Both", "Reset-Server"]
REPO_ROOT = Path(__file__).resolve().parent.parent

AUDIT_LEVELS = {
    "rule_context": ["Rule Group"],
    "rule_context_plus_threat_type": ["Rule Group", "Threat/Content Type"],
    "rule_context_plus_direction": ["Rule Group", "Direction"],
    "rule_context_plus_application": ["Rule Group", "Application"],
    "rule_context_plus_threat_type_direction": ["Rule Group", "Threat/Content Type", "Direction"],
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/processed/threat_five_class.csv"))
    parser.add_argument("--outdir", type=Path, default=Path("results/policy_action_entropy_audit"))
    parser.add_argument(
        "--no-anonymize-rules",
        action="store_true",
        help="Write raw Rule values instead of stable anonymous rule_context_* identifiers.",
    )
    parser.add_argument("--low-purity-threshold", type=float, default=0.80)
    return parser.parse_args()


def entropy_bits(counts: pd.Series) -> float:
    values = counts.to_numpy(dtype=float)
    values = values[values > 0]
    if values.size == 0:
        return float("nan")
    probabilities = values / values.sum()
    return float(-(probabilities * np.log2(probabilities)).sum())


def add_rule_groups(df: pd.DataFrame, anonymize_rules: bool) -> tuple[pd.DataFrame, dict[str, str]]:
    result = df.copy()
    raw_rule = result["Rule"].astype("string").fillna(MISSING_RULE).astype(str)
    result["Rule Group"] = raw_rule
    if not anonymize_rules:
        return result, {}

    non_missing = raw_rule[raw_rule.ne(MISSING_RULE)]
    counts = non_missing.value_counts()
    ordered_rules = sorted(counts.index, key=lambda value: (-int(counts[value]), str(value)))
    mapping = {rule: f"rule_context_{idx:02d}" for idx, rule in enumerate(ordered_rules, start=1)}
    result["Rule Group"] = raw_rule.map(lambda value: MISSING_RULE if value == MISSING_RULE else mapping[value])
    return result, mapping


def label_columns(df: pd.DataFrame) -> list[str]:
    observed = set(df["target"].astype(str).unique())
    ordered = [label for label in LABEL_ORDER if label in observed]
    ordered.extend(sorted(observed - set(ordered)))
    return ordered


def summarize_groups(
    df: pd.DataFrame,
    key_columns: list[str],
    level_name: str,
    labels: list[str],
    low_purity_threshold: float,
) -> tuple[dict[str, object], pd.DataFrame]:
    max_entropy = math.log2(len(labels))
    rows: list[dict[str, object]] = []

    for key, group in df.groupby(key_columns, dropna=False):
        counts = group["target"].astype(str).value_counts()
        records = int(counts.sum())
        entropy = entropy_bits(counts)
        dominant_class = str(counts.idxmax())
        dominant_count = int(counts.max())
        row: dict[str, object] = {
            "audit_level": level_name,
            "context_key": " | ".join(map(str, key if isinstance(key, tuple) else (key,))),
            "records": records,
            "nonzero_classes": int((counts > 0).sum()),
            "dominant_class": dominant_class,
            "dominant_count": dominant_count,
            "dominant_fraction": float(dominant_count / records),
            "entropy_bits": entropy,
            "normalized_entropy": float(entropy / max_entropy) if max_entropy else float("nan"),
            "records_not_in_dominant_class": int(records - dominant_count),
        }
        row.update({f"class_{label}": int(counts.get(label, 0)) for label in labels})
        rows.append(row)

    detail = pd.DataFrame(rows).sort_values(["audit_level", "records", "context_key"], ascending=[True, False, True])
    mixed = detail["nonzero_classes"] > 1
    low_purity = detail["dominant_fraction"] < low_purity_threshold
    weights = detail["records"].to_numpy(dtype=float)
    summary = {
        "audit_level": level_name,
        "key_columns": " + ".join(key_columns),
        "contexts": int(len(detail)),
        "records": int(detail["records"].sum()),
        "mixed_contexts": int(mixed.sum()),
        "mixed_context_fraction": float(mixed.mean()),
        "records_in_mixed_contexts": int(detail.loc[mixed, "records"].sum()),
        "low_purity_threshold": float(low_purity_threshold),
        "low_purity_contexts": int(low_purity.sum()),
        "records_in_low_purity_contexts": int(detail.loc[low_purity, "records"].sum()),
        "record_weighted_mean_entropy_bits": float(np.average(detail["entropy_bits"], weights=weights)),
        "record_weighted_normalized_entropy": float(np.average(detail["normalized_entropy"], weights=weights)),
        "record_weighted_dominant_fraction": float(np.average(detail["dominant_fraction"], weights=weights)),
        "record_weighted_majority_error_floor": float(
            np.average(detail["records_not_in_dominant_class"] / detail["records"], weights=weights)
        ),
        "median_context_dominant_fraction": float(detail["dominant_fraction"].median()),
    }
    return summary, detail


def main() -> None:
    args = parse_args()
    if args.no_anonymize_rules and args.outdir.resolve().is_relative_to(REPO_ROOT):
        raise SystemExit(
            "Non-anonymized rule outputs must be written outside the public repository."
        )
    args.outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data, low_memory=False)
    df["target"] = df["target"].astype(str)
    df, rule_mapping = add_rule_groups(df, anonymize_rules=not args.no_anonymize_rules)
    labels = label_columns(df)

    summaries = []
    details = {}
    for level_name, columns in AUDIT_LEVELS.items():
        summary, detail = summarize_groups(df, columns, level_name, labels, args.low_purity_threshold)
        summaries.append(summary)
        details[level_name] = detail

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(args.outdir / "policy_action_entropy_summary.csv", index=False)
    details["rule_context"].drop(columns=["audit_level"]).to_csv(
        args.outdir / "policy_action_rule_purity.csv", index=False
    )

    metadata = {
        "data_identifier": args.data.name,
        "data_sha256": sha256(args.data),
        "rows": int(len(df)),
        "target_labels": labels,
        "anonymize_rules": not args.no_anonymize_rules,
        "non_missing_rule_contexts": int(len(rule_mapping)) if rule_mapping else int(df["Rule Group"].nunique()),
        "missing_rule_records": int(df["Rule Group"].eq(MISSING_RULE).sum()),
        "low_purity_threshold": float(args.low_purity_threshold),
        "outputs": [
            "policy_action_entropy_summary.csv",
            "policy_action_rule_purity.csv",
        ],
        "note": (
            "Rule values are anonymized by default. This is a model-independent audit of "
            "policy-conditioned action regularity, not a classifier benchmark."
        ),
    }
    (args.outdir / "policy_action_entropy_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
