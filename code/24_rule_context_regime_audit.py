"""Aggregate-only Rule-missingness and context-support audit.

This script does not fit a model. It describes where Rule values are missing
and recomputes pooled held-out-context metrics after excluding any context
whose removal eliminates more than 90% of one target class. The latter is a
post-review sensitivity, not a replacement for the seven-context primary
audit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)


MISSING_RULE = "__MISSING_RULE__"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/processed/threat_five_class.csv"),
    )
    parser.add_argument(
        "--heldout-confusions",
        type=Path,
        default=Path(
            "results/policy_context_robustness/"
            "policy_context_heldout_confusions.csv"
        ),
    )
    parser.add_argument(
        "--rule-summary",
        type=Path,
        default=Path(
            "results/policy_context_robustness/"
            "policy_context_rule_summary.csv"
        ),
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("results/reviewer_revision_sensitivity"),
    )
    parser.add_argument("--removal-threshold", type=float, default=0.90)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def missingness_profile(df: pd.DataFrame) -> pd.DataFrame:
    working = df.copy()
    working["target"] = working["target"].replace({"Deny": "Block"})
    working["_rule_missing"] = working["Rule"].isna()
    generated = pd.to_datetime(working["Generate Time"], errors="coerce")
    working["_generate_hour"] = generated.dt.strftime("%H")
    rows: list[dict[str, object]] = []

    def append_scope(dimension: str, value: str, group: pd.DataFrame) -> None:
        missing_rows = int(group["_rule_missing"].sum())
        rows.append(
            {
                "dimension": dimension,
                "value": value,
                "rows": int(len(group)),
                "missing_rule_rows": missing_rows,
                "missing_rule_fraction": float(missing_rows / len(group)),
            }
        )

    append_scope("overall", "all_rows", working)
    for source_column, dimension in [
        ("target", "target"),
        ("Threat/Content Type", "threat_content_type"),
        ("_generate_hour", "generate_hour"),
    ]:
        for value, group in working.groupby(source_column, dropna=False, sort=True):
            append_scope(dimension, str(value), group)
    return pd.DataFrame(rows)


def retained_contexts(
    rule_summary: pd.DataFrame,
    removal_threshold: float,
) -> tuple[set[str], list[dict[str, object]]]:
    class_columns = [
        column for column in rule_summary.columns if column.startswith("class_")
    ]
    totals = rule_summary[class_columns].sum(axis=0)
    eligible = rule_summary.loc[
        (~rule_summary["is_missing_rule"].astype(bool))
        & (rule_summary["records"] >= 50)
    ].copy()
    retained: set[str] = set()
    diagnostics: list[dict[str, object]] = []
    for _, row in eligible.iterrows():
        removed = {
            column.removeprefix("class_"): (
                float(row[column] / totals[column]) if totals[column] else 0.0
            )
            for column in class_columns
        }
        maximum_class = max(removed, key=removed.get)
        maximum_fraction = removed[maximum_class]
        keep = maximum_fraction <= removal_threshold
        if keep:
            retained.add(str(row["rule_group"]))
        diagnostics.append(
            {
                "rule_group": str(row["rule_group"]),
                "records": int(row["records"]),
                "maximum_removed_class": maximum_class,
                "maximum_removed_fraction": maximum_fraction,
                "retained": keep,
            }
        )
    return retained, diagnostics


def pooled_metrics(
    saved: pd.DataFrame,
    retained: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    named = saved.loc[saved["rule_group"].isin(retained)].copy()
    class_columns = [
        column
        for column in named.columns
        if column not in {"feature_set", "rule_group", "true"}
    ]
    labels = sorted(set(named["true"].dropna().astype(str)) | set(class_columns))
    aggregate_rows: list[dict[str, object]] = []
    class_rows: list[dict[str, object]] = []

    for feature_set, group in named.groupby("feature_set", sort=True):
        matrix = np.zeros((len(labels), len(labels)), dtype=np.int64)
        label_index = {label: index for index, label in enumerate(labels)}
        for _, row in group.iterrows():
            true_label = str(row["true"])
            for predicted_label in class_columns:
                value = row[predicted_label]
                if pd.notna(value):
                    matrix[label_index[true_label], label_index[predicted_label]] += int(
                        value
                    )

        true_ids: list[int] = []
        predicted_ids: list[int] = []
        for true_id in range(len(labels)):
            for predicted_id in range(len(labels)):
                count = int(matrix[true_id, predicted_id])
                if count:
                    true_ids.extend([true_id] * count)
                    predicted_ids.extend([predicted_id] * count)
        y_true = np.asarray(true_ids, dtype=int)
        y_pred = np.asarray(predicted_ids, dtype=int)
        label_ids = np.arange(len(labels))
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=label_ids,
            zero_division=0,
        )
        aggregate_rows.append(
            {
                "feature_set": feature_set,
                "heldout_named_contexts": int(group["rule_group"].nunique()),
                "pooled_rows": int(matrix.sum()),
                "pooled_errors": int(matrix.sum() - np.trace(matrix)),
                "pooled_accuracy": float(accuracy_score(y_true, y_pred)),
                "pooled_balanced_accuracy": float(
                    balanced_accuracy_score(y_true, y_pred)
                ),
                "pooled_macro_f1_all_classes": float(
                    f1_score(
                        y_true,
                        y_pred,
                        labels=label_ids,
                        average="macro",
                        zero_division=0,
                    )
                ),
                "pooled_weighted_f1": float(
                    f1_score(
                        y_true,
                        y_pred,
                        labels=label_ids,
                        average="weighted",
                        zero_division=0,
                    )
                ),
            }
        )
        for label_id, label in enumerate(labels):
            class_rows.append(
                {
                    "feature_set": feature_set,
                    "class": label,
                    "precision": float(precision[label_id]),
                    "recall": float(recall[label_id]),
                    "f1": float(f1[label_id]),
                    "support": int(support[label_id]),
                }
            )
    return pd.DataFrame(aggregate_rows), pd.DataFrame(class_rows)


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(
        args.data,
        usecols=[
            "target",
            "Generate Time",
            "Threat/Content Type",
            "Rule",
        ],
        low_memory=False,
    )
    rule_summary = pd.read_csv(args.rule_summary)
    saved_confusions = pd.read_csv(args.heldout_confusions)

    profile = missingness_profile(df)
    retained, diagnostics = retained_contexts(
        rule_summary,
        args.removal_threshold,
    )
    aggregate, classwise = pooled_metrics(saved_confusions, retained)

    profile.to_csv(args.outdir / "rule_missingness_profile.csv", index=False)
    aggregate.to_csv(
        args.outdir / "heldout_named_context_support_qualified_metrics.csv",
        index=False,
    )
    classwise.to_csv(
        args.outdir / "heldout_named_context_support_qualified_classwise.csv",
        index=False,
    )

    metadata = {
        "analysis": "rule_context_regime_audit",
        "aggregate_outputs_only": True,
        "data_sha256": sha256(args.data),
        "heldout_confusions_sha256": sha256(args.heldout_confusions),
        "rule_summary_sha256": sha256(args.rule_summary),
        "script_sha256": sha256(Path(__file__)),
        "removal_threshold": args.removal_threshold,
        "purpose": (
            "post-review sensitivity that separates the fold removing over 90% "
            "of one class from the remaining held-out-context folds"
        ),
        "release_label_normalization": "Deny is reported as Block; class membership is unchanged",
        "retained_contexts": sorted(retained),
        "context_diagnostics": diagnostics,
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    (args.outdir / "rule_context_regime_audit_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    print(profile.to_string(index=False))
    print(aggregate.to_string(index=False))


if __name__ == "__main__":
    main()
