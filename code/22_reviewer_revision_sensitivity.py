"""Reviewer-requested sensitivity checks for the firewall case study.

This script performs two bounded checks without changing the published event
labels or the original train/test split:

1. It compares the selected XGBoost model under the original most-frequent
   categorical imputation with a variant that represents missing ``Rule``
   values as an explicit category.
2. It pools the saved confusion counts from the seven eligible named-rule
   exclusion tests and reports standard classwise and macro F1 scores.

The script stores aggregate metrics only. It does not export event-level
predictions or enterprise rule names.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from xgboost import XGBClassifier


EXCLUDE_ALWAYS = {
    "target",
    "raw_action",
    "raw_traffic_subtype",
    "raw_session_end_reason",
    "Receive Time",
    "Generate Time",
    "High Res Timestamp",
    "Type",
    "Session ID",
}

POLICY_FEATURES = {"Rule", "Action Source"}
MISSING_RULE = "__MISSING_RULE__"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/processed/threat_five_class.csv"))
    parser.add_argument(
        "--heldout-confusions",
        type=Path,
        default=Path(
            "github-threat-xai-action/results/policy_context_robustness/"
            "policy_context_heldout_confusions.csv"
        ),
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("github-threat-xai-action/results/reviewer_revision_sensitivity"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=6)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def feature_columns(df: pd.DataFrame, include_policy: bool) -> list[str]:
    excluded = set(EXCLUDE_ALWAYS)
    if not include_policy:
        excluded |= POLICY_FEATURES
    return [column for column in df.columns if column not in excluded]


def build_pipeline(X: pd.DataFrame, args: argparse.Namespace) -> Pipeline:
    categorical = [column for column in X.columns if not pd.api.types.is_numeric_dtype(X[column])]
    numeric = [column for column in X.columns if column not in categorical]
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                        ),
                    ]
                ),
                categorical,
            ),
            ("num", Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]), numeric),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    model = XGBClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=args.seed,
        n_jobs=-1,
    )
    return Pipeline(steps=[("preprocess", preprocessor), ("model", model)])


def metric_rows(
    configuration: str,
    labels: list[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    rule_missing: np.ndarray,
) -> list[dict[str, object]]:
    all_label_ids = np.arange(len(labels))
    scopes = {
        "all_test_rows": np.ones(len(y_true), dtype=bool),
        "named_rule_test_rows": ~rule_missing,
        "missing_rule_test_rows": rule_missing,
    }
    rows: list[dict[str, object]] = []
    for scope, mask in scopes.items():
        scoped_true = y_true[mask]
        scoped_pred = y_pred[mask]
        observed_ids = np.unique(scoped_true)
        rows.append(
            {
                "configuration": configuration,
                "scope": scope,
                "rows": int(mask.sum()),
                "observed_classes": ";".join(labels[label_id] for label_id in observed_ids),
                "accuracy": float(accuracy_score(scoped_true, scoped_pred)),
                "balanced_accuracy": float(balanced_accuracy_score(scoped_true, scoped_pred)),
                "macro_f1_observed_classes": float(
                    f1_score(
                        scoped_true,
                        scoped_pred,
                        average="macro",
                        labels=observed_ids,
                        zero_division=0,
                    )
                ),
                "macro_f1_all_classes": float(
                    f1_score(
                        scoped_true,
                        scoped_pred,
                        average="macro",
                        labels=all_label_ids,
                        zero_division=0,
                    )
                ),
                "weighted_f1": float(
                    f1_score(scoped_true, scoped_pred, average="weighted", zero_division=0)
                ),
                "errors": int((scoped_true != scoped_pred).sum()),
            }
        )
    return rows


def run_rule_missingness_sensitivity(
    df: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    target = df["target"].astype(str).replace({"Deny": "Block"})
    encoder = LabelEncoder()
    y = encoder.fit_transform(target)
    labels = list(encoder.classes_)
    indices = np.arange(len(df))
    train_indices, test_indices = train_test_split(
        indices,
        test_size=0.2,
        random_state=args.seed,
        stratify=y,
    )
    raw_rule_missing = df["Rule"].isna().to_numpy()

    configurations: list[tuple[str, pd.DataFrame]] = []
    core_columns = feature_columns(df, include_policy=False)
    configurations.append(("core_without_policy_fields", df[core_columns].copy()))

    policy_columns = feature_columns(df, include_policy=True)
    configurations.append(
        ("with_policy_most_frequent_rule_imputation", df[policy_columns].copy())
    )
    explicit_missing = df[policy_columns].copy()
    explicit_missing["Rule"] = explicit_missing["Rule"].astype("string").fillna(MISSING_RULE)
    configurations.append(("with_policy_explicit_missing_rule", explicit_missing))

    metric_output: list[dict[str, object]] = []
    confusion_output: list[dict[str, object]] = []
    timing: dict[str, float] = {}
    all_label_ids = np.arange(len(labels))

    for configuration, X in configurations:
        pipeline = build_pipeline(X, args)
        started = time.perf_counter()
        pipeline.fit(X.iloc[train_indices], y[train_indices])
        predictions = pipeline.predict(X.iloc[test_indices])
        timing[configuration] = float(time.perf_counter() - started)
        metric_output.extend(
            metric_rows(
                configuration=configuration,
                labels=labels,
                y_true=y[test_indices],
                y_pred=predictions,
                rule_missing=raw_rule_missing[test_indices],
            )
        )
        matrix = confusion_matrix(y[test_indices], predictions, labels=all_label_ids)
        for true_id, true_label in enumerate(labels):
            for predicted_id, predicted_label in enumerate(labels):
                confusion_output.append(
                    {
                        "configuration": configuration,
                        "true": true_label,
                        "predicted": predicted_label,
                        "count": int(matrix[true_id, predicted_id]),
                    }
                )

    metadata = {
        "rows": int(len(df)),
        "train_rows": int(len(train_indices)),
        "test_rows": int(len(test_indices)),
        "named_rule_test_rows": int((~raw_rule_missing[test_indices]).sum()),
        "missing_rule_test_rows": int(raw_rule_missing[test_indices].sum()),
        "labels": labels,
        "seed": args.seed,
        "test_size": 0.2,
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "learning_rate": 0.08,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "tree_method": "hist",
        "fit_predict_seconds": timing,
    }
    return pd.DataFrame(metric_output), pd.DataFrame(confusion_output), metadata


def pooled_metrics_from_saved_confusions(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    saved = pd.read_csv(path)
    named = saved.loc[~saved["rule_group"].eq(MISSING_RULE)].copy()
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
                    matrix[label_index[true_label], label_index[predicted_label]] += int(value)

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
                    f1_score(y_true, y_pred, labels=label_ids, average="macro", zero_division=0)
                ),
                "pooled_weighted_f1": float(
                    f1_score(y_true, y_pred, labels=label_ids, average="weighted", zero_division=0)
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
    wall_started = time.perf_counter()
    args.outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.data, low_memory=False)

    sensitivity, sensitivity_confusions, sensitivity_metadata = (
        run_rule_missingness_sensitivity(df, args)
    )
    pooled, pooled_classwise = pooled_metrics_from_saved_confusions(
        args.heldout_confusions
    )

    sensitivity.to_csv(args.outdir / "rule_missingness_sensitivity.csv", index=False)
    sensitivity_confusions.to_csv(
        args.outdir / "rule_missingness_sensitivity_confusions.csv", index=False
    )
    pooled.to_csv(args.outdir / "heldout_named_context_pooled_metrics.csv", index=False)
    pooled_classwise.to_csv(
        args.outdir / "heldout_named_context_pooled_classwise.csv", index=False
    )

    metadata = {
        "analysis": "reviewer_revision_sensitivity",
        "aggregate_outputs_only": True,
        "data_path": str(args.data),
        "data_sha256": sha256(args.data),
        "heldout_confusions_path": str(args.heldout_confusions),
        "heldout_confusions_sha256": sha256(args.heldout_confusions),
        "script_sha256": sha256(Path(__file__)),
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "wall_seconds": float(time.perf_counter() - wall_started),
        },
        "rule_missingness_sensitivity": sensitivity_metadata,
    }
    (args.outdir / "reviewer_revision_sensitivity_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print(sensitivity.to_string(index=False))
    print(pooled.to_string(index=False))
    print(f"Wrote aggregate outputs to {args.outdir}")


if __name__ == "__main__":
    main()
