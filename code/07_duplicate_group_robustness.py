"""Duplicate-aware robustness checks for the firewall action manuscript.

The main random split can place identical structured feature vectors in both
train and test partitions. This script tests how much of the result survives
when exact feature signatures are kept on only one side of a grouped split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from xgboost import XGBClassifier


CORE_FEATURES = [
    "Threat/Content Type",
    "Application",
    "Source Zone",
    "Destination Zone",
    "Inbound Interface",
    "Outbound Interface",
    "IP Protocol",
    "Source Port",
    "Destination Port",
    "Source Country",
    "Destination Country",
    "Threat/Content Name",
    "Category",
    "Severity",
    "Direction",
    "Subcategory of app",
    "Category of app",
    "Technology of app",
    "Risk of app",
    "SaaS of app",
]

FEATURE_SETS = {
    "core": CORE_FEATURES,
    "no_threat_descriptors": [
        c for c in CORE_FEATURES if c not in {"Threat/Content Type", "Threat/Content Name", "Severity"}
    ],
    "minimal_context": [
        "Application",
        "Source Zone",
        "Destination Zone",
        "Inbound Interface",
        "Outbound Interface",
        "IP Protocol",
        "Source Port",
        "Destination Port",
        "Source Country",
        "Destination Country",
        "Direction",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/processed/threat_five_class.csv"))
    parser.add_argument("--outdir", type=Path, default=Path("results/duplicate_group_robustness"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    return parser.parse_args()


def build_pipeline(features: list[str], args: argparse.Namespace) -> Pipeline:
    numeric_cols = [c for c in ["Source Port", "Destination Port", "Risk of app"] if c in features]
    categorical_cols = [c for c in features if c not in numeric_cols]
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
                categorical_cols,
            ),
            ("num", Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]), numeric_cols),
        ],
        remainder="drop",
    )
    model = XGBClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softprob",
        eval_metric="mlogloss",
        tree_method="hist",
        device=args.device,
        random_state=args.seed,
        n_jobs=-1,
    )
    return Pipeline(steps=[("preprocess", preprocessor), ("model", model)])


def feature_signature(df: pd.DataFrame, features: list[str]) -> pd.Series:
    values = df[features].astype("string").fillna("<NA>")
    joined = values.agg("\x1f".join, axis=1)
    return joined.map(lambda value: hashlib.sha1(value.encode("utf-8")).hexdigest())


def summarize_duplicates(df: pd.DataFrame, features: list[str], feature_set: str) -> dict[str, object]:
    groups = feature_signature(df, features)
    group_sizes = groups.value_counts()
    label_counts = df.assign(_group=groups).groupby("_group", dropna=False)["target"].nunique()
    return {
        "feature_set": feature_set,
        "features": len(features),
        "rows": int(len(df)),
        "unique_signatures": int(group_sizes.shape[0]),
        "duplicate_rows": int(groups.duplicated(keep=False).sum()),
        "duplicate_row_fraction": float(groups.duplicated(keep=False).mean()),
        "mixed_label_signature_groups": int((label_counts > 1).sum()),
        "largest_signature_group": int(group_sizes.iloc[0]),
        "median_signature_group": float(group_sizes.median()),
    }


def encode_labels(y_train: pd.Series, y_test: pd.Series) -> tuple[np.ndarray, np.ndarray, LabelEncoder]:
    encoder = LabelEncoder()
    y_train_encoded = encoder.fit_transform(y_train.astype(str))
    y_test_encoded = encoder.transform(y_test.astype(str))
    return y_train_encoded, y_test_encoded, encoder


def evaluate_split(df: pd.DataFrame, features: list[str], groups: pd.Series, feature_set: str, args: argparse.Namespace) -> pd.DataFrame:
    splitter = GroupShuffleSplit(n_splits=args.splits, test_size=args.test_size, random_state=args.seed)
    rows: list[dict[str, object]] = []
    dummy_x = np.zeros(len(df))
    for split_id, (train_idx, test_idx) in enumerate(splitter.split(dummy_x, df["target"], groups), start=1):
        train_targets = set(df.iloc[train_idx]["target"].astype(str))
        test_targets = set(df.iloc[test_idx]["target"].astype(str))
        if not test_targets.issubset(train_targets):
            missing = sorted(test_targets - train_targets)
            rows.append(
                {
                    "feature_set": feature_set,
                    "split_id": split_id,
                    "status": "skipped_missing_train_class",
                    "missing_classes": ";".join(missing),
                    "train_rows": int(len(train_idx)),
                    "test_rows": int(len(test_idx)),
                }
            )
            continue
        y_train, y_test, encoder = encode_labels(df.iloc[train_idx]["target"], df.iloc[test_idx]["target"])
        pipe = build_pipeline(features, args)
        start = time.perf_counter()
        pipe.fit(df.iloc[train_idx][features], y_train)
        fit_seconds = time.perf_counter() - start
        start = time.perf_counter()
        y_pred = pipe.predict(df.iloc[test_idx][features])
        predict_seconds = time.perf_counter() - start
        train_groups = set(groups.iloc[train_idx])
        test_groups = set(groups.iloc[test_idx])
        rows.append(
            {
                "feature_set": feature_set,
                "split_id": split_id,
                "status": "ok",
                "features": len(features),
                "train_rows": int(len(train_idx)),
                "test_rows": int(len(test_idx)),
                "train_groups": int(len(train_groups)),
                "test_groups": int(len(test_groups)),
                "group_overlap": int(len(train_groups.intersection(test_groups))),
                "fit_seconds": float(fit_seconds),
                "predict_seconds": float(predict_seconds),
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
                "macro_f1": float(f1_score(y_test, y_pred, average="macro")),
                "weighted_f1": float(f1_score(y_test, y_pred, average="weighted")),
                "errors": int((y_test != y_pred).sum()),
                "labels": ";".join(encoder.classes_),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    started_at = pd.Timestamp.now().isoformat()
    wall_start = time.perf_counter()
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.data, low_memory=False)
    df["target"] = df["target"].astype(str)

    duplicate_summaries = []
    result_frames = []
    for feature_set, features in FEATURE_SETS.items():
        duplicate_summaries.append(summarize_duplicates(df, features, feature_set))
        groups = feature_signature(df, features)
        result_frames.append(evaluate_split(df, features, groups, feature_set, args))

    duplicate_summary = pd.DataFrame(duplicate_summaries)
    grouped_results = pd.concat(result_frames, ignore_index=True)
    ok_results = grouped_results[grouped_results["status"].eq("ok")].copy()
    grouped_summary = (
        ok_results.groupby("feature_set", as_index=False)
        .agg(
            splits=("split_id", "count"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            balanced_accuracy_std=("balanced_accuracy", "std"),
            accuracy_mean=("accuracy", "mean"),
            errors_mean=("errors", "mean"),
            test_rows_mean=("test_rows", "mean"),
            fit_seconds_sum=("fit_seconds", "sum"),
            predict_seconds_sum=("predict_seconds", "sum"),
        )
        .sort_values("feature_set")
    )

    duplicate_summary.to_csv(args.outdir / "duplicate_signature_summary.csv", index=False)
    grouped_results.to_csv(args.outdir / "duplicate_group_split_results.csv", index=False)
    grouped_summary.to_csv(args.outdir / "duplicate_group_split_summary.csv", index=False)
    metadata = {
        "started_at": started_at,
        "ended_at": pd.Timestamp.now().isoformat(),
        "wall_seconds": float(time.perf_counter() - wall_start),
        "data": str(args.data),
        "rows": int(len(df)),
        "splits": args.splits,
        "test_size": args.test_size,
        "device": args.device,
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "note": "Grouped splits keep exact feature signatures out of both train and test at the same time.",
    }
    (args.outdir / "duplicate_group_robustness_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(grouped_summary.to_string(index=False))


if __name__ == "__main__":
    main()
