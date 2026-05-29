"""Policy-context robustness checks for firewall action prediction.

The manuscript treats the Rule field as high-leakage policy context and excludes
it from the core feature set. This script uses Rule only as a grouping variable:
one policy context is held out at a time, while the model is trained without the
Rule field. The goal is to stress-test whether the learned action mapping
survives when a named policy context is not seen during training.

These checks are not a replacement for cross-enterprise validation. They are a
same-dataset robustness audit for local policy-context sensitivity.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
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
        feature
        for feature in CORE_FEATURES
        if feature not in {"Threat/Content Type", "Threat/Content Name", "Severity"}
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

MISSING_RULE = "__MISSING_RULE__"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/processed/threat_five_class.csv"))
    parser.add_argument("--outdir", type=Path, default=Path("results/policy_context_robustness"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-rule-rows", type=int, default=50)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    return parser.parse_args()


def build_pipeline(features: list[str], args: argparse.Namespace) -> Pipeline:
    numeric_cols = [column for column in ["Source Port", "Destination Port", "Risk of app"] if column in features]
    categorical_cols = [column for column in features if column not in numeric_cols]
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
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


def prepare_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df["target"] = df["target"].astype(str)
    df["Rule Group"] = df["Rule"].astype("string").fillna(MISSING_RULE).astype(str)
    return df.reset_index(drop=True)


def summarize_rules(df: pd.DataFrame) -> pd.DataFrame:
    class_counts = pd.crosstab(df["Rule Group"], df["target"])
    for label in sorted(df["target"].unique()):
        if label not in class_counts.columns:
            class_counts[label] = 0
    class_counts = class_counts[sorted(class_counts.columns)]
    rows = []
    for rule, counts in class_counts.iterrows():
        total = int(counts.sum())
        dominant_class = str(counts.idxmax())
        dominant_fraction = float(counts.max() / total) if total else np.nan
        nonzero_classes = int((counts > 0).sum())
        row = {
            "rule_group": rule,
            "records": total,
            "nonzero_classes": nonzero_classes,
            "dominant_class": dominant_class,
            "dominant_fraction": dominant_fraction,
            "is_missing_rule": bool(rule == MISSING_RULE),
        }
        row.update({f"class_{label}": int(counts[label]) for label in class_counts.columns})
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["records", "rule_group"], ascending=[False, True])


def local_label_encode(y_train: pd.Series, y_test: pd.Series) -> tuple[np.ndarray, np.ndarray, LabelEncoder]:
    encoder = LabelEncoder()
    y_train_encoded = encoder.fit_transform(y_train.astype(str))
    y_test_encoded = encoder.transform(y_test.astype(str))
    return y_train_encoded, y_test_encoded, encoder


def evaluate_rule_holdout(
    df: pd.DataFrame,
    rule_group: str,
    feature_set: str,
    features: list[str],
    all_labels: list[str],
    args: argparse.Namespace,
) -> tuple[dict[str, object], pd.DataFrame]:
    test_mask = df["Rule Group"].eq(rule_group).to_numpy()
    train_mask = ~test_mask
    train_df = df.loc[train_mask]
    test_df = df.loc[test_mask]
    y_train_classes = set(train_df["target"].astype(str))
    y_test_classes = set(test_df["target"].astype(str))

    base_row: dict[str, object] = {
        "feature_set": feature_set,
        "rule_group": rule_group,
        "is_missing_rule": bool(rule_group == MISSING_RULE),
        "features": len(features),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "test_classes": ";".join(sorted(y_test_classes)),
    }

    if not y_test_classes.issubset(y_train_classes):
        missing = sorted(y_test_classes - y_train_classes)
        row = {
            **base_row,
            "status": "skipped_missing_train_class",
            "missing_classes": ";".join(missing),
        }
        return row, pd.DataFrame()

    y_train, y_test, encoder = local_label_encode(train_df["target"], test_df["target"])
    pipeline = build_pipeline(features, args)
    fit_start = time.perf_counter()
    pipeline.fit(train_df[features], y_train)
    fit_seconds = time.perf_counter() - fit_start
    predict_start = time.perf_counter()
    y_pred = pipeline.predict(test_df[features])
    predict_seconds = time.perf_counter() - predict_start

    all_label_ids = encoder.transform([label for label in all_labels if label in set(encoder.classes_)])
    observed_label_ids = encoder.transform(sorted(y_test_classes))
    observed_recalls = []
    for label_id in observed_label_ids:
        mask = y_test == label_id
        observed_recalls.append(float((y_pred[mask] == label_id).mean()))
    accuracy = accuracy_score(y_test, y_pred)
    balanced = float(np.mean(observed_recalls)) if observed_recalls else np.nan
    macro_observed = f1_score(y_test, y_pred, average="macro", labels=observed_label_ids, zero_division=0)
    macro_all = f1_score(y_test, y_pred, average="macro", labels=all_label_ids, zero_division=0)
    weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    row = {
        **base_row,
        "status": "ok",
        "missing_classes": "",
        "fit_seconds": float(fit_seconds),
        "predict_seconds": float(predict_seconds),
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced),
        "macro_f1_observed_classes": float(macro_observed),
        "macro_f1_all_classes": float(macro_all),
        "weighted_f1": float(weighted),
        "errors": int((y_test != y_pred).sum()),
    }

    confusion = pd.crosstab(
        pd.Series(encoder.inverse_transform(y_test), name="true"),
        pd.Series(encoder.inverse_transform(y_pred), name="predicted"),
    ).reset_index()
    confusion.insert(0, "feature_set", feature_set)
    confusion.insert(1, "rule_group", rule_group)
    return row, confusion


def aggregate_predictions(
    results: pd.DataFrame,
    scope_name: str,
    include_missing_rule: bool,
    min_rule_rows: int,
) -> pd.DataFrame:
    part = results[
        results["status"].eq("ok")
        & (results["test_rows"] >= min_rule_rows)
        & (results["is_missing_rule"].eq(include_missing_rule) if include_missing_rule else ~results["is_missing_rule"])
    ].copy()
    if part.empty:
        return pd.DataFrame()
    rows = []
    for feature_set, group in part.groupby("feature_set", sort=True):
        total_rows = group["test_rows"].sum()
        total_errors = group["errors"].sum()
        weights = group["test_rows"] / total_rows
        rows.append(
            {
                "scope": scope_name,
                "feature_set": feature_set,
                "heldout_contexts": int(group.shape[0]),
                "test_rows": int(total_rows),
                "errors": int(total_errors),
                "support_weighted_accuracy": float(1.0 - total_errors / total_rows),
                "support_weighted_balanced_accuracy": float((group["balanced_accuracy"] * weights).sum()),
                "support_weighted_observed_macro_f1": float((group["macro_f1_observed_classes"] * weights).sum()),
                "unweighted_observed_macro_f1_mean": float(group["macro_f1_observed_classes"].mean()),
                "unweighted_observed_macro_f1_std": float(group["macro_f1_observed_classes"].std()),
                "fit_seconds_sum": float(group["fit_seconds"].sum()),
                "predict_seconds_sum": float(group["predict_seconds"].sum()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    started_at = pd.Timestamp.now().isoformat()
    wall_start = time.perf_counter()
    args.outdir.mkdir(parents=True, exist_ok=True)
    df = prepare_data(args.data)
    all_labels = sorted(df["target"].unique())

    rule_summary = summarize_rules(df)
    candidate_rules = rule_summary.loc[rule_summary["records"].ge(args.min_rule_rows), "rule_group"].tolist()

    result_rows: list[dict[str, object]] = []
    confusion_frames: list[pd.DataFrame] = []
    for feature_set, features in FEATURE_SETS.items():
        for rule_group in candidate_rules:
            print(f"Running {feature_set} with held-out rule context: {rule_group}")
            row, confusion = evaluate_rule_holdout(df, rule_group, feature_set, features, all_labels, args)
            result_rows.append(row)
            if not confusion.empty:
                confusion_frames.append(confusion)

    results = pd.DataFrame(result_rows)
    aggregate = pd.concat(
        [
            aggregate_predictions(results, "named_rule_contexts_only", include_missing_rule=False, min_rule_rows=args.min_rule_rows),
            aggregate_predictions(results, "missing_rule_context_only", include_missing_rule=True, min_rule_rows=args.min_rule_rows),
        ],
        ignore_index=True,
    )

    rule_summary.to_csv(args.outdir / "policy_context_rule_summary.csv", index=False)
    results.to_csv(args.outdir / "policy_context_heldout_results.csv", index=False)
    aggregate.to_csv(args.outdir / "policy_context_heldout_aggregate.csv", index=False)
    if confusion_frames:
        pd.concat(confusion_frames, ignore_index=True).to_csv(
            args.outdir / "policy_context_heldout_confusions.csv", index=False
        )

    metadata = {
        "started_at": started_at,
        "ended_at": pd.Timestamp.now().isoformat(),
        "wall_seconds": float(time.perf_counter() - wall_start),
        "data": str(args.data),
        "rows": int(len(df)),
        "rule_groups": int(df["Rule Group"].nunique()),
        "candidate_rule_groups": int(len(candidate_rules)),
        "min_rule_rows": int(args.min_rule_rows),
        "device": args.device,
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "note": "Rule is used only as a held-out group variable and is not included in any feature set.",
    }
    (args.outdir / "policy_context_robustness_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(aggregate.to_string(index=False))


if __name__ == "__main__":
    main()
