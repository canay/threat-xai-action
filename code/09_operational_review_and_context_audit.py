"""Operational review-queue and context holdout audit.

This script adds two applied-security validation layers for the Paper 1
manuscript:

1. Review-queue simulation. The model is treated as an action-consistency
   reviewer: records are routed to analyst review when the predicted action
   disagrees with the observed firewall action, confidence is low, or a reset
   action is involved.
2. Leave-one-context-out robustness. Operational context values are held out
   one at a time while the model is trained without Rule, to stress-test how
   well the action mapping transfers to unseen zones, directions, application
   categories, and high-support threat/application contexts.

These checks use only the processed paper-specific dataset. They are not a
substitute for cross-enterprise or independent-period validation.
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
from sklearn.model_selection import train_test_split
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

NO_THREAT_DESCRIPTORS = [
    feature
    for feature in CORE_FEATURES
    if feature not in {"Threat/Content Type", "Threat/Content Name", "Severity"}
]

FEATURE_SETS = {
    "core": CORE_FEATURES,
    "no_threat_descriptors": NO_THREAT_DESCRIPTORS,
}

CONTEXT_COLUMNS = [
    "Direction",
    "Source Zone",
    "Destination Zone",
    "Category of app",
    "Subcategory of app",
    "Application",
    "Threat/Content Type",
]

RESET_LABELS = {"Reset-Both", "Reset-Server"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/processed/threat_five_class.csv"))
    parser.add_argument("--outdir", type=Path, default=Path("results/operational_review_context_audit"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--min-context-rows", type=int, default=1000)
    parser.add_argument("--max-contexts-per-column", type=int, default=8)
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
    model_kwargs = {
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "learning_rate": 0.08,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "objective": "multi:softprob",
        "eval_metric": "mlogloss",
        "tree_method": "hist",
        "random_state": args.seed,
        "n_jobs": -1,
    }
    if args.device == "cuda":
        model_kwargs["device"] = "cuda"
    return Pipeline(steps=[("preprocess", preprocessor), ("model", XGBClassifier(**model_kwargs))])


def macro_f1_observed(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    labels = np.array(sorted(np.unique(y_true)))
    return float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0))


def balanced_accuracy_observed(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    recalls = []
    for label in sorted(np.unique(y_true)):
        mask = y_true == label
        recalls.append(float((y_pred[mask] == label).mean()))
    return float(np.mean(recalls)) if recalls else np.nan


def fit_predict(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    features: list[str],
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], float, float]:
    encoder = LabelEncoder()
    y_train_text = df.iloc[train_idx]["target"].astype(str)
    y_test_text = df.iloc[test_idx]["target"].astype(str)
    encoder.fit(y_train_text)
    if not set(y_test_text).issubset(set(encoder.classes_)):
        missing = sorted(set(y_test_text) - set(encoder.classes_))
        raise ValueError(f"Test classes absent from training data: {missing}")
    y_train = encoder.transform(y_train_text)
    y_test = encoder.transform(y_test_text)
    pipe = build_pipeline(features, args)
    started = time.perf_counter()
    pipe.fit(df.iloc[train_idx][features], y_train)
    fit_seconds = time.perf_counter() - started
    started = time.perf_counter()
    y_pred = pipe.predict(df.iloc[test_idx][features]).astype(int)
    proba = pipe.predict_proba(df.iloc[test_idx][features])
    predict_seconds = time.perf_counter() - started
    return y_test, y_pred, proba, list(encoder.classes_), fit_seconds, predict_seconds


def summarize_prediction(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
    prefix: dict[str, object],
    fit_seconds: float,
    predict_seconds: float,
) -> dict[str, object]:
    return {
        **prefix,
        "status": "ok",
        "train_rows": int(prefix["train_rows"]),
        "test_rows": int(prefix["test_rows"]),
        "fit_seconds": float(fit_seconds),
        "predict_seconds": float(predict_seconds),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy_observed": balanced_accuracy_observed(y_true, y_pred),
        "macro_f1_observed": macro_f1_observed(y_true, y_pred),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "errors": int((y_true != y_pred).sum()),
        "observed_classes": ";".join(labels[label] for label in sorted(np.unique(y_true))),
    }


def review_queue_rows(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    proba: np.ndarray,
    labels: list[str],
    run_id: str,
    split_name: str,
    feature_set: str,
) -> list[dict[str, object]]:
    confidence = proba.max(axis=1)
    true_text = np.array([labels[value] for value in y_true])
    pred_text = np.array([labels[value] for value in y_pred])
    mismatch = y_true != y_pred
    reset_related = np.isin(true_text, list(RESET_LABELS)) | np.isin(pred_text, list(RESET_LABELS))
    low_1 = confidence <= np.quantile(confidence, 0.01)
    low_5 = confidence <= np.quantile(confidence, 0.05)
    low_10 = confidence <= np.quantile(confidence, 0.10)
    scenarios = {
        "mismatch_only": mismatch,
        "low_confidence_bottom_5pct": low_5,
        "reset_related": reset_related,
        "mismatch_or_low5": mismatch | low_5,
        "mismatch_or_low5_or_reset": mismatch | low_5 | reset_related,
        "low1_or_reset": low_1 | reset_related,
        "low10_or_reset": low_10 | reset_related,
    }
    total_errors = int(mismatch.sum())
    rows = []
    for scenario, selected in scenarios.items():
        queue_rows = int(selected.sum())
        captured_errors = int((selected & mismatch).sum())
        reset_rows = int((selected & reset_related).sum())
        rows.append(
            {
                "run_id": run_id,
                "split": split_name,
                "feature_set": feature_set,
                "scenario": scenario,
                "test_rows": int(len(y_true)),
                "queue_rows": queue_rows,
                "queue_fraction": float(queue_rows / len(y_true)),
                "total_errors": total_errors,
                "captured_errors": captured_errors,
                "error_capture_rate": float(captured_errors / total_errors) if total_errors else 1.0,
                "queue_error_precision": float(captured_errors / queue_rows) if queue_rows else np.nan,
                "reset_related_in_queue": reset_rows,
                "mean_confidence_queue": float(confidence[selected].mean()) if queue_rows else np.nan,
                "mean_confidence_not_queue": float(confidence[~selected].mean()) if (~selected).any() else np.nan,
            }
        )
    return rows


def review_queue_detail_rows(
    df: pd.DataFrame,
    test_idx: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    proba: np.ndarray,
    labels: list[str],
    run_id: str,
) -> pd.DataFrame:
    confidence = proba.max(axis=1)
    true_text = np.array([labels[value] for value in y_true])
    pred_text = np.array([labels[value] for value in y_pred])
    mismatch = y_true != y_pred
    reset_related = np.isin(true_text, list(RESET_LABELS)) | np.isin(pred_text, list(RESET_LABELS))
    low_5 = confidence <= np.quantile(confidence, 0.05)
    selected = mismatch | low_5 | reset_related
    detail = df.iloc[test_idx].copy()
    detail = detail.loc[selected, ["target", "Application", "Direction", "Source Zone", "Destination Zone", "Category of app"]]
    detail.insert(0, "run_id", run_id)
    detail["predicted_action"] = pred_text[selected]
    detail["confidence"] = confidence[selected]
    detail["mismatch"] = mismatch[selected]
    detail["reset_related"] = reset_related[selected]
    detail["low_confidence_bottom_5pct"] = low_5[selected]
    return detail


def context_candidates(df: pd.DataFrame, args: argparse.Namespace) -> list[tuple[str, str, np.ndarray]]:
    candidates = []
    for column in CONTEXT_COLUMNS:
        counts = df[column].astype("string").fillna("__MISSING__").value_counts()
        counts = counts[counts >= args.min_context_rows].head(args.max_contexts_per_column)
        for value in counts.index:
            mask = df[column].astype("string").fillna("__MISSING__").eq(value).to_numpy()
            candidates.append((column, str(value), mask))
    return candidates


def run_context_holdouts(df: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    confusions = []
    for column, value, test_mask in context_candidates(df, args):
        train_idx = np.flatnonzero(~test_mask)
        test_idx = np.flatnonzero(test_mask)
        for feature_set, features in FEATURE_SETS.items():
            run_id = f"{feature_set}__holdout_{column}={value}"
            prefix = {
                "run_id": run_id,
                "feature_set": feature_set,
                "context_column": column,
                "context_value": value,
                "train_rows": int(len(train_idx)),
                "test_rows": int(len(test_idx)),
            }
            try:
                y_true, y_pred, _, labels, fit_seconds, predict_seconds = fit_predict(
                    df, train_idx, test_idx, features, args
                )
            except ValueError as exc:
                rows.append({**prefix, "status": "skipped", "reason": str(exc)})
                continue
            rows.append(summarize_prediction(y_true, y_pred, labels, prefix, fit_seconds, predict_seconds))
            pairs = pd.crosstab(
                pd.Series([labels[value] for value in y_true], name="true"),
                pd.Series([labels[value] for value in y_pred], name="predicted"),
            )
            for true_label, pair_row in pairs.iterrows():
                for pred_label, count in pair_row.items():
                    if int(count):
                        confusions.append(
                            {
                                "run_id": run_id,
                                "feature_set": feature_set,
                                "context_column": column,
                                "context_value": value,
                                "true": true_label,
                                "predicted": pred_label,
                                "count": int(count),
                            }
                        )
    return pd.DataFrame(rows), pd.DataFrame(confusions)


def aggregate_context_results(results: pd.DataFrame) -> pd.DataFrame:
    ok = results[results["status"].eq("ok")].copy()
    rows = []
    for (feature_set, column), group in ok.groupby(["feature_set", "context_column"], sort=True):
        total_rows = group["test_rows"].sum()
        weights = group["test_rows"] / total_rows
        rows.append(
            {
                "feature_set": feature_set,
                "context_column": column,
                "heldout_values": int(group.shape[0]),
                "support_rows": int(total_rows),
                "support_weighted_accuracy": float((group["accuracy"] * weights).sum()),
                "support_weighted_macro_f1_observed": float((group["macro_f1_observed"] * weights).sum()),
                "support_weighted_balanced_accuracy_observed": float(
                    (group["balanced_accuracy_observed"] * weights).sum()
                ),
                "worst_context_value": str(group.loc[group["macro_f1_observed"].idxmin(), "context_value"]),
                "worst_macro_f1_observed": float(group["macro_f1_observed"].min()),
                "fit_seconds_sum": float(group["fit_seconds"].sum()),
                "predict_seconds_sum": float(group["predict_seconds"].sum()),
            }
        )
    return pd.DataFrame(rows)


def run_review_queue(df: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_idx, test_idx = train_test_split(
        np.arange(len(df)),
        test_size=0.2,
        stratify=df["target"],
        random_state=args.seed,
    )
    rows = []
    summary_rows = []
    details = []
    for feature_set, features in FEATURE_SETS.items():
        run_id = f"stratified_holdout__{feature_set}"
        y_true, y_pred, proba, labels, fit_seconds, predict_seconds = fit_predict(df, train_idx, test_idx, features, args)
        prefix = {
            "run_id": run_id,
            "feature_set": feature_set,
            "split": "stratified_holdout",
            "train_rows": int(len(train_idx)),
            "test_rows": int(len(test_idx)),
        }
        summary_rows.append(summarize_prediction(y_true, y_pred, labels, prefix, fit_seconds, predict_seconds))
        rows.extend(review_queue_rows(y_true, y_pred, proba, labels, run_id, "stratified_holdout", feature_set))
        details.append(review_queue_detail_rows(df, test_idx, y_true, y_pred, proba, labels, run_id))
    return pd.DataFrame(summary_rows), pd.DataFrame(rows), pd.concat(details, ignore_index=True)


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    wall_start = time.perf_counter()
    started_at = pd.Timestamp.now().isoformat()
    df = pd.read_csv(args.data, low_memory=False)
    df["target"] = df["target"].astype(str)

    review_summary, review_queue, review_detail = run_review_queue(df, args)
    context_results, context_confusions = run_context_holdouts(df, args)
    context_aggregate = aggregate_context_results(context_results)

    review_summary.to_csv(args.outdir / "operational_review_model_summary.csv", index=False)
    review_queue.to_csv(args.outdir / "operational_review_queue_summary.csv", index=False)
    review_detail.to_csv(args.outdir / "operational_review_queue_records.csv", index=False)
    context_results.to_csv(args.outdir / "context_holdout_results.csv", index=False)
    context_confusions.to_csv(args.outdir / "context_holdout_confusions.csv", index=False)
    context_aggregate.to_csv(args.outdir / "context_holdout_aggregate.csv", index=False)

    metadata = {
        "started_at": started_at,
        "ended_at": pd.Timestamp.now().isoformat(),
        "wall_seconds": float(time.perf_counter() - wall_start),
        "data": str(args.data),
        "rows": int(len(df)),
        "device": args.device,
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "min_context_rows": args.min_context_rows,
        "max_contexts_per_column": args.max_contexts_per_column,
        "note": (
            "Review queue simulation uses observed action as an audit reference. "
            "Leave-one-context-out checks are same-dataset stress tests, not independent external validation."
        ),
    }
    (args.outdir / "operational_review_context_audit_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(review_summary.to_string(index=False))
    print(review_queue.to_string(index=False))
    print(context_aggregate.to_string(index=False))


if __name__ == "__main__":
    main()
