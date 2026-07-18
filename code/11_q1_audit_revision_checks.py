"""Q1 audit revision checks for the firewall action manuscript.

This script adds two lightweight checks used in the revision text:

1. Simple model-free baselines that do not use direct policy fields or direct
   threat/signature descriptors.
2. Test-set bootstrap confidence intervals from the saved fixed-model
   true/predicted confusion-cell counts.

The bootstrap samples the empirical confusion-cell distribution from one saved
fitted-model output; it is not a repeated-training uncertainty estimate. Pass
``--fit-xgb`` to refit the XGBoost probes locally instead.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from xgboost import XGBClassifier


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


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

DIRECT_THREAT_DESCRIPTORS = {"Threat/Content Name", "Threat/Content Type", "Severity"}

FEATURE_SETS = {
    "core": CORE_FEATURES,
    "no_threat_descriptors": [f for f in CORE_FEATURES if f not in DIRECT_THREAT_DESCRIPTORS],
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

CONTEXT_BASELINES = {
    "train_majority": [],
    "context_majority_app_dir_zones": [
        "Application",
        "Direction",
        "Source Zone",
        "Destination Zone",
    ],
    "context_majority_operational_minimal": [
        "Application",
        "Direction",
        "Source Zone",
        "Destination Zone",
        "Inbound Interface",
        "Outbound Interface",
        "IP Protocol",
        "Destination Port",
        "Source Country",
        "Destination Country",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/processed/threat_five_class.csv"))
    parser.add_argument("--outdir", type=Path, default=Path("results/q1_audit_revision"))
    parser.add_argument("--confusions", type=Path, default=Path("results/extensions/q1_validation_confusions.csv"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-iters", type=int, default=500)
    parser.add_argument("--fit-xgb", action="store_true", help="Refit XGBoost probes instead of using saved confusions.")
    parser.add_argument(
        "--confusion-only",
        action="store_true",
        help="Only compute CI outputs from saved confusion counts; skip controlled-data baselines.",
    )
    return parser.parse_args()


def chronological_indices(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    ordered = df.sort_values(["Generate Time Parsed"], kind="mergesort").index.to_numpy()
    nominal_cut = int(len(ordered) * 0.8)
    boundary_time = df.loc[ordered[nominal_cut], "Generate Time Parsed"]
    boundary_positions = np.flatnonzero(
        df.loc[ordered, "Generate Time Parsed"].to_numpy() == boundary_time.to_datetime64()
    )
    candidates = [int(boundary_positions[0]), int(boundary_positions[-1] + 1)]
    candidates = [candidate for candidate in candidates if 0 < candidate < len(ordered)]
    cut = min(candidates, key=lambda candidate: (abs(candidate - nominal_cut), candidate))
    return ordered[:cut], ordered[cut:]


def majority_label(labels: pd.Series) -> str:
    counts = labels.astype(str).value_counts()
    top = counts[counts == counts.max()].index.tolist()
    return sorted(top)[0]


def fixed_label_order(series: pd.Series) -> list[str]:
    preferred = ["Allow", "Block", "Drop", "Reset-Both", "Reset-Server"]
    observed = set(series.astype(str))
    ordered = [label for label in preferred if label in observed]
    ordered.extend(sorted(observed - set(ordered)))
    return ordered


def evaluate_string_predictions(
    y_true: pd.Series,
    y_pred: list[str] | np.ndarray,
    labels: list[str],
) -> dict:
    true = y_true.astype(str).to_numpy()
    pred = np.asarray(y_pred).astype(str)
    return {
        "accuracy": float(accuracy_score(true, pred)),
        "balanced_accuracy_fixed_labels": float(recall_score(true, pred, labels=labels, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(true, pred, labels=labels, average="macro", zero_division=0)),
        "errors": int((true != pred).sum()),
    }


def make_context_keys(frame: pd.DataFrame, columns: list[str]) -> list[tuple[str, ...]]:
    if not columns:
        return []
    values = frame[columns].fillna("__MISSING__").astype(str)
    return list(values.itertuples(index=False, name=None))


def evaluate_context_baseline(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    split_name: str,
    baseline_name: str,
    context_columns: list[str],
    labels: list[str],
) -> dict:
    train = df.loc[train_idx]
    test = df.loc[test_idx]
    global_label = majority_label(train["target"])

    if not context_columns:
        predictions = [global_label] * len(test)
        matched_fraction = 0.0
        mapping_size = 0
    else:
        mapping: dict[tuple[str, ...], str] = {}
        grouped = train.assign(_key=make_context_keys(train, context_columns)).groupby("_key")["target"]
        for key, group in grouped:
            mapping[key] = majority_label(group)
        test_keys = make_context_keys(test, context_columns)
        matched = [key in mapping for key in test_keys]
        predictions = [mapping.get(key, global_label) for key in test_keys]
        matched_fraction = float(np.mean(matched))
        mapping_size = len(mapping)

    metrics = evaluate_string_predictions(test["target"], predictions, labels)
    return {
        "split": split_name,
        "baseline": baseline_name,
        "context_columns": ";".join(context_columns) if context_columns else "(none)",
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "mapping_size": int(mapping_size),
        "matched_context_fraction": matched_fraction,
        **metrics,
    }


def build_xgb_pipeline(features: list[str], seed: int) -> Pipeline:
    numeric_cols = [c for c in ["Source Port", "Destination Port", "Risk of app"] if c in features]
    categorical_cols = [c for c in features if c not in numeric_cols]
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
        verbose_feature_names_out=False,
    )
    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=seed,
        n_jobs=-1,
    )
    return Pipeline(steps=[("preprocess", preprocessor), ("model", model)])


def fixed_model_summary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels_encoded: list[int],
) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy_fixed_labels": float(
            recall_score(y_true, y_pred, labels=labels_encoded, average="macro", zero_division=0)
        ),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels_encoded, average="macro", zero_division=0)),
        "errors": int((y_true != y_pred).sum()),
    }


def bootstrap_fixed_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels_encoded: list[int],
    label_names: list[str],
    rng: np.random.Generator,
    iterations: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = len(y_true)
    metric_samples = {"accuracy": [], "balanced_accuracy_fixed_labels": [], "macro_f1": []}
    per_class_samples = {label: [] for label in label_names}

    for _ in range(iterations):
        sample = rng.integers(0, n, size=n)
        yt = y_true[sample]
        yp = y_pred[sample]
        metric_samples["accuracy"].append(accuracy_score(yt, yp))
        metric_samples["balanced_accuracy_fixed_labels"].append(
            recall_score(yt, yp, labels=labels_encoded, average="macro", zero_division=0)
        )
        metric_samples["macro_f1"].append(f1_score(yt, yp, labels=labels_encoded, average="macro", zero_division=0))
        recalls = recall_score(yt, yp, labels=labels_encoded, average=None, zero_division=0)
        for label_name, recall in zip(label_names, recalls):
            per_class_samples[label_name].append(recall)

    metric_rows = []
    for metric, values in metric_samples.items():
        arr = np.asarray(values, dtype=float)
        metric_rows.append(
            {
                "metric": metric,
                "ci_low_95": float(np.quantile(arr, 0.025)),
                "ci_high_95": float(np.quantile(arr, 0.975)),
                "bootstrap_mean": float(np.mean(arr)),
            }
        )

    recall_rows = []
    for label, values in per_class_samples.items():
        arr = np.asarray(values, dtype=float)
        recall_rows.append(
            {
                "class": label,
                "recall_ci_low_95": float(np.quantile(arr, 0.025)),
                "recall_ci_high_95": float(np.quantile(arr, 0.975)),
                "recall_bootstrap_mean": float(np.mean(arr)),
            }
        )
    return pd.DataFrame(metric_rows), pd.DataFrame(recall_rows)


def metrics_from_confusion_matrix(matrix: np.ndarray) -> dict:
    matrix = matrix.astype(float)
    total = matrix.sum()
    diagonal = np.diag(matrix)
    row_sums = matrix.sum(axis=1)
    col_sums = matrix.sum(axis=0)
    recall = np.divide(diagonal, row_sums, out=np.zeros_like(diagonal), where=row_sums > 0)
    precision = np.divide(diagonal, col_sums, out=np.zeros_like(diagonal), where=col_sums > 0)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(diagonal),
        where=(precision + recall) > 0,
    )
    return {
        "accuracy": float(diagonal.sum() / total) if total else 0.0,
        "balanced_accuracy_fixed_labels": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "errors": int(total - diagonal.sum()),
    }


def bootstrap_confusion_matrix(
    matrix: np.ndarray,
    labels: list[str],
    rng: np.random.Generator,
    iterations: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = int(matrix.sum())
    probs = matrix.astype(float).ravel() / n
    samples = rng.multinomial(n, probs, size=iterations).reshape(iterations, matrix.shape[0], matrix.shape[1])

    metric_samples = {"accuracy": [], "balanced_accuracy_fixed_labels": [], "macro_f1": []}
    per_class_samples = {label: [] for label in labels}
    for sample_matrix in samples:
        metrics = metrics_from_confusion_matrix(sample_matrix)
        for metric in metric_samples:
            metric_samples[metric].append(metrics[metric])
        diagonal = np.diag(sample_matrix).astype(float)
        row_sums = sample_matrix.sum(axis=1).astype(float)
        recalls = np.divide(diagonal, row_sums, out=np.zeros_like(diagonal), where=row_sums > 0)
        for label, recall in zip(labels, recalls):
            per_class_samples[label].append(float(recall))

    metric_rows = []
    for metric, values in metric_samples.items():
        arr = np.asarray(values, dtype=float)
        metric_rows.append(
            {
                "metric": metric,
                "ci_low_95": float(np.quantile(arr, 0.025)),
                "ci_high_95": float(np.quantile(arr, 0.975)),
                "bootstrap_mean": float(np.mean(arr)),
            }
        )

    recall_rows = []
    for label, values in per_class_samples.items():
        arr = np.asarray(values, dtype=float)
        recall_rows.append(
            {
                "class": label,
                "recall_ci_low_95": float(np.quantile(arr, 0.025)),
                "recall_ci_high_95": float(np.quantile(arr, 0.975)),
                "recall_bootstrap_mean": float(np.mean(arr)),
            }
        )
    return pd.DataFrame(metric_rows), pd.DataFrame(recall_rows)


def evaluate_xgb(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    split_name: str,
    feature_set: str,
    encoder: LabelEncoder,
    seed: int,
    iterations: int,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    features = FEATURE_SETS[feature_set]
    train = df.loc[train_idx]
    test = df.loc[test_idx]
    y_train = encoder.transform(train["target"].astype(str))
    y_test = encoder.transform(test["target"].astype(str))

    pipe = build_xgb_pipeline(features, seed)
    pipe.fit(train[features], y_train)
    y_pred = pipe.predict(test[features])

    label_names = list(encoder.classes_)
    labels_encoded = list(range(len(label_names)))
    summary = fixed_model_summary(y_test, y_pred, labels_encoded)
    rng = np.random.default_rng(seed + len(feature_set) + len(split_name))
    metric_ci, class_ci = bootstrap_fixed_predictions(
        y_test, y_pred, labels_encoded, label_names, rng, iterations
    )

    metric_ci.insert(0, "split", split_name)
    metric_ci.insert(1, "feature_set", feature_set)
    metric_ci.insert(2, "point_estimate", metric_ci["metric"].map(summary))
    metric_ci["bootstrap_iterations"] = iterations
    metric_ci["uncertainty_scope"] = "test_set_bootstrap_fixed_fitted_model"

    class_ci.insert(0, "split", split_name)
    class_ci.insert(1, "feature_set", feature_set)
    observed_recalls = recall_score(y_test, y_pred, labels=labels_encoded, average=None, zero_division=0)
    class_ci["point_recall"] = observed_recalls
    class_ci["test_support"] = [int((y_test == label).sum()) for label in labels_encoded]
    class_ci["bootstrap_iterations"] = iterations
    class_ci["uncertainty_scope"] = "test_set_bootstrap_fixed_fitted_model"

    return {
        "split": split_name,
        "feature_set": feature_set,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        **summary,
    }, metric_ci, class_ci


def xgb_ci_from_saved_confusions(
    confusion_path: Path,
    labels: list[str],
    seed: int,
    iterations: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    confusions = pd.read_csv(confusion_path)
    summary_rows = []
    metric_frames = []
    class_frames = []
    for (feature_set, split_name), group in confusions.groupby(["feature_set", "split"], sort=False):
        matrix = np.zeros((len(labels), len(labels)), dtype=int)
        label_to_index = {label: idx for idx, label in enumerate(labels)}
        for row in group.itertuples(index=False):
            matrix[label_to_index[row.true], label_to_index[row.predicted]] += int(row.count)
        summary = metrics_from_confusion_matrix(matrix)
        rng = np.random.default_rng(seed + len(feature_set) + len(split_name))
        metric_ci, class_ci = bootstrap_confusion_matrix(matrix, labels, rng, iterations)

        metric_ci.insert(0, "split", split_name)
        metric_ci.insert(1, "feature_set", feature_set)
        metric_ci.insert(2, "point_estimate", metric_ci["metric"].map(summary))
        metric_ci["bootstrap_iterations"] = iterations
        metric_ci["uncertainty_scope"] = "confusion_cell_bootstrap_from_saved_fixed_model_output"

        class_ci.insert(0, "split", split_name)
        class_ci.insert(1, "feature_set", feature_set)
        diagonal = np.diag(matrix).astype(float)
        row_sums = matrix.sum(axis=1).astype(float)
        observed_recalls = np.divide(diagonal, row_sums, out=np.zeros_like(diagonal), where=row_sums > 0)
        class_ci["point_recall"] = observed_recalls
        class_ci["test_support"] = row_sums.astype(int)
        class_ci["bootstrap_iterations"] = iterations
        class_ci["uncertainty_scope"] = "confusion_cell_bootstrap_from_saved_fixed_model_output"

        summary_rows.append(
            {
                "split": split_name,
                "feature_set": feature_set,
                "train_rows": "",
                "test_rows": int(matrix.sum()),
                **summary,
            }
        )
        metric_frames.append(metric_ci)
        class_frames.append(class_ci)

    return (
        pd.DataFrame(summary_rows),
        pd.concat(metric_frames, ignore_index=True),
        pd.concat(class_frames, ignore_index=True),
    )


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    if args.confusion_only:
        preview = pd.read_csv(args.confusions)
        labels = fixed_label_order(pd.concat([preview["true"], preview["predicted"]], ignore_index=True))
        xgb_summary, xgb_metric_ci, xgb_class_ci = xgb_ci_from_saved_confusions(
            args.confusions, labels, args.seed, args.bootstrap_iters
        )
        xgb_summary.to_csv(args.outdir / "xgb_fixed_model_summary.csv", index=False)
        xgb_metric_ci.to_csv(args.outdir / "xgb_fixed_model_bootstrap_ci.csv", index=False)
        xgb_class_ci.to_csv(args.outdir / "xgb_fixed_model_per_class_recall_ci.csv", index=False)
        metadata = {
            "seed": args.seed,
            "bootstrap_iterations": args.bootstrap_iters,
            "labels": labels,
            "xgb_ci_source": args.confusions.name,
            "xgb_ci_source_sha256": sha256(args.confusions),
            "note": "Confusion-only mode samples saved fixed-model confusion-cell counts and does not read controlled data.",
        }
        (args.outdir / "q1_audit_revision_metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        print("XGBoost fixed-model summaries:")
        print(xgb_summary.to_string(index=False))
        return

    df = pd.read_csv(args.data, low_memory=False)
    df["target"] = df["target"].astype(str)
    df["Generate Time Parsed"] = pd.to_datetime(df["Generate Time"], errors="coerce")
    df = df[df["Generate Time Parsed"].notna()].copy().reset_index(drop=True)

    labels = fixed_label_order(df["target"])
    encoder = LabelEncoder()
    encoder.fit(labels)

    strat_train, strat_test = train_test_split(
        np.arange(len(df)),
        test_size=0.2,
        stratify=df["target"],
        random_state=args.seed,
    )
    chrono_train, chrono_test = chronological_indices(df)
    splits = {
        "stratified_holdout": (strat_train, strat_test),
        "chronological_holdout": (chrono_train, chrono_test),
    }

    baseline_rows = []
    for split_name, (train_idx, test_idx) in splits.items():
        for baseline_name, context_columns in CONTEXT_BASELINES.items():
            baseline_rows.append(
                evaluate_context_baseline(
                    df,
                    train_idx,
                    test_idx,
                    split_name,
                    baseline_name,
                    context_columns,
                    labels,
                )
            )
    baseline_df = pd.DataFrame(baseline_rows)
    baseline_df.to_csv(args.outdir / "simple_context_baselines.csv", index=False)

    if args.fit_xgb:
        xgb_rows = []
        metric_ci_frames = []
        class_ci_frames = []
        for split_name, (train_idx, test_idx) in splits.items():
            for feature_set in ["core", "no_threat_descriptors", "minimal_context"]:
                summary, metric_ci, class_ci = evaluate_xgb(
                    df,
                    train_idx,
                    test_idx,
                    split_name,
                    feature_set,
                    encoder,
                    args.seed,
                    args.bootstrap_iters,
                )
                xgb_rows.append(summary)
                metric_ci_frames.append(metric_ci)
                class_ci_frames.append(class_ci)
        xgb_summary = pd.DataFrame(xgb_rows)
        xgb_metric_ci = pd.concat(metric_ci_frames, ignore_index=True)
        xgb_class_ci = pd.concat(class_ci_frames, ignore_index=True)
        xgb_source = "local_refit"
        xgb_source_sha256 = None
    else:
        xgb_summary, xgb_metric_ci, xgb_class_ci = xgb_ci_from_saved_confusions(
            args.confusions, labels, args.seed, args.bootstrap_iters
        )
        xgb_source = args.confusions.name
        xgb_source_sha256 = sha256(args.confusions)

    xgb_summary.to_csv(args.outdir / "xgb_fixed_model_summary.csv", index=False)
    xgb_metric_ci.to_csv(args.outdir / "xgb_fixed_model_bootstrap_ci.csv", index=False)
    xgb_class_ci.to_csv(args.outdir / "xgb_fixed_model_per_class_recall_ci.csv", index=False)

    metadata = {
        "seed": args.seed,
        "bootstrap_iterations": args.bootstrap_iters,
        "data_rows_after_timestamp_filter": int(len(df)),
        "labels": labels,
        "feature_sets": FEATURE_SETS,
        "context_baselines": CONTEXT_BASELINES,
        "xgb_ci_source": xgb_source,
        "xgb_ci_source_sha256": xgb_source_sha256,
        "note": (
            "With --fit-xgb, one model is fitted per split and feature set; bootstrap iterations "
            "resample that model's fixed test predictions and do not refit the model. Without "
            "--fit-xgb, the same fixed-prediction scope is reconstructed from saved confusions."
        ),
    }
    (args.outdir / "q1_audit_revision_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print("Simple baselines:")
    print(baseline_df.to_string(index=False))
    print("\nXGBoost fixed-model summaries:")
    print(xgb_summary.to_string(index=False))


if __name__ == "__main__":
    main()
