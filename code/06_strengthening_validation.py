"""Strengthening checks for the firewall action manuscript.

This script adds three reviewer-facing validation layers:

1. Feature-group ablations to test whether performance depends on a narrow
   family of policy-proxy fields.
2. Forward-in-time within-day splits to stress the random-split evidence.
3. Selective prediction diagnostics to quantify whether low-confidence records
   can be routed to manual review in an analyst-facing workflow.

The timestamp span in the processed dataset is short, so the temporal analysis is
reported as a within-day forward-shift stress test rather than long-term temporal
generalization.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
)
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

FEATURE_GROUPS = {
    "threat_descriptors": ["Threat/Content Type", "Threat/Content Name", "Severity"],
    "application_context": [
        "Application",
        "Subcategory of app",
        "Category of app",
        "Technology of app",
        "Risk of app",
        "SaaS of app",
        "Category",
    ],
    "zones_interfaces": ["Source Zone", "Destination Zone", "Inbound Interface", "Outbound Interface"],
    "network_endpoint_context": ["Source Country", "Destination Country", "Source Port", "Destination Port", "IP Protocol"],
    "direction": ["Direction"],
}

BLUE = "#3D79A8"
TEAL = "#2A9D8F"
SLATE = "#5E7182"
CYAN = "#4CB3C7"
GRID = "#B8C6D1"
EDGE = "#243342"


@dataclass
class PredictionBundle:
    y_true: np.ndarray
    y_pred: np.ndarray
    proba: np.ndarray
    labels: list[str]
    fit_seconds: float
    predict_seconds: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/processed/threat_five_class.csv"))
    parser.add_argument("--outdir", type=Path, default=Path("results/strengthening"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=0,
        help="Optional stratified sample size for smoke tests. Use 0 for the full dataset.",
    )
    return parser.parse_args()


def load_data(path: Path, sample_rows: int, seed: int) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df["target"] = df["target"].astype(str)
    df["Generate Time Parsed"] = pd.to_datetime(df["Generate Time"], errors="coerce")
    df = df[df["Generate Time Parsed"].notna()].copy()
    if sample_rows and sample_rows < len(df):
        frac = sample_rows / len(df)
        sampled_parts = []
        for _, part in df.groupby("target", sort=False):
            sampled_parts.append(part.sample(max(1, int(round(len(part) * frac))), random_state=seed))
        df = pd.concat(sampled_parts, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df.reset_index(drop=True)


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
                        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
                    ]
                ),
                categorical_cols,
            ),
            ("num", Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]), numeric_cols),
        ],
        verbose_feature_names_out=False,
    )
    xgb_kwargs = {
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "learning_rate": 0.08,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "eval_metric": "mlogloss",
        "tree_method": "hist",
        "random_state": args.seed,
        "n_jobs": -1,
    }
    if args.device == "cuda":
        xgb_kwargs["device"] = "cuda"
    model = XGBClassifier(**xgb_kwargs)
    return Pipeline(steps=[("preprocess", preprocessor), ("model", model)])


def multiclass_brier(y_true: np.ndarray, proba: np.ndarray) -> float:
    target = np.zeros_like(proba)
    target[np.arange(len(y_true)), y_true] = 1.0
    return float(np.mean(np.sum((proba - target) ** 2, axis=1)))


def expected_calibration_error(y_true: np.ndarray, proba: np.ndarray, bins: int = 10) -> float:
    pred = proba.argmax(axis=1)
    confidence = proba.max(axis=1)
    correct = (pred == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi == 1.0:
            mask = (confidence >= lo) & (confidence <= hi)
        else:
            mask = (confidence >= lo) & (confidence < hi)
        count = int(mask.sum())
        if count:
            ece += (count / len(y_true)) * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    return float(ece)


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, proba: np.ndarray, labels: list[str]) -> dict:
    label_ids = np.arange(len(labels))
    confidence = proba.max(axis=1)
    correct = y_true == y_pred
    recalls = []
    for label_id in label_ids:
        mask = y_true == label_id
        if mask.any():
            recalls.append(float((y_pred[mask] == label_id).mean()))
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": float(np.mean(recalls)) if recalls else np.nan,
        "macro_f1": f1_score(y_true, y_pred, average="macro", labels=label_ids, zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "log_loss": log_loss(y_true, proba, labels=label_ids),
        "brier": multiclass_brier(y_true, proba),
        "ece_10": expected_calibration_error(y_true, proba),
        "errors": int((~correct).sum()),
        "mean_confidence": float(confidence.mean()),
        "mean_confidence_correct": float(confidence[correct].mean()) if correct.any() else np.nan,
        "mean_confidence_wrong": float(confidence[~correct].mean()) if (~correct).any() else np.nan,
    }


def fit_predict(
    df: pd.DataFrame,
    encoder: LabelEncoder,
    features: list[str],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    args: argparse.Namespace,
) -> PredictionBundle:
    y_train = encoder.transform(df.iloc[train_idx]["target"])
    y_test = encoder.transform(df.iloc[test_idx]["target"])
    train_classes = np.array(sorted(np.unique(y_train)))
    local_map = {global_label: local_label for local_label, global_label in enumerate(train_classes)}
    y_train_local = np.array([local_map[value] for value in y_train], dtype=int)
    pipe = build_pipeline(features, args)
    start = time.perf_counter()
    pipe.fit(df.iloc[train_idx][features], y_train_local)
    fit_seconds = time.perf_counter() - start
    start = time.perf_counter()
    local_pred = pipe.predict(df.iloc[test_idx][features]).astype(int)
    local_proba = pipe.predict_proba(df.iloc[test_idx][features])
    predict_seconds = time.perf_counter() - start
    y_pred = train_classes[local_pred]
    proba = np.zeros((len(test_idx), len(encoder.classes_)), dtype=float)
    for local_label, global_label in enumerate(train_classes):
        proba[:, global_label] = local_proba[:, local_label]
    row_sums = proba.sum(axis=1, keepdims=True)
    proba = np.divide(proba, row_sums, out=np.zeros_like(proba), where=row_sums > 0)
    return PredictionBundle(y_test, y_pred, proba, list(encoder.classes_), fit_seconds, predict_seconds)


def chronological_indices(df: pd.DataFrame, train_fraction: float = 0.8) -> tuple[np.ndarray, np.ndarray]:
    ordered = df.sort_values("Generate Time Parsed").index.to_numpy()
    cut = int(len(ordered) * train_fraction)
    return ordered[:cut], ordered[cut:]


def make_forward_splits(df: pd.DataFrame, min_test_rows: int = 5000) -> list[tuple[str, np.ndarray, np.ndarray]]:
    rows = []
    quantiles = [0.50, 0.60, 0.70, 0.80]
    ordered = df.sort_values("Generate Time Parsed").index.to_numpy()
    for q in quantiles:
        cut = int(len(ordered) * q)
        train_idx = ordered[:cut]
        test_idx = ordered[cut:]
        if len(test_idx) < min_test_rows:
            continue
        train_end = df.loc[train_idx, "Generate Time Parsed"].max()
        test_start = df.loc[test_idx, "Generate Time Parsed"].min()
        name = f"forward_q{int(q * 100)}_train_until_{train_end:%H%M}_test_from_{test_start:%H%M}"
        rows.append((name, train_idx, test_idx))
    return rows


def confusion_rows(bundle: PredictionBundle, run_id: str, feature_set: str, split: str) -> list[dict]:
    cm = confusion_matrix(bundle.y_true, bundle.y_pred, labels=np.arange(len(bundle.labels)))
    rows = []
    for i, true_label in enumerate(bundle.labels):
        for j, pred_label in enumerate(bundle.labels):
            count = int(cm[i, j])
            if count:
                rows.append(
                    {
                        "run_id": run_id,
                        "feature_set": feature_set,
                        "split": split,
                        "true": true_label,
                        "predicted": pred_label,
                        "count": count,
                    }
                )
    return rows


def selective_rows(bundle: PredictionBundle, run_id: str, feature_set: str, split: str) -> list[dict]:
    rows = []
    confidence = bundle.proba.max(axis=1)
    coverages = [1.00, 0.95, 0.90, 0.80, 0.70, 0.60, 0.50]
    for coverage in coverages:
        threshold = float(np.quantile(confidence, 1.0 - coverage))
        keep = confidence >= threshold
        if keep.sum() == 0:
            continue
        metrics = evaluate_predictions(bundle.y_true[keep], bundle.y_pred[keep], bundle.proba[keep], bundle.labels)
        class_counts = pd.Series([bundle.labels[i] for i in bundle.y_true[keep]]).value_counts()
        rows.append(
            {
                "run_id": run_id,
                "feature_set": feature_set,
                "split": split,
                "coverage_target": coverage,
                "confidence_threshold": threshold,
                "kept_rows": int(keep.sum()),
                "referred_rows": int((~keep).sum()),
                "kept_fraction": float(keep.mean()),
                "minority_reset_server_kept": int(class_counts.get("Reset-Server", 0)),
                **metrics,
            }
        )
    return rows


def confidence_decile_rows(bundle: PredictionBundle, run_id: str, feature_set: str, split: str) -> list[dict]:
    confidence = bundle.proba.max(axis=1)
    decile = pd.qcut(pd.Series(confidence), q=10, labels=False, duplicates="drop")
    rows = []
    for d in sorted(decile.dropna().unique()):
        mask = decile.to_numpy() == d
        rows.append(
            {
                "run_id": run_id,
                "feature_set": feature_set,
                "split": split,
                "confidence_decile": int(d),
                "rows": int(mask.sum()),
                "confidence_min": float(confidence[mask].min()),
                "confidence_max": float(confidence[mask].max()),
                "error_rate": float((bundle.y_true[mask] != bundle.y_pred[mask]).mean()),
            }
        )
    return rows


def evaluate_run(
    df: pd.DataFrame,
    encoder: LabelEncoder,
    run_id: str,
    feature_set: str,
    split: str,
    features: list[str],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    args: argparse.Namespace,
) -> tuple[dict, PredictionBundle]:
    bundle = fit_predict(df, encoder, features, train_idx, test_idx, args)
    metrics = evaluate_predictions(bundle.y_true, bundle.y_pred, bundle.proba, bundle.labels)
    return (
        {
            "run_id": run_id,
            "model": "XGBoost",
            "feature_set": feature_set,
            "split": split,
            "features": len(features),
            "train_rows": len(train_idx),
            "test_rows": len(test_idx),
            "fit_seconds": bundle.fit_seconds,
            "predict_seconds": bundle.predict_seconds,
            **metrics,
        },
        bundle,
    )


def save_figures(summary: pd.DataFrame, selective: pd.DataFrame, outdir: Path) -> None:
    ablation = summary[summary["split"].isin(["stratified_holdout", "chronological_80_20"])].copy()
    preferred_order = [
        "core",
        "drop_threat_descriptors",
        "drop_application_context",
        "drop_zones_interfaces",
        "drop_network_endpoint_context",
        "drop_direction",
    ]
    ablation["feature_set"] = pd.Categorical(ablation["feature_set"], preferred_order, ordered=True)
    ablation = ablation.sort_values(["split", "feature_set"])

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.8), sharey=True)
    for ax, split, letter in zip(axes, ["stratified_holdout", "chronological_80_20"], ["(a)", "(b)"]):
        part = ablation[ablation["split"] == split]
        x = np.arange(len(part))
        ax.bar(x, part["macro_f1"], color=BLUE, edgecolor=EDGE, linewidth=0.45, label="Macro-F1")
        ax.scatter(x, part["balanced_accuracy"], color=TEAL, s=26, label="Balanced acc.", zorder=3)
        ax.set_ylim(0.70, 1.01)
        ax.set_xticks(x)
        ax.set_xticklabels([str(v).replace("drop_", "-") for v in part["feature_set"]], rotation=38, ha="right", fontsize=7)
        ax.grid(axis="y", color=GRID, linestyle=":", linewidth=0.6)
        ax.text(0.5, -0.44, letter, transform=ax.transAxes, ha="center", va="top", fontweight="bold", fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("Score")
    axes[0].legend(frameon=False, fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(outdir / "fig_feature_group_ablation.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    selected = selective[
        (selective["feature_set"] == "core")
        & (selective["split"].isin(["stratified_holdout", "chronological_80_20"]))
    ].copy()
    fig, ax = plt.subplots(figsize=(5.8, 3.5))
    for split, color, marker in [
        ("stratified_holdout", BLUE, "o"),
        ("chronological_80_20", TEAL, "s"),
    ]:
        part = selected[selected["split"] == split].sort_values("kept_fraction")
        ax.plot(part["kept_fraction"], part["macro_f1"], marker=marker, color=color, label=split.replace("_", " "))
    ax.set_xlabel("Retained fraction after confidence filtering")
    ax.set_ylabel("Macro-F1 on retained records")
    ax.set_ylim(0.70, 1.01)
    ax.grid(axis="both", color=GRID, linestyle=":", linewidth=0.6)
    ax.legend(frameon=False, fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(outdir / "fig_selective_prediction.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    wall_start = time.perf_counter()
    wall_start_iso = pd.Timestamp.now().isoformat()
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    df = load_data(args.data, args.sample_rows, args.seed)
    encoder = LabelEncoder()
    encoder.fit(df["target"])

    strat_train, strat_test = train_test_split(
        np.arange(len(df)),
        test_size=0.2,
        stratify=df["target"],
        random_state=args.seed,
    )
    chrono_train, chrono_test = chronological_indices(df, train_fraction=0.8)

    feature_sets = {"core": CORE_FEATURES}
    for group_name, group_features in FEATURE_GROUPS.items():
        feature_sets[f"drop_{group_name}"] = [f for f in CORE_FEATURES if f not in set(group_features)]

    summary_rows: list[dict] = []
    confusion_all: list[dict] = []
    selective_all: list[dict] = []
    decile_all: list[dict] = []

    main_splits = [
        ("stratified_holdout", strat_train, strat_test),
        ("chronological_80_20", chrono_train, chrono_test),
    ]

    for split_name, train_idx, test_idx in main_splits:
        for feature_set, features in feature_sets.items():
            run_id = f"{split_name}__{feature_set}"
            print(f"Running {run_id} with {len(features)} features")
            row, bundle = evaluate_run(df, encoder, run_id, feature_set, split_name, features, train_idx, test_idx, args)
            summary_rows.append(row)
            confusion_all.extend(confusion_rows(bundle, run_id, feature_set, split_name))
            if feature_set in {"core", "drop_threat_descriptors"}:
                selective_all.extend(selective_rows(bundle, run_id, feature_set, split_name))
                decile_all.extend(confidence_decile_rows(bundle, run_id, feature_set, split_name))

    for split_name, train_idx, test_idx in make_forward_splits(df):
        for feature_set in ["core", "drop_threat_descriptors"]:
            run_id = f"{split_name}__{feature_set}"
            print(f"Running {run_id} with {len(feature_sets[feature_set])} features")
            row, bundle = evaluate_run(
                df, encoder, run_id, feature_set, split_name, feature_sets[feature_set], train_idx, test_idx, args
            )
            summary_rows.append(row)
            confusion_all.extend(confusion_rows(bundle, run_id, feature_set, split_name))

    summary = pd.DataFrame(summary_rows)
    selective = pd.DataFrame(selective_all)
    pd.DataFrame(confusion_all).to_csv(args.outdir / "strengthening_confusions.csv", index=False)
    pd.DataFrame(decile_all).to_csv(args.outdir / "confidence_deciles.csv", index=False)
    selective.to_csv(args.outdir / "selective_prediction.csv", index=False)
    summary.to_csv(args.outdir / "strengthening_summary.csv", index=False)
    save_figures(summary, selective, args.outdir)

    metadata = {
        "rows": len(df),
        "sample_rows_argument": args.sample_rows,
        "time_min": str(df["Generate Time Parsed"].min()),
        "time_max": str(df["Generate Time Parsed"].max()),
        "device": args.device,
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "started_at": wall_start_iso,
        "ended_at": pd.Timestamp.now().isoformat(),
        "wall_seconds": float(time.perf_counter() - wall_start),
        "sum_fit_seconds": float(summary["fit_seconds"].sum()),
        "sum_predict_seconds": float(summary["predict_seconds"].sum()),
        "note": "Temporal splits are within-day forward-shift stress tests, not long-term temporal validation.",
        "feature_groups": FEATURE_GROUPS,
        "outputs": [
            "strengthening_summary.csv",
            "strengthening_confusions.csv",
            "selective_prediction.csv",
            "confidence_deciles.csv",
            "fig_feature_group_ablation.png",
            "fig_selective_prediction.png",
        ],
    }
    (args.outdir / "strengthening_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    if not selective.empty:
        print(selective.to_string(index=False))


if __name__ == "__main__":
    main()
