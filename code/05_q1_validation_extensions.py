"""Additional validation checks for the firewall action manuscript.

The script adds manuscript-facing robustness evidence beyond the main VPS
benchmark: chronological holdout, a stricter minimal-context feature audit,
probability calibration diagnostics, error-pair summaries, and SHAP rank
stability for the selected XGBoost model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
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

FEATURE_SETS = {
    "core": CORE_FEATURES,
    "no_threat_descriptors": [
        feature
        for feature in CORE_FEATURES
        if feature not in {"Threat/Content Name", "Threat/Content Type", "Severity"}
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

BLUE = "#3D79A8"
TEAL = "#2A9D8F"
SLATE = "#5E7182"
GRID = "#B8C6D1"
EDGE = "#243342"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/processed/threat_five_class.csv"))
    parser.add_argument("--outdir", type=Path, default=Path("results/extensions"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shap-sample", type=int, default=650)
    return parser.parse_args()


def build_pipeline(features: list[str], seed: int) -> Pipeline:
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


def expected_calibration_error(y_true: np.ndarray, proba: np.ndarray, bins: int = 10) -> tuple[float, pd.DataFrame]:
    pred = proba.argmax(axis=1)
    confidence = proba.max(axis=1)
    correct = (pred == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi == 1.0:
            mask = (confidence >= lo) & (confidence <= hi)
        else:
            mask = (confidence >= lo) & (confidence < hi)
        count = int(mask.sum())
        if count == 0:
            acc = np.nan
            conf = np.nan
        else:
            acc = float(correct[mask].mean())
            conf = float(confidence[mask].mean())
            ece += (count / len(y_true)) * abs(acc - conf)
        rows.append({"bin_low": lo, "bin_high": hi, "count": count, "accuracy": acc, "confidence": conf})
    return float(ece), pd.DataFrame(rows)


def multiclass_brier(y_true: np.ndarray, proba: np.ndarray) -> float:
    target = np.zeros_like(proba)
    target[np.arange(len(y_true)), y_true] = 1.0
    return float(np.mean(np.sum((proba - target) ** 2, axis=1)))


def evaluate_split(
    df: pd.DataFrame,
    labels: list[str],
    encoder: LabelEncoder,
    feature_set: str,
    split_name: str,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    seed: int,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    features = FEATURE_SETS[feature_set]
    x_train = df.iloc[train_idx][features]
    x_test = df.iloc[test_idx][features]
    y_train = encoder.transform(df.iloc[train_idx]["target"].astype(str))
    y_test = encoder.transform(df.iloc[test_idx]["target"].astype(str))

    pipe = build_pipeline(features, seed)
    pipe.fit(x_train, y_train)
    pred = pipe.predict(x_test)
    proba = pipe.predict_proba(x_test)
    ece, calibration_bins = expected_calibration_error(y_test, proba)
    confidence = proba.max(axis=1)
    correct = pred == y_test

    summary = {
        "model": "XGBoost",
        "feature_set": feature_set,
        "split": split_name,
        "train_rows": len(train_idx),
        "test_rows": len(test_idx),
        "features": len(features),
        "accuracy": accuracy_score(y_test, pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, pred),
        "macro_f1": f1_score(y_test, pred, average="macro", labels=np.arange(len(labels)), zero_division=0),
        "weighted_f1": f1_score(y_test, pred, average="weighted", zero_division=0),
        "log_loss": log_loss(y_test, proba, labels=np.arange(len(labels))),
        "brier": multiclass_brier(y_test, proba),
        "ece_10": ece,
        "errors": int((~correct).sum()),
        "mean_confidence_correct": float(confidence[correct].mean()) if correct.any() else np.nan,
        "mean_confidence_wrong": float(confidence[~correct].mean()) if (~correct).any() else np.nan,
    }

    cm = confusion_matrix(y_test, pred, labels=np.arange(len(labels)))
    confusion_rows = []
    for i, true_label in enumerate(labels):
        for j, pred_label in enumerate(labels):
            count = int(cm[i, j])
            if count:
                confusion_rows.append(
                    {
                        "feature_set": feature_set,
                        "split": split_name,
                        "true": true_label,
                        "predicted": pred_label,
                        "count": count,
                    }
                )
    calibration_bins.insert(0, "feature_set", feature_set)
    calibration_bins.insert(1, "split", split_name)
    return summary, pd.DataFrame(confusion_rows), calibration_bins


def chronological_indices(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, pd.Timestamp, pd.Timestamp]:
    ordered = df.sort_values("Generate Time Parsed").index.to_numpy()
    cut = int(len(ordered) * 0.8)
    return ordered[:cut], ordered[cut:], df.loc[ordered[:cut], "Generate Time Parsed"].max(), df.loc[ordered[cut:], "Generate Time Parsed"].min()


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    return ranks


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = rankdata(a)
    rb = rankdata(b)
    if np.std(ra) == 0 or np.std(rb) == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def shap_stability(df: pd.DataFrame, encoder: LabelEncoder, outdir: Path, seed: int, sample_size: int) -> pd.DataFrame:
    features = FEATURE_SETS["core"]
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["target"], random_state=seed)
    y_train = encoder.transform(train_df["target"].astype(str))
    pipe = build_pipeline(features, seed)
    pipe.fit(train_df[features], y_train)

    transformed_test = pipe.named_steps["preprocess"].transform(test_df[features])
    model = pipe.named_steps["model"]
    feature_names = list(pipe.named_steps["preprocess"].get_feature_names_out())
    labels = list(encoder.classes_)

    baseline_values = None
    baseline_top = None
    rows = []
    for sample_seed in [11, 23, 42, 71, 97]:
        sampled = (
            test_df.assign(_encoded_target=encoder.transform(test_df["target"].astype(str)))
            .groupby("_encoded_target", group_keys=False)
            .apply(lambda part: part.sample(min(len(part), max(1, sample_size // len(labels))), random_state=sample_seed))
        )
        sample_positions = test_df.index.get_indexer(sampled.index)
        x_sample = transformed_test[sample_positions]
        import shap.explainers._tree
        _orig_float = float
        def _patched_float(val):
            if isinstance(val, str) and val.startswith('['):
                return 0.5
            return _orig_float(val)
        
        try:
            builtins = __import__('builtins')
            builtins.float = _patched_float
            explainer = shap.TreeExplainer(model)
        finally:
            builtins.float = _orig_float
        shap_values = explainer.shap_values(x_sample)
        shap_array = np.asarray(shap_values)
        if shap_array.ndim == 3 and shap_array.shape[0] == len(labels):
            shap_array = np.moveaxis(shap_array, 0, 2)
        global_values = np.mean(np.abs(shap_array), axis=(0, 2))
        top_idx = np.argsort(global_values)[::-1][:10]
        top_features = [feature_names[i] for i in top_idx]
        if baseline_values is None:
            baseline_values = global_values
            baseline_top = set(top_features)
            rho = 1.0
            overlap = 1.0
        else:
            rho = spearman(baseline_values, global_values)
            overlap = len(baseline_top & set(top_features)) / len(baseline_top)
        rows.append(
            {
                "sample_seed": sample_seed,
                "sample_rows": len(sampled),
                "top10_overlap_with_first_sample": overlap,
                "spearman_with_first_sample": rho,
                "top_features": "; ".join(top_features),
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(outdir / "q1_shap_stability.csv", index=False)
    return result


def save_summary_figure(summary: pd.DataFrame, outdir: Path) -> None:
    order = [
        ("stratified_holdout", "core"),
        ("stratified_holdout", "no_threat_descriptors"),
        ("stratified_holdout", "minimal_context"),
        ("chronological_holdout", "core"),
        ("chronological_holdout", "no_threat_descriptors"),
        ("chronological_holdout", "minimal_context"),
    ]
    labels = [f"{split.replace('_', ' ')}\n{feature.replace('_', ' ')}" for split, feature in order]
    rows = [summary[(summary["split"] == split) & (summary["feature_set"] == feature)].iloc[0] for split, feature in order]
    macro = [row["macro_f1"] for row in rows]
    bacc = [row["balanced_accuracy"] for row in rows]
    ece = [row["ece_10"] for row in rows]
    x = np.arange(len(order))

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.55), gridspec_kw={"width_ratios": [1.55, 1.0]})
    ax = axes[0]
    width = 0.36
    ax.bar(x - width / 2, macro, width, label="Macro-F1", color=BLUE, edgecolor=EDGE, linewidth=0.45)
    ax.bar(x + width / 2, bacc, width, label="Balanced acc.", color=TEAL, edgecolor=EDGE, linewidth=0.45)
    ax.set_ylim(0.80, 1.005)
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=32, ha="right", fontsize=7)
    ax.grid(axis="y", color=GRID, linestyle=":", linewidth=0.6)
    ax.legend(frameon=False, fontsize=7, loc="lower left")
    ax.text(0.5, -0.39, "(a)", transform=ax.transAxes, ha="center", va="top", fontweight="bold", fontsize=8)

    ax = axes[1]
    ax.bar(x, ece, color=SLATE, edgecolor=EDGE, linewidth=0.45)
    ax.set_ylabel("ECE, 10 bins")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=32, ha="right", fontsize=7)
    ax.grid(axis="y", color=GRID, linestyle=":", linewidth=0.6)
    ax.text(0.5, -0.39, "(b)", transform=ax.transAxes, ha="center", va="top", fontweight="bold", fontsize=8)
    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(outdir / "fig_q1_validation_checks.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data, low_memory=False)
    df["target"] = df["target"].astype(str)
    df["Generate Time Parsed"] = pd.to_datetime(df["Generate Time"], errors="coerce")
    encoder = LabelEncoder()
    encoder.fit(df["target"])
    labels = list(encoder.classes_)

    strat_train, strat_test = train_test_split(
        np.arange(len(df)),
        test_size=0.2,
        stratify=df["target"],
        random_state=args.seed,
    )
    chrono_train, chrono_test, chrono_train_end, chrono_test_start = chronological_indices(df)

    class_count_rows = []
    for split_name, train_idx, test_idx in [
        ("stratified_holdout", strat_train, strat_test),
        ("chronological_holdout", chrono_train, chrono_test),
    ]:
        for part_name, idx in [("train", train_idx), ("test", test_idx)]:
            counts = df.iloc[idx]["target"].value_counts()
            for label in labels:
                class_count_rows.append(
                    {
                        "split": split_name,
                        "partition": part_name,
                        "class": label,
                        "count": int(counts.get(label, 0)),
                    }
                )

    summaries = []
    confusions = []
    calibration_bins = []
    for split_name, train_idx, test_idx in [
        ("stratified_holdout", strat_train, strat_test),
        ("chronological_holdout", chrono_train, chrono_test),
    ]:
        for feature_set in ["core", "no_threat_descriptors", "minimal_context"]:
            summary, confusion, calibration = evaluate_split(
                df, labels, encoder, feature_set, split_name, train_idx, test_idx, args.seed
            )
            summaries.append(summary)
            confusions.append(confusion)
            calibration_bins.append(calibration)

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(args.outdir / "q1_validation_summary.csv", index=False)
    pd.concat(confusions, ignore_index=True).to_csv(args.outdir / "q1_validation_confusions.csv", index=False)
    pd.concat(calibration_bins, ignore_index=True).to_csv(args.outdir / "q1_calibration_bins.csv", index=False)
    pd.DataFrame(class_count_rows).to_csv(args.outdir / "q1_validation_class_counts.csv", index=False)
    shap_df = shap_stability(df, encoder, args.outdir, args.seed, args.shap_sample)
    save_summary_figure(summary_df, args.outdir)

    metadata = {
        "chronological_split": {
            "train_until": str(chrono_train_end),
            "test_from": str(chrono_test_start),
            "note": "Single-day chronological holdout; this is a temporal stress check, not cross-period validation.",
        },
        "labels": labels,
        "feature_sets": FEATURE_SETS,
        "shap_stability_mean_top10_overlap": float(shap_df["top10_overlap_with_first_sample"].iloc[1:].mean()),
        "shap_stability_mean_spearman": float(shap_df["spearman_with_first_sample"].iloc[1:].mean()),
    }
    (args.outdir / "q1_validation_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(summary_df.to_string(index=False))
    print(shap_df.to_string(index=False))


if __name__ == "__main__":
    main()
