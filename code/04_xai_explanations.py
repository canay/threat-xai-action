"""Generate SHAP and LIME explanation artifacts for Paper 1.

The script uses the processed paper-specific dataset and the core feature
configuration. It retrains the XGBoost model with the manuscript parameters,
then writes global SHAP, class-wise SHAP, summary swarm SHAP, and local LIME outputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from xgboost import XGBClassifier


# Publication-neutral figure palette
PRIMARY = "#00539C"     # Deep Sapphire
TEAL = "#008080"        # Muted Teal
CRIMSON = "#C00000"     # Dark Red
GRAY_TEXT = "#333333"
GRAY_LINE = "#888888"
GRAY_GRID = "#E2E8F0"
GRAY_BG = "#F8FAFC"

SHAP_CMAP = LinearSegmentedColormap.from_list("q1_shap", [PRIMARY, "#DCE4EE", CRIMSON])
HEATMAP_CMAP = LinearSegmentedColormap.from_list("q1_heat", [GRAY_BG, "#80B7D8", PRIMARY])


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


def set_q1_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "axes.titleweight": "bold",
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": GRAY_LINE,
        "axes.linewidth": 1.0,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.color": GRAY_TEXT,
        "ytick.color": GRAY_TEXT,
        "text.color": GRAY_TEXT,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/processed/threat_five_class.csv"),
        help="Processed Paper 1 dataset.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("results/xai"),
        help="Directory for explanation artifacts.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_preprocessor(x: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[int]]:
    numeric_cols = [c for c in ["Source Port", "Destination Port", "Risk of app"] if c in x.columns]
    categorical_cols = [c for c in CORE_FEATURES if c not in numeric_cols]
    categorical_idx = list(range(len(categorical_cols)))
    preprocessor = ColumnTransformer(
        [
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                        ),
                    ]
                ),
                categorical_cols,
            ),
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_cols),
        ]
    )
    return preprocessor, categorical_cols + numeric_cols, categorical_idx


def save_global_shap(outdir: Path, feature_names: list[str], shap_array: np.ndarray) -> pd.DataFrame:
    global_mean = np.mean(np.abs(shap_array), axis=(0, 2))
    global_df = pd.DataFrame({"feature": feature_names, "mean_abs_shap": global_mean})
    global_df = global_df.sort_values("mean_abs_shap", ascending=False)
    global_df.to_csv(outdir / "xai_shap_global_importance.csv", index=False)

    top = global_df.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(3.6, 3.8))
    
    # Premium background striping
    y_pos = np.arange(len(top))
    for yi in y_pos[1::2]:
        ax.axhspan(yi - 0.5, yi + 0.5, color=GRAY_BG, zorder=0, lw=0)

    ax.barh(top["feature"], top["mean_abs_shap"], color=PRIMARY, edgecolor="white", linewidth=0.8, zorder=3)
    ax.set_xlabel("Mean Absolute SHAP Value", fontweight="bold")
    ax.grid(axis="x", color=GRAY_GRID, linestyle="--", linewidth=1.0, zorder=1)
    ax.set_axisbelow(True)
    ax.set_yticklabels(top["feature"], fontweight="bold")

    fig.tight_layout()
    fig.savefig(outdir / "fig_xai_shap_global.png")
    plt.close(fig)
    return global_df


def save_classwise_shap(
    outdir: Path, feature_names: list[str], class_names: list[str], shap_array: np.ndarray, global_df: pd.DataFrame
) -> None:
    # This is a Single Column Heatmap with Y-axis labels!
    class_mean = np.mean(np.abs(shap_array), axis=0)
    top_features = global_df.head(10)["feature"].tolist()
    top_idx = [feature_names.index(f) for f in top_features]

    fig, ax = plt.subplots(figsize=(3.6, 4.0))
    heat = class_mean[top_idx, :]
    image = ax.imshow(heat, aspect="auto", cmap=HEATMAP_CMAP)
    
    ax.set_yticks(np.arange(len(top_idx)))
    ax.set_yticklabels(top_features, fontweight="bold", color=GRAY_TEXT, fontsize=7.5)
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=35, ha="right", fontweight="bold", color=GRAY_TEXT, fontsize=8)
    
    max_heat = np.nanmax(heat) if np.isfinite(heat).any() else 0.0
    for i, feature_idx in enumerate(top_idx):
        for j in range(len(class_names)):
            value = class_mean[feature_idx, j]
            text_color = "white" if max_heat > 0 and value > 0.65 * max_heat else GRAY_TEXT
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7.5, color=text_color, fontweight="bold")
            
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Mean Absolute SHAP", fontweight="bold")
    
    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_visible(False)
        
    fig.tight_layout()
    fig.savefig(outdir / "fig_xai_shap_classwise.png")
    plt.close(fig)


def save_shap_summary_swarm(
    outdir: Path,
    feature_names: list[str],
    shap_array: np.ndarray,
    global_df: pd.DataFrame,
    x_sample_enc: np.ndarray,
    pred_classes: np.ndarray,
) -> None:
    # Double column swarm plot!
    top_features = global_df.head(10)["feature"].tolist()
    top_idx = [feature_names.index(f) for f in top_features]
    predicted_class_shap = shap_array[np.arange(shap_array.shape[0]), :, pred_classes]
    rng = np.random.default_rng(42)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))

    max_abs = np.nanpercentile(np.abs(predicted_class_shap[:, top_idx]), 99.2)
    if not np.isfinite(max_abs) or max_abs <= 0:
        max_abs = 1.0

    # Background striping
    for row in range(len(top_idx)):
        if row % 2 == 1:
            ax.axhspan(row - 0.5, row + 0.5, color=GRAY_BG, zorder=0, lw=0)

    scatter_handle = None
    for row, feature_idx in enumerate(top_idx):
        shap_values = predicted_class_shap[:, feature_idx]
        feature_values = np.asarray(x_sample_enc[:, feature_idx], dtype=float)
        finite = np.isfinite(feature_values)
        if finite.any() and np.nanmax(feature_values[finite]) > np.nanmin(feature_values[finite]):
            colors = (feature_values - np.nanmin(feature_values[finite])) / (
                np.nanmax(feature_values[finite]) - np.nanmin(feature_values[finite])
            )
        else:
            colors = np.full_like(feature_values, 0.5, dtype=float)
        y_jitter = rng.normal(0.0, 0.08, size=shap_values.shape[0])
        scatter_handle = ax.scatter(
            shap_values,
            row + y_jitter,
            c=colors,
            cmap=SHAP_CMAP,
            s=8,
            alpha=0.6,
            linewidths=0,
            rasterized=True,
            zorder=3
        )

    ax.axvline(0, color=GRAY_LINE, linewidth=1.2, zorder=2)
    ax.set_yticks(np.arange(len(top_idx)))
    ax.set_yticklabels(top_features, fontweight="bold", color=GRAY_TEXT, fontsize=9)
    ax.set_ylim(len(top_idx) - 0.5, -0.5)
    ax.set_xlim(-1.12 * max_abs, 1.12 * max_abs)
    ax.set_xlabel("SHAP Value for Predicted Class", fontweight="bold", fontsize=10)
    
    ax.grid(axis="x", color=GRAY_GRID, linestyle="--", linewidth=1.0, zorder=1)
    ax.set_axisbelow(True)
    
    if scatter_handle is not None:
        colorbar = fig.colorbar(scatter_handle, ax=ax, fraction=0.03, pad=0.02)
        colorbar.set_ticks([0, 1])
        colorbar.set_ticklabels(["Low", "High"], fontweight="bold")
        colorbar.set_label("Feature Value", fontweight="bold")

    fig.tight_layout()
    fig.savefig(outdir / "fig_xai_shap_summary.png")
    plt.close(fig)


def save_lime_group(
    outdir: Path,
    lime_df: pd.DataFrame,
    classes: list[str],
    filename: str,
    figsize: tuple[float, float],
) -> None:
    letters = ["(a)", "(b)", "(c)"]
    fig, axes = plt.subplots(len(classes), 1, figsize=figsize)
    if len(classes) == 1:
        axes = [axes]

    for ax, class_name, letter in zip(axes, classes, letters):
        rows = lime_df[lime_df["class"] == class_name].iloc[::-1]
        y_pos = np.arange(len(rows))
        weights = rows["weight"].to_numpy()
        colors = [PRIMARY if weight >= 0 else CRIMSON for weight in weights]
        
        # Background striping
        for yi in y_pos[1::2]:
            ax.axhspan(yi - 0.5, yi + 0.5, color=GRAY_BG, zorder=0, lw=0)
            
        ax.barh(y_pos, weights, color=colors, edgecolor="white", linewidth=0.8, zorder=3)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(rows["feature_rule"], fontsize=8, fontweight="bold")
        ax.axvline(0, color=GRAY_LINE, linewidth=1.2, zorder=2)
        ax.set_title(class_name, fontsize=9, loc="left", fontweight="bold", pad=4)
        
        ax.grid(axis="x", color=GRAY_GRID, linestyle="--", linewidth=1.0, zorder=1)
        ax.set_axisbelow(True)

    axes[-1].set_xlabel("LIME Local Surrogate Weight", fontweight="bold")
    fig.tight_layout()
    fig.savefig(outdir / filename)
    plt.close(fig)


def main() -> None:
    set_q1_style()
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    args.outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data, usecols=["target"] + CORE_FEATURES, low_memory=False)
    for col in ["Source Port", "Destination Port", "Risk of app"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    x = df[CORE_FEATURES]
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["target"])
    class_names = list(label_encoder.classes_)

    preprocessor, feature_names, categorical_idx = build_preprocessor(x)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=args.seed
    )
    x_train_enc = preprocessor.fit_transform(x_train)
    x_test_enc = preprocessor.transform(x_test)

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        tree_method="hist",
        objective="multi:softprob",
        num_class=len(class_names),
        base_score=0.5,
        eval_metric="mlogloss",
        random_state=args.seed,
        n_jobs=4,
    )
    model.fit(x_train_enc, y_train)
    y_pred = model.predict(x_test_enc)

    shap_indices: list[int] = []
    for cls in range(len(class_names)):
        idx = np.where(y_test == cls)[0]
        take = min(len(idx), 250)
        shap_indices.extend(rng.choice(idx, size=take, replace=False).tolist())
    shap_indices = np.array(sorted(shap_indices))

    import shap.explainers._tree
    original_decode = getattr(shap.explainers._tree, "decode_ubjson_buffer", None)
    
    if original_decode:
        def patched_decode(*args, **kwargs):
            jmodel = original_decode(*args, **kwargs)
            try:
                base_score = jmodel["learner"]["learner_model_param"]["base_score"]
                if isinstance(base_score, str) and base_score.startswith("["):
                    jmodel["learner"]["learner_model_param"]["base_score"] = "0.5"
            except KeyError:
                pass
            return jmodel
        shap.explainers._tree.decode_ubjson_buffer = patched_decode

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(x_test_enc[shap_indices])
    finally:
        if original_decode:
            shap.explainers._tree.decode_ubjson_buffer = original_decode
    if isinstance(shap_values, list):
        shap_array = np.stack(shap_values, axis=2)
    else:
        shap_array = np.asarray(shap_values)
        if shap_array.shape[1] != len(feature_names) and shap_array.shape[2] == len(feature_names):
            shap_array = np.transpose(shap_array, (0, 2, 1))

    global_df = save_global_shap(args.outdir, feature_names, shap_array)
    save_classwise_shap(args.outdir, feature_names, class_names, shap_array, global_df)
    save_shap_summary_swarm(
        args.outdir,
        feature_names,
        shap_array,
        global_df,
        np.asarray(x_test_enc[shap_indices]),
        np.asarray(y_pred[shap_indices]),
    )

    train_sample_size = min(6000, x_train_enc.shape[0])
    train_sample_idx = rng.choice(np.arange(x_train_enc.shape[0]), size=train_sample_size, replace=False)
    lime_explainer = LimeTabularExplainer(
        np.asarray(x_train_enc[train_sample_idx]),
        feature_names=feature_names,
        class_names=class_names,
        categorical_features=categorical_idx,
        mode="classification",
        discretize_continuous=True,
        random_state=args.seed,
    )

    selected = []
    for cls, class_name in enumerate(class_names):
        idx = np.where((y_test == cls) & (y_pred == cls))[0]
        if len(idx):
            selected.append((cls, class_name, int(idx[0])))

    lime_records = []
    x_test_np = np.asarray(x_test_enc)
    for cls, class_name, idx in selected:
        explanation = lime_explainer.explain_instance(
            x_test_np[idx],
            lambda rows: model.predict_proba(np.asarray(rows)),
            labels=[cls],
            num_features=7,
            num_samples=1200,
        )
        values = explanation.as_list(label=cls)
        lime_records.extend(
            {"class": class_name, "test_index": idx, "feature_rule": name, "weight": weight}
            for name, weight in values
        )
    lime_df = pd.DataFrame(lime_records)
    lime_df.to_csv(args.outdir / "xai_lime_local_examples.csv", index=False)
    save_lime_group(
        args.outdir,
        lime_df,
        ["Allow", "Deny", "Drop"],
        "fig_xai_lime_local_primary.png",
        (3.6, 6.0),
    )
    save_lime_group(
        args.outdir,
        lime_df,
        ["Reset-Both", "Reset-Server"],
        "fig_xai_lime_local_reset.png",
        (3.6, 4.0),
    )

    summary = {
        "model": "XGBoost",
        "split": "stratified 80/20, random_state=42",
        "shap_sample_size": int(len(shap_indices)),
        "lime_training_sample_size": int(train_sample_size),
        "classes": class_names,
        "top_global_shap_features": global_df.head(10).to_dict(orient="records"),
        "lime_selected_classes": [class_name for _, class_name, _ in selected],
    }
    (args.outdir / "xai_generation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
