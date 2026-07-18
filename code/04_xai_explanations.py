"""Generate SHAP and LIME explanation artifacts for Paper 1.

The script uses the processed paper-specific dataset and the core feature
configuration. It retrains the XGBoost model with the manuscript parameters,
then writes aggregate SHAP artifacts and deidentified manuscript-facing LIME figures.
Row-level local-surrogate tables remain controlled and are not written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import shap
import xgboost
from lime.lime_tabular import LimeTabularExplainer
from scipy.special import softmax
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
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
RENDERER_VERSION = "1.3.0"
AXIS_LABEL_SIZE_PT = 8
AXIS_TICK_SIZE_PT = 8


def resolve_lato_regular() -> tuple[font_manager.FontProperties, Path]:
    """Resolve the author-selected regular face for axis labels."""
    explicit = os.environ.get("LEAF_LATO_REGULAR")
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise RuntimeError(f"LEAF_LATO_REGULAR is not a font file: {path}")
        font_manager.fontManager.addfont(path)
        return font_manager.FontProperties(fname=str(path), weight="normal"), path
    for font_path in font_manager.findSystemFonts():
        path = Path(font_path)
        if path.name.lower() == "lato-regular.ttf":
            font_manager.fontManager.addfont(path)
            return font_manager.FontProperties(fname=str(path), weight="normal"), path
    raise RuntimeError("Lato Regular is required to render manuscript axis labels.")


def resolve_inter_regular() -> tuple[font_manager.FontProperties, Path]:
    """Resolve the author-selected regular face for internal plot text."""
    explicit = os.environ.get("LEAF_INTER_REGULAR")
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise RuntimeError(f"LEAF_INTER_REGULAR is not a font file: {path}")
        font_manager.fontManager.addfont(path)
        return font_manager.FontProperties(fname=str(path), weight="normal"), path
    preferred = {
        "inter-regular.ttf",
        "inter-variablefont_opsz,wght.ttf",
    }
    for font_path in font_manager.findSystemFonts():
        path = Path(font_path)
        if path.name.lower() in preferred:
            font_manager.fontManager.addfont(path)
            return font_manager.FontProperties(fname=str(path), weight="normal"), path
    raise RuntimeError("Inter Regular is required to render manuscript plot text.")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


AXIS_LABEL_FONT, AXIS_LABEL_FONT_PATH = resolve_lato_regular()
INTERNAL_TEXT_FONT, INTERNAL_TEXT_FONT_PATH = resolve_inter_regular()
INTERNAL_TEXT_FAMILY = INTERNAL_TEXT_FONT.get_name()


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
        "font.sans-serif": [INTERNAL_TEXT_FAMILY, "DejaVu Sans"],
        "font.size": 8,
        "axes.labelsize": AXIS_LABEL_SIZE_PT,
        "axes.titlesize": 9,
        "axes.titleweight": "normal",
        "xtick.labelsize": AXIS_TICK_SIZE_PT,
        "ytick.labelsize": AXIS_TICK_SIZE_PT,
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
        # At 600 dpi, 0.02 inches gives a physical 12-pixel outer margin.
        "savefig.pad_inches": 0.02,
    })


def apply_internal_tick_font(ax: plt.Axes) -> None:
    """Apply Inter Regular at the locked axis-tick size without bold faces."""
    for label in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
        label.set_fontproperties(INTERNAL_TEXT_FONT)
        label.set_fontweight("normal")
        label.set_fontsize(AXIS_TICK_SIZE_PT)


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

    ax.barh(y_pos, top["mean_abs_shap"], color=PRIMARY, edgecolor="white", linewidth=0.8, zorder=3)
    ax.set_xlabel(
        "Mean Absolute SHAP Value",
        fontproperties=AXIS_LABEL_FONT,
        fontsize=AXIS_LABEL_SIZE_PT,
    )
    ax.grid(axis="x", color=GRAY_GRID, linestyle="--", linewidth=1.0, zorder=1)
    ax.set_axisbelow(True)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top["feature"], fontproperties=INTERNAL_TEXT_FONT)
    apply_internal_tick_font(ax)

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
    ax.set_yticklabels(
        top_features,
        fontproperties=INTERNAL_TEXT_FONT,
        color=GRAY_TEXT,
        fontsize=AXIS_TICK_SIZE_PT,
    )
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_xticklabels(
        class_names,
        rotation=35,
        ha="right",
        fontproperties=INTERNAL_TEXT_FONT,
        color=GRAY_TEXT,
        fontsize=AXIS_TICK_SIZE_PT,
    )
    
    max_heat = np.nanmax(heat) if np.isfinite(heat).any() else 0.0
    for i, feature_idx in enumerate(top_idx):
        for j in range(len(class_names)):
            value = class_mean[feature_idx, j]
            text_color = "white" if max_heat > 0 and value > 0.65 * max_heat else GRAY_TEXT
            ax.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=7.5,
                color=text_color,
                fontproperties=INTERNAL_TEXT_FONT,
            )
            
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label(
        "Mean Absolute SHAP",
        fontproperties=AXIS_LABEL_FONT,
        fontsize=AXIS_LABEL_SIZE_PT,
    )
    apply_internal_tick_font(ax)
    for label in colorbar.ax.get_yticklabels():
        label.set_fontproperties(INTERNAL_TEXT_FONT)
        label.set_fontweight("normal")
        label.set_fontsize(AXIS_TICK_SIZE_PT)
    
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
    ax.set_yticklabels(
        top_features,
        fontproperties=INTERNAL_TEXT_FONT,
        color=GRAY_TEXT,
        fontsize=AXIS_TICK_SIZE_PT,
    )
    ax.set_ylim(len(top_idx) - 0.5, -0.5)
    ax.set_xlim(-1.12 * max_abs, 1.12 * max_abs)
    ax.set_xlabel(
        "SHAP Value for Predicted-Class Raw Margin",
        fontproperties=AXIS_LABEL_FONT,
        fontsize=AXIS_LABEL_SIZE_PT,
    )
    
    ax.grid(axis="x", color=GRAY_GRID, linestyle="--", linewidth=1.0, zorder=1)
    ax.set_axisbelow(True)
    
    if scatter_handle is not None:
        colorbar = fig.colorbar(scatter_handle, ax=ax, fraction=0.03, pad=0.02)
        colorbar.set_ticks([0, 1])
        colorbar.set_ticklabels(["Low", "High"])
        colorbar.set_label(
            "Feature Value",
            fontproperties=AXIS_LABEL_FONT,
            fontsize=AXIS_LABEL_SIZE_PT,
        )
        for label in colorbar.ax.get_yticklabels():
            label.set_fontproperties(INTERNAL_TEXT_FONT)
            label.set_fontweight("normal")
            label.set_fontsize(AXIS_TICK_SIZE_PT)

    apply_internal_tick_font(ax)

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
        ax.set_yticklabels(
            rows["feature_rule"],
            fontsize=AXIS_TICK_SIZE_PT,
            fontproperties=INTERNAL_TEXT_FONT,
        )
        ax.axvline(0, color=GRAY_LINE, linewidth=1.2, zorder=2)
        ax.set_title(class_name, fontsize=9, loc="left", fontproperties=INTERNAL_TEXT_FONT, pad=4)
        
        ax.grid(axis="x", color=GRAY_GRID, linestyle="--", linewidth=1.0, zorder=1)
        ax.set_axisbelow(True)
        apply_internal_tick_font(ax)

    axes[-1].set_xlabel(
        "LIME surrogate weight",
        fontproperties=AXIS_LABEL_FONT,
        fontsize=AXIS_LABEL_SIZE_PT,
    )
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
        eval_metric="mlogloss",
        random_state=args.seed,
        n_jobs=-1,
    )
    # Preserve XGBoost's fitted multiclass vector intercept. Overriding
    # base_score would create a different model from the canonical benchmark.
    model.fit(x_train_enc, y_train)
    y_pred = model.predict(x_test_enc)

    shap_indices: list[int] = []
    for cls in range(len(class_names)):
        idx = np.where(y_test == cls)[0]
        take = min(len(idx), 250)
        shap_indices.extend(rng.choice(idx, size=take, replace=False).tolist())
    shap_indices = np.array(sorted(shap_indices))

    # SHAP 0.51 reads the XGBoost 3.2 multiclass vector base score directly;
    # no parser monkeypatch or intercept substitution is applied.
    explainer = shap.TreeExplainer(
        model,
        data=None,
        feature_perturbation="tree_path_dependent",
        model_output="raw",
    )
    shap_values = explainer.shap_values(
        x_test_enc[shap_indices],
        check_additivity=True,
    )
    if isinstance(shap_values, list):
        shap_array = np.stack(shap_values, axis=2)
    else:
        shap_array = np.asarray(shap_values)
        if shap_array.shape[1] != len(feature_names) and shap_array.shape[2] == len(feature_names):
            shap_array = np.transpose(shap_array, (0, 2, 1))

    # Durable semantic check: TreeSHAP values must reconstruct the class-wise
    # pre-softmax XGBoost margins. The probability check separately verifies
    # that softmax(raw margins) matches predict_proba.
    n_diag = min(64, len(shap_indices))
    x_diag = np.asarray(x_test_enc[shap_indices[:n_diag]])
    phi_diag = np.asarray(shap_array[:n_diag])
    raw_margin = np.asarray(model.predict(x_diag, output_margin=True))
    expected_value = np.asarray(explainer.expected_value, dtype=float)
    if expected_value.ndim == 0:
        expected_value = np.repeat(expected_value, len(class_names))
    expected_value = expected_value.reshape(1, -1)
    reconstructed_margin = phi_diag.sum(axis=1) + expected_value
    probability_from_margin = softmax(raw_margin, axis=1)
    predicted_probability = np.asarray(model.predict_proba(x_diag))
    max_additivity_error = float(np.max(np.abs(reconstructed_margin - raw_margin)))
    max_probability_error = float(np.max(np.abs(probability_from_margin - predicted_probability)))
    if not np.allclose(reconstructed_margin, raw_margin, rtol=1e-4, atol=1e-4):
        raise RuntimeError(f"TreeSHAP raw-margin additivity check failed: {max_additivity_error:.3e}")
    if not np.allclose(probability_from_margin, predicted_probability, rtol=1e-6, atol=1e-6):
        raise RuntimeError(f"XGBoost raw-margin probability check failed: {max_probability_error:.3e}")

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
            example_id = f"lime_{class_name.lower().replace('-', '_')}_01"
            selected.append((cls, class_name, int(idx[0]), example_id))

    lime_records = []
    x_test_np = np.asarray(x_test_enc)
    for cls, class_name, idx, example_id in selected:
        explanation = lime_explainer.explain_instance(
            x_test_np[idx],
            lambda rows: model.predict_proba(np.asarray(rows)),
            labels=[cls],
            num_features=7,
            num_samples=1200,
        )
        values = explanation.as_list(label=cls)
        lime_records.extend(
            {"class": class_name, "example_id": example_id, "feature_rule": name, "weight": weight}
            for name, weight in values
        )
    lime_df = pd.DataFrame(lime_records)
    save_lime_group(
        args.outdir,
        lime_df,
        ["Allow", "Block", "Drop"],
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
    save_lime_group(
        args.outdir,
        lime_df,
        ["Reset-Server"],
        "fig_lime_local_explanation.png",
        (3.6, 2.2),
    )

    summary = {
        "renderer_version": RENDERER_VERSION,
        "figure_typography": {
            "axis_label_font_family": "Lato",
            "axis_label_font_weight": "regular",
            "axis_label_font_file_basename": AXIS_LABEL_FONT_PATH.name,
            "axis_label_font_file_sha256": file_sha256(AXIS_LABEL_FONT_PATH),
            "axis_label_font_size_pt": AXIS_LABEL_SIZE_PT,
            "axis_tick_font_size_pt": AXIS_TICK_SIZE_PT,
            "internal_text_font_family": "Inter",
            "internal_text_font_weight": "regular",
            "internal_text_font_file_basename": INTERNAL_TEXT_FONT_PATH.name,
            "internal_text_font_file_sha256": file_sha256(INTERNAL_TEXT_FONT_PATH),
            "bold_internal_text": False,
        },
        "model": "XGBoost",
        "split": "stratified 80/20, random_state=42",
        "shap_sample_size": int(len(shap_indices)),
        "lime_training_sample_size": int(train_sample_size),
        "classes": class_names,
        "selected_model_holdout_check": {
            "test_rows": int(len(y_test)),
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
            "macro_f1": float(f1_score(y_test, y_pred, average="macro")),
            "errors": int(np.sum(y_test != y_pred)),
        },
        "top_global_shap_features": global_df.head(10).to_dict(orient="records"),
        "lime_selected_classes": [class_name for _, class_name, _, _ in selected],
        "lime_release_boundary": "deidentified figures only; row-level local-surrogate table not written",
        "shap_semantics": {
            "shap_version": shap.__version__,
            "xgboost_version": xgboost.__version__,
            "feature_perturbation": "tree_path_dependent",
            "model_output": "raw",
            "background_rows": 0,
            "explained_output": "class-wise pre-softmax margin",
            "aggregation_global": "mean absolute SHAP over records and classes",
            "aggregation_classwise": "mean absolute SHAP over records for each class",
            "beeswarm": "signed SHAP for each record's predicted class",
        },
        "shap_additivity_diagnostic": {
            "rows": n_diag,
            "phi_shape": list(phi_diag.shape),
            "raw_margin_shape": list(raw_margin.shape),
            "expected_value_shape": list(expected_value.shape),
            "max_abs_margin_reconstruction_error": max_additivity_error,
            "max_abs_probability_error": max_probability_error,
            "margin_tolerance": {"rtol": 1e-4, "atol": 1e-4},
            "probability_tolerance": {"rtol": 1e-6, "atol": 1e-6},
            "passed": True,
        },
    }
    (args.outdir / "xai_generation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
