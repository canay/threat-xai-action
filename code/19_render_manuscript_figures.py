"""Render manuscript-facing descriptive and benchmark figures from saved evidence.

The script consumes only the aggregate processing manifest and canonical
benchmark CSV files. It does not require, read, or export event-level records.
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
import numpy as np
import pandas as pd


PRIMARY = "#00539C"
TEAL = "#008080"
CORAL = "#EEA47F"
GRAY_TEXT = "#333333"
GRAY_LINE = "#888888"
GRAY_GRID = "#E2E8F0"
GRAY_BG = "#F8FAFC"

MODEL_ORDER = [
    "LightGBM",
    "CatBoost",
    "XGBoost",
    "Random Forest",
    "Extra Trees",
    "Decision Tree",
]
FEATURE_SET_ORDER = [
    "core",
    "drop_threat_descriptors",
    "drop_application_context",
    "drop_zones_interfaces",
    "drop_network_endpoint_context",
    "drop_direction",
]
EXPECTED_CLASSES = {"Allow", "Block", "Drop", "Reset-Both", "Reset-Server"}
RENDERER_VERSION = "1.4.0"
FIG2_FIG3_AXIS_LABEL_SIZE_PT = 8
FIG2_FIG3_AXIS_TICK_SIZE_PT = 7


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


AXIS_LABEL_FONT, AXIS_LABEL_FONT_PATH = resolve_lato_regular()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--processing-manifest",
        type=Path,
        default=Path("data/processed/threat_dataset_processing_manifest.json"),
        help="Aggregate raw-to-processed manifest containing class_counts.",
    )
    parser.add_argument("--core-holdout", type=Path, required=True)
    parser.add_argument("--no-threat-holdout", type=Path, required=True)
    parser.add_argument("--core-cv", type=Path, required=True)
    parser.add_argument("--no-threat-cv", type=Path, required=True)
    parser.add_argument("--strengthening", type=Path, required=True)
    parser.add_argument(
        "--rolling-origin",
        type=Path,
        default=Path("results/q1_audit_revision/forward_chaining_chronological.csv"),
    )
    parser.add_argument(
        "--bootstrap-ci",
        type=Path,
        default=Path("results/q1_audit_revision/xgb_fixed_model_bootstrap_ci.csv"),
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("results/manuscript_figures"),
    )
    return parser.parse_args()


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": GRAY_LINE,
            "axes.linewidth": 1.0,
            "xtick.color": GRAY_TEXT,
            "ytick.color": GRAY_TEXT,
            "text.color": GRAY_TEXT,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            # At 600 dpi, 0.02 inches gives a physical 12-pixel outer margin.
            "savefig.pad_inches": 0.02,
        }
    )


def ordered_models(values: pd.Series) -> list[str]:
    present = set(values.astype(str))
    ordered = [model for model in MODEL_ORDER if model in present]
    ordered.extend(sorted(present.difference(ordered)))
    return ordered


def score_limits(values: np.ndarray, pad: float = 0.003) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        raise ValueError("No finite score values were found for plotting.")
    lower = max(0.0, float(finite.min()) - pad)
    upper = min(1.0, float(finite.max()) + pad)
    if upper - lower < 0.01:
        lower = max(0.0, upper - 0.01)
    return lower, upper


def require_probability_scores(frame: pd.DataFrame, columns: list[str], source: Path) -> None:
    values = frame[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0.0).any() or (values > 1.0).any():
        raise ValueError(f"{source} contains non-finite values or scores outside [0, 1].")


def sha256(path: Path, *, normalize_text_eol: bool = False) -> str:
    digest = hashlib.sha256()
    if normalize_text_eol:
        payload = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        digest.update(payload)
        return digest.hexdigest().upper()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def render_class_distribution(manifest_path: Path, output: Path) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    class_counts = manifest.get("class_counts")
    if not isinstance(class_counts, dict) or not class_counts:
        raise ValueError("Processing manifest must contain a non-empty class_counts object.")
    if set(class_counts) != EXPECTED_CLASSES:
        raise ValueError(f"Expected exactly the five manuscript classes: {sorted(EXPECTED_CLASSES)}")
    if int(manifest.get("rows_out", -1)) != sum(int(value) for value in class_counts.values()):
        raise ValueError("rows_out must equal the sum of class_counts in the processing manifest.")

    preferred = ["Drop", "Block", "Reset-Both", "Allow", "Reset-Server"]
    labels = [label for label in preferred if label in class_counts]
    labels.extend(sorted(set(class_counts).difference(labels)))
    counts = np.array([int(class_counts[label]) for label in labels], dtype=int)
    if (counts <= 0).any():
        raise ValueError("All class counts must be positive integers.")

    fig, ax = plt.subplots(figsize=(3.6, 2.5))
    y_pos = np.arange(len(labels))
    ax.barh(y_pos, counts, height=0.42, color=PRIMARY, zorder=3)
    ax.set_xscale("log")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(
        labels,
        fontweight="normal",
        fontsize=FIG2_FIG3_AXIS_TICK_SIZE_PT,
    )
    ax.invert_yaxis()
    ax.set_xlabel(
        "Number of records (log scale)",
        fontproperties=AXIS_LABEL_FONT,
        fontsize=FIG2_FIG3_AXIS_LABEL_SIZE_PT,
    )
    ax.tick_params(axis="x", labelsize=FIG2_FIG3_AXIS_TICK_SIZE_PT)
    ax.grid(axis="x", color=GRAY_GRID, linestyle="--", linewidth=0.8, zorder=1)
    ax.set_axisbelow(True)
    for row, value in enumerate(counts):
        ax.text(value * 1.12, row, f"{value:,}", va="center", fontsize=7.5, fontweight="bold")
    ax.set_xlim(max(1.0, float(counts.min()) / 5.0), float(counts.max()) * 5.0)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def load_holdout(core_path: Path, no_threat_path: Path) -> pd.DataFrame:
    core = pd.read_csv(core_path, usecols=["model", "f1_macro"]).rename(columns={"f1_macro": "core"})
    no_threat = pd.read_csv(no_threat_path, usecols=["model", "f1_macro"]).rename(
        columns={"f1_macro": "no_threat_descriptors"}
    )
    merged = core.merge(no_threat, on="model", how="inner", validate="one_to_one")
    if len(merged) != len(core) or len(merged) != len(no_threat):
        raise ValueError("Core and no-threat holdout files must contain the same models exactly once.")
    require_probability_scores(merged, ["core", "no_threat_descriptors"], core_path)
    order = ordered_models(merged["model"])
    merged["model"] = pd.Categorical(merged["model"], categories=order, ordered=True)
    return merged.sort_values("model").reset_index(drop=True)


def load_cv(core_path: Path, no_threat_path: Path) -> pd.DataFrame:
    columns = ["model", "test_f1_macro.mean", "test_f1_macro.std"]
    core = pd.read_csv(core_path, usecols=columns).rename(
        columns={"test_f1_macro.mean": "core_mean", "test_f1_macro.std": "core_std"}
    )
    no_threat = pd.read_csv(no_threat_path, usecols=columns).rename(
        columns={
            "test_f1_macro.mean": "no_threat_mean",
            "test_f1_macro.std": "no_threat_std",
        }
    )
    merged = core.merge(no_threat, on="model", how="inner", validate="one_to_one")
    if len(merged) != len(core) or len(merged) != len(no_threat):
        raise ValueError("Core and no-threat CV files must contain the same models exactly once.")
    require_probability_scores(
        merged,
        ["core_mean", "core_std", "no_threat_mean", "no_threat_std"],
        core_path,
    )
    order = ordered_models(merged["model"])
    merged["model"] = pd.Categorical(merged["model"], categories=order, ordered=True)
    return merged.sort_values("model").reset_index(drop=True)


def render_combined_benchmark(
    core_holdout: Path,
    no_threat_holdout: Path,
    core_cv: Path,
    no_threat_cv: Path,
    output: Path,
) -> None:
    holdout = load_holdout(core_holdout, no_threat_holdout)
    cv = load_cv(core_cv, no_threat_cv)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.9))

    y_holdout = np.arange(len(holdout))
    for row in y_holdout[1::2]:
        ax1.axhspan(row - 0.5, row + 0.5, color=GRAY_BG, zorder=0, lw=0)
    for row, item in holdout.iterrows():
        ax1.plot(
            [item["no_threat_descriptors"], item["core"]],
            [row, row],
            color=GRAY_LINE,
            lw=1.7,
            zorder=1,
        )
        delta = float(item["no_threat_descriptors"] - item["core"])
        ax1.annotate(
            f"{delta:+.3f}",
            xy=(max(item["core"], item["no_threat_descriptors"]), row),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            ha="left",
            fontsize=7,
            color=GRAY_LINE,
            fontweight="bold",
        )
    ax1.scatter(
        holdout["core"], y_holdout, color=PRIMARY, s=39, label="Core", zorder=3,
        edgecolor="white", linewidth=0.7,
    )
    ax1.scatter(
        holdout["no_threat_descriptors"], y_holdout, color=TEAL, s=39,
        label="No threat descriptors", zorder=3, edgecolor="white", linewidth=0.7,
    )
    ax1.set_yticks(y_holdout)
    ax1.set_yticklabels(
        holdout["model"].astype(str),
        fontweight="bold",
        fontsize=FIG2_FIG3_AXIS_TICK_SIZE_PT,
    )
    ax1.invert_yaxis()
    ax1.set_xlabel(
        "Holdout macro-F1",
        fontproperties=AXIS_LABEL_FONT,
        fontsize=FIG2_FIG3_AXIS_LABEL_SIZE_PT,
    )
    ax1.tick_params(axis="x", labelsize=FIG2_FIG3_AXIS_TICK_SIZE_PT)
    ax1.set_xlim(*score_limits(holdout[["core", "no_threat_descriptors"]].to_numpy(), pad=0.004))
    ax1.grid(axis="x", color=GRAY_GRID, linestyle="--", linewidth=0.8)
    ax1.set_axisbelow(True)
    ax1.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2)
    ax1.text(0.5, -0.25, "(a)", transform=ax1.transAxes, ha="center", va="top", fontweight="bold")

    y_cv = np.arange(len(cv))
    for row in y_cv[1::2]:
        ax2.axhspan(row - 0.5, row + 0.5, color=GRAY_BG, zorder=0, lw=0)
    ax2.errorbar(
        cv["core_mean"], y_cv - 0.12, xerr=cv["core_std"], fmt="o", color=PRIMARY,
        ecolor=PRIMARY, elinewidth=1.2, capsize=3.2, markersize=6,
        markeredgecolor="white", label="Core", zorder=3,
    )
    ax2.errorbar(
        cv["no_threat_mean"], y_cv + 0.12, xerr=cv["no_threat_std"], fmt="o", color=TEAL,
        ecolor=CORAL, elinewidth=1.2, capsize=3.2, markersize=6,
        markeredgecolor="white", label="No threat descriptors", zorder=3,
    )
    ax2.set_yticks(y_cv)
    ax2.set_yticklabels(
        cv["model"].astype(str),
        fontweight="bold",
        fontsize=FIG2_FIG3_AXIS_TICK_SIZE_PT,
    )
    ax2.invert_yaxis()
    ax2.set_xlabel(
        "Cross-validation macro-F1",
        fontproperties=AXIS_LABEL_FONT,
        fontsize=FIG2_FIG3_AXIS_LABEL_SIZE_PT,
    )
    ax2.tick_params(axis="x", labelsize=FIG2_FIG3_AXIS_TICK_SIZE_PT)
    means = cv[["core_mean", "no_threat_mean"]].to_numpy()
    stds = cv[["core_std", "no_threat_std"]].to_numpy()
    ax2.set_xlim(*score_limits(np.concatenate([(means - stds).ravel(), (means + stds).ravel()]), pad=0.003))
    ax2.grid(axis="x", color=GRAY_GRID, linestyle="--", linewidth=0.8)
    ax2.set_axisbelow(True)
    ax2.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2)
    ax2.text(0.5, -0.25, "(b)", transform=ax2.transAxes, ha="center", va="top", fontweight="bold")

    fig.tight_layout(pad=0.3, w_pad=4.5)
    fig.savefig(output)
    plt.close(fig)


def render_feature_group_validation(strengthening_path: Path, output: Path) -> None:
    required = {"split", "feature_set", "macro_f1", "balanced_accuracy"}
    frame = pd.read_csv(strengthening_path)
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{strengthening_path} is missing columns: {sorted(missing)}")
    frame = frame[
        frame["split"].isin(["stratified_holdout", "chronological_80_20"])
        & frame["feature_set"].isin(FEATURE_SET_ORDER)
    ].copy()
    expected_pairs = {
        (split, feature_set)
        for split in ["stratified_holdout", "chronological_80_20"]
        for feature_set in FEATURE_SET_ORDER
    }
    actual_pairs = set(zip(frame["split"], frame["feature_set"]))
    if actual_pairs != expected_pairs or frame.duplicated(["split", "feature_set"]).any():
        raise ValueError("Strengthening input must contain each required split/feature-set pair exactly once.")
    require_probability_scores(frame, ["macro_f1", "balanced_accuracy"], strengthening_path)
    frame["feature_set"] = pd.Categorical(
        frame["feature_set"], categories=FEATURE_SET_ORDER, ordered=True
    )

    display_labels = {
        "core": "Core",
        "drop_threat_descriptors": "− threat descriptors",
        "drop_application_context": "− application context",
        "drop_zones_interfaces": "− zones/interfaces",
        "drop_network_endpoint_context": "− endpoint context",
        "drop_direction": "− direction",
    }
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    for ax, split, title in zip(
        axes,
        ["stratified_holdout", "chronological_80_20"],
        ["Stratified holdout", "Chronological 80/20"],
    ):
        part = frame[frame["split"] == split].sort_values("feature_set")
        y_pos = np.arange(len(part))
        for row in y_pos[1::2]:
            ax.axhspan(row - 0.5, row + 0.5, color=GRAY_BG, zorder=0, lw=0)
        ax.scatter(part["macro_f1"], y_pos - 0.10, color=PRIMARY, s=35, label="Macro-F1", zorder=3)
        ax.scatter(
            part["balanced_accuracy"], y_pos + 0.10, color=TEAL, marker="s", s=30,
            label="Balanced accuracy", zorder=3,
        )
        for row, item in enumerate(part.to_dict("records")):
            ax.plot(
                [item["macro_f1"], item["balanced_accuracy"]],
                [row - 0.10, row + 0.10],
                color=GRAY_LINE,
                lw=0.8,
                zorder=1,
            )
        ax.set_yticks(y_pos)
        ax.set_yticklabels([display_labels[str(value)] for value in part["feature_set"]])
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Score", fontproperties=AXIS_LABEL_FONT)
        limits = score_limits(part[["macro_f1", "balanced_accuracy"]].to_numpy(), pad=0.025)
        ax.set_xlim(*limits)
        ax.grid(axis="x", color=GRAY_GRID, linestyle="--", linewidth=0.8)
        ax.set_axisbelow(True)
    axes[0].invert_yaxis()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.01), ncol=2)
    axes[0].text(0.5, -0.24, "(a)", transform=axes[0].transAxes, ha="center", va="top", fontweight="bold")
    axes[1].text(0.5, -0.24, "(b)", transform=axes[1].transAxes, ha="center", va="top", fontweight="bold")
    fig.tight_layout(pad=0.35, w_pad=4.5, rect=(0, 0, 1, 0.92))
    fig.savefig(output)
    plt.close(fig)


def load_rolling_origin(path: Path) -> pd.DataFrame:
    required = {
        "train_pct",
        "test_window",
        "macro_f1",
        "balanced_accuracy",
        "errors",
    }
    frame = pd.read_csv(path)
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    frame = frame.sort_values("train_pct").reset_index(drop=True)
    if len(frame) != 4 or frame["train_pct"].duplicated().any():
        raise ValueError("Rolling-origin input must contain exactly four unique training cutoffs.")
    require_probability_scores(frame, ["macro_f1", "balanced_accuracy"], path)
    errors = pd.to_numeric(frame["errors"], errors="coerce")
    if not np.isfinite(errors).all() or (errors < 0).any():
        raise ValueError("Rolling-origin errors must be finite non-negative counts.")
    frame["errors"] = errors.astype(int)
    return frame


def load_bootstrap_intervals(path: Path) -> pd.DataFrame:
    required = {
        "split",
        "feature_set",
        "point_estimate",
        "metric",
        "ci_low_95",
        "ci_high_95",
        "bootstrap_iterations",
        "uncertainty_scope",
    }
    frame = pd.read_csv(path)
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    splits = ["stratified_holdout", "chronological_holdout"]
    feature_sets = ["core", "no_threat_descriptors", "minimal_context"]
    metrics = ["macro_f1", "balanced_accuracy_fixed_labels"]
    frame = frame[
        frame["split"].isin(splits)
        & frame["feature_set"].isin(feature_sets)
        & frame["metric"].isin(metrics)
    ].copy()
    expected = {
        (split, feature_set, metric)
        for split in splits
        for feature_set in feature_sets
        for metric in metrics
    }
    actual = set(zip(frame["split"], frame["feature_set"], frame["metric"]))
    if actual != expected or frame.duplicated(["split", "feature_set", "metric"]).any():
        raise ValueError("Bootstrap input must contain each split/feature/metric interval exactly once.")
    require_probability_scores(frame, ["point_estimate", "ci_low_95", "ci_high_95"], path)
    if (frame["ci_low_95"] > frame["point_estimate"]).any() or (
        frame["point_estimate"] > frame["ci_high_95"]
    ).any():
        raise ValueError("Every point estimate must fall inside its 95% interval.")
    if set(pd.to_numeric(frame["bootstrap_iterations"])) != {1000}:
        raise ValueError("Bootstrap intervals must use exactly 1000 resamples.")
    if set(frame["uncertainty_scope"].astype(str)) != {
        "test_set_bootstrap_fixed_fitted_model"
    }:
        raise ValueError("Unexpected uncertainty scope in bootstrap input.")
    frame["split"] = pd.Categorical(frame["split"], categories=splits, ordered=True)
    frame["feature_set"] = pd.Categorical(
        frame["feature_set"], categories=feature_sets, ordered=True
    )
    frame["metric"] = pd.Categorical(frame["metric"], categories=metrics, ordered=True)
    return frame.sort_values(["split", "feature_set", "metric"]).reset_index(drop=True)


def render_temporal_uncertainty_validation(
    rolling_path: Path,
    bootstrap_path: Path,
    output_png: Path,
    output_pdf: Path,
) -> None:
    rolling = load_rolling_origin(rolling_path)
    intervals = load_bootstrap_intervals(bootstrap_path)
    support_font = font_manager.FontProperties(family="DejaVu Sans", size=7.2)

    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(7.2, 3.25),
        gridspec_kw={"width_ratios": [0.88, 1.25]},
    )

    x_pos = np.arange(len(rolling))
    ax1.plot(
        x_pos,
        rolling["macro_f1"],
        color=PRIMARY,
        marker="o",
        markersize=5.6,
        markeredgecolor="white",
        linewidth=1.7,
        label="Macro-F1",
        zorder=3,
    )
    ax1.plot(
        x_pos,
        rolling["balanced_accuracy"],
        color=TEAL,
        marker="s",
        markersize=5.2,
        markeredgecolor="white",
        linewidth=1.7,
        label="Balanced accuracy",
        zorder=3,
    )
    for x_value, error_count in zip(x_pos, rolling["errors"]):
        ax1.annotate(
            f"Err. {error_count}",
            xy=(x_value, max(rolling.loc[x_value, "macro_f1"], rolling.loc[x_value, "balanced_accuracy"])),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontproperties=support_font,
            color=GRAY_TEXT,
        )
    window_labels = [f"{str(value).replace('-', '–')}%" for value in rolling["test_window"]]
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(window_labels, fontproperties=support_font)
    ax1.set_xlabel(
        "Following 10% test window",
        fontproperties=AXIS_LABEL_FONT,
        fontsize=FIG2_FIG3_AXIS_LABEL_SIZE_PT,
    )
    ax1.set_ylabel(
        "Score",
        fontproperties=AXIS_LABEL_FONT,
        fontsize=FIG2_FIG3_AXIS_LABEL_SIZE_PT,
    )
    ax1.set_ylim(0.76, 1.02)
    ax1.set_yticks(np.arange(0.80, 1.01, 0.05))
    ax1.grid(axis="y", color=GRAY_GRID, linestyle="--", linewidth=0.8)
    ax1.set_axisbelow(True)
    ax1.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2)

    split_labels = {
        "stratified_holdout": "Stratified",
        "chronological_holdout": "Chronological",
    }
    feature_labels = {
        "core": "Core",
        "no_threat_descriptors": "No descriptors",
        "minimal_context": "Minimal context",
    }
    rows = [
        (split, feature_set)
        for split in ["stratified_holdout", "chronological_holdout"]
        for feature_set in ["core", "no_threat_descriptors", "minimal_context"]
    ]
    y_pos = np.arange(len(rows))
    for row in y_pos[1::2]:
        ax2.axhspan(row - 0.5, row + 0.5, color=GRAY_BG, zorder=0, lw=0)
    metric_specs = [
        ("macro_f1", -0.11, "o", PRIMARY, "Macro-F1"),
        ("balanced_accuracy_fixed_labels", 0.11, "s", TEAL, "Balanced accuracy"),
    ]
    for metric, offset, marker, color, label in metric_specs:
        part = intervals[intervals["metric"] == metric].copy()
        lookup = {
            (str(item["split"]), str(item["feature_set"])): item
            for item in part.to_dict("records")
        }
        points = np.array([lookup[row]["point_estimate"] for row in rows], dtype=float)
        lows = np.array([lookup[row]["ci_low_95"] for row in rows], dtype=float)
        highs = np.array([lookup[row]["ci_high_95"] for row in rows], dtype=float)
        ax2.errorbar(
            points,
            y_pos + offset,
            xerr=np.vstack([points - lows, highs - points]),
            fmt=marker,
            color=color,
            ecolor=color,
            elinewidth=1.1,
            capsize=2.6,
            markersize=5.2,
            markeredgecolor="white",
            label=label,
            zorder=3,
        )
    ax2.axhline(2.5, color=GRAY_LINE, linewidth=0.8)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(
        [f"{split_labels[split]} / {feature_labels[feature]}" for split, feature in rows],
        fontproperties=support_font,
    )
    ax2.invert_yaxis()
    ax2.set_xlim(0.76, 1.005)
    ax2.set_xticks(np.arange(0.80, 1.001, 0.05))
    ax2.set_xlabel(
        "Fixed-prediction estimate and 95% interval",
        fontproperties=AXIS_LABEL_FONT,
        fontsize=FIG2_FIG3_AXIS_LABEL_SIZE_PT,
    )
    ax2.grid(axis="x", color=GRAY_GRID, linestyle="--", linewidth=0.8)
    ax2.set_axisbelow(True)
    ax2.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2)

    ax1.text(0.5, -0.24, "(a)", transform=ax1.transAxes, ha="center", va="top")
    ax2.text(0.5, -0.24, "(b)", transform=ax2.transAxes, ha="center", va="top")
    fig.tight_layout(pad=0.35, w_pad=3.2, rect=(0, 0.02, 1, 0.96))
    fig.savefig(output_png, dpi=600)
    fig.savefig(output_pdf)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    set_style()
    render_class_distribution(args.processing_manifest, args.outdir / "fig_class_distribution.png")
    render_combined_benchmark(
        args.core_holdout,
        args.no_threat_holdout,
        args.core_cv,
        args.no_threat_cv,
        args.outdir / "fig_results_ablation_cv_combined.png",
    )
    render_feature_group_validation(
        args.strengthening,
        args.outdir / "fig_feature_group_validation.png",
    )
    render_temporal_uncertainty_validation(
        args.rolling_origin,
        args.bootstrap_ci,
        args.outdir / "fig_temporal_uncertainty_validation.png",
        args.outdir / "fig_temporal_uncertainty_validation.pdf",
    )
    input_paths = {
        "processing_manifest": args.processing_manifest,
        "core_holdout": args.core_holdout,
        "no_threat_holdout": args.no_threat_holdout,
        "core_cv": args.core_cv,
        "no_threat_cv": args.no_threat_cv,
        "strengthening": args.strengthening,
        "rolling_origin": args.rolling_origin,
        "bootstrap_ci": args.bootstrap_ci,
    }
    output_names = [
        "fig_class_distribution.png",
        "fig_results_ablation_cv_combined.png",
        "fig_feature_group_validation.png",
        "fig_temporal_uncertainty_validation.png",
        "fig_temporal_uncertainty_validation.pdf",
    ]
    metadata = {
        "renderer": Path(__file__).name,
        "renderer_version": RENDERER_VERSION,
        "inputs": {
            name: {"basename": path.name, "sha256": sha256(path, normalize_text_eol=True)}
            for name, path in input_paths.items()
        },
        "input_hash_semantics": "SHA-256 after normalizing CRLF/CR text inputs to LF, matching Git blob bytes.",
        "render_style": {
            "axis_label_font_family": "Lato",
            "axis_label_font_weight": "regular",
            "axis_label_font_file_basename": AXIS_LABEL_FONT_PATH.name,
            "axis_label_font_file_sha256": sha256(AXIS_LABEL_FONT_PATH),
            "vector_font_type": 42,
            "temporal_figure_support_font_family": "DejaVu Sans",
            "temporal_figure_support_font_size_pt": 7.2,
            "fig2_fig3_axis_label_size_pt": FIG2_FIG3_AXIS_LABEL_SIZE_PT,
            "fig2_fig3_axis_tick_size_pt": FIG2_FIG3_AXIS_TICK_SIZE_PT,
            "fig4_axis_sizes_unchanged": True,
        },
        "checks": {
            "class_count_total": sum(
                json.loads(args.processing_manifest.read_text(encoding="utf-8"))["class_counts"].values()
            ),
            "holdout_models": len(load_holdout(args.core_holdout, args.no_threat_holdout)),
            "cv_models": len(load_cv(args.core_cv, args.no_threat_cv)),
            "feature_group_rows": 12,
            "rolling_origin_rows": len(load_rolling_origin(args.rolling_origin)),
            "bootstrap_interval_rows": len(load_bootstrap_intervals(args.bootstrap_ci)),
        },
        "outputs": {
            name: {"sha256": sha256(args.outdir / name)} for name in output_names
        },
    }
    (args.outdir / "manuscript_figure_render_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote manuscript figures to {args.outdir}")


if __name__ == "__main__":
    main()
