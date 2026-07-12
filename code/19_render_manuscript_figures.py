"""Render manuscript-facing descriptive and benchmark figures from saved evidence.

The script consumes only the aggregate processing manifest and canonical
benchmark CSV files. It does not require, read, or export event-level records.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
EXPECTED_CLASSES = {"Allow", "Deny", "Drop", "Reset-Both", "Reset-Server"}
RENDERER_VERSION = "1.0.2"


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

    preferred = ["Drop", "Deny", "Reset-Both", "Allow", "Reset-Server"]
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
    ax.set_yticklabels(labels, fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlabel("Number of records (log scale)", fontweight="bold")
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
    ax1.set_yticklabels(holdout["model"].astype(str), fontweight="bold")
    ax1.invert_yaxis()
    ax1.set_xlabel("Holdout macro-F1", fontweight="bold")
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
    ax2.set_yticklabels(cv["model"].astype(str), fontweight="bold")
    ax2.invert_yaxis()
    ax2.set_xlabel("Cross-validation macro-F1", fontweight="bold")
    means = cv[["core_mean", "no_threat_mean"]].to_numpy()
    stds = cv[["core_std", "no_threat_std"]].to_numpy()
    ax2.set_xlim(*score_limits(np.concatenate([(means - stds).ravel(), (means + stds).ravel()]), pad=0.003))
    ax2.grid(axis="x", color=GRAY_GRID, linestyle="--", linewidth=0.8)
    ax2.set_axisbelow(True)
    ax2.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2)
    ax2.text(0.5, -0.25, "(b)", transform=ax2.transAxes, ha="center", va="top", fontweight="bold")

    fig.tight_layout(pad=0.3)
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
        ax.set_xlabel("Score", fontweight="bold")
        limits = score_limits(part[["macro_f1", "balanced_accuracy"]].to_numpy(), pad=0.025)
        ax.set_xlim(*limits)
        ax.grid(axis="x", color=GRAY_GRID, linestyle="--", linewidth=0.8)
        ax.set_axisbelow(True)
    axes[0].invert_yaxis()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.01), ncol=2)
    axes[0].text(0.5, -0.24, "(a)", transform=axes[0].transAxes, ha="center", va="top", fontweight="bold")
    axes[1].text(0.5, -0.24, "(b)", transform=axes[1].transAxes, ha="center", va="top", fontweight="bold")
    fig.tight_layout(pad=0.35, rect=(0, 0, 1, 0.92))
    fig.savefig(output)
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
    input_paths = {
        "processing_manifest": args.processing_manifest,
        "core_holdout": args.core_holdout,
        "no_threat_holdout": args.no_threat_holdout,
        "core_cv": args.core_cv,
        "no_threat_cv": args.no_threat_cv,
        "strengthening": args.strengthening,
    }
    output_names = [
        "fig_class_distribution.png",
        "fig_results_ablation_cv_combined.png",
        "fig_feature_group_validation.png",
    ]
    metadata = {
        "renderer": Path(__file__).name,
        "renderer_version": RENDERER_VERSION,
        "inputs": {
            name: {"basename": path.name, "sha256": sha256(path, normalize_text_eol=True)}
            for name, path in input_paths.items()
        },
        "input_hash_semantics": "SHA-256 after normalizing CRLF/CR text inputs to LF, matching Git blob bytes.",
        "checks": {
            "class_count_total": sum(
                json.loads(args.processing_manifest.read_text(encoding="utf-8"))["class_counts"].values()
            ),
            "holdout_models": len(load_holdout(args.core_holdout, args.no_threat_holdout)),
            "cv_models": len(load_cv(args.core_cv, args.no_threat_cv)),
            "feature_group_rows": 12,
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
