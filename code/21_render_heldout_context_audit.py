"""Render the manuscript figure for the held-out named-rule context audit.

This renderer consumes the stored aggregate CSV only; it does not fit a model
or alter any scientific result.
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


RENDERER_VERSION = "2.0.0"
PRIMARY = "#00539C"
TEAL = "#008080"
CORAL = "#EEA47F"
INK = "#333333"
LINE = "#888888"
GRID = "#E2E8F0"
ROW_BG = "#F8FAFC"
AXIS_LABEL_SIZE_PT = 8
AXIS_TICK_SIZE_PT = 7
PANEL_TITLE_SIZE_PT = 8.2


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def resolve_font(
    env_name: str,
    accepted_names: set[str],
    label: str,
    *,
    weight: str = "normal",
) -> tuple[font_manager.FontProperties, Path]:
    explicit = os.environ.get(env_name)
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise RuntimeError(f"{env_name} is not a font file: {path}")
        font_manager.fontManager.addfont(path)
        return font_manager.FontProperties(fname=str(path), weight=weight), path
    for font_path in font_manager.findSystemFonts():
        path = Path(font_path)
        if path.name.lower() in accepted_names:
            font_manager.fontManager.addfont(path)
            return font_manager.FontProperties(fname=str(path), weight=weight), path
    raise RuntimeError(f"{label} is required to render the manuscript figure.")


LATO, LATO_PATH = resolve_font("LEAF_LATO_REGULAR", {"lato-regular.ttf"}, "Lato Regular")
LATO_BOLD, LATO_BOLD_PATH = resolve_font(
    "LEAF_LATO_BOLD",
    {"lato-bold.ttf"},
    "Lato Bold",
    weight="bold",
)
INTER, INTER_PATH = resolve_font(
    "LEAF_INTER_REGULAR",
    {"inter-regular.ttf", "inter-variablefont_opsz,wght.ttf"},
    "Inter Regular",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--aggregate",
        type=Path,
        default=Path("results/policy_context_robustness/policy_context_heldout_aggregate.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/policy_context_robustness/fig_heldout_policy_context_audit.png"),
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("results/policy_context_robustness/fig_heldout_policy_context_audit.metadata.json"),
    )
    return parser.parse_args()


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [INTER.get_name(), "DejaVu Sans"],
            "font.size": 7,
            "axes.labelsize": AXIS_LABEL_SIZE_PT,
            "axes.titlesize": PANEL_TITLE_SIZE_PT,
            "axes.titleweight": "bold",
            "xtick.labelsize": AXIS_TICK_SIZE_PT,
            "ytick.labelsize": AXIS_TICK_SIZE_PT,
            "legend.fontsize": 7.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": LINE,
            "axes.linewidth": 1.0,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )


def apply_inter(ax: plt.Axes) -> None:
    """Apply Inter without allowing FontProperties to reset the locked size."""
    for text in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
        text.set_fontproperties(INTER)
        text.set_fontweight("normal")
        text.set_fontsize(AXIS_TICK_SIZE_PT)


def main() -> None:
    args = parse_args()
    set_style()
    source = pd.read_csv(args.aggregate)
    frame = source[source["scope"] == "named_rule_contexts_only"].copy()
    order = ["core", "no_threat_descriptors", "minimal_context"]
    frame["feature_set"] = pd.Categorical(frame["feature_set"], categories=order, ordered=True)
    frame = frame.sort_values("feature_set").reset_index(drop=True)
    if frame["feature_set"].astype(str).tolist() != order:
        raise RuntimeError("Held-out aggregate CSV does not contain the expected three feature settings.")

    y = np.arange(len(frame))
    fig = plt.figure(figsize=(3.60, 2.42))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.85, 1.0], wspace=0.30)
    ax_score = fig.add_subplot(grid[0, 0])
    ax_error = fig.add_subplot(grid[0, 1], sharey=ax_score)

    accuracy = frame["support_weighted_accuracy"].to_numpy()
    macro_f1 = frame["support_weighted_observed_macro_f1"].to_numpy()
    errors = frame["errors"].to_numpy(dtype=int)

    for row in y[1::2]:
        ax_score.axhspan(row - 0.5, row + 0.5, color=ROW_BG, zorder=0, lw=0)
        ax_error.axhspan(row - 0.5, row + 0.5, color=ROW_BG, zorder=0, lw=0)
    for yi, acc, mf1 in zip(y, accuracy, macro_f1):
        ax_score.plot([mf1, acc], [yi, yi], color=LINE, lw=1.1, zorder=1)
    ax_score.scatter(
        accuracy,
        y,
        color=PRIMARY,
        marker="o",
        s=34,
        label="Accuracy",
        edgecolor="white",
        linewidth=0.6,
        zorder=3,
    )
    ax_score.scatter(
        macro_f1,
        y,
        color=TEAL,
        marker="s",
        s=31,
        label="Observed-class macro-F1",
        edgecolor="white",
        linewidth=0.6,
        zorder=3,
    )
    ax_score.set_xlim(0.30, 0.88)
    ax_score.set_xticks([0.4, 0.6, 0.8])
    ax_score.set_yticks(y)
    ax_score.set_yticklabels(["Core", "No threat\ndescriptors", "Minimal context"], fontproperties=INTER)
    ax_score.invert_yaxis()
    ax_score.set_title("Scores", fontproperties=LATO_BOLD, fontsize=PANEL_TITLE_SIZE_PT, pad=5)
    ax_score.set_xlabel(
        "Support-weighted score",
        fontproperties=LATO,
        fontsize=AXIS_LABEL_SIZE_PT,
    )
    ax_score.grid(axis="x", color=GRID, linewidth=0.8, linestyle="--")
    ax_score.set_axisbelow(True)
    handles, legend_labels = ax_score.get_legend_handles_labels()
    legend = fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.59, 0.985),
        frameon=False,
        ncol=2,
        columnspacing=0.9,
        handlelength=1.0,
        handletextpad=0.45,
        borderaxespad=0.0,
        prop=INTER,
    )
    for text in legend.get_texts():
        text.set_fontweight("normal")
        text.set_fontsize(7.0)

    for yi, value in zip(y, errors):
        ax_error.plot([0, value], [yi, yi], color=LINE, lw=1.1, zorder=1)
    ax_error.scatter(
        errors,
        y,
        color=CORAL,
        marker="s",
        s=31,
        edgecolor="white",
        linewidth=0.6,
        zorder=3,
    )
    ax_error.set_xlim(0, 6600)
    ax_error.set_xticks([0, 3000, 6000])
    ax_error.set_xticklabels(["0", "3k", "6k"], fontproperties=INTER)
    ax_error.set_title("Errors", fontproperties=LATO_BOLD, fontsize=PANEL_TITLE_SIZE_PT, pad=5)
    ax_error.set_xlabel("Records", fontproperties=LATO, fontsize=AXIS_LABEL_SIZE_PT)
    ax_error.grid(axis="x", color=GRID, linewidth=0.8, linestyle="--")
    ax_error.set_axisbelow(True)
    ax_error.tick_params(axis="y", left=False, labelleft=False)
    for yi, value in zip(y, errors):
        label_x = value + 170 if value < 5000 else value - 170
        ax_error.text(
            label_x,
            yi,
            f"{value:,}",
            va="center",
            ha="left" if value < 5000 else "right",
            fontproperties=INTER,
            fontsize=6.4,
            color=INK,
            bbox={
                "facecolor": ROW_BG if yi % 2 else "white",
                "edgecolor": "none",
                "pad": 0.35,
            },
            zorder=4,
        )

    ax_score.text(0.5, -0.31, "(a)", transform=ax_score.transAxes, ha="center", va="top", fontproperties=INTER, fontsize=7.2)
    ax_error.text(0.5, -0.31, "(b)", transform=ax_error.transAxes, ha="center", va="top", fontproperties=INTER, fontsize=7.2)
    apply_inter(ax_score)
    apply_inter(ax_error)
    fig.subplots_adjust(left=0.34, right=0.98, top=0.76, bottom=0.25)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, facecolor="white")
    plt.close(fig)

    payload = {
        "renderer_version": RENDERER_VERSION,
        "source": str(args.aggregate.as_posix()),
        "source_sha256": file_sha256(args.aggregate),
        "output": str(args.output.as_posix()),
        "output_sha256": file_sha256(args.output),
        "scope": "named_rule_contexts_only",
        "test_rows": int(frame["test_rows"].iloc[0]),
        "settings": [
            {
                "feature_set": row.feature_set,
                "support_weighted_accuracy": float(row.support_weighted_accuracy),
                "support_weighted_observed_macro_f1": float(row.support_weighted_observed_macro_f1),
                "errors": int(row.errors),
            }
            for row in frame.itertuples(index=False)
        ],
        "typography": {
            "axis_label_font": "Lato Regular",
            "axis_label_font_sha256": file_sha256(LATO_PATH),
            "panel_title_font": "Lato Bold",
            "panel_title_font_sha256": file_sha256(LATO_BOLD_PATH),
            "internal_text_font": "Inter Regular",
            "internal_text_font_sha256": file_sha256(INTER_PATH),
            "bold_internal_text": False,
            "axis_label_size_pt": AXIS_LABEL_SIZE_PT,
            "axis_tick_size_pt": AXIS_TICK_SIZE_PT,
            "panel_title_size_pt": PANEL_TITLE_SIZE_PT,
        },
        "palette": {
            "accuracy": PRIMARY,
            "observed_class_macro_f1": TEAL,
            "errors": CORAL,
            "connector": LINE,
            "row_background": ROW_BG,
        },
    }
    args.metadata.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
