"""Render the manuscript methodology workflow from aggregate cohort metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


INK = "#243342"
PRIMARY = "#00539C"
TEAL = "#008080"
PHASE_FILL = "#F5F8FB"
BOX_FILL = "#FFFFFF"
PHASE_EDGE = "#9AA9B5"
RENDERER_VERSION = "1.0.0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--processing-manifest",
        type=Path,
        default=Path("data/processed/threat_dataset_processing_manifest.json"),
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("results/manuscript_figures"),
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def add_phase(ax: plt.Axes, x: float, y: float, width: float, height: float, label: str) -> None:
    ax.add_patch(
        Rectangle(
            (x, y),
            width,
            height,
            facecolor=PHASE_FILL,
            edgecolor=PHASE_EDGE,
            linewidth=1.0,
            linestyle=(0, (4, 3)),
            zorder=0,
        )
    )
    ax.text(
        x + 0.12,
        y + height - 0.13,
        label,
        ha="left",
        va="top",
        fontsize=8.2,
        fontweight="bold",
        color=INK,
        zorder=3,
    )


def add_box(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    body: str,
) -> tuple[float, float, float, float]:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.025,rounding_size=0.05",
            facecolor=BOX_FILL,
            edgecolor=INK,
            linewidth=1.15,
            zorder=2,
        )
    )
    ax.text(
        x + width / 2,
        y + height * 0.70,
        title,
        ha="center",
        va="center",
        fontsize=8.0,
        fontweight="bold",
        color=INK,
        zorder=3,
    )
    ax.text(
        x + width / 2,
        y + height * 0.34,
        body,
        ha="center",
        va="center",
        fontsize=7.2,
        color=INK,
        linespacing=1.25,
        zorder=3,
    )
    return x, y, width, height


def side(box: tuple[float, float, float, float], direction: str) -> tuple[float, float]:
    x, y, width, height = box
    points = {
        "left": (x, y + height / 2),
        "right": (x + width, y + height / 2),
        "top": (x + width / 2, y + height),
        "bottom": (x + width / 2, y),
    }
    return points[direction]


def add_arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    dashed: bool = False,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=1.25,
            linestyle="--" if dashed else "-",
            color=TEAL if dashed else PRIMARY,
            shrinkA=3,
            shrinkB=3,
            zorder=4,
        )
    )


def render(manifest_path: Path, output: Path) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows_in = int(manifest["rows_in"])
    rows_out = int(manifest["rows_out"])
    excluded_alert = int(manifest["excluded_by_normalized_action"]["alert"])
    if rows_in - excluded_alert != rows_out:
        raise ValueError("Cohort manifest is internally inconsistent.")

    fig, ax = plt.subplots(figsize=(10.0, 6.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    add_phase(ax, 0.12, 4.13, 9.76, 1.68, "PHASE I: CONTROLLED DATA AND LEAKAGE DESIGN")
    add_phase(ax, 0.12, 2.18, 9.76, 1.68, "PHASE II: RECONSTRUCTION AND STRESS AUDIT")
    add_phase(ax, 0.12, 0.23, 9.76, 1.68, "PHASE III: MODEL INSPECTION AND INTERPRETATION BOUNDARY")

    width = 2.78
    height = 1.08
    row1_y = 4.40
    row2_y = 2.45
    row3_y = 0.50

    b1 = add_box(
        ax,
        0.40,
        row1_y,
        width,
        height,
        "1. Controlled threat-log export",
        f"{rows_in:,} raw records\nOne firewall and policy state",
    )
    b2 = add_box(
        ax,
        3.61,
        row1_y,
        width,
        height,
        "2. Deterministic cohort construction",
        f"Exclude {excluded_alert:,} alert records\nMap {rows_out:,} records to five actions",
    )
    b3 = add_box(
        ax,
        6.82,
        row1_y,
        width,
        height,
        "3. Leakage-graded feature settings",
        "Core excludes Rule\nDescriptor removal and with-policy bound",
    )

    b4 = add_box(
        ax,
        6.82,
        row2_y,
        width,
        height,
        "4. Policy-action regularity audit",
        "Model-independent context entropy\nNamed and missing-rule scopes separated",
    )
    b5 = add_box(
        ax,
        3.61,
        row2_y,
        width,
        height,
        "5. Core reconstruction evidence",
        "Stratified 80/20 holdout and 5-fold CV\nSix tree-based measurement probes",
    )
    b6 = add_box(
        ax,
        0.40,
        row2_y,
        width,
        height,
        "6. Sensitivity and transfer stress",
        "Descriptor and duplicate-group tests\nTemporal and context stress tests\nConditional intervals and referral audit",
    )

    b7 = add_box(
        ax,
        0.40,
        row3_y,
        3.55,
        height,
        "7. Selected-model inspection",
        "TreeSHAP on class-wise raw margins\nCategorical-aware LIME local surrogates",
    )
    b8 = add_box(
        ax,
        4.37,
        row3_y,
        5.23,
        height,
        "8. Interpretation and release boundary",
        "Retrospective policy audit; non-causal and non-autonomous\nPublic code/aggregates; event-level data under controlled access",
    )

    add_arrow(ax, side(b1, "right"), side(b2, "left"))
    add_arrow(ax, side(b2, "right"), side(b3, "left"))
    add_arrow(ax, side(b3, "bottom"), side(b4, "top"))
    add_arrow(ax, side(b4, "left"), side(b5, "right"))
    add_arrow(ax, side(b5, "left"), side(b6, "right"))
    add_arrow(ax, side(b6, "bottom"), side(b7, "top"))
    add_arrow(ax, side(b7, "right"), side(b8, "left"), dashed=True)

    fig.savefig(output, dpi=600, bbox_inches="tight", pad_inches=0.06, facecolor="white")
    plt.close(fig)
    return {
        "rows_in": rows_in,
        "excluded_alert": excluded_alert,
        "rows_out": rows_out,
    }


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    output = args.outdir / "fig_methodology_workflow.png"
    checks = render(args.processing_manifest, output)
    metadata = {
        "renderer": Path(__file__).name,
        "renderer_version": RENDERER_VERSION,
        "input": {
            "basename": args.processing_manifest.name,
            "sha256": sha256(args.processing_manifest),
        },
        "cohort_checks": checks,
        "output": {"basename": output.name, "sha256": sha256(output)},
    }
    (args.outdir / "methodology_workflow_render_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
