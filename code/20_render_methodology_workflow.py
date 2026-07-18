"""Render the manuscript methodology workflow from aggregate cohort metadata."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, Rectangle
from PIL import Image, ImageChops


PRIMARY = "#596F7A"
BOUNDARY = "#765C86"
PHASE_TEXT = "#7A263A"
BOX_TITLE_TEXT = "#17365D"
BOX_BODY_TEXT = "#5B3A78"
PHASE_PALETTES = {
    "I": {
        "phase_fill": "#FAFAFA",
        "box_fill": "#EDF5FB",
        "phase_edge": "#A8C4D6",
        "box_edge": "#7FA8C2",
    },
    "II": {
        "phase_fill": "#F7F7F7",
        "box_fill": "#FFF8E5",
        "phase_edge": "#D7C58E",
        "box_edge": "#BFA35B",
    },
    "III": {
        "phase_fill": "#F4F4F4",
        "box_fill": "#EEF8F1",
        "phase_edge": "#ACCBB5",
        "box_edge": "#82A98E",
    },
}
RENDERER_VERSION = "1.8.1"
PNG_DPI = 600
CROP_PADDING_PIXELS = 12


def ensure_inter_static_fonts() -> None:
    """Instantiate static Inter Regular/SemiBold faces from an installed
    variable font so Matplotlib can select real weights."""
    variable_source = None
    for font_path in font_manager.findSystemFonts():
        name = Path(font_path).name.lower()
        if name.startswith("inter-variablefont") and "italic" not in name:
            variable_source = Path(font_path)
            break
    if variable_source is None:
        return
    try:
        from fontTools.ttLib import TTFont
        from fontTools.varLib.instancer import instantiateVariableFont
    except ImportError:
        return
    import tempfile

    cache_dir = Path(tempfile.gettempdir()) / "leaf_inter_static"
    cache_dir.mkdir(parents=True, exist_ok=True)
    for weight, subfamily in ((400, "Regular"), (600, "SemiBold")):
        target = cache_dir / f"Inter-{subfamily}.ttf"
        if not target.is_file():
            font = TTFont(str(variable_source))
            pins = {axis.axisTag: axis.defaultValue for axis in font["fvar"].axes}
            pins["wght"] = weight
            instantiateVariableFont(font, pins, inplace=True, updateFontNames=True)
            name_table = font["name"]
            for name_id, value in (
                (1, "Inter"),
                (2, subfamily),
                (4, f"Inter {subfamily}"),
                (6, f"Inter-{subfamily}"),
                (16, "Inter"),
                (17, subfamily),
            ):
                name_table.setName(value, name_id, 3, 1, 0x409)
            font.save(str(target))
        font_manager.fontManager.addfont(str(target))


def register_additional_fonts() -> None:
    """Register publication fonts that Matplotlib may not cache automatically."""
    ensure_inter_static_fonts()
    kpsewhich = shutil.which("kpsewhich")
    if kpsewhich:
        for filename in ("SourceSans3-Regular.otf", "SourceSans3-Semibold.otf"):
            completed = subprocess.run(
                [kpsewhich, filename],
                check=False,
                capture_output=True,
                text=True,
            )
            font_path = Path(completed.stdout.strip())
            if completed.returncode == 0 and font_path.is_file():
                font_manager.fontManager.addfont(font_path)

    for font_path in font_manager.findSystemFonts():
        filename = Path(font_path).name.lower()
        if filename.startswith(("lato-", "liberationsans-")):
            font_manager.fontManager.addfont(font_path)
        elif (
            filename.startswith("inter-")
            and "italic" not in filename
            and "variablefont" not in filename
        ):
            font_manager.fontManager.addfont(font_path)


def resolve_fonts() -> tuple[str, Path, str, Path]:
    """Resolve the author-selected title and summary font faces."""
    register_additional_fonts()
    title_family = "Lato"
    summary_family = "Liberation Sans"
    try:
        title_path = Path(
            font_manager.findfont(
                font_manager.FontProperties(family=title_family, weight="bold"),
                fallback_to_default=False,
            )
        )
        summary_path = Path(
            font_manager.findfont(
                font_manager.FontProperties(family=summary_family, weight="normal"),
                fallback_to_default=False,
            )
        )
    except ValueError as exc:
        raise RuntimeError(
            "Figure 1 requires Lato Bold and Liberation Sans Regular."
        ) from exc
    return title_family, title_path, summary_family, summary_path


TITLE_FONT_FAMILY, TITLE_FONT_PATH, SUMMARY_FONT_FAMILY, SUMMARY_FONT_PATH = resolve_fonts()
matplotlib.rcParams.update(
    {
        "font.family": SUMMARY_FONT_FAMILY,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


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


def add_phase(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    label: str,
    phase_key: str,
    *,
    align: str = "left",
) -> None:
    palette = PHASE_PALETTES[phase_key]
    ax.add_patch(
        Rectangle(
            (x, y),
            width,
            height,
            facecolor=palette["phase_fill"],
            edgecolor=palette["phase_edge"],
            linewidth=0.8,
            linestyle=(0, (4, 3)),
            zorder=0,
        )
    )
    inset = 0.12
    if align == "center":
        text_x, ha = x + width / 2, "center"
    elif align == "right":
        text_x, ha = x + width - inset, "right"
    else:
        text_x, ha = x + inset, "left"
    ax.text(
        text_x,
        y + height - 0.13,
        label,
        ha=ha,
        va="top",
        fontsize=7.8,
        fontfamily=TITLE_FONT_FAMILY,
        fontweight="bold",
        color=PHASE_TEXT,
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
    phase_key: str,
    *,
    title_fontsize: float = 8.6,
    body_fontsize: float = 7.8,
) -> tuple[float, float, float, float]:
    palette = PHASE_PALETTES[phase_key]
    ax.add_patch(
        Rectangle(
            (x, y),
            width,
            height,
            facecolor=palette["box_fill"],
            edgecolor=palette["box_edge"],
            linewidth=0.9,
            zorder=2,
        )
    )
    ax.text(
        x + width / 2,
        y + height * 0.68,
        title,
        ha="center",
        va="center",
        fontsize=title_fontsize,
        fontfamily=TITLE_FONT_FAMILY,
        fontweight="bold",
        color=BOX_TITLE_TEXT,
        zorder=3,
    )
    ax.text(
        x + width / 2,
        y + height * 0.29,
        body,
        ha="center",
        va="center",
        fontsize=body_fontsize,
        fontfamily=SUMMARY_FONT_FAMILY,
        color=BOX_BODY_TEXT,
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
            mutation_scale=9.5,
            linewidth=1.0,
            linestyle="--" if dashed else "-",
            color=BOUNDARY if dashed else PRIMARY,
            shrinkA=3,
            shrinkB=3,
            zorder=4,
        )
    )


def save_content_cropped_png(fig: plt.Figure, output: Path) -> dict:
    """Rasterize, then physically crop white margins around the drawn content."""
    buffer = io.BytesIO()
    fig.savefig(
        buffer,
        format="png",
        dpi=PNG_DPI,
        bbox_inches="tight",
        pad_inches=0,
        facecolor="white",
        edgecolor="none",
    )
    buffer.seek(0)
    with Image.open(buffer) as rendered:
        image = rendered.convert("RGB")

    background = Image.new("RGB", image.size, "white")
    content_bbox = ImageChops.difference(image, background).getbbox()
    if content_bbox is None:
        raise RuntimeError("Rendered methodology workflow contains no visible content.")

    left, top, right, bottom = content_bbox
    crop_bbox = (
        max(0, left - CROP_PADDING_PIXELS),
        max(0, top - CROP_PADDING_PIXELS),
        min(image.width, right + CROP_PADDING_PIXELS),
        min(image.height, bottom + CROP_PADDING_PIXELS),
    )
    cropped = image.crop(crop_bbox)
    cropped.save(output, format="PNG", dpi=(PNG_DPI, PNG_DPI), optimize=True)
    return {
        "uncropped_pixels": {"width": image.width, "height": image.height},
        "content_bbox_pixels": {
            "left": left,
            "top": top,
            "right": right,
            "bottom": bottom,
        },
        "crop_bbox_pixels": {
            "left": crop_bbox[0],
            "top": crop_bbox[1],
            "right": crop_bbox[2],
            "bottom": crop_bbox[3],
        },
        "final_pixels": {"width": cropped.width, "height": cropped.height},
        "padding_pixels": CROP_PADDING_PIXELS,
        "dpi": PNG_DPI,
    }


def render(manifest_path: Path, output: Path) -> tuple[dict, dict]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows_in = int(manifest["rows_in"])
    rows_out = int(manifest["rows_out"])
    excluded_alert = int(manifest["excluded_by_normalized_action"]["alert"])
    if rows_in - excluded_alert != rows_out:
        raise ValueError("Cohort manifest is internally inconsistent.")

    fig, ax = plt.subplots(figsize=(10.0, 4.1))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.1)
    ax.axis("off")

    phase_height = 1.10
    phase1_y = 2.85
    phase2_y = 1.50
    phase3_y = 0.15
    add_phase(
        ax,
        0.12,
        phase1_y,
        9.76,
        phase_height,
        "PHASE I: CONTROLLED DATA AND LEAKAGE DESIGN",
        "I",
        align="left",
    )
    add_phase(
        ax,
        0.12,
        phase2_y,
        9.76,
        phase_height,
        "PHASE II: RECONSTRUCTION AND STRESS AUDIT",
        "II",
        align="center",
    )
    add_phase(
        ax,
        0.12,
        phase3_y,
        9.76,
        phase_height,
        "PHASE III: MODEL INSPECTION AND INTERPRETATION BOUNDARY",
        "III",
        align="right",
    )

    width = 2.78
    height = 0.64
    row1_y = phase1_y + 0.16
    row2_y = phase2_y + 0.16
    row3_y = phase3_y + 0.16

    b1 = add_box(
        ax,
        0.40,
        row1_y,
        width,
        height,
        "1. Controlled threat-log export",
        "Raw records, one policy state",
        "I",
    )
    b2 = add_box(
        ax,
        3.61,
        row1_y,
        width,
        height,
        "2. Deterministic cohort construction",
        "Alert filtering and action mapping",
        "I",
    )
    b3 = add_box(
        ax,
        6.82,
        row1_y,
        width,
        height,
        "3. Leakage-graded feature settings",
        "Core, no-descriptor, and with-policy",
        "I",
    )

    b4 = add_box(
        ax,
        6.82,
        row2_y,
        width,
        height,
        "4. Policy-action regularity audit",
        "Rule-context entropy",
        "II",
    )
    b5 = add_box(
        ax,
        3.61,
        row2_y,
        width,
        height,
        "5. Core reconstruction evidence",
        "Holdout and cross-validation",
        "II",
    )
    b6 = add_box(
        ax,
        0.40,
        row2_y,
        width,
        height,
        "6. Sensitivity and transfer stress",
        "Ablation, duplicate, temporal, and context",
        "II",
    )

    b7 = add_box(
        ax,
        0.40,
        row3_y,
        width,
        height,
        "7. Selected-model inspection",
        "TreeSHAP and LIME",
        "III",
    )
    b8 = add_box(
        ax,
        3.61,
        row3_y,
        5.99,
        height,
        "8. Interpretation and release boundary",
        "Offline, non-causal, controlled access",
        "III",
    )

    add_arrow(ax, side(b1, "right"), side(b2, "left"))
    add_arrow(ax, side(b2, "right"), side(b3, "left"))
    add_arrow(ax, side(b3, "bottom"), side(b4, "top"))
    add_arrow(ax, side(b4, "left"), side(b5, "right"))
    add_arrow(ax, side(b5, "left"), side(b6, "right"))
    add_arrow(ax, side(b6, "bottom"), side(b7, "top"))
    add_arrow(ax, side(b7, "right"), side(b8, "left"), dashed=True)

    crop_metadata = save_content_cropped_png(fig, output)
    plt.close(fig)
    return (
        {
            "rows_in": rows_in,
            "excluded_alert": excluded_alert,
            "rows_out": rows_out,
        },
        crop_metadata,
    )


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    output = args.outdir / "fig_methodology_workflow.png"
    checks, crop_metadata = render(args.processing_manifest, output)
    metadata = {
        "renderer": Path(__file__).name,
        "renderer_version": RENDERER_VERSION,
        "render_style": {
            "title_font_family": TITLE_FONT_FAMILY,
            "title_font_weight": "bold",
            "title_font_file_basename": TITLE_FONT_PATH.name,
            "title_font_file_sha256": sha256(TITLE_FONT_PATH),
            "summary_font_family": SUMMARY_FONT_FAMILY,
            "summary_font_weight": "regular",
            "summary_font_file_basename": SUMMARY_FONT_PATH.name,
            "summary_font_file_sha256": sha256(SUMMARY_FONT_PATH),
            "phase_font_points": 7.8,
            "box_title_font_points": 8.6,
            "box_body_font_points": 7.8,
            "box_text_size_uniform_across_steps": True,
            "palette": {
                "phase_groups": PHASE_PALETTES,
                "phase_label_text": PHASE_TEXT,
                "box_title_text": BOX_TITLE_TEXT,
                "box_body_text": BOX_BODY_TEXT,
                "primary_connector": PRIMARY,
                "boundary_connector": BOUNDARY,
            },
            "box_corner_style": "square",
            "box_height_inches": 0.64,
            "figure_height_inches": 4.1,
            "box_text_structure": "one title and one single-line summary",
            "phase_label_alignment": ["left", "center", "right"],
            "content_aware_png_crop": True,
        },
        "input": {
            "basename": args.processing_manifest.name,
            "sha256": sha256(args.processing_manifest),
        },
        "cohort_checks": checks,
        "crop": crop_metadata,
        "output": {"basename": output.name, "sha256": sha256(output)},
    }
    (args.outdir / "methodology_workflow_render_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
