"""Render a constant-hue manuscript beeswarm from the verified SHAP PNG.

The SHAP coordinates are not recomputed. Saturated red/blue point pixels are
mapped to one blue hue while retaining local intensity, and the obsolete
encoded-value color bar is blanked. This avoids implying ordinal semantics for
nominal category codes without changing attribution positions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


VERIFIED_INPUT_SHA256 = (
    "6FD0ED8F9AE6D4D63AFEE664EE5B935B5989E8BD369975E097D6E60B3B21E5C5"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--legend-mask-x", type=int, default=3665)
    parser.add_argument("--manifest", type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> None:
    args = parse_args()
    input_hash = sha256(args.input)
    if input_hash != VERIFIED_INPUT_SHA256:
        raise RuntimeError(
            f"Unexpected input hash {input_hash}; expected {VERIFIED_INPUT_SHA256}"
        )

    opened = Image.open(args.input)
    input_dpi = opened.info.get("dpi")
    image = opened.convert("RGB")
    pixels = np.asarray(image).copy()
    if not 0 < args.legend_mask_x < pixels.shape[1]:
        raise ValueError("--legend-mask-x must lie inside the image width")

    working = pixels[:, : args.legend_mask_x, :].astype(np.float32)
    maximum = working.max(axis=2)
    minimum = working.min(axis=2)
    chroma = maximum - minimum
    saturated = (chroma >= 18.0) & (maximum >= 80.0)

    neutral = working.mean(axis=2)
    strength = np.clip(chroma / 105.0, 0.28, 0.82)
    base_blue = np.array([31.0, 119.0, 180.0], dtype=np.float32)
    recolored = (
        neutral[..., None] * (1.0 - strength[..., None])
        + base_blue[None, None, :] * strength[..., None]
    )
    working[saturated] = recolored[saturated]
    pixels[:, : args.legend_mask_x, :] = np.clip(
        working,
        0,
        255,
    ).astype(np.uint8)
    pixels[:, args.legend_mask_x :, :] = 255

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs = {"optimize": True}
    if input_dpi:
        save_kwargs["dpi"] = input_dpi
    Image.fromarray(pixels, mode="RGB").save(args.output, **save_kwargs)

    manifest_path = args.manifest or args.output.with_suffix(".render.json")
    manifest = {
        "operation": "constant_hue_shap_beeswarm_render",
        "renderer": Path(__file__).name,
        "input": args.input.as_posix(),
        "input_sha256": input_hash,
        "output": args.output.as_posix(),
        "output_sha256": sha256(args.output),
        "width": int(pixels.shape[1]),
        "height": int(pixels.shape[0]),
        "legend_mask_x": args.legend_mask_x,
        "dpi": list(input_dpi) if input_dpi else None,
        "saturated_pixels_recolored": int(saturated.sum()),
        "scientific_geometry_changed": False,
        "interpretation": (
            "SHAP coordinates retained; nominal-code high/low color semantics removed"
        ),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
