"""
Augment a GE-Lab environment with deterministic visual transforms.

Creates a new environment directory with augmented page images and
identical JSON structure files. Only color/intensity transforms are
applied, so bounding boxes remain valid without modification.

Usage:
    # Apply a named preset
    python data_engine/augment_env.py \
        --env_dir data_engine/sim2real_envs/tree_full \
        --output_dir data_engine/sim2real_envs/tree_full_bright \
        --preset bright

    # Apply custom parameters
    python data_engine/augment_env.py \
        --env_dir data_engine/sim2real_envs/tree_full \
        --output_dir data_engine/sim2real_envs/tree_full_v1 \
        --brightness 1.2 --contrast 0.9

    # Generate all presets at once
    python data_engine/augment_env.py \
        --env_dir data_engine/sim2real_envs/tree_full \
        --output_dir data_engine/sim2real_envs/augmented \
        --all_presets

After augmentation, generate training data with:
    python data_engine/generate_sft_data.py \
        --env_dir <augmented_output_dir> \
        --output_dir <augmented_output_dir>
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance

PRESETS = {
    "bright":        {"brightness": 1.25, "contrast": 1.1},
    "dim":           {"brightness": 0.75, "contrast": 0.9},
    "warm":          {"hue_shift": 12, "saturation": 1.2},
    "cool":          {"hue_shift": -12, "saturation": 0.85},
    "high_contrast": {"contrast": 1.4},
    "low_contrast":  {"contrast": 0.65},
    "desaturated":   {"saturation": 0.5},
    "vivid":         {"saturation": 1.5, "contrast": 1.1},
    "noisy":         {"noise_sigma": 8},
}


def apply_transforms(img, brightness=1.0, contrast=1.0, saturation=1.0,
                     hue_shift=0, noise_sigma=0, noise_seed=0):
    """Apply color/intensity transforms to a PIL RGB image."""
    if brightness != 1.0:
        img = ImageEnhance.Brightness(img).enhance(brightness)
    if contrast != 1.0:
        img = ImageEnhance.Contrast(img).enhance(contrast)
    if saturation != 1.0:
        img = ImageEnhance.Color(img).enhance(saturation)
    if hue_shift != 0:
        arr = np.array(img.convert("HSV"), dtype=np.int16)
        arr[:, :, 0] = (arr[:, :, 0] + int(hue_shift * 255 / 360)) % 256
        img = Image.fromarray(arr.astype(np.uint8), "HSV").convert("RGB")
    if noise_sigma > 0:
        arr = np.array(img, dtype=np.float32)
        rng = np.random.default_rng(noise_seed)
        arr = np.clip(arr + rng.normal(0, noise_sigma, arr.shape), 0, 255)
        img = Image.fromarray(arr.astype(np.uint8))
    return img


def augment_environment(env_dir, output_dir, params):
    """Create an augmented copy of a GE-Lab environment.

    Copies JSON structure files unchanged. Applies visual transforms
    to each page image.
    """
    env_dir = Path(env_dir)
    output_dir = Path(output_dir)
    pages_in = env_dir / "pages"
    pages_out = output_dir / "pages"

    if not pages_in.exists():
        raise FileNotFoundError(f"No pages/ directory in {env_dir}")

    pages_out.mkdir(parents=True, exist_ok=True)

    for name in ["ui_structure.json", "ui_structure_layer.json"]:
        src = env_dir / name
        if src.exists():
            shutil.copy2(src, output_dir / name)

    page_files = sorted(pages_in.glob("*.png"))
    for i, page_path in enumerate(page_files):
        img = Image.open(page_path).convert("RGB")
        augmented = apply_transforms(img, noise_seed=i, **params)
        augmented.save(pages_out / page_path.name)
        if (i + 1) % 50 == 0 or i + 1 == len(page_files):
            print(f"  [{i + 1}/{len(page_files)}] pages")

    print(f"Output: {output_dir}/ ({len(page_files)} pages)")


def main():
    parser = argparse.ArgumentParser(
        description="Augment a GE-Lab environment with visual transforms")
    parser.add_argument("--env_dir", required=True,
                        help="Source environment directory")
    parser.add_argument("--output_dir", required=True,
                        help="Output directory for augmented environment")
    parser.add_argument("--preset", choices=list(PRESETS.keys()),
                        help="Named augmentation preset")
    parser.add_argument("--all_presets", action="store_true",
                        help="Generate all presets (appends preset name to output_dir)")
    parser.add_argument("--brightness", type=float, default=1.0)
    parser.add_argument("--contrast", type=float, default=1.0)
    parser.add_argument("--saturation", type=float, default=1.0)
    parser.add_argument("--hue_shift", type=float, default=0)
    parser.add_argument("--noise_sigma", type=float, default=0)
    args = parser.parse_args()

    if args.all_presets:
        for name, params in PRESETS.items():
            out = f"{args.output_dir}_{name}"
            print(f"\n=== {name} -> {out} ===")
            augment_environment(args.env_dir, out, params)
    elif args.preset:
        params = PRESETS[args.preset]
        print(f"Preset: {args.preset} {params}")
        augment_environment(args.env_dir, args.output_dir, params)
    else:
        params = {
            "brightness": args.brightness,
            "contrast": args.contrast,
            "saturation": args.saturation,
            "hue_shift": args.hue_shift,
            "noise_sigma": args.noise_sigma,
        }
        active = {k: v for k, v in params.items()
                  if v != (0 if k in ("hue_shift", "noise_sigma") else 1.0)}
        if not active:
            parser.error("No transforms specified. Use --preset or set transform parameters.")
        print(f"Transforms: {active}")
        augment_environment(args.env_dir, args.output_dir, params)


if __name__ == "__main__":
    main()
