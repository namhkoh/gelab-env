"""
Label extracted real-world icons using GPT-5-mini vision.

Creates a montage grid of icons and asks GPT to label each one.
Saves labels to icon_labels.json for use by sim2real_tree.py.

Usage:
    export OPENAI_API_KEY="sk-..."
    python data_engine/sim2real_label_icons.py \
        --icon_dir data_engine/real_icons \
        --batch_size 25 --seed 42
"""

import argparse
import base64
import json
import os
import re
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image, ImageDraw, ImageFont


def load_icons(icon_dir: str) -> List[Tuple[str, Image.Image]]:
    """Load all icons from directory. Returns [(filename, image), ...]."""
    import glob
    pattern = os.path.join(icon_dir, "*/PNG/*.png")
    paths = sorted(glob.glob(pattern))
    if not paths:
        pattern = os.path.join(icon_dir, "**/*.png")
        paths = sorted(glob.glob(pattern, recursive=True))
    icons = []
    for p in paths:
        try:
            img = Image.open(p).convert("RGB")
            icons.append((os.path.basename(p), img))
        except Exception:
            continue
    return icons


def make_montage(icons: List[Tuple[str, Image.Image]],
                 cell_size: int = 80, cols: int = 5) -> Tuple[Image.Image, List[str]]:
    """Create a labeled montage grid of icons.

    Each cell shows the icon with a number label.
    Returns (montage_image, list_of_filenames).
    """
    n = len(icons)
    rows = (n + cols - 1) // cols
    padding = 4
    label_h = 16
    full_cell = cell_size + label_h + padding * 2

    w = cols * full_cell + padding
    h = rows * full_cell + padding
    montage = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(montage)

    try:
        font = ImageFont.truetype("font/helvetica.ttf", 11)
    except Exception:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
        except Exception:
            font = ImageFont.load_default()

    filenames = []
    for i, (fname, img) in enumerate(icons):
        r, c = divmod(i, cols)
        x = padding + c * full_cell
        y = padding + r * full_cell

        # Draw number label
        label = str(i + 1)
        draw.text((x + 2, y + 2), label, fill=(200, 0, 0), font=font)

        # Resize and paste icon
        icon_resized = img.resize((cell_size - padding * 2, cell_size - padding * 2),
                                  Image.LANCZOS)
        montage.paste(icon_resized, (x + padding, y + label_h))
        filenames.append(fname)

    return montage, filenames


def image_to_base64(img: Image.Image) -> str:
    """Convert PIL image to base64 data URL."""
    buf = BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def label_batch(client, model_name: str,
                montage: Image.Image, filenames: List[str],
                max_retries: int = 2) -> Dict[str, str]:
    """Send montage to GPT vision and get labels for each icon."""
    n = len(filenames)
    prompt = (
        f"This image shows a grid of {n} cropped UI elements from mobile apps, numbered 1 to {n}.\n"
        f"For EACH one, provide a short label (1-3 words) describing what it is.\n"
        f"Many are app logos (e.g. Facebook, Chrome, eBay), UI icons (search, settings, back arrow),\n"
        f"or buttons. Some may be dark or small — try your best to identify them.\n"
        f"Format: one line per item, exactly like:\n"
        f"1: search\n"
        f"2: settings gear\n"
        f"3: facebook\n"
        f"...\n"
        f"Only label as 'unknown' if truly unrecognizable (solid color, noise, etc).\n"
        f"Dark icons with visible shapes should still be labeled (e.g. 'dark menu icon').\n"
        f"Reply with ONLY the numbered labels, nothing else."
    )

    img_url = image_to_base64(montage)

    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": img_url}},
                    ],
                }],
                max_completion_tokens=4096,
            )
            text = response.choices[0].message.content or ""
            labels = parse_labels(text, n)
            if labels:
                result = {}
                for i, fname in enumerate(filenames):
                    result[fname] = labels.get(i + 1, "unknown")
                return result
            if attempt < max_retries:
                print(" [retry]", end="", flush=True)
        except Exception as e:
            if attempt < max_retries:
                print(f" [error: {e}, retrying]", end="", flush=True)
            else:
                print(f" [failed: {e}]", end="", flush=True)

    # Fallback: all unknown
    return {fname: "unknown" for fname in filenames}


def parse_labels(text: str, expected: int) -> Dict[int, str]:
    """Parse numbered label lines from GPT response."""
    labels = {}
    for line in text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        m = re.match(r"(\d+)[:\.\)\-]\s*(.+)", line)
        if m:
            idx = int(m.group(1))
            label = m.group(2).strip().strip('"\'').lower()
            # Clean up label
            label = re.sub(r'[^a-z0-9_ ]', '', label).strip()
            if label:
                labels[idx] = label
    return labels if len(labels) >= max(1, expected * 0.2) else {}


def run(args):
    from openai import OpenAI

    # Load icons
    icons = load_icons(args.icon_dir)
    print(f"Loaded {len(icons)} icons from {args.icon_dir}")

    # Load existing labels (for resume)
    labels_path = os.path.join(args.icon_dir, "icon_labels.json")
    if os.path.exists(labels_path) and not args.overwrite:
        with open(labels_path) as f:
            existing = json.load(f)
        print(f"Loaded {len(existing)} existing labels (use --overwrite to redo)")
    else:
        existing = {}

    # Filter to unlabeled icons
    todo = [(fname, img) for fname, img in icons if fname not in existing]
    print(f"Icons to label: {len(todo)} (already done: {len(existing)})")

    if not todo:
        print("All icons already labeled.")
        return

    # Init client
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    client = OpenAI(api_key=key)
    print(f"Model: {args.model_name}")

    # Process in batches
    bs = args.batch_size
    total_batches = (len(todo) + bs - 1) // bs
    all_labels = dict(existing)

    for batch_idx in range(total_batches):
        start = batch_idx * bs
        end = min(start + bs, len(todo))
        batch = todo[start:end]

        print(f"  Batch {batch_idx + 1}/{total_batches} "
              f"(icons {start + 1}-{end}/{len(todo)})",
              end="", flush=True)

        montage, filenames = make_montage(batch)
        batch_labels = label_batch(client, args.model_name, montage, filenames)
        all_labels.update(batch_labels)

        # Show a few labels
        samples = list(batch_labels.items())[:3]
        sample_str = ", ".join(f"{f}: {l}" for f, l in samples)
        print(f" -> {len(batch_labels)} labels ({sample_str}...)")

        # Save after each batch (resume-safe)
        with open(labels_path, "w") as f:
            json.dump(all_labels, f, indent=2)

    # Summary
    labeled = sum(1 for v in all_labels.values() if v != "unknown")
    print(f"\nDone. {len(all_labels)} total labels ({labeled} named, "
          f"{len(all_labels) - labeled} unknown)")
    print(f"Saved to: {labels_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Label extracted icons using GPT-5-mini vision")
    parser.add_argument("--icon_dir", type=str, default="data_engine/real_icons",
                        help="Directory with icons (expects */PNG/*.png or flat)")
    parser.add_argument("--model_name", type=str, default="gpt-5-mini-2025-08-07")
    parser.add_argument("--batch_size", type=int, default=25,
                        help="Icons per montage grid (default 25 = 5x5)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing labels")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
