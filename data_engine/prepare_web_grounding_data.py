"""Generate WEB-domain grounding + understanding SFT samples from Mind2Web.

Web analogue of prepare_real_grounding_data.py (AMEX/mobile): produces the
same two task formats with the same prompt templates and 0-1000 center
normalization, sourced from osunlp/Multimodal-Mind2Web train split.

Mind2Web screenshots are full-page renders (1280 x up to ~20000 px), while
bounding_box_rect coordinates in pos_candidates are page pixels aligned 1:1
with the screenshot. Unlike process_mind2web in prepare_continue_train_data.py
(which saves the full tall page and normalizes over it, compressing y), this
script crops a viewport-sized window around the target element -- with random
vertical jitter so the element does not always sit at the same height -- and
normalizes the center against the CROP so emitted coordinates are correct for
the saved image.

Example:
    HF_HOME=/workspace/nhkoh/hf_cache python data_engine/prepare_web_grounding_data.py \
        --image_dir datas/real_world_images/mind2web_grounding \
        --output datas/web_grounding_m2w.json \
        --ground_max 10000 --understand_frac 0.3
"""
import argparse
import json
import os
import random
import re

from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

VALID_OPS = {"CLICK", "TYPE", "SELECT"}


def parse_json_maybe(obj, default):
    if isinstance(obj, str):
        try:
            return json.loads(obj)
        except (json.JSONDecodeError, TypeError):
            return default
    return obj if obj is not None else default


def get_bbox_and_attrs(pos_candidates):
    """Extract pixel bbox [x, y, w, h] + attributes from Mind2Web pos_candidates."""
    for cand in pos_candidates:
        cand = parse_json_maybe(cand, None)
        if not isinstance(cand, dict):
            continue
        attrs = parse_json_maybe(cand.get("attributes", "{}"), {})
        rect = attrs.get("bounding_box_rect")
        if not rect:
            continue
        try:
            parts = [float(v) for v in str(rect).split(",")]
        except ValueError:
            continue
        if len(parts) >= 4 and parts[2] > 0 and parts[3] > 0:
            return parts[:4], attrs
    return None, None


def clean_name(raw):
    return re.sub(r"\s+", " ", str(raw)).strip()


def element_name(action_repr, attrs):
    """Human-readable element name: action_reprs text > aria_label > title."""
    text = ""
    if action_repr:
        # "[heading]  CAR -> CLICK" -> "CAR"
        text = str(action_repr).split("->")[0]
        text = re.sub(r"^\s*\[[^\]]*\]", "", text)
        text = clean_name(text)
    if len(text) < 2:
        text = clean_name(attrs.get("aria_label", ""))
    if len(text) < 2:
        text = clean_name(attrs.get("title", ""))
    return text


def crop_viewport(img, bbox, viewport_h, rng):
    """Crop a full-width, viewport_h-tall window containing bbox.

    The element center lands at a jittered vertical position so grounding
    targets are not always mid-frame. Returns (cropped_img, cx, cy) with the
    center in crop-pixel coordinates.
    """
    img_w, img_h = img.size
    x, y, w, h = bbox
    cx = x + w / 2
    cy = y + h / 2
    if img_h <= viewport_h:
        return img, cx, cy
    # Element (vertically) fully inside the crop when possible, else centered.
    pad = 8
    lo = min(h / 2 + pad, viewport_h / 2)
    hi = max(viewport_h - h / 2 - pad, viewport_h / 2)
    offset = rng.uniform(lo, hi)  # element center's y within the crop
    top = int(round(cy - offset))
    top = max(0, min(top, img_h - viewport_h))
    img = img.crop((0, top, img_w, top + viewport_h))
    return img, cx, cy - top


def center_to_1000(cx, cy, canvas_size):
    w, h = canvas_size
    x = max(0, min(1000, int(cx / w * 1000)))
    y = max(0, min(1000, int(cy / h * 1000)))
    return x, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="osunlp/Multimodal-Mind2Web")
    parser.add_argument("--split", default="train")
    parser.add_argument("--image_dir", default="datas/real_world_images/mind2web_grounding")
    parser.add_argument("--output", default="datas/web_grounding_m2w.json")
    parser.add_argument("--ground_max", type=int, default=10000)
    parser.add_argument("--understand_frac", type=float, default=0.3,
                        help="fraction of elements that also emit an understanding sample")
    parser.add_argument("--viewport_h", type=int, default=1080,
                        help="crop height (px) for tall full-page screenshots")
    parser.add_argument("--jpeg_quality", type=int, default=92)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    os.makedirs(args.image_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    ds = load_dataset(args.dataset, split=args.split)
    print(f"[Mind2Web] Loaded {len(ds)} raw rows.")
    # Drop heavy unused columns so per-row access only decodes what we need.
    drop = [c for c in ("raw_html", "cleaned_html", "neg_candidates", "action_reprs")
            if c in ds.column_names]
    ds = ds.remove_columns(drop)

    ground, understand = [], []
    skipped = {"op": 0, "bbox": 0, "name": 0, "image": 0, "off_image": 0}
    for i in tqdm(range(len(ds)), desc="[Mind2Web] Converting"):
        if len(ground) >= args.ground_max:
            break
        row = ds[i]
        op = parse_json_maybe(row.get("operation", ""), {}).get("op", "")
        if op not in VALID_OPS:
            skipped["op"] += 1
            continue
        bbox, attrs = get_bbox_and_attrs(row.get("pos_candidates") or [])
        if bbox is None:
            skipped["bbox"] += 1
            continue
        name = element_name(row.get("target_action_reprs", ""), attrs)
        if len(name) < 2 or len(name) > 80:
            skipped["name"] += 1
            continue

        img = row.get("screenshot")
        if not isinstance(img, Image.Image):
            skipped["image"] += 1
            continue
        img_w, img_h = img.size
        if not (0 <= bbox[0] + bbox[2] / 2 < img_w and 0 <= bbox[1] + bbox[3] / 2 < img_h):
            skipped["off_image"] += 1
            continue

        crop, cx_px, cy_px = crop_viewport(img.convert("RGB"), bbox, args.viewport_h, rng)
        cx, cy = center_to_1000(cx_px, cy_px, crop.size)

        img_path = os.path.join(args.image_dir, f"m2w_{len(ground):06d}.jpg")
        crop.save(img_path, quality=args.jpeg_quality)
        abs_img = os.path.abspath(img_path)

        ground.append({
            "messages": [
                {"role": "user", "content": f"<image>I want to click on {name}. Please locate the target element I should interact with. (with point)"},
                {"role": "assistant", "content": f"Action: click(start_box='<|box_start|>({cx},{cy})<|box_end|>')"},
            ],
            "images": [abs_img],
            "source": "mind2web_grounding",
        })
        if rng.random() < args.understand_frac:
            understand.append({
                "messages": [
                    {"role": "user", "content": f"<image>What is the icon at point ({cx},{cy}) in the image?"},
                    {"role": "assistant", "content": name},
                ],
                "images": [abs_img],
                "source": "mind2web_understanding",
            })

    out = ground + understand
    rng.shuffle(out)
    for i, s in enumerate(out):
        s["idx"] = i
    with open(args.output, "w") as fp:
        json.dump(out, fp, ensure_ascii=False)
    print(f"skipped: {skipped}")
    print(f"grounding: {len(ground)}, understanding: {len(understand)}")
    print(f"total: {len(out)} -> {args.output}")


if __name__ == "__main__":
    main()
