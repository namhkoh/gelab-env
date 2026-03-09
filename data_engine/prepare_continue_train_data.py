"""
Prepare 24k real-world GUI training data for continue-train (Section 6.2).

Downloads AITW, AMEX, and Mind2Web from HuggingFace, converts to unified
ms-swift SFT format with the real-world system prompt action space:
  click, TYPE("text"), SCROLL(N), WAIT(N), complete

Coordinate system: (0,0) top-left to (1000,1000) bottom-right.

Usage:
    # Default: AITW (16k) + Mind2Web (8k) = 24k total
    python data_engine/prepare_continue_train_data.py

    # Custom distribution:
    python data_engine/prepare_continue_train_data.py \
        --sources aitw mind2web --total_samples 24000

    # Include AMEX (requires ~87GB manual download first):
    python data_engine/prepare_continue_train_data.py \
        --sources aitw amex mind2web --total_samples 24000
"""

import argparse
import json
import os
import random
from pathlib import Path

from datasets import load_dataset
from PIL import Image
from tqdm import tqdm


# ---------------------------------------------------------------------------
# AITW conversion
# ---------------------------------------------------------------------------

AITW_ACTION_MAP = {
    3: "type",
    4: "dual_point",  # click or scroll depending on touch==lift
    10: "complete",
}


def _aitw_to_action_str(action_type, touch_yx, lift_yx, type_text, img_h, img_w):
    """Convert AITW action fields to the real-world output format."""
    if action_type == 10:  # STATUS_TASK_COMPLETE
        return "complete", "this is the target page."
    if action_type == 3:  # TYPE
        text = type_text[0] if isinstance(type_text, list) and type_text else str(type_text)
        return f'TYPE("{text}")', f'type "{text}" into the input field.'
    if action_type == 4:  # DUAL_POINT
        ty, tx = touch_yx
        ly, lx = lift_yx
        # Distinguish click vs scroll: if touch ~= lift, it's a click
        if abs(ty - ly) < 0.02 and abs(tx - lx) < 0.02:
            x = int(tx * 1000)
            y = int(ty * 1000)
            x = max(0, min(1000, x))
            y = max(0, min(1000, y))
            action = f"click(start_box='<|box_start|>({x},{y})<|box_end|>')"
            explain = f"click the element at ({x},{y})."
            return action, explain
        else:
            # Scroll: compute direction and magnitude
            dy = ly - ty
            dx = lx - tx
            if abs(dy) > abs(dx):
                dist = int(abs(dy) * 10)
                direction = "down" if dy > 0 else "up"
            else:
                dist = int(abs(dx) * 10)
                direction = "right" if dx > 0 else "left"
            dist = max(1, min(10, dist))
            return f"SCROLL({dist})", f"scroll {direction}."
    return None, None


def process_aitw(max_samples, image_dir, cache_dir):
    """Load and process AITW dataset from HuggingFace.

    cjfcsjt/AITW_Single requires a config name: 'unseen_subject' or 'unseen_verb'.
    We load one config (unseen_subject has 65k samples, plenty for 16k).

    Optimization: filter by action type FIRST using HF Dataset.filter() (fast,
    no image decoding), then shuffle and iterate only the filtered subset.
    """
    print(f"[AITW] Loading dataset (requesting {max_samples} samples)...")
    try:
        ds = load_dataset(
            "cjfcsjt/AITW_Single",
            "unseen_subject",
            split="train",
            cache_dir=cache_dir,
            trust_remote_code=True,
        )
        print(f"[AITW] Loaded 'unseen_subject': {len(ds)} samples")
    except Exception as e:
        print(f"[AITW] Could not load cjfcsjt/AITW_Single: {e}")
        return []

    # Step 1: Filter to supported action types (fast, no image decoding)
    valid_actions = set(AITW_ACTION_MAP.keys())
    print(f"[AITW] Filtering to action types {valid_actions}...")
    ds_filtered = ds.filter(
        lambda row: row["results_action_type"] in valid_actions,
        num_proc=4,
    )
    print(f"[AITW] After filtering: {len(ds_filtered)} valid samples")

    # Step 2: Shuffle and take only what we need
    ds_filtered = ds_filtered.shuffle(seed=42)
    # Take more than needed to account for skips (bad images, etc.)
    take_n = min(len(ds_filtered), max_samples * 2)

    samples = []
    img_subdir = os.path.join(image_dir, "aitw")
    os.makedirs(img_subdir, exist_ok=True)

    for i in tqdm(range(take_n), desc="[AITW] Converting"):
        if len(samples) >= max_samples:
            break
        row = ds_filtered[i]

        action_type = row["results_action_type"]
        touch_yx = row.get("results_yx_touch", [-1, -1])
        lift_yx = row.get("results_yx_lift", [-1, -1])
        type_text = row.get("results_type_action", "")
        goal = row.get("goal_info", "")

        # Get image (cjfcsjt/AITW_Single uses 'image_encoded' as Image feature)
        img = row.get("image_encoded", row.get("image"))
        if not isinstance(img, Image.Image):
            continue

        img_h, img_w = img.height, img.width
        action_str, explain = _aitw_to_action_str(
            action_type, touch_yx, lift_yx, type_text, img_h, img_w
        )
        if action_str is None:
            continue

        # Save image
        img_path = os.path.join(img_subdir, f"aitw_{len(samples):06d}.png")
        img.save(img_path)

        user_content = f"<image>{goal}"
        assistant_content = f"Explain: {explain}\tAction: {action_str}"

        samples.append({
            "messages": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content},
            ],
            "images": [os.path.abspath(img_path)],
            "source": "aitw",
        })

    print(f"[AITW] Converted {len(samples)} samples.")
    return samples


# ---------------------------------------------------------------------------
# AMEX conversion
# ---------------------------------------------------------------------------

AMEX_ACTION_MAP = {
    "TAP": "click",
    "SCROLL": "scroll",
    "TYPE": "type",
    "TASK_COMPLETE": "complete",
    "PRESS_BACK": None,
    "PRESS_HOME": None,
    "PRESS_ENTER": None,
    "TASK_IMPOSSIBLE": None,
}


def _amex_to_action_str(action, touch_coord, lift_coord, type_text, device_dim):
    """Convert AMEX action fields to the real-world output format."""
    if action == "TASK_COMPLETE":
        return "complete", "this is the target page."
    if action == "TYPE":
        text = type_text if type_text else ""
        return f'TYPE("{text}")', f'type "{text}" into the input field.'
    if action == "TAP":
        w, h = device_dim
        x = int((touch_coord[0] / w) * 1000)
        y = int((touch_coord[1] / h) * 1000)
        x = max(0, min(1000, x))
        y = max(0, min(1000, y))
        action_str = f"click(start_box='<|box_start|>({x},{y})<|box_end|>')"
        explain = f"click the element at ({x},{y})."
        return action_str, explain
    if action == "SCROLL":
        w, h = device_dim
        ty, tx = touch_coord[1] / h, touch_coord[0] / w
        ly, lx = lift_coord[1] / h, lift_coord[0] / w
        dy = ly - ty
        dx = lx - tx
        if abs(dy) > abs(dx):
            dist = int(abs(dy) * 10)
            direction = "down" if dy > 0 else "up"
        else:
            dist = int(abs(dx) * 10)
            direction = "right" if dx > 0 else "left"
        dist = max(1, min(10, dist))
        return f"SCROLL({dist})", f"scroll {direction}."
    return None, None


def process_amex(max_samples, image_dir, cache_dir):
    """Load and process AMEX dataset from HuggingFace.

    WARNING: Yuxiang007/AMEX is a file-based repo (~87GB of zipped screenshots),
    NOT a standard tabular HF dataset. load_dataset() will likely fail.
    To use AMEX, manually download and extract from:
      https://huggingface.co/datasets/Yuxiang007/AMEX
    Then provide processed data via a local JSON file instead.
    """
    print(f"[AMEX] Loading dataset (requesting {max_samples} samples)...")
    print("[AMEX] WARNING: AMEX is ~87GB of zipped files, not a standard HF dataset.")
    print("[AMEX] This will likely fail. Use --sources aitw mind2web to skip AMEX.")
    try:
        ds = load_dataset(
            "Yuxiang007/AMEX",
            split="train",
            cache_dir=cache_dir,
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"[AMEX] Could not load Yuxiang007/AMEX: {e}")
        print("[AMEX] To use AMEX data, download manually from:")
        print("[AMEX]   https://huggingface.co/datasets/Yuxiang007/AMEX")
        print("[AMEX] Skipping AMEX.")
        return []

    print(f"[AMEX] Loaded {len(ds)} raw samples. Processing...")
    samples = []
    indices = list(range(len(ds)))
    random.shuffle(indices)

    img_subdir = os.path.join(image_dir, "amex")
    os.makedirs(img_subdir, exist_ok=True)

    for i in tqdm(indices, desc="[AMEX] Converting", total=len(indices)):
        if len(samples) >= max_samples:
            break
        row = ds[i]

        action = row.get("action", "")
        if AMEX_ACTION_MAP.get(action) is None:
            continue

        touch_coord = row.get("touch_coord", [0, 0])
        lift_coord = row.get("lift_coord", [0, 0])
        type_text = row.get("type_text", "")
        device_dim = row.get("device_dim", [1080, 1920])
        instruction = row.get("instruction", row.get("goal", ""))

        img = row.get("image")
        if img is None:
            continue
        if not isinstance(img, Image.Image):
            continue

        action_str, explain = _amex_to_action_str(
            action, touch_coord, lift_coord, type_text, device_dim
        )
        if action_str is None:
            continue

        img_path = os.path.join(img_subdir, f"amex_{len(samples):06d}.png")
        img.save(img_path)

        user_content = f"<image>{instruction}"
        assistant_content = f"Explain: {explain}\tAction: {action_str}"

        samples.append({
            "messages": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content},
            ],
            "images": [os.path.abspath(img_path)],
            "source": "amex",
        })

    print(f"[AMEX] Converted {len(samples)} samples.")
    return samples


# ---------------------------------------------------------------------------
# Mind2Web conversion
# ---------------------------------------------------------------------------

def _mind2web_get_bbox(pos_candidates):
    """Extract pixel bbox from Mind2Web pos_candidates."""
    for cand in pos_candidates:
        attrs = cand.get("attributes", "{}")
        if isinstance(attrs, str):
            try:
                attrs = json.loads(attrs)
            except (json.JSONDecodeError, TypeError):
                attrs = {}
        bbox = attrs.get("bounding_box_rect") or attrs.get("bbox")
        if bbox:
            if isinstance(bbox, str):
                parts = [float(x) for x in bbox.split(",")]
                if len(parts) >= 4:
                    return parts[:4]
            elif isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                return list(bbox[:4])
    return None


def process_mind2web(max_samples, image_dir, cache_dir):
    """Load and process Multimodal-Mind2Web from HuggingFace."""
    print(f"[Mind2Web] Loading dataset (requesting {max_samples} samples)...")
    try:
        ds = load_dataset(
            "osunlp/Multimodal-Mind2Web",
            split="train",
            cache_dir=cache_dir,
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"[Mind2Web] Could not load osunlp/Multimodal-Mind2Web: {e}")
        print("[Mind2Web] Skipping. Check HuggingFace access or download manually.")
        return []

    print(f"[Mind2Web] Loaded {len(ds)} raw samples.")

    # Filter to supported ops first (fast, no image decoding)
    valid_ops = {"CLICK", "TYPE", "SELECT"}
    print(f"[Mind2Web] Filtering to ops {valid_ops}...")
    ds_filtered = ds.filter(
        lambda row: row.get("operation", {}).get("op", "") in valid_ops,
        num_proc=4,
    )
    print(f"[Mind2Web] After filtering: {len(ds_filtered)} valid samples")
    ds_filtered = ds_filtered.shuffle(seed=42)

    samples = []
    img_subdir = os.path.join(image_dir, "mind2web")
    os.makedirs(img_subdir, exist_ok=True)

    for i in tqdm(range(len(ds_filtered)), desc="[Mind2Web] Converting"):
        if len(samples) >= max_samples:
            break
        row = ds_filtered[i]

        operation = row.get("operation", {})
        op = operation.get("op", "")
        value = operation.get("value", "")
        task = row.get("confirmed_task", "")
        action_repr = row.get("target_action_reprs", "")

        img = row.get("screenshot")
        if img is None or not isinstance(img, Image.Image):
            continue

        img_w, img_h = img.size
        pos_candidates = row.get("pos_candidates", [])
        bbox = _mind2web_get_bbox(pos_candidates)

        if op == "CLICK":
            if bbox is None:
                continue
            # bbox is [x, y, width, height] in pixels
            cx = bbox[0] + bbox[2] / 2 if len(bbox) == 4 and bbox[2] > bbox[0] else (bbox[0] + bbox[2]) / 2
            cy = bbox[1] + bbox[3] / 2 if len(bbox) == 4 and bbox[3] > bbox[1] else (bbox[1] + bbox[3]) / 2
            # Heuristic: if values look like [x, y, w, h] with w,h < x,y
            if bbox[2] < bbox[0] or bbox[3] < bbox[1]:
                cx = (bbox[0] + bbox[2]) / 2
                cy = (bbox[1] + bbox[3]) / 2
            x = int((cx / img_w) * 1000)
            y = int((cy / img_h) * 1000)
            x = max(0, min(1000, x))
            y = max(0, min(1000, y))
            action_str = f"click(start_box='<|box_start|>({x},{y})<|box_end|>')"
            explain = action_repr if action_repr else f"click the element at ({x},{y})."
        elif op == "TYPE":
            action_str = f'TYPE("{value}")'
            explain = action_repr if action_repr else f'type "{value}" into the field.'
        elif op == "SELECT":
            # Map SELECT to TYPE for the expanded action space
            action_str = f'TYPE("{value}")'
            explain = action_repr if action_repr else f'select "{value}" from the dropdown.'
        else:
            continue

        img_path = os.path.join(img_subdir, f"m2w_{len(samples):06d}.png")
        img.save(img_path)

        user_content = f"<image>{task}"
        assistant_content = f"Explain: {explain}\tAction: {action_str}"

        samples.append({
            "messages": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content},
            ],
            "images": [os.path.abspath(img_path)],
            "source": "mind2web",
        })

    print(f"[Mind2Web] Converted {len(samples)} samples.")
    return samples


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

SOURCE_PROCESSORS = {
    "aitw": process_aitw,
    "amex": process_amex,
    "mind2web": process_mind2web,
}


def main():
    parser = argparse.ArgumentParser(description="Prepare 24k continue-train data")
    parser.add_argument("--output", default="datas/continue_train_24k.json")
    parser.add_argument("--image_dir", default="datas/real_world_images")
    parser.add_argument("--total_samples", type=int, default=24000)
    parser.add_argument(
        "--sources",
        nargs="+",
        default=["aitw", "mind2web"],  # AMEX excluded: 87GB manual download
        choices=list(SOURCE_PROCESSORS.keys()),
        help="Data sources. Default: aitw + mind2web (16k + 8k). "
             "AMEX excluded by default (requires 87GB manual zip download).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache_dir", default="/ext_hdd2/nhkoh/.cache/huggingface/datasets")
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(args.image_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # Allocate samples proportionally across sources
    n_sources = len(args.sources)
    per_source = args.total_samples // n_sources
    remainder = args.total_samples % n_sources

    all_samples = []
    for idx, source in enumerate(args.sources):
        n = per_source + (1 if idx < remainder else 0)
        processor = SOURCE_PROCESSORS[source]
        samples = processor(n, args.image_dir, args.cache_dir)
        all_samples.extend(samples)

    # Shuffle and trim to exact total
    random.shuffle(all_samples)
    all_samples = all_samples[: args.total_samples]

    # Add indices
    for i, s in enumerate(all_samples):
        s["idx"] = i

    # Write output
    with open(args.output, "w") as f:
        json.dump(all_samples, f, indent=2, ensure_ascii=False)

    # Print statistics
    source_counts = {}
    for s in all_samples:
        src = s.get("source", "unknown")
        source_counts[src] = source_counts.get(src, 0) + 1

    print(f"\nTotal samples: {len(all_samples)}")
    print(f"Output: {args.output}")
    print("Source distribution:")
    for src, count in sorted(source_counts.items()):
        print(f"  {src}: {count}")


if __name__ == "__main__":
    main()
