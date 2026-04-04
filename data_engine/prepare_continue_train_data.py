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
# AITZ conversion
# ---------------------------------------------------------------------------

AITZ_ACTION_MAP = {
    "click": "click",
    "scroll": "scroll",
    "type": "type",
    "navigate_home": None,
    "navigate_back": None,
    "open_app": None,
    "wait": "wait",
}


def _aitz_to_action_str(action_dict, img_w, img_h):
    """Convert AITZ action dict to the real-world output format."""
    name = action_dict.get("action_name", "")
    explain = action_dict.get("action_explanation", "")

    if name == "click":
        touch = action_dict.get("start_point") or action_dict.get("end_point")
        if not touch or len(touch) < 2:
            return None, None
        x = int(float(touch[0]) * 1000)
        y = int(float(touch[1]) * 1000)
        x = max(0, min(1000, x))
        y = max(0, min(1000, y))
        action_str = f"click(start_box='<|box_start|>({x},{y})<|box_end|>')"
        return action_str, explain or f"click the element at ({x},{y})."
    if name == "scroll":
        direction = action_dict.get("direction", "down")
        dist = 5
        scroll_dist = action_dict.get("scroll_distance")
        if scroll_dist and float(scroll_dist) > 0:
            dist = max(1, min(10, int(float(scroll_dist) * 10)))
        return f"SCROLL({dist})", explain or f"scroll {direction}."
    if name == "type":
        text = action_dict.get("keys") or ""
        return f'TYPE("{text}")', explain or f'type "{text}".'
    if name == "wait":
        return "WAIT(3)", explain or "wait for the page to load."
    return None, None


def process_aitz(max_samples, image_dir, cache_dir):
    """Load and process AITZ dataset from HuggingFace (xwm/AITZ)."""
    print(f"[AITZ] Loading dataset (requesting {max_samples} samples)...")
    try:
        ds = load_dataset(
            "xwm/AITZ",
            split="train",
            cache_dir=cache_dir,
            trust_remote_code=True,
        )
        print(f"[AITZ] Loaded: {len(ds)} samples")
    except Exception as e:
        print(f"[AITZ] Could not load xwm/AITZ: {e}")
        return []

    # Filter to supported actions
    valid_actions = {"click", "scroll", "type", "wait"}
    print(f"[AITZ] Filtering to actions {valid_actions}...")
    ds_filtered = ds.filter(
        lambda row: (row.get("action") or {}).get("action_name", "") in valid_actions,
        num_proc=4,
    )
    print(f"[AITZ] After filtering: {len(ds_filtered)} valid samples")
    ds_filtered = ds_filtered.shuffle(seed=42)

    samples = []
    img_subdir = os.path.join(image_dir, "aitz")
    os.makedirs(img_subdir, exist_ok=True)

    for i in tqdm(range(len(ds_filtered)), desc="[AITZ] Converting"):
        if len(samples) >= max_samples:
            break
        row = ds_filtered[i]

        action_dict = row.get("action", {})
        img_w = row.get("image_width", 270)
        img_h = row.get("image_height", 600)

        action_str, explain = _aitz_to_action_str(action_dict, img_w, img_h)
        if action_str is None:
            continue

        instruction = row.get("instruction", "")

        # Decode image from base64
        img_b64 = row.get("image_base64", "")
        if not img_b64:
            continue
        try:
            import base64
            from io import BytesIO
            img = Image.open(BytesIO(base64.b64decode(img_b64))).convert("RGB")
        except Exception:
            continue

        img_path = os.path.join(img_subdir, f"aitz_{len(samples):06d}.png")
        img.save(img_path)

        user_content = f"<image>{instruction}"
        assistant_content = f"Explain: {explain}\tAction: {action_str}"

        samples.append({
            "messages": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content},
            ],
            "images": [os.path.abspath(img_path)],
            "source": "aitz",
        })

    print(f"[AITZ] Converted {len(samples)} samples.")
    return samples


# ---------------------------------------------------------------------------
# AMEX conversion (local files)
# ---------------------------------------------------------------------------

AMEX_ACTION_MAP = {
    "TAP": "click",
    "SCROLL": "scroll",
    "SWIPE": "scroll",
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


AMEX_LOCAL_ANNOTATIONS = "/ext_hdd2/tsyou/AMEX_dataset/AMEX/instruction_anno"
AMEX_LOCAL_SCREENSHOTS = "/ext_hdd2/tsyou/AMEX_dataset/AMEX/screenshot"


def process_amex(max_samples, image_dir, cache_dir):
    """Load and process AMEX from local files."""
    print(f"[AMEX] Loading from local files (requesting {max_samples} samples)...")

    annot_dir = Path(AMEX_LOCAL_ANNOTATIONS)
    screenshot_dir = Path(AMEX_LOCAL_SCREENSHOTS)
    if not annot_dir.exists():
        print(f"[AMEX] Annotations not found at {annot_dir}. Skipping.")
        return []

    annot_files = sorted(annot_dir.glob("*.json"))
    print(f"[AMEX] Found {len(annot_files)} annotation files")

    # Collect all step-level samples across trajectories
    all_steps = []
    for annot_path in annot_files:
        try:
            with open(annot_path) as f:
                traj = json.load(f)
        except Exception:
            continue
        episode_id = traj.get("episode_id", annot_path.stem)
        instruction = traj.get("instruction", "")
        for step in traj.get("steps", []):
            action = step.get("action", "")
            if AMEX_ACTION_MAP.get(action) is None:
                continue
            screenshot_name = step.get("image_path", step.get("screenshot", ""))
            if not screenshot_name:
                continue
            screenshot_path = screenshot_dir / screenshot_name
            if not screenshot_path.exists():
                continue
            device_dim = step.get("device_dim", [1080, 1920])
            all_steps.append({
                "action": action,
                "touch_coord": step.get("touch_coord", [0, 0]),
                "lift_coord": step.get("lift_coord", [0, 0]),
                "type_text": step.get("type_text", ""),
                "device_dim": device_dim,
                "instruction": instruction,
                "screenshot_path": str(screenshot_path),
            })

    print(f"[AMEX] Collected {len(all_steps)} valid steps")
    random.shuffle(all_steps)

    samples = []
    img_subdir = os.path.join(image_dir, "amex")
    os.makedirs(img_subdir, exist_ok=True)

    for step_data in tqdm(all_steps, desc="[AMEX] Converting"):
        if len(samples) >= max_samples:
            break

        action_str, explain = _amex_to_action_str(
            step_data["action"],
            step_data["touch_coord"],
            step_data["lift_coord"],
            step_data["type_text"],
            step_data["device_dim"],
        )
        if action_str is None:
            continue

        try:
            img = Image.open(step_data["screenshot_path"]).convert("RGB")
        except Exception:
            continue

        img_path = os.path.join(img_subdir, f"amex_{len(samples):06d}.png")
        img.save(img_path)

        user_content = f"<image>{step_data['instruction']}"
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
        if isinstance(cand, str):
            try:
                cand = json.loads(cand)
            except (json.JSONDecodeError, TypeError):
                continue
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
    # operation is a JSON string, not a dict
    valid_ops = {"CLICK", "TYPE", "SELECT"}
    print(f"[Mind2Web] Filtering to ops {valid_ops}...")
    def _parse_op(row):
        op_field = row.get("operation", "")
        if isinstance(op_field, str):
            try:
                op_field = json.loads(op_field)
            except (json.JSONDecodeError, TypeError):
                return False
        return op_field.get("op", "") in valid_ops
    ds_filtered = ds.filter(_parse_op, num_proc=4)
    print(f"[Mind2Web] After filtering: {len(ds_filtered)} valid samples")
    ds_filtered = ds_filtered.shuffle(seed=42)

    samples = []
    img_subdir = os.path.join(image_dir, "mind2web")
    os.makedirs(img_subdir, exist_ok=True)

    for i in tqdm(range(len(ds_filtered)), desc="[Mind2Web] Converting"):
        if len(samples) >= max_samples:
            break
        row = ds_filtered[i]

        operation = row.get("operation", "")
        if isinstance(operation, str):
            try:
                operation = json.loads(operation)
            except (json.JSONDecodeError, TypeError):
                continue
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
    "aitz": process_aitz,
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
        default=["aitw", "aitz", "amex", "mind2web"],
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
