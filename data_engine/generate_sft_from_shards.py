"""
Generate sft_amex.json from extracted HF tar shards (no annotation files needed).

Uses ui_structure.json from each trajectory to generate navigation, grounding,
and captioning samples. Based on generate_amex_sft_data.py.

Usage:
    python /workspace/data_engine/generate_sft_from_shards.py \
        --shards_dir /data/amex-gelab/shards \
        --extract_dir /data/amex_envs \
        --output /data/datas/sft_amex.json
"""

import argparse
import json
import os
import random
import tarfile
from collections import Counter
from typing import Any, Dict, List, Tuple


def _find_closest_layout_element(
    action_coord: List[int],
    layout: Dict[str, Any],
) -> Tuple[str, List[int]]:
    ax, ay = action_coord[0], action_coord[1]
    best_key = ""
    best_bbox = [0, 0, 0, 0]
    best_dist = float("inf")

    for key, value in layout.items():
        if key in ("back", "home"):
            continue
        if isinstance(value, dict):
            bbox = value.get("bbox", [0, 0, 0, 0])
        elif isinstance(value, list) and len(value) == 4:
            bbox = value
        else:
            continue

        x1, y1, x2, y2 = bbox
        if x1 <= ax <= x2 and y1 <= ay <= y2:
            area = (x2 - x1) * (y2 - y1)
            if area < best_dist:
                best_dist = area
                best_key = key
                best_bbox = bbox
            continue

        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        dist = (ax - cx) ** 2 + (ay - cy) ** 2
        if dist < best_dist and best_key == "":
            best_dist = dist
            best_key = key
            best_bbox = bbox

    return best_key or "element", best_bbox


def _infer_swipe_direction(start: List[int], end: List[int]) -> str:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    if abs(dy) > abs(dx):
        return "up" if dy < 0 else "down"
    return "left" if dx < 0 else "right"


def _bbox_center(bbox: List[int]) -> Tuple[int, int]:
    if not bbox or bbox == [0, 0, 0, 0]:
        return (0, 0)
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) // 2, (y1 + y2) // 2)


def _format_user_content(instruction: str, history: str) -> str:
    parts = ["<image>"]
    if instruction:
        parts.append(f"Goal: {instruction}")
    parts.append(f"History: {history}")
    return " ".join(parts)


# -- Action formatters (same as generate_amex_sft_data.py) --

def _format_tap_action(action_name: str, coord: List[int]) -> str:
    cx, cy = int(coord[0]), int(coord[1])
    return (
        f"Explain: tap {action_name}.\t"
        f"Action: tap(start_box='<|box_start|>({cx},{cy})<|box_end|>')"
    )

def _format_swipe_action(start: List[int], end: List[int], direction: str) -> str:
    sx, sy = int(start[0]), int(start[1])
    ex, ey = int(end[0]), int(end[1])
    return (
        f"Explain: swipe {direction}.\t"
        f"Action: swipe(start_box='<|box_start|>({sx},{sy})<|box_end|>', "
        f"end_box='<|box_start|>({ex},{ey})<|box_end|>')"
    )

def _format_type_action(text: str, coord: List[int]) -> str:
    cx, cy = int(coord[0]), int(coord[1])
    return (
        f"Explain: type \"{text}\".\t"
        f"Action: type(start_box='<|box_start|>({cx},{cy})<|box_end|>', text='{text}')"
    )

def _format_press_enter_action() -> str:
    return "Explain: press enter to confirm.\tAction: press_enter()"

def _format_press_back_action() -> str:
    return "Explain: press back.\tAction: press_back()"

def _format_press_home_action() -> str:
    return "Explain: press home.\tAction: press_home()"

def _format_complete_action() -> str:
    return "Explain: task is complete.\tAction: complete"

def _format_impossible_action() -> str:
    return "Explain: task cannot be completed.\tAction: impossible"


# -- Sample generators --

def generate_trajectory_samples(
    ui_structure: dict,
    pages_dir: str,
    source_label: str,
    instruction: str = "",
) -> List[dict]:
    samples = []
    pages = ui_structure.get("pages", {})
    page_ids = sorted(
        pages.keys(),
        key=lambda x: int(x.split("_")[1]) if len(x.split("_")) > 1 and x.split("_")[1].isdigit() else 0,
    )
    if not page_ids:
        return samples

    start_page = page_ids[0]
    end_page = page_ids[-1]
    if not instruction:
        instruction = f"Navigate from {start_page} to {end_page}"

    history_parts: List[str] = []

    for page_id in page_ids:
        page = pages.get(page_id)
        if not page:
            continue

        transitions = page.get("transitions", [])
        if not transitions:
            continue

        t = transitions[0]
        action = str(t.get("action", "")).strip().upper()
        action_coord = t.get("action_coord", [0, 0])

        history = "; ".join(history_parts) if history_parts else "Null"
        user_content = _format_user_content(instruction, history)
        image_path = os.path.join(pages_dir, page.get("image", f"{page_id}.png"))

        if action == "TASK_COMPLETE":
            samples.append({
                "messages": [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": _format_complete_action()},
                ],
                "images": [image_path],
                "source": source_label,
                "action_type": "TASK_COMPLETE",
            })
            continue

        if action == "TASK_IMPOSSIBLE":
            samples.append({
                "messages": [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": _format_impossible_action()},
                ],
                "images": [image_path],
                "source": source_label,
                "action_type": "TASK_IMPOSSIBLE",
            })
            continue

        assistant_content = None
        bbox_norm = [0, 0, 0, 0]
        action_type = None

        if action in ("TAP", "CLICK"):
            elem_key, elem_bbox = _find_closest_layout_element(action_coord, page.get("layout", {}))
            assistant_content = _format_tap_action(elem_key, action_coord)
            bbox_norm = list(elem_bbox) if elem_bbox != [0, 0, 0, 0] else [
                action_coord[0] - 20, action_coord[1] - 20,
                action_coord[0] + 20, action_coord[1] + 20,
            ]
            history_parts.append(f"step{len(history_parts)+1}: tap {elem_key} on {page_id}")
            action_type = "TAP"

        elif action in ("SWIPE", "SCROLL"):
            lift_coord = t.get("lift_coord", action_coord)
            direction = _infer_swipe_direction(action_coord, lift_coord)
            assistant_content = _format_swipe_action(action_coord, lift_coord, direction)
            x_coords = sorted([action_coord[0], lift_coord[0]])
            y_coords = sorted([action_coord[1], lift_coord[1]])
            bbox_norm = [x_coords[0], y_coords[0], x_coords[1], y_coords[1]]
            history_parts.append(f"step{len(history_parts)+1}: swipe {direction} on {page_id}")
            action_type = "SWIPE"

        elif action in ("TYPE", "TEXT"):
            type_text = str(t.get("type_text", "")).strip()
            assistant_content = _format_type_action(type_text, action_coord)
            bbox_norm = [
                action_coord[0] - 40, action_coord[1] - 20,
                action_coord[0] + 40, action_coord[1] + 20,
            ]
            history_parts.append(f"step{len(history_parts)+1}: type \"{type_text}\" on {page_id}")
            action_type = "TYPE"

        elif action == "PRESS_ENTER":
            assistant_content = _format_press_enter_action()
            history_parts.append(f"step{len(history_parts)+1}: press enter on {page_id}")
            action_type = "PRESS_ENTER"

        elif action == "PRESS_BACK":
            assistant_content = _format_press_back_action()
            history_parts.append(f"step{len(history_parts)+1}: press back on {page_id}")
            action_type = "PRESS_BACK"

        elif action == "PRESS_HOME":
            assistant_content = _format_press_home_action()
            history_parts.append(f"step{len(history_parts)+1}: press home on {page_id}")
            action_type = "PRESS_HOME"

        else:
            continue

        samples.append({
            "task": instruction,
            "route": f"From {start_page} to {end_page}",
            "messages": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content},
            ],
            "images": [image_path],
            "bbox_norm": bbox_norm,
            "source": source_label,
            "action_type": action_type,
        })

    return samples


def generate_grounding_samples(
    ui_structure: dict,
    pages_dir: str,
    source_label: str,
    max_samples: int = 20,
) -> List[dict]:
    all_icons = []
    for page_id, page in ui_structure.get("pages", {}).items():
        layout = page.get("layout", {})
        for key, value in layout.items():
            if key in ("back", "home"):
                continue
            bbox = value.get("bbox", [0, 0, 0, 0]) if isinstance(value, dict) else value
            if bbox == [0, 0, 0, 0]:
                continue
            all_icons.append((page_id, key, bbox))

    if not all_icons:
        return []

    sampled = random.choices(all_icons, k=min(max_samples, len(all_icons)))
    samples = []
    for page_id, label, bbox in sampled:
        cx, cy = _bbox_center(bbox)
        image_path = os.path.join(
            pages_dir, ui_structure["pages"][page_id].get("image", f"{page_id}.png")
        )
        samples.append({
            "task": "grounding",
            "messages": [
                {"role": "user", "content": f"<image>Tap on {label} in the image."},
                {"role": "assistant", "content": f"Action: tap(start_box='<|box_start|>({cx},{cy})<|box_end|>')"},
            ],
            "images": [image_path],
            "bbox_norm": list(bbox),
            "source": source_label,
            "action_type": "grounding",
        })
    return samples


def generate_captioning_samples(
    ui_structure: dict,
    pages_dir: str,
    source_label: str,
    max_samples: int = 20,
) -> List[dict]:
    all_icons = []
    for page_id, page in ui_structure.get("pages", {}).items():
        layout = page.get("layout", {})
        for key, value in layout.items():
            if key in ("back", "home"):
                continue
            bbox = value.get("bbox", [0, 0, 0, 0]) if isinstance(value, dict) else value
            if bbox == [0, 0, 0, 0]:
                continue
            all_icons.append((page_id, key, bbox))

    if not all_icons:
        return []

    sampled = random.choices(all_icons, k=min(max_samples, len(all_icons)))
    samples = []
    for page_id, label, bbox in sampled:
        cx, cy = _bbox_center(bbox)
        image_path = os.path.join(
            pages_dir, ui_structure["pages"][page_id].get("image", f"{page_id}.png")
        )
        samples.append({
            "task": "captioning",
            "messages": [
                {"role": "user", "content": f"<image>What is the icon at point ({cx}, {cy}) in the image?"},
                {"role": "assistant", "content": label.replace("_", " ")},
            ],
            "images": [image_path],
            "bbox_norm": list(bbox),
            "source": source_label,
            "action_type": "captioning",
        })
    return samples


# -- Main --

def extract_shards(shards_dir: str, extract_dir: str):
    os.makedirs(extract_dir, exist_ok=True)
    tar_files = sorted(f for f in os.listdir(shards_dir) if f.endswith(".tar"))
    print(f"Found {len(tar_files)} tar shards")
    for i, tar_name in enumerate(tar_files):
        tar_path = os.path.join(shards_dir, tar_name)
        print(f"  [{i+1}/{len(tar_files)}] Extracting {tar_name}...")
        with tarfile.open(tar_path, "r") as tf:
            tf.extractall(path=extract_dir)
    print(f"Extraction complete -> {extract_dir}")


def find_trajectory_dirs(extract_dir: str) -> List[str]:
    traj_dirs = []
    for root, dirs, files in os.walk(extract_dir):
        if "ui_structure.json" in files:
            traj_dirs.append(root)
    return sorted(traj_dirs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards_dir", default="/data/amex-gelab/shards")
    parser.add_argument("--extract_dir", default="/data/amex_envs")
    parser.add_argument("--output", default="/data/datas/sft_amex.json")
    parser.add_argument("--grounding_per_traj", type=int, default=20)
    parser.add_argument("--captioning_per_traj", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_extract", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)

    if not args.skip_extract:
        extract_shards(args.shards_dir, args.extract_dir)
    else:
        print("Skipping extraction (--skip_extract)")

    traj_dirs = find_trajectory_dirs(args.extract_dir)
    print(f"\nFound {len(traj_dirs)} trajectories")

    all_samples = []
    action_counts = Counter()
    errors = 0

    for i, traj_dir in enumerate(traj_dirs):
        traj_id = os.path.basename(traj_dir)
        source_label = f"amex_{traj_id}"

        try:
            with open(os.path.join(traj_dir, "ui_structure.json")) as f:
                ui = json.load(f)

            pages_dir = os.path.join(traj_dir, "pages")
            if not os.path.isdir(pages_dir):
                pages_dir = os.path.join(traj_dir, "extracted_assets")
            if not os.path.isdir(pages_dir):
                pages_dir = traj_dir

            instruction = ""
            manifest_path = os.path.join(traj_dir, "trajectory_assets_manifest.json")
            if os.path.exists(manifest_path):
                with open(manifest_path) as f:
                    manifest = json.load(f)
                instruction = manifest.get("instruction", "")

            nav = generate_trajectory_samples(ui, pages_dir, source_label, instruction)
            grounding = generate_grounding_samples(ui, pages_dir, source_label, args.grounding_per_traj)
            captioning = generate_captioning_samples(ui, pages_dir, source_label, args.captioning_per_traj)

            samples = nav + grounding + captioning
            all_samples.extend(samples)
            for s in samples:
                action_counts[s.get("action_type", "unknown")] += 1

            if (i + 1) % 100 == 0:
                print(f"  [{i+1}/{len(traj_dirs)}] samples: {len(all_samples)}")

        except Exception as e:
            print(f"  ERROR [{traj_id}]: {e}")
            errors += 1

    for i, s in enumerate(all_samples):
        s["idx"] = i

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(all_samples, f, indent=2)

    print(f"\n{'='*60}")
    print(f"SFT DATASET GENERATED")
    print(f"{'='*60}")
    print(f"Trajectories: {len(traj_dirs) - errors} (errors: {errors})")
    print(f"Total samples: {len(all_samples)}")
    print(f"Action types:")
    for atype, count in action_counts.most_common():
        print(f"  {atype:20s}: {count}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
