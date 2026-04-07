"""
Generate SFT training data from pre-composed AMEX trajectories (448x448).

Reads ui_structure.json + page images from --env_dir and produces
sft_amex.json compatible with swift sft training.

Usage:
    python data_engine/generate_amex_sft_data.py \
        --env_dir /home/irteam/data-vol1/amex_sft \
        --output /home/irteam/data-vol1/gelab-env/datas/sft_amex.json

    python data_engine/generate_amex_sft_data.py \
        --env_dir /home/irteam/data-vol1/amex_sft_hf_448 \
        --output /home/irteam/data-vol1/gelab-env/datas/sft_amex.json \
        --max_trajectories 100
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _normalize_coord(x: int, y: int, canvas_w: int, canvas_h: int) -> Tuple[int, int]:
    """Normalize pixel coordinates to 0-1000 range."""
    nx = int(round(x * 1000 / canvas_w)) if canvas_w > 0 else x
    ny = int(round(y * 1000 / canvas_h)) if canvas_h > 0 else y
    return (max(0, min(1000, nx)), max(0, min(1000, ny)))


def _bbox_center(bbox: List[int], canvas_w: int = 0, canvas_h: int = 0) -> Tuple[int, int]:
    if not bbox or bbox == [0, 0, 0, 0]:
        return (0, 0)
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    if canvas_w > 0 and canvas_h > 0:
        return _normalize_coord(cx, cy, canvas_w, canvas_h)
    return (cx, cy)


def _find_closest_layout_element(
    action_coord: List[int],
    layout: Dict[str, Any],
) -> Tuple[str, List[int]]:
    """Find the layout element whose bbox contains or is closest to action_coord."""
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


def _format_tap_action(action_name: str, page_id: str, coord: List[int]) -> str:
    cx, cy = coord[0], coord[1]
    return (
        f"Explain: tap {action_name} on {page_id}.\t"
        f"Action: tap(start_box='<|box_start|>({cx},{cy})<|box_end|>')"
    )


def _format_swipe_action(page_id: str, start: List[int], end: List[int], direction: str) -> str:
    sx, sy = start[0], start[1]
    ex, ey = end[0], end[1]
    return (
        f"Explain: swipe {direction} on {page_id}.\t"
        f"Action: swipe(start_box='<|box_start|>({sx},{sy})<|box_end|>', "
        f"end_box='<|box_start|>({ex},{ey})<|box_end|>')"
    )


def _format_type_action(text: str, page_id: str, coord: List[int]) -> str:
    cx, cy = coord[0], coord[1]
    return (
        f"Explain: type \"{text}\" on {page_id}.\t"
        f"Action: type(start_box='<|box_start|>({cx},{cy})<|box_end|>', text='{text}')"
    )


def _format_press_enter_action(page_id: str) -> str:
    return f"Explain: press enter to confirm on {page_id}.\tAction: press_enter()"


def _format_press_back_action(page_id: str) -> str:
    return f"Explain: press back on {page_id}.\tAction: press_back()"


def _format_press_home_action(page_id: str) -> str:
    return f"Explain: press home on {page_id}.\tAction: press_home()"


def _format_complete_action() -> str:
    return "Explain: task is complete.\tAction: complete"


def _format_impossible_action() -> str:
    return "Explain: task cannot be completed.\tAction: impossible"


def _infer_swipe_direction(start: List[int], end: List[int]) -> str:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    if abs(dy) > abs(dx):
        return "up" if dy < 0 else "down"
    return "left" if dx < 0 else "right"


def _format_user_content(instruction: str, history: str) -> str:
    parts = ["<image>"]
    if instruction:
        parts.append(f"Instruction: {instruction}.")
    parts.append(f"History: {history}")
    return " ".join(parts)


def generate_trajectory_samples(
    ui_structure: dict,
    pages_dir: str,
    source_label: str,
) -> List[dict]:
    """Generate SFT samples from a single pre-composed trajectory."""
    samples: List[dict] = []
    pages = ui_structure.get("pages", {})
    metadata = ui_structure.get("metadata", {})
    instruction = str(metadata.get("instruction", "")).strip()
    canvas_size = metadata.get("canvas_size", [0, 0])
    canvas_w, canvas_h = canvas_size[0], canvas_size[1]

    page_ids = sorted(
        pages.keys(),
        key=lambda x: int(x.split("_")[1]) if x.split("_")[1].isdigit() else 0,
    )
    if not page_ids:
        return samples

    history_parts: List[str] = []

    for page_id in page_ids:
        page = pages.get(page_id)
        if not page:
            continue

        transitions = page.get("transitions", [])
        if not transitions:
            continue

        # Use first transition (the forward navigation action)
        t = transitions[0]
        action = str(t.get("action", "")).strip().upper()
        action_coord = t.get("action_coord", [0, 0])
        target_page = t.get("target_page", page_id)

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

        if action in ("TAP", "CLICK"):
            icon_name = t.get("icon_name")
            if icon_name:
                elem_key = icon_name
            else:
                elem_key, _ = _find_closest_layout_element(action_coord, page.get("layout", {}))
            norm_coord = list(_normalize_coord(action_coord[0], action_coord[1], canvas_w, canvas_h))
            assistant_content = _format_tap_action(elem_key, page_id, norm_coord)
            history_parts.append(f"step{len(history_parts)+1}: tap {elem_key}")
            action_type = "TAP"

        elif action in ("SWIPE", "SCROLL"):
            lift_coord = t.get("lift_coord", action_coord)
            direction = _infer_swipe_direction(action_coord, lift_coord)
            norm_start = list(_normalize_coord(action_coord[0], action_coord[1], canvas_w, canvas_h))
            norm_end = list(_normalize_coord(lift_coord[0], lift_coord[1], canvas_w, canvas_h))
            assistant_content = _format_swipe_action(page_id, norm_start, norm_end, direction)
            history_parts.append(f"step{len(history_parts)+1}: swipe {direction}")
            action_type = "SWIPE"

        elif action in ("TYPE", "TEXT"):
            type_text = str(t.get("type_text", "")).strip()
            if action_coord and action_coord != [0, 0]:
                coord = action_coord
            else:
                # Find an input-like element from layout
                coord = [canvas_w // 2, canvas_h // 2] if canvas_w > 0 else [224, 224]
                for key, value in page.get("layout", {}).items():
                    if key in ("back", "home"):
                        continue
                    bbox = value.get("bbox", [0, 0, 0, 0]) if isinstance(value, dict) else value
                    if bbox != [0, 0, 0, 0]:
                        coord = [(bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2]
                        break
            norm_coord = list(_normalize_coord(coord[0], coord[1], canvas_w, canvas_h))
            assistant_content = _format_type_action(type_text, page_id, norm_coord)
            history_parts.append(f"step{len(history_parts)+1}: type \"{type_text}\"")
            action_type = "TYPE"

        elif action == "PRESS_ENTER":
            assistant_content = _format_press_enter_action(page_id)
            history_parts.append(f"step{len(history_parts)+1}: press enter")
            action_type = "PRESS_ENTER"

        elif action == "PRESS_BACK":
            assistant_content = _format_press_back_action(page_id)
            history_parts.append(f"step{len(history_parts)+1}: press back")
            action_type = "PRESS_BACK"

        elif action == "PRESS_HOME":
            assistant_content = _format_press_home_action(page_id)
            history_parts.append(f"step{len(history_parts)+1}: press home")
            action_type = "PRESS_HOME"

        else:
            continue

        samples.append({
            "messages": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content},
            ],
            "images": [image_path],
            "source": source_label,
            "action_type": action_type,
        })

    return samples


def generate_grounding_samples(
    ui_structure: dict,
    pages_dir: str,
    source_label: str,
    max_per_trajectory: int = 20,
) -> List[dict]:
    """Generate icon grounding samples from layout elements."""
    metadata = ui_structure.get("metadata", {})
    canvas_size = metadata.get("canvas_size", [0, 0])
    canvas_w, canvas_h = canvas_size[0], canvas_size[1]

    all_icons: List[Tuple[str, str, List[int]]] = []
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

    sampled = random.choices(all_icons, k=min(max_per_trajectory, len(all_icons)))
    samples = []
    for page_id, label, bbox in sampled:
        cx, cy = _bbox_center(bbox, canvas_w, canvas_h)
        image_path = os.path.join(pages_dir, ui_structure["pages"][page_id].get("image", f"{page_id}.png"))
        samples.append({
            "messages": [
                {"role": "user", "content": f"<image>Tap on {label} in the image."},
                {"role": "assistant", "content": f"Action: tap(start_box='<|box_start|>({cx},{cy})<|box_end|>')"},
            ],
            "images": [image_path],
            "source": source_label,
            "action_type": "grounding",
        })
    return samples


def generate_captioning_samples(
    ui_structure: dict,
    pages_dir: str,
    source_label: str,
    max_per_trajectory: int = 20,
) -> List[dict]:
    """Generate icon captioning samples from layout elements."""
    metadata = ui_structure.get("metadata", {})
    canvas_size = metadata.get("canvas_size", [0, 0])
    canvas_w, canvas_h = canvas_size[0], canvas_size[1]

    all_icons: List[Tuple[str, str, List[int]]] = []
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

    sampled = random.choices(all_icons, k=min(max_per_trajectory, len(all_icons)))
    samples = []
    for page_id, label, bbox in sampled:
        cx, cy = _bbox_center(bbox, canvas_w, canvas_h)
        image_path = os.path.join(pages_dir, ui_structure["pages"][page_id].get("image", f"{page_id}.png"))
        samples.append({
            "messages": [
                {"role": "user", "content": f"<image>What is the icon at point ({cx}, {cy}) in the image?"},
                {"role": "assistant", "content": label.replace("_", " ")},
            ],
            "images": [image_path],
            "source": source_label,
            "action_type": "captioning",
        })
    return samples


def parse_args():
    parser = argparse.ArgumentParser(description="Generate SFT data from pre-composed AMEX trajectories")
    parser.add_argument("--env_dir", type=str, required=True,
                        help="Directory with pre-composed trajectories (e.g. amex_sft_hf_448)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output JSON path (e.g. datas/sft_amex.json)")
    parser.add_argument("--max_trajectories", type=int, default=None)
    parser.add_argument("--grounding_per_traj", type=int, default=20)
    parser.add_argument("--captioning_per_traj", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    traj_dirs = sorted([
        d for d in os.listdir(args.env_dir)
        if os.path.isdir(os.path.join(args.env_dir, d))
        and os.path.exists(os.path.join(args.env_dir, d, "ui_structure.json"))
    ])

    if args.max_trajectories:
        traj_dirs = traj_dirs[:args.max_trajectories]

    print(f"{'='*60}")
    print(f"AMEX SFT DATA GENERATION (from pre-composed 448x448)")
    print(f"{'='*60}")
    print(f"env_dir: {args.env_dir}")
    print(f"Trajectories: {len(traj_dirs)}")
    print(f"Output: {args.output}")
    print()

    all_samples: List[dict] = []
    action_type_counts: Dict[str, int] = {}

    for traj_idx, traj_stem in enumerate(traj_dirs):
        traj_path = os.path.join(args.env_dir, traj_stem)
        ui_path = os.path.join(traj_path, "ui_structure.json")
        pages_dir = os.path.join(traj_path, "pages")

        with open(ui_path, "r", encoding="utf-8") as f:
            ui_structure = json.load(f)

        source_label = f"amex_{traj_stem}"

        nav_samples = generate_trajectory_samples(ui_structure, pages_dir, source_label)
        grounding_samples = generate_grounding_samples(
            ui_structure, pages_dir, source_label, args.grounding_per_traj)
        captioning_samples = generate_captioning_samples(
            ui_structure, pages_dir, source_label, args.captioning_per_traj)

        samples = nav_samples + grounding_samples + captioning_samples
        all_samples.extend(samples)

        for s in samples:
            atype = s.get("action_type", "unknown")
            action_type_counts[atype] = action_type_counts.get(atype, 0) + 1

        if (traj_idx + 1) % 500 == 0 or traj_idx == len(traj_dirs) - 1:
            print(f"  [{traj_idx+1}/{len(traj_dirs)}] cumulative samples: {len(all_samples)}")

    for i, s in enumerate(all_samples):
        s["idx"] = i

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_samples, f, indent=2)

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Trajectories: {len(traj_dirs)}")
    print(f"Total samples: {len(all_samples)}")
    print(f"Action types:")
    for atype, count in sorted(action_type_counts.items()):
        print(f"  {atype:20s}: {count}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
