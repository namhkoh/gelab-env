"""
Generate SFT training data from pre-composed AMEX environments.

Reads already-composed trajectory envs from
    /ext_hdd2/tsyou/gelab-env/data_engine/amex_sft/<traj_stem>/
(each containing ui_structure.json + pages/) and emits sft_amex.json.

No AMEX raw dataset (annotation/screenshot/element_anno/icons) is needed —
composition is assumed to have been done already.

Usage:
    python data_engine/generate_amex_sft_data.py
    python data_engine/generate_amex_sft_data.py --max_trajectories 100
"""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import Any, Dict, List, Tuple

DEFAULT_ENVS_DIR = "/data/amex_envs"
DEFAULT_OUTPUT_PATH = "/data/datas/sft_amex.json"


def _bbox_to_normalized(bbox: List[int]) -> List[int]:
    if not bbox or bbox == [0, 0, 0, 0]:
        return [0, 0, 0, 0]
    return [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]


def _point_to_normalized(point: List[int]) -> Tuple[int, int]:
    return int(point[0]), int(point[1])


def _bbox_center_normalized(bbox: List[int]) -> Tuple[int, int]:
    return (int(bbox[0]) + int(bbox[2])) // 2, (int(bbox[1]) + int(bbox[3])) // 2


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


def _format_tap_action(action_name: str, coord: List[int]) -> str:
    cx, cy = _point_to_normalized(coord)
    return (
        f"Explain: tap {action_name}.\t"
        f"Action: tap(start_box='<|box_start|>({cx},{cy})<|box_end|>')"
    )


def _format_swipe_action(start: List[int], end: List[int], direction: str) -> str:
    sx, sy = _point_to_normalized(start)
    ex, ey = _point_to_normalized(end)
    return (
        f"Explain: swipe {direction}.\t"
        f"Action: swipe(start_box='<|box_start|>({sx},{sy})<|box_end|>', "
        f"end_box='<|box_start|>({ex},{ey})<|box_end|>')"
    )


def _format_type_action(text: str, coord: List[int]) -> str:
    cx, cy = _point_to_normalized(coord)
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


def _infer_swipe_direction(start: List[int], end: List[int]) -> str:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    if abs(dy) > abs(dx):
        return "up" if dy < 0 else "down"
    return "left" if dx < 0 else "right"


def _format_user_content(instruction: str, history: str) -> str:
    parts = ["<image>"]
    if instruction:
        parts.append(f"Goal: {instruction}")
    parts.append(f"History: {history}")
    return " ".join(parts)


def generate_trajectory_samples(
    ui_structure: dict,
    pages_dir: str,
    source_label: str,
) -> List[dict]:
    samples: List[dict] = []
    pages = ui_structure.get("pages", {})
    if not isinstance(pages, dict):
        return samples
    metadata = ui_structure.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    instruction = str(metadata.get("instruction") or ui_structure.get("instruction") or "").strip()
    page_ids = sorted(
        pages.keys(),
        key=lambda x: int(x.split("_")[1]) if x.split("_")[1].isdigit() else 0,
    )

    if not page_ids:
        return samples

    task_text = instruction or "complete the task"
    history_parts: List[str] = []

    for page_id in page_ids:
        page = pages.get(page_id)
        if not page or not isinstance(page, dict):
            continue

        transitions = page.get("transitions", [])
        if not transitions:
            continue

        t = transitions[0]
        action = str(t.get("action", "")).strip().upper()
        action_coord = t.get("action_coord", [0, 0])

        history = "; ".join(history_parts) if history_parts else "Null"
        user_content = _format_user_content(instruction, history)
        image_path = os.path.join(pages_dir, page.get("image", ""))

        if action == "TASK_COMPLETE":
            samples.append({
                "task": task_text,
                "messages": [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": _format_complete_action()},
                ],
                "images": [image_path],
                "bbox_norm": [0, 0, 0, 0],
                "source": source_label,
                "action_type": "TASK_COMPLETE",
            })
            continue

        if action == "TASK_IMPOSSIBLE":
            samples.append({
                "task": task_text,
                "messages": [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": _format_impossible_action()},
                ],
                "images": [image_path],
                "bbox_norm": [0, 0, 0, 0],
                "source": source_label,
                "action_type": "TASK_IMPOSSIBLE",
            })
            continue

        if action in ("TAP", "CLICK"):
            elem_key, elem_bbox = _find_closest_layout_element(action_coord, page.get("layout", {}))
            assistant_content = _format_tap_action(elem_key, action_coord)
            bbox_norm = _bbox_to_normalized(elem_bbox) if elem_bbox != [0, 0, 0, 0] else _bbox_to_normalized(
                [action_coord[0] - 20, action_coord[1] - 20, action_coord[0] + 20, action_coord[1] + 20]
            )
            history_parts.append(f"step{len(history_parts)+1}: tap {elem_key}")
            action_type = "TAP"

        elif action in ("SWIPE", "SCROLL"):
            lift_coord = t.get("lift_coord", action_coord)
            direction = _infer_swipe_direction(action_coord, lift_coord)
            assistant_content = _format_swipe_action(action_coord, lift_coord, direction)
            x_coords = sorted([action_coord[0], lift_coord[0]])
            y_coords = sorted([action_coord[1], lift_coord[1]])
            bbox_norm = _bbox_to_normalized([x_coords[0], y_coords[0], x_coords[1], y_coords[1]])
            history_parts.append(f"step{len(history_parts)+1}: swipe {direction}")
            action_type = "SWIPE"

        elif action in ("TYPE", "TEXT"):
            type_text = str(t.get("type_text", "")).strip()
            assistant_content = _format_type_action(type_text, action_coord)
            bbox_norm = _bbox_to_normalized(
                [action_coord[0] - 40, action_coord[1] - 20, action_coord[0] + 40, action_coord[1] + 20]
            )
            history_parts.append(f"step{len(history_parts)+1}: type \"{type_text}\"")
            action_type = "TYPE"

        elif action == "PRESS_ENTER":
            assistant_content = _format_press_enter_action()
            bbox_norm = [0, 0, 0, 0]
            history_parts.append(f"step{len(history_parts)+1}: press enter")
            action_type = "PRESS_ENTER"

        elif action == "PRESS_BACK":
            assistant_content = _format_press_back_action()
            bbox_norm = [0, 0, 0, 0]
            history_parts.append(f"step{len(history_parts)+1}: press back")
            action_type = "PRESS_BACK"

        elif action == "PRESS_HOME":
            assistant_content = _format_press_home_action()
            bbox_norm = [0, 0, 0, 0]
            history_parts.append(f"step{len(history_parts)+1}: press home")
            action_type = "PRESS_HOME"

        else:
            continue

        samples.append({
            "task": task_text,
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
    max_per_trajectory: int = 50,
) -> List[dict]:
    all_icons: List[Tuple[str, str, List[int]]] = []
    pages = ui_structure.get("pages", {})
    if not isinstance(pages, dict):
        return []
    for page_id, page in pages.items():
        if not isinstance(page, dict):
            continue
        layout = page.get("layout", {})
        if not isinstance(layout, dict):
            continue
        for key, value in layout.items():
            if key in ("back", "home"):
                continue
            bbox = value.get("bbox", [0, 0, 0, 0]) if isinstance(value, dict) else value
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue
            if bbox == [0, 0, 0, 0]:
                continue
            all_icons.append((page_id, key, bbox))

    if not all_icons:
        return []

    sampled = random.choices(all_icons, k=min(max_per_trajectory, len(all_icons)))
    samples = []
    for page_id, label, bbox in sampled:
        cx, cy = _bbox_center_normalized(bbox)
        image_path = os.path.join(pages_dir, ui_structure["pages"][page_id].get("image", ""))
        samples.append({
            "task": "grounding",
            "messages": [
                {"role": "user", "content": f"<image>Tap on {label} in the image."},
                {"role": "assistant", "content": f"Action: tap(start_box='<|box_start|>({cx},{cy})<|box_end|>')"},
            ],
            "images": [image_path],
            "bbox_norm": _bbox_to_normalized(bbox),
            "source": source_label,
            "action_type": "grounding",
        })
    return samples


def generate_captioning_samples(
    ui_structure: dict,
    pages_dir: str,
    source_label: str,
    max_per_trajectory: int = 50,
) -> List[dict]:
    all_icons: List[Tuple[str, str, List[int]]] = []
    pages = ui_structure.get("pages", {})
    if not isinstance(pages, dict):
        return []
    for page_id, page in pages.items():
        if not isinstance(page, dict):
            continue
        layout = page.get("layout", {})
        if not isinstance(layout, dict):
            continue
        for key, value in layout.items():
            if key in ("back", "home"):
                continue
            bbox = value.get("bbox", [0, 0, 0, 0]) if isinstance(value, dict) else value
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue
            if bbox == [0, 0, 0, 0]:
                continue
            all_icons.append((page_id, key, bbox))

    if not all_icons:
        return []

    sampled = random.choices(all_icons, k=min(max_per_trajectory, len(all_icons)))
    samples = []
    for page_id, label, bbox in sampled:
        cx, cy = _bbox_center_normalized(bbox)
        image_path = os.path.join(pages_dir, ui_structure["pages"][page_id].get("image", ""))
        samples.append({
            "task": "captioning",
            "messages": [
                {"role": "user", "content": f"<image>What is the icon at point ({cx}, {cy}) in the image?"},
                {"role": "assistant", "content": label.replace("_", " ")},
            ],
            "images": [image_path],
            "bbox_norm": _bbox_to_normalized(bbox),
            "source": source_label,
            "action_type": "captioning",
        })
    return samples


def process_single_env(
    env_dir: str,
    grounding_per_traj: int,
    captioning_per_traj: int,
) -> Tuple[List[dict], dict]:
    ui_path = os.path.join(env_dir, "ui_structure.json")
    if not os.path.exists(ui_path):
        return [], {}

    with open(ui_path, "r", encoding="utf-8") as f:
        ui_structure = json.load(f)

    pages_dir = os.path.join(env_dir, "pages")
    traj_stem = os.path.basename(env_dir.rstrip("/"))
    metadata = ui_structure.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    episode_id = metadata.get("episode_id") or traj_stem
    source_label = f"amex_{episode_id}"

    nav_samples = generate_trajectory_samples(ui_structure, pages_dir, source_label)
    grounding_samples = generate_grounding_samples(ui_structure, pages_dir, source_label, grounding_per_traj)
    captioning_samples = generate_captioning_samples(ui_structure, pages_dir, source_label, captioning_per_traj)

    all_samples = nav_samples + grounding_samples + captioning_samples
    stats = {
        "traj_stem": traj_stem,
        "episode_id": episode_id,
        "instruction": metadata.get("instruction") or ui_structure.get("instruction", ""),
        "nav_samples": len(nav_samples),
        "grounding_samples": len(grounding_samples),
        "captioning_samples": len(captioning_samples),
        "total_samples": len(all_samples),
    }
    return all_samples, stats


def parse_args():
    parser = argparse.ArgumentParser(description="Generate SFT data from pre-composed AMEX envs")
    parser.add_argument("--envs_dir", type=str, default=DEFAULT_ENVS_DIR,
                        help="Directory containing <traj_stem>/ui_structure.json + pages/")
    parser.add_argument("--output_path", type=str, default=DEFAULT_OUTPUT_PATH,
                        help="Output sft_amex.json path")
    parser.add_argument("--max_trajectories", type=int, default=None)
    parser.add_argument("--grounding_per_traj", type=int, default=20)
    parser.add_argument("--captioning_per_traj", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    traj_stems = sorted(
        d for d in os.listdir(args.envs_dir)
        if os.path.isdir(os.path.join(args.envs_dir, d))
        and os.path.exists(os.path.join(args.envs_dir, d, "ui_structure.json"))
    )
    if args.max_trajectories:
        traj_stems = traj_stems[:args.max_trajectories]

    print(f"{'='*60}")
    print("AMEX SFT DATA GENERATION (from pre-composed envs)")
    print(f"{'='*60}")
    print(f"Source:       {args.envs_dir}")
    print(f"Trajectories: {len(traj_stems)}")
    print(f"Output:       {args.output_path}")
    print()

    all_samples: List[dict] = []
    all_stats: List[dict] = []
    action_type_counts: Dict[str, int] = {}

    for i, traj_stem in enumerate(traj_stems):
        env_dir = os.path.join(args.envs_dir, traj_stem)
        try:
            samples, stats = process_single_env(
                env_dir, args.grounding_per_traj, args.captioning_per_traj
            )
        except Exception as e:
            print(f"[{i+1}/{len(traj_stems)}] {traj_stem}: ERROR {e}")
            continue

        all_samples.extend(samples)
        all_stats.append(stats)
        for s in samples:
            atype = s.get("action_type", "unknown")
            action_type_counts[atype] = action_type_counts.get(atype, 0) + 1
        print(f"[{i+1}/{len(traj_stems)}] {traj_stem}: {stats.get('total_samples', 0)} samples "
              f"(nav={stats.get('nav_samples', 0)}, "
              f"grounding={stats.get('grounding_samples', 0)}, "
              f"captioning={stats.get('captioning_samples', 0)})")

    for i, s in enumerate(all_samples):
        s["idx"] = i

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(all_samples, f, indent=2)

    stats_path = os.path.join(os.path.dirname(args.output_path) or ".", "generation_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_trajectories": len(all_stats),
            "total_samples": len(all_samples),
            "action_type_distribution": action_type_counts,
            "per_trajectory": all_stats,
        }, f, indent=2)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Trajectories processed: {len(all_stats)}")
    print(f"Total SFT samples:      {len(all_samples)}")
    print("Action type distribution:")
    for atype, count in sorted(action_type_counts.items()):
        print(f"  {atype:20s}: {count}")
    print(f"Output: {args.output_path}")


if __name__ == "__main__":
    main()
