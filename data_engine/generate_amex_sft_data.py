"""
Generate SFT training data from AMEX trajectories.

Pipeline:
1. For each AMEX trajectory, deterministically compose pages using
   pre-extracted icons + element_anno bboxes (via amex_compose_deterministic.py)
2. Generate SFT samples following the existing GE-Lab format:
   - Navigation (click): TAP steps -> click(start_box=...)
   - Navigation (swipe): SWIPE steps -> swipe(start_box=..., end_box=...)
   - Complete: TASK_COMPLETE -> complete
   - Grounding: element label -> coordinates
   - Captioning: coordinates -> element label
3. Output sft_amex.json compatible with swift sft training

Usage:
    python data_engine/generate_amex_sft_data.py \
        --output_dir datas_amex \
        --max_trajectories 100

    python data_engine/generate_amex_sft_data.py \
        --trajectory_id 2024_3_20_16_19_c7de79edba6b442681ddcdc40adb3900 \
        --output_dir datas_amex
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from amex_compose_deterministic import (
    CANVAS_H,
    CANVAS_W,
    compose_trajectory,
)  # noqa: E402

NORM_SIZE = 1000


def _bbox_to_normalized(bbox: List[int]) -> List[int]:
    if not bbox or bbox == [0, 0, 0, 0]:
        return [0, 0, 0, 0]
    x1, y1, x2, y2 = bbox
    return [
        int(x1 * NORM_SIZE / CANVAS_W),
        int(y1 * NORM_SIZE / CANVAS_H),
        int(x2 * NORM_SIZE / CANVAS_W),
        int(y2 * NORM_SIZE / CANVAS_H),
    ]


def _point_to_normalized(point: List[int]) -> Tuple[int, int]:
    x, y = point[0], point[1]
    return (
        int(x * NORM_SIZE / CANVAS_W),
        int(y * NORM_SIZE / CANVAS_H),
    )


def _bbox_center_normalized(bbox: List[int]) -> Tuple[int, int]:
    norm = _bbox_to_normalized(bbox)
    return (norm[0] + norm[2]) // 2, (norm[1] + norm[3]) // 2


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


def _format_click_action(action_name: str, page_id: str, coord: List[int]) -> str:
    cx, cy = _point_to_normalized(coord)
    return (
        f"Explain: click {action_name} icon on {page_id}.\t"
        f"Action: click(start_box='<|box_start|>({cx},{cy})<|box_end|>')"
    )


def _format_swipe_action(page_id: str, start: List[int], end: List[int], direction: str) -> str:
    sx, sy = _point_to_normalized(start)
    ex, ey = _point_to_normalized(end)
    return (
        f"Explain: swipe {direction} on {page_id}.\t"
        f"Action: swipe(start_box='<|box_start|>({sx},{sy})<|box_end|>', "
        f"end_box='<|box_start|>({ex},{ey})<|box_end|>')"
    )


def _format_type_action(text: str, page_id: str, coord: List[int]) -> str:
    cx, cy = _point_to_normalized(coord)
    return (
        f"Explain: type \"{text}\" on {page_id}.\t"
        f"Action: type(start_box='<|box_start|>({cx},{cy})<|box_end|>', text='{text}')"
    )


def _format_complete_action() -> str:
    return "Explain: this is target page.\tAction: complete"


def _infer_swipe_direction(start: List[int], end: List[int]) -> str:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    if abs(dy) > abs(dx):
        return "up" if dy < 0 else "down"
    return "left" if dx < 0 else "right"


def _format_user_content(instruction: str, route_text: str, history: str) -> str:
    parts = ["<image>"]
    if instruction:
        parts.append(f"Goal: {instruction}")
    parts.append(route_text)
    parts.append(f"History: {history}")
    return " ".join(parts)


def generate_trajectory_samples(
    ui_structure: dict,
    trajectory: dict,
    pages_dir: str,
    source_label: str,
) -> List[dict]:
    """Generate SFT samples from a single composed trajectory."""
    samples: List[dict] = []
    pages = ui_structure.get("pages", {})
    instruction = str(trajectory.get("instruction", "")).strip()
    steps = trajectory.get("steps", [])
    page_ids = sorted(pages.keys(), key=lambda x: int(x.split("_")[1]) if x.split("_")[1].isdigit() else 0)

    if not page_ids:
        return samples

    start_page = page_ids[0]
    end_page = page_ids[-1]
    route_text = f"Instruction: from {start_page} to {end_page}."
    history_parts: List[str] = []

    for i, page_id in enumerate(page_ids):
        page = pages.get(page_id)
        if not page:
            continue

        transitions = page.get("transitions", [])
        if not transitions:
            continue

        t = transitions[0]
        action = str(t.get("action", "")).strip().upper()
        action_coord = t.get("action_coord", [0, 0])
        target_page = t.get("target_page", page_id)

        history = "; ".join(history_parts) if history_parts else "Null"
        user_content = _format_user_content(instruction, route_text, history)
        image_path = os.path.join(pages_dir, page.get("image", f"{page_id}.png"))

        if action == "TASK_COMPLETE":
            sample = {
                "task": instruction or f"From {start_page} to {end_page}",
                "route": f"From {start_page} to {end_page}",
                "messages": [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": _format_complete_action()},
                ],
                "images": [image_path],
                "bbox_norm": [0, 0, 0, 0],
                "source": source_label,
                "action_type": "complete",
            }
            samples.append(sample)
            continue

        if action in ("TAP", "CLICK"):
            elem_key, elem_bbox = _find_closest_layout_element(action_coord, page.get("layout", {}))
            assistant_content = _format_click_action(elem_key, page_id, action_coord)
            bbox_norm = _bbox_to_normalized(elem_bbox) if elem_bbox != [0, 0, 0, 0] else _bbox_to_normalized(
                [action_coord[0] - 20, action_coord[1] - 20, action_coord[0] + 20, action_coord[1] + 20]
            )
            history_parts.append(f"step{len(history_parts)+1}: click {elem_key} icon on {page_id}")
            action_type = "click"

        elif action in ("SWIPE", "SCROLL"):
            lift_coord = t.get("lift_coord", action_coord)
            direction = _infer_swipe_direction(action_coord, lift_coord)
            assistant_content = _format_swipe_action(page_id, action_coord, lift_coord, direction)
            x_coords = sorted([action_coord[0], lift_coord[0]])
            y_coords = sorted([action_coord[1], lift_coord[1]])
            bbox_norm = _bbox_to_normalized([x_coords[0], y_coords[0], x_coords[1], y_coords[1]])
            history_parts.append(f"step{len(history_parts)+1}: swipe {direction} on {page_id}")
            action_type = "swipe"

        elif action in ("TYPE", "TEXT"):
            type_text = str(t.get("type_text", "")).strip()
            assistant_content = _format_type_action(type_text, page_id, action_coord)
            bbox_norm = _bbox_to_normalized(
                [action_coord[0] - 40, action_coord[1] - 20, action_coord[0] + 40, action_coord[1] + 20]
            )
            history_parts.append(f"step{len(history_parts)+1}: type \"{type_text}\" on {page_id}")
            action_type = "type"

        elif action in ("PRESS_BACK", "PRESS_HOME", "PRESS_ENTER"):
            semantic = action.lower().replace("press_", "")
            assistant_content = (
                f"Explain: press {semantic} on {page_id}.\t"
                f"Action: press_{semantic}()"
            )
            bbox_norm = [0, 0, 0, 0]
            history_parts.append(f"step{len(history_parts)+1}: press {semantic} on {page_id}")
            action_type = f"press_{semantic}"

        else:
            continue

        sample = {
            "task": instruction or f"From {start_page} to {end_page}",
            "route": f"From {start_page} to {end_page}",
            "messages": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content},
            ],
            "images": [image_path],
            "bbox_norm": bbox_norm,
            "source": source_label,
            "action_type": action_type,
        }
        samples.append(sample)

    return samples


def generate_grounding_samples(
    ui_structure: dict,
    pages_dir: str,
    source_label: str,
    max_per_trajectory: int = 50,
) -> List[dict]:
    """Generate icon grounding samples from layout elements."""
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
        cx, cy = _bbox_center_normalized(bbox)
        image_path = os.path.join(pages_dir, ui_structure["pages"][page_id].get("image", f"{page_id}.png"))
        samples.append({
            "task": "grounding",
            "messages": [
                {"role": "user", "content": f"<image>Click on {label} in the image."},
                {"role": "assistant", "content": f"Action: click(start_box='<|box_start|>({cx},{cy})<|box_end|>')"},
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
    """Generate icon captioning samples from layout elements."""
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
        cx, cy = _bbox_center_normalized(bbox)
        image_path = os.path.join(pages_dir, ui_structure["pages"][page_id].get("image", f"{page_id}.png"))
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


def _list_trajectory_stems(annotations_dir: str) -> List[str]:
    stems = []
    for f in sorted(os.listdir(annotations_dir)):
        if f.endswith(".json"):
            stems.append(f[:-5])
    return stems


def _has_matching_icons(traj_stem: str, icons_base_dir: str) -> bool:
    return os.path.isdir(os.path.join(icons_base_dir, traj_stem))


def process_single_trajectory(
    traj_stem: str,
    annotations_dir: str,
    screenshots_dir: str,
    element_anno_dir: str,
    icons_base_dir: str,
    output_base_dir: str,
    grounding_per_traj: int,
    captioning_per_traj: int,
    crop_mode: str = "icons_on_screenshot",
    model_name: str = "gpt-5-mini-2025-08-07",
    client: Any = None,
) -> Tuple[List[dict], dict]:
    """Process one trajectory end-to-end: compose pages + generate SFT samples."""
    anno_path = os.path.join(annotations_dir, f"{traj_stem}.json")
    with open(anno_path, "r", encoding="utf-8") as f:
        trajectory = json.load(f)

    env_dir = os.path.join(output_base_dir, "envs", traj_stem)
    os.makedirs(env_dir, exist_ok=True)

    compose_trajectory(
        trajectory=trajectory,
        traj_stem=traj_stem,
        screenshots_dir=screenshots_dir,
        element_anno_dir=element_anno_dir,
        icons_base_dir=icons_base_dir,
        output_dir=env_dir,
        max_steps=None,
        crop_mode=crop_mode,
        model_name=model_name,
        client=client,
    )

    ui_path = os.path.join(env_dir, "ui_structure.json")
    if not os.path.exists(ui_path):
        return [], {}

    with open(ui_path, "r", encoding="utf-8") as f:
        ui_structure = json.load(f)

    pages_dir = os.path.join(env_dir, "pages")
    source_label = f"amex_{trajectory.get('episode_id', traj_stem)}"

    nav_samples = generate_trajectory_samples(ui_structure, trajectory, pages_dir, source_label)
    grounding_samples = generate_grounding_samples(ui_structure, pages_dir, source_label, grounding_per_traj)
    captioning_samples = generate_captioning_samples(ui_structure, pages_dir, source_label, captioning_per_traj)

    all_samples = nav_samples + grounding_samples + captioning_samples

    stats = {
        "traj_stem": traj_stem,
        "episode_id": trajectory.get("episode_id", ""),
        "instruction": trajectory.get("instruction", ""),
        "total_steps": len(trajectory.get("steps", [])),
        "nav_samples": len(nav_samples),
        "grounding_samples": len(grounding_samples),
        "captioning_samples": len(captioning_samples),
        "total_samples": len(all_samples),
    }

    return all_samples, stats


def parse_args():
    parser = argparse.ArgumentParser(description="Generate SFT data from AMEX trajectories")
    parser.add_argument("--trajectory_id", type=str, default=None,
                        help="Single trajectory stem to process (for testing)")
    parser.add_argument("--annotations_dir", type=str,
                        default="/ext_hdd2/tsyou/AMEX_dataset/AMEX/instruction_anno")
    parser.add_argument("--screenshots_dir", type=str,
                        default="/ext_hdd2/tsyou/AMEX_dataset/AMEX/screenshot")
    parser.add_argument("--element_anno_dir", type=str,
                        default="/ext_hdd2/tsyou/AMEX_dataset/AMEX/element_anno")
    parser.add_argument("--icons_dir", type=str,
                        default="/ext_hdd2/ttran/datasets/AMEX_dataset/ui_elements_output_clean/elements")
    parser.add_argument("--output_dir", type=str, default="datas_amex",
                        help="Output directory for SFT data and composed environments")
    parser.add_argument("--max_trajectories", type=int, default=None,
                        help="Limit number of trajectories to process")
    parser.add_argument("--grounding_per_traj", type=int, default=20,
                        help="Grounding samples per trajectory")
    parser.add_argument("--captioning_per_traj", type=int, default=20,
                        help="Captioning samples per trajectory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--crop_mode", type=str, default="deterministic_compose",
                        choices=["screenshot", "icons_on_screenshot", "icons_only",
                                 "deterministic_compose", "gpt_compose"],
                        help="Page composition mode (default: deterministic_compose)")
    parser.add_argument("--model_name", type=str, default="gpt-5-mini-2025-08-07",
                        help="OpenAI model for gpt_compose mode")
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.trajectory_id:
        traj_stems = [args.trajectory_id]
    else:
        all_stems = _list_trajectory_stems(args.annotations_dir)
        traj_stems = [s for s in all_stems if _has_matching_icons(s, args.icons_dir)]
        print(f"Found {len(all_stems)} annotations, {len(traj_stems)} have matching icon directories")
        if args.max_trajectories:
            traj_stems = traj_stems[:args.max_trajectories]

    client = None
    if args.crop_mode == "gpt_compose":
        from amex_compose_deterministic import _load_gpt_compose
        mod = _load_gpt_compose()
        client = mod.load_api_client()

    print(f"\n{'='*60}")
    print(f"AMEX SFT DATA GENERATION")
    print(f"{'='*60}")
    print(f"Trajectories to process: {len(traj_stems)}")
    print(f"Crop mode: {args.crop_mode}")
    print(f"Output: {args.output_dir}")
    print()

    all_samples: List[dict] = []
    all_stats: List[dict] = []
    action_type_counts: Dict[str, int] = {}

    for traj_idx, traj_stem in enumerate(traj_stems):
        print(f"\n[{traj_idx+1}/{len(traj_stems)}] {traj_stem}")
        try:
            samples, stats = process_single_trajectory(
                traj_stem=traj_stem,
                annotations_dir=args.annotations_dir,
                screenshots_dir=args.screenshots_dir,
                element_anno_dir=args.element_anno_dir,
                icons_base_dir=args.icons_dir,
                output_base_dir=args.output_dir,
                grounding_per_traj=args.grounding_per_traj,
                captioning_per_traj=args.captioning_per_traj,
                crop_mode=args.crop_mode,
                model_name=args.model_name,
                client=client,
            )
            all_samples.extend(samples)
            all_stats.append(stats)
            for s in samples:
                atype = s.get("action_type", "unknown")
                action_type_counts[atype] = action_type_counts.get(atype, 0) + 1
            print(f"  -> {stats.get('total_samples', 0)} samples "
                  f"(nav={stats.get('nav_samples', 0)}, "
                  f"grounding={stats.get('grounding_samples', 0)}, "
                  f"captioning={stats.get('captioning_samples', 0)})")
        except Exception as e:
            print(f"  -> ERROR: {e}")
            continue

    for i, s in enumerate(all_samples):
        s["idx"] = i

    sft_path = os.path.join(args.output_dir, "sft_amex.json")
    with open(sft_path, "w", encoding="utf-8") as f:
        json.dump(all_samples, f, indent=2)

    stats_path = os.path.join(args.output_dir, "generation_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_trajectories": len(all_stats),
            "total_samples": len(all_samples),
            "action_type_distribution": action_type_counts,
            "per_trajectory": all_stats,
        }, f, indent=2)

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Trajectories processed: {len(all_stats)}")
    print(f"Total SFT samples:      {len(all_samples)}")
    print(f"Action type distribution:")
    for atype, count in sorted(action_type_counts.items()):
        print(f"  {atype:20s}: {count}")
    print(f"Output: {sft_path}")


if __name__ == "__main__":
    main()
