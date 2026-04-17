"""
Generate SFT training data from augmented AMEX trajectories.

Reads augmented trajectories from datas/amex_gelab_augmented/ and produces
SFT training JSON compatible with Qwen2.5-VL + ms-swift.

Coordinate system: Normalized to (0,0)-(1000,1000).

The augmentation pipeline (augment_trajectories.py) updates layout bboxes
but NOT transition action_coords. This script uses the layout bboxes for
TAP targets so that coordinates match the augmented screenshots.

Usage:
    python data_engine/generate_augmented_sft_data.py
    python data_engine/generate_augmented_sft_data.py --max_trajectories 100
    python data_engine/generate_augmented_sft_data.py --output datas/sft_augmented_custom.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import Any, Dict, List, Tuple

DEFAULT_AUG_DIR = "datas/amex_gelab_augmented"
DEFAULT_OUTPUT = "datas/sft_amex_augmented.json"


# ===================================================================
#  Coordinate helpers
# ===================================================================

def normalize_coord(x: float, y: float, cw: int, ch: int) -> Tuple[int, int]:
    """Normalize raw pixel (x, y) to (0..1000, 0..1000)."""
    nx = max(0, min(1000, int(x / cw * 1000)))
    ny = max(0, min(1000, int(y / ch * 1000)))
    return nx, ny


def bbox_center(bbox: List[int]) -> Tuple[float, float]:
    return (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0


# ===================================================================
#  Format helpers  (match existing amex_gelab SFT conventions)
# ===================================================================

def fmt_click(icon_name: str, page_id: str, target_page: str,
              nx: int, ny: int) -> str:
    return (
        f"Explain: click {icon_name} on {page_id} to navigate to {target_page}.\t"
        f"Action: click(start_box='<|box_start|>({nx},{ny})<|box_end|>')"
    )


def fmt_swipe(direction: str, sx: int, sy: int, ex: int, ey: int) -> str:
    return (
        f"Explain: swipe {direction}.\t"
        f"Action: swipe(start_box='<|box_start|>({sx},{sy})<|box_end|>', "
        f"end_box='<|box_start|>({ex},{ey})<|box_end|>')"
    )


def fmt_type(text: str, nx: int, ny: int) -> str:
    return (
        f"Explain: type \"{text}\".\t"
        f"Action: type(start_box='<|box_start|>({nx},{ny})<|box_end|>', text='{text}')"
    )


def fmt_press_enter() -> str:
    return "Explain: press enter to confirm.\tAction: press_enter()"


def fmt_press_back() -> str:
    return "Explain: press back.\tAction: press_back()"


def fmt_complete() -> str:
    return "Explain: task is complete.\tAction: complete"


def fmt_impossible() -> str:
    return "Explain: task cannot be completed.\tAction: impossible"


def swipe_direction(start: List[int], end: List[int]) -> str:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    if abs(dy) > abs(dx):
        return "up" if dy < 0 else "down"
    return "left" if dx < 0 else "right"


# ===================================================================
#  Nav sample generation
# ===================================================================

def generate_nav_samples(
    ui: dict, pages_dir: str, canvas_w: int, canvas_h: int,
) -> List[dict]:
    """Generate one nav sample per forward trajectory step.

    Uses transitions[0] on each page as the trajectory-following action.
    For TAP, coordinates come from the layout bbox (augmentation-aware).
    """
    samples: List[dict] = []
    pages = ui.get("pages", {})
    instruction = (ui.get("metadata", {}) or {}).get("instruction", "")
    if not instruction:
        return samples

    # Sort pages in trajectory order
    def page_sort_key(pid: str):
        parts = pid.replace("page_", "").split("_")
        num = int(parts[0]) if parts[0].isdigit() else 999
        suffix = 1 if "popup" in pid else 0
        return (num, suffix)

    page_ids = sorted(pages.keys(), key=page_sort_key)

    for page_id in page_ids:
        page = pages[page_id]
        transitions = page.get("transitions", [])
        if not transitions:
            continue

        # First transition = forward trajectory action
        t = transitions[0]
        action = t["action"]
        target = t.get("target_page", "")
        icon_name = t.get("icon_name", "")
        image_path = os.path.join(pages_dir, page.get("image", f"{page_id}.png"))
        layout = page.get("layout", {})

        if action in ("TAP", "CLICK"):
            # Use layout bbox (augmentation-updated) if available
            if icon_name and icon_name in layout:
                entry = layout[icon_name]
                bbox = entry.get("bbox", [0, 0, 0, 0]) if isinstance(entry, dict) else entry
                cx, cy = bbox_center(bbox)
            else:
                cx, cy = t["action_coord"][0], t["action_coord"][1]
            nx, ny = normalize_coord(cx, cy, canvas_w, canvas_h)
            assistant = fmt_click(icon_name or "element", page_id, target, nx, ny)

        elif action in ("SWIPE", "SCROLL"):
            start = t.get("action_coord")
            end = t.get("lift_coord", start)
            if not start:
                continue
            sx, sy = normalize_coord(start[0], start[1], canvas_w, canvas_h)
            ex, ey = normalize_coord(end[0], end[1], canvas_w, canvas_h)
            direction = swipe_direction(start, end)
            assistant = fmt_swipe(direction, sx, sy, ex, ey)

        elif action in ("TYPE", "TEXT"):
            text = str(t.get("type_text", "")).strip()
            coord = t.get("action_coord")
            if coord:
                nx, ny = normalize_coord(coord[0], coord[1], canvas_w, canvas_h)
            elif icon_name and icon_name in layout:
                entry = layout[icon_name]
                bbox = entry.get("bbox", [0, 0, 0, 0]) if isinstance(entry, dict) else entry
                cx, cy = bbox_center(bbox)
                nx, ny = normalize_coord(cx, cy, canvas_w, canvas_h)
            else:
                # No coordinate info — use center of screen as fallback
                nx, ny = 500, 500
            assistant = fmt_type(text, nx, ny)

        elif action == "PRESS_ENTER":
            assistant = fmt_press_enter()

        elif action == "PRESS_BACK":
            if not t.get("action_coord"):
                assistant = fmt_press_back()
            else:
                assistant = fmt_press_back()

        elif action == "TASK_COMPLETE":
            assistant = fmt_complete()

        elif action == "TASK_IMPOSSIBLE":
            assistant = fmt_impossible()

        else:
            continue

        samples.append({
            "messages": [
                {"role": "user", "content": f"<image>{instruction}"},
                {"role": "assistant", "content": assistant},
            ],
            "images": [os.path.abspath(image_path)],
            "source": "amex_augmented_nav",
        })

    return samples


# ===================================================================
#  Grounding sample generation
# ===================================================================

def generate_grounding_samples(
    ui: dict, pages_dir: str, canvas_w: int, canvas_h: int,
    max_per_traj: int = 50,
) -> List[dict]:
    """Generate icon grounding samples: text -> click coordinates."""
    all_icons: List[Tuple[str, str, List[int]]] = []
    pages = ui.get("pages", {})

    for page_id, page in pages.items():
        layout = page.get("layout", {})
        for key, value in layout.items():
            if key in ("back", "home", "popup_overlay"):
                continue
            bbox = value.get("bbox", [0, 0, 0, 0]) if isinstance(value, dict) else value
            if bbox == [0, 0, 0, 0]:
                continue
            img_name = page.get("image", f"{page_id}.png")
            all_icons.append((page_id, key, bbox, img_name))

    if not all_icons:
        return []

    sampled = random.choices(all_icons, k=min(max_per_traj, len(all_icons)))
    samples = []
    for page_id, label, bbox, img_name in sampled:
        cx, cy = bbox_center(bbox)
        nx, ny = normalize_coord(cx, cy, canvas_w, canvas_h)
        image_path = os.path.join(pages_dir, img_name)
        samples.append({
            "messages": [
                {"role": "user", "content": (
                    f"<image>I want to click on {label.replace('_', ' ')}. "
                    "Please locate the target element I should interact with. (with point)"
                )},
                {"role": "assistant", "content": (
                    f"Action: click(start_box='<|box_start|>({nx},{ny})<|box_end|>')"
                )},
            ],
            "images": [os.path.abspath(image_path)],
            "source": "amex_augmented_grounding",
        })
    return samples


# ===================================================================
#  Understanding sample generation
# ===================================================================

def generate_understanding_samples(
    ui: dict, pages_dir: str, canvas_w: int, canvas_h: int,
    max_per_traj: int = 50,
) -> List[dict]:
    """Generate icon understanding samples: coordinates -> label."""
    all_icons: List[Tuple[str, str, List[int]]] = []
    pages = ui.get("pages", {})

    for page_id, page in pages.items():
        layout = page.get("layout", {})
        for key, value in layout.items():
            if key in ("back", "home", "popup_overlay"):
                continue
            bbox = value.get("bbox", [0, 0, 0, 0]) if isinstance(value, dict) else value
            if bbox == [0, 0, 0, 0]:
                continue
            img_name = page.get("image", f"{page_id}.png")
            all_icons.append((page_id, key, bbox, img_name))

    if not all_icons:
        return []

    sampled = random.choices(all_icons, k=min(max_per_traj, len(all_icons)))
    samples = []
    for page_id, label, bbox, img_name in sampled:
        cx, cy = bbox_center(bbox)
        nx, ny = normalize_coord(cx, cy, canvas_w, canvas_h)
        image_path = os.path.join(pages_dir, img_name)
        samples.append({
            "messages": [
                {"role": "user", "content": f"<image>What is the icon at point ({nx},{ny}) in the image?"},
                {"role": "assistant", "content": label.replace("_", " ")},
            ],
            "images": [os.path.abspath(image_path)],
            "source": "amex_augmented_understanding",
        })
    return samples


# ===================================================================
#  Per-trajectory processing
# ===================================================================

def process_trajectory(
    traj_dir: str,
    grounding_per_traj: int,
    understanding_per_traj: int,
) -> Tuple[List[dict], dict]:
    ui_path = os.path.join(traj_dir, "ui_structure.json")
    if not os.path.exists(ui_path):
        return [], {}

    with open(ui_path, "r", encoding="utf-8") as f:
        ui = json.load(f)

    pages_dir = os.path.join(traj_dir, "pages")
    metadata = ui.get("metadata", {}) or {}
    canvas_size = metadata.get("canvas_size", [1080, 2400])
    canvas_w, canvas_h = canvas_size[0], canvas_size[1]
    episode_id = metadata.get("episode_id", os.path.basename(traj_dir))

    nav = generate_nav_samples(ui, pages_dir, canvas_w, canvas_h)
    grounding = generate_grounding_samples(
        ui, pages_dir, canvas_w, canvas_h, grounding_per_traj)
    understanding = generate_understanding_samples(
        ui, pages_dir, canvas_w, canvas_h, understanding_per_traj)

    all_samples = nav + grounding + understanding
    stats = {
        "episode_id": episode_id,
        "instruction": metadata.get("instruction", ""),
        "nav": len(nav),
        "grounding": len(grounding),
        "understanding": len(understanding),
        "total": len(all_samples),
    }
    return all_samples, stats


# ===================================================================
#  Main
# ===================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Generate SFT data from augmented trajectories")
    p.add_argument("--aug_dir", default=DEFAULT_AUG_DIR,
                    help="Augmented trajectories directory")
    p.add_argument("--output", default=DEFAULT_OUTPUT,
                    help="Output SFT JSON path")
    p.add_argument("--max_trajectories", type=int, default=None)
    p.add_argument("--grounding_per_traj", type=int, default=20)
    p.add_argument("--understanding_per_traj", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    aug_dir = os.path.abspath(args.aug_dir)
    traj_stems = sorted(
        d for d in os.listdir(aug_dir)
        if os.path.isdir(os.path.join(aug_dir, d))
        and os.path.exists(os.path.join(aug_dir, d, "ui_structure.json"))
    )

    if args.max_trajectories:
        traj_stems = traj_stems[: args.max_trajectories]

    print(f"{'=' * 60}")
    print("AUGMENTED SFT DATA GENERATION")
    print(f"{'=' * 60}")
    print(f"Source:       {aug_dir}")
    print(f"Trajectories: {len(traj_stems)}")
    print(f"Output:       {args.output}")
    print()

    all_samples: List[dict] = []
    all_stats: List[dict] = []
    source_counts: Dict[str, int] = {}

    for i, stem in enumerate(traj_stems):
        traj_dir = os.path.join(aug_dir, stem)
        try:
            samples, stats = process_trajectory(
                traj_dir, args.grounding_per_traj, args.understanding_per_traj)
        except Exception as e:
            print(f"[{i + 1}/{len(traj_stems)}] {stem}: ERROR {e}")
            continue

        all_samples.extend(samples)
        all_stats.append(stats)
        for s in samples:
            src = s["source"]
            source_counts[src] = source_counts.get(src, 0) + 1

        if (i + 1) % 500 == 0 or i == len(traj_stems) - 1:
            print(f"[{i + 1}/{len(traj_stems)}] processed, "
                  f"running total: {len(all_samples)} samples")

    # Add indices
    for i, s in enumerate(all_samples):
        s["idx"] = i

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_samples, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"Trajectories processed: {len(all_stats)}")
    print(f"Total SFT samples:      {len(all_samples)}")
    print("Source distribution:")
    for src, count in sorted(source_counts.items()):
        print(f"  {src:35s}: {count}")
    nav_total = sum(s["nav"] for s in all_stats)
    print(f"\nNav breakdown: {nav_total} samples across {len(all_stats)} trajectories")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
