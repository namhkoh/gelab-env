"""
Generate action-aware ST-RL data from composed AMEX sim2real episodes.

This script scans one AMEX trajectory environment or a directory of many
trajectory environments and converts them into step-by-step ST-RL samples.

Expected environment layout per episode:
  <episode_dir>/
    ui_structure.json
    pages/
      page_0.png
      ...

The generated dataset keeps the classic GE-Lab ST-RL sample structure:
  - problem
  - solution
  - image
  - bbox_px

and additionally stores action-specific metadata for later reward updates:
  - action_type
  - start_box_px
  - end_box_px
  - type_text
  - page_id
  - target_page
  - episode_id

Coordinate note:
  This generator stores raw pixel coordinates from the AMEX environment.
  The previous `*_norm` names were misleading because these values are not
  normalized to 0-1000.

Usage:
  python data_engine/generate_amex_st_rl_data.py \
    --env_dir /home/irteam/data-vol1/amex_sft \
    --output /home/irteam/data-vol1/gelab-env/datas/st_rl_amex.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _sort_page_ids(page_ids: Iterable[str]) -> List[str]:
    def _key(page_id: str) -> Tuple[int, str]:
        try:
            return int(page_id.split("_")[1]), page_id
        except Exception:
            return 10**9, page_id

    return sorted(page_ids, key=_key)


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _canvas_size(ui_structure: Dict[str, Any]) -> Tuple[int, int]:
    metadata = ui_structure.get("metadata", {})
    canvas = metadata.get("canvas_size")
    if isinstance(canvas, list) and len(canvas) == 2 and canvas[0] and canvas[1]:
        return int(canvas[0]), int(canvas[1])
    return 1080, 2400


def _bbox_to_pixels(bbox: Optional[List[int]], canvas_w: int, canvas_h: int) -> List[int]:
    if not bbox or len(bbox) != 4:
        return [0, 0, 0, 0]
    return [int(b) for b in bbox]


def _point_to_pixels(point: Optional[List[int]], canvas_w: int, canvas_h: int) -> Optional[Tuple[int, int]]:
    if not point or len(point) < 2:
        return None
    return (int(point[0]), int(point[1]))


def _bbox_center(bbox: List[int]) -> List[int]:
    return [(bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2]


def _valid_bbox(bbox: Optional[List[int]]) -> bool:
    return bool(bbox) and len(bbox) == 4 and bbox != [0, 0, 0, 0]


def _point_box(point: List[int], radius: int = 24) -> List[int]:
    x, y = point[0], point[1]
    return [x - radius, y - radius, x + radius, y + radius]


def _find_closest_layout_element(
    point: Optional[List[int]],
    bbox: Optional[List[int]],
    layout: Dict[str, Any],
    *,
    include_system: bool = False,
) -> Tuple[str, List[int]]:
    best_name = "element"
    best_bbox = [0, 0, 0, 0]
    best_score = float("inf")
    best_iou = -1.0

    ref_x = None
    ref_y = None
    if point and len(point) >= 2:
        ref_x, ref_y = point[0], point[1]
    elif bbox and len(bbox) == 4:
        ref_x, ref_y = _bbox_center(bbox)

    for key, value in layout.items():
        if not include_system and key in {"back", "home"}:
            continue
        if isinstance(value, dict):
            cand_bbox = value.get("bbox", [0, 0, 0, 0])
        elif isinstance(value, list) and len(value) == 4:
            cand_bbox = value
        else:
            continue
        if len(cand_bbox) != 4:
            continue

        x1, y1, x2, y2 = cand_bbox
        if bbox and len(bbox) == 4:
            bx1, by1, bx2, by2 = bbox
            inter_x1 = max(x1, bx1)
            inter_y1 = max(y1, by1)
            inter_x2 = min(x2, bx2)
            inter_y2 = min(y2, by2)
            inter_w = max(0, inter_x2 - inter_x1)
            inter_h = max(0, inter_y2 - inter_y1)
            inter_area = inter_w * inter_h
            cand_area = max((x2 - x1) * (y2 - y1), 1)
            bbox_area = max((bx2 - bx1) * (by2 - by1), 1)
            union_area = cand_area + bbox_area - inter_area
            iou = inter_area / union_area if union_area > 0 else 0.0
            if iou > 0:
                area_delta = abs(cand_area - bbox_area)
                if iou > best_iou or (iou == best_iou and area_delta < best_score):
                    best_iou = iou
                    best_score = area_delta
                    best_name = key
                    best_bbox = cand_bbox
                continue

        if ref_x is not None and ref_y is not None and x1 <= ref_x <= x2 and y1 <= ref_y <= y2:
            area = max((x2 - x1) * (y2 - y1), 1)
            if area < best_score:
                best_score = area
                best_name = key
                best_bbox = cand_bbox
            continue

        cx, cy = _bbox_center(cand_bbox)
        if ref_x is None or ref_y is None:
            dist = 0
        else:
            dist = (ref_x - cx) ** 2 + (ref_y - cy) ** 2
        if dist < best_score:
            best_score = dist
            best_name = key
            best_bbox = cand_bbox

    return best_name, best_bbox


def _infer_swipe_direction(start: Optional[List[int]], end: Optional[List[int]]) -> str:
    if not start or not end:
        return "unknown"
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    if abs(dy) > abs(dx):
        return "up" if dy < 0 else "down"
    return "left" if dx < 0 else "right"


def _escape_text(text: str) -> str:
    return text.replace("\\", "\\\\").replace("'", "\\'")


def _format_tap_action(action_name: str, point_norm: Tuple[int, int]) -> str:
    x, y = point_norm
    return (
        f"Explain: tap {action_name}.\t"
        f"Action: tap(start_box='<|box_start|>({x},{y})<|box_end|>')"
    )


def _format_swipe_action(
    start_norm: Optional[Tuple[int, int]],
    end_norm: Optional[Tuple[int, int]],
    direction: str,
) -> str:
    if start_norm and end_norm:
        sx, sy = start_norm
        ex, ey = end_norm
        return (
            f"Explain: swipe {direction}.\t"
            f"Action: swipe(start_box='<|box_start|>({sx},{sy})<|box_end|>', "
            f"end_box='<|box_start|>({ex},{ey})<|box_end|>')"
        )
    return f"Explain: swipe {direction}.\tAction: swipe()"


def _format_type_action(
    text: str,
    start_norm: Optional[Tuple[int, int]],
) -> str:
    safe_text = _escape_text(text)
    if start_norm:
        x, y = start_norm
        return (
            f"Explain: type \"{text}\" into the input field.\t"
            f"Action: type(start_box='<|box_start|>({x},{y})<|box_end|>', text='{safe_text}')"
        )
    return f"Explain: type \"{text}\" into the input field.\tAction: type(text='{safe_text}')"


def _format_press_enter_action() -> str:
    return "Explain: press enter.\tAction: press_enter()"


def _format_press_back_action(start_norm: Optional[Tuple[int, int]]) -> str:
    if start_norm:
        x, y = start_norm
        return (
            f"Explain: press back.\t"
            f"Action: press_back(start_box='<|box_start|>({x},{y})<|box_end|>')"
        )
    return "Explain: press back.\tAction: press_back()"


def _format_press_home_action(start_norm: Optional[Tuple[int, int]]) -> str:
    if start_norm:
        x, y = start_norm
        return (
            f"Explain: press home.\t"
            f"Action: press_home(start_box='<|box_start|>({x},{y})<|box_end|>')"
        )
    return "Explain: press home.\tAction: press_home()"


def _format_complete_action() -> str:
    return "Explain: task is complete.\tAction: task complete"


def _format_impossible_action() -> str:
    return "Explain: task cannot be completed.\tAction: task impossible"


def _history_to_text(history_parts: List[str]) -> str:
    return "; ".join(history_parts) if history_parts else "Null"


def _problem_text(instruction: str, history_parts: List[str]) -> str:
    instruction = instruction.strip() or "Complete the task shown on the current mobile interface."
    return f"<image>Instruction: {instruction}. History: {_history_to_text(history_parts)}"


def _resolve_type_point(transition: Dict[str, Any], page: Dict[str, Any]) -> Tuple[Optional[List[int]], List[int], str]:
    action_coord = transition.get("action_coord")
    icon_bbox = transition.get("icon_bbox")
    layout = page.get("layout", {})

    if action_coord and len(action_coord) >= 2:
        bbox = icon_bbox if icon_bbox and len(icon_bbox) == 4 else _point_box(action_coord)
        name, _ = _find_closest_layout_element(action_coord, bbox, layout)
        return action_coord, bbox, name

    if icon_bbox and len(icon_bbox) == 4:
        center = _bbox_center(icon_bbox)
        name, _ = _find_closest_layout_element(center, icon_bbox, layout)
        return center, icon_bbox, name

    name, bbox = _find_closest_layout_element(None, None, layout)
    if bbox != [0, 0, 0, 0]:
        return _bbox_center(bbox), bbox, name

    return None, [0, 0, 0, 0], "input"


def _direct_episode_dirs(root: Path) -> List[str]:
    episode_dirs: List[str] = []
    for child in sorted(root.iterdir()):
        if child.is_dir() and (child / "ui_structure.json").is_file():
            episode_dirs.append(str(child))
    return episode_dirs


def _episode_dirs(env_dir: str) -> List[str]:
    env_path = Path(env_dir)
    direct_episode_dirs = _direct_episode_dirs(env_path)
    if direct_episode_dirs:
        return direct_episode_dirs

    nested_envs_dir = env_path / "envs"
    if nested_envs_dir.is_dir():
        nested_episode_dirs = _direct_episode_dirs(nested_envs_dir)
        if nested_episode_dirs:
            return nested_episode_dirs

    if (env_path / "ui_structure.json").is_file():
        return [str(env_path)]

    return []


def generate_episode_samples(episode_dir: str) -> List[Dict[str, Any]]:
    ui_path = os.path.join(episode_dir, "ui_structure.json")
    pages_dir = os.path.join(episode_dir, "pages")
    ui_structure = _load_json(ui_path)
    pages = ui_structure.get("pages", {})
    metadata = ui_structure.get("metadata", {})
    instruction = str(metadata.get("instruction", "")).strip()
    episode_id = str(metadata.get("episode_id", os.path.basename(episode_dir)))
    canvas_w, canvas_h = _canvas_size(ui_structure)

    samples: List[Dict[str, Any]] = []
    history_parts: List[str] = []

    for page_idx, page_id in enumerate(_sort_page_ids(pages.keys()), start=1):
        page = pages.get(page_id, {})
        transitions = page.get("transitions", [])
        if not transitions:
            continue

        transition = transitions[0]
        raw_action = str(transition.get("action", "")).strip().upper()
        target_page = str(transition.get("target_page", page_id))
        image_rel = page.get("image", f"{page_id}.png")
        image_path = os.path.abspath(os.path.join(pages_dir, image_rel))
        layout = page.get("layout", {})

        action_coord = transition.get("action_coord")
        icon_bbox = transition.get("icon_bbox")
        lift_coord = transition.get("lift_coord")
        type_text = str(transition.get("type_text", "") or "").strip()
        icon_name = transition.get("icon_name") or None

        action_type = raw_action
        bbox = icon_bbox if isinstance(icon_bbox, list) else [0, 0, 0, 0]
        start_point = action_coord if isinstance(action_coord, list) else None
        end_point = lift_coord if isinstance(lift_coord, list) else None
        solution = None
        history_entry = None

        if raw_action in {"TAP", "CLICK"}:
            elem_name, elem_bbox = _find_closest_layout_element(start_point, bbox, layout)
            if icon_name:
                elem_name = icon_name
            if start_point is None and _valid_bbox(bbox):
                start_point = _bbox_center(bbox)
            elif start_point is None and elem_bbox != [0, 0, 0, 0]:
                start_point = _bbox_center(elem_bbox)
            if not _valid_bbox(bbox):
                if _valid_bbox(elem_bbox):
                    bbox = elem_bbox
                elif start_point is not None:
                    bbox = _point_box(start_point)
                else:
                    bbox = [0, 0, 0, 0]
            if start_point is None:
                continue
            start_norm = _point_to_pixels(start_point, canvas_w, canvas_h)
            solution = _format_tap_action(elem_name, start_norm)
            history_entry = f"step{len(history_parts) + 1}: tap {elem_name}"

        elif raw_action in {"SWIPE", "SCROLL"}:
            direction = _infer_swipe_direction(start_point, end_point)
            start_norm = _point_to_pixels(start_point, canvas_w, canvas_h)
            end_norm = _point_to_pixels(end_point, canvas_w, canvas_h)
            if bbox == [0, 0, 0, 0] and start_point and end_point:
                x1, x2 = sorted([start_point[0], end_point[0]])
                y1, y2 = sorted([start_point[1], end_point[1]])
                bbox = [x1, y1, x2, y2]
            solution = _format_swipe_action(start_norm, end_norm, direction)
            history_entry = f"step{len(history_parts) + 1}: swipe {direction}"

        elif raw_action in {"TYPE", "TEXT"}:
            start_point, bbox, target_name = _resolve_type_point(transition, page)
            start_norm = _point_to_pixels(start_point, canvas_w, canvas_h)
            solution = _format_type_action(type_text, start_norm)
            history_entry = f"step{len(history_parts) + 1}: type \"{type_text}\""
            if not type_text:
                type_text = target_name

        elif raw_action == "PRESS_ENTER":
            solution = _format_press_enter_action()
            history_entry = f"step{len(history_parts) + 1}: press enter"

        elif raw_action == "PRESS_BACK":
            start_norm = _point_to_pixels(start_point, canvas_w, canvas_h)
            if not _valid_bbox(bbox) and start_point is not None:
                bbox = _point_box(start_point)
            solution = _format_press_back_action(start_norm)
            history_entry = f"step{len(history_parts) + 1}: press back"

        elif raw_action == "PRESS_HOME":
            start_norm = _point_to_pixels(start_point, canvas_w, canvas_h)
            if not _valid_bbox(bbox) and start_point is not None:
                bbox = _point_box(start_point)
            solution = _format_press_home_action(start_norm)
            history_entry = f"step{len(history_parts) + 1}: press home"

        elif raw_action == "TASK_COMPLETE":
            solution = _format_complete_action()
            history_entry = f"step{len(history_parts) + 1}: task complete"

        elif raw_action == "TASK_IMPOSSIBLE":
            solution = _format_impossible_action()
            history_entry = f"step{len(history_parts) + 1}: task impossible"

        else:
            continue

        start_box_px = [0, 0, 0, 0]
        end_box_px = [0, 0, 0, 0]
        if raw_action in {"SWIPE", "SCROLL"}:
            if start_point:
                start_box_px = _bbox_to_pixels(_point_box(start_point), canvas_w, canvas_h)
            if end_point:
                end_box_px = _bbox_to_pixels(_point_box(end_point), canvas_w, canvas_h)

        sample = {
            "idx": -1,
            "episode_id": episode_id,
            "step": page_idx,
            "page_id": page_id,
            "target_page": target_page,
            "action_type": action_type,
            "type_text": type_text,
            "bbox_px": _bbox_to_pixels(bbox, canvas_w, canvas_h),
            "start_box_px": start_box_px,
            "end_box_px": end_box_px,
            "image": image_path,
            "problem": _problem_text(instruction, history_parts),
            "solution": solution,
        }
        samples.append(sample)

        if history_entry:
            history_parts.append(history_entry)

    return samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate ST-RL data from composed AMEX sim2real episodes")
    parser.add_argument(
        "--env_dir",
        default="/home/irteam/data-vol1/amex_sft",
        help="Single episode directory or root directory containing many episode directories",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output json path. Defaults to <env_dir>/st_rl_amex.json for roots or <episode_dir>/st_rl_amex.json for single episodes.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    episode_dirs = _episode_dirs(args.env_dir)
    if not episode_dirs:
        raise FileNotFoundError(f"No episode directories with ui_structure.json found under: {args.env_dir}")

    output_path = args.output
    if output_path is None:
        output_path = os.path.join(args.env_dir, "st_rl_amex.json")
    output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    all_samples: List[Dict[str, Any]] = []
    total_errors = 0

    for episode_dir in episode_dirs:
        episode_name = os.path.basename(os.path.abspath(episode_dir))
        try:
            samples = generate_episode_samples(episode_dir)
            all_samples.extend(samples)
            print(f"[ok] {episode_name}: {len(samples)} samples")
        except Exception as e:
            total_errors += 1
            print(f"[error] {episode_name}: {e}")

    for idx, sample in enumerate(all_samples):
        sample["idx"] = idx

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_samples, f, indent=2, ensure_ascii=False)

    print()
    print("=" * 60)
    print("AMEX ST-RL DATASET")
    print("=" * 60)
    print(f"Episodes processed: {len(episode_dirs) - total_errors}/{len(episode_dirs)}")
    print(f"Total samples:      {len(all_samples)}")
    print(f"Output:             {output_path}")


if __name__ == "__main__":
    main()
