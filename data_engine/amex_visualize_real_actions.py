"""
Visualize AMEX trajectory actions on the original screenshots.

Given a trajectory id or annotation json, this script loads the AMEX steps,
opens the real screenshots, and saves per-step overlay images showing the
recorded action coordinates directly in screenshot space.

Example:
    python data_engine/amex_visualize_real_actions.py \
        --trajectory_id e8ba0101cbc74242b48af70a57dafdf5

    python data_engine/amex_visualize_real_actions.py \
        --trajectory_id 2024_3_18_17_19_e8ba0101cbc74242b48af70a57dafdf5 \
        --max_steps 4
"""

import argparse
import json
import math
import os
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont


DEFAULT_ANNOTATIONS_DIR = "/ext_hdd2/tsyou/AMEX_dataset/AMEX/instruction_anno"
DEFAULT_FALLBACK_ANNOTATIONS_DIR = "/ext_hdd2/tsyou/AMEX-dataset-subset/AMEX/instruction_anno"
DEFAULT_SCREENSHOTS_DIR = "/ext_hdd2/tsyou/AMEX_dataset/AMEX/screenshot"
DEFAULT_OUTPUT_ROOT = "data_engine/amex_real_action_overlays"

ACTION_COLORS = {
    "TAP": (255, 72, 72, 255),
    "CLICK": (255, 72, 72, 255),
    "SWIPE": (255, 150, 40, 255),
    "SCROLL": (255, 150, 40, 255),
    "TYPE": (255, 72, 72, 255),
    "TEXT": (255, 72, 72, 255),
    "PRESS_BACK": (235, 90, 90, 255),
    "PRESS_HOME": (90, 190, 90, 255),
    "PRESS_ENTER": (255, 72, 72, 255),
    "TASK_COMPLETE": (70, 180, 95, 255),
    "TASK_IMPOSSIBLE": (220, 90, 90, 255),
}


def _try_load_font(size: int) -> ImageFont.FreeTypeFont:
    for name in ("DejaVuSans.ttf", "FreeSans.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def _normalize_action(action: str) -> str:
    return str(action or "").strip().upper()


def _safe_step_index(step: dict, fallback: int) -> int:
    raw = step.get("step_index")
    if isinstance(raw, int) and raw > 0:
        return raw
    return fallback


def _resolve_annotation_path(trajectory_id: str,
                             annotations_dir: str,
                             fallback_annotations_dir: str) -> str:
    candidate_dirs = [annotations_dir]
    if fallback_annotations_dir and fallback_annotations_dir != annotations_dir:
        candidate_dirs.append(fallback_annotations_dir)

    if trajectory_id.endswith(".json") and os.path.exists(trajectory_id):
        return trajectory_id

    for base_dir in candidate_dirs:
        direct = os.path.join(base_dir, f"{trajectory_id}.json")
        if os.path.exists(direct):
            return direct

    for base_dir in candidate_dirs:
        if not base_dir or not os.path.isdir(base_dir):
            continue
        for annot_path in sorted(Path(base_dir).glob("*.json")):
            try:
                with open(annot_path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
            except Exception:
                continue
            episode_id = str(payload.get("episode_id", "")).strip()
            stem = annot_path.stem
            if trajectory_id == episode_id or trajectory_id == stem or stem.endswith(trajectory_id):
                return str(annot_path)

    raise FileNotFoundError(f"Could not find annotation json for trajectory_id={trajectory_id}")


def _resolve_screenshot_path(step: dict,
                             screenshots_dir: str,
                             episode_id: str,
                             step_number: int) -> Tuple[str, str]:
    screenshot_name = (
        step.get("image_path")
        or step.get("screenshot")
        or f"{episode_id}-{step_number}.png"
    )
    screenshot_name = str(screenshot_name)
    if os.path.isabs(screenshot_name) and os.path.exists(screenshot_name):
        return os.path.basename(screenshot_name), screenshot_name
    return screenshot_name, os.path.join(screenshots_dir, screenshot_name)


def _scale_point(coord: List[int],
                 device_dim: List[int],
                 image_size: Tuple[int, int]) -> Optional[Tuple[int, int]]:
    if not isinstance(coord, (list, tuple)) or len(coord) != 2:
        return None
    x, y = int(coord[0]), int(coord[1])
    if x == 0 and y == 0:
        return None

    img_w, img_h = image_size
    if isinstance(device_dim, (list, tuple)) and len(device_dim) == 2:
        dev_w, dev_h = int(device_dim[0]), int(device_dim[1])
        if dev_w > 0 and dev_h > 0:
            x = int(round(x * img_w / dev_w))
            y = int(round(y * img_h / dev_h))

    x = min(max(0, x), max(img_w - 1, 0))
    y = min(max(0, y), max(img_h - 1, 0))
    return x, y


def _scale_interest_region(step: dict,
                           image_size: Tuple[int, int]) -> Optional[List[int]]:
    region = step.get("interest_region")
    if not isinstance(region, (list, tuple)) or len(region) != 2:
        return None
    top_left = _scale_point(region[0], step.get("device_dim") or [], image_size)
    bottom_right = _scale_point(region[1], step.get("device_dim") or [], image_size)
    if top_left is None or bottom_right is None:
        return None
    x1, y1 = top_left
    x2, y2 = bottom_right
    left, right = sorted((x1, x2))
    top, bottom = sorted((y1, y2))
    if right - left < 3 or bottom - top < 3:
        return None
    return [left, top, right, bottom]


def _text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont) -> Tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def _clamp_box(box: List[int], image_size: Tuple[int, int]) -> List[int]:
    width, height = image_size
    left = min(max(0, int(box[0])), max(width - 1, 0))
    top = min(max(0, int(box[1])), max(height - 1, 0))
    right = min(max(left + 1, int(box[2])), width)
    bottom = min(max(top + 1, int(box[3])), height)
    return [left, top, right, bottom]


def _box_from_point(point: Tuple[int, int],
                    image_size: Tuple[int, int],
                    box_width: int,
                    box_height: int) -> List[int]:
    x, y = point
    return _clamp_box(
        [
            int(round(x - box_width / 2)),
            int(round(y - box_height / 2)),
            int(round(x + box_width / 2)),
            int(round(y + box_height / 2)),
        ],
        image_size,
    )


def _expand_box(box: List[int],
                image_size: Tuple[int, int],
                pad_x: int,
                pad_y: int) -> List[int]:
    return _clamp_box(
        [box[0] - pad_x, box[1] - pad_y, box[2] + pad_x, box[3] + pad_y],
        image_size,
    )


def _draw_crosshair(draw: ImageDraw.ImageDraw,
                    point: Tuple[int, int],
                    color: Tuple[int, int, int, int]):
    x, y = point
    draw.ellipse([x - 12, y - 12, x + 12, y + 12], outline=(255, 255, 255, 255), width=4)
    draw.ellipse([x - 6, y - 6, x + 6, y + 6], fill=color)
    draw.line([x - 28, y, x - 8, y], fill=(255, 255, 255, 255), width=4)
    draw.line([x + 8, y, x + 28, y], fill=(255, 255, 255, 255), width=4)
    draw.line([x, y - 28, x, y - 8], fill=(255, 255, 255, 255), width=4)
    draw.line([x, y + 8, x, y + 28], fill=(255, 255, 255, 255), width=4)


def _draw_highlight_box(draw: ImageDraw.ImageDraw,
                        box: List[int],
                        color: Tuple[int, int, int, int],
                        point: Optional[Tuple[int, int]] = None,
                        label: str = ""):
    fill = (color[0], color[1], color[2], 36)
    soft = (255, 255, 255, 220)
    draw.rounded_rectangle(box, radius=10, fill=fill, outline=soft, width=3)
    inset = _expand_box(box, (10**9, 10**9), -6, -6)
    draw.rounded_rectangle(inset, radius=8, outline=color, width=5)
    if point is not None:
        _draw_crosshair(draw, point, color)
    if label:
        font = _try_load_font(22)
        tw, th = _text_size(draw, label, font)
        pad_x = 10
        pad_y = 6
        bx1 = max(0, box[0])
        by1 = max(0, box[1] - th - pad_y * 2 - 8)
        bx2 = bx1 + tw + pad_x * 2
        by2 = by1 + th + pad_y * 2
        draw.rounded_rectangle([bx1, by1, bx2, by2], radius=10, fill=(255, 255, 255, 235), outline=color, width=3)
        draw.text((bx1 + pad_x, by1 + pad_y), label, fill=color, font=font)


def _draw_arrow(draw: ImageDraw.ImageDraw,
                start: Tuple[int, int],
                end: Tuple[int, int],
                color: Tuple[int, int, int, int],
                width: int = 8):
    draw.line([start, end], fill=color, width=width)
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = math.hypot(dx, dy)
    if length < 1:
        return
    ux = dx / length
    uy = dy / length
    arrow_len = min(28, max(18, int(length * 0.12)))
    left = (
        int(end[0] - ux * arrow_len - uy * (arrow_len * 0.45)),
        int(end[1] - uy * arrow_len + ux * (arrow_len * 0.45)),
    )
    right = (
        int(end[0] - ux * arrow_len + uy * (arrow_len * 0.45)),
        int(end[1] - uy * arrow_len - ux * (arrow_len * 0.45)),
    )
    draw.polygon([end, left, right], fill=color)


def _action_title(step: dict) -> str:
    action = _normalize_action(step.get("action", ""))
    if action == "PRESS_ENTER":
        return "PRESS ENTER"
    if action == "TASK_COMPLETE":
        return "TASK COMPLETE"
    if action == "TASK_IMPOSSIBLE":
        return "TASK IMPOSSIBLE"
    return action or "UNKNOWN"


def _header_lines(step: dict, step_number: int) -> List[str]:
    type_text = str(step.get("type_text", "") or "").strip()
    action_label = f"{step_number}:{_action_title(step)}"
    if type_text:
        action_label = f"{action_label} {type_text}"
    lines = [action_label]
    package_name = str(step.get("package_name", "") or "").strip()
    if package_name:
        lines.append(package_name)
    return lines


def _draw_header_box(draw: ImageDraw.ImageDraw,
                     image_size: Tuple[int, int],
                     lines: List[str],
                     fill: Tuple[int, int, int, int]):
    title_font = _try_load_font(24)
    sub_font = _try_load_font(16)
    padding_x = 12
    padding_y = 8
    line_gap = 4
    widths = []
    heights = []
    fonts = []
    wrapped_lines: List[str] = []

    for idx, line in enumerate(lines):
        font = title_font if idx == 0 else sub_font
        wrapped = textwrap.wrap(line, width=26) or [line]
        for wrapped_line in wrapped:
            widths.append(_text_size(draw, wrapped_line, font)[0])
            heights.append(_text_size(draw, wrapped_line, font)[1])
            fonts.append(font)
            wrapped_lines.append(wrapped_line)

    box_width = min(image_size[0] - 20, max(widths, default=0) + padding_x * 2)
    box_height = sum(heights) + line_gap * max(len(wrapped_lines) - 1, 0) + padding_y * 2
    x1 = 10
    y1 = 10
    draw.rounded_rectangle([x1, y1, x1 + box_width, y1 + box_height], radius=10, fill=(255, 255, 255, 235), outline=fill, width=3)
    y = y1 + padding_y
    for line, font, height in zip(wrapped_lines, fonts, heights):
        draw.text((x1 + padding_x, y), line, fill=fill, font=font)
        y += height + line_gap


def _render_step_overlay(trajectory: dict,
                         step: dict,
                         step_number: int,
                         total_steps: int,
                         screenshot_name: str,
                         screenshot_path: str) -> Tuple[Image.Image, Dict[str, object]]:
    with Image.open(screenshot_path) as img_handle:
        base = img_handle.convert("RGBA")

    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    action = _normalize_action(step.get("action", ""))
    color = ACTION_COLORS.get(action, (255, 220, 80, 255))
    device_dim = step.get("device_dim") or []
    touch_point = _scale_point(step.get("touch_coord") or [], device_dim, base.size)
    lift_point = _scale_point(step.get("lift_coord") or [], device_dim, base.size)
    interest_region = _scale_interest_region(step, base.size)
    del trajectory, total_steps, screenshot_name

    _draw_header_box(draw, base.size, _header_lines(step, step_number), fill=color)

    spatial_drawn = False
    type_text = str(step.get("type_text", "") or "").strip()
    if interest_region is not None:
        interest_region = _expand_box(interest_region, base.size, 18, 18)

    if action in ("SWIPE", "SCROLL") and touch_point and lift_point:
        _draw_arrow(draw, touch_point, lift_point, color)
        _draw_highlight_box(
            draw,
            _expand_box(_box_from_point(touch_point, base.size, 72, 72), base.size, 8, 8),
            color,
            point=touch_point,
            label="START",
        )
        _draw_highlight_box(
            draw,
            _expand_box(_box_from_point(lift_point, base.size, 72, 72), base.size, 8, 8),
            color,
            point=lift_point,
            label="END",
        )
        spatial_drawn = True
    elif action in ("TAP", "CLICK") and touch_point is not None:
        target_box = interest_region or _box_from_point(
            touch_point,
            base.size,
            max(150, int(base.size[0] * 0.24)),
            max(90, int(base.size[1] * 0.08)),
        )
        _draw_highlight_box(draw, target_box, color, point=touch_point, label=action)
        spatial_drawn = True
    elif touch_point is not None:
        if action in ("TYPE", "TEXT", "PRESS_ENTER"):
            target_box = interest_region or _box_from_point(
                touch_point,
                base.size,
                max(220, int(base.size[0] * 0.6)),
                max(90, int(base.size[1] * 0.08)),
            )
            label = _action_title(step)
        else:
            target_box = interest_region or _box_from_point(
                touch_point,
                base.size,
                max(120, int(base.size[0] * 0.18)),
                max(80, int(base.size[1] * 0.07)),
            )
            label = action or "ACT"
        _draw_highlight_box(draw, target_box, color, point=touch_point, label=label)
        if lift_point is not None and lift_point != touch_point and action not in ("TYPE", "TEXT", "PRESS_ENTER"):
            _draw_highlight_box(
                draw,
                _box_from_point(lift_point, base.size, 72, 72),
                (255, 255, 255, 255),
                point=lift_point,
                label="LIFT",
            )
        spatial_drawn = True

    composed = Image.alpha_composite(base, overlay).convert("RGB")
    return composed, {
        "action": action,
        "touch_coord": step.get("touch_coord") or [0, 0],
        "lift_coord": step.get("lift_coord") or [0, 0],
        "scaled_touch_coord": list(touch_point) if touch_point is not None else [0, 0],
        "scaled_lift_coord": list(lift_point) if lift_point is not None else [0, 0],
        "interest_region_bbox": interest_region or [0, 0, 0, 0],
        "spatial_overlay": spatial_drawn,
    }


def _build_overview(output_dir: str, overlay_paths: List[str], labels: List[str], cols: int = 3) -> Optional[str]:
    valid_paths = [path for path in overlay_paths if os.path.exists(path)]
    if not valid_paths:
        return None

    thumb_w = 280
    thumb_h = 520
    header_h = 46
    gap = 14
    cols = max(1, cols)
    rows = int(math.ceil(len(valid_paths) / cols))
    sheet_w = cols * thumb_w + (cols + 1) * gap
    sheet_h = rows * (thumb_h + header_h) + (rows + 1) * gap

    sheet = Image.new("RGB", (sheet_w, sheet_h), (244, 244, 247))
    draw = ImageDraw.Draw(sheet)
    font = _try_load_font(16)

    for idx, (path, label) in enumerate(zip(valid_paths, labels)):
        row = idx // cols
        col = idx % cols
        x = gap + col * thumb_w
        y = gap + row * (thumb_h + header_h)
        with Image.open(path) as img_handle:
            image = img_handle.convert("RGB")
        thumb = image.copy()
        thumb.thumbnail((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        draw.rounded_rectangle([x, y, x + thumb_w - 1, y + header_h + thumb_h - 1], radius=12, fill=(255, 255, 255))
        draw.text((x + 10, y + 10), label, fill=(20, 20, 20), font=font)
        paste_x = x + (thumb_w - thumb.size[0]) // 2
        paste_y = y + header_h + (thumb_h - thumb.size[1]) // 2
        sheet.paste(thumb, (paste_x, paste_y))

    output_path = os.path.join(output_dir, "overview.png")
    sheet.save(output_path)
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize AMEX real actions on original screenshots")
    parser.add_argument("--trajectory_id", type=str, required=True,
                        help="AMEX episode_id, annotation stem, or annotation json path")
    parser.add_argument("--annotations_dir", type=str, default=DEFAULT_ANNOTATIONS_DIR,
                        help="Primary AMEX annotation directory")
    parser.add_argument("--fallback_annotations_dir", type=str, default=DEFAULT_FALLBACK_ANNOTATIONS_DIR,
                        help="Fallback annotation directory")
    parser.add_argument("--screenshots_dir", type=str, default=DEFAULT_SCREENSHOTS_DIR,
                        help="Directory containing original screenshots")
    parser.add_argument("--output_root", type=str, default=DEFAULT_OUTPUT_ROOT,
                        help="Root directory for generated overlay images")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Optional limit for quick inspection")
    parser.add_argument("--overview_cols", type=int, default=3,
                        help="Number of columns in overview contact sheet")
    return parser.parse_args()


def main():
    args = parse_args()

    annotation_path = _resolve_annotation_path(
        args.trajectory_id,
        args.annotations_dir,
        args.fallback_annotations_dir,
    )
    with open(annotation_path, "r", encoding="utf-8") as f:
        trajectory = json.load(f)

    episode_id = str(trajectory.get("episode_id", "") or Path(annotation_path).stem)
    stem = Path(annotation_path).stem
    output_dir = os.path.join(args.output_root, stem)
    os.makedirs(output_dir, exist_ok=True)

    steps = list(trajectory.get("steps") or [])
    if args.max_steps is not None:
        steps = steps[:max(0, args.max_steps)]

    manifest_steps = []
    overlay_paths = []
    overview_labels = []

    for zero_idx, step in enumerate(steps):
        step_number = zero_idx + 1
        screenshot_name, screenshot_path = _resolve_screenshot_path(
            step,
            args.screenshots_dir,
            episode_id,
            step_number,
        )

        step_record = {
            "step_number": step_number,
            "step_id": step.get("step_id"),
            "step_index": _safe_step_index(step, step_number),
            "action": _normalize_action(step.get("action", "")),
            "type_text": str(step.get("type_text", "") or ""),
            "screenshot_name": screenshot_name,
            "screenshot_path": screenshot_path,
            "overlay_path": "",
            "exists": os.path.exists(screenshot_path),
        }

        if not os.path.exists(screenshot_path):
            manifest_steps.append(step_record)
            continue

        rendered, overlay_meta = _render_step_overlay(
            trajectory,
            step,
            step_number,
            len(steps),
            screenshot_name,
            screenshot_path,
        )

        file_stub = f"step_{step_number:02d}_{_normalize_action(step.get('action', '')) or 'UNKNOWN'}"
        output_path = os.path.join(output_dir, f"{file_stub}.png")
        rendered.save(output_path)

        step_record.update(overlay_meta)
        step_record["overlay_path"] = output_path
        manifest_steps.append(step_record)
        overlay_paths.append(output_path)
        overview_labels.append(f"{step_number:02d} {_action_title(step)}")

    overview_path = _build_overview(output_dir, overlay_paths, overview_labels, cols=args.overview_cols)

    manifest = {
        "trajectory_id_input": args.trajectory_id,
        "annotation_path": annotation_path,
        "episode_id": episode_id,
        "annotation_stem": stem,
        "instruction": trajectory.get("instruction", ""),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "screenshots_dir": args.screenshots_dir,
        "output_dir": output_dir,
        "overview_path": overview_path or "",
        "total_steps": len(steps),
        "steps": manifest_steps,
    }

    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Annotation: {annotation_path}")
    print(f"Episode: {episode_id}")
    print(f"Saved: {output_dir}")
    print(f"Manifest: {manifest_path}")
    if overview_path:
        print(f"Overview: {overview_path}")
    print(f"Rendered overlays: {len(overlay_paths)}/{len(steps)}")


if __name__ == "__main__":
    main()
