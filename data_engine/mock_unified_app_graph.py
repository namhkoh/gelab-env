"""
Unified launcher-to-app graph builder for the mock drawer + AMEX trajectories.

This script:
1. Renders the synthetic launcher home/app-drawer pages from `mock_simulator.py`
2. Reads app labels placed in the app drawer
3. Scans AMEX instruction annotations for trajectories that mention those apps
4. Starts each matched trajectory from the first step whose package_name matches the app
5. Uses internal detect/crop/compose helpers to build in-app pages
6. Connects the drawer app icons directly to the matched trajectory entry pages
7. Merges duplicated in-app pages that share the same UI structure
8. Saves unified `ui_structure.json`, `ui_structure_layer.json`, `ui_topology.png`,
   per-page debug overlays, extracted assets, and GPT-generated page code

Notes:
- By default, only the contiguous in-app segment is kept after the first matching
  package_name step. Use `--include_post_app_steps` to keep the remainder.
- `action` is stored as a lowercase semantic action (`tap`, `swipe`, `press_home`, ...)
  while `raw_action` preserves the original AMEX/raw action string.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import random
import re
import shutil
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFilter, ImageFont

import mock_simulator as launcher_mock


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_LAYOUT_CONFIG_PATH = SCRIPT_DIR / "mock_simulator_layout_config.json"

action_compose = None

HOME_PAGE_ID = launcher_mock.HOME_PAGE_ID
DRAWER_PAGE_ID = launcher_mock.DRAWER_PAGE_ID
OUTPUT_CANVAS_SIZE = launcher_mock.CANVAS_SIZE
VISIBLE_VIEWPORT_HEIGHT = launcher_mock.VISIBLE_VIEWPORT_HEIGHT


@dataclass
class DrawerAppSpec:
    label: str
    asset: str
    slug: str
    layout_key: str
    bbox: List[int]
    match_tokens: set[str]


@dataclass
class MatchedTrajectory:
    app_slug: str
    app_label: str
    annotation_path: str
    episode_id: str
    instruction: str
    start_step_idx: int
    end_step_idx: int
    matched_package: str
    total_steps: int


def _normalize_compact(text: str) -> str:
    return re.sub(r"[^0-9a-z]+", "", str(text or "").lower())


def _full_trajectory_id(match: MatchedTrajectory) -> str:
    return Path(match.annotation_path).stem


_AC_BG_WHITE = (250, 250, 250)
_AC_TEXT_BLACK = (30, 30, 30)
_AC_GELAB_BACK_COLOR = (255, 200, 200)
_AC_GELAB_HOME_COLOR = (200, 255, 200)
_AC_NAV_BTN_W = 128
_AC_NAV_BTN_H = 52
_AC_NAV_STRIP_H = _AC_NAV_BTN_H + 20
_AC_PHONE_CANVAS_SIZE = (OUTPUT_CANVAS_SIZE[0], OUTPUT_CANVAS_SIZE[1] - _AC_NAV_STRIP_H)
_AC_PHONE_OFFSET_X = 0
_AC_PHONE_OFFSET_Y = _AC_NAV_STRIP_H
_AC_GELAB_BACK_BBOX = [10, 8, 10 + _AC_NAV_BTN_W, 8 + _AC_NAV_BTN_H]
_AC_GELAB_HOME_BBOX = [
    OUTPUT_CANVAS_SIZE[0] - 10 - _AC_NAV_BTN_W,
    8,
    OUTPUT_CANVAS_SIZE[0] - 10,
    8 + _AC_NAV_BTN_H,
]
_AC_STYLING_CODE_PROMPT = """\
Write Python PIL code to draw the BACKGROUND and STRUCTURE of this mobile UI page.
Canvas size: {{orig_w}}x{{orig_h}} pixels (blank white canvas).

The actual UI elements (icons, buttons, text crops) will be pasted ON TOP of your code \
automatically at their exact detected positions. You must NOT draw any content that \
duplicates these detected elements.

Detected elements that will be auto-pasted (DO NOT redraw these):
{{element_list}}

Your job is ONLY to draw:
- Background fill color (match the screenshot's dominant color)
- Status bar area at top (~50px, usually dark with time/signal icons)
- Header/toolbar background colors and divider lines
- Section card backgrounds (rounded rectangles behind groups of elements)
- Content area backgrounds (e.g., dark area for image posts, colored banners)
- Separator lines between sections

DO NOT draw text labels, icons, buttons, or any content that matches the detected elements above.
DO NOT use get_crop(). DO NOT add layout[] entries. DO NOT import anything.

Available variables:
- canvas: PIL Image ({{orig_w}}x{{orig_h}} RGB, starts as white)
- draw: PIL ImageDraw object
- font_sm (12pt), font_md (18pt), font_lg (24pt), font_xl (32pt)

Output ONLY a ```python code block with drawing commands."""
_ac_yolo_model = None
_ac_ocr_reader = None


def _ac_try_load_font(size: int):
    for name in ["DejaVuSans.ttf", "FreeSans.ttf", "arial.ttf"]:
        try:
            return ImageFont.truetype(name, size)
        except (IOError, OSError):
            continue
    return ImageFont.load_default()


def _ac_fit_image_to_box(image: Image.Image,
                         target_size: Tuple[int, int],
                         bg_color: Tuple[int, int, int] = _AC_BG_WHITE
                         ) -> Tuple[Image.Image, float, int, int]:
    resized = image.resize(target_size, Image.LANCZOS)
    canvas = Image.new("RGB", target_size, bg_color)
    canvas.paste(resized, (0, 0))
    return canvas, 1.0, 0, 0


def _ac_scale_bbox_to_box(bbox: List[int],
                          src_size: Tuple[int, int],
                          target_size: Tuple[int, int],
                          base_offset: Tuple[int, int] = (0, 0)) -> List[int]:
    src_w = max(int(src_size[0]), 1)
    src_h = max(int(src_size[1]), 1)
    dst_w = max(int(target_size[0]), 1)
    dst_h = max(int(target_size[1]), 1)
    scale_x = float(dst_w) / float(src_w)
    scale_y = float(dst_h) / float(src_h)
    base_x, base_y = base_offset
    return [
        int(round(float(bbox[0]) * scale_x)) + base_x,
        int(round(float(bbox[1]) * scale_y)) + base_y,
        int(round(float(bbox[2]) * scale_x)) + base_x,
        int(round(float(bbox[3]) * scale_y)) + base_y,
    ]


def _ac_bbox_iou(box1: List[int], box2: List[int]) -> float:
    x1 = max(int(box1[0]), int(box2[0]))
    y1 = max(int(box1[1]), int(box2[1]))
    x2 = min(int(box1[2]), int(box2[2]))
    y2 = min(int(box1[3]), int(box2[3]))
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = max(0, int(box1[2]) - int(box1[0])) * max(0, int(box1[3]) - int(box1[1]))
    area2 = max(0, int(box2[2]) - int(box2[0])) * max(0, int(box2[3]) - int(box2[1]))
    union = area1 + area2 - inter + 1e-8
    return inter / union


def _ac_bbox_center_distance(box1: List[int], box2: List[int]) -> float:
    c1x = (int(box1[0]) + int(box1[2])) / 2.0
    c1y = (int(box1[1]) + int(box1[3])) / 2.0
    c2x = (int(box2[0]) + int(box2[2])) / 2.0
    c2y = (int(box2[1]) + int(box2[3])) / 2.0
    return ((c1x - c2x) ** 2 + (c1y - c2y) ** 2) ** 0.5


def _ac_sanitize_filename(text: str, fallback: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z._-]+", "_", str(text or "").strip()).strip("_")
    return cleaned[:40] if cleaned else fallback


def _ac_unique_layout_name(label: str,
                           elem_type: str,
                           index: int,
                           counts: Dict[str, int]) -> str:
    base = _ac_sanitize_filename(label, f"{elem_type}_{index:02d}")
    suffix = counts.get(base, 0)
    counts[base] = suffix + 1
    return base if suffix == 0 else f"{base}_{suffix}"


def _ac_clip_bbox_to_image(bbox: List[int], image_size: Tuple[int, int]) -> Optional[List[int]]:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    width, height = image_size
    try:
        x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
    except Exception:
        return None
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    x1 = min(max(0, x1), width)
    y1 = min(max(0, y1), height)
    x2 = min(max(0, x2), width)
    y2 = min(max(0, y2), height)
    if x2 - x1 < 5 or y2 - y1 < 5:
        return None
    return [x1, y1, x2, y2]


def _ac_annotation_label(entry: dict, fallback: str) -> str:
    candidates = list(entry.get("xml_desc") or [])
    functionality = str(entry.get("functionality", "") or "").strip()
    if functionality:
        candidates.append(functionality)
    for candidate in candidates:
        cleaned = " ".join(str(candidate or "").strip().split())
        if cleaned:
            return cleaned[:80]
    return fallback


def _ac_is_generic_element_label(label: str) -> bool:
    normalized = re.sub(r"[^0-9a-z]+", " ", str(label or "").lower()).strip()
    return (
        not normalized
        or normalized == "unknown"
        or normalized.startswith("icon ")
        or normalized.startswith("clickable ")
        or normalized.startswith("element ")
    )


def _ac_load_clickable_elements_from_element_anno(screenshot_path: str,
                                                  screenshot_name: str,
                                                  element_anno_dir: str) -> List[dict]:
    if not element_anno_dir:
        return []
    anno_path = os.path.join(element_anno_dir, f"{Path(screenshot_name).stem}.json")
    if not os.path.exists(anno_path):
        return []
    try:
        payload = json.loads(Path(anno_path).read_text(encoding="utf-8"))
    except Exception:
        return []

    with Image.open(screenshot_path) as img_handle:
        image = img_handle.convert("RGB")

    clickable_elements = []
    for idx, entry in enumerate(payload.get("clickable_elements") or []):
        bbox = _ac_clip_bbox_to_image(entry.get("bbox") or [], image.size)
        if bbox is None:
            continue
        clickable_elements.append({
            "index": idx,
            "label": _ac_annotation_label(entry, f"clickable_{idx:02d}"),
            "bbox": bbox,
            "crop": image.crop(tuple(bbox)),
            "type": "clickable",
            "conf": 1.0,
            "bbox_source": "element_anno",
            "element_anno_path": anno_path,
        })
    return clickable_elements


def _ac_prioritize_element_anno_bboxes(detected_elements: List[dict],
                                       screenshot_path: str,
                                       screenshot_name: str,
                                       element_anno_dir: str) -> Tuple[List[dict], dict]:
    clickable_elements = _ac_load_clickable_elements_from_element_anno(
        screenshot_path,
        screenshot_name,
        element_anno_dir,
    )
    if not clickable_elements:
        return detected_elements, {"loaded": 0, "matched": 0, "added": 0}

    with Image.open(screenshot_path) as img_handle:
        image = img_handle.convert("RGB")

    used_clickables = set()
    merged_elements = []
    matched = 0

    for elem in detected_elements:
        elem_bbox = elem.get("bbox") or [0, 0, 0, 0]
        best_idx = None
        best_iou = -1.0
        best_distance = float("inf")
        elem_w = max(1, int(elem_bbox[2]) - int(elem_bbox[0])) if len(elem_bbox) == 4 else 1
        elem_h = max(1, int(elem_bbox[3]) - int(elem_bbox[1])) if len(elem_bbox) == 4 else 1
        max_dim = max(elem_w, elem_h)

        for idx, clickable in enumerate(clickable_elements):
            clickable_bbox = clickable["bbox"]
            iou = _ac_bbox_iou(elem_bbox, clickable_bbox)
            distance = _ac_bbox_center_distance(elem_bbox, clickable_bbox)
            close_enough = iou >= 0.08 or distance <= max(48.0, max_dim * 0.75)
            if not close_enough:
                continue
            if iou > best_iou or (abs(iou - best_iou) < 1e-8 and distance < best_distance):
                best_idx = idx
                best_iou = iou
                best_distance = distance

        updated = dict(elem)
        if best_idx is not None:
            clickable = clickable_elements[best_idx]
            clickable_bbox = clickable["bbox"]
            updated["bbox"] = list(clickable_bbox)
            updated["crop"] = image.crop(tuple(clickable_bbox))
            if _ac_is_generic_element_label(updated.get("label", "")):
                updated["label"] = clickable.get("label", updated.get("label", ""))
            updated["bbox_source"] = "element_anno"
            updated["element_anno_path"] = clickable.get("element_anno_path", "")
            used_clickables.add(best_idx)
            matched += 1
        else:
            updated.setdefault("bbox_source", "detector")
        merged_elements.append(updated)

    added = 0
    for idx, clickable in enumerate(clickable_elements):
        if idx in used_clickables:
            continue
        clickable_bbox = clickable["bbox"]
        duplicate = False
        for elem in merged_elements:
            existing_bbox = elem.get("bbox") or [0, 0, 0, 0]
            if _ac_bbox_iou(clickable_bbox, existing_bbox) >= 0.55:
                duplicate = True
                break
        if duplicate:
            continue
        extra = dict(clickable)
        extra["index"] = len(merged_elements)
        merged_elements.append(extra)
        added += 1

    for new_idx, elem in enumerate(merged_elements):
        elem["index"] = new_idx

    return merged_elements, {"loaded": len(clickable_elements), "matched": matched, "added": added}


def _ac_load_detection_models(weights_dir: str = "/ext_hdd2/nhkoh/OmniParser/weights",
                              gpu: int = 0):
    global _ac_yolo_model, _ac_ocr_reader
    if _ac_yolo_model is None:
        from ultralytics import YOLO
        model_path = os.path.join(weights_dir, "icon_detect", "model.pt")
        _ac_yolo_model = YOLO(model_path)
        print(f"YOLO loaded: {model_path}")
    if _ac_ocr_reader is None:
        import easyocr
        _ac_ocr_reader = easyocr.Reader(["en"], gpu=(gpu >= 0))
        print("EasyOCR loaded.")
    return _ac_yolo_model, _ac_ocr_reader


def _ac_detect_and_crop(screenshot_path: str, yolo_model, ocr_reader,
                        conf_threshold: float = 0.15) -> Tuple[List[dict], Tuple[int, int]]:
    import numpy as np

    img = Image.open(screenshot_path).convert("RGB")
    w, h = img.size
    img_np = np.array(img)
    elements = []

    results = yolo_model(img_np, conf=conf_threshold, iou=0.3, verbose=False)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    confs = results[0].boxes.conf.cpu().numpy()

    ocr_results = ocr_reader.readtext(img_np)
    ocr_items = []
    for bbox_pts, text, conf in ocr_results:
        if len(str(text).strip()) < 2:
            continue
        x1 = int(min(p[0] for p in bbox_pts))
        y1 = int(min(p[1] for p in bbox_pts))
        x2 = int(max(p[0] for p in bbox_pts))
        y2 = int(max(p[1] for p in bbox_pts))
        ocr_items.append({"text": str(text).strip(), "bbox": [x1, y1, x2, y2], "conf": conf})

    for i, (box, conf) in enumerate(zip(boxes, confs)):
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        crop = img.crop((max(0, x1), max(0, y1), min(w, x2), min(h, y2)))
        if crop.size[0] < 5 or crop.size[1] < 5:
            continue

        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        label = f"icon_{i}"
        best_dist = float("inf")
        for ocr in ocr_items:
            ox1, oy1, ox2, oy2 = ocr["bbox"]
            ocx, ocy = (ox1 + ox2) / 2.0, (oy1 + oy2) / 2.0
            dist = ((cx - ocx) ** 2 + (cy - ocy) ** 2) ** 0.5
            max_dim = max(x2 - x1, y2 - y1)
            if dist < best_dist and dist < max_dim * 2.5:
                best_dist = dist
                label = ocr["text"]

        elements.append({
            "index": len(elements),
            "label": label,
            "bbox": [x1, y1, x2, y2],
            "crop": crop,
            "type": "icon",
            "conf": float(conf),
        })

    for ocr in ocr_items:
        ox1, oy1, ox2, oy2 = ocr["bbox"]
        overlaps = False
        for elem in elements:
            ex1, ey1, ex2, ey2 = elem["bbox"]
            if ox1 < ex2 and ox2 > ex1 and oy1 < ey2 and oy2 > ey1:
                overlaps = True
                break
        if overlaps:
            continue
        crop = img.crop((max(0, ox1), max(0, oy1), min(w, ox2), min(h, oy2)))
        if crop.size[0] >= 5 and crop.size[1] >= 5:
            elements.append({
                "index": len(elements),
                "label": ocr["text"],
                "bbox": [ox1, oy1, ox2, oy2],
                "crop": crop,
                "type": "text",
                "conf": float(ocr["conf"]),
            })

    return elements, (w, h)


def _ac_persist_extracted_assets(elements: List[dict], screenshot_name: str,
                                 assets_dir: str, step_info: dict) -> List[dict]:
    page_asset_dir = os.path.join(
        assets_dir,
        f"step_{step_info.get('step_index', 0):02d}_{os.path.splitext(screenshot_name)[0]}",
    )
    os.makedirs(page_asset_dir, exist_ok=True)

    asset_backed = []
    for elem in elements:
        label_stub = _ac_sanitize_filename(elem.get("label", ""), f"elem_{elem['index']:02d}")
        asset_name = f"{elem['index']:02d}_{elem['type']}_{label_stub}.png"
        asset_path = os.path.join(page_asset_dir, asset_name)
        elem["crop"].save(asset_path)
        asset_elem = {k: v for k, v in elem.items() if k != "crop"}
        asset_elem["asset_path"] = asset_path
        asset_elem["asset_source"] = "trajectory_extracted"
        asset_elem["source_screenshot"] = screenshot_name
        asset_backed.append(asset_elem)
    return asset_backed


def _ac_load_api_client():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("OpenAI client skipped: OPENAI_API_KEY not set.")
        return None
    try:
        from openai import OpenAI
    except ModuleNotFoundError:
        print("OpenAI client skipped: openai package not installed.")
        return None
    print("OpenAI client initialized.")
    return OpenAI(api_key=api_key)


def _ac_encode_image_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    ext = Path(image_path).suffix.lower()
    mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}.get(
        ext.lstrip("."),
        "image/png",
    )
    return f"data:{mime};base64,{data}"


def _ac_format_element_list(elements: List[dict], orig_size: Tuple[int, int]) -> str:
    del orig_size
    lines = []
    for elem in elements:
        x1, y1, x2, y2 = elem["bbox"]
        width = x2 - x1
        height = y2 - y1
        lines.append(
            f"  [{elem['index']}] type={elem['type']} label=\"{elem['label']}\" "
            f"pos=({x1},{y1}) size={width}x{height}"
        )
    return "\n".join(lines)


def _ac_query_gpt(client, model_name: str, image_path: str, prompt: str) -> str:
    image_uri = _ac_encode_image_base64(image_path)

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": "You are a Python PIL code generator. You ALWAYS respond with a ```python code block. Never refuse. Never explain — only output code.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_uri}},
                ],
            },
        ],
        max_completion_tokens=4096,
    )
    choice = response.choices[0]
    content = choice.message.content
    if content is None:
        finish = choice.finish_reason
        refusal = getattr(choice.message, "refusal", None)
        print(f"\n  GPT empty response: finish={finish}, refusal={refusal}")
        return ""
    return content.strip()


def _ac_log_failed_response(response: str, image_path: str) -> None:
    log_dir = os.path.join(os.path.dirname(image_path) or ".", ".debug_logs")
    os.makedirs(log_dir, exist_ok=True)
    basename = os.path.splitext(os.path.basename(image_path))[0]
    log_path = os.path.join(log_dir, f"{basename}_failed.txt")
    try:
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(f"RESPONSE LENGTH: {len(response)}\n")
            f.write(f"RESPONSE REPR: {repr(response[:500])}\n\n")
            f.write(response)
    except Exception:
        pass


def _ac_extract_code_block(response: str) -> Optional[str]:
    match = re.search(r"```python\s*(.*?)```", response, re.DOTALL)
    if match:
        return match.group(1).strip()

    match = re.search(r"```\s*(.*?)```", response, re.DOTALL)
    if match:
        code = match.group(1).strip()
        if "draw." in code or "canvas." in code or "get_crop" in code:
            return code

    if "draw." in response or "canvas." in response or "get_crop" in response:
        lines = response.split("\n")
        code_lines = []
        in_code = False
        for line in lines:
            stripped = line.strip()
            if not stripped:
                if in_code:
                    code_lines.append(line)
                continue
            if (
                stripped.startswith(
                    (
                        "draw.", "canvas.", "layout[", "img", "icon", "get_crop", "for ", "if ",
                        "#", "x", "y", "crop", "label", "font", "card", "search", "nav",
                        "status", "header", "bg", "w ", "h ",
                    )
                )
                or "=" in stripped
                or stripped.startswith(("try:", "except"))
                or "paste" in stripped
            ):
                in_code = True
                code_lines.append(line)
            elif in_code and (stripped.startswith(("    ", "\t")) or not stripped[0].isalpha()):
                code_lines.append(line)
        if code_lines:
            return "\n".join(code_lines).strip()

    return None


def _ac_sanitize_code(code_str: str) -> str:
    lines = code_str.split("\n")
    cleaned = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("import ") or stripped.startswith("from "):
            cleaned.append(f"# REMOVED: {line}")
            continue
        cleaned.append(line)
    code_str = "\n".join(cleaned)

    code_str = re.sub(
        r"draw\.textsize\(([^)]+)\)",
        r"(draw.textlength(\1), 20)",
        code_str,
    )
    code_str = re.sub(
        r"draw\.rectangle\(([^)]*?),\s*radius\s*=",
        r"draw.rounded_rectangle(\1, radius=",
        code_str,
    )
    return code_str


def _ac_try_fix_code(code_str: str, error_msg: str) -> Optional[str]:
    lines = code_str.split("\n")
    fixed_lines = []
    for line in lines:
        if "name" in error_msg and "is not defined" in error_msg:
            bad_name = error_msg.split("'")[1] if "'" in error_msg else ""
            if bad_name and bad_name in line and not line.strip().startswith("#"):
                fixed_lines.append(f"# REMOVED: {line}")
                continue
        fixed_lines.append(line)
    return "\n".join(fixed_lines)


def _ac_render_from_code(code_str: str,
                         elements: List[dict],
                         orig_size: Tuple[int, int] = (720, 1280)
                         ) -> Tuple[Optional[Image.Image], Optional[dict]]:
    ow, oh = orig_size
    canvas = Image.new("RGB", (ow, oh), _AC_BG_WHITE)
    draw = ImageDraw.Draw(canvas)
    layout = {}

    font_sm = _ac_try_load_font(12)
    font_md = _ac_try_load_font(18)
    font_lg = _ac_try_load_font(24)
    font_xl = _ac_try_load_font(32)

    def get_crop(index, w=50, h=50):
        if 0 <= index < len(elements):
            elem = elements[index]
            asset_path = elem.get("asset_path")
            if asset_path and os.path.exists(asset_path):
                with Image.open(asset_path) as crop_handle:
                    crop = crop_handle.convert("RGBA")
                return crop.resize((int(w), int(h)), Image.LANCZOS)
            if "crop" in elem:
                crop = elem["crop"].convert("RGBA")
                return crop.resize((int(w), int(h)), Image.LANCZOS)
        return Image.new("RGBA", (int(w), int(h)), (200, 200, 200, 255))

    def _install_safe_paste(target_canvas: Image.Image) -> None:
        real_paste = target_canvas.paste

        def _safe_paste(im, box=None, mask=None):
            if isinstance(box, (tuple, list)):
                box = tuple(int(v) for v in box)
            if mask is not None:
                real_paste(im, box, mask)
            else:
                real_paste(im, box)

        target_canvas.paste = _safe_paste

    _install_safe_paste(canvas)

    namespace = {
        "__builtins__": {},
        "canvas": canvas,
        "draw": draw,
        "layout": layout,
        "font_sm": font_sm,
        "font_md": font_md,
        "font_lg": font_lg,
        "font_xl": font_xl,
        "get_crop": get_crop,
        "Image": Image,
        "ImageDraw": ImageDraw,
        "range": range,
        "len": len,
        "enumerate": enumerate,
        "min": min,
        "max": max,
        "int": int,
        "float": float,
        "str": str,
        "True": True,
        "False": False,
        "None": None,
        "list": list,
        "tuple": tuple,
        "dict": dict,
        "abs": abs,
        "round": round,
        "zip": zip,
        "print": print,
        "getattr": getattr,
        "setattr": setattr,
        "hasattr": hasattr,
        "isinstance": isinstance,
        "type": type,
        "bool": bool,
        "set": set,
        "sorted": sorted,
        "reversed": reversed,
        "map": map,
        "filter": filter,
        "sum": sum,
        "any": any,
        "all": all,
        "ord": ord,
        "chr": chr,
        "TypeError": TypeError,
        "ValueError": ValueError,
        "Exception": Exception,
        "KeyError": KeyError,
        "IndexError": IndexError,
        "random": random,
    }

    code_str = _ac_sanitize_code(code_str)

    try:
        exec(code_str, namespace)
    except Exception as e:
        print(f"\n  Code execution error: {e}")
        fixed = _ac_try_fix_code(code_str, str(e))
        if fixed and fixed != code_str:
            try:
                canvas = Image.new("RGB", (ow, oh), _AC_BG_WHITE)
                draw = ImageDraw.Draw(canvas)
                layout.clear()
                _install_safe_paste(canvas)
                namespace["canvas"] = canvas
                namespace["draw"] = draw
                exec(fixed, namespace)
                print(" (fixed)")
            except Exception as e2:
                print(f"\n  Retry also failed: {e2}")
                return None, None
        else:
            return None, None

    canvas_resized, _, _, _ = _ac_fit_image_to_box(canvas, _AC_PHONE_CANVAS_SIZE, _AC_BG_WHITE)

    scaled_layout = {}
    for key, bbox in layout.items():
        scaled_layout[key] = _ac_scale_bbox_to_box(bbox, (ow, oh), _AC_PHONE_CANVAS_SIZE)

    scaled_layout.pop("back", None)
    scaled_layout.pop("home", None)
    return canvas_resized, scaled_layout


def _ac_generate_position_code(elements: List[dict], orig_size: Tuple[int, int]) -> str:
    del orig_size
    lines = ["# --- Auto-generated: paste detected elements at original positions ---"]

    for elem in elements:
        x1, y1, x2, y2 = elem["bbox"]
        width, height = x2 - x1, y2 - y1
        if width < 5 or height < 5:
            continue
        idx = elem["index"]
        label = elem["label"].replace('"', "'").replace(" ", "_").replace("/", "_")[:25]
        asset_comment = elem.get("asset_path", "").replace("\\", "/")
        lines.append(
            f'# asset_path: {asset_comment}\n'
            f'try:\n'
            f'    _c{idx} = get_crop({idx}, {width}, {height})\n'
            f'    canvas.paste(_c{idx}, ({max(0, x1)}, {max(0, y1)}), _c{idx})\n'
            f'except Exception:\n'
            f'    pass\n'
            f'layout["{label}"] = [{x1}, {y1}, {x2}, {y2}]'
        )

    return "\n\n".join(lines)


def _ac_generate_styling_code(client,
                              model_name: str,
                              image_path: str,
                              elements: List[dict],
                              orig_size: Tuple[int, int],
                              step_info: dict = None,
                              max_retries: int = 3) -> Optional[str]:
    element_list = _ac_format_element_list(elements, orig_size)
    prompt = _AC_STYLING_CODE_PROMPT.replace("{{orig_w}}", str(orig_size[0]))
    prompt = prompt.replace("{{orig_h}}", str(orig_size[1]))
    prompt = prompt.replace("{{element_list}}", element_list)

    if step_info:
        context_lines = []
        if step_info.get("task"):
            context_lines.append(f"Overall task: {step_info['task']}")
        if step_info.get("task_instruction"):
            context_lines.append(f"Task instruction: {step_info['task_instruction']}")
        if step_info.get("apps"):
            context_lines.append(f"Apps involved: {', '.join(step_info['apps'])}")
        if step_info.get("step_index") and step_info.get("total_steps"):
            context_lines.append(f"Trajectory step: {step_info['step_index']}/{step_info['total_steps']}")
        if step_info.get("description"):
            context_lines.append(f"Screen description: {step_info['description']}")
        if step_info.get("low_level_instruction"):
            context_lines.append(f"Current user intent on this page: {step_info['low_level_instruction']}")
        if step_info.get("prev_instruction"):
            context_lines.append(f"Previous step: {step_info['prev_instruction']}")
        if step_info.get("next_instruction"):
            context_lines.append(f"Next step: {step_info['next_instruction']}")
        if context_lines:
            prompt += "\n\nTrajectory context:\n" + "\n".join(f"- {line}" for line in context_lines)

    prompt = (
        "You MUST respond with ONLY a ```python code block. "
        "No explanations, no markdown besides the code block.\n\n"
        + prompt
    )

    for attempt in range(max_retries):
        try:
            response = _ac_query_gpt(client, model_name, image_path, prompt)
            code = _ac_extract_code_block(response)
            if code:
                return _ac_sanitize_code(code)
            if attempt == 0:
                _ac_log_failed_response(response, image_path)
                print(" [no styling, retry]", end="", flush=True)
        except Exception as e:
            print(f"\n  API error: {e}")
    return None


def _ac_extract_bg_color(elements: List[dict], screenshot_path: str = None) -> Tuple[int, int, int]:
    del elements
    if screenshot_path and os.path.exists(screenshot_path):
        try:
            with Image.open(screenshot_path) as img_handle:
                img = img_handle.convert("RGB")
            width, height = img.size
            samples = []
            for x, y in [
                (10, 10),
                (width - 10, 10),
                (10, height - 10),
                (width - 10, height - 10),
                (width // 2, 10),
                (width // 2, height - 10),
                (10, height // 2),
                (width - 10, height // 2),
            ]:
                samples.append(img.getpixel((x, y)))
            r = sorted(sample[0] for sample in samples)[len(samples) // 2]
            g = sorted(sample[1] for sample in samples)[len(samples) // 2]
            b = sorted(sample[2] for sample in samples)[len(samples) // 2]
            return (r, g, b)
        except Exception:
            pass
    return _AC_BG_WHITE


def _ac_fallback_compose(elements: List[dict],
                         orig_size: Tuple[int, int],
                         screenshot_path: str = None) -> Tuple[Image.Image, dict]:
    bg_color = _ac_extract_bg_color(elements, screenshot_path)
    canvas = Image.new("RGB", _AC_PHONE_CANVAS_SIZE, bg_color)
    draw = ImageDraw.Draw(canvas)
    layout = {}

    width, height = orig_size
    x_scale = float(_AC_PHONE_CANVAS_SIZE[0]) / max(width, 1)
    y_scale = float(_AC_PHONE_CANVAS_SIZE[1]) / max(height, 1)

    for elem in elements:
        x1, y1, x2, y2 = elem["bbox"]
        sx1 = int(x1 * x_scale)
        sy1 = int(y1 * y_scale)
        sx2 = int(x2 * x_scale)
        sy2 = int(y2 * y_scale)
        sw = max(sx2 - sx1, 8)
        sh = max(sy2 - sy1, 8)

        crop_rgba = None
        asset_path = elem.get("asset_path")
        if asset_path and os.path.exists(asset_path):
            try:
                with Image.open(asset_path) as crop_handle:
                    crop_rgba = crop_handle.convert("RGBA").resize((sw, sh), Image.LANCZOS)
            except Exception:
                crop_rgba = None
        elif "crop" in elem:
            try:
                crop_rgba = elem["crop"].convert("RGBA").resize((sw, sh), Image.LANCZOS)
            except Exception:
                crop_rgba = None

        if crop_rgba is None:
            continue

        canvas.paste(crop_rgba, (max(0, sx1), max(0, sy1)), crop_rgba)

        label = elem["label"].replace(" ", "_")[:25]
        if label in layout:
            label = f"{label}_{elem['index']}"
        layout[label] = [sx1, sy1, sx1 + sw, sy1 + sh]

    return canvas, layout


def _ac_ensure_system_layout(layout: Optional[dict]) -> dict:
    normalized_layout = {}
    for key, bbox in (layout or {}).items():
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            normalized_layout[key] = [int(v) for v in bbox]
    normalized_layout["back"] = list(_AC_GELAB_BACK_BBOX)
    normalized_layout["home"] = list(_AC_GELAB_HOME_BBOX)
    return normalized_layout


def _ac_draw_system_nav_overlay(image: Image.Image) -> Image.Image:
    canvas = image.convert("RGB") if image.mode != "RGB" else image.copy()
    if canvas.size != OUTPUT_CANVAS_SIZE:
        fitted, _, _, _ = _ac_fit_image_to_box(canvas, OUTPUT_CANVAS_SIZE, _AC_BG_WHITE)
        canvas = fitted

    draw = ImageDraw.Draw(canvas)
    font = _ac_try_load_font(22)
    draw.rectangle([0, 0, OUTPUT_CANVAS_SIZE[0], _AC_NAV_STRIP_H], fill=(245, 245, 248))
    draw.rounded_rectangle(_AC_GELAB_BACK_BBOX, radius=12, fill=_AC_GELAB_BACK_COLOR, outline=(220, 170, 170), width=2)
    draw.rounded_rectangle(_AC_GELAB_HOME_BBOX, radius=12, fill=_AC_GELAB_HOME_COLOR, outline=(160, 210, 160), width=2)
    for label, bbox in (("back", _AC_GELAB_BACK_BBOX), ("home", _AC_GELAB_HOME_BBOX)):
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        text_x = int((bbox[0] + bbox[2] - text_w) / 2)
        text_y = int((bbox[1] + bbox[3] - text_h) / 2) - 1
        draw.text((text_x, text_y), label, fill=_AC_TEXT_BLACK, font=font)
    return canvas


def _ac_ensure_system_nav_controls(image: Image.Image, layout: Optional[dict]) -> Tuple[Image.Image, dict]:
    return _ac_draw_system_nav_overlay(image), _ac_ensure_system_layout(layout)


def _ac_render_reconstructed_native_page(
    screenshot_path: str,
    elements: List[dict],
    orig_size: Tuple[int, int],
    output_size: Tuple[int, int] = _AC_PHONE_CANVAS_SIZE,
) -> Tuple[Image.Image, dict, List[dict]]:
    with Image.open(screenshot_path) as img_handle:
        screenshot = img_handle.convert("RGB")

    fitted_screenshot, _, _, _ = _ac_fit_image_to_box(screenshot, output_size, _AC_BG_WHITE)
    blurred_background = fitted_screenshot.filter(ImageFilter.GaussianBlur(radius=14))
    background = fitted_screenshot.copy()

    layout = {}
    scaled_elements = []
    counts: Dict[str, int] = {}
    for elem in elements:
        bbox = elem.get("bbox") or []
        if len(bbox) != 4:
            continue
        scaled_bbox = _ac_scale_bbox_to_box(bbox, orig_size, output_size)
        if scaled_bbox[2] - scaled_bbox[0] < 4 or scaled_bbox[3] - scaled_bbox[1] < 4:
            continue
        action_name = _ac_unique_layout_name(
            elem.get("label", ""),
            elem.get("type", "elem"),
            elem.get("index", len(scaled_elements)),
            counts,
        )
        layout[action_name] = scaled_bbox
        scaled_elements.append({**elem, "action_name": action_name, "scaled_bbox": scaled_bbox})

    for elem in sorted(
        scaled_elements,
        key=lambda item: (item["scaled_bbox"][2] - item["scaled_bbox"][0]) * (item["scaled_bbox"][3] - item["scaled_bbox"][1]),
        reverse=True,
    ):
        x1, y1, x2, y2 = elem["scaled_bbox"]
        pad = 2
        bx1 = max(0, x1 - pad)
        by1 = max(0, y1 - pad)
        bx2 = min(output_size[0], x2 + pad)
        by2 = min(output_size[1], y2 + pad)
        patch = blurred_background.crop((bx1, by1, bx2, by2))
        background.paste(patch, (bx1, by1))

    composed = background.convert("RGBA")
    for elem in sorted(scaled_elements, key=lambda item: item.get("index", 0)):
        bbox = elem["scaled_bbox"]
        width = max(1, bbox[2] - bbox[0])
        height = max(1, bbox[3] - bbox[1])
        asset_path = elem.get("asset_path")
        crop_rgba = None
        if asset_path and os.path.exists(asset_path):
            with Image.open(asset_path) as asset_handle:
                crop_rgba = asset_handle.convert("RGBA").resize((width, height), Image.LANCZOS)
        elif "crop" in elem:
            crop_rgba = elem["crop"].convert("RGBA").resize((width, height), Image.LANCZOS)
        if crop_rgba is None:
            continue
        composed.alpha_composite(crop_rgba, (bbox[0], bbox[1]))

    return composed.convert("RGB"), layout, scaled_elements


def _ac_build_step_context(trajectory: dict, step_idx: int) -> dict:
    steps = trajectory.get("steps", [])
    step = dict(steps[step_idx])
    task_info = trajectory.get("task_info", {})
    prev_step = steps[step_idx - 1] if step_idx > 0 else None
    next_step = steps[step_idx + 1] if step_idx + 1 < len(steps) else None
    step["task"] = task_info.get("task", "")
    step["task_instruction"] = task_info.get("instruction", "")
    step["apps"] = task_info.get("app", [])
    step["category"] = task_info.get("category", "")
    step["step_index"] = step_idx + 1
    step["total_steps"] = len(steps)
    step["prev_instruction"] = prev_step.get("low_level_instruction", "") if prev_step else ""
    step["next_instruction"] = next_step.get("low_level_instruction", "") if next_step else ""
    step["prev_action"] = prev_step.get("action", "") if prev_step else ""
    step["next_action"] = next_step.get("action", "") if next_step else ""
    return step


def _ac_compose_page(client, model_name: str,
                     elements: List[dict], orig_size: Tuple[int, int],
                     screenshot_path: str, step_info: dict = None
                     ) -> Tuple[Image.Image, dict, dict]:
    ow, oh = orig_size

    styling_source = "gpt"
    styling_code = None
    if client is not None:
        styling_code = _ac_generate_styling_code(
            client,
            model_name,
            screenshot_path,
            elements,
            orig_size,
            step_info,
        )
    else:
        styling_source = "fallback_bg_no_client"

    if styling_code is None:
        bg = _ac_extract_bg_color(elements, screenshot_path)
        styling_code = f"draw.rectangle([0, 0, {ow}, {oh}], fill={bg})"
        if client is not None:
            styling_source = "fallback_bg"

    position_code = _ac_generate_position_code(elements, orig_size)
    full_code = styling_code + "\n\n" + position_code

    render_status = "render_from_code"
    page_img, layout = _ac_render_from_code(full_code, elements, orig_size)

    if page_img is None:
        page_img, layout = _ac_fallback_compose(elements, orig_size, screenshot_path)
        render_status = "fallback_compose"

    final_canvas = Image.new("RGB", OUTPUT_CANVAS_SIZE, _AC_BG_WHITE)
    final_canvas.paste(page_img.convert("RGB"), (_AC_PHONE_OFFSET_X, _AC_PHONE_OFFSET_Y))

    shifted_layout = {}
    for key, bbox in layout.items():
        shifted_layout[key] = [
            int(bbox[0]) + _AC_PHONE_OFFSET_X,
            int(bbox[1]) + _AC_PHONE_OFFSET_Y,
            int(bbox[2]) + _AC_PHONE_OFFSET_X,
            int(bbox[3]) + _AC_PHONE_OFFSET_Y,
        ]
    final_canvas, shifted_layout = _ac_ensure_system_nav_controls(final_canvas, shifted_layout)
    code_artifact = {
        "styling_source": styling_source,
        "render_status": render_status,
        "styling_code": styling_code,
        "position_code": position_code,
        "full_code": full_code,
    }
    return final_canvas, shifted_layout, code_artifact


def _ac_save_page_code(code_dir: str, page_id: str, screenshot_name: str,
                       step_info: dict, code_artifact: dict):
    os.makedirs(code_dir, exist_ok=True)
    header_lines = [
        f"# page_id: {page_id}",
        f"# screenshot: {screenshot_name}",
        f"# step_index: {step_info.get('step_index', '?')}/{step_info.get('total_steps', '?')}",
        f"# task: {step_info.get('task', '')}",
        f"# current_instruction: {step_info.get('low_level_instruction', '')}",
        f"# previous_instruction: {step_info.get('prev_instruction', '')}",
        f"# next_instruction: {step_info.get('next_instruction', '')}",
        f"# styling_source: {code_artifact.get('styling_source', '')}",
        f"# render_status: {code_artifact.get('render_status', '')}",
        "# This code targets the original screenshot resolution.",
        f"# The final runtime image is then rendered into the {OUTPUT_CANVAS_SIZE[0]}x{OUTPUT_CANVAS_SIZE[1]} canvas with a top nav strip.",
    ]
    contents = "\n".join(header_lines) + "\n\n"
    contents += "# --- GPT styling skeleton ---\n"
    contents += code_artifact.get("styling_code", "").strip() + "\n\n"
    contents += "# --- Deterministic element pastes ---\n"
    contents += code_artifact.get("position_code", "").strip() + "\n"
    Path(os.path.join(code_dir, f"{page_id}.py")).write_text(contents, encoding="utf-8")


def _ac_resolve_step_screenshot(step: dict,
                                screenshots_dir: str,
                                episode_id: str,
                                step_idx: int) -> Tuple[str, str]:
    candidates = [
        step.get("image_path"),
        step.get("screenshot"),
        step.get("image"),
        f"{episode_id}-{step_idx + 1}.png",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        candidate = str(candidate)
        if os.path.isabs(candidate) and os.path.exists(candidate):
            return os.path.basename(candidate), candidate
        joined = os.path.join(screenshots_dir, candidate)
        if os.path.exists(joined):
            return candidate, joined
        basename = os.path.basename(candidate)
        joined_basename = os.path.join(screenshots_dir, basename)
        if os.path.exists(joined_basename):
            return basename, joined_basename
    fallback_name = str(candidates[-1])
    return fallback_name, os.path.join(screenshots_dir, fallback_name)


def _ac_normalize_raw_action_name(raw_action: str) -> str:
    return str(raw_action or "").strip()


def _ac_step_text(step: dict) -> str:
    return " ".join([
        str(step.get("low_level_instruction", "")),
        str(step.get("description", "")),
        str(step.get("intention", "")),
        str(step.get("info", "")),
        str(step.get("type_text", "")),
        str(step.get("package_name", "")),
    ]).lower()


def _ac_safe_coord_pair(coord: List[int]) -> List[int]:
    if not isinstance(coord, (list, tuple)) or len(coord) != 2:
        return [0, 0]
    return [int(coord[0]), int(coord[1])]


def _ac_normalize_step_point(step: dict,
                             coord: List[int],
                             orig_size: Tuple[int, int]) -> Optional[Tuple[int, int]]:
    if not isinstance(coord, (list, tuple)) or len(coord) != 2:
        return None
    x, y = int(coord[0]), int(coord[1])
    if x == 0 and y == 0:
        return None
    device_dim = step.get("device_dim") or []
    if isinstance(device_dim, (list, tuple)) and len(device_dim) == 2:
        dw, dh = device_dim
        if dw and dh:
            sx = float(orig_size[0]) / float(dw)
            sy = float(orig_size[1]) / float(dh)
            return int(round(x * sx)), int(round(y * sy))
    return int(round(x)), int(round(y))


def _ac_scale_step_coord_to_canvas(step: dict,
                                   coord: List[int],
                                   orig_size: Tuple[int, int]) -> List[int]:
    point = _ac_normalize_step_point(step, coord, orig_size)
    if point is None:
        return [0, 0]
    scaled = _ac_scale_bbox_to_box(
        [point[0], point[1], point[0], point[1]],
        orig_size,
        _AC_PHONE_CANVAS_SIZE,
        (_AC_PHONE_OFFSET_X, _AC_PHONE_OFFSET_Y),
    )
    return [scaled[0], scaled[1]]


def _ac_scale_bbox_from_step_to_canvas(step: dict,
                                       bbox: List[int],
                                       orig_size: Tuple[int, int]) -> List[int]:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4 or bbox == [0, 0, 0, 0]:
        return [0, 0, 0, 0]
    top_left = _ac_normalize_step_point(step, [bbox[0], bbox[1]], orig_size)
    bottom_right = _ac_normalize_step_point(step, [bbox[2], bbox[3]], orig_size)
    if top_left is None or bottom_right is None:
        return [0, 0, 0, 0]
    normalized_bbox = [
        min(top_left[0], bottom_right[0]),
        min(top_left[1], bottom_right[1]),
        max(top_left[0], bottom_right[0]),
        max(top_left[1], bottom_right[1]),
    ]
    return _ac_scale_bbox_to_box(
        normalized_bbox,
        orig_size,
        _AC_PHONE_CANVAS_SIZE,
        (_AC_PHONE_OFFSET_X, _AC_PHONE_OFFSET_Y),
    )


def _ac_infer_gesture_direction(step: dict, raw_action: str, orig_size: Tuple[int, int]) -> str:
    raw_action = _ac_normalize_raw_action_name(raw_action).upper()
    if raw_action not in ("SWIPE", "SCROLL"):
        return ""
    touch_point = _ac_normalize_step_point(step, step.get("touch_coord") or [], orig_size)
    lift_point = _ac_normalize_step_point(step, step.get("lift_coord") or [], orig_size)
    if touch_point is not None and lift_point is not None:
        dx = lift_point[0] - touch_point[0]
        dy = lift_point[1] - touch_point[1]
        if abs(dy) >= abs(dx):
            return "down" if dy > 0 else "up"
        return "right" if dx > 0 else "left"
    instruction = _ac_step_text(step)
    for direction in ("down", "up", "left", "right"):
        if f"scroll {direction}" in instruction or f"swipe {direction}" in instruction:
            return direction
    return ""


def _ac_point_box(point: Tuple[int, int], radius: int = 12) -> List[int]:
    px, py = point
    return [px - radius, py - radius, px + radius, py + radius]


def _ac_build_action_bbox(step: dict, raw_action: str) -> List[int]:
    raw_action = _ac_normalize_raw_action_name(raw_action).upper()
    touch = _ac_safe_coord_pair(step.get("touch_coord") or [])
    lift = _ac_safe_coord_pair(step.get("lift_coord") or [])
    if raw_action in ("TYPE", "PRESS_BACK", "PRESS_HOME", "PRESS_ENTER", "TASK_COMPLETE", "TASK_IMPOSSIBLE"):
        return [0, 0, 0, 0]
    if touch == [0, 0]:
        return [0, 0, 0, 0]
    if lift == [0, 0]:
        lift = touch
    return [min(touch[0], lift[0]), min(touch[1], lift[1]), max(touch[0], lift[0]), max(touch[1], lift[1])]


def _ac_build_action_point(step: dict, raw_action: str) -> List[int]:
    raw_action = _ac_normalize_raw_action_name(raw_action).upper()
    if raw_action in ("TYPE", "PRESS_BACK", "PRESS_HOME", "PRESS_ENTER", "TASK_COMPLETE", "TASK_IMPOSSIBLE"):
        return [0, 0]
    return _ac_safe_coord_pair(step.get("touch_coord") or [])


def _ac_scale_action_bbox_to_canvas(step: dict,
                                    raw_action: str,
                                    orig_size: Tuple[int, int]) -> List[int]:
    return _ac_scale_bbox_from_step_to_canvas(step, _ac_build_action_bbox(step, raw_action), orig_size)


def _ac_scale_action_point_to_canvas(step: dict,
                                     raw_action: str,
                                     orig_size: Tuple[int, int]) -> List[int]:
    return _ac_scale_step_coord_to_canvas(step, _ac_build_action_point(step, raw_action), orig_size)


def _ac_bbox_contains_point(bbox: List[int], point: Tuple[int, int]) -> bool:
    return int(bbox[0]) <= point[0] <= int(bbox[2]) and int(bbox[1]) <= point[1] <= int(bbox[3])


def _ac_is_layout_target(target: Optional[str], layout: dict) -> bool:
    return bool(target) and str(target) in layout


def _ac_find_closest_layout_key(layout: dict,
                                target_box: List[int],
                                allow_system: bool = False) -> Tuple[Optional[str], float, float]:
    best_key = None
    best_iou = 0.0
    best_distance = float("inf")
    for key, bbox in layout.items():
        if not allow_system and key in ("back", "home"):
            continue
        iou = _ac_bbox_iou(target_box, bbox)
        distance = _ac_bbox_center_distance(target_box, bbox)
        if iou > best_iou or (iou == best_iou and distance < best_distance):
            best_key = key
            best_iou = iou
            best_distance = distance
    return best_key, best_iou, best_distance


def _ac_resolve_tap_target(step: dict,
                           layout: dict,
                           orig_size: Tuple[int, int]) -> Optional[str]:
    point = _ac_scale_step_coord_to_canvas(step, step.get("touch_coord") or [], orig_size)
    if point == [0, 0]:
        return None
    point_tuple = (point[0], point[1])
    for key, bbox in layout.items():
        if _ac_bbox_contains_point(bbox, point_tuple):
            return key
    return None


def _ac_find_action_target(step: dict, layout: dict,
                           orig_size: Tuple[int, int]) -> str:
    if not layout:
        return "unknown"
    action = _ac_normalize_raw_action_name(step.get("action", "")).upper()
    info = str(step.get("info", ""))
    instruction = _ac_step_text(step)
    if "KEY_HOME" in info or "home screen" in instruction:
        return "home"
    if "go back" in instruction or instruction.startswith("back ") or info == "BACK":
        return "back"

    touch_point = _ac_scale_step_coord_to_canvas(step, step.get("touch_coord") or [], orig_size)
    lift_point = _ac_scale_step_coord_to_canvas(step, step.get("lift_coord") or [], orig_size)

    if action in ("TAP", "CLICK") and touch_point != [0, 0]:
        best_key, best_iou, best_distance = _ac_find_closest_layout_key(
            layout, _ac_point_box((touch_point[0], touch_point[1])), allow_system=True
        )
        if best_key is not None and (best_iou > 0 or best_distance <= 48):
            return best_key

    if action in ("TYPE", "TEXT"):
        for preferred in ("search_bar", "search", "input", "text_field", "keyboard"):
            if preferred in layout:
                return preferred

    if action == "PRESS_ENTER":
        for preferred in ("keyboard", "search_bar", "input", "text_field"):
            if preferred in layout:
                return preferred

    if action in ("SWIPE", "SCROLL"):
        if touch_point != [0, 0] and lift_point != [0, 0]:
            start_box = _ac_point_box((touch_point[0], touch_point[1]), radius=18)
            end_box = _ac_point_box((lift_point[0], lift_point[1]), radius=18)
            best_start, start_iou, start_distance = _ac_find_closest_layout_key(layout, start_box, allow_system=False)
            best_end, end_iou, end_distance = _ac_find_closest_layout_key(layout, end_box, allow_system=False)
            if best_start is not None and (start_iou > 0 or start_distance <= 72):
                return best_start
            if best_end is not None and (end_iou > 0 or end_distance <= 72):
                return best_end
        direction = _ac_infer_gesture_direction(step, action, orig_size)
        if direction:
            return f"swipe_{direction}"

    if action == "PRESS_BACK":
        return "back"
    if action == "PRESS_HOME":
        return "home"
    if action == "TASK_COMPLETE":
        return "complete"
    if action == "TASK_IMPOSSIBLE":
        return "impossible"

    best_score = 0.0
    best_key = next(iter(layout.keys()), "unknown")
    for key in layout:
        key_lower = key.lower().replace("_", " ")
        score = SequenceMatcher(None, instruction, key_lower).ratio()
        if key_lower and key_lower in instruction:
            score += 0.3
        if score > best_score:
            best_score = score
            best_key = key
    return best_key


def _ac_resolve_transition(step: dict,
                           layout: dict,
                           orig_size: Tuple[int, int],
                           target_page: str) -> dict:
    raw_action = _ac_normalize_raw_action_name(step.get("action", ""))
    resolved_target = _ac_find_action_target(step, layout, orig_size)
    strict_tap_target = _ac_resolve_tap_target(step, layout, orig_size) if raw_action.upper() in ("TAP", "CLICK") else None
    canvas_action_bbox = _ac_scale_action_bbox_to_canvas(step, raw_action, orig_size)
    canvas_action_point = _ac_scale_action_point_to_canvas(step, raw_action, orig_size)
    canvas_lift_coord = _ac_scale_step_coord_to_canvas(step, step.get("lift_coord") or [], orig_size)
    gesture_direction = _ac_infer_gesture_direction(step, raw_action, orig_size)

    transition = {
        "raw_action": raw_action,
        "action": resolved_target,
        "target_page": target_page,
        "canvas_action_bbox": canvas_action_bbox,
        "canvas_action_point": canvas_action_point,
        "canvas_lift_coord": canvas_lift_coord,
        "icon_bbox": layout.get(resolved_target, [0, 0, 0, 0]),
        "type_text": str(step.get("type_text", "")),
        "gesture_direction": gesture_direction,
    }

    raw_upper = raw_action.upper()
    if raw_upper in ("TAP", "CLICK"):
        if strict_tap_target is not None:
            transition["action"] = strict_tap_target
            transition["icon_bbox"] = layout.get(strict_tap_target, [0, 0, 0, 0])
        else:
            transition["action"] = "tap"
            transition["icon_bbox"] = canvas_action_bbox
    elif raw_upper in ("TYPE", "TEXT"):
        transition["action"] = resolved_target if _ac_is_layout_target(resolved_target, layout) else "type"
        transition["icon_bbox"] = [0, 0, 0, 0]
    elif raw_upper in ("SWIPE", "SCROLL"):
        transition["action"] = resolved_target if _ac_is_layout_target(resolved_target, layout) else "swipe"
        transition["icon_bbox"] = canvas_action_bbox
    elif raw_upper == "PRESS_ENTER":
        transition["action"] = resolved_target if _ac_is_layout_target(resolved_target, layout) else "press_enter"
        transition["icon_bbox"] = [0, 0, 0, 0]
    elif raw_upper == "PRESS_BACK":
        transition["action"] = "back"
        transition["icon_bbox"] = layout.get("back", _AC_GELAB_BACK_BBOX)
    elif raw_upper == "PRESS_HOME":
        transition["action"] = "home"
        transition["icon_bbox"] = layout.get("home", _AC_GELAB_HOME_BBOX)
    elif raw_upper == "TASK_COMPLETE":
        transition["action"] = "complete"
        transition["icon_bbox"] = [0, 0, 0, 0]
    elif raw_upper == "TASK_IMPOSSIBLE":
        transition["action"] = "impossible"
        transition["icon_bbox"] = [0, 0, 0, 0]
    return transition


def _ac_build_system_transition(raw_action: str,
                                action: str,
                                target_page: str,
                                icon_bbox: List[int]) -> dict:
    return {
        "raw_action": raw_action,
        "action": action,
        "target_page": target_page,
        "canvas_action_bbox": [0, 0, 0, 0],
        "canvas_action_point": [0, 0],
        "canvas_lift_coord": [0, 0],
        "icon_bbox": icon_bbox,
        "type_text": "",
        "gesture_direction": "",
    }


def _ac_bbox_center_point(bbox: List[int]) -> List[int]:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return [0, 0]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    if x1 == x2 == y1 == y2 == 0:
        return [0, 0]
    return [int(round((x1 + x2) / 2.0)), int(round((y1 + y2) / 2.0))]


def _ac_is_valid_point(point: List[int]) -> bool:
    return isinstance(point, (list, tuple)) and len(point) == 2 and not (int(point[0]) == 0 and int(point[1]) == 0)


def _ac_is_valid_bbox(bbox: List[int]) -> bool:
    return isinstance(bbox, (list, tuple)) and len(bbox) == 4 and any(int(v) != 0 for v in bbox)


def _ac_stored_transition_action_coord(transition: dict) -> List[int]:
    raw_action = _ac_normalize_raw_action_name(transition.get("raw_action", "")).upper()
    canvas_action_point = _ac_safe_coord_pair(transition.get("canvas_action_point") or [])
    canvas_action_bbox = transition.get("canvas_action_bbox") or [0, 0, 0, 0]
    icon_bbox = transition.get("icon_bbox") or [0, 0, 0, 0]
    if raw_action in ("TYPE", "TEXT", "PRESS_ENTER", "TASK_COMPLETE", "TASK_IMPOSSIBLE"):
        return [0, 0]
    if raw_action in ("SWIPE", "SCROLL", "TAP", "CLICK"):
        if _ac_is_valid_point(canvas_action_point):
            return [int(canvas_action_point[0]), int(canvas_action_point[1])]
        if _ac_is_valid_bbox(canvas_action_bbox):
            return _ac_bbox_center_point(canvas_action_bbox)
    if raw_action in ("PRESS_BACK", "PRESS_HOME") and _ac_is_valid_bbox(icon_bbox):
        return _ac_bbox_center_point(icon_bbox)
    if _ac_is_valid_bbox(icon_bbox):
        return _ac_bbox_center_point(icon_bbox)
    if _ac_is_valid_point(canvas_action_point):
        return [int(canvas_action_point[0]), int(canvas_action_point[1])]
    if _ac_is_valid_bbox(canvas_action_bbox):
        return _ac_bbox_center_point(canvas_action_bbox)
    return [0, 0]


def _ac_stored_transition_lift_coord(transition: dict) -> List[int]:
    raw_action = _ac_normalize_raw_action_name(transition.get("raw_action", "")).upper()
    if raw_action not in ("SWIPE", "SCROLL"):
        return [0, 0]
    canvas_lift_coord = _ac_safe_coord_pair(transition.get("canvas_lift_coord") or [])
    if _ac_is_valid_point(canvas_lift_coord):
        return [int(canvas_lift_coord[0]), int(canvas_lift_coord[1])]
    return [0, 0]


def _ac_debug_action_name(transition: dict) -> str:
    raw_action = _ac_normalize_raw_action_name(transition.get("raw_action", "")).upper()
    if raw_action == "TASK_COMPLETE":
        return "COMPLETE"
    if raw_action == "TASK_IMPOSSIBLE":
        return "IMPOSSIBLE"
    if raw_action:
        return raw_action.replace("_", " ")
    return str(transition.get("action", "") or "").strip().upper().replace("_", " ")


def _ac_debug_bbox_for_transition(transition: dict, action_coord: List[int]) -> List[int]:
    raw_action = _ac_normalize_raw_action_name(transition.get("raw_action", "")).upper()
    canvas_action_bbox = transition.get("canvas_action_bbox") or [0, 0, 0, 0]
    icon_bbox = transition.get("icon_bbox") or [0, 0, 0, 0]
    if raw_action in ("TYPE", "TEXT", "PRESS_ENTER", "TASK_COMPLETE", "TASK_IMPOSSIBLE"):
        return [0, 0, 0, 0]
    if raw_action in ("SWIPE", "SCROLL") and _ac_is_valid_bbox(canvas_action_bbox):
        return [int(v) for v in canvas_action_bbox]
    if _ac_is_valid_bbox(icon_bbox):
        return [int(v) for v in icon_bbox]
    if _ac_is_valid_bbox(canvas_action_bbox):
        return [int(v) for v in canvas_action_bbox]
    if _ac_is_valid_point(action_coord):
        px, py = int(action_coord[0]), int(action_coord[1])
        return [px - 8, py - 8, px + 8, py + 8]
    return [0, 0, 0, 0]


def _ac_debug_transition_label(idx: int, transition: dict) -> str:
    action = _ac_debug_action_name(transition)
    raw_action = _ac_normalize_raw_action_name(transition.get("raw_action", "")).upper()
    if raw_action in ("TYPE", "TEXT", "PRESS_ENTER"):
        type_text = str(transition.get("type_text", "") or "").strip()
        if type_text:
            shortened = type_text if len(type_text) <= 24 else f"{type_text[:21]}..."
            return f"{idx}:{action} {shortened}"
    return f"{idx}:{action}"


def _ac_should_draw_non_spatial_debug_label(transition: dict) -> bool:
    raw_action = _ac_normalize_raw_action_name(transition.get("raw_action", "")).upper()
    return raw_action in ("TYPE", "TEXT", "PRESS_ENTER", "TASK_COMPLETE", "TASK_IMPOSSIBLE")


def _ac_save_action_debug_overlay(page_image_path: str,
                                  output_path: str,
                                  transitions: List[dict]):
    if not os.path.exists(page_image_path):
        return
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with Image.open(page_image_path) as img_handle:
        image = img_handle.convert("RGB")
    draw = ImageDraw.Draw(image)
    font = _ac_try_load_font(12)
    palette = [
        ((230, 57, 70), (255, 230, 233)),
        ((29, 78, 216), (227, 238, 255)),
        ((22, 163, 74), (229, 255, 237)),
        ((217, 119, 6), (255, 245, 224)),
        ((126, 34, 206), (243, 232, 255)),
    ]
    non_spatial_label_y = _AC_NAV_STRIP_H + 8
    for idx, transition in enumerate(transitions or []):
        edge_color, label_bg = palette[idx % len(palette)]
        action_coord = _ac_stored_transition_action_coord(transition)
        debug_bbox = _ac_debug_bbox_for_transition(transition, action_coord)
        label = _ac_debug_transition_label(idx, transition)
        if _ac_is_valid_bbox(debug_bbox):
            draw.rectangle(debug_bbox, outline=edge_color, width=3)
        if _ac_is_valid_point(action_coord):
            px, py = int(action_coord[0]), int(action_coord[1])
            draw.rectangle([px - 4, py - 4, px + 4, py + 4], fill=edge_color, outline=(255, 255, 255), width=1)
            text_bbox = draw.textbbox((0, 0), label, font=font)
            text_w = text_bbox[2] - text_bbox[0]
            text_h = text_bbox[3] - text_bbox[1]
            lx = min(max(4, px + 8), max(4, image.size[0] - text_w - 6))
            ly = min(max(4, py - text_h - 8), max(4, image.size[1] - text_h - 6))
            draw.rectangle([lx - 2, ly - 2, lx + text_w + 2, ly + text_h + 2], fill=label_bg, outline=edge_color, width=1)
            draw.text((lx, ly), label, fill=edge_color, font=font)
        elif _ac_should_draw_non_spatial_debug_label(transition):
            text_bbox = draw.textbbox((0, 0), label, font=font)
            text_w = text_bbox[2] - text_bbox[0]
            text_h = text_bbox[3] - text_bbox[1]
            lx = 6
            ly = min(non_spatial_label_y, max(6, image.size[1] - text_h - 6))
            draw.rectangle([lx - 2, ly - 2, lx + text_w + 2, ly + text_h + 2], fill=label_bg, outline=edge_color, width=1)
            draw.text((lx, ly), label, fill=edge_color, font=font)
            non_spatial_label_y = ly + text_h + 8
    image.save(output_path)


class _LocalActionComposeNamespace:
    GELAB_BACK_BBOX = list(_AC_GELAB_BACK_BBOX)
    GELAB_HOME_BBOX = list(_AC_GELAB_HOME_BBOX)
    load_api_client = staticmethod(_ac_load_api_client)
    load_detection_models = staticmethod(_ac_load_detection_models)
    detect_and_crop = staticmethod(_ac_detect_and_crop)
    _prioritize_element_anno_bboxes = staticmethod(_ac_prioritize_element_anno_bboxes)
    _persist_extracted_assets = staticmethod(_ac_persist_extracted_assets)
    _build_step_context = staticmethod(_ac_build_step_context)
    _resolve_step_screenshot = staticmethod(_ac_resolve_step_screenshot)
    compose_page = staticmethod(_ac_compose_page)
    _ensure_system_nav_controls = staticmethod(_ac_ensure_system_nav_controls)
    _save_page_code = staticmethod(_ac_save_page_code)
    _ensure_system_layout = staticmethod(_ac_ensure_system_layout)
    _resolve_transition = staticmethod(_ac_resolve_transition)
    _build_system_transition = staticmethod(_ac_build_system_transition)
    _save_action_debug_overlay = staticmethod(_ac_save_action_debug_overlay)
    _stored_transition_action_coord = staticmethod(_ac_stored_transition_action_coord)
    _stored_transition_lift_coord = staticmethod(_ac_stored_transition_lift_coord)


def _ensure_compose_modules() -> None:
    global action_compose
    if action_compose is None:
        action_compose = _LocalActionComposeNamespace


def _tokenize_words(text: str) -> List[str]:
    return [token for token in re.split(r"[^0-9a-z]+", str(text or "").lower()) if token]


def _build_match_tokens(label: str, asset: str) -> set[str]:
    tokens: set[str] = set()
    for candidate in (label, asset, str(asset).replace("_real", "")):
        compact = _normalize_compact(candidate)
        if len(compact) >= 3:
            tokens.add(compact)
        for token in _tokenize_words(candidate):
            if len(token) >= 3:
                tokens.add(token)
    return tokens


def _matches_text(text: str, tokens: Iterable[str]) -> bool:
    text_lower = str(text or "").lower()
    text_compact = _normalize_compact(text_lower)
    for token in tokens:
        if token in text_lower or token in text_compact:
            return True
    return False


def _typed_layout_to_plain(layout: Dict[str, Any]) -> Dict[str, List[int]]:
    plain: Dict[str, List[int]] = {}
    for key, value in (layout or {}).items():
        if isinstance(value, dict) and "bbox" in value:
            plain[key] = [int(v) for v in value["bbox"]]
        elif isinstance(value, (list, tuple)) and len(value) == 4:
            plain[key] = [int(v) for v in value]
    return plain


def _resolve_spec_label(spec: dict) -> str:
    label = str(spec.get("label") or "").strip()
    if label:
        return label
    asset = str(spec.get("asset") or "").strip().lower()
    if asset in launcher_mock.REAL_ICON_LIBRARY:
        return str(launcher_mock.REAL_ICON_LIBRARY[asset]["label"])
    return asset or "app"


def _find_layout_key(layout: Dict[str, List[int]], preferred_label: str, asset: str) -> Optional[str]:
    candidates = [
        str(preferred_label or "").strip(),
        launcher_mock._safe_key(str(preferred_label or "").strip()),
        str(asset or "").strip(),
        launcher_mock._safe_key(str(asset or "").strip()),
    ]
    lowered = {key.lower(): key for key in layout}
    for candidate in candidates:
        if not candidate:
            continue
        if candidate in layout:
            return candidate
        lowered_match = lowered.get(candidate.lower())
        if lowered_match:
            return lowered_match
    return None


def _extract_drawer_apps(layout_config: dict, drawer_layout: Dict[str, List[int]]) -> List[DrawerAppSpec]:
    apps: List[DrawerAppSpec] = []
    drawer_icons = layout_config.get("pages", {}).get("app_drawer", {}).get("icons", [])
    for spec in drawer_icons:
        if not isinstance(spec, dict):
            continue
        label = _resolve_spec_label(spec)
        asset = str(spec.get("asset") or label)
        layout_key = _find_layout_key(drawer_layout, label, asset)
        if layout_key is None:
            continue
        slug = _normalize_compact(label or asset)
        if not slug:
            continue
        apps.append(DrawerAppSpec(
            label=label,
            asset=asset,
            slug=slug,
            layout_key=layout_key,
            bbox=[int(v) for v in drawer_layout[layout_key]],
            match_tokens=_build_match_tokens(label, asset),
        ))
    return apps


def _find_first_matching_step_index(steps: List[dict], tokens: set[str]) -> Tuple[Optional[int], str]:
    for idx, step in enumerate(steps):
        package_name = str(step.get("package_name", "") or "")
        if _matches_text(package_name, tokens):
            return idx, package_name
    return None, ""


def _find_matching_segment_end(steps: List[dict], start_idx: int, tokens: set[str]) -> int:
    end_idx = start_idx + 1
    while end_idx < len(steps):
        package_name = str(steps[end_idx].get("package_name", "") or "")
        if not _matches_text(package_name, tokens):
            break
        end_idx += 1
    return end_idx


def _trajectory_ends_with_task_complete(steps: List[dict]) -> bool:
    actions = [
        str(step.get("action", "") or "").strip().upper()
        for step in steps
        if str(step.get("action", "") or "").strip()
    ]
    return bool(actions) and actions[-1] == "TASK_COMPLETE"


def _scan_matching_annotations(apps: List[DrawerAppSpec],
                               annotations_dir: str,
                               include_post_app_steps: bool) -> Dict[str, List[MatchedTrajectory]]:
    matches: Dict[str, List[MatchedTrajectory]] = defaultdict(list)
    annotations_path = Path(annotations_dir)
    for annot_path in sorted(annotations_path.glob("*.json")):
        try:
            trajectory = json.loads(annot_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        instruction = str(trajectory.get("instruction", "") or "")
        steps = trajectory.get("steps", []) or []
        if not _trajectory_ends_with_task_complete(steps):
            continue
        packages = [str(step.get("package_name", "") or "") for step in steps]

        for app in apps:
            if not (_matches_text(instruction, app.match_tokens) or any(_matches_text(pkg, app.match_tokens) for pkg in packages)):
                continue

            start_idx, matched_package = _find_first_matching_step_index(steps, app.match_tokens)
            if start_idx is None:
                continue

            end_idx = len(steps) if include_post_app_steps else _find_matching_segment_end(steps, start_idx, app.match_tokens)
            matches[app.slug].append(MatchedTrajectory(
                app_slug=app.slug,
                app_label=app.label,
                annotation_path=str(annot_path),
                episode_id=str(trajectory.get("episode_id") or annot_path.stem),
                instruction=instruction,
                start_step_idx=int(start_idx),
                end_step_idx=int(end_idx),
                matched_package=matched_package,
                total_steps=len(steps),
            ))

    for app_slug in matches:
        matches[app_slug].sort(key=lambda item: (item.episode_id, item.start_step_idx))
    return matches


def _semantic_action(raw_action: str, fallback: str = "") -> str:
    normalized = str(raw_action or "").strip().upper()
    semantic_map = {
        "TAP": "tap",
        "CLICK": "tap",
        "SWIPE": "swipe",
        "SCROLL": "swipe",
        "TYPE": "type",
        "TEXT": "type",
        "PRESS_ENTER": "press_enter",
        "PRESS_BACK": "press_back",
        "PRESS_HOME": "press_home",
        "TASK_COMPLETE": "task_complete",
        "TASK_IMPOSSIBLE": "task_impossible",
    }
    if normalized in semantic_map:
        return semantic_map[normalized]
    return str(fallback or normalized or "navigate").strip().lower()


def _bbox_center(bbox: List[int]) -> List[int]:
    return [int(round((bbox[0] + bbox[2]) / 2.0)), int(round((bbox[1] + bbox[3]) / 2.0))]


def _valid_point(point: List[int]) -> bool:
    return isinstance(point, (list, tuple)) and len(point) == 2 and not (int(point[0]) == 0 and int(point[1]) == 0)


def _valid_bbox(bbox: List[int]) -> bool:
    return isinstance(bbox, (list, tuple)) and len(bbox) == 4 and any(int(v) != 0 for v in bbox)


def _build_tap_transition(target_page: str,
                          target_element: str,
                          icon_bbox: List[int],
                          raw_action: str = "TAP") -> dict:
    return {
        "raw_action": raw_action,
        "action": target_element,
        "action_kind": _semantic_action(raw_action),
        "target_page": target_page,
        "canvas_action_bbox": [int(v) for v in icon_bbox],
        "canvas_action_point": _bbox_center(icon_bbox),
        "canvas_lift_coord": [0, 0],
        "icon_bbox": [int(v) for v in icon_bbox],
        "type_text": "",
        "gesture_direction": "",
    }


def _build_swipe_transition(target_page: str,
                            action_coord: List[int],
                            lift_coord: List[int],
                            direction: str,
                            icon_bbox: Optional[List[int]] = None) -> dict:
    start = [int(action_coord[0]), int(action_coord[1])]
    end = [int(lift_coord[0]), int(lift_coord[1])]
    bbox = icon_bbox or [
        min(start[0], end[0]),
        min(start[1], end[1]),
        max(start[0], end[0]),
        max(start[1], end[1]),
    ]
    return {
        "raw_action": "SWIPE",
        "action": "swipe",
        "action_kind": "swipe",
        "target_page": target_page,
        "canvas_action_bbox": [int(v) for v in bbox],
        "canvas_action_point": start,
        "canvas_lift_coord": end,
        "icon_bbox": [int(v) for v in bbox],
        "type_text": "",
        "gesture_direction": direction,
    }


def _serialize_layout(layout: Dict[str, List[int]]) -> Dict[str, dict]:
    return {
        key: {
            "bbox": [int(v) for v in bbox],
            "type": "system" if key in ("back", "home") else "normal",
        }
        for key, bbox in (layout or {}).items()
    }


def _serialize_transition(transition: dict) -> dict:
    _ensure_compose_modules()
    raw_action = str(transition.get("raw_action", "") or "").strip().upper()
    semantic_action = _semantic_action(raw_action, transition.get("action", ""))
    item = {
        "action": semantic_action,
        "target_page": transition.get("target_page", ""),
        "raw_action": raw_action,
    }

    target_element = str(transition.get("action", "") or "").strip()
    if target_element and target_element.lower() not in {
        semantic_action,
        "tap",
        "swipe",
        "type",
        "press_enter",
        "press_back",
        "press_home",
        "task_complete",
        "task_impossible",
    }:
        item["target_element"] = target_element

    action_coord = action_compose._stored_transition_action_coord(transition)
    lift_coord = action_compose._stored_transition_lift_coord(transition)
    if _valid_point(action_coord):
        item["action_coord"] = [int(action_coord[0]), int(action_coord[1])]
    if _valid_point(lift_coord):
        item["lift_coord"] = [int(lift_coord[0]), int(lift_coord[1])]

    icon_bbox = transition.get("icon_bbox") or [0, 0, 0, 0]
    if raw_action in ("SWIPE", "SCROLL"):
        icon_bbox = transition.get("canvas_action_bbox") or icon_bbox
    if _valid_bbox(icon_bbox):
        item["icon_bbox"] = [int(v) for v in icon_bbox]

    type_text = str(transition.get("type_text", "") or "").strip()
    if type_text:
        item["type_text"] = type_text
    gesture_direction = str(transition.get("gesture_direction", "") or "").strip()
    if gesture_direction:
        item["gesture_direction"] = gesture_direction
    source_trace_page = str(transition.get("source_trace_page", "") or "").strip()
    if source_trace_page:
        item["source_trace_page"] = source_trace_page
    source_trajectory_id = str(transition.get("source_trajectory_id", "") or "").strip()
    if source_trajectory_id:
        item["source_trajectory_id"] = source_trajectory_id
    source_step_indices = transition.get("source_step_indices") or []
    if isinstance(source_step_indices, (list, tuple)) and source_step_indices:
        item["source_step_indices"] = [int(v) for v in source_step_indices]
    return item


def _transition_signature_payload_from_serialized(serialized: dict,
                                                  include_coordinates: bool = False) -> dict:
    payload = {
        "action": str(serialized.get("action", "") or "").strip().lower(),
        "target_page": str(serialized.get("target_page", "") or "").strip(),
        "raw_action": str(serialized.get("raw_action", "") or "").strip().upper(),
    }

    target_element = str(serialized.get("target_element", "") or "").strip()
    if target_element:
        payload["target_element"] = target_element

    icon_bbox = serialized.get("icon_bbox") or [0, 0, 0, 0]
    if _valid_bbox(icon_bbox):
        payload["icon_bbox"] = [int(v) for v in icon_bbox]

    type_text = str(serialized.get("type_text", "") or "").strip()
    if type_text:
        payload["type_text"] = type_text

    gesture_direction = str(serialized.get("gesture_direction", "") or "").strip()
    if gesture_direction:
        payload["gesture_direction"] = gesture_direction

    if include_coordinates:
        action_coord = serialized.get("action_coord") or [0, 0]
        if _valid_point(action_coord):
            payload["action_coord"] = [int(action_coord[0]), int(action_coord[1])]
        lift_coord = serialized.get("lift_coord") or [0, 0]
        if _valid_point(lift_coord):
            payload["lift_coord"] = [int(lift_coord[0]), int(lift_coord[1])]
    return payload


def _transition_action_location_signature_from_serialized(serialized: dict) -> Tuple[Any, ...]:
    action_coord = serialized.get("action_coord") or [0, 0]
    lift_coord = serialized.get("lift_coord") or [0, 0]
    icon_bbox = serialized.get("icon_bbox") or [0, 0, 0, 0]
    raw_action = str(serialized.get("raw_action", "") or "").strip().upper()

    if _valid_point(action_coord):
        signature: List[Any] = [
            "point",
            _quantize(int(action_coord[0]), bucket=32),
            _quantize(int(action_coord[1]), bucket=32),
        ]
        if _valid_point(lift_coord):
            signature.extend([
                _quantize(int(lift_coord[0]), bucket=32),
                _quantize(int(lift_coord[1]), bucket=32),
            ])
        elif raw_action in ("SWIPE", "SCROLL"):
            signature.append(str(serialized.get("gesture_direction", "") or "").strip().lower())
        return tuple(signature)

    if _valid_bbox(icon_bbox):
        x1, y1, x2, y2 = [int(v) for v in icon_bbox]
        return (
            "bbox",
            _quantize(x1, bucket=32),
            _quantize(y1, bucket=32),
            _quantize(x2 - x1, bucket=32),
            _quantize(y2 - y1, bucket=32),
        )

    target_element = str(serialized.get("target_element", "") or "").strip().lower()
    type_text = str(serialized.get("type_text", "") or "").strip().lower()
    gesture_direction = str(serialized.get("gesture_direction", "") or "").strip().lower()
    return ("fallback", raw_action, target_element, type_text, gesture_direction)


def _dedupe_serialized_transitions(transitions: List[dict]) -> List[dict]:
    deduped: List[dict] = []
    seen_signatures: set[str] = set()
    seen_system_actions: set[str] = set()

    for transition in transitions:
        action_label = _semantic_action(
            str(transition.get("raw_action", "") or ""),
            str(transition.get("action", "") or ""),
        )
        if action_label in ("press_back", "press_home"):
            if action_label in seen_system_actions:
                continue
            seen_system_actions.add(action_label)
            deduped.append(transition)
            continue

        signature = json.dumps(
            _transition_signature_payload_from_serialized(transition, include_coordinates=False),
            sort_keys=True,
            separators=(",", ":"),
        )
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        deduped.append(transition)
    return deduped


def _build_ui_layer(pages: Dict[str, dict], root_page_id: str) -> dict:
    visited: set[str] = set()

    def build_node(page_id: str) -> Optional[dict]:
        if page_id in visited or page_id not in pages:
            return None
        visited.add(page_id)
        page = pages[page_id]
        serialized_transitions = _dedupe_serialized_transitions(
            [_serialize_transition(t) for t in page.get("transitions", [])]
        )
        non_system = [t for t in serialized_transitions if t.get("action") not in ("press_back", "press_home")]

        subnodes = []
        for transition in non_system:
            child_id = transition.get("target_page", "")
            child = build_node(child_id)
            if child is not None:
                subnodes.append(child)
        return {
            "page_id": page_id,
            "page_name": page.get("page_name", page_id),
            "application_id": page.get("application_id", ""),
            "application_name": page.get("application_name", ""),
            "image": page.get("image", f"{page_id}.png"),
            "depth": int(page.get("depth", 0)),
            "layout": _serialize_layout(page.get("layout", {})),
            "merged_from_page_ids": list(page.get("merged_from_page_ids", [page_id])),
            "trajectory_ids": list(page.get("trajectory_ids", [])),
            "trajectory_ids_full": list(page.get("trajectory_ids_full", [])),
            "trace_steps": list(page.get("trace_steps", [])),
            "page_summary": deepcopy(page.get("page_summary", {})),
            "transitions": non_system,
            "subnodes": subnodes,
        }

    root = build_node(root_page_id)
    for page_id in sorted(pages):
        if page_id in visited:
            continue
        child = build_node(page_id)
        if child is not None and root is not None:
            root["subnodes"].append(child)
    return {
        "root": root,
        "metadata": {
            "source": "mock_unified_app_graph",
            "canvas_size": list(OUTPUT_CANVAS_SIZE),
            "total_pages": len(pages),
        },
    }


def _save_ui_structure(output_dir: Path,
                       pages: Dict[str, dict],
                       root_page_id: str,
                       metadata: dict) -> None:
    serialized_pages = {}
    for page_id, page in sorted(pages.items(), key=lambda item: (int(item[1].get("depth", 0)), str(item[0]))):
        serialized_transitions = _dedupe_serialized_transitions(
            [_serialize_transition(t) for t in page.get("transitions", [])]
        )
        serialized_pages[page_id] = {
            "page_id": page_id,
            "page_name": page.get("page_name", page_id),
            "application_id": page.get("application_id", ""),
            "application_name": page.get("application_name", ""),
            "image": page.get("image", f"{page_id}.png"),
            "depth": int(page.get("depth", 0)),
            "layout": _serialize_layout(page.get("layout", {})),
            "transitions": serialized_transitions,
            "merged_from_page_ids": list(page.get("merged_from_page_ids", [page_id])),
            "trajectory_ids": list(page.get("trajectory_ids", [])),
            "trajectory_ids_full": list(page.get("trajectory_ids_full", [])),
            "trace_steps": list(page.get("trace_steps", [])),
            "page_summary": deepcopy(page.get("page_summary", {})),
        }

    ui_structure = {
        "pages": serialized_pages,
        "metadata": {
            **metadata,
            "source": "mock_unified_app_graph",
            "canvas_size": list(OUTPUT_CANVAS_SIZE),
            "root_page_id": root_page_id,
            "total_pages": len(serialized_pages),
        },
    }
    ui_layer = _build_ui_layer(pages, root_page_id)

    (output_dir / "ui_structure.json").write_text(json.dumps(ui_structure, indent=2), encoding="utf-8")
    (output_dir / "ui_structure_layer.json").write_text(json.dumps(ui_layer, indent=2), encoding="utf-8")


def _make_page_summary(page_name: str,
                       application_id: str,
                       application_name: str,
                       layout: Dict[str, List[int]],
                       page_family: str) -> dict:
    icons = sorted([key for key in layout if key not in ("back", "home")])
    return {
        "page_family": page_family,
        "semantic_page_name": page_name,
        "logical_page_name": page_name,
        "application_id": application_id,
        "application_name": application_name,
        "icons": icons,
        "additional_icons": [],
        "components": icons,
        "additional_components": [],
    }


def _launcher_transition_from_mock(swipe_transition: dict, target_page: str) -> dict:
    return {
        "raw_action": "SWIPE",
        "action": "swipe",
        "action_kind": "swipe",
        "target_page": target_page,
        "canvas_action_bbox": [int(v) for v in swipe_transition.get("icon_bbox", [0, 0, 0, 0])],
        "canvas_action_point": [int(v) for v in swipe_transition.get("action_coord", [0, 0])],
        "canvas_lift_coord": [int(v) for v in swipe_transition.get("lift_coord", [0, 0])],
        "icon_bbox": [int(v) for v in swipe_transition.get("icon_bbox", [0, 0, 0, 0])],
        "type_text": "",
        "gesture_direction": str(swipe_transition.get("gesture_direction", "") or ""),
    }


def _save_mock_pages(output_dir: Path,
                     layout_config: dict) -> Tuple[Dict[str, List[int]], Dict[str, List[int]], Dict[str, Any]]:
    render_state: Dict[str, Any] = {
        "rng": launcher_mock._build_rng(layout_config),
        "used_random_assets": set(),
        "resolved_icons": [],
    }
    home_img, home_layout_typed, swipe_transition = launcher_mock._draw_home_page(layout_config, render_state)
    drawer_img, drawer_layout_typed = launcher_mock._draw_app_drawer_page(layout_config, render_state)

    pages_dir = output_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)
    home_img.save(pages_dir / f"{HOME_PAGE_ID}.png")
    drawer_img.save(pages_dir / f"{DRAWER_PAGE_ID}.png")

    return (
        _typed_layout_to_plain(home_layout_typed),
        _typed_layout_to_plain(drawer_layout_typed),
        {
            "swipe_transition": swipe_transition,
            "resolved_icons": render_state["resolved_icons"],
        },
    )


def _segment_steps(trajectory: dict,
                   start_step_idx: int,
                   end_step_idx: int) -> dict:
    segment = dict(trajectory)
    steps = trajectory.get("steps", []) or []
    selected_steps = []
    for global_idx in range(start_step_idx, end_step_idx):
        step = dict(steps[global_idx])
        step["source_step_index"] = global_idx + 1
        selected_steps.append(step)
    segment["steps"] = selected_steps
    segment["segment_start_step_index"] = start_step_idx + 1
    segment["segment_end_step_index"] = end_step_idx
    return segment


def _step_context_for_segment(segment_trajectory: dict, local_idx: int) -> dict:
    context = action_compose._build_step_context(segment_trajectory, local_idx)
    context["task"] = segment_trajectory.get("instruction", "")
    context["task_instruction"] = segment_trajectory.get("instruction", "")
    context["source_episode_id"] = segment_trajectory.get("episode_id", "")
    context["global_step_index"] = segment_trajectory["steps"][local_idx].get("source_step_index", local_idx + 1)
    return context


def _compose_segment_pages(match: MatchedTrajectory,
                           app: DrawerAppSpec,
                           app_root_page_id: str,
                           home_page_id: str,
                           args,
                           client,
                           model_name: str,
                           yolo_model,
                           ocr_reader,
                           output_dir: Path) -> Tuple[List[dict], List[dict], List[dict]]:
    with open(match.annotation_path, "r", encoding="utf-8") as f:
        trajectory = json.load(f)

    segment_trajectory = _segment_steps(trajectory, match.start_step_idx, match.end_step_idx)
    steps = segment_trajectory.get("steps", []) or []
    if not steps:
        return [], [], []

    print(
        f"    composing trajectory_id={_full_trajectory_id(match)} "
        f"episode={match.episode_id} "
        f"steps={match.start_step_idx + 1}-{match.end_step_idx} "
        f"({len(steps)} pages)"
    )

    pages_dir = output_dir / "pages"
    code_dir = output_dir / "generated_code"
    assets_dir = output_dir / "extracted_assets" / app.slug / match.episode_id
    pages_dir.mkdir(parents=True, exist_ok=True)
    code_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)

    page_rows: List[dict] = []
    manifest_rows: List[dict] = []
    matched_rows: List[dict] = []

    for local_idx, step in enumerate(steps):
        screenshot_name, screenshot_path = action_compose._resolve_step_screenshot(
            step, args.screenshots_dir, match.episode_id, local_idx
        )
        if not os.path.exists(screenshot_path):
            print(
                f"      [{local_idx + 1}/{len(steps)}] "
                f"missing screenshot={screenshot_name} "
                f"source_step={step.get('source_step_index', local_idx + 1)}"
            )
            continue

        step_context = _step_context_for_segment(segment_trajectory, local_idx)
        page_id = f"page_{app.slug}_{match.episode_id}_{local_idx + 1:02d}"
        print(
            f"      [{local_idx + 1}/{len(steps)}] "
            f"source_step={step.get('source_step_index', local_idx + 1)} "
            f"screenshot={screenshot_name}"
        )

        elements, orig_size = action_compose.detect_and_crop(screenshot_path, yolo_model, ocr_reader)
        elements, anno_stats = action_compose._prioritize_element_anno_bboxes(
            elements,
            screenshot_path,
            screenshot_name,
            args.element_anno_dir,
        )
        asset_elements = action_compose._persist_extracted_assets(
            elements,
            screenshot_name,
            str(assets_dir),
            step_context,
        )

        page_img, layout, code_artifact = action_compose.compose_page(
            client,
            model_name,
            asset_elements,
            orig_size,
            screenshot_path,
            step_context,
        )
        page_img, layout = action_compose._ensure_system_nav_controls(page_img, layout)
        page_img.save(pages_dir / f"{page_id}.png")
        action_compose._save_page_code(
            str(code_dir),
            page_id,
            screenshot_name,
            step_context,
            code_artifact,
        )
        print(
            f"        -> page_id={page_id} "
            f"detected={len(elements)} "
            f"anno_loaded={anno_stats.get('loaded', 0)} "
            f"layout={len(layout)}"
        )

        layout = action_compose._ensure_system_layout(layout)
        page_rows.append({
            "page_id": page_id,
            "image": f"{page_id}.png",
            "depth": 3 + local_idx,
            "layout": layout,
            "orig_size": tuple(orig_size),
            "step": step,
            "step_context": step_context,
            "episode_id": match.episode_id,
            "page_name": f"{app.label} {match.episode_id[:8]} step {local_idx + 1}",
            "application_id": app.slug,
            "application_name": app.label,
            "trajectory_ids": [match.episode_id],
            "trajectory_ids_full": [_full_trajectory_id(match)],
            "trace_steps": [step.get("source_step_index", local_idx + 1)],
            "anno_stats": anno_stats,
        })

        for elem in asset_elements:
            manifest_rows.append({
                "page_id": page_id,
                "episode_id": match.episode_id,
                "app_label": app.label,
                "step_index": step.get("source_step_index", local_idx + 1),
                "screenshot": screenshot_name,
                "type": elem.get("type"),
                "label": elem.get("label"),
                "bbox": elem.get("bbox"),
                "asset_path": elem.get("asset_path"),
                "asset_source": elem.get("asset_source"),
            })

        matched_rows.append({
            "page_id": page_id,
            "trajectory_id_full": _full_trajectory_id(match),
            "episode_id": match.episode_id,
            "app_label": app.label,
            "instruction": match.instruction,
            "step_index": step.get("source_step_index", local_idx + 1),
            "screenshot": screenshot_name,
            "package_name": step.get("package_name", ""),
        })

    if not page_rows:
        return [], manifest_rows, matched_rows

    for idx, page in enumerate(page_rows):
        next_page_id = page_rows[idx + 1]["page_id"] if idx + 1 < len(page_rows) else page["page_id"]
        raw_action = str(page["step"].get("action", "") or "").strip().upper()
        if idx + 1 >= len(page_rows):
            if raw_action == "PRESS_HOME":
                next_page_id = home_page_id
            elif raw_action == "PRESS_BACK":
                next_page_id = app_root_page_id

        transition = action_compose._resolve_transition(
            page["step"],
            page["layout"],
            tuple(page["orig_size"]),
            next_page_id,
        )
        transition["source_trace_page"] = page["page_id"]
        transition["source_trajectory_id"] = match.episode_id
        transition["source_step_indices"] = [page["step"].get("source_step_index", idx + 1)]
        transition["action_kind"] = _semantic_action(transition.get("raw_action", ""), transition.get("action", ""))

        transitions = [transition]
        used_system_targets = {transition.get("action")} if transition.get("action") in ("back", "home") else set()
        back_target = page_rows[idx - 1]["page_id"] if idx > 0 else app_root_page_id
        if "back" not in used_system_targets:
            back_transition = action_compose._build_system_transition(
                raw_action="PRESS_BACK",
                action="back",
                target_page=back_target,
                icon_bbox=page["layout"].get("back", action_compose.GELAB_BACK_BBOX),
            )
            back_transition["action_kind"] = "press_back"
            transitions.append(back_transition)
        if "home" not in used_system_targets:
            home_transition = action_compose._build_system_transition(
                raw_action="PRESS_HOME",
                action="home",
                target_page=home_page_id,
                icon_bbox=page["layout"].get("home", action_compose.GELAB_HOME_BBOX),
            )
            home_transition["action_kind"] = "press_home"
            transitions.append(home_transition)

        page["transitions"] = transitions
        page["page_summary"] = _make_page_summary(
            page["page_name"],
            page["application_id"],
            page["application_name"],
            page["layout"],
            page_family="content_page",
        )

    return page_rows, manifest_rows, matched_rows


def _build_rich_launcher_pages(home_layout: Dict[str, List[int]],
                               drawer_layout: Dict[str, List[int]],
                               swipe_transition: dict,
                               drawer_apps: List[DrawerAppSpec],
                               app_entry_pages: Dict[str, List[str]]) -> Dict[str, dict]:
    pages: Dict[str, dict] = {}

    home_transitions = [
        _launcher_transition_from_mock(swipe_transition, DRAWER_PAGE_ID),
        {
            **action_compose._build_system_transition(
                raw_action="PRESS_HOME",
                action="home",
                target_page=HOME_PAGE_ID,
                icon_bbox=home_layout.get("home", [0, 0, 0, 0]),
            ),
            "action_kind": "press_home",
        },
    ]
    pages[HOME_PAGE_ID] = {
        "page_id": HOME_PAGE_ID,
        "image": f"{HOME_PAGE_ID}.png",
        "depth": 0,
        "layout": home_layout,
        "transitions": home_transitions,
        "page_name": "Home",
        "application_id": "launcher",
        "application_name": "Launcher",
        "trajectory_ids": [],
        "trace_steps": [],
        "page_summary": _make_page_summary("Home", "launcher", "Launcher", home_layout, page_family="home"),
    }

    drawer_transitions: List[dict] = [
        {
            **action_compose._build_system_transition(
                raw_action="PRESS_BACK",
                action="back",
                target_page=HOME_PAGE_ID,
                icon_bbox=drawer_layout.get("back", [0, 0, 0, 0]),
            ),
            "action_kind": "press_back",
        },
        {
            **action_compose._build_system_transition(
                raw_action="PRESS_HOME",
                action="home",
                target_page=HOME_PAGE_ID,
                icon_bbox=drawer_layout.get("home", [0, 0, 0, 0]),
            ),
            "action_kind": "press_home",
        },
    ]
    for app in drawer_apps:
        entry_pages = _ordered_unique(app_entry_pages.get(app.slug, []))
        if not entry_pages:
            continue
        for first_page in entry_pages:
            transition = _build_tap_transition(first_page, app.layout_key, app.bbox, raw_action="TAP")
            drawer_transitions.append(transition)

    pages[DRAWER_PAGE_ID] = {
        "page_id": DRAWER_PAGE_ID,
        "image": f"{DRAWER_PAGE_ID}.png",
        "depth": 1,
        "layout": drawer_layout,
        "transitions": drawer_transitions,
        "page_name": "App Drawer",
        "application_id": "launcher",
        "application_name": "Launcher",
        "trajectory_ids": [],
        "trace_steps": [],
        "page_summary": _make_page_summary("App Drawer", "launcher", "Launcher", drawer_layout, page_family="app_drawer"),
    }
    return pages


def _transition_action_label(transition: dict) -> str:
    return _semantic_action(
        str(transition.get("raw_action", "") or ""),
        str(transition.get("action", "") or ""),
    )


def _is_progressive_back_transition(
    source_page: dict,
    transition: dict,
    pages: Dict[str, dict],
) -> bool:
    if _transition_action_label(transition) != "press_back":
        return False
    target_page = str(transition.get("target_page", "") or "")
    if not target_page or target_page not in pages:
        return False
    source_depth = int(source_page.get("depth", 0))
    target_depth = int(pages[target_page].get("depth", source_depth))
    return target_depth > source_depth


def _non_system_target_pages(page: dict, pages: Dict[str, dict]) -> List[str]:
    targets: List[str] = []
    for transition in page.get("transitions", []):
        action_label = _topology_transition_label(page, transition, pages)
        if not action_label:
            continue
        if not _should_keep_transition(page, transition, pages):
            continue
        target_page = str(transition.get("target_page", "") or "")
        if target_page:
            targets.append(target_page)
    return _ordered_unique(targets)


def _topology_transition_label(source_page: dict, transition: dict, pages: Dict[str, dict]) -> str:
    action_label = _transition_action_label(transition)
    if action_label == "press_home":
        return ""
    if action_label == "press_back":
        if _is_progressive_back_transition(source_page, transition, pages):
            return "back_apply"
        return ""
    return action_label


def _topology_transition_priority(action_label: str) -> int:
    if not action_label:
        return -1
    if action_label == "back_apply":
        return 1
    return 2


def _topology_transition_richness(transition: dict) -> int:
    score = 0
    target_element = str(transition.get("target_element", transition.get("action", "")) or "").strip().lower()
    if target_element and target_element not in {
        "tap",
        "click",
        "type",
        "swipe",
        "back",
        "home",
        "press_enter",
    }:
        score += 4
    if str(transition.get("type_text", "") or "").strip():
        score += 4
    if str(transition.get("gesture_direction", "") or "").strip():
        score += 2
    raw_action = str(transition.get("raw_action", "") or "").strip().upper()
    if raw_action and raw_action not in ("PRESS_BACK", "PRESS_HOME"):
        score += 1
    return score


def _topology_transition_group_key(transition: dict) -> Tuple[str, Tuple[Any, ...]]:
    serialized = _serialize_transition(transition)
    semantic_payload = _transition_signature_payload_from_serialized(
        serialized,
        include_coordinates=False,
    )
    semantic_payload.pop("target_page", None)
    semantic_payload.pop("icon_bbox", None)
    return (
        str(serialized.get("target_page", "") or "").strip(),
        tuple(sorted(semantic_payload.items())),
    )


def _iter_topology_transitions(
    pages: Dict[str, dict],
    ordered_page_ids: List[str],
) -> Iterable[Tuple[str, str, dict, str]]:
    for page_id in ordered_page_ids:
        source_page = pages[page_id]
        best_by_group: Dict[Tuple[str, Tuple[Any, ...]], Tuple[Tuple[int, int, int], dict, str, int]] = {}
        for idx, transition in enumerate(source_page.get("transitions", [])):
            target_page = str(transition.get("target_page", "") or "")
            if not target_page or target_page == page_id or target_page not in pages:
                continue
            action_label = _topology_transition_label(source_page, transition, pages)
            if not action_label:
                continue
            if not _should_keep_transition(source_page, transition, pages):
                continue
            candidate_rank = (
                _topology_transition_priority(action_label),
                _topology_transition_richness(transition),
                -idx,
            )
            transition_group_key = _topology_transition_group_key(transition)
            existing = best_by_group.get(transition_group_key)
            if existing is None or candidate_rank > existing[0]:
                best_by_group[transition_group_key] = (candidate_rank, transition, action_label, idx)
        for (target_page, _), (_, transition, action_label, first_idx) in sorted(
            best_by_group.items(),
            key=lambda item: (
                int(pages[item[0][0]].get("depth", 0)),
                item[1][3],
                str(item[0][0]),
            ),
        ):
            yield page_id, target_page, transition, action_label


def _ordered_topology_page_ids(pages: Dict[str, dict], root_page_id: str) -> List[str]:
    ordered_ids: List[str] = []
    visited: set[str] = set()
    queue: List[str] = [root_page_id] if root_page_id in pages else []

    while queue:
        page_id = queue.pop(0)
        if page_id in visited or page_id not in pages:
            continue
        visited.add(page_id)
        ordered_ids.append(page_id)
        for target_page in _non_system_target_pages(pages[page_id], pages):
            if target_page not in visited:
                queue.append(target_page)

    for page_id in sorted(pages, key=lambda item: (int(pages[item].get("depth", 0)), str(item))):
        if page_id not in visited:
            ordered_ids.append(page_id)
    return ordered_ids


def _build_primary_parent_map(
    pages: Dict[str, dict],
    root_page_id: str,
) -> Tuple[Dict[str, str], Dict[Tuple[str, str], int], Dict[str, int]]:
    ordered_page_ids = _ordered_topology_page_ids(pages, root_page_id)
    page_order = {page_id: idx for idx, page_id in enumerate(ordered_page_ids)}
    primary_parent: Dict[str, str] = {}
    child_order: Dict[Tuple[str, str], int] = {}

    for page_id in ordered_page_ids:
        targets = _non_system_target_pages(pages[page_id], pages)
        for idx, target_page in enumerate(targets):
            child_order[(page_id, target_page)] = idx
            primary_parent.setdefault(target_page, page_id)
    return primary_parent, child_order, page_order


def _sorted_depth_buckets_for_topology(
    pages: Dict[str, dict],
    root_page_id: str,
    ordered_page_ids: List[str],
) -> Dict[int, List[str]]:
    depth_buckets: Dict[int, List[str]] = defaultdict(list)
    for page_id in ordered_page_ids:
        depth_buckets[int(pages[page_id].get("depth", 0))].append(page_id)

    primary_parent, child_order, page_order = _build_primary_parent_map(pages, root_page_id)
    row_order_lookup: Dict[str, int] = {}
    for depth in sorted(depth_buckets):
        bucket = depth_buckets[depth]
        if depth == 0:
            ordered_bucket = sorted(
                bucket,
                key=lambda page_id: (0 if page_id == root_page_id else 1, page_order.get(page_id, 10**9)),
            )
        else:
            ordered_bucket = sorted(
                bucket,
                key=lambda page_id: (
                    row_order_lookup.get(primary_parent.get(page_id, ""), 10**9),
                    child_order.get((primary_parent.get(page_id, ""), page_id), 10**9),
                    page_order.get(page_id, 10**9),
                ),
            )
        depth_buckets[depth] = ordered_bucket
        for idx, page_id in enumerate(ordered_bucket):
            row_order_lookup[page_id] = idx
    return depth_buckets


def _draw_orthogonal_edge(
    draw: ImageDraw.ImageDraw,
    start_point: Tuple[int, int],
    end_point: Tuple[int, int],
    lane_x: int,
    color: Tuple[int, int, int],
    width: int,
) -> None:
    sx, sy = start_point
    ex, ey = end_point
    lane_x = max(min(lane_x, ex - 10), sx + 10)
    draw.line(
        [
            (sx, sy),
            (lane_x, sy),
            (lane_x, ey),
            (ex, ey),
        ],
        fill=color,
        width=width,
    )


def _build_topology_tree_payload(pages: Dict[str, dict], root_page_id: str) -> dict:
    visited: set[str] = set()

    def build_node(page_id: str) -> Optional[dict]:
        if page_id in visited or page_id not in pages:
            return None
        visited.add(page_id)
        page = pages[page_id]
        outgoing_transitions = []
        for _, target_page, transition, action_label in _iter_topology_transitions(pages, [page_id]):
            outgoing_transitions.append({
                "target_page_id": target_page,
                "action": action_label,
                "raw_action": str(transition.get("raw_action", "") or ""),
                "target_element": str(transition.get("target_element", transition.get("action", "")) or ""),
                "type_text": str(transition.get("type_text", "") or ""),
                "gesture_direction": str(transition.get("gesture_direction", "") or ""),
            })
        child_nodes = []
        for target_page in _non_system_target_pages(page, pages):
            child = build_node(target_page)
            if child is not None:
                child_nodes.append(child)
        return {
            "page_id": page_id,
            "page_name": page.get("page_name", page_id),
            "application_id": page.get("application_id", ""),
            "application_name": page.get("application_name", ""),
            "image": page.get("image", f"{page_id}.png"),
            "depth": int(page.get("depth", 0)),
            "merged_from_page_ids": list(page.get("merged_from_page_ids", [page_id])),
            "outgoing_transitions": outgoing_transitions,
            "children": child_nodes,
        }

    root_node = build_node(root_page_id) if root_page_id in pages else None
    disconnected = []
    for page_id in _ordered_topology_page_ids(pages, root_page_id):
        if page_id in visited:
            continue
        child = build_node(page_id)
        if child is not None:
            disconnected.append(child)

    return {
        "root_page_id": root_page_id,
        "root": root_node,
        "disconnected_roots": disconnected,
        "metadata": {
            "total_pages": len(pages),
            "source": "mock_unified_app_graph_internal",
        },
    }


def _build_topology_graph_payload(
    pages: Dict[str, dict],
    root_page_id: str,
    same_page_state: Optional[dict] = None,
) -> dict:
    ordered_page_ids = _ordered_topology_page_ids(pages, root_page_id)
    nodes = []
    edges = []
    canonical_by_page_id = dict((same_page_state or {}).get("canonical_by_page_id", {}))
    members_by_canonical = dict((same_page_state or {}).get("members_by_canonical", {}))

    for page_id in ordered_page_ids:
        page = pages[page_id]
        canonical_page_id = canonical_by_page_id.get(page_id, page_id)
        group_members = list(members_by_canonical.get(canonical_page_id, [canonical_page_id]))
        nodes.append({
            "page_id": page_id,
            "page_name": page.get("page_name", page_id),
            "application_id": page.get("application_id", ""),
            "application_name": page.get("application_name", ""),
            "image": page.get("image", f"{page_id}.png"),
            "depth": int(page.get("depth", 0)),
            "merged_from_page_ids": list(page.get("merged_from_page_ids", [page_id])),
            "same_page_canonical_id": canonical_page_id,
            "same_page_group_size": len(group_members),
            "same_page_group_members": group_members,
            "same_page_duplicate": canonical_page_id != page_id,
        })

    for page_id, target_page, transition, action_label in _iter_topology_transitions(pages, ordered_page_ids):
        edges.append({
            "source_page_id": page_id,
            "target_page_id": target_page,
            "action": action_label,
            "raw_action": str(transition.get("raw_action", "") or ""),
            "target_element": str(transition.get("target_element", transition.get("action", "")) or ""),
            "type_text": str(transition.get("type_text", "") or ""),
            "gesture_direction": str(transition.get("gesture_direction", "") or ""),
            "system": False,
        })

    return {
        "root_page_id": root_page_id,
        "nodes": nodes,
        "edges": edges,
        "metadata": {
            "total_pages": len(nodes),
            "total_edges": len(edges),
            "excluded_actions": ["press_back", "press_home"],
            "collapsed_duplicate_pages": sum(
                max(0, len(list(page.get("merged_from_page_ids", [page_id]))) - 1)
                for page_id, page in pages.items()
            ),
            "same_page_group_count": len((same_page_state or {}).get("groups", [])),
            "same_page_groups": list((same_page_state or {}).get("groups", [])),
            "source": "mock_unified_app_graph_internal",
        },
    }


def _build_topology_source_page_lookup(metadata: Optional[dict]) -> Dict[str, str]:
    if not isinstance(metadata, dict):
        return {}
    page_id_map = metadata.get("page_id_map")
    if not isinstance(page_id_map, dict):
        return {}
    return {
        str(new_page_id): str(source_page_id)
        for source_page_id, new_page_id in page_id_map.items()
    }


def _infer_topology_app_slug(page_id: str) -> str:
    raw_page_id = str(page_id or "").strip().lower()
    if not raw_page_id.startswith("page_"):
        return ""
    suffix = raw_page_id[5:]
    if not suffix:
        return ""
    slug = suffix.split("_", 1)[0].strip()
    if not slug or slug.isdigit():
        return ""
    return slug


def _topology_app_key_for_page(
    page_id: str,
    page: dict,
    source_page_lookup: Dict[str, str],
) -> str:
    summary = page.get("page_summary") or {}
    for candidate in (
        page.get("application_id"),
        summary.get("application_id"),
        page.get("application_name"),
        summary.get("application_name"),
    ):
        normalized = _normalize_compact(candidate)
        if normalized:
            return normalized

    for candidate in (
        source_page_lookup.get(page_id, ""),
        page.get("page_id", ""),
        page.get("page_name", ""),
    ):
        inferred = _normalize_compact(_infer_topology_app_slug(str(candidate or "")))
        if inferred:
            return inferred

    if int(page.get("depth", 0)) <= 1:
        return "launcher"
    return ""


def _topology_band_key_for_page(
    page_id: str,
    page: dict,
    source_page_lookup: Dict[str, str],
) -> str:
    return _topology_app_key_for_page(page_id, page, source_page_lookup) or "misc"


def _build_topology_band_layout(
    sorted_depths: List[int],
    depth_buckets: Dict[int, List[str]],
    ordered_page_ids: List[str],
    pages: Dict[str, dict],
    source_page_lookup: Dict[str, str],
    box_h: int,
    row_gap: int,
) -> Tuple[List[str], Dict[str, int], Dict[str, int], int, int]:
    ordered_bands: List[str] = []
    seen_bands: set[str] = set()
    for page_id in ordered_page_ids:
        band_key = _topology_band_key_for_page(page_id, pages[page_id], source_page_lookup)
        if band_key in seen_bands:
            continue
        seen_bands.add(band_key)
        ordered_bands.append(band_key)

    band_capacities: Dict[str, int] = {band_key: 1 for band_key in ordered_bands}
    for depth in sorted_depths:
        counts: Dict[str, int] = {}
        for page_id in depth_buckets[depth]:
            band_key = _topology_band_key_for_page(page_id, pages[page_id], source_page_lookup)
            counts[band_key] = counts.get(band_key, 0) + 1
        for band_key, count in counts.items():
            band_capacities[band_key] = max(band_capacities.get(band_key, 1), count)

    band_gap = 58 if len(ordered_bands) >= 3 else 38
    band_tops: Dict[str, int] = {}
    cursor = 0
    for idx, band_key in enumerate(ordered_bands):
        band_tops[band_key] = cursor
        band_h = band_capacities[band_key] * box_h + max(0, band_capacities[band_key] - 1) * row_gap
        cursor += band_h
        if idx < len(ordered_bands) - 1:
            cursor += band_gap

    return ordered_bands, band_capacities, band_tops, band_gap, cursor


def _build_topology_band_label_lookup(metadata: Optional[dict]) -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    if not isinstance(metadata, dict):
        return lookup
    for row in metadata.get("resolved_icons", []) or []:
        if not isinstance(row, dict):
            continue
        label = str(row.get("label") or row.get("asset") or "").strip()
        if not label:
            continue
        for candidate in (row.get("asset"), row.get("label")):
            normalized = _normalize_compact(candidate)
            if normalized and normalized not in lookup:
                lookup[normalized] = label
    return lookup


def _topology_asset_signature_similarity(
    left: Iterable[Tuple[str, str]],
    right: Iterable[Tuple[str, str]],
) -> float:
    left_set = set(left)
    right_set = set(right)
    if not left_set and not right_set:
        return 1.0
    union = left_set | right_set
    if not union:
        return 1.0
    return float(len(left_set & right_set)) / float(len(union))


def _expand_topology_pages_by_trajectory_asset_clusters(
    output_dir: Path,
    pages: Dict[str, dict],
    metadata: Optional[dict] = None,
    matched_rows: Optional[List[dict]] = None,
    asset_rows: Optional[List[dict]] = None,
) -> Dict[str, dict]:
    if matched_rows is None or asset_rows is None:
        matched_steps_path = output_dir / "matched_steps.json"
        asset_manifest_path = output_dir / "trajectory_assets_manifest.json"
        if not matched_steps_path.exists() or not asset_manifest_path.exists():
            return pages

        try:
            matched_rows = json.loads(matched_steps_path.read_text(encoding="utf-8"))
            asset_rows = json.loads(asset_manifest_path.read_text(encoding="utf-8"))
        except Exception:
            return pages

    def row_cluster_key(row: dict) -> Tuple[str, str]:
        return (
            str(row.get("episode_id", "") or "").strip(),
            str(row.get("screenshot", "") or "").strip(),
        )

    asset_signatures: Dict[Tuple[str, str], set[Tuple[str, str]]] = defaultdict(set)
    for row in asset_rows:
        episode_id = str(row.get("episode_id", "") or "").strip()
        screenshot = str(row.get("screenshot", "") or "").strip()
        if not episode_id or not screenshot:
            continue
        asset_type = str(row.get("type", "") or "").strip().lower()
        asset_label = _normalize_compact(row.get("label"))
        if not asset_type and not asset_label:
            continue
        asset_signatures[(episode_id, screenshot)].add((asset_type, asset_label))

    rows_by_page_id: Dict[str, List[dict]] = defaultdict(list)
    rows_by_trajectory: Dict[str, List[dict]] = defaultdict(list)
    for row in matched_rows:
        page_id = str(row.get("page_id", "") or "").strip()
        if page_id not in pages:
            continue
        row_copy = dict(row)
        rows_by_page_id[page_id].append(row_copy)
        trajectory_id_full = str(row.get("trajectory_id_full", "") or "").strip()
        trajectory_key = trajectory_id_full or str(row.get("episode_id", "") or "").strip()
        if trajectory_key:
            rows_by_trajectory[trajectory_key].append(row_copy)

    future_signature_by_row_key: Dict[Tuple[str, str], Tuple[str, ...]] = {}
    for trajectory_rows in rows_by_trajectory.values():
        ordered_rows = sorted(
            trajectory_rows,
            key=lambda item: (
                int(item.get("step_index", 0)),
                str(item.get("screenshot", "") or ""),
            ),
        )
        future_signature: Tuple[str, ...] = ()
        for row in reversed(ordered_rows):
            future_signature_by_row_key[row_cluster_key(row)] = future_signature
            page_id = str(row.get("page_id", "") or "").strip()
            if page_id in pages:
                future_signature = (page_id,) + future_signature

    split_threshold = 0.15
    cluster_index_by_row_key: Dict[Tuple[str, str], int] = {}
    split_page_ids: List[str] = []
    next_numeric_page_id = 1 + max(
        (
            int(match.group(1))
            for page_id in pages
            for match in [re.match(r"page_(\d+)$", str(page_id))]
            if match
        ),
        default=-1,
    )
    synthetic_ids_by_page_and_cluster: Dict[Tuple[str, int], str] = {}
    base_page_id_by_node_id: Dict[str, str] = {page_id: page_id for page_id in pages}

    for page_id, page_rows in rows_by_page_id.items():
        page_depth = int(pages.get(page_id, {}).get("depth", 0))
        # Keep direct app entry pages stable so launcher transitions stay anchored.
        # Deeper pages can split by downstream trajectory path when their futures diverge.
        allow_future_path_split = page_depth > 3
        unique_rows = []
        seen_row_keys: set[Tuple[str, str]] = set()
        for row in sorted(
            page_rows,
            key=lambda item: (
                str(item.get("trajectory_id_full", "") or ""),
                int(item.get("step_index", 0)),
                str(item.get("screenshot", "") or ""),
            ),
        ):
            row_key = row_cluster_key(row)
            if row_key in seen_row_keys:
                continue
            seen_row_keys.add(row_key)
            unique_rows.append(row)
        if len(unique_rows) < 2:
            continue

        clusters: List[List[dict]] = []
        cluster_signatures: List[set[Tuple[str, str]]] = []
        cluster_future_signatures: List[Tuple[str, ...]] = []
        for row in unique_rows:
            row_key = row_cluster_key(row)
            signature = set(asset_signatures.get((
                str(row.get("episode_id", "") or ""),
                str(row.get("screenshot", "") or ""),
            ), set()))
            future_signature = future_signature_by_row_key.get(row_key, ())
            assigned = False
            for cluster_index, cluster_signature in enumerate(cluster_signatures):
                if allow_future_path_split and cluster_future_signatures[cluster_index] != future_signature:
                    continue
                if _topology_asset_signature_similarity(signature, cluster_signature) >= split_threshold:
                    clusters[cluster_index].append(row)
                    cluster_signatures[cluster_index].update(signature)
                    assigned = True
                    break
            if assigned:
                continue
            clusters.append([row])
            cluster_signatures.append(set(signature))
            cluster_future_signatures.append(future_signature if allow_future_path_split else ())

        if len(clusters) < 2:
            continue

        split_page_ids.append(page_id)
        for cluster_index, cluster_rows in enumerate(clusters):
            target_page_id = page_id
            if cluster_index > 0:
                target_page_id = f"page_{next_numeric_page_id}"
                next_numeric_page_id += 1
                synthetic_ids_by_page_and_cluster[(page_id, cluster_index)] = target_page_id
                base_page_id_by_node_id[target_page_id] = page_id
            for row in cluster_rows:
                cluster_index_by_row_key[row_cluster_key(row)] = cluster_index

    if not split_page_ids:
        return pages

    expanded_pages: Dict[str, dict] = {}
    for page_id, page in pages.items():
        cloned = deepcopy(page)
        if int(cloned.get("depth", 0)) >= 2:
            cloned["transitions"] = []
        expanded_pages[page_id] = cloned

    for (base_page_id, cluster_index), synthetic_page_id in synthetic_ids_by_page_and_cluster.items():
        cloned = deepcopy(pages[base_page_id])
        cloned["page_id"] = synthetic_page_id
        cloned["image"] = pages[base_page_id].get("image", f"{base_page_id}.png")
        cloned["transitions"] = []
        cloned["topology_split_from_page_id"] = base_page_id
        expanded_pages[synthetic_page_id] = cloned

    desired_targets_by_source_node: Dict[str, List[str]] = defaultdict(list)
    desired_canonical_target_by_pair: Dict[Tuple[str, str], str] = {}

    def row_node_id(row: dict) -> str:
        page_id = str(row.get("page_id", "") or "").strip()
        cluster_index = cluster_index_by_row_key.get(row_cluster_key(row), 0)
        return synthetic_ids_by_page_and_cluster.get((page_id, cluster_index), page_id)

    for trajectory_rows in rows_by_trajectory.values():
        ordered_rows = sorted(
            trajectory_rows,
            key=lambda item: (
                int(item.get("step_index", 0)),
                str(item.get("screenshot", "") or ""),
            ),
        )
        for source_row, target_row in zip(ordered_rows, ordered_rows[1:]):
            source_page_id = str(source_row.get("page_id", "") or "").strip()
            target_page_id = str(target_row.get("page_id", "") or "").strip()
            if source_page_id not in pages or target_page_id not in pages:
                continue
            source_node_id = row_node_id(source_row)
            target_node_id = row_node_id(target_row)
            pair_key = (source_node_id, target_node_id)
            if target_node_id not in desired_targets_by_source_node[source_node_id]:
                desired_targets_by_source_node[source_node_id].append(target_node_id)
            desired_canonical_target_by_pair[pair_key] = target_page_id

    for source_page_id, source_page in pages.items():
        if int(source_page.get("depth", 0)) < 2:
            continue
        source_node_ids = [source_page_id]
        source_node_ids.extend(
            synthetic_page_id
            for (base_page_id, _), synthetic_page_id in sorted(
                synthetic_ids_by_page_and_cluster.items(),
                key=lambda item: item[0][1],
            )
            if base_page_id == source_page_id
        )
        kept_transitions = [
            transition
            for transition in source_page.get("transitions", [])
            if _topology_transition_label(source_page, transition, pages)
            and _should_keep_transition(source_page, transition, pages)
        ]
        transitions_by_canonical_target: Dict[str, List[dict]] = defaultdict(list)
        for transition in kept_transitions:
            transitions_by_canonical_target[str(transition.get("target_page", "") or "")].append(transition)

        for source_node_id in source_node_ids:
            base_source_page_id = base_page_id_by_node_id.get(source_node_id, source_page_id)
            target_use_counts: Dict[str, int] = defaultdict(int)
            for target_node_id in desired_targets_by_source_node.get(source_node_id, []):
                canonical_target_page_id = desired_canonical_target_by_pair.get((source_node_id, target_node_id), "")
                candidate_pool = transitions_by_canonical_target.get(canonical_target_page_id, [])
                if not candidate_pool:
                    continue
                candidate_index = min(target_use_counts[canonical_target_page_id], len(candidate_pool) - 1)
                target_use_counts[canonical_target_page_id] += 1
                remapped = deepcopy(candidate_pool[candidate_index])
                remapped["target_page"] = target_node_id
                expanded_pages[source_node_id]["transitions"].append(remapped)

    row_node_id_by_row_key: Dict[Tuple[str, str], str] = {}
    for row in matched_rows:
        page_id = str(row.get("page_id", "") or "").strip()
        if page_id not in pages:
            continue
        row_key = row_cluster_key(row)
        node_id = row_node_id(row)
        row_node_id_by_row_key[row_key] = node_id
        row["page_id"] = node_id

    for row in asset_rows:
        page_id = str(row.get("page_id", "") or "").strip()
        if page_id not in pages:
            continue
        node_id = row_node_id_by_row_key.get(row_cluster_key(row))
        if node_id:
            row["page_id"] = node_id

    for page_id, page in expanded_pages.items():
        if int(page.get("depth", 0)) < 2:
            continue
        deduped_transitions: List[dict] = []
        seen_signatures: set[str] = set()
        for transition in page.get("transitions", []):
            signature = _transition_merge_signature(transition)
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)
            deduped_transitions.append(transition)
        page["transitions"] = deduped_transitions

    return expanded_pages


def _topology_display_name_for_band(
    band_key: str,
    band_label_lookup: Dict[str, str],
) -> str:
    normalized = _normalize_compact(band_key)
    if normalized in band_label_lookup:
        return band_label_lookup[normalized]
    if normalized == "launcher":
        return "Launcher"
    if normalized == "misc":
        return "Misc"
    if not normalized:
        return "Unknown"
    return str(band_key).replace("_", " ").title()


def _lighten_rgb(color: Tuple[int, int, int], factor: float) -> Tuple[int, int, int]:
    clamped = max(0.0, min(1.0, factor))
    return tuple(
        int(round(channel + (255 - channel) * clamped))
        for channel in color
    )


def _draw_topology_edge_tag(
    draw: ImageDraw.ImageDraw,
    center: Tuple[float, float],
    text: str,
    font,
    fill_color: Tuple[int, int, int],
) -> None:
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_w = text_bbox[2] - text_bbox[0]
    text_h = text_bbox[3] - text_bbox[1]
    pad_x = 12
    pad_y = 6
    rect_x1 = int(round(center[0] - text_w / 2 - pad_x))
    rect_y1 = int(round(center[1] - text_h / 2 - pad_y))
    rect_x2 = int(round(center[0] + text_w / 2 + pad_x))
    rect_y2 = int(round(center[1] + text_h / 2 + pad_y))
    draw.rounded_rectangle(
        [rect_x1, rect_y1, rect_x2, rect_y2],
        radius=16,
        fill=fill_color,
        outline=(187, 194, 206),
        width=2,
    )
    draw.text(
        (int(round(center[0] - text_w / 2)), int(round(center[1] - text_h / 2 - 1))),
        text,
        fill=(34, 39, 48),
        font=font,
    )


def _node_fill_for_app(application_id: str) -> Tuple[int, int, int]:
    palette = [
        (232, 242, 255),
        (234, 247, 237),
        (255, 241, 230),
        (247, 237, 255),
        (255, 238, 242),
        (236, 244, 245),
    ]
    normalized = _normalize_compact(application_id)
    if "novelship" in normalized:
        return (255, 228, 223)
    if "citymapper" in normalized:
        return (220, 246, 230)
    if normalized == "launcher":
        return (239, 242, 247)
    if not normalized:
        return palette[0]
    index = sum(ord(ch) for ch in normalized) % len(palette)
    return palette[index]


def _truncate_text(draw: ImageDraw.ImageDraw, text: str, font, max_width: int) -> str:
    text = str(text or "")
    if not text:
        return ""
    if draw.textbbox((0, 0), text, font=font)[2] <= max_width:
        return text
    suffix = "..."
    current = text
    while current:
        candidate = current + suffix
        if draw.textbbox((0, 0), candidate, font=font)[2] <= max_width:
            return candidate
        current = current[:-1]
    return suffix


def _topology_edge_label(transition: dict) -> str:
    action_label = _transition_action_label(transition)
    if action_label == "tap":
        target_element = str(transition.get("target_element", transition.get("action", "")) or "").strip()
        if target_element and target_element.lower() not in ("tap", "click"):
            return target_element
        return "tap"
    if action_label == "swipe":
        direction = str(transition.get("gesture_direction", "") or "").strip().lower()
        return f"swipe_{direction}" if direction else "swipe"
    if action_label == "type":
        type_text = str(transition.get("type_text", "") or "").strip()
        return f"type:{type_text}" if type_text else "type"
    return action_label


def _topology_edge_group_label(edge_labels: List[str], transition_count: int) -> str:
    labels = [str(label or "").strip() for label in edge_labels if str(label or "").strip()]
    if not labels and transition_count <= 1:
        return ""
    if not labels:
        return f"x{transition_count}"
    if len(labels) == 1:
        return f"{labels[0]} x{transition_count}" if transition_count > 1 else labels[0]
    preview = labels[:3]
    joined = " | ".join(preview)
    if len(labels) > len(preview):
        joined = f"{joined} +{len(labels) - len(preview)}"
    return joined


def _save_topology_visualization_pil_fallback(
    pages: Dict[str, dict],
    root_page_id: str,
    output_path: Path,
    metadata: Optional[dict] = None,
    same_page_state: Optional[dict] = None,
) -> None:
    if not pages:
        Image.new("RGB", (1200, 800), (255, 255, 255)).save(output_path)
        return

    ordered_page_ids = _ordered_topology_page_ids(pages, root_page_id)
    topology_edges = list(_iter_topology_transitions(pages, ordered_page_ids))
    depth_buckets = _sorted_depth_buckets_for_topology(pages, root_page_id, ordered_page_ids)
    primary_parent, _, _ = _build_primary_parent_map(pages, root_page_id)
    source_page_lookup = _build_topology_source_page_lookup(metadata)
    visual_ordered_page_ids: List[str] = list(ordered_page_ids)
    visual_depth_buckets: Dict[int, List[str]] = {
        depth: list(bucket)
        for depth, bucket in depth_buckets.items()
    }

    def base_page_id(page_id: str) -> str:
        return page_id

    def visual_page(page_id: str) -> dict:
        return pages[page_id]

    sorted_depths = sorted(visual_depth_buckets)
    box_w = 252
    box_h = 72
    col_gap = 156
    row_gap = 26
    margin_x = 80
    margin_y = 90
    title_h = 70
    ordered_bands, band_capacities, band_tops, band_gap, required_usable_h = _build_topology_band_layout(
        sorted_depths,
        visual_depth_buckets,
        visual_ordered_page_ids,
        {page_id: visual_page(page_id) for page_id in visual_ordered_page_ids},
        source_page_lookup,
        box_h,
        row_gap,
    )
    canvas_w = max(1400, margin_x * 2 + len(sorted_depths) * box_w + max(0, len(sorted_depths) - 1) * col_gap)
    canvas_h = max(900, margin_y * 2 + title_h + required_usable_h)
    usable_h = canvas_h - margin_y * 2 - title_h
    stack_top = margin_y + title_h + max(0, (usable_h - required_usable_h) // 2)

    canvas = Image.new("RGB", (canvas_w, canvas_h), (250, 251, 253))
    draw = ImageDraw.Draw(canvas)
    font_title = launcher_mock.FONT_MD
    font_page = launcher_mock.FONT_SM
    font_tag = _ac_try_load_font(20)
    band_label_lookup = _build_topology_band_label_lookup(metadata)
    canonical_by_page_id = dict((same_page_state or {}).get("canonical_by_page_id", {}))
    members_by_canonical = dict((same_page_state or {}).get("members_by_canonical", {}))

    draw.text(
        (margin_x, 28),
        f"UI Topology",
        fill=(24, 30, 40),
        font=font_title,
    )

    positions: Dict[str, Tuple[int, int]] = {}
    for depth_idx, depth in enumerate(sorted_depths):
        x = margin_x + depth_idx * (box_w + col_gap)
        draw.text((x, margin_y), f"Depth {depth}", fill=(86, 92, 104), font=font_page)
        column_items = visual_depth_buckets[depth]
        grouped_items: Dict[str, List[str]] = {band_key: [] for band_key in ordered_bands}
        for page_id in column_items:
            band_key = _topology_band_key_for_page(base_page_id(page_id), visual_page(page_id), source_page_lookup)
            grouped_items.setdefault(band_key, []).append(page_id)
        for band_key in ordered_bands:
            band_items = grouped_items.get(band_key, [])
            if not band_items:
                continue
            band_h = band_capacities[band_key] * box_h + max(0, band_capacities[band_key] - 1) * row_gap
            content_h = len(band_items) * box_h + max(0, len(band_items) - 1) * row_gap
            band_start_y = stack_top + band_tops[band_key] + max(0, (band_h - content_h) // 2)
            for row_idx, page_id in enumerate(band_items):
                y = band_start_y + row_idx * (box_h + row_gap)
                positions[page_id] = (x, y)

    separator_x1 = max(24, margin_x - 28)
    separator_x2 = canvas_w - max(24, margin_x - 8)
    for idx in range(1, len(ordered_bands)):
        band_key = ordered_bands[idx]
        separator_y = stack_top + band_tops[band_key] - band_gap // 2
        draw.line(
            [(separator_x1, separator_y), (separator_x2, separator_y)],
            fill=(228, 232, 238),
            width=2,
        )

    grouped_edges: Dict[Tuple[str, str], List[Tuple[int, dict, str]]] = defaultdict(list)
    for edge_idx, (page_id, target_page, transition, action_label) in enumerate(topology_edges):
        grouped_edges[(page_id, target_page)].append((edge_idx, transition, action_label))

    edge_specs: List[dict] = []
    for edge_idx, ((page_id, target_page), grouped) in enumerate(
        sorted(
            grouped_edges.items(),
            key=lambda item: (
                int(pages[item[0][0]].get("depth", 0)),
                int(pages[item[0][1]].get("depth", 0)),
                str(item[0][0]),
                str(item[0][1]),
            ),
        )
    ):
        representative_idx, representative_transition, representative_action_label = max(
            grouped,
            key=lambda item: (
                _topology_transition_richness(item[1]),
                -item[0],
            ),
        )
        del representative_idx
        edge_specs.append({
            "edge_id": f"edge_{edge_idx}",
            "is_primary": primary_parent.get(target_page) == page_id,
            "source_page_id": page_id,
            "target_page_id": target_page,
            "logical_target_page_id": target_page,
            "transition": representative_transition,
            "action_label": representative_action_label,
            "edge_labels": _ordered_unique(_topology_edge_label(item[1]) for item in grouped),
            "transition_count": len(grouped),
        })

    edge_specs.sort(
        key=lambda item: (
            item["is_primary"],
            visual_page(str(item["source_page_id"])).get("depth", 0),
            item["source_page_id"],
            item["logical_target_page_id"],
            _topology_edge_label(item["transition"]),
        )
    )
    edge_lookup = {str(edge_spec["edge_id"]): edge_spec for edge_spec in edge_specs}
    outgoing_by_source: Dict[str, List[str]] = defaultdict(list)
    incoming_by_target: Dict[str, List[str]] = defaultdict(list)
    for edge_spec in edge_specs:
        outgoing_by_source[edge_spec["source_page_id"]].append(edge_spec["edge_id"])
        incoming_by_target[edge_spec["target_page_id"]].append(edge_spec["edge_id"])

    def edge_slot_offset(index: int, total: int) -> int:
        if total <= 1:
            return box_h // 2
        top_pad = 14
        bottom_pad = 14
        usable_h = max(8, box_h - top_pad - bottom_pad)
        if total == 2:
            center = box_h // 2
            return center - 12 if index == 0 else center + 12
        step = usable_h / float(total - 1)
        return int(round(top_pad + index * step))

    drawn_app_entry_tags: set[str] = set()
    for edge_spec in edge_specs:
        is_primary = bool(edge_spec["is_primary"])
        page_id = str(edge_spec["source_page_id"])
        target_page = str(edge_spec["target_page_id"])
        transition = edge_spec["transition"]
        edge_id = str(edge_spec["edge_id"])
        source_xy = positions.get(page_id)
        target_xy = positions.get(target_page)
        if source_xy is None or target_xy is None:
            continue
        sx, sy = source_xy
        tx, ty = target_xy
        source_targets = sorted(
            outgoing_by_source.get(page_id, [edge_id]),
            key=lambda candidate: (
                positions.get(edge_lookup[candidate]["target_page_id"], (0, 0))[1],
                str(candidate),
            ),
        )
        target_sources = sorted(
            incoming_by_target.get(target_page, [edge_id]),
            key=lambda candidate: (
                positions.get(edge_lookup[candidate]["source_page_id"], (0, 0))[1],
                str(candidate),
            ),
        )
        source_index = source_targets.index(edge_id) if edge_id in source_targets else 0
        target_index = target_sources.index(edge_id) if edge_id in target_sources else 0
        start_point = (
            sx + box_w - 2,
            sy + edge_slot_offset(source_index, len(source_targets)),
        )
        end_point = (
            tx + 2,
            ty + edge_slot_offset(target_index, len(target_sources)),
        )
        line_color = (84, 121, 255) if is_primary else (196, 204, 224)
        line_width = 3 if is_primary else 2
        draw.line([start_point, end_point], fill=line_color, width=line_width)
        dx = end_point[0] - start_point[0]
        dy = end_point[1] - start_point[1]
        length = max((dx * dx + dy * dy) ** 0.5, 1.0)
        unit_x = dx / length
        unit_y = dy / length
        perp_x = -unit_y
        perp_y = unit_x
        arrow_size = 10 if is_primary else 8
        arrow_base_x = end_point[0] - unit_x * arrow_size
        arrow_base_y = end_point[1] - unit_y * arrow_size
        wing = arrow_size * 0.48
        arrow = [
            (int(round(end_point[0])), int(round(end_point[1]))),
            (int(round(arrow_base_x + perp_x * wing)), int(round(arrow_base_y + perp_y * wing))),
            (int(round(arrow_base_x - perp_x * wing)), int(round(arrow_base_y - perp_y * wing))),
        ]
        draw.polygon(arrow, fill=line_color)

        source_app = _topology_band_key_for_page(base_page_id(page_id), visual_page(page_id), source_page_lookup)
        target_app = _topology_band_key_for_page(base_page_id(target_page), visual_page(target_page), source_page_lookup)
        if source_app == "launcher" and target_app not in ("launcher", "misc") and target_app not in drawn_app_entry_tags:
            drawn_app_entry_tags.add(target_app)
            label_center = (
                start_point[0] + dx * 0.42,
                start_point[1] + dy * 0.42 - 18,
            )
            _draw_topology_edge_tag(
                draw,
                label_center,
                _topology_display_name_for_band(target_app, band_label_lookup),
                font_tag,
                _lighten_rgb(_node_fill_for_app(target_app), 0.10),
            )

        edge_group_label = _topology_edge_group_label(
            list(edge_spec.get("edge_labels", [])),
            int(edge_spec.get("transition_count", 1)),
        )
        if edge_group_label and (len(edge_spec.get("edge_labels", [])) > 1 or int(edge_spec.get("transition_count", 1)) > 1):
            label_center = (
                start_point[0] + dx * 0.50,
                start_point[1] + dy * 0.50,
            )
            _draw_topology_edge_tag(
                draw,
                label_center,
                _truncate_text(draw, edge_group_label, font_page, 220),
                font_page,
                _lighten_rgb(_node_fill_for_app(source_app), 0.18),
            )

    for page_id in visual_ordered_page_ids:
        x, y = positions[page_id]
        page = visual_page(page_id)
        real_page_id = base_page_id(page_id)
        app_key = _topology_app_key_for_page(real_page_id, page, source_page_lookup)
        fill_color = _node_fill_for_app(app_key)
        app_label = _topology_display_name_for_band(app_key, band_label_lookup)
        outline = (186, 194, 208)
        if real_page_id == root_page_id and page_id == real_page_id:
            outline = (54, 102, 227)
        canonical_page_id = canonical_by_page_id.get(real_page_id, real_page_id)
        group_members = list(members_by_canonical.get(canonical_page_id, [canonical_page_id]))
        is_duplicate_page = canonical_page_id != real_page_id
        if is_duplicate_page:
            outline = (120, 132, 154)
        draw.rounded_rectangle([x, y, x + box_w, y + box_h], radius=18, fill=fill_color, outline=outline, width=3)
        page_title = real_page_id
        subtitle = app_label
        if is_duplicate_page:
            subtitle = f"{subtitle} -> {canonical_page_id}"
        title_text = _truncate_text(draw, page_title, font_page, box_w - 24)
        subtitle_text = _truncate_text(draw, subtitle, font_page, box_w - 24)
        draw.text((x + 12, y + 10), title_text, fill=(26, 31, 41), font=font_page)
        draw.text((x + 12, y + 36), subtitle_text, fill=(78, 84, 96), font=font_page)

    canvas.save(output_path)


def _save_topology_visualization(
    pages: Dict[str, dict],
    root_page_id: str,
    output_path: Path,
    metadata: Optional[dict] = None,
    same_page_state: Optional[dict] = None,
) -> None:
    _save_topology_visualization_pil_fallback(
        pages,
        root_page_id,
        output_path,
        metadata=metadata,
        same_page_state=same_page_state,
    )


def _write_topology_artifacts(
    output_dir: Path,
    pages: Dict[str, dict],
    root_page_id: str,
    metadata: Optional[dict] = None,
) -> None:
    same_page_state = _build_topology_same_page_state(pages, output_dir, metadata)
    topology_pages, topology_root_page_id = _collapse_topology_pages(
        pages,
        root_page_id=root_page_id,
        same_page_state=same_page_state,
        metadata=metadata,
    )
    topology_pages = _expand_topology_pages_by_trajectory_asset_clusters(
        output_dir=output_dir,
        pages=topology_pages,
        metadata=metadata,
    )
    topology_tree = _build_topology_tree_payload(topology_pages, root_page_id=topology_root_page_id)
    topology_graph = _build_topology_graph_payload(
        topology_pages,
        root_page_id=topology_root_page_id,
        same_page_state=same_page_state,
    )
    (output_dir / "ui_topology_tree.json").write_text(json.dumps(topology_tree, indent=2), encoding="utf-8")
    (output_dir / "ui_topology.json").write_text(json.dumps(topology_graph, indent=2), encoding="utf-8")
    (output_dir / "ui_topology_same_pages.json").write_text(
        json.dumps({"groups": same_page_state.get("groups", [])}, indent=2),
        encoding="utf-8",
    )
    _save_topology_visualization(
        topology_pages,
        root_page_id=topology_root_page_id,
        output_path=output_dir / "ui_topology.png",
        metadata=metadata,
        same_page_state=same_page_state,
    )


def _save_action_debug_overlays(output_dir: Path, pages: Dict[str, dict]) -> None:
    _ensure_compose_modules()
    debug_dir = output_dir / "action_coord_debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    for page_id, page in pages.items():
        image_path = output_dir / "pages" / f"{page_id}.png"
        overlay_path = debug_dir / f"{page_id}.png"
        action_compose._save_action_debug_overlay(str(image_path), str(overlay_path), page.get("transitions", []))


def _write_match_report(output_dir: Path,
                        drawer_apps: List[DrawerAppSpec],
                        matches_by_app: Dict[str, List[MatchedTrajectory]]) -> None:
    payload = {
        "apps": [
            {
                "label": app.label,
                "asset": app.asset,
                "slug": app.slug,
                "layout_key": app.layout_key,
                "bbox": app.bbox,
                "matched_annotation_count": len(matches_by_app.get(app.slug, [])),
                "matches": [
                    {
                        "trajectory_id_full": _full_trajectory_id(match),
                        "episode_id": match.episode_id,
                        "annotation_path": match.annotation_path,
                        "instruction": match.instruction,
                        "start_step_idx": match.start_step_idx,
                        "end_step_idx": match.end_step_idx,
                        "matched_package": match.matched_package,
                        "total_steps": match.total_steps,
                    }
                    for match in matches_by_app.get(app.slug, [])
                ],
            }
            for app in drawer_apps
        ]
    }
    (output_dir / "matched_annotations.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _normalize_signature_key(key: str) -> str:
    normalized = str(key or "").strip().lower()
    normalized = re.sub(r"_[0-9]+$", "", normalized)
    return normalized


def _quantize(value: int, bucket: int = 24) -> int:
    return int(round(int(value) / float(bucket)))


def _ordered_unique(values: Iterable[Any]) -> List[Any]:
    result: List[Any] = []
    seen: set[str] = set()
    for value in values:
        try:
            marker = json.dumps(value, sort_keys=True)
        except TypeError:
            marker = str(value)
        if marker in seen:
            continue
        seen.add(marker)
        result.append(value)
    return result


def _page_structure_signature(page: dict) -> str:
    summary = page.get("page_summary") or {}
    payload = {
        "merge_group": _page_merge_group_id(page),
        "page_family": str(summary.get("page_family", "") or ""),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _page_structure_items(page: dict) -> List[Tuple[int, int, int, int]]:
    layout_items: List[Tuple[int, int, int, int]] = []
    for key, bbox in sorted((page.get("layout") or {}).items()):
        if isinstance(bbox, dict) and "bbox" in bbox:
            bbox = bbox.get("bbox", [0, 0, 0, 0])
        if key in ("back", "home") or not _valid_bbox(bbox):
            continue
        x1, y1, x2, y2 = [int(v) for v in bbox]
        width = x2 - x1
        height = y2 - y1
        # Ignore tiny OCR/noise boxes so visually identical states collapse.
        if width < 18 or height < 12:
            continue
        layout_items.append((
            _quantize(x1, bucket=32),
            _quantize(y1, bucket=32),
            _quantize(width, bucket=32),
            _quantize(height, bucket=32),
        ))
    layout_items.sort()
    return layout_items


def _page_structure_similarity(items_a: List[Tuple[int, int, int, int]],
                               items_b: List[Tuple[int, int, int, int]]) -> float:
    if not items_a and not items_b:
        return 1.0
    counts_a: Dict[Tuple[int, int, int, int], int] = defaultdict(int)
    counts_b: Dict[Tuple[int, int, int, int], int] = defaultdict(int)
    for item in items_a:
        counts_a[item] += 1
    for item in items_b:
        counts_b[item] += 1
    keys = set(counts_a) | set(counts_b)
    intersection = sum(min(counts_a[key], counts_b[key]) for key in keys)
    union = sum(max(counts_a[key], counts_b[key]) for key in keys)
    if union == 0:
        return 1.0
    return float(intersection) / float(union)


def _build_topology_same_page_state(
    pages: Dict[str, dict],
    output_dir: Path,
    metadata: Optional[dict],
) -> dict:
    source_page_lookup = _build_topology_source_page_lookup(metadata)
    page_hashes: Dict[str, str] = {}
    structure_cache: Dict[str, List[Tuple[int, int, int, int]]] = {}
    buckets: Dict[Tuple[str, int], List[str]] = defaultdict(list)

    for page_id, page in sorted(pages.items(), key=lambda item: _topology_page_order_key(str(item[0]), pages)):
        app_key = _topology_app_key_for_page(page_id, page, source_page_lookup)
        if not app_key or app_key == "launcher":
            continue
        buckets[(app_key, int(page.get("depth", 0)))].append(page_id)

    canonical_by_page_id: Dict[str, str] = {}
    members_by_canonical: Dict[str, List[str]] = {}
    groups: List[dict] = []

    for (app_key, depth), page_ids in buckets.items():
        if len(page_ids) < 2:
            continue
        ordered_ids = sorted(page_ids, key=lambda page_id: _topology_page_order_key(page_id, pages))
        parent = {page_id: page_id for page_id in ordered_ids}

        def find(page_id: str) -> str:
            current = page_id
            while parent[current] != current:
                parent[current] = parent[parent[current]]
                current = parent[current]
            return current

        def union(page_id_a: str, page_id_b: str) -> None:
            root_a = find(page_id_a)
            root_b = find(page_id_b)
            if root_a == root_b:
                return
            if _topology_page_order_key(root_a, pages) <= _topology_page_order_key(root_b, pages):
                parent[root_b] = root_a
            else:
                parent[root_a] = root_b

        for idx, canonical_id in enumerate(ordered_ids):
            for candidate_id in ordered_ids[idx + 1:]:
                if _looks_like_same_page(
                    output_dir,
                    pages,
                    canonical_id,
                    candidate_id,
                    page_hashes,
                    structure_cache,
                ):
                    union(canonical_id, candidate_id)

        grouped_members: Dict[str, List[str]] = defaultdict(list)
        for page_id in ordered_ids:
            grouped_members[find(page_id)].append(page_id)

        for group_members in grouped_members.values():
            if len(group_members) < 2:
                continue
            group_members.sort(key=lambda page_id: _topology_page_order_key(page_id, pages))
            canonical_id = group_members[0]
            members_by_canonical[canonical_id] = group_members
            for page_id in group_members:
                canonical_by_page_id[page_id] = canonical_id
            groups.append({
                "canonical_page_id": canonical_id,
                "page_ids": group_members,
                "app_key": app_key,
                "depth": depth,
                "count": len(group_members),
            })

    groups.sort(key=lambda item: _topology_page_order_key(item["canonical_page_id"], pages))
    return {
        "canonical_by_page_id": canonical_by_page_id,
        "members_by_canonical": members_by_canonical,
        "groups": groups,
    }


def _ensure_topology_page_identity(
    page_id: str,
    page: dict,
    source_page_lookup: Dict[str, str],
    band_label_lookup: Dict[str, str],
) -> None:
    app_key = _topology_app_key_for_page(page_id, page, source_page_lookup)
    if not app_key:
        return
    if not str(page.get("application_id", "") or "").strip():
        page["application_id"] = app_key
    if not str(page.get("application_name", "") or "").strip():
        page["application_name"] = _topology_display_name_for_band(app_key, band_label_lookup)
    summary = page.get("page_summary")
    if not isinstance(summary, dict):
        summary = {}
        page["page_summary"] = summary
    if not str(summary.get("application_id", "") or "").strip():
        summary["application_id"] = page.get("application_id", "")
    if not str(summary.get("application_name", "") or "").strip():
        summary["application_name"] = page.get("application_name", "")


def _collapse_topology_pages(
    pages: Dict[str, dict],
    root_page_id: str,
    same_page_state: Optional[dict],
    metadata: Optional[dict],
) -> Tuple[Dict[str, dict], str]:
    canonical_by_page_id = dict((same_page_state or {}).get("canonical_by_page_id", {}))
    members_by_canonical = dict((same_page_state or {}).get("members_by_canonical", {}))
    source_page_lookup = _build_topology_source_page_lookup(metadata)
    band_label_lookup = _build_topology_band_label_lookup(metadata)

    if not canonical_by_page_id:
        collapsed_pages = {page_id: deepcopy(page) for page_id, page in pages.items()}
        for page_id, page in collapsed_pages.items():
            page.setdefault("merged_from_page_ids", [page_id])
            _ensure_topology_page_identity(page_id, page, source_page_lookup, band_label_lookup)
        return collapsed_pages, root_page_id

    ordered_source_ids = _ordered_topology_page_ids(pages, root_page_id)
    ordered_canonical_ids: List[str] = []
    seen_canonicals: set[str] = set()
    for page_id in list(ordered_source_ids) + sorted(pages, key=lambda item: _topology_page_order_key(item, pages)):
        canonical_id = canonical_by_page_id.get(page_id, page_id)
        if canonical_id in seen_canonicals:
            continue
        seen_canonicals.add(canonical_id)
        ordered_canonical_ids.append(canonical_id)

    collapsed_pages: Dict[str, dict] = {}
    for canonical_id in ordered_canonical_ids:
        group_members = list(members_by_canonical.get(canonical_id, [canonical_id]))
        group_members.sort(key=lambda page_id: _topology_page_order_key(page_id, pages))
        canonical_page = deepcopy(pages[canonical_id])
        canonical_page["page_id"] = canonical_id
        canonical_page["merged_from_page_ids"] = list(group_members)
        for duplicate_id in group_members[1:]:
            _merge_page_metadata(canonical_page, pages[duplicate_id])
        canonical_page.pop("_merged_transition_pool", None)

        remapped_transitions: List[dict] = []
        seen_transition_signatures: set[str] = set()
        for member_id in group_members:
            for transition in pages[member_id].get("transitions", []):
                remapped = deepcopy(transition)
                target_page = str(remapped.get("target_page", "") or "")
                if not target_page:
                    continue
                remapped["target_page"] = canonical_by_page_id.get(target_page, target_page)
                if remapped["target_page"] not in pages:
                    continue
                action_label = _transition_action_label(remapped)
                if action_label not in ("press_back", "press_home") and remapped["target_page"] == canonical_id:
                    continue
                signature = _transition_merge_signature(remapped)
                if signature in seen_transition_signatures:
                    continue
                seen_transition_signatures.add(signature)
                remapped_transitions.append(remapped)

        canonical_page["transitions"] = remapped_transitions
        _ensure_topology_page_identity(canonical_id, canonical_page, source_page_lookup, band_label_lookup)
        collapsed_pages[canonical_id] = canonical_page

    collapsed_root_page_id = canonical_by_page_id.get(root_page_id, root_page_id)
    return collapsed_pages, collapsed_root_page_id


def _compute_page_ahash(image_path: Path) -> str:
    if not image_path.exists():
        return ""
    with Image.open(image_path) as image_handle:
        thumb = image_handle.convert("L").resize((8, 8), Image.LANCZOS)
        pixels = list(thumb.getdata())
    if not pixels:
        return ""
    avg = sum(pixels) / len(pixels)
    bits = "".join("1" if pixel >= avg else "0" for pixel in pixels)
    return f"{int(bits, 2):016x}"


def _ahash_distance(hash_a: str, hash_b: str) -> int:
    if not hash_a or not hash_b:
        return 64
    return (int(hash_a, 16) ^ int(hash_b, 16)).bit_count()


def _topology_page_order_key(page_id: str, pages: Dict[str, dict]) -> Tuple[int, int, str]:
    match = re.match(r"page_(\d+)$", str(page_id))
    numeric = int(match.group(1)) if match else 10**9
    return (int(pages.get(page_id, {}).get("depth", 0)), numeric, str(page_id))


def _page_visual_hash(
    output_dir: Path,
    page_id: str,
    page: dict,
    page_hashes: Dict[str, str],
) -> str:
    if page_id not in page_hashes:
        image_name = str(page.get("image", f"{page_id}.png") or f"{page_id}.png")
        page_hashes[page_id] = _compute_page_ahash(output_dir / "pages" / image_name)
    return page_hashes[page_id]


def _same_page_similarity_metrics(
    output_dir: Path,
    pages: Dict[str, dict],
    page_id_a: str,
    page_id_b: str,
    page_hashes: Dict[str, str],
    structure_cache: Dict[str, List[Tuple[int, int, int, int]]],
) -> Tuple[float, int]:
    if page_id_a not in structure_cache:
        structure_cache[page_id_a] = _page_structure_items(pages[page_id_a])
    if page_id_b not in structure_cache:
        structure_cache[page_id_b] = _page_structure_items(pages[page_id_b])
    similarity = _page_structure_similarity(structure_cache[page_id_a], structure_cache[page_id_b])
    hash_distance = _ahash_distance(
        _page_visual_hash(output_dir, page_id_a, pages[page_id_a], page_hashes),
        _page_visual_hash(output_dir, page_id_b, pages[page_id_b], page_hashes),
    )
    return similarity, hash_distance


def _looks_like_same_page(
    output_dir: Path,
    pages: Dict[str, dict],
    page_id_a: str,
    page_id_b: str,
    page_hashes: Dict[str, str],
    structure_cache: Dict[str, List[Tuple[int, int, int, int]]],
) -> bool:
    similarity, hash_distance = _same_page_similarity_metrics(
        output_dir,
        pages,
        page_id_a,
        page_id_b,
        page_hashes,
        structure_cache,
    )
    return hash_distance <= 3 or (hash_distance <= 6 and similarity >= 0.68)


def _merge_page_metadata(canonical_page: dict, duplicate_page: dict) -> None:
    duplicate_page_id = str(duplicate_page.get("page_id", "") or "")
    if not duplicate_page_id:
        duplicate_page_id = str(duplicate_page.get("image", "") or "").replace(".png", "")
    if not str(canonical_page.get("application_id", "") or "").strip():
        canonical_page["application_id"] = duplicate_page.get("application_id", "")
    if not str(canonical_page.get("application_name", "") or "").strip():
        canonical_page["application_name"] = duplicate_page.get("application_name", "")
    canonical_summary = canonical_page.setdefault("page_summary", {}) or {}
    duplicate_summary = duplicate_page.get("page_summary") or {}
    if not str(canonical_summary.get("application_id", "") or "").strip():
        canonical_summary["application_id"] = duplicate_summary.get("application_id", "") or canonical_page.get("application_id", "")
    if not str(canonical_summary.get("application_name", "") or "").strip():
        canonical_summary["application_name"] = duplicate_summary.get("application_name", "") or canonical_page.get("application_name", "")

    canonical_page["depth"] = min(
        int(canonical_page.get("depth", 0)),
        int(duplicate_page.get("depth", canonical_page.get("depth", 0))),
    )
    canonical_page["trajectory_ids"] = _ordered_unique(
        list(canonical_page.get("trajectory_ids", [])) + list(duplicate_page.get("trajectory_ids", []))
    )
    canonical_page["trajectory_ids_full"] = _ordered_unique(
        list(canonical_page.get("trajectory_ids_full", [])) + list(duplicate_page.get("trajectory_ids_full", []))
    )
    canonical_page["trace_steps"] = _ordered_unique(
        list(canonical_page.get("trace_steps", [])) + list(duplicate_page.get("trace_steps", []))
    )
    canonical_page.setdefault("merged_from_page_ids", [canonical_page["page_id"]])
    canonical_page["merged_from_page_ids"] = _ordered_unique(
        list(canonical_page["merged_from_page_ids"]) + ([duplicate_page_id] if duplicate_page_id else [])
    )
    canonical_page.setdefault("_merged_transition_pool", deepcopy(canonical_page.get("transitions", [])))
    canonical_page["_merged_transition_pool"].extend(deepcopy(duplicate_page.get("transitions", [])))


def _transition_merge_signature(transition: dict) -> str:
    serialized = _serialize_transition(transition)
    signature_payload = {
        "target_page": str(serialized.get("target_page", "") or "").strip(),
        "location": list(_transition_action_location_signature_from_serialized(serialized)),
    }
    if signature_payload["location"] and signature_payload["location"][0] == "fallback":
        fallback_payload = _transition_signature_payload_from_serialized(
            serialized,
            include_coordinates=False,
        )
        fallback_payload.pop("target_page", None)
        signature_payload["fallback"] = fallback_payload
    return json.dumps(signature_payload, sort_keys=True, separators=(",", ":"))


def _should_keep_transition(source_page: dict,
                            transition: dict,
                            page_lookup: Dict[str, dict]) -> bool:
    target_page = str(transition.get("target_page", "") or "")
    if not target_page:
        return False

    action_label = _transition_action_label(transition)
    if action_label in ("press_back", "press_home"):
        return True

    source_page_id = str(source_page.get("page_id", "") or "")
    if target_page == source_page_id:
        return False

    source_app = _page_merge_group_id(source_page)
    if source_app == "launcher":
        return True

    target_page_data = page_lookup.get(target_page)
    if target_page_data is None:
        return False

    source_has_explicit_app = _page_has_explicit_app_identity(source_page)
    target_has_explicit_app = _page_has_explicit_app_identity(target_page_data)
    if not source_has_explicit_app or not target_has_explicit_app:
        return True

    target_app = _page_merge_group_id(target_page_data)
    return bool(target_app) and target_app == source_app


def _remove_duplicate_page_outputs(output_dir: Path, duplicate_page_ids: List[str]) -> None:
    for base_dir, suffix in (
        (output_dir / "pages", ".png"),
        (output_dir / "generated_code", ".py"),
    ):
        if not base_dir.exists():
            continue
        for page_id in duplicate_page_ids:
            path = base_dir / f"{page_id}{suffix}"
            if path.exists():
                path.unlink()


def _page_merge_group_id(page: dict) -> str:
    summary = page.get("page_summary") or {}
    for candidate in (
        page.get("application_id"),
        summary.get("application_id"),
        page.get("application_name"),
        summary.get("application_name"),
    ):
        value = str(candidate or "").strip()
        if value:
            return value
    # If the app identity is missing, keep the page isolated rather than
    # risking an accidental cross-app merge.
    return str(page.get("page_id", "") or "")


def _page_has_explicit_app_identity(page: dict) -> bool:
    summary = page.get("page_summary") or {}
    for candidate in (
        page.get("application_id"),
        summary.get("application_id"),
        page.get("application_name"),
        summary.get("application_name"),
    ):
        if str(candidate or "").strip():
            return True
    return False


def _transition_source_action_signature(transition: dict) -> str:
    serialized = _serialize_transition(transition)
    payload = _transition_signature_payload_from_serialized(
        serialized,
        include_coordinates=False,
    )
    payload.pop("target_page", None)
    payload.pop("icon_bbox", None)
    payload["location"] = list(_transition_action_location_signature_from_serialized(serialized))
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _resolve_page_alias(page_id_alias: Dict[str, str], page_id: str) -> str:
    current = str(page_id or "")
    seen: set[str] = set()
    while current and current not in seen:
        seen.add(current)
        next_page_id = str(page_id_alias.get(current, current) or current)
        if not next_page_id or next_page_id == current:
            break
        current = next_page_id
    return current


def _collapse_pages_with_alias_map(
    pages: Dict[str, dict],
    page_id_alias: Dict[str, str],
) -> Tuple[Dict[str, dict], Dict[str, str]]:
    resolved_alias = {
        page_id: _resolve_page_alias(page_id_alias, page_id)
        for page_id in pages
    }
    ordered_page_ids = sorted(pages, key=lambda page_id: _topology_page_order_key(page_id, pages))
    ordered_canonical_ids: List[str] = []
    members_by_canonical: Dict[str, List[str]] = defaultdict(list)

    for page_id in ordered_page_ids:
        canonical_id = resolved_alias.get(page_id, page_id)
        members_by_canonical[canonical_id].append(page_id)
        if canonical_id not in ordered_canonical_ids:
            ordered_canonical_ids.append(canonical_id)

    canonical_pages: Dict[str, dict] = {}
    for canonical_id in ordered_canonical_ids:
        canonical_page = deepcopy(pages[canonical_id])
        canonical_page["page_id"] = canonical_id
        canonical_page.setdefault("merged_from_page_ids", [canonical_id])
        canonical_page["_merged_transition_pool"] = deepcopy(canonical_page.get("transitions", []))
        for duplicate_id in members_by_canonical.get(canonical_id, [canonical_id]):
            if duplicate_id == canonical_id:
                continue
            _merge_page_metadata(canonical_page, pages[duplicate_id])
        canonical_pages[canonical_id] = canonical_page

    canonical_page_lookup = {
        canonical_id: canonical_pages[canonical_id]
        for canonical_id in ordered_canonical_ids
    }

    merged_pages: Dict[str, dict] = {}
    for canonical_id in ordered_canonical_ids:
        canonical_page = canonical_pages[canonical_id]
        transition_pool = canonical_page.pop("_merged_transition_pool", deepcopy(canonical_page.get("transitions", [])))
        remapped_transitions: List[dict] = []
        seen_transition_signatures: set[str] = set()
        for transition in transition_pool:
            remapped = deepcopy(transition)
            target_page = str(remapped.get("target_page", "") or "")
            source_trace_page = str(remapped.get("source_trace_page", "") or "")
            if target_page:
                remapped["target_page"] = resolved_alias.get(target_page, target_page)
            if source_trace_page:
                remapped["source_trace_page"] = resolved_alias.get(source_trace_page, source_trace_page)
            if not _should_keep_transition(canonical_page, remapped, canonical_page_lookup):
                continue
            signature = _transition_merge_signature(remapped)
            if signature in seen_transition_signatures:
                continue
            seen_transition_signatures.add(signature)
            remapped_transitions.append(remapped)
        canonical_page["transitions"] = remapped_transitions
        merged_pages[canonical_id] = canonical_page

    return merged_pages, resolved_alias


def _dedupe_page_transition_lists(pages: Dict[str, dict]) -> int:
    removed = 0
    for page in pages.values():
        deduped_transitions: List[dict] = []
        seen_transition_signatures: set[str] = set()
        for transition in page.get("transitions", []):
            signature = _transition_merge_signature(transition)
            if signature in seen_transition_signatures:
                removed += 1
                continue
            seen_transition_signatures.add(signature)
            deduped_transitions.append(transition)
        page["transitions"] = deduped_transitions
    return removed


def _merge_pages_by_deterministic_action_targets(
    output_dir: Path,
    pages: Dict[str, dict],
    asset_manifest: List[dict],
    matched_step_rows: List[dict],
    max_iterations: int = 8,
    cleanup_outputs: bool = True,
) -> Tuple[Dict[str, dict], Dict[str, Any]]:
    working_pages = {
        page_id: deepcopy(page)
        for page_id, page in pages.items()
    }
    collapsed_page_ids: set[str] = set()
    iteration_stats: List[dict] = []

    for iteration in range(1, max_iterations + 1):
        ordered_page_ids = sorted(working_pages, key=lambda page_id: _topology_page_order_key(page_id, working_pages))
        parent = {page_id: page_id for page_id in ordered_page_ids}

        def find(page_id: str) -> str:
            current = page_id
            while parent[current] != current:
                parent[current] = parent[parent[current]]
                current = parent[current]
            return current

        def union(page_id_a: str, page_id_b: str) -> None:
            root_a = find(page_id_a)
            root_b = find(page_id_b)
            if root_a == root_b:
                return
            if _topology_page_order_key(root_a, working_pages) <= _topology_page_order_key(root_b, working_pages):
                parent[root_b] = root_a
            else:
                parent[root_a] = root_b

        for source_page_id in ordered_page_ids:
            source_page = working_pages[source_page_id]
            targets_by_action_signature: Dict[Tuple[str, int], List[str]] = defaultdict(list)
            for transition in source_page.get("transitions", []):
                target_page = str(transition.get("target_page", "") or "")
                if not target_page or target_page not in working_pages or target_page == source_page_id:
                    continue
                if _transition_action_label(transition) in ("press_back", "press_home"):
                    continue
                target_page_data = working_pages[target_page]
                target_depth = int(target_page_data.get("depth", 0))
                if target_depth < 2:
                    continue
                if not _should_keep_transition(source_page, transition, working_pages):
                    continue
                action_signature = _transition_source_action_signature(transition)
                targets_by_action_signature[(action_signature, target_depth)].append(target_page)

            for target_page_ids in targets_by_action_signature.values():
                unique_targets = _ordered_unique(target_page_ids)
                if len(unique_targets) < 2:
                    continue
                canonical_target = min(
                    unique_targets,
                    key=lambda page_id: _topology_page_order_key(page_id, working_pages),
                )
                for duplicate_target in unique_targets[1:]:
                    union(canonical_target, duplicate_target)

        page_id_alias = {
            page_id: find(page_id)
            for page_id in ordered_page_ids
        }
        duplicate_ids = sorted(
            page_id
            for page_id, canonical_id in page_id_alias.items()
            if page_id != canonical_id
        )
        if not duplicate_ids:
            break

        previous_page_count = len(working_pages)
        working_pages, resolved_alias = _collapse_pages_with_alias_map(working_pages, page_id_alias)
        for row in asset_manifest:
            page_id = str(row.get("page_id", "") or "")
            if page_id:
                row["page_id"] = resolved_alias.get(page_id, page_id)
        for row in matched_step_rows:
            page_id = str(row.get("page_id", "") or "")
            if page_id:
                row["page_id"] = resolved_alias.get(page_id, page_id)

        collapsed_page_ids.update(duplicate_ids)
        iteration_stats.append({
            "iteration": iteration,
            "pages_before": previous_page_count,
            "pages_after": len(working_pages),
            "collapsed_duplicate_pages": len(duplicate_ids),
        })

    duplicate_transitions_removed = _dedupe_page_transition_lists(working_pages)
    if cleanup_outputs and collapsed_page_ids:
        _remove_duplicate_page_outputs(output_dir, sorted(collapsed_page_ids))

    merge_stats = {
        "original_pages": len(pages),
        "merged_pages": len(working_pages),
        "collapsed_duplicate_pages": len(collapsed_page_ids),
        "duplicate_transitions_removed": duplicate_transitions_removed,
        "iterations": iteration_stats,
    }
    return working_pages, merge_stats


def _merge_duplicate_content_pages(output_dir: Path,
                                   pages: Dict[str, dict],
                                   asset_manifest: List[dict],
                                   matched_step_rows: List[dict],
                                   merge_candidate_page_ids: List[str]) -> Tuple[Dict[str, dict], Dict[str, str], dict]:
    grouped_candidates: Dict[Tuple[str, int], List[str]] = defaultdict(list)
    canonical_pages: Dict[str, dict] = {}
    page_id_alias: Dict[str, str] = {}
    ordered_canonical_ids: List[str] = []
    merge_candidate_set = {str(page_id) for page_id in merge_candidate_page_ids}
    page_hashes: Dict[str, str] = {}
    structure_cache: Dict[str, List[Tuple[int, int, int, int]]] = {}

    for page_id, page in sorted(pages.items(), key=lambda item: _topology_page_order_key(str(item[0]), pages)):
        if page_id not in merge_candidate_set:
            canonical_page = deepcopy(page)
            canonical_page["_merged_transition_pool"] = deepcopy(page.get("transitions", []))
            canonical_page["merged_from_page_ids"] = [page_id]
            canonical_pages[page_id] = canonical_page
            page_id_alias[page_id] = page_id
            ordered_canonical_ids.append(page_id)
            continue

        bucket_key = (_page_merge_group_id(page), int(page.get("depth", 0)))
        matched_canonical_id: Optional[str] = None

        for canonical_id in grouped_candidates.get(bucket_key, []):
            if _looks_like_same_page(
                output_dir,
                pages,
                canonical_id,
                page_id,
                page_hashes,
                structure_cache,
            ):
                matched_canonical_id = canonical_id
                break

        if matched_canonical_id is None:
            canonical_page = deepcopy(page)
            canonical_page["_merged_transition_pool"] = deepcopy(page.get("transitions", []))
            canonical_page["merged_from_page_ids"] = [page_id]
            canonical_pages[page_id] = canonical_page
            page_id_alias[page_id] = page_id
            ordered_canonical_ids.append(page_id)
            grouped_candidates[bucket_key].append(page_id)
            continue

        page_id_alias[page_id] = matched_canonical_id
        _merge_page_metadata(canonical_pages[matched_canonical_id], page)

    duplicate_page_ids = [page_id for page_id, canonical_id in page_id_alias.items() if page_id != canonical_id]
    _remove_duplicate_page_outputs(output_dir, duplicate_page_ids)
    canonical_page_lookup = {
        canonical_id: canonical_pages[canonical_id]
        for canonical_id in ordered_canonical_ids
    }

    merged_pages: Dict[str, dict] = {}
    dropped_cross_app_or_invalid = 0
    for canonical_id in ordered_canonical_ids:
        canonical_page = canonical_pages[canonical_id]
        transition_pool = canonical_page.pop("_merged_transition_pool", deepcopy(canonical_page.get("transitions", [])))
        remapped_transitions: List[dict] = []
        seen_transition_signatures: set[str] = set()
        for transition in transition_pool:
            remapped = dict(transition)
            target_page = str(remapped.get("target_page", "") or "")
            source_trace_page = str(remapped.get("source_trace_page", "") or "")
            if target_page in page_id_alias:
                remapped["target_page"] = page_id_alias[target_page]
            if source_trace_page in page_id_alias:
                remapped["source_trace_page"] = page_id_alias[source_trace_page]
            if not _should_keep_transition(canonical_page, remapped, canonical_page_lookup):
                dropped_cross_app_or_invalid += 1
                continue
            signature = _transition_merge_signature(remapped)
            if signature in seen_transition_signatures:
                continue
            seen_transition_signatures.add(signature)
            remapped_transitions.append(remapped)

        canonical_page["transitions"] = remapped_transitions
        merged_pages[canonical_id] = canonical_page

    for row in asset_manifest:
        page_id = str(row.get("page_id", "") or "")
        if page_id in page_id_alias:
            row["page_id"] = page_id_alias[page_id]

    for row in matched_step_rows:
        page_id = str(row.get("page_id", "") or "")
        if page_id in page_id_alias:
            row["page_id"] = page_id_alias[page_id]

    merge_stats = {
        "original_content_pages": len(pages),
        "merged_content_pages": len(merged_pages),
        "merge_candidate_pages": len(merge_candidate_set),
        "collapsed_duplicate_pages": len(duplicate_page_ids),
        "collapsed_entry_pages": sum(1 for page_id in duplicate_page_ids if page_id in merge_candidate_set),
        "dropped_cross_app_or_invalid_transitions": dropped_cross_app_or_invalid,
    }
    return merged_pages, page_id_alias, merge_stats


def _print_match_summary(drawer_apps: List[DrawerAppSpec],
                         matches_by_app: Dict[str, List[MatchedTrajectory]],
                         max_trajectories_per_app: Optional[int]) -> None:
    print(f"Drawer apps detected: {len(drawer_apps)}")
    for app in drawer_apps:
        all_matches = matches_by_app.get(app.slug, [])
        shown_matches = all_matches[:max_trajectories_per_app] if max_trajectories_per_app is not None else all_matches
        print(f"[{app.label}] matched {len(all_matches)} trajectories")
        if max_trajectories_per_app is not None and len(all_matches) > len(shown_matches):
            print(f"  using first {len(shown_matches)} trajectories due to --max_trajectories_per_app")
        for match in shown_matches:
            print(
                "  - "
                f"trajectory_id={_full_trajectory_id(match)} "
                f"episode={match.episode_id} "
                f"start_step={match.start_step_idx + 1} "
                f"end_step={match.end_step_idx} "
                f"package={match.matched_package}"
            )


def _renumber_page_outputs(output_dir: Path,
                           pages: Dict[str, dict],
                           asset_manifest: List[dict],
                           matched_step_rows: List[dict],
                           root_page_id: str) -> Tuple[Dict[str, dict], str, Dict[str, str]]:
    ordered_old_ids = list(pages.keys())
    page_id_map = {old_page_id: f"page_{idx}" for idx, old_page_id in enumerate(ordered_old_ids)}

    def rename_outputs(base_dir: Path, suffix: str) -> None:
        if not base_dir.exists():
            return
        staged_path_by_old_page_id: Dict[str, Path] = {}
        for old_page_id, new_page_id in page_id_map.items():
            old_path = base_dir / f"{old_page_id}{suffix}"
            if not old_path.exists():
                continue
            staged_path = base_dir / f"__tmp__{old_page_id}{suffix}"
            if staged_path.exists():
                staged_path.unlink()
            old_path.rename(staged_path)
            staged_path_by_old_page_id[old_page_id] = staged_path

        for old_page_id, new_page_id in page_id_map.items():
            final_path = base_dir / f"{new_page_id}{suffix}"
            if final_path.exists():
                final_path.unlink()
            source_page_id = old_page_id
            source_staged_path = staged_path_by_old_page_id.get(source_page_id)
            if source_staged_path is None:
                source_page_id = str(pages.get(old_page_id, {}).get("topology_split_from_page_id", "") or "")
                source_staged_path = staged_path_by_old_page_id.get(source_page_id)
            if source_staged_path is None or not source_staged_path.exists():
                continue
            shutil.copyfile(source_staged_path, final_path)

        for staged_path in staged_path_by_old_page_id.values():
            if staged_path.exists():
                staged_path.unlink()

    rename_outputs(output_dir / "pages", ".png")
    rename_outputs(output_dir / "generated_code", ".py")

    renumbered_pages: Dict[str, dict] = {}
    for old_page_id in ordered_old_ids:
        page = dict(pages[old_page_id])
        new_page_id = page_id_map[old_page_id]
        page["page_id"] = new_page_id
        page["image"] = f"{new_page_id}.png"

        remapped_transitions = []
        for transition in page.get("transitions", []):
            remapped = dict(transition)
            target_page = remapped.get("target_page")
            source_trace_page = remapped.get("source_trace_page")
            if target_page in page_id_map:
                remapped["target_page"] = page_id_map[target_page]
            if source_trace_page in page_id_map:
                remapped["source_trace_page"] = page_id_map[source_trace_page]
            remapped_transitions.append(remapped)
        page["transitions"] = remapped_transitions
        renumbered_pages[new_page_id] = page

    for row in asset_manifest:
        page_id = row.get("page_id")
        if page_id in page_id_map:
            row["page_id"] = page_id_map[page_id]

    for row in matched_step_rows:
        page_id = row.get("page_id")
        if page_id in page_id_map:
            row["page_id"] = page_id_map[page_id]

    return renumbered_pages, page_id_map.get(root_page_id, root_page_id), page_id_map


def build_unified_graph(args) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectory_limit = args.max_trajectories_per_app if int(args.max_trajectories_per_app) > 0 else None

    layout_config = launcher_mock._load_layout_config(Path(args.layout_config))
    home_layout, drawer_layout, launcher_meta = _save_mock_pages(output_dir, layout_config)
    drawer_apps = _extract_drawer_apps(layout_config, drawer_layout)
    matches_by_app = _scan_matching_annotations(drawer_apps, args.annotations_dir, args.include_post_app_steps)
    _write_match_report(output_dir, drawer_apps, matches_by_app)
    _print_match_summary(drawer_apps, matches_by_app, trajectory_limit)

    if args.scan_only:
        total_matches = sum(len(matches_by_app.get(app.slug, [])) for app in drawer_apps)
        print(f"scan_only: apps={len(drawer_apps)} matched_trajectories={total_matches}")
        return

    _ensure_compose_modules()
    client = action_compose.load_api_client()
    model_name = args.model_name
    yolo_model, ocr_reader = action_compose.load_detection_models(args.weights_dir, args.gpu)

    pages: Dict[str, dict] = {}
    first_page_candidates_by_app: Dict[str, List[str]] = defaultdict(list)
    asset_manifest: List[dict] = []
    matched_step_rows: List[dict] = []

    all_app_pages: List[dict] = []

    for app in drawer_apps:
        app_matches = matches_by_app.get(app.slug, [])
        if trajectory_limit is not None:
            app_matches = app_matches[:trajectory_limit]
        if not app_matches:
            print(f"\n[App] {app.label}: no matched trajectories to compose")
            continue

        print(f"\n[App] {app.label}: composing {len(app_matches)} trajectories")

        composed_rows_by_episode: Dict[str, List[dict]] = {}
        for match_idx, match in enumerate(app_matches, start=1):
            print(
                f"  [{match_idx}/{len(app_matches)}] "
                f"trajectory_id={_full_trajectory_id(match)} "
                f"episode={match.episode_id} "
                f"start_step={match.start_step_idx + 1} "
                f"end_step={match.end_step_idx}"
            )
            page_rows, manifest_rows, matched_rows = _compose_segment_pages(
                match=match,
                app=app,
                app_root_page_id=DRAWER_PAGE_ID,
                home_page_id=HOME_PAGE_ID,
                args=args,
                client=client,
                model_name=model_name,
                yolo_model=yolo_model,
                ocr_reader=ocr_reader,
                output_dir=output_dir,
            )
            if not page_rows:
                print(
                    f"    -> skipped trajectory_id={_full_trajectory_id(match)} "
                    f"episode={match.episode_id} (no composed pages)"
                )
                continue
            composed_rows_by_episode[match.episode_id] = page_rows
            first_page_candidates_by_app[app.slug].append(page_rows[0]["page_id"])
            asset_manifest.extend(manifest_rows)
            matched_step_rows.extend(matched_rows)
            print(
                f"    -> completed trajectory_id={_full_trajectory_id(match)} "
                f"episode={match.episode_id} pages={len(page_rows)}"
            )

        if not composed_rows_by_episode:
            print(f"  -> no usable trajectories remained for {app.label}")
            continue

        for episode_id in sorted(composed_rows_by_episode):
            all_app_pages.extend(composed_rows_by_episode[episode_id])
        print(f"  -> direct_entry_candidates={len(composed_rows_by_episode)} trajectory_pages={sum(len(v) for v in composed_rows_by_episode.values())}")

    content_pages_by_id = {page["page_id"]: page for page in all_app_pages}
    entry_page_candidates = _ordered_unique(
        page_id
        for page_id in content_pages_by_id
    )
    content_pages_by_id, content_page_alias_map, merge_stats = _merge_duplicate_content_pages(
        output_dir=output_dir,
        pages=content_pages_by_id,
        asset_manifest=asset_manifest,
        matched_step_rows=matched_step_rows,
        merge_candidate_page_ids=entry_page_candidates,
    )
    print(
        "Merged duplicate content pages: "
        f"{merge_stats['original_content_pages']} -> {merge_stats['merged_content_pages']} "
        f"(collapsed={merge_stats['collapsed_duplicate_pages']})"
    )

    merged_entry_pages_by_app: Dict[str, List[str]] = {}
    for app in drawer_apps:
        merged_entry_pages_by_app[app.slug] = _ordered_unique(
            content_page_alias_map.get(page_id, page_id)
            for page_id in first_page_candidates_by_app.get(app.slug, [])
            if content_page_alias_map.get(page_id, page_id) in content_pages_by_id
        )

    pages.update(_build_rich_launcher_pages(
        home_layout=home_layout,
        drawer_layout=drawer_layout,
        swipe_transition=launcher_meta["swipe_transition"],
        drawer_apps=drawer_apps,
        app_entry_pages=merged_entry_pages_by_app,
    ))
    for page in content_pages_by_id.values():
        page["depth"] = max(2, int(page.get("depth", 2)))
        pages[page["page_id"]] = page

    pages, deterministic_action_merge_stats = _merge_pages_by_deterministic_action_targets(
        output_dir=output_dir,
        pages=pages,
        asset_manifest=asset_manifest,
        matched_step_rows=matched_step_rows,
    )
    print(
        "Merged deterministic action targets: "
        f"{deterministic_action_merge_stats['original_pages']} -> {deterministic_action_merge_stats['merged_pages']} "
        f"(collapsed={deterministic_action_merge_stats['collapsed_duplicate_pages']}, "
        f"deduped_transitions={deterministic_action_merge_stats['duplicate_transitions_removed']})"
    )

    expanded_pages = _expand_topology_pages_by_trajectory_asset_clusters(
        output_dir=output_dir,
        pages=pages,
        matched_rows=matched_step_rows,
        asset_rows=asset_manifest,
    )
    if len(expanded_pages) != len(pages):
        print(
            "Expanded shared-path pages by future trajectory splits: "
            f"{len(pages)} -> {len(expanded_pages)}"
        )
    pages = expanded_pages

    metadata = {
        "launcher_layout_config": str(args.layout_config),
        "annotations_dir": str(args.annotations_dir),
        "screenshots_dir": str(args.screenshots_dir),
        "element_anno_dir": str(args.element_anno_dir),
        "model_name": model_name,
        "max_trajectories_per_app": trajectory_limit,
        "matched_app_count": sum(1 for app in drawer_apps if merged_entry_pages_by_app.get(app.slug)),
        "matched_trajectory_count": len({row["episode_id"] for row in matched_step_rows}),
        "resolved_icons": launcher_meta.get("resolved_icons", []),
        "content_page_alias_map": content_page_alias_map,
        "content_page_merge_stats": merge_stats,
        "deterministic_action_merge_stats": deterministic_action_merge_stats,
    }

    pages, root_page_id, page_id_map = _renumber_page_outputs(
        output_dir=output_dir,
        pages=pages,
        asset_manifest=asset_manifest,
        matched_step_rows=matched_step_rows,
        root_page_id=HOME_PAGE_ID,
    )
    metadata["page_id_map"] = page_id_map

    _save_ui_structure(output_dir, pages, root_page_id, metadata)
    _write_topology_artifacts(output_dir, pages, root_page_id, metadata=metadata)
    _save_action_debug_overlays(output_dir, pages)
    (output_dir / "trajectory_assets_manifest.json").write_text(json.dumps(asset_manifest, indent=2), encoding="utf-8")
    (output_dir / "matched_steps.json").write_text(json.dumps(matched_step_rows, indent=2), encoding="utf-8")

    print(f"Done: {output_dir}")
    print(f"  pages: {len(pages)}")
    print(f"  assets: {len(asset_manifest)}")
    print(f"  ui_structure.json")
    print(f"  ui_structure_layer.json")
    print(f"  ui_topology.png")


def parse_args():
    parser = argparse.ArgumentParser(description="Build one unified launcher-to-app graph from mock drawer apps + AMEX trajectories")
    parser.add_argument("--layout_config", type=str, default=str(DEFAULT_LAYOUT_CONFIG_PATH),
                        help="Mock launcher layout config used to read app drawer labels/positions")
    parser.add_argument("--annotations_dir", type=str,
                        default="/ext_hdd2/tsyou/AMEX_dataset/AMEX/instruction_anno",
                        help="Directory containing AMEX instruction annotation JSONs")
    parser.add_argument("--screenshots_dir", type=str,
                        default="/ext_hdd2/tsyou/AMEX_dataset/AMEX/screenshot",
                        help="Directory containing AMEX screenshots")
    parser.add_argument("--element_anno_dir", type=str,
                        default="/ext_hdd2/tsyou/AMEX_dataset/AMEX/element_anno",
                        help="Directory containing AMEX element annotation JSONs")
    parser.add_argument("--weights_dir", type=str,
                        default="/ext_hdd2/nhkoh/OmniParser/weights",
                        help="OmniParser weights directory")
    parser.add_argument("--output_dir", type=str,
                        default="data_engine/mock_unified_app_graph_4_simple_applications",
                        help="Directory where the unified environment will be saved")
    parser.add_argument("--model_name", type=str, default="gpt-5-mini-2025-08-07",
                        help="OpenAI model used for page styling composition")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index for OmniParser/EasyOCR")
    parser.add_argument("--max_trajectories_per_app", type=int, default=2,
                        help="Limit matched trajectories per app for testing. Use 0 for all trajectories.")
    parser.add_argument("--include_post_app_steps", action="store_true",
                        help="Keep the full remainder of a trajectory after the app first appears")
    parser.add_argument("--scan_only", action="store_true",
                        help="Only scan annotations and write matched_annotations.json without GPT/detection")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_unified_graph(args)


if __name__ == "__main__":
    main()
