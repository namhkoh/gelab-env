"""
Sim2Real Compose Pipeline (Stages 3-4): Detection-guided page composition.

Takes AMEX trajectories, detects UI elements with OmniParser (YOLO+OCR),
and uses GPT-5-mini to compose GE-Lab pages (252x448 phone ratio) from the actual cropped elements.

Pipeline:
  Stage 1 (Detect): YOLO + OCR detect UI elements on each screenshot
  Stage 2 (Crop): Crop actual icons/text from the screenshot
  Stage 3 (Compose): GPT-5-mini arranges cropped elements on canvas (252x448 phone ratio)
  Stage 4 (Structure): Build ui_structure.json + ui_structure_layer.json with AMEX action coordinates

Prerequisites:
    - OmniParser weights at /ext_hdd2/nhkoh/OmniParser/weights/
    - AMEX annotations + screenshots downloaded
    - OPENAI_API_KEY environment variable set

Usage:
    export OPENAI_API_KEY="sk-..."
    python data_engine/sim2real_compose.py \
    --trajectory_id 2024_3_18_17_19_e8ba0101cbc74242b48af70a57dafdf5 \
    --output_dir data_engine/sim2real_envs/trajectory_001 \
    --gpu 0


Outputs:
    - `pages/`: rendered GE-Lab page PNGs
    - `generated_code/`: saved GPT styling code + deterministic paste code per page
    - `extracted_assets/`: cropped UI assets extracted from each AMEX screenshot
    - `trajectory_assets_manifest.json`: manifest for extracted assets
    - `ui_structure.json`: minimal page structure with stored action and action_coord
    - `ui_structure_layer.json`: layer/tree view of the same page graph
    - `action_coord/`: overlay images for inspecting stored action points

If `--trajectory_id` is omitted, one output subdirectory is created per AMEX episode.

"""

import argparse
import base64
import json
import os
import random
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from openai import OpenAI
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from ultralytics import YOLO

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# The main change here are:
# 1) AMEX annotation/screenshot loading and metadata handling.
# 2) Action-coordinate preservation from raw AMEX touch/lift annotations.

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
OUTPUT_CANVAS_SIZE = (1080, 2400)  # Portrait canvas without side gutters
OUTPUT_W, OUTPUT_H = OUTPUT_CANVAS_SIZE
ICON_SIZE = 50
NAV_BAR_HEIGHT = 45
HEADER_HEIGHT = 45
STATUS_BAR_HEIGHT = 20
MARGIN = 10

# Colors
BG_DARK = (25, 25, 35)
BG_WHITE = (250, 250, 250)
HEADER_BLUE = (35, 47, 62)
NAV_BAR_COLOR = (40, 40, 50)
BUTTON_YELLOW = (255, 216, 20)
BUTTON_ORANGE = (255, 164, 28)
BUTTON_GREEN = (76, 175, 80)
TEXT_WHITE = (255, 255, 255)
TEXT_BLACK = (30, 30, 30)
TEXT_GRAY = (130, 130, 130)
PRIME_BLUE = (0, 100, 200)

# ---------------------------------------------------------------------------
# OmniParser Detection: YOLO + OCR per screenshot
# ---------------------------------------------------------------------------

_yolo_model = None
_ocr_reader = None


def load_detection_models(weights_dir: str = "/ext_hdd2/nhkoh/OmniParser/weights",
                          gpu: int = 0):
    """Load YOLO icon detector and EasyOCR reader."""
    global _yolo_model, _ocr_reader
    if _yolo_model is None:
        model_path = os.path.join(weights_dir, "icon_detect", "model.pt")
        _yolo_model = YOLO(model_path)
        print(f"YOLO loaded: {model_path}")
    if _ocr_reader is None:
        import easyocr
        _ocr_reader = easyocr.Reader(["en"], gpu=(gpu >= 0))
        print("EasyOCR loaded.")
    return _yolo_model, _ocr_reader


def detect_and_crop(screenshot_path: str, yolo_model, ocr_reader,
                    conf_threshold: float = 0.15) -> Tuple[List[dict], Tuple[int, int]]:
    """Detect UI elements on a screenshot and crop them.

    Returns (elements, (width, height)) where each element has:
        index, label, bbox [x1,y1,x2,y2] in pixels, crop (PIL Image), type
    """
    img = Image.open(screenshot_path).convert("RGB")
    w, h = img.size
    img_np = np.array(img)

    elements = []

    # --- YOLO icon detection ---
    results = yolo_model(img_np, conf=conf_threshold, iou=0.3, verbose=False)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    confs = results[0].boxes.conf.cpu().numpy()

    # --- OCR text detection ---
    ocr_results = ocr_reader.readtext(img_np)
    ocr_items = []
    for bbox_pts, text, conf in ocr_results:
        if len(text.strip()) < 2:
            continue
        x1 = int(min(p[0] for p in bbox_pts))
        y1 = int(min(p[1] for p in bbox_pts))
        x2 = int(max(p[0] for p in bbox_pts))
        y2 = int(max(p[1] for p in bbox_pts))
        ocr_items.append({"text": text.strip(), "bbox": [x1, y1, x2, y2], "conf": conf})

    # --- Process YOLO detections, label with nearest OCR ---
    for i, (box, conf) in enumerate(zip(boxes, confs)):
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        crop = img.crop((max(0, x1), max(0, y1), min(w, x2), min(h, y2)))
        if crop.size[0] < 5 or crop.size[1] < 5:
            continue

        # Find nearest OCR text for label
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        label = f"icon_{i}"
        best_dist = float("inf")
        for ocr in ocr_items:
            ox1, oy1, ox2, oy2 = ocr["bbox"]
            ocx, ocy = (ox1 + ox2) / 2, (oy1 + oy2) / 2
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

    # --- Add OCR-only text elements (not overlapping YOLO) ---
    for ocr in ocr_items:
        ox1, oy1, ox2, oy2 = ocr["bbox"]
        overlaps = False
        for elem in elements:
            ex1, ey1, ex2, ey2 = elem["bbox"]
            # Simple overlap check
            if ox1 < ex2 and ox2 > ex1 and oy1 < ey2 and oy2 > ey1:
                overlaps = True
                break
        if not overlaps:
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


def _clip_bbox_to_image(bbox: List[int], image_size: Tuple[int, int]) -> Optional[List[int]]:
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


def _annotation_label(entry: dict, fallback: str) -> str:
    candidates = list(entry.get("xml_desc") or [])
    functionality = str(entry.get("functionality", "") or "").strip()
    if functionality:
        candidates.append(functionality)
    for candidate in candidates:
        cleaned = " ".join(str(candidate or "").strip().split())
        if cleaned:
            return cleaned[:80]
    return fallback


def _is_generic_element_label(label: str) -> bool:
    normalized = re.sub(r"[^0-9a-z]+", " ", str(label or "").lower()).strip()
    return (
        not normalized
        or normalized == "unknown"
        or normalized.startswith("icon ")
        or normalized.startswith("clickable ")
        or normalized.startswith("element ")
    )


def _load_clickable_elements_from_element_anno(screenshot_path: str,
                                               screenshot_name: str,
                                               element_anno_dir: str) -> List[dict]:
    if not element_anno_dir:
        return []
    anno_path = os.path.join(element_anno_dir, f"{Path(screenshot_name).stem}.json")
    if not os.path.exists(anno_path):
        return []

    try:
        with open(anno_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return []

    with Image.open(screenshot_path) as img_handle:
        image = img_handle.convert("RGB")

    clickable_elements = []
    for idx, entry in enumerate(payload.get("clickable_elements") or []):
        bbox = _clip_bbox_to_image(entry.get("bbox") or [], image.size)
        if bbox is None:
            continue
        label = _annotation_label(entry, f"clickable_{idx:02d}")
        clickable_elements.append({
            "index": idx,
            "label": label,
            "bbox": bbox,
            "crop": image.crop(tuple(bbox)),
            "type": "clickable",
            "conf": 1.0,
            "bbox_source": "element_anno",
            "element_anno_path": anno_path,
        })
    return clickable_elements


def _prioritize_element_anno_bboxes(detected_elements: List[dict],
                                    screenshot_path: str,
                                    screenshot_name: str,
                                    element_anno_dir: str) -> Tuple[List[dict], dict]:
    clickable_elements = _load_clickable_elements_from_element_anno(
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
            iou = _bbox_iou(elem_bbox, clickable_bbox)
            distance = _bbox_center_distance(elem_bbox, clickable_bbox)
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
            if _is_generic_element_label(updated.get("label", "")):
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
            if _bbox_iou(clickable_bbox, existing_bbox) >= 0.55:
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

    return merged_elements, {
        "loaded": len(clickable_elements),
        "matched": matched,
        "added": added,
    }


def format_element_list(elements: List[dict], orig_size: Tuple[int, int]) -> str:
    """Format detected elements as text for the GPT prompt."""
    w, h = orig_size
    lines = []
    for e in elements:
        x1, y1, x2, y2 = e["bbox"]
        ew, eh = x2 - x1, y2 - y1
        lines.append(
            f"  [{e['index']}] type={e['type']} label=\"{e['label']}\" "
            f"pos=({x1},{y1}) size={ew}x{eh}"
        )
    return "\n".join(lines)


def _sanitize_filename(text: str, fallback: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z._-]+", "_", (text or "").strip()).strip("_")
    return cleaned[:40] if cleaned else fallback


def _unique_layout_name(label: str,
                        elem_type: str,
                        index: int,
                        counts: Dict[str, int]) -> str:
    base = _sanitize_filename(label, f"{elem_type}_{index:02d}")
    suffix = counts.get(base, 0)
    counts[base] = suffix + 1
    return base if suffix == 0 else f"{base}_{suffix}"


def _persist_extracted_assets(elements: List[dict], screenshot_name: str,
                              assets_dir: str, step_info: dict) -> List[dict]:
    """Persist extracted trajectory assets to disk and return asset-backed elements."""
    page_asset_dir = os.path.join(
        assets_dir,
        f"step_{step_info.get('step_index', 0):02d}_{os.path.splitext(screenshot_name)[0]}",
    )
    os.makedirs(page_asset_dir, exist_ok=True)

    asset_backed = []
    for elem in elements:
        label_stub = _sanitize_filename(elem.get("label", ""), f"elem_{elem['index']:02d}")
        asset_name = f"{elem['index']:02d}_{elem['type']}_{label_stub}.png"
        asset_path = os.path.join(page_asset_dir, asset_name)
        elem["crop"].save(asset_path)

        asset_elem = {k: v for k, v in elem.items() if k != "crop"}
        asset_elem["asset_path"] = asset_path
        asset_elem["asset_source"] = "trajectory_extracted"
        asset_elem["source_screenshot"] = screenshot_name
        asset_backed.append(asset_elem)

    return asset_backed


def _save_asset_manifest(output_dir: str, pages_detection_data: List[dict]):
    """Save one manifest describing all extracted assets used for trajectory composition."""
    manifest = []
    for page in pages_detection_data:
        for elem in page["elements"]:
            manifest.append({
                "page_id": page["page_id"],
                "screenshot": page["screenshot_name"],
                "step_index": page["step"].get("step_index"),
                "type": elem.get("type"),
                "label": elem.get("label"),
                "bbox": elem.get("bbox"),
                "asset_path": elem.get("asset_path"),
                "asset_source": elem.get("asset_source"),
            })

    with open(os.path.join(output_dir, "trajectory_assets_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def _normalize_saved_layout(layout_blob: dict) -> dict:
    """Normalize saved layout blobs into {name: [x1, y1, x2, y2]}."""
    normalized = {}
    if not isinstance(layout_blob, dict):
        return normalized

    for key, value in layout_blob.items():
        bbox = value.get("bbox") if isinstance(value, dict) else value
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        try:
            normalized[str(key)] = [int(round(float(v))) for v in bbox]
        except Exception:
            continue
    return normalized


def _load_existing_ui_pages(output_dir: str) -> dict:
    ui_path = os.path.join(output_dir, "ui_structure.json")
    if not os.path.exists(ui_path):
        return {}
    try:
        with open(ui_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload.get("pages") or {}
    except Exception:
        return {}


def _load_existing_manifest_by_page(output_dir: str) -> Dict[str, List[dict]]:
    manifest_path = os.path.join(output_dir, "trajectory_assets_manifest.json")
    if not os.path.exists(manifest_path):
        return {}

    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except Exception:
        return {}

    grouped: Dict[str, List[dict]] = {}
    for item in manifest or []:
        if not isinstance(item, dict):
            continue
        page_id = str(item.get("page_id", "") or "").strip()
        if not page_id:
            continue
        grouped.setdefault(page_id, []).append(item)
    return grouped


def _extract_layout_from_saved_code(code_path: str) -> dict:
    """Parse layout assignments from a saved generated_code/page_X.py artifact."""
    if not os.path.exists(code_path):
        return {}

    try:
        with open(code_path, "r", encoding="utf-8") as f:
            contents = f.read()
    except Exception:
        return {}

    layout = {}
    pattern = re.compile(r'layout\[(["\'])(.*?)\1\]\s*=\s*\[\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\]')
    for match in pattern.finditer(contents):
        layout[match.group(2)] = [
            int(match.group(3)),
            int(match.group(4)),
            int(match.group(5)),
            int(match.group(6)),
        ]
    return layout


def _load_existing_page_layout(output_dir: str,
                               page_id: str,
                               existing_ui_pages: dict) -> Optional[dict]:
    page_entry = existing_ui_pages.get(page_id) or {}
    layout = _normalize_saved_layout(page_entry.get("layout") or {})
    if layout:
        return layout

    code_path = os.path.join(output_dir, "generated_code", f"{page_id}.py")
    layout = _extract_layout_from_saved_code(code_path)
    return layout or None


def render_native_page(screenshot_path: str,
                       elements: List[dict],
                       orig_size: Tuple[int, int],
                       output_size: Tuple[int, int] = OUTPUT_CANVAS_SIZE
                       ) -> Tuple[Image.Image, dict, List[dict]]:
    """Fit the original screenshot into the output canvas and scale element bboxes with it.

    This preserves the native AMEX visual appearance while still producing a
    GE-Lab-compatible layout dict in output pixel coordinates.
    """
    with Image.open(screenshot_path) as img_handle:
        screenshot = img_handle.convert("RGB")
    page_img, _, _, _ = _fit_image_to_box(screenshot, output_size, BG_WHITE)

    layout = {}
    scaled_elements = []
    counts: Dict[str, int] = {}
    for elem in elements:
        bbox = elem.get("bbox") or []
        if len(bbox) != 4:
            continue
        scaled_bbox = _scale_bbox_to_box(bbox, orig_size, output_size)
        if scaled_bbox[2] - scaled_bbox[0] < 4 or scaled_bbox[3] - scaled_bbox[1] < 4:
            continue
        action_name = _unique_layout_name(
            elem.get("label", ""),
            elem.get("type", "elem"),
            elem.get("index", len(scaled_elements)),
            counts,
        )
        layout[action_name] = scaled_bbox
        scaled_elements.append({
            **elem,
            "action_name": action_name,
            "scaled_bbox": scaled_bbox,
        })

    return page_img, layout, scaled_elements


def render_reconstructed_native_page(
    screenshot_path: str,
    elements: List[dict],
    orig_size: Tuple[int, int],
    output_size: Tuple[int, int] = OUTPUT_CANVAS_SIZE,
) -> Tuple[Image.Image, dict, List[dict]]:
    """Rebuild a page from a screenshot-derived background plus extracted crops.

    The background stays visually native to the original AMEX page, but the
    interactive/text regions are reintroduced explicitly from the persisted crops.
    """
    with Image.open(screenshot_path) as img_handle:
        screenshot = img_handle.convert("RGB")

    fitted_screenshot, _, _, _ = _fit_image_to_box(screenshot, output_size, BG_WHITE)
    blurred_background = fitted_screenshot.filter(ImageFilter.GaussianBlur(radius=14))
    background = fitted_screenshot.copy()

    layout = {}
    scaled_elements = []
    counts: Dict[str, int] = {}
    for elem in elements:
        bbox = elem.get("bbox") or []
        if len(bbox) != 4:
            continue
        scaled_bbox = _scale_bbox_to_box(bbox, orig_size, output_size)
        if scaled_bbox[2] - scaled_bbox[0] < 4 or scaled_bbox[3] - scaled_bbox[1] < 4:
            continue
        action_name = _unique_layout_name(
            elem.get("label", ""),
            elem.get("type", "elem"),
            elem.get("index", len(scaled_elements)),
            counts,
        )
        layout[action_name] = scaled_bbox
        scaled_elements.append({
            **elem,
            "action_name": action_name,
            "scaled_bbox": scaled_bbox,
        })

    # Remove detected regions from the background skeleton before re-pasting crops.
    for elem in sorted(scaled_elements, key=lambda item: (item["scaled_bbox"][2] - item["scaled_bbox"][0]) * (item["scaled_bbox"][3] - item["scaled_bbox"][1]), reverse=True):
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
        if not asset_path or not os.path.exists(asset_path):
            continue
        with Image.open(asset_path) as asset_handle:
            crop = asset_handle.convert("RGBA").resize((width, height), Image.LANCZOS)
        composed.alpha_composite(crop, (bbox[0], bbox[1]))

    return composed.convert("RGB"), layout, scaled_elements


# ---------------------------------------------------------------------------
# OpenAI API client
# ---------------------------------------------------------------------------

_api_client_local = threading.local()


def load_api_client(verbose: bool = True) -> OpenAI:
    """Initialize OpenAI API client."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable not set")
    client = OpenAI(api_key=api_key)
    if verbose:
        print("OpenAI client initialized.")
    return client


def _get_thread_api_client() -> OpenAI:
    client = getattr(_api_client_local, "client", None)
    if client is None:
        client = load_api_client(verbose=False)
        _api_client_local.client = client
    return client


def _encode_image_base64(image_path: str) -> str:
    """Encode an image file to base64 data URI."""
    with open(image_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    # Detect format from extension
    ext = Path(image_path).suffix.lower()
    mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}.get(
        ext.lstrip("."), "image/png"
    )
    return f"data:{mime};base64,{data}"


# GE-Lab nav button style: pink "back" top-left, green "home" top-right
# Dedicated nav strip at top so back/home don't occlude page content.
NAV_BTN_W = 128
NAV_BTN_H = 52
NAV_STRIP_H = NAV_BTN_H + 20
PHONE_CANVAS_H = OUTPUT_H - NAV_STRIP_H
PHONE_CANVAS_W = OUTPUT_W
CANVAS_SIZE = (PHONE_CANVAS_W, PHONE_CANVAS_H)
CANVAS_W, CANVAS_H = CANVAS_SIZE
PHONE_OFFSET_X = 0
PHONE_OFFSET_Y = NAV_STRIP_H
GELAB_BACK_COLOR = (255, 200, 200)  # pink
GELAB_HOME_COLOR = (200, 255, 200)  # green
GELAB_BACK_BBOX = [10, 8, 10 + NAV_BTN_W, 8 + NAV_BTN_H]
GELAB_HOME_BBOX = [OUTPUT_W - 10 - NAV_BTN_W, 8, OUTPUT_W - 10, 8 + NAV_BTN_H]

STYLING_CODE_PROMPT = """\
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

# Backward-compatible alias for the older direct rendering path.
RENDER_CODE_PROMPT = STYLING_CODE_PROMPT

# Legacy JSON prompt (used as fallback)
PAGE_ANALYSIS_PROMPT = """\
You are a UI layout analyst. Given this mobile screenshot, describe the UI layout \
as a JSON object for rendering on a 252x448 portrait canvas.

Output ONLY valid JSON with this schema:
{
  "page_type": "home_screen|app_page|search_page|list_page|detail_page|confirmation_page|settings_page|other",
  "background_color": [R, G, B],
  "header": {
    "visible": true/false,
    "text": "header title",
    "color": [R, G, B],
    "has_back_button": true/false,
    "has_search": true/false,
    "search_placeholder": "Search text..."
  },
  "components": [
    {"type": "icon_grid", "columns": 4, "icons": [{"label": "AppName"}]},
    {"type": "search_bar", "placeholder": "Search..."},
    {"type": "text_list", "items": ["item1", "item2"]},
    {"type": "product_card", "title": "...", "price": "$X.XX", "has_prime": true},
    {"type": "button", "text": "Button text", "color": "yellow|orange|green|blue|gray"},
    {"type": "text_block", "text": "...", "style": "title|subtitle|body|caption"},
    {"type": "keyboard"},
    {"type": "image_placeholder", "label": "product image"},
    {"type": "category_tabs", "tabs": ["Tab1", "Tab2"]},
    {"type": "divider"}
  ],
  "interactive_elements": [
    {"id": "element_name", "action": "CLICK|TEXT|SCROLL"}
  ]
}

Rules:
- Identify the main app/page being shown
- List only the most prominent UI components (max 8-10)
- For icon grids (like home screens), list the visible app icons by name
- For product pages, capture title, price, and key buttons
- Use descriptive labels for icons (e.g., "Chrome", "Instagram", not "icon_1")
- Do NOT include markdown formatting, only raw JSON"""


def _query_gpt(client: OpenAI, model_name: str, image_path: str, prompt: str) -> str:
    """Send image+prompt to GPT and return text response."""
    image_uri = _encode_image_base64(image_path)

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a Python PIL code generator. You ALWAYS respond with a ```python code block. Never refuse. Never explain — only output code."},
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_uri}},
            ]}
        ],
        max_completion_tokens=4096,
    )
    choice = response.choices[0]
    content = choice.message.content
    if content is None:
        # Log refusal / filter reason
        finish = choice.finish_reason
        refusal = getattr(choice.message, "refusal", None)
        print(f"\n  GPT empty response: finish={finish}, refusal={refusal}")
        return ""
    return content.strip()


def generate_page_code(client: OpenAI, model_name: str,
                       image_path: str, elements: List[dict],
                       orig_size: Tuple[int, int],
                       step_info: dict = None,
                       max_retries: int = 3) -> Optional[str]:
    """Ask GPT to generate PIL rendering code using detected elements."""
    element_list = format_element_list(elements, orig_size)

    # Fill in the template
    prompt = RENDER_CODE_PROMPT.replace("{{orig_w}}", str(orig_size[0]))
    prompt = prompt.replace("{{orig_h}}", str(orig_size[1]))
    prompt = prompt.replace("{{element_list}}", element_list)

    # Add trajectory context
    if step_info:
        context = ""
        desc = step_info.get("description", "")
        instruction = step_info.get("low_level_instruction", "")
        if desc:
            context += f"\nScreen description: {desc}"
        if instruction:
            context += f"\nUser action on this screen: {instruction}"
        if context:
            prompt += f"\n\nAdditional context:{context}"

    # Always prefix with forceful instruction (GPT-5-mini often returns empty without it)
    prompt = (
        "You MUST respond with ONLY a ```python code block. "
        "No explanations, no markdown besides the code block.\n\n"
        + prompt
    )

    for attempt in range(max_retries):
        try:
            response = _query_gpt(client, model_name, image_path, prompt)
            code = _extract_code_block(response)
            if code:
                return code
            if attempt == 0:
                _log_failed_response(response, image_path)
                print(f" [no code, retry]", end="", flush=True)
            else:
                print(f" [retry {attempt+1}]", end="", flush=True)
        except Exception as e:
            print(f"\n  API error: {e}")
            if attempt == max_retries - 1:
                return None
    return None


def _log_failed_response(response: str, image_path: str):
    """Log failed GPT responses for debugging."""
    log_dir = os.path.join(os.path.dirname(image_path) or ".", ".debug_logs")
    os.makedirs(log_dir, exist_ok=True)
    basename = os.path.splitext(os.path.basename(image_path))[0]
    log_path = os.path.join(log_dir, f"{basename}_failed.txt")
    try:
        with open(log_path, "w") as f:
            f.write(f"RESPONSE LENGTH: {len(response)}\n")
            f.write(f"RESPONSE REPR: {repr(response[:500])}\n\n")
            f.write(response)
    except Exception:
        pass


def _extract_code_block(response: str) -> Optional[str]:
    """Extract python code block from VLM response."""
    # Try ```python ... ``` block
    match = re.search(r"```python\s*(.*?)```", response, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Try ``` ... ``` block
    match = re.search(r"```\s*(.*?)```", response, re.DOTALL)
    if match:
        code = match.group(1).strip()
        # Verify it looks like drawing code
        if "draw." in code or "canvas." in code or "get_crop" in code:
            return code

    # If response looks like code (has draw. or canvas. calls), extract code lines
    if "draw." in response or "canvas." in response or "get_crop" in response:
        # Filter to lines that look like code (strip prose)
        lines = response.split("\n")
        code_lines = []
        in_code = False
        for line in lines:
            stripped = line.strip()
            # Skip empty lines and comments at start
            if not stripped:
                if in_code:
                    code_lines.append(line)
                continue
            # Detect code-like lines
            if (stripped.startswith(("draw.", "canvas.", "layout[", "img", "icon",
                                     "get_crop", "for ", "if ", "#", "x", "y",
                                     "crop", "label", "font", "card", "search",
                                     "nav", "status", "header", "bg", "w ", "h "))
                or "=" in stripped
                or stripped.startswith(("try:", "except"))
                or "paste" in stripped):
                in_code = True
                code_lines.append(line)
            elif in_code and (stripped.startswith(("    ", "\t")) or not stripped[0].isalpha()):
                code_lines.append(line)
            # Skip prose lines
        if code_lines:
            return "\n".join(code_lines).strip()

    return None


def _detect_max_coordinate(code_str: str) -> int:
    """Detect the max coordinate used in generated code (excluding color values)."""
    # Strip fill=(...) and outline=(...) patterns to avoid counting RGB values
    cleaned = re.sub(r'(?:fill|outline)\s*=\s*\([^)]+\)', '', code_str)
    # Strip font= references
    cleaned = re.sub(r'font\s*=\s*\w+', '', cleaned)
    # Strip radius= values
    cleaned = re.sub(r'radius\s*=\s*\d+', '', cleaned)
    # Find all remaining integers
    numbers = [int(n) for n in re.findall(r'\b(\d+)\b', cleaned)]
    return max(numbers) if numbers else CANVAS_H


def _fit_size(src_size: Tuple[int, int], target_size: Tuple[int, int]) -> Tuple[int, int, float, int, int]:
    """Fit src_size inside target_size while preserving aspect ratio."""
    src_w, src_h = src_size
    target_w, target_h = target_size
    scale = min(target_w / max(src_w, 1), target_h / max(src_h, 1))
    fitted_w = max(1, int(round(src_w * scale)))
    fitted_h = max(1, int(round(src_h * scale)))
    offset_x = (target_w - fitted_w) // 2
    offset_y = (target_h - fitted_h) // 2
    return fitted_w, fitted_h, scale, offset_x, offset_y


def _fit_image_to_box(image: Image.Image,
                      target_size: Tuple[int, int],
                      bg_color: Tuple[int, int, int] = BG_WHITE
                      ) -> Tuple[Image.Image, float, int, int]:
    """Resize an image to fill the target canvas without side gutters."""
    resized = image.resize(target_size, Image.LANCZOS)
    canvas = Image.new("RGB", target_size, bg_color)
    canvas.paste(resized, (0, 0))
    return canvas, 1.0, 0, 0


def _scale_bbox_to_box(bbox: List[int],
                       src_size: Tuple[int, int],
                       target_size: Tuple[int, int],
                       base_offset: Tuple[int, int] = (0, 0)) -> List[int]:
    """Scale a bbox with the same full-canvas resize used for the page image."""
    src_w = max(src_size[0], 1)
    src_h = max(src_size[1], 1)
    dst_w = max(target_size[0], 1)
    dst_h = max(target_size[1], 1)
    scale_x = float(dst_w) / float(src_w)
    scale_y = float(dst_h) / float(src_h)
    base_x, base_y = base_offset
    return [
        int(round(bbox[0] * scale_x)) + base_x,
        int(round(bbox[1] * scale_y)) + base_y,
        int(round(bbox[2] * scale_x)) + base_x,
        int(round(bbox[3] * scale_y)) + base_y,
    ]


def render_from_code(code_str: str, elements: List[dict],
                     orig_size: Tuple[int, int] = (720, 1280)
                     ) -> Tuple[Optional[Image.Image], Optional[dict]]:
    """Execute GPT-generated PIL code at original resolution, then resize it to CANVAS_SIZE.

    GPT composes at the original phone resolution (e.g., 720x1280) using the real
    detected coordinates, then we resize it into the GE-Lab viewport without side gutters.
    """
    ow, oh = orig_size
    canvas = Image.new("RGB", (ow, oh), BG_WHITE)
    draw = ImageDraw.Draw(canvas)
    layout = {}

    # Larger fonts for original resolution (will be scaled down with resize)
    font_sm = _try_load_font(12)
    font_md = _try_load_font(18)
    font_lg = _try_load_font(24)
    font_xl = _try_load_font(32)

    def get_crop(index, w=50, h=50):
        """Return an extracted trajectory asset resized to w x h."""
        if 0 <= index < len(elements):
            elem = elements[index]
            asset_path = elem.get("asset_path")
            if asset_path and os.path.exists(asset_path):
                crop = Image.open(asset_path).convert("RGBA")
                return crop.resize((int(w), int(h)), Image.LANCZOS)
            if "crop" in elem:
                crop = elem["crop"].convert("RGBA")
                return crop.resize((int(w), int(h)), Image.LANCZOS)
        ph = Image.new("RGBA", (int(w), int(h)), (200, 200, 200, 255))
        return ph

    # Wrap canvas.paste to auto-convert float coordinates to int
    _real_paste = canvas.paste

    def _safe_paste(im, box=None, mask=None):
        if isinstance(box, (tuple, list)):
            box = tuple(int(v) for v in box)
        if mask is not None:
            _real_paste(im, box, mask)
        else:
            _real_paste(im, box)
    canvas.paste = _safe_paste

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
        # Safe builtins
        "range": range, "len": len, "enumerate": enumerate,
        "min": min, "max": max, "int": int, "float": float, "str": str,
        "True": True, "False": False, "None": None,
        "list": list, "tuple": tuple, "dict": dict,
        "abs": abs, "round": round, "zip": zip, "print": print,
        "getattr": getattr, "setattr": setattr, "hasattr": hasattr,
        "isinstance": isinstance, "type": type, "bool": bool, "set": set,
        "sorted": sorted, "reversed": reversed, "map": map, "filter": filter,
        "sum": sum, "any": any, "all": all, "ord": ord, "chr": chr,
        "TypeError": TypeError, "ValueError": ValueError, "Exception": Exception,
        "KeyError": KeyError, "IndexError": IndexError,
        "random": random,
        "math": __import__("math"),
    }

    code_str = _sanitize_code(code_str)

    try:
        exec(code_str, namespace)
    except Exception as e:
        print(f"\n  Code execution error: {e}")
        fixed = _try_fix_code(code_str, str(e))
        if fixed and fixed != code_str:
            try:
                canvas = Image.new("RGB", (ow, oh), BG_WHITE)
                draw = ImageDraw.Draw(canvas)
                layout.clear()
                namespace["canvas"] = canvas
                namespace["draw"] = draw
                exec(fixed, namespace)
                print(f" (fixed)")
            except Exception as e2:
                print(f"\n  Retry also failed: {e2}")
                return None, None
        else:
            return None, None

    canvas_resized, _, _, _ = _fit_image_to_box(canvas, CANVAS_SIZE, BG_WHITE)

    scaled_layout = {}
    for key, bbox in layout.items():
        scaled_layout[key] = _scale_bbox_to_box(bbox, (ow, oh), CANVAS_SIZE)

    # back/home are added by compose_page in the nav strip (not here)
    scaled_layout.pop("back", None)
    scaled_layout.pop("home", None)

    return canvas_resized, scaled_layout


def _sanitize_code(code_str: str) -> str:
    """Pre-process GPT code to remove/fix common issues before exec."""
    lines = code_str.split("\n")
    cleaned = []
    for line in lines:
        stripped = line.strip()
        # Remove import statements
        if stripped.startswith("import ") or stripped.startswith("from "):
            cleaned.append(f"# REMOVED: {line}")
            continue
        cleaned.append(line)
    code_str = "\n".join(cleaned)

    # Fix deprecated draw.textsize() -> draw.textlength() (Pillow 10+)
    # Common pattern: w, h = draw.textsize(text, font=font) or draw.textsize(text)
    code_str = re.sub(
        r'draw\.textsize\(([^)]+)\)',
        r'(draw.textlength(\1), 20)',  # approximate height as 20
        code_str
    )

    # Fix draw.rectangle(..., radius=N) -> draw.rounded_rectangle(..., radius=N)
    code_str = re.sub(
        r'draw\.rectangle\(([^)]*?),\s*radius\s*=',
        r'draw.rounded_rectangle(\1, radius=',
        code_str
    )

    return code_str


def _try_fix_code(code_str: str, error_msg: str) -> Optional[str]:
    """Try to fix common code execution errors."""
    lines = code_str.split("\n")
    fixed_lines = []
    for line in lines:
        # Remove lines that reference undefined names
        if "name" in error_msg and "is not defined" in error_msg:
            bad_name = error_msg.split("'")[1] if "'" in error_msg else ""
            if bad_name and bad_name in line and not line.strip().startswith("#"):
                fixed_lines.append(f"# REMOVED: {line}")
                continue
        fixed_lines.append(line)
    return "\n".join(fixed_lines)


def analyze_screenshot(client: OpenAI, model_name: str,
                       image_path: str, step_info: dict = None) -> dict:
    """Fallback: Use GPT to analyze a screenshot and output a JSON page spec."""
    context = ""
    if step_info:
        desc = step_info.get("description", "")
        instruction = step_info.get("low_level_instruction", "")
        if desc:
            context += f"\nScreen description: {desc}"
        if instruction:
            context += f"\nUser action on this screen: {instruction}"

    prompt = PAGE_ANALYSIS_PROMPT
    if context:
        prompt += f"\n\nAdditional context:{context}"

    response = _query_gpt(client, model_name, image_path, prompt)
    return _parse_json_response(response)


def _parse_json_response(response: str) -> dict:
    """Extract JSON from VLM response, handling markdown fences and noise."""
    # Try direct parse
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown code block
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Try finding first { ... } block
    brace_start = response.find("{")
    if brace_start >= 0:
        depth = 0
        for i in range(brace_start, len(response)):
            if response[i] == "{":
                depth += 1
            elif response[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(response[brace_start : i + 1])
                    except json.JSONDecodeError:
                        break

    print(f"  WARNING: Failed to parse VLM JSON. Using fallback spec.")
    print(f"  Response preview: {response[:200]}")
    return _fallback_page_spec()


def _fallback_page_spec() -> dict:
    return {
        "page_type": "other",
        "background_color": list(BG_WHITE),
        "header": {"visible": True, "text": "Page", "color": list(HEADER_BLUE),
                    "has_back_button": True, "has_search": False},
        "components": [
            {"type": "text_block", "text": "Content", "style": "body"},
        ],
        "interactive_elements": [],
    }


# ---------------------------------------------------------------------------
# Icon matching: link VLM labels to extracted icon files
# ---------------------------------------------------------------------------

def load_icon_pool(icons_dir: str) -> Dict[str, Image.Image]:
    """Load all extracted icons and index by label."""
    pool = {}
    icons_path = Path(icons_dir)
    for png in icons_path.rglob("*.png"):
        pool[png.stem] = Image.open(png).convert("RGBA")
    return pool


def load_icon_metadata(metadata_path: str) -> List[dict]:
    """Load icons_metadata.json from sim2real.py output."""
    if not os.path.exists(metadata_path):
        return []
    with open(metadata_path) as f:
        return json.load(f)


def find_best_icon(label: str, icon_pool: Dict[str, Image.Image],
                   metadata: List[dict] = None) -> Optional[Image.Image]:
    """Find the best matching icon for a given label using fuzzy matching."""
    label_lower = label.lower().strip()
    if not label_lower or not icon_pool:
        return None

    best_score = 0
    best_key = None

    for key in icon_pool:
        # Extract the label part from icon filename (e.g., "icon_00657_Instagram" -> "instagram")
        parts = key.split("_", 2)
        icon_label = parts[2].lower() if len(parts) > 2 else key.lower()

        if icon_label == "unknown":
            continue

        score = SequenceMatcher(None, label_lower, icon_label).ratio()
        if score > best_score:
            best_score = score
            best_key = key

    # Also check metadata content field
    if metadata:
        for m in metadata:
            content = (m.get("content") or "").lower()
            if not content or content == "unknown":
                continue
            score = SequenceMatcher(None, label_lower, content).ratio()
            if score > best_score:
                best_score = score
                icon_path = m.get("icon_path", "")
                if icon_path:
                    stem = Path(icon_path).stem
                    if stem in icon_pool:
                        best_key = stem

    if best_score >= 0.4 and best_key:
        return icon_pool[best_key]
    return None


def get_random_icon(icon_pool: Dict[str, Image.Image], used: set) -> Optional[Image.Image]:
    """Get a random unused icon from the pool."""
    import random
    available = [k for k in icon_pool if k not in used]
    if not available:
        available = list(icon_pool.keys())
    if not available:
        return None
    key = random.choice(available)
    used.add(key)
    return icon_pool[key]


# ---------------------------------------------------------------------------
# Page Renderer: JSON spec -> portrait PIL image + layout dict
# ---------------------------------------------------------------------------

def _try_load_font(size: int):
    """Try to load a font, fall back to default."""
    for name in ["DejaVuSans.ttf", "FreeSans.ttf", "arial.ttf"]:
        try:
            return ImageFont.truetype(name, size)
        except (IOError, OSError):
            continue
    return ImageFont.load_default()


def _ensure_system_layout(layout: Optional[dict]) -> dict:
    """Ensure every composed page exposes GE-Lab back/home controls in layout."""
    # Action-aware variant: force system controls into layout so PRESS_BACK /
    # PRESS_HOME can resolve to concrete boxes even if the source screenshot
    # did not contain explicit GE-Lab navigation widgets.
    normalized_layout = {}
    for key, bbox in (layout or {}).items():
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            normalized_layout[key] = [int(v) for v in bbox]
    normalized_layout["back"] = list(GELAB_BACK_BBOX)
    normalized_layout["home"] = list(GELAB_HOME_BBOX)
    return normalized_layout


def _draw_system_nav_overlay(image: Image.Image) -> Image.Image:
    """Draw the fixed GE-Lab top navigation strip with back/home buttons."""
    if image.mode != "RGB":
        canvas = image.convert("RGB")
    else:
        canvas = image.copy()

    if canvas.size != OUTPUT_CANVAS_SIZE:
        fitted, _, paste_x, paste_y = _fit_image_to_box(canvas, OUTPUT_CANVAS_SIZE, BG_WHITE)
        resized_canvas = Image.new("RGB", OUTPUT_CANVAS_SIZE, BG_WHITE)
        resized_canvas.paste(fitted, (paste_x, paste_y))
        canvas = resized_canvas

    draw = ImageDraw.Draw(canvas)
    font = _try_load_font(22)

    draw.rectangle([0, 0, OUTPUT_W, NAV_STRIP_H], fill=(245, 245, 248))
    draw.rounded_rectangle(GELAB_BACK_BBOX, radius=12, fill=GELAB_BACK_COLOR, outline=(220, 170, 170), width=2)
    draw.rounded_rectangle(GELAB_HOME_BBOX, radius=12, fill=GELAB_HOME_COLOR, outline=(160, 210, 160), width=2)
    for label, bbox in (("back", GELAB_BACK_BBOX), ("home", GELAB_HOME_BBOX)):
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        text_x = int((bbox[0] + bbox[2] - text_w) / 2)
        text_y = int((bbox[1] + bbox[3] - text_h) / 2) - 1
        draw.text((text_x, text_y), label, fill=TEXT_BLACK, font=font)

    phone_frame = [
        PHONE_OFFSET_X - 1,
        PHONE_OFFSET_Y - 1,
        PHONE_OFFSET_X + CANVAS_W,
        PHONE_OFFSET_Y + CANVAS_H,
    ]
    draw.rounded_rectangle(phone_frame, radius=10, outline=(220, 220, 226), width=1)
    return canvas


def _ensure_system_nav_controls(image: Image.Image, layout: Optional[dict]) -> Tuple[Image.Image, dict]:
    """Finalize a page so the rendered image and layout both include back/home."""
    return _draw_system_nav_overlay(image), _ensure_system_layout(layout)


def render_page(spec: dict, icon_pool: Dict[str, Image.Image],
                metadata: List[dict], used_icons: set) -> Tuple[Image.Image, dict]:
    """Render a page specification to the configured portrait canvas.

    Returns (image, layout_dict) where layout_dict maps element IDs to bboxes.
    """
    bg = tuple(spec.get("background_color", list(BG_WHITE)))
    img = Image.new("RGB", CANVAS_SIZE, bg)
    draw = ImageDraw.Draw(img)

    font_sm = _try_load_font(9)
    font_md = _try_load_font(11)
    font_lg = _try_load_font(14)
    font_title = _try_load_font(16)

    layout = {}
    y_cursor = 0

    # --- Status bar ---
    draw.rectangle([0, 0, CANVAS_W, STATUS_BAR_HEIGHT], fill=(20, 20, 25))
    y_cursor = STATUS_BAR_HEIGHT

    # --- Header ---
    header = spec.get("header", {})
    if header.get("visible", False):
        h_color = tuple(header.get("color", list(HEADER_BLUE)))
        draw.rectangle([0, y_cursor, CANVAS_W, y_cursor + HEADER_HEIGHT], fill=h_color)

        # Back button
        if header.get("has_back_button", False):
            bx, by = 8, y_cursor + 10
            draw.text((bx, by), "<", fill=TEXT_WHITE, font=font_lg)
            layout["back"] = [bx, by, bx + 25, by + 25]

        # Title
        title = header.get("text", "")
        if title:
            tx = 40 if header.get("has_back_button") else 10
            draw.text((tx, y_cursor + 13), title[:30], fill=TEXT_WHITE, font=font_md)

        # Search in header
        if header.get("has_search", False):
            sx, sy = 35, y_cursor + 8
            sw, sh = 350, 28
            draw.rounded_rectangle([sx, sy, sx + sw, sy + sh], radius=5, fill=(255, 255, 255))
            placeholder = header.get("search_placeholder", "Search...")
            draw.text((sx + 8, sy + 6), placeholder[:35], fill=TEXT_GRAY, font=font_sm)
            layout["search_bar"] = [sx, sy, sx + sw, sy + sh]

        y_cursor += HEADER_HEIGHT

    # --- Components ---
    components = spec.get("components", [])
    for comp in components:
        if y_cursor >= CANVAS_SIZE[1] - NAV_BAR_HEIGHT - 10:
            break

        ctype = comp.get("type", "")
        remaining = CANVAS_SIZE[1] - NAV_BAR_HEIGHT - y_cursor

        if ctype == "icon_grid":
            y_cursor = _render_icon_grid(
                img, draw, comp, y_cursor, icon_pool, metadata, used_icons, layout, font_sm
            )

        elif ctype == "search_bar":
            if remaining < 40:
                continue
            sx, sy = MARGIN, y_cursor + 5
            sw, sh = CANVAS_SIZE[0] - 2 * MARGIN, 30
            draw.rounded_rectangle([sx, sy, sx + sw, sy + sh], radius=5,
                                   fill=(240, 240, 240), outline=(200, 200, 200))
            placeholder = comp.get("placeholder", "Search...")
            draw.text((sx + 10, sy + 7), placeholder[:40], fill=TEXT_GRAY, font=font_sm)
            layout["search_bar"] = [sx, sy, sx + sw, sy + sh]
            y_cursor = sy + sh + 5

        elif ctype == "text_list":
            items = comp.get("items", [])
            for item in items[:8]:
                if y_cursor >= CANVAS_SIZE[1] - NAV_BAR_HEIGHT - 20:
                    break
                draw.text((MARGIN + 5, y_cursor + 3), str(item)[:45],
                          fill=TEXT_BLACK if bg[0] > 128 else TEXT_WHITE, font=font_sm)
                item_id = str(item).replace(" ", "_")[:20]
                layout[item_id] = [MARGIN, y_cursor, CANVAS_SIZE[0] - MARGIN, y_cursor + 22]
                draw.line([MARGIN, y_cursor + 22, CANVAS_SIZE[0] - MARGIN, y_cursor + 22],
                          fill=(220, 220, 220))
                y_cursor += 24

        elif ctype == "product_card":
            if remaining < 55:
                continue
            y_cursor = _render_product_card(img, draw, comp, y_cursor, layout, font_sm, font_md)

        elif ctype == "button":
            if remaining < 40:
                continue
            y_cursor = _render_button(draw, comp, y_cursor, layout, font_md)

        elif ctype == "text_block":
            if remaining < 20:
                continue
            text = comp.get("text", "")
            style = comp.get("style", "body")
            font = {"title": font_title, "subtitle": font_lg, "body": font_md, "caption": font_sm}.get(style, font_md)
            color = TEXT_BLACK if bg[0] > 128 else TEXT_WHITE
            draw.text((MARGIN + 5, y_cursor + 3), text[:50], fill=color, font=font)
            y_cursor += {"title": 28, "subtitle": 24, "body": 20, "caption": 16}.get(style, 20)

        elif ctype == "keyboard":
            if remaining < 100:
                continue
            y_cursor = _render_keyboard(draw, y_cursor, layout, font_sm)

        elif ctype == "image_placeholder":
            if remaining < 80:
                continue
            label = comp.get("label", "Image")
            ph = min(80, remaining - 10)
            draw.rounded_rectangle(
                [MARGIN, y_cursor + 3, CANVAS_SIZE[0] - MARGIN, y_cursor + ph],
                radius=5, fill=(230, 230, 230), outline=(200, 200, 200)
            )
            draw.text((CANVAS_SIZE[0] // 2 - 20, y_cursor + ph // 2 - 5),
                       label[:20], fill=TEXT_GRAY, font=font_sm)
            layout[label.replace(" ", "_")] = [MARGIN, y_cursor + 3,
                                                CANVAS_SIZE[0] - MARGIN, y_cursor + ph]
            y_cursor += ph + 5

        elif ctype == "category_tabs":
            if remaining < 25:
                continue
            tabs = comp.get("tabs", [])
            tab_w = (CANVAS_SIZE[0] - 2 * MARGIN) // max(len(tabs), 1)
            for i, tab in enumerate(tabs[:5]):
                tx = MARGIN + i * tab_w
                draw.text((tx + 5, y_cursor + 5), tab[:12], fill=PRIME_BLUE, font=font_sm)
            y_cursor += 25

        elif ctype == "divider":
            draw.line([MARGIN, y_cursor + 3, CANVAS_SIZE[0] - MARGIN, y_cursor + 3],
                      fill=(200, 200, 200))
            y_cursor += 6

    # --- Navigation bar (always present) ---
    nav_y = CANVAS_H - NAV_BAR_HEIGHT
    draw.rectangle([0, nav_y, CANVAS_W, CANVAS_H], fill=NAV_BAR_COLOR)

    # Back button
    bx1, by1 = 10, nav_y + 8
    bw, bh = 60, 30
    draw.rounded_rectangle([bx1, by1, bx1 + bw, by1 + bh], radius=4,
                           fill=(255, 200, 200))
    draw.text((bx1 + 10, by1 + 7), "Back", fill=TEXT_BLACK, font=font_sm)
    layout["back"] = [bx1, by1, bx1 + bw, by1 + bh]

    # Home button
    hx1 = CANVAS_W - 70
    draw.rounded_rectangle([hx1, by1, hx1 + bw, by1 + bh], radius=4,
                           fill=(200, 255, 200))
    draw.text((hx1 + 10, by1 + 7), "Home", fill=TEXT_BLACK, font=font_sm)
    layout["home"] = [hx1, by1, hx1 + bw, by1 + bh]

    return img, layout


def _render_icon_grid(img, draw, comp, y_cursor, icon_pool, metadata,
                      used_icons, layout, font) -> int:
    """Render a grid of icons (e.g., home screen app grid)."""
    columns = comp.get("columns", 4)
    icons_spec = comp.get("icons", [])
    if not icons_spec:
        return y_cursor

    cell_w = (CANVAS_SIZE[0] - 2 * MARGIN) // columns
    cell_h = ICON_SIZE + 18  # icon + label
    rows_needed = (len(icons_spec) + columns - 1) // columns

    for i, icon_spec in enumerate(icons_spec):
        row, col = divmod(i, columns)
        if y_cursor + (row + 1) * cell_h >= CANVAS_SIZE[1] - NAV_BAR_HEIGHT:
            break

        if isinstance(icon_spec, str):
            label = icon_spec
        else:
            label = icon_spec.get("label", f"App_{i}")
        cx = MARGIN + col * cell_w + (cell_w - ICON_SIZE) // 2
        cy = y_cursor + row * cell_h + 3

        # Find matching icon or use random
        icon_img = find_best_icon(label, icon_pool, metadata)
        if icon_img is None:
            icon_img = get_random_icon(icon_pool, used_icons)

        if icon_img is not None:
            icon_resized = icon_img.resize((ICON_SIZE, ICON_SIZE), Image.LANCZOS)
            img.paste(icon_resized, (cx, cy), icon_resized if icon_resized.mode == "RGBA" else None)

        # Label below icon
        draw.text((cx, cy + ICON_SIZE + 1), label[:10], fill=TEXT_WHITE if img.getpixel((cx, cy + ICON_SIZE + 5))[0] < 128 else TEXT_BLACK, font=font)

        layout[label.replace(" ", "_")] = [cx, cy, cx + ICON_SIZE, cy + ICON_SIZE]

    total_rows = min(rows_needed, (CANVAS_SIZE[1] - NAV_BAR_HEIGHT - y_cursor) // cell_h)
    return y_cursor + total_rows * cell_h + 5


def _render_product_card(img, draw, comp, y_cursor, layout, font_sm, font_md) -> int:
    """Render a product listing card."""
    title = comp.get("title", "Product")
    price = comp.get("price", "")
    has_prime = comp.get("has_prime", False)

    card_h = 50
    cx, cy = MARGIN, y_cursor + 3
    cw = CANVAS_SIZE[0] - 2 * MARGIN

    draw.rounded_rectangle([cx, cy, cx + cw, cy + card_h], radius=3,
                           fill=(255, 255, 255), outline=(230, 230, 230))

    draw.text((cx + 8, cy + 5), title[:35], fill=TEXT_BLACK, font=font_md)

    if price:
        draw.text((cx + 8, cy + 22), price, fill=(180, 30, 30), font=font_md)

    if has_prime:
        draw.rounded_rectangle([cx + 8, cy + 36, cx + 50, cy + 46],
                               radius=2, fill=PRIME_BLUE)
        draw.text((cx + 12, cy + 37), "Prime", fill=TEXT_WHITE, font=font_sm)

    card_id = title.replace(" ", "_")[:25]
    layout[card_id] = [cx, cy, cx + cw, cy + card_h]
    return cy + card_h + 4


def _render_button(draw, comp, y_cursor, layout, font) -> int:
    """Render a button."""
    text = comp.get("text", "Button")
    color_name = comp.get("color", "gray")
    color_map = {
        "yellow": BUTTON_YELLOW, "orange": BUTTON_ORANGE,
        "green": BUTTON_GREEN, "blue": PRIME_BLUE,
        "gray": (180, 180, 180), "red": (220, 50, 50),
    }
    bg_color = color_map.get(color_name, (180, 180, 180))
    text_color = TEXT_BLACK if color_name in ("yellow", "gray") else TEXT_WHITE

    bx = MARGIN + 20
    bw = CANVAS_SIZE[0] - 2 * MARGIN - 40
    bh = 32

    draw.rounded_rectangle([bx, y_cursor + 3, bx + bw, y_cursor + 3 + bh],
                           radius=5, fill=bg_color)
    # Center text
    draw.text((bx + bw // 2 - len(text) * 3, y_cursor + 10), text[:25],
              fill=text_color, font=font)

    btn_id = text.replace(" ", "_")[:25]
    layout[btn_id] = [bx, y_cursor + 3, bx + bw, y_cursor + 3 + bh]
    return y_cursor + bh + 8


def _render_keyboard(draw, y_cursor, layout, font) -> int:
    """Render a simplified keyboard."""
    kb_y = y_cursor + 3
    draw.rectangle([0, kb_y, CANVAS_W, kb_y + 95], fill=(210, 210, 215))

    rows = ["qwertyuiop", "asdfghjkl", "zxcvbnm"]
    key_h = 22
    key_w = max(16, (CANVAS_W - 20) // 10)
    for r, row in enumerate(rows):
        row_w = len(row) * key_w
        start_x = (CANVAS_W - row_w) // 2
        for c, ch in enumerate(row):
            kx = start_x + c * key_w
            ky = kb_y + 5 + r * (key_h + 4)
            draw.rounded_rectangle([kx, ky, kx + key_w - 4, ky + key_h],
                                   radius=3, fill=(255, 255, 255))
            draw.text((kx + 5, ky + 4), ch, fill=TEXT_BLACK, font=font)

    layout["keyboard"] = [0, kb_y, CANVAS_W, kb_y + 95]
    return kb_y + 100


# ---------------------------------------------------------------------------
# Stage 4: Build structure (ui_structure.json + ui_structure_layer.json)
# ---------------------------------------------------------------------------

def build_structure(pages_data: List[dict], trajectory: dict,
                    output_dir: str) -> dict:
    """Build per-trajectory UI structure files with aligned AMEX action coordinates."""
    # Compared with the AMEX baseline structure builder, this version
    # keeps one AMEX step's action geometry, then injects fallback back/home
    # transitions so GE-Lab always has navigable system actions.
    steps = trajectory.get("steps", [])

    ui_structure = {"pages": {}, "metadata": {
        "source": "amex_sim2real_compose_sft",
        "episode_id": trajectory.get("episode_id", ""),
        "instruction": trajectory.get("instruction", ""),
        "total_pages": len(pages_data),
        "canvas_size": list(OUTPUT_CANVAS_SIZE),
        "phone_canvas_size": list(CANVAS_SIZE),
        "nav_strip_height": NAV_STRIP_H,
    }}

    home_page_id = _detect_home_page_id(pages_data)

    for i, pdata in enumerate(pages_data):
        page_id = pdata["page_id"]
        layout = _ensure_system_layout(pdata["layout"])
        pdata["layout"] = layout
        step = pdata.get("step", steps[i] if i < len(steps) else {})
        orig_size = tuple(pdata.get("orig_size", (720, 1280)))

        layout_typed = {}
        for key, bbox in layout.items():
            layout_typed[key] = {
                "bbox": bbox,
                "type": "system" if key in ("back", "home") else "normal",
            }

        transitions = []
        used_system_targets = set()

        next_page = pdata.get("next_trace_page_id", page_id)
        if next_page:
            transition = _resolve_transition(step, layout, orig_size, next_page)
            transitions.append(transition)
            if transition.get("action") in ("back", "home"):
                used_system_targets.add(transition["action"])

        if i > 0 and "back" not in used_system_targets:
            transitions.append(_build_system_transition(
                raw_action="PRESS_BACK",
                action="back",
                target_page=pages_data[i - 1]["page_id"],
                icon_bbox=layout.get("back", GELAB_BACK_BBOX),
            ))

        if "home" not in used_system_targets:
            transitions.append(_build_system_transition(
                raw_action="PRESS_HOME",
                action="home",
                target_page=home_page_id,
                icon_bbox=layout.get("home", GELAB_HOME_BBOX),
            ))

        ui_structure["pages"][page_id] = {
            "image": f"{page_id}.png",
            "depth": i,
            "layout": layout_typed,
            "transitions": _serialize_transitions_minimal(transitions),
        }
        _save_action_debug_overlay(
            os.path.join(output_dir, "pages", f"{page_id}.png"),
            os.path.join(output_dir, "action_coord", f"{page_id}.png"),
            transitions,
        )

    layer = _build_layer_structure(pages_data, ui_structure["pages"])

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "ui_structure.json"), "w", encoding="utf-8") as f:
        json.dump(ui_structure, f, indent=2)
    with open(os.path.join(output_dir, "ui_structure_layer.json"), "w", encoding="utf-8") as f:
        json.dump(layer, f, indent=2)

    return ui_structure


def _detect_home_page_id(pages_data: List[dict]) -> str:
    """Pick the most likely launcher/root page for a trajectory."""
    launcher_tokens = ("launcher", "home screen", "app drawer")
    for pdata in pages_data:
        step = pdata.get("step", {})
        package_name = str(step.get("package_name", "")).lower()
        desc = " ".join([
            str(step.get("description", "")),
            str(step.get("low_level_instruction", "")),
            str(step.get("info", "")),
            package_name,
        ]).lower()
        if any(token in desc for token in launcher_tokens):
            return pdata["page_id"]
    return pages_data[0]["page_id"] if pages_data else "page_0"


def _bbox_center_point(bbox: List[int]) -> List[int]:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return [0, 0]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    if x1 == x2 == y1 == y2 == 0:
        return [0, 0]
    return [int(round((x1 + x2) / 2.0)), int(round((y1 + y2) / 2.0))]


def _is_valid_point(point: List[int]) -> bool:
    return isinstance(point, (list, tuple)) and len(point) == 2 and not (int(point[0]) == 0 and int(point[1]) == 0)


def _is_valid_bbox(bbox: List[int]) -> bool:
    return isinstance(bbox, (list, tuple)) and len(bbox) == 4 and any(int(v) != 0 for v in bbox)


def _stored_transition_action(transition: dict) -> str:
    """Preserve the raw dataset action string when available."""
    raw_action = _normalize_raw_action_name(transition.get("raw_action", "")).upper()
    return raw_action if raw_action else str(transition.get("action", "") or "")


def _debug_action_name(transition: dict) -> str:
    raw_action = _normalize_raw_action_name(transition.get("raw_action", "")).upper()
    if raw_action == "TASK_COMPLETE":
        return "COMPLETE"
    if raw_action == "TASK_IMPOSSIBLE":
        return "IMPOSSIBLE"
    if raw_action:
        return raw_action.replace("_", " ")
    return str(transition.get("action", "") or "").strip().upper().replace("_", " ")


def _stored_transition_action_coord(transition: dict) -> List[int]:
    """Pick one canvas-space point that best represents the stored action."""
    # Action-coordinate addition: ui_structure.json stores one representative
    # point per transition, so we collapse richer AMEX geometry down to the
    # best click/swipe anchor in final canvas coordinates.
    raw_action = _normalize_raw_action_name(transition.get("raw_action", "")).upper()
    canvas_action_point = _safe_coord_pair(transition.get("canvas_action_point") or [])
    canvas_action_bbox = transition.get("canvas_action_bbox") or [0, 0, 0, 0]
    icon_bbox = transition.get("icon_bbox") or [0, 0, 0, 0]

    if raw_action in ("TYPE", "TEXT", "PRESS_ENTER"):
        return [0, 0]
    if raw_action in ("TASK_COMPLETE", "TASK_IMPOSSIBLE"):
        return [0, 0]

    if raw_action in ("SWIPE", "SCROLL", "TAP", "CLICK"):
        if _is_valid_point(canvas_action_point):
            return [int(canvas_action_point[0]), int(canvas_action_point[1])]
        if _is_valid_bbox(canvas_action_bbox):
            return _bbox_center_point(canvas_action_bbox)

    if raw_action in ("PRESS_BACK", "PRESS_HOME") and _is_valid_bbox(icon_bbox):
        return _bbox_center_point(icon_bbox)

    if _is_valid_bbox(icon_bbox):
        return _bbox_center_point(icon_bbox)
    if _is_valid_point(canvas_action_point):
        return [int(canvas_action_point[0]), int(canvas_action_point[1])]
    if _is_valid_bbox(canvas_action_bbox):
        return _bbox_center_point(canvas_action_bbox)
    return [0, 0]


def _stored_transition_lift_coord(transition: dict) -> List[int]:
    """Preserve the canvas-space end point for swipe/scroll gestures."""
    raw_action = _normalize_raw_action_name(transition.get("raw_action", "")).upper()
    if raw_action not in ("SWIPE", "SCROLL"):
        return [0, 0]
    canvas_lift_coord = _safe_coord_pair(transition.get("canvas_lift_coord") or [])
    if _is_valid_point(canvas_lift_coord):
        return [int(canvas_lift_coord[0]), int(canvas_lift_coord[1])]
    return [0, 0]


def _debug_bbox_for_transition(transition: dict, action_coord: List[int]) -> List[int]:
    raw_action = _normalize_raw_action_name(transition.get("raw_action", "")).upper()
    canvas_action_bbox = transition.get("canvas_action_bbox") or [0, 0, 0, 0]
    icon_bbox = transition.get("icon_bbox") or [0, 0, 0, 0]

    if raw_action in ("TYPE", "TEXT", "PRESS_ENTER"):
        return [0, 0, 0, 0]
    if raw_action in ("TASK_COMPLETE", "TASK_IMPOSSIBLE"):
        return [0, 0, 0, 0]
    if raw_action in ("SWIPE", "SCROLL") and _is_valid_bbox(canvas_action_bbox):
        return [int(v) for v in canvas_action_bbox]
    if _is_valid_bbox(icon_bbox):
        return [int(v) for v in icon_bbox]
    if _is_valid_bbox(canvas_action_bbox):
        return [int(v) for v in canvas_action_bbox]
    if _is_valid_point(action_coord):
        px, py = int(action_coord[0]), int(action_coord[1])
        radius = 8
        return [px - radius, py - radius, px + radius, py + radius]
    return [0, 0, 0, 0]


def _serialize_transition_minimal(transition: dict) -> dict:
    """Keep only the final action plus one canvas-space action coordinate."""
    # This minimal serializer is specific to the action_coord branch; the
    # baseline compose script did not need to preserve action points per page.
    raw_action = _normalize_raw_action_name(transition.get("raw_action", "")).upper()
    icon_bbox = transition.get("icon_bbox", [0, 0, 0, 0])
    if raw_action in ("SWIPE", "SCROLL"):
        icon_bbox = transition.get("canvas_action_bbox", icon_bbox)
    if raw_action in ("TYPE", "TEXT", "PRESS_ENTER"):
        icon_bbox = [0, 0, 0, 0]
    action_coord = _stored_transition_action_coord(transition)
    lift_coord = _stored_transition_lift_coord(transition)
    item = {
        "action": _stored_transition_action(transition),
        "target_page": transition.get("target_page", ""),
    }
    if _is_valid_point(action_coord):
        item["action_coord"] = [int(action_coord[0]), int(action_coord[1])]
    if _is_valid_point(lift_coord):
        item["lift_coord"] = [int(lift_coord[0]), int(lift_coord[1])]
    if isinstance(icon_bbox, (list, tuple)) and len(icon_bbox) == 4 and _is_valid_bbox(icon_bbox):
        item["icon_bbox"] = [int(v) for v in icon_bbox]

    type_text = str(transition.get("type_text", "") or "").strip()
    if raw_action in ("TYPE", "TEXT", "PRESS_ENTER") and type_text:
        item["type_text"] = type_text

    return item


def _serialize_transitions_minimal(transitions: List[dict]) -> List[dict]:
    return [_serialize_transition_minimal(transition) for transition in transitions or []]


def _debug_transition_label(idx: int, transition: dict) -> str:
    action = _debug_action_name(transition)
    raw_action = _normalize_raw_action_name(transition.get("raw_action", "")).upper()
    if raw_action in ("TYPE", "TEXT"):
        type_text = str(transition.get("type_text", "") or "").strip()
        if type_text:
            shortened = type_text if len(type_text) <= 24 else f"{type_text[:21]}..."
            return f"{idx}:{action} {shortened}"
    if raw_action == "PRESS_ENTER":
        type_text = str(transition.get("type_text", "") or "").strip()
        if type_text:
            shortened = type_text if len(type_text) <= 24 else f"{type_text[:21]}..."
            return f"{idx}:{action} {shortened}"
    return f"{idx}:{action}"


def _should_draw_non_spatial_debug_label(transition: dict) -> bool:
    raw_action = _normalize_raw_action_name(transition.get("raw_action", "")).upper()
    return raw_action in ("TYPE", "TEXT", "PRESS_ENTER", "TASK_COMPLETE", "TASK_IMPOSSIBLE")


def _save_action_debug_overlay(page_image_path: str,
                               output_path: str,
                               transitions: List[dict]):
    """Render an overlay showing the stored action point/box for quick inspection."""
    if not os.path.exists(page_image_path):
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with Image.open(page_image_path) as img_handle:
        image = img_handle.convert("RGB")
    draw = ImageDraw.Draw(image)
    font = _try_load_font(12)

    palette = [
        ((230, 57, 70), (255, 230, 233)),
        ((29, 78, 216), (227, 238, 255)),
        ((22, 163, 74), (229, 255, 237)),
        ((217, 119, 6), (255, 245, 224)),
        ((126, 34, 206), (243, 232, 255)),
    ]
    non_spatial_label_y = NAV_STRIP_H + 8

    for idx, transition in enumerate(transitions or []):
        edge_color, label_bg = palette[idx % len(palette)]
        action_coord = _stored_transition_action_coord(transition)
        debug_bbox = _debug_bbox_for_transition(transition, action_coord)
        label = _debug_transition_label(idx, transition)

        if _is_valid_bbox(debug_bbox):
            draw.rectangle(debug_bbox, outline=edge_color, width=3)
        if _is_valid_point(action_coord):
            px, py = int(action_coord[0]), int(action_coord[1])
            draw.rectangle([px - 4, py - 4, px + 4, py + 4], fill=edge_color, outline=(255, 255, 255), width=1)

            text_bbox = draw.textbbox((0, 0), label, font=font)
            text_w = text_bbox[2] - text_bbox[0]
            text_h = text_bbox[3] - text_bbox[1]
            lx = min(max(4, px + 8), max(4, image.size[0] - text_w - 6))
            ly = min(max(4, py - text_h - 8), max(4, image.size[1] - text_h - 6))
            draw.rectangle([lx - 2, ly - 2, lx + text_w + 2, ly + text_h + 2], fill=label_bg, outline=edge_color, width=1)
            draw.text((lx, ly), label, fill=edge_color, font=font)
        elif _should_draw_non_spatial_debug_label(transition):
            text_bbox = draw.textbbox((0, 0), label, font=font)
            text_w = text_bbox[2] - text_bbox[0]
            text_h = text_bbox[3] - text_bbox[1]
            lx = 6
            ly = min(non_spatial_label_y, max(6, image.size[1] - text_h - 6))
            draw.rectangle([lx - 2, ly - 2, lx + text_w + 2, ly + text_h + 2], fill=label_bg, outline=edge_color, width=1)
            draw.text((lx, ly), label, fill=edge_color, font=font)
            non_spatial_label_y = ly + text_h + 8

    image.save(output_path)


def _bbox_iou(box1: List[int], box2: List[int]) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union = area1 + area2 - inter + 1e-8
    return inter / union


def _bbox_center_distance(box1: List[int], box2: List[int]) -> float:
    c1x = (box1[0] + box1[2]) / 2
    c1y = (box1[1] + box1[3]) / 2
    c2x = (box2[0] + box2[2]) / 2
    c2y = (box2[1] + box2[3]) / 2
    return ((c1x - c2x) ** 2 + (c1y - c2y) ** 2) ** 0.5


def _normalize_raw_action_name(raw_action: str) -> str:
    return str(raw_action or "").strip()


def _step_text(step: dict) -> str:
    return " ".join([
        str(step.get("low_level_instruction", "")),
        str(step.get("description", "")),
        str(step.get("intention", "")),
        str(step.get("info", "")),
        str(step.get("type_text", "")),
        str(step.get("package_name", "")),
    ]).lower()


def _normalize_step_point(step: dict,
                          coord: List[int],
                          orig_size: Tuple[int, int]) -> Optional[Tuple[int, int]]:
    """Map AMEX device coordinates into the original screenshot coordinate system."""
    # AMEX-specific: touch/lift coordinates are recorded in device space, so
    # we first undo that scale before projecting into the GE-Lab canvas.
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


def _scale_step_coord_to_canvas(step: dict,
                                coord: List[int],
                                orig_size: Tuple[int, int]) -> List[int]:
    point = _normalize_step_point(step, coord, orig_size)
    if point is None:
        return [0, 0]
    scaled = _scale_bbox_to_box(
        [point[0], point[1], point[0], point[1]],
        orig_size,
        CANVAS_SIZE,
        (PHONE_OFFSET_X, PHONE_OFFSET_Y),
    )
    return [scaled[0], scaled[1]]


def _scale_bbox_from_step_to_canvas(step: dict,
                                    bbox: List[int],
                                    orig_size: Tuple[int, int]) -> List[int]:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return [0, 0, 0, 0]
    if bbox == [0, 0, 0, 0]:
        return [0, 0, 0, 0]

    top_left = _normalize_step_point(step, [bbox[0], bbox[1]], orig_size)
    bottom_right = _normalize_step_point(step, [bbox[2], bbox[3]], orig_size)
    if top_left is None or bottom_right is None:
        return [0, 0, 0, 0]

    normalized_bbox = [
        min(top_left[0], bottom_right[0]),
        min(top_left[1], bottom_right[1]),
        max(top_left[0], bottom_right[0]),
        max(top_left[1], bottom_right[1]),
    ]
    return _scale_bbox_to_box(
        normalized_bbox,
        orig_size,
        CANVAS_SIZE,
        (PHONE_OFFSET_X, PHONE_OFFSET_Y),
    )


def _infer_gesture_direction(step: dict, raw_action: str, orig_size: Tuple[int, int]) -> str:
    raw_action = _normalize_raw_action_name(raw_action)
    if raw_action not in ("SWIPE", "SCROLL"):
        return ""

    touch_point = _normalize_step_point(step, step.get("touch_coord") or [], orig_size)
    lift_point = _normalize_step_point(step, step.get("lift_coord") or [], orig_size)
    if touch_point is not None and lift_point is not None:
        dx = lift_point[0] - touch_point[0]
        dy = lift_point[1] - touch_point[1]
        if abs(dy) >= abs(dx):
            return "down" if dy > 0 else "up"
        return "right" if dx > 0 else "left"

    instruction = _step_text(step)
    for direction in ("down", "up", "left", "right"):
        if f"scroll {direction}" in instruction or f"swipe {direction}" in instruction:
            return direction
    return ""


def _safe_coord_pair(coord: List[int]) -> List[int]:
    if not isinstance(coord, (list, tuple)) or len(coord) != 2:
        return [0, 0]
    return [int(coord[0]), int(coord[1])]


def _point_box(point: Tuple[int, int], radius: int = 12) -> List[int]:
    px, py = point
    return [px - radius, py - radius, px + radius, py + radius]


def _build_action_bbox(step: dict, raw_action: str) -> List[int]:
    raw_action = _normalize_raw_action_name(raw_action)
    touch = _safe_coord_pair(step.get("touch_coord") or [])
    lift = _safe_coord_pair(step.get("lift_coord") or [])

    if raw_action in ("TYPE", "PRESS_BACK", "PRESS_HOME", "PRESS_ENTER", "TASK_COMPLETE", "TASK_IMPOSSIBLE"):
        return [0, 0, 0, 0]
    if touch == [0, 0]:
        return [0, 0, 0, 0]
    if lift == [0, 0]:
        lift = touch
    return [
        min(touch[0], lift[0]),
        min(touch[1], lift[1]),
        max(touch[0], lift[0]),
        max(touch[1], lift[1]),
    ]


def _build_action_point(step: dict, raw_action: str) -> List[int]:
    raw_action = _normalize_raw_action_name(raw_action)
    if raw_action in ("TYPE", "PRESS_BACK", "PRESS_HOME", "PRESS_ENTER", "TASK_COMPLETE", "TASK_IMPOSSIBLE"):
        return [0, 0]
    return _safe_coord_pair(step.get("touch_coord") or [])


def _scale_action_bbox_to_canvas(step: dict,
                                 raw_action: str,
                                 orig_size: Tuple[int, int]) -> List[int]:
    return _scale_bbox_from_step_to_canvas(step, _build_action_bbox(step, raw_action), orig_size)


def _scale_action_point_to_canvas(step: dict,
                                  raw_action: str,
                                  orig_size: Tuple[int, int]) -> List[int]:
    return _scale_step_coord_to_canvas(step, _build_action_point(step, raw_action), orig_size)


def _bbox_contains_point(bbox: List[int], point: Tuple[int, int]) -> bool:
    return bbox[0] <= point[0] <= bbox[2] and bbox[1] <= point[1] <= bbox[3]


def _is_layout_target(target: Optional[str], layout: dict) -> bool:
    return bool(target) and target in layout


def _resolve_tap_target(step: dict,
                        layout: dict,
                        orig_size: Tuple[int, int]) -> Optional[str]:
    point = _scale_step_coord_to_canvas(step, step.get("touch_coord") or [], orig_size)
    if point == [0, 0]:
        return None

    point_tuple = (point[0], point[1])
    for key, bbox in layout.items():
        if _bbox_contains_point(bbox, point_tuple):
            return key
    return None


def _find_closest_layout_key(layout: dict,
                             target_box: List[int],
                             allow_system: bool = False) -> Tuple[Optional[str], float, float]:
    best_key = None
    best_iou = 0.0
    best_distance = float("inf")

    for key, bbox in layout.items():
        if not allow_system and key in ("back", "home"):
            continue
        iou = _bbox_iou(target_box, bbox)
        distance = _bbox_center_distance(target_box, bbox)
        if iou > best_iou or (iou == best_iou and distance < best_distance):
            best_key = key
            best_iou = iou
            best_distance = distance

    return best_key, best_iou, best_distance


def _find_action_target(step: dict, layout: dict,
                        orig_size: Tuple[int, int]) -> str:
    """Find the layout element or gesture most likely targeted by a raw AMEX step."""
    if not layout:
        return "unknown"

    action = _normalize_raw_action_name(step.get("action", "")).upper()
    info = str(step.get("info", ""))
    instruction = _step_text(step)

    if "KEY_HOME" in info or "home screen" in instruction:
        return "home"
    if "go back" in instruction or instruction.startswith("back ") or info == "BACK":
        return "back"

    touch_point = _scale_step_coord_to_canvas(step, step.get("touch_coord") or [], orig_size)
    lift_point = _scale_step_coord_to_canvas(step, step.get("lift_coord") or [], orig_size)

    if action in ("TAP", "CLICK") and touch_point != [0, 0]:
        best_key, best_iou, best_distance = _find_closest_layout_key(
            layout, _point_box((touch_point[0], touch_point[1])), allow_system=True
        )
        if best_key is not None and (best_iou > 0 or best_distance <= 48):
            return best_key

    if action in ("TYPE", "TEXT"):
        for preferred in ("search_bar", "search", "input", "text_field", "keyboard"):
            if preferred in layout:
                return preferred
        if "search" in instruction:
            for key in layout:
                if "search" in key.lower():
                    return key

    if action == "PRESS_ENTER":
        if "search" in instruction:
            for key in layout:
                if "search" in key.lower():
                    return key
        for preferred in ("keyboard", "search_bar", "input", "text_field"):
            if preferred in layout:
                return preferred

    if action in ("SWIPE", "SCROLL"):
        if touch_point != [0, 0] and lift_point != [0, 0]:
            start_box = _point_box((touch_point[0], touch_point[1]), radius=18)
            end_box = _point_box((lift_point[0], lift_point[1]), radius=18)
            best_start, start_iou, start_distance = _find_closest_layout_key(
                layout, start_box, allow_system=False
            )
            best_end, end_iou, end_distance = _find_closest_layout_key(
                layout, end_box, allow_system=False
            )
            if best_start is not None and (start_iou > 0 or start_distance <= 72):
                return best_start
            if best_end is not None and (end_iou > 0 or end_distance <= 72):
                return best_end

        direction = _infer_gesture_direction(step, action, orig_size)
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


def _resolve_transition(step: dict,
                        layout: dict,
                        orig_size: Tuple[int, int],
                        target_page: str) -> dict:
    # Action-coordinate addition: keep both the semantic target label and the
    # raw/canvas AMEX geometry so downstream code can use exact action points.
    raw_action = _normalize_raw_action_name(step.get("action", ""))
    resolved_target = _find_action_target(step, layout, orig_size)
    strict_tap_target = None
    if raw_action in ("TAP", "CLICK"):
        strict_tap_target = _resolve_tap_target(step, layout, orig_size)

    canvas_action_bbox = _scale_action_bbox_to_canvas(step, raw_action, orig_size)
    canvas_action_point = _scale_action_point_to_canvas(step, raw_action, orig_size)
    canvas_lift_coord = _scale_step_coord_to_canvas(step, step.get("lift_coord") or [], orig_size)
    gesture_direction = _infer_gesture_direction(step, raw_action, orig_size)

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

    if raw_action in ("TAP", "CLICK"):
        if strict_tap_target is not None:
            transition["action"] = strict_tap_target
            transition["icon_bbox"] = layout.get(strict_tap_target, [0, 0, 0, 0])
        else:
            transition["action"] = "tap"
            transition["icon_bbox"] = canvas_action_bbox
    elif raw_action in ("TYPE", "TEXT"):
        transition["action"] = resolved_target if _is_layout_target(resolved_target, layout) else "type"
        transition["icon_bbox"] = [0, 0, 0, 0]
    elif raw_action in ("SWIPE", "SCROLL"):
        if _is_layout_target(resolved_target, layout):
            transition["action"] = resolved_target
            transition["icon_bbox"] = canvas_action_bbox
        else:
            transition["action"] = resolved_target if resolved_target.startswith("swipe_") else "swipe"
            transition["icon_bbox"] = canvas_action_bbox
    elif raw_action == "PRESS_ENTER":
        transition["action"] = resolved_target if _is_layout_target(resolved_target, layout) else "press_enter"
        transition["icon_bbox"] = [0, 0, 0, 0]
    elif raw_action == "PRESS_BACK":
        transition["action"] = "back"
        transition["icon_bbox"] = layout.get("back", GELAB_BACK_BBOX)
    elif raw_action == "PRESS_HOME":
        transition["action"] = "home"
        transition["icon_bbox"] = layout.get("home", GELAB_HOME_BBOX)
    elif raw_action == "TASK_COMPLETE":
        transition["action"] = "complete"
        transition["icon_bbox"] = [0, 0, 0, 0]
    elif raw_action == "TASK_IMPOSSIBLE":
        transition["action"] = "impossible"
        transition["icon_bbox"] = [0, 0, 0, 0]

    return transition


def _build_system_transition(raw_action: str,
                             action: str,
                             target_page: str,
                             icon_bbox: List[int]) -> dict:
    # Synthetic system transitions are injected only in this action-aware
    # branch so every page still exposes back/home actions after serialization.
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


def _build_layer_structure(pages_data: List[dict],
                           pages_full: Dict[str, dict]) -> dict:
    """Build layer structure matching GE-Lab ui_structure_layer.json format."""
    if not pages_data:
        return {"root": None, "metadata": {}}

    page_order = [pdata["page_id"] for pdata in pages_data]

    def build_node(page_id, visited):
        visited.add(page_id)
        page_data = pages_full.get(page_id, {})

        all_transitions = page_data.get("transitions", [])
        non_system = [t for t in all_transitions if t.get("action") not in ("back", "home")]

        node = {
            "image": f"{page_id}.png",
            "depth": page_data.get("depth", 0),
            "layout": page_data.get("layout", {}),
            "transitions": non_system,
            "subnodes": [],
        }
        for transition in non_system:
            child_id = transition.get("target_page")
            if child_id in pages_full and child_id not in visited:
                node["subnodes"].append(build_node(child_id, visited))
        return node

    visited = set()
    root_id = page_order[0]
    root = build_node(root_id, visited)
    for page_id in page_order[1:]:
        if page_id not in visited:
            root["subnodes"].append(build_node(page_id, visited))

    return {
        "root": root,
        "metadata": {
            "type": "trajectory",
            "canvas_size": list(OUTPUT_CANVAS_SIZE),
            "phone_canvas_size": list(CANVAS_SIZE),
        },
    }


# ---------------------------------------------------------------------------
# Crop saving: labeled detection output for inspection
# ---------------------------------------------------------------------------

def _save_labeled_crops(elements: List[dict], orig_size: Tuple[int, int],
                        screenshot_path: str, output_dir: str):
    """Save each detected crop as a labeled PNG + annotated overview image."""
    os.makedirs(output_dir, exist_ok=True)

    # Save individual crops
    for e in elements:
        label = e["label"].replace("/", "_").replace(" ", "_")[:30]
        fname = f"{e['index']:02d}_{e['type']}_{label}.png"
        e["crop"].save(os.path.join(output_dir, fname))

    # Save annotated screenshot showing all detections with labels
    img = Image.open(screenshot_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    font = _try_load_font(14)
    for e in elements:
        x1, y1, x2, y2 = e["bbox"]
        color = (0, 255, 0) if e["type"] == "icon" else (255, 255, 0)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        draw.text((x1, max(0, y1 - 16)),
                  f"[{e['index']}] {e['label'][:20]}", fill=color, font=font)
    img.save(os.path.join(output_dir, "_annotated.png"))

    # Save element manifest
    manifest = []
    for e in elements:
        manifest.append({
            "index": e["index"], "label": e["label"], "type": e["type"],
            "bbox": e["bbox"], "conf": round(e["conf"], 3),
        })
    with open(os.path.join(output_dir, "_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)


# ---------------------------------------------------------------------------
# Two-phase composition: GPT styling + deterministic positioning
# ---------------------------------------------------------------------------

def _build_step_context(trajectory: dict, step_idx: int) -> dict:
    """Build trajectory-aware context for one page composition step."""
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

def _generate_position_code(elements: List[dict], orig_size: Tuple[int, int]) -> str:
    """Generate deterministic PIL code that pastes all crops at detected positions.

    This code is auto-generated (not LLM) to guarantee correct positioning.
    """
    ow, oh = orig_size
    lines = ["# --- Auto-generated: paste detected elements at original positions ---"]

    for e in elements:
        x1, y1, x2, y2 = e["bbox"]
        ew, eh = x2 - x1, y2 - y1
        if ew < 5 or eh < 5:
            continue
        idx = e["index"]
        label = e["label"].replace('"', "'").replace(" ", "_").replace("/", "_")[:25]
        asset_comment = e.get("asset_path", "").replace("\\", "/")

        lines.append(
            f'# asset_path: {asset_comment}\n'
            f'try:\n'
            f'    _c{idx} = get_crop({idx}, {ew}, {eh})\n'
            f'    canvas.paste(_c{idx}, ({max(0, x1)}, {max(0, y1)}), _c{idx})\n'
            f'except Exception:\n'
            f'    pass\n'
            f'layout["{label}"] = [{x1}, {y1}, {x2}, {y2}]'
        )

    # back/home drawn separately by compose_page after resize
    return "\n\n".join(lines)


def generate_styling_code(client: OpenAI, model_name: str,
                          image_path: str, elements: List[dict],
                          orig_size: Tuple[int, int],
                          step_info: dict = None,
                          max_retries: int = 3) -> Optional[str]:
    """Ask GPT to generate background/styling code (no element content)."""
    element_list = format_element_list(elements, orig_size)
    prompt = STYLING_CODE_PROMPT.replace("{{orig_w}}", str(orig_size[0]))
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
            response = _query_gpt(client, model_name, image_path, prompt)
            code = _extract_code_block(response)
            if code:
                return _sanitize_code(code)
            if attempt == 0:
                _log_failed_response(response, image_path)
                print(f" [no styling, retry]", end="", flush=True)
        except Exception as e:
            print(f"\n  API error: {e}")
    return None


def compose_page(client: OpenAI, model_name: str,
                 elements: List[dict], orig_size: Tuple[int, int],
                 screenshot_path: str, step_info: dict = None
                 ) -> Tuple[Image.Image, dict, dict]:
    """Compose a GE-Lab page: GPT styling + detected crops + GE-Lab nav.

    Phase 1 (GPT): Generate background/styling on blank canvas — colors,
        status bar, headers, section cards, dividers. No element content.
    Phase 2 (deterministic): Paste real YOLO-detected crops at exact positions.
    Phase 3 (deterministic): Draw GE-Lab back/home buttons at top.

    Returns (image at OUTPUT_CANVAS_SIZE, layout dict with bboxes in OUTPUT_CANVAS_SIZE coords).
    """
    ow, oh = orig_size

    # Phase 1: GPT styling on blank canvas
    styling_source = "gpt"
    styling_code = generate_styling_code(
        client, model_name, screenshot_path, elements, orig_size, step_info
    )
    if styling_code is None:
        bg = _extract_bg_color(elements, screenshot_path)
        styling_code = f"draw.rectangle([0, 0, {ow}, {oh}], fill={bg})"
        styling_source = "fallback_bg"

    # Phase 2: Deterministic crop positioning
    position_code = _generate_position_code(elements, orig_size)

    # Combine: styling first, then crop pastes on top
    full_code = styling_code + "\n\n" + position_code

    # Execute on blank canvas at original resolution
    render_status = "render_from_code"
    page_img, layout = render_from_code(full_code, elements, orig_size)

    if page_img is None:
        # Fallback: just bg color + crops
        page_img, layout = _fallback_compose(elements, orig_size, screenshot_path)
        render_status = "fallback_compose"

    final_canvas = Image.new("RGB", OUTPUT_CANVAS_SIZE, BG_WHITE)
    final_canvas.paste(page_img.convert("RGB"), (PHONE_OFFSET_X, PHONE_OFFSET_Y))

    shifted_layout = {}
    for key, bbox in layout.items():
        if key in ("back", "home"):
            continue  # will be set below
        shifted_layout[key] = [
            bbox[0] + PHONE_OFFSET_X,
            bbox[1] + PHONE_OFFSET_Y,
            bbox[2] + PHONE_OFFSET_X,
            bbox[3] + PHONE_OFFSET_Y,
        ]
    final_canvas, shifted_layout = _ensure_system_nav_controls(final_canvas, shifted_layout)

    code_artifact = {
        "styling_source": styling_source,
        "render_status": render_status,
        "styling_code": styling_code,
        "position_code": position_code,
        "full_code": full_code,
    }
    return final_canvas, shifted_layout, code_artifact


def _save_page_code(code_dir: str, page_id: str, screenshot_name: str,
                    step_info: dict, code_artifact: dict):
    """Persist the PIL code used to build one trajectory page."""
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
        f"# The final runtime image is then rendered into the {OUTPUT_W}x{OUTPUT_H} canvas with a top nav strip.",
    ]

    contents = "\n".join(header_lines) + "\n\n"
    contents += "# --- GPT styling skeleton ---\n"
    contents += code_artifact.get("styling_code", "").strip() + "\n\n"
    contents += "# --- Deterministic element pastes ---\n"
    contents += code_artifact.get("position_code", "").strip() + "\n"

    code_path = os.path.join(code_dir, f"{page_id}.py")
    with open(code_path, "w", encoding="utf-8") as f:
        f.write(contents)


def _compose_page_record(page: dict,
                         pages_dir: str,
                         code_dir: str,
                         model_name: str,
                         should_resume: bool,
                         client: Optional[OpenAI] = None) -> dict:
    """Compose or reuse one page record while keeping page ordering external."""
    page_id = page["page_id"]
    screenshot_name = page["screenshot_name"]
    screenshot_path = page["screenshot_path"]
    orig_size = tuple(page["orig_size"])
    step_context = page["step"]
    elements = page["elements"]
    page_output_path = os.path.join(pages_dir, f"{page_id}.png")
    existing_layout = page.get("existing_layout")

    if should_resume and page.get("reuse_existing_page") and existing_layout and os.path.exists(page_output_path):
        layout = {key: list(bbox) for key, bbox in existing_layout.items()}
        return {
            "message": f"  compose {page_id} -> reuse existing layout ({len(layout)} layout elems)",
            "page_data": {
                "page_id": page_id,
                "layout": layout,
                "orig_size": list(orig_size),
                "step": step_context,
                "next_trace_page_id": page.get("next_trace_page_id", page_id),
            },
        }

    compose_client = client if client is not None else _get_thread_api_client()
    try:
        page_img, layout, code_artifact = compose_page(
            compose_client, model_name, elements, orig_size, screenshot_path, step_context
        )
        page_img, layout = _ensure_system_nav_controls(page_img, layout)
        page_img.save(page_output_path)
        _save_page_code(code_dir, page_id, screenshot_name, step_context, code_artifact)
    except Exception as exc:
        raise RuntimeError(f"compose failed for {page_id}: {exc}") from exc

    return {
        "message": f"  compose {page_id} -> {len(layout)} layout elems",
        "page_data": {
            "page_id": page_id,
            "layout": layout,
            "orig_size": list(orig_size),
            "step": step_context,
            "next_trace_page_id": page.get("next_trace_page_id", page_id),
        },
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def _collect_annotation_jobs(args) -> List[Tuple[str, str]]:
    """Collect one AMEX trajectory or all trajectories in instruction_anno."""
    # AMEX adaptation from the AMEX baseline: enumerate episode JSONs
    # from the instruction annotation directory instead of AMEX exports.
    annotations_dir = Path(args.annotations_dir)
    if not annotations_dir.exists():
        raise FileNotFoundError(f"Annotations directory not found: {annotations_dir}")

    trajectory_id = getattr(args, "trajectory_id", None)
    max_trajectories = getattr(args, "max_trajectories", None)

    if trajectory_id:
        by_filename = annotations_dir / f"{trajectory_id}.json"
        if by_filename.exists():
            return [(trajectory_id, str(by_filename))]

        for annot_path in sorted(annotations_dir.glob("*.json")):
            try:
                with open(annot_path, "r", encoding="utf-8") as f:
                    episode_id = json.load(f).get("episode_id", "")
                if str(episode_id) == trajectory_id:
                    return [(trajectory_id, str(annot_path))]
            except Exception:
                continue
        raise FileNotFoundError(f"Annotation not found for trajectory_id: {trajectory_id}")

    jobs = []
    for annot_path in sorted(annotations_dir.glob("*.json")):
        try:
            with open(annot_path, "r", encoding="utf-8") as f:
                trajectory = json.load(f)
            episode_id = trajectory.get("episode_id") or annot_path.stem
            jobs.append((str(episode_id), str(annot_path)))
        except Exception as exc:
            print(f"SKIP annotation parse failure: {annot_path.name} ({exc})")

    start_index = getattr(args, "start_index", 0)
    if start_index > 0:
        jobs = jobs[start_index:]
    if max_trajectories is not None:
        jobs = jobs[:max_trajectories]
    return jobs


def _resolve_step_screenshot(step: dict,
                             screenshots_dir: str,
                             episode_id: str,
                             step_idx: int) -> Tuple[str, str]:
    # AMEX screenshot names are less uniform than AMEX's fixed export
    # scheme, so we try several filename conventions used in the dataset.
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


def _resolve_trajectory_output_dir(base_output_dir: str,
                                   episode_id: str,
                                   use_subdir: bool) -> str:
    if not use_subdir:
        return base_output_dir
    return os.path.join(base_output_dir, _sanitize_filename(str(episode_id), "trajectory"))


def _count_expected_pages(trajectory: dict, args) -> int:
    episode_id = trajectory.get("episode_id", "unknown_episode")
    count = 0
    for i, step in enumerate(trajectory.get("steps", [])):
        _, screenshot_path = _resolve_step_screenshot(step, args.screenshots_dir, episode_id, i)
        if os.path.exists(screenshot_path):
            count += 1
    return count


def _has_complete_page_outputs(output_dir: str, expected_pages: int) -> bool:
    if expected_pages <= 0:
        return False
    pages_dir = Path(output_dir) / "pages"
    if not pages_dir.exists():
        return False
    for idx in range(expected_pages):
        if not (pages_dir / f"page_{idx}.png").exists():
            return False
    return True


def _process_trajectory(trajectory: dict,
                        output_dir: str,
                        args,
                        client: OpenAI,
                        model_name: str,
                        yolo_model,
                        ocr_reader) -> int:
    steps = trajectory.get("steps", [])
    episode_id = trajectory.get("episode_id", "unknown_episode")
    print(f"Trajectory: {episode_id}")
    print(f"Instruction: {trajectory.get('instruction', '')}")
    print(f"Steps: {len(steps)}")

    pages_dir = os.path.join(output_dir, "pages")
    code_dir = os.path.join(output_dir, "generated_code")
    assets_dir = os.path.join(output_dir, "extracted_assets")
    os.makedirs(pages_dir, exist_ok=True)
    os.makedirs(code_dir, exist_ok=True)
    os.makedirs(assets_dir, exist_ok=True)

    should_resume = getattr(args, "resume", False)
    existing_ui_pages = _load_existing_ui_pages(output_dir) if should_resume else {}
    existing_manifest_by_page = _load_existing_manifest_by_page(output_dir) if should_resume else {}

    pages_detection_data = []

    for i, step in enumerate(steps):
        screenshot_name, screenshot_path = _resolve_step_screenshot(
            step, args.screenshots_dir, episode_id, i
        )

        if not os.path.exists(screenshot_path):
            print(f"  [{i + 1}/{len(steps)}] SKIP (screenshot missing: {screenshot_name})")
            continue

        page_id = f"page_{len(pages_detection_data)}"
        step_context = _build_step_context(trajectory, i)
        page_png_path = os.path.join(pages_dir, f"{page_id}.png")
        existing_layout = None
        existing_elements = []
        can_skip_detection = False
        can_reuse_page = False

        if should_resume and os.path.exists(page_png_path):
            existing_layout = _load_existing_page_layout(output_dir, page_id, existing_ui_pages)
            can_reuse_page = existing_layout is not None
            if can_reuse_page:
                existing_elements = [dict(item) for item in existing_manifest_by_page.get(page_id, [])]
                can_skip_detection = bool(existing_elements)

        if can_skip_detection:
            with Image.open(screenshot_path) as img_handle:
                orig_size = img_handle.size
            print(f"  [{i + 1}/{len(steps)}] {screenshot_name} [resume existing assets/layout]", flush=True)
            pages_detection_data.append({
                "page_id": page_id,
                "screenshot_name": screenshot_name,
                "screenshot_path": screenshot_path,
                "orig_size": list(orig_size),
                "step": step_context,
                "elements": existing_elements,
                "trajectory_local_page_index": i,
                "existing_layout": existing_layout,
                "reuse_existing_page": True,
            })
            continue

        print(f"  [{i + 1}/{len(steps)}] {screenshot_name}", end="", flush=True)
        elements, orig_size = detect_and_crop(screenshot_path, yolo_model, ocr_reader)
        print(f" ({len(elements)} detected)", end="", flush=True)
        elements, anno_stats = _prioritize_element_anno_bboxes(
            elements,
            screenshot_path,
            screenshot_name,
            getattr(args, "element_anno_dir", ""),
        )
        if anno_stats.get("loaded", 0):
            print(
                f" [anno clickable:{anno_stats['loaded']} matched:{anno_stats['matched']} added:{anno_stats['added']}]",
                end="",
                flush=True,
            )

        if args.save_crops:
            _save_labeled_crops(
                elements,
                orig_size,
                screenshot_path,
                os.path.join(output_dir, "crops", page_id),
            )

        asset_elements = _persist_extracted_assets(
            elements, screenshot_name, assets_dir, step_context
        )
        print(f" [assets:{len(asset_elements)}]", flush=True)

        pages_detection_data.append({
            "page_id": page_id,
            "screenshot_name": screenshot_name,
            "screenshot_path": screenshot_path,
            "orig_size": list(orig_size),
            "step": step_context,
            "elements": asset_elements,
            "trajectory_local_page_index": i,
            "existing_layout": existing_layout,
            "reuse_existing_page": can_reuse_page,
        })

    if not pages_detection_data:
        print(f"  No usable screenshots found for {episode_id}, skipping.")
        return 0

    page_ids_by_local_index = {
        page["trajectory_local_page_index"]: page["page_id"]
        for page in pages_detection_data
    }
    ordered_local_indices = sorted(page_ids_by_local_index)
    next_page_map = {}
    for pos, local_idx in enumerate(ordered_local_indices):
        if pos + 1 < len(ordered_local_indices):
            next_page_map[local_idx] = page_ids_by_local_index[ordered_local_indices[pos + 1]]
        else:
            next_page_map[local_idx] = page_ids_by_local_index[local_idx]

    for page in pages_detection_data:
        local_idx = page["trajectory_local_page_index"]
        page["next_trace_page_id"] = next_page_map.get(local_idx, page["page_id"])

    _save_asset_manifest(output_dir, pages_detection_data)

    pages_data: List[Optional[dict]] = [None] * len(pages_detection_data)
    success_count = 0
    api_concurrency = max(1, int(getattr(args, "api_concurrency", 1) or 1))

    if api_concurrency <= 1 or len(pages_detection_data) <= 1:
        for page_index, page in enumerate(pages_detection_data):
            result = _compose_page_record(
                page=page,
                pages_dir=pages_dir,
                code_dir=code_dir,
                model_name=model_name,
                should_resume=should_resume,
                client=client,
            )
            print(result["message"])
            success_count += 1
            pages_data[page_index] = result["page_data"]
    else:
        print(f"Composing pages with api_concurrency={api_concurrency}")
        with ThreadPoolExecutor(max_workers=api_concurrency) as executor:
            future_to_index = {
                executor.submit(
                    _compose_page_record,
                    page=page,
                    pages_dir=pages_dir,
                    code_dir=code_dir,
                    model_name=model_name,
                    should_resume=should_resume,
                    client=None,
                ): page_index
                for page_index, page in enumerate(pages_detection_data)
            }
            for future in as_completed(future_to_index):
                page_index = future_to_index[future]
                result = future.result()
                print(result["message"])
                success_count += 1
                pages_data[page_index] = result["page_data"]

    pages_data = [page for page in pages_data if page is not None]

    print(f"\nComposed: {success_count}/{len(pages_data)} pages")
    print(f"Building structure ({len(pages_data)} pages)...")
    build_structure(pages_data, trajectory, output_dir)

    print(f"\nDone. Output: {output_dir}/")
    print(f"  pages/             {len(pages_data)} PNG files ({OUTPUT_W}x{OUTPUT_H})")
    print(f"  generated_code/    {len(pages_data)} PIL code files")
    print(f"  extracted_assets/  saved extracted trajectory crops")
    print(f"  trajectory_assets_manifest.json")
    print(f"  ui_structure.json")
    print(f"  ui_structure_layer.json")
    print(f"  action_coord/ per-page action overlay images")
    return len(pages_data)


def run_pipeline(args):
    client = load_api_client()
    model_name = args.model_name
    print(f"Model: {model_name}")
    print(f"API concurrency: {max(1, int(getattr(args, 'api_concurrency', 1) or 1))}")

    yolo_model, ocr_reader = load_detection_models(args.weights_dir, args.gpu)

    annotation_jobs = _collect_annotation_jobs(args)
    if not annotation_jobs:
        raise RuntimeError(f"No annotations found in: {args.annotations_dir}")

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Trajectories: {len(annotation_jobs)}")

    use_subdir = args.trajectory_id is None
    processed_pages = 0

    should_resume = getattr(args, "resume", False)

    for idx, (_, annot_path) in enumerate(annotation_jobs, start=1):
        with open(annot_path, "r", encoding="utf-8") as f:
            trajectory = json.load(f)
        episode_id = trajectory.get("episode_id") or Path(annot_path).stem
        trajectory_output_dir = _resolve_trajectory_output_dir(
            args.output_dir,
            episode_id,
            use_subdir=use_subdir,
        )

        if should_resume:
            expected_pages = _count_expected_pages(trajectory, args)
            ui_structure_path = os.path.join(trajectory_output_dir, "ui_structure.json")
            if os.path.exists(ui_structure_path) and _has_complete_page_outputs(trajectory_output_dir, expected_pages):
                print(f"\n[{idx}/{len(annotation_jobs)}] SKIP (already done) episode={episode_id}")
                continue

        print(f"\n[{idx}/{len(annotation_jobs)}] episode={episode_id}")
        processed_pages += _process_trajectory(
            trajectory=trajectory,
            output_dir=trajectory_output_dir,
            args=args,
            client=client,
            model_name=model_name,
            yolo_model=yolo_model,
            ocr_reader=ocr_reader,
        )

    print(f"\nFinished trajectories: {len(annotation_jobs)}")
    print(f"Total composed pages: {processed_pages}")


def _extract_bg_color(elements: List[dict], screenshot_path: str = None) -> Tuple[int, int, int]:
    """Extract dominant background color from the original screenshot."""
    if screenshot_path and os.path.exists(screenshot_path):
        try:
            img = Image.open(screenshot_path).convert("RGB")
            # Sample corners and edges to get background color
            w, h = img.size
            samples = []
            for x, y in [(10, 10), (w-10, 10), (10, h-10), (w-10, h-10),
                          (w//2, 10), (w//2, h-10), (10, h//2), (w-10, h//2)]:
                samples.append(img.getpixel((x, y)))
            # Use median of samples
            r = sorted([s[0] for s in samples])[len(samples)//2]
            g = sorted([s[1] for s in samples])[len(samples)//2]
            b = sorted([s[2] for s in samples])[len(samples)//2]
            return (r, g, b)
        except Exception:
            pass
    return BG_WHITE


def _fallback_compose(elements: List[dict],
                      orig_size: Tuple[int, int],
                      screenshot_path: str = None) -> Tuple[Image.Image, dict]:
    """Fallback: scale detected elements and paste at proportional positions.

    Uses the original screenshot's background color and preserves spatial
    relationships between elements for a natural-looking result.
    """
    bg_color = _extract_bg_color(elements, screenshot_path)
    canvas = Image.new("RGB", CANVAS_SIZE, bg_color)
    draw = ImageDraw.Draw(canvas)
    layout = {}
    font_sm = _try_load_font(9)

    w, h = orig_size
    content_h = CANVAS_H - STATUS_BAR_HEIGHT - NAV_BAR_HEIGHT
    x_scale = float(CANVAS_W) / w
    y_scale = float(content_h) / h

    # Status bar
    draw.rectangle([0, 0, CANVAS_W, STATUS_BAR_HEIGHT], fill=(15, 15, 20))

    # Paste all elements at proportionally scaled positions
    nav_top = CANVAS_H - NAV_BAR_HEIGHT
    for e in elements:
        x1, y1, x2, y2 = e["bbox"]

        # Scale positions
        sx1 = int(x1 * x_scale)
        sy1 = int(y1 * y_scale) + STATUS_BAR_HEIGHT
        sx2 = int(x2 * x_scale)
        sy2 = int(y2 * y_scale) + STATUS_BAR_HEIGHT

        sw, sh = max(sx2 - sx1, 8), max(sy2 - sy1, 8)

        # Clamp to content area
        if sy1 >= nav_top:
            continue
        if sy2 > nav_top:
            sy2 = nav_top
            sh = sy2 - sy1

        # Ensure minimum readability for small crops
        if sw < 12 and sh < 12:
            continue

        try:
            if "crop" in e:
                crop = e["crop"].convert("RGBA").resize((sw, sh), Image.LANCZOS)
            elif e.get("asset_path") and os.path.exists(e["asset_path"]):
                crop = Image.open(e["asset_path"]).convert("RGBA").resize((sw, sh), Image.LANCZOS)
            else:
                continue
            canvas.paste(crop, (max(0, sx1), sy1), crop)
        except Exception:
            continue

        label = e["label"].replace(" ", "_")[:25]
        if label in layout:
            label = f"{label}_{e['index']}"
        layout[label] = [sx1, sy1, sx1 + sw, sy1 + sh]

    # back/home handled by compose_page nav strip (not here)
    return canvas, layout


def parse_args():
    parser = argparse.ArgumentParser(description="Sim2Real Compose: detection-guided page composition")
    parser.add_argument("--trajectory_id", type=str, default=None,
                        help="Single trajectory episode_id or annotation filename stem to process")
    parser.add_argument("--max_trajectories", type=int, default=None,
                        help="When processing a directory, limit the number of trajectories")
    parser.add_argument("--screenshots_dir", type=str,
                        default="/ext_hdd2/tsyou/AMEX_dataset/AMEX/screenshot",
                        help="Directory with AMEX screenshots")
    parser.add_argument("--annotations_dir", type=str,
                        default="/ext_hdd2/tsyou/AMEX_dataset/AMEX/instruction_anno",
                        help="Directory with AMEX instruction annotation JSONs")
    parser.add_argument("--element_anno_dir", type=str,
                        default="/ext_hdd2/tsyou/AMEX_dataset/AMEX/element_anno",
                        help="Directory with per-screenshot element annotation JSONs")
    parser.add_argument("--output_dir", type=str,
                        default="data_engine/sim2real_envs/amex_sft",
                        help="Output root directory for generated trajectory environments")
    parser.add_argument("--weights_dir", type=str,
                        default="/ext_hdd2/nhkoh/OmniParser/weights",
                        help="OmniParser weights directory")
    parser.add_argument("--model_name", type=str,
                        default="gpt-5-mini-2025-08-07",
                        help="OpenAI model for styling code generation")
    parser.add_argument("--api_concurrency", type=int, default=1,
                        help="Number of concurrent GPT compose requests for the page compose stage")
    parser.add_argument("--gpu", type=int, default=0, help="GPU for YOLO detection")
    parser.add_argument("--save_crops", action="store_true",
                        help="Save labeled crops and annotated screenshots for inspection")
    parser.add_argument("--no_save_code", action="store_true",
                        help="Skip saving GPT-generated .py code files (saves disk at scale)")
    parser.add_argument("--start_index", type=int, default=0,
                        help="Start processing from this trajectory index (for batch splitting)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip trajectories that already have ui_structure.json in output_dir")
    return parser.parse_args()
    


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args)
