"""
Sim2Real Compose Pipeline (Stages 3-4): Detection-guided page composition.

Takes GUIOdyssey trajectories, detects UI elements with OmniParser (YOLO+OCR),
and uses GPT-5-mini to compose GE-Lab pages (252x448 phone ratio) from the actual cropped elements.

Pipeline:
  Stage 1 (Detect): YOLO + OCR detect UI elements on each screenshot
  Stage 2 (Crop): Crop actual icons/text from the screenshot
  Stage 3 (Compose): GPT-5-mini arranges cropped elements on canvas (252x448 phone ratio)
  Stage 4 (Structure): Build ui_structure.json + transition graph

Prerequisites:
    - OmniParser weights at /ext_hdd/nhkoh/OmniParser/weights/
    - GUIOdyssey annotations + screenshots downloaded
    - OPENAI_API_KEY environment variable set

Usage:
    export OPENAI_API_KEY="sk-..."
    python data_engine/sim2real_compose.py \
        --trajectory_id 0055763512444649 \
        --output_dir data_engine/sim2real_envs/shopping_001
"""

import argparse
import base64
import json
import os
import random
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from openai import OpenAI
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from ultralytics import YOLO

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
OUTPUT_CANVAS_SIZE = (448, 448)  # GE-Lab-compatible square canvas
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


def load_detection_models(weights_dir: str = "/ext_hdd/nhkoh/OmniParser/weights",
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


def render_native_page(screenshot_path: str,
                       elements: List[dict],
                       orig_size: Tuple[int, int],
                       output_size: Tuple[int, int] = OUTPUT_CANVAS_SIZE
                       ) -> Tuple[Image.Image, dict, List[dict]]:
    """Fit the original screenshot into the output canvas and scale element bboxes with it.

    This preserves the native GUIOdyssey visual appearance while still producing a
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

    The background stays visually native to the original GUIOdyssey page, but the
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

def load_api_client() -> OpenAI:
    """Initialize OpenAI API client."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable not set")
    client = OpenAI(api_key=api_key)
    print("OpenAI client initialized.")
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
NAV_BTN_W = 40
NAV_BTN_H = 24
NAV_STRIP_H = NAV_BTN_H + 8  # 4px padding top + bottom
PHONE_CANVAS_H = OUTPUT_H - NAV_STRIP_H
PHONE_CANVAS_W = min(OUTPUT_W, int(round(PHONE_CANVAS_H * 9 / 16)))
CANVAS_SIZE = (PHONE_CANVAS_W, PHONE_CANVAS_H)
CANVAS_W, CANVAS_H = CANVAS_SIZE
PHONE_OFFSET_X = (OUTPUT_W - CANVAS_W) // 2
PHONE_OFFSET_Y = NAV_STRIP_H
GELAB_BACK_COLOR = (255, 200, 200)  # pink
GELAB_HOME_COLOR = (200, 255, 200)  # green
GELAB_BACK_BBOX = [4, 4, 4 + NAV_BTN_W, 4 + NAV_BTN_H]
GELAB_HOME_BBOX = [OUTPUT_W - 4 - NAV_BTN_W, 4, OUTPUT_W - 4, 4 + NAV_BTN_H]

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
as a JSON object for rendering on a 448x448 pixel canvas.

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
    """Resize an image into a target box without stretching it."""
    fitted_w, fitted_h, scale, offset_x, offset_y = _fit_size(image.size, target_size)
    resized = image.resize((fitted_w, fitted_h), Image.LANCZOS)
    canvas = Image.new("RGB", target_size, bg_color)
    canvas.paste(resized, (offset_x, offset_y))
    return canvas, scale, offset_x, offset_y


def _scale_bbox_to_box(bbox: List[int],
                       src_size: Tuple[int, int],
                       target_size: Tuple[int, int],
                       base_offset: Tuple[int, int] = (0, 0)) -> List[int]:
    """Scale a bbox with the same fit-to-box transform used for the page image."""
    _, _, scale, offset_x, offset_y = _fit_size(src_size, target_size)
    base_x, base_y = base_offset
    return [
        int(round(bbox[0] * scale)) + offset_x + base_x,
        int(round(bbox[1] * scale)) + offset_y + base_y,
        int(round(bbox[2] * scale)) + offset_x + base_x,
        int(round(bbox[3] * scale)) + offset_y + base_y,
    ]


def render_from_code(code_str: str, elements: List[dict],
                     orig_size: Tuple[int, int] = (720, 1280)
                     ) -> Tuple[Optional[Image.Image], Optional[dict]]:
    """Execute GPT-generated PIL code at original resolution, then fit it to CANVAS_SIZE.

    GPT composes at the original phone resolution (e.g., 720x1280) using the real
    detected coordinates, then we fit it into the GE-Lab viewport without stretching it.
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
# Page Renderer: JSON spec -> 448x448 PIL image + layout dict
# ---------------------------------------------------------------------------

def _try_load_font(size: int):
    """Try to load a font, fall back to default."""
    for name in ["DejaVuSans.ttf", "FreeSans.ttf", "arial.ttf"]:
        try:
            return ImageFont.truetype(name, size)
        except (IOError, OSError):
            continue
    return ImageFont.load_default()


def render_page(spec: dict, icon_pool: Dict[str, Image.Image],
                metadata: List[dict], used_icons: set) -> Tuple[Image.Image, dict]:
    """Render a page specification to a 448x448 image.

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
    """Build GE-Lab compatible structure files from rendered pages.

    Produces ui_structure.json and ui_structure_layer.json matching the exact
    GE-Lab format used by env_utils.py, generate_sft_data.py, and evaluate.py.

    pages_data: list of {"page_id": str, "layout": dict}
    trajectory: GUIOdyssey annotation dict
    """
    steps = trajectory.get("steps", [])
    task_info = trajectory.get("task_info", {})

    ui_structure = {"pages": {}, "metadata": {
        "source": "sim2real_compose",
        "episode_id": trajectory.get("episode_id", ""),
        "task": task_info.get("task", ""),
        "category": task_info.get("category", ""),
        "apps": task_info.get("app", []),
        "total_pages": len(pages_data),
        "canvas_size": list(OUTPUT_CANVAS_SIZE),
        "phone_canvas_size": list(CANVAS_SIZE),
    }}

    home_page_id = _detect_home_page_id(pages_data)

    # Build pages with transitions (GE-Lab list format)
    for i, pdata in enumerate(pages_data):
        page_id = pdata["page_id"]
        layout = pdata["layout"]
        step = pdata.get("step", steps[i] if i < len(steps) else {})
        orig_size = tuple(pdata.get("orig_size", (720, 1280)))

        # Build layout dict with type: "normal" or "system"
        layout_typed = {}
        for k, bbox in layout.items():
            ltype = "system" if k in ("back", "home") else "normal"
            layout_typed[k] = {"bbox": bbox, "type": ltype}

        # Build transitions as list: [{action, target_page, icon_bbox}]
        transitions = []
        used_system_targets = set()
        if i + 1 < len(pages_data):
            next_page = pages_data[i + 1]["page_id"]
            target_elem = _find_action_target(step, layout, orig_size)
            icon_bbox = layout.get(target_elem, [0, 0, 0, 0])
            transitions.append({
                "action": target_elem,
                "target_page": next_page,
                "icon_bbox": icon_bbox,
            })
            if target_elem in ("back", "home"):
                used_system_targets.add(target_elem)

        # Back transition (except root page)
        if i > 0 and "back" not in used_system_targets:
            back_bbox = layout.get("back", GELAB_BACK_BBOX)
            transitions.append({
                "action": "back",
                "target_page": pages_data[i - 1]["page_id"],
                "icon_bbox": back_bbox,
            })

        # Home transition
        if "home" not in used_system_targets:
            home_bbox = layout.get("home", GELAB_HOME_BBOX)
            transitions.append({
                "action": "home",
                "target_page": home_page_id,
                "icon_bbox": home_bbox,
            })

        ui_structure["pages"][page_id] = {
            "image": f"{page_id}.png",
            "depth": i,
            "layout": layout_typed,
            "transitions": transitions,
        }

    # Build layer structure (matches GE-Lab ui_structure_layer.json format)
    layer = _build_layer_structure(pages_data, ui_structure["pages"])

    # Save
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "ui_structure.json"), "w") as f:
        json.dump(ui_structure, f, indent=2)
    with open(os.path.join(output_dir, "ui_structure_layer.json"), "w") as f:
        json.dump(layer, f, indent=2)

    return ui_structure


def _detect_home_page_id(pages_data: List[dict]) -> str:
    """Pick the most likely home/root page using the trajectory descriptions."""
    for pdata in pages_data:
        step = pdata.get("step", {})
        desc = " ".join([
            str(step.get("description", "")),
            str(step.get("low_level_instruction", "")),
            str(step.get("info", "")),
        ]).lower()
        if "home screen" in desc or "launcher" in desc:
            return pdata["page_id"]
    return pages_data[0]["page_id"] if pages_data else "page_0"


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


def _find_action_target(step: dict, layout: dict,
                        orig_size: Tuple[int, int]) -> str:
    """Find the layout element most likely targeted by the real trajectory action."""
    if not layout:
        return "unknown"

    action = str(step.get("action", "")).upper()
    info = str(step.get("info", ""))
    instruction = " ".join([
        str(step.get("low_level_instruction", "")),
        str(step.get("description", "")),
        str(step.get("intention", "")),
        info,
    ]).lower()

    if "KEY_HOME" in info or "home screen" in instruction:
        return "home"
    if "go back" in instruction or instruction.startswith("back ") or info == "BACK":
        return "back"

    sam2_bbox = step.get("sam2_bbox") or []
    if action == "CLICK" and len(sam2_bbox) == 4:
        scaled_bbox = _scale_bbox_to_box(
            sam2_bbox, orig_size, CANVAS_SIZE, (PHONE_OFFSET_X, PHONE_OFFSET_Y)
        )
        best_key = None
        best_iou = 0.0
        best_distance = float("inf")
        for key, bbox in layout.items():
            if key in ("back", "home"):
                continue
            iou = _bbox_iou(scaled_bbox, bbox)
            distance = _bbox_center_distance(scaled_bbox, bbox)
            if iou > best_iou or (iou == best_iou and distance < best_distance):
                best_key = key
                best_iou = iou
                best_distance = distance
        if best_key is not None and (best_iou > 0 or best_distance <= 48):
            return best_key

    if action == "TEXT":
        for preferred in ("search_bar", "keyboard"):
            if preferred in layout:
                return preferred

    best_score = 0
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


def _build_layer_structure(pages_data: List[dict],
                           pages_full: Dict[str, dict]) -> dict:
    """Build layer structure matching GE-Lab ui_structure_layer.json format.

    Each node has: image, depth, layout, transitions (non-system only), subnodes.
    """
    if not pages_data:
        return {"root": None, "metadata": {}}

    page_order = [pdata["page_id"] for pdata in pages_data]

    def build_node(page_id, visited):
        visited.add(page_id)
        page_data = pages_full.get(page_id, {})

        # Filter transitions to exclude system actions (back, home)
        all_trans = page_data.get("transitions", [])
        non_system = [t for t in all_trans if t["action"] not in ("back", "home")]

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

    # Phase 3: Compose final 448x448 canvas with a dedicated nav strip at top.
    final = Image.new("RGB", OUTPUT_CANVAS_SIZE, (245, 245, 245))
    final.paste(page_img, (PHONE_OFFSET_X, PHONE_OFFSET_Y))

    # Shift all layout bboxes into the final GE-Lab canvas
    shifted_layout = {}
    for key, bbox in layout.items():
        if key in ("back", "home"):
            continue  # will be set below
        shifted_layout[key] = [
            bbox[0] + PHONE_OFFSET_X, bbox[1] + PHONE_OFFSET_Y,
            bbox[2] + PHONE_OFFSET_X, bbox[3] + PHONE_OFFSET_Y,
        ]

    # Draw nav strip: white background + separator + rounded back/home buttons
    draw_final = ImageDraw.Draw(final)
    draw_final.rectangle([0, 0, OUTPUT_W, NAV_STRIP_H], fill=(255, 255, 255))
    draw_final.line([0, NAV_STRIP_H - 1, OUTPUT_W, NAV_STRIP_H - 1], fill=(200, 200, 200))
    font_nav = _try_load_font(12)

    bx1, by1, bx2, by2 = GELAB_BACK_BBOX
    draw_final.rounded_rectangle([bx1, by1, bx2, by2], radius=4, fill=GELAB_BACK_COLOR)
    tw = draw_final.textlength("back", font=font_nav) if hasattr(draw_final, "textlength") else 24
    draw_final.text((bx1 + (NAV_BTN_W - tw) / 2, by1 + (NAV_BTN_H - 12) / 2),
                    "back", fill=TEXT_BLACK, font=font_nav)
    shifted_layout["back"] = [bx1, by1, bx2, by2]

    hx1, hy1, hx2, hy2 = GELAB_HOME_BBOX
    draw_final.rounded_rectangle([hx1, hy1, hx2, hy2], radius=4, fill=GELAB_HOME_COLOR)
    tw = draw_final.textlength("home", font=font_nav) if hasattr(draw_final, "textlength") else 28
    draw_final.text((hx1 + (NAV_BTN_W - tw) / 2, hy1 + (NAV_BTN_H - 12) / 2),
                    "home", fill=TEXT_BLACK, font=font_nav)
    shifted_layout["home"] = [hx1, hy1, hx2, hy2]

    code_artifact = {
        "styling_source": styling_source,
        "render_status": render_status,
        "styling_code": styling_code,
        "position_code": position_code,
        "full_code": full_code,
    }
    return final, shifted_layout, code_artifact


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
        "# The final runtime image is then fit into the 448x448 GE-Lab canvas with a top nav strip.",
    ]

    contents = "\n".join(header_lines) + "\n\n"
    contents += "# --- GPT styling skeleton ---\n"
    contents += code_artifact.get("styling_code", "").strip() + "\n\n"
    contents += "# --- Deterministic element pastes ---\n"
    contents += code_artifact.get("position_code", "").strip() + "\n"

    code_path = os.path.join(code_dir, f"{page_id}.py")
    with open(code_path, "w", encoding="utf-8") as f:
        f.write(contents)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(args):
    # Initialize OpenAI client for GPT styling
    client = load_api_client()
    model_name = args.model_name
    print(f"Model: {model_name}")

    # Load detection models (YOLO + OCR)
    yolo_model, ocr_reader = load_detection_models(args.weights_dir, args.gpu)

    # Load trajectory
    annot_path = os.path.join(args.annotations_dir, f"{args.trajectory_id}.json")
    if not os.path.exists(annot_path):
        raise FileNotFoundError(f"Annotation not found: {annot_path}")
    with open(annot_path) as f:
        trajectory = json.load(f)

    steps = trajectory.get("steps", [])
    episode_id = trajectory.get("episode_id", args.trajectory_id)
    task_info = trajectory.get("task_info", {})
    print(f"Trajectory: {episode_id}")
    print(f"Task: {task_info.get('task', 'N/A')}")
    print(f"Apps: {task_info.get('app', [])}")
    print(f"Steps: {len(steps)}")

    # Stage 1-2: detect and extract trajectory assets first
    pages_dir = os.path.join(args.output_dir, "pages")
    code_dir = os.path.join(args.output_dir, "generated_code")
    assets_dir = os.path.join(args.output_dir, "extracted_assets")
    os.makedirs(pages_dir, exist_ok=True)
    os.makedirs(code_dir, exist_ok=True)
    os.makedirs(assets_dir, exist_ok=True)

    pages_detection_data = []

    for i, step in enumerate(steps):
        screenshot_name = step.get("screenshot", f"{episode_id}_{i}.png")
        screenshot_path = os.path.join(args.screenshots_dir, screenshot_name)

        if not os.path.exists(screenshot_path):
            print(f"  [{i+1}/{len(steps)}] SKIP (screenshot missing: {screenshot_name})")
            continue

        page_id = f"page_{i}"
        step_context = _build_step_context(trajectory, i)

        # Stage 1-2: Detect + crop UI elements from this screenshot
        print(f"  [{i+1}/{len(steps)}] {screenshot_name}", end="", flush=True)
        elements, orig_size = detect_and_crop(screenshot_path, yolo_model, ocr_reader)
        print(f" ({len(elements)} detected)", end="", flush=True)

        # Optionally save labeled crops for inspection
        if args.save_crops:
            _save_labeled_crops(elements, orig_size, screenshot_path,
                                os.path.join(args.output_dir, "crops", page_id))

        asset_elements = _persist_extracted_assets(elements, screenshot_name, assets_dir, step_context)
        print(f" [assets:{len(asset_elements)}]", end="", flush=True)

        pages_detection_data.append({
            "page_id": page_id,
            "screenshot_name": screenshot_name,
            "screenshot_path": screenshot_path,
            "orig_size": list(orig_size),
            "step": step_context,
            "elements": asset_elements,
        })

    _save_asset_manifest(args.output_dir, pages_detection_data)

    # Stage 3-4: compose from extracted assets and build structure
    pages_data = []
    success_count = 0

    for page in pages_detection_data:
        page_id = page["page_id"]
        screenshot_name = page["screenshot_name"]
        screenshot_path = page["screenshot_path"]
        orig_size = tuple(page["orig_size"])
        step_context = page["step"]
        elements = page["elements"]

        # Stage 3: Compose page
        #   Phase 1: GPT generates background/styling on blank canvas
        #   Phase 2: Extracted trajectory assets pasted at exact positions
        #   Phase 3: GE-Lab back/home buttons at top
        page_img, layout, code_artifact = compose_page(
            client, model_name, elements, orig_size, screenshot_path, step_context
        )
        print(f"  compose {page_id} -> {len(layout)} layout elems")
        success_count += 1

        page_img.save(os.path.join(pages_dir, f"{page_id}.png"))
        _save_page_code(code_dir, page_id, screenshot_name, step_context, code_artifact)

        pages_data.append({
            "page_id": page_id,
            "layout": layout,
            "orig_size": list(orig_size),
            "step": step_context,
        })

    print(f"\nComposed: {success_count}/{len(pages_data)} pages")

    # Stage 4: Build structure
    print(f"Building structure ({len(pages_data)} pages)...")
    structure = build_structure(pages_data, trajectory, args.output_dir)

    print(f"\nDone. Output: {args.output_dir}/")
    print(f"  pages/             {len(pages_data)} PNG files ({OUTPUT_W}x{OUTPUT_H})")
    print(f"  generated_code/    {len(pages_data)} PIL code files")
    print(f"  extracted_assets/  saved extracted trajectory crops")
    print(f"  trajectory_assets_manifest.json")
    print(f"  ui_structure.json")
    print(f"  ui_structure_layer.json")


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
            crop = e["crop"].convert("RGBA").resize((sw, sh), Image.LANCZOS)
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
    parser.add_argument("--trajectory_id", type=str, required=True,
                        help="GUIOdyssey episode ID to process")
    parser.add_argument("--screenshots_dir", type=str,
                        default="/ext_hdd/nhkoh/dataset/GUIOdyssey/screenshots",
                        help="Directory with GUIOdyssey screenshots")
    parser.add_argument("--annotations_dir", type=str,
                        default="/ext_hdd/nhkoh/dataset/GUIOdyssey/annotations",
                        help="Directory with GUIOdyssey annotation JSONs")
    parser.add_argument("--output_dir", type=str,
                        default="data_engine/sim2real_envs/trajectory_001",
                        help="Output directory for the generated environment")
    parser.add_argument("--weights_dir", type=str,
                        default="/ext_hdd/nhkoh/OmniParser/weights",
                        help="OmniParser weights directory")
    parser.add_argument("--model_name", type=str,
                        default="gpt-5-mini-2025-08-07",
                        help="OpenAI model for styling code generation")
    parser.add_argument("--gpu", type=int, default=0, help="GPU for YOLO detection")
    parser.add_argument("--save_crops", action="store_true",
                        help="Save labeled crops and annotated screenshots for inspection")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args)
