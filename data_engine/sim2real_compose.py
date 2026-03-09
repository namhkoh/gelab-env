"""
Sim2Real Compose Pipeline (Stages 3-4): Detection-guided page composition.

Takes GUIOdyssey trajectories, detects UI elements with OmniParser (YOLO+OCR),
and uses GPT-5-mini to compose 448x448 GE-Lab pages from the actual cropped elements.

Pipeline:
  Stage 1 (Detect): YOLO + OCR detect UI elements on each screenshot
  Stage 2 (Crop): Crop actual icons/text from the screenshot
  Stage 3 (Compose): GPT-5-mini arranges cropped elements on 448x448 canvas
  Stage 4 (Structure): Build ui_structure.json + transition graph

Prerequisites:
    - OmniParser weights at /ext_hdd2/nhkoh/OmniParser/weights/
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
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CANVAS_SIZE = (448, 448)
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


RENDER_CODE_PROMPT = """\
Write Python PIL code to recreate this mobile screenshot as a 448x448 GE-Lab page, \
using the ACTUAL detected UI elements listed below.

Original screenshot: {{orig_w}}x{{orig_h}} pixels.

Detected UI elements (from OmniParser YOLO+OCR):
{{element_list}}

RULES:
1. Use get_crop(index, w, h) to paste REAL cropped elements from the screenshot.
2. ALL coordinates MUST be within 0-448 (canvas is 448x448).
3. Canvas zones: y=0..20 status bar, y=20..65 header, y=65..403 content, y=403..448 nav bar.
4. x margins: 15 left, 433 right.
5. Do NOT import anything. Do NOT define functions.
6. Every pasted element MUST have a layout entry: layout["name"] = [x1, y1, x2, y2]
7. All crops are RGBA, so always use: canvas.paste(img, (x, y), img)

Available variables:
- canvas: PIL Image (448x448 RGBA-compatible)
- draw: PIL ImageDraw object
- get_crop(index, w, h): returns detected element [index] resized to w x h (RGBA)
- font_sm (9pt), font_md (12pt), font_lg (16pt), font_xl (20pt)
- layout: dict — fill with layout["element_name"] = [x1, y1, x2, y2]
- Image, ImageDraw, range, len, enumerate, min, max, int, float, str, etc.

Layout patterns:
- Home screen: 4-column icon grid at x=[30,135,240,345], rows at y=[70,140,210,280]
- Settings/list page: stack items at y=70,100,130,... with full width rows
- App page: header at y=20..65, main content centered in y=65..403
- Search page: search bar at y=70, results below

REQUIRED nav bar at end:
draw.rectangle([0, 403, 448, 448], fill=(40, 40, 50))
draw.rounded_rectangle([20, 410, 100, 440], radius=4, fill=(255, 200, 200))
draw.text((40, 417), "Back", fill=(30, 30, 30), font=font_sm)
layout["back"] = [20, 410, 100, 440]
draw.rounded_rectangle([348, 410, 428, 440], radius=4, fill=(200, 255, 200))
draw.text((368, 417), "Home", fill=(30, 30, 30), font=font_sm)
layout["home"] = [348, 410, 428, 440]

Output ONLY a ```python code block. Start with background color, then paste elements, end with nav bar."""

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
                       max_retries: int = 2) -> Optional[str]:
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

    for attempt in range(max_retries):
        try:
            cur_prompt = prompt
            if attempt > 0:
                # Retry with a more forceful prompt
                cur_prompt = (
                    "You MUST respond with ONLY a ```python code block. "
                    "No explanations, no markdown besides the code block.\n\n"
                    + prompt
                )
            response = _query_gpt(client, model_name, image_path, cur_prompt)
            code = _extract_code_block(response)
            if code:
                return code
            if attempt == 0:
                # Log first 200 chars of failed response for debugging
                _log_failed_response(response, image_path)
                print(f" [no code block, retry]", end="", flush=True)
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
    return max(numbers) if numbers else 448


def render_from_code(code_str: str, elements: List[dict]) -> Tuple[Optional[Image.Image], Optional[dict]]:
    """Execute GPT-generated PIL code with actual cropped elements."""
    canvas = Image.new("RGB", CANVAS_SIZE, BG_WHITE)
    draw = ImageDraw.Draw(canvas)
    layout = {}

    font_sm = _try_load_font(9)
    font_md = _try_load_font(12)
    font_lg = _try_load_font(16)
    font_xl = _try_load_font(20)

    def get_crop(index, w=50, h=50):
        """Return the actual cropped element resized to w x h."""
        if 0 <= index < len(elements):
            crop = elements[index]["crop"].convert("RGBA")
            return crop.resize((int(w), int(h)), Image.LANCZOS)
        # Fallback placeholder
        ph = Image.new("RGBA", (int(w), int(h)), (200, 200, 200, 255))
        return ph

    # Use a single dict as both globals and locals to avoid Python scoping issues
    # with exec() where separate globals/locals can't find names in locals.
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
        # Safe builtins — comprehensive set to avoid exec failures
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
    }

    try:
        exec(code_str, namespace)
    except Exception as e:
        print(f"\n  Code execution error: {e}")
        # Try stripping problematic lines and retry
        fixed = _try_fix_code(code_str, str(e))
        if fixed and fixed != code_str:
            try:
                # Reset canvas
                canvas = Image.new("RGB", CANVAS_SIZE, BG_WHITE)
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

    # Ensure layout has at least back/home
    if "back" not in layout:
        layout["back"] = [20, 410, 100, 440]
    if "home" not in layout:
        layout["home"] = [348, 410, 428, 440]

    return canvas, layout


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
    draw.rectangle([0, 0, 448, STATUS_BAR_HEIGHT], fill=(20, 20, 25))
    y_cursor = STATUS_BAR_HEIGHT

    # --- Header ---
    header = spec.get("header", {})
    if header.get("visible", False):
        h_color = tuple(header.get("color", list(HEADER_BLUE)))
        draw.rectangle([0, y_cursor, 448, y_cursor + HEADER_HEIGHT], fill=h_color)

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
    nav_y = CANVAS_SIZE[1] - NAV_BAR_HEIGHT
    draw.rectangle([0, nav_y, 448, 448], fill=NAV_BAR_COLOR)

    # Back button
    bx1, by1 = 20, nav_y + 8
    bw, bh = 80, 30
    draw.rounded_rectangle([bx1, by1, bx1 + bw, by1 + bh], radius=4,
                           fill=(255, 200, 200))
    draw.text((bx1 + 20, by1 + 7), "Back", fill=TEXT_BLACK, font=font_sm)
    layout["back"] = [bx1, by1, bx1 + bw, by1 + bh]

    # Home button
    hx1 = CANVAS_SIZE[0] - 100
    draw.rounded_rectangle([hx1, by1, hx1 + bw, by1 + bh], radius=4,
                           fill=(200, 255, 200))
    draw.text((hx1 + 20, by1 + 7), "Home", fill=TEXT_BLACK, font=font_sm)
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
    draw.rectangle([0, kb_y, 448, kb_y + 95], fill=(210, 210, 215))

    rows = ["qwertyuiop", "asdfghjkl", "zxcvbnm"]
    key_h = 26
    for r, row in enumerate(rows):
        row_w = len(row) * 32
        start_x = (448 - row_w) // 2
        for c, ch in enumerate(row):
            kx = start_x + c * 32
            ky = kb_y + 5 + r * (key_h + 4)
            draw.rounded_rectangle([kx, ky, kx + 28, ky + key_h],
                                   radius=3, fill=(255, 255, 255))
            draw.text((kx + 9, ky + 6), ch, fill=TEXT_BLACK, font=font)

    layout["keyboard"] = [0, kb_y, 448, kb_y + 95]
    return kb_y + 100


# ---------------------------------------------------------------------------
# Stage 4: Build structure (ui_structure.json + ui_structure_layer.json)
# ---------------------------------------------------------------------------

def build_structure(pages_data: List[dict], trajectory: dict,
                    output_dir: str) -> dict:
    """Build GE-Lab compatible structure files from rendered pages.

    pages_data: list of {"page_id": str, "layout": dict, "spec": dict}
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
        "num_pages": len(pages_data),
    }}

    # Build pages with transitions
    for i, pdata in enumerate(pages_data):
        page_id = pdata["page_id"]
        layout = pdata["layout"]

        # Determine transitions from trajectory steps
        transitions = {}
        if i < len(steps):
            step = steps[i]
            action = step.get("action", "")
            instruction = step.get("low_level_instruction", "")

            if i + 1 < len(pages_data):
                next_page = pages_data[i + 1]["page_id"]

                # Find the interactive element that was clicked
                target_elem = _find_action_target(action, instruction, layout)

                if action == "CLICK":
                    transitions[target_elem] = {
                        "target": next_page,
                        "action_type": "CLICK",
                        "instruction": instruction,
                    }
                elif action == "TYPE":
                    text_content = step.get("info", "")
                    if isinstance(text_content, list):
                        text_content = str(text_content)
                    transitions[target_elem] = {
                        "target": next_page,
                        "action_type": "TEXT",
                        "text": text_content,
                        "instruction": instruction,
                    }
                elif action == "SCROLL":
                    transitions[target_elem] = {
                        "target": next_page,
                        "action_type": "SCROLL",
                        "instruction": instruction,
                    }
                elif action in ("COMPLETE", "INCOMPLETE"):
                    transitions["complete"] = {
                        "target": None,
                        "action_type": "COMPLETE",
                        "instruction": instruction,
                    }
                else:
                    transitions[target_elem] = {
                        "target": next_page,
                        "action_type": action,
                        "instruction": instruction,
                    }

        # Back transitions (except root page)
        if i > 0:
            transitions["back"] = {
                "target": pages_data[i - 1]["page_id"],
                "action_type": "CLICK",
            }
        # Home transition
        transitions["home"] = {
            "target": pages_data[0]["page_id"],
            "action_type": "CLICK",
        }

        ui_structure["pages"][page_id] = {
            "image": f"{page_id}.png",
            "depth": i,
            "layout": {k: {"bbox": v, "type": "normal"} for k, v in layout.items()},
            "transitions": transitions,
        }

    # Build layer structure (linear chain for trajectories)
    layer = _build_layer_structure(pages_data)

    # Save
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "ui_structure.json"), "w") as f:
        json.dump(ui_structure, f, indent=2)
    with open(os.path.join(output_dir, "ui_structure_layer.json"), "w") as f:
        json.dump(layer, f, indent=2)

    return ui_structure


def _find_action_target(action: str, instruction: str, layout: dict) -> str:
    """Find the layout element most likely targeted by the action."""
    if not layout or not instruction:
        return "unknown"

    instruction_lower = instruction.lower()
    best_score = 0
    best_key = list(layout.keys())[0] if layout else "unknown"

    for key in layout:
        key_lower = key.lower().replace("_", " ")
        score = SequenceMatcher(None, instruction_lower, key_lower).ratio()
        # Boost score if key appears as substring in instruction
        if key_lower in instruction_lower:
            score += 0.3
        if score > best_score:
            best_score = score
            best_key = key

    return best_key


def _build_layer_structure(pages_data: List[dict]) -> dict:
    """Build a linear layer structure (trajectory = chain of pages)."""
    if not pages_data:
        return {"root": None, "metadata": {}}

    def build_node(idx):
        node = {
            "name": pages_data[idx]["page_id"],
            "image": f"{pages_data[idx]['page_id']}.png",
            "subnodes": [],
        }
        if idx + 1 < len(pages_data):
            node["subnodes"].append(build_node(idx + 1))
        return node

    return {"root": build_node(0), "metadata": {"type": "trajectory"}}


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(args):
    # Initialize OpenAI client
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

    # Process each step
    pages_dir = os.path.join(args.output_dir, "pages")
    codes_dir = os.path.join(args.output_dir, "generated_code")
    os.makedirs(pages_dir, exist_ok=True)
    os.makedirs(codes_dir, exist_ok=True)

    pages_data = []
    code_success = 0
    fallback_count = 0

    for i, step in enumerate(steps):
        screenshot_name = step.get("screenshot", f"{episode_id}_{i}.png")
        screenshot_path = os.path.join(args.screenshots_dir, screenshot_name)

        if not os.path.exists(screenshot_path):
            print(f"  [{i+1}/{len(steps)}] SKIP (screenshot missing: {screenshot_name})")
            continue

        page_id = f"step_{i}"

        # Stage 1-2: Detect + crop UI elements from this screenshot
        print(f"  [{i+1}/{len(steps)}] {screenshot_name}", end="", flush=True)
        elements, orig_size = detect_and_crop(screenshot_path, yolo_model, ocr_reader)
        print(f" ({len(elements)} detected)", end="", flush=True)

        # Stage 3: GPT arranges cropped elements on 448x448 canvas
        page_img = None
        layout = None

        code_str = generate_page_code(
            client, model_name, screenshot_path, elements, orig_size, step
        )
        if code_str:
            with open(os.path.join(codes_dir, f"{page_id}.py"), "w") as f:
                f.write(code_str)
            page_img, layout = render_from_code(code_str, elements)

        if page_img is not None:
            print(f" [code] ({len(layout)} elems)")
            code_success += 1
        else:
            # Fallback: scaled paste preserving spatial layout + background color
            print(f" [fallback]", end="", flush=True)
            page_img, layout = _fallback_compose(elements, orig_size, screenshot_path)
            print(f" ({len(layout)} elems)")
            fallback_count += 1

        page_img.save(os.path.join(pages_dir, f"{page_id}.png"))

        pages_data.append({
            "page_id": page_id,
            "layout": layout,
            "spec": {},
        })

    print(f"\nRendering: {code_success} code, {fallback_count} fallback")

    # Stage 4: Build structure
    print(f"Building structure ({len(pages_data)} pages)...")
    structure = build_structure(pages_data, trajectory, args.output_dir)

    print(f"\nDone. Output: {args.output_dir}/")
    print(f"  pages/           {len(pages_data)} PNG files")
    print(f"  generated_code/  GPT PIL code")
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
    x_scale = 448.0 / w
    y_scale = 383.0 / h  # content area = 383px (y=20..403)

    # Status bar
    draw.rectangle([0, 0, 448, 20], fill=(15, 15, 20))

    # Separate icons and text elements for better layout
    icon_elems = [e for e in elements if e["type"] == "icon"]
    text_elems = [e for e in elements if e["type"] == "text"]

    # Paste all elements at proportionally scaled positions
    for e in elements:
        x1, y1, x2, y2 = e["bbox"]
        ew, eh = x2 - x1, y2 - y1

        # Scale positions
        sx1 = int(x1 * x_scale)
        sy1 = int(y1 * y_scale) + 20
        sx2 = int(x2 * x_scale)
        sy2 = int(y2 * y_scale) + 20

        sw, sh = max(sx2 - sx1, 8), max(sy2 - sy1, 8)

        # Clamp to content area
        if sy1 >= 400:
            continue
        if sy2 > 403:
            sy2 = 403
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
        # Avoid duplicate layout keys
        if label in layout:
            label = f"{label}_{e['index']}"
        layout[label] = [sx1, sy1, sx1 + sw, sy1 + sh]

    # Nav bar
    draw.rectangle([0, 403, 448, 448], fill=NAV_BAR_COLOR)
    draw.rounded_rectangle([20, 410, 100, 440], radius=4, fill=(255, 200, 200))
    draw.text((40, 417), "Back", fill=TEXT_BLACK, font=font_sm)
    layout["back"] = [20, 410, 100, 440]
    draw.rounded_rectangle([348, 410, 428, 440], radius=4, fill=(200, 255, 200))
    draw.text((368, 417), "Home", fill=TEXT_BLACK, font=font_sm)
    layout["home"] = [348, 410, 428, 440]

    return canvas, layout


def parse_args():
    parser = argparse.ArgumentParser(description="Sim2Real Compose: detection-guided page composition")
    parser.add_argument("--trajectory_id", type=str, required=True,
                        help="GUIOdyssey episode ID to process")
    parser.add_argument("--screenshots_dir", type=str,
                        default="/ext_hdd2/nhkoh/GUI-Odyssey/screenshots",
                        help="Directory with GUIOdyssey screenshots")
    parser.add_argument("--annotations_dir", type=str,
                        default="/ext_hdd2/nhkoh/GUI-Odyssey/annotations",
                        help="Directory with GUIOdyssey annotation JSONs")
    parser.add_argument("--output_dir", type=str,
                        default="data_engine/sim2real_envs/trajectory_001",
                        help="Output directory for the generated environment")
    parser.add_argument("--weights_dir", type=str,
                        default="/ext_hdd2/nhkoh/OmniParser/weights",
                        help="OmniParser weights directory")
    parser.add_argument("--model_name", type=str,
                        default="gpt-5-mini-2025-08-07",
                        help="OpenAI model name for page composition")
    parser.add_argument("--gpu", type=int, default=0, help="GPU for YOLO detection")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args)
