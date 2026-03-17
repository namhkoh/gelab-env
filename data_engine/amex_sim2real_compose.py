"""
Sim2Real Compose Pipeline (Stages 3-4): Detection-guided page composition.

Takes AMEX trajectories, detects UI elements with OmniParser (YOLO+OCR),
and uses GPT-5-mini to compose a full-width GE-Lab phone viewport inside
the final 448x448 canvas.

This variant is designed to merge multiple AMEX trajectories into one shared
canonical UI graph. Repeated states across traces are deduplicated, their
layouts/actions are merged, and the final outputs describe the merged
navigation structure rather than a single isolated trajectory. (Still working on it)

Pipeline:
  Stage 1 (Detect): YOLO + OCR detect UI elements on each screenshot
  Stage 2 (Crop): Crop actual icons/text from the screenshot
  Stage 3 (Compose): GPT-5-mini arranges cropped elements on a full-width phone viewport
  Stage 4 (Merge + Structure): Merge repeated pages across trajectories into a canonical graph and export ui_structure.json, ui_structure_layer.json, and ui_topology.png

Prerequisites:
    - OmniParser weights at /ext_hdd2/nhkoh/OmniParser/weights/
    - AMEX annotations + screenshots downloaded
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
from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps
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
DEFAULT_STATE_MATCH_THRESHOLD = 0.82
DEFAULT_LAYOUT_MATCH_IOU = 0.72
DEFAULT_LLM_PAGE_MATCH_TOP_K = 3

ACTION_KIND_MAP = {
    "TAP": "TAP",
    "SCROLL": "SCROLL",
    "TYPE": "TYPE",
    "PRESS_BACK": "PRESS_BACK",
    "PRESS_HOME": "PRESS_HOME",
    "PRESS_ENTER": "PRESS_ENTER",
    "TASK_COMPLETE": "TASK_COMPLETE",
    "TASK_IMPOSSIBLE": "TASK_IMPOSSIBLE",
}

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


def _normalize_label(label: str) -> str:
    normalized = re.sub(r"[^0-9a-z]+", " ", str(label or "").lower()).strip()
    return normalized or "unknown"


def _quantize_bbox(bbox: List[int], canvas_size: Tuple[int, int]) -> Tuple[int, int, int, int]:
    if len(bbox) != 4:
        return (0, 0, 0, 0)
    w = max(canvas_size[0], 1)
    h = max(canvas_size[1], 1)
    return (
        int(round(bbox[0] * 20 / w)),
        int(round(bbox[1] * 20 / h)),
        int(round(bbox[2] * 20 / w)),
        int(round(bbox[3] * 20 / h)),
    )


def _layout_signature(layout: dict,
                      canvas_size: Tuple[int, int] = OUTPUT_CANVAS_SIZE,
                      include_system: bool = False) -> List[Tuple[str, Tuple[int, int, int, int]]]:
    """Serialize a layout into a coarse, order-invariant signature for state matching."""
    signature = []
    for key, bbox in layout.items():
        if not include_system and key in ("back", "home"):
            continue
        signature.append((_normalize_label(key), _quantize_bbox(bbox, canvas_size)))
    signature.sort()
    return signature


def _average_hash(image: Image.Image, hash_size: int = 16) -> str:
    gray = image.convert("L").resize((hash_size, hash_size), Image.Resampling.LANCZOS)
    pixels = np.asarray(gray, dtype=np.float32)
    mean_value = float(pixels.mean())
    bits = ["1" if val >= mean_value else "0" for val in pixels.flatten()]
    return "".join(bits)


def _hamming_distance(hash_a: str, hash_b: str) -> int:
    if not hash_a or not hash_b or len(hash_a) != len(hash_b):
        return max(len(hash_a), len(hash_b), 9999)
    return sum(ch_a != ch_b for ch_a, ch_b in zip(hash_a, hash_b))


def _layout_similarity(sig_a: List[Tuple[str, Tuple[int, int, int, int]]],
                       sig_b: List[Tuple[str, Tuple[int, int, int, int]]]) -> float:
    if not sig_a and not sig_b:
        return 1.0
    set_a = set(sig_a)
    set_b = set(sig_b)
    overlap = len(set_a & set_b)
    union = max(len(set_a | set_b), 1)
    return overlap / union


def _label_tokens(label: str) -> List[str]:
    return [tok for tok in _normalize_label(label).split() if len(tok) >= 2]


def _is_launcher_package(app_name: str) -> bool:
    normalized = _normalize_label(app_name)
    if not normalized or normalized == "unknown":
        return False
    launcher_tokens = (
        "launcher",
        "launcher3",
        "quickstep",
        "trebuchet",
        "pixel",
        "one ui home",
        "oneuihome",
        "miui home",
        "miuihome",
        "systemui",
        "desktop",
        "springboard",
        "home",
    )
    return any(token in normalized for token in launcher_tokens)


def _token_similarity(tokens_a: List[str], tokens_b: List[str]) -> float:
    set_a = set(tokens_a)
    set_b = set(tokens_b)
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / max(len(set_a | set_b), 1)


def _page_layout_summary(layout: dict,
                         canvas_size: Tuple[int, int] = OUTPUT_CANVAS_SIZE,
                         primary_app: str = "",
                         llm_semantics: Optional[dict] = None) -> dict:
    """Extract coarse page semantics used for page-family merging and reporting."""
    _, canvas_h = canvas_size
    title_candidates = []
    nav_candidates = []
    icon_labels = []
    component_types = set()
    generic_icon_titles = 0

    for key, bbox in layout.items():
        if key in ("back", "home"):
            continue
        normalized = _normalize_label(key)
        tokens = _label_tokens(key)
        center_y = (bbox[1] + bbox[3]) / 2 if len(bbox) == 4 else 0
        width = max(0, bbox[2] - bbox[0]) if len(bbox) == 4 else 0
        height = max(0, bbox[3] - bbox[1]) if len(bbox) == 4 else 0

        if center_y <= canvas_h * 0.22 and tokens:
            title_candidates.append((width * height, normalized))

        if center_y >= canvas_h * 0.82:
            nav_candidates.append(normalized)

        is_small = width <= canvas_size[0] * 0.22 and height <= canvas_size[1] * 0.18
        if is_small:
            icon_labels.append(normalized)
            if normalized.startswith("icon "):
                generic_icon_titles += 1

        if any(tok in normalized for tok in ("search", "input", "field", "keyboard")):
            component_types.add("Search/Input")
        elif any(tok in normalized for tok in ("tab", "nav", "menu", "bottom")):
            component_types.add("Navigation Bar")
        elif any(tok in normalized for tok in ("list", "feed", "item", "row")):
            component_types.add("List/Feed")
        elif any(tok in normalized for tok in ("card", "product", "detail", "content")):
            component_types.add("Content Card")
        elif any(tok in normalized for tok in ("button", "buy", "checkout", "continue", "submit")):
            component_types.add("Action Button")
        elif any(tok in normalized for tok in ("image", "photo", "banner")):
            component_types.add("Media")
        else:
            component_types.add("Interactive Element")

    title_candidates.sort(reverse=True)
    page_title = title_candidates[0][1].title() if title_candidates else "Untitled Page"

    unique_icons = sorted({label.title() for label in icon_labels if label and label != "unknown"})
    unique_nav = sorted({label.title() for label in nav_candidates if label})
    if unique_nav:
        component_types.add("Navigation Bar")

    title_tokens = _label_tokens(page_title)
    launcher_app = _is_launcher_package(primary_app)
    icon_count = len(unique_icons)
    is_search_apps = "search" in title_tokens and "apps" in title_tokens
    title_has_apps = "apps" in title_tokens or "all" in title_tokens
    has_search_component = "Search/Input" in component_types
    component_count = len(component_types)
    icon_density = icon_count / max(1, component_count)
    dense_icon_grid = icon_count >= 6 and (
        icon_density >= 2.0
        or generic_icon_titles >= max(3, icon_count // 3)
    )
    is_app_drawer_like = (
        icon_count >= 6 and (
            is_search_apps
            or (title_has_apps and has_search_component)
            or (has_search_component and dense_icon_grid and generic_icon_titles >= max(2, icon_count // 4))
        )
    )
    is_home_like = (
        dense_icon_grid and (
            launcher_app
            or page_title in ("Untitled Page",)
            or page_title.startswith("Icon ")
            or generic_icon_titles >= max(3, icon_count // 3)
            or icon_count >= 8
        )
    )
    launcher_surface = is_home_like or is_app_drawer_like
    if launcher_app and icon_count >= 5:
        launcher_surface = True

    page_family = "home_page" if launcher_surface else "content_page"
    semantic_page_name = "Home Page" if page_family == "home_page" else page_title

    application_id = "launcher" if page_family == "home_page" else _normalize_application_id(primary_app)
    application_name = "Launcher" if application_id == "launcher" else (
        primary_app.replace("_", " ").replace(".", " ").strip().title() or "Unknown App"
    )
    logical_page_name = semantic_page_name

    llm_semantics = llm_semantics or {}
    llm_family = _normalize_page_family(str(llm_semantics.get("page_family", "")))
    if llm_family != "other" or llm_semantics.get("semantic_page_name") or llm_semantics.get("logical_page_name"):
        if llm_family != "other":
            page_family = llm_family
        if llm_semantics.get("semantic_page_name"):
            semantic_page_name = _stable_title(llm_semantics.get("semantic_page_name", ""), semantic_page_name)
        if llm_semantics.get("logical_page_name"):
            logical_page_name = _stable_title(llm_semantics.get("logical_page_name", ""), semantic_page_name)
        else:
            logical_page_name = semantic_page_name
        application_id = _normalize_application_id(
            llm_semantics.get("application_id", application_id),
            fallback=application_id,
        )
        application_name = _stable_title(
            llm_semantics.get("application_name", ""),
            application_name,
        )
        launcher_app = bool(llm_semantics.get("is_launcher_surface", False)) or _is_launcher_page_family(page_family)
        if launcher_app:
            page_family = "home_page"
            application_id = "launcher"
            application_name = "Launcher"
            is_home_like = True
        if _is_launcher_page_family(page_family):
            logical_page_name = "Home Page"
            semantic_page_name = "Home Page"

    return {
        "page_title": page_title,
        "semantic_page_name": semantic_page_name,
        "logical_page_name": logical_page_name,
        "logical_page_tokens": _label_tokens(logical_page_name),
        "title_tokens": title_tokens,
        "nav_tokens": [tok for label in unique_nav for tok in _label_tokens(label)],
        "icons": unique_icons,
        "additional_icons": [],
        "components": sorted(component_types),
        "additional_components": [],
        "nav_items": unique_nav,
        "icon_count": icon_count,
        "page_family": page_family,
        "is_home_like": page_family == "home_page",
        "is_launcher_app": launcher_app,
        "dense_icon_grid": dense_icon_grid,
        "application_id": application_id,
        "application_name": application_name,
        "llm_confidence": float(llm_semantics.get("confidence", 0.0) or 0.0),
    }


def _page_similarity_details(page: dict, candidate: dict) -> dict:
    """Score whether two screenshots belong to the same canonical page."""
    profile = page["state_profile"]
    candidate_profile = candidate["state_profile"]

    layout_score = _layout_similarity(
        profile["layout_signature"],
        candidate_profile["layout_signature"],
    )
    header_score = _token_similarity(
        profile["layout_summary"]["title_tokens"],
        candidate_profile["layout_summary"]["title_tokens"],
    )
    nav_score = _token_similarity(
        profile["layout_summary"]["nav_tokens"],
        candidate_profile["layout_summary"]["nav_tokens"],
    )
    component_score = _token_similarity(
        candidate_profile["layout_summary"]["components"],
        profile["layout_summary"]["components"],
    )
    logical_name_score = _token_similarity(
        profile["layout_summary"].get("logical_page_tokens", []),
        candidate_profile["layout_summary"].get("logical_page_tokens", []),
    )
    hash_distance = _hamming_distance(profile["visual_hash"], candidate_profile["visual_hash"])
    visual_score = max(0.0, 1.0 - (hash_distance / 48.0))
    same_page_family = (
        profile["layout_summary"].get("page_family")
        == candidate_profile["layout_summary"].get("page_family")
    )
    both_home_like = (
        profile["layout_summary"].get("is_home_like", False)
        and candidate_profile["layout_summary"].get("is_home_like", False)
    )
    both_launcher_family = (
        profile["layout_summary"].get("is_launcher_app", False)
        and candidate_profile["layout_summary"].get("is_launcher_app", False)
    )
    icon_count_gap = abs(
        int(profile["layout_summary"].get("icon_count", 0))
        - int(candidate_profile["layout_summary"].get("icon_count", 0))
    )
    icon_score = _token_similarity(
        profile["layout_summary"].get("icons", []),
        candidate_profile["layout_summary"].get("icons", []),
    )
    home_similarity_bonus = 0.0
    if both_home_like:
        home_similarity_bonus += 0.08
        if same_page_family:
            home_similarity_bonus += 0.05
        if icon_count_gap <= 4:
            home_similarity_bonus += 0.04
        if icon_score >= 0.45:
            home_similarity_bonus += 0.07
        if both_launcher_family:
            home_similarity_bonus += 0.05

    confidence = (
        0.36 * layout_score
        + 0.18 * header_score
        + 0.12 * nav_score
        + 0.11 * component_score
        + 0.08 * icon_score
        + 0.10 * logical_name_score
        + 0.05 * visual_score
        + home_similarity_bonus
    )
    return {
        "confidence": confidence,
        "layout_score": layout_score,
        "header_score": header_score,
        "nav_score": nav_score,
        "component_score": component_score,
        "visual_score": visual_score,
        "logical_name_score": logical_name_score,
        "hash_distance": hash_distance,
        "icon_score": icon_score,
        "same_primary_app": profile["primary_app"] == candidate_profile["primary_app"],
        "same_page_family": same_page_family,
        "both_home_like": both_home_like,
        "both_launcher_family": both_launcher_family,
        "icon_count_gap": icon_count_gap,
        "is_uncertain": 0.60 <= confidence < 0.70,
    }


def _merge_page_summaries(base: dict, incoming: dict) -> dict:
    base_icons = set(base.get("icons", []))
    incoming_icons = set(incoming.get("icons", []))
    base_components = set(base.get("components", []))
    incoming_components = set(incoming.get("components", []))
    base_is_launcher = _is_launcher_page_family(base.get("page_family"))
    incoming_is_launcher = _is_launcher_page_family(incoming.get("page_family"))
    merged_family = (
        "home_page"
        if (base_is_launcher or incoming_is_launcher)
        else (
            base.get("page_family")
            if base.get("page_family") == incoming.get("page_family")
            else "home_page"
            if base.get("is_home_like") and incoming.get("is_home_like")
            else base.get("page_family") or incoming.get("page_family") or "content_page"
        )
    )
    semantic_page_name = (
        "Home Page"
        if merged_family == "home_page"
        else incoming.get("semantic_page_name") or base.get("semantic_page_name") or "Untitled Page"
    )
    logical_page_name = (
        "Home Page"
        if merged_family == "home_page"
        else incoming.get("logical_page_name") or base.get("logical_page_name") or semantic_page_name
    )
    application_id = (
        "launcher"
        if merged_family == "home_page"
        else incoming.get("application_id") or base.get("application_id") or "unknown_app"
    )
    application_name = (
        "Launcher"
        if merged_family == "home_page"
        else incoming.get("application_name") or base.get("application_name") or "Unknown App"
    )
    return {
        "page_title": base.get("page_title") or incoming.get("page_title") or "Untitled Page",
        "semantic_page_name": semantic_page_name,
        "logical_page_name": logical_page_name,
        "logical_page_tokens": sorted(
            set(_label_tokens(logical_page_name))
            | set(base.get("logical_page_tokens", []))
            | set(incoming.get("logical_page_tokens", []))
        ),
        "title_tokens": sorted(set(base.get("title_tokens", [])) | set(incoming.get("title_tokens", []))),
        "nav_tokens": sorted(set(base.get("nav_tokens", [])) | set(incoming.get("nav_tokens", []))),
        "icons": sorted(base_icons | incoming_icons),
        "additional_icons": sorted(set(base.get("additional_icons", [])) | (incoming_icons - base_icons)),
        "components": sorted(base_components | incoming_components),
        "additional_components": sorted(
            set(base.get("additional_components", [])) | (incoming_components - base_components)
        ),
        "nav_items": sorted(set(base.get("nav_items", [])) | set(incoming.get("nav_items", []))),
        "icon_count": max(int(base.get("icon_count", 0)), int(incoming.get("icon_count", 0))),
        "page_family": merged_family,
        "is_home_like": merged_family == "home_page" or bool(base.get("is_home_like")) or bool(incoming.get("is_home_like")),
        "application_id": application_id,
        "application_name": application_name,
        "llm_confidence": max(float(base.get("llm_confidence", 0.0)), float(incoming.get("llm_confidence", 0.0))),
    }


def _primary_app(step: dict) -> str:
    apps = step.get("apps") or []
    if isinstance(apps, list) and apps:
        return str(apps[0]).lower()
    return str(step.get("package_name", "") or "unknown").lower()


def _variation_defaults(preset: str) -> Tuple[float, int, int]:
    if preset == "strong":
        return 0.35, 8, 18
    if preset == "mild":
        return 0.18, 4, 10
    return 0.0, 0, 0


def _resolve_variation_config(args) -> dict:
    """Resolve CLI overrides into one deterministic variation config."""
    tint_strength, layout_jitter_px, corner_radius = _variation_defaults(args.variation_preset)
    if args.icon_color_jitter is not None:
        tint_strength = args.icon_color_jitter
    if args.layout_jitter_px is not None:
        layout_jitter_px = args.layout_jitter_px
    if args.icon_corner_radius is not None:
        corner_radius = args.icon_corner_radius
    return {
        "preset": args.variation_preset,
        "seed": args.variation_seed,
        "enabled": args.variation_preset != "none" or tint_strength > 0 or layout_jitter_px > 0 or corner_radius > 0,
        "icon_tint_strength": max(0.0, float(tint_strength)),
        "layout_jitter_px": max(0, int(layout_jitter_px)),
        "icon_corner_radius": max(0, int(corner_radius)),
    }


def _rng_for_step(base_seed: int, step_index: int, elem_index: int = 0) -> random.Random:
    return random.Random((base_seed + 1) * 100_003 + step_index * 9_973 + elem_index * 433)


def _enhance_crop_quality(crop: Image.Image) -> Image.Image:
    rgba = crop.convert("RGBA")
    alpha = rgba.getchannel("A")
    rgb = Image.merge("RGB", rgba.split()[:3])
    rgb = ImageOps.autocontrast(rgb, cutoff=1)
    rgb = ImageEnhance.Sharpness(rgb).enhance(1.35)
    rgb = ImageEnhance.Contrast(rgb).enhance(1.08)
    enhanced = Image.merge("RGBA", (*rgb.split(), alpha))
    bbox = alpha.getbbox()
    return enhanced.crop(bbox) if bbox else enhanced


def _rounded_alpha_mask(size: Tuple[int, int], radius: int) -> Optional[Image.Image]:
    if radius <= 0:
        return None
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle([0, 0, size[0] - 1, size[1] - 1], radius=radius, fill=255)
    return mask


def _tint_icon_crop(crop: Image.Image, rng: random.Random, strength: float) -> Image.Image:
    if strength <= 0:
        return crop
    rgba = crop.convert("RGBA")
    alpha = rgba.getchannel("A")
    grayscale = ImageOps.grayscale(Image.merge("RGB", rgba.split()[:3]))
    palette = [
        (64, 122, 232),
        (27, 148, 92),
        (210, 86, 64),
        (142, 84, 233),
        (224, 160, 32),
        (47, 163, 176),
    ]
    tint = rng.choice(palette)
    colorized = ImageOps.colorize(grayscale, black=(18, 18, 18), white=tint).convert("RGBA")
    mixed = Image.blend(rgba, colorized, min(0.6, strength))
    mixed.putalpha(alpha)
    return mixed


def _apply_visual_variation(elem: dict, variation_cfg: dict, step_index: int) -> Tuple[Image.Image, List[int], dict]:
    """Apply deterministic asset-level appearance/layout changes for AMEX variation control."""
    crop = elem["crop"].convert("RGBA")
    bbox = list(elem.get("bbox", [0, 0, 0, 0]))
    metadata = {"variation_preset": variation_cfg["preset"]}
    if not variation_cfg.get("enabled"):
        return crop, bbox, metadata

    rng = _rng_for_step(variation_cfg["seed"], step_index, elem.get("index", 0))

    if elem.get("type") == "icon" and variation_cfg["icon_tint_strength"] > 0:
        crop = _tint_icon_crop(crop, rng, variation_cfg["icon_tint_strength"])
        metadata["color_variant"] = True

    radius = min(variation_cfg["icon_corner_radius"], crop.size[0] // 4, crop.size[1] // 4)
    if elem.get("type") == "icon" and radius > 0:
        mask = _rounded_alpha_mask(crop.size, radius)
        if mask is not None:
            crop.putalpha(ImageChops.multiply(crop.getchannel("A"), mask))
            metadata["corner_radius"] = radius

    jitter_px = variation_cfg["layout_jitter_px"]
    if jitter_px > 0 and len(bbox) == 4:
        dx = rng.randint(-jitter_px, jitter_px)
        dy = rng.randint(-jitter_px, jitter_px)
        bbox = [bbox[0] + dx, bbox[1] + dy, bbox[2] + dx, bbox[3] + dy]
        metadata["layout_jitter"] = [dx, dy]

    return crop, bbox, metadata


def _postprocess_page_image(page_img: Image.Image) -> Image.Image:
    img = page_img.convert("RGB")
    img = ImageEnhance.Color(img).enhance(1.04)
    img = ImageEnhance.Contrast(img).enhance(1.05)
    img = ImageEnhance.Sharpness(img).enhance(1.2)
    return img


def _persist_extracted_assets(elements: List[dict], screenshot_name: str,
                              assets_dir: str, step_info: dict,
                              variation_cfg: Optional[dict] = None) -> List[dict]:
    """Persist extracted AMEX assets to disk and return asset-backed elements."""
    page_asset_dir = os.path.join(
        assets_dir,
        f"step_{step_info.get('step_index', 0):02d}_{os.path.splitext(screenshot_name)[0]}",
    )
    os.makedirs(page_asset_dir, exist_ok=True)

    asset_backed = []
    step_index = int(step_info.get("step_index", 0))
    for elem in elements:
        label_stub = _sanitize_filename(elem.get("label", ""), f"elem_{elem['index']:02d}")
        asset_name = f"{elem['index']:02d}_{elem['type']}_{label_stub}.png"
        asset_path = os.path.join(page_asset_dir, asset_name)
        enhanced_crop = _enhance_crop_quality(elem["crop"])
        varied_crop, varied_bbox, variation_meta = _apply_visual_variation(
            {**elem, "crop": enhanced_crop},
            variation_cfg or {"enabled": False, "preset": "none", "seed": 0, "icon_tint_strength": 0.0, "layout_jitter_px": 0, "icon_corner_radius": 0},
            step_index,
        )
        varied_crop.save(asset_path, compress_level=1)

        asset_elem = {k: v for k, v in elem.items() if k != "crop"}
        asset_elem["raw_bbox"] = list(elem.get("bbox", []))
        asset_elem["bbox"] = varied_bbox
        asset_elem["asset_path"] = asset_path
        asset_elem["asset_source"] = "trajectory_extracted"
        asset_elem["source_screenshot"] = screenshot_name
        asset_elem["variation"] = variation_meta
        asset_backed.append(asset_elem)

    return asset_backed


def _save_asset_manifest(output_dir: str, pages_detection_data: List[dict]):
    """Save one manifest describing all extracted assets used for AMEX composition."""
    manifest = []
    for page in pages_detection_data:
        for elem in page["elements"]:
            manifest.append({
                "page_id": page["page_id"],
                "trajectory_id": page.get("trajectory_id"),
                "screenshot": page["screenshot_name"],
                "step_index": page["step"].get("step_index"),
                "type": elem.get("type"),
                "label": elem.get("label"),
                "raw_bbox": elem.get("raw_bbox"),
                "bbox": elem.get("bbox"),
                "asset_path": elem.get("asset_path"),
                "asset_source": elem.get("asset_source"),
                "variation": elem.get("variation", {}),
            })

    with open(os.path.join(output_dir, "trajectory_assets_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def _cleanup_auxiliary_outputs(output_dir: str):
    """Remove intermediate files while keeping the useful visual debug overlays."""
    removable_files = (
        "trajectory_assets_manifest.json",
        "llm_page_semantics.json",
        "llm_pair_matches.json",
        "ui_structure_full.json",
        "ui_structure_layer_full.json",
        "ui_topology.json",
        "ui_topology_tree.json",
        "ui_page_graph.json",
        "ui_topology.txt",
    )

    for filename in removable_files:
        path = os.path.join(output_dir, filename)
        if os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass


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
# Use the full available width so the composed screen does not keep left/right gutters.
PHONE_CANVAS_W = OUTPUT_W
CANVAS_SIZE = (PHONE_CANVAS_W, PHONE_CANVAS_H)
CANVAS_W, CANVAS_H = CANVAS_SIZE
PHONE_OFFSET_X = 0
PHONE_OFFSET_Y = NAV_STRIP_H
GELAB_BACK_COLOR = (255, 200, 200)  # pink
GELAB_HOME_COLOR = (200, 255, 200)  # green
GELAB_BACK_BBOX = [4, 4, 4 + NAV_BTN_W, 4 + NAV_BTN_H]
GELAB_HOME_BBOX = [OUTPUT_W - 4 - NAV_BTN_W, 4, OUTPUT_W - 4, 4 + NAV_BTN_H]


def _build_position_layout_entries(elements: List[dict],
                                   orig_size: Tuple[int, int],
                                   target_size: Tuple[int, int] = CANVAS_SIZE) -> List[dict]:
    """Create deterministic, unique element layout entries for one rendered page."""
    counts: Dict[str, int] = {}
    entries: List[dict] = []

    for elem in elements:
        bbox = elem.get("bbox") or []
        if len(bbox) != 4:
            continue
        x1, y1, x2, y2 = [int(round(v)) for v in bbox]
        if x2 - x1 < 5 or y2 - y1 < 5:
            continue

        action_name = _unique_layout_name(
            elem.get("label", ""),
            elem.get("type", "elem"),
            elem.get("index", len(entries)),
            counts,
        )
        scaled_bbox = _clip_bbox_to_canvas(
            _scale_bbox_to_box([x1, y1, x2, y2], orig_size, target_size),
            target_size,
        )
        if not _has_valid_bbox(scaled_bbox):
            continue

        entries.append({
            **elem,
            "orig_bbox": [x1, y1, x2, y2],
            "bbox": [int(v) for v in scaled_bbox],
            "action_name": action_name,
            "source_label": str(elem.get("label", "")),
        })

    return entries


def _sync_rendered_elements_with_layout(rendered_elements: List[dict],
                                        layout: dict,
                                        canvas_size: Tuple[int, int] = OUTPUT_CANVAS_SIZE) -> List[dict]:
    """Update tracked rendered elements so their boxes match the latest layout dict."""
    synced = []
    seen = set()
    for elem in rendered_elements:
        action_name = elem.get("action_name")
        if not action_name or action_name in seen or action_name not in layout:
            continue
        clipped_bbox = _clip_bbox_to_canvas(layout[action_name], canvas_size)
        if not _has_valid_bbox(clipped_bbox):
            continue
        synced.append({
            **elem,
            "bbox": [int(v) for v in clipped_bbox],
        })
        seen.add(action_name)
    return synced

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

PAGE_SEMANTIC_SYSTEM_PROMPT = """You analyze mobile UI screenshots for navigation graph construction.
Return ONLY a JSON object. Be concise, literal, and avoid speculation."""

PAGE_SEMANTIC_PROMPT = """\
Analyze this mobile screenshot and assign stable navigation labels.

Return ONLY valid JSON with this schema:
{
  "page_family": "home_page|content_page|search_page|list_page|detail_page|settings_page|form_page|confirmation_page|other",
  "semantic_page_name": "human readable short name",
  "logical_page_name": "stable canonical page name used to merge visually different variants of the same page",
  "application_name": "app or launcher name",
  "application_id": "short_normalized_identifier",
  "is_launcher_surface": true,
  "confidence": 0.0
}

Rules:
- Android launcher surfaces, including all-apps screens, must use page_family "home_page", application_name "Launcher", and application_id "launcher".
- If two screenshots would reasonably merge to the same node, they should share the same logical_page_name.
- Use short stable names like "Home Page", "Search Results", "Product Detail", "Settings".
- Do not use the file name as evidence.

Context:
{context}
"""

PAGE_MATCH_SYSTEM_PROMPT = """You compare two mobile UI screenshots for navigation graph canonicalization.
Return ONLY a JSON object. Prefer false when uncertain."""

PAGE_MATCH_PROMPT = """\
Decide whether these two mobile screenshots should map to the same logical navigation node.

Return ONLY valid JSON with this schema:
{
  "same_logical_page": true,
  "same_application": true,
  "shared_page_family": "home_page|content_page|search_page|list_page|detail_page|settings_page|form_page|confirmation_page|other",
  "canonical_application_id": "launcher",
  "canonical_logical_page_name": "Home Page",
  "confidence": 0.0,
  "reason": "short reason"
}

Rules:
- Different visual variants of the same Android launcher surface, including all-apps screens, count as the same logical page.
- Two screens inside the same app are NOT the same logical page unless they show the same page/state.
- If they are from different applications and not both launcher surfaces, usually return same_logical_page false.
- Prefer false when unsure.

Page A:
{page_a}

Page B:
{page_b}
"""


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


def _query_multimodal_gpt(client: OpenAI,
                          model_name: str,
                          image_paths: List[str],
                          prompt: str,
                          system_prompt: str,
                          max_completion_tokens: int = 1200) -> str:
    """Send one or more images plus a prompt to GPT and return text response."""
    content = [{"type": "text", "text": prompt}]
    for image_path in image_paths:
        if image_path and os.path.exists(image_path):
            content.append({"type": "image_url", "image_url": {"url": _encode_image_base64(image_path)}})

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ],
        max_completion_tokens=max_completion_tokens,
    )
    choice = response.choices[0]
    message_content = choice.message.content
    if message_content is None:
        finish = choice.finish_reason
        refusal = getattr(choice.message, "refusal", None)
        print(f"\n  GPT empty response: finish={finish}, refusal={refusal}")
        return ""
    return message_content.strip()


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


def _fit_image_to_box(image: Image.Image,
                      target_size: Tuple[int, int],
                      bg_color: Tuple[int, int, int] = BG_WHITE
                      ) -> Tuple[Image.Image, float, int, int]:
    """Resize an image to fill the target canvas without left/right gutters."""
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
        """Return an extracted AMEX asset resized to w x h."""
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
        clipped_bbox = _clip_bbox_to_canvas(
            _scale_bbox_to_box(bbox, (ow, oh), CANVAS_SIZE),
            CANVAS_SIZE,
        )
        if _has_valid_bbox(clipped_bbox):
            scaled_layout[key] = clipped_bbox

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


def _load_json_cache(cache_path: Optional[str]) -> dict:
    if not cache_path or not os.path.exists(cache_path):
        return {}
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_json_cache(cache_path: Optional[str], payload: dict):
    if not cache_path:
        return
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)


def _page_cache_key(page: dict) -> str:
    return "::".join([
        str(page.get("trajectory_id", "")),
        str(page.get("trajectory_local_page_index", "")),
        str(page.get("screenshot_name", page.get("page_id", ""))),
    ])


def _pair_cache_key(page_a: dict, page_b: dict) -> str:
    left, right = sorted([_page_cache_key(page_a), _page_cache_key(page_b)])
    return f"{left}||{right}"


def _normalize_page_family(value: str) -> str:
    normalized = _normalize_label(value)
    aliases = {
        "home page": "home_page",
        "home screen": "home_page",
        "home": "home_page",
        "launcher": "home_page",
        "launcher home": "home_page",
        "app drawer": "home_page",
        "appdrawer": "home_page",
        "all apps": "home_page",
        "search": "search_page",
        "list": "list_page",
        "detail": "detail_page",
        "settings": "settings_page",
        "form": "form_page",
        "confirmation": "confirmation_page",
        "content": "content_page",
    }
    normalized = aliases.get(normalized, normalized.replace(" ", "_"))
    allowed = {
        "home_page",
        "content_page",
        "search_page",
        "list_page",
        "detail_page",
        "settings_page",
        "form_page",
        "confirmation_page",
        "other",
    }
    return normalized if normalized in allowed else "other"


def _is_launcher_page_family(value: str) -> bool:
    return _normalize_page_family(str(value or "")) == "home_page"


def _normalize_application_id(value: str, fallback: str = "unknown_app") -> str:
    normalized = _normalize_label(value).replace(" ", "_")
    return normalized if normalized and normalized != "unknown" else fallback


def _stable_title(text: str, fallback: str) -> str:
    cleaned = " ".join(str(text or "").strip().split())
    return cleaned[:80] if cleaned else fallback


def _page_context_for_llm(page: dict, heuristic_summary: dict) -> str:
    step = page.get("step", {})
    visible_keys = sorted(list((page.get("layout") or {}).keys()))[:18]
    elements = page.get("elements") or []
    visible_labels = sorted({
        str(elem.get("label", "")).strip()
        for elem in elements
        if str(elem.get("label", "")).strip()
    })[:18]
    context = {
        "task": step.get("task", ""),
        "task_instruction": step.get("task_instruction", ""),
        "raw_apps": step.get("apps", []),
        "package_name": step.get("package_name", ""),
        "current_instruction": step.get("low_level_instruction", ""),
        "prev_instruction": step.get("prev_instruction", ""),
        "next_instruction": step.get("next_instruction", ""),
        "heuristic_page_family": heuristic_summary.get("page_family", "other"),
        "heuristic_page_title": heuristic_summary.get("page_title", "Untitled Page"),
        "heuristic_icons": heuristic_summary.get("icons", [])[:12],
        "layout_keys": visible_keys,
        "detected_labels": visible_labels,
    }
    return json.dumps(context, ensure_ascii=True, indent=2)


def _normalize_llm_page_semantics(raw: dict, page: dict, heuristic_summary: dict) -> dict:
    if not isinstance(raw, dict):
        raw = {}

    family = _normalize_page_family(str(raw.get("page_family", heuristic_summary.get("page_family", "other"))))
    fallback_app = _primary_app(page.get("step", {}))
    app_name = _stable_title(raw.get("application_name", ""), fallback_app.replace("_", " ").title() or "Unknown App")
    app_id = _normalize_application_id(raw.get("application_id", app_name), _normalize_application_id(fallback_app))
    semantic_name = _stable_title(raw.get("semantic_page_name", ""), heuristic_summary.get("semantic_page_name", "Untitled Page"))
    logical_name = _stable_title(raw.get("logical_page_name", ""), semantic_name)
    is_launcher_surface = bool(raw.get("is_launcher_surface", False))
    is_app_drawer = bool(raw.get("is_app_drawer", False))

    if _is_launcher_page_family(family) or is_app_drawer:
        family = "home_page"
        semantic_name = "Home Page"
        logical_name = "Home Page"
        is_launcher_surface = True
        app_id = "launcher"
        app_name = "Launcher"

    try:
        confidence = float(raw.get("confidence", 0.0))
    except Exception:
        confidence = 0.0
    confidence = max(0.0, min(confidence, 1.0))

    return {
        "page_family": family,
        "semantic_page_name": semantic_name,
        "logical_page_name": logical_name,
        "application_name": app_name,
        "application_id": app_id,
        "is_launcher_surface": is_launcher_surface,
        "is_app_drawer": False,
        "confidence": confidence,
        "source": "llm",
    }


def _merge_llm_semantics(base: dict, incoming: dict) -> dict:
    if not base:
        return dict(incoming or {})
    if not incoming:
        return dict(base or {})

    base = dict(base)
    incoming = dict(incoming)
    base_conf = float(base.get("confidence", 0.0) or 0.0)
    incoming_conf = float(incoming.get("confidence", 0.0) or 0.0)
    chosen = incoming if incoming_conf >= base_conf else base
    other = base if chosen is incoming else incoming

    if _is_launcher_page_family(base.get("page_family")) or _is_launcher_page_family(incoming.get("page_family")):
        return {
            "page_family": "home_page",
            "semantic_page_name": "Home Page",
            "logical_page_name": "Home Page",
            "application_name": "Launcher",
            "application_id": "launcher",
            "is_launcher_surface": True,
            "is_app_drawer": False,
            "confidence": max(base_conf, incoming_conf),
            "source": "llm_merged",
        }

    merged = dict(chosen)
    merged["application_name"] = chosen.get("application_name") or other.get("application_name") or "Unknown App"
    merged["application_id"] = chosen.get("application_id") or other.get("application_id") or "unknown_app"
    merged["semantic_page_name"] = chosen.get("semantic_page_name") or other.get("semantic_page_name") or "Untitled Page"
    merged["logical_page_name"] = chosen.get("logical_page_name") or other.get("logical_page_name") or merged["semantic_page_name"]
    merged["confidence"] = max(base_conf, incoming_conf)
    merged["source"] = "llm_merged"
    return merged


def _classify_page_with_llm(client: OpenAI, model_name: str, page: dict) -> dict:
    screenshot_path = page.get("screenshot_path")
    if not screenshot_path or not os.path.exists(screenshot_path):
        return {}

    heuristic_summary = _page_layout_summary(
        page.get("layout", {}),
        OUTPUT_CANVAS_SIZE,
        primary_app=_primary_app(page.get("step", {})),
    )
    prompt = PAGE_SEMANTIC_PROMPT.replace(
        "{context}",
        _page_context_for_llm(page, heuristic_summary),
    )
    try:
        response = _query_multimodal_gpt(
            client,
            model_name,
            [screenshot_path],
            prompt,
            PAGE_SEMANTIC_SYSTEM_PROMPT,
            max_completion_tokens=900,
        )
        raw = _parse_json_response(response)
    except Exception as exc:
        print(f"\n  LLM page semantic classification failed for {page.get('page_id')}: {exc}")
        raw = {}
    return _normalize_llm_page_semantics(raw, page, heuristic_summary)


def _annotate_pages_with_llm_semantics(client: OpenAI,
                                       model_name: str,
                                       pages_data: List[dict],
                                       output_dir: str,
                                       enabled: bool = True):
    """Attach cached LLM page semantics to every composed page."""
    cache_path = os.path.join(output_dir, "llm_page_semantics.json")
    cache = _load_json_cache(cache_path)
    updated = False

    for idx, page in enumerate(pages_data, start=1):
        cache_key = _page_cache_key(page)
        semantics = cache.get(cache_key)
        heuristic_summary = _page_layout_summary(
            page.get("layout", {}),
            OUTPUT_CANVAS_SIZE,
            primary_app=_primary_app(page.get("step", {})),
        )
        if not semantics and enabled:
            print(f"  llm-semantics [{idx}/{len(pages_data)}] {page.get('page_id')}...", end="", flush=True)
            semantics = _classify_page_with_llm(client, model_name, page)
            print(f" {semantics.get('page_family', 'other')} / {semantics.get('application_id', 'unknown')}")
        normalized_semantics = _normalize_llm_page_semantics(semantics or {}, page, heuristic_summary)
        if cache.get(cache_key) != normalized_semantics:
            cache[cache_key] = normalized_semantics
            updated = True
        semantics = normalized_semantics
        page["llm_semantics"] = semantics or {}

    if updated:
        _save_json_cache(cache_path, cache)


def _pair_context_for_llm(page: dict) -> str:
    summary = page.get("llm_semantics") or {}
    heuristic = _page_layout_summary(
        page.get("layout", {}),
        OUTPUT_CANVAS_SIZE,
        primary_app=_primary_app(page.get("step", {})),
    )
    payload = {
        "page_id": page.get("page_id"),
        "trajectory_id": page.get("trajectory_id", ""),
        "raw_package": _primary_app(page.get("step", {})),
        "llm_page_family": summary.get("page_family", ""),
        "llm_logical_page_name": summary.get("logical_page_name", ""),
        "llm_application_id": summary.get("application_id", ""),
        "heuristic_page_family": heuristic.get("page_family", ""),
        "heuristic_title": heuristic.get("page_title", ""),
        "visible_icons": heuristic.get("icons", [])[:12],
        "layout_keys": sorted(list((page.get("layout") or {}).keys()))[:18],
    }
    return json.dumps(payload, ensure_ascii=True, indent=2)


def _normalize_llm_pair_match(raw: dict,
                              page: dict,
                              candidate: dict) -> dict:
    if not isinstance(raw, dict):
        raw = {}
    try:
        confidence = float(raw.get("confidence", 0.0))
    except Exception:
        confidence = 0.0
    confidence = max(0.0, min(confidence, 1.0))
    family = _normalize_page_family(str(raw.get("shared_page_family", "")))
    same_logical_page = bool(raw.get("same_logical_page", False))
    same_application = bool(raw.get("same_application", False))
    canonical_application_id = _normalize_application_id(
        str(raw.get("canonical_application_id", "")),
        fallback=_normalize_application_id(
            (page.get("llm_semantics") or {}).get("application_id", "")
            or (candidate.get("llm_semantics") or {}).get("application_id", "")
            or _primary_app(page.get("step", {}))
        ),
    )
    canonical_logical_page_name = _stable_title(
        raw.get("canonical_logical_page_name", ""),
        (page.get("llm_semantics") or {}).get("logical_page_name", "")
        or (candidate.get("llm_semantics") or {}).get("logical_page_name", "")
        or "Unknown Page",
    )
    if _is_launcher_page_family(family):
        family = "home_page"
        canonical_application_id = "launcher"
        canonical_logical_page_name = "Home Page"

    return {
        "same_logical_page": same_logical_page,
        "same_application": same_application,
        "shared_page_family": family,
        "canonical_application_id": canonical_application_id,
        "canonical_logical_page_name": canonical_logical_page_name,
        "confidence": confidence,
        "reason": str(raw.get("reason", "")).strip()[:200],
        "source": "llm_pair_match",
    }


def _match_pages_with_llm(client: OpenAI,
                          model_name: str,
                          page: dict,
                          candidate: dict,
                          pair_cache: dict,
                          pair_cache_path: Optional[str] = None) -> dict:
    cache_key = _pair_cache_key(page, candidate)
    if cache_key in pair_cache:
        return pair_cache[cache_key]

    image_paths = [page.get("screenshot_path", ""), candidate.get("screenshot_path", "")]
    prompt = PAGE_MATCH_PROMPT.replace(
        "{page_a}",
        _pair_context_for_llm(page),
    ).replace(
        "{page_b}",
        _pair_context_for_llm(candidate),
    )
    try:
        response = _query_multimodal_gpt(
            client,
            model_name,
            image_paths,
            prompt,
            PAGE_MATCH_SYSTEM_PROMPT,
            max_completion_tokens=900,
        )
        raw = _parse_json_response(response)
    except Exception as exc:
        print(f"\n  LLM pair match failed for {page.get('page_id')} vs {candidate.get('page_id')}: {exc}")
        raw = {}

    normalized = _normalize_llm_pair_match(raw, page, candidate)
    pair_cache[cache_key] = normalized
    _save_json_cache(pair_cache_path, pair_cache)
    return normalized


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


def _ensure_system_layout(layout: Optional[dict]) -> dict:
    """Ensure every composed page exposes GE-Lab back/home controls in layout."""
    normalized_layout = {}
    for key, bbox in (layout or {}).items():
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            normalized_layout[key] = [int(v) for v in bbox]
    normalized_layout["back"] = list(GELAB_BACK_BBOX)
    normalized_layout["home"] = list(GELAB_HOME_BBOX)
    return normalized_layout


def _wrap_phone_canvas(phone_canvas: Image.Image) -> Image.Image:
    """Place the rendered page directly under the fixed top navigation strip."""
    canvas = phone_canvas.convert("RGB") if phone_canvas.mode != "RGB" else phone_canvas.copy()
    if canvas.size != CANVAS_SIZE:
        canvas, _, _, _ = _fit_image_to_box(canvas, CANVAS_SIZE, BG_WHITE)

    final_canvas = Image.new("RGB", OUTPUT_CANVAS_SIZE, BG_WHITE)
    final_canvas.paste(canvas, (PHONE_OFFSET_X, PHONE_OFFSET_Y))
    return final_canvas


def _draw_system_nav_overlay(image: Image.Image) -> Image.Image:
    """Draw the fixed GE-Lab top navigation strip with back/home buttons."""
    canvas = image.convert("RGB") if image.mode != "RGB" else image.copy()

    if canvas.size == CANVAS_SIZE:
        canvas = _wrap_phone_canvas(canvas)
    elif canvas.size != OUTPUT_CANVAS_SIZE:
        canvas = _wrap_phone_canvas(canvas)

    draw = ImageDraw.Draw(canvas)
    font = _try_load_font(12)

    draw.rectangle([0, 0, OUTPUT_W, NAV_STRIP_H], fill=(245, 245, 248))
    draw.rounded_rectangle(GELAB_BACK_BBOX, radius=6, fill=GELAB_BACK_COLOR, outline=(220, 170, 170))
    draw.rounded_rectangle(GELAB_HOME_BBOX, radius=6, fill=GELAB_HOME_COLOR, outline=(160, 210, 160))
    draw.text((GELAB_BACK_BBOX[0] + 8, GELAB_BACK_BBOX[1] + 6), "back", fill=TEXT_BLACK, font=font)
    draw.text((GELAB_HOME_BBOX[0] + 6, GELAB_HOME_BBOX[1] + 6), "home", fill=TEXT_BLACK, font=font)

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

def build_structure(pages_data: List[dict], trajectories: List[dict],
                    output_dir: str,
                    state_match_threshold: float = DEFAULT_STATE_MATCH_THRESHOLD,
                    layout_match_iou: float = DEFAULT_LAYOUT_MATCH_IOU,
                    client: Optional[OpenAI] = None,
                    model_name: str = "",
                    enable_llm_pair_matching: bool = True,
                    pair_match_top_k: int = DEFAULT_LLM_PAGE_MATCH_TOP_K,
                    canonical_pages: Optional[List[dict]] = None,
                    canonical_page_id_map: Optional[Dict[str, str]] = None) -> dict:
    """Build GE-Lab compatible structure files from rendered pages.

    Produces ui_structure.json and ui_structure_layer.json matching the exact
    GE-Lab format used by env_utils.py, generate_sft_data.py, and evaluate.py.

    pages_data: list of {"page_id": str, "layout": dict}
    trajectories: AMEX annotation dicts merged into one unified graph
    """
    episode_ids = [traj.get("episode_id", "") for traj in trajectories]
    tasks = [traj.get("task_info", {}).get("task", "") for traj in trajectories]
    categories = [traj.get("task_info", {}).get("category", "") for traj in trajectories]
    apps = sorted({
        str(app)
        for traj in trajectories
        for app in (traj.get("task_info", {}).get("app", []) or [])
        if app
    })

    # Merge repeated AMEX states across traces before serializing the final graph.
    if canonical_pages is None or canonical_page_id_map is None:
        canonical_pages, page_id_map = _deduplicate_pages(
            pages_data,
            state_match_threshold=state_match_threshold,
            layout_match_iou=layout_match_iou,
            client=client,
            model_name=model_name,
            enable_llm_pair_matching=enable_llm_pair_matching,
            pair_match_top_k=pair_match_top_k,
            pair_cache_path=os.path.join(output_dir, "llm_pair_matches.json"),
        )
        canonical_pages, page_id_map, _ = _renumber_canonical_pages(canonical_pages, page_id_map)
    else:
        canonical_pages = [dict(page) for page in canonical_pages]
        page_id_map = dict(canonical_page_id_map)
    for page in canonical_pages:
        page["layout"] = _ensure_system_layout(page.get("layout", {}))
        if isinstance(page.get("image"), Image.Image):
            page["image"], page["layout"] = _ensure_system_nav_controls(page["image"], page["layout"])
        if isinstance(page.get("base_image"), Image.Image):
            page["base_image"], _ = _ensure_system_nav_controls(page["base_image"], {})

    root_trace_page_id = _detect_home_page_id(pages_data)
    root_page_id = page_id_map.get(root_trace_page_id, root_trace_page_id)

    ui_structure = {"pages": {}, "metadata": {
        "source": "sim2real_compose",
        "episode_id": episode_ids[0] if len(episode_ids) == 1 else "",
        "episode_ids": episode_ids,
        "trajectory_count": len(trajectories),
        "task": tasks[0] if len(set(filter(None, tasks))) == 1 and tasks else "",
        "tasks": tasks,
        "category": categories[0] if len(set(filter(None, categories))) == 1 and categories else "",
        "categories": categories,
        "apps": apps,
        "total_pages": len(canonical_pages),
        "trace_pages": len(pages_data),
        "deduplicated_pages": len(pages_data) - len(canonical_pages),
        "canvas_size": list(OUTPUT_CANVAS_SIZE),
        "phone_canvas_size": list(CANVAS_SIZE),
        "state_match_threshold": state_match_threshold,
        "layout_match_iou": layout_match_iou,
        "variation": pages_data[0].get("variation_cfg", {}) if pages_data else {},
        "page_similarity_rule": "merge trajectory page_0 into a shared union root, relocate overlapping new bboxes to empty space, and merge non-root pages only when non-system UI elements share the same positions",
    }}

    canonical_index = {page["page_id"]: idx for idx, page in enumerate(canonical_pages)}
    # Keep a trace-layout -> canonical-layout key map so actions still point to merged nodes.
    canonical_trace_mapping = {}
    for canonical_page in canonical_pages:
        canonical_trace_mapping.update(canonical_page.get("trace_action_mappings", {}))

    for page in canonical_pages:
        layout_typed = {}
        for key, bbox in page["layout"].items():
            layout_typed[key] = {"bbox": bbox, "type": "system" if key in ("back", "home") else "normal"}
        ui_structure["pages"][page["page_id"]] = {
            "image": f"{page['page_id']}.png",
            "depth": canonical_index[page["page_id"]],
            "step_id": page.get("step", {}).get("step_id", canonical_index[page["page_id"]] + 1),
            "source_step_index": page.get("step", {}).get("step_index", canonical_index[page["page_id"]] + 1),
            "aliases": page.get("aliases", []),
            "trace_steps": [int(s) for s in page.get("trace_steps", []) if s is not None],
            "trajectory_ids": page.get("trajectory_ids", []),
            "state_profile": {
                "primary_app": page["state_profile"]["primary_app"],
                "layout_signature_size": len(page["state_profile"]["layout_signature"]),
                "visual_hash": page["state_profile"]["visual_hash"],
            },
            "llm_semantics": page.get("llm_semantics", {}),
            "page_summary": page["state_profile"].get("layout_summary", {}),
            "merge_notes": page.get("merge_notes", []),
            "layout": layout_typed,
            "transitions": [],
        }

    for i, trace_page in enumerate(pages_data):
        source_page_id = page_id_map[trace_page["page_id"]]
        source_page = ui_structure["pages"][source_page_id]
        step = trace_page.get("step", {})
        orig_size = tuple(trace_page.get("orig_size", (720, 1280)))
        trace_layout = _ensure_system_layout(trace_page.get("layout", {}))
        trace_page["layout"] = trace_layout
        next_trace_page = trace_page.get("next_trace_page_id") or trace_page["page_id"]
        target_trace_page = next_trace_page if next_trace_page in page_id_map else trace_page["page_id"]
        target_page_id = page_id_map[target_trace_page]
        transition = _resolve_transition(step, trace_layout, orig_size, target_page_id, i + 1)
        used_system_targets = set()

        action_key_mapping = canonical_trace_mapping.get(trace_page["page_id"], {})

        if transition.get("action_source") == "layout":
            mapped_action = action_key_mapping.get(transition.get("action"))
            if mapped_action:
                transition["action"] = mapped_action
                transition["icon_bbox"] = source_page["layout"].get(mapped_action, {}).get("bbox", transition.get("icon_bbox", [0, 0, 0, 0]))

        raw_action = _normalize_raw_action_name(step.get("action", "")).upper()
        transition["action_kind"] = ACTION_KIND_MAP.get(raw_action, raw_action if raw_action else "UNKNOWN")
        transition = _hydrate_transition_canvas_geometry(transition)
        transition["normalized_icon_bbox"] = _normalize_bbox_to_canvas(transition.get("icon_bbox", [0, 0, 0, 0]))
        transition["normalized_action_bbox"] = _normalize_bbox_to_canvas(transition["canvas_action_bbox"])
        transition["normalized_action_point"] = _normalize_point_to_canvas(transition["canvas_action_point"])
        transition["gesture_direction"] = _infer_gesture_direction(step, raw_action, orig_size)
        transition["source_page"] = source_page_id
        transition["source_trace_page"] = trace_page["page_id"]
        transition["source_trajectory_id"] = trace_page.get("trajectory_id", "")
        transition["source_step_indices"] = [int(step.get("step_index", i + 1))]

        action_payload = {}
        if raw_action in ("TYPE", "PRESS_ENTER"):
            action_payload["text"] = str(step.get("type_text", ""))
        if raw_action == "SCROLL":
            action_payload["direction"] = transition["gesture_direction"]
        transition["action_payload"] = action_payload

        _append_unique_transition(source_page["transitions"], transition)
        if transition.get("action") in ("back", "home"):
            used_system_targets.add(transition["action"])

        prev_trace_page = trace_page.get("prev_trace_page_id") or ""
        if prev_trace_page and "back" not in used_system_targets:
            prev_target_page = page_id_map.get(prev_trace_page, prev_trace_page)
            back_transition = _build_system_transition(
                raw_action="PRESS_BACK",
                action="back",
                target_page=prev_target_page,
                icon_bbox=trace_layout.get("back", GELAB_BACK_BBOX),
            )
            back_transition["source_page"] = source_page_id
            back_transition["source_trace_page"] = trace_page["page_id"]
            back_transition["source_trajectory_id"] = trace_page.get("trajectory_id", "")
            back_transition["source_step_indices"] = [int(step.get("step_index", i + 1))]
            _append_unique_transition(source_page["transitions"], back_transition)

        if root_page_id and "home" not in used_system_targets:
            home_transition = _build_system_transition(
                raw_action="PRESS_HOME",
                action="home",
                target_page=root_page_id,
                icon_bbox=trace_layout.get("home", GELAB_HOME_BBOX),
            )
            home_transition["source_page"] = source_page_id
            home_transition["source_trace_page"] = trace_page["page_id"]
            home_transition["source_trajectory_id"] = trace_page.get("trajectory_id", "")
            home_transition["source_step_indices"] = [int(step.get("step_index", i + 1))]
            _append_unique_transition(source_page["transitions"], home_transition)

    page_graph = _build_page_graph(ui_structure["pages"], root_page_id=root_page_id)
    logical_depths = _compute_group_depths(page_graph)
    for page_id, page_node in page_graph.get("canonical_pages", {}).items():
        if page_id not in ui_structure["pages"]:
            continue
        logical_page_id = page_node.get("logical_page_id", page_id)
        logical_page = page_graph.get("pages", {}).get(logical_page_id, {})
        ui_structure["pages"][page_id]["depth"] = logical_depths.get(
            logical_page_id,
            ui_structure["pages"][page_id].get("depth", 0),
        )
        ui_structure["pages"][page_id]["page_name"] = page_node.get("page_name", page_id)
        ui_structure["pages"][page_id]["detected_icons"] = page_node.get("detected_icons", [])
        ui_structure["pages"][page_id]["merged_elements"] = page_node.get("merged_elements", [])
        ui_structure["pages"][page_id]["outgoing_navigation"] = logical_page.get("outgoing_navigation", [])
        ui_structure["pages"][page_id]["logical_page_id"] = logical_page_id
        ui_structure["pages"][page_id]["application_id"] = page_node.get("application_id", "")
        ui_structure["pages"][page_id]["application_name"] = page_node.get("application_name", "")

    # Build layer structure (matches GE-Lab ui_structure_layer.json format)
    canonical_pages_data = []
    for page in canonical_pages:
        canonical_pages_data.append({
            "page_id": page["page_id"],
            "layout": page["layout"],
        })
    layer = _build_layer_structure(
        canonical_pages_data,
        ui_structure["pages"],
        root_page_id=root_page_id,
    )
    topology_tree = _build_topology_tree(ui_structure["pages"], root_page_id=root_page_id)
    full_topology = _build_full_topology_graph(ui_structure["pages"], root_page_id=root_page_id)
    ui_structure["metadata"]["root_page_id"] = root_page_id
    ui_structure["metadata"]["root_group_id"] = page_graph.get("root_group_id", root_page_id)
    ui_structure["metadata"]["topology_type"] = "merged_page_tree"
    ui_structure["metadata"]["topology_tree"] = topology_tree["root"]
    ui_structure["metadata"]["topology_graph"] = full_topology
    ui_structure["metadata"]["page_graph_type"] = "page_aware_graph"
    ui_structure["metadata"]["page_id_to_group"] = page_graph.get("page_id_to_group", {})
    ui_structure["metadata"]["application_id_to_pages"] = page_graph.get("application_id_to_pages", {})
    ui_structure["metadata"]["logical_page_depths"] = logical_depths
    ui_structure["page_graph"] = {
        "root_page_id": root_page_id,
        "root_group_id": page_graph.get("root_group_id", root_page_id),
        "pages": page_graph.get("pages", {}),
        "canonical_pages": page_graph.get("canonical_pages", {}),
        "page_id_to_group": page_graph.get("page_id_to_group", {}),
        "application_id_to_pages": page_graph.get("application_id_to_pages", {}),
        "logical_page_depths": logical_depths,
    }
    structure_minimal = _serialize_ui_structure_minimal(ui_structure["pages"])
    layer_minimal = _serialize_ui_layer_minimal(layer, structure_minimal["pages"])

    # Save
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "ui_structure.json"), "w") as f:
        json.dump(structure_minimal, f, indent=2)
    with open(os.path.join(output_dir, "ui_structure_layer.json"), "w") as f:
        json.dump(layer_minimal, f, indent=2)
    _save_topology_visualization(ui_structure["pages"], os.path.join(output_dir, "ui_topology.png"))
    _save_action_debug_overlays(ui_structure["pages"], output_dir)

    return ui_structure


def _detect_home_page_id(pages_data: List[dict]) -> str:
    """Pick the most likely home/root page using the trajectory descriptions."""
    for pdata in pages_data:
        if int(pdata.get("trajectory_local_page_index", -1)) == 0:
            return pdata["page_id"]
    for pdata in pages_data:
        llm_semantics = pdata.get("llm_semantics", {})
        if llm_semantics.get("page_family") == "home_page":
            return pdata["page_id"]
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


def _bbox_area(bbox: List[int]) -> int:
    return max(0, int(bbox[2]) - int(bbox[0])) * max(0, int(bbox[3]) - int(bbox[1]))


def _bbox_geometry_similarity(box1: List[int], box2: List[int]) -> float:
    iou = _bbox_iou(box1, box2)
    distance = _bbox_center_distance(box1, box2)
    width_1 = max(1, int(box1[2]) - int(box1[0]))
    height_1 = max(1, int(box1[3]) - int(box1[1]))
    width_2 = max(1, int(box2[2]) - int(box2[0]))
    height_2 = max(1, int(box2[3]) - int(box2[1]))
    width_score = min(width_1, width_2) / max(width_1, width_2)
    height_score = min(height_1, height_2) / max(height_1, height_2)
    distance_norm = distance / max((width_1 + height_1 + width_2 + height_2) / 4.0, 1.0)
    distance_score = max(0.0, 1.0 - min(distance_norm, 2.0) / 2.0)
    return 0.60 * iou + 0.25 * ((width_score + height_score) / 2.0) + 0.15 * distance_score


def _layout_position_similarity(layout_a: dict, layout_b: dict) -> dict:
    """Compare two layouts using only non-system bbox geometry."""
    boxes_a = [
        [int(v) for v in bbox]
        for key, bbox in (layout_a or {}).items()
        if key not in ("back", "home") and len(bbox) == 4
    ]
    boxes_b = [
        [int(v) for v in bbox]
        for key, bbox in (layout_b or {}).items()
        if key not in ("back", "home") and len(bbox) == 4
    ]

    if not boxes_a and not boxes_b:
        return {
            "mean_score": 1.0,
            "incoming_coverage": 1.0,
            "canonical_coverage": 1.0,
            "matched_ratio": 1.0,
            "count_gap": 0,
            "min_match_score": 1.0,
            "exact_match": True,
        }
    if not boxes_a or not boxes_b:
        return {
            "mean_score": 0.0,
            "incoming_coverage": 0.0,
            "canonical_coverage": 0.0,
            "matched_ratio": 0.0,
            "count_gap": abs(len(boxes_a) - len(boxes_b)),
            "min_match_score": 0.0,
            "exact_match": False,
        }

    unmatched = list(range(len(boxes_b)))
    scores: List[float] = []
    matched_scores: List[float] = []
    for box_a in sorted(boxes_a, key=_bbox_area, reverse=True):
        best_j = None
        best_score = 0.0
        for box_idx in unmatched:
            score = _bbox_geometry_similarity(box_a, boxes_b[box_idx])
            if score > best_score:
                best_score = score
                best_j = box_idx
        scores.append(best_score)
        if best_j is not None and best_score >= 0.74:
            matched_scores.append(best_score)
            unmatched.remove(best_j)

    matched = len(matched_scores)
    mean_score = sum(scores) / max(len(boxes_a), len(boxes_b), 1)
    incoming_coverage = matched / max(len(boxes_a), 1)
    canonical_coverage = matched / max(len(boxes_b), 1)
    matched_ratio = matched / max(len(boxes_a), len(boxes_b), 1)
    min_match_score = min(matched_scores) if matched_scores else 0.0
    exact_match = (
        abs(len(boxes_a) - len(boxes_b)) <= 1
        and incoming_coverage >= 0.95
        and canonical_coverage >= 0.95
        and min_match_score >= 0.74
    )
    return {
        "mean_score": mean_score,
        "incoming_coverage": incoming_coverage,
        "canonical_coverage": canonical_coverage,
        "matched_ratio": matched_ratio,
        "count_gap": abs(len(boxes_a) - len(boxes_b)),
        "min_match_score": min_match_score,
        "exact_match": exact_match,
    }


def _clip_bbox_to_canvas(bbox: List[int], canvas_size: Tuple[int, int]) -> List[int]:
    """Hard-clip a bbox so the saved page/layout never extends outside the canvas."""
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return [0, 0, 0, 0]

    canvas_w = max(int(canvas_size[0]), 1)
    canvas_h = max(int(canvas_size[1]), 1)
    left = max(0, min(int(min(bbox[0], bbox[2])), canvas_w))
    top = max(0, min(int(min(bbox[1], bbox[3])), canvas_h))
    right = max(0, min(int(max(bbox[0], bbox[2])), canvas_w))
    bottom = max(0, min(int(max(bbox[1], bbox[3])), canvas_h))
    if right <= left or bottom <= top:
        return [0, 0, 0, 0]
    return [left, top, right, bottom]


def _bboxes_overlap(box1: List[int], box2: List[int], padding: int = 6) -> bool:
    return not (
        box1[2] + padding <= box2[0]
        or box2[2] + padding <= box1[0]
        or box1[3] + padding <= box2[1]
        or box2[3] + padding <= box1[1]
    )


def _find_non_overlapping_bbox(preferred_bbox: List[int],
                               occupied_bboxes: List[List[int]],
                               canvas_size: Tuple[int, int],
                               padding: int = 6) -> List[int]:
    """Move a bbox to the nearest empty spot when a merged element would collide."""
    candidate = _clip_bbox_to_canvas(preferred_bbox, canvas_size)
    if not _has_valid_bbox(candidate):
        return [0, 0, 0, 0]
    if not any(_bboxes_overlap(candidate, box, padding=padding) for box in occupied_bboxes):
        return candidate

    width = max(1, candidate[2] - candidate[0])
    height = max(1, candidate[3] - candidate[1])
    max_x = max(int(canvas_size[0]) - width, 0)
    max_y = max(int(canvas_size[1]) - height, 0)
    step = max(10, min(width, height, 32) // 2)

    x_positions = sorted(set([candidate[0]] + list(range(0, max_x + 1, step)) + [max_x]))
    y_positions = sorted(set([candidate[1]] + list(range(0, max_y + 1, step)) + [max_y]))
    ranked_positions = []
    for x1 in x_positions:
        for y1 in y_positions:
            ranked_positions.append((
                abs(x1 - candidate[0]) + abs(y1 - candidate[1]),
                abs((x1 + width / 2.0) - (candidate[0] + width / 2.0))
                + abs((y1 + height / 2.0) - (candidate[1] + height / 2.0)),
                x1,
                y1,
            ))
    ranked_positions.sort()

    for _, _, x1, y1 in ranked_positions:
        placed = [int(x1), int(y1), int(x1) + width, int(y1) + height]
        if not any(_bboxes_overlap(placed, box, padding=padding) for box in occupied_bboxes):
            return placed
    return candidate


def _stabilize_layout_boxes(layout: dict,
                            canvas_size: Tuple[int, int],
                            padding: int = 8) -> dict:
    """Keep merged interactive elements inside the screen and separate collisions."""
    stabilized = {}
    occupied: List[List[int]] = []

    movable_items = []
    for key, bbox in (layout or {}).items():
        if key in ("back", "home"):
            stabilized[key] = [int(v) for v in bbox]
            continue
        clipped_bbox = _clip_bbox_to_canvas(bbox, canvas_size)
        if _has_valid_bbox(clipped_bbox):
            movable_items.append((key, clipped_bbox))

    # Preserve the visual reading order and only move boxes that truly collide.
    movable_items.sort(key=lambda item: (item[1][1], item[1][0], -_bbox_area(item[1])))
    for key, bbox in movable_items:
        placed_bbox = _find_non_overlapping_bbox(bbox, occupied, canvas_size, padding=padding)
        if _has_valid_bbox(placed_bbox):
            stabilized[key] = placed_bbox
            occupied.append(placed_bbox)

    return stabilized


def _render_canonical_page_image(base_image: Image.Image, rendered_elements: List[dict]) -> Image.Image:
    """Overlay the currently merged element set on top of the canonical base image."""
    composed = base_image.convert("RGBA").copy()
    for elem in sorted(
        rendered_elements,
        key=lambda item: (_bbox_area(item.get("bbox", [0, 0, 0, 0])), item.get("index", 0)),
        reverse=True,
    ):
        bbox = _clip_bbox_to_canvas(elem.get("bbox") or [], base_image.size)
        if not _has_valid_bbox(bbox):
            continue
        width = max(1, int(bbox[2]) - int(bbox[0]))
        height = max(1, int(bbox[3]) - int(bbox[1]))
        asset_path = elem.get("asset_path")
        if not asset_path or not os.path.exists(asset_path):
            continue
        try:
            with Image.open(asset_path) as asset_handle:
                crop = asset_handle.convert("RGBA").resize((width, height), Image.LANCZOS)
            composed.alpha_composite(crop, (int(bbox[0]), int(bbox[1])))
        except Exception:
            continue
    return composed.convert("RGB")


def _layout_entry_similarity(key_a: str,
                             bbox_a: List[int],
                             key_b: str,
                             bbox_b: List[int],
                             prefer_position: bool = False) -> float:
    label_score = SequenceMatcher(None, _normalize_label(key_a), _normalize_label(key_b)).ratio()
    iou = _bbox_iou(bbox_a, bbox_b)
    distance = _bbox_center_distance(bbox_a, bbox_b)
    distance_score = max(0.0, 1.0 - (distance / 80.0))
    geometry_score = _bbox_geometry_similarity(bbox_a, bbox_b)
    if prefer_position:
        return max(iou, (0.75 * geometry_score) + (0.15 * label_score) + (0.10 * distance_score))
    return max(iou, (0.50 * label_score) + (0.35 * geometry_score) + (0.15 * distance_score))


def _merge_layouts_into_canonical(canonical_layout: dict,
                                  incoming_layout: dict,
                                  counts: Dict[str, int],
                                  match_iou_threshold: float,
                                  canvas_size: Tuple[int, int] = CANVAS_SIZE,
                                  prefer_position: bool = False) -> Tuple[dict, Dict[str, str], List[dict]]:
    """Merge another AMEX page layout into a canonical state and return key remapping."""
    merged_layout = dict(canonical_layout)
    key_mapping = {}
    relocation_notes = []

    for incoming_key, incoming_bbox in incoming_layout.items():
        if incoming_key in ("back", "home"):
            merged_layout[incoming_key] = incoming_bbox
            key_mapping[incoming_key] = incoming_key
            continue
        incoming_bbox = _clip_bbox_to_canvas(incoming_bbox, canvas_size)
        if not _has_valid_bbox(incoming_bbox):
            continue

        # Reuse an existing canonical element when label/position are close enough.
        best_key = None
        best_score = 0.0
        for existing_key, existing_bbox in merged_layout.items():
            if existing_key in ("back", "home"):
                continue
            score = _layout_entry_similarity(
                incoming_key,
                incoming_bbox,
                existing_key,
                existing_bbox,
                prefer_position=prefer_position,
            )
            if score > best_score:
                best_key = existing_key
                best_score = score

        if best_key is not None and best_score >= match_iou_threshold:
            merged_layout[best_key] = _clip_bbox_to_canvas([
                int(round((merged_layout[best_key][0] + incoming_bbox[0]) / 2)),
                int(round((merged_layout[best_key][1] + incoming_bbox[1]) / 2)),
                int(round((merged_layout[best_key][2] + incoming_bbox[2]) / 2)),
                int(round((merged_layout[best_key][3] + incoming_bbox[3]) / 2)),
            ], canvas_size)
            key_mapping[incoming_key] = best_key
            continue

        # Otherwise keep the new element so merged states accumulate all valid actions.
        new_key = _unique_layout_name(incoming_key, "merged", len(merged_layout), counts)
        occupied = [
            box for key, box in merged_layout.items()
            if key not in ("back", "home") and len(box) == 4
        ]
        placed_bbox = _find_non_overlapping_bbox(incoming_bbox, occupied, canvas_size)
        merged_layout[new_key] = placed_bbox
        key_mapping[incoming_key] = new_key
        if placed_bbox != incoming_bbox:
            relocation_notes.append({
                "incoming_key": incoming_key,
                "canonical_key": new_key,
                "from_bbox": [int(v) for v in incoming_bbox],
                "to_bbox": [int(v) for v in placed_bbox],
                "reason": "bbox_overlap",
            })

    merged_layout = _stabilize_layout_boxes(merged_layout, canvas_size, padding=10)
    return merged_layout, key_mapping, relocation_notes


def _is_root_trace_page(page: dict) -> bool:
    return int(page.get("trajectory_local_page_index", -1)) == 0


def _merge_rendered_elements_into_canonical(canonical_page: dict,
                                            incoming_page: dict,
                                            key_mapping: Dict[str, str]) -> List[dict]:
    """Merge tracked rendered element assets so canonical images can reflect union pages."""
    canonical_elements = []
    by_name = {}
    occupied_bboxes: List[List[int]] = []
    canvas_size = tuple(getattr(canonical_page.get("image"), "size", CANVAS_SIZE))
    root_union = bool(canonical_page.get("is_root_union", False))

    for elem in canonical_page.get("rendered_elements", []):
        action_name = elem.get("action_name")
        if not action_name:
            continue
        synced_bbox = _clip_bbox_to_canvas(
            canonical_page.get("layout", {}).get(action_name, elem.get("bbox", [0, 0, 0, 0])),
            canvas_size,
        )
        if not _has_valid_bbox(synced_bbox):
            continue
        synced = {
            **elem,
            "bbox": [int(v) for v in synced_bbox],
        }
        canonical_elements.append(synced)
        by_name[action_name] = synced
        if _has_valid_bbox(synced["bbox"]):
            occupied_bboxes.append([int(v) for v in synced["bbox"]])

    for incoming_elem in incoming_page.get("rendered_elements", []):
        source_name = incoming_elem.get("action_name")
        target_name = key_mapping.get(source_name)
        if not target_name or target_name in ("back", "home"):
            continue

        target_bbox = _clip_bbox_to_canvas(
            canonical_page.get("layout", {}).get(target_name, incoming_elem.get("bbox", [0, 0, 0, 0])),
            canvas_size,
        )
        if not _has_valid_bbox(target_bbox):
            continue
        if target_name in by_name:
            by_name[target_name]["bbox"] = [int(v) for v in target_bbox]
            continue

        placed_bbox = [int(v) for v in target_bbox]
        if root_union and _has_valid_bbox(placed_bbox):
            placed_bbox = _find_non_overlapping_bbox(
                placed_bbox,
                occupied_bboxes,
                canvas_size,
                padding=10,
            )
            canonical_page.setdefault("layout", {})[target_name] = placed_bbox

        merged_elem = {
            **incoming_elem,
            "action_name": target_name,
            "bbox": placed_bbox,
            "merged_from_page": incoming_page.get("page_id"),
        }
        canonical_elements.append(merged_elem)
        by_name[target_name] = merged_elem
        if _has_valid_bbox(placed_bbox):
            occupied_bboxes.append(placed_bbox)

    return canonical_elements


def _page_state_profile(page: dict) -> dict:
    page_img = page.get("image")
    llm_semantics = page.get("llm_semantics") or {}
    primary_app = llm_semantics.get("application_id") or _primary_app(page.get("step", {}))
    layout_summary = _page_layout_summary(
        page.get("layout", {}),
        OUTPUT_CANVAS_SIZE,
        primary_app=primary_app,
        llm_semantics=llm_semantics,
    )
    return {
        "primary_app": primary_app,
        "layout_signature": _layout_signature(page.get("layout", {}), OUTPUT_CANVAS_SIZE),
        "position_signature": sorted([
            _quantize_bbox(bbox, OUTPUT_CANVAS_SIZE)
            for key, bbox in (page.get("layout", {}) or {}).items()
            if key not in ("back", "home")
        ]),
        "visual_hash": _average_hash(page_img) if isinstance(page_img, Image.Image) else "",
        "layout_summary": layout_summary,
    }


def _find_matching_state(page: dict,
                         canonical_pages: List[dict],
                         state_match_threshold: float,
                         client: Optional[OpenAI] = None,
                         model_name: str = "",
                         enable_llm_pair_matching: bool = True,
                         pair_match_top_k: int = DEFAULT_LLM_PAGE_MATCH_TOP_K,
                         pair_cache: Optional[dict] = None,
                         pair_cache_path: Optional[str] = None) -> Optional[dict]:
    """Match pages with deterministic UI-element geometry rules.

    Rules:
    - Every trajectory-local `page_0` merges into one shared root union page.
    - Non-root pages merge only when their non-system UI elements occupy the
      same positions (allowing tiny bbox noise).
    """
    del state_match_threshold, client, model_name, enable_llm_pair_matching, pair_match_top_k, pair_cache, pair_cache_path

    page_is_root = _is_root_trace_page(page)
    if page_is_root:
        for candidate in canonical_pages:
            if candidate.get("is_root_union", False):
                page["_merge_strategy"] = {
                    "mode": "root_union",
                    "reason": "trajectory_page_0_union",
                    "geometry": _layout_position_similarity(page.get("layout", {}), candidate.get("layout", {})),
                }
                candidate.setdefault("merge_candidates", []).append({
                    "page_id": page.get("page_id"),
                    "confidence": 1.0,
                    "pair_verified": False,
                })
                return candidate
        return None

    best_match = None
    best_geometry = None
    best_score = 0.0
    for candidate in canonical_pages:
        if candidate.get("is_root_union", False):
            continue
        geometry = _layout_position_similarity(page.get("layout", {}), candidate.get("layout", {}))
        if not geometry["exact_match"]:
            continue
        score = geometry["mean_score"] + geometry["matched_ratio"]
        if score > best_score:
            best_score = score
            best_match = candidate
            best_geometry = geometry

    if best_match is not None:
        page["_merge_strategy"] = {
            "mode": "position_match",
            "reason": "ui_elements_share_same_positions",
            "geometry": best_geometry or {},
        }
        best_match.setdefault("merge_candidates", []).append({
            "page_id": page.get("page_id"),
            "confidence": round(best_score / 2.0, 4),
            "pair_verified": False,
        })
    return best_match


def _deduplicate_pages(pages_data: List[dict],
                       state_match_threshold: float,
                       layout_match_iou: float,
                       client: Optional[OpenAI] = None,
                       model_name: str = "",
                       enable_llm_pair_matching: bool = True,
                       pair_match_top_k: int = DEFAULT_LLM_PAGE_MATCH_TOP_K,
                       pair_cache_path: Optional[str] = None) -> Tuple[List[dict], Dict[str, str]]:
    """Collapse duplicate AMEX pages into canonical graph nodes across all traces."""
    canonical_pages = []
    page_id_map: Dict[str, str] = {}
    pair_cache = _load_json_cache(pair_cache_path)

    for page in pages_data:
        page = dict(page)
        page["state_profile"] = _page_state_profile(page)
        matched = _find_matching_state(
            page,
            canonical_pages,
            state_match_threshold,
            client=client,
            model_name=model_name,
            enable_llm_pair_matching=enable_llm_pair_matching,
            pair_match_top_k=pair_match_top_k,
            pair_cache=pair_cache,
            pair_cache_path=pair_cache_path,
        )

        if matched is None:
            # The first time a state appears, it becomes the canonical node.
            clipped_layout = {}
            for key, bbox in (page.get("layout", {}) or {}).items():
                clipped_bbox = _clip_bbox_to_canvas(bbox, OUTPUT_CANVAS_SIZE)
                if key in ("back", "home") or _has_valid_bbox(clipped_bbox):
                    clipped_layout[key] = clipped_bbox
            page["layout"] = clipped_layout
            page["aliases"] = [page["page_id"]]
            page["trace_steps"] = [page.get("step", {}).get("step_index")]
            page["trajectory_ids"] = [page.get("trajectory_id", "")]
            base_image_source = page.get("base_image")
            if not isinstance(base_image_source, Image.Image):
                base_image_source = page.get("image")
            page["base_image"] = base_image_source.copy() if isinstance(base_image_source, Image.Image) else base_image_source
            page["is_root_union"] = _is_root_trace_page(page)
            page["layout_key_counts"] = {}
            for key in page.get("layout", {}):
                page["layout_key_counts"][_sanitize_filename(key, key)] = 1
            page["trace_action_mappings"] = {
                page["page_id"]: {key: key for key in page.get("layout", {})}
            }
            page["merge_notes"] = []
            page["merge_candidates"] = []
            page["rendered_elements"] = _sync_rendered_elements_with_layout(
                page.get("rendered_elements", []),
                page.get("layout", {}),
                canvas_size=OUTPUT_CANVAS_SIZE,
            )
            canonical_pages.append(page)
            page_id_map[page["page_id"]] = page["page_id"]
            continue

        # Later duplicate states only contribute extra layout/actions into the canonical node.
        merge_strategy = page.get("_merge_strategy", {})
        prefer_position = merge_strategy.get("mode") == "position_match"
        merge_threshold = layout_match_iou if prefer_position else min(layout_match_iou, 0.58)
        canvas_size = tuple(getattr(matched.get("image"), "size", CANVAS_SIZE))
        merged_layout, key_mapping, relocation_notes = _merge_layouts_into_canonical(
            matched["layout"],
            page.get("layout", {}),
            matched["layout_key_counts"],
            merge_threshold,
            canvas_size=canvas_size,
            prefer_position=prefer_position,
        )
        matched["layout"] = merged_layout
        matched["rendered_elements"] = _merge_rendered_elements_into_canonical(matched, page, key_mapping)
        matched["rendered_elements"] = _sync_rendered_elements_with_layout(
            matched.get("rendered_elements", []),
            matched.get("layout", {}),
            canvas_size=canvas_size,
        )
        if isinstance(matched.get("base_image"), Image.Image):
            matched["image"] = _render_canonical_page_image(
                matched["base_image"],
                matched.get("rendered_elements", []),
            )
        matched["llm_semantics"] = _merge_llm_semantics(
            matched.get("llm_semantics", {}),
            page.get("llm_semantics", {}),
        )
        matched["state_profile"]["layout_summary"] = _merge_page_summaries(
            matched["state_profile"].get("layout_summary", {}),
            page["state_profile"].get("layout_summary", {}),
        )
        matched["state_profile"] = _page_state_profile(matched)
        matched["aliases"].append(page["page_id"])
        matched["trace_steps"].append(page.get("step", {}).get("step_index"))
        if page.get("trajectory_id") and page.get("trajectory_id") not in matched["trajectory_ids"]:
            matched["trajectory_ids"].append(page.get("trajectory_id"))
        matched["trace_action_mappings"][page["page_id"]] = key_mapping
        similarity = _page_similarity_details(page, matched)
        geometry = _layout_position_similarity(page.get("layout", {}), matched.get("layout", {}))
        merge_strategy = page.get("_merge_strategy", {})
        matched["merge_notes"].append({
            "merged_page_id": page["page_id"],
            "merge_mode": merge_strategy.get("mode", "unknown"),
            "confidence": round(similarity["confidence"], 4),
            "layout_score": round(similarity["layout_score"], 4),
            "header_score": round(similarity["header_score"], 4),
            "nav_score": round(similarity["nav_score"], 4),
            "component_score": round(similarity["component_score"], 4),
            "logical_name_score": round(similarity["logical_name_score"], 4),
            "position_match": geometry,
            "reason": merge_strategy.get("reason", "merged and expanded canonical node"),
            "relocated_elements": relocation_notes,
        })
        page_id_map[page["page_id"]] = matched["page_id"]

    _save_json_cache(pair_cache_path, pair_cache)
    return canonical_pages, page_id_map


def _renumber_canonical_pages(canonical_pages: List[dict],
                              page_id_map: Dict[str, str]) -> Tuple[List[dict], Dict[str, str], Dict[str, str]]:
    """Rewrite merged canonical node ids to dense page_0..page_n numbering."""
    old_to_new = {
        page["page_id"]: f"page_{idx}"
        for idx, page in enumerate(canonical_pages)
    }
    renumbered_pages = []
    for page in canonical_pages:
        page_copy = dict(page)
        page_copy["page_id"] = old_to_new.get(page["page_id"], page["page_id"])
        renumbered_pages.append(page_copy)

    renumbered_page_id_map = {
        trace_page_id: old_to_new.get(canonical_page_id, canonical_page_id)
        for trace_page_id, canonical_page_id in page_id_map.items()
    }
    return renumbered_pages, renumbered_page_id_map, old_to_new


def _transition_signature(transition: dict) -> Tuple:
    return (
        transition.get("raw_action"),
        transition.get("action_kind"),
        transition.get("action"),
        transition.get("target_page"),
        transition.get("gesture_direction"),
        transition.get("type_text", ""),
        tuple(transition.get("canvas_action_bbox", [])),
        tuple(transition.get("canvas_action_point", [])),
    )


def _append_unique_transition(transitions: List[dict], transition: dict):
    signature = _transition_signature(transition)
    for existing in transitions:
        if _transition_signature(existing) == signature:
            trace_steps = sorted(set(existing.get("source_step_indices", []) + transition.get("source_step_indices", [])))
            existing["source_step_indices"] = trace_steps
            return
    transitions.append(transition)


def _page_sort_key(page_id: str):
    match = re.search(r"(\d+)$", str(page_id))
    return (int(match.group(1)), str(page_id)) if match else (10**9, str(page_id))


def _page_display_name(page_id: str, page: dict) -> str:
    summary = page.get("page_summary", {})
    title = summary.get("semantic_page_name") or summary.get("page_title") or page_id
    return f"{title} ({page_id})"


def _normalize_app_group_name(app_name: str) -> str:
    normalized = _normalize_label(app_name)
    return normalized.replace(" ", "_") if normalized and normalized != "unknown" else "unknown_app"


def _application_group(page: dict) -> Tuple[str, str]:
    summary = page.get("page_summary", {})
    if _is_launcher_page_family(summary.get("page_family")):
        return "launcher", "Launcher"

    if summary.get("application_id"):
        return (
            _normalize_application_id(str(summary.get("application_id")), "unknown_app"),
            str(summary.get("application_name", "")).strip() or "Unknown App",
        )

    primary_app = (
        page.get("state_profile", {}).get("primary_app")
        or page.get("step", {}).get("package_name")
        or "unknown_app"
    )
    app_group_id = _normalize_app_group_name(str(primary_app))
    app_group_name = str(primary_app).replace("_", " ").replace(".", " ").strip().title() or "Unknown App"
    return app_group_id, app_group_name


def _logical_page_name(page: dict) -> str:
    summary = page.get("page_summary", {})
    name = (
        summary.get("logical_page_name")
        or summary.get("semantic_page_name")
        or summary.get("page_title")
        or page.get("page_id", "Page")
    )
    return str(name)


def _logical_page_group_id(page: dict) -> str:
    summary = page.get("page_summary", {})
    page_family = summary.get("page_family", "content_page")
    page_name = _logical_page_name(page)
    app_group_id, _ = _application_group(page)
    if _is_launcher_page_family(page_family):
        return page_family
    return f"{app_group_id}::{_normalize_label(page_name).replace(' ', '_') or page.get('page_id', 'page')}"


def _merged_element_names(page: dict) -> List[str]:
    """Return the merged set of non-system layout element names for one page node."""
    return sorted([
        key for key in page.get("layout", {})
        if key not in ("back", "home")
    ])


def _build_page_graph(pages: Dict[str, dict], root_page_id: Optional[str] = None) -> dict:
    """Build a page-aware graph with logical page grouping and application clusters."""
    if not pages:
        return {
            "root_page_id": None,
            "root_group_id": None,
            "page_id_to_group": {},
            "application_id_to_pages": {},
            "canonical_pages": {},
            "pages": {},
        }

    if root_page_id not in pages:
        root_page_id = min(pages.keys(), key=lambda pid: (pages[pid].get("depth", 0), _page_sort_key(pid)))

    canonical_pages = {}
    page_id_to_group = {}
    logical_pages: Dict[str, dict] = {}
    application_id_to_pages: Dict[str, set] = {}

    for page_id in sorted(pages, key=_page_sort_key):
        page = pages[page_id]
        summary = page.get("page_summary", {})
        detected_icons = sorted(set(summary.get("icons", [])) | set(summary.get("additional_icons", [])))
        merged_elements = _merged_element_names(page)
        page_name = _logical_page_name(page)
        app_group_id, app_group_name = _application_group(page)
        logical_group_id = _logical_page_group_id(page)

        canonical_pages[page_id] = {
            "page_id": page_id,
            "page_name": page_name,
            "detected_icons": detected_icons,
            "merged_elements": merged_elements,
            "outgoing_navigation": [],
            "aliases": page.get("aliases", []),
            "trace_steps": page.get("trace_steps", []),
            "trajectory_ids": page.get("trajectory_ids", []),
            "components": summary.get("components", []),
            "additional_components": summary.get("additional_components", []),
            "page_family": summary.get("page_family", "content_page"),
            "application_id": app_group_id,
            "application_name": app_group_name,
            "logical_page_id": logical_group_id,
        }
        page_id_to_group[page_id] = logical_group_id
        for alias in page.get("aliases", []):
            page_id_to_group[str(alias)] = logical_group_id

        logical_page = logical_pages.setdefault(logical_group_id, {
            "page_id": logical_group_id,
            "page_name": page_name,
            "detected_icons": set(),
            "merged_elements": set(),
            "outgoing_navigation": {},
            "member_page_ids": [],
            "aliases": [],
            "trace_steps": set(),
            "trajectory_ids": set(),
            "components": set(),
            "additional_components": set(),
            "page_family": summary.get("page_family", "content_page"),
            "application_id": app_group_id,
            "application_name": app_group_name,
        })
        logical_page["detected_icons"].update(detected_icons)
        logical_page["merged_elements"].update(merged_elements)
        logical_page["member_page_ids"].append(page_id)
        logical_page["aliases"].extend(page.get("aliases", []))
        logical_page["trace_steps"].update(int(step) for step in page.get("trace_steps", []) if step is not None)
        logical_page["trajectory_ids"].update(page.get("trajectory_ids", []))
        logical_page["components"].update(summary.get("components", []))
        logical_page["additional_components"].update(summary.get("additional_components", []))
        application_id_to_pages.setdefault(app_group_id, set()).add(logical_group_id)

    for page_id in sorted(pages, key=_page_sort_key):
        source_page = canonical_pages[page_id]
        source_group_id = source_page["logical_page_id"]
        branch_index = logical_pages[source_group_id]["outgoing_navigation"]

        for transition in pages[page_id].get("transitions", []):
            target_page = transition.get("target_page")
            if not target_page or target_page not in canonical_pages:
                continue
            target_group_id = canonical_pages[target_page]["logical_page_id"]
            if target_group_id == source_group_id:
                continue

            branch = branch_index.setdefault(target_group_id, {
                "target_page_id": target_group_id,
                "target_page_name": logical_pages[target_group_id]["page_name"],
                "target_application_id": logical_pages[target_group_id]["application_id"],
                "target_application_name": logical_pages[target_group_id]["application_name"],
                "actions": set(),
                "action_kinds": set(),
                "source_trace_pages": set(),
                "source_step_indices": set(),
                "source_canonical_pages": set(),
                "target_canonical_pages": set(),
                "transition_count": 0,
            })
            if transition.get("raw_action") or transition.get("action"):
                branch["actions"].add(str(transition.get("raw_action") or transition.get("action")))
            if transition.get("action_kind"):
                branch["action_kinds"].add(str(transition.get("action_kind")))
            branch["source_trace_pages"].add(str(transition.get("source_trace_page", page_id)))
            branch["source_step_indices"].update(int(idx) for idx in transition.get("source_step_indices", []) if idx is not None)
            branch["source_canonical_pages"].add(page_id)
            branch["target_canonical_pages"].add(target_page)
            branch["transition_count"] += 1

    for logical_group_id, logical_page in logical_pages.items():
        outgoing_navigation = []
        for target_group_id in sorted(logical_page["outgoing_navigation"]):
            branch = logical_page["outgoing_navigation"][target_group_id]
            outgoing_navigation.append({
                "target_page_id": branch["target_page_id"],
                "target_page_name": branch["target_page_name"],
                "target_application_id": branch["target_application_id"],
                "target_application_name": branch["target_application_name"],
                "actions": sorted(branch["actions"]),
                "action_kinds": sorted(branch["action_kinds"]),
                "source_trace_pages": sorted(branch["source_trace_pages"], key=_page_sort_key),
                "source_step_indices": sorted(branch["source_step_indices"]),
                "source_canonical_pages": sorted(branch["source_canonical_pages"], key=_page_sort_key),
                "target_canonical_pages": sorted(branch["target_canonical_pages"], key=_page_sort_key),
                "transition_count": branch["transition_count"],
            })
        logical_page["outgoing_navigation"] = outgoing_navigation
        logical_page["detected_icons"] = sorted(logical_page["detected_icons"])
        logical_page["merged_elements"] = sorted(logical_page["merged_elements"])
        logical_page["member_page_ids"] = sorted(logical_page["member_page_ids"], key=_page_sort_key)
        logical_page["aliases"] = sorted(set(logical_page["aliases"]), key=_page_sort_key)
        logical_page["trace_steps"] = sorted(logical_page["trace_steps"])
        logical_page["trajectory_ids"] = sorted(tid for tid in logical_page["trajectory_ids"] if tid)
        logical_page["components"] = sorted(logical_page["components"])
        logical_page["additional_components"] = sorted(logical_page["additional_components"])

    root_group_id = page_id_to_group.get(root_page_id, root_page_id)

    return {
        "root_page_id": root_page_id,
        "root_group_id": root_group_id,
        "page_id_to_group": page_id_to_group,
        "application_id_to_pages": {
            app_id: sorted(page_ids)
            for app_id, page_ids in sorted(application_id_to_pages.items())
        },
        "canonical_pages": canonical_pages,
        "pages": logical_pages,
    }


def _compute_group_depths(page_graph: dict) -> Dict[str, int]:
    """Compute BFS depths on logical merged-page graph from the chosen root."""
    logical_pages = page_graph.get("pages", {}) or {}
    root_group_id = page_graph.get("root_group_id")
    if not logical_pages:
        return {}
    if root_group_id not in logical_pages:
        root_group_id = sorted(logical_pages, key=_page_sort_key)[0]

    depths: Dict[str, int] = {root_group_id: 0}
    queue = [root_group_id]
    while queue:
        current = queue.pop(0)
        current_depth = depths[current]
        for branch in logical_pages.get(current, {}).get("outgoing_navigation", []):
            target = branch.get("target_page_id")
            if not target or target not in logical_pages or target in depths:
                continue
            depths[target] = current_depth + 1
            queue.append(target)

    for page_id in logical_pages:
        depths.setdefault(page_id, max(depths.values(), default=0) + 1)
    return depths


def _build_topology_tree(pages: Dict[str, dict], root_page_id: Optional[str] = None) -> dict:
    """Build a tree-oriented topology summary from the merged page graph."""
    page_graph = _build_page_graph(pages, root_page_id=root_page_id)
    graph_pages = page_graph.get("pages", {})
    root_page_id = page_graph.get("root_group_id")

    if not graph_pages:
        return {"root": None, "nodes": {}}

    def build_node(page_id: str, path: Tuple[str, ...]) -> dict:
        graph_node = graph_pages[page_id]
        children = []
        for branch in graph_node.get("outgoing_navigation", []):
            target = branch.get("target_page_id")
            if not target or target not in graph_pages or target == page_id or target in path:
                continue
            children.append({
                "via_actions": branch.get("actions", []),
                "via_action_kinds": branch.get("action_kinds", []),
                "branch": branch,
                "target": build_node(target, path + (page_id,)),
            })
        return {
            "page_id": page_id,
            "name": f"{graph_node.get('page_name', page_id)} ({page_id})",
            "page_name": graph_node.get("page_name", page_id),
            "icons": graph_node.get("detected_icons", []),
            "merged_elements": graph_node.get("merged_elements", []),
            "components": graph_node.get("components", []),
            "additional_components": graph_node.get("additional_components", []),
            "aliases": graph_node.get("aliases", []),
            "application_name": graph_node.get("application_name", ""),
            "member_page_ids": graph_node.get("member_page_ids", []),
            "outgoing_navigation": graph_node.get("outgoing_navigation", []),
            "children": children,
        }

    nodes = {page_id: build_node(page_id, tuple()) for page_id in sorted(graph_pages, key=_page_sort_key)}
    return {"root": nodes[root_page_id], "nodes": nodes}


def _build_full_topology_graph(pages: Dict[str, dict], root_page_id: Optional[str] = None) -> dict:
    """Build a full page-level topology graph that preserves every page and transition."""
    if not pages:
        return {"root_page_id": root_page_id, "nodes": {}, "edges": [], "page_count": 0, "transition_count": 0}

    if root_page_id not in pages:
        root_page_id = min(pages.keys(), key=lambda pid: (pages[pid].get("depth", 0), _page_sort_key(pid)))

    nodes = {}
    edges = []
    for page_id in sorted(pages, key=lambda pid: (pages[pid].get("depth", 0), _page_sort_key(pid))):
        page = pages[page_id]
        nodes[page_id] = {
            "page_id": page_id,
            "page_name": page.get("page_name", page_id),
            "application_id": page.get("application_id", ""),
            "application_name": page.get("application_name", ""),
            "logical_page_id": page.get("logical_page_id", page_id),
            "depth": int(page.get("depth", 0)),
            "aliases": page.get("aliases", []),
            "trajectory_ids": page.get("trajectory_ids", []),
            "trace_steps": page.get("trace_steps", []),
            "detected_icons": page.get("detected_icons", []),
            "merged_elements": page.get("merged_elements", []),
        }
        for transition in page.get("transitions", []):
            edges.append({
                "source_page": page_id,
                "source_page_name": page.get("page_name", page_id),
                "target_page": transition.get("target_page"),
                "target_page_name": pages.get(
                    transition.get("target_page", ""),
                    {},
                ).get("page_name", transition.get("target_page", "")),
                "raw_action": transition.get("raw_action", ""),
                "action": transition.get("action", ""),
                "action_kind": transition.get("action_kind", ""),
                "gesture_direction": transition.get("gesture_direction", ""),
                "source_trace_page": transition.get("source_trace_page", ""),
                "source_trajectory_id": transition.get("source_trajectory_id", ""),
                "source_step_indices": transition.get("source_step_indices", []),
                "action_payload": transition.get("action_payload", {}),
            })

    return {
        "root_page_id": root_page_id,
        "page_count": len(nodes),
        "transition_count": len(edges),
        "nodes": nodes,
        "edges": edges,
    }


def _summarize_structure_metadata(pages: Dict[str, dict]) -> dict:
    """Create ui_environment_448-style metadata for saved structure files."""
    if not pages:
        return {
            "total_pages": 0,
            "tree_depth": 0,
            "nodes_per_level": [],
        }

    depth_to_pages: Dict[int, List[str]] = {}
    for page_id, page in pages.items():
        depth_to_pages.setdefault(int(page.get("depth", 0)), []).append(page_id)

    max_depth = max(depth_to_pages) if depth_to_pages else 0
    nodes_per_level = [len(depth_to_pages.get(depth, [])) for depth in range(1, max_depth + 1)]
    return {
        "total_pages": len(pages),
        "tree_depth": max_depth + 1,
        "nodes_per_level": nodes_per_level,
    }


def _serialize_layout_minimal(layout: dict) -> dict:
    serialized = {}
    for key, value in (layout or {}).items():
        if isinstance(value, dict):
            serialized[key] = {
                "bbox": value.get("bbox", [0, 0, 0, 0]),
                "type": value.get("type", "normal"),
            }
        else:
            serialized[key] = {
                "bbox": value,
                "type": "system" if key in ("back", "home") else "normal",
            }
    return serialized


def _serialize_transitions_minimal(transitions: List[dict]) -> List[dict]:
    return [
        _serialize_transition_minimal(transition)
        for transition in (transitions or [])
    ]


def _serialize_ui_structure_minimal(pages: Dict[str, dict]) -> dict:
    """Convert rich page data into ui_environment_448-style ui_structure.json."""
    serialized_pages = {}
    for page_id, page in sorted(pages.items(), key=lambda item: (int(item[1].get("depth", 0)), _page_sort_key(item[0]))):
        serialized_pages[page_id] = {
            "image": page.get("image", f"{page_id}.png"),
            "depth": int(page.get("depth", 0)),
            "layout": _serialize_layout_minimal(page.get("layout", {})),
            "transitions": _serialize_transitions_minimal(page.get("transitions", [])),
        }
    return {
        "pages": serialized_pages,
        "metadata": _summarize_structure_metadata(serialized_pages),
    }


def _serialize_layer_node_minimal(node: Optional[dict]) -> Optional[dict]:
    if node is None:
        return None
    return {
        "image": node.get("image", ""),
        "depth": int(node.get("depth", 0)),
        "layout": _serialize_layout_minimal(node.get("layout", {})),
        "transitions": _serialize_transitions_minimal(node.get("transitions", [])),
        "subnodes": [
            child for child in (
                _serialize_layer_node_minimal(subnode)
                for subnode in node.get("subnodes", [])
            )
            if child is not None
        ],
    }


def _serialize_ui_layer_minimal(layer: dict, structure_pages: Dict[str, dict]) -> dict:
    """Convert rich layer data into ui_environment_448-style ui_structure_layer.json."""
    return {
        "root": _serialize_layer_node_minimal(layer.get("root")),
        "metadata": _summarize_structure_metadata(structure_pages),
    }


def _stored_transition_action(transition: dict) -> str:
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


def _debug_transition_label(idx: int, transition: dict) -> str:
    label_action = _debug_action_name(transition)
    raw_action = _normalize_raw_action_name(transition.get("raw_action", "")).upper()
    if raw_action in ("TYPE", "TEXT", "PRESS_ENTER"):
        type_text = str(transition.get("type_text", "") or "").strip()
        if type_text:
            shortened = type_text if len(type_text) <= 24 else f"{type_text[:21]}..."
            return f"{idx}:{label_action} {shortened}"
    return f"{idx}:{label_action}"


def _should_draw_non_spatial_debug_label(transition: dict) -> bool:
    raw_action = _normalize_raw_action_name(transition.get("raw_action", "")).upper()
    return raw_action in ("TYPE", "TEXT", "PRESS_ENTER", "TASK_COMPLETE", "TASK_IMPOSSIBLE")


def _stored_transition_action_coord(transition: dict) -> List[int]:
    raw_action = _normalize_raw_action_name(transition.get("raw_action", "")).upper()
    canvas_action_point = _safe_coord_pair(transition.get("canvas_action_point") or [])
    canvas_action_bbox = transition.get("canvas_action_bbox") or [0, 0, 0, 0]
    icon_bbox = transition.get("icon_bbox") or [0, 0, 0, 0]

    if raw_action in ("TYPE", "TEXT", "PRESS_ENTER", "TASK_COMPLETE", "TASK_IMPOSSIBLE"):
        return [0, 0]
    if raw_action in ("TAP", "CLICK") and _has_valid_bbox(icon_bbox):
        return _bbox_center_point(icon_bbox)
    if raw_action in ("SWIPE", "SCROLL", "TAP", "CLICK"):
        if _has_valid_point(canvas_action_point):
            return [int(canvas_action_point[0]), int(canvas_action_point[1])]
        if _has_valid_bbox(canvas_action_bbox):
            return _bbox_center_point(canvas_action_bbox)
    if raw_action in ("PRESS_BACK", "PRESS_HOME") and _has_valid_bbox(icon_bbox):
        return _bbox_center_point(icon_bbox)
    if _has_valid_bbox(icon_bbox):
        return _bbox_center_point(icon_bbox)
    if _has_valid_point(canvas_action_point):
        return [int(canvas_action_point[0]), int(canvas_action_point[1])]
    if _has_valid_bbox(canvas_action_bbox):
        return _bbox_center_point(canvas_action_bbox)
    return [0, 0]


def _serialize_transition_minimal(transition: dict) -> dict:
    raw_action = _normalize_raw_action_name(transition.get("raw_action", "")).upper()
    icon_bbox = transition.get("icon_bbox", [0, 0, 0, 0])
    if raw_action in ("TYPE", "TEXT", "PRESS_ENTER"):
        icon_bbox = [0, 0, 0, 0]
    action_coord = _stored_transition_action_coord(transition)
    item = {
        "action": _stored_transition_action(transition),
        "target_page": transition.get("target_page", ""),
    }
    if _has_valid_point(action_coord):
        item["action_coord"] = [int(action_coord[0]), int(action_coord[1])]
    if isinstance(icon_bbox, (list, tuple)) and len(icon_bbox) == 4 and _has_valid_bbox(icon_bbox):
        item["icon_bbox"] = [int(v) for v in icon_bbox]
    type_text = str(transition.get("type_text", "") or "").strip()
    if raw_action in ("TYPE", "TEXT", "PRESS_ENTER") and type_text:
        item["type_text"] = type_text
    if raw_action in ("SWIPE", "SCROLL") and transition.get("gesture_direction"):
        item["gesture_direction"] = transition.get("gesture_direction")
    return item


def _transition_debug_bbox(transition: dict, action_coord: List[int]) -> List[int]:
    raw_action = _normalize_raw_action_name(transition.get("raw_action", "")).upper()
    canvas_action_bbox = transition.get("canvas_action_bbox", [0, 0, 0, 0])
    icon_bbox = transition.get("icon_bbox", [0, 0, 0, 0])

    if raw_action in ("TYPE", "TEXT", "PRESS_ENTER", "TASK_COMPLETE", "TASK_IMPOSSIBLE"):
        return [0, 0, 0, 0]
    if raw_action in ("SWIPE", "SCROLL") and _has_valid_bbox(canvas_action_bbox):
        return [int(v) for v in canvas_action_bbox]
    if _has_valid_bbox(icon_bbox):
        return [int(v) for v in icon_bbox]
    if _has_valid_bbox(canvas_action_bbox):
        return [int(v) for v in canvas_action_bbox]
    if _has_valid_point(action_coord):
        px, py = [int(v) for v in action_coord]
        radius = 8
        return [px - radius, py - radius, px + radius, py + radius]
    return [0, 0, 0, 0]


def _save_action_debug_overlays(pages: Dict[str, dict], output_dir: str):
    """Save per-page overlays showing serialized action coordinates and target boxes."""
    debug_dir = os.path.join(output_dir, "action_coord_debug")
    os.makedirs(debug_dir, exist_ok=True)
    for existing in os.listdir(debug_dir):
        if existing.endswith(".png"):
            try:
                os.remove(os.path.join(debug_dir, existing))
            except OSError:
                pass

    palette = [
        ((230, 57, 70), (255, 230, 233)),
        ((29, 78, 216), (227, 238, 255)),
        ((22, 163, 74), (229, 255, 237)),
        ((217, 119, 6), (255, 245, 224)),
        ((126, 34, 206), (243, 232, 255)),
    ]
    font = _try_load_font(12)

    for page_id, page in pages.items():
        image_path = os.path.join(output_dir, "pages", f"{page_id}.png")
        if not os.path.exists(image_path):
            continue

        with Image.open(image_path) as img_handle:
            image = img_handle.convert("RGB")
        draw = ImageDraw.Draw(image)
        non_spatial_label_y = NAV_STRIP_H + 8

        for idx, transition in enumerate(page.get("transitions", [])):
            edge_color, label_bg = palette[idx % len(palette)]
            action_coord = transition.get("action_coord", _fallback_canvas_action_point(transition))
            if not _has_valid_point(action_coord):
                action_coord = _fallback_canvas_action_point(transition)
            debug_bbox = _transition_debug_bbox(transition, action_coord)
            label = _debug_transition_label(idx, transition)

            if _has_valid_bbox(debug_bbox):
                draw.rectangle(debug_bbox, outline=edge_color, width=3)
            if _has_valid_point(action_coord):
                px, py = [int(v) for v in action_coord]
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

        image.save(os.path.join(debug_dir, f"{page_id}.png"))


def _topology_action_label(transition: dict) -> str:
    raw_action = str(transition.get("raw_action", "") or "").strip()
    return raw_action or str(transition.get("action", "") or "").strip() or "navigate"


def _save_topology_visualization(pages: Dict[str, dict], output_path: str):
    """Render a cleaner canonical-page topology view with every page shown."""
    try:
        import matplotlib.pyplot as plt
        import networkx as nx
    except ImportError as exc:
        print(f"Topology PNG skipped: missing dependency ({exc})")
        return

    graph = nx.DiGraph()
    root_id = next(iter(sorted(pages, key=_page_sort_key)), None)
    for page_id, page in sorted(pages.items(), key=lambda item: (int(item[1].get("depth", 0)), _page_sort_key(item[0]))):
        graph.add_node(
            page_id,
            depth=int(page.get("depth", 0)),
            page_name=str(page.get("page_name", page_id)),
            application_id=str(page.get("application_id", "unknown_app")),
            member_count=max(1, len(page.get("aliases", []) or page.get("trajectory_ids", []))),
        )

        aggregated_edges: Dict[Tuple[str, str], dict] = {}
        for transition in page.get("transitions", []):
            target_id = transition.get("target_page")
            if not target_id or target_id == page_id:
                continue
            edge_key = (page_id, target_id)
            edge_meta = aggregated_edges.setdefault(edge_key, {"labels": set(), "weight": 0})
            edge_meta["labels"].add(_topology_action_label(transition)[:26])
            edge_meta["weight"] += 1

        for (_, target_id), edge_meta in aggregated_edges.items():
            label = ", ".join(sorted(edge_meta["labels"])[:2])
            if len(edge_meta["labels"]) > 2:
                label += "..."
            graph.add_edge(
                page_id,
                target_id,
                label=label or "navigate",
                weight=max(1, int(edge_meta["weight"])),
            )

    if not graph.nodes:
        return

    if root_id not in graph.nodes:
        root_id = sorted(graph.nodes(), key=_page_sort_key)[0]

    nodes_by_depth: Dict[int, List[str]] = {}
    for node, data in graph.nodes(data=True):
        nodes_by_depth.setdefault(int(data.get("depth", 0)), []).append(node)

    pos = {}
    for depth in sorted(nodes_by_depth):
        nodes = sorted(
            nodes_by_depth[depth],
            key=lambda node_id: (
                graph.nodes[node_id].get("application_id", ""),
                graph.nodes[node_id].get("page_name", node_id),
                _page_sort_key(node_id),
            ),
        )
        count = len(nodes)
        for idx, node in enumerate(nodes):
            pos[node] = (depth * 3.8, -(idx - (count - 1) / 2.0) * 2.3)

    palette = [
        "#8ecae6",
        "#b8e0a5",
        "#ffd6a5",
        "#f4b6c2",
        "#cdb4db",
        "#bde0fe",
        "#d9ed92",
        "#f1c0e8",
    ]
    app_ids = sorted({
        str(graph.nodes[node].get("application_id", "unknown_app"))
        for node in graph.nodes()
    })
    app_colors = {
        app_id: palette[idx % len(palette)]
        for idx, app_id in enumerate(app_ids)
    }
    node_colors = [
        "#ffd166" if node == root_id else app_colors.get(graph.nodes[node].get("application_id", "unknown_app"), "#8ecae6")
        for node in graph.nodes()
    ]
    node_sizes = [
        1500 + 180 * max(1, int(graph.nodes[node].get("member_count", 1)))
        for node in graph.nodes()
    ]
    labels = {
        node: (
            f"{str(graph.nodes[node].get('page_name', node))[:24]}\n"
            f"({node})"
        )
        for node in graph.nodes()
    }
    edge_labels = {
        (u, v): data.get("label", "navigate")
        for u, v, data in graph.edges(data=True)
    }
    edge_widths = [
        1.5 + 0.25 * max(1, int(data.get("weight", 1)))
        for _, _, data in graph.edges(data=True)
    ]

    fig_w = min(40, max(16, len(nodes_by_depth) * 4.4))
    fig_h = min(30, max(10, max(len(nodes) for nodes in nodes_by_depth.values()) * 1.5))
    plt.figure(figsize=(fig_w, fig_h))

    nx.draw_networkx_nodes(
        graph,
        pos,
        node_color=node_colors,
        node_size=node_sizes,
        linewidths=1.6,
        edgecolors="#264653",
    )
    nx.draw_networkx_labels(graph, pos, labels=labels, font_size=7, font_weight="bold")
    nx.draw_networkx_edges(
        graph,
        pos,
        edge_color="#6c757d",
        arrows=True,
        arrowsize=14,
        width=edge_widths,
        connectionstyle="arc3,rad=0.06",
    )
    nx.draw_networkx_edge_labels(
        graph,
        pos,
        edge_labels=edge_labels,
        font_size=6,
        rotate=False,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.82),
    )

    plt.title("Merged UI Topology", fontsize=14)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=220)
    plt.close()


def _scale_action_bbox_to_canvas(step: dict,
                                 raw_action: str,
                                 orig_size: Tuple[int, int]) -> List[int]:
    return _scale_bbox_from_step_to_canvas(step, _build_action_bbox(step, raw_action), orig_size)


def _scale_action_point_to_canvas(step: dict,
                                  raw_action: str,
                                  orig_size: Tuple[int, int]) -> List[int]:
    return _scale_step_coord_to_canvas(step, _build_action_point(step, raw_action), orig_size)


def _normalize_bbox_to_canvas(bbox: List[int],
                              canvas_size: Tuple[int, int] = OUTPUT_CANVAS_SIZE) -> List[int]:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return [0, 0, 0, 0]
    width = max(canvas_size[0], 1)
    height = max(canvas_size[1], 1)
    return [
        int(round(int(bbox[0]) * 1000 / width)),
        int(round(int(bbox[1]) * 1000 / height)),
        int(round(int(bbox[2]) * 1000 / width)),
        int(round(int(bbox[3]) * 1000 / height)),
    ]


def _normalize_point_to_canvas(point: List[int],
                               canvas_size: Tuple[int, int] = OUTPUT_CANVAS_SIZE) -> List[int]:
    if not isinstance(point, (list, tuple)) or len(point) != 2:
        return [0, 0]
    width = max(canvas_size[0], 1)
    height = max(canvas_size[1], 1)
    return [
        int(round(int(point[0]) * 1000 / width)),
        int(round(int(point[1]) * 1000 / height)),
    ]


def _has_valid_point(point: List[int]) -> bool:
    return isinstance(point, (list, tuple)) and len(point) == 2 and any(int(v) != 0 for v in point)


def _has_valid_bbox(bbox: List[int]) -> bool:
    return isinstance(bbox, (list, tuple)) and len(bbox) == 4 and any(int(v) != 0 for v in bbox)


def _bbox_center_point(bbox: List[int]) -> List[int]:
    if not _has_valid_bbox(bbox):
        return [0, 0]
    return [
        int(round((int(bbox[0]) + int(bbox[2])) / 2.0)),
        int(round((int(bbox[1]) + int(bbox[3])) / 2.0)),
    ]


def _fallback_canvas_action_bbox(transition: dict) -> List[int]:
    canvas_action_bbox = transition.get("canvas_action_bbox", [0, 0, 0, 0])
    if _has_valid_bbox(canvas_action_bbox):
        return [int(v) for v in canvas_action_bbox]

    icon_bbox = transition.get("icon_bbox", [0, 0, 0, 0])
    if _has_valid_bbox(icon_bbox):
        return [int(v) for v in icon_bbox]

    point = transition.get("canvas_action_point", [0, 0])
    if _has_valid_point(point):
        px, py = [int(v) for v in point]
        radius = 12
        return [px - radius, py - radius, px + radius, py + radius]
    return [0, 0, 0, 0]


def _fallback_canvas_action_point(transition: dict) -> List[int]:
    raw_action = _normalize_raw_action_name(transition.get("raw_action", "")).upper()
    canvas_action_point = transition.get("canvas_action_point", [0, 0])

    icon_bbox = transition.get("icon_bbox", [0, 0, 0, 0])
    if raw_action in ("TAP", "CLICK", "PRESS_BACK", "PRESS_HOME") and _has_valid_bbox(icon_bbox):
        return _bbox_center_point(icon_bbox)

    if _has_valid_point(canvas_action_point):
        return [int(v) for v in canvas_action_point]
    if _has_valid_bbox(icon_bbox):
        return _bbox_center_point(icon_bbox)

    canvas_action_bbox = transition.get("canvas_action_bbox", [0, 0, 0, 0])
    if _has_valid_bbox(canvas_action_bbox):
        return _bbox_center_point(canvas_action_bbox)
    return [0, 0]


def _hydrate_transition_canvas_geometry(transition: dict) -> dict:
    """Ensure every serialized transition has the best available canvas-space geometry."""
    transition["canvas_action_bbox"] = _fallback_canvas_action_bbox(transition)
    transition["canvas_action_point"] = _fallback_canvas_action_point(transition)
    transition["action_coord"] = list(transition["canvas_action_point"])
    return transition


def _normalize_raw_action_name(raw_action: str) -> str:
    return str(raw_action or "").strip()


def _infer_gesture_direction(step: dict, raw_action: str, orig_size: Tuple[int, int]) -> str:
    raw_action = _normalize_raw_action_name(raw_action).upper()
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


def _step_text(step: dict) -> str:
    """Collect all textual hints attached to one AMEX step."""
    return " ".join([
        str(step.get("low_level_instruction", "")),
        str(step.get("description", "")),
        str(step.get("intention", "")),
        str(step.get("info", "")),
        str(step.get("type_text", "")),
        str(step.get("task_instruction", "")),
        str(step.get("task", "")),
    ]).lower()


def _normalize_step_point(step: dict,
                          coord: List[int],
                          orig_size: Tuple[int, int]) -> Optional[Tuple[int, int]]:
    """Map a raw AMEX point into the current page/layout coordinate system."""
    if not isinstance(coord, (list, tuple)) or len(coord) != 2:
        return None

    x, y = coord
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


def _safe_coord_pair(coord: List[int]) -> List[int]:
    if not isinstance(coord, (list, tuple)) or len(coord) != 2:
        return [0, 0]
    return [int(coord[0]), int(coord[1])]


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
    return [
        min(max(PHONE_OFFSET_X, scaled[0]), PHONE_OFFSET_X + CANVAS_W - 1),
        min(max(PHONE_OFFSET_Y, scaled[1]), PHONE_OFFSET_Y + CANVAS_H - 1),
    ]


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
    return _clip_bbox_to_canvas(
        _scale_bbox_to_box(
            normalized_bbox,
            orig_size,
            CANVAS_SIZE,
            (PHONE_OFFSET_X, PHONE_OFFSET_Y),
        ),
        OUTPUT_CANVAS_SIZE,
    )


def _point_box(point: Tuple[int, int], radius: int = 12) -> List[int]:
    px, py = point
    return [px - radius, py - radius, px + radius, py + radius]


def _build_action_bbox(step: dict, raw_action: str) -> List[int]:
    """Build an action bbox in raw instruction-annotation coordinates."""
    raw_action = _normalize_raw_action_name(raw_action).upper()
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
    """Store the representative raw point for actions driven by a touch point."""
    raw_action = _normalize_raw_action_name(raw_action).upper()
    if raw_action in ("TYPE", "PRESS_BACK", "PRESS_HOME", "PRESS_ENTER", "TASK_COMPLETE", "TASK_IMPOSSIBLE"):
        return [0, 0]
    return _safe_coord_pair(step.get("touch_coord") or [])


def _bbox_contains_point(bbox: List[int], point: Tuple[int, int]) -> bool:
    return bbox[0] <= point[0] <= bbox[2] and bbox[1] <= point[1] <= bbox[3]


def _is_layout_target(target: Optional[str], layout: dict) -> bool:
    return bool(target) and target in layout


def _resolve_tap_target(step: dict,
                        layout: dict,
                        orig_size: Tuple[int, int]) -> Optional[str]:
    """Use a strict point-in-bbox match for tap-like actions."""
    point = _scale_step_coord_to_canvas(step, step.get("touch_coord") or [], orig_size)
    if point == [0, 0]:
        return None

    for key, bbox in layout.items():
        if _bbox_contains_point(bbox, (point[0], point[1])):
            return key
    return None


def _resolve_transition(step: dict,
                        layout: dict,
                        orig_size: Tuple[int, int],
                        target_page: str,
                        default_step_id: int) -> dict:
    """Serialize one AMEX step into a transition record."""
    raw_action = _normalize_raw_action_name(step.get("action", "")).upper()
    resolved_target = _find_action_target(step, layout, orig_size)
    strict_tap_target = None
    if raw_action in ("TAP", "CLICK"):
        strict_tap_target = _resolve_tap_target(step, layout, orig_size)

    touch_coord = _safe_coord_pair(step.get("touch_coord") or [])
    lift_coord = _safe_coord_pair(step.get("lift_coord") or [])
    canvas_touch = _scale_step_coord_to_canvas(step, touch_coord, orig_size)
    canvas_lift = _scale_step_coord_to_canvas(step, lift_coord, orig_size)
    action_bbox = _build_action_bbox(step, raw_action)
    canvas_action_bbox = _scale_action_bbox_to_canvas(step, raw_action, orig_size)
    action_point = _build_action_point(step, raw_action)
    canvas_action_point = _scale_action_point_to_canvas(step, raw_action, orig_size)
    gesture_direction = _infer_gesture_direction(step, raw_action, orig_size)

    transition = {
        "step_id": step.get("step_id", default_step_id),
        "step_index": step.get("step_index", default_step_id),
        "raw_action": raw_action,
        "action_kind": ACTION_KIND_MAP.get(raw_action, raw_action if raw_action else "UNKNOWN"),
        "action": resolved_target,
        "action_source": "layout" if _is_layout_target(resolved_target, layout) else "derived",
        "target_page": target_page,
        "action_bbox": action_bbox,
        "action_point": action_point,
        "touch_coord": touch_coord,
        "lift_coord": lift_coord,
        "canvas_action_bbox": canvas_action_bbox,
        "canvas_action_point": canvas_action_point,
        "canvas_touch_coord": canvas_touch,
        "canvas_lift_coord": canvas_lift,
        "normalized_action_bbox": _normalize_bbox_to_canvas(canvas_action_bbox),
        "normalized_action_point": _normalize_point_to_canvas(canvas_action_point),
        "normalized_touch_coord": _normalize_point_to_canvas(canvas_touch),
        "normalized_lift_coord": _normalize_point_to_canvas(canvas_lift),
        "icon_bbox": layout.get(resolved_target, [0, 0, 0, 0]),
        "gesture_direction": gesture_direction,
        "type_text": str(step.get("type_text", "")),
    }

    if raw_action in ("TAP", "CLICK"):
        if strict_tap_target is not None:
            transition["action"] = strict_tap_target
            transition["action_source"] = "layout"
            transition["icon_bbox"] = layout.get(strict_tap_target, [0, 0, 0, 0])
        else:
            transition["action"] = "tap"
            transition["action_source"] = "coord"
            transition["icon_bbox"] = canvas_action_bbox
    elif raw_action in ("TYPE", "TEXT"):
        transition["action"] = resolved_target if _is_layout_target(resolved_target, layout) else "type"
        transition["action_source"] = "text_input"
        transition["icon_bbox"] = [0, 0, 0, 0]
    elif raw_action in ("SWIPE", "SCROLL"):
        if isinstance(resolved_target, str) and resolved_target.startswith("scroll_"):
            transition["action"] = resolved_target
            transition["action_source"] = "gesture"
            transition["icon_bbox"] = canvas_action_bbox
        elif _is_layout_target(resolved_target, layout):
            transition["action"] = resolved_target
            transition["action_source"] = "layout"
            transition["icon_bbox"] = layout.get(resolved_target, [0, 0, 0, 0])
        else:
            transition["action"] = "scroll"
            transition["action_source"] = "gesture"
            transition["icon_bbox"] = canvas_action_bbox
    elif raw_action == "PRESS_ENTER":
        transition["action"] = resolved_target if _is_layout_target(resolved_target, layout) else "press_enter"
        transition["action_source"] = "enter_key"
        transition["icon_bbox"] = [0, 0, 0, 0]
    elif raw_action == "PRESS_BACK":
        transition["action"] = "back"
        transition["action_source"] = "system"
        transition["icon_bbox"] = layout.get("back", GELAB_BACK_BBOX)
    elif raw_action == "PRESS_HOME":
        transition["action"] = "home"
        transition["action_source"] = "system"
        transition["icon_bbox"] = layout.get("home", GELAB_HOME_BBOX)
    elif raw_action == "TASK_COMPLETE":
        transition["action"] = "complete"
        transition["action_source"] = "task"
        transition["icon_bbox"] = [0, 0, 0, 0]
    elif raw_action == "TASK_IMPOSSIBLE":
        transition["action"] = "impossible"
        transition["action_source"] = "task"
        transition["icon_bbox"] = [0, 0, 0, 0]

    transition["normalized_icon_bbox"] = _normalize_bbox_to_canvas(transition["icon_bbox"])
    transition["action_payload"] = {
        "text": transition["type_text"] if raw_action in ("TYPE", "TEXT", "PRESS_ENTER") else "",
        "direction": gesture_direction if raw_action in ("SWIPE", "SCROLL") else "",
    }
    return transition


def _build_system_transition(raw_action: str,
                             action: str,
                             target_page: str,
                             icon_bbox: List[int]) -> dict:
    return {
        "step_id": None,
        "step_index": None,
        "raw_action": raw_action,
        "action_kind": ACTION_KIND_MAP.get(raw_action, raw_action),
        "action": action,
        "action_source": "synthetic_system",
        "target_page": target_page,
        "action_bbox": [0, 0, 0, 0],
        "action_point": [0, 0],
        "touch_coord": [0, 0],
        "lift_coord": [0, 0],
        "canvas_action_bbox": [0, 0, 0, 0],
        "canvas_action_point": [0, 0],
        "canvas_touch_coord": [0, 0],
        "canvas_lift_coord": [0, 0],
        "normalized_action_bbox": [0, 0, 0, 0],
        "normalized_action_point": [0, 0],
        "normalized_touch_coord": [0, 0],
        "normalized_lift_coord": [0, 0],
        "icon_bbox": [int(v) for v in icon_bbox],
        "normalized_icon_bbox": _normalize_bbox_to_canvas(icon_bbox),
        "gesture_direction": "",
        "type_text": "",
        "action_payload": {},
        "synthetic": True,
    }


def _find_closest_layout_key(layout: dict,
                             target_box: List[int],
                             allow_system: bool = False
                             ) -> Tuple[Optional[str], float, float]:
    """Find the layout element with best overlap/center distance to target_box."""
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
    """Find the layout element or gesture most likely targeted by an AMEX step."""
    if not layout:
        return "unknown"

    action = _normalize_raw_action_name(step.get("action", "")).upper()
    info = str(step.get("info", "")).upper()
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
        if best_key is not None and (best_iou > 0 or best_distance <= 64):
            return best_key

    if action in ("TYPE", "TEXT"):
        if touch_point != [0, 0]:
            best_key, best_iou, best_distance = _find_closest_layout_key(
                layout, _point_box((touch_point[0], touch_point[1])), allow_system=False
            )
            if best_key is not None and (best_iou > 0 or best_distance <= 96):
                return best_key
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
        start_point = touch_point
        end_point = lift_point
        if start_point != [0, 0] and end_point != [0, 0]:
            start_box = _point_box((start_point[0], start_point[1]), radius=18)
            end_box = _point_box((end_point[0], end_point[1]), radius=18)
            best_start, start_iou, start_distance = _find_closest_layout_key(
                layout, start_box, allow_system=False
            )
            best_end, end_iou, end_distance = _find_closest_layout_key(
                layout, end_box, allow_system=False
            )
            if best_start is not None and (start_iou > 0 or start_distance <= 96):
                return best_start
            if best_end is not None and (end_iou > 0 or end_distance <= 96):
                return best_end

            dy = end_point[1] - start_point[1]
            dx = end_point[0] - start_point[0]
            if abs(dy) >= abs(dx):
                return "scroll_down" if dy > 0 else "scroll_up"
            return "scroll_right" if dx > 0 else "scroll_left"

        if "scroll down" in instruction or "swipe down" in instruction:
            return "scroll_down"
        if "scroll up" in instruction or "swipe up" in instruction:
            return "scroll_up"
        if "scroll left" in instruction or "swipe left" in instruction:
            return "scroll_left"
        if "scroll right" in instruction or "swipe right" in instruction:
            return "scroll_right"

    if action == "PRESS_BACK":
        return "back"

    if action == "PRESS_HOME":
        return "home"

    if action == "TASK_COMPLETE":
        return "complete"

    if action == "TASK_IMPOSSIBLE":
        return "impossible"

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
                           pages_full: Dict[str, dict],
                           root_page_id: Optional[str] = None) -> dict:
    """Build layer structure matching GE-Lab ui_structure_layer.json format.

    Each node has: image, depth, layout, one per-step transition, subnodes.
    """
    if not pages_data:
        return {"root": None, "metadata": {}}

    page_order = [pdata["page_id"] for pdata in pages_data]

    def build_node(page_id, visited):
        visited.add(page_id)
        page_data = pages_full.get(page_id, {})

        transitions = page_data.get("transitions", [])

        node = {
            "image": f"{page_id}.png",
            "depth": page_data.get("depth", 0),
            "layout": page_data.get("layout", {}),
            "transitions": transitions,
            "subnodes": [],
        }
        for transition in transitions:
            child_id = transition.get("target_page")
            if child_id in pages_full and child_id not in visited:
                node["subnodes"].append(build_node(child_id, visited))
        return node

    visited = set()
    root_id = root_page_id if root_page_id in pages_full else page_order[0]
    root = build_node(root_id, visited)
    for page_id in page_order[1:]:
        if page_id not in visited:
            root["subnodes"].append(build_node(page_id, visited))

    return {
        "root": root,
        "metadata": {
            "type": "amex_unified_graph",
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
    """Build AMEX-step-aware context for one page composition step."""
    steps = trajectory.get("steps", [])
    step = dict(steps[step_idx])
    task_info = trajectory.get("task_info", {})

    prev_step = steps[step_idx - 1] if step_idx > 0 else None
    next_step = steps[step_idx + 1] if step_idx + 1 < len(steps) else None

    step["task"] = task_info.get("task", "") or trajectory.get("task", "") or trajectory.get("instruction", "")
    step["task_instruction"] = task_info.get("instruction", "") or trajectory.get("instruction", "")
    step["apps"] = task_info.get("app", [])
    if not step["apps"] and step.get("package_name"):
        step["apps"] = [step.get("package_name")]
    step["category"] = task_info.get("category", "")
    step["step_index"] = step_idx + 1
    step["total_steps"] = len(steps)
    step["prev_instruction"] = (
        prev_step.get("low_level_instruction")
        or prev_step.get("type_text")
        or prev_step.get("action", "")
    ) if prev_step else ""
    step["next_instruction"] = (
        next_step.get("low_level_instruction")
        or next_step.get("type_text")
        or next_step.get("action", "")
    ) if next_step else ""
    step["prev_action"] = prev_step.get("action", "") if prev_step else ""
    step["next_action"] = next_step.get("action", "") if next_step else ""
    return step

def _generate_position_code(layout_entries: List[dict]) -> str:
    """Generate deterministic PIL code that pastes all crops at tracked positions."""
    lines = ["# --- Auto-generated: paste detected elements at original positions ---"]

    for e in layout_entries:
        x1, y1, x2, y2 = e["orig_bbox"]
        ew, eh = x2 - x1, y2 - y1
        if ew < 5 or eh < 5:
            continue
        idx = e["index"]
        action_name = e["action_name"].replace('"', "'")
        asset_comment = e.get("asset_path", "").replace("\\", "/")

        lines.append(
            f'# asset_path: {asset_comment}\n'
            f'try:\n'
            f'    _c{idx} = get_crop({idx}, {ew}, {eh})\n'
            f'    canvas.paste(_c{idx}, ({max(0, x1)}, {max(0, y1)}), _c{idx})\n'
            f'except Exception:\n'
            f'    pass\n'
            f'layout["{action_name}"] = [{x1}, {y1}, {x2}, {y2}]'
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
                 ) -> Tuple[Image.Image, dict, dict, List[dict], Image.Image]:
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
    layout_entries = _build_position_layout_entries(elements, orig_size)
    position_code = _generate_position_code(layout_entries)

    # Combine: styling first, then crop pastes on top
    full_code = styling_code + "\n\n" + position_code

    base_render_status = "render_from_code"
    base_page_img, _ = render_from_code(styling_code, elements, orig_size)
    if base_page_img is None:
        base_render_status = "fallback_bg"
        base_page_img = Image.new("RGB", CANVAS_SIZE, _extract_bg_color(elements, screenshot_path))

    # Execute on blank canvas at original resolution
    render_status = "render_from_code"
    page_img, layout = render_from_code(full_code, elements, orig_size)

    if page_img is None:
        # Fallback: just bg color + crops
        page_img, layout = _fallback_compose(elements, orig_size, screenshot_path)
        render_status = "fallback_compose"

    page_img = _postprocess_page_image(page_img)
    base_page_img = _postprocess_page_image(base_page_img)

    # Shift all layout boxes into the final GE-Lab canvas and clamp them so
    # no clickable region ends up outside the visible phone viewport.
    shifted_layout = {}
    for key, bbox in layout.items():
        if key in ("back", "home"):
            continue  # will be set below
        shifted_bbox = _clip_bbox_to_canvas([
            bbox[0] + PHONE_OFFSET_X,
            bbox[1] + PHONE_OFFSET_Y,
            bbox[2] + PHONE_OFFSET_X,
            bbox[3] + PHONE_OFFSET_Y,
        ], OUTPUT_CANVAS_SIZE)
        if _has_valid_bbox(shifted_bbox):
            shifted_layout[key] = shifted_bbox

    final_canvas, shifted_layout = _ensure_system_nav_controls(page_img, shifted_layout)
    base_canvas, _ = _ensure_system_nav_controls(base_page_img, {})

    rendered_elements = _sync_rendered_elements_with_layout(
        layout_entries,
        shifted_layout,
        canvas_size=OUTPUT_CANVAS_SIZE,
    )

    code_artifact = {
        "styling_source": styling_source,
        "render_status": render_status,
        "background_render_status": base_render_status,
        "styling_code": styling_code,
        "position_code": position_code,
        "full_code": full_code,
    }
    return final_canvas, shifted_layout, code_artifact, rendered_elements, base_canvas


def _save_page_code(code_dir: str, page_id: str, screenshot_name: str,
                    step_info: dict, code_artifact: dict):
    """Persist the PIL code used to build one AMEX-derived page."""
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
        "# The final runtime image is then wrapped into the 448x448 GE-Lab canvas with a top nav strip.",
    ]

    contents = "\n".join(header_lines) + "\n\n"
    contents += "# --- GPT styling skeleton ---\n"
    contents += code_artifact.get("styling_code", "").strip() + "\n\n"
    contents += "# --- Deterministic element pastes ---\n"
    contents += code_artifact.get("position_code", "").strip() + "\n"

    code_path = os.path.join(code_dir, f"{page_id}.py")
    with open(code_path, "w", encoding="utf-8") as f:
        f.write(contents)


def _collect_annotation_jobs(args) -> List[Tuple[str, str]]:
    """Collect one AMEX annotation or batch all annotation files in the directory."""
    annotations_dir = Path(args.annotations_dir)
    if not annotations_dir.exists():
        raise FileNotFoundError(f"Annotations directory not found: {annotations_dir}")

    trajectory_id = getattr(args, "trajectory_id", None)
    max_trajectories = getattr(args, "max_trajectories", None)

    if trajectory_id:
        candidate_paths = []
        by_filename = annotations_dir / f"{trajectory_id}.json"
        if by_filename.exists():
            candidate_paths.append(by_filename)
        else:
            for annot_path in sorted(annotations_dir.glob("*.json")):
                try:
                    with open(annot_path, "r", encoding="utf-8") as f:
                        episode_id = json.load(f).get("episode_id", "")
                    if episode_id == trajectory_id:
                        candidate_paths.append(annot_path)
                        break
                except Exception:
                    continue
        if not candidate_paths:
            raise FileNotFoundError(f"Annotation not found for trajectory_id: {trajectory_id}")
        return [(trajectory_id, str(candidate_paths[0]))]

    jobs = []
    for annot_path in sorted(annotations_dir.glob("*.json")):
        try:
            with open(annot_path, "r", encoding="utf-8") as f:
                trajectory = json.load(f)
            episode_id = trajectory.get("episode_id") or annot_path.stem
            jobs.append((str(episode_id), str(annot_path)))
        except Exception as exc:
            print(f"SKIP annotation parse failure: {annot_path.name} ({exc})")
            continue

    if max_trajectories is not None:
        jobs = jobs[:max_trajectories]
    return jobs


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

    annotation_jobs = _collect_annotation_jobs(args)
    if not annotation_jobs:
        raise RuntimeError(f"No annotations found in: {args.annotations_dir}")
    print(f"Trajectories: {len(annotation_jobs)}")
    variation_cfg = _resolve_variation_config(args)
    print(f"Variation preset: {variation_cfg['preset']}")

    # Stage 1-2: detect and extract AMEX assets first
    pages_dir = os.path.join(args.output_dir, "pages")
    code_dir = os.path.join(args.output_dir, "generated_code")
    assets_dir = os.path.join(args.output_dir, "extracted_assets")
    os.makedirs(pages_dir, exist_ok=True)
    os.makedirs(code_dir, exist_ok=True)
    os.makedirs(assets_dir, exist_ok=True)

    pages_detection_data = []
    trajectories = []
    global_page_idx = 0

    for traj_idx, (trajectory_id, annot_path) in enumerate(annotation_jobs, start=1):
        with open(annot_path, "r", encoding="utf-8") as f:
            trajectory = json.load(f)
        trajectories.append(trajectory)

        steps = trajectory.get("steps", [])
        episode_id = trajectory.get("episode_id", trajectory_id)
        task_info = trajectory.get("task_info", {})
        print(f"[{traj_idx}/{len(annotation_jobs)}] amex_episode={episode_id} task={task_info.get('task', 'N/A')} steps={len(steps)}")

        # Map original AMEX step index -> generated trace page id so each trajectory
        # only connects to its own next state before global state merging happens.
        page_lookup = {}
        for i, step in enumerate(steps):
            screenshot_name = (
                step.get("image_path")
                or step.get("screenshot")
                or f"{episode_id}-{i+1}.png"
            )
            screenshot_path = os.path.join(args.screenshots_dir, screenshot_name)

            if not os.path.exists(screenshot_path):
                print(f"  [{i+1}/{len(steps)}] SKIP (screenshot missing: {screenshot_name})")
                continue

            page_id = f"page_{global_page_idx}"
            global_page_idx += 1
            page_lookup[i] = page_id
            step_context = _build_step_context(trajectory, i)

            print(f"  [{i+1}/{len(steps)}] {screenshot_name}", end="", flush=True)
            elements, orig_size = detect_and_crop(screenshot_path, yolo_model, ocr_reader)
            print(f" ({len(elements)} detected)", end="", flush=True)

            if args.save_crops:
                _save_labeled_crops(elements, orig_size, screenshot_path,
                                    os.path.join(args.output_dir, "crops", page_id))

            asset_elements = _persist_extracted_assets(
                elements,
                screenshot_name,
                assets_dir,
                step_context,
                variation_cfg=variation_cfg,
            )
            print(f" [assets:{len(asset_elements)}]", end="", flush=True)

            pages_detection_data.append({
                "page_id": page_id,
                "trajectory_id": episode_id,
                "trajectory_index": traj_idx - 1,
                "trajectory_local_page_index": i,
                "screenshot_name": screenshot_name,
                "screenshot_path": screenshot_path,
                "orig_size": list(orig_size),
                "step": step_context,
                "elements": asset_elements,
            })

        for page in pages_detection_data:
            if page.get("trajectory_id") != episode_id:
                continue
            local_idx = page.get("trajectory_local_page_index")
            prev_page_id = page_lookup.get(local_idx - 1, "")
            next_page_id = page_lookup.get(local_idx + 1, page["page_id"])
            page["prev_trace_page_id"] = prev_page_id
            page["next_trace_page_id"] = next_page_id

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
        #   Phase 2: Extracted AMEX assets pasted at exact positions
        #   Phase 3: GE-Lab back/home buttons at top
        page_img, layout, code_artifact, rendered_elements, base_image = compose_page(
            client, model_name, elements, orig_size, screenshot_path, step_context
        )
        print(f"  compose {page_id} -> {len(layout)} layout elems")
        success_count += 1

        pages_data.append({
            "page_id": page_id,
            "image": page_img,
            "layout": layout,
            "orig_size": list(orig_size),
            "step": step_context,
            "trajectory_id": page["trajectory_id"],
            "trajectory_index": page["trajectory_index"],
            "trajectory_local_page_index": page["trajectory_local_page_index"],
            "prev_trace_page_id": page.get("prev_trace_page_id", ""),
            "next_trace_page_id": page.get("next_trace_page_id", page_id),
            "screenshot_name": screenshot_name,
            "screenshot_path": screenshot_path,
            "elements": elements,
            "rendered_elements": rendered_elements,
            "base_image": base_image,
            "code_artifact": code_artifact,
            "variation_cfg": variation_cfg,
        })

    print(f"\nComposed: {success_count}/{len(pages_data)} pages")

    canonical_pages, page_id_map = _deduplicate_pages(
        pages_data,
        state_match_threshold=args.state_match_threshold,
        layout_match_iou=args.layout_match_iou,
        client=client,
        model_name=model_name,
        enable_llm_pair_matching=not args.disable_llm_pair_matching,
        pair_match_top_k=args.llm_pair_match_top_k,
        pair_cache_path=os.path.join(args.output_dir, "llm_pair_matches.json"),
    )
    canonical_pages, canonical_page_id_map, _ = _renumber_canonical_pages(canonical_pages, page_id_map)
    for existing in os.listdir(pages_dir):
        if existing.endswith(".png"):
            try:
                os.remove(os.path.join(pages_dir, existing))
            except OSError:
                pass
    for existing in os.listdir(code_dir):
        if existing.endswith(".py"):
            try:
                os.remove(os.path.join(code_dir, existing))
            except OSError:
                pass
    for page in canonical_pages:
        page["image"].save(os.path.join(pages_dir, f"{page['page_id']}.png"))
        _save_page_code(
            code_dir,
            page["page_id"],
            page.get("screenshot_name", ""),
            page.get("step", {}),
            page.get("code_artifact", {}),
        )

    # Stage 4: Build structure
    print(f"Building structure ({len(canonical_pages)} canonical pages from {len(pages_data)} traces)...")
    structure = build_structure(
        pages_data,
        trajectories,
        args.output_dir,
        state_match_threshold=args.state_match_threshold,
        layout_match_iou=args.layout_match_iou,
        client=client,
        model_name=model_name,
        enable_llm_pair_matching=not args.disable_llm_pair_matching,
        pair_match_top_k=args.llm_pair_match_top_k,
        canonical_pages=canonical_pages,
        canonical_page_id_map=canonical_page_id_map,
    )
    _cleanup_auxiliary_outputs(args.output_dir)

    print(f"\nDone. Output: {args.output_dir}/")
    print(f"  pages/             {len(canonical_pages)} PNG files ({OUTPUT_W}x{OUTPUT_H})")
    print(f"  generated_code/    {len(canonical_pages)} PIL code files")
    print(f"  extracted_assets/  saved extracted AMEX crops")
    print(f"  ui_structure.json")
    print(f"  ui_structure_layer.json")
    print(f"  ui_topology.png")
    print(f"  action_coord_debug/")


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

    layout_entries = _build_position_layout_entries(elements, orig_size, target_size=CANVAS_SIZE)

    # Status bar
    draw.rectangle([0, 0, CANVAS_W, STATUS_BAR_HEIGHT], fill=(15, 15, 20))

    # Paste all elements at proportionally scaled positions
    nav_top = CANVAS_H - NAV_BAR_HEIGHT
    for e in layout_entries:
        sx1, sy1, sx2, sy2 = [int(v) for v in e["bbox"]]
        sw, sh = max(sx2 - sx1, 8), max(sy2 - sy1, 8)
        if sy1 >= nav_top:
            continue
        if sy2 > nav_top:
            sy2 = nav_top
            sh = sy2 - sy1
        if sw < 12 and sh < 12:
            continue

        try:
            asset_path = e.get("asset_path")
            if asset_path and os.path.exists(asset_path):
                with Image.open(asset_path) as asset_handle:
                    crop = asset_handle.convert("RGBA").resize((sw, sh), Image.LANCZOS)
            else:
                crop = e["crop"].convert("RGBA").resize((sw, sh), Image.LANCZOS)
            canvas.paste(crop, (max(0, sx1), sy1), crop)
        except Exception:
            continue

        layout[e["action_name"]] = [sx1, sy1, sx1 + sw, sy1 + sh]

    # back/home handled by compose_page nav strip (not here)
    return canvas, layout


def parse_args():
    parser = argparse.ArgumentParser(description="Sim2Real Compose: detection-guided page composition")
    parser.add_argument("--screenshots_dir", type=str,
                        default="/ext_hdd2/tsyou/AMEX_dataset/AMEX/screenshot",
                        help="Directory with AMEX screenshots")
    parser.add_argument("--annotations_dir", type=str,
                        default="/ext_hdd2/tsyou/AMEX-dataset-subset/AMEX/instruction_anno",
                        help="Directory with AMEX annotation JSONs")
    parser.add_argument("--output_dir", type=str,
                        default="data_engine/sim2real_envs/trajectory_001",
                        help="Output directory for the generated environment")
    parser.add_argument("--weights_dir", type=str,
                        default="/ext_hdd2/nhkoh/OmniParser/weights",
                        help="OmniParser weights directory")
    parser.add_argument("--model_name", type=str,
                        default="gpt-5-mini-2025-08-07",
                        help="OpenAI model for styling code generation")
    parser.add_argument("--gpu", type=int, default=0, help="GPU for YOLO detection")
    parser.add_argument("--save_crops", action="store_true",
                        help="Save labeled crops and annotated screenshots for inspection")
    parser.add_argument("--max_trajectories", type=int, default=None,
                        help="Optional cap on the number of annotation files to process when batching")
    parser.add_argument("--variation_preset", type=str, default="none",
                        choices=["none", "mild", "strong"],
                        help="Controllable visual/layout variation strength for extracted assets")
    parser.add_argument("--variation_seed", type=int, default=0,
                        help="Seed used for deterministic asset variation")
    parser.add_argument("--icon_color_jitter", type=float, default=None,
                        help="Override icon recoloring strength (0 disables)")
    parser.add_argument("--layout_jitter_px", type=int, default=None,
                        help="Override bbox jitter amount in source pixels")
    parser.add_argument("--icon_corner_radius", type=int, default=None,
                        help="Override icon corner rounding radius in source pixels")
    parser.add_argument("--state_match_threshold", type=float, default=DEFAULT_STATE_MATCH_THRESHOLD,
                        help="Minimum state similarity for page deduplication")
    parser.add_argument("--layout_match_iou", type=float, default=DEFAULT_LAYOUT_MATCH_IOU,
                        help="Minimum layout-element similarity when merging duplicate states")
    parser.add_argument("--disable_llm_page_semantics", action="store_true",
                        help="Disable per-page GPT semantic labeling for page_family/application/logical_page_name")
    parser.add_argument("--disable_llm_pair_matching", action="store_true",
                        help="Disable GPT pair verification for uncertain page merges")
    parser.add_argument("--llm_pair_match_top_k", type=int, default=DEFAULT_LLM_PAGE_MATCH_TOP_K,
                        help="How many top heuristic merge candidates to verify with GPT per page")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args)
