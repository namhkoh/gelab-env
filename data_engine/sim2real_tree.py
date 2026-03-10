"""
Sim2Real Tree: Generate GE-Lab tree environments from GUIOdyssey screenshots.

Two modes:
  1. Grounded mode (--screenshots_dir): Each tree page is based on a real
     GUIOdyssey screenshot. YOLO detects elements, GPT generates a background
     inspired by the screenshot, and extracted icon crops are pasted on top.
     Pages look like real mobile apps but are wired into a GE-Lab tree.

  2. Icon-pool mode (default): Uses pre-extracted icon pool with deterministic
     rendering. Fast fallback for testing without GUIOdyssey data.

Usage:
    # Grounded mode (real screenshots + GPT composition)
    export OPENAI_API_KEY="sk-..."
    python data_engine/sim2real_tree.py \
        --screenshots_dir /ext_hdd2/nhkoh/GUI-Odyssey/screenshots \
        --output_dir data_engine/sim2real_envs/tree_001 \
        --tree_depth 4 --nodes_per_level 3 2 1

    # Icon-pool mode (no API key needed)
    python data_engine/sim2real_tree.py \
        --icon_dir data_engine/real_icons \
        --output_dir data_engine/sim2real_envs/tree_001
"""

import argparse
import base64
import json
import os
import random
import re
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
OUTPUT_W, OUTPUT_H = 252, 448
NAV_BTN_W = 40
NAV_BTN_H = 24
NAV_STRIP_H = NAV_BTN_H + 8  # 32px
CONTENT_W = OUTPUT_W
CONTENT_H = OUTPUT_H - NAV_STRIP_H  # 416px

ICON_SIZE = 40
TITLE_HEIGHT = 36

GELAB_BACK_COLOR = (255, 200, 200)
GELAB_HOME_COLOR = (200, 255, 200)
GELAB_BACK_BBOX = [4, 4, 4 + NAV_BTN_W, 4 + NAV_BTN_H]
GELAB_HOME_BBOX = [OUTPUT_W - 4 - NAV_BTN_W, 4, OUTPUT_W - 4, 4 + NAV_BTN_H]

TEXT_BLACK = (30, 30, 30)
BG_WHITE = (255, 255, 255)

# ---------------------------------------------------------------------------
# Font loading
# ---------------------------------------------------------------------------
_font_cache = {}


def _load_font(size: int):
    if size in _font_cache:
        return _font_cache[size]
    for path in ["font/helvetica.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                 "/usr/share/fonts/liberation/LiberationSans-Regular.ttf"]:
        try:
            f = ImageFont.truetype(path, size)
            _font_cache[size] = f
            return f
        except Exception:
            continue
    f = ImageFont.load_default()
    _font_cache[size] = f
    return f


# ---------------------------------------------------------------------------
# Screenshot detection (YOLO + OCR)
# ---------------------------------------------------------------------------
def detect_elements(screenshot_path: str, yolo_model, ocr_reader,
                    conf_threshold: float = 0.15) -> List[dict]:
    """Detect UI elements on a screenshot. Returns list of element dicts.

    Each element has: label, bbox [x1,y1,x2,y2] pixels, crop (PIL Image), type.
    Reuses sim2real_compose.py detection logic.
    """
    import numpy as np
    img = Image.open(screenshot_path).convert("RGB")
    w, h = img.size
    img_np = np.array(img)

    elements = []

    # YOLO icon detection
    results = yolo_model(img_np, conf=conf_threshold, iou=0.3, verbose=False)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    confs = results[0].boxes.conf.cpu().numpy()

    # OCR text detection
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

    # Process YOLO detections, label with nearest OCR text
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
            "label": _sanitize_label(label, f"icon_{i}"),
            "bbox": [x1, y1, x2, y2],
            "crop": crop,
            "type": "icon",
            "conf": float(conf),
        })

    return elements


def _sanitize_label(text: str, fallback: str) -> str:
    cleaned = re.sub(r'[^0-9A-Za-z_]', '_', (text or '').strip()).strip('_')
    return cleaned[:30] if cleaned else fallback


# ---------------------------------------------------------------------------
# Detection cache: avoid re-running YOLO on every invocation
# ---------------------------------------------------------------------------
def build_detection_cache(screenshots_dir: str, cache_dir: str,
                          gpu: int = 0) -> List[dict]:
    """Run YOLO+OCR on screenshots, cache results. Returns pool of detections.

    Each entry: {screenshot_path, orig_size, elements: [{label, bbox, crop_path}]}
    """
    import glob

    cache_meta_path = os.path.join(cache_dir, "detection_cache.json")
    crops_dir = os.path.join(cache_dir, "crops")

    # Load existing cache
    existing = {}
    if os.path.exists(cache_meta_path):
        with open(cache_meta_path) as f:
            for entry in json.load(f):
                existing[entry["screenshot"]] = entry

    # Find screenshots
    patterns = [os.path.join(screenshots_dir, "*.png"),
                os.path.join(screenshots_dir, "*.jpg")]
    paths = []
    for pat in patterns:
        paths.extend(sorted(glob.glob(pat)))
    print(f"Found {len(paths)} screenshots in {screenshots_dir}")

    # Filter to uncached
    todo = [p for p in paths if os.path.basename(p) not in existing]
    if not todo:
        print(f"All {len(paths)} screenshots cached.")
        return list(existing.values())

    print(f"Running detection on {len(todo)} new screenshots...")

    # Load models
    from ultralytics import YOLO
    import easyocr
    weights_dir = "/ext_hdd2/nhkoh/OmniParser/weights"
    yolo_model = YOLO(os.path.join(weights_dir, "icon_detect", "model.pt"))
    ocr_reader = easyocr.Reader(["en"], gpu=(gpu >= 0))

    os.makedirs(crops_dir, exist_ok=True)

    for i, spath in enumerate(todo):
        fname = os.path.basename(spath)
        print(f"  [{i+1}/{len(todo)}] {fname}", end="", flush=True)

        elements = detect_elements(spath, yolo_model, ocr_reader)
        img = Image.open(spath)
        orig_size = img.size
        img.close()

        # Save crops
        elem_records = []
        for elem in elements:
            crop_name = f"{Path(fname).stem}_{elem['index']:02d}_{elem['label'][:20]}.png"
            crop_path = os.path.join(crops_dir, crop_name)
            elem["crop"].save(crop_path)
            elem_records.append({
                "label": elem["label"],
                "bbox": elem["bbox"],
                "type": elem["type"],
                "crop_path": crop_path,
            })

        existing[fname] = {
            "screenshot": fname,
            "screenshot_path": spath,
            "orig_size": list(orig_size),
            "elements": elem_records,
            "n_icons": len([e for e in elem_records if e["type"] == "icon"]),
        }
        print(f" -> {len(elem_records)} elements")

    # Save cache
    with open(cache_meta_path, "w") as f:
        json.dump(list(existing.values()), f, indent=2)

    return list(existing.values())


# ---------------------------------------------------------------------------
# Icon pool loading (fallback mode)
# ---------------------------------------------------------------------------
def load_icon_pool(icon_dir: str) -> List[Tuple[Image.Image, str]]:
    """Load icon images from directory with GPT-generated labels if available."""
    import glob

    labels_path = os.path.join(icon_dir, "icon_labels.json")
    gpt_labels = {}
    if os.path.exists(labels_path):
        with open(labels_path) as f:
            gpt_labels = json.load(f)
        named = sum(1 for v in gpt_labels.values() if v != "unknown")
        print(f"Loaded {len(gpt_labels)} icon labels ({named} named)")

    pattern = os.path.join(icon_dir, "*/PNG/*.png")
    paths = sorted(glob.glob(pattern))
    if not paths:
        pattern = os.path.join(icon_dir, "**/*.png")
        paths = sorted(glob.glob(pattern, recursive=True))
    if not paths:
        raise ValueError(f"No PNG icons found in {icon_dir}")

    icons = []
    label_counts = {}
    for p in paths:
        try:
            img = Image.open(p).convert("RGBA")
            fname = os.path.basename(p)
            if fname in gpt_labels and gpt_labels[fname] != "unknown":
                base_label = gpt_labels[fname].replace(" ", "_")
            else:
                parts = Path(p).parts
                if "PNG" in parts:
                    idx = list(parts).index("PNG")
                    category = parts[idx - 1].replace(" ", "_") if idx > 0 else "icon"
                    name = Path(p).stem
                    base_label = f"{category}_{name}"
                else:
                    base_label = Path(p).stem
            count = label_counts.get(base_label, 0)
            label_counts[base_label] = count + 1
            label = f"{base_label}_{count + 1}" if count > 0 else base_label
            icons.append((img, label))
        except Exception:
            continue

    print(f"Loaded {len(icons)} icons from {icon_dir}")
    return icons


# ---------------------------------------------------------------------------
# Tree topology
# ---------------------------------------------------------------------------
class TreeNode:
    def __init__(self, page_id: str, depth: int, parent_id: Optional[str] = None):
        self.page_id = page_id
        self.depth = depth
        self.parent_id = parent_id
        self.children: List['TreeNode'] = []
        self.icons: List[Tuple[Image.Image, str]] = []  # nav icons only
        self.children_map: Dict[str, str] = {}
        self.layout: Dict[str, List[int]] = {}
        # Grounded mode: source screenshot info
        self.screenshot_path: Optional[str] = None
        self.screenshot_elements: List[dict] = []
        self.orig_size: Optional[Tuple[int, int]] = None
        # All display elements: (crop, label, orig_bbox) — nav + decoration
        self.all_display_elements: List[Tuple[Image.Image, str, List[int]]] = []


def build_tree(icons: List[Tuple[Image.Image, str]],
               tree_depth: int,
               nodes_per_level: List[int],
               seed: int = 42) -> Tuple[TreeNode, List[TreeNode]]:
    """Build tree topology and assign icons from pool."""
    rng = random.Random(seed)
    icon_pool = list(icons)
    rng.shuffle(icon_pool)
    icon_idx = 0

    root = TreeNode("page_0", depth=0)
    all_nodes = [root]
    current_level = [root]
    page_counter = 1

    for level in range(tree_depth - 1):
        branching = nodes_per_level[level]
        next_level = []
        for parent in current_level:
            page_icons = []
            for _ in range(branching):
                child_id = f"page_{page_counter}"
                child = TreeNode(child_id, depth=level + 1,
                                 parent_id=parent.page_id)
                parent.children.append(child)
                all_nodes.append(child)
                next_level.append(child)
                page_counter += 1

                img, label = icon_pool[icon_idx % len(icon_pool)]
                icon_idx += 1
                used = {l for _, l in page_icons}
                if label in used:
                    label = f"{label}_{icon_idx}"
                page_icons.append((img, label))
                parent.children_map[label] = child_id
            parent.icons = page_icons
        current_level = next_level

    print(f"Tree: {len(all_nodes)} pages, depth {tree_depth}, branching {nodes_per_level}")
    return root, all_nodes


def build_tree_grounded(detection_pool: List[dict],
                        tree_depth: int,
                        nodes_per_level: List[int],
                        seed: int = 42) -> Tuple[TreeNode, List[TreeNode]]:
    """Build tree topology, assigning a GUIOdyssey screenshot to each node.

    Non-leaf pages get screenshots with enough detected icons for the branching
    factor. Each detected icon becomes a navigation element in the tree.
    """
    rng = random.Random(seed)

    # Sort pool by number of icons (descending) for greedy assignment
    pool = list(detection_pool)
    rng.shuffle(pool)

    root = TreeNode("page_0", depth=0)
    all_nodes = [root]
    current_level = [root]
    page_counter = 1
    pool_idx = 0

    def pick_screenshot(min_icons: int) -> Optional[dict]:
        """Pick a screenshot with at least min_icons detected elements."""
        nonlocal pool_idx
        # Try from current position forward, wrapping around
        for offset in range(len(pool)):
            entry = pool[(pool_idx + offset) % len(pool)]
            if entry["n_icons"] >= min_icons:
                pool_idx = (pool_idx + offset + 1) % len(pool)
                return entry
        # Fallback: pick any screenshot
        entry = pool[pool_idx % len(pool)]
        pool_idx = (pool_idx + 1) % len(pool)
        return entry

    # Assign screenshot to root
    branching_root = nodes_per_level[0]
    root_entry = pick_screenshot(branching_root)
    _assign_screenshot_to_node(root, root_entry, branching_root)

    for level in range(tree_depth - 1):
        branching = nodes_per_level[level]
        next_level = []

        for parent in current_level:
            for i in range(branching):
                child_id = f"page_{page_counter}"
                child = TreeNode(child_id, depth=level + 1,
                                 parent_id=parent.page_id)
                parent.children.append(child)
                all_nodes.append(child)
                next_level.append(child)
                page_counter += 1

            # Icons on parent page are the navigation elements to children
            # These are already set in _assign_screenshot_to_node

        # Assign screenshots to next level nodes
        for child in next_level:
            is_leaf = (level + 1 == tree_depth - 1)
            if is_leaf:
                entry = pick_screenshot(0)
                _assign_screenshot_to_node(child, entry, 0)
            else:
                next_branching = nodes_per_level[level + 1]
                entry = pick_screenshot(next_branching)
                _assign_screenshot_to_node(child, entry, next_branching)

        current_level = next_level

    print(f"Tree: {len(all_nodes)} pages, depth {tree_depth}, branching {nodes_per_level}")
    return root, all_nodes


def _assign_screenshot_to_node(node: TreeNode, entry: dict, n_nav_icons: int):
    """Assign a screenshot's detected elements to a tree node.

    Loads ALL detected elements as display items.
    Picks n_nav_icons of them as navigation icons (wired to tree transitions).
    The rest are visual decoration for realistic page density.
    """
    node.screenshot_path = entry["screenshot_path"]
    node.orig_size = tuple(entry["orig_size"])
    node.screenshot_elements = entry["elements"]

    all_elems = entry["elements"]
    if not all_elems:
        return

    # Load ALL crops with unique labels
    label_counts = {}
    all_display = []
    for elem in all_elems:
        try:
            crop = Image.open(elem["crop_path"]).convert("RGBA")
        except Exception:
            continue
        base_label = _sanitize_label(elem["label"], f"elem_{len(all_display)}")
        count = label_counts.get(base_label, 0)
        label_counts[base_label] = count + 1
        label = f"{base_label}_{count + 1}" if count > 0 else base_label
        all_display.append((crop, label, elem["bbox"]))

    node.all_display_elements = all_display

    # Pick nav icons from among the detected elements
    # Prefer medium-sized elements (not too big, not too tiny) for nav icons
    if n_nav_icons > 0 and all_display:
        def _icon_score(item):
            _, _, bbox = item
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            area = w * h
            # Prefer icons with area between 1000 and 50000 pixels
            if 1000 < area < 50000:
                return 0
            return abs(area - 25000)

        candidates = sorted(all_display, key=_icon_score)
        nav_items = candidates[:n_nav_icons]
        node.icons = [(img, label) for img, label, _ in nav_items]
    else:
        node.icons = []


# ---------------------------------------------------------------------------
# Layout computation
# ---------------------------------------------------------------------------
def compute_page_layout(icons: List[Tuple[Image.Image, str]],
                        canvas_w: int = CONTENT_W,
                        canvas_h: int = CONTENT_H,
                        icon_size: int = ICON_SIZE) -> Dict[str, List[int]]:
    """Compute grid positions for icons on the content canvas."""
    layout = {}
    layout["page_title"] = [10, 5, canvas_w - 10, TITLE_HEIGHT]

    if not icons:
        return layout

    n = len(icons)
    margin_x = 16
    margin_y = 20
    label_h = 16
    cell_h = icon_size + label_h

    max_cols = max(1, (canvas_w - margin_x) // (icon_size + margin_x))
    cols = min(n, max_cols)
    rows = (n + cols - 1) // cols

    grid_w = cols * icon_size + (cols - 1) * margin_x
    grid_h = rows * cell_h + (rows - 1) * margin_y
    start_x = (canvas_w - grid_w) // 2
    start_y = TITLE_HEIGHT + max(10, (canvas_h - TITLE_HEIGHT - grid_h) // 2)

    for i, (_, label) in enumerate(icons):
        r, c = divmod(i, cols)
        x = start_x + c * (icon_size + margin_x)
        y = start_y + r * (cell_h + margin_y)
        layout[label] = [x, y, x + icon_size, y + icon_size]

    return layout


def compute_page_layout_grounded(node: 'TreeNode',
                                  canvas_w: int = CONTENT_W,
                                  canvas_h: int = CONTENT_H) -> Dict[str, List[int]]:
    """Position ALL elements proportionally from their original screenshot coords.

    Scales from original GUIOdyssey resolution (e.g. 720x1280) to our canvas.
    Nav icons get their real position (not forced into a grid).
    """
    layout = {}
    if not node.all_display_elements or not node.orig_size:
        return layout

    ow, oh = node.orig_size
    sx = canvas_w / ow
    sy = canvas_h / oh

    for _, label, orig_bbox in node.all_display_elements:
        ox1, oy1, ox2, oy2 = orig_bbox
        x1 = int(ox1 * sx)
        y1 = int(oy1 * sy)
        x2 = int(ox2 * sx)
        y2 = int(oy2 * sy)
        # Clamp to canvas
        x1 = max(0, min(x1, canvas_w - 2))
        y1 = max(0, min(y1, canvas_h - 2))
        x2 = max(x1 + 1, min(x2, canvas_w))
        y2 = max(y1 + 1, min(y2, canvas_h))
        layout[label] = [x1, y1, x2, y2]

    return layout


# ---------------------------------------------------------------------------
# Icon fitting helper
# ---------------------------------------------------------------------------
def _fit_icon(img: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
    tw, th = target_size
    source = img.convert("RGBA")
    scale = min(tw / max(source.width, 1), th / max(source.height, 1))
    rw = max(1, int(round(source.width * scale)))
    rh = max(1, int(round(source.height * scale)))
    resized = source.resize((rw, rh), Image.LANCZOS)
    canvas = Image.new("RGBA", (tw, th), (0, 0, 0, 0))
    canvas.paste(resized, ((tw - rw) // 2, (th - rh) // 2), resized)
    return canvas


# ---------------------------------------------------------------------------
# Rendering: deterministic (fallback)
# ---------------------------------------------------------------------------
def render_page_deterministic(node: TreeNode,
                              canvas_size: Tuple[int, int] = (CONTENT_W, CONTENT_H),
                              ) -> Image.Image:
    """White background, centered title, icons in grid."""
    cw, ch = canvas_size
    layout = node.layout

    content_img = Image.new("RGB", canvas_size, BG_WHITE)
    draw = ImageDraw.Draw(content_img)

    title_font = _load_font(14)
    title = node.page_id
    text_bbox = draw.textbbox((0, 0), title, font=title_font)
    tw = text_bbox[2] - text_bbox[0]
    th = text_bbox[3] - text_bbox[1]
    draw.text(((cw - tw) // 2, (TITLE_HEIGHT - th) // 2),
              title, fill=TEXT_BLACK, font=title_font)

    for icon_img, icon_label in node.icons:
        bbox = layout.get(icon_label)
        if bbox is None:
            continue
        x1, y1, x2, y2 = bbox
        fitted = _fit_icon(icon_img, (x2 - x1, y2 - y1))
        content_img.paste(fitted, (x1, y1), fitted)

    return content_img


# ---------------------------------------------------------------------------
# Rendering: grounded (screenshot reference + GPT background + crop paste)
# ---------------------------------------------------------------------------
def _image_to_base64(img: Image.Image, max_dim: int = 512) -> str:
    """Resize and encode PIL image to base64 data URL."""
    w, h = img.size
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=80)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def _build_grounded_prompt(content_w, content_h, elements_desc, n_total_elements):
    """GPT prompt that uses the screenshot as visual reference."""
    lines = [
        "You MUST respond with ONLY a ```python code block. No explanations.",
        "",
        f"Write Python PIL code to draw the BACKGROUND of this mobile UI page.",
        f"Canvas size: {content_w}x{content_h} pixels (starts white).",
        "",
        "Look at the attached screenshot for visual reference.",
        f"I will paste {n_total_elements} detected UI elements on top of this background.",
        "Draw ONLY the background/chrome that matches the screenshot:",
        "- Fill canvas with the screenshot's dominant background color (NOT white unless screenshot is white)",
        "- Draw header/toolbar/status-bar colored bands",
        "- Draw section card backgrounds as rounded rectangles",
        "- Draw separator lines, bottom tab-bar background",
        "- Match the color scheme closely (e.g., dark mode = dark bg)",
        "",
        "DO NOT draw icons, text labels, or any interactive elements.",
        "The background should look like the screenshot WITH elements removed.",
        "",
        f"Variables: canvas (PIL Image {content_w}x{content_h}), draw (ImageDraw),",
        "font_sm (12pt), font_md (18pt), font_lg (24pt).",
        "Do not import anything.",
    ]
    return "\n".join(lines)


def _extract_code(text: str) -> Optional[str]:
    for pattern in [r"```python\s*\n(.*?)```", r"```\s*\n(.*?)```"]:
        m = re.search(pattern, text, re.DOTALL)
        if m:
            code = m.group(1).strip()
            if code and len(code) > 20:
                return code
    text = text.strip()
    if text and len(text) > 20 and ("draw." in text or "canvas" in text):
        return text
    return None


def _sanitize_code(code: str) -> str:
    lines = []
    for line in code.split("\n"):
        s = line.strip()
        if s.startswith("import ") or s.startswith("from "):
            lines.append(f"# REMOVED: {line}")
        elif "open(" in s or "os." in s or "subprocess" in s or "eval(" in s:
            lines.append(f"# REMOVED: {line}")
        else:
            lines.append(line)
    # Fix deprecated textsize
    result = "\n".join(lines)
    result = re.sub(r'draw\.textsize\(([^)]+)\)', r'(draw.textlength(\1), 20)', result)
    return result


def render_page_grounded(client, model_name: str, node: TreeNode,
                         canvas_size: Tuple[int, int] = (CONTENT_W, CONTENT_H),
                         max_retries: int = 2) -> Image.Image:
    """Render page using screenshot as visual reference + extracted crops.

    1. Send screenshot to GPT -> generates background PIL code
    2. Execute code on blank canvas
    3. Paste extracted icon crops at grid positions
    """
    cw, ch = canvas_size
    layout = node.layout

    n_total = len(node.all_display_elements)
    prompt = _build_grounded_prompt(cw, ch, "", n_total)

    # Load and encode screenshot
    screenshot = Image.open(node.screenshot_path).convert("RGB")
    img_url = _image_to_base64(screenshot)

    # Call GPT with screenshot as vision input
    content_img = None
    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": img_url}},
                    ],
                }],
                max_completion_tokens=2048,
            )
            text = response.choices[0].message.content or ""
            code = _extract_code(text)
            if code:
                code = _sanitize_code(code)
                canvas = Image.new("RGB", canvas_size, BG_WHITE)
                draw = ImageDraw.Draw(canvas)
                namespace = {
                    "__builtins__": {},
                    "canvas": canvas, "draw": draw,
                    "font_sm": _load_font(12), "font_md": _load_font(18),
                    "font_lg": _load_font(24),
                    "Image": Image, "ImageDraw": ImageDraw,
                    "range": range, "len": len, "enumerate": enumerate,
                    "min": min, "max": max, "int": int, "float": float, "str": str,
                    "True": True, "False": False, "None": None,
                    "list": list, "tuple": tuple, "dict": dict,
                    "abs": abs, "round": round, "print": print,
                    "random": random,
                }
                exec(code, namespace)
                content_img = canvas
                break
            if attempt < max_retries:
                print(" [retry]", end="", flush=True)
        except Exception as e:
            if attempt < max_retries:
                print(" [retry]", end="", flush=True)
            else:
                print(f" [error: {e}]", end="", flush=True)

    if content_img is None:
        print(" [fallback]", end="", flush=True)
        content_img = Image.new("RGB", canvas_size, BG_WHITE)
        draw = ImageDraw.Draw(content_img)
        draw.rectangle([0, 0, cw, TITLE_HEIGHT], fill=(240, 240, 245))
        draw.line([0, TITLE_HEIGHT, cw, TITLE_HEIGHT], fill=(210, 210, 210))

    # Paste ALL detected element crops at their proportional positions
    for icon_img, icon_label, _ in node.all_display_elements:
        bbox = layout.get(icon_label)
        if bbox is None:
            continue
        x1, y1, x2, y2 = bbox
        bw, bh = x2 - x1, y2 - y1
        if bw < 2 or bh < 2:
            continue
        fitted = _fit_icon(icon_img, (bw, bh))
        content_img.paste(fitted, (x1, y1), fitted)

    return content_img


# ---------------------------------------------------------------------------
# Nav strip assembly
# ---------------------------------------------------------------------------
def assemble_with_nav_strip(content_img: Image.Image,
                            layout: Dict[str, List[int]],
                            ) -> Tuple[Image.Image, Dict[str, List[int]]]:
    final = Image.new("RGB", (OUTPUT_W, OUTPUT_H), BG_WHITE)
    final.paste(content_img, (0, NAV_STRIP_H))

    final_layout = {}
    for key, bbox in layout.items():
        if key == "page_title":
            continue
        final_layout[key] = [bbox[0], bbox[1] + NAV_STRIP_H,
                             bbox[2], bbox[3] + NAV_STRIP_H]

    draw_nav = ImageDraw.Draw(final)
    draw_nav.rectangle([0, 0, OUTPUT_W, NAV_STRIP_H], fill=BG_WHITE)
    draw_nav.line([0, NAV_STRIP_H - 1, OUTPUT_W, NAV_STRIP_H - 1],
                  fill=(200, 200, 200))
    font_nav = _load_font(12)

    bx1, by1, bx2, by2 = GELAB_BACK_BBOX
    draw_nav.rounded_rectangle([bx1, by1, bx2, by2], radius=4, fill=GELAB_BACK_COLOR)
    try:
        btw = draw_nav.textlength("back", font=font_nav)
    except AttributeError:
        btw = 24
    draw_nav.text((bx1 + (NAV_BTN_W - btw) / 2, by1 + (NAV_BTN_H - 12) / 2),
                  "back", fill=TEXT_BLACK, font=font_nav)
    final_layout["back"] = list(GELAB_BACK_BBOX)

    hx1, hy1, hx2, hy2 = GELAB_HOME_BBOX
    draw_nav.rounded_rectangle([hx1, hy1, hx2, hy2], radius=4, fill=GELAB_HOME_COLOR)
    try:
        htw = draw_nav.textlength("home", font=font_nav)
    except AttributeError:
        htw = 28
    draw_nav.text((hx1 + (NAV_BTN_W - htw) / 2, hy1 + (NAV_BTN_H - 12) / 2),
                  "home", fill=TEXT_BLACK, font=font_nav)
    final_layout["home"] = list(GELAB_HOME_BBOX)

    return final, final_layout


# ---------------------------------------------------------------------------
# GE-Lab JSON structure builder
# ---------------------------------------------------------------------------
def build_gelab_structure(root, all_nodes, layouts, tree_depth,
                          nodes_per_level, seed) -> Tuple[dict, dict]:
    metadata = {
        "canvas_size": [OUTPUT_W, OUTPUT_H],
        "nav_strip_height": NAV_STRIP_H,
        "icon_size": ICON_SIZE,
        "tree_depth": tree_depth,
        "nodes_per_level": nodes_per_level,
        "total_pages": len(all_nodes),
        "seed": seed,
        "source": "sim2real_tree",
    }

    ui_structure = {"pages": {}, "metadata": metadata}
    for node in all_nodes:
        layout = layouts.get(node.page_id, {})
        layout_typed = {}
        for key, bbox in layout.items():
            ltype = "system" if key in ("back", "home") else "normal"
            layout_typed[key] = {"bbox": bbox, "type": ltype}

        transitions = []
        for icon_label, child_id in node.children_map.items():
            icon_bbox = layout.get(icon_label, GELAB_BACK_BBOX)
            transitions.append({
                "action": icon_label, "target_page": child_id,
                "icon_bbox": icon_bbox,
            })
        if node.parent_id is not None:
            transitions.append({
                "action": "back", "target_page": node.parent_id,
                "icon_bbox": list(GELAB_BACK_BBOX),
            })
        transitions.append({
            "action": "home", "target_page": "page_0",
            "icon_bbox": list(GELAB_HOME_BBOX),
        })
        ui_structure["pages"][node.page_id] = {
            "image": f"{node.page_id}.png", "depth": node.depth,
            "layout": layout_typed, "transitions": transitions,
        }

    def build_layer(node):
        layout = layouts.get(node.page_id, {})
        non_system_trans = []
        for icon_label, child_id in node.children_map.items():
            icon_bbox = layout.get(icon_label)
            if icon_bbox:
                non_system_trans.append({
                    "action": icon_label, "target_page": child_id,
                    "icon_bbox": icon_bbox,
                })
        layer_layout = {}
        for key, bbox in layout.items():
            if key in ("back", "home"):
                continue
            layer_layout[key] = {"bbox": bbox, "type": "normal"}
        return {
            "image": f"{node.page_id}.png", "depth": node.depth,
            "layout": layer_layout, "transitions": non_system_trans,
            "subnodes": [build_layer(c) for c in node.children],
        }

    ui_layer = {"root": build_layer(root), "metadata": metadata}
    return ui_structure, ui_layer


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def run_pipeline(args):
    grounded = args.screenshots_dir is not None

    if grounded:
        # --- Grounded mode: real screenshots ---
        print("=== Grounded mode: GUIOdyssey screenshots ===")
        cache_dir = os.path.join(args.output_dir, ".detection_cache")
        detection_pool = build_detection_cache(
            args.screenshots_dir, cache_dir, gpu=args.gpu,
        )
        print(f"Detection pool: {len(detection_pool)} screenshots")

        # OpenAI client for GPT composition
        from openai import OpenAI
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError("OPENAI_API_KEY required for grounded mode")
        client = OpenAI(api_key=key)
        print(f"Model: {args.model_name}")

        # Build tree with screenshot assignment
        root, all_nodes = build_tree_grounded(
            detection_pool, args.tree_depth, args.nodes_per_level, seed=args.seed,
        )

        # Compute layout from proportional screenshot positions
        for node in all_nodes:
            node.layout = compute_page_layout_grounded(node)
            # Wire children_map (nav icon label -> child page_id)
            if node.children and node.icons:
                for child, (_, label) in zip(node.children, node.icons):
                    node.children_map[label] = child.page_id

    else:
        # --- Icon-pool mode: pre-extracted icons ---
        print("=== Icon-pool mode ===")
        icons = load_icon_pool(args.icon_dir)
        client = None
        if args.use_gpt:
            from openai import OpenAI
            key = os.environ.get("OPENAI_API_KEY")
            if not key:
                raise ValueError("OPENAI_API_KEY required when --use_gpt is set")
            client = OpenAI(api_key=key)
            print(f"GPT mode: {args.model_name}")
        root, all_nodes = build_tree(
            icons, args.tree_depth, args.nodes_per_level, seed=args.seed,
        )
        for node in all_nodes:
            node.layout = compute_page_layout(node.icons)

    # Render all pages
    pages_dir = os.path.join(args.output_dir, "pages")
    os.makedirs(pages_dir, exist_ok=True)
    layouts = {}

    for i, node in enumerate(all_nodes):
        n_all = len(node.all_display_elements)
        n_nav = len(node.icons)
        print(f"  [{i+1}/{len(all_nodes)}] {node.page_id} "
              f"(depth={node.depth}, {n_all} elems, {n_nav} nav)",
              end="", flush=True)

        if grounded:
            content_img = render_page_grounded(
                client, args.model_name, node,
            )
        elif client is not None:
            content_img = render_page_grounded(
                client, args.model_name, node,
            ) if node.screenshot_path else render_page_deterministic(node)
        else:
            content_img = render_page_deterministic(node)

        page_img, layout = assemble_with_nav_strip(content_img, node.layout)
        page_path = os.path.join(pages_dir, f"{node.page_id}.png")
        page_img.save(page_path)
        layouts[node.page_id] = layout
        print(f" -> {len(layout)} layout elems")

    print(f"\nRendered: {len(all_nodes)} pages")

    # Build and save GE-Lab JSONs
    print(f"Building structure ({len(all_nodes)} pages)...")
    ui_structure, ui_layer = build_gelab_structure(
        root, all_nodes, layouts,
        args.tree_depth, args.nodes_per_level, args.seed,
    )
    with open(os.path.join(args.output_dir, "ui_structure.json"), "w") as f:
        json.dump(ui_structure, f, indent=2)
    with open(os.path.join(args.output_dir, "ui_structure_layer.json"), "w") as f:
        json.dump(ui_layer, f, indent=2)

    print(f"\nDone. Output: {args.output_dir}/")
    print(f"  pages/                 {len(all_nodes)} PNG files ({OUTPUT_W}x{OUTPUT_H})")
    print(f"  ui_structure.json")
    print(f"  ui_structure_layer.json")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sim2Real Tree: GE-Lab environment from GUIOdyssey screenshots")
    # Grounded mode
    parser.add_argument("--screenshots_dir", type=str, default=None,
                        help="GUIOdyssey screenshots directory (enables grounded mode)")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU for YOLO detection (grounded mode)")
    # Icon-pool mode
    parser.add_argument("--icon_dir", type=str, default="data_engine/real_icons",
                        help="Icon directory (icon-pool mode)")
    parser.add_argument("--use_gpt", action="store_true",
                        help="Use GPT for backgrounds (icon-pool mode)")
    # Shared
    parser.add_argument("--output_dir", type=str,
                        default="data_engine/sim2real_envs/tree_001")
    parser.add_argument("--tree_depth", type=int, default=4)
    parser.add_argument("--nodes_per_level", type=int, nargs="+", default=[3, 2, 1])
    parser.add_argument("--model_name", type=str, default="gpt-5-mini-2025-08-07")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if len(args.nodes_per_level) != args.tree_depth - 1:
        raise ValueError(
            f"nodes_per_level length ({len(args.nodes_per_level)}) "
            f"must equal tree_depth - 1 ({args.tree_depth - 1})")
    run_pipeline(args)
