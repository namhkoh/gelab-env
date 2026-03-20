"""
Page composition from AMEX trajectories using pre-extracted icons.

Given an AMEX trajectory, this script:
1. Loads element annotations (bbox + labels) for each step
2. Matches extracted icon PNGs from the ui_elements_output_clean directory
3. Places icons on a GE-Lab canvas (1080x2400) at their scaled positions
4. Produces side-by-side comparisons with original screenshots
5. Generates an icon quality report

Supports four crop modes:
  screenshot          - Direct GT bbox crops (deterministic, no GPT)
  icons_on_screenshot - Extracted icons on scaled GT background (deterministic)
  icons_only          - Extracted icons on flat color (deterministic)
  gpt_compose         - GPT-5-Mini generates background styling, icons pasted on top
                        (requires OPENAI_API_KEY)

Usage:
    python data_engine/amex_compose_deterministic.py \
        --trajectory_id 2024_3_20_16_19_c7de79edba6b442681ddcdc40adb3900

    python data_engine/amex_compose_deterministic.py \
        --trajectory_id 2024_3_20_16_19_c7de79edba6b442681ddcdc40adb3900 \
        --crop_mode gpt_compose --max_steps 3
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont, ImageOps

CANVAS_W, CANVAS_H = 1080, 2400

NAV_BTN_W = 88
NAV_BTN_H = 38
NAV_STRIP_H = NAV_BTN_H + 18
BACK_BBOX = [10, 8, 10 + NAV_BTN_W, 8 + NAV_BTN_H]
HOME_BBOX = [CANVAS_W - 10 - NAV_BTN_W, 8, CANVAS_W - 10, 8 + NAV_BTN_H]
BACK_COLOR = (255, 200, 200)
HOME_COLOR = (200, 255, 200)

DEFAULT_ANNOTATIONS_DIR = "/ext_hdd2/tsyou/AMEX_dataset/AMEX/instruction_anno"
DEFAULT_SCREENSHOTS_DIR = "/ext_hdd2/tsyou/AMEX_dataset/AMEX/screenshot"
DEFAULT_ELEMENT_ANNO_DIR = "/ext_hdd2/tsyou/AMEX_dataset/AMEX/element_anno"
DEFAULT_ICONS_DIR = "/ext_hdd2/ttran/datasets/AMEX_dataset/ui_elements_output_clean/elements"


def _try_font(size: int, bold: bool = False):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]
    for c in candidates:
        try:
            return ImageFont.truetype(c, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


FONT_SM = _try_font(22)
FONT_MD = _try_font(28, bold=True)


def _scale_bbox(bbox: List[int], device_w: int, device_h: int) -> List[int]:
    """Scale bbox from device coordinates to canvas coordinates."""
    x1, y1, x2, y2 = bbox
    sx = CANVAS_W / device_w
    sy = (CANVAS_H - NAV_STRIP_H) / device_h
    return [
        int(round(x1 * sx)),
        int(round(y1 * sy)) + NAV_STRIP_H,
        int(round(x2 * sx)),
        int(round(y2 * sy)) + NAV_STRIP_H,
    ]


def _scale_point(coord: List[int], device_w: int, device_h: int) -> List[int]:
    """Scale a point from device coordinates to canvas coordinates."""
    x, y = coord[0], coord[1]
    sx = CANVAS_W / device_w
    sy = (CANVAS_H - NAV_STRIP_H) / device_h
    return [
        int(round(x * sx)),
        int(round(y * sy)) + NAV_STRIP_H,
    ]


def _extract_bg_color(screenshot_path: str) -> Tuple[int, int, int]:
    """Sample the dominant background color from a screenshot."""
    try:
        img = Image.open(screenshot_path).convert("RGB")
        small = img.resize((1, 1), Image.Resampling.LANCZOS)
        return small.getpixel((0, 0))
    except Exception:
        return (245, 245, 248)


def _sample_screen_colors(screenshot_path: str) -> Dict[str, Tuple[int, int, int]]:
    """Sample dominant colors from different regions of the screenshot."""
    defaults = {
        "top": (30, 30, 40),
        "mid": (245, 245, 248),
        "bottom": (240, 240, 243),
        "overall": (245, 245, 248),
    }
    try:
        img = Image.open(screenshot_path).convert("RGB")
        w, h = img.size
        top = img.crop((0, 0, w, min(80, h))).resize((1, 1), Image.Resampling.LANCZOS).getpixel((0, 0))
        mid = img.crop((0, h // 3, w, 2 * h // 3)).resize((1, 1), Image.Resampling.LANCZOS).getpixel((0, 0))
        bottom = img.crop((0, max(0, h - 120), w, h)).resize((1, 1), Image.Resampling.LANCZOS).getpixel((0, 0))
        overall = img.resize((1, 1), Image.Resampling.LANCZOS).getpixel((0, 0))
        return {"top": top, "mid": mid, "bottom": bottom, "overall": overall}
    except Exception:
        return defaults


def _draw_deterministic_background(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    colors: Dict[str, Tuple[int, int, int]],
) -> None:
    """Draw a clean deterministic background using sampled colors."""
    top_c = colors["top"]
    mid_c = colors["mid"]
    bottom_c = colors["bottom"]

    for y in range(NAV_STRIP_H, CANVAS_H):
        t = (y - NAV_STRIP_H) / max(CANVAS_H - NAV_STRIP_H - 1, 1)
        if t < 0.05:
            s = t / 0.05
            r = int(top_c[0] * (1 - s) + mid_c[0] * s)
            g = int(top_c[1] * (1 - s) + mid_c[1] * s)
            b = int(top_c[2] * (1 - s) + mid_c[2] * s)
        elif t > 0.9:
            s = (t - 0.9) / 0.1
            r = int(mid_c[0] * (1 - s) + bottom_c[0] * s)
            g = int(mid_c[1] * (1 - s) + bottom_c[1] * s)
            b = int(mid_c[2] * (1 - s) + bottom_c[2] * s)
        else:
            r, g, b = mid_c
        draw.line([(0, y), (CANVAS_W, y)], fill=(r, g, b))

    draw.rectangle([0, NAV_STRIP_H, CANVAS_W, NAV_STRIP_H + 48], fill=top_c)


def _draw_nav_strip(draw: ImageDraw.ImageDraw, layout: Dict[str, Any]) -> None:
    draw.rectangle([0, 0, CANVAS_W, NAV_STRIP_H], fill=(240, 240, 243))
    draw.rounded_rectangle(BACK_BBOX, radius=10, fill=BACK_COLOR, outline=(220, 154, 154), width=2)
    draw.rounded_rectangle(HOME_BBOX, radius=10, fill=HOME_COLOR, outline=(136, 191, 136), width=2)
    for label, bbox in [("back", BACK_BBOX), ("home", HOME_BBOX)]:
        tb = draw.textbbox((0, 0), label, font=FONT_SM)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        draw.text(
            (bbox[0] + (bbox[2] - bbox[0] - tw) // 2, bbox[1] + (bbox[3] - bbox[1] - th) // 2),
            label, fill=(32, 36, 44), font=FONT_SM,
        )
    layout["back"] = BACK_BBOX
    layout["home"] = HOME_BBOX


def _find_icon_file(icons_dir: str, traj_stem: str, step_id: int, elem_idx: int) -> Optional[str]:
    """Find the extracted icon file for a given trajectory/step/element index."""
    prefix = f"{traj_stem}-{step_id}_"
    suffix = f"_elem{elem_idx}.png"
    bare_suffix = f"elem{elem_idx}.png"
    for fname in os.listdir(icons_dir):
        if not fname.startswith(prefix):
            continue
        if fname.endswith(suffix) or fname.endswith(bare_suffix):
            return os.path.join(icons_dir, fname)
    return None


def _label_from_icon_filename(fname: str) -> str:
    """Extract a human-readable label from the icon filename."""
    stem = Path(fname).stem
    match = re.search(r'-\d+_(.+)_elem\d+$', stem)
    if match:
        return match.group(1).replace("_", " ").title()
    match = re.search(r'-\d+_(elem\d+)$', stem)
    if match:
        return match.group(1)
    return stem


_gpt_compose_module = None


def _load_gpt_compose():
    """Lazy-load the GPT compose module to avoid import overhead when not needed."""
    global _gpt_compose_module
    if _gpt_compose_module is not None:
        return _gpt_compose_module

    import sys
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    import amex_sim2real_compose_action_coord as mod
    _gpt_compose_module = mod
    return mod


def _build_elements_from_anno(
    element_anno_path: str,
    icons_dir: str,
    traj_stem: str,
    step_id: int,
    screenshot_path: str = "",
    output_dir: str = "",
) -> List[dict]:
    """Build the element list format expected by the compose pipeline.

    Each element dict has: index, label, type, bbox, asset_path.
    When an extracted icon is missing, falls back to cropping directly
    from the GT screenshot (saves to output_dir).
    """
    if not os.path.exists(element_anno_path):
        return []

    with open(element_anno_path, "r", encoding="utf-8") as f:
        anno = json.load(f)

    screenshot = None
    if screenshot_path and os.path.exists(screenshot_path):
        screenshot = Image.open(screenshot_path).convert("RGBA")
        ss_w, ss_h = screenshot.size
    else:
        ss_w, ss_h = 0, 0

    elements = []
    for idx, elem in enumerate(anno.get("clickable_elements", [])):
        bbox = elem.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        if w < 5 or h < 5:
            continue

        xml_desc = elem.get("xml_desc", [])
        label = xml_desc[0] if xml_desc else f"elem{idx}"

        icon_path = _find_icon_file(icons_dir, traj_stem, step_id, idx)

        if not icon_path and screenshot is not None and output_dir:
            x1 = max(0, min(bbox[0], ss_w))
            y1 = max(0, min(bbox[1], ss_h))
            x2 = max(0, min(bbox[2], ss_w))
            y2 = max(0, min(bbox[3], ss_h))
            if x2 - x1 >= 5 and y2 - y1 >= 5:
                crop = screenshot.crop((x1, y1, x2, y2))
                fallback_dir = os.path.join(output_dir, "_screenshot_fallbacks")
                os.makedirs(fallback_dir, exist_ok=True)
                safe_label = re.sub(r"[^0-9a-zA-Z]+", "_", label).strip("_") or f"elem{idx}"
                fallback_path = os.path.join(fallback_dir, f"step{step_id}_elem{idx}_{safe_label}.png")
                crop.save(fallback_path)
                icon_path = fallback_path

        elements.append({
            "index": idx,
            "label": label,
            "type": "icon" if w < 300 and h < 300 else "content",
            "bbox": bbox,
            "asset_path": icon_path or "",
        })

    return elements


def _generate_centered_position_code(elements: List[dict], orig_size: Tuple[int, int]) -> str:
    """Generate PIL code that pastes icons centered at their bbox center, preserving aspect ratio."""
    lines = ["# --- Auto-generated: paste icons centered at bbox positions ---"]

    for e in elements:
        x1, y1, x2, y2 = e["bbox"]
        ew, eh = x2 - x1, y2 - y1
        if ew < 5 or eh < 5:
            continue
        idx = e["index"]
        label = e["label"].replace('"', "'").replace(" ", "_").replace("/", "_")[:25]

        lines.append(
            f'try:\n'
            f'    _c{idx} = get_crop({idx}, {ew}, {eh})\n'
            f'    _iw, _ih = _c{idx}.size\n'
            f'    _cx = {x1} + ({ew} - _iw) // 2\n'
            f'    _cy = {y1} + ({eh} - _ih) // 2\n'
            f'    canvas.paste(_c{idx}, ({max(0, x1)} if _iw >= {ew} else _cx, {max(0, y1)} if _ih >= {eh} else _cy), _c{idx})\n'
            f'except Exception:\n'
            f'    pass\n'
            f'layout["{label}"] = [{x1}, {y1}, {x2}, {y2}]'
        )

    return "\n\n".join(lines)


def _compose_step_gpt(
    screenshot_path: str,
    element_anno_path: str,
    icons_dir: str,
    traj_stem: str,
    step_id: int,
    device_dim: List[int],
    model_name: str,
    client: Any,
    output_dir: str = "",
) -> Tuple[Image.Image, Dict[str, Any], List[dict]]:
    """Compose a page using GPT-5-Mini for styling + deterministic icon placement."""
    mod = _load_gpt_compose()

    dev_w = device_dim[0] if device_dim else 1440
    dev_h = device_dim[1] if device_dim else 2960

    elements = _build_elements_from_anno(
        element_anno_path, icons_dir, traj_stem, step_id, screenshot_path, output_dir,
    )

    if not elements:
        canvas = Image.new("RGB", (CANVAS_W, CANVAS_H), (245, 245, 248))
        draw = ImageDraw.Draw(canvas)
        layout: Dict[str, Any] = {}
        _draw_nav_strip(draw, layout)
        return canvas, layout, []

    orig_size = (dev_w, dev_h)

    styling_code = mod.generate_styling_code(
        client, model_name, screenshot_path, elements, orig_size
    )
    if styling_code is None:
        bg = _extract_bg_color(screenshot_path)
        styling_code = f"draw.rectangle([0, 0, {dev_w}, {dev_h}], fill={bg})"

    position_code = _generate_centered_position_code(elements, orig_size)
    full_code = styling_code + "\n\n" + position_code

    page_img, layout_raw = _render_with_natural_sizing(mod, full_code, elements, orig_size)

    if page_img is None:
        page_img, layout_raw = _fallback_compose_from_assets(elements, orig_size, screenshot_path, mod)

    if layout_raw is None:
        layout_raw = {}

    final_canvas = Image.new("RGB", (CANVAS_W, CANVAS_H), (250, 250, 250))
    final_canvas.paste(
        page_img.convert("RGB"),
        (mod.PHONE_OFFSET_X, mod.PHONE_OFFSET_Y),
    )

    shifted_layout: Dict[str, Any] = {}
    for key, bbox in layout_raw.items():
        if key in ("back", "home"):
            continue
        shifted_layout[key] = [
            bbox[0] + mod.PHONE_OFFSET_X,
            bbox[1] + mod.PHONE_OFFSET_Y,
            bbox[2] + mod.PHONE_OFFSET_X,
            bbox[3] + mod.PHONE_OFFSET_Y,
        ]

    final_canvas, shifted_layout = mod._ensure_system_nav_controls(
        final_canvas, shifted_layout
    )

    icon_report = []
    for elem in elements:
        has_asset = bool(elem.get("asset_path") and os.path.exists(elem["asset_path"]))
        canvas_bbox = _scale_bbox(elem["bbox"], dev_w, dev_h)
        icon_report.append({
            "step_id": step_id,
            "elem_idx": elem["index"],
            "label": elem["label"],
            "device_bbox": elem["bbox"],
            "canvas_bbox": canvas_bbox,
            "crop_source": "gpt_compose" if has_asset else "gpt_compose_no_icon",
        })

    return final_canvas, shifted_layout, icon_report


def _fallback_compose_from_assets(
    elements: List[dict],
    orig_size: Tuple[int, int],
    screenshot_path: str,
    mod,
) -> Tuple[Image.Image, dict]:
    """Fallback when GPT code fails: place icons at scaled positions on bg color."""
    bg = _extract_bg_color(screenshot_path)
    canvas = Image.new("RGB", mod.CANVAS_SIZE, bg)
    layout = {}

    ow, oh = orig_size
    cw, ch = mod.CANVAS_SIZE
    sx = cw / ow
    sy = ch / oh

    for e in elements:
        x1, y1, x2, y2 = e["bbox"]
        cx1 = int(x1 * sx)
        cy1 = int(y1 * sy)
        cx2 = int(x2 * sx)
        cy2 = int(y2 * sy)
        bw = max(cx2 - cx1, 4)
        bh = max(cy2 - cy1, 4)

        asset_path = e.get("asset_path")
        if asset_path and os.path.exists(asset_path):
            icon = Image.open(asset_path).convert("RGBA")
            if icon.width > bw or icon.height > bh:
                icon = ImageOps.contain(icon, (bw, bh))
            px = cx1 + (bw - icon.width) // 2
            py = cy1 + (bh - icon.height) // 2
            canvas.paste(icon, (max(0, px), max(0, py)), icon)

        label = e["label"].replace(" ", "_")[:25]
        if label in layout:
            label = f"{label}_{e['index']}"
        layout[label] = [cx1, cy1, cx2, cy2]

    return canvas, layout


def _render_with_natural_sizing(mod, code_str: str, elements: List[dict],
                                 orig_size: Tuple[int, int]):
    """Like render_from_code but get_crop returns icons at natural size, not stretched."""
    import random as _random
    ow, oh = orig_size
    canvas = Image.new("RGB", (ow, oh), (250, 250, 250))
    draw = ImageDraw.Draw(canvas)
    layout = {}

    font_sm = mod._try_load_font(12)
    font_md = mod._try_load_font(18)
    font_lg = mod._try_load_font(24)
    font_xl = mod._try_load_font(32)

    def get_crop(index, w=50, h=50):
        if 0 <= index < len(elements):
            elem = elements[index]
            asset_path = elem.get("asset_path")
            if asset_path and os.path.exists(asset_path):
                crop = Image.open(asset_path).convert("RGBA")
                iw, ih = crop.size
                if iw > int(w) or ih > int(h):
                    crop = ImageOps.contain(crop, (int(w), int(h)))
                return crop
        ph = Image.new("RGBA", (int(w), int(h)), (200, 200, 200, 255))
        return ph

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
        "canvas": canvas, "draw": draw, "layout": layout,
        "font_sm": font_sm, "font_md": font_md,
        "font_lg": font_lg, "font_xl": font_xl,
        "get_crop": get_crop,
        "Image": Image, "ImageDraw": ImageDraw,
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
        "random": _random,
    }

    code_str = mod._sanitize_code(code_str)

    try:
        exec(code_str, namespace)
    except Exception as e:
        print(f"\n  Code execution error: {e}")
        fixed = mod._try_fix_code(code_str, str(e))
        if fixed and fixed != code_str:
            try:
                canvas = Image.new("RGB", (ow, oh), (250, 250, 250))
                draw = ImageDraw.Draw(canvas)
                layout.clear()
                namespace["canvas"] = canvas
                namespace["draw"] = draw
                exec(fixed, namespace)
            except Exception as e2:
                print(f"\n  Retry also failed: {e2}")
                return None, None
        else:
            return None, None

    canvas_resized, _, _, _ = mod._fit_image_to_box(canvas, mod.CANVAS_SIZE, (250, 250, 250))

    scaled_layout = {}
    for key, bbox in layout.items():
        scaled_layout[key] = mod._scale_bbox_to_box(bbox, (ow, oh), mod.CANVAS_SIZE)

    scaled_layout.pop("back", None)
    scaled_layout.pop("home", None)

    return canvas_resized, scaled_layout


def _scale_screenshot_to_canvas(screenshot: Image.Image, dev_w: int, dev_h: int) -> Image.Image:
    """Scale the full GT screenshot to canvas size, accounting for the nav strip."""
    phone_h = CANVAS_H - NAV_STRIP_H
    scaled = screenshot.resize((CANVAS_W, phone_h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (CANVAS_W, CANVAS_H), (240, 240, 243))
    canvas.paste(scaled, (0, NAV_STRIP_H))
    return canvas


def compose_step(
    screenshot_path: str,
    element_anno_path: str,
    icons_dir: str,
    traj_stem: str,
    step_id: int,
    device_dim: List[int],
    crop_mode: str = "icons_on_screenshot",
    model_name: str = "gpt-5-mini-2025-08-07",
    client: Any = None,
    output_dir: str = "",
) -> Tuple[Image.Image, Dict[str, Any], List[dict]]:
    """Compose a single page using one of five crop modes.

    crop_mode options:
      "screenshot"            - Crop elements directly from the GT screenshot (rectangular).
      "icons_on_screenshot"   - Extracted icons on scaled GT background.
      "icons_only"            - Extracted icons on flat background color.
      "deterministic_compose" - Extracted icons on a deterministic gradient background
                                sampled from the GT screenshot. No GPT needed. Falls back
                                to GT screenshot crops for elements without extracted icons.
      "gpt_compose"           - GPT-5-Mini generates background styling PIL code, then
                                extracted icons are pasted deterministically.

    Returns (canvas_image, layout_dict, icon_report_rows).
    """
    if crop_mode == "gpt_compose":
        return _compose_step_gpt(
            screenshot_path=screenshot_path,
            element_anno_path=element_anno_path,
            icons_dir=icons_dir,
            traj_stem=traj_stem,
            step_id=step_id,
            device_dim=device_dim,
            model_name=model_name,
            client=client,
            output_dir=output_dir,
        )
    layout: Dict[str, Any] = {}
    icon_report: List[dict] = []

    dev_w = device_dim[0] if device_dim else 1440
    dev_h = device_dim[1] if device_dim else 2960

    screenshot = None
    if os.path.exists(screenshot_path):
        screenshot = Image.open(screenshot_path).convert("RGB")
        ss_w, ss_h = screenshot.size
    else:
        ss_w, ss_h = dev_w, dev_h

    if crop_mode == "icons_on_screenshot" and screenshot is not None:
        canvas = _scale_screenshot_to_canvas(screenshot, dev_w, dev_h)
    elif crop_mode == "deterministic_compose":
        colors = _sample_screen_colors(screenshot_path)
        canvas = Image.new("RGB", (CANVAS_W, CANVAS_H), colors["mid"])
        draw_bg = ImageDraw.Draw(canvas)
        _draw_deterministic_background(canvas, draw_bg, colors)
    else:
        bg_color = _extract_bg_color(screenshot_path)
        canvas = Image.new("RGB", (CANVAS_W, CANVAS_H), bg_color)

    draw = ImageDraw.Draw(canvas)
    _draw_nav_strip(draw, layout)

    if not os.path.exists(element_anno_path):
        return canvas, layout, icon_report

    with open(element_anno_path, "r", encoding="utf-8") as f:
        anno = json.load(f)

    elements = anno.get("clickable_elements", [])

    for idx, elem in enumerate(elements):
        raw_bbox = elem.get("bbox")
        if not raw_bbox or len(raw_bbox) != 4:
            continue

        canvas_bbox = _scale_bbox(raw_bbox, dev_w, dev_h)
        cb_w = canvas_bbox[2] - canvas_bbox[0]
        cb_h = canvas_bbox[3] - canvas_bbox[1]
        if cb_w < 4 or cb_h < 4:
            continue

        xml_desc = elem.get("xml_desc", [])
        label = xml_desc[0] if xml_desc else f"elem{idx}"

        crop_source = "none"

        if crop_mode == "screenshot":
            if screenshot is not None:
                x1 = max(0, min(raw_bbox[0], ss_w))
                y1 = max(0, min(raw_bbox[1], ss_h))
                x2 = max(0, min(raw_bbox[2], ss_w))
                y2 = max(0, min(raw_bbox[3], ss_h))
                if x2 - x1 >= 4 and y2 - y1 >= 4:
                    crop_img = screenshot.crop((x1, y1, x2, y2))
                    resized = crop_img.resize((cb_w, cb_h), Image.Resampling.LANCZOS)
                    canvas.paste(resized, (canvas_bbox[0], canvas_bbox[1]))
                    crop_source = "screenshot"

        elif crop_mode in ("icons_on_screenshot", "icons_only", "deterministic_compose"):
            icon_path = _find_icon_file(icons_dir, traj_stem, step_id, idx)
            if icon_path and os.path.exists(icon_path):
                icon_img = Image.open(icon_path).convert("RGBA")
                cx = canvas_bbox[0] + cb_w // 2
                cy = canvas_bbox[1] + cb_h // 2
                paste_x = cx - icon_img.width // 2
                paste_y = cy - icon_img.height // 2
                canvas.paste(icon_img, (paste_x, paste_y), icon_img)
                crop_source = "extracted_icon"
            elif crop_mode == "deterministic_compose" and screenshot is not None:
                x1 = max(0, min(raw_bbox[0], ss_w))
                y1 = max(0, min(raw_bbox[1], ss_h))
                x2 = max(0, min(raw_bbox[2], ss_w))
                y2 = max(0, min(raw_bbox[3], ss_h))
                if x2 - x1 >= 4 and y2 - y1 >= 4:
                    crop_img = screenshot.crop((x1, y1, x2, y2))
                    resized = crop_img.resize((cb_w, cb_h), Image.Resampling.LANCZOS)
                    canvas.paste(resized, (canvas_bbox[0], canvas_bbox[1]))
                    crop_source = "screenshot_fallback"

        if crop_source == "none":
            icon_path = _find_icon_file(icons_dir, traj_stem, step_id, idx)
            if icon_path and os.path.exists(icon_path):
                icon_img = Image.open(icon_path).convert("RGBA")
                cx = canvas_bbox[0] + cb_w // 2
                cy = canvas_bbox[1] + cb_h // 2
                paste_x = cx - icon_img.width // 2
                paste_y = cy - icon_img.height // 2
                canvas.paste(icon_img, (paste_x, paste_y), icon_img)
                crop_source = "extracted_icon_fallback"
            else:
                draw.rounded_rectangle(canvas_bbox, radius=8, fill=(220, 220, 225), outline=(180, 180, 190), width=1)

        report_row = {
            "step_id": step_id,
            "elem_idx": idx,
            "label": label,
            "device_bbox": raw_bbox,
            "canvas_bbox": canvas_bbox,
            "crop_source": crop_source,
        }

        safe_label = re.sub(r"[^0-9a-zA-Z]+", "_", label).strip("_") or f"elem{idx}"
        layout[safe_label] = canvas_bbox
        icon_report.append(report_row)

    return canvas, layout, icon_report


def build_side_by_side(screenshot_path: str, composed: Image.Image) -> Image.Image:
    """Place original screenshot and composed page side by side."""
    orig = Image.open(screenshot_path).convert("RGB")
    orig_resized = ImageOps.contain(orig, (CANVAS_W, CANVAS_H))

    gap = 20
    sheet_w = CANVAS_W * 2 + gap
    sheet_h = CANVAS_H
    sheet = Image.new("RGB", (sheet_w, sheet_h), (230, 230, 233))
    sheet.paste(orig_resized, ((CANVAS_W - orig_resized.width) // 2, 0))
    sheet.paste(composed, (CANVAS_W + gap, 0))

    draw = ImageDraw.Draw(sheet)
    draw.text((10, 10), "Original", fill=(255, 60, 60), font=FONT_MD)
    draw.text((CANVAS_W + gap + 10, 10), "Composed", fill=(60, 60, 255), font=FONT_MD)
    return sheet


def compose_trajectory(
    trajectory: dict,
    traj_stem: str,
    screenshots_dir: str,
    element_anno_dir: str,
    icons_base_dir: str,
    output_dir: str,
    max_steps: Optional[int] = None,
    crop_mode: str = "icons_on_screenshot",
    model_name: str = "gpt-5-mini-2025-08-07",
    client: Any = None,
) -> dict:
    """Compose all steps of a trajectory. Returns a manifest dict."""
    pages_dir = os.path.join(output_dir, "pages")
    compare_dir = os.path.join(output_dir, "comparisons")
    os.makedirs(pages_dir, exist_ok=True)
    os.makedirs(compare_dir, exist_ok=True)

    episode_id = trajectory.get("episode_id", "")
    steps = trajectory.get("steps", [])
    if max_steps is not None:
        steps = steps[:max_steps]

    icons_dir = os.path.join(icons_base_dir, traj_stem)
    if not os.path.isdir(icons_dir):
        print(f"WARNING: icons directory not found: {icons_dir}")
        icons_dir = ""

    all_icon_reports: List[dict] = []
    pages_data: List[dict] = []

    for local_idx, step in enumerate(steps):
        step_id = local_idx + 1
        image_name = step.get("image_path") or f"{traj_stem}-{step_id}.png"
        screenshot_path = os.path.join(screenshots_dir, image_name)
        anno_stem = Path(image_name).stem
        element_anno_path = os.path.join(element_anno_dir, f"{anno_stem}.json")
        device_dim = step.get("device_dim") or [1440, 2960]

        page_id = f"page_{local_idx}"

        if not os.path.exists(screenshot_path):
            print(f"  [{step_id}/{len(steps)}] SKIP - screenshot missing: {image_name}")
            continue

        composed, layout, icon_report = compose_step(
            screenshot_path=screenshot_path,
            element_anno_path=element_anno_path,
            icons_dir=icons_dir,
            traj_stem=traj_stem,
            step_id=step_id,
            device_dim=device_dim,
            crop_mode=crop_mode,
            model_name=model_name,
            client=client,
            output_dir=output_dir,
        )

        composed.save(os.path.join(pages_dir, f"{page_id}.png"))
        comparison = build_side_by_side(screenshot_path, composed)
        comparison.save(os.path.join(compare_dir, f"step_{step_id:02d}.png"))
        all_icon_reports.extend(icon_report)

        action = str(step.get("action", "")).strip().upper()
        touch_coord = step.get("touch_coord") or [0, 0]
        lift_coord = step.get("lift_coord") or [0, 0]
        canvas_touch = _scale_point(touch_coord, device_dim[0], device_dim[1])
        canvas_lift = _scale_point(lift_coord, device_dim[0], device_dim[1])

        next_page = f"page_{local_idx + 1}" if local_idx + 1 < len(steps) else page_id
        if action == "TASK_COMPLETE":
            next_page = page_id

        transition = {
            "action": action,
            "target_page": next_page,
            "action_coord": canvas_touch,
            "icon_bbox": [0, 0, 0, 0],
        }
        if action in ("SWIPE", "SCROLL"):
            transition["lift_coord"] = canvas_lift
        if action in ("TYPE", "TEXT"):
            transition["type_text"] = str(step.get("type_text", "") or "")

        pages_data.append({
            "page_id": page_id,
            "image": f"{page_id}.png",
            "depth": local_idx,
            "layout": layout,
            "transitions": [transition],
            "step": step,
        })

        placed = sum(1 for r in icon_report if r.get("crop_source") != "none")
        print(f"  [{step_id}/{len(steps)}] {action} | elements={len(icon_report)} placed={placed}/{len(icon_report)}")

    ui_structure = _build_ui_structure(pages_data, trajectory)
    with open(os.path.join(output_dir, "ui_structure.json"), "w", encoding="utf-8") as f:
        json.dump(ui_structure, f, indent=2)

    report = _build_quality_report(all_icon_reports)
    with open(os.path.join(output_dir, "icon_quality_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    _print_quality_summary(report)
    return report


def _build_ui_structure(pages_data: List[dict], trajectory: dict) -> dict:
    pages = {}
    for pd in pages_data:
        page_id = pd["page_id"]
        serialized_layout = {
            key: {"bbox": bbox, "type": "system" if key in ("back", "home") else "normal"}
            for key, bbox in pd["layout"].items()
        }
        serialized_transitions = []
        for t in pd["transitions"]:
            st = {
                "action": t["action"],
                "target_page": t["target_page"],
                "action_coord": t["action_coord"],
            }
            if "lift_coord" in t:
                st["lift_coord"] = t["lift_coord"]
            if "icon_bbox" in t and t["icon_bbox"] != [0, 0, 0, 0]:
                st["icon_bbox"] = t["icon_bbox"]
            if "type_text" in t and t["type_text"]:
                st["type_text"] = t["type_text"]
            serialized_transitions.append(st)

        pages[page_id] = {
            "image": pd["image"],
            "depth": pd["depth"],
            "layout": serialized_layout,
            "transitions": serialized_transitions,
        }

    return {
        "pages": pages,
        "metadata": {
            "source": "amex_compose_deterministic",
            "episode_id": trajectory.get("episode_id", ""),
            "instruction": trajectory.get("instruction", ""),
            "total_pages": len(pages),
            "canvas_size": [CANVAS_W, CANVAS_H],
        },
    }


def _build_quality_report(icon_reports: List[dict]) -> dict:
    total = len(icon_reports)
    source_counts: Dict[str, int] = {}
    for r in icon_reports:
        src = r.get("crop_source", "none")
        source_counts[src] = source_counts.get(src, 0) + 1
    missing = source_counts.get("none", 0)

    return {
        "summary": {
            "total_elements": total,
            "source_distribution": source_counts,
            "missing": missing,
            "placement_rate": round((total - missing) / max(total, 1), 3),
        },
        "per_icon": icon_reports,
    }


def _print_quality_summary(report: dict) -> None:
    s = report["summary"]
    print("\n=== Composition Report ===")
    print(f"  Total elements:  {s['total_elements']}")
    for src, count in sorted(s["source_distribution"].items()):
        print(f"    {src:30s}: {count}")
    print(f"  Missing:         {s['missing']}")
    print(f"  Placement rate:  {s['placement_rate']:.1%}")


def parse_args():
    parser = argparse.ArgumentParser(description="Deterministic AMEX page composition for icon quality evaluation")
    parser.add_argument("--trajectory_id", type=str,
                        default="2024_3_20_16_19_c7de79edba6b442681ddcdc40adb3900",
                        help="AMEX trajectory stem (annotation filename without .json)")
    parser.add_argument("--annotations_dir", type=str, default=DEFAULT_ANNOTATIONS_DIR)
    parser.add_argument("--screenshots_dir", type=str, default=DEFAULT_SCREENSHOTS_DIR)
    parser.add_argument("--element_anno_dir", type=str, default=DEFAULT_ELEMENT_ANNO_DIR)
    parser.add_argument("--icons_dir", type=str, default=DEFAULT_ICONS_DIR)
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: data_engine/amex_deterministic_compose/<trajectory_id>)")
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--crop_mode", type=str, default="deterministic_compose",
                        choices=["screenshot", "icons_on_screenshot", "icons_only",
                                 "deterministic_compose", "gpt_compose"],
                        help="screenshot: GT bbox crops. "
                             "icons_on_screenshot: extracted icons on scaled GT background. "
                             "icons_only: extracted icons on flat color. "
                             "deterministic_compose: extracted icons on sampled gradient bg "
                             "(no GPT, fully deterministic, screenshot crop fallback). "
                             "gpt_compose: GPT-5-Mini styling + icon paste (requires OPENAI_API_KEY).")
    parser.add_argument("--model_name", type=str, default="gpt-5-mini-2025-08-07",
                        help="OpenAI model for gpt_compose mode.")
    return parser.parse_args()


def main():
    args = parse_args()
    traj_stem = args.trajectory_id
    anno_path = os.path.join(args.annotations_dir, f"{traj_stem}.json")
    if not os.path.exists(anno_path):
        raise FileNotFoundError(f"Annotation not found: {anno_path}")

    with open(anno_path, "r", encoding="utf-8") as f:
        trajectory = json.load(f)

    output_dir = args.output_dir or os.path.join("data_engine", "amex_deterministic_compose", traj_stem)
    os.makedirs(output_dir, exist_ok=True)

    client = None
    if args.crop_mode == "gpt_compose":
        mod = _load_gpt_compose()
        client = mod.load_api_client()

    print(f"Trajectory: {traj_stem}")
    print(f"Episode:    {trajectory.get('episode_id', '')}")
    print(f"Instruction: {trajectory.get('instruction', '')[:120]}")
    print(f"Steps:      {len(trajectory.get('steps', []))}")
    print(f"Crop mode:  {args.crop_mode}")
    print(f"Output:     {output_dir}")
    print()

    compose_trajectory(
        trajectory=trajectory,
        traj_stem=traj_stem,
        screenshots_dir=args.screenshots_dir,
        element_anno_dir=args.element_anno_dir,
        icons_base_dir=args.icons_dir,
        output_dir=output_dir,
        max_steps=args.max_steps,
        crop_mode=args.crop_mode,
        model_name=args.model_name,
        client=client,
    )


if __name__ == "__main__":
    main()
