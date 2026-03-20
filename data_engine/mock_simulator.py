"""
Standalone mock AMEX-style environment builder for a home page + app drawer.

This script creates:
  - pages/page_0_home.png
  - pages/page_1_app_drawer.png
  - ui_structure.json
  - ui_structure_layer.json
  - action_coord_debug/*.png

The full canvas is kept at 1080x24000 as requested, while the visible phone UI
is drawn in the upper 2400px region so it still looks like a normal screen.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Tuple

from PIL import Image, ImageDraw, ImageFont, ImageOps


CANVAS_SIZE = (1080, 2400)
VISIBLE_VIEWPORT_HEIGHT = 2400
STATUS_BAR_HEIGHT = 72
TOP_MARGIN = 48
SIDE_MARGIN = 56
SYSTEM_BUTTON_W = 248
SYSTEM_BUTTON_H = 96
SYSTEM_BUTTON_Y_OFFSET = -34
SEARCH_BAR_H = 110
ICON_TILE_SIZE = 156
APP_LABEL_GAP = 18
ICON_GRID_COLS = 4

HOME_PAGE_ID = "page_0_home"
DRAWER_PAGE_ID = "page_1_app_drawer"

EVENTBRITE_ICON_PATH = Path(
    "/ext_hdd2/ttran/datasets/AMEX_dataset/ui_elements_output_clean/elements/"
    "2024_3_20_16_19_c7de79edba6b442681ddcdc40adb3900/"
    "2024_3_20_16_19_c7de79edba6b442681ddcdc40adb3900-2_eventbrite_elem20.png"
)
SEATGEEK_ICON_PATH = Path(
    "/ext_hdd2/ttran/datasets/AMEX_dataset/ui_elements_output_clean/elements/"
    "2024_3_20_16_19_c7de79edba6b442681ddcdc40adb3900/"
    "2024_3_20_16_19_c7de79edba6b442681ddcdc40adb3900-3_seatgeek_elem22.png"
)
GALLERY_REAL_ICON_PATH = Path(
    "/ext_hdd2/ttran/datasets/AMEX_dataset/ui_elements_output_clean/elements/"
    "2024_4_9_10_38_1037a32bd0de450985efaa31575e2023/"
    "2024_4_9_10_38_1037a32bd0de450985efaa31575e2023-2_gallery_elem15.png"
)
NEWS_REAL_ICON_PATH = Path(
    "/ext_hdd2/ttran/datasets/AMEX_dataset/ui_elements_output_clean/elements/"
    "2024_4_9_10_38_1037a32bd0de450985efaa31575e2023/"
    "2024_4_9_10_38_1037a32bd0de450985efaa31575e2023-2_news_elem22.png"
)
SMARTNEWS_REAL_ICON_PATH = Path(
    "/ext_hdd2/ttran/datasets/AMEX_dataset/ui_elements_output_clean/elements/"
    "2024_4_9_10_38_1037a32bd0de450985efaa31575e2023/"
    "2024_4_9_10_38_1037a32bd0de450985efaa31575e2023-2_smartnews_elem27.png"
)
KKBOX_REAL_ICON_PATH = Path(
    "/ext_hdd2/ttran/datasets/AMEX_dataset/ui_elements_output_clean/elements/"
    "2024_4_9_10_38_1037a32bd0de450985efaa31575e2023/"
    "2024_4_9_10_38_1037a32bd0de450985efaa31575e2023-2_kkbox_elem19.png"
)
NOVELSHIP_REAL_ICON_PATH = Path(
    "/ext_hdd2/ttran/datasets/AMEX_dataset/ui_elements_output_robust/elements/"
    "2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197/"
    "2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-2_novelship_elem35.png"
)
CITYMAPPER_REAL_ICON_PATH = Path(
    "/ext_hdd2/ttran/datasets/AMEX_dataset/ui_elements_output_robust/elements/"
    "2024_6_20_14_27_962363f8d8cd4f60bf2ea2d0f578d023/"
    "2024_6_20_14_27_962363f8d8cd4f60bf2ea2d0f578d023-2_citymapper_elem0.png"
)
AUDIO_MACK_REAL_ICON_PATH = Path(
    "/ext_hdd2/ttran/datasets/AMEX_dataset/ui_elements_output_clean/elements/"
    "2024_4_9_10_38_1037a32bd0de450985efaa31575e2023/"
    "2024_4_9_10_38_1037a32bd0de450985efaa31575e2023-2_audio_mack_elem4.png"
)
SUPERUSER_REAL_ICON_PATH = Path(
    "/ext_hdd2/ttran/datasets/AMEX_dataset/ui_elements_output_clean/elements/"
    "2024_4_9_10_38_1037a32bd0de450985efaa31575e2023/"
    "2024_4_9_10_38_1037a32bd0de450985efaa31575e2023-2_superuser_elem29.png"
)
BROWSER_REAL_ICON_PATH = Path(
    "/ext_hdd2/ttran/datasets/AMEX_dataset/ui_elements_output_clean/elements/"
    "2024_4_9_10_38_1037a32bd0de450985efaa31575e2023/"
    "2024_4_9_10_38_1037a32bd0de450985efaa31575e2023-2_webview_browser_tester_elem31.png"
)
MUSIC_REAL_ICON_PATH = Path(
    "/ext_hdd2/ttran/datasets/AMEX_dataset/ui_elements_output_clean/elements/"
    "2024_4_9_10_38_1037a32bd0de450985efaa31575e2023/"
    "2024_4_9_10_38_1037a32bd0de450985efaa31575e2023-3_music_elem2.png"
)
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_LAYOUT_CONFIG_PATH = SCRIPT_DIR / "mock_simulator_layout_config.json"

REAL_ICON_LIBRARY = {
    "eventbrite": {"label": "Eventbrite", "path": EVENTBRITE_ICON_PATH},
    "seatgeek": {"label": "SeatGeek", "path": SEATGEEK_ICON_PATH},
    "gallery_real": {"label": "Gallery", "path": GALLERY_REAL_ICON_PATH},
    "news_real": {"label": "News", "path": NEWS_REAL_ICON_PATH},
    "smartnews_real": {"label": "SmartNews", "path": SMARTNEWS_REAL_ICON_PATH},
    "kkbox": {"label": "KKBOX", "path": KKBOX_REAL_ICON_PATH},
    "kkbox_real": {"label": "KKBOX", "path": KKBOX_REAL_ICON_PATH},
    "novelship": {"label": "Novelship", "path": NOVELSHIP_REAL_ICON_PATH},
    "novelship_real": {"label": "Novelship", "path": NOVELSHIP_REAL_ICON_PATH},
    "citymapper": {"label": "Citymapper", "path": CITYMAPPER_REAL_ICON_PATH},
    "citymapper_real": {"label": "Citymapper", "path": CITYMAPPER_REAL_ICON_PATH},
    "audiomack_real": {"label": "Audiomack", "path": AUDIO_MACK_REAL_ICON_PATH},
    "superuser_real": {"label": "Superuser", "path": SUPERUSER_REAL_ICON_PATH},
    "browser_real": {"label": "Browser", "path": BROWSER_REAL_ICON_PATH},
    "music_real": {"label": "Music", "path": MUSIC_REAL_ICON_PATH},
}
RANDOM_HOME_REAL_ASSETS = [
    "gallery_real",
    "news_real",
    "smartnews_real",
    "kkbox_real",
    "audiomack_real",
    "superuser_real",
    "browser_real",
    "music_real",
]

CANONICAL_REAL_ICON_ASSETS = {
    "kkbox": "kkbox_real",
    "kkbox_real": "kkbox_real",
    "novelship": "novelship_real",
    "novelship_real": "novelship_real",
    "citymapper": "citymapper_real",
    "citymapper_real": "citymapper_real",
}

DEFAULT_LAYOUT_CONFIG = {
    "metadata": {
        "style": "realistic_launcher_v2",
        "random_seed": None,
    },
    "_instructions": [
        "Each icon can use either 'slot' or direct 'x' and 'y' coordinates.",
        "x and y are the top-left coordinates of the icon tile in pixels.",
        "You can swap icons by changing slot names, or place them freely with x/y.",
        "Use asset=random_home_real to sample a realistic app icon from the bundled AMEX asset pool.",
    ],
    "slots": {
        "home_slot_1": {"x": 99, "y": 760},
        "home_slot_2": {"x": 341, "y": 760},
        "home_slot_3": {"x": 583, "y": 760},
        "home_slot_4": {"x": 825, "y": 760},
        "home_slot_5": {"x": 99, "y": 1032},
        "home_slot_6": {"x": 341, "y": 1032},
        "home_slot_7": {"x": 583, "y": 1032},
        "home_slot_8": {"x": 825, "y": 1032},
        "drawer_slot_1": {"x": 132, "y": 560},
        "drawer_slot_2": {"x": 362, "y": 560},
        "drawer_slot_3": {"x": 592, "y": 560},
        "drawer_slot_4": {"x": 822, "y": 560},
        "drawer_slot_5": {"x": 132, "y": 828},
        "drawer_slot_6": {"x": 362, "y": 828},
        "drawer_slot_7": {"x": 592, "y": 828},
        "drawer_slot_8": {"x": 822, "y": 828},
        "drawer_slot_9": {"x": 132, "y": 1096},
        "drawer_slot_10": {"x": 362, "y": 1096},
        "drawer_slot_11": {"x": 592, "y": 1096},
        "drawer_slot_12": {"x": 822, "y": 1096},
        "drawer_slot_13": {"x": 132, "y": 1364},
        "drawer_slot_14": {"x": 362, "y": 1364},
        "drawer_slot_15": {"x": 592, "y": 1364},
        "drawer_slot_16": {"x": 822, "y": 1364},
        "drawer_slot_17": {"x": 132, "y": 1632},
        "drawer_slot_18": {"x": 362, "y": 1632},
        "drawer_slot_19": {"x": 592, "y": 1632},
        "drawer_slot_20": {"x": 822, "y": 1632},
        "drawer_slot_21": {"x": 132, "y": 1900},
        "drawer_slot_22": {"x": 362, "y": 1900},
        "drawer_slot_23": {"x": 592, "y": 1900},
        "drawer_slot_24": {"x": 822, "y": 1900},
        "dock_slot_1": {"x": 204, "y": 1988},
        "dock_slot_2": {"x": 382, "y": 1988},
        "dock_slot_3": {"x": 560, "y": 1988},
        "dock_slot_4": {"x": 738, "y": 1988},
    },
    "pages": {
        "home": {
            "icons": [
                {"label": "SeatGeek", "asset": "seatgeek", "slot": "home_slot_1", "show_label": False},
                {"label": "KKBOX", "asset": "kkbox", "slot": "home_slot_2", "show_label": False},
                {"label": "Novelship", "asset": "novelship", "slot": "home_slot_3", "show_label": True},
                {"label": "Citymapper", "asset": "citymapper", "slot": "home_slot_4", "show_label": True},
                {"asset": "random_home_real", "slot": "home_slot_5", "show_label": False},
                {"asset": "random_home_real", "slot": "home_slot_6", "show_label": False},
                {"asset": "random_home_real", "slot": "home_slot_7", "show_label": False},
                {"asset": "random_home_real", "slot": "home_slot_8", "show_label": False},
            ],
            "dock_icons": [
                {"label": "Phone", "asset": "phone", "slot": "dock_slot_1", "size": 112, "show_label": True},
                {"label": "Chrome", "asset": "chrome", "slot": "dock_slot_2", "size": 112, "show_label": True},
                {"label": "Messages", "asset": "messages", "slot": "dock_slot_3", "size": 112, "show_label": True},
                {"label": "Camera", "asset": "camera", "slot": "dock_slot_4", "size": 112, "show_label": True},
            ],
        },
        "app_drawer": {
            "icons": [
                {"label": "SeatGeek", "asset": "seatgeek", "slot": "drawer_slot_1", "show_label": False},
                {"label": "KKBOX", "asset": "kkbox", "slot": "drawer_slot_2", "show_label": False},
                {"label": "Novelship", "asset": "novelship", "slot": "drawer_slot_3", "show_label": True},
                {"label": "Citymapper", "asset": "citymapper", "slot": "drawer_slot_4", "show_label": True},
            ]
        },
    },
}


def _try_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


FONT_SM = _try_font(26)
FONT_MD = _try_font(34)
FONT_LG = _try_font(52, bold=True)
FONT_DATE = _try_font(64, bold=True)
FONT_XL = _try_font(92, bold=True)


def _safe_key(label: str) -> str:
    key = re.sub(r"[^0-9a-zA-Z]+", "_", label).strip("_")
    return key or "item"


def _unique_layout_key(layout: Dict[str, dict], label: str) -> str:
    base_key = _safe_key(label)
    if base_key not in layout:
        return base_key
    suffix = 2
    while f"{base_key}_{suffix}" in layout:
        suffix += 1
    return f"{base_key}_{suffix}"


def _bbox_center(bbox: List[int]) -> List[int]:
    x1, y1, x2, y2 = bbox
    return [int(round((x1 + x2) / 2.0)), int(round((y1 + y2) / 2.0))]


def _merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_layout_config(config_path: Path | None) -> dict:
    config = deepcopy(DEFAULT_LAYOUT_CONFIG)
    if config_path and config_path.exists():
        override = json.loads(config_path.read_text(encoding="utf-8"))
        if isinstance(override, dict):
            config = _merge_dict(config, override)
    return config


def _build_rng(layout_config: dict) -> random.Random:
    metadata = layout_config.get("metadata", {})
    seed = metadata.get("random_seed")
    return random.Random(seed)


def _resolve_icon_position(spec: dict, slots: Dict[str, dict]) -> Tuple[int, int]:
    position = spec.get("position") if isinstance(spec.get("position"), dict) else {}
    slot_name = spec.get("slot") or position.get("slot")
    if slot_name:
        slot = slots.get(str(slot_name))
        if not isinstance(slot, dict):
            raise ValueError(f"Unknown slot: {slot_name}")
        return int(slot["x"]), int(slot["y"])

    x = spec.get("x", position.get("x"))
    y = spec.get("y", position.get("y"))
    if x is None or y is None:
        raise ValueError(f"Icon spec must provide slot or x/y: {spec}")
    return int(x), int(y)


def _new_canvas(bg_color: Tuple[int, int, int]) -> Image.Image:
    canvas = Image.new("RGB", CANVAS_SIZE, bg_color)
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, 0, CANVAS_SIZE[0], VISIBLE_VIEWPORT_HEIGHT], fill=bg_color)
    return canvas


def _blend_color(start: Tuple[int, int, int], end: Tuple[int, int, int], t: float) -> Tuple[int, int, int]:
    return tuple(int(round(start[i] + (end[i] - start[i]) * t)) for i in range(3))


def _draw_vertical_gradient(draw: ImageDraw.ImageDraw, top: Tuple[int, int, int], bottom: Tuple[int, int, int]) -> None:
    for y in range(VISIBLE_VIEWPORT_HEIGHT):
        t = y / max(VISIBLE_VIEWPORT_HEIGHT - 1, 1)
        draw.line([(0, y), (CANVAS_SIZE[0], y)], fill=_blend_color(top, bottom, t))


def _draw_home_wallpaper(draw: ImageDraw.ImageDraw) -> None:
    _draw_vertical_gradient(draw, (97, 132, 219), (224, 233, 252))
    draw.ellipse([-120, -120, 420, 320], fill=(133, 166, 241))
    draw.ellipse([650, -40, 1180, 420], fill=(187, 205, 247))
    draw.ellipse([710, 100, 1220, 610], fill=(237, 242, 252))
    draw.polygon([(0, 1660), (230, 1420), (540, 1710), (860, 1380), (1080, 1560), (1080, 2400), (0, 2400)], fill=(210, 221, 244))
    draw.polygon([(0, 1820), (250, 1620), (480, 1870), (760, 1695), (1080, 1900), (1080, 2400), (0, 2400)], fill=(224, 232, 248))


def _draw_app_drawer_background(draw: ImageDraw.ImageDraw) -> None:
    _draw_vertical_gradient(draw, (197, 210, 238), (241, 244, 250))
    draw.ellipse([-180, -60, 360, 420], fill=(172, 191, 234))
    draw.ellipse([760, -90, 1250, 360], fill=(223, 231, 246))
    draw.rounded_rectangle([20, 160, 1060, 2390], radius=72, fill=(248, 249, 252), outline=(224, 228, 237), width=2)


def _draw_chip(draw: ImageDraw.ImageDraw, bbox: List[int], text: str, fill: Tuple[int, int, int], text_fill: Tuple[int, int, int]) -> None:
    draw.rounded_rectangle(bbox, radius=24, fill=fill)
    text_box = draw.textbbox((0, 0), text, font=FONT_SM)
    text_w = text_box[2] - text_box[0]
    text_h = text_box[3] - text_box[1]
    draw.text((bbox[0] + (bbox[2] - bbox[0] - text_w) / 2, bbox[1] + (bbox[3] - bbox[1] - text_h) / 2 - 2), text, fill=text_fill, font=FONT_SM)


def _draw_status_bar(draw: ImageDraw.ImageDraw) -> None:
    text_fill = (18, 22, 30)
    draw.text((SIDE_MARGIN, 18), "9:41", fill=text_fill, font=FONT_MD)
    bar_x = CANVAS_SIZE[0] - 200
    for idx, height in enumerate((10, 14, 18, 22)):
        x = bar_x + idx * 10
        draw.rounded_rectangle([x, 46 - height, x + 6, 46], radius=2, fill=text_fill)
    draw.arc([bar_x + 54, 18, bar_x + 92, 56], start=210, end=330, fill=text_fill, width=3)
    draw.arc([bar_x + 60, 24, bar_x + 86, 50], start=215, end=325, fill=text_fill, width=3)
    draw.ellipse([bar_x + 70, 40, bar_x + 76, 46], fill=text_fill)
    battery = [CANVAS_SIZE[0] - 86, 22, CANVAS_SIZE[0] - 24, 48]
    draw.rounded_rectangle(battery, radius=7, outline=text_fill, width=3)
    draw.rectangle([battery[2], battery[1] + 7, battery[2] + 6, battery[3] - 7], fill=text_fill)
    draw.rounded_rectangle([battery[0] + 4, battery[1] + 4, battery[0] + 38, battery[3] - 4], radius=4, fill=text_fill)


def _draw_system_buttons(draw: ImageDraw.ImageDraw, layout: Dict[str, dict]) -> None:
    button_top = TOP_MARGIN + STATUS_BAR_HEIGHT + SYSTEM_BUTTON_Y_OFFSET
    back_bbox = [SIDE_MARGIN, button_top, SIDE_MARGIN + SYSTEM_BUTTON_W, button_top + SYSTEM_BUTTON_H]
    home_bbox = [
        CANVAS_SIZE[0] - SIDE_MARGIN - SYSTEM_BUTTON_W,
        button_top,
        CANVAS_SIZE[0] - SIDE_MARGIN,
        button_top + SYSTEM_BUTTON_H,
    ]
    draw.rounded_rectangle(back_bbox, radius=28, fill=(255, 205, 205), outline=(220, 154, 154), width=3)
    draw.rounded_rectangle(home_bbox, radius=28, fill=(205, 247, 205), outline=(136, 191, 136), width=3)

    for label, bbox in (("back", back_bbox), ("home", home_bbox)):
        text_box = draw.textbbox((0, 0), label, font=FONT_MD)
        text_w = text_box[2] - text_box[0]
        text_h = text_box[3] - text_box[1]
        text_x = int((bbox[0] + bbox[2] - text_w) / 2)
        text_y = int((bbox[1] + bbox[3] - text_h) / 2) - 2
        draw.text((text_x, text_y), label, fill=(32, 36, 44), font=FONT_MD)
    layout["back"] = {"bbox": back_bbox, "type": "system"}
    layout["home"] = {"bbox": home_bbox, "type": "system"}


def _draw_search_bar(draw: ImageDraw.ImageDraw, y: int, placeholder: str) -> List[int]:
    bbox = [SIDE_MARGIN, y, CANVAS_SIZE[0] - SIDE_MARGIN, y + SEARCH_BAR_H]
    draw.rounded_rectangle(bbox, radius=48, fill=(255, 255, 255), outline=(223, 228, 236), width=2)
    draw.ellipse([bbox[0] + 30, y + 32, bbox[0] + 64, y + 66], outline=(132, 138, 150), width=4)
    draw.line([bbox[0] + 57, y + 58, bbox[0] + 76, y + 77], fill=(132, 138, 150), width=4)
    draw.text((bbox[0] + 106, y + 28), placeholder, fill=(109, 115, 127), font=FONT_MD)
    return bbox


def _placeholder_icon(label: str, fill: Tuple[int, int, int], size: int) -> Image.Image:
    icon = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    icon_draw = ImageDraw.Draw(icon)
    icon_draw.rounded_rectangle([0, 0, size - 1, size - 1], radius=34, fill=fill)
    initials = "".join(part[0] for part in label.split()[:2]).upper() or "A"
    text_box = icon_draw.textbbox((0, 0), initials, font=FONT_LG)
    text_w = text_box[2] - text_box[0]
    text_h = text_box[3] - text_box[1]
    icon_draw.text(
        ((size - text_w) / 2, (size - text_h) / 2 - 6),
        initials,
        fill=(255, 255, 255),
        font=FONT_LG,
    )
    return icon


def _load_real_icon(path: Path, size: int) -> Image.Image:
    if not path.exists():
        return _placeholder_icon(path.stem, (108, 132, 255), size)
    icon = Image.open(path).convert("RGBA")
    fitted = ImageOps.contain(icon, (size, size))
    tile = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    offset = ((size - fitted.width) // 2, (size - fitted.height) // 2)
    tile.paste(fitted, offset, fitted)
    return tile


def _icon_image_for_asset(asset_name: str, label: str, size: int) -> Image.Image:
    normalized = str(asset_name or "").strip().lower()
    placeholder_colors = {
        "maps": (72, 192, 124),
        "camera": (102, 102, 118),
        "chat": (255, 178, 87),
        "files": (119, 184, 255),
        "clock": (94, 108, 245),
        "photos": (233, 118, 168),
        "phone": (76, 190, 124),
        "chrome": (76, 126, 250),
        "messages": (92, 214, 167),
    }

    if normalized == "eventbrite":
        return _load_real_icon(EVENTBRITE_ICON_PATH, size)
    if normalized == "seatgeek":
        return _load_real_icon(SEATGEEK_ICON_PATH, size)
    if normalized in REAL_ICON_LIBRARY:
        return _load_real_icon(REAL_ICON_LIBRARY[normalized]["path"], size)
    return _placeholder_icon(label, placeholder_colors.get(normalized, (108, 132, 255)), size)


def _resolve_icon_asset(
    asset_name: str,
    label: str,
    size: int,
    render_state: Dict[str, Any],
) -> Tuple[Image.Image, str, str]:
    normalized = str(asset_name or "").strip().lower()
    final_label = label
    resolved_asset = normalized
    canonical_asset = CANONICAL_REAL_ICON_ASSETS.get(normalized, normalized)

    if normalized == "random_home_real":
        candidates = [
            asset
            for asset in RANDOM_HOME_REAL_ASSETS
            if CANONICAL_REAL_ICON_ASSETS.get(asset, asset) not in render_state["used_random_assets"]
        ]
        if not candidates:
            render_state["used_random_assets"].clear()
            candidates = list(RANDOM_HOME_REAL_ASSETS)
        resolved_asset = render_state["rng"].choice(candidates)
        render_state["used_random_assets"].add(CANONICAL_REAL_ICON_ASSETS.get(resolved_asset, resolved_asset))
        final_label = REAL_ICON_LIBRARY[resolved_asset]["label"]
        return _icon_image_for_asset(resolved_asset, final_label, size), final_label, resolved_asset

    if normalized in REAL_ICON_LIBRARY:
        render_state["used_random_assets"].add(canonical_asset)
        final_label = label or REAL_ICON_LIBRARY[normalized]["label"]
    return _icon_image_for_asset(normalized, final_label or label, size), (final_label or label), resolved_asset


def _draw_icon_with_label(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    label: str,
    icon_x: int,
    icon_y: int,
    icon_image: Image.Image,
    layout: Dict[str, dict],
    layout_key: str | None = None,
    label_fill: Tuple[int, int, int] = (26, 30, 38),
    icon_size: int = ICON_TILE_SIZE,
    show_label: bool = True,
) -> None:
    fitted = ImageOps.contain(icon_image.convert("RGBA"), (icon_size, icon_size))
    tile = Image.new("RGBA", (icon_size, icon_size), (0, 0, 0, 0))
    offset = ((icon_size - fitted.width) // 2, (icon_size - fitted.height) // 2)
    tile.paste(fitted, offset, fitted)
    canvas.paste(tile, (icon_x, icon_y), tile)

    key = _unique_layout_key(layout, layout_key or label or "icon")
    layout[key] = {"bbox": [icon_x, icon_y, icon_x + icon_size, icon_y + icon_size], "type": "normal"}

    if show_label and label:
        display_label = label if len(label) <= 10 else f"{label[:9]}..."
        text_box = draw.textbbox((0, 0), display_label, font=FONT_SM)
        text_w = text_box[2] - text_box[0]
        label_x = int(icon_x + (icon_size - text_w) / 2)
        label_y = icon_y + icon_size + APP_LABEL_GAP
        draw.text((label_x, label_y), display_label, fill=label_fill, font=FONT_SM)


def _draw_configured_icons(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    icon_specs: List[dict],
    slots: Dict[str, dict],
    layout: Dict[str, dict],
    render_state: Dict[str, Any],
    page_name: str,
) -> None:
    for idx, spec in enumerate(icon_specs):
        if not isinstance(spec, dict):
            continue
        label = str(spec.get("label") or "")
        asset = str(spec.get("asset") or label or f"icon_{idx}")
        icon_size = int(spec.get("size", ICON_TILE_SIZE))
        icon_x, icon_y = _resolve_icon_position(spec, slots)
        icon_image, resolved_label, resolved_asset = _resolve_icon_asset(asset, label, icon_size, render_state)
        _draw_icon_with_label(
            canvas,
            draw,
            label=resolved_label,
            icon_x=icon_x,
            icon_y=icon_y,
            icon_image=icon_image,
            layout=layout,
            layout_key=str(spec.get("layout_key") or resolved_label or f"icon_{idx}"),
            icon_size=icon_size,
            show_label=bool(spec.get("show_label", True)),
        )
        render_state["resolved_icons"].append(
            {
                "page": page_name,
                "label": resolved_label,
                "asset": resolved_asset,
                "x": icon_x,
                "y": icon_y,
                "size": icon_size,
                "slot": spec.get("slot"),
                "source_path": str(REAL_ICON_LIBRARY[resolved_asset]["path"]) if resolved_asset in REAL_ICON_LIBRARY else "",
            }
        )


def _icon_specs_bounds(icon_specs: List[dict], slots: Dict[str, dict]) -> List[int] | None:
    boxes: List[List[int]] = []
    for spec in icon_specs:
        if not isinstance(spec, dict):
            continue
        try:
            icon_x, icon_y = _resolve_icon_position(spec, slots)
        except Exception:
            continue
        icon_size = int(spec.get("size", ICON_TILE_SIZE))
        boxes.append([icon_x, icon_y, icon_x + icon_size, icon_y + icon_size])
    if not boxes:
        return None
    min_x = min(box[0] for box in boxes)
    min_y = min(box[1] for box in boxes)
    max_x = max(box[2] for box in boxes)
    max_y = max(box[3] for box in boxes)
    return [min_x, min_y, max_x, max_y]


def _draw_home_page(layout_config: dict, render_state: Dict[str, Any]) -> Tuple[Image.Image, Dict[str, dict], dict]:
    canvas = _new_canvas((239, 245, 255))
    draw = ImageDraw.Draw(canvas)
    layout: Dict[str, dict] = {}
    slots = layout_config.get("slots", {})
    home_config = layout_config.get("pages", {}).get("home", {})

    _draw_home_wallpaper(draw)
    draw.rounded_rectangle([40, 224, 520, 450], radius=56, fill=(255, 255, 255))
    draw.rounded_rectangle([560, 224, 1040, 450], radius=56, fill=(255, 255, 255))
    draw.text((74, 258), "17", fill=(45, 53, 74), font=FONT_XL)
    draw.text((232, 274), "Wed", fill=(64, 72, 92), font=FONT_DATE)
    draw.text((596, 270), "18°C", fill=(45, 53, 74), font=FONT_XL)

    _draw_status_bar(draw)
    _draw_system_buttons(draw, layout)
    _draw_configured_icons(canvas, draw, home_config.get("icons", []), slots, layout, render_state, "home")

    dock_icons = home_config.get("dock_icons", [])
    dock_bounds = _icon_specs_bounds(dock_icons, slots)
    if dock_bounds is None:
        dock_bbox = [94, 1934, 960, 2226]
    else:
        min_x, min_y, max_x, max_y = dock_bounds
        dock_bbox = [
            max(72, min_x - 110),
            max(1860, min_y - 34),
            min(CANVAS_SIZE[0] - 72, max_x + 110),
            min(VISIBLE_VIEWPORT_HEIGHT - 140, max_y + 116),
        ]
    draw.rounded_rectangle(dock_bbox, radius=98, fill=(249, 250, 253))
    _draw_configured_icons(canvas, draw, dock_icons, slots, layout, render_state, "home_dock")

    swipe_zone = [420, 2326, 660, 2360]
    draw.rounded_rectangle(swipe_zone, radius=18, fill=(42, 46, 56))
    layout["drawer_handle"] = {"bbox": swipe_zone, "type": "normal"}

    swipe_transition = {
        "action": "swipe",
        "target_page": DRAWER_PAGE_ID,
        "action_coord": [540, 2343],
        "lift_coord": [540, 1460],
        "icon_bbox": [500, 1460, 580, 2343],
        "gesture_direction": "up",
    }
    return canvas, layout, swipe_transition


def _draw_app_drawer_page(layout_config: dict, render_state: Dict[str, Any]) -> Tuple[Image.Image, Dict[str, dict]]:
    canvas = _new_canvas((246, 247, 251))
    draw = ImageDraw.Draw(canvas)
    layout: Dict[str, dict] = {}
    slots = layout_config.get("slots", {})
    drawer_config = layout_config.get("pages", {}).get("app_drawer", {})

    _draw_app_drawer_background(draw)
    _draw_status_bar(draw)
    _draw_system_buttons(draw, layout)
    layout["app_drawer_search"] = {"bbox": _draw_search_bar(draw, 220, "Search apps"), "type": "normal"}
    draw.text((SIDE_MARGIN, 402), "All apps", fill=(28, 31, 39), font=FONT_LG)
    drawer_icons = drawer_config.get("icons", [])
    _draw_configured_icons(canvas, draw, drawer_icons, slots, layout, render_state, "app_drawer")

    draw.rounded_rectangle([446, 2340, 634, 2374], radius=18, fill=(74, 81, 94))
    layout["drawer_handle"] = {"bbox": [446, 2340, 634, 2374], "type": "normal"}

    return canvas, layout


def _build_ui_structure(home_layout: Dict[str, dict], drawer_layout: Dict[str, dict], swipe_transition: dict) -> dict:
    home_button_bbox = home_layout["home"]["bbox"]
    back_button_bbox = drawer_layout["back"]["bbox"]
    drawer_home_bbox = drawer_layout["home"]["bbox"]
    return {
        "pages": {
            HOME_PAGE_ID: {
                "image": f"{HOME_PAGE_ID}.png",
                "depth": 0,
                "layout": home_layout,
                "transitions": [
                    swipe_transition,
                    {
                        "action": "PRESS_HOME",
                        "target_page": HOME_PAGE_ID,
                        "action_coord": _bbox_center(home_button_bbox),
                        "icon_bbox": home_button_bbox,
                    },
                ],
            },
            DRAWER_PAGE_ID: {
                "image": f"{DRAWER_PAGE_ID}.png",
                "depth": 1,
                "layout": drawer_layout,
                "transitions": [
                    {
                        "action": "PRESS_BACK",
                        "target_page": HOME_PAGE_ID,
                        "action_coord": _bbox_center(back_button_bbox),
                        "icon_bbox": back_button_bbox,
                    },
                    {
                        "action": "PRESS_HOME",
                        "target_page": HOME_PAGE_ID,
                        "action_coord": _bbox_center(drawer_home_bbox),
                        "icon_bbox": drawer_home_bbox,
                    },
                    {
                        "action": "swipe",
                        "target_page": HOME_PAGE_ID,
                        "action_coord": [540, 1460],
                        "lift_coord": [540, 2343],
                        "icon_bbox": [500, 1460, 580, 2343],
                        "gesture_direction": "down",
                    },
                ],
            },
        },
        "metadata": {
            "source": "mock_simulator",
            "canvas_size": list(CANVAS_SIZE),
            "visible_viewport_height": VISIBLE_VIEWPORT_HEIGHT,
            "total_pages": 2,
            "icon_sources": {
                "eventbrite": str(EVENTBRITE_ICON_PATH),
                "seatgeek": str(SEATGEEK_ICON_PATH),
            },
        },
    }


def _build_ui_layer(ui_structure: dict) -> dict:
    home_page = ui_structure["pages"][HOME_PAGE_ID]
    drawer_page = ui_structure["pages"][DRAWER_PAGE_ID]
    non_system_home = [t for t in home_page["transitions"] if t["action"] not in ("PRESS_BACK", "PRESS_HOME")]
    non_system_drawer = [t for t in drawer_page["transitions"] if t["action"] not in ("PRESS_BACK", "PRESS_HOME")]
    return {
        "root": {
            "image": home_page["image"],
            "depth": home_page["depth"],
            "layout": home_page["layout"],
            "transitions": non_system_home,
            "subnodes": [
                {
                    "image": drawer_page["image"],
                    "depth": drawer_page["depth"],
                    "layout": drawer_page["layout"],
                    "transitions": non_system_drawer,
                    "subnodes": [],
                }
            ],
        },
        "metadata": {
            "type": "trajectory",
            "canvas_size": list(CANVAS_SIZE),
            "visible_viewport_height": VISIBLE_VIEWPORT_HEIGHT,
        },
    }


def _save_debug_overlay(page_image_path: Path, transitions: List[dict], output_path: Path) -> None:
    image = Image.open(page_image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    palette = [
        (255, 99, 132),
        (84, 192, 118),
        (72, 147, 255),
        (255, 170, 64),
    ]
    for idx, transition in enumerate(transitions):
        color = palette[idx % len(palette)]
        icon_bbox = transition.get("icon_bbox")
        if isinstance(icon_bbox, list) and len(icon_bbox) == 4:
            draw.rectangle(icon_bbox, outline=color, width=6)
        action_coord = transition.get("action_coord")
        lift_coord = transition.get("lift_coord")
        if isinstance(action_coord, list) and len(action_coord) == 2:
            x, y = action_coord
            draw.ellipse([x - 16, y - 16, x + 16, y + 16], fill=color)
        if isinstance(action_coord, list) and len(action_coord) == 2 and isinstance(lift_coord, list) and len(lift_coord) == 2:
            draw.line([tuple(action_coord), tuple(lift_coord)], fill=color, width=10)
        label = transition["action"]
        draw.text((60, 2280 + idx * 40), label, fill=color, font=FONT_SM)
    image.save(output_path)


def build_mock_environment(output_dir: Path, layout_config: dict) -> None:
    pages_dir = output_dir / "pages"
    debug_dir = output_dir / "action_coord_debug"
    pages_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    render_state: Dict[str, Any] = {
        "rng": _build_rng(layout_config),
        "used_random_assets": set(),
        "resolved_icons": [],
    }

    home_img, home_layout, swipe_transition = _draw_home_page(layout_config, render_state)
    drawer_img, drawer_layout = _draw_app_drawer_page(layout_config, render_state)

    home_path = pages_dir / f"{HOME_PAGE_ID}.png"
    drawer_path = pages_dir / f"{DRAWER_PAGE_ID}.png"
    home_img.save(home_path)
    drawer_img.save(drawer_path)

    ui_structure = _build_ui_structure(home_layout, drawer_layout, swipe_transition)
    ui_layer = _build_ui_layer(ui_structure)

    (output_dir / "ui_structure.json").write_text(json.dumps(ui_structure, indent=2), encoding="utf-8")
    (output_dir / "ui_structure_layer.json").write_text(json.dumps(ui_layer, indent=2), encoding="utf-8")
    layout_config_used = deepcopy(layout_config)
    layout_config_used["resolved_icons"] = render_state["resolved_icons"]
    (output_dir / "layout_config_used.json").write_text(json.dumps(layout_config_used, indent=2), encoding="utf-8")

    _save_debug_overlay(home_path, ui_structure["pages"][HOME_PAGE_ID]["transitions"], debug_dir / f"{HOME_PAGE_ID}.png")
    _save_debug_overlay(drawer_path, ui_structure["pages"][DRAWER_PAGE_ID]["transitions"], debug_dir / f"{DRAWER_PAGE_ID}.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a mock home/app-drawer environment with PIL.")
    parser.add_argument(
        "--output_dir",
        default="data_engine/mock_simulator",
        help="Directory where pages and ui_structure files will be saved.",
    )
    parser.add_argument(
        "--layout_config",
        default=str(DEFAULT_LAYOUT_CONFIG_PATH),
        help="Optional JSON config for icon slots or direct x/y placement.",
    )
    args = parser.parse_args()
    config_path = Path(args.layout_config) if args.layout_config else None
    layout_config = _load_layout_config(config_path)
    build_mock_environment(Path(args.output_dir), layout_config)


if __name__ == "__main__":
    main()
