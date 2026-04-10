#!/usr/bin/env python3
"""
Pipeline Visualizer — shows how the unified graph builder constructs pages step by step.

For ONE trajectory, saves intermediate images at each pipeline stage:
  - Launcher: wallpaper → widgets → status bar → system buttons → icons → dock → final
  - App drawer: background → status bar → search bar → icons → final
  - App pages: original screenshot → YOLO+OCR detection overlay → blur background
              → elements pasted one-by-one → final composed page

Usage (conda run -n gelab):
    python pipeline_visualizer.py \
        --annotations_dir /ext_hdd2/tsyou/AMEX_dataset/AMEX/instruction_anno \
        --screenshots_dir /ext_hdd2/tsyou/AMEX_dataset/AMEX/screenshot \
        --element_anno_dir /ext_hdd2/tsyou/AMEX_dataset/AMEX/element_anno \
        --weights_dir /ext_hdd2/nhkoh/OmniParser/weights \
        --output_dir /ext_hdd2/tsyou/pipeline_viz
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Tuple

from PIL import Image, ImageDraw

import mock_unified_app_graph as G

CANVAS_SIZE = G.CANVAS_SIZE
VISIBLE_VIEWPORT_HEIGHT = G.VISIBLE_VIEWPORT_HEIGHT


# ---------------------------------------------------------------------------
# Step counter for ordering saved images
# ---------------------------------------------------------------------------
class StepSaver:
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.counter = 0

    def save(self, image: Image.Image, label: str, subfolder: str = ""):
        self.counter += 1
        folder = self.output_dir / subfolder if subfolder else self.output_dir
        folder.mkdir(parents=True, exist_ok=True)
        fname = f"step_{self.counter:03d}_{label}.png"
        image.save(folder / fname)
        print(f"  [step {self.counter:03d}] {subfolder + '/' if subfolder else ''}{fname}")
        return folder / fname


# ---------------------------------------------------------------------------
# Launcher page: instrument _draw_home_page to save each layer
# ---------------------------------------------------------------------------
def visualize_home_page(layout_config: dict, render_state: dict, saver: StepSaver) -> Tuple[Image.Image, dict, dict]:
    """Draw home page step-by-step, saving intermediate images."""
    canvas = G._new_canvas((239, 245, 255))
    draw = ImageDraw.Draw(canvas)
    layout: Dict[str, dict] = {}
    slots = layout_config.get("slots", {})
    home_config = layout_config.get("pages", {}).get("home", {})

    # Step 1: blank canvas
    saver.save(canvas.copy(), "home_01_blank_canvas", "home")

    # Step 2: wallpaper
    G._draw_home_wallpaper(draw)
    saver.save(canvas.copy(), "home_02_wallpaper", "home")

    # Step 3: widgets
    draw.rounded_rectangle([40, 224, 520, 450], radius=56, fill=(255, 255, 255))
    draw.rounded_rectangle([560, 224, 1040, 450], radius=56, fill=(255, 255, 255))
    draw.text((74, 258), "17", fill=(45, 53, 74), font=G.FONT_XL)
    draw.text((232, 274), "Wed", fill=(64, 72, 92), font=G.FONT_DATE)
    draw.text((596, 270), "18°C", fill=(45, 53, 74), font=G.FONT_XL)
    saver.save(canvas.copy(), "home_03_widgets", "home")

    # Step 4: status bar
    G._draw_status_bar(draw)
    saver.save(canvas.copy(), "home_04_status_bar", "home")

    # Step 5: system buttons
    G._draw_system_buttons(draw, layout)
    saver.save(canvas.copy(), "home_05_system_buttons", "home")

    # Step 6: app icons
    G._draw_configured_icons(canvas, draw, home_config.get("icons", []), slots, layout, render_state, "home")
    saver.save(canvas.copy(), "home_06_app_icons", "home")

    # Step 7: dock
    dock_icons = home_config.get("dock_icons", [])
    dock_bounds = G._icon_specs_bounds(dock_icons, slots)
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
    G._draw_configured_icons(canvas, draw, dock_icons, slots, layout, render_state, "home_dock")
    saver.save(canvas.copy(), "home_07_dock", "home")

    # Step 8: swipe handle (final)
    swipe_zone = [420, 2326, 660, 2360]
    draw.rounded_rectangle(swipe_zone, radius=18, fill=(42, 46, 56))
    layout["drawer_handle"] = {"bbox": swipe_zone, "type": "normal"}
    saver.save(canvas.copy(), "home_08_final", "home")

    swipe_transition = G._build_swipe_transition_mock(
        G.drawer_page_id(0),
        action_coord=[540, 2343],
        lift_coord=[540, 1460],
        direction="up",
        icon_bbox=[500, 1460, 580, 2343],
    )
    return canvas, layout, swipe_transition


def visualize_app_drawer_page(layout_config: dict, render_state: dict,
                              saver: StepSaver, page_index: int = 0,
                              drawer_icons=None, total_pages=None, page_id=None):
    """Draw app drawer page step-by-step, saving intermediate images."""
    canvas = G._new_canvas((246, 247, 251))
    draw = ImageDraw.Draw(canvas)
    layout: Dict[str, dict] = {}
    slots = layout_config.get("slots", {})
    drawer_pages = G._chunk_app_drawer_icons(layout_config)
    resolved_total_pages = int(total_pages) if total_pages is not None else len(drawer_pages)
    resolved_page_icons = drawer_icons if drawer_icons is not None else drawer_pages[min(page_index, len(drawer_pages) - 1)]
    resolved_page_icons = G._normalize_drawer_page_icons(resolved_page_icons, slots)
    resolved_page_id = page_id or G.drawer_page_id(page_index)

    sub = f"app_drawer_p{page_index}"

    # Step 1: blank canvas
    saver.save(canvas.copy(), f"drawer_p{page_index}_01_blank", sub)

    # Step 2: background
    G._draw_app_drawer_background(draw)
    saver.save(canvas.copy(), f"drawer_p{page_index}_02_background", sub)

    # Step 3: status bar
    G._draw_status_bar(draw)
    saver.save(canvas.copy(), f"drawer_p{page_index}_03_status_bar", sub)

    # Step 4: system buttons
    G._draw_system_buttons(draw, layout)
    saver.save(canvas.copy(), f"drawer_p{page_index}_04_system_buttons", sub)

    # Step 5: search bar + header
    layout["app_drawer_search"] = {"bbox": G._draw_search_bar(draw, 220, "Search apps"), "type": "normal"}
    draw.text((G.SIDE_MARGIN, 402), "All apps", fill=(28, 31, 39), font=G.FONT_LG)
    if resolved_total_pages > 1:
        indicator_text = f"{page_index + 1}/{resolved_total_pages}"
        text_box = draw.textbbox((0, 0), indicator_text, font=G.FONT_SM)
        text_w = text_box[2] - text_box[0]
        draw.text((CANVAS_SIZE[0] - G.SIDE_MARGIN - text_w, 416), indicator_text, fill=(90, 98, 112), font=G.FONT_SM)
        G._draw_page_indicator(draw, page_index, resolved_total_pages)
    saver.save(canvas.copy(), f"drawer_p{page_index}_05_search_header", sub)

    # Step 6: icons
    G._draw_configured_icons(canvas, draw, resolved_page_icons, slots, layout, render_state, resolved_page_id)
    saver.save(canvas.copy(), f"drawer_p{page_index}_06_icons", sub)

    # Step 7: handle (final)
    draw.rounded_rectangle([446, 2340, 634, 2374], radius=18, fill=(74, 81, 94))
    layout["drawer_handle"] = {"bbox": [446, 2340, 634, 2374], "type": "normal"}
    saver.save(canvas.copy(), f"drawer_p{page_index}_07_final", sub)

    return canvas, layout


# ---------------------------------------------------------------------------
# App page: GPT compose pipeline with intermediate saves
# ---------------------------------------------------------------------------
def visualize_app_page_gpt(screenshot_path: str, elements: List[dict],
                           orig_size: Tuple[int, int], saver: StepSaver,
                           page_label: str, client, model_name: str,
                           step_info: dict = None) -> Tuple[Image.Image, dict, dict]:
    """Compose an app page via GPT styling, saving intermediate images."""
    ow, oh = orig_size
    sub = f"app_{page_label}"

    # Step 1: original screenshot
    with Image.open(screenshot_path) as img_handle:
        screenshot = img_handle.convert("RGB")
    saver.save(screenshot.copy(), f"{page_label}_01_original_screenshot", sub)

    # Step 2: detection overlay — draw bboxes on original screenshot
    overlay = screenshot.copy()
    draw_ov = ImageDraw.Draw(overlay)
    colors = [(255, 0, 0), (0, 200, 0), (0, 0, 255), (255, 165, 0), (128, 0, 128)]
    for i, elem in enumerate(elements):
        bbox = elem.get("bbox") or []
        if len(bbox) != 4:
            continue
        color = colors[i % len(colors)]
        draw_ov.rectangle(bbox, outline=color, width=3)
        label_text = elem.get("label", f"elem_{i}")[:20]
        try:
            draw_ov.text((bbox[0] + 2, bbox[1] + 2), label_text, fill=color, font=G.FONT_SM)
        except Exception:
            draw_ov.text((bbox[0] + 2, bbox[1] + 2), label_text, fill=color)
    saver.save(overlay, f"{page_label}_02_detection_overlay", sub)

    # Step 3: GPT generates styling code
    styling_source = "gpt"
    styling_code = None
    if client is not None:
        styling_code = G._ac_generate_styling_code(
            client, model_name, screenshot_path,
            elements, orig_size, step_info,
        )
    else:
        styling_source = "fallback_bg_no_client"

    if styling_code is None:
        bg = G._ac_extract_bg_color(elements, screenshot_path)
        styling_code = f"draw.rectangle([0, 0, {ow}, {oh}], fill={bg})"
        if client is not None:
            styling_source = "fallback_bg"

    # Save the styling code as text file for reference
    code_file = saver.output_dir / sub / f"{page_label}_03_styling_code.py"
    (saver.output_dir / sub).mkdir(parents=True, exist_ok=True)
    code_file.write_text(
        f"# styling_source: {styling_source}\n"
        f"# orig_size: {ow}x{oh}\n\n"
        f"{styling_code}\n",
        encoding="utf-8",
    )
    print(f"  [code] {sub}/{page_label}_03_styling_code.py ({styling_source})")

    # Step 4: render styling-only canvas (background without elements)
    styling_only_canvas = Image.new("RGB", (ow, oh), G._AC_BG_WHITE)
    styling_only_draw = ImageDraw.Draw(styling_only_canvas)
    try:
        # Execute only the styling code (no element pastes) to see the background
        _exec_styling_only(styling_code, styling_only_canvas, styling_only_draw, elements, (ow, oh))
    except Exception as e:
        print(f"    styling-only exec error: {e}")
    saver.save(styling_only_canvas.copy(), f"{page_label}_04_styling_bg_only", sub)

    # Step 5: generate position code
    position_code = G._ac_generate_position_code(elements, orig_size)
    full_code = styling_code + "\n\n" + position_code

    # Step 6: render full code (styling + element pastes)
    render_status = "render_from_code"
    page_img, layout = G._ac_render_from_code(full_code, elements, orig_size)

    if page_img is None:
        page_img, layout = G._ac_fallback_compose(elements, orig_size, screenshot_path)
        render_status = "fallback_compose"

    saver.save(page_img.copy(), f"{page_label}_05_rendered_page_raw", sub)

    # Step 7: place on final canvas with nav controls
    final_canvas = Image.new("RGB", CANVAS_SIZE, G._AC_BG_WHITE)
    final_canvas.paste(page_img.convert("RGB"), (G._AC_PHONE_OFFSET_X, G._AC_PHONE_OFFSET_Y))
    saver.save(final_canvas.copy(), f"{page_label}_06_on_canvas", sub)

    shifted_layout = {}
    for key, bbox in (layout or {}).items():
        shifted_layout[key] = [
            int(bbox[0]) + G._AC_PHONE_OFFSET_X,
            int(bbox[1]) + G._AC_PHONE_OFFSET_Y,
            int(bbox[2]) + G._AC_PHONE_OFFSET_X,
            int(bbox[3]) + G._AC_PHONE_OFFSET_Y,
        ]
    final_canvas, shifted_layout = G._ac_ensure_system_nav_controls(final_canvas, shifted_layout)
    saver.save(final_canvas.copy(), f"{page_label}_07_final_with_nav", sub)

    code_artifact = {
        "styling_source": styling_source,
        "render_status": render_status,
        "styling_code": styling_code,
        "position_code": position_code,
        "full_code": full_code,
    }
    return final_canvas, shifted_layout, code_artifact


def _exec_styling_only(styling_code: str, canvas: Image.Image, draw: ImageDraw.ImageDraw,
                       elements: List[dict], orig_size: Tuple[int, int]):
    """Execute only the GPT styling code to produce the background canvas."""
    import random as _random
    font_sm = G._ac_try_load_font(12)
    font_md = G._ac_try_load_font(18)
    font_lg = G._ac_try_load_font(24)
    font_xl = G._ac_try_load_font(32)

    def get_crop(index, w=50, h=50):
        # Return transparent placeholder — we don't want element pastes
        return Image.new("RGBA", (int(w), int(h)), (0, 0, 0, 0))

    real_paste = canvas.paste
    def _safe_paste(im, box=None, mask=None):
        if isinstance(box, (tuple, list)):
            box = tuple(int(v) for v in box)
        if mask is not None:
            real_paste(im, box, mask)
        else:
            real_paste(im, box)
    canvas.paste = _safe_paste

    namespace = {
        "__builtins__": {},
        "canvas": canvas, "draw": draw, "layout": {},
        "font_sm": font_sm, "font_md": font_md, "font_lg": font_lg, "font_xl": font_xl,
        "get_crop": get_crop,
        "Image": Image, "ImageDraw": ImageDraw,
        "range": range, "len": len, "enumerate": enumerate,
        "min": min, "max": max, "int": int, "float": float, "str": str,
        "True": True, "False": False, "None": None,
        "list": list, "tuple": tuple, "dict": dict, "abs": abs, "round": round,
        "zip": zip, "print": print, "getattr": getattr, "setattr": setattr,
        "hasattr": hasattr, "isinstance": isinstance, "type": type, "bool": bool,
        "set": set, "sorted": sorted, "reversed": reversed, "map": map,
        "filter": filter, "sum": sum, "any": any, "all": all,
        "ord": ord, "chr": chr,
        "TypeError": TypeError, "ValueError": ValueError, "Exception": Exception,
        "KeyError": KeyError, "IndexError": IndexError,
        "random": _random,
    }
    sanitized = G._ac_sanitize_code(styling_code)
    exec(sanitized, namespace)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def run_visualization(args):
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saver = StepSaver(str(output_dir))

    print("=" * 60)
    print("Pipeline Visualizer — intermediate step saver")
    print("=" * 60)

    # --- Phase 1: Launcher ---
    print("\n[Phase 1] Launcher pages")
    annotations_dir = Path(args.annotations_dir) if args.annotations_dir else None
    layout_config = G._load_layout_config(
        Path(args.layout_config),
        annotations_dir=annotations_dir,
    )
    render_state: Dict[str, Any] = {
        "rng": G._build_rng(layout_config),
        "used_random_assets": set(),
        "resolved_icons": [],
    }

    # Home page with intermediate steps
    print("\n  Drawing home page...")
    home_img, home_layout, swipe_transition = visualize_home_page(layout_config, render_state, saver)

    # App drawer pages with intermediate steps
    print("\n  Drawing app drawer pages...")
    drawer_icon_pages = G._chunk_app_drawer_icons(layout_config)
    total_pages = len(drawer_icon_pages)
    slots = layout_config.get("slots", {})
    drawer_pages = []
    # Only visualize first 2 drawer pages to keep output manageable
    for page_index in range(min(total_pages, 2)):
        page_icons = drawer_icon_pages[page_index]
        normalized = G._normalize_drawer_page_icons(page_icons, slots)
        page_id = G.drawer_page_id(page_index)
        print(f"\n  Drawing app drawer page {page_index}...")
        img, layout = visualize_app_drawer_page(
            layout_config, render_state, saver,
            page_index=page_index,
            drawer_icons=normalized,
            total_pages=total_pages,
            page_id=page_id,
        )
        drawer_pages.append({
            "page_id": page_id,
            "image": img,
            "layout": layout,
            "icons": normalized,
            "depth": page_index + 1,
        })

    # --- Phase 2: Find one trajectory to compose ---
    print("\n[Phase 2] Scanning for one trajectory")
    launcher_bundle = G._render_launcher_bundle(layout_config)
    launcher_pages = {}
    for page_id, page in launcher_bundle["ui_structure"]["pages"].items():
        launcher_pages[page_id] = {
            "page_id": page_id,
            "image": page["image"],
            "depth": int(page["depth"]),
            "layout": G._typed_layout_to_plain(page["layout"]),
            "transitions": deepcopy(page.get("transitions", [])),
        }
    drawer_page_specs = deepcopy(launcher_bundle.get("drawer_page_specs", []))
    drawer_apps = G._extract_drawer_apps(drawer_page_specs, launcher_pages)
    matches_by_app = G._scan_matching_annotations(drawer_apps, args.annotations_dir, False)

    # Find the first app with at least one trajectory
    chosen_app = None
    chosen_match = None
    for app in drawer_apps:
        app_matches = matches_by_app.get(app.slug, [])
        if app_matches:
            chosen_app = app
            chosen_match = app_matches[0]
            break

    if chosen_app is None or chosen_match is None:
        print("No matching trajectories found. Visualization stops after launcher.")
        print(f"\nDone. Output saved to: {output_dir}")
        return

    print(f"  Chosen app: {chosen_app.label}")
    print(f"  Episode: {chosen_match.episode_id}")
    print(f"  Steps: {chosen_match.start_step_idx + 1} - {chosen_match.end_step_idx}")

    # --- Phase 3: Detection & GPT Compose for app pages ---
    print("\n[Phase 3] Loading detection models + GPT client...")
    G._ensure_compose_modules()
    client = G.action_compose.load_api_client()
    model_name = args.model_name
    yolo_model, ocr_reader = G.action_compose.load_detection_models(args.weights_dir, args.gpu)

    if client is None:
        print("  WARNING: No OpenAI client — will use fallback bg color instead of GPT styling")
    else:
        print(f"  Model: {model_name}")

    # Load trajectory
    with open(chosen_match.annotation_path, "r", encoding="utf-8") as f:
        trajectory = json.load(f)
    segment = G._segment_steps(trajectory, chosen_match.start_step_idx, chosen_match.end_step_idx)
    steps = segment.get("steps", []) or []

    # Limit to first 5 steps for brevity
    max_steps = min(len(steps), 5)
    print(f"\n  Processing {max_steps} of {len(steps)} steps")

    assets_dir = output_dir / "extracted_assets" / chosen_app.slug / chosen_match.episode_id
    assets_dir.mkdir(parents=True, exist_ok=True)

    for local_idx in range(max_steps):
        step = steps[local_idx]
        screenshot_name, screenshot_path = G.action_compose._resolve_step_screenshot(
            step, args.screenshots_dir, chosen_match.episode_id, local_idx
        )
        if not os.path.exists(screenshot_path):
            print(f"\n  [{local_idx + 1}/{max_steps}] Missing screenshot: {screenshot_name}")
            continue

        print(f"\n  [{local_idx + 1}/{max_steps}] Processing: {screenshot_name}")

        # Detect elements
        elements, orig_size = G.action_compose.detect_and_crop(screenshot_path, yolo_model, ocr_reader)
        print(f"    Detected {len(elements)} elements (YOLO+OCR)")

        # Prioritize annotation bboxes
        elements, anno_stats = G.action_compose._prioritize_element_anno_bboxes(
            elements,
            screenshot_path,
            screenshot_name,
            args.element_anno_dir,
        )
        print(f"    After annotation priority: {len(elements)} elements, anno_loaded={anno_stats.get('loaded', 0)}")

        # Persist extracted assets
        step_context = G._step_context_for_segment(segment, local_idx)
        asset_elements = G.action_compose._persist_extracted_assets(
            elements, screenshot_name, str(assets_dir), step_context
        )

        # Save individual extracted element crops
        elem_sub = f"app_step{local_idx + 1:02d}_elements"
        elem_folder = output_dir / elem_sub
        elem_folder.mkdir(parents=True, exist_ok=True)
        for ei, elem in enumerate(asset_elements[:20]):  # first 20 elements
            ap = elem.get("asset_path")
            if ap and os.path.exists(ap):
                with Image.open(ap) as crop_img:
                    label_safe = str(elem.get("label", f"elem_{ei}"))[:20].replace("/", "_").replace(" ", "_")
                    crop_img.save(elem_folder / f"elem_{ei:02d}_{label_safe}.png")

        saver.save(
            _make_element_grid(asset_elements, max_items=24),
            f"step{local_idx + 1:02d}_extracted_elements_grid",
            f"app_step{local_idx + 1:02d}",
        )

        # GPT compose pipeline with intermediate saves
        page_label = f"step{local_idx + 1:02d}"
        visualize_app_page_gpt(
            screenshot_path,
            asset_elements,
            orig_size,
            saver,
            page_label,
            client=client,
            model_name=model_name,
            step_info=step_context,
        )

    print(f"\n{'=' * 60}")
    print(f"Done. Total steps saved: {saver.counter}")
    print(f"Output directory: {output_dir}")
    print(f"{'=' * 60}")


def _make_element_grid(elements: List[dict], max_items: int = 24,
                       cell_size: int = 150, cols: int = 6) -> Image.Image:
    """Create a grid image of extracted element crops."""
    items = elements[:max_items]
    rows = (len(items) + cols - 1) // cols
    grid = Image.new("RGB", (cols * cell_size, max(1, rows) * cell_size), (240, 240, 240))
    draw = ImageDraw.Draw(grid)

    for i, elem in enumerate(items):
        row, col = divmod(i, cols)
        cx, cy = col * cell_size, row * cell_size

        # Draw cell border
        draw.rectangle([cx, cy, cx + cell_size - 1, cy + cell_size - 1], outline=(200, 200, 200), width=1)

        # Load crop
        asset_path = elem.get("asset_path")
        crop = None
        if asset_path and os.path.exists(asset_path):
            try:
                crop = Image.open(asset_path).convert("RGBA")
            except Exception:
                pass
        elif "crop" in elem:
            crop = elem["crop"].convert("RGBA")

        if crop is not None:
            # Fit crop into cell with padding
            pad = 8
            max_w, max_h = cell_size - 2 * pad, cell_size - 2 * pad - 16
            ratio = min(max_w / max(crop.width, 1), max_h / max(crop.height, 1), 1.0)
            new_w = max(1, int(crop.width * ratio))
            new_h = max(1, int(crop.height * ratio))
            resized = crop.resize((new_w, new_h), Image.LANCZOS)
            paste_x = cx + pad + (max_w - new_w) // 2
            paste_y = cy + pad + (max_h - new_h) // 2
            grid.paste(resized.convert("RGB"), (paste_x, paste_y))

        # Label
        label = str(elem.get("label", f"#{i}"))[:12]
        try:
            draw.text((cx + 4, cy + cell_size - 14), label, fill=(60, 60, 60), font=G.FONT_SM)
        except Exception:
            draw.text((cx + 4, cy + cell_size - 14), label, fill=(60, 60, 60))

    return grid


def parse_args():
    parser = argparse.ArgumentParser(description="Pipeline step visualizer for unified graph builder")
    parser.add_argument("--layout_config", type=str,
                        default=str(G.DEFAULT_LAYOUT_CONFIG_PATH),
                        help="Mock launcher layout config")
    parser.add_argument("--annotations_dir", type=str,
                        default="/ext_hdd2/tsyou/AMEX_dataset/AMEX/instruction_anno",
                        help="AMEX instruction annotation JSONs directory")
    parser.add_argument("--screenshots_dir", type=str,
                        default="/ext_hdd2/tsyou/AMEX_dataset/AMEX/screenshot",
                        help="AMEX screenshots directory")
    parser.add_argument("--element_anno_dir", type=str,
                        default="/ext_hdd2/tsyou/AMEX_dataset/AMEX/element_anno",
                        help="AMEX element annotation JSONs directory")
    parser.add_argument("--weights_dir", type=str,
                        default="/ext_hdd2/nhkoh/OmniParser/weights",
                        help="OmniParser weights directory")
    parser.add_argument("--output_dir", type=str,
                        default="/ext_hdd2/tsyou/pipeline_viz",
                        help="Output directory for intermediate step images")
    parser.add_argument("--model_name", type=str,
                        default="gpt-5-mini-2025-08-07",
                        help="OpenAI model for page styling composition")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_visualization(args)
