"""
Page Augmentor: Augmentation for mock simulator UI pages.

Two modes:
  [launcher] Home + App Drawer pages
    - Config-based: modifies slot coords in layout_config.json → re-renders
      via _render_launcher_bundle
  [app] In-app pages (e.g. Amazon, BBC News, ...)
    - Image-based: reads ui_structure.json bboxes → crops elements from
      the rendered screenshot → erases → pastes at shifted positions

Augmentation types:
  1. shift   — move element positions (config slots or image bboxes)
  2. bgcolor — change background color
  3. popup   — add popup/dialog overlay

Usage:
    # --- Launcher mode (home + app drawer) ---
    python page_augmentor.py launcher \
        --config mock_simulator_layout_config.json \
        --output_dir /ext_hdd2/tsyou/aug_launcher \
        --augment shift --shift_range 40 --shift_target home

    # --- App mode (in-app pages) ---
    python page_augmentor.py app \
        --env_dir ../amex_all_apps_graph_0330 \
        --output_dir /ext_hdd2/tsyou/aug_app \
        --augment shift --pages page_5,page_10 --shift_range 30

    # App mode: change background + popup
    python page_augmentor.py app \
        --env_dir ../amex_all_apps_graph_0330 \
        --output_dir /ext_hdd2/tsyou/aug_app_bg \
        --augment bgcolor --pages page_5 --bg_color "#1A1A2E"

    python page_augmentor.py app \
        --env_dir ../amex_all_apps_graph_0330 \
        --output_dir /ext_hdd2/tsyou/aug_app_popup \
        --augment popup --pages page_5 --popup_text "Enable location?"
"""

import argparse
import copy
import json
import os
import random
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFilter, ImageFont

# ---------------------------------------------------------------------------
# Import renderer (for launcher mode)
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

try:
    from mock_unified_app_graph import (
        _render_launcher_bundle,
        _load_layout_config,
    )
    HAS_RENDERER = True
except ImportError:
    HAS_RENDERER = False


# ===================================================================
#  Common helpers
# ===================================================================

def hex_to_rgb(h: str) -> Tuple[int, int, int]:
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(v, hi))


def try_load_font(size: int = 28):
    for fp in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]:
        if os.path.exists(fp):
            try:
                return ImageFont.truetype(fp, size)
            except Exception:
                continue
    return ImageFont.load_default()


def _boxes_overlap(a, b, margin: int = 8) -> bool:
    return not (a[2] + margin <= b[0] or b[2] + margin <= a[0] or
                a[3] + margin <= b[1] or b[3] + margin <= a[1])


def _sample_local_bg(img: Image.Image, x1, y1, x2, y2, margin=12) -> Tuple[int, ...]:
    """Sample pixels around bbox to estimate local bg color."""
    W, H = img.size
    samples = []
    for _ in range(80):
        side = random.choice(["top", "bottom", "left", "right"])
        if side == "top":
            sx, sy = random.randint(max(0, x1), min(W-1, x2)), max(0, y1 - random.randint(1, margin))
        elif side == "bottom":
            sx, sy = random.randint(max(0, x1), min(W-1, x2)), min(H-1, y2 + random.randint(1, margin))
        elif side == "left":
            sx, sy = max(0, x1 - random.randint(1, margin)), random.randint(max(0, y1), min(H-1, y2))
        else:
            sx, sy = min(W-1, x2 + random.randint(1, margin)), random.randint(max(0, y1), min(H-1, y2))
        samples.append(img.getpixel((sx, sy))[:3])
    rounded = [tuple((c // 8) * 8 for c in p) for p in samples]
    return Counter(rounded).most_common(1)[0][0]


def _sample_edge_bg(img: Image.Image, n=200) -> Tuple[int, int, int]:
    w, h = img.size
    samples = []
    for _ in range(n):
        edge = random.choice(["top", "bottom", "left", "right"])
        if edge == "top":
            x, y = random.randint(0, w-1), random.randint(0, max(0, int(h*0.03)))
        elif edge == "bottom":
            x, y = random.randint(0, w-1), random.randint(int(h*0.97), h-1)
        elif edge == "left":
            x, y = random.randint(0, max(0, int(w*0.03))), random.randint(0, h-1)
        else:
            x, y = random.randint(int(w*0.97), w-1), random.randint(0, h-1)
        samples.append(img.getpixel((x, y))[:3])
    rounded = [tuple((c // 16) * 16 for c in p) for p in samples]
    return Counter(rounded).most_common(1)[0][0]


# ===================================================================
#  LAUNCHER MODE — config-based (home + app drawer)
# ===================================================================

def launcher_shuffle(config: dict, shuffle_target: str = "home+drawer",
                     seed: Optional[int] = None) -> dict:
    """
    Shuffle which icon goes to which slot (not the slot coordinates).

    For home page:  icons get randomly reassigned to home_slot_1..8
    For app drawer: icons get randomly reassigned to drawer_slot_1..24
    For dock:       icons get randomly reassigned to dock_slot_1..4

    The grid layout stays intact — only icon-to-slot mapping changes.
    """
    rng = random.Random(seed)
    config = copy.deepcopy(config)
    target_parts = set(shuffle_target.lower().replace(" ", "").split("+"))
    pages = config.get("pages", {})

    # Home icons
    if "all" in target_parts or "home" in target_parts:
        home_icons = pages.get("home", {}).get("icons", [])
        if home_icons:
            home_slots = [ic["slot"] for ic in home_icons if "slot" in ic]
            rng.shuffle(home_slots)
            for ic, new_slot in zip(home_icons, home_slots):
                ic["slot"] = new_slot
            print(f"[launcher-shuffle] home: shuffled {len(home_icons)} icons across {len(home_slots)} slots")

    # Dock icons
    if "all" in target_parts or "dock" in target_parts:
        dock_icons = pages.get("home", {}).get("dock_icons", [])
        if dock_icons:
            dock_slots = [ic["slot"] for ic in dock_icons if "slot" in ic]
            rng.shuffle(dock_slots)
            for ic, new_slot in zip(dock_icons, dock_slots):
                ic["slot"] = new_slot
            print(f"[launcher-shuffle] dock: shuffled {len(dock_icons)} icons")

    # App drawer icons
    if "all" in target_parts or "drawer" in target_parts:
        drawer_config = pages.get("app_drawer", {})
        drawer_icons = drawer_config.get("icons", [])
        if drawer_icons:
            has_slots = [ic for ic in drawer_icons if "slot" in ic]
            if has_slots:
                # Slot-based: shuffle slot assignments
                slots = [ic["slot"] for ic in has_slots]
                rng.shuffle(slots)
                for ic, new_slot in zip(has_slots, slots):
                    ic["slot"] = new_slot
                print(f"[launcher-shuffle] drawer: shuffled {len(has_slots)} slot-based icons")
            else:
                # List-order-based (annotations): shuffle the list itself
                rng.shuffle(drawer_icons)
                drawer_config["icons"] = drawer_icons
                print(f"[launcher-shuffle] drawer: shuffled {len(drawer_icons)} icons (list order)")

    return config


def launcher_bgcolor(page_images: Dict[str, Image.Image],
                     target_color: str, tolerance: int = 50) -> Dict[str, Image.Image]:
    """Recolor background of rendered launcher page images."""
    target_rgb = hex_to_rgb(target_color)
    for page_id, img in page_images.items():
        img = img.convert("RGB")
        bg = _sample_edge_bg(img)
        px = img.load()
        W, H = img.size
        for y in range(H):
            for x in range(W):
                r, g, b = px[x, y]
                if abs(r-bg[0])<tolerance and abs(g-bg[1])<tolerance and abs(b-bg[2])<tolerance:
                    d = (r-bg[0], g-bg[1], b-bg[2])
                    px[x, y] = (clamp(target_rgb[0]+d[0],0,255),
                                clamp(target_rgb[1]+d[1],0,255),
                                clamp(target_rgb[2]+d[2],0,255))
        page_images[page_id] = img
        print(f"[launcher-bgcolor] {page_id}: {bg} -> {target_color}")
    return page_images


# ===================================================================
#  APP MODE — image-based (in-app pages)
# ===================================================================

def _load_page_assets(env_dir: str, page_id: str, manifest: List[dict]) -> List[dict]:
    """Load extracted asset images for a given page."""
    page_assets = [r for r in manifest if r.get("page_id") == page_id]
    result = []
    for r in page_assets:
        # Resolve asset path: manifest uses "amex_all_apps_graph/..." but
        # the actual dir might be "amex_all_apps_graph_0330/..."
        raw_path = r["asset_path"]
        # Try: env_dir / relative part after first "extracted_assets/"
        if "extracted_assets/" in raw_path:
            rel = "extracted_assets/" + raw_path.split("extracted_assets/", 1)[1]
            full = os.path.join(env_dir, rel)
        else:
            full = os.path.join(os.path.dirname(env_dir), raw_path)

        if os.path.exists(full):
            result.append({**r, "resolved_path": full})
    return result


def app_shift(
    structure: dict,
    pages_dir: str,
    output_pages_dir: str,
    page_ids: List[str],
    env_dir: str = "",
    seed: Optional[int] = None,
) -> dict:
    """
    Reconstruct app pages from assets with shuffled positions.

    Uses the same approach as _ac_render_reconstructed_native_page:
      1. Take original screenshot → blur it as background
      2. Load extracted asset images for each element
      3. Shuffle element positions (similar-size swap + random shift)
      4. Paste assets at new positions on the blurred background

    This produces clean, large-scale rearrangements because each element
    is rendered fresh from its asset image, not cropped from the composite.
    """
    rng = random.Random(seed)
    struct = copy.deepcopy(structure)
    os.makedirs(output_pages_dir, exist_ok=True)

    # Load asset manifest
    manifest_path = os.path.join(env_dir, "trajectory_assets_manifest.json")
    manifest = []
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)

    for page_id in page_ids:
        page_data = struct["pages"].get(page_id)
        if not page_data:
            print(f"[app-shift] {page_id} not found, skip"); continue

        img_path = os.path.join(pages_dir, page_data["image"])
        img = Image.open(img_path).convert("RGB")
        W, H = img.size
        layout = page_data["layout"]

        # Load assets for this page
        assets = _load_page_assets(env_dir, page_id, manifest)
        asset_by_label = {}
        for a in assets:
            # Match asset to layout element by label
            label = a["label"]
            for layout_name in layout:
                # Fuzzy match: layout name often contains label
                if label in layout_name or layout_name in label:
                    asset_by_label[layout_name] = a
                    break

        # Classify elements
        fixed_names = {n for n, info in layout.items() if info.get("type") == "system"}
        shiftable = []
        pinned = []
        for n, info in layout.items():
            if info.get("type") != "normal":
                continue
            b = info["bbox"]
            if (b[2] - b[0]) >= W * 0.85:
                pinned.append(n)
            else:
                shiftable.append(n)

        if not shiftable:
            print(f"[app-shift] {page_id}: no shiftable elements, skip"); continue

        sys_bottom = max((layout[n]["bbox"][3] for n in fixed_names), default=0)

        # --- Step 1: Create blurred background ---
        blurred_bg = img.filter(ImageFilter.GaussianBlur(radius=14))
        canvas = img.copy()

        # Erase shiftable element areas with blur
        for name in shiftable:
            b = layout[name]["bbox"]
            pad = 2
            bx1, by1 = max(0, b[0]-pad), max(0, b[1]-pad)
            bx2, by2 = min(W, b[2]+pad), min(H, b[3]+pad)
            patch = blurred_bg.crop((bx1, by1, bx2, by2))
            canvas.paste(patch, (bx1, by1))

        canvas = canvas.convert("RGBA")

        # --- Step 2: Collect element images (from assets or crop) ---
        elem_images = {}
        for name in shiftable:
            b = layout[name]["bbox"]
            ew, eh = b[2]-b[0], b[3]-b[1]
            if name in asset_by_label:
                apath = asset_by_label[name]["resolved_path"]
                elem_images[name] = Image.open(apath).convert("RGBA").resize((ew, eh), Image.LANCZOS)
            else:
                # Fallback: crop from original image
                elem_images[name] = img.crop((b[0], b[1], b[2], b[3])).convert("RGBA")

        # --- Step 3: Group by similar size and shuffle positions ---
        orig_bboxes = {n: list(layout[n]["bbox"]) for n in shiftable}

        # Group elements by similar dimensions
        items = sorted(shiftable, key=lambda n: (
            (orig_bboxes[n][2]-orig_bboxes[n][0]) * (orig_bboxes[n][3]-orig_bboxes[n][1])
        ))
        groups = []
        current = [items[0]]
        for i in range(1, len(items)):
            pb = orig_bboxes[current[-1]]
            cb = orig_bboxes[items[i]]
            pw, ph = pb[2]-pb[0], pb[3]-pb[1]
            cw, ch = cb[2]-cb[0], cb[3]-cb[1]
            if pw > 0 and ph > 0 and abs(cw-pw)/max(pw,1) < 0.5 and abs(ch-ph)/max(ph,1) < 0.5:
                current.append(items[i])
            else:
                groups.append(current)
                current = [items[i]]
        groups.append(current)

        # Shuffle positions within each group
        new_positions = {}
        swap_count = 0
        for group in groups:
            if len(group) < 2:
                new_positions[group[0]] = orig_bboxes[group[0]]
                continue
            positions = [orig_bboxes[n] for n in group]
            shuffled = list(positions)
            for _ in range(20):
                rng.shuffle(shuffled)
                if shuffled != positions:
                    break
            for name, new_pos in zip(group, shuffled):
                old_b = orig_bboxes[name]
                ew, eh = old_b[2]-old_b[0], old_b[3]-old_b[1]
                # Place element centered on new_pos center
                ncx = (new_pos[0] + new_pos[2]) // 2
                ncy = (new_pos[1] + new_pos[3]) // 2
                nx = clamp(ncx - ew//2, 0, W - ew)
                ny = clamp(ncy - eh//2, max(sys_bottom+5, 0), H - eh)
                new_positions[name] = [nx, ny, nx+ew, ny+eh]
                if nx != old_b[0] or ny != old_b[1]:
                    swap_count += 1

        # --- Step 4: Paste elements at new positions ---
        # Sort by area descending (large elements behind small ones)
        paste_order = sorted(shiftable,
            key=lambda n: (new_positions[n][2]-new_positions[n][0])*(new_positions[n][3]-new_positions[n][1]),
            reverse=True)

        for name in paste_order:
            nb = new_positions[name]
            if name in elem_images:
                canvas.alpha_composite(elem_images[name], (nb[0], nb[1]))
            layout[name]["bbox"] = nb

        # Also paste pinned elements (from original crop) to preserve them
        for name in pinned:
            b = layout[name]["bbox"]
            crop = img.crop((b[0], b[1], b[2], b[3])).convert("RGBA")
            canvas.alpha_composite(crop, (b[0], b[1]))

        # Sync transitions
        all_moved = set(shiftable)
        for trans in page_data.get("transitions", []):
            act = trans.get("action", "")
            tgt = trans.get("target_element", "")
            matched = act if act in all_moved else (tgt if tgt in all_moved else None)
            if matched and matched in layout:
                nb = layout[matched]["bbox"]
                trans["icon_bbox"] = nb
                trans["action_coord"] = [(nb[0]+nb[2])//2, (nb[1]+nb[3])//2]

        out_path = os.path.join(output_pages_dir, page_data["image"])
        canvas.convert("RGB").save(out_path)

        # Save original for comparison
        orig_out = os.path.join(output_pages_dir, f"ORIG_{page_data['image']}")
        img.save(orig_out)

        print(f"[app-shift] {page_id}: {swap_count}/{len(shiftable)} swapped "
              f"({len(asset_by_label)} assets loaded, {len(pinned)} pinned)")

    return struct


def app_bgcolor(
    structure: dict,
    pages_dir: str,
    output_pages_dir: str,
    page_ids: List[str],
    target_color: str = "#1A1A2E",
    tolerance: int = 50,
) -> dict:
    """Recolor background of in-app page images."""
    struct = copy.deepcopy(structure)
    os.makedirs(output_pages_dir, exist_ok=True)
    target_rgb = hex_to_rgb(target_color)

    for page_id in page_ids:
        page_data = struct["pages"].get(page_id)
        if not page_data:
            print(f"[app-bgcolor] {page_id} not found, skip"); continue

        img_path = os.path.join(pages_dir, page_data["image"])
        img = Image.open(img_path).convert("RGB")
        bg = _sample_edge_bg(img)
        px = img.load()
        W, H = img.size
        for y in range(H):
            for x in range(W):
                r, g, b = px[x, y]
                if abs(r-bg[0])<tolerance and abs(g-bg[1])<tolerance and abs(b-bg[2])<tolerance:
                    d = (r-bg[0], g-bg[1], b-bg[2])
                    px[x, y] = (clamp(target_rgb[0]+d[0],0,255),
                                clamp(target_rgb[1]+d[1],0,255),
                                clamp(target_rgb[2]+d[2],0,255))
        out_path = os.path.join(output_pages_dir, page_data["image"])
        img.save(out_path)
        print(f"[app-bgcolor] {page_id}: {bg} -> {target_color}")

    return struct


# ===================================================================
#  Popup overlay (shared — works on any PIL Image dict)
# ===================================================================

def add_popup(
    page_images: Dict[str, Image.Image],
    ui_structure: Optional[dict],
    target_pages: List[str],
    popup_text: str = "Allow access?",
    popup_buttons: Optional[List[str]] = None,
) -> Tuple[Dict[str, Image.Image], Optional[dict]]:
    if popup_buttons is None:
        popup_buttons = ["Allow", "Deny"]
    struct = copy.deepcopy(ui_structure) if ui_structure else None

    for page_id in target_pages:
        if page_id not in page_images:
            print(f"[popup] {page_id} not found, skip"); continue

        img = page_images[page_id].convert("RGBA")
        W, H = img.size

        # Dim
        img = Image.alpha_composite(img, Image.new("RGBA", (W, H), (0, 0, 0, 100)))

        # Popup box
        pw, ph = int(W*0.75), int(H*0.22)
        px, py = (W-pw)//2, (H-ph)//2
        layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        d.rounded_rectangle([px, py, px+pw, py+ph], radius=30,
                            fill=(255,255,255,240), outline=(200,200,200,255))

        font_t = try_load_font(36)
        font_b = try_load_font(30)
        tb = d.textbbox((0, 0), popup_text, font=font_t)
        d.text((px+(pw-(tb[2]-tb[0]))//2, py+40), popup_text,
               fill=(30,30,30,255), font=font_t)

        # Buttons
        btn_h, btn_m = 70, 30
        sbw = (pw-2*btn_m)//len(popup_buttons)
        btn_y = py+ph-btn_h-30
        colors = [(66,133,244,255),(220,220,220,255),(234,67,53,255),(52,168,83,255)]

        page_layout = None
        if struct:
            for p in struct.get("pages", {}).values():
                if p.get("page_id") == page_id:
                    page_layout = p.get("layout", {}); break

        for i, lbl in enumerate(popup_buttons):
            bx = px + btn_m + i*sbw + 10
            bw = sbw - 20
            d.rounded_rectangle([bx, btn_y, bx+bw, btn_y+btn_h], radius=15,
                                fill=colors[i%len(colors)])
            btb = d.textbbox((0, 0), lbl, font=font_b)
            d.text((bx+(bw-(btb[2]-btb[0]))//2, btn_y+(btn_h-(btb[3]-btb[1]))//2),
                   lbl, fill=(255,255,255,255), font=font_b)
            if page_layout is not None:
                page_layout[f"popup_btn_{lbl.lower().replace(' ','_')}"] = {
                    "bbox": [bx, btn_y, bx+bw, btn_y+btn_h], "type": "popup"}

        if page_layout is not None:
            page_layout["popup_overlay"] = {
                "bbox": [px, py, px+pw, py+ph], "type": "popup"}

        page_images[page_id] = Image.alpha_composite(img, layer).convert("RGB")
        print(f"[popup] {page_id}: popup added")

    return page_images, struct


# ===================================================================
#  Save
# ===================================================================

def save_output(output_dir: str, page_images: Dict[str, Image.Image],
                ui_structure: Optional[dict] = None, config: Optional[dict] = None):
    out = Path(output_dir)
    pages_dir = out / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)
    for pid, img in page_images.items():
        fname = f"{pid}.png" if not pid.endswith(".png") else pid
        img.save(pages_dir / fname)
    if ui_structure:
        (out / "ui_structure.json").write_text(json.dumps(ui_structure, indent=2))
    if config:
        (out / "layout_config_augmented.json").write_text(
            json.dumps(config, indent=2, ensure_ascii=False))
    print(f"[save] {len(page_images)} pages -> {out}")


# ===================================================================
#  Page ID resolver (for app mode)
# ===================================================================

def resolve_page_ids(structure: dict, pages_arg: str) -> List[str]:
    all_ids = list(structure["pages"].keys())
    if pages_arg == "all":
        return all_ids
    if pages_arg.startswith("depth="):
        d = int(pages_arg.split("=")[1])
        return [pid for pid, p in structure["pages"].items() if p.get("depth") == d]
    if pages_arg.startswith("app="):
        app = pages_arg.split("=", 1)[1].lower()
        return [pid for pid, p in structure["pages"].items()
                if app in p.get("application_name", "").lower()
                or app in p.get("application_id", "").lower()]
    requested = [p.strip() for p in pages_arg.split(",")]
    return [p for p in requested if p in structure["pages"]]


# ===================================================================
#  CLI
# ===================================================================

def main():
    parser = argparse.ArgumentParser(description="Page Augmentor (launcher + app modes)")
    sub = parser.add_subparsers(dest="mode", required=True)

    # ---- Launcher mode ----
    lp = sub.add_parser("launcher", help="Config-based augmentation for home + app drawer")
    lp.add_argument("--config", type=str, default=str(SCRIPT_DIR / "mock_simulator_layout_config.json"))
    lp.add_argument("--annotations_dir", type=str, default=None,
                    help="AMEX annotations dir to auto-populate drawer icons")
    lp.add_argument("--output_dir", required=True)
    lp.add_argument("--augment", required=True, choices=["shift", "bgcolor", "popup", "all"])
    lp.add_argument("--seed", type=int, default=None)
    lp.add_argument("--shuffle_target", default="home+drawer",
                    help="Which icon groups to shuffle: home, drawer, dock, all, home+drawer")
    lp.add_argument("--bg_color", default="#FFE0B2")
    lp.add_argument("--bg_tolerance", type=int, default=50)
    lp.add_argument("--popup_text", default="Allow access?")
    lp.add_argument("--popup_buttons", default="Allow,Deny")
    lp.add_argument("--popup_pages", default="page_0",
                    help="page_0, all, page_0,page_1_app_drawer")

    # ---- App mode ----
    ap = sub.add_parser("app", help="Image-based augmentation for in-app pages")
    ap.add_argument("--env_dir", required=True,
                    help="Environment dir (contains ui_structure.json + pages/)")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--augment", required=True, choices=["shift", "bgcolor", "popup", "all"])
    ap.add_argument("--pages", default="page_5",
                    help="page_5, page_5,page_10, depth=4, app=amazon, all")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--shift_range", type=int, default=150,
                    help="Max shift in px (adaptive per element size, default: 150)")
    ap.add_argument("--bg_color", default="#1A1A2E")
    ap.add_argument("--bg_tolerance", type=int, default=50)
    ap.add_argument("--popup_text", default="Allow access?")
    ap.add_argument("--popup_buttons", default="Allow,Deny")

    args = parser.parse_args()
    augments = [args.augment] if args.augment != "all" else ["shift", "bgcolor", "popup"]

    # ================================================================
    # LAUNCHER MODE
    # ================================================================
    if args.mode == "launcher":
        if not HAS_RENDERER:
            print("[error] Cannot import mock_unified_app_graph. Is it in the same directory?")
            return

        ann_dir = Path(args.annotations_dir) if args.annotations_dir else None
        config = _load_layout_config(Path(args.config), annotations_dir=ann_dir)

        # Render original first for comparison
        orig_bundle = _render_launcher_bundle(config)
        orig_images = orig_bundle["page_images"]

        if "shift" in augments:
            config = launcher_shuffle(config, args.shuffle_target, args.seed)

        bundle = _render_launcher_bundle(config)
        page_images = bundle["page_images"]
        ui_structure = bundle["ui_structure"]
        print(f"[render] {len(page_images)} launcher pages rendered")

        # Save originals for comparison
        orig_dir = Path(args.output_dir) / "pages"
        orig_dir.mkdir(parents=True, exist_ok=True)
        for pid, img in orig_images.items():
            img.save(orig_dir / f"ORIG_{pid}.png")
        print(f"[save] {len(orig_images)} original pages saved for comparison")

        if "bgcolor" in augments:
            page_images = launcher_bgcolor(page_images, args.bg_color, args.bg_tolerance)

        if "popup" in augments:
            popup_pages = args.popup_pages.split(",")
            if "all" in popup_pages:
                popup_pages = list(page_images.keys())
            page_images, ui_structure = add_popup(
                page_images, ui_structure, popup_pages,
                args.popup_text, [b.strip() for b in args.popup_buttons.split(",")])

        save_output(args.output_dir, page_images, ui_structure, config)

    # ================================================================
    # APP MODE
    # ================================================================
    elif args.mode == "app":
        env_dir = args.env_dir
        struct_path = os.path.join(env_dir, "ui_structure.json")
        with open(struct_path) as f:
            structure = json.load(f)
        pages_dir = os.path.join(env_dir, "pages")
        output_pages_dir = os.path.join(args.output_dir, "pages")

        page_ids = resolve_page_ids(structure, args.pages)
        if not page_ids:
            print("[error] No pages matched"); return
        print(f"[info] Target: {len(page_ids)} page(s): {page_ids[:5]}{'...' if len(page_ids)>5 else ''}")

        if "shift" in augments:
            structure = app_shift(structure, pages_dir, output_pages_dir,
                                  page_ids, env_dir=env_dir, seed=args.seed)

        if "bgcolor" in augments:
            src = output_pages_dir if "shift" in augments else pages_dir
            structure = app_bgcolor(structure, src, output_pages_dir,
                                    page_ids, args.bg_color, args.bg_tolerance)

        if "popup" in augments:
            src = output_pages_dir if ("shift" in augments or "bgcolor" in augments) else pages_dir
            # Load rendered images
            pimages = {}
            for pid in page_ids:
                pd = structure["pages"][pid]
                ip = os.path.join(src, pd["image"])
                if not os.path.exists(ip):
                    ip = os.path.join(pages_dir, pd["image"])
                pimages[pid] = Image.open(ip)
            pimages, structure = add_popup(
                pimages, structure, page_ids,
                args.popup_text, [b.strip() for b in args.popup_buttons.split(",")])
            os.makedirs(output_pages_dir, exist_ok=True)
            for pid, im in pimages.items():
                im.save(os.path.join(output_pages_dir, structure["pages"][pid]["image"]))

        # Save structure
        out_struct_path = os.path.join(args.output_dir, "ui_structure.json")
        os.makedirs(args.output_dir, exist_ok=True)
        with open(out_struct_path, "w") as f:
            json.dump(structure, f, indent=2)

        print(f"[save] -> {args.output_dir}")

    print("[done]")


if __name__ == "__main__":
    main()
