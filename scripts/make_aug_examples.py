"""Build side-by-side before/after comparison PNGs for each augmentation operator.

Picks one trajectory per operator (applied alone) from augmentation_log.json,
loads the matching original + augmented page, and writes a 1-row / 2-column
comparison image with labels to demo/aug_examples/.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

ROOT = Path("/workspace/gelab-env/datas")
ORIG_ROOT = ROOT / "amex_gelab_extracted/amex_sft"
AUG_ROOT = ROOT / "amex_gelab_augmented"
LOG_PATH = AUG_ROOT / "augmentation_log.json"
OUT_DIR = Path("/workspace/gelab-env/demo/aug_examples")
OUT_DIR.mkdir(parents=True, exist_ok=True)

log = json.load(open(LOG_PATH))
traj = log["trajectories"]


def pick_one(operator: str, exclusive: bool = True) -> Optional[str]:
    """Find a trajectory where the *only* applied operator is `operator`."""
    for tid, meta in traj.items():
        augs = meta.get("augmentations") or []
        if meta.get("status") != "ok":
            continue
        if exclusive and augs == [operator]:
            return tid
        if (not exclusive) and operator in augs:
            return tid
    return None


def find_common_page(tid: str, prefer_page: Optional[str] = None) -> Optional[str]:
    """Return a page filename present in both the original and augmented dir."""
    orig_pages = set(os.listdir(ORIG_ROOT / tid / "pages"))
    aug_pages = set(os.listdir(AUG_ROOT / tid / "pages"))
    common = sorted(orig_pages & aug_pages)
    if not common:
        return None
    if prefer_page:
        for p in common:
            if prefer_page in p:
                return p
    return common[0]


def side_by_side(left: Image.Image, right: Image.Image,
                 left_label: str, right_label: str,
                 caption: str) -> Image.Image:
    """Create a 2-panel image with labels + caption strip."""
    max_h = 1600
    # Resize both to a common height, preserving aspect
    def fit(im: Image.Image) -> Image.Image:
        if im.height > max_h:
            r = max_h / im.height
            return im.resize((int(im.width * r), max_h), Image.LANCZOS)
        return im

    L, R = fit(left), fit(right)
    H = max(L.height, R.height)
    pad = 24
    label_h = 56
    caption_h = 72
    W = L.width + R.width + pad * 3
    total_h = H + label_h + caption_h + pad
    canvas = Image.new("RGB", (W, total_h), (250, 250, 250))

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 36)
        small = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 28)
    except Exception:
        font = ImageFont.load_default()
        small = ImageFont.load_default()

    draw = ImageDraw.Draw(canvas)
    # Caption
    draw.text((pad, pad // 2), caption, fill=(30, 30, 30), font=font)
    # Panels
    y0 = label_h + pad // 2
    draw.text((pad, y0 - 40), left_label, fill=(30, 30, 30), font=small)
    canvas.paste(L, (pad, y0))
    draw.text((L.width + pad * 2, y0 - 40), right_label, fill=(30, 30, 30), font=small)
    canvas.paste(R, (L.width + pad * 2, y0))
    return canvas


def build(operator: str, caption: str, prefer_page: Optional[str] = None) -> Optional[Path]:
    tid = pick_one(operator)
    if tid is None:
        print(f"[{operator}] no exclusive-operator trajectory found")
        return None
    page = find_common_page(tid, prefer_page)
    if not page:
        print(f"[{operator}] no common page in traj {tid}")
        return None
    orig = Image.open(ORIG_ROOT / tid / "pages" / page).convert("RGB")
    aug = Image.open(AUG_ROOT / tid / "pages" / page).convert("RGB")
    meta = traj[tid]
    extra = []
    if operator == "bgcolor" and meta.get("bgcolor_target"):
        extra.append(f"target={meta['bgcolor_target']}")
    if operator == "popup" and meta.get("popup_text"):
        extra.append(f'text="{meta["popup_text"]}"')
        extra.append(f'inserted_at={meta.get("popup_page")}')
    extra_s = "  ·  ".join(extra)
    full_caption = f"{caption} — traj {tid[:8]}… / {page}"
    if extra_s:
        full_caption += f"\n{extra_s}"
    # side_by_side takes a single-line caption; fold to one line
    full_caption = full_caption.replace("\n", "   |   ")
    img = side_by_side(orig, aug, "Original", f"Augmented: {operator}", full_caption)
    out = OUT_DIR / f"aug_{operator}.png"
    img.save(out, optimize=True)
    print(f"[{operator}] wrote {out}  (orig={orig.size}, aug={aug.size})")
    return out


def build_popup_traj(tid: str, caption: str) -> Optional[Path]:
    """For popup, the augmented dir has an extra page_X_popup not in the original.
    Pair the popup page vs the original 'page_X' it was inserted before."""
    meta = traj[tid]
    popup_page = meta.get("popup_page")  # e.g. "page_1"
    if not popup_page:
        return None
    orig_file = f"{popup_page}.png"
    aug_file_candidates = [f"{popup_page}_popup.png", f"{popup_page}.png"]
    orig_path = ORIG_ROOT / tid / "pages" / orig_file
    aug_path = None
    for cand in aug_file_candidates:
        p = AUG_ROOT / tid / "pages" / cand
        if p.exists():
            if cand.endswith("_popup.png"):
                aug_path = p
                break
    if aug_path is None or not orig_path.exists():
        return None
    orig = Image.open(orig_path).convert("RGB")
    aug = Image.open(aug_path).convert("RGB")
    extra = f'text="{meta.get("popup_text","")}"  ·  inserted_before={popup_page}'
    full_caption = f"{caption} — traj {tid[:8]}… / {popup_page}   |   {extra}"
    img = side_by_side(orig, aug, f"Original {popup_page}",
                       f"Inserted popup ({popup_page}_popup)",
                       full_caption)
    out = OUT_DIR / "aug_popup.png"
    img.save(out, optimize=True)
    print(f"[popup] wrote {out}  (orig={orig.size}, aug={aug.size})")
    return out


if __name__ == "__main__":
    # Shift: pages same name, content rearranged
    build("shift", "Shift — element positions rearranged on the same page")
    # Bgcolor: pages same name, background recolored
    build("bgcolor", "Background recolor — pastel palette substitution")
    # Popup: new *_popup.png page inserted — handle specially
    tid = pick_one("popup")
    if tid:
        build_popup_traj(tid, "Popup — permission dialog inserted before the original page")
    print("\nDone. See:", OUT_DIR)
