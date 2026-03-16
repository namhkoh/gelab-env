#!/usr/bin/env python3
"""One-shot: build placeholder+cache for GUIOdyssey episode 2493102722960871, then run tree.py."""
import json
import os
import subprocess
import sys
from pathlib import Path
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
os.chdir(REPO)

EPISODE = "2493102722960871"
DATA_DIR = REPO / "datas" / f"guiodyssey_{EPISODE}"
OUT_DIR = REPO / "datas" / f"guiodyssey_real_{EPISODE}" / "out"

def main():
    ann_path = DATA_DIR / "annotations" / f"{EPISODE}.json"
    if not ann_path.exists():
        print("Missing annotation. Run: python scripts/run_trajectory_real_guiodyssey.py --no_run")
        return 1
    with open(ann_path, encoding="utf-8") as f:
        ann = json.load(f)
    steps = ann.get("steps", [])
    device = ann.get("device_info", {})
    w, h = device.get("w", 720), device.get("h", 1280)
    canvas_w, canvas_h = 252, 448

    screenshots_dir = DATA_DIR / "screenshots"
    screenshots_dir.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    family_cache_dir = OUT_DIR / "page_families"
    family_cache_dir.mkdir(parents=True, exist_ok=True)

    for step_idx, step in enumerate(steps):
        name = step.get("screenshot", f"{EPISODE}_{step_idx}.png")
        img = Image.new("RGB", (w, h), color=(240, 240, 240))
        img.save(screenshots_dir / name)

        bbox = step.get("sam2_bbox") or []
        if len(bbox) >= 4:
            x1, y1, x2, y2 = [int(bbox[i]) for i in range(4)]
            c1, c2 = int(x1 * canvas_w / w), int(y1 * canvas_h / h)
            c3 = max(c1 + 1, int(x2 * canvas_w / w))
            c4 = max(c2 + 1, int(y2 * canvas_h / h))
            layout = {"click_0": [c1, c2, c3, c4]}
        else:
            layout = {"dummy": [10, 10, 30, 30]}

        step_ctx = dict(step)
        step_ctx.setdefault("task", ann.get("task_info", {}).get("task", ""))
        step_ctx.setdefault("step_index", step_idx + 1)
        step_ctx.setdefault("total_steps", len(steps))
        family = {
            "page_id": f"page_{step_idx}",
            "page_family_id": f"family_{step_idx:03d}",
            "source_step_index": step_idx,
            "screenshot_name": name,
            "screenshot_path": str(screenshots_dir / name),
            "orig_size": [w, h],
            "canvas_size": [canvas_w, canvas_h],
            "step": step_ctx,
            "elements": [],
            "layout": layout,
            "action_name_counts": {k: 1 for k in layout},
            "render_mode": "crop_reconstructed",
        }
        with open(family_cache_dir / f"family_{step_idx:03d}.json", "w", encoding="utf-8") as f:
            json.dump(family, f, indent=2)
        canonical = Image.new("RGB", (canvas_w, canvas_h), color=(248, 248, 248))
        canonical.save(family_cache_dir / f"family_{step_idx:03d}_canonical.png")

    print("Running tree.py...")
    r = subprocess.run([
        sys.executable, "data_engine/tree.py",
        "--icon_source", "trajectory",
        "--trajectory_id", EPISODE,
        "--annotations_dir", str(DATA_DIR / "annotations"),
        "--screenshots_dir", str(screenshots_dir),
        "--output_dir", str(OUT_DIR),
    ], cwd=REPO)
    if r.returncode != 0:
        return r.returncode
    ui_path = OUT_DIR / "ui_structure.json"
    if ui_path.exists():
        with open(ui_path, encoding="utf-8") as f:
            ui = json.load(f)
        print("effective_spine_page_ids:", ui.get("metadata", {}).get("effective_spine_page_ids"))
        print("total_pages:", ui.get("metadata", {}).get("total_pages"))
        print("Saved:", ui_path)
    return 0

if __name__ == "__main__":
    sys.exit(main())
