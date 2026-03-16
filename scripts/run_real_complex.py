#!/usr/bin/env python3
"""
복잡한 태스크: 18스텝 GUIOdyssey 에피소드로 실행.
merge를 보려면 step 0과 step 5를 같은 layout으로 맞춤 (같은 상태 재방문 시뮬레이션).
"""
import json
import os
import subprocess
import sys
from pathlib import Path
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
os.chdir(REPO)

# 18스텝 에피소드 (Opera + Simplenote, Circle 문서)
EPISODE = "7872483543119388"
DATA_DIR = REPO / "datas" / f"guiodyssey_{EPISODE}"
OUT_DIR = REPO / "datas" / f"guiodyssey_real_{EPISODE}" / "out"

# 같은 layout으로 만들 step 쌍 → merge 발생
MERGE_SAME_LAYOUT_STEPS = (0, 5)  # step 0과 step 5를 동일 layout으로


def main():
    ann_path = DATA_DIR / "annotations" / f"{EPISODE}.json"
    if not ann_path.exists():
        print("Missing annotation. Run: python scripts/run_trajectory_real_guiodyssey.py --episode_id 7872483543119388 --no_run")
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

    def bbox_to_layout(bbox, step_idx):
        if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            x1, y1, x2, y2 = [int(bbox[i]) for i in range(4)]
            c1 = int(x1 * canvas_w / w)
            c2 = int(y1 * canvas_h / h)
            c3 = max(c1 + 1, int(x2 * canvas_w / w))
            c4 = max(c2 + 1, int(y2 * canvas_h / h))
            return {"click_0": [c1, c2, c3, c4]}
        return {"dummy": [10 + step_idx, 10, 30 + step_idx, 30]}

    layouts = []
    for step_idx, step in enumerate(steps):
        bbox = step.get("sam2_bbox") or []
        layouts.append(bbox_to_layout(bbox, step_idx))

    # merge 시연: MERGE_SAME_LAYOUT_STEPS에 해당하는 step들은 같은 layout 사용
    if MERGE_SAME_LAYOUT_STEPS:
        i0, i1 = MERGE_SAME_LAYOUT_STEPS[0], MERGE_SAME_LAYOUT_STEPS[1]
        if i0 < len(layouts) and i1 < len(layouts):
            layouts[i1] = dict(layouts[i0])
            print(f"Merge 시연: step {i0}와 step {i1}을 같은 layout으로 설정 → spine에서 step {i1}이 page_{i0}로 합쳐짐")

    for step_idx, step in enumerate(steps):
        name = step.get("screenshot", f"{EPISODE}_{step_idx}.png")
        img = Image.new("RGB", (w, h), color=(240, 240, 240))
        img.save(screenshots_dir / name)

        layout = layouts[step_idx]
        step_ctx = {k: v for k, v in step.items() if k != "sam2_bbox"}
        step_ctx.setdefault("task", ann.get("task_info", {}).get("task", ""))
        step_ctx.setdefault("step_index", step_idx + 1)
        step_ctx.setdefault("total_steps", len(steps))
        step_ctx["sam2_bbox"] = []  # avoid _choose_click_target importing sim2real_compose
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

    print(f"Running tree.py (episode {EPISODE}, {len(steps)} steps)...")
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
        meta = ui.get("metadata", {})
        eff = meta.get("effective_spine_page_ids", [])
        n = meta.get("total_pages", 0)
        print("--- 결과 ---")
        print("total_pages:", n)
        print("effective_spine_page_ids (처음 10 + ...):", eff[:10], "..." if len(eff) > 10 else "")
        if MERGE_SAME_LAYOUT_STEPS:
            i0, i1 = MERGE_SAME_LAYOUT_STEPS[0], MERGE_SAME_LAYOUT_STEPS[1]
            if i1 < len(eff) and eff[i1] == f"page_{i0}":
                print(f"OK: step {i1}이 page_{i0}로 merge됨 (effective_spine[{i1}] = {eff[i1]})")
            else:
                print(f"effective_spine[{i1}] = {eff[i1] if i1 < len(eff) else '?'} (기대: page_{i0})")
        print("Saved:", ui_path)
    return 0

if __name__ == "__main__":
    sys.exit(main())
