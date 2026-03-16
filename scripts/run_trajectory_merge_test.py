#!/usr/bin/env python3
"""
Run trajectory mode with pre-seeded family cache so that:
- No OmniParser/download needed (cache hit for all steps).
- Layout of step 0 and step 2 are identical -> merge into one node.

Uses server paths by default: /ext_hdd/nhkoh/dataset/GUIOdyssey/
If that is not writable, uses local datas/trajectory_merge_test/.
"""
import json
import os
import shutil
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO_ROOT)

# Prefer server path; fallback to local
BASE_SERVER = "/ext_hdd/nhkoh/dataset/GUIOdyssey"
BASE_LOCAL = os.path.join(REPO_ROOT, "datas", "trajectory_merge_test")

def _pick_base():
    if os.path.isdir("/ext_hdd/nhkoh") and os.access("/ext_hdd/nhkoh", os.W_OK):
        base = BASE_SERVER
    else:
        base = BASE_LOCAL
    os.makedirs(base, exist_ok=True)
    return base

def main():
    base = _pick_base()
    annotations_dir = os.path.join(base, "annotations")
    screenshots_dir = os.path.join(base, "screenshots")
    output_dir = os.path.join(base, "out")
    family_cache_dir = os.path.join(output_dir, "page_families")
    os.makedirs(annotations_dir, exist_ok=True)
    os.makedirs(screenshots_dir, exist_ok=True)
    os.makedirs(family_cache_dir, exist_ok=True)

    trajectory_id = "test_merge"
    # 3 steps: step0.png, step1.png, step2.png. Step 0 and 2 same layout -> merge.
    trajectory = {
        "episode_id": trajectory_id,
        "task_info": {"task": "Merge test", "instruction": "Test", "app": [], "category": ""},
        "steps": [
            {"screenshot": "step0.png", "action": "CLICK", "low_level_instruction": "click a"},
            {"screenshot": "step1.png", "action": "CLICK", "low_level_instruction": "click b"},
            {"screenshot": "step2.png", "action": "COMPLETE", "low_level_instruction": "done"},
        ],
    }
    with open(os.path.join(annotations_dir, f"{trajectory_id}.json"), "w", encoding="utf-8") as f:
        json.dump(trajectory, f, indent=2)

    # Dummy screenshots (must exist)
    for i in range(3):
        dst = os.path.join(screenshots_dir, f"step{i}.png")
        if not os.path.exists(dst):
            src = os.path.join(REPO_ROOT, "datas", "images", f"page_{i}.png")
            if os.path.exists(src):
                shutil.copy(src, dst)
            else:
                from PIL import Image
                Image.new("RGB", (400, 700), color=(200, 200, 200)).save(dst)
                print(f"Created placeholder {dst}")

    # Same layout for step 0 and step 2 so they merge
    layout_a = {"icon_a": [100, 200, 150, 250], "icon_b": [200, 200, 250, 250]}
    layout_b = {"icon_c": [150, 300, 200, 350], "icon_d": [250, 300, 300, 350]}

    from PIL import Image
    def write_family(step_idx: int, layout: dict, step: dict):
        family = {
            "page_id": f"page_{step_idx}",
            "page_family_id": f"family_{step_idx:03d}",
            "source_step_index": step_idx,
            "screenshot_name": f"step{step_idx}.png",
            "screenshot_path": os.path.join(screenshots_dir, f"step{step_idx}.png"),
            "orig_size": [400, 700],
            "canvas_size": [252, 448],
            "step": step,
            "elements": [],
            "layout": layout,
            "action_name_counts": {name: 1 for name in layout},
            "render_mode": "crop_reconstructed",
        }
        with open(os.path.join(family_cache_dir, f"family_{step_idx:03d}.json"), "w", encoding="utf-8") as f:
            json.dump(family, f, indent=2)
        # Minimal 252x448 image
        img = Image.new("RGB", (252, 448), color=(240, 240, 240))
        img.save(os.path.join(family_cache_dir, f"family_{step_idx:03d}_canonical.png"))

    write_family(0, layout_a, trajectory["steps"][0])
    write_family(1, layout_b, trajectory["steps"][1])
    write_family(2, dict(layout_a), trajectory["steps"][2])  # same as 0 -> merge

    cmd = [
        sys.executable,
        "data_engine/tree.py",
        "--icon_source", "trajectory",
        "--trajectory_id", trajectory_id,
        "--annotations_dir", annotations_dir,
        "--screenshots_dir", screenshots_dir,
        "--output_dir", output_dir,
    ]
    print("Running:", " ".join(cmd))
    r = subprocess.run(cmd, cwd=REPO_ROOT)
    if r.returncode != 0:
        print("Command failed with code", r.returncode)
        return r.returncode

    # Check merge happened
    ui_path = os.path.join(output_dir, "ui_structure.json")
    if os.path.exists(ui_path):
        with open(ui_path, "r", encoding="utf-8") as f:
            ui = json.load(f)
        pages = ui.get("pages", {})
        meta = ui.get("metadata", {})
        eff = meta.get("effective_spine_page_ids", [])
        print("\n--- Result ---")
        print("total pages (spine+branch):", len(pages))
        print("effective_spine_page_ids:", eff)
        merged = eff == ["page_0", "page_1", "page_0"]
        if merged:
            print("OK: Merge verified (step 2 merged into page_0; effective_spine = [page_0, page_1, page_0]).")
        else:
            print("Unexpected: expected effective_spine [page_0, page_1, page_0].")
    return 0

if __name__ == "__main__":
    sys.exit(main())
