#!/usr/bin/env python3
"""
Run trajectory mode with real GUIOdyssey data:
1. Download one episode annotation from HuggingFace (hflqf88888/GUIOdyssey).
2. Use screenshots from: env GUIODYSSEY_SCREENSHOTS_DIR, or
   /ext_hdd/nhkoh/dataset/GUIOdyssey/screenshots (if extracted), or
   --download-screenshots to download the zip and extract only needed PNGs (slow).
3. Run tree.py with --icon_source trajectory.

Requires: OmniParser weights for detection (or pre-seeded family cache).
"""
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
os.chdir(REPO_ROOT)

# Episode: 2-step episode to minimize screenshots needed. Or pass --episode_id
DEFAULT_EPISODE_ID = "2493102722960871"
HF_REPO = "hflqf88888/GUIOdyssey"


def get_annotation(episode_id: str, out_dir: Path) -> Path:
    """Download episode annotation from HF to out_dir/annotations/{id}.json."""
    from huggingface_hub import hf_hub_download
    out_dir.mkdir(parents=True, exist_ok=True)
    ann_dir = out_dir / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)
    dest = ann_dir / f"{episode_id}.json"
    if dest.exists():
        print(f"Annotation exists: {dest}")
        return dest
    try:
        path = hf_hub_download(
            repo_id=HF_REPO,
            filename=f"annotations/{episode_id}.json",
            repo_type="dataset",
            local_dir=str(out_dir),
            local_dir_use_symlinks=False,
        )
        # hf_hub_download with local_dir puts file at out_dir/annotations/...
        dest = out_dir / "annotations" / f"{episode_id}.json"
        if not dest.exists():
            # might be in cache; copy to our dir
            import shutil
            shutil.copy(path, dest)
        print(f"Annotation saved: {dest}")
        return dest
    except Exception as e:
        print(f"Download annotation failed: {e}")
        raise


def find_screenshots_dir(episode_id: str, steps: list) -> Path | None:
    """Return a directory that contains the episode's screenshot PNGs, or None."""
    candidates = [
        os.environ.get("GUIODYSSEY_SCREENSHOTS_DIR"),
        "/ext_hdd/nhkoh/dataset/GUIOdyssey/screenshots",
        REPO_ROOT / "datas" / "guiodyssey_screenshots",
    ]
    names = {s.get("screenshot") for s in steps if s.get("screenshot")}
    for d in candidates:
        if not d:
            continue
        p = Path(d)
        if not p.is_dir():
            continue
        found = sum(1 for n in names if (p / n).exists())
        if found == len(names):
            return p
        if found > 0:
            print(f"Partial: {p} has {found}/{len(names)} screenshots")
    return None


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode_id", default=DEFAULT_EPISODE_ID, help="GUIOdyssey episode ID")
    ap.add_argument("--output_dir", default=None, help="Output dir for tree.py (default: datas/guiodyssey_real_<id>)")
    ap.add_argument("--data_dir", default=None, help="Dir for annotations + screenshots (default: datas/guiodyssey_<id>)")
    ap.add_argument("--download_screenshots", action="store_true",
                    help="Download screenshots zip from HF and extract only this episode (slow, ~7GB download)")
    ap.add_argument("--use_placeholder_screenshots", action="store_true",
                    help="Create placeholder images + family cache from annotation (real layout from sam2_bbox); run without OmniParser")
    ap.add_argument("--no_run", action="store_true", help="Only prepare annotation and check screenshots")
    args = ap.parse_args()

    episode_id = args.episode_id
    data_dir = Path(args.data_dir or REPO_ROOT / "datas" / f"guiodyssey_{episode_id}")
    data_dir.mkdir(parents=True, exist_ok=True)

    # 1. Annotation
    ann_path = get_annotation(episode_id, data_dir)
    with open(ann_path, encoding="utf-8") as f:
        ann = json.load(f)
    steps = ann.get("steps", [])
    if not steps:
        print("No steps in annotation")
        return 1
    print(f"Episode {episode_id}: {len(steps)} steps")

    # 2. Screenshots
    screenshots_dir = find_screenshots_dir(episode_id, steps)
    if not screenshots_dir and args.download_screenshots:
        print("Downloading screenshots zip from HF (this may take a long time)...")
        try:
            screenshots_dir = download_episode_screenshots(HF_REPO, episode_id, steps, data_dir)
        except Exception as e:
            print(f"Download failed: {e}")
            screenshots_dir = None
    output_dir = args.output_dir or str(REPO_ROOT / "datas" / f"guiodyssey_real_{episode_id}" / "out")
    if not screenshots_dir and args.use_placeholder_screenshots:
        print("Creating placeholder screenshots and family cache from real annotation (no OmniParser needed)...")
        screenshots_dir = build_placeholder_and_cache(episode_id, ann, steps, data_dir, output_dir)
    if not screenshots_dir:
        print("Screenshots not found. Options:")
        print("  1. Extract GUIOdyssey screenshots zip on server to /ext_hdd/nhkoh/dataset/GUIOdyssey/screenshots/")
        print("  2. Set GUIODYSSEY_SCREENSHOTS_DIR to a dir containing the episode PNGs")
        print("  3. Run with --download_screenshots (downloads ~7GB zip from HF, then extracts needed PNGs)")
        print("  4. Run with --use_placeholder_screenshots to run pipeline with real annotation + placeholder images (merge logic still applies)")
        return 1
    screenshots_dir = Path(screenshots_dir)
    print(f"Screenshots dir: {screenshots_dir}")

    if args.no_run:
        print("--no_run: skipping tree.py")
        return 0

    # 3. Run tree.py
    annotations_dir = data_dir / "annotations"
    cmd = [
        sys.executable,
        "data_engine/tree.py",
        "--icon_source", "trajectory",
        "--trajectory_id", episode_id,
        "--annotations_dir", str(annotations_dir),
        "--screenshots_dir", str(screenshots_dir),
        "--output_dir", output_dir,
    ]
    print("Running:", " ".join(cmd))
    r = subprocess.run(cmd, cwd=REPO_ROOT)
    if r.returncode != 0:
        return r.returncode
    # Summary
    ui_path = Path(output_dir) / "ui_structure.json"
    if ui_path.exists():
        with open(ui_path, encoding="utf-8") as f:
            ui = json.load(f)
        meta = ui.get("metadata", {})
        print("\n--- Result ---")
        print("total_pages:", meta.get("total_pages"))
        print("effective_spine_page_ids:", meta.get("effective_spine_page_ids"))
        print("Saved:", ui_path)
    return 0


def build_placeholder_and_cache(episode_id: str, ann: dict, steps: list, data_dir: Path, output_dir: str) -> Path:
    """Create placeholder PNGs and family cache from real annotation (sam2_bbox -> layout)."""
    from PIL import Image
    device = ann.get("device_info", {})
    w, h = device.get("w", 720), device.get("h", 1280)
    screenshots_dir = data_dir / "screenshots"
    screenshots_dir.mkdir(parents=True, exist_ok=True)
    out = Path(output_dir)
    family_cache_dir = out / "page_families"
    family_cache_dir.mkdir(parents=True, exist_ok=True)
    canvas_w, canvas_h = 252, 448

    for step_idx, step in enumerate(steps):
        name = step.get("screenshot", f"{episode_id}_{step_idx}.png")
        (screenshots_dir / name).parent.mkdir(parents=True, exist_ok=True)
        img = Image.new("RGB", (w, h), color=(240, 240, 240))
        img.save(screenshots_dir / name)

        bbox = step.get("sam2_bbox") or []
        if len(bbox) >= 4:
            x1, y1, x2, y2 = [int(bbox[i]) for i in range(4)]
            # scale to canvas
            c1 = int(x1 * canvas_w / w)
            c2 = int(y1 * canvas_h / h)
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
    return screenshots_dir


def download_episode_screenshots(repo_id: str, episode_id: str, steps: list, data_dir: Path) -> Path:
    """Download screenshots.zip, extract only episode PNGs. Requires zip to be single-file (not split)."""
    from huggingface_hub import hf_hub_download
    import zipfile
    names = [s["screenshot"] for s in steps if s.get("screenshot")]
    screenshots_dir = data_dir / "screenshots"
    screenshots_dir.mkdir(parents=True, exist_ok=True)
    zip_path = hf_hub_download(
        repo_id=repo_id,
        filename="screenshots/screenshots.zip",
        repo_type="dataset",
    )
    # Open zip and extract only our files
    with zipfile.ZipFile(zip_path, "r") as z:
        for name in names:
            try:
                z.extract(name, screenshots_dir)
            except KeyError:
                # Try without path (zip might have flat structure)
                for info in z.infolist():
                    if info.filename.endswith(name) or info.filename == name:
                        z.extract(info, screenshots_dir)
                        break
    return screenshots_dir


if __name__ == "__main__":
    sys.exit(main())
