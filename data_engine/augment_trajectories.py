"""
Trajectory-level data augmentation orchestrator for amex-gelab.

Applies a random subset (>= 1) of {shift, bgcolor, popup} augmentations
globally to each trajectory, writes augmented copies to an output directory,
and produces an augmentation log.

Usage:
    # Test on 1 trajectory
    python data_engine/augment_trajectories.py \
        --input_dir datas/amex_gelab_extracted/amex_sft \
        --output_dir /tmp/aug_test \
        --max_trajectories 1 --seed 42

    # Full run (8 parallel workers)
    python data_engine/augment_trajectories.py \
        --input_dir datas/amex_gelab_extracted/amex_sft \
        --output_dir datas/amex_gelab_augmented \
        --seed 42 --workers 8

    # Dry run (compute assignments only, no I/O)
    python data_engine/augment_trajectories.py \
        --input_dir datas/amex_gelab_extracted/amex_sft \
        --output_dir /tmp/dry --dry_run
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import shutil
import time
from datetime import datetime, timezone
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from page_augmentor import app_bgcolor, app_shift, insert_popup_page

# ===================================================================
#  Constants
# ===================================================================

BG_PALETTE = [
    "#D6E4F0",  # soft blue
    "#E8D5E0",  # soft mauve
    "#D4E8D1",  # soft mint
    "#F5E6CC",  # warm sand
    "#E0E0E0",  # light grey
    "#D5DDE8",  # steel blue
    "#F0DDD4",  # peach
    "#E6E2D3",  # warm cream
]

POPUP_TEXTS = [
    "Allow access to your location?",
    "Allow notifications?",
    "Allow access to contacts?",
    "Allow access to storage?",
    "Allow camera access?",
    "Allow microphone access?",
    "Enable Bluetooth?",
    "Allow this app to make calls?",
    "Allow access to photos?",
    "This app requires your permission",
]

AUGMENTATION_ORDER = ["shift", "bgcolor", "popup"]


# ===================================================================
#  Selection helpers
# ===================================================================

def select_augmentations(rng: random.Random) -> List[str]:
    """Select a random non-empty subset of augmentations.

    Each augmentation has an independent 50% chance.  If none are selected,
    one is chosen at random (guaranteeing >= 1).
    """
    chosen = [a for a in AUGMENTATION_ORDER if rng.random() < 0.5]
    if not chosen:
        chosen = [rng.choice(AUGMENTATION_ORDER)]
    return sorted(chosen, key=AUGMENTATION_ORDER.index)


def select_popup_target(
    structure: dict, rng: random.Random,
) -> Optional[str]:
    """Pick a random page eligible for popup insertion.

    Eligible pages:
      - Not page_0 (no predecessor to rewire)
      - Not the final page (TASK_COMPLETE / TASK_IMPOSSIBLE)
      - Has a predecessor whose forward transition (transitions[0]) targets it

    Prefers depth 1-3 (where permission dialogs naturally appear) with 70%
    probability; otherwise any eligible page.
    """
    pages = structure.get("pages", {})
    if len(pages) < 3:
        return None

    # Build forward-transition target set (what each page's transitions[0] points to)
    forward_targets: Dict[str, str] = {}
    for pid, pdata in pages.items():
        trans = pdata.get("transitions", [])
        if trans:
            t0 = trans[0]
            if t0.get("action") not in ("PRESS_BACK", "PRESS_HOME",
                                         "TASK_COMPLETE", "TASK_IMPOSSIBLE"):
                forward_targets[pid] = t0["target_page"]

    reachable = set(forward_targets.values())

    # Find last page (has TASK_COMPLETE or TASK_IMPOSSIBLE as transitions[0])
    last_pages = set()
    for pid, pdata in pages.items():
        trans = pdata.get("transitions", [])
        if trans and trans[0].get("action") in ("TASK_COMPLETE", "TASK_IMPOSSIBLE"):
            last_pages.add(pid)

    eligible = []
    shallow = []  # depth 1-3
    for pid, pdata in pages.items():
        if pid == "page_0":
            continue
        if pid in last_pages:
            continue
        if pid not in reachable:
            continue
        eligible.append(pid)
        depth = pdata.get("depth", 99)
        if 1 <= depth <= 3:
            shallow.append(pid)

    if not eligible:
        return None

    # Prefer shallow pages 70% of the time
    if shallow and rng.random() < 0.7:
        return rng.choice(shallow)
    return rng.choice(eligible)


def _trajectory_seed(global_seed: int, episode_id: str) -> int:
    """Deterministic per-trajectory seed from global seed + episode id."""
    h = hashlib.sha256(f"{global_seed}:{episode_id}".encode()).hexdigest()
    return int(h[:8], 16)


# ===================================================================
#  Single-trajectory augmentation
# ===================================================================

def augment_single_trajectory(
    episode_id: str,
    input_base: str,
    output_base: str,
    augmentations: List[str],
    seed: int,
    bg_color: Optional[str],
    popup_text: Optional[str],
    popup_target: Optional[str],
) -> Dict[str, Any]:
    """Apply augmentations to one trajectory, writing results to output_base."""
    input_dir = os.path.join(input_base, episode_id)
    output_dir = os.path.join(output_base, episode_id)
    input_pages = os.path.join(input_dir, "pages")
    output_pages = os.path.join(output_dir, "pages")

    # Load structure
    ui_path = os.path.join(input_dir, "ui_structure.json")
    with open(ui_path) as f:
        structure = json.load(f)

    num_pages_original = len(structure.get("pages", {}))

    # Prepare output directory
    os.makedirs(output_pages, exist_ok=True)

    # Copy all page images to output (augmentations overwrite in-place)
    for fname in os.listdir(input_pages):
        if fname.endswith(".png"):
            shutil.copy2(os.path.join(input_pages, fname), output_pages)

    # Symlink extracted_assets
    src_assets = os.path.join(input_dir, "extracted_assets")
    dst_assets = os.path.join(output_dir, "extracted_assets")
    if os.path.exists(src_assets) and not os.path.exists(dst_assets):
        os.symlink(os.path.abspath(src_assets), dst_assets)

    # Copy trajectory_assets_manifest.json
    manifest_src = os.path.join(input_dir, "trajectory_assets_manifest.json")
    if os.path.exists(manifest_src):
        shutil.copy2(manifest_src, output_dir)

    # Collect all page ids for global augmentation
    all_page_ids = list(structure["pages"].keys())

    # --- Apply augmentations in order ---

    if "shift" in augmentations:
        structure = app_shift(
            structure, input_pages, output_pages,
            all_page_ids, env_dir=output_dir, seed=seed,
        )

    if "bgcolor" in augmentations:
        structure = app_bgcolor(
            structure, output_pages, output_pages,
            all_page_ids, target_color=bg_color, tolerance=50,
        )

    if "popup" in augmentations and popup_target:
        structure = insert_popup_page(
            structure, input_pages, output_pages,
            popup_target, popup_text=popup_text,
        )

    # Write augmented ui_structure.json
    with open(os.path.join(output_dir, "ui_structure.json"), "w") as f:
        json.dump(structure, f, indent=2)

    num_pages_augmented = len(structure.get("pages", {}))

    return {
        "episode_id": episode_id,
        "augmentations": augmentations,
        "seed": seed,
        "bgcolor_target": bg_color if "bgcolor" in augmentations else None,
        "popup_page": popup_target if "popup" in augmentations else None,
        "popup_text": popup_text if "popup" in augmentations else None,
        "num_pages_original": num_pages_original,
        "num_pages_augmented": num_pages_augmented,
        "status": "ok",
    }


def _worker(args: Tuple) -> Dict[str, Any]:
    """Multiprocessing wrapper."""
    try:
        return augment_single_trajectory(*args)
    except Exception as e:
        return {
            "episode_id": args[0],
            "augmentations": args[3],
            "status": f"error: {e}",
        }


# ===================================================================
#  Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Trajectory-level data augmentation for amex-gelab",
    )
    parser.add_argument(
        "--input_dir", default="datas/amex_gelab_extracted/amex_sft",
        help="Directory with <episode_id>/ trajectory folders",
    )
    parser.add_argument(
        "--output_dir", default="datas/amex_gelab_augmented",
        help="Output directory for augmented trajectories",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--max_trajectories", type=int, default=None,
        help="Process at most N trajectories (for testing)",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Only compute augmentation assignments, no I/O",
    )
    args = parser.parse_args()

    # Discover trajectories
    input_dir = args.input_dir
    episode_ids = sorted([
        d for d in os.listdir(input_dir)
        if os.path.isdir(os.path.join(input_dir, d))
        and os.path.exists(os.path.join(input_dir, d, "ui_structure.json"))
    ])
    if args.max_trajectories:
        episode_ids = episode_ids[:args.max_trajectories]

    print(f"[augment] Found {len(episode_ids)} trajectories in {input_dir}")
    print(f"[augment] Output: {args.output_dir}")
    print(f"[augment] Seed: {args.seed}, Workers: {args.workers}")
    os.makedirs(args.output_dir, exist_ok=True)

    # Pre-compute augmentation assignments
    aug_counts = {"shift": 0, "bgcolor": 0, "popup": 0}
    work_items: List[Tuple] = []

    for eid in episode_ids:
        tseed = _trajectory_seed(args.seed, eid)
        rng = random.Random(tseed)

        augmentations = select_augmentations(rng)

        # Select bgcolor target
        bg_color = rng.choice(BG_PALETTE) if "bgcolor" in augmentations else None

        # Select popup target page
        popup_target = None
        popup_text = None
        if "popup" in augmentations:
            ui_path = os.path.join(input_dir, eid, "ui_structure.json")
            with open(ui_path) as f:
                structure = json.load(f)
            popup_target = select_popup_target(structure, rng)
            if popup_target:
                popup_text = rng.choice(POPUP_TEXTS)
            else:
                # No eligible page — remove popup from this trajectory
                augmentations = [a for a in augmentations if a != "popup"]
                if not augmentations:
                    augmentations = [rng.choice(["shift", "bgcolor"])]

        for a in augmentations:
            aug_counts[a] += 1

        work_items.append((
            eid, input_dir, args.output_dir,
            augmentations, tseed, bg_color, popup_text, popup_target,
        ))

    # Report assignments
    print(f"\n[augment] Augmentation distribution:")
    for aug, count in aug_counts.items():
        pct = count / len(episode_ids) * 100 if episode_ids else 0
        print(f"  {aug}: {count} ({pct:.1f}%)")

    combo_counts: Dict[str, int] = {}
    for item in work_items:
        key = "+".join(item[3])
        combo_counts[key] = combo_counts.get(key, 0) + 1
    print(f"\n[augment] Combination distribution:")
    for combo, count in sorted(combo_counts.items(), key=lambda x: -x[1]):
        print(f"  {combo}: {count}")

    if args.dry_run:
        print("\n[augment] Dry run — no files written.")
        return

    # Process trajectories
    t0 = time.time()
    results: List[Dict[str, Any]] = []

    if args.workers <= 1:
        for i, item in enumerate(work_items):
            result = _worker(item)
            results.append(result)
            if (i + 1) % 50 == 0 or (i + 1) == len(work_items):
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (len(work_items) - i - 1) / rate if rate > 0 else 0
                print(f"[augment] {i+1}/{len(work_items)} "
                      f"({elapsed:.0f}s elapsed, ETA {eta:.0f}s)")
    else:
        with Pool(args.workers) as pool:
            for i, result in enumerate(pool.imap_unordered(_worker, work_items)):
                results.append(result)
                if (i + 1) % 100 == 0 or (i + 1) == len(work_items):
                    elapsed = time.time() - t0
                    rate = (i + 1) / elapsed
                    eta = (len(work_items) - i - 1) / rate if rate > 0 else 0
                    print(f"[augment] {i+1}/{len(work_items)} "
                          f"({elapsed:.0f}s elapsed, ETA {eta:.0f}s)")

    elapsed = time.time() - t0

    # Count results
    ok = sum(1 for r in results if r.get("status") == "ok")
    errors = [r for r in results if r.get("status", "").startswith("error")]

    print(f"\n[augment] Done: {ok}/{len(results)} ok, {len(errors)} errors "
          f"in {elapsed:.1f}s")
    if errors:
        for e in errors[:10]:
            print(f"  ERROR {e['episode_id']}: {e['status']}")

    # Write augmentation log
    log = {
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "total_trajectories": len(results),
        "augmentation_counts": aug_counts,
        "combination_counts": combo_counts,
        "ok": ok,
        "errors": len(errors),
        "elapsed_seconds": round(elapsed, 1),
        "trajectories": {r["episode_id"]: r for r in results},
    }
    log_path = os.path.join(args.output_dir, "augmentation_log.json")
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"[augment] Log written to {log_path}")


if __name__ == "__main__":
    main()
