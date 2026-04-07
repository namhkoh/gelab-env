"""
Fix lift_coord in amex_sft_hf_448 ui_structure.json files:
  Scale from original 1080x2400 to 448x448 space.

Then upload the fixed dataset to HuggingFace.

Usage:
    python data_engine/fix_lift_coord_and_upload.py \
        --env_dir /home/irteam/data-vol1/amex_sft_hf_448 \
        --hf_repo luca0621/amex-gelab-448 \
        --dry_run   # preview changes without writing

    python data_engine/fix_lift_coord_and_upload.py \
        --env_dir /home/irteam/data-vol1/amex_sft_hf_448 \
        --hf_repo luca0621/amex-gelab-448
"""

import argparse
import json
import os

ORIG_W, ORIG_H = 1080, 2400
TARGET_W, TARGET_H = 448, 448


def scale_coord(coord, orig_w=ORIG_W, orig_h=ORIG_H, target_w=TARGET_W, target_h=TARGET_H):
    x, y = coord[0], coord[1]
    return [int(round(x * target_w / orig_w)), int(round(y * target_h / orig_h))]


def fix_ui_structure(ui_structure):
    fixed = 0
    for page_id, page in ui_structure.get("pages", {}).items():
        for t in page.get("transitions", []):
            if t.get("action") in ("SWIPE", "SCROLL"):
                lc = t.get("lift_coord")
                if lc and (lc[0] > TARGET_W or lc[1] > TARGET_H):
                    t["lift_coord"] = scale_coord(lc)
                    fixed += 1
    # Update metadata
    meta = ui_structure.get("metadata", {})
    if meta.get("canvas_size") == [ORIG_W, ORIG_H]:
        meta["canvas_size"] = [TARGET_W, TARGET_H]
    if meta.get("phone_canvas_size") == [ORIG_W, ORIG_H - 72]:
        meta["phone_canvas_size"] = [TARGET_W, TARGET_H]
    if "nav_strip_height" in meta:
        meta["nav_strip_height"] = int(round(meta["nav_strip_height"] * TARGET_H / ORIG_H))
    return fixed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_dir", type=str, required=True)
    parser.add_argument("--hf_repo", type=str, default="luca0621/amex-gelab-448")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    traj_dirs = sorted([
        d for d in os.listdir(args.env_dir)
        if os.path.isdir(os.path.join(args.env_dir, d))
    ])

    total_fixed = 0
    total_files = 0

    for traj in traj_dirs:
        ui_path = os.path.join(args.env_dir, traj, "ui_structure.json")
        if not os.path.exists(ui_path):
            continue

        with open(ui_path, "r", encoding="utf-8") as f:
            ui = json.load(f)

        fixed = fix_ui_structure(ui)
        total_files += 1
        total_fixed += fixed

        if fixed > 0 and not args.dry_run:
            with open(ui_path, "w", encoding="utf-8") as f:
                json.dump(ui, f, indent=2)

        if fixed > 0 and total_fixed <= 5:
            print(f"  [{traj}] fixed {fixed} lift_coord(s)")

    print(f"\nTotal: {total_fixed} lift_coords fixed across {total_files} files")

    if args.dry_run:
        print("\n[DRY RUN] No files written. Remove --dry_run to apply.")
        return

    # Upload to HuggingFace
    print(f"\nUploading to {args.hf_repo}...")
    from huggingface_hub import HfApi
    api = HfApi()
    token = os.environ.get("HF_TOKEN", "")
    #working

    api.upload_folder(
        folder_path=args.env_dir,
        repo_id=args.hf_repo,
        repo_type="dataset",
        token=token,
        commit_message="fix: scale lift_coord from 1080x2400 to 448x448",
    )
    print(f"Upload complete: https://huggingface.co/datasets/{args.hf_repo}")


if __name__ == "__main__":
    main()
