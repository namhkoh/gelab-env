"""Build nav-only SFT datasets from successful AMEX trajectories only.

Success = last step action == TASK_COMPLETE (2,828/3,046 trajectories).
Two outputs:
  - pb_success_nav_orig.json : nav samples from non-augmented AMEX pages
  - pb_success_nav_aug.json  : nav samples from augmented AMEX pages (shift/bgcolor/popup)

Both have absolute image paths for cross-cwd robustness.
"""
from __future__ import annotations
import glob
import json
import os
import re
from collections import Counter

ANNO_DIR = "/workspace/gelab-env/datas/amex_raw/AMEX/instruction_anno"
ORIG_SFT = "/workspace/gelab-env/datas/amex_gelab_sft_full.json"
AUG_ROOT = "/workspace/gelab-env/datas_amex/amex-augmented-sft"
AUG_SFT  = f"{AUG_ROOT}/train.json"
OUT_DIR = "/workspace/gelab-env/datas_amex"

TRAJ_HASH_RE = re.compile(r"([0-9a-f]{32})_")


def traj_id(sample: dict) -> str | None:
    for p in sample.get("images", []):
        m = TRAJ_HASH_RE.search(os.path.basename(p))
        if m: return m.group(1)
    return None


def absolutize(samples):
    for s in samples:
        s["images"] = [
            p if os.path.isabs(p) else os.path.join(AUG_ROOT, p)
            for p in s["images"]
        ]
    return samples


# 1. Build success set
print("Scanning AMEX annotations for successful trajectories...")
success = set()
last_actions = Counter()
for f in glob.glob(ANNO_DIR + "/*.json"):
    d = json.load(open(f))
    steps = d.get("steps", [])
    if not steps:
        continue
    last = steps[-1].get("action")
    last_actions[last] += 1
    if last == "TASK_COMPLETE":
        success.add(d["episode_id"])
print(f"  success: {len(success):,}  (last-action dist: {dict(last_actions)})")

# 2. Filter orig nav
print("\nFiltering orig nav...")
orig = json.load(open(ORIG_SFT))
orig_nav = [s for s in orig if s.get("source") == "amex_gelab_nav"]
orig_nav_success = [s for s in orig_nav if (t := traj_id(s)) and t in success]
unique_trajs = {traj_id(s) for s in orig_nav_success}
print(f"  orig_nav total: {len(orig_nav):,}")
print(f"  orig_nav from successful trajectories: {len(orig_nav_success):,} "
      f"(from {len(unique_trajs):,} unique trajectories)")
out = f"{OUT_DIR}/pb_success_nav_orig.json"
json.dump(orig_nav_success, open(out, "w"))
print(f"  -> wrote {out}")

# 3. Filter aug nav
print("\nFiltering aug nav...")
aug = json.load(open(AUG_SFT))
aug_nav = [s for s in aug if s.get("source") == "amex_augmented_nav"]
aug_nav_success = [s for s in aug_nav if (t := traj_id(s)) and t in success]
aug_nav_success = absolutize(aug_nav_success)
unique_trajs = {traj_id(s) for s in aug_nav_success}
miss = sum(1 for s in aug_nav_success[:500] if not os.path.exists(s["images"][0]))
print(f"  aug_nav total: {len(aug_nav):,}")
print(f"  aug_nav from successful trajectories: {len(aug_nav_success):,} "
      f"(from {len(unique_trajs):,} unique trajectories)")
print(f"  first-500 missing images: {miss}")
out = f"{OUT_DIR}/pb_success_nav_aug.json"
json.dump(aug_nav_success, open(out, "w"))
print(f"  -> wrote {out}")

print("\nSummary")
print(f"  successful trajectories (TASK_COMPLETE):  {len(success):,} / 3,046 "
      f"({len(success)/3046*100:.1f}%)")
print(f"  pb_success_nav_orig.json samples:  {len(orig_nav_success):,}")
print(f"  pb_success_nav_aug.json samples:   {len(aug_nav_success):,}")
