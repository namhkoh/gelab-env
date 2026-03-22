"""
Collect SFT data from all completed AMEX trajectory compositions.

Incrementally generates SFT samples from newly completed trajectories
and merges them into a single sft_amex.json. Tracks processed trajectories
in sft_processed.json to avoid re-processing on subsequent runs.

Usage:
    python data_engine/collect_amex_sft.py \
        --env_dir data_engine/sim2real_envs/amex_sft \
        --anno_dir /ext_hdd2/tsyou/AMEX_dataset/AMEX/instruction_anno
"""

import argparse
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_amex_sft_data import (
    generate_trajectory_samples,
    generate_grounding_samples,
    generate_captioning_samples,
)

ANNO_DIRS = [
    "/ext_hdd2/tsyou/AMEX_dataset/AMEX/instruction_anno",
    "/ext_hdd2/tsyou/AMEX_dataset/AMEX/instruction_anno_1",
    "/ext_hdd2/tsyou/AMEX_dataset/AMEX/instruction_anno_2",
    "/ext_hdd2/tsyou/AMEX_dataset/AMEX/instruction_anno_3",
    "/ext_hdd2/tsyou/AMEX_dataset/AMEX/instruction_anno_4",
    "/ext_hdd2/tsyou/AMEX_dataset/AMEX/instruction_anno_5",
    "/ext_hdd2/tsyou/AMEX_dataset/AMEX/instruction_anno_6",
]


def find_annotation(episode_id: str) -> str:
    """Search all annotation directories for a matching trajectory."""
    for anno_dir in ANNO_DIRS:
        if not os.path.isdir(anno_dir):
            continue
        for f in os.listdir(anno_dir):
            if episode_id in f and f.endswith(".json"):
                return os.path.join(anno_dir, f)
    return ""


def main():
    parser = argparse.ArgumentParser(description="Collect SFT data from completed trajectory compositions")
    parser.add_argument("--env_dir", default="data_engine/sim2real_envs/amex_sft",
                        help="Root directory with composed trajectory subdirectories")
    parser.add_argument("--grounding_per_traj", type=int, default=20)
    parser.add_argument("--captioning_per_traj", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    env_dir = args.env_dir

    processed_path = os.path.join(env_dir, "sft_processed.json")
    output_path = os.path.join(env_dir, "sft_amex.json")

    already_processed = set()
    existing_samples = []
    if os.path.exists(processed_path):
        with open(processed_path) as f:
            already_processed = set(json.load(f))
    if os.path.exists(output_path):
        with open(output_path) as f:
            existing_samples = json.load(f)

    completed = []
    for d in sorted(os.listdir(env_dir)):
        ui_path = os.path.join(env_dir, d, "ui_structure.json")
        if os.path.isfile(ui_path):
            completed.append(d)

    new_trajs = [t for t in completed if t not in already_processed]

    print(f"Total completed trajectories: {len(completed)}")
    print(f"Already in SFT dataset:       {len(already_processed)}")
    print(f"New to process:               {len(new_trajs)}")

    if not new_trajs:
        print("Nothing new to process.")
        return

    new_samples = []
    errors = 0
    cwd = os.getcwd()

    for i, episode_id in enumerate(new_trajs):
        traj_dir = os.path.join(env_dir, episode_id)
        ui_path = os.path.join(traj_dir, "ui_structure.json")
        pages_dir = os.path.join(traj_dir, "pages")
        anno_path = find_annotation(episode_id)

        if not anno_path:
            print(f"  SKIP {episode_id}: annotation not found")
            errors += 1
            continue

        try:
            with open(ui_path) as f:
                ui = json.load(f)
            with open(anno_path) as f:
                traj = json.load(f)

            source = f"amex_{episode_id}"
            nav = generate_trajectory_samples(ui, traj, pages_dir, source)
            grounding = generate_grounding_samples(ui, pages_dir, source, args.grounding_per_traj)
            captioning = generate_captioning_samples(ui, pages_dir, source, args.captioning_per_traj)

            samples = nav + grounding + captioning
            for s in samples:
                s["images"] = [os.path.join(cwd, p) if not os.path.isabs(p) else p for p in s["images"]]

            new_samples.extend(samples)
            already_processed.add(episode_id)

            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{len(new_trajs)}] new samples: {len(new_samples)}")
        except Exception as e:
            print(f"  ERROR {episode_id}: {e}")
            errors += 1

    all_samples = existing_samples + new_samples
    for i, s in enumerate(all_samples):
        s["idx"] = i

    with open(output_path, "w") as f:
        json.dump(all_samples, f, indent=2)
    with open(processed_path, "w") as f:
        json.dump(sorted(already_processed), f, indent=2)

    types = Counter(s.get("action_type", "?") for s in all_samples)
    print(f"\n=== SFT DATASET UPDATED ===")
    print(f"Previous samples:  {len(existing_samples)}")
    print(f"New samples:       {len(new_samples)} (from {len(new_trajs) - errors} trajectories)")
    print(f"Total samples:     {len(all_samples)}")
    print(f"Errors:            {errors}")
    print(f"Action types:")
    for t, c in types.most_common():
        print(f"  {t:20s}: {c}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
