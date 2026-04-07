"""
Generate ST-RL data for AMEX matching sft_amex.json coordinate space (672x1512).

Coordinates are scaled from raw 1080x2400 canvas to 672x1512 resized space,
matching Qwen2.5-VL smart_resize output for max_pixels=1048576.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_amex_st_rl_data import generate_episode_samples, _episode_dirs


SRC_W, SRC_H = 1080, 2400
DST_W, DST_H = 672, 1512
SX = DST_W / SRC_W
SY = DST_H / SRC_H


def scale_bbox(bbox):
    if not bbox or len(bbox) != 4 or bbox == [0, 0, 0, 0]:
        return bbox
    return [
        round(bbox[0] * SX), round(bbox[1] * SY),
        round(bbox[2] * SX), round(bbox[3] * SY),
    ]


def scale_coord_str(m):
    x, y = int(m.group(1)), int(m.group(2))
    return f"({round(x * SX)},{round(y * SY)})"


COORD_RE = re.compile(r"\((\d+),(\d+)\)")


def scale_solution(sol):
    if not sol:
        return sol
    return COORD_RE.sub(scale_coord_str, sol)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_dir", default="/home/irteam/data-vol1/amex_sft")
    parser.add_argument("--output", default="/home/irteam/data-vol1/gelab-env/datas/st_rl_amex.json")
    args = parser.parse_args()

    episode_dirs = _episode_dirs(args.env_dir)
    print(f"Found {len(episode_dirs)} episodes")

    all_samples = []
    errors = 0
    for i, ep_dir in enumerate(episode_dirs):
        try:
            samples = generate_episode_samples(ep_dir)
            for s in samples:
                s["bbox_px"] = scale_bbox(s.get("bbox_px"))
                s["start_box_px"] = scale_bbox(s.get("start_box_px"))
                s["end_box_px"] = scale_bbox(s.get("end_box_px"))
                s["solution"] = scale_solution(s.get("solution"))
            all_samples.extend(samples)
        except Exception as e:
            errors += 1
            print(f"[error] {os.path.basename(ep_dir)}: {e}")
        if (i + 1) % 500 == 0:
            print(f"  processed {i+1}/{len(episode_dirs)} ({len(all_samples)} samples)")

    for idx, s in enumerate(all_samples):
        s["idx"] = idx

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_samples, f, indent=2, ensure_ascii=False)

    print()
    print("=" * 60)
    print(f"Episodes: {len(episode_dirs) - errors}/{len(episode_dirs)}")
    print(f"Samples:  {len(all_samples)}")
    print(f"Output:   {args.output}")
    print(f"Coord:    1080x2400 -> 672x1512")


if __name__ == "__main__":
    main()

