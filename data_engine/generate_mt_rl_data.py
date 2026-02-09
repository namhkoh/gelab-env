"""
Generate MT-RL training data.

Paper specification:
- Multi-turn RL uses path tasks from subtrees 2-3
- Each task is a (start, end) pair with first-step solution
- Model navigates multi-step paths with sparse A2B reward

Expected output: 2,200 samples (1,100 per subtree)
"""

import json
import os
import random
import argparse

from env_utils import GELabEnvUtils


class MTRLDataGenerator(GELabEnvUtils):
    """Generate MT-RL training data."""

    def generate_mt_rl_samples(self, subtrees, max_per_subtree=1100):
        """Generate MT-RL samples (path starting points)."""
        samples = []
        idx = 0

        for subtree_idx in subtrees:
            subtree_pages = list(self.get_pages_in_subtrees([subtree_idx]))
            count = 0
            attempted = set()

            while count < max_per_subtree and len(attempted) < len(subtree_pages) ** 2:
                start = random.choice(subtree_pages)
                end = random.choice(subtree_pages)

                if start == end or (start, end) in attempted:
                    continue
                attempted.add((start, end))

                path = self.get_path_with_actions(start, end)
                if not path:
                    continue

                from_page, action, bbox = path[0]
                norm_bbox = self.bbox_to_normalized(bbox)
                cx, cy = self.bbox_center_normalized(bbox)

                sample = {
                    "idx": idx,
                    "path": len(path),
                    "task": f"From {start} to {end}",
                    "bbox_norm": norm_bbox,
                    "image": os.path.join(self.pages_dir, f"{from_page}.png"),
                    "problem": f"<image>Instruction: from {start} to {end}. History: Null",
                    "solution": f"Explain: click {action} icon on {from_page}.\tAction: click(start_box='<|box_start|>({cx},{cy})<|box_end|>')",
                    "source": f"sub{subtree_idx}",
                }
                samples.append(sample)
                idx += 1
                count += 1

            print(f"  Subtree {subtree_idx}: {count} MT-RL samples")

        return samples


def main():
    parser = argparse.ArgumentParser(description="Generate MT-RL training data")
    parser.add_argument("--env_dir", default="datas", help="Path to environment directory")
    parser.add_argument("--output_dir", default="datas", help="Output directory")
    parser.add_argument("--max_per_subtree", type=int, default=1100,
                        help="Max samples per subtree (default: 1100)")
    args = parser.parse_args()

    if not os.path.isabs(args.env_dir):
        args.env_dir = os.path.join(os.getcwd(), args.env_dir)

    random.seed(42)

    print("=" * 60)
    print("MT-RL DATA GENERATION")
    print("=" * 60)
    print(f"Environment: {args.env_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Max per subtree: {args.max_per_subtree}")
    print()

    generator = MTRLDataGenerator(args.env_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    print("Generating MT-RL samples (subtrees 2-3)...")
    samples = generator.generate_mt_rl_samples(
        subtrees=[2, 3],
        max_per_subtree=args.max_per_subtree,
    )

    output_path = os.path.join(args.output_dir, "mt_rl_aligned.json")
    with open(output_path, "w") as f:
        json.dump(samples, f, indent=2)

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total MT-RL: {len(samples)}")
    print(f"Output:      {output_path}")


if __name__ == "__main__":
    main()
