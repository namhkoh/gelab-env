"""
Generate ST-RL training data for subtrees 2-3.

Paper specification:
- ST-RL uses Path data from RL subtrees (2-3)
- Each subtree has ~12,439 path instances
- Pre-constructed trajectories where o_{0:t} and a_{0:t-1} are given
- Model predicts a_t (next action)

Format: problem/solution for GRPO training
Expected output: ~24,878 path samples (12,439 per subtree)
"""

import os
import json
import random
import argparse

from env_utils import GELabEnvUtils


class STRLDataGenerator(GELabEnvUtils):
    """Generate ST-RL training data with Path data from RL subtrees."""

    def generate_st_rl_path_data(self, subtrees, target_samples=12439):
        """
        Generate ST-RL Path data for specified subtrees.

        For each path, generate samples for each step in the trajectory.
        Each sample has the history of previous steps and asks model to predict next action.
        """
        samples_per_subtree = {s: [] for s in subtrees}
        idx = 0

        subtree_pages = list(self.get_pages_in_subtrees(subtrees))
        print(f"Pages in subtrees {subtrees}: {len(subtree_pages)}")

        # Generate paths between all pairs
        all_paths = []
        for start in subtree_pages:
            for end in subtree_pages:
                if start == end:
                    continue
                path = self.get_path_with_actions(start, end)
                if path:
                    all_paths.append((start, end, path))

        print(f"Total unique paths found: {len(all_paths)}")

        for start, end, path in all_paths:
            path_len = len(path)
            subtree_idx = self.page_to_subtree.get(start, 2)

            for step_idx, (from_page, action, bbox) in enumerate(path):
                if step_idx == 0:
                    history = "Null"
                else:
                    history_steps = []
                    for h_idx in range(step_idx):
                        h_page, h_action, _ = path[h_idx]
                        history_steps.append(f"step{h_idx+1}: click {h_action} icon")
                    history = "; ".join(history_steps)

                norm_bbox = self.bbox_to_normalized(bbox)
                cx, cy = self.bbox_center_normalized(bbox)

                sample = {
                    "idx": idx,
                    "path_length": path_len,
                    "step": step_idx + 1,
                    "bbox_norm": norm_bbox,
                    "image": os.path.join(self.pages_dir, f"{from_page}.png"),
                    "problem": f"<image>Instruction: from {start} to {end}. History: {history}",
                    "solution": f"Explain: click {action} icon on {from_page}.\tAction: click(start_box='<|box_start|>({cx},{cy})<|box_end|>')",
                    "source": f"sub{subtree_idx}_path",
                }
                samples_per_subtree[subtree_idx].append(sample)
                idx += 1

        # Balance and limit samples per subtree
        samples = []
        for subtree_idx in subtrees:
            subtree_samples = samples_per_subtree.get(subtree_idx, [])
            if len(subtree_samples) > target_samples:
                subtree_samples = random.sample(subtree_samples, target_samples)
            samples.extend(subtree_samples)
            print(f"  Subtree {subtree_idx}: {len(subtree_samples)} path samples")

        # Reindex and shuffle
        for i, sample in enumerate(samples):
            sample["idx"] = i
        random.shuffle(samples)

        return samples


def main():
    parser = argparse.ArgumentParser(description="Generate ST-RL training data")
    parser.add_argument("--env_dir", default="datas", help="Path to environment directory")
    parser.add_argument("--output_dir", default="datas", help="Output directory")
    parser.add_argument("--target_samples", type=int, default=12439,
                        help="Target samples per subtree (paper: 12,439)")
    args = parser.parse_args()

    if not os.path.isabs(args.env_dir):
        args.env_dir = os.path.join(os.getcwd(), args.env_dir)

    random.seed(42)

    print("=" * 60)
    print("ST-RL DATA GENERATION")
    print("=" * 60)
    print(f"Environment: {args.env_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Target samples per subtree: {args.target_samples}")
    print()

    generator = STRLDataGenerator(args.env_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    print("Generating ST-RL Path data (subtrees 2-3)...")
    path_samples = generator.generate_st_rl_path_data(
        subtrees=[2, 3],
        target_samples=args.target_samples,
    )

    output_path = os.path.join(args.output_dir, "st_rl_path_only.json")
    with open(output_path, "w") as f:
        json.dump(path_samples, f, indent=2)

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Path samples: {len(path_samples)}")
    print(f"Output:       {output_path}")


if __name__ == "__main__":
    main()
