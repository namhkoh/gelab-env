"""
Generate SFT training data.

Paper specifications (Appendix A.1):
- Path data from subtrees 0-1 (with page_0 as valid endpoint)
- Edge data from ALL subtrees (0-4)
- 2,320 Icon Grounding + 2,320 Icon Captioning auxiliary samples

Expected output: ~30,888 samples (24,878 path + 1,370 edge + 2,320 grounding + 2,320 captioning)
"""

import json
import os
import random
import argparse

from env_utils import GELabEnvUtils


class SFTDataGenerator(GELabEnvUtils):
    """Generate SFT training data."""

    def generate_path_samples(self, subtrees, include_page_0=True):
        """
        Generate path samples for specified subtrees.

        For each path of length N, generate N+1 samples:
        - Samples 0 to N-1: Click actions with history
        - Sample N: Complete action at target
        """
        samples = []
        idx = 0

        for subtree_idx in subtrees:
            subtree_pages = self.get_pages_in_subtrees([subtree_idx])
            subtree_pages_set = set(subtree_pages)

            valid_endpoints = list(subtree_pages)
            if include_page_0 and "page_0" not in subtree_pages_set:
                valid_endpoints.append("page_0")
                valid_endpoints.sort()

            pair_count = 0
            sample_count = 0

            for start in valid_endpoints:
                for end in valid_endpoints:
                    if start == end:
                        continue
                    if start not in subtree_pages_set and end not in subtree_pages_set:
                        continue

                    path = self.get_path_with_actions(start, end)
                    if not path:
                        continue

                    pair_count += 1
                    path_length = len(path)
                    task = f"From {start} to {end}"
                    source = f"sub{subtree_idx}_path"
                    history_steps = []

                    for step_idx, (page_id, action, bbox) in enumerate(path):
                        if step_idx == 0:
                            history = "Null"
                        else:
                            history = "; ".join(
                                [f"step{i+1}: click {h[1]} icon on {h[0]}"
                                 for i, h in enumerate(history_steps)]
                            )

                        user_content = f"<image>Instruction: from {start} to {end}. History: {history}"
                        assistant_content = self.format_click_action(action, page_id, bbox)

                        sample = {
                            "idx": idx,
                            "path": path_length,
                            "task": task,
                            "messages": [
                                {"role": "user", "content": user_content},
                                {"role": "assistant", "content": assistant_content},
                            ],
                            "images": [os.path.join(self.pages_dir, f"{page_id}.png")],
                            "bbox_norm": self.bbox_to_normalized(bbox),
                            "source": source,
                        }
                        samples.append(sample)
                        idx += 1
                        sample_count += 1
                        history_steps.append((page_id, action))

                    # Complete action at target
                    final_history = "; ".join(
                        [f"step{i+1}: click {h[1]} icon on {h[0]}"
                         for i, h in enumerate(history_steps)]
                    )
                    user_content = f"<image>Instruction: from {start} to {end}. History: {final_history}"
                    assistant_content = self.format_complete_action()

                    sample = {
                        "idx": idx,
                        "path": path_length,
                        "task": task,
                        "messages": [
                            {"role": "user", "content": user_content},
                            {"role": "assistant", "content": assistant_content},
                        ],
                        "images": [os.path.join(self.pages_dir, f"{end}.png")],
                        "bbox_norm": [0, 0, 0, 0],
                        "source": source,
                    }
                    samples.append(sample)
                    idx += 1
                    sample_count += 1

            print(f"    Subtree {subtree_idx}: {pair_count} pairs -> {sample_count} samples")

        return samples

    def generate_edge_samples(self, subtrees):
        """
        Generate edge (single-step) samples for specified subtrees.

        For each transition A->B, generates 2 samples:
        1. Click sample: At page A, action = click to reach B
        2. Complete sample: At page B, action = complete
        """
        samples = []
        idx = 0

        for subtree_idx in subtrees:
            subtree_pages = self.get_pages_in_subtrees([subtree_idx])
            transition_count = 0

            for page_id in subtree_pages:
                for target, action, bbox in self.graph.get(page_id, []):
                    task = f"From {page_id} to {target}"
                    source = f"sub{subtree_idx}_edge"

                    # Sample 1: Click action at source page
                    user_1 = f"<image>Instruction: from {page_id} to {target}. History: Null"
                    assistant_1 = self.format_click_action(action, page_id, bbox)

                    sample_1 = {
                        "idx": idx,
                        "path": 1,
                        "task": task,
                        "messages": [
                            {"role": "user", "content": user_1},
                            {"role": "assistant", "content": assistant_1},
                        ],
                        "images": [os.path.join(self.pages_dir, f"{page_id}.png")],
                        "bbox_norm": self.bbox_to_normalized(bbox),
                        "source": source,
                    }
                    samples.append(sample_1)
                    idx += 1

                    # Sample 2: Complete action at target page
                    history = f"step1: click {action} icon on {page_id}"
                    user_2 = f"<image>Instruction: from {page_id} to {target}. History: {history}"
                    assistant_2 = self.format_complete_action()

                    sample_2 = {
                        "idx": idx,
                        "path": 1,
                        "task": task,
                        "messages": [
                            {"role": "user", "content": user_2},
                            {"role": "assistant", "content": assistant_2},
                        ],
                        "images": [os.path.join(self.pages_dir, f"{target}.png")],
                        "bbox_norm": [0, 0, 0, 0],
                        "source": source,
                    }
                    samples.append(sample_2)
                    idx += 1
                    transition_count += 1

            # Add entry transitions from page_0 to subtree
            for target, action, bbox in self.graph.get("page_0", []):
                if self.page_to_subtree.get(target) == subtree_idx:
                    task = f"From page_0 to {target}"
                    source = f"sub{subtree_idx}_edge"

                    sample_1 = {
                        "idx": idx,
                        "path": 1,
                        "task": task,
                        "messages": [
                            {"role": "user", "content": f"<image>Instruction: from page_0 to {target}. History: Null"},
                            {"role": "assistant", "content": self.format_click_action(action, "page_0", bbox)},
                        ],
                        "images": [os.path.join(self.pages_dir, "page_0.png")],
                        "bbox_norm": self.bbox_to_normalized(bbox),
                        "source": source,
                    }
                    samples.append(sample_1)
                    idx += 1

                    sample_2 = {
                        "idx": idx,
                        "path": 1,
                        "task": task,
                        "messages": [
                            {"role": "user", "content": f"<image>Instruction: from page_0 to {target}. History: step1: click {action} icon on page_0"},
                            {"role": "assistant", "content": self.format_complete_action()},
                        ],
                        "images": [os.path.join(self.pages_dir, f"{target}.png")],
                        "bbox_norm": [0, 0, 0, 0],
                        "source": source,
                    }
                    samples.append(sample_2)
                    idx += 1
                    transition_count += 1

            print(f"    Subtree {subtree_idx}: {transition_count} transitions x 2 = {transition_count * 2} samples")

        return samples

    def generate_grounding_samples(self, num_samples=2320):
        """Generate Icon Grounding samples (text -> coordinates)."""
        samples = []

        all_icons = []
        for page_id, page_data in self.pages.items():
            for trans in page_data.get("transitions", []):
                action = trans["action"]
                if action in ["back", "home"]:
                    continue
                bbox = trans.get("icon_bbox", [0, 0, 0, 0])
                all_icons.append((page_id, action, bbox))

        sampled = random.choices(all_icons, k=num_samples)

        for idx, (page_id, action, bbox) in enumerate(sampled):
            cx, cy = self.bbox_center_normalized(bbox)

            sample = {
                "idx": idx,
                "task": "grounding",
                "messages": [
                    {"role": "user", "content": f"<image>Click on {action} in the image."},
                    {"role": "assistant", "content": f"Action: click(start_box='<|box_start|>({cx},{cy})<|box_end|>')"},
                ],
                "images": [os.path.join(self.pages_dir, f"{page_id}.png")],
                "bbox_norm": self.bbox_to_normalized(bbox),
                "source": "grounding",
            }
            samples.append(sample)

        return samples

    def generate_captioning_samples(self, num_samples=2320):
        """Generate Icon Captioning samples (coordinates -> text)."""
        samples = []

        all_icons = []
        for page_id, page_data in self.pages.items():
            for trans in page_data.get("transitions", []):
                action = trans["action"]
                if action in ["back", "home"]:
                    continue
                bbox = trans.get("icon_bbox", [0, 0, 0, 0])
                all_icons.append((page_id, action, bbox))

        sampled = random.choices(all_icons, k=num_samples)

        for idx, (page_id, action, bbox) in enumerate(sampled):
            cx, cy = self.bbox_center_normalized(bbox)

            sample = {
                "idx": idx,
                "task": "captioning",
                "messages": [
                    {"role": "user", "content": f"<image>What is the icon at point ({cx}, {cy}) in the image?"},
                    {"role": "assistant", "content": action},
                ],
                "images": [os.path.join(self.pages_dir, f"{page_id}.png")],
                "bbox_norm": self.bbox_to_normalized(bbox),
                "source": "captioning",
            }
            samples.append(sample)

        return samples


def main():
    parser = argparse.ArgumentParser(description="Generate SFT training data")
    parser.add_argument("--env_dir", default="datas", help="Path to environment directory")
    parser.add_argument("--output_dir", default="datas", help="Output directory")
    args = parser.parse_args()

    if not os.path.isabs(args.env_dir):
        args.env_dir = os.path.join(os.getcwd(), args.env_dir)

    random.seed(42)

    print("=" * 60)
    print("SFT DATA GENERATION")
    print("=" * 60)
    print(f"Environment: {args.env_dir}")
    print(f"Output: {args.output_dir}")
    print()

    generator = SFTDataGenerator(args.env_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    # Path samples from subtrees 0-1
    print("Path samples from subtrees 0-1...")
    path_samples = generator.generate_path_samples(subtrees=[0, 1])
    print(f"  Total path samples: {len(path_samples)}")

    # Edge samples from all subtrees
    print("Edge samples from all subtrees (0-4)...")
    edge_samples = generator.generate_edge_samples(subtrees=[0, 1, 2, 3, 4])
    print(f"  Total edge samples: {len(edge_samples)}")

    # Grounding samples
    print("Grounding samples...")
    grounding_samples = generator.generate_grounding_samples(2320)
    print(f"  Total grounding samples: {len(grounding_samples)}")

    # Captioning samples
    print("Captioning samples...")
    captioning_samples = generator.generate_captioning_samples(2320)
    print(f"  Total captioning samples: {len(captioning_samples)}")

    # Combine and reindex
    sft_samples = path_samples + edge_samples + grounding_samples + captioning_samples
    for i, s in enumerate(sft_samples):
        s["idx"] = i

    output_path = os.path.join(args.output_dir, "sft_aligned.json")
    with open(output_path, "w") as f:
        json.dump(sft_samples, f, indent=2)

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Path:       {len(path_samples)}")
    print(f"Edge:       {len(edge_samples)}")
    print(f"Grounding:  {len(grounding_samples)}")
    print(f"Captioning: {len(captioning_samples)}")
    print(f"Total SFT:  {len(sft_samples)}")
    print(f"Output:     {output_path}")


if __name__ == "__main__":
    main()
