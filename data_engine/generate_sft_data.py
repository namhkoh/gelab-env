"""
Generate SFT training data.

Supports:
- Legacy paper-aligned subtree environments
- GT-spine-with-branches environments built from GUIOdyssey trajectories
"""

import json
import os
import random
import argparse

from env_utils import GELabEnvUtils


class SFTDataGenerator(GELabEnvUtils):
    """Generate SFT training data."""

    def is_gt_spine_topology(self):
        return self.metadata.get("topology_type") == "gt_spine_branches"

    def _goal_text(self):
        task = str(self.metadata.get("trajectory_task") or self.metadata.get("task") or "").strip()
        instruction = str(self.metadata.get("trajectory_instruction") or "").strip()
        if task and instruction and instruction != task:
            return f"Goal: {task} Context: {instruction}"
        return f"Goal: {task or instruction}" if (task or instruction) else ""

    def _task_label(self, fallback):
        return self.metadata.get("trajectory_task") or self.metadata.get("task") or fallback

    def _format_user_content(self, route_text, history):
        parts = ["<image>"]
        goal_text = self._goal_text()
        if goal_text:
            parts.append(goal_text)
        parts.append(route_text)
        parts.append(f"History: {history}")
        return " ".join(parts)

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
                    route_text = f"Instruction: from {start} to {end}."
                    task = self._task_label(f"From {start} to {end}")
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

                        user_content = self._format_user_content(route_text, history)
                        assistant_content = self.format_click_action(action, page_id, bbox)

                        sample = {
                            "idx": idx,
                            "path": path_length,
                            "task": task,
                            "route": f"From {start} to {end}",
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
                    user_content = self._format_user_content(route_text, final_history)
                    assistant_content = self.format_complete_action()

                    sample = {
                        "idx": idx,
                        "path": path_length,
                        "task": task,
                        "route": f"From {start} to {end}",
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

    def generate_gt_spine_path_samples(self):
        """Generate path samples directly from the GT-spine-plus-branches graph."""
        samples = []
        idx = 0
        page_ids = self.get_sorted_page_ids()
        pair_count = 0

        for start in page_ids:
            for end in page_ids:
                if start == end:
                    continue

                path = self.get_path_with_actions(start, end)
                if not path:
                    continue

                pair_count += 1
                path_length = len(path)
                route_text = f"Instruction: from {start} to {end} within the current app flow."
                task = self._task_label(f"From {start} to {end}")
                history_steps = []

                for step_idx, (page_id, action, bbox) in enumerate(path):
                    if step_idx == 0:
                        history = "Null"
                    else:
                        history = "; ".join(
                            [f"step{i+1}: click {h[1]} icon on {h[0]}" for i, h in enumerate(history_steps)]
                        )

                    sample = {
                        "idx": idx,
                        "path": path_length,
                        "task": task,
                        "route": f"From {start} to {end}",
                        "messages": [
                            {"role": "user", "content": self._format_user_content(route_text, history)},
                            {"role": "assistant", "content": self.format_click_action(action, page_id, bbox)},
                        ],
                        "images": [os.path.join(self.pages_dir, self.pages[page_id]["image"])],
                        "bbox_norm": self.bbox_to_normalized(bbox),
                        "source": "gt_spine_path",
                    }
                    samples.append(sample)
                    idx += 1
                    history_steps.append((page_id, action))

                final_history = "; ".join(
                    [f"step{i+1}: click {h[1]} icon on {h[0]}" for i, h in enumerate(history_steps)]
                ) or "Null"
                samples.append({
                    "idx": idx,
                    "path": path_length,
                    "task": task,
                    "route": f"From {start} to {end}",
                    "messages": [
                        {"role": "user", "content": self._format_user_content(route_text, final_history)},
                        {"role": "assistant", "content": self.format_complete_action()},
                    ],
                    "images": [os.path.join(self.pages_dir, self.pages[end]["image"])],
                    "bbox_norm": [0, 0, 0, 0],
                    "source": "gt_spine_path",
                })
                idx += 1

        print(f"    GT spine paths: {pair_count} reachable pairs -> {len(samples)} samples")
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
                    route_text = f"Instruction: from {page_id} to {target}."
                    task = self._task_label(f"From {page_id} to {target}")
                    source = f"sub{subtree_idx}_edge"

                    # Sample 1: Click action at source page
                    user_1 = self._format_user_content(route_text, "Null")
                    assistant_1 = self.format_click_action(action, page_id, bbox)

                    sample_1 = {
                        "idx": idx,
                        "path": 1,
                        "task": task,
                        "route": f"From {page_id} to {target}",
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
                    user_2 = self._format_user_content(route_text, history)
                    assistant_2 = self.format_complete_action()

                    sample_2 = {
                        "idx": idx,
                        "path": 1,
                        "task": task,
                        "route": f"From {page_id} to {target}",
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
                    route_text = f"Instruction: from page_0 to {target}."
                    task = self._task_label(f"From page_0 to {target}")
                    source = f"sub{subtree_idx}_edge"

                    sample_1 = {
                        "idx": idx,
                        "path": 1,
                        "task": task,
                        "route": f"From page_0 to {target}",
                        "messages": [
                            {"role": "user", "content": self._format_user_content(route_text, "Null")},
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
                        "route": f"From page_0 to {target}",
                        "messages": [
                            {"role": "user", "content": self._format_user_content(route_text, f"step1: click {action} icon on page_0")},
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

    def generate_gt_spine_edge_samples(self):
        """Generate one-step edge samples across the full GT-spine-plus-branches graph."""
        samples = []
        idx = 0
        transition_count = 0

        for page_id in self.get_sorted_page_ids():
            for target, action, bbox in self.get_public_transitions(page_id):
                route_text = f"Instruction: from {page_id} to {target} within the current app flow."
                task = self._task_label(f"From {page_id} to {target}")

                samples.append({
                    "idx": idx,
                    "path": 1,
                    "task": task,
                    "route": f"From {page_id} to {target}",
                    "messages": [
                        {"role": "user", "content": self._format_user_content(route_text, "Null")},
                        {"role": "assistant", "content": self.format_click_action(action, page_id, bbox)},
                    ],
                    "images": [os.path.join(self.pages_dir, self.pages[page_id]["image"])],
                    "bbox_norm": self.bbox_to_normalized(bbox),
                    "source": "gt_spine_edge",
                })
                idx += 1

                samples.append({
                    "idx": idx,
                    "path": 1,
                    "task": task,
                    "route": f"From {page_id} to {target}",
                    "messages": [
                        {"role": "user", "content": self._format_user_content(route_text, f"step1: click {action} icon on {page_id}")},
                        {"role": "assistant", "content": self.format_complete_action()},
                    ],
                    "images": [os.path.join(self.pages_dir, self.pages[target]["image"])],
                    "bbox_norm": [0, 0, 0, 0],
                    "source": "gt_spine_edge",
                })
                idx += 1
                transition_count += 1

        print(f"    GT spine edges: {transition_count} transitions x 2 = {len(samples)} samples")
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

        if not all_icons:
            return samples
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

        if not all_icons:
            return samples
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

    if generator.is_gt_spine_topology():
        print("Path samples from GT spine + branches...")
        path_samples = generator.generate_gt_spine_path_samples()
        print(f"  Total path samples: {len(path_samples)}")

        print("Edge samples from GT spine + branches...")
        edge_samples = generator.generate_gt_spine_edge_samples()
        print(f"  Total edge samples: {len(edge_samples)}")
    else:
        print("Path samples from subtrees 0-1...")
        path_samples = generator.generate_path_samples(subtrees=[0, 1])
        print(f"  Total path samples: {len(path_samples)}")

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
