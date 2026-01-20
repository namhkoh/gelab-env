"""
Dataset Generation Script for GE-Lab

Generates SFT, ST-RL, and MT-RL training datasets from the generated UI environment.
Uses the ui_structure.json to build navigation graphs and create training samples.

Aligned with paper settings:
- Tree depth: 7 (configurable via nodes_per_level)
- Path lengths: 1-7 steps
- Source naming: subX (multi-step), subX_edge (single-step)
- Includes complete actions and edge cases
"""

import json
import os
import random
import shutil
from collections import deque, defaultdict
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import argparse


class DatasetGenerator:
    """Generates training datasets from UI structure."""
    
    # Canvas dimensions from the environment
    CANVAS_WIDTH = 1179
    CANVAS_HEIGHT = 2556
    # Normalized coordinate system (used in Action strings)
    NORM_SIZE = 1000
    
    def __init__(self, ui_structure_path: str, pages_dir: str, output_dir: str):
        """
        Initialize the dataset generator.
        
        Args:
            ui_structure_path: Path to ui_structure.json
            pages_dir: Path to the pages/ directory containing page images
            output_dir: Directory to output datasets (e.g., datas/)
        """
        self.ui_structure_path = ui_structure_path
        self.pages_dir = pages_dir
        self.output_dir = output_dir
        
        # Load UI structure
        with open(ui_structure_path, 'r') as f:
            data = json.load(f)
        
        self.pages = data['pages']
        self.metadata = data.get('metadata', {})
        
        # Build navigation graph (excluding system actions: back, home)
        self.graph = self._build_graph()
        
        # Build reverse graph for finding paths
        self.reverse_graph = self._build_reverse_graph()
        
        # Precompute all shortest paths
        self.shortest_paths = self._compute_all_shortest_paths()
        
        # Group paths by length for balanced sampling
        self.paths_by_length = self._group_paths_by_length()
        
        print(f"Loaded {len(self.pages)} pages")
        print(f"Graph has {len(self.graph)} nodes")
        print(f"Path distribution: {[(k, len(v)) for k, v in sorted(self.paths_by_length.items())]}")
    
    def _build_graph(self) -> Dict[str, List[Tuple[str, str, List[int]]]]:
        """
        Build adjacency list from page transitions.
        Excludes system actions (back, home) for forward navigation paths.
        Returns: {page_id: [(target_page, action_name, icon_bbox), ...]}
        """
        graph = {}
        for page_id, page_data in self.pages.items():
            graph[page_id] = []
            for transition in page_data.get('transitions', []):
                target = transition['target_page']
                action = transition['action']
                # Exclude system actions for path computation
                if action in ['back', 'home']:
                    continue
                bbox = transition.get('icon_bbox', [0, 0, 0, 0])
                graph[page_id].append((target, action, bbox))
        return graph
    
    def _build_reverse_graph(self) -> Dict[str, List[str]]:
        """Build reverse graph to find parent pages."""
        reverse = defaultdict(list)
        for page_id, transitions in self.graph.items():
            for target, _, _ in transitions:
                reverse[target].append(page_id)
        return dict(reverse)
    
    def _compute_all_shortest_paths(self) -> Dict[Tuple[str, str], List[Tuple[str, str, List[int]]]]:
        """
        Compute shortest paths between all pairs of pages using BFS.
        Returns: {(start, end): [(page_id, action, bbox), ...]}
        """
        paths = {}
        page_ids = list(self.pages.keys())
        
        for start in page_ids:
            # BFS from start
            visited = {start: None}  # page -> (prev_page, action, bbox)
            queue = deque([start])
            
            while queue:
                current = queue.popleft()
                for target, action, bbox in self.graph.get(current, []):
                    if target not in visited:
                        visited[target] = (current, action, bbox)
                        queue.append(target)
            
            # Reconstruct paths to all reachable pages
            for end in page_ids:
                if end == start:
                    paths[(start, end)] = []
                elif end in visited:
                    path = []
                    current = end
                    while visited[current] is not None:
                        prev, action, bbox = visited[current]
                        path.append((prev, action, bbox))
                        current = prev
                    path.reverse()
                    paths[(start, end)] = path
        
        return paths
    
    def _group_paths_by_length(self) -> Dict[int, List[Tuple[str, str, List]]]:
        """Group all paths by their length for balanced sampling."""
        grouped = defaultdict(list)
        for (start, end), path in self.shortest_paths.items():
            if len(path) > 0:
                grouped[len(path)].append((start, end, path))
        return dict(grouped)
    
    def _bbox_to_normalized(self, bbox: List[int]) -> List[int]:
        """
        Convert pixel bbox [x1, y1, x2, y2] to normalized 0-1000 coordinates.
        This is used for bbox_norm in training data.
        """
        if not bbox or len(bbox) != 4:
            return [0, 0, 0, 0]
        
        x1, y1, x2, y2 = bbox
        return [
            int(x1 * self.NORM_SIZE / self.CANVAS_WIDTH),
            int(y1 * self.NORM_SIZE / self.CANVAS_HEIGHT),
            int(x2 * self.NORM_SIZE / self.CANVAS_WIDTH),
            int(y2 * self.NORM_SIZE / self.CANVAS_HEIGHT)
        ]
    
    def _bbox_to_center_normalized(self, bbox: List[int]) -> Tuple[int, int]:
        """
        Convert bbox [x1, y1, x2, y2] to normalized center coordinates.
        Normalized to 1000x1000 coordinate system as per paper.
        """
        if not bbox or len(bbox) != 4:
            return (500, 500)  # Default center
        
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # Normalize to 1000x1000
        norm_x = int(center_x * self.NORM_SIZE / self.CANVAS_WIDTH)
        norm_y = int(center_y * self.NORM_SIZE / self.CANVAS_HEIGHT)
        
        return (norm_x, norm_y)
    
    def _format_action(self, action_name: str, page_id: str, bbox: List[int]) -> str:
        """Format the action string with coordinates (paper format)."""
        x, y = self._bbox_to_center_normalized(bbox)
        return f"Explain:click {action_name} icon on {page_id}.\tAction: click(start_box='<|box_start|>({x},{y})<|box_end|>')"
    
    def _format_complete_action(self) -> str:
        """Format the completion action (paper format)."""
        return "Explain: this is target page.\tAction: complete"
    
    def _get_source_label(self, depth: int, is_edge: bool = False) -> str:
        """
        Generate source label following paper convention.
        - subX for multi-step navigation
        - subX_edge for single-step transitions (edge cases)
        """
        if is_edge:
            return f"sub{depth}_edge"
        return f"sub{depth}"
    
    def generate_sft_dataset(
        self,
        num_samples: int = 5000,
        min_path_length: int = 1,
        max_path_length: int = 7,
        include_complete_action: bool = True,
        balance_path_lengths: bool = True
    ) -> List[Dict]:
        """
        Generate SFT dataset with multi-turn conversation format.
        
        Following paper methodology:
        - Each sample traces a navigation path with history tracking
        - Includes complete actions when reaching target
        - Balanced distribution across path lengths
        """
        samples = []
        idx = 0
        
        # Calculate samples per path length for balance
        available_lengths = [l for l in range(min_path_length, max_path_length + 1) 
                           if l in self.paths_by_length]
        
        if not available_lengths:
            print(f"Warning: No valid paths found with length {min_path_length}-{max_path_length}")
            return samples
        
        if balance_path_lengths:
            # Distribute samples evenly across path lengths
            samples_per_length = num_samples // len(available_lengths)
            remainder = num_samples % len(available_lengths)
        else:
            # Proportional to available paths
            total_paths = sum(len(self.paths_by_length[l]) for l in available_lengths)
            samples_per_length = None
        
        for length_idx, path_length in enumerate(available_lengths):
            available_paths = self.paths_by_length[path_length]
            
            if balance_path_lengths:
                n_samples = samples_per_length + (1 if length_idx < remainder else 0)
            else:
                n_samples = int(num_samples * len(available_paths) / total_paths)
            
            # Sample paths for this length
            sampled_paths = random.choices(available_paths, k=n_samples)
            
            for start, end, path in sampled_paths:
                task = f"From {start} to {end}"
                history_steps = []
                
                # Generate sample for each step in the path
                for step_idx, (page_id, action, bbox) in enumerate(path):
                    depth = self.pages[page_id].get('depth', 0)
                    
                    # Build history string
                    if step_idx == 0:
                        history = "Null"
                        is_edge = True  # First step has no history
                    else:
                        history_parts = []
                        for i, (h_page, h_action, _) in enumerate(history_steps):
                            history_parts.append(f"step{i+1}: click {h_action} icon on {h_page}.")
                        history = " ".join(history_parts)
                        is_edge = False
                    
                    # User content
                    user_content = f"<image>Instruction: from {start} to {end}. History: {history}"
                    
                    # Assistant response
                    assistant_content = self._format_action(action, page_id, bbox)
                    
                    sample = {
                        "idx": idx,
                        "path": path_length,
                        "task": task,
                        "messages": [
                            {"role": "user", "content": user_content},
                            {"role": "assistant", "content": assistant_content}
                        ],
                        "images": [f"datas/images/{page_id}.png"],
                        "bbox_norm": self._bbox_to_normalized(bbox),
                        "source": self._get_source_label(depth, is_edge=(step_idx == 0 and path_length == 1))
                    }
                    samples.append(sample)
                    idx += 1
                    
                    # Update history
                    history_steps.append((page_id, action, bbox))
                
                # Add completion step (arriving at target)
                if include_complete_action and path:
                    # Find the target page from the last transition
                    last_page, last_action, _ = path[-1]
                    for target, action, _ in self.graph.get(last_page, []):
                        if action == last_action:
                            target_page = target
                            break
                    else:
                        target_page = end
                    
                    history_parts = []
                    for i, (h_page, h_action, _) in enumerate(history_steps):
                        history_parts.append(f"step{i+1}: click {h_action} icon on {h_page}.")
                    history = " ".join(history_parts)
                    
                    user_content = f"<image>Instruction: from {start} to {end}. History: {history}"
                    assistant_content = self._format_complete_action()
                    
                    target_depth = self.pages.get(target_page, {}).get('depth', 0)
                    
                    sample = {
                        "idx": idx,
                        "path": path_length,
                        "task": task,
                        "messages": [
                            {"role": "user", "content": user_content},
                            {"role": "assistant", "content": assistant_content}
                        ],
                        "images": [f"datas/images/{target_page}.png"],
                        "bbox_norm": [0, 0, 0, 0],  # No bbox for complete action
                        "source": self._get_source_label(target_depth)
                    }
                    samples.append(sample)
                    idx += 1
        
        print(f"Generated {len(samples)} SFT samples")
        return samples
    
    def generate_st_rl_dataset(
        self,
        num_samples: int = 3000,
        min_path_length: int = 1,
        max_path_length: int = 7,
        coordinate_noise: int = 15,
        balance_path_lengths: bool = True
    ) -> List[Dict]:
        """
        Generate ST-RL dataset with single-turn format.
        
        Following paper methodology:
        - Single-step transitions (edge cases) with History: Null
        - Uses bbox_norm for reward calculation
        - Source labeled as subX_edge
        """
        samples = []
        idx = 0
        
        available_lengths = [l for l in range(min_path_length, max_path_length + 1) 
                           if l in self.paths_by_length]
        
        if not available_lengths:
            print("Warning: No valid paths found for ST-RL")
            return samples
        
        if balance_path_lengths:
            samples_per_length = num_samples // len(available_lengths)
            remainder = num_samples % len(available_lengths)
        
        for length_idx, path_length in enumerate(available_lengths):
            available_paths = self.paths_by_length[path_length]
            
            if balance_path_lengths:
                n_samples = samples_per_length + (1 if length_idx < remainder else 0)
            else:
                n_samples = num_samples // len(available_lengths)
            
            sampled_paths = random.choices(available_paths, k=n_samples)
            
            for start, end, path in sampled_paths:
                # Take first step of the path (single-turn, edge case)
                page_id, action, bbox = path[0]
                depth = self.pages[page_id].get('depth', 0)
                
                # Add small noise to coordinates for diversity (paper methodology)
                x, y = self._bbox_to_center_normalized(bbox)
                if coordinate_noise > 0:
                    x += random.randint(-coordinate_noise, coordinate_noise)
                    y += random.randint(-coordinate_noise, coordinate_noise)
                    x = max(0, min(self.NORM_SIZE, x))
                    y = max(0, min(self.NORM_SIZE, y))
                
                problem = f"<image>Instruction: from {start} to {end}. History: Null"
                solution = f"explain:click {action} icon on {page_id}.\tAction: click(start_box='<|box_start|>({x},{y})<|box_end|>')"
                
                sample = {
                    "idx": idx,
                    "bbox_norm": self._bbox_to_normalized(bbox),
                    "image": f"datas/images/{page_id}.png",
                    "problem": problem,
                    "solution": solution,
                    "source": self._get_source_label(depth, is_edge=True)
                }
                samples.append(sample)
                idx += 1
        
        print(f"Generated {len(samples)} ST-RL samples")
        return samples
    
    def generate_mt_rl_dataset(
        self,
        num_samples: int = 3000,
        min_path_length: int = 1,
        max_path_length: int = 7,
        coordinate_noise: int = 15,
        balance_path_lengths: bool = True
    ) -> List[Dict]:
        """
        Generate MT-RL dataset with multi-turn exploration format.
        
        Following paper methodology:
        - Includes path length and task info for multi-turn exploration
        - Used with a2b reward function
        """
        samples = []
        idx = 0
        
        available_lengths = [l for l in range(min_path_length, max_path_length + 1) 
                           if l in self.paths_by_length]
        
        if not available_lengths:
            print("Warning: No valid paths found for MT-RL")
            return samples
        
        if balance_path_lengths:
            samples_per_length = num_samples // len(available_lengths)
            remainder = num_samples % len(available_lengths)
        
        for length_idx, path_length in enumerate(available_lengths):
            available_paths = self.paths_by_length[path_length]
            
            if balance_path_lengths:
                n_samples = samples_per_length + (1 if length_idx < remainder else 0)
            else:
                n_samples = num_samples // len(available_lengths)
            
            sampled_paths = random.choices(available_paths, k=n_samples)
            
            for start, end, path in sampled_paths:
                task = f"From {start} to {end}"
                
                # Take first step
                page_id, action, bbox = path[0]
                depth = self.pages[page_id].get('depth', 0)
                
                # Add noise
                x, y = self._bbox_to_center_normalized(bbox)
                if coordinate_noise > 0:
                    x += random.randint(-coordinate_noise, coordinate_noise)
                    y += random.randint(-coordinate_noise, coordinate_noise)
                    x = max(0, min(self.NORM_SIZE, x))
                    y = max(0, min(self.NORM_SIZE, y))
                
                problem = f"<image>Instruction: from {start} to {end}. History: Null"
                solution = f"Explain:click {action} icon on {page_id}.\tAction: click(start_box='<|box_start|>({x},{y})<|box_end|>')"
                
                sample = {
                    "idx": idx,
                    "path": path_length,
                    "task": task,
                    "bbox_norm": self._bbox_to_normalized(bbox),
                    "image": f"datas/images/{page_id}.png",
                    "problem": problem,
                    "solution": solution,
                    "source": self._get_source_label(depth)
                }
                samples.append(sample)
                idx += 1
        
        print(f"Generated {len(samples)} MT-RL samples")
        return samples
    
    def generate_test_dataset(
        self,
        num_samples: int = 500,
        min_path_length: int = 1,
        max_path_length: int = 7,
        balance_path_lengths: bool = True
    ) -> List[Dict]:
        """
        Generate test dataset for evaluation.
        
        Follows same format as SFT for compatibility with evaluation scripts.
        Uses deterministic sampling (seed=42) for reproducibility.
        """
        samples = []
        idx = 0
        
        available_lengths = [l for l in range(min_path_length, max_path_length + 1) 
                           if l in self.paths_by_length]
        
        if not available_lengths:
            return samples
        
        # Use fixed seed for reproducible test set
        rng = random.Random(42)
        
        if balance_path_lengths:
            samples_per_length = num_samples // len(available_lengths)
            remainder = num_samples % len(available_lengths)
        
        for length_idx, path_length in enumerate(available_lengths):
            available_paths = self.paths_by_length[path_length]
            
            if balance_path_lengths:
                n_samples = samples_per_length + (1 if length_idx < remainder else 0)
            else:
                n_samples = num_samples // len(available_lengths)
            
            # Sample without replacement for test set
            n_samples = min(n_samples, len(available_paths))
            sampled_paths = rng.sample(available_paths, n_samples)
            
            for start, end, path in sampled_paths:
                page_id, action, bbox = path[0]
                depth = self.pages[page_id].get('depth', 0)
                
                # Test set uses exact coordinates (no noise)
                x, y = self._bbox_to_center_normalized(bbox)
                
                user_content = f"<image>Instruction: from {start} to {end}. History: Null"
                assistant_content = f"Explain:click {action} icon on {page_id}.\tAction: click(start_box='<|box_start|>({x},{y})<|box_end|>')"
                
                sample = {
                    "idx": idx,
                    "path": path_length,
                    "task": f"From {start} to {end}",
                    "messages": [
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": assistant_content}
                    ],
                    "images": [f"datas/images/{page_id}.png"],
                    "bbox_norm": self._bbox_to_normalized(bbox),
                    "source": self._get_source_label(depth, is_edge=True)
                }
                samples.append(sample)
                idx += 1
        
        print(f"Generated {len(samples)} test samples")
        return samples
    
    def copy_images(self):
        """Copy page images to the output images directory."""
        images_dir = os.path.join(self.output_dir, "images")
        os.makedirs(images_dir, exist_ok=True)
        
        copied = 0
        for page_id in self.pages.keys():
            src = os.path.join(self.pages_dir, f"{page_id}.png")
            dst = os.path.join(images_dir, f"{page_id}.png")
            if os.path.exists(src):
                shutil.copy2(src, dst)
                copied += 1
        
        print(f"Copied {copied} images to {images_dir}")
    
    def setup_environment_for_training(self, project_root: str = None):
        """
        Setup environment files for reward functions and multi-turn training.
        
        The reward functions (A2B, SoftA2B) and multi-turn functions expect:
        - environment/demo/ui_structure.json
        - environment/demo/ui_structure_layer_fixed.json
        
        This method copies the generated UI structure to these locations.
        """
        if project_root is None:
            project_root = os.path.dirname(os.path.dirname(self.ui_structure_path))
            # Go up to project root (assuming data_engine/ui_environment/timestamp/)
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(self.ui_structure_path)))
        
        env_dir = os.path.join(project_root, "environment", "demo")
        os.makedirs(env_dir, exist_ok=True)
        
        # Copy ui_structure.json
        src_structure = self.ui_structure_path
        dst_structure = os.path.join(env_dir, "ui_structure.json")
        shutil.copy2(src_structure, dst_structure)
        print(f"Copied ui_structure.json to {dst_structure}")
        
        # Copy ui_structure_layer.json as ui_structure_layer_fixed.json
        src_layer = self.ui_structure_path.replace("ui_structure.json", "ui_structure_layer.json")
        dst_layer = os.path.join(env_dir, "ui_structure_layer_fixed.json")
        if os.path.exists(src_layer):
            shutil.copy2(src_layer, dst_layer)
            print(f"Copied ui_structure_layer.json to {dst_layer}")
        else:
            print(f"Warning: {src_layer} not found, skipping")
        
        # Also copy pages directory for multi-turn training
        pages_src = self.pages_dir
        pages_dst = os.path.join(env_dir, "pages")
        if os.path.exists(pages_dst):
            shutil.rmtree(pages_dst)
        shutil.copytree(pages_src, pages_dst)
        print(f"Copied pages to {pages_dst}")
    
    def save_dataset(self, data: List[Dict], filename: str):
        """Save dataset to JSON file."""
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(data)} samples to {filepath}")
    
    def print_statistics(self):
        """Print dataset statistics for verification."""
        print("\n=== Dataset Statistics ===")
        print(f"Total pages: {len(self.pages)}")
        print(f"Tree depth: {self.metadata.get('tree_depth', 'unknown')}")
        print(f"Nodes per level: {self.metadata.get('nodes_per_level', 'unknown')}")
        
        # Path length distribution
        print("\nPath length distribution:")
        for length in sorted(self.paths_by_length.keys()):
            count = len(self.paths_by_length[length])
            print(f"  Path@{length}: {count} paths")
        
        # Depth distribution
        depth_counts = defaultdict(int)
        for page_id, page_data in self.pages.items():
            depth_counts[page_data.get('depth', 0)] += 1
        
        print("\nPage depth distribution:")
        for depth in sorted(depth_counts.keys()):
            print(f"  Depth {depth}: {depth_counts[depth]} pages")
    
    def generate_all(
        self,
        sft_samples: int = 5000,
        st_rl_samples: int = 3000,
        mt_rl_samples: int = 3000,
        test_samples: int = 500,
        copy_images: bool = True,
        balance_path_lengths: bool = True,
        setup_env: bool = True,
        project_root: str = None
    ):
        """Generate all datasets following paper methodology."""
        print("\n=== Generating Datasets (Paper-aligned) ===\n")
        
        # Print statistics first
        self.print_statistics()
        
        # Copy images first
        if copy_images:
            print("\nCopying page images...")
            self.copy_images()
        
        # Setup environment for reward functions
        if setup_env:
            print("\nSetting up environment for training...")
            self.setup_environment_for_training(project_root)
        
        # Generate SFT dataset
        print("\nGenerating SFT dataset...")
        sft_data = self.generate_sft_dataset(
            num_samples=sft_samples,
            balance_path_lengths=balance_path_lengths
        )
        self.save_dataset(sft_data, "sft.json")
        
        # Generate ST-RL dataset
        print("\nGenerating ST-RL dataset...")
        st_rl_data = self.generate_st_rl_dataset(
            num_samples=st_rl_samples,
            balance_path_lengths=balance_path_lengths
        )
        self.save_dataset(st_rl_data, "st_rl.json")
        
        # Generate MT-RL dataset
        print("\nGenerating MT-RL dataset...")
        mt_rl_data = self.generate_mt_rl_dataset(
            num_samples=mt_rl_samples,
            balance_path_lengths=balance_path_lengths
        )
        self.save_dataset(mt_rl_data, "mt_rl.json")
        
        # Generate test dataset
        print("\nGenerating test dataset...")
        test_data = self.generate_test_dataset(
            num_samples=test_samples,
            balance_path_lengths=balance_path_lengths
        )
        self.save_dataset(test_data, "test.json")
        
        print("\n=== Dataset Generation Complete ===")
        print(f"SFT: {len(sft_data)} samples")
        print(f"ST-RL: {len(st_rl_data)} samples")
        print(f"MT-RL: {len(mt_rl_data)} samples")
        print(f"Test: {len(test_data)} samples")
        print(f"\nOutput directory: {self.output_dir}")
        if setup_env:
            print(f"Environment setup at: environment/demo/")


def main():
    parser = argparse.ArgumentParser(
        description="Generate training datasets from UI environment (paper-aligned)"
    )
    parser.add_argument("--ui_structure", type=str, required=True,
                        help="Path to ui_structure.json")
    parser.add_argument("--pages_dir", type=str, required=True,
                        help="Path to pages/ directory")
    parser.add_argument("--output_dir", type=str, default="datas",
                        help="Output directory for datasets")
    parser.add_argument("--project_root", type=str, default=None,
                        help="Project root for setting up environment/demo/ (auto-detected if not set)")
    parser.add_argument("--sft_samples", type=int, default=5000,
                        help="Number of SFT navigation tasks (generates more samples due to multi-step)")
    parser.add_argument("--st_rl_samples", type=int, default=3000,
                        help="Number of ST-RL samples (single-step edge cases)")
    parser.add_argument("--mt_rl_samples", type=int, default=3000,
                        help="Number of MT-RL samples (multi-turn exploration)")
    parser.add_argument("--test_samples", type=int, default=500,
                        help="Number of test samples")
    parser.add_argument("--no_copy_images", action="store_true",
                        help="Skip copying images to output directory")
    parser.add_argument("--no_balance", action="store_true",
                        help="Disable balanced sampling across path lengths")
    parser.add_argument("--no_setup_env", action="store_true",
                        help="Skip setting up environment/demo/ for reward functions")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    if args.seed is not None:
        random.seed(args.seed)
    
    generator = DatasetGenerator(
        ui_structure_path=args.ui_structure,
        pages_dir=args.pages_dir,
        output_dir=args.output_dir
    )
    
    generator.generate_all(
        sft_samples=args.sft_samples,
        st_rl_samples=args.st_rl_samples,
        mt_rl_samples=args.mt_rl_samples,
        test_samples=args.test_samples,
        copy_images=not args.no_copy_images,
        balance_path_lengths=not args.no_balance,
        setup_env=not args.no_setup_env,
        project_root=args.project_root
    )


if __name__ == "__main__":
    main()
