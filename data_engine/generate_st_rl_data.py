"""
Generate ST-RL training data for subtrees 2-3.

Paper specification:
- ST-RL uses Path data from RL subtrees (2-3)
- Each subtree has ~12,439 path instances
- Pre-constructed trajectories where o_{0:t} and a_{0:t-1} are given
- Model predicts a_t (next action)

Format: problem/solution for GRPO training
"""

import os
import sys
import json
import random
import argparse
import networkx as nx
from collections import defaultdict
from typing import Dict, List, Tuple, Set

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class STRLDataGenerator:
    """Generate ST-RL training data with Path data from RL subtrees."""
    
    def __init__(self, env_dir: str):
        self.env_dir = env_dir
        self.pages_dir = os.path.join(env_dir, "pages")
        
        # Load ui_structure.json (contains pages and transitions)
        ui_structure_path = os.path.join(env_dir, "ui_structure.json")
        with open(ui_structure_path) as f:
            ui_structure = json.load(f)
        
        self.pages = ui_structure.get("pages", {})
        
        # Build graph from transitions
        self.graph = {}
        for page_id, page_data in self.pages.items():
            transitions = page_data.get("transitions", [])
            self.graph[page_id] = [
                (t["target_page"], t["action"], t["icon_bbox"])
                for t in transitions
            ]
        
        # Build NetworkX graph for pathfinding
        self.nx_graph = nx.DiGraph()
        for src, edges in self.graph.items():
            for target, action, bbox in edges:
                self.nx_graph.add_edge(src, target, action=action, bbox=bbox)
        
        # Map pages to subtrees
        self.page_to_subtree = {}
        self._build_subtree_mapping()
        
        # Canvas dimensions (448x448)
        self.canvas_width = 448
        self.canvas_height = 448
    
    def _build_subtree_mapping(self):
        """Map each page to its subtree index."""
        root_children = list(self.nx_graph.successors("page_0"))
        root_children = [c for c in root_children if c != "page_0"]
        root_children.sort(key=lambda x: int(x.split('_')[1]))
        
        for subtree_idx, subtree_root in enumerate(root_children):
            self.page_to_subtree[subtree_root] = subtree_idx
            # BFS to find all descendants
            queue = [subtree_root]
            while queue:
                current = queue.pop(0)
                for neighbor in self.nx_graph.successors(current):
                    if neighbor not in self.page_to_subtree and neighbor != "page_0":
                        self.page_to_subtree[neighbor] = subtree_idx
                        queue.append(neighbor)
    
    def _get_pages_in_subtrees(self, subtree_indices: List[int]) -> Set[str]:
        """Get all page IDs belonging to specified subtrees."""
        pages = set()
        for page_id, subtree_idx in self.page_to_subtree.items():
            if subtree_idx in subtree_indices:
                pages.add(page_id)
        return pages
    
    def _normalize_coord(self, x: int, y: int) -> Tuple[int, int]:
        """Normalize coordinates to 0-1000 range."""
        norm_x = int(x / self.canvas_width * 1000)
        norm_y = int(y / self.canvas_height * 1000)
        return min(norm_x, 1000), min(norm_y, 1000)
    
    def _bbox_to_normalized(self, bbox: List[int]) -> List[int]:
        """Convert bbox to normalized coordinates."""
        x1, y1, x2, y2 = bbox
        nx1, ny1 = self._normalize_coord(x1, y1)
        nx2, ny2 = self._normalize_coord(x2, y2)
        return [nx1, ny1, nx2, ny2]
    
    def _bbox_center_normalized(self, bbox: List[int]) -> Tuple[int, int]:
        """Get normalized center of bbox."""
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        return self._normalize_coord(cx, cy)
    
    def _get_path_with_actions(self, start: str, end: str) -> List[Tuple[str, str, List[int]]]:
        """Get path from start to end with actions and bboxes."""
        try:
            path_nodes = nx.shortest_path(self.nx_graph, start, end)
        except nx.NetworkXNoPath:
            return []
        
        if len(path_nodes) < 2:
            return []
        
        result = []
        for i in range(len(path_nodes) - 1):
            current = path_nodes[i]
            next_node = path_nodes[i + 1]
            
            # Find the edge data
            for target, action, bbox in self.graph.get(current, []):
                if target == next_node:
                    result.append((current, action, bbox))
                    break
        
        return result
    
    def generate_st_rl_path_data(self, subtrees: List[int], target_samples: int = 12439) -> List[Dict]:
        """
        Generate ST-RL Path data for specified subtrees.
        
        For each path, we generate samples for each step in the trajectory.
        Each sample has the history of previous steps and asks model to predict next action.
        
        Note: The problem shows original start->end task, but current page changes as 
        history progresses. This matches paper's formulation where task instruction
        stays constant but history captures previous actions.
        """
        samples = []
        idx = 0
        
        subtree_pages = list(self._get_pages_in_subtrees(subtrees))
        print(f"Pages in subtrees {subtrees}: {len(subtree_pages)}")
        
        # Generate paths between all pairs
        all_paths = []
        for start in subtree_pages:
            for end in subtree_pages:
                if start == end:
                    continue
                path = self._get_path_with_actions(start, end)
                if path:
                    all_paths.append((start, end, path))
        
        print(f"Total unique paths found: {len(all_paths)}")
        
        # Generate samples from paths
        samples_per_subtree = {s: [] for s in subtrees}
        
        for start, end, path in all_paths:
            path_len = len(path)
            task = f"From {start} to {end}"
            subtree_idx = self.page_to_subtree.get(start, 2)
            
            # For each step in the path, create a training sample
            for step_idx, (from_page, action, bbox) in enumerate(path):
                # Build history - shows what actions were taken previously
                if step_idx == 0:
                    history = "Null"
                else:
                    history_steps = []
                    for h_idx in range(step_idx):
                        h_page, h_action, _ = path[h_idx]
                        history_steps.append(f"step{h_idx+1}: click {h_action} icon on {h_page}")
                    history = "; ".join(history_steps)
                
                # Normalize coordinates
                norm_bbox = self._bbox_to_normalized(bbox)
                cx, cy = self._bbox_center_normalized(bbox)
                
                # from_page is the CURRENT page where action should be taken
                # The instruction shows original task (start->end)
                sample = {
                    "idx": idx,
                    "path_length": path_len,
                    "step": step_idx + 1,
                    "bbox_norm": norm_bbox,
                    "image": os.path.join(self.pages_dir, f"{from_page}.png"),
                    "problem": f"<image>Instruction: from {start} to {end}. History: {history}",
                    "solution": f"Explain: click {action} icon on {from_page}.\tAction: click(start_box='<|box_start|>({cx},{cy})<|box_end|>')",
                    "source": f"sub{subtree_idx}_path"
                }
                samples_per_subtree[subtree_idx].append(sample)
                idx += 1
        
        # Balance and limit samples
        samples = []
        for subtree_idx in subtrees:
            subtree_samples = samples_per_subtree.get(subtree_idx, [])
            # Limit to target per subtree
            if len(subtree_samples) > target_samples:
                subtree_samples = random.sample(subtree_samples, target_samples)
            samples.extend(subtree_samples)
            print(f"  Subtree {subtree_idx}: {len(subtree_samples)} path samples")
        
        # Reindex
        for i, sample in enumerate(samples):
            sample["idx"] = i
        
        random.shuffle(samples)
        return samples
    
    def generate_st_rl_edge_data(self, subtrees: List[int]) -> List[Dict]:
        """
        Generate ST-RL Edge data (single-step) for specified subtrees.
        These are simpler samples with no history.
        """
        samples = []
        idx = 0
        
        for subtree_idx in subtrees:
            subtree_pages = self._get_pages_in_subtrees([subtree_idx])
            
            for page_id in subtree_pages:
                for target, action, bbox in self.graph.get(page_id, []):
                    if action in ['back', 'home']:
                        continue
                    
                    norm_bbox = self._bbox_to_normalized(bbox)
                    cx, cy = self._bbox_center_normalized(bbox)
                    
                    sample = {
                        "idx": idx,
                        "path_length": 1,
                        "step": 1,
                        "bbox_norm": norm_bbox,
                        "image": os.path.join(self.pages_dir, f"{page_id}.png"),
                        "problem": f"<image>Instruction: from {page_id} to {target}. History: Null",
                        "solution": f"Explain: click {action} icon on {page_id}.\tAction: click(start_box='<|box_start|>({cx},{cy})<|box_end|>')",
                        "source": f"sub{subtree_idx}_edge"
                    }
                    samples.append(sample)
                    idx += 1
            
            print(f"  Subtree {subtree_idx}: {idx - sum(1 for s in samples if s['source'] != f'sub{subtree_idx}_edge')} edge samples")
        
        return samples


def main():
    parser = argparse.ArgumentParser(description="Generate ST-RL training data")
    parser.add_argument("--env_dir", default="data_engine/ui_environment_448/latest",
                       help="Path to environment directory")
    parser.add_argument("--output_dir", default="datas/448_paper",
                       help="Output directory for datasets")
    parser.add_argument("--target_samples", type=int, default=12439,
                       help="Target samples per subtree (paper: 12,439)")
    args = parser.parse_args()
    
    # Find environment directory
    if not os.path.isabs(args.env_dir):
        args.env_dir = os.path.join(os.getcwd(), args.env_dir)
    
    print("="*60)
    print("ST-RL DATA GENERATION")
    print("="*60)
    print(f"Environment: {args.env_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Target samples per subtree: {args.target_samples}")
    print()
    
    generator = STRLDataGenerator(args.env_dir)
    
    # Generate Path data for ST-RL (main training data)
    print("Generating ST-RL Path data (subtrees 2-3)...")
    path_samples = generator.generate_st_rl_path_data(
        subtrees=[2, 3],
        target_samples=args.target_samples
    )
    
    # Generate Edge data for ST-RL (supplementary)
    print("\nGenerating ST-RL Edge data (subtrees 2-3)...")
    edge_samples = generator.generate_st_rl_edge_data(subtrees=[2, 3])
    
    # Combine: Path + Edge
    all_samples = path_samples + edge_samples
    
    # Reindex
    for i, sample in enumerate(all_samples):
        sample["idx"] = i
    
    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save combined ST-RL data
    output_path = os.path.join(args.output_dir, "st_rl_448.json")
    with open(output_path, 'w') as f:
        json.dump(all_samples, f, indent=2)
    
    print()
    print("="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Path samples: {len(path_samples)}")
    print(f"Edge samples: {len(edge_samples)}")
    print(f"Total ST-RL samples: {len(all_samples)}")
    print(f"Output: {output_path}")
    
    # Also save path-only version
    path_only_output = os.path.join(args.output_dir, "st_rl_path_only.json")
    with open(path_only_output, 'w') as f:
        json.dump(path_samples, f, indent=2)
    print(f"Path-only output: {path_only_output}")


if __name__ == "__main__":
    main()
