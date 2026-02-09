"""
Generate balanced ST-RL training data for subtrees 2-3.

Paper specification (Table 9):
- ST-RL uses 12,439 samples total
- Each subtree has ~12,439 path instances

Key insight from analysis:
- Cross-subtree paths (sub2 -> sub3) heavily bias toward "home" action (97.7%)
- This causes the model to always predict "home"

Solution:
- Use only INTRA-SUBTREE paths (within sub2 OR within sub3)
- This gives ~11,580 samples with balanced distribution:
  - home: 14.6%
  - back: 9.2%
  - other (specific icons): 76.2%

Format: problem/solution for GRPO training
"""

import os
import sys
import json
import random
import argparse
import networkx as nx
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Set
import re

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class BalancedSTRLDataGenerator:
    """Generate balanced ST-RL training data using intra-subtree paths only."""
    
    def __init__(self, env_dir: str):
        self.env_dir = env_dir
        self.pages_dir = os.path.join(env_dir, "pages")
        
        # Load ui_structure.json
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
        
        # Canvas dimensions
        self.canvas_width = 448
        self.canvas_height = 448
    
    def _build_subtree_mapping(self):
        """Map each page to its subtree index."""
        root_children = list(self.nx_graph.successors("page_0"))
        root_children = [c for c in root_children if c != "page_0"]
        root_children.sort(key=lambda x: int(x.split('_')[1]))
        
        for subtree_idx, subtree_root in enumerate(root_children):
            self.page_to_subtree[subtree_root] = subtree_idx
            queue = [subtree_root]
            while queue:
                current = queue.pop(0)
                for neighbor in self.nx_graph.successors(current):
                    if neighbor not in self.page_to_subtree and neighbor != "page_0":
                        self.page_to_subtree[neighbor] = subtree_idx
                        queue.append(neighbor)
    
    def _get_pages_in_subtree(self, subtree_idx: int) -> Set[str]:
        """Get all page IDs belonging to a specific subtree."""
        return {page_id for page_id, idx in self.page_to_subtree.items() if idx == subtree_idx}
    
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
            
            for target, action, bbox in self.graph.get(current, []):
                if target == next_node:
                    result.append((current, action, bbox))
                    break
        
        return result
    
    def generate_intra_subtree_data(self, subtrees: List[int], target_samples: int = 12439) -> List[Dict]:
        """
        Generate ST-RL data using only INTRA-SUBTREE paths.
        
        This avoids the cross-subtree paths that heavily bias toward "home" action.
        """
        samples = []
        idx = 0
        
        for subtree_idx in subtrees:
            subtree_pages = list(self._get_pages_in_subtree(subtree_idx))
            print(f"Subtree {subtree_idx}: {len(subtree_pages)} pages")
            
            # Generate all intra-subtree paths
            subtree_samples = []
            
            for start in subtree_pages:
                for end in subtree_pages:
                    if start == end:
                        continue
                    
                    path = self._get_path_with_actions(start, end)
                    if not path:
                        continue
                    
                    path_len = len(path)
                    task = f"From {start} to {end}"
                    
                    # Generate sample for each step in the path
                    for step_idx, (from_page, action, bbox) in enumerate(path):
                        # Build history
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
                        
                        sample = {
                            "idx": idx,
                            "path_length": path_len,
                            "step": step_idx + 1,
                            "bbox_norm": norm_bbox,
                            "image": os.path.join(self.pages_dir, f"{from_page}.png"),
                            "problem": f"<image>Instruction: from {start} to {end}. History: {history}",
                            "solution": f"Explain: click {action} icon on {from_page}.\tAction: click(start_box='<|box_start|>({cx},{cy})<|box_end|>')",
                            "source": f"sub{subtree_idx}_intra_path",
                            "first_action": action if step_idx == 0 else None
                        }
                        subtree_samples.append(sample)
                        idx += 1
            
            print(f"  Generated {len(subtree_samples)} samples from subtree {subtree_idx}")
            samples.extend(subtree_samples)
        
        # Analyze action distribution
        self._print_action_distribution(samples)
        
        # If we have more than target, sample down
        if len(samples) > target_samples:
            print(f"\nSampling down from {len(samples)} to {target_samples} samples")
            samples = self._balanced_sample(samples, target_samples)
        
        # Reindex and shuffle
        random.shuffle(samples)
        for i, sample in enumerate(samples):
            sample["idx"] = i
            if "first_action" in sample:
                del sample["first_action"]
        
        return samples
    
    def _balanced_sample(self, samples: List[Dict], target: int) -> List[Dict]:
        """
        Sample while maintaining diversity across:
        - Path lengths
        - Subtrees
        - First-step actions (for step=1 samples)
        """
        # Group by path (defined by source->target in problem)
        path_groups = defaultdict(list)
        for sample in samples:
            prob = sample["problem"]
            match = re.search(r'from (\S+) to (\S+)', prob, re.IGNORECASE)
            if match:
                path_key = f"{match.group(1)}->{match.group(2)}"
                path_groups[path_key].append(sample)
        
        # Calculate how many paths to include
        total_paths = len(path_groups)
        
        # Sample paths to get roughly target samples
        # Avg samples per path = target / num_paths_to_keep
        # We want all steps of selected paths, so sample paths not individual samples
        
        # First, stratify by path length
        paths_by_length = defaultdict(list)
        for path_key, path_samples in path_groups.items():
            path_len = path_samples[0]["path_length"]
            paths_by_length[path_len].append(path_key)
        
        # Sample paths proportionally by length
        selected_paths = []
        total_samples_in_paths = 0
        
        for path_len in sorted(paths_by_length.keys()):
            paths = paths_by_length[path_len]
            # Estimate samples from this length: path_len * num_paths
            # We want proportional representation
            
            # Sample all if few
            if total_samples_in_paths < target:
                for path_key in paths:
                    selected_paths.append(path_key)
                    total_samples_in_paths += path_len
                    if total_samples_in_paths >= target:
                        break
        
        # Collect samples from selected paths
        result = []
        for path_key in selected_paths:
            result.extend(path_groups[path_key])
        
        # If still over target, randomly subsample
        if len(result) > target:
            result = random.sample(result, target)
        
        return result
    
    def _print_action_distribution(self, samples: List[Dict]):
        """Print action distribution for analysis."""
        first_step_actions = Counter()
        all_actions = Counter()
        
        for sample in samples:
            sol = sample["solution"]
            match = re.search(r'click (\S+) icon', sol, re.IGNORECASE)
            if match:
                action = match.group(1).lower()
                all_actions[action] += 1
                if sample.get("step", 1) == 1:
                    first_step_actions[action] += 1
        
        print("\n=== ACTION DISTRIBUTION ===")
        print("First-step actions:")
        total_first = sum(first_step_actions.values())
        home = first_step_actions.get("home", 0)
        back = first_step_actions.get("back", 0)
        other = total_first - home - back
        print(f"  home: {home} ({home/total_first*100:.1f}%)")
        print(f"  back: {back} ({back/total_first*100:.1f}%)")
        print(f"  other: {other} ({other/total_first*100:.1f}%)")
        
        print("\nAll actions:")
        total_all = sum(all_actions.values())
        home = all_actions.get("home", 0)
        back = all_actions.get("back", 0)
        other = total_all - home - back
        print(f"  home: {home} ({home/total_all*100:.1f}%)")
        print(f"  back: {back} ({back/total_all*100:.1f}%)")
        print(f"  other: {other} ({other/total_all*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Generate balanced ST-RL training data")
    parser.add_argument("--env_dir", default="data_engine/ui_environment_448/latest",
                       help="Path to environment directory")
    parser.add_argument("--output_dir", default="datas/448_retrain",
                       help="Output directory for datasets")
    parser.add_argument("--target_samples", type=int, default=12439,
                       help="Target total samples (paper: 12,439)")
    args = parser.parse_args()
    
    # Find environment directory
    if not os.path.isabs(args.env_dir):
        args.env_dir = os.path.join(os.getcwd(), args.env_dir)
    
    print("="*60)
    print("BALANCED ST-RL DATA GENERATION")
    print("="*60)
    print(f"Environment: {args.env_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Target samples: {args.target_samples}")
    print()
    print("Strategy: Use only INTRA-SUBTREE paths to avoid home-action bias")
    print()
    
    generator = BalancedSTRLDataGenerator(args.env_dir)
    
    # Generate intra-subtree path data
    print("Generating intra-subtree path data (subtrees 2-3)...")
    samples = generator.generate_intra_subtree_data(
        subtrees=[2, 3],
        target_samples=args.target_samples
    )
    
    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    
    output_path = os.path.join(args.output_dir, "st_rl_balanced.json")
    with open(output_path, 'w') as f:
        json.dump(samples, f, indent=2)
    
    print()
    print("="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total samples: {len(samples)}")
    print(f"Output: {output_path}")
    
    # Final distribution check
    from collections import Counter
    import re
    
    first_actions = []
    for s in samples:
        if s.get("step", 1) == 1:
            match = re.search(r'click (\S+) icon', s["solution"], re.IGNORECASE)
            if match:
                action = match.group(1).lower()
                first_actions.append("home" if action == "home" else "back" if action == "back" else "icon")
    
    dist = Counter(first_actions)
    total = sum(dist.values())
    print(f"\nFinal first-step action distribution:")
    for action in ["home", "back", "icon"]:
        count = dist.get(action, 0)
        print(f"  {action}: {count} ({count/total*100:.1f}%)")


if __name__ == "__main__":
    main()
