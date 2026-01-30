"""
SFT Dataset Generation - Aligned with Paper

Paper specifications (Appendix A.1):
- 5 subtrees from root, each with 46 nodes
- SFT training: Path data from subtrees 0-1, Edge data from ALL subtrees
- Per subtree: ~12,439 path instances, 274 edge instances
- Auxiliary: 2,320 Icon Grounding + 2,320 Icon Captioning

PATH DATA:
  Key insight: page_0 is included as valid endpoint
  For each path of length N, generate N+1 samples:
  - Samples 0 to N-1: Click actions with history
  - Sample N: Complete action at target

EDGE DATA:
  For each transition A→B, generates 2 samples:
  1. Click sample: At page A, action = click to reach B
  2. Complete sample: At page B, action = complete
"""

import json
import os
import random
from collections import deque, defaultdict
from typing import Dict, List, Tuple, Set
import argparse
import networkx as nx


class AlignedDatasetGenerator:
    """Generate dataset aligned with paper specifications."""
    
    # Updated to match author's 448x448 square format (max_pixels=200704 in paper)
    CANVAS_WIDTH = 448
    CANVAS_HEIGHT = 448
    NORM_SIZE = 1000
    
    def __init__(self, ui_structure_path: str, ui_layer_path: str, pages_dir: str):
        self.pages_dir = pages_dir
        
        # Load UI structure
        with open(ui_structure_path, 'r') as f:
            data = json.load(f)
        self.pages = data['pages']
        
        # Build page-to-subtree mapping
        self.page_to_subtree = self._build_subtree_mapping(ui_layer_path)
        
        # Build navigation graph (full with back/home)
        self.graph = self._build_graph()
        
        # Build NetworkX graph for path finding
        self.nx_graph = self._build_nx_graph()
        
        print(f"Loaded {len(self.pages)} pages")
        print(f"Subtree distribution: {self._count_subtrees()}")
    
    def _build_subtree_mapping(self, ui_layer_path: str) -> Dict[str, int]:
        """Build mapping from page_id to subtree index (0-4)."""
        with open(ui_layer_path, 'r') as f:
            ui = json.load(f)
        
        root = ui.get('root', {})
        page_to_subtree = {'page_0': -1}  # root is special
        
        def collect_pages(node, subtree_idx):
            img = node.get('image', '')
            page_id = img.replace('.png', '') if img else None
            if page_id:
                page_to_subtree[page_id] = subtree_idx
            for child in node.get('subnodes', []):
                collect_pages(child, subtree_idx)
        
        for i, child in enumerate(root.get('subnodes', [])):
            collect_pages(child, i)
        
        return page_to_subtree
    
    def _count_subtrees(self) -> Dict[int, int]:
        """Count pages per subtree."""
        counts = defaultdict(int)
        for page_id, subtree in self.page_to_subtree.items():
            counts[subtree] += 1
        return dict(counts)
    
    def _build_graph(self) -> Dict[str, List[Tuple[str, str, List[int]]]]:
        """Build navigation graph INCLUDING back/home for full connectivity."""
        graph = {}
        for page_id, page_data in self.pages.items():
            graph[page_id] = []
            for trans in page_data.get('transitions', []):
                action = trans['action']
                target = trans['target_page']
                bbox = trans.get('icon_bbox', [0, 0, 0, 0])
                graph[page_id].append((target, action, bbox))
        return graph
    
    def _build_nx_graph(self) -> nx.DiGraph:
        """Build NetworkX graph for efficient path finding."""
        G = nx.DiGraph()
        
        for page_id, page_data in self.pages.items():
            G.add_node(page_id, depth=page_data.get('depth', 0))
            for trans in page_data.get('transitions', []):
                action = trans['action']
                target = trans['target_page']
                bbox = trans.get('icon_bbox', [0, 0, 0, 0])
                G.add_edge(page_id, target, action=action, bbox=bbox)
        
        return G
    
    def _bbox_to_normalized(self, bbox: List[int]) -> List[int]:
        """Convert pixel bbox to normalized 0-1000 coordinates."""
        if not bbox or bbox == [0, 0, 0, 0]:
            return [0, 0, 0, 0]
        x1, y1, x2, y2 = bbox
        return [
            int(x1 * self.NORM_SIZE / self.CANVAS_WIDTH),
            int(y1 * self.NORM_SIZE / self.CANVAS_HEIGHT),
            int(x2 * self.NORM_SIZE / self.CANVAS_WIDTH),
            int(y2 * self.NORM_SIZE / self.CANVAS_HEIGHT)
        ]
    
    def _bbox_center_normalized(self, bbox: List[int]) -> Tuple[int, int]:
        """Get center of bbox in normalized coordinates."""
        norm = self._bbox_to_normalized(bbox)
        cx = (norm[0] + norm[2]) // 2
        cy = (norm[1] + norm[3]) // 2
        return cx, cy
    
    def _format_click_action(self, action_name: str, page_id: str, bbox: List[int]) -> str:
        """Format click action string."""
        cx, cy = self._bbox_center_normalized(bbox)
        return f"Explain: click {action_name} icon on {page_id}.\tAction: click(start_box='<|box_start|>({cx},{cy})<|box_end|>')"
    
    def _format_complete_action(self) -> str:
        """Format complete action string."""
        return "Explain: this is target page.\tAction: complete"
    
    def _get_pages_in_subtrees(self, subtree_indices: List[int]) -> Set[str]:
        """Get all pages belonging to specified subtrees."""
        return {p for p, s in self.page_to_subtree.items() if s in subtree_indices}
    
    def _get_path_with_actions(self, start: str, end: str) -> List[Tuple[str, str, List[int]]]:
        """Get shortest path with action and bbox info."""
        try:
            path_nodes = nx.shortest_path(self.nx_graph, start, end)
        except nx.NetworkXNoPath:
            return None
        
        if len(path_nodes) < 2:
            return None
        
        path = []
        for i in range(len(path_nodes) - 1):
            from_node = path_nodes[i]
            to_node = path_nodes[i + 1]
            edge_data = self.nx_graph.get_edge_data(from_node, to_node)
            if edge_data:
                path.append((from_node, edge_data['action'], edge_data['bbox']))
        
        return path
    
    def generate_path_samples(self, subtrees: List[int], include_page_0: bool = True) -> List[Dict]:
        """
        Generate path samples for specified subtrees.
        
        For each path of length N, generate N+1 samples:
        - Samples 0 to N-1: Click actions with history
        - Sample N: Complete action at target
        """
        samples = []
        idx = 0
        
        for subtree_idx in subtrees:
            subtree_pages = self._get_pages_in_subtrees([subtree_idx])
            
            # Valid endpoints include page_0
            valid_endpoints = subtree_pages.copy()
            if include_page_0:
                valid_endpoints.add('page_0')
            
            pair_count = 0
            sample_count = 0
            
            for start in valid_endpoints:
                for end in valid_endpoints:
                    if start == end:
                        continue
                    
                    # At least one endpoint must be in subtree
                    if start not in subtree_pages and end not in subtree_pages:
                        continue
                    
                    path = self._get_path_with_actions(start, end)
                    if not path:
                        continue
                    
                    pair_count += 1
                    path_length = len(path)
                    task = f"From {start} to {end}"
                    source = f"sub{subtree_idx}_path"
                    history_steps = []
                    
                    # Generate sample for each step (click actions)
                    for step_idx, (page_id, action, bbox) in enumerate(path):
                        if step_idx == 0:
                            history = "Null"
                        else:
                            history = "; ".join([f"step{i+1}: click {h[1]} icon on {h[0]}" 
                                               for i, h in enumerate(history_steps)])
                        
                        user_content = f"<image>Instruction: from {start} to {end}. History: {history}"
                        assistant_content = self._format_click_action(action, page_id, bbox)
                        
                        sample = {
                            "idx": idx,
                            "path": path_length,
                            "task": task,
                            "messages": [
                                {"role": "user", "content": user_content},
                                {"role": "assistant", "content": assistant_content}
                            ],
                            "images": [os.path.join(self.pages_dir, f"{page_id}.png")],
                            "bbox_norm": self._bbox_to_normalized(bbox),
                            "source": source
                        }
                        samples.append(sample)
                        idx += 1
                        sample_count += 1
                        
                        history_steps.append((page_id, action))
                    
                    # Generate complete action sample at target
                    final_history = "; ".join([f"step{i+1}: click {h[1]} icon on {h[0]}" 
                                              for i, h in enumerate(history_steps)])
                    
                    user_content = f"<image>Instruction: from {start} to {end}. History: {final_history}"
                    assistant_content = self._format_complete_action()
                    
                    sample = {
                        "idx": idx,
                        "path": path_length,
                        "task": task,
                        "messages": [
                            {"role": "user", "content": user_content},
                            {"role": "assistant", "content": assistant_content}
                        ],
                        "images": [os.path.join(self.pages_dir, f"{end}.png")],
                        "bbox_norm": [0, 0, 0, 0],
                        "source": source
                    }
                    samples.append(sample)
                    idx += 1
                    sample_count += 1
            
            print(f"    Subtree {subtree_idx}: {pair_count} pairs -> {sample_count} samples")
        
        return samples
    
    def generate_edge_samples(self, subtrees: List[int]) -> List[Dict]:
        """
        Generate edge (single-step) samples for specified subtrees.
        
        For each transition A->B, generates 2 samples:
        1. Click sample: At page A, action = click to reach B
        2. Complete sample: At page B, action = complete
        """
        samples = []
        idx = 0
        
        for subtree_idx in subtrees:
            subtree_pages = self._get_pages_in_subtrees([subtree_idx])
            transition_count = 0
            
            # Process all transitions FROM pages in this subtree
            for page_id in subtree_pages:
                for target, action, bbox in self.graph.get(page_id, []):
                    task = f"From {page_id} to {target}"
                    source = f"sub{subtree_idx}_edge"
                    
                    # Sample 1: Click action at source page
                    user_1 = f"<image>Instruction: from {page_id} to {target}. History: Null"
                    assistant_1 = self._format_click_action(action, page_id, bbox)
                    
                    sample_1 = {
                        "idx": idx,
                        "path": 1,
                        "task": task,
                        "messages": [
                            {"role": "user", "content": user_1},
                            {"role": "assistant", "content": assistant_1}
                        ],
                        "images": [os.path.join(self.pages_dir, f"{page_id}.png")],
                        "bbox_norm": self._bbox_to_normalized(bbox),
                        "source": source
                    }
                    samples.append(sample_1)
                    idx += 1
                    
                    # Sample 2: Complete action at target page
                    history = f"step1: click {action} icon on {page_id}"
                    user_2 = f"<image>Instruction: from {page_id} to {target}. History: {history}"
                    assistant_2 = self._format_complete_action()
                    
                    sample_2 = {
                        "idx": idx,
                        "path": 1,
                        "task": task,
                        "messages": [
                            {"role": "user", "content": user_2},
                            {"role": "assistant", "content": assistant_2}
                        ],
                        "images": [os.path.join(self.pages_dir, f"{target}.png")],
                        "bbox_norm": [0, 0, 0, 0],
                        "source": source
                    }
                    samples.append(sample_2)
                    idx += 1
                    
                    transition_count += 1
            
            # Add entry transitions from page_0 to subtree
            for target, action, bbox in self.graph.get('page_0', []):
                if self.page_to_subtree.get(target) == subtree_idx:
                    task = f"From page_0 to {target}"
                    source = f"sub{subtree_idx}_edge"
                    
                    # Click at page_0
                    sample_1 = {
                        "idx": idx,
                        "path": 1,
                        "task": task,
                        "messages": [
                            {"role": "user", "content": f"<image>Instruction: from page_0 to {target}. History: Null"},
                            {"role": "assistant", "content": self._format_click_action(action, "page_0", bbox)}
                        ],
                        "images": [os.path.join(self.pages_dir, "page_0.png")],
                        "bbox_norm": self._bbox_to_normalized(bbox),
                        "source": source
                    }
                    samples.append(sample_1)
                    idx += 1
                    
                    # Complete at target
                    sample_2 = {
                        "idx": idx,
                        "path": 1,
                        "task": task,
                        "messages": [
                            {"role": "user", "content": f"<image>Instruction: from page_0 to {target}. History: step1: click {action} icon on page_0"},
                            {"role": "assistant", "content": self._format_complete_action()}
                        ],
                        "images": [os.path.join(self.pages_dir, f"{target}.png")],
                        "bbox_norm": [0, 0, 0, 0],
                        "source": source
                    }
                    samples.append(sample_2)
                    idx += 1
                    
                    transition_count += 1
            
            print(f"    Subtree {subtree_idx}: {transition_count} transitions x 2 = {transition_count * 2} samples")
        
        return samples
    
    def generate_grounding_samples(self, num_samples: int = 2320) -> List[Dict]:
        """Generate Icon Grounding samples (text -> coordinates)."""
        samples = []
        
        all_icons = []
        for page_id, page_data in self.pages.items():
            for trans in page_data.get('transitions', []):
                action = trans['action']
                if action in ['back', 'home']:
                    continue
                bbox = trans.get('icon_bbox', [0, 0, 0, 0])
                all_icons.append((page_id, action, bbox))
        
        sampled = random.choices(all_icons, k=num_samples)
        
        for idx, (page_id, action, bbox) in enumerate(sampled):
            cx, cy = self._bbox_center_normalized(bbox)
            
            sample = {
                "idx": idx,
                "task": "grounding",
                "messages": [
                    {"role": "user", "content": f"<image>Click on {action} in the image."},
                    {"role": "assistant", "content": f"Action: click(start_box='<|box_start|>({cx},{cy})<|box_end|>')"}
                ],
                "images": [os.path.join(self.pages_dir, f"{page_id}.png")],
                "bbox_norm": self._bbox_to_normalized(bbox),
                "source": "grounding"
            }
            samples.append(sample)
        
        return samples
    
    def generate_captioning_samples(self, num_samples: int = 2320) -> List[Dict]:
        """Generate Icon Captioning samples (coordinates -> text)."""
        samples = []
        
        all_icons = []
        for page_id, page_data in self.pages.items():
            for trans in page_data.get('transitions', []):
                action = trans['action']
                if action in ['back', 'home']:
                    continue
                bbox = trans.get('icon_bbox', [0, 0, 0, 0])
                all_icons.append((page_id, action, bbox))
        
        sampled = random.choices(all_icons, k=num_samples)
        
        for idx, (page_id, action, bbox) in enumerate(sampled):
            cx, cy = self._bbox_center_normalized(bbox)
            
            sample = {
                "idx": idx,
                "task": "captioning",
                "messages": [
                    {"role": "user", "content": f"<image>What is the icon at point ({cx}, {cy}) in the image?"},
                    {"role": "assistant", "content": action}
                ],
                "images": [os.path.join(self.pages_dir, f"{page_id}.png")],
                "bbox_norm": self._bbox_to_normalized(bbox),
                "source": "captioning"
            }
            samples.append(sample)
        
        return samples
    
    def generate_st_rl_samples(self, subtrees: List[int]) -> List[Dict]:
        """Generate ST-RL samples (edge-only, problem/solution format)."""
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
                        "bbox_norm": norm_bbox,
                        "image": os.path.join(self.pages_dir, f"{page_id}.png"),
                        "problem": f"<image>Instruction: from {page_id} to {target}. History: Null",
                        "solution": f"Explain: click {action} icon on {page_id}.\tAction: click(start_box='<|box_start|>({cx},{cy})<|box_end|>')",
                        "source": f"sub{subtree_idx}_edge"
                    }
                    samples.append(sample)
                    idx += 1
        
        return samples
    
    def generate_mt_rl_samples(self, subtrees: List[int], max_per_subtree: int = 1100) -> List[Dict]:
        """Generate MT-RL samples (path starting points)."""
        samples = []
        idx = 0
        
        for subtree_idx in subtrees:
            subtree_pages = list(self._get_pages_in_subtrees([subtree_idx]))
            count = 0
            attempted = set()
            
            while count < max_per_subtree and len(attempted) < len(subtree_pages) ** 2:
                start = random.choice(subtree_pages)
                end = random.choice(subtree_pages)
                
                if start == end or (start, end) in attempted:
                    continue
                attempted.add((start, end))
                
                path = self._get_path_with_actions(start, end)
                if not path:
                    continue
                
                from_page, action, bbox = path[0]
                norm_bbox = self._bbox_to_normalized(bbox)
                cx, cy = self._bbox_center_normalized(bbox)
                
                sample = {
                    "idx": idx,
                    "path": len(path),
                    "task": f"From {start} to {end}",
                    "bbox_norm": norm_bbox,
                    "image": os.path.join(self.pages_dir, f"{from_page}.png"),
                    "problem": f"<image>Instruction: from {start} to {end}. History: Null",
                    "solution": f"Explain: click {action} icon on {from_page}.\tAction: click(start_box='<|box_start|>({cx},{cy})<|box_end|>')",
                    "source": f"sub{subtree_idx}"
                }
                samples.append(sample)
                idx += 1
                count += 1
        
        return samples


def calculate_expected_dataset_size(nodes_per_level: List[int]):
    """Validate expected dataset size using NetworkX."""
    G = nx.DiGraph()
    
    # Build tree
    levels = {0: ["page_0"]}
    G.add_node("page_0", level=0)
    current_idx = 1
    
    for level, num_children in enumerate(nodes_per_level):
        levels[level + 1] = []
        for parent in levels[level]:
            for _ in range(num_children):
                child_id = f"page_{current_idx}"
                G.add_node(child_id, level=level + 1)
                G.add_edge(parent, child_id, action="forward")
                levels[level + 1].append(child_id)
                current_idx += 1
    
    # Add back/home edges
    for node in G.nodes():
        if node == "page_0":
            continue
        level = G.nodes[node]['level']
        
        # Back: find parent
        for pred in G.predecessors(node):
            if G.nodes[pred]['level'] == level - 1:
                G.add_edge(node, pred, action="back")
                break
        
        # Home: level >= 2
        if level >= 2:
            G.add_edge(node, "page_0", action="home")
    
    # Calculate path statistics
    total_instances = 0
    path_lengths = []
    
    for start in G.nodes():
        for end in G.nodes():
            if start == end:
                continue
            try:
                path = nx.shortest_path(G, start, end)
                instances = len(path)  # N nodes = N samples (N-1 clicks + 1 complete)
                total_instances += instances
                path_lengths.append(len(path))
            except nx.NetworkXNoPath:
                pass
    
    return {
        "total_nodes": G.number_of_nodes(),
        "total_edges": G.number_of_edges(),
        "total_pairs": len(path_lengths),
        "avg_path_length": sum(path_lengths) / len(path_lengths) if path_lengths else 0,
        "total_instances": total_instances
    }


def main():
    parser = argparse.ArgumentParser(description="Generate aligned dataset")
    parser.add_argument("--env_dir", type=str, required=True,
                       help="Path to UI environment directory")
    parser.add_argument("--output_dir", type=str, default="datas",
                       help="Output directory")
    parser.add_argument("--validate", action="store_true",
                       help="Run validation only")
    args = parser.parse_args()
    
    if args.validate:
        print("=== Dataset Size Validation ===")
        result = calculate_expected_dataset_size([5, 3, 2, 2, 1, 1])
        print(f"Total nodes: {result['total_nodes']}")
        print(f"Total edges: {result['total_edges']}")
        print(f"Total pairs: {result['total_pairs']}")
        print(f"Avg path length: {result['avg_path_length']:.2f}")
        print(f"Total instances: {result['total_instances']}")
        return
    
    ui_structure = os.path.join(args.env_dir, "ui_structure.json")
    ui_layer = os.path.join(args.env_dir, "ui_structure_layer.json")
    pages_dir = os.path.join(args.env_dir, "pages")
    
    generator = AlignedDatasetGenerator(ui_structure, ui_layer, pages_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate SFT dataset
    print("\n=== Generating SFT Dataset ===")
    print("Path samples from subtrees 0-1...")
    path_samples = generator.generate_path_samples(subtrees=[0, 1])
    print(f"  Total path samples: {len(path_samples)}")
    
    print("Edge samples from all subtrees (0-4)...")
    edge_samples = generator.generate_edge_samples(subtrees=[0, 1, 2, 3, 4])
    print(f"  Total edge samples: {len(edge_samples)}")
    
    print("Grounding samples...")
    grounding_samples = generator.generate_grounding_samples(2320)
    print(f"  Total grounding samples: {len(grounding_samples)}")
    
    print("Captioning samples...")
    captioning_samples = generator.generate_captioning_samples(2320)
    print(f"  Total captioning samples: {len(captioning_samples)}")
    
    # Combine SFT
    sft_samples = path_samples + edge_samples + grounding_samples + captioning_samples
    for i, s in enumerate(sft_samples):
        s['idx'] = i
    
    sft_path = os.path.join(args.output_dir, "sft_aligned.json")
    with open(sft_path, 'w') as f:
        json.dump(sft_samples, f, indent=2)
    print(f"\nSFT dataset saved: {sft_path} ({len(sft_samples)} samples)")
    
    # Generate ST-RL dataset
    print("\n=== Generating ST-RL Dataset ===")
    st_rl_samples = generator.generate_st_rl_samples(subtrees=[2, 3])
    
    st_rl_path = os.path.join(args.output_dir, "st_rl_aligned.json")
    with open(st_rl_path, 'w') as f:
        json.dump(st_rl_samples, f, indent=2)
    print(f"ST-RL dataset saved: {st_rl_path} ({len(st_rl_samples)} samples)")
    
    # Generate MT-RL dataset
    print("\n=== Generating MT-RL Dataset ===")
    mt_rl_samples = generator.generate_mt_rl_samples(subtrees=[2, 3], max_per_subtree=1100)
    
    mt_rl_path = os.path.join(args.output_dir, "mt_rl_aligned.json")
    with open(mt_rl_path, 'w') as f:
        json.dump(mt_rl_samples, f, indent=2)
    print(f"MT-RL dataset saved: {mt_rl_path} ({len(mt_rl_samples)} samples)")
    
    # Generate Test dataset
    print("\n=== Generating Test Dataset ===")
    test_samples = generator.generate_path_samples(subtrees=[4])
    
    test_path = os.path.join(args.output_dir, "test_aligned.json")
    with open(test_path, 'w') as f:
        json.dump(test_samples, f, indent=2)
    print(f"Test dataset saved: {test_path} ({len(test_samples)} samples)")
    
    # Summary
    print("\n=== Summary ===")
    print(f"SFT: {len(sft_samples)} samples")
    print(f"  - Path: {len(path_samples)}")
    print(f"  - Edge: {len(edge_samples)}")
    print(f"  - Grounding: {len(grounding_samples)}")
    print(f"  - Captioning: {len(captioning_samples)}")
    print(f"ST-RL: {len(st_rl_samples)} samples")
    print(f"MT-RL: {len(mt_rl_samples)} samples")
    print(f"Test: {len(test_samples)} samples")


if __name__ == "__main__":
    main()
