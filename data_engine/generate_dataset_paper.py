#!/usr/bin/env python3
"""
Data generation matching paper's specification exactly.

Paper specs:
- 5 subtrees partitioned 2:2:1 (SFT:RL:Test)
- Per subtree: 12,439 path data + 274 edge data
- Test: 2,162 interactive tasks (all pairs from test subtree + root)
- Test distribution follows Table 6
"""

import json
import os
import random
import shutil
from collections import defaultdict, deque
from typing import List, Dict, Set, Tuple, Optional
from pathlib import Path

# Constants
CANVAS_WIDTH = 1179
CANVAS_HEIGHT = 2556
NORM_SIZE = 1000

# Paper's test task distribution (Table 6)
TEST_DISTRIBUTION = {
    1: 137, 2: 147, 3: 222, 4: 324, 5: 492, 6: 456, 7: 384
}


class PaperDatasetGenerator:
    def __init__(self, ui_structure_path: str, pages_dir: str):
        with open(ui_structure_path, 'r') as f:
            self.ui_data = json.load(f)
        
        self.pages = self.ui_data['pages']
        self.pages_dir = pages_dir
        
        # Build adjacency and identify subtrees
        self._build_adjacency()
        self._identify_subtrees()
        self._compute_depths()
        
    def _build_adjacency(self):
        """Build adjacency lists."""
        self.adj_forward = defaultdict(list)  # Forward edges only
        self.adj_full = defaultdict(list)      # All edges
        
        for page_id, page_data in self.pages.items():
            for t in page_data.get('transitions', []):
                action = t['action']
                target = t['target_page']
                bbox = t.get('icon_bbox', t.get('bbox', [0, 0, 0, 0]))
                
                edge = {
                    'action': action,
                    'target': target,
                    'bbox': bbox
                }
                
                self.adj_full[page_id].append(edge)
                
                if action not in ['back', 'home']:
                    self.adj_forward[page_id].append(edge)
    
    def _identify_subtrees(self):
        """Identify 5 subtrees from root's children."""
        self.subtrees = []
        
        for t in self.pages['page_0'].get('transitions', []):
            if t['action'] not in ['back', 'home']:
                root = t['target_page']
                pages = self._get_subtree_pages(root)
                self.subtrees.append({
                    'root': root,
                    'pages': pages
                })
        
        # Paper partition: 2:2:1 for SFT:RL:Test
        self.sft_subtrees = [0, 1]      # Subtrees 1, 2
        self.rl_subtrees = [2, 3]       # Subtrees 3, 4
        self.test_subtree = 4           # Subtree 5
        
        # Get page sets
        self.sft_pages = set()
        for idx in self.sft_subtrees:
            self.sft_pages |= self.subtrees[idx]['pages']
        self.sft_pages.add('page_0')  # Include root
        
        self.rl_pages = set()
        for idx in self.rl_subtrees:
            self.rl_pages |= self.subtrees[idx]['pages']
        self.rl_pages.add('page_0')  # Include root
        
        self.test_pages = self.subtrees[self.test_subtree]['pages'] | {'page_0'}
        
        print(f"SFT pages: {len(self.sft_pages)} (subtrees 1,2 + root)")
        print(f"RL pages: {len(self.rl_pages)} (subtrees 3,4 + root)")
        print(f"Test pages: {len(self.test_pages)} (subtree 5 + root)")
    
    def _get_subtree_pages(self, root: str) -> Set[str]:
        """Get all pages in subtree via forward edges."""
        visited = {root}
        queue = deque([root])
        
        while queue:
            current = queue.popleft()
            for edge in self.adj_forward[current]:
                neighbor = edge['target']
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        return visited
    
    def _compute_depths(self):
        """Compute depth of each page."""
        self.depths = {'page_0': 0}
        queue = deque(['page_0'])
        
        while queue:
            current = queue.popleft()
            for edge in self.adj_forward[current]:
                neighbor = edge['target']
                if neighbor not in self.depths:
                    self.depths[neighbor] = self.depths[current] + 1
                    queue.append(neighbor)
    
    def _normalize_bbox(self, bbox: List[int]) -> List[int]:
        """Normalize bbox to 0-1000 range."""
        if not bbox or len(bbox) != 4:
            return [0, 0, 0, 0]
        x1, y1, x2, y2 = bbox
        return [
            int(x1 * NORM_SIZE / CANVAS_WIDTH),
            int(y1 * NORM_SIZE / CANVAS_HEIGHT),
            int(x2 * NORM_SIZE / CANVAS_WIDTH),
            int(y2 * NORM_SIZE / CANVAS_HEIGHT)
        ]
    
    def _bbox_center(self, bbox_norm: List[int]) -> Tuple[int, int]:
        """Get center point of normalized bbox."""
        x1, y1, x2, y2 = bbox_norm
        return ((x1 + x2) // 2, (y1 + y2) // 2)
    
    def _find_path_bfs(self, start: str, end: str, allowed_pages: Set[str] = None) -> Optional[List[Dict]]:
        """Find shortest path using BFS."""
        if start == end:
            return []
        
        queue = deque([(start, [])])
        visited = {start}
        
        while queue:
            current, path = queue.popleft()
            
            for edge in self.adj_full[current]:
                neighbor = edge['target']
                
                # If allowed_pages specified, only use pages in that set
                if allowed_pages and neighbor not in allowed_pages:
                    continue
                
                new_path = path + [{
                    'from_page': current,
                    'action': edge['action'],
                    'to_page': neighbor,
                    'bbox': edge['bbox']
                }]
                
                if neighbor == end:
                    return new_path
                
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, new_path))
        
        return None
    
    def _get_source_label(self, page: str) -> str:
        """Get source label based on depth."""
        depth = self.depths.get(page, 0)
        return f"sub{depth}"
    
    def _format_action(self, action_name: str, page_id: str, bbox: List[int]) -> str:
        """Format action string."""
        bbox_norm = self._normalize_bbox(bbox)
        x, y = self._bbox_center(bbox_norm)
        return f"Explain:click {action_name} icon on {page_id}.\tAction: click(start_box='<|box_start|>({x},{y})<|box_end|>')"
    
    def generate_sft_data(self) -> List[Dict]:
        """Generate SFT data from SFT subtrees (paper: 2 subtrees × 12,439 path data)."""
        sft_data = []
        idx = 0
        
        sft_page_list = list(self.sft_pages)
        
        # Generate all possible (start, end) pairs within SFT pages
        all_pairs = []
        for start in sft_page_list:
            for end in sft_page_list:
                if start != end:
                    all_pairs.append((start, end))
        
        print(f"  Generating from {len(all_pairs)} possible pairs...")
        
        for start, end in all_pairs:
            path = self._find_path_bfs(start, end, self.sft_pages)
            if path is None:
                continue
            
            # Generate multi-turn samples for this trajectory
            history_steps = []
            task = f"From {start} to {end}"
            
            for step in path:
                current_page = step['from_page']
                action_name = step['action']
                bbox = step['bbox']
                
                # Build history
                if not history_steps:
                    history = "Null"
                else:
                    history = ".".join([f"step{i+1}: click {h['action']} icon on {h['page']}" 
                                      for i, h in enumerate(history_steps)])
                
                sample = {
                    "idx": idx,
                    "path": len(path),
                    "task": task,
                    "messages": [
                        {"role": "user", "content": f"<image>Instruction: from {start} to {end}. History: {history}"},
                        {"role": "assistant", "content": self._format_action(action_name, current_page, bbox)}
                    ],
                    "images": [f"datas/images/{current_page}.png"],
                    "source": self._get_source_label(current_page)
                }
                
                sft_data.append(sample)
                idx += 1
                history_steps.append({'action': action_name, 'page': current_page})
            
            # Add complete step
            final_page = path[-1]['to_page']
            history = ".".join([f"step{i+1}: click {h['action']} icon on {h['page']}" 
                              for i, h in enumerate(history_steps)])
            
            sample = {
                "idx": idx,
                "path": len(path),
                "task": task,
                "messages": [
                    {"role": "user", "content": f"<image>Instruction: from {start} to {end}. History: {history}"},
                    {"role": "assistant", "content": "Explain: this is target page.\tAction: complete"}
                ],
                "images": [f"datas/images/{final_page}.png"],
                "source": self._get_source_label(final_page)
            }
            sft_data.append(sample)
            idx += 1
        
        return sft_data
    
    def generate_test_data(self) -> List[Dict]:
        """Generate test data matching Table 6 distribution exactly."""
        test_data = []
        idx = 0
        
        test_page_list = list(self.test_pages)
        
        # Generate all possible pairs and compute path lengths
        pairs_by_length = defaultdict(list)
        
        for start in test_page_list:
            for end in test_page_list:
                if start != end:
                    path = self._find_path_bfs(start, end)
                    if path:
                        pairs_by_length[len(path)].append((start, end, path))
        
        print(f"  Found pairs by path length: {dict((k, len(v)) for k, v in sorted(pairs_by_length.items()))}")
        
        # Sample according to Table 6 distribution
        for path_len, target_count in TEST_DISTRIBUTION.items():
            available = pairs_by_length.get(path_len, [])
            if len(available) < target_count:
                print(f"    Path@{path_len}: Only {len(available)} available, need {target_count}")
                selected = available
            else:
                selected = random.sample(available, target_count)
            
            for start, end, path in selected:
                # Use first step for single-step test
                step = path[0]
                current_page = step['from_page']
                action_name = step['action']
                bbox = step['bbox']
                bbox_norm = self._normalize_bbox(bbox)
                
                sample = {
                    "idx": idx,
                    "path": path_len,
                    "task": f"From {start} to {end}",
                    "bbox": bbox,
                    "bbox_norm": bbox_norm,
                    "messages": [
                        {"role": "user", "content": f"<image>Instruction: from {start} to {end}. History: Null"},
                        {"role": "assistant", "content": self._format_action(action_name, current_page, bbox)}
                    ],
                    "images": [f"datas/images/{current_page}.png"]
                }
                
                test_data.append(sample)
                idx += 1
        
        return test_data
    
    def generate_st_rl_data(self, num_samples: int = 3000) -> List[Dict]:
        """Generate ST-RL data from RL subtrees."""
        st_rl_data = []
        idx = 0
        
        rl_page_list = [p for p in self.rl_pages if p != 'page_0']
        
        for _ in range(num_samples):
            # Pick page with forward edges
            pages_with_edges = [p for p in rl_page_list if self.adj_forward[p]]
            if not pages_with_edges:
                continue
            
            current_page = random.choice(pages_with_edges)
            edge = random.choice(self.adj_forward[current_page])
            
            target = edge['target']
            action_name = edge['action']
            bbox = edge['bbox']
            bbox_norm = self._normalize_bbox(bbox)
            
            # Add noise
            x, y = self._bbox_center(bbox_norm)
            x += random.randint(-10, 10)
            y += random.randint(-10, 10)
            x = max(0, min(1000, x))
            y = max(0, min(1000, y))
            
            sample = {
                "idx": idx,
                "bbox_norm": bbox_norm,
                "image": f"datas/images/{current_page}.png",
                "problem": f"<image>Instruction: from {current_page} to {target}. History: Null",
                "solution": f"explain:click {action_name} icon on {current_page}.\tAction: click(start_box='<|box_start|>({x},{y})<|box_end|>')",
                "source": f"{self._get_source_label(current_page)}_edge"
            }
            
            st_rl_data.append(sample)
            idx += 1
        
        return st_rl_data
    
    def generate_mt_rl_data(self, num_samples: int = 3000) -> List[Dict]:
        """Generate MT-RL data from RL subtrees."""
        mt_rl_data = []
        idx = 0
        
        rl_page_list = list(self.rl_pages)
        samples_per_path = num_samples // 7
        
        for target_path_len in range(1, 8):
            generated = 0
            attempts = 0
            
            while generated < samples_per_path and attempts < samples_per_path * 50:
                attempts += 1
                
                start = random.choice(rl_page_list)
                end = random.choice(rl_page_list)
                
                if start == end:
                    continue
                
                path = self._find_path_bfs(start, end, self.rl_pages)
                if path is None or len(path) != target_path_len:
                    continue
                
                step = path[0]
                current_page = step['from_page']
                action_name = step['action']
                bbox = step['bbox']
                bbox_norm = self._normalize_bbox(bbox)
                
                x, y = self._bbox_center(bbox_norm)
                x += random.randint(-10, 10)
                y += random.randint(-10, 10)
                x = max(0, min(1000, x))
                y = max(0, min(1000, y))
                
                sample = {
                    "idx": idx,
                    "path": target_path_len,
                    "task": f"From {start} to {end}",
                    "bbox_norm": bbox_norm,
                    "image": f"datas/images/{current_page}.png",
                    "problem": f"<image>Instruction: from {start} to {end}. History: Null",
                    "solution": f"Explain:click {action_name} icon on {current_page}.\tAction: click(start_box='<|box_start|>({x},{y})<|box_end|>')",
                    "source": self._get_source_label(current_page)
                }
                
                mt_rl_data.append(sample)
                idx += 1
                generated += 1
        
        return mt_rl_data


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--ui_structure', type=str, default='environment/demo/ui_structure.json')
    parser.add_argument('--pages_dir', type=str, default='environment/demo/pages')
    parser.add_argument('--output_dir', type=str, default='datas')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    print("Loading UI structure...")
    generator = PaperDatasetGenerator(args.ui_structure, args.pages_dir)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Copy images
    images_dir = os.path.join(args.output_dir, 'images')
    if os.path.exists(images_dir):
        print(f"Images already exist at {images_dir}")
    else:
        print(f"Copying images to {images_dir}")
        shutil.copytree(args.pages_dir, images_dir)
    
    # Generate SFT
    print("\nGenerating SFT data (from subtrees 1,2)...")
    sft_data = generator.generate_sft_data()
    with open(os.path.join(args.output_dir, 'sft.json'), 'w') as f:
        json.dump(sft_data, f, indent=2)
    print(f"  Generated {len(sft_data)} SFT samples")
    
    # Generate test
    print("\nGenerating test data (from subtree 5, matching Table 6)...")
    test_data = generator.generate_test_data()
    with open(os.path.join(args.output_dir, 'test.json'), 'w') as f:
        json.dump(test_data, f, indent=2)
    print(f"  Generated {len(test_data)} test samples")
    
    # Generate ST-RL
    print("\nGenerating ST-RL data (from subtrees 3,4)...")
    st_rl_data = generator.generate_st_rl_data()
    with open(os.path.join(args.output_dir, 'st_rl.json'), 'w') as f:
        json.dump(st_rl_data, f, indent=2)
    print(f"  Generated {len(st_rl_data)} ST-RL samples")
    
    # Generate MT-RL
    print("\nGenerating MT-RL data (from subtrees 3,4)...")
    mt_rl_data = generator.generate_mt_rl_data()
    with open(os.path.join(args.output_dir, 'mt_rl.json'), 'w') as f:
        json.dump(mt_rl_data, f, indent=2)
    print(f"  Generated {len(mt_rl_data)} MT-RL samples")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
