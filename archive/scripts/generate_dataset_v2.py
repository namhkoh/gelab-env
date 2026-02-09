#!/usr/bin/env python3
"""
Data generation script matching original GE-Lab format exactly.

Generates:
- sft.json: Multi-turn SFT data with source field, NO bbox_norm
- test.json: Single-step test data with bbox AND bbox_norm, NO source
- st_rl.json: Single-turn RL data with bbox_norm, source with _edge suffix
- mt_rl.json: Multi-turn RL data with bbox_norm, source WITHOUT _edge suffix
"""

import json
import os
import random
import shutil
from collections import defaultdict, deque
from typing import List, Dict, Tuple, Optional
from pathlib import Path

# Constants matching original
CANVAS_WIDTH = 1179
CANVAS_HEIGHT = 2556
NORM_SIZE = 1000


class DatasetGenerator:
    def __init__(self, ui_structure_path: str, pages_dir: str):
        with open(ui_structure_path, 'r') as f:
            self.ui_data = json.load(f)
        
        self.pages = self.ui_data['pages']
        self.pages_dir = pages_dir
        
        # Build adjacency lists
        self._build_adjacency()
        self._compute_depths()
        
    def _build_adjacency(self):
        """Build forward-only and full adjacency lists."""
        self.adj_forward = defaultdict(list)  # Forward edges only
        self.adj_full = defaultdict(list)      # All edges (including back/home)
        
        for page_id, page_data in self.pages.items():
            for t in page_data.get('transitions', []):
                action = t['action']
                target = t['target_page']
                bbox = t.get('icon_bbox', t.get('bbox', [0, 0, 0, 0]))
                
                self.adj_full[page_id].append({
                    'action': action,
                    'target': target,
                    'bbox': bbox
                })
                
                if action not in ['back', 'home']:
                    self.adj_forward[page_id].append({
                        'action': action,
                        'target': target,
                        'bbox': bbox
                    })
    
    def _compute_depths(self):
        """Compute depth of each page from root."""
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
    
    def _find_path_bfs(self, start: str, end: str) -> Optional[List[Dict]]:
        """Find shortest path using BFS (allows all transitions)."""
        if start == end:
            return []
        
        queue = deque([(start, [])])
        visited = {start}
        
        while queue:
            current, path = queue.popleft()
            
            for edge in self.adj_full[current]:
                neighbor = edge['target']
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
    
    def _get_source_label(self, current_page: str) -> str:
        """Get source label based on page depth."""
        depth = self.depths.get(current_page, 0)
        return f"sub{depth}"
    
    def _format_action(self, action_name: str, page_id: str, bbox: List[int]) -> str:
        """Format action string with coordinates."""
        bbox_norm = self._normalize_bbox(bbox)
        x, y = self._bbox_center(bbox_norm)
        return f"Explain:click {action_name} icon on {page_id}.\tAction: click(start_box='<|box_start|>({x},{y})<|box_end|>')"
    
    def _format_action_lowercase(self, action_name: str, page_id: str, bbox: List[int]) -> str:
        """Format action string with lowercase 'explain' (for st_rl)."""
        bbox_norm = self._normalize_bbox(bbox)
        x, y = self._bbox_center(bbox_norm)
        return f"explain:click {action_name} icon on {page_id}.\tAction: click(start_box='<|box_start|>({x},{y})<|box_end|>')"
    
    def generate_sft_data(self, num_samples: int = 20000) -> List[Dict]:
        """Generate SFT data matching original format."""
        sft_data = []
        idx = 0
        
        all_pages = list(self.pages.keys())
        
        # Target samples per path (total / 7 paths / avg_path_len for multi-turn)
        # Each path generates (path_len + 1) samples, avg is ~4.5
        trajectories_per_path = num_samples // 7 // 5  # ~570 trajectories per path
        
        for target_path_len in range(1, 8):
            generated = 0
            attempts = 0
            max_attempts = trajectories_per_path * 50
            
            while generated < trajectories_per_path and attempts < max_attempts:
                attempts += 1
                
                # Pick random start and end
                start = random.choice(all_pages)
                end = random.choice(all_pages)
                
                if start == end:
                    continue
                
                path = self._find_path_bfs(start, end)
                if path is None or len(path) != target_path_len:
                    continue
                
                # Generate multi-turn samples for this path
                history_steps = []
                task = f"From {start} to {end}"
                
                for step_idx, step in enumerate(path):
                    current_page = step['from_page']
                    action_name = step['action']
                    bbox = step['bbox']
                    
                    # Build history string
                    if not history_steps:
                        history = "Null"
                    else:
                        history = ".".join([f"step{i+1}: click {h['action']} icon on {h['page']}" 
                                          for i, h in enumerate(history_steps)])
                    
                    # User message
                    user_content = f"<image>Instruction: from {start} to {end}. History: {history}"
                    
                    # Assistant message
                    assistant_content = self._format_action(action_name, current_page, bbox)
                    
                    sample = {
                        "idx": idx,
                        "path": target_path_len,
                        "task": task,
                        "messages": [
                            {"role": "user", "content": user_content},
                            {"role": "assistant", "content": assistant_content}
                        ],
                        "images": [f"datas/images/{current_page}.png"],
                        "source": self._get_source_label(current_page)
                    }
                    
                    sft_data.append(sample)
                    idx += 1
                    
                    # Update history
                    history_steps.append({'action': action_name, 'page': current_page})
                
                # Add final "complete" step
                final_page = path[-1]['to_page']
                if not history_steps:
                    history = "Null"
                else:
                    history = ".".join([f"step{i+1}: click {h['action']} icon on {h['page']}" 
                                      for i, h in enumerate(history_steps)])
                
                sample = {
                    "idx": idx,
                    "path": target_path_len,
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
                
                generated += 1
        
        return sft_data
    
    def generate_test_data(self, num_samples: int = 500) -> List[Dict]:
        """Generate test data matching original format (has bbox AND bbox_norm, NO source)."""
        test_data = []
        idx = 0
        
        all_pages = list(self.pages.keys())
        samples_per_path = num_samples // 7  # paths 1-7 including indirect navigation
        
        for target_path_len in range(1, 8):
            generated = 0
            attempts = 0
            max_attempts = samples_per_path * 100  # More attempts for harder paths
            
            while generated < samples_per_path and attempts < max_attempts:
                attempts += 1
                
                start = random.choice(all_pages)
                end = random.choice(all_pages)
                
                if start == end:
                    continue
                
                path = self._find_path_bfs(start, end)
                if path is None or len(path) != target_path_len:
                    continue
                
                # Use first step only for test (single-step evaluation)
                step = path[0]
                current_page = step['from_page']
                action_name = step['action']
                bbox = step['bbox']
                bbox_norm = self._normalize_bbox(bbox)
                x, y = self._bbox_center(bbox_norm)
                
                sample = {
                    "idx": idx,
                    "path": target_path_len,
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
                generated += 1
        
        return test_data
    
    def generate_st_rl_data(self, num_samples: int = 3000) -> List[Dict]:
        """Generate ST-RL data matching original format."""
        st_rl_data = []
        idx = 0
        
        all_pages = list(self.pages.keys())
        
        for _ in range(num_samples):
            # Pick pages with forward edges
            pages_with_edges = [p for p in all_pages if self.adj_forward[p]]
            if not pages_with_edges:
                continue
                
            current_page = random.choice(pages_with_edges)
            edge = random.choice(self.adj_forward[current_page])
            
            target = edge['target']
            action_name = edge['action']
            bbox = edge['bbox']
            bbox_norm = self._normalize_bbox(bbox)
            
            # Add small random noise to coordinates
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
        """Generate MT-RL data matching original format."""
        mt_rl_data = []
        idx = 0
        
        all_pages = list(self.pages.keys())
        samples_per_path = num_samples // 6
        
        for target_path_len in range(1, 7):
            generated = 0
            attempts = 0
            max_attempts = samples_per_path * 20
            
            while generated < samples_per_path and attempts < max_attempts:
                attempts += 1
                
                start = random.choice(all_pages)
                end = random.choice(all_pages)
                
                if start == end:
                    continue
                
                path = self._find_path_bfs(start, end)
                if path is None or len(path) != target_path_len:
                    continue
                
                # Use first step
                step = path[0]
                current_page = step['from_page']
                action_name = step['action']
                bbox = step['bbox']
                bbox_norm = self._normalize_bbox(bbox)
                x, y = self._bbox_center(bbox_norm)
                
                # Add small random noise
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
    
    parser = argparse.ArgumentParser(description='Generate datasets matching original GE-Lab format')
    parser.add_argument('--ui_structure', type=str, default='environment/demo/ui_structure.json')
    parser.add_argument('--pages_dir', type=str, default='environment/demo/pages')
    parser.add_argument('--output_dir', type=str, default='datas')
    parser.add_argument('--sft_samples', type=int, default=20000)
    parser.add_argument('--test_samples', type=int, default=500)
    parser.add_argument('--st_rl_samples', type=int, default=3000)
    parser.add_argument('--mt_rl_samples', type=int, default=3000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    print(f"Loading UI structure from {args.ui_structure}")
    generator = DatasetGenerator(args.ui_structure, args.pages_dir)
    
    print(f"Total pages: {len(generator.pages)}")
    print(f"Max depth: {max(generator.depths.values())}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Copy images
    images_dir = os.path.join(args.output_dir, 'images')
    if not os.path.exists(images_dir):
        print(f"Copying images to {images_dir}")
        shutil.copytree(args.pages_dir, images_dir)
    
    # Generate SFT data
    print(f"\nGenerating SFT data ({args.sft_samples} target samples)...")
    sft_data = generator.generate_sft_data(args.sft_samples)
    with open(os.path.join(args.output_dir, 'sft.json'), 'w') as f:
        json.dump(sft_data, f, indent=2)
    print(f"  Generated {len(sft_data)} SFT samples")
    
    # Generate test data
    print(f"\nGenerating test data ({args.test_samples} target samples)...")
    test_data = generator.generate_test_data(args.test_samples)
    with open(os.path.join(args.output_dir, 'test.json'), 'w') as f:
        json.dump(test_data, f, indent=2)
    print(f"  Generated {len(test_data)} test samples")
    
    # Generate ST-RL data
    print(f"\nGenerating ST-RL data ({args.st_rl_samples} target samples)...")
    st_rl_data = generator.generate_st_rl_data(args.st_rl_samples)
    with open(os.path.join(args.output_dir, 'st_rl.json'), 'w') as f:
        json.dump(st_rl_data, f, indent=2)
    print(f"  Generated {len(st_rl_data)} ST-RL samples")
    
    # Generate MT-RL data
    print(f"\nGenerating MT-RL data ({args.mt_rl_samples} target samples)...")
    mt_rl_data = generator.generate_mt_rl_data(args.mt_rl_samples)
    with open(os.path.join(args.output_dir, 'mt_rl.json'), 'w') as f:
        json.dump(mt_rl_data, f, indent=2)
    print(f"  Generated {len(mt_rl_data)} MT-RL samples")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
