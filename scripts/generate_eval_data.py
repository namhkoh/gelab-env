#!/usr/bin/env python3
"""
Generate evaluation data matching the paper's Table 6 distribution:
- Path@1: 137 tasks
- Path@2: 147 tasks
- Path@3: 222 tasks
- Path@4: 324 tasks
- Path@5: 492 tasks
- Path@6: 456 tasks
- Path@7: 384 tasks
- Total: 2162 tasks

Each task generates multiple step-by-step samples for evaluation.
"""

import json
import os
import sys
import argparse
import random
from glob import glob
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


# Paper's Table 6 distribution
PAPER_TASK_DISTRIBUTION = {
    1: 137,
    2: 147,
    3: 222,
    4: 324,
    5: 492,
    6: 456,
    7: 384
}


def load_ui_structure(ui_dir: str = None, ui_structure_path: str = None) -> Dict:
    """Load UI structure from the environment directory or direct path."""
    
    if ui_structure_path:
        # Direct path to ui_structure.json (flat format)
        print(f"Using UI structure: {ui_structure_path}")
        with open(ui_structure_path, 'r') as f:
            ui_structure = json.load(f)
        ui_env_dir = os.path.dirname(ui_structure_path)
        return ui_structure, ui_env_dir
    
    # Find the most recent UI environment
    ui_env_dirs = sorted(glob(os.path.join(ui_dir, '*')), reverse=True)
    if not ui_env_dirs:
        raise FileNotFoundError(f"No UI environment found in {ui_dir}")
    
    ui_env_dir = ui_env_dirs[0]
    print(f"Using UI environment: {ui_env_dir}")
    
    # Load ui_structure_layer.json
    structure_path = os.path.join(ui_env_dir, 'ui_structure_layer.json')
    if not os.path.exists(structure_path):
        structure_path = os.path.join(ui_env_dir, 'ui_structure.json')
    
    with open(structure_path, 'r') as f:
        ui_structure = json.load(f)
    
    return ui_structure, ui_env_dir


def build_graph_from_structure(ui_structure: Dict) -> Dict:
    """Build adjacency graph and page info from UI structure."""
    
    graph = {}  # page_id -> list of (target_page, icon_name, bbox)
    page_info = {}  # page_id -> {depth, subtree_idx, ...}
    
    # Check if this is flat format (has 'pages' key) or hierarchical format (has 'root' key)
    if 'pages' in ui_structure:
        # Flat format (like demo/ui_structure.json)
        pages = ui_structure['pages']
        
        # Determine subtree indices based on page_0's children
        page_0 = pages.get('page_0', {})
        subtree_roots = []
        for trans in page_0.get('transitions', []):
            if trans['action'] not in ['home', 'back']:
                subtree_roots.append(trans['target_page'])
        subtree_roots.sort(key=lambda x: int(x.split('_')[1]))
        
        # Build graph
        for page_id, page_data in pages.items():
            depth = page_data.get('depth', 0)
            
            # Determine subtree index
            subtree_idx = -1
            if page_id == 'page_0':
                subtree_idx = -1
            elif page_id in subtree_roots:
                subtree_idx = subtree_roots.index(page_id)
            else:
                # Trace back to find subtree root
                for i, root in enumerate(subtree_roots):
                    root_num = int(root.split('_')[1])
                    page_num = int(page_id.split('_')[1])
                    # Simple heuristic: pages are numbered sequentially per subtree
                    if i < len(subtree_roots) - 1:
                        next_root_num = int(subtree_roots[i+1].split('_')[1])
                        if root_num <= page_num < next_root_num:
                            subtree_idx = i
                            break
                    else:
                        if page_num >= root_num:
                            subtree_idx = i
            
            page_info[page_id] = {'depth': depth, 'subtree_idx': subtree_idx}
            
            graph[page_id] = []
            for trans in page_data.get('transitions', []):
                target = trans.get('target_page', '')
                action = trans.get('action', 'unknown')
                bbox = trans.get('icon_bbox', [0, 0, 100, 100])
                if target:
                    graph[page_id].append((target, action, bbox))
        
        return graph, page_info
    
    # Hierarchical format (original)
    def process_node(node_data: Dict, depth: int = 0, subtree_idx: int = -1):
        # Get page_id from image name
        img_name = node_data.get('image', '')
        page_id = img_name.replace('.png', '') if img_name else f'page_unknown_{depth}'
        
        page_info[page_id] = {'depth': depth, 'subtree_idx': subtree_idx}
        
        # Process transitions
        transitions = node_data.get('transitions', [])
        graph[page_id] = []
        
        for trans in transitions:
            target = trans.get('target_page', '')
            action = trans.get('action', 'unknown')
            bbox = trans.get('icon_bbox', [0, 0, 100, 100])
            if target:
                graph[page_id].append((target, action, bbox))
        
        # Process subnodes recursively (subnodes is a list)
        subnodes = node_data.get('subnodes', [])
        for i, subnode in enumerate(subnodes):
            # For depth 1 (children of root), assign subtree index
            child_subtree_idx = i if depth == 0 else subtree_idx
            process_node(subnode, depth + 1, child_subtree_idx)
    
    # Start from root
    if 'root' in ui_structure:
        process_node(ui_structure['root'], 0, -1)
    
    return graph, page_info


def find_all_paths_bfs(graph: Dict, max_depth: int = 7) -> Dict[int, List[Tuple[str, str, List[str]]]]:
    """Find all paths grouped by length using BFS."""
    
    paths_by_length = defaultdict(list)
    
    # For each starting node
    for start in graph.keys():
        # BFS to find all reachable nodes
        visited = {start: [start]}
        queue = [(start, [start])]
        
        while queue:
            current, path = queue.pop(0)
            
            if len(path) - 1 > max_depth:
                continue
            
            # Record this path if it's more than just the start
            if len(path) > 1:
                path_length = len(path) - 1
                paths_by_length[path_length].append((start, current, path))
            
            # Explore neighbors
            for target, _, _ in graph.get(current, []):
                if target and target not in visited:
                    new_path = path + [target]
                    visited[target] = new_path
                    queue.append((target, new_path))
        
        # Also add paths via home button (for non-root pages to page_0)
        if start != 'page_0' and 'page_0' not in visited:
            paths_by_length[1].append((start, 'page_0', [start, 'page_0']))
    
    return dict(paths_by_length)


def get_subtree_pages(graph: Dict, page_info: Dict, subtree_idx: int) -> set:
    """Get pages belonging to a specific subtree."""
    
    subtree_pages = {'page_0'}  # Root is shared
    
    for page_id, info in page_info.items():
        if info.get('subtree_idx') == subtree_idx:
            subtree_pages.add(page_id)
    
    return subtree_pages


def find_path(graph: Dict, source: str, target: str, max_depth: int = 7) -> Optional[List[str]]:
    """Find shortest path from source to target using BFS."""
    
    if source == target:
        return None
    
    visited = {source: [source]}
    queue = [(source, [source])]
    
    while queue:
        current, path = queue.pop(0)
        
        if len(path) > max_depth + 1:
            continue
        
        # Get all possible next pages
        next_pages = []
        
        # Regular transitions from graph
        for next_page, _, _ in graph.get(current, []):
            if next_page:
                next_pages.append(next_page)
        
        # Home button (any non-root page can go to page_0)
        if current != 'page_0':
            next_pages.append('page_0')
        
        # Check each possible next page
        for next_page in next_pages:
            if next_page == target:
                return path + [target]
            
            if next_page not in visited:
                visited[next_page] = path + [next_page]
                queue.append((next_page, path + [next_page]))
    
    return None


def get_transition_info(graph: Dict, source: str, target: str) -> Tuple[str, List[int]]:
    """Get icon name and bbox for transition from source to target."""
    
    for next_page, icon_name, bbox in graph.get(source, []):
        if next_page == target:
            return icon_name, bbox
    
    # Check if it's a home button transition
    if target == 'page_0':
        return 'home', [959, 100, 1079, 220]
    
    return 'unknown', [500, 500, 600, 600]


def format_box_string(bbox: List[int]) -> str:
    """Format bounding box as click coordinates."""
    if not bbox or len(bbox) < 4:
        return "<|box_start|>(500,500)<|box_end|>"
    
    x_center = (bbox[0] + bbox[2]) // 2
    y_center = (bbox[1] + bbox[3]) // 2
    return f"<|box_start|>({x_center},{y_center})<|box_end|>"


def generate_task_samples(graph: Dict, source: str, target: str, 
                         path: List[str], task_idx: int, images_dir: str = 'datas/images') -> List[Dict]:
    """Generate step-by-step samples for a single navigation task."""
    
    samples = []
    history_steps = []
    path_length = len(path) - 1
    task_name = f"From {source} to {target}"
    
    for step_idx in range(len(path) - 1):
        current_page = path[step_idx]
        next_page = path[step_idx + 1]
        
        # Get action for this transition
        icon_name, bbox = get_transition_info(graph, current_page, next_page)
        box_str = format_box_string(bbox)
        
        # Build history string
        if not history_steps:
            history_str = "Null"
        else:
            history_str = " ".join([f"step{i+1}: {h}" for i, h in enumerate(history_steps)])
        
        # Create user prompt
        user_content = f"<image>Instruction: from {source} to {target}. History: {history_str}"
        
        # Create assistant response
        assistant_content = f"Explain:click {icon_name} icon on {current_page}.\tAction: click(start_box='{box_str}')"
        
        sample = {
            "idx": 0,  # Will be renumbered later
            "path": path_length,
            "task": task_name,
            "messages": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content}
            ],
            "images": [f"{images_dir}/{current_page}.png"],
            "bbox_norm": bbox if bbox else [0, 0, 100, 100]
        }
        samples.append(sample)
        
        # Update history for next step
        history_steps.append(f"click {icon_name} icon on {current_page}.")
    
    # Add completion step
    final_page = path[-1]
    
    history_str = " ".join([f"step{i+1}: {h}" for i, h in enumerate(history_steps)])
    user_content = f"<image>Instruction: from {source} to {target}. History: {history_str}"
    assistant_content = "Explain: this is target page.\tAction: complete"
    
    complete_sample = {
        "idx": 0,
        "path": path_length,
        "task": task_name,
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content}
        ],
        "images": [f"{images_dir}/{final_page}.png"],
        "bbox_norm": None
    }
    samples.append(complete_sample)
    
    return samples


def main():
    parser = argparse.ArgumentParser(description='Generate evaluation data matching paper Table 6')
    parser.add_argument('--ui_dir', type=str, default=None,
                       help='Directory containing UI environment')
    parser.add_argument('--ui_structure', type=str, default=None,
                       help='Direct path to ui_structure.json (alternative to ui_dir)')
    parser.add_argument('--images_dir', type=str, default='datas/images',
                       help='Directory containing page images')
    parser.add_argument('--output_dir', type=str, default='datas',
                       help='Output directory for evaluation data')
    parser.add_argument('--test_subtree', type=int, default=4,
                       help='Subtree index to use for testing (default: 4)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for task selection')
    
    args = parser.parse_args()
    random.seed(args.seed)
    
    if not args.ui_structure and not args.ui_dir:
        args.ui_dir = 'data_engine/ui_environment'
    
    print("Loading UI structure...")
    ui_structure, ui_env_dir = load_ui_structure(args.ui_dir, args.ui_structure)
    
    print("Building navigation graph...")
    graph, page_info = build_graph_from_structure(ui_structure)
    print(f"Graph has {len(graph)} pages")
    
    # Get test subtree pages
    print(f"\nGetting pages for test subtree {args.test_subtree}...")
    test_pages = get_subtree_pages(graph, page_info, args.test_subtree)
    print(f"Test subtree has {len(test_pages)} pages")
    
    # Find all possible paths within test subtree
    print("\nFinding all valid paths in test subtree...")
    paths_by_length = defaultdict(list)
    
    for source in test_pages:
        for target in test_pages:
            if source == target:
                continue
            
            path = find_path(graph, source, target)
            if path and len(path) >= 2:
                path_length = len(path) - 1
                if path_length <= 7:
                    paths_by_length[path_length].append((source, target, path))
    
    print("\nAvailable paths per length:")
    for length in range(1, 8):
        print(f"  Path@{length}: {len(paths_by_length[length])} available")
    
    # Sample tasks to match paper distribution
    print("\n=== Sampling tasks to match paper Table 6 ===")
    selected_tasks = []
    
    for path_length, target_count in PAPER_TASK_DISTRIBUTION.items():
        available = paths_by_length[path_length]
        
        if len(available) >= target_count:
            selected = random.sample(available, target_count)
        else:
            # If not enough unique paths, sample with replacement
            selected = random.choices(available, k=target_count) if available else []
            print(f"  Warning: Path@{path_length} has only {len(available)} paths, needed {target_count}")
        
        for source, target, path in selected:
            selected_tasks.append({
                'source': source,
                'target': target,
                'path': path,
                'path_length': path_length
            })
        
        print(f"  Path@{path_length}: selected {len(selected)} tasks (target: {target_count})")
    
    print(f"\nTotal tasks selected: {len(selected_tasks)}")
    
    # Generate step-by-step samples for all tasks
    print("\nGenerating step-by-step evaluation samples...")
    all_samples = []
    
    for task_idx, task in enumerate(selected_tasks):
        samples = generate_task_samples(
            graph, 
            task['source'], 
            task['target'], 
            task['path'],
            task_idx,
            args.images_dir
        )
        all_samples.extend(samples)
    
    # Renumber indices
    for i, sample in enumerate(all_samples):
        sample['idx'] = i
    
    # Print statistics
    print(f"\n=== Final Statistics ===")
    print(f"Total tasks: {len(selected_tasks)}")
    print(f"Total samples (steps): {len(all_samples)}")
    
    task_counts = defaultdict(int)
    for task in selected_tasks:
        task_counts[task['path_length']] += 1
    
    print("\nTask distribution:")
    for length in range(1, 8):
        paper_count = PAPER_TASK_DISTRIBUTION[length]
        actual_count = task_counts[length]
        match = "✓" if actual_count == paper_count else "✗"
        print(f"  Path@{length}: {actual_count} tasks (paper: {paper_count}) {match}")
    
    # Save evaluation data
    os.makedirs(args.output_dir, exist_ok=True)
    output_filename = f'eval_sub{args.test_subtree}_test.json'
    output_path = os.path.join(args.output_dir, output_filename)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_samples, f, indent=2, ensure_ascii=False)
    
    print(f"\nEvaluation data saved to: {output_path}")
    
    # Save task summary
    task_summary = {
        "total_tasks": len(selected_tasks),
        "total_samples": len(all_samples),
        "distribution": {str(k): task_counts[k] for k in range(1, 8)},
        "paper_target": {str(k): v for k, v in PAPER_TASK_DISTRIBUTION.items()},
        "seed": args.seed,
        "test_subtree": args.test_subtree
    }
    
    summary_path = os.path.join(args.output_dir, f'eval_sub{args.test_subtree}_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(task_summary, f, indent=2)
    
    print(f"Summary saved to: {summary_path}")


if __name__ == '__main__':
    main()
