"""
Dataset Generation Script for GE-Lab

Generates training data from UI environment structure:
- Edge data: Single-step transitions
- Path data: Multi-step navigation trajectories
- Allocates data by subtree for SFT/RL/Test splits

Usage:
    python generate_dataset.py --env_dir <path_to_ui_environment> --output_dir <output_path>
"""

import json
import os
import argparse
import random
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional
import copy


def load_ui_structure(env_dir: str) -> Dict:
    """Load UI structure from environment directory."""
    structure_path = os.path.join(env_dir, "ui_structure.json")
    with open(structure_path, "r") as f:
        return json.load(f)


def build_navigation_graph(pages: Dict) -> Dict[str, List[Tuple[str, str, List[int]]]]:
    """Build adjacency list from page transitions."""
    graph = defaultdict(list)
    for page_id, page_data in pages.items():
        for transition in page_data.get("transitions", []):
            action = transition["action"]
            target = transition["target_page"]
            bbox = transition.get("icon_bbox", [0, 0, 0, 0])
            graph[page_id].append((target, action, bbox))
    return graph


def get_subtree_pages(pages: Dict, root_children: List[str]) -> Dict[int, set]:
    """Identify which pages belong to which subtree."""
    subtrees = {}
    
    for idx, child in enumerate(root_children):
        subtree_pages = set()
        queue = deque([child])
        
        while queue:
            page = queue.popleft()
            if page in subtree_pages:
                continue
            subtree_pages.add(page)
            
            for transition in pages.get(page, {}).get("transitions", []):
                target = transition["target_page"]
                action = transition["action"]
                if action not in ["back", "home"] and target != "page_0":
                    queue.append(target)
        
        subtrees[idx] = subtree_pages
    
    return subtrees


def find_shortest_path(graph: Dict, start: str, end: str, pages: Dict) -> Optional[List[Tuple[str, str, List[int]]]]:
    """Find shortest path between two pages using BFS."""
    if start == end:
        return []
    
    queue = deque([(start, [])])
    visited = {start}
    
    while queue:
        current, path = queue.popleft()
        
        for target, action, bbox in graph.get(current, []):
            if target == end:
                return path + [(current, action, bbox, target)]
            
            if target not in visited:
                visited.add(target)
                queue.append((target, path + [(current, action, bbox, target)]))
    
    return None


def normalize_bbox(bbox: List[int], canvas_w: int = 1179, canvas_h: int = 2556) -> List[int]:
    """Normalize bbox to 0-1000 coordinate system."""
    x1, y1, x2, y2 = bbox
    return [
        int(x1 * 1000 / canvas_w),
        int(y1 * 1000 / canvas_h),
        int(x2 * 1000 / canvas_w),
        int(y2 * 1000 / canvas_h)
    ]


def bbox_center(bbox: List[int]) -> Tuple[int, int]:
    """Get center point of normalized bbox."""
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) // 2, (y1 + y2) // 2)


def generate_edge_data(pages: Dict, graph: Dict, subtree_pages: set, 
                       subtree_name: str, pages_dir: str) -> List[Dict]:
    """Generate single-step transition data."""
    edge_data = []
    idx = 0
    
    for page_id in subtree_pages:
        if page_id not in pages:
            continue
            
        page_data = pages[page_id]
        
        for transition in page_data.get("transitions", []):
            action = transition["action"]
            target = transition["target_page"]
            bbox = transition.get("icon_bbox", [0, 0, 0, 0])
            
            if action in ["back", "home"]:
                continue
            
            norm_bbox = normalize_bbox(bbox)
            cx, cy = bbox_center(norm_bbox)
            
            problem = f"<image>Instruction: from {page_id} to {target}. History: Null"
            solution = f"explain:click {action} icon on {page_id}.\tAction: click(start_box='<|box_start|>({cx},{cy})<|box_end|>')"
            
            edge_data.append({
                "idx": idx,
                "bbox_norm": norm_bbox,
                "image": os.path.join(pages_dir, f"{page_id}.png"),
                "problem": problem,
                "solution": solution,
                "source": f"{subtree_name}_edge"
            })
            idx += 1
    
    return edge_data


def generate_path_data_sft(pages: Dict, graph: Dict, subtree_pages: set,
                           subtree_name: str, pages_dir: str, 
                           max_paths: int = 15000) -> List[Dict]:
    """Generate multi-step path data for SFT."""
    path_data = []
    idx = 0
    pages_list = list(subtree_pages)
    
    attempted = set()
    
    while len(path_data) < max_paths and len(attempted) < len(pages_list) ** 2:
        start = random.choice(pages_list)
        end = random.choice(pages_list)
        
        if start == end or (start, end) in attempted:
            continue
        attempted.add((start, end))
        
        path = find_shortest_path(graph, start, end, pages)
        if path is None or len(path) == 0:
            continue
        
        path_length = len(path)
        history = "Null"
        
        for step_idx, (from_page, action, bbox, to_page) in enumerate(path):
            is_last = (step_idx == len(path) - 1)
            norm_bbox = normalize_bbox(bbox)
            cx, cy = bbox_center(norm_bbox)
            
            user_content = f"<image>Instruction: from {start} to {end}. History: {history}"
            
            if is_last and to_page == end:
                assistant_content = f"Explain:click {action} icon on {from_page}.\tAction: click(start_box='<|box_start|>({cx},{cy})<|box_end|>')"
            else:
                assistant_content = f"Explain:click {action} icon on {from_page}.\tAction: click(start_box='<|box_start|>({cx},{cy})<|box_end|>')"
            
            path_data.append({
                "idx": idx,
                "path": path_length,
                "task": f"From {start} to {end}",
                "messages": [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": assistant_content}
                ],
                "images": [os.path.join(pages_dir, f"{from_page}.png")],
                "source": subtree_name
            })
            idx += 1
            
            history = f"{history}; step{step_idx+1}: click {action} icon on {from_page}" if history != "Null" else f"step{step_idx+1}: click {action} icon on {from_page}"
        
        path_data.append({
            "idx": idx,
            "path": path_length,
            "task": f"From {start} to {end}",
            "messages": [
                {"role": "user", "content": f"<image>Instruction: from {start} to {end}. History: {history}"},
                {"role": "assistant", "content": "Explain: this is target page.\tAction: complete"}
            ],
            "images": [os.path.join(pages_dir, f"{end}.png")],
            "source": subtree_name
        })
        idx += 1
    
    return path_data


def generate_path_data_rl(pages: Dict, graph: Dict, subtree_pages: set,
                          subtree_name: str, pages_dir: str,
                          max_tasks: int = 8000) -> List[Dict]:
    """Generate path starting points for RL training."""
    path_data = []
    idx = 0
    pages_list = list(subtree_pages)
    
    attempted = set()
    
    while len(path_data) < max_tasks and len(attempted) < len(pages_list) ** 2:
        start = random.choice(pages_list)
        end = random.choice(pages_list)
        
        if start == end or (start, end) in attempted:
            continue
        attempted.add((start, end))
        
        path = find_shortest_path(graph, start, end, pages)
        if path is None or len(path) == 0:
            continue
        
        first_step = path[0]
        from_page, action, bbox, to_page = first_step
        norm_bbox = normalize_bbox(bbox)
        cx, cy = bbox_center(norm_bbox)
        
        path_data.append({
            "idx": idx,
            "path": len(path),
            "task": f"From {start} to {end}",
            "bbox_norm": norm_bbox,
            "image": os.path.join(pages_dir, f"{from_page}.png"),
            "problem": f"<image>Instruction: from {start} to {end}. History: Null",
            "solution": f"Explain:click {action} icon on {from_page}.\tAction: click(start_box='<|box_start|>({cx},{cy})<|box_end|>')",
            "source": subtree_name
        })
        idx += 1
    
    return path_data


def generate_test_data(pages: Dict, graph: Dict, subtree_pages: set,
                       subtree_name: str, pages_dir: str,
                       max_tests: int = 2500) -> List[Dict]:
    """Generate test data in SFT format."""
    test_data = []
    idx = 0
    pages_list = list(subtree_pages)
    
    attempted = set()
    
    while len(test_data) < max_tests and len(attempted) < len(pages_list) ** 2:
        start = random.choice(pages_list)
        end = random.choice(pages_list)
        
        if start == end or (start, end) in attempted:
            continue
        attempted.add((start, end))
        
        path = find_shortest_path(graph, start, end, pages)
        if path is None or len(path) == 0:
            continue
        
        first_step = path[0]
        from_page, action, bbox, to_page = first_step
        norm_bbox = normalize_bbox(bbox)
        cx, cy = bbox_center(norm_bbox)
        
        test_data.append({
            "idx": idx,
            "path": len(path),
            "task": f"From {start} to {end}",
            "bbox": bbox,
            "bbox_norm": norm_bbox,
            "messages": [
                {"role": "user", "content": f"<image>Instruction: from {start} to {end}. History: Null"},
                {"role": "assistant", "content": f"Explain:click {action} icon on {from_page}.\tAction: click(start_box='<|box_start|>({cx},{cy})<|box_end|>')"}
            ],
            "images": [os.path.join(pages_dir, f"{from_page}.png")]
        })
        idx += 1
    
    return test_data


def main():
    parser = argparse.ArgumentParser(description="Generate GE-Lab training dataset")
    parser.add_argument("--env_dir", type=str, required=True,
                        help="Path to UI environment directory")
    parser.add_argument("--output_dir", type=str, default="datas",
                        help="Output directory for generated datasets")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    print(f"Loading UI structure from {args.env_dir}...")
    structure = load_ui_structure(args.env_dir)
    pages = structure["pages"]
    
    print(f"Total pages: {len(pages)}")
    
    graph = build_navigation_graph(pages)
    
    root_children = []
    for transition in pages["page_0"].get("transitions", []):
        if transition["action"] not in ["back", "home"]:
            root_children.append(transition["target_page"])
    
    print(f"Root children (subtrees): {root_children}")
    
    subtrees = get_subtree_pages(pages, root_children)
    for idx, subtree in subtrees.items():
        print(f"  Subtree {idx} ({root_children[idx]}): {len(subtree)} pages")
    
    subtrees[5] = {"page_0"}
    
    pages_dir = os.path.join(args.env_dir, "pages")
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("\n=== Generating SFT Data ===")
    sft_data = []
    
    for subtree_idx in [0, 1]:
        subtree_name = f"sub{subtree_idx}"
        print(f"Processing subtree {subtree_idx} for SFT...")
        
        edge_data = generate_edge_data(pages, graph, subtrees[subtree_idx], 
                                       subtree_name, pages_dir)
        print(f"  Edge data: {len(edge_data)} samples")
        
        path_data = generate_path_data_sft(pages, graph, subtrees[subtree_idx],
                                           subtree_name, pages_dir, max_paths=12500)
        print(f"  Path data: {len(path_data)} samples")
        
        for sample in edge_data:
            sft_sample = {
                "idx": len(sft_data),
                "path": 1,
                "task": sample["problem"].split("Instruction: ")[1].split(". History")[0].replace("from ", "From ").replace(" to ", " to "),
                "messages": [
                    {"role": "user", "content": sample["problem"]},
                    {"role": "assistant", "content": sample["solution"].replace("explain:", "Explain:")}
                ],
                "images": [sample["image"]],
                "source": sample["source"]
            }
            sft_data.append(sft_sample)
        
        sft_data.extend(path_data)
    
    for subtree_idx in range(5):
        subtree_name = f"sub{subtree_idx}"
        edge_data = generate_edge_data(pages, graph, subtrees[subtree_idx],
                                       subtree_name, pages_dir)
        for sample in edge_data:
            sft_sample = {
                "idx": len(sft_data),
                "path": 1,
                "task": sample["problem"].split("Instruction: ")[1].split(". History")[0].replace("from ", "From ").replace(" to ", " to "),
                "messages": [
                    {"role": "user", "content": sample["problem"]},
                    {"role": "assistant", "content": sample["solution"].replace("explain:", "Explain:")}
                ],
                "images": [sample["image"]],
                "source": f"{subtree_name}_edge"
            }
            if subtree_idx not in [0, 1]:
                sft_data.append(sft_sample)
    
    sft_path = os.path.join(args.output_dir, "sft_full.json")
    with open(sft_path, "w") as f:
        json.dump(sft_data, f, indent=2)
    print(f"\nSFT data saved to {sft_path}: {len(sft_data)} samples")
    
    print("\n=== Generating ST-RL Data ===")
    st_rl_data = []
    
    for subtree_idx in [2, 3]:
        subtree_name = f"sub{subtree_idx}"
        print(f"Processing subtree {subtree_idx} for ST-RL...")
        
        edge_data = generate_edge_data(pages, graph, subtrees[subtree_idx],
                                       subtree_name, pages_dir)
        st_rl_data.extend(edge_data)
        print(f"  Added {len(edge_data)} edge samples")
    
    for i, sample in enumerate(st_rl_data):
        sample["idx"] = i
    
    st_rl_path = os.path.join(args.output_dir, "st_rl_full.json")
    with open(st_rl_path, "w") as f:
        json.dump(st_rl_data, f, indent=2)
    print(f"\nST-RL data saved to {st_rl_path}: {len(st_rl_data)} samples")
    
    print("\n=== Generating MT-RL Data ===")
    mt_rl_data = []
    
    for subtree_idx in [2, 3]:
        subtree_name = f"sub{subtree_idx}"
        print(f"Processing subtree {subtree_idx} for MT-RL...")
        
        path_data = generate_path_data_rl(pages, graph, subtrees[subtree_idx],
                                          subtree_name, pages_dir, max_tasks=1100)
        mt_rl_data.extend(path_data)
        print(f"  Added {len(path_data)} path tasks")
    
    for i, sample in enumerate(mt_rl_data):
        sample["idx"] = i
    
    mt_rl_path = os.path.join(args.output_dir, "mt_rl_full.json")
    with open(mt_rl_path, "w") as f:
        json.dump(mt_rl_data, f, indent=2)
    print(f"\nMT-RL data saved to {mt_rl_path}: {len(mt_rl_data)} samples")
    
    print("\n=== Generating Test Data ===")
    test_data = generate_test_data(pages, graph, subtrees[4], "test", 
                                   pages_dir, max_tests=2200)
    
    test_path = os.path.join(args.output_dir, "test_full.json")
    with open(test_path, "w") as f:
        json.dump(test_data, f, indent=2)
    print(f"\nTest data saved to {test_path}: {len(test_data)} samples")
    
    print("\n=== Summary ===")
    print(f"SFT training data: {len(sft_data)} samples")
    print(f"ST-RL training data: {len(st_rl_data)} samples")
    print(f"MT-RL training data: {len(mt_rl_data)} tasks")
    print(f"Test data: {len(test_data)} samples")
    print(f"\nAll datasets saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
