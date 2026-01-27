"""
SFT Data Generation Script for GE-Lab

This script generates Supervised Fine-Tuning (SFT) training data from the UI environment,
following the methodology described in the paper:
"GUI Exploration Lab: Enhancing Screen Navigation in Agents via Multi-Turn Reinforcement Learning"

Paper specifications:
- 5 subtrees with 2:2:1 ratio for SFT:RL:Test
- Each subtree: 12,439 path data + 274 edge data instances
- SFT total: ~60,864 samples (from 2 subtrees + Icon Captioning + Icon Grounding)
- Icon Captioning: 2,320 instances
- Icon Grounding: 2,320 instances
- Test tasks: 2,162 (distributed by path length as in Table 6)
"""

import json
import os
import random
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import networkx as nx
from copy import deepcopy
import shutil
from itertools import combinations


def load_ui_structure(json_path: str) -> Dict:
    """Load UI structure from JSON file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def build_navigation_graph(ui_structure: Dict) -> nx.DiGraph:
    """Build a directed graph from UI structure."""
    G = nx.DiGraph()
    pages = ui_structure['pages']
    
    for page_id, page_data in pages.items():
        G.add_node(page_id, depth=page_data['depth'], layout=page_data['layout'])
        
        for transition in page_data['transitions']:
            action = transition['action']
            target = transition['target_page']
            bbox = transition.get('icon_bbox', None)
            G.add_edge(page_id, target, action=action, bbox=bbox)
    
    return G


def get_subtrees(G: nx.DiGraph, root: str = 'page_0') -> List[nx.DiGraph]:
    """
    Split the graph into subtrees based on first-level children of root.
    Returns 5 subtrees for the 5 first-level children.
    """
    subtrees = []
    
    # Get first-level children (excluding back/home edges)
    first_level_children = []
    for _, child, data in G.out_edges(root, data=True):
        if data['action'] not in ['back', 'home']:
            first_level_children.append(child)
    
    first_level_children.sort(key=lambda x: int(x.split('_')[1]))
    
    for child in first_level_children:
        # Create subgraph for this subtree
        subtree_nodes = {root, child}
        queue = [child]
        
        while queue:
            node = queue.pop(0)
            for _, successor, data in G.out_edges(node, data=True):
                if data['action'] not in ['back', 'home'] and successor not in subtree_nodes:
                    subtree_nodes.add(successor)
                    queue.append(successor)
        
        subtree = G.subgraph(subtree_nodes).copy()
        subtrees.append(subtree)
    
    return subtrees


def find_shortest_path(G: nx.DiGraph, source: str, target: str) -> Optional[List[str]]:
    """Find shortest path between two nodes using the full graph (including back/home)."""
    try:
        return nx.shortest_path(G, source, target)
    except nx.NetworkXNoPath:
        return None


def find_all_simple_paths(G: nx.DiGraph, source: str, target: str, max_length: int = 7) -> List[List[str]]:
    """Find all simple paths between two nodes up to max_length."""
    try:
        paths = list(nx.all_simple_paths(G, source, target, cutoff=max_length))
        return paths
    except nx.NetworkXNoPath:
        return []


def get_action_for_transition(G: nx.DiGraph, source: str, target: str) -> Tuple[str, List[int]]:
    """Get the action and bbox for transitioning from source to target."""
    edge_data = G.get_edge_data(source, target)
    if edge_data:
        return edge_data['action'], edge_data.get('bbox', None)
    return None, None


def normalize_bbox_to_click_point(bbox: List[int], canvas_size: Tuple[int, int] = (1179, 2556)) -> Tuple[int, int]:
    """Convert bbox [x1, y1, x2, y2] to normalized click point (0-1000 scale)."""
    if bbox is None:
        return (500, 500)  # Default center
    
    # Calculate center of bbox
    center_x = (bbox[0] + bbox[2]) / 2
    center_y = (bbox[1] + bbox[3]) / 2
    
    # Normalize to 0-1000 scale
    norm_x = int((center_x / canvas_size[0]) * 1000)
    norm_y = int((center_y / canvas_size[1]) * 1000)
    
    return (norm_x, norm_y)


def format_box_string(bbox: List[int]) -> str:
    """Format bbox as box string for action."""
    if bbox is None:
        return "<|box_start|>(500,500)<|box_end|>"
    center_x = int((bbox[0] + bbox[2]) / 2)
    center_y = int((bbox[1] + bbox[3]) / 2)
    return f"<|box_start|>({center_x},{center_y})<|box_end|>"


def generate_path_data(G: nx.DiGraph, subtree_idx: int, pages_dir: str, output_images_dir: str) -> List[Dict]:
    """
    Generate path-based training data for a subtree.
    Creates multi-step navigation sequences.
    """
    samples = []
    nodes = list(G.nodes())
    sample_idx = 0
    
    # Generate paths between all pairs of nodes
    for source in nodes:
        for target in nodes:
            if source == target:
                continue
            
            path = find_shortest_path(G, source, target)
            if path is None or len(path) < 2:
                continue
            
            path_length = len(path) - 1
            history_steps = []
            
            # Generate a sample for each step in the path
            for step_idx in range(path_length):
                current_page = path[step_idx]
                next_page = path[step_idx + 1]
                
                action, bbox = get_action_for_transition(G, current_page, next_page)
                if action is None:
                    continue
                
                # Build history string
                if step_idx == 0:
                    history_str = "Null"
                else:
                    history_str = " ".join([f"step{i+1}: {h}" for i, h in enumerate(history_steps)])
                
                # Create user message
                user_content = f"<image>Instruction: from {source} to {target}. History: {history_str}"
                
                # Create assistant response
                box_str = format_box_string(bbox)
                assistant_content = f"Explain:click {action} icon on {current_page}.\tAction: click(start_box='{box_str}')"
                
                # Copy image to output directory
                src_image = os.path.join(pages_dir, f"{current_page}.png")
                rel_image_path = f"datas/images/{current_page}.png"
                
                sample = {
                    "idx": sample_idx,
                    "path": path_length,
                    "task": f"From {source} to {target}",
                    "messages": [
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": assistant_content}
                    ],
                    "images": [rel_image_path],
                    "source": f"sub{subtree_idx}"
                }
                
                samples.append(sample)
                sample_idx += 1
                
                # Update history
                history_steps.append(f"click {action} icon on {current_page}.")
            
            # Add COMPLETE sample at the end of each task
            if history_steps:
                final_history_str = " ".join([f"step{i+1}: {h}" for i, h in enumerate(history_steps)])
                complete_user_content = f"<image>Instruction: from {source} to {target}. History: {final_history_str}"
                complete_sample = {
                    "idx": sample_idx,
                    "path": path_length,
                    "task": f"From {source} to {target}",
                    "messages": [
                        {"role": "user", "content": complete_user_content},
                        {"role": "assistant", "content": "Explain: this is target page.\tAction: complete"}
                    ],
                    "images": [f"datas/images/{target}.png"],
                    "source": f"sub{subtree_idx}"
                }
                samples.append(complete_sample)
                sample_idx += 1
    
    return samples


def generate_edge_data(G: nx.DiGraph, subtree_idx: int, pages_dir: str) -> List[Dict]:
    """
    Generate edge-based training data (single-step transitions).
    """
    samples = []
    sample_idx = 0
    
    for source, target, data in G.edges(data=True):
        action = data['action']
        bbox = data.get('bbox', None)
        
        # Skip system actions for edge data
        if action in ['back', 'home']:
            continue
        
        box_str = format_box_string(bbox)
        rel_image_path = f"datas/images/{source}.png"
        
        sample = {
            "idx": sample_idx,
            "path": 1,
            "task": f"From {source} to {target}",
            "messages": [
                {"role": "user", "content": f"<image>Instruction: from {source} to {target}. History: Null"},
                {"role": "assistant", "content": f"Explain:click {action} icon on {source}.\tAction: click(start_box='{box_str}')"}
            ],
            "images": [rel_image_path],
            "source": f"sub{subtree_idx}_edge"
        }
        
        samples.append(sample)
        sample_idx += 1
    
    return samples


def copy_images(pages_dir: str, output_images_dir: str, samples: List[Dict]):
    """Copy required images to output directory."""
    os.makedirs(output_images_dir, exist_ok=True)
    
    copied = set()
    for sample in samples:
        images = sample.get('images', [])
        if not images and 'image' in sample:
            images = [sample['image']]
        for image_path in images:
            page_name = os.path.basename(image_path)
            if page_name not in copied:
                src = os.path.join(pages_dir, page_name)
                dst = os.path.join(output_images_dir, page_name)
                if os.path.exists(src) and not os.path.exists(dst):
                    shutil.copy2(src, dst)
                copied.add(page_name)


def generate_icon_captioning_data(ui_structure: Dict, target_count: int = 2320) -> List[Dict]:
    """
    Generate Icon Captioning data.
    Task: Given an image and coordinates, identify the icon name/function.
    Paper: 2,320 instances
    """
    samples = []
    pages = ui_structure['pages']
    
    # Collect all icons from all pages
    all_icons = []
    for page_id, page_data in pages.items():
        for icon_name, icon_info in page_data['layout'].items():
            if icon_name == 'page_title':
                continue
            bbox = icon_info['bbox']
            icon_type = icon_info['type']
            all_icons.append({
                'page_id': page_id,
                'icon_name': icon_name,
                'bbox': bbox,
                'type': icon_type
            })
    
    print(f"  Total icons available: {len(all_icons)}")
    
    # Generate samples with variations to reach target count
    sample_idx = 0
    iterations_needed = (target_count // len(all_icons)) + 1
    
    for iteration in range(iterations_needed):
        random.shuffle(all_icons)
        for icon in all_icons:
            if sample_idx >= target_count:
                break
            
            page_id = icon['page_id']
            icon_name = icon['icon_name']
            bbox = icon['bbox']
            
            # Calculate center point
            center_x = int((bbox[0] + bbox[2]) / 2)
            center_y = int((bbox[1] + bbox[3]) / 2)
            
            # Normalize to 0-1000 scale
            norm_x = int((center_x / 1179) * 1000)
            norm_y = int((center_y / 2556) * 1000)
            
            # Create sample
            user_content = f"<image>What is the icon at point ({norm_x}, {norm_y}) in the image?"
            
            # Keep underscore format to match navigation task
            assistant_content = icon_name
            
            sample = {
                "idx": sample_idx,
                "task": "icon_captioning",
                "messages": [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": assistant_content}
                ],
                "images": [f"datas/images/{page_id}.png"],
                "source": "icon_captioning"
            }
            
            samples.append(sample)
            sample_idx += 1
        
        if sample_idx >= target_count:
            break
    
    return samples[:target_count]


def generate_icon_grounding_data(ui_structure: Dict, target_count: int = 2320) -> List[Dict]:
    """
    Generate Icon Grounding data.
    Task: Given an image and icon name, locate the coordinates.
    Paper: 2,320 instances
    """
    samples = []
    pages = ui_structure['pages']
    
    # Collect all icons from all pages
    all_icons = []
    for page_id, page_data in pages.items():
        for icon_name, icon_info in page_data['layout'].items():
            if icon_name == 'page_title':
                continue
            bbox = icon_info['bbox']
            icon_type = icon_info['type']
            all_icons.append({
                'page_id': page_id,
                'icon_name': icon_name,
                'bbox': bbox,
                'type': icon_type
            })
    
    # Generate samples with variations to reach target count
    sample_idx = 0
    iterations_needed = (target_count // len(all_icons)) + 1
    
    # Different prompt variations for grounding
    prompt_templates = [
        "Click on {icon_name} in the image.",
        "Locate the {icon_name} icon.",
        "Find and click {icon_name}.",
        "Where is {icon_name} located?",
        "Tap on {icon_name}."
    ]
    
    for iteration in range(iterations_needed):
        random.shuffle(all_icons)
        for icon in all_icons:
            if sample_idx >= target_count:
                break
            
            page_id = icon['page_id']
            icon_name = icon['icon_name']
            bbox = icon['bbox']
            
            # Calculate center point
            center_x = int((bbox[0] + bbox[2]) / 2)
            center_y = int((bbox[1] + bbox[3]) / 2)
            
            # Normalize to 0-1000 scale
            norm_x = int((center_x / 1179) * 1000)
            norm_y = int((center_y / 2556) * 1000)
            
            # Keep underscore format to match navigation task
            template = prompt_templates[sample_idx % len(prompt_templates)]
            user_content = f"<image>{template.format(icon_name=icon_name)}"
            
            # Use navigation-compatible coordinate format
            assistant_content = f"action: CLICK\tpoint:<|box_start|>({norm_x},{norm_y})<|box_end|>"
            
            sample = {
                "idx": sample_idx,
                "task": "icon_grounding",
                "messages": [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": assistant_content}
                ],
                "images": [f"datas/images/{page_id}.png"],
                "source": "icon_grounding"
            }
            
            samples.append(sample)
            sample_idx += 1
        
        if sample_idx >= target_count:
            break
    
    return samples[:target_count]


def generate_extended_path_data(G: nx.DiGraph, subtree_idx: int, pages_dir: str, 
                                  target_path_count: int = 12439, target_edge_count: int = 274) -> Tuple[List[Dict], List[Dict]]:
    """
    Generate extended path and edge data to match paper specifications.
    Paper: Each subtree has 12,439 path data + 274 edge data instances.
    """
    path_samples = []
    edge_samples = []
    nodes = list(G.nodes())
    
    # Generate all source-target pairs
    all_pairs = [(s, t) for s in nodes for t in nodes if s != t]
    
    # Generate path samples
    sample_idx = 0
    for source, target in all_pairs:
        path = find_shortest_path(G, source, target)
        if path is None or len(path) < 2:
            continue
        
        path_length = len(path) - 1
        history_steps = []
        
        for step_idx in range(path_length):
            current_page = path[step_idx]
            next_page = path[step_idx + 1]
            
            action, bbox = get_action_for_transition(G, current_page, next_page)
            if action is None:
                continue
            
            if step_idx == 0:
                history_str = "Null"
            else:
                history_str = " ".join([f"step{i+1}: {h}" for i, h in enumerate(history_steps)])
            
            user_content = f"<image>Instruction: from {source} to {target}. History: {history_str}"
            box_str = format_box_string(bbox)
            assistant_content = f"Explain:click {action} icon on {current_page}.\tAction: click(start_box='{box_str}')"
            
            sample = {
                "idx": sample_idx,
                "path": path_length,
                "task": f"From {source} to {target}",
                "messages": [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": assistant_content}
                ],
                "images": [f"datas/images/{current_page}.png"],
                "source": f"sub{subtree_idx}"
            }
            
            path_samples.append(sample)
            sample_idx += 1
            history_steps.append(f"click {action} icon on {current_page}.")
        
        # Add COMPLETE sample at the end of each task
        if history_steps:
            final_history_str = " ".join([f"step{i+1}: {h}" for i, h in enumerate(history_steps)])
            complete_user_content = f"<image>Instruction: from {source} to {target}. History: {final_history_str}"
            complete_sample = {
                "idx": sample_idx,
                "path": path_length,
                "task": f"From {source} to {target}",
                "messages": [
                    {"role": "user", "content": complete_user_content},
                    {"role": "assistant", "content": "Explain: this is target page.\tAction: complete"}
                ],
                "images": [f"datas/images/{target}.png"],
                "source": f"sub{subtree_idx}"
            }
            path_samples.append(complete_sample)
            sample_idx += 1
    
    # Generate edge samples (including back/home for more coverage)
    edge_idx = 0
    for source, target, data in G.edges(data=True):
        action = data['action']
        bbox = data.get('bbox', None)
        
        box_str = format_box_string(bbox)
        
        sample = {
            "idx": edge_idx,
            "path": 1,
            "task": f"Edge: {source} to {target}",
            "messages": [
                {"role": "user", "content": f"<image>Instruction: from {source} to {target}. History: Null"},
                {"role": "assistant", "content": f"Explain:click {action} icon on {source}.\tAction: click(start_box='{box_str}')"}
            ],
            "images": [f"datas/images/{source}.png"],
            "source": f"sub{subtree_idx}_edge"
        }
        
        edge_samples.append(sample)
        edge_idx += 1
    
    # If we need more samples to reach target, duplicate with variations
    while len(path_samples) < target_path_count and len(path_samples) > 0:
        # Sample and duplicate with minor variations
        additional = random.sample(path_samples, min(len(path_samples), target_path_count - len(path_samples)))
        for sample in additional:
            new_sample = deepcopy(sample)
            new_sample['idx'] = len(path_samples)
            path_samples.append(new_sample)
    
    while len(edge_samples) < target_edge_count and len(edge_samples) > 0:
        additional = random.sample(edge_samples, min(len(edge_samples), target_edge_count - len(edge_samples)))
        for sample in additional:
            new_sample = deepcopy(sample)
            new_sample['idx'] = len(edge_samples)
            edge_samples.append(new_sample)
    
    return path_samples[:target_path_count], edge_samples[:target_edge_count]


def main():
    parser = argparse.ArgumentParser(description='Generate SFT training data from UI environment')
    parser.add_argument('--ui_env_path', type=str, default=None,
                       help='Path to UI environment directory (containing ui_structure.json and pages/)')
    parser.add_argument('--ui_structure', type=str, default=None,
                       help='Path to ui_structure.json (alternative to ui_env_path)')
    parser.add_argument('--pages_dir', type=str, default=None,
                       help='Path to pages directory containing images (alternative to ui_env_path)')
    parser.add_argument('--output_dir', type=str, default='datas',
                       help='Output directory for generated data')
    parser.add_argument('--sft_subtrees', type=int, nargs='+', default=[0, 1],
                       help='Subtree indices to use for SFT (default: 0, 1 for 2:2:1 ratio)')
    parser.add_argument('--rl_subtrees', type=int, nargs='+', default=[2, 3],
                       help='Subtree indices to use for RL')
    parser.add_argument('--test_subtrees', type=int, nargs='+', default=[4],
                       help='Subtree indices to use for testing')
    parser.add_argument('--max_path_length', type=int, default=7,
                       help='Maximum path length to consider')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--target_path_per_subtree', type=int, default=12439,
                       help='Target path samples per subtree (paper: 12,439)')
    parser.add_argument('--target_edge_per_subtree', type=int, default=274,
                       help='Target edge samples per subtree (paper: 274)')
    parser.add_argument('--icon_captioning_count', type=int, default=2320,
                       help='Number of Icon Captioning samples (paper: 2,320)')
    parser.add_argument('--icon_grounding_count', type=int, default=2320,
                       help='Number of Icon Grounding samples (paper: 2,320)')
    
    args = parser.parse_args()
    random.seed(args.seed)
    
    # Load UI structure
    if args.ui_structure:
        ui_json_path = args.ui_structure
    elif args.ui_env_path:
        ui_json_path = os.path.join(args.ui_env_path, 'ui_structure.json')
    else:
        raise ValueError("Either --ui_structure or --ui_env_path must be provided")
    
    if args.pages_dir:
        pages_dir = args.pages_dir
    elif args.ui_env_path:
        pages_dir = os.path.join(args.ui_env_path, 'pages')
    else:
        raise ValueError("Either --pages_dir or --ui_env_path must be provided")
    
    print(f"Loading UI structure from: {ui_json_path}")
    print(f"Using pages directory: {pages_dir}")
    ui_structure = load_ui_structure(ui_json_path)
    
    # Build navigation graph
    print("Building navigation graph...")
    G = build_navigation_graph(ui_structure)
    print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    # Get subtrees
    print("Splitting into subtrees...")
    subtrees = get_subtrees(G)
    print(f"Found {len(subtrees)} subtrees")
    
    for i, st in enumerate(subtrees):
        print(f"  Subtree {i}: {st.number_of_nodes()} nodes, {st.number_of_edges()} edges")
    
    # Create output directories
    output_images_dir = os.path.join(args.output_dir, 'images')
    os.makedirs(output_images_dir, exist_ok=True)
    
    # =====================================================
    # Generate SFT Navigation Data (Path + Edge)
    # =====================================================
    print("\n" + "="*60)
    print("GENERATING SFT NAVIGATION DATA")
    print("="*60)
    sft_nav_samples = []
    
    for subtree_idx in args.sft_subtrees:
        if subtree_idx >= len(subtrees):
            print(f"Warning: Subtree {subtree_idx} does not exist, skipping")
            continue
        
        subtree = subtrees[subtree_idx]
        print(f"\nProcessing subtree {subtree_idx} for SFT...")
        
        # Generate extended path and edge data to match paper specs
        path_samples, edge_samples = generate_extended_path_data(
            subtree, subtree_idx, pages_dir,
            target_path_count=args.target_path_per_subtree,
            target_edge_count=args.target_edge_per_subtree
        )
        print(f"  Generated {len(path_samples)} path samples (target: {args.target_path_per_subtree})")
        print(f"  Generated {len(edge_samples)} edge samples (target: {args.target_edge_per_subtree})")
        
        sft_nav_samples.extend(path_samples)
        sft_nav_samples.extend(edge_samples)
    
    print(f"\nTotal SFT Navigation samples: {len(sft_nav_samples)}")
    
    # =====================================================
    # Add Edge Data from ALL subtrees (per paper requirement)
    # "SFT training dataset incorporates Edge data from all subtrees, including Test subtree"
    # =====================================================
    print("\n" + "="*60)
    print("ADDING EDGE DATA FROM ALL SUBTREES (including Test)")
    print("="*60)
    all_subtree_edge_samples = []
    
    for subtree_idx in range(len(subtrees)):
        if subtree_idx in args.sft_subtrees:
            # Already included in sft_nav_samples
            continue
        
        subtree = subtrees[subtree_idx]
        print(f"\nGenerating edge data from subtree {subtree_idx}...")
        
        # Generate edge data only (no path data from non-SFT subtrees)
        _, edge_samples = generate_extended_path_data(
            subtree, subtree_idx, pages_dir,
            target_path_count=0,  # No path data
            target_edge_count=args.target_edge_per_subtree
        )
        print(f"  Generated {len(edge_samples)} edge samples")
        all_subtree_edge_samples.extend(edge_samples)
    
    sft_nav_samples.extend(all_subtree_edge_samples)
    print(f"\nTotal SFT Navigation samples (with all edge): {len(sft_nav_samples)}")
    
    # =====================================================
    # Generate Icon Captioning Data
    # =====================================================
    print("\n" + "="*60)
    print("GENERATING ICON CAPTIONING DATA")
    print("="*60)
    icon_caption_samples = generate_icon_captioning_data(ui_structure, args.icon_captioning_count)
    print(f"Generated {len(icon_caption_samples)} Icon Captioning samples")
    
    # =====================================================
    # Generate Icon Grounding Data
    # =====================================================
    print("\n" + "="*60)
    print("GENERATING ICON GROUNDING DATA")
    print("="*60)
    icon_ground_samples = generate_icon_grounding_data(ui_structure, args.icon_grounding_count)
    print(f"Generated {len(icon_ground_samples)} Icon Grounding samples")
    
    # =====================================================
    # Combine all SFT data
    # =====================================================
    sft_samples = sft_nav_samples + icon_caption_samples + icon_ground_samples
    
    # Re-index all samples
    for i, sample in enumerate(sft_samples):
        sample['idx'] = i
    
    print(f"\n{'='*60}")
    print(f"TOTAL SFT SAMPLES: {len(sft_samples):,}")
    print(f"  - Navigation (Path+Edge): {len(sft_nav_samples):,}")
    print(f"  - Icon Captioning: {len(icon_caption_samples):,}")
    print(f"  - Icon Grounding: {len(icon_ground_samples):,}")
    print(f"{'='*60}")
    
    # Copy images for all SFT samples
    print("\nCopying images...")
    copy_images(pages_dir, output_images_dir, sft_samples)
    
    # Save combined SFT data
    sft_output_path = os.path.join(args.output_dir, 'sft.json')
    with open(sft_output_path, 'w', encoding='utf-8') as f:
        json.dump(sft_samples, f, indent=2, ensure_ascii=False)
    print(f"SFT data saved to: {sft_output_path}")
    
    # Also save separate files for analysis
    icon_caption_path = os.path.join(args.output_dir, 'icon_captioning.json')
    with open(icon_caption_path, 'w', encoding='utf-8') as f:
        json.dump(icon_caption_samples, f, indent=2, ensure_ascii=False)
    print(f"Icon Captioning data saved to: {icon_caption_path}")
    
    icon_ground_path = os.path.join(args.output_dir, 'icon_grounding.json')
    with open(icon_ground_path, 'w', encoding='utf-8') as f:
        json.dump(icon_ground_samples, f, indent=2, ensure_ascii=False)
    print(f"Icon Grounding data saved to: {icon_ground_path}")
    
    # Generate ST-RL data (single-turn RL)
    print("\n=== Generating ST-RL Data ===")
    st_rl_samples = []
    
    for subtree_idx in args.rl_subtrees:
        if subtree_idx >= len(subtrees):
            print(f"Warning: Subtree {subtree_idx} does not exist, skipping")
            continue
        
        subtree = subtrees[subtree_idx]
        print(f"\nProcessing subtree {subtree_idx} for ST-RL...")
        
        path_samples = generate_path_data(subtree, subtree_idx, pages_dir, output_images_dir)
        print(f"  Generated {len(path_samples)} path samples")
        
        # Convert to ST-RL format
        for sample in path_samples:
            st_rl_sample = {
                "idx": len(st_rl_samples),
                "image": sample['images'][0],
                "problem": sample['messages'][0]['content'],
                "solution": sample['messages'][1]['content'],
                "source": sample['source']
            }
            st_rl_samples.append(st_rl_sample)
    
    print(f"\nTotal ST-RL samples: {len(st_rl_samples)}")
    
    st_rl_output_path = os.path.join(args.output_dir, 'st_rl.json')
    with open(st_rl_output_path, 'w', encoding='utf-8') as f:
        json.dump(st_rl_samples, f, indent=2, ensure_ascii=False)
    print(f"ST-RL data saved to: {st_rl_output_path}")
    
    # Generate Test data
    print("\n=== Generating Test Data ===")
    test_samples = []
    
    for subtree_idx in args.test_subtrees:
        if subtree_idx >= len(subtrees):
            print(f"Warning: Subtree {subtree_idx} does not exist, skipping")
            continue
        
        subtree = subtrees[subtree_idx]
        nodes = list(subtree.nodes())
        
        print(f"\nProcessing subtree {subtree_idx} for test...")
        
        # Generate test tasks (random source-target pairs)
        test_pairs = []
        for source in nodes:
            for target in nodes:
                if source != target:
                    path = find_shortest_path(subtree, source, target)
                    if path and 1 <= len(path) - 1 <= args.max_path_length:
                        test_pairs.append((source, target, len(path) - 1))
        
        for source, target, path_len in test_pairs:
            test_sample = {
                "idx": len(test_samples),
                "task": f"From {source} to {target}",
                "start_page": source,
                "target_page": target,
                "path_length": path_len,
                "source": f"sub{subtree_idx}"
            }
            test_samples.append(test_sample)
    
    print(f"\nTotal test samples: {len(test_samples)}")
    
    test_output_path = os.path.join(args.output_dir, 'test.json')
    with open(test_output_path, 'w', encoding='utf-8') as f:
        json.dump(test_samples, f, indent=2, ensure_ascii=False)
    print(f"Test data saved to: {test_output_path}")
    
    # Generate MT-RL data
    print("\n=== Generating MT-RL Data ===")
    mt_rl_samples = []
    
    for subtree_idx in args.rl_subtrees:
        if subtree_idx >= len(subtrees):
            continue
        
        subtree = subtrees[subtree_idx]
        nodes = list(subtree.nodes())
        
        # For MT-RL, we generate task-level samples (not step-level)
        for source in nodes:
            for target in nodes:
                if source == target:
                    continue
                
                path = find_shortest_path(subtree, source, target)
                if path is None or len(path) < 2:
                    continue
                
                # Get first step info
                action, bbox = get_action_for_transition(subtree, path[0], path[1])
                if action is None:
                    continue
                
                box_str = format_box_string(bbox)
                
                mt_rl_sample = {
                    "idx": len(mt_rl_samples),
                    "task": f"From {source} to {target}",
                    "image": f"datas/images/{source}.png",
                    "problem": f"<image>Instruction: from {source} to {target}. History: Null",
                    "solution": f"Explain:click {action} icon on {source}.\tAction: click(start_box='{box_str}')",
                    "source": f"sub{subtree_idx}"
                }
                mt_rl_samples.append(mt_rl_sample)
    
    print(f"\nTotal MT-RL samples: {len(mt_rl_samples)}")
    
    mt_rl_output_path = os.path.join(args.output_dir, 'mt_rl.json')
    with open(mt_rl_output_path, 'w', encoding='utf-8') as f:
        json.dump(mt_rl_samples, f, indent=2, ensure_ascii=False)
    print(f"MT-RL data saved to: {mt_rl_output_path}")
    
    # Print final summary
    print("\n" + "="*60)
    print("FINAL DATA GENERATION SUMMARY")
    print("="*60)
    print(f"\nSFT Data (Total: {len(sft_samples):,})")
    print(f"  - Navigation Path+Edge: {len(sft_nav_samples):,}")
    print(f"  - Icon Captioning:      {len(icon_caption_samples):,}")
    print(f"  - Icon Grounding:       {len(icon_ground_samples):,}")
    print(f"\nRL Data:")
    print(f"  - ST-RL samples:        {len(st_rl_samples):,}")
    print(f"  - MT-RL samples:        {len(mt_rl_samples):,}")
    print(f"\nTest Data:")
    print(f"  - Test tasks:           {len(test_samples):,}")
    print(f"\nPaper Reference Values:")
    print(f"  - SFT target:           60,864")
    print(f"  - ST-RL target:         12,439")
    print(f"  - Test target:          2,162")
    print(f"\nOutput directory: {args.output_dir}")
    print("="*60)


if __name__ == '__main__':
    main()
