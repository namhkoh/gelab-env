"""
Generate evaluation splits matching paper methodology:
- ID Edge/Path: from subtrees 0-1 (training data)
- OOD Edge/Path: from subtree 4 (test data)

Paper quote:
"The current environment root node comprises five subtrees: two are designated for SFT training, 
two for RL training, and one for OOD testing."
"""

import json
import os
import random
from collections import defaultdict


def get_subtree_pages(layer_structure):
    """
    Extract pages belonging to each subtree.
    Subtrees are the direct children of page_0 (root).
    """
    subtrees = {}
    root = layer_structure.get("root", {})
    
    def collect_pages(node):
        """Recursively collect all pages in a subtree."""
        pages = []
        image = node.get("image", "")
        if image:
            page_name = image.replace(".png", "")
            pages.append(page_name)
        for subnode in node.get("subnodes", []):
            pages.extend(collect_pages(subnode))
        return pages
    
    # Get direct children of root (these are the subtrees)
    for i, subtree_node in enumerate(root.get("subnodes", [])):
        subtrees[i] = collect_pages(subtree_node)
    
    return subtrees


def generate_edge_samples(ui_structure, pages, env_path, num_samples=None):
    """
    Generate Edge (single-step) evaluation samples.
    Each sample: click one icon to transition to adjacent page.
    """
    samples = []
    pages_data = ui_structure.get("pages", {})
    
    for page_name in pages:
        if page_name not in pages_data:
            continue
        
        page_info = pages_data[page_name]
        transitions = page_info.get("transitions", [])
        
        for trans in transitions:
            icon_name = trans.get("action", "")
            target_page = trans.get("target_page", "")
            bbox = trans.get("icon_bbox", [0, 0, 0, 0])
            
            if not icon_name or not target_page:
                continue
            
            # Skip system icons (back, home) for cleaner evaluation
            if icon_name.lower() in ["back", "home"]:
                continue
            
            # Calculate center coordinate (normalized 0-1000)
            center_x = int((bbox[0] + bbox[2]) / 2 / 448 * 1000)
            center_y = int((bbox[1] + bbox[3]) / 2 / 448 * 1000)
            
            # Normalize bbox to 0-1000
            bbox_norm = [
                int(bbox[0] / 448 * 1000),
                int(bbox[1] / 448 * 1000),
                int(bbox[2] / 448 * 1000),
                int(bbox[3] / 448 * 1000)
            ]
            
            image_path = os.path.join(env_path, "pages", f"{page_name}.png")
            
            sample = {
                "idx": len(samples),
                "type": "edge",
                "source_page": page_name,
                "target_page": target_page,
                "icon_name": icon_name,
                "bbox_norm": bbox_norm,
                "messages": [
                    {"role": "user", "content": f"<image>Instruction: from {page_name} to {target_page}. History: Null"},
                    {"role": "assistant", "content": f"Explain: click {icon_name} icon on {page_name}.\tAction: click(start_box='<|box_start|>({center_x},{center_y})<|box_end|>')"}
                ],
                "images": [image_path]
            }
            samples.append(sample)
    
    if num_samples and len(samples) > num_samples:
        samples = random.sample(samples, num_samples)
    
    return samples


def generate_path_samples(ui_structure, layer_structure, pages, env_path, max_samples=500):
    """
    Generate Path (multi-step) evaluation samples.
    Each sample: navigate from source to target through multiple steps.
    """
    samples = []
    pages_data = ui_structure.get("pages", {})
    
    # Build adjacency graph
    graph = defaultdict(list)
    for page_name, page_info in pages_data.items():
        for trans in page_info.get("transitions", []):
            target = trans.get("target_page")
            icon = trans.get("action")
            bbox = trans.get("icon_bbox", [0, 0, 0, 0])
            if target and icon:
                graph[page_name].append({
                    "target": target,
                    "icon": icon,
                    "bbox": bbox
                })
    
    # Generate path tasks using BFS
    def find_path(start, end, max_depth=7):
        """Find shortest path between two pages."""
        if start == end:
            return []
        
        visited = {start}
        queue = [(start, [])]
        
        while queue:
            current, path = queue.pop(0)
            
            if len(path) >= max_depth:
                continue
            
            for edge in graph.get(current, []):
                next_page = edge["target"]
                if next_page == end:
                    return path + [edge]
                if next_page not in visited:
                    visited.add(next_page)
                    queue.append((next_page, path + [edge]))
        
        return None
    
    # Generate tasks for different path lengths
    pages_list = [p for p in pages if p in pages_data]
    
    for path_length in range(1, 8):
        count = 0
        attempts = 0
        max_attempts = 1000
        
        while count < max_samples // 7 and attempts < max_attempts:
            attempts += 1
            start = random.choice(pages_list)
            end = random.choice(pages_list)
            
            if start == end:
                continue
            
            path = find_path(start, end, max_depth=path_length + 1)
            
            if path and len(path) == path_length:
                # Create sample for first step
                first_step = path[0]
                bbox = first_step["bbox"]
                icon_name = first_step["icon"]
                
                center_x = int((bbox[0] + bbox[2]) / 2 / 448 * 1000)
                center_y = int((bbox[1] + bbox[3]) / 2 / 448 * 1000)
                bbox_norm = [
                    int(bbox[0] / 448 * 1000),
                    int(bbox[1] / 448 * 1000),
                    int(bbox[2] / 448 * 1000),
                    int(bbox[3] / 448 * 1000)
                ]
                
                image_path = os.path.join(env_path, "pages", f"{start}.png")
                
                sample = {
                    "idx": len(samples),
                    "type": "path",
                    "path_length": path_length,
                    "source_page": start,
                    "target_page": end,
                    "full_path": [start] + [e["target"] for e in path],
                    "first_step_icon": icon_name,
                    "bbox_norm": bbox_norm,
                    "messages": [
                        {"role": "user", "content": f"<image>Instruction: from {start} to {end}. History: Null"},
                        {"role": "assistant", "content": f"Explain: click {icon_name} icon on {start}.\tAction: click(start_box='<|box_start|>({center_x},{center_y})<|box_end|>')"}
                    ],
                    "images": [image_path]
                }
                samples.append(sample)
                count += 1
    
    return samples


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_path", default="datas")
    parser.add_argument("--output_dir", default="datas")
    args = parser.parse_args()
    
    random.seed(42)

    # Load structures
    with open(os.path.join(args.env_path, "ui_structure.json")) as f:
        ui_structure = json.load(f)
    
    with open(os.path.join(args.env_path, "ui_structure_layer.json")) as f:
        layer_structure = json.load(f)
    
    # Get subtrees
    subtrees = get_subtree_pages(layer_structure)
    print(f"Found {len(subtrees)} subtrees:")
    for i, pages in subtrees.items():
        print(f"  Subtree {i}: {len(pages)} pages")
    
    # ID pages: subtrees 0-1 (SFT training)
    id_pages = subtrees.get(0, []) + subtrees.get(1, [])
    # OOD pages: subtree 4 (test)
    ood_pages = subtrees.get(4, subtrees.get(len(subtrees)-1, []))  # Use last subtree if 4 doesn't exist
    
    print(f"\nID pages (subtrees 0-1): {len(id_pages)}")
    print(f"OOD pages (subtree 4): {len(ood_pages)}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate ID Edge samples
    id_edge = generate_edge_samples(ui_structure, id_pages, args.env_path)
    with open(os.path.join(args.output_dir, "test_id_edge.json"), "w") as f:
        json.dump(id_edge, f, indent=2)
    print(f"\nGenerated ID Edge: {len(id_edge)} samples")
    
    # Generate OOD Edge samples
    ood_edge = generate_edge_samples(ui_structure, ood_pages, args.env_path)
    with open(os.path.join(args.output_dir, "test_ood_edge.json"), "w") as f:
        json.dump(ood_edge, f, indent=2)
    print(f"Generated OOD Edge: {len(ood_edge)} samples")
    
    # Generate ID Path samples
    id_path = generate_path_samples(ui_structure, layer_structure, id_pages, args.env_path)
    with open(os.path.join(args.output_dir, "test_id_path.json"), "w") as f:
        json.dump(id_path, f, indent=2)
    print(f"Generated ID Path: {len(id_path)} samples")
    
    # Generate OOD Path samples
    ood_path = generate_path_samples(ui_structure, layer_structure, ood_pages, args.env_path)
    with open(os.path.join(args.output_dir, "test_ood_path.json"), "w") as f:
        json.dump(ood_path, f, indent=2)
    print(f"Generated OOD Path: {len(ood_path)} samples")
    
    print("\n" + "="*50)
    print("Evaluation splits generated successfully!")
    print("="*50)
    print(f"\nFiles created in {args.output_dir}:")
    print("  - test_id_edge.json   (for ID Edge evaluation)")
    print("  - test_ood_edge.json  (for OOD Edge evaluation)")
    print("  - test_id_path.json   (for ID Path evaluation)")
    print("  - test_ood_path.json  (for OOD Path evaluation)")


if __name__ == "__main__":
    main()
