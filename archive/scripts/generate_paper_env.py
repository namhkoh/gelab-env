"""
Generate environment matching paper's exact specification:

Paper quote (Section 4.1 & Appendix):
- "tree-structured graph with maximum depth of 7"
- "branching structure follows a node distribution of [5,3,2,2,1,1,0] at each level"
- "five independent subtrees, allocated according to a 2:2:1 ratio for SFT, RL, and test"
- "each subtree contains 12,439 path data instances and 274 edge data instances"

Expected structure:
- Level 0 (root): 1 node (page_0)
- Level 1: 5 nodes (5 subtree roots)
- Level 2: 5 * 3 = 15 nodes
- Level 3: 15 * 2 = 30 nodes
- Level 4: 30 * 2 = 60 nodes
- Level 5: 60 * 1 = 60 nodes
- Level 6: 60 * 1 = 60 nodes (leaves, no children)
Total: 1 + 5 + 15 + 30 + 60 + 60 + 60 = 231 pages
Per subtree: 46 pages (1 + 3 + 6 + 12 + 12 + 12)
"""

import os
import sys
import json
from PIL import Image
import glob

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tree import DynamicTopoEnv, UIManager, load_icons_from_directory


def calculate_pages_per_subtree(nodes_per_level):
    """Calculate number of pages per subtree given the node distribution."""
    # nodes_per_level[0] is children of root (number of subtrees)
    # nodes_per_level[1:] defines the structure within each subtree
    
    subtree_distribution = nodes_per_level[1:]  # [3, 2, 2, 1, 1, 0]
    
    # Calculate pages in each subtree
    pages = 1  # subtree root
    multiplier = 1
    
    for children_per_node in subtree_distribution:
        if children_per_node == 0:
            break
        multiplier *= children_per_node
        pages += multiplier
    
    return pages


def count_required_icons(tree_depth, nodes_per_level):
    """Calculate minimum icons needed for the tree structure."""
    # We need icons for transitions between pages
    # Each non-leaf page needs icons equal to its number of children
    
    total_icons = 0
    nodes_at_level = 1  # Start with root
    
    for level in range(tree_depth - 1):
        children_per_node = nodes_per_level[level]
        total_icons += nodes_at_level * children_per_node
        nodes_at_level = nodes_at_level * children_per_node
    
    return total_icons


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate paper-aligned environment")
    parser.add_argument("--output_dir", default="ui_environment_448_paper", help="Output directory")
    parser.add_argument("--icon_dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "icons"), help="Directory containing icon images")
    args = parser.parse_args()
    
    # Paper's exact specification
    TREE_DEPTH = 7
    NODES_PER_LEVEL = [5, 3, 2, 2, 1, 1]  # Length must be tree_depth - 1
    
    # Calculate expected structure
    num_subtrees = NODES_PER_LEVEL[0]
    pages_per_subtree = calculate_pages_per_subtree(NODES_PER_LEVEL)
    total_pages = 1 + num_subtrees * pages_per_subtree  # 1 for root
    required_icons = count_required_icons(TREE_DEPTH, NODES_PER_LEVEL)
    
    print("="*60)
    print("PAPER ENVIRONMENT SPECIFICATION")
    print("="*60)
    print(f"Tree depth: {TREE_DEPTH}")
    print(f"Nodes per level: {NODES_PER_LEVEL}")
    print(f"Number of subtrees: {num_subtrees}")
    print(f"Pages per subtree: {pages_per_subtree}")
    print(f"Total pages: {total_pages}")
    print(f"Required icons (transitions): {required_icons}")
    print("="*60)
    
    # Load icons
    print(f"\nLoading icons from {args.icon_dir}...")
    available_icons = len(glob.glob(os.path.join(args.icon_dir, "*/PNG/*.png")))
    icons_to_load = min(required_icons + 20, available_icons)
    print(f"Available icons: {available_icons}, loading: {icons_to_load}")
    
    icons_data = load_icons_from_directory(
        args.icon_dir, 
        required_count=icons_to_load
    )
    # Unpack list of tuples
    icon_images = [item[0] for item in icons_data]
    func_descs = [item[1] for item in icons_data]
    print(f"Loaded {len(icon_images)} icons")
    
    if len(icon_images) < required_icons:
        print(f"ERROR: Need at least {required_icons} icons, but only have {len(icon_images)}")
        print("Please add more icons to the icons directory")
        return
    
    # Create environment
    print(f"\nGenerating environment...")
    env = DynamicTopoEnv(
        icon_images=icon_images,
        func_descs=func_descs,
        tree_depth=TREE_DEPTH,
        nodes_per_level=NODES_PER_LEVEL,
        is_random_node=False  # Exact structure, no randomness
    )
    
    # Verify structure
    actual_pages = len(env.transition_graph.nodes)
    print(f"Generated {actual_pages} pages (expected {total_pages})")
    
    if actual_pages != total_pages:
        print(f"WARNING: Page count mismatch!")
    
    # Count pages per subtree
    subtree_counts = {}
    for page_id in env.transition_graph.nodes:
        if page_id == "page_0":
            continue
        # Get subtree by finding which child of root this page belongs to
        # Simple heuristic: check the path from root
        path_to_root = []
        current = page_id
        while current != "page_0":
            parent = env.transition_graph.nodes[current]['page'].parent
            if parent is None:
                break
            path_to_root.append(parent)
            current = parent
        
        if path_to_root:
            subtree_root = path_to_root[-1] if path_to_root[-1] != "page_0" else path_to_root[-2] if len(path_to_root) > 1 else page_id
            # Find which subtree index
            children_of_root = [n for n in env.transition_graph.successors("page_0") 
                              if n not in ["page_0"]]
            for i, child in enumerate(sorted(children_of_root)):
                if subtree_root == child or page_id == child:
                    subtree_counts[i] = subtree_counts.get(i, 0) + 1
                    break
    
    print(f"\nSubtree distribution:")
    for i in range(num_subtrees):
        count = subtree_counts.get(i, 0) + 1  # +1 for subtree root
        print(f"  Subtree {i}: {count} pages")
    
    # Save environment
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.output_dir)
    env.save_environment_data(output_path)
    print(f"\nEnvironment saved to: {output_path}")
    
    # Create symlink to latest
    latest_link = os.path.join(os.path.dirname(output_path), "ui_environment_448", "latest")
    os.makedirs(os.path.dirname(latest_link), exist_ok=True)
    if os.path.islink(latest_link):
        os.unlink(latest_link)
    elif os.path.exists(latest_link):
        import shutil
        shutil.rmtree(latest_link)
    os.symlink(output_path, latest_link)
    print(f"Created symlink: {latest_link} -> {output_path}")
    
    print("\n" + "="*60)
    print("ENVIRONMENT GENERATION COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
