import random
from typing import Any, Dict, List, Optional, Tuple
import networkx as nx
from PIL import Image, ImageDraw, ImageEnhance
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import time
from io import BytesIO
from PIL import ImageFont
import glob
import re
# --------------------------
# Data Structure Definitions
# --------------------------
class UIElement:
    """UI element class, containing icon and function description"""
    # Icon size for 448x448 canvas (must match LayoutGenerator.ICON_WIDTH/HEIGHT)
    ICON_SIZE = 50
    
    def __init__(self, image: Image.Image, func: str):
        self.raw_image = fit_image_to_canvas(image, (self.ICON_SIZE, self.ICON_SIZE))
        self.func_desc = func
        self.used = False

    def mark_used(self):
        self.used = True

class UIPage:
    """UI page class"""
    def __init__(self, page_id: str, elements: List[UIElement], layout: Dict[str, Tuple[int, int]], parent: Optional[str] = None):
        self.page_id = page_id
        self.elements = elements
        self.layout = layout
        self.parent = parent


# --------------------------
# Item 1: Unified graph — state fingerprint for identical-state detection
# --------------------------
SYSTEM_LAYOUT_KEYS = frozenset({"back", "home", "page_title"})


def state_fingerprint_page(page: UIPage) -> str:
    """Canonical fingerprint for a UIPage (synthetic). Same layout => same state.
    Ignores back, home, page_title so that only content icons define the state."""
    parts = []
    for name, bbox in sorted(page.layout.items()):
        if name in SYSTEM_LAYOUT_KEYS:
            continue
        bbox_tuple = tuple(bbox) if hasattr(bbox, "__iter__") else bbox
        parts.append((name, bbox_tuple))
    return json.dumps(parts, sort_keys=True)


def state_fingerprint_layout(layout: Dict[str, List[int]]) -> str:
    """Canonical fingerprint for a serialized layout dict (trajectory / JSON).
    Same layout => same state. Use for deduplicating nodes when building the graph."""
    parts = []
    for name, bbox in sorted(layout.items()):
        if name in SYSTEM_LAYOUT_KEYS:
            continue
        if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            parts.append((name, tuple(int(x) for x in bbox[:4])))
        else:
            parts.append((name, tuple(bbox) if hasattr(bbox, "__iter__") else (0, 0, 0, 0)))
    return json.dumps(parts, sort_keys=True)


def fit_image_to_canvas(image: Image.Image, size: Tuple[int, int]) -> Image.Image:
    """Resize an icon into a fixed canvas without distorting the original aspect ratio."""
    canvas_w, canvas_h = size
    source = image.convert("RGBA")
    scale = min(canvas_w / max(source.width, 1), canvas_h / max(source.height, 1))
    resized_w = max(1, int(round(source.width * scale)))
    resized_h = max(1, int(round(source.height * scale)))
    resized = source.resize((resized_w, resized_h), Image.Resampling.LANCZOS)

    canvas = Image.new("RGBA", size, (0, 0, 0, 0))
    offset_x = (canvas_w - resized_w) // 2
    offset_y = (canvas_h - resized_h) // 2
    canvas.paste(resized, (offset_x, offset_y), resized)
    return canvas

# --------------------------
# Core Environment Class
# --------------------------
class DynamicTopoEnv:
    def __init__(self, 
                icon_images: List[Image.Image], 
                func_descs: List[str],
                tree_depth: int = 3,
                nodes_per_level: List[int] = None,
                is_random_node: bool = False):
        """
        Initialize simulation environment
        :param icon_images: List of icon images
        :param func_descs: List of corresponding function descriptions
        :param tree_depth: Maximum tree depth
        :param nodes_per_level: List of child nodes per level, length should be tree_depth-1
        :param is_random_node: Whether to randomly generate number of child nodes (not exceeding specified number)
        """
        # Input validation
        if len(icon_images) != len(func_descs):
            raise ValueError("Number of icons does not match number of function descriptions")
        
        if nodes_per_level is None:
            nodes_per_level = [2, 3]
        
        if len(nodes_per_level) != tree_depth - 1:
            raise ValueError(f"nodes_per_level length({len(nodes_per_level)}) must equal tree_depth-1({tree_depth-1})")
        
        # Initialize components
        self.ui_manager = UIManager(icon_images, func_descs)
        self.topo_generator = TopologyGenerator(tree_depth, nodes_per_level, is_random_node)
        self.render_engine = RenderEngine()
        self.transition_graph = nx.DiGraph()
      
        # Build environment
        self._build_environment()
        self.reset()
      
      
    def _build_environment(self):
        """Build complete environment topology"""
        hierarchy, pages = self.topo_generator.generate(self.ui_manager)
        self.transition_graph = TopologyBuilder.build(hierarchy, pages)
      
    def reset(self) -> Tuple[Image.Image, dict]:
        """Reset environment state"""
        self.current_page = "page_0"
        return self.get_observation()
  
    def get_observation(self) -> Tuple[Image.Image, dict]:
        """Get current observation"""
        page = self.transition_graph.nodes[self.current_page]['page']
        return self.render_engine.render(page), page.layout
  
    def step(self, action: str) -> Tuple[Tuple[Image.Image, dict], float, bool]:
        """
        Execute action
        :return: (observation image, layout info), reward, is_terminal
        """
        # Find valid transition
        new_page = self._find_transition(action)
        reward = self._calculate_reward(action, new_page)
      
        # Update state
        if new_page is not None:
            self.current_page = new_page
      
        return self.get_observation(), reward, False
  
    def _find_transition(self, action: str) -> Optional[str]:
        """Find valid transition target"""
        for successor in self.transition_graph.successors(self.current_page):
            edge_data = self.transition_graph.get_edge_data(self.current_page, successor)
            if edge_data['action'] == action:
                return successor
        return None
  
    def _calculate_reward(self, action: str, new_page: Optional[str]) -> float:
        """Calculate immediate reward"""
        if new_page is None:
            return -1.0  # Invalid operation penalty
        if new_page == "page_goal":
            return 10.0  # Example goal reward
        return -0.1  # Small penalty for normal operations (encourage shortest path)

    def visualize_topology(self, save_path: str = 'topology.png'):
        """Visualize UI interface transition relationships (only showing normal transitions, not system operations)"""
        # Get all nodes and their hierarchical relationships
        nodes_by_level = {}  # Store nodes at each level
        root = "page_0"
        nodes_by_level[0] = [root]
        
        # Use BFS to traverse all nodes and determine their levels
        visited = {root}
        queue = [(root, 0)]
        
        while queue:
            node, depth = queue.pop(0)
            children = [v for u, v, d in self.transition_graph.edges(node, data=True)
                       if d['action'] not in ['back', 'home'] and u == node]
            
            if depth + 1 not in nodes_by_level:
                nodes_by_level[depth + 1] = []
            
            for child in children:
                if child not in visited:
                    visited.add(child)
                    nodes_by_level[depth + 1].append(child)
                    queue.append((child, depth + 1))
        
        # Remove empty levels
        nodes_by_level = {k: v for k, v in nodes_by_level.items() if v}
        
        # Calculate canvas size and node size
        max_depth = max(nodes_by_level.keys())
        max_nodes_in_level = max(len(nodes) for nodes in nodes_by_level.values())
        
        # Dynamically adjust canvas size
        figsize_width = min(20, max(15, max_depth * 3))  # Adjust width based on depth
        figsize_height = min(15, max(10, max_nodes_in_level * 1.5))  # Adjust height based on max nodes per level
        
        # Modify: Adjust node size calculation method to make nodes smaller
        node_size = max(500, min(1000, 4000 / max_nodes_in_level))  # Originally max(1000, min(2000, 8000 / max_nodes_in_level))
        
        # Modify: Adjust font size calculation method to make text smaller
        font_size = max(4, min(8, 15 / max_nodes_in_level))  # Originally max(6, min(10, 20 / max_nodes_in_level))
        
        plt.figure(figsize=(figsize_width, figsize_height))
        
        # Create custom hierarchical layout
        pos = {}
        x_spacing = 2.0
        y_spacing = max(0.8, 2.0 / max_nodes_in_level)  # Dynamically adjust vertical spacing
        
        # Calculate node positions
        for level, nodes in nodes_by_level.items():
            # Modify: Sort nodes by their numeric part
            nodes.sort(key=lambda x: int(x.split('_')[1]))  # Convert 'page_X' to number for sorting
            count = len(nodes)
            for i, node in enumerate(nodes):
                x = level * x_spacing
                y = (i - (count - 1) / 2) * y_spacing
                pos[node] = (x, y)
        
        # Ensure all nodes have position information
        for node in self.transition_graph.nodes():
            if node not in pos:
                print(f"Warning: Node {node} has no position, assigning default position")
                pos[node] = (0, 0)
        
        # Draw nodes
        nx.draw_networkx_nodes(self.transition_graph, pos,
                              node_color='lightblue',
                              node_size=node_size)
        
        # Process edges
        normal_edges = [(u, v) for u, v, d in self.transition_graph.edges(data=True)
                        if d['action'] not in ['back', 'home']]
        
        # Modify: Reduce arrow size
        nx.draw_networkx_edges(self.transition_graph, pos,
                              edgelist=normal_edges,
                              edge_color='gray',
                              arrows=True,
                              arrowsize=10)  # Originally 15
        
        # Edge labels
        edge_labels = {(u, v): d['action']
                      for u, v, d in self.transition_graph.edges(data=True)
                      if d['action'] not in ['back', 'home']}
        
        # Adjust edge labels
        nx.draw_networkx_edge_labels(self.transition_graph, pos,
                                    edge_labels=edge_labels,
                                    font_size=font_size,
                                    bbox=dict(facecolor='white',
                                            edgecolor='none',
                                            alpha=0.7))
        
        # Node labels
        nx.draw_networkx_labels(self.transition_graph, pos, font_size=font_size)
        
        plt.title("UI Graph")
        plt.axis('off')
        plt.tight_layout()  # Automatically adjust layout
        plt.savefig(save_path, bbox_inches='tight', dpi=300, pad_inches=0.5)
        plt.close()

    def save_environment_data(self, output_dir: str = "output", seed: int = None,
                              extra_metadata: Optional[dict] = None):
        """Save environment data, including page images, transition relationships, and configuration parameters"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Create pages subdirectory
        pages_dir = os.path.join(output_dir, "pages")
        os.makedirs(pages_dir, exist_ok=True)
        
        # Save configuration parameters
        config_path = os.path.join(output_dir, "config.json")
        config_data = {
            "seed": seed,
            "tree_depth": self.topo_generator.max_depth,
            "nodes_per_level": self.topo_generator.nodes_per_level,
            "is_random_node": self.topo_generator.is_random_node,
            "canvas_size": LayoutGenerator.CANVAS_SIZE,
            "icon_size": (LayoutGenerator.ICON_WIDTH, LayoutGenerator.ICON_HEIGHT),
            "margin": LayoutGenerator.MARGIN,
            "top_margin": LayoutGenerator.TOP_MARGIN
        }
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
        
        # Modify: More comprehensive depth calculation
        node_depths = {}
        root = "page_0"
        node_depths[root] = 0
        
        # First get all nodes in the graph to ensure all nodes have depth
        all_nodes = list(self.transition_graph.nodes())
        
        # Use BFS to calculate depth of all nodes, including leaf nodes
        visited = {root}
        queue = [(root, 0)]
        
        while queue:
            node, depth = queue.pop(0)
            # Consider all outgoing edges, not excluding any action type
            children = [v for u, v in self.transition_graph.out_edges(node)]
            
            for child in children:
                if child not in visited:
                    visited.add(child)
                    node_depths[child] = depth + 1
                    queue.append((child, depth + 1))
        
        # Ensure all nodes have depth values
        for node in all_nodes:
            if node not in node_depths:
                # Try to find depth of the node through incoming edges
                parents = [u for u, v in self.transition_graph.in_edges(node)]
                if parents:
                    # Use depth of the first parent node + 1
                    parent = parents[0]
                    if parent in node_depths:
                        node_depths[node] = node_depths[parent] + 1
                    else:
                        # If parent node also has no depth, use default value
                        node_depths[node] = -1
                else:
                    # Isolated node uses default depth value
                    node_depths[node] = -1
        
        # Save all page images
        pages_data = {}
        for node_id, node_data in self.transition_graph.nodes(data=True):
            page = node_data['page']
            
            # Save page image to pages subdirectory
            image_filename = f"{node_id}.png"
            image_path = os.path.join(pages_dir, image_filename)
            page_image = self.render_engine.render(page)
            page_image.save(image_path)
            
            # Collect page data, path needs to include pages subdirectory
            pages_data[node_id] = {
                "image": f"{image_filename}",  # Modify reference path to relative path
                "depth": node_depths[node_id],
                "layout": {
                    icon.func_desc: {
                        "bbox": list(page.layout[icon.func_desc]),
                        "type": "system" if icon.func_desc in ['back', 'home'] else "normal"
                    }
                    for icon in page.elements
                },
                "transitions": []
            }
        
        # Add transition relationships, ensuring correct bbox information
        for u, v, data in self.transition_graph.edges(data=True):
            action = data['action']
            source_page = self.transition_graph.nodes[u]['page']
            
            # Find icon bbox that triggers the transition
            icon_bbox = None
            if action in source_page.layout:
                icon_bbox = list(source_page.layout[action])
            
            pages_data[u]["transitions"].append({
                "action": action,
                "target_page": v,
                "icon_bbox": icon_bbox  # Use bbox instead of coordinates
            })
        
        # Modify JSON data saving, add reference to configuration information
        json_path = os.path.join(output_dir, "ui_structure.json")
        metadata = {
            "total_pages": len(pages_data),
            "tree_depth": self.topo_generator.max_depth,
            "nodes_per_level": self.topo_generator.nodes_per_level,
            "config_file": "config.json",
            "action_space": ["click", "complete"],
        }
        if extra_metadata:
            metadata.update(extra_metadata)

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump({
                "pages": pages_data,
                "metadata": metadata
            }, f, indent=2, ensure_ascii=False)

        # New: Generate JSON file with hierarchical structure
        self.generate_layered_structure(pages_data, output_dir, extra_metadata=extra_metadata)

        return json_path

    def generate_layered_structure(self, pages_data, output_dir, extra_metadata: Optional[dict] = None):
        """Generate JSON file with hierarchical structure, child node information placed under parent's subnodes"""
        # Build parent-child relationship mapping
        parent_child_map = {}
        
        # Initialize all nodes' child node list
        for node_id in pages_data:
            parent_child_map[node_id] = []
        
        # Fill child node information
        for node_id, node_info in pages_data.items():
            for transition in node_info["transitions"]:
                # Only consider normal transitions, ignore 'back' and 'home' system operations
                action = transition["action"]
                target = transition["target_page"]
                if action not in ['back', 'home']:
                    parent_child_map[node_id].append(target)
        
        # Create tree structure
        def create_node_tree(node_id):
            node_data = pages_data[node_id].copy()
            # Only retain normal transitions, ignore 'back' and 'home' system operations
            normal_transitions = [t for t in node_data["transitions"] 
                                if t["action"] not in ['back', 'home']]
            node_data["transitions"] = normal_transitions
            
            # Add child nodes
            subnodes = []
            for child_id in parent_child_map[node_id]:
                # Ensure no circular references
                if child_id != node_id:
                    subnodes.append(create_node_tree(child_id))
            
            node_data["subnodes"] = subnodes
            return node_data
        
        # Build tree from root node
        root = "page_0"
        tree_structure = create_node_tree(root)
        
        # Save to file
        layer_json_path = os.path.join(output_dir, "ui_structure_layer.json")
        metadata = {
            "total_pages": len(pages_data),
            "tree_depth": self.topo_generator.max_depth,
            "nodes_per_level": self.topo_generator.nodes_per_level,
            "config_file": "config.json",
            "action_space": ["click", "complete"],
        }
        if extra_metadata:
            metadata.update(extra_metadata)

        with open(layer_json_path, 'w', encoding='utf-8') as f:
            json.dump({
                "root": tree_structure,
                "metadata": metadata
            }, f, indent=2, ensure_ascii=False)
        
        print(f"Hierarchical structure JSON file saved to: {layer_json_path}")

# --------------------------
# Subsystem Component Implementations
# --------------------------
class UIManager:
    """UI resource manager"""
    def __init__(self, images: List[Image.Image], funcs: List[str]):
        # Create element pool, filter out back and home
        self.all_elements = [UIElement(img, f) for img, f in zip(images, funcs) 
                           if f not in ['back', 'home']]
        
        # System icon size for 448x448 canvas (matches UIElement.ICON_SIZE)
        sys_icon_size = UIElement.ICON_SIZE
        
        # Modify system icon colors and styles
        back_img = Image.new('RGB', (sys_icon_size, sys_icon_size), (255, 200, 200))  # Light red
        home_img = Image.new('RGB', (sys_icon_size, sys_icon_size), (200, 255, 200))  # Light green
        
        # Add text with scaled font for smaller icons
        font_size = max(10, sys_icon_size // 4)  # Scale font to icon size
        for img, text in [(back_img, 'back'), (home_img, 'home')]:
            draw = ImageDraw.Draw(img)
            try:
                font = ImageFont.truetype("font/helvetica.ttf", font_size)
            except:
                font = None
            # Calculate text size to center display
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            x = (sys_icon_size - text_width) // 2
            y = (sys_icon_size - text_height) // 2
            draw.text((x, y), text, fill=(0, 0, 0), font=font)  # Black text
        
        self.sys_elements = {
            'back': UIElement(back_img, 'back'),
            'home': UIElement(home_img, 'home')
        }
        
        # Maintain icon list
        self.total_icons = self.all_elements.copy()
        self.used_icons = []
        self.available_icons = self.total_icons.copy()
        
        # Add debug information
        print(f"Total icon count: {len(self.total_icons)}")
        print(f"System icons: {[e.func_desc for e in self.sys_elements.values()]}")
        print(f"Available icons: {[e.func_desc for e in self.available_icons]}")
    
    def allocate_icons(self, count: int) -> List[UIElement]:
        """Allocate normal icons, allocate in sequence rather than randomly"""
        if count > len(self.available_icons):
            raise ValueError(
                f"Insufficient number of icons! Need: {count}, Available: {len(self.available_icons)}\n"
                f"Used icons: {[e.func_desc for e in self.used_icons]}\n"
                f"Available icons: {[e.func_desc for e in self.available_icons]}"
            )
        
        # Modify: Use first count icons, rather than randomly selecting
        selected = self.available_icons[:count]
        
        # Update icon status
        for elem in selected:
            elem.mark_used()
            self.available_icons.remove(elem)
            self.used_icons.append(elem)
            
        # Print debug information
        print(f"Allocate {count} icons: {[e.func_desc for e in selected]}")
        print(f"Remaining available icons: {len(self.available_icons)}")
        print(f"Remaining icon list: {[e.func_desc for e in self.available_icons]}")
        
        return selected
    
    def get_system_icons(self) -> List[UIElement]:
        """Get system icons"""
        return list(self.sys_elements.values())
    
    def get_icon_status(self) -> Dict[str, int]:
        """Get icon usage status"""
        return {
            "total": len(self.total_icons),
            "used": len(self.used_icons),
            "available": len(self.available_icons)
        }

class TopologyGenerator:
    """Topological generator - pure tree structure"""
    def __init__(self, max_depth: int, nodes_per_level: List[int], is_random_node: bool = False):
        """
        Initialize topological generator
        :param max_depth: Maximum tree depth
        :param nodes_per_level: List of child nodes per level, length should be max_depth-1
        :param is_random_node: Whether to randomly generate number of child nodes (not exceeding specified number)
        """
        if len(nodes_per_level) != max_depth - 1:
            raise ValueError(f"nodes_per_level length({len(nodes_per_level)}) must equal tree_depth-1({max_depth-1})")
        
        self.max_depth = max_depth
        self.nodes_per_level = nodes_per_level
        self.is_random_node = is_random_node
        self.all_pages = {}

    def generate(self, ui_mgr: UIManager) -> Tuple[Dict[str, List[str]], Dict[str, UIPage]]:
        """Generate pure tree structure page hierarchy, ensure generate exactly tree_depth layer tree, and last layer only system icons"""
        hierarchy = {}
        pages = {}
        page_counter = 0
        
        # Create root page (depth 0)
        root = self._create_page(ui_mgr, None, page_counter, depth=0)
        hierarchy[root.page_id] = []
        pages[root.page_id] = root
        self.all_pages[root.page_id] = root
        page_counter += 1
        
        # Use BFS to generate tree layer by layer
        nodes_by_level = {0: [root.page_id]}
        
        # Process each layer until tree_depth-1 layer (one layer before the last layer)
        for depth in range(self.max_depth - 1):
            # If current layer does not exist, end
            if depth not in nodes_by_level:
                break
                
            # Record next layer nodes
            nodes_by_level[depth + 1] = []
            
            # Get current layer node count
            current_level_nodes = nodes_by_level[depth]
            
            # Determine child node count
            children_per_node = self.nodes_per_level[depth]
            
            # Create child nodes for each node in current layer
            for parent_id in current_level_nodes:
                # If random node count is enabled, randomly generate child node count for each parent node
                if self.is_random_node:
                    actual_children = random.randint(1, children_per_node)
                else:
                    actual_children = children_per_node
                
                # Check if there are enough icons - only non-leaf nodes need icons
                if depth + 1 < self.max_depth - 1 and len(ui_mgr.available_icons) < actual_children:
                    print(f"Warning: Insufficient number of icons, cannot create {actual_children} child nodes for depth {depth}")
                    print(f"Need {actual_children} icons, but only {len(ui_mgr.available_icons)} are available")
                    break
                
                # Create all child nodes for current parent node
                for _ in range(actual_children):
                    # Here change judgment, ensure last layer is leaf node
                    is_leaf = (depth + 1 == self.max_depth - 1)
                    
                    # Create child node page
                    child = self._create_page(ui_mgr, parent_id, page_counter, depth=depth+1, is_leaf=is_leaf)
                    child_id = child.page_id
                    
                    # Update data structure
                    hierarchy[parent_id].append(child_id)
                    hierarchy[child_id] = []
                    pages[child_id] = child
                    self.all_pages[child_id] = child
                    nodes_by_level[depth + 1].append(child_id)
                    
                    page_counter += 1
        
        # Print debug information
        print("\n=== Tree structure generation ===")
        for level, nodes in nodes_by_level.items():
            print(f"Depth {level}: {len(nodes)} nodes")
            
        return hierarchy, pages

    def _create_page(self, ui_mgr: UIManager, parent: Optional[str], pid: int, depth: int, is_leaf: bool = False) -> UIPage:
        """Create new page, ensure leaf node only system icons"""
        # Determine actual needed icon count
        if is_leaf:
            # Leaf node does not need normal icons
            needed_icons = 0
            normal_icons = []
        else:
            # Non-leaf nodes need icons equal to number of child nodes
            if depth < self.max_depth - 2:  # Ancestor nodes of non-leaf nodes
                max_icons = self.nodes_per_level[depth]
                if self.is_random_node:
                    needed_icons = random.randint(1, max_icons)
                else:
                    needed_icons = max_icons
                    
                # Allocate icons
                normal_icons = ui_mgr.allocate_icons(needed_icons) if needed_icons > 0 else []
            else:  # Second last layer, need to allocate icons to last layer
                max_icons = self.nodes_per_level[depth]
                if self.is_random_node:
                    needed_icons = random.randint(1, max_icons)
                else:
                    needed_icons = max_icons
                    
                # Allocate icons
                normal_icons = ui_mgr.allocate_icons(needed_icons) if needed_icons > 0 else []
        
        # Print debug information
        print(f"\nCreate page page_{pid}:")
        print(f"Depth: {depth}, Parent node: {parent}, Is leaf node: {is_leaf}")
        print(f"Needed icon count: {0 if is_leaf else needed_icons}")
        
        # Add system icons - Add system icons based on depth
        sys_icons = []
        
        # Root node (page_0) does not add any system icons
        if depth > 0:
            # All non-root nodes add back button
            sys_icons.append(ui_mgr.sys_elements['back'])
            # Only nodes with depth>1 (2nd level and above child nodes) add home button
            if depth > 1:
                sys_icons.append(ui_mgr.sys_elements['home'])
        
        # Print allocation result
        print(f"Page elements: System icons({len(sys_icons)}), Normal icons({len(normal_icons)}")
        
        # Generate layout
        layout = LayoutGenerator.generate([*normal_icons, *sys_icons])
        
        return UIPage(
            page_id=f"page_{pid}",
            elements=[*normal_icons, *sys_icons],
            layout=layout,
            parent=parent
        )

class LayoutGenerator:
    """Layout generation engine"""
    # Modified to match author's 448x448 square format
    CANVAS_SIZE = (448, 448)
    ICON_HEIGHT = 50  # Scaled down for 448x448 canvas
    ICON_WIDTH = 50   # Scaled down for 448x448 canvas
    MARGIN = 20  # Scaled margin for smaller canvas
    TOP_MARGIN = 50  # Scaled top margin for smaller canvas
    
    @classmethod
    def _generate_predefined_positions(cls) -> List[Tuple[int, int]]:
        """Generate predefined center point positions, provide candidate positions for icon placement, no random offset"""
        positions = []
        
        # Calculate actual usable area
        usable_width = cls.CANVAS_SIZE[0] - 2 * cls.MARGIN
        usable_height = cls.CANVAS_SIZE[1] - cls.TOP_MARGIN - cls.MARGIN
        
        # Top system icon and title space
        reserved_top = cls.TOP_MARGIN + cls.ICON_HEIGHT + 50  # Reserve enough space for top
        
        # Minimum spacing between icons
        min_spacing_x = cls.ICON_WIDTH + 30  # Horizontal spacing
        min_spacing_y = cls.ICON_HEIGHT + 30  # Vertical spacing
        
        # Calculate grid point count
        num_cols = usable_width // min_spacing_x
        num_rows = (usable_height - reserved_top) // min_spacing_y
        
        # Limit maximum grid point count
        num_cols = min(num_cols, 5)  # Maximum 5 columns
        num_rows = min(num_rows, 8)  # Maximum 8 rows
        
        # Calculate grid starting position (to center grid)
        start_x = cls.MARGIN + (usable_width - (num_cols - 1) * min_spacing_x) // 2
        start_y = reserved_top + (usable_height - reserved_top - (num_rows - 1) * min_spacing_y) // 2
        
        # Generate grid points - No random offset
        for row in range(num_rows):
            for col in range(num_cols):
                x = start_x + col * min_spacing_x
                y = start_y + row * min_spacing_y
                positions.append((int(x), int(y)))
        
        return positions
    
    @classmethod
    def generate(cls, elements: List[UIElement]) -> Dict[str, Tuple[int, int, int, int]]:
        """Generate non-overlapping layout, return bbox format (x1, y1, x2, y2)"""
        positions = {}
        
        # Pre-define default value of y1 to prevent undefined error
        y1 = cls.MARGIN
        
        # Find system icons
        back_element = next((e for e in elements if e.func_desc == 'back'), None)
        home_element = next((e for e in elements if e.func_desc == 'home'), None)
        
        # System icon fixed positions - Add positions if corresponding system icons exist
        if back_element:
            x1, y1 = cls.MARGIN, cls.MARGIN
            positions['back'] = (x1, y1, x1 + cls.ICON_WIDTH, y1 + cls.ICON_HEIGHT)
        
        if home_element:
            x1 = cls.CANVAS_SIZE[0] - cls.ICON_WIDTH - cls.MARGIN
            positions['home'] = (x1, y1, x1 + cls.ICON_WIDTH, y1 + cls.ICON_HEIGHT)
        
        # Page title position (Reserved for RenderEngine use)
        # Scale title width proportionally to canvas (about 50% of canvas width)
        title_width = cls.CANVAS_SIZE[0] // 2
        title_x = (cls.CANVAS_SIZE[0] - title_width) // 2
        positions['page_title'] = (title_x, y1, title_x + title_width, y1 + cls.ICON_HEIGHT)
        
        # Get non-system icons
        normal_elements = [e for e in elements if e.func_desc not in ['back', 'home']]
        
        # If no normal icons, return directly
        if not normal_elements:
            return positions
        
        # Generate predefined position list
        predefined_positions = cls._generate_predefined_positions()
        
        # Ensure enough positions
        if len(predefined_positions) < len(normal_elements):
            print(f"Warning: Insufficient predefined positions! Need {len(normal_elements)}, but only {len(predefined_positions)}")
            # Repeat some positions when needed
            while len(predefined_positions) < len(normal_elements):
                predefined_positions.append(random.choice(predefined_positions))
        
        # Randomly select positions without repetition
        selected_positions = random.sample(predefined_positions, len(normal_elements))
        
        # Allocate positions to each normal icon
        for elem, (x, y) in zip(normal_elements, selected_positions):
            positions[elem.func_desc] = (x, y, x + cls.ICON_WIDTH, y + cls.ICON_HEIGHT)
        
        return positions

class TopologyBuilder:
    """Topological structure builder"""
    @staticmethod
    def build(hierarchy: Dict[str, List[str]], pages: Dict[str, UIPage]) -> nx.DiGraph:
        """Build transfer relationship graph"""
        graph = nx.DiGraph()
        
        # Add all nodes to graph
        for page_id, page in pages.items():
            graph.add_node(page_id, page=page)
            
        # Add node and normal transfer edges
        for parent, children in hierarchy.items():
            parent_page = pages[parent]
            # Get non-system icons on current page
            available_icons = [e.func_desc for e in parent_page.elements 
                              if e.func_desc not in ['back', 'home']]
            
            print(f"\nProcessing page {parent}:")
            print(f"Child node count: {len(children)}")
            print(f"Available icons: {available_icons}")
            
            # Check if there are enough icons, if not, only use available icons to connect part of child nodes
            max_connections = min(len(available_icons), len(children))
            if max_connections < len(children):
                print(f"Warning: Insufficient number of icons on page {parent}, can only connect {max_connections}/{len(children)} child nodes")
                # Cut child node list, retain part that can be connected
                children = children[:max_connections]
            
            # Allocate a icon to each child node
            for child, action in zip(children, available_icons):
                print(f"Connect: {parent} --[{action}]--> {child}")
                graph.add_edge(parent, child, action=action)
        
        # Ensure system icons have correct transfer relationships
        for node in graph.nodes:
            page = pages[node]
            
            # Check if page has system button and update its transfer relationships
            has_back = next((True for e in page.elements if e.func_desc == 'back'), False)
            has_home = next((True for e in page.elements if e.func_desc == 'home'), False)
            
            # Find and remove all existing edges related to system buttons
            edges_to_remove = []
            for succ in list(graph.successors(node)):
                action = graph[node][succ]['action']
                if action in ['back', 'home']:
                    edges_to_remove.append((node, succ))
            
            for u, v in edges_to_remove:
                print(f"Remove incorrect system transfer: {u} --[{graph[u][v]['action']}]--> {v}")
                graph.remove_edge(u, v)
            
            # Add correct system transfer edges
            if has_back and page.parent is not None:
                graph.add_edge(node, page.parent, action='back')
                print(f"Add back edge: {node} --> {page.parent}")
                
            if has_home:
                graph.add_edge(node, "page_0", action='home')
                print(f"Add home edge: {node} --> page_0")
        
        # Check repeated transfer and repair
        for node_id in graph.nodes:
            # Get all transfer actions from that node
            outgoing_edges = list(graph.out_edges(node_id, data=True))
            actions = [d['action'] for _, _, d in outgoing_edges]
            
            # Check repeated transfer
            if len(actions) != len(set(actions)):
                print(f"Warning: Page {node_id} has repeated transfer action")
                # Delete repeated edges, retain first appearing
                seen_actions = set()
                edges_to_remove = []
                
                for u, v, data in outgoing_edges:
                    action = data['action']
                    if action in seen_actions:
                        edges_to_remove.append((u, v))
                    else:
                        seen_actions.add(action)
                
                for u, v in edges_to_remove:
                    print(f"Delete repeated transfer: {u} --[{graph[u][v]['action']}]--> {v}")
                    graph.remove_edge(u, v)
                    
        # Ensure leaf nodes are also added to graph (even if no outgoing edges)
        for page_id in pages:
            if page_id not in graph:
                graph.add_node(page_id, page=pages[page_id])
                
        return graph

# --------------------------
# Rendering Engine
# --------------------------
class RenderEngine:
    def render(self, page: UIPage) -> Image.Image:
        """Render page image"""
        # Use LayoutGenerator canvas size for consistency
        img = Image.new('RGB', LayoutGenerator.CANVAS_SIZE, (255, 255, 255))
        draw = ImageDraw.Draw(img)
        
        # Draw page title
        title_bbox = page.layout['page_title']
        title_text = page.page_id
        # Use smaller font for 448x448 canvas
        try:
            from PIL import ImageFont
            font = ImageFont.truetype("font/helvetica.ttf", 24, encoding="unic")  # Scaled font for smaller canvas
        except:
            font = None
        
        # Get text size
        text_bbox = draw.textbbox((0, 0), title_text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        # Calculate text center position
        x = title_bbox[0] + (title_bbox[2] - title_bbox[0] - text_width) // 2
        y = title_bbox[1] + (title_bbox[3] - title_bbox[1] - text_height) // 2
        
        # Draw text
        draw.text((x, y), title_text, fill=(0, 0, 0), font=font)
        
        # Draw all elements
        for elem in page.elements:
            bbox = page.layout[elem.func_desc]
            img.paste(elem.raw_image, (bbox[0], bbox[1]))
        
        return img

# --------------------------
# Usage Example
# --------------------------
def create_dummy_icon(color: tuple, text: str) -> Image.Image:
    """Create test icon"""
    img = Image.new('RGB', (64, 64), color)
    draw = ImageDraw.Draw(img)
    draw.text((10, 10), text, fill=(255, 255, 255))
    return img

def load_icons_from_directory(dirname: str, required_count: int, output_dir: str = None) -> List[Tuple[Image.Image, str]]:
    """Load icon images from specified directory
    :param dirname: Icon directory path (local path)
    :param required_count: Number of required icons
    :param output_dir: Output directory for saving used_icons.json
    :return: List of [(image object, function description)]
    """
    # Use os.path to get all PNG file paths
    pattern = os.path.join(dirname, "*/PNG/*.png")
    all_icon_paths = sorted(glob.glob(pattern))

    if len(all_icon_paths) < required_count:
        raise ValueError(f"Insufficient number of icons! Need {required_count}, but directory only has {len(all_icon_paths)}")
    
    # Randomly sample required number of icons
    selected_paths = random.sample(all_icon_paths, required_count)
    
    icons = []
    used_icons_info = []
    
    for filepath in selected_paths:
        # Parse path to get category name and filename
        parts = filepath.split(os.sep)  # Use os.sep as separator for cross-platform compatibility
        category = parts[-3].replace(" ", "_")  # Category name is third from last
        filename = os.path.splitext(parts[-1])[0]  # Filename without extension
        func_name = f"{category}_{filename}"
        
        # Read image data using regular open
        with open(filepath, 'rb') as f:
            img_data = f.read()
            img = Image.open(BytesIO(img_data)).convert('RGBA')
        
        # Scale proportionally using nearest neighbor interpolation
        aspect_ratio = img.width / img.height
        new_height = 200  # Modified to 200, was 50
        new_width = int(aspect_ratio * new_height)
        img = img.resize((new_width, new_height), Image.Resampling.NEAREST)
        
        icons.append((img, func_name))
        used_icons_info.append({
            "original_path": filepath,
            "func_name": func_name
        })
    
    # If output directory specified, save used_icons.json to that directory
    if output_dir:
        json_path = os.path.join(output_dir, "used_icons.json")
        os.makedirs(output_dir, exist_ok=True)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(used_icons_info, f, indent=2, ensure_ascii=False)
    
    return icons


def _sanitize_func_name(text: str, fallback: str) -> str:
    cleaned = re.sub(r'[^0-9A-Za-z]+', '_', (text or '').strip()).strip('_')
    return cleaned[:40] if cleaned else fallback


def _tokenize(text: str) -> set:
    return {token for token in re.findall(r'[a-z0-9]+', (text or '').lower()) if len(token) > 1}


def _bbox_iou(box1: List[float], box2: List[float]) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = area1 + area2 - inter + 1e-8
    return inter / union


def _build_icon_path_index(icon_root: str) -> Dict[str, str]:
    index = {}
    for dirpath, _, filenames in os.walk(icon_root):
        for filename in filenames:
            if not filename.lower().endswith(".png"):
                continue
            index.setdefault(filename, os.path.join(dirpath, filename))
    return index


def _load_trajectory_annotation(trajectory_id: str, annotations_dir: str) -> dict:
    annot_path = os.path.join(annotations_dir, f"{trajectory_id}.json")
    if not os.path.exists(annot_path):
        raise FileNotFoundError(f"Trajectory annotation not found: {annot_path}")
    with open(annot_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def build_trajectory_metadata(trajectory_id: str, annotations_dir: str) -> dict:
    trajectory = _load_trajectory_annotation(trajectory_id, annotations_dir)
    task_info = trajectory.get("task_info", {})
    return {
        "icon_source": "trajectory",
        "trajectory_id": trajectory_id,
        "episode_id": trajectory.get("episode_id", trajectory_id),
        "trajectory_task": task_info.get("task", ""),
        "trajectory_instruction": task_info.get("instruction", ""),
        "trajectory_meta_task": task_info.get("meta_task", ""),
        "trajectory_apps": task_info.get("app", []),
        "trajectory_category": task_info.get("category", ""),
        "trajectory_step_count": len(trajectory.get("steps", [])),
        "action_space": ["click", "complete"],
    }


def _extract_live_trajectory_icon_metadata(trajectory_id: str,
                                           annotations_dir: str,
                                           screenshots_dir: str,
                                           output_dir: Optional[str] = None,
                                           weights_dir: str = "/ext_hdd2/nhkoh/OmniParser/weights",
                                           box_threshold: float = 0.05,
                                           iou_threshold: float = 0.1) -> List[dict]:
    trajectory = _load_trajectory_annotation(trajectory_id, annotations_dir)
    cache_path = os.path.join(output_dir, "trajectory_icon_metadata.json") if output_dir else None
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    from sim2real import (
        crop_and_save_icon,
        detect_icons_yolo,
        detect_text_ocr,
        load_ocr,
        load_yolo_model,
        merge_detections,
    )

    yolo_model = load_yolo_model(weights_dir)
    ocr_reader = load_ocr()

    extracted_root = os.path.join(output_dir, "trajectory_icon_pool") if output_dir else None
    if extracted_root:
        os.makedirs(extracted_root, exist_ok=True)

    metadata = []
    for step_idx, step in enumerate(trajectory.get("steps", [])):
        screenshot_name = step.get("screenshot")
        if not screenshot_name:
            continue

        screenshot_path = os.path.join(screenshots_dir, screenshot_name)
        if not os.path.exists(screenshot_path):
            continue

        with Image.open(screenshot_path) as img_handle:
            image_pil = img_handle.convert("RGB")
            img_w, img_h = image_pil.size
            yolo_boxes, yolo_confs = detect_icons_yolo(
                yolo_model, image_pil, box_threshold=box_threshold, iou_threshold=iou_threshold
            )
            ocr_texts, ocr_bboxes = detect_text_ocr(ocr_reader, image_pil)
            elements = merge_detections(
                yolo_boxes, yolo_confs, ocr_texts, ocr_bboxes, img_w, img_h
            )

            step_dir = None
            if extracted_root:
                step_dir = os.path.join(
                    extracted_root,
                    f"step_{step.get('step', step_idx):02d}_{os.path.splitext(screenshot_name)[0]}",
                )
                os.makedirs(step_dir, exist_ok=True)

            icon_counter = 0
            for elem in elements:
                if elem.get("type") != "icon":
                    continue

                bbox = list(elem.get("bbox") or [])
                if len(bbox) != 4:
                    continue

                content = (elem.get("content") or "").strip()
                icon_path = None
                if step_dir:
                    file_stub = _sanitize_func_name(content, f"icon_{icon_counter}")
                    icon_path = os.path.join(step_dir, f"{icon_counter:03d}_{file_stub}.png")
                    if not crop_and_save_icon(image_pil, bbox, icon_path):
                        icon_path = None

                metadata.append({
                    "type": "icon",
                    "bbox": bbox,
                    "content": content,
                    "interactivity": bool(elem.get("interactivity", True)),
                    "confidence": float(elem.get("confidence", 0.0)),
                    "source": elem.get("source", "trajectory_live"),
                    "source_screenshot": screenshot_name,
                    "icon_path": icon_path,
                    "step_index": step.get("step", step_idx),
                })
                icon_counter += 1

    if cache_path:
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

    return metadata


def _load_icon_from_metadata_entry(entry: Dict[str, Any],
                                   screenshots_dir: str) -> Optional[Image.Image]:
    icon_path = entry.get("icon_path")
    if icon_path and os.path.exists(icon_path):
        with open(icon_path, 'rb') as f:
            return Image.open(BytesIO(f.read())).convert('RGBA')

    screenshot_name = entry.get("source_screenshot")
    bbox = entry.get("bbox") or []
    if not screenshot_name or len(bbox) != 4:
        return None

    screenshot_path = os.path.join(screenshots_dir, screenshot_name)
    if not os.path.exists(screenshot_path):
        return None

    with Image.open(screenshot_path) as screenshot_img:
        screenshot = screenshot_img.convert("RGBA")
        width, height = screenshot.size
        x1 = max(0, int(round(bbox[0] * width)))
        y1 = max(0, int(round(bbox[1] * height)))
        x2 = min(width, int(round(bbox[2] * width)))
        y2 = min(height, int(round(bbox[3] * height)))
        if x2 - x1 < 5 or y2 - y1 < 5:
            return None
        return screenshot.crop((x1, y1, x2, y2))


def load_icons_from_trajectory(trajectory_id: str,
                               required_count: int,
                               metadata_path: Optional[str],
                               annotations_dir: str,
                               screenshots_dir: str,
                               output_dir: str = None,
                               extract_mode: str = "live",
                               weights_dir: str = "/ext_hdd2/nhkoh/OmniParser/weights"
                               ) -> List[Tuple[Image.Image, str]]:
    """Load a trajectory-scoped icon pool for full GE-Lab tree generation."""
    trajectory = _load_trajectory_annotation(trajectory_id, annotations_dir)
    if extract_mode == "live":
        metadata = _extract_live_trajectory_icon_metadata(
            trajectory_id=trajectory_id,
            annotations_dir=annotations_dir,
            screenshots_dir=screenshots_dir,
            output_dir=output_dir,
            weights_dir=weights_dir,
        )
        icon_path_index = {}
    else:
        if not metadata_path or not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Icon metadata not found: {metadata_path}")
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        icon_root = os.path.dirname(metadata_path)
        icon_path_index = _build_icon_path_index(icon_root)

    steps = trajectory.get("steps", [])
    task_info = trajectory.get("task_info", {})
    screenshot_names = {step.get("screenshot") for step in steps if step.get("screenshot")}
    if not screenshot_names:
        raise ValueError(f"Trajectory {trajectory_id} does not contain screenshots")

    screenshot_sizes = {}
    for screenshot_name in screenshot_names:
        screenshot_path = os.path.join(screenshots_dir, screenshot_name)
        if os.path.exists(screenshot_path):
            with Image.open(screenshot_path) as screenshot_img:
                screenshot_sizes[screenshot_name] = screenshot_img.size

    task_tokens = set()
    task_tokens.update(_tokenize(task_info.get("task", "")))
    task_tokens.update(_tokenize(task_info.get("instruction", "")))
    task_tokens.update(_tokenize(task_info.get("meta_task", "")))
    for app_name in task_info.get("app", []):
        task_tokens.update(_tokenize(app_name))

    step_boxes = {}
    step_tokens = {}
    for step in steps:
        screenshot_name = step.get("screenshot")
        if not screenshot_name:
            continue
        step_tokens[screenshot_name] = set()
        step_tokens[screenshot_name].update(_tokenize(step.get("low_level_instruction", "")))
        step_tokens[screenshot_name].update(_tokenize(step.get("description", "")))
        step_tokens[screenshot_name].update(_tokenize(step.get("intention", "")))
        step_tokens[screenshot_name].update(_tokenize(str(step.get("info", ""))))

        sam2_bbox = step.get("sam2_bbox") or []
        img_size = screenshot_sizes.get(screenshot_name)
        if len(sam2_bbox) == 4 and img_size is not None:
            width, height = img_size
            step_boxes.setdefault(screenshot_name, []).append([
                sam2_bbox[0] / width,
                sam2_bbox[1] / height,
                sam2_bbox[2] / width,
                sam2_bbox[3] / height,
            ])

    ranked_entries = []
    for index, entry in enumerate(metadata):
        if entry.get("type") != "icon":
            continue
        screenshot_name = entry.get("source_screenshot")
        if screenshot_name not in screenshot_names:
            continue
        icon_path = entry.get("icon_path")
        if icon_path and not os.path.exists(icon_path):
            icon_path = icon_path_index.get(os.path.basename(icon_path))

        bbox = entry.get("bbox") or []
        if len(bbox) != 4:
            continue

        width = max(0.0, bbox[2] - bbox[0])
        height = max(0.0, bbox[3] - bbox[1])
        area = width * height
        if area < 0.0002 or area > 0.12:
            continue
        aspect_ratio = width / max(height, 1e-6)
        if aspect_ratio > 4.0 or aspect_ratio < 0.25:
            continue

        score = 0.1
        content = entry.get("content") or ""
        content_tokens = _tokenize(content)
        if content_tokens & task_tokens:
            score += 0.5
        if content_tokens & step_tokens.get(screenshot_name, set()):
            score += 0.35

        max_overlap = 0.0
        for action_box in step_boxes.get(screenshot_name, []):
            max_overlap = max(max_overlap, _bbox_iou(bbox, action_box))
        score += max_overlap * 2.0

        if not content_tokens and max_overlap < 0.1:
            score -= 0.2

        ranked_entries.append({
            "score": score,
            "entry": {**entry, "icon_path": icon_path},
            "max_overlap": max_overlap,
            "content": content,
            "index": index,
        })

    ranked_entries.sort(
        key=lambda item: (
            item["score"],
            item["max_overlap"],
            bool(item["content"]),
            item["content"].lower(),
            -item["index"],
        ),
        reverse=True,
    )

    selected_entries = ranked_entries[:required_count]
    initial_selected_count = len(selected_entries)
    if not selected_entries:
        raise ValueError(
            f"Trajectory {trajectory_id} did not yield any usable icons after trajectory filtering."
        )
    if initial_selected_count < required_count:
        base_pool = list(selected_entries)
        reuse_index = 0
        while len(selected_entries) < required_count:
            selected_entries.append(base_pool[reuse_index % len(base_pool)])
            reuse_index += 1

    icons = []
    used_icons_info = []
    name_counts = {}
    for selection_idx, item in enumerate(selected_entries):
        entry = item["entry"]
        img = _load_icon_from_metadata_entry(entry, screenshots_dir)
        if img is None:
            continue

        base_name = _sanitize_func_name(
            entry.get("content") or os.path.splitext(os.path.basename(entry.get("icon_path") or ""))[0],
            f"traj_icon_{len(icons)}"
        )
        suffix = name_counts.get(base_name, 0)
        name_counts[base_name] = suffix + 1
        func_name = base_name if suffix == 0 else f"{base_name}_{suffix}"

        icons.append((img, func_name))
        used_icons_info.append({
            "icon_path": entry.get("icon_path"),
            "func_name": func_name,
            "content": entry.get("content"),
            "source_screenshot": entry.get("source_screenshot"),
            "score": round(item["score"], 4),
            "action_overlap": round(item["max_overlap"], 4),
            "selection_index": selection_idx,
            "reused_from_ranked_pool": selection_idx >= initial_selected_count,
        })

    if output_dir:
        json_path = os.path.join(output_dir, "used_icons.json")
        os.makedirs(output_dir, exist_ok=True)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(used_icons_info, f, indent=2, ensure_ascii=False)

    if len(icons) < required_count:
        raise ValueError(
            f"Trajectory {trajectory_id} resolved only {len(icons)} usable icon crops, need {required_count}."
        )

    return icons

def calculate_required_icons(tree_depth: int, nodes_per_level: List[int]) -> int:
    """
    Calculate minimum number of icons needed for tree structure
    :param tree_depth: Tree depth
    :param nodes_per_level: List of child nodes per level
    :return: Minimum number of required icons
    """
    total_icons = 0
    current_level_nodes = 1  # Start from root node
    
    # For each layer (except the last layer), calculate needed icon count
    for depth in range(tree_depth - 1):
        # Current layer all nodes count * Each node needed child node count
        icons_needed = current_level_nodes * nodes_per_level[depth]
        total_icons += icons_needed
        # Update next layer total node count
        current_level_nodes = icons_needed
    
    return total_icons


def _page_sort_key(page_id: str):
    try:
        return int(page_id.split("_")[1])
    except Exception:
        return page_id


def _typed_layout(layout: Dict[str, List[int]]) -> Dict[str, dict]:
    return {
        name: {"bbox": [int(v) for v in bbox], "type": "normal"}
        for name, bbox in layout.items()
    }


def _build_spine_with_merge(
    families: List[dict],
    spine_page_ids: List[str],
) -> Tuple[Dict[str, dict], Dict[str, List[str]], List[str]]:
    """
    Build spine pages and edges with identical-state merge.
    Families must have: page_id, layout, canonical_action_name, canonical_action_bbox (set for non-last).
    Returns (pages, tree_children, effective_spine_id). No I/O.
    """
    tree_children: Dict[str, List[str]] = {}
    pages: Dict[str, dict] = {}
    fp2page: Dict[str, str] = {}
    effective_spine_id: List[str] = []

    for family_idx, family in enumerate(families):
        page_id = family["page_id"]
        next_page_id = spine_page_ids[family_idx + 1] if family_idx + 1 < len(spine_page_ids) else None
        # Use pre-mutation layout for fingerprint so merge is by content, not by _choose_click_target additions
        layout = family.get("_original_layout", family["layout"])
        fp = state_fingerprint_layout(layout)

        if fp in fp2page:
            existing_id = fp2page[fp]
            effective_spine_id.append(existing_id)
            existing_page = pages[existing_id]

            # Merge additional layout elements from this family into the existing page.
            # If a name already exists with the same bbox, skip; if bbox differs, keep the existing one
            # (we treat the first-seen layout as canonical for that name).
            existing_layout = existing_page.get("layout", {})
            typed_new_layout = _typed_layout(layout)
            for name, obj in typed_new_layout.items():
                if name in SYSTEM_LAYOUT_KEYS:
                    continue
                new_bbox = obj.get("bbox")
                if name in existing_layout:
                    old_bbox = existing_layout[name].get("bbox")
                    if isinstance(old_bbox, list) and isinstance(new_bbox, list) and old_bbox == new_bbox:
                        continue
                    # Different bbox for same name: keep the original mapping and ignore the new one for now.
                    continue
                existing_layout[name] = obj

            # Track all source step indices that map to this unified state.
            src_list = existing_page.setdefault("source_steps", [])
            if not src_list:
                # Seed with the original source_step_index if present.
                base_idx = existing_page.get("source_step_index")
                if base_idx is not None:
                    src_list.append(base_idx)
            if family_idx not in src_list:
                src_list.append(family_idx)

            prev_id = effective_spine_id[family_idx - 1]
            for t in pages[prev_id]["transitions"]:
                if t["target_page"] == page_id:
                    t["target_page"] = existing_id
                    break
            tree_children[prev_id] = [existing_id if c == page_id else c for c in tree_children[prev_id]]
            if next_page_id is not None:
                pages[existing_id]["transitions"].append({
                    "action": family["canonical_action_name"],
                    "target_page": next_page_id,
                    "icon_bbox": family["canonical_action_bbox"],
                    "transition_role": "spine",
                })
                tree_children.setdefault(existing_id, []).append(next_page_id)
            # Ensure the new transition's action name exists in layout (transitions reference mutated layout).
            if family.get("canonical_action_name") and family["canonical_action_name"] not in existing_layout:
                existing_layout[family["canonical_action_name"]] = {
                    "bbox": list(family.get("canonical_action_bbox") or []),
                }
            continue

        effective_spine_id.append(page_id)
        fp2page[fp] = page_id
        # Use mutated layout (family["layout"]) so transition action names exist in saved layout.
        page_record = {
            "image": f"{page_id}.png",
            "depth": family_idx,
            "layout": _typed_layout(family["layout"]),
            "transitions": [],
            "source_step_index": family_idx,
            "source_steps": [family_idx],
            "page_family_id": family.get("page_family_id", f"family_{family_idx:03d}"),
            "is_canonical": True,
            "branch_parent_page": None,
            "branch_parent_action": None,
            "merge_target_page": next_page_id,
        }
        if next_page_id is not None:
            page_record["transitions"].append({
                "action": family["canonical_action_name"],
                "target_page": next_page_id,
                "icon_bbox": family["canonical_action_bbox"],
                "transition_role": "spine",
            })
            tree_children[page_id] = [next_page_id]
        else:
            tree_children[page_id] = []
        pages[page_id] = page_record

    tree_children = {k: v for k, v in tree_children.items() if k in pages}
    return pages, tree_children, effective_spine_id


def _dedupe_name(name: str, counts: Dict[str, int]) -> str:
    suffix = counts.get(name, 0)
    counts[name] = suffix + 1
    return name if suffix == 0 else f"{name}_{suffix}"


def _normalize_transition_action(raw_label: str, step: dict, fallback: str) -> str:
    action_type = str(step.get("action", "")).upper()
    info_text = str(step.get("info", ""))
    instruction = " ".join([
        str(step.get("low_level_instruction", "")),
        str(step.get("description", "")),
        str(step.get("intention", "")),
        info_text,
    ]).lower()

    label = _sanitize_func_name(raw_label, fallback)
    lowered = label.lower()
    if "key_home" in info_text.lower() or "home screen" in instruction or lowered == "home":
        return "launcher_button"
    if info_text == "BACK" or "go back" in instruction or lowered == "back":
        return "previous_button"
    if action_type == "TEXT":
        return "input_field" if not re.search(r"[A-Za-z]", lowered) else label
    if action_type == "SCROLL" and lowered in {"scroll", "feed", "content"}:
        return "content_region"
    if not re.search(r"[A-Za-z]", label):
        return fallback
    return label[:40]


def _bbox_area(bbox: List[int]) -> int:
    return max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])


def _bbox_center(bbox: List[int]) -> Tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def _bbox_distance(box1: List[int], box2: List[int]) -> float:
    c1x, c1y = _bbox_center(box1)
    c2x, c2y = _bbox_center(box2)
    return ((c1x - c2x) ** 2 + (c1y - c2y) ** 2) ** 0.5


def _metadata_action_bbox(kind: str, canvas_size: Tuple[int, int]) -> List[int]:
    width, height = canvas_size
    if kind == "launcher_button":
        return [width // 2 - 34, height - 48, width // 2 + 34, height - 10]
    if kind == "previous_button":
        return [10, height - 48, 68, height - 10]
    if kind == "input_field":
        return [int(width * 0.08), int(height * 0.04), int(width * 0.92), int(height * 0.12)]
    return [int(width * 0.08), int(height * 0.18), int(width * 0.92), int(height * 0.86)]


def _alias_layout_action(family: dict, raw_name: str, new_name: str) -> str:
    if not raw_name or raw_name == new_name:
        return raw_name
    if raw_name in family["layout"] and new_name not in family["layout"]:
        family["layout"][new_name] = family["layout"].pop(raw_name)
        for elem in family["scaled_elements"]:
            if elem["action_name"] == raw_name:
                elem["action_name"] = new_name
                break
        return new_name
    return new_name


def _add_metadata_action(family: dict, action_name: str, bbox: List[int]) -> Tuple[str, List[int]]:
    counts = family.setdefault("action_name_counts", {})
    final_name = _dedupe_name(action_name, counts)
    family["layout"][final_name] = [int(v) for v in bbox]
    return final_name, family["layout"][final_name]


def _choose_click_target(family: dict) -> Tuple[str, List[int]]:
    step = family["step"]
    action_type = str(step.get("action", "")).upper()
    info_text = str(step.get("info", ""))
    instruction = " ".join([
        str(step.get("low_level_instruction", "")),
        str(step.get("description", "")),
        str(step.get("intention", "")),
        info_text,
    ]).lower()
    canvas_size = tuple(family["canvas_size"])
    if "scaled_elements" not in family and "layout" in family:
        layout = family["layout"]
        family["scaled_elements"] = [
            {"action_name": name, "scaled_bbox": bbox if isinstance(bbox, (list, tuple)) else bbox.get("bbox", [0, 0, 10, 10])}
            for name, bbox in layout.items()
        ]

    if "KEY_HOME" in info_text or "home screen" in instruction:
        return _add_metadata_action(family, "launcher_button", _metadata_action_bbox("launcher_button", canvas_size))
    if info_text == "BACK" or "go back" in instruction or instruction.startswith("back "):
        return _add_metadata_action(family, "previous_button", _metadata_action_bbox("previous_button", canvas_size))

    scaled_elements = family["scaled_elements"]
    sam2_bbox = step.get("sam2_bbox") or []
    if action_type == "CLICK" and len(sam2_bbox) == 4:
        from sim2real_compose import _scale_bbox_to_box as compose_scale_bbox

        scaled_bbox = compose_scale_bbox(sam2_bbox, tuple(family["orig_size"]), canvas_size)
        best_elem = None
        best_iou = 0.0
        best_distance = float("inf")
        for elem in scaled_elements:
            bbox = elem["scaled_bbox"]
            iou = _bbox_iou(
                [bbox[0], bbox[1], bbox[2], bbox[3]],
                [scaled_bbox[0], scaled_bbox[1], scaled_bbox[2], scaled_bbox[3]],
            )
            distance = _bbox_distance(bbox, scaled_bbox)
            if iou > best_iou or (iou == best_iou and distance < best_distance):
                best_elem = elem
                best_iou = iou
                best_distance = distance
        if best_elem is not None and (best_iou > 0 or best_distance <= 56):
            action_name = _normalize_transition_action(
                best_elem["action_name"], step, f"step_{family['source_step_index']:02d}_click"
            )
            action_name = _alias_layout_action(family, best_elem["action_name"], action_name)
            return action_name, family["layout"].get(action_name, best_elem["scaled_bbox"])

    if action_type == "TEXT":
        preferred = []
        for elem in scaled_elements:
            label = str(elem.get("action_name", "")).lower()
            bbox = elem["scaled_bbox"]
            is_wide = (bbox[2] - bbox[0]) >= int(canvas_size[0] * 0.45)
            if any(token in label for token in ("search", "input", "field", "text", "query")) or is_wide:
                preferred.append((is_wide, _bbox_area(bbox), elem))
        if preferred:
            preferred.sort(key=lambda item: (item[0], item[1]), reverse=True)
            elem = preferred[0][2]
            action_name = _normalize_transition_action(elem["action_name"], step, "input_field")
            action_name = _alias_layout_action(family, elem["action_name"], action_name)
            return action_name, family["layout"].get(action_name, elem["scaled_bbox"])
        return _add_metadata_action(family, "input_field", _metadata_action_bbox("input_field", canvas_size))

    if action_type == "SCROLL":
        ranked = []
        for elem in scaled_elements:
            bbox = elem["scaled_bbox"]
            cx, cy = _bbox_center(bbox)
            area = _bbox_area(bbox)
            if area < 1200:
                continue
            center_bonus = -abs(cx - canvas_size[0] / 2.0) - abs(cy - canvas_size[1] / 2.0) * 0.5
            ranked.append((area + center_bonus, elem))
        if ranked:
            ranked.sort(key=lambda item: item[0], reverse=True)
            elem = ranked[0][1]
            action_name = _normalize_transition_action(elem["action_name"], step, "content_region")
            action_name = _alias_layout_action(family, elem["action_name"], action_name)
            return action_name, family["layout"].get(action_name, elem["scaled_bbox"])
        return _add_metadata_action(family, "content_region", _metadata_action_bbox("content_region", canvas_size))

    best_score = -1.0
    best_elem = None
    instruction_tokens = _tokenize(instruction)
    for elem in scaled_elements:
        label = str(elem.get("action_name", ""))
        label_tokens = _tokenize(label)
        overlap = len(instruction_tokens & label_tokens)
        score = float(overlap) * 2.0 + _bbox_area(elem["scaled_bbox"]) / 5000.0
        if score > best_score:
            best_score = score
            best_elem = elem
    if best_elem is not None:
        action_name = _normalize_transition_action(
            best_elem["action_name"], step, f"step_{family['source_step_index']:02d}_continue"
        )
        action_name = _alias_layout_action(family, best_elem["action_name"], action_name)
        return action_name, family["layout"].get(action_name, best_elem["scaled_bbox"])
    return _add_metadata_action(
        family,
        f"step_{family['source_step_index']:02d}_continue",
        _metadata_action_bbox("content_region", canvas_size),
    )


def _select_branch_actions(family: dict, max_branches: int) -> List[str]:
    candidates = []
    excluded = {family.get("canonical_action_name")}
    for elem in family["scaled_elements"]:
        action_name = elem["action_name"]
        if action_name in excluded:
            continue
        bbox = elem["scaled_bbox"]
        area = _bbox_area(bbox)
        if area < 320:
            continue
        alpha_bonus = 1 if re.search(r"[A-Za-z]", str(elem.get("label", ""))) else 0
        type_bonus = 1 if elem.get("type") == "icon" else 0
        score = type_bonus * 100000 + alpha_bonus * 10000 + area
        candidates.append((score, action_name))
    candidates.sort(reverse=True)
    return [name for _, name in candidates[:max_branches]]


def _build_branch_asset_pool(families: List[dict], family_idx: int, window: int = 2) -> List[dict]:
    asset_pool = []
    start = max(0, family_idx - window)
    end = min(len(families), family_idx + window + 1)
    for idx in range(start, end):
        for elem in families[idx]["scaled_elements"]:
            asset_path = elem.get("asset_path")
            bbox = elem.get("bbox") or []
            if not asset_path or len(bbox) != 4:
                continue
            asset_pool.append({
                "asset_path": asset_path,
                "type": elem.get("type"),
                "label": elem.get("label"),
                "asset_size": (max(1, bbox[2] - bbox[0]), max(1, bbox[3] - bbox[1])),
                "source_step_index": families[idx]["source_step_index"],
            })
    return asset_pool


def _choose_replacement_asset(pool: List[dict], elem: dict, family_idx: int, branch_idx: int,
                              used_paths: set) -> Optional[dict]:
    target_bbox = elem["scaled_bbox"]
    target_w = max(1, target_bbox[2] - target_bbox[0])
    target_h = max(1, target_bbox[3] - target_bbox[1])
    target_ratio = target_w / float(target_h)
    ranked = []
    for item in pool:
        if item["asset_path"] in used_paths:
            continue
        if item["type"] != elem.get("type"):
            continue
        if item["label"] == elem.get("label") and item["source_step_index"] == family_idx:
            continue
        asset_w, asset_h = item["asset_size"]
        asset_ratio = asset_w / float(max(asset_h, 1))
        ratio_penalty = abs(target_ratio - asset_ratio)
        distance_penalty = abs(item["source_step_index"] - family_idx) * 0.15
        label_bonus = 0.5 if item["label"] != elem.get("label") else 0.0
        score = label_bonus - ratio_penalty - distance_penalty
        ranked.append((score, item))
    if not ranked:
        return None
    ranked.sort(key=lambda item: item[0], reverse=True)
    return ranked[branch_idx % len(ranked)][1]


def _apply_branch_variant(family: dict, focus_action: str, branch_idx: int,
                          asset_pool: List[dict]) -> Image.Image:
    image = family["canonical_image"].convert("RGBA")
    layout = family["layout"]
    element_by_action = {elem["action_name"]: elem for elem in family["scaled_elements"]}
    focus_bbox = layout[focus_action]
    ordered_actions = [focus_action]
    nearby = []
    for elem in family["scaled_elements"]:
        action_name = elem["action_name"]
        if action_name == focus_action or action_name == family.get("canonical_action_name"):
            continue
        nearby.append((_bbox_distance(focus_bbox, elem["scaled_bbox"]), action_name))
    nearby.sort(key=lambda item: item[0])
    ordered_actions.extend(name for _, name in nearby[:2])

    used_paths = set()
    for action_name in ordered_actions:
        elem = element_by_action.get(action_name)
        if elem is None:
            continue
        bbox = layout[action_name]
        replacement = _choose_replacement_asset(asset_pool, elem, family["source_step_index"], branch_idx, used_paths)
        if replacement is not None:
            used_paths.add(replacement["asset_path"])
            with Image.open(replacement["asset_path"]) as asset_handle:
                fitted = fit_image_to_canvas(
                    asset_handle.convert("RGBA"),
                    (max(1, bbox[2] - bbox[0]), max(1, bbox[3] - bbox[1])),
                )
            image.alpha_composite(fitted, (bbox[0], bbox[1]))
            continue

        patch = image.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
        factor = 1.08 if branch_idx % 2 == 0 else 0.92
        patch = ImageEnhance.Brightness(patch).enhance(factor)
        image.paste(patch, (bbox[0], bbox[1], bbox[2], bbox[3]))

    return image.convert("RGB")


def _visualize_serialized_graph(pages: Dict[str, dict], output_path: str):
    graph = nx.DiGraph()
    for page_id, page in pages.items():
        graph.add_node(page_id, depth=page.get("depth", 0))
        for transition in page.get("transitions", []):
            graph.add_edge(page_id, transition["target_page"], action=transition["action"])
    if not graph.nodes:
        return

    nodes_by_depth: Dict[int, List[str]] = {}
    for node, data in graph.nodes(data=True):
        nodes_by_depth.setdefault(int(data.get("depth", 0)), []).append(node)

    pos = {}
    for depth in sorted(nodes_by_depth):
        nodes = sorted(nodes_by_depth[depth], key=_page_sort_key)
        for idx, node in enumerate(nodes):
            pos[node] = (depth, -idx)

    plt.figure(figsize=(18, 10))
    nx.draw_networkx_nodes(graph, pos, node_color="lightblue", node_size=550)
    nx.draw_networkx_edges(graph, pos, arrows=True, arrowsize=12, edge_color="gray")
    nx.draw_networkx_labels(graph, pos, font_size=7)
    edge_labels = {(u, v): data["action"] for u, v, data in graph.edges(data=True)}
    nx.draw_networkx_edge_labels(
        graph,
        pos,
        edge_labels=edge_labels,
        font_size=6,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.8),
    )
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=200)
    plt.close()


def _build_gt_layer_tree(root_id: str, pages: Dict[str, dict], tree_children: Dict[str, List[str]]) -> dict:
    def build_node(page_id: str, visited: Optional[set] = None) -> dict:
        if visited is None:
            visited = set()
        if page_id in visited:
            return None
        visited.add(page_id)
        page = pages[page_id]
        subnodes = []
        for child_id in tree_children.get(page_id, []):
            if child_id not in visited:
                child_node = build_node(child_id, visited)
                if child_node is not None:
                    subnodes.append(child_node)
        return {
            "image": page["image"],
            "depth": page["depth"],
            "layout": page["layout"],
            "transitions": [t for t in page["transitions"] if t.get("transition_role") != "merge"],
            "source_step_index": page.get("source_step_index"),
            "page_family_id": page.get("page_family_id"),
            "is_canonical": page.get("is_canonical", False),
            "branch_parent_page": page.get("branch_parent_page"),
            "merge_target_page": page.get("merge_target_page"),
            "subnodes": subnodes,
        }

    return build_node(root_id)


def _save_trajectory_asset_manifest(output_dir: str, pages_detection_data: List[dict]) -> None:
    """Save manifest of extracted assets (used when trajectory uses detection or cache)."""
    manifest = []
    for page in pages_detection_data:
        for elem in page.get("elements", []):
            manifest.append({
                "page_id": page.get("page_id"),
                "screenshot": page.get("screenshot_name"),
                "step_index": page.get("step", {}).get("step_index"),
                "type": elem.get("type"),
                "label": elem.get("label"),
                "bbox": elem.get("bbox"),
                "asset_path": elem.get("asset_path"),
                "asset_source": elem.get("asset_source"),
            })
    with open(os.path.join(output_dir, "trajectory_assets_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def generate_trajectory_family_environment(args, output_dir: str) -> dict:
    trajectory = _load_trajectory_annotation(args.trajectory_id, args.annotations_dir)
    steps = trajectory.get("steps", [])
    if not steps:
        raise ValueError(f"Trajectory {args.trajectory_id} has no steps")

    pages_dir = os.path.join(output_dir, "pages")
    assets_dir = os.path.join(output_dir, "trajectory_assets")
    family_cache_dir = os.path.join(output_dir, "page_families")
    os.makedirs(pages_dir, exist_ok=True)
    os.makedirs(assets_dir, exist_ok=True)
    os.makedirs(family_cache_dir, exist_ok=True)

    yolo_model, ocr_reader = None, None

    families = []
    pages_detection_data = []
    for step_idx, step in enumerate(steps):
        screenshot_name = step.get("screenshot", f"{args.trajectory_id}_{step_idx}.png")
        screenshot_path = os.path.join(args.screenshots_dir, screenshot_name)
        if not os.path.exists(screenshot_path):
            raise FileNotFoundError(f"Missing screenshot: {screenshot_path}")

        family_json_path = os.path.join(family_cache_dir, f"family_{step_idx:03d}.json")
        family_png_path = os.path.join(family_cache_dir, f"family_{step_idx:03d}_canonical.png")
        if os.path.exists(family_json_path) and os.path.exists(family_png_path):
            with open(family_json_path, "r", encoding="utf-8") as f:
                family = json.load(f)
            if family.get("render_mode") == "crop_reconstructed":
                with Image.open(family_png_path) as img_handle:
                    family["canonical_image"] = img_handle.convert("RGB")
                families.append(family)
                pages_detection_data.append({
                    "page_id": family["page_id"],
                    "screenshot_name": screenshot_name,
                    "step": family["step"],
                    "elements": family["elements"],
                })
                continue

        if yolo_model is None:
            import sim2real_compose as _sim2real
            yolo_model, ocr_reader = _sim2real.load_detection_models(args.omniparser_weights, 0)
        step_context = _sim2real._build_step_context(trajectory, step_idx)
        elements, orig_size = _sim2real.detect_and_crop(screenshot_path, yolo_model, ocr_reader)
        asset_elements = _sim2real._persist_extracted_assets(elements, screenshot_name, assets_dir, step_context)
        canonical_image, layout, scaled_elements = _sim2real.render_reconstructed_native_page(
            screenshot_path,
            asset_elements,
            orig_size,
        )
        family = {
            "page_id": f"page_{step_idx}",
            "page_family_id": f"family_{step_idx:03d}",
            "source_step_index": step_idx,
            "screenshot_name": screenshot_name,
            "screenshot_path": screenshot_path,
            "orig_size": list(orig_size),
            "canvas_size": list(_sim2real.OUTPUT_CANVAS_SIZE),
            "step": step_context,
            "elements": asset_elements,
            "scaled_elements": scaled_elements,
            "layout": {name: [int(v) for v in bbox] for name, bbox in layout.items()},
            "action_name_counts": {name: 1 for name in layout},
            "render_mode": "crop_reconstructed",
        }
        family["canonical_image"] = canonical_image

        with open(family_json_path, "w", encoding="utf-8") as f:
            json.dump({k: v for k, v in family.items() if k != "canonical_image"}, f, indent=2)
        canonical_image.save(family_png_path)

        families.append(family)
        pages_detection_data.append({
            "page_id": family["page_id"],
            "screenshot_name": screenshot_name,
            "step": step_context,
            "elements": asset_elements,
        })

    _save_trajectory_asset_manifest(output_dir, pages_detection_data)

    for family in families:
        if "scaled_elements" not in family and "layout" in family:
            layout = family["layout"]
            family["scaled_elements"] = [
                {
                    "action_name": name,
                    "scaled_bbox": list(bbox) if isinstance(bbox, (list, tuple)) else list(bbox.get("bbox", [0, 0, 10, 10])),
                }
                for name, bbox in layout.items()
            ]

    # Snapshot layout before _choose_click_target mutates it (e.g. _add_metadata_action); merge uses this.
    for family in families:
        layout = family.get("layout", {})
        family["_original_layout"] = {k: list(v) if isinstance(v, (list, tuple)) else v for k, v in layout.items()}

    spine_page_ids = [family["page_id"] for family in families]
    for family_idx, family in enumerate(families):
        next_page_id = spine_page_ids[family_idx + 1] if family_idx + 1 < len(spine_page_ids) else None
        if next_page_id is not None:
            action_name, action_bbox = _choose_click_target(family)
            family["canonical_action_name"] = action_name
            family["canonical_action_bbox"] = [int(v) for v in action_bbox]
        else:
            family["canonical_action_name"] = None
            family["canonical_action_bbox"] = None

    pages, tree_children, effective_spine_id = _build_spine_with_merge(families, spine_page_ids)
    for family in families:
        page_id = family["page_id"]
        if page_id in pages:
            page_path = os.path.join(pages_dir, f"{page_id}.png")
            family["canonical_image"].save(page_path)
    branch_pages = []
    # Use max existing page index + 1 to avoid colliding with spine page IDs after merge.
    page_counter = (max(int(p.split("_")[1]) for p in pages) + 1) if pages else 0
    max_branches = max(0, int(getattr(args, "branches_per_step", 2)))
    for family_idx, family in enumerate(families[:-1]):
        canonical_page_id = family["page_id"]
        if canonical_page_id not in pages:
            continue
        merge_target_page = effective_spine_id[family_idx + 1]
        branch_actions = _select_branch_actions(family, max_branches)
        if not branch_actions:
            continue
        asset_pool = _build_branch_asset_pool(families, family_idx)
        for branch_idx, branch_action in enumerate(branch_actions):
            branch_page_id = f"page_{page_counter}"
            page_counter += 1
            branch_img = _apply_branch_variant(family, branch_action, branch_idx, asset_pool)
            branch_path = os.path.join(pages_dir, f"{branch_page_id}.png")
            branch_img.save(branch_path)

            pages[canonical_page_id]["transitions"].append({
                "action": branch_action,
                "target_page": branch_page_id,
                "icon_bbox": family["layout"][branch_action],
                "transition_role": "branch",
            })
            tree_children.setdefault(canonical_page_id, []).append(branch_page_id)
            tree_children.setdefault(branch_page_id, [])

            branch_pages.append(branch_page_id)
            pages[branch_page_id] = {
                "image": f"{branch_page_id}.png",
                "depth": family_idx + 1,
                "layout": _typed_layout({name: bbox[:] for name, bbox in family["layout"].items()}),
                "transitions": [{
                    "action": family["canonical_action_name"],
                    "target_page": merge_target_page,
                    "icon_bbox": family["canonical_action_bbox"],
                    "transition_role": "merge",
                }],
                "source_step_index": family_idx,
                "page_family_id": family["page_family_id"],
                "is_canonical": False,
                "branch_parent_page": canonical_page_id,
                "branch_parent_action": branch_action,
                "merge_target_page": merge_target_page,
            }

    metadata = build_trajectory_metadata(args.trajectory_id, args.annotations_dir)
    metadata.update({
        "topology_type": "gt_spine_branches",
        "root_page_id": spine_page_ids[0],
        "spine_page_ids": spine_page_ids,
        "effective_spine_page_ids": effective_spine_id,
        "canonical_page_count": len(spine_page_ids),
        "branch_page_count": len(branch_pages),
        "page_family_count": len(families),
        "total_pages": len(pages),
        "output_canvas_size": list(families[0]["canvas_size"]),
        "canvas_size": list(families[0]["canvas_size"]),  # env_utils expects this for bbox normalization
        "branch_pages_per_step": max_branches,
        "visual_mode": "crop_reconstructed_native",
        "canonical_render_mode": "extracted_crops",
    })

    ui_structure = {"pages": pages, "metadata": metadata}
    layer = {
        "root": _build_gt_layer_tree(spine_page_ids[0], pages, tree_children),
        "metadata": metadata,
    }

    with open(os.path.join(output_dir, "ui_structure.json"), "w", encoding="utf-8") as f:
        json.dump(ui_structure, f, indent=2, ensure_ascii=False)
    with open(os.path.join(output_dir, "ui_structure_layer.json"), "w", encoding="utf-8") as f:
        json.dump(layer, f, indent=2, ensure_ascii=False)

    _visualize_serialized_graph(pages, os.path.join(output_dir, "ui_topology.png"))
    return ui_structure

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate UI environment tree")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--icon_source", type=str, default="synthetic",
                        choices=["synthetic", "trajectory"],
                        help="Whether to sample icons from the built-in synthetic pool or a GUIOdyssey trajectory")
    parser.add_argument("--trajectory_id", type=str, default=None,
                        help="GUIOdyssey trajectory ID when --icon_source trajectory")
    parser.add_argument("--icons_dir", type=str, default="icons",
                        help="Synthetic icon directory used when --icon_source synthetic")
    parser.add_argument("--icons_metadata", type=str,
                        default="data_engine/real_icons/icons_metadata.json",
                        help="Sim2real icon metadata JSON used when --icon_source trajectory")
    parser.add_argument("--trajectory_extract_mode", type=str, default="live",
                        choices=["live", "metadata"],
                        help="When using trajectory icons, extract directly from the selected GUIOdyssey trajectory or reuse a prebuilt metadata file")
    parser.add_argument("--omniparser_weights", type=str,
                        default="/ext_hdd/nhkoh/OmniParser/weights",
                        help="OmniParser weights directory used for live trajectory icon extraction")
    parser.add_argument("--annotations_dir", type=str,
                        default="/ext_hdd/nhkoh/dataset/GUIOdyssey/annotations",
                        help="GUIOdyssey annotations directory used when --icon_source trajectory")
    parser.add_argument("--screenshots_dir", type=str,
                        default="/ext_hdd/nhkoh/dataset/GUIOdyssey/screenshots",
                        help="GUIOdyssey screenshots directory used when --icon_source trajectory")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Optional output directory. Defaults to a timestamped directory under ui_environment_448/")
    parser.add_argument("--branches_per_step", type=int, default=2,
                        help="Number of deterministic branch pages to attach to each GT step in trajectory mode")
    args = parser.parse_args()

    # Set random seed for reproducibility
    random.seed(args.seed)

    # Paper-aligned configuration (GE-Lab paper Section 3.1)
    # Creates 5-subtree structure with 231 total pages
    # - Subtrees 0-1: SFT Path training
    # - Subtrees 2-3: RL training
    # - Subtree 4: OOD testing (held out)
    nodes_per_level = [5, 3, 2, 2, 1, 1]  # 5 subtrees, ~46 pages each = 231 total
    tree_depth = len(nodes_per_level) + 1  # depth = 7
    is_random_node = False
    
    # Calculate required number of icons
    required_icons = calculate_required_icons(tree_depth, nodes_per_level)
    print(f"Minimum number of icons needed for tree structure: {required_icons}")
    
    # Create output directory (448x448 paper-aligned)
    output_dir = args.output_dir or os.path.join("ui_environment_448", time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(output_dir, exist_ok=True)

    if args.icon_source == "trajectory":
        if not args.trajectory_id:
            parser.error("--trajectory_id is required when --icon_source trajectory")
        ui_structure = generate_trajectory_family_environment(args, output_dir)
        metadata = ui_structure.get("metadata", {})
        print(f"Environment data saved to: {os.path.join(output_dir, 'ui_structure.json')}")
        print(
            "Generated GT-spine family environment: "
            f"{metadata.get('canonical_page_count', 0)} canonical pages, "
            f"{metadata.get('branch_page_count', 0)} branch pages, "
            f"{metadata.get('total_pages', 0)} total."
        )
    else:
        icon_data = load_icons_from_directory(args.icons_dir, required_icons, output_dir)
        test_icons = [img for img, _ in icon_data]
        test_funcs = [func for _, func in icon_data]

        available_icons = len(test_funcs)
        if available_icons < required_icons:
            raise ValueError(
                f"Insufficient number of icons! Need {required_icons}, but only have {available_icons}\n"
                f"Current configuration:\n"
                f"- Tree depth: {tree_depth}\n"
                f"- Nodes per level: {nodes_per_level}"
            )

        print(f"Icon count check passed: Need {required_icons}, Available {available_icons}")

        env = DynamicTopoEnv(
            icon_images=test_icons,
            func_descs=test_funcs,
            tree_depth=tree_depth,
            nodes_per_level=nodes_per_level,
            is_random_node=is_random_node
        )

        json_path = env.save_environment_data(output_dir, seed=args.seed, extra_metadata=None)
        print(f"Environment data saved to: {json_path}")

        env.visualize_topology(os.path.join(os.path.dirname(json_path), 'ui_topology.png'))

        print("\n=== Initial Icon Status ===")
        print(f"Total icons: {len(env.ui_manager.total_icons)}")
        print(f"Icon list: {[e.func_desc for e in env.ui_manager.total_icons]}")

        print("\n=== Page Icon Distribution ===")
        for page_id, page_data in env.transition_graph.nodes(data=True):
            page = page_data['page']
            normal_icons = [e.func_desc for e in page.elements if e.func_desc not in ['back', 'home']]
            children = [v for u, v in env.transition_graph.edges(page_id)
                       if env.transition_graph[u][v]['action'] not in ['back', 'home']]
            print(f"\nPage {page_id}:")
            print(f"Icons owned: {normal_icons}")
            print(f"Connected child nodes: {children}")

        obs_img, obs_layout = env.reset()

        actions = ['icon_0', 'icon_1', 'back', 'home', 'icon_2', 'icon_3']
        for action in actions:
            print(f"Execute action: {action}")
            (obs, layout), reward, done = env.step(action)
            print(f"Reward: {reward}, Current page: {env.current_page}")

        status = env.ui_manager.get_icon_status()
        print(f"Icon status: {status}")
