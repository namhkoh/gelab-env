import random
from typing import List, Dict, Tuple, Optional
import networkx as nx
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import time
from megfile import smart_glob, smart_open, smart_path_join
from io import BytesIO
from PIL import ImageFont
import glob
# --------------------------
# Data Structure Definitions
# --------------------------
class UIElement:
    """UI element class, containing icon and function description"""
    def __init__(self, image: Image.Image, func: str):
        self.raw_image = image.resize((200, 200))  # Modified to 200x200, was 50x50
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

    def save_environment_data(self, output_dir: str = "output"):
        """Save environment data, including page images, transition relationships, and configuration parameters"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Create pages subdirectory
        pages_dir = os.path.join(output_dir, "pages")
        os.makedirs(pages_dir, exist_ok=True)
        
        # Save configuration parameters
        config_path = os.path.join(output_dir, "config.json")
        config_data = {
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
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump({
                "pages": pages_data,
                "metadata": {
                    "total_pages": len(pages_data),
                    "tree_depth": self.topo_generator.max_depth,
                    "nodes_per_level": self.topo_generator.nodes_per_level,
                    "config_file": "config.json"  # Add configuration file reference
                }
            }, f, indent=2, ensure_ascii=False)
        
        # New: Generate JSON file with hierarchical structure
        self.generate_layered_structure(pages_data, output_dir)
        
        return json_path

    def generate_layered_structure(self, pages_data, output_dir):
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
        with open(layer_json_path, 'w', encoding='utf-8') as f:
            json.dump({
                "root": tree_structure,
                "metadata": {
                    "total_pages": len(pages_data),
                    "tree_depth": self.topo_generator.max_depth,
                    "nodes_per_level": self.topo_generator.nodes_per_level,
                    "config_file": "config.json"
                }
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
        
        # Modify system icon colors and styles
        back_img = Image.new('RGB', (200, 200), (255, 200, 200))  # Light red, size modified to 200x200
        home_img = Image.new('RGB', (200, 200), (200, 255, 200))  # Light green, size modified to 200x200
        
        # Add text
        for img, text in [(back_img, 'back'), (home_img, 'home')]:
            draw = ImageDraw.Draw(img)

            font = ImageFont.truetype("font/helvetica.ttf", 60)  # Increase system button font size
            # Calculate text size to center display
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            x = (200 - text_width) // 2
            y = (200 - text_height) // 2
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
    CANVAS_SIZE = (1179, 2556)
    ICON_HEIGHT = 200  # Modified to 200, was 50
    ICON_WIDTH = 200   # Modified to 200, was 50
    MARGIN = 60  # Slightly increase edge white space
    TOP_MARGIN = 150  # Slightly increase top margin
    
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
        title_width = 400  # Increase reserved title width, adapt to wider screen
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
        img = Image.new('RGB', (1179, 2556), (255, 255, 255))  # Modified canvas size
        draw = ImageDraw.Draw(img)
        
        # Draw page title
        title_bbox = page.layout['page_title']
        title_text = page.page_id
        # Use larger font
        try:
            from PIL import ImageFont
            font = ImageFont.truetype("font/helvetica.ttf", 80, encoding="unic")  # Increase title font size
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
    all_icon_paths = glob.glob(pattern)
    
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

if __name__ == "__main__":
    # Modify configuration parameters
    # tree_depth = 5         # Tree depth
    # nodes_per_level = [15,4,5,3]  # List of child nodes per level, length should be tree_depth-1
    # nodes_per_level = [4,2,3,2]  # List of child nodes per level, length should be tree_depth-1
    nodes_per_level = [5, 3, 2, 2, 1, 1]
    tree_depth = len(nodes_per_level)+1
    is_random_node = False
    
    # Calculate required number of icons
    required_icons = calculate_required_icons(tree_depth, nodes_per_level)
    print(f"Minimum number of icons needed for tree structure: {required_icons}")
    
    # Create output directory
    output_dir = os.path.join("ui_environment", time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(output_dir, exist_ok=True)
    
    # Modify icon directory to local path
    icons_dir = "icons"
    icon_data = load_icons_from_directory(icons_dir, required_icons, output_dir)
    test_icons = [img for img, _ in icon_data]
    test_funcs = [func for _, func in icon_data]
    
    # Check if number of icons is sufficient
    available_icons = len(test_funcs)
    if available_icons < required_icons:
        raise ValueError(
            f"Insufficient number of icons! Need {required_icons}, but only have {available_icons}\n"
            f"Current configuration:\n"
            f"- Tree depth: {tree_depth}\n"
            f"- Nodes per level: {nodes_per_level}"
        )
    
    print(f"Icon count check passed: Need {required_icons}, Available {available_icons}")
    
    # Initialize environment
    env = DynamicTopoEnv(
        icon_images=test_icons,
        func_descs=test_funcs,
        tree_depth=tree_depth,
        nodes_per_level=nodes_per_level,
        is_random_node=is_random_node
    )
    
    # Save environment data to same directory
    json_path = env.save_environment_data(output_dir)
    print(f"Environment data saved to: {json_path}")
    
    # Visualize topology structure
    env.visualize_topology(os.path.join(os.path.dirname(json_path), 'ui_topology.png'))
    
    # Add debug information
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
    
    # Simulation run
    obs_img, obs_layout = env.reset()
    
    # Example operation sequence
    actions = ['icon_0', 'icon_1', 'back', 'home', 'icon_2', 'icon_3']
    for action in actions:
        print(f"Execute action: {action}")
        (obs, layout), reward, done = env.step(action)
        print(f"Reward: {reward}, Current page: {env.current_page}")

    # Check icon status anytime
    status = env.ui_manager.get_icon_status()
    print(f"Icon status: {status}")