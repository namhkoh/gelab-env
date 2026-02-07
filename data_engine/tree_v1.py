"""
GE-Lab V1: Extended environment with TEXT action support.

Minimal extension to add:
1. Search bar component on select pages
2. TEXT action that navigates to result pages
3. Maintains existing CLICK navigation

Usage:
    python data_engine/tree_v1.py
"""

import os
import json
import random
from typing import List, Dict, Tuple, Optional
from PIL import Image, ImageDraw, ImageFont
import networkx as nx

from tree import (
    UIElement, UIPage, UIManager, TopologyGenerator, 
    LayoutGenerator, TopologyBuilder, RenderEngine,
    load_icons_from_directory, calculate_required_icons
)


class SearchBarComponent:
    """Search bar UI component."""
    
    def __init__(self, placeholder: str = "Search..."):
        self.placeholder = placeholder
        self.value = ""
        self.is_active = False
    
    def render(self, draw: ImageDraw.Draw, bbox: Tuple[int, int, int, int], font=None):
        """Render search bar to image."""
        x1, y1, x2, y2 = bbox
        
        # Background - darker for visibility
        draw.rounded_rectangle([x1, y1, x2, y2], radius=8, fill=(245, 245, 245), outline=(100, 100, 100), width=2)
        
        # Search icon (simple magnifying glass)
        icon_size = (y2 - y1) - 10
        icon_x = x1 + 8
        icon_y = y1 + 5
        draw.ellipse([icon_x, icon_y, icon_x + icon_size - 5, icon_y + icon_size - 5], 
                     outline=(150, 150, 150), width=2)
        draw.line([icon_x + icon_size - 8, icon_y + icon_size - 8, 
                   icon_x + icon_size, icon_y + icon_size], 
                  fill=(150, 150, 150), width=2)
        
        # Text
        text = self.value if self.value else self.placeholder
        text_color = (50, 50, 50) if self.value else (150, 150, 150)
        text_x = icon_x + icon_size + 8
        text_y = y1 + (y2 - y1) // 2 - 8
        draw.text((text_x, text_y), text, fill=text_color, font=font)


class SearchResultPage:
    """Represents a search result page linked to a query."""
    
    def __init__(self, query: str, result_pages: List[str]):
        self.query = query
        self.result_pages = result_pages


class LayoutGeneratorV1(LayoutGenerator):
    """Extended layout generator with search bar support."""
    
    SEARCH_BAR_HEIGHT = 40
    SEARCH_BAR_MARGIN = 15
    
    @classmethod
    def generate_with_search(cls, elements: List[UIElement], has_search: bool = False) -> Dict[str, Tuple[int, int, int, int]]:
        """Generate layout with optional search bar at top."""
        positions = {}
        
        # Search bar position (if enabled)
        if has_search:
            sb_x1 = cls.MARGIN
            sb_y1 = cls.MARGIN
            sb_x2 = cls.CANVAS_SIZE[0] - cls.MARGIN
            sb_y2 = sb_y1 + cls.SEARCH_BAR_HEIGHT
            positions['search_bar'] = (sb_x1, sb_y1, sb_x2, sb_y2)
            
            # Adjust top margin for other elements
            adjusted_top = sb_y2 + cls.SEARCH_BAR_MARGIN
        else:
            adjusted_top = cls.MARGIN
        
        # Rest of layout generation (icons)
        y1 = adjusted_top
        
        # Find system icons
        back_element = next((e for e in elements if e.func_desc == 'back'), None)
        home_element = next((e for e in elements if e.func_desc == 'home'), None)
        
        # System icon positions
        if back_element:
            x1 = cls.MARGIN
            positions['back'] = (x1, y1, x1 + cls.ICON_WIDTH, y1 + cls.ICON_HEIGHT)
        
        if home_element:
            x1 = cls.CANVAS_SIZE[0] - cls.ICON_WIDTH - cls.MARGIN
            positions['home'] = (x1, y1, x1 + cls.ICON_WIDTH, y1 + cls.ICON_HEIGHT)
        
        # Page title
        title_width = cls.CANVAS_SIZE[0] // 2
        title_x = (cls.CANVAS_SIZE[0] - title_width) // 2
        positions['page_title'] = (title_x, y1, title_x + title_width, y1 + cls.ICON_HEIGHT)
        
        # Normal icons
        normal_elements = [e for e in elements if e.func_desc not in ['back', 'home']]
        
        if not normal_elements:
            return positions
        
        # Generate grid positions below search bar and system icons
        icon_start_y = y1 + cls.ICON_HEIGHT + 30
        usable_width = cls.CANVAS_SIZE[0] - 2 * cls.MARGIN
        usable_height = cls.CANVAS_SIZE[1] - icon_start_y - cls.MARGIN
        
        min_spacing_x = cls.ICON_WIDTH + 30
        min_spacing_y = cls.ICON_HEIGHT + 30
        
        num_cols = min(usable_width // min_spacing_x, 5)
        num_rows = min(usable_height // min_spacing_y, 6)
        
        start_x = cls.MARGIN + (usable_width - (num_cols - 1) * min_spacing_x) // 2
        start_y = icon_start_y
        
        predefined_positions = []
        for row in range(num_rows):
            for col in range(num_cols):
                x = start_x + col * min_spacing_x
                y = start_y + row * min_spacing_y
                predefined_positions.append((int(x), int(y)))
        
        if len(predefined_positions) < len(normal_elements):
            while len(predefined_positions) < len(normal_elements):
                predefined_positions.append(random.choice(predefined_positions))
        
        selected_positions = random.sample(predefined_positions, len(normal_elements))
        
        for elem, (x, y) in zip(normal_elements, selected_positions):
            positions[elem.func_desc] = (x, y, x + cls.ICON_WIDTH, y + cls.ICON_HEIGHT)
        
        return positions


class RenderEngineV1(RenderEngine):
    """Extended render engine with search bar support."""
    
    def __init__(self):
        super().__init__()
        self.search_bars = {}
    
    def render(self, page: UIPage, search_bar: SearchBarComponent = None) -> Image.Image:
        """Render page with optional search bar."""
        img = Image.new('RGB', LayoutGenerator.CANVAS_SIZE, (255, 255, 255))
        draw = ImageDraw.Draw(img)
        
        try:
            font = ImageFont.truetype("font/helvetica.ttf", 14)
            title_font = ImageFont.truetype("font/helvetica.ttf", 24)
        except:
            font = None
            title_font = None
        
        # Draw search bar if present
        if search_bar and 'search_bar' in page.layout:
            search_bar.render(draw, page.layout['search_bar'], font)
        
        # Draw page title
        if 'page_title' in page.layout:
            title_bbox = page.layout['page_title']
            title_text = page.page_id
            text_bbox = draw.textbbox((0, 0), title_text, font=title_font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            x = title_bbox[0] + (title_bbox[2] - title_bbox[0] - text_width) // 2
            y = title_bbox[1] + (title_bbox[3] - title_bbox[1] - text_height) // 2
            draw.text((x, y), title_text, fill=(0, 0, 0), font=title_font)
        
        # Draw icons
        for elem in page.elements:
            if elem.func_desc in page.layout:
                bbox = page.layout[elem.func_desc]
                img.paste(elem.raw_image, (bbox[0], bbox[1]))
        
        return img


class UIPageV1(UIPage):
    """Extended page with search bar support."""
    
    def __init__(self, page_id: str, elements: List[UIElement], layout: Dict, 
                 parent: Optional[str] = None, has_search: bool = False,
                 search_queries: List[str] = None):
        super().__init__(page_id, elements, layout, parent)
        self.has_search = has_search
        self.search_queries = search_queries or []
        self.search_bar = SearchBarComponent() if has_search else None


class DynamicTopoEnvV1:
    """Extended environment with TEXT action support."""
    
    def __init__(self, 
                 icon_images: List[Image.Image], 
                 func_descs: List[str],
                 tree_depth: int = 3,
                 nodes_per_level: List[int] = None,
                 search_enabled_pages: List[str] = None,
                 search_query_mapping: Dict[str, List[str]] = None):
        """
        Initialize V1 environment.
        
        Args:
            search_enabled_pages: List of page_ids that have search bars
            search_query_mapping: Dict mapping query strings to target page_ids
        """
        if len(icon_images) != len(func_descs):
            raise ValueError("Number of icons does not match function descriptions")
        
        if nodes_per_level is None:
            nodes_per_level = [2, 3]
        
        self.ui_manager = UIManager(icon_images, func_descs)
        self.topo_generator = TopologyGenerator(tree_depth, nodes_per_level, False)
        self.render_engine = RenderEngineV1()
        self.transition_graph = nx.DiGraph()
        
        self.search_enabled_pages = set(search_enabled_pages or [])
        self.search_query_mapping = search_query_mapping or {}
        
        self._build_environment()
        self.reset()
    
    def _build_environment(self):
        """Build environment with search-enabled pages."""
        hierarchy, pages = self.topo_generator.generate(self.ui_manager)
        
        # Convert pages to V1 format with search support
        pages_v1 = {}
        for page_id, page in pages.items():
            has_search = page_id in self.search_enabled_pages
            
            if has_search:
                layout = LayoutGeneratorV1.generate_with_search(page.elements, has_search=True)
            else:
                layout = page.layout
            
            queries = [q for q, targets in self.search_query_mapping.items() 
                      if page_id in self.search_enabled_pages]
            
            pages_v1[page_id] = UIPageV1(
                page_id=page.page_id,
                elements=page.elements,
                layout=layout,
                parent=page.parent,
                has_search=has_search,
                search_queries=queries
            )
        
        self.transition_graph = TopologyBuilder.build(hierarchy, pages_v1)
        
        # Add TEXT transitions for search queries
        for query, target_pages in self.search_query_mapping.items():
            for source_page in self.search_enabled_pages:
                if source_page in self.transition_graph:
                    for target in target_pages:
                        if target in self.transition_graph:
                            self.transition_graph.add_edge(
                                source_page, target,
                                action=f"TEXT:{query}",
                                action_type="TEXT"
                            )
    
    def reset(self) -> Tuple[Image.Image, dict]:
        """Reset to initial state."""
        self.current_page = "page_0"
        return self.get_observation()
    
    def get_observation(self) -> Tuple[Image.Image, dict]:
        """Get current observation."""
        page = self.transition_graph.nodes[self.current_page]['page']
        search_bar = page.search_bar if hasattr(page, 'search_bar') else None
        return self.render_engine.render(page, search_bar), page.layout
    
    def step(self, action: str, text_input: str = None) -> Tuple[Tuple[Image.Image, dict], float, bool]:
        """
        Execute action.
        
        Args:
            action: "CLICK" or "TEXT"
            text_input: For TEXT action, the query string
        
        Returns:
            (observation, layout), reward, done
        """
        if action == "TEXT" and text_input:
            new_page = self._handle_text_action(text_input)
        else:
            new_page = self._find_transition(action)
        
        reward = self._calculate_reward(action, new_page)
        
        if new_page is not None:
            self.current_page = new_page
        
        return self.get_observation(), reward, False
    
    def _handle_text_action(self, query: str) -> Optional[str]:
        """Handle TEXT action by finding matching query transition."""
        action_key = f"TEXT:{query}"
        for successor in self.transition_graph.successors(self.current_page):
            edge_data = self.transition_graph.get_edge_data(self.current_page, successor)
            if edge_data.get('action') == action_key:
                return successor
        return None
    
    def _find_transition(self, action: str) -> Optional[str]:
        """Find valid CLICK transition."""
        for successor in self.transition_graph.successors(self.current_page):
            edge_data = self.transition_graph.get_edge_data(self.current_page, successor)
            if edge_data['action'] == action and edge_data.get('action_type') != 'TEXT':
                return successor
        return None
    
    def _calculate_reward(self, action: str, new_page: Optional[str]) -> float:
        """Calculate reward."""
        if new_page is None:
            return -1.0
        return 0.0
    
    def get_available_actions(self) -> Dict[str, List]:
        """Get available actions from current page."""
        actions = {"CLICK": [], "TEXT": []}
        
        page = self.transition_graph.nodes[self.current_page]['page']
        
        # CLICK actions (icons)
        for elem in page.elements:
            if elem.func_desc in page.layout:
                bbox = page.layout[elem.func_desc]
                actions["CLICK"].append({
                    "target": elem.func_desc,
                    "bbox": bbox
                })
        
        # TEXT actions (if search bar present)
        if hasattr(page, 'has_search') and page.has_search:
            for successor in self.transition_graph.successors(self.current_page):
                edge_data = self.transition_graph.get_edge_data(self.current_page, successor)
                if edge_data.get('action_type') == 'TEXT':
                    query = edge_data['action'].replace("TEXT:", "")
                    actions["TEXT"].append({
                        "query": query,
                        "target_page": successor
                    })
        
        return actions
    
    def save_environment_data(self, output_dir: str) -> str:
        """Save environment data including V1 extensions."""
        os.makedirs(output_dir, exist_ok=True)
        pages_dir = os.path.join(output_dir, "pages")
        os.makedirs(pages_dir, exist_ok=True)
        
        pages_data = {}
        
        for node_id, node_data in self.transition_graph.nodes(data=True):
            page = node_data['page']
            
            # Render and save image
            search_bar = page.search_bar if hasattr(page, 'search_bar') else None
            page_image = self.render_engine.render(page, search_bar)
            image_path = os.path.join(pages_dir, f"{node_id}.png")
            page_image.save(image_path)
            
            # Collect page data
            pages_data[node_id] = {
                "image": f"{node_id}.png",
                "has_search": getattr(page, 'has_search', False),
                "layout": {
                    key: list(bbox) for key, bbox in page.layout.items()
                },
                "transitions": []
            }
        
        # Add transitions
        for u, v, data in self.transition_graph.edges(data=True):
            action = data['action']
            action_type = data.get('action_type', 'CLICK')
            
            source_page = self.transition_graph.nodes[u]['page']
            icon_bbox = None
            
            if action_type == 'CLICK' and action in source_page.layout:
                icon_bbox = list(source_page.layout[action])
            elif action_type == 'TEXT' and 'search_bar' in source_page.layout:
                # TEXT actions use the search_bar bbox as the actionable area
                icon_bbox = list(source_page.layout['search_bar'])
            
            transition_data = {
                "action": action,
                "action_type": action_type,
                "target_page": v,
                "icon_bbox": icon_bbox
            }
            
            # Add text_input field for TEXT actions
            if action_type == 'TEXT':
                # Extract query from "TEXT:query" format
                text_query = action.replace("TEXT:", "")
                transition_data["text_input"] = text_query
            
            pages_data[u]["transitions"].append(transition_data)
        
        # Save JSON
        json_path = os.path.join(output_dir, "ui_structure_v1.json")
        with open(json_path, 'w') as f:
            json.dump({
                "version": "1.0",
                "pages": pages_data,
                "search_query_mapping": self.search_query_mapping,
                "metadata": {
                    "total_pages": len(pages_data),
                    "search_enabled_pages": list(self.search_enabled_pages),
                    "supported_actions": ["CLICK", "TEXT"]
                }
            }, f, indent=2)
        
        return json_path


def generate_sample_queries(pages: List[str], num_queries: int = 5) -> Dict[str, List[str]]:
    """Generate sample search queries mapping to random pages."""
    sample_queries = [
        "smart light bulb",
        "wireless headphones", 
        "laptop stand",
        "phone charger",
        "bluetooth speaker"
    ]
    
    query_mapping = {}
    for i, query in enumerate(sample_queries[:num_queries]):
        # Each query maps to 1-3 random result pages
        num_results = random.randint(1, min(3, len(pages)))
        query_mapping[query] = random.sample(pages, num_results)
    
    return query_mapping


if __name__ == "__main__":
    import time
    
    # Configuration
    nodes_per_level = [3, 2, 2]  # Smaller tree for testing
    tree_depth = len(nodes_per_level) + 1
    
    required_icons = calculate_required_icons(tree_depth, nodes_per_level)
    print(f"Required icons: {required_icons}")
    
    # Output directory
    output_dir = os.path.join("data_engine/ui_environment_v1", time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(output_dir, exist_ok=True)
    
    # Load icons
    icons_dir = "data_engine/icons"
    if not os.path.exists(icons_dir):
        icons_dir = "icons"
    
    icon_data = load_icons_from_directory(icons_dir, required_icons, output_dir)
    test_icons = [img for img, _ in icon_data]
    test_funcs = [func for _, func in icon_data]
    
    # Define which pages have search bars (e.g., root and first-level pages)
    search_enabled = ["page_0", "page_1", "page_2", "page_3"]
    
    # Generate query mapping (will be populated after we know all page IDs)
    # For now, use placeholder
    query_mapping = {
        "smart light bulb": ["page_4", "page_5"],
        "wireless headphones": ["page_6", "page_7"],
        "laptop stand": ["page_8"]
    }
    
    # Create V1 environment
    env = DynamicTopoEnvV1(
        icon_images=test_icons,
        func_descs=test_funcs,
        tree_depth=tree_depth,
        nodes_per_level=nodes_per_level,
        search_enabled_pages=search_enabled,
        search_query_mapping=query_mapping
    )
    
    # Save environment
    json_path = env.save_environment_data(output_dir)
    print(f"V1 Environment saved to: {json_path}")
    
    # Test actions
    print("\n=== Testing V1 Environment ===")
    obs, layout = env.reset()
    print(f"Current page: {env.current_page}")
    print(f"Available actions: {env.get_available_actions()}")
    
    # Test CLICK action
    if env.get_available_actions()["CLICK"]:
        first_click = env.get_available_actions()["CLICK"][0]["target"]
        print(f"\nExecuting CLICK on: {first_click}")
        obs, reward, done = env.step(first_click)
        print(f"New page: {env.current_page}, Reward: {reward}")
    
    # Reset and test TEXT action
    env.reset()
    if env.get_available_actions()["TEXT"]:
        first_text = env.get_available_actions()["TEXT"][0]["query"]
        print(f"\nExecuting TEXT: '{first_text}'")
        obs, reward, done = env.step("TEXT", text_input=first_text)
        print(f"New page: {env.current_page}, Reward: {reward}")
    
    print("\n=== V1 Environment Test Complete ===")
