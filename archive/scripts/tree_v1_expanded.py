"""
GE-Lab V1 Expanded: Realistic environment matching ground truth trajectory.

Supports:
- CLICK: Tap on UI elements
- TEXT: Type into search bars
- SCROLL: Scroll up/down on pages
- KEY_HOME: Return to home screen
- COMPLETE: Task completion

Based on Sequence/2451545728360121.json structure.
"""

import os
import json
import random
from typing import List, Dict, Tuple, Optional
from PIL import Image, ImageDraw, ImageFont
import networkx as nx

# Canvas size matching existing dataset
CANVAS_SIZE = (448, 448)
ICON_SIZE = 50
MARGIN = 20


class UIComponent:
    """Base class for UI components."""
    
    def __init__(self, component_id: str, bbox: Tuple[int, int, int, int]):
        self.component_id = component_id
        self.bbox = bbox
    
    def render(self, draw: ImageDraw.Draw, font=None):
        raise NotImplementedError


class AppIcon(UIComponent):
    """App icon on home screen."""
    
    # Class-level icon cache
    _icon_cache = {}
    
    def __init__(self, component_id: str, bbox: Tuple[int, int, int, int], 
                 name: str, color: Tuple[int, int, int], icon_image: Image.Image = None):
        super().__init__(component_id, bbox)
        self.name = name
        self.color = color
        self.icon_image = icon_image
        self._load_icon_if_available()
    
    def _load_icon_if_available(self):
        """Try to load icon from assets folder."""
        if self.icon_image is not None:
            return
        
        # Map component_id to asset name
        asset_map = {
            "app_chrome": "chrome",
            "app_amazon": "amazon",
            "app_tiktok": "tiktok",
            "app_ebay": "ebay",
            "app_settings": "settings",
            "app_photos": "photos",
        }
        
        asset_name = asset_map.get(self.component_id)
        if asset_name:
            asset_path = f"data_engine/assets/{asset_name}.png"
            if os.path.exists(asset_path):
                if asset_path not in AppIcon._icon_cache:
                    AppIcon._icon_cache[asset_path] = Image.open(asset_path).convert('RGBA')
                self.icon_image = AppIcon._icon_cache[asset_path]
    
    def render(self, draw: ImageDraw.Draw, font=None):
        x1, y1, x2, y2 = self.bbox
        
        # Draw app name below icon
        try:
            small_font = ImageFont.truetype("font/helvetica.ttf", 10)
        except:
            small_font = None
        
        text_bbox = draw.textbbox((0, 0), self.name[:8], font=small_font)
        text_w = text_bbox[2] - text_bbox[0]
        text_x = x1 + (x2 - x1 - text_w) // 2
        text_y = y2 + 2
        draw.text((text_x, text_y), self.name[:8], fill=(0, 0, 0), font=small_font)
        
        if not self.icon_image:
            # Fallback: Draw colored rectangle
            draw.rounded_rectangle([x1, y1, x2, y2], radius=10, fill=self.color, outline=(50, 50, 50))


class SearchBar(UIComponent):
    """Search bar component."""
    
    def __init__(self, component_id: str, bbox: Tuple[int, int, int, int], 
                 placeholder: str = "Search...", value: str = ""):
        super().__init__(component_id, bbox)
        self.placeholder = placeholder
        self.value = value
    
    def render(self, draw: ImageDraw.Draw, font=None):
        x1, y1, x2, y2 = self.bbox
        
        # Background
        draw.rounded_rectangle([x1, y1, x2, y2], radius=8, fill=(245, 245, 245), outline=(100, 100, 100), width=2)
        
        # Search icon
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
        draw.text((text_x, text_y), text[:30], fill=text_color, font=font)


class NavButton(UIComponent):
    """Navigation button (back, home, tabs)."""
    
    def __init__(self, component_id: str, bbox: Tuple[int, int, int, int], 
                 label: str, bg_color: Tuple[int, int, int] = (220, 220, 220)):
        super().__init__(component_id, bbox)
        self.label = label
        self.bg_color = bg_color
    
    def render(self, draw: ImageDraw.Draw, font=None):
        x1, y1, x2, y2 = self.bbox
        draw.rounded_rectangle([x1, y1, x2, y2], radius=5, fill=self.bg_color, outline=(150, 150, 150))
        
        try:
            small_font = ImageFont.truetype("font/helvetica.ttf", 12)
        except:
            small_font = None
        
        text_bbox = draw.textbbox((0, 0), self.label, font=small_font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        text_x = x1 + (x2 - x1 - text_w) // 2
        text_y = y1 + (y2 - y1 - text_h) // 2
        draw.text((text_x, text_y), self.label, fill=(0, 0, 0), font=small_font)


class ContentCard(UIComponent):
    """Content card (search result, product listing)."""
    
    def __init__(self, component_id: str, bbox: Tuple[int, int, int, int],
                 title: str, subtitle: str = "", color: Tuple[int, int, int] = (240, 240, 250)):
        super().__init__(component_id, bbox)
        self.title = title
        self.subtitle = subtitle
        self.color = color
    
    def render(self, draw: ImageDraw.Draw, font=None):
        x1, y1, x2, y2 = self.bbox
        
        # Card background
        draw.rounded_rectangle([x1, y1, x2, y2], radius=8, fill=self.color, outline=(200, 200, 200))
        
        # Thumbnail placeholder
        thumb_size = min(y2 - y1 - 10, 40)
        draw.rectangle([x1 + 5, y1 + 5, x1 + 5 + thumb_size, y1 + 5 + thumb_size], 
                      fill=(200, 200, 200), outline=(180, 180, 180))
        
        try:
            title_font = ImageFont.truetype("font/helvetica.ttf", 11)
            sub_font = ImageFont.truetype("font/helvetica.ttf", 9)
        except:
            title_font = sub_font = None
        
        # Title
        text_x = x1 + thumb_size + 15
        draw.text((text_x, y1 + 8), self.title[:25], fill=(30, 30, 30), font=title_font)
        
        # Subtitle
        if self.subtitle:
            draw.text((text_x, y1 + 25), self.subtitle[:30], fill=(100, 100, 100), font=sub_font)


class ActionButton(UIComponent):
    """Action button (Add to Cart, Buy Now)."""
    
    def __init__(self, component_id: str, bbox: Tuple[int, int, int, int],
                 label: str, bg_color: Tuple[int, int, int] = (255, 153, 0)):
        super().__init__(component_id, bbox)
        self.label = label
        self.bg_color = bg_color
    
    def render(self, draw: ImageDraw.Draw, font=None):
        x1, y1, x2, y2 = self.bbox
        draw.rounded_rectangle([x1, y1, x2, y2], radius=5, fill=self.bg_color, outline=(200, 120, 0))
        
        try:
            btn_font = ImageFont.truetype("font/helvetica.ttf", 12)
        except:
            btn_font = None
        
        text_bbox = draw.textbbox((0, 0), self.label, font=btn_font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        text_x = x1 + (x2 - x1 - text_w) // 2
        text_y = y1 + (y2 - y1 - text_h) // 2
        draw.text((text_x, text_y), self.label, fill=(0, 0, 0), font=btn_font)


class ScrollIndicator(UIComponent):
    """Visual indicator that page is scrollable."""
    
    def __init__(self, component_id: str, bbox: Tuple[int, int, int, int], scroll_position: float = 0.0):
        super().__init__(component_id, bbox)
        self.scroll_position = scroll_position  # 0.0 = top, 1.0 = bottom
    
    def render(self, draw: ImageDraw.Draw, font=None):
        x1, y1, x2, y2 = self.bbox
        
        # Scrollbar track
        draw.rectangle([x1, y1, x2, y2], fill=(230, 230, 230), outline=(200, 200, 200))
        
        # Scrollbar thumb
        track_height = y2 - y1
        thumb_height = max(20, track_height // 4)
        thumb_y = y1 + int((track_height - thumb_height) * self.scroll_position)
        draw.rectangle([x1 + 2, thumb_y, x2 - 2, thumb_y + thumb_height], 
                      fill=(150, 150, 150), outline=(120, 120, 120))


class PageV1:
    """Extended page with multiple component types."""
    
    def __init__(self, page_id: str, page_type: str, components: List[UIComponent],
                 parent: Optional[str] = None, is_scrollable: bool = False,
                 scroll_position: float = 0.0, max_scroll: float = 1.0):
        self.page_id = page_id
        self.page_type = page_type  # 'home', 'app', 'search_results', 'detail', 'product'
        self.components = components
        self.parent = parent
        self.is_scrollable = is_scrollable
        self.scroll_position = scroll_position
        self.max_scroll = max_scroll
    
    def get_layout(self) -> Dict[str, Tuple[int, int, int, int]]:
        """Get layout dict for all components."""
        return {c.component_id: c.bbox for c in self.components}
    
    def get_component(self, component_id: str) -> Optional[UIComponent]:
        """Get component by ID."""
        for c in self.components:
            if c.component_id == component_id:
                return c
        return None


class RenderEngineV1Expanded:
    """Render engine for expanded V1 pages."""
    
    def __init__(self):
        try:
            self.font = ImageFont.truetype("font/helvetica.ttf", 14)
            self.title_font = ImageFont.truetype("font/helvetica.ttf", 18)
        except:
            self.font = None
            self.title_font = None
    
    def render(self, page: PageV1, icon_images: Dict[str, Image.Image] = None) -> Image.Image:
        """Render page to image."""
        img = Image.new('RGB', CANVAS_SIZE, (255, 255, 255))
        draw = ImageDraw.Draw(img)
        
        # Draw page title/header based on page type
        if page.page_type == 'home':
            header_text = ""  # No header for home screen
        else:
            header_text = page.page_type.replace('_', ' ').upper()
            draw.text((CANVAS_SIZE[0] // 2 - 50, 5), header_text, fill=(100, 100, 100), font=self.font)
        
        # Draw all components
        for component in page.components:
            component.render(draw, self.font)
            
            # Paste icon images for AppIcon components
            if isinstance(component, AppIcon) and component.icon_image:
                icon_img = component.icon_image.resize(
                    (component.bbox[2] - component.bbox[0], component.bbox[3] - component.bbox[1]),
                    Image.Resampling.LANCZOS
                )
                # Handle RGBA images
                if icon_img.mode == 'RGBA':
                    img.paste(icon_img, (component.bbox[0], component.bbox[1]), icon_img)
                else:
                    img.paste(icon_img, (component.bbox[0], component.bbox[1]))
        
        # Draw scroll indicator if scrollable
        if page.is_scrollable:
            scroll_bbox = (CANVAS_SIZE[0] - 15, 60, CANVAS_SIZE[0] - 5, CANVAS_SIZE[1] - 20)
            scroll_indicator = ScrollIndicator("scroll", scroll_bbox, page.scroll_position)
            scroll_indicator.render(draw, self.font)
        
        return img


class ExpandedEnvironment:
    """
    Expanded V1 environment matching ground truth trajectory structure.
    
    Page types:
    - home: Home screen with app icons
    - app_chrome: Chrome browser main page
    - app_amazon: Amazon app main page
    - search_results: Search results listing
    - article: Article/content page (scrollable)
    - product: Product detail page (scrollable)
    """
    
    def __init__(self):
        self.pages: Dict[str, PageV1] = {}
        self.transition_graph = nx.MultiDiGraph()  # MultiDiGraph for multiple edges between same nodes
        self.current_page = "home"
        self.render_engine = RenderEngineV1Expanded()
        
        self._build_environment()
    
    def _build_environment(self):
        """Build the environment structure."""
        
        # === HOME SCREEN ===
        home_components = [
            AppIcon("app_chrome", (80, 100, 130, 150), "Chrome", (66, 133, 244)),
            AppIcon("app_amazon", (170, 100, 220, 150), "Amazon", (255, 153, 0)),
            AppIcon("app_tiktok", (260, 100, 310, 150), "TikTok", (0, 0, 0)),
            AppIcon("app_ebay", (350, 100, 400, 150), "eBay", (0, 100, 210)),
            AppIcon("app_settings", (80, 200, 130, 250), "Settings", (128, 128, 128)),
            AppIcon("app_photos", (170, 200, 220, 250), "Photos", (76, 175, 80)),
        ]
        self.pages["home"] = PageV1("home", "home", home_components)
        
        # === CHROME APP ===
        chrome_components = [
            NavButton("back", (MARGIN, MARGIN, MARGIN + 50, MARGIN + 35), "< Back", (255, 200, 200)),
            NavButton("tabs", (CANVAS_SIZE[0] - 70, MARGIN, CANVAS_SIZE[0] - MARGIN, MARGIN + 35), "Tabs"),
            SearchBar("search_bar", (MARGIN, 70, CANVAS_SIZE[0] - MARGIN, 110), "Search or type URL"),
            ContentCard("shortcut_1", (MARGIN, 130, CANVAS_SIZE[0] // 2 - 10, 190), "Google", "google.com"),
            ContentCard("shortcut_2", (CANVAS_SIZE[0] // 2 + 10, 130, CANVAS_SIZE[0] - MARGIN, 190), "YouTube", "youtube.com"),
        ]
        self.pages["app_chrome"] = PageV1("app_chrome", "app", chrome_components, parent="home")
        
        # === CHROME SEARCH RESULTS ===
        search_results_components = [
            NavButton("back", (MARGIN, MARGIN, MARGIN + 50, MARGIN + 35), "< Back", (255, 200, 200)),
            NavButton("home_btn", (CANVAS_SIZE[0] - 70, MARGIN, CANVAS_SIZE[0] - MARGIN, MARGIN + 35), "Home", (200, 255, 200)),
            SearchBar("search_bar", (MARGIN, 70, CANVAS_SIZE[0] - MARGIN, 110), value="smart light bulbs"),
            ContentCard("result_1", (MARGIN, 125, CANVAS_SIZE[0] - MARGIN, 185), "Best Smart Lights 2024", "CNET - Expert Reviews", (240, 248, 255)),
            ContentCard("result_2", (MARGIN, 195, CANVAS_SIZE[0] - MARGIN, 255), "Top 10 Smart Bulbs", "Tom's Guide", (255, 248, 240)),
            ContentCard("result_3", (MARGIN, 265, CANVAS_SIZE[0] - MARGIN, 325), "Smart Light Comparison", "TechRadar", (248, 255, 240)),
        ]
        self.pages["chrome_search_results"] = PageV1("chrome_search_results", "search_results", 
                                                     search_results_components, parent="app_chrome")
        
        # === ARTICLE PAGE (scrollable) ===
        article_components = [
            NavButton("back", (MARGIN, MARGIN, MARGIN + 50, MARGIN + 35), "< Back", (255, 200, 200)),
            NavButton("home_btn", (CANVAS_SIZE[0] - 70, MARGIN, CANVAS_SIZE[0] - MARGIN, MARGIN + 35), "Home", (200, 255, 200)),
            ContentCard("article_header", (MARGIN, 60, CANVAS_SIZE[0] - MARGIN - 20, 120), 
                       "Best Smart Lights 2024", "By CNET Editors", (230, 240, 255)),
            ContentCard("product_1", (MARGIN, 135, CANVAS_SIZE[0] - MARGIN - 20, 195), 
                       "Wyze Bulb - $8", "Best Budget Option", (255, 250, 240)),
            ContentCard("product_2", (MARGIN, 210, CANVAS_SIZE[0] - MARGIN - 20, 270), 
                       "Philips Hue - $50", "Best Overall", (255, 250, 240)),
            ContentCard("product_3", (MARGIN, 285, CANVAS_SIZE[0] - MARGIN - 20, 345), 
                       "LIFX Mini - $25", "Best Colors", (255, 250, 240)),
            ContentCard("product_4", (MARGIN, 360, CANVAS_SIZE[0] - MARGIN - 20, 420), 
                       "Sengled Smart - $10", "Best Value", (255, 250, 240)),
        ]
        self.pages["article_page"] = PageV1("article_page", "article", article_components,
                                            parent="chrome_search_results", is_scrollable=True)
        
        # === AMAZON APP ===
        amazon_components = [
            NavButton("back", (MARGIN, MARGIN, MARGIN + 50, MARGIN + 35), "< Back", (255, 200, 200)),
            SearchBar("search_bar", (MARGIN, 70, CANVAS_SIZE[0] - MARGIN, 110), "Search Amazon"),
            ContentCard("deal_1", (MARGIN, 130, CANVAS_SIZE[0] // 2 - 10, 200), "Today's Deals", "Up to 50% off"),
            ContentCard("deal_2", (CANVAS_SIZE[0] // 2 + 10, 130, CANVAS_SIZE[0] - MARGIN, 200), "Prime Day", "Exclusive"),
            ContentCard("recent_1", (MARGIN, 220, CANVAS_SIZE[0] - MARGIN, 280), "Recently Viewed", "Smart Thermostat"),
        ]
        self.pages["app_amazon"] = PageV1("app_amazon", "app", amazon_components, parent="home")
        
        # === AMAZON SEARCH RESULTS ===
        amazon_search_components = [
            NavButton("back", (MARGIN, MARGIN, MARGIN + 50, MARGIN + 35), "< Back", (255, 200, 200)),
            NavButton("home_btn", (CANVAS_SIZE[0] - 70, MARGIN, CANVAS_SIZE[0] - MARGIN, MARGIN + 35), "Home", (200, 255, 200)),
            SearchBar("search_bar", (MARGIN, 70, CANVAS_SIZE[0] - MARGIN, 110), value="Wyze Bulb"),
            ContentCard("product_result_1", (MARGIN, 125, CANVAS_SIZE[0] - MARGIN, 195), 
                       "Wyze Bulb White", "$7.99 - 4.5 stars", (255, 250, 240)),
            ContentCard("product_result_2", (MARGIN, 205, CANVAS_SIZE[0] - MARGIN, 275), 
                       "Wyze Bulb Color", "$11.99 - 4.3 stars", (255, 250, 240)),
            ContentCard("product_result_3", (MARGIN, 285, CANVAS_SIZE[0] - MARGIN, 355), 
                       "Wyze Bulb 4-Pack", "$26.98 - 4.6 stars", (255, 250, 240)),
        ]
        self.pages["amazon_search_results"] = PageV1("amazon_search_results", "search_results",
                                                     amazon_search_components, parent="app_amazon")
        
        # === PRODUCT DETAIL PAGE (scrollable) ===
        product_components = [
            NavButton("back", (MARGIN, MARGIN, MARGIN + 50, MARGIN + 35), "< Back", (255, 200, 200)),
            NavButton("home_btn", (CANVAS_SIZE[0] - 70, MARGIN, CANVAS_SIZE[0] - MARGIN, MARGIN + 35), "Home", (200, 255, 200)),
            ContentCard("product_image", (MARGIN, 60, CANVAS_SIZE[0] - MARGIN - 20, 180), 
                       "Wyze Bulb", "Smart LED Light", (250, 250, 250)),
            ContentCard("product_price", (MARGIN, 190, CANVAS_SIZE[0] - MARGIN - 20, 240), 
                       "$7.99", "Free delivery with Prime", (255, 255, 240)),
            ActionButton("add_to_cart", (MARGIN + 20, 260, CANVAS_SIZE[0] // 2 - 10, 300), 
                        "Add to Cart", (255, 216, 20)),
            ActionButton("buy_now", (CANVAS_SIZE[0] // 2 + 10, 260, CANVAS_SIZE[0] - MARGIN - 20, 300), 
                        "Buy Now", (255, 164, 28)),
            ContentCard("product_desc", (MARGIN, 320, CANVAS_SIZE[0] - MARGIN - 20, 420), 
                       "Product Details", "Works with Alexa, Google", (248, 248, 255)),
        ]
        self.pages["product_detail"] = PageV1("product_detail", "product", product_components,
                                              parent="amazon_search_results", is_scrollable=True)
        
        # === CART/CONFIRMATION PAGE ===
        cart_components = [
            NavButton("back", (MARGIN, MARGIN, MARGIN + 50, MARGIN + 35), "< Back", (255, 200, 200)),
            NavButton("home_btn", (CANVAS_SIZE[0] - 70, MARGIN, CANVAS_SIZE[0] - MARGIN, MARGIN + 35), "Home", (200, 255, 200)),
            ContentCard("cart_item", (MARGIN, 80, CANVAS_SIZE[0] - MARGIN, 160), 
                       "Wyze Bulb", "Added to Cart - $7.99", (220, 255, 220)),
            ActionButton("checkout", (MARGIN + 50, 200, CANVAS_SIZE[0] - MARGIN - 50, 250), 
                        "Proceed to Checkout", (255, 164, 28)),
            ActionButton("complete", (MARGIN + 50, 280, CANVAS_SIZE[0] - MARGIN - 50, 330), 
                        "COMPLETE TASK", (76, 175, 80)),
        ]
        self.pages["cart_page"] = PageV1("cart_page", "cart", cart_components, parent="product_detail")
        
        # === BUILD TRANSITION GRAPH ===
        self._build_transitions()
    
    def _build_transitions(self):
        """Build the transition graph with all action types."""
        
        # Add all pages to graph
        for page_id, page in self.pages.items():
            self.transition_graph.add_node(page_id, page=page)
        
        # === CLICK transitions ===
        click_transitions = [
            # From home
            ("home", "app_chrome", "app_chrome", "CLICK"),
            ("home", "app_amazon", "app_amazon", "CLICK"),
            
            # Chrome navigation
            ("app_chrome", "home", "back", "CLICK"),
            
            # Chrome search results
            ("chrome_search_results", "app_chrome", "back", "CLICK"),
            ("chrome_search_results", "home", "home_btn", "CLICK"),
            ("chrome_search_results", "article_page", "result_1", "CLICK"),
            
            # Article page
            ("article_page", "chrome_search_results", "back", "CLICK"),
            ("article_page", "home", "home_btn", "CLICK"),
            
            # Amazon navigation
            ("app_amazon", "home", "back", "CLICK"),
            
            # Amazon search results
            ("amazon_search_results", "app_amazon", "back", "CLICK"),
            ("amazon_search_results", "home", "home_btn", "CLICK"),
            ("amazon_search_results", "product_detail", "product_result_1", "CLICK"),
            ("amazon_search_results", "product_detail", "product_result_2", "CLICK"),
            ("amazon_search_results", "product_detail", "product_result_3", "CLICK"),
            
            # Product detail
            ("product_detail", "amazon_search_results", "back", "CLICK"),
            ("product_detail", "home", "home_btn", "CLICK"),
            ("product_detail", "cart_page", "add_to_cart", "CLICK"),
            
            # Cart page
            ("cart_page", "product_detail", "back", "CLICK"),
            ("cart_page", "home", "home_btn", "CLICK"),
        ]
        
        for from_page, to_page, component_id, action_type in click_transitions:
            component = self.pages[from_page].get_component(component_id)
            bbox = component.bbox if component else None
            self.transition_graph.add_edge(
                from_page, to_page,
                action=f"CLICK:{component_id}",
                action_type="CLICK",
                component_id=component_id,
                bbox=list(bbox) if bbox else None
            )
        
        # === TEXT transitions ===
        text_transitions = [
            ("app_chrome", "chrome_search_results", "search_bar", "smart light bulbs"),
            ("app_chrome", "chrome_search_results", "search_bar", "best smart bulbs 2024"),
            ("app_amazon", "amazon_search_results", "search_bar", "Wyze Bulb"),
            ("app_amazon", "amazon_search_results", "search_bar", "smart light bulb"),
        ]
        
        for from_page, to_page, component_id, text_input in text_transitions:
            component = self.pages[from_page].get_component(component_id)
            bbox = component.bbox if component else None
            self.transition_graph.add_edge(
                from_page, to_page,
                action=f"TEXT:{text_input}",
                action_type="TEXT",
                component_id=component_id,
                bbox=list(bbox) if bbox else None,
                text_input=text_input
            )
        
        # === SCROLL transitions (same page, different scroll state) ===
        # For simplicity, SCROLL doesn't change pages but would in a full implementation
        scrollable_pages = ["article_page", "product_detail"]
        for page_id in scrollable_pages:
            component = self.pages[page_id].get_component("back")
            # SCROLL action available on scrollable pages
            self.transition_graph.add_edge(
                page_id, page_id,
                action="SCROLL:down",
                action_type="SCROLL",
                scroll_direction="down",
                bbox=[MARGIN, 60, CANVAS_SIZE[0] - MARGIN, CANVAS_SIZE[1] - MARGIN]  # Content area
            )
            self.transition_graph.add_edge(
                page_id, page_id,
                action="SCROLL:up",
                action_type="SCROLL",
                scroll_direction="up",
                bbox=[MARGIN, 60, CANVAS_SIZE[0] - MARGIN, CANVAS_SIZE[1] - MARGIN]
            )
        
        # === KEY_HOME transitions (from any non-home page) ===
        for page_id in self.pages:
            if page_id != "home":
                self.transition_graph.add_edge(
                    page_id, "home",
                    action="KEY_HOME",
                    action_type="KEY_HOME",
                    bbox=None  # System action, no specific bbox
                )
        
        # === COMPLETE transition ===
        self.transition_graph.add_edge(
            "cart_page", "cart_page",
            action="COMPLETE",
            action_type="COMPLETE",
            bbox=list(self.pages["cart_page"].get_component("complete").bbox)
        )
    
    def reset(self) -> Tuple[Image.Image, dict]:
        """Reset to home screen."""
        self.current_page = "home"
        return self.get_observation()
    
    def get_observation(self) -> Tuple[Image.Image, dict]:
        """Get current observation."""
        page = self.pages[self.current_page]
        img = self.render_engine.render(page)
        return img, page.get_layout()
    
    def step(self, action_type: str, target: str = None, text_input: str = None) -> Tuple[Tuple[Image.Image, dict], float, bool]:
        """
        Execute action.
        
        Args:
            action_type: CLICK, TEXT, SCROLL, KEY_HOME, COMPLETE
            target: Component ID for CLICK, or direction for SCROLL
            text_input: Text for TEXT action
        """
        done = False
        new_page = None
        
        if action_type == "COMPLETE":
            done = True
            return self.get_observation(), 1.0, done
        
        if action_type == "KEY_HOME":
            new_page = "home"
        elif action_type == "SCROLL":
            # SCROLL stays on same page (could update scroll position)
            page = self.pages[self.current_page]
            if target == "down":
                page.scroll_position = min(page.max_scroll, page.scroll_position + 0.25)
            elif target == "up":
                page.scroll_position = max(0, page.scroll_position - 0.25)
            new_page = self.current_page
        else:
            # CLICK or TEXT - find matching transition
            action_key = f"{action_type}:{target or text_input}"
            for u, v, key, data in self.transition_graph.out_edges(self.current_page, keys=True, data=True):
                if data['action'] == action_key:
                    new_page = v
                    break
        
        if new_page:
            self.current_page = new_page
        
        return self.get_observation(), 0.0, done
    
    def get_available_actions(self) -> Dict[str, List]:
        """Get all available actions from current page."""
        actions = {"CLICK": [], "TEXT": [], "SCROLL": [], "KEY_HOME": [], "COMPLETE": []}
        
        for u, v, key, data in self.transition_graph.out_edges(self.current_page, keys=True, data=True):
            action_type = data['action_type']
            
            action_info = {
                "action": data['action'],
                "target_page": v,
                "bbox": data.get('bbox'),
            }
            
            if action_type == "TEXT":
                action_info["text_input"] = data.get('text_input')
            if action_type == "SCROLL":
                action_info["direction"] = data.get('scroll_direction')
            
            actions[action_type].append(action_info)
        
        return actions
    
    def save_environment(self, output_dir: str) -> str:
        """Save environment data."""
        os.makedirs(output_dir, exist_ok=True)
        pages_dir = os.path.join(output_dir, "pages")
        os.makedirs(pages_dir, exist_ok=True)
        
        pages_data = {}
        
        for page_id, page in self.pages.items():
            # Render and save image
            img = self.render_engine.render(page)
            img.save(os.path.join(pages_dir, f"{page_id}.png"))
            
            pages_data[page_id] = {
                "page_type": page.page_type,
                "is_scrollable": page.is_scrollable,
                "layout": {c.component_id: list(c.bbox) for c in page.components},
                "transitions": []
            }
        
        # Add transitions (MultiDiGraph returns key as well)
        for u, v, key, data in self.transition_graph.edges(keys=True, data=True):
            pages_data[u]["transitions"].append({
                "action": data['action'],
                "action_type": data['action_type'],
                "target_page": v,
                "bbox": data.get('bbox'),
                "text_input": data.get('text_input'),
                "scroll_direction": data.get('scroll_direction'),
            })
        
        # Save JSON
        json_path = os.path.join(output_dir, "ui_structure_v1_expanded.json")
        with open(json_path, 'w') as f:
            json.dump({
                "version": "1.1",
                "pages": pages_data,
                "metadata": {
                    "total_pages": len(pages_data),
                    "supported_actions": ["CLICK", "TEXT", "SCROLL", "KEY_HOME", "COMPLETE"],
                    "based_on": "Sequence/2451545728360121.json"
                }
            }, f, indent=2)
        
        return json_path


def generate_sample_trajectory():
    """Generate a sample trajectory similar to ground truth."""
    
    trajectory = [
        {"action_type": "CLICK", "target": "app_chrome", "description": "Open Chrome browser"},
        {"action_type": "TEXT", "text_input": "smart light bulbs", "description": "Search for smart light bulbs"},
        {"action_type": "CLICK", "target": "result_1", "description": "Click on CNET article"},
        {"action_type": "SCROLL", "target": "down", "description": "Scroll down to see products"},
        {"action_type": "SCROLL", "target": "down", "description": "Continue scrolling"},
        {"action_type": "KEY_HOME", "description": "Return to home screen"},
        {"action_type": "CLICK", "target": "app_amazon", "description": "Open Amazon app"},
        {"action_type": "TEXT", "text_input": "Wyze Bulb", "description": "Search for Wyze Bulb"},
        {"action_type": "CLICK", "target": "product_result_1", "description": "Click on product"},
        {"action_type": "SCROLL", "target": "down", "description": "Scroll to see details"},
        {"action_type": "CLICK", "target": "add_to_cart", "description": "Add to cart"},
        {"action_type": "COMPLETE", "description": "Task complete"},
    ]
    
    return trajectory


if __name__ == "__main__":
    import time
    
    output_dir = f"data_engine/ui_environment_v1_expanded/{time.strftime('%Y%m%d_%H%M%S')}"
    
    # Create environment
    env = ExpandedEnvironment()
    
    # Save environment
    json_path = env.save_environment(output_dir)
    print(f"Environment saved to: {json_path}")
    
    # Test trajectory
    print("\n=== Testing Sample Trajectory ===")
    trajectory = generate_sample_trajectory()
    
    obs, layout = env.reset()
    print(f"Start: {env.current_page}")
    
    for i, step in enumerate(trajectory):
        print(f"\nStep {i}: {step['description']}")
        print(f"  Action: {step['action_type']}", end="")
        if step.get('target'):
            print(f", Target: {step['target']}", end="")
        if step.get('text_input'):
            print(f", Text: '{step['text_input']}'", end="")
        print()
        
        obs, reward, done = env.step(
            step['action_type'],
            target=step.get('target'),
            text_input=step.get('text_input')
        )
        print(f"  -> Page: {env.current_page}, Done: {done}")
        
        if done:
            print("\n=== Task Complete! ===")
            break
    
    print(f"\nPages generated: {len(env.pages)}")
    print("Action types: CLICK, TEXT, SCROLL, KEY_HOME, COMPLETE")
