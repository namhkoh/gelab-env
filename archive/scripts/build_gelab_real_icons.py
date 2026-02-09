"""
Build GE-Lab style screens using real extracted icons.
Creates synthetic backgrounds with only essential icons imported.
"""

import os
import json
import time
from PIL import Image, ImageDraw, ImageFont
from typing import List, Dict, Tuple, Optional


# Canvas size (GE-Lab style, but with mobile aspect ratio)
CANVAS_SIZE = (336, 748)

# Colors
BG_DARK = (25, 25, 35)  # Home screen background
BG_WHITE = (255, 255, 255)  # App backgrounds
BG_AMAZON = (255, 255, 255)  # Amazon white
HEADER_AMAZON = (35, 47, 62)  # Amazon dark header
NAV_BAR = (245, 245, 245)  # Bottom nav bar


class IconExtractor:
    """Extract real icons from original screenshots."""
    
    def __init__(self, icons_info_path: str, screenshots_dir: str):
        self.screenshots_dir = screenshots_dir
        self.icons_info = {}
        self.screenshots_cache = {}
        
        with open(icons_info_path, 'r') as f:
            self.icons_info = json.load(f)
    
    def _load_screenshot(self, step_idx: int) -> Optional[Image.Image]:
        """Load and cache original screenshot."""
        if step_idx in self.screenshots_cache:
            return self.screenshots_cache[step_idx]
        
        path = os.path.join(self.screenshots_dir, f"2451545728360121_{step_idx}.png")
        if os.path.exists(path):
            img = Image.open(path)
            self.screenshots_cache[step_idx] = img
            return img
        return None
    
    def extract_icon(self, step_idx: int, icon_id: int) -> Optional[Image.Image]:
        """Extract a specific icon from screenshot."""
        step_key = f"step_{step_idx}"
        if step_key not in self.icons_info:
            return None
        
        icons = self.icons_info[step_key]
        icon_info = None
        for ic in icons:
            if ic['id'] == icon_id:
                icon_info = ic
                break
        
        if icon_info is None:
            return None
        
        screenshot = self._load_screenshot(step_idx)
        if screenshot is None:
            return None
        
        # Get bbox and crop
        bbox_norm = icon_info.get('bbox_norm', [])
        if not bbox_norm:
            return None
        
        w, h = screenshot.size
        x1 = int(bbox_norm[0] * w)
        y1 = int(bbox_norm[1] * h)
        x2 = int(bbox_norm[2] * w)
        y2 = int(bbox_norm[3] * h)
        
        try:
            return screenshot.crop((x1, y1, x2, y2))
        except:
            return None
    
    def find_icon_by_content(self, step_idx: int, content_match: str) -> Optional[Tuple[int, Dict]]:
        """Find icon by content match."""
        step_key = f"step_{step_idx}"
        if step_key not in self.icons_info:
            return None
        
        for icon in self.icons_info[step_key]:
            if content_match.lower() in icon['content'].lower():
                return icon['id'], icon
        return None


class GELabPageBuilder:
    """Build GE-Lab style pages with real icons."""
    
    def __init__(self, extractor: IconExtractor):
        self.extractor = extractor
        
        try:
            self.font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", 11)
            self.font_small = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", 9)
            self.font_title = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf", 12)
        except:
            self.font = self.font_small = self.font_title = None
    
    def _draw_nav_bar(self, img: Image.Image, layout: Dict) -> None:
        """Draw GE-Lab style navigation bar at bottom."""
        draw = ImageDraw.Draw(img)
        w, h = img.size
        
        # Nav bar background
        nav_y = h - 45
        draw.rectangle([0, nav_y, w, h], fill=NAV_BAR)
        
        # Back button
        back_bbox = [15, nav_y + 8, 65, nav_y + 35]
        draw.rounded_rectangle(back_bbox, radius=5, fill=(200, 200, 200), outline=(150, 150, 150))
        draw.text((25, nav_y + 14), "Back", fill=(50, 50, 50), font=self.font_small)
        layout["nav_back"] = {"bbox": back_bbox, "type": "nav", "action": "BACK"}
        
        # Home button
        home_bbox = [w//2 - 30, nav_y + 8, w//2 + 30, nav_y + 35]
        draw.rounded_rectangle(home_bbox, radius=5, fill=(200, 200, 200), outline=(150, 150, 150))
        draw.text((w//2 - 18, nav_y + 14), "Home", fill=(50, 50, 50), font=self.font_small)
        layout["nav_home"] = {"bbox": home_bbox, "type": "nav", "action": "KEY_HOME"}
    
    def create_home_screen(self, step_idx: int = 10) -> Dict:
        """Create home screen with real app icons."""
        img = Image.new("RGB", CANVAS_SIZE, BG_DARK)
        draw = ImageDraw.Draw(img)
        layout = {}
        
        # Essential app icons to extract (in grid order)
        # Note: icon_id 19 is the clock/time display (2:31), not an app - skip it
        essential_apps = [
            (10, "TikTok"),
            (9, "Amazon"),
            (15, "eBay"),
            (16, "AliExpress"),
            (14, "Flipkart"),
            (7, "Alibaba"),
            (8, "Target"),
            (12, "Shop"),
            (11, "Instagram"),
            (17, "Threads"),
            (5, "Edge"),
            (13, "Tripadvisor"),
            (6, "Quora"),
            (3, "Booking"),
            (1, "Agoda"),
            (18, "Airbnb"),
            (4, "Facebook"),
            (0, "Play Books"),
            (2, "OfferUp"),
            (23, "Chrome"),  # Chrome icon from dock
        ]
        
        # Grid layout - improved spacing
        cols, rows = 4, 5
        icon_display_size = 58  # Display size (square) - slightly larger
        margin_x = 15
        margin_top = 45
        
        # Calculate even spacing
        cell_width = (CANVAS_SIZE[0] - 2 * margin_x) // cols
        cell_height = 85  # Compact since icons include labels
        
        placed = 0
        for icon_id, name in essential_apps:
            if placed >= cols * rows:
                break
            
            row = placed // cols
            col = placed % cols
            
            # Center icon in cell
            cell_x = margin_x + col * cell_width
            cell_y = margin_top + row * cell_height
            
            # Extract real icon
            icon_img = self.extractor.extract_icon(step_idx, icon_id)
            
            if icon_img:
                # Maintain aspect ratio while fitting in display size
                orig_w, orig_h = icon_img.size
                aspect = orig_w / orig_h
                
                if aspect > 1:  # Wider than tall
                    new_w = icon_display_size
                    new_h = int(icon_display_size / aspect)
                else:  # Taller than wide or square
                    new_h = icon_display_size
                    new_w = int(icon_display_size * aspect)
                
                # Resize maintaining aspect ratio
                icon_img = icon_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                
                # Center in cell
                x = cell_x + (cell_width - new_w) // 2
                y = cell_y + (icon_display_size - new_h) // 2
                
                # Paste (handle RGBA)
                if icon_img.mode == "RGBA":
                    img.paste(icon_img, (x, y), icon_img)
                else:
                    img.paste(icon_img, (x, y))
                
                # Note: Extracted icons already include app name labels
                # No need to add additional text
                
                # Add to layout (use cell bounds for click area)
                safe_name = name.replace(" ", "_").replace(".", "")
                layout[safe_name] = {
                    "bbox": [cell_x, cell_y, cell_x + cell_width, cell_y + cell_height - 10],
                    "content": name,
                    "type": "app_icon"
                }
                
                placed += 1
        
        # Draw nav bar
        self._draw_nav_bar(img, layout)
        
        return {"page_id": "home", "image": img, "layout": layout, "step_idx": step_idx}
    
    def create_amazon_home(self, step_idx: int = 11) -> Dict:
        """Create Amazon home screen with essential elements."""
        img = Image.new("RGB", CANVAS_SIZE, BG_AMAZON)
        draw = ImageDraw.Draw(img)
        layout = {}
        
        # Amazon header
        draw.rectangle([0, 0, CANVAS_SIZE[0], 55], fill=HEADER_AMAZON)
        
        # Search bar
        search_bbox = [15, 15, CANVAS_SIZE[0] - 50, 45]
        draw.rounded_rectangle(search_bbox, radius=8, fill=(255, 255, 255), outline=(200, 200, 200))
        draw.text((25, 22), "Search Amazon", fill=(100, 100, 100), font=self.font)
        layout["search_bar"] = {"bbox": search_bbox, "type": "search_bar", "action": "CLICK"}
        
        # Cart icon area
        cart_bbox = [CANVAS_SIZE[0] - 45, 15, CANVAS_SIZE[0] - 10, 45]
        # Try to extract cart icon
        cart_icon = self.extractor.extract_icon(step_idx, 18)  # Cart icon
        if cart_icon:
            cart_icon = cart_icon.resize((30, 30), Image.Resampling.LANCZOS)
            img.paste(cart_icon, (CANVAS_SIZE[0] - 42, 18))
        else:
            draw.rounded_rectangle(cart_bbox, radius=5, fill=(50, 60, 75))
            draw.text((CANVAS_SIZE[0] - 38, 22), "Cart", fill=(255, 255, 255), font=self.font_small)
        layout["cart"] = {"bbox": cart_bbox, "type": "icon"}
        
        # Category tabs
        categories = ["Groceries", "Medical Care", "Pharmacy", "In-Store"]
        tab_y = 65
        tab_w = (CANVAS_SIZE[0] - 20) // len(categories)
        for i, cat in enumerate(categories):
            x = 10 + i * tab_w
            draw.text((x, tab_y), cat[:10], fill=(0, 100, 100), font=self.font_small)
        
        # Featured product area (extract from screenshot)
        promo_y = 100
        promo_bbox = [15, promo_y, CANVAS_SIZE[0] - 15, promo_y + 150]
        draw.rounded_rectangle(promo_bbox, radius=10, fill=(248, 248, 255), outline=(230, 230, 240))
        
        # Try to extract product image
        product_img = self.extractor.extract_icon(step_idx, 16)  # Amazon Smart Air Quality
        if product_img:
            # Resize and center
            pw, ph = 200, 120
            product_img = product_img.resize((pw, ph), Image.Resampling.LANCZOS)
            px = (CANVAS_SIZE[0] - pw) // 2
            img.paste(product_img, (px, promo_y + 15))
        
        layout["featured_product"] = {"bbox": promo_bbox, "type": "content"}
        
        # Shopping sections
        section_y = 280
        sections = ["Keep shopping", "Pick up where you left off", "Continue shopping"]
        for i, section in enumerate(sections):
            y = section_y + i * 70
            draw.text((20, y), section, fill=(30, 30, 30), font=self.font_title)
            
            # Section content area
            sec_bbox = [15, y + 20, CANVAS_SIZE[0] - 15, y + 60]
            draw.rounded_rectangle(sec_bbox, radius=5, fill=(250, 250, 250), outline=(230, 230, 230))
            layout[f"section_{i}"] = {"bbox": sec_bbox, "type": "content"}
        
        # Sign in button
        signin_y = 520
        signin_bbox = [30, signin_y, CANVAS_SIZE[0] - 30, signin_y + 40]
        draw.rounded_rectangle(signin_bbox, radius=5, fill=(255, 216, 20), outline=(200, 170, 0))
        draw.text((100, signin_y + 12), "Sign in securely", fill=(0, 0, 0), font=self.font)
        layout["sign_in"] = {"bbox": signin_bbox, "type": "button"}
        
        # Nav bar
        self._draw_nav_bar(img, layout)
        
        return {"page_id": "amazon_home", "image": img, "layout": layout, "step_idx": step_idx}
    
    def create_amazon_search(self, step_idx: int = 13) -> Dict:
        """Create Amazon search input screen."""
        img = Image.new("RGB", CANVAS_SIZE, BG_AMAZON)
        draw = ImageDraw.Draw(img)
        layout = {}
        
        # Header with back arrow and search
        draw.rectangle([0, 0, CANVAS_SIZE[0], 55], fill=HEADER_AMAZON)
        
        # Back arrow
        back_bbox = [10, 15, 40, 45]
        draw.text((15, 22), "<", fill=(255, 255, 255), font=self.font_title)
        layout["back_arrow"] = {"bbox": back_bbox, "type": "nav", "action": "BACK"}
        
        # Search input
        search_bbox = [50, 15, CANVAS_SIZE[0] - 15, 45]
        draw.rounded_rectangle(search_bbox, radius=8, fill=(255, 255, 255))
        draw.text((60, 22), "Search Amazon", fill=(100, 100, 100), font=self.font)
        layout["search_input"] = {"bbox": search_bbox, "type": "text_input"}
        
        # Search suggestions
        suggestions = [
            "wyze bulb",
            "wyze bulb white",
            "wyze bulb color",
            "wyze bulbs 2 pack",
            "wyze bulb camera",
            "wyze bulbs 4 pack",
        ]
        
        sug_y = 70
        for i, sug in enumerate(suggestions):
            y = sug_y + i * 45
            sug_bbox = [15, y, CANVAS_SIZE[0] - 15, y + 40]
            draw.rectangle([15, y, CANVAS_SIZE[0] - 15, y + 40], fill=(255, 255, 255))
            draw.line([15, y + 40, CANVAS_SIZE[0] - 15, y + 40], fill=(230, 230, 230))
            
            # Search icon
            draw.text((25, y + 12), "🔍", font=self.font_small)
            draw.text((50, y + 12), sug, fill=(30, 30, 30), font=self.font)
            
            layout[f"suggestion_{i}"] = {"bbox": sug_bbox, "type": "suggestion", "content": sug}
        
        # Keyboard area (simplified)
        kb_y = 450
        draw.rectangle([0, kb_y, CANVAS_SIZE[0], CANVAS_SIZE[1]], fill=(210, 215, 220))
        
        # Keyboard rows
        kb_rows = [
            "q w e r t y u i o p",
            "a s d f g h j k l",
            "z x c v b n m"
        ]
        for i, row in enumerate(kb_rows):
            y = kb_y + 20 + i * 50
            keys = row.split()
            key_w = CANVAS_SIZE[0] // (len(keys) + 1)
            for j, key in enumerate(keys):
                x = 15 + j * key_w
                key_bbox = [x, y, x + key_w - 5, y + 40]
                draw.rounded_rectangle(key_bbox, radius=5, fill=(255, 255, 255))
                draw.text((x + key_w//3, y + 12), key, fill=(0, 0, 0), font=self.font)
        
        layout["keyboard"] = {"bbox": [0, kb_y, CANVAS_SIZE[0], CANVAS_SIZE[1]], "type": "keyboard"}
        
        return {"page_id": "amazon_search", "image": img, "layout": layout, "step_idx": step_idx}
    
    def create_search_results(self, step_idx: int = 14) -> Dict:
        """Create search results page with product listings."""
        img = Image.new("RGB", CANVAS_SIZE, BG_AMAZON)
        draw = ImageDraw.Draw(img)
        layout = {}
        
        # Header
        draw.rectangle([0, 0, CANVAS_SIZE[0], 55], fill=HEADER_AMAZON)
        
        # Back arrow
        draw.text((15, 22), "<", fill=(255, 255, 255), font=self.font_title)
        layout["back"] = {"bbox": [10, 15, 40, 45], "type": "nav", "action": "BACK"}
        
        # Search bar with query
        search_bbox = [50, 15, CANVAS_SIZE[0] - 15, 45]
        draw.rounded_rectangle(search_bbox, radius=8, fill=(255, 255, 255))
        draw.text((60, 22), "Wyze Bulb", fill=(30, 30, 30), font=self.font)
        layout["search_bar"] = {"bbox": search_bbox, "type": "search_bar"}
        
        # Filter bar
        filter_y = 60
        draw.rectangle([0, filter_y, CANVAS_SIZE[0], filter_y + 35], fill=(245, 245, 245))
        draw.text((15, filter_y + 10), "Filters | Prime | Price", fill=(0, 100, 100), font=self.font_small)
        layout["filters"] = {"bbox": [0, filter_y, CANVAS_SIZE[0], filter_y + 35], "type": "filters"}
        
        # Product listings
        products = [
            ("Wyze Bulb White", "$7.99", "4.5 ★ (12,345)"),
            ("Wyze Bulb Color", "$11.99", "4.3 ★ (8,234)"),
            ("Wyze Bulb 2-Pack", "$14.98", "4.6 ★ (15,678)"),
            ("Wyze Bulb BR30", "$9.99", "4.4 ★ (5,432)"),
        ]
        
        prod_y = 105
        for i, (name, price, rating) in enumerate(products):
            y = prod_y + i * 140
            
            # Product card
            card_bbox = [10, y, CANVAS_SIZE[0] - 10, y + 130]
            draw.rounded_rectangle(card_bbox, radius=8, fill=(255, 255, 255), outline=(230, 230, 230))
            
            # Product image placeholder with extracted icon if available
            img_bbox = [20, y + 10, 100, y + 90]
            draw.rectangle(img_bbox, fill=(248, 248, 255), outline=(230, 230, 240))
            
            # Product info
            draw.text((110, y + 15), name[:25], fill=(0, 0, 0), font=self.font_title)
            draw.text((110, y + 40), price, fill=(177, 39, 4), font=self.font_title)
            draw.text((110, y + 60), rating, fill=(100, 100, 100), font=self.font_small)
            draw.text((110, y + 80), "Prime FREE Delivery", fill=(0, 100, 0), font=self.font_small)
            
            # Prime badge
            draw.rounded_rectangle([110, y + 100, 160, y + 118], radius=3, fill=(0, 120, 180))
            draw.text((115, y + 103), "prime", fill=(255, 255, 255), font=self.font_small)
            
            layout[f"product_{i}"] = {
                "bbox": card_bbox, 
                "type": "product", 
                "content": name,
                "action": "CLICK"
            }
        
        # Nav bar
        self._draw_nav_bar(img, layout)
        
        return {"page_id": "search_results", "image": img, "layout": layout, "step_idx": step_idx}
    
    def create_product_detail(self, step_idx: int = 17) -> Dict:
        """Create product detail page."""
        img = Image.new("RGB", CANVAS_SIZE, BG_AMAZON)
        draw = ImageDraw.Draw(img)
        layout = {}
        
        # Header
        draw.rectangle([0, 0, CANVAS_SIZE[0], 55], fill=HEADER_AMAZON)
        draw.text((15, 22), "<", fill=(255, 255, 255), font=self.font_title)
        layout["back"] = {"bbox": [10, 15, 40, 45], "type": "nav", "action": "BACK"}
        
        # Search bar
        search_bbox = [50, 15, CANVAS_SIZE[0] - 50, 45]
        draw.rounded_rectangle(search_bbox, radius=8, fill=(255, 255, 255))
        draw.text((60, 22), "Wyze Bulb", fill=(100, 100, 100), font=self.font)
        layout["search_bar"] = {"bbox": search_bbox, "type": "search_bar"}
        
        # Product image area
        img_bbox = [20, 65, CANVAS_SIZE[0] - 20, 220]
        draw.rectangle(img_bbox, fill=(250, 250, 255), outline=(230, 230, 240))
        
        # Extract product image if available
        # Draw placeholder light bulb shape
        cx, cy = CANVAS_SIZE[0] // 2, 140
        draw.ellipse([cx - 40, cy - 50, cx + 40, cy + 30], fill=(255, 250, 200), outline=(200, 200, 150))
        draw.rectangle([cx - 15, cy + 25, cx + 15, cy + 55], fill=(200, 200, 200))
        
        layout["product_image"] = {"bbox": img_bbox, "type": "image"}
        
        # Product title
        draw.text((20, 230), "Wyze Bulb White", fill=(0, 0, 0), font=self.font_title)
        draw.text((20, 250), "Smart LED Bulb, WiFi, Dimmable", fill=(80, 80, 80), font=self.font_small)
        
        # Rating
        draw.text((20, 275), "★★★★☆ 4.5", fill=(255, 153, 0), font=self.font)
        draw.text((100, 275), "(12,345 ratings)", fill=(0, 100, 100), font=self.font_small)
        
        # Price
        draw.text((20, 305), "$7.99", fill=(177, 39, 4), font=self.font_title)
        draw.text((80, 308), "($7.99 / Count)", fill=(100, 100, 100), font=self.font_small)
        
        # Prime delivery
        draw.text((20, 335), "FREE delivery", fill=(0, 100, 0), font=self.font)
        draw.text((110, 335), "Tuesday, May 28", fill=(30, 30, 30), font=self.font)
        
        # Delivery/Pickup options
        opt_y = 370
        draw.rounded_rectangle([20, opt_y, 160, opt_y + 35], radius=5, fill=(0, 100, 100), outline=(0, 80, 80))
        draw.text((55, opt_y + 10), "Delivery", fill=(255, 255, 255), font=self.font)
        layout["delivery_option"] = {"bbox": [20, opt_y, 160, opt_y + 35], "type": "option"}
        
        draw.rounded_rectangle([170, opt_y, CANVAS_SIZE[0] - 20, opt_y + 35], radius=5, fill=(255, 255, 255), outline=(200, 200, 200))
        draw.text((215, opt_y + 10), "Pickup", fill=(80, 80, 80), font=self.font)
        layout["pickup_option"] = {"bbox": [170, opt_y, CANVAS_SIZE[0] - 20, opt_y + 35], "type": "option"}
        
        # Quantity
        qty_y = 420
        draw.text((20, qty_y), "Quantity:", fill=(30, 30, 30), font=self.font)
        draw.rounded_rectangle([90, qty_y - 5, 150, qty_y + 25], radius=5, fill=(255, 255, 255), outline=(200, 200, 200))
        draw.text((115, qty_y), "1", fill=(30, 30, 30), font=self.font)
        layout["quantity"] = {"bbox": [90, qty_y - 5, 150, qty_y + 25], "type": "selector"}
        
        # Add to Cart button
        cart_y = 460
        cart_bbox = [20, cart_y, CANVAS_SIZE[0] - 20, cart_y + 45]
        draw.rounded_rectangle(cart_bbox, radius=8, fill=(255, 216, 20), outline=(200, 170, 0))
        draw.text((110, cart_y + 13), "Add to Cart", fill=(0, 0, 0), font=self.font_title)
        layout["add_to_cart"] = {"bbox": cart_bbox, "type": "button", "action": "CLICK"}
        
        # Buy Now button
        buy_y = 515
        buy_bbox = [20, buy_y, CANVAS_SIZE[0] - 20, buy_y + 45]
        draw.rounded_rectangle(buy_bbox, radius=8, fill=(255, 164, 28), outline=(200, 130, 20))
        draw.text((125, buy_y + 13), "Buy Now", fill=(0, 0, 0), font=self.font_title)
        layout["buy_now"] = {"bbox": buy_bbox, "type": "button", "action": "CLICK"}
        
        # Additional info
        info_y = 575
        draw.text((20, info_y), "Ships from: Amazon.com", fill=(80, 80, 80), font=self.font_small)
        draw.text((20, info_y + 20), "Sold by: Amazon.com", fill=(80, 80, 80), font=self.font_small)
        draw.text((20, info_y + 40), "Returns: 30-day returns", fill=(0, 100, 100), font=self.font_small)
        
        # Nav bar
        self._draw_nav_bar(img, layout)
        
        return {"page_id": "product_detail", "image": img, "layout": layout, "step_idx": step_idx}
    
    def create_cart_confirmation(self, step_idx: int = 19) -> Dict:
        """Create cart added confirmation page."""
        img = Image.new("RGB", CANVAS_SIZE, BG_AMAZON)
        draw = ImageDraw.Draw(img)
        layout = {}
        
        # Header
        draw.rectangle([0, 0, CANVAS_SIZE[0], 55], fill=HEADER_AMAZON)
        draw.text((15, 22), "<", fill=(255, 255, 255), font=self.font_title)
        layout["back"] = {"bbox": [10, 15, 40, 45], "type": "nav", "action": "BACK"}
        
        # Search bar
        search_bbox = [50, 15, CANVAS_SIZE[0] - 50, 45]
        draw.rounded_rectangle(search_bbox, radius=8, fill=(255, 255, 255))
        draw.text((60, 22), "Wyze Bulb", fill=(100, 100, 100), font=self.font)
        
        # Success message
        success_y = 70
        success_bbox = [15, success_y, CANVAS_SIZE[0] - 15, success_y + 50]
        draw.rounded_rectangle(success_bbox, radius=8, fill=(220, 255, 220), outline=(150, 200, 150))
        draw.text((25, success_y + 8), "✓ Added to Cart", fill=(0, 120, 0), font=self.font_title)
        draw.text((25, success_y + 28), "Wyze Bulb White", fill=(50, 50, 50), font=self.font)
        layout["success_message"] = {"bbox": success_bbox, "type": "notification"}
        
        # Cart summary
        cart_y = 140
        draw.text((20, cart_y), "Cart subtotal (1 item): $7.99", fill=(30, 30, 30), font=self.font)
        
        # Proceed to Checkout button
        checkout_y = 180
        checkout_bbox = [20, checkout_y, CANVAS_SIZE[0] - 20, checkout_y + 45]
        draw.rounded_rectangle(checkout_bbox, radius=8, fill=(255, 164, 28), outline=(200, 130, 20))
        draw.text((70, checkout_y + 13), "Proceed to Checkout", fill=(0, 0, 0), font=self.font_title)
        layout["checkout"] = {"bbox": checkout_bbox, "type": "button", "action": "CLICK"}
        
        # Continue shopping
        cont_y = 240
        draw.text((90, cont_y), "Continue Shopping", fill=(0, 100, 100), font=self.font)
        layout["continue_shopping"] = {"bbox": [80, cont_y - 5, 250, cont_y + 20], "type": "link"}
        
        # Product in cart
        prod_y = 290
        prod_bbox = [15, prod_y, CANVAS_SIZE[0] - 15, prod_y + 120]
        draw.rounded_rectangle(prod_bbox, radius=8, fill=(255, 255, 255), outline=(230, 230, 230))
        
        # Product image placeholder
        draw.rectangle([25, prod_y + 10, 95, prod_y + 80], fill=(250, 250, 255), outline=(230, 230, 240))
        
        # Product info
        draw.text((105, prod_y + 15), "Wyze Bulb White", fill=(0, 0, 0), font=self.font_title)
        draw.text((105, prod_y + 40), "$7.99", fill=(177, 39, 4), font=self.font)
        draw.text((105, prod_y + 60), "In Stock", fill=(0, 100, 0), font=self.font_small)
        draw.text((105, prod_y + 80), "Qty: 1", fill=(80, 80, 80), font=self.font_small)
        
        layout["cart_item"] = {"bbox": prod_bbox, "type": "cart_item"}
        
        # COMPLETE button (for task completion)
        complete_y = 440
        complete_bbox = [40, complete_y, CANVAS_SIZE[0] - 40, complete_y + 50]
        draw.rounded_rectangle(complete_bbox, radius=8, fill=(76, 175, 80), outline=(60, 140, 64))
        draw.text((95, complete_y + 15), "TASK COMPLETE", fill=(255, 255, 255), font=self.font_title)
        layout["complete"] = {"bbox": complete_bbox, "type": "complete", "action": "COMPLETE"}
        
        # Nav bar
        self._draw_nav_bar(img, layout)
        
        return {"page_id": "cart_confirmation", "image": img, "layout": layout, "step_idx": step_idx}


def build_10_screen_sequence(output_dir: str) -> str:
    """Build 10-screen GE-Lab shopping sequence."""
    
    os.makedirs(output_dir, exist_ok=True)
    pages_dir = os.path.join(output_dir, "pages")
    os.makedirs(pages_dir, exist_ok=True)
    
    # Initialize
    icons_info_path = "data_engine/ui_environment_real/20260202_225737/icons_info.json"
    screenshots_dir = "Sequence/screenshots"
    
    extractor = IconExtractor(icons_info_path, screenshots_dir)
    builder = GELabPageBuilder(extractor)
    
    # Define 6 screens for Amazon shopping sequence (no Chrome)
    # Format: (enum, page_id, builder_func, step_idx)
    screens = [
        (0, "home", builder.create_home_screen, 10),
        (1, "amazon_home", builder.create_amazon_home, 11),
        (2, "amazon_search", builder.create_amazon_search, 14),
        (3, "search_results", builder.create_search_results, 15),
        (4, "product_detail", builder.create_product_detail, 18),
        (5, "cart_confirmation", builder.create_cart_confirmation, 19),
    ]
    
    # No additional Chrome screens needed
    additional_screens = []
    
    pages_data = {}
    
    print(f"Building 10-screen GE-Lab environment...")
    print(f"Canvas size: {CANVAS_SIZE}")
    
    # Build main screens
    for enum, page_id, builder_func, step_idx in screens:
        print(f"  Creating '{enum}_{page_id}' (step {step_idx})...")
        page = builder_func(step_idx)
        
        # Save image with enumeration prefix
        filename = f"{enum}_{page_id}.png"
        img_path = os.path.join(pages_dir, filename)
        page['image'].save(img_path)
        
        pages_data[page_id] = {
            'image': filename,
            'enum': enum,
            'step_idx': step_idx,
            'layout': page['layout'],
            'transitions': []
        }
    
    # Add Chrome/research screens (simplified versions)
    for enum, page_id, step_idx in additional_screens:
        print(f"  Creating '{enum}_{page_id}' (step {step_idx})...")
        
        # Create simple Chrome screens
        img = Image.new("RGB", CANVAS_SIZE, (255, 255, 255))
        draw = ImageDraw.Draw(img)
        layout = {}
        
        try:
            font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", 11)
            font_title = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf", 12)
        except:
            font = font_title = None
        
        # Chrome header
        draw.rectangle([0, 0, CANVAS_SIZE[0], 50], fill=(245, 245, 245))
        
        # URL bar
        url_bbox = [15, 10, CANVAS_SIZE[0] - 50, 40]
        draw.rounded_rectangle(url_bbox, radius=8, fill=(255, 255, 255), outline=(200, 200, 200))
        
        if page_id == "chrome_home":
            draw.text((25, 18), "Search or type URL", fill=(150, 150, 150), font=font)
            layout["url_bar"] = {"bbox": url_bbox, "type": "search_bar", "action": "CLICK"}
        elif page_id == "chrome_search":
            draw.text((25, 18), "smart light bulbs", fill=(30, 30, 30), font=font)
            layout["search_bar"] = {"bbox": url_bbox, "type": "text_input"}
        else:
            draw.text((25, 18), "cnet.com/smart-lights", fill=(30, 30, 30), font=font)
        
        # Content area
        content_y = 60
        if "results" in page_id or "article" in page_id:
            # Search results or article
            draw.text((20, content_y), "Best Smart Lights 2024", fill=(0, 0, 0), font=font_title)
            draw.text((20, content_y + 25), "CNET | Expert Reviews", fill=(100, 100, 100), font=font)
            
            # Article items
            items = [
                "1. Wyze Bulb White - Best Value",
                "2. Philips Hue - Premium Pick",
                "3. LIFX Mini - Best Colors",
            ]
            for i, item in enumerate(items):
                y = content_y + 70 + i * 60
                item_bbox = [15, y, CANVAS_SIZE[0] - 15, y + 50]
                draw.rounded_rectangle(item_bbox, radius=5, fill=(250, 250, 255), outline=(230, 230, 240))
                draw.text((25, y + 15), item, fill=(30, 30, 30), font=font)
                layout[f"item_{i}"] = {"bbox": item_bbox, "type": "content"}
        
        # Nav bar
        nav_y = CANVAS_SIZE[1] - 45
        draw.rectangle([0, nav_y, CANVAS_SIZE[0], CANVAS_SIZE[1]], fill=(245, 245, 245))
        
        back_bbox = [15, nav_y + 8, 65, nav_y + 35]
        draw.rounded_rectangle(back_bbox, radius=5, fill=(200, 200, 200))
        draw.text((25, nav_y + 14), "Back", fill=(50, 50, 50), font=font)
        layout["nav_back"] = {"bbox": back_bbox, "type": "nav", "action": "BACK"}
        
        home_bbox = [CANVAS_SIZE[0]//2 - 30, nav_y + 8, CANVAS_SIZE[0]//2 + 30, nav_y + 35]
        draw.rounded_rectangle(home_bbox, radius=5, fill=(200, 200, 200))
        draw.text((CANVAS_SIZE[0]//2 - 18, nav_y + 14), "Home", fill=(50, 50, 50), font=font)
        layout["nav_home"] = {"bbox": home_bbox, "type": "nav", "action": "KEY_HOME"}
        
        # Save with enumeration prefix
        filename = f"{enum}_{page_id}.png"
        img_path = os.path.join(pages_dir, filename)
        img.save(img_path)
        
        pages_data[page_id] = {
            'image': filename,
            'enum': enum,
            'step_idx': step_idx,
            'layout': layout,
            'transitions': []
        }
    
    # Define transitions for Amazon shopping flow only
    transitions = [
        # Home -> Amazon
        ("home", "CLICK", "Amazon", "amazon_home", None),
        
        # Amazon flow
        ("amazon_home", "CLICK", "search_bar", "amazon_search", None),
        ("amazon_home", "BACK", "nav_back", "home", None),
        ("amazon_home", "KEY_HOME", "nav_home", "home", None),
        
        ("amazon_search", "TEXT", "search_input", "search_results", "wyze bulb"),
        ("amazon_search", "CLICK", "suggestion_0", "search_results", None),
        ("amazon_search", "BACK", "nav_back", "amazon_home", None),
        ("amazon_search", "KEY_HOME", "nav_home", "home", None),
        
        ("search_results", "CLICK", "product_0", "product_detail", None),
        ("search_results", "BACK", "nav_back", "amazon_search", None),
        ("search_results", "KEY_HOME", "nav_home", "home", None),
        
        ("product_detail", "CLICK", "add_to_cart", "cart_confirmation", None),
        ("product_detail", "BACK", "nav_back", "search_results", None),
        ("product_detail", "KEY_HOME", "nav_home", "home", None),
        
        ("cart_confirmation", "COMPLETE", "complete", "cart_confirmation", None),
        ("cart_confirmation", "KEY_HOME", "nav_home", "home", None),
    ]
    
    # Add transitions to pages
    for from_page, action_type, component, to_page, text_input in transitions:
        if from_page not in pages_data:
            continue
        
        bbox = None
        layout = pages_data[from_page]['layout']
        for key, val in layout.items():
            if component.lower() in key.lower():
                bbox = val.get('bbox')
                break
        
        transition = {
            'action': f"{action_type}:{component}",
            'action_type': action_type,
            'target_page': to_page,
            'component_id': component,
            'bbox': bbox
        }
        if text_input:
            transition['text_input'] = text_input
        
        pages_data[from_page]['transitions'].append(transition)
    
    # Save structure
    structure = {
        'version': '2.0',
        'source': 'gelab_real_icons',
        'pages': pages_data,
        'metadata': {
            'total_pages': len(pages_data),
            'canvas_size': list(CANVAS_SIZE),
            'supported_actions': ['CLICK', 'TEXT', 'SCROLL', 'KEY_HOME', 'BACK', 'COMPLETE'],
            'task': {
                'category': 'Web_Shopping',
                'instruction': 'Research smart light bulbs on Chrome, then purchase on Amazon.'
            }
        }
    }
    
    json_path = os.path.join(output_dir, "ui_structure.json")
    with open(json_path, 'w') as f:
        json.dump(structure, f, indent=2)
    
    print(f"\n✓ GE-Lab environment created: {output_dir}")
    print(f"  Pages: {len(pages_data)}")
    print(f"  Structure: {json_path}")
    
    return json_path


def generate_tree_topology(output_dir: str) -> str:
    """Generate a clean tree topology for Amazon shopping flow."""
    from PIL import Image, ImageDraw, ImageFont
    
    # Canvas size - simple vertical flow
    width, height = 500, 600
    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", 11)
        font_bold = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf", 13)
        font_small = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", 10)
        font_title = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf", 16)
    except:
        font = font_bold = font_small = font_title = None
    
    # Colors
    BLUE = (100, 149, 237)      # Home
    ORANGE = (255, 152, 0)      # Amazon
    GREEN = (76, 175, 80)       # Complete
    GRAY = (120, 120, 120)      # Lines
    
    node_w, node_h = 160, 38
    center_x = width // 2
    
    # 6 screens in vertical flow
    nodes = [
        ("0. home", BLUE, 55),
        ("1. amazon_home", ORANGE, 130),
        ("2. amazon_search", ORANGE, 205),
        ("3. search_results", ORANGE, 280),
        ("4. product_detail", ORANGE, 355),
        ("5. cart_confirm", GREEN, 430),
    ]
    
    edge_labels = [
        "CLICK: Amazon",
        "CLICK: search_bar",
        "TEXT: wyze bulb",
        "CLICK: product",
        "CLICK: Add to Cart",
    ]
    
    # Draw edges and labels
    for i in range(len(nodes) - 1):
        y1 = nodes[i][2] + node_h // 2
        y2 = nodes[i + 1][2] - node_h // 2
        
        # Draw line
        draw.line([(center_x, y1), (center_x, y2)], fill=GRAY, width=2)
        
        # Draw arrow head
        draw.polygon([
            (center_x, y2),
            (center_x - 6, y2 - 10),
            (center_x + 6, y2 - 10)
        ], fill=GRAY)
        
        # Draw label to the right
        label = edge_labels[i]
        mid_y = (y1 + y2) // 2
        draw.text((center_x + 15, mid_y - 6), label, fill=(80, 80, 80), font=font_small)
    
    # Draw nodes
    for label, color, y in nodes:
        x = center_x
        bbox = [x - node_w//2, y - node_h//2, x + node_w//2, y + node_h//2]
        draw.rounded_rectangle(bbox, radius=10, fill=color, outline=(50, 50, 50), width=2)
        
        # Draw label (centered)
        text_bbox = draw.textbbox((0, 0), label, font=font_bold)
        text_w = text_bbox[2] - text_bbox[0]
        text_x = x - text_w // 2
        text_y = y - 7
        draw.text((text_x, text_y), label, fill=(255, 255, 255), font=font_bold)
    
    # TASK COMPLETE indicator
    draw.text((center_x - 55, 475), "══════════════════", fill=GREEN, font=font)
    draw.text((center_x - 50, 495), "TASK COMPLETE", fill=GREEN, font=font_bold)
    
    # Title
    draw.text((center_x - 100, 15), "Amazon Shopping Flow", fill=(0, 0, 0), font=font_title)
    
    # Legend
    legend_y = height - 35
    legend_items = [(100, BLUE, "Home"), (200, ORANGE, "Amazon"), (320, GREEN, "Complete")]
    for lx, color, text in legend_items:
        draw.rounded_rectangle([lx, legend_y, lx + 18, legend_y + 18], radius=3, fill=color, outline=(50, 50, 50))
        draw.text((lx + 22, legend_y + 2), text, fill=(0, 0, 0), font=font)
    
    # Save
    tree_path = os.path.join(output_dir, "tree_topology.png")
    img.save(tree_path)
    
    # Text version
    text_tree = """
Amazon Shopping Flow
====================

    ┌─────────────────────┐
    │      0. home        │  Home screen with app icons
    └──────────┬──────────┘
               │
               │ CLICK: Amazon
               ▼
    ┌─────────────────────┐
    │   1. amazon_home    │  Search bar, categories
    └──────────┬──────────┘
               │
               │ CLICK: search_bar
               ▼
    ┌─────────────────────┐
    │   2. amazon_search  │  Wyze bulb suggestions
    └──────────┬──────────┘
               │
               │ TEXT: wyze bulb
               ▼
    ┌─────────────────────┐
    │  3. search_results  │  Product listings
    └──────────┬──────────┘
               │
               │ CLICK: product
               ▼
    ┌─────────────────────┐
    │  4. product_detail  │  Add to Cart, Buy Now
    └──────────┬──────────┘
               │
               │ CLICK: Add to Cart
               ▼
    ┌─────────────────────┐
    │   5. cart_confirm   │  "Added to cart"
    │  ═══════════════════│
    │    TASK COMPLETE    │
    └─────────────────────┘

Navigation: [Back] → previous | [Home] → page 0
"""
    
    text_path = os.path.join(output_dir, "tree_topology.txt")
    with open(text_path, 'w') as f:
        f.write(text_tree)
    
    print(f"  Tree topology saved: {tree_path}")
    print(f"  Text topology saved: {text_path}")
    
    return tree_path


if __name__ == "__main__":
    output_dir = f"data_engine/ui_environment_gelab_real/{time.strftime('%Y%m%d_%H%M%S')}"
    build_10_screen_sequence(output_dir)
    generate_tree_topology(output_dir)
