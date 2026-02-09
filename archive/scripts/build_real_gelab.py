"""
Build a GE-Lab style environment using real extracted icons.
Creates a shopping sequence with realistic app icons and UI components.
"""

import os
import json
import random
from PIL import Image, ImageDraw, ImageFont
from typing import List, Dict, Tuple, Optional
import glob


# Canvas settings (matching original GE-Lab)
CANVAS_SIZE = (448, 448)
ICON_SIZE = (60, 60)
MARGIN = 15
TOP_MARGIN = 40


class RealIconLoader:
    """Load and manage real extracted icons."""
    
    def __init__(self, assets_base_dir: str):
        self.assets_base_dir = assets_base_dir
        self.icons_by_step = {}
        self.icons_by_name = {}
        self._load_icons()
    
    def _load_icons(self):
        """Load all extracted icons."""
        step_dirs = glob.glob(os.path.join(self.assets_base_dir, "step_*"))
        
        for step_dir in step_dirs:
            step_name = os.path.basename(step_dir)
            step_idx = int(step_name.split("_")[1])
            self.icons_by_step[step_idx] = []
            
            # Load non-resized icons
            icon_files = [f for f in glob.glob(os.path.join(step_dir, "*.png"))
                         if "resized" not in f]
            
            for icon_path in icon_files:
                filename = os.path.basename(icon_path)
                # Extract name from filename like "09_Amazon_Sho__.png"
                name = "_".join(filename.split("_")[1:]).replace(".png", "")
                
                icon_info = {
                    "path": icon_path,
                    "name": name,
                    "step": step_idx,
                    "image": None  # Lazy load
                }
                
                self.icons_by_step[step_idx].append(icon_info)
                
                # Also index by name for easy lookup
                clean_name = name.lower().replace("_", "").replace(" ", "")
                self.icons_by_name[clean_name] = icon_info
    
    def get_icon_image(self, icon_info: Dict, size: Tuple[int, int] = ICON_SIZE) -> Image.Image:
        """Load and resize icon image."""
        if icon_info.get("image") is None:
            img = Image.open(icon_info["path"]).convert("RGBA")
            icon_info["image"] = img
        
        return icon_info["image"].resize(size, Image.Resampling.LANCZOS)
    
    def find_icon(self, name: str) -> Optional[Dict]:
        """Find icon by name (fuzzy match)."""
        clean_name = name.lower().replace("_", "").replace(" ", "")
        
        # Exact match
        if clean_name in self.icons_by_name:
            return self.icons_by_name[clean_name]
        
        # Partial match
        for key, icon in self.icons_by_name.items():
            if clean_name in key or key in clean_name:
                return icon
        
        return None
    
    def get_app_icons(self) -> List[Dict]:
        """Get app icons from home screen (step 0)."""
        # Filter to get only app icons (not system UI)
        app_names = ["tiktok", "amazon", "ebay", "chrome", "instagram", 
                     "facebook", "alibaba", "target", "flipkart", "airbnb",
                     "tripadvisor", "quora", "booking", "agoda", "offerup",
                     "edge", "playbooks", "aliexpress", "shop", "threads"]
        
        icons = []
        for name in app_names:
            icon = self.find_icon(name)
            if icon:
                icons.append(icon)
        
        return icons


class RealGELabPageBuilder:
    """Build GE-Lab style pages using real icons."""
    
    def __init__(self, icon_loader: RealIconLoader):
        self.icon_loader = icon_loader
        self.pages = {}
        self.transitions = []
    
    def create_home_page(self, page_id: str, app_icons: List[Dict], 
                         rows: int = 4, cols: int = 4) -> Dict:
        """Create a home screen page with app icons in a grid."""
        img = Image.new("RGB", CANVAS_SIZE, (30, 30, 40))  # Dark background
        draw = ImageDraw.Draw(img)
        
        layout = {}
        
        # Calculate grid positions
        icon_w, icon_h = ICON_SIZE
        spacing_x = (CANVAS_SIZE[0] - 2 * MARGIN) // cols
        spacing_y = (CANVAS_SIZE[1] - TOP_MARGIN - MARGIN) // rows
        
        # Place icons in grid
        icon_idx = 0
        for row in range(rows):
            for col in range(cols):
                if icon_idx >= len(app_icons):
                    break
                
                icon_info = app_icons[icon_idx]
                
                x = MARGIN + col * spacing_x + (spacing_x - icon_w) // 2
                y = TOP_MARGIN + row * spacing_y + (spacing_y - icon_h) // 2
                
                # Load and paste icon
                icon_img = self.icon_loader.get_icon_image(icon_info, ICON_SIZE)
                
                # Handle RGBA
                if icon_img.mode == "RGBA":
                    img.paste(icon_img, (x, y), icon_img)
                else:
                    img.paste(icon_img, (x, y))
                
                # Store layout
                clean_name = icon_info["name"].replace(" ", "_")[:15]
                layout[clean_name] = {
                    "bbox": [x, y, x + icon_w, y + icon_h],
                    "type": "app_icon",
                    "name": icon_info["name"]
                }
                
                icon_idx += 1
        
        # Add status bar area
        draw.rectangle([0, 0, CANVAS_SIZE[0], 25], fill=(20, 20, 25))
        
        return {
            "page_id": page_id,
            "page_type": "home",
            "image": img,
            "layout": layout
        }
    
    def create_app_page(self, page_id: str, app_name: str,
                        has_search: bool = True,
                        content_items: List[Dict] = None) -> Dict:
        """Create an app page with search bar and content."""
        img = Image.new("RGB", CANVAS_SIZE, (255, 255, 255))
        draw = ImageDraw.Draw(img)
        
        layout = {}
        
        try:
            font = ImageFont.truetype("font/helvetica.ttf", 14)
            small_font = ImageFont.truetype("font/helvetica.ttf", 11)
        except:
            font = small_font = None
        
        # Header bar
        draw.rectangle([0, 0, CANVAS_SIZE[0], 50], fill=(245, 245, 245))
        
        # Back button
        back_bbox = [MARGIN, 10, MARGIN + 50, 40]
        draw.rounded_rectangle(back_bbox, radius=5, fill=(255, 200, 200), outline=(200, 150, 150))
        draw.text((MARGIN + 10, 15), "Back", fill=(0, 0, 0), font=small_font)
        layout["back"] = {"bbox": back_bbox, "type": "nav_button"}
        
        # Home button
        home_bbox = [CANVAS_SIZE[0] - MARGIN - 50, 10, CANVAS_SIZE[0] - MARGIN, 40]
        draw.rounded_rectangle(home_bbox, radius=5, fill=(200, 255, 200), outline=(150, 200, 150))
        draw.text((CANVAS_SIZE[0] - MARGIN - 40, 15), "Home", fill=(0, 0, 0), font=small_font)
        layout["home"] = {"bbox": home_bbox, "type": "nav_button"}
        
        # App title
        draw.text((CANVAS_SIZE[0] // 2 - 30, 15), app_name, fill=(50, 50, 50), font=font)
        
        # Search bar
        if has_search:
            search_bbox = [MARGIN, 60, CANVAS_SIZE[0] - MARGIN, 95]
            draw.rounded_rectangle(search_bbox, radius=8, fill=(240, 240, 240), outline=(100, 100, 100), width=2)
            draw.text((MARGIN + 40, 70), f"Search {app_name}...", fill=(150, 150, 150), font=font)
            layout["search_bar"] = {"bbox": search_bbox, "type": "search_bar"}
        
        # Content area with items
        if content_items:
            y_offset = 110 if has_search else 60
            for i, item in enumerate(content_items):
                item_bbox = [MARGIN, y_offset, CANVAS_SIZE[0] - MARGIN, y_offset + 60]
                
                # Draw content card
                color = item.get("color", (250, 250, 255))
                draw.rounded_rectangle(item_bbox, radius=8, fill=color, outline=(220, 220, 220))
                
                # Draw item content
                title = item.get("title", f"Item {i}")[:30]
                subtitle = item.get("subtitle", "")[:40]
                draw.text((MARGIN + 10, y_offset + 10), title, fill=(30, 30, 30), font=font)
                if subtitle:
                    draw.text((MARGIN + 10, y_offset + 30), subtitle, fill=(100, 100, 100), font=small_font)
                
                item_id = item.get("id", f"item_{i}")
                layout[item_id] = {"bbox": item_bbox, "type": "content_item", "title": title}
                
                y_offset += 70
        
        return {
            "page_id": page_id,
            "page_type": "app",
            "app_name": app_name,
            "image": img,
            "layout": layout
        }
    
    def create_product_page(self, page_id: str, product_name: str,
                           product_icon: Optional[Dict] = None) -> Dict:
        """Create a product detail page."""
        img = Image.new("RGB", CANVAS_SIZE, (255, 255, 255))
        draw = ImageDraw.Draw(img)
        
        layout = {}
        
        try:
            font = ImageFont.truetype("font/helvetica.ttf", 14)
            title_font = ImageFont.truetype("font/helvetica.ttf", 16)
            small_font = ImageFont.truetype("font/helvetica.ttf", 11)
        except:
            font = title_font = small_font = None
        
        # Header
        draw.rectangle([0, 0, CANVAS_SIZE[0], 50], fill=(35, 47, 62))
        
        # Back button
        back_bbox = [MARGIN, 10, MARGIN + 50, 40]
        draw.rounded_rectangle(back_bbox, radius=5, fill=(255, 200, 200))
        draw.text((MARGIN + 10, 15), "Back", fill=(0, 0, 0), font=small_font)
        layout["back"] = {"bbox": back_bbox, "type": "nav_button"}
        
        # Product image area
        product_bbox = [MARGIN, 60, CANVAS_SIZE[0] - MARGIN, 200]
        draw.rectangle(product_bbox, fill=(250, 250, 250), outline=(230, 230, 230))
        
        if product_icon:
            prod_img = self.icon_loader.get_icon_image(product_icon, (120, 120))
            paste_x = (CANVAS_SIZE[0] - 120) // 2
            if prod_img.mode == "RGBA":
                img.paste(prod_img, (paste_x, 90), prod_img)
            else:
                img.paste(prod_img, (paste_x, 90))
        
        layout["product_image"] = {"bbox": product_bbox, "type": "product_image"}
        
        # Product title
        draw.text((MARGIN, 210), product_name[:35], fill=(0, 0, 0), font=title_font)
        
        # Price
        draw.text((MARGIN, 235), "$7.99", fill=(177, 39, 4), font=title_font)
        draw.text((MARGIN + 60, 238), "Prime FREE Delivery", fill=(0, 100, 0), font=small_font)
        
        # Add to Cart button
        cart_bbox = [MARGIN, 280, CANVAS_SIZE[0] // 2 - 10, 320]
        draw.rounded_rectangle(cart_bbox, radius=5, fill=(255, 216, 20), outline=(200, 170, 0))
        draw.text((MARGIN + 20, 292), "Add to Cart", fill=(0, 0, 0), font=font)
        layout["add_to_cart"] = {"bbox": cart_bbox, "type": "action_button"}
        
        # Buy Now button
        buy_bbox = [CANVAS_SIZE[0] // 2 + 10, 280, CANVAS_SIZE[0] - MARGIN, 320]
        draw.rounded_rectangle(buy_bbox, radius=5, fill=(255, 164, 28), outline=(200, 130, 20))
        draw.text((CANVAS_SIZE[0] // 2 + 40, 292), "Buy Now", fill=(0, 0, 0), font=font)
        layout["buy_now"] = {"bbox": buy_bbox, "type": "action_button"}
        
        # Product details
        details_bbox = [MARGIN, 340, CANVAS_SIZE[0] - MARGIN, 430]
        draw.rounded_rectangle(details_bbox, radius=5, fill=(248, 248, 255), outline=(230, 230, 240))
        draw.text((MARGIN + 10, 350), "Product Details", fill=(0, 0, 0), font=font)
        draw.text((MARGIN + 10, 375), "• Works with Alexa & Google", fill=(80, 80, 80), font=small_font)
        draw.text((MARGIN + 10, 395), "• 800 Lumens, Dimmable", fill=(80, 80, 80), font=small_font)
        draw.text((MARGIN + 10, 415), "• 2700K Soft White", fill=(80, 80, 80), font=small_font)
        layout["product_details"] = {"bbox": details_bbox, "type": "content"}
        
        return {
            "page_id": page_id,
            "page_type": "product",
            "image": img,
            "layout": layout
        }
    
    def create_cart_page(self, page_id: str, product_name: str) -> Dict:
        """Create a cart/confirmation page."""
        img = Image.new("RGB", CANVAS_SIZE, (255, 255, 255))
        draw = ImageDraw.Draw(img)
        
        layout = {}
        
        try:
            font = ImageFont.truetype("font/helvetica.ttf", 14)
            title_font = ImageFont.truetype("font/helvetica.ttf", 16)
            small_font = ImageFont.truetype("font/helvetica.ttf", 11)
        except:
            font = title_font = small_font = None
        
        # Header
        draw.rectangle([0, 0, CANVAS_SIZE[0], 50], fill=(35, 47, 62))
        draw.text((CANVAS_SIZE[0] // 2 - 30, 15), "Cart", fill=(255, 255, 255), font=title_font)
        
        # Back button
        back_bbox = [MARGIN, 10, MARGIN + 50, 40]
        draw.rounded_rectangle(back_bbox, radius=5, fill=(255, 200, 200))
        draw.text((MARGIN + 10, 15), "Back", fill=(0, 0, 0), font=small_font)
        layout["back"] = {"bbox": back_bbox, "type": "nav_button"}
        
        # Success message
        draw.rounded_rectangle([MARGIN, 70, CANVAS_SIZE[0] - MARGIN, 130], 
                              radius=8, fill=(220, 255, 220), outline=(150, 200, 150))
        draw.text((MARGIN + 20, 85), "Added to Cart", fill=(0, 100, 0), font=title_font)
        draw.text((MARGIN + 20, 110), product_name[:30], fill=(50, 50, 50), font=font)
        
        # Cart item
        item_bbox = [MARGIN, 150, CANVAS_SIZE[0] - MARGIN, 230]
        draw.rounded_rectangle(item_bbox, radius=8, fill=(255, 255, 255), outline=(220, 220, 220))
        draw.text((MARGIN + 10, 160), product_name[:25], fill=(0, 0, 0), font=font)
        draw.text((MARGIN + 10, 185), "$7.99", fill=(177, 39, 4), font=font)
        draw.text((MARGIN + 10, 205), "Qty: 1", fill=(100, 100, 100), font=small_font)
        layout["cart_item"] = {"bbox": item_bbox, "type": "cart_item"}
        
        # Checkout button
        checkout_bbox = [MARGIN + 30, 260, CANVAS_SIZE[0] - MARGIN - 30, 305]
        draw.rounded_rectangle(checkout_bbox, radius=5, fill=(255, 164, 28), outline=(200, 130, 20))
        draw.text((CANVAS_SIZE[0] // 2 - 60, 275), "Proceed to Checkout", fill=(0, 0, 0), font=font)
        layout["checkout"] = {"bbox": checkout_bbox, "type": "action_button"}
        
        # Complete button (for task completion)
        complete_bbox = [MARGIN + 30, 330, CANVAS_SIZE[0] - MARGIN - 30, 375]
        draw.rounded_rectangle(complete_bbox, radius=5, fill=(76, 175, 80), outline=(60, 140, 64))
        draw.text((CANVAS_SIZE[0] // 2 - 50, 345), "TASK COMPLETE", fill=(255, 255, 255), font=font)
        layout["complete"] = {"bbox": complete_bbox, "type": "complete_button"}
        
        return {
            "page_id": page_id,
            "page_type": "cart",
            "image": img,
            "layout": layout
        }


def build_shopping_environment(assets_dir: str, output_dir: str) -> str:
    """Build a complete shopping sequence environment."""
    
    os.makedirs(output_dir, exist_ok=True)
    pages_dir = os.path.join(output_dir, "pages")
    os.makedirs(pages_dir, exist_ok=True)
    
    # Load real icons
    print("Loading extracted icons...")
    loader = RealIconLoader(assets_dir)
    builder = RealGELabPageBuilder(loader)
    
    # Get app icons for home screen
    app_icons = loader.get_app_icons()
    print(f"Found {len(app_icons)} app icons")
    
    pages_data = {}
    
    # 1. Create Home Page
    print("Creating pages...")
    home_page = builder.create_home_page("home", app_icons)
    home_page["image"].save(os.path.join(pages_dir, "home.png"))
    pages_data["home"] = {
        "image": "home.png",
        "page_type": home_page["page_type"],
        "layout": {k: v for k, v in home_page["layout"].items()},
        "transitions": []
    }
    
    # 2. Create Chrome App Page
    chrome_page = builder.create_app_page("chrome", "Chrome", has_search=True, content_items=[
        {"id": "shortcut_1", "title": "Google", "subtitle": "google.com", "color": (250, 250, 255)},
        {"id": "shortcut_2", "title": "YouTube", "subtitle": "youtube.com", "color": (255, 250, 250)},
    ])
    chrome_page["image"].save(os.path.join(pages_dir, "chrome.png"))
    pages_data["chrome"] = {
        "image": "chrome.png",
        "page_type": chrome_page["page_type"],
        "layout": {k: v for k, v in chrome_page["layout"].items()},
        "transitions": []
    }
    
    # 3. Create Chrome Search Results
    search_results = builder.create_app_page("search_results", "Search Results", has_search=False, content_items=[
        {"id": "result_1", "title": "Best Smart Lights 2024", "subtitle": "CNET - Expert Reviews", "color": (240, 248, 255)},
        {"id": "result_2", "title": "Top 10 Smart Bulbs", "subtitle": "Tom's Guide", "color": (255, 248, 240)},
        {"id": "result_3", "title": "Wyze Bulb Review", "subtitle": "The Verge", "color": (248, 255, 240)},
    ])
    search_results["image"].save(os.path.join(pages_dir, "search_results.png"))
    pages_data["search_results"] = {
        "image": "search_results.png",
        "page_type": "search_results",
        "layout": {k: v for k, v in search_results["layout"].items()},
        "transitions": []
    }
    
    # 4. Create Amazon App Page
    amazon_page = builder.create_app_page("amazon", "Amazon", has_search=True, content_items=[
        {"id": "deal_1", "title": "Today's Deals", "subtitle": "Up to 50% off", "color": (255, 250, 240)},
        {"id": "deal_2", "title": "Prime Day Deals", "subtitle": "Exclusive savings", "color": (240, 250, 255)},
    ])
    amazon_page["image"].save(os.path.join(pages_dir, "amazon.png"))
    pages_data["amazon"] = {
        "image": "amazon.png",
        "page_type": amazon_page["page_type"],
        "layout": {k: v for k, v in amazon_page["layout"].items()},
        "transitions": []
    }
    
    # 5. Create Amazon Search Results
    amazon_search = builder.create_app_page("amazon_search", "Wyze Bulb", has_search=False, content_items=[
        {"id": "product_1", "title": "Wyze Bulb White", "subtitle": "$7.99 - 4.5 stars", "color": (255, 252, 245)},
        {"id": "product_2", "title": "Wyze Bulb Color", "subtitle": "$11.99 - 4.3 stars", "color": (255, 252, 245)},
        {"id": "product_3", "title": "Wyze Bulb 4-Pack", "subtitle": "$26.98 - 4.6 stars", "color": (255, 252, 245)},
    ])
    amazon_search["image"].save(os.path.join(pages_dir, "amazon_search.png"))
    pages_data["amazon_search"] = {
        "image": "amazon_search.png",
        "page_type": "search_results",
        "layout": {k: v for k, v in amazon_search["layout"].items()},
        "transitions": []
    }
    
    # 6. Create Product Page
    product_page = builder.create_product_page("product", "Wyze Bulb Smart LED")
    product_page["image"].save(os.path.join(pages_dir, "product.png"))
    pages_data["product"] = {
        "image": "product.png",
        "page_type": product_page["page_type"],
        "layout": {k: v for k, v in product_page["layout"].items()},
        "transitions": []
    }
    
    # 7. Create Cart Page
    cart_page = builder.create_cart_page("cart", "Wyze Bulb Smart LED")
    cart_page["image"].save(os.path.join(pages_dir, "cart.png"))
    pages_data["cart"] = {
        "image": "cart.png",
        "page_type": cart_page["page_type"],
        "layout": {k: v for k, v in cart_page["layout"].items()},
        "transitions": []
    }
    
    # Build transitions
    transitions = [
        # From home
        ("home", "chrome", "Chrome", "CLICK"),
        ("home", "amazon", "Amazon_Sho", "CLICK"),
        
        # Chrome flow
        ("chrome", "home", "home", "CLICK"),
        ("chrome", "search_results", "search_bar", "TEXT", "smart light bulbs"),
        
        # Search results
        ("search_results", "chrome", "back", "CLICK"),
        ("search_results", "home", "home", "CLICK"),
        
        # Amazon flow
        ("amazon", "home", "home", "CLICK"),
        ("amazon", "amazon_search", "search_bar", "TEXT", "Wyze Bulb"),
        
        # Amazon search results
        ("amazon_search", "amazon", "back", "CLICK"),
        ("amazon_search", "home", "home", "CLICK"),
        ("amazon_search", "product", "product_1", "CLICK"),
        
        # Product page
        ("product", "amazon_search", "back", "CLICK"),
        ("product", "cart", "add_to_cart", "CLICK"),
        
        # Cart page
        ("cart", "product", "back", "CLICK"),
    ]
    
    for t in transitions:
        from_page, to_page, component, action_type = t[0], t[1], t[2], t[3]
        text_input = t[4] if len(t) > 4 else None
        
        # Get bbox from layout
        layout = pages_data[from_page]["layout"]
        bbox = None
        for key, val in layout.items():
            if component.lower() in key.lower():
                bbox = val.get("bbox")
                break
        
        transition = {
            "action": f"{action_type}:{component}",
            "action_type": action_type,
            "target_page": to_page,
            "component_id": component,
            "bbox": bbox
        }
        
        if text_input:
            transition["text_input"] = text_input
        
        pages_data[from_page]["transitions"].append(transition)
    
    # Add COMPLETE transition
    pages_data["cart"]["transitions"].append({
        "action": "COMPLETE",
        "action_type": "COMPLETE",
        "target_page": "cart",
        "bbox": pages_data["cart"]["layout"]["complete"]["bbox"]
    })
    
    # Save structure
    structure = {
        "version": "2.0",
        "source": "real_icons_gelab",
        "pages": pages_data,
        "metadata": {
            "total_pages": len(pages_data),
            "supported_actions": ["CLICK", "TEXT", "SCROLL", "KEY_HOME", "COMPLETE"],
            "canvas_size": list(CANVAS_SIZE)
        }
    }
    
    json_path = os.path.join(output_dir, "ui_structure.json")
    with open(json_path, "w") as f:
        json.dump(structure, f, indent=2)
    
    print(f"\nEnvironment created: {output_dir}")
    print(f"  Pages: {len(pages_data)}")
    print(f"  Structure: {json_path}")
    
    return json_path


if __name__ == "__main__":
    import time
    
    assets_dir = "data_engine/ui_environment_real/20260202_225737/assets"
    output_dir = f"data_engine/ui_environment_real_gelab/{time.strftime('%Y%m%d_%H%M%S')}"
    
    build_shopping_environment(assets_dir, output_dir)
