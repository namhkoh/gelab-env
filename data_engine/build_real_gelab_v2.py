"""
Build a realistic GE-Lab environment using real extracted icons.
Maintains original aspect ratio and uses actual UI components from the Amazon shopping sequence.
"""

import os
import json
import time
from PIL import Image, ImageDraw, ImageFont
from typing import List, Dict, Tuple, Optional
import glob


# Original Pixel 8 Pro resolution
ORIGINAL_SIZE = (1344, 2992)
# Scaled canvas size maintaining aspect ratio (~0.449)
CANVAS_SIZE = (336, 748)  # Maintains 1344:2992 ratio scaled down
SCALE_FACTOR = CANVAS_SIZE[0] / ORIGINAL_SIZE[0]


class RealAssetsManager:
    """Manage real extracted UI assets from the shopping sequence."""
    
    def __init__(self, base_dir: str, sequence_dir: str, icons_info_path: str):
        self.base_dir = base_dir
        self.sequence_dir = sequence_dir
        self.icons_info = {}
        self.screenshots = {}
        
        # Load icons info
        if os.path.exists(icons_info_path):
            with open(icons_info_path, 'r') as f:
                self.icons_info = json.load(f)
        
        # Load original screenshots
        self._load_screenshots()
    
    def _load_screenshots(self):
        """Load original full-resolution screenshots."""
        screenshot_files = glob.glob(os.path.join(self.sequence_dir, "*.png"))
        for path in screenshot_files:
            filename = os.path.basename(path)
            # Extract step index from filename like "2451545728360121_10.png"
            parts = filename.replace(".png", "").split("_")
            if len(parts) >= 2:
                try:
                    step_idx = int(parts[-1])
                    self.screenshots[step_idx] = path
                except ValueError:
                    continue
    
    def get_screenshot(self, step_idx: int) -> Optional[Image.Image]:
        """Get original screenshot for a step."""
        if step_idx in self.screenshots:
            return Image.open(self.screenshots[step_idx])
        return None
    
    def extract_icon(self, step_key: str, icon_content: str) -> Optional[Image.Image]:
        """Extract an icon from the original screenshot based on icons_info."""
        if step_key not in self.icons_info:
            return None
        
        icons = self.icons_info[step_key]
        for icon in icons:
            if icon_content.lower() in icon['content'].lower():
                # Get step index
                step_idx = int(step_key.split("_")[1])
                screenshot = self.get_screenshot(step_idx)
                if screenshot is None:
                    continue
                
                # Get bbox and crop
                bbox = icon['bbox_pixel']
                # bbox is [x1, y1, x2, y2] but in original resolution
                # Need to convert from normalized to pixel
                img_w, img_h = screenshot.size
                
                # Use normalized bbox if pixel bbox seems off
                bbox_norm = icon.get('bbox_norm', [])
                if bbox_norm:
                    x1 = int(bbox_norm[0] * img_w)
                    y1 = int(bbox_norm[1] * img_h)
                    x2 = int(bbox_norm[2] * img_w)
                    y2 = int(bbox_norm[3] * img_h)
                else:
                    x1, y1, x2, y2 = bbox
                
                try:
                    return screenshot.crop((x1, y1, x2, y2))
                except Exception as e:
                    print(f"Error extracting {icon_content}: {e}")
                    return None
        
        return None
    
    def get_icons_for_step(self, step_idx: int) -> List[Dict]:
        """Get all icons info for a specific step."""
        step_key = f"step_{step_idx}"
        return self.icons_info.get(step_key, [])


class RealGELabBuilder:
    """Build GE-Lab pages using real UI assets."""
    
    def __init__(self, assets_manager: RealAssetsManager):
        self.assets = assets_manager
        self.pages = {}
        
    def create_page_from_screenshot(self, step_idx: int, page_id: str,
                                    add_nav_buttons: bool = True) -> Dict:
        """Create a GE-Lab page from a real screenshot with navigation buttons."""
        screenshot = self.assets.get_screenshot(step_idx)
        if screenshot is None:
            raise ValueError(f"Screenshot not found for step {step_idx}")
        
        # Resize to canvas size maintaining aspect ratio
        img = screenshot.resize(CANVAS_SIZE, Image.Resampling.LANCZOS)
        
        # Get icons info for this step
        icons = self.assets.get_icons_for_step(step_idx)
        
        # Build layout with scaled bboxes
        layout = {}
        for icon in icons:
            content = icon.get('content', '')
            bbox_norm = icon.get('bbox_norm', [])
            
            if bbox_norm:
                # Scale to canvas size
                x1 = int(bbox_norm[0] * CANVAS_SIZE[0])
                y1 = int(bbox_norm[1] * CANVAS_SIZE[1])
                x2 = int(bbox_norm[2] * CANVAS_SIZE[0])
                y2 = int(bbox_norm[3] * CANVAS_SIZE[1])
                
                # Create safe key
                key = "".join(c if c.isalnum() else '_' for c in content)[:25]
                if not key:
                    key = f"icon_{icon['id']}"
                
                layout[key] = {
                    'bbox': [x1, y1, x2, y2],
                    'content': content,
                    'interactivity': icon.get('interactivity', True),
                    'type': 'extracted_icon'
                }
        
        # Add navigation buttons overlay
        if add_nav_buttons:
            draw = ImageDraw.Draw(img)
            
            try:
                font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", 10)
            except:
                font = None
            
            # Home button (bottom center, like Android nav)
            home_w, home_h = 50, 25
            home_x = (CANVAS_SIZE[0] - home_w) // 2
            home_y = CANVAS_SIZE[1] - 35
            home_bbox = [home_x, home_y, home_x + home_w, home_y + home_h]
            
            draw.rounded_rectangle(home_bbox, radius=5, fill=(60, 60, 60, 200), outline=(100, 100, 100))
            draw.text((home_x + 10, home_y + 6), "Home", fill=(255, 255, 255), font=font)
            layout["nav_home"] = {
                'bbox': home_bbox,
                'type': 'nav_button',
                'action': 'KEY_HOME'
            }
            
            # Back button (bottom left)
            back_w, back_h = 40, 25
            back_x = 10
            back_y = CANVAS_SIZE[1] - 35
            back_bbox = [back_x, back_y, back_x + back_w, back_y + back_h]
            
            draw.rounded_rectangle(back_bbox, radius=5, fill=(60, 60, 60, 200), outline=(100, 100, 100))
            draw.text((back_x + 8, back_y + 6), "Back", fill=(255, 255, 255), font=font)
            layout["nav_back"] = {
                'bbox': back_bbox,
                'type': 'nav_button',
                'action': 'BACK'
            }
        
        return {
            'page_id': page_id,
            'step_idx': step_idx,
            'image': img,
            'layout': layout
        }


def build_shopping_sequence(output_dir: str) -> str:
    """Build a shopping sequence GE-Lab environment from real UI assets."""
    
    os.makedirs(output_dir, exist_ok=True)
    pages_dir = os.path.join(output_dir, "pages")
    os.makedirs(pages_dir, exist_ok=True)
    
    # Initialize assets manager
    base_dir = "data_engine/ui_environment_real/20260202_225737"
    sequence_dir = "Sequence/screenshots"
    icons_info_path = os.path.join(base_dir, "icons_info.json")
    
    assets = RealAssetsManager(base_dir, sequence_dir, icons_info_path)
    builder = RealGELabBuilder(assets)
    
    # Define the shopping sequence steps we want to use
    # Step mapping: step_idx -> (page_id, description)
    sequence_steps = [
        (10, "home", "Home screen with app icons"),
        (11, "amazon_home", "Amazon home page"),
        (12, "amazon_search", "Amazon search with keyboard"),
        (14, "search_results", "Wyze bulb search results"),
        (18, "product_page", "Product detail with Add to Cart"),
        (19, "cart_added", "Cart confirmation"),
    ]
    
    pages_data = {}
    
    print(f"Building GE-Lab environment with {len(sequence_steps)} pages...")
    print(f"Canvas size: {CANVAS_SIZE} (aspect ratio: {CANVAS_SIZE[0]/CANVAS_SIZE[1]:.3f})")
    
    for step_idx, page_id, description in sequence_steps:
        print(f"  Creating page '{page_id}' from step {step_idx}...")
        
        try:
            page = builder.create_page_from_screenshot(step_idx, page_id)
            
            # Save page image
            img_path = os.path.join(pages_dir, f"{page_id}.png")
            page['image'].save(img_path)
            
            # Store page data
            pages_data[page_id] = {
                'image': f"{page_id}.png",
                'step_idx': step_idx,
                'description': description,
                'layout': page['layout'],
                'transitions': []
            }
            
        except Exception as e:
            print(f"    Error: {e}")
            continue
    
    # Define transitions for the shopping flow
    transitions = [
        # From home
        ("home", "CLICK", "Amazon_Sho__", "amazon_home", None),
        ("home", "KEY_HOME", "nav_home", "home", None),
        
        # From Amazon home
        ("amazon_home", "CLICK", "search_bar", "amazon_search", None),
        ("amazon_home", "KEY_HOME", "nav_home", "home", None),
        ("amazon_home", "BACK", "nav_back", "home", None),
        
        # From Amazon search
        ("amazon_search", "TEXT", "search_input", "search_results", "wyze bulb"),
        ("amazon_search", "KEY_HOME", "nav_home", "home", None),
        ("amazon_search", "BACK", "nav_back", "amazon_home", None),
        
        # From search results
        ("search_results", "CLICK", "wyze_bulb_color_", "product_page", None),
        ("search_results", "KEY_HOME", "nav_home", "home", None),
        ("search_results", "BACK", "nav_back", "amazon_search", None),
        
        # From product page  
        ("product_page", "CLICK", "Add_to_Cart_", "cart_added", None),
        ("product_page", "KEY_HOME", "nav_home", "home", None),
        ("product_page", "BACK", "nav_back", "search_results", None),
        
        # From cart - task complete
        ("cart_added", "COMPLETE", "complete", "cart_added", None),
        ("cart_added", "KEY_HOME", "nav_home", "home", None),
    ]
    
    # Add transitions to pages
    for from_page, action_type, component, to_page, text_input in transitions:
        if from_page not in pages_data:
            continue
        
        # Find bbox for component
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
    
    # Build final structure
    structure = {
        'version': '2.0',
        'source': 'real_ui_shopping_sequence',
        'sequence_id': '2451545728360121',
        'pages': pages_data,
        'metadata': {
            'total_pages': len(pages_data),
            'canvas_size': list(CANVAS_SIZE),
            'original_size': list(ORIGINAL_SIZE),
            'scale_factor': SCALE_FACTOR,
            'supported_actions': ['CLICK', 'TEXT', 'SCROLL', 'KEY_HOME', 'BACK', 'COMPLETE'],
            'task': {
                'category': 'Web_Shopping',
                'instruction': 'Find a highly recommended smart light bulb and purchase it on Amazon.'
            }
        }
    }
    
    # Save structure
    json_path = os.path.join(output_dir, "ui_structure.json")
    with open(json_path, 'w') as f:
        json.dump(structure, f, indent=2)
    
    print(f"\n✓ GE-Lab environment created: {output_dir}")
    print(f"  Pages: {len(pages_data)}")
    print(f"  Structure: {json_path}")
    
    return json_path


if __name__ == "__main__":
    output_dir = f"data_engine/ui_environment_real_gelab_v2/{time.strftime('%Y%m%d_%H%M%S')}"
    build_shopping_sequence(output_dir)
