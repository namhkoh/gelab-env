"""
Generate two demo variations of the shopping sequence to demonstrate
automated perturbation of the UI environment pipeline.

Variation 1 (color): Theme color changes across all pages.
Variation 2 (layout): Icon removal + grid shift on home screen.
"""

import os
import sys
import json
from PIL import Image, ImageDraw, ImageFont
from typing import Dict, Optional

sys.path.insert(0, os.path.dirname(__file__))
from build_gelab_real_icons import (
    IconExtractor, GELabPageBuilder, CANVAS_SIZE,
    BG_DARK, BG_AMAZON, HEADER_AMAZON, NAV_BAR,
)


class ColorVariationBuilder(GELabPageBuilder):
    """Variation 1: Change color theme across all pages."""

    BG_DARK_V = (15, 25, 55)
    HEADER_V = (0, 100, 100)
    NAV_BAR_V = (220, 230, 245)
    CART_BTN_V = (76, 175, 80)
    BUY_BTN_V = (56, 142, 60)

    def _draw_nav_bar(self, img, layout):
        draw = ImageDraw.Draw(img)
        w, h = img.size
        nav_y = h - 45
        draw.rectangle([0, nav_y, w, h], fill=self.NAV_BAR_V)
        back_bbox = [15, nav_y + 8, 65, nav_y + 35]
        draw.rounded_rectangle(back_bbox, radius=5, fill=(180, 190, 210), outline=(130, 140, 160))
        draw.text((25, nav_y + 14), "Back", fill=(30, 30, 50), font=self.font_small)
        layout["nav_back"] = {"bbox": back_bbox, "type": "nav", "action": "BACK"}
        home_bbox = [w // 2 - 30, nav_y + 8, w // 2 + 30, nav_y + 35]
        draw.rounded_rectangle(home_bbox, radius=5, fill=(180, 190, 210), outline=(130, 140, 160))
        draw.text((w // 2 - 18, nav_y + 14), "Home", fill=(30, 30, 50), font=self.font_small)
        layout["nav_home"] = {"bbox": home_bbox, "type": "nav", "action": "KEY_HOME"}

    def create_home_screen(self, step_idx=10):
        result = super().create_home_screen(step_idx)
        old_img = result['image']
        img = Image.new("RGB", CANVAS_SIZE, self.BG_DARK_V)
        # Re-paste icons onto new background
        draw = ImageDraw.Draw(img)
        layout = {}
        essential_apps = [
            (10, "TikTok"), (9, "Amazon"), (15, "eBay"), (16, "AliExpress"),
            (14, "Flipkart"), (7, "Alibaba"), (8, "Target"), (12, "Shop"),
            (11, "Instagram"), (17, "Threads"), (5, "Edge"), (13, "Tripadvisor"),
            (6, "Quora"), (3, "Booking"), (1, "Agoda"), (18, "Airbnb"),
            (4, "Facebook"), (0, "Play Books"), (2, "OfferUp"), (23, "Chrome"),
        ]
        cols, rows = 4, 5
        icon_display_size = 58
        margin_x, margin_top = 15, 45
        cell_width = (CANVAS_SIZE[0] - 2 * margin_x) // cols
        cell_height = 85
        placed = 0
        for icon_id, name in essential_apps:
            if placed >= cols * rows:
                break
            row, col = placed // cols, placed % cols
            cell_x = margin_x + col * cell_width
            cell_y = margin_top + row * cell_height
            icon_img = self.extractor.extract_icon(step_idx, icon_id)
            if icon_img:
                ow, oh = icon_img.size
                aspect = ow / oh
                new_w = icon_display_size if aspect > 1 else int(icon_display_size * aspect)
                new_h = int(icon_display_size / aspect) if aspect > 1 else icon_display_size
                icon_img = icon_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                x = cell_x + (cell_width - new_w) // 2
                y = cell_y + (icon_display_size - new_h) // 2
                if icon_img.mode == "RGBA":
                    img.paste(icon_img, (x, y), icon_img)
                else:
                    img.paste(icon_img, (x, y))
                safe_name = name.replace(" ", "_").replace(".", "")
                layout[safe_name] = {
                    "bbox": [cell_x, cell_y, cell_x + cell_width, cell_y + cell_height - 10],
                    "content": name, "type": "app_icon"
                }
                placed += 1
        self._draw_nav_bar(img, layout)
        result['image'] = img
        result['layout'] = layout
        return result

    def create_amazon_home(self, step_idx=11):
        img = Image.new("RGB", CANVAS_SIZE, BG_AMAZON)
        draw = ImageDraw.Draw(img)
        layout = {}
        draw.rectangle([0, 0, CANVAS_SIZE[0], 55], fill=self.HEADER_V)
        search_bbox = [15, 15, CANVAS_SIZE[0] - 50, 45]
        draw.rounded_rectangle(search_bbox, radius=8, fill=(255, 255, 255), outline=(200, 200, 200))
        draw.text((25, 22), "Search Amazon", fill=(100, 100, 100), font=self.font)
        layout["search_bar"] = {"bbox": search_bbox, "type": "search_bar", "action": "CLICK"}
        cart_bbox = [CANVAS_SIZE[0] - 45, 15, CANVAS_SIZE[0] - 10, 45]
        cart_icon = self.extractor.extract_icon(step_idx, 18)
        if cart_icon:
            cart_icon = cart_icon.resize((30, 30), Image.Resampling.LANCZOS)
            img.paste(cart_icon, (CANVAS_SIZE[0] - 42, 18))
        else:
            draw.rounded_rectangle(cart_bbox, radius=5, fill=(0, 80, 80))
            draw.text((CANVAS_SIZE[0] - 38, 22), "Cart", fill=(255, 255, 255), font=self.font_small)
        layout["cart"] = {"bbox": cart_bbox, "type": "icon"}
        categories = ["Groceries", "Medical Care", "Pharmacy", "In-Store"]
        tab_y = 65
        tab_w = (CANVAS_SIZE[0] - 20) // len(categories)
        for i, cat in enumerate(categories):
            draw.text((10 + i * tab_w, tab_y), cat[:10], fill=(0, 100, 100), font=self.font_small)
        promo_y = 100
        promo_bbox = [15, promo_y, CANVAS_SIZE[0] - 15, promo_y + 150]
        draw.rounded_rectangle(promo_bbox, radius=10, fill=(248, 248, 255), outline=(230, 230, 240))
        product_img = self.extractor.extract_icon(step_idx, 16)
        if product_img:
            product_img = product_img.resize((200, 120), Image.Resampling.LANCZOS)
            img.paste(product_img, ((CANVAS_SIZE[0] - 200) // 2, promo_y + 15))
        layout["featured_product"] = {"bbox": promo_bbox, "type": "content"}
        section_y = 280
        for i, section in enumerate(["Keep shopping", "Pick up where you left off", "Continue shopping"]):
            y = section_y + i * 70
            draw.text((20, y), section, fill=(30, 30, 30), font=self.font_title)
            sec_bbox = [15, y + 20, CANVAS_SIZE[0] - 15, y + 60]
            draw.rounded_rectangle(sec_bbox, radius=5, fill=(250, 250, 250), outline=(230, 230, 230))
            layout[f"section_{i}"] = {"bbox": sec_bbox, "type": "content"}
        signin_y = 520
        signin_bbox = [30, signin_y, CANVAS_SIZE[0] - 30, signin_y + 40]
        draw.rounded_rectangle(signin_bbox, radius=5, fill=(0, 150, 150), outline=(0, 120, 120))
        draw.text((100, signin_y + 12), "Sign in securely", fill=(255, 255, 255), font=self.font)
        layout["sign_in"] = {"bbox": signin_bbox, "type": "button"}
        self._draw_nav_bar(img, layout)
        return {"page_id": "amazon_home", "image": img, "layout": layout, "step_idx": step_idx}

    def create_product_detail(self, step_idx=17):
        result = super().create_product_detail(step_idx)
        img = result['image']
        draw = ImageDraw.Draw(img)
        layout = result['layout']
        # Redraw header with new color
        draw.rectangle([0, 0, CANVAS_SIZE[0], 55], fill=self.HEADER_V)
        draw.text((15, 22), "<", fill=(255, 255, 255), font=self.font_title)
        search_bbox = [50, 15, CANVAS_SIZE[0] - 50, 45]
        draw.rounded_rectangle(search_bbox, radius=8, fill=(255, 255, 255))
        draw.text((60, 22), "Wyze Bulb", fill=(100, 100, 100), font=self.font)
        # Redraw buttons with new colors
        cart_y = 460
        cart_bbox = [20, cart_y, CANVAS_SIZE[0] - 20, cart_y + 45]
        draw.rounded_rectangle(cart_bbox, radius=8, fill=self.CART_BTN_V, outline=(60, 140, 64))
        draw.text((110, cart_y + 13), "Add to Cart", fill=(255, 255, 255), font=self.font_title)
        layout["add_to_cart"] = {"bbox": cart_bbox, "type": "button", "action": "CLICK"}
        buy_y = 515
        buy_bbox = [20, buy_y, CANVAS_SIZE[0] - 20, buy_y + 45]
        draw.rounded_rectangle(buy_bbox, radius=8, fill=self.BUY_BTN_V, outline=(40, 110, 44))
        draw.text((125, buy_y + 13), "Buy Now", fill=(255, 255, 255), font=self.font_title)
        layout["buy_now"] = {"bbox": buy_bbox, "type": "button", "action": "CLICK"}
        return result

    def create_cart_confirmation(self, step_idx=19):
        result = super().create_cart_confirmation(step_idx)
        img = result['image']
        draw = ImageDraw.Draw(img)
        # Redraw header
        draw.rectangle([0, 0, CANVAS_SIZE[0], 55], fill=self.HEADER_V)
        draw.text((15, 22), "<", fill=(255, 255, 255), font=self.font_title)
        search_bbox = [50, 15, CANVAS_SIZE[0] - 50, 45]
        draw.rounded_rectangle(search_bbox, radius=8, fill=(255, 255, 255))
        draw.text((60, 22), "Wyze Bulb", fill=(100, 100, 100), font=self.font)
        # Redraw checkout button
        checkout_y = 180
        checkout_bbox = [20, checkout_y, CANVAS_SIZE[0] - 20, checkout_y + 45]
        draw.rounded_rectangle(checkout_bbox, radius=8, fill=(0, 150, 150), outline=(0, 120, 120))
        draw.text((70, checkout_y + 13), "Proceed to Checkout", fill=(255, 255, 255), font=self.font_title)
        return result


class LayoutVariationBuilder(GELabPageBuilder):
    """Variation 2: Remove 5 icons, shift to 3x5 grid, larger icons."""

    def create_home_screen(self, step_idx=10):
        img = Image.new("RGB", CANVAS_SIZE, BG_DARK)
        draw = ImageDraw.Draw(img)
        layout = {}
        # 15 icons (removed eBay, AliExpress, Flipkart, Quora, Play Books)
        essential_apps = [
            (10, "TikTok"), (9, "Amazon"), (7, "Alibaba"),
            (8, "Target"), (12, "Shop"), (11, "Instagram"),
            (17, "Threads"), (5, "Edge"), (13, "Tripadvisor"),
            (3, "Booking"), (1, "Agoda"), (18, "Airbnb"),
            (4, "Facebook"), (2, "OfferUp"), (23, "Chrome"),
        ]
        cols, rows = 3, 5
        icon_display_size = 68
        margin_x = 20
        margin_top = 55
        cell_width = (CANVAS_SIZE[0] - 2 * margin_x) // cols
        cell_height = 90
        placed = 0
        for icon_id, name in essential_apps:
            if placed >= cols * rows:
                break
            row, col = placed // cols, placed % cols
            cell_x = margin_x + col * cell_width
            cell_y = margin_top + row * cell_height
            icon_img = self.extractor.extract_icon(step_idx, icon_id)
            if icon_img:
                ow, oh = icon_img.size
                aspect = ow / oh
                new_w = icon_display_size if aspect > 1 else int(icon_display_size * aspect)
                new_h = int(icon_display_size / aspect) if aspect > 1 else icon_display_size
                icon_img = icon_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                x = cell_x + (cell_width - new_w) // 2
                y = cell_y + (icon_display_size - new_h) // 2
                if icon_img.mode == "RGBA":
                    img.paste(icon_img, (x, y), icon_img)
                else:
                    img.paste(icon_img, (x, y))
                safe_name = name.replace(" ", "_").replace(".", "")
                layout[safe_name] = {
                    "bbox": [cell_x, cell_y, cell_x + cell_width, cell_y + cell_height - 10],
                    "content": name, "type": "app_icon"
                }
                placed += 1
        self._draw_nav_bar(img, layout)
        return {"page_id": "home", "image": img, "layout": layout, "step_idx": step_idx}


def build_variation(builder, output_dir, screens):
    os.makedirs(output_dir, exist_ok=True)
    pages_dir = os.path.join(output_dir, "pages")
    os.makedirs(pages_dir, exist_ok=True)
    pages_data = {}
    for enum, page_id, builder_func, step_idx in screens:
        page = builder_func(step_idx)
        filename = f"{enum}_{page_id}.png"
        page['image'].save(os.path.join(pages_dir, filename))
        pages_data[page_id] = {
            'image': filename, 'enum': enum, 'step_idx': step_idx,
            'layout': page['layout'], 'transitions': []
        }
    # Same transitions as original
    transitions = [
        ("home", "CLICK", "Amazon", "amazon_home", None),
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
    for from_page, action_type, component, to_page, text_input in transitions:
        if from_page not in pages_data:
            continue
        bbox = None
        for key, val in pages_data[from_page]['layout'].items():
            if component.lower() in key.lower():
                bbox = val.get('bbox')
                break
        t = {'action': f"{action_type}:{component}", 'action_type': action_type,
             'target_page': to_page, 'component_id': component, 'bbox': bbox}
        if text_input:
            t['text_input'] = text_input
        pages_data[from_page]['transitions'].append(t)
    structure = {
        'version': '2.0', 'source': 'gelab_real_icons_variation',
        'pages': pages_data,
        'metadata': {'total_pages': len(pages_data), 'canvas_size': list(CANVAS_SIZE),
                     'supported_actions': ['CLICK', 'TEXT', 'SCROLL', 'KEY_HOME', 'BACK', 'COMPLETE'],
                     'task': {'category': 'Web_Shopping',
                              'instruction': 'Research smart light bulbs on Chrome, then purchase on Amazon.'}}
    }
    with open(os.path.join(output_dir, "ui_structure.json"), 'w') as f:
        json.dump(structure, f, indent=2)
    print(f"  Created {len(pages_data)} pages in {output_dir}")


if __name__ == "__main__":
    icons_info_path = "data_engine/ui_environment_real/20260202_225737/icons_info.json"
    screenshots_dir = "Sequence/screenshots"
    extractor = IconExtractor(icons_info_path, screenshots_dir)
    base_dir = "data_engine/ui_environment_gelab_real"

    # Variation 1: Color theme
    print("Variation 1: Color theme change")
    color_builder = ColorVariationBuilder(extractor)
    screens_color = [
        (0, "home", color_builder.create_home_screen, 10),
        (1, "amazon_home", color_builder.create_amazon_home, 11),
        (2, "amazon_search", color_builder.create_amazon_search, 13),
        (3, "search_results", color_builder.create_search_results, 14),
        (4, "product_detail", color_builder.create_product_detail, 17),
        (5, "cart_confirmation", color_builder.create_cart_confirmation, 19),
    ]
    build_variation(color_builder, os.path.join(base_dir, "variation_color"), screens_color)

    # Variation 2: Layout shift
    print("Variation 2: Icon removal + layout shift")
    layout_builder = LayoutVariationBuilder(extractor)
    screens_layout = [
        (0, "home", layout_builder.create_home_screen, 10),
        (1, "amazon_home", layout_builder.create_amazon_home, 11),
        (2, "amazon_search", layout_builder.create_amazon_search, 13),
        (3, "search_results", layout_builder.create_search_results, 14),
        (4, "product_detail", layout_builder.create_product_detail, 17),
        (5, "cart_confirmation", layout_builder.create_cart_confirmation, 19),
    ]
    build_variation(layout_builder, os.path.join(base_dir, "variation_layout"), screens_layout)

    print("\nDone. Compare:")
    print(f"  Original: {base_dir}/20260202_233723/pages/")
    print(f"  Color:    {base_dir}/variation_color/pages/")
    print(f"  Layout:   {base_dir}/variation_layout/pages/")
