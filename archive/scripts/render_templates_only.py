"""
Render the hardcoded Ge-LAB page templates WITHOUT extracted icons.
Shows the pure structural skeleton of each page in the sequence.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

from PIL import Image, ImageDraw, ImageFont
from build_gelab_real_icons import GELabPageBuilder, CANVAS_SIZE


class DummyExtractor:
    """Returns None for all icon extractions, showing only the template."""
    def extract_icon(self, step_idx, icon_id):
        return None
    def find_icon_by_content(self, step_idx, content_match):
        return None


def render_templates(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    builder = GELabPageBuilder(DummyExtractor())

    screens = [
        (0, "home", builder.create_home_screen, 10),
        (1, "amazon_home", builder.create_amazon_home, 11),
        (2, "amazon_search", builder.create_amazon_search, 14),
        (3, "search_results", builder.create_search_results, 15),
        (4, "product_detail", builder.create_product_detail, 18),
        (5, "cart_confirmation", builder.create_cart_confirmation, 19),
    ]

    paths = []
    for enum, page_id, builder_func, step_idx in screens:
        page = builder_func(step_idx)
        filename = f"{enum}_{page_id}_template.png"
        path = os.path.join(output_dir, filename)
        page['image'].save(path)
        paths.append(path)
        print(f"  Saved: {filename}  ({CANVAS_SIZE[0]}x{CANVAS_SIZE[1]})")

    # Also create a single horizontal strip showing the full sequence
    images = [Image.open(p) for p in paths]
    gap = 10
    strip_w = sum(im.width for im in images) + gap * (len(images) - 1)
    strip_h = images[0].height
    strip = Image.new("RGB", (strip_w, strip_h), (240, 240, 240))

    x = 0
    for im in images:
        strip.paste(im, (x, 0))
        x += im.width + gap

    strip_path = os.path.join(output_dir, "sequence_strip.png")
    strip.save(strip_path)
    print(f"\n  Sequence strip: {strip_path}  ({strip_w}x{strip_h})")


if __name__ == "__main__":
    out = "data_engine/ui_environment_gelab_real/templates_only"
    render_templates(out)
    print("\nDone.")
