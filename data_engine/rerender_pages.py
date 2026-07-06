"""Re-render GE-Lab environment pages from a saved ui_structure.json.

The original datas/pages/ renders were lost (gitignored, generated on another
machine). tree.py's seed-42 regeneration does not reproduce the committed
environment because the original icon pool differed, so instead we re-render
each page deterministically from the committed structure: icon func names map
directly to files (Business_171 -> icons/Business/PNG/171.png) and every bbox
is recorded in the layout. Reuses tree.py's own classes so the icon processing
chain (NEAREST resize to h=200, then LANCZOS fit to 50x50) and text rendering
are identical to the original pipeline.

Run from data_engine/ (font/helvetica.ttf and icons/ are cwd-relative):
    python rerender_pages.py --structure ../datas/ui_structure.json --output ../datas/pages
"""
import argparse
import json
import os
from io import BytesIO

from PIL import Image

from tree import UIElement, UIPage, RenderEngine, UIManager


def load_normal_icon(func_name: str, icons_dir: str = "icons") -> UIElement:
    category, _, filename = func_name.partition("_")
    path = os.path.join(icons_dir, category, "PNG", f"{filename}.png")
    # Replicate load_icons_from_directory: bytes -> RGBA -> NEAREST resize to height 200
    with open(path, "rb") as f:
        img = Image.open(BytesIO(f.read())).convert("RGBA")
    aspect_ratio = img.width / img.height
    new_height = 200
    new_width = int(aspect_ratio * new_height)
    img = img.resize((new_width, new_height), Image.Resampling.NEAREST)
    return UIElement(img, func_name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--structure", default="../datas/ui_structure.json")
    parser.add_argument("--output", default="../datas/pages")
    parser.add_argument("--icons_dir", default="icons")
    args = parser.parse_args()

    with open(args.structure, "r", encoding="utf-8") as f:
        pages = json.load(f)["pages"]

    os.makedirs(args.output, exist_ok=True)

    # back/home come from UIManager's procedural drawing; an empty manager
    # still builds sys_elements exactly as the original run did.
    sys_elements = UIManager([], []).sys_elements

    # page_title bbox is deterministic (LayoutGenerator.generate): y1 stays at
    # MARGIN because 'back' (or the default) pins it to 20.
    title_bbox = (112, 20, 336, 70)

    renderer = RenderEngine()
    icon_cache = {}
    for page_id in sorted(pages, key=lambda p: int(p.rsplit("_", 1)[1])):
        info = pages[page_id]
        elements, layout = [], {"page_title": title_bbox}
        for func_name, meta in info["layout"].items():
            layout[func_name] = tuple(meta["bbox"])
            if meta["type"] == "system":
                elements.append(sys_elements[func_name])
            else:
                if func_name not in icon_cache:
                    icon_cache[func_name] = load_normal_icon(func_name, args.icons_dir)
                elements.append(icon_cache[func_name])
        page = UIPage(page_id, elements, layout)
        renderer.render(page).save(os.path.join(args.output, f"{page_id}.png"))

    print(f"Rendered {len(pages)} pages to {args.output}")


if __name__ == "__main__":
    main()
