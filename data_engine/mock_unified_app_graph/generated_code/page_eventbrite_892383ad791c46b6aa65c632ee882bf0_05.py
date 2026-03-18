# page_id: page_eventbrite_892383ad791c46b6aa65c632ee882bf0_05
# screenshot: 2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-7.png
# step_index: 5/12
# task: Open Eventbrite. Search for online "Music" events happening next weekend.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Canvas and draw are provided (PIL Image and ImageDraw)
# Fonts: font_sm, font_md, font_lg, font_xl

# Colors
BG_WHITE = (255, 255, 255)
STATUS_BAR = (196, 196, 196)        # light gray status bar
STATUS_DIV = (170, 170, 170)        # darker divider under status bar
CARD_BORDER = (240, 240, 240)       # very light card border
SEPARATOR = (235, 235, 235)         # subtle separators between list items
ACCENT_ORANGE = (216, 78, 26)       # accent color (for selection/stripe)
HEADER_DIV = (245, 245, 245)        # faint header divider

w, h = canvas.size

# Background fill (dominant color)
draw.rectangle([(0, 0), (w, h)], fill=BG_WHITE)

# Status bar area (top)
status_h = 72
draw.rectangle([(0, 0), (w, status_h)], fill=STATUS_BAR)
# Divider line under status bar
draw.line([(0, status_h), (w, status_h)], fill=STATUS_DIV, width=2)

# Header area divider (subtle line to separate toolbar/header)
header_div_y = 170
draw.line([(0, header_div_y), (w, header_div_y)], fill=HEADER_DIV, width=1)

# Section/list item positions (from detected elements)
item_positions = [
    (48, 234, 1344, 144),   # "When do you want to go out?" header area (background grouping)
    (48, 414, 1344, 144),   # "Today"
    (48, 594, 1344, 144),   # "Tomorrow"
    (48, 774, 1344, 144),   # "This Week"
    (48, 954, 1344, 144),   # "This Weekend"
    (48, 1134, 1344, 144),  # "Choose a date..."
]

# Draw subtle rounded card outlines and separators for each section item
for (lx, ly, width_item, height_item) in item_positions:
    rx = lx + width_item
    ry = ly + height_item
    # Slightly inset rounded rectangle to act as a card/background border
    bbox = [lx - 8, ly - 8, rx + 8, ry + 8]
    try:
        # Use rounded rectangle if available
        draw.rounded_rectangle(bbox, radius=18, outline=CARD_BORDER, width=1, fill=None)
    except Exception:
        # Fallback: draw normal rectangle border if rounded not available
        draw.rectangle(bbox, outline=CARD_BORDER, width=1)
    # Separator line directly under the item (clean, subtle)
    draw.line([(lx - 8, ry + 4), (rx + 8, ry + 4)], fill=SEPARATOR, width=1)

# Accent stripe for the selected item ("Anytime" at first listed content item)
# Use a slim vertical stripe at the left of the first content item
first_item = item_positions[0]
fx, fy, fw_item, fh_item = first_item
stripe_x0 = fx
stripe_x1 = fx + 24
stripe_y0 = fy
stripe_y1 = fy + fh_item
draw.rectangle([(stripe_x0, stripe_y0), (stripe_x1, stripe_y1)], fill=ACCENT_ORANGE)

# Large content area background hint (subtle shadow under top area)
# This provides a subtle visual separation between header and main content
shadow_bbox = [32, header_div_y + 6, w - 32, header_div_y + 18]
draw.rectangle(shadow_bbox, fill=(250, 250, 250))

# Final faint full-width separators for visual rhythm down the page
separator_ys = [378, 558, 738, 918, 1278]
for sy in separator_ys:
    draw.line([(32, sy), (w - 32, sy)], fill=SEPARATOR, width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_05_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-7/00_icon_5.23.png
try:
    _c0 = get_crop(0, 60, 62)
    canvas.paste(_c0, (180, 2), _c0)
except Exception:
    pass
layout["5.23"] = [180, 2, 240, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_05_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-7/01_icon_5.23.png
try:
    _c1 = get_crop(1, 56, 62)
    canvas.paste(_c1, (116, 3), _c1)
except Exception:
    pass
layout["5.23"] = [116, 3, 172, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_05_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-7/02_icon_icon_2.png
try:
    _c2 = get_crop(2, 64, 61)
    canvas.paste(_c2, (308, 3), _c2)
except Exception:
    pass
layout["icon_2"] = [308, 3, 372, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_05_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-7/03_icon_5.23.png
try:
    _c3 = get_crop(3, 144, 144)
    canvas.paste(_c3, (12, 72), _c3)
except Exception:
    pass
layout["5.23"] = [12, 72, 156, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_05_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-7/04_icon_Anytime.png
try:
    _c4 = get_crop(4, 1344, 144)
    canvas.paste(_c4, (48, 234), _c4)
except Exception:
    pass
layout["Anytime"] = [48, 234, 1392, 378]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_05_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-7/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 51, 58)
    canvas.paste(_c5, (248, 5), _c5)
except Exception:
    pass
layout["icon_5"] = [248, 5, 299, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_05_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-7/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 100, 61)
    canvas.paste(_c6, (1212, 0), _c6)
except Exception:
    pass
layout["icon_6"] = [1212, 0, 1312, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_05_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-7/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 44, 60)
    canvas.paste(_c7, (1326, 3), _c7)
except Exception:
    pass
layout["icon_7"] = [1326, 3, 1370, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_05_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-7/08_icon_5.23.png
try:
    _c8 = get_crop(8, 91, 61)
    canvas.paste(_c8, (17, 3), _c8)
except Exception:
    pass
layout["5.23"] = [17, 3, 108, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_05_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-7/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 123, 129)
    canvas.paste(_c9, (1291, 246), _c9)
except Exception:
    pass
layout["icon_9"] = [1291, 246, 1414, 375]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_05_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-7/10_icon_Tomorrow.png
try:
    _c10 = get_crop(10, 1344, 144)
    canvas.paste(_c10, (48, 594), _c10)
except Exception:
    pass
layout["Tomorrow"] = [48, 594, 1392, 738]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_05_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-7/11_text_When_do_you_want_to_go_out.png
try:
    _c11 = get_crop(11, 1344, 144)
    canvas.paste(_c11, (48, 234), _c11)
except Exception:
    pass
layout["When_do_you_want_to_go_ou"] = [48, 234, 1392, 378]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_05_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-7/12_text_Today.png
try:
    _c12 = get_crop(12, 1344, 144)
    canvas.paste(_c12, (48, 414), _c12)
except Exception:
    pass
layout["Today"] = [48, 414, 1392, 558]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_05_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-7/13_text_This_Week.png
try:
    _c13 = get_crop(13, 1344, 144)
    canvas.paste(_c13, (48, 774), _c13)
except Exception:
    pass
layout["This_Week"] = [48, 774, 1392, 918]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_05_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-7/14_text_This_Weekend.png
try:
    _c14 = get_crop(14, 1344, 144)
    canvas.paste(_c14, (48, 954), _c14)
except Exception:
    pass
layout["This_Weekend"] = [48, 954, 1392, 1098]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_05_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-7/15_text_Choose_a_date-.png
try:
    _c15 = get_crop(15, 1344, 144)
    canvas.paste(_c15, (48, 1134), _c15)
except Exception:
    pass
layout["Choose_a_date-"] = [48, 1134, 1392, 1278]
