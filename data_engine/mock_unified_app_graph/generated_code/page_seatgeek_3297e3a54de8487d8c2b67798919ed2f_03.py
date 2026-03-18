# page_id: page_seatgeek_3297e3a54de8487d8c2b67798919ed2f_03
# screenshot: 2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-6.png
# step_index: 3/11
# task: Open SeatGeek. Search "Comedy Show in Los Angeles". Find the top recommendation. When is the event?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw background and structural UI elements for the provided canvas.
# Available variables:
# - canvas: PIL Image (1440x2960 RGB)
# - draw: PIL ImageDraw object
# - font_sm, font_md, font_lg, font_xl

w, h = canvas.size

# Colors (subtle, matching screenshot)
BG = (255, 255, 255)               # main white background
STATUS_BG = (245, 245, 245)       # light grey status bar
SEARCH_BG = (249, 249, 249)       # search field background
SEARCH_BORDER = (230, 230, 230)   # search border / input outline
DIVIDER = (235, 235, 235)         # section dividers
SUBTLE_DIV = (245, 245, 245)      # very subtle divider

# Fill overall background (canvas starts white but ensure uniformity)
draw.rectangle([(0, 0), (w, h)], fill=BG)

# Status bar area at the top (~120px to match screenshot proportions)
status_h = 120
draw.rectangle([(0, 0), (w, status_h)], fill=STATUS_BG)

# Thin bottom hairline of status bar
draw.line([(0, status_h - 1), (w, status_h - 1)], fill=DIVIDER, width=1)

# Search bar (rounded) near the top
search_left = 32
search_right = w - 32
search_top = 56
search_bottom = 200
search_radius = 36

draw.rounded_rectangle(
    [(search_left, search_top), (search_right, search_bottom)],
    radius=search_radius,
    fill=SEARCH_BG,
    outline=SEARCH_BORDER,
    width=1
)

# Subtle drop-shadow/hairline under search field
draw.line([(search_left + 8, search_bottom + 6), (search_right - 8, search_bottom + 6)], fill=DIVIDER, width=1)

# Divider below the search area (full width subtle)
divider_y = search_bottom + 14
draw.line([(24, divider_y), (w - 24, divider_y)], fill=DIVIDER, width=1)

# "Use my current location" row boundaries (visual separators only)
use_row_top = divider_y + 10
use_row_bottom = use_row_top + 190  # matches screenshot row height approx
# Top separator for the row
draw.line([(24, use_row_top), (w - 24, use_row_top)], fill=SUBTLE_DIV, width=1)
# Bottom separator for the row
draw.line([(24, use_row_bottom), (w - 24, use_row_bottom)], fill=DIVIDER, width=1)

# Divider above Recent locations header
recent_div_y = use_row_bottom + 28
draw.line([(24, recent_div_y), (w - 24, recent_div_y)], fill=DIVIDER, width=1)

# Recent locations header area (no text drawn, just spacing and subtle separator beneath)
recent_header_y = recent_div_y + 20
draw.line([(24, recent_header_y + 64), (w - 24, recent_header_y + 64)], fill=SUBTLE_DIV, width=1)

# Rows for recent location items - draw separators between items
first_item_top = 657
first_item_bottom = first_item_top + 168
second_item_top = first_item_bottom
second_item_bottom = second_item_top + 168

# Top border for the first item (subtle)
draw.line([(24, first_item_top), (w - 24, first_item_top)], fill=SUBTLE_DIV, width=1)
# Divider between first and second item
draw.line([(24, first_item_bottom), (w - 24, first_item_bottom)], fill=DIVIDER, width=1)
# Bottom border for the second item
draw.line([(24, second_item_bottom), (w - 24, second_item_bottom)], fill=SUBTLE_DIV, width=1)

# Final subtle footer divider near bottom (visual balance)
draw.line([(24, h - 48), (w - 24, h - 48)], fill=SUBTLE_DIV, width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_03_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-6/00_icon_icon_0.png
try:
    _c0 = get_crop(0, 51, 68)
    canvas.paste(_c0, (1153, 0), _c0)
except Exception:
    pass
layout["icon_0"] = [1153, 0, 1204, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_03_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-6/01_icon_icon_1.png
try:
    _c1 = get_crop(1, 100, 70)
    canvas.paste(_c1, (1214, 0), _c1)
except Exception:
    pass
layout["icon_1"] = [1214, 0, 1314, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_03_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-6/02_icon_icon_2.png
try:
    _c2 = get_crop(2, 63, 61)
    canvas.paste(_c2, (242, 3), _c2)
except Exception:
    pass
layout["icon_2"] = [242, 3, 305, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_03_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-6/03_icon_7_10_my.png
try:
    _c3 = get_crop(3, 168, 144)
    canvas.paste(_c3, (0, 122), _c3)
except Exception:
    pass
layout["7:10_my"] = [0, 122, 168, 266]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_03_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-6/04_icon_icon_4.png
try:
    _c4 = get_crop(4, 58, 62)
    canvas.paste(_c4, (313, 2), _c4)
except Exception:
    pass
layout["icon_4"] = [313, 2, 371, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_03_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-6/05_icon_Recent_locations.png
try:
    _c5 = get_crop(5, 115, 119)
    canvas.paste(_c5, (45, 356), _c5)
except Exception:
    pass
layout["Recent_locations"] = [45, 356, 160, 475]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_03_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-6/06_icon_Search_by_city.png
try:
    _c6 = get_crop(6, 85, 112)
    canvas.paste(_c6, (232, 144), _c6)
except Exception:
    pass
layout["Search_by_city"] = [232, 144, 317, 256]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_03_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-6/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 50, 67)
    canvas.paste(_c7, (1320, 1), _c7)
except Exception:
    pass
layout["icon_7"] = [1320, 1, 1370, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_03_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-6/08_icon_7_10_my.png
try:
    _c8 = get_crop(8, 47, 64)
    canvas.paste(_c8, (186, 1), _c8)
except Exception:
    pass
layout["7:10_my"] = [186, 1, 233, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_03_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-6/09_icon_Los_Angeles_CA.png
try:
    _c9 = get_crop(9, 123, 140)
    canvas.paste(_c9, (44, 846), _c9)
except Exception:
    pass
layout["Los_Angeles,_CA"] = [44, 846, 167, 986]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_03_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-6/10_icon_Clear.png
try:
    _c10 = get_crop(10, 144, 144)
    canvas.paste(_c10, (1248, 122), _c10)
except Exception:
    pass
layout["Clear"] = [1248, 122, 1392, 266]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_03_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-6/11_icon_7_10_my.png
try:
    _c11 = get_crop(11, 54, 66)
    canvas.paste(_c11, (116, 0), _c11)
except Exception:
    pass
layout["7:10_my"] = [116, 0, 170, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_03_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-6/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 47, 61)
    canvas.paste(_c12, (384, 3), _c12)
except Exception:
    pass
layout["icon_12"] = [384, 3, 431, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_03_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-6/13_text_Search_by_city.png
try:
    _c13 = get_crop(13, 936, 144)
    canvas.paste(_c13, (312, 122), _c13)
except Exception:
    pass
layout["Search_by_city"] = [312, 122, 1248, 266]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_03_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-6/14_text_Use_my_current_location.png
try:
    _c14 = get_crop(14, 1440, 194)
    canvas.paste(_c14, (0, 316), _c14)
except Exception:
    pass
layout["Use_my_current_location"] = [0, 316, 1440, 510]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_03_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-6/15_text_Recent_locations.png
try:
    _c15 = get_crop(15, 441, 54)
    canvas.paste(_c15, (44, 562), _c15)
except Exception:
    pass
layout["Recent_locations"] = [44, 562, 485, 616]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_03_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-6/16_text_New_York_NY.png
try:
    _c16 = get_crop(16, 299, 55)
    canvas.paste(_c16, (213, 714), _c16)
except Exception:
    pass
layout["New_York,_NY"] = [213, 714, 512, 769]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_03_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-6/17_text_Los_Angeles_CA.png
try:
    _c17 = get_crop(17, 1440, 168)
    canvas.paste(_c17, (0, 825), _c17)
except Exception:
    pass
layout["Los_Angeles,_CA"] = [0, 825, 1440, 993]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_03_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-6/18_clickable_New_York_NY.png
try:
    _c18 = get_crop(18, 1440, 168)
    canvas.paste(_c18, (0, 657), _c18)
except Exception:
    pass
layout["New_York,_NY"] = [0, 657, 1440, 825]
