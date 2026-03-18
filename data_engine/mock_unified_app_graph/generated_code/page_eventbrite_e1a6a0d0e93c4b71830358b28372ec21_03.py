# page_id: page_eventbrite_e1a6a0d0e93c4b71830358b28372ec21_03
# screenshot: 2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-5.png
# step_index: 3/9
# task: Open Eventbrite. Search for "Language Learning". Filter only online events. Note how many events are available for "Spanish".
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: fallback_compose
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw UI background and structural elements for the provided canvas
# Available variables:
# - canvas: PIL Image (1440x2960 RGB)
# - draw: PIL ImageDraw object
# - font_sm, font_md, font_lg, font_xl

# Overall background (slightly off-white to match screenshot tone)
draw.rectangle([(0, 0), (1440, 2960)], fill="#FBFBFD")

# Status bar (top ~50-64px) - light gray bar behind system icons
status_h = 64
draw.rectangle([(0, 0), (1440, status_h)], fill="#BDBDBD")

# Header / search area background (white, sits under status bar)
header_top = status_h
header_bottom = 232
draw.rectangle([(0, header_top), (1440, header_bottom)], fill="#FFFFFF")

# Blue underline for the search field (accent line)
underline_left = 48
underline_right = 1392
underline_y = 192
draw.rectangle([(underline_left, underline_y - 4), (underline_right, underline_y), fill := "#1E56FF"], fill=fill)

# subtle divider/shadow below header area
draw.line([(0, header_bottom), (1440, header_bottom)], fill="#E6E6E9", width=1)

# Section title "Events" would be pasted later; draw a faint separator above the first content row
draw.line([(48, 350), (1392, 350)], fill="#EFEFF1", width=1)

# Card/background blocks for each event list item (rounded rectangles)
card_x1 = 48
card_x2 = 1392
card_width = card_x2 - card_x1
card_h = 396
card_positions_y = [390, 786, 1182, 1578, 1974]

for y in card_positions_y:
    # Slightly off-white card background to separate from page background
    rect_bbox = [card_x1, y, card_x2, y + card_h]
    draw.rounded_rectangle(rect_bbox, radius=12, fill="#FFFFFF", outline="#E8E8EA", width=1)
    # subtle shadow line under each card
    shadow_y = y + card_h + 2
    draw.line([(card_x1 + 6, shadow_y), (card_x2 - 6, shadow_y)], fill="#F3F3F5", width=1)

# Bottom navigation bar background region (reserved area)
nav_top = 2804
nav_bottom = 2960
draw.rectangle([(0, nav_top), (1440, nav_bottom)], fill="#FFFFFF")
# top border for nav bar
draw.line([(0, nav_top), (1440, nav_top)], fill="#E6E6E9", width=1)

# Additional subtle separators between content groups (in content area)
# (Drawn at approximate logical boundaries to match screenshot structure)
draw.line([(48, 560), (1392, 560)], fill="#F0F0F2", width=1)
draw.line([(48, 952), (1392, 952)], fill="#F0F0F2", width=1)
draw.line([(48, 1348), (1392, 1348)], fill="#F0F0F2", width=1)
draw.line([(48, 1736), (1392, 1736)], fill="#F0F0F2", width=1)

# Small left margin guide lines (subtle) to visually align content columns without drawing any icons/text
draw.line([(48, header_bottom + 8), (48, nav_top - 8)], fill="#FFFFFF", width=0)  # no-op visually but keeps alignment logic

# End of structural/background drawing

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_03_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-5/00_icon_Fri_Apr_26.png
try:
    _c0 = get_crop(0, 1344, 396)
    canvas.paste(_c0, (48, 1974), _c0)
except Exception:
    pass
layout["Fri,_Apr_26"] = [48, 1974, 1392, 2370]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_03_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-5/01_icon_Language_Learning.png
try:
    _c1 = get_crop(1, 1344, 191)
    canvas.paste(_c1, (48, 72), _c1)
except Exception:
    pass
layout["Language_Learning}"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_03_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-5/02_icon_Events.png
try:
    _c2 = get_crop(2, 1344, 396)
    canvas.paste(_c2, (48, 390), _c2)
except Exception:
    pass
layout["Events"] = [48, 390, 1392, 786]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_03_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-5/03_icon_al.png
try:
    _c3 = get_crop(3, 1344, 396)
    canvas.paste(_c3, (48, 1578), _c3)
except Exception:
    pass
layout["al"] = [48, 1578, 1392, 1974]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_03_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-5/04_icon_icon_4.png
try:
    _c4 = get_crop(4, 50, 55)
    canvas.paste(_c4, (316, 6), _c4)
except Exception:
    pass
layout["icon_4"] = [316, 6, 366, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_03_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-5/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 40, 52)
    canvas.paste(_c5, (255, 8), _c5)
except Exception:
    pass
layout["icon_5"] = [255, 8, 295, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_03_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-5/06_icon_5.17.png
try:
    _c6 = get_crop(6, 50, 58)
    canvas.paste(_c6, (185, 4), _c6)
except Exception:
    pass
layout["5.17"] = [185, 4, 235, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_03_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-5/07_icon_Tickets.png
try:
    _c7 = get_crop(7, 288, 156)
    canvas.paste(_c7, (864, 2804), _c7)
except Exception:
    pass
layout["Tickets"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_03_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-5/08_icon_5.17.png
try:
    _c8 = get_crop(8, 55, 58)
    canvas.paste(_c8, (114, 5), _c8)
except Exception:
    pass
layout["5.17"] = [114, 5, 169, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_03_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-5/09_icon_8_2194_creator_followers.png
try:
    _c9 = get_crop(9, 1344, 396)
    canvas.paste(_c9, (48, 786), _c9)
except Exception:
    pass
layout["8_2194_creator_followers"] = [48, 786, 1392, 1182]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_03_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-5/10_icon_8_2194_creator_followers.png
try:
    _c10 = get_crop(10, 1344, 396)
    canvas.paste(_c10, (48, 1182), _c10)
except Exception:
    pass
layout["8_2194_creator_followers"] = [48, 1182, 1392, 1578]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_03_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-5/11_icon_More.png
try:
    _c11 = get_crop(11, 288, 156)
    canvas.paste(_c11, (1152, 2804), _c11)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_03_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-5/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 44, 55)
    canvas.paste(_c12, (1323, 5), _c12)
except Exception:
    pass
layout["icon_12"] = [1323, 5, 1367, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_03_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-5/13_icon_5.17.png
try:
    _c13 = get_crop(13, 110, 98)
    canvas.paste(_c13, (64, 121), _c13)
except Exception:
    pass
layout["5.17"] = [64, 121, 174, 219]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_03_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-5/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 52, 59)
    canvas.paste(_c14, (1217, 3), _c14)
except Exception:
    pass
layout["icon_14"] = [1217, 3, 1269, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_03_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-5/15_icon_Cancel.png
try:
    _c15 = get_crop(15, 41, 56)
    canvas.paste(_c15, (1272, 5), _c15)
except Exception:
    pass
layout["Cancel"] = [1272, 5, 1313, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_03_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-5/16_icon_Book_Launch_Party_Language_Learning_in.png
try:
    _c16 = get_crop(16, 1344, 396)
    canvas.paste(_c16, (48, 1974), _c16)
except Exception:
    pass
layout["Book_Launch_Party:_Langua"] = [48, 1974, 1392, 2370]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_03_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-5/17_icon_Online.png
try:
    _c17 = get_crop(17, 113, 52)
    canvas.paste(_c17, (390, 1023), _c17)
except Exception:
    pass
layout["Online"] = [390, 1023, 503, 1075]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_03_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-5/18_icon_Search_events.png
try:
    _c18 = get_crop(18, 288, 156)
    canvas.paste(_c18, (288, 2804), _c18)
except Exception:
    pass
layout["Search_events"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_03_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-5/19_icon_Language_Learning.png
try:
    _c19 = get_crop(19, 1344, 396)
    canvas.paste(_c19, (48, 1182), _c19)
except Exception:
    pass
layout["Language_Learning"] = [48, 1182, 1392, 1578]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_03_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-5/20_icon_Home.png
try:
    _c20 = get_crop(20, 288, 156)
    canvas.paste(_c20, (0, 2804), _c20)
except Exception:
    pass
layout["Home"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_03_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-5/21_icon_Cancel.png
try:
    _c21 = get_crop(21, 144, 144)
    canvas.paste(_c21, (1099, 96), _c21)
except Exception:
    pass
layout["Cancel"] = [1099, 96, 1243, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_03_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-5/22_icon_Cancel.png
try:
    _c22 = get_crop(22, 149, 144)
    canvas.paste(_c22, (1243, 97), _c22)
except Exception:
    pass
layout["Cancel"] = [1243, 97, 1392, 241]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_03_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-5/23_icon_icon_23.png
try:
    _c23 = get_crop(23, 43, 55)
    canvas.paste(_c23, (386, 6), _c23)
except Exception:
    pass
layout["icon_23"] = [386, 6, 429, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_03_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-5/24_icon_Favorites.png
try:
    _c24 = get_crop(24, 288, 156)
    canvas.paste(_c24, (576, 2804), _c24)
except Exception:
    pass
layout["Favorites"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_03_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-5/25_icon_Language_Learning_Through.png
try:
    _c25 = get_crop(25, 1344, 396)
    canvas.paste(_c25, (48, 1578), _c25)
except Exception:
    pass
layout["Language_Learning_Through"] = [48, 1578, 1392, 1974]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_03_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-5/26_icon_SALT_presents_Delivering_Primary.png
try:
    _c26 = get_crop(26, 1344, 396)
    canvas.paste(_c26, (48, 1578), _c26)
except Exception:
    pass
layout["SALT_presents:_Delivering"] = [48, 1578, 1392, 1974]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_03_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-5/27_icon_Planning_and_Strategies_for_Language.png
try:
    _c27 = get_crop(27, 1344, 396)
    canvas.paste(_c27, (48, 390), _c27)
except Exception:
    pass
layout["Planning_and_Strategies_f"] = [48, 390, 1392, 786]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_03_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-5/28_icon_Planning_and_Strategies_for_Language.png
try:
    _c28 = get_crop(28, 1344, 396)
    canvas.paste(_c28, (48, 390), _c28)
except Exception:
    pass
layout["Planning_and_Strategies_f"] = [48, 390, 1392, 786]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_03_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-5/29_icon_Online.png
try:
    _c29 = get_crop(29, 114, 54)
    canvas.paste(_c29, (390, 1813), _c29)
except Exception:
    pass
layout["Online"] = [390, 1813, 504, 1867]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_03_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-5/30_icon_Gateway_Bible_Church.png
try:
    _c30 = get_crop(30, 1344, 396)
    canvas.paste(_c30, (48, 390), _c30)
except Exception:
    pass
layout["Gateway_Bible_Church"] = [48, 390, 1392, 786]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_03_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-5/31_text_5.17.png
try:
    _c31 = get_crop(31, 87, 43)
    canvas.paste(_c31, (22, 17), _c31)
except Exception:
    pass
layout["5.17"] = [22, 17, 109, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_03_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-5/32_text_Events.png
try:
    _c32 = get_crop(32, 186, 56)
    canvas.paste(_c32, (46, 301), _c32)
except Exception:
    pass
layout["Events"] = [46, 301, 232, 357]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_03_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-5/33_text_8_14_creator_followers.png
try:
    _c33 = get_crop(33, 1344, 396)
    canvas.paste(_c33, (48, 1974), _c33)
except Exception:
    pass
layout["8_14_creator_followers"] = [48, 1974, 1392, 2370]
