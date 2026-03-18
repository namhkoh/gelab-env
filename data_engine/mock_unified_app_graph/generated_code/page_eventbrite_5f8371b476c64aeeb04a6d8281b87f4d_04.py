# page_id: page_eventbrite_5f8371b476c64aeeb04a6d8281b87f4d_04
# screenshot: 2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-6.png
# step_index: 4/7
# task: Open Eventbrite. Search Science & Tech event. Select the first one that is not promoted. If it is free, add it to Favorites. If it is not free, record its price in Google Keep Notes.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Background and structural elements for Eventbrite "Science & Tech" UI mock
# Uses provided variables: canvas (PIL Image), draw (ImageDraw), font_sm, font_md, font_lg, font_xl

# Overall background (very light off-white similar to screenshot)
draw.rectangle([(0, 0), (1440, 2960)], fill=(250, 251, 253))

# Status bar (top area with darker gray tone)
status_h = 96
draw.rectangle([(0, 0), (1440, status_h)], fill=(200, 200, 200))

# Header / toolbar area (white to contrast with status bar)
header_top = status_h
header_bottom = 260
draw.rectangle([(0, header_top), (1440, header_bottom)], fill=(255, 255, 255))

# Subtle divider under the header
draw.line([(48, header_bottom), (1392, header_bottom)], fill=(225, 225, 230), width=2)

# Light subtle separator above the content chips area
draw.line([(48, 520), (1392, 520)], fill=(240, 240, 245), width=1)

# First event card grouping background (rounded rectangle with subtle shadow)
card1 = (48, 676, 1392, 1870)  # x0,y0,x1,y1 from detected grouping
shadow_offset = (6, 8)
shadow_box1 = (card1[0] + shadow_offset[0], card1[1] + shadow_offset[1],
               card1[2] + shadow_offset[0], card1[3] + shadow_offset[1])
draw.rounded_rectangle(shadow_box1, radius=28, fill=(232, 232, 237))
draw.rounded_rectangle(card1, radius=24, fill=(255, 255, 255))

# Divider line between first card and following content
draw.line([(48, card1[3] + 16), (1392, card1[3] + 16)], fill=(235, 235, 240), width=1)

# Second event card grouping background (rounded rectangle with subtle shadow)
card2 = (48, 1918, 1392, 2816)
shadow_box2 = (card2[0] + shadow_offset[0], card2[1] + shadow_offset[1],
               card2[2] + shadow_offset[0], card2[3] + shadow_offset[1])
draw.rounded_rectangle(shadow_box2, radius=28, fill=(232, 232, 237))
draw.rounded_rectangle(card2, radius=24, fill=(255, 255, 255))

# Subtle separator line above the bottom navigation
nav_top = 2816
draw.line([(0, nav_top), (1440, nav_top)], fill=(230, 230, 235), width=2)

# Bottom navigation bar background
draw.rectangle([(0, nav_top), (1440, 2960)], fill=(255, 255, 255))

# Small top border for nav to emphasize separation
draw.line([(0, nav_top), (1440, nav_top)], fill=(220, 220, 225), width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_04_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-6/00_icon_Music.png
try:
    _c0 = get_crop(0, 187, 103)
    canvas.paste(_c0, (837, 410), _c0)
except Exception:
    pass
layout["Music"] = [837, 410, 1024, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_04_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-6/01_icon_Business.png
try:
    _c1 = get_crop(1, 241, 103)
    canvas.paste(_c1, (1036, 410), _c1)
except Exception:
    pass
layout["Business"] = [1036, 410, 1277, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_04_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-6/02_icon_Anytime.png
try:
    _c2 = get_crop(2, 400, 103)
    canvas.paste(_c2, (425, 410), _c2)
except Exception:
    pass
layout["Anytime"] = [425, 410, 825, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_04_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-6/03_icon_Filters.png
try:
    _c3 = get_crop(3, 359, 103)
    canvas.paste(_c3, (54, 410), _c3)
except Exception:
    pass
layout["Filters"] = [54, 410, 413, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_04_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-6/04_icon_Foo.png
try:
    _c4 = get_crop(4, 148, 110)
    canvas.paste(_c4, (1283, 406), _c4)
except Exception:
    pass
layout["Foo"] = [1283, 406, 1431, 516]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_04_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-6/05_icon_Breakout.png
try:
    _c5 = get_crop(5, 144, 144)
    canvas.paste(_c5, (1092, 2434), _c5)
except Exception:
    pass
layout["Breakout"] = [1092, 2434, 1236, 2578]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_04_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-6/06_icon_Favorite_button.png
try:
    _c6 = get_crop(6, 144, 144)
    canvas.paste(_c6, (1092, 1192), _c6)
except Exception:
    pass
layout["Favorite_button"] = [1092, 1192, 1236, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_04_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-6/07_icon_Breakout.png
try:
    _c7 = get_crop(7, 144, 144)
    canvas.paste(_c7, (1236, 2434), _c7)
except Exception:
    pass
layout["Breakout"] = [1236, 2434, 1380, 2578]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_04_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-6/08_icon_Overflow_menu_button.png
try:
    _c8 = get_crop(8, 144, 144)
    canvas.paste(_c8, (1236, 1192), _c8)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 1192, 1380, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_04_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-6/09_icon_The_Tiny_Cupboard_Comedy_Club_s_Stand-Up.png
try:
    _c9 = get_crop(9, 1344, 1194)
    canvas.paste(_c9, (48, 676), _c9)
except Exception:
    pass
layout["The_Tiny_Cupboard_Comedy_"] = [48, 676, 1392, 1870]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_04_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-6/10_icon_The_Tiny_Cupboard_Comedy_Club_s_Stand-Up.png
try:
    _c10 = get_crop(10, 1344, 1194)
    canvas.paste(_c10, (48, 676), _c10)
except Exception:
    pass
layout["The_Tiny_Cupboard_Comedy_"] = [48, 676, 1392, 1870]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_04_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-6/11_icon_Foo.png
try:
    _c11 = get_crop(11, 144, 144)
    canvas.paste(_c11, (1248, 96), _c11)
except Exception:
    pass
layout["Foo"] = [1248, 96, 1392, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_04_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-6/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 57, 62)
    canvas.paste(_c12, (246, 1), _c12)
except Exception:
    pass
layout["icon_12"] = [246, 1, 303, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_04_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-6/13_icon_9.37.png
try:
    _c13 = get_crop(13, 124, 118)
    canvas.paste(_c13, (55, 112), _c13)
except Exception:
    pass
layout["9.37"] = [55, 112, 179, 230]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_04_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-6/14_icon_9.37.png
try:
    _c14 = get_crop(14, 56, 62)
    canvas.paste(_c14, (182, 0), _c14)
except Exception:
    pass
layout["9.37"] = [182, 0, 238, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_04_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-6/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 104, 61)
    canvas.paste(_c15, (1206, 0), _c15)
except Exception:
    pass
layout["icon_15"] = [1206, 0, 1310, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_04_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-6/16_icon_icon_16.png
try:
    _c16 = get_crop(16, 57, 63)
    canvas.paste(_c16, (313, 1), _c16)
except Exception:
    pass
layout["icon_16"] = [313, 1, 370, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_04_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-6/17_icon_9.37.png
try:
    _c17 = get_crop(17, 57, 64)
    canvas.paste(_c17, (113, 0), _c17)
except Exception:
    pass
layout["9.37"] = [113, 0, 170, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_04_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-6/18_icon_IN_Conversation_Women_in_STEAM.png
try:
    _c18 = get_crop(18, 288, 156)
    canvas.paste(_c18, (288, 2804), _c18)
except Exception:
    pass
layout["IN_Conversation:_Women_in"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_04_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-6/19_icon_New_York.png
try:
    _c19 = get_crop(19, 434, 144)
    canvas.paste(_c19, (0, 259), _c19)
except Exception:
    pass
layout["New_York"] = [0, 259, 434, 403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_04_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-6/20_icon_icon_20.png
try:
    _c20 = get_crop(20, 61, 61)
    canvas.paste(_c20, (1317, 0), _c20)
except Exception:
    pass
layout["icon_20"] = [1317, 0, 1378, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_04_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-6/21_icon_Science_Tech.png
try:
    _c21 = get_crop(21, 1344, 191)
    canvas.paste(_c21, (48, 72), _c21)
except Exception:
    pass
layout["Science_&_Tech"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_04_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-6/22_icon_Ticket_sales_end_soon.png
try:
    _c22 = get_crop(22, 288, 156)
    canvas.paste(_c22, (288, 2804), _c22)
except Exception:
    pass
layout["Ticket_sales_end_soon"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_04_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-6/23_icon_Science_Tech.png
try:
    _c23 = get_crop(23, 48, 61)
    canvas.paste(_c23, (383, 2), _c23)
except Exception:
    pass
layout["Science_&_Tech"] = [383, 2, 431, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_04_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-6/24_icon_IN_Conversation_Women_in_STEAM.png
try:
    _c24 = get_crop(24, 1344, 898)
    canvas.paste(_c24, (48, 1918), _c24)
except Exception:
    pass
layout["IN_Conversation:_Women_in"] = [48, 1918, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_04_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-6/25_icon_Ticket_sales_end_soon.png
try:
    _c25 = get_crop(25, 288, 156)
    canvas.paste(_c25, (0, 2804), _c25)
except Exception:
    pass
layout["Ticket_sales_end_soon"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_04_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-6/26_icon_IN_Conversation_Women_in_STEAM.png
try:
    _c26 = get_crop(26, 288, 156)
    canvas.paste(_c26, (576, 2804), _c26)
except Exception:
    pass
layout["IN_Conversation:_Women_in"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_04_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-6/27_icon_Breakout.png
try:
    _c27 = get_crop(27, 288, 156)
    canvas.paste(_c27, (1152, 2804), _c27)
except Exception:
    pass
layout["Breakout"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_04_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-6/28_icon_Promoted.png
try:
    _c28 = get_crop(28, 249, 63)
    canvas.paste(_c28, (84, 1764), _c28)
except Exception:
    pass
layout["Promoted"] = [84, 1764, 333, 1827]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_04_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-6/29_icon_Cupboard.png
try:
    _c29 = get_crop(29, 42, 58)
    canvas.paste(_c29, (286, 1767), _c29)
except Exception:
    pass
layout["Cupboard"] = [286, 1767, 328, 1825]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_04_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-6/30_icon_Breakout.png
try:
    _c30 = get_crop(30, 288, 156)
    canvas.paste(_c30, (864, 2804), _c30)
except Exception:
    pass
layout["Breakout"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_04_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-6/31_text_9.37.png
try:
    _c31 = get_crop(31, 89, 43)
    canvas.paste(_c31, (20, 17), _c31)
except Exception:
    pass
layout["9.37"] = [20, 17, 109, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_04_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-6/32_text_2_154events.png
try:
    _c32 = get_crop(32, 359, 103)
    canvas.paste(_c32, (54, 410), _c32)
except Exception:
    pass
layout["2,154events"] = [54, 410, 413, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_04_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-6/33_text_The.png
try:
    _c33 = get_crop(33, 75, 43)
    canvas.paste(_c33, (94, 1708), _c33)
except Exception:
    pass
layout["The"] = [94, 1708, 169, 1751]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_04_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-6/34_text_Cupboard.png
try:
    _c34 = get_crop(34, 195, 55)
    canvas.paste(_c34, (250, 1704), _c34)
except Exception:
    pass
layout["Cupboard"] = [250, 1704, 445, 1759]
