# page_id: page_eventbrite_cee80d2c889b45cd86e30248baf590b8_11
# screenshot: 2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-13.png
# step_index: 11/13
# task: Open Eventbrite. Search Food & Drink party event in New York. Select the first one that is not promoted. Follow the organizer. Share to google keep notes.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Paint the page background and structural elements for the mobile UI mockup
# Uses provided: canvas (1440x2960 PIL Image), draw (ImageDraw), font_sm, font_md, font_lg, font_xl

# Overall background (slightly off-white to match screenshot)
bg_color = (250, 251, 253)
draw.rectangle([(0, 0), (1440, 2960)], fill=bg_color)

# Status bar area (top ~84px) - subtle muted grey
status_bar_h = 84
status_color = (231, 232, 234)
draw.rectangle([(0, 0), (1440, status_bar_h)], fill=status_color)

# Top header / toolbar background (white) below status bar
header_top = status_bar_h
header_h = 100
draw.rectangle([(0, header_top), (1440, header_top + header_h)], fill=(255, 255, 255))

# Thin divider under header
divider_y = header_top + header_h
draw.line([(32, divider_y), (1408, divider_y)], fill=(217, 219, 222), width=2)

# Filter / chips background strip (light subtle tint)
chips_top = 240
chips_bottom = 360
chips_bg = (249, 250, 251)
draw.rectangle([(0, chips_top), (1440, chips_bottom)], fill=chips_bg)

# Subtle separator under chips area
draw.line([(32, chips_bottom + 6), (1408, chips_bottom + 6)], fill=(230, 231, 233), width=1)

# Card geometry (left/right margins follow detected content: 48 px margin)
card_x0 = 48
card_x1 = 48 + 1344  # 1392
card_radius = 26

# First event card (background + shadow)
first_card_top = 620
first_card_bottom = 1660
shadow_offset = 8
# shadow
draw.rounded_rectangle(
    [(card_x0, first_card_top + shadow_offset), (card_x1, first_card_bottom + shadow_offset)],
    radius=card_radius + 2,
    fill=(235, 237, 240)
)
# white card
draw.rounded_rectangle(
    [(card_x0, first_card_top), (card_x1, first_card_bottom)],
    radius=card_radius,
    fill=(255, 255, 255)
)

# Thin divider/separator below first card (subtle)
sep_y = first_card_bottom + 18
draw.line([(card_x0 + 8, sep_y), (card_x1 - 8, sep_y)], fill=(235, 237, 240), width=1)

# Second event card (background + shadow)
second_card_top = 1680
second_card_bottom = 2780  # stop above bottom nav
# shadow
draw.rounded_rectangle(
    [(card_x0, second_card_top + shadow_offset), (card_x1, second_card_bottom + shadow_offset)],
    radius=card_radius + 2,
    fill=(235, 237, 240)
)
# white card
draw.rounded_rectangle(
    [(card_x0, second_card_top), (card_x1, second_card_bottom)],
    radius=card_radius,
    fill=(255, 255, 255)
)

# Divider line between content and bottom navigation
bottom_nav_top = 2804
draw.line([(0, bottom_nav_top), (1440, bottom_nav_top)], fill=(220, 222, 224), width=2)

# Bottom navigation bar background (white)
draw.rectangle([(0, bottom_nav_top), (1440, 2960)], fill=(255, 255, 255))

# Very subtle page left/right content gutters (visual framing)
gutter_color = (248, 249, 250)
draw.rectangle([(0, 0), (24, 2960)], fill=gutter_color)
draw.rectangle([(1440 - 24, 0), (1440, 2960)], fill=gutter_color)

# Additional subtle horizontal rhythm lines to separate sections (visual structure only)
section_lines = [
    header_top + 48,
    header_top + header_h + 36,
    chips_bottom + 56,
    first_card_top - 40,
    second_card_top - 28
]
for y in section_lines:
    draw.line([(48, y), (1392, y)], fill=(246, 246, 247), width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_11_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-13/00_icon_Music.png
try:
    _c0 = get_crop(0, 196, 112)
    canvas.paste(_c0, (829, 405), _c0)
except Exception:
    pass
layout["Music"] = [829, 405, 1025, 517]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_11_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-13/01_icon_Business.png
try:
    _c1 = get_crop(1, 251, 111)
    canvas.paste(_c1, (1029, 406), _c1)
except Exception:
    pass
layout["Business"] = [1029, 406, 1280, 517]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_11_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-13/02_icon_Anytime.png
try:
    _c2 = get_crop(2, 1344, 124)
    canvas.paste(_c2, (48, 525), _c2)
except Exception:
    pass
layout["Anytime"] = [48, 525, 1392, 649]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_11_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-13/03_icon_Filters.png
try:
    _c3 = get_crop(3, 434, 144)
    canvas.paste(_c3, (0, 259), _c3)
except Exception:
    pass
layout["Filters"] = [0, 259, 434, 403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_11_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-13/04_icon_Favorite_button.png
try:
    _c4 = get_crop(4, 144, 144)
    canvas.paste(_c4, (1092, 1213), _c4)
except Exception:
    pass
layout["Favorite_button"] = [1092, 1213, 1236, 1357]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_11_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-13/05_icon_Favorite_button.png
try:
    _c5 = get_crop(5, 144, 144)
    canvas.paste(_c5, (1092, 2206), _c5)
except Exception:
    pass
layout["Favorite_button"] = [1092, 2206, 1236, 2350]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_11_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-13/06_icon_Foo.png
try:
    _c6 = get_crop(6, 126, 110)
    canvas.paste(_c6, (1284, 406), _c6)
except Exception:
    pass
layout["Foo"] = [1284, 406, 1410, 516]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_11_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-13/07_icon_Overflow_menu_button.png
try:
    _c7 = get_crop(7, 144, 144)
    canvas.paste(_c7, (1236, 2206), _c7)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 2206, 1380, 2350]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_11_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-13/08_icon_Overflow_menu_button.png
try:
    _c8 = get_crop(8, 144, 144)
    canvas.paste(_c8, (1236, 1213), _c8)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 1213, 1380, 1357]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_11_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-13/09_icon_Inspired_by_Italian_Food_An_Award-Winnin.png
try:
    _c9 = get_crop(9, 1344, 1126)
    canvas.paste(_c9, (48, 1690), _c9)
except Exception:
    pass
layout["Inspired_by_Italian_Food:"] = [48, 1690, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_11_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-13/10_icon_Foo.png
try:
    _c10 = get_crop(10, 144, 144)
    canvas.paste(_c10, (1248, 96), _c10)
except Exception:
    pass
layout["Foo"] = [1248, 96, 1392, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_11_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-13/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 59, 62)
    canvas.paste(_c11, (245, 1), _c11)
except Exception:
    pass
layout["icon_11"] = [245, 1, 304, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_11_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-13/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 53, 64)
    canvas.paste(_c12, (1151, 1), _c12)
except Exception:
    pass
layout["icon_12"] = [1151, 1, 1204, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_11_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-13/13_icon_9.45.png
try:
    _c13 = get_crop(13, 124, 117)
    canvas.paste(_c13, (55, 112), _c13)
except Exception:
    pass
layout["9.45"] = [55, 112, 179, 229]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_11_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-13/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 99, 63)
    canvas.paste(_c14, (1212, 0), _c14)
except Exception:
    pass
layout["icon_14"] = [1212, 0, 1311, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_11_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-13/15_icon_9.45.png
try:
    _c15 = get_crop(15, 56, 63)
    canvas.paste(_c15, (182, 0), _c15)
except Exception:
    pass
layout["9.45"] = [182, 0, 238, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_11_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-13/16_icon_icon_16.png
try:
    _c16 = get_crop(16, 60, 63)
    canvas.paste(_c16, (312, 1), _c16)
except Exception:
    pass
layout["icon_16"] = [312, 1, 372, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_11_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-13/17_icon_New_York.png
try:
    _c17 = get_crop(17, 434, 144)
    canvas.paste(_c17, (0, 259), _c17)
except Exception:
    pass
layout["New_York"] = [0, 259, 434, 403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_11_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-13/18_icon_icon_18.png
try:
    _c18 = get_crop(18, 56, 61)
    canvas.paste(_c18, (1319, 0), _c18)
except Exception:
    pass
layout["icon_18"] = [1319, 0, 1375, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_11_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-13/19_icon_Author_on_Cuisine_Community.png
try:
    _c19 = get_crop(19, 288, 156)
    canvas.paste(_c19, (576, 2804), _c19)
except Exception:
    pass
layout["Author_on_Cuisine_&_Commu"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_11_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-13/20_icon_Cheese.png
try:
    _c20 = get_crop(20, 1344, 945)
    canvas.paste(_c20, (48, 697), _c20)
except Exception:
    pass
layout["Cheese"] = [48, 697, 1392, 1642]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_11_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-13/21_icon_9.45.png
try:
    _c21 = get_crop(21, 57, 65)
    canvas.paste(_c21, (114, 0), _c21)
except Exception:
    pass
layout["9.45"] = [114, 0, 171, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_11_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-13/22_icon_Wed_Mar_27_._7_00_PM_EDT.png
try:
    _c22 = get_crop(22, 288, 156)
    canvas.paste(_c22, (288, 2804), _c22)
except Exception:
    pass
layout["Wed,_Mar_27_._7:00_PM_EDT"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_11_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-13/23_icon_Food_Drink.png
try:
    _c23 = get_crop(23, 1344, 191)
    canvas.paste(_c23, (48, 72), _c23)
except Exception:
    pass
layout["Food_&_Drink"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_11_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-13/24_icon_Inspired_by_Italian_Food_An_Award-Winnin.png
try:
    _c24 = get_crop(24, 288, 156)
    canvas.paste(_c24, (864, 2804), _c24)
except Exception:
    pass
layout["Inspired_by_Italian_Food:"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_11_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-13/25_icon_Food_Drink.png
try:
    _c25 = get_crop(25, 49, 62)
    canvas.paste(_c25, (383, 2), _c25)
except Exception:
    pass
layout["Food_&_Drink"] = [383, 2, 432, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_11_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-13/26_icon_More.png
try:
    _c26 = get_crop(26, 288, 156)
    canvas.paste(_c26, (1152, 2804), _c26)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_11_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-13/27_icon_6.30_PM_EDT.png
try:
    _c27 = get_crop(27, 1344, 945)
    canvas.paste(_c27, (48, 697), _c27)
except Exception:
    pass
layout["6.30_PM_EDT"] = [48, 697, 1392, 1642]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_11_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-13/28_text_9.45.png
try:
    _c28 = get_crop(28, 94, 43)
    canvas.paste(_c28, (20, 15), _c28)
except Exception:
    pass
layout["9.45"] = [20, 15, 114, 58]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_11_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-13/29_text_At_a_traffic_island_across_street_from.png
try:
    _c29 = get_crop(29, 1344, 124)
    canvas.paste(_c29, (48, 525), _c29)
except Exception:
    pass
layout["At_a_traffic_island_acros"] = [48, 525, 1392, 649]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_11_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-13/30_text_Thursday_April_25th.png
try:
    _c30 = get_crop(30, 1344, 124)
    canvas.paste(_c30, (48, 525), _c30)
except Exception:
    pass
layout["Thursday_April_25th_"] = [48, 525, 1392, 649]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_11_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-13/31_text_2024.png
try:
    _c31 = get_crop(31, 157, 66)
    canvas.paste(_c31, (1217, 730), _c31)
except Exception:
    pass
layout["2024"] = [1217, 730, 1374, 796]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_11_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-13/32_text_Sociale.png
try:
    _c32 = get_crop(32, 145, 43)
    canvas.paste(_c32, (91, 2723), _c32)
except Exception:
    pass
layout["Sociale"] = [91, 2723, 236, 2766]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_11_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-13/33_clickable_Home.png
try:
    _c33 = get_crop(33, 288, 156)
    canvas.paste(_c33, (0, 2804), _c33)
except Exception:
    pass
layout["Home"] = [0, 2804, 288, 2960]
