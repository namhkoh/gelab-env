# page_id: page_eventbrite_c7c81d1bf6744774b99294e9f124dda3_02
# screenshot: 2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-4.png
# step_index: 2/10
# task: Open Eventbrite. Search for "Fitness". Select the events in the location "Chicago". What is the price of the first event in listing?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw background and structural UI elements for the canvas (1440x2960)
# Available variables: canvas (PIL Image), draw (PIL ImageDraw), font_sm, font_md, font_lg, font_xl

w, h = canvas.size

# Colors
bg_white = "#FFFFFF"
status_gray = "#BDBDBD"        # status bar background
divider_light = "#EDEEF1"      # thin separators
divider_subtle = "#F5F6F8"
blue_accent = "#2F51FF"        # search underline accent
card_outline = "#F0F1F4"
nav_divider = "#E6E6E9"

# Fill overall background (canvas may already be white)
draw.rectangle((0, 0, w, h), fill=bg_white)

# Status bar (top area with signal/time background)
status_h = 72
draw.rectangle((0, 0, w, status_h), fill=status_gray)
# thin bottom edge of status bar
draw.line((0, status_h - 1, w, status_h - 1), fill=divider_light, width=1)

# Header / Search area background (keeps white but ensure clean area)
header_top = status_h
header_bottom = 160
draw.rectangle((0, header_top, w, header_bottom), fill=bg_white)

# Search field underline (accent) aligned with typical horizontal margins used by detected elements
search_left = 48
search_right = w - 48
underline_thickness = 6
underline_y1 = header_bottom - 8
underline_y2 = underline_y1 + underline_thickness
draw.rectangle((search_left, underline_y1, search_right, underline_y2), fill=blue_accent)

# Section container / card area for "Recent" list (subtle rounded card outline)
list_card_top = 260
list_card_bottom = 1860
card_left = 24
card_right = w - 24
draw.rounded_rectangle((card_left, list_card_top, card_right, list_card_bottom),
                       radius=12, fill=bg_white, outline=card_outline, width=1)

# Separator lines for each list row
# Based on detected row start at y=390 and row height ~144
first_row_y = 390
row_height = 144
num_rows = 10
for i in range(num_rows):
    sep_y = first_row_y + (i + 1) * row_height  # separator under each row
    # draw line only across the main content area (matching search field margins)
    draw.line((search_left, sep_y, search_right, sep_y), fill=divider_subtle, width=1)

# Thin left margin rule under the search underline to separate header from content (subtle)
draw.line((search_left, underline_y2 + 8, search_right, underline_y2 + 8), fill=divider_subtle, width=1)

# Bottom navigation bar background and top divider
nav_top = 2804
nav_bottom = h
draw.rectangle((0, nav_top, w, nav_bottom), fill=bg_white)
draw.line((0, nav_top, w, nav_top), fill=nav_divider, width=2)

# Very subtle shadow above nav to separate from content
shadow_h = 4
draw.rectangle((0, nav_top - shadow_h, w, nav_top), fill="#FBFBFC")

# Additional subtle vertical guide at left for content alignment (does not draw any icons/text)
# provides visual structure only
guide_x = search_left
draw.line((guide_x, list_card_top + 6, guide_x, list_card_bottom - 6), fill=divider_light, width=1)

# Final subtle right alignment guide (matching the search_right)
draw.line((search_right, list_card_top + 6, search_right, list_card_bottom - 6), fill=divider_light, width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_02_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-4/00_icon_7.09.png
try:
    _c0 = get_crop(0, 60, 65)
    canvas.paste(_c0, (114, 1), _c0)
except Exception:
    pass
layout["7.09"] = [114, 1, 174, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_02_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-4/01_icon_7.09.png
try:
    _c1 = get_crop(1, 58, 64)
    canvas.paste(_c1, (181, 0), _c1)
except Exception:
    pass
layout["7.09"] = [181, 0, 239, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_02_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-4/02_icon_Search_for__..png
try:
    _c2 = get_crop(2, 66, 64)
    canvas.paste(_c2, (308, 2), _c2)
except Exception:
    pass
layout["[Search_for__."] = [308, 2, 374, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_02_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-4/03_icon_icon_3.png
try:
    _c3 = get_crop(3, 49, 63)
    canvas.paste(_c3, (249, 2), _c3)
except Exception:
    pass
layout["icon_3"] = [249, 2, 298, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_02_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-4/04_icon_Cancel.png
try:
    _c4 = get_crop(4, 99, 62)
    canvas.paste(_c4, (1212, 0), _c4)
except Exception:
    pass
layout["Cancel"] = [1212, 0, 1311, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_02_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-4/05_icon_Cancel.png
try:
    _c5 = get_crop(5, 58, 62)
    canvas.paste(_c5, (1316, 0), _c5)
except Exception:
    pass
layout["Cancel"] = [1316, 0, 1374, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_02_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-4/06_icon_Cancel.png
try:
    _c6 = get_crop(6, 149, 144)
    canvas.paste(_c6, (1243, 97), _c6)
except Exception:
    pass
layout["Cancel"] = [1243, 97, 1392, 241]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_02_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-4/07_icon_Education.png
try:
    _c7 = get_crop(7, 1344, 144)
    canvas.paste(_c7, (48, 534), _c7)
except Exception:
    pass
layout["Education"] = [48, 534, 1392, 678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_02_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-4/08_icon_7.09.png
try:
    _c8 = get_crop(8, 125, 111)
    canvas.paste(_c8, (53, 113), _c8)
except Exception:
    pass
layout["7.09"] = [53, 113, 178, 224]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_02_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-4/09_icon_Search_for__..png
try:
    _c9 = get_crop(9, 1344, 144)
    canvas.paste(_c9, (48, 390), _c9)
except Exception:
    pass
layout["[Search_for__."] = [48, 390, 1392, 534]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_02_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-4/10_icon_Favorites.png
try:
    _c10 = get_crop(10, 288, 156)
    canvas.paste(_c10, (576, 2804), _c10)
except Exception:
    pass
layout["Favorites"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_02_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-4/11_icon_Close_current_screen.png
try:
    _c11 = get_crop(11, 144, 144)
    canvas.paste(_c11, (1248, 822), _c11)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 822, 1392, 966]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_02_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-4/12_icon_Tickets.png
try:
    _c12 = get_crop(12, 288, 156)
    canvas.paste(_c12, (864, 2804), _c12)
except Exception:
    pass
layout["Tickets"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_02_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-4/13_icon_Close_current_screen.png
try:
    _c13 = get_crop(13, 144, 144)
    canvas.paste(_c13, (1248, 1254), _c13)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 1254, 1392, 1398]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_02_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-4/14_icon_Close_current_screen.png
try:
    _c14 = get_crop(14, 144, 144)
    canvas.paste(_c14, (1248, 534), _c14)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 534, 1392, 678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_02_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-4/15_icon_Close_current_screen.png
try:
    _c15 = get_crop(15, 144, 144)
    canvas.paste(_c15, (1248, 1398), _c15)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 1398, 1392, 1542]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_02_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-4/16_icon_Close_current_screen.png
try:
    _c16 = get_crop(16, 144, 144)
    canvas.paste(_c16, (1248, 678), _c16)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 678, 1392, 822]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_02_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-4/17_icon_Search_for__..png
try:
    _c17 = get_crop(17, 1344, 191)
    canvas.paste(_c17, (48, 72), _c17)
except Exception:
    pass
layout["[Search_for__."] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_02_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-4/18_icon_Close_current_screen.png
try:
    _c18 = get_crop(18, 144, 144)
    canvas.paste(_c18, (1248, 1110), _c18)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 1110, 1392, 1254]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_02_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-4/19_icon_Close_current_screen.png
try:
    _c19 = get_crop(19, 144, 144)
    canvas.paste(_c19, (1248, 1686), _c19)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 1686, 1392, 1830]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_02_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-4/20_icon_Close_current_screen.png
try:
    _c20 = get_crop(20, 144, 144)
    canvas.paste(_c20, (1248, 1542), _c20)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 1542, 1392, 1686]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_02_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-4/21_icon_Home.png
try:
    _c21 = get_crop(21, 288, 156)
    canvas.paste(_c21, (0, 2804), _c21)
except Exception:
    pass
layout["Home"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_02_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-4/22_icon_Search_events.png
try:
    _c22 = get_crop(22, 288, 156)
    canvas.paste(_c22, (288, 2804), _c22)
except Exception:
    pass
layout["Search_events"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_02_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-4/23_icon_Food_Drink.png
try:
    _c23 = get_crop(23, 1344, 144)
    canvas.paste(_c23, (48, 678), _c23)
except Exception:
    pass
layout["Food_&_Drink"] = [48, 678, 1392, 822]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_02_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-4/24_icon_Festival.png
try:
    _c24 = get_crop(24, 1344, 144)
    canvas.paste(_c24, (48, 1542), _c24)
except Exception:
    pass
layout["Festival"] = [48, 1542, 1392, 1686]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_02_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-4/25_icon_Cancel.png
try:
    _c25 = get_crop(25, 144, 144)
    canvas.paste(_c25, (1248, 390), _c25)
except Exception:
    pass
layout["Cancel"] = [1248, 390, 1392, 534]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_02_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-4/26_icon_More.png
try:
    _c26 = get_crop(26, 288, 156)
    canvas.paste(_c26, (1152, 2804), _c26)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_02_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-4/27_icon_Food_Drink.png
try:
    _c27 = get_crop(27, 1344, 144)
    canvas.paste(_c27, (48, 822), _c27)
except Exception:
    pass
layout["Food_&_Drink"] = [48, 822, 1392, 966]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_02_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-4/28_icon_Close_current_screen.png
try:
    _c28 = get_crop(28, 144, 144)
    canvas.paste(_c28, (1248, 966), _c28)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 966, 1392, 1110]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_02_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-4/29_icon_Science_Tech.png
try:
    _c29 = get_crop(29, 1344, 144)
    canvas.paste(_c29, (48, 1110), _c29)
except Exception:
    pass
layout["Science_&_Tech"] = [48, 1110, 1392, 1254]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_02_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-4/30_icon_Basketball.png
try:
    _c30 = get_crop(30, 1344, 144)
    canvas.paste(_c30, (48, 1254), _c30)
except Exception:
    pass
layout["Basketball"] = [48, 1254, 1392, 1398]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_02_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-4/31_icon_Exhibition.png
try:
    _c31 = get_crop(31, 1344, 144)
    canvas.paste(_c31, (48, 1398), _c31)
except Exception:
    pass
layout["Exhibition"] = [48, 1398, 1392, 1542]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_02_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-4/32_icon_Search_for__..png
try:
    _c32 = get_crop(32, 48, 66)
    canvas.paste(_c32, (383, 2), _c32)
except Exception:
    pass
layout["[Search_for__."] = [383, 2, 431, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_02_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-4/33_icon_Science_Tech.png
try:
    _c33 = get_crop(33, 1344, 144)
    canvas.paste(_c33, (48, 966), _c33)
except Exception:
    pass
layout["Science_&_Tech"] = [48, 966, 1392, 1110]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_02_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-4/34_text_7.09.png
try:
    _c34 = get_crop(34, 89, 43)
    canvas.paste(_c34, (22, 17), _c34)
except Exception:
    pass
layout["7.09"] = [22, 17, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_02_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-4/35_text_Recent.png
try:
    _c35 = get_crop(35, 203, 62)
    canvas.paste(_c35, (45, 299), _c35)
except Exception:
    pass
layout["Recent"] = [45, 299, 248, 361]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_02_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-4/36_text_Taylor_Swift.png
try:
    _c36 = get_crop(36, 229, 57)
    canvas.paste(_c36, (161, 1734), _c36)
except Exception:
    pass
layout["Taylor_Swift"] = [161, 1734, 390, 1791]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_02_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-4/37_clickable_Taylor_Swift.png
try:
    _c37 = get_crop(37, 1344, 144)
    canvas.paste(_c37, (48, 1686), _c37)
except Exception:
    pass
layout["Taylor_Swift"] = [48, 1686, 1392, 1830]
