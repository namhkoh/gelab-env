# page_id: page_eventbrite_5059e61823f845cf95cbd1d8d54254c6_08
# screenshot: 2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-10.png
# step_index: 8/19
# task: Open Eventbrite. Look for 'Education' in Los Angeles happening on May 4. Filter to show only free events. How many events are posted?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Canvas and draw are provided.
w, h = canvas.size

# Colors
bg_white = "#FFFFFF"
status_gray = "#9EA0A2"     # status bar background
header_bg = "#FFFFFF"       # header/background area
divider_light = "#E9E9EC"   # subtle separators
primary_blue = "#2F5BE6"    # search underline / accent
card_bg = "#FBFBFD"         # slight off-white for group container
nav_divider = "#E6E6EA"     # bottom nav top line

# Fill overall background
draw.rectangle((0, 0, w, h), fill=bg_white)

# Status bar area (top)
status_height = 88
draw.rectangle((0, 0, w, status_height), fill=status_gray)

# Header / search area background (below status bar)
header_top = status_height
header_bottom = 280
draw.rectangle((0, header_top, w, header_bottom), fill=header_bg)

# Blue underline for the search area (divider under the search field)
underline_x0 = 48
underline_x1 = w - 48
underline_y = header_bottom - 20
draw.rectangle((underline_x0, underline_y, underline_x1, underline_y + 4), fill=primary_blue)

# Large rounded container behind the list of recent items
list_top = 300
list_bottom = 1780
list_left = 48
list_right = w - 48
draw.rounded_rectangle((list_left, list_top, list_right, list_bottom), radius=10, fill=card_bg, outline=None)

# Subtle separators between list rows (match detected row positions)
row_tops = [390, 534, 678, 822, 966, 1110, 1254, 1398, 1542, 1686]
for y in row_tops:
    # draw a light hairline across the content area
    draw.line((list_left + 24, y, list_right - 24, y), fill=divider_light, width=1)

# Bottom navigation bar area and top divider
nav_top = 2804
draw.line((0, nav_top, w, nav_top), fill=nav_divider, width=2)
draw.rectangle((0, nav_top, w, h), fill=bg_white)

# Subtle bottom padding divider (very light)
draw.line((list_left, list_bottom + 8, list_right, list_bottom + 8), fill=divider_light, width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_08_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-10/00_icon_icon_0.png
try:
    _c0 = get_crop(0, 50, 67)
    canvas.paste(_c0, (1154, 1), _c0)
except Exception:
    pass
layout["icon_0"] = [1154, 1, 1204, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_08_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-10/01_icon_7.35.png
try:
    _c1 = get_crop(1, 57, 61)
    canvas.paste(_c1, (181, 2), _c1)
except Exception:
    pass
layout["7.35"] = [181, 2, 238, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_08_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-10/02_icon_7.35.png
try:
    _c2 = get_crop(2, 58, 62)
    canvas.paste(_c2, (115, 2), _c2)
except Exception:
    pass
layout["7.35"] = [115, 2, 173, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_08_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-10/03_icon_Search_forae.png
try:
    _c3 = get_crop(3, 61, 61)
    canvas.paste(_c3, (310, 3), _c3)
except Exception:
    pass
layout["Search_forae"] = [310, 3, 371, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_08_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-10/04_icon_icon_4.png
try:
    _c4 = get_crop(4, 47, 58)
    canvas.paste(_c4, (250, 4), _c4)
except Exception:
    pass
layout["icon_4"] = [250, 4, 297, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_08_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-10/05_icon_Cancel.png
try:
    _c5 = get_crop(5, 97, 64)
    canvas.paste(_c5, (1212, 1), _c5)
except Exception:
    pass
layout["Cancel"] = [1212, 1, 1309, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_08_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-10/06_icon_Cancel.png
try:
    _c6 = get_crop(6, 149, 144)
    canvas.paste(_c6, (1243, 97), _c6)
except Exception:
    pass
layout["Cancel"] = [1243, 97, 1392, 241]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_08_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-10/07_icon_Cancel.png
try:
    _c7 = get_crop(7, 54, 63)
    canvas.paste(_c7, (1318, 1), _c7)
except Exception:
    pass
layout["Cancel"] = [1318, 1, 1372, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_08_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-10/08_icon_Coding_Workshop.png
try:
    _c8 = get_crop(8, 1344, 144)
    canvas.paste(_c8, (48, 534), _c8)
except Exception:
    pass
layout["Coding_Workshop"] = [48, 534, 1392, 678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_08_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-10/09_icon_Tickets.png
try:
    _c9 = get_crop(9, 288, 156)
    canvas.paste(_c9, (864, 2804), _c9)
except Exception:
    pass
layout["Tickets"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_08_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-10/10_icon_Coding_Workshop.png
try:
    _c10 = get_crop(10, 1344, 144)
    canvas.paste(_c10, (48, 390), _c10)
except Exception:
    pass
layout["Coding_Workshop"] = [48, 390, 1392, 534]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_08_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-10/11_icon_Food_Drink.png
try:
    _c11 = get_crop(11, 1344, 144)
    canvas.paste(_c11, (48, 1398), _c11)
except Exception:
    pass
layout["Food_&_Drink"] = [48, 1398, 1392, 1542]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_08_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-10/12_icon_7.35.png
try:
    _c12 = get_crop(12, 121, 111)
    canvas.paste(_c12, (57, 116), _c12)
except Exception:
    pass
layout["7.35"] = [57, 116, 178, 227]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_08_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-10/13_icon_Close_current_screen.png
try:
    _c13 = get_crop(13, 144, 144)
    canvas.paste(_c13, (1248, 1254), _c13)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 1254, 1392, 1398]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_08_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-10/14_icon_Close_current_screen.png
try:
    _c14 = get_crop(14, 144, 144)
    canvas.paste(_c14, (1248, 822), _c14)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 822, 1392, 966]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_08_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-10/15_icon_Close_current_screen.png
try:
    _c15 = get_crop(15, 144, 144)
    canvas.paste(_c15, (1248, 1398), _c15)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 1398, 1392, 1542]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_08_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-10/16_icon_Science_Tech.png
try:
    _c16 = get_crop(16, 1344, 144)
    canvas.paste(_c16, (48, 1542), _c16)
except Exception:
    pass
layout["Science_&_Tech"] = [48, 1542, 1392, 1686]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_08_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-10/17_icon_Close_current_screen.png
try:
    _c17 = get_crop(17, 144, 144)
    canvas.paste(_c17, (1248, 1110), _c17)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 1110, 1392, 1254]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_08_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-10/18_icon_Close_current_screen.png
try:
    _c18 = get_crop(18, 144, 144)
    canvas.paste(_c18, (1248, 1686), _c18)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 1686, 1392, 1830]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_08_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-10/19_icon_Close_current_screen.png
try:
    _c19 = get_crop(19, 144, 144)
    canvas.paste(_c19, (1248, 678), _c19)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 678, 1392, 822]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_08_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-10/20_icon_Close_current_screen.png
try:
    _c20 = get_crop(20, 144, 144)
    canvas.paste(_c20, (1248, 534), _c20)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 534, 1392, 678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_08_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-10/21_icon_Show.png
try:
    _c21 = get_crop(21, 116, 132)
    canvas.paste(_c21, (26, 1696), _c21)
except Exception:
    pass
layout["Show"] = [26, 1696, 142, 1828]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_08_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-10/22_icon_Close_current_screen.png
try:
    _c22 = get_crop(22, 144, 144)
    canvas.paste(_c22, (1248, 1542), _c22)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 1542, 1392, 1686]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_08_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-10/23_icon_Close_current_screen.png
try:
    _c23 = get_crop(23, 144, 144)
    canvas.paste(_c23, (1248, 966), _c23)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 966, 1392, 1110]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_08_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-10/24_icon_Cancel.png
try:
    _c24 = get_crop(24, 144, 144)
    canvas.paste(_c24, (1248, 390), _c24)
except Exception:
    pass
layout["Cancel"] = [1248, 390, 1392, 534]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_08_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-10/25_icon_Favorites.png
try:
    _c25 = get_crop(25, 288, 156)
    canvas.paste(_c25, (576, 2804), _c25)
except Exception:
    pass
layout["Favorites"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_08_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-10/26_icon_Food_Drink.png
try:
    _c26 = get_crop(26, 1344, 144)
    canvas.paste(_c26, (48, 1254), _c26)
except Exception:
    pass
layout["Food_&_Drink"] = [48, 1254, 1392, 1398]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_08_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-10/27_icon_Coding_Workshop.png
try:
    _c27 = get_crop(27, 1344, 144)
    canvas.paste(_c27, (48, 678), _c27)
except Exception:
    pass
layout["Coding_Workshop"] = [48, 678, 1392, 822]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_08_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-10/28_icon_Home.png
try:
    _c28 = get_crop(28, 288, 156)
    canvas.paste(_c28, (0, 2804), _c28)
except Exception:
    pass
layout["Home"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_08_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-10/29_icon_Search_events.png
try:
    _c29 = get_crop(29, 288, 156)
    canvas.paste(_c29, (288, 2804), _c29)
except Exception:
    pass
layout["Search_events"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_08_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-10/30_icon_Search_forae.png
try:
    _c30 = get_crop(30, 1344, 191)
    canvas.paste(_c30, (48, 72), _c30)
except Exception:
    pass
layout["Search_forae"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_08_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-10/31_icon_7.35.png
try:
    _c31 = get_crop(31, 92, 62)
    canvas.paste(_c31, (15, 2), _c31)
except Exception:
    pass
layout["7.35"] = [15, 2, 107, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_08_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-10/32_icon_More.png
try:
    _c32 = get_crop(32, 288, 156)
    canvas.paste(_c32, (1152, 2804), _c32)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_08_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-10/33_icon_Education.png
try:
    _c33 = get_crop(33, 1344, 144)
    canvas.paste(_c33, (48, 1110), _c33)
except Exception:
    pass
layout["Education"] = [48, 1110, 1392, 1254]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_08_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-10/34_icon_Search_forae.png
try:
    _c34 = get_crop(34, 47, 62)
    canvas.paste(_c34, (383, 3), _c34)
except Exception:
    pass
layout["Search_forae"] = [383, 3, 430, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_08_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-10/35_icon_Education.png
try:
    _c35 = get_crop(35, 1344, 144)
    canvas.paste(_c35, (48, 966), _c35)
except Exception:
    pass
layout["Education"] = [48, 966, 1392, 1110]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_08_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-10/36_icon_Music_Festival.png
try:
    _c36 = get_crop(36, 1344, 144)
    canvas.paste(_c36, (48, 822), _c36)
except Exception:
    pass
layout["Music_Festival"] = [48, 822, 1392, 966]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_08_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-10/37_text_Recent.png
try:
    _c37 = get_crop(37, 200, 56)
    canvas.paste(_c37, (46, 301), _c37)
except Exception:
    pass
layout["Recent"] = [46, 301, 246, 357]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_08_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-10/38_text_Show.png
try:
    _c38 = get_crop(38, 112, 43)
    canvas.paste(_c38, (163, 1740), _c38)
except Exception:
    pass
layout["Show"] = [163, 1740, 275, 1783]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_08_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-10/39_clickable_Show.png
try:
    _c39 = get_crop(39, 1344, 144)
    canvas.paste(_c39, (48, 1686), _c39)
except Exception:
    pass
layout["Show"] = [48, 1686, 1392, 1830]
