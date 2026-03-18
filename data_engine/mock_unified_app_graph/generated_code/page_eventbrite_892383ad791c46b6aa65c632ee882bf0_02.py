# page_id: page_eventbrite_892383ad791c46b6aa65c632ee882bf0_02
# screenshot: 2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-4.png
# step_index: 2/12
# task: Open Eventbrite. Search for online "Music" events happening next weekend.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Top status bar
draw.rectangle([(0, 0), (1440, 56)], fill="#c0c0c0")

# Header / search area background (subtle off-white)
draw.rectangle([(0, 56), (1440, 156)], fill="#fbfbfc")

# Active search underline (material blue)
underline_y = 150
draw.rectangle([(48, underline_y - 3), (1392, underline_y + 3)], fill="#2E55FF")

# Main list container (rounded) background with subtle border
container_x0, container_y0 = 32, 280
container_x1, container_y1 = 1408, 1900
draw.rounded_rectangle([(container_x0, container_y0), (container_x1, container_y1)],
                       radius=12, fill="#ffffff", outline="#ececee", width=1)

# Draw separators for each list row (using detected top positions + row height 144)
row_tops = [390, 534, 678, 822, 966, 1110, 1254, 1398, 1542, 1686]
row_height = 144
sep_color = "#ecebf0"
for top in row_tops:
    y = top + row_height
    # limit separators to be inside container
    if container_y0 + 10 < y < container_y1 - 10:
        draw.line([(container_x0 + 12, y), (container_x1 - 12, y)], fill=sep_color, width=1)

# Thin divider under Recent/search header area
draw.line([(32, 276), (1408, 276)], fill="#e0e0e5", width=1)

# Bottom navigation bar background and top divider
nav_top = 2804
draw.line([(0, nav_top), (1440, nav_top)], fill="#e6e6ef", width=2)
draw.rectangle([(0, nav_top), (1440, 2960)], fill="#ffffff")

# Subtle left and right page margins (visual rail)
rail_color = "#fafbfd"
draw.rectangle([(0, 156), (32, 2800)], fill=rail_color)
draw.rectangle([(1408, 156), (1440, 2800)], fill=rail_color)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_02_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-4/00_icon_5.22.png
try:
    _c0 = get_crop(0, 59, 61)
    canvas.paste(_c0, (180, 2), _c0)
except Exception:
    pass
layout["5.22"] = [180, 2, 239, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_02_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-4/01_icon_5.22.png
try:
    _c1 = get_crop(1, 59, 62)
    canvas.paste(_c1, (114, 2), _c1)
except Exception:
    pass
layout["5.22"] = [114, 2, 173, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_02_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-4/02_icon_Search_forae.png
try:
    _c2 = get_crop(2, 62, 62)
    canvas.paste(_c2, (310, 2), _c2)
except Exception:
    pass
layout["Search_forae"] = [310, 2, 372, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_02_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-4/03_icon_icon_3.png
try:
    _c3 = get_crop(3, 48, 60)
    canvas.paste(_c3, (250, 3), _c3)
except Exception:
    pass
layout["icon_3"] = [250, 3, 298, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_02_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-4/04_icon_Language_Learning.png
try:
    _c4 = get_crop(4, 1344, 144)
    canvas.paste(_c4, (48, 534), _c4)
except Exception:
    pass
layout["Language_Learning"] = [48, 534, 1392, 678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_02_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-4/05_icon_Cancel.png
try:
    _c5 = get_crop(5, 149, 144)
    canvas.paste(_c5, (1243, 97), _c5)
except Exception:
    pass
layout["Cancel"] = [1243, 97, 1392, 241]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_02_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-4/06_icon_Cancel.png
try:
    _c6 = get_crop(6, 56, 63)
    canvas.paste(_c6, (1317, 0), _c6)
except Exception:
    pass
layout["Cancel"] = [1317, 0, 1373, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_02_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-4/07_icon_Photography.png
try:
    _c7 = get_crop(7, 1344, 144)
    canvas.paste(_c7, (48, 1398), _c7)
except Exception:
    pass
layout["Photography"] = [48, 1398, 1392, 1542]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_02_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-4/08_icon_Cancel.png
try:
    _c8 = get_crop(8, 97, 63)
    canvas.paste(_c8, (1212, 0), _c8)
except Exception:
    pass
layout["Cancel"] = [1212, 0, 1309, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_02_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-4/09_icon_Open_Mic_Night.png
try:
    _c9 = get_crop(9, 1344, 144)
    canvas.paste(_c9, (48, 1254), _c9)
except Exception:
    pass
layout["Open_Mic_Night"] = [48, 1254, 1392, 1398]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_02_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-4/10_icon_Language_Learning.png
try:
    _c10 = get_crop(10, 1344, 144)
    canvas.paste(_c10, (48, 678), _c10)
except Exception:
    pass
layout["Language_Learning"] = [48, 678, 1392, 822]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_02_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-4/11_icon_Open_Mic_Night.png
try:
    _c11 = get_crop(11, 1344, 144)
    canvas.paste(_c11, (48, 1110), _c11)
except Exception:
    pass
layout["Open_Mic_Night"] = [48, 1110, 1392, 1254]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_02_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-4/12_icon_5.22.png
try:
    _c12 = get_crop(12, 120, 112)
    canvas.paste(_c12, (57, 116), _c12)
except Exception:
    pass
layout["5.22"] = [57, 116, 177, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_02_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-4/13_icon_Tickets.png
try:
    _c13 = get_crop(13, 288, 156)
    canvas.paste(_c13, (864, 2804), _c13)
except Exception:
    pass
layout["Tickets"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_02_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-4/14_icon_Close_current_screen.png
try:
    _c14 = get_crop(14, 144, 144)
    canvas.paste(_c14, (1248, 822), _c14)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 822, 1392, 966]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_02_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-4/15_icon_5.22.png
try:
    _c15 = get_crop(15, 93, 62)
    canvas.paste(_c15, (15, 1), _c15)
except Exception:
    pass
layout["5.22"] = [15, 1, 108, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_02_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-4/16_icon_Business_Seminar.png
try:
    _c16 = get_crop(16, 1344, 144)
    canvas.paste(_c16, (48, 390), _c16)
except Exception:
    pass
layout["Business_Seminar"] = [48, 390, 1392, 534]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_02_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-4/17_icon_Close_current_screen.png
try:
    _c17 = get_crop(17, 144, 144)
    canvas.paste(_c17, (1248, 678), _c17)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 678, 1392, 822]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_02_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-4/18_icon_Wellness.png
try:
    _c18 = get_crop(18, 1344, 144)
    canvas.paste(_c18, (48, 1542), _c18)
except Exception:
    pass
layout["Wellness"] = [48, 1542, 1392, 1686]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_02_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-4/19_icon_Close_current_screen.png
try:
    _c19 = get_crop(19, 144, 144)
    canvas.paste(_c19, (1248, 534), _c19)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 534, 1392, 678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_02_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-4/20_icon_Close_current_screen.png
try:
    _c20 = get_crop(20, 144, 144)
    canvas.paste(_c20, (1248, 1398), _c20)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 1398, 1392, 1542]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_02_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-4/21_icon_Close_current_screen.png
try:
    _c21 = get_crop(21, 144, 144)
    canvas.paste(_c21, (1248, 1686), _c21)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 1686, 1392, 1830]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_02_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-4/22_icon_Close_current_screen.png
try:
    _c22 = get_crop(22, 144, 144)
    canvas.paste(_c22, (1248, 1110), _c22)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 1110, 1392, 1254]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_02_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-4/23_icon_Close_current_screen.png
try:
    _c23 = get_crop(23, 144, 144)
    canvas.paste(_c23, (1248, 1254), _c23)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 1254, 1392, 1398]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_02_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-4/24_icon_Favorites.png
try:
    _c24 = get_crop(24, 288, 156)
    canvas.paste(_c24, (576, 2804), _c24)
except Exception:
    pass
layout["Favorites"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_02_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-4/25_icon_Close_current_screen.png
try:
    _c25 = get_crop(25, 144, 144)
    canvas.paste(_c25, (1248, 1542), _c25)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 1542, 1392, 1686]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_02_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-4/26_icon_Cancel.png
try:
    _c26 = get_crop(26, 144, 144)
    canvas.paste(_c26, (1248, 390), _c26)
except Exception:
    pass
layout["Cancel"] = [1248, 390, 1392, 534]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_02_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-4/27_icon_Cooking.png
try:
    _c27 = get_crop(27, 117, 130)
    canvas.paste(_c27, (25, 1696), _c27)
except Exception:
    pass
layout["Cooking"] = [25, 1696, 142, 1826]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_02_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-4/28_icon_Open_Mic_Night.png
try:
    _c28 = get_crop(28, 1344, 144)
    canvas.paste(_c28, (48, 966), _c28)
except Exception:
    pass
layout["Open_Mic_Night"] = [48, 966, 1392, 1110]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_02_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-4/29_icon_Close_current_screen.png
try:
    _c29 = get_crop(29, 144, 144)
    canvas.paste(_c29, (1248, 966), _c29)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 966, 1392, 1110]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_02_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-4/30_icon_Gardening.png
try:
    _c30 = get_crop(30, 1344, 144)
    canvas.paste(_c30, (48, 822), _c30)
except Exception:
    pass
layout["Gardening"] = [48, 822, 1392, 966]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_02_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-4/31_icon_Home.png
try:
    _c31 = get_crop(31, 288, 156)
    canvas.paste(_c31, (0, 2804), _c31)
except Exception:
    pass
layout["Home"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_02_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-4/32_icon_Search_forae.png
try:
    _c32 = get_crop(32, 1344, 191)
    canvas.paste(_c32, (48, 72), _c32)
except Exception:
    pass
layout["Search_forae"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_02_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-4/33_icon_Search_events.png
try:
    _c33 = get_crop(33, 288, 156)
    canvas.paste(_c33, (288, 2804), _c33)
except Exception:
    pass
layout["Search_events"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_02_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-4/34_icon_Search_forae.png
try:
    _c34 = get_crop(34, 47, 63)
    canvas.paste(_c34, (383, 3), _c34)
except Exception:
    pass
layout["Search_forae"] = [383, 3, 430, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_02_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-4/35_icon_More.png
try:
    _c35 = get_crop(35, 288, 156)
    canvas.paste(_c35, (1152, 2804), _c35)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_02_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-4/36_text_Recent.png
try:
    _c36 = get_crop(36, 200, 56)
    canvas.paste(_c36, (46, 301), _c36)
except Exception:
    pass
layout["Recent"] = [46, 301, 246, 357]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_02_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-4/37_text_Cooking.png
try:
    _c37 = get_crop(37, 167, 60)
    canvas.paste(_c37, (160, 1733), _c37)
except Exception:
    pass
layout["Cooking"] = [160, 1733, 327, 1793]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_02_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-4/38_clickable_Cooking.png
try:
    _c38 = get_crop(38, 1344, 144)
    canvas.paste(_c38, (48, 1686), _c38)
except Exception:
    pass
layout["Cooking"] = [48, 1686, 1392, 1830]
