# page_id: page_eventbrite_5362d15941a843c5a00f4a85c7ff0a9a_07
# screenshot: 2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-9.png
# step_index: 7/12
# task: Open Eventbrite. Set the city to 'Los Angeles'. Search 'Business'. Filter 'French' speaking events. Add the first event to favorite.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Paint UI background and structural elements for the given canvas.
# Available globals: canvas (PIL Image), draw (ImageDraw), font_sm, font_md, font_lg, font_xl

# Colors
status_bar_color = (200, 200, 200)      # light grey for status bar
header_bg = (255, 255, 255)             # white header area
divider_light = (235, 235, 240)         # very light divider for list rows
divider_subtle = (245, 245, 248)        # subtle divider under toolbars
nav_border = (225, 225, 230)            # nav top border

# 1) Status bar area (top ~64px)
draw.rectangle([(0, 0), (1440, 64)], fill=status_bar_color)

# 2) Header / toolbar background below status bar
draw.rectangle([(0, 64), (1440, 220)], fill=header_bg)
# subtle divider under header (do not duplicate detected search visuals)
draw.line([(0, 220), (1440, 220)], fill=divider_subtle, width=1)

# 3) Main list/group background (rounded container behind rows)
# Keep a light inset so pasted rows/icons/text align above it
list_container_bbox = (36, 510, 1404, 1860)
try:
    draw.rounded_rectangle(list_container_bbox, radius=12, fill=(255, 255, 255))
except Exception:
    # Fallback if rounded_rectangle isn't available
    draw.rectangle(list_container_bbox, fill=(255, 255, 255))

# 4) Row separator lines for the list (do not draw any icons/text)
row_tops = [534, 678, 822, 966, 1110, 1254, 1398, 1542, 1686, 1830]
for y in row_tops:
    draw.line([(48, y), (1392, y)], fill=divider_light, width=1)

# 5) Additional faint divider below the top search area (across content width)
draw.line([(48, 390), (1392, 390)], fill=divider_subtle, width=1)

# 6) Bottom navigation bar background and top border
nav_top = 2804
draw.rectangle([(0, nav_top), (1440, 2960)], fill=(255, 255, 255))
draw.line([(0, nav_top), (1440, nav_top)], fill=nav_border, width=2)

# 7) Subtle page bottom edge line
draw.line([(0, 2958), (1440, 2958)], fill=(245, 245, 248), width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_07_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-9/00_icon_icon_0.png
try:
    _c0 = get_crop(0, 50, 67)
    canvas.paste(_c0, (1154, 1), _c0)
except Exception:
    pass
layout["icon_0"] = [1154, 1, 1204, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_07_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-9/01_icon_8.02.png
try:
    _c1 = get_crop(1, 60, 64)
    canvas.paste(_c1, (114, 1), _c1)
except Exception:
    pass
layout["8.02"] = [114, 1, 174, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_07_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-9/02_icon_8.02.png
try:
    _c2 = get_crop(2, 58, 63)
    canvas.paste(_c2, (181, 1), _c2)
except Exception:
    pass
layout["8.02"] = [181, 1, 239, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_07_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-9/03_icon_Search_for__..png
try:
    _c3 = get_crop(3, 65, 64)
    canvas.paste(_c3, (308, 1), _c3)
except Exception:
    pass
layout["[Search_for__."] = [308, 1, 373, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_07_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-9/04_icon_Cancel.png
try:
    _c4 = get_crop(4, 100, 64)
    canvas.paste(_c4, (1212, 0), _c4)
except Exception:
    pass
layout["Cancel"] = [1212, 0, 1312, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_07_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-9/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 49, 62)
    canvas.paste(_c5, (249, 2), _c5)
except Exception:
    pass
layout["icon_5"] = [249, 2, 298, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_07_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-9/06_icon_Cancel.png
try:
    _c6 = get_crop(6, 56, 62)
    canvas.paste(_c6, (1317, 0), _c6)
except Exception:
    pass
layout["Cancel"] = [1317, 0, 1373, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_07_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-9/07_icon_Cancel.png
try:
    _c7 = get_crop(7, 149, 144)
    canvas.paste(_c7, (1243, 97), _c7)
except Exception:
    pass
layout["Cancel"] = [1243, 97, 1392, 241]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_07_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-9/08_icon_Art.png
try:
    _c8 = get_crop(8, 1344, 144)
    canvas.paste(_c8, (48, 534), _c8)
except Exception:
    pass
layout["Art"] = [48, 534, 1392, 678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_07_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-9/09_icon_8.02.png
try:
    _c9 = get_crop(9, 126, 109)
    canvas.paste(_c9, (52, 114), _c9)
except Exception:
    pass
layout["8.02"] = [52, 114, 178, 223]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_07_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-9/10_icon_Tickets.png
try:
    _c10 = get_crop(10, 288, 156)
    canvas.paste(_c10, (864, 2804), _c10)
except Exception:
    pass
layout["Tickets"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_07_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-9/11_icon_Search_for__..png
try:
    _c11 = get_crop(11, 1344, 191)
    canvas.paste(_c11, (48, 72), _c11)
except Exception:
    pass
layout["[Search_for__."] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_07_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-9/12_icon_Close_current_screen.png
try:
    _c12 = get_crop(12, 144, 144)
    canvas.paste(_c12, (1248, 822), _c12)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 822, 1392, 966]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_07_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-9/13_icon_Close_current_screen.png
try:
    _c13 = get_crop(13, 144, 144)
    canvas.paste(_c13, (1248, 534), _c13)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 534, 1392, 678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_07_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-9/14_icon_Favorites.png
try:
    _c14 = get_crop(14, 288, 156)
    canvas.paste(_c14, (576, 2804), _c14)
except Exception:
    pass
layout["Favorites"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_07_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-9/15_icon_Close_current_screen.png
try:
    _c15 = get_crop(15, 144, 144)
    canvas.paste(_c15, (1248, 678), _c15)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 678, 1392, 822]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_07_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-9/16_icon_Close_current_screen.png
try:
    _c16 = get_crop(16, 144, 144)
    canvas.paste(_c16, (1248, 1254), _c16)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 1254, 1392, 1398]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_07_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-9/17_icon_Close_current_screen.png
try:
    _c17 = get_crop(17, 144, 144)
    canvas.paste(_c17, (1248, 1398), _c17)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 1398, 1392, 1542]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_07_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-9/18_icon_Close_current_screen.png
try:
    _c18 = get_crop(18, 144, 144)
    canvas.paste(_c18, (1248, 1686), _c18)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 1686, 1392, 1830]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_07_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-9/19_icon_Close_current_screen.png
try:
    _c19 = get_crop(19, 144, 144)
    canvas.paste(_c19, (1248, 1110), _c19)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 1110, 1392, 1254]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_07_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-9/20_icon_Close_current_screen.png
try:
    _c20 = get_crop(20, 144, 144)
    canvas.paste(_c20, (1248, 1542), _c20)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 1542, 1392, 1686]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_07_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-9/21_icon_Home.png
try:
    _c21 = get_crop(21, 288, 156)
    canvas.paste(_c21, (0, 2804), _c21)
except Exception:
    pass
layout["Home"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_07_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-9/22_icon_Search_events.png
try:
    _c22 = get_crop(22, 288, 156)
    canvas.paste(_c22, (288, 2804), _c22)
except Exception:
    pass
layout["Search_events"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_07_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-9/23_icon_Search_for__..png
try:
    _c23 = get_crop(23, 1344, 144)
    canvas.paste(_c23, (48, 390), _c23)
except Exception:
    pass
layout["[Search_for__."] = [48, 390, 1392, 534]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_07_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-9/24_icon_Cancel.png
try:
    _c24 = get_crop(24, 144, 144)
    canvas.paste(_c24, (1248, 390), _c24)
except Exception:
    pass
layout["Cancel"] = [1248, 390, 1392, 534]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_07_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-9/25_icon_Food_and_Drink.png
try:
    _c25 = get_crop(25, 1344, 144)
    canvas.paste(_c25, (48, 678), _c25)
except Exception:
    pass
layout["Food_and_Drink"] = [48, 678, 1392, 822]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_07_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-9/26_icon_Close_current_screen.png
try:
    _c26 = get_crop(26, 144, 144)
    canvas.paste(_c26, (1248, 966), _c26)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 966, 1392, 1110]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_07_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-9/27_icon_More.png
try:
    _c27 = get_crop(27, 288, 156)
    canvas.paste(_c27, (1152, 2804), _c27)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_07_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-9/28_icon_Food_Drink.png
try:
    _c28 = get_crop(28, 1344, 144)
    canvas.paste(_c28, (48, 1542), _c28)
except Exception:
    pass
layout["Food_&_Drink"] = [48, 1542, 1392, 1686]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_07_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-9/29_icon_Music_Festival.png
try:
    _c29 = get_crop(29, 1344, 144)
    canvas.paste(_c29, (48, 1398), _c29)
except Exception:
    pass
layout["Music_Festival"] = [48, 1398, 1392, 1542]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_07_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-9/30_icon_Search_for__..png
try:
    _c30 = get_crop(30, 48, 65)
    canvas.paste(_c30, (383, 2), _c30)
except Exception:
    pass
layout["[Search_for__."] = [383, 2, 431, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_07_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-9/31_icon_Science_Tech.png
try:
    _c31 = get_crop(31, 112, 128)
    canvas.paste(_c31, (27, 1698), _c31)
except Exception:
    pass
layout["Science_&_Tech"] = [27, 1698, 139, 1826]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_07_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-9/32_text_8.02.png
try:
    _c32 = get_crop(32, 91, 43)
    canvas.paste(_c32, (20, 17), _c32)
except Exception:
    pass
layout["8.02"] = [20, 17, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_07_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-9/33_text_Education.png
try:
    _c33 = get_crop(33, 195, 50)
    canvas.paste(_c33, (162, 872), _c33)
except Exception:
    pass
layout["Education"] = [162, 872, 357, 922]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_07_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-9/34_text_session.png
try:
    _c34 = get_crop(34, 148, 43)
    canvas.paste(_c34, (262, 1021), _c34)
except Exception:
    pass
layout["session"] = [262, 1021, 410, 1064]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_07_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-9/35_text_Coding_Workshop.png
try:
    _c35 = get_crop(35, 1344, 144)
    canvas.paste(_c35, (48, 1110), _c35)
except Exception:
    pass
layout["Coding_Workshop"] = [48, 1110, 1392, 1254]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_07_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-9/36_text_Music_Festival.png
try:
    _c36 = get_crop(36, 272, 48)
    canvas.paste(_c36, (164, 1304), _c36)
except Exception:
    pass
layout["Music_Festival"] = [164, 1304, 436, 1352]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_07_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-9/37_text_Science_Tech.png
try:
    _c37 = get_crop(37, 292, 49)
    canvas.paste(_c37, (160, 1735), _c37)
except Exception:
    pass
layout["Science_&_Tech"] = [160, 1735, 452, 1784]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_07_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-9/38_clickable_Education.png
try:
    _c38 = get_crop(38, 1344, 144)
    canvas.paste(_c38, (48, 822), _c38)
except Exception:
    pass
layout["Education"] = [48, 822, 1392, 966]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_07_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-9/39_clickable_Yoga_session.png
try:
    _c39 = get_crop(39, 1344, 144)
    canvas.paste(_c39, (48, 966), _c39)
except Exception:
    pass
layout["Yoga_session"] = [48, 966, 1392, 1110]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_07_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-9/40_clickable_Music_Festival.png
try:
    _c40 = get_crop(40, 1344, 144)
    canvas.paste(_c40, (48, 1254), _c40)
except Exception:
    pass
layout["Music_Festival"] = [48, 1254, 1392, 1398]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_07_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-9/41_clickable_Science_Tech.png
try:
    _c41 = get_crop(41, 1344, 144)
    canvas.paste(_c41, (48, 1686), _c41)
except Exception:
    pass
layout["Science_&_Tech"] = [48, 1686, 1392, 1830]
