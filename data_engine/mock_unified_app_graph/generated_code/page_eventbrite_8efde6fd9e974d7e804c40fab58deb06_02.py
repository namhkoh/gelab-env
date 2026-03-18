# page_id: page_eventbrite_8efde6fd9e974d7e804c40fab58deb06_02
# screenshot: 2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-4.png
# step_index: 2/8
# task: Open Eventbrite. Search for "Education". Filter only online events. Note how many events are available.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw overall background (canvas already provided as white, but ensure fill)
draw.rectangle([(0, 0), (1440, 2960)], fill="#ffffff")

# Status bar area (top strip)
status_h = 72
draw.rectangle([(0, 0), (1440, status_h)], fill="#bfbfbf")  # neutral gray status bar

# Subtle header area under status bar (search area background)
search_area_top = 72
search_area_height = 191  # follows detected search crop at y=72 height=191
draw.rectangle([(0, search_area_top), (1440, search_area_top + search_area_height)], fill="#ffffff")

# Blue underline beneath the search field (prominent accent)
underline_y = search_area_top + search_area_height  # place at bottom of search area
underline_thickness = 6
draw.rectangle([(48, underline_y - underline_thickness//2), (1440-48, underline_y + underline_thickness//2)],
               fill="#2D5AE8")

# Light divider immediately below the search area
draw.line([(48, underline_y + 10), (1440-48, underline_y + 10)], fill="#e6e6e6", width=1)

# Card / list background area (subtle slightly off-white to separate from page)
list_top = underline_y + 24
list_bottom = 1760
list_left = 24
list_right = 1440 - 24
# rounded rectangle background for the list region
try:
    draw.rounded_rectangle([(list_left, list_top), (list_right, list_bottom)], radius=8, fill="#ffffff", outline=None)
except Exception:
    # Fallback if rounded_rectangle not available
    draw.rectangle([(list_left, list_top), (list_right, list_bottom)], fill="#ffffff")

# Draw separators for each list row (use detected row Y positions as guides)
row_tops = [534, 678, 822, 966, 1110, 1254, 1398, 1542, 1686]
sep_left = 48
sep_right = 1440 - 48
for y in row_tops:
    # top hairline (very light)
    draw.line([(sep_left, y), (sep_right, y)], fill="#efeff2", width=1)
    # bottom hairline for the row to enhance separation (at y + row height)
    # The detected rows are 144px tall; draw bottom divider too
    bottom_y = y + 144
    draw.line([(sep_left, bottom_y), (sep_right, bottom_y)], fill="#f4f4f6", width=1)

# Small section divider under the "Recent" header area
recent_header_bottom = 360
draw.line([(48, recent_header_bottom), (1440-48, recent_header_bottom)], fill="#f0f0f2", width=1)

# Bottom navigation background and top border
nav_top = 2804
nav_bottom = 2960
draw.rectangle([(0, nav_top), (1440, nav_bottom)], fill="#fafafa")
# top border of nav
draw.line([(0, nav_top), (1440, nav_top)], fill="#e6e6e9", width=2)

# Subtle left/right page margins (visual guides)
draw.line([(24, status_h), (24, 2960 - 156)], fill="#ffffff", width=1)
draw.line([(1440-24, status_h), (1440-24, 2960 - 156)], fill="#ffffff", width=1)

# faint bottom edge
draw.line([(0, 2959), (1440, 2959)], fill="#e9e9ec", width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_02_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-4/00_icon_6.58.png
try:
    _c0 = get_crop(0, 60, 64)
    canvas.paste(_c0, (114, 1), _c0)
except Exception:
    pass
layout["6.58"] = [114, 1, 174, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_02_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-4/01_icon_6.58.png
try:
    _c1 = get_crop(1, 58, 63)
    canvas.paste(_c1, (181, 1), _c1)
except Exception:
    pass
layout["6.58"] = [181, 1, 239, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_02_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-4/02_icon_Search_for__..png
try:
    _c2 = get_crop(2, 65, 64)
    canvas.paste(_c2, (308, 1), _c2)
except Exception:
    pass
layout["[Search_for__."] = [308, 1, 373, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_02_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-4/03_icon_icon_3.png
try:
    _c3 = get_crop(3, 51, 62)
    canvas.paste(_c3, (248, 2), _c3)
except Exception:
    pass
layout["icon_3"] = [248, 2, 299, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_02_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-4/04_icon_Cancel.png
try:
    _c4 = get_crop(4, 149, 144)
    canvas.paste(_c4, (1243, 97), _c4)
except Exception:
    pass
layout["Cancel"] = [1243, 97, 1392, 241]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_02_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-4/05_icon_Cancel.png
try:
    _c5 = get_crop(5, 99, 62)
    canvas.paste(_c5, (1212, 0), _c5)
except Exception:
    pass
layout["Cancel"] = [1212, 0, 1311, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_02_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-4/06_icon_Cancel.png
try:
    _c6 = get_crop(6, 58, 63)
    canvas.paste(_c6, (1316, 0), _c6)
except Exception:
    pass
layout["Cancel"] = [1316, 0, 1374, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_02_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-4/07_icon_Food_Drink.png
try:
    _c7 = get_crop(7, 1344, 144)
    canvas.paste(_c7, (48, 534), _c7)
except Exception:
    pass
layout["Food_&_Drink"] = [48, 534, 1392, 678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_02_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-4/08_icon_Favorites.png
try:
    _c8 = get_crop(8, 288, 156)
    canvas.paste(_c8, (576, 2804), _c8)
except Exception:
    pass
layout["Favorites"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_02_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-4/09_icon_Search_for__..png
try:
    _c9 = get_crop(9, 1344, 144)
    canvas.paste(_c9, (48, 390), _c9)
except Exception:
    pass
layout["[Search_for__."] = [48, 390, 1392, 534]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_02_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-4/10_icon_Close_current_screen.png
try:
    _c10 = get_crop(10, 144, 144)
    canvas.paste(_c10, (1248, 822), _c10)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 822, 1392, 966]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_02_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-4/11_icon_Close_current_screen.png
try:
    _c11 = get_crop(11, 144, 144)
    canvas.paste(_c11, (1248, 534), _c11)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 534, 1392, 678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_02_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-4/12_icon_Science_Tech.png
try:
    _c12 = get_crop(12, 1344, 144)
    canvas.paste(_c12, (48, 678), _c12)
except Exception:
    pass
layout["Science_&_Tech"] = [48, 678, 1392, 822]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_02_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-4/13_icon_Close_current_screen.png
try:
    _c13 = get_crop(13, 144, 144)
    canvas.paste(_c13, (1248, 1254), _c13)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 1254, 1392, 1398]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_02_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-4/14_icon_Close_current_screen.png
try:
    _c14 = get_crop(14, 144, 144)
    canvas.paste(_c14, (1248, 1398), _c14)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 1398, 1392, 1542]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_02_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-4/15_icon_6.58.png
try:
    _c15 = get_crop(15, 126, 109)
    canvas.paste(_c15, (52, 114), _c15)
except Exception:
    pass
layout["6.58"] = [52, 114, 178, 223]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_02_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-4/16_icon_Close_current_screen.png
try:
    _c16 = get_crop(16, 144, 144)
    canvas.paste(_c16, (1248, 678), _c16)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 678, 1392, 822]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_02_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-4/17_icon_Close_current_screen.png
try:
    _c17 = get_crop(17, 144, 144)
    canvas.paste(_c17, (1248, 1110), _c17)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 1110, 1392, 1254]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_02_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-4/18_icon_Tickets.png
try:
    _c18 = get_crop(18, 288, 156)
    canvas.paste(_c18, (864, 2804), _c18)
except Exception:
    pass
layout["Tickets"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_02_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-4/19_icon_Cancel.png
try:
    _c19 = get_crop(19, 144, 144)
    canvas.paste(_c19, (1248, 390), _c19)
except Exception:
    pass
layout["Cancel"] = [1248, 390, 1392, 534]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_02_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-4/20_icon_Close_current_screen.png
try:
    _c20 = get_crop(20, 144, 144)
    canvas.paste(_c20, (1248, 1542), _c20)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 1542, 1392, 1686]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_02_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-4/21_icon_Close_current_screen.png
try:
    _c21 = get_crop(21, 144, 144)
    canvas.paste(_c21, (1248, 1686), _c21)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 1686, 1392, 1830]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_02_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-4/22_icon_Home.png
try:
    _c22 = get_crop(22, 288, 156)
    canvas.paste(_c22, (0, 2804), _c22)
except Exception:
    pass
layout["Home"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_02_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-4/23_icon_Science_Tech.png
try:
    _c23 = get_crop(23, 1344, 144)
    canvas.paste(_c23, (48, 822), _c23)
except Exception:
    pass
layout["Science_&_Tech"] = [48, 822, 1392, 966]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_02_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-4/24_icon_Search_for__..png
try:
    _c24 = get_crop(24, 1344, 191)
    canvas.paste(_c24, (48, 72), _c24)
except Exception:
    pass
layout["[Search_for__."] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_02_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-4/25_icon_Close_current_screen.png
try:
    _c25 = get_crop(25, 144, 144)
    canvas.paste(_c25, (1248, 966), _c25)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 966, 1392, 1110]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_02_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-4/26_icon_Search_events.png
try:
    _c26 = get_crop(26, 288, 156)
    canvas.paste(_c26, (288, 2804), _c26)
except Exception:
    pass
layout["Search_events"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_02_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-4/27_icon_More.png
try:
    _c27 = get_crop(27, 288, 156)
    canvas.paste(_c27, (1152, 2804), _c27)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_02_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-4/28_icon_Search_for__..png
try:
    _c28 = get_crop(28, 48, 65)
    canvas.paste(_c28, (383, 2), _c28)
except Exception:
    pass
layout["[Search_for__."] = [383, 2, 431, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_02_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-4/29_icon_Talkshow.png
try:
    _c29 = get_crop(29, 1344, 144)
    canvas.paste(_c29, (48, 1542), _c29)
except Exception:
    pass
layout["Talkshow"] = [48, 1542, 1392, 1686]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_02_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-4/30_icon_Taylor_Swift.png
try:
    _c30 = get_crop(30, 1344, 144)
    canvas.paste(_c30, (48, 1398), _c30)
except Exception:
    pass
layout["Taylor_Swift"] = [48, 1398, 1392, 1542]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_02_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-4/31_icon_Basketball.png
try:
    _c31 = get_crop(31, 1344, 144)
    canvas.paste(_c31, (48, 966), _c31)
except Exception:
    pass
layout["Basketball"] = [48, 966, 1392, 1110]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_02_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-4/32_text_6.58.png
try:
    _c32 = get_crop(32, 89, 43)
    canvas.paste(_c32, (22, 17), _c32)
except Exception:
    pass
layout["6.58"] = [22, 17, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_02_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-4/33_text_Recent.png
try:
    _c33 = get_crop(33, 203, 62)
    canvas.paste(_c33, (45, 299), _c33)
except Exception:
    pass
layout["Recent"] = [45, 299, 248, 361]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_02_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-4/34_text_Exhibition.png
try:
    _c34 = get_crop(34, 191, 49)
    canvas.paste(_c34, (164, 1159), _c34)
except Exception:
    pass
layout["Exhibition"] = [164, 1159, 355, 1208]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_02_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-4/35_text_Festival.png
try:
    _c35 = get_crop(35, 154, 48)
    canvas.paste(_c35, (162, 1304), _c35)
except Exception:
    pass
layout["Festival"] = [162, 1304, 316, 1352]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_02_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-4/36_text_Broadway.png
try:
    _c36 = get_crop(36, 195, 53)
    canvas.paste(_c36, (163, 1736), _c36)
except Exception:
    pass
layout["Broadway"] = [163, 1736, 358, 1789]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_02_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-4/37_clickable_Exhibition.png
try:
    _c37 = get_crop(37, 1344, 144)
    canvas.paste(_c37, (48, 1110), _c37)
except Exception:
    pass
layout["Exhibition"] = [48, 1110, 1392, 1254]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_02_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-4/38_clickable_Festival.png
try:
    _c38 = get_crop(38, 1344, 144)
    canvas.paste(_c38, (48, 1254), _c38)
except Exception:
    pass
layout["Festival"] = [48, 1254, 1392, 1398]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_02_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-4/39_clickable_Broadway.png
try:
    _c39 = get_crop(39, 1344, 144)
    canvas.paste(_c39, (48, 1686), _c39)
except Exception:
    pass
layout["Broadway"] = [48, 1686, 1392, 1830]
