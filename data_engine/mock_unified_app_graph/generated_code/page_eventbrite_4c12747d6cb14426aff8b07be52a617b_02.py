# page_id: page_eventbrite_4c12747d6cb14426aff8b07be52a617b_02
# screenshot: 2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-4.png
# step_index: 2/11
# task: Open Eventbrite. Search 'Art'. Filter event type "Performance". Select the first event. Follow the organizer and save the event to favorite. What is the price of the ticket?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw background structure for the mobile UI (uses provided canvas and draw)

# Status bar
status_h = 72
draw.rectangle([(0, 0), (1440, status_h)], fill=(220, 220, 220))
# subtle bottom edge for status bar
draw.line([(0, status_h - 1), (1440, status_h - 1)], fill=(200, 200, 200), width=1)

# Search area background (keeps blank so icons/text pasted on top remain visible)
search_x = 48
search_y = 72
search_w = 1344
search_h = 191
search_rect = (search_x, search_y, search_x + search_w, search_y + search_h)
# white rounded background (very subtle so pasted content sits above it)
draw.rounded_rectangle(search_rect, radius=6, fill=(255, 255, 255), outline=None)

# Blue underline under search input (prominent accent line)
underline_y = search_y + 64
draw.line([(search_x, underline_y), (search_x + search_w, underline_y)], fill=(47, 96, 212), width=4)

# Divider directly under the search area (thin grey)
draw.line([(search_x, search_y + search_h), (search_x + search_w, search_y + search_h)], fill=(235, 235, 240), width=1)

# Section separators for the recent items list (match rows, left/right padded)
separator_color = (235, 235, 240)
separator_x1 = search_x
separator_x2 = search_x + search_w

# Known boundary y-positions (bottom edges of rows / sections)
separators = [
    search_y + search_h,  # bottom of search area
    534,  # bottom of first row (Food and Drink -> Education boundary)
    678,  # next
    822,
    966,
    1110,
    1254,
    1398,
    1542,
    1686,
    2804  # top of bottom navigation
]
for y in separators:
    draw.line([(separator_x1, y), (separator_x2, y)], fill=separator_color, width=1)

# "Recent" header area subtle emphasis (no text drawn)
recent_header_top = search_y + 92
recent_header_bottom = recent_header_top + 48
# very light transparent-like band (solid RGB since canvas is RGB)
draw.rectangle([(search_x, recent_header_top), (search_x + 260, recent_header_bottom)], fill=(250, 250, 251))

# Footer navigation area top border and background
nav_top = 2804
draw.rectangle([(0, nav_top), (1440, 2960)], fill=(255, 255, 255))
draw.line([(0, nav_top), (1440, nav_top)], fill=(225, 225, 230), width=1)

# Subtle left and right content margins guide (non-intrusive)
margin_color = (245, 245, 247)
draw.line([(24, status_h), (24, 2960)], fill=margin_color, width=1)
draw.line([(1440 - 24, status_h), (1440 - 24, 2960)], fill=margin_color, width=1)

# Optional faint vertical rule to visually separate search area from content (light)
draw.line([(search_x, search_y + 8), (search_x, nav_top - 8)], fill=(248, 248, 249), width=1)
draw.line([(search_x + search_w, search_y + 8), (search_x + search_w, nav_top - 8)], fill=(248, 248, 249), width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_02_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-4/00_icon_7.51.png
try:
    _c0 = get_crop(0, 59, 62)
    canvas.paste(_c0, (180, 2), _c0)
except Exception:
    pass
layout["7.51"] = [180, 2, 239, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_02_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-4/01_icon_7.51.png
try:
    _c1 = get_crop(1, 61, 62)
    canvas.paste(_c1, (113, 2), _c1)
except Exception:
    pass
layout["7.51"] = [113, 2, 174, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_02_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-4/02_icon_Search_for__..png
try:
    _c2 = get_crop(2, 64, 62)
    canvas.paste(_c2, (309, 2), _c2)
except Exception:
    pass
layout["[Search_for__."] = [309, 2, 373, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_02_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-4/03_icon_Cancel.png
try:
    _c3 = get_crop(3, 149, 144)
    canvas.paste(_c3, (1243, 97), _c3)
except Exception:
    pass
layout["Cancel"] = [1243, 97, 1392, 241]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_02_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-4/04_icon_icon_4.png
try:
    _c4 = get_crop(4, 49, 60)
    canvas.paste(_c4, (249, 3), _c4)
except Exception:
    pass
layout["icon_4"] = [249, 3, 298, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_02_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-4/05_icon_Education.png
try:
    _c5 = get_crop(5, 1344, 144)
    canvas.paste(_c5, (48, 534), _c5)
except Exception:
    pass
layout["Education"] = [48, 534, 1392, 678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_02_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-4/06_icon_Cancel.png
try:
    _c6 = get_crop(6, 56, 63)
    canvas.paste(_c6, (1317, 0), _c6)
except Exception:
    pass
layout["Cancel"] = [1317, 0, 1373, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_02_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-4/07_icon_Cancel.png
try:
    _c7 = get_crop(7, 99, 63)
    canvas.paste(_c7, (1212, 0), _c7)
except Exception:
    pass
layout["Cancel"] = [1212, 0, 1311, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_02_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-4/08_icon_Yoga_session.png
try:
    _c8 = get_crop(8, 1344, 144)
    canvas.paste(_c8, (48, 678), _c8)
except Exception:
    pass
layout["Yoga_session"] = [48, 678, 1392, 822]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_02_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-4/09_icon_7.51.png
try:
    _c9 = get_crop(9, 122, 107)
    canvas.paste(_c9, (55, 115), _c9)
except Exception:
    pass
layout["7.51"] = [55, 115, 177, 222]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_02_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-4/10_icon_Close_current_screen.png
try:
    _c10 = get_crop(10, 144, 144)
    canvas.paste(_c10, (1248, 822), _c10)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 822, 1392, 966]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_02_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-4/11_icon_Close_current_screen.png
try:
    _c11 = get_crop(11, 144, 144)
    canvas.paste(_c11, (1248, 534), _c11)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 534, 1392, 678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_02_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-4/12_icon_Tickets.png
try:
    _c12 = get_crop(12, 288, 156)
    canvas.paste(_c12, (864, 2804), _c12)
except Exception:
    pass
layout["Tickets"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_02_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-4/13_icon_Close_current_screen.png
try:
    _c13 = get_crop(13, 144, 144)
    canvas.paste(_c13, (1248, 678), _c13)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 678, 1392, 822]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_02_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-4/14_icon_Close_current_screen.png
try:
    _c14 = get_crop(14, 144, 144)
    canvas.paste(_c14, (1248, 1110), _c14)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 1110, 1392, 1254]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_02_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-4/15_icon_Cancel.png
try:
    _c15 = get_crop(15, 144, 144)
    canvas.paste(_c15, (1248, 390), _c15)
except Exception:
    pass
layout["Cancel"] = [1248, 390, 1392, 534]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_02_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-4/16_icon_Close_current_screen.png
try:
    _c16 = get_crop(16, 144, 144)
    canvas.paste(_c16, (1248, 1398), _c16)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 1398, 1392, 1542]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_02_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-4/17_icon_Close_current_screen.png
try:
    _c17 = get_crop(17, 144, 144)
    canvas.paste(_c17, (1248, 1254), _c17)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 1254, 1392, 1398]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_02_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-4/18_icon_Close_current_screen.png
try:
    _c18 = get_crop(18, 144, 144)
    canvas.paste(_c18, (1248, 1542), _c18)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 1542, 1392, 1686]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_02_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-4/19_icon_Close_current_screen.png
try:
    _c19 = get_crop(19, 144, 144)
    canvas.paste(_c19, (1248, 1686), _c19)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 1686, 1392, 1830]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_02_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-4/20_icon_Close_current_screen.png
try:
    _c20 = get_crop(20, 144, 144)
    canvas.paste(_c20, (1248, 966), _c20)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 966, 1392, 1110]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_02_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-4/21_icon_Music.png
try:
    _c21 = get_crop(21, 1344, 144)
    canvas.paste(_c21, (48, 1398), _c21)
except Exception:
    pass
layout["Music"] = [48, 1398, 1392, 1542]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_02_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-4/22_icon_Search_for__..png
try:
    _c22 = get_crop(22, 1344, 191)
    canvas.paste(_c22, (48, 72), _c22)
except Exception:
    pass
layout["[Search_for__."] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_02_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-4/23_icon_Favorites.png
try:
    _c23 = get_crop(23, 288, 156)
    canvas.paste(_c23, (576, 2804), _c23)
except Exception:
    pass
layout["Favorites"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_02_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-4/24_icon_Search_events.png
try:
    _c24 = get_crop(24, 288, 156)
    canvas.paste(_c24, (288, 2804), _c24)
except Exception:
    pass
layout["Search_events"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_02_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-4/25_icon_Home.png
try:
    _c25 = get_crop(25, 288, 156)
    canvas.paste(_c25, (0, 2804), _c25)
except Exception:
    pass
layout["Home"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_02_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-4/26_icon_7.51.png
try:
    _c26 = get_crop(26, 91, 62)
    canvas.paste(_c26, (14, 2), _c26)
except Exception:
    pass
layout["7.51"] = [14, 2, 105, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_02_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-4/27_icon_Food_and_Drink.png
try:
    _c27 = get_crop(27, 1344, 144)
    canvas.paste(_c27, (48, 390), _c27)
except Exception:
    pass
layout["Food_and_Drink"] = [48, 390, 1392, 534]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_02_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-4/28_icon_Food_Drink.png
try:
    _c28 = get_crop(28, 1344, 144)
    canvas.paste(_c28, (48, 1542), _c28)
except Exception:
    pass
layout["Food_&_Drink"] = [48, 1542, 1392, 1686]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_02_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-4/29_icon_Search_for__..png
try:
    _c29 = get_crop(29, 47, 63)
    canvas.paste(_c29, (383, 3), _c29)
except Exception:
    pass
layout["[Search_for__."] = [383, 3, 430, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_02_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-4/30_icon_More.png
try:
    _c30 = get_crop(30, 288, 156)
    canvas.paste(_c30, (1152, 2804), _c30)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_02_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-4/31_icon_Science_Tech.png
try:
    _c31 = get_crop(31, 113, 131)
    canvas.paste(_c31, (27, 1696), _c31)
except Exception:
    pass
layout["Science_&_Tech"] = [27, 1696, 140, 1827]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_02_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-4/32_text_Coding_Workshop.png
try:
    _c32 = get_crop(32, 1344, 144)
    canvas.paste(_c32, (48, 822), _c32)
except Exception:
    pass
layout["Coding_Workshop"] = [48, 822, 1392, 966]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_02_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-4/33_text_Music_Festival.png
try:
    _c33 = get_crop(33, 273, 53)
    canvas.paste(_c33, (163, 1014), _c33)
except Exception:
    pass
layout["Music_Festival"] = [163, 1014, 436, 1067]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_02_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-4/34_text_Fitness.png
try:
    _c34 = get_crop(34, 140, 43)
    canvas.paste(_c34, (165, 1164), _c34)
except Exception:
    pass
layout["Fitness"] = [165, 1164, 305, 1207]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_02_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-4/35_text_Art.png
try:
    _c35 = get_crop(35, 69, 48)
    canvas.paste(_c35, (163, 1305), _c35)
except Exception:
    pass
layout["Art"] = [163, 1305, 232, 1353]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_02_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-4/36_text_Science_Tech.png
try:
    _c36 = get_crop(36, 292, 49)
    canvas.paste(_c36, (160, 1735), _c36)
except Exception:
    pass
layout["Science_&_Tech"] = [160, 1735, 452, 1784]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_02_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-4/37_clickable_Music_Festival.png
try:
    _c37 = get_crop(37, 1344, 144)
    canvas.paste(_c37, (48, 966), _c37)
except Exception:
    pass
layout["Music_Festival"] = [48, 966, 1392, 1110]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_02_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-4/38_clickable_Fitness.png
try:
    _c38 = get_crop(38, 1344, 144)
    canvas.paste(_c38, (48, 1110), _c38)
except Exception:
    pass
layout["Fitness"] = [48, 1110, 1392, 1254]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_02_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-4/39_clickable_Art.png
try:
    _c39 = get_crop(39, 1344, 144)
    canvas.paste(_c39, (48, 1254), _c39)
except Exception:
    pass
layout["Art"] = [48, 1254, 1392, 1398]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_02_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-4/40_clickable_Science_Tech.png
try:
    _c40 = get_crop(40, 1344, 144)
    canvas.paste(_c40, (48, 1686), _c40)
except Exception:
    pass
layout["Science_&_Tech"] = [48, 1686, 1392, 1830]
