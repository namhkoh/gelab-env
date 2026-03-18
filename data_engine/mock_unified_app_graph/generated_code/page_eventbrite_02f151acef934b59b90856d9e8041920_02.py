# page_id: page_eventbrite_02f151acef934b59b90856d9e8041920_02
# screenshot: 2024_4_24_17_24_02f151acef934b59b90856d9e8041920-4.png
# step_index: 2/11
# task: Open Eventbrite. Check the "Tech" events happening this month. Open the first event and check its date and time.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw UI background and structural elements for the provided canvas/draw
# Assumes variables provided in the environment:
# - canvas: PIL.Image (1440x2960 RGB)
# - draw: PIL.ImageDraw.Draw(canvas)
# - font_sm, font_md, font_lg, font_xl (not used here)

# Fill overall background (dominant color is white)
draw.rectangle([(0, 0), (1440, 2960)], fill="#ffffff")

# Status bar area (top ~72px) - light grey background to match screenshot status bar
status_bar_h = 72
draw.rectangle([(0, 0), (1440, status_bar_h)], fill="#bfbfbf")
# Status bar bottom hairline
draw.line([(0, status_bar_h), (1440, status_bar_h)], fill="#a8a8a8", width=1)

# Header / Search area background (beneath status bar)
search_area_top = status_bar_h
search_area_bottom = 160
draw.rectangle([(0, search_area_top), (1440, search_area_bottom)], fill="#ffffff")
# Blue underline under the search box (prominent blue divider)
underline_y = search_area_bottom
draw.rectangle([(48, underline_y - 3), (1392, underline_y + 1)], fill="#2546ff")

# Thin divider under the header/search region
draw.line([(0, underline_y + 1), (1440, underline_y + 1)], fill="#e6e6ea", width=1)

# "Recent" section area background strip (subtle off-white block to anchor section)
recent_strip_top = 188
recent_strip_bottom = 240
draw.rectangle([(48, recent_strip_top), (1392, recent_strip_bottom)], fill="#ffffff")

# Rounded card container behind the list of recent items (subtle border)
list_container_top = 520
list_container_bottom = 1760
list_container_left = 48
list_container_right = 1392
draw.rounded_rectangle(
    [(list_container_left, list_container_top), (list_container_right, list_container_bottom)],
    radius=12,
    fill="#ffffff",
    outline="#f0f0f3",
    width=1
)

# Light separators between list rows (use detected item vertical positions as guides)
item_positions = [534, 678, 822, 966, 1110, 1254, 1398, 1542, 1696]
for y in item_positions:
    # draw a subtle divider at the bottom edge of each list item area
    draw.line([(list_container_left + 8, y + 144), (list_container_right - 8, y + 144)], fill="#f3f3f6", width=1)

# Additional faint horizontal guides to separate major sections (near top of list)
draw.line([(48, 300), (1392, 300)], fill="#f3f3f6", width=1)

# Bottom navigation bar area (around y=2804) - keep background and top border only
nav_top = 2804
nav_bottom = 2960
draw.rectangle([(0, nav_top), (1440, nav_bottom)], fill="#ffffff")
# Top border for the nav bar
draw.line([(0, nav_top), (1440, nav_top)], fill="#e6e6ea", width=2)

# Subtle left and right page gutters (visual guides) - very faint
draw.line([(48, 0), (48, 2960)], fill="#fafafb", width=1)
draw.line([(1392, 0), (1392, 2960)], fill="#fafafb", width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_02_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-4/00_icon_5.24.png
try:
    _c0 = get_crop(0, 58, 62)
    canvas.paste(_c0, (180, 2), _c0)
except Exception:
    pass
layout["5.24"] = [180, 2, 238, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_02_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-4/01_icon_5.24.png
try:
    _c1 = get_crop(1, 59, 63)
    canvas.paste(_c1, (114, 2), _c1)
except Exception:
    pass
layout["5.24"] = [114, 2, 173, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_02_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-4/02_icon_Search_for_-..png
try:
    _c2 = get_crop(2, 64, 63)
    canvas.paste(_c2, (309, 2), _c2)
except Exception:
    pass
layout["(Search_for:-."] = [309, 2, 373, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_02_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-4/03_icon_icon_3.png
try:
    _c3 = get_crop(3, 49, 61)
    canvas.paste(_c3, (249, 3), _c3)
except Exception:
    pass
layout["icon_3"] = [249, 3, 298, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_02_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-4/04_icon_Business_Seminar.png
try:
    _c4 = get_crop(4, 1344, 144)
    canvas.paste(_c4, (48, 534), _c4)
except Exception:
    pass
layout["Business_Seminar"] = [48, 534, 1392, 678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_02_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-4/05_icon_Cancel.png
try:
    _c5 = get_crop(5, 149, 144)
    canvas.paste(_c5, (1243, 97), _c5)
except Exception:
    pass
layout["Cancel"] = [1243, 97, 1392, 241]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_02_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-4/06_icon_Search_for_-..png
try:
    _c6 = get_crop(6, 1344, 144)
    canvas.paste(_c6, (48, 390), _c6)
except Exception:
    pass
layout["(Search_for:-."] = [48, 390, 1392, 534]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_02_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-4/07_icon_Cancel.png
try:
    _c7 = get_crop(7, 57, 63)
    canvas.paste(_c7, (1316, 0), _c7)
except Exception:
    pass
layout["Cancel"] = [1316, 0, 1373, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_02_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-4/08_icon_Cancel.png
try:
    _c8 = get_crop(8, 99, 63)
    canvas.paste(_c8, (1212, 0), _c8)
except Exception:
    pass
layout["Cancel"] = [1212, 0, 1311, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_02_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-4/09_icon_Language_Learning.png
try:
    _c9 = get_crop(9, 1344, 144)
    canvas.paste(_c9, (48, 678), _c9)
except Exception:
    pass
layout["Language_Learning"] = [48, 678, 1392, 822]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_02_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-4/10_icon_Open_Mic_Night.png
try:
    _c10 = get_crop(10, 1344, 144)
    canvas.paste(_c10, (48, 1398), _c10)
except Exception:
    pass
layout["Open_Mic_Night"] = [48, 1398, 1392, 1542]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_02_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-4/11_icon_5.24.png
try:
    _c11 = get_crop(11, 124, 109)
    canvas.paste(_c11, (53, 114), _c11)
except Exception:
    pass
layout["5.24"] = [53, 114, 177, 223]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_02_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-4/12_icon_Tickets.png
try:
    _c12 = get_crop(12, 288, 156)
    canvas.paste(_c12, (864, 2804), _c12)
except Exception:
    pass
layout["Tickets"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_02_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-4/13_icon_Photography.png
try:
    _c13 = get_crop(13, 1344, 144)
    canvas.paste(_c13, (48, 1542), _c13)
except Exception:
    pass
layout["Photography"] = [48, 1542, 1392, 1686]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_02_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-4/14_icon_5.24.png
try:
    _c14 = get_crop(14, 93, 62)
    canvas.paste(_c14, (15, 1), _c14)
except Exception:
    pass
layout["5.24"] = [15, 1, 108, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_02_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-4/15_icon_Close_current_screen.png
try:
    _c15 = get_crop(15, 144, 144)
    canvas.paste(_c15, (1248, 822), _c15)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 822, 1392, 966]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_02_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-4/16_icon_Open_Mic_Night.png
try:
    _c16 = get_crop(16, 1344, 144)
    canvas.paste(_c16, (48, 1254), _c16)
except Exception:
    pass
layout["Open_Mic_Night"] = [48, 1254, 1392, 1398]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_02_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-4/17_icon_Close_current_screen.png
try:
    _c17 = get_crop(17, 144, 144)
    canvas.paste(_c17, (1248, 534), _c17)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 534, 1392, 678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_02_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-4/18_icon_Favorites.png
try:
    _c18 = get_crop(18, 288, 156)
    canvas.paste(_c18, (576, 2804), _c18)
except Exception:
    pass
layout["Favorites"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_02_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-4/19_icon_Close_current_screen.png
try:
    _c19 = get_crop(19, 144, 144)
    canvas.paste(_c19, (1248, 678), _c19)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 678, 1392, 822]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_02_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-4/20_icon_Close_current_screen.png
try:
    _c20 = get_crop(20, 144, 144)
    canvas.paste(_c20, (1248, 1254), _c20)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 1254, 1392, 1398]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_02_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-4/21_icon_Close_current_screen.png
try:
    _c21 = get_crop(21, 144, 144)
    canvas.paste(_c21, (1248, 1398), _c21)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 1398, 1392, 1542]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_02_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-4/22_icon_Close_current_screen.png
try:
    _c22 = get_crop(22, 144, 144)
    canvas.paste(_c22, (1248, 1110), _c22)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 1110, 1392, 1254]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_02_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-4/23_icon_Close_current_screen.png
try:
    _c23 = get_crop(23, 144, 144)
    canvas.paste(_c23, (1248, 1542), _c23)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 1542, 1392, 1686]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_02_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-4/24_icon_Close_current_screen.png
try:
    _c24 = get_crop(24, 144, 144)
    canvas.paste(_c24, (1248, 1686), _c24)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 1686, 1392, 1830]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_02_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-4/25_icon_Cancel.png
try:
    _c25 = get_crop(25, 144, 144)
    canvas.paste(_c25, (1248, 390), _c25)
except Exception:
    pass
layout["Cancel"] = [1248, 390, 1392, 534]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_02_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-4/26_icon_Close_current_screen.png
try:
    _c26 = get_crop(26, 144, 144)
    canvas.paste(_c26, (1248, 966), _c26)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 966, 1392, 1110]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_02_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-4/27_icon_Home.png
try:
    _c27 = get_crop(27, 288, 156)
    canvas.paste(_c27, (0, 2804), _c27)
except Exception:
    pass
layout["Home"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_02_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-4/28_icon_Language_Learning.png
try:
    _c28 = get_crop(28, 1344, 144)
    canvas.paste(_c28, (48, 822), _c28)
except Exception:
    pass
layout["Language_Learning"] = [48, 822, 1392, 966]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_02_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-4/29_icon_Search_events.png
try:
    _c29 = get_crop(29, 288, 156)
    canvas.paste(_c29, (288, 2804), _c29)
except Exception:
    pass
layout["Search_events"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_02_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-4/30_icon_Search_for_-..png
try:
    _c30 = get_crop(30, 1344, 191)
    canvas.paste(_c30, (48, 72), _c30)
except Exception:
    pass
layout["(Search_for:-."] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_02_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-4/31_icon_Gardening.png
try:
    _c31 = get_crop(31, 1344, 144)
    canvas.paste(_c31, (48, 966), _c31)
except Exception:
    pass
layout["Gardening"] = [48, 966, 1392, 1110]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_02_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-4/32_icon_More.png
try:
    _c32 = get_crop(32, 288, 156)
    canvas.paste(_c32, (1152, 2804), _c32)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_02_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-4/33_icon_Open_Mic_Night.png
try:
    _c33 = get_crop(33, 1344, 144)
    canvas.paste(_c33, (48, 1110), _c33)
except Exception:
    pass
layout["Open_Mic_Night"] = [48, 1110, 1392, 1254]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_02_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-4/34_icon_Wellness.png
try:
    _c34 = get_crop(34, 115, 132)
    canvas.paste(_c34, (26, 1696), _c34)
except Exception:
    pass
layout["Wellness"] = [26, 1696, 141, 1828]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_02_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-4/35_icon_Search_for_-..png
try:
    _c35 = get_crop(35, 47, 65)
    canvas.paste(_c35, (383, 2), _c35)
except Exception:
    pass
layout["(Search_for:-."] = [383, 2, 430, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_02_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-4/36_text_Wellness.png
try:
    _c36 = get_crop(36, 173, 43)
    canvas.paste(_c36, (165, 1740), _c36)
except Exception:
    pass
layout["Wellness"] = [165, 1740, 338, 1783]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_02_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-4/37_clickable_Wellness.png
try:
    _c37 = get_crop(37, 1344, 144)
    canvas.paste(_c37, (48, 1686), _c37)
except Exception:
    pass
layout["Wellness"] = [48, 1686, 1392, 1830]
