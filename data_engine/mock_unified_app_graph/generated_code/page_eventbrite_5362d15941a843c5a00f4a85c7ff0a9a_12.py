# page_id: page_eventbrite_5362d15941a843c5a00f4a85c7ff0a9a_12
# screenshot: 2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-14.png
# step_index: 12/12
# task: Open Eventbrite. Set the city to 'Los Angeles'. Search 'Business'. Filter 'French' speaking events. Add the first event to favorite.
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

# Ensure full background is white (canvas is already white, but reinforce)
draw.rectangle([(0, 0), (1440, 2960)], fill=(255, 255, 255))

# Top status bar area (~96px tall) - light grey band
status_bar_h = 96
draw.rectangle([(0, 0), (1440, status_bar_h)], fill=(189, 189, 189))

# Header area divider (thin line under the header/search area)
header_div_y = 240
draw.line([(24, header_div_y), (1416, header_div_y)], fill=(220, 220, 220), width=2)

# Subtle separator between header and filters row
filters_sep_y = 344
draw.line([(24, filters_sep_y), (1416, filters_sep_y)], fill=(245, 245, 245), width=1)

# Light rounded background behind the horizontal filter chips row
# (chips themselves will be pasted on top; this is just a subtle background card)
filters_bg_top = 360
filters_bg_bottom = 468
draw.rounded_rectangle(
    [(24, filters_bg_top), (1416, filters_bg_bottom)],
    radius=56,
    fill=(237, 246, 255),
    outline=None
)

# Subtle thin divider below the filters area (visual separation before content)
draw.line(
    [(24, filters_bg_bottom + 8), (1416, filters_bg_bottom + 8)],
    fill=(245, 245, 245),
    width=1
)

# Main content area: keep white (no content drawn to avoid overlapping pasted elements)
# Optionally draw a very subtle background band where "no results" sits (kept minimal)
# (Do not draw icons/text that will be pasted)
content_hint_top = filters_bg_bottom + 40
content_hint_bottom = 2000
draw.rectangle([(0, content_hint_top), (1440, content_hint_bottom)], fill=(255, 255, 255))

# Bottom navigation bar area - top border and white background
nav_top = 2804
draw.line([(0, nav_top), (1440, nav_top)], fill=(225, 225, 225), width=2)
draw.rectangle([(0, nav_top), (1440, 2960)], fill=(255, 255, 255))

# Very subtle shadow line above the nav to give slight elevation
draw.line([(0, nav_top - 6), (1440, nav_top - 6)], fill=(245, 245, 245), width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_12_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-14/00_icon_Anytime.png
try:
    _c0 = get_crop(0, 400, 103)
    canvas.paste(_c0, (438, 410), _c0)
except Exception:
    pass
layout["Anytime"] = [438, 410, 838, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_12_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-14/01_icon_Music.png
try:
    _c1 = get_crop(1, 187, 103)
    canvas.paste(_c1, (850, 410), _c1)
except Exception:
    pass
layout["Music"] = [850, 410, 1037, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_12_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-14/02_icon_Business.png
try:
    _c2 = get_crop(2, 241, 103)
    canvas.paste(_c2, (1049, 410), _c2)
except Exception:
    pass
layout["Business"] = [1049, 410, 1290, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_12_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-14/03_icon_1_Filter.png
try:
    _c3 = get_crop(3, 372, 103)
    canvas.paste(_c3, (54, 410), _c3)
except Exception:
    pass
layout["1_Filter"] = [54, 410, 426, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_12_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-14/04_icon_No_results_found.png
try:
    _c4 = get_crop(4, 293, 296)
    canvas.paste(_c4, (572, 643), _c4)
except Exception:
    pass
layout["No_results_found"] = [572, 643, 865, 939]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_12_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-14/05_icon_Fo.png
try:
    _c5 = get_crop(5, 135, 111)
    canvas.paste(_c5, (1295, 406), _c5)
except Exception:
    pass
layout["Fo("] = [1295, 406, 1430, 517]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_12_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-14/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 52, 67)
    canvas.paste(_c6, (1152, 0), _c6)
except Exception:
    pass
layout["icon_6"] = [1152, 0, 1204, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_12_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-14/07_icon_Close_current_screen.png
try:
    _c7 = get_crop(7, 144, 144)
    canvas.paste(_c7, (1248, 96), _c7)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 96, 1392, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_12_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-14/08_icon_8.02.png
try:
    _c8 = get_crop(8, 112, 107)
    canvas.paste(_c8, (60, 117), _c8)
except Exception:
    pass
layout["8.02"] = [60, 117, 172, 224]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_12_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-14/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 99, 65)
    canvas.paste(_c9, (1212, 0), _c9)
except Exception:
    pass
layout["icon_9"] = [1212, 0, 1311, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_12_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-14/10_icon_Los_Angeles.png
try:
    _c10 = get_crop(10, 492, 144)
    canvas.paste(_c10, (0, 259), _c10)
except Exception:
    pass
layout["Los_Angeles"] = [0, 259, 492, 403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_12_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-14/11_icon_Business.png
try:
    _c11 = get_crop(11, 65, 64)
    canvas.paste(_c11, (308, 0), _c11)
except Exception:
    pass
layout["Business"] = [308, 0, 373, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_12_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-14/12_icon_8.02.png
try:
    _c12 = get_crop(12, 56, 63)
    canvas.paste(_c12, (182, 1), _c12)
except Exception:
    pass
layout["8.02"] = [182, 1, 238, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_12_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-14/13_icon_8.02.png
try:
    _c13 = get_crop(13, 56, 65)
    canvas.paste(_c13, (116, 0), _c13)
except Exception:
    pass
layout["8.02"] = [116, 0, 172, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_12_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-14/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 48, 62)
    canvas.paste(_c14, (251, 1), _c14)
except Exception:
    pass
layout["icon_14"] = [251, 1, 299, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_12_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-14/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 53, 63)
    canvas.paste(_c15, (1319, 0), _c15)
except Exception:
    pass
layout["icon_15"] = [1319, 0, 1372, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_12_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-14/16_icon_Search_events.png
try:
    _c16 = get_crop(16, 288, 156)
    canvas.paste(_c16, (288, 2804), _c16)
except Exception:
    pass
layout["Search_events"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_12_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-14/17_icon_Business.png
try:
    _c17 = get_crop(17, 1344, 191)
    canvas.paste(_c17, (48, 72), _c17)
except Exception:
    pass
layout["Business"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_12_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-14/18_icon_Tickets.png
try:
    _c18 = get_crop(18, 288, 156)
    canvas.paste(_c18, (864, 2804), _c18)
except Exception:
    pass
layout["Tickets"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_12_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-14/19_icon_Business.png
try:
    _c19 = get_crop(19, 45, 62)
    canvas.paste(_c19, (385, 2), _c19)
except Exception:
    pass
layout["Business"] = [385, 2, 430, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_12_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-14/20_icon_Favorites.png
try:
    _c20 = get_crop(20, 288, 156)
    canvas.paste(_c20, (576, 2804), _c20)
except Exception:
    pass
layout["Favorites"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_12_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-14/21_icon_More.png
try:
    _c21 = get_crop(21, 288, 156)
    canvas.paste(_c21, (1152, 2804), _c21)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_12_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-14/22_icon_Home.png
try:
    _c22 = get_crop(22, 288, 156)
    canvas.paste(_c22, (0, 2804), _c22)
except Exception:
    pass
layout["Home"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_12_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-14/23_icon_Tickets.png
try:
    _c23 = get_crop(23, 288, 156)
    canvas.paste(_c23, (864, 2804), _c23)
except Exception:
    pass
layout["Tickets"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_12_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-14/24_text_8.02.png
try:
    _c24 = get_crop(24, 91, 43)
    canvas.paste(_c24, (20, 17), _c24)
except Exception:
    pass
layout["8.02"] = [20, 17, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_12_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-14/25_text_No_results_found.png
try:
    _c25 = get_crop(25, 445, 56)
    canvas.paste(_c25, (500, 1032), _c25)
except Exception:
    pass
layout["No_results_found"] = [500, 1032, 945, 1088]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_12_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-14/26_text_Expand_your_search_and.png
try:
    _c26 = get_crop(26, 516, 63)
    canvas.paste(_c26, (365, 1174), _c26)
except Exception:
    pass
layout["Expand_your_search_and"] = [365, 1174, 881, 1237]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_12_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-14/27_text_again.png
try:
    _c27 = get_crop(27, 138, 69)
    canvas.paste(_c27, (940, 1173), _c27)
except Exception:
    pass
layout["again"] = [940, 1173, 1078, 1242]
