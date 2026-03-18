# page_id: page_eventbrite_45f56b06f31541079045047b6d542613_07
# screenshot: 2024_4_23_19_27_45f56b06f31541079045047b6d542613-9.png
# step_index: 7/21
# task: Open Eventbrite. Search events 'Yoga session' in New York. Filter free events and set date from May 3 to May 6. What is the location of the first promoted event?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# draw background and structure for the mobile UI page
# available variables: canvas (1440x2960 RGB), draw (ImageDraw), font_sm/font_md/font_lg/font_xl

# Full canvas background (slightly off-white to match screenshot)
draw.rectangle([(0, 0), (1440, 2960)], fill="#fbfbfd")

# Status bar area (top ~56px, light grey)
status_h = 56
draw.rectangle([(0, 0), (1440, status_h)], fill="#d0d0d0")

# Subtle divider at bottom of status bar
draw.line([(0, status_h), (1440, status_h)], fill="#c6c6c6", width=1)

# Header / Search area background (white)
header_top = status_h
header_bottom = 140
draw.rectangle([(0, header_top), (1440, header_bottom)], fill="#ffffff")

# Blue underline for the search field (thin prominent line)
underline_left = 48
underline_right = 1440 - 48
underline_y = 128
draw.line([(underline_left, underline_y), (underline_right, underline_y)], fill="#2347d6", width=4)

# Thin separator below header
draw.line([(0, header_bottom), (1440, header_bottom)], fill="#efeff2", width=1)

# "Recent" section heading area background (no text drawn)
recent_top = header_bottom + 12
recent_bottom = recent_top + 48
# keep it white but add a subtle area separator to indicate the heading region
draw.rectangle([(0, recent_top), (1440, recent_bottom)], fill="#ffffff")
draw.line([(48, recent_bottom), (1392, recent_bottom)], fill="#efeff2", width=1)

# Large rounded container behind the list of recent items
list_left = 48
list_right = 1392
list_top = 390 - 12   # slightly above first item
list_bottom = 1686 + 144 + 12  # slightly below last item
draw.rounded_rectangle([(list_left, list_top), (list_right, list_bottom)],
                       radius=10, fill="#ffffff", outline="#f0f0f3", width=1)

# Draw subtle separators between list items (list items are 144px tall starting at various y)
item_tops = [390, 534, 678, 822, 966, 1110, 1254, 1398, 1542, 1686]
for y in item_tops:
    # top separator (subtle)
    draw.line([(list_left + 24, y), (list_right - 24, y)], fill="#f0f0f3", width=1)
    # bottom separator for each item to ensure clear separation
    bottom_y = y + 144
    draw.line([(list_left + 24, bottom_y), (list_right - 24, bottom_y)], fill="#fafafb", width=1)

# Subtle vertical divider area on the right where close icons will be pasted (do not draw icons)
right_action_area_x = 1248
draw.line([(right_action_area_x - 12, list_top), (right_action_area_x - 12, list_bottom)], fill="#ffffff", width=1)
# faint guideline to separate action area (very subtle, nearly invisible)
draw.line([(right_action_area_x, list_top + 8), (right_action_area_x, list_bottom - 8)], fill="#f7f7f8", width=1)

# Additional grouped section backgrounds (e.g., for categories that appear as full-width rows)
# Draw light rounded backgrounds behind each detected category row to create subtle cards
category_rows = [48, 390, 534, 678, 822, 966, 1110, 1254, 1398, 1542, 1686]
for y in category_rows:
    # background spans same width as list but is very subtle (almost white) and rounded on small radius
    top = y + 6
    bottom = y + 144 - 6
    draw.rounded_rectangle([(list_left, top), (list_right, bottom)], radius=8,
                           fill="#ffffff", outline=None)

# Bottom navigation bar background and top border
nav_top = 2804
draw.rectangle([(0, nav_top), (1440, 2960)], fill="#ffffff")
draw.line([(0, nav_top), (1440, nav_top)], fill="#e6e6e9", width=2)

# Subtle shadow above the navigation bar to separate content
shadow_top = nav_top - 6
for i, alpha in enumerate([18, 12, 8, 5]):
    y = shadow_top + i
    # very faint horizontal lines to simulate shadow fade
    draw.line([(0, y), (1440, y)], fill="#000000" + "", width=1)  # placeholder: near-invisible line
# NOTE: above shadow lines are extremely faint on purpose (they won't interfere with pasted icons)

# Final top/bottom edge enhancements (very subtle)
draw.line([(0, 0), (1440, 0)], fill="#e9e9ea", width=1)
draw.line([(0, 2959), (1440, 2959)], fill="#e9e9ea", width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_07_2024_4_23_19_27_45f56b06f31541079045047b6d542613-9/00_icon_icon_0.png
try:
    _c0 = get_crop(0, 50, 67)
    canvas.paste(_c0, (1154, 1), _c0)
except Exception:
    pass
layout["icon_0"] = [1154, 1, 1204, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_07_2024_4_23_19_27_45f56b06f31541079045047b6d542613-9/01_icon_7.28.png
try:
    _c1 = get_crop(1, 58, 62)
    canvas.paste(_c1, (115, 2), _c1)
except Exception:
    pass
layout["7.28"] = [115, 2, 173, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_07_2024_4_23_19_27_45f56b06f31541079045047b6d542613-9/02_icon_7.28.png
try:
    _c2 = get_crop(2, 57, 61)
    canvas.paste(_c2, (181, 2), _c2)
except Exception:
    pass
layout["7.28"] = [181, 2, 238, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_07_2024_4_23_19_27_45f56b06f31541079045047b6d542613-9/03_icon_Search_forae.png
try:
    _c3 = get_crop(3, 60, 61)
    canvas.paste(_c3, (311, 3), _c3)
except Exception:
    pass
layout["Search_forae"] = [311, 3, 371, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_07_2024_4_23_19_27_45f56b06f31541079045047b6d542613-9/04_icon_icon_4.png
try:
    _c4 = get_crop(4, 46, 58)
    canvas.paste(_c4, (251, 4), _c4)
except Exception:
    pass
layout["icon_4"] = [251, 4, 297, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_07_2024_4_23_19_27_45f56b06f31541079045047b6d542613-9/05_icon_Cancel.png
try:
    _c5 = get_crop(5, 96, 64)
    canvas.paste(_c5, (1212, 1), _c5)
except Exception:
    pass
layout["Cancel"] = [1212, 1, 1308, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_07_2024_4_23_19_27_45f56b06f31541079045047b6d542613-9/06_icon_Cancel.png
try:
    _c6 = get_crop(6, 149, 144)
    canvas.paste(_c6, (1243, 97), _c6)
except Exception:
    pass
layout["Cancel"] = [1243, 97, 1392, 241]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_07_2024_4_23_19_27_45f56b06f31541079045047b6d542613-9/07_icon_Cancel.png
try:
    _c7 = get_crop(7, 53, 63)
    canvas.paste(_c7, (1318, 1), _c7)
except Exception:
    pass
layout["Cancel"] = [1318, 1, 1371, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_07_2024_4_23_19_27_45f56b06f31541079045047b6d542613-9/08_icon_Coding_Workshop.png
try:
    _c8 = get_crop(8, 1344, 144)
    canvas.paste(_c8, (48, 534), _c8)
except Exception:
    pass
layout["Coding_Workshop"] = [48, 534, 1392, 678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_07_2024_4_23_19_27_45f56b06f31541079045047b6d542613-9/09_icon_Science_Tech.png
try:
    _c9 = get_crop(9, 1344, 144)
    canvas.paste(_c9, (48, 1542), _c9)
except Exception:
    pass
layout["Science_&_Tech"] = [48, 1542, 1392, 1686]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_07_2024_4_23_19_27_45f56b06f31541079045047b6d542613-9/10_icon_Science_Tech.png
try:
    _c10 = get_crop(10, 1344, 144)
    canvas.paste(_c10, (48, 1398), _c10)
except Exception:
    pass
layout["Science_&_Tech"] = [48, 1398, 1392, 1542]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_07_2024_4_23_19_27_45f56b06f31541079045047b6d542613-9/11_icon_Tickets.png
try:
    _c11 = get_crop(11, 288, 156)
    canvas.paste(_c11, (864, 2804), _c11)
except Exception:
    pass
layout["Tickets"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_07_2024_4_23_19_27_45f56b06f31541079045047b6d542613-9/12_icon_7.28.png
try:
    _c12 = get_crop(12, 119, 111)
    canvas.paste(_c12, (58, 116), _c12)
except Exception:
    pass
layout["7.28"] = [58, 116, 177, 227]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_07_2024_4_23_19_27_45f56b06f31541079045047b6d542613-9/13_icon_Close_current_screen.png
try:
    _c13 = get_crop(13, 144, 144)
    canvas.paste(_c13, (1248, 1254), _c13)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 1254, 1392, 1398]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_07_2024_4_23_19_27_45f56b06f31541079045047b6d542613-9/14_icon_Close_current_screen.png
try:
    _c14 = get_crop(14, 144, 144)
    canvas.paste(_c14, (1248, 822), _c14)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 822, 1392, 966]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_07_2024_4_23_19_27_45f56b06f31541079045047b6d542613-9/15_icon_Close_current_screen.png
try:
    _c15 = get_crop(15, 144, 144)
    canvas.paste(_c15, (1248, 1398), _c15)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 1398, 1392, 1542]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_07_2024_4_23_19_27_45f56b06f31541079045047b6d542613-9/16_icon_Close_current_screen.png
try:
    _c16 = get_crop(16, 144, 144)
    canvas.paste(_c16, (1248, 1110), _c16)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 1110, 1392, 1254]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_07_2024_4_23_19_27_45f56b06f31541079045047b6d542613-9/17_icon_Close_current_screen.png
try:
    _c17 = get_crop(17, 144, 144)
    canvas.paste(_c17, (1248, 534), _c17)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 534, 1392, 678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_07_2024_4_23_19_27_45f56b06f31541079045047b6d542613-9/18_icon_Close_current_screen.png
try:
    _c18 = get_crop(18, 144, 144)
    canvas.paste(_c18, (1248, 678), _c18)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 678, 1392, 822]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_07_2024_4_23_19_27_45f56b06f31541079045047b6d542613-9/19_icon_Close_current_screen.png
try:
    _c19 = get_crop(19, 144, 144)
    canvas.paste(_c19, (1248, 1686), _c19)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 1686, 1392, 1830]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_07_2024_4_23_19_27_45f56b06f31541079045047b6d542613-9/20_icon_Close_current_screen.png
try:
    _c20 = get_crop(20, 144, 144)
    canvas.paste(_c20, (1248, 1542), _c20)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 1542, 1392, 1686]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_07_2024_4_23_19_27_45f56b06f31541079045047b6d542613-9/21_icon_Cancel.png
try:
    _c21 = get_crop(21, 144, 144)
    canvas.paste(_c21, (1248, 390), _c21)
except Exception:
    pass
layout["Cancel"] = [1248, 390, 1392, 534]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_07_2024_4_23_19_27_45f56b06f31541079045047b6d542613-9/22_icon_Close_current_screen.png
try:
    _c22 = get_crop(22, 144, 144)
    canvas.paste(_c22, (1248, 966), _c22)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 966, 1392, 1110]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_07_2024_4_23_19_27_45f56b06f31541079045047b6d542613-9/23_icon_Food_Drink.png
try:
    _c23 = get_crop(23, 1344, 144)
    canvas.paste(_c23, (48, 1254), _c23)
except Exception:
    pass
layout["Food_&_Drink"] = [48, 1254, 1392, 1398]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_07_2024_4_23_19_27_45f56b06f31541079045047b6d542613-9/24_icon_Favorites.png
try:
    _c24 = get_crop(24, 288, 156)
    canvas.paste(_c24, (576, 2804), _c24)
except Exception:
    pass
layout["Favorites"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_07_2024_4_23_19_27_45f56b06f31541079045047b6d542613-9/25_icon_7.28.png
try:
    _c25 = get_crop(25, 92, 61)
    canvas.paste(_c25, (15, 2), _c25)
except Exception:
    pass
layout["7.28"] = [15, 2, 107, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_07_2024_4_23_19_27_45f56b06f31541079045047b6d542613-9/26_icon_Coding_Workshop.png
try:
    _c26 = get_crop(26, 1344, 144)
    canvas.paste(_c26, (48, 390), _c26)
except Exception:
    pass
layout["Coding_Workshop"] = [48, 390, 1392, 534]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_07_2024_4_23_19_27_45f56b06f31541079045047b6d542613-9/27_icon_Fitness.png
try:
    _c27 = get_crop(27, 1344, 144)
    canvas.paste(_c27, (48, 678), _c27)
except Exception:
    pass
layout["Fitness"] = [48, 678, 1392, 822]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_07_2024_4_23_19_27_45f56b06f31541079045047b6d542613-9/28_icon_Search_forae.png
try:
    _c28 = get_crop(28, 1344, 191)
    canvas.paste(_c28, (48, 72), _c28)
except Exception:
    pass
layout["Search_forae"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_07_2024_4_23_19_27_45f56b06f31541079045047b6d542613-9/29_icon_Home.png
try:
    _c29 = get_crop(29, 288, 156)
    canvas.paste(_c29, (0, 2804), _c29)
except Exception:
    pass
layout["Home"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_07_2024_4_23_19_27_45f56b06f31541079045047b6d542613-9/30_icon_Search_events.png
try:
    _c30 = get_crop(30, 288, 156)
    canvas.paste(_c30, (288, 2804), _c30)
except Exception:
    pass
layout["Search_events"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_07_2024_4_23_19_27_45f56b06f31541079045047b6d542613-9/31_icon_Basketball.png
try:
    _c31 = get_crop(31, 116, 131)
    canvas.paste(_c31, (26, 1696), _c31)
except Exception:
    pass
layout["Basketball"] = [26, 1696, 142, 1827]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_07_2024_4_23_19_27_45f56b06f31541079045047b6d542613-9/32_icon_Search_forae.png
try:
    _c32 = get_crop(32, 46, 62)
    canvas.paste(_c32, (384, 3), _c32)
except Exception:
    pass
layout["Search_forae"] = [384, 3, 430, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_07_2024_4_23_19_27_45f56b06f31541079045047b6d542613-9/33_icon_Food_Drink.png
try:
    _c33 = get_crop(33, 1344, 144)
    canvas.paste(_c33, (48, 1110), _c33)
except Exception:
    pass
layout["Food_&_Drink"] = [48, 1110, 1392, 1254]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_07_2024_4_23_19_27_45f56b06f31541079045047b6d542613-9/34_icon_More.png
try:
    _c34 = get_crop(34, 288, 156)
    canvas.paste(_c34, (1152, 2804), _c34)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_07_2024_4_23_19_27_45f56b06f31541079045047b6d542613-9/35_icon_Education.png
try:
    _c35 = get_crop(35, 1344, 144)
    canvas.paste(_c35, (48, 822), _c35)
except Exception:
    pass
layout["Education"] = [48, 822, 1392, 966]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_07_2024_4_23_19_27_45f56b06f31541079045047b6d542613-9/36_text_Education.png
try:
    _c36 = get_crop(36, 195, 50)
    canvas.paste(_c36, (162, 1015), _c36)
except Exception:
    pass
layout["Education"] = [162, 1015, 357, 1065]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_07_2024_4_23_19_27_45f56b06f31541079045047b6d542613-9/37_text_Basketball.png
try:
    _c37 = get_crop(37, 201, 43)
    canvas.paste(_c37, (165, 1740), _c37)
except Exception:
    pass
layout["Basketball"] = [165, 1740, 366, 1783]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_07_2024_4_23_19_27_45f56b06f31541079045047b6d542613-9/38_clickable_Education.png
try:
    _c38 = get_crop(38, 1344, 144)
    canvas.paste(_c38, (48, 966), _c38)
except Exception:
    pass
layout["Education"] = [48, 966, 1392, 1110]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_07_2024_4_23_19_27_45f56b06f31541079045047b6d542613-9/39_clickable_Basketball.png
try:
    _c39 = get_crop(39, 1344, 144)
    canvas.paste(_c39, (48, 1686), _c39)
except Exception:
    pass
layout["Basketball"] = [48, 1686, 1392, 1830]
