# page_id: page_eventbrite_31d4c984d33c4fe69d15d4e595fa95f6_10
# screenshot: 2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-12.png
# step_index: 10/14
# task: Open Eventbrite. Look for 'community events' in 'Chicago'. Select the first event happening tomorrow that is not promoted. Check if they have an option for 'refund policy'.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw overall background (ensure clean white)
draw.rectangle([(0, 0), (1440, 2960)], fill=(255, 255, 255))

# Status bar (top area)
status_h = 50
draw.rectangle([(0, 0), (1440, status_h)], fill=(210, 210, 210))  # light grey status bar

# Subtle hairline under status bar
draw.line([(0, status_h), (1440, status_h)], fill=(200, 200, 200), width=1)

# Header / search area background (keeps it visually distinct but do not draw any icons/text)
header_top = status_h
header_bottom = 160
draw.rectangle([(0, header_top), (1440, header_bottom)], fill=(255, 255, 255))

# Blue underline for the search field area (prominent accent)
underline_y = 132
draw.rectangle([(48, underline_y - 2), (1392, underline_y + 2)], fill=(39, 76, 255))  # blue accent

# Divider below header area
draw.line([(48, header_bottom + 10), (1392, header_bottom + 10)], fill=(230, 230, 235), width=1)

# "Recent" section separator (subtle)
recent_div_y = 300
draw.line([(48, recent_div_y), (1392, recent_div_y)], fill=(240, 240, 243), width=1)

# Draw separators between list rows (rows are 144px high starting at y=390 based on detected elements)
list_start_y = 390
row_h = 144
list_left = 48
list_right = 1392
num_rows = 10  # based on detected rows
for i in range(1, num_rows + 1):
    y = list_start_y + i * row_h
    draw.line([(list_left, y), (list_right, y)], fill=(243, 243, 246), width=1)

# Optional subtle rounded card behind the list area (very light to avoid duplicating content)
card_top = list_start_y - 18
card_bottom = list_start_y + num_rows * row_h + 18
# Pillow's rounded_rectangle may be available; fall back to drawing a rounded rect rectangle-like area
try:
    draw.rounded_rectangle([(32, card_top), (1408, card_bottom)], radius=12, fill=(255, 255, 255), outline=(245, 245, 248))
except Exception:
    draw.rectangle([(32, card_top), (1408, card_bottom)], fill=(255, 255, 255), outline=(245, 245, 248))

# Bottom navigation bar background and top divider
nav_top = 2804
draw.line([(0, nav_top), (1440, nav_top)], fill=(225, 225, 229), width=2)
draw.rectangle([(0, nav_top), (1440, 2960)], fill=(255, 255, 255))

# Slight shadow above nav bar
draw.rectangle([(0, nav_top - 6), (1440, nav_top)], fill=(248, 248, 250))

# Final subtle bottom border
draw.line([(0, 2959), (1440, 2959)], fill=(230, 230, 233), width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_10_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-12/00_icon_icon_0.png
try:
    _c0 = get_crop(0, 51, 67)
    canvas.paste(_c0, (1153, 1), _c0)
except Exception:
    pass
layout["icon_0"] = [1153, 1, 1204, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_10_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-12/01_icon_8.07.png
try:
    _c1 = get_crop(1, 61, 64)
    canvas.paste(_c1, (113, 1), _c1)
except Exception:
    pass
layout["8.07"] = [113, 1, 174, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_10_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-12/02_icon_8.07.png
try:
    _c2 = get_crop(2, 58, 63)
    canvas.paste(_c2, (181, 0), _c2)
except Exception:
    pass
layout["8.07"] = [181, 0, 239, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_10_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-12/03_icon_Search_forae.png
try:
    _c3 = get_crop(3, 64, 64)
    canvas.paste(_c3, (309, 1), _c3)
except Exception:
    pass
layout["Search_forae"] = [309, 1, 373, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_10_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-12/04_icon_Cancel.png
try:
    _c4 = get_crop(4, 101, 64)
    canvas.paste(_c4, (1211, 0), _c4)
except Exception:
    pass
layout["Cancel"] = [1211, 0, 1312, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_10_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-12/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 50, 61)
    canvas.paste(_c5, (248, 2), _c5)
except Exception:
    pass
layout["icon_5"] = [248, 2, 298, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_10_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-12/06_icon_Cancel.png
try:
    _c6 = get_crop(6, 56, 62)
    canvas.paste(_c6, (1317, 0), _c6)
except Exception:
    pass
layout["Cancel"] = [1317, 0, 1373, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_10_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-12/07_icon_Art.png
try:
    _c7 = get_crop(7, 1344, 144)
    canvas.paste(_c7, (48, 678), _c7)
except Exception:
    pass
layout["Art"] = [48, 678, 1392, 822]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_10_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-12/08_icon_Cancel.png
try:
    _c8 = get_crop(8, 149, 144)
    canvas.paste(_c8, (1243, 97), _c8)
except Exception:
    pass
layout["Cancel"] = [1243, 97, 1392, 241]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_10_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-12/09_icon_Fitness.png
try:
    _c9 = get_crop(9, 1344, 144)
    canvas.paste(_c9, (48, 534), _c9)
except Exception:
    pass
layout["Fitness"] = [48, 534, 1392, 678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_10_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-12/10_icon_Music_Festival.png
try:
    _c10 = get_crop(10, 1344, 144)
    canvas.paste(_c10, (48, 1542), _c10)
except Exception:
    pass
layout["Music_Festival"] = [48, 1542, 1392, 1686]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_10_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-12/11_icon_8.07.png
try:
    _c11 = get_crop(11, 127, 112)
    canvas.paste(_c11, (53, 115), _c11)
except Exception:
    pass
layout["8.07"] = [53, 115, 180, 227]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_10_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-12/12_icon_Tickets.png
try:
    _c12 = get_crop(12, 288, 156)
    canvas.paste(_c12, (864, 2804), _c12)
except Exception:
    pass
layout["Tickets"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_10_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-12/13_icon_Close_current_screen.png
try:
    _c13 = get_crop(13, 144, 144)
    canvas.paste(_c13, (1248, 822), _c13)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 822, 1392, 966]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_10_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-12/14_icon_Search_forae.png
try:
    _c14 = get_crop(14, 1344, 144)
    canvas.paste(_c14, (48, 390), _c14)
except Exception:
    pass
layout["Search_forae"] = [48, 390, 1392, 534]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_10_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-12/15_icon_Coding_Workshop.png
try:
    _c15 = get_crop(15, 1344, 144)
    canvas.paste(_c15, (48, 1398), _c15)
except Exception:
    pass
layout["Coding_Workshop"] = [48, 1398, 1392, 1542]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_10_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-12/16_icon_Close_current_screen.png
try:
    _c16 = get_crop(16, 144, 144)
    canvas.paste(_c16, (1248, 534), _c16)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 534, 1392, 678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_10_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-12/17_icon_Close_current_screen.png
try:
    _c17 = get_crop(17, 144, 144)
    canvas.paste(_c17, (1248, 678), _c17)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 678, 1392, 822]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_10_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-12/18_icon_Close_current_screen.png
try:
    _c18 = get_crop(18, 144, 144)
    canvas.paste(_c18, (1248, 1254), _c18)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 1254, 1392, 1398]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_10_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-12/19_icon_Close_current_screen.png
try:
    _c19 = get_crop(19, 144, 144)
    canvas.paste(_c19, (1248, 1398), _c19)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 1398, 1392, 1542]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_10_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-12/20_icon_Close_current_screen.png
try:
    _c20 = get_crop(20, 144, 144)
    canvas.paste(_c20, (1248, 1686), _c20)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 1686, 1392, 1830]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_10_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-12/21_icon_Close_current_screen.png
try:
    _c21 = get_crop(21, 144, 144)
    canvas.paste(_c21, (1248, 1110), _c21)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 1110, 1392, 1254]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_10_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-12/22_icon_Favorites.png
try:
    _c22 = get_crop(22, 288, 156)
    canvas.paste(_c22, (576, 2804), _c22)
except Exception:
    pass
layout["Favorites"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_10_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-12/23_icon_Close_current_screen.png
try:
    _c23 = get_crop(23, 144, 144)
    canvas.paste(_c23, (1248, 1542), _c23)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 1542, 1392, 1686]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_10_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-12/24_icon_Search_forae.png
try:
    _c24 = get_crop(24, 1344, 191)
    canvas.paste(_c24, (48, 72), _c24)
except Exception:
    pass
layout["Search_forae"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_10_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-12/25_icon_Cancel.png
try:
    _c25 = get_crop(25, 144, 144)
    canvas.paste(_c25, (1248, 390), _c25)
except Exception:
    pass
layout["Cancel"] = [1248, 390, 1392, 534]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_10_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-12/26_icon_Home.png
try:
    _c26 = get_crop(26, 288, 156)
    canvas.paste(_c26, (0, 2804), _c26)
except Exception:
    pass
layout["Home"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_10_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-12/27_icon_Search_forae.png
try:
    _c27 = get_crop(27, 48, 65)
    canvas.paste(_c27, (383, 2), _c27)
except Exception:
    pass
layout["Search_forae"] = [383, 2, 431, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_10_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-12/28_icon_Search_events.png
try:
    _c28 = get_crop(28, 288, 156)
    canvas.paste(_c28, (288, 2804), _c28)
except Exception:
    pass
layout["Search_events"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_10_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-12/29_icon_Close_current_screen.png
try:
    _c29 = get_crop(29, 144, 144)
    canvas.paste(_c29, (1248, 966), _c29)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 966, 1392, 1110]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_10_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-12/30_icon_Music.png
try:
    _c30 = get_crop(30, 115, 130)
    canvas.paste(_c30, (26, 1697), _c30)
except Exception:
    pass
layout["Music"] = [26, 1697, 141, 1827]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_10_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-12/31_icon_More.png
try:
    _c31 = get_crop(31, 288, 156)
    canvas.paste(_c31, (1152, 2804), _c31)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_10_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-12/32_icon_Coding_Workshop.png
try:
    _c32 = get_crop(32, 1344, 144)
    canvas.paste(_c32, (48, 1254), _c32)
except Exception:
    pass
layout["Coding_Workshop"] = [48, 1254, 1392, 1398]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_10_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-12/33_icon_8.07.png
try:
    _c33 = get_crop(33, 93, 61)
    canvas.paste(_c33, (13, 3), _c33)
except Exception:
    pass
layout["8.07"] = [13, 3, 106, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_10_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-12/34_icon_Coding_Workshop.png
try:
    _c34 = get_crop(34, 1344, 144)
    canvas.paste(_c34, (48, 1110), _c34)
except Exception:
    pass
layout["Coding_Workshop"] = [48, 1110, 1392, 1254]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_10_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-12/35_text_Recent.png
try:
    _c35 = get_crop(35, 200, 56)
    canvas.paste(_c35, (46, 301), _c35)
except Exception:
    pass
layout["Recent"] = [46, 301, 246, 357]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_10_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-12/36_text_Food_and_Drink.png
try:
    _c36 = get_crop(36, 286, 51)
    canvas.paste(_c36, (164, 872), _c36)
except Exception:
    pass
layout["Food_and_Drink"] = [164, 872, 450, 923]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_10_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-12/37_text_Education.png
try:
    _c37 = get_crop(37, 195, 50)
    canvas.paste(_c37, (162, 1015), _c37)
except Exception:
    pass
layout["Education"] = [162, 1015, 357, 1065]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_10_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-12/38_text_Food_Drink.png
try:
    _c38 = get_crop(38, 245, 52)
    canvas.paste(_c38, (164, 1734), _c38)
except Exception:
    pass
layout["Food_&_Drink"] = [164, 1734, 409, 1786]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_10_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-12/39_clickable_Food_and_Drink.png
try:
    _c39 = get_crop(39, 1344, 144)
    canvas.paste(_c39, (48, 822), _c39)
except Exception:
    pass
layout["Food_and_Drink"] = [48, 822, 1392, 966]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_10_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-12/40_clickable_Education.png
try:
    _c40 = get_crop(40, 1344, 144)
    canvas.paste(_c40, (48, 966), _c40)
except Exception:
    pass
layout["Education"] = [48, 966, 1392, 1110]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_10_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-12/41_clickable_Food_Drink.png
try:
    _c41 = get_crop(41, 1344, 144)
    canvas.paste(_c41, (48, 1686), _c41)
except Exception:
    pass
layout["Food_&_Drink"] = [48, 1686, 1392, 1830]
