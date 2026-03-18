# page_id: page_seatgeek_6d3c2be0a0b34daf904d1c72c351bd6e_04
# screenshot: 2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-7.png
# step_index: 4/9
# task: Open SeatGeek. Look up "Phoenix Suns" tickets for next upcoming event. Which section are tickets in?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Paint overall background
draw.rectangle([(0, 0), (1440, 2960)], fill="#FBFBFB")

# Status bar area (top)
status_h = 72
draw.rectangle([(0, 0), (1440, status_h)], fill="#EFEFEF")

# Subtle bottom line under status bar
draw.line([(24, status_h), (1416, status_h)], fill="#E6E6E6", width=1)

# Search bar background (rounded rectangle)
search_left = 40
search_top = 88
search_right = 1400
search_bottom = 232
draw.rounded_rectangle(
    [(search_left, search_top), (search_right, search_bottom)],
    radius=32,
    fill="#FFFFFF",
    outline="#E9E9E9",
    width=2
)

# Thin divider under search area
draw.line([(24, 280), (1416, 280)], fill="#EDEDED", width=1)

# Section separators (light)
separator_ys = [650, 820, 1216, 1604, 1784, 1963, 2351, 2792]
for y in separator_ys:
    draw.line([(24, y), (1416, y)], fill="#EFEFEF", width=1)

# Subtle grouping backgrounds for major sections (very light cards)
# Top results card background
draw.rounded_rectangle([(24, 300), (1416, 640)], radius=8, fill="#FFFFFF", outline=None)

# Performers card background
draw.rounded_rectangle([(24, 860), (1416, 1188)], radius=8, fill="#FFFFFF", outline=None)

# Events card background
draw.rounded_rectangle([(24, 1236), (1416, 1588)], radius=8, fill="#FFFFFF", outline=None)

# Venues card background
draw.rounded_rectangle([(24, 1628), (1416, 1948)], radius=8, fill="#FFFFFF", outline=None)

# Recent searches area background (just above bottom nav)
draw.rounded_rectangle([(24, 2375), (1416, 2768)], radius=8, fill="#FFFFFF", outline=None)

# Top subtle shadow under search bar (thin)
draw.line([(search_left+4, search_bottom+6), (search_right-4, search_bottom+6)], fill="#F4F4F4", width=2)

# Bottom navigation bar background and top divider
nav_top = 2792
draw.rectangle([(0, nav_top), (1440, 2960)], fill="#FFFFFF")
draw.line([(24, nav_top), (1416, nav_top)], fill="#E9E9E9", width=1)

# Left and right safe-area guides (very subtle, to match screenshot margins)
draw.line([(24, 0), (24, 2960)], fill="#FCFCFC", width=1)
draw.line([(1416, 0), (1416, 2960)], fill="#FCFCFC", width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_04_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-7/00_icon_Top_results.png
try:
    _c0 = get_crop(0, 1440, 179)
    canvas.paste(_c0, (0, 471), _c0)
except Exception:
    pass
layout["Top_results"] = [0, 471, 1440, 650]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_04_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-7/01_icon_Tomorrow.png
try:
    _c1 = get_crop(1, 1440, 179)
    canvas.paste(_c1, (0, 829), _c1)
except Exception:
    pass
layout["Tomorrow"] = [0, 829, 1440, 1008]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_04_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-7/02_icon_Performers.png
try:
    _c2 = get_crop(2, 1440, 179)
    canvas.paste(_c2, (0, 1217), _c2)
except Exception:
    pass
layout["Performers"] = [0, 1217, 1440, 1396]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_04_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-7/03_icon_Events.png
try:
    _c3 = get_crop(3, 1440, 179)
    canvas.paste(_c3, (0, 1605), _c3)
except Exception:
    pass
layout["Events"] = [0, 1605, 1440, 1784]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_04_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-7/04_icon_Phoenix_Suns.png
try:
    _c4 = get_crop(4, 1032, 144)
    canvas.paste(_c4, (216, 120), _c4)
except Exception:
    pass
layout["Phoenix_Suns"] = [216, 120, 1248, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_04_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-7/05_icon_Venues.png
try:
    _c5 = get_crop(5, 1440, 179)
    canvas.paste(_c5, (0, 1963), _c5)
except Exception:
    pass
layout["Venues"] = [0, 1963, 1440, 2142]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_04_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-7/06_icon_Tomorrow.png
try:
    _c6 = get_crop(6, 1440, 179)
    canvas.paste(_c6, (0, 1784), _c6)
except Exception:
    pass
layout["Tomorrow"] = [0, 1784, 1440, 1963]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_04_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-7/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 60, 62)
    canvas.paste(_c7, (244, 3), _c7)
except Exception:
    pass
layout["icon_7"] = [244, 3, 304, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_04_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-7/08_icon_7_06_my.png
try:
    _c8 = get_crop(8, 168, 144)
    canvas.paste(_c8, (48, 120), _c8)
except Exception:
    pass
layout["7:06_my"] = [48, 120, 216, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_04_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-7/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 43, 70)
    canvas.paste(_c9, (1155, 0), _c9)
except Exception:
    pass
layout["icon_9"] = [1155, 0, 1198, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_04_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-7/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 93, 69)
    canvas.paste(_c10, (1219, 0), _c10)
except Exception:
    pass
layout["icon_10"] = [1219, 0, 1312, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_04_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-7/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 54, 61)
    canvas.paste(_c11, (315, 3), _c11)
except Exception:
    pass
layout["icon_11"] = [315, 3, 369, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_04_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-7/12_icon_Phoenix_AZ.png
try:
    _c12 = get_crop(12, 1440, 179)
    canvas.paste(_c12, (0, 1963), _c12)
except Exception:
    pass
layout["Phoenix,_AZ"] = [0, 1963, 1440, 2142]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_04_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-7/13_icon_Recent_searches.png
try:
    _c13 = get_crop(13, 288, 162)
    canvas.paste(_c13, (288, 2792), _c13)
except Exception:
    pass
layout["Recent_searches"] = [288, 2792, 576, 2954]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_04_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-7/14_icon_Phoenix_AZ.png
try:
    _c14 = get_crop(14, 1440, 179)
    canvas.paste(_c14, (0, 1784), _c14)
except Exception:
    pass
layout["Phoenix,_AZ"] = [0, 1784, 1440, 1963]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_04_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-7/15_icon_7_06_my.png
try:
    _c15 = get_crop(15, 46, 61)
    canvas.paste(_c15, (186, 2), _c15)
except Exception:
    pass
layout["7:06_my"] = [186, 2, 232, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_04_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-7/16_icon_Western_Conference_First_Round_Phoenix_S.png
try:
    _c16 = get_crop(16, 1440, 179)
    canvas.paste(_c16, (0, 1605), _c16)
except Exception:
    pass
layout["Western_Conference_First_"] = [0, 1605, 1440, 1784]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_04_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-7/17_icon_icon_17.png
try:
    _c17 = get_crop(17, 45, 66)
    canvas.paste(_c17, (1326, 2), _c17)
except Exception:
    pass
layout["icon_17"] = [1326, 2, 1371, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_04_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-7/18_icon_Clear.png
try:
    _c18 = get_crop(18, 144, 144)
    canvas.paste(_c18, (1248, 120), _c18)
except Exception:
    pass
layout["Clear"] = [1248, 120, 1392, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_04_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-7/19_icon_Western_Conference_First_Round_Phoenix_S.png
try:
    _c19 = get_crop(19, 1440, 179)
    canvas.paste(_c19, (0, 650), _c19)
except Exception:
    pass
layout["Western_Conference_First_"] = [0, 650, 1440, 829]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_04_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-7/20_icon_Account.png
try:
    _c20 = get_crop(20, 288, 168)
    canvas.paste(_c20, (1152, 2792), _c20)
except Exception:
    pass
layout["Account"] = [1152, 2792, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_04_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-7/21_icon_Recent_searches.png
try:
    _c21 = get_crop(21, 288, 168)
    canvas.paste(_c21, (0, 2792), _c21)
except Exception:
    pass
layout["Recent_searches"] = [0, 2792, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_04_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-7/22_icon_Footprint_Center.png
try:
    _c22 = get_crop(22, 1440, 179)
    canvas.paste(_c22, (0, 2351), _c22)
except Exception:
    pass
layout["Footprint_Center"] = [0, 2351, 1440, 2530]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_04_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-7/23_icon_Western_Conference_First_Round_Phoenix_S.png
try:
    _c23 = get_crop(23, 1440, 179)
    canvas.paste(_c23, (0, 471), _c23)
except Exception:
    pass
layout["Western_Conference_First_"] = [0, 471, 1440, 650]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_04_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-7/24_text_7_06_my.png
try:
    _c24 = get_crop(24, 156, 52)
    canvas.paste(_c24, (19, 9), _c24)
except Exception:
    pass
layout["7:06_my"] = [19, 9, 175, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_04_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-7/25_text_Top_results.png
try:
    _c25 = get_crop(25, 295, 72)
    canvas.paste(_c25, (40, 373), _c25)
except Exception:
    pass
layout["Top_results"] = [40, 373, 335, 445]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_04_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-7/26_text_Performers.png
try:
    _c26 = get_crop(26, 293, 54)
    canvas.paste(_c26, (44, 1122), _c26)
except Exception:
    pass
layout["Performers"] = [44, 1122, 337, 1176]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_04_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-7/27_text_Phoenix_Suns.png
try:
    _c27 = get_crop(27, 292, 52)
    canvas.paste(_c27, (234, 1251), _c27)
except Exception:
    pass
layout["Phoenix_Suns"] = [234, 1251, 526, 1303]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_04_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-7/28_text_18_events.png
try:
    _c28 = get_crop(28, 194, 52)
    canvas.paste(_c28, (234, 1314), _c28)
except Exception:
    pass
layout["18_events"] = [234, 1314, 428, 1366]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_04_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-7/29_text_Events.png
try:
    _c29 = get_crop(29, 177, 54)
    canvas.paste(_c29, (46, 1510), _c29)
except Exception:
    pass
layout["Events"] = [46, 1510, 223, 1564]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_04_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-7/30_text_Venues.png
try:
    _c30 = get_crop(30, 195, 56)
    canvas.paste(_c30, (43, 2256), _c30)
except Exception:
    pass
layout["Venues"] = [43, 2256, 238, 2312]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_04_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-7/31_text_Recent_searches.png
try:
    _c31 = get_crop(31, 288, 168)
    canvas.paste(_c31, (0, 2792), _c31)
except Exception:
    pass
layout["Recent_searches"] = [0, 2792, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_04_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-7/32_clickable_Tickets.png
try:
    _c32 = get_crop(32, 288, 168)
    canvas.paste(_c32, (576, 2792), _c32)
except Exception:
    pass
layout["Tickets"] = [576, 2792, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_04_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-7/33_clickable_Tracking.png
try:
    _c33 = get_crop(33, 288, 168)
    canvas.paste(_c33, (864, 2792), _c33)
except Exception:
    pass
layout["Tracking"] = [864, 2792, 1152, 2960]
