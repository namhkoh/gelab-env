# page_id: page_seatgeek_2d936b7d0af74c20ba6cd92218729838_04
# screenshot: 2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-7.png
# step_index: 4/12
# task: Open SeatGeek. Track "Los Angeles Clippers" and "Golden State Warriors".
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw background and structural UI elements for a 1440x2960 mobile UI
# Uses provided 'canvas' (PIL.Image) and 'draw' (PIL.ImageDraw)

# Overall canvas background (dominant white)
draw.rectangle((0, 0, 1440, 2960), fill=(255, 255, 255))

# Status bar area (light gray background)
status_h = 64
draw.rectangle((0, 0, 1440, status_h), fill=(240, 240, 240))

# Subtle bottom edge for status bar to separate from content
draw.line((0, status_h - 1, 1440, status_h - 1), fill=(225, 225, 225), width=1)

# Search bar / toolbar background (rounded pill behind icons/text)
search_left = 40
search_right = 1400
search_top = 72
search_bottom = 140
search_radius = 22
draw.rounded_rectangle((search_left, search_top, search_right, search_bottom),
                       radius=search_radius, fill=(250, 250, 250), outline=(230, 230, 230), width=1)

# Thin divider under toolbar area
toolbar_divider_y = search_bottom + 12
draw.line((24, toolbar_divider_y, 1416, toolbar_divider_y), fill=(230, 230, 230), width=1)

# Top "cards"/group container (subtle off-white block behind Top results list)
top_group_top = toolbar_divider_y + 18
top_group_bottom = 760
draw.rounded_rectangle((24, top_group_top, 1416, top_group_bottom),
                       radius=12, fill=(254, 254, 254), outline=None)

# Inner separators for items in top group (positions approximate, leave space for pasted items)
top_item_y = top_group_top + 120
for i in range(1, 3):
    y = top_group_top + i * 170
    draw.line((40, y, 1400, y), fill=(235, 235, 235), width=1)

# Divider below Top results group
draw.line((24, top_group_bottom + 6, 1416, top_group_bottom + 6), fill=(230, 230, 230), width=1)

# Performers section container
performers_top = top_group_bottom + 28
performers_bottom = 1360
draw.rounded_rectangle((24, performers_top, 1416, performers_bottom),
                       radius=12, fill=(255, 255, 255), outline=None)

# Separators between performer items
for j in range(1, 3):
    y = performers_top + j * 160
    draw.line((40, y, 1400, y), fill=(235, 235, 235), width=1)

# Divider under Performers
draw.line((24, performers_bottom + 6, 1416, performers_bottom + 6), fill=(230, 230, 230), width=1)

# Events section container
events_top = performers_bottom + 28
events_bottom = 2020
draw.rounded_rectangle((24, events_top, 1416, events_bottom),
                       radius=12, fill=(254, 254, 254), outline=None)

# Separators for events list (approximate rows)
for k in range(1, 4):
    y = events_top + k * 170
    draw.line((40, y, 1400, y), fill=(235, 235, 235), width=1)

# Divider under Events
draw.line((24, events_bottom + 6, 1416, events_bottom + 6), fill=(230, 230, 230), width=1)

# Recent searches container (subtle background)
recent_top = events_bottom + 28
recent_bottom = 2600
draw.rounded_rectangle((24, recent_top, 1416, recent_bottom),
                       radius=12, fill=(255, 255, 255), outline=None)

# Separators for recent search rows
for m in range(1, 4):
    y = recent_top + m * 160
    draw.line((40, y, 1400, y), fill=(240, 240, 240), width=1)

# Top shadow line above bottom navigation bar area
nav_top = 2730
draw.line((24, nav_top, 1416, nav_top), fill=(225, 225, 225), width=1)

# Bottom navigation background (keep white but add slight top gradient/rect to emphasize separation)
draw.rectangle((0, nav_top, 1440, 2960), fill=(255, 255, 255))
draw.rectangle((0, nav_top, 1440, nav_top + 6), fill=(245, 245, 245))

# Safe padding guides (non-intrusive, light) - do not obscure content; optional faint margins
draw.line((24, 0, 24, 2960), fill=(250, 250, 250), width=1)
draw.line((1416, 0, 1416, 2960), fill=(250, 250, 250), width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_04_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-7/00_icon_Performers.png
try:
    _c0 = get_crop(0, 1440, 179)
    canvas.paste(_c0, (0, 1217), _c0)
except Exception:
    pass
layout["Performers"] = [0, 1217, 1440, 1396]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_04_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-7/01_icon_No_events.png
try:
    _c1 = get_crop(1, 1440, 179)
    canvas.paste(_c1, (0, 1396), _c1)
except Exception:
    pass
layout["No_events"] = [0, 1396, 1440, 1575]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_04_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-7/02_icon_Top_results.png
try:
    _c2 = get_crop(2, 1440, 179)
    canvas.paste(_c2, (0, 471), _c2)
except Exception:
    pass
layout["Top_results"] = [0, 471, 1440, 650]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_04_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-7/03_icon_Events.png
try:
    _c3 = get_crop(3, 1440, 179)
    canvas.paste(_c3, (0, 1784), _c3)
except Exception:
    pass
layout["Events"] = [0, 1784, 1440, 1963]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_04_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-7/04_icon_The.png
try:
    _c4 = get_crop(4, 288, 168)
    canvas.paste(_c4, (0, 2792), _c4)
except Exception:
    pass
layout["The"] = [0, 2792, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_04_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-7/05_icon_Morm.png
try:
    _c5 = get_crop(5, 1440, 168)
    canvas.paste(_c5, (0, 2530), _c5)
except Exception:
    pass
layout["Morm"] = [0, 2530, 1440, 2698]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_04_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-7/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 58, 57)
    canvas.paste(_c6, (245, 5), _c6)
except Exception:
    pass
layout["icon_6"] = [245, 5, 303, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_04_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-7/07_icon_Western_Conference_First_Round_Dallas_Ma.png
try:
    _c7 = get_crop(7, 1440, 179)
    canvas.paste(_c7, (0, 471), _c7)
except Exception:
    pass
layout["Western_Conference_First_"] = [0, 471, 1440, 650]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_04_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-7/08_icon_Fri_Apr_26.png
try:
    _c8 = get_crop(8, 1440, 179)
    canvas.paste(_c8, (0, 829), _c8)
except Exception:
    pass
layout["Fri,_Apr_26,"] = [0, 829, 1440, 1008]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_04_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-7/09_icon_Los_Angels_Clippers.png
try:
    _c9 = get_crop(9, 1440, 179)
    canvas.paste(_c9, (0, 1396), _c9)
except Exception:
    pass
layout["Los_Angels_Clippers"] = [0, 1396, 1440, 1575]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_04_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-7/10_icon_Fri.png
try:
    _c10 = get_crop(10, 1440, 179)
    canvas.paste(_c10, (0, 1963), _c10)
except Exception:
    pass
layout["Fri,"] = [0, 1963, 1440, 2142]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_04_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-7/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 42, 69)
    canvas.paste(_c11, (1156, 0), _c11)
except Exception:
    pass
layout["icon_11"] = [1156, 0, 1198, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_04_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-7/12_icon_Los_Angeles_CA.png
try:
    _c12 = get_crop(12, 1440, 179)
    canvas.paste(_c12, (0, 1784), _c12)
except Exception:
    pass
layout["Los_Angeles,_CA"] = [0, 1784, 1440, 1963]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_04_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-7/13_icon_Tomorrow.png
try:
    _c13 = get_crop(13, 1440, 179)
    canvas.paste(_c13, (0, 650), _c13)
except Exception:
    pass
layout["Tomorrow"] = [0, 650, 1440, 829]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_04_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-7/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 95, 67)
    canvas.paste(_c14, (1219, 0), _c14)
except Exception:
    pass
layout["icon_14"] = [1219, 0, 1314, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_04_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-7/15_icon_Los_Angeles_Clippers.png
try:
    _c15 = get_crop(15, 1032, 144)
    canvas.paste(_c15, (216, 120), _c15)
except Exception:
    pass
layout["Los_Angeles_Clippers]"] = [216, 120, 1248, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_04_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-7/16_icon_Fri.png
try:
    _c16 = get_crop(16, 1440, 179)
    canvas.paste(_c16, (0, 2142), _c16)
except Exception:
    pass
layout["Fri,"] = [0, 2142, 1440, 2321]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_04_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-7/17_icon_Dallas_TX.png
try:
    _c17 = get_crop(17, 1440, 179)
    canvas.paste(_c17, (0, 829), _c17)
except Exception:
    pass
layout["Dallas,_TX"] = [0, 829, 1440, 1008]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_04_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-7/18_icon_6.53_Wy.png
try:
    _c18 = get_crop(18, 44, 60)
    canvas.paste(_c18, (186, 3), _c18)
except Exception:
    pass
layout["6.53_Wy"] = [186, 3, 230, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_04_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-7/19_icon_icon_19.png
try:
    _c19 = get_crop(19, 44, 63)
    canvas.paste(_c19, (1326, 3), _c19)
except Exception:
    pass
layout["icon_19"] = [1326, 3, 1370, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_04_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-7/20_icon_6.53_Wy.png
try:
    _c20 = get_crop(20, 168, 144)
    canvas.paste(_c20, (48, 120), _c20)
except Exception:
    pass
layout["6.53_Wy"] = [48, 120, 216, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_04_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-7/21_icon_Los_Angeles_CA.png
try:
    _c21 = get_crop(21, 1440, 179)
    canvas.paste(_c21, (0, 650), _c21)
except Exception:
    pass
layout["Los_Angeles,_CA"] = [0, 650, 1440, 829]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_04_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-7/22_icon_icon_22.png
try:
    _c22 = get_crop(22, 45, 56)
    canvas.paste(_c22, (319, 7), _c22)
except Exception:
    pass
layout["icon_22"] = [319, 7, 364, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_04_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-7/23_icon_Dallas_TX.png
try:
    _c23 = get_crop(23, 1440, 179)
    canvas.paste(_c23, (0, 2142), _c23)
except Exception:
    pass
layout["Dallas,_TX"] = [0, 2142, 1440, 2321]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_04_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-7/24_icon_Los_Angeles_Clippers.png
try:
    _c24 = get_crop(24, 1440, 179)
    canvas.paste(_c24, (0, 1217), _c24)
except Exception:
    pass
layout["Los_Angeles_Clippers"] = [0, 1217, 1440, 1396]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_04_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-7/25_icon_Account.png
try:
    _c25 = get_crop(25, 288, 168)
    canvas.paste(_c25, (1152, 2792), _c25)
except Exception:
    pass
layout["Account"] = [1152, 2792, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_04_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-7/26_icon_Clear.png
try:
    _c26 = get_crop(26, 144, 144)
    canvas.paste(_c26, (1248, 120), _c26)
except Exception:
    pass
layout["Clear"] = [1248, 120, 1392, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_04_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-7/27_icon_ion_Kina.png
try:
    _c27 = get_crop(27, 288, 162)
    canvas.paste(_c27, (288, 2792), _c27)
except Exception:
    pass
layout["ion_Kina"] = [288, 2792, 576, 2954]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_04_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-7/28_icon_Tracking.png
try:
    _c28 = get_crop(28, 288, 168)
    canvas.paste(_c28, (864, 2792), _c28)
except Exception:
    pass
layout["Tracking"] = [864, 2792, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_04_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-7/29_text_6.53_Wy.png
try:
    _c29 = get_crop(29, 153, 49)
    canvas.paste(_c29, (19, 12), _c29)
except Exception:
    pass
layout["6.53_Wy"] = [19, 12, 172, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_04_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-7/30_text_Top_results.png
try:
    _c30 = get_crop(30, 295, 72)
    canvas.paste(_c30, (40, 373), _c30)
except Exception:
    pass
layout["Top_results"] = [40, 373, 335, 445]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_04_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-7/31_text_Performers.png
try:
    _c31 = get_crop(31, 293, 54)
    canvas.paste(_c31, (44, 1122), _c31)
except Exception:
    pass
layout["Performers"] = [44, 1122, 337, 1176]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_04_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-7/32_text_No_events.png
try:
    _c32 = get_crop(32, 201, 43)
    canvas.paste(_c32, (239, 1497), _c32)
except Exception:
    pass
layout["No_events"] = [239, 1497, 440, 1540]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_04_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-7/33_text_Events.png
try:
    _c33 = get_crop(33, 177, 54)
    canvas.paste(_c33, (46, 1691), _c33)
except Exception:
    pass
layout["Events"] = [46, 1691, 223, 1745]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_04_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-7/34_text_Recent_searches.png
try:
    _c34 = get_crop(34, 436, 54)
    canvas.paste(_c34, (44, 2435), _c34)
except Exception:
    pass
layout["Recent_searches"] = [44, 2435, 480, 2489]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_04_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-7/35_text_The_Book_of_Mormon.png
try:
    _c35 = get_crop(35, 1440, 168)
    canvas.paste(_c35, (0, 2530), _c35)
except Exception:
    pass
layout["The_Book_of_Mormon"] = [0, 2530, 1440, 2698]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_04_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-7/36_text_The.png
try:
    _c36 = get_crop(36, 87, 41)
    canvas.paste(_c36, (237, 2760), _c36)
except Exception:
    pass
layout["The"] = [237, 2760, 324, 2801]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_04_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-7/37_clickable_Tickets.png
try:
    _c37 = get_crop(37, 288, 168)
    canvas.paste(_c37, (576, 2792), _c37)
except Exception:
    pass
layout["Tickets"] = [576, 2792, 864, 2960]
