# page_id: page_seatgeek_094b5cdb02e246858451240263e6ef7f_04
# screenshot: 2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-7.png
# step_index: 4/9
# task: Open SeatGeek. Find the soonest upcoming NBA game in Boston with "Celtics". What is the highest price for each ticket?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Top-level background
draw.rectangle([(0, 0), (1440, 2960)], fill="#fcfcfc")

# Status bar (top ~60px) - light grey area
status_h = 60
draw.rectangle([(0, 0), (1440, status_h)], fill="#e9e9e9")

# Search bar / header area (rounded)
search_x0, search_y0 = 32, 70
search_x1, search_y1 = 1408, 190
draw.rounded_rectangle([(search_x0, search_y0), (search_x1, search_y1)],
                       radius=28, fill="#fafafa", outline="#e6e6e6", width=2)

# subtle shadow under search bar
draw.line([(search_x0, search_y1+6), (search_x1, search_y1+6)], fill="#f0f0f0", width=4)

# Section background cards (soft off-white panels to group items)
# Top results card
draw.rounded_rectangle([(24, 320), (1416, 720)],
                       radius=12, fill="#ffffff", outline="#f0f0f0", width=1)

# Performers card
draw.rounded_rectangle([(24, 760), (1416, 1680)],
                       radius=12, fill="#ffffff", outline="#f0f0f0", width=1)

# Events card
draw.rounded_rectangle([(24, 1820), (1416, 2740)],
                       radius=12, fill="#ffffff", outline="#f0f0f0", width=1)

# Horizontal separators between major sections (thin lines)
sep_color = "#e6e6e6"
sep_positions = [search_y1 + 30, 650, 1008, 1700, 2320, 2750]
for y in sep_positions:
    draw.line([(24, y), (1416, y)], fill=sep_color, width=2)

# Light dividers inside the cards to suggest row separation (subtle, wider margins)
row_x0, row_x1 = 160, 1360
row_seps = [420, 520, 600, 900, 1080, 1260, 1960, 2140, 2320]
for y in row_seps:
    draw.line([(row_x0, y), (row_x1, y)], fill="#f2f2f2", width=1)

# Bottom navigation bar background (area where icons will be pasted)
nav_top = 2792
draw.rectangle([(0, nav_top), (1440, 2960)], fill="#ffffff")
# top border/shadow of nav bar
draw.line([(0, nav_top), (1440, nav_top)], fill="#e6e6e6", width=2)
draw.line([(0, nav_top+3), (1440, nav_top+3)], fill="#f7f7f7", width=1)

# subtle overall vignette strips (very faint) to match screenshot feel
draw.rectangle([(0, status_h), (1440, status_h+2)], fill="#eaeaea")
draw.rectangle([(0, 2960-2), (1440, 2960)], fill="#f6f6f6")

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_04_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-7/00_icon_29_events.png
try:
    _c0 = get_crop(0, 1440, 179)
    canvas.paste(_c0, (0, 1396), _c0)
except Exception:
    pass
layout["29_events"] = [0, 1396, 1440, 1575]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_04_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-7/01_icon_No_events.png
try:
    _c1 = get_crop(1, 1440, 179)
    canvas.paste(_c1, (0, 1575), _c1)
except Exception:
    pass
layout["No_events"] = [0, 1575, 1440, 1754]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_04_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-7/02_icon_Performers.png
try:
    _c2 = get_crop(2, 1440, 179)
    canvas.paste(_c2, (0, 1217), _c2)
except Exception:
    pass
layout["Performers"] = [0, 1217, 1440, 1396]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_04_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-7/03_icon_Events.png
try:
    _c3 = get_crop(3, 1440, 179)
    canvas.paste(_c3, (0, 1963), _c3)
except Exception:
    pass
layout["Events"] = [0, 1963, 1440, 2142]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_04_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-7/04_icon_Top_results.png
try:
    _c4 = get_crop(4, 1440, 179)
    canvas.paste(_c4, (0, 471), _c4)
except Exception:
    pass
layout["Top_results"] = [0, 471, 1440, 650]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_04_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-7/05_icon_Celtics.png
try:
    _c5 = get_crop(5, 1032, 144)
    canvas.paste(_c5, (216, 120), _c5)
except Exception:
    pass
layout["Celtics"] = [216, 120, 1248, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_04_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-7/06_icon_Performers.png
try:
    _c6 = get_crop(6, 1440, 179)
    canvas.paste(_c6, (0, 829), _c6)
except Exception:
    pass
layout["Performers"] = [0, 829, 1440, 1008]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_04_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-7/07_icon_Yesterday.png
try:
    _c7 = get_crop(7, 1440, 179)
    canvas.paste(_c7, (0, 2142), _c7)
except Exception:
    pass
layout["Yesterday"] = [0, 2142, 1440, 2321]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_04_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-7/08_icon_Eastern_Conference_First_Round_Miami_Hea.png
try:
    _c8 = get_crop(8, 1440, 179)
    canvas.paste(_c8, (0, 471), _c8)
except Exception:
    pass
layout["Eastern_Conference_First_"] = [0, 471, 1440, 650]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_04_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-7/09_icon_I_1A.png
try:
    _c9 = get_crop(9, 288, 162)
    canvas.paste(_c9, (288, 2792), _c9)
except Exception:
    pass
layout["I_1A"] = [288, 2792, 576, 2954]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_04_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-7/10_icon_Wed.png
try:
    _c10 = get_crop(10, 1440, 179)
    canvas.paste(_c10, (0, 2321), _c10)
except Exception:
    pass
layout["Wed,"] = [0, 2321, 1440, 2500]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_04_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-7/11_icon_Yesterday.png
try:
    _c11 = get_crop(11, 1440, 179)
    canvas.paste(_c11, (0, 650), _c11)
except Exception:
    pass
layout["Yesterday"] = [0, 650, 1440, 829]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_04_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-7/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 43, 70)
    canvas.paste(_c12, (1155, 0), _c12)
except Exception:
    pass
layout["icon_12"] = [1155, 0, 1198, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_04_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-7/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 57, 57)
    canvas.paste(_c13, (246, 5), _c13)
except Exception:
    pass
layout["icon_13"] = [246, 5, 303, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_04_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-7/14_icon_4.59_Wy.png
try:
    _c14 = get_crop(14, 168, 144)
    canvas.paste(_c14, (48, 120), _c14)
except Exception:
    pass
layout["4.59_Wy"] = [48, 120, 216, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_04_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-7/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 92, 68)
    canvas.paste(_c15, (1220, 0), _c15)
except Exception:
    pass
layout["icon_15"] = [1220, 0, 1312, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_04_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-7/16_icon_Celtic_Woman.png
try:
    _c16 = get_crop(16, 308, 59)
    canvas.paste(_c16, (235, 1426), _c16)
except Exception:
    pass
layout["Celtic_Woman"] = [235, 1426, 543, 1485]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_04_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-7/17_icon_Boston_MA.png
try:
    _c17 = get_crop(17, 1440, 179)
    canvas.paste(_c17, (0, 1963), _c17)
except Exception:
    pass
layout["Boston,_MA"] = [0, 1963, 1440, 2142]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_04_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-7/18_icon_Boston_MA.png
try:
    _c18 = get_crop(18, 1440, 179)
    canvas.paste(_c18, (0, 650), _c18)
except Exception:
    pass
layout["Boston,_MA"] = [0, 650, 1440, 829]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_04_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-7/19_icon_Maine_Celtics.png
try:
    _c19 = get_crop(19, 300, 57)
    canvas.paste(_c19, (233, 1606), _c19)
except Exception:
    pass
layout["Maine_Celtics"] = [233, 1606, 533, 1663]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_04_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-7/20_icon_Boston_MA.png
try:
    _c20 = get_crop(20, 1440, 179)
    canvas.paste(_c20, (0, 829), _c20)
except Exception:
    pass
layout["Boston,_MA"] = [0, 829, 1440, 1008]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_04_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-7/21_icon_4.59_Wy.png
try:
    _c21 = get_crop(21, 45, 59)
    canvas.paste(_c21, (186, 3), _c21)
except Exception:
    pass
layout["4.59_Wy"] = [186, 3, 231, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_04_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-7/22_icon_Boston_MA.png
try:
    _c22 = get_crop(22, 1440, 179)
    canvas.paste(_c22, (0, 2321), _c22)
except Exception:
    pass
layout["Boston,_MA"] = [0, 2321, 1440, 2500]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_04_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-7/23_icon_icon_23.png
try:
    _c23 = get_crop(23, 44, 63)
    canvas.paste(_c23, (1326, 3), _c23)
except Exception:
    pass
layout["icon_23"] = [1326, 3, 1370, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_04_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-7/24_icon_I_1A.png
try:
    _c24 = get_crop(24, 288, 168)
    canvas.paste(_c24, (0, 2792), _c24)
except Exception:
    pass
layout["I_1A"] = [0, 2792, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_04_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-7/25_icon_Tickets.png
try:
    _c25 = get_crop(25, 288, 168)
    canvas.paste(_c25, (576, 2792), _c25)
except Exception:
    pass
layout["Tickets"] = [576, 2792, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_04_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-7/26_icon_Clear.png
try:
    _c26 = get_crop(26, 144, 144)
    canvas.paste(_c26, (1248, 120), _c26)
except Exception:
    pass
layout["Clear"] = [1248, 120, 1392, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_04_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-7/27_icon_Tracking.png
try:
    _c27 = get_crop(27, 288, 168)
    canvas.paste(_c27, (864, 2792), _c27)
except Exception:
    pass
layout["Tracking"] = [864, 2792, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_04_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-7/28_icon_Account.png
try:
    _c28 = get_crop(28, 288, 168)
    canvas.paste(_c28, (1152, 2792), _c28)
except Exception:
    pass
layout["Account"] = [1152, 2792, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_04_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-7/29_icon_4.59_Wy.png
try:
    _c29 = get_crop(29, 51, 61)
    canvas.paste(_c29, (117, 1), _c29)
except Exception:
    pass
layout["4.59_Wy"] = [117, 1, 168, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_04_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-7/30_icon_icon_30.png
try:
    _c30 = get_crop(30, 47, 57)
    canvas.paste(_c30, (318, 6), _c30)
except Exception:
    pass
layout["icon_30"] = [318, 6, 365, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_04_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-7/31_text_Top_results.png
try:
    _c31 = get_crop(31, 295, 72)
    canvas.paste(_c31, (40, 373), _c31)
except Exception:
    pass
layout["Top_results"] = [40, 373, 335, 445]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_04_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-7/32_text_Performers.png
try:
    _c32 = get_crop(32, 293, 54)
    canvas.paste(_c32, (44, 1122), _c32)
except Exception:
    pass
layout["Performers"] = [44, 1122, 337, 1176]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_04_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-7/33_text_Boston_Celtics.png
try:
    _c33 = get_crop(33, 317, 52)
    canvas.paste(_c33, (235, 1253), _c33)
except Exception:
    pass
layout["Boston_Celtics"] = [235, 1253, 552, 1305]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_04_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-7/34_text_18_events.png
try:
    _c34 = get_crop(34, 193, 52)
    canvas.paste(_c34, (234, 1314), _c34)
except Exception:
    pass
layout["18_events"] = [234, 1314, 427, 1366]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_04_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-7/35_text_29_events.png
try:
    _c35 = get_crop(35, 200, 45)
    canvas.paste(_c35, (235, 1495), _c35)
except Exception:
    pass
layout["29_events"] = [235, 1495, 435, 1540]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_04_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-7/36_text_No_events.png
try:
    _c36 = get_crop(36, 201, 40)
    canvas.paste(_c36, (239, 1678), _c36)
except Exception:
    pass
layout["No_events"] = [239, 1678, 440, 1718]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_04_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-7/37_text_Events.png
try:
    _c37 = get_crop(37, 178, 57)
    canvas.paste(_c37, (46, 1868), _c37)
except Exception:
    pass
layout["Events"] = [46, 1868, 224, 1925]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_04_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-7/38_text_Recent_searches.png
try:
    _c38 = get_crop(38, 288, 168)
    canvas.paste(_c38, (0, 2792), _c38)
except Exception:
    pass
layout["Recent_searches"] = [0, 2792, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_04_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-7/39_text_I_1A.png
try:
    _c39 = get_crop(39, 166, 30)
    canvas.paste(_c39, (236, 2770), _c39)
except Exception:
    pass
layout["I_1A"] = [236, 2770, 402, 2800]
