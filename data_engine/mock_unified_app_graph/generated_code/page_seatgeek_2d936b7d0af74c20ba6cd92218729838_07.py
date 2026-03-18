# page_id: page_seatgeek_2d936b7d0af74c20ba6cd92218729838_07
# screenshot: 2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-10.png
# step_index: 7/12
# task: Open SeatGeek. Track "Los Angeles Clippers" and "Golden State Warriors".
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# draw background, status bar, headers, section cards, and separators for SeatGeek UI mock
# available variables: canvas (PIL Image 1440x2960), draw (ImageDraw), fonts: font_sm, font_md, font_lg, font_xl

# Colors
bg = (249, 249, 249)            # page background
status_bg = (241, 241, 241)     # status bar background
stroke = (230, 230, 230)        # divider / subtle strokes
card_fill = (255, 255, 255)     # card surface
muted_stroke = (240, 240, 240)  # lighter stroke for inner separators
shadow = (220, 220, 220)        # faint shadow line

# Fill entire canvas background
draw.rectangle([(0, 0), canvas.size], fill=bg)

# Status bar area (top)
status_h = 72
draw.rectangle([(0, 0), (1440, status_h)], fill=status_bg)
# subtle bottom divider under status bar
draw.line([(0, status_h), (1440, status_h)], fill=stroke, width=1)

# Search bar background (rounded rectangle under status bar)
search_left = 40
search_right = 1400
search_top = 56
search_bottom = 156
search_radius = 24
draw.rounded_rectangle([(search_left, search_top), (search_right, search_bottom)],
                       radius=search_radius, fill=card_fill, outline=stroke, width=1)

# Thin divider below search area
draw.line([(32, search_bottom + 24), (1408, search_bottom + 24)], fill=muted_stroke, width=1)

# Section "Top results" card area (subtle elevated white card)
top_card_top = 420
top_card_bottom = 640
card_margin = 28
draw.rounded_rectangle([(card_margin, top_card_top), (1440 - card_margin, top_card_bottom)],
                       radius=12, fill=card_fill, outline=stroke, width=1)
# separators inside top card (two rows)
row_y1 = top_card_top + 84
row_y2 = top_card_top + 164
draw.line([(card_margin + 24, row_y1), (1440 - card_margin - 24, row_y1)], fill=muted_stroke, width=1)
draw.line([(card_margin + 24, row_y2), (1440 - card_margin - 24, row_y2)], fill=muted_stroke, width=1)

# Divider after top results
divider_y = top_card_bottom + 28
draw.line([(40, divider_y), (1400, divider_y)], fill=stroke, width=1)

# Performers card area
perf_card_top = 1140
perf_card_bottom = 1440
draw.rounded_rectangle([(24, perf_card_top), (1440 - 24, perf_card_bottom)],
                       radius=12, fill=card_fill, outline=stroke, width=1)
# internal separators for performers list (two items)
p_row1 = perf_card_top + 90
p_row2 = perf_card_top + 190
draw.line([(40, p_row1), (1400, p_row1)], fill=muted_stroke, width=1)
draw.line([(40, p_row2), (1400, p_row2)], fill=muted_stroke, width=1)

# Divider after performers
draw.line([(40, perf_card_bottom + 24), (1400, perf_card_bottom + 24)], fill=stroke, width=1)

# Events card area
events_card_top = 1680
events_card_bottom = 1990
draw.rounded_rectangle([(24, events_card_top), (1440 - 24, events_card_bottom)],
                       radius=12, fill=card_fill, outline=stroke, width=1)
# separators for three event rows
e_row1 = events_card_top + 96
e_row2 = events_card_top + 180
e_row3 = events_card_top + 260
draw.line([(40, e_row1), (1400, e_row1)], fill=muted_stroke, width=1)
draw.line([(40, e_row2), (1400, e_row2)], fill=muted_stroke, width=1)
draw.line([(40, e_row3), (1400, e_row3)], fill=muted_stroke, width=1)

# Divider below events
draw.line([(40, events_card_bottom + 24), (1400, events_card_bottom + 24)], fill=stroke, width=1)

# Recent searches card area
recent_card_top = 2320
recent_card_bottom = 2640
draw.rounded_rectangle([(24, recent_card_top), (1440 - 24, recent_card_bottom)],
                       radius=12, fill=card_fill, outline=stroke, width=1)
# separators for two recent items
r_row1 = recent_card_top + 96
r_row2 = recent_card_top + 192
draw.line([(40, r_row1), (1400, r_row1)], fill=muted_stroke, width=1)
draw.line([(40, r_row2), (1400, r_row2)], fill=muted_stroke, width=1)

# Bottom navigation bar background and top divider
nav_top = 2792
nav_height = 168
draw.rectangle([(0, nav_top), (1440, nav_top + nav_height)], fill=card_fill)
draw.line([(0, nav_top), (1440, nav_top)], fill=shadow, width=1)

# Subtle left/right edge shadows for cards to imply elevation
# small vertical shadow lines next to main cards
for y0, y1 in [(top_card_top, top_card_bottom), (perf_card_top, perf_card_bottom),
               (events_card_top, events_card_bottom), (recent_card_top, recent_card_bottom)]:
    # left shadow
    draw.line([(12, y0 + 6), (12, y1 - 6)], fill=(245, 245, 245), width=6)
    # right shadow
    draw.line([(1440 - 12, y0 + 6), (1440 - 12, y1 - 6)], fill=(245, 245, 245), width=6)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_07_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-10/00_icon_Mormi.png
try:
    _c0 = get_crop(0, 1440, 168)
    canvas.paste(_c0, (0, 2530), _c0)
except Exception:
    pass
layout["Mormi"] = [0, 2530, 1440, 2698]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_07_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-10/01_icon_Performers.png
try:
    _c1 = get_crop(1, 1440, 179)
    canvas.paste(_c1, (0, 1217), _c1)
except Exception:
    pass
layout["Performers"] = [0, 1217, 1440, 1396]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_07_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-10/02_icon_No_events.png
try:
    _c2 = get_crop(2, 1440, 179)
    canvas.paste(_c2, (0, 1396), _c2)
except Exception:
    pass
layout["No_events"] = [0, 1396, 1440, 1575]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_07_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-10/03_icon_Top_results.png
try:
    _c3 = get_crop(3, 1440, 179)
    canvas.paste(_c3, (0, 471), _c3)
except Exception:
    pass
layout["Top_results"] = [0, 471, 1440, 650]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_07_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-10/04_icon_Los_Angeles_CA.png
try:
    _c4 = get_crop(4, 1440, 179)
    canvas.paste(_c4, (0, 1784), _c4)
except Exception:
    pass
layout["Los_Angeles,_CA"] = [0, 1784, 1440, 1963]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_07_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-10/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 59, 59)
    canvas.paste(_c5, (245, 4), _c5)
except Exception:
    pass
layout["icon_5"] = [245, 4, 304, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_07_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-10/06_icon_The_Rook_of_Mormon.png
try:
    _c6 = get_crop(6, 288, 162)
    canvas.paste(_c6, (288, 2792), _c6)
except Exception:
    pass
layout["The_Rook_of_Mormon"] = [288, 2792, 576, 2954]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_07_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-10/07_icon_Events.png
try:
    _c7 = get_crop(7, 1440, 179)
    canvas.paste(_c7, (0, 1784), _c7)
except Exception:
    pass
layout["Events"] = [0, 1784, 1440, 1963]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_07_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-10/08_icon_6.53_Wy.png
try:
    _c8 = get_crop(8, 54, 62)
    canvas.paste(_c8, (115, 0), _c8)
except Exception:
    pass
layout["6.53_Wy"] = [115, 0, 169, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_07_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-10/09_icon_Los_Angels_Clippers.png
try:
    _c9 = get_crop(9, 1440, 179)
    canvas.paste(_c9, (0, 1396), _c9)
except Exception:
    pass
layout["Los_Angels_Clippers"] = [0, 1396, 1440, 1575]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_07_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-10/10_icon_Western_Conference_First_Round_Dallas_Ma.png
try:
    _c10 = get_crop(10, 1440, 179)
    canvas.paste(_c10, (0, 471), _c10)
except Exception:
    pass
layout["Western_Conference_First_"] = [0, 471, 1440, 650]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_07_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-10/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 44, 70)
    canvas.paste(_c11, (1155, 0), _c11)
except Exception:
    pass
layout["icon_11"] = [1155, 0, 1199, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_07_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-10/12_icon_6.53_Wy.png
try:
    _c12 = get_crop(12, 45, 61)
    canvas.paste(_c12, (186, 2), _c12)
except Exception:
    pass
layout["6.53_Wy"] = [186, 2, 231, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_07_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-10/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 93, 68)
    canvas.paste(_c13, (1219, 0), _c13)
except Exception:
    pass
layout["icon_13"] = [1219, 0, 1312, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_07_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-10/14_icon_Mormi.png
try:
    _c14 = get_crop(14, 288, 168)
    canvas.paste(_c14, (0, 2792), _c14)
except Exception:
    pass
layout["Mormi"] = [0, 2792, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_07_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-10/15_icon_Tracking.png
try:
    _c15 = get_crop(15, 288, 168)
    canvas.paste(_c15, (864, 2792), _c15)
except Exception:
    pass
layout["Tracking"] = [864, 2792, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_07_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-10/16_icon_Dallas_TX.png
try:
    _c16 = get_crop(16, 1440, 179)
    canvas.paste(_c16, (0, 829), _c16)
except Exception:
    pass
layout["Dallas,_TX"] = [0, 829, 1440, 1008]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_07_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-10/17_icon_Tomorrow.png
try:
    _c17 = get_crop(17, 1440, 179)
    canvas.paste(_c17, (0, 650), _c17)
except Exception:
    pass
layout["Tomorrow"] = [0, 650, 1440, 829]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_07_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-10/18_icon_Fri_Apr_26.png
try:
    _c18 = get_crop(18, 1440, 179)
    canvas.paste(_c18, (0, 829), _c18)
except Exception:
    pass
layout["Fri,_Apr_26,"] = [0, 829, 1440, 1008]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_07_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-10/19_icon_Los_Angeles_CA.png
try:
    _c19 = get_crop(19, 1440, 179)
    canvas.paste(_c19, (0, 650), _c19)
except Exception:
    pass
layout["Los_Angeles,_CA"] = [0, 650, 1440, 829]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_07_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-10/20_icon_Account.png
try:
    _c20 = get_crop(20, 288, 168)
    canvas.paste(_c20, (1152, 2792), _c20)
except Exception:
    pass
layout["Account"] = [1152, 2792, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_07_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-10/21_icon_Tickets.png
try:
    _c21 = get_crop(21, 288, 168)
    canvas.paste(_c21, (576, 2792), _c21)
except Exception:
    pass
layout["Tickets"] = [576, 2792, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_07_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-10/22_icon_Fri.png
try:
    _c22 = get_crop(22, 1440, 179)
    canvas.paste(_c22, (0, 1963), _c22)
except Exception:
    pass
layout["Fri,"] = [0, 1963, 1440, 2142]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_07_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-10/23_icon_Dallas_TX.png
try:
    _c23 = get_crop(23, 1440, 179)
    canvas.paste(_c23, (0, 2142), _c23)
except Exception:
    pass
layout["Dallas,_TX"] = [0, 2142, 1440, 2321]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_07_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-10/24_icon_icon_24.png
try:
    _c24 = get_crop(24, 44, 64)
    canvas.paste(_c24, (1326, 3), _c24)
except Exception:
    pass
layout["icon_24"] = [1326, 3, 1370, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_07_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-10/25_icon_icon_25.png
try:
    _c25 = get_crop(25, 44, 56)
    canvas.paste(_c25, (320, 6), _c25)
except Exception:
    pass
layout["icon_25"] = [320, 6, 364, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_07_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-10/26_icon_Clear.png
try:
    _c26 = get_crop(26, 144, 144)
    canvas.paste(_c26, (1248, 120), _c26)
except Exception:
    pass
layout["Clear"] = [1248, 120, 1392, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_07_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-10/27_icon_Los_Angeles_Clippers.png
try:
    _c27 = get_crop(27, 1440, 179)
    canvas.paste(_c27, (0, 1217), _c27)
except Exception:
    pass
layout["Los_Angeles_Clippers"] = [0, 1217, 1440, 1396]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_07_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-10/28_icon_6.53_Wy.png
try:
    _c28 = get_crop(28, 168, 144)
    canvas.paste(_c28, (48, 120), _c28)
except Exception:
    pass
layout["6.53_Wy"] = [48, 120, 216, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_07_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-10/29_icon_Fri.png
try:
    _c29 = get_crop(29, 1440, 179)
    canvas.paste(_c29, (0, 2142), _c29)
except Exception:
    pass
layout["Fri,"] = [0, 2142, 1440, 2321]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_07_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-10/30_icon_Dallas_TX.png
try:
    _c30 = get_crop(30, 1440, 179)
    canvas.paste(_c30, (0, 1963), _c30)
except Exception:
    pass
layout["Dallas,_TX"] = [0, 1963, 1440, 2142]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_07_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-10/31_icon_icon_31.png
try:
    _c31 = get_crop(31, 42, 54)
    canvas.paste(_c31, (386, 8), _c31)
except Exception:
    pass
layout["icon_31"] = [386, 8, 428, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_07_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-10/32_icon_Western_Conference_First_Round_Dallas_Ma.png
try:
    _c32 = get_crop(32, 1440, 179)
    canvas.paste(_c32, (0, 650), _c32)
except Exception:
    pass
layout["Western_Conference_First_"] = [0, 650, 1440, 829]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_07_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-10/33_icon_Western_Conference_First_Round_LA_Clippe.png
try:
    _c33 = get_crop(33, 1440, 179)
    canvas.paste(_c33, (0, 829), _c33)
except Exception:
    pass
layout["Western_Conference_First_"] = [0, 829, 1440, 1008]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_07_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-10/34_text_Los_Angeles_Clippers.png
try:
    _c34 = get_crop(34, 1032, 144)
    canvas.paste(_c34, (216, 120), _c34)
except Exception:
    pass
layout["Los_Angeles_Clippers"] = [216, 120, 1248, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_07_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-10/35_text_Top_results.png
try:
    _c35 = get_crop(35, 295, 72)
    canvas.paste(_c35, (40, 373), _c35)
except Exception:
    pass
layout["Top_results"] = [40, 373, 335, 445]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_07_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-10/36_text_Performers.png
try:
    _c36 = get_crop(36, 293, 54)
    canvas.paste(_c36, (44, 1122), _c36)
except Exception:
    pass
layout["Performers"] = [44, 1122, 337, 1176]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_07_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-10/37_text_No_events.png
try:
    _c37 = get_crop(37, 201, 43)
    canvas.paste(_c37, (239, 1497), _c37)
except Exception:
    pass
layout["No_events"] = [239, 1497, 440, 1540]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_07_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-10/38_text_Events.png
try:
    _c38 = get_crop(38, 177, 54)
    canvas.paste(_c38, (46, 1691), _c38)
except Exception:
    pass
layout["Events"] = [46, 1691, 223, 1745]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_07_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-10/39_text_Recent_searches.png
try:
    _c39 = get_crop(39, 436, 54)
    canvas.paste(_c39, (44, 2435), _c39)
except Exception:
    pass
layout["Recent_searches"] = [44, 2435, 480, 2489]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_07_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-10/40_text_Los_Angeles_Clippers.png
try:
    _c40 = get_crop(40, 1440, 168)
    canvas.paste(_c40, (0, 2530), _c40)
except Exception:
    pass
layout["Los_Angeles_Clippers"] = [0, 2530, 1440, 2698]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_07_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-10/41_text_The_Rook_of_Mormon.png
try:
    _c41 = get_crop(41, 288, 162)
    canvas.paste(_c41, (288, 2792), _c41)
except Exception:
    pass
layout["The_Rook_of_Mormon"] = [288, 2792, 576, 2954]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_07_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-10/42_text_Mormi.png
try:
    _c42 = get_crop(42, 61, 36)
    canvas.paste(_c42, (75, 2763), _c42)
except Exception:
    pass
layout["Mormi"] = [75, 2763, 136, 2799]
