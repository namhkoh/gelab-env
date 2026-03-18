# page_id: page_seatgeek_094b5cdb02e246858451240263e6ef7f_06
# screenshot: 2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-9.png
# step_index: 6/9
# task: Open SeatGeek. Find the soonest upcoming NBA game in Boston with "Celtics". What is the highest price for each ticket?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# draw mobile UI background and structural elements for SeatGeek-like page

# Colors
BG = (243, 245, 247)         # overall light grey background
STATUS_BG = (150, 155, 160)  # status bar darker grey
HEADER_BG = (255, 255, 255)  # white header pill
CARD_BG = (255, 255, 255)    # white cards
BORDER = (219, 222, 224)     # light border color
DIVIDER = (224, 226, 228)    # subtle divider
SHADOW = (200, 203, 205)     # simple shadow color for offset "shadow" rectangles

w, h = canvas.size

# Fill overall background
draw.rectangle([(0, 0), (w, h)], fill=BG)

# Status bar area (~50-64px tall)
status_h = 64
draw.rectangle([(0, 0), (w, status_h)], fill=STATUS_BG)

# Header / toolbar background (rounded pill) - use detected header region as the pill background
# Detected header pos=(48,120) size=1344x156
header_x0, header_y0 = 48, 120
header_w, header_h = 1344, 156
header_x1 = header_x0 + header_w
header_y1 = header_y0 + header_h
header_radius = 78

# subtle drop shadow (offset)
shadow_offset = 6
draw.rounded_rectangle(
    [(header_x0, header_y0 + shadow_offset), (header_x1, header_y1 + shadow_offset)],
    radius=header_radius,
    fill=SHADOW
)
# header background
draw.rounded_rectangle(
    [(header_x0, header_y0), (header_x1, header_y1)],
    radius=header_radius,
    fill=HEADER_BG,
    outline=BORDER,
    width=1
)

# Divider line to separate header from filter chips area
divider_y = header_y1 + 24
draw.line([(48, divider_y), (w - 48, divider_y)], fill=DIVIDER, width=1)

# Large seating map card background (rounded) centered under chips
# Positioning approximated to match screenshot composition
map_x0, map_y0 = 36, divider_y + 36
map_x1, map_y1 = w - 36, 1840
map_radius = 48

# shadow for map card
draw.rounded_rectangle(
    [(map_x0 + 6, map_y0 + 8), (map_x1 + 6, map_y1 + 8)],
    radius=map_radius,
    fill=SHADOW
)
# map card white background with light border
draw.rounded_rectangle(
    [(map_x0, map_y0), (map_x1, map_y1)],
    radius=map_radius,
    fill=CARD_BG,
    outline=BORDER,
    width=2
)

# Subtle inner ring background behind the seating map to mimic slight inset
inset = 16
draw.rounded_rectangle(
    [(map_x0 + inset, map_y0 + inset), (map_x1 - inset, map_y1 - inset)],
    radius=map_radius - 12,
    outline=(236, 238, 239),
    width=1
)

# Listings container card at bottom (white with rounded top corners)
# Detected "1097 Listings" area around y ~2029, draw a full-width card from ~1960 down
list_x0, list_y0 = 0, 1960
list_x1, list_y1 = w, h
list_radius = 32

# shadow for listings card
draw.rounded_rectangle(
    [(list_x0 + 4, list_y0 + 6), (list_x1 + 4, list_y1 + 6)],
    radius=list_radius,
    fill=SHADOW
)

draw.rounded_rectangle(
    [(list_x0, list_y0), (list_x1, list_y1)],
    radius=list_radius,
    fill=CARD_BG,
    outline=BORDER,
    width=1
)

# Thin top divider under the listings header area
top_div_y = list_y0 + 110
draw.line([(24, top_div_y), (w - 24, top_div_y)], fill=DIVIDER, width=1)

# Draw separators for two list items (approximate positions)
item1_y = top_div_y + 170  # first listing card vertical center area
item_sep1 = item1_y + 180
item_sep2 = item_sep1 + 220

draw.line([(24, item_sep1), (w - 24, item_sep1)], fill=DIVIDER, width=1)
draw.line([(24, item_sep2), (w - 24, item_sep2)], fill=DIVIDER, width=1)

# Small subtle pill behind the "Sort by price" control area on the listings header (background only)
# Detected sort control pos ~ (961,1989) size=455x144 - we'll draw a faint rounded rect behind it
sort_x0, sort_y0 = 961, list_y0 + 29
sort_w, sort_h = 455, 144
draw.rounded_rectangle(
    [(sort_x0, sort_y0), (sort_x0 + sort_w, sort_y0 + sort_h)],
    radius=72,
    fill=BG,
    outline=BORDER,
    width=1
)

# Final subtle bottom safe area line
draw.line([(0, h - 1), (w, h - 1)], fill=DIVIDER, width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_06_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-9/00_icon_Include_fees.png
try:
    _c0 = get_crop(0, 335, 108)
    canvas.paste(_c0, (544, 312), _c0)
except Exception:
    pass
layout["Include_fees"] = [544, 312, 879, 420]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_06_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-9/01_icon_Courtside.png
try:
    _c1 = get_crop(1, 286, 108)
    canvas.paste(_c1, (915, 312), _c1)
except Exception:
    pass
layout["Courtside"] = [915, 312, 1201, 420]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_06_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-9/02_icon_Quantity.png
try:
    _c2 = get_crop(2, 268, 108)
    canvas.paste(_c2, (240, 312), _c2)
except Exception:
    pass
layout["Quantity"] = [240, 312, 508, 420]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_06_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-9/03_icon_Tit.png
try:
    _c3 = get_crop(3, 156, 108)
    canvas.paste(_c3, (48, 312), _c3)
except Exception:
    pass
layout["Tit"] = [48, 312, 204, 420]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_06_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-9/04_icon_Center.png
try:
    _c4 = get_crop(4, 203, 108)
    canvas.paste(_c4, (1237, 312), _c4)
except Exception:
    pass
layout["Center"] = [1237, 312, 1440, 420]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_06_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-9/05_icon_9.5.png
try:
    _c5 = get_crop(5, 1440, 371)
    canvas.paste(_c5, (0, 2589), _c5)
except Exception:
    pass
layout["9.5"] = [0, 2589, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_06_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-9/06_icon_E_Conf_Ist_Rnd_TBD_at_Celtics_Gm_2_HG_2.png
try:
    _c6 = get_crop(6, 1344, 156)
    canvas.paste(_c6, (48, 120), _c6)
except Exception:
    pass
layout["E_Conf_Ist_Rnd:_TBD_at_Ce"] = [48, 120, 1392, 276]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_06_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-9/07_icon_1097_Listings.png
try:
    _c7 = get_crop(7, 1440, 455)
    canvas.paste(_c7, (0, 2134), _c7)
except Exception:
    pass
layout["1097_Listings"] = [0, 2134, 1440, 2589]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_06_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-9/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 48, 65)
    canvas.paste(_c8, (1154, 1), _c8)
except Exception:
    pass
layout["icon_8"] = [1154, 1, 1202, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_06_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-9/09_icon_5.00_Wy.png
try:
    _c9 = get_crop(9, 54, 60)
    canvas.paste(_c9, (181, 2), _c9)
except Exception:
    pass
layout["5.00_Wy"] = [181, 2, 235, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_06_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-9/10_icon_5.00_Wy.png
try:
    _c10 = get_crop(10, 64, 62)
    canvas.paste(_c10, (113, 1), _c10)
except Exception:
    pass
layout["5.00_Wy"] = [113, 1, 177, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_06_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-9/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 101, 63)
    canvas.paste(_c11, (1214, 1), _c11)
except Exception:
    pass
layout["icon_11"] = [1214, 1, 1315, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_06_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-9/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 64, 60)
    canvas.paste(_c12, (242, 2), _c12)
except Exception:
    pass
layout["icon_12"] = [242, 2, 306, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_06_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-9/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 53, 58)
    canvas.paste(_c13, (1320, 3), _c13)
except Exception:
    pass
layout["icon_13"] = [1320, 3, 1373, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_06_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-9/14_icon_Amazing_deal.png
try:
    _c14 = get_crop(14, 1440, 455)
    canvas.paste(_c14, (0, 2134), _c14)
except Exception:
    pass
layout["Amazing_deal"] = [0, 2134, 1440, 2589]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_06_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-9/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 59, 61)
    canvas.paste(_c15, (313, 2), _c15)
except Exception:
    pass
layout["icon_15"] = [313, 2, 372, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_06_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-9/16_icon_icon_16.png
try:
    _c16 = get_crop(16, 51, 63)
    canvas.paste(_c16, (382, 1), _c16)
except Exception:
    pass
layout["icon_16"] = [382, 1, 433, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_06_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-9/17_icon_Center.png
try:
    _c17 = get_crop(17, 156, 156)
    canvas.paste(_c17, (1236, 120), _c17)
except Exception:
    pass
layout["Center"] = [1236, 120, 1392, 276]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_06_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-9/18_icon_Sort_by_price.png
try:
    _c18 = get_crop(18, 455, 144)
    canvas.paste(_c18, (961, 1989), _c18)
except Exception:
    pass
layout["Sort_by_price"] = [961, 1989, 1416, 2133]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_06_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-9/19_text_STANDING.png
try:
    _c19 = get_crop(19, 97, 25)
    canvas.paste(_c19, (670, 689), _c19)
except Exception:
    pass
layout["~STANDING"] = [670, 689, 767, 714]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_06_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-9/20_text_STUI.png
try:
    _c20 = get_crop(20, 62, 29)
    canvas.paste(_c20, (488, 717), _c20)
except Exception:
    pass
layout["STUI"] = [488, 717, 550, 746]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_06_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-9/21_text_STU2.png
try:
    _c21 = get_crop(21, 65, 29)
    canvas.paste(_c21, (566, 717), _c21)
except Exception:
    pass
layout["~STU2"] = [566, 717, 631, 746]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_06_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-9/22_text_STUS.png
try:
    _c22 = get_crop(22, 64, 29)
    canvas.paste(_c22, (807, 717), _c22)
except Exception:
    pass
layout["STUS"] = [807, 717, 871, 746]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_06_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-9/23_text_STU6.png
try:
    _c23 = get_crop(23, 64, 29)
    canvas.paste(_c23, (888, 717), _c23)
except Exception:
    pass
layout["STU6"] = [888, 717, 952, 746]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_06_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-9/24_text_S608.png
try:
    _c24 = get_crop(24, 60, 27)
    canvas.paste(_c24, (543, 876), _c24)
except Exception:
    pass
layout["S608"] = [543, 876, 603, 903]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_06_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-9/25_text_S611.png
try:
    _c25 = get_crop(25, 57, 27)
    canvas.paste(_c25, (668, 876), _c25)
except Exception:
    pass
layout["S611"] = [668, 876, 725, 903]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_06_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-9/26_text_613.png
try:
    _c26 = get_crop(26, 60, 27)
    canvas.paste(_c26, (751, 876), _c26)
except Exception:
    pass
layout["613"] = [751, 876, 811, 903]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_06_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-9/27_text_S615.png
try:
    _c27 = get_crop(27, 59, 29)
    canvas.paste(_c27, (837, 874), _c27)
except Exception:
    pass
layout["S615"] = [837, 874, 896, 903]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_06_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-9/28_text_20.png
try:
    _c28 = get_crop(28, 37, 27)
    canvas.paste(_c28, (469, 953), _c28)
except Exception:
    pass
layout["20"] = [469, 953, 506, 980]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_06_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-9/29_text_19.png
try:
    _c29 = get_crop(29, 32, 27)
    canvas.paste(_c29, (448, 1022), _c29)
except Exception:
    pass
layout["19"] = [448, 1022, 480, 1049]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_06_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-9/30_text_L12.png
try:
    _c30 = get_crop(30, 44, 30)
    canvas.paste(_c30, (270, 1045), _c30)
except Exception:
    pass
layout["L12"] = [270, 1045, 314, 1075]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_06_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-9/31_text_Log.png
try:
    _c31 = get_crop(31, 45, 27)
    canvas.paste(_c31, (250, 1108), _c31)
except Exception:
    pass
layout["Log"] = [250, 1108, 295, 1135]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_06_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-9/32_text_SCORERS.png
try:
    _c32 = get_crop(32, 83, 25)
    canvas.paste(_c32, (677, 1121), _c32)
except Exception:
    pass
layout["~SCORERS"] = [677, 1121, 760, 1146]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_06_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-9/33_text_Lo5.png
try:
    _c33 = get_crop(33, 44, 27)
    canvas.paste(_c33, (247, 1186), _c33)
except Exception:
    pass
layout["Lo5"] = [247, 1186, 291, 1213]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_06_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-9/34_text_16.png
try:
    _c34 = get_crop(34, 34, 27)
    canvas.paste(_c34, (361, 1357), _c34)
except Exception:
    pass
layout["16"] = [361, 1357, 395, 1384]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_06_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-9/35_text_S645.png
try:
    _c35 = get_crop(35, 60, 27)
    canvas.paste(_c35, (543, 1517), _c35)
except Exception:
    pass
layout["S645"] = [543, 1517, 603, 1544]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_06_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-9/36_text_S642.png
try:
    _c36 = get_crop(36, 60, 27)
    canvas.paste(_c36, (668, 1517), _c36)
except Exception:
    pass
layout["S642"] = [668, 1517, 728, 1544]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_06_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-9/37_text_S640.png
try:
    _c37 = get_crop(37, 62, 27)
    canvas.paste(_c37, (751, 1517), _c37)
except Exception:
    pass
layout["S640"] = [751, 1517, 813, 1544]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_06_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-9/38_text_S638.png
try:
    _c38 = get_crop(38, 59, 27)
    canvas.paste(_c38, (837, 1517), _c38)
except Exception:
    pass
layout["S638"] = [837, 1517, 896, 1544]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_06_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-9/39_text_RAF28.png
try:
    _c39 = get_crop(39, 74, 27)
    canvas.paste(_c39, (485, 1674), _c39)
except Exception:
    pass
layout["RAF28"] = [485, 1674, 559, 1701]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_06_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-9/40_text_STANDING.png
try:
    _c40 = get_crop(40, 97, 25)
    canvas.paste(_c40, (670, 1683), _c40)
except Exception:
    pass
layout["~STANDING"] = [670, 1683, 767, 1708]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_06_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-9/41_text_1097_Listings.png
try:
    _c41 = get_crop(41, 359, 74)
    canvas.paste(_c41, (51, 2029), _c41)
except Exception:
    pass
layout["1097_Listings"] = [51, 2029, 410, 2103]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_06_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-9/42_text_S177_each.png
try:
    _c42 = get_crop(42, 1440, 371)
    canvas.paste(_c42, (0, 2589), _c42)
except Exception:
    pass
layout["S177_each"] = [0, 2589, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_06_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-9/43_text_Price_includes_fees.png
try:
    _c43 = get_crop(43, 1440, 371)
    canvas.paste(_c43, (0, 2589), _c43)
except Exception:
    pass
layout["Price_includes_fees"] = [0, 2589, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_06_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-9/44_text_9.5.png
try:
    _c44 = get_crop(44, 52, 36)
    canvas.paste(_c44, (501, 2809), _c44)
except Exception:
    pass
layout["9.5"] = [501, 2809, 553, 2845]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_06_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-9/45_text_Amazing_deal.png
try:
    _c45 = get_crop(45, 1440, 371)
    canvas.paste(_c45, (0, 2589), _c45)
except Exception:
    pass
layout["Amazing_deal"] = [0, 2589, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_06_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-9/46_text_1-3_tickets.png
try:
    _c46 = get_crop(46, 219, 48)
    canvas.paste(_c46, (488, 2872), _c46)
except Exception:
    pass
layout["1-3_tickets"] = [488, 2872, 707, 2920]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_06_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-9/47_text_an.png
try:
    _c47 = get_crop(47, 54, 9)
    canvas.paste(_c47, (524, 2948), _c47)
except Exception:
    pass
layout["an"] = [524, 2948, 578, 2957]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_06_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-9/48_text_RAF38.png
try:
    _c48 = get_crop(48, 84, 53)
    canvas.paste(_c48, (393, 721), _c48)
except Exception:
    pass
layout["RAF38"] = [393, 721, 477, 774]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_06_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-9/49_text_RAFZ.png
try:
    _c49 = get_crop(49, 73, 54)
    canvas.paste(_c49, (966, 721), _c49)
except Exception:
    pass
layout["~RAFZ"] = [966, 721, 1039, 775]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_06_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-9/50_text_RAF37.png
try:
    _c50 = get_crop(50, 84, 75)
    canvas.paste(_c50, (285, 758), _c50)
except Exception:
    pass
layout["RAF37"] = [285, 758, 369, 833]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_06_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-9/51_text_RAF8.png
try:
    _c51 = get_crop(51, 80, 71)
    canvas.paste(_c51, (1073, 760), _c51)
except Exception:
    pass
layout["~RAF8"] = [1073, 760, 1153, 831]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_06_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-9/52_text_S605.png
try:
    _c52 = get_crop(52, 71, 50)
    canvas.paste(_c52, (407, 886), _c52)
except Exception:
    pass
layout["S605"] = [407, 886, 478, 936]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_06_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-9/53_text_-5618.png
try:
    _c53 = get_crop(53, 71, 50)
    canvas.paste(_c53, (960, 887), _c53)
except Exception:
    pass
layout["-5618"] = [960, 887, 1031, 937]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_06_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-9/54_text_NLOUNGE.png
try:
    _c54 = get_crop(54, 106, 62)
    canvas.paste(_c54, (931, 1440), _c54)
except Exception:
    pass
layout["~NLOUNGE"] = [931, 1440, 1037, 1502]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_06_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-9/55_text_S548.png
try:
    _c55 = get_crop(55, 75, 55)
    canvas.paste(_c55, (418, 1441), _c55)
except Exception:
    pass
layout["~S548"] = [418, 1441, 493, 1496]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_06_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-9/56_text_RAF1Z.png
try:
    _c56 = get_crop(56, 83, 52)
    canvas.paste(_c56, (949, 1649), _c56)
except Exception:
    pass
layout["RAF1Z"] = [949, 1649, 1032, 1701]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_06_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-9/57_clickable_Back.png
try:
    _c57 = get_crop(57, 156, 156)
    canvas.paste(_c57, (48, 120), _c57)
except Exception:
    pass
layout["Back"] = [48, 120, 204, 276]
