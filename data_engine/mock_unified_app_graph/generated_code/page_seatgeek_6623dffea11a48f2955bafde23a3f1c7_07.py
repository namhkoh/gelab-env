# page_id: page_seatgeek_6623dffea11a48f2955bafde23a3f1c7_07
# screenshot: 2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-10.png
# step_index: 7/9
# task: Open SeatGeek. Search "New York Knicks" and select the second upcoming event, show the location of the event and track the event.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Background and structural UI drawing for SeatGeek-like page
# Uses provided 'canvas' (PIL Image) and 'draw' (ImageDraw.Draw) and fonts.

# Full-page background (dominant light gray-blue)
draw.rectangle([(0, 0), (1440, 2960)], fill="#eef1f4")

# Status bar area at top (~80px tall)
status_h = 80
draw.rectangle([(0, 0), (1440, status_h)], fill="#e2e5e8")

# Subtle bottom divider under status bar
draw.line([(0, status_h), (1440, status_h)], fill="#d6d9dc", width=1)

# Header / toolbar pill (background behind title & icons)
header_box = (40, 100, 1400, 260)  # left, top, right, bottom
draw.rounded_rectangle(header_box, radius=40, fill="#ffffff", outline="#dde0e3", width=1)

# Slight highlight above header (very subtle)
draw.line([(header_box[0]+8, header_box[1]+8), (header_box[2]-8, header_box[1]+8)], fill="#f7f8f9", width=1)

# Divider below header separating filters area
header_bottom = header_box[3] + 20
draw.line([(24, header_bottom), (1416, header_bottom)], fill="#e6e9ec", width=1)

# Filters row background area (small subtle background band)
filters_band = (0, header_bottom, 1440, header_bottom + 150)
draw.rectangle(filters_band, fill="#eef2f5")

# Light horizontal rule to anchor filter chips
draw.line([(24, header_bottom + 150), (1416, header_bottom + 150)], fill="#e1e4e7", width=1)

# Large seating-map card background (rounded) - center upper area
map_box = (80, header_bottom + 60, 1360, 1700)  # left, top, right, bottom
# outer card shadow (simple darker band)
shadow_offset = 8
draw.rounded_rectangle(
    (map_box[0] + shadow_offset, map_box[1] + shadow_offset, map_box[2] + shadow_offset, map_box[3] + shadow_offset),
    radius=40, fill="#d8dbe0"
)
# main card
draw.rounded_rectangle(map_box, radius=40, fill="#ffffff", outline="#d7d9dc", width=1)

# inner map background (slightly cooler gray inside the card to separate from card border)
inner_inset = 18
inner_box = (map_box[0] + inner_inset, map_box[1] + inner_inset, map_box[2] - inner_inset, map_box[3] - inner_inset)
draw.rounded_rectangle(inner_box, radius=34, fill="#f6f8fa")

# central oval background for arena seating plan (visual frame behind dynamic content)
oval_inset = 50
oval_box = (inner_box[0] + oval_inset, inner_box[1] + oval_inset, inner_box[2] - oval_inset, inner_box[3] - oval_inset)
draw.ellipse(oval_box, fill="#f1f4f6", outline="#dfe3e6", width=2)

# Court area placeholder background (subtle rectangle where the court graphic will be pasted)
court_h = 170
court_w = 420
court_x = (inner_box[0] + inner_box[2]) // 2 - court_w // 2
court_y = inner_box[1] + (inner_box[3] - inner_box[1]) // 2 - court_h // 2
court_box = (court_x, court_y, court_x + court_w, court_y + court_h)
draw.rounded_rectangle(court_box, radius=12, fill="#f7fbff", outline="#cfd6db", width=1)

# Top-to-listings separation (a faint divider under the map area)
map_bottom = map_box[3] + 24
draw.line([(24, map_bottom), (1416, map_bottom)], fill="#e4e7ea", width=1)

# Listings panel: large white rounded panel from about y=1960 to bottom
list_panel_top = 1960
panel_box = (0, list_panel_top, 1440, 2960)
# panel shadow (simple band)
draw.rectangle((0, list_panel_top - 10, 1440, list_panel_top), fill="#e8ebee")
# panel body
draw.rounded_rectangle(panel_box, radius=28, fill="#ffffff", outline="#e1e4e7", width=1)

# Small top divider inside panel (under the header area of the panel)
panel_inner_top = list_panel_top + 80
draw.line([(32, panel_inner_top), (1408, panel_inner_top)], fill="#eceded", width=1)

# Listing item separators (two visible items; draw separators between them)
first_item_top = panel_inner_top + 40
item_height = 240
sep_y1 = first_item_top + item_height
sep_y2 = sep_y1 + item_height
draw.line([(24, sep_y1), (1416, sep_y1)], fill="#f0f1f2", width=1)
draw.line([(24, sep_y2), (1416, sep_y2)], fill="#f0f1f2", width=1)

# Right-side "Sort" area background (subtle pill area top-right of listing panel)
sort_pill = (1040, list_panel_top + 20, 1416, list_panel_top + 80)
draw.rounded_rectangle(sort_pill, radius=28, fill="#ffffff", outline="#eceff1", width=1)

# Final subtle bottom inset to suggest more content below fold
bottom_inset = 2880
draw.rectangle([(0, bottom_inset), (1440, 2960)], fill="#ffffff")

# End of structural drawing.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_07_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-10/00_icon_Include_fees.png
try:
    _c0 = get_crop(0, 335, 108)
    canvas.paste(_c0, (544, 312), _c0)
except Exception:
    pass
layout["Include_fees"] = [544, 312, 879, 420]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_07_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-10/01_icon_Courtside.png
try:
    _c1 = get_crop(1, 286, 108)
    canvas.paste(_c1, (915, 312), _c1)
except Exception:
    pass
layout["Courtside"] = [915, 312, 1201, 420]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_07_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-10/02_icon_Quantity.png
try:
    _c2 = get_crop(2, 268, 108)
    canvas.paste(_c2, (240, 312), _c2)
except Exception:
    pass
layout["Quantity"] = [240, 312, 508, 420]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_07_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-10/03_icon_Tit.png
try:
    _c3 = get_crop(3, 156, 108)
    canvas.paste(_c3, (48, 312), _c3)
except Exception:
    pass
layout["Tit"] = [48, 312, 204, 420]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_07_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-10/04_icon_1201_Listings.png
try:
    _c4 = get_crop(4, 1440, 455)
    canvas.paste(_c4, (0, 2134), _c4)
except Exception:
    pass
layout["1201_Listings"] = [0, 2134, 1440, 2589]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_07_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-10/05_icon_7.7.png
try:
    _c5 = get_crop(5, 1440, 371)
    canvas.paste(_c5, (0, 2589), _c5)
except Exception:
    pass
layout["7.7"] = [0, 2589, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_07_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-10/06_icon_Center.png
try:
    _c6 = get_crop(6, 203, 108)
    canvas.paste(_c6, (1237, 312), _c6)
except Exception:
    pass
layout["Center"] = [1237, 312, 1440, 420]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_07_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-10/07_icon_E_Conf_Ist_Rnd_Knicks_at_76ers_Gm_3_HG_1.png
try:
    _c7 = get_crop(7, 1344, 156)
    canvas.paste(_c7, (48, 120), _c7)
except Exception:
    pass
layout["E_Conf_Ist_Rnd:_Knicks_at"] = [48, 120, 1392, 276]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_07_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-10/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 50, 65)
    canvas.paste(_c8, (1153, 1), _c8)
except Exception:
    pass
layout["icon_8"] = [1153, 1, 1203, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_07_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-10/09_icon_6.58_my.png
try:
    _c9 = get_crop(9, 68, 62)
    canvas.paste(_c9, (110, 1), _c9)
except Exception:
    pass
layout["6.58_my"] = [110, 1, 178, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_07_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-10/10_icon_6.58_my.png
try:
    _c10 = get_crop(10, 54, 61)
    canvas.paste(_c10, (181, 2), _c10)
except Exception:
    pass
layout["6.58_my"] = [181, 2, 235, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_07_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-10/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 67, 61)
    canvas.paste(_c11, (240, 2), _c11)
except Exception:
    pass
layout["icon_11"] = [240, 2, 307, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_07_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-10/12_icon_i.png
try:
    _c12 = get_crop(12, 104, 63)
    canvas.paste(_c12, (1212, 1), _c12)
except Exception:
    pass
layout["i_"] = [1212, 1, 1316, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_07_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-10/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 53, 56)
    canvas.paste(_c13, (1320, 4), _c13)
except Exception:
    pass
layout["icon_13"] = [1320, 4, 1373, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_07_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-10/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 60, 64)
    canvas.paste(_c14, (313, 1), _c14)
except Exception:
    pass
layout["icon_14"] = [313, 1, 373, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_07_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-10/15_icon_i.png
try:
    _c15 = get_crop(15, 156, 156)
    canvas.paste(_c15, (1236, 120), _c15)
except Exception:
    pass
layout["i_"] = [1236, 120, 1392, 276]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_07_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-10/16_icon_icon_16.png
try:
    _c16 = get_crop(16, 51, 65)
    canvas.paste(_c16, (382, 0), _c16)
except Exception:
    pass
layout["icon_16"] = [382, 0, 433, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_07_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-10/17_icon_Sort_by_price.png
try:
    _c17 = get_crop(17, 455, 144)
    canvas.paste(_c17, (961, 1989), _c17)
except Exception:
    pass
layout["Sort_by_price"] = [961, 1989, 1416, 2133]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_07_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-10/18_text_ASSEMBLY_ROOM.png
try:
    _c18 = get_crop(18, 152, 25)
    canvas.paste(_c18, (643, 696), _c18)
except Exception:
    pass
layout["ASSEMBLY_ROOM"] = [643, 696, 795, 721]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_07_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-10/19_text_LS41.png
try:
    _c19 = get_crop(19, 55, 27)
    canvas.paste(_c19, (652, 867), _c19)
except Exception:
    pass
layout["LS41"] = [652, 867, 707, 894]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_07_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-10/20_text_LS35.png
try:
    _c20 = get_crop(20, 66, 38)
    canvas.paste(_c20, (420, 883), _c20)
except Exception:
    pass
layout["~LS35"] = [420, 883, 486, 921]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_07_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-10/21_text_LS49.png
try:
    _c21 = get_crop(21, 59, 29)
    canvas.paste(_c21, (955, 886), _c21)
except Exception:
    pass
layout["LS49"] = [955, 886, 1014, 915]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_07_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-10/22_text_LS32.png
try:
    _c22 = get_crop(22, 60, 27)
    canvas.paste(_c22, (335, 941), _c22)
except Exception:
    pass
layout["LS32"] = [335, 941, 395, 968]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_07_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-10/23_text_CB22.png
try:
    _c23 = get_crop(23, 64, 29)
    canvas.paste(_c23, (925, 948), _c23)
except Exception:
    pass
layout["CB22"] = [925, 948, 989, 977]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_07_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-10/24_text_LS30.png
try:
    _c24 = get_crop(24, 57, 29)
    canvas.paste(_c24, (291, 990), _c24)
except Exception:
    pass
layout["LS30"] = [291, 990, 348, 1019]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_07_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-10/25_text_LS28.png
try:
    _c25 = get_crop(25, 57, 30)
    canvas.paste(_c25, (259, 1047), _c25)
except Exception:
    pass
layout["LS28"] = [259, 1047, 316, 1077]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_07_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-10/26_text_LS56.png
try:
    _c26 = get_crop(26, 57, 30)
    canvas.paste(_c26, (1124, 1047), _c26)
except Exception:
    pass
layout["LS56"] = [1124, 1047, 1181, 1077]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_07_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-10/27_text_LS26.png
try:
    _c27 = get_crop(27, 60, 29)
    canvas.paste(_c27, (240, 1101), _c27)
except Exception:
    pass
layout["LS26"] = [240, 1101, 300, 1130]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_07_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-10/28_text_SCORERS.png
try:
    _c28 = get_crop(28, 83, 25)
    canvas.paste(_c28, (677, 1126), _c28)
except Exception:
    pass
layout["SCORERS"] = [677, 1126, 760, 1151]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_07_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-10/29_text_LS24H.png
try:
    _c29 = get_crop(29, 87, 37)
    canvas.paste(_c29, (235, 1150), _c29)
except Exception:
    pass
layout["LS24H"] = [235, 1150, 322, 1187]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_07_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-10/30_text_LLS.png
try:
    _c30 = get_crop(30, 41, 27)
    canvas.paste(_c30, (76, 1212), _c30)
except Exception:
    pass
layout["LLS"] = [76, 1212, 117, 1239]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_07_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-10/31_text_LS21.png
try:
    _c31 = get_crop(31, 55, 29)
    canvas.paste(_c31, (236, 1235), _c31)
except Exception:
    pass
layout["LS21"] = [236, 1235, 291, 1264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_07_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-10/32_text_R118.png
try:
    _c32 = get_crop(32, 62, 30)
    canvas.paste(_c32, (874, 1230), _c32)
except Exception:
    pass
layout["R118"] = [874, 1230, 936, 1260]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_07_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-10/33_text_LLA.png
try:
    _c33 = get_crop(33, 41, 27)
    canvas.paste(_c33, (78, 1267), _c33)
except Exception:
    pass
layout["LLA"] = [78, 1267, 119, 1294]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_07_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-10/34_text_LS19.png
try:
    _c34 = get_crop(34, 60, 30)
    canvas.paste(_c34, (240, 1290), _c34)
except Exception:
    pass
layout["LS19"] = [240, 1290, 300, 1320]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_07_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-10/35_text_111.png
try:
    _c35 = get_crop(35, 43, 27)
    canvas.paste(_c35, (557, 1304), _c35)
except Exception:
    pass
layout["111"] = [557, 1304, 600, 1331]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_07_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-10/36_text_115.png
try:
    _c36 = get_crop(36, 46, 27)
    canvas.paste(_c36, (837, 1304), _c36)
except Exception:
    pass
layout["115"] = [837, 1304, 883, 1331]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_07_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-10/37_text_LL3.png
try:
    _c37 = get_crop(37, 44, 27)
    canvas.paste(_c37, (85, 1327), _c37)
except Exception:
    pass
layout["LL3"] = [85, 1327, 129, 1354]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_07_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-10/38_text_LS17.png
try:
    _c38 = get_crop(38, 57, 30)
    canvas.paste(_c38, (259, 1343), _c38)
except Exception:
    pass
layout["LS17"] = [259, 1343, 316, 1373]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_07_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-10/39_text_LS67.png
try:
    _c39 = get_crop(39, 58, 28)
    canvas.paste(_c39, (1121, 1343), _c39)
except Exception:
    pass
layout["LS67"] = [1121, 1343, 1179, 1371]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_07_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-10/40_text_LL2.png
try:
    _c40 = get_crop(40, 43, 31)
    canvas.paste(_c40, (104, 1383), _c40)
except Exception:
    pass
layout["LL2"] = [104, 1383, 147, 1414]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_07_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-10/41_text_LS69.png
try:
    _c41 = get_crop(41, 60, 27)
    canvas.paste(_c41, (1089, 1401), _c41)
except Exception:
    pass
layout["LS69"] = [1089, 1401, 1149, 1428]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_07_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-10/42_text_LLI.png
try:
    _c42 = get_crop(42, 41, 29)
    canvas.paste(_c42, (125, 1436), _c42)
except Exception:
    pass
layout["LLI"] = [125, 1436, 166, 1465]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_07_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-10/43_text_C16W.png
try:
    _c43 = get_crop(43, 68, 27)
    canvas.paste(_c43, (976, 1445), _c43)
except Exception:
    pass
layout["C16W"] = [976, 1445, 1044, 1472]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_07_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-10/44_text_LS12.png
try:
    _c44 = get_crop(44, 59, 29)
    canvas.paste(_c44, (361, 1471), _c44)
except Exception:
    pass
layout["LS12"] = [361, 1471, 420, 1500]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_07_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-10/45_text_C1OW.png
try:
    _c45 = get_crop(45, 69, 32)
    canvas.paste(_c45, (448, 1468), _c45)
except Exception:
    pass
layout["C1OW"] = [448, 1468, 517, 1500]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_07_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-10/46_text_Ls9.png
try:
    _c46 = get_crop(46, 48, 29)
    canvas.paste(_c46, (446, 1510), _c46)
except Exception:
    pass
layout["Ls9"] = [446, 1510, 494, 1539]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_07_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-10/47_text_LS74.png
try:
    _c47 = get_crop(47, 60, 32)
    canvas.paste(_c47, (966, 1496), _c47)
except Exception:
    pass
layout["LS74"] = [966, 1496, 1026, 1528]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_07_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-10/48_text_LS2.png
try:
    _c48 = get_crop(48, 43, 27)
    canvas.paste(_c48, (666, 1526), _c48)
except Exception:
    pass
layout["LS2"] = [666, 1526, 709, 1553]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_07_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-10/49_text_S82.png
try:
    _c49 = get_crop(49, 60, 29)
    canvas.paste(_c49, (721, 1524), _c49)
except Exception:
    pass
layout["[S82"] = [721, 1524, 781, 1553]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_07_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-10/50_text_PRESS_BOX_1.png
try:
    _c50 = get_crop(50, 108, 20)
    canvas.paste(_c50, (808, 1696), _c50)
except Exception:
    pass
layout["PRESS_BOX_1"] = [808, 1696, 916, 1716]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_07_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-10/51_text_1201_Listings.png
try:
    _c51 = get_crop(51, 349, 78)
    canvas.paste(_c51, (51, 2026), _c51)
except Exception:
    pass
layout["1201_Listings"] = [51, 2026, 400, 2104]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_07_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-10/52_text_S146_each.png
try:
    _c52 = get_crop(52, 1440, 455)
    canvas.paste(_c52, (0, 2134), _c52)
except Exception:
    pass
layout["S146_each"] = [0, 2134, 1440, 2589]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_07_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-10/53_text_Price_includes_fees.png
try:
    _c53 = get_crop(53, 1440, 455)
    canvas.paste(_c53, (0, 2134), _c53)
except Exception:
    pass
layout["Price_includes_fees"] = [0, 2134, 1440, 2589]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_07_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-10/54_text_8.5.png
try:
    _c54 = get_crop(54, 50, 30)
    canvas.paste(_c54, (502, 2356), _c54)
except Exception:
    pass
layout["8.5"] = [502, 2356, 552, 2386]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_07_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-10/55_text_Great_deal.png
try:
    _c55 = get_crop(55, 1440, 455)
    canvas.paste(_c55, (0, 2134), _c55)
except Exception:
    pass
layout["Great_deal"] = [0, 2134, 1440, 2589]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_07_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-10/56_text_2_tickets.png
try:
    _c56 = get_crop(56, 180, 43)
    canvas.paste(_c56, (489, 2420), _c56)
except Exception:
    pass
layout["2_tickets"] = [489, 2420, 669, 2463]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_07_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-10/57_text_Assembly_Room.png
try:
    _c57 = get_crop(57, 1440, 455)
    canvas.paste(_c57, (0, 2134), _c57)
except Exception:
    pass
layout["Assembly_Room"] = [0, 2134, 1440, 2589]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_07_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-10/58_text_S149_each.png
try:
    _c58 = get_crop(58, 1440, 371)
    canvas.paste(_c58, (0, 2589), _c58)
except Exception:
    pass
layout["S149_each"] = [0, 2589, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_07_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-10/59_text_Price_includes_fees.png
try:
    _c59 = get_crop(59, 1440, 371)
    canvas.paste(_c59, (0, 2589), _c59)
except Exception:
    pass
layout["Price_includes_fees"] = [0, 2589, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_07_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-10/60_text_7.7.png
try:
    _c60 = get_crop(60, 41, 31)
    canvas.paste(_c60, (504, 2810), _c60)
except Exception:
    pass
layout["7.7"] = [504, 2810, 545, 2841]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_07_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-10/61_text_Great_deal.png
try:
    _c61 = get_crop(61, 1440, 371)
    canvas.paste(_c61, (0, 2589), _c61)
except Exception:
    pass
layout["Great_deal"] = [0, 2589, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_07_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-10/62_text_1ticket.png
try:
    _c62 = get_crop(62, 147, 43)
    canvas.paste(_c62, (491, 2876), _c62)
except Exception:
    pass
layout["1ticket"] = [491, 2876, 638, 2919]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_07_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-10/63_text_PRESS.png
try:
    _c63 = get_crop(63, 63, 35)
    canvas.paste(_c63, (408, 1673), _c63)
except Exception:
    pass
layout["PRESS"] = [408, 1673, 471, 1708]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_07_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-10/64_clickable_Back.png
try:
    _c64 = get_crop(64, 156, 156)
    canvas.paste(_c64, (48, 120), _c64)
except Exception:
    pass
layout["Back"] = [48, 120, 204, 276]
