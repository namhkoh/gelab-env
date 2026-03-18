# page_id: page_seatgeek_1cc69540849e491bb4fc78ed1f09c554_06
# screenshot: 2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-9.png
# step_index: 6/7
# task: Open SeatGeek. Search "Madison Square Garden". Select the next upcoming event. Who are the performers of the event?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Background fill
draw.rectangle((0, 0, 1440, 2960), fill="#f1f4f6")  # overall pale bluish-gray background

# Status bar area (top)
status_h = 80
draw.rectangle((0, 0, 1440, status_h), fill="#e6e9eb")
draw.line((0, status_h, 1440, status_h), fill="#d0d4d6", width=1)

# Header pill (rounded) with subtle shadow
header_x0, header_x1 = 48, 1392
header_y0, header_y1 = 56, 176
# shadow
draw.rounded_rectangle((header_x0+2, header_y0+6, header_x1+2, header_y1+6), radius=40, fill="#e0e3e5")
# main pill
draw.rounded_rectangle((header_x0, header_y0, header_x1, header_y1), radius=40, fill="#ffffff", outline="#d7d9db", width=1)
# thin divider under header
draw.line((48, header_y1+8, 1392, header_y1+8), fill="#e6e8ea", width=1)

# Light band behind filter chips area (keeps chips visually floating)
chips_band_y0 = header_y1 + 24
chips_band_y1 = chips_band_y0 + 220
draw.rectangle((0, chips_band_y0, 1440, chips_band_y1), fill="#f1f4f6")

# Large circular seating map background / arena container
circle_margin = 120
circle_top = chips_band_y0 + 40  # roughly where the map begins
circle_size = 1200
circle_bbox = (circle_margin, circle_top, circle_margin + circle_size, circle_top + circle_size)
# outer ring shadow
draw.ellipse((circle_bbox[0]-6, circle_bbox[1]-6, circle_bbox[2]+6, circle_bbox[3]+6), fill="#e9ebec")
# outer circle
draw.ellipse(circle_bbox, fill="#ffffff", outline="#c6c9cc", width=8)
# inner background slightly off-white
inner_inset = 24
draw.ellipse((circle_bbox[0]+inner_inset, circle_bbox[1]+inner_inset, circle_bbox[2]-inner_inset, circle_bbox[3]-inner_inset), fill="#fbfcfd")

# Center court area hint (subtle rectangular court background, but do not draw any markings)
court_w = 420
court_h = 220
cx = (circle_bbox[0] + circle_bbox[2]) / 2
cy = (circle_bbox[1] + circle_bbox[3]) / 2
court_bbox = (cx - court_w/2, cy - court_h/2, cx + court_w/2, cy + court_h/2)
draw.rounded_rectangle(court_bbox, radius=18, fill="#f4f7f9", outline="#d7d9db", width=2)

# Divider line between map area and listings card
map_bottom = circle_bbox[3] + 28
draw.line((48, map_bottom, 1392, map_bottom), fill="#e6e8ea", width=1)

# Listings header card (rounded, white) that contains "973 Listings" and sort control area
list_header_y0 = map_bottom + 48
list_header_y1 = list_header_y0 + 88
draw.rounded_rectangle((24, list_header_y0, 1416, list_header_y1), radius=28, fill="#ffffff", outline="#dcdfe1", width=1)
# subtle top shadow for the listings card
draw.line((24, list_header_y1+1, 1416, list_header_y1+1), fill="#eef0f1", width=1)

# Content area background (listings area) - large rounded white panel
content_y0 = list_header_y1 + 28
draw.rounded_rectangle((0, content_y0, 1440, 2960), radius=36, fill="#ffffff", outline=None)

# Faint separators between expected listing items (do not draw any text or thumbnails)
sep_x0 = 24
sep_x1 = 1416
# approximate positions for separators (spaced for multiple cards)
separators = [content_y0 + 220, content_y0 + 220 + 480, content_y0 + 220 + 480 + 480]
for y in separators:
    draw.line((sep_x0, y, sep_x1, y), fill="#f0f2f3", width=1)

# Placeholder rounded rectangles for listing card backgrounds (only backgrounds, no content)
card_margin_left = 48
card_margin_right = 1392
card_height = 200
first_card_y = content_y0 + 24
for i in range(3):
    y0 = first_card_y + i * (card_height + 24)
    y1 = y0 + card_height
    draw.rounded_rectangle((card_margin_left, y0, card_margin_right, y1), radius=18, fill="#ffffff", outline="#ebeef0", width=1)

# Bottom safe area faint divider
draw.line((0, 2880, 1440, 2880), fill="#e9ebec", width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_06_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-9/00_icon_Include_fees.png
try:
    _c0 = get_crop(0, 335, 108)
    canvas.paste(_c0, (544, 312), _c0)
except Exception:
    pass
layout["Include_fees"] = [544, 312, 879, 420]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_06_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-9/01_icon_Quantity.png
try:
    _c1 = get_crop(1, 268, 108)
    canvas.paste(_c1, (240, 312), _c1)
except Exception:
    pass
layout["Quantity"] = [240, 312, 508, 420]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_06_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-9/02_icon_Courtside.png
try:
    _c2 = get_crop(2, 286, 108)
    canvas.paste(_c2, (915, 312), _c2)
except Exception:
    pass
layout["Courtside"] = [915, 312, 1201, 420]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_06_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-9/03_icon_9.9.png
try:
    _c3 = get_crop(3, 1440, 455)
    canvas.paste(_c3, (0, 2355), _c3)
except Exception:
    pass
layout["9.9"] = [0, 2355, 1440, 2810]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_06_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-9/04_icon_Tit.png
try:
    _c4 = get_crop(4, 156, 108)
    canvas.paste(_c4, (48, 312), _c4)
except Exception:
    pass
layout["Tit"] = [48, 312, 204, 420]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_06_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-9/05_icon_Center.png
try:
    _c5 = get_crop(5, 203, 108)
    canvas.paste(_c5, (1237, 312), _c5)
except Exception:
    pass
layout["Center"] = [1237, 312, 1440, 420]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_06_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-9/06_icon_E_Conf_Ist_Rnd_76ers_at_Knicks_Gm_2_HG_2.png
try:
    _c6 = get_crop(6, 1344, 156)
    canvas.paste(_c6, (48, 120), _c6)
except Exception:
    pass
layout["E_Conf_Ist_Rnd:_76ers_at_"] = [48, 120, 1392, 276]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_06_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-9/07_icon_7.45_Wy.png
try:
    _c7 = get_crop(7, 67, 63)
    canvas.paste(_c7, (111, 0), _c7)
except Exception:
    pass
layout["7.45_Wy"] = [111, 0, 178, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_06_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-9/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 50, 64)
    canvas.paste(_c8, (1153, 2), _c8)
except Exception:
    pass
layout["icon_8"] = [1153, 2, 1203, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_06_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-9/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 57, 61)
    canvas.paste(_c9, (312, 3), _c9)
except Exception:
    pass
layout["icon_9"] = [312, 3, 369, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_06_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-9/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 62, 61)
    canvas.paste(_c10, (242, 2), _c10)
except Exception:
    pass
layout["icon_10"] = [242, 2, 304, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_06_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-9/11_icon_7.45_Wy.png
try:
    _c11 = get_crop(11, 53, 60)
    canvas.paste(_c11, (182, 2), _c11)
except Exception:
    pass
layout["7.45_Wy"] = [182, 2, 235, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_06_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-9/12_icon_6.png
try:
    _c12 = get_crop(12, 105, 62)
    canvas.paste(_c12, (1212, 1), _c12)
except Exception:
    pass
layout["@6"] = [1212, 1, 1317, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_06_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-9/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 54, 57)
    canvas.paste(_c13, (1320, 3), _c13)
except Exception:
    pass
layout["icon_13"] = [1320, 3, 1374, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_06_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-9/14_icon_Amazing_deal.png
try:
    _c14 = get_crop(14, 1440, 455)
    canvas.paste(_c14, (0, 2355), _c14)
except Exception:
    pass
layout["Amazing_deal"] = [0, 2355, 1440, 2810]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_06_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-9/15_icon_6.png
try:
    _c15 = get_crop(15, 156, 156)
    canvas.paste(_c15, (1236, 120), _c15)
except Exception:
    pass
layout["@6"] = [1236, 120, 1392, 276]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_06_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-9/16_icon_Sort_by_deal.png
try:
    _c16 = get_crop(16, 440, 144)
    canvas.paste(_c16, (976, 1989), _c16)
except Exception:
    pass
layout["Sort_by_deal"] = [976, 1989, 1416, 2133]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_06_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-9/17_icon_icon_17.png
try:
    _c17 = get_crop(17, 49, 64)
    canvas.paste(_c17, (383, 1), _c17)
except Exception:
    pass
layout["icon_17"] = [383, 1, 432, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_06_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-9/18_icon_MER.png
try:
    _c18 = get_crop(18, 384, 105)
    canvas.paste(_c18, (53, 2854), _c18)
except Exception:
    pass
layout["MER"] = [53, 2854, 437, 2959]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_06_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-9/19_icon_7.45_Wy.png
try:
    _c19 = get_crop(19, 102, 65)
    canvas.paste(_c19, (4, 0), _c19)
except Exception:
    pass
layout["7.45_Wy"] = [4, 0, 106, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_06_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-9/20_text_212.png
try:
    _c20 = get_crop(20, 48, 30)
    canvas.paste(_c20, (610, 712), _c20)
except Exception:
    pass
layout["212"] = [610, 712, 658, 742]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_06_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-9/21_text_210.png
try:
    _c21 = get_crop(21, 48, 30)
    canvas.paste(_c21, (779, 712), _c21)
except Exception:
    pass
layout["210"] = [779, 712, 827, 742]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_06_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-9/22_text_213.png
try:
    _c22 = get_crop(22, 48, 29)
    canvas.paste(_c22, (506, 731), _c22)
except Exception:
    pass
layout["213"] = [506, 731, 554, 760]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_06_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-9/23_text_209.png
try:
    _c23 = get_crop(23, 45, 29)
    canvas.paste(_c23, (886, 731), _c23)
except Exception:
    pass
layout["209"] = [886, 731, 931, 760]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_06_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-9/24_text_208.png
try:
    _c24 = get_crop(24, 46, 28)
    canvas.paste(_c24, (987, 781), _c24)
except Exception:
    pass
layout["208"] = [987, 781, 1033, 809]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_06_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-9/25_text_SS18.png
try:
    _c25 = get_crop(25, 62, 28)
    canvas.paste(_c25, (1091, 818), _c25)
except Exception:
    pass
layout["SS18"] = [1091, 818, 1153, 846]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_06_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-9/26_text_S4Z.png
try:
    _c26 = get_crop(26, 57, 30)
    canvas.paste(_c26, (673, 897), _c26)
except Exception:
    pass
layout["[S4Z"] = [673, 897, 730, 927]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_06_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-9/27_text_S45__543.png
try:
    _c27 = get_crop(27, 128, 36)
    canvas.paste(_c27, (739, 898), _c27)
except Exception:
    pass
layout["[S45_[543"] = [739, 898, 867, 934]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_06_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-9/28_text_S16.png
try:
    _c28 = get_crop(28, 60, 29)
    canvas.paste(_c28, (1149, 888), _c28)
except Exception:
    pass
layout["S16"] = [1149, 888, 1209, 917]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_06_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-9/29_text_LS52.png
try:
    _c29 = get_crop(29, 62, 29)
    canvas.paste(_c29, (483, 923), _c29)
except Exception:
    pass
layout["LS52"] = [483, 923, 545, 952]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_06_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-9/30_text_SS15.png
try:
    _c30 = get_crop(30, 59, 27)
    canvas.paste(_c30, (1175, 932), _c30)
except Exception:
    pass
layout["SS15"] = [1175, 932, 1234, 959]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_06_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-9/31_text_LS54.png
try:
    _c31 = get_crop(31, 60, 29)
    canvas.paste(_c31, (418, 948), _c31)
except Exception:
    pass
layout["LS54"] = [418, 948, 478, 977]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_06_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-9/32_text_Ls39.png
try:
    _c32 = get_crop(32, 60, 29)
    canvas.paste(_c32, (957, 946), _c32)
except Exception:
    pass
layout["Ls39"] = [957, 946, 1017, 975]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_06_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-9/33_text_LS57.png
try:
    _c33 = get_crop(33, 57, 28)
    canvas.paste(_c33, (340, 1003), _c33)
except Exception:
    pass
layout["LS57"] = [340, 1003, 397, 1031]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_06_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-9/34_text_LS36.png
try:
    _c34 = get_crop(34, 58, 28)
    canvas.paste(_c34, (1040, 1003), _c34)
except Exception:
    pass
layout["LS36"] = [1040, 1003, 1098, 1031]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_06_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-9/35_text_206.png
try:
    _c35 = get_crop(35, 48, 27)
    canvas.paste(_c35, (1119, 1006), _c35)
except Exception:
    pass
layout["206"] = [1119, 1006, 1167, 1033]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_06_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-9/36_text_SS13.png
try:
    _c36 = get_crop(36, 59, 29)
    canvas.paste(_c36, (1214, 1027), _c36)
except Exception:
    pass
layout["SS13"] = [1214, 1027, 1273, 1056]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_06_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-9/37_text_205.png
try:
    _c37 = get_crop(37, 46, 28)
    canvas.paste(_c37, (1149, 1077), _c37)
except Exception:
    pass
layout["205"] = [1149, 1077, 1195, 1105]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_06_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-9/38_text_SS11.png
try:
    _c38 = get_crop(38, 57, 27)
    canvas.paste(_c38, (1235, 1126), _c38)
except Exception:
    pass
layout["SS11"] = [1235, 1126, 1292, 1153]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_06_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-9/39_text_SS8.png
try:
    _c39 = get_crop(39, 48, 29)
    canvas.paste(_c39, (1242, 1265), _c39)
except Exception:
    pass
layout["SS8"] = [1242, 1265, 1290, 1294]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_06_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-9/40_text_LS1.png
try:
    _c40 = get_crop(40, 43, 27)
    canvas.paste(_c40, (326, 1364), _c40)
except Exception:
    pass
layout["LS1"] = [326, 1364, 369, 1391]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_06_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-9/41_text_SS6.png
try:
    _c41 = get_crop(41, 45, 27)
    canvas.paste(_c41, (1221, 1362), _c41)
except Exception:
    pass
layout["SS6"] = [1221, 1362, 1266, 1389]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_06_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-9/42_text_LS24.png
try:
    _c42 = get_crop(42, 60, 27)
    canvas.paste(_c42, (1038, 1387), _c42)
except Exception:
    pass
layout["LS24"] = [1038, 1387, 1098, 1414]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_06_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-9/43_text_LS5.png
try:
    _c43 = get_crop(43, 46, 29)
    canvas.paste(_c43, (425, 1441), _c43)
except Exception:
    pass
layout["LS5"] = [425, 1441, 471, 1470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_06_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-9/44_text_Ls7.png
try:
    _c44 = get_crop(44, 44, 29)
    canvas.paste(_c44, (485, 1464), _c44)
except Exception:
    pass
layout["Ls7"] = [485, 1464, 529, 1493]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_06_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-9/45_text_LS19.png
try:
    _c45 = get_crop(45, 60, 29)
    canvas.paste(_c45, (899, 1466), _c45)
except Exception:
    pass
layout["LS19"] = [899, 1466, 959, 1495]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_06_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-9/46_text_SS4.png
try:
    _c46 = get_crop(46, 48, 30)
    canvas.paste(_c46, (1179, 1454), _c46)
except Exception:
    pass
layout["SS4"] = [1179, 1454, 1227, 1484]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_06_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-9/47_text_LS12.png
try:
    _c47 = get_crop(47, 57, 29)
    canvas.paste(_c47, (654, 1494), _c47)
except Exception:
    pass
layout["LS12"] = [654, 1494, 711, 1523]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_06_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-9/48_text_LS15_LS17.png
try:
    _c48 = get_crop(48, 134, 41)
    canvas.paste(_c48, (760, 1479), _c48)
except Exception:
    pass
layout["LS15_LS17"] = [760, 1479, 894, 1520]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_06_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-9/49_text_SS3.png
try:
    _c49 = get_crop(49, 48, 27)
    canvas.paste(_c49, (1154, 1501), _c49)
except Exception:
    pass
layout["SS3"] = [1154, 1501, 1202, 1528]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_06_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-9/50_text_222.png
try:
    _c50 = get_crop(50, 50, 36)
    canvas.paste(_c50, (491, 1530), _c50)
except Exception:
    pass
layout["222"] = [491, 1530, 541, 1566]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_06_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-9/51_text_MEDIA.png
try:
    _c51 = get_crop(51, 62, 25)
    canvas.paste(_c51, (689, 1535), _c51)
except Exception:
    pass
layout["MEDIA"] = [689, 1535, 751, 1560]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_06_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-9/52_text_226.png
try:
    _c52 = get_crop(52, 48, 30)
    canvas.paste(_c52, (895, 1528), _c52)
except Exception:
    pass
layout["226"] = [895, 1528, 943, 1558]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_06_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-9/53_text_225.png
try:
    _c53 = get_crop(53, 46, 27)
    canvas.paste(_c53, (809, 1547), _c53)
except Exception:
    pass
layout["225"] = [809, 1547, 855, 1574]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_06_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-9/54_text_SS1.png
try:
    _c54 = get_crop(54, 46, 30)
    canvas.paste(_c54, (1098, 1572), _c54)
except Exception:
    pass
layout["SS1"] = [1098, 1572, 1144, 1602]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_06_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-9/55_text_224UWC.png
try:
    _c55 = get_crop(55, 101, 28)
    canvas.paste(_c55, (684, 1699), _c55)
except Exception:
    pass
layout["224UWC"] = [684, 1699, 785, 1727]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_06_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-9/56_text_324.png
try:
    _c56 = get_crop(56, 58, 27)
    canvas.paste(_c56, (411, 1739), _c56)
except Exception:
    pass
layout["[324"] = [411, 1739, 469, 1766]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_06_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-9/57_text_325.png
try:
    _c57 = get_crop(57, 58, 27)
    canvas.paste(_c57, (499, 1741), _c57)
except Exception:
    pass
layout["[325"] = [499, 1741, 557, 1768]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_06_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-9/58_text_326.png
try:
    _c58 = get_crop(58, 59, 27)
    canvas.paste(_c58, (622, 1741), _c58)
except Exception:
    pass
layout["[326"] = [622, 1741, 681, 1768]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_06_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-9/59_text_327.png
try:
    _c59 = get_crop(59, 46, 27)
    canvas.paste(_c59, (825, 1741), _c59)
except Exception:
    pass
layout["327"] = [825, 1741, 871, 1768]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_06_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-9/60_text_328.png
try:
    _c60 = get_crop(60, 48, 27)
    canvas.paste(_c60, (936, 1741), _c60)
except Exception:
    pass
layout["328"] = [936, 1741, 984, 1768]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_06_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-9/61_text_973_Listings.png
try:
    _c61 = get_crop(61, 332, 77)
    canvas.paste(_c61, (54, 2029), _c61)
except Exception:
    pass
layout["973_Listings"] = [54, 2029, 386, 2106]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_06_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-9/62_text_We_sell_resale_tickets._Resale_tickets_m.png
try:
    _c62 = get_crop(62, 1440, 455)
    canvas.paste(_c62, (0, 2355), _c62)
except Exception:
    pass
layout["€_We_sell_resale_tickets."] = [0, 2355, 1440, 2810]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_06_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-9/63_text_face_value.png
try:
    _c63 = get_crop(63, 218, 43)
    canvas.paste(_c63, (57, 2256), _c63)
except Exception:
    pass
layout["face_value:"] = [57, 2256, 275, 2299]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_06_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-9/64_text_S559_each.png
try:
    _c64 = get_crop(64, 274, 61)
    canvas.paste(_c64, (487, 2862), _c64)
except Exception:
    pass
layout["S559_each"] = [487, 2862, 761, 2923]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_06_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-9/65_text_J1O8WC.png
try:
    _c65 = get_crop(65, 87, 41)
    canvas.paste(_c65, (552, 921), _c65)
except Exception:
    pass
layout["J1O8WC"] = [552, 921, 639, 962]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_06_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-9/66_clickable_Back.png
try:
    _c66 = get_crop(66, 156, 156)
    canvas.paste(_c66, (48, 120), _c66)
except Exception:
    pass
layout["Back"] = [48, 120, 204, 276]
