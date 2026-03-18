# page_id: page_seatgeek_21b637d11bea46b8adb3c2efc9f03501_08
# screenshot: 2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-11.png
# step_index: 8/10
# task: Open SeatGeek and find the soonest upcoming NBA game in New York with "Nets", record the cheapest price in google keep notes.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Top-level background
draw.rectangle([(0, 0), (1440, 2960)], fill="#eef1f4")  # overall light bluish-gray background

# Status bar area (top ~56px)
status_h = 56
draw.rectangle([(0, 0), (1440, status_h)], fill="#d0d2d4")  # darker status bar background
# thin divider under status bar
draw.line([(0, status_h), (1440, status_h)], fill="#c6c8ca", width=1)

# App header / toolbar background (rounded pill behind the title area)
header_top = 64
header_bottom = 160
header_inset = 40
draw.rounded_rectangle(
    [(header_inset, header_top), (1440 - header_inset, header_bottom)],
    radius=40,
    fill="#f7f8f9",
    outline="#d9dbdd",
    width=2
)

# subtle divider below header area
draw.line([(20, header_bottom + 12), (1420, header_bottom + 12)], fill="#e0e2e3", width=1)

# Large circular seating-map background (centered circular area)
map_center_x = 720
map_center_y = 900
map_radius = 560
map_bbox = [
    (map_center_x - map_radius, map_center_y - map_radius),
    (map_center_x + map_radius, map_center_y + map_radius),
]
draw.ellipse(map_bbox, fill="#e6eaec", outline="#bfc3c6", width=6)

# Slight inner ring to mimic stadium ring structure (large ring)
inner_ring_radius = map_radius - 24
draw.ellipse(
    [
        (map_center_x - inner_ring_radius, map_center_y - inner_ring_radius),
        (map_center_x + inner_ring_radius, map_center_y + inner_ring_radius),
    ],
    outline="#d1d5d7",
    width=8
)

# Court background area (subtle centered rectangle behind the court)
court_w = 300
court_h = 160
court_left = map_center_x - court_w // 2
court_top = map_center_y - court_h // 2
draw.rounded_rectangle(
    [(court_left, court_top), (court_left + court_w, court_top + court_h)],
    radius=8,
    fill="#f0f3f5",
    outline="#c9ced0",
    width=2
)

# Dim overlay gradient band behind top-of-map filters (subtle)
# (represented by two translucent rectangles stacked as approximation)
band_top = header_bottom + 24
band_bottom = map_center_y - map_radius + 40
draw.rectangle([(0, band_top), (1440, band_bottom)], fill="#eef1f4")

# Separator line above the bottom modal sheet
modal_top = 1640
draw.line([(40, modal_top), (1400, modal_top)], fill="#e6e8e9", width=1)

# Modal sheet (bottom rounded sheet)
modal_left = 0
modal_right = 1440
modal_radius = 36
draw.rounded_rectangle(
    [(modal_left, modal_top), (modal_right, 2960)],
    radius=modal_radius,
    fill="#ffffff",
    outline="#e6e7e8",
    width=1
)

# Modal top handle (small rounded bar)
handle_w = 120
handle_h = 8
handle_x0 = (1440 - handle_w) // 2
handle_y0 = modal_top + 12
draw.rounded_rectangle(
    [(handle_x0, handle_y0), (handle_x0 + handle_w, handle_y0 + handle_h)],
    radius=8,
    fill="#e3e5e6",
    outline=None
)

# Title divider (thin line between title area and content inside modal)
title_div_y = modal_top + 96
draw.line([(60, title_div_y), (1380, title_div_y)], fill="#f0f1f2", width=1)

# Card containers inside modal - three option cards (rounded rectangles)
card_x0 = 60
card_x1 = 60 + 1320  # width 1320 as in detected layout
card_h = 200
gap = 30

card1_y0 = modal_top + 120
card1_y1 = card1_y0 + card_h

card2_y0 = card1_y1 + gap
card2_y1 = card2_y0 + card_h

card3_y0 = card2_y1 + gap
card3_y1 = card3_y0 + card_h

card_radius = 20

# Card 1 - neutral background with subtle border
draw.rounded_rectangle(
    [(card_x0, card1_y0), (card_x1, card1_y1)],
    radius=card_radius,
    fill="#ffffff",
    outline="#e5e6e7",
    width=2
)

# Card 2 - selected style (thicker darker outline to indicate active selection)
draw.rounded_rectangle(
    [(card_x0, card2_y0), (card_x1, card2_y1)],
    radius=card_radius,
    fill="#ffffff",
    outline="#0b0b0b",
    width=4
)
# inner subtle shadow line for selected card
draw.line([(card_x0 + 12, card2_y0 + 6), (card_x1 - 12, card2_y0 + 6)], fill="#000000", width=1)

# Card 3 - neutral
draw.rounded_rectangle(
    [(card_x0, card3_y0), (card_x1, card3_y1)],
    radius=card_radius,
    fill="#ffffff",
    outline="#e5e6e7",
    width=2
)

# Light divider lines between cards (subtle)
draw.line([(card_x0 + 12, card1_y1 + gap // 2), (card_x1 - 12, card1_y1 + gap // 2)], fill="#f0f1f2", width=1)
draw.line([(card_x0 + 12, card2_y1 + gap // 2), (card_x1 - 12, card2_y1 + gap // 2)], fill="#f0f1f2", width=1)

# Bottom area filler (space below cards)
bottom_space_top = card3_y1 + 24
draw.rectangle([(0, bottom_space_top), (1440, 2960)], fill="#ffffff")

# Thin separators for content above modal (subtle)
draw.line([(40, map_center_y + map_radius + 20), (1400, map_center_y + map_radius + 20)], fill="#e6e8e9", width=1)

# Side paddings shadow lines to give depth to modal edges
draw.line([(48, modal_top + 4), (48, 2956)], fill="#f3f4f5", width=1)
draw.line([(1392, modal_top + 4), (1392, 2956)], fill="#f3f4f5", width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_08_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-11/00_icon_We_rate_each_ticket_based_on_location_pr.png
try:
    _c0 = get_crop(0, 1320, 267)
    canvas.paste(_c0, (60, 2318), _c0)
except Exception:
    pass
layout["We_rate_each_ticket_based"] = [60, 2318, 1380, 2585]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_08_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-11/01_icon_Include.png
try:
    _c1 = get_crop(1, 341, 117)
    canvas.paste(_c1, (537, 309), _c1)
except Exception:
    pass
layout["Include"] = [537, 309, 878, 426]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_08_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-11/02_icon_Courtside.png
try:
    _c2 = get_crop(2, 295, 119)
    canvas.paste(_c2, (909, 308), _c2)
except Exception:
    pass
layout["Courtside"] = [909, 308, 1204, 427]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_08_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-11/03_icon_Quantity.png
try:
    _c3 = get_crop(3, 280, 120)
    canvas.paste(_c3, (231, 307), _c3)
except Exception:
    pass
layout["Quantity"] = [231, 307, 511, 427]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_08_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-11/04_icon_Tit.png
try:
    _c4 = get_crop(4, 169, 122)
    canvas.paste(_c4, (37, 308), _c4)
except Exception:
    pass
layout["Tit"] = [37, 308, 206, 430]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_08_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-11/05_icon_We_rate_each_ticket_based_on_location_pr.png
try:
    _c5 = get_crop(5, 1320, 329)
    canvas.paste(_c5, (60, 1941), _c5)
except Exception:
    pass
layout["We_rate_each_ticket_based"] = [60, 1941, 1380, 2270]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_08_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-11/06_icon_Center.png
try:
    _c6 = get_crop(6, 211, 119)
    canvas.paste(_c6, (1229, 308), _c6)
except Exception:
    pass
layout["Center"] = [1229, 308, 1440, 427]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_08_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-11/07_icon_Include.png
try:
    _c7 = get_crop(7, 1342, 165)
    canvas.paste(_c7, (39, 118), _c7)
except Exception:
    pass
layout["Include"] = [39, 118, 1381, 283]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_08_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-11/08_icon_Center.png
try:
    _c8 = get_crop(8, 101, 109)
    canvas.paste(_c8, (1256, 146), _c8)
except Exception:
    pass
layout["Center"] = [1256, 146, 1357, 255]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_08_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-11/09_icon_GK.png
try:
    _c9 = get_crop(9, 57, 55)
    canvas.paste(_c9, (180, 6), _c9)
except Exception:
    pass
layout["GK"] = [180, 6, 237, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_08_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-11/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 55, 66)
    canvas.paste(_c10, (1151, 1), _c10)
except Exception:
    pass
layout["icon_10"] = [1151, 1, 1206, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_08_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-11/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 52, 57)
    canvas.paste(_c11, (1320, 3), _c11)
except Exception:
    pass
layout["icon_11"] = [1320, 3, 1372, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_08_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-11/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 106, 62)
    canvas.paste(_c12, (1211, 1), _c12)
except Exception:
    pass
layout["icon_12"] = [1211, 1, 1317, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_08_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-11/13_icon_GK.png
try:
    _c13 = get_crop(13, 54, 58)
    canvas.paste(_c13, (117, 3), _c13)
except Exception:
    pass
layout["GK"] = [117, 3, 171, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_08_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-11/14_text_6.38.png
try:
    _c14 = get_crop(14, 89, 41)
    canvas.paste(_c14, (22, 17), _c14)
except Exception:
    pass
layout["6.38"] = [22, 17, 111, 58]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_08_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-11/15_text_B1o.png
try:
    _c15 = get_crop(15, 53, 27)
    canvas.paste(_c15, (971, 650), _c15)
except Exception:
    pass
layout["B1o]"] = [971, 650, 1024, 677]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_08_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-11/16_text_212.png
try:
    _c16 = get_crop(16, 48, 30)
    canvas.paste(_c16, (610, 712), _c16)
except Exception:
    pass
layout["212"] = [610, 712, 658, 742]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_08_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-11/17_text_210.png
try:
    _c17 = get_crop(17, 48, 27)
    canvas.paste(_c17, (779, 712), _c17)
except Exception:
    pass
layout["210"] = [779, 712, 827, 739]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_08_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-11/18_text_213.png
try:
    _c18 = get_crop(18, 48, 29)
    canvas.paste(_c18, (506, 731), _c18)
except Exception:
    pass
layout["213"] = [506, 731, 554, 760]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_08_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-11/19_text_209.png
try:
    _c19 = get_crop(19, 45, 29)
    canvas.paste(_c19, (886, 731), _c19)
except Exception:
    pass
layout["209"] = [886, 731, 931, 760]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_08_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-11/20_text_208.png
try:
    _c20 = get_crop(20, 46, 28)
    canvas.paste(_c20, (987, 781), _c20)
except Exception:
    pass
layout["208"] = [987, 781, 1033, 809]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_08_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-11/21_text_SS18.png
try:
    _c21 = get_crop(21, 62, 28)
    canvas.paste(_c21, (1091, 818), _c21)
except Exception:
    pass
layout["SS18"] = [1091, 818, 1153, 846]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_08_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-11/22_text_S47.png
try:
    _c22 = get_crop(22, 57, 27)
    canvas.paste(_c22, (673, 897), _c22)
except Exception:
    pass
layout["[S47"] = [673, 897, 730, 924]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_08_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-11/23_text_S45__543.png
try:
    _c23 = get_crop(23, 128, 36)
    canvas.paste(_c23, (739, 898), _c23)
except Exception:
    pass
layout["[S45_[543"] = [739, 898, 867, 934]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_08_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-11/24_text_SS16.png
try:
    _c24 = get_crop(24, 60, 29)
    canvas.paste(_c24, (1149, 888), _c24)
except Exception:
    pass
layout["SS16"] = [1149, 888, 1209, 917]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_08_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-11/25_text_LS52.png
try:
    _c25 = get_crop(25, 62, 29)
    canvas.paste(_c25, (483, 923), _c25)
except Exception:
    pass
layout["LS52"] = [483, 923, 545, 952]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_08_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-11/26_text_s41.png
try:
    _c26 = get_crop(26, 63, 37)
    canvas.paste(_c26, (871, 913), _c26)
except Exception:
    pass
layout["~[s41"] = [871, 913, 934, 950]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_08_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-11/27_text_SS15.png
try:
    _c27 = get_crop(27, 59, 27)
    canvas.paste(_c27, (1175, 932), _c27)
except Exception:
    pass
layout["SS15"] = [1175, 932, 1234, 959]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_08_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-11/28_text_LS54.png
try:
    _c28 = get_crop(28, 60, 29)
    canvas.paste(_c28, (418, 948), _c28)
except Exception:
    pass
layout["LS54"] = [418, 948, 478, 977]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_08_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-11/29_text_Ls39.png
try:
    _c29 = get_crop(29, 60, 29)
    canvas.paste(_c29, (957, 946), _c29)
except Exception:
    pass
layout["Ls39"] = [957, 946, 1017, 975]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_08_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-11/30_text_LS57.png
try:
    _c30 = get_crop(30, 57, 28)
    canvas.paste(_c30, (340, 1003), _c30)
except Exception:
    pass
layout["LS57"] = [340, 1003, 397, 1031]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_08_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-11/31_text_LS36.png
try:
    _c31 = get_crop(31, 58, 28)
    canvas.paste(_c31, (1040, 1003), _c31)
except Exception:
    pass
layout["LS36"] = [1040, 1003, 1098, 1031]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_08_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-11/32_text_206.png
try:
    _c32 = get_crop(32, 48, 27)
    canvas.paste(_c32, (1119, 1006), _c32)
except Exception:
    pass
layout["206"] = [1119, 1006, 1167, 1033]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_08_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-11/33_text_SS13.png
try:
    _c33 = get_crop(33, 59, 27)
    canvas.paste(_c33, (1214, 1029), _c33)
except Exception:
    pass
layout["SS13"] = [1214, 1029, 1273, 1056]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_08_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-11/34_text_205.png
try:
    _c34 = get_crop(34, 46, 28)
    canvas.paste(_c34, (1149, 1077), _c34)
except Exception:
    pass
layout["205"] = [1149, 1077, 1195, 1105]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_08_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-11/35_text_SS11.png
try:
    _c35 = get_crop(35, 57, 27)
    canvas.paste(_c35, (1235, 1126), _c35)
except Exception:
    pass
layout["SS11"] = [1235, 1126, 1292, 1153]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_08_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-11/36_text_SS10.png
try:
    _c36 = get_crop(36, 60, 30)
    canvas.paste(_c36, (1239, 1172), _c36)
except Exception:
    pass
layout["SS10"] = [1239, 1172, 1299, 1202]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_08_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-11/37_text_218.png
try:
    _c37 = get_crop(37, 45, 28)
    canvas.paste(_c37, (220, 1225), _c37)
except Exception:
    pass
layout["218"] = [220, 1225, 265, 1253]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_08_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-11/38_text_SS9.png
try:
    _c38 = get_crop(38, 48, 27)
    canvas.paste(_c38, (1246, 1219), _c38)
except Exception:
    pass
layout["SS9"] = [1246, 1219, 1294, 1246]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_08_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-11/39_text_SS8.png
try:
    _c39 = get_crop(39, 48, 29)
    canvas.paste(_c39, (1242, 1265), _c39)
except Exception:
    pass
layout["SS8"] = [1242, 1265, 1290, 1294]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_08_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-11/40_text_203.png
try:
    _c40 = get_crop(40, 46, 27)
    canvas.paste(_c40, (1149, 1313), _c40)
except Exception:
    pass
layout["203"] = [1149, 1313, 1195, 1340]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_08_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-11/41_text_SS6.png
try:
    _c41 = get_crop(41, 45, 27)
    canvas.paste(_c41, (1221, 1362), _c41)
except Exception:
    pass
layout["SS6"] = [1221, 1362, 1266, 1389]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_08_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-11/42_text_LS24.png
try:
    _c42 = get_crop(42, 58, 27)
    canvas.paste(_c42, (1040, 1387), _c42)
except Exception:
    pass
layout["LS24"] = [1040, 1387, 1098, 1414]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_08_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-11/43_text_LSS.png
try:
    _c43 = get_crop(43, 50, 33)
    canvas.paste(_c43, (423, 1440), _c43)
except Exception:
    pass
layout["LSS"] = [423, 1440, 473, 1473]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_08_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-11/44_text_SS4.png
try:
    _c44 = get_crop(44, 48, 27)
    canvas.paste(_c44, (1179, 1457), _c44)
except Exception:
    pass
layout["SS4"] = [1179, 1457, 1227, 1484]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_08_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-11/45_text_Ls7.png
try:
    _c45 = get_crop(45, 44, 29)
    canvas.paste(_c45, (485, 1464), _c45)
except Exception:
    pass
layout["Ls7"] = [485, 1464, 529, 1493]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_08_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-11/46_text_LS19.png
try:
    _c46 = get_crop(46, 60, 29)
    canvas.paste(_c46, (899, 1466), _c46)
except Exception:
    pass
layout["LS19"] = [899, 1466, 959, 1495]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_08_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-11/47_text_LS12.png
try:
    _c47 = get_crop(47, 59, 29)
    canvas.paste(_c47, (652, 1494), _c47)
except Exception:
    pass
layout["LS12"] = [652, 1494, 711, 1523]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_08_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-11/48_text_1S15_LS1Z.png
try:
    _c48 = get_crop(48, 134, 41)
    canvas.paste(_c48, (760, 1479), _c48)
except Exception:
    pass
layout["1S15_LS1Z"] = [760, 1479, 894, 1520]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_08_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-11/49_text_SS3.png
try:
    _c49 = get_crop(49, 48, 27)
    canvas.paste(_c49, (1154, 1501), _c49)
except Exception:
    pass
layout["SS3"] = [1154, 1501, 1202, 1528]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_08_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-11/50_text_222.png
try:
    _c50 = get_crop(50, 50, 36)
    canvas.paste(_c50, (491, 1530), _c50)
except Exception:
    pass
layout["222"] = [491, 1530, 541, 1566]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_08_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-11/51_text_MEDIA.png
try:
    _c51 = get_crop(51, 60, 20)
    canvas.paste(_c51, (690, 1539), _c51)
except Exception:
    pass
layout["MEDIA"] = [690, 1539, 750, 1559]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_08_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-11/52_text_226.png
try:
    _c52 = get_crop(52, 48, 30)
    canvas.paste(_c52, (895, 1528), _c52)
except Exception:
    pass
layout["226"] = [895, 1528, 943, 1558]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_08_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-11/53_text_225.png
try:
    _c53 = get_crop(53, 46, 27)
    canvas.paste(_c53, (809, 1547), _c53)
except Exception:
    pass
layout["225"] = [809, 1547, 855, 1574]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_08_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-11/54_text_SS1.png
try:
    _c54 = get_crop(54, 46, 30)
    canvas.paste(_c54, (1098, 1572), _c54)
except Exception:
    pass
layout["SS1"] = [1098, 1572, 1144, 1602]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_08_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-11/55_text_Sort_by.png
try:
    _c55 = get_crop(55, 188, 68)
    canvas.paste(_c55, (626, 1740), _c55)
except Exception:
    pass
layout["Sort_by"] = [626, 1740, 814, 1808]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_08_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-11/56_text_Best_Seats.png
try:
    _c56 = get_crop(56, 269, 55)
    canvas.paste(_c56, (118, 2703), _c56)
except Exception:
    pass
layout["Best_Seats"] = [118, 2703, 387, 2758]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_08_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-11/57_text_Get_close_to_the_action_with_the_best_se.png
try:
    _c57 = get_crop(57, 1320, 267)
    canvas.paste(_c57, (60, 2633), _c57)
except Exception:
    pass
layout["Get_close_to_the_action_w"] = [60, 2633, 1380, 2900]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_08_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-11/58_text_JO8WC.png
try:
    _c58 = get_crop(58, 87, 41)
    canvas.paste(_c58, (552, 921), _c58)
except Exception:
    pass
layout["JO8WC"] = [552, 921, 639, 962]
