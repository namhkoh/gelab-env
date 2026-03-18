# page_id: page_seatgeek_094b5cdb02e246858451240263e6ef7f_07
# screenshot: 2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-10.png
# step_index: 7/9
# task: Open SeatGeek. Find the soonest upcoming NBA game in Boston with "Celtics". What is the highest price for each ticket?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw background and structural UI elements for the mobile page
# Uses provided variables: canvas (PIL.Image), draw (PIL.ImageDraw), font_* available but not used.

W, H = canvas.size

# Colors
bg_color = (236, 239, 241)        # overall page background (soft light gray)
status_bar_color = (208, 213, 217) # top status bar slightly darker
header_bg = (250, 251, 252)       # header pill background
header_border = (200, 205, 210)
chips_strip = (245, 247, 248)     # background strip for chips row
map_bg = (226, 230, 233)          # seating/map area background
map_border = (185, 194, 200)
sheet_shadow = (190, 192, 194)    # subtle shadow for bottom sheet
sheet_bg = (255, 255, 255)        # bottom sheet white
grabber = (224, 226, 228)

# Clear full canvas with main background
draw.rectangle([(0, 0), (W, H)], fill=bg_color)

# Status bar area (top strip)
status_h = 64
draw.rectangle([(0, 0), (W, status_h)], fill=status_bar_color)

# Header pill (rounded) under status bar
header_margin_x = 40
header_top = 80
header_bottom = 180
header_radius = 48
draw.rounded_rectangle(
    [(header_margin_x, header_top), (W - header_margin_x, header_bottom)],
    radius=header_radius,
    fill=header_bg,
    outline=header_border,
    width=2
)

# Divider line under header (subtle)
divider_y = header_bottom + 8
draw.line([(header_margin_x + 8, divider_y), (W - header_margin_x - 8, divider_y)], fill=(220,223,226), width=1)

# Chips row background strip (subtle rounded long capsule behind filter chips)
chips_top = 220
chips_bottom = 380
chips_radius = 40
draw.rounded_rectangle(
    [(30, chips_top), (W - 30, chips_bottom)],
    radius=chips_radius,
    fill=chips_strip,
    outline=None
)

# Seating map / arena background area (rounded rectangle)
map_left = 60
map_top = chips_bottom + 40
map_right = W - 60
map_bottom = int(map_top + (H * 0.42))  # occupy central region
map_radius = 40
draw.rounded_rectangle(
    [(map_left, map_top), (map_right, map_bottom)],
    radius=map_radius,
    fill=map_bg,
    outline=map_border,
    width=4
)

# Large oval "arena" subtle shape inside map area to create structure
oval_margin = 100
draw.ellipse(
    [(map_left + oval_margin, map_top + oval_margin), (map_right - oval_margin, map_bottom - oval_margin)],
    fill=(235, 238, 240),
    outline=(200, 206, 210),
    width=6
)

# Subtle inner rings to suggest seating tiers (concentric rounded rectangles/ellipses)
ring_colors = [(215, 220, 223), (230, 233, 235)]
for i, color in enumerate(ring_colors):
    inset = 40 + i * 30
    draw.ellipse(
        [(map_left + inset, map_top + inset), (map_right - inset, map_bottom - inset)],
        outline=color,
        width=3
    )

# Shadow strip above bottom sheet to separate map and sheet
sheet_top = int(map_bottom - 40 + 300) if (map_bottom + 300) < (H - 300) else (int(H * 0.56))
# Better fixed placement: place sheet starting at around 1680 for 2960 canvas resemblance
sheet_top = 1680
shadow_top = sheet_top - 18
draw.rectangle([(0, shadow_top), (W, shadow_top + 18)], fill=sheet_shadow)

# Bottom sheet (rounded white card)
sheet_margin_x = 40
sheet_bottom = H - 20
sheet_radius = 36
draw.rounded_rectangle(
    [(sheet_margin_x, sheet_top), (W - sheet_margin_x, sheet_bottom)],
    radius=sheet_radius,
    fill=sheet_bg,
    outline=None
)

# Small center grabber at top of sheet
grabber_w = 160
grabber_h = 12
grabber_x0 = (W - grabber_w) // 2
grabber_x1 = grabber_x0 + grabber_w
grabber_y0 = sheet_top + 18
grabber_y1 = grabber_y0 + grabber_h
draw.rounded_rectangle([(grabber_x0, grabber_y0), (grabber_x1, grabber_y1)], radius=10, fill=grabber)

# Top notch divider under the sheet header area (subtle)
header_div_y = grabber_y1 + 28
draw.line([(sheet_margin_x + 20, header_div_y), (W - sheet_margin_x - 20, header_div_y)], fill=(240,242,243), width=1)

# Subtle horizontal separators hinting at list separation (light and thin; avoid drawing heavy card content)
sep_start_x = sheet_margin_x + 12
sep_end_x = W - sheet_margin_x - 12
# Place separators at spaced intervals but keep them faint
seps = [header_div_y + 84, header_div_y + 204, header_div_y + 324]
for y in seps:
    draw.line([(sep_start_x, y), (sep_end_x, y)], fill=(245,246,247), width=1)

# Final little accent: faint border on entire canvas edges to frame UI
frame_border = (230, 233, 235)
draw.rectangle([(0, 0), (W-1, H-1)], outline=frame_border, width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_07_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-10/00_icon_We_rate_each_ticket_based_on_location_pr.png
try:
    _c0 = get_crop(0, 1320, 267)
    canvas.paste(_c0, (60, 2318), _c0)
except Exception:
    pass
layout["We_rate_each_ticket_based"] = [60, 2318, 1380, 2585]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_07_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-10/01_icon_Include.png
try:
    _c1 = get_crop(1, 341, 117)
    canvas.paste(_c1, (537, 309), _c1)
except Exception:
    pass
layout["Include"] = [537, 309, 878, 426]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_07_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-10/02_icon_Courtside.png
try:
    _c2 = get_crop(2, 295, 119)
    canvas.paste(_c2, (908, 308), _c2)
except Exception:
    pass
layout["Courtside"] = [908, 308, 1203, 427]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_07_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-10/03_icon_Quantity.png
try:
    _c3 = get_crop(3, 280, 120)
    canvas.paste(_c3, (231, 307), _c3)
except Exception:
    pass
layout["Quantity"] = [231, 307, 511, 427]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_07_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-10/04_icon_Tit.png
try:
    _c4 = get_crop(4, 169, 120)
    canvas.paste(_c4, (37, 308), _c4)
except Exception:
    pass
layout["Tit"] = [37, 308, 206, 428]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_07_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-10/05_icon_Center.png
try:
    _c5 = get_crop(5, 211, 119)
    canvas.paste(_c5, (1229, 308), _c5)
except Exception:
    pass
layout["Center"] = [1229, 308, 1440, 427]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_07_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-10/06_icon_We_rate_each_ticket_based_on_location_pr.png
try:
    _c6 = get_crop(6, 1320, 329)
    canvas.paste(_c6, (60, 1941), _c6)
except Exception:
    pass
layout["We_rate_each_ticket_based"] = [60, 1941, 1380, 2270]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_07_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-10/07_icon_5.00_my.png
try:
    _c7 = get_crop(7, 64, 63)
    canvas.paste(_c7, (112, 1), _c7)
except Exception:
    pass
layout["5.00_my"] = [112, 1, 176, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_07_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-10/08_icon_5.00_my.png
try:
    _c8 = get_crop(8, 54, 61)
    canvas.paste(_c8, (181, 2), _c8)
except Exception:
    pass
layout["5.00_my"] = [181, 2, 235, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_07_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-10/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 66, 62)
    canvas.paste(_c9, (240, 2), _c9)
except Exception:
    pass
layout["icon_9"] = [240, 2, 306, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_07_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-10/10_icon_6.png
try:
    _c10 = get_crop(10, 103, 112)
    canvas.paste(_c10, (1255, 145), _c10)
except Exception:
    pass
layout["6_"] = [1255, 145, 1358, 257]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_07_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-10/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 60, 65)
    canvas.paste(_c11, (313, 1), _c11)
except Exception:
    pass
layout["icon_11"] = [313, 1, 373, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_07_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-10/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 52, 54)
    canvas.paste(_c12, (1320, 6), _c12)
except Exception:
    pass
layout["icon_12"] = [1320, 6, 1372, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_07_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-10/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 50, 64)
    canvas.paste(_c13, (1153, 2), _c13)
except Exception:
    pass
layout["icon_13"] = [1153, 2, 1203, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_07_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-10/14_icon_E_Conf_Ist_Rnd_TBD_at_Celtics_Gm_2_HG_2.png
try:
    _c14 = get_crop(14, 1359, 162)
    canvas.paste(_c14, (39, 120), _c14)
except Exception:
    pass
layout["E_Conf_Ist_Rnd:_TBD_at_Ce"] = [39, 120, 1398, 282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_07_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-10/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 49, 67)
    canvas.paste(_c15, (382, 0), _c15)
except Exception:
    pass
layout["icon_15"] = [382, 0, 431, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_07_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-10/16_icon_6.png
try:
    _c16 = get_crop(16, 104, 60)
    canvas.paste(_c16, (1213, 3), _c16)
except Exception:
    pass
layout["6_"] = [1213, 3, 1317, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_07_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-10/17_text_STANDING.png
try:
    _c17 = get_crop(17, 97, 25)
    canvas.paste(_c17, (670, 689), _c17)
except Exception:
    pass
layout["~STANDING"] = [670, 689, 767, 714]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_07_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-10/18_text_STUI.png
try:
    _c18 = get_crop(18, 62, 29)
    canvas.paste(_c18, (488, 717), _c18)
except Exception:
    pass
layout["STUI"] = [488, 717, 550, 746]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_07_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-10/19_text_STU2.png
try:
    _c19 = get_crop(19, 65, 29)
    canvas.paste(_c19, (566, 717), _c19)
except Exception:
    pass
layout["STU2"] = [566, 717, 631, 746]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_07_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-10/20_text_STUS.png
try:
    _c20 = get_crop(20, 64, 29)
    canvas.paste(_c20, (807, 717), _c20)
except Exception:
    pass
layout["STUS"] = [807, 717, 871, 746]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_07_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-10/21_text_STU6.png
try:
    _c21 = get_crop(21, 64, 29)
    canvas.paste(_c21, (888, 717), _c21)
except Exception:
    pass
layout["STU6"] = [888, 717, 952, 746]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_07_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-10/22_text_S608.png
try:
    _c22 = get_crop(22, 60, 27)
    canvas.paste(_c22, (543, 876), _c22)
except Exception:
    pass
layout["S608"] = [543, 876, 603, 903]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_07_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-10/23_text_S611.png
try:
    _c23 = get_crop(23, 57, 27)
    canvas.paste(_c23, (668, 876), _c23)
except Exception:
    pass
layout["S611"] = [668, 876, 725, 903]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_07_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-10/24_text_S613.png
try:
    _c24 = get_crop(24, 60, 27)
    canvas.paste(_c24, (751, 876), _c24)
except Exception:
    pass
layout["S613"] = [751, 876, 811, 903]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_07_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-10/25_text_S615.png
try:
    _c25 = get_crop(25, 59, 29)
    canvas.paste(_c25, (837, 874), _c25)
except Exception:
    pass
layout["S615"] = [837, 874, 896, 903]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_07_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-10/26_text_20.png
try:
    _c26 = get_crop(26, 34, 27)
    canvas.paste(_c26, (472, 953), _c26)
except Exception:
    pass
layout["20"] = [472, 953, 506, 980]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_07_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-10/27_text_19.png
try:
    _c27 = get_crop(27, 32, 27)
    canvas.paste(_c27, (448, 1022), _c27)
except Exception:
    pass
layout["19"] = [448, 1022, 480, 1049]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_07_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-10/28_text_L12.png
try:
    _c28 = get_crop(28, 44, 30)
    canvas.paste(_c28, (270, 1045), _c28)
except Exception:
    pass
layout["L12"] = [270, 1045, 314, 1075]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_07_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-10/29_text_Log.png
try:
    _c29 = get_crop(29, 43, 27)
    canvas.paste(_c29, (250, 1108), _c29)
except Exception:
    pass
layout["Log"] = [250, 1108, 293, 1135]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_07_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-10/30_text_SCORERS.png
try:
    _c30 = get_crop(30, 83, 25)
    canvas.paste(_c30, (677, 1121), _c30)
except Exception:
    pass
layout["~SCORERS"] = [677, 1121, 760, 1146]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_07_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-10/31_text_Los.png
try:
    _c31 = get_crop(31, 44, 27)
    canvas.paste(_c31, (247, 1186), _c31)
except Exception:
    pass
layout["Los"] = [247, 1186, 291, 1213]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_07_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-10/32_text_16.png
try:
    _c32 = get_crop(32, 34, 27)
    canvas.paste(_c32, (361, 1357), _c32)
except Exception:
    pass
layout["16"] = [361, 1357, 395, 1384]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_07_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-10/33_text_S645.png
try:
    _c33 = get_crop(33, 60, 27)
    canvas.paste(_c33, (543, 1517), _c33)
except Exception:
    pass
layout["S645"] = [543, 1517, 603, 1544]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_07_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-10/34_text_S642.png
try:
    _c34 = get_crop(34, 60, 27)
    canvas.paste(_c34, (668, 1517), _c34)
except Exception:
    pass
layout["S642"] = [668, 1517, 728, 1544]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_07_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-10/35_text_S640.png
try:
    _c35 = get_crop(35, 62, 27)
    canvas.paste(_c35, (751, 1517), _c35)
except Exception:
    pass
layout["S640"] = [751, 1517, 813, 1544]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_07_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-10/36_text_S638.png
try:
    _c36 = get_crop(36, 59, 27)
    canvas.paste(_c36, (837, 1517), _c36)
except Exception:
    pass
layout["S638"] = [837, 1517, 896, 1544]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_07_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-10/37_text_Sort_by.png
try:
    _c37 = get_crop(37, 188, 68)
    canvas.paste(_c37, (626, 1740), _c37)
except Exception:
    pass
layout["Sort_by"] = [626, 1740, 814, 1808]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_07_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-10/38_text_Best_Seats.png
try:
    _c38 = get_crop(38, 269, 55)
    canvas.paste(_c38, (118, 2703), _c38)
except Exception:
    pass
layout["Best_Seats"] = [118, 2703, 387, 2758]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_07_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-10/39_text_Get_close_to_the_action_with_the_best_se.png
try:
    _c39 = get_crop(39, 1320, 267)
    canvas.paste(_c39, (60, 2633), _c39)
except Exception:
    pass
layout["Get_close_to_the_action_w"] = [60, 2633, 1380, 2900]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_07_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-10/40_text_RAF38.png
try:
    _c40 = get_crop(40, 84, 53)
    canvas.paste(_c40, (393, 721), _c40)
except Exception:
    pass
layout["RAF38"] = [393, 721, 477, 774]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_07_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-10/41_text_RAFZ.png
try:
    _c41 = get_crop(41, 73, 54)
    canvas.paste(_c41, (966, 721), _c41)
except Exception:
    pass
layout["_RAFZ"] = [966, 721, 1039, 775]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_07_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-10/42_text_RAF37.png
try:
    _c42 = get_crop(42, 87, 76)
    canvas.paste(_c42, (284, 758), _c42)
except Exception:
    pass
layout["RAF37"] = [284, 758, 371, 834]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_07_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-10/43_text_RAF8.png
try:
    _c43 = get_crop(43, 80, 71)
    canvas.paste(_c43, (1073, 760), _c43)
except Exception:
    pass
layout["~RAF8_"] = [1073, 760, 1153, 831]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_07_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-10/44_text_5605.png
try:
    _c44 = get_crop(44, 74, 51)
    canvas.paste(_c44, (405, 886), _c44)
except Exception:
    pass
layout["5605"] = [405, 886, 479, 937]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_07_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-10/45_text_-5618.png
try:
    _c45 = get_crop(45, 71, 50)
    canvas.paste(_c45, (960, 887), _c45)
except Exception:
    pass
layout["-5618"] = [960, 887, 1031, 937]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_07_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-10/46_text_NLOUNGE.png
try:
    _c46 = get_crop(46, 105, 61)
    canvas.paste(_c46, (932, 1441), _c46)
except Exception:
    pass
layout["~NLOUNGE"] = [932, 1441, 1037, 1502]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_07_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-10/47_text_S548.png
try:
    _c47 = get_crop(47, 74, 55)
    canvas.paste(_c47, (418, 1441), _c47)
except Exception:
    pass
layout["~S548"] = [418, 1441, 492, 1496]
