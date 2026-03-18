# page_id: page_seatgeek_21b637d11bea46b8adb3c2efc9f03501_09
# screenshot: 2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-12.png
# step_index: 9/10
# task: Open SeatGeek and find the soonest upcoming NBA game in New York with "Nets", record the cheapest price in google keep notes.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw UI background and structural elements for a mobile SeatGeek-like page
# Available variables: canvas (PIL Image 1440x2960 RGB), draw (ImageDraw), font_sm, font_md, font_lg, font_xl

# Colors
bg = (242, 244, 246)           # overall light gray background
status_bar_col = (229, 231, 233)
header_bg = (255, 255, 255)
card_bg = (255, 255, 255)
muted_div = (220, 224, 227)
map_border = (200, 204, 208)
soft_shadow = (235, 238, 240)

w, h = 1440, 2960

# Fill background
draw.rectangle([(0, 0), (w, h)], fill=bg)

# Status bar area (top)
status_h = 64
draw.rectangle([(0, 0), (w, status_h)], fill=status_bar_col)

# Top header / toolbar background (rounded white pill)
hdr_x0, hdr_y0 = 40, 80
hdr_x1, hdr_y1 = w - 40, 220
hdr_radius = 48
draw.rounded_rectangle([(hdr_x0, hdr_y0), (hdr_x1, hdr_y1)], radius=hdr_radius, fill=header_bg, outline=muted_div, width=1)

# Subtle divider below header area (separates header from chips/map)
div_y = 240
draw.line([(40, div_y), (w - 40, div_y)], fill=muted_div, width=1)

# Light background band where chips live (behind filter chips)
chips_band_y0 = 240
chips_band_y1 = 420
draw.rectangle([(0, chips_band_y0), (w, chips_band_y1)], fill=bg)

# Big circular stadium/map area with border and inner white surface
# Outer ring (border)
map_cx, map_cy = w // 2, 940
rx, ry = 600, 580  # radius x and y
outer_bbox = [map_cx - rx - 10, map_cy - ry - 10, map_cx + rx + 10, map_cy + ry + 10]
draw.ellipse(outer_bbox, fill=map_border)

# Inner white circle (map surface)
inner_bbox = [map_cx - rx + 6, map_cy - ry + 6, map_cx + rx - 6, map_cy + ry - 6]
draw.ellipse(inner_bbox, fill=card_bg)

# Very subtle inner ring to mimic stadium rim
ring_bbox = [map_cx - rx + 16, map_cy - ry + 16, map_cx + rx - 16, map_cy + ry - 16]
draw.ellipse(ring_bbox, outline=muted_div, width=2)

# Small shadow under the map to lift it slightly
shadow_bbox = [map_cx - rx - 40, map_cy + ry - 20, map_cx + rx + 40, map_cy + ry + 40]
draw.ellipse(shadow_bbox, fill=soft_shadow)

# Divider line between map and listings area
map_bottom = inner_bbox[3] + 20
draw.line([(40, map_bottom), (w - 40, map_bottom)], fill=muted_div, width=1)

# Listings card background (rounded white card anchored near bottom)
list_top = map_bottom + 60
list_margin = 20
list_radius = 28
draw.rounded_rectangle([(list_margin, list_top), (w - list_margin, h - 20)], radius=list_radius, fill=card_bg, outline=muted_div, width=1)

# Thin top shadow under the listings card to separate from map
draw.rectangle([(list_margin+2, list_top), (w - list_margin-2, list_top+6)], fill=soft_shadow)

# Inner content divider lines inside listings card (to suggest header and items separation)
# Header separator (just under the top area of the card)
card_header_y = list_top + 120
draw.line([(list_margin + 20, card_header_y), (w - list_margin - 20, card_header_y)], fill=muted_div, width=1)

# Subtle separators for list items (draw a few to suggest structure)
for i in range(0, 3):
    y = card_header_y + 220 + i * 380
    draw.line([(list_margin + 20, y), (w - list_margin - 20, y)], fill=(240,240,241), width=1)

# Small rounded thumbnail placeholder boxes on left of list items (structure only)
thumb_w, thumb_h = 220, 140
thumb_x = list_margin + 40
# Draw 3 thumbnail placeholders vertically aligned (background only)
for i in range(0, 3):
    ty = card_header_y + 30 + i * 380
    box = (thumb_x, ty, thumb_x + thumb_w, ty + thumb_h)
    draw.rounded_rectangle(box, radius=12, fill=(245,245,246), outline=(235,235,236), width=1)

# Right-side subtle vertical divider in listings header (sort area background)
sort_box = (w - list_margin - 320, list_top + 30, w - list_margin - 20, list_top + 110)
draw.rounded_rectangle(sort_box, radius=16, fill=(252,252,252), outline=muted_div, width=1)

# Final subtle bottom padding line
draw.line([(list_margin + 20, h - 160), (w - list_margin - 20, h - 160)], fill=(245,245,246), width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_09_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-12/00_icon_Include_fees.png
try:
    _c0 = get_crop(0, 335, 108)
    canvas.paste(_c0, (544, 312), _c0)
except Exception:
    pass
layout["Include_fees"] = [544, 312, 879, 420]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_09_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-12/01_icon_Courtside.png
try:
    _c1 = get_crop(1, 286, 108)
    canvas.paste(_c1, (915, 312), _c1)
except Exception:
    pass
layout["Courtside"] = [915, 312, 1201, 420]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_09_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-12/02_icon_Quantity.png
try:
    _c2 = get_crop(2, 268, 108)
    canvas.paste(_c2, (240, 312), _c2)
except Exception:
    pass
layout["Quantity"] = [240, 312, 508, 420]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_09_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-12/03_icon_8.0.png
try:
    _c3 = get_crop(3, 1440, 455)
    canvas.paste(_c3, (0, 2355), _c3)
except Exception:
    pass
layout["8.0"] = [0, 2355, 1440, 2810]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_09_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-12/04_icon_Tit.png
try:
    _c4 = get_crop(4, 156, 108)
    canvas.paste(_c4, (48, 312), _c4)
except Exception:
    pass
layout["Tit"] = [48, 312, 204, 420]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_09_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-12/05_icon_Center.png
try:
    _c5 = get_crop(5, 203, 108)
    canvas.paste(_c5, (1237, 312), _c5)
except Exception:
    pass
layout["Center"] = [1237, 312, 1440, 420]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_09_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-12/06_icon_Include_fees.png
try:
    _c6 = get_crop(6, 1344, 156)
    canvas.paste(_c6, (48, 120), _c6)
except Exception:
    pass
layout["Include_fees"] = [48, 120, 1392, 276]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_09_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-12/07_icon_Great_deal.png
try:
    _c7 = get_crop(7, 1440, 455)
    canvas.paste(_c7, (0, 2355), _c7)
except Exception:
    pass
layout["Great_deal"] = [0, 2355, 1440, 2810]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_09_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-12/08_icon_GK.png
try:
    _c8 = get_crop(8, 58, 56)
    canvas.paste(_c8, (179, 5), _c8)
except Exception:
    pass
layout["GK"] = [179, 5, 237, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_09_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-12/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 53, 65)
    canvas.paste(_c9, (1151, 1), _c9)
except Exception:
    pass
layout["icon_9"] = [1151, 1, 1204, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_09_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-12/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 103, 62)
    canvas.paste(_c10, (1212, 1), _c10)
except Exception:
    pass
layout["icon_10"] = [1212, 1, 1315, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_09_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-12/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 54, 59)
    canvas.paste(_c11, (1319, 2), _c11)
except Exception:
    pass
layout["icon_11"] = [1319, 2, 1373, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_09_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-12/12_icon_Sort_by_price.png
try:
    _c12 = get_crop(12, 455, 144)
    canvas.paste(_c12, (961, 1989), _c12)
except Exception:
    pass
layout["Sort_by_price"] = [961, 1989, 1416, 2133]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_09_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-12/13_icon_Center.png
try:
    _c13 = get_crop(13, 156, 156)
    canvas.paste(_c13, (1236, 120), _c13)
except Exception:
    pass
layout["Center"] = [1236, 120, 1392, 276]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_09_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-12/14_icon_S219_each.png
try:
    _c14 = get_crop(14, 381, 106)
    canvas.paste(_c14, (53, 2854), _c14)
except Exception:
    pass
layout["S219_each"] = [53, 2854, 434, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_09_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-12/15_icon_GK.png
try:
    _c15 = get_crop(15, 53, 59)
    canvas.paste(_c15, (117, 2), _c15)
except Exception:
    pass
layout["GK"] = [117, 2, 170, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_09_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-12/16_icon_6.38.png
try:
    _c16 = get_crop(16, 132, 62)
    canvas.paste(_c16, (6, 1), _c16)
except Exception:
    pass
layout["6.38"] = [6, 1, 138, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_09_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-12/17_text_B1o.png
try:
    _c17 = get_crop(17, 53, 27)
    canvas.paste(_c17, (971, 650), _c17)
except Exception:
    pass
layout["B1o]"] = [971, 650, 1024, 677]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_09_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-12/18_text_212.png
try:
    _c18 = get_crop(18, 48, 30)
    canvas.paste(_c18, (610, 712), _c18)
except Exception:
    pass
layout["212"] = [610, 712, 658, 742]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_09_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-12/19_text_210.png
try:
    _c19 = get_crop(19, 48, 27)
    canvas.paste(_c19, (779, 712), _c19)
except Exception:
    pass
layout["210"] = [779, 712, 827, 739]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_09_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-12/20_text_213.png
try:
    _c20 = get_crop(20, 48, 29)
    canvas.paste(_c20, (506, 731), _c20)
except Exception:
    pass
layout["213"] = [506, 731, 554, 760]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_09_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-12/21_text_209.png
try:
    _c21 = get_crop(21, 45, 29)
    canvas.paste(_c21, (886, 731), _c21)
except Exception:
    pass
layout["209"] = [886, 731, 931, 760]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_09_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-12/22_text_208.png
try:
    _c22 = get_crop(22, 46, 28)
    canvas.paste(_c22, (987, 781), _c22)
except Exception:
    pass
layout["208"] = [987, 781, 1033, 809]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_09_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-12/23_text_SS18.png
try:
    _c23 = get_crop(23, 62, 28)
    canvas.paste(_c23, (1091, 818), _c23)
except Exception:
    pass
layout["SS18"] = [1091, 818, 1153, 846]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_09_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-12/24_text_S47.png
try:
    _c24 = get_crop(24, 57, 27)
    canvas.paste(_c24, (673, 897), _c24)
except Exception:
    pass
layout["[S47"] = [673, 897, 730, 924]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_09_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-12/25_text_S45__543.png
try:
    _c25 = get_crop(25, 128, 36)
    canvas.paste(_c25, (739, 898), _c25)
except Exception:
    pass
layout["[S45_[543"] = [739, 898, 867, 934]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_09_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-12/26_text_S16.png
try:
    _c26 = get_crop(26, 60, 29)
    canvas.paste(_c26, (1149, 888), _c26)
except Exception:
    pass
layout["S16"] = [1149, 888, 1209, 917]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_09_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-12/27_text_LS52.png
try:
    _c27 = get_crop(27, 62, 29)
    canvas.paste(_c27, (483, 923), _c27)
except Exception:
    pass
layout["LS52"] = [483, 923, 545, 952]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_09_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-12/28_text_LS41.png
try:
    _c28 = get_crop(28, 57, 31)
    canvas.paste(_c28, (874, 916), _c28)
except Exception:
    pass
layout["LS41"] = [874, 916, 931, 947]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_09_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-12/29_text_SS15.png
try:
    _c29 = get_crop(29, 59, 27)
    canvas.paste(_c29, (1175, 932), _c29)
except Exception:
    pass
layout["SS15"] = [1175, 932, 1234, 959]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_09_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-12/30_text_LS54.png
try:
    _c30 = get_crop(30, 60, 29)
    canvas.paste(_c30, (418, 948), _c30)
except Exception:
    pass
layout["LS54"] = [418, 948, 478, 977]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_09_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-12/31_text_Ls39.png
try:
    _c31 = get_crop(31, 60, 29)
    canvas.paste(_c31, (957, 946), _c31)
except Exception:
    pass
layout["Ls39"] = [957, 946, 1017, 975]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_09_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-12/32_text_LS57.png
try:
    _c32 = get_crop(32, 57, 28)
    canvas.paste(_c32, (340, 1003), _c32)
except Exception:
    pass
layout["LS57"] = [340, 1003, 397, 1031]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_09_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-12/33_text_LS36.png
try:
    _c33 = get_crop(33, 58, 28)
    canvas.paste(_c33, (1040, 1003), _c33)
except Exception:
    pass
layout["LS36"] = [1040, 1003, 1098, 1031]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_09_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-12/34_text_206.png
try:
    _c34 = get_crop(34, 48, 27)
    canvas.paste(_c34, (1119, 1006), _c34)
except Exception:
    pass
layout["206"] = [1119, 1006, 1167, 1033]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_09_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-12/35_text_SS13.png
try:
    _c35 = get_crop(35, 59, 29)
    canvas.paste(_c35, (1214, 1027), _c35)
except Exception:
    pass
layout["SS13"] = [1214, 1027, 1273, 1056]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_09_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-12/36_text_205.png
try:
    _c36 = get_crop(36, 46, 28)
    canvas.paste(_c36, (1149, 1077), _c36)
except Exception:
    pass
layout["205"] = [1149, 1077, 1195, 1105]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_09_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-12/37_text_SS11.png
try:
    _c37 = get_crop(37, 57, 27)
    canvas.paste(_c37, (1235, 1126), _c37)
except Exception:
    pass
layout["SS11"] = [1235, 1126, 1292, 1153]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_09_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-12/38_text_SS10.png
try:
    _c38 = get_crop(38, 60, 30)
    canvas.paste(_c38, (1239, 1172), _c38)
except Exception:
    pass
layout["SS10"] = [1239, 1172, 1299, 1202]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_09_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-12/39_text_218.png
try:
    _c39 = get_crop(39, 45, 30)
    canvas.paste(_c39, (220, 1225), _c39)
except Exception:
    pass
layout["218"] = [220, 1225, 265, 1255]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_09_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-12/40_text_sS9.png
try:
    _c40 = get_crop(40, 48, 27)
    canvas.paste(_c40, (1246, 1219), _c40)
except Exception:
    pass
layout["sS9"] = [1246, 1219, 1294, 1246]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_09_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-12/41_text_SS8.png
try:
    _c41 = get_crop(41, 48, 29)
    canvas.paste(_c41, (1242, 1265), _c41)
except Exception:
    pass
layout["SS8"] = [1242, 1265, 1290, 1294]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_09_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-12/42_text_203.png
try:
    _c42 = get_crop(42, 46, 27)
    canvas.paste(_c42, (1149, 1313), _c42)
except Exception:
    pass
layout["203"] = [1149, 1313, 1195, 1340]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_09_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-12/43_text_SS6.png
try:
    _c43 = get_crop(43, 45, 27)
    canvas.paste(_c43, (1221, 1362), _c43)
except Exception:
    pass
layout["SS6"] = [1221, 1362, 1266, 1389]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_09_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-12/44_text_LS24.png
try:
    _c44 = get_crop(44, 60, 27)
    canvas.paste(_c44, (1038, 1387), _c44)
except Exception:
    pass
layout["LS24"] = [1038, 1387, 1098, 1414]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_09_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-12/45_text_LSS.png
try:
    _c45 = get_crop(45, 52, 36)
    canvas.paste(_c45, (422, 1438), _c45)
except Exception:
    pass
layout["LSS"] = [422, 1438, 474, 1474]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_09_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-12/46_text_SS4.png
try:
    _c46 = get_crop(46, 48, 30)
    canvas.paste(_c46, (1179, 1454), _c46)
except Exception:
    pass
layout["SS4"] = [1179, 1454, 1227, 1484]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_09_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-12/47_text_Ls7.png
try:
    _c47 = get_crop(47, 44, 31)
    canvas.paste(_c47, (485, 1464), _c47)
except Exception:
    pass
layout["Ls7"] = [485, 1464, 529, 1495]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_09_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-12/48_text_LS19.png
try:
    _c48 = get_crop(48, 60, 29)
    canvas.paste(_c48, (899, 1466), _c48)
except Exception:
    pass
layout["LS19"] = [899, 1466, 959, 1495]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_09_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-12/49_text_LS12.png
try:
    _c49 = get_crop(49, 57, 29)
    canvas.paste(_c49, (654, 1494), _c49)
except Exception:
    pass
layout["LS12"] = [654, 1494, 711, 1523]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_09_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-12/50_text_LS15_LS17.png
try:
    _c50 = get_crop(50, 134, 41)
    canvas.paste(_c50, (760, 1479), _c50)
except Exception:
    pass
layout["LS15_LS17"] = [760, 1479, 894, 1520]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_09_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-12/51_text_SS3.png
try:
    _c51 = get_crop(51, 48, 27)
    canvas.paste(_c51, (1154, 1501), _c51)
except Exception:
    pass
layout["SS3"] = [1154, 1501, 1202, 1528]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_09_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-12/52_text_222.png
try:
    _c52 = get_crop(52, 50, 36)
    canvas.paste(_c52, (491, 1530), _c52)
except Exception:
    pass
layout["222"] = [491, 1530, 541, 1566]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_09_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-12/53_text_MEDIA.png
try:
    _c53 = get_crop(53, 62, 25)
    canvas.paste(_c53, (689, 1535), _c53)
except Exception:
    pass
layout["MEDIA"] = [689, 1535, 751, 1560]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_09_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-12/54_text_226.png
try:
    _c54 = get_crop(54, 48, 30)
    canvas.paste(_c54, (895, 1528), _c54)
except Exception:
    pass
layout["226"] = [895, 1528, 943, 1558]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_09_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-12/55_text_225.png
try:
    _c55 = get_crop(55, 46, 27)
    canvas.paste(_c55, (809, 1547), _c55)
except Exception:
    pass
layout["225"] = [809, 1547, 855, 1574]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_09_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-12/56_text_SS1.png
try:
    _c56 = get_crop(56, 46, 30)
    canvas.paste(_c56, (1098, 1572), _c56)
except Exception:
    pass
layout["SS1"] = [1098, 1572, 1144, 1602]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_09_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-12/57_text_224UWC.png
try:
    _c57 = get_crop(57, 101, 28)
    canvas.paste(_c57, (684, 1699), _c57)
except Exception:
    pass
layout["224UWC"] = [684, 1699, 785, 1727]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_09_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-12/58_text_324.png
try:
    _c58 = get_crop(58, 58, 27)
    canvas.paste(_c58, (411, 1739), _c58)
except Exception:
    pass
layout["[324"] = [411, 1739, 469, 1766]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_09_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-12/59_text_325.png
try:
    _c59 = get_crop(59, 58, 27)
    canvas.paste(_c59, (499, 1741), _c59)
except Exception:
    pass
layout["[325"] = [499, 1741, 557, 1768]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_09_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-12/60_text_326.png
try:
    _c60 = get_crop(60, 59, 27)
    canvas.paste(_c60, (622, 1741), _c60)
except Exception:
    pass
layout["[326"] = [622, 1741, 681, 1768]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_09_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-12/61_text_327.png
try:
    _c61 = get_crop(61, 46, 27)
    canvas.paste(_c61, (825, 1741), _c61)
except Exception:
    pass
layout["327"] = [825, 1741, 871, 1768]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_09_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-12/62_text_328.png
try:
    _c62 = get_crop(62, 48, 27)
    canvas.paste(_c62, (936, 1741), _c62)
except Exception:
    pass
layout["328"] = [936, 1741, 984, 1768]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_09_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-12/63_text_679_Listings.png
try:
    _c63 = get_crop(63, 330, 79)
    canvas.paste(_c63, (56, 2027), _c63)
except Exception:
    pass
layout["679_Listings"] = [56, 2027, 386, 2106]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_09_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-12/64_text_We_sell_resale_tickets._Resale_tickets_m.png
try:
    _c64 = get_crop(64, 1440, 455)
    canvas.paste(_c64, (0, 2355), _c64)
except Exception:
    pass
layout["€_We_sell_resale_tickets."] = [0, 2355, 1440, 2810]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_09_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-12/65_text_face_value.png
try:
    _c65 = get_crop(65, 218, 43)
    canvas.paste(_c65, (57, 2256), _c65)
except Exception:
    pass
layout["face_value:"] = [57, 2256, 275, 2299]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_09_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-12/66_text_S219_each.png
try:
    _c66 = get_crop(66, 263, 65)
    canvas.paste(_c66, (485, 2862), _c66)
except Exception:
    pass
layout["S219_each"] = [485, 2862, 748, 2927]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_09_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-12/67_text_J1O8WC.png
try:
    _c67 = get_crop(67, 87, 41)
    canvas.paste(_c67, (552, 921), _c67)
except Exception:
    pass
layout["J1O8WC"] = [552, 921, 639, 962]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_09_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-12/68_clickable_Back.png
try:
    _c68 = get_crop(68, 156, 156)
    canvas.paste(_c68, (48, 120), _c68)
except Exception:
    pass
layout["Back"] = [48, 120, 204, 276]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_09_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-12/69_clickable_Nets_at_Knicks.png
try:
    _c69 = get_crop(69, 317, 156)
    canvas.paste(_c69, (204, 120), _c69)
except Exception:
    pass
layout["Nets_at_Knicks"] = [204, 120, 521, 276]
