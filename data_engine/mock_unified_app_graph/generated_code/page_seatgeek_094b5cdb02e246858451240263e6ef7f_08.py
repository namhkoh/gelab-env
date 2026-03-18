# page_id: page_seatgeek_094b5cdb02e246858451240263e6ef7f_08
# screenshot: 2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-11.png
# step_index: 8/9
# task: Open SeatGeek. Find the soonest upcoming NBA game in Boston with "Celtics". What is the highest price for each ticket?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Background fill (dominant light gray-blue)
w, h = canvas.size
draw.rectangle([(0, 0), (w, h)], fill=(242, 245, 247))

# Status bar area at top (~50px high, slightly darker)
status_h = 80
draw.rectangle([(0, 0), (w, status_h)], fill=(224, 228, 231))

# Header/toolbar background (rounded pill behind title)
header_left = 48
header_top = 120
header_w = 1344
header_h = 156
header_bbox_shadow = (header_left-4, header_top+8, header_left+header_w+4, header_top+header_h+8)
draw.rounded_rectangle(header_bbox_shadow, radius=80, fill=(226, 229, 231))
header_bbox = (header_left, header_top, header_left+header_w, header_top+header_h)
draw.rounded_rectangle(header_bbox, radius=76, fill=(255, 255, 255), outline=(213, 217, 219), width=2)

# Divider line below header (subtle)
divider_y = header_top + header_h + 16
draw.line([(48, divider_y), (w-48, divider_y)], fill=(226, 229, 231), width=1)

# Filter pills area separator (thin line under filter row)
filter_row_y = 312 + 120  # visually below filters
draw.line([(36, filter_row_y), (w-36, filter_row_y)], fill=(234, 237, 239), width=1)

# Large seating-map card background (rounded rectangle with drop shadow)
map_left = 60
map_top = 420
map_right = w - 60
map_bottom = 1680
# shadow
draw.rounded_rectangle((map_left+8, map_top+12, map_right+8, map_bottom+12), radius=40, fill=(230, 232, 234))
# white card
draw.rounded_rectangle((map_left, map_top, map_right, map_bottom), radius=40, fill=(255, 255, 255), outline=(218, 221, 224), width=2)
# subtle inner border ring to emulate seating outline
inner_ring_margin = 18
draw.rounded_rectangle((map_left+inner_ring_margin, map_top+inner_ring_margin, map_right-inner_ring_margin, map_bottom-inner_ring_margin),
                       radius=32, outline=(236, 238, 240), width=2)

# Thin separator between map and listings area
sep_y = map_bottom + 40
draw.line([(36, sep_y), (w-36, sep_y)], fill=(226, 229, 231), width=1)

# Listings container (white sheet with rounded top corners)
list_top = sep_y + 24
list_left = 12
list_right = w - 12
list_bottom = h
# shadow for listings panel
draw.rounded_rectangle((list_left+6, list_top+8, list_right+6, list_bottom+8), radius=28, fill=(229, 231, 233))
# main listings background
draw.rounded_rectangle((list_left, list_top, list_right, list_bottom), radius=28, fill=(255, 255, 255), outline=(220, 223, 225), width=1)

# Listings header strip (thin area for "1097 Listings" / sort)
header_strip_h = 120
draw.rectangle((list_left+22, list_top+18, list_right-22, list_top+18+header_strip_h), fill=(255,255,255))
# subtle bottom divider
draw.line([(list_left+22, list_top+18+header_strip_h), (list_right-22, list_top+18+header_strip_h)], fill=(234,236,238), width=1)

# Individual listing card backgrounds (rounded rectangles), two sample rows as structure only
card_margin_x = list_left + 32
card_width = list_right - 32
card_h = 380
first_card_top = list_top + 18 + header_strip_h + 24
second_card_top = first_card_top + card_h + 28

# First listing card
draw.rounded_rectangle((card_margin_x, first_card_top, card_width, first_card_top + card_h),
                       radius=20, fill=(255,255,255), outline=(236,238,239), width=1)
# subtle shadow under first card
draw.rectangle((card_margin_x+4, first_card_top+card_h+2, card_width+4, first_card_top+card_h+6), fill=(240,241,242))

# Separator line between first and second card area
draw.line([(card_margin_x, first_card_top + card_h + 16), (card_width, first_card_top + card_h + 16)], fill=(240,241,242), width=1)

# Second listing card
draw.rounded_rectangle((card_margin_x, second_card_top, card_width, second_card_top + card_h),
                       radius=20, fill=(255,255,255), outline=(236,238,239), width=1)
draw.rectangle((card_margin_x+4, second_card_top+card_h+2, card_width+4, second_card_top+card_h+6), fill=(240,241,242))

# Thin separators for list items (additional subtle lines)
for y in range(second_card_top + card_h + 24, list_bottom - 40, 120):
    draw.line([(36, y), (w-36, y)], fill=(245,246,247), width=1)

# Small accent divider under top status bar (to separate icons area)
draw.line([(0, status_h), (w, status_h)], fill=(218, 221, 223), width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_08_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-11/00_icon_Include_fees.png
try:
    _c0 = get_crop(0, 335, 108)
    canvas.paste(_c0, (544, 312), _c0)
except Exception:
    pass
layout["Include_fees"] = [544, 312, 879, 420]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_08_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-11/01_icon_Courtside.png
try:
    _c1 = get_crop(1, 286, 108)
    canvas.paste(_c1, (915, 312), _c1)
except Exception:
    pass
layout["Courtside"] = [915, 312, 1201, 420]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_08_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-11/02_icon_Quantity.png
try:
    _c2 = get_crop(2, 268, 108)
    canvas.paste(_c2, (240, 312), _c2)
except Exception:
    pass
layout["Quantity"] = [240, 312, 508, 420]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_08_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-11/03_icon_Tit.png
try:
    _c3 = get_crop(3, 156, 108)
    canvas.paste(_c3, (48, 312), _c3)
except Exception:
    pass
layout["Tit"] = [48, 312, 204, 420]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_08_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-11/04_icon_Center.png
try:
    _c4 = get_crop(4, 203, 108)
    canvas.paste(_c4, (1237, 312), _c4)
except Exception:
    pass
layout["Center"] = [1237, 312, 1440, 420]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_08_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-11/05_icon_9.5.png
try:
    _c5 = get_crop(5, 1440, 371)
    canvas.paste(_c5, (0, 2589), _c5)
except Exception:
    pass
layout["9.5"] = [0, 2589, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_08_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-11/06_icon_E_Conf_Ist_Rnd_TBD_at_Celtics_Gm_2_HG_2.png
try:
    _c6 = get_crop(6, 1344, 156)
    canvas.paste(_c6, (48, 120), _c6)
except Exception:
    pass
layout["E_Conf_Ist_Rnd:_TBD_at_Ce"] = [48, 120, 1392, 276]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_08_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-11/07_icon_1097_Listings.png
try:
    _c7 = get_crop(7, 1440, 455)
    canvas.paste(_c7, (0, 2134), _c7)
except Exception:
    pass
layout["1097_Listings"] = [0, 2134, 1440, 2589]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_08_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-11/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 48, 65)
    canvas.paste(_c8, (1154, 1), _c8)
except Exception:
    pass
layout["icon_8"] = [1154, 1, 1202, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_08_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-11/09_icon_5.00_Wy.png
try:
    _c9 = get_crop(9, 54, 60)
    canvas.paste(_c9, (181, 2), _c9)
except Exception:
    pass
layout["5.00_Wy"] = [181, 2, 235, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_08_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-11/10_icon_5.00_Wy.png
try:
    _c10 = get_crop(10, 64, 62)
    canvas.paste(_c10, (113, 1), _c10)
except Exception:
    pass
layout["5.00_Wy"] = [113, 1, 177, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_08_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-11/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 101, 63)
    canvas.paste(_c11, (1214, 1), _c11)
except Exception:
    pass
layout["icon_11"] = [1214, 1, 1315, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_08_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-11/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 64, 60)
    canvas.paste(_c12, (242, 2), _c12)
except Exception:
    pass
layout["icon_12"] = [242, 2, 306, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_08_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-11/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 53, 58)
    canvas.paste(_c13, (1320, 3), _c13)
except Exception:
    pass
layout["icon_13"] = [1320, 3, 1373, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_08_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-11/14_icon_Amazing_deal.png
try:
    _c14 = get_crop(14, 1440, 455)
    canvas.paste(_c14, (0, 2134), _c14)
except Exception:
    pass
layout["Amazing_deal"] = [0, 2134, 1440, 2589]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_08_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-11/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 59, 61)
    canvas.paste(_c15, (313, 2), _c15)
except Exception:
    pass
layout["icon_15"] = [313, 2, 372, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_08_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-11/16_icon_icon_16.png
try:
    _c16 = get_crop(16, 51, 63)
    canvas.paste(_c16, (382, 1), _c16)
except Exception:
    pass
layout["icon_16"] = [382, 1, 433, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_08_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-11/17_icon_Center.png
try:
    _c17 = get_crop(17, 156, 156)
    canvas.paste(_c17, (1236, 120), _c17)
except Exception:
    pass
layout["Center"] = [1236, 120, 1392, 276]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_08_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-11/18_icon_Sort_by_price.png
try:
    _c18 = get_crop(18, 455, 144)
    canvas.paste(_c18, (961, 1989), _c18)
except Exception:
    pass
layout["Sort_by_price"] = [961, 1989, 1416, 2133]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_08_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-11/19_text_STANDING.png
try:
    _c19 = get_crop(19, 97, 25)
    canvas.paste(_c19, (670, 689), _c19)
except Exception:
    pass
layout["~STANDING"] = [670, 689, 767, 714]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_08_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-11/20_text_STUI.png
try:
    _c20 = get_crop(20, 62, 29)
    canvas.paste(_c20, (488, 717), _c20)
except Exception:
    pass
layout["STUI"] = [488, 717, 550, 746]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_08_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-11/21_text_STU2.png
try:
    _c21 = get_crop(21, 65, 29)
    canvas.paste(_c21, (566, 717), _c21)
except Exception:
    pass
layout["~STU2"] = [566, 717, 631, 746]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_08_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-11/22_text_STUS.png
try:
    _c22 = get_crop(22, 64, 29)
    canvas.paste(_c22, (807, 717), _c22)
except Exception:
    pass
layout["STUS"] = [807, 717, 871, 746]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_08_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-11/23_text_STU6.png
try:
    _c23 = get_crop(23, 64, 29)
    canvas.paste(_c23, (888, 717), _c23)
except Exception:
    pass
layout["STU6"] = [888, 717, 952, 746]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_08_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-11/24_text_S608.png
try:
    _c24 = get_crop(24, 60, 27)
    canvas.paste(_c24, (543, 876), _c24)
except Exception:
    pass
layout["S608"] = [543, 876, 603, 903]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_08_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-11/25_text_S611.png
try:
    _c25 = get_crop(25, 57, 27)
    canvas.paste(_c25, (668, 876), _c25)
except Exception:
    pass
layout["S611"] = [668, 876, 725, 903]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_08_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-11/26_text_613.png
try:
    _c26 = get_crop(26, 60, 27)
    canvas.paste(_c26, (751, 876), _c26)
except Exception:
    pass
layout["613"] = [751, 876, 811, 903]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_08_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-11/27_text_S615.png
try:
    _c27 = get_crop(27, 59, 29)
    canvas.paste(_c27, (837, 874), _c27)
except Exception:
    pass
layout["S615"] = [837, 874, 896, 903]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_08_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-11/28_text_20.png
try:
    _c28 = get_crop(28, 37, 27)
    canvas.paste(_c28, (469, 953), _c28)
except Exception:
    pass
layout["20"] = [469, 953, 506, 980]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_08_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-11/29_text_19.png
try:
    _c29 = get_crop(29, 32, 27)
    canvas.paste(_c29, (448, 1022), _c29)
except Exception:
    pass
layout["19"] = [448, 1022, 480, 1049]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_08_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-11/30_text_L12.png
try:
    _c30 = get_crop(30, 44, 30)
    canvas.paste(_c30, (270, 1045), _c30)
except Exception:
    pass
layout["L12"] = [270, 1045, 314, 1075]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_08_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-11/31_text_Log.png
try:
    _c31 = get_crop(31, 45, 27)
    canvas.paste(_c31, (250, 1108), _c31)
except Exception:
    pass
layout["Log"] = [250, 1108, 295, 1135]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_08_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-11/32_text_SCORERS.png
try:
    _c32 = get_crop(32, 83, 25)
    canvas.paste(_c32, (677, 1121), _c32)
except Exception:
    pass
layout["~SCORERS"] = [677, 1121, 760, 1146]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_08_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-11/33_text_Lo5.png
try:
    _c33 = get_crop(33, 44, 27)
    canvas.paste(_c33, (247, 1186), _c33)
except Exception:
    pass
layout["Lo5"] = [247, 1186, 291, 1213]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_08_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-11/34_text_16.png
try:
    _c34 = get_crop(34, 34, 27)
    canvas.paste(_c34, (361, 1357), _c34)
except Exception:
    pass
layout["16"] = [361, 1357, 395, 1384]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_08_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-11/35_text_S645.png
try:
    _c35 = get_crop(35, 60, 27)
    canvas.paste(_c35, (543, 1517), _c35)
except Exception:
    pass
layout["S645"] = [543, 1517, 603, 1544]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_08_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-11/36_text_S642.png
try:
    _c36 = get_crop(36, 60, 27)
    canvas.paste(_c36, (668, 1517), _c36)
except Exception:
    pass
layout["S642"] = [668, 1517, 728, 1544]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_08_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-11/37_text_S640.png
try:
    _c37 = get_crop(37, 62, 27)
    canvas.paste(_c37, (751, 1517), _c37)
except Exception:
    pass
layout["S640"] = [751, 1517, 813, 1544]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_08_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-11/38_text_S638.png
try:
    _c38 = get_crop(38, 59, 27)
    canvas.paste(_c38, (837, 1517), _c38)
except Exception:
    pass
layout["S638"] = [837, 1517, 896, 1544]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_08_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-11/39_text_RAF28.png
try:
    _c39 = get_crop(39, 74, 27)
    canvas.paste(_c39, (485, 1674), _c39)
except Exception:
    pass
layout["RAF28"] = [485, 1674, 559, 1701]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_08_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-11/40_text_STANDING.png
try:
    _c40 = get_crop(40, 97, 25)
    canvas.paste(_c40, (670, 1683), _c40)
except Exception:
    pass
layout["~STANDING"] = [670, 1683, 767, 1708]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_08_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-11/41_text_1097_Listings.png
try:
    _c41 = get_crop(41, 359, 74)
    canvas.paste(_c41, (51, 2029), _c41)
except Exception:
    pass
layout["1097_Listings"] = [51, 2029, 410, 2103]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_08_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-11/42_text_S177_each.png
try:
    _c42 = get_crop(42, 1440, 371)
    canvas.paste(_c42, (0, 2589), _c42)
except Exception:
    pass
layout["S177_each"] = [0, 2589, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_08_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-11/43_text_Price_includes_fees.png
try:
    _c43 = get_crop(43, 1440, 371)
    canvas.paste(_c43, (0, 2589), _c43)
except Exception:
    pass
layout["Price_includes_fees"] = [0, 2589, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_08_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-11/44_text_9.5.png
try:
    _c44 = get_crop(44, 52, 36)
    canvas.paste(_c44, (501, 2809), _c44)
except Exception:
    pass
layout["9.5"] = [501, 2809, 553, 2845]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_08_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-11/45_text_Amazing_deal.png
try:
    _c45 = get_crop(45, 1440, 371)
    canvas.paste(_c45, (0, 2589), _c45)
except Exception:
    pass
layout["Amazing_deal"] = [0, 2589, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_08_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-11/46_text_1-3_tickets.png
try:
    _c46 = get_crop(46, 219, 48)
    canvas.paste(_c46, (488, 2872), _c46)
except Exception:
    pass
layout["1-3_tickets"] = [488, 2872, 707, 2920]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_08_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-11/47_text_an.png
try:
    _c47 = get_crop(47, 54, 9)
    canvas.paste(_c47, (524, 2948), _c47)
except Exception:
    pass
layout["an"] = [524, 2948, 578, 2957]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_08_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-11/48_text_RAF38.png
try:
    _c48 = get_crop(48, 84, 53)
    canvas.paste(_c48, (393, 721), _c48)
except Exception:
    pass
layout["RAF38"] = [393, 721, 477, 774]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_08_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-11/49_text_RAFZ.png
try:
    _c49 = get_crop(49, 73, 54)
    canvas.paste(_c49, (966, 721), _c49)
except Exception:
    pass
layout["~RAFZ"] = [966, 721, 1039, 775]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_08_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-11/50_text_RAF37.png
try:
    _c50 = get_crop(50, 84, 75)
    canvas.paste(_c50, (285, 758), _c50)
except Exception:
    pass
layout["RAF37"] = [285, 758, 369, 833]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_08_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-11/51_text_RAF8.png
try:
    _c51 = get_crop(51, 80, 71)
    canvas.paste(_c51, (1073, 760), _c51)
except Exception:
    pass
layout["~RAF8"] = [1073, 760, 1153, 831]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_08_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-11/52_text_S605.png
try:
    _c52 = get_crop(52, 71, 50)
    canvas.paste(_c52, (407, 886), _c52)
except Exception:
    pass
layout["S605"] = [407, 886, 478, 936]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_08_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-11/53_text_-5618.png
try:
    _c53 = get_crop(53, 71, 50)
    canvas.paste(_c53, (960, 887), _c53)
except Exception:
    pass
layout["-5618"] = [960, 887, 1031, 937]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_08_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-11/54_text_NLOUNGE.png
try:
    _c54 = get_crop(54, 106, 62)
    canvas.paste(_c54, (931, 1440), _c54)
except Exception:
    pass
layout["~NLOUNGE"] = [931, 1440, 1037, 1502]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_08_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-11/55_text_S548.png
try:
    _c55 = get_crop(55, 75, 55)
    canvas.paste(_c55, (418, 1441), _c55)
except Exception:
    pass
layout["~S548"] = [418, 1441, 493, 1496]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_08_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-11/56_text_RAF1Z.png
try:
    _c56 = get_crop(56, 83, 52)
    canvas.paste(_c56, (949, 1649), _c56)
except Exception:
    pass
layout["RAF1Z"] = [949, 1649, 1032, 1701]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_08_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-11/57_clickable_Back.png
try:
    _c57 = get_crop(57, 156, 156)
    canvas.paste(_c57, (48, 120), _c57)
except Exception:
    pass
layout["Back"] = [48, 120, 204, 276]
