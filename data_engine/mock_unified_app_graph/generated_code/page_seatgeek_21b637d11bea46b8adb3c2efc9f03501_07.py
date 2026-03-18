# page_id: page_seatgeek_21b637d11bea46b8adb3c2efc9f03501_07
# screenshot: 2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-10.png
# step_index: 7/10
# task: Open SeatGeek and find the soonest upcoming NBA game in New York with "Nets", record the cheapest price in google keep notes.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw overall background (dominant pale bluish-gray)
bg_color = (236, 239, 241)
draw.rectangle([0, 0, 1440, 2960], fill=bg_color)

# Status bar area at top (~80px)
status_h = 80
status_color = (220, 223, 226)
draw.rectangle([0, 0, 1440, status_h], fill=status_color)

# Thin bottom divider under status bar
draw.line([(0, status_h), (1440, status_h)], fill=(200, 203, 206), width=1)

# Header/toolbar background (rounded white pill with subtle shadow)
header_margin_x = 30
header_top = status_h + 18
header_bottom = header_top + 100
header_box = [header_margin_x, header_top, 1440 - header_margin_x, header_bottom]
# shadow
shadow_box = [header_box[0], header_box[1] + 6, header_box[2], header_box[3] + 6]
draw.rounded_rectangle(shadow_box, radius=52, fill=(210, 213, 215))
# main header
draw.rounded_rectangle(header_box, radius=52, fill=(255, 255, 255))
# subtle vertical divider on the right side of header (to imply info icon area)
divider_x = header_box[2] - 110
draw.line([(divider_x, header_box[1]+18), (divider_x, header_box[3]-18)], fill=(230,230,232), width=1)

# Row of filter pill backgrounds (placeholders behind actual pills)
pills_y_center = header_bottom + 100
pill_height = 110
pill_radius = pill_height // 2
pill_spacing = 40
# approximate pill widths and positions (left-to-right)
pill_positions = [
    (60, pills_y_center - pill_height//2, 60 + 160, pills_y_center + pill_height//2),
    (220, pills_y_center - pill_height//2, 220 + 280, pills_y_center + pill_height//2),
    (520, pills_y_center - pill_height//2, 520 + 340, pills_y_center + pill_height//2),
    (880, pills_y_center - pill_height//2, 880 + 295, pills_y_center + pill_height//2),
    (1180, pills_y_center - pill_height//2, 1180 + 210, pills_y_center + pill_height//2),
]
pill_bg = (247, 249, 250)
selected_pill_bg = (25, 25, 25)  # selected dark pill behind "Include fees"
for i, p in enumerate(pill_positions):
    col = selected_pill_bg if i == 2 else pill_bg  # make middle pill appear selected
    # draw pill shadow
    shadow = [p[0], p[1]+6, p[2], p[3]+6]
    draw.rounded_rectangle(shadow, radius=pill_radius, fill=(220,222,224))
    draw.rounded_rectangle(p, radius=pill_radius, fill=col)

# Subtle separator line under pills
sep_y = pills_y_center + pill_height//2 + 30
draw.line([(40, sep_y), (1400, sep_y)], fill=(220,223,225), width=1)

# Large circular content background for the arena map (light ring)
# Place it centered roughly below the pills; leave margins
map_top = sep_y + 20
map_left = 60
map_right = 1440 - 60
map_bottom = map_top + 1200  # large circle/oval
# main map background (very light)
map_bg = (236, 239, 241)
draw.ellipse([map_left, map_top, map_right, map_bottom], fill=map_bg, outline=(200,203,206), width=6)
# inner subtle darker ring to suggest map boundary
inner_margin = 30
draw.ellipse([map_left+inner_margin, map_top+inner_margin, map_right-inner_margin, map_bottom-inner_margin],
             outline=(210,213,216), width=4)

# Faint overlay circle to give depth (center highlight)
center_highlight = (255,255,255)
cx0 = (map_left + map_right)//2 - 260
cy0 = (map_top + map_bottom)//2 - 200
draw.ellipse([cx0, cy0, cx0+520, cy0+520], fill=center_highlight)

# Divider line above bottom modal sheet (to visually separate content)
modal_start_y = 1680
draw.line([(0, modal_start_y), (1440, modal_start_y)], fill=(210,213,216), width=1)

# Bottom modal sheet (rounded top corners)
sheet_top = modal_start_y
sheet_radius = 36
sheet_box = [0, sheet_top, 1440, 2960]
# subtle shadow for sheet (slightly darker strip above)
draw.rectangle([0, sheet_top-6, 1440, sheet_top], fill=(220,223,225))
draw.rounded_rectangle(sheet_box, radius=sheet_radius, fill=(255,255,255))

# Modal top drag indicator
drag_w = 160
drag_h = 10
drag_x1 = (1440 - drag_w)//2
drag_x2 = drag_x1 + drag_w
drag_y1 = sheet_top + 18
drag_y2 = drag_y1 + drag_h
draw.rounded_rectangle([drag_x1, drag_y1, drag_x2, drag_y2], radius=6, fill=(230,232,235))

# Cards inside modal: three rounded cards with light borders
card_margin_x = 48
card_w_left = card_margin_x
card_w_right = 1440 - card_margin_x

first_card_top = sheet_top + 86
card_height = 200
card_gap = 36

# Deal Score card (top)
deal_card = [card_w_left, first_card_top, card_w_right, first_card_top + card_height]
draw.rounded_rectangle(deal_card, radius=24, fill=(255,255,255), outline=(230,232,234), width=2)

# Price card (middle - emphasized with a thicker darker border)
price_top = deal_card[3] + card_gap
price_card = [card_w_left, price_top, card_w_right, price_top + card_height]
draw.rounded_rectangle(price_card, radius=24, fill=(255,255,255), outline=(30,30,30), width=4)

# Best Seats card (bottom)
best_top = price_card[3] + card_gap
best_card = [card_w_left, best_top, card_w_right, best_top + card_height]
draw.rounded_rectangle(best_card, radius=24, fill=(255,255,255), outline=(230,232,234), width=2)

# Very light separators between cards (subtle lines)
draw.line([(card_w_left+20, deal_card[3]+12), (card_w_right-20, deal_card[3]+12)], fill=(245,245,246), width=1)
draw.line([(card_w_left+20, price_card[3]+12), (card_w_right-20, price_card[3]+12)], fill=(245,245,246), width=1)

# Add subtle shadow under modal inner cards (soft rectangles offset)
shadow_offset = 8
for c in (deal_card, price_card, best_card):
    s = [c[0], c[1]+shadow_offset, c[2], c[3]+shadow_offset]
    draw.rounded_rectangle(s, radius=24, fill=(245,246,247))

# Small top border for overall page (thin)
draw.line([(0, 0), (1440, 0)], fill=(200,200,200))

# Final subtle vignette at the very bottom (to anchor sheet)
draw.rectangle([0, 2920, 1440, 2960], fill=(240,241,242))

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_07_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-10/00_icon_We_rate_each_ticket_based_on_location_pr.png
try:
    _c0 = get_crop(0, 1320, 267)
    canvas.paste(_c0, (60, 2318), _c0)
except Exception:
    pass
layout["We_rate_each_ticket_based"] = [60, 2318, 1380, 2585]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_07_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-10/01_icon_Include.png
try:
    _c1 = get_crop(1, 341, 117)
    canvas.paste(_c1, (537, 309), _c1)
except Exception:
    pass
layout["Include"] = [537, 309, 878, 426]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_07_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-10/02_icon_Courtside.png
try:
    _c2 = get_crop(2, 295, 119)
    canvas.paste(_c2, (909, 308), _c2)
except Exception:
    pass
layout["Courtside"] = [909, 308, 1204, 427]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_07_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-10/03_icon_Quantity.png
try:
    _c3 = get_crop(3, 280, 120)
    canvas.paste(_c3, (231, 307), _c3)
except Exception:
    pass
layout["Quantity"] = [231, 307, 511, 427]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_07_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-10/04_icon_Tit.png
try:
    _c4 = get_crop(4, 169, 122)
    canvas.paste(_c4, (37, 308), _c4)
except Exception:
    pass
layout["Tit"] = [37, 308, 206, 430]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_07_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-10/05_icon_We_rate_each_ticket_based_on_location_pr.png
try:
    _c5 = get_crop(5, 1320, 329)
    canvas.paste(_c5, (60, 1941), _c5)
except Exception:
    pass
layout["We_rate_each_ticket_based"] = [60, 1941, 1380, 2270]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_07_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-10/06_icon_Center.png
try:
    _c6 = get_crop(6, 211, 119)
    canvas.paste(_c6, (1229, 308), _c6)
except Exception:
    pass
layout["Center"] = [1229, 308, 1440, 427]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_07_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-10/07_icon_Include.png
try:
    _c7 = get_crop(7, 1342, 165)
    canvas.paste(_c7, (39, 118), _c7)
except Exception:
    pass
layout["Include"] = [39, 118, 1381, 283]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_07_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-10/08_icon_Center.png
try:
    _c8 = get_crop(8, 101, 109)
    canvas.paste(_c8, (1256, 146), _c8)
except Exception:
    pass
layout["Center"] = [1256, 146, 1357, 255]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_07_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-10/09_icon_GK.png
try:
    _c9 = get_crop(9, 57, 55)
    canvas.paste(_c9, (180, 6), _c9)
except Exception:
    pass
layout["GK"] = [180, 6, 237, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_07_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-10/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 55, 66)
    canvas.paste(_c10, (1151, 1), _c10)
except Exception:
    pass
layout["icon_10"] = [1151, 1, 1206, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_07_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-10/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 52, 57)
    canvas.paste(_c11, (1320, 3), _c11)
except Exception:
    pass
layout["icon_11"] = [1320, 3, 1372, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_07_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-10/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 106, 62)
    canvas.paste(_c12, (1211, 1), _c12)
except Exception:
    pass
layout["icon_12"] = [1211, 1, 1317, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_07_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-10/13_icon_GK.png
try:
    _c13 = get_crop(13, 54, 58)
    canvas.paste(_c13, (117, 3), _c13)
except Exception:
    pass
layout["GK"] = [117, 3, 171, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_07_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-10/14_text_6.38.png
try:
    _c14 = get_crop(14, 89, 41)
    canvas.paste(_c14, (22, 17), _c14)
except Exception:
    pass
layout["6.38"] = [22, 17, 111, 58]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_07_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-10/15_text_B1o.png
try:
    _c15 = get_crop(15, 53, 27)
    canvas.paste(_c15, (971, 650), _c15)
except Exception:
    pass
layout["B1o]"] = [971, 650, 1024, 677]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_07_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-10/16_text_212.png
try:
    _c16 = get_crop(16, 48, 30)
    canvas.paste(_c16, (610, 712), _c16)
except Exception:
    pass
layout["212"] = [610, 712, 658, 742]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_07_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-10/17_text_210.png
try:
    _c17 = get_crop(17, 48, 27)
    canvas.paste(_c17, (779, 712), _c17)
except Exception:
    pass
layout["210"] = [779, 712, 827, 739]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_07_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-10/18_text_213.png
try:
    _c18 = get_crop(18, 48, 29)
    canvas.paste(_c18, (506, 731), _c18)
except Exception:
    pass
layout["213"] = [506, 731, 554, 760]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_07_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-10/19_text_209.png
try:
    _c19 = get_crop(19, 45, 29)
    canvas.paste(_c19, (886, 731), _c19)
except Exception:
    pass
layout["209"] = [886, 731, 931, 760]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_07_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-10/20_text_208.png
try:
    _c20 = get_crop(20, 46, 28)
    canvas.paste(_c20, (987, 781), _c20)
except Exception:
    pass
layout["208"] = [987, 781, 1033, 809]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_07_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-10/21_text_SS18.png
try:
    _c21 = get_crop(21, 62, 28)
    canvas.paste(_c21, (1091, 818), _c21)
except Exception:
    pass
layout["SS18"] = [1091, 818, 1153, 846]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_07_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-10/22_text_S47.png
try:
    _c22 = get_crop(22, 57, 27)
    canvas.paste(_c22, (673, 897), _c22)
except Exception:
    pass
layout["[S47"] = [673, 897, 730, 924]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_07_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-10/23_text_S45__543.png
try:
    _c23 = get_crop(23, 128, 36)
    canvas.paste(_c23, (739, 898), _c23)
except Exception:
    pass
layout["[S45_[543"] = [739, 898, 867, 934]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_07_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-10/24_text_SS16.png
try:
    _c24 = get_crop(24, 60, 29)
    canvas.paste(_c24, (1149, 888), _c24)
except Exception:
    pass
layout["SS16"] = [1149, 888, 1209, 917]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_07_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-10/25_text_LS52.png
try:
    _c25 = get_crop(25, 62, 29)
    canvas.paste(_c25, (483, 923), _c25)
except Exception:
    pass
layout["LS52"] = [483, 923, 545, 952]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_07_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-10/26_text_s41.png
try:
    _c26 = get_crop(26, 63, 37)
    canvas.paste(_c26, (871, 913), _c26)
except Exception:
    pass
layout["~[s41"] = [871, 913, 934, 950]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_07_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-10/27_text_SS15.png
try:
    _c27 = get_crop(27, 59, 27)
    canvas.paste(_c27, (1175, 932), _c27)
except Exception:
    pass
layout["SS15"] = [1175, 932, 1234, 959]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_07_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-10/28_text_LS54.png
try:
    _c28 = get_crop(28, 60, 29)
    canvas.paste(_c28, (418, 948), _c28)
except Exception:
    pass
layout["LS54"] = [418, 948, 478, 977]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_07_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-10/29_text_Ls39.png
try:
    _c29 = get_crop(29, 60, 29)
    canvas.paste(_c29, (957, 946), _c29)
except Exception:
    pass
layout["Ls39"] = [957, 946, 1017, 975]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_07_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-10/30_text_LS57.png
try:
    _c30 = get_crop(30, 57, 28)
    canvas.paste(_c30, (340, 1003), _c30)
except Exception:
    pass
layout["LS57"] = [340, 1003, 397, 1031]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_07_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-10/31_text_LS36.png
try:
    _c31 = get_crop(31, 58, 28)
    canvas.paste(_c31, (1040, 1003), _c31)
except Exception:
    pass
layout["LS36"] = [1040, 1003, 1098, 1031]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_07_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-10/32_text_206.png
try:
    _c32 = get_crop(32, 48, 27)
    canvas.paste(_c32, (1119, 1006), _c32)
except Exception:
    pass
layout["206"] = [1119, 1006, 1167, 1033]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_07_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-10/33_text_SS13.png
try:
    _c33 = get_crop(33, 59, 27)
    canvas.paste(_c33, (1214, 1029), _c33)
except Exception:
    pass
layout["SS13"] = [1214, 1029, 1273, 1056]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_07_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-10/34_text_205.png
try:
    _c34 = get_crop(34, 46, 28)
    canvas.paste(_c34, (1149, 1077), _c34)
except Exception:
    pass
layout["205"] = [1149, 1077, 1195, 1105]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_07_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-10/35_text_SS11.png
try:
    _c35 = get_crop(35, 57, 27)
    canvas.paste(_c35, (1235, 1126), _c35)
except Exception:
    pass
layout["SS11"] = [1235, 1126, 1292, 1153]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_07_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-10/36_text_SS10.png
try:
    _c36 = get_crop(36, 60, 30)
    canvas.paste(_c36, (1239, 1172), _c36)
except Exception:
    pass
layout["SS10"] = [1239, 1172, 1299, 1202]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_07_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-10/37_text_218.png
try:
    _c37 = get_crop(37, 45, 28)
    canvas.paste(_c37, (220, 1225), _c37)
except Exception:
    pass
layout["218"] = [220, 1225, 265, 1253]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_07_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-10/38_text_SS9.png
try:
    _c38 = get_crop(38, 48, 27)
    canvas.paste(_c38, (1246, 1219), _c38)
except Exception:
    pass
layout["SS9"] = [1246, 1219, 1294, 1246]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_07_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-10/39_text_SS8.png
try:
    _c39 = get_crop(39, 48, 29)
    canvas.paste(_c39, (1242, 1265), _c39)
except Exception:
    pass
layout["SS8"] = [1242, 1265, 1290, 1294]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_07_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-10/40_text_203.png
try:
    _c40 = get_crop(40, 46, 27)
    canvas.paste(_c40, (1149, 1313), _c40)
except Exception:
    pass
layout["203"] = [1149, 1313, 1195, 1340]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_07_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-10/41_text_SS6.png
try:
    _c41 = get_crop(41, 45, 27)
    canvas.paste(_c41, (1221, 1362), _c41)
except Exception:
    pass
layout["SS6"] = [1221, 1362, 1266, 1389]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_07_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-10/42_text_LS24.png
try:
    _c42 = get_crop(42, 58, 27)
    canvas.paste(_c42, (1040, 1387), _c42)
except Exception:
    pass
layout["LS24"] = [1040, 1387, 1098, 1414]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_07_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-10/43_text_LSS.png
try:
    _c43 = get_crop(43, 50, 33)
    canvas.paste(_c43, (423, 1440), _c43)
except Exception:
    pass
layout["LSS"] = [423, 1440, 473, 1473]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_07_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-10/44_text_SS4.png
try:
    _c44 = get_crop(44, 48, 27)
    canvas.paste(_c44, (1179, 1457), _c44)
except Exception:
    pass
layout["SS4"] = [1179, 1457, 1227, 1484]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_07_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-10/45_text_Ls7.png
try:
    _c45 = get_crop(45, 44, 29)
    canvas.paste(_c45, (485, 1464), _c45)
except Exception:
    pass
layout["Ls7"] = [485, 1464, 529, 1493]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_07_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-10/46_text_LS19.png
try:
    _c46 = get_crop(46, 60, 29)
    canvas.paste(_c46, (899, 1466), _c46)
except Exception:
    pass
layout["LS19"] = [899, 1466, 959, 1495]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_07_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-10/47_text_LS12.png
try:
    _c47 = get_crop(47, 59, 29)
    canvas.paste(_c47, (652, 1494), _c47)
except Exception:
    pass
layout["LS12"] = [652, 1494, 711, 1523]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_07_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-10/48_text_1S15_LS1Z.png
try:
    _c48 = get_crop(48, 134, 41)
    canvas.paste(_c48, (760, 1479), _c48)
except Exception:
    pass
layout["1S15_LS1Z"] = [760, 1479, 894, 1520]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_07_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-10/49_text_SS3.png
try:
    _c49 = get_crop(49, 48, 27)
    canvas.paste(_c49, (1154, 1501), _c49)
except Exception:
    pass
layout["SS3"] = [1154, 1501, 1202, 1528]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_07_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-10/50_text_222.png
try:
    _c50 = get_crop(50, 50, 36)
    canvas.paste(_c50, (491, 1530), _c50)
except Exception:
    pass
layout["222"] = [491, 1530, 541, 1566]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_07_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-10/51_text_MEDIA.png
try:
    _c51 = get_crop(51, 60, 20)
    canvas.paste(_c51, (690, 1539), _c51)
except Exception:
    pass
layout["MEDIA"] = [690, 1539, 750, 1559]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_07_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-10/52_text_226.png
try:
    _c52 = get_crop(52, 48, 30)
    canvas.paste(_c52, (895, 1528), _c52)
except Exception:
    pass
layout["226"] = [895, 1528, 943, 1558]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_07_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-10/53_text_225.png
try:
    _c53 = get_crop(53, 46, 27)
    canvas.paste(_c53, (809, 1547), _c53)
except Exception:
    pass
layout["225"] = [809, 1547, 855, 1574]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_07_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-10/54_text_SS1.png
try:
    _c54 = get_crop(54, 46, 30)
    canvas.paste(_c54, (1098, 1572), _c54)
except Exception:
    pass
layout["SS1"] = [1098, 1572, 1144, 1602]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_07_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-10/55_text_Sort_by.png
try:
    _c55 = get_crop(55, 188, 68)
    canvas.paste(_c55, (626, 1740), _c55)
except Exception:
    pass
layout["Sort_by"] = [626, 1740, 814, 1808]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_07_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-10/56_text_Best_Seats.png
try:
    _c56 = get_crop(56, 269, 55)
    canvas.paste(_c56, (118, 2703), _c56)
except Exception:
    pass
layout["Best_Seats"] = [118, 2703, 387, 2758]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_07_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-10/57_text_Get_close_to_the_action_with_the_best_se.png
try:
    _c57 = get_crop(57, 1320, 267)
    canvas.paste(_c57, (60, 2633), _c57)
except Exception:
    pass
layout["Get_close_to_the_action_w"] = [60, 2633, 1380, 2900]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_07_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-10/58_text_JO8WC.png
try:
    _c58 = get_crop(58, 87, 41)
    canvas.paste(_c58, (552, 921), _c58)
except Exception:
    pass
layout["JO8WC"] = [552, 921, 639, 962]
