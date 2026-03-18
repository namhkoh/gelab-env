# page_id: page_seatgeek_21b637d11bea46b8adb3c2efc9f03501_06
# screenshot: 2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-9.png
# step_index: 6/10
# task: Open SeatGeek and find the soonest upcoming NBA game in New York with "Nets", record the cheapest price in google keep notes.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw background and structural UI elements for the mobile UI page
# Variables available: canvas (PIL Image), draw (ImageDraw), font_sm, font_md, font_lg, font_xl

# Overall background (subtle bluish-gray)
bg_color = (238, 242, 245)  # light blue-gray
draw.rectangle([0, 0, 1440, 2960], fill=bg_color)

# Status bar (top thin band)
status_h = 64
status_color = (225, 228, 231)  # slightly darker than background
draw.rectangle([0, 0, 1440, status_h], fill=status_color)

# Thin bottom stroke under status bar
draw.line([(0, status_h), (1440, status_h)], fill=(216, 219, 222), width=1)

# Header / toolbar pill (large rounded pill under status bar)
header_margin = 48
header_top = status_h + 8
header_bottom = header_top + 156
header_rect = [header_margin, header_top, 1440 - header_margin, header_bottom]
draw.rounded_rectangle(header_rect, radius=44, fill=(255, 255, 255), outline=(226, 228, 230), width=1)

# Vertical divider inside header on the right side (background structural divider)
divider_x = header_rect[2] - 88
draw.line([(divider_x, header_top + 16), (divider_x, header_bottom - 16)], fill=(236, 238, 239), width=2)

# Small circular background on right side of header (icon background area)
right_icon_bbox = [header_rect[2] - 88 + 16, header_top + 16, header_rect[2] - 16, header_top + 136]
draw.ellipse(right_icon_bbox, fill=(255, 255, 255), outline=(226, 228, 230))

# Row of filter pills (background shapes only)
# Using the detected pill sizes/positions as guides but only drawing backgrounds
# Pill specs: (x_center,y_center,width,height) derived from detections (converted to bbox)
pills = [
    {"pos": (48, 312), "size": (156, 108)},   # leftmost small icon pill (background)
    {"pos": (240, 312), "size": (268, 108)},  # Quantity
    {"pos": (544, 312), "size": (335, 108), "selected": True},  # Include fees (selected - black)
    {"pos": (915, 312), "size": (286, 108)},  # Courtside
    {"pos": (1237, 312), "size": (203, 108)}  # Center
]

for p in pills:
    x, y = p["pos"]
    w, h = p["size"]
    # The detected pos are given as top-left in the metadata; use that directly for bbox
    left = x
    top = y - int(h/2)  # adjust to center-like vertical alignment approximated
    # But to match screenshot better, use y (detected y is center-ish); we instead align using detected top approximations:
    # Many detected entries use pos with top-left coordinates; to be safe, treat pos as left,top
    # For these entries, the given pos seems to be left coordinate. We'll use left, top = x, y - h/2 earlier,
    # but ensure pill vertical placement roughly where filter row sits.
    top = y - int(h/2)
    bbox = [left, top, left + w, top + h]
    radius = int(h / 2)
    if p.get("selected"):
        draw.rounded_rectangle(bbox, radius=radius, fill=(18, 18, 18))  # dark selected pill
    else:
        draw.rounded_rectangle(bbox, radius=radius, fill=(255, 255, 255), outline=(226, 228, 230), width=1)

# Large circular seating map card background (centered)
circle_diameter = 1000
circle_left = int((1440 - circle_diameter) / 2)
circle_top = 420
circle_bbox = [circle_left, circle_top, circle_left + circle_diameter, circle_top + circle_diameter]
# Soft outer shadow (faint)
shadow_bbox = [circle_bbox[0] + 6, circle_bbox[1] + 8, circle_bbox[2] + 6, circle_bbox[3] + 8]
draw.ellipse(shadow_bbox, fill=(226, 229, 231))
# Main white circular card
draw.ellipse(circle_bbox, fill=(255, 255, 255), outline=(200, 203, 206), width=6)
# Inner subtle ring to suggest the map container
inner_inset = 18
inner_bbox = [circle_bbox[0] + inner_inset, circle_bbox[1] + inner_inset, circle_bbox[2] - inner_inset, circle_bbox[3] - inner_inset]
draw.ellipse(inner_bbox, outline=(242, 243, 244), width=4)

# Bottom listings card (large white rounded rectangle with grab handle)
card_top = 1880
card_rect = [0, card_top, 1440, 2960]
card_radius = 40
draw.rounded_rectangle(card_rect, radius=card_radius, fill=(255, 255, 255))

# Card top shadow line
draw.line([(24, card_top + 2), (1440 - 24, card_top + 2)], fill=(220, 223, 225), width=1)

# Center grab handle on the top of the card
handle_w = 160
handle_h = 10
handle_x1 = int((1440 - handle_w) / 2)
handle_x2 = handle_x1 + handle_w
handle_y1 = card_top + 18
draw.rounded_rectangle([handle_x1, handle_y1, handle_x2, handle_y1 + handle_h], radius=6, fill=(235, 237, 239))

# Header area inside listings card (space for "679 Listings" and sort control)
header_inner_top = card_top + 36
header_inner_bottom = header_inner_top + 120
# subtle bottom divider under header area
draw.line([(24, header_inner_bottom), (1440 - 24, header_inner_bottom)], fill=(236, 238, 239), width=1)

# Sort-by rounded background on the header's right side (structural only)
sort_box_w = 360
sort_box_h = 96
sort_box_right = 1440 - 36
sort_box_left = sort_box_right - sort_box_w
sort_box_top = header_inner_top + 12
sort_box_bbox = [sort_box_left, sort_box_top, sort_box_right, sort_box_top + sort_box_h]
draw.rounded_rectangle(sort_box_bbox, radius=30, fill=(255, 255, 255), outline=(226, 228, 230), width=1)

# Listing thumbnail placeholders and separators (structural only)
# First listing thumbnail (left), rest will be separators; avoid drawing text/content
thumb_margin_left = 48
thumb_w = 336
thumb_h = 226
first_thumb_top = header_inner_bottom + 32
thumb_bbox = [thumb_margin_left, first_thumb_top, thumb_margin_left + thumb_w, first_thumb_top + thumb_h]
draw.rounded_rectangle(thumb_bbox, radius=16, fill=(38, 38, 38))  # dark image placeholder

# Second listing thumbnail further down for structural repeat
second_thumb_top = first_thumb_top + thumb_h + 84
thumb2_bbox = [thumb_margin_left, second_thumb_top, thumb_margin_left + thumb_w, second_thumb_top + thumb_h]
draw.rounded_rectangle(thumb2_bbox, radius=16, fill=(38, 38, 38))

# Horizontal separators between listing rows (structural)
sep_y1 = first_thumb_top + thumb_h + 22
sep_y2 = second_thumb_top + thumb_h + 22
for y in (sep_y1, sep_y2):
    draw.line([(24, y), (1440 - 24, y)], fill=(243, 244, 245), width=1)

# Subtle vertical divider to visually separate thumbnails from listing text area (no text drawn)
divider_x = thumb_margin_left + thumb_w + 36
draw.line([(divider_x, first_thumb_top - 8), (divider_x, first_thumb_top + thumb_h + 8)], fill=(245, 246, 247), width=1)
draw.line([(divider_x, second_thumb_top - 8), (divider_x, second_thumb_top + thumb_h + 8)], fill=(245, 246, 247), width=1)

# Finishing subtle accents: a light bottom shadow on the card to separate from app background
draw.rectangle([0, 2960 - 8, 1440, 2960], fill=(232, 234, 235))

# End of structural/background drawing.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_06_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-9/00_icon_Include_fees.png
try:
    _c0 = get_crop(0, 335, 108)
    canvas.paste(_c0, (544, 312), _c0)
except Exception:
    pass
layout["Include_fees"] = [544, 312, 879, 420]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_06_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-9/01_icon_Courtside.png
try:
    _c1 = get_crop(1, 286, 108)
    canvas.paste(_c1, (915, 312), _c1)
except Exception:
    pass
layout["Courtside"] = [915, 312, 1201, 420]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_06_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-9/02_icon_Quantity.png
try:
    _c2 = get_crop(2, 268, 108)
    canvas.paste(_c2, (240, 312), _c2)
except Exception:
    pass
layout["Quantity"] = [240, 312, 508, 420]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_06_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-9/03_icon_8.0.png
try:
    _c3 = get_crop(3, 1440, 455)
    canvas.paste(_c3, (0, 2355), _c3)
except Exception:
    pass
layout["8.0"] = [0, 2355, 1440, 2810]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_06_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-9/04_icon_Tit.png
try:
    _c4 = get_crop(4, 156, 108)
    canvas.paste(_c4, (48, 312), _c4)
except Exception:
    pass
layout["Tit"] = [48, 312, 204, 420]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_06_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-9/05_icon_Center.png
try:
    _c5 = get_crop(5, 203, 108)
    canvas.paste(_c5, (1237, 312), _c5)
except Exception:
    pass
layout["Center"] = [1237, 312, 1440, 420]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_06_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-9/06_icon_Include_fees.png
try:
    _c6 = get_crop(6, 1344, 156)
    canvas.paste(_c6, (48, 120), _c6)
except Exception:
    pass
layout["Include_fees"] = [48, 120, 1392, 276]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_06_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-9/07_icon_Great_deal.png
try:
    _c7 = get_crop(7, 1440, 455)
    canvas.paste(_c7, (0, 2355), _c7)
except Exception:
    pass
layout["Great_deal"] = [0, 2355, 1440, 2810]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_06_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-9/08_icon_GK.png
try:
    _c8 = get_crop(8, 58, 56)
    canvas.paste(_c8, (179, 5), _c8)
except Exception:
    pass
layout["GK"] = [179, 5, 237, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_06_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-9/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 53, 65)
    canvas.paste(_c9, (1151, 1), _c9)
except Exception:
    pass
layout["icon_9"] = [1151, 1, 1204, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_06_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-9/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 103, 62)
    canvas.paste(_c10, (1212, 1), _c10)
except Exception:
    pass
layout["icon_10"] = [1212, 1, 1315, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_06_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-9/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 54, 59)
    canvas.paste(_c11, (1319, 2), _c11)
except Exception:
    pass
layout["icon_11"] = [1319, 2, 1373, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_06_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-9/12_icon_Sort_by_price.png
try:
    _c12 = get_crop(12, 455, 144)
    canvas.paste(_c12, (961, 1989), _c12)
except Exception:
    pass
layout["Sort_by_price"] = [961, 1989, 1416, 2133]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_06_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-9/13_icon_Center.png
try:
    _c13 = get_crop(13, 156, 156)
    canvas.paste(_c13, (1236, 120), _c13)
except Exception:
    pass
layout["Center"] = [1236, 120, 1392, 276]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_06_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-9/14_icon_S219_each.png
try:
    _c14 = get_crop(14, 381, 106)
    canvas.paste(_c14, (53, 2854), _c14)
except Exception:
    pass
layout["S219_each"] = [53, 2854, 434, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_06_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-9/15_icon_GK.png
try:
    _c15 = get_crop(15, 53, 59)
    canvas.paste(_c15, (117, 2), _c15)
except Exception:
    pass
layout["GK"] = [117, 2, 170, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_06_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-9/16_icon_6.38.png
try:
    _c16 = get_crop(16, 132, 62)
    canvas.paste(_c16, (6, 1), _c16)
except Exception:
    pass
layout["6.38"] = [6, 1, 138, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_06_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-9/17_text_B1o.png
try:
    _c17 = get_crop(17, 53, 27)
    canvas.paste(_c17, (971, 650), _c17)
except Exception:
    pass
layout["B1o]"] = [971, 650, 1024, 677]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_06_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-9/18_text_212.png
try:
    _c18 = get_crop(18, 48, 30)
    canvas.paste(_c18, (610, 712), _c18)
except Exception:
    pass
layout["212"] = [610, 712, 658, 742]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_06_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-9/19_text_210.png
try:
    _c19 = get_crop(19, 48, 27)
    canvas.paste(_c19, (779, 712), _c19)
except Exception:
    pass
layout["210"] = [779, 712, 827, 739]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_06_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-9/20_text_213.png
try:
    _c20 = get_crop(20, 48, 29)
    canvas.paste(_c20, (506, 731), _c20)
except Exception:
    pass
layout["213"] = [506, 731, 554, 760]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_06_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-9/21_text_209.png
try:
    _c21 = get_crop(21, 45, 29)
    canvas.paste(_c21, (886, 731), _c21)
except Exception:
    pass
layout["209"] = [886, 731, 931, 760]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_06_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-9/22_text_208.png
try:
    _c22 = get_crop(22, 46, 28)
    canvas.paste(_c22, (987, 781), _c22)
except Exception:
    pass
layout["208"] = [987, 781, 1033, 809]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_06_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-9/23_text_SS18.png
try:
    _c23 = get_crop(23, 62, 28)
    canvas.paste(_c23, (1091, 818), _c23)
except Exception:
    pass
layout["SS18"] = [1091, 818, 1153, 846]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_06_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-9/24_text_S47.png
try:
    _c24 = get_crop(24, 57, 27)
    canvas.paste(_c24, (673, 897), _c24)
except Exception:
    pass
layout["[S47"] = [673, 897, 730, 924]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_06_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-9/25_text_S45__543.png
try:
    _c25 = get_crop(25, 128, 36)
    canvas.paste(_c25, (739, 898), _c25)
except Exception:
    pass
layout["[S45_[543"] = [739, 898, 867, 934]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_06_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-9/26_text_S16.png
try:
    _c26 = get_crop(26, 60, 29)
    canvas.paste(_c26, (1149, 888), _c26)
except Exception:
    pass
layout["S16"] = [1149, 888, 1209, 917]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_06_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-9/27_text_LS52.png
try:
    _c27 = get_crop(27, 62, 29)
    canvas.paste(_c27, (483, 923), _c27)
except Exception:
    pass
layout["LS52"] = [483, 923, 545, 952]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_06_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-9/28_text_LS41.png
try:
    _c28 = get_crop(28, 57, 31)
    canvas.paste(_c28, (874, 916), _c28)
except Exception:
    pass
layout["LS41"] = [874, 916, 931, 947]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_06_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-9/29_text_SS15.png
try:
    _c29 = get_crop(29, 59, 27)
    canvas.paste(_c29, (1175, 932), _c29)
except Exception:
    pass
layout["SS15"] = [1175, 932, 1234, 959]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_06_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-9/30_text_LS54.png
try:
    _c30 = get_crop(30, 60, 29)
    canvas.paste(_c30, (418, 948), _c30)
except Exception:
    pass
layout["LS54"] = [418, 948, 478, 977]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_06_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-9/31_text_Ls39.png
try:
    _c31 = get_crop(31, 60, 29)
    canvas.paste(_c31, (957, 946), _c31)
except Exception:
    pass
layout["Ls39"] = [957, 946, 1017, 975]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_06_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-9/32_text_LS57.png
try:
    _c32 = get_crop(32, 57, 28)
    canvas.paste(_c32, (340, 1003), _c32)
except Exception:
    pass
layout["LS57"] = [340, 1003, 397, 1031]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_06_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-9/33_text_LS36.png
try:
    _c33 = get_crop(33, 58, 28)
    canvas.paste(_c33, (1040, 1003), _c33)
except Exception:
    pass
layout["LS36"] = [1040, 1003, 1098, 1031]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_06_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-9/34_text_206.png
try:
    _c34 = get_crop(34, 48, 27)
    canvas.paste(_c34, (1119, 1006), _c34)
except Exception:
    pass
layout["206"] = [1119, 1006, 1167, 1033]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_06_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-9/35_text_SS13.png
try:
    _c35 = get_crop(35, 59, 29)
    canvas.paste(_c35, (1214, 1027), _c35)
except Exception:
    pass
layout["SS13"] = [1214, 1027, 1273, 1056]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_06_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-9/36_text_205.png
try:
    _c36 = get_crop(36, 46, 28)
    canvas.paste(_c36, (1149, 1077), _c36)
except Exception:
    pass
layout["205"] = [1149, 1077, 1195, 1105]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_06_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-9/37_text_SS11.png
try:
    _c37 = get_crop(37, 57, 27)
    canvas.paste(_c37, (1235, 1126), _c37)
except Exception:
    pass
layout["SS11"] = [1235, 1126, 1292, 1153]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_06_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-9/38_text_SS10.png
try:
    _c38 = get_crop(38, 60, 30)
    canvas.paste(_c38, (1239, 1172), _c38)
except Exception:
    pass
layout["SS10"] = [1239, 1172, 1299, 1202]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_06_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-9/39_text_218.png
try:
    _c39 = get_crop(39, 45, 30)
    canvas.paste(_c39, (220, 1225), _c39)
except Exception:
    pass
layout["218"] = [220, 1225, 265, 1255]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_06_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-9/40_text_sS9.png
try:
    _c40 = get_crop(40, 48, 27)
    canvas.paste(_c40, (1246, 1219), _c40)
except Exception:
    pass
layout["sS9"] = [1246, 1219, 1294, 1246]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_06_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-9/41_text_SS8.png
try:
    _c41 = get_crop(41, 48, 29)
    canvas.paste(_c41, (1242, 1265), _c41)
except Exception:
    pass
layout["SS8"] = [1242, 1265, 1290, 1294]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_06_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-9/42_text_203.png
try:
    _c42 = get_crop(42, 46, 27)
    canvas.paste(_c42, (1149, 1313), _c42)
except Exception:
    pass
layout["203"] = [1149, 1313, 1195, 1340]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_06_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-9/43_text_SS6.png
try:
    _c43 = get_crop(43, 45, 27)
    canvas.paste(_c43, (1221, 1362), _c43)
except Exception:
    pass
layout["SS6"] = [1221, 1362, 1266, 1389]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_06_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-9/44_text_LS24.png
try:
    _c44 = get_crop(44, 60, 27)
    canvas.paste(_c44, (1038, 1387), _c44)
except Exception:
    pass
layout["LS24"] = [1038, 1387, 1098, 1414]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_06_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-9/45_text_LSS.png
try:
    _c45 = get_crop(45, 52, 36)
    canvas.paste(_c45, (422, 1438), _c45)
except Exception:
    pass
layout["LSS"] = [422, 1438, 474, 1474]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_06_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-9/46_text_SS4.png
try:
    _c46 = get_crop(46, 48, 30)
    canvas.paste(_c46, (1179, 1454), _c46)
except Exception:
    pass
layout["SS4"] = [1179, 1454, 1227, 1484]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_06_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-9/47_text_Ls7.png
try:
    _c47 = get_crop(47, 44, 31)
    canvas.paste(_c47, (485, 1464), _c47)
except Exception:
    pass
layout["Ls7"] = [485, 1464, 529, 1495]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_06_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-9/48_text_LS19.png
try:
    _c48 = get_crop(48, 60, 29)
    canvas.paste(_c48, (899, 1466), _c48)
except Exception:
    pass
layout["LS19"] = [899, 1466, 959, 1495]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_06_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-9/49_text_LS12.png
try:
    _c49 = get_crop(49, 57, 29)
    canvas.paste(_c49, (654, 1494), _c49)
except Exception:
    pass
layout["LS12"] = [654, 1494, 711, 1523]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_06_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-9/50_text_LS15_LS17.png
try:
    _c50 = get_crop(50, 134, 41)
    canvas.paste(_c50, (760, 1479), _c50)
except Exception:
    pass
layout["LS15_LS17"] = [760, 1479, 894, 1520]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_06_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-9/51_text_SS3.png
try:
    _c51 = get_crop(51, 48, 27)
    canvas.paste(_c51, (1154, 1501), _c51)
except Exception:
    pass
layout["SS3"] = [1154, 1501, 1202, 1528]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_06_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-9/52_text_222.png
try:
    _c52 = get_crop(52, 50, 36)
    canvas.paste(_c52, (491, 1530), _c52)
except Exception:
    pass
layout["222"] = [491, 1530, 541, 1566]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_06_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-9/53_text_MEDIA.png
try:
    _c53 = get_crop(53, 62, 25)
    canvas.paste(_c53, (689, 1535), _c53)
except Exception:
    pass
layout["MEDIA"] = [689, 1535, 751, 1560]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_06_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-9/54_text_226.png
try:
    _c54 = get_crop(54, 48, 30)
    canvas.paste(_c54, (895, 1528), _c54)
except Exception:
    pass
layout["226"] = [895, 1528, 943, 1558]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_06_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-9/55_text_225.png
try:
    _c55 = get_crop(55, 46, 27)
    canvas.paste(_c55, (809, 1547), _c55)
except Exception:
    pass
layout["225"] = [809, 1547, 855, 1574]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_06_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-9/56_text_SS1.png
try:
    _c56 = get_crop(56, 46, 30)
    canvas.paste(_c56, (1098, 1572), _c56)
except Exception:
    pass
layout["SS1"] = [1098, 1572, 1144, 1602]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_06_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-9/57_text_224UWC.png
try:
    _c57 = get_crop(57, 101, 28)
    canvas.paste(_c57, (684, 1699), _c57)
except Exception:
    pass
layout["224UWC"] = [684, 1699, 785, 1727]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_06_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-9/58_text_324.png
try:
    _c58 = get_crop(58, 58, 27)
    canvas.paste(_c58, (411, 1739), _c58)
except Exception:
    pass
layout["[324"] = [411, 1739, 469, 1766]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_06_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-9/59_text_325.png
try:
    _c59 = get_crop(59, 58, 27)
    canvas.paste(_c59, (499, 1741), _c59)
except Exception:
    pass
layout["[325"] = [499, 1741, 557, 1768]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_06_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-9/60_text_326.png
try:
    _c60 = get_crop(60, 59, 27)
    canvas.paste(_c60, (622, 1741), _c60)
except Exception:
    pass
layout["[326"] = [622, 1741, 681, 1768]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_06_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-9/61_text_327.png
try:
    _c61 = get_crop(61, 46, 27)
    canvas.paste(_c61, (825, 1741), _c61)
except Exception:
    pass
layout["327"] = [825, 1741, 871, 1768]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_06_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-9/62_text_328.png
try:
    _c62 = get_crop(62, 48, 27)
    canvas.paste(_c62, (936, 1741), _c62)
except Exception:
    pass
layout["328"] = [936, 1741, 984, 1768]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_06_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-9/63_text_679_Listings.png
try:
    _c63 = get_crop(63, 330, 79)
    canvas.paste(_c63, (56, 2027), _c63)
except Exception:
    pass
layout["679_Listings"] = [56, 2027, 386, 2106]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_06_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-9/64_text_We_sell_resale_tickets._Resale_tickets_m.png
try:
    _c64 = get_crop(64, 1440, 455)
    canvas.paste(_c64, (0, 2355), _c64)
except Exception:
    pass
layout["€_We_sell_resale_tickets."] = [0, 2355, 1440, 2810]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_06_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-9/65_text_face_value.png
try:
    _c65 = get_crop(65, 218, 43)
    canvas.paste(_c65, (57, 2256), _c65)
except Exception:
    pass
layout["face_value:"] = [57, 2256, 275, 2299]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_06_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-9/66_text_S219_each.png
try:
    _c66 = get_crop(66, 263, 65)
    canvas.paste(_c66, (485, 2862), _c66)
except Exception:
    pass
layout["S219_each"] = [485, 2862, 748, 2927]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_06_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-9/67_text_J1O8WC.png
try:
    _c67 = get_crop(67, 87, 41)
    canvas.paste(_c67, (552, 921), _c67)
except Exception:
    pass
layout["J1O8WC"] = [552, 921, 639, 962]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_06_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-9/68_clickable_Back.png
try:
    _c68 = get_crop(68, 156, 156)
    canvas.paste(_c68, (48, 120), _c68)
except Exception:
    pass
layout["Back"] = [48, 120, 204, 276]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_06_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-9/69_clickable_Nets_at_Knicks.png
try:
    _c69 = get_crop(69, 317, 156)
    canvas.paste(_c69, (204, 120), _c69)
except Exception:
    pass
layout["Nets_at_Knicks"] = [204, 120, 521, 276]
