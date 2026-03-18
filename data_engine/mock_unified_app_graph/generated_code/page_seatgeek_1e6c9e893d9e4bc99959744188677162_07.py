# page_id: page_seatgeek_1e6c9e893d9e4bc99959744188677162_07
# screenshot: 2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-10.png
# step_index: 7/8
# task: Open SeatGeek. Search "Radio City Music Hall" and then add the venue to favorite. Who are the performers of the top recommended event?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Fill overall background with the app's pale bluish-gray canvas color
draw.rectangle([(0, 0), (1440, 2960)], fill="#eef2f4")

# Status bar area at top (~50-80px)
status_h = 80
draw.rectangle([(0, 0), (1440, status_h)], fill="#d7dbdd")

# Subtle bottom divider under status bar
draw.line([(0, status_h), (1440, status_h)], fill="#e6eaec", width=1)

# Header / toolbar background (rounded pill) behind the title and back button area
header_left, header_top = 48, 120
header_right, header_bottom = header_left + 1344, header_top + 156
header_radius = 78
draw.rounded_rectangle(
    [(header_left, header_top), (header_right, header_bottom)],
    radius=header_radius,
    fill="#ffffff",
    outline="#e8eaec",
    width=2
)

# Thin vertical divider inside the header pill (to the right area for info icon)
divider_x = header_right - 112  # approximate divider location inside pill
draw.line(
    [(divider_x, header_top + 18), (divider_x, header_bottom - 18)],
    fill="#edf0f1",
    width=2
)

# Subtle shadow line beneath header pill to give separation
shadow_y = header_bottom + 8
draw.line([(48, shadow_y), (1392, shadow_y)], fill="#e9ecef", width=1)

# Separator line under filter chips area (chips themselves will be pasted on top)
filters_sep_y = 360
draw.line([(24, filters_sep_y), (1416, filters_sep_y)], fill="#eceff0", width=1)

# Main seating-map content background block (slightly darker rounded panel)
map_left, map_top = 120, 420
map_right, map_bottom = 1320, 1740
map_radius = 32
draw.rounded_rectangle(
    [(map_left, map_top), (map_right, map_bottom)],
    radius=map_radius,
    fill="#f6f8f9",
    outline="#dde1e3",
    width=2
)

# Stage background rectangle at the top of the seating-map panel
stage_w, stage_h = 440, 96
stage_cx = (map_left + map_right) // 2
stage_box = [
    stage_cx - stage_w // 2,
    map_top + 24,
    stage_cx + stage_w // 2,
    map_top + 24 + stage_h
]
draw.rectangle(stage_box, fill="#5f6467", outline="#4f5356", width=3)

# Tier separators (soft rounded arcs) to hint at seating tiers — subtle outlines only
tier_colors = {"outline": "#d8dcde", "fill": "#f6f8f9"}
tier_boxes = [
    (map_left + 60, map_top + 160, map_right - 60, map_top + 360),
    (map_left + 120, map_top + 420, map_right - 120, map_top + 620),
    (map_left + 160, map_top + 700, map_right - 160, map_top + 900),
    (map_left + 200, map_top + 980, map_right - 200, map_top + 1160),
]
for b in tier_boxes:
    draw.rounded_rectangle([ (b[0], b[1]), (b[2], b[3]) ],
                           radius=48,
                           fill=tier_colors["fill"],
                           outline=tier_colors["outline"],
                           width=3)

# Light central vertical guide (very subtle) to suggest centerline of map
draw.line([(stage_cx, map_top + 24), (stage_cx, map_bottom - 24)], fill="#eef1f2", width=2)

# Large white card for listings at the bottom with rounded corners and subtle shadow
list_card_top = 1960
list_card_left = 24
list_card_right = 1416
list_card_bottom = 2940
list_card_radius = 40

# Drop shadow for the listings card (simulated with a soft gray rounded rect behind)
shadow_offset = 10
draw.rounded_rectangle(
    [
        (list_card_left, list_card_top + shadow_offset),
        (list_card_right, list_card_bottom + shadow_offset)
    ],
    radius=list_card_radius,
    fill="#eaedf0"
)

# Main listings card
draw.rounded_rectangle(
    [(list_card_left, list_card_top), (list_card_right, list_card_bottom)],
    radius=list_card_radius,
    fill="#ffffff",
    outline="#e9ecef",
    width=2
)

# Top hairline divider of the listings card
draw.line(
    [(list_card_left + 16, list_card_top + 100), (list_card_right - 16, list_card_top + 100)],
    fill="#f0f2f3",
    width=1
)

# Small separator between listings header area and list content (subtle)
draw.line(
    [(list_card_left + 16, list_card_top + 200), (list_card_right - 16, list_card_top + 200)],
    fill="#f4f6f7",
    width=1
)

# Bottom edge subtle fade line to anchor the page
draw.line([(0, 2958), (1440, 2958)], fill="#e9ecef", width=2)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_07_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-10/00_icon_Include_fees.png
try:
    _c0 = get_crop(0, 335, 108)
    canvas.paste(_c0, (544, 312), _c0)
except Exception:
    pass
layout["Include_fees"] = [544, 312, 879, 420]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_07_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-10/01_icon_VIP.png
try:
    _c1 = get_crop(1, 221, 108)
    canvas.paste(_c1, (915, 312), _c1)
except Exception:
    pass
layout["VIP"] = [915, 312, 1136, 420]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_07_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-10/02_icon_Quantity.png
try:
    _c2 = get_crop(2, 268, 108)
    canvas.paste(_c2, (240, 312), _c2)
except Exception:
    pass
layout["Quantity"] = [240, 312, 508, 420]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_07_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-10/03_icon_9.1.png
try:
    _c3 = get_crop(3, 1440, 455)
    canvas.paste(_c3, (0, 2355), _c3)
except Exception:
    pass
layout["9.1"] = [0, 2355, 1440, 2810]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_07_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-10/04_icon_Tit.png
try:
    _c4 = get_crop(4, 156, 108)
    canvas.paste(_c4, (48, 312), _c4)
except Exception:
    pass
layout["Tit"] = [48, 312, 204, 420]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_07_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-10/05_icon_Best_seats.png
try:
    _c5 = get_crop(5, 268, 108)
    canvas.paste(_c5, (1172, 312), _c5)
except Exception:
    pass
layout["Best_seats"] = [1172, 312, 1440, 420]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_07_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-10/06_icon_Include_fees.png
try:
    _c6 = get_crop(6, 1344, 156)
    canvas.paste(_c6, (48, 120), _c6)
except Exception:
    pass
layout["Include_fees"] = [48, 120, 1392, 276]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_07_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-10/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 58, 63)
    canvas.paste(_c7, (312, 2), _c7)
except Exception:
    pass
layout["icon_7"] = [312, 2, 370, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_07_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-10/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 63, 62)
    canvas.paste(_c8, (242, 2), _c8)
except Exception:
    pass
layout["icon_8"] = [242, 2, 305, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_07_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-10/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 102, 62)
    canvas.paste(_c9, (1213, 1), _c9)
except Exception:
    pass
layout["icon_9"] = [1213, 1, 1315, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_07_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-10/10_icon_8.33_my.png
try:
    _c10 = get_crop(10, 68, 64)
    canvas.paste(_c10, (110, 0), _c10)
except Exception:
    pass
layout["8.33_my"] = [110, 0, 178, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_07_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-10/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 52, 64)
    canvas.paste(_c11, (1151, 1), _c11)
except Exception:
    pass
layout["icon_11"] = [1151, 1, 1203, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_07_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-10/12_icon_8.33_my.png
try:
    _c12 = get_crop(12, 55, 60)
    canvas.paste(_c12, (182, 2), _c12)
except Exception:
    pass
layout["8.33_my"] = [182, 2, 237, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_07_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-10/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 53, 57)
    canvas.paste(_c13, (1320, 3), _c13)
except Exception:
    pass
layout["icon_13"] = [1320, 3, 1373, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_07_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-10/14_icon_Sort_by_price.png
try:
    _c14 = get_crop(14, 455, 144)
    canvas.paste(_c14, (961, 1989), _c14)
except Exception:
    pass
layout["Sort_by_price"] = [961, 1989, 1416, 2133]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_07_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-10/15_icon_Laufey_with_Wasia_Project.png
try:
    _c15 = get_crop(15, 49, 64)
    canvas.paste(_c15, (383, 1), _c15)
except Exception:
    pass
layout["Laufey_with_Wasia_Project"] = [383, 1, 432, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_07_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-10/16_icon_Amazing_deal.png
try:
    _c16 = get_crop(16, 1440, 455)
    canvas.paste(_c16, (0, 2355), _c16)
except Exception:
    pass
layout["Amazing_deal"] = [0, 2355, 1440, 2810]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_07_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-10/17_icon_Best_seats.png
try:
    _c17 = get_crop(17, 156, 156)
    canvas.paste(_c17, (1236, 120), _c17)
except Exception:
    pass
layout["Best_seats"] = [1236, 120, 1392, 276]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_07_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-10/18_icon_8.33_my.png
try:
    _c18 = get_crop(18, 108, 65)
    canvas.paste(_c18, (3, 0), _c18)
except Exception:
    pass
layout["8.33_my"] = [3, 0, 111, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_07_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-10/19_text_STAGE.png
try:
    _c19 = get_crop(19, 42, 15)
    canvas.paste(_c19, (699, 665), _c19)
except Exception:
    pass
layout["STAGE"] = [699, 665, 741, 680]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_07_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-10/20_text_ORCH.png
try:
    _c20 = get_crop(20, 69, 29)
    canvas.paste(_c20, (432, 925), _c20)
except Exception:
    pass
layout["ORCH"] = [432, 925, 501, 954]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_07_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-10/21_text_ORCH.png
try:
    _c21 = get_crop(21, 71, 29)
    canvas.paste(_c21, (918, 925), _c21)
except Exception:
    pass
layout["ORCH"] = [918, 925, 989, 954]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_07_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-10/22_text_459_Listings.png
try:
    _c22 = get_crop(22, 333, 74)
    canvas.paste(_c22, (56, 2029), _c22)
except Exception:
    pass
layout["459_Listings"] = [56, 2029, 389, 2103]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_07_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-10/23_text_We_sell_resale_tickets._Resale_tickets_m.png
try:
    _c23 = get_crop(23, 1440, 455)
    canvas.paste(_c23, (0, 2355), _c23)
except Exception:
    pass
layout["€_We_sell_resale_tickets."] = [0, 2355, 1440, 2810]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_07_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-10/24_text_face_value.png
try:
    _c24 = get_crop(24, 218, 43)
    canvas.paste(_c24, (57, 2256), _c24)
except Exception:
    pass
layout["face_value:"] = [57, 2256, 275, 2299]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_07_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-10/25_text_S188_each.png
try:
    _c25 = get_crop(25, 262, 61)
    canvas.paste(_c25, (485, 2862), _c25)
except Exception:
    pass
layout["S188_each"] = [485, 2862, 747, 2923]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_07_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-10/26_clickable_Back.png
try:
    _c26 = get_crop(26, 156, 156)
    canvas.paste(_c26, (48, 120), _c26)
except Exception:
    pass
layout["Back"] = [48, 120, 204, 276]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_07_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-10/27_clickable_Laufey_with_Wasia_Project.png
try:
    _c27 = get_crop(27, 520, 156)
    canvas.paste(_c27, (204, 120), _c27)
except Exception:
    pass
layout["Laufey_with_Wasia_Project"] = [204, 120, 724, 276]
