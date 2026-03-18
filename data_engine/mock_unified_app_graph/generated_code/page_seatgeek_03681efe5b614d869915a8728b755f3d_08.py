# page_id: page_seatgeek_03681efe5b614d869915a8728b755f3d_08
# screenshot: 2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-11.png
# step_index: 8/10
# task: Open SeatGeek. Search "Metropolitan Opera". Find the next available show. Filter by "best seats". What section are they in for the lowest price tickets?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Background fill
bg_color = (236, 239, 241)  # light bluish-gray background similar to screenshot
draw.rectangle([(0, 0), canvas.size], fill=bg_color)

# Status bar (top area)
status_h = 60
status_color = (38, 40, 42)  # dark status bar
draw.rectangle([(0, 0), (canvas.width, status_h)], fill=status_color)

# Header pill (rounded search bar area)
header_left = 40
header_right = canvas.width - 40
header_top = 64
header_height = 96
header_radius = 48

# subtle shadow for header pill (draw first, then pill on top)
shadow_color = (210, 212, 214)
shadow_offset = 6
draw.rounded_rectangle(
    [(header_left, header_top + shadow_offset), (header_right, header_top + header_height + shadow_offset)],
    radius=header_radius,
    fill=shadow_color
)
# white pill
pill_color = (255, 255, 255)
draw.rounded_rectangle(
    [(header_left, header_top), (header_right, header_top + header_height)],
    radius=header_radius,
    fill=pill_color,
    outline=(225,225,225)
)

# thin divider line below header area
divider_y = header_top + header_height + 18
draw.line([(20, divider_y), (canvas.width - 20, divider_y)], fill=(220, 221, 223), width=1)

# Large content area background for seating map region
map_area_top = divider_y + 16
map_area_bottom = 1720
map_area_margin = 80
map_bg_color = (239, 241, 243)  # slightly different pale shade
draw.rounded_rectangle(
    [(map_area_margin, map_area_top), (canvas.width - map_area_margin, map_area_bottom)],
    radius=24,
    fill=map_bg_color,
    outline=(222, 223, 225)
)

# Add subtle horizontal separators inside the map background to create visual grouping (no text/icons)
sep_x0 = map_area_margin + 24
sep_x1 = canvas.width - map_area_margin - 24
y_positions = [
    map_area_top + 200,
    map_area_top + 460,
    map_area_top + 720,
    map_area_top + 980
]
for y in y_positions:
    draw.line([(sep_x0, y), (sep_x1, y)], fill=(232, 233, 235), width=1)

# Thin vertical center guide (soft) that won't conflict with pasted elements
draw.line([(canvas.width//2, map_area_top + 20), (canvas.width//2, map_area_bottom - 20)], fill=(235,236,237), width=1)

# Modal sheet (bottom area) "Sort by" panel
modal_top = 1760
modal_radius = 28

# shadow above modal
shadow_h = 12
draw.rectangle([(0, modal_top - shadow_h), (canvas.width, modal_top)], fill=(210, 212, 214))

# white modal rounded rectangle
modal_color = (255, 255, 255)
draw.rounded_rectangle(
    [(0, modal_top), (canvas.width, canvas.height)],
    radius=modal_radius,
    fill=modal_color
)

# small handle at modal top center
handle_w = 160
handle_h = 10
handle_x0 = (canvas.width - handle_w) // 2
handle_x1 = handle_x0 + handle_w
handle_y0 = modal_top + 18
handle_radius = handle_h // 2
draw.rounded_rectangle([(handle_x0, handle_y0), (handle_x1, handle_y0 + handle_h)], radius=handle_radius, fill=(230,231,233))

# Cards inside modal (rounded rectangles)
card_left = 40
card_right = canvas.width - 40
card_width = card_right - card_left
card_height = 160
card_spacing = 28

first_card_top = modal_top + 100
cards = []
for i in range(3):
    top = first_card_top + i * (card_height + card_spacing)
    bottom = top + card_height
    cards.append((top, bottom))

# Draw card outlines and subtle fills
card_fill = (255, 255, 255)
card_border_light = (221, 222, 224)
card_border_strong = (30, 30, 30)

# Card 0: Deal Score (light border)
draw.rounded_rectangle([(card_left, cards[0][0]), (card_right, cards[0][1])], radius=18, fill=card_fill, outline=card_border_light, width=2)

# Card 1: Price (selected style - stronger border)
draw.rounded_rectangle([(card_left, cards[1][0]), (card_right, cards[1][1])], radius=18, fill=card_fill, outline=card_border_strong, width=3)

# Card 2: Best Seats (light border)
draw.rounded_rectangle([(card_left, cards[2][0]), (card_right, cards[2][1])], radius=18, fill=card_fill, outline=card_border_light, width=2)

# Divider lines between cards (subtle)
for i in range(3):
    top, bottom = cards[i]
    # inner subtle divider near top to suggest header area inside each card
    draw.line([(card_left + 20, top + 72), (card_right - 20, top + 72)], fill=(245,246,247), width=1)

# Bottom safe area shadow/line
draw.line([(0, canvas.height - 60), (canvas.width, canvas.height - 60)], fill=(238,239,240), width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_08_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-11/00_icon_We_rate_each_ticket_based_on_location_pr.png
try:
    _c0 = get_crop(0, 1320, 267)
    canvas.paste(_c0, (60, 2318), _c0)
except Exception:
    pass
layout["We_rate_each_ticket_based"] = [60, 2318, 1380, 2585]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_08_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-11/01_icon_Include.png
try:
    _c1 = get_crop(1, 341, 118)
    canvas.paste(_c1, (537, 309), _c1)
except Exception:
    pass
layout["Include"] = [537, 309, 878, 427]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_08_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-11/02_icon_Best_seats.png
try:
    _c2 = get_crop(2, 309, 118)
    canvas.paste(_c2, (910, 309), _c2)
except Exception:
    pass
layout["Best_seats"] = [910, 309, 1219, 427]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_08_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-11/03_icon_Quantity.png
try:
    _c3 = get_crop(3, 279, 120)
    canvas.paste(_c3, (231, 307), _c3)
except Exception:
    pass
layout["Quantity"] = [231, 307, 510, 427]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_08_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-11/04_icon_Tit.png
try:
    _c4 = get_crop(4, 169, 121)
    canvas.paste(_c4, (37, 308), _c4)
except Exception:
    pass
layout["Tit"] = [37, 308, 206, 429]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_08_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-11/05_icon_Low_pri.png
try:
    _c5 = get_crop(5, 193, 119)
    canvas.paste(_c5, (1247, 308), _c5)
except Exception:
    pass
layout["Low_pri"] = [1247, 308, 1440, 427]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_08_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-11/06_icon_We_rate_each_ticket_based_on_location_pr.png
try:
    _c6 = get_crop(6, 1320, 329)
    canvas.paste(_c6, (60, 1941), _c6)
except Exception:
    pass
layout["We_rate_each_ticket_based"] = [60, 1941, 1380, 2270]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_08_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-11/07_icon_Include.png
try:
    _c7 = get_crop(7, 1353, 166)
    canvas.paste(_c7, (40, 118), _c7)
except Exception:
    pass
layout["Include"] = [40, 118, 1393, 284]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_08_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-11/08_icon_El_Nino.png
try:
    _c8 = get_crop(8, 64, 63)
    canvas.paste(_c8, (241, 2), _c8)
except Exception:
    pass
layout["El_Nino"] = [241, 2, 305, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_08_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-11/09_icon_7.58_my.png
try:
    _c9 = get_crop(9, 68, 64)
    canvas.paste(_c9, (110, 0), _c9)
except Exception:
    pass
layout["7.58_my"] = [110, 0, 178, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_08_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-11/10_icon_El_Nino.png
try:
    _c10 = get_crop(10, 60, 63)
    canvas.paste(_c10, (311, 2), _c10)
except Exception:
    pass
layout["El_Nino"] = [311, 2, 371, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_08_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-11/11_icon_7.58_my.png
try:
    _c11 = get_crop(11, 54, 63)
    canvas.paste(_c11, (182, 1), _c11)
except Exception:
    pass
layout["7.58_my"] = [182, 1, 236, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_08_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-11/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 53, 56)
    canvas.paste(_c12, (1319, 4), _c12)
except Exception:
    pass
layout["icon_12"] = [1319, 4, 1372, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_08_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-11/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 52, 65)
    canvas.paste(_c13, (1152, 1), _c13)
except Exception:
    pass
layout["icon_13"] = [1152, 1, 1204, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_08_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-11/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 104, 61)
    canvas.paste(_c14, (1213, 2), _c14)
except Exception:
    pass
layout["icon_14"] = [1213, 2, 1317, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_08_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-11/15_icon_New_York.png
try:
    _c15 = get_crop(15, 47, 65)
    canvas.paste(_c15, (384, 1), _c15)
except Exception:
    pass
layout["New_York"] = [384, 1, 431, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_08_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-11/16_icon_Sort_by.png
try:
    _c16 = get_crop(16, 1320, 329)
    canvas.paste(_c16, (60, 1941), _c16)
except Exception:
    pass
layout["Sort_by"] = [60, 1941, 1380, 2270]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_08_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-11/17_icon_Low_pri.png
try:
    _c17 = get_crop(17, 96, 112)
    canvas.paste(_c17, (1258, 146), _c17)
except Exception:
    pass
layout["Low_pri"] = [1258, 146, 1354, 258]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_08_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-11/18_text_STAGE.png
try:
    _c18 = get_crop(18, 42, 16)
    canvas.paste(_c18, (470, 611), _c18)
except Exception:
    pass
layout["STAGE"] = [470, 611, 512, 627]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_08_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-11/19_text_ORCHESTRA_PIT.png
try:
    _c19 = get_crop(19, 136, 25)
    canvas.paste(_c19, (421, 684), _c19)
except Exception:
    pass
layout["ORCHESTRA_PIT"] = [421, 684, 557, 709]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_08_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-11/20_text_ORCH_L.png
try:
    _c20 = get_crop(20, 87, 27)
    canvas.paste(_c20, (354, 821), _c20)
except Exception:
    pass
layout["ORCH_L"] = [354, 821, 441, 848]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_08_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-11/21_text_49.png
try:
    _c21 = get_crop(21, 36, 27)
    canvas.paste(_c21, (844, 920), _c21)
except Exception:
    pass
layout["49"] = [844, 920, 880, 947]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_08_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-11/22_text_18.png
try:
    _c22 = get_crop(22, 34, 29)
    canvas.paste(_c22, (1057, 911), _c22)
except Exception:
    pass
layout["18"] = [1057, 911, 1091, 940]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_08_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-11/23_text_27.png
try:
    _c23 = get_crop(23, 36, 29)
    canvas.paste(_c23, (923, 941), _c23)
except Exception:
    pass
layout["27"] = [923, 941, 959, 970]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_08_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-11/24_text_26.png
try:
    _c24 = get_crop(24, 34, 27)
    canvas.paste(_c24, (983, 939), _c24)
except Exception:
    pass
layout["26"] = [983, 939, 1017, 966]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_08_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-11/25_text_LEVEL.png
try:
    _c25 = get_crop(25, 57, 25)
    canvas.paste(_c25, (402, 1017), _c25)
except Exception:
    pass
layout["LEVEL"] = [402, 1017, 459, 1042]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_08_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-11/26_text_ORCHESTRA.png
try:
    _c26 = get_crop(26, 106, 25)
    canvas.paste(_c26, (469, 1017), _c26)
except Exception:
    pass
layout["ORCHESTRA"] = [469, 1017, 575, 1042]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_08_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-11/27_text_LEVEL_2_PARTERRE.png
try:
    _c27 = get_crop(27, 159, 25)
    canvas.paste(_c27, (879, 1017), _c27)
except Exception:
    pass
layout["LEVEL_2_PARTERRE"] = [879, 1017, 1038, 1042]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_08_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-11/28_text_33.png
try:
    _c28 = get_crop(28, 31, 27)
    canvas.paste(_c28, (331, 1149), _c28)
except Exception:
    pass
layout["33"] = [331, 1149, 362, 1176]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_08_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-11/29_text_ROOM.png
try:
    _c29 = get_crop(29, 64, 27)
    canvas.paste(_c29, (950, 1367), _c29)
except Exception:
    pass
layout["ROOM"] = [950, 1367, 1014, 1394]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_08_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-11/30_text_Best_Seats.png
try:
    _c30 = get_crop(30, 269, 55)
    canvas.paste(_c30, (118, 2703), _c30)
except Exception:
    pass
layout["Best_Seats"] = [118, 2703, 387, 2758]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_08_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-11/31_text_Get_close_to_the_action_with_the_best_se.png
try:
    _c31 = get_crop(31, 1320, 267)
    canvas.paste(_c31, (60, 2633), _c31)
except Exception:
    pass
layout["Get_close_to_the_action_w"] = [60, 2633, 1380, 2900]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_08_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-11/32_text_STANDING.png
try:
    _c32 = get_crop(32, 102, 46)
    canvas.paste(_c32, (539, 933), _c32)
except Exception:
    pass
layout["STANDING"] = [539, 933, 641, 979]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_08_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-11/33_text_STANDING.png
try:
    _c33 = get_crop(33, 104, 46)
    canvas.paste(_c33, (338, 934), _c33)
except Exception:
    pass
layout["STANDING"] = [338, 934, 442, 980]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_08_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-11/34_text_STANDING.png
try:
    _c34 = get_crop(34, 101, 66)
    canvas.paste(_c34, (790, 1300), _c34)
except Exception:
    pass
layout["STANDING"] = [790, 1300, 891, 1366]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_08_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-11/35_text_STANDING.png
try:
    _c35 = get_crop(35, 103, 60)
    canvas.paste(_c35, (322, 1317), _c35)
except Exception:
    pass
layout["STANDING"] = [322, 1317, 425, 1377]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_08_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-11/36_text_STANDING.png
try:
    _c36 = get_crop(36, 104, 56)
    canvas.paste(_c36, (554, 1319), _c36)
except Exception:
    pass
layout["~STANDING"] = [554, 1319, 658, 1375]
