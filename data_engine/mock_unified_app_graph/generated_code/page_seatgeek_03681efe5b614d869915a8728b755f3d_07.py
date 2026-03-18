# page_id: page_seatgeek_03681efe5b614d869915a8728b755f3d_07
# screenshot: 2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-10.png
# step_index: 7/10
# task: Open SeatGeek. Search "Metropolitan Opera". Find the next available show. Filter by "best seats". What section are they in for the lowest price tickets?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Top-level background
draw.rectangle([(0, 0), (1440, 2960)], fill="#f2f4f4")

# Status bar area (top ~50px)
status_h = 50
draw.rectangle([(0, 0), (1440, status_h)], fill="#e6e7e8")

# Subtle bottom border under status bar
draw.line([(0, status_h), (1440, status_h)], fill="#d1d3d4", width=1)

# Header / toolbar container (rounded pill-style background behind title area)
header_margin_x = 48
header_top = status_h + 18
header_bottom = header_top + 120
draw.rounded_rectangle(
    [(header_margin_x, header_top), (1440 - header_margin_x, header_bottom)],
    radius=40, fill="#ffffff", outline="#dbdbdd", width=1
)

# Small divider under header (separates header and filters)
divider_y = header_bottom + 18
draw.line([(32, divider_y), (1408, divider_y)], fill="#e1e2e3", width=1)

# Filter area background zone (subtle group background behind the pill filters)
filters_top = divider_y + 14
filters_bottom = filters_top + 120
# A faint transparent-looking strip -- draw as very light fill
draw.rectangle([(0, filters_top), (1440, filters_bottom)], fill="#f6f7f8")

# Main seating map card (rounded rectangle centered)
map_margin_x = 160
map_top = filters_bottom + 20
map_bottom = map_top + 1180
draw.rounded_rectangle(
    [(map_margin_x, map_top), (1440 - map_margin_x, map_bottom)],
    radius=28, fill="#ffffff", outline="#d7d8da", width=2
)

# Inner subtle backdrop for maps to suggest a panel area (slightly off-white)
inner_pad = 22
draw.rounded_rectangle(
    [(map_margin_x + inner_pad, map_top + inner_pad),
     (1440 - map_margin_x - inner_pad, map_bottom - inner_pad)],
    radius=22, fill="#f4f6f7"
)

# Thin separators between the rows of map thumbnails (visual guides only)
# There are roughly three rows of maps; draw faint horizontal separator lines
sep1 = map_top + 320
sep2 = map_top + 680
draw.line([(map_margin_x + 12, sep1), (1440 - map_margin_x - 12, sep1)], fill="#e6e7e8", width=1)
draw.line([(map_margin_x + 12, sep2), (1440 - map_margin_x - 12, sep2)], fill="#e6e7e8", width=1)

# Divider between map area and listings section
listings_top = map_bottom + 18
draw.line([(24, listings_top), (1416, listings_top)], fill="#dcdedf", width=1)

# Listings container background (white rounded rectangle card)
list_card_top = listings_top + 16
list_card_bottom = 2960 - 160
draw.rounded_rectangle(
    [(16, list_card_top), (1440 - 16, list_card_bottom)],
    radius=24, fill="#ffffff", outline="#e0e1e2", width=1
)

# A subtle header area inside listings card for "16 Listings" and sort control
list_header_h = 120
draw.rectangle(
    [(32, list_card_top), (1440 - 32, list_card_top + list_header_h)],
    fill="#ffffff"
)
# Divider under list header
draw.line([(32, list_card_top + list_header_h), (1440 - 32, list_card_top + list_header_h)], fill="#ececee", width=1)

# Background blocks for each listing row (to give structure where detected listing thumbnails will be pasted)
# We'll draw three placeholder row backgrounds (only background shapes, not icons/text)
row_height = 260
row_gap = 28
first_row_top = list_card_top + list_header_h + 20
for i in range(3):
    top = first_row_top + i * (row_height + row_gap)
    left = 32
    right = 1440 - 32
    bottom = top + row_height
    # light rounded rectangle per listing
    draw.rounded_rectangle([(left, top), (right, bottom)], radius=16, fill="#fafbfc", outline="#ececee", width=1)

    # subtle separator under each row
    draw.line([(left + 14, bottom + 8), (right - 14, bottom + 8)], fill="#efeff0", width=1)

# Sticky bottom info bar background (where price summary appears)
bottom_bar_top = 2355  # align with detected sticky overlay region top
bottom_bar_bottom = 2960
draw.rectangle([(0, bottom_bar_top), (1440, bottom_bar_bottom)], fill="#ffffff")
# subtle top border for bottom bar
draw.line([(0, bottom_bar_top), (1440, bottom_bar_top)], fill="#e1e2e3", width=1)

# Final subtle vignette shadow under main map card to lift it slightly
shadow_top = map_bottom + 2
for i, alpha in enumerate([0, 1, 2, 3, 4]):
    y = shadow_top + i
    draw.line([(map_margin_x + 6, y), (1440 - map_margin_x - 6, y)], fill="#e9eaeb", width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_07_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-10/00_icon_Include.png
try:
    _c0 = get_crop(0, 335, 108)
    canvas.paste(_c0, (544, 312), _c0)
except Exception:
    pass
layout["Include"] = [544, 312, 879, 420]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_07_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-10/01_icon_Best_seats.png
try:
    _c1 = get_crop(1, 303, 108)
    canvas.paste(_c1, (915, 312), _c1)
except Exception:
    pass
layout["Best_seats"] = [915, 312, 1218, 420]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_07_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-10/02_icon_Quantity.png
try:
    _c2 = get_crop(2, 268, 108)
    canvas.paste(_c2, (240, 312), _c2)
except Exception:
    pass
layout["Quantity"] = [240, 312, 508, 420]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_07_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-10/03_icon_Tit.png
try:
    _c3 = get_crop(3, 156, 108)
    canvas.paste(_c3, (48, 312), _c3)
except Exception:
    pass
layout["Tit"] = [48, 312, 204, 420]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_07_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-10/04_icon_Low_pri.png
try:
    _c4 = get_crop(4, 186, 108)
    canvas.paste(_c4, (1254, 312), _c4)
except Exception:
    pass
layout["Low_pri"] = [1254, 312, 1440, 420]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_07_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-10/05_icon_0y.png
try:
    _c5 = get_crop(5, 1440, 455)
    canvas.paste(_c5, (0, 2355), _c5)
except Exception:
    pass
layout["0y"] = [0, 2355, 1440, 2810]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_07_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-10/06_icon_Include.png
try:
    _c6 = get_crop(6, 1344, 156)
    canvas.paste(_c6, (48, 120), _c6)
except Exception:
    pass
layout["Include"] = [48, 120, 1392, 276]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_07_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-10/07_icon_El_Nino.png
try:
    _c7 = get_crop(7, 56, 61)
    canvas.paste(_c7, (313, 3), _c7)
except Exception:
    pass
layout["El_Nino"] = [313, 3, 369, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_07_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-10/08_icon_7.57_Wy.png
try:
    _c8 = get_crop(8, 70, 64)
    canvas.paste(_c8, (109, 0), _c8)
except Exception:
    pass
layout["7.57_Wy"] = [109, 0, 179, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_07_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-10/09_icon_El_Nino.png
try:
    _c9 = get_crop(9, 61, 61)
    canvas.paste(_c9, (243, 2), _c9)
except Exception:
    pass
layout["El_Nino"] = [243, 2, 304, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_07_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-10/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 51, 65)
    canvas.paste(_c10, (1152, 1), _c10)
except Exception:
    pass
layout["icon_10"] = [1152, 1, 1203, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_07_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-10/11_icon_7.57_Wy.png
try:
    _c11 = get_crop(11, 55, 59)
    canvas.paste(_c11, (182, 2), _c11)
except Exception:
    pass
layout["7.57_Wy"] = [182, 2, 237, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_07_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-10/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 102, 63)
    canvas.paste(_c12, (1213, 1), _c12)
except Exception:
    pass
layout["icon_12"] = [1213, 1, 1315, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_07_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-10/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 52, 57)
    canvas.paste(_c13, (1321, 3), _c13)
except Exception:
    pass
layout["icon_13"] = [1321, 3, 1373, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_07_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-10/14_icon_New_York.png
try:
    _c14 = get_crop(14, 49, 63)
    canvas.paste(_c14, (383, 1), _c14)
except Exception:
    pass
layout["New_York"] = [383, 1, 432, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_07_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-10/15_icon_Amazing_deal.png
try:
    _c15 = get_crop(15, 1440, 455)
    canvas.paste(_c15, (0, 2355), _c15)
except Exception:
    pass
layout["Amazing_deal"] = [0, 2355, 1440, 2810]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_07_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-10/16_icon_Sort_by_price.png
try:
    _c16 = get_crop(16, 455, 144)
    canvas.paste(_c16, (961, 1989), _c16)
except Exception:
    pass
layout["Sort_by_price"] = [961, 1989, 1416, 2133]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_07_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-10/17_icon_Low_pri.png
try:
    _c17 = get_crop(17, 156, 156)
    canvas.paste(_c17, (1236, 120), _c17)
except Exception:
    pass
layout["Low_pri"] = [1236, 120, 1392, 276]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_07_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-10/18_icon_0y.png
try:
    _c18 = get_crop(18, 382, 106)
    canvas.paste(_c18, (52, 2854), _c18)
except Exception:
    pass
layout["0y"] = [52, 2854, 434, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_07_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-10/19_icon_7.57_Wy.png
try:
    _c19 = get_crop(19, 99, 64)
    canvas.paste(_c19, (6, 0), _c19)
except Exception:
    pass
layout["7.57_Wy"] = [6, 0, 105, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_07_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-10/20_text_STAGE.png
try:
    _c20 = get_crop(20, 42, 16)
    canvas.paste(_c20, (470, 611), _c20)
except Exception:
    pass
layout["STAGE"] = [470, 611, 512, 627]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_07_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-10/21_text_ORCHESTRA_PIT.png
try:
    _c21 = get_crop(21, 136, 25)
    canvas.paste(_c21, (421, 684), _c21)
except Exception:
    pass
layout["ORCHESTRA_PIT"] = [421, 684, 557, 709]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_07_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-10/22_text_ORCH_L.png
try:
    _c22 = get_crop(22, 87, 27)
    canvas.paste(_c22, (354, 821), _c22)
except Exception:
    pass
layout["ORCH_L"] = [354, 821, 441, 848]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_07_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-10/23_text_49.png
try:
    _c23 = get_crop(23, 36, 30)
    canvas.paste(_c23, (844, 920), _c23)
except Exception:
    pass
layout["49"] = [844, 920, 880, 950]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_07_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-10/24_text_18.png
try:
    _c24 = get_crop(24, 34, 29)
    canvas.paste(_c24, (1057, 911), _c24)
except Exception:
    pass
layout["18"] = [1057, 911, 1091, 940]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_07_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-10/25_text_27.png
try:
    _c25 = get_crop(25, 36, 27)
    canvas.paste(_c25, (923, 941), _c25)
except Exception:
    pass
layout["27"] = [923, 941, 959, 968]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_07_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-10/26_text_26.png
try:
    _c26 = get_crop(26, 34, 27)
    canvas.paste(_c26, (983, 939), _c26)
except Exception:
    pass
layout["26"] = [983, 939, 1017, 966]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_07_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-10/27_text_LEVEL_1_ORCHESTRA.png
try:
    _c27 = get_crop(27, 173, 25)
    canvas.paste(_c27, (402, 1017), _c27)
except Exception:
    pass
layout["LEVEL_1_ORCHESTRA"] = [402, 1017, 575, 1042]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_07_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-10/28_text_LEVEL_2_PARTERRE.png
try:
    _c28 = get_crop(28, 159, 25)
    canvas.paste(_c28, (879, 1017), _c28)
except Exception:
    pass
layout["LEVEL_2_PARTERRE"] = [879, 1017, 1038, 1042]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_07_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-10/29_text_30.png
try:
    _c29 = get_crop(29, 32, 27)
    canvas.paste(_c29, (615, 1126), _c29)
except Exception:
    pass
layout["30"] = [615, 1126, 647, 1153]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_07_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-10/30_text_33.png
try:
    _c30 = get_crop(30, 31, 27)
    canvas.paste(_c30, (331, 1149), _c30)
except Exception:
    pass
layout["33"] = [331, 1149, 362, 1176]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_07_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-10/31_text_ROOM.png
try:
    _c31 = get_crop(31, 64, 27)
    canvas.paste(_c31, (950, 1367), _c31)
except Exception:
    pass
layout["~ROOM"] = [950, 1367, 1014, 1394]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_07_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-10/32_text_BALC_L.png
try:
    _c32 = get_crop(32, 80, 27)
    canvas.paste(_c32, (342, 1739), _c32)
except Exception:
    pass
layout["BALC_L"] = [342, 1739, 422, 1766]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_07_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-10/33_text_BALC_R.png
try:
    _c33 = get_crop(33, 83, 27)
    canvas.paste(_c33, (557, 1739), _c33)
except Exception:
    pass
layout["BALC_R"] = [557, 1739, 640, 1766]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_07_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-10/34_text_LEVEL_5_BALCONY.png
try:
    _c34 = get_crop(34, 153, 20)
    canvas.paste(_c34, (412, 1844), _c34)
except Exception:
    pass
layout["LEVEL_5_BALCONY"] = [412, 1844, 565, 1864]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_07_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-10/35_text_16_Listings.png
try:
    _c35 = get_crop(35, 291, 81)
    canvas.paste(_c35, (54, 2024), _c35)
except Exception:
    pass
layout["16_Listings"] = [54, 2024, 345, 2105]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_07_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-10/36_text_We_sell_resale_tickets._Resale_tickets_m.png
try:
    _c36 = get_crop(36, 1440, 455)
    canvas.paste(_c36, (0, 2355), _c36)
except Exception:
    pass
layout["€_We_sell_resale_tickets."] = [0, 2355, 1440, 2810]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_07_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-10/37_text_face_value.png
try:
    _c37 = get_crop(37, 218, 43)
    canvas.paste(_c37, (57, 2256), _c37)
except Exception:
    pass
layout["face_value:"] = [57, 2256, 275, 2299]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_07_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-10/38_text_S623_each.png
try:
    _c38 = get_crop(38, 273, 66)
    canvas.paste(_c38, (485, 2862), _c38)
except Exception:
    pass
layout["S623_each"] = [485, 2862, 758, 2928]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_07_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-10/39_text_STANDING.png
try:
    _c39 = get_crop(39, 104, 46)
    canvas.paste(_c39, (538, 933), _c39)
except Exception:
    pass
layout["STANDING"] = [538, 933, 642, 979]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_07_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-10/40_text_STANDING.png
try:
    _c40 = get_crop(40, 104, 45)
    canvas.paste(_c40, (338, 935), _c40)
except Exception:
    pass
layout["STANDING"] = [338, 935, 442, 980]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_07_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-10/41_text_STANDING.png
try:
    _c41 = get_crop(41, 101, 66)
    canvas.paste(_c41, (790, 1300), _c41)
except Exception:
    pass
layout["STANDING"] = [790, 1300, 891, 1366]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_07_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-10/42_text_STANDING.png
try:
    _c42 = get_crop(42, 103, 60)
    canvas.paste(_c42, (322, 1317), _c42)
except Exception:
    pass
layout["STANDING"] = [322, 1317, 425, 1377]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_07_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-10/43_text_STANDING.png
try:
    _c43 = get_crop(43, 104, 56)
    canvas.paste(_c43, (554, 1319), _c43)
except Exception:
    pass
layout["~STANDING"] = [554, 1319, 658, 1375]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_07_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-10/44_text_STANDING.png
try:
    _c44 = get_crop(44, 103, 51)
    canvas.paste(_c44, (1001, 1765), _c44)
except Exception:
    pass
layout["~STANDING"] = [1001, 1765, 1104, 1816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_07_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-10/45_text_STANDING.png
try:
    _c45 = get_crop(45, 105, 51)
    canvas.paste(_c45, (811, 1765), _c45)
except Exception:
    pass
layout["STANDING"] = [811, 1765, 916, 1816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_07_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-10/46_clickable_Back.png
try:
    _c46 = get_crop(46, 156, 156)
    canvas.paste(_c46, (48, 120), _c46)
except Exception:
    pass
layout["Back"] = [48, 120, 204, 276]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_07_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-10/47_clickable_El_Nino_-_New_York.png
try:
    _c47 = get_crop(47, 363, 156)
    canvas.paste(_c47, (204, 120), _c47)
except Exception:
    pass
layout["El_Nino_-_New_York"] = [204, 120, 567, 276]
