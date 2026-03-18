# page_id: page_seatgeek_3297e3a54de8487d8c2b67798919ed2f_10
# screenshot: 2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-13.png
# step_index: 10/11
# task: Open SeatGeek. Search "Comedy Show in Los Angeles". Find the top recommendation. When is the event?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw background and UI structure for SeatGeek-like page
W, H = canvas.size

# Colors
bg = (242, 244, 246)        # overall page background (very light gray)
status_bg = (230, 233, 235) # status bar background
card_white = (255, 255, 255)
muted_div = (226, 229, 232) # divider / subtle shadow color
map_bg = (237, 239, 241)    # seating-map backdrop
list_card_bg = (255, 255, 255)
sep_color = (220, 223, 226)

# Fill overall background
draw.rectangle((0, 0, W, H), fill=bg)

# Status bar area at top (~0-72 px)
status_h = 72
draw.rectangle((0, 0, W, status_h), fill=status_bg)

# Header card (rounded, centered) with subtle offset "shadow"
hdr_left = 36
hdr_top = status_h + 8
hdr_right = W - 36
hdr_bottom = hdr_top + 128
hdr_radius = 40

# Shadow (slightly darker rounded rect offset)
shadow_offset = 6
draw.rounded_rectangle(
    (hdr_left, hdr_top + shadow_offset, hdr_right, hdr_bottom + shadow_offset),
    radius=hdr_radius, fill=muted_div
)
# Header background (white pill)
draw.rounded_rectangle(
    (hdr_left, hdr_top, hdr_right, hdr_bottom),
    radius=hdr_radius, fill=card_white, outline=None
)

# Thin divider below header
div_y = hdr_bottom + 16
draw.rectangle((24, div_y, W - 24, div_y + 1), fill=sep_color)

# Filters row background area (a subtle backdrop behind pill controls)
filters_top = div_y + 18
filters_bottom = filters_top + 120
# a very light rounded rect background band
draw.rounded_rectangle(
    (24, filters_top, W - 24, filters_bottom),
    radius=36, fill=bg, outline=None
)

# Large seating-map area backdrop (centered, light grey card)
map_left = 120
map_right = W - 120
map_top = filters_bottom + 30
map_bottom = map_top + 1200  # tall central area to host the image map
map_radius = 28

# Map "card" shadow simulation (offset pale border)
draw.rounded_rectangle(
    (map_left - 6, map_top + 6, map_right + 6, map_bottom + 6),
    radius=map_radius + 6, fill=muted_div
)
draw.rounded_rectangle(
    (map_left, map_top, map_right, map_bottom),
    radius=map_radius, fill=map_bg, outline=(215,218,221), width=4
)

# Subtle inner clipping area for the map (lighter center)
inner_inset = 28
draw.rounded_rectangle(
    (map_left + inner_inset, map_top + inner_inset,
     map_right - inner_inset, map_bottom - inner_inset),
    radius=18, fill=(249,250,251)
)

# Thin separator line between map and listings area
list_area_top = map_bottom + 36
draw.rectangle((24, list_area_top, W - 24, list_area_top + 1), fill=sep_color)

# Listings container (bottom card)
list_left = 20
list_top = list_area_top + 12
list_right = W - 20
list_bottom = H - 20
list_radius = 28

# Slight shadow for listings card
draw.rounded_rectangle(
    (list_left, list_top + 8, list_right, list_bottom + 8),
    radius=list_radius, fill=muted_div
)
# White listings card
draw.rounded_rectangle(
    (list_left, list_top, list_right, list_bottom),
    radius=list_radius, fill=list_card_bg, outline=(230,233,236)
)

# Divider under the "367 Listings" heading area (approximate)
heading_h = 92
heading_y = list_top + 8 + heading_h
draw.rectangle((list_left + 20, heading_y, list_right - 20, heading_y + 1), fill=sep_color)

# Draw a couple of listing row background placeholders (rounded cards)
row_height = 160
first_row_top = heading_y + 22
row_gap = 28
for i in range(3):
    r_top = first_row_top + i * (row_height + row_gap)
    r_bottom = r_top + row_height
    # card background
    draw.rounded_rectangle(
        (list_left + 24, r_top, list_right - 24, r_bottom),
        radius=16, fill=(250,251,252), outline=(235,238,241)
    )
    # light horizontal separator below each card
    draw.rectangle((list_left + 24, r_bottom + 6, list_right - 24, r_bottom + 7), fill=sep_color)

# Small top divider under status bar near left (visual cue)
draw.rectangle((0, status_h - 1, W, status_h), fill=sep_color)

# Final subtle vertical side paddings (visual)
side_pad_w = 12
draw.rectangle((0, 0, side_pad_w, H), fill=bg)
draw.rectangle((W - side_pad_w, 0, W, H), fill=bg)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_10_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-13/00_icon_Include_fees.png
try:
    _c0 = get_crop(0, 335, 108)
    canvas.paste(_c0, (544, 312), _c0)
except Exception:
    pass
layout["Include_fees"] = [544, 312, 879, 420]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_10_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-13/01_icon_Best_seats.png
try:
    _c1 = get_crop(1, 303, 108)
    canvas.paste(_c1, (915, 312), _c1)
except Exception:
    pass
layout["Best_seats"] = [915, 312, 1218, 420]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_10_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-13/02_icon_367_Listings.png
try:
    _c2 = get_crop(2, 1440, 455)
    canvas.paste(_c2, (0, 2134), _c2)
except Exception:
    pass
layout["367_Listings"] = [0, 2134, 1440, 2589]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_10_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-13/03_icon_Quantity.png
try:
    _c3 = get_crop(3, 268, 108)
    canvas.paste(_c3, (240, 312), _c3)
except Exception:
    pass
layout["Quantity"] = [240, 312, 508, 420]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_10_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-13/04_icon_Tit.png
try:
    _c4 = get_crop(4, 156, 108)
    canvas.paste(_c4, (48, 312), _c4)
except Exception:
    pass
layout["Tit"] = [48, 312, 204, 420]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_10_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-13/05_icon_9.8.png
try:
    _c5 = get_crop(5, 1440, 371)
    canvas.paste(_c5, (0, 2589), _c5)
except Exception:
    pass
layout["9.8"] = [0, 2589, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_10_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-13/06_icon_Shane_Gillis.png
try:
    _c6 = get_crop(6, 1344, 156)
    canvas.paste(_c6, (48, 120), _c6)
except Exception:
    pass
layout["Shane_Gillis"] = [48, 120, 1392, 276]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_10_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-13/07_icon_Low_pri.png
try:
    _c7 = get_crop(7, 186, 108)
    canvas.paste(_c7, (1254, 312), _c7)
except Exception:
    pass
layout["Low_pri"] = [1254, 312, 1440, 420]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_10_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-13/08_icon_Netflix_Is_A_Joke.png
try:
    _c8 = get_crop(8, 59, 64)
    canvas.paste(_c8, (311, 2), _c8)
except Exception:
    pass
layout["Netflix_Is_A_Joke"] = [311, 2, 370, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_10_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-13/09_icon_7_12_my.png
try:
    _c9 = get_crop(9, 67, 63)
    canvas.paste(_c9, (111, 1), _c9)
except Exception:
    pass
layout["7:12_my"] = [111, 1, 178, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_10_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-13/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 51, 64)
    canvas.paste(_c10, (1152, 1), _c10)
except Exception:
    pass
layout["icon_10"] = [1152, 1, 1203, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_10_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-13/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 63, 62)
    canvas.paste(_c11, (242, 2), _c11)
except Exception:
    pass
layout["icon_11"] = [242, 2, 305, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_10_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-13/12_icon_0.png
try:
    _c12 = get_crop(12, 103, 63)
    canvas.paste(_c12, (1212, 0), _c12)
except Exception:
    pass
layout["0#"] = [1212, 0, 1315, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_10_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-13/13_icon_7_12_my.png
try:
    _c13 = get_crop(13, 53, 61)
    canvas.paste(_c13, (182, 2), _c13)
except Exception:
    pass
layout["7:12_my"] = [182, 2, 235, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_10_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-13/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 54, 60)
    canvas.paste(_c14, (1319, 2), _c14)
except Exception:
    pass
layout["icon_14"] = [1319, 2, 1373, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_10_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-13/15_icon_0.png
try:
    _c15 = get_crop(15, 156, 156)
    canvas.paste(_c15, (1236, 120), _c15)
except Exception:
    pass
layout["0#"] = [1236, 120, 1392, 276]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_10_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-13/16_icon_Netflix_Is_A_Joke.png
try:
    _c16 = get_crop(16, 49, 65)
    canvas.paste(_c16, (383, 1), _c16)
except Exception:
    pass
layout["Netflix_Is_A_Joke"] = [383, 1, 432, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_10_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-13/17_icon_Sort_by_deal.png
try:
    _c17 = get_crop(17, 440, 144)
    canvas.paste(_c17, (976, 1989), _c17)
except Exception:
    pass
layout["Sort_by_deal"] = [976, 1989, 1416, 2133]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_10_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-13/18_icon_7_12_my.png
try:
    _c18 = get_crop(18, 98, 65)
    canvas.paste(_c18, (6, 0), _c18)
except Exception:
    pass
layout["7:12_my"] = [6, 0, 104, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_10_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-13/19_icon_Amazing_deal.png
try:
    _c19 = get_crop(19, 1440, 455)
    canvas.paste(_c19, (0, 2134), _c19)
except Exception:
    pass
layout["Amazing_deal"] = [0, 2134, 1440, 2589]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_10_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-13/20_icon_STAGE.png
try:
    _c20 = get_crop(20, 287, 192)
    canvas.paste(_c20, (575, 690), _c20)
except Exception:
    pass
layout["STAGE"] = [575, 690, 862, 882]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_10_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-13/21_text_PIT.png
try:
    _c21 = get_crop(21, 39, 25)
    canvas.paste(_c21, (610, 890), _c21)
except Exception:
    pass
layout["PIT"] = [610, 890, 649, 915]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_10_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-13/22_text_PIT.png
try:
    _c22 = get_crop(22, 38, 27)
    canvas.paste(_c22, (791, 890), _c22)
except Exception:
    pass
layout["PIT"] = [791, 890, 829, 917]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_10_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-13/23_text_SOUTH.png
try:
    _c23 = get_crop(23, 68, 24)
    canvas.paste(_c23, (1034, 1064), _c23)
except Exception:
    pass
layout["SOUTH"] = [1034, 1064, 1102, 1088]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_10_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-13/24_text_VIP_BOXES.png
try:
    _c24 = get_crop(24, 120, 27)
    canvas.paste(_c24, (446, 1091), _c24)
except Exception:
    pass
layout["VIP_BOXES"] = [446, 1091, 566, 1118]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_10_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-13/25_text_VIP_BOXES.png
try:
    _c25 = get_crop(25, 119, 30)
    canvas.paste(_c25, (872, 1091), _c25)
except Exception:
    pass
layout["VIP_BOXES"] = [872, 1091, 991, 1121]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_10_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-13/26_text_NORTH.png
try:
    _c26 = get_crop(26, 64, 20)
    canvas.paste(_c26, (362, 1178), _c26)
except Exception:
    pass
layout["NORTH"] = [362, 1178, 426, 1198]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_10_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-13/27_text_SOUTH.png
try:
    _c27 = get_crop(27, 69, 25)
    canvas.paste(_c27, (1010, 1177), _c27)
except Exception:
    pass
layout["SOUTH"] = [1010, 1177, 1079, 1202]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_10_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-13/28_text_BENCH.png
try:
    _c28 = get_crop(28, 89, 38)
    canvas.paste(_c28, (740, 1663), _c28)
except Exception:
    pass
layout["BENCH"] = [740, 1663, 829, 1701]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_10_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-13/29_text_367_Listings.png
try:
    _c29 = get_crop(29, 330, 77)
    canvas.paste(_c29, (56, 2027), _c29)
except Exception:
    pass
layout["367_Listings"] = [56, 2027, 386, 2104]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_10_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-13/30_text_S77_each.png
try:
    _c30 = get_crop(30, 1440, 371)
    canvas.paste(_c30, (0, 2589), _c30)
except Exception:
    pass
layout["S77_each"] = [0, 2589, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_10_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-13/31_text_Price_includes_fees.png
try:
    _c31 = get_crop(31, 1440, 371)
    canvas.paste(_c31, (0, 2589), _c31)
except Exception:
    pass
layout["Price_includes_fees"] = [0, 2589, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_10_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-13/32_text_9.8.png
try:
    _c32 = get_crop(32, 50, 31)
    canvas.paste(_c32, (502, 2810), _c32)
except Exception:
    pass
layout["9.8"] = [502, 2810, 552, 2841]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_10_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-13/33_text_Amazing_deal.png
try:
    _c33 = get_crop(33, 1440, 371)
    canvas.paste(_c33, (0, 2589), _c33)
except Exception:
    pass
layout["Amazing_deal"] = [0, 2589, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_10_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-13/34_text_1ticket.png
try:
    _c34 = get_crop(34, 147, 43)
    canvas.paste(_c34, (491, 2876), _c34)
except Exception:
    pass
layout["1ticket"] = [491, 2876, 638, 2919]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_10_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-13/35_text_IBENCHI.png
try:
    _c35 = get_crop(35, 92, 46)
    canvas.paste(_c35, (495, 1645), _c35)
except Exception:
    pass
layout["IBENCHI"] = [495, 1645, 587, 1691]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_10_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-13/36_text_BENCHI.png
try:
    _c36 = get_crop(36, 89, 43)
    canvas.paste(_c36, (852, 1646), _c36)
except Exception:
    pass
layout["BENCHI"] = [852, 1646, 941, 1689]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_10_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-13/37_clickable_Back.png
try:
    _c37 = get_crop(37, 156, 156)
    canvas.paste(_c37, (48, 120), _c37)
except Exception:
    pass
layout["Back"] = [48, 120, 204, 276]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_10_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-13/38_clickable_Netflix_Is_A_Joke_-_Shane_Gillis.png
try:
    _c38 = get_crop(38, 582, 156)
    canvas.paste(_c38, (204, 120), _c38)
except Exception:
    pass
layout["Netflix_Is_A_Joke_-_Shane"] = [204, 120, 786, 276]
