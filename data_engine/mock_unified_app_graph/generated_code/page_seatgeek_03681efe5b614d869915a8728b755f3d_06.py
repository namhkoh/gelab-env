# page_id: page_seatgeek_03681efe5b614d869915a8728b755f3d_06
# screenshot: 2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-9.png
# step_index: 6/10
# task: Open SeatGeek. Search "Metropolitan Opera". Find the next available show. Filter by "best seats". What section are they in for the lowest price tickets?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Paint overall page background to match the app's light gray canvas
draw.rectangle((0, 0, 1440, 2960), fill="#eef1f3")

# Status bar area (top ~80px) - slightly darker than background
draw.rectangle((0, 0, 1440, 80), fill="#e3e6e8")

# Subtle bottom hairline under status bar
draw.line((0, 80, 1440, 80), fill="#d6d9db", width=1)

# Header card shadow (slight offset) and header rounded card background
# Header card bounds correspond to detected clickable/header region: (48,120) size 1344x156
shadow_box = (46, 132, 1394, 304)  # slightly offset for shadow
draw.rounded_rectangle(shadow_box, radius=36, fill="#e9ebec")

header_box = (48, 120, 1392, 276)
draw.rounded_rectangle(header_box, radius=32, fill="#ffffff")

# Subtle divider below header to separate header from filter chips area
draw.line((64, 276, 1376, 276), fill="#ebedef", width=1)

# Light backdrop area for the seat-map region (center area)
# Create a muted panel background to set apart the seating diagrams
seat_panel = (48, 300, 1392, 1880)
draw.rounded_rectangle(seat_panel, radius=24, fill="#f5f7f8")

# Very subtle inner border for the seat map area
draw.rounded_rectangle(seat_panel, radius=24, outline="#e6e9ea", width=1)

# Horizontal separator between seating area and listings
sep_y = 1888
draw.line((48, sep_y, 1392, sep_y), fill="#e0e3e5", width=1)

# Listings card: white rounded rectangle starting below the seating area
listings_top = 1960
listings_box = (24, listings_top, 1416, 2956)
draw.rounded_rectangle(listings_box, radius=40, fill="#ffffff")

# Subtle top shadow/hairline for listings card to give elevation
draw.line((40, listings_top, 1400, listings_top), fill="#e6e9ea", width=1)

# Small centered subtle handle near top of listings card (visual affordance)
handle_w = 120
handle_h = 6
handle_x0 = (1440 - handle_w) // 2
handle_y0 = listings_top + 12
draw.rectangle((handle_x0, handle_y0, handle_x0 + handle_w, handle_y0 + handle_h), fill="#f0f2f3")

# Internal separators in the listings area (suggestive dividers only)
# Keep these faint so pasted content (thumbnails/text) will remain primary
divider_positions = [listings_top + 220, listings_top + 720, listings_top + 1220]
for y in divider_positions:
    draw.line((40, y, 1400, y), fill="#f1f3f4", width=1)

# Right-side thin vertical rule near the sort area (visual structure only)
draw.line((1360, listings_top + 16, 1360, listings_top + 80), fill="#eceff0", width=1)

# Footer safe-area bottom hairline
draw.line((0, 2956, 1440, 2956), fill="#e9ebec", width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_06_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-9/00_icon_Include_fees.png
try:
    _c0 = get_crop(0, 335, 108)
    canvas.paste(_c0, (544, 312), _c0)
except Exception:
    pass
layout["Include_fees"] = [544, 312, 879, 420]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_06_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-9/01_icon_Best_seats.png
try:
    _c1 = get_crop(1, 303, 108)
    canvas.paste(_c1, (915, 312), _c1)
except Exception:
    pass
layout["Best_seats"] = [915, 312, 1218, 420]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_06_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-9/02_icon_Quantity.png
try:
    _c2 = get_crop(2, 268, 108)
    canvas.paste(_c2, (240, 312), _c2)
except Exception:
    pass
layout["Quantity"] = [240, 312, 508, 420]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_06_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-9/03_icon_Tit.png
try:
    _c3 = get_crop(3, 156, 108)
    canvas.paste(_c3, (48, 312), _c3)
except Exception:
    pass
layout["Tit"] = [48, 312, 204, 420]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_06_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-9/04_icon_74.png
try:
    _c4 = get_crop(4, 1440, 455)
    canvas.paste(_c4, (0, 2355), _c4)
except Exception:
    pass
layout["74"] = [0, 2355, 1440, 2810]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_06_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-9/05_icon_Low_pri.png
try:
    _c5 = get_crop(5, 186, 108)
    canvas.paste(_c5, (1254, 312), _c5)
except Exception:
    pass
layout["Low_pri"] = [1254, 312, 1440, 420]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_06_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-9/06_icon_Include_fees.png
try:
    _c6 = get_crop(6, 1344, 156)
    canvas.paste(_c6, (48, 120), _c6)
except Exception:
    pass
layout["Include_fees"] = [48, 120, 1392, 276]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_06_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-9/07_icon_El_Nino.png
try:
    _c7 = get_crop(7, 56, 62)
    canvas.paste(_c7, (313, 2), _c7)
except Exception:
    pass
layout["El_Nino"] = [313, 2, 369, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_06_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-9/08_icon_7.57_Wy.png
try:
    _c8 = get_crop(8, 70, 64)
    canvas.paste(_c8, (109, 0), _c8)
except Exception:
    pass
layout["7.57_Wy"] = [109, 0, 179, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_06_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-9/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 52, 64)
    canvas.paste(_c9, (1151, 1), _c9)
except Exception:
    pass
layout["icon_9"] = [1151, 1, 1203, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_06_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-9/10_icon_El_Nino.png
try:
    _c10 = get_crop(10, 61, 61)
    canvas.paste(_c10, (243, 2), _c10)
except Exception:
    pass
layout["El_Nino"] = [243, 2, 304, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_06_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-9/11_icon_0.png
try:
    _c11 = get_crop(11, 102, 63)
    canvas.paste(_c11, (1213, 0), _c11)
except Exception:
    pass
layout["0#"] = [1213, 0, 1315, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_06_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-9/12_icon_7.57_Wy.png
try:
    _c12 = get_crop(12, 55, 60)
    canvas.paste(_c12, (182, 1), _c12)
except Exception:
    pass
layout["7.57_Wy"] = [182, 1, 237, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_06_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-9/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 53, 57)
    canvas.paste(_c13, (1320, 3), _c13)
except Exception:
    pass
layout["icon_13"] = [1320, 3, 1373, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_06_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-9/14_icon_Sort_by_price.png
try:
    _c14 = get_crop(14, 455, 144)
    canvas.paste(_c14, (961, 1989), _c14)
except Exception:
    pass
layout["Sort_by_price"] = [961, 1989, 1416, 2133]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_06_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-9/15_icon_New_York.png
try:
    _c15 = get_crop(15, 49, 63)
    canvas.paste(_c15, (383, 1), _c15)
except Exception:
    pass
layout["New_York"] = [383, 1, 432, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_06_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-9/16_icon_0.png
try:
    _c16 = get_crop(16, 156, 156)
    canvas.paste(_c16, (1236, 120), _c16)
except Exception:
    pass
layout["0#"] = [1236, 120, 1392, 276]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_06_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-9/17_icon_Great_deal.png
try:
    _c17 = get_crop(17, 1440, 455)
    canvas.paste(_c17, (0, 2355), _c17)
except Exception:
    pass
layout["Great_deal"] = [0, 2355, 1440, 2810]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_06_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-9/18_icon_S82_each.png
try:
    _c18 = get_crop(18, 383, 106)
    canvas.paste(_c18, (52, 2854), _c18)
except Exception:
    pass
layout["S82_each"] = [52, 2854, 435, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_06_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-9/19_icon_7.57_Wy.png
try:
    _c19 = get_crop(19, 99, 64)
    canvas.paste(_c19, (5, 0), _c19)
except Exception:
    pass
layout["7.57_Wy"] = [5, 0, 104, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_06_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-9/20_text_STAGE.png
try:
    _c20 = get_crop(20, 42, 16)
    canvas.paste(_c20, (470, 611), _c20)
except Exception:
    pass
layout["STAGE"] = [470, 611, 512, 627]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_06_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-9/21_text_ORCHESTRA_PIT.png
try:
    _c21 = get_crop(21, 136, 25)
    canvas.paste(_c21, (421, 684), _c21)
except Exception:
    pass
layout["ORCHESTRA_PIT"] = [421, 684, 557, 709]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_06_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-9/22_text_49.png
try:
    _c22 = get_crop(22, 36, 30)
    canvas.paste(_c22, (844, 920), _c22)
except Exception:
    pass
layout["49"] = [844, 920, 880, 950]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_06_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-9/23_text_18.png
try:
    _c23 = get_crop(23, 34, 29)
    canvas.paste(_c23, (1057, 911), _c23)
except Exception:
    pass
layout["18"] = [1057, 911, 1091, 940]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_06_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-9/24_text_27.png
try:
    _c24 = get_crop(24, 36, 27)
    canvas.paste(_c24, (923, 941), _c24)
except Exception:
    pass
layout["27"] = [923, 941, 959, 968]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_06_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-9/25_text_26.png
try:
    _c25 = get_crop(25, 34, 27)
    canvas.paste(_c25, (983, 939), _c25)
except Exception:
    pass
layout["26"] = [983, 939, 1017, 966]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_06_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-9/26_text_LEVEL_1_ORCHESTRA.png
try:
    _c26 = get_crop(26, 173, 25)
    canvas.paste(_c26, (402, 1017), _c26)
except Exception:
    pass
layout["LEVEL_1_ORCHESTRA"] = [402, 1017, 575, 1042]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_06_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-9/27_text_LEVEL_2_PARTERRE.png
try:
    _c27 = get_crop(27, 159, 25)
    canvas.paste(_c27, (879, 1017), _c27)
except Exception:
    pass
layout["LEVEL_2_PARTERRE"] = [879, 1017, 1038, 1042]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_06_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-9/28_text_30.png
try:
    _c28 = get_crop(28, 32, 27)
    canvas.paste(_c28, (615, 1126), _c28)
except Exception:
    pass
layout["30"] = [615, 1126, 647, 1153]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_06_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-9/29_text_33.png
try:
    _c29 = get_crop(29, 31, 27)
    canvas.paste(_c29, (331, 1149), _c29)
except Exception:
    pass
layout["33"] = [331, 1149, 362, 1176]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_06_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-9/30_text_ROOM.png
try:
    _c30 = get_crop(30, 64, 27)
    canvas.paste(_c30, (950, 1367), _c30)
except Exception:
    pass
layout["~ROOM"] = [950, 1367, 1014, 1394]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_06_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-9/31_text_154_Listings.png
try:
    _c31 = get_crop(31, 322, 68)
    canvas.paste(_c31, (55, 2032), _c31)
except Exception:
    pass
layout["154_Listings"] = [55, 2032, 377, 2100]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_06_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-9/32_text_We_sell_resale_tickets._Resale_tickets_m.png
try:
    _c32 = get_crop(32, 1440, 455)
    canvas.paste(_c32, (0, 2355), _c32)
except Exception:
    pass
layout["€_We_sell_resale_tickets."] = [0, 2355, 1440, 2810]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_06_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-9/33_text_face_value.png
try:
    _c33 = get_crop(33, 218, 43)
    canvas.paste(_c33, (57, 2256), _c33)
except Exception:
    pass
layout["face_value:"] = [57, 2256, 275, 2299]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_06_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-9/34_text_S82_each.png
try:
    _c34 = get_crop(34, 239, 66)
    canvas.paste(_c34, (485, 2862), _c34)
except Exception:
    pass
layout["S82_each"] = [485, 2862, 724, 2928]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_06_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-9/35_text_STANDING.png
try:
    _c35 = get_crop(35, 104, 46)
    canvas.paste(_c35, (538, 933), _c35)
except Exception:
    pass
layout["STANDING"] = [538, 933, 642, 979]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_06_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-9/36_text_STANDING.png
try:
    _c36 = get_crop(36, 105, 46)
    canvas.paste(_c36, (337, 934), _c36)
except Exception:
    pass
layout["STANDING"] = [337, 934, 442, 980]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_06_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-9/37_text_STANDING.png
try:
    _c37 = get_crop(37, 104, 56)
    canvas.paste(_c37, (554, 1319), _c37)
except Exception:
    pass
layout["~STANDING"] = [554, 1319, 658, 1375]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_06_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-9/38_clickable_Back.png
try:
    _c38 = get_crop(38, 156, 156)
    canvas.paste(_c38, (48, 120), _c38)
except Exception:
    pass
layout["Back"] = [48, 120, 204, 276]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_06_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-9/39_clickable_El_Nino_-_New_York.png
try:
    _c39 = get_crop(39, 363, 156)
    canvas.paste(_c39, (204, 120), _c39)
except Exception:
    pass
layout["El_Nino_-_New_York"] = [204, 120, 567, 276]
