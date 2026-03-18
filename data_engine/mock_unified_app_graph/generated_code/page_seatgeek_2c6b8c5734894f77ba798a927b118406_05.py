# page_id: page_seatgeek_2c6b8c5734894f77ba798a927b118406_05
# screenshot: 2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-8.png
# step_index: 5/5
# task: Open SeatGeek. Search "Wembley Stadium". Show the next five football matches. Add to watch list.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw overall background (dominant color: white)
draw.rectangle([(0, 0), (1440, 2960)], fill=(255, 255, 255))

# Status bar area (top)
status_h = 72
draw.rectangle([(0, 0), (1440, status_h)], fill=(242, 242, 242))

# subtle divider under status bar
draw.line([(24, status_h), (1440-24, status_h)], fill=(225, 225, 225), width=2)

# NOTE: the map image area is detected and will be pasted at y=72..776 so we MUST NOT draw over it.
# Start drawing page content below the detected map area (map ends at y = 72 + 704 = 776)
map_bottom = 776

# Header / Title card background (full-width band starting immediately below map)
header_top = map_bottom
header_bottom = header_top + 152   # approx height for the title card area
# Use a white card with slight inset and a subtle top border (do not draw above map_bottom)
draw.rectangle([(0, header_top), (1440, header_bottom)], fill=(255, 255, 255))

# subtle top border line for separation (draw at map_bottom so it doesn't overlap map content)
draw.line([(24, header_top), (1440-24, header_top)], fill=(235, 235, 235), width=1)

# thin divider under header/title card
divider_y = header_bottom + 24
draw.line([(24, divider_y), (1440-24, divider_y)], fill=(235, 235, 235), width=1)

# A faint centered content band (empty content area background behind possible listings)
content_top = divider_y + 24
# Keep content area visually distinct by a very subtle off-white tint for the top section
content_band_height = 260
draw.rectangle([(0, content_top), (1440, content_top + content_band_height)], fill=(255, 255, 255))

# Secondary subtle horizontal separators to suggest sections further down the page
for i in range(3):
    y = content_top + (i + 1) * 80 + 20
    draw.line([(48, y), (1440-48, y)], fill=(245, 245, 245), width=1)

# Bottom padding area remains white (no additional elements drawn)
# End of UI background and structure drawing

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_05_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-8/00_icon_28.png
try:
    _c0 = get_crop(0, 144, 144)
    canvas.paste(_c0, (36, 84), _c0)
except Exception:
    pass
layout["28"] = [36, 84, 180, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_05_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-8/01_icon_A4088.png
try:
    _c1 = get_crop(1, 158, 199)
    canvas.paste(_c1, (808, 61), _c1)
except Exception:
    pass
layout["A4088"] = [808, 61, 966, 260]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_05_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-8/02_icon_Reservoir.png
try:
    _c2 = get_crop(2, 49, 69)
    canvas.paste(_c2, (1154, 1), _c2)
except Exception:
    pass
layout["Reservoir"] = [1154, 1, 1203, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_05_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-8/03_icon_icon_3.png
try:
    _c3 = get_crop(3, 46, 63)
    canvas.paste(_c3, (1325, 3), _c3)
except Exception:
    pass
layout["icon_3"] = [1325, 3, 1371, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_05_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-8/04_icon_B4557.png
try:
    _c4 = get_crop(4, 1440, 704)
    canvas.paste(_c4, (0, 72), _c4)
except Exception:
    pass
layout["B4557"] = [0, 72, 1440, 776]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_05_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-8/05_icon_See_more_options.png
try:
    _c5 = get_crop(5, 204, 174)
    canvas.paste(_c5, (1236, 806), _c5)
except Exception:
    pass
layout["See_more_options"] = [1236, 806, 1440, 980]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_05_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-8/06_icon_870.png
try:
    _c6 = get_crop(6, 70, 79)
    canvas.paste(_c6, (466, 509), _c6)
except Exception:
    pass
layout["870"] = [466, 509, 536, 588]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_05_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-8/07_icon_A407.png
try:
    _c7 = get_crop(7, 69, 58)
    canvas.paste(_c7, (1034, 609), _c7)
except Exception:
    pass
layout["A407"] = [1034, 609, 1103, 667]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_05_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-8/08_icon_Whitton.png
try:
    _c8 = get_crop(8, 87, 92)
    canvas.paste(_c8, (127, 497), _c8)
except Exception:
    pass
layout["Whitton"] = [127, 497, 214, 589]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_05_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-8/09_icon_Fulton_Rd.png
try:
    _c9 = get_crop(9, 54, 85)
    canvas.paste(_c9, (629, 317), _c9)
except Exception:
    pass
layout["Fulton_Rd"] = [629, 317, 683, 402]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_05_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-8/10_icon_Reservoir.png
try:
    _c10 = get_crop(10, 62, 69)
    canvas.paste(_c10, (1213, 3), _c10)
except Exception:
    pass
layout["Reservoir"] = [1213, 3, 1275, 72]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_05_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-8/11_text_7_05_my.png
try:
    _c11 = get_crop(11, 153, 52)
    canvas.paste(_c11, (19, 9), _c11)
except Exception:
    pass
layout["7:05_my"] = [19, 9, 172, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_05_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-8/12_text_0.png
try:
    _c12 = get_crop(12, 32, 27)
    canvas.paste(_c12, (543, 67), _c12)
except Exception:
    pass
layout["'0"] = [543, 67, 575, 94]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_05_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-8/13_text_A4088.png
try:
    _c13 = get_crop(13, 79, 38)
    canvas.paste(_c13, (272, 266), _c13)
except Exception:
    pass
layout["A4088"] = [272, 266, 351, 304]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_05_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-8/14_text_60.png
try:
    _c14 = get_crop(14, 50, 27)
    canvas.paste(_c14, (372, 268), _c14)
except Exception:
    pass
layout["60"] = [372, 268, 422, 295]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_05_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-8/15_text_Rd.png
try:
    _c15 = get_crop(15, 37, 32)
    canvas.paste(_c15, (143, 377), _c15)
except Exception:
    pass
layout["Rd"] = [143, 377, 180, 409]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_05_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-8/16_text_A407.png
try:
    _c16 = get_crop(16, 60, 27)
    canvas.paste(_c16, (1151, 553), _c16)
except Exception:
    pass
layout["A407"] = [1151, 553, 1211, 580]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_05_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-8/17_text_0_0.png
try:
    _c17 = get_crop(17, 75, 27)
    canvas.paste(_c17, (738, 664), _c17)
except Exception:
    pass
layout["0*0"] = [738, 664, 813, 691]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_05_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-8/18_text_Wembley_Stadium.png
try:
    _c18 = get_crop(18, 522, 86)
    canvas.paste(_c18, (44, 849), _c18)
except Exception:
    pass
layout["Wembley_Stadium"] = [44, 849, 566, 935]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_05_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-8/19_text_London_UK.png
try:
    _c19 = get_crop(19, 268, 56)
    canvas.paste(_c19, (41, 942), _c19)
except Exception:
    pass
layout["London,_UK"] = [41, 942, 309, 998]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_05_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-8/20_text_No_events_found.png
try:
    _c20 = get_crop(20, 350, 55)
    canvas.paste(_c20, (546, 1237), _c20)
except Exception:
    pass
layout["No_events_found"] = [546, 1237, 896, 1292]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_05_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-8/21_text_Brent.png
try:
    _c21 = get_crop(21, 89, 59)
    canvas.paste(_c21, (1042, 143), _c21)
except Exception:
    pass
layout["Brent"] = [1042, 143, 1131, 202]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_05_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-8/22_text_P4088.png
try:
    _c22 = get_crop(22, 79, 50)
    canvas.paste(_c22, (587, 205), _c22)
except Exception:
    pass
layout["P4088"] = [587, 205, 666, 255]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_05_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-8/23_text_Sudbury..png
try:
    _c23 = get_crop(23, 91, 86)
    canvas.paste(_c23, (39, 228), _c23)
except Exception:
    pass
layout["Sudbury."] = [39, 228, 130, 314]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_05_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-8/24_text_A4088.png
try:
    _c24 = get_crop(24, 85, 54)
    canvas.paste(_c24, (426, 260), _c24)
except Exception:
    pass
layout["A4088"] = [426, 260, 511, 314]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_05_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-8/25_text_Harrow_Rd.png
try:
    _c25 = get_crop(25, 116, 99)
    canvas.paste(_c25, (710, 559), _c25)
except Exception:
    pass
layout["Harrow_Rd"] = [710, 559, 826, 658]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_05_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-8/26_text_A4005.png
try:
    _c26 = get_crop(26, 88, 84)
    canvas.paste(_c26, (297, 585), _c26)
except Exception:
    pass
layout["A4005"] = [297, 585, 385, 669]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_05_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-8/27_text_A404.png
try:
    _c27 = get_crop(27, 71, 53)
    canvas.paste(_c27, (890, 701), _c27)
except Exception:
    pass
layout["A404"] = [890, 701, 961, 754]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_05_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-8/28_clickable_Tracking.png
try:
    _c28 = get_crop(28, 144, 144)
    canvas.paste(_c28, (1260, 84), _c28)
except Exception:
    pass
layout["Tracking"] = [1260, 84, 1404, 228]
