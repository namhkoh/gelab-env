# page_id: page_seatgeek_1cc69540849e491bb4fc78ed1f09c554_04
# screenshot: 2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-7.png
# step_index: 4/7
# task: Open SeatGeek. Search "Madison Square Garden". Select the next upcoming event. Who are the performers of the event?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Background fill (dominant off-white)
draw.rectangle((0, 0, 1440, 2960), fill="#FBFBFB")

# Status bar area at top (~72px)
draw.rectangle((0, 0, 1440, 72), fill="#EFEFF0")

# Subtle bottom hairline of status bar
draw.line((24, 72, 1416, 72), fill="#E6E6E6", width=1)

# Search/header bar background (rounded rectangle)
search_left, search_top = 32, 84
search_right, search_bottom = 1408, 196
# faint shadow behind search bar
draw.rounded_rectangle((search_left+2, search_top+6, search_right+2, search_bottom+6),
                       radius=30, fill="#F6F6F6", outline=None)
# main search bar
draw.rounded_rectangle((search_left, search_top, search_right, search_bottom),
                       radius=30, fill="#FFFFFF", outline="#EAEAEA", width=1)

# Divider under search area
draw.line((24, 216, 1416, 216), fill="#EDEDED", width=1)

# Section cards (rounded white cards grouping list areas)
# Top results / results card
draw.rounded_rectangle((24, 260, 1416, 1040), radius=12, fill="#FFFFFF", outline="#F1F1F1", width=1)
# Performers small card (separating area)
draw.rounded_rectangle((24, 1048, 1416, 1320), radius=12, fill="#FFFFFF", outline="#F1F1F1", width=1)
# Events card
draw.rounded_rectangle((24, 1360, 1416, 2320), radius=12, fill="#FFFFFF", outline="#F1F1F1", width=1)
# Venues card
draw.rounded_rectangle((24, 2328, 1416, 2656), radius=12, fill="#FFFFFF", outline="#F1F1F1", width=1)

# Subtle separators between individual rows/sections (left/right margin aligned)
separator_x1, separator_x2 = 40, 1400
separator_ys = [216, 328, 520, 700, 840, 1008, 1320, 1396, 1645, 1880, 1963, 2320, 2630, 2792]
for y in separator_ys:
    draw.line((separator_x1, y, separator_x2, y), fill="#ECECEC", width=1)

# Light background band for the main content area to separate from nav area
draw.rectangle((0, 2656, 1440, 2960), fill="#FFFFFF")

# Top subtle shadow under header/search card to give depth
draw.line((24, 200, 1416, 200), fill="#F4F4F4", width=2)

# Bottom navigation bar top divider
draw.line((0, 2792, 1440, 2792), fill="#EAEAEA", width=1)

# Slight inner divider for deeper separation near the very bottom
draw.line((24, 2936, 1416, 2936), fill="#F7F7F7", width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_04_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-7/00_icon_Performers.png
try:
    _c0 = get_crop(0, 1440, 179)
    canvas.paste(_c0, (0, 1217), _c0)
except Exception:
    pass
layout["Performers"] = [0, 1217, 1440, 1396]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_04_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-7/01_icon_Tomorrow.png
try:
    _c1 = get_crop(1, 1440, 179)
    canvas.paste(_c1, (0, 829), _c1)
except Exception:
    pass
layout["Tomorrow"] = [0, 829, 1440, 1008]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_04_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-7/02_icon_Events.png
try:
    _c2 = get_crop(2, 1440, 179)
    canvas.paste(_c2, (0, 1605), _c2)
except Exception:
    pass
layout["Events"] = [0, 1605, 1440, 1784]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_04_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-7/03_icon_Billy_Joel.png
try:
    _c3 = get_crop(3, 1440, 179)
    canvas.paste(_c3, (0, 1963), _c3)
except Exception:
    pass
layout["Billy_Joel"] = [0, 1963, 1440, 2142]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_04_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-7/04_icon_Tonight.png
try:
    _c4 = get_crop(4, 1440, 179)
    canvas.paste(_c4, (0, 1784), _c4)
except Exception:
    pass
layout["Tonight"] = [0, 1784, 1440, 1963]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_04_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-7/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 57, 61)
    canvas.paste(_c5, (245, 3), _c5)
except Exception:
    pass
layout["icon_5"] = [245, 3, 302, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_04_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-7/06_icon_Madison_Square_Garden.png
try:
    _c6 = get_crop(6, 1440, 179)
    canvas.paste(_c6, (0, 471), _c6)
except Exception:
    pass
layout["Madison_Square_Garden"] = [0, 471, 1440, 650]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_04_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-7/07_icon_New_York_NY.png
try:
    _c7 = get_crop(7, 1440, 179)
    canvas.paste(_c7, (0, 1784), _c7)
except Exception:
    pass
layout["New_York,_NY"] = [0, 1784, 1440, 1963]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_04_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-7/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 42, 70)
    canvas.paste(_c8, (1156, 0), _c8)
except Exception:
    pass
layout["icon_8"] = [1156, 0, 1198, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_04_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-7/09_icon_Tracking.png
try:
    _c9 = get_crop(9, 288, 168)
    canvas.paste(_c9, (864, 2792), _c9)
except Exception:
    pass
layout["Tracking"] = [864, 2792, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_04_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-7/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 53, 61)
    canvas.paste(_c10, (315, 3), _c10)
except Exception:
    pass
layout["icon_10"] = [315, 3, 368, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_04_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-7/11_icon_Madison_Square_Garden_Holiday_Festival.png
try:
    _c11 = get_crop(11, 1440, 179)
    canvas.paste(_c11, (0, 1217), _c11)
except Exception:
    pass
layout["Madison_Square_Garden_Hol"] = [0, 1217, 1440, 1396]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_04_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-7/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 96, 68)
    canvas.paste(_c12, (1218, 0), _c12)
except Exception:
    pass
layout["icon_12"] = [1218, 0, 1314, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_04_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-7/13_icon_7.45_Wy.png
try:
    _c13 = get_crop(13, 168, 144)
    canvas.paste(_c13, (48, 120), _c13)
except Exception:
    pass
layout["7.45_Wy"] = [48, 120, 216, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_04_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-7/14_icon_7.45_Wy.png
try:
    _c14 = get_crop(14, 43, 61)
    canvas.paste(_c14, (188, 2), _c14)
except Exception:
    pass
layout["7.45_Wy"] = [188, 2, 231, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_04_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-7/15_icon_No_events.png
try:
    _c15 = get_crop(15, 288, 168)
    canvas.paste(_c15, (0, 2792), _c15)
except Exception:
    pass
layout["No_events"] = [0, 2792, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_04_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-7/16_icon_Account.png
try:
    _c16 = get_crop(16, 288, 168)
    canvas.paste(_c16, (1152, 2792), _c16)
except Exception:
    pass
layout["Account"] = [1152, 2792, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_04_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-7/17_icon_New_York_NY.png
try:
    _c17 = get_crop(17, 1440, 179)
    canvas.paste(_c17, (0, 1605), _c17)
except Exception:
    pass
layout["New_York,_NY"] = [0, 1605, 1440, 1784]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_04_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-7/18_icon_Madison_Square_Garden.png
try:
    _c18 = get_crop(18, 1440, 179)
    canvas.paste(_c18, (0, 2351), _c18)
except Exception:
    pass
layout["Madison_Square_Garden"] = [0, 2351, 1440, 2530]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_04_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-7/19_icon_icon_19.png
try:
    _c19 = get_crop(19, 43, 64)
    canvas.paste(_c19, (1327, 3), _c19)
except Exception:
    pass
layout["icon_19"] = [1327, 3, 1370, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_04_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-7/20_icon_No_events.png
try:
    _c20 = get_crop(20, 288, 162)
    canvas.paste(_c20, (288, 2792), _c20)
except Exception:
    pass
layout["No_events"] = [288, 2792, 576, 2954]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_04_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-7/21_icon_New_York_NY.png
try:
    _c21 = get_crop(21, 1440, 179)
    canvas.paste(_c21, (0, 1963), _c21)
except Exception:
    pass
layout["New_York,_NY"] = [0, 1963, 1440, 2142]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_04_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-7/22_icon_Tickets.png
try:
    _c22 = get_crop(22, 288, 168)
    canvas.paste(_c22, (576, 2792), _c22)
except Exception:
    pass
layout["Tickets"] = [576, 2792, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_04_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-7/23_icon_7.45_Wy.png
try:
    _c23 = get_crop(23, 51, 61)
    canvas.paste(_c23, (117, 2), _c23)
except Exception:
    pass
layout["7.45_Wy"] = [117, 2, 168, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_04_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-7/24_icon_Clear.png
try:
    _c24 = get_crop(24, 144, 144)
    canvas.paste(_c24, (1248, 120), _c24)
except Exception:
    pass
layout["Clear"] = [1248, 120, 1392, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_04_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-7/25_icon_Eastern_Conference_First_Round_Washingto.png
try:
    _c25 = get_crop(25, 1440, 179)
    canvas.paste(_c25, (0, 650), _c25)
except Exception:
    pass
layout["Eastern_Conference_First_"] = [0, 650, 1440, 829]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_04_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-7/26_text_Madison_Square_Garden.png
try:
    _c26 = get_crop(26, 1032, 144)
    canvas.paste(_c26, (216, 120), _c26)
except Exception:
    pass
layout["Madison_Square_Garden"] = [216, 120, 1248, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_04_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-7/27_text_Top_results.png
try:
    _c27 = get_crop(27, 295, 72)
    canvas.paste(_c27, (40, 373), _c27)
except Exception:
    pass
layout["Top_results"] = [40, 373, 335, 445]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_04_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-7/28_text_Eastern_Conference_First_Round_Washingto.png
try:
    _c28 = get_crop(28, 1440, 179)
    canvas.paste(_c28, (0, 829), _c28)
except Exception:
    pass
layout["Eastern_Conference_First_"] = [0, 829, 1440, 1008]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_04_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-7/29_text_Tomorrow.png
try:
    _c29 = get_crop(29, 209, 51)
    canvas.paste(_c29, (235, 929), _c29)
except Exception:
    pass
layout["Tomorrow"] = [235, 929, 444, 980]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_04_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-7/30_text_7_PM.png
try:
    _c30 = get_crop(30, 107, 40)
    canvas.paste(_c30, (464, 931), _c30)
except Exception:
    pass
layout["7_PM"] = [464, 931, 571, 971]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_04_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-7/31_text_New_York_NY.png
try:
    _c31 = get_crop(31, 1440, 179)
    canvas.paste(_c31, (0, 829), _c31)
except Exception:
    pass
layout["New_York;_NY"] = [0, 829, 1440, 1008]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_04_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-7/32_text_Performers.png
try:
    _c32 = get_crop(32, 293, 54)
    canvas.paste(_c32, (44, 1122), _c32)
except Exception:
    pass
layout["Performers"] = [44, 1122, 337, 1176]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_04_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-7/33_text_Events.png
try:
    _c33 = get_crop(33, 177, 54)
    canvas.paste(_c33, (46, 1510), _c33)
except Exception:
    pass
layout["Events"] = [46, 1510, 223, 1564]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_04_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-7/34_text_Venues.png
try:
    _c34 = get_crop(34, 197, 60)
    canvas.paste(_c34, (42, 2253), _c34)
except Exception:
    pass
layout["Venues"] = [42, 2253, 239, 2313]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_04_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-7/35_text_Madison_Square_Garden.png
try:
    _c35 = get_crop(35, 1440, 179)
    canvas.paste(_c35, (0, 2530), _c35)
except Exception:
    pass
layout["Madison_Square_Garden"] = [0, 2530, 1440, 2709]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_04_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-7/36_text_No_events.png
try:
    _c36 = get_crop(36, 201, 40)
    canvas.paste(_c36, (239, 2633), _c36)
except Exception:
    pass
layout["No_events"] = [239, 2633, 440, 2673]
