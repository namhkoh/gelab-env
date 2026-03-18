# page_id: page_seatgeek_21b637d11bea46b8adb3c2efc9f03501_04
# screenshot: 2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-7.png
# step_index: 4/10
# task: Open SeatGeek and find the soonest upcoming NBA game in New York with "Nets", record the cheapest price in google keep notes.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# fill overall background (canvas is provided)
draw.rectangle((0, 0, 1440, 2960), fill="#ffffff")

# Status bar area (top)
status_h = 80
draw.rectangle((0, 0, 1440, status_h), fill="#f3f3f3")  # light gray status background
# subtle bottom divider under status bar
draw.line((0, status_h-1, 1440, status_h-1), fill="#e6e6e6", width=1)

# Search box background (rounded)
search_left = 40
search_top = 110
search_right = 1400
search_bottom = 264
draw.rounded_rectangle(
    (search_left, search_top, search_right, search_bottom),
    radius=18,
    fill="#fbfbfb",
    outline="#e6e6e6",
    width=2
)

# Thin divider below search area
draw.line((32, search_bottom + 28, 1408, search_bottom + 28), fill="#ededed", width=1)

# Section group background cards (subtle off-white panels to define groups)
groups = [
    (24, 320, 1416, 760),   # Top results group area
    (24, 760, 1416, 1200),  # Performers group area
    (24, 1200, 1416, 1880), # Events group area
    (24, 1880, 2600, 2680)  # Recent searches area (extends downward)
]
for (l, t, r, b) in groups:
    draw.rounded_rectangle(
        (l, t, r, b),
        radius=10,
        fill="#ffffff",
        outline="#f2f2f2",
        width=1
    )
    # subtle inner top shadow line to separate header from list
    draw.line((l+8, t+62, r-8, t+62), fill="#f5f5f5", width=1)

# Horizontal separators between logical sections (full-width subtle rules)
separator_ys = [search_bottom + 28, 660, 1020, 1500, 1880, 2260, 2708]
for y in separator_ys:
    draw.line((24, y, 1416, y), fill="#ececec", width=1)

# Bottom navigation bar area
nav_top = 2720
draw.rectangle((0, nav_top, 1440, 2960), fill="#ffffff")
# top divider for nav bar
draw.line((0, nav_top, 1440, nav_top), fill="#e9e9e9", width=1)

# Add subtle left/right safe-area padding guides (very faint, purely structural)
draw.line((24, status_h + 8, 24, 2960), fill="#fbfbfb", width=1)
draw.line((1416, status_h + 8, 1416, 2960), fill="#fbfbfb", width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_04_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-7/00_icon_Top_results.png
try:
    _c0 = get_crop(0, 1440, 179)
    canvas.paste(_c0, (0, 471), _c0)
except Exception:
    pass
layout["Top_results"] = [0, 471, 1440, 650]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_04_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-7/01_icon_Performers.png
try:
    _c1 = get_crop(1, 1440, 179)
    canvas.paste(_c1, (0, 1217), _c1)
except Exception:
    pass
layout["Performers"] = [0, 1217, 1440, 1396]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_04_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-7/02_icon_Netsi.png
try:
    _c2 = get_crop(2, 1032, 144)
    canvas.paste(_c2, (216, 120), _c2)
except Exception:
    pass
layout["Netsi"] = [216, 120, 1248, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_04_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-7/03_icon_Events.png
try:
    _c3 = get_crop(3, 1440, 179)
    canvas.paste(_c3, (0, 1963), _c3)
except Exception:
    pass
layout["Events"] = [0, 1963, 1440, 2142]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_04_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-7/04_icon_5_events.png
try:
    _c4 = get_crop(4, 1440, 179)
    canvas.paste(_c4, (0, 829), _c4)
except Exception:
    pass
layout["5_events"] = [0, 829, 1440, 1008]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_04_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-7/05_icon_167_events.png
try:
    _c5 = get_crop(5, 1440, 179)
    canvas.paste(_c5, (0, 650), _c5)
except Exception:
    pass
layout["167_events"] = [0, 650, 1440, 829]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_04_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-7/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 46, 70)
    canvas.paste(_c6, (1153, 0), _c6)
except Exception:
    pass
layout["icon_6"] = [1153, 0, 1199, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_04_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-7/07_icon_6.37.png
try:
    _c7 = get_crop(7, 168, 144)
    canvas.paste(_c7, (48, 120), _c7)
except Exception:
    pass
layout["6.37"] = [48, 120, 216, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_04_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-7/08_icon_167_events.png
try:
    _c8 = get_crop(8, 1440, 179)
    canvas.paste(_c8, (0, 1396), _c8)
except Exception:
    pass
layout["167_events"] = [0, 1396, 1440, 1575]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_04_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-7/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 93, 67)
    canvas.paste(_c9, (1219, 0), _c9)
except Exception:
    pass
layout["icon_9"] = [1219, 0, 1312, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_04_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-7/10_icon_5_events.png
try:
    _c10 = get_crop(10, 1440, 179)
    canvas.paste(_c10, (0, 1575), _c10)
except Exception:
    pass
layout["5_events"] = [0, 1575, 1440, 1754]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_04_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-7/11_icon_New_York_NY.png
try:
    _c11 = get_crop(11, 1440, 179)
    canvas.paste(_c11, (0, 1963), _c11)
except Exception:
    pass
layout["New_York,_NY"] = [0, 1963, 1440, 2142]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_04_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-7/12_icon_AIct.png
try:
    _c12 = get_crop(12, 288, 168)
    canvas.paste(_c12, (576, 2792), _c12)
except Exception:
    pass
layout["AIct^"] = [576, 2792, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_04_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-7/13_icon_Yesterday.png
try:
    _c13 = get_crop(13, 1440, 179)
    canvas.paste(_c13, (0, 2142), _c13)
except Exception:
    pass
layout["Yesterday"] = [0, 2142, 1440, 2321]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_04_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-7/14_icon_New_Orleans_Pelicans_at_Brooklyn_Nets.png
try:
    _c14 = get_crop(14, 1440, 179)
    canvas.paste(_c14, (0, 2142), _c14)
except Exception:
    pass
layout["New_Orleans_Pelicans_at_B"] = [0, 2142, 1440, 2321]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_04_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-7/15_icon_Tracking.png
try:
    _c15 = get_crop(15, 288, 168)
    canvas.paste(_c15, (864, 2792), _c15)
except Exception:
    pass
layout["Tracking"] = [864, 2792, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_04_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-7/16_icon_icon_16.png
try:
    _c16 = get_crop(16, 50, 65)
    canvas.paste(_c16, (1319, 1), _c16)
except Exception:
    pass
layout["icon_16"] = [1319, 1, 1369, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_04_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-7/17_icon_Clear.png
try:
    _c17 = get_crop(17, 144, 144)
    canvas.paste(_c17, (1248, 120), _c17)
except Exception:
    pass
layout["Clear"] = [1248, 120, 1392, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_04_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-7/18_icon_AIct.png
try:
    _c18 = get_crop(18, 288, 162)
    canvas.paste(_c18, (288, 2792), _c18)
except Exception:
    pass
layout["AIct^"] = [288, 2792, 576, 2954]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_04_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-7/19_icon_Milwaukee_WI.png
try:
    _c19 = get_crop(19, 1440, 179)
    canvas.paste(_c19, (0, 2321), _c19)
except Exception:
    pass
layout["Milwaukee,_WI"] = [0, 2321, 1440, 2500]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_04_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-7/20_icon_Tomorrow.png
try:
    _c20 = get_crop(20, 1440, 179)
    canvas.paste(_c20, (0, 2321), _c20)
except Exception:
    pass
layout["Tomorrow"] = [0, 2321, 1440, 2500]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_04_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-7/21_icon_Account.png
try:
    _c21 = get_crop(21, 288, 168)
    canvas.paste(_c21, (1152, 2792), _c21)
except Exception:
    pass
layout["Account"] = [1152, 2792, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_04_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-7/22_icon_Drcallr.png
try:
    _c22 = get_crop(22, 288, 168)
    canvas.paste(_c22, (0, 2792), _c22)
except Exception:
    pass
layout["Drcallr"] = [0, 2792, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_04_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-7/23_icon_13_events.png
try:
    _c23 = get_crop(23, 186, 58)
    canvas.paste(_c23, (235, 1309), _c23)
except Exception:
    pass
layout["13_events"] = [235, 1309, 421, 1367]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_04_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-7/24_icon_Brooklyn_Nets.png
try:
    _c24 = get_crop(24, 1440, 179)
    canvas.paste(_c24, (0, 471), _c24)
except Exception:
    pass
layout["Brooklyn_Nets"] = [0, 471, 1440, 650]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_04_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-7/25_icon_New_York_Mets.png
try:
    _c25 = get_crop(25, 323, 55)
    canvas.paste(_c25, (232, 1429), _c25)
except Exception:
    pass
layout["New_York_Mets"] = [232, 1429, 555, 1484]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_04_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-7/26_icon_Long_Island_Nets.png
try:
    _c26 = get_crop(26, 1440, 179)
    canvas.paste(_c26, (0, 829), _c26)
except Exception:
    pass
layout["Long_Island_Nets"] = [0, 829, 1440, 1008]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_04_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-7/27_icon_Brooklyn_Nets.png
try:
    _c27 = get_crop(27, 307, 57)
    canvas.paste(_c27, (233, 1249), _c27)
except Exception:
    pass
layout["Brooklyn_Nets"] = [233, 1249, 540, 1306]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_04_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-7/28_icon_Long_Island_Nets.png
try:
    _c28 = get_crop(28, 1440, 179)
    canvas.paste(_c28, (0, 1575), _c28)
except Exception:
    pass
layout["Long_Island_Nets"] = [0, 1575, 1440, 1754]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_04_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-7/29_icon_Long_Island_Nets.png
try:
    _c29 = get_crop(29, 1440, 179)
    canvas.paste(_c29, (0, 829), _c29)
except Exception:
    pass
layout["Long_Island_Nets"] = [0, 829, 1440, 1008]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_04_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-7/30_text_6.37.png
try:
    _c30 = get_crop(30, 89, 45)
    canvas.paste(_c30, (20, 15), _c30)
except Exception:
    pass
layout["6.37"] = [20, 15, 109, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_04_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-7/31_text_EK.png
try:
    _c31 = get_crop(31, 50, 39)
    canvas.paste(_c31, (253, 24), _c31)
except Exception:
    pass
layout["EK"] = [253, 24, 303, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_04_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-7/32_text_Top_results.png
try:
    _c32 = get_crop(32, 295, 72)
    canvas.paste(_c32, (40, 373), _c32)
except Exception:
    pass
layout["Top_results"] = [40, 373, 335, 445]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_04_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-7/33_text_New_York_Mets.png
try:
    _c33 = get_crop(33, 328, 55)
    canvas.paste(_c33, (234, 685), _c33)
except Exception:
    pass
layout["New_York_Mets"] = [234, 685, 562, 740]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_04_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-7/34_text_167_events.png
try:
    _c34 = get_crop(34, 214, 45)
    canvas.paste(_c34, (235, 748), _c34)
except Exception:
    pass
layout["167_events"] = [235, 748, 449, 793]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_04_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-7/35_text_Performers.png
try:
    _c35 = get_crop(35, 293, 54)
    canvas.paste(_c35, (44, 1122), _c35)
except Exception:
    pass
layout["Performers"] = [44, 1122, 337, 1176]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_04_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-7/36_text_167_events.png
try:
    _c36 = get_crop(36, 214, 45)
    canvas.paste(_c36, (235, 1495), _c36)
except Exception:
    pass
layout["167_events"] = [235, 1495, 449, 1540]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_04_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-7/37_text_Events.png
try:
    _c37 = get_crop(37, 181, 57)
    canvas.paste(_c37, (43, 1868), _c37)
except Exception:
    pass
layout["Events"] = [43, 1868, 224, 1925]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_04_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-7/38_text_Recent_searches.png
try:
    _c38 = get_crop(38, 288, 168)
    canvas.paste(_c38, (0, 2792), _c38)
except Exception:
    pass
layout["Recent_searches"] = [0, 2792, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_04_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-7/39_text_Drcallr.png
try:
    _c39 = get_crop(39, 288, 162)
    canvas.paste(_c39, (288, 2792), _c39)
except Exception:
    pass
layout["Drcallr"] = [288, 2792, 576, 2954]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_04_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-7/40_text_AIct.png
try:
    _c40 = get_crop(40, 103, 27)
    canvas.paste(_c40, (435, 2773), _c40)
except Exception:
    pass
layout["AIct^"] = [435, 2773, 538, 2800]
