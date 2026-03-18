# page_id: page_seatgeek_2c8f932b941840c18364dd035f1c8473_04
# screenshot: 2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-7.png
# step_index: 4/8
# task: Open SeatGeek. Search "Beatles Love". Select the soonest upcoming event. Choose 2 tickets and continue. What is the lowest price for each ticket?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Top-level UI background and structural elements for the mobile UI
# Uses provided `canvas` (PIL Image) and `draw` (ImageDraw) objects.
# Available fonts: font_sm, font_md, font_lg, font_xl

# Canvas size: 1440x2960

# Fill overall background (white / very light)
draw.rectangle([(0, 0), (1440, 2960)], fill=(255, 255, 255))

# Status bar area (top ~60px) - subtle light gray background
status_h = 60
draw.rectangle([(0, 0), (1440, status_h)], fill=(238, 238, 238))

# Top subtle divider under status bar
draw.line([(0, status_h), (1440, status_h)], fill=(220, 220, 220), width=1)

# Search bar background (rounded rectangle). Positioned under status bar.
search_left = 40
search_top = 60
search_right = 1400
search_bottom = 180
search_radius = 36
draw.rounded_rectangle(
    [(search_left, search_top), (search_right, search_bottom)],
    radius=search_radius,
    fill=(243, 243, 243),
    outline=(227, 227, 227),
    width=1
)

# Subtle shadow line below search bar
shadow_y = search_bottom + 6
draw.line([(30, shadow_y), (1410, shadow_y)], fill=(240, 240, 240), width=1)

# Section/background cards (subtle off-white rounded areas behind groups)
card_margin_h = 28
card_left = 24
card_right = 1440 - 24

# Top results card area
top_card_top = 240
top_card_bottom = 640
draw.rounded_rectangle(
    [(card_left, top_card_top), (card_right, top_card_bottom)],
    radius=12,
    fill=(255, 255, 255),
    outline=(245, 245, 245),
    width=1
)
# subtle inner divider under Top results header area
draw.line([(card_left + 24, top_card_top + 68), (card_right - 24, top_card_top + 68)], fill=(235, 235, 235), width=1)

# Performers card area
perf_card_top = 680
perf_card_bottom = 1120
draw.rounded_rectangle(
    [(card_left, perf_card_top), (card_right, perf_card_bottom)],
    radius=12,
    fill=(255, 255, 255),
    outline=(245, 245, 245),
    width=1
)
draw.line([(card_left + 24, perf_card_top + 78), (card_right - 24, perf_card_top + 78)], fill=(235, 235, 235), width=1)

# Events card area
events_card_top = 1160
events_card_bottom = 1760
draw.rounded_rectangle(
    [(card_left, events_card_top), (card_right, events_card_bottom)],
    radius=12,
    fill=(255, 255, 255),
    outline=(245, 245, 245),
    width=1
)
draw.line([(card_left + 24, events_card_top + 78), (card_right - 24, events_card_top + 78)], fill=(235, 235, 235), width=1)

# Recent searches area background (near bottom, above bottom nav)
recent_top = 2320
recent_bottom = 2760
draw.rectangle([(0, recent_top), (1440, recent_bottom)], fill=(255, 255, 255))
draw.line([(24, recent_top), (1416, recent_top)], fill=(230, 230, 230), width=1)

# Full-width separators between main sections (subtle)
separators = [
    search_bottom + 40,  # below search area
    top_card_bottom + 20,
    perf_card_bottom + 20,
    events_card_bottom + 20,
    recent_top
]
for y in separators:
    draw.line([(24, y), (1416, y)], fill=(235, 235, 235), width=1)

# Bottom navigation background and top divider (leave icons empty)
bottom_nav_top = 2792
draw.rectangle([(0, bottom_nav_top), (1440, 2960)], fill=(255, 255, 255))
# top border for bottom nav
draw.line([(0, bottom_nav_top), (1440, bottom_nav_top)], fill=(225, 225, 225), width=2)
# subtle shadow above the nav to lift it
draw.line([(0, bottom_nav_top - 4), (1440, bottom_nav_top - 4)], fill=(245, 245, 245), width=1)

# Additional subtle vertical margins on the left to mimic app safe area
draw.rectangle([(0, 0), (24, 2960)], fill=(255, 255, 255))
draw.rectangle([(1440 - 24, 0), (1440, 2960)], fill=(255, 255, 255))

# End of structural/background drawing

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_04_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-7/00_icon_Performers.png
try:
    _c0 = get_crop(0, 1440, 179)
    canvas.paste(_c0, (0, 1217), _c0)
except Exception:
    pass
layout["Performers"] = [0, 1217, 1440, 1396]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_04_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-7/01_icon_No_events.png
try:
    _c1 = get_crop(1, 1440, 179)
    canvas.paste(_c1, (0, 1575), _c1)
except Exception:
    pass
layout["No_events"] = [0, 1575, 1440, 1754]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_04_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-7/02_icon_Top_results.png
try:
    _c2 = get_crop(2, 1440, 179)
    canvas.paste(_c2, (0, 471), _c2)
except Exception:
    pass
layout["Top_results"] = [0, 471, 1440, 650]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_04_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-7/03_icon_No_events.png
try:
    _c3 = get_crop(3, 1440, 179)
    canvas.paste(_c3, (0, 1396), _c3)
except Exception:
    pass
layout["No_events"] = [0, 1396, 1440, 1575]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_04_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-7/04_icon_Events.png
try:
    _c4 = get_crop(4, 1440, 179)
    canvas.paste(_c4, (0, 1963), _c4)
except Exception:
    pass
layout["Events"] = [0, 1963, 1440, 2142]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_04_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-7/05_icon_No_events.png
try:
    _c5 = get_crop(5, 1440, 179)
    canvas.paste(_c5, (0, 829), _c5)
except Exception:
    pass
layout["No_events"] = [0, 829, 1440, 1008]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_04_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-7/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 42, 70)
    canvas.paste(_c6, (1156, 0), _c6)
except Exception:
    pass
layout["icon_6"] = [1156, 0, 1198, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_04_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-7/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 60, 58)
    canvas.paste(_c7, (245, 5), _c7)
except Exception:
    pass
layout["icon_7"] = [245, 5, 305, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_04_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-7/08_icon_Cirque_du_Soleil_The_Beatles.png
try:
    _c8 = get_crop(8, 1440, 179)
    canvas.paste(_c8, (0, 471), _c8)
except Exception:
    pass
layout["Cirque_du_Soleil:_The_Bea"] = [0, 471, 1440, 650]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_04_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-7/09_icon_TL_DL_rim.png
try:
    _c9 = get_crop(9, 288, 162)
    canvas.paste(_c9, (288, 2792), _c9)
except Exception:
    pass
layout["TL^_DL~rim"] = [288, 2792, 576, 2954]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_04_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-7/10_icon_Las_Vegas_NV.png
try:
    _c10 = get_crop(10, 1440, 179)
    canvas.paste(_c10, (0, 2142), _c10)
except Exception:
    pass
layout["Las_Vegas,_NV"] = [0, 2142, 1440, 2321]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_04_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-7/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 92, 68)
    canvas.paste(_c11, (1220, 0), _c11)
except Exception:
    pass
layout["icon_11"] = [1220, 0, 1312, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_04_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-7/12_icon_Las_Vegas_NV.png
try:
    _c12 = get_crop(12, 1440, 179)
    canvas.paste(_c12, (0, 1963), _c12)
except Exception:
    pass
layout["Las_Vegas,_NV"] = [0, 1963, 1440, 2142]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_04_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-7/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 47, 56)
    canvas.paste(_c13, (318, 7), _c13)
except Exception:
    pass
layout["icon_13"] = [318, 7, 365, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_04_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-7/14_icon_Las_Vegas_NV.png
try:
    _c14 = get_crop(14, 1440, 179)
    canvas.paste(_c14, (0, 2321), _c14)
except Exception:
    pass
layout["Las_Vegas,_NV"] = [0, 2321, 1440, 2500]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_04_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-7/15_icon_5.06_my.png
try:
    _c15 = get_crop(15, 45, 59)
    canvas.paste(_c15, (186, 4), _c15)
except Exception:
    pass
layout["5.06_my"] = [186, 4, 231, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_04_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-7/16_icon_Tomorrow.png
try:
    _c16 = get_crop(16, 1440, 179)
    canvas.paste(_c16, (0, 2321), _c16)
except Exception:
    pass
layout["Tomorrow"] = [0, 2321, 1440, 2500]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_04_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-7/17_icon_icon_17.png
try:
    _c17 = get_crop(17, 44, 63)
    canvas.paste(_c17, (1326, 3), _c17)
except Exception:
    pass
layout["icon_17"] = [1326, 3, 1370, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_04_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-7/18_icon_Tomorrow.png
try:
    _c18 = get_crop(18, 1440, 179)
    canvas.paste(_c18, (0, 2142), _c18)
except Exception:
    pass
layout["Tomorrow"] = [0, 2142, 1440, 2321]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_04_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-7/19_icon_Ocz.png
try:
    _c19 = get_crop(19, 288, 168)
    canvas.paste(_c19, (576, 2792), _c19)
except Exception:
    pass
layout["Ocz~"] = [576, 2792, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_04_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-7/20_icon_Las.png
try:
    _c20 = get_crop(20, 1440, 179)
    canvas.paste(_c20, (0, 650), _c20)
except Exception:
    pass
layout["Las"] = [0, 650, 1440, 829]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_04_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-7/21_icon_5.06_my.png
try:
    _c21 = get_crop(21, 168, 144)
    canvas.paste(_c21, (48, 120), _c21)
except Exception:
    pass
layout["5.06_my"] = [48, 120, 216, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_04_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-7/22_icon_TL_DL_rim.png
try:
    _c22 = get_crop(22, 288, 168)
    canvas.paste(_c22, (0, 2792), _c22)
except Exception:
    pass
layout["TL^_DL~rim"] = [0, 2792, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_04_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-7/23_icon_Clear.png
try:
    _c23 = get_crop(23, 144, 144)
    canvas.paste(_c23, (1248, 120), _c23)
except Exception:
    pass
layout["Clear"] = [1248, 120, 1392, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_04_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-7/24_icon_Ocz.png
try:
    _c24 = get_crop(24, 288, 168)
    canvas.paste(_c24, (864, 2792), _c24)
except Exception:
    pass
layout["Ocz~"] = [864, 2792, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_04_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-7/25_icon_91_events.png
try:
    _c25 = get_crop(25, 1440, 179)
    canvas.paste(_c25, (0, 650), _c25)
except Exception:
    pass
layout["91_events"] = [0, 650, 1440, 829]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_04_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-7/26_icon_Las.png
try:
    _c26 = get_crop(26, 1440, 179)
    canvas.paste(_c26, (0, 829), _c26)
except Exception:
    pass
layout["Las"] = [0, 829, 1440, 1008]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_04_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-7/27_icon_Account.png
try:
    _c27 = get_crop(27, 288, 168)
    canvas.paste(_c27, (1152, 2792), _c27)
except Exception:
    pass
layout["Account"] = [1152, 2792, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_04_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-7/28_icon_Cirque_du_Soleil_The_Beatles.png
try:
    _c28 = get_crop(28, 1440, 179)
    canvas.paste(_c28, (0, 1217), _c28)
except Exception:
    pass
layout["Cirque_du_Soleil:_The_Bea"] = [0, 1217, 1440, 1396]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_04_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-7/29_icon_icon_29.png
try:
    _c29 = get_crop(29, 41, 55)
    canvas.paste(_c29, (387, 8), _c29)
except Exception:
    pass
layout["icon_29"] = [387, 8, 428, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_04_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-7/30_icon_5.06_my.png
try:
    _c30 = get_crop(30, 51, 61)
    canvas.paste(_c30, (117, 2), _c30)
except Exception:
    pass
layout["5.06_my"] = [117, 2, 168, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_04_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-7/31_text_Beatles_Love.png
try:
    _c31 = get_crop(31, 1032, 144)
    canvas.paste(_c31, (216, 120), _c31)
except Exception:
    pass
layout["Beatles_Love"] = [216, 120, 1248, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_04_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-7/32_text_Top_results.png
try:
    _c32 = get_crop(32, 295, 72)
    canvas.paste(_c32, (40, 373), _c32)
except Exception:
    pass
layout["Top_results"] = [40, 373, 335, 445]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_04_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-7/33_text_Performers.png
try:
    _c33 = get_crop(33, 293, 54)
    canvas.paste(_c33, (44, 1122), _c33)
except Exception:
    pass
layout["Performers"] = [44, 1122, 337, 1176]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_04_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-7/34_text_The_LOVE_Beatles_Tribute.png
try:
    _c34 = get_crop(34, 1440, 179)
    canvas.paste(_c34, (0, 1396), _c34)
except Exception:
    pass
layout["The_LOVE_Beatles_Tribute"] = [0, 1396, 1440, 1575]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_04_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-7/35_text_No_events.png
try:
    _c35 = get_crop(35, 201, 43)
    canvas.paste(_c35, (239, 1497), _c35)
except Exception:
    pass
layout["No_events"] = [239, 1497, 440, 1540]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_04_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-7/36_text_Love_Songs.png
try:
    _c36 = get_crop(36, 252, 62)
    canvas.paste(_c36, (235, 1610), _c36)
except Exception:
    pass
layout["Love_Songs"] = [235, 1610, 487, 1672]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_04_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-7/37_text_The_Beatles_Vol._7_Tribute_Show.png
try:
    _c37 = get_crop(37, 1440, 179)
    canvas.paste(_c37, (0, 1575), _c37)
except Exception:
    pass
layout["The_Beatles_Vol._7_Tribut"] = [0, 1575, 1440, 1754]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_04_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-7/38_text_No_events.png
try:
    _c38 = get_crop(38, 201, 40)
    canvas.paste(_c38, (239, 1678), _c38)
except Exception:
    pass
layout["No_events"] = [239, 1678, 440, 1718]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_04_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-7/39_text_Events.png
try:
    _c39 = get_crop(39, 181, 57)
    canvas.paste(_c39, (43, 1868), _c39)
except Exception:
    pass
layout["Events"] = [43, 1868, 224, 1925]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_04_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-7/40_text_Recent_searches.png
try:
    _c40 = get_crop(40, 288, 168)
    canvas.paste(_c40, (0, 2792), _c40)
except Exception:
    pass
layout["Recent_searches"] = [0, 2792, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_04_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-7/41_text_TL_DL_rim.png
try:
    _c41 = get_crop(41, 288, 162)
    canvas.paste(_c41, (288, 2792), _c41)
except Exception:
    pass
layout["TL^_DL~rim"] = [288, 2792, 576, 2954]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_04_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-7/42_text_L.png
try:
    _c42 = get_crop(42, 129, 32)
    canvas.paste(_c42, (532, 2769), _c42)
except Exception:
    pass
layout["~+L^"] = [532, 2769, 661, 2801]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_04_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-7/43_text_Ocz.png
try:
    _c43 = get_crop(43, 288, 168)
    canvas.paste(_c43, (576, 2792), _c43)
except Exception:
    pass
layout["Ocz~"] = [576, 2792, 864, 2960]
