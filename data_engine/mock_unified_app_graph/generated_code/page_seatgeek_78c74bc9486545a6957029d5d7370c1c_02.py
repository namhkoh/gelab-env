# page_id: page_seatgeek_78c74bc9486545a6957029d5d7370c1c_02
# screenshot: 2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-5.png
# step_index: 2/9
# task: Open SeatGeek and search by category "Comedy". Select the first one in New York and check its information. Track the performer of this event.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Background and structural UI elements for the SeatGeek-like mobile page
# (Uses provided `canvas` (1440x2960 RGB) and `draw` ImageDraw)

# Fill overall background (ensure clean white)
draw.rectangle([(0, 0), (1440, 2960)], fill="#ffffff")

# 1) Status bar area at top (~72px)
status_h = 72
draw.rectangle([(0, 0), (1440, status_h)], fill="#f3f4f5")

# subtle thin divider under status bar
draw.line([(0, status_h), (1440, status_h)], fill="#e2e2e2", width=1)

# 2) Header / toolbar background (large white area with subtle shadow/divider)
header_top = status_h
header_bottom = 320
draw.rectangle([(0, header_top), (1440, header_bottom)], fill="#ffffff")
# header bottom shadow / divider
draw.line([(24, header_bottom), (1416, header_bottom)], fill="#e9e9e9", width=2)

# 3) Large event row card background just under header (light white card with subtle border)
event_card = (24, 300, 1416, 380)
draw.rounded_rectangle(event_card, radius=10, fill="#ffffff", outline="#f0f0f0", width=1)

# subtle divider below this event row
draw.line([(24, 384), (1416, 384)], fill="#f1f1f1", width=1)

# 4) Divider between sections (between main event row and 'Recently viewed' area)
draw.line([(24, 520), (1416, 520)], fill="#efefef", width=1)

# 5) Recently viewed cards backgrounds (rounded dark rectangles)
recent_cards = [
    (48, 812, 48 + 462, 812 + 533),   # left card
    (546, 812, 546 + 462, 812 + 519), # middle card
    (1044, 812, 1044 + 396, 812 + 519) # right card
]
for (x1, y1, x2, y2) in recent_cards:
    # shadow
    draw.rounded_rectangle([x1+4, y1+6, x2+6, y2+10], radius=20, fill="#e9e9e9")
    # dark card background (images will be pasted on top)
    draw.rounded_rectangle([x1, y1, x2, y2], radius=20, fill="#0b0b0b")

# separator line under recently viewed section title/cards
draw.line([(24, 1070), (1416, 1070)], fill="#f2f2f2", width=1)

# 6) Browse by category section title area
# (we keep background white but add a separator above and below)
draw.line([(24, 1500), (1416, 1500)], fill="#f3f3f3", width=1)

# 7) Category card backgrounds (rounded dark rectangles)
category_cards = [
    (48, 1580, 48 + 462, 1580 + 312),   # Sports
    (546, 1580, 546 + 462, 1580 + 312), # Concerts
    (1044, 1580, 1044 + 396, 1580 + 312) # Broadway
]
for (x1, y1, x2, y2) in category_cards:
    # subtle shadow
    draw.rounded_rectangle([x1+3, y1+4, x2+5, y2+8], radius=16, fill="#ededed")
    draw.rounded_rectangle([x1, y1, x2, y2], radius=16, fill="#0b0b0b")

# bottom separator for category area
draw.line([(24, 1910), (1416, 1910)], fill="#efefef", width=1)

# 8) Just announced section separator and card backgrounds
draw.line([(24, 1980), (1416, 1980)], fill="#f3f3f3", width=1)

just_announced_cards = [
    (48, 2127, 48 + 462, 2127 + 519),   # left
    (546, 2127, 546 + 462, 2127 + 519)  # right
]
for (x1, y1, x2, y2) in just_announced_cards:
    draw.rounded_rectangle([x1+3, y1+6, x2+6, y2+10], radius=18, fill="#eaeaea")
    draw.rounded_rectangle([x1, y1, x2, y2], radius=18, fill="#0b0b0b")

# subtle divider below just announced area
draw.line([(24, 2650), (1416, 2650)], fill="#f2f2f2", width=1)

# 9) Bottom navigation bar area (with top divider and subtle shadow)
nav_top = 2792
draw.rectangle([(0, nav_top), (1440, 2960)], fill="#ffffff")
# top divider line
draw.line([(0, nav_top), (1440, nav_top)], fill="#e6e6e6", width=1)
# slight shadow above nav for separation
draw.rectangle([(0, nav_top-6), (1440, nav_top)], fill="#fafafa")

# 10) Final subtle vertical gutters on left and right to match app margins
# (light thin lines to mimic app content inset)
draw.line([(24, header_bottom), (24, 2700)], fill="#ffffff", width=1)
draw.line([(1416, header_bottom), (1416, 2700)], fill="#ffffff", width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_02_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-5/00_icon_Broadway.png
try:
    _c0 = get_crop(0, 396, 312)
    canvas.paste(_c0, (1044, 1580), _c0)
except Exception:
    pass
layout["Broadway"] = [1044, 1580, 1440, 1892]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_02_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-5/01_icon_Concerts.png
try:
    _c1 = get_crop(1, 462, 312)
    canvas.paste(_c1, (546, 1580), _c1)
except Exception:
    pass
layout["Concerts"] = [546, 1580, 1008, 1892]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_02_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-5/02_icon_Sports.png
try:
    _c2 = get_crop(2, 462, 312)
    canvas.paste(_c2, (48, 1580), _c2)
except Exception:
    pass
layout["Sports"] = [48, 1580, 510, 1892]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_02_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-5/03_icon_884.png
try:
    _c3 = get_crop(3, 105, 151)
    canvas.paste(_c3, (1335, 379), _c3)
except Exception:
    pass
layout["884"] = [1335, 379, 1440, 530]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_02_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-5/04_icon_S273.png
try:
    _c4 = get_crop(4, 462, 533)
    canvas.paste(_c4, (48, 812), _c4)
except Exception:
    pass
layout["S273+"] = [48, 812, 510, 1345]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_02_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-5/05_icon_8.27_my.png
try:
    _c5 = get_crop(5, 56, 58)
    canvas.paste(_c5, (182, 4), _c5)
except Exception:
    pass
layout["8.27_my"] = [182, 4, 238, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_02_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-5/06_icon_Viom_Il.png
try:
    _c6 = get_crop(6, 288, 168)
    canvas.paste(_c6, (864, 2792), _c6)
except Exception:
    pass
layout["Viom_~Il"] = [864, 2792, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_02_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-5/07_icon_8.27_my.png
try:
    _c7 = get_crop(7, 53, 56)
    canvas.paste(_c7, (115, 6), _c7)
except Exception:
    pass
layout["8.27_my"] = [115, 6, 168, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_02_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-5/08_icon_884.png
try:
    _c8 = get_crop(8, 96, 62)
    canvas.paste(_c8, (1217, 2), _c8)
except Exception:
    pass
layout["884"] = [1217, 2, 1313, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_02_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-5/09_icon_S120.png
try:
    _c9 = get_crop(9, 396, 519)
    canvas.paste(_c9, (1044, 812), _c9)
except Exception:
    pass
layout["S120+"] = [1044, 812, 1440, 1331]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_02_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-5/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 50, 54)
    canvas.paste(_c10, (316, 6), _c10)
except Exception:
    pass
layout["icon_10"] = [316, 6, 366, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_02_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-5/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 48, 56)
    canvas.paste(_c11, (1321, 5), _c11)
except Exception:
    pass
layout["icon_11"] = [1321, 5, 1369, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_02_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-5/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 43, 66)
    canvas.paste(_c12, (1156, 0), _c12)
except Exception:
    pass
layout["icon_12"] = [1156, 0, 1199, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_02_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-5/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 52, 55)
    canvas.paste(_c13, (247, 6), _c13)
except Exception:
    pass
layout["icon_13"] = [247, 6, 299, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_02_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-5/14_icon_884.png
try:
    _c14 = get_crop(14, 144, 240)
    canvas.paste(_c14, (1260, 72), _c14)
except Exception:
    pass
layout["884"] = [1260, 72, 1404, 312]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_02_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-5/15_icon_Viom_Il.png
try:
    _c15 = get_crop(15, 288, 168)
    canvas.paste(_c15, (1152, 2792), _c15)
except Exception:
    pass
layout["Viom_~Il"] = [1152, 2792, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_02_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-5/16_icon_Cnawo.png
try:
    _c16 = get_crop(16, 288, 168)
    canvas.paste(_c16, (288, 2792), _c16)
except Exception:
    pass
layout["Cnawo"] = [288, 2792, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_02_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-5/17_icon_Tickets.png
try:
    _c17 = get_crop(17, 288, 168)
    canvas.paste(_c17, (576, 2792), _c17)
except Exception:
    pass
layout["Tickets"] = [576, 2792, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_02_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-5/18_icon_Cnawo.png
try:
    _c18 = get_crop(18, 288, 162)
    canvas.paste(_c18, (0, 2792), _c18)
except Exception:
    pass
layout["Cnawo"] = [0, 2792, 288, 2954]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_02_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-5/19_icon_View_all.png
try:
    _c19 = get_crop(19, 135, 132)
    canvas.paste(_c19, (1133, 388), _c19)
except Exception:
    pass
layout["View_all"] = [1133, 388, 1268, 520]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_02_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-5/20_icon_icon_20.png
try:
    _c20 = get_crop(20, 48, 55)
    canvas.paste(_c20, (383, 5), _c20)
except Exception:
    pass
layout["icon_20"] = [383, 5, 431, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_02_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-5/21_text_New_York_NY.png
try:
    _c21 = get_crop(21, 384, 68)
    canvas.paste(_c21, (48, 133), _c21)
except Exception:
    pass
layout["New_York,_NY"] = [48, 133, 432, 201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_02_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-5/22_text_date.png
try:
    _c22 = get_crop(22, 117, 52)
    canvas.paste(_c22, (134, 208), _c22)
except Exception:
    pass
layout["date"] = [134, 208, 251, 260]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_02_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-5/23_text_Hadestown.png
try:
    _c23 = get_crop(23, 251, 49)
    canvas.paste(_c23, (229, 389), _c23)
except Exception:
    pass
layout["Hadestown"] = [229, 389, 480, 438]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_02_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-5/24_text_Mar_20.png
try:
    _c24 = get_crop(24, 162, 52)
    canvas.paste(_c24, (232, 463), _c24)
except Exception:
    pass
layout["Mar_20"] = [232, 463, 394, 515]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_02_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-5/25_text_Walter_Kerr_Theatre.png
try:
    _c25 = get_crop(25, 1309, 234)
    canvas.paste(_c25, (0, 333), _c25)
except Exception:
    pass
layout["Walter_Kerr_Theatre"] = [0, 333, 1309, 567]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_02_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-5/26_text_Recently_viewed_events.png
try:
    _c26 = get_crop(26, 72, 72)
    canvas.paste(_c26, (408, 836), _c26)
except Exception:
    pass
layout["Recently_viewed_events"] = [408, 836, 480, 908]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_02_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-5/27_text_View_all.png
try:
    _c27 = get_crop(27, 264, 183)
    canvas.paste(_c27, (1176, 629), _c27)
except Exception:
    pass
layout["View_all"] = [1176, 629, 1440, 812]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_02_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-5/28_text_S155.png
try:
    _c28 = get_crop(28, 112, 55)
    canvas.paste(_c28, (591, 1038), _c28)
except Exception:
    pass
layout["S155"] = [591, 1038, 703, 1093]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_02_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-5/29_text_Drake_Rescheduled.png
try:
    _c29 = get_crop(29, 462, 533)
    canvas.paste(_c29, (48, 812), _c29)
except Exception:
    pass
layout["Drake_(Rescheduled"] = [48, 812, 510, 1345]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_02_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-5/30_text_Hadestown.png
try:
    _c30 = get_crop(30, 462, 519)
    canvas.paste(_c30, (546, 812), _c30)
except Exception:
    pass
layout["Hadestown"] = [546, 812, 1008, 1331]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_02_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-5/31_text_Matt_Rife.png
try:
    _c31 = get_crop(31, 204, 49)
    canvas.paste(_c31, (1041, 1166), _c31)
except Exception:
    pass
layout["Matt_Rife"] = [1041, 1166, 1245, 1215]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_02_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-5/32_text_from_3_15_2024.png
try:
    _c32 = get_crop(32, 462, 533)
    canvas.paste(_c32, (48, 812), _c32)
except Exception:
    pass
layout["from_3_15_2024)"] = [48, 812, 510, 1345]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_02_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-5/33_text_Today.png
try:
    _c33 = get_crop(33, 135, 62)
    canvas.paste(_c33, (538, 1227), _c33)
except Exception:
    pass
layout["Today"] = [538, 1227, 673, 1289]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_02_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-5/34_text_2_PM.png
try:
    _c34 = get_crop(34, 103, 43)
    canvas.paste(_c34, (693, 1236), _c34)
except Exception:
    pass
layout["2_PM"] = [693, 1236, 796, 1279]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_02_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-5/35_text_Tonight.png
try:
    _c35 = get_crop(35, 161, 54)
    canvas.paste(_c35, (1038, 1233), _c35)
except Exception:
    pass
layout["Tonight"] = [1038, 1233, 1199, 1287]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_02_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-5/36_text_7_PM.png
try:
    _c36 = get_crop(36, 103, 41)
    canvas.paste(_c36, (1220, 1238), _c36)
except Exception:
    pass
layout["7_PM"] = [1220, 1238, 1323, 1279]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_02_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-5/37_text_Fri_Mar_29_8_PM.png
try:
    _c37 = get_crop(37, 462, 533)
    canvas.paste(_c37, (48, 812), _c37)
except Exception:
    pass
layout["Fri,_Mar_29,_8_PM"] = [48, 812, 510, 1345]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_02_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-5/38_text_Browse_by_category.png
try:
    _c38 = get_crop(38, 462, 312)
    canvas.paste(_c38, (48, 1580), _c38)
except Exception:
    pass
layout["Browse_by_category"] = [48, 1580, 510, 1892]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_02_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-5/39_text_View_all.png
try:
    _c39 = get_crop(39, 264, 183)
    canvas.paste(_c39, (1176, 1397), _c39)
except Exception:
    pass
layout["View_all"] = [1176, 1397, 1440, 1580]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_02_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-5/40_text_Just_announced.png
try:
    _c40 = get_crop(40, 72, 72)
    canvas.paste(_c40, (408, 2151), _c40)
except Exception:
    pass
layout["Just_announced"] = [408, 2151, 480, 2223]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_02_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-5/41_text_View_all.png
try:
    _c41 = get_crop(41, 264, 183)
    canvas.paste(_c41, (1176, 1944), _c41)
except Exception:
    pass
layout["View_all"] = [1176, 1944, 1440, 2127]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_02_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-5/42_text_BARCLAYS_CEMTE.png
try:
    _c42 = get_crop(42, 78, 16)
    canvas.paste(_c42, (725, 2140), _c42)
except Exception:
    pass
layout["BARCLAYS_CEMTE"] = [725, 2140, 803, 2156]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_02_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-5/43_text_S116.png
try:
    _c43 = get_crop(43, 126, 49)
    canvas.paste(_c43, (95, 2357), _c43)
except Exception:
    pass
layout["S116+"] = [95, 2357, 221, 2406]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_02_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-5/44_text_S91.png
try:
    _c44 = get_crop(44, 104, 45)
    canvas.paste(_c44, (595, 2360), _c44)
except Exception:
    pass
layout["S91+"] = [595, 2360, 699, 2405]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_02_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-5/45_text_Matt_Rife_18.png
try:
    _c45 = get_crop(45, 462, 519)
    canvas.paste(_c45, (48, 2127), _c45)
except Exception:
    pass
layout["Matt_Rife_(18+)"] = [48, 2127, 510, 2646]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_02_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-5/46_text_Bulls_at_Nets.png
try:
    _c46 = get_crop(46, 462, 519)
    canvas.paste(_c46, (546, 2127), _c46)
except Exception:
    pass
layout["Bulls_at_Nets"] = [546, 2127, 1008, 2646]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_02_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-5/47_text_Sun_Jul_28_7.30_PM.png
try:
    _c47 = get_crop(47, 462, 519)
    canvas.paste(_c47, (48, 2127), _c47)
except Exception:
    pass
layout["Sun,_Jul_28,7.30_PM"] = [48, 2127, 510, 2646]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_02_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-5/48_text_Fri_Mar_29_7.30_PM.png
try:
    _c48 = get_crop(48, 462, 519)
    canvas.paste(_c48, (546, 2127), _c48)
except Exception:
    pass
layout["Fri,_Mar_29,7.30_PM"] = [546, 2127, 1008, 2646]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_02_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-5/49_text_Viom_Il.png
try:
    _c49 = get_crop(49, 288, 168)
    canvas.paste(_c49, (1152, 2792), _c49)
except Exception:
    pass
layout["Viom_~Il"] = [1152, 2792, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_02_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-5/50_clickable_Tracking.png
try:
    _c50 = get_crop(50, 72, 72)
    canvas.paste(_c50, (906, 836), _c50)
except Exception:
    pass
layout["Tracking"] = [906, 836, 978, 908]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_02_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-5/51_clickable_Tracking.png
try:
    _c51 = get_crop(51, 72, 72)
    canvas.paste(_c51, (906, 2151), _c51)
except Exception:
    pass
layout["Tracking"] = [906, 2151, 978, 2223]
