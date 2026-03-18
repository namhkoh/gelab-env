# page_id: page_seatgeek_78c74bc9486545a6957029d5d7370c1c_03
# screenshot: 2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-6.png
# step_index: 3/9
# task: Open SeatGeek and search by category "Comedy". Select the first one in New York and check its information. Track the performer of this event.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Paint overall background
draw.rectangle([(0, 0), canvas.size], fill="#ffffff")

# Status bar (top area)
status_h = 72
draw.rectangle([(0, 0), (canvas.width, status_h)], fill="#efefef")

# Header / toolbar background (just below status bar)
header_top = status_h
header_bottom = 160
draw.rectangle([(0, header_top), (canvas.width, header_bottom)], fill="#ffffff")

# Subtle bottom divider under header
draw.line([(24, header_bottom), (canvas.width - 24, header_bottom)], fill="#e6e6e6", width=2)

# Section separators (light horizontal rules between major groups)
for y in (1040, 1790, 2430):
    draw.line([(24, y), (canvas.width - 24, y)], fill="#eeeef0", width=2)

# Card shadow + thumbnail card backgrounds (rows of event thumbnails)
# Each tuple: (x, y, width, height)
thumbnail_cards = [
    (48, 495, 462, 519),
    (546, 495, 462, 533),
    (1044, 495, 396, 533),

    (48, 1261, 462, 533),
    (546, 1261, 462, 519),
    (1044, 1261, 396, 533),

    (48, 2029, 462, 519),
    (546, 2029, 462, 533),
    (1044, 2029, 396, 519),
]

radius = 28
shadow_offset = (6, 6)
for (x, y, w, h) in thumbnail_cards:
    # shadow
    sx0 = x + shadow_offset[0]
    sy0 = y + shadow_offset[1]
    sx1 = x + w + shadow_offset[0]
    sy1 = y + h + shadow_offset[1]
    draw.rounded_rectangle([(sx0, sy0), (sx1, sy1)], radius=radius, fill="#e9ecef")
    # main dark thumbnail background
    draw.rounded_rectangle([(x, y), (x + w, y + h)], radius=radius, fill="#0f1720")

# Additional subtle card backgrounds for any lower "Comedy" row (structural only)
# These are placed lower on the screen as backgrounds for the next horizontal list.
comedy_row_y = 2600
comedy_card_w = 420
comedy_card_h = 460
left_margins = [48, 546, 1044]
for ix, lx in enumerate(left_margins):
    cy = comedy_row_y
    # keep within canvas
    if cy + comedy_card_h + 10 > canvas.height:
        cy = canvas.height - comedy_card_h - 24
    # shadow
    sx0 = lx + 5
    sy0 = cy + 6
    sx1 = lx + comedy_card_w + 5
    sy1 = cy + comedy_card_h + 6
    draw.rounded_rectangle([(sx0, sy0), (sx1, sy1)], radius=24, fill="#eceff1")
    # card fill (darker to simulate image card background)
    draw.rounded_rectangle([(lx, cy), (lx + comedy_card_w, cy + comedy_card_h)], radius=24, fill="#0d1114")

# Thin dividers between each card row area (within content area)
# These give structure below the thumbnail rows where titles/dates will be placed.
inner_dividers = [ (48, 1020, canvas.width - 48, 1020),
                   (48, 1770, canvas.width - 48, 1770),
                   (48, 2410, canvas.width - 48, 2410) ]
for x0, y0, x1, y1 in inner_dividers:
    draw.line([(x0, y0), (x1, y1)], fill="#f1f1f3", width=1)

# Light background band behind section titles to subtly separate header area
title_band_top = header_bottom + 12
title_band_bottom = title_band_top + 64
draw.rectangle([(0, title_band_top), (canvas.width, title_band_bottom)], fill="#ffffff")

# Final subtle vignette line near bottom to ground the page
draw.line([(24, canvas.height - 120), (canvas.width - 24, canvas.height - 120)], fill="#f0f0f2", width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_03_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-6/00_icon_S69.png
try:
    _c0 = get_crop(0, 462, 533)
    canvas.paste(_c0, (546, 495), _c0)
except Exception:
    pass
layout["S69+"] = [546, 495, 1008, 1028]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_03_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-6/01_icon_S273.png
try:
    _c1 = get_crop(1, 462, 533)
    canvas.paste(_c1, (48, 1261), _c1)
except Exception:
    pass
layout["S273+"] = [48, 1261, 510, 1794]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_03_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-6/02_icon_8.28_my.png
try:
    _c2 = get_crop(2, 54, 58)
    canvas.paste(_c2, (114, 4), _c2)
except Exception:
    pass
layout["8.28_my"] = [114, 4, 168, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_03_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-6/03_icon_8.28_my.png
try:
    _c3 = get_crop(3, 55, 58)
    canvas.paste(_c3, (182, 4), _c3)
except Exception:
    pass
layout["8.28_my"] = [182, 4, 237, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_03_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-6/04_icon_icon_4.png
try:
    _c4 = get_crop(4, 51, 55)
    canvas.paste(_c4, (316, 5), _c4)
except Exception:
    pass
layout["icon_4"] = [316, 5, 367, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_03_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-6/05_icon_888.png
try:
    _c5 = get_crop(5, 96, 60)
    canvas.paste(_c5, (1217, 3), _c5)
except Exception:
    pass
layout["888"] = [1217, 3, 1313, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_03_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-6/06_icon_S164.png
try:
    _c6 = get_crop(6, 462, 519)
    canvas.paste(_c6, (48, 495), _c6)
except Exception:
    pass
layout["S164+"] = [48, 495, 510, 1014]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_03_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-6/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 49, 57)
    canvas.paste(_c7, (1321, 5), _c7)
except Exception:
    pass
layout["icon_7"] = [1321, 5, 1370, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_03_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-6/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 48, 55)
    canvas.paste(_c8, (384, 5), _c8)
except Exception:
    pass
layout["icon_8"] = [384, 5, 432, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_03_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-6/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 54, 57)
    canvas.paste(_c9, (246, 5), _c9)
except Exception:
    pass
layout["icon_9"] = [246, 5, 300, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_03_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-6/10_icon_8.28_my.png
try:
    _c10 = get_crop(10, 144, 240)
    canvas.paste(_c10, (0, 72), _c10)
except Exception:
    pass
layout["8.28_my"] = [0, 72, 144, 312]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_03_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-6/11_icon_888.png
try:
    _c11 = get_crop(11, 144, 240)
    canvas.paste(_c11, (1260, 72), _c11)
except Exception:
    pass
layout["888"] = [1260, 72, 1404, 312]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_03_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-6/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 45, 59)
    canvas.paste(_c12, (1155, 5), _c12)
except Exception:
    pass
layout["icon_12"] = [1155, 5, 1200, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_03_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-6/13_icon_MORM.png
try:
    _c13 = get_crop(13, 396, 519)
    canvas.paste(_c13, (1044, 2029), _c13)
except Exception:
    pass
layout["MORM"] = [1044, 2029, 1440, 2548]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_03_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-6/14_icon_8_PM.png
try:
    _c14 = get_crop(14, 72, 72)
    canvas.paste(_c14, (906, 2821), _c14)
except Exception:
    pass
layout["8_PM"] = [906, 2821, 978, 2893]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_03_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-6/15_icon_View_all.png
try:
    _c15 = get_crop(15, 48, 91)
    canvas.paste(_c15, (1392, 515), _c15)
except Exception:
    pass
layout["View_all"] = [1392, 515, 1440, 606]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_03_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-6/16_icon_Comedy.png
try:
    _c16 = get_crop(16, 72, 72)
    canvas.paste(_c16, (408, 2821), _c16)
except Exception:
    pass
layout["Comedy"] = [408, 2821, 480, 2893]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_03_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-6/17_icon_View_all.png
try:
    _c17 = get_crop(17, 264, 183)
    canvas.paste(_c17, (1176, 2614), _c17)
except Exception:
    pass
layout["View_all"] = [1176, 2614, 1440, 2797]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_03_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-6/18_icon_8.28_my.png
try:
    _c18 = get_crop(18, 100, 62)
    canvas.paste(_c18, (8, 0), _c18)
except Exception:
    pass
layout["8.28_my"] = [8, 0, 108, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_03_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-6/19_icon_S326.png
try:
    _c19 = get_crop(19, 396, 533)
    canvas.paste(_c19, (1044, 495), _c19)
except Exception:
    pass
layout["S326+"] = [1044, 495, 1440, 1028]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_03_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-6/20_icon_S326.png
try:
    _c20 = get_crop(20, 72, 72)
    canvas.paste(_c20, (906, 519), _c20)
except Exception:
    pass
layout["S326+"] = [906, 519, 978, 591]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_03_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-6/21_icon_View_all.png
try:
    _c21 = get_crop(21, 58, 103)
    canvas.paste(_c21, (1382, 1274), _c21)
except Exception:
    pass
layout["View_all"] = [1382, 1274, 1440, 1377]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_03_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-6/22_icon_S371.png
try:
    _c22 = get_crop(22, 462, 519)
    canvas.paste(_c22, (546, 1261), _c22)
except Exception:
    pass
layout["S371+"] = [546, 1261, 1008, 1780]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_03_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-6/23_text_Browse_by_category.png
try:
    _c23 = get_crop(23, 72, 72)
    canvas.paste(_c23, (408, 519), _c23)
except Exception:
    pass
layout["Browse_by_category"] = [408, 519, 480, 591]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_03_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-6/24_text_Sports.png
try:
    _c24 = get_crop(24, 182, 74)
    canvas.paste(_c24, (39, 374), _c24)
except Exception:
    pass
layout["Sports"] = [39, 374, 221, 448]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_03_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-6/25_text_View_all.png
try:
    _c25 = get_crop(25, 264, 183)
    canvas.paste(_c25, (1176, 312), _c25)
except Exception:
    pass
layout["View_all"] = [1176, 312, 1440, 495]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_03_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-6/26_text_Braves_at_Phillies.png
try:
    _c26 = get_crop(26, 462, 519)
    canvas.paste(_c26, (48, 495), _c26)
except Exception:
    pass
layout["Braves_at_Phillies"] = [48, 495, 510, 1014]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_03_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-6/27_text_Inter_Miami_CF_at.png
try:
    _c27 = get_crop(27, 462, 533)
    canvas.paste(_c27, (546, 495), _c27)
except Exception:
    pass
layout["Inter_Miami_CF_at"] = [546, 495, 1008, 1028]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_03_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-6/28_text_WWE_WrestleMan.png
try:
    _c28 = get_crop(28, 396, 533)
    canvas.paste(_c28, (1044, 495), _c28)
except Exception:
    pass
layout["WWE_WrestleMan="] = [1044, 495, 1440, 1028]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_03_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-6/29_text_Thu_Mar_28_3.05_PM.png
try:
    _c29 = get_crop(29, 462, 519)
    canvas.paste(_c29, (48, 495), _c29)
except Exception:
    pass
layout["Thu,_Mar_28,3.05_PM"] = [48, 495, 510, 1014]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_03_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-6/30_text_New_York_Red_Bulls.png
try:
    _c30 = get_crop(30, 462, 533)
    canvas.paste(_c30, (546, 495), _c30)
except Exception:
    pass
layout["New_York_Red_Bulls"] = [546, 495, 1008, 1028]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_03_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-6/31_text_40.png
try:
    _c31 = get_crop(31, 68, 48)
    canvas.paste(_c31, (1041, 911), _c31)
except Exception:
    pass
layout["40"] = [1041, 911, 1109, 959]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_03_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-6/32_text_Night_2.png
try:
    _c32 = get_crop(32, 160, 57)
    canvas.paste(_c32, (1143, 909), _c32)
except Exception:
    pass
layout["Night_2"] = [1143, 909, 1303, 966]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_03_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-6/33_text_Sat.png
try:
    _c33 = get_crop(33, 83, 43)
    canvas.paste(_c33, (542, 979), _c33)
except Exception:
    pass
layout["Sat;"] = [542, 979, 625, 1022]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_03_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-6/34_text_Mar_23.png
try:
    _c34 = get_crop(34, 147, 45)
    canvas.paste(_c34, (637, 977), _c34)
except Exception:
    pass
layout["Mar_23,"] = [637, 977, 784, 1022]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_03_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-6/35_text_2_PM.png
try:
    _c35 = get_crop(35, 103, 40)
    canvas.paste(_c35, (794, 982), _c35)
except Exception:
    pass
layout["2_PM"] = [794, 982, 897, 1022]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_03_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-6/36_text_Sun_Apr_7_7.30_PM.png
try:
    _c36 = get_crop(36, 396, 533)
    canvas.paste(_c36, (1044, 495), _c36)
except Exception:
    pass
layout["Sun,_Apr_7,_7.30_PM"] = [1044, 495, 1440, 1028]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_03_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-6/37_text_Concerts.png
try:
    _c37 = get_crop(37, 242, 57)
    canvas.paste(_c37, (44, 1140), _c37)
except Exception:
    pass
layout["Concerts"] = [44, 1140, 286, 1197]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_03_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-6/38_text_View_all.png
try:
    _c38 = get_crop(38, 264, 183)
    canvas.paste(_c38, (1176, 1078), _c38)
except Exception:
    pass
layout["View_all"] = [1176, 1078, 1440, 1261]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_03_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-6/39_text_S464.png
try:
    _c39 = get_crop(39, 396, 533)
    canvas.paste(_c39, (1044, 1261), _c39)
except Exception:
    pass
layout["S464+"] = [1044, 1261, 1440, 1794]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_03_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-6/40_text_Drake_Rescheduled.png
try:
    _c40 = get_crop(40, 462, 533)
    canvas.paste(_c40, (48, 1261), _c40)
except Exception:
    pass
layout["Drake_(Rescheduled"] = [48, 1261, 510, 1794]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_03_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-6/41_text_Billy_Joel.png
try:
    _c41 = get_crop(41, 193, 60)
    canvas.paste(_c41, (538, 1613), _c41)
except Exception:
    pass
layout["Billy_Joel"] = [538, 1613, 731, 1673]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_03_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-6/42_text_Olivia_Rodrigo_with.png
try:
    _c42 = get_crop(42, 396, 533)
    canvas.paste(_c42, (1044, 1261), _c42)
except Exception:
    pass
layout["Olivia_Rodrigo_with"] = [1044, 1261, 1440, 1794]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_03_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-6/43_text_from_3_15_2024.png
try:
    _c43 = get_crop(43, 462, 533)
    canvas.paste(_c43, (48, 1261), _c43)
except Exception:
    pass
layout["from_3_15_2024)"] = [48, 1261, 510, 1794]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_03_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-6/44_text_Thu_Mar_28_8_PM.png
try:
    _c44 = get_crop(44, 462, 519)
    canvas.paste(_c44, (546, 1261), _c44)
except Exception:
    pass
layout["Thu,_Mar_28,8_PM"] = [546, 1261, 1008, 1780]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_03_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-6/45_text_Fri_Mar_29_8_PM.png
try:
    _c45 = get_crop(45, 462, 533)
    canvas.paste(_c45, (48, 1261), _c45)
except Exception:
    pass
layout["Fri,_Mar_29,8_PM"] = [48, 1261, 510, 1794]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_03_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-6/46_text_Tue_Apr_9_7.30_PM.png
try:
    _c46 = get_crop(46, 396, 533)
    canvas.paste(_c46, (1044, 1261), _c46)
except Exception:
    pass
layout["Tue,_Apr_9,_7.30_PM"] = [1044, 1261, 1440, 1794]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_03_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-6/47_text_Broadway_Shows.png
try:
    _c47 = get_crop(47, 72, 72)
    canvas.paste(_c47, (408, 2053), _c47)
except Exception:
    pass
layout["Broadway_Shows"] = [408, 2053, 480, 2125]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_03_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-6/48_text_View_all.png
try:
    _c48 = get_crop(48, 264, 183)
    canvas.paste(_c48, (1176, 1846), _c48)
except Exception:
    pass
layout["View_all"] = [1176, 1846, 1440, 2029]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_03_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-6/49_text_Oata.png
try:
    _c49 = get_crop(49, 56, 9)
    canvas.paste(_c49, (677, 2097), _c49)
except Exception:
    pass
layout["Oata"] = [677, 2097, 733, 2106]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_03_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-6/50_text_HTtera_NeT_C.png
try:
    _c50 = get_crop(50, 113, 9)
    canvas.paste(_c50, (749, 2097), _c50)
except Exception:
    pass
layout["HTtera_NeT_C"] = [749, 2097, 862, 2106]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_03_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-6/51_text_Tete.png
try:
    _c51 = get_crop(51, 462, 533)
    canvas.paste(_c51, (546, 2029), _c51)
except Exception:
    pass
layout["Tete"] = [546, 2029, 1008, 2562]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_03_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-6/52_text_5155_ES_OWN.png
try:
    _c52 = get_crop(52, 462, 519)
    canvas.paste(_c52, (48, 2029), _c52)
except Exception:
    pass
layout["5155+ES_OWN"] = [48, 2029, 510, 2548]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_03_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-6/53_text_S88.png
try:
    _c53 = get_crop(53, 117, 52)
    canvas.paste(_c53, (592, 2257), _c53)
except Exception:
    pass
layout["S88+"] = [592, 2257, 709, 2309]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_03_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-6/54_text_Hadestown.png
try:
    _c54 = get_crop(54, 462, 519)
    canvas.paste(_c54, (48, 2029), _c54)
except Exception:
    pass
layout["Hadestown"] = [48, 2029, 510, 2548]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_03_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-6/55_text_Moulin_Rougel_The.png
try:
    _c55 = get_crop(55, 462, 533)
    canvas.paste(_c55, (546, 2029), _c55)
except Exception:
    pass
layout["Moulin_Rougel_The"] = [546, 2029, 1008, 2562]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_03_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-6/56_text_The_Book_of_Morme.png
try:
    _c56 = get_crop(56, 396, 519)
    canvas.paste(_c56, (1044, 2029), _c56)
except Exception:
    pass
layout["The_Book_of_Morme"] = [1044, 2029, 1440, 2548]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_03_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-6/57_text_Tonight.png
try:
    _c57 = get_crop(57, 166, 62)
    canvas.paste(_c57, (40, 2447), _c57)
except Exception:
    pass
layout["Tonight"] = [40, 2447, 206, 2509]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_03_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-6/58_text_7.30_PM.png
try:
    _c58 = get_crop(58, 165, 50)
    canvas.paste(_c58, (222, 2449), _c58)
except Exception:
    pass
layout["7.30_PM"] = [222, 2449, 387, 2499]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_03_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-6/59_text_Tonight.png
try:
    _c59 = get_crop(59, 164, 62)
    canvas.paste(_c59, (1037, 2447), _c59)
except Exception:
    pass
layout["Tonight"] = [1037, 2447, 1201, 2509]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_03_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-6/60_text_7_PM.png
try:
    _c60 = get_crop(60, 103, 43)
    canvas.paste(_c60, (1220, 2452), _c60)
except Exception:
    pass
layout["7_PM"] = [1220, 2452, 1323, 2495]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_03_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-6/61_text_Tonight.png
try:
    _c61 = get_crop(61, 164, 62)
    canvas.paste(_c61, (537, 2507), _c61)
except Exception:
    pass
layout["Tonight"] = [537, 2507, 701, 2569]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_03_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-6/62_text_8_PM.png
try:
    _c62 = get_crop(62, 106, 43)
    canvas.paste(_c62, (720, 2513), _c62)
except Exception:
    pass
layout["8_PM"] = [720, 2513, 826, 2556]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_03_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-6/63_text_Comedy.png
try:
    _c63 = get_crop(63, 227, 68)
    canvas.paste(_c63, (43, 2677), _c63)
except Exception:
    pass
layout["Comedy"] = [43, 2677, 270, 2745]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_03_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-6/64_text_View_all.png
try:
    _c64 = get_crop(64, 264, 183)
    canvas.paste(_c64, (1176, 2614), _c64)
except Exception:
    pass
layout["View_all"] = [1176, 2614, 1440, 2797]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_03_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-6/65_clickable_Tracking.png
try:
    _c65 = get_crop(65, 72, 72)
    canvas.paste(_c65, (408, 1285), _c65)
except Exception:
    pass
layout["Tracking"] = [408, 1285, 480, 1357]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_03_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-6/66_clickable_Tracking.png
try:
    _c66 = get_crop(66, 72, 72)
    canvas.paste(_c66, (906, 1285), _c66)
except Exception:
    pass
layout["Tracking"] = [906, 1285, 978, 1357]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_03_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-6/67_clickable_Tracking.png
try:
    _c67 = get_crop(67, 72, 72)
    canvas.paste(_c67, (906, 2053), _c67)
except Exception:
    pass
layout["Tracking"] = [906, 2053, 978, 2125]
