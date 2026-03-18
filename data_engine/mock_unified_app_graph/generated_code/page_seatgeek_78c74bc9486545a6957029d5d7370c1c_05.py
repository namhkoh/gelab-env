# page_id: page_seatgeek_78c74bc9486545a6957029d5d7370c1c_05
# screenshot: 2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-8.png
# step_index: 5/9
# task: Open SeatGeek and search by category "Comedy". Select the first one in New York and check its information. Track the performer of this event.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Top-level background
draw.rectangle((0, 0, 1440, 2960), fill=(250, 250, 250))

# Status bar (top ~80px)
status_h = 80
draw.rectangle((0, 0, 1440, status_h), fill=(236, 236, 236))
# subtle bottom divider under status bar
draw.line((0, status_h, 1440, status_h), fill=(220, 220, 220), width=1)

# Header / toolbar area (white card-like)
header_y0 = status_h
header_y1 = status_h + 100
draw.rectangle((0, header_y0, 1440, header_y1), fill=(255, 255, 255))
# divider under header
draw.line((24, header_y1, 1440-24, header_y1), fill=(230, 230, 230), width=1)

# First event image/banner background (warm mustard)
banner1_y0 = header_y1 + 20
banner1_y1 = banner1_y0 + 180
draw.rounded_rectangle((0, banner1_y0, 1440, banner1_y1), radius=6, fill=(241, 180, 66))
# subtle darker bottom edge for separation
draw.line((0, banner1_y1, 1440, banner1_y1), fill=(230, 170, 50), width=2)

# First event card body (white)
card1_y0 = banner1_y1
card1_y1 = card1_y0 + 180
draw.rectangle((0, card1_y0, 1440, card1_y1), fill=(255, 255, 255))
# horizontal thin separators inside card
draw.line((24, card1_y0 + 120, 1440-24, card1_y0 + 120), fill=(235, 235, 235), width=1)
draw.line((24, card1_y1, 1440-24, card1_y1), fill=(230, 230, 230), width=1)

# Track/share row divider below (subtle)
divider1 = card1_y1 + 8
draw.line((0, divider1, 1440, divider1), fill=(240, 240, 240), width=8)

# Second event image/banner background (vibrant blue)
banner2_y0 = divider1 + 24
banner2_y1 = banner2_y0 + 280
draw.rounded_rectangle((0, banner2_y0, 1440, banner2_y1), radius=6, fill=(18, 97, 229))
# bottom edge highlight
draw.line((0, banner2_y1, 1440, banner2_y1), fill=(12, 75, 177), width=2)

# Second event card body (white)
card2_y0 = banner2_y1
card2_y1 = card2_y0 + 180
draw.rectangle((0, card2_y0, 1440, card2_y1), fill=(255, 255, 255))
draw.line((24, card2_y0 + 120, 1440-24, card2_y0 + 120), fill=(235, 235, 235), width=1)
draw.line((24, card2_y1, 1440-24, card2_y1), fill=(230, 230, 230), width=1)

# Divider under second card
divider2 = card2_y1 + 8
draw.line((0, divider2, 1440, divider2), fill=(240, 240, 240), width=8)

# Third event big image/banner background (dark stage image area)
banner3_y0 = divider2 + 24
banner3_y1 = banner3_y0 + 340
draw.rectangle((0, banner3_y0, 1440, banner3_y1), fill=(20, 20, 20))
# soft top highlight to indicate image content (thin)
draw.line((0, banner3_y0, 1440, banner3_y0), fill=(40, 40, 40), width=2)

# Third event white card area below dark banner
card3_y0 = banner3_y1
card3_y1 = card3_y0 + 240
draw.rectangle((0, card3_y0, 1440, card3_y1), fill=(255, 255, 255))
draw.line((24, card3_y0 + 120, 1440-24, card3_y0 + 120), fill=(235, 235, 235), width=1)
draw.line((24, card3_y1, 1440-24, card3_y1), fill=(230, 230, 230), width=1)

# Global separators between major sections (subtle)
for y in (header_y1, card1_y1, banner2_y1, card2_y1, banner3_y1, card3_y1):
    draw.line((24, y+1, 1440-24, y+1), fill=(245, 245, 245), width=1)

# Left and right padding guide lines (very subtle) to suggest content margins
margin_x = 24
draw.line((margin_x, header_y0, margin_x, card3_y1), fill=(250, 250, 250), width=1)
draw.line((1440-margin_x, header_y0, 1440-margin_x, card3_y1), fill=(250, 250, 250), width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_05_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-8/00_icon_Track.png
try:
    _c0 = get_crop(0, 267, 185)
    canvas.paste(_c0, (0, 820), _c0)
except Exception:
    pass
layout["Track"] = [0, 820, 267, 1005]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_05_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-8/01_icon_Track.png
try:
    _c1 = get_crop(1, 267, 185)
    canvas.paste(_c1, (0, 1955), _c1)
except Exception:
    pass
layout["Track"] = [0, 1955, 267, 2140]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_05_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-8/02_icon_Share.png
try:
    _c2 = get_crop(2, 248, 162)
    canvas.paste(_c2, (267, 836), _c2)
except Exception:
    pass
layout["Share"] = [267, 836, 515, 998]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_05_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-8/03_icon_848.png
try:
    _c3 = get_crop(3, 144, 240)
    canvas.paste(_c3, (1260, 72), _c3)
except Exception:
    pass
layout["848"] = [1260, 72, 1404, 312]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_05_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-8/04_icon_Share.png
try:
    _c4 = get_crop(4, 248, 162)
    canvas.paste(_c4, (267, 1971), _c4)
except Exception:
    pass
layout["Share"] = [267, 1971, 515, 2133]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_05_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-8/05_icon_Comedy.png
try:
    _c5 = get_crop(5, 51, 57)
    canvas.paste(_c5, (316, 4), _c5)
except Exception:
    pass
layout["Comedy"] = [316, 4, 367, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_05_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-8/06_icon_Share.png
try:
    _c6 = get_crop(6, 1440, 796)
    canvas.paste(_c6, (0, 2164), _c6)
except Exception:
    pass
layout["Share"] = [0, 2164, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_05_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-8/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 45, 68)
    canvas.paste(_c7, (1156, 0), _c7)
except Exception:
    pass
layout["icon_7"] = [1156, 0, 1201, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_05_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-8/08_icon_8.28_my.png
try:
    _c8 = get_crop(8, 53, 57)
    canvas.paste(_c8, (115, 6), _c8)
except Exception:
    pass
layout["8.28_my"] = [115, 6, 168, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_05_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-8/09_icon_8.28_my.png
try:
    _c9 = get_crop(9, 53, 56)
    canvas.paste(_c9, (182, 6), _c9)
except Exception:
    pass
layout["8.28_my"] = [182, 6, 235, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_05_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-8/10_icon_Comedy.png
try:
    _c10 = get_crop(10, 52, 57)
    canvas.paste(_c10, (247, 5), _c10)
except Exception:
    pass
layout["Comedy"] = [247, 5, 299, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_05_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-8/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 47, 60)
    canvas.paste(_c11, (1322, 2), _c11)
except Exception:
    pass
layout["icon_11"] = [1322, 2, 1369, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_05_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-8/12_icon_8.28_my.png
try:
    _c12 = get_crop(12, 144, 240)
    canvas.paste(_c12, (0, 72), _c12)
except Exception:
    pass
layout["8.28_my"] = [0, 72, 144, 312]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_05_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-8/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 56, 65)
    canvas.paste(_c13, (1217, 1), _c13)
except Exception:
    pass
layout["icon_13"] = [1217, 1, 1273, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_05_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-8/14_icon_New_York_NY.png
try:
    _c14 = get_crop(14, 1440, 1135)
    canvas.paste(_c14, (0, 1029), _c14)
except Exception:
    pass
layout["New_York,_NY"] = [0, 1029, 1440, 2164]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_05_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-8/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 56, 64)
    canvas.paste(_c15, (1259, 1), _c15)
except Exception:
    pass
layout["icon_15"] = [1259, 1, 1315, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_05_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-8/16_icon_icon_16.png
try:
    _c16 = get_crop(16, 44, 60)
    canvas.paste(_c16, (385, 1), _c16)
except Exception:
    pass
layout["icon_16"] = [385, 1, 429, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_05_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-8/17_icon_S86.png
try:
    _c17 = get_crop(17, 164, 64)
    canvas.paste(_c17, (43, 1622), _c17)
except Exception:
    pass
layout["S86+"] = [43, 1622, 207, 1686]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_05_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-8/18_icon_Comedy.png
try:
    _c18 = get_crop(18, 251, 76)
    canvas.paste(_c18, (181, 127), _c18)
except Exception:
    pass
layout["Comedy"] = [181, 127, 432, 203]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_05_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-8/19_text_New_York_NY.png
try:
    _c19 = get_crop(19, 307, 60)
    canvas.paste(_c19, (185, 205), _c19)
except Exception:
    pass
layout["New_York,_NY"] = [185, 205, 492, 265]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_05_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-8/20_text_date.png
try:
    _c20 = get_crop(20, 117, 52)
    canvas.paste(_c20, (606, 208), _c20)
except Exception:
    pass
layout["date"] = [606, 208, 723, 260]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_05_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-8/21_text_S82.png
try:
    _c21 = get_crop(21, 114, 52)
    canvas.paste(_c21, (65, 493), _c21)
except Exception:
    pass
layout["S82+"] = [65, 493, 179, 545]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_05_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-8/22_text_Jo.png
try:
    _c22 = get_crop(22, 61, 52)
    canvas.paste(_c22, (42, 643), _c22)
except Exception:
    pass
layout["Jo"] = [42, 643, 103, 695]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_05_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-8/23_text_Fri_Mar_22_8_PM.png
try:
    _c23 = get_crop(23, 267, 185)
    canvas.paste(_c23, (0, 820), _c23)
except Exception:
    pass
layout["Fri,_Mar_22,8_PM"] = [0, 820, 267, 1005]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_05_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-8/24_text_Philadelphia_PA.png
try:
    _c24 = get_crop(24, 1440, 717)
    canvas.paste(_c24, (0, 312), _c24)
except Exception:
    pass
layout["Philadelphia,_PA"] = [0, 312, 1440, 1029]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_05_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-8/25_text_Wells_Fargo_Center.png
try:
    _c25 = get_crop(25, 1440, 717)
    canvas.paste(_c25, (0, 312), _c25)
except Exception:
    pass
layout["Wells_Fargo_Center"] = [0, 312, 1440, 1029]
