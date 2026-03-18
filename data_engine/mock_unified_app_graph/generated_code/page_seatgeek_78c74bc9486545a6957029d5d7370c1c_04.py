# page_id: page_seatgeek_78c74bc9486545a6957029d5d7370c1c_04
# screenshot: 2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-7.png
# step_index: 4/9
# task: Open SeatGeek and search by category "Comedy". Select the first one in New York and check its information. Track the performer of this event.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Top status bar
draw.rectangle([(0, 0), (1440, 72)], fill="#f3f4f5")

# Header / toolbar area (below status bar)
draw.rectangle([(0, 72), (1440, 200)], fill="#ffffff")
# subtle bottom divider under toolbar
draw.line([(24, 200), (1416, 200)], fill="#e6e6e6", width=1)

# Category chips background strip (rounded pill container behind chips)
chips_box = (16, 260, 1424, 392)
draw.rounded_rectangle(chips_box, radius=28, fill="#ffffff", outline="#e9e9e9", width=1)

# First large hero/background area (purple)
hero1_box = (0, 456, 1440, 1591)
draw.rectangle(hero1_box, fill="#6f34e6")  # vibrant purple backdrop
# subtle top inner darker overlay near left to echo screenshot shading
draw.rectangle([(0, 456), (720, 920)], fill="#5a28c6")

# White info card for first event (placed overlapping bottom of hero1)
card1_box = (0, 1270, 1440, 1470)
draw.rectangle(card1_box, fill="#ffffff")
# card top divider
draw.line([(24, 1270), (1416, 1270)], fill="#e6e6e6", width=1)
# faint shadow under this card
draw.line([(24, 1470), (1416, 1470)], fill="#dcdcdc", width=2)

# Divider separating info area and action row for first card
draw.line([(24, 1396), (1416, 1396)], fill="#f0f0f0", width=1)

# Second large hero/background area (yellow)
hero2_box = (0, 1591, 1440, 2726)
draw.rectangle(hero2_box, fill="#f2b400")  # warm golden backdrop
# subtle textured darker patch to the left
draw.rectangle([(0, 1790), (640, 2120)], fill="#e2a300")

# White info card for second event (placed overlapping bottom of hero2)
card2_box = (0, 2430, 1440, 2670)
draw.rectangle(card2_box, fill="#ffffff")
# card top divider
draw.line([(24, 2430), (1416, 2430)], fill="#e6e6e6", width=1)
# faint shadow under this card
draw.line([(24, 2670), (1416, 2670)], fill="#dcdcdc", width=2)

# Thin separators between major sections
draw.line([(0, 1591), (1440, 1591)], fill="#efefef", width=1)
draw.line([(0, 2726), (1440, 2726)], fill="#efefef", width=1)

# Bottom safe area background (slightly off-white)
draw.rectangle([(0, 2726), (1440, 2960)], fill="#fbfbfb")

# Small rounded separators / card edges for subtle elevation on hero images
draw.rounded_rectangle([(16, 1266), (1424, 1474)], radius=6, outline="#f3f3f3", width=1)
draw.rounded_rectangle([(16, 2426), (1424, 2674)], radius=6, outline="#f3f3f3", width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_04_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-7/00_icon_Broadway.png
try:
    _c0 = get_crop(0, 294, 97)
    canvas.paste(_c0, (21, 335), _c0)
except Exception:
    pass
layout["Broadway"] = [21, 335, 315, 432]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_04_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-7/01_icon_Family.png
try:
    _c1 = get_crop(1, 221, 97)
    canvas.paste(_c1, (624, 335), _c1)
except Exception:
    pass
layout["Family"] = [624, 335, 845, 432]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_04_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-7/02_icon_Dance.png
try:
    _c2 = get_crop(2, 224, 97)
    canvas.paste(_c2, (869, 335), _c2)
except Exception:
    pass
layout["Dance"] = [869, 335, 1093, 432]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_04_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-7/03_icon_Comedy.png
try:
    _c3 = get_crop(3, 261, 97)
    canvas.paste(_c3, (339, 335), _c3)
except Exception:
    pass
layout["Comedy"] = [339, 335, 600, 432]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_04_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-7/04_icon_Cirque_Du_Sole.png
try:
    _c4 = get_crop(4, 323, 97)
    canvas.paste(_c4, (1117, 335), _c4)
except Exception:
    pass
layout["Cirque_Du_Sole"] = [1117, 335, 1440, 432]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_04_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-7/05_icon_Track.png
try:
    _c5 = get_crop(5, 267, 185)
    canvas.paste(_c5, (0, 1382), _c5)
except Exception:
    pass
layout["Track"] = [0, 1382, 267, 1567]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_04_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-7/06_icon_Track.png
try:
    _c6 = get_crop(6, 267, 185)
    canvas.paste(_c6, (0, 2517), _c6)
except Exception:
    pass
layout["Track"] = [0, 2517, 267, 2702]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_04_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-7/07_icon_Share.png
try:
    _c7 = get_crop(7, 248, 162)
    canvas.paste(_c7, (267, 2533), _c7)
except Exception:
    pass
layout["Share"] = [267, 2533, 515, 2695]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_04_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-7/08_icon_Share.png
try:
    _c8 = get_crop(8, 248, 162)
    canvas.paste(_c8, (267, 1398), _c8)
except Exception:
    pass
layout["Share"] = [267, 1398, 515, 1560]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_04_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-7/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 49, 52)
    canvas.paste(_c9, (317, 8), _c9)
except Exception:
    pass
layout["icon_9"] = [317, 8, 366, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_04_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-7/10_icon_884.png
try:
    _c10 = get_crop(10, 144, 240)
    canvas.paste(_c10, (1260, 72), _c10)
except Exception:
    pass
layout["884"] = [1260, 72, 1404, 312]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_04_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-7/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 52, 54)
    canvas.paste(_c11, (248, 7), _c11)
except Exception:
    pass
layout["icon_11"] = [248, 7, 300, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_04_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-7/12_icon_8.28_my.png
try:
    _c12 = get_crop(12, 53, 54)
    canvas.paste(_c12, (115, 7), _c12)
except Exception:
    pass
layout["8.28_my"] = [115, 7, 168, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_04_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-7/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 46, 64)
    canvas.paste(_c13, (1155, 2), _c13)
except Exception:
    pass
layout["icon_13"] = [1155, 2, 1201, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_04_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-7/14_icon_8.28_my.png
try:
    _c14 = get_crop(14, 53, 56)
    canvas.paste(_c14, (183, 5), _c14)
except Exception:
    pass
layout["8.28_my"] = [183, 5, 236, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_04_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-7/15_icon_884.png
try:
    _c15 = get_crop(15, 95, 55)
    canvas.paste(_c15, (1219, 6), _c15)
except Exception:
    pass
layout["884"] = [1219, 6, 1314, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_04_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-7/16_icon_icon_16.png
try:
    _c16 = get_crop(16, 47, 52)
    canvas.paste(_c16, (1322, 7), _c16)
except Exception:
    pass
layout["icon_16"] = [1322, 7, 1369, 59]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_04_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-7/17_icon_8.28_my.png
try:
    _c17 = get_crop(17, 144, 240)
    canvas.paste(_c17, (0, 72), _c17)
except Exception:
    pass
layout["8.28_my"] = [0, 72, 144, 312]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_04_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-7/18_icon_Philadelphia_PA.png
try:
    _c18 = get_crop(18, 1440, 1135)
    canvas.paste(_c18, (0, 1591), _c18)
except Exception:
    pass
layout["Philadelphia,_PA"] = [0, 1591, 1440, 2726]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_04_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-7/19_icon_Philadelphia_PA.png
try:
    _c19 = get_crop(19, 1440, 1135)
    canvas.paste(_c19, (0, 456), _c19)
except Exception:
    pass
layout["Philadelphia,_PA"] = [0, 456, 1440, 1591]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_04_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-7/20_icon_icon_20.png
try:
    _c20 = get_crop(20, 43, 56)
    canvas.paste(_c20, (385, 4), _c20)
except Exception:
    pass
layout["icon_20"] = [385, 4, 428, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_04_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-7/21_icon_S132.png
try:
    _c21 = get_crop(21, 180, 65)
    canvas.paste(_c21, (40, 1048), _c21)
except Exception:
    pass
layout["S132+"] = [40, 1048, 220, 1113]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_04_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-7/22_icon_8.28_my.png
try:
    _c22 = get_crop(22, 93, 61)
    canvas.paste(_c22, (15, 2), _c22)
except Exception:
    pass
layout["8.28_my"] = [15, 2, 108, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_04_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-7/23_icon_Share.png
try:
    _c23 = get_crop(23, 257, 75)
    canvas.paste(_c23, (638, 2885), _c23)
except Exception:
    pass
layout["Share"] = [638, 2885, 895, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_04_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-7/24_text_Comedy.png
try:
    _c24 = get_crop(24, 251, 77)
    canvas.paste(_c24, (185, 132), _c24)
except Exception:
    pass
layout["Comedy"] = [185, 132, 436, 209]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_04_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-7/25_text_date.png
try:
    _c25 = get_crop(25, 117, 52)
    canvas.paste(_c25, (606, 208), _c25)
except Exception:
    pass
layout["date"] = [606, 208, 723, 260]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_04_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-7/26_text_Sun_Apr_7_7_PM.png
try:
    _c26 = get_crop(26, 267, 185)
    canvas.paste(_c26, (0, 1382), _c26)
except Exception:
    pass
layout["Sun,_Apr_7,7_PM"] = [0, 1382, 267, 1567]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_04_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-7/27_text_Philadelphia_PA.png
try:
    _c27 = get_crop(27, 248, 162)
    canvas.paste(_c27, (267, 1398), _c27)
except Exception:
    pass
layout["Philadelphia,_PA"] = [267, 1398, 515, 1560]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_04_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-7/28_text_Liacouras_Center.png
try:
    _c28 = get_crop(28, 372, 52)
    canvas.paste(_c28, (781, 1279), _c28)
except Exception:
    pass
layout["Liacouras_Center"] = [781, 1279, 1153, 1331]
