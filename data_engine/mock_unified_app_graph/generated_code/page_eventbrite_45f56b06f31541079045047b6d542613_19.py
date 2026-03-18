# page_id: page_eventbrite_45f56b06f31541079045047b6d542613_19
# screenshot: 2024_4_23_19_27_45f56b06f31541079045047b6d542613-21.png
# step_index: 19/21
# task: Open Eventbrite. Search events 'Yoga session' in New York. Filter free events and set date from May 3 to May 6. What is the location of the first promoted event?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw overall background and UI structural elements for the mobile date-picker screen.
# Uses provided variables: canvas (PIL Image), draw (ImageDraw), font_sm, font_md, font_lg, font_xl

# Fill whole canvas with dominant background color (white)
draw.rectangle([(0, 0), (1440, 2960)], fill="#FFFFFF")

# Status bar background (top area)
status_h = 72
draw.rectangle([(0, 0), (1440, status_h)], fill="#CFCFCF")

# Header / toolbar background (below status bar)
header_top = status_h
header_bottom = 168
draw.rectangle([(0, header_top), (1440, header_bottom)], fill="#FFFFFF")

# Subtle divider under header
draw.line([(24, header_bottom), (1440-24, header_bottom)], fill="#EAE6F2", width=2)

# Calendar container background (subtle card-like area behind the calendar grid)
# Keep this above the apply-button area (do not overlap bottom detected apply button region)
cal_top = 220
cal_left = 48
cal_right = 1440 - 48
cal_bottom = 1600  # well above apply-button at ~2768
draw.rounded_rectangle([(cal_left, cal_top), (cal_right, cal_bottom)],
                       radius=20, fill="#FFFFFF", outline="#F2F0F6", width=2)

# Subtle inner separators to visually separate month header from calendar grid
month_header_y = 640
draw.line([(cal_left+30, month_header_y), (cal_right-30, month_header_y)], fill="#F3F1F7", width=1)

# Light horizontal guides for calendar area (very subtle, to indicate structure only)
for y in range(760, 1360, 160):
    draw.line([(cal_left+60, y), (cal_right-60, y)], fill="#FBFAFC", width=1)

# Thin vertical center guide (subtle)
draw.line([(720, cal_top+20), (720, cal_bottom-20)], fill="#FBFAFC", width=1)

# Slight shadow below the calendar card to lift it from the background
shadow_y = cal_bottom + 6
draw.line([(cal_left+10, shadow_y), (cal_right-10, shadow_y)], fill="#F0EDF4", width=3)

# Top edge subtle border for the content region (below header)
content_top = header_bottom + 12
draw.line([(24, content_top), (1440-24, content_top)], fill="#F7F6FA", width=1)

# Safe-area bottom separator (do not draw full button; only a thin guide above the bottom area)
bottom_sep_y = 2728
draw.line([(24, bottom_sep_y), (1440-24, bottom_sep_y)], fill="#E8E5ED", width=2)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_19_2024_4_23_19_27_45f56b06f31541079045047b6d542613-21/00_icon_Apply_date_range.png
try:
    _c0 = get_crop(0, 1344, 144)
    canvas.paste(_c0, (48, 2768), _c0)
except Exception:
    pass
layout["Apply_date_range"] = [48, 2768, 1392, 2912]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_19_2024_4_23_19_27_45f56b06f31541079045047b6d542613-21/01_icon_13.png
try:
    _c1 = get_crop(1, 132, 120)
    canvas.paste(_c1, (192, 1004), _c1)
except Exception:
    pass
layout["13"] = [192, 1004, 324, 1124]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_19_2024_4_23_19_27_45f56b06f31541079045047b6d542613-21/02_icon_icon_2.png
try:
    _c2 = get_crop(2, 107, 110)
    canvas.paste(_c2, (70, 774), _c2)
except Exception:
    pass
layout["icon_2"] = [70, 774, 177, 884]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_19_2024_4_23_19_27_45f56b06f31541079045047b6d542613-21/03_icon_icon_3.png
try:
    _c3 = get_crop(3, 50, 71)
    canvas.paste(_c3, (1154, 0), _c3)
except Exception:
    pass
layout["icon_3"] = [1154, 0, 1204, 71]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_19_2024_4_23_19_27_45f56b06f31541079045047b6d542613-21/04_icon_7.29.png
try:
    _c4 = get_crop(4, 60, 63)
    canvas.paste(_c4, (180, 2), _c4)
except Exception:
    pass
layout["7.29"] = [180, 2, 240, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_19_2024_4_23_19_27_45f56b06f31541079045047b6d542613-21/05_icon_7.29.png
try:
    _c5 = get_crop(5, 60, 65)
    canvas.paste(_c5, (114, 1), _c5)
except Exception:
    pass
layout["7.29"] = [114, 1, 174, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_19_2024_4_23_19_27_45f56b06f31541079045047b6d542613-21/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 61, 62)
    canvas.paste(_c6, (310, 3), _c6)
except Exception:
    pass
layout["icon_6"] = [310, 3, 371, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_19_2024_4_23_19_27_45f56b06f31541079045047b6d542613-21/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 100, 71)
    canvas.paste(_c7, (1210, 0), _c7)
except Exception:
    pass
layout["icon_7"] = [1210, 0, 1310, 71]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_19_2024_4_23_19_27_45f56b06f31541079045047b6d542613-21/08_icon_May_2024.png
try:
    _c8 = get_crop(8, 120, 110)
    canvas.paste(_c8, (200, 774), _c8)
except Exception:
    pass
layout["May_2024"] = [200, 774, 320, 884]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_19_2024_4_23_19_27_45f56b06f31541079045047b6d542613-21/09_icon_26.png
try:
    _c9 = get_crop(9, 132, 120)
    canvas.paste(_c9, (60, 1364), _c9)
except Exception:
    pass
layout["26"] = [60, 1364, 192, 1484]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_19_2024_4_23_19_27_45f56b06f31541079045047b6d542613-21/10_icon_27.png
try:
    _c10 = get_crop(10, 132, 120)
    canvas.paste(_c10, (192, 1364), _c10)
except Exception:
    pass
layout["27"] = [192, 1364, 324, 1484]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_19_2024_4_23_19_27_45f56b06f31541079045047b6d542613-21/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 50, 60)
    canvas.paste(_c11, (249, 5), _c11)
except Exception:
    pass
layout["icon_11"] = [249, 5, 299, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_19_2024_4_23_19_27_45f56b06f31541079045047b6d542613-21/12_icon_28.png
try:
    _c12 = get_crop(12, 132, 120)
    canvas.paste(_c12, (324, 1364), _c12)
except Exception:
    pass
layout["28"] = [324, 1364, 456, 1484]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_19_2024_4_23_19_27_45f56b06f31541079045047b6d542613-21/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 53, 70)
    canvas.paste(_c13, (1318, 0), _c13)
except Exception:
    pass
layout["icon_13"] = [1318, 0, 1371, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_19_2024_4_23_19_27_45f56b06f31541079045047b6d542613-21/14_icon_May_2024.png
try:
    _c14 = get_crop(14, 132, 111)
    canvas.paste(_c14, (324, 772), _c14)
except Exception:
    pass
layout["May_2024"] = [324, 772, 456, 883]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_19_2024_4_23_19_27_45f56b06f31541079045047b6d542613-21/15_icon_7.29.png
try:
    _c15 = get_crop(15, 144, 144)
    canvas.paste(_c15, (12, 72), _c15)
except Exception:
    pass
layout["7.29"] = [12, 72, 156, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_19_2024_4_23_19_27_45f56b06f31541079045047b6d542613-21/16_icon_May_2024.png
try:
    _c16 = get_crop(16, 123, 111)
    canvas.paste(_c16, (461, 772), _c16)
except Exception:
    pass
layout["May_2024"] = [461, 772, 584, 883]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_19_2024_4_23_19_27_45f56b06f31541079045047b6d542613-21/17_icon_7.29.png
try:
    _c17 = get_crop(17, 90, 62)
    canvas.paste(_c17, (17, 2), _c17)
except Exception:
    pass
layout["7.29"] = [17, 2, 107, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_19_2024_4_23_19_27_45f56b06f31541079045047b6d542613-21/18_icon_What_date.png
try:
    _c18 = get_crop(18, 319, 73)
    canvas.paste(_c18, (558, 111), _c18)
except Exception:
    pass
layout["What_date?"] = [558, 111, 877, 184]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_19_2024_4_23_19_27_45f56b06f31541079045047b6d542613-21/19_icon_icon_19.png
try:
    _c19 = get_crop(19, 50, 65)
    canvas.paste(_c19, (382, 2), _c19)
except Exception:
    pass
layout["icon_19"] = [382, 2, 432, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_19_2024_4_23_19_27_45f56b06f31541079045047b6d542613-21/20_icon_29.png
try:
    _c20 = get_crop(20, 132, 120)
    canvas.paste(_c20, (456, 1364), _c20)
except Exception:
    pass
layout["29"] = [456, 1364, 588, 1484]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_19_2024_4_23_19_27_45f56b06f31541079045047b6d542613-21/21_icon_Next_month.png
try:
    _c21 = get_crop(21, 144, 144)
    canvas.paste(_c21, (846, 620), _c21)
except Exception:
    pass
layout["Next_month"] = [846, 620, 990, 764]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_19_2024_4_23_19_27_45f56b06f31541079045047b6d542613-21/22_icon_24.png
try:
    _c22 = get_crop(22, 132, 120)
    canvas.paste(_c22, (720, 1244), _c22)
except Exception:
    pass
layout["24"] = [720, 1244, 852, 1364]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_19_2024_4_23_19_27_45f56b06f31541079045047b6d542613-21/23_icon_14.png
try:
    _c23 = get_crop(23, 132, 120)
    canvas.paste(_c23, (324, 1004), _c23)
except Exception:
    pass
layout["14"] = [324, 1004, 456, 1124]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_19_2024_4_23_19_27_45f56b06f31541079045047b6d542613-21/24_icon_13.png
try:
    _c24 = get_crop(24, 132, 120)
    canvas.paste(_c24, (192, 1124), _c24)
except Exception:
    pass
layout["13"] = [192, 1124, 324, 1244]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_19_2024_4_23_19_27_45f56b06f31541079045047b6d542613-21/25_text_Start_Date.png
try:
    _c25 = get_crop(25, 613, 144)
    canvas.paste(_c25, (48, 313), _c25)
except Exception:
    pass
layout["Start_Date"] = [48, 313, 661, 457]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_19_2024_4_23_19_27_45f56b06f31541079045047b6d542613-21/26_text_End_Date.png
try:
    _c26 = get_crop(26, 617, 114)
    canvas.paste(_c26, (48, 476), _c26)
except Exception:
    pass
layout["End_Date"] = [48, 476, 665, 590]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_19_2024_4_23_19_27_45f56b06f31541079045047b6d542613-21/27_text_May_2024.png
try:
    _c27 = get_crop(27, 198, 56)
    canvas.paste(_c27, (423, 666), _c27)
except Exception:
    pass
layout["May_2024"] = [423, 666, 621, 722]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_19_2024_4_23_19_27_45f56b06f31541079045047b6d542613-21/28_text_10.png
try:
    _c28 = get_crop(28, 132, 120)
    canvas.paste(_c28, (720, 1004), _c28)
except Exception:
    pass
layout["10"] = [720, 1004, 852, 1124]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_19_2024_4_23_19_27_45f56b06f31541079045047b6d542613-21/29_text_11.png
try:
    _c29 = get_crop(29, 132, 120)
    canvas.paste(_c29, (852, 1004), _c29)
except Exception:
    pass
layout["11"] = [852, 1004, 984, 1124]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_19_2024_4_23_19_27_45f56b06f31541079045047b6d542613-21/30_text_12.png
try:
    _c30 = get_crop(30, 132, 120)
    canvas.paste(_c30, (60, 1124), _c30)
except Exception:
    pass
layout["12"] = [60, 1124, 192, 1244]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_19_2024_4_23_19_27_45f56b06f31541079045047b6d542613-21/31_text_14.png
try:
    _c31 = get_crop(31, 132, 120)
    canvas.paste(_c31, (324, 1124), _c31)
except Exception:
    pass
layout["14"] = [324, 1124, 456, 1244]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_19_2024_4_23_19_27_45f56b06f31541079045047b6d542613-21/32_text_15.png
try:
    _c32 = get_crop(32, 132, 120)
    canvas.paste(_c32, (456, 1124), _c32)
except Exception:
    pass
layout["15"] = [456, 1124, 588, 1244]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_19_2024_4_23_19_27_45f56b06f31541079045047b6d542613-21/33_text_16.png
try:
    _c33 = get_crop(33, 132, 120)
    canvas.paste(_c33, (588, 1124), _c33)
except Exception:
    pass
layout["16"] = [588, 1124, 720, 1244]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_19_2024_4_23_19_27_45f56b06f31541079045047b6d542613-21/34_text_17.png
try:
    _c34 = get_crop(34, 132, 120)
    canvas.paste(_c34, (720, 1124), _c34)
except Exception:
    pass
layout["17"] = [720, 1124, 852, 1244]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_19_2024_4_23_19_27_45f56b06f31541079045047b6d542613-21/35_text_18.png
try:
    _c35 = get_crop(35, 132, 120)
    canvas.paste(_c35, (852, 1124), _c35)
except Exception:
    pass
layout["18"] = [852, 1124, 984, 1244]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_19_2024_4_23_19_27_45f56b06f31541079045047b6d542613-21/36_text_19.png
try:
    _c36 = get_crop(36, 132, 120)
    canvas.paste(_c36, (60, 1244), _c36)
except Exception:
    pass
layout["19"] = [60, 1244, 192, 1364]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_19_2024_4_23_19_27_45f56b06f31541079045047b6d542613-21/37_text_20.png
try:
    _c37 = get_crop(37, 132, 120)
    canvas.paste(_c37, (192, 1244), _c37)
except Exception:
    pass
layout["20"] = [192, 1244, 324, 1364]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_19_2024_4_23_19_27_45f56b06f31541079045047b6d542613-21/38_text_21.png
try:
    _c38 = get_crop(38, 132, 120)
    canvas.paste(_c38, (324, 1244), _c38)
except Exception:
    pass
layout["21"] = [324, 1244, 456, 1364]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_19_2024_4_23_19_27_45f56b06f31541079045047b6d542613-21/39_text_22.png
try:
    _c39 = get_crop(39, 132, 120)
    canvas.paste(_c39, (456, 1244), _c39)
except Exception:
    pass
layout["22"] = [456, 1244, 588, 1364]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_19_2024_4_23_19_27_45f56b06f31541079045047b6d542613-21/40_text_23.png
try:
    _c40 = get_crop(40, 132, 120)
    canvas.paste(_c40, (588, 1244), _c40)
except Exception:
    pass
layout["23"] = [588, 1244, 720, 1364]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_19_2024_4_23_19_27_45f56b06f31541079045047b6d542613-21/41_text_25.png
try:
    _c41 = get_crop(41, 132, 120)
    canvas.paste(_c41, (852, 1244), _c41)
except Exception:
    pass
layout["25"] = [852, 1244, 984, 1364]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_19_2024_4_23_19_27_45f56b06f31541079045047b6d542613-21/42_text_30.png
try:
    _c42 = get_crop(42, 132, 120)
    canvas.paste(_c42, (588, 1364), _c42)
except Exception:
    pass
layout["30"] = [588, 1364, 720, 1484]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_19_2024_4_23_19_27_45f56b06f31541079045047b6d542613-21/43_text_31.png
try:
    _c43 = get_crop(43, 132, 120)
    canvas.paste(_c43, (720, 1364), _c43)
except Exception:
    pass
layout["31"] = [720, 1364, 852, 1484]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_19_2024_4_23_19_27_45f56b06f31541079045047b6d542613-21/44_clickable_1.png
try:
    _c44 = get_crop(44, 132, 120)
    canvas.paste(_c44, (456, 884), _c44)
except Exception:
    pass
layout["1"] = [456, 884, 588, 1004]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_19_2024_4_23_19_27_45f56b06f31541079045047b6d542613-21/45_clickable_2.png
try:
    _c45 = get_crop(45, 132, 120)
    canvas.paste(_c45, (588, 884), _c45)
except Exception:
    pass
layout["2"] = [588, 884, 720, 1004]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_19_2024_4_23_19_27_45f56b06f31541079045047b6d542613-21/46_clickable_3.png
try:
    _c46 = get_crop(46, 132, 120)
    canvas.paste(_c46, (720, 884), _c46)
except Exception:
    pass
layout["3"] = [720, 884, 852, 1004]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_19_2024_4_23_19_27_45f56b06f31541079045047b6d542613-21/47_clickable_4.png
try:
    _c47 = get_crop(47, 132, 120)
    canvas.paste(_c47, (852, 884), _c47)
except Exception:
    pass
layout["4"] = [852, 884, 984, 1004]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_19_2024_4_23_19_27_45f56b06f31541079045047b6d542613-21/48_clickable_5.png
try:
    _c48 = get_crop(48, 132, 120)
    canvas.paste(_c48, (60, 1004), _c48)
except Exception:
    pass
layout["5"] = [60, 1004, 192, 1124]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_19_2024_4_23_19_27_45f56b06f31541079045047b6d542613-21/49_clickable_8.png
try:
    _c49 = get_crop(49, 132, 120)
    canvas.paste(_c49, (456, 1004), _c49)
except Exception:
    pass
layout["8"] = [456, 1004, 588, 1124]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_19_2024_4_23_19_27_45f56b06f31541079045047b6d542613-21/50_clickable_9.png
try:
    _c50 = get_crop(50, 132, 120)
    canvas.paste(_c50, (588, 1004), _c50)
except Exception:
    pass
layout["9"] = [588, 1004, 720, 1124]
