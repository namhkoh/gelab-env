# page_id: page_eventbrite_a9f633a394e74f78843aa30bd2792346_16
# screenshot: 2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-18.png
# step_index: 16/18
# task: Open Eventbrite. Set the city to "Los Angeles". Look for Photography workshops happening next week. What is the price of the tickets for first non-promoted event?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Top-level background
draw.rectangle([(0, 0), (1440, 2960)], fill=(255, 255, 255))

# Status bar area (top subtle grey background)
status_bar_height = 72
draw.rectangle([(0, 0), (1440, status_bar_height)], fill=(226, 226, 226))

# Header divider (thin line below header area)
header_divider_y = 200
draw.line([(48, header_divider_y), (1392, header_divider_y)], fill=(235, 231, 243), width=2)

# Light left content background behind the Start/End date block (soft card)
left_card = (48, 240, 860, 560)
try:
    draw.rounded_rectangle(left_card, radius=14, fill=(250, 250, 255), outline=(235, 231, 243), width=1)
except Exception:
    # fallback if rounded_rectangle not supported
    draw.rectangle(left_card, fill=(250, 250, 255), outline=(235, 231, 243))

# Calendar container background (large subtle rounded rectangle)
cal_top = 580
cal_bottom = 1500
cal_left = 48
cal_right = 1392
try:
    draw.rounded_rectangle([cal_left, cal_top, cal_right, cal_bottom],
                           radius=18, fill=(254, 254, 255), outline=(240, 236, 248), width=1)
except Exception:
    draw.rectangle([cal_left, cal_top, cal_right, cal_bottom],
                   fill=(254, 254, 255), outline=(240, 236, 248))

# Subtle month header background band inside calendar container
month_band_top = cal_top + 40
month_band_bottom = cal_top + 120
draw.rectangle([(cal_left + 20, month_band_top), (cal_right - 20, month_band_bottom)], fill=(255, 255, 255))

# Very light horizontal separators for calendar area (do not draw numbers/text)
sep_color = (245, 243, 250)
for i in range(5):
    y = month_band_bottom + 100 + i * 160
    draw.line([(cal_left + 20, y), (cal_right - 20, y)], fill=sep_color, width=1)

# Subtle vertical gutters in calendar area to suggest columns (7 columns)
col_color = (248, 247, 252)
cols = 7
inner_left = cal_left + 40
inner_right = cal_right - 40
width = inner_right - inner_left
col_w = width / cols
for i in range(1, cols):
    x = int(inner_left + i * col_w)
    draw.line([(x, month_band_bottom + 20), (x, cal_bottom - 20)], fill=col_color, width=1)

# Top-to-content soft shadow under header area
shadow_top = header_divider_y + 2
for i, a in enumerate([30, 20, 10]):
    y = shadow_top + i * 2
    alpha = max(1, a)
    color = (240, 238, 244)
    draw.line([(48, y), (1392, y)], fill=color, width=1)

# Separator line above the bottom action area (leave the actual button area free)
bottom_sep_y = 2720
draw.line([(48, bottom_sep_y), (1392, bottom_sep_y)], fill=(235, 231, 243), width=2)

# Rounded background inset at very bottom (subtle) to visually ground the bottom button area
bottom_bg = (32, bottom_sep_y + 12, 1408, 2960 - 12)
try:
    draw.rounded_rectangle(bottom_bg, radius=12, fill=(255, 255, 255), outline=(245, 243, 248), width=1)
except Exception:
    draw.rectangle(bottom_bg, fill=(255, 255, 255), outline=(245, 243, 248))

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_16_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-18/00_icon_Apply_date_range.png
try:
    _c0 = get_crop(0, 1344, 144)
    canvas.paste(_c0, (48, 2768), _c0)
except Exception:
    pass
layout["Apply_date_range"] = [48, 2768, 1392, 2912]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_16_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-18/01_icon_icon_1.png
try:
    _c1 = get_crop(1, 52, 71)
    canvas.paste(_c1, (1153, 0), _c1)
except Exception:
    pass
layout["icon_1"] = [1153, 0, 1205, 71]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_16_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-18/02_icon_4.51.png
try:
    _c2 = get_crop(2, 60, 65)
    canvas.paste(_c2, (180, 1), _c2)
except Exception:
    pass
layout["4.51"] = [180, 1, 240, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_16_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-18/03_icon_4.51.png
try:
    _c3 = get_crop(3, 63, 66)
    canvas.paste(_c3, (112, 1), _c3)
except Exception:
    pass
layout["4.51"] = [112, 1, 175, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_16_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-18/04_icon_icon_4.png
try:
    _c4 = get_crop(4, 101, 70)
    canvas.paste(_c4, (1210, 0), _c4)
except Exception:
    pass
layout["icon_4"] = [1210, 0, 1311, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_16_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-18/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 63, 63)
    canvas.paste(_c5, (309, 2), _c5)
except Exception:
    pass
layout["icon_5"] = [309, 2, 372, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_16_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-18/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 53, 70)
    canvas.paste(_c6, (1319, 0), _c6)
except Exception:
    pass
layout["icon_6"] = [1319, 0, 1372, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_16_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-18/07_icon_4.51.png
try:
    _c7 = get_crop(7, 144, 144)
    canvas.paste(_c7, (12, 72), _c7)
except Exception:
    pass
layout["4.51"] = [12, 72, 156, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_16_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-18/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 51, 63)
    canvas.paste(_c8, (248, 2), _c8)
except Exception:
    pass
layout["icon_8"] = [248, 2, 299, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_16_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-18/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 97, 107)
    canvas.paste(_c9, (74, 777), _c9)
except Exception:
    pass
layout["icon_9"] = [74, 777, 171, 884]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_16_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-18/10_icon_What_date.png
try:
    _c10 = get_crop(10, 319, 72)
    canvas.paste(_c10, (558, 111), _c10)
except Exception:
    pass
layout["What_date?"] = [558, 111, 877, 183]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_16_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-18/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 110, 108)
    canvas.paste(_c11, (201, 775), _c11)
except Exception:
    pass
layout["icon_11"] = [201, 775, 311, 883]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_16_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-18/12_icon_4.51.png
try:
    _c12 = get_crop(12, 90, 62)
    canvas.paste(_c12, (16, 2), _c12)
except Exception:
    pass
layout["4.51"] = [16, 2, 106, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_16_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-18/13_icon_End_Date.png
try:
    _c13 = get_crop(13, 144, 144)
    canvas.paste(_c13, (54, 620), _c13)
except Exception:
    pass
layout["End_Date"] = [54, 620, 198, 764]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_16_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-18/14_icon_Next_month.png
try:
    _c14 = get_crop(14, 144, 144)
    canvas.paste(_c14, (846, 620), _c14)
except Exception:
    pass
layout["Next_month"] = [846, 620, 990, 764]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_16_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-18/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 49, 66)
    canvas.paste(_c15, (382, 1), _c15)
except Exception:
    pass
layout["icon_15"] = [382, 1, 431, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_16_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-18/16_text_Start_Date.png
try:
    _c16 = get_crop(16, 583, 144)
    canvas.paste(_c16, (48, 313), _c16)
except Exception:
    pass
layout["Start_Date"] = [48, 313, 631, 457]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_16_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-18/17_text_End_Date.png
try:
    _c17 = get_crop(17, 620, 114)
    canvas.paste(_c17, (48, 476), _c17)
except Exception:
    pass
layout["End_Date"] = [48, 476, 668, 590]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_16_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-18/18_text_May_2024.png
try:
    _c18 = get_crop(18, 198, 56)
    canvas.paste(_c18, (423, 666), _c18)
except Exception:
    pass
layout["May_2024"] = [423, 666, 621, 722]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_16_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-18/19_text_10.png
try:
    _c19 = get_crop(19, 132, 120)
    canvas.paste(_c19, (720, 1004), _c19)
except Exception:
    pass
layout["10"] = [720, 1004, 852, 1124]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_16_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-18/20_text_11.png
try:
    _c20 = get_crop(20, 132, 120)
    canvas.paste(_c20, (852, 1004), _c20)
except Exception:
    pass
layout["11"] = [852, 1004, 984, 1124]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_16_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-18/21_text_12.png
try:
    _c21 = get_crop(21, 132, 120)
    canvas.paste(_c21, (60, 1124), _c21)
except Exception:
    pass
layout["12"] = [60, 1124, 192, 1244]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_16_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-18/22_text_13.png
try:
    _c22 = get_crop(22, 132, 120)
    canvas.paste(_c22, (192, 1124), _c22)
except Exception:
    pass
layout["13"] = [192, 1124, 324, 1244]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_16_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-18/23_text_14.png
try:
    _c23 = get_crop(23, 132, 120)
    canvas.paste(_c23, (324, 1124), _c23)
except Exception:
    pass
layout["14"] = [324, 1124, 456, 1244]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_16_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-18/24_text_15.png
try:
    _c24 = get_crop(24, 132, 120)
    canvas.paste(_c24, (456, 1124), _c24)
except Exception:
    pass
layout["15"] = [456, 1124, 588, 1244]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_16_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-18/25_text_16.png
try:
    _c25 = get_crop(25, 132, 120)
    canvas.paste(_c25, (588, 1124), _c25)
except Exception:
    pass
layout["16"] = [588, 1124, 720, 1244]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_16_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-18/26_text_17.png
try:
    _c26 = get_crop(26, 132, 120)
    canvas.paste(_c26, (720, 1124), _c26)
except Exception:
    pass
layout["17"] = [720, 1124, 852, 1244]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_16_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-18/27_text_18.png
try:
    _c27 = get_crop(27, 132, 120)
    canvas.paste(_c27, (852, 1124), _c27)
except Exception:
    pass
layout["18"] = [852, 1124, 984, 1244]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_16_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-18/28_text_19.png
try:
    _c28 = get_crop(28, 132, 120)
    canvas.paste(_c28, (60, 1244), _c28)
except Exception:
    pass
layout["19"] = [60, 1244, 192, 1364]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_16_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-18/29_text_20.png
try:
    _c29 = get_crop(29, 132, 120)
    canvas.paste(_c29, (192, 1244), _c29)
except Exception:
    pass
layout["20"] = [192, 1244, 324, 1364]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_16_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-18/30_text_21.png
try:
    _c30 = get_crop(30, 132, 120)
    canvas.paste(_c30, (324, 1244), _c30)
except Exception:
    pass
layout["21"] = [324, 1244, 456, 1364]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_16_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-18/31_text_22.png
try:
    _c31 = get_crop(31, 132, 120)
    canvas.paste(_c31, (456, 1244), _c31)
except Exception:
    pass
layout["22"] = [456, 1244, 588, 1364]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_16_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-18/32_text_23.png
try:
    _c32 = get_crop(32, 132, 120)
    canvas.paste(_c32, (588, 1244), _c32)
except Exception:
    pass
layout["23"] = [588, 1244, 720, 1364]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_16_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-18/33_text_24.png
try:
    _c33 = get_crop(33, 132, 120)
    canvas.paste(_c33, (720, 1244), _c33)
except Exception:
    pass
layout["24"] = [720, 1244, 852, 1364]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_16_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-18/34_text_25.png
try:
    _c34 = get_crop(34, 132, 120)
    canvas.paste(_c34, (852, 1244), _c34)
except Exception:
    pass
layout["25"] = [852, 1244, 984, 1364]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_16_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-18/35_text_26.png
try:
    _c35 = get_crop(35, 132, 120)
    canvas.paste(_c35, (60, 1364), _c35)
except Exception:
    pass
layout["26"] = [60, 1364, 192, 1484]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_16_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-18/36_text_27.png
try:
    _c36 = get_crop(36, 132, 120)
    canvas.paste(_c36, (192, 1364), _c36)
except Exception:
    pass
layout["27"] = [192, 1364, 324, 1484]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_16_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-18/37_text_28.png
try:
    _c37 = get_crop(37, 132, 120)
    canvas.paste(_c37, (324, 1364), _c37)
except Exception:
    pass
layout["28"] = [324, 1364, 456, 1484]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_16_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-18/38_text_29.png
try:
    _c38 = get_crop(38, 132, 120)
    canvas.paste(_c38, (456, 1364), _c38)
except Exception:
    pass
layout["29"] = [456, 1364, 588, 1484]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_16_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-18/39_text_30.png
try:
    _c39 = get_crop(39, 132, 120)
    canvas.paste(_c39, (588, 1364), _c39)
except Exception:
    pass
layout["30"] = [588, 1364, 720, 1484]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_16_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-18/40_text_31.png
try:
    _c40 = get_crop(40, 132, 120)
    canvas.paste(_c40, (720, 1364), _c40)
except Exception:
    pass
layout["31"] = [720, 1364, 852, 1484]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_16_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-18/41_clickable_1.png
try:
    _c41 = get_crop(41, 132, 120)
    canvas.paste(_c41, (456, 884), _c41)
except Exception:
    pass
layout["1"] = [456, 884, 588, 1004]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_16_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-18/42_clickable_2.png
try:
    _c42 = get_crop(42, 132, 120)
    canvas.paste(_c42, (588, 884), _c42)
except Exception:
    pass
layout["2"] = [588, 884, 720, 1004]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_16_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-18/43_clickable_3.png
try:
    _c43 = get_crop(43, 132, 120)
    canvas.paste(_c43, (720, 884), _c43)
except Exception:
    pass
layout["3"] = [720, 884, 852, 1004]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_16_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-18/44_clickable_4.png
try:
    _c44 = get_crop(44, 132, 120)
    canvas.paste(_c44, (852, 884), _c44)
except Exception:
    pass
layout["4"] = [852, 884, 984, 1004]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_16_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-18/45_clickable_5.png
try:
    _c45 = get_crop(45, 132, 120)
    canvas.paste(_c45, (60, 1004), _c45)
except Exception:
    pass
layout["5"] = [60, 1004, 192, 1124]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_16_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-18/46_clickable_6.png
try:
    _c46 = get_crop(46, 132, 120)
    canvas.paste(_c46, (192, 1004), _c46)
except Exception:
    pass
layout["6"] = [192, 1004, 324, 1124]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_16_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-18/47_clickable_7.png
try:
    _c47 = get_crop(47, 132, 120)
    canvas.paste(_c47, (324, 1004), _c47)
except Exception:
    pass
layout["7"] = [324, 1004, 456, 1124]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_16_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-18/48_clickable_8.png
try:
    _c48 = get_crop(48, 132, 120)
    canvas.paste(_c48, (456, 1004), _c48)
except Exception:
    pass
layout["8"] = [456, 1004, 588, 1124]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_16_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-18/49_clickable_9.png
try:
    _c49 = get_crop(49, 132, 120)
    canvas.paste(_c49, (588, 1004), _c49)
except Exception:
    pass
layout["9"] = [588, 1004, 720, 1124]
