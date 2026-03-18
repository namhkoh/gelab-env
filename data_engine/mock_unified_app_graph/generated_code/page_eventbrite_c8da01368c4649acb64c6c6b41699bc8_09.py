# page_id: page_eventbrite_c8da01368c4649acb64c6c6b41699bc8_09
# screenshot: 2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-11.png
# step_index: 9/13
# task: Open Eventbrite. Look up "Animal" events. Filter by events happening next week. Select the first event - who is the organizer?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Background fill (match screenshot's very light off-white)
draw.rectangle((0, 0, 1440, 2960), fill=(250, 250, 252))

# Status bar area at top (~72px) - light grey background like Android status bar
status_bar_h = 72
draw.rectangle((0, 0, 1440, status_bar_h), fill=(198, 198, 198))

# Header / toolbar area (below status bar) - keep very subtle off-white but add a soft divider/shadow
header_top = status_bar_h
header_bottom = 220
draw.rectangle((0, header_top, 1440, header_bottom), fill=(250, 250, 252))
# subtle bottom divider line for header
draw.line((48, header_bottom, 1392, header_bottom), fill=(232, 228, 240), width=1)

# Card background for the Start Date / End Date area (rounded rectangle behind text groups)
card1_x0, card1_y0 = 48, 240
card1_x1, card1_y1 = 1392, 520
draw.rounded_rectangle((card1_x0, card1_y0, card1_x1, card1_y1),
                       radius=14,
                       fill=(255, 255, 255),
                       outline=(230, 225, 240),
                       width=2)

# Soft shadow under that card (very subtle)
shadow_y = card1_y1 + 6
draw.line((card1_x0 + 6, shadow_y, card1_x1 - 6, shadow_y), fill=(245, 244, 247), width=8)

# Calendar container background (area holding month title and grid)
cal_x0, cal_y0 = 96, 560
cal_x1, cal_y1 = 1344, 1560
draw.rounded_rectangle((cal_x0, cal_y0, cal_x1, cal_y1),
                       radius=18,
                       fill=(250, 250, 252),
                       outline=(245, 244, 247),
                       width=1)

# Subtle divider between month title area and calendar grid
month_div_y = cal_y0 + 80
draw.line((cal_x0 + 28, month_div_y, cal_x1 - 28, month_div_y), fill=(240, 238, 244), width=1)

# Light row separators for calendar grid (do not draw numbers or cells, only subtle separators)
# Draw 6 horizontal separators to suggest rows
row_count = 6
row_height = (cal_y1 - month_div_y - 24) / row_count
for i in range(1, row_count):
    y = int(month_div_y + 24 + i * row_height)
    draw.line((cal_x0 + 24, y, cal_x1 - 24, y), fill=(247, 246, 249), width=1)

# Vertical separators for calendar columns (7 days)
col_count = 7
col_width = (cal_x1 - cal_x0 - 48) / col_count
for i in range(1, col_count):
    x = int(cal_x0 + 24 + i * col_width)
    draw.line((x, month_div_y + 20, x, cal_y1 - 20), fill=(247, 246, 249), width=1)

# Large empty content background area (main white space below calendar)
content_top = cal_y1 + 24
draw.rectangle((48, content_top, 1392, 2680), fill=(250, 250, 252))

# Separator line above the bottom action area (keep it above the auto-pasted button)
separator_y = 2728
draw.line((48, separator_y, 1392, separator_y), fill=(230, 225, 236), width=2)

# Soft rounded outline to suggest the bottom action area without drawing the actual button
# (outline stops short of the exact button area to avoid duplicating detected element)
draw.rounded_rectangle((72, separator_y + 12, 1368, separator_y + 12 + 96),
                       radius=12,
                       outline=(210, 205, 216),
                       width=2,
                       fill=None)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_09_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-11/00_icon_Apply_date_range.png
try:
    _c0 = get_crop(0, 1344, 144)
    canvas.paste(_c0, (48, 2768), _c0)
except Exception:
    pass
layout["Apply_date_range"] = [48, 2768, 1392, 2912]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_09_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-11/01_icon_28.png
try:
    _c1 = get_crop(1, 132, 120)
    canvas.paste(_c1, (60, 1364), _c1)
except Exception:
    pass
layout["28"] = [60, 1364, 192, 1484]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_09_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-11/02_icon_5.15.png
try:
    _c2 = get_crop(2, 59, 64)
    canvas.paste(_c2, (181, 1), _c2)
except Exception:
    pass
layout["5.15"] = [181, 1, 240, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_09_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-11/03_icon_icon_3.png
try:
    _c3 = get_crop(3, 60, 62)
    canvas.paste(_c3, (311, 3), _c3)
except Exception:
    pass
layout["icon_3"] = [311, 3, 371, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_09_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-11/04_icon_5.15.png
try:
    _c4 = get_crop(4, 58, 66)
    canvas.paste(_c4, (116, 1), _c4)
except Exception:
    pass
layout["5.15"] = [116, 1, 174, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_09_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-11/05_icon_5.15.png
try:
    _c5 = get_crop(5, 144, 144)
    canvas.paste(_c5, (12, 72), _c5)
except Exception:
    pass
layout["5.15"] = [12, 72, 156, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_09_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-11/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 52, 64)
    canvas.paste(_c6, (247, 2), _c6)
except Exception:
    pass
layout["icon_6"] = [247, 2, 299, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_09_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-11/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 57, 70)
    canvas.paste(_c7, (1316, 0), _c7)
except Exception:
    pass
layout["icon_7"] = [1316, 0, 1373, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_09_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-11/08_icon_29.png
try:
    _c8 = get_crop(8, 132, 120)
    canvas.paste(_c8, (192, 1364), _c8)
except Exception:
    pass
layout["29"] = [192, 1364, 324, 1484]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_09_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-11/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 82, 69)
    canvas.paste(_c9, (1212, 0), _c9)
except Exception:
    pass
layout["icon_9"] = [1212, 0, 1294, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_09_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-11/10_icon_What_date.png
try:
    _c10 = get_crop(10, 319, 71)
    canvas.paste(_c10, (558, 112), _c10)
except Exception:
    pass
layout["What_date?"] = [558, 112, 877, 183]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_09_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-11/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 43, 65)
    canvas.paste(_c11, (1272, 2), _c11)
except Exception:
    pass
layout["icon_11"] = [1272, 2, 1315, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_09_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-11/12_icon_30.png
try:
    _c12 = get_crop(12, 132, 120)
    canvas.paste(_c12, (324, 1364), _c12)
except Exception:
    pass
layout["30"] = [324, 1364, 456, 1484]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_09_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-11/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 50, 64)
    canvas.paste(_c13, (382, 2), _c13)
except Exception:
    pass
layout["icon_13"] = [382, 2, 432, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_09_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-11/14_icon_Next_month.png
try:
    _c14 = get_crop(14, 144, 144)
    canvas.paste(_c14, (846, 620), _c14)
except Exception:
    pass
layout["Next_month"] = [846, 620, 990, 764]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_09_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-11/15_icon_27.png
try:
    _c15 = get_crop(15, 132, 120)
    canvas.paste(_c15, (852, 1244), _c15)
except Exception:
    pass
layout["27"] = [852, 1244, 984, 1364]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_09_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-11/16_text_5.15.png
try:
    _c16 = get_crop(16, 92, 43)
    canvas.paste(_c16, (22, 17), _c16)
except Exception:
    pass
layout["5.15"] = [22, 17, 114, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_09_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-11/17_text_Start_Date.png
try:
    _c17 = get_crop(17, 583, 144)
    canvas.paste(_c17, (48, 313), _c17)
except Exception:
    pass
layout["Start_Date"] = [48, 313, 631, 457]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_09_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-11/18_text_End_Date.png
try:
    _c18 = get_crop(18, 638, 114)
    canvas.paste(_c18, (48, 476), _c18)
except Exception:
    pass
layout["End_Date"] = [48, 476, 686, 590]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_09_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-11/19_text_April_2024.png
try:
    _c19 = get_crop(19, 202, 54)
    canvas.paste(_c19, (421, 666), _c19)
except Exception:
    pass
layout["April_2024"] = [421, 666, 623, 720]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_09_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-11/20_text_10.png
try:
    _c20 = get_crop(20, 132, 120)
    canvas.paste(_c20, (456, 1004), _c20)
except Exception:
    pass
layout["10"] = [456, 1004, 588, 1124]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_09_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-11/21_text_11.png
try:
    _c21 = get_crop(21, 132, 120)
    canvas.paste(_c21, (588, 1004), _c21)
except Exception:
    pass
layout["11"] = [588, 1004, 720, 1124]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_09_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-11/22_text_12.png
try:
    _c22 = get_crop(22, 132, 120)
    canvas.paste(_c22, (720, 1004), _c22)
except Exception:
    pass
layout["12"] = [720, 1004, 852, 1124]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_09_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-11/23_text_13.png
try:
    _c23 = get_crop(23, 132, 120)
    canvas.paste(_c23, (852, 1004), _c23)
except Exception:
    pass
layout["13"] = [852, 1004, 984, 1124]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_09_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-11/24_text_14.png
try:
    _c24 = get_crop(24, 132, 120)
    canvas.paste(_c24, (60, 1124), _c24)
except Exception:
    pass
layout["14"] = [60, 1124, 192, 1244]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_09_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-11/25_text_15.png
try:
    _c25 = get_crop(25, 132, 120)
    canvas.paste(_c25, (192, 1124), _c25)
except Exception:
    pass
layout["15"] = [192, 1124, 324, 1244]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_09_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-11/26_text_16.png
try:
    _c26 = get_crop(26, 132, 120)
    canvas.paste(_c26, (324, 1124), _c26)
except Exception:
    pass
layout["16"] = [324, 1124, 456, 1244]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_09_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-11/27_text_17.png
try:
    _c27 = get_crop(27, 132, 120)
    canvas.paste(_c27, (456, 1124), _c27)
except Exception:
    pass
layout["17"] = [456, 1124, 588, 1244]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_09_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-11/28_text_18.png
try:
    _c28 = get_crop(28, 132, 120)
    canvas.paste(_c28, (588, 1124), _c28)
except Exception:
    pass
layout["18"] = [588, 1124, 720, 1244]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_09_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-11/29_text_19.png
try:
    _c29 = get_crop(29, 132, 120)
    canvas.paste(_c29, (720, 1124), _c29)
except Exception:
    pass
layout["19"] = [720, 1124, 852, 1244]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_09_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-11/30_text_20.png
try:
    _c30 = get_crop(30, 132, 120)
    canvas.paste(_c30, (852, 1124), _c30)
except Exception:
    pass
layout["20"] = [852, 1124, 984, 1244]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_09_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-11/31_text_21.png
try:
    _c31 = get_crop(31, 132, 120)
    canvas.paste(_c31, (60, 1244), _c31)
except Exception:
    pass
layout["21"] = [60, 1244, 192, 1364]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_09_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-11/32_text_22.png
try:
    _c32 = get_crop(32, 132, 120)
    canvas.paste(_c32, (192, 1244), _c32)
except Exception:
    pass
layout["22"] = [192, 1244, 324, 1364]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_09_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-11/33_text_23.png
try:
    _c33 = get_crop(33, 132, 120)
    canvas.paste(_c33, (324, 1244), _c33)
except Exception:
    pass
layout["23"] = [324, 1244, 456, 1364]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_09_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-11/34_text_24.png
try:
    _c34 = get_crop(34, 132, 120)
    canvas.paste(_c34, (456, 1244), _c34)
except Exception:
    pass
layout["24"] = [456, 1244, 588, 1364]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_09_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-11/35_text_25.png
try:
    _c35 = get_crop(35, 132, 120)
    canvas.paste(_c35, (588, 1244), _c35)
except Exception:
    pass
layout["25"] = [588, 1244, 720, 1364]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_09_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-11/36_text_26.png
try:
    _c36 = get_crop(36, 132, 120)
    canvas.paste(_c36, (720, 1244), _c36)
except Exception:
    pass
layout["26"] = [720, 1244, 852, 1364]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_09_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-11/37_clickable_1.png
try:
    _c37 = get_crop(37, 132, 120)
    canvas.paste(_c37, (192, 884), _c37)
except Exception:
    pass
layout["1"] = [192, 884, 324, 1004]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_09_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-11/38_clickable_2.png
try:
    _c38 = get_crop(38, 132, 120)
    canvas.paste(_c38, (324, 884), _c38)
except Exception:
    pass
layout["2"] = [324, 884, 456, 1004]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_09_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-11/39_clickable_3.png
try:
    _c39 = get_crop(39, 132, 120)
    canvas.paste(_c39, (456, 884), _c39)
except Exception:
    pass
layout["3"] = [456, 884, 588, 1004]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_09_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-11/40_clickable_4.png
try:
    _c40 = get_crop(40, 132, 120)
    canvas.paste(_c40, (588, 884), _c40)
except Exception:
    pass
layout["4"] = [588, 884, 720, 1004]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_09_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-11/41_clickable_5.png
try:
    _c41 = get_crop(41, 132, 120)
    canvas.paste(_c41, (720, 884), _c41)
except Exception:
    pass
layout["5"] = [720, 884, 852, 1004]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_09_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-11/42_clickable_6.png
try:
    _c42 = get_crop(42, 132, 120)
    canvas.paste(_c42, (852, 884), _c42)
except Exception:
    pass
layout["6"] = [852, 884, 984, 1004]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_09_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-11/43_clickable_7.png
try:
    _c43 = get_crop(43, 132, 120)
    canvas.paste(_c43, (60, 1004), _c43)
except Exception:
    pass
layout["7"] = [60, 1004, 192, 1124]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_09_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-11/44_clickable_8.png
try:
    _c44 = get_crop(44, 132, 120)
    canvas.paste(_c44, (192, 1004), _c44)
except Exception:
    pass
layout["8"] = [192, 1004, 324, 1124]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_09_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-11/45_clickable_9.png
try:
    _c45 = get_crop(45, 132, 120)
    canvas.paste(_c45, (324, 1004), _c45)
except Exception:
    pass
layout["9"] = [324, 1004, 456, 1124]
