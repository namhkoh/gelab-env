# page_id: page_eventbrite_f502e886c78146dfb2f1efc2a331c781_16
# screenshot: 2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-18.png
# step_index: 16/18
# task: Open Eventbrite. Search for 'music festival' in San Francisco. Set date available from April 30 to May 3. How many events are listed?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Background and structural UI elements for the calendar/date-picker screen.
# Uses provided variables: canvas (PIL Image), draw (ImageDraw), font_sm, font_md, font_lg, font_xl

W, H = canvas.size

# 1) Overall background (slightly warm white to match screenshot)
draw.rectangle((0, 0, W, H), fill=(250, 250, 250))

# 2) Status bar area at top (~72px) - light gray background (icons will be pasted on top)
status_h = 72
draw.rectangle((0, 0, W, status_h), fill=(224, 224, 224))
# subtle bottom divider for the status bar
draw.line((0, status_h - 1, W, status_h - 1), fill=(200, 200, 200))

# 3) Header / toolbar area (below status bar)
header_top = status_h
header_bottom = 200
# keep header background visually consistent (very slightly different to give depth)
draw.rectangle((0, header_top, W, header_bottom), fill=(250, 250, 250))
# thin divider under header
draw.line((24, header_bottom, W - 24, header_bottom), fill=(235, 235, 235), width=1)

# 4) Top content group card (rounded subtle card behind the "Start Date / End Date" area)
card_x0, card_y0 = 36, 240
card_x1, card_y1 = W - 36, 420
draw.rounded_rectangle((card_x0, card_y0, card_x1, card_y1),
                       radius=12,
                       fill=(255, 255, 255),
                       outline=(235, 232, 240))
# faint inner divider lines to suggest sections (not overlapping detected text/icons)
draw.line((card_x0 + 20, card_y0 + 86, card_x1 - 20, card_y0 + 86), fill=(245, 243, 248), width=1)

# 5) Calendar background area (large white rounded rectangle to hold month + grid)
cal_x0, cal_y0 = 48, 636
cal_x1, cal_y1 = W - 48, 1480
draw.rounded_rectangle((cal_x0, cal_y0, cal_x1, cal_y1),
                       radius=8,
                       fill=(255, 255, 255),
                       outline=(245, 243, 248))

# 6) Month header area (centered within calendar) - subtle background band
month_band_h = 64
mb_top = cal_y0 + 8
mb_bottom = mb_top + month_band_h
draw.rectangle((cal_x0 + 24, mb_top, cal_x1 - 24, mb_bottom), fill=(255, 255, 255))
# month band divider line under header
draw.line((cal_x0 + 24, mb_bottom, cal_x1 - 24, mb_bottom), fill=(240, 236, 245), width=1)

# 7) Weekday header row (subtle text-area band) - only background/lines (text/icons will be pasted)
weekday_top = mb_bottom + 24
weekday_bottom = weekday_top + 44
draw.rectangle((cal_x0 + 24, weekday_top, cal_x1 - 24, weekday_bottom), fill=(255, 255, 255))
# faint separator below weekdays
draw.line((cal_x0 + 24, weekday_bottom, cal_x1 - 24, weekday_bottom), fill=(245, 243, 247), width=1)

# 8) Calendar grid lines (behind date numbers) - very light to avoid duplicating numbers
# Based on detected grid positions: 7 columns, columns roughly start near x=60 and step ~132
col_left = 60
col_step = 132
cols = [col_left + i * col_step for i in range(8)]  # 8 boundaries for 7 columns
# Draw vertical guide lines (very faint)
for x in cols:
    # ensure lines stay inside calendar area horizontal padding
    if cal_x0 + 20 < x < cal_x1 - 20:
        draw.line((x, weekday_bottom + 8, x, cal_y1 - 24), fill=(245, 243, 247), width=1)

# Draw horizontal guide lines for rows (approx positions for 5 rows of dates)
row_top = weekday_bottom + 16
row_height = 120
for r in range(1, 6):  # 5 horizontal separators (after each row)
    y = row_top + r * row_height
    if y < cal_y1 - 8:
        draw.line((cal_x0 + 36, y, cal_x1 - 36, y), fill=(245, 243, 247), width=1)

# 9) Subtle left and right safe-area vertical guides (visual structure only)
draw.line((36, header_bottom + 6, 36, H - 200), fill=(248, 247, 250), width=1)
draw.line((W - 36, header_bottom + 6, W - 36, H - 200), fill=(248, 247, 250), width=1)

# 10) Top-to-bottom subtle page divider near bottom to separate content area from apply button
apply_top = 2768  # detected apply button Y
divider_y = apply_top - 20
draw.line((24, divider_y, W - 24, divider_y), fill=(235, 232, 240), width=1)
# light shadow above the bottom interactive area
draw.rectangle((12, divider_y + 2, W - 12, divider_y + 6), fill=(250, 249, 251))

# 11) Peripheral rounded outlines to suggest card/grouping (do not draw or approximate any detected button contents)
# top-left decorative rounded corner stroke behind header content
draw.arc((18, 88, 120, 190), start=180, end=270, fill=(245, 243, 247), width=1)

# Done - structural/background elements only. Icons/text/buttons will be pasted on top separately.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_16_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-18/00_icon_Apply_date_range.png
try:
    _c0 = get_crop(0, 1344, 144)
    canvas.paste(_c0, (48, 2768), _c0)
except Exception:
    pass
layout["Apply_date_range"] = [48, 2768, 1392, 2912]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_16_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-18/01_icon_icon_1.png
try:
    _c1 = get_crop(1, 50, 71)
    canvas.paste(_c1, (1154, 0), _c1)
except Exception:
    pass
layout["icon_1"] = [1154, 0, 1204, 71]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_16_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-18/02_icon_7.19.png
try:
    _c2 = get_crop(2, 57, 63)
    canvas.paste(_c2, (182, 1), _c2)
except Exception:
    pass
layout["7.19"] = [182, 1, 239, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_16_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-18/03_icon_icon_3.png
try:
    _c3 = get_crop(3, 98, 108)
    canvas.paste(_c3, (74, 776), _c3)
except Exception:
    pass
layout["icon_3"] = [74, 776, 172, 884]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_16_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-18/04_icon_icon_4.png
try:
    _c4 = get_crop(4, 101, 71)
    canvas.paste(_c4, (1210, 0), _c4)
except Exception:
    pass
layout["icon_4"] = [1210, 0, 1311, 71]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_16_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-18/05_icon_7.19.png
try:
    _c5 = get_crop(5, 58, 65)
    canvas.paste(_c5, (115, 1), _c5)
except Exception:
    pass
layout["7.19"] = [115, 1, 173, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_16_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-18/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 60, 61)
    canvas.paste(_c6, (311, 3), _c6)
except Exception:
    pass
layout["icon_6"] = [311, 3, 371, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_16_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-18/07_icon_May_2024.png
try:
    _c7 = get_crop(7, 119, 110)
    canvas.paste(_c7, (200, 773), _c7)
except Exception:
    pass
layout["May_2024"] = [200, 773, 319, 883]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_16_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-18/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 53, 70)
    canvas.paste(_c8, (1318, 0), _c8)
except Exception:
    pass
layout["icon_8"] = [1318, 0, 1371, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_16_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-18/09_icon_7.19.png
try:
    _c9 = get_crop(9, 144, 144)
    canvas.paste(_c9, (12, 72), _c9)
except Exception:
    pass
layout["7.19"] = [12, 72, 156, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_16_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-18/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 51, 62)
    canvas.paste(_c10, (248, 3), _c10)
except Exception:
    pass
layout["icon_10"] = [248, 3, 299, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_16_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-18/11_icon_May_2024.png
try:
    _c11 = get_crop(11, 143, 112)
    canvas.paste(_c11, (320, 771), _c11)
except Exception:
    pass
layout["May_2024"] = [320, 771, 463, 883]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_16_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-18/12_icon_May_2024.png
try:
    _c12 = get_crop(12, 131, 113)
    canvas.paste(_c12, (456, 770), _c12)
except Exception:
    pass
layout["May_2024"] = [456, 770, 587, 883]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_16_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-18/13_icon_What_date.png
try:
    _c13 = get_crop(13, 319, 73)
    canvas.paste(_c13, (558, 111), _c13)
except Exception:
    pass
layout["What_date?"] = [558, 111, 877, 184]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_16_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-18/14_icon_End_Date.png
try:
    _c14 = get_crop(14, 144, 144)
    canvas.paste(_c14, (54, 620), _c14)
except Exception:
    pass
layout["End_Date"] = [54, 620, 198, 764]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_16_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-18/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 48, 64)
    canvas.paste(_c15, (383, 2), _c15)
except Exception:
    pass
layout["icon_15"] = [383, 2, 431, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_16_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-18/16_icon_26.png
try:
    _c16 = get_crop(16, 132, 120)
    canvas.paste(_c16, (60, 1364), _c16)
except Exception:
    pass
layout["26"] = [60, 1364, 192, 1484]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_16_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-18/17_icon_27.png
try:
    _c17 = get_crop(17, 132, 120)
    canvas.paste(_c17, (192, 1364), _c17)
except Exception:
    pass
layout["27"] = [192, 1364, 324, 1484]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_16_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-18/18_icon_Next_month.png
try:
    _c18 = get_crop(18, 144, 144)
    canvas.paste(_c18, (846, 620), _c18)
except Exception:
    pass
layout["Next_month"] = [846, 620, 990, 764]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_16_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-18/19_icon_7.19.png
try:
    _c19 = get_crop(19, 92, 63)
    canvas.paste(_c19, (16, 2), _c19)
except Exception:
    pass
layout["7.19"] = [16, 2, 108, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_16_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-18/20_text_Start_Date.png
try:
    _c20 = get_crop(20, 587, 144)
    canvas.paste(_c20, (48, 313), _c20)
except Exception:
    pass
layout["Start_Date"] = [48, 313, 635, 457]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_16_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-18/21_text_End_Date.png
try:
    _c21 = get_crop(21, 638, 114)
    canvas.paste(_c21, (48, 476), _c21)
except Exception:
    pass
layout["End_Date"] = [48, 476, 686, 590]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_16_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-18/22_text_May_2024.png
try:
    _c22 = get_crop(22, 198, 56)
    canvas.paste(_c22, (423, 666), _c22)
except Exception:
    pass
layout["May_2024"] = [423, 666, 621, 722]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_16_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-18/23_text_10.png
try:
    _c23 = get_crop(23, 132, 120)
    canvas.paste(_c23, (720, 1004), _c23)
except Exception:
    pass
layout["10"] = [720, 1004, 852, 1124]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_16_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-18/24_text_11.png
try:
    _c24 = get_crop(24, 132, 120)
    canvas.paste(_c24, (852, 1004), _c24)
except Exception:
    pass
layout["11"] = [852, 1004, 984, 1124]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_16_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-18/25_text_12.png
try:
    _c25 = get_crop(25, 132, 120)
    canvas.paste(_c25, (60, 1124), _c25)
except Exception:
    pass
layout["12"] = [60, 1124, 192, 1244]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_16_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-18/26_text_13.png
try:
    _c26 = get_crop(26, 132, 120)
    canvas.paste(_c26, (192, 1124), _c26)
except Exception:
    pass
layout["13"] = [192, 1124, 324, 1244]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_16_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-18/27_text_14.png
try:
    _c27 = get_crop(27, 132, 120)
    canvas.paste(_c27, (324, 1124), _c27)
except Exception:
    pass
layout["14"] = [324, 1124, 456, 1244]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_16_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-18/28_text_15.png
try:
    _c28 = get_crop(28, 132, 120)
    canvas.paste(_c28, (456, 1124), _c28)
except Exception:
    pass
layout["15"] = [456, 1124, 588, 1244]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_16_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-18/29_text_16.png
try:
    _c29 = get_crop(29, 132, 120)
    canvas.paste(_c29, (588, 1124), _c29)
except Exception:
    pass
layout["16"] = [588, 1124, 720, 1244]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_16_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-18/30_text_17.png
try:
    _c30 = get_crop(30, 132, 120)
    canvas.paste(_c30, (720, 1124), _c30)
except Exception:
    pass
layout["17"] = [720, 1124, 852, 1244]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_16_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-18/31_text_18.png
try:
    _c31 = get_crop(31, 132, 120)
    canvas.paste(_c31, (852, 1124), _c31)
except Exception:
    pass
layout["18"] = [852, 1124, 984, 1244]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_16_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-18/32_text_19.png
try:
    _c32 = get_crop(32, 132, 120)
    canvas.paste(_c32, (60, 1244), _c32)
except Exception:
    pass
layout["19"] = [60, 1244, 192, 1364]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_16_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-18/33_text_20.png
try:
    _c33 = get_crop(33, 132, 120)
    canvas.paste(_c33, (192, 1244), _c33)
except Exception:
    pass
layout["20"] = [192, 1244, 324, 1364]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_16_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-18/34_text_21.png
try:
    _c34 = get_crop(34, 132, 120)
    canvas.paste(_c34, (324, 1244), _c34)
except Exception:
    pass
layout["21"] = [324, 1244, 456, 1364]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_16_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-18/35_text_22.png
try:
    _c35 = get_crop(35, 132, 120)
    canvas.paste(_c35, (456, 1244), _c35)
except Exception:
    pass
layout["22"] = [456, 1244, 588, 1364]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_16_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-18/36_text_23.png
try:
    _c36 = get_crop(36, 132, 120)
    canvas.paste(_c36, (588, 1244), _c36)
except Exception:
    pass
layout["23"] = [588, 1244, 720, 1364]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_16_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-18/37_text_24.png
try:
    _c37 = get_crop(37, 132, 120)
    canvas.paste(_c37, (720, 1244), _c37)
except Exception:
    pass
layout["24"] = [720, 1244, 852, 1364]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_16_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-18/38_text_25.png
try:
    _c38 = get_crop(38, 132, 120)
    canvas.paste(_c38, (852, 1244), _c38)
except Exception:
    pass
layout["25"] = [852, 1244, 984, 1364]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_16_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-18/39_text_28.png
try:
    _c39 = get_crop(39, 132, 120)
    canvas.paste(_c39, (324, 1364), _c39)
except Exception:
    pass
layout["28"] = [324, 1364, 456, 1484]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_16_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-18/40_text_29.png
try:
    _c40 = get_crop(40, 132, 120)
    canvas.paste(_c40, (456, 1364), _c40)
except Exception:
    pass
layout["29"] = [456, 1364, 588, 1484]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_16_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-18/41_text_30.png
try:
    _c41 = get_crop(41, 132, 120)
    canvas.paste(_c41, (588, 1364), _c41)
except Exception:
    pass
layout["30"] = [588, 1364, 720, 1484]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_16_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-18/42_text_31.png
try:
    _c42 = get_crop(42, 132, 120)
    canvas.paste(_c42, (720, 1364), _c42)
except Exception:
    pass
layout["31"] = [720, 1364, 852, 1484]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_16_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-18/43_clickable_1.png
try:
    _c43 = get_crop(43, 132, 120)
    canvas.paste(_c43, (456, 884), _c43)
except Exception:
    pass
layout["1"] = [456, 884, 588, 1004]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_16_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-18/44_clickable_2.png
try:
    _c44 = get_crop(44, 132, 120)
    canvas.paste(_c44, (588, 884), _c44)
except Exception:
    pass
layout["2"] = [588, 884, 720, 1004]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_16_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-18/45_clickable_3.png
try:
    _c45 = get_crop(45, 132, 120)
    canvas.paste(_c45, (720, 884), _c45)
except Exception:
    pass
layout["3"] = [720, 884, 852, 1004]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_16_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-18/46_clickable_4.png
try:
    _c46 = get_crop(46, 132, 120)
    canvas.paste(_c46, (852, 884), _c46)
except Exception:
    pass
layout["4"] = [852, 884, 984, 1004]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_16_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-18/47_clickable_5.png
try:
    _c47 = get_crop(47, 132, 120)
    canvas.paste(_c47, (60, 1004), _c47)
except Exception:
    pass
layout["5"] = [60, 1004, 192, 1124]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_16_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-18/48_clickable_6.png
try:
    _c48 = get_crop(48, 132, 120)
    canvas.paste(_c48, (192, 1004), _c48)
except Exception:
    pass
layout["6"] = [192, 1004, 324, 1124]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_16_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-18/49_clickable_7.png
try:
    _c49 = get_crop(49, 132, 120)
    canvas.paste(_c49, (324, 1004), _c49)
except Exception:
    pass
layout["7"] = [324, 1004, 456, 1124]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_16_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-18/50_clickable_8.png
try:
    _c50 = get_crop(50, 132, 120)
    canvas.paste(_c50, (456, 1004), _c50)
except Exception:
    pass
layout["8"] = [456, 1004, 588, 1124]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_16_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-18/51_clickable_9.png
try:
    _c51 = get_crop(51, 132, 120)
    canvas.paste(_c51, (588, 1004), _c51)
except Exception:
    pass
layout["9"] = [588, 1004, 720, 1124]
