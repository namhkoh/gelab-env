# page_id: page_eventbrite_a9f633a394e74f78843aa30bd2792346_15
# screenshot: 2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-17.png
# step_index: 15/18
# task: Open Eventbrite. Set the city to "Los Angeles". Look for Photography workshops happening next week. What is the price of the tickets for first non-promoted event?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Canvas and draw are provided (canvas: PIL Image, draw: ImageDraw)
# Fonts available: font_sm, font_md, font_lg, font_xl

# Colors
BG = "#FFFFFF"             # main background (dominant)
STATUS_BG = "#D6D6D6"      # top status bar grey
CARD_BORDER = "#E7E0EA"    # subtle card border
CARD_SHADOW = "#F3F1F5"    # shadow under cards
ACCENT = "#5A3A66"         # purple accent for outlines/dividers
DIVIDER = "#F0ECF4"        # very light divider

w, h = canvas.size

# Fill main background (ensure consistent)
draw.rectangle((0, 0, w, h), fill=BG)

# Status bar area (top ~72px)
status_h = 72
draw.rectangle((0, 0, w, status_h), fill=STATUS_BG)

# Header / toolbar area (below status bar)
header_top = status_h
header_bottom = 180
# keep header area white but add subtle divider under it
draw.rectangle((0, header_top, w, header_bottom), fill=BG)
draw.line((24, header_bottom - 6, w - 24, header_bottom - 6), fill=DIVIDER, width=1)

# Card background behind Start Date / End Date group
card1_x0, card1_y0 = 32, 200
card1_x1, card1_y1 = w - 32, 580
card_radius = 20
# shadow below card
draw.rectangle((card1_x0 + 4, card1_y1 + 6, card1_x1 + 6, card1_y1 + 10), fill=CARD_SHADOW)
# card body (white fill to keep text/icons visible when pasted) with subtle border
draw.rounded_rectangle((card1_x0, card1_y0, card1_x1, card1_y1),
                       radius=card_radius, fill=BG, outline=CARD_BORDER, width=2)

# Thin separator between the top card and calendar area
sep_y = card1_y1 + 16
draw.line((36, sep_y, w - 36, sep_y), fill=DIVIDER, width=1)

# Calendar container card (holds month and grid)
cal_x0, cal_y0 = 48, 640
cal_x1, cal_y1 = w - 48, 1500
cal_radius = 18
# subtle shadow for calendar card
draw.rectangle((cal_x0 + 6, cal_y1 + 8, cal_x1 + 10, cal_y1 + 12), fill=CARD_SHADOW)
# calendar card background and border
draw.rounded_rectangle((cal_x0, cal_y0, cal_x1, cal_y1),
                       radius=cal_radius, fill=BG, outline=CARD_BORDER, width=2)

# Weekday header divider line (top of calendar grid area)
weekday_div_y = cal_y0 + 60
draw.line((cal_x0 + 20, weekday_div_y, cal_x1 - 20, weekday_div_y), fill=DIVIDER, width=1)

# Subtle vertical grid guides for calendar columns (light, not duplicating numbers)
cols = 7
col_width = (cal_x1 - cal_x0 - 40) / cols
for i in range(1, cols):
    x = cal_x0 + 20 + i * col_width
    draw.line((x, weekday_div_y + 8, x, cal_y1 - 24), fill="#FBF9FB", width=1)

# Horizontal grid guide lines for calendar rows (light)
rows = 6
row_height = (cal_y1 - (weekday_div_y + 24) - 24) / rows
for r in range(1, rows):
    y = weekday_div_y + 24 + r * row_height
    draw.line((cal_x0 + 20, y, cal_x1 - 20, y), fill="#FBF9FB", width=1)

# Bottom "Apply date range" button background (rounded rectangle border and slight shadow)
btn_x0, btn_y0 = 48, 2768
btn_w, btn_h = 1344, 144  # given by detection
btn_x1, btn_y1 = btn_x0 + btn_w, btn_y0 + btn_h
btn_radius = 12
# shadow under button
draw.rectangle((btn_x0 + 4, btn_y1 + 6, btn_x1 + 6, btn_y1 + 12), fill=CARD_SHADOW)
# button border and background (white fill to avoid duplicating pasted label)
draw.rounded_rectangle((btn_x0, btn_y0, btn_x1, btn_y1),
                       radius=btn_radius, fill=BG, outline=ACCENT, width=4)

# Top-to-content subtle left margin divider (visual alignment aid)
draw.line((32, header_bottom + 4, 32, btn_y0 - 24), fill=DIVIDER, width=1)

# End - background and structure drawn. Icons/text/buttons will be pasted on top.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_15_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-17/00_icon_Apply_date_range.png
try:
    _c0 = get_crop(0, 1344, 144)
    canvas.paste(_c0, (48, 2768), _c0)
except Exception:
    pass
layout["Apply_date_range"] = [48, 2768, 1392, 2912]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_15_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-17/01_icon_icon_1.png
try:
    _c1 = get_crop(1, 50, 71)
    canvas.paste(_c1, (1154, 0), _c1)
except Exception:
    pass
layout["icon_1"] = [1154, 0, 1204, 71]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_15_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-17/02_icon_4.51.png
try:
    _c2 = get_crop(2, 59, 65)
    canvas.paste(_c2, (180, 1), _c2)
except Exception:
    pass
layout["4.51"] = [180, 1, 239, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_15_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-17/03_icon_4.51.png
try:
    _c3 = get_crop(3, 61, 66)
    canvas.paste(_c3, (113, 1), _c3)
except Exception:
    pass
layout["4.51"] = [113, 1, 174, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_15_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-17/04_icon_icon_4.png
try:
    _c4 = get_crop(4, 98, 108)
    canvas.paste(_c4, (74, 776), _c4)
except Exception:
    pass
layout["icon_4"] = [74, 776, 172, 884]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_15_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-17/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 61, 62)
    canvas.paste(_c5, (310, 3), _c5)
except Exception:
    pass
layout["icon_5"] = [310, 3, 371, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_15_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-17/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 100, 71)
    canvas.paste(_c6, (1210, 0), _c6)
except Exception:
    pass
layout["icon_6"] = [1210, 0, 1310, 71]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_15_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-17/07_icon_May_2024.png
try:
    _c7 = get_crop(7, 119, 110)
    canvas.paste(_c7, (200, 773), _c7)
except Exception:
    pass
layout["May_2024"] = [200, 773, 319, 883]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_15_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-17/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 53, 70)
    canvas.paste(_c8, (1318, 0), _c8)
except Exception:
    pass
layout["icon_8"] = [1318, 0, 1371, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_15_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-17/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 52, 64)
    canvas.paste(_c9, (247, 2), _c9)
except Exception:
    pass
layout["icon_9"] = [247, 2, 299, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_15_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-17/10_icon_4.51.png
try:
    _c10 = get_crop(10, 144, 144)
    canvas.paste(_c10, (12, 72), _c10)
except Exception:
    pass
layout["4.51"] = [12, 72, 156, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_15_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-17/11_icon_May_2024.png
try:
    _c11 = get_crop(11, 142, 112)
    canvas.paste(_c11, (320, 771), _c11)
except Exception:
    pass
layout["May_2024"] = [320, 771, 462, 883]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_15_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-17/12_icon_May_2024.png
try:
    _c12 = get_crop(12, 130, 113)
    canvas.paste(_c12, (456, 770), _c12)
except Exception:
    pass
layout["May_2024"] = [456, 770, 586, 883]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_15_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-17/13_icon_What_date.png
try:
    _c13 = get_crop(13, 319, 73)
    canvas.paste(_c13, (558, 111), _c13)
except Exception:
    pass
layout["What_date?"] = [558, 111, 877, 184]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_15_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-17/14_icon_4.51.png
try:
    _c14 = get_crop(14, 87, 60)
    canvas.paste(_c14, (17, 3), _c14)
except Exception:
    pass
layout["4.51"] = [17, 3, 104, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_15_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-17/15_icon_End_Date.png
try:
    _c15 = get_crop(15, 144, 144)
    canvas.paste(_c15, (54, 620), _c15)
except Exception:
    pass
layout["End_Date"] = [54, 620, 198, 764]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_15_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-17/16_icon_icon_16.png
try:
    _c16 = get_crop(16, 50, 65)
    canvas.paste(_c16, (382, 2), _c16)
except Exception:
    pass
layout["icon_16"] = [382, 2, 432, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_15_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-17/17_icon_26.png
try:
    _c17 = get_crop(17, 132, 120)
    canvas.paste(_c17, (60, 1364), _c17)
except Exception:
    pass
layout["26"] = [60, 1364, 192, 1484]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_15_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-17/18_icon_27.png
try:
    _c18 = get_crop(18, 132, 120)
    canvas.paste(_c18, (192, 1364), _c18)
except Exception:
    pass
layout["27"] = [192, 1364, 324, 1484]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_15_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-17/19_icon_Next_month.png
try:
    _c19 = get_crop(19, 144, 144)
    canvas.paste(_c19, (846, 620), _c19)
except Exception:
    pass
layout["Next_month"] = [846, 620, 990, 764]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_15_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-17/20_text_Start_Date.png
try:
    _c20 = get_crop(20, 583, 144)
    canvas.paste(_c20, (48, 313), _c20)
except Exception:
    pass
layout["Start_Date"] = [48, 313, 631, 457]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_15_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-17/21_text_End_Date.png
try:
    _c21 = get_crop(21, 638, 114)
    canvas.paste(_c21, (48, 476), _c21)
except Exception:
    pass
layout["End_Date"] = [48, 476, 686, 590]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_15_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-17/22_text_May_2024.png
try:
    _c22 = get_crop(22, 198, 56)
    canvas.paste(_c22, (423, 666), _c22)
except Exception:
    pass
layout["May_2024"] = [423, 666, 621, 722]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_15_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-17/23_text_10.png
try:
    _c23 = get_crop(23, 132, 120)
    canvas.paste(_c23, (720, 1004), _c23)
except Exception:
    pass
layout["10"] = [720, 1004, 852, 1124]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_15_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-17/24_text_11.png
try:
    _c24 = get_crop(24, 132, 120)
    canvas.paste(_c24, (852, 1004), _c24)
except Exception:
    pass
layout["11"] = [852, 1004, 984, 1124]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_15_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-17/25_text_12.png
try:
    _c25 = get_crop(25, 132, 120)
    canvas.paste(_c25, (60, 1124), _c25)
except Exception:
    pass
layout["12"] = [60, 1124, 192, 1244]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_15_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-17/26_text_13.png
try:
    _c26 = get_crop(26, 132, 120)
    canvas.paste(_c26, (192, 1124), _c26)
except Exception:
    pass
layout["13"] = [192, 1124, 324, 1244]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_15_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-17/27_text_14.png
try:
    _c27 = get_crop(27, 132, 120)
    canvas.paste(_c27, (324, 1124), _c27)
except Exception:
    pass
layout["14"] = [324, 1124, 456, 1244]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_15_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-17/28_text_15.png
try:
    _c28 = get_crop(28, 132, 120)
    canvas.paste(_c28, (456, 1124), _c28)
except Exception:
    pass
layout["15"] = [456, 1124, 588, 1244]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_15_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-17/29_text_16.png
try:
    _c29 = get_crop(29, 132, 120)
    canvas.paste(_c29, (588, 1124), _c29)
except Exception:
    pass
layout["16"] = [588, 1124, 720, 1244]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_15_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-17/30_text_17.png
try:
    _c30 = get_crop(30, 132, 120)
    canvas.paste(_c30, (720, 1124), _c30)
except Exception:
    pass
layout["17"] = [720, 1124, 852, 1244]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_15_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-17/31_text_18.png
try:
    _c31 = get_crop(31, 132, 120)
    canvas.paste(_c31, (852, 1124), _c31)
except Exception:
    pass
layout["18"] = [852, 1124, 984, 1244]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_15_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-17/32_text_19.png
try:
    _c32 = get_crop(32, 132, 120)
    canvas.paste(_c32, (60, 1244), _c32)
except Exception:
    pass
layout["19"] = [60, 1244, 192, 1364]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_15_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-17/33_text_20.png
try:
    _c33 = get_crop(33, 132, 120)
    canvas.paste(_c33, (192, 1244), _c33)
except Exception:
    pass
layout["20"] = [192, 1244, 324, 1364]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_15_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-17/34_text_21.png
try:
    _c34 = get_crop(34, 132, 120)
    canvas.paste(_c34, (324, 1244), _c34)
except Exception:
    pass
layout["21"] = [324, 1244, 456, 1364]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_15_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-17/35_text_22.png
try:
    _c35 = get_crop(35, 132, 120)
    canvas.paste(_c35, (456, 1244), _c35)
except Exception:
    pass
layout["22"] = [456, 1244, 588, 1364]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_15_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-17/36_text_23.png
try:
    _c36 = get_crop(36, 132, 120)
    canvas.paste(_c36, (588, 1244), _c36)
except Exception:
    pass
layout["23"] = [588, 1244, 720, 1364]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_15_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-17/37_text_24.png
try:
    _c37 = get_crop(37, 132, 120)
    canvas.paste(_c37, (720, 1244), _c37)
except Exception:
    pass
layout["24"] = [720, 1244, 852, 1364]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_15_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-17/38_text_25.png
try:
    _c38 = get_crop(38, 132, 120)
    canvas.paste(_c38, (852, 1244), _c38)
except Exception:
    pass
layout["25"] = [852, 1244, 984, 1364]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_15_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-17/39_text_28.png
try:
    _c39 = get_crop(39, 132, 120)
    canvas.paste(_c39, (324, 1364), _c39)
except Exception:
    pass
layout["28"] = [324, 1364, 456, 1484]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_15_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-17/40_text_29.png
try:
    _c40 = get_crop(40, 132, 120)
    canvas.paste(_c40, (456, 1364), _c40)
except Exception:
    pass
layout["29"] = [456, 1364, 588, 1484]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_15_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-17/41_text_30.png
try:
    _c41 = get_crop(41, 132, 120)
    canvas.paste(_c41, (588, 1364), _c41)
except Exception:
    pass
layout["30"] = [588, 1364, 720, 1484]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_15_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-17/42_text_31.png
try:
    _c42 = get_crop(42, 132, 120)
    canvas.paste(_c42, (720, 1364), _c42)
except Exception:
    pass
layout["31"] = [720, 1364, 852, 1484]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_15_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-17/43_clickable_1.png
try:
    _c43 = get_crop(43, 132, 120)
    canvas.paste(_c43, (456, 884), _c43)
except Exception:
    pass
layout["1"] = [456, 884, 588, 1004]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_15_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-17/44_clickable_2.png
try:
    _c44 = get_crop(44, 132, 120)
    canvas.paste(_c44, (588, 884), _c44)
except Exception:
    pass
layout["2"] = [588, 884, 720, 1004]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_15_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-17/45_clickable_3.png
try:
    _c45 = get_crop(45, 132, 120)
    canvas.paste(_c45, (720, 884), _c45)
except Exception:
    pass
layout["3"] = [720, 884, 852, 1004]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_15_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-17/46_clickable_4.png
try:
    _c46 = get_crop(46, 132, 120)
    canvas.paste(_c46, (852, 884), _c46)
except Exception:
    pass
layout["4"] = [852, 884, 984, 1004]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_15_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-17/47_clickable_5.png
try:
    _c47 = get_crop(47, 132, 120)
    canvas.paste(_c47, (60, 1004), _c47)
except Exception:
    pass
layout["5"] = [60, 1004, 192, 1124]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_15_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-17/48_clickable_6.png
try:
    _c48 = get_crop(48, 132, 120)
    canvas.paste(_c48, (192, 1004), _c48)
except Exception:
    pass
layout["6"] = [192, 1004, 324, 1124]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_15_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-17/49_clickable_7.png
try:
    _c49 = get_crop(49, 132, 120)
    canvas.paste(_c49, (324, 1004), _c49)
except Exception:
    pass
layout["7"] = [324, 1004, 456, 1124]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_15_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-17/50_clickable_8.png
try:
    _c50 = get_crop(50, 132, 120)
    canvas.paste(_c50, (456, 1004), _c50)
except Exception:
    pass
layout["8"] = [456, 1004, 588, 1124]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_15_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-17/51_clickable_9.png
try:
    _c51 = get_crop(51, 132, 120)
    canvas.paste(_c51, (588, 1004), _c51)
except Exception:
    pass
layout["9"] = [588, 1004, 720, 1124]
