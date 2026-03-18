# page_id: page_eventbrite_892383ad791c46b6aa65c632ee882bf0_11
# screenshot: 2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-13.png
# step_index: 11/12
# task: Open Eventbrite. Search for online "Music" events happening next weekend.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw UI background and structural elements for the calendar page.
# Available variables: canvas (PIL Image 1440x2960), draw (ImageDraw), fonts: font_sm, font_md, font_lg, font_xl

# Fill overall background (dominant color: white/off-white)
draw.rectangle([(0, 0), (1440, 2960)], fill="#FFFFFF")

# Status bar area (top ~72px) - light gray background to match screenshot status bar
status_bar_h = 72
draw.rectangle([(0, 0), (1440, status_bar_h)], fill="#E9E9E9")

# Header area (toolbar) under status bar (keeps white but provide a subtle bottom divider)
header_top = status_bar_h
header_bottom = 168
draw.rectangle([(0, header_top), (1440, header_bottom)], fill="#FFFFFF")
# subtle divider under header
draw.line([(32, header_bottom), (1408, header_bottom)], fill="#EDEAF5", width=2)

# Rounded background card for Start Date block (behind detected Start Date text)
start_card = (28, 260, 1412, 420)  # left, top, right, bottom
draw.rounded_rectangle(start_card, radius=14, fill="#FBF9FF", outline=None)

# Rounded background card for End Date block (behind detected End Date text)
end_card = (28, 420, 1412, 540)
draw.rounded_rectangle(end_card, radius=14, fill="#FBF9FF", outline=None)
# subtle inner separator between start & end card areas (thin)
draw.line([(40, 440), (1400, 440)], fill="#F0EDF6", width=1)

# Month navigation / calendar container background (subtle grouping behind calendar days)
cal_x0 = 40
cal_x1 = 1000
cal_y0 = 720
cal_y1 = 1440
draw.rectangle([(cal_x0, cal_y0), (cal_x1, cal_y1)], fill="#FFFFFF")  # keep white but group area

# Subtle top divider for calendar container
draw.line([(cal_x0 + 20, cal_y0 + 36), (cal_x1 - 20, cal_y0 + 36)], fill="#F3F1F8", width=1)

# Light guide lines for calendar rows (very subtle separators, not interfering with day content)
# Rows roughly spaced; we'll draw faint horizontal rules as visual structure
row_y = cal_y0 + 80
for i in range(5):
    draw.line([(cal_x0 + 10, row_y), (cal_x1 - 10, row_y)], fill="#FBF8FD", width=1)
    row_y += 120

# Vertical subtle separators for weekday columns (very faint)
col_x = cal_x0 + 60
for i in range(6):
    draw.line([(col_x, cal_y0 + 40), (col_x, cal_y1 - 20)], fill="#FBF8FD", width=1)
    col_x += 132

# Bottom area: leave space for the Apply button (do not draw the button itself).
# Add a faint top border to the button area so pasted button has a clear separation.
apply_bar_top = 2768 - 8  # slightly above detected apply button area
draw.line([(40, apply_bar_top), (1400, apply_bar_top)], fill="#ECE9F2", width=2)

# Tiny decorative left and right safe-area rounded borders (subtle)
draw.rounded_rectangle([(28, 2768), (1412, 2920)], radius=10, outline="#E6E0EE", width=2)

# Small shadow under header & cards to provide depth (very subtle using thin strokes)
draw.line([(40, header_bottom + 6), (1400, header_bottom + 6)], fill="#F6F4FA", width=1)
draw.line([(40, end_card[3] + 6), (1400, end_card[3] + 6)], fill="#F6F4FA", width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_11_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-13/00_icon_Apply_date_range.png
try:
    _c0 = get_crop(0, 1344, 144)
    canvas.paste(_c0, (48, 2768), _c0)
except Exception:
    pass
layout["Apply_date_range"] = [48, 2768, 1392, 2912]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_11_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-13/01_icon_5.23.png
try:
    _c1 = get_crop(1, 60, 65)
    canvas.paste(_c1, (180, 1), _c1)
except Exception:
    pass
layout["5.23"] = [180, 1, 240, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_11_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-13/02_icon_5.23.png
try:
    _c2 = get_crop(2, 60, 65)
    canvas.paste(_c2, (114, 1), _c2)
except Exception:
    pass
layout["5.23"] = [114, 1, 174, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_11_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-13/03_icon_icon_3.png
try:
    _c3 = get_crop(3, 63, 63)
    canvas.paste(_c3, (309, 2), _c3)
except Exception:
    pass
layout["icon_3"] = [309, 2, 372, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_11_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-13/04_icon_5.23.png
try:
    _c4 = get_crop(4, 144, 144)
    canvas.paste(_c4, (12, 72), _c4)
except Exception:
    pass
layout["5.23"] = [12, 72, 156, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_11_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-13/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 56, 71)
    canvas.paste(_c5, (1316, 0), _c5)
except Exception:
    pass
layout["icon_5"] = [1316, 0, 1372, 71]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_11_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-13/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 51, 63)
    canvas.paste(_c6, (248, 2), _c6)
except Exception:
    pass
layout["icon_6"] = [248, 2, 299, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_11_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-13/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 91, 70)
    canvas.paste(_c7, (1212, 0), _c7)
except Exception:
    pass
layout["icon_7"] = [1212, 0, 1303, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_11_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-13/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 97, 107)
    canvas.paste(_c8, (74, 777), _c8)
except Exception:
    pass
layout["icon_8"] = [74, 777, 171, 884]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_11_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-13/09_icon_What_date.png
try:
    _c9 = get_crop(9, 319, 72)
    canvas.paste(_c9, (558, 111), _c9)
except Exception:
    pass
layout["What_date?"] = [558, 111, 877, 183]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_11_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-13/10_icon_5.23.png
try:
    _c10 = get_crop(10, 91, 62)
    canvas.paste(_c10, (17, 2), _c10)
except Exception:
    pass
layout["5.23"] = [17, 2, 108, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_11_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-13/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 110, 108)
    canvas.paste(_c11, (201, 775), _c11)
except Exception:
    pass
layout["icon_11"] = [201, 775, 311, 883]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_11_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-13/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 49, 67)
    canvas.paste(_c12, (382, 1), _c12)
except Exception:
    pass
layout["icon_12"] = [382, 1, 431, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_11_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-13/13_icon_End_Date.png
try:
    _c13 = get_crop(13, 144, 144)
    canvas.paste(_c13, (54, 620), _c13)
except Exception:
    pass
layout["End_Date"] = [54, 620, 198, 764]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_11_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-13/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 41, 66)
    canvas.paste(_c14, (1274, 0), _c14)
except Exception:
    pass
layout["icon_14"] = [1274, 0, 1315, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_11_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-13/15_icon_Next_month.png
try:
    _c15 = get_crop(15, 144, 144)
    canvas.paste(_c15, (846, 620), _c15)
except Exception:
    pass
layout["Next_month"] = [846, 620, 990, 764]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_11_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-13/16_icon_May_2024.png
try:
    _c16 = get_crop(16, 118, 110)
    canvas.paste(_c16, (330, 774), _c16)
except Exception:
    pass
layout["May_2024"] = [330, 774, 448, 884]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_11_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-13/17_text_Start_Date.png
try:
    _c17 = get_crop(17, 583, 144)
    canvas.paste(_c17, (48, 313), _c17)
except Exception:
    pass
layout["Start_Date"] = [48, 313, 631, 457]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_11_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-13/18_text_End_Date.png
try:
    _c18 = get_crop(18, 620, 114)
    canvas.paste(_c18, (48, 476), _c18)
except Exception:
    pass
layout["End_Date"] = [48, 476, 668, 590]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_11_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-13/19_text_May_2024.png
try:
    _c19 = get_crop(19, 198, 56)
    canvas.paste(_c19, (423, 666), _c19)
except Exception:
    pass
layout["May_2024"] = [423, 666, 621, 722]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_11_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-13/20_text_10.png
try:
    _c20 = get_crop(20, 132, 120)
    canvas.paste(_c20, (720, 1004), _c20)
except Exception:
    pass
layout["10"] = [720, 1004, 852, 1124]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_11_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-13/21_text_11.png
try:
    _c21 = get_crop(21, 132, 120)
    canvas.paste(_c21, (852, 1004), _c21)
except Exception:
    pass
layout["11"] = [852, 1004, 984, 1124]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_11_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-13/22_text_12.png
try:
    _c22 = get_crop(22, 132, 120)
    canvas.paste(_c22, (60, 1124), _c22)
except Exception:
    pass
layout["12"] = [60, 1124, 192, 1244]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_11_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-13/23_text_13.png
try:
    _c23 = get_crop(23, 132, 120)
    canvas.paste(_c23, (192, 1124), _c23)
except Exception:
    pass
layout["13"] = [192, 1124, 324, 1244]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_11_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-13/24_text_14.png
try:
    _c24 = get_crop(24, 132, 120)
    canvas.paste(_c24, (324, 1124), _c24)
except Exception:
    pass
layout["14"] = [324, 1124, 456, 1244]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_11_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-13/25_text_15.png
try:
    _c25 = get_crop(25, 132, 120)
    canvas.paste(_c25, (456, 1124), _c25)
except Exception:
    pass
layout["15"] = [456, 1124, 588, 1244]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_11_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-13/26_text_16.png
try:
    _c26 = get_crop(26, 132, 120)
    canvas.paste(_c26, (588, 1124), _c26)
except Exception:
    pass
layout["16"] = [588, 1124, 720, 1244]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_11_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-13/27_text_17.png
try:
    _c27 = get_crop(27, 132, 120)
    canvas.paste(_c27, (720, 1124), _c27)
except Exception:
    pass
layout["17"] = [720, 1124, 852, 1244]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_11_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-13/28_text_18.png
try:
    _c28 = get_crop(28, 132, 120)
    canvas.paste(_c28, (852, 1124), _c28)
except Exception:
    pass
layout["18"] = [852, 1124, 984, 1244]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_11_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-13/29_text_19.png
try:
    _c29 = get_crop(29, 132, 120)
    canvas.paste(_c29, (60, 1244), _c29)
except Exception:
    pass
layout["19"] = [60, 1244, 192, 1364]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_11_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-13/30_text_20.png
try:
    _c30 = get_crop(30, 132, 120)
    canvas.paste(_c30, (192, 1244), _c30)
except Exception:
    pass
layout["20"] = [192, 1244, 324, 1364]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_11_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-13/31_text_21.png
try:
    _c31 = get_crop(31, 132, 120)
    canvas.paste(_c31, (324, 1244), _c31)
except Exception:
    pass
layout["21"] = [324, 1244, 456, 1364]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_11_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-13/32_text_22.png
try:
    _c32 = get_crop(32, 132, 120)
    canvas.paste(_c32, (456, 1244), _c32)
except Exception:
    pass
layout["22"] = [456, 1244, 588, 1364]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_11_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-13/33_text_23.png
try:
    _c33 = get_crop(33, 132, 120)
    canvas.paste(_c33, (588, 1244), _c33)
except Exception:
    pass
layout["23"] = [588, 1244, 720, 1364]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_11_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-13/34_text_24.png
try:
    _c34 = get_crop(34, 132, 120)
    canvas.paste(_c34, (720, 1244), _c34)
except Exception:
    pass
layout["24"] = [720, 1244, 852, 1364]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_11_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-13/35_text_25.png
try:
    _c35 = get_crop(35, 132, 120)
    canvas.paste(_c35, (852, 1244), _c35)
except Exception:
    pass
layout["25"] = [852, 1244, 984, 1364]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_11_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-13/36_text_26.png
try:
    _c36 = get_crop(36, 132, 120)
    canvas.paste(_c36, (60, 1364), _c36)
except Exception:
    pass
layout["26"] = [60, 1364, 192, 1484]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_11_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-13/37_text_27.png
try:
    _c37 = get_crop(37, 132, 120)
    canvas.paste(_c37, (192, 1364), _c37)
except Exception:
    pass
layout["27"] = [192, 1364, 324, 1484]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_11_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-13/38_text_28.png
try:
    _c38 = get_crop(38, 132, 120)
    canvas.paste(_c38, (324, 1364), _c38)
except Exception:
    pass
layout["28"] = [324, 1364, 456, 1484]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_11_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-13/39_text_29.png
try:
    _c39 = get_crop(39, 132, 120)
    canvas.paste(_c39, (456, 1364), _c39)
except Exception:
    pass
layout["29"] = [456, 1364, 588, 1484]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_11_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-13/40_text_30.png
try:
    _c40 = get_crop(40, 132, 120)
    canvas.paste(_c40, (588, 1364), _c40)
except Exception:
    pass
layout["30"] = [588, 1364, 720, 1484]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_11_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-13/41_text_31.png
try:
    _c41 = get_crop(41, 132, 120)
    canvas.paste(_c41, (720, 1364), _c41)
except Exception:
    pass
layout["31"] = [720, 1364, 852, 1484]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_11_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-13/42_clickable_1.png
try:
    _c42 = get_crop(42, 132, 120)
    canvas.paste(_c42, (456, 884), _c42)
except Exception:
    pass
layout["1"] = [456, 884, 588, 1004]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_11_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-13/43_clickable_2.png
try:
    _c43 = get_crop(43, 132, 120)
    canvas.paste(_c43, (588, 884), _c43)
except Exception:
    pass
layout["2"] = [588, 884, 720, 1004]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_11_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-13/44_clickable_3.png
try:
    _c44 = get_crop(44, 132, 120)
    canvas.paste(_c44, (720, 884), _c44)
except Exception:
    pass
layout["3"] = [720, 884, 852, 1004]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_11_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-13/45_clickable_4.png
try:
    _c45 = get_crop(45, 132, 120)
    canvas.paste(_c45, (852, 884), _c45)
except Exception:
    pass
layout["4"] = [852, 884, 984, 1004]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_11_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-13/46_clickable_5.png
try:
    _c46 = get_crop(46, 132, 120)
    canvas.paste(_c46, (60, 1004), _c46)
except Exception:
    pass
layout["5"] = [60, 1004, 192, 1124]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_11_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-13/47_clickable_6.png
try:
    _c47 = get_crop(47, 132, 120)
    canvas.paste(_c47, (192, 1004), _c47)
except Exception:
    pass
layout["6"] = [192, 1004, 324, 1124]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_11_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-13/48_clickable_7.png
try:
    _c48 = get_crop(48, 132, 120)
    canvas.paste(_c48, (324, 1004), _c48)
except Exception:
    pass
layout["7"] = [324, 1004, 456, 1124]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_11_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-13/49_clickable_8.png
try:
    _c49 = get_crop(49, 132, 120)
    canvas.paste(_c49, (456, 1004), _c49)
except Exception:
    pass
layout["8"] = [456, 1004, 588, 1124]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_11_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-13/50_clickable_9.png
try:
    _c50 = get_crop(50, 132, 120)
    canvas.paste(_c50, (588, 1004), _c50)
except Exception:
    pass
layout["9"] = [588, 1004, 720, 1124]
