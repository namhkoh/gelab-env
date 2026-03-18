# page_id: page_eventbrite_c8da01368c4649acb64c6c6b41699bc8_11
# screenshot: 2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-13.png
# step_index: 11/13
# task: Open Eventbrite. Look up "Animal" events. Filter by events happening next week. Select the first event - who is the organizer?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# draw background and UI structure for the calendar/date-range screen
# Available variables:
# - canvas: PIL Image (1440x2960 RGB)
# - draw: PIL ImageDraw object
# - font_sm, font_md, font_lg, font_xl

W, H = canvas.size

# Background fill (dominant color: near white)
draw.rectangle(((0, 0), (W, H)), fill=(255, 255, 255))

# Status bar area at top (~72px)
status_h = 72
draw.rectangle(((0, 0), (W, status_h)), fill=(210, 210, 210))  # light grey status bar

# Header / toolbar region (below status bar)
header_top = status_h
header_h = 128
header_bottom = header_top + header_h
# keep header white, but draw a subtle bottom divider and faint shadow
draw.rectangle(((0, header_top), (W, header_bottom)), fill=(255, 255, 255))
# subtle divider line
divider_y = header_bottom - 1
draw.line(((24, divider_y), (W - 24, divider_y)), fill=(232, 231, 235), width=1)
# faint shadow under header
shadow_y = header_bottom + 1
draw.line(((24, shadow_y), (W - 24, shadow_y)), fill=(245, 244, 246), width=1)

# Section separators for the Start/End date area (visual structure only)
# these separators sit between large date labels but do not duplicate text/icons
sep_x0 = 32
sep_x1 = W - 32
# first separator (below Start Date block)
sep1_y = 420
draw.line(((sep_x0, sep1_y), (sep_x1, sep1_y)), fill=(243, 242, 245), width=1)
# second separator (below End Date block)
sep2_y = 540
draw.line(((sep_x0, sep2_y), (sep_x1, sep2_y)), fill=(243, 242, 245), width=1)

# Calendar container background (rounded rectangle behind the month & grid)
cal_left = 40
cal_right = W - 40
cal_top = 620
cal_bottom = 1480
cal_radius = 16
# very subtle off-white / very light lavender to set the calendar area apart
draw.rounded_rectangle(
    ((cal_left, cal_top), (cal_right, cal_bottom)),
    radius=cal_radius,
    fill=(255, 255, 255),
    outline=None,
)

# Add a slightly darker horizontal rule where the month/nav would be (visual cue)
month_rule_y = cal_top + 48
draw.line(((cal_left + 24, month_rule_y), (cal_right - 24, month_rule_y)), fill=(250, 249, 251), width=1)

# Grid area suggestion: subtle faint grid lines (row separators) - very light so they don't duplicate numbers
grid_start_y = month_rule_y + 24
row_height = 120
for i in range(1, 6):
    y = grid_start_y + i * row_height
    if y < cal_bottom - 24:
        draw.line(((cal_left + 24, y), (cal_right - 24, y)), fill=(250, 249, 251), width=1)

# Column vertical hints (do not draw numbers or day letters)
col_count = 7
col_margin = 24
grid_width = cal_right - cal_left - 2 * col_margin
col_w = grid_width / col_count
for i in range(1, col_count):
    x = cal_left + col_margin + int(i * col_w)
    # very faint vertical guide lines
    draw.line(((x, grid_start_y), (x, cal_bottom - 24)), fill=(250, 249, 251), width=1)

# Bottom area: subtle divider above the action button region (do not draw the button itself)
button_area_top = 2768
top_rule_y = max(button_area_top - 24, H - 220)
draw.line(((24, top_rule_y), (W - 24, top_rule_y)), fill=(228, 226, 233), width=1)
# faint shadow below the rule
draw.line(((24, top_rule_y + 1), (W - 24, top_rule_y + 1)), fill=(247, 247, 247), width=1)

# Safe-area bottom padding background (very subtle)
safe_bottom_h = H - (button_area_top + 144)
if safe_bottom_h > 0:
    # fill a tiny bit of area under the button with very light color to match screenshot spacing
    draw.rectangle(((0, H - safe_bottom_h), (W, H)), fill=(255, 255, 255))

# subtle left/right page margins (vertical thin lines near edges for composition)
edge_line_y0 = header_bottom + 8
edge_line_y1 = H - 180
draw.line(((24, edge_line_y0), (24, edge_line_y1)), fill=(255, 255, 255), width=1)
draw.line(((W - 24, edge_line_y0), (W - 24, edge_line_y1)), fill=(255, 255, 255), width=1)

# finished structural drawing

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_11_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-13/00_icon_Apply_date_range.png
try:
    _c0 = get_crop(0, 1344, 144)
    canvas.paste(_c0, (48, 2768), _c0)
except Exception:
    pass
layout["Apply_date_range"] = [48, 2768, 1392, 2912]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_11_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-13/01_icon_5.15.png
try:
    _c1 = get_crop(1, 59, 65)
    canvas.paste(_c1, (181, 0), _c1)
except Exception:
    pass
layout["5.15"] = [181, 0, 240, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_11_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-13/02_icon_icon_2.png
try:
    _c2 = get_crop(2, 62, 62)
    canvas.paste(_c2, (310, 3), _c2)
except Exception:
    pass
layout["icon_2"] = [310, 3, 372, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_11_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-13/03_icon_icon_3.png
try:
    _c3 = get_crop(3, 57, 70)
    canvas.paste(_c3, (1316, 0), _c3)
except Exception:
    pass
layout["icon_3"] = [1316, 0, 1373, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_11_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-13/04_icon_5.15.png
try:
    _c4 = get_crop(4, 144, 144)
    canvas.paste(_c4, (12, 72), _c4)
except Exception:
    pass
layout["5.15"] = [12, 72, 156, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_11_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-13/05_icon_5.15.png
try:
    _c5 = get_crop(5, 59, 66)
    canvas.paste(_c5, (116, 0), _c5)
except Exception:
    pass
layout["5.15"] = [116, 0, 175, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_11_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-13/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 51, 63)
    canvas.paste(_c6, (249, 2), _c6)
except Exception:
    pass
layout["icon_6"] = [249, 2, 300, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_11_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-13/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 92, 70)
    canvas.paste(_c7, (1212, 0), _c7)
except Exception:
    pass
layout["icon_7"] = [1212, 0, 1304, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_11_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-13/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 97, 107)
    canvas.paste(_c8, (74, 777), _c8)
except Exception:
    pass
layout["icon_8"] = [74, 777, 171, 884]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_11_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-13/09_icon_What_date.png
try:
    _c9 = get_crop(9, 319, 72)
    canvas.paste(_c9, (558, 111), _c9)
except Exception:
    pass
layout["What_date?"] = [558, 111, 877, 183]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_11_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-13/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 110, 108)
    canvas.paste(_c10, (201, 775), _c10)
except Exception:
    pass
layout["icon_10"] = [201, 775, 311, 883]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_11_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-13/11_icon_End_Date.png
try:
    _c11 = get_crop(11, 144, 144)
    canvas.paste(_c11, (54, 620), _c11)
except Exception:
    pass
layout["End_Date"] = [54, 620, 198, 764]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_11_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-13/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 41, 66)
    canvas.paste(_c12, (1274, 0), _c12)
except Exception:
    pass
layout["icon_12"] = [1274, 0, 1315, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_11_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-13/13_icon_Next_month.png
try:
    _c13 = get_crop(13, 144, 144)
    canvas.paste(_c13, (846, 620), _c13)
except Exception:
    pass
layout["Next_month"] = [846, 620, 990, 764]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_11_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-13/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 49, 66)
    canvas.paste(_c14, (382, 1), _c14)
except Exception:
    pass
layout["icon_14"] = [382, 1, 431, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_11_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-13/15_text_5.15.png
try:
    _c15 = get_crop(15, 92, 43)
    canvas.paste(_c15, (22, 17), _c15)
except Exception:
    pass
layout["5.15"] = [22, 17, 114, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_11_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-13/16_text_Start_Date.png
try:
    _c16 = get_crop(16, 583, 144)
    canvas.paste(_c16, (48, 313), _c16)
except Exception:
    pass
layout["Start_Date"] = [48, 313, 631, 457]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_11_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-13/17_text_End_Date.png
try:
    _c17 = get_crop(17, 620, 114)
    canvas.paste(_c17, (48, 476), _c17)
except Exception:
    pass
layout["End_Date"] = [48, 476, 668, 590]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_11_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-13/18_text_May_2024.png
try:
    _c18 = get_crop(18, 198, 56)
    canvas.paste(_c18, (423, 666), _c18)
except Exception:
    pass
layout["May_2024"] = [423, 666, 621, 722]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_11_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-13/19_text_10.png
try:
    _c19 = get_crop(19, 132, 120)
    canvas.paste(_c19, (720, 1004), _c19)
except Exception:
    pass
layout["10"] = [720, 1004, 852, 1124]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_11_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-13/20_text_11.png
try:
    _c20 = get_crop(20, 132, 120)
    canvas.paste(_c20, (852, 1004), _c20)
except Exception:
    pass
layout["11"] = [852, 1004, 984, 1124]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_11_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-13/21_text_12.png
try:
    _c21 = get_crop(21, 132, 120)
    canvas.paste(_c21, (60, 1124), _c21)
except Exception:
    pass
layout["12"] = [60, 1124, 192, 1244]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_11_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-13/22_text_13.png
try:
    _c22 = get_crop(22, 132, 120)
    canvas.paste(_c22, (192, 1124), _c22)
except Exception:
    pass
layout["13"] = [192, 1124, 324, 1244]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_11_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-13/23_text_14.png
try:
    _c23 = get_crop(23, 132, 120)
    canvas.paste(_c23, (324, 1124), _c23)
except Exception:
    pass
layout["14"] = [324, 1124, 456, 1244]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_11_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-13/24_text_15.png
try:
    _c24 = get_crop(24, 132, 120)
    canvas.paste(_c24, (456, 1124), _c24)
except Exception:
    pass
layout["15"] = [456, 1124, 588, 1244]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_11_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-13/25_text_16.png
try:
    _c25 = get_crop(25, 132, 120)
    canvas.paste(_c25, (588, 1124), _c25)
except Exception:
    pass
layout["16"] = [588, 1124, 720, 1244]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_11_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-13/26_text_17.png
try:
    _c26 = get_crop(26, 132, 120)
    canvas.paste(_c26, (720, 1124), _c26)
except Exception:
    pass
layout["17"] = [720, 1124, 852, 1244]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_11_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-13/27_text_18.png
try:
    _c27 = get_crop(27, 132, 120)
    canvas.paste(_c27, (852, 1124), _c27)
except Exception:
    pass
layout["18"] = [852, 1124, 984, 1244]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_11_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-13/28_text_19.png
try:
    _c28 = get_crop(28, 132, 120)
    canvas.paste(_c28, (60, 1244), _c28)
except Exception:
    pass
layout["19"] = [60, 1244, 192, 1364]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_11_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-13/29_text_20.png
try:
    _c29 = get_crop(29, 132, 120)
    canvas.paste(_c29, (192, 1244), _c29)
except Exception:
    pass
layout["20"] = [192, 1244, 324, 1364]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_11_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-13/30_text_21.png
try:
    _c30 = get_crop(30, 132, 120)
    canvas.paste(_c30, (324, 1244), _c30)
except Exception:
    pass
layout["21"] = [324, 1244, 456, 1364]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_11_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-13/31_text_22.png
try:
    _c31 = get_crop(31, 132, 120)
    canvas.paste(_c31, (456, 1244), _c31)
except Exception:
    pass
layout["22"] = [456, 1244, 588, 1364]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_11_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-13/32_text_23.png
try:
    _c32 = get_crop(32, 132, 120)
    canvas.paste(_c32, (588, 1244), _c32)
except Exception:
    pass
layout["23"] = [588, 1244, 720, 1364]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_11_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-13/33_text_24.png
try:
    _c33 = get_crop(33, 132, 120)
    canvas.paste(_c33, (720, 1244), _c33)
except Exception:
    pass
layout["24"] = [720, 1244, 852, 1364]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_11_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-13/34_text_25.png
try:
    _c34 = get_crop(34, 132, 120)
    canvas.paste(_c34, (852, 1244), _c34)
except Exception:
    pass
layout["25"] = [852, 1244, 984, 1364]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_11_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-13/35_text_26.png
try:
    _c35 = get_crop(35, 132, 120)
    canvas.paste(_c35, (60, 1364), _c35)
except Exception:
    pass
layout["26"] = [60, 1364, 192, 1484]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_11_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-13/36_text_27.png
try:
    _c36 = get_crop(36, 132, 120)
    canvas.paste(_c36, (192, 1364), _c36)
except Exception:
    pass
layout["27"] = [192, 1364, 324, 1484]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_11_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-13/37_text_28.png
try:
    _c37 = get_crop(37, 132, 120)
    canvas.paste(_c37, (324, 1364), _c37)
except Exception:
    pass
layout["28"] = [324, 1364, 456, 1484]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_11_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-13/38_text_29.png
try:
    _c38 = get_crop(38, 132, 120)
    canvas.paste(_c38, (456, 1364), _c38)
except Exception:
    pass
layout["29"] = [456, 1364, 588, 1484]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_11_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-13/39_text_30.png
try:
    _c39 = get_crop(39, 132, 120)
    canvas.paste(_c39, (588, 1364), _c39)
except Exception:
    pass
layout["30"] = [588, 1364, 720, 1484]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_11_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-13/40_text_31.png
try:
    _c40 = get_crop(40, 132, 120)
    canvas.paste(_c40, (720, 1364), _c40)
except Exception:
    pass
layout["31"] = [720, 1364, 852, 1484]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_11_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-13/41_clickable_1.png
try:
    _c41 = get_crop(41, 132, 120)
    canvas.paste(_c41, (456, 884), _c41)
except Exception:
    pass
layout["1"] = [456, 884, 588, 1004]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_11_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-13/42_clickable_2.png
try:
    _c42 = get_crop(42, 132, 120)
    canvas.paste(_c42, (588, 884), _c42)
except Exception:
    pass
layout["2"] = [588, 884, 720, 1004]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_11_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-13/43_clickable_3.png
try:
    _c43 = get_crop(43, 132, 120)
    canvas.paste(_c43, (720, 884), _c43)
except Exception:
    pass
layout["3"] = [720, 884, 852, 1004]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_11_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-13/44_clickable_4.png
try:
    _c44 = get_crop(44, 132, 120)
    canvas.paste(_c44, (852, 884), _c44)
except Exception:
    pass
layout["4"] = [852, 884, 984, 1004]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_11_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-13/45_clickable_5.png
try:
    _c45 = get_crop(45, 132, 120)
    canvas.paste(_c45, (60, 1004), _c45)
except Exception:
    pass
layout["5"] = [60, 1004, 192, 1124]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_11_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-13/46_clickable_6.png
try:
    _c46 = get_crop(46, 132, 120)
    canvas.paste(_c46, (192, 1004), _c46)
except Exception:
    pass
layout["6"] = [192, 1004, 324, 1124]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_11_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-13/47_clickable_7.png
try:
    _c47 = get_crop(47, 132, 120)
    canvas.paste(_c47, (324, 1004), _c47)
except Exception:
    pass
layout["7"] = [324, 1004, 456, 1124]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_11_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-13/48_clickable_8.png
try:
    _c48 = get_crop(48, 132, 120)
    canvas.paste(_c48, (456, 1004), _c48)
except Exception:
    pass
layout["8"] = [456, 1004, 588, 1124]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_11_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-13/49_clickable_9.png
try:
    _c49 = get_crop(49, 132, 120)
    canvas.paste(_c49, (588, 1004), _c49)
except Exception:
    pass
layout["9"] = [588, 1004, 720, 1124]
