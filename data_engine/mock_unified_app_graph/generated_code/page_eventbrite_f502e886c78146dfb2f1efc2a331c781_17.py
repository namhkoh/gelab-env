# page_id: page_eventbrite_f502e886c78146dfb2f1efc2a331c781_17
# screenshot: 2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-19.png
# step_index: 17/18
# task: Open Eventbrite. Search for 'music festival' in San Francisco. Set date available from April 30 to May 3. How many events are listed?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Canvas assumed to exist: 1440x2960 (canvas), and draw is an ImageDraw object.
w, h = canvas.size

# Background (dominant color is white)
draw.rectangle([(0, 0), (w, h)], fill=(255, 255, 255))

# Status bar area at top (~72px) - light gray background
status_h = 72
draw.rectangle([(0, 0), (w, status_h)], fill=(200, 200, 200))

# Header / toolbar area below status bar
header_top = status_h
header_bottom = 176
draw.rectangle([(0, header_top), (w, header_bottom)], fill=(255, 255, 255))

# Header bottom divider (subtle)
divider_y = header_bottom
draw.line([(24, divider_y), (w - 24, divider_y)], fill=(234, 230, 246), width=1)

# Group background for Start Date and End Date area (two subtle rounded cards)
# These are background shapes only; text/icons are auto-pasted on top.
start_card = (36, 240, w - 36, 420)
end_card = (36, 420, w - 36, 560)
draw.rounded_rectangle(start_card, radius=14, fill=(249, 247, 252), outline=None)
draw.rounded_rectangle(end_card, radius=14, fill=(249, 247, 252), outline=None)

# Light divider between the two cards to visually separate sections
draw.line([(48, 420), (w - 48, 420)], fill=(238, 235, 245), width=1)

# Calendar background area (subtle white-on-white container with faint shadow line)
cal_top = 600
cal_bottom = 1600
cal_left = 48
cal_right = w - 48
# faint background tint to anchor the calendar area
draw.rounded_rectangle((cal_left, cal_top, cal_right, cal_bottom), radius=12, fill=(255, 255, 255), outline=None)
# very faint top border/shadow
draw.line([(cal_left + 8, cal_top + 1), (cal_right - 8, cal_top + 1)], fill=(245, 243, 250), width=1)

# Calendar grid separators (subtle)
# Based on detected grid cell size ~132x120 and starting x around 60, starting y around 884
cell_w = 132
cell_h = 120
grid_start_x = 60
grid_start_y = 884
cols = 7
rows = 6

sep_color = (231, 227, 239)  # very light purple-gray for separators

# Vertical grid lines
for c in range(cols + 1):
    x = grid_start_x + c * cell_w
    # limit grid lines to reasonable vertical span within calendar area
    y1 = grid_start_y
    y2 = grid_start_y + rows * cell_h
    draw.line([(x, y1), (x, y2)], fill=sep_color, width=1)

# Horizontal grid lines
for r in range(rows + 1):
    y = grid_start_y + r * cell_h
    x1 = grid_start_x
    x2 = grid_start_x + cols * cell_w
    draw.line([(x1, y), (x2, y)], fill=sep_color, width=1)

# Weekday header separator (just above the first row of dates)
weekday_sep_y = grid_start_y - 60
draw.line([(grid_start_x, weekday_sep_y), (grid_start_x + cols * cell_w, weekday_sep_y)], fill=(245, 243, 250), width=1)

# Subtle vertical centering guides (very faint) to suggest column alignment (background-only)
guide_color = (248, 247, 250)
for c in range(cols):
    cx = grid_start_x + c * cell_w + cell_w // 2
    draw.line([(cx, grid_start_y - 80), (cx, grid_start_y + rows * cell_h + 20)], fill=guide_color, width=1)

# Top-left large decorative accent behind title area (very subtle, won't overlap text/icons)
accent_rect = (24, header_top + 8, 220, header_top + 56)
draw.rounded_rectangle(accent_rect, radius=8, fill=(250, 249, 253))

# Separator line above the bottom action area (the "Apply date range" button will be pasted on top;
# avoid drawing inside the button area itself). Button top is at y=2768, so draw a separator just above it.
bottom_sep_y = 2728
draw.line([(24, bottom_sep_y), (w - 24, bottom_sep_y)], fill=(230, 227, 236), width=2)
# a faint shadow band between separator and button region (keeps it subtle and not duplicating the button)
draw.rectangle([(0, bottom_sep_y + 2), (w, bottom_sep_y + 12)], fill=(250, 249, 252))

# Final subtle edge frame around the whole screen (very faint)
draw.rectangle([(2, 2), (w - 2, h - 2)], outline=(245, 244, 247), width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_17_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-19/00_icon_Apply_date_range.png
try:
    _c0 = get_crop(0, 1344, 144)
    canvas.paste(_c0, (48, 2768), _c0)
except Exception:
    pass
layout["Apply_date_range"] = [48, 2768, 1392, 2912]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_17_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-19/01_icon_icon_1.png
try:
    _c1 = get_crop(1, 51, 71)
    canvas.paste(_c1, (1154, 0), _c1)
except Exception:
    pass
layout["icon_1"] = [1154, 0, 1205, 71]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_17_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-19/02_icon_icon_2.png
try:
    _c2 = get_crop(2, 101, 70)
    canvas.paste(_c2, (1210, 0), _c2)
except Exception:
    pass
layout["icon_2"] = [1210, 0, 1311, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_17_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-19/03_icon_7.19.png
try:
    _c3 = get_crop(3, 59, 63)
    canvas.paste(_c3, (181, 1), _c3)
except Exception:
    pass
layout["7.19"] = [181, 1, 240, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_17_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-19/04_icon_icon_4.png
try:
    _c4 = get_crop(4, 52, 70)
    canvas.paste(_c4, (1319, 0), _c4)
except Exception:
    pass
layout["icon_4"] = [1319, 0, 1371, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_17_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-19/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 60, 62)
    canvas.paste(_c5, (311, 3), _c5)
except Exception:
    pass
layout["icon_5"] = [311, 3, 371, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_17_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-19/06_icon_7.19.png
try:
    _c6 = get_crop(6, 58, 64)
    canvas.paste(_c6, (116, 1), _c6)
except Exception:
    pass
layout["7.19"] = [116, 1, 174, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_17_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-19/07_icon_7.19.png
try:
    _c7 = get_crop(7, 144, 144)
    canvas.paste(_c7, (12, 72), _c7)
except Exception:
    pass
layout["7.19"] = [12, 72, 156, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_17_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-19/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 51, 61)
    canvas.paste(_c8, (249, 3), _c8)
except Exception:
    pass
layout["icon_8"] = [249, 3, 300, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_17_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-19/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 96, 107)
    canvas.paste(_c9, (75, 777), _c9)
except Exception:
    pass
layout["icon_9"] = [75, 777, 171, 884]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_17_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-19/10_icon_What_date.png
try:
    _c10 = get_crop(10, 319, 73)
    canvas.paste(_c10, (558, 111), _c10)
except Exception:
    pass
layout["What_date?"] = [558, 111, 877, 184]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_17_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-19/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 108, 108)
    canvas.paste(_c11, (202, 775), _c11)
except Exception:
    pass
layout["icon_11"] = [202, 775, 310, 883]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_17_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-19/12_icon_27.png
try:
    _c12 = get_crop(12, 132, 120)
    canvas.paste(_c12, (192, 1364), _c12)
except Exception:
    pass
layout["27"] = [192, 1364, 324, 1484]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_17_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-19/13_icon_End_Date.png
try:
    _c13 = get_crop(13, 144, 144)
    canvas.paste(_c13, (54, 620), _c13)
except Exception:
    pass
layout["End_Date"] = [54, 620, 198, 764]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_17_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-19/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 49, 64)
    canvas.paste(_c14, (382, 2), _c14)
except Exception:
    pass
layout["icon_14"] = [382, 2, 431, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_17_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-19/15_icon_Next_month.png
try:
    _c15 = get_crop(15, 144, 144)
    canvas.paste(_c15, (846, 620), _c15)
except Exception:
    pass
layout["Next_month"] = [846, 620, 990, 764]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_17_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-19/16_icon_May_2024.png
try:
    _c16 = get_crop(16, 107, 110)
    canvas.paste(_c16, (463, 774), _c16)
except Exception:
    pass
layout["May_2024"] = [463, 774, 570, 884]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_17_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-19/17_icon_May_2024.png
try:
    _c17 = get_crop(17, 116, 109)
    canvas.paste(_c17, (331, 775), _c17)
except Exception:
    pass
layout["May_2024"] = [331, 775, 447, 884]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_17_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-19/18_text_7.19.png
try:
    _c18 = get_crop(18, 91, 45)
    canvas.paste(_c18, (20, 15), _c18)
except Exception:
    pass
layout["7.19"] = [20, 15, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_17_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-19/19_text_Start_Date.png
try:
    _c19 = get_crop(19, 587, 144)
    canvas.paste(_c19, (48, 313), _c19)
except Exception:
    pass
layout["Start_Date"] = [48, 313, 635, 457]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_17_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-19/20_text_End_Date.png
try:
    _c20 = get_crop(20, 613, 114)
    canvas.paste(_c20, (48, 476), _c20)
except Exception:
    pass
layout["End_Date"] = [48, 476, 661, 590]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_17_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-19/21_text_May_2024.png
try:
    _c21 = get_crop(21, 198, 56)
    canvas.paste(_c21, (423, 666), _c21)
except Exception:
    pass
layout["May_2024"] = [423, 666, 621, 722]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_17_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-19/22_text_10.png
try:
    _c22 = get_crop(22, 132, 120)
    canvas.paste(_c22, (720, 1004), _c22)
except Exception:
    pass
layout["10"] = [720, 1004, 852, 1124]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_17_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-19/23_text_11.png
try:
    _c23 = get_crop(23, 132, 120)
    canvas.paste(_c23, (852, 1004), _c23)
except Exception:
    pass
layout["11"] = [852, 1004, 984, 1124]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_17_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-19/24_text_12.png
try:
    _c24 = get_crop(24, 132, 120)
    canvas.paste(_c24, (60, 1124), _c24)
except Exception:
    pass
layout["12"] = [60, 1124, 192, 1244]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_17_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-19/25_text_13.png
try:
    _c25 = get_crop(25, 132, 120)
    canvas.paste(_c25, (192, 1124), _c25)
except Exception:
    pass
layout["13"] = [192, 1124, 324, 1244]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_17_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-19/26_text_14.png
try:
    _c26 = get_crop(26, 132, 120)
    canvas.paste(_c26, (324, 1124), _c26)
except Exception:
    pass
layout["14"] = [324, 1124, 456, 1244]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_17_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-19/27_text_15.png
try:
    _c27 = get_crop(27, 132, 120)
    canvas.paste(_c27, (456, 1124), _c27)
except Exception:
    pass
layout["15"] = [456, 1124, 588, 1244]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_17_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-19/28_text_16.png
try:
    _c28 = get_crop(28, 132, 120)
    canvas.paste(_c28, (588, 1124), _c28)
except Exception:
    pass
layout["16"] = [588, 1124, 720, 1244]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_17_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-19/29_text_17.png
try:
    _c29 = get_crop(29, 132, 120)
    canvas.paste(_c29, (720, 1124), _c29)
except Exception:
    pass
layout["17"] = [720, 1124, 852, 1244]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_17_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-19/30_text_18.png
try:
    _c30 = get_crop(30, 132, 120)
    canvas.paste(_c30, (852, 1124), _c30)
except Exception:
    pass
layout["18"] = [852, 1124, 984, 1244]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_17_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-19/31_text_19.png
try:
    _c31 = get_crop(31, 132, 120)
    canvas.paste(_c31, (60, 1244), _c31)
except Exception:
    pass
layout["19"] = [60, 1244, 192, 1364]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_17_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-19/32_text_20.png
try:
    _c32 = get_crop(32, 132, 120)
    canvas.paste(_c32, (192, 1244), _c32)
except Exception:
    pass
layout["20"] = [192, 1244, 324, 1364]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_17_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-19/33_text_21.png
try:
    _c33 = get_crop(33, 132, 120)
    canvas.paste(_c33, (324, 1244), _c33)
except Exception:
    pass
layout["21"] = [324, 1244, 456, 1364]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_17_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-19/34_text_22.png
try:
    _c34 = get_crop(34, 132, 120)
    canvas.paste(_c34, (456, 1244), _c34)
except Exception:
    pass
layout["22"] = [456, 1244, 588, 1364]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_17_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-19/35_text_23.png
try:
    _c35 = get_crop(35, 132, 120)
    canvas.paste(_c35, (588, 1244), _c35)
except Exception:
    pass
layout["23"] = [588, 1244, 720, 1364]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_17_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-19/36_text_24.png
try:
    _c36 = get_crop(36, 132, 120)
    canvas.paste(_c36, (720, 1244), _c36)
except Exception:
    pass
layout["24"] = [720, 1244, 852, 1364]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_17_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-19/37_text_25.png
try:
    _c37 = get_crop(37, 132, 120)
    canvas.paste(_c37, (852, 1244), _c37)
except Exception:
    pass
layout["25"] = [852, 1244, 984, 1364]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_17_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-19/38_text_26.png
try:
    _c38 = get_crop(38, 132, 120)
    canvas.paste(_c38, (60, 1364), _c38)
except Exception:
    pass
layout["26"] = [60, 1364, 192, 1484]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_17_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-19/39_text_28.png
try:
    _c39 = get_crop(39, 132, 120)
    canvas.paste(_c39, (324, 1364), _c39)
except Exception:
    pass
layout["28"] = [324, 1364, 456, 1484]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_17_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-19/40_text_29.png
try:
    _c40 = get_crop(40, 132, 120)
    canvas.paste(_c40, (456, 1364), _c40)
except Exception:
    pass
layout["29"] = [456, 1364, 588, 1484]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_17_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-19/41_text_30.png
try:
    _c41 = get_crop(41, 132, 120)
    canvas.paste(_c41, (588, 1364), _c41)
except Exception:
    pass
layout["30"] = [588, 1364, 720, 1484]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_17_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-19/42_text_31.png
try:
    _c42 = get_crop(42, 132, 120)
    canvas.paste(_c42, (720, 1364), _c42)
except Exception:
    pass
layout["31"] = [720, 1364, 852, 1484]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_17_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-19/43_clickable_1.png
try:
    _c43 = get_crop(43, 132, 120)
    canvas.paste(_c43, (456, 884), _c43)
except Exception:
    pass
layout["1"] = [456, 884, 588, 1004]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_17_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-19/44_clickable_2.png
try:
    _c44 = get_crop(44, 132, 120)
    canvas.paste(_c44, (588, 884), _c44)
except Exception:
    pass
layout["2"] = [588, 884, 720, 1004]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_17_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-19/45_clickable_3.png
try:
    _c45 = get_crop(45, 132, 120)
    canvas.paste(_c45, (720, 884), _c45)
except Exception:
    pass
layout["3"] = [720, 884, 852, 1004]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_17_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-19/46_clickable_4.png
try:
    _c46 = get_crop(46, 132, 120)
    canvas.paste(_c46, (852, 884), _c46)
except Exception:
    pass
layout["4"] = [852, 884, 984, 1004]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_17_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-19/47_clickable_5.png
try:
    _c47 = get_crop(47, 132, 120)
    canvas.paste(_c47, (60, 1004), _c47)
except Exception:
    pass
layout["5"] = [60, 1004, 192, 1124]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_17_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-19/48_clickable_6.png
try:
    _c48 = get_crop(48, 132, 120)
    canvas.paste(_c48, (192, 1004), _c48)
except Exception:
    pass
layout["6"] = [192, 1004, 324, 1124]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_17_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-19/49_clickable_7.png
try:
    _c49 = get_crop(49, 132, 120)
    canvas.paste(_c49, (324, 1004), _c49)
except Exception:
    pass
layout["7"] = [324, 1004, 456, 1124]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_17_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-19/50_clickable_8.png
try:
    _c50 = get_crop(50, 132, 120)
    canvas.paste(_c50, (456, 1004), _c50)
except Exception:
    pass
layout["8"] = [456, 1004, 588, 1124]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_17_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-19/51_clickable_9.png
try:
    _c51 = get_crop(51, 132, 120)
    canvas.paste(_c51, (588, 1004), _c51)
except Exception:
    pass
layout["9"] = [588, 1004, 720, 1124]
