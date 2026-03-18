# page_id: page_eventbrite_c8da01368c4649acb64c6c6b41699bc8_10
# screenshot: 2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-12.png
# step_index: 10/13
# task: Open Eventbrite. Look up "Animal" events. Filter by events happening next week. Select the first event - who is the organizer?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw UI background & structural elements for a 1440x2960 canvas.
# Available variables: canvas (PIL Image), draw (ImageDraw), font_sm, font_md, font_lg, font_xl

# Colors
bg_color = (255, 255, 255)            # page background (white)
status_bar_color = (200, 200, 200)    # top status bar (light gray)
header_divider = (230, 225, 235)      # subtle divider under header (very light purple/gray)
card_bg = (250, 249, 252)             # pale card background
card_border = (240, 235, 245)         # card border tint
calendar_card_bg = (255, 255, 255)    # calendar card (white)
calendar_border = (242, 238, 247)     # calendar border tint
separator = (235, 230, 240)           # light separator line

W, H = canvas.size

# Fill overall background
draw.rectangle([(0,0),(W,H)], fill=bg_color)

# Status bar area at the very top (for time/signal icons)
status_h = 72
draw.rectangle([(0,0),(W,status_h)], fill=status_bar_color)

# Thin top edge to separate status bar from rest (subtle)
draw.line([(0,status_h-1),(W,status_h-1)], fill=(190,190,190), width=1)

# Header / toolbar area (title and back arrow will be pasted on top)
header_y0 = status_h
header_y1 = 172
# keep header background same as page but add a faint divider below
draw.rectangle([(0,header_y0),(W,header_y1)], fill=bg_color)
draw.line([(24, header_y1),(W-24, header_y1)], fill=header_divider, width=1)

# Subtle left-side back-navigation safe area shadow (behind arrow icon that will be pasted)
# This is intentionally minimal and won't overlap icons' pixels (provides structure only)
left_nav_box = (20, header_y0+12, 120, header_y1-12)
draw.rounded_rectangle(left_nav_box, radius=10, fill=bg_color, outline=None)

# Top content group card (behind Start Date / End Date text blocks)
# Place as a light rounded card that groups the date fields
group_card_x0 = 28
group_card_x1 = W - 28
group_card_y0 = 220
group_card_y1 = 520
draw.rounded_rectangle([ (group_card_x0, group_card_y0), (group_card_x1, group_card_y1) ],
                       radius=16, fill=card_bg, outline=card_border, width=2)

# Divider between the two rows inside the group card (subtle)
mid_y = group_card_y0 + 170
draw.line([(group_card_x0+20, mid_y),(group_card_x1-20, mid_y)], fill=separator, width=1)

# Calendar container card (holds month header and calendar grid)
cal_x0 = 120
cal_x1 = W - 120
cal_y0 = 600
cal_y1 = 1420
draw.rounded_rectangle([ (cal_x0, cal_y0), (cal_x1, cal_y1) ],
                       radius=18, fill=calendar_card_bg, outline=calendar_border, width=2)

# Month header band inside the calendar card (background only; month label will be pasted)
month_band_h = 88
month_band_y0 = cal_y0 + 20
month_band_y1 = month_band_y0 + month_band_h
# Keep band same white but give subtle bottom divider line
draw.rectangle([(cal_x0+20, month_band_y0),(cal_x1-20, month_band_y1)], fill=calendar_card_bg)
draw.line([(cal_x0+36, month_band_y1),(cal_x1-36, month_band_y1)], fill=separator, width=1)

# Weekday row hint (thin separators)
weekday_y = month_band_y1 + 48
draw.line([(cal_x0+40, weekday_y),(cal_x1-40, weekday_y)], fill=separator, width=1)

# Vertical spacing guide lines for calendar columns (very faint, purely structural)
col_count = 7
col_width = (cal_x1 - cal_x0 - 80) / col_count
for i in range(1, col_count):
    x = cal_x0 + 40 + i * col_width
    draw.line([(x, month_band_y1+12),(x, cal_y1-24)], fill=(248,246,249), width=1)

# Horizontal separators for weeks (subtle)
week_rows = 6
row_height = (cal_y1 - (weekday_y + 20) - 48) / week_rows
for r in range(1, week_rows):
    y = weekday_y + 20 + r * row_height
    draw.line([(cal_x0+36, y),(cal_x1-36, y)], fill=(248,246,249), width=1)

# Large empty content area below calendar (leave mostly blank but draw a faint divider)
content_divider_y = cal_y1 + 24
draw.line([(28, content_divider_y),(W-28, content_divider_y)], fill=(245,243,246), width=1)

# Bottom area: draw a faint inset border for the screen bottom area (above the apply button)
# Make sure not to draw over the exact apply button area at y >= 2768 (we stop well above)
bottom_inset_y0 = H - 220
bottom_inset_y1 = H - 120
draw.rounded_rectangle([ (40, bottom_inset_y0), (W-40, bottom_inset_y1) ],
                       radius=12, outline=(235,230,240), width=2, fill=bg_color)

# Final subtle overall vignette lines to define major sections (purely structural)
# Top-mid separator under header group
draw.line([(28, group_card_y0-12),(W-28, group_card_y0-12)], fill=separator, width=1)
# Separator above calendar
draw.line([(28, cal_y0-12),(W-28, cal_y0-12)], fill=separator, width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_10_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-12/00_icon_Apply_date_range.png
try:
    _c0 = get_crop(0, 1344, 144)
    canvas.paste(_c0, (48, 2768), _c0)
except Exception:
    pass
layout["Apply_date_range"] = [48, 2768, 1392, 2912]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_10_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-12/01_icon_5.15.png
try:
    _c1 = get_crop(1, 59, 64)
    canvas.paste(_c1, (181, 1), _c1)
except Exception:
    pass
layout["5.15"] = [181, 1, 240, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_10_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-12/02_icon_icon_2.png
try:
    _c2 = get_crop(2, 99, 109)
    canvas.paste(_c2, (74, 775), _c2)
except Exception:
    pass
layout["icon_2"] = [74, 775, 173, 884]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_10_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-12/03_icon_May_2024.png
try:
    _c3 = get_crop(3, 120, 110)
    canvas.paste(_c3, (200, 773), _c3)
except Exception:
    pass
layout["May_2024"] = [200, 773, 320, 883]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_10_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-12/04_icon_icon_4.png
try:
    _c4 = get_crop(4, 61, 62)
    canvas.paste(_c4, (310, 3), _c4)
except Exception:
    pass
layout["icon_4"] = [310, 3, 371, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_10_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-12/05_icon_5.15.png
try:
    _c5 = get_crop(5, 58, 66)
    canvas.paste(_c5, (116, 1), _c5)
except Exception:
    pass
layout["5.15"] = [116, 1, 174, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_10_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-12/06_icon_5.15.png
try:
    _c6 = get_crop(6, 144, 144)
    canvas.paste(_c6, (12, 72), _c6)
except Exception:
    pass
layout["5.15"] = [12, 72, 156, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_10_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-12/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 57, 70)
    canvas.paste(_c7, (1316, 0), _c7)
except Exception:
    pass
layout["icon_7"] = [1316, 0, 1373, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_10_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-12/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 52, 64)
    canvas.paste(_c8, (247, 2), _c8)
except Exception:
    pass
layout["icon_8"] = [247, 2, 299, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_10_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-12/09_icon_May_2024.png
try:
    _c9 = get_crop(9, 143, 112)
    canvas.paste(_c9, (320, 771), _c9)
except Exception:
    pass
layout["May_2024"] = [320, 771, 463, 883]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_10_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-12/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 89, 69)
    canvas.paste(_c10, (1212, 0), _c10)
except Exception:
    pass
layout["icon_10"] = [1212, 0, 1301, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_10_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-12/11_icon_May_2024.png
try:
    _c11 = get_crop(11, 131, 113)
    canvas.paste(_c11, (456, 770), _c11)
except Exception:
    pass
layout["May_2024"] = [456, 770, 587, 883]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_10_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-12/12_icon_What_date.png
try:
    _c12 = get_crop(12, 319, 71)
    canvas.paste(_c12, (558, 112), _c12)
except Exception:
    pass
layout["What_date?"] = [558, 112, 877, 183]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_10_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-12/13_icon_End_Date.png
try:
    _c13 = get_crop(13, 144, 144)
    canvas.paste(_c13, (54, 620), _c13)
except Exception:
    pass
layout["End_Date"] = [54, 620, 198, 764]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_10_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-12/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 40, 65)
    canvas.paste(_c14, (1274, 0), _c14)
except Exception:
    pass
layout["icon_14"] = [1274, 0, 1314, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_10_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-12/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 49, 65)
    canvas.paste(_c15, (382, 2), _c15)
except Exception:
    pass
layout["icon_15"] = [382, 2, 431, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_10_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-12/16_icon_26.png
try:
    _c16 = get_crop(16, 132, 120)
    canvas.paste(_c16, (60, 1364), _c16)
except Exception:
    pass
layout["26"] = [60, 1364, 192, 1484]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_10_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-12/17_icon_27.png
try:
    _c17 = get_crop(17, 132, 120)
    canvas.paste(_c17, (192, 1364), _c17)
except Exception:
    pass
layout["27"] = [192, 1364, 324, 1484]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_10_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-12/18_icon_Next_month.png
try:
    _c18 = get_crop(18, 144, 144)
    canvas.paste(_c18, (846, 620), _c18)
except Exception:
    pass
layout["Next_month"] = [846, 620, 990, 764]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_10_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-12/19_text_5.15.png
try:
    _c19 = get_crop(19, 92, 43)
    canvas.paste(_c19, (22, 17), _c19)
except Exception:
    pass
layout["5.15"] = [22, 17, 114, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_10_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-12/20_text_Start_Date.png
try:
    _c20 = get_crop(20, 583, 144)
    canvas.paste(_c20, (48, 313), _c20)
except Exception:
    pass
layout["Start_Date"] = [48, 313, 631, 457]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_10_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-12/21_text_End_Date.png
try:
    _c21 = get_crop(21, 638, 114)
    canvas.paste(_c21, (48, 476), _c21)
except Exception:
    pass
layout["End_Date"] = [48, 476, 686, 590]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_10_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-12/22_text_May_2024.png
try:
    _c22 = get_crop(22, 198, 56)
    canvas.paste(_c22, (423, 666), _c22)
except Exception:
    pass
layout["May_2024"] = [423, 666, 621, 722]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_10_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-12/23_text_10.png
try:
    _c23 = get_crop(23, 132, 120)
    canvas.paste(_c23, (720, 1004), _c23)
except Exception:
    pass
layout["10"] = [720, 1004, 852, 1124]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_10_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-12/24_text_11.png
try:
    _c24 = get_crop(24, 132, 120)
    canvas.paste(_c24, (852, 1004), _c24)
except Exception:
    pass
layout["11"] = [852, 1004, 984, 1124]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_10_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-12/25_text_12.png
try:
    _c25 = get_crop(25, 132, 120)
    canvas.paste(_c25, (60, 1124), _c25)
except Exception:
    pass
layout["12"] = [60, 1124, 192, 1244]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_10_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-12/26_text_13.png
try:
    _c26 = get_crop(26, 132, 120)
    canvas.paste(_c26, (192, 1124), _c26)
except Exception:
    pass
layout["13"] = [192, 1124, 324, 1244]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_10_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-12/27_text_14.png
try:
    _c27 = get_crop(27, 132, 120)
    canvas.paste(_c27, (324, 1124), _c27)
except Exception:
    pass
layout["14"] = [324, 1124, 456, 1244]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_10_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-12/28_text_15.png
try:
    _c28 = get_crop(28, 132, 120)
    canvas.paste(_c28, (456, 1124), _c28)
except Exception:
    pass
layout["15"] = [456, 1124, 588, 1244]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_10_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-12/29_text_16.png
try:
    _c29 = get_crop(29, 132, 120)
    canvas.paste(_c29, (588, 1124), _c29)
except Exception:
    pass
layout["16"] = [588, 1124, 720, 1244]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_10_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-12/30_text_17.png
try:
    _c30 = get_crop(30, 132, 120)
    canvas.paste(_c30, (720, 1124), _c30)
except Exception:
    pass
layout["17"] = [720, 1124, 852, 1244]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_10_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-12/31_text_18.png
try:
    _c31 = get_crop(31, 132, 120)
    canvas.paste(_c31, (852, 1124), _c31)
except Exception:
    pass
layout["18"] = [852, 1124, 984, 1244]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_10_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-12/32_text_19.png
try:
    _c32 = get_crop(32, 132, 120)
    canvas.paste(_c32, (60, 1244), _c32)
except Exception:
    pass
layout["19"] = [60, 1244, 192, 1364]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_10_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-12/33_text_20.png
try:
    _c33 = get_crop(33, 132, 120)
    canvas.paste(_c33, (192, 1244), _c33)
except Exception:
    pass
layout["20"] = [192, 1244, 324, 1364]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_10_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-12/34_text_21.png
try:
    _c34 = get_crop(34, 132, 120)
    canvas.paste(_c34, (324, 1244), _c34)
except Exception:
    pass
layout["21"] = [324, 1244, 456, 1364]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_10_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-12/35_text_22.png
try:
    _c35 = get_crop(35, 132, 120)
    canvas.paste(_c35, (456, 1244), _c35)
except Exception:
    pass
layout["22"] = [456, 1244, 588, 1364]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_10_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-12/36_text_23.png
try:
    _c36 = get_crop(36, 132, 120)
    canvas.paste(_c36, (588, 1244), _c36)
except Exception:
    pass
layout["23"] = [588, 1244, 720, 1364]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_10_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-12/37_text_24.png
try:
    _c37 = get_crop(37, 132, 120)
    canvas.paste(_c37, (720, 1244), _c37)
except Exception:
    pass
layout["24"] = [720, 1244, 852, 1364]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_10_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-12/38_text_25.png
try:
    _c38 = get_crop(38, 132, 120)
    canvas.paste(_c38, (852, 1244), _c38)
except Exception:
    pass
layout["25"] = [852, 1244, 984, 1364]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_10_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-12/39_text_28.png
try:
    _c39 = get_crop(39, 132, 120)
    canvas.paste(_c39, (324, 1364), _c39)
except Exception:
    pass
layout["28"] = [324, 1364, 456, 1484]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_10_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-12/40_text_29.png
try:
    _c40 = get_crop(40, 132, 120)
    canvas.paste(_c40, (456, 1364), _c40)
except Exception:
    pass
layout["29"] = [456, 1364, 588, 1484]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_10_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-12/41_text_30.png
try:
    _c41 = get_crop(41, 132, 120)
    canvas.paste(_c41, (588, 1364), _c41)
except Exception:
    pass
layout["30"] = [588, 1364, 720, 1484]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_10_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-12/42_text_31.png
try:
    _c42 = get_crop(42, 132, 120)
    canvas.paste(_c42, (720, 1364), _c42)
except Exception:
    pass
layout["31"] = [720, 1364, 852, 1484]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_10_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-12/43_clickable_1.png
try:
    _c43 = get_crop(43, 132, 120)
    canvas.paste(_c43, (456, 884), _c43)
except Exception:
    pass
layout["1"] = [456, 884, 588, 1004]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_10_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-12/44_clickable_2.png
try:
    _c44 = get_crop(44, 132, 120)
    canvas.paste(_c44, (588, 884), _c44)
except Exception:
    pass
layout["2"] = [588, 884, 720, 1004]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_10_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-12/45_clickable_3.png
try:
    _c45 = get_crop(45, 132, 120)
    canvas.paste(_c45, (720, 884), _c45)
except Exception:
    pass
layout["3"] = [720, 884, 852, 1004]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_10_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-12/46_clickable_4.png
try:
    _c46 = get_crop(46, 132, 120)
    canvas.paste(_c46, (852, 884), _c46)
except Exception:
    pass
layout["4"] = [852, 884, 984, 1004]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_10_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-12/47_clickable_5.png
try:
    _c47 = get_crop(47, 132, 120)
    canvas.paste(_c47, (60, 1004), _c47)
except Exception:
    pass
layout["5"] = [60, 1004, 192, 1124]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_10_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-12/48_clickable_6.png
try:
    _c48 = get_crop(48, 132, 120)
    canvas.paste(_c48, (192, 1004), _c48)
except Exception:
    pass
layout["6"] = [192, 1004, 324, 1124]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_10_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-12/49_clickable_7.png
try:
    _c49 = get_crop(49, 132, 120)
    canvas.paste(_c49, (324, 1004), _c49)
except Exception:
    pass
layout["7"] = [324, 1004, 456, 1124]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_10_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-12/50_clickable_8.png
try:
    _c50 = get_crop(50, 132, 120)
    canvas.paste(_c50, (456, 1004), _c50)
except Exception:
    pass
layout["8"] = [456, 1004, 588, 1124]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_10_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-12/51_clickable_9.png
try:
    _c51 = get_crop(51, 132, 120)
    canvas.paste(_c51, (588, 1004), _c51)
except Exception:
    pass
layout["9"] = [588, 1004, 720, 1124]
