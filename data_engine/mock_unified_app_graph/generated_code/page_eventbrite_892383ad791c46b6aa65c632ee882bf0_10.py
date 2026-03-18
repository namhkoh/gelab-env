# page_id: page_eventbrite_892383ad791c46b6aa65c632ee882bf0_10
# screenshot: 2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-12.png
# step_index: 10/12
# task: Open Eventbrite. Search for online "Music" events happening next weekend.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw UI background and structural elements for a 1440x2960 canvas.
# Available variables: canvas (PIL Image), draw (ImageDraw), font_sm, font_md, font_lg, font_xl

W, H = canvas.size

# Color palette (approximate to screenshot)
WHITE = (255, 255, 255)
STATUS_GRAY = (217, 217, 217)      # top status bar
DIVIDER = (225, 219, 233)          # subtle divider / border (light purple-gray)
CARD_BG = (250, 249, 251)          # very light card background
SUBTLE_SHADOW = (240, 236, 245)    # faint shadow tone
ACCENT_DEEP = (45, 18, 59)         # deep purple used for thin accent lines

# 1) Status bar area at top (~50px high on screenshot; use slightly taller to match layout)
status_h = 80
draw.rectangle([0, 0, W, status_h], fill=STATUS_GRAY)

# 2) Header / toolbar area (below status bar). Keep background white; add a subtle bottom divider.
header_top = status_h
header_h = 120
header_bottom = header_top + header_h
draw.rectangle([0, header_top, W, header_bottom], fill=WHITE)
# bottom divider line
draw.line([40, header_bottom, W-40, header_bottom], fill=DIVIDER, width=2)

# 3) Subtle left alignment guideline area (visual structural hint)
# A faint vertical guide (not text or icon) to mirror content margins (very subtle)
draw.line([48, header_bottom+12, 48, H-220], fill=SUBTLE_SHADOW, width=1)

# 4) Large rounded card background for the calendar/content group
# Positioned under the header; rounded rect with soft border and pale fill
cal_x0 = 40
cal_x1 = W - 40
cal_y0 = header_bottom + 40   # start a bit below the header
cal_y1 = cal_y0 + 1240        # tall area to contain calendar grid
card_radius = 20
draw.rounded_rectangle([cal_x0, cal_y0, cal_x1, cal_y1],
                       radius=card_radius,
                       fill=CARD_BG,
                       outline=DIVIDER,
                       width=2)

# 5) Inner subtle separators inside the calendar card to indicate logical rows/areas
# (thin horizontal lines that don't replicate text/icons)
row_y = cal_y0 + 120
for i in range(4):
    y = row_y + i * 220
    if y < cal_y1 - 24:
        draw.line([cal_x0 + 28, y, cal_x1 - 28, y], fill=SUBTLE_SHADOW, width=1)

# 6) Weekdays header separator (thin line to suggest the weekday row boundary)
weekday_line_y = cal_y0 + 68
draw.line([cal_x0 + 60, weekday_line_y, cal_x1 - 60, weekday_line_y], fill=DIVIDER, width=1)

# 7) A faint grid hint for the calendar area (only light dividing lines, not numbers)
grid_left = cal_x0 + 40
grid_right = cal_x1 - 40
grid_top = weekday_line_y + 20
grid_bottom = cal_y1 - 40
cols = 7
rows = 6
cell_w = (grid_right - grid_left) / cols
cell_h = (grid_bottom - grid_top) / rows

for c in range(1, cols):
    x = int(grid_left + c * cell_w)
    draw.line([x, grid_top, x, grid_bottom], fill=(245,244,247), width=1)
for r in range(1, rows):
    y = int(grid_top + r * cell_h)
    draw.line([grid_left, y, grid_right, y], fill=(245,244,247), width=1)

# 8) Subtle decorative left chevron/hint area (background only) where back icon sits - do NOT draw icon
# Provide a faint rounded touch-area background at left edge of header (keeps layout shape)
back_hint_w = 92
back_hint_h = 92
back_hint_x = 40
back_hint_y = header_top + (header_h - back_hint_h)//2
draw.rounded_rectangle([back_hint_x, back_hint_y, back_hint_x + back_hint_w, back_hint_y + back_hint_h],
                       radius=20,
                       fill=WHITE,
                       outline=(240,240,245),
                       width=1)

# 9) Separator line above the bottom action bar (Apply date range). Do not draw the button itself.
# The detected apply button sits at y ~2768 with height 144; draw a faint shadow/divider above it.
apply_top = 2768
divider_y = apply_top - 16
draw.line([36, divider_y, W-36, divider_y], fill=DIVIDER, width=2)
# slight shadow band above the button area to suggest separation (very faint)
draw.rectangle([36, divider_y+2, W-36, divider_y+8], fill=SUBTLE_SHADOW)

# 10) Final subtle accents: a thin left and right margin lines to frame content area
draw.line([48, header_bottom+8, 48, H-220], fill=DIVIDER, width=1)
draw.line([W-48, header_bottom+8, W-48, H-220], fill=DIVIDER, width=1)

# Done. The actual texts, icons and the "Apply date range" button will be pasted on top separately.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_10_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-12/00_icon_Apply_date_range.png
try:
    _c0 = get_crop(0, 1344, 144)
    canvas.paste(_c0, (48, 2768), _c0)
except Exception:
    pass
layout["Apply_date_range"] = [48, 2768, 1392, 2912]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_10_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-12/01_icon_5.23.png
try:
    _c1 = get_crop(1, 59, 64)
    canvas.paste(_c1, (180, 2), _c1)
except Exception:
    pass
layout["5.23"] = [180, 2, 239, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_10_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-12/02_icon_5.23.png
try:
    _c2 = get_crop(2, 59, 65)
    canvas.paste(_c2, (115, 1), _c2)
except Exception:
    pass
layout["5.23"] = [115, 1, 174, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_10_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-12/03_icon_icon_3.png
try:
    _c3 = get_crop(3, 98, 109)
    canvas.paste(_c3, (74, 775), _c3)
except Exception:
    pass
layout["icon_3"] = [74, 775, 172, 884]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_10_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-12/04_icon_icon_4.png
try:
    _c4 = get_crop(4, 61, 62)
    canvas.paste(_c4, (310, 3), _c4)
except Exception:
    pass
layout["icon_4"] = [310, 3, 371, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_10_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-12/05_icon_May_2024.png
try:
    _c5 = get_crop(5, 119, 110)
    canvas.paste(_c5, (200, 773), _c5)
except Exception:
    pass
layout["May_2024"] = [200, 773, 319, 883]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_10_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-12/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 50, 61)
    canvas.paste(_c6, (249, 4), _c6)
except Exception:
    pass
layout["icon_6"] = [249, 4, 299, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_10_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-12/07_icon_5.23.png
try:
    _c7 = get_crop(7, 144, 144)
    canvas.paste(_c7, (12, 72), _c7)
except Exception:
    pass
layout["5.23"] = [12, 72, 156, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_10_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-12/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 56, 70)
    canvas.paste(_c8, (1316, 0), _c8)
except Exception:
    pass
layout["icon_8"] = [1316, 0, 1372, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_10_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-12/09_icon_May_2024.png
try:
    _c9 = get_crop(9, 142, 112)
    canvas.paste(_c9, (320, 771), _c9)
except Exception:
    pass
layout["May_2024"] = [320, 771, 462, 883]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_10_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-12/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 88, 69)
    canvas.paste(_c10, (1213, 0), _c10)
except Exception:
    pass
layout["icon_10"] = [1213, 0, 1301, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_10_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-12/11_icon_May_2024.png
try:
    _c11 = get_crop(11, 130, 113)
    canvas.paste(_c11, (456, 770), _c11)
except Exception:
    pass
layout["May_2024"] = [456, 770, 586, 883]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_10_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-12/12_icon_What_date.png
try:
    _c12 = get_crop(12, 319, 72)
    canvas.paste(_c12, (558, 112), _c12)
except Exception:
    pass
layout["What_date?"] = [558, 112, 877, 184]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_10_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-12/13_icon_5.23.png
try:
    _c13 = get_crop(13, 91, 62)
    canvas.paste(_c13, (17, 2), _c13)
except Exception:
    pass
layout["5.23"] = [17, 2, 108, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_10_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-12/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 50, 65)
    canvas.paste(_c14, (382, 2), _c14)
except Exception:
    pass
layout["icon_14"] = [382, 2, 432, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_10_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-12/15_icon_End_Date.png
try:
    _c15 = get_crop(15, 144, 144)
    canvas.paste(_c15, (54, 620), _c15)
except Exception:
    pass
layout["End_Date"] = [54, 620, 198, 764]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_10_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-12/16_icon_icon_16.png
try:
    _c16 = get_crop(16, 40, 65)
    canvas.paste(_c16, (1274, 0), _c16)
except Exception:
    pass
layout["icon_16"] = [1274, 0, 1314, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_10_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-12/17_icon_26.png
try:
    _c17 = get_crop(17, 132, 120)
    canvas.paste(_c17, (60, 1364), _c17)
except Exception:
    pass
layout["26"] = [60, 1364, 192, 1484]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_10_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-12/18_icon_27.png
try:
    _c18 = get_crop(18, 132, 120)
    canvas.paste(_c18, (192, 1364), _c18)
except Exception:
    pass
layout["27"] = [192, 1364, 324, 1484]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_10_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-12/19_icon_Next_month.png
try:
    _c19 = get_crop(19, 144, 144)
    canvas.paste(_c19, (846, 620), _c19)
except Exception:
    pass
layout["Next_month"] = [846, 620, 990, 764]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_10_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-12/20_text_Start_Date.png
try:
    _c20 = get_crop(20, 583, 144)
    canvas.paste(_c20, (48, 313), _c20)
except Exception:
    pass
layout["Start_Date"] = [48, 313, 631, 457]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_10_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-12/21_text_End_Date.png
try:
    _c21 = get_crop(21, 638, 114)
    canvas.paste(_c21, (48, 476), _c21)
except Exception:
    pass
layout["End_Date"] = [48, 476, 686, 590]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_10_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-12/22_text_May_2024.png
try:
    _c22 = get_crop(22, 198, 56)
    canvas.paste(_c22, (423, 666), _c22)
except Exception:
    pass
layout["May_2024"] = [423, 666, 621, 722]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_10_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-12/23_text_10.png
try:
    _c23 = get_crop(23, 132, 120)
    canvas.paste(_c23, (720, 1004), _c23)
except Exception:
    pass
layout["10"] = [720, 1004, 852, 1124]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_10_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-12/24_text_11.png
try:
    _c24 = get_crop(24, 132, 120)
    canvas.paste(_c24, (852, 1004), _c24)
except Exception:
    pass
layout["11"] = [852, 1004, 984, 1124]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_10_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-12/25_text_12.png
try:
    _c25 = get_crop(25, 132, 120)
    canvas.paste(_c25, (60, 1124), _c25)
except Exception:
    pass
layout["12"] = [60, 1124, 192, 1244]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_10_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-12/26_text_13.png
try:
    _c26 = get_crop(26, 132, 120)
    canvas.paste(_c26, (192, 1124), _c26)
except Exception:
    pass
layout["13"] = [192, 1124, 324, 1244]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_10_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-12/27_text_14.png
try:
    _c27 = get_crop(27, 132, 120)
    canvas.paste(_c27, (324, 1124), _c27)
except Exception:
    pass
layout["14"] = [324, 1124, 456, 1244]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_10_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-12/28_text_15.png
try:
    _c28 = get_crop(28, 132, 120)
    canvas.paste(_c28, (456, 1124), _c28)
except Exception:
    pass
layout["15"] = [456, 1124, 588, 1244]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_10_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-12/29_text_16.png
try:
    _c29 = get_crop(29, 132, 120)
    canvas.paste(_c29, (588, 1124), _c29)
except Exception:
    pass
layout["16"] = [588, 1124, 720, 1244]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_10_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-12/30_text_17.png
try:
    _c30 = get_crop(30, 132, 120)
    canvas.paste(_c30, (720, 1124), _c30)
except Exception:
    pass
layout["17"] = [720, 1124, 852, 1244]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_10_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-12/31_text_18.png
try:
    _c31 = get_crop(31, 132, 120)
    canvas.paste(_c31, (852, 1124), _c31)
except Exception:
    pass
layout["18"] = [852, 1124, 984, 1244]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_10_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-12/32_text_19.png
try:
    _c32 = get_crop(32, 132, 120)
    canvas.paste(_c32, (60, 1244), _c32)
except Exception:
    pass
layout["19"] = [60, 1244, 192, 1364]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_10_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-12/33_text_20.png
try:
    _c33 = get_crop(33, 132, 120)
    canvas.paste(_c33, (192, 1244), _c33)
except Exception:
    pass
layout["20"] = [192, 1244, 324, 1364]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_10_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-12/34_text_21.png
try:
    _c34 = get_crop(34, 132, 120)
    canvas.paste(_c34, (324, 1244), _c34)
except Exception:
    pass
layout["21"] = [324, 1244, 456, 1364]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_10_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-12/35_text_22.png
try:
    _c35 = get_crop(35, 132, 120)
    canvas.paste(_c35, (456, 1244), _c35)
except Exception:
    pass
layout["22"] = [456, 1244, 588, 1364]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_10_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-12/36_text_23.png
try:
    _c36 = get_crop(36, 132, 120)
    canvas.paste(_c36, (588, 1244), _c36)
except Exception:
    pass
layout["23"] = [588, 1244, 720, 1364]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_10_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-12/37_text_24.png
try:
    _c37 = get_crop(37, 132, 120)
    canvas.paste(_c37, (720, 1244), _c37)
except Exception:
    pass
layout["24"] = [720, 1244, 852, 1364]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_10_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-12/38_text_25.png
try:
    _c38 = get_crop(38, 132, 120)
    canvas.paste(_c38, (852, 1244), _c38)
except Exception:
    pass
layout["25"] = [852, 1244, 984, 1364]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_10_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-12/39_text_28.png
try:
    _c39 = get_crop(39, 132, 120)
    canvas.paste(_c39, (324, 1364), _c39)
except Exception:
    pass
layout["28"] = [324, 1364, 456, 1484]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_10_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-12/40_text_29.png
try:
    _c40 = get_crop(40, 132, 120)
    canvas.paste(_c40, (456, 1364), _c40)
except Exception:
    pass
layout["29"] = [456, 1364, 588, 1484]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_10_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-12/41_text_30.png
try:
    _c41 = get_crop(41, 132, 120)
    canvas.paste(_c41, (588, 1364), _c41)
except Exception:
    pass
layout["30"] = [588, 1364, 720, 1484]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_10_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-12/42_text_31.png
try:
    _c42 = get_crop(42, 132, 120)
    canvas.paste(_c42, (720, 1364), _c42)
except Exception:
    pass
layout["31"] = [720, 1364, 852, 1484]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_10_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-12/43_clickable_1.png
try:
    _c43 = get_crop(43, 132, 120)
    canvas.paste(_c43, (456, 884), _c43)
except Exception:
    pass
layout["1"] = [456, 884, 588, 1004]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_10_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-12/44_clickable_2.png
try:
    _c44 = get_crop(44, 132, 120)
    canvas.paste(_c44, (588, 884), _c44)
except Exception:
    pass
layout["2"] = [588, 884, 720, 1004]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_10_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-12/45_clickable_3.png
try:
    _c45 = get_crop(45, 132, 120)
    canvas.paste(_c45, (720, 884), _c45)
except Exception:
    pass
layout["3"] = [720, 884, 852, 1004]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_10_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-12/46_clickable_4.png
try:
    _c46 = get_crop(46, 132, 120)
    canvas.paste(_c46, (852, 884), _c46)
except Exception:
    pass
layout["4"] = [852, 884, 984, 1004]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_10_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-12/47_clickable_5.png
try:
    _c47 = get_crop(47, 132, 120)
    canvas.paste(_c47, (60, 1004), _c47)
except Exception:
    pass
layout["5"] = [60, 1004, 192, 1124]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_10_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-12/48_clickable_6.png
try:
    _c48 = get_crop(48, 132, 120)
    canvas.paste(_c48, (192, 1004), _c48)
except Exception:
    pass
layout["6"] = [192, 1004, 324, 1124]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_10_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-12/49_clickable_7.png
try:
    _c49 = get_crop(49, 132, 120)
    canvas.paste(_c49, (324, 1004), _c49)
except Exception:
    pass
layout["7"] = [324, 1004, 456, 1124]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_10_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-12/50_clickable_8.png
try:
    _c50 = get_crop(50, 132, 120)
    canvas.paste(_c50, (456, 1004), _c50)
except Exception:
    pass
layout["8"] = [456, 1004, 588, 1124]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_10_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-12/51_clickable_9.png
try:
    _c51 = get_crop(51, 132, 120)
    canvas.paste(_c51, (588, 1004), _c51)
except Exception:
    pass
layout["9"] = [588, 1004, 720, 1124]
