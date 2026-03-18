# page_id: page_eventbrite_1a166da440f24e2e9152f2c0e40eb7aa_12
# screenshot: 2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-14.png
# step_index: 12/16
# task: Open Eventbrite. Check "Sports" category. Filter events happening next month. Add the first event to your wishlist.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Background and structural elements for the UI page
# Available variables: canvas (PIL Image 1440x2960), draw (ImageDraw), fonts: font_sm, font_md, font_lg, font_xl

w, h = canvas.size

# Colors
bg_white = (255, 255, 255)
status_gray = (198, 198, 198)       # status bar background
status_div = (180, 180, 180)        # status bar bottom divider
header_div = (235, 232, 243)        # subtle purple-ish divider
card_border = (236, 233, 245)       # card outline
card_shadow = (245, 244, 247)       # card subtle fill
muted_line = (245, 243, 247)        # separators inside calendar
accent_purple = (47, 21, 59)        # deep purple for thin separators/accents
bottom_outline = (200, 198, 206)    # bottom apply container outline

# 1) Canvas background (dominant color)
draw.rectangle([(0, 0), (w, h)], fill=bg_white)

# 2) Status bar area at top (~84px)
status_h = 84
draw.rectangle([(0, 0), (w, status_h)], fill=status_gray)
# fine divider under status bar
draw.line([(0, status_h), (w, status_h)], fill=status_div, width=1)

# 3) Header / toolbar divider (subtle line under header/title area)
header_bottom = 220
draw.line([(40, header_bottom), (w - 40, header_bottom)], fill=header_div, width=1)

# 4) Calendar card background (rounded rect behind month and grid)
# Position tuned to leave margins where detected elements will be pasted on top
card_left = 48
card_top = 620
card_right = w - 48
card_bottom = 1480
card_radius = 18
# Slight shadow/backing rectangle (very subtle)
shadow_offset = 6
draw.rounded_rectangle(
    [(card_left + shadow_offset, card_top + shadow_offset),
     (card_right + shadow_offset, card_bottom + shadow_offset)],
    radius=card_radius,
    fill=card_shadow
)
# Main card
draw.rounded_rectangle(
    [(card_left, card_top), (card_right, card_bottom)],
    radius=card_radius,
    fill=bg_white,
    outline=card_border,
    width=1
)

# 5) Month header divider (thin accent near month title area)
month_header_y = card_top + 64
draw.line([(card_left + 60, month_header_y), (card_right - 60, month_header_y)], fill=muted_line, width=1)

# 6) Calendar grid subtle separators (rows and columns) - very faint so they act as structure only
# Grid geometry based on detected cell size ~132x120
cell_w = 132
cell_h = 120
grid_left = card_left + 12    # align slightly inward
grid_top = month_header_y + 40
cols = 7
rows = 5  # typical month view rows (visual structure only)
# vertical lines
for c in range(1, cols):
    x = grid_left + c * cell_w
    if x < card_right - 12:
        draw.line([(x, grid_top), (x, grid_top + rows * cell_h)], fill=muted_line, width=1)
# horizontal lines
for r in range(1, rows):
    y = grid_top + r * cell_h
    if y < card_bottom - 40:
        draw.line([(grid_left, y), (card_right - 12, y)], fill=muted_line, width=1)

# 7) Subtle weekday header baseline (just a light line to anchor weekday labels)
weekday_y = grid_top - 40
draw.line([(grid_left, weekday_y), (card_right - 12, weekday_y)], fill=muted_line, width=1)

# 8) Small accent dot on right of month to indicate chevron alignment (non-icon structural cue)
accent_x = card_right - 80
accent_y = card_top + 36
draw.ellipse([(accent_x - 6, accent_y - 6), (accent_x + 6, accent_y + 6)], fill=card_border)

# 9) Bottom "Apply" container background / outline (under detected button)
apply_top = 2720
apply_bottom = 2908
apply_left = 32
apply_right = w - 32
apply_radius = 12
# Outer outline box (to give the button area a container)
draw.rounded_rectangle(
    [(apply_left, apply_top), (apply_right, apply_bottom)],
    radius=apply_radius,
    fill=bg_white,
    outline=bottom_outline,
    width=6
)
# subtle top divider/shadow for the apply area
draw.line([(apply_left + 12, apply_top), (apply_right - 12, apply_top)], fill=muted_line, width=1)

# 10) Left-side vertical margin guideline (subtle) and right-side guideline
draw.line([(32, header_bottom + 12), (32, apply_top - 12)], fill=muted_line, width=1)
draw.line([(w - 32, header_bottom + 12), (w - 32, apply_top - 12)], fill=muted_line, width=1)

# 11) Tiny corner accents on the calendar card (to emphasize grouping) - very subtle dots
for dx, dy in [(12, 12), (card_right - card_left - 12, 12), (12, card_bottom - card_top - 12), (card_right - card_left - 12, card_bottom - card_top - 12)]:
    px = card_left + dx
    py = card_top + dy
    draw.ellipse([(px - 3, py - 3), (px + 3, py + 3)], fill=card_border)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_12_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-14/00_icon_Apply_date_range.png
try:
    _c0 = get_crop(0, 1344, 144)
    canvas.paste(_c0, (48, 2768), _c0)
except Exception:
    pass
layout["Apply_date_range"] = [48, 2768, 1392, 2912]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_12_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-14/01_icon_5.31.png
try:
    _c1 = get_crop(1, 60, 63)
    canvas.paste(_c1, (180, 2), _c1)
except Exception:
    pass
layout["5.31"] = [180, 2, 240, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_12_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-14/02_icon_5.31.png
try:
    _c2 = get_crop(2, 61, 65)
    canvas.paste(_c2, (113, 1), _c2)
except Exception:
    pass
layout["5.31"] = [113, 1, 174, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_12_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-14/03_icon_icon_3.png
try:
    _c3 = get_crop(3, 61, 62)
    canvas.paste(_c3, (310, 3), _c3)
except Exception:
    pass
layout["icon_3"] = [310, 3, 371, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_12_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-14/04_icon_5.31.png
try:
    _c4 = get_crop(4, 144, 144)
    canvas.paste(_c4, (12, 72), _c4)
except Exception:
    pass
layout["5.31"] = [12, 72, 156, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_12_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-14/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 50, 60)
    canvas.paste(_c5, (249, 5), _c5)
except Exception:
    pass
layout["icon_5"] = [249, 5, 299, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_12_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-14/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 57, 70)
    canvas.paste(_c6, (1316, 0), _c6)
except Exception:
    pass
layout["icon_6"] = [1316, 0, 1373, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_12_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-14/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 102, 103)
    canvas.paste(_c7, (73, 778), _c7)
except Exception:
    pass
layout["icon_7"] = [73, 778, 175, 881]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_12_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-14/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 88, 69)
    canvas.paste(_c8, (1212, 0), _c8)
except Exception:
    pass
layout["icon_8"] = [1212, 0, 1300, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_12_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-14/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 107, 100)
    canvas.paste(_c9, (202, 780), _c9)
except Exception:
    pass
layout["icon_9"] = [202, 780, 309, 880]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_12_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-14/10_icon_May_2024.png
try:
    _c10 = get_crop(10, 117, 91)
    canvas.paste(_c10, (458, 782), _c10)
except Exception:
    pass
layout["May_2024"] = [458, 782, 575, 873]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_12_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-14/11_icon_What_date.png
try:
    _c11 = get_crop(11, 319, 71)
    canvas.paste(_c11, (558, 112), _c11)
except Exception:
    pass
layout["What_date?"] = [558, 112, 877, 183]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_12_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-14/12_icon_27.png
try:
    _c12 = get_crop(12, 132, 120)
    canvas.paste(_c12, (192, 1364), _c12)
except Exception:
    pass
layout["27"] = [192, 1364, 324, 1484]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_12_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-14/13_icon_26.png
try:
    _c13 = get_crop(13, 132, 120)
    canvas.paste(_c13, (60, 1364), _c13)
except Exception:
    pass
layout["26"] = [60, 1364, 192, 1484]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_12_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-14/14_icon_5.31.png
try:
    _c14 = get_crop(14, 91, 62)
    canvas.paste(_c14, (16, 3), _c14)
except Exception:
    pass
layout["5.31"] = [16, 3, 107, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_12_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-14/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 43, 66)
    canvas.paste(_c15, (1272, 1), _c15)
except Exception:
    pass
layout["icon_15"] = [1272, 1, 1315, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_12_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-14/16_icon_icon_16.png
try:
    _c16 = get_crop(16, 50, 64)
    canvas.paste(_c16, (382, 2), _c16)
except Exception:
    pass
layout["icon_16"] = [382, 2, 432, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_12_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-14/17_icon_28.png
try:
    _c17 = get_crop(17, 132, 120)
    canvas.paste(_c17, (324, 1364), _c17)
except Exception:
    pass
layout["28"] = [324, 1364, 456, 1484]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_12_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-14/18_icon_Next_month.png
try:
    _c18 = get_crop(18, 144, 144)
    canvas.paste(_c18, (846, 620), _c18)
except Exception:
    pass
layout["Next_month"] = [846, 620, 990, 764]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_12_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-14/19_icon_15.png
try:
    _c19 = get_crop(19, 132, 120)
    canvas.paste(_c19, (456, 1004), _c19)
except Exception:
    pass
layout["15"] = [456, 1004, 588, 1124]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_12_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-14/20_text_Start_Date.png
try:
    _c20 = get_crop(20, 591, 144)
    canvas.paste(_c20, (48, 313), _c20)
except Exception:
    pass
layout["Start_Date"] = [48, 313, 639, 457]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_12_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-14/21_text_End_Date.png
try:
    _c21 = get_crop(21, 638, 114)
    canvas.paste(_c21, (48, 476), _c21)
except Exception:
    pass
layout["End_Date"] = [48, 476, 686, 590]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_12_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-14/22_text_May_2024.png
try:
    _c22 = get_crop(22, 198, 56)
    canvas.paste(_c22, (423, 666), _c22)
except Exception:
    pass
layout["May_2024"] = [423, 666, 621, 722]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_12_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-14/23_text_10.png
try:
    _c23 = get_crop(23, 132, 120)
    canvas.paste(_c23, (720, 1004), _c23)
except Exception:
    pass
layout["10"] = [720, 1004, 852, 1124]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_12_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-14/24_text_11.png
try:
    _c24 = get_crop(24, 132, 120)
    canvas.paste(_c24, (852, 1004), _c24)
except Exception:
    pass
layout["11"] = [852, 1004, 984, 1124]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_12_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-14/25_text_12.png
try:
    _c25 = get_crop(25, 132, 120)
    canvas.paste(_c25, (60, 1124), _c25)
except Exception:
    pass
layout["12"] = [60, 1124, 192, 1244]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_12_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-14/26_text_13.png
try:
    _c26 = get_crop(26, 132, 120)
    canvas.paste(_c26, (192, 1124), _c26)
except Exception:
    pass
layout["13"] = [192, 1124, 324, 1244]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_12_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-14/27_text_14.png
try:
    _c27 = get_crop(27, 132, 120)
    canvas.paste(_c27, (324, 1124), _c27)
except Exception:
    pass
layout["14"] = [324, 1124, 456, 1244]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_12_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-14/28_text_15.png
try:
    _c28 = get_crop(28, 132, 120)
    canvas.paste(_c28, (456, 1124), _c28)
except Exception:
    pass
layout["15"] = [456, 1124, 588, 1244]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_12_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-14/29_text_16.png
try:
    _c29 = get_crop(29, 132, 120)
    canvas.paste(_c29, (588, 1124), _c29)
except Exception:
    pass
layout["16"] = [588, 1124, 720, 1244]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_12_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-14/30_text_17.png
try:
    _c30 = get_crop(30, 132, 120)
    canvas.paste(_c30, (720, 1124), _c30)
except Exception:
    pass
layout["17"] = [720, 1124, 852, 1244]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_12_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-14/31_text_18.png
try:
    _c31 = get_crop(31, 132, 120)
    canvas.paste(_c31, (852, 1124), _c31)
except Exception:
    pass
layout["18"] = [852, 1124, 984, 1244]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_12_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-14/32_text_19.png
try:
    _c32 = get_crop(32, 132, 120)
    canvas.paste(_c32, (60, 1244), _c32)
except Exception:
    pass
layout["19"] = [60, 1244, 192, 1364]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_12_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-14/33_text_20.png
try:
    _c33 = get_crop(33, 132, 120)
    canvas.paste(_c33, (192, 1244), _c33)
except Exception:
    pass
layout["20"] = [192, 1244, 324, 1364]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_12_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-14/34_text_21.png
try:
    _c34 = get_crop(34, 132, 120)
    canvas.paste(_c34, (324, 1244), _c34)
except Exception:
    pass
layout["21"] = [324, 1244, 456, 1364]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_12_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-14/35_text_22.png
try:
    _c35 = get_crop(35, 132, 120)
    canvas.paste(_c35, (456, 1244), _c35)
except Exception:
    pass
layout["22"] = [456, 1244, 588, 1364]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_12_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-14/36_text_23.png
try:
    _c36 = get_crop(36, 132, 120)
    canvas.paste(_c36, (588, 1244), _c36)
except Exception:
    pass
layout["23"] = [588, 1244, 720, 1364]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_12_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-14/37_text_24.png
try:
    _c37 = get_crop(37, 132, 120)
    canvas.paste(_c37, (720, 1244), _c37)
except Exception:
    pass
layout["24"] = [720, 1244, 852, 1364]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_12_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-14/38_text_25.png
try:
    _c38 = get_crop(38, 132, 120)
    canvas.paste(_c38, (852, 1244), _c38)
except Exception:
    pass
layout["25"] = [852, 1244, 984, 1364]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_12_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-14/39_text_29.png
try:
    _c39 = get_crop(39, 132, 120)
    canvas.paste(_c39, (456, 1364), _c39)
except Exception:
    pass
layout["29"] = [456, 1364, 588, 1484]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_12_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-14/40_text_30.png
try:
    _c40 = get_crop(40, 132, 120)
    canvas.paste(_c40, (588, 1364), _c40)
except Exception:
    pass
layout["30"] = [588, 1364, 720, 1484]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_12_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-14/41_text_31.png
try:
    _c41 = get_crop(41, 132, 120)
    canvas.paste(_c41, (720, 1364), _c41)
except Exception:
    pass
layout["31"] = [720, 1364, 852, 1484]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_12_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-14/42_clickable_1.png
try:
    _c42 = get_crop(42, 132, 120)
    canvas.paste(_c42, (456, 884), _c42)
except Exception:
    pass
layout["1"] = [456, 884, 588, 1004]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_12_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-14/43_clickable_2.png
try:
    _c43 = get_crop(43, 132, 120)
    canvas.paste(_c43, (588, 884), _c43)
except Exception:
    pass
layout["2"] = [588, 884, 720, 1004]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_12_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-14/44_clickable_3.png
try:
    _c44 = get_crop(44, 132, 120)
    canvas.paste(_c44, (720, 884), _c44)
except Exception:
    pass
layout["3"] = [720, 884, 852, 1004]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_12_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-14/45_clickable_4.png
try:
    _c45 = get_crop(45, 132, 120)
    canvas.paste(_c45, (852, 884), _c45)
except Exception:
    pass
layout["4"] = [852, 884, 984, 1004]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_12_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-14/46_clickable_5.png
try:
    _c46 = get_crop(46, 132, 120)
    canvas.paste(_c46, (60, 1004), _c46)
except Exception:
    pass
layout["5"] = [60, 1004, 192, 1124]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_12_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-14/47_clickable_6.png
try:
    _c47 = get_crop(47, 132, 120)
    canvas.paste(_c47, (192, 1004), _c47)
except Exception:
    pass
layout["6"] = [192, 1004, 324, 1124]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_12_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-14/48_clickable_7.png
try:
    _c48 = get_crop(48, 132, 120)
    canvas.paste(_c48, (324, 1004), _c48)
except Exception:
    pass
layout["7"] = [324, 1004, 456, 1124]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_12_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-14/49_clickable_9.png
try:
    _c49 = get_crop(49, 132, 120)
    canvas.paste(_c49, (588, 1004), _c49)
except Exception:
    pass
layout["9"] = [588, 1004, 720, 1124]
