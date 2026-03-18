# page_id: page_eventbrite_eb32c51543d749539b68e6c61ff72fb8_13
# screenshot: 2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-15.png
# step_index: 13/19
# task: Open Eventbrite. Set the city to San Francisco. Filter for events occurring between May 1st and May 15th under the category 'Music'. Select the first event and check the pricing options available.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw background and structural elements for the UI page
# Available variables:
# - canvas: PIL Image (1440x2960 RGB)
# - draw: PIL ImageDraw object
# - font_sm, font_md, font_lg, font_xl

# Colors
bg_white = (255, 255, 255)
status_bar_gray = (200, 200, 200)        # top status bar background
status_bar_dark = (170, 170, 170)        # subtle bottom edge of status bar
divider_gray = (230, 230, 235)           # thin dividers / subtle backgrounds
panel_bg = (250, 250, 252)               # very light panel background
panel_border = (220, 218, 235)           # soft purple-gray border for panels
muted_purple = (85, 57, 113)             # theme purple for accents (used only sparingly)
shadow_color = (240, 240, 244)

W, H = canvas.size

# Fill overall background (canvas is already white, but ensure consistent color)
draw.rectangle([0, 0, W, H], fill=bg_white)

# Status bar (top area ~72px tall)
status_h = 72
draw.rectangle([0, 0, W, status_h], fill=status_bar_gray)
# subtle bottom edge/shadow under status bar
draw.line([(0, status_h-1), (W, status_h-1)], fill=status_bar_dark, width=2)

# Header area (toolbar) beneath status bar
header_top = status_h
header_bottom = 180
draw.rectangle([0, header_top, W, header_bottom], fill=bg_white)
# subtle divider under header
draw.line([(24, header_bottom-1), (W-24, header_bottom-1)], fill=divider_gray, width=1)

# Drop shadow under header (soft)
for i, a in enumerate([6, 4, 2]):
    y = header_bottom + i
    draw.line([(24, y), (W-24, y)], fill=(245 - i*3, 245 - i*3, 247 - i*2), width=1)

# Calendar / content panel background (rounded rectangle)
# Keep this panel above the large whitespace but avoid touching the bottom action button area.
panel_left = 48
panel_right = W - 48
panel_top = 620
panel_bottom = 1488  # leave space below for content and the bottom button area
panel_radius = 28

# Outer soft shadow (subtle)
shadow_box = [panel_left + 6, panel_top + 10, panel_right + 6, panel_bottom + 14]
draw.rounded_rectangle(shadow_box, radius=panel_radius+2, fill=shadow_color)

# Panel background
panel_box = [panel_left, panel_top, panel_right, panel_bottom]
draw.rounded_rectangle(panel_box, radius=panel_radius, fill=panel_bg, outline=panel_border, width=2)

# Horizontal separators inside the panel to suggest grouping (do not draw any text or icons)
sep_y1 = panel_top + 72
sep_y2 = panel_top + 160
draw.line([(panel_left + 24, sep_y1), (panel_right - 24, sep_y1)], fill=divider_gray, width=1)
draw.line([(panel_left + 24, sep_y2), (panel_right - 24, sep_y2)], fill=divider_gray, width=1)

# Subtle grid hints for the calendar area (light strokes only, no numbers)
# These lines mark approximate rows/columns but are very faint so as not to duplicate content.
cal_left = panel_left + 36
cal_right = panel_right - 36
cal_top = sep_y2 + 24
cal_bottom = panel_bottom - 40

# Horizontal row lines
rows = 6
row_h = (cal_bottom - cal_top) / rows
for i in range(1, rows):
    y = int(cal_top + i * row_h)
    draw.line([(cal_left, y), (cal_right, y)], fill=(245,245,247), width=1)

# Vertical column hints (7 columns for weekdays)
cols = 7
col_w = (cal_right - cal_left) / cols
for j in range(1, cols):
    x = int(cal_left + j * col_w)
    draw.line([(x, cal_top), (x, cal_bottom)], fill=(245,245,247), width=1)

# Small decorative month navigation block background (behind month label area)
month_block_w = 260
month_block_h = 56
month_block_x = (W - month_block_w) // 2
month_block_y = panel_top + 16
draw.rectangle([month_block_x - 8, month_block_y - 6, month_block_x + month_block_w + 8, month_block_y + month_block_h + 6],
               fill=bg_white, outline=(245,245,247), width=1)

# A right-side chevron touch area hint (very faint), positioned where the "next month" chevron sits in the UI.
chev_x = panel_right - 72
chev_y = month_block_y + month_block_h // 2
draw.rectangle([chev_x - 28, month_block_y - 6, chev_x + 28, month_block_y + month_block_h + 6], fill=(255,255,255,0), outline=(245,245,247))
# draw a tiny faint accent dot to the right (not an arrow icon)
draw.ellipse((chev_x - 2, chev_y - 2, chev_x + 2, chev_y + 2), fill=divider_gray)

# Large empty content area below the calendar (keeps it visually separated)
content_top = panel_bottom + 24
content_left = 48
content_right = W - 48
content_bottom = 2640  # keep above bottom action area
draw.rectangle([content_left, content_top, content_right, content_bottom], fill=bg_white)

# Subtle separator line to indicate end of main content before the action area (do not overlap the action button)
sep_before_action = content_bottom + 40
draw.line([(24, sep_before_action), (W-24, sep_before_action)], fill=divider_gray, width=1)

# Small rounded inset near the very bottom to suggest the button zone (but do NOT draw the button itself)
inset_y1 = sep_before_action + 18
inset_y2 = inset_y1 + 80
inset_left = 36
inset_right = W - 36
draw.rounded_rectangle([inset_left, inset_y1, inset_right, inset_y2], radius=12, outline=(220,220,225), width=2, fill=(255,255,255))

# Final subtle side margins/shadows for depth
draw.line([(12, header_bottom), (12, H-12)], fill=(250,250,252), width=1)
draw.line([(W-12, header_bottom), (W-12, H-12)], fill=(250,250,252), width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_13_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-15/00_icon_Apply_date_range.png
try:
    _c0 = get_crop(0, 1344, 144)
    canvas.paste(_c0, (48, 2768), _c0)
except Exception:
    pass
layout["Apply_date_range"] = [48, 2768, 1392, 2912]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_13_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-15/01_icon_icon_1.png
try:
    _c1 = get_crop(1, 50, 71)
    canvas.paste(_c1, (1154, 0), _c1)
except Exception:
    pass
layout["icon_1"] = [1154, 0, 1204, 71]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_13_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-15/02_icon_7.48.png
try:
    _c2 = get_crop(2, 58, 63)
    canvas.paste(_c2, (181, 2), _c2)
except Exception:
    pass
layout["7.48"] = [181, 2, 239, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_13_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-15/03_icon_7.48.png
try:
    _c3 = get_crop(3, 59, 65)
    canvas.paste(_c3, (115, 1), _c3)
except Exception:
    pass
layout["7.48"] = [115, 1, 174, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_13_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-15/04_icon_icon_4.png
try:
    _c4 = get_crop(4, 61, 62)
    canvas.paste(_c4, (310, 3), _c4)
except Exception:
    pass
layout["icon_4"] = [310, 3, 371, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_13_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-15/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 100, 71)
    canvas.paste(_c5, (1210, 0), _c5)
except Exception:
    pass
layout["icon_5"] = [1210, 0, 1310, 71]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_13_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-15/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 50, 59)
    canvas.paste(_c6, (249, 5), _c6)
except Exception:
    pass
layout["icon_6"] = [249, 5, 299, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_13_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-15/07_icon_7.48.png
try:
    _c7 = get_crop(7, 144, 144)
    canvas.paste(_c7, (12, 72), _c7)
except Exception:
    pass
layout["7.48"] = [12, 72, 156, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_13_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-15/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 53, 70)
    canvas.paste(_c8, (1318, 0), _c8)
except Exception:
    pass
layout["icon_8"] = [1318, 0, 1371, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_13_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-15/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 102, 103)
    canvas.paste(_c9, (73, 778), _c9)
except Exception:
    pass
layout["icon_9"] = [73, 778, 175, 881]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_13_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-15/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 105, 99)
    canvas.paste(_c10, (203, 780), _c10)
except Exception:
    pass
layout["icon_10"] = [203, 780, 308, 879]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_13_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-15/11_icon_May_2024.png
try:
    _c11 = get_crop(11, 117, 91)
    canvas.paste(_c11, (458, 782), _c11)
except Exception:
    pass
layout["May_2024"] = [458, 782, 575, 873]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_13_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-15/12_icon_27.png
try:
    _c12 = get_crop(12, 132, 120)
    canvas.paste(_c12, (192, 1364), _c12)
except Exception:
    pass
layout["27"] = [192, 1364, 324, 1484]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_13_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-15/13_icon_26.png
try:
    _c13 = get_crop(13, 132, 120)
    canvas.paste(_c13, (60, 1364), _c13)
except Exception:
    pass
layout["26"] = [60, 1364, 192, 1484]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_13_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-15/14_icon_What_date.png
try:
    _c14 = get_crop(14, 319, 73)
    canvas.paste(_c14, (558, 111), _c14)
except Exception:
    pass
layout["What_date?"] = [558, 111, 877, 184]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_13_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-15/15_icon_7.48.png
try:
    _c15 = get_crop(15, 90, 62)
    canvas.paste(_c15, (17, 3), _c15)
except Exception:
    pass
layout["7.48"] = [17, 3, 107, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_13_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-15/16_icon_28.png
try:
    _c16 = get_crop(16, 132, 120)
    canvas.paste(_c16, (324, 1364), _c16)
except Exception:
    pass
layout["28"] = [324, 1364, 456, 1484]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_13_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-15/17_icon_icon_17.png
try:
    _c17 = get_crop(17, 50, 65)
    canvas.paste(_c17, (382, 2), _c17)
except Exception:
    pass
layout["icon_17"] = [382, 2, 432, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_13_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-15/18_icon_15.png
try:
    _c18 = get_crop(18, 132, 120)
    canvas.paste(_c18, (456, 1004), _c18)
except Exception:
    pass
layout["15"] = [456, 1004, 588, 1124]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_13_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-15/19_icon_Next_month.png
try:
    _c19 = get_crop(19, 144, 144)
    canvas.paste(_c19, (846, 620), _c19)
except Exception:
    pass
layout["Next_month"] = [846, 620, 990, 764]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_13_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-15/20_text_Start_Date.png
try:
    _c20 = get_crop(20, 591, 144)
    canvas.paste(_c20, (48, 313), _c20)
except Exception:
    pass
layout["Start_Date"] = [48, 313, 639, 457]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_13_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-15/21_text_End_Date.png
try:
    _c21 = get_crop(21, 638, 114)
    canvas.paste(_c21, (48, 476), _c21)
except Exception:
    pass
layout["End_Date"] = [48, 476, 686, 590]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_13_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-15/22_text_May_2024.png
try:
    _c22 = get_crop(22, 198, 56)
    canvas.paste(_c22, (423, 666), _c22)
except Exception:
    pass
layout["May_2024"] = [423, 666, 621, 722]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_13_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-15/23_text_10.png
try:
    _c23 = get_crop(23, 132, 120)
    canvas.paste(_c23, (720, 1004), _c23)
except Exception:
    pass
layout["10"] = [720, 1004, 852, 1124]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_13_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-15/24_text_11.png
try:
    _c24 = get_crop(24, 132, 120)
    canvas.paste(_c24, (852, 1004), _c24)
except Exception:
    pass
layout["11"] = [852, 1004, 984, 1124]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_13_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-15/25_text_12.png
try:
    _c25 = get_crop(25, 132, 120)
    canvas.paste(_c25, (60, 1124), _c25)
except Exception:
    pass
layout["12"] = [60, 1124, 192, 1244]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_13_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-15/26_text_13.png
try:
    _c26 = get_crop(26, 132, 120)
    canvas.paste(_c26, (192, 1124), _c26)
except Exception:
    pass
layout["13"] = [192, 1124, 324, 1244]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_13_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-15/27_text_14.png
try:
    _c27 = get_crop(27, 132, 120)
    canvas.paste(_c27, (324, 1124), _c27)
except Exception:
    pass
layout["14"] = [324, 1124, 456, 1244]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_13_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-15/28_text_15.png
try:
    _c28 = get_crop(28, 132, 120)
    canvas.paste(_c28, (456, 1124), _c28)
except Exception:
    pass
layout["15"] = [456, 1124, 588, 1244]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_13_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-15/29_text_16.png
try:
    _c29 = get_crop(29, 132, 120)
    canvas.paste(_c29, (588, 1124), _c29)
except Exception:
    pass
layout["16"] = [588, 1124, 720, 1244]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_13_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-15/30_text_17.png
try:
    _c30 = get_crop(30, 132, 120)
    canvas.paste(_c30, (720, 1124), _c30)
except Exception:
    pass
layout["17"] = [720, 1124, 852, 1244]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_13_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-15/31_text_18.png
try:
    _c31 = get_crop(31, 132, 120)
    canvas.paste(_c31, (852, 1124), _c31)
except Exception:
    pass
layout["18"] = [852, 1124, 984, 1244]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_13_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-15/32_text_19.png
try:
    _c32 = get_crop(32, 132, 120)
    canvas.paste(_c32, (60, 1244), _c32)
except Exception:
    pass
layout["19"] = [60, 1244, 192, 1364]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_13_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-15/33_text_20.png
try:
    _c33 = get_crop(33, 132, 120)
    canvas.paste(_c33, (192, 1244), _c33)
except Exception:
    pass
layout["20"] = [192, 1244, 324, 1364]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_13_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-15/34_text_21.png
try:
    _c34 = get_crop(34, 132, 120)
    canvas.paste(_c34, (324, 1244), _c34)
except Exception:
    pass
layout["21"] = [324, 1244, 456, 1364]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_13_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-15/35_text_22.png
try:
    _c35 = get_crop(35, 132, 120)
    canvas.paste(_c35, (456, 1244), _c35)
except Exception:
    pass
layout["22"] = [456, 1244, 588, 1364]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_13_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-15/36_text_23.png
try:
    _c36 = get_crop(36, 132, 120)
    canvas.paste(_c36, (588, 1244), _c36)
except Exception:
    pass
layout["23"] = [588, 1244, 720, 1364]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_13_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-15/37_text_24.png
try:
    _c37 = get_crop(37, 132, 120)
    canvas.paste(_c37, (720, 1244), _c37)
except Exception:
    pass
layout["24"] = [720, 1244, 852, 1364]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_13_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-15/38_text_25.png
try:
    _c38 = get_crop(38, 132, 120)
    canvas.paste(_c38, (852, 1244), _c38)
except Exception:
    pass
layout["25"] = [852, 1244, 984, 1364]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_13_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-15/39_text_29.png
try:
    _c39 = get_crop(39, 132, 120)
    canvas.paste(_c39, (456, 1364), _c39)
except Exception:
    pass
layout["29"] = [456, 1364, 588, 1484]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_13_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-15/40_text_30.png
try:
    _c40 = get_crop(40, 132, 120)
    canvas.paste(_c40, (588, 1364), _c40)
except Exception:
    pass
layout["30"] = [588, 1364, 720, 1484]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_13_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-15/41_text_31.png
try:
    _c41 = get_crop(41, 132, 120)
    canvas.paste(_c41, (720, 1364), _c41)
except Exception:
    pass
layout["31"] = [720, 1364, 852, 1484]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_13_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-15/42_clickable_1.png
try:
    _c42 = get_crop(42, 132, 120)
    canvas.paste(_c42, (456, 884), _c42)
except Exception:
    pass
layout["1"] = [456, 884, 588, 1004]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_13_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-15/43_clickable_2.png
try:
    _c43 = get_crop(43, 132, 120)
    canvas.paste(_c43, (588, 884), _c43)
except Exception:
    pass
layout["2"] = [588, 884, 720, 1004]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_13_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-15/44_clickable_3.png
try:
    _c44 = get_crop(44, 132, 120)
    canvas.paste(_c44, (720, 884), _c44)
except Exception:
    pass
layout["3"] = [720, 884, 852, 1004]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_13_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-15/45_clickable_4.png
try:
    _c45 = get_crop(45, 132, 120)
    canvas.paste(_c45, (852, 884), _c45)
except Exception:
    pass
layout["4"] = [852, 884, 984, 1004]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_13_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-15/46_clickable_5.png
try:
    _c46 = get_crop(46, 132, 120)
    canvas.paste(_c46, (60, 1004), _c46)
except Exception:
    pass
layout["5"] = [60, 1004, 192, 1124]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_13_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-15/47_clickable_6.png
try:
    _c47 = get_crop(47, 132, 120)
    canvas.paste(_c47, (192, 1004), _c47)
except Exception:
    pass
layout["6"] = [192, 1004, 324, 1124]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_13_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-15/48_clickable_7.png
try:
    _c48 = get_crop(48, 132, 120)
    canvas.paste(_c48, (324, 1004), _c48)
except Exception:
    pass
layout["7"] = [324, 1004, 456, 1124]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_13_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-15/49_clickable_9.png
try:
    _c49 = get_crop(49, 132, 120)
    canvas.paste(_c49, (588, 1004), _c49)
except Exception:
    pass
layout["9"] = [588, 1004, 720, 1124]
