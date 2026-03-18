# page_id: page_eventbrite_1a166da440f24e2e9152f2c0e40eb7aa_13
# screenshot: 2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-15.png
# step_index: 13/16
# task: Open Eventbrite. Check "Sports" category. Filter events happening next month. Add the first event to your wishlist.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw general background and UI structure for the calendar/date range screen.
# Available variables: canvas (PIL Image 1440x2960), draw (ImageDraw), font_sm, font_md, font_lg, font_xl

# Colors
status_bar_color = (200, 200, 200)        # light gray status bar
divider_color = (230, 229, 235)           # very light divider
card_bg = (250, 250, 252)                 # off-white card background
card_border = (235, 232, 238)             # subtle border for cards
muted_line = (210, 206, 220)              # muted separators
accent_purple = (55, 20, 70)              # deep purple for thin accents

W, H = canvas.size

# 1) Status bar (top area with icons will be pasted on top)
status_h = 72
draw.rectangle([(0, 0), (W, status_h)], fill=status_bar_color)

# subtle darker top edge to mimic device bezel
draw.line([(0, 0), (W, 1)], fill=(190, 190, 190))

# 2) Header area background (below status bar)
header_top = status_h
header_bottom = 220
# keep it white but draw a faint accent band behind where title sits
draw.rectangle([(0, header_top), (W, header_bottom)], fill=(255, 255, 255))

# faint horizontal divider under header
draw.line([(40, header_bottom), (W - 40, header_bottom)], fill=divider_color, width=1)

# 3) Large calendar/card container (rounded rectangle) to group calendar UI
# Positioned to leave safe margins and not obscure bottom action area.
card_x0 = 36
card_x1 = W - 36
card_y0 = 240
card_y1 = 1700
card_radius = 28

# subtle shadow rectangle under the card (very light, offset)
shadow_offset = 8
draw.rounded_rectangle(
    [(card_x0 + shadow_offset, card_y0 + shadow_offset),
     (card_x1 + shadow_offset, card_y1 + shadow_offset)],
    radius=card_radius + 2,
    fill=(245, 244, 246)
)

# main card background
draw.rounded_rectangle(
    [(card_x0, card_y0), (card_x1, card_y1)],
    radius=card_radius,
    fill=card_bg,
    outline=card_border,
    width=2
)

# 4) Sub-section separators inside the card to create visual grouping
# Separator under "Start/End Date" area (approx location; content will be pasted over)
sep_y = 520
draw.line([(card_x0 + 24, sep_y), (card_x1 - 24, sep_y)], fill=muted_line, width=1)

# Another subtle separator where the calendar grid visually begins
calendar_sep_y = 780
draw.line([(card_x0 + 24, calendar_sep_y), (card_x1 - 24, calendar_sep_y)], fill=muted_line, width=1)

# 5) Small decorative accent near the month title area (no text drawn)
# a tiny purple chevron-like block to hint at navigation (keeps it abstract)
accent_w = 10
accent_h = 40
accent_x = (W // 2) + 160
accent_y = 640
draw.rectangle([(accent_x, accent_y), (accent_x + accent_w, accent_y + accent_h)], fill=accent_purple)

# 6) Light grid guide lines for calendar area (very faint, non-intrusive)
# these are background guides only; numbers/dates will be pasted on top
grid_left = card_x0 + 60
grid_right = card_x1 - 240
grid_top = 820
grid_cell_w = 132
grid_cell_h = 120
cols = 7
rows = 6

for r in range(rows):
    y = grid_top + r * grid_cell_h
    # horizontal guide line
    draw.line([(grid_left, y), (grid_right, y)], fill=(248, 247, 249), width=1)
for c in range(cols + 1):
    x = grid_left + c * grid_cell_w
    draw.line([(x, grid_top), (x, grid_top + rows * grid_cell_h)], fill=(248, 247, 249), width=1)

# 7) Subtle vertical divider between left margin and content inside the card
draw.line([(card_x0 + 24, card_y0 + 24), (card_x0 + 24, card_y1 - 24)], fill=divider_color, width=1)

# 8) Bottom safe area shading (above the action button) to separate content from the bottom action bar
# Leave the exact apply-button bounding box untouched; just draw a faint band above it.
bottom_band_top = 2660
bottom_band_bottom = 2720
draw.rectangle([(24, bottom_band_top), (W - 24, bottom_band_bottom)], fill=(255, 255, 255, 0), outline=divider_color)

# 9) Fine top-left/back area accent (do not draw icon)
# small rounded background behind back navigation area (subtle)
back_bg_w = 84
back_bg_h = 84
back_bg_x = 36
back_bg_y = header_top + 12
draw.rounded_rectangle(
    [(back_bg_x, back_bg_y), (back_bg_x + back_bg_w, back_bg_y + back_bg_h)],
    radius=18,
    fill=(255, 255, 255),
    outline=divider_color,
    width=1
)

# Note: All textual labels, icons, and interactive controls are intentionally omitted.
# The crops for those elements will be pasted on top of this structural background by the pipeline.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_13_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-15/00_icon_Apply_date_range.png
try:
    _c0 = get_crop(0, 1344, 144)
    canvas.paste(_c0, (48, 2768), _c0)
except Exception:
    pass
layout["Apply_date_range"] = [48, 2768, 1392, 2912]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_13_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-15/01_icon_31.png
try:
    _c1 = get_crop(1, 132, 120)
    canvas.paste(_c1, (720, 1364), _c1)
except Exception:
    pass
layout["31"] = [720, 1364, 852, 1484]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_13_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-15/02_icon_28.png
try:
    _c2 = get_crop(2, 132, 120)
    canvas.paste(_c2, (324, 1364), _c2)
except Exception:
    pass
layout["28"] = [324, 1364, 456, 1484]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_13_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-15/03_icon_5.31.png
try:
    _c3 = get_crop(3, 60, 63)
    canvas.paste(_c3, (180, 2), _c3)
except Exception:
    pass
layout["5.31"] = [180, 2, 240, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_13_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-15/04_icon_27.png
try:
    _c4 = get_crop(4, 132, 120)
    canvas.paste(_c4, (192, 1364), _c4)
except Exception:
    pass
layout["27"] = [192, 1364, 324, 1484]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_13_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-15/05_icon_5.31.png
try:
    _c5 = get_crop(5, 61, 65)
    canvas.paste(_c5, (113, 1), _c5)
except Exception:
    pass
layout["5.31"] = [113, 1, 174, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_13_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-15/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 61, 62)
    canvas.paste(_c6, (310, 3), _c6)
except Exception:
    pass
layout["icon_6"] = [310, 3, 371, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_13_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-15/07_icon_29.png
try:
    _c7 = get_crop(7, 132, 120)
    canvas.paste(_c7, (456, 1364), _c7)
except Exception:
    pass
layout["29"] = [456, 1364, 588, 1484]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_13_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-15/08_icon_26.png
try:
    _c8 = get_crop(8, 132, 120)
    canvas.paste(_c8, (60, 1364), _c8)
except Exception:
    pass
layout["26"] = [60, 1364, 192, 1484]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_13_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-15/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 50, 60)
    canvas.paste(_c9, (249, 5), _c9)
except Exception:
    pass
layout["icon_9"] = [249, 5, 299, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_13_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-15/10_icon_5.31.png
try:
    _c10 = get_crop(10, 144, 144)
    canvas.paste(_c10, (12, 72), _c10)
except Exception:
    pass
layout["5.31"] = [12, 72, 156, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_13_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-15/11_icon_23.png
try:
    _c11 = get_crop(11, 132, 120)
    canvas.paste(_c11, (588, 1244), _c11)
except Exception:
    pass
layout["23"] = [588, 1244, 720, 1364]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_13_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-15/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 57, 70)
    canvas.paste(_c12, (1316, 0), _c12)
except Exception:
    pass
layout["icon_12"] = [1316, 0, 1373, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_13_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-15/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 87, 69)
    canvas.paste(_c13, (1212, 0), _c13)
except Exception:
    pass
layout["icon_13"] = [1212, 0, 1299, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_13_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-15/14_icon_24.png
try:
    _c14 = get_crop(14, 132, 120)
    canvas.paste(_c14, (720, 1244), _c14)
except Exception:
    pass
layout["24"] = [720, 1244, 852, 1364]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_13_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-15/15_icon_5.31.png
try:
    _c15 = get_crop(15, 91, 62)
    canvas.paste(_c15, (16, 3), _c15)
except Exception:
    pass
layout["5.31"] = [16, 3, 107, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_13_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-15/16_icon_icon_16.png
try:
    _c16 = get_crop(16, 109, 105)
    canvas.paste(_c16, (202, 777), _c16)
except Exception:
    pass
layout["icon_16"] = [202, 777, 311, 882]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_13_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-15/17_icon_22.png
try:
    _c17 = get_crop(17, 132, 120)
    canvas.paste(_c17, (456, 1244), _c17)
except Exception:
    pass
layout["22"] = [456, 1244, 588, 1364]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_13_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-15/18_icon_What_date.png
try:
    _c18 = get_crop(18, 319, 71)
    canvas.paste(_c18, (558, 112), _c18)
except Exception:
    pass
layout["What_date?"] = [558, 112, 877, 183]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_13_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-15/19_icon_icon_19.png
try:
    _c19 = get_crop(19, 106, 109)
    canvas.paste(_c19, (71, 776), _c19)
except Exception:
    pass
layout["icon_19"] = [71, 776, 177, 885]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_13_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-15/20_icon_10.png
try:
    _c20 = get_crop(20, 132, 120)
    canvas.paste(_c20, (588, 884), _c20)
except Exception:
    pass
layout["10"] = [588, 884, 720, 1004]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_13_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-15/21_icon_icon_21.png
try:
    _c21 = get_crop(21, 50, 64)
    canvas.paste(_c21, (382, 2), _c21)
except Exception:
    pass
layout["icon_21"] = [382, 2, 432, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_13_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-15/22_icon_icon_22.png
try:
    _c22 = get_crop(22, 44, 67)
    canvas.paste(_c22, (1272, 1), _c22)
except Exception:
    pass
layout["icon_22"] = [1272, 1, 1316, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_13_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-15/23_icon_Next_month.png
try:
    _c23 = get_crop(23, 144, 144)
    canvas.paste(_c23, (846, 620), _c23)
except Exception:
    pass
layout["Next_month"] = [846, 620, 990, 764]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_13_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-15/24_icon_21.png
try:
    _c24 = get_crop(24, 132, 120)
    canvas.paste(_c24, (324, 1244), _c24)
except Exception:
    pass
layout["21"] = [324, 1244, 456, 1364]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_13_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-15/25_text_Start_Date.png
try:
    _c25 = get_crop(25, 591, 144)
    canvas.paste(_c25, (48, 313), _c25)
except Exception:
    pass
layout["Start_Date"] = [48, 313, 639, 457]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_13_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-15/26_text_End_Date.png
try:
    _c26 = get_crop(26, 583, 114)
    canvas.paste(_c26, (48, 476), _c26)
except Exception:
    pass
layout["End_Date"] = [48, 476, 631, 590]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_13_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-15/27_text_May_2024.png
try:
    _c27 = get_crop(27, 198, 56)
    canvas.paste(_c27, (423, 666), _c27)
except Exception:
    pass
layout["May_2024"] = [423, 666, 621, 722]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_13_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-15/28_text_10.png
try:
    _c28 = get_crop(28, 132, 120)
    canvas.paste(_c28, (720, 1004), _c28)
except Exception:
    pass
layout["10"] = [720, 1004, 852, 1124]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_13_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-15/29_text_11.png
try:
    _c29 = get_crop(29, 132, 120)
    canvas.paste(_c29, (852, 1004), _c29)
except Exception:
    pass
layout["11"] = [852, 1004, 984, 1124]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_13_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-15/30_text_12.png
try:
    _c30 = get_crop(30, 132, 120)
    canvas.paste(_c30, (60, 1124), _c30)
except Exception:
    pass
layout["12"] = [60, 1124, 192, 1244]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_13_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-15/31_text_13.png
try:
    _c31 = get_crop(31, 132, 120)
    canvas.paste(_c31, (192, 1124), _c31)
except Exception:
    pass
layout["13"] = [192, 1124, 324, 1244]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_13_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-15/32_text_14.png
try:
    _c32 = get_crop(32, 132, 120)
    canvas.paste(_c32, (324, 1124), _c32)
except Exception:
    pass
layout["14"] = [324, 1124, 456, 1244]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_13_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-15/33_text_15.png
try:
    _c33 = get_crop(33, 132, 120)
    canvas.paste(_c33, (456, 1124), _c33)
except Exception:
    pass
layout["15"] = [456, 1124, 588, 1244]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_13_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-15/34_text_16.png
try:
    _c34 = get_crop(34, 132, 120)
    canvas.paste(_c34, (588, 1124), _c34)
except Exception:
    pass
layout["16"] = [588, 1124, 720, 1244]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_13_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-15/35_text_17.png
try:
    _c35 = get_crop(35, 132, 120)
    canvas.paste(_c35, (720, 1124), _c35)
except Exception:
    pass
layout["17"] = [720, 1124, 852, 1244]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_13_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-15/36_text_18.png
try:
    _c36 = get_crop(36, 132, 120)
    canvas.paste(_c36, (852, 1124), _c36)
except Exception:
    pass
layout["18"] = [852, 1124, 984, 1244]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_13_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-15/37_text_19.png
try:
    _c37 = get_crop(37, 132, 120)
    canvas.paste(_c37, (60, 1244), _c37)
except Exception:
    pass
layout["19"] = [60, 1244, 192, 1364]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_13_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-15/38_text_20.png
try:
    _c38 = get_crop(38, 132, 120)
    canvas.paste(_c38, (192, 1244), _c38)
except Exception:
    pass
layout["20"] = [192, 1244, 324, 1364]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_13_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-15/39_text_25.png
try:
    _c39 = get_crop(39, 132, 120)
    canvas.paste(_c39, (852, 1244), _c39)
except Exception:
    pass
layout["25"] = [852, 1244, 984, 1364]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_13_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-15/40_text_30.png
try:
    _c40 = get_crop(40, 132, 120)
    canvas.paste(_c40, (588, 1364), _c40)
except Exception:
    pass
layout["30"] = [588, 1364, 720, 1484]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_13_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-15/41_clickable_1.png
try:
    _c41 = get_crop(41, 132, 120)
    canvas.paste(_c41, (456, 884), _c41)
except Exception:
    pass
layout["1"] = [456, 884, 588, 1004]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_13_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-15/42_clickable_3.png
try:
    _c42 = get_crop(42, 132, 120)
    canvas.paste(_c42, (720, 884), _c42)
except Exception:
    pass
layout["3"] = [720, 884, 852, 1004]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_13_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-15/43_clickable_4.png
try:
    _c43 = get_crop(43, 132, 120)
    canvas.paste(_c43, (852, 884), _c43)
except Exception:
    pass
layout["4"] = [852, 884, 984, 1004]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_13_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-15/44_clickable_5.png
try:
    _c44 = get_crop(44, 132, 120)
    canvas.paste(_c44, (60, 1004), _c44)
except Exception:
    pass
layout["5"] = [60, 1004, 192, 1124]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_13_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-15/45_clickable_6.png
try:
    _c45 = get_crop(45, 132, 120)
    canvas.paste(_c45, (192, 1004), _c45)
except Exception:
    pass
layout["6"] = [192, 1004, 324, 1124]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_13_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-15/46_clickable_7.png
try:
    _c46 = get_crop(46, 132, 120)
    canvas.paste(_c46, (324, 1004), _c46)
except Exception:
    pass
layout["7"] = [324, 1004, 456, 1124]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_13_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-15/47_clickable_8.png
try:
    _c47 = get_crop(47, 132, 120)
    canvas.paste(_c47, (456, 1004), _c47)
except Exception:
    pass
layout["8"] = [456, 1004, 588, 1124]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_13_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-15/48_clickable_9.png
try:
    _c48 = get_crop(48, 132, 120)
    canvas.paste(_c48, (588, 1004), _c48)
except Exception:
    pass
layout["9"] = [588, 1004, 720, 1124]
