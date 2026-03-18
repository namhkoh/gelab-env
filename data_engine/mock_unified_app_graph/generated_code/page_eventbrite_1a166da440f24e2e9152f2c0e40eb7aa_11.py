# page_id: page_eventbrite_1a166da440f24e2e9152f2c0e40eb7aa_11
# screenshot: 2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-13.png
# step_index: 11/16
# task: Open Eventbrite. Check "Sports" category. Filter events happening next month. Add the first event to your wishlist.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Background / structural drawing for the mobile UI (calendar/date picker)
# Uses provided canvas (1440x2960) and draw (ImageDraw)

# Colors
bg_white = (255, 255, 255)
status_gray = (214, 214, 214)       # status bar background
muted_divider = (220, 215, 227)     # very light purple/gray divider
card_border = (235, 233, 240)       # subtle card outline
card_fill = (255, 255, 255)         # cards keep white
section_bg = (250, 249, 252)        # faint section background
accent_light = (244, 240, 250)

W, H = canvas.size

# Fill entire background (canvas may already be white, but ensure consistent tone)
draw.rectangle((0, 0, W, H), fill=bg_white)

# Status bar area (top)
status_h = 72
draw.rectangle((0, 0, W, status_h), fill=status_gray)

# Subtle bottom border under status bar
draw.line((0, status_h - 1, W, status_h - 1), fill=muted_divider, width=1)

# Header area: keep transparent/white but add a subtle divider under header
header_top = status_h
header_bottom = 180
# Slightly different tint to separate from content (very subtle)
draw.rectangle((0, header_top, W, header_bottom), fill=bg_white)
draw.line((48, header_bottom, W - 48, header_bottom), fill=muted_divider, width=1)

# Main calendar container card (rounded)
cal_left = 48
cal_right = W - 48
cal_top = 220
cal_bottom = 1380
cal_radius = 14
# light shadow simulated by a faint offset tinted rectangle behind (very subtle)
shadow_color = (245, 244, 247)
draw.rounded_rectangle((cal_left + 6, cal_top + 8, cal_right + 6, cal_bottom + 8),
                       radius=cal_radius + 2, fill=shadow_color, outline=None)
# actual card
draw.rounded_rectangle((cal_left, cal_top, cal_right, cal_bottom),
                       radius=cal_radius, fill=card_fill, outline=card_border, width=1)

# Month header area inside calendar (no icons/text, only background divider)
month_header_top = cal_top + 30
month_header_bottom = month_header_top + 80
draw.rectangle((cal_left + 24, month_header_top, cal_right - 24, month_header_bottom),
               fill=card_fill, outline=None)
# faint separator under month header
draw.line((cal_left + 24, month_header_bottom + 6, cal_right - 24, month_header_bottom + 6),
          fill=muted_divider, width=1)

# Weekdays row background (very subtle)
week_row_top = month_header_bottom + 28
week_row_bottom = week_row_top + 36
draw.rectangle((cal_left + 24, week_row_top, cal_right - 24, week_row_bottom),
               fill=accent_light, outline=None)

# Calendar grid background area (keeps white but with a faint grid guide using very light lines)
grid_top = week_row_bottom + 12
grid_bottom = cal_bottom - 24
grid_left = cal_left + 24
grid_right = cal_right - 24
# Draw faint horizontal separators for weeks (no text/numbers)
num_weeks = 6
week_h = (grid_bottom - grid_top) / num_weeks
for i in range(1, num_weeks):
    y = int(grid_top + i * week_h)
    draw.line((grid_left, y, grid_right, y), fill=card_border, width=1)

# Draw faint vertical guides for 7 day columns (subtle, not strong)
num_cols = 7
col_w = (grid_right - grid_left) / num_cols
for i in range(1, num_cols):
    x = int(grid_left + i * col_w)
    draw.line((x, grid_top, x, grid_bottom), fill=card_border, width=1)

# End Date section area (card-like background)
end_section_top = cal_bottom + 60
end_section_left = cal_left
end_section_right = cal_right
end_section_bottom = end_section_top + 220
end_radius = 12
draw.rounded_rectangle((end_section_left, end_section_top, end_section_right, end_section_bottom),
                       radius=end_radius, fill=section_bg, outline=card_border, width=1)

# Separator line between calendar and end-date area
sep_y = cal_bottom + 36
draw.line((cal_left, sep_y, cal_right, sep_y), fill=muted_divider, width=1)

# Large empty content area (below end-date) - keep background consistent
content_top = end_section_bottom + 20
draw.rectangle((0, content_top, W, H - 200), fill=bg_white)

# Top of bottom area: a faint separator to visually separate main content from bottom control zone
bottom_control_top = H - 220
draw.line((48, bottom_control_top, W - 48, bottom_control_top), fill=muted_divider, width=1)

# Note: The actual "Apply date range" button and all icons/text will be pasted on top at exact positions.
# This code intentionally draws only background, cards, dividers, and general structure.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_11_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-13/00_icon_Apply_date_range.png
try:
    _c0 = get_crop(0, 1344, 144)
    canvas.paste(_c0, (48, 2768), _c0)
except Exception:
    pass
layout["Apply_date_range"] = [48, 2768, 1392, 2912]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_11_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-13/01_icon_5.31.png
try:
    _c1 = get_crop(1, 62, 66)
    canvas.paste(_c1, (179, 1), _c1)
except Exception:
    pass
layout["5.31"] = [179, 1, 241, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_11_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-13/02_icon_28.png
try:
    _c2 = get_crop(2, 132, 120)
    canvas.paste(_c2, (324, 1201), _c2)
except Exception:
    pass
layout["28"] = [324, 1201, 456, 1321]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_11_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-13/03_icon_5.31.png
try:
    _c3 = get_crop(3, 66, 68)
    canvas.paste(_c3, (110, 0), _c3)
except Exception:
    pass
layout["5.31"] = [110, 0, 176, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_11_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-13/04_icon_icon_4.png
try:
    _c4 = get_crop(4, 65, 65)
    canvas.paste(_c4, (308, 2), _c4)
except Exception:
    pass
layout["icon_4"] = [308, 2, 373, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_11_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-13/05_icon_26.png
try:
    _c5 = get_crop(5, 132, 120)
    canvas.paste(_c5, (60, 1201), _c5)
except Exception:
    pass
layout["26"] = [60, 1201, 192, 1321]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_11_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-13/06_icon_27.png
try:
    _c6 = get_crop(6, 132, 120)
    canvas.paste(_c6, (192, 1201), _c6)
except Exception:
    pass
layout["27"] = [192, 1201, 324, 1321]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_11_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-13/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 53, 65)
    canvas.paste(_c7, (247, 1), _c7)
except Exception:
    pass
layout["icon_7"] = [247, 1, 300, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_11_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-13/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 57, 70)
    canvas.paste(_c8, (1316, 0), _c8)
except Exception:
    pass
layout["icon_8"] = [1316, 0, 1373, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_11_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-13/09_icon_5.31.png
try:
    _c9 = get_crop(9, 144, 144)
    canvas.paste(_c9, (12, 72), _c9)
except Exception:
    pass
layout["5.31"] = [12, 72, 156, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_11_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-13/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 97, 70)
    canvas.paste(_c10, (1211, 0), _c10)
except Exception:
    pass
layout["icon_10"] = [1211, 0, 1308, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_11_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-13/11_icon_May.png
try:
    _c11 = get_crop(11, 144, 144)
    canvas.paste(_c11, (54, 457), _c11)
except Exception:
    pass
layout["May"] = [54, 457, 198, 601]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_11_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-13/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 93, 105)
    canvas.paste(_c12, (76, 617), _c12)
except Exception:
    pass
layout["icon_12"] = [76, 617, 169, 722]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_11_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-13/13_icon_29.png
try:
    _c13 = get_crop(13, 132, 120)
    canvas.paste(_c13, (456, 1201), _c13)
except Exception:
    pass
layout["29"] = [456, 1201, 588, 1321]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_11_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-13/14_icon_5.31.png
try:
    _c14 = get_crop(14, 94, 65)
    canvas.paste(_c14, (14, 1), _c14)
except Exception:
    pass
layout["5.31"] = [14, 1, 108, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_11_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-13/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 50, 68)
    canvas.paste(_c15, (382, 1), _c15)
except Exception:
    pass
layout["icon_15"] = [382, 1, 432, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_11_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-13/16_icon_15.png
try:
    _c16 = get_crop(16, 132, 120)
    canvas.paste(_c16, (456, 841), _c16)
except Exception:
    pass
layout["15"] = [456, 841, 588, 961]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_11_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-13/17_icon_What_date.png
try:
    _c17 = get_crop(17, 321, 71)
    canvas.paste(_c17, (558, 113), _c17)
except Exception:
    pass
layout["What_date?"] = [558, 113, 879, 184]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_11_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-13/18_icon_May.png
try:
    _c18 = get_crop(18, 114, 93)
    canvas.paste(_c18, (462, 619), _c18)
except Exception:
    pass
layout["May"] = [462, 619, 576, 712]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_11_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-13/19_icon_Next_month.png
try:
    _c19 = get_crop(19, 144, 144)
    canvas.paste(_c19, (846, 457), _c19)
except Exception:
    pass
layout["Next_month"] = [846, 457, 990, 601]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_11_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-13/20_icon_icon_20.png
try:
    _c20 = get_crop(20, 43, 58)
    canvas.paste(_c20, (1273, 4), _c20)
except Exception:
    pass
layout["icon_20"] = [1273, 4, 1316, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_11_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-13/21_icon_Choose_a_date.png
try:
    _c21 = get_crop(21, 638, 144)
    canvas.paste(_c21, (48, 1490), _c21)
except Exception:
    pass
layout["Choose_a_date"] = [48, 1490, 686, 1634]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_11_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-13/22_icon_2024.png
try:
    _c22 = get_crop(22, 76, 89)
    canvas.paste(_c22, (615, 620), _c22)
except Exception:
    pass
layout["2024"] = [615, 620, 691, 709]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_11_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-13/23_icon_May.png
try:
    _c23 = get_crop(23, 103, 92)
    canvas.paste(_c23, (338, 619), _c23)
except Exception:
    pass
layout["May"] = [338, 619, 441, 711]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_11_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-13/24_text_Start_Date.png
try:
    _c24 = get_crop(24, 591, 114)
    canvas.paste(_c24, (48, 313), _c24)
except Exception:
    pass
layout["Start_Date"] = [48, 313, 639, 427]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_11_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-13/25_text_10.png
try:
    _c25 = get_crop(25, 132, 120)
    canvas.paste(_c25, (720, 841), _c25)
except Exception:
    pass
layout["10"] = [720, 841, 852, 961]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_11_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-13/26_text_11.png
try:
    _c26 = get_crop(26, 132, 120)
    canvas.paste(_c26, (852, 841), _c26)
except Exception:
    pass
layout["11"] = [852, 841, 984, 961]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_11_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-13/27_text_12.png
try:
    _c27 = get_crop(27, 132, 120)
    canvas.paste(_c27, (60, 961), _c27)
except Exception:
    pass
layout["12"] = [60, 961, 192, 1081]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_11_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-13/28_text_13.png
try:
    _c28 = get_crop(28, 132, 120)
    canvas.paste(_c28, (192, 961), _c28)
except Exception:
    pass
layout["13"] = [192, 961, 324, 1081]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_11_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-13/29_text_14.png
try:
    _c29 = get_crop(29, 132, 120)
    canvas.paste(_c29, (324, 961), _c29)
except Exception:
    pass
layout["14"] = [324, 961, 456, 1081]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_11_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-13/30_text_15.png
try:
    _c30 = get_crop(30, 132, 120)
    canvas.paste(_c30, (456, 961), _c30)
except Exception:
    pass
layout["15"] = [456, 961, 588, 1081]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_11_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-13/31_text_16.png
try:
    _c31 = get_crop(31, 132, 120)
    canvas.paste(_c31, (588, 961), _c31)
except Exception:
    pass
layout["16"] = [588, 961, 720, 1081]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_11_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-13/32_text_17.png
try:
    _c32 = get_crop(32, 132, 120)
    canvas.paste(_c32, (720, 961), _c32)
except Exception:
    pass
layout["17"] = [720, 961, 852, 1081]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_11_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-13/33_text_18.png
try:
    _c33 = get_crop(33, 132, 120)
    canvas.paste(_c33, (852, 961), _c33)
except Exception:
    pass
layout["18"] = [852, 961, 984, 1081]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_11_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-13/34_text_19.png
try:
    _c34 = get_crop(34, 132, 120)
    canvas.paste(_c34, (60, 1081), _c34)
except Exception:
    pass
layout["19"] = [60, 1081, 192, 1201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_11_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-13/35_text_20.png
try:
    _c35 = get_crop(35, 132, 120)
    canvas.paste(_c35, (192, 1081), _c35)
except Exception:
    pass
layout["20"] = [192, 1081, 324, 1201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_11_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-13/36_text_21.png
try:
    _c36 = get_crop(36, 132, 120)
    canvas.paste(_c36, (324, 1081), _c36)
except Exception:
    pass
layout["21"] = [324, 1081, 456, 1201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_11_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-13/37_text_22.png
try:
    _c37 = get_crop(37, 132, 120)
    canvas.paste(_c37, (456, 1081), _c37)
except Exception:
    pass
layout["22"] = [456, 1081, 588, 1201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_11_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-13/38_text_23.png
try:
    _c38 = get_crop(38, 132, 120)
    canvas.paste(_c38, (588, 1081), _c38)
except Exception:
    pass
layout["23"] = [588, 1081, 720, 1201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_11_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-13/39_text_24.png
try:
    _c39 = get_crop(39, 132, 120)
    canvas.paste(_c39, (720, 1081), _c39)
except Exception:
    pass
layout["24"] = [720, 1081, 852, 1201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_11_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-13/40_text_25.png
try:
    _c40 = get_crop(40, 132, 120)
    canvas.paste(_c40, (852, 1081), _c40)
except Exception:
    pass
layout["25"] = [852, 1081, 984, 1201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_11_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-13/41_text_30.png
try:
    _c41 = get_crop(41, 132, 120)
    canvas.paste(_c41, (588, 1201), _c41)
except Exception:
    pass
layout["30"] = [588, 1201, 720, 1321]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_11_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-13/42_text_31.png
try:
    _c42 = get_crop(42, 132, 120)
    canvas.paste(_c42, (720, 1201), _c42)
except Exception:
    pass
layout["31"] = [720, 1201, 852, 1321]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_11_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-13/43_clickable_1.png
try:
    _c43 = get_crop(43, 132, 120)
    canvas.paste(_c43, (456, 721), _c43)
except Exception:
    pass
layout["1"] = [456, 721, 588, 841]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_11_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-13/44_clickable_2.png
try:
    _c44 = get_crop(44, 132, 120)
    canvas.paste(_c44, (588, 721), _c44)
except Exception:
    pass
layout["2"] = [588, 721, 720, 841]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_11_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-13/45_clickable_3.png
try:
    _c45 = get_crop(45, 132, 120)
    canvas.paste(_c45, (720, 721), _c45)
except Exception:
    pass
layout["3"] = [720, 721, 852, 841]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_11_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-13/46_clickable_4.png
try:
    _c46 = get_crop(46, 132, 120)
    canvas.paste(_c46, (852, 721), _c46)
except Exception:
    pass
layout["4"] = [852, 721, 984, 841]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_11_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-13/47_clickable_5.png
try:
    _c47 = get_crop(47, 132, 120)
    canvas.paste(_c47, (60, 841), _c47)
except Exception:
    pass
layout["5"] = [60, 841, 192, 961]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_11_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-13/48_clickable_6.png
try:
    _c48 = get_crop(48, 132, 120)
    canvas.paste(_c48, (192, 841), _c48)
except Exception:
    pass
layout["6"] = [192, 841, 324, 961]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_11_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-13/49_clickable_7.png
try:
    _c49 = get_crop(49, 132, 120)
    canvas.paste(_c49, (324, 841), _c49)
except Exception:
    pass
layout["7"] = [324, 841, 456, 961]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_11_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-13/50_clickable_9.png
try:
    _c50 = get_crop(50, 132, 120)
    canvas.paste(_c50, (588, 841), _c50)
except Exception:
    pass
layout["9"] = [588, 841, 720, 961]
