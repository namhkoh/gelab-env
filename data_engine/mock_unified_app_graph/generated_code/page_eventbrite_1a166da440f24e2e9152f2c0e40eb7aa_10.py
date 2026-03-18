# page_id: page_eventbrite_1a166da440f24e2e9152f2c0e40eb7aa_10
# screenshot: 2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-12.png
# step_index: 10/16
# task: Open Eventbrite. Check "Sports" category. Filter events happening next month. Add the first event to your wishlist.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Canvas and draw are provided (canvas: PIL Image 1440x2960 RGB, draw: ImageDraw)
W, H = canvas.size

# Colors (matching the screenshot's subtle neutral + lavender accent)
STATUS_BG = "#CFCFCF"      # status bar grey
HEADER_DIV = "#EDE8F5"     # subtle purple divider under header
CARD_BORDER = "#E9E0F6"    # light purple border for cards
CARD_ACCENT = "#F6F2FB"    # very light purple fill for small accents
GRID_LINE = "#F4EFF9"      # faint grid/separator lines

# STATUS BAR (top area)
status_h = 72
draw.rectangle([(0, 0), (W, status_h)], fill=STATUS_BG)

# HEADER area (below status) - keep mostly white but add a bottom divider line
header_top = status_h
header_bottom = 168
draw.rectangle([(0, header_top), (W, header_bottom)], fill="white")
draw.line([(32, header_bottom), (W-32, header_bottom)], fill=HEADER_DIV, width=2)

# Large calendar card background (rounded rect) - leaves space for pasted calendar content
cal_x0, cal_y0 = 48, 220
cal_x1, cal_y1 = W - 48, 1520
draw.rounded_rectangle([(cal_x0, cal_y0), (cal_x1, cal_y1)],
                       radius=20, fill="white", outline=CARD_BORDER, width=2)

# Subtle left accent on the calendar card (decorative, not duplicating any icons/text)
accent_w = 36
draw.rounded_rectangle([(cal_x0, cal_y0+8), (cal_x0+accent_w, cal_y0+120)],
                       radius=8, fill=CARD_ACCENT, outline=None)

# Horizontal separators for calendar rows (faint lines)
# Use positions aligned with the calendar grid rows (placed under/between rows)
row_positions = [721, 841, 961, 1081, 1201]  # detected row y's (numbers will be pasted over)
for y in row_positions:
    y_line = y - 40  # place separator between rows
    if cal_y0 + 12 < y_line < cal_y1 - 12:
        draw.line([(cal_x0+16, y_line), (cal_x1-16, y_line)], fill=GRID_LINE, width=2)

# Vertical separators between calendar columns (faint)
col_x = [60, 192, 324, 456, 588, 720, 852]  # detected column x's
# Draw midlines between consecutive columns to act as subtle separators
for i in range(len(col_x)-1):
    x_line = (col_x[i] + col_x[i+1]) // 2
    if cal_x0+16 < x_line < cal_x1-16:
        draw.line([(x_line, cal_y0+16), (x_line, cal_y1-16)], fill=GRID_LINE, width=1)

# Divider separating calendar area from the End Date section
end_div_y = cal_y1 + 40
draw.line([(32, end_div_y), (W-32, end_div_y)], fill=HEADER_DIV, width=2)

# End Date card background area (subtle card for the lower section)
end_x0, end_y0 = 48, end_div_y + 24
end_x1, end_y1 = W - 48, end_y0 + 260
draw.rounded_rectangle([(end_x0, end_y0), (end_x1, end_y1)],
                       radius=16, fill="white", outline=CARD_BORDER, width=2)

# Subtle decorative horizontal rule under the "End Date" area to hint structure
rule_y = end_y1 + 18
draw.line([(end_x0+8, rule_y), (end_x1-8, rule_y)], fill=GRID_LINE, width=1)

# Ensure we do NOT draw anything overlapping the bottom action area where the "Apply date range"
# element will be pasted (detected at y ~2768). So leave the bottom region clear.
# (No drawing commands below y = 2600)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_10_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-12/00_icon_Apply_date_range.png
try:
    _c0 = get_crop(0, 1344, 144)
    canvas.paste(_c0, (48, 2768), _c0)
except Exception:
    pass
layout["Apply_date_range"] = [48, 2768, 1392, 2912]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_10_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-12/01_icon_May.png
try:
    _c1 = get_crop(1, 129, 115)
    canvas.paste(_c1, (194, 610), _c1)
except Exception:
    pass
layout["May"] = [194, 610, 323, 725]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_10_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-12/02_icon_28.png
try:
    _c2 = get_crop(2, 132, 120)
    canvas.paste(_c2, (324, 1201), _c2)
except Exception:
    pass
layout["28"] = [324, 1201, 456, 1321]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_10_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-12/03_icon_26.png
try:
    _c3 = get_crop(3, 132, 120)
    canvas.paste(_c3, (60, 1201), _c3)
except Exception:
    pass
layout["26"] = [60, 1201, 192, 1321]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_10_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-12/04_icon_May.png
try:
    _c4 = get_crop(4, 139, 115)
    canvas.paste(_c4, (321, 608), _c4)
except Exception:
    pass
layout["May"] = [321, 608, 460, 723]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_10_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-12/05_icon_5.31.png
try:
    _c5 = get_crop(5, 62, 66)
    canvas.paste(_c5, (179, 1), _c5)
except Exception:
    pass
layout["5.31"] = [179, 1, 241, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_10_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-12/06_icon_5.31.png
try:
    _c6 = get_crop(6, 64, 68)
    canvas.paste(_c6, (111, 0), _c6)
except Exception:
    pass
layout["5.31"] = [111, 0, 175, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_10_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-12/07_icon_27.png
try:
    _c7 = get_crop(7, 132, 120)
    canvas.paste(_c7, (192, 1201), _c7)
except Exception:
    pass
layout["27"] = [192, 1201, 324, 1321]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_10_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-12/08_icon_May.png
try:
    _c8 = get_crop(8, 131, 116)
    canvas.paste(_c8, (455, 608), _c8)
except Exception:
    pass
layout["May"] = [455, 608, 586, 724]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_10_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-12/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 65, 65)
    canvas.paste(_c9, (308, 2), _c9)
except Exception:
    pass
layout["icon_9"] = [308, 2, 373, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_10_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-12/10_icon_29.png
try:
    _c10 = get_crop(10, 132, 120)
    canvas.paste(_c10, (456, 1201), _c10)
except Exception:
    pass
layout["29"] = [456, 1201, 588, 1321]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_10_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-12/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 53, 64)
    canvas.paste(_c11, (247, 2), _c11)
except Exception:
    pass
layout["icon_11"] = [247, 2, 300, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_10_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-12/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 57, 70)
    canvas.paste(_c12, (1316, 0), _c12)
except Exception:
    pass
layout["icon_12"] = [1316, 0, 1373, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_10_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-12/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 106, 117)
    canvas.paste(_c13, (70, 611), _c13)
except Exception:
    pass
layout["icon_13"] = [70, 611, 176, 728]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_10_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-12/14_icon_May.png
try:
    _c14 = get_crop(14, 144, 144)
    canvas.paste(_c14, (54, 457), _c14)
except Exception:
    pass
layout["May"] = [54, 457, 198, 601]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_10_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-12/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 95, 69)
    canvas.paste(_c15, (1211, 0), _c15)
except Exception:
    pass
layout["icon_15"] = [1211, 0, 1306, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_10_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-12/16_icon_5.31.png
try:
    _c16 = get_crop(16, 144, 144)
    canvas.paste(_c16, (12, 72), _c16)
except Exception:
    pass
layout["5.31"] = [12, 72, 156, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_10_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-12/17_icon_2024.png
try:
    _c17 = get_crop(17, 131, 113)
    canvas.paste(_c17, (590, 609), _c17)
except Exception:
    pass
layout["2024"] = [590, 609, 721, 722]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_10_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-12/18_icon_icon_18.png
try:
    _c18 = get_crop(18, 50, 68)
    canvas.paste(_c18, (382, 1), _c18)
except Exception:
    pass
layout["icon_18"] = [382, 1, 432, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_10_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-12/19_icon_5.31.png
try:
    _c19 = get_crop(19, 93, 65)
    canvas.paste(_c19, (15, 1), _c19)
except Exception:
    pass
layout["5.31"] = [15, 1, 108, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_10_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-12/20_icon_22.png
try:
    _c20 = get_crop(20, 132, 120)
    canvas.paste(_c20, (456, 1081), _c20)
except Exception:
    pass
layout["22"] = [456, 1081, 588, 1201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_10_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-12/21_icon_icon_21.png
try:
    _c21 = get_crop(21, 41, 65)
    canvas.paste(_c21, (1274, 0), _c21)
except Exception:
    pass
layout["icon_21"] = [1274, 0, 1315, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_10_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-12/22_icon_Next_month.png
try:
    _c22 = get_crop(22, 144, 144)
    canvas.paste(_c22, (846, 457), _c22)
except Exception:
    pass
layout["Next_month"] = [846, 457, 990, 601]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_10_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-12/23_icon_23.png
try:
    _c23 = get_crop(23, 132, 120)
    canvas.paste(_c23, (588, 1081), _c23)
except Exception:
    pass
layout["23"] = [588, 1081, 720, 1201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_10_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-12/24_icon_What_date.png
try:
    _c24 = get_crop(24, 319, 70)
    canvas.paste(_c24, (558, 113), _c24)
except Exception:
    pass
layout["What_date?"] = [558, 113, 877, 183]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_10_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-12/25_icon_30.png
try:
    _c25 = get_crop(25, 132, 120)
    canvas.paste(_c25, (588, 1201), _c25)
except Exception:
    pass
layout["30"] = [588, 1201, 720, 1321]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_10_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-12/26_icon_Choose_a_date.png
try:
    _c26 = get_crop(26, 638, 144)
    canvas.paste(_c26, (48, 1490), _c26)
except Exception:
    pass
layout["Choose_a_date"] = [48, 1490, 686, 1634]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_10_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-12/27_text_Start_Date.png
try:
    _c27 = get_crop(27, 589, 114)
    canvas.paste(_c27, (48, 313), _c27)
except Exception:
    pass
layout["Start_Date"] = [48, 313, 637, 427]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_10_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-12/28_text_10.png
try:
    _c28 = get_crop(28, 132, 120)
    canvas.paste(_c28, (720, 841), _c28)
except Exception:
    pass
layout["10"] = [720, 841, 852, 961]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_10_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-12/29_text_11.png
try:
    _c29 = get_crop(29, 132, 120)
    canvas.paste(_c29, (852, 841), _c29)
except Exception:
    pass
layout["11"] = [852, 841, 984, 961]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_10_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-12/30_text_12.png
try:
    _c30 = get_crop(30, 132, 120)
    canvas.paste(_c30, (60, 961), _c30)
except Exception:
    pass
layout["12"] = [60, 961, 192, 1081]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_10_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-12/31_text_13.png
try:
    _c31 = get_crop(31, 132, 120)
    canvas.paste(_c31, (192, 961), _c31)
except Exception:
    pass
layout["13"] = [192, 961, 324, 1081]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_10_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-12/32_text_14.png
try:
    _c32 = get_crop(32, 132, 120)
    canvas.paste(_c32, (324, 961), _c32)
except Exception:
    pass
layout["14"] = [324, 961, 456, 1081]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_10_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-12/33_text_15.png
try:
    _c33 = get_crop(33, 132, 120)
    canvas.paste(_c33, (456, 961), _c33)
except Exception:
    pass
layout["15"] = [456, 961, 588, 1081]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_10_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-12/34_text_16.png
try:
    _c34 = get_crop(34, 132, 120)
    canvas.paste(_c34, (588, 961), _c34)
except Exception:
    pass
layout["16"] = [588, 961, 720, 1081]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_10_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-12/35_text_17.png
try:
    _c35 = get_crop(35, 132, 120)
    canvas.paste(_c35, (720, 961), _c35)
except Exception:
    pass
layout["17"] = [720, 961, 852, 1081]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_10_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-12/36_text_18.png
try:
    _c36 = get_crop(36, 132, 120)
    canvas.paste(_c36, (852, 961), _c36)
except Exception:
    pass
layout["18"] = [852, 961, 984, 1081]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_10_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-12/37_text_19.png
try:
    _c37 = get_crop(37, 132, 120)
    canvas.paste(_c37, (60, 1081), _c37)
except Exception:
    pass
layout["19"] = [60, 1081, 192, 1201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_10_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-12/38_text_20.png
try:
    _c38 = get_crop(38, 132, 120)
    canvas.paste(_c38, (192, 1081), _c38)
except Exception:
    pass
layout["20"] = [192, 1081, 324, 1201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_10_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-12/39_text_21.png
try:
    _c39 = get_crop(39, 132, 120)
    canvas.paste(_c39, (324, 1081), _c39)
except Exception:
    pass
layout["21"] = [324, 1081, 456, 1201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_10_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-12/40_text_24.png
try:
    _c40 = get_crop(40, 132, 120)
    canvas.paste(_c40, (720, 1081), _c40)
except Exception:
    pass
layout["24"] = [720, 1081, 852, 1201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_10_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-12/41_text_25.png
try:
    _c41 = get_crop(41, 132, 120)
    canvas.paste(_c41, (852, 1081), _c41)
except Exception:
    pass
layout["25"] = [852, 1081, 984, 1201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_10_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-12/42_text_31.png
try:
    _c42 = get_crop(42, 132, 120)
    canvas.paste(_c42, (720, 1201), _c42)
except Exception:
    pass
layout["31"] = [720, 1201, 852, 1321]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_10_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-12/43_clickable_1.png
try:
    _c43 = get_crop(43, 132, 120)
    canvas.paste(_c43, (456, 721), _c43)
except Exception:
    pass
layout["1"] = [456, 721, 588, 841]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_10_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-12/44_clickable_2.png
try:
    _c44 = get_crop(44, 132, 120)
    canvas.paste(_c44, (588, 721), _c44)
except Exception:
    pass
layout["2"] = [588, 721, 720, 841]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_10_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-12/45_clickable_3.png
try:
    _c45 = get_crop(45, 132, 120)
    canvas.paste(_c45, (720, 721), _c45)
except Exception:
    pass
layout["3"] = [720, 721, 852, 841]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_10_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-12/46_clickable_4.png
try:
    _c46 = get_crop(46, 132, 120)
    canvas.paste(_c46, (852, 721), _c46)
except Exception:
    pass
layout["4"] = [852, 721, 984, 841]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_10_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-12/47_clickable_5.png
try:
    _c47 = get_crop(47, 132, 120)
    canvas.paste(_c47, (60, 841), _c47)
except Exception:
    pass
layout["5"] = [60, 841, 192, 961]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_10_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-12/48_clickable_6.png
try:
    _c48 = get_crop(48, 132, 120)
    canvas.paste(_c48, (192, 841), _c48)
except Exception:
    pass
layout["6"] = [192, 841, 324, 961]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_10_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-12/49_clickable_7.png
try:
    _c49 = get_crop(49, 132, 120)
    canvas.paste(_c49, (324, 841), _c49)
except Exception:
    pass
layout["7"] = [324, 841, 456, 961]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_10_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-12/50_clickable_8.png
try:
    _c50 = get_crop(50, 132, 120)
    canvas.paste(_c50, (456, 841), _c50)
except Exception:
    pass
layout["8"] = [456, 841, 588, 961]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_10_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-12/51_clickable_9.png
try:
    _c51 = get_crop(51, 132, 120)
    canvas.paste(_c51, (588, 841), _c51)
except Exception:
    pass
layout["9"] = [588, 841, 720, 961]
