# page_id: page_eventbrite_f502e886c78146dfb2f1efc2a331c781_13
# screenshot: 2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-15.png
# step_index: 13/18
# task: Open Eventbrite. Search for 'music festival' in San Francisco. Set date available from April 30 to May 3. How many events are listed?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# draw background and structural elements for the UI page
# assumes variables provided in environment:
# - canvas: PIL Image (1440x2960 RGB)
# - draw: PIL.ImageDraw.Draw(canvas)
# - font_sm, font_md, font_lg, font_xl (not used here)

# Fill overall background (dominant color from screenshot: white)
draw.rectangle((0, 0, 1440, 2960), fill="#FFFFFF")

# Status bar area at top (~72px tall) - light gray background
status_h = 72
draw.rectangle((0, 0, 1440, status_h), fill="#D8D8D8")

# Header / toolbar area below status bar
toolbar_y0 = status_h
toolbar_y1 = 200
draw.rectangle((0, toolbar_y0, 1440, toolbar_y1), fill="#FFFFFF")

# Subtle bottom divider under the toolbar
draw.line((40, toolbar_y1, 1400, toolbar_y1), fill="#E9E4F2", width=1)

# Calendar / Start Date card background (large rounded container)
cal_x0, cal_y0 = 36, 260
cal_x1, cal_y1 = 1404, 1260
draw.rounded_rectangle((cal_x0, cal_y0, cal_x1, cal_y1),
                       radius=28,
                       fill="#FBF9FF",
                       outline="#F1ECF8")

# Month header band inside the calendar card (separates heading from grid)
month_y0 = cal_y0 + 120
month_y1 = month_y0 + 100
draw.rectangle((cal_x0 + 18, month_y0, cal_x1 - 18, month_y1), fill="#FFFFFF")
# subtle divider under month header
draw.line((cal_x0 + 18, month_y1 + 8, cal_x1 - 18, month_y1 + 8), fill="#ECE7F3", width=1)

# Calendar grid background area (slightly different tint)
grid_y0 = month_y1 + 24
grid_y1 = cal_y1 - 36
draw.rectangle((cal_x0 + 18, grid_y0, cal_x1 - 18, grid_y1), fill="#FFFFFF")

# Thin separators for weeks area (suggest structure without drawing numbers)
# Draw faint horizontal guide lines (do not draw numbers)
week_count = 6
week_height = (grid_y1 - grid_y0) / week_count
for i in range(1, week_count):
    y = int(grid_y0 + i * week_height)
    draw.line((cal_x0 + 18, y, cal_x1 - 18, y), fill="#F4F2F7", width=1)

# Vertical column guides for 7-day grid (very faint)
col_count = 7
col_width = (cal_x1 - cal_x0 - 36) / col_count
for i in range(1, col_count):
    x = int(cal_x0 + 18 + i * col_width)
    draw.line((x, grid_y0, x, grid_y1), fill="#F4F2F7", width=1)

# "End Date" section card background (separate rounded container)
end_x0, end_y0 = 36, 1380
end_x1, end_y1 = 1404, 1680
draw.rounded_rectangle((end_x0, end_y0, end_x1, end_y1),
                       radius=20,
                       fill="#FFFFFF",
                       outline="#F0EBF6")

# Subtle divider line below the calendar area separating from "End Date"
sep_y = cal_y1 + 40
draw.line((36, sep_y, 1404, sep_y), fill="#EFEAF0", width=1)

# Large empty content region remains white (no extra drawing) to match screenshot
# Draw a faint top border for the bottom "Apply" area (so button pasted on top has subtle separation)
apply_btn_top = 2728  # based on detected Apply date range at y=2768; a small shadow above it
draw.rectangle((36, apply_btn_top - 12, 1404, apply_btn_top), fill="#F3F0F6")

# Add a very subtle inner shadow bar just above the button for depth
draw.line((36, apply_btn_top - 6, 1404, apply_btn_top - 6), fill="#E6E0EA", width=1)

# Optional: very pale outer frame around the whole content area to match app subtle framing
draw.rounded_rectangle((24, 24, 1416, 2936), radius=8, outline="#FAF8FB", width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_13_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-15/00_icon_23.png
try:
    _c0 = get_crop(0, 132, 120)
    canvas.paste(_c0, (324, 1081), _c0)
except Exception:
    pass
layout["23"] = [324, 1081, 456, 1201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_13_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-15/01_icon_Apply_date_range.png
try:
    _c1 = get_crop(1, 1344, 144)
    canvas.paste(_c1, (48, 2768), _c1)
except Exception:
    pass
layout["Apply_date_range"] = [48, 2768, 1392, 2912]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_13_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-15/02_icon_28.png
try:
    _c2 = get_crop(2, 132, 120)
    canvas.paste(_c2, (60, 1201), _c2)
except Exception:
    pass
layout["28"] = [60, 1201, 192, 1321]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_13_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-15/03_icon_24.png
try:
    _c3 = get_crop(3, 132, 120)
    canvas.paste(_c3, (456, 1081), _c3)
except Exception:
    pass
layout["24"] = [456, 1081, 588, 1201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_13_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-15/04_icon_29.png
try:
    _c4 = get_crop(4, 132, 120)
    canvas.paste(_c4, (192, 1201), _c4)
except Exception:
    pass
layout["29"] = [192, 1201, 324, 1321]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_13_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-15/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 52, 71)
    canvas.paste(_c5, (1153, 0), _c5)
except Exception:
    pass
layout["icon_5"] = [1153, 0, 1205, 71]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_13_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-15/06_icon_22.png
try:
    _c6 = get_crop(6, 132, 120)
    canvas.paste(_c6, (192, 1081), _c6)
except Exception:
    pass
layout["22"] = [192, 1081, 324, 1201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_13_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-15/07_icon_30.png
try:
    _c7 = get_crop(7, 132, 120)
    canvas.paste(_c7, (324, 1201), _c7)
except Exception:
    pass
layout["30"] = [324, 1201, 456, 1321]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_13_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-15/08_icon_7.19.png
try:
    _c8 = get_crop(8, 61, 65)
    canvas.paste(_c8, (180, 0), _c8)
except Exception:
    pass
layout["7.19"] = [180, 0, 241, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_13_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-15/09_icon_25.png
try:
    _c9 = get_crop(9, 132, 120)
    canvas.paste(_c9, (588, 1081), _c9)
except Exception:
    pass
layout["25"] = [588, 1081, 720, 1201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_13_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-15/10_icon_26.png
try:
    _c10 = get_crop(10, 132, 120)
    canvas.paste(_c10, (720, 1081), _c10)
except Exception:
    pass
layout["26"] = [720, 1081, 852, 1201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_13_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-15/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 100, 70)
    canvas.paste(_c11, (1210, 0), _c11)
except Exception:
    pass
layout["icon_11"] = [1210, 0, 1310, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_13_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-15/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 64, 63)
    canvas.paste(_c12, (309, 3), _c12)
except Exception:
    pass
layout["icon_12"] = [309, 3, 373, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_13_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-15/13_icon_7.19.png
try:
    _c13 = get_crop(13, 60, 66)
    canvas.paste(_c13, (115, 0), _c13)
except Exception:
    pass
layout["7.19"] = [115, 0, 175, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_13_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-15/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 54, 70)
    canvas.paste(_c14, (1318, 0), _c14)
except Exception:
    pass
layout["icon_14"] = [1318, 0, 1372, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_13_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-15/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 53, 63)
    canvas.paste(_c15, (247, 2), _c15)
except Exception:
    pass
layout["icon_15"] = [247, 2, 300, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_13_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-15/16_icon_27.png
try:
    _c16 = get_crop(16, 132, 120)
    canvas.paste(_c16, (852, 1081), _c16)
except Exception:
    pass
layout["27"] = [852, 1081, 984, 1201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_13_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-15/17_icon_21.png
try:
    _c17 = get_crop(17, 132, 120)
    canvas.paste(_c17, (60, 1081), _c17)
except Exception:
    pass
layout["21"] = [60, 1081, 192, 1201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_13_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-15/18_icon_7.19.png
try:
    _c18 = get_crop(18, 144, 144)
    canvas.paste(_c18, (12, 72), _c18)
except Exception:
    pass
layout["7.19"] = [12, 72, 156, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_13_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-15/19_icon_11.png
try:
    _c19 = get_crop(19, 132, 120)
    canvas.paste(_c19, (588, 721), _c19)
except Exception:
    pass
layout["11"] = [588, 721, 720, 841]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_13_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-15/20_icon_April_2024.png
try:
    _c20 = get_crop(20, 126, 110)
    canvas.paste(_c20, (593, 611), _c20)
except Exception:
    pass
layout["April_2024"] = [593, 611, 719, 721]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_13_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-15/21_icon_icon_21.png
try:
    _c21 = get_crop(21, 49, 67)
    canvas.paste(_c21, (382, 1), _c21)
except Exception:
    pass
layout["icon_21"] = [382, 1, 431, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_13_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-15/22_icon_Next_month.png
try:
    _c22 = get_crop(22, 144, 144)
    canvas.paste(_c22, (846, 457), _c22)
except Exception:
    pass
layout["Next_month"] = [846, 457, 990, 601]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_13_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-15/23_icon_18.png
try:
    _c23 = get_crop(23, 132, 120)
    canvas.paste(_c23, (588, 961), _c23)
except Exception:
    pass
layout["18"] = [588, 961, 720, 1081]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_13_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-15/24_icon_Choose_a_date.png
try:
    _c24 = get_crop(24, 638, 144)
    canvas.paste(_c24, (48, 1490), _c24)
except Exception:
    pass
layout["Choose_a_date"] = [48, 1490, 686, 1634]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_13_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-15/25_icon_12.png
try:
    _c25 = get_crop(25, 132, 120)
    canvas.paste(_c25, (720, 721), _c25)
except Exception:
    pass
layout["12"] = [720, 721, 852, 841]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_13_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-15/26_icon_12.png
try:
    _c26 = get_crop(26, 102, 106)
    canvas.paste(_c26, (735, 615), _c26)
except Exception:
    pass
layout["12"] = [735, 615, 837, 721]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_13_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-15/27_icon_19.png
try:
    _c27 = get_crop(27, 132, 120)
    canvas.paste(_c27, (720, 961), _c27)
except Exception:
    pass
layout["19"] = [720, 961, 852, 1081]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_13_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-15/28_text_7.19.png
try:
    _c28 = get_crop(28, 91, 45)
    canvas.paste(_c28, (20, 15), _c28)
except Exception:
    pass
layout["7.19"] = [20, 15, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_13_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-15/29_text_What_date.png
try:
    _c29 = get_crop(29, 318, 63)
    canvas.paste(_c29, (563, 117), _c29)
except Exception:
    pass
layout["What_date?"] = [563, 117, 881, 180]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_13_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-15/30_text_Start_Date.png
try:
    _c30 = get_crop(30, 580, 114)
    canvas.paste(_c30, (48, 313), _c30)
except Exception:
    pass
layout["Start_Date"] = [48, 313, 628, 427]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_13_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-15/31_text_April_2024.png
try:
    _c31 = get_crop(31, 203, 54)
    canvas.paste(_c31, (420, 504), _c31)
except Exception:
    pass
layout["April_2024"] = [420, 504, 623, 558]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_13_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-15/32_text_10.png
try:
    _c32 = get_crop(32, 132, 120)
    canvas.paste(_c32, (456, 841), _c32)
except Exception:
    pass
layout["10"] = [456, 841, 588, 961]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_13_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-15/33_text_11.png
try:
    _c33 = get_crop(33, 132, 120)
    canvas.paste(_c33, (588, 841), _c33)
except Exception:
    pass
layout["11"] = [588, 841, 720, 961]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_13_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-15/34_text_12.png
try:
    _c34 = get_crop(34, 132, 120)
    canvas.paste(_c34, (720, 841), _c34)
except Exception:
    pass
layout["12"] = [720, 841, 852, 961]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_13_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-15/35_text_13.png
try:
    _c35 = get_crop(35, 132, 120)
    canvas.paste(_c35, (852, 841), _c35)
except Exception:
    pass
layout["13"] = [852, 841, 984, 961]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_13_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-15/36_text_14.png
try:
    _c36 = get_crop(36, 132, 120)
    canvas.paste(_c36, (60, 961), _c36)
except Exception:
    pass
layout["14"] = [60, 961, 192, 1081]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_13_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-15/37_text_15.png
try:
    _c37 = get_crop(37, 132, 120)
    canvas.paste(_c37, (192, 961), _c37)
except Exception:
    pass
layout["15"] = [192, 961, 324, 1081]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_13_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-15/38_text_16.png
try:
    _c38 = get_crop(38, 132, 120)
    canvas.paste(_c38, (324, 961), _c38)
except Exception:
    pass
layout["16"] = [324, 961, 456, 1081]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_13_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-15/39_text_17.png
try:
    _c39 = get_crop(39, 132, 120)
    canvas.paste(_c39, (456, 961), _c39)
except Exception:
    pass
layout["17"] = [456, 961, 588, 1081]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_13_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-15/40_text_20.png
try:
    _c40 = get_crop(40, 132, 120)
    canvas.paste(_c40, (852, 961), _c40)
except Exception:
    pass
layout["20"] = [852, 961, 984, 1081]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_13_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-15/41_clickable_1.png
try:
    _c41 = get_crop(41, 132, 120)
    canvas.paste(_c41, (192, 721), _c41)
except Exception:
    pass
layout["1"] = [192, 721, 324, 841]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_13_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-15/42_clickable_2.png
try:
    _c42 = get_crop(42, 132, 120)
    canvas.paste(_c42, (324, 721), _c42)
except Exception:
    pass
layout["2"] = [324, 721, 456, 841]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_13_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-15/43_clickable_3.png
try:
    _c43 = get_crop(43, 132, 120)
    canvas.paste(_c43, (456, 721), _c43)
except Exception:
    pass
layout["3"] = [456, 721, 588, 841]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_13_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-15/44_clickable_6.png
try:
    _c44 = get_crop(44, 132, 120)
    canvas.paste(_c44, (852, 721), _c44)
except Exception:
    pass
layout["6"] = [852, 721, 984, 841]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_13_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-15/45_clickable_7.png
try:
    _c45 = get_crop(45, 132, 120)
    canvas.paste(_c45, (60, 841), _c45)
except Exception:
    pass
layout["7"] = [60, 841, 192, 961]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_13_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-15/46_clickable_8.png
try:
    _c46 = get_crop(46, 132, 120)
    canvas.paste(_c46, (192, 841), _c46)
except Exception:
    pass
layout["8"] = [192, 841, 324, 961]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_13_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-15/47_clickable_9.png
try:
    _c47 = get_crop(47, 132, 120)
    canvas.paste(_c47, (324, 841), _c47)
except Exception:
    pass
layout["9"] = [324, 841, 456, 961]
