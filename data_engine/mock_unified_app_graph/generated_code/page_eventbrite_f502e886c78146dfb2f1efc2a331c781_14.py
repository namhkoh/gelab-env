# page_id: page_eventbrite_f502e886c78146dfb2f1efc2a331c781_14
# screenshot: 2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-16.png
# step_index: 14/18
# task: Open Eventbrite. Search for 'music festival' in San Francisco. Set date available from April 30 to May 3. How many events are listed?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Background base
draw.rectangle([(0, 0), (1440, 2960)], fill="#ffffff")

# Status bar area (top)
status_h = 72
draw.rectangle([(0, 0), (1440, status_h)], fill="#d0d0d0")
# subtle bottom edge for status bar
draw.line([(0, status_h), (1440, status_h)], fill="#bfbfbf", width=1)

# Header area separator (under the title area)
header_h = 180
draw.rectangle([(0, status_h), (1440, header_h)], fill="#ffffff")
draw.line([(48, header_h), (1392, header_h)], fill="#f0edf6", width=1)

# Calendar section card (soft rounded panel behind calendar grid)
cal_x0, cal_x1 = 48, 1392
cal_y0, cal_y1 = 220, 1360
draw.rounded_rectangle(
    [(cal_x0, cal_y0), (cal_x1, cal_y1)],
    radius=28,
    fill="#fbfbfe",
    outline="#e9e6f2",
    width=1
)

# Subtle inner highlight on the calendar card (top edge)
draw.line([(cal_x0+8, cal_y0+6), (cal_x1-8, cal_y0+6)], fill="#f6f5fb", width=1)

# Divider between calendar region and the "End Date" area
sep_y = 1480
draw.line([(48, sep_y), (1392, sep_y)], fill="#f1eef4", width=1)

# End Date area background hint (very light band to group area)
end_y0, end_y1 = 1488, 1880
draw.rectangle([(48, end_y0), (1392, end_y1)], fill="#ffffff")

# Subtle vertical guides for calendar columns (very faint)
col_w = 132 + 60  # approximate spacing between date columns
for i in range(1, 6):
    x = 48 + i * col_w
    if 120 < x < 1392:
        draw.line([(x, cal_y0+40), (x, cal_y1-24)], fill="#fbfbfc", width=1)

# Top-of-bottom-action area separator (above the Apply button region)
apply_top = 2768
# subtle shadow band above the button area
draw.rectangle([(0, apply_top-14), (1440, apply_top)], fill="#f6f6f8")
# faint divider line at top of the action area
draw.line([(48, apply_top), (1392, apply_top)], fill="#e6e3ea", width=1)

# Page left/right padding guides (very faint) to visually group content (non-intrusive)
pad_x = 48
draw.line([(pad_x, header_h+8), (pad_x, 2500)], fill="#ffffff", width=1)
draw.line([(1440-pad_x, header_h+8), (1440-pad_x, 2500)], fill="#ffffff", width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_14_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-16/00_icon_30.png
try:
    _c0 = get_crop(0, 132, 120)
    canvas.paste(_c0, (324, 1201), _c0)
except Exception:
    pass
layout["30"] = [324, 1201, 456, 1321]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_14_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-16/01_icon_Apply_date_range.png
try:
    _c1 = get_crop(1, 1344, 144)
    canvas.paste(_c1, (48, 2768), _c1)
except Exception:
    pass
layout["Apply_date_range"] = [48, 2768, 1392, 2912]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_14_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-16/02_icon_29.png
try:
    _c2 = get_crop(2, 132, 120)
    canvas.paste(_c2, (192, 1201), _c2)
except Exception:
    pass
layout["29"] = [192, 1201, 324, 1321]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_14_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-16/03_icon_23.png
try:
    _c3 = get_crop(3, 132, 120)
    canvas.paste(_c3, (324, 1081), _c3)
except Exception:
    pass
layout["23"] = [324, 1081, 456, 1201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_14_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-16/04_icon_24.png
try:
    _c4 = get_crop(4, 132, 120)
    canvas.paste(_c4, (456, 1081), _c4)
except Exception:
    pass
layout["24"] = [456, 1081, 588, 1201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_14_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-16/05_icon_25.png
try:
    _c5 = get_crop(5, 132, 120)
    canvas.paste(_c5, (588, 1081), _c5)
except Exception:
    pass
layout["25"] = [588, 1081, 720, 1201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_14_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-16/06_icon_28.png
try:
    _c6 = get_crop(6, 132, 120)
    canvas.paste(_c6, (60, 1201), _c6)
except Exception:
    pass
layout["28"] = [60, 1201, 192, 1321]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_14_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-16/07_icon_22.png
try:
    _c7 = get_crop(7, 132, 120)
    canvas.paste(_c7, (192, 1081), _c7)
except Exception:
    pass
layout["22"] = [192, 1081, 324, 1201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_14_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-16/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 52, 71)
    canvas.paste(_c8, (1153, 0), _c8)
except Exception:
    pass
layout["icon_8"] = [1153, 0, 1205, 71]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_14_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-16/09_icon_26.png
try:
    _c9 = get_crop(9, 132, 120)
    canvas.paste(_c9, (720, 1081), _c9)
except Exception:
    pass
layout["26"] = [720, 1081, 852, 1201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_14_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-16/10_icon_7.19.png
try:
    _c10 = get_crop(10, 61, 65)
    canvas.paste(_c10, (180, 0), _c10)
except Exception:
    pass
layout["7.19"] = [180, 0, 241, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_14_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-16/11_icon_21.png
try:
    _c11 = get_crop(11, 132, 120)
    canvas.paste(_c11, (60, 1081), _c11)
except Exception:
    pass
layout["21"] = [60, 1081, 192, 1201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_14_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-16/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 100, 70)
    canvas.paste(_c12, (1210, 0), _c12)
except Exception:
    pass
layout["icon_12"] = [1210, 0, 1310, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_14_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-16/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 63, 62)
    canvas.paste(_c13, (309, 3), _c13)
except Exception:
    pass
layout["icon_13"] = [309, 3, 372, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_14_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-16/14_icon_7.19.png
try:
    _c14 = get_crop(14, 61, 67)
    canvas.paste(_c14, (114, 0), _c14)
except Exception:
    pass
layout["7.19"] = [114, 0, 175, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_14_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-16/15_icon_27.png
try:
    _c15 = get_crop(15, 132, 120)
    canvas.paste(_c15, (852, 1081), _c15)
except Exception:
    pass
layout["27"] = [852, 1081, 984, 1201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_14_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-16/16_icon_icon_16.png
try:
    _c16 = get_crop(16, 54, 70)
    canvas.paste(_c16, (1318, 0), _c16)
except Exception:
    pass
layout["icon_16"] = [1318, 0, 1372, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_14_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-16/17_icon_icon_17.png
try:
    _c17 = get_crop(17, 52, 63)
    canvas.paste(_c17, (248, 2), _c17)
except Exception:
    pass
layout["icon_17"] = [248, 2, 300, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_14_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-16/18_icon_7.19.png
try:
    _c18 = get_crop(18, 144, 144)
    canvas.paste(_c18, (12, 72), _c18)
except Exception:
    pass
layout["7.19"] = [12, 72, 156, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_14_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-16/19_icon_11.png
try:
    _c19 = get_crop(19, 132, 120)
    canvas.paste(_c19, (588, 721), _c19)
except Exception:
    pass
layout["11"] = [588, 721, 720, 841]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_14_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-16/20_icon_icon_20.png
try:
    _c20 = get_crop(20, 49, 67)
    canvas.paste(_c20, (382, 1), _c20)
except Exception:
    pass
layout["icon_20"] = [382, 1, 431, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_14_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-16/21_icon_Next_month.png
try:
    _c21 = get_crop(21, 144, 144)
    canvas.paste(_c21, (846, 457), _c21)
except Exception:
    pass
layout["Next_month"] = [846, 457, 990, 601]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_14_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-16/22_icon_Choose_a_date.png
try:
    _c22 = get_crop(22, 638, 144)
    canvas.paste(_c22, (48, 1490), _c22)
except Exception:
    pass
layout["Choose_a_date"] = [48, 1490, 686, 1634]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_14_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-16/23_icon_18.png
try:
    _c23 = get_crop(23, 132, 120)
    canvas.paste(_c23, (588, 961), _c23)
except Exception:
    pass
layout["18"] = [588, 961, 720, 1081]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_14_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-16/24_icon_What_date.png
try:
    _c24 = get_crop(24, 321, 71)
    canvas.paste(_c24, (558, 113), _c24)
except Exception:
    pass
layout["What_date?"] = [558, 113, 879, 184]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_14_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-16/25_icon_19.png
try:
    _c25 = get_crop(25, 132, 120)
    canvas.paste(_c25, (720, 961), _c25)
except Exception:
    pass
layout["19"] = [720, 961, 852, 1081]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_14_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-16/26_text_7.19.png
try:
    _c26 = get_crop(26, 91, 45)
    canvas.paste(_c26, (20, 15), _c26)
except Exception:
    pass
layout["7.19"] = [20, 15, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_14_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-16/27_text_Start_Date.png
try:
    _c27 = get_crop(27, 587, 114)
    canvas.paste(_c27, (48, 313), _c27)
except Exception:
    pass
layout["Start_Date"] = [48, 313, 635, 427]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_14_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-16/28_text_April_2024.png
try:
    _c28 = get_crop(28, 203, 54)
    canvas.paste(_c28, (420, 504), _c28)
except Exception:
    pass
layout["April_2024"] = [420, 504, 623, 558]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_14_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-16/29_text_10.png
try:
    _c29 = get_crop(29, 132, 120)
    canvas.paste(_c29, (456, 841), _c29)
except Exception:
    pass
layout["10"] = [456, 841, 588, 961]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_14_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-16/30_text_11.png
try:
    _c30 = get_crop(30, 132, 120)
    canvas.paste(_c30, (588, 841), _c30)
except Exception:
    pass
layout["11"] = [588, 841, 720, 961]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_14_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-16/31_text_12.png
try:
    _c31 = get_crop(31, 132, 120)
    canvas.paste(_c31, (720, 841), _c31)
except Exception:
    pass
layout["12"] = [720, 841, 852, 961]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_14_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-16/32_text_13.png
try:
    _c32 = get_crop(32, 132, 120)
    canvas.paste(_c32, (852, 841), _c32)
except Exception:
    pass
layout["13"] = [852, 841, 984, 961]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_14_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-16/33_text_14.png
try:
    _c33 = get_crop(33, 132, 120)
    canvas.paste(_c33, (60, 961), _c33)
except Exception:
    pass
layout["14"] = [60, 961, 192, 1081]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_14_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-16/34_text_15.png
try:
    _c34 = get_crop(34, 132, 120)
    canvas.paste(_c34, (192, 961), _c34)
except Exception:
    pass
layout["15"] = [192, 961, 324, 1081]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_14_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-16/35_text_16.png
try:
    _c35 = get_crop(35, 132, 120)
    canvas.paste(_c35, (324, 961), _c35)
except Exception:
    pass
layout["16"] = [324, 961, 456, 1081]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_14_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-16/36_text_17.png
try:
    _c36 = get_crop(36, 132, 120)
    canvas.paste(_c36, (456, 961), _c36)
except Exception:
    pass
layout["17"] = [456, 961, 588, 1081]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_14_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-16/37_text_20.png
try:
    _c37 = get_crop(37, 132, 120)
    canvas.paste(_c37, (852, 961), _c37)
except Exception:
    pass
layout["20"] = [852, 961, 984, 1081]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_14_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-16/38_clickable_1.png
try:
    _c38 = get_crop(38, 132, 120)
    canvas.paste(_c38, (192, 721), _c38)
except Exception:
    pass
layout["1"] = [192, 721, 324, 841]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_14_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-16/39_clickable_2.png
try:
    _c39 = get_crop(39, 132, 120)
    canvas.paste(_c39, (324, 721), _c39)
except Exception:
    pass
layout["2"] = [324, 721, 456, 841]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_14_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-16/40_clickable_3.png
try:
    _c40 = get_crop(40, 132, 120)
    canvas.paste(_c40, (456, 721), _c40)
except Exception:
    pass
layout["3"] = [456, 721, 588, 841]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_14_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-16/41_clickable_5.png
try:
    _c41 = get_crop(41, 132, 120)
    canvas.paste(_c41, (720, 721), _c41)
except Exception:
    pass
layout["5"] = [720, 721, 852, 841]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_14_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-16/42_clickable_6.png
try:
    _c42 = get_crop(42, 132, 120)
    canvas.paste(_c42, (852, 721), _c42)
except Exception:
    pass
layout["6"] = [852, 721, 984, 841]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_14_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-16/43_clickable_7.png
try:
    _c43 = get_crop(43, 132, 120)
    canvas.paste(_c43, (60, 841), _c43)
except Exception:
    pass
layout["7"] = [60, 841, 192, 961]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_14_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-16/44_clickable_8.png
try:
    _c44 = get_crop(44, 132, 120)
    canvas.paste(_c44, (192, 841), _c44)
except Exception:
    pass
layout["8"] = [192, 841, 324, 961]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_14_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-16/45_clickable_9.png
try:
    _c45 = get_crop(45, 132, 120)
    canvas.paste(_c45, (324, 841), _c45)
except Exception:
    pass
layout["9"] = [324, 841, 456, 961]
