# page_id: page_eventbrite_5059e61823f845cf95cbd1d8d54254c6_13
# screenshot: 2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-15.png
# step_index: 13/19
# task: Open Eventbrite. Look for 'Education' in Los Angeles happening on May 4. Filter to show only free events. How many events are posted?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Top-level background
draw.rectangle([0, 0, 1440, 2960], fill=(255, 255, 255))

# Status bar (top area)
status_h = 72
draw.rectangle([0, 0, 1440, status_h], fill=(236, 234, 232))  # soft warm gray similar to screenshot

# Subtle bottom border/shadow under status bar
draw.line([(0, status_h), (1440, status_h)], fill=(220, 218, 216), width=1)

# Header area (title row) - keep background white but add subtle shadow
header_top = status_h
header_bottom = 200
draw.rectangle([0, header_top, 1440, header_bottom], fill=(255, 255, 255))
# shadow line
draw.line([(48, header_bottom), (1392, header_bottom)], fill=(235, 230, 240), width=1)

# Main calendar/card background behind calendar grid (light neutral panel)
cal_left, cal_top, cal_right, cal_bottom = 48, 420, 1392, 1160
draw.rounded_rectangle([cal_left, cal_top, cal_right, cal_bottom],
                       radius=18,
                       fill=(250, 250, 252),
                       outline=None)

# Month navigation background accent (behind month text area)
nav_w = 220
nav_h = 60
nav_x = 420
nav_y = 466
draw.rounded_rectangle([nav_x - 20, nav_y - 10, nav_x + nav_w, nav_y + nav_h],
                       radius=12,
                       fill=(255, 255, 255, 0),
                       outline=(235, 230, 240))

# Thin separators to divide Start Date section and End Date section
sep1_y = 320
sep2_y = 1480
draw.line([(48, sep1_y), (1392, sep1_y)], fill=(235, 232, 240), width=1)
draw.line([(48, sep2_y), (1392, sep2_y)], fill=(235, 232, 240), width=1)

# Light background block for the "End Date" section area (subtle card)
end_left, end_top, end_right, end_bottom = 48, 1320, 1392, 1700
draw.rounded_rectangle([end_left, end_top, end_right, end_bottom],
                       radius=16,
                       fill=(255, 255, 255),
                       outline=(245, 242, 247))

# Large empty content area remains white (no drawing of text or icons)
# Add a faint vertical guideline grid for alignment (very subtle, purely structural)
for x in (48, 180, 312, 444, 576, 708, 840, 972, 1104, 1236, 1392):
    draw.line([(x, 220), (x, 1320)], fill=(252, 251, 253), width=1)

# Top-left back area subtle pill (background only, icon will be pasted on top)
back_pill_box = (36, status_h + 18, 120, status_h + 72)
draw.rounded_rectangle(back_pill_box, radius=28, fill=(255, 255, 255, 0), outline=(235, 230, 240))

# Small accent divider above the bottom action area (separates main content from footer button)
bottom_sep_y = 2720
draw.line([(48, bottom_sep_y), (1392, bottom_sep_y)], fill=(230, 227, 235), width=1)

# Soft shadow band above the footer area (to imply elevation for the action area)
shadow_top = bottom_sep_y + 6
shadow_bottom = bottom_sep_y + 36
draw.rectangle([48, shadow_top, 1392, shadow_bottom], fill=(248, 246, 249))

# Light rounded outline area representing the safe region where the "Apply date range" button will be pasted
# (Only draw a faint outline behind it — do not duplicate the button itself)
btn_box = [48, 2768 - 8, 1392, 2912 + 8]
draw.rounded_rectangle(btn_box, radius=12, outline=(200, 190, 210), width=3, fill=None)

# Final subtle vignette at very bottom to anchor the layout
draw.rectangle([0, 2920, 1440, 2960], fill=(255, 255, 255, 0))

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_13_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-15/00_icon_23.png
try:
    _c0 = get_crop(0, 132, 120)
    canvas.paste(_c0, (324, 1081), _c0)
except Exception:
    pass
layout["23"] = [324, 1081, 456, 1201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_13_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-15/01_icon_Apply_date_range.png
try:
    _c1 = get_crop(1, 1344, 144)
    canvas.paste(_c1, (48, 2768), _c1)
except Exception:
    pass
layout["Apply_date_range"] = [48, 2768, 1392, 2912]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_13_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-15/02_icon_28.png
try:
    _c2 = get_crop(2, 132, 120)
    canvas.paste(_c2, (60, 1201), _c2)
except Exception:
    pass
layout["28"] = [60, 1201, 192, 1321]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_13_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-15/03_icon_24.png
try:
    _c3 = get_crop(3, 132, 120)
    canvas.paste(_c3, (456, 1081), _c3)
except Exception:
    pass
layout["24"] = [456, 1081, 588, 1201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_13_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-15/04_icon_29.png
try:
    _c4 = get_crop(4, 132, 120)
    canvas.paste(_c4, (192, 1201), _c4)
except Exception:
    pass
layout["29"] = [192, 1201, 324, 1321]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_13_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-15/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 52, 71)
    canvas.paste(_c5, (1153, 0), _c5)
except Exception:
    pass
layout["icon_5"] = [1153, 0, 1205, 71]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_13_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-15/06_icon_22.png
try:
    _c6 = get_crop(6, 132, 120)
    canvas.paste(_c6, (192, 1081), _c6)
except Exception:
    pass
layout["22"] = [192, 1081, 324, 1201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_13_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-15/07_icon_30.png
try:
    _c7 = get_crop(7, 132, 120)
    canvas.paste(_c7, (324, 1201), _c7)
except Exception:
    pass
layout["30"] = [324, 1201, 456, 1321]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_13_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-15/08_icon_7.35.png
try:
    _c8 = get_crop(8, 62, 65)
    canvas.paste(_c8, (179, 1), _c8)
except Exception:
    pass
layout["7.35"] = [179, 1, 241, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_13_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-15/09_icon_25.png
try:
    _c9 = get_crop(9, 132, 120)
    canvas.paste(_c9, (588, 1081), _c9)
except Exception:
    pass
layout["25"] = [588, 1081, 720, 1201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_13_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-15/10_icon_7.35.png
try:
    _c10 = get_crop(10, 62, 66)
    canvas.paste(_c10, (113, 1), _c10)
except Exception:
    pass
layout["7.35"] = [113, 1, 175, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_13_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-15/11_icon_26.png
try:
    _c11 = get_crop(11, 132, 120)
    canvas.paste(_c11, (720, 1081), _c11)
except Exception:
    pass
layout["26"] = [720, 1081, 852, 1201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_13_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-15/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 64, 63)
    canvas.paste(_c12, (309, 3), _c12)
except Exception:
    pass
layout["icon_12"] = [309, 3, 373, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_13_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-15/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 100, 70)
    canvas.paste(_c13, (1210, 0), _c13)
except Exception:
    pass
layout["icon_13"] = [1210, 0, 1310, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_13_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-15/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 52, 64)
    canvas.paste(_c14, (248, 2), _c14)
except Exception:
    pass
layout["icon_14"] = [248, 2, 300, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_13_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-15/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 54, 69)
    canvas.paste(_c15, (1318, 0), _c15)
except Exception:
    pass
layout["icon_15"] = [1318, 0, 1372, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_13_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-15/16_icon_27.png
try:
    _c16 = get_crop(16, 132, 120)
    canvas.paste(_c16, (852, 1081), _c16)
except Exception:
    pass
layout["27"] = [852, 1081, 984, 1201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_13_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-15/17_icon_21.png
try:
    _c17 = get_crop(17, 132, 120)
    canvas.paste(_c17, (60, 1081), _c17)
except Exception:
    pass
layout["21"] = [60, 1081, 192, 1201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_13_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-15/18_icon_7.35.png
try:
    _c18 = get_crop(18, 144, 144)
    canvas.paste(_c18, (12, 72), _c18)
except Exception:
    pass
layout["7.35"] = [12, 72, 156, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_13_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-15/19_icon_11.png
try:
    _c19 = get_crop(19, 132, 120)
    canvas.paste(_c19, (588, 721), _c19)
except Exception:
    pass
layout["11"] = [588, 721, 720, 841]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_13_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-15/20_icon_icon_20.png
try:
    _c20 = get_crop(20, 49, 67)
    canvas.paste(_c20, (382, 1), _c20)
except Exception:
    pass
layout["icon_20"] = [382, 1, 431, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_13_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-15/21_icon_April_2024.png
try:
    _c21 = get_crop(21, 126, 110)
    canvas.paste(_c21, (593, 611), _c21)
except Exception:
    pass
layout["April_2024"] = [593, 611, 719, 721]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_13_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-15/22_icon_Next_month.png
try:
    _c22 = get_crop(22, 144, 144)
    canvas.paste(_c22, (846, 457), _c22)
except Exception:
    pass
layout["Next_month"] = [846, 457, 990, 601]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_13_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-15/23_icon_18.png
try:
    _c23 = get_crop(23, 132, 120)
    canvas.paste(_c23, (588, 961), _c23)
except Exception:
    pass
layout["18"] = [588, 961, 720, 1081]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_13_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-15/24_icon_Choose_a_date.png
try:
    _c24 = get_crop(24, 638, 144)
    canvas.paste(_c24, (48, 1490), _c24)
except Exception:
    pass
layout["Choose_a_date"] = [48, 1490, 686, 1634]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_13_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-15/25_icon_12.png
try:
    _c25 = get_crop(25, 132, 120)
    canvas.paste(_c25, (720, 721), _c25)
except Exception:
    pass
layout["12"] = [720, 721, 852, 841]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_13_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-15/26_icon_12.png
try:
    _c26 = get_crop(26, 103, 107)
    canvas.paste(_c26, (734, 614), _c26)
except Exception:
    pass
layout["12"] = [734, 614, 837, 721]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_13_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-15/27_icon_7.35.png
try:
    _c27 = get_crop(27, 94, 63)
    canvas.paste(_c27, (15, 2), _c27)
except Exception:
    pass
layout["7.35"] = [15, 2, 109, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_13_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-15/28_icon_19.png
try:
    _c28 = get_crop(28, 132, 120)
    canvas.paste(_c28, (720, 961), _c28)
except Exception:
    pass
layout["19"] = [720, 961, 852, 1081]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_13_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-15/29_text_What_date.png
try:
    _c29 = get_crop(29, 318, 63)
    canvas.paste(_c29, (563, 117), _c29)
except Exception:
    pass
layout["What_date?"] = [563, 117, 881, 180]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_13_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-15/30_text_Start_Date.png
try:
    _c30 = get_crop(30, 580, 114)
    canvas.paste(_c30, (48, 313), _c30)
except Exception:
    pass
layout["Start_Date"] = [48, 313, 628, 427]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_13_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-15/31_text_April_2024.png
try:
    _c31 = get_crop(31, 203, 54)
    canvas.paste(_c31, (420, 504), _c31)
except Exception:
    pass
layout["April_2024"] = [420, 504, 623, 558]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_13_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-15/32_text_10.png
try:
    _c32 = get_crop(32, 132, 120)
    canvas.paste(_c32, (456, 841), _c32)
except Exception:
    pass
layout["10"] = [456, 841, 588, 961]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_13_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-15/33_text_11.png
try:
    _c33 = get_crop(33, 132, 120)
    canvas.paste(_c33, (588, 841), _c33)
except Exception:
    pass
layout["11"] = [588, 841, 720, 961]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_13_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-15/34_text_12.png
try:
    _c34 = get_crop(34, 132, 120)
    canvas.paste(_c34, (720, 841), _c34)
except Exception:
    pass
layout["12"] = [720, 841, 852, 961]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_13_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-15/35_text_13.png
try:
    _c35 = get_crop(35, 132, 120)
    canvas.paste(_c35, (852, 841), _c35)
except Exception:
    pass
layout["13"] = [852, 841, 984, 961]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_13_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-15/36_text_14.png
try:
    _c36 = get_crop(36, 132, 120)
    canvas.paste(_c36, (60, 961), _c36)
except Exception:
    pass
layout["14"] = [60, 961, 192, 1081]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_13_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-15/37_text_15.png
try:
    _c37 = get_crop(37, 132, 120)
    canvas.paste(_c37, (192, 961), _c37)
except Exception:
    pass
layout["15"] = [192, 961, 324, 1081]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_13_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-15/38_text_16.png
try:
    _c38 = get_crop(38, 132, 120)
    canvas.paste(_c38, (324, 961), _c38)
except Exception:
    pass
layout["16"] = [324, 961, 456, 1081]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_13_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-15/39_text_17.png
try:
    _c39 = get_crop(39, 132, 120)
    canvas.paste(_c39, (456, 961), _c39)
except Exception:
    pass
layout["17"] = [456, 961, 588, 1081]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_13_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-15/40_text_20.png
try:
    _c40 = get_crop(40, 132, 120)
    canvas.paste(_c40, (852, 961), _c40)
except Exception:
    pass
layout["20"] = [852, 961, 984, 1081]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_13_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-15/41_clickable_1.png
try:
    _c41 = get_crop(41, 132, 120)
    canvas.paste(_c41, (192, 721), _c41)
except Exception:
    pass
layout["1"] = [192, 721, 324, 841]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_13_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-15/42_clickable_2.png
try:
    _c42 = get_crop(42, 132, 120)
    canvas.paste(_c42, (324, 721), _c42)
except Exception:
    pass
layout["2"] = [324, 721, 456, 841]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_13_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-15/43_clickable_3.png
try:
    _c43 = get_crop(43, 132, 120)
    canvas.paste(_c43, (456, 721), _c43)
except Exception:
    pass
layout["3"] = [456, 721, 588, 841]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_13_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-15/44_clickable_6.png
try:
    _c44 = get_crop(44, 132, 120)
    canvas.paste(_c44, (852, 721), _c44)
except Exception:
    pass
layout["6"] = [852, 721, 984, 841]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_13_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-15/45_clickable_7.png
try:
    _c45 = get_crop(45, 132, 120)
    canvas.paste(_c45, (60, 841), _c45)
except Exception:
    pass
layout["7"] = [60, 841, 192, 961]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_13_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-15/46_clickable_8.png
try:
    _c46 = get_crop(46, 132, 120)
    canvas.paste(_c46, (192, 841), _c46)
except Exception:
    pass
layout["8"] = [192, 841, 324, 961]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_13_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-15/47_clickable_9.png
try:
    _c47 = get_crop(47, 132, 120)
    canvas.paste(_c47, (324, 841), _c47)
except Exception:
    pass
layout["9"] = [324, 841, 456, 961]
