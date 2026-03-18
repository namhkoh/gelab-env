# page_id: page_eventbrite_5059e61823f845cf95cbd1d8d54254c6_15
# screenshot: 2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-17.png
# step_index: 15/19
# task: Open Eventbrite. Look for 'Education' in Los Angeles happening on May 4. Filter to show only free events. How many events are posted?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Paint UI background and structural elements for the calendar screen
# Uses provided variables: canvas (PIL.Image) and draw (PIL.ImageDraw)

# Full white background (dominant color)
draw.rectangle([(0, 0), (1440, 2960)], fill=(255, 255, 255))

# Status bar area (top ~72px) - light neutral background for icons
status_h = 72
draw.rectangle([(0, 0), (1440, status_h)], fill=(237, 238, 240))

# Subtle bottom border of status bar to separate from header
draw.line([(0, status_h), (1440, status_h)], fill=(230, 228, 235), width=1)

# Header / toolbar area (under status bar)
header_top = status_h
header_bottom = 156
draw.rectangle([(0, header_top), (1440, header_bottom)], fill=(255, 255, 255))

# Thin divider under header
draw.line([(36, header_bottom), (1404, header_bottom)], fill=(245, 241, 249), width=1)

# Calendar/content card background (light purple-tinted panel behind calendar)
card_left = 36
card_right = 1404
card_top = 220
card_bottom = 1340
draw.rounded_rectangle(
    [(card_left, card_top), (card_right, card_bottom)],
    radius=28,
    fill=(250, 249, 252),
    outline=None
)

# Add a very subtle inner horizontal guide to visually separate month nav area
nav_line_y = card_top + 80
draw.line([(card_left + 24, nav_line_y), (card_right - 24, nav_line_y)], fill=(244, 240, 248), width=1)

# Separator between calendar area and the "End Date" section
sep_y = 1376
draw.line([(36, sep_y), (1404, sep_y)], fill=(245, 241, 249), width=1)

# Light large whitespace band behind the End Date area to provide visual grouping
end_card_top = sep_y + 16
end_card_bottom = end_card_top + 220
draw.rectangle([(36, end_card_top), (1404, end_card_bottom)], fill=(255, 255, 255))

# Subtle divider/shadow near the bottom to separate main content from bottom action area
bottom_shadow_y = 2720
draw.line([(24, bottom_shadow_y), (1416, bottom_shadow_y)], fill=(236, 234, 240), width=2)
draw.line([(24, bottom_shadow_y + 2), (1416, bottom_shadow_y + 2)], fill=(250, 249, 251), width=1)

# Gentle corner highlights on page edges (very faint) to match app feel
# Left edge vertical fade rectangle (very subtle)
draw.rectangle([(0, 200), (8, 1600)], fill=(255, 255, 255, 10))
draw.rectangle([(1432, 200), (1440, 1600)], fill=(255, 255, 255, 10))

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_15_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-17/00_icon_Apply_date_range.png
try:
    _c0 = get_crop(0, 1344, 144)
    canvas.paste(_c0, (48, 2768), _c0)
except Exception:
    pass
layout["Apply_date_range"] = [48, 2768, 1392, 2912]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_15_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-17/01_icon_icon_1.png
try:
    _c1 = get_crop(1, 52, 71)
    canvas.paste(_c1, (1153, 0), _c1)
except Exception:
    pass
layout["icon_1"] = [1153, 0, 1205, 71]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_15_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-17/02_icon_7.35.png
try:
    _c2 = get_crop(2, 61, 65)
    canvas.paste(_c2, (179, 1), _c2)
except Exception:
    pass
layout["7.35"] = [179, 1, 240, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_15_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-17/03_icon_icon_3.png
try:
    _c3 = get_crop(3, 65, 64)
    canvas.paste(_c3, (308, 2), _c3)
except Exception:
    pass
layout["icon_3"] = [308, 2, 373, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_15_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-17/04_icon_7.35.png
try:
    _c4 = get_crop(4, 62, 66)
    canvas.paste(_c4, (113, 1), _c4)
except Exception:
    pass
layout["7.35"] = [113, 1, 175, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_15_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-17/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 101, 70)
    canvas.paste(_c5, (1210, 0), _c5)
except Exception:
    pass
layout["icon_5"] = [1210, 0, 1311, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_15_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-17/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 54, 71)
    canvas.paste(_c6, (1318, 0), _c6)
except Exception:
    pass
layout["icon_6"] = [1318, 0, 1372, 71]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_15_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-17/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 52, 64)
    canvas.paste(_c7, (248, 2), _c7)
except Exception:
    pass
layout["icon_7"] = [248, 2, 300, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_15_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-17/08_icon_26.png
try:
    _c8 = get_crop(8, 132, 120)
    canvas.paste(_c8, (60, 1201), _c8)
except Exception:
    pass
layout["26"] = [60, 1201, 192, 1321]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_15_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-17/09_icon_7.35.png
try:
    _c9 = get_crop(9, 144, 144)
    canvas.paste(_c9, (12, 72), _c9)
except Exception:
    pass
layout["7.35"] = [12, 72, 156, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_15_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-17/10_icon_28.png
try:
    _c10 = get_crop(10, 132, 120)
    canvas.paste(_c10, (324, 1201), _c10)
except Exception:
    pass
layout["28"] = [324, 1201, 456, 1321]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_15_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-17/11_icon_27.png
try:
    _c11 = get_crop(11, 132, 120)
    canvas.paste(_c11, (192, 1201), _c11)
except Exception:
    pass
layout["27"] = [192, 1201, 324, 1321]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_15_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-17/12_icon_Next_month.png
try:
    _c12 = get_crop(12, 144, 144)
    canvas.paste(_c12, (846, 457), _c12)
except Exception:
    pass
layout["Next_month"] = [846, 457, 990, 601]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_15_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-17/13_icon_May.png
try:
    _c13 = get_crop(13, 144, 144)
    canvas.paste(_c13, (54, 457), _c13)
except Exception:
    pass
layout["May"] = [54, 457, 198, 601]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_15_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-17/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 49, 68)
    canvas.paste(_c14, (382, 1), _c14)
except Exception:
    pass
layout["icon_14"] = [382, 1, 431, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_15_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-17/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 91, 103)
    canvas.paste(_c15, (76, 618), _c15)
except Exception:
    pass
layout["icon_15"] = [76, 618, 167, 721]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_15_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-17/16_icon_Start_Date.png
try:
    _c16 = get_crop(16, 144, 144)
    canvas.paste(_c16, (54, 457), _c16)
except Exception:
    pass
layout["Start_Date"] = [54, 457, 198, 601]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_15_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-17/17_icon_29.png
try:
    _c17 = get_crop(17, 132, 120)
    canvas.paste(_c17, (456, 1201), _c17)
except Exception:
    pass
layout["29"] = [456, 1201, 588, 1321]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_15_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-17/18_icon_May.png
try:
    _c18 = get_crop(18, 112, 110)
    canvas.paste(_c18, (200, 612), _c18)
except Exception:
    pass
layout["May"] = [200, 612, 312, 722]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_15_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-17/19_text_What_date.png
try:
    _c19 = get_crop(19, 318, 63)
    canvas.paste(_c19, (563, 117), _c19)
except Exception:
    pass
layout["What_date?"] = [563, 117, 881, 180]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_15_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-17/20_text_Start_Date.png
try:
    _c20 = get_crop(20, 620, 114)
    canvas.paste(_c20, (48, 313), _c20)
except Exception:
    pass
layout["Start_Date"] = [48, 313, 668, 427]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_15_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-17/21_text_10.png
try:
    _c21 = get_crop(21, 132, 120)
    canvas.paste(_c21, (720, 841), _c21)
except Exception:
    pass
layout["10"] = [720, 841, 852, 961]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_15_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-17/22_text_11.png
try:
    _c22 = get_crop(22, 132, 120)
    canvas.paste(_c22, (852, 841), _c22)
except Exception:
    pass
layout["11"] = [852, 841, 984, 961]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_15_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-17/23_text_12.png
try:
    _c23 = get_crop(23, 132, 120)
    canvas.paste(_c23, (60, 961), _c23)
except Exception:
    pass
layout["12"] = [60, 961, 192, 1081]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_15_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-17/24_text_13.png
try:
    _c24 = get_crop(24, 132, 120)
    canvas.paste(_c24, (192, 961), _c24)
except Exception:
    pass
layout["13"] = [192, 961, 324, 1081]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_15_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-17/25_text_14.png
try:
    _c25 = get_crop(25, 132, 120)
    canvas.paste(_c25, (324, 961), _c25)
except Exception:
    pass
layout["14"] = [324, 961, 456, 1081]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_15_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-17/26_text_15.png
try:
    _c26 = get_crop(26, 132, 120)
    canvas.paste(_c26, (456, 961), _c26)
except Exception:
    pass
layout["15"] = [456, 961, 588, 1081]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_15_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-17/27_text_16.png
try:
    _c27 = get_crop(27, 132, 120)
    canvas.paste(_c27, (588, 961), _c27)
except Exception:
    pass
layout["16"] = [588, 961, 720, 1081]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_15_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-17/28_text_17.png
try:
    _c28 = get_crop(28, 132, 120)
    canvas.paste(_c28, (720, 961), _c28)
except Exception:
    pass
layout["17"] = [720, 961, 852, 1081]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_15_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-17/29_text_18.png
try:
    _c29 = get_crop(29, 132, 120)
    canvas.paste(_c29, (852, 961), _c29)
except Exception:
    pass
layout["18"] = [852, 961, 984, 1081]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_15_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-17/30_text_19.png
try:
    _c30 = get_crop(30, 132, 120)
    canvas.paste(_c30, (60, 1081), _c30)
except Exception:
    pass
layout["19"] = [60, 1081, 192, 1201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_15_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-17/31_text_20.png
try:
    _c31 = get_crop(31, 132, 120)
    canvas.paste(_c31, (192, 1081), _c31)
except Exception:
    pass
layout["20"] = [192, 1081, 324, 1201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_15_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-17/32_text_21.png
try:
    _c32 = get_crop(32, 132, 120)
    canvas.paste(_c32, (324, 1081), _c32)
except Exception:
    pass
layout["21"] = [324, 1081, 456, 1201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_15_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-17/33_text_22.png
try:
    _c33 = get_crop(33, 132, 120)
    canvas.paste(_c33, (456, 1081), _c33)
except Exception:
    pass
layout["22"] = [456, 1081, 588, 1201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_15_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-17/34_text_23.png
try:
    _c34 = get_crop(34, 132, 120)
    canvas.paste(_c34, (588, 1081), _c34)
except Exception:
    pass
layout["23"] = [588, 1081, 720, 1201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_15_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-17/35_text_24.png
try:
    _c35 = get_crop(35, 132, 120)
    canvas.paste(_c35, (720, 1081), _c35)
except Exception:
    pass
layout["24"] = [720, 1081, 852, 1201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_15_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-17/36_text_25.png
try:
    _c36 = get_crop(36, 132, 120)
    canvas.paste(_c36, (852, 1081), _c36)
except Exception:
    pass
layout["25"] = [852, 1081, 984, 1201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_15_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-17/37_text_30.png
try:
    _c37 = get_crop(37, 132, 120)
    canvas.paste(_c37, (588, 1201), _c37)
except Exception:
    pass
layout["30"] = [588, 1201, 720, 1321]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_15_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-17/38_text_31.png
try:
    _c38 = get_crop(38, 132, 120)
    canvas.paste(_c38, (720, 1201), _c38)
except Exception:
    pass
layout["31"] = [720, 1201, 852, 1321]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_15_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-17/39_text_End_Date.png
try:
    _c39 = get_crop(39, 252, 63)
    canvas.paste(_c39, (45, 1453), _c39)
except Exception:
    pass
layout["End_Date"] = [45, 1453, 297, 1516]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_15_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-17/40_clickable_1.png
try:
    _c40 = get_crop(40, 132, 120)
    canvas.paste(_c40, (456, 721), _c40)
except Exception:
    pass
layout["1"] = [456, 721, 588, 841]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_15_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-17/41_clickable_2.png
try:
    _c41 = get_crop(41, 132, 120)
    canvas.paste(_c41, (588, 721), _c41)
except Exception:
    pass
layout["2"] = [588, 721, 720, 841]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_15_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-17/42_clickable_3.png
try:
    _c42 = get_crop(42, 132, 120)
    canvas.paste(_c42, (720, 721), _c42)
except Exception:
    pass
layout["3"] = [720, 721, 852, 841]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_15_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-17/43_clickable_4.png
try:
    _c43 = get_crop(43, 132, 120)
    canvas.paste(_c43, (852, 721), _c43)
except Exception:
    pass
layout["4"] = [852, 721, 984, 841]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_15_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-17/44_clickable_5.png
try:
    _c44 = get_crop(44, 132, 120)
    canvas.paste(_c44, (60, 841), _c44)
except Exception:
    pass
layout["5"] = [60, 841, 192, 961]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_15_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-17/45_clickable_6.png
try:
    _c45 = get_crop(45, 132, 120)
    canvas.paste(_c45, (192, 841), _c45)
except Exception:
    pass
layout["6"] = [192, 841, 324, 961]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_15_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-17/46_clickable_7.png
try:
    _c46 = get_crop(46, 132, 120)
    canvas.paste(_c46, (324, 841), _c46)
except Exception:
    pass
layout["7"] = [324, 841, 456, 961]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_15_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-17/47_clickable_8.png
try:
    _c47 = get_crop(47, 132, 120)
    canvas.paste(_c47, (456, 841), _c47)
except Exception:
    pass
layout["8"] = [456, 841, 588, 961]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_15_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-17/48_clickable_9.png
try:
    _c48 = get_crop(48, 132, 120)
    canvas.paste(_c48, (588, 841), _c48)
except Exception:
    pass
layout["9"] = [588, 841, 720, 961]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_15_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-17/49_clickable_Choose_a_date.png
try:
    _c49 = get_crop(49, 638, 144)
    canvas.paste(_c49, (48, 1490), _c49)
except Exception:
    pass
layout["Choose_a_date"] = [48, 1490, 686, 1634]
