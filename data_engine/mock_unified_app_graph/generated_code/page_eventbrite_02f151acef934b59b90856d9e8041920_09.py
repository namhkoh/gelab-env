# page_id: page_eventbrite_02f151acef934b59b90856d9e8041920_09
# screenshot: 2024_4_24_17_24_02f151acef934b59b90856d9e8041920-11.png
# step_index: 9/11
# task: Open Eventbrite. Check the "Tech" events happening this month. Open the first event and check its date and time.
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

# Full background
draw.rectangle([(0, 0), (1440, 2960)], fill=(255, 255, 255))

# Status bar area (top)
status_h = 72
draw.rectangle([(0, 0), (1440, status_h)], fill=(230, 230, 230))

# Header / toolbar area (under status bar)
header_top = status_h
header_bottom = 200
draw.rectangle([(0, header_top), (1440, header_bottom)], fill=(255, 255, 255))

# Header bottom divider (subtle purple-gray)
draw.line([(32, header_bottom), (1408, header_bottom)], fill=(115, 86, 140), width=1)

# Card background behind Start Date / End Date area (light lavender tint)
card1_x0, card1_y0 = 32, 232
card1_x1, card1_y1 = 1408, 576
draw.rounded_rectangle([(card1_x0, card1_y0), (card1_x1, card1_y1)],
                       radius=20, fill=(250, 247, 255), outline=None)

# Soft inner top highlight for the card
draw.line([(card1_x0 + 2, card1_y0 + 2), (card1_x1 - 2, card1_y0 + 2)], fill=(245, 243, 250), width=1)

# Subtle shadow under the card
shadow_y = card1_y1 + 6
draw.rectangle([(card1_x0 + 6, card1_y1 + 2), (card1_x1 - 6, shadow_y)], fill=(245, 243, 247))

# Month header background (centered area where "April 2024" sits)
month_x0, month_y0 = 360, 620
month_x1, month_y1 = 1080, 760
draw.rounded_rectangle([(month_x0, month_y0), (month_x1, month_y1)],
                       radius=12, fill=(255, 255, 255), outline=None)
# small chevron area separator hint (right side of month header)
draw.line([(month_x1 - 40, month_y0 + 20), (month_x1 - 40, month_y1 - 20)], fill=(245, 245, 247), width=1)

# Calendar grid background block
cal_x0, cal_y0 = 32, 760
cal_x1, cal_y1 = 1408, 1400
draw.rectangle([(cal_x0, cal_y0), (cal_x1, cal_y1)], fill=(255, 255, 255))

# Faint horizontal separators between weeks (kept slightly offset to avoid direct overlap with detected date boxes)
week_lines = [820, 940, 1060, 1180, 1300]
for y in week_lines:
    draw.line([(120, y), (1320, y)], fill=(245, 243, 250), width=1)

# Light vertical guides for calendar columns (do not draw over the full height to remain subtle)
col_x = [60, 192, 324, 456, 588, 720, 852, 984]  # approximate column x positions (subtle guide lines)
for x in col_x:
    draw.line([(x - 6, cal_y0 + 12), (x - 6, cal_y1 - 12)], fill=(250, 249, 252), width=1)

# Decorative left margin strip (subtle)
draw.rectangle([(0, header_bottom + 12), (8, 1600)], fill=(250, 248, 255))

# Bottom area: subtle top separator above the "Apply date range" button region
# The auto-pasted button occupies y ~2768..2912, so draw a separator above that.
sep_y = 2728
draw.line([(48, sep_y), (1392, sep_y)], fill=(220, 217, 228), width=2)

# Shadow band above bottom button (very subtle)
draw.rectangle([(48, sep_y + 6), (1392, sep_y + 12)], fill=(250, 249, 252))

# Outer page margins (very light)
draw.rectangle([(16, 16), (1424, 40)], outline=(245, 243, 250), width=1)
draw.rectangle([(16, 2920), (1424, 2944)], outline=(245, 243, 250), width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_09_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-11/00_icon_Apply_date_range.png
try:
    _c0 = get_crop(0, 1344, 144)
    canvas.paste(_c0, (48, 2768), _c0)
except Exception:
    pass
layout["Apply_date_range"] = [48, 2768, 1392, 2912]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_09_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-11/01_icon_30.png
try:
    _c1 = get_crop(1, 132, 120)
    canvas.paste(_c1, (324, 1364), _c1)
except Exception:
    pass
layout["30"] = [324, 1364, 456, 1484]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_09_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-11/02_icon_28.png
try:
    _c2 = get_crop(2, 132, 120)
    canvas.paste(_c2, (60, 1364), _c2)
except Exception:
    pass
layout["28"] = [60, 1364, 192, 1484]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_09_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-11/03_icon_24.png
try:
    _c3 = get_crop(3, 132, 120)
    canvas.paste(_c3, (456, 1244), _c3)
except Exception:
    pass
layout["24"] = [456, 1244, 588, 1364]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_09_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-11/04_icon_29.png
try:
    _c4 = get_crop(4, 132, 120)
    canvas.paste(_c4, (192, 1364), _c4)
except Exception:
    pass
layout["29"] = [192, 1364, 324, 1484]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_09_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-11/05_icon_5.25.png
try:
    _c5 = get_crop(5, 60, 64)
    canvas.paste(_c5, (180, 2), _c5)
except Exception:
    pass
layout["5.25"] = [180, 2, 240, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_09_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-11/06_icon_25.png
try:
    _c6 = get_crop(6, 132, 120)
    canvas.paste(_c6, (588, 1244), _c6)
except Exception:
    pass
layout["25"] = [588, 1244, 720, 1364]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_09_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-11/07_icon_5.25.png
try:
    _c7 = get_crop(7, 59, 65)
    canvas.paste(_c7, (115, 1), _c7)
except Exception:
    pass
layout["5.25"] = [115, 1, 174, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_09_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-11/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 61, 62)
    canvas.paste(_c8, (310, 3), _c8)
except Exception:
    pass
layout["icon_8"] = [310, 3, 371, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_09_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-11/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 50, 60)
    canvas.paste(_c9, (249, 5), _c9)
except Exception:
    pass
layout["icon_9"] = [249, 5, 299, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_09_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-11/10_icon_24.png
try:
    _c10 = get_crop(10, 112, 134)
    canvas.paste(_c10, (471, 1358), _c10)
except Exception:
    pass
layout["24"] = [471, 1358, 583, 1492]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_09_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-11/11_icon_5.25.png
try:
    _c11 = get_crop(11, 144, 144)
    canvas.paste(_c11, (12, 72), _c11)
except Exception:
    pass
layout["5.25"] = [12, 72, 156, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_09_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-11/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 57, 70)
    canvas.paste(_c12, (1316, 0), _c12)
except Exception:
    pass
layout["icon_12"] = [1316, 0, 1373, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_09_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-11/13_icon_23.png
try:
    _c13 = get_crop(13, 132, 120)
    canvas.paste(_c13, (324, 1244), _c13)
except Exception:
    pass
layout["23"] = [324, 1244, 456, 1364]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_09_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-11/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 83, 69)
    canvas.paste(_c14, (1212, 0), _c14)
except Exception:
    pass
layout["icon_14"] = [1212, 0, 1295, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_09_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-11/15_icon_26.png
try:
    _c15 = get_crop(15, 132, 120)
    canvas.paste(_c15, (720, 1244), _c15)
except Exception:
    pass
layout["26"] = [720, 1244, 852, 1364]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_09_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-11/16_icon_22.png
try:
    _c16 = get_crop(16, 132, 120)
    canvas.paste(_c16, (192, 1244), _c16)
except Exception:
    pass
layout["22"] = [192, 1244, 324, 1364]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_09_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-11/17_icon_27.png
try:
    _c17 = get_crop(17, 132, 120)
    canvas.paste(_c17, (852, 1244), _c17)
except Exception:
    pass
layout["27"] = [852, 1244, 984, 1364]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_09_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-11/18_icon_What_date.png
try:
    _c18 = get_crop(18, 319, 72)
    canvas.paste(_c18, (558, 111), _c18)
except Exception:
    pass
layout["What_date?"] = [558, 111, 877, 183]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_09_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-11/19_icon_21.png
try:
    _c19 = get_crop(19, 132, 120)
    canvas.paste(_c19, (60, 1244), _c19)
except Exception:
    pass
layout["21"] = [60, 1244, 192, 1364]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_09_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-11/20_icon_icon_20.png
try:
    _c20 = get_crop(20, 43, 66)
    canvas.paste(_c20, (1272, 1), _c20)
except Exception:
    pass
layout["icon_20"] = [1272, 1, 1315, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_09_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-11/21_icon_Next_month.png
try:
    _c21 = get_crop(21, 144, 144)
    canvas.paste(_c21, (846, 620), _c21)
except Exception:
    pass
layout["Next_month"] = [846, 620, 990, 764]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_09_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-11/22_icon_icon_22.png
try:
    _c22 = get_crop(22, 50, 65)
    canvas.paste(_c22, (382, 2), _c22)
except Exception:
    pass
layout["icon_22"] = [382, 2, 432, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_09_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-11/23_icon_5.25.png
try:
    _c23 = get_crop(23, 91, 62)
    canvas.paste(_c23, (17, 3), _c23)
except Exception:
    pass
layout["5.25"] = [17, 3, 108, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_09_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-11/24_text_Start_Date.png
try:
    _c24 = get_crop(24, 589, 144)
    canvas.paste(_c24, (48, 313), _c24)
except Exception:
    pass
layout["Start_Date"] = [48, 313, 637, 457]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_09_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-11/25_text_End_Date.png
try:
    _c25 = get_crop(25, 587, 114)
    canvas.paste(_c25, (48, 476), _c25)
except Exception:
    pass
layout["End_Date"] = [48, 476, 635, 590]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_09_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-11/26_text_April_2024.png
try:
    _c26 = get_crop(26, 202, 54)
    canvas.paste(_c26, (421, 666), _c26)
except Exception:
    pass
layout["April_2024"] = [421, 666, 623, 720]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_09_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-11/27_text_10.png
try:
    _c27 = get_crop(27, 132, 120)
    canvas.paste(_c27, (456, 1004), _c27)
except Exception:
    pass
layout["10"] = [456, 1004, 588, 1124]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_09_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-11/28_text_11.png
try:
    _c28 = get_crop(28, 132, 120)
    canvas.paste(_c28, (588, 1004), _c28)
except Exception:
    pass
layout["11"] = [588, 1004, 720, 1124]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_09_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-11/29_text_12.png
try:
    _c29 = get_crop(29, 132, 120)
    canvas.paste(_c29, (720, 1004), _c29)
except Exception:
    pass
layout["12"] = [720, 1004, 852, 1124]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_09_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-11/30_text_13.png
try:
    _c30 = get_crop(30, 132, 120)
    canvas.paste(_c30, (852, 1004), _c30)
except Exception:
    pass
layout["13"] = [852, 1004, 984, 1124]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_09_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-11/31_text_14.png
try:
    _c31 = get_crop(31, 132, 120)
    canvas.paste(_c31, (60, 1124), _c31)
except Exception:
    pass
layout["14"] = [60, 1124, 192, 1244]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_09_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-11/32_text_15.png
try:
    _c32 = get_crop(32, 132, 120)
    canvas.paste(_c32, (192, 1124), _c32)
except Exception:
    pass
layout["15"] = [192, 1124, 324, 1244]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_09_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-11/33_text_16.png
try:
    _c33 = get_crop(33, 132, 120)
    canvas.paste(_c33, (324, 1124), _c33)
except Exception:
    pass
layout["16"] = [324, 1124, 456, 1244]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_09_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-11/34_text_17.png
try:
    _c34 = get_crop(34, 132, 120)
    canvas.paste(_c34, (456, 1124), _c34)
except Exception:
    pass
layout["17"] = [456, 1124, 588, 1244]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_09_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-11/35_text_18.png
try:
    _c35 = get_crop(35, 132, 120)
    canvas.paste(_c35, (588, 1124), _c35)
except Exception:
    pass
layout["18"] = [588, 1124, 720, 1244]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_09_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-11/36_text_19.png
try:
    _c36 = get_crop(36, 132, 120)
    canvas.paste(_c36, (720, 1124), _c36)
except Exception:
    pass
layout["19"] = [720, 1124, 852, 1244]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_09_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-11/37_text_20.png
try:
    _c37 = get_crop(37, 132, 120)
    canvas.paste(_c37, (852, 1124), _c37)
except Exception:
    pass
layout["20"] = [852, 1124, 984, 1244]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_09_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-11/38_clickable_1.png
try:
    _c38 = get_crop(38, 132, 120)
    canvas.paste(_c38, (192, 884), _c38)
except Exception:
    pass
layout["1"] = [192, 884, 324, 1004]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_09_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-11/39_clickable_2.png
try:
    _c39 = get_crop(39, 132, 120)
    canvas.paste(_c39, (324, 884), _c39)
except Exception:
    pass
layout["2"] = [324, 884, 456, 1004]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_09_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-11/40_clickable_3.png
try:
    _c40 = get_crop(40, 132, 120)
    canvas.paste(_c40, (456, 884), _c40)
except Exception:
    pass
layout["3"] = [456, 884, 588, 1004]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_09_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-11/41_clickable_4.png
try:
    _c41 = get_crop(41, 132, 120)
    canvas.paste(_c41, (588, 884), _c41)
except Exception:
    pass
layout["4"] = [588, 884, 720, 1004]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_09_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-11/42_clickable_5.png
try:
    _c42 = get_crop(42, 132, 120)
    canvas.paste(_c42, (720, 884), _c42)
except Exception:
    pass
layout["5"] = [720, 884, 852, 1004]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_09_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-11/43_clickable_6.png
try:
    _c43 = get_crop(43, 132, 120)
    canvas.paste(_c43, (852, 884), _c43)
except Exception:
    pass
layout["6"] = [852, 884, 984, 1004]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_09_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-11/44_clickable_7.png
try:
    _c44 = get_crop(44, 132, 120)
    canvas.paste(_c44, (60, 1004), _c44)
except Exception:
    pass
layout["7"] = [60, 1004, 192, 1124]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_09_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-11/45_clickable_8.png
try:
    _c45 = get_crop(45, 132, 120)
    canvas.paste(_c45, (192, 1004), _c45)
except Exception:
    pass
layout["8"] = [192, 1004, 324, 1124]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_09_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-11/46_clickable_9.png
try:
    _c46 = get_crop(46, 132, 120)
    canvas.paste(_c46, (324, 1004), _c46)
except Exception:
    pass
layout["9"] = [324, 1004, 456, 1124]
