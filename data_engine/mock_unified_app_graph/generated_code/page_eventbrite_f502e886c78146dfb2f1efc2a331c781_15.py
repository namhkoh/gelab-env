# page_id: page_eventbrite_f502e886c78146dfb2f1efc2a331c781_15
# screenshot: 2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-17.png
# step_index: 15/18
# task: Open Eventbrite. Search for 'music festival' in San Francisco. Set date available from April 30 to May 3. How many events are listed?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Paint UI background and structural elements on provided canvas/draw objects.
w, h = canvas.size

# Colors
status_gray = (220, 220, 220)        # status bar background
card_bg = (250, 248, 255)            # very light purple card background
card_outline = (230, 225, 240)       # subtle outline for card
divider = (235, 232, 240)            # light divider lines
subtle_shadow = (245, 244, 247)      # faint area shading

# 1) Status bar area (top ~72px)
status_h = 72
draw.rectangle([0, 0, w, status_h], fill=status_gray)

# 2) Header area below status bar (keep white but add a faint top accent bar to separate)
header_top = status_h
header_h = 160
draw.rectangle([0, header_top, w, header_top + header_h], fill=(255, 255, 255))
# faint bottom divider under header
draw.line([(48, header_top + header_h - 2), (w - 48, header_top + header_h - 2)], fill=divider, width=1)

# 3) Large rounded card background that groups the date selection/calendar area
card_x1, card_y1 = 48, header_top + 120   # starts lower so text/icons (Start/End) will be pasted on top
card_x2, card_y2 = w - 48, 1560
card_radius = 18
draw.rounded_rectangle([card_x1, card_y1, card_x2, card_y2], radius=card_radius, fill=card_bg, outline=card_outline, width=1)

# 4) Subtle inner shading band behind month header area (to give structure)
month_band_y = 600
draw.rectangle([card_x1 + 12, month_band_y - 32, card_x2 - 12, month_band_y + 32], fill=subtle_shadow)

# 5) Thin separators within the card to organize sections:
# Separator under the "Start/End" area (approx where end date label area finishes)
sep_y = 520
draw.line([(card_x1 + 8, sep_y), (card_x2 - 8, sep_y)], fill=divider, width=1)

# Separator under the calendar grid area (above large empty content)
cal_bottom_y = 1500
draw.line([(card_x1 + 8, cal_bottom_y), (card_x2 - 8, cal_bottom_y)], fill=divider, width=1)

# 6) Faint horizontal grid guide lines for week rows (subtle, so they won't conflict with pasted numbers)
# Only draw very light horizontal guides spaced similarly to calendar rows for visual structure
row_start_y = 720
row_height = 120
for i in range(1, 6):  # 5 horizontal guides inside calendar area
    y = row_start_y + i * row_height
    if y < card_y2 - 40:
        draw.line([(card_x1 + 40, y), (card_x2 - 40, y)], fill=(245,243,250), width=1)

# 7) Vertical padding guides for the weekday header (very faint)
weekday_top = month_band_y + 44
weekday_bottom = weekday_top + 40
draw.rectangle([card_x1 + 40, weekday_top, card_x2 - 40, weekday_bottom], outline=(250,250,250), width=1)

# 8) Bottom area separator above the actionable "Apply date range" (button will be pasted)
apply_top = 2768  # given position of detected apply button; add a subtle divider above it
draw.line([(48, apply_top - 12), (w - 48, apply_top - 12)], fill=divider, width=2)
# Add faint rounded shadow band just above the button area
draw.rounded_rectangle([48, apply_top - 8, w - 48, apply_top + 8], radius=8, fill=(250,250,252), outline=None)

# 9) Subtle left and right edge vertical guides on the main card to finish visual structure
draw.line([(card_x1, card_y1 + 8), (card_x1, card_y2 - 8)], fill=card_outline, width=1)
draw.line([(card_x2, card_y1 + 8), (card_x2, card_y2 - 8)], fill=card_outline, width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_15_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-17/00_icon_Apply_date_range.png
try:
    _c0 = get_crop(0, 1344, 144)
    canvas.paste(_c0, (48, 2768), _c0)
except Exception:
    pass
layout["Apply_date_range"] = [48, 2768, 1392, 2912]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_15_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-17/01_icon_30.png
try:
    _c1 = get_crop(1, 132, 120)
    canvas.paste(_c1, (324, 1364), _c1)
except Exception:
    pass
layout["30"] = [324, 1364, 456, 1484]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_15_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-17/02_icon_icon_2.png
try:
    _c2 = get_crop(2, 50, 71)
    canvas.paste(_c2, (1154, 0), _c2)
except Exception:
    pass
layout["icon_2"] = [1154, 0, 1204, 71]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_15_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-17/03_icon_7.19.png
try:
    _c3 = get_crop(3, 57, 63)
    canvas.paste(_c3, (182, 1), _c3)
except Exception:
    pass
layout["7.19"] = [182, 1, 239, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_15_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-17/04_icon_23.png
try:
    _c4 = get_crop(4, 132, 120)
    canvas.paste(_c4, (324, 1244), _c4)
except Exception:
    pass
layout["23"] = [324, 1244, 456, 1364]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_15_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-17/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 100, 71)
    canvas.paste(_c5, (1210, 0), _c5)
except Exception:
    pass
layout["icon_5"] = [1210, 0, 1310, 71]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_15_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-17/06_icon_7.19.png
try:
    _c6 = get_crop(6, 58, 65)
    canvas.paste(_c6, (115, 1), _c6)
except Exception:
    pass
layout["7.19"] = [115, 1, 173, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_15_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-17/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 60, 61)
    canvas.paste(_c7, (311, 4), _c7)
except Exception:
    pass
layout["icon_7"] = [311, 4, 371, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_15_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-17/08_icon_29.png
try:
    _c8 = get_crop(8, 132, 120)
    canvas.paste(_c8, (192, 1364), _c8)
except Exception:
    pass
layout["29"] = [192, 1364, 324, 1484]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_15_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-17/09_icon_28.png
try:
    _c9 = get_crop(9, 132, 120)
    canvas.paste(_c9, (60, 1364), _c9)
except Exception:
    pass
layout["28"] = [60, 1364, 192, 1484]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_15_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-17/10_icon_7.19.png
try:
    _c10 = get_crop(10, 144, 144)
    canvas.paste(_c10, (12, 72), _c10)
except Exception:
    pass
layout["7.19"] = [12, 72, 156, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_15_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-17/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 53, 70)
    canvas.paste(_c11, (1318, 0), _c11)
except Exception:
    pass
layout["icon_11"] = [1318, 0, 1371, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_15_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-17/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 51, 62)
    canvas.paste(_c12, (248, 3), _c12)
except Exception:
    pass
layout["icon_12"] = [248, 3, 299, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_15_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-17/13_icon_24.png
try:
    _c13 = get_crop(13, 115, 136)
    canvas.paste(_c13, (470, 1358), _c13)
except Exception:
    pass
layout["24"] = [470, 1358, 585, 1494]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_15_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-17/14_icon_What_date.png
try:
    _c14 = get_crop(14, 319, 72)
    canvas.paste(_c14, (558, 111), _c14)
except Exception:
    pass
layout["What_date?"] = [558, 111, 877, 183]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_15_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-17/15_icon_27.png
try:
    _c15 = get_crop(15, 132, 120)
    canvas.paste(_c15, (852, 1244), _c15)
except Exception:
    pass
layout["27"] = [852, 1244, 984, 1364]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_15_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-17/16_icon_22.png
try:
    _c16 = get_crop(16, 132, 120)
    canvas.paste(_c16, (192, 1244), _c16)
except Exception:
    pass
layout["22"] = [192, 1244, 324, 1364]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_15_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-17/17_icon_icon_17.png
try:
    _c17 = get_crop(17, 49, 64)
    canvas.paste(_c17, (382, 2), _c17)
except Exception:
    pass
layout["icon_17"] = [382, 2, 431, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_15_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-17/18_icon_Next_month.png
try:
    _c18 = get_crop(18, 144, 144)
    canvas.paste(_c18, (846, 620), _c18)
except Exception:
    pass
layout["Next_month"] = [846, 620, 990, 764]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_15_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-17/19_icon_24.png
try:
    _c19 = get_crop(19, 132, 120)
    canvas.paste(_c19, (456, 1244), _c19)
except Exception:
    pass
layout["24"] = [456, 1244, 588, 1364]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_15_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-17/20_icon_7.19.png
try:
    _c20 = get_crop(20, 92, 63)
    canvas.paste(_c20, (16, 2), _c20)
except Exception:
    pass
layout["7.19"] = [16, 2, 108, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_15_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-17/21_icon_21.png
try:
    _c21 = get_crop(21, 132, 120)
    canvas.paste(_c21, (60, 1244), _c21)
except Exception:
    pass
layout["21"] = [60, 1244, 192, 1364]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_15_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-17/22_text_Start_Date.png
try:
    _c22 = get_crop(22, 587, 144)
    canvas.paste(_c22, (48, 313), _c22)
except Exception:
    pass
layout["Start_Date"] = [48, 313, 635, 457]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_15_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-17/23_text_End_Date.png
try:
    _c23 = get_crop(23, 638, 114)
    canvas.paste(_c23, (48, 476), _c23)
except Exception:
    pass
layout["End_Date"] = [48, 476, 686, 590]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_15_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-17/24_text_April_2024.png
try:
    _c24 = get_crop(24, 202, 54)
    canvas.paste(_c24, (421, 666), _c24)
except Exception:
    pass
layout["April_2024"] = [421, 666, 623, 720]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_15_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-17/25_text_10.png
try:
    _c25 = get_crop(25, 132, 120)
    canvas.paste(_c25, (456, 1004), _c25)
except Exception:
    pass
layout["10"] = [456, 1004, 588, 1124]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_15_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-17/26_text_11.png
try:
    _c26 = get_crop(26, 132, 120)
    canvas.paste(_c26, (588, 1004), _c26)
except Exception:
    pass
layout["11"] = [588, 1004, 720, 1124]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_15_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-17/27_text_12.png
try:
    _c27 = get_crop(27, 132, 120)
    canvas.paste(_c27, (720, 1004), _c27)
except Exception:
    pass
layout["12"] = [720, 1004, 852, 1124]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_15_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-17/28_text_13.png
try:
    _c28 = get_crop(28, 132, 120)
    canvas.paste(_c28, (852, 1004), _c28)
except Exception:
    pass
layout["13"] = [852, 1004, 984, 1124]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_15_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-17/29_text_14.png
try:
    _c29 = get_crop(29, 132, 120)
    canvas.paste(_c29, (60, 1124), _c29)
except Exception:
    pass
layout["14"] = [60, 1124, 192, 1244]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_15_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-17/30_text_15.png
try:
    _c30 = get_crop(30, 132, 120)
    canvas.paste(_c30, (192, 1124), _c30)
except Exception:
    pass
layout["15"] = [192, 1124, 324, 1244]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_15_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-17/31_text_16.png
try:
    _c31 = get_crop(31, 132, 120)
    canvas.paste(_c31, (324, 1124), _c31)
except Exception:
    pass
layout["16"] = [324, 1124, 456, 1244]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_15_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-17/32_text_17.png
try:
    _c32 = get_crop(32, 132, 120)
    canvas.paste(_c32, (456, 1124), _c32)
except Exception:
    pass
layout["17"] = [456, 1124, 588, 1244]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_15_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-17/33_text_18.png
try:
    _c33 = get_crop(33, 132, 120)
    canvas.paste(_c33, (588, 1124), _c33)
except Exception:
    pass
layout["18"] = [588, 1124, 720, 1244]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_15_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-17/34_text_19.png
try:
    _c34 = get_crop(34, 132, 120)
    canvas.paste(_c34, (720, 1124), _c34)
except Exception:
    pass
layout["19"] = [720, 1124, 852, 1244]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_15_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-17/35_text_20.png
try:
    _c35 = get_crop(35, 132, 120)
    canvas.paste(_c35, (852, 1124), _c35)
except Exception:
    pass
layout["20"] = [852, 1124, 984, 1244]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_15_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-17/36_text_25.png
try:
    _c36 = get_crop(36, 132, 120)
    canvas.paste(_c36, (588, 1244), _c36)
except Exception:
    pass
layout["25"] = [588, 1244, 720, 1364]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_15_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-17/37_text_26.png
try:
    _c37 = get_crop(37, 132, 120)
    canvas.paste(_c37, (720, 1244), _c37)
except Exception:
    pass
layout["26"] = [720, 1244, 852, 1364]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_15_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-17/38_clickable_1.png
try:
    _c38 = get_crop(38, 132, 120)
    canvas.paste(_c38, (192, 884), _c38)
except Exception:
    pass
layout["1"] = [192, 884, 324, 1004]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_15_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-17/39_clickable_2.png
try:
    _c39 = get_crop(39, 132, 120)
    canvas.paste(_c39, (324, 884), _c39)
except Exception:
    pass
layout["2"] = [324, 884, 456, 1004]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_15_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-17/40_clickable_3.png
try:
    _c40 = get_crop(40, 132, 120)
    canvas.paste(_c40, (456, 884), _c40)
except Exception:
    pass
layout["3"] = [456, 884, 588, 1004]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_15_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-17/41_clickable_4.png
try:
    _c41 = get_crop(41, 132, 120)
    canvas.paste(_c41, (588, 884), _c41)
except Exception:
    pass
layout["4"] = [588, 884, 720, 1004]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_15_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-17/42_clickable_5.png
try:
    _c42 = get_crop(42, 132, 120)
    canvas.paste(_c42, (720, 884), _c42)
except Exception:
    pass
layout["5"] = [720, 884, 852, 1004]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_15_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-17/43_clickable_6.png
try:
    _c43 = get_crop(43, 132, 120)
    canvas.paste(_c43, (852, 884), _c43)
except Exception:
    pass
layout["6"] = [852, 884, 984, 1004]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_15_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-17/44_clickable_7.png
try:
    _c44 = get_crop(44, 132, 120)
    canvas.paste(_c44, (60, 1004), _c44)
except Exception:
    pass
layout["7"] = [60, 1004, 192, 1124]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_15_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-17/45_clickable_8.png
try:
    _c45 = get_crop(45, 132, 120)
    canvas.paste(_c45, (192, 1004), _c45)
except Exception:
    pass
layout["8"] = [192, 1004, 324, 1124]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_15_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-17/46_clickable_9.png
try:
    _c46 = get_crop(46, 132, 120)
    canvas.paste(_c46, (324, 1004), _c46)
except Exception:
    pass
layout["9"] = [324, 1004, 456, 1124]
