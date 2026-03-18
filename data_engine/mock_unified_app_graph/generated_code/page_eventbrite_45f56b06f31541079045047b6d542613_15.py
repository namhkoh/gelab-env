# page_id: page_eventbrite_45f56b06f31541079045047b6d542613_15
# screenshot: 2024_4_23_19_27_45f56b06f31541079045047b6d542613-17.png
# step_index: 15/21
# task: Open Eventbrite. Search events 'Yoga session' in New York. Filter free events and set date from May 3 to May 6. What is the location of the first promoted event?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw background and structural UI elements for the calendar/date-picker page.
# Uses provided `canvas` (PIL Image) and `draw` (ImageDraw.Draw).
# Assumes canvas is 1440x2960.

w, h = canvas.size

# Colors
bg_color = "#ffffff"           # main background (white)
status_bar_color = "#d7d7d7"   # light grey status bar
divider_color = "#efe8f6"      # very light purple divider
card_outline = "#efe6f6"       # card subtle outline
card_fill = "#ffffff"          # card fill (white, slightly different if needed)
muted_bg = "#faf9fb"           # subtle off-white for section backgrounds

# Clear full canvas to base background
draw.rectangle([(0, 0), (w, h)], fill=bg_color)

# Status bar area (~72px tall to cover system icons area)
status_bar_height = 72
draw.rectangle([(0, 0), (w, status_bar_height)], fill=status_bar_color)

# Header / toolbar area (below status bar)
header_top = status_bar_height
header_bottom = 192
# Keep header visually distinct: white background with a subtle bottom divider
draw.rectangle([(0, header_top), (w, header_bottom)], fill=card_fill)
draw.line([(48, header_bottom), (w - 48, header_bottom)], fill=divider_color, width=2)

# Main calendar card background (rounded rectangle)
card_left = 48
card_right = w - 48
card_top = 232
card_bottom = 1240
card_radius = 20
try:
    draw.rounded_rectangle([(card_left, card_top), (card_right, card_bottom)],
                           radius=card_radius, fill=card_fill, outline=card_outline, width=1)
except Exception:
    # Fallback if rounded_rectangle isn't available
    draw.rectangle([(card_left, card_top), (card_right, card_bottom)], fill=card_fill, outline=card_outline)

# Subtle divider under the calendar card to separate "End Date" section
sep_y = 1400
draw.line([(card_left, sep_y), (card_right, sep_y)], fill=divider_color, width=2)

# "End Date" section background area (large open area)
end_section_top = sep_y + 24
end_section_bottom = 2320
try:
    draw.rounded_rectangle([(card_left, end_section_top), (card_right, end_section_bottom)],
                           radius=16, fill=muted_bg, outline=None)
except Exception:
    draw.rectangle([(card_left, end_section_top), (card_right, end_section_bottom)], fill=muted_bg)

# Subtle horizontal rule above the bottom action area (keeps space from Apply button)
bottom_rule_y = 2720
draw.line([(24, bottom_rule_y), (w - 24, bottom_rule_y)], fill="#ece7ef", width=1)

# Outer edge top/bottom padding accents (very faint)
draw.line([(24, card_top - 8), (w - 24, card_top - 8)], fill="#fbf7fc", width=1)
draw.line([(24, end_section_bottom + 8), (w - 24, end_section_bottom + 8)], fill="#fbf7fc", width=1)

# Small left gutter guideline (purely structural, very faint)
draw.line([(48, header_bottom + 8), (48, h - 160)], fill="#fbf7fc", width=1)

# Decorative subtle drop shadow under the calendar card (light)
shadow_top = card_bottom
shadow_height = 10
for i in range(6):
    alpha_shade = 240 - i * 30
    # create slightly darker lines to simulate shadow (using grey values)
    shade = (230 - i * 8, 227 - i * 8, 235 - i * 8)
    y = shadow_top + i
    draw.line([(card_left + 6, y), (card_right - 6, y)], fill=shade, width=1)

# End: keep all actual icons/text areas untouched (background/structural elements only)
# (Do not draw any icon/text content — those will be pasted later.)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_15_2024_4_23_19_27_45f56b06f31541079045047b6d542613-17/00_icon_23.png
try:
    _c0 = get_crop(0, 132, 120)
    canvas.paste(_c0, (324, 1081), _c0)
except Exception:
    pass
layout["23"] = [324, 1081, 456, 1201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_15_2024_4_23_19_27_45f56b06f31541079045047b6d542613-17/01_icon_Apply_date_range.png
try:
    _c1 = get_crop(1, 1344, 144)
    canvas.paste(_c1, (48, 2768), _c1)
except Exception:
    pass
layout["Apply_date_range"] = [48, 2768, 1392, 2912]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_15_2024_4_23_19_27_45f56b06f31541079045047b6d542613-17/02_icon_28.png
try:
    _c2 = get_crop(2, 132, 120)
    canvas.paste(_c2, (60, 1201), _c2)
except Exception:
    pass
layout["28"] = [60, 1201, 192, 1321]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_15_2024_4_23_19_27_45f56b06f31541079045047b6d542613-17/03_icon_24.png
try:
    _c3 = get_crop(3, 132, 120)
    canvas.paste(_c3, (456, 1081), _c3)
except Exception:
    pass
layout["24"] = [456, 1081, 588, 1201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_15_2024_4_23_19_27_45f56b06f31541079045047b6d542613-17/04_icon_29.png
try:
    _c4 = get_crop(4, 132, 120)
    canvas.paste(_c4, (192, 1201), _c4)
except Exception:
    pass
layout["29"] = [192, 1201, 324, 1321]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_15_2024_4_23_19_27_45f56b06f31541079045047b6d542613-17/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 52, 71)
    canvas.paste(_c5, (1153, 0), _c5)
except Exception:
    pass
layout["icon_5"] = [1153, 0, 1205, 71]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_15_2024_4_23_19_27_45f56b06f31541079045047b6d542613-17/06_icon_22.png
try:
    _c6 = get_crop(6, 132, 120)
    canvas.paste(_c6, (192, 1081), _c6)
except Exception:
    pass
layout["22"] = [192, 1081, 324, 1201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_15_2024_4_23_19_27_45f56b06f31541079045047b6d542613-17/07_icon_30.png
try:
    _c7 = get_crop(7, 132, 120)
    canvas.paste(_c7, (324, 1201), _c7)
except Exception:
    pass
layout["30"] = [324, 1201, 456, 1321]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_15_2024_4_23_19_27_45f56b06f31541079045047b6d542613-17/08_icon_7.29.png
try:
    _c8 = get_crop(8, 62, 65)
    canvas.paste(_c8, (179, 1), _c8)
except Exception:
    pass
layout["7.29"] = [179, 1, 241, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_15_2024_4_23_19_27_45f56b06f31541079045047b6d542613-17/09_icon_25.png
try:
    _c9 = get_crop(9, 132, 120)
    canvas.paste(_c9, (588, 1081), _c9)
except Exception:
    pass
layout["25"] = [588, 1081, 720, 1201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_15_2024_4_23_19_27_45f56b06f31541079045047b6d542613-17/10_icon_7.29.png
try:
    _c10 = get_crop(10, 62, 66)
    canvas.paste(_c10, (113, 1), _c10)
except Exception:
    pass
layout["7.29"] = [113, 1, 175, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_15_2024_4_23_19_27_45f56b06f31541079045047b6d542613-17/11_icon_26.png
try:
    _c11 = get_crop(11, 132, 120)
    canvas.paste(_c11, (720, 1081), _c11)
except Exception:
    pass
layout["26"] = [720, 1081, 852, 1201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_15_2024_4_23_19_27_45f56b06f31541079045047b6d542613-17/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 64, 63)
    canvas.paste(_c12, (309, 3), _c12)
except Exception:
    pass
layout["icon_12"] = [309, 3, 373, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_15_2024_4_23_19_27_45f56b06f31541079045047b6d542613-17/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 100, 70)
    canvas.paste(_c13, (1210, 0), _c13)
except Exception:
    pass
layout["icon_13"] = [1210, 0, 1310, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_15_2024_4_23_19_27_45f56b06f31541079045047b6d542613-17/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 52, 63)
    canvas.paste(_c14, (248, 2), _c14)
except Exception:
    pass
layout["icon_14"] = [248, 2, 300, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_15_2024_4_23_19_27_45f56b06f31541079045047b6d542613-17/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 54, 69)
    canvas.paste(_c15, (1318, 0), _c15)
except Exception:
    pass
layout["icon_15"] = [1318, 0, 1372, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_15_2024_4_23_19_27_45f56b06f31541079045047b6d542613-17/16_icon_27.png
try:
    _c16 = get_crop(16, 132, 120)
    canvas.paste(_c16, (852, 1081), _c16)
except Exception:
    pass
layout["27"] = [852, 1081, 984, 1201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_15_2024_4_23_19_27_45f56b06f31541079045047b6d542613-17/17_icon_21.png
try:
    _c17 = get_crop(17, 132, 120)
    canvas.paste(_c17, (60, 1081), _c17)
except Exception:
    pass
layout["21"] = [60, 1081, 192, 1201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_15_2024_4_23_19_27_45f56b06f31541079045047b6d542613-17/18_icon_7.29.png
try:
    _c18 = get_crop(18, 144, 144)
    canvas.paste(_c18, (12, 72), _c18)
except Exception:
    pass
layout["7.29"] = [12, 72, 156, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_15_2024_4_23_19_27_45f56b06f31541079045047b6d542613-17/19_icon_11.png
try:
    _c19 = get_crop(19, 132, 120)
    canvas.paste(_c19, (588, 721), _c19)
except Exception:
    pass
layout["11"] = [588, 721, 720, 841]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_15_2024_4_23_19_27_45f56b06f31541079045047b6d542613-17/20_icon_icon_20.png
try:
    _c20 = get_crop(20, 49, 67)
    canvas.paste(_c20, (382, 1), _c20)
except Exception:
    pass
layout["icon_20"] = [382, 1, 431, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_15_2024_4_23_19_27_45f56b06f31541079045047b6d542613-17/21_icon_April_2024.png
try:
    _c21 = get_crop(21, 126, 110)
    canvas.paste(_c21, (593, 611), _c21)
except Exception:
    pass
layout["April_2024"] = [593, 611, 719, 721]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_15_2024_4_23_19_27_45f56b06f31541079045047b6d542613-17/22_icon_Next_month.png
try:
    _c22 = get_crop(22, 144, 144)
    canvas.paste(_c22, (846, 457), _c22)
except Exception:
    pass
layout["Next_month"] = [846, 457, 990, 601]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_15_2024_4_23_19_27_45f56b06f31541079045047b6d542613-17/23_icon_18.png
try:
    _c23 = get_crop(23, 132, 120)
    canvas.paste(_c23, (588, 961), _c23)
except Exception:
    pass
layout["18"] = [588, 961, 720, 1081]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_15_2024_4_23_19_27_45f56b06f31541079045047b6d542613-17/24_icon_Choose_a_date.png
try:
    _c24 = get_crop(24, 638, 144)
    canvas.paste(_c24, (48, 1490), _c24)
except Exception:
    pass
layout["Choose_a_date"] = [48, 1490, 686, 1634]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_15_2024_4_23_19_27_45f56b06f31541079045047b6d542613-17/25_icon_7.29.png
try:
    _c25 = get_crop(25, 92, 63)
    canvas.paste(_c25, (16, 1), _c25)
except Exception:
    pass
layout["7.29"] = [16, 1, 108, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_15_2024_4_23_19_27_45f56b06f31541079045047b6d542613-17/26_icon_12.png
try:
    _c26 = get_crop(26, 132, 120)
    canvas.paste(_c26, (720, 721), _c26)
except Exception:
    pass
layout["12"] = [720, 721, 852, 841]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_15_2024_4_23_19_27_45f56b06f31541079045047b6d542613-17/27_icon_12.png
try:
    _c27 = get_crop(27, 103, 107)
    canvas.paste(_c27, (734, 614), _c27)
except Exception:
    pass
layout["12"] = [734, 614, 837, 721]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_15_2024_4_23_19_27_45f56b06f31541079045047b6d542613-17/28_icon_19.png
try:
    _c28 = get_crop(28, 132, 120)
    canvas.paste(_c28, (720, 961), _c28)
except Exception:
    pass
layout["19"] = [720, 961, 852, 1081]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_15_2024_4_23_19_27_45f56b06f31541079045047b6d542613-17/29_text_What_date.png
try:
    _c29 = get_crop(29, 318, 63)
    canvas.paste(_c29, (563, 117), _c29)
except Exception:
    pass
layout["What_date?"] = [563, 117, 881, 180]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_15_2024_4_23_19_27_45f56b06f31541079045047b6d542613-17/30_text_Start_Date.png
try:
    _c30 = get_crop(30, 580, 114)
    canvas.paste(_c30, (48, 313), _c30)
except Exception:
    pass
layout["Start_Date"] = [48, 313, 628, 427]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_15_2024_4_23_19_27_45f56b06f31541079045047b6d542613-17/31_text_April_2024.png
try:
    _c31 = get_crop(31, 203, 54)
    canvas.paste(_c31, (420, 504), _c31)
except Exception:
    pass
layout["April_2024"] = [420, 504, 623, 558]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_15_2024_4_23_19_27_45f56b06f31541079045047b6d542613-17/32_text_10.png
try:
    _c32 = get_crop(32, 132, 120)
    canvas.paste(_c32, (456, 841), _c32)
except Exception:
    pass
layout["10"] = [456, 841, 588, 961]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_15_2024_4_23_19_27_45f56b06f31541079045047b6d542613-17/33_text_11.png
try:
    _c33 = get_crop(33, 132, 120)
    canvas.paste(_c33, (588, 841), _c33)
except Exception:
    pass
layout["11"] = [588, 841, 720, 961]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_15_2024_4_23_19_27_45f56b06f31541079045047b6d542613-17/34_text_12.png
try:
    _c34 = get_crop(34, 132, 120)
    canvas.paste(_c34, (720, 841), _c34)
except Exception:
    pass
layout["12"] = [720, 841, 852, 961]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_15_2024_4_23_19_27_45f56b06f31541079045047b6d542613-17/35_text_13.png
try:
    _c35 = get_crop(35, 132, 120)
    canvas.paste(_c35, (852, 841), _c35)
except Exception:
    pass
layout["13"] = [852, 841, 984, 961]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_15_2024_4_23_19_27_45f56b06f31541079045047b6d542613-17/36_text_14.png
try:
    _c36 = get_crop(36, 132, 120)
    canvas.paste(_c36, (60, 961), _c36)
except Exception:
    pass
layout["14"] = [60, 961, 192, 1081]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_15_2024_4_23_19_27_45f56b06f31541079045047b6d542613-17/37_text_15.png
try:
    _c37 = get_crop(37, 132, 120)
    canvas.paste(_c37, (192, 961), _c37)
except Exception:
    pass
layout["15"] = [192, 961, 324, 1081]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_15_2024_4_23_19_27_45f56b06f31541079045047b6d542613-17/38_text_16.png
try:
    _c38 = get_crop(38, 132, 120)
    canvas.paste(_c38, (324, 961), _c38)
except Exception:
    pass
layout["16"] = [324, 961, 456, 1081]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_15_2024_4_23_19_27_45f56b06f31541079045047b6d542613-17/39_text_17.png
try:
    _c39 = get_crop(39, 132, 120)
    canvas.paste(_c39, (456, 961), _c39)
except Exception:
    pass
layout["17"] = [456, 961, 588, 1081]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_15_2024_4_23_19_27_45f56b06f31541079045047b6d542613-17/40_text_20.png
try:
    _c40 = get_crop(40, 132, 120)
    canvas.paste(_c40, (852, 961), _c40)
except Exception:
    pass
layout["20"] = [852, 961, 984, 1081]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_15_2024_4_23_19_27_45f56b06f31541079045047b6d542613-17/41_clickable_1.png
try:
    _c41 = get_crop(41, 132, 120)
    canvas.paste(_c41, (192, 721), _c41)
except Exception:
    pass
layout["1"] = [192, 721, 324, 841]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_15_2024_4_23_19_27_45f56b06f31541079045047b6d542613-17/42_clickable_2.png
try:
    _c42 = get_crop(42, 132, 120)
    canvas.paste(_c42, (324, 721), _c42)
except Exception:
    pass
layout["2"] = [324, 721, 456, 841]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_15_2024_4_23_19_27_45f56b06f31541079045047b6d542613-17/43_clickable_3.png
try:
    _c43 = get_crop(43, 132, 120)
    canvas.paste(_c43, (456, 721), _c43)
except Exception:
    pass
layout["3"] = [456, 721, 588, 841]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_15_2024_4_23_19_27_45f56b06f31541079045047b6d542613-17/44_clickable_6.png
try:
    _c44 = get_crop(44, 132, 120)
    canvas.paste(_c44, (852, 721), _c44)
except Exception:
    pass
layout["6"] = [852, 721, 984, 841]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_15_2024_4_23_19_27_45f56b06f31541079045047b6d542613-17/45_clickable_7.png
try:
    _c45 = get_crop(45, 132, 120)
    canvas.paste(_c45, (60, 841), _c45)
except Exception:
    pass
layout["7"] = [60, 841, 192, 961]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_15_2024_4_23_19_27_45f56b06f31541079045047b6d542613-17/46_clickable_8.png
try:
    _c46 = get_crop(46, 132, 120)
    canvas.paste(_c46, (192, 841), _c46)
except Exception:
    pass
layout["8"] = [192, 841, 324, 961]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_15_2024_4_23_19_27_45f56b06f31541079045047b6d542613-17/47_clickable_9.png
try:
    _c47 = get_crop(47, 132, 120)
    canvas.paste(_c47, (324, 841), _c47)
except Exception:
    pass
layout["9"] = [324, 841, 456, 961]
