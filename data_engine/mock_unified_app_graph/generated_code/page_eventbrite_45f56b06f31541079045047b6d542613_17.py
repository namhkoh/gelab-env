# page_id: page_eventbrite_45f56b06f31541079045047b6d542613_17
# screenshot: 2024_4_23_19_27_45f56b06f31541079045047b6d542613-19.png
# step_index: 17/21
# task: Open Eventbrite. Search events 'Yoga session' in New York. Filter free events and set date from May 3 to May 6. What is the location of the first promoted event?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# draw background and structural UI elements for the mobile calendar page
w, h = canvas.size

# Colors
bg_white = "#FFFFFF"
status_bg = "#D7D7D7"       # status bar background (light gray)
divider = "#EDE9F3"         # subtle divider / card outline (very light purple-gray)
card_outline = "#E6E1EE"    # card border
card_shadow = "#F6F5F9"     # shadow / subtle lift
muted_bg = "#FBFAFC"        # slightly off-white for grouped sections

# Fill full background (dominant color)
draw.rectangle((0, 0, w, h), fill=bg_white)

# Status bar area at top (~72px)
status_h = 72
draw.rectangle((0, 0, w, status_h), fill=status_bg)
# subtle bottom divider under status bar
draw.line((0, status_h - 1, w, status_h - 1), fill=divider, width=1)

# Header / toolbar area (under status bar)
header_h = 96
header_top = status_h
header_bottom = header_top + header_h
# keep header background very slightly different to separate from page
draw.rectangle((0, header_top, w, header_bottom), fill=bg_white)
# header bottom divider
draw.line((48, header_bottom - 1, w - 48, header_bottom - 1), fill=divider, width=2)

# Start Date section card (rounded rectangle container)
card_margin_x = 48
card_top = header_bottom + 24
card_bottom = card_top + 1120  # cover calendar area
card_radius = 28
# draw a soft shadow band below the card for subtle elevation (thin strip)
shadow_top = card_top + card_bottom * 0 // 1  # no heavy calculations; keep subtle below card
draw.rectangle((card_margin_x + 6, card_top + 10, w - card_margin_x + 6, card_bottom + 10), fill=card_shadow)
# draw main rounded card
draw.rounded_rectangle(
    (card_margin_x, card_top, w - card_margin_x, card_bottom),
    radius=card_radius,
    fill=muted_bg,
    outline=card_outline,
    width=1
)

# Internal subtle horizontal divider inside Start Date card (to visually separate large title area from calendar)
divider_y = card_top + 120
draw.line((card_margin_x + 24, divider_y, w - card_margin_x - 24, divider_y), fill=divider, width=1)

# End Date section header area (no text drawn)
end_section_top = card_bottom + 36
end_section_height = 220
end_section_radius = 16
draw.rounded_rectangle(
    (card_margin_x, end_section_top, w - card_margin_x, end_section_top + end_section_height),
    radius=end_section_radius,
    fill=bg_white,
    outline=card_outline,
    width=1
)
# small divider under the End Date header area
draw.line((card_margin_x + 12, end_section_top + 56, w - card_margin_x - 12, end_section_top + 56), fill=divider, width=1)

# Large content area (blank) below End Date (keep white background but frame it slightly)
content_top = end_section_top + end_section_height + 24
draw.rectangle((card_margin_x, content_top, w - card_margin_x, h - 320), fill=bg_white, outline=None)

# Top separator above the bottom action area (do not draw the button itself)
apply_area_top = h - 320
draw.line((24, apply_area_top, w - 24, apply_area_top), fill=divider, width=2)

# Rounded container outline above the actual apply button area to echo the app layout (but do NOT draw the button content)
apply_outline_top = apply_area_top + 12
apply_outline_bottom = h - 32
apply_outline_radius = 12
# Draw only the outline to suggest a zone (transparent inside)
draw.rounded_rectangle(
    (48, apply_outline_top, w - 48, apply_outline_bottom),
    radius=apply_outline_radius,
    outline=card_outline,
    width=3,
    fill=None
)

# Final subtle horizontal rule at very top of page (under status bar) to ground header
draw.line((0, status_h + 0.5, w, status_h + 0.5), fill=divider, width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_17_2024_4_23_19_27_45f56b06f31541079045047b6d542613-19/00_icon_Apply_date_range.png
try:
    _c0 = get_crop(0, 1344, 144)
    canvas.paste(_c0, (48, 2768), _c0)
except Exception:
    pass
layout["Apply_date_range"] = [48, 2768, 1392, 2912]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_17_2024_4_23_19_27_45f56b06f31541079045047b6d542613-19/01_icon_icon_1.png
try:
    _c1 = get_crop(1, 52, 71)
    canvas.paste(_c1, (1153, 0), _c1)
except Exception:
    pass
layout["icon_1"] = [1153, 0, 1205, 71]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_17_2024_4_23_19_27_45f56b06f31541079045047b6d542613-19/02_icon_7.29.png
try:
    _c2 = get_crop(2, 61, 65)
    canvas.paste(_c2, (179, 1), _c2)
except Exception:
    pass
layout["7.29"] = [179, 1, 240, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_17_2024_4_23_19_27_45f56b06f31541079045047b6d542613-19/03_icon_7.29.png
try:
    _c3 = get_crop(3, 63, 66)
    canvas.paste(_c3, (112, 1), _c3)
except Exception:
    pass
layout["7.29"] = [112, 1, 175, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_17_2024_4_23_19_27_45f56b06f31541079045047b6d542613-19/04_icon_icon_4.png
try:
    _c4 = get_crop(4, 65, 65)
    canvas.paste(_c4, (308, 2), _c4)
except Exception:
    pass
layout["icon_4"] = [308, 2, 373, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_17_2024_4_23_19_27_45f56b06f31541079045047b6d542613-19/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 102, 70)
    canvas.paste(_c5, (1210, 0), _c5)
except Exception:
    pass
layout["icon_5"] = [1210, 0, 1312, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_17_2024_4_23_19_27_45f56b06f31541079045047b6d542613-19/06_icon_28.png
try:
    _c6 = get_crop(6, 132, 120)
    canvas.paste(_c6, (324, 1201), _c6)
except Exception:
    pass
layout["28"] = [324, 1201, 456, 1321]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_17_2024_4_23_19_27_45f56b06f31541079045047b6d542613-19/07_icon_27.png
try:
    _c7 = get_crop(7, 132, 120)
    canvas.paste(_c7, (192, 1201), _c7)
except Exception:
    pass
layout["27"] = [192, 1201, 324, 1321]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_17_2024_4_23_19_27_45f56b06f31541079045047b6d542613-19/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 52, 64)
    canvas.paste(_c8, (248, 2), _c8)
except Exception:
    pass
layout["icon_8"] = [248, 2, 300, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_17_2024_4_23_19_27_45f56b06f31541079045047b6d542613-19/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 53, 71)
    canvas.paste(_c9, (1318, 0), _c9)
except Exception:
    pass
layout["icon_9"] = [1318, 0, 1371, 71]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_17_2024_4_23_19_27_45f56b06f31541079045047b6d542613-19/10_icon_26.png
try:
    _c10 = get_crop(10, 132, 120)
    canvas.paste(_c10, (60, 1201), _c10)
except Exception:
    pass
layout["26"] = [60, 1201, 192, 1321]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_17_2024_4_23_19_27_45f56b06f31541079045047b6d542613-19/11_icon_7.29.png
try:
    _c11 = get_crop(11, 144, 144)
    canvas.paste(_c11, (12, 72), _c11)
except Exception:
    pass
layout["7.29"] = [12, 72, 156, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_17_2024_4_23_19_27_45f56b06f31541079045047b6d542613-19/12_icon_29.png
try:
    _c12 = get_crop(12, 132, 120)
    canvas.paste(_c12, (456, 1201), _c12)
except Exception:
    pass
layout["29"] = [456, 1201, 588, 1321]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_17_2024_4_23_19_27_45f56b06f31541079045047b6d542613-19/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 96, 112)
    canvas.paste(_c13, (75, 614), _c13)
except Exception:
    pass
layout["icon_13"] = [75, 614, 171, 726]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_17_2024_4_23_19_27_45f56b06f31541079045047b6d542613-19/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 49, 68)
    canvas.paste(_c14, (382, 1), _c14)
except Exception:
    pass
layout["icon_14"] = [382, 1, 431, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_17_2024_4_23_19_27_45f56b06f31541079045047b6d542613-19/15_icon_May.png
try:
    _c15 = get_crop(15, 113, 112)
    canvas.paste(_c15, (203, 612), _c15)
except Exception:
    pass
layout["May"] = [203, 612, 316, 724]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_17_2024_4_23_19_27_45f56b06f31541079045047b6d542613-19/16_icon_7.29.png
try:
    _c16 = get_crop(16, 93, 63)
    canvas.paste(_c16, (15, 1), _c16)
except Exception:
    pass
layout["7.29"] = [15, 1, 108, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_17_2024_4_23_19_27_45f56b06f31541079045047b6d542613-19/17_icon_What_date.png
try:
    _c17 = get_crop(17, 321, 71)
    canvas.paste(_c17, (558, 113), _c17)
except Exception:
    pass
layout["What_date?"] = [558, 113, 879, 184]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_17_2024_4_23_19_27_45f56b06f31541079045047b6d542613-19/18_icon_Next_month.png
try:
    _c18 = get_crop(18, 144, 144)
    canvas.paste(_c18, (846, 457), _c18)
except Exception:
    pass
layout["Next_month"] = [846, 457, 990, 601]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_17_2024_4_23_19_27_45f56b06f31541079045047b6d542613-19/19_icon_23.png
try:
    _c19 = get_crop(19, 132, 120)
    canvas.paste(_c19, (588, 1081), _c19)
except Exception:
    pass
layout["23"] = [588, 1081, 720, 1201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_17_2024_4_23_19_27_45f56b06f31541079045047b6d542613-19/20_icon_25.png
try:
    _c20 = get_crop(20, 132, 120)
    canvas.paste(_c20, (852, 1081), _c20)
except Exception:
    pass
layout["25"] = [852, 1081, 984, 1201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_17_2024_4_23_19_27_45f56b06f31541079045047b6d542613-19/21_icon_Start_Date.png
try:
    _c21 = get_crop(21, 144, 144)
    canvas.paste(_c21, (54, 457), _c21)
except Exception:
    pass
layout["Start_Date"] = [54, 457, 198, 601]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_17_2024_4_23_19_27_45f56b06f31541079045047b6d542613-19/22_icon_May.png
try:
    _c22 = get_crop(22, 144, 144)
    canvas.paste(_c22, (54, 457), _c22)
except Exception:
    pass
layout["May"] = [54, 457, 198, 601]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_17_2024_4_23_19_27_45f56b06f31541079045047b6d542613-19/23_icon_24.png
try:
    _c23 = get_crop(23, 132, 120)
    canvas.paste(_c23, (720, 1081), _c23)
except Exception:
    pass
layout["24"] = [720, 1081, 852, 1201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_17_2024_4_23_19_27_45f56b06f31541079045047b6d542613-19/24_icon_10.png
try:
    _c24 = get_crop(24, 132, 120)
    canvas.paste(_c24, (720, 841), _c24)
except Exception:
    pass
layout["10"] = [720, 841, 852, 961]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_17_2024_4_23_19_27_45f56b06f31541079045047b6d542613-19/25_text_Start_Date.png
try:
    _c25 = get_crop(25, 613, 114)
    canvas.paste(_c25, (48, 313), _c25)
except Exception:
    pass
layout["Start_Date"] = [48, 313, 661, 427]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_17_2024_4_23_19_27_45f56b06f31541079045047b6d542613-19/26_text_11.png
try:
    _c26 = get_crop(26, 132, 120)
    canvas.paste(_c26, (852, 841), _c26)
except Exception:
    pass
layout["11"] = [852, 841, 984, 961]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_17_2024_4_23_19_27_45f56b06f31541079045047b6d542613-19/27_text_12.png
try:
    _c27 = get_crop(27, 132, 120)
    canvas.paste(_c27, (60, 961), _c27)
except Exception:
    pass
layout["12"] = [60, 961, 192, 1081]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_17_2024_4_23_19_27_45f56b06f31541079045047b6d542613-19/28_text_13.png
try:
    _c28 = get_crop(28, 132, 120)
    canvas.paste(_c28, (192, 961), _c28)
except Exception:
    pass
layout["13"] = [192, 961, 324, 1081]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_17_2024_4_23_19_27_45f56b06f31541079045047b6d542613-19/29_text_14.png
try:
    _c29 = get_crop(29, 132, 120)
    canvas.paste(_c29, (324, 961), _c29)
except Exception:
    pass
layout["14"] = [324, 961, 456, 1081]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_17_2024_4_23_19_27_45f56b06f31541079045047b6d542613-19/30_text_15.png
try:
    _c30 = get_crop(30, 132, 120)
    canvas.paste(_c30, (456, 961), _c30)
except Exception:
    pass
layout["15"] = [456, 961, 588, 1081]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_17_2024_4_23_19_27_45f56b06f31541079045047b6d542613-19/31_text_16.png
try:
    _c31 = get_crop(31, 132, 120)
    canvas.paste(_c31, (588, 961), _c31)
except Exception:
    pass
layout["16"] = [588, 961, 720, 1081]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_17_2024_4_23_19_27_45f56b06f31541079045047b6d542613-19/32_text_17.png
try:
    _c32 = get_crop(32, 132, 120)
    canvas.paste(_c32, (720, 961), _c32)
except Exception:
    pass
layout["17"] = [720, 961, 852, 1081]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_17_2024_4_23_19_27_45f56b06f31541079045047b6d542613-19/33_text_18.png
try:
    _c33 = get_crop(33, 132, 120)
    canvas.paste(_c33, (852, 961), _c33)
except Exception:
    pass
layout["18"] = [852, 961, 984, 1081]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_17_2024_4_23_19_27_45f56b06f31541079045047b6d542613-19/34_text_19.png
try:
    _c34 = get_crop(34, 132, 120)
    canvas.paste(_c34, (60, 1081), _c34)
except Exception:
    pass
layout["19"] = [60, 1081, 192, 1201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_17_2024_4_23_19_27_45f56b06f31541079045047b6d542613-19/35_text_20.png
try:
    _c35 = get_crop(35, 132, 120)
    canvas.paste(_c35, (192, 1081), _c35)
except Exception:
    pass
layout["20"] = [192, 1081, 324, 1201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_17_2024_4_23_19_27_45f56b06f31541079045047b6d542613-19/36_text_21.png
try:
    _c36 = get_crop(36, 132, 120)
    canvas.paste(_c36, (324, 1081), _c36)
except Exception:
    pass
layout["21"] = [324, 1081, 456, 1201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_17_2024_4_23_19_27_45f56b06f31541079045047b6d542613-19/37_text_22.png
try:
    _c37 = get_crop(37, 132, 120)
    canvas.paste(_c37, (456, 1081), _c37)
except Exception:
    pass
layout["22"] = [456, 1081, 588, 1201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_17_2024_4_23_19_27_45f56b06f31541079045047b6d542613-19/38_text_30.png
try:
    _c38 = get_crop(38, 132, 120)
    canvas.paste(_c38, (588, 1201), _c38)
except Exception:
    pass
layout["30"] = [588, 1201, 720, 1321]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_17_2024_4_23_19_27_45f56b06f31541079045047b6d542613-19/39_text_31.png
try:
    _c39 = get_crop(39, 132, 120)
    canvas.paste(_c39, (720, 1201), _c39)
except Exception:
    pass
layout["31"] = [720, 1201, 852, 1321]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_17_2024_4_23_19_27_45f56b06f31541079045047b6d542613-19/40_text_End_Date.png
try:
    _c40 = get_crop(40, 252, 63)
    canvas.paste(_c40, (45, 1453), _c40)
except Exception:
    pass
layout["End_Date"] = [45, 1453, 297, 1516]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_17_2024_4_23_19_27_45f56b06f31541079045047b6d542613-19/41_clickable_1.png
try:
    _c41 = get_crop(41, 132, 120)
    canvas.paste(_c41, (456, 721), _c41)
except Exception:
    pass
layout["1"] = [456, 721, 588, 841]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_17_2024_4_23_19_27_45f56b06f31541079045047b6d542613-19/42_clickable_2.png
try:
    _c42 = get_crop(42, 132, 120)
    canvas.paste(_c42, (588, 721), _c42)
except Exception:
    pass
layout["2"] = [588, 721, 720, 841]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_17_2024_4_23_19_27_45f56b06f31541079045047b6d542613-19/43_clickable_3.png
try:
    _c43 = get_crop(43, 132, 120)
    canvas.paste(_c43, (720, 721), _c43)
except Exception:
    pass
layout["3"] = [720, 721, 852, 841]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_17_2024_4_23_19_27_45f56b06f31541079045047b6d542613-19/44_clickable_4.png
try:
    _c44 = get_crop(44, 132, 120)
    canvas.paste(_c44, (852, 721), _c44)
except Exception:
    pass
layout["4"] = [852, 721, 984, 841]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_17_2024_4_23_19_27_45f56b06f31541079045047b6d542613-19/45_clickable_5.png
try:
    _c45 = get_crop(45, 132, 120)
    canvas.paste(_c45, (60, 841), _c45)
except Exception:
    pass
layout["5"] = [60, 841, 192, 961]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_17_2024_4_23_19_27_45f56b06f31541079045047b6d542613-19/46_clickable_6.png
try:
    _c46 = get_crop(46, 132, 120)
    canvas.paste(_c46, (192, 841), _c46)
except Exception:
    pass
layout["6"] = [192, 841, 324, 961]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_17_2024_4_23_19_27_45f56b06f31541079045047b6d542613-19/47_clickable_7.png
try:
    _c47 = get_crop(47, 132, 120)
    canvas.paste(_c47, (324, 841), _c47)
except Exception:
    pass
layout["7"] = [324, 841, 456, 961]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_17_2024_4_23_19_27_45f56b06f31541079045047b6d542613-19/48_clickable_8.png
try:
    _c48 = get_crop(48, 132, 120)
    canvas.paste(_c48, (456, 841), _c48)
except Exception:
    pass
layout["8"] = [456, 841, 588, 961]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_17_2024_4_23_19_27_45f56b06f31541079045047b6d542613-19/49_clickable_9.png
try:
    _c49 = get_crop(49, 132, 120)
    canvas.paste(_c49, (588, 841), _c49)
except Exception:
    pass
layout["9"] = [588, 841, 720, 961]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_17_2024_4_23_19_27_45f56b06f31541079045047b6d542613-19/50_clickable_Choose_a_date.png
try:
    _c50 = get_crop(50, 638, 144)
    canvas.paste(_c50, (48, 1490), _c50)
except Exception:
    pass
layout["Choose_a_date"] = [48, 1490, 686, 1634]
