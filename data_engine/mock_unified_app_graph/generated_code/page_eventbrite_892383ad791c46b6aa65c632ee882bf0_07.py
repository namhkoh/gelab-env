# page_id: page_eventbrite_892383ad791c46b6aa65c632ee882bf0_07
# screenshot: 2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-9.png
# step_index: 7/12
# task: Open Eventbrite. Search for online "Music" events happening next weekend.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Background & UI structural artwork for the mobile date-picker page.
# Uses provided canvas (1440x2960) and draw (ImageDraw) objects.

# Colors (approximate to screenshot)
_bg = (250, 250, 252)            # very light off-white background
_status_bar = (211, 211, 211)    # light grey status bar
_divider = (230, 230, 235)       # subtle dividers
_card_bg = (247, 247, 249)       # very light card background
_subtle_shadow = (240, 238, 246) # faint tint for header/card accents

# Fill overall background
draw.rectangle((0, 0, 1440, 2960), fill=_bg)

# Top status bar area (~72-88px tall)
status_h = 88
draw.rectangle((0, 0, 1440, status_h), fill=_status_bar)
# subtle bottom edge for status bar
draw.line((0, status_h - 1, 1440, status_h - 1), fill=_divider, width=1)

# Header area (contains title). Keep background same as canvas but add a divider below.
header_top = status_h
header_bottom = 170
draw.rectangle((0, header_top, 1440, header_bottom), fill=_bg)
# header bottom divider
draw.line((48, header_bottom, 1392, header_bottom), fill=_divider, width=1)

# Large calendar "card" background to sit behind the month & grid
card_x0, card_x1 = 48, 1392
card_y0, card_y1 = 220, 1320
draw.rounded_rectangle((card_x0, card_y0, card_x1, card_y1), radius=28, fill=_card_bg)
# very faint inner top accent to imply separation for the month heading
draw.line((card_x0 + 20, card_y0 + 100, card_x1 - 20, card_y0 + 100), fill=_subtle_shadow, width=1)

# Subtle horizontal divider separating calendar grid area from the following content
divider_y = 1440
draw.line((48, divider_y, 1392, divider_y), fill=_divider, width=1)

# End Date section background hint (large whitespace area but add a faint block behind label area)
end_section_x0, end_section_x1 = 48, 1392
end_section_y0, end_section_y1 = 1460, 2000
draw.rectangle((end_section_x0, end_section_y0, end_section_x1, end_section_y1), fill=_bg)
# a faint left accent to lead the eye (does not duplicate text/icons)
draw.rectangle((end_section_x0 + 12, end_section_y0 + 12, end_section_x0 + 18, end_section_y0 + 80), fill=_subtle_shadow)

# Bottom area separation above the persistent "Apply date range" control (control itself will be pasted)
button_top = 2768
sep_y = button_top - 56
# draw a faint horizontal separator and a subtle shadow band above the control area
draw.line((36, sep_y, 1404, sep_y), fill=_divider, width=1)
draw.rectangle((36, sep_y + 4, 1404, sep_y + 10), fill=(249, 249, 250))

# Small decorative left & right margins at very bottom to hint safe-area (no button content drawn)
draw.rectangle((24, 2928, 1416, 2956), fill=_bg)

# Add a faint vertical guide on the left for visual balance (non-interactive background only)
draw.line((48, header_bottom + 8, 48, 2600), fill=(245,245,247), width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_07_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-9/00_icon_24.png
try:
    _c0 = get_crop(0, 132, 120)
    canvas.paste(_c0, (456, 1081), _c0)
except Exception:
    pass
layout["24"] = [456, 1081, 588, 1201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_07_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-9/01_icon_Apply_date_range.png
try:
    _c1 = get_crop(1, 1344, 144)
    canvas.paste(_c1, (48, 2768), _c1)
except Exception:
    pass
layout["Apply_date_range"] = [48, 2768, 1392, 2912]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_07_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-9/02_icon_28.png
try:
    _c2 = get_crop(2, 132, 120)
    canvas.paste(_c2, (60, 1201), _c2)
except Exception:
    pass
layout["28"] = [60, 1201, 192, 1321]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_07_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-9/03_icon_29.png
try:
    _c3 = get_crop(3, 132, 120)
    canvas.paste(_c3, (192, 1201), _c3)
except Exception:
    pass
layout["29"] = [192, 1201, 324, 1321]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_07_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-9/04_icon_23.png
try:
    _c4 = get_crop(4, 132, 120)
    canvas.paste(_c4, (324, 1081), _c4)
except Exception:
    pass
layout["23"] = [324, 1081, 456, 1201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_07_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-9/05_icon_25.png
try:
    _c5 = get_crop(5, 132, 120)
    canvas.paste(_c5, (588, 1081), _c5)
except Exception:
    pass
layout["25"] = [588, 1081, 720, 1201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_07_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-9/06_icon_30.png
try:
    _c6 = get_crop(6, 132, 120)
    canvas.paste(_c6, (324, 1201), _c6)
except Exception:
    pass
layout["30"] = [324, 1201, 456, 1321]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_07_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-9/07_icon_27.png
try:
    _c7 = get_crop(7, 132, 120)
    canvas.paste(_c7, (852, 1081), _c7)
except Exception:
    pass
layout["27"] = [852, 1081, 984, 1201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_07_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-9/08_icon_26.png
try:
    _c8 = get_crop(8, 132, 120)
    canvas.paste(_c8, (720, 1081), _c8)
except Exception:
    pass
layout["26"] = [720, 1081, 852, 1201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_07_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-9/09_icon_5.23.png
try:
    _c9 = get_crop(9, 62, 65)
    canvas.paste(_c9, (179, 1), _c9)
except Exception:
    pass
layout["5.23"] = [179, 1, 241, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_07_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-9/10_icon_5.23.png
try:
    _c10 = get_crop(10, 62, 66)
    canvas.paste(_c10, (113, 1), _c10)
except Exception:
    pass
layout["5.23"] = [113, 1, 175, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_07_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-9/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 63, 64)
    canvas.paste(_c11, (309, 2), _c11)
except Exception:
    pass
layout["icon_11"] = [309, 2, 372, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_07_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-9/12_icon_21.png
try:
    _c12 = get_crop(12, 132, 120)
    canvas.paste(_c12, (60, 1081), _c12)
except Exception:
    pass
layout["21"] = [60, 1081, 192, 1201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_07_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-9/13_icon_22.png
try:
    _c13 = get_crop(13, 132, 120)
    canvas.paste(_c13, (192, 1081), _c13)
except Exception:
    pass
layout["22"] = [192, 1081, 324, 1201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_07_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-9/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 53, 64)
    canvas.paste(_c14, (247, 2), _c14)
except Exception:
    pass
layout["icon_14"] = [247, 2, 300, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_07_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-9/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 57, 70)
    canvas.paste(_c15, (1316, 0), _c15)
except Exception:
    pass
layout["icon_15"] = [1316, 0, 1373, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_07_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-9/16_icon_5.23.png
try:
    _c16 = get_crop(16, 144, 144)
    canvas.paste(_c16, (12, 72), _c16)
except Exception:
    pass
layout["5.23"] = [12, 72, 156, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_07_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-9/17_icon_18.png
try:
    _c17 = get_crop(17, 132, 120)
    canvas.paste(_c17, (588, 961), _c17)
except Exception:
    pass
layout["18"] = [588, 961, 720, 1081]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_07_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-9/18_icon_icon_18.png
try:
    _c18 = get_crop(18, 90, 69)
    canvas.paste(_c18, (1211, 0), _c18)
except Exception:
    pass
layout["icon_18"] = [1211, 0, 1301, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_07_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-9/19_icon_11.png
try:
    _c19 = get_crop(19, 132, 120)
    canvas.paste(_c19, (588, 721), _c19)
except Exception:
    pass
layout["11"] = [588, 721, 720, 841]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_07_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-9/20_icon_5.23.png
try:
    _c20 = get_crop(20, 92, 64)
    canvas.paste(_c20, (16, 1), _c20)
except Exception:
    pass
layout["5.23"] = [16, 1, 108, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_07_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-9/21_icon_19.png
try:
    _c21 = get_crop(21, 132, 120)
    canvas.paste(_c21, (720, 961), _c21)
except Exception:
    pass
layout["19"] = [720, 961, 852, 1081]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_07_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-9/22_icon_icon_22.png
try:
    _c22 = get_crop(22, 49, 67)
    canvas.paste(_c22, (382, 1), _c22)
except Exception:
    pass
layout["icon_22"] = [382, 1, 431, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_07_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-9/23_icon_April_2024.png
try:
    _c23 = get_crop(23, 126, 110)
    canvas.paste(_c23, (593, 611), _c23)
except Exception:
    pass
layout["April_2024"] = [593, 611, 719, 721]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_07_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-9/24_icon_icon_24.png
try:
    _c24 = get_crop(24, 41, 65)
    canvas.paste(_c24, (1274, 0), _c24)
except Exception:
    pass
layout["icon_24"] = [1274, 0, 1315, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_07_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-9/25_icon_Next_month.png
try:
    _c25 = get_crop(25, 144, 144)
    canvas.paste(_c25, (846, 457), _c25)
except Exception:
    pass
layout["Next_month"] = [846, 457, 990, 601]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_07_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-9/26_icon_12.png
try:
    _c26 = get_crop(26, 132, 120)
    canvas.paste(_c26, (720, 721), _c26)
except Exception:
    pass
layout["12"] = [720, 721, 852, 841]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_07_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-9/27_icon_12.png
try:
    _c27 = get_crop(27, 104, 107)
    canvas.paste(_c27, (733, 614), _c27)
except Exception:
    pass
layout["12"] = [733, 614, 837, 721]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_07_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-9/28_icon_icon_28.png
try:
    _c28 = get_crop(28, 103, 100)
    canvas.paste(_c28, (72, 618), _c28)
except Exception:
    pass
layout["icon_28"] = [72, 618, 175, 718]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_07_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-9/29_icon_Choose_a_date.png
try:
    _c29 = get_crop(29, 638, 144)
    canvas.paste(_c29, (48, 1490), _c29)
except Exception:
    pass
layout["Choose_a_date"] = [48, 1490, 686, 1634]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_07_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-9/30_icon_What_date.png
try:
    _c30 = get_crop(30, 322, 71)
    canvas.paste(_c30, (558, 113), _c30)
except Exception:
    pass
layout["What_date?"] = [558, 113, 880, 184]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_07_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-9/31_icon_16.png
try:
    _c31 = get_crop(31, 132, 120)
    canvas.paste(_c31, (324, 961), _c31)
except Exception:
    pass
layout["16"] = [324, 961, 456, 1081]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_07_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-9/32_icon_10.png
try:
    _c32 = get_crop(32, 132, 120)
    canvas.paste(_c32, (324, 721), _c32)
except Exception:
    pass
layout["10"] = [324, 721, 456, 841]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_07_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-9/33_text_Start_Date.png
try:
    _c33 = get_crop(33, 589, 114)
    canvas.paste(_c33, (48, 313), _c33)
except Exception:
    pass
layout["Start_Date"] = [48, 313, 637, 427]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_07_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-9/34_text_April_2024.png
try:
    _c34 = get_crop(34, 203, 54)
    canvas.paste(_c34, (420, 504), _c34)
except Exception:
    pass
layout["April_2024"] = [420, 504, 623, 558]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_07_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-9/35_text_10.png
try:
    _c35 = get_crop(35, 132, 120)
    canvas.paste(_c35, (456, 841), _c35)
except Exception:
    pass
layout["10"] = [456, 841, 588, 961]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_07_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-9/36_text_11.png
try:
    _c36 = get_crop(36, 132, 120)
    canvas.paste(_c36, (588, 841), _c36)
except Exception:
    pass
layout["11"] = [588, 841, 720, 961]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_07_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-9/37_text_12.png
try:
    _c37 = get_crop(37, 132, 120)
    canvas.paste(_c37, (720, 841), _c37)
except Exception:
    pass
layout["12"] = [720, 841, 852, 961]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_07_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-9/38_text_13.png
try:
    _c38 = get_crop(38, 132, 120)
    canvas.paste(_c38, (852, 841), _c38)
except Exception:
    pass
layout["13"] = [852, 841, 984, 961]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_07_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-9/39_text_14.png
try:
    _c39 = get_crop(39, 132, 120)
    canvas.paste(_c39, (60, 961), _c39)
except Exception:
    pass
layout["14"] = [60, 961, 192, 1081]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_07_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-9/40_text_15.png
try:
    _c40 = get_crop(40, 132, 120)
    canvas.paste(_c40, (192, 961), _c40)
except Exception:
    pass
layout["15"] = [192, 961, 324, 1081]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_07_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-9/41_text_17.png
try:
    _c41 = get_crop(41, 132, 120)
    canvas.paste(_c41, (456, 961), _c41)
except Exception:
    pass
layout["17"] = [456, 961, 588, 1081]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_07_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-9/42_text_20.png
try:
    _c42 = get_crop(42, 132, 120)
    canvas.paste(_c42, (852, 961), _c42)
except Exception:
    pass
layout["20"] = [852, 961, 984, 1081]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_07_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-9/43_clickable_1.png
try:
    _c43 = get_crop(43, 132, 120)
    canvas.paste(_c43, (192, 721), _c43)
except Exception:
    pass
layout["1"] = [192, 721, 324, 841]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_07_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-9/44_clickable_3.png
try:
    _c44 = get_crop(44, 132, 120)
    canvas.paste(_c44, (456, 721), _c44)
except Exception:
    pass
layout["3"] = [456, 721, 588, 841]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_07_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-9/45_clickable_6.png
try:
    _c45 = get_crop(45, 132, 120)
    canvas.paste(_c45, (852, 721), _c45)
except Exception:
    pass
layout["6"] = [852, 721, 984, 841]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_07_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-9/46_clickable_7.png
try:
    _c46 = get_crop(46, 132, 120)
    canvas.paste(_c46, (60, 841), _c46)
except Exception:
    pass
layout["7"] = [60, 841, 192, 961]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_07_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-9/47_clickable_8.png
try:
    _c47 = get_crop(47, 132, 120)
    canvas.paste(_c47, (192, 841), _c47)
except Exception:
    pass
layout["8"] = [192, 841, 324, 961]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_07_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-9/48_clickable_9.png
try:
    _c48 = get_crop(48, 132, 120)
    canvas.paste(_c48, (324, 841), _c48)
except Exception:
    pass
layout["9"] = [324, 841, 456, 961]
