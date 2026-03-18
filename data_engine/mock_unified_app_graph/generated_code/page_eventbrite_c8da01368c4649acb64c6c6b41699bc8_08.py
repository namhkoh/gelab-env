# page_id: page_eventbrite_c8da01368c4649acb64c6c6b41699bc8_08
# screenshot: 2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-10.png
# step_index: 8/13
# task: Open Eventbrite. Look up "Animal" events. Filter by events happening next week. Select the first event - who is the organizer?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw background and structural elements for the mobile UI page
# Assumes provided variables: canvas (PIL.Image), draw (ImageDraw), font_sm, font_md, font_lg, font_xl

W, H = canvas.size

# Global colors that match the screenshot
bg_offwhite = (255, 255, 255)            # main background (very white)
status_bar_gray = (200, 200, 200)        # top status bar
divider_light = (236, 233, 241)          # very light divider / pale purple-gray
button_outline = (120, 110, 145)         # muted purple/gray for button border
muted_purple = (60, 20, 60)              # used for subtle accents (not for text)

# Fill overall background (canvas starts white but ensure consistent tone)
draw.rectangle([(0, 0), (W, H)], fill=bg_offwhite)

# Status bar area (top) - leave icons to be pasted on top
status_bar_height = 72
draw.rectangle([(0, 0), (W, status_bar_height)], fill=status_bar_gray)

# Header / toolbar area beneath status bar
header_top = status_bar_height
header_height = 96
draw.rectangle([(0, header_top), (W, header_top + header_height)], fill=bg_offwhite)

# Subtle bottom divider under header
divider_y = header_top + header_height
draw.line([(48, divider_y), (W - 48, divider_y)], fill=divider_light, width=2)

# Calendar / content area background block (subtle, mostly white so we keep very subtle fill)
# This defines a region behind the calendar numbers and month view
cal_margin_x = 48
cal_top = divider_y + 24
cal_bottom = 1320
draw.rectangle([(cal_margin_x, cal_top), (W - cal_margin_x, cal_bottom)], fill=bg_offwhite)

# A faint separator line below the calendar region before the "End Date" section
end_section_top = 1480
draw.line([(cal_margin_x, end_section_top), (W - cal_margin_x, end_section_top)], fill=divider_light, width=2)

# End Date section background area (keeps white but mark it with a very subtle tint on edges)
end_section_height = 300
draw.rectangle([(cal_margin_x, end_section_top), (W - cal_margin_x, end_section_top + end_section_height)], fill=bg_offwhite)

# Bottom action button background (rounded rectangle) - leave label and inner content to be pasted
btn_x0, btn_y0 = 48, 2768
btn_x1, btn_y1 = W - 48, btn_y0 + 144
btn_radius = 12
# White fill with subtle outline to match screenshot
draw.rounded_rectangle([(btn_x0, btn_y0), (btn_x1, btn_y1)], radius=btn_radius, fill=bg_offwhite,
                       outline=button_outline, width=6)

# Subtle shadow/line above the button to separate it from content (very faint)
draw.line([(btn_x0 + 6, btn_y0 - 18), (btn_x1 - 6, btn_y0 - 18)], fill=divider_light, width=1)

# Additional faint separators to structure the page:
#  - a faint divider between the large Start Date area and the calendar block
start_date_div_y = cal_top - 48
draw.line([(cal_margin_x, start_date_div_y), (W - cal_margin_x, start_date_div_y)], fill=divider_light, width=1)

#  - faint bottom edge near the very bottom of the screen for grounding the layout
ground_y = btn_y1 + 8
draw.line([(24, ground_y), (W - 24, ground_y)], fill=divider_light, width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_08_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-10/00_icon_28.png
try:
    _c0 = get_crop(0, 132, 120)
    canvas.paste(_c0, (60, 1201), _c0)
except Exception:
    pass
layout["28"] = [60, 1201, 192, 1321]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_08_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-10/01_icon_Apply_date_range.png
try:
    _c1 = get_crop(1, 1344, 144)
    canvas.paste(_c1, (48, 2768), _c1)
except Exception:
    pass
layout["Apply_date_range"] = [48, 2768, 1392, 2912]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_08_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-10/02_icon_29.png
try:
    _c2 = get_crop(2, 132, 120)
    canvas.paste(_c2, (192, 1201), _c2)
except Exception:
    pass
layout["29"] = [192, 1201, 324, 1321]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_08_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-10/03_icon_24.png
try:
    _c3 = get_crop(3, 132, 120)
    canvas.paste(_c3, (456, 1081), _c3)
except Exception:
    pass
layout["24"] = [456, 1081, 588, 1201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_08_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-10/04_icon_30.png
try:
    _c4 = get_crop(4, 132, 120)
    canvas.paste(_c4, (324, 1201), _c4)
except Exception:
    pass
layout["30"] = [324, 1201, 456, 1321]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_08_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-10/05_icon_23.png
try:
    _c5 = get_crop(5, 132, 120)
    canvas.paste(_c5, (324, 1081), _c5)
except Exception:
    pass
layout["23"] = [324, 1081, 456, 1201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_08_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-10/06_icon_5.15.png
try:
    _c6 = get_crop(6, 61, 66)
    canvas.paste(_c6, (180, 0), _c6)
except Exception:
    pass
layout["5.15"] = [180, 0, 241, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_08_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-10/07_icon_25.png
try:
    _c7 = get_crop(7, 132, 120)
    canvas.paste(_c7, (588, 1081), _c7)
except Exception:
    pass
layout["25"] = [588, 1081, 720, 1201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_08_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-10/08_icon_22.png
try:
    _c8 = get_crop(8, 132, 120)
    canvas.paste(_c8, (192, 1081), _c8)
except Exception:
    pass
layout["22"] = [192, 1081, 324, 1201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_08_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-10/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 63, 64)
    canvas.paste(_c9, (309, 2), _c9)
except Exception:
    pass
layout["icon_9"] = [309, 2, 372, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_08_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-10/10_icon_5.15.png
try:
    _c10 = get_crop(10, 61, 68)
    canvas.paste(_c10, (114, 0), _c10)
except Exception:
    pass
layout["5.15"] = [114, 0, 175, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_08_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-10/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 53, 64)
    canvas.paste(_c11, (247, 2), _c11)
except Exception:
    pass
layout["icon_11"] = [247, 2, 300, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_08_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-10/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 57, 69)
    canvas.paste(_c12, (1316, 0), _c12)
except Exception:
    pass
layout["icon_12"] = [1316, 0, 1373, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_08_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-10/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 92, 69)
    canvas.paste(_c13, (1211, 0), _c13)
except Exception:
    pass
layout["icon_13"] = [1211, 0, 1303, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_08_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-10/14_icon_5.15.png
try:
    _c14 = get_crop(14, 144, 144)
    canvas.paste(_c14, (12, 72), _c14)
except Exception:
    pass
layout["5.15"] = [12, 72, 156, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_08_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-10/15_icon_26.png
try:
    _c15 = get_crop(15, 132, 120)
    canvas.paste(_c15, (720, 1081), _c15)
except Exception:
    pass
layout["26"] = [720, 1081, 852, 1201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_08_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-10/16_icon_27.png
try:
    _c16 = get_crop(16, 132, 120)
    canvas.paste(_c16, (852, 1081), _c16)
except Exception:
    pass
layout["27"] = [852, 1081, 984, 1201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_08_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-10/17_icon_icon_17.png
try:
    _c17 = get_crop(17, 41, 65)
    canvas.paste(_c17, (1274, 0), _c17)
except Exception:
    pass
layout["icon_17"] = [1274, 0, 1315, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_08_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-10/18_icon_11.png
try:
    _c18 = get_crop(18, 132, 120)
    canvas.paste(_c18, (588, 721), _c18)
except Exception:
    pass
layout["11"] = [588, 721, 720, 841]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_08_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-10/19_icon_icon_19.png
try:
    _c19 = get_crop(19, 49, 67)
    canvas.paste(_c19, (382, 1), _c19)
except Exception:
    pass
layout["icon_19"] = [382, 1, 431, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_08_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-10/20_icon_Choose_a_date.png
try:
    _c20 = get_crop(20, 638, 144)
    canvas.paste(_c20, (48, 1490), _c20)
except Exception:
    pass
layout["Choose_a_date"] = [48, 1490, 686, 1634]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_08_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-10/21_icon_Next_month.png
try:
    _c21 = get_crop(21, 144, 144)
    canvas.paste(_c21, (846, 457), _c21)
except Exception:
    pass
layout["Next_month"] = [846, 457, 990, 601]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_08_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-10/22_icon_April_2024.png
try:
    _c22 = get_crop(22, 121, 109)
    canvas.paste(_c22, (596, 611), _c22)
except Exception:
    pass
layout["April_2024"] = [596, 611, 717, 720]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_08_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-10/23_icon_What_date.png
try:
    _c23 = get_crop(23, 322, 71)
    canvas.paste(_c23, (558, 113), _c23)
except Exception:
    pass
layout["What_date?"] = [558, 113, 880, 184]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_08_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-10/24_text_5.15.png
try:
    _c24 = get_crop(24, 92, 43)
    canvas.paste(_c24, (22, 17), _c24)
except Exception:
    pass
layout["5.15"] = [22, 17, 114, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_08_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-10/25_text_Start_Date.png
try:
    _c25 = get_crop(25, 583, 114)
    canvas.paste(_c25, (48, 313), _c25)
except Exception:
    pass
layout["Start_Date"] = [48, 313, 631, 427]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_08_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-10/26_text_April_2024.png
try:
    _c26 = get_crop(26, 203, 54)
    canvas.paste(_c26, (420, 504), _c26)
except Exception:
    pass
layout["April_2024"] = [420, 504, 623, 558]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_08_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-10/27_text_10.png
try:
    _c27 = get_crop(27, 132, 120)
    canvas.paste(_c27, (456, 841), _c27)
except Exception:
    pass
layout["10"] = [456, 841, 588, 961]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_08_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-10/28_text_11.png
try:
    _c28 = get_crop(28, 132, 120)
    canvas.paste(_c28, (588, 841), _c28)
except Exception:
    pass
layout["11"] = [588, 841, 720, 961]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_08_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-10/29_text_12.png
try:
    _c29 = get_crop(29, 132, 120)
    canvas.paste(_c29, (720, 841), _c29)
except Exception:
    pass
layout["12"] = [720, 841, 852, 961]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_08_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-10/30_text_13.png
try:
    _c30 = get_crop(30, 132, 120)
    canvas.paste(_c30, (852, 841), _c30)
except Exception:
    pass
layout["13"] = [852, 841, 984, 961]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_08_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-10/31_text_14.png
try:
    _c31 = get_crop(31, 132, 120)
    canvas.paste(_c31, (60, 961), _c31)
except Exception:
    pass
layout["14"] = [60, 961, 192, 1081]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_08_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-10/32_text_15.png
try:
    _c32 = get_crop(32, 132, 120)
    canvas.paste(_c32, (192, 961), _c32)
except Exception:
    pass
layout["15"] = [192, 961, 324, 1081]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_08_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-10/33_text_16.png
try:
    _c33 = get_crop(33, 132, 120)
    canvas.paste(_c33, (324, 961), _c33)
except Exception:
    pass
layout["16"] = [324, 961, 456, 1081]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_08_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-10/34_text_17.png
try:
    _c34 = get_crop(34, 132, 120)
    canvas.paste(_c34, (456, 961), _c34)
except Exception:
    pass
layout["17"] = [456, 961, 588, 1081]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_08_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-10/35_text_18.png
try:
    _c35 = get_crop(35, 132, 120)
    canvas.paste(_c35, (588, 961), _c35)
except Exception:
    pass
layout["18"] = [588, 961, 720, 1081]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_08_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-10/36_text_19.png
try:
    _c36 = get_crop(36, 132, 120)
    canvas.paste(_c36, (720, 961), _c36)
except Exception:
    pass
layout["19"] = [720, 961, 852, 1081]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_08_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-10/37_text_20.png
try:
    _c37 = get_crop(37, 132, 120)
    canvas.paste(_c37, (852, 961), _c37)
except Exception:
    pass
layout["20"] = [852, 961, 984, 1081]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_08_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-10/38_text_21.png
try:
    _c38 = get_crop(38, 132, 120)
    canvas.paste(_c38, (60, 1081), _c38)
except Exception:
    pass
layout["21"] = [60, 1081, 192, 1201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_08_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-10/39_clickable_1.png
try:
    _c39 = get_crop(39, 132, 120)
    canvas.paste(_c39, (192, 721), _c39)
except Exception:
    pass
layout["1"] = [192, 721, 324, 841]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_08_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-10/40_clickable_2.png
try:
    _c40 = get_crop(40, 132, 120)
    canvas.paste(_c40, (324, 721), _c40)
except Exception:
    pass
layout["2"] = [324, 721, 456, 841]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_08_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-10/41_clickable_3.png
try:
    _c41 = get_crop(41, 132, 120)
    canvas.paste(_c41, (456, 721), _c41)
except Exception:
    pass
layout["3"] = [456, 721, 588, 841]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_08_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-10/42_clickable_5.png
try:
    _c42 = get_crop(42, 132, 120)
    canvas.paste(_c42, (720, 721), _c42)
except Exception:
    pass
layout["5"] = [720, 721, 852, 841]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_08_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-10/43_clickable_6.png
try:
    _c43 = get_crop(43, 132, 120)
    canvas.paste(_c43, (852, 721), _c43)
except Exception:
    pass
layout["6"] = [852, 721, 984, 841]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_08_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-10/44_clickable_7.png
try:
    _c44 = get_crop(44, 132, 120)
    canvas.paste(_c44, (60, 841), _c44)
except Exception:
    pass
layout["7"] = [60, 841, 192, 961]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_08_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-10/45_clickable_8.png
try:
    _c45 = get_crop(45, 132, 120)
    canvas.paste(_c45, (192, 841), _c45)
except Exception:
    pass
layout["8"] = [192, 841, 324, 961]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_08_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-10/46_clickable_9.png
try:
    _c46 = get_crop(46, 132, 120)
    canvas.paste(_c46, (324, 841), _c46)
except Exception:
    pass
layout["9"] = [324, 841, 456, 961]
