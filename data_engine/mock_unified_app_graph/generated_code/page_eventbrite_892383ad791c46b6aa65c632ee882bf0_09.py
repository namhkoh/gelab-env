# page_id: page_eventbrite_892383ad791c46b6aa65c632ee882bf0_09
# screenshot: 2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-11.png
# step_index: 9/12
# task: Open Eventbrite. Search for online "Music" events happening next weekend.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw background and UI structure for the calendar/date-picker screen.
# Uses provided variables: canvas (PIL Image), draw (ImageDraw), font_sm, font_md, font_lg, font_xl

w, h = canvas.size

# Colors
status_bar_color = (226, 226, 226)      # light grey status bar
header_divider = (230, 226, 240)        # faint purple-ish divider under header
calendar_card_fill = (250, 249, 252)    # very light off-white for calendar card
calendar_card_stroke = (240, 236, 246)  # subtle stroke around the card
grid_line = (242, 240, 246)             # faint grid lines between weeks
section_shadow = (235, 233, 243)        # faint shadow under card
bottom_separator = (220, 218, 226)      # faint line above bottom button area

# 1) Status bar area (top ~80px)
status_bar_height = 80
draw.rectangle([(0, 0), (w, status_bar_height)], fill=status_bar_color)

# 2) Header area below status bar (leave background mostly white, add subtle divider)
header_top = status_bar_height
header_bottom = 180
draw.rectangle([(0, header_top), (w, header_bottom)], fill=(255, 255, 255))
# subtle divider line below header
draw.line([(40, header_bottom - 8), (w - 40, header_bottom - 8)], fill=header_divider, width=2)

# 3) Calendar card background (rounded rectangle grouping the month + calendar)
# Keep the card in the upper-middle area so it doesn't overlap bottom controls.
card_left = 40
card_right = w - 40
card_top = 560
card_bottom = 1480
card_radius = 28

# shadow (very subtle, behind card)
shadow_offset = 8
draw.rounded_rectangle(
    [(card_left + shadow_offset, card_top + shadow_offset),
     (card_right + shadow_offset, card_bottom + shadow_offset)],
    radius=card_radius, fill=section_shadow)

# main card
draw.rounded_rectangle(
    [(card_left, card_top), (card_right, card_bottom)],
    radius=card_radius, fill=calendar_card_fill, outline=calendar_card_stroke, width=1)

# 4) Month title area: subtle horizontal divider under the month label region
month_title_y = card_top + 80
draw.line([(card_left + 40, month_title_y), (card_right - 40, month_title_y)], fill=grid_line, width=1)

# 5) Calendar grid separators (horizontal lines between week rows)
# The detected date rows appear around these y positions; we'll draw faint separators between them.
week_rows = [
    card_top + 120,  # space under weekdays row
    card_top + 244,  # row 1
    card_top + 364,  # row 2
    card_top + 484,  # row 3
    card_top + 604,  # row 4
    card_top + 724,  # row 5
]
for y in week_rows:
    draw.line([(card_left + 40, y), (card_right - 40, y)], fill=grid_line, width=1)

# 6) Vertical faint separators between weekday columns (do not draw over interactive cells content)
# Calendar typically has 7 columns; compute x positions based on a left margin within card.
col_left = card_left + 20
col_width = (card_right - card_left - 40) / 7.0
for i in range(1, 7):
    x = int(col_left + i * col_width)
    draw.line([(x, card_top + 120), (x, card_bottom - 40)], fill=grid_line, width=1)

# 7) Additional subtle decorative divider between the top content (start/end date) and calendar card
# This is above the card; place a light divider so the UI groups are visually separated.
draw.line([(40, card_top - 48), (w - 40, card_top - 48)], fill=header_divider, width=1)

# 8) Large empty content area remains white (no drawing) so icons/text can be pasted without duplication

# 9) Subtle top edge underline for the whole screen near the bottom 'Apply date range' control
# Draw a faint separator a little above the detected apply-button area to frame the control without duplicating it.
bottom_control_top = 2768
sep_y = bottom_control_top - 48
draw.line([(32, sep_y), (w - 32, sep_y)], fill=bottom_separator, width=2)

# 10) Left/right side safe margins - thin vertical guides (very faint) to suggest layout bounds
margin_color = (245, 244, 247)
draw.line([(40, header_bottom), (40, h - 200)], fill=margin_color, width=1)
draw.line([(w - 40, header_bottom), (w - 40, h - 200)], fill=margin_color, width=1)

# End of structural/background drawing.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_09_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-11/00_icon_Apply_date_range.png
try:
    _c0 = get_crop(0, 1344, 144)
    canvas.paste(_c0, (48, 2768), _c0)
except Exception:
    pass
layout["Apply_date_range"] = [48, 2768, 1392, 2912]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_09_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-11/01_icon_28.png
try:
    _c1 = get_crop(1, 132, 120)
    canvas.paste(_c1, (60, 1364), _c1)
except Exception:
    pass
layout["28"] = [60, 1364, 192, 1484]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_09_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-11/02_icon_5.23.png
try:
    _c2 = get_crop(2, 60, 63)
    canvas.paste(_c2, (180, 2), _c2)
except Exception:
    pass
layout["5.23"] = [180, 2, 240, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_09_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-11/03_icon_5.23.png
try:
    _c3 = get_crop(3, 58, 65)
    canvas.paste(_c3, (115, 1), _c3)
except Exception:
    pass
layout["5.23"] = [115, 1, 173, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_09_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-11/04_icon_icon_4.png
try:
    _c4 = get_crop(4, 61, 62)
    canvas.paste(_c4, (310, 3), _c4)
except Exception:
    pass
layout["icon_4"] = [310, 3, 371, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_09_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-11/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 50, 60)
    canvas.paste(_c5, (249, 5), _c5)
except Exception:
    pass
layout["icon_5"] = [249, 5, 299, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_09_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-11/06_icon_5.23.png
try:
    _c6 = get_crop(6, 144, 144)
    canvas.paste(_c6, (12, 72), _c6)
except Exception:
    pass
layout["5.23"] = [12, 72, 156, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_09_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-11/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 56, 70)
    canvas.paste(_c7, (1316, 0), _c7)
except Exception:
    pass
layout["icon_7"] = [1316, 0, 1372, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_09_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-11/08_icon_29.png
try:
    _c8 = get_crop(8, 132, 120)
    canvas.paste(_c8, (192, 1364), _c8)
except Exception:
    pass
layout["29"] = [192, 1364, 324, 1484]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_09_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-11/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 80, 69)
    canvas.paste(_c9, (1213, 0), _c9)
except Exception:
    pass
layout["icon_9"] = [1213, 0, 1293, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_09_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-11/10_icon_5.23.png
try:
    _c10 = get_crop(10, 91, 62)
    canvas.paste(_c10, (17, 2), _c10)
except Exception:
    pass
layout["5.23"] = [17, 2, 108, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_09_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-11/11_icon_What_date.png
try:
    _c11 = get_crop(11, 319, 71)
    canvas.paste(_c11, (558, 112), _c11)
except Exception:
    pass
layout["What_date?"] = [558, 112, 877, 183]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_09_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-11/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 43, 65)
    canvas.paste(_c12, (1272, 2), _c12)
except Exception:
    pass
layout["icon_12"] = [1272, 2, 1315, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_09_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-11/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 50, 65)
    canvas.paste(_c13, (382, 2), _c13)
except Exception:
    pass
layout["icon_13"] = [382, 2, 432, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_09_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-11/14_icon_30.png
try:
    _c14 = get_crop(14, 132, 120)
    canvas.paste(_c14, (324, 1364), _c14)
except Exception:
    pass
layout["30"] = [324, 1364, 456, 1484]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_09_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-11/15_icon_Next_month.png
try:
    _c15 = get_crop(15, 144, 144)
    canvas.paste(_c15, (846, 620), _c15)
except Exception:
    pass
layout["Next_month"] = [846, 620, 990, 764]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_09_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-11/16_icon_27.png
try:
    _c16 = get_crop(16, 132, 120)
    canvas.paste(_c16, (852, 1244), _c16)
except Exception:
    pass
layout["27"] = [852, 1244, 984, 1364]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_09_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-11/17_text_Start_Date.png
try:
    _c17 = get_crop(17, 583, 144)
    canvas.paste(_c17, (48, 313), _c17)
except Exception:
    pass
layout["Start_Date"] = [48, 313, 631, 457]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_09_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-11/18_text_End_Date.png
try:
    _c18 = get_crop(18, 638, 114)
    canvas.paste(_c18, (48, 476), _c18)
except Exception:
    pass
layout["End_Date"] = [48, 476, 686, 590]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_09_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-11/19_text_April_2024.png
try:
    _c19 = get_crop(19, 202, 54)
    canvas.paste(_c19, (421, 666), _c19)
except Exception:
    pass
layout["April_2024"] = [421, 666, 623, 720]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_09_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-11/20_text_10.png
try:
    _c20 = get_crop(20, 132, 120)
    canvas.paste(_c20, (456, 1004), _c20)
except Exception:
    pass
layout["10"] = [456, 1004, 588, 1124]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_09_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-11/21_text_11.png
try:
    _c21 = get_crop(21, 132, 120)
    canvas.paste(_c21, (588, 1004), _c21)
except Exception:
    pass
layout["11"] = [588, 1004, 720, 1124]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_09_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-11/22_text_12.png
try:
    _c22 = get_crop(22, 132, 120)
    canvas.paste(_c22, (720, 1004), _c22)
except Exception:
    pass
layout["12"] = [720, 1004, 852, 1124]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_09_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-11/23_text_13.png
try:
    _c23 = get_crop(23, 132, 120)
    canvas.paste(_c23, (852, 1004), _c23)
except Exception:
    pass
layout["13"] = [852, 1004, 984, 1124]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_09_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-11/24_text_14.png
try:
    _c24 = get_crop(24, 132, 120)
    canvas.paste(_c24, (60, 1124), _c24)
except Exception:
    pass
layout["14"] = [60, 1124, 192, 1244]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_09_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-11/25_text_15.png
try:
    _c25 = get_crop(25, 132, 120)
    canvas.paste(_c25, (192, 1124), _c25)
except Exception:
    pass
layout["15"] = [192, 1124, 324, 1244]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_09_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-11/26_text_16.png
try:
    _c26 = get_crop(26, 132, 120)
    canvas.paste(_c26, (324, 1124), _c26)
except Exception:
    pass
layout["16"] = [324, 1124, 456, 1244]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_09_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-11/27_text_17.png
try:
    _c27 = get_crop(27, 132, 120)
    canvas.paste(_c27, (456, 1124), _c27)
except Exception:
    pass
layout["17"] = [456, 1124, 588, 1244]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_09_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-11/28_text_18.png
try:
    _c28 = get_crop(28, 132, 120)
    canvas.paste(_c28, (588, 1124), _c28)
except Exception:
    pass
layout["18"] = [588, 1124, 720, 1244]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_09_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-11/29_text_19.png
try:
    _c29 = get_crop(29, 132, 120)
    canvas.paste(_c29, (720, 1124), _c29)
except Exception:
    pass
layout["19"] = [720, 1124, 852, 1244]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_09_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-11/30_text_20.png
try:
    _c30 = get_crop(30, 132, 120)
    canvas.paste(_c30, (852, 1124), _c30)
except Exception:
    pass
layout["20"] = [852, 1124, 984, 1244]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_09_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-11/31_text_21.png
try:
    _c31 = get_crop(31, 132, 120)
    canvas.paste(_c31, (60, 1244), _c31)
except Exception:
    pass
layout["21"] = [60, 1244, 192, 1364]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_09_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-11/32_text_22.png
try:
    _c32 = get_crop(32, 132, 120)
    canvas.paste(_c32, (192, 1244), _c32)
except Exception:
    pass
layout["22"] = [192, 1244, 324, 1364]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_09_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-11/33_text_23.png
try:
    _c33 = get_crop(33, 132, 120)
    canvas.paste(_c33, (324, 1244), _c33)
except Exception:
    pass
layout["23"] = [324, 1244, 456, 1364]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_09_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-11/34_text_24.png
try:
    _c34 = get_crop(34, 132, 120)
    canvas.paste(_c34, (456, 1244), _c34)
except Exception:
    pass
layout["24"] = [456, 1244, 588, 1364]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_09_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-11/35_text_25.png
try:
    _c35 = get_crop(35, 132, 120)
    canvas.paste(_c35, (588, 1244), _c35)
except Exception:
    pass
layout["25"] = [588, 1244, 720, 1364]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_09_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-11/36_text_26.png
try:
    _c36 = get_crop(36, 132, 120)
    canvas.paste(_c36, (720, 1244), _c36)
except Exception:
    pass
layout["26"] = [720, 1244, 852, 1364]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_09_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-11/37_clickable_1.png
try:
    _c37 = get_crop(37, 132, 120)
    canvas.paste(_c37, (192, 884), _c37)
except Exception:
    pass
layout["1"] = [192, 884, 324, 1004]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_09_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-11/38_clickable_2.png
try:
    _c38 = get_crop(38, 132, 120)
    canvas.paste(_c38, (324, 884), _c38)
except Exception:
    pass
layout["2"] = [324, 884, 456, 1004]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_09_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-11/39_clickable_3.png
try:
    _c39 = get_crop(39, 132, 120)
    canvas.paste(_c39, (456, 884), _c39)
except Exception:
    pass
layout["3"] = [456, 884, 588, 1004]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_09_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-11/40_clickable_4.png
try:
    _c40 = get_crop(40, 132, 120)
    canvas.paste(_c40, (588, 884), _c40)
except Exception:
    pass
layout["4"] = [588, 884, 720, 1004]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_09_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-11/41_clickable_5.png
try:
    _c41 = get_crop(41, 132, 120)
    canvas.paste(_c41, (720, 884), _c41)
except Exception:
    pass
layout["5"] = [720, 884, 852, 1004]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_09_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-11/42_clickable_6.png
try:
    _c42 = get_crop(42, 132, 120)
    canvas.paste(_c42, (852, 884), _c42)
except Exception:
    pass
layout["6"] = [852, 884, 984, 1004]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_09_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-11/43_clickable_7.png
try:
    _c43 = get_crop(43, 132, 120)
    canvas.paste(_c43, (60, 1004), _c43)
except Exception:
    pass
layout["7"] = [60, 1004, 192, 1124]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_09_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-11/44_clickable_8.png
try:
    _c44 = get_crop(44, 132, 120)
    canvas.paste(_c44, (192, 1004), _c44)
except Exception:
    pass
layout["8"] = [192, 1004, 324, 1124]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_09_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-11/45_clickable_9.png
try:
    _c45 = get_crop(45, 132, 120)
    canvas.paste(_c45, (324, 1004), _c45)
except Exception:
    pass
layout["9"] = [324, 1004, 456, 1124]
