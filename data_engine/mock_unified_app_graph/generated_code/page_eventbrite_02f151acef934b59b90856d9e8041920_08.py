# page_id: page_eventbrite_02f151acef934b59b90856d9e8041920_08
# screenshot: 2024_4_24_17_24_02f151acef934b59b90856d9e8041920-10.png
# step_index: 8/11
# task: Open Eventbrite. Check the "Tech" events happening this month. Open the first event and check its date and time.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw background and structural UI elements for the calendar screen
# Uses provided variables: canvas (PIL Image) and draw (ImageDraw), fonts available but unused.

# Colors
bg_white = (255, 255, 255)
status_bar_gray = (232, 232, 232)
header_divider = (235, 228, 245)   # very light purple divider
card_bg = (250, 250, 252)          # subtle off-white card
card_border = (228, 224, 235)      # light border for cards
muted_divider = (240, 240, 244)
shadow_line = (220, 220, 225)

W, H = canvas.size

# Base background (canvas starts white, but fill explicitly)
draw.rectangle((0, 0, W, H), fill=bg_white)

# Status bar area (top) - a subtle light gray strip behind status icons
status_h = 72
draw.rectangle((0, 0, W, status_h), fill=status_bar_gray)

# Header / toolbar area below status bar
header_top = status_h
header_bottom = 160
draw.rectangle((0, header_top, W, header_bottom), fill=bg_white)

# Subtle bottom divider for header
draw.line((36, header_bottom, W-36, header_bottom), fill=header_divider, width=1)

# Top content card (group behind "Start Date" / "End Date")
# Rounded rectangle card to visually group the date selection header portion
card_left = 48
card_right = W - 48
card_top = 232
card_bottom = 620
draw.rounded_rectangle(
    (card_left, card_top, card_right, card_bottom),
    radius=18,
    fill=card_bg,
    outline=card_border,
    width=2
)

# Thin subtle divider below the card to separate calendar area
divider_y = card_bottom + 24
draw.line((card_left + 8, divider_y, card_right - 8, divider_y), fill=muted_divider, width=1)

# Calendar panel area background (light, almost white) to anchor the month & grid
cal_top = divider_y + 20
cal_bottom = 1400
draw.rectangle((card_left, cal_top, card_right, cal_bottom), fill=bg_white, outline=None)

# Light guide lines for calendar grouping (do not draw any numbers or icons)
# Horizontal helper lines to subtly structure the calendar rows
row_height = 96
for i in range(1, 6):
    y = cal_top + i * row_height
    # stop drawing lines if they go too far down
    if y < cal_bottom:
        draw.line((card_left + 12, y, card_right - 12, y), fill=(247,247,249), width=1)

# Subtle centered month row separator to emphasize month label area
month_row_y = cal_top + 24
draw.line((card_left + 120, month_row_y + 40, card_right - 120, month_row_y + 40), fill=shadow_line, width=1)

# Right-side subtle chevron area (visual divider for next-month icon placement)
# Draw a faint vertical guide (not an icon) so pasted chevron sits on a background
chev_x = card_right - 120
draw.line((chev_x, cal_top + 16, chev_x, cal_top + 96), fill=(246,246,247), width=1)

# Large empty content area remains white below calendar (no drawing to avoid duplicating future content)
content_top = cal_bottom + 20
# draw a faint horizontal separator above bottom action region (keep above detected button)
footer_sep_y = 2720
draw.line((48, footer_sep_y, W-48, footer_sep_y), fill=(238,238,242), width=2)

# Add a very subtle inset border on the screen edges to match UI safe area
edge_pad = 24
draw.rounded_rectangle((edge_pad, edge_pad, W-edge_pad, H-edge_pad-220), radius=6, outline=(249,249,250), width=1)

# NOTE: All actual icons, texts and the bottom "Apply date range" button are intentionally NOT drawn here.
# This code only draws backgrounds, cards, dividers and structural elements.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_08_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-10/00_icon_Apply_date_range.png
try:
    _c0 = get_crop(0, 1344, 144)
    canvas.paste(_c0, (48, 2768), _c0)
except Exception:
    pass
layout["Apply_date_range"] = [48, 2768, 1392, 2912]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_08_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-10/01_icon_24.png
try:
    _c1 = get_crop(1, 132, 120)
    canvas.paste(_c1, (456, 1244), _c1)
except Exception:
    pass
layout["24"] = [456, 1244, 588, 1364]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_08_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-10/02_icon_28.png
try:
    _c2 = get_crop(2, 132, 120)
    canvas.paste(_c2, (60, 1364), _c2)
except Exception:
    pass
layout["28"] = [60, 1364, 192, 1484]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_08_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-10/03_icon_25.png
try:
    _c3 = get_crop(3, 132, 120)
    canvas.paste(_c3, (588, 1244), _c3)
except Exception:
    pass
layout["25"] = [588, 1244, 720, 1364]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_08_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-10/04_icon_30.png
try:
    _c4 = get_crop(4, 132, 120)
    canvas.paste(_c4, (324, 1364), _c4)
except Exception:
    pass
layout["30"] = [324, 1364, 456, 1484]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_08_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-10/05_icon_29.png
try:
    _c5 = get_crop(5, 132, 120)
    canvas.paste(_c5, (192, 1364), _c5)
except Exception:
    pass
layout["29"] = [192, 1364, 324, 1484]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_08_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-10/06_icon_26.png
try:
    _c6 = get_crop(6, 132, 120)
    canvas.paste(_c6, (720, 1244), _c6)
except Exception:
    pass
layout["26"] = [720, 1244, 852, 1364]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_08_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-10/07_icon_5.25.png
try:
    _c7 = get_crop(7, 58, 63)
    canvas.paste(_c7, (181, 2), _c7)
except Exception:
    pass
layout["5.25"] = [181, 2, 239, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_08_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-10/08_icon_5.25.png
try:
    _c8 = get_crop(8, 59, 64)
    canvas.paste(_c8, (115, 2), _c8)
except Exception:
    pass
layout["5.25"] = [115, 2, 174, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_08_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-10/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 61, 62)
    canvas.paste(_c9, (310, 3), _c9)
except Exception:
    pass
layout["icon_9"] = [310, 3, 371, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_08_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-10/10_icon_27.png
try:
    _c10 = get_crop(10, 132, 120)
    canvas.paste(_c10, (852, 1244), _c10)
except Exception:
    pass
layout["27"] = [852, 1244, 984, 1364]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_08_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-10/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 49, 60)
    canvas.paste(_c11, (249, 5), _c11)
except Exception:
    pass
layout["icon_11"] = [249, 5, 298, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_08_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-10/12_icon_5.25.png
try:
    _c12 = get_crop(12, 144, 144)
    canvas.paste(_c12, (12, 72), _c12)
except Exception:
    pass
layout["5.25"] = [12, 72, 156, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_08_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-10/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 56, 70)
    canvas.paste(_c13, (1316, 0), _c13)
except Exception:
    pass
layout["icon_13"] = [1316, 0, 1372, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_08_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-10/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 81, 69)
    canvas.paste(_c14, (1212, 0), _c14)
except Exception:
    pass
layout["icon_14"] = [1212, 0, 1293, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_08_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-10/15_icon_What_date.png
try:
    _c15 = get_crop(15, 319, 72)
    canvas.paste(_c15, (558, 111), _c15)
except Exception:
    pass
layout["What_date?"] = [558, 111, 877, 183]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_08_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-10/16_icon_21.png
try:
    _c16 = get_crop(16, 132, 120)
    canvas.paste(_c16, (60, 1244), _c16)
except Exception:
    pass
layout["21"] = [60, 1244, 192, 1364]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_08_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-10/17_icon_icon_17.png
try:
    _c17 = get_crop(17, 43, 66)
    canvas.paste(_c17, (1272, 1), _c17)
except Exception:
    pass
layout["icon_17"] = [1272, 1, 1315, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_08_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-10/18_icon_22.png
try:
    _c18 = get_crop(18, 132, 120)
    canvas.paste(_c18, (192, 1244), _c18)
except Exception:
    pass
layout["22"] = [192, 1244, 324, 1364]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_08_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-10/19_icon_Next_month.png
try:
    _c19 = get_crop(19, 144, 144)
    canvas.paste(_c19, (846, 620), _c19)
except Exception:
    pass
layout["Next_month"] = [846, 620, 990, 764]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_08_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-10/20_icon_icon_20.png
try:
    _c20 = get_crop(20, 50, 65)
    canvas.paste(_c20, (382, 2), _c20)
except Exception:
    pass
layout["icon_20"] = [382, 2, 432, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_08_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-10/21_icon_23.png
try:
    _c21 = get_crop(21, 132, 120)
    canvas.paste(_c21, (324, 1244), _c21)
except Exception:
    pass
layout["23"] = [324, 1244, 456, 1364]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_08_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-10/22_icon_5.25.png
try:
    _c22 = get_crop(22, 91, 61)
    canvas.paste(_c22, (17, 3), _c22)
except Exception:
    pass
layout["5.25"] = [17, 3, 108, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_08_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-10/23_icon_12.png
try:
    _c23 = get_crop(23, 132, 120)
    canvas.paste(_c23, (720, 884), _c23)
except Exception:
    pass
layout["12"] = [720, 884, 852, 1004]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_08_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-10/24_icon_icon_24.png
try:
    _c24 = get_crop(24, 101, 102)
    canvas.paste(_c24, (72, 781), _c24)
except Exception:
    pass
layout["icon_24"] = [72, 781, 173, 883]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_08_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-10/25_text_Start_Date.png
try:
    _c25 = get_crop(25, 589, 144)
    canvas.paste(_c25, (48, 313), _c25)
except Exception:
    pass
layout["Start_Date"] = [48, 313, 637, 457]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_08_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-10/26_text_End_Date.png
try:
    _c26 = get_crop(26, 638, 114)
    canvas.paste(_c26, (48, 476), _c26)
except Exception:
    pass
layout["End_Date"] = [48, 476, 686, 590]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_08_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-10/27_text_April_2024.png
try:
    _c27 = get_crop(27, 202, 54)
    canvas.paste(_c27, (421, 666), _c27)
except Exception:
    pass
layout["April_2024"] = [421, 666, 623, 720]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_08_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-10/28_text_10.png
try:
    _c28 = get_crop(28, 132, 120)
    canvas.paste(_c28, (456, 1004), _c28)
except Exception:
    pass
layout["10"] = [456, 1004, 588, 1124]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_08_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-10/29_text_11.png
try:
    _c29 = get_crop(29, 132, 120)
    canvas.paste(_c29, (588, 1004), _c29)
except Exception:
    pass
layout["11"] = [588, 1004, 720, 1124]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_08_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-10/30_text_12.png
try:
    _c30 = get_crop(30, 132, 120)
    canvas.paste(_c30, (720, 1004), _c30)
except Exception:
    pass
layout["12"] = [720, 1004, 852, 1124]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_08_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-10/31_text_13.png
try:
    _c31 = get_crop(31, 132, 120)
    canvas.paste(_c31, (852, 1004), _c31)
except Exception:
    pass
layout["13"] = [852, 1004, 984, 1124]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_08_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-10/32_text_14.png
try:
    _c32 = get_crop(32, 132, 120)
    canvas.paste(_c32, (60, 1124), _c32)
except Exception:
    pass
layout["14"] = [60, 1124, 192, 1244]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_08_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-10/33_text_15.png
try:
    _c33 = get_crop(33, 132, 120)
    canvas.paste(_c33, (192, 1124), _c33)
except Exception:
    pass
layout["15"] = [192, 1124, 324, 1244]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_08_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-10/34_text_16.png
try:
    _c34 = get_crop(34, 132, 120)
    canvas.paste(_c34, (324, 1124), _c34)
except Exception:
    pass
layout["16"] = [324, 1124, 456, 1244]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_08_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-10/35_text_17.png
try:
    _c35 = get_crop(35, 132, 120)
    canvas.paste(_c35, (456, 1124), _c35)
except Exception:
    pass
layout["17"] = [456, 1124, 588, 1244]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_08_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-10/36_text_18.png
try:
    _c36 = get_crop(36, 132, 120)
    canvas.paste(_c36, (588, 1124), _c36)
except Exception:
    pass
layout["18"] = [588, 1124, 720, 1244]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_08_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-10/37_text_19.png
try:
    _c37 = get_crop(37, 132, 120)
    canvas.paste(_c37, (720, 1124), _c37)
except Exception:
    pass
layout["19"] = [720, 1124, 852, 1244]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_08_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-10/38_text_20.png
try:
    _c38 = get_crop(38, 132, 120)
    canvas.paste(_c38, (852, 1124), _c38)
except Exception:
    pass
layout["20"] = [852, 1124, 984, 1244]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_08_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-10/39_clickable_1.png
try:
    _c39 = get_crop(39, 132, 120)
    canvas.paste(_c39, (192, 884), _c39)
except Exception:
    pass
layout["1"] = [192, 884, 324, 1004]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_08_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-10/40_clickable_2.png
try:
    _c40 = get_crop(40, 132, 120)
    canvas.paste(_c40, (324, 884), _c40)
except Exception:
    pass
layout["2"] = [324, 884, 456, 1004]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_08_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-10/41_clickable_3.png
try:
    _c41 = get_crop(41, 132, 120)
    canvas.paste(_c41, (456, 884), _c41)
except Exception:
    pass
layout["3"] = [456, 884, 588, 1004]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_08_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-10/42_clickable_4.png
try:
    _c42 = get_crop(42, 132, 120)
    canvas.paste(_c42, (588, 884), _c42)
except Exception:
    pass
layout["4"] = [588, 884, 720, 1004]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_08_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-10/43_clickable_6.png
try:
    _c43 = get_crop(43, 132, 120)
    canvas.paste(_c43, (852, 884), _c43)
except Exception:
    pass
layout["6"] = [852, 884, 984, 1004]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_08_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-10/44_clickable_7.png
try:
    _c44 = get_crop(44, 132, 120)
    canvas.paste(_c44, (60, 1004), _c44)
except Exception:
    pass
layout["7"] = [60, 1004, 192, 1124]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_08_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-10/45_clickable_8.png
try:
    _c45 = get_crop(45, 132, 120)
    canvas.paste(_c45, (192, 1004), _c45)
except Exception:
    pass
layout["8"] = [192, 1004, 324, 1124]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_08_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-10/46_clickable_9.png
try:
    _c46 = get_crop(46, 132, 120)
    canvas.paste(_c46, (324, 1004), _c46)
except Exception:
    pass
layout["9"] = [324, 1004, 456, 1124]
