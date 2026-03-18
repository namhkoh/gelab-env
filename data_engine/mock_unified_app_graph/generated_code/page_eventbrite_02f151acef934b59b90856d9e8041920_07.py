# page_id: page_eventbrite_02f151acef934b59b90856d9e8041920_07
# screenshot: 2024_4_24_17_24_02f151acef934b59b90856d9e8041920-9.png
# step_index: 7/11
# task: Open Eventbrite. Check the "Tech" events happening this month. Open the first event and check its date and time.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw background and structural UI elements for the calendar page.
# Available variables:
# - canvas: PIL Image (1440x2960 RGB)
# - draw: PIL ImageDraw object
# - font_sm, font_md, font_lg, font_xl (not used)

# Full background (ensure clean white)
draw.rectangle([(0, 0), (1440, 2960)], fill="#FFFFFF")

# Status bar area at the very top (light gray background)
status_h = 80
draw.rectangle([(0, 0), (1440, status_h)], fill="#D0D0D0")

# Thin top inner divider under status bar
draw.line([(0, status_h), (1440, status_h)], fill="#C8C6CC", width=1)

# Header/toolbar area (keeps white but with a subtle divider)
header_top = status_h
header_bottom = 220
draw.rectangle([(0, header_top), (1440, header_bottom)], fill="#FFFFFF")
# Divider under header
draw.line([(32, header_bottom), (1440-32, header_bottom)], fill="#EDECF2", width=1)

# Calendar card shadow + rounded card background
card_x0 = 48
card_x1 = 1440 - 48
card_y0 = 240
card_y1 = 1320
card_radius = 22

# shadow (slight offset)
shadow_offset = 8
draw.rounded_rectangle(
    [(card_x0 + shadow_offset, card_y0 + shadow_offset),
     (card_x1 + shadow_offset, card_y1 + shadow_offset)],
    radius=card_radius,
    fill="#F5F5F7",
    outline=None
)

# main calendar card (white with pale border)
draw.rounded_rectangle(
    [(card_x0, card_y0), (card_x1, card_y1)],
    radius=card_radius,
    fill="#FFFFFF",
    outline="#EDEAF0",
    width=1
)

# Month header divider inside the card (subtle horizontal rule where month label area would be)
month_header_y = card_y0 + 80
draw.line([(card_x0 + 24, month_header_y), (card_x1 - 24, month_header_y)], fill="#F0EDF4", width=1)

# Calendar grid skeleton (weekday row + 5 rows of date slots)
grid_top = month_header_y + 36
grid_bottom = card_y1 - 60
grid_left = card_x0 + 40
grid_right = card_x1 - 40

# Draw 6 horizontal separators (for weekday labels + 5 rows)
rows = 6  # top for weekday labels + 5 weeks
row_height = (grid_bottom - grid_top) / rows
for i in range(rows + 1):
    y = int(grid_top + i * row_height)
    draw.line([(grid_left, y), (grid_right, y)], fill="#F4F3F8", width=1)

# Draw 7 vertical separators for 7 days
cols = 7
col_width = (grid_right - grid_left) / cols
for j in range(cols + 1):
    x = int(grid_left + j * col_width)
    draw.line([(x, grid_top), (x, grid_bottom)], fill="#F4F3F8", width=1)

# Faint weekday labels background row (no text)
weekday_bg_top = grid_top - int(row_height)
weekday_bg_bottom = grid_top
draw.rectangle([(grid_left, weekday_bg_top), (grid_right, weekday_bg_bottom)], fill="#FFFFFF")

# Separator between calendar card and "End Date" section
sep_y = card_y1 + 40
draw.line([(32, sep_y), (1440-32, sep_y)], fill="#F0EDF4", width=1)

# End Date card area (subtle rounded area / placeholder background)
end_x0 = 48
end_x1 = 1440 - 48
end_y0 = sep_y + 24
end_y1 = end_y0 + 360
end_radius = 20
draw.rounded_rectangle(
    [(end_x0, end_y0), (end_x1, end_y1)],
    radius=end_radius,
    fill="#FFFFFF",
    outline="#EDEAF0",
    width=1
)

# Add a faint horizontal guideline inside the End Date card to suggest content region
draw.line([(end_x0 + 24, end_y0 + 110), (end_x1 - 24, end_y0 + 110)], fill="#F4F3F8", width=1)

# Large empty content area remains white (no text or icons drawn)

# Footer divider above the bottom action area (do not draw the button itself)
footer_divider_y = 2700
draw.line([(24, footer_divider_y), (1440-24, footer_divider_y)], fill="#E9E7EE", width=2)

# A subtle background band behind the bottom action area (keeps structure but not button)
footer_band_top = footer_divider_y + 8
footer_band_bottom = footer_divider_y + 120
draw.rounded_rectangle(
    [(32, footer_band_top), (1440-32, footer_band_bottom)],
    radius=14,
    fill="#FFFFFF",
    outline="#E6E3EA",
    width=1
)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_07_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-9/00_icon_24.png
try:
    _c0 = get_crop(0, 132, 120)
    canvas.paste(_c0, (456, 1081), _c0)
except Exception:
    pass
layout["24"] = [456, 1081, 588, 1201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_07_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-9/01_icon_Apply_date_range.png
try:
    _c1 = get_crop(1, 1344, 144)
    canvas.paste(_c1, (48, 2768), _c1)
except Exception:
    pass
layout["Apply_date_range"] = [48, 2768, 1392, 2912]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_07_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-9/02_icon_28.png
try:
    _c2 = get_crop(2, 132, 120)
    canvas.paste(_c2, (60, 1201), _c2)
except Exception:
    pass
layout["28"] = [60, 1201, 192, 1321]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_07_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-9/03_icon_29.png
try:
    _c3 = get_crop(3, 132, 120)
    canvas.paste(_c3, (192, 1201), _c3)
except Exception:
    pass
layout["29"] = [192, 1201, 324, 1321]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_07_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-9/04_icon_23.png
try:
    _c4 = get_crop(4, 132, 120)
    canvas.paste(_c4, (324, 1081), _c4)
except Exception:
    pass
layout["23"] = [324, 1081, 456, 1201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_07_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-9/05_icon_25.png
try:
    _c5 = get_crop(5, 132, 120)
    canvas.paste(_c5, (588, 1081), _c5)
except Exception:
    pass
layout["25"] = [588, 1081, 720, 1201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_07_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-9/06_icon_30.png
try:
    _c6 = get_crop(6, 132, 120)
    canvas.paste(_c6, (324, 1201), _c6)
except Exception:
    pass
layout["30"] = [324, 1201, 456, 1321]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_07_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-9/07_icon_27.png
try:
    _c7 = get_crop(7, 132, 120)
    canvas.paste(_c7, (852, 1081), _c7)
except Exception:
    pass
layout["27"] = [852, 1081, 984, 1201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_07_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-9/08_icon_26.png
try:
    _c8 = get_crop(8, 132, 120)
    canvas.paste(_c8, (720, 1081), _c8)
except Exception:
    pass
layout["26"] = [720, 1081, 852, 1201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_07_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-9/09_icon_5.25.png
try:
    _c9 = get_crop(9, 62, 66)
    canvas.paste(_c9, (179, 1), _c9)
except Exception:
    pass
layout["5.25"] = [179, 1, 241, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_07_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-9/10_icon_5.25.png
try:
    _c10 = get_crop(10, 62, 66)
    canvas.paste(_c10, (113, 1), _c10)
except Exception:
    pass
layout["5.25"] = [113, 1, 175, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_07_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-9/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 63, 64)
    canvas.paste(_c11, (309, 2), _c11)
except Exception:
    pass
layout["icon_11"] = [309, 2, 372, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_07_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-9/12_icon_21.png
try:
    _c12 = get_crop(12, 132, 120)
    canvas.paste(_c12, (60, 1081), _c12)
except Exception:
    pass
layout["21"] = [60, 1081, 192, 1201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_07_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-9/13_icon_22.png
try:
    _c13 = get_crop(13, 132, 120)
    canvas.paste(_c13, (192, 1081), _c13)
except Exception:
    pass
layout["22"] = [192, 1081, 324, 1201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_07_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-9/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 52, 64)
    canvas.paste(_c14, (248, 2), _c14)
except Exception:
    pass
layout["icon_14"] = [248, 2, 300, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_07_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-9/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 57, 70)
    canvas.paste(_c15, (1316, 0), _c15)
except Exception:
    pass
layout["icon_15"] = [1316, 0, 1373, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_07_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-9/16_icon_18.png
try:
    _c16 = get_crop(16, 132, 120)
    canvas.paste(_c16, (588, 961), _c16)
except Exception:
    pass
layout["18"] = [588, 961, 720, 1081]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_07_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-9/17_icon_5.25.png
try:
    _c17 = get_crop(17, 144, 144)
    canvas.paste(_c17, (12, 72), _c17)
except Exception:
    pass
layout["5.25"] = [12, 72, 156, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_07_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-9/18_icon_icon_18.png
try:
    _c18 = get_crop(18, 90, 69)
    canvas.paste(_c18, (1211, 0), _c18)
except Exception:
    pass
layout["icon_18"] = [1211, 0, 1301, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_07_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-9/19_icon_11.png
try:
    _c19 = get_crop(19, 132, 120)
    canvas.paste(_c19, (588, 721), _c19)
except Exception:
    pass
layout["11"] = [588, 721, 720, 841]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_07_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-9/20_icon_19.png
try:
    _c20 = get_crop(20, 132, 120)
    canvas.paste(_c20, (720, 961), _c20)
except Exception:
    pass
layout["19"] = [720, 961, 852, 1081]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_07_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-9/21_icon_icon_21.png
try:
    _c21 = get_crop(21, 49, 67)
    canvas.paste(_c21, (382, 1), _c21)
except Exception:
    pass
layout["icon_21"] = [382, 1, 431, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_07_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-9/22_icon_icon_22.png
try:
    _c22 = get_crop(22, 41, 65)
    canvas.paste(_c22, (1274, 0), _c22)
except Exception:
    pass
layout["icon_22"] = [1274, 0, 1315, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_07_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-9/23_icon_April_2024.png
try:
    _c23 = get_crop(23, 126, 110)
    canvas.paste(_c23, (593, 611), _c23)
except Exception:
    pass
layout["April_2024"] = [593, 611, 719, 721]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_07_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-9/24_icon_Next_month.png
try:
    _c24 = get_crop(24, 144, 144)
    canvas.paste(_c24, (846, 457), _c24)
except Exception:
    pass
layout["Next_month"] = [846, 457, 990, 601]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_07_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-9/25_icon_12.png
try:
    _c25 = get_crop(25, 132, 120)
    canvas.paste(_c25, (720, 721), _c25)
except Exception:
    pass
layout["12"] = [720, 721, 852, 841]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_07_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-9/26_icon_12.png
try:
    _c26 = get_crop(26, 104, 107)
    canvas.paste(_c26, (733, 614), _c26)
except Exception:
    pass
layout["12"] = [733, 614, 837, 721]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_07_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-9/27_icon_Choose_a_date.png
try:
    _c27 = get_crop(27, 638, 144)
    canvas.paste(_c27, (48, 1490), _c27)
except Exception:
    pass
layout["Choose_a_date"] = [48, 1490, 686, 1634]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_07_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-9/28_icon_icon_28.png
try:
    _c28 = get_crop(28, 104, 100)
    canvas.paste(_c28, (71, 618), _c28)
except Exception:
    pass
layout["icon_28"] = [71, 618, 175, 718]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_07_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-9/29_icon_What_date.png
try:
    _c29 = get_crop(29, 322, 71)
    canvas.paste(_c29, (558, 113), _c29)
except Exception:
    pass
layout["What_date?"] = [558, 113, 880, 184]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_07_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-9/30_icon_16.png
try:
    _c30 = get_crop(30, 132, 120)
    canvas.paste(_c30, (324, 961), _c30)
except Exception:
    pass
layout["16"] = [324, 961, 456, 1081]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_07_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-9/31_icon_5.25.png
try:
    _c31 = get_crop(31, 92, 63)
    canvas.paste(_c31, (16, 2), _c31)
except Exception:
    pass
layout["5.25"] = [16, 2, 108, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_07_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-9/32_icon_10.png
try:
    _c32 = get_crop(32, 132, 120)
    canvas.paste(_c32, (324, 721), _c32)
except Exception:
    pass
layout["10"] = [324, 721, 456, 841]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_07_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-9/33_text_Start_Date.png
try:
    _c33 = get_crop(33, 589, 114)
    canvas.paste(_c33, (48, 313), _c33)
except Exception:
    pass
layout["Start_Date"] = [48, 313, 637, 427]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_07_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-9/34_text_April_2024.png
try:
    _c34 = get_crop(34, 203, 54)
    canvas.paste(_c34, (420, 504), _c34)
except Exception:
    pass
layout["April_2024"] = [420, 504, 623, 558]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_07_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-9/35_text_10.png
try:
    _c35 = get_crop(35, 132, 120)
    canvas.paste(_c35, (456, 841), _c35)
except Exception:
    pass
layout["10"] = [456, 841, 588, 961]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_07_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-9/36_text_11.png
try:
    _c36 = get_crop(36, 132, 120)
    canvas.paste(_c36, (588, 841), _c36)
except Exception:
    pass
layout["11"] = [588, 841, 720, 961]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_07_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-9/37_text_12.png
try:
    _c37 = get_crop(37, 132, 120)
    canvas.paste(_c37, (720, 841), _c37)
except Exception:
    pass
layout["12"] = [720, 841, 852, 961]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_07_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-9/38_text_13.png
try:
    _c38 = get_crop(38, 132, 120)
    canvas.paste(_c38, (852, 841), _c38)
except Exception:
    pass
layout["13"] = [852, 841, 984, 961]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_07_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-9/39_text_14.png
try:
    _c39 = get_crop(39, 132, 120)
    canvas.paste(_c39, (60, 961), _c39)
except Exception:
    pass
layout["14"] = [60, 961, 192, 1081]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_07_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-9/40_text_15.png
try:
    _c40 = get_crop(40, 132, 120)
    canvas.paste(_c40, (192, 961), _c40)
except Exception:
    pass
layout["15"] = [192, 961, 324, 1081]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_07_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-9/41_text_17.png
try:
    _c41 = get_crop(41, 132, 120)
    canvas.paste(_c41, (456, 961), _c41)
except Exception:
    pass
layout["17"] = [456, 961, 588, 1081]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_07_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-9/42_text_20.png
try:
    _c42 = get_crop(42, 132, 120)
    canvas.paste(_c42, (852, 961), _c42)
except Exception:
    pass
layout["20"] = [852, 961, 984, 1081]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_07_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-9/43_clickable_1.png
try:
    _c43 = get_crop(43, 132, 120)
    canvas.paste(_c43, (192, 721), _c43)
except Exception:
    pass
layout["1"] = [192, 721, 324, 841]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_07_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-9/44_clickable_3.png
try:
    _c44 = get_crop(44, 132, 120)
    canvas.paste(_c44, (456, 721), _c44)
except Exception:
    pass
layout["3"] = [456, 721, 588, 841]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_07_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-9/45_clickable_6.png
try:
    _c45 = get_crop(45, 132, 120)
    canvas.paste(_c45, (852, 721), _c45)
except Exception:
    pass
layout["6"] = [852, 721, 984, 841]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_07_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-9/46_clickable_7.png
try:
    _c46 = get_crop(46, 132, 120)
    canvas.paste(_c46, (60, 841), _c46)
except Exception:
    pass
layout["7"] = [60, 841, 192, 961]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_07_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-9/47_clickable_8.png
try:
    _c47 = get_crop(47, 132, 120)
    canvas.paste(_c47, (192, 841), _c47)
except Exception:
    pass
layout["8"] = [192, 841, 324, 961]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_07_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-9/48_clickable_9.png
try:
    _c48 = get_crop(48, 132, 120)
    canvas.paste(_c48, (324, 841), _c48)
except Exception:
    pass
layout["9"] = [324, 841, 456, 961]
