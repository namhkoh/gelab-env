# page_id: page_eventbrite_a9f633a394e74f78843aa30bd2792346_14
# screenshot: 2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-16.png
# step_index: 14/18
# task: Open Eventbrite. Set the city to "Los Angeles". Look for Photography workshops happening next week. What is the price of the tickets for first non-promoted event?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Background and structural UI elements for 1440x2960 canvas.
# Variables provided: canvas (PIL Image), draw (ImageDraw), font_sm, font_md, font_lg, font_xl

W, H = canvas.size

# 1) Base background (dominant white)
draw.rectangle((0, 0, W, H), fill="#FFFFFF")

# 2) Status bar area at top (~72px high) - subtle grey
status_h = 72
draw.rectangle((0, 0, W, status_h), fill="#D0D0D3")

# 3) Header area (below status) with subtle bottom divider
header_top = status_h
header_bottom = 180
draw.rectangle((0, header_top, W, header_bottom), fill="#FFFFFF")
# bottom divider line
draw.line((40, header_bottom, W-40, header_bottom), fill="#E9E6EE", width=2)

# 4) Main content card for date selection (rounded rect)
card_left = 48
card_right = W - 48
card_top = 200
card_bottom = 1400
card_radius = 22
draw.rounded_rectangle(
    (card_left, card_top, card_right, card_bottom),
    radius=card_radius,
    fill="#FBFBFF",
    outline="#EFEAF3",
    width=1
)
# subtle drop shadow below the card
shadow_top = card_bottom
shadow_bottom = card_bottom + 18
draw.rectangle((card_left+8, shadow_top, card_right-8, shadow_bottom), fill="#F4F4F6")

# 5) Calendar grid background area inside the card - faint tinted band for the grid region
grid_area_top = 620
grid_area_bottom = 1320
grid_inset = 40
draw.rectangle(
    (card_left + grid_inset, grid_area_top, card_right - grid_inset, grid_area_bottom),
    fill="#FFFFFF"
)

# 6) Calendar grid lines (faint) - match layout of 7 columns x multiple rows
# Column layout inferred from detected positions: start x ~60, cell width ~132
grid_start_x = 60
cell_w = 132
cols = 7
# vertical grid lines
for i in range(cols + 1):
    x = grid_start_x + i * cell_w
    # only draw inside the card
    if card_left + grid_inset <= x <= card_right - grid_inset:
        draw.line(
            (x, grid_area_top + 20, x, grid_area_bottom - 20),
            fill="#F0EDF6",
            width=1
        )

# horizontal grid lines for weekly rows (approx)
row_height = 120
rows = 5
for r in range(rows + 1):
    y = (grid_area_top + 20) + r * row_height
    if grid_area_top + 20 <= y <= grid_area_bottom - 20:
        draw.line(
            (card_left + grid_inset, y, card_right - grid_inset, y),
            fill="#F0EDF6",
            width=1
        )

# 7) Month header area divider (centered above grid) - subtle
month_header_y = 666
draw.line((card_left + 120, month_header_y + 40, card_right - 120, month_header_y + 40), fill="#FFFFFF", width=8)
# small decorative dot/chevron background (no icon)
chev_box_w = 44
draw.rectangle((card_right - 200, month_header_y + 10, card_right - 200 + chev_box_w, month_header_y + 10 + chev_box_w), fill="#FFFFFF")

# 8) Large empty content area below calendar left as white (no content duplication)
# No draw needed because background is already white; but add a very light vertical guideline on left side to hint structure
draw.line((card_left, card_bottom + 24, card_left, H - 240), fill="#FFFFFF", width=1)

# 9) Bottom control area background & top separator (behind "Apply date range" control)
bottom_control_top = 2680
draw.rectangle((0, bottom_control_top, W, H), fill="#FFFFFF")
# top separator shadow line
draw.line((40, bottom_control_top, W-40, bottom_control_top), fill="#E6E3E8", width=2)
# subtle inner border to suggest a recessed control area (rounded)
control_margin = 40
control_rect = (control_margin, bottom_control_top + 12, W - control_margin, bottom_control_top + 12 + 160)
draw.rounded_rectangle(control_rect, radius=12, fill="#FFFFFF", outline="#D8D6DB", width=2)

# 10) Overall subtle edge framing to match app look
draw.rectangle((0, 0, W-1, H-1), outline="#FFFFFF")

# End of structural/background drawing

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_14_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-16/00_icon_Apply_date_range.png
try:
    _c0 = get_crop(0, 1344, 144)
    canvas.paste(_c0, (48, 2768), _c0)
except Exception:
    pass
layout["Apply_date_range"] = [48, 2768, 1392, 2912]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_14_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-16/01_icon_28.png
try:
    _c1 = get_crop(1, 132, 120)
    canvas.paste(_c1, (60, 1364), _c1)
except Exception:
    pass
layout["28"] = [60, 1364, 192, 1484]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_14_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-16/02_icon_icon_2.png
try:
    _c2 = get_crop(2, 50, 71)
    canvas.paste(_c2, (1154, 0), _c2)
except Exception:
    pass
layout["icon_2"] = [1154, 0, 1204, 71]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_14_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-16/03_icon_4.51.png
try:
    _c3 = get_crop(3, 59, 64)
    canvas.paste(_c3, (181, 1), _c3)
except Exception:
    pass
layout["4.51"] = [181, 1, 240, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_14_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-16/04_icon_4.51.png
try:
    _c4 = get_crop(4, 61, 65)
    canvas.paste(_c4, (113, 1), _c4)
except Exception:
    pass
layout["4.51"] = [113, 1, 174, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_14_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-16/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 61, 62)
    canvas.paste(_c5, (310, 3), _c5)
except Exception:
    pass
layout["icon_5"] = [310, 3, 371, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_14_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-16/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 100, 71)
    canvas.paste(_c6, (1210, 0), _c6)
except Exception:
    pass
layout["icon_6"] = [1210, 0, 1310, 71]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_14_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-16/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 50, 60)
    canvas.paste(_c7, (249, 5), _c7)
except Exception:
    pass
layout["icon_7"] = [249, 5, 299, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_14_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-16/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 53, 70)
    canvas.paste(_c8, (1318, 0), _c8)
except Exception:
    pass
layout["icon_8"] = [1318, 0, 1371, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_14_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-16/09_icon_4.51.png
try:
    _c9 = get_crop(9, 144, 144)
    canvas.paste(_c9, (12, 72), _c9)
except Exception:
    pass
layout["4.51"] = [12, 72, 156, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_14_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-16/10_icon_29.png
try:
    _c10 = get_crop(10, 132, 120)
    canvas.paste(_c10, (192, 1364), _c10)
except Exception:
    pass
layout["29"] = [192, 1364, 324, 1484]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_14_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-16/11_icon_What_date.png
try:
    _c11 = get_crop(11, 319, 72)
    canvas.paste(_c11, (558, 111), _c11)
except Exception:
    pass
layout["What_date?"] = [558, 111, 877, 183]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_14_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-16/12_icon_30.png
try:
    _c12 = get_crop(12, 132, 120)
    canvas.paste(_c12, (324, 1364), _c12)
except Exception:
    pass
layout["30"] = [324, 1364, 456, 1484]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_14_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-16/13_icon_4.51.png
try:
    _c13 = get_crop(13, 92, 62)
    canvas.paste(_c13, (15, 3), _c13)
except Exception:
    pass
layout["4.51"] = [15, 3, 107, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_14_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-16/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 50, 65)
    canvas.paste(_c14, (382, 2), _c14)
except Exception:
    pass
layout["icon_14"] = [382, 2, 432, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_14_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-16/15_icon_Next_month.png
try:
    _c15 = get_crop(15, 144, 144)
    canvas.paste(_c15, (846, 620), _c15)
except Exception:
    pass
layout["Next_month"] = [846, 620, 990, 764]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_14_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-16/16_icon_27.png
try:
    _c16 = get_crop(16, 132, 120)
    canvas.paste(_c16, (852, 1244), _c16)
except Exception:
    pass
layout["27"] = [852, 1244, 984, 1364]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_14_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-16/17_text_Start_Date.png
try:
    _c17 = get_crop(17, 583, 144)
    canvas.paste(_c17, (48, 313), _c17)
except Exception:
    pass
layout["Start_Date"] = [48, 313, 631, 457]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_14_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-16/18_text_End_Date.png
try:
    _c18 = get_crop(18, 638, 114)
    canvas.paste(_c18, (48, 476), _c18)
except Exception:
    pass
layout["End_Date"] = [48, 476, 686, 590]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_14_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-16/19_text_April_2024.png
try:
    _c19 = get_crop(19, 202, 54)
    canvas.paste(_c19, (421, 666), _c19)
except Exception:
    pass
layout["April_2024"] = [421, 666, 623, 720]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_14_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-16/20_text_10.png
try:
    _c20 = get_crop(20, 132, 120)
    canvas.paste(_c20, (456, 1004), _c20)
except Exception:
    pass
layout["10"] = [456, 1004, 588, 1124]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_14_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-16/21_text_11.png
try:
    _c21 = get_crop(21, 132, 120)
    canvas.paste(_c21, (588, 1004), _c21)
except Exception:
    pass
layout["11"] = [588, 1004, 720, 1124]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_14_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-16/22_text_12.png
try:
    _c22 = get_crop(22, 132, 120)
    canvas.paste(_c22, (720, 1004), _c22)
except Exception:
    pass
layout["12"] = [720, 1004, 852, 1124]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_14_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-16/23_text_13.png
try:
    _c23 = get_crop(23, 132, 120)
    canvas.paste(_c23, (852, 1004), _c23)
except Exception:
    pass
layout["13"] = [852, 1004, 984, 1124]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_14_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-16/24_text_14.png
try:
    _c24 = get_crop(24, 132, 120)
    canvas.paste(_c24, (60, 1124), _c24)
except Exception:
    pass
layout["14"] = [60, 1124, 192, 1244]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_14_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-16/25_text_15.png
try:
    _c25 = get_crop(25, 132, 120)
    canvas.paste(_c25, (192, 1124), _c25)
except Exception:
    pass
layout["15"] = [192, 1124, 324, 1244]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_14_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-16/26_text_16.png
try:
    _c26 = get_crop(26, 132, 120)
    canvas.paste(_c26, (324, 1124), _c26)
except Exception:
    pass
layout["16"] = [324, 1124, 456, 1244]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_14_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-16/27_text_17.png
try:
    _c27 = get_crop(27, 132, 120)
    canvas.paste(_c27, (456, 1124), _c27)
except Exception:
    pass
layout["17"] = [456, 1124, 588, 1244]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_14_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-16/28_text_18.png
try:
    _c28 = get_crop(28, 132, 120)
    canvas.paste(_c28, (588, 1124), _c28)
except Exception:
    pass
layout["18"] = [588, 1124, 720, 1244]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_14_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-16/29_text_19.png
try:
    _c29 = get_crop(29, 132, 120)
    canvas.paste(_c29, (720, 1124), _c29)
except Exception:
    pass
layout["19"] = [720, 1124, 852, 1244]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_14_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-16/30_text_20.png
try:
    _c30 = get_crop(30, 132, 120)
    canvas.paste(_c30, (852, 1124), _c30)
except Exception:
    pass
layout["20"] = [852, 1124, 984, 1244]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_14_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-16/31_text_21.png
try:
    _c31 = get_crop(31, 132, 120)
    canvas.paste(_c31, (60, 1244), _c31)
except Exception:
    pass
layout["21"] = [60, 1244, 192, 1364]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_14_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-16/32_text_22.png
try:
    _c32 = get_crop(32, 132, 120)
    canvas.paste(_c32, (192, 1244), _c32)
except Exception:
    pass
layout["22"] = [192, 1244, 324, 1364]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_14_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-16/33_text_23.png
try:
    _c33 = get_crop(33, 132, 120)
    canvas.paste(_c33, (324, 1244), _c33)
except Exception:
    pass
layout["23"] = [324, 1244, 456, 1364]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_14_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-16/34_text_24.png
try:
    _c34 = get_crop(34, 132, 120)
    canvas.paste(_c34, (456, 1244), _c34)
except Exception:
    pass
layout["24"] = [456, 1244, 588, 1364]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_14_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-16/35_text_25.png
try:
    _c35 = get_crop(35, 132, 120)
    canvas.paste(_c35, (588, 1244), _c35)
except Exception:
    pass
layout["25"] = [588, 1244, 720, 1364]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_14_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-16/36_text_26.png
try:
    _c36 = get_crop(36, 132, 120)
    canvas.paste(_c36, (720, 1244), _c36)
except Exception:
    pass
layout["26"] = [720, 1244, 852, 1364]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_14_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-16/37_clickable_1.png
try:
    _c37 = get_crop(37, 132, 120)
    canvas.paste(_c37, (192, 884), _c37)
except Exception:
    pass
layout["1"] = [192, 884, 324, 1004]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_14_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-16/38_clickable_2.png
try:
    _c38 = get_crop(38, 132, 120)
    canvas.paste(_c38, (324, 884), _c38)
except Exception:
    pass
layout["2"] = [324, 884, 456, 1004]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_14_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-16/39_clickable_3.png
try:
    _c39 = get_crop(39, 132, 120)
    canvas.paste(_c39, (456, 884), _c39)
except Exception:
    pass
layout["3"] = [456, 884, 588, 1004]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_14_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-16/40_clickable_4.png
try:
    _c40 = get_crop(40, 132, 120)
    canvas.paste(_c40, (588, 884), _c40)
except Exception:
    pass
layout["4"] = [588, 884, 720, 1004]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_14_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-16/41_clickable_5.png
try:
    _c41 = get_crop(41, 132, 120)
    canvas.paste(_c41, (720, 884), _c41)
except Exception:
    pass
layout["5"] = [720, 884, 852, 1004]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_14_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-16/42_clickable_6.png
try:
    _c42 = get_crop(42, 132, 120)
    canvas.paste(_c42, (852, 884), _c42)
except Exception:
    pass
layout["6"] = [852, 884, 984, 1004]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_14_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-16/43_clickable_7.png
try:
    _c43 = get_crop(43, 132, 120)
    canvas.paste(_c43, (60, 1004), _c43)
except Exception:
    pass
layout["7"] = [60, 1004, 192, 1124]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_14_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-16/44_clickable_8.png
try:
    _c44 = get_crop(44, 132, 120)
    canvas.paste(_c44, (192, 1004), _c44)
except Exception:
    pass
layout["8"] = [192, 1004, 324, 1124]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_14_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-16/45_clickable_9.png
try:
    _c45 = get_crop(45, 132, 120)
    canvas.paste(_c45, (324, 1004), _c45)
except Exception:
    pass
layout["9"] = [324, 1004, 456, 1124]
