# page_id: page_seatgeek_78c74bc9486545a6957029d5d7370c1c_06
# screenshot: 2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-9.png
# step_index: 6/9
# task: Open SeatGeek and search by category "Comedy". Select the first one in New York and check its information. Track the performer of this event.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Background/base fills and UI structure for the SeatGeek-like page
# Uses provided `canvas` (1440x2960 RGB) and `draw` (ImageDraw)
# Fonts: font_sm, font_md, font_lg, font_xl are available but not used (no text drawing)

# Full-page background (soft bluish-gray)
draw.rectangle([0, 0, 1440, 2960], fill="#eef1f4")

# Status bar area at top (slightly darker strip)
status_h = 80
draw.rectangle([0, 0, 1440, status_h], fill="#e6e9eb")

# Subtle divider under status bar
draw.line([(0, status_h), (1440, status_h)], fill="#d6d9db", width=1)

# Header / toolbar card (rounded pill)
header_top = 92
header_bottom = 216
header_left = 48
header_right = 1392
header_radius = 64

# Shadow for header (simple offset shadow)
draw.rounded_rectangle(
    [header_left, header_top + 6, header_right, header_bottom + 12],
    radius=header_radius,
    fill="#e9ecef",
    outline=None
)

# Header background (white pill)
draw.rounded_rectangle(
    [header_left, header_top, header_right, header_bottom],
    radius=header_radius,
    fill="#ffffff",
    outline="#dcdfe1",
    width=1
)

# Vertical subtle divider at right side of header (visual separation, not icons/text)
divider_x = header_right - 92
draw.line([(divider_x, header_top + 20), (divider_x, header_bottom - 20)], fill="#e1e4e6", width=2)

# Large circular seating-map card (centered)
center_x = 720
center_y = 820
map_radius = 520

# Outer drop shadow for circle (a simple offset filled ellipse)
shadow_offset = 12
draw.ellipse(
    [
        (center_x - map_radius + shadow_offset, center_y - map_radius + shadow_offset),
        (center_x + map_radius + shadow_offset, center_y + map_radius + shadow_offset)
    ],
    fill="#e6e9eb"
)

# Outer ring / border
draw.ellipse(
    [
        (center_x - map_radius, center_y - map_radius),
        (center_x + map_radius, center_y + map_radius)
    ],
    fill="#ffffff",
    outline="#bfc4c7",
    width=8
)

# Inner subtle background for the seating graphic (very light)
inner_inset = 22
draw.ellipse(
    [
        (center_x - map_radius + inner_inset, center_y - map_radius + inner_inset),
        (center_x + map_radius - inner_inset, center_y + map_radius - inner_inset)
    ],
    fill="#fbfcfd",
    outline=None
)

# A faint circular guide ring (to match card styling, not content)
guide_inset = 120
draw.ellipse(
    [
        (center_x - map_radius + guide_inset, center_y - map_radius + guide_inset),
        (center_x + map_radius - guide_inset, center_y + map_radius - guide_inset)
    ],
    outline="#f0f2f3",
    width=2
)

# Listings container card (white rounded rectangle anchored near bottom of the map)
list_top = 1860
list_left = 24
list_right = 1416
list_bottom = 2940
list_radius = 28

# Shadow behind listings container
draw.rounded_rectangle(
    [list_left, list_top + 8, list_right, list_bottom + 12],
    radius=list_radius,
    fill="#e9ecef"
)

# Listings container white background
draw.rounded_rectangle(
    [list_left, list_top, list_right, list_bottom],
    radius=list_radius,
    fill="#ffffff",
    outline="#e6e8ea",
    width=1
)

# Top divider for the listings header area
header_area_y = list_top + 28
draw.line([(list_left + 28, header_area_y), (list_right - 28, header_area_y)], fill="#eceff0", width=1)

# Draw a subtle right-aligned sort-region background (only background rectangle, no icons/text)
sort_region_w = 300
sort_region_h = 96
sort_region_x0 = list_right - 48 - sort_region_w
sort_region_y0 = list_top + 18
draw.rounded_rectangle(
    [sort_region_x0, sort_region_y0, sort_region_x0 + sort_region_w, sort_region_y0 + sort_region_h],
    radius=22,
    fill="#ffffff",
    outline="#edeeef",
    width=1
)

# Divider line below header text area in listings card
draw.line([(list_left + 24, list_top + 130), (list_right - 24, list_top + 130)], fill="#eef1f2", width=1)

# Individual listing card backgrounds (rounded rectangles) - only the card backings, no content
item_left = list_left + 24
item_right = list_right - 24
item_w = item_right - item_left
item_h = 280
first_item_top = list_top + 160
item_gap = 28

for i in range(3):
    top = first_item_top + i * (item_h + item_gap)
    bottom = top + item_h
    draw.rounded_rectangle(
        [item_left, top, item_right, bottom],
        radius=20,
        fill="#ffffff",
        outline="#eceff0",
        width=1
    )
    # separator line under each item (except last)
    sep_y = bottom + (item_gap // 2)
    draw.line([(item_left + 12, sep_y), (item_right - 12, sep_y)], fill="#f0f2f3", width=1)

# Subtle horizontal separators inside the first listing card to suggest content rows (no text/icons)
card_inset = 32
first_card_top = first_item_top
for y_offset in (80, 140, 200):
    draw.line([(item_left + card_inset, first_card_top + y_offset), (item_right - card_inset, first_card_top + y_offset)], fill="#f6f7f8", width=1)

# Small decorative faint divider near bottom of the page
draw.line([(48, list_bottom - 120), (1392, list_bottom - 120)], fill="#f3f5f6", width=1)

# End of structural drawing

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_06_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-9/00_icon_Include_fees.png
try:
    _c0 = get_crop(0, 335, 108)
    canvas.paste(_c0, (544, 312), _c0)
except Exception:
    pass
layout["Include_fees"] = [544, 312, 879, 420]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_06_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-9/01_icon_Best_seats.png
try:
    _c1 = get_crop(1, 303, 108)
    canvas.paste(_c1, (915, 312), _c1)
except Exception:
    pass
layout["Best_seats"] = [915, 312, 1218, 420]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_06_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-9/02_icon_Quantity.png
try:
    _c2 = get_crop(2, 268, 108)
    canvas.paste(_c2, (240, 312), _c2)
except Exception:
    pass
layout["Quantity"] = [240, 312, 508, 420]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_06_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-9/03_icon_8.5.png
try:
    _c3 = get_crop(3, 1440, 455)
    canvas.paste(_c3, (0, 2355), _c3)
except Exception:
    pass
layout["8.5"] = [0, 2355, 1440, 2810]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_06_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-9/04_icon_Tit.png
try:
    _c4 = get_crop(4, 156, 108)
    canvas.paste(_c4, (48, 312), _c4)
except Exception:
    pass
layout["Tit"] = [48, 312, 204, 420]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_06_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-9/05_icon_Include_fees.png
try:
    _c5 = get_crop(5, 310, 156)
    canvas.paste(_c5, (204, 120), _c5)
except Exception:
    pass
layout["Include_fees"] = [204, 120, 514, 276]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_06_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-9/06_icon_Low_pri.png
try:
    _c6 = get_crop(6, 186, 108)
    canvas.paste(_c6, (1254, 312), _c6)
except Exception:
    pass
layout["Low_pri"] = [1254, 312, 1440, 420]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_06_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-9/07_icon_Great_deal.png
try:
    _c7 = get_crop(7, 1440, 455)
    canvas.paste(_c7, (0, 2355), _c7)
except Exception:
    pass
layout["Great_deal"] = [0, 2355, 1440, 2810]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_06_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-9/08_icon_8.28_my.png
try:
    _c8 = get_crop(8, 61, 62)
    canvas.paste(_c8, (111, 1), _c8)
except Exception:
    pass
layout["8.28_my"] = [111, 1, 172, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_06_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-9/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 54, 64)
    canvas.paste(_c9, (1150, 1), _c9)
except Exception:
    pass
layout["icon_9"] = [1150, 1, 1204, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_06_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-9/10_icon_0.png
try:
    _c10 = get_crop(10, 103, 61)
    canvas.paste(_c10, (1212, 1), _c10)
except Exception:
    pass
layout["0#"] = [1212, 1, 1315, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_06_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-9/11_icon_8.28_my.png
try:
    _c11 = get_crop(11, 57, 61)
    canvas.paste(_c11, (181, 1), _c11)
except Exception:
    pass
layout["8.28_my"] = [181, 1, 238, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_06_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-9/12_icon_Andrew_Schulz.png
try:
    _c12 = get_crop(12, 53, 59)
    canvas.paste(_c12, (315, 2), _c12)
except Exception:
    pass
layout["Andrew_Schulz"] = [315, 2, 368, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_06_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-9/13_icon_Sort_by_price.png
try:
    _c13 = get_crop(13, 455, 144)
    canvas.paste(_c13, (961, 1989), _c13)
except Exception:
    pass
layout["Sort_by_price"] = [961, 1989, 1416, 2133]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_06_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-9/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 55, 60)
    canvas.paste(_c14, (247, 2), _c14)
except Exception:
    pass
layout["icon_14"] = [247, 2, 302, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_06_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-9/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 55, 58)
    canvas.paste(_c15, (1319, 2), _c15)
except Exception:
    pass
layout["icon_15"] = [1319, 2, 1374, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_06_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-9/16_icon_0.png
try:
    _c16 = get_crop(16, 156, 156)
    canvas.paste(_c16, (1236, 120), _c16)
except Exception:
    pass
layout["0#"] = [1236, 120, 1392, 276]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_06_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-9/17_icon_S110_each.png
try:
    _c17 = get_crop(17, 384, 106)
    canvas.paste(_c17, (51, 2854), _c17)
except Exception:
    pass
layout["S110_each"] = [51, 2854, 435, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_06_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-9/18_icon_Andrew_Schulz.png
try:
    _c18 = get_crop(18, 50, 62)
    canvas.paste(_c18, (382, 0), _c18)
except Exception:
    pass
layout["Andrew_Schulz"] = [382, 0, 432, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_06_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-9/19_icon_8.28_my.png
try:
    _c19 = get_crop(19, 110, 64)
    canvas.paste(_c19, (3, 0), _c19)
except Exception:
    pass
layout["8.28_my"] = [3, 0, 113, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_06_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-9/20_text_B16_315.png
try:
    _c20 = get_crop(20, 101, 30)
    canvas.paste(_c20, (416, 647), _c20)
except Exception:
    pass
layout["B16]_315"] = [416, 647, 517, 677]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_06_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-9/21_text_212.png
try:
    _c21 = get_crop(21, 48, 30)
    canvas.paste(_c21, (610, 712), _c21)
except Exception:
    pass
layout["212"] = [610, 712, 658, 742]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_06_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-9/22_text_210.png
try:
    _c22 = get_crop(22, 48, 27)
    canvas.paste(_c22, (779, 712), _c22)
except Exception:
    pass
layout["210"] = [779, 712, 827, 739]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_06_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-9/23_text_213.png
try:
    _c23 = get_crop(23, 48, 27)
    canvas.paste(_c23, (506, 733), _c23)
except Exception:
    pass
layout["213"] = [506, 733, 554, 760]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_06_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-9/24_text_209.png
try:
    _c24 = get_crop(24, 45, 29)
    canvas.paste(_c24, (886, 731), _c24)
except Exception:
    pass
layout["209"] = [886, 731, 931, 760]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_06_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-9/25_text_208.png
try:
    _c25 = get_crop(25, 46, 28)
    canvas.paste(_c25, (987, 781), _c25)
except Exception:
    pass
layout["208"] = [987, 781, 1033, 809]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_06_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-9/26_text_214.png
try:
    _c26 = get_crop(26, 46, 27)
    canvas.paste(_c26, (411, 807), _c26)
except Exception:
    pass
layout["214"] = [411, 807, 457, 834]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_06_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-9/27_text_SS18.png
try:
    _c27 = get_crop(27, 62, 28)
    canvas.paste(_c27, (1091, 818), _c27)
except Exception:
    pass
layout["SS18"] = [1091, 818, 1153, 846]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_06_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-9/28_text_215.png
try:
    _c28 = get_crop(28, 48, 27)
    canvas.paste(_c28, (326, 881), _c28)
except Exception:
    pass
layout["215"] = [326, 881, 374, 908]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_06_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-9/29_text_S16.png
try:
    _c29 = get_crop(29, 60, 29)
    canvas.paste(_c29, (1149, 888), _c29)
except Exception:
    pass
layout["S16"] = [1149, 888, 1209, 917]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_06_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-9/30_text_47.png
try:
    _c30 = get_crop(30, 20, 23)
    canvas.paste(_c30, (172, 905), _c30)
except Exception:
    pass
layout["47"] = [172, 905, 192, 928]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_06_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-9/31_text_S48.png
try:
    _c31 = get_crop(31, 58, 28)
    canvas.paste(_c31, (640, 899), _c31)
except Exception:
    pass
layout["[S48"] = [640, 899, 698, 927]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_06_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-9/32_text_S45__543.png
try:
    _c32 = get_crop(32, 126, 36)
    canvas.paste(_c32, (739, 898), _c32)
except Exception:
    pass
layout["[S45_[543"] = [739, 898, 865, 934]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_06_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-9/33_text_LS51.png
try:
    _c33 = get_crop(33, 57, 30)
    canvas.paste(_c33, (520, 913), _c33)
except Exception:
    pass
layout["LS51"] = [520, 913, 577, 943]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_06_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-9/34_text_207.png
try:
    _c34 = get_crop(34, 45, 27)
    canvas.paste(_c34, (1043, 927), _c34)
except Exception:
    pass
layout["207"] = [1043, 927, 1088, 954]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_06_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-9/35_text_2215P.png
try:
    _c35 = get_crop(35, 61, 27)
    canvas.paste(_c35, (317, 953), _c35)
except Exception:
    pass
layout["2215P"] = [317, 953, 378, 980]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_06_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-9/36_text_Ls39.png
try:
    _c36 = get_crop(36, 60, 27)
    canvas.paste(_c36, (957, 948), _c36)
except Exception:
    pass
layout["Ls39"] = [957, 948, 1017, 975]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_06_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-9/37_text_109.png
try:
    _c37 = get_crop(37, 46, 28)
    canvas.paste(_c37, (506, 966), _c37)
except Exception:
    pass
layout["109"] = [506, 966, 552, 994]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_06_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-9/38_text_110.png
try:
    _c38 = get_crop(38, 48, 27)
    canvas.paste(_c38, (423, 990), _c38)
except Exception:
    pass
layout["110"] = [423, 990, 471, 1017]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_06_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-9/39_text_415.png
try:
    _c39 = get_crop(39, 46, 30)
    canvas.paste(_c39, (178, 1008), _c39)
except Exception:
    pass
layout["415"] = [178, 1008, 224, 1038]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_06_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-9/40_text_216.png
try:
    _c40 = get_crop(40, 48, 29)
    canvas.paste(_c40, (270, 1006), _c40)
except Exception:
    pass
layout["216"] = [270, 1006, 318, 1035]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_06_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-9/41_text_LS57.png
try:
    _c41 = get_crop(41, 55, 28)
    canvas.paste(_c41, (342, 1003), _c41)
except Exception:
    pass
layout["LS57"] = [342, 1003, 397, 1031]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_06_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-9/42_text_LS36.png
try:
    _c42 = get_crop(42, 58, 28)
    canvas.paste(_c42, (1040, 1003), _c42)
except Exception:
    pass
layout["LS36"] = [1040, 1003, 1098, 1031]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_06_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-9/43_text_206.png
try:
    _c43 = get_crop(43, 48, 27)
    canvas.paste(_c43, (1119, 1006), _c43)
except Exception:
    pass
layout["206"] = [1119, 1006, 1167, 1033]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_06_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-9/44_text_I106D.png
try:
    _c44 = get_crop(44, 61, 27)
    canvas.paste(_c44, (768, 1043), _c44)
except Exception:
    pass
layout["I106D"] = [768, 1043, 829, 1070]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_06_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-9/45_text_111.png
try:
    _c45 = get_crop(45, 41, 27)
    canvas.paste(_c45, (351, 1064), _c45)
except Exception:
    pass
layout["111"] = [351, 1064, 392, 1091]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_06_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-9/46_text_205.png
try:
    _c46 = get_crop(46, 46, 28)
    canvas.paste(_c46, (1149, 1077), _c46)
except Exception:
    pass
layout["205"] = [1149, 1077, 1195, 1105]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_06_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-9/47_text_SS12.png
try:
    _c47 = get_crop(47, 59, 28)
    canvas.paste(_c47, (1228, 1077), _c47)
except Exception:
    pass
layout["SS12"] = [1228, 1077, 1287, 1105]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_06_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-9/48_text_416.png
try:
    _c48 = get_crop(48, 48, 29)
    canvas.paste(_c48, (150, 1117), _c48)
except Exception:
    pass
layout["416"] = [150, 1117, 198, 1146]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_06_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-9/49_text_SS11.png
try:
    _c49 = get_crop(49, 57, 27)
    canvas.paste(_c49, (1237, 1126), _c49)
except Exception:
    pass
layout["SS11"] = [1237, 1126, 1294, 1153]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_06_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-9/50_text_112.png
try:
    _c50 = get_crop(50, 46, 27)
    canvas.paste(_c50, (314, 1170), _c50)
except Exception:
    pass
layout["112"] = [314, 1170, 360, 1197]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_06_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-9/51_text_SS10.png
try:
    _c51 = get_crop(51, 60, 30)
    canvas.paste(_c51, (1239, 1172), _c51)
except Exception:
    pass
layout["SS10"] = [1239, 1172, 1299, 1202]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_06_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-9/52_text_STAGE.png
try:
    _c52 = get_crop(52, 44, 16)
    canvas.paste(_c52, (512, 1203), _c52)
except Exception:
    pass
layout["STAGE"] = [512, 1203, 556, 1219]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_06_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-9/53_text_MIX.png
try:
    _c53 = get_crop(53, 39, 25)
    canvas.paste(_c53, (888, 1198), _c53)
except Exception:
    pass
layout["MIX"] = [888, 1198, 927, 1223]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_06_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-9/54_text_417.png
try:
    _c54 = get_crop(54, 45, 27)
    canvas.paste(_c54, (148, 1230), _c54)
except Exception:
    pass
layout["417"] = [148, 1230, 193, 1257]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_06_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-9/55_text_sS9.png
try:
    _c55 = get_crop(55, 48, 27)
    canvas.paste(_c55, (1246, 1219), _c55)
except Exception:
    pass
layout["sS9"] = [1246, 1219, 1294, 1246]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_06_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-9/56_text_MC64.png
try:
    _c56 = get_crop(56, 69, 27)
    canvas.paste(_c56, (268, 1260), _c56)
except Exception:
    pass
layout["MC64"] = [268, 1260, 337, 1287]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_06_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-9/57_text_113.png
try:
    _c57 = get_crop(57, 45, 28)
    canvas.paste(_c57, (391, 1269), _c57)
except Exception:
    pass
layout["113"] = [391, 1269, 436, 1297]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_06_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-9/58_text_SS8.png
try:
    _c58 = get_crop(58, 48, 29)
    canvas.paste(_c58, (1242, 1265), _c58)
except Exception:
    pass
layout["SS8"] = [1242, 1265, 1290, 1294]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_06_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-9/59_text_203.png
try:
    _c59 = get_crop(59, 46, 27)
    canvas.paste(_c59, (1149, 1313), _c59)
except Exception:
    pass
layout["203"] = [1149, 1313, 1195, 1340]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_06_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-9/60_text_418.png
try:
    _c60 = get_crop(60, 48, 30)
    canvas.paste(_c60, (164, 1336), _c60)
except Exception:
    pass
layout["418"] = [164, 1336, 212, 1366]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_06_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-9/61_text_219.png
try:
    _c61 = get_crop(61, 48, 27)
    canvas.paste(_c61, (247, 1341), _c61)
except Exception:
    pass
layout["219"] = [247, 1341, 295, 1368]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_06_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-9/62_text_114.png
try:
    _c62 = get_crop(62, 48, 27)
    canvas.paste(_c62, (446, 1330), _c62)
except Exception:
    pass
layout["114"] = [446, 1330, 494, 1357]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_06_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-9/63_text_115.png
try:
    _c63 = get_crop(63, 46, 27)
    canvas.paste(_c63, (527, 1346), _c63)
except Exception:
    pass
layout["115"] = [527, 1346, 573, 1373]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_06_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-9/64_text_ES1S.png
try:
    _c64 = get_crop(64, 62, 29)
    canvas.paste(_c64, (640, 1346), _c64)
except Exception:
    pass
layout["ES1S]"] = [640, 1346, 702, 1375]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_06_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-9/65_text_ES12.png
try:
    _c65 = get_crop(65, 61, 29)
    canvas.paste(_c65, (738, 1346), _c65)
except Exception:
    pass
layout["ES12"] = [738, 1346, 799, 1375]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_06_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-9/66_text_LS1.png
try:
    _c66 = get_crop(66, 43, 27)
    canvas.paste(_c66, (326, 1364), _c66)
except Exception:
    pass
layout["LS1"] = [326, 1364, 369, 1391]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_06_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-9/67_text_SS6.png
try:
    _c67 = get_crop(67, 45, 27)
    canvas.paste(_c67, (1221, 1362), _c67)
except Exception:
    pass
layout["SS6"] = [1221, 1362, 1266, 1389]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_06_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-9/68_text_LS24.png
try:
    _c68 = get_crop(68, 60, 27)
    canvas.paste(_c68, (1038, 1387), _c68)
except Exception:
    pass
layout["LS24"] = [1038, 1387, 1098, 1414]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_06_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-9/69_text_202.png
try:
    _c69 = get_crop(69, 48, 30)
    canvas.paste(_c69, (1121, 1373), _c69)
except Exception:
    pass
layout["202"] = [1121, 1373, 1169, 1403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_06_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-9/70_text_419.png
try:
    _c70 = get_crop(70, 45, 27)
    canvas.paste(_c70, (206, 1441), _c70)
except Exception:
    pass
layout["419"] = [206, 1441, 251, 1468]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_06_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-9/71_text_220.png
try:
    _c71 = get_crop(71, 48, 27)
    canvas.paste(_c71, (326, 1436), _c71)
except Exception:
    pass
layout["220"] = [326, 1436, 374, 1463]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_06_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-9/72_text_LS4.png
try:
    _c72 = get_crop(72, 48, 27)
    canvas.paste(_c72, (398, 1427), _c72)
except Exception:
    pass
layout["LS4"] = [398, 1427, 446, 1454]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_06_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-9/73_text_201.png
try:
    _c73 = get_crop(73, 46, 27)
    canvas.paste(_c73, (1068, 1434), _c73)
except Exception:
    pass
layout["201"] = [1068, 1434, 1114, 1461]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_06_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-9/74_text_Ls7.png
try:
    _c74 = get_crop(74, 44, 27)
    canvas.paste(_c74, (485, 1466), _c74)
except Exception:
    pass
layout["Ls7"] = [485, 1466, 529, 1493]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_06_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-9/75_text_LS19.png
try:
    _c75 = get_crop(75, 60, 29)
    canvas.paste(_c75, (899, 1466), _c75)
except Exception:
    pass
layout["LS19"] = [899, 1466, 959, 1495]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_06_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-9/76_text_SS4.png
try:
    _c76 = get_crop(76, 48, 27)
    canvas.paste(_c76, (1179, 1457), _c76)
except Exception:
    pass
layout["SS4"] = [1179, 1457, 1227, 1484]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_06_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-9/77_text_221.png
try:
    _c77 = get_crop(77, 43, 27)
    canvas.paste(_c77, (409, 1498), _c77)
except Exception:
    pass
layout["221"] = [409, 1498, 452, 1525]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_06_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-9/78_text_LS12.png
try:
    _c78 = get_crop(78, 57, 29)
    canvas.paste(_c78, (654, 1494), _c78)
except Exception:
    pass
layout["LS12"] = [654, 1494, 711, 1523]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_06_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-9/79_text_LS15_LS17.png
try:
    _c79 = get_crop(79, 134, 41)
    canvas.paste(_c79, (760, 1479), _c79)
except Exception:
    pass
layout["LS15_LS17"] = [760, 1479, 894, 1520]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_06_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-9/80_text_SS3.png
try:
    _c80 = get_crop(80, 48, 27)
    canvas.paste(_c80, (1154, 1501), _c80)
except Exception:
    pass
layout["SS3"] = [1154, 1501, 1202, 1528]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_06_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-9/81_text_MEDIA.png
try:
    _c81 = get_crop(81, 62, 25)
    canvas.paste(_c81, (689, 1535), _c81)
except Exception:
    pass
layout["MEDIA"] = [689, 1535, 751, 1560]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_06_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-9/82_text_226.png
try:
    _c82 = get_crop(82, 48, 30)
    canvas.paste(_c82, (895, 1528), _c82)
except Exception:
    pass
layout["226"] = [895, 1528, 943, 1558]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_06_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-9/83_text_225.png
try:
    _c83 = get_crop(83, 46, 27)
    canvas.paste(_c83, (809, 1547), _c83)
except Exception:
    pass
layout["225"] = [809, 1547, 855, 1574]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_06_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-9/84_text_SS1.png
try:
    _c84 = get_crop(84, 46, 30)
    canvas.paste(_c84, (1098, 1572), _c84)
except Exception:
    pass
layout["SS1"] = [1098, 1572, 1144, 1602]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_06_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-9/85_text_0324.png
try:
    _c85 = get_crop(85, 58, 25)
    canvas.paste(_c85, (411, 1741), _c85)
except Exception:
    pass
layout["0324"] = [411, 1741, 469, 1766]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_06_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-9/86_text_325.png
try:
    _c86 = get_crop(86, 58, 27)
    canvas.paste(_c86, (499, 1741), _c86)
except Exception:
    pass
layout["[325"] = [499, 1741, 557, 1768]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_06_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-9/87_text_289_Listings.png
try:
    _c87 = get_crop(87, 332, 76)
    canvas.paste(_c87, (54, 2029), _c87)
except Exception:
    pass
layout["289_Listings"] = [54, 2029, 386, 2105]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_06_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-9/88_text_We_sell_resale_tickets._Resale_tickets_m.png
try:
    _c88 = get_crop(88, 1440, 455)
    canvas.paste(_c88, (0, 2355), _c88)
except Exception:
    pass
layout["€_We_sell_resale_tickets."] = [0, 2355, 1440, 2810]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_06_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-9/89_text_face_value.png
try:
    _c89 = get_crop(89, 218, 43)
    canvas.paste(_c89, (57, 2256), _c89)
except Exception:
    pass
layout["face_value:"] = [57, 2256, 275, 2299]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_06_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-9/90_text_S110_each.png
try:
    _c90 = get_crop(90, 250, 61)
    canvas.paste(_c90, (485, 2862), _c90)
except Exception:
    pass
layout["S110_each"] = [485, 2862, 735, 2923]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_06_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-9/91_text_223LWC.png
try:
    _c91 = get_crop(91, 103, 46)
    canvas.paste(_c91, (530, 1519), _c91)
except Exception:
    pass
layout["~223LWC"] = [530, 1519, 633, 1565]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/78c74bc9486545a6957029d5d7370c1c/step_06_2024_3_20_16_27_78c74bc9486545a6957029d5d7370c1c-9/92_clickable_Back.png
try:
    _c92 = get_crop(92, 156, 156)
    canvas.paste(_c92, (48, 120), _c92)
except Exception:
    pass
layout["Back"] = [48, 120, 204, 276]
