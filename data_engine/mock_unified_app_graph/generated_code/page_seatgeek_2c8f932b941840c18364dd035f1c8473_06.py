# page_id: page_seatgeek_2c8f932b941840c18364dd035f1c8473_06
# screenshot: 2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-9.png
# step_index: 6/8
# task: Open SeatGeek. Search "Beatles Love". Select the soonest upcoming event. Choose 2 tickets and continue. What is the lowest price for each ticket?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Canvas and draw are provided. Fonts: font_sm, font_md, font_lg, font_xl
# Draw the overall page background and structural UI elements only.

# Full background (dominant app background: very light cool gray)
draw.rectangle((0, 0, 1440, 2960), fill="#eef1f4")

# Status bar area at very top (slightly darker to contrast icons)
status_h = 88
draw.rectangle((0, 0, 1440, status_h), fill="#e2e5e8")
# subtle bottom divider under status bar
draw.line((0, status_h, 1440, status_h), fill="#d1d5d8", width=1)

# Top header "pill" background (rounded white bar behind title + back button)
header_top = status_h + 16
header_bottom = header_top + 100
padding_h = 48
draw.rounded_rectangle(
    (padding_h, header_top, 1440 - padding_h, header_bottom),
    radius=52,
    fill="#ffffff",
    outline="#e0e3e6",
    width=1
)

# Soft shadow / separator under header pill
draw.line((padding_h+8, header_bottom+6, 1440-padding_h-8, header_bottom+6), fill="#e8eaec", width=1)

# Filters area background (subtle band where filter pills sit)
filters_top = header_bottom + 28
filters_bottom = filters_top + 120
draw.rectangle((0, filters_top, 1440, filters_bottom), fill="#f3f5f7")
# Add a faint top separator for the filter band
draw.line((24, filters_top, 1440-24, filters_top), fill="#e6e8ea", width=1)

# Large seating-map background container (rounded card)
map_top = filters_bottom + 40
map_bottom = map_top + 1100
map_left = 84
map_right = 1440 - 84
draw.rounded_rectangle(
    (map_left, map_top, map_right, map_bottom),
    radius=36,
    fill="#eef1f4",
    outline="#d9dce0",
    width=2
)
# subtle inner panel where the seating graphic sits (slightly darker)
inner_inset = 16
draw.rounded_rectangle(
    (map_left + inner_inset, map_top + inner_inset, map_right - inner_inset, map_bottom - inner_inset),
    radius=28,
    fill="#f7f8fa",
    outline=None
)

# Central stage/background area (a dark content-area background to indicate the stage region)
# This is a simple large rounded rectangle to represent a dark content region behind map content.
stage_w = 740
stage_h = 500
stage_cx = (map_left + map_right) // 2
stage_cy = map_top + 510
stage_box = (stage_cx - stage_w//2, stage_cy - stage_h//2, stage_cx + stage_w//2, stage_cy + stage_h//2)
draw.rounded_rectangle(stage_box, radius=18, fill="#4a4c4f", outline="#3b3c3e", width=2)

# Slight horizontal and vertical guides on the map card to section off area (very subtle separators)
# (These are only structural lines, not labels/icons)
draw.line((map_left + 24, map_top + 220, map_right - 24, map_top + 220), fill="#eceff1", width=2)
draw.line((map_left + 24, map_top + 740, map_right - 24, map_top + 740), fill="#eceff1", width=2)
draw.line((map_left + (map_right-map_left)//2, map_top + 40, map_left + (map_right-map_left)//2, map_bottom - 40), fill="#eceff1", width=2)

# Bottom "listings" sheet: white rounded sheet coming up from bottom
sheet_top = map_bottom + 80
if sheet_top < 1960:
    sheet_top = 1960  # match visual placement similar to screenshot
sheet_radius = 28
draw.rounded_rectangle((0, sheet_top, 1440, 2960), radius=sheet_radius, fill="#ffffff", outline="#e6e8ea", width=1)

# Header area of the listings sheet (space for "222 Listings" and sort)
sheet_header_h = 120
draw.rectangle((0, sheet_top, 1440, sheet_top + sheet_header_h), fill="#ffffff")
# thin separator under header
draw.line((24, sheet_top + sheet_header_h, 1440 - 24, sheet_top + sheet_header_h), fill="#eceef0", width=1)

# Listing item background cards placeholders (stacked rows). These are only backgrounds.
item_h = 220
item_margin_x = 36
item_gap = 28
first_item_top = sheet_top + sheet_header_h + 28
for i in range(3):  # draw 3 listing card backgrounds to indicate repeated items
    top = first_item_top + i * (item_h + item_gap)
    left = item_margin_x
    right = 1440 - item_margin_x
    bottom = top + item_h
    # card background
    draw.rounded_rectangle((left, top, right, bottom), radius=18, fill="#ffffff", outline="#e9ebed", width=1)
    # thumbnail placeholder (rounded rect) on left of each card (background only)
    thumb_w = 200
    thumb_h = 160
    thumb_left = left + 28
    thumb_top = top + (item_h - thumb_h) // 2
    thumb_right = thumb_left + thumb_w
    thumb_bottom = thumb_top + thumb_h
    draw.rounded_rectangle((thumb_left, thumb_top, thumb_right, thumb_bottom), radius=16, fill="#eef1f4", outline="#d7d9db", width=1)
    # vertical separator line between thumbnail and details area
    sep_x = thumb_right + 28
    draw.line((sep_x, top + 18, sep_x, bottom - 18), fill="#f0f2f4", width=1)
    # price/value area background band on the right (subtle block where price chips appear)
    price_block_w = 300
    pb_left = right - 28 - price_block_w
    pb_top = top + 28
    pb_bottom = bottom - 28
    draw.rounded_rectangle((pb_left, pb_top, right - 28, pb_bottom), radius=12, fill="#ffffff", outline="#f1f3f4", width=1)
    # separator between listing cards
    draw.line((left + 12, bottom + item_gap//2, right - 12, bottom + item_gap//2), fill="#f3f5f6", width=1)

# Final bottom safe area subtle shadow line
draw.line((0, 2958, 1440, 2958), fill="#e9ebed", width=2)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_06_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-9/00_icon_Include_fees.png
try:
    _c0 = get_crop(0, 335, 108)
    canvas.paste(_c0, (544, 312), _c0)
except Exception:
    pass
layout["Include_fees"] = [544, 312, 879, 420]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_06_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-9/01_icon_Best_seats.png
try:
    _c1 = get_crop(1, 303, 108)
    canvas.paste(_c1, (915, 312), _c1)
except Exception:
    pass
layout["Best_seats"] = [915, 312, 1218, 420]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_06_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-9/02_icon_Quantity.png
try:
    _c2 = get_crop(2, 268, 108)
    canvas.paste(_c2, (240, 312), _c2)
except Exception:
    pass
layout["Quantity"] = [240, 312, 508, 420]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_06_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-9/03_icon_222_Listings.png
try:
    _c3 = get_crop(3, 1440, 455)
    canvas.paste(_c3, (0, 2134), _c3)
except Exception:
    pass
layout["222_Listings"] = [0, 2134, 1440, 2589]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_06_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-9/04_icon_Tit.png
try:
    _c4 = get_crop(4, 156, 108)
    canvas.paste(_c4, (48, 312), _c4)
except Exception:
    pass
layout["Tit"] = [48, 312, 204, 420]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_06_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-9/05_icon_8.9.png
try:
    _c5 = get_crop(5, 1440, 371)
    canvas.paste(_c5, (0, 2589), _c5)
except Exception:
    pass
layout["8.9"] = [0, 2589, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_06_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-9/06_icon_Low_pri.png
try:
    _c6 = get_crop(6, 186, 108)
    canvas.paste(_c6, (1254, 312), _c6)
except Exception:
    pass
layout["Low_pri"] = [1254, 312, 1440, 420]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_06_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-9/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 52, 64)
    canvas.paste(_c7, (1152, 1), _c7)
except Exception:
    pass
layout["icon_7"] = [1152, 1, 1204, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_06_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-9/08_icon_5.08_Wy.png
try:
    _c8 = get_crop(8, 67, 62)
    canvas.paste(_c8, (111, 1), _c8)
except Exception:
    pass
layout["5.08_Wy"] = [111, 1, 178, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_06_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-9/09_icon_6.png
try:
    _c9 = get_crop(9, 103, 63)
    canvas.paste(_c9, (1212, 1), _c9)
except Exception:
    pass
layout["6_"] = [1212, 1, 1315, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_06_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-9/10_icon_5.08_Wy.png
try:
    _c10 = get_crop(10, 54, 61)
    canvas.paste(_c10, (181, 2), _c10)
except Exception:
    pass
layout["5.08_Wy"] = [181, 2, 235, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_06_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-9/11_icon_Love.png
try:
    _c11 = get_crop(11, 1344, 156)
    canvas.paste(_c11, (48, 120), _c11)
except Exception:
    pass
layout["Love"] = [48, 120, 1392, 276]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_06_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-9/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 65, 61)
    canvas.paste(_c12, (242, 2), _c12)
except Exception:
    pass
layout["icon_12"] = [242, 2, 307, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_06_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-9/13_icon_6.png
try:
    _c13 = get_crop(13, 156, 156)
    canvas.paste(_c13, (1236, 120), _c13)
except Exception:
    pass
layout["6_"] = [1236, 120, 1392, 276]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_06_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-9/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 54, 60)
    canvas.paste(_c14, (1319, 2), _c14)
except Exception:
    pass
layout["icon_14"] = [1319, 2, 1373, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_06_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-9/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 60, 62)
    canvas.paste(_c15, (313, 2), _c15)
except Exception:
    pass
layout["icon_15"] = [313, 2, 373, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_06_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-9/16_icon_Sort_by_price.png
try:
    _c16 = get_crop(16, 455, 144)
    canvas.paste(_c16, (961, 1989), _c16)
except Exception:
    pass
layout["Sort_by_price"] = [961, 1989, 1416, 2133]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_06_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-9/17_icon_icon_17.png
try:
    _c17 = get_crop(17, 52, 63)
    canvas.paste(_c17, (381, 1), _c17)
except Exception:
    pass
layout["icon_17"] = [381, 1, 433, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_06_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-9/18_icon_Amazing_deal.png
try:
    _c18 = get_crop(18, 1440, 455)
    canvas.paste(_c18, (0, 2134), _c18)
except Exception:
    pass
layout["Amazing_deal"] = [0, 2134, 1440, 2589]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_06_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-9/19_text_207.png
try:
    _c19 = get_crop(19, 45, 27)
    canvas.paste(_c19, (553, 721), _c19)
except Exception:
    pass
layout["207"] = [553, 721, 598, 748]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_06_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-9/20_text_209.png
try:
    _c20 = get_crop(20, 45, 27)
    canvas.paste(_c20, (613, 712), _c20)
except Exception:
    pass
layout["209"] = [613, 712, 658, 739]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_06_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-9/21_text_208.png
try:
    _c21 = get_crop(21, 48, 27)
    canvas.paste(_c21, (851, 721), _c21)
except Exception:
    pass
layout["208"] = [851, 721, 899, 748]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_06_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-9/22_text_305.png
try:
    _c22 = get_crop(22, 48, 28)
    canvas.paste(_c22, (226, 788), _c22)
except Exception:
    pass
layout["305"] = [226, 788, 274, 816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_06_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-9/23_text_306.png
try:
    _c23 = get_crop(23, 46, 28)
    canvas.paste(_c23, (1186, 781), _c23)
except Exception:
    pass
layout["306"] = [1186, 781, 1232, 809]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_06_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-9/24_text_205.png
try:
    _c24 = get_crop(24, 48, 29)
    canvas.paste(_c24, (351, 821), _c24)
except Exception:
    pass
layout["205"] = [351, 821, 399, 850]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_06_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-9/25_text_206.png
try:
    _c25 = get_crop(25, 46, 29)
    canvas.paste(_c25, (1052, 821), _c25)
except Exception:
    pass
layout["206"] = [1052, 821, 1098, 850]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_06_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-9/26_text_305.png
try:
    _c26 = get_crop(26, 46, 27)
    canvas.paste(_c26, (171, 939), _c26)
except Exception:
    pass
layout["305"] = [171, 939, 217, 966]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_06_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-9/27_text_205.png
try:
    _c27 = get_crop(27, 45, 27)
    canvas.paste(_c27, (354, 932), _c27)
except Exception:
    pass
layout["205"] = [354, 932, 399, 959]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_06_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-9/28_text_206.png
try:
    _c28 = get_crop(28, 46, 27)
    canvas.paste(_c28, (1052, 932), _c28)
except Exception:
    pass
layout["206"] = [1052, 932, 1098, 959]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_06_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-9/29_text_306.png
try:
    _c29 = get_crop(29, 48, 27)
    canvas.paste(_c29, (1239, 932), _c29)
except Exception:
    pass
layout["306"] = [1239, 932, 1287, 959]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_06_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-9/30_text_STAGE.png
try:
    _c30 = get_crop(30, 41, 16)
    canvas.paste(_c30, (706, 1203), _c30)
except Exception:
    pass
layout["STAGE"] = [706, 1203, 747, 1219]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_06_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-9/31_text_203.png
try:
    _c31 = get_crop(31, 48, 29)
    canvas.paste(_c31, (344, 1279), _c31)
except Exception:
    pass
layout["203"] = [344, 1279, 392, 1308]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_06_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-9/32_text_204.png
try:
    _c32 = get_crop(32, 48, 27)
    canvas.paste(_c32, (1001, 1279), _c32)
except Exception:
    pass
layout["204"] = [1001, 1279, 1049, 1306]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_06_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-9/33_text_303.png
try:
    _c33 = get_crop(33, 46, 27)
    canvas.paste(_c33, (180, 1364), _c33)
except Exception:
    pass
layout["303"] = [180, 1364, 226, 1391]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_06_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-9/34_text_304.png
try:
    _c34 = get_crop(34, 46, 27)
    canvas.paste(_c34, (1232, 1371), _c34)
except Exception:
    pass
layout["304"] = [1232, 1371, 1278, 1398]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_06_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-9/35_text_203.png
try:
    _c35 = get_crop(35, 45, 30)
    canvas.paste(_c35, (347, 1558), _c35)
except Exception:
    pass
layout["203"] = [347, 1558, 392, 1588]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_06_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-9/36_text_204.png
try:
    _c36 = get_crop(36, 46, 30)
    canvas.paste(_c36, (1059, 1558), _c36)
except Exception:
    pass
layout["204"] = [1059, 1558, 1105, 1588]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_06_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-9/37_text_222_Listings.png
try:
    _c37 = get_crop(37, 330, 80)
    canvas.paste(_c37, (54, 2025), _c37)
except Exception:
    pass
layout["222_Listings"] = [54, 2025, 384, 2105]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_06_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-9/38_text_S151each.png
try:
    _c38 = get_crop(38, 1440, 371)
    canvas.paste(_c38, (0, 2589), _c38)
except Exception:
    pass
layout["S151each"] = [0, 2589, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_06_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-9/39_text_Price_includes_fees.png
try:
    _c39 = get_crop(39, 1440, 371)
    canvas.paste(_c39, (0, 2589), _c39)
except Exception:
    pass
layout["Price_includes_fees"] = [0, 2589, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_06_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-9/40_text_8.9.png
try:
    _c40 = get_crop(40, 50, 29)
    canvas.paste(_c40, (502, 2812), _c40)
except Exception:
    pass
layout["8.9"] = [502, 2812, 552, 2841]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_06_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-9/41_text_Amazing_deal.png
try:
    _c41 = get_crop(41, 1440, 371)
    canvas.paste(_c41, (0, 2589), _c41)
except Exception:
    pass
layout["Amazing_deal"] = [0, 2589, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_06_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-9/42_text_2-6_tickets.png
try:
    _c42 = get_crop(42, 1440, 371)
    canvas.paste(_c42, (0, 2589), _c42)
except Exception:
    pass
layout["2-6_tickets"] = [0, 2589, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_06_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-9/43_clickable_Back.png
try:
    _c43 = get_crop(43, 156, 156)
    canvas.paste(_c43, (48, 120), _c43)
except Exception:
    pass
layout["Back"] = [48, 120, 204, 276]
