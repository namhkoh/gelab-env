# page_id: page_seatgeek_275896c29e4341b1a61db6ff596ab44c_06
# screenshot: 2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-9.png
# step_index: 6/9
# task: Open SeatGeek. Look up "Seattle Mariners" tickets. Select the next upcoming event in Los Angeles. Set quantity to 2 and select the best value tickets. Which section are tickets in?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Background fill (app background)
draw.rectangle((0, 0, 1440, 2960), fill=(242, 244, 245))

# Status bar area at top
status_h = 86
draw.rectangle((0, 0, 1440, status_h), fill=(224, 226, 228))
# subtle divider under status bar
draw.line((0, status_h, 1440, status_h), fill=(213, 215, 218), width=1)

# Subtle toolbar band under status bar (background only, don't draw any icons/text)
toolbar_top = status_h
toolbar_bottom = 220
draw.rectangle((0, toolbar_top, 1440, toolbar_bottom), fill=(242, 244, 245))

# Main seating-map card background (rounded)
seat_card_bbox = (48, 360, 1392, 1960)
draw.rounded_rectangle(seat_card_bbox, radius=36, fill=(255, 255, 255), outline=(225, 228, 230), width=2)

# Inner seating/map area (slightly different tone to separate from card)
inner_inset = 44
inner_bbox = (seat_card_bbox[0] + inner_inset, seat_card_bbox[1] + inner_inset,
              seat_card_bbox[2] - inner_inset, seat_card_bbox[3] - inner_inset)
draw.rounded_rectangle(inner_bbox, radius=28, fill=(249, 250, 251), outline=None)

# Soft vignette / subtle background behind stadium (to help pasted map stand out)
vignette_bbox = (inner_bbox[0] + 24, inner_bbox[1] + 24, inner_bbox[2] - 24, inner_bbox[3] - 24)
draw.rectangle(vignette_bbox, fill=(245, 246, 247))

# Divider line between seating area and listings region
divider_y = 2000
draw.line((48, divider_y, 1392, divider_y), fill=(225, 227, 229), width=1)

# Bottom listings panel (rounded top corners)
list_panel_bbox = (0, 2020, 1440, 2960)
draw.rounded_rectangle(list_panel_bbox, radius=36, fill=(255, 255, 255), outline=(230, 232, 234), width=1)

# Listings header background area (behind "846 Listings" and sort control)
list_header_h = 2120
draw.rectangle((0, 2020, 1440, list_header_h), fill=(255, 255, 255))
# thin separator under the header
draw.line((48, list_header_h, 1392, list_header_h), fill=(235, 237, 238), width=1)

# Two listing card backgrounds (placeholders only — content will be pasted)
card_margin_x = 48
card_w = 1392 - card_margin_x
first_card_y0 = 2160
first_card_y1 = 2400
second_card_y0 = 2480
second_card_y1 = 2720
card_radius = 18
card_fill = (255, 255, 255)
card_outline = (236, 238, 239)

draw.rounded_rectangle((card_margin_x, first_card_y0, 1392, first_card_y1),
                       radius=card_radius, fill=card_fill, outline=card_outline, width=1)
draw.rounded_rectangle((card_margin_x, second_card_y0, 1392, second_card_y1),
                       radius=card_radius, fill=card_fill, outline=card_outline, width=1)

# subtle separators between listing rows (in case more items are present)
sep_y = first_card_y1 + 36
draw.line((card_margin_x + 12, sep_y, 1392 - 12, sep_y), fill=(245, 246, 247), width=1)
sep_y2 = second_card_y1 + 36
draw.line((card_margin_x + 12, sep_y2, 1392 - 12, sep_y2), fill=(245, 246, 247), width=1)

# Small top shadow under the seating card to lift it slightly
shadow_y0 = seat_card_bbox[3] - 6
for i, alpha in enumerate([18, 12, 6], start=0):
    y = shadow_y0 + i
    shade = 230 + i*2
    draw.line((seat_card_bbox[0]+8, y, seat_card_bbox[2]-8, y), fill=(shade, shade, shade), width=1)

# Left and right subtle edge insets to frame the content
edge_inset_color = (245, 246, 247)
draw.rectangle((0, 0, 24, 2960), fill=edge_inset_color)
draw.rectangle((1440-24, 0, 1440, 2960), fill=edge_inset_color)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_06_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-9/00_icon_Include_fees.png
try:
    _c0 = get_crop(0, 335, 108)
    canvas.paste(_c0, (544, 312), _c0)
except Exception:
    pass
layout["Include_fees"] = [544, 312, 879, 420]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_06_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-9/01_icon_Home_plate.png
try:
    _c1 = get_crop(1, 325, 108)
    canvas.paste(_c1, (915, 312), _c1)
except Exception:
    pass
layout["Home_plate"] = [915, 312, 1240, 420]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_06_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-9/02_icon_Quantity.png
try:
    _c2 = get_crop(2, 268, 108)
    canvas.paste(_c2, (240, 312), _c2)
except Exception:
    pass
layout["Quantity"] = [240, 312, 508, 420]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_06_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-9/03_icon_Tit.png
try:
    _c3 = get_crop(3, 156, 108)
    canvas.paste(_c3, (48, 312), _c3)
except Exception:
    pass
layout["Tit"] = [48, 312, 204, 420]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_06_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-9/04_icon_Include_fees.png
try:
    _c4 = get_crop(4, 1344, 156)
    canvas.paste(_c4, (48, 120), _c4)
except Exception:
    pass
layout["Include_fees"] = [48, 120, 1392, 276]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_06_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-9/05_icon_Ist_ba.png
try:
    _c5 = get_crop(5, 164, 108)
    canvas.paste(_c5, (1276, 312), _c5)
except Exception:
    pass
layout["Ist_ba"] = [1276, 312, 1440, 420]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_06_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-9/06_icon_9.8.png
try:
    _c6 = get_crop(6, 1440, 371)
    canvas.paste(_c6, (0, 2589), _c6)
except Exception:
    pass
layout["9.8"] = [0, 2589, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_06_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-9/07_icon_7.49_W.png
try:
    _c7 = get_crop(7, 67, 63)
    canvas.paste(_c7, (111, 0), _c7)
except Exception:
    pass
layout["7.49_W"] = [111, 0, 178, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_06_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-9/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 63, 62)
    canvas.paste(_c8, (242, 2), _c8)
except Exception:
    pass
layout["icon_8"] = [242, 2, 305, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_06_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-9/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 101, 63)
    canvas.paste(_c9, (1213, 0), _c9)
except Exception:
    pass
layout["icon_9"] = [1213, 0, 1314, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_06_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-9/10_icon_Mariners_at_Angels.png
try:
    _c10 = get_crop(10, 58, 63)
    canvas.paste(_c10, (312, 2), _c10)
except Exception:
    pass
layout["Mariners_at_Angels"] = [312, 2, 370, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_06_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-9/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 52, 65)
    canvas.paste(_c11, (1151, 0), _c11)
except Exception:
    pass
layout["icon_11"] = [1151, 0, 1203, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_06_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-9/12_icon_7.49_W.png
try:
    _c12 = get_crop(12, 54, 61)
    canvas.paste(_c12, (182, 1), _c12)
except Exception:
    pass
layout["7.49_W"] = [182, 1, 236, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_06_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-9/13_icon_846_Listings.png
try:
    _c13 = get_crop(13, 1440, 455)
    canvas.paste(_c13, (0, 2134), _c13)
except Exception:
    pass
layout["846_Listings"] = [0, 2134, 1440, 2589]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_06_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-9/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 54, 61)
    canvas.paste(_c14, (1319, 1), _c14)
except Exception:
    pass
layout["icon_14"] = [1319, 1, 1373, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_06_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-9/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 74, 71)
    canvas.paste(_c15, (1265, 863), _c15)
except Exception:
    pass
layout["icon_15"] = [1265, 863, 1339, 934]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_06_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-9/16_icon_icon_16.png
try:
    _c16 = get_crop(16, 69, 69)
    canvas.paste(_c16, (1284, 930), _c16)
except Exception:
    pass
layout["icon_16"] = [1284, 930, 1353, 999]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_06_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-9/17_icon_Ist_ba.png
try:
    _c17 = get_crop(17, 156, 156)
    canvas.paste(_c17, (1236, 120), _c17)
except Exception:
    pass
layout["Ist_ba"] = [1236, 120, 1392, 276]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_06_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-9/18_icon_Sort_by_deal.png
try:
    _c18 = get_crop(18, 440, 144)
    canvas.paste(_c18, (976, 1989), _c18)
except Exception:
    pass
layout["Sort_by_deal"] = [976, 1989, 1416, 2133]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_06_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-9/19_icon_Mariners_at_Angels.png
try:
    _c19 = get_crop(19, 50, 64)
    canvas.paste(_c19, (383, 1), _c19)
except Exception:
    pass
layout["Mariners_at_Angels"] = [383, 1, 433, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_06_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-9/20_icon_Amazing_deal.png
try:
    _c20 = get_crop(20, 1440, 455)
    canvas.paste(_c20, (0, 2134), _c20)
except Exception:
    pass
layout["Amazing_deal"] = [0, 2134, 1440, 2589]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_06_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-9/21_icon_icon_21.png
try:
    _c21 = get_crop(21, 47, 61)
    canvas.paste(_c21, (1238, 935), _c21)
except Exception:
    pass
layout["icon_21"] = [1238, 935, 1285, 996]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_06_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-9/22_icon_icon_22.png
try:
    _c22 = get_crop(22, 67, 71)
    canvas.paste(_c22, (1290, 997), _c22)
except Exception:
    pass
layout["icon_22"] = [1290, 997, 1357, 1068]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_06_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-9/23_icon_icon_23.png
try:
    _c23 = get_crop(23, 52, 67)
    canvas.paste(_c23, (1241, 995), _c23)
except Exception:
    pass
layout["icon_23"] = [1241, 995, 1293, 1062]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_06_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-9/24_icon_icon_24.png
try:
    _c24 = get_crop(24, 69, 72)
    canvas.paste(_c24, (1199, 865), _c24)
except Exception:
    pass
layout["icon_24"] = [1199, 865, 1268, 937]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_06_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-9/25_icon_7.49_W.png
try:
    _c25 = get_crop(25, 108, 64)
    canvas.paste(_c25, (3, 0), _c25)
except Exception:
    pass
layout["7.49_W"] = [3, 0, 111, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_06_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-9/26_text_L5.png
try:
    _c26 = get_crop(26, 34, 28)
    canvas.paste(_c26, (615, 1558), _c26)
except Exception:
    pass
layout["L5"] = [615, 1558, 649, 1586]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_06_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-9/27_text_216.png
try:
    _c27 = get_crop(27, 48, 29)
    canvas.paste(_c27, (643, 1598), _c27)
except Exception:
    pass
layout["216"] = [643, 1598, 691, 1627]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_06_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-9/28_text_1218.png
try:
    _c28 = get_crop(28, 48, 30)
    canvas.paste(_c28, (724, 1602), _c28)
except Exception:
    pass
layout["1218"] = [724, 1602, 772, 1632]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_06_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-9/29_text_846_Listings.png
try:
    _c29 = get_crop(29, 336, 72)
    canvas.paste(_c29, (56, 2029), _c29)
except Exception:
    pass
layout["846_Listings"] = [56, 2029, 392, 2101]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_06_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-9/30_text_S18_each.png
try:
    _c30 = get_crop(30, 1440, 371)
    canvas.paste(_c30, (0, 2589), _c30)
except Exception:
    pass
layout["S18_each"] = [0, 2589, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_06_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-9/31_text_Price_includes_fees.png
try:
    _c31 = get_crop(31, 1440, 371)
    canvas.paste(_c31, (0, 2589), _c31)
except Exception:
    pass
layout["Price_includes_fees"] = [0, 2589, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_06_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-9/32_text_9.8.png
try:
    _c32 = get_crop(32, 50, 31)
    canvas.paste(_c32, (502, 2810), _c32)
except Exception:
    pass
layout["9.8"] = [502, 2810, 552, 2841]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_06_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-9/33_text_Amazing_deal.png
try:
    _c33 = get_crop(33, 1440, 371)
    canvas.paste(_c33, (0, 2589), _c33)
except Exception:
    pass
layout["Amazing_deal"] = [0, 2589, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_06_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-9/34_text_1-8_tickets.png
try:
    _c34 = get_crop(34, 221, 48)
    canvas.paste(_c34, (488, 2872), _c34)
except Exception:
    pass
layout["1-8_tickets"] = [488, 2872, 709, 2920]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_06_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-9/35_text_GM.png
try:
    _c35 = get_crop(35, 48, 35)
    canvas.paste(_c35, (782, 1554), _c35)
except Exception:
    pass
layout["GM"] = [782, 1554, 830, 1589]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_06_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-9/36_clickable_Back.png
try:
    _c36 = get_crop(36, 156, 156)
    canvas.paste(_c36, (48, 120), _c36)
except Exception:
    pass
layout["Back"] = [48, 120, 204, 276]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_06_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-9/37_clickable_Mariners_at_Angels.png
try:
    _c37 = get_crop(37, 373, 156)
    canvas.paste(_c37, (204, 120), _c37)
except Exception:
    pass
layout["Mariners_at_Angels"] = [204, 120, 577, 276]
