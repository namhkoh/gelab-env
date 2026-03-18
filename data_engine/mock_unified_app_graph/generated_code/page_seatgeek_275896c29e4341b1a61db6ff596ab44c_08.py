# page_id: page_seatgeek_275896c29e4341b1a61db6ff596ab44c_08
# screenshot: 2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-11.png
# step_index: 8/9
# task: Open SeatGeek. Look up "Seattle Mariners" tickets. Select the next upcoming event in Los Angeles. Set quantity to 2 and select the best value tickets. Which section are tickets in?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Background fill (dominant page color)
bg_color = (242, 244, 246)  # light cool gray
draw.rectangle([(0, 0), canvas.size], fill=bg_color)

# Status bar area at top (~80px)
status_h = 80
status_color = (230, 232, 234)
draw.rectangle([(0, 0), (canvas.width, status_h)], fill=status_color)

# Header / toolbar background (rounded white pill behind title area)
header_left = 48
header_top = 88
header_right = canvas.width - 48
header_bottom = 228
draw.rounded_rectangle(
    [(header_left, header_top), (header_right, header_bottom)],
    radius=64,
    fill=(255, 255, 255),
    outline=(220, 221, 223),
    width=1
)

# Subtle divider below header
draw.line([(header_left, header_bottom + 1), (header_right, header_bottom + 1)], fill=(232, 233, 235), width=1)

# Large central content container behind the stadium map
content_left = 40
content_top = header_bottom + 32
content_right = canvas.width - 40
content_bottom = 1580
draw.rounded_rectangle(
    [(content_left, content_top), (content_right, content_bottom)],
    radius=48,
    fill=(246, 247, 248),
    outline=(220, 221, 224),
    width=2
)

# Thin separator above the listings area
listings_top = 1960
draw.line([(24, listings_top), (canvas.width - 24, listings_top)], fill=(225, 226, 228), width=1)

# Bottom sheet / listings panel with rounded top corners
panel_top = listings_top
panel_bottom = canvas.height
panel_radius = 36
# Use a full-width rounded rectangle; for visual top-only rounding, draw a rounded rect that fills whole bottom
draw.rounded_rectangle(
    [(0, panel_top), (canvas.width, panel_bottom)],
    radius=panel_radius,
    fill=(255, 255, 255),
    outline=(220, 221, 223),
    width=1
)

# Small handle at top of bottom sheet
handle_w = 120
handle_h = 8
handle_x0 = (canvas.width - handle_w) // 2
handle_y0 = panel_top + 12
draw.rounded_rectangle(
    [(handle_x0, handle_y0), (handle_x0 + handle_w, handle_y0 + handle_h)],
    radius=4,
    fill=(233, 234, 236)
)

# Listing card background placeholders (cards will be layered on top by pasted crops)
card_left = 40
card_right = canvas.width - 40
card1_top = panel_top + 72
card1_bottom = card1_top + 180
card2_top = card1_bottom + 28
card2_bottom = card2_top + 180

card_outline = (236, 237, 238)
draw.rounded_rectangle(
    [(card_left, card1_top), (card_right, card1_bottom)],
    radius=20,
    fill=(255, 255, 255),
    outline=card_outline,
    width=1
)
draw.rounded_rectangle(
    [(card_left, card2_top), (card_right, card2_bottom)],
    radius=20,
    fill=(255, 255, 255),
    outline=card_outline,
    width=1
)

# Subtle separators between listing cards and panel content
draw.line([(card_left + 12, card1_bottom + 14), (card_right - 12, card1_bottom + 14)], fill=(240, 241, 242), width=1)
draw.line([(card_left + 12, card2_bottom + 14), (card_right - 12, card2_bottom + 14)], fill=(240, 241, 242), width=1)

# Light vertical edges/shadows to suggest elevation for central content
edge_shadow_color = (238, 239, 241)
draw.line([(content_left - 1, content_top + 6), (content_left - 1, content_bottom - 6)], fill=edge_shadow_color, width=1)
draw.line([(content_right + 1, content_top + 6), (content_right + 1, content_bottom - 6)], fill=edge_shadow_color, width=1)

# Final subtle divider above panel to reinforce section separation
draw.line([(24, panel_top - 8), (canvas.width - 24, panel_top - 8)], fill=(230, 231, 233), width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_08_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-11/00_icon_Include_fees.png
try:
    _c0 = get_crop(0, 335, 108)
    canvas.paste(_c0, (542, 312), _c0)
except Exception:
    pass
layout["Include_fees"] = [542, 312, 877, 420]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_08_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-11/01_icon_2_tickets.png
try:
    _c1 = get_crop(1, 266, 108)
    canvas.paste(_c1, (240, 312), _c1)
except Exception:
    pass
layout["2_tickets"] = [240, 312, 506, 420]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_08_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-11/02_icon_Home_plate.png
try:
    _c2 = get_crop(2, 325, 108)
    canvas.paste(_c2, (913, 312), _c2)
except Exception:
    pass
layout["Home_plate"] = [913, 312, 1238, 420]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_08_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-11/03_icon_Tit.png
try:
    _c3 = get_crop(3, 156, 108)
    canvas.paste(_c3, (48, 312), _c3)
except Exception:
    pass
layout["Tit"] = [48, 312, 204, 420]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_08_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-11/04_icon_Include_fees.png
try:
    _c4 = get_crop(4, 1344, 156)
    canvas.paste(_c4, (48, 120), _c4)
except Exception:
    pass
layout["Include_fees"] = [48, 120, 1392, 276]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_08_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-11/05_icon_Ist_ba.png
try:
    _c5 = get_crop(5, 166, 108)
    canvas.paste(_c5, (1274, 312), _c5)
except Exception:
    pass
layout["Ist_ba"] = [1274, 312, 1440, 420]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_08_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-11/06_icon_9.8.png
try:
    _c6 = get_crop(6, 1440, 371)
    canvas.paste(_c6, (0, 2589), _c6)
except Exception:
    pass
layout["9.8"] = [0, 2589, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_08_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-11/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 100, 63)
    canvas.paste(_c7, (1214, 0), _c7)
except Exception:
    pass
layout["icon_7"] = [1214, 0, 1314, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_08_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-11/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 63, 62)
    canvas.paste(_c8, (242, 2), _c8)
except Exception:
    pass
layout["icon_8"] = [242, 2, 305, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_08_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-11/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 51, 65)
    canvas.paste(_c9, (1152, 0), _c9)
except Exception:
    pass
layout["icon_9"] = [1152, 0, 1203, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_08_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-11/10_icon_7.50_my.png
try:
    _c10 = get_crop(10, 67, 63)
    canvas.paste(_c10, (111, 0), _c10)
except Exception:
    pass
layout["7.50_my"] = [111, 0, 178, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_08_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-11/11_icon_Mariners_at_Angels.png
try:
    _c11 = get_crop(11, 58, 63)
    canvas.paste(_c11, (312, 2), _c11)
except Exception:
    pass
layout["Mariners_at_Angels"] = [312, 2, 370, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_08_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-11/12_icon_7.50_my.png
try:
    _c12 = get_crop(12, 53, 62)
    canvas.paste(_c12, (182, 1), _c12)
except Exception:
    pass
layout["7.50_my"] = [182, 1, 235, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_08_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-11/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 54, 60)
    canvas.paste(_c13, (1319, 2), _c13)
except Exception:
    pass
layout["icon_13"] = [1319, 2, 1373, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_08_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-11/14_icon_Ist_ba.png
try:
    _c14 = get_crop(14, 156, 156)
    canvas.paste(_c14, (1236, 120), _c14)
except Exception:
    pass
layout["Ist_ba"] = [1236, 120, 1392, 276]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_08_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-11/15_icon_775_Listings.png
try:
    _c15 = get_crop(15, 1440, 455)
    canvas.paste(_c15, (0, 2134), _c15)
except Exception:
    pass
layout["775_Listings"] = [0, 2134, 1440, 2589]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_08_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-11/16_icon_Sort_by_deal.png
try:
    _c16 = get_crop(16, 440, 144)
    canvas.paste(_c16, (976, 1989), _c16)
except Exception:
    pass
layout["Sort_by_deal"] = [976, 1989, 1416, 2133]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_08_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-11/17_icon_Mariners_at_Angels.png
try:
    _c17 = get_crop(17, 49, 64)
    canvas.paste(_c17, (383, 1), _c17)
except Exception:
    pass
layout["Mariners_at_Angels"] = [383, 1, 432, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_08_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-11/18_icon_Amazing_deal.png
try:
    _c18 = get_crop(18, 1440, 455)
    canvas.paste(_c18, (0, 2134), _c18)
except Exception:
    pass
layout["Amazing_deal"] = [0, 2134, 1440, 2589]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_08_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-11/19_icon_Cue.png
try:
    _c19 = get_crop(19, 335, 108)
    canvas.paste(_c19, (542, 312), _c19)
except Exception:
    pass
layout["Cue?"] = [542, 312, 877, 420]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_08_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-11/20_text_775_Listings.png
try:
    _c20 = get_crop(20, 326, 68)
    canvas.paste(_c20, (55, 2032), _c20)
except Exception:
    pass
layout["775_Listings"] = [55, 2032, 381, 2100]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_08_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-11/21_text_S18_each.png
try:
    _c21 = get_crop(21, 1440, 371)
    canvas.paste(_c21, (0, 2589), _c21)
except Exception:
    pass
layout["S18_each"] = [0, 2589, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_08_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-11/22_text_Price_includes_fees.png
try:
    _c22 = get_crop(22, 1440, 371)
    canvas.paste(_c22, (0, 2589), _c22)
except Exception:
    pass
layout["Price_includes_fees"] = [0, 2589, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_08_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-11/23_text_9.8.png
try:
    _c23 = get_crop(23, 50, 31)
    canvas.paste(_c23, (502, 2810), _c23)
except Exception:
    pass
layout["9.8"] = [502, 2810, 552, 2841]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_08_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-11/24_text_Amazing_deal.png
try:
    _c24 = get_crop(24, 1440, 371)
    canvas.paste(_c24, (0, 2589), _c24)
except Exception:
    pass
layout["Amazing_deal"] = [0, 2589, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_08_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-11/25_text_1-8_tickets.png
try:
    _c25 = get_crop(25, 221, 48)
    canvas.paste(_c25, (488, 2872), _c25)
except Exception:
    pass
layout["1-8_tickets"] = [488, 2872, 709, 2920]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_08_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-11/26_clickable_Back.png
try:
    _c26 = get_crop(26, 156, 156)
    canvas.paste(_c26, (48, 120), _c26)
except Exception:
    pass
layout["Back"] = [48, 120, 204, 276]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_08_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-11/27_clickable_Mariners_at_Angels.png
try:
    _c27 = get_crop(27, 373, 156)
    canvas.paste(_c27, (204, 120), _c27)
except Exception:
    pass
layout["Mariners_at_Angels"] = [204, 120, 577, 276]
