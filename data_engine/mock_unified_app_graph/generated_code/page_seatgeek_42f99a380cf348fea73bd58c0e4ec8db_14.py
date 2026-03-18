# page_id: page_seatgeek_42f99a380cf348fea73bd58c0e4ec8db_14
# screenshot: 2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-17.png
# step_index: 14/14
# task: Open SeatGeek and search for the broadway show "lion king" on March 22. I need 3 tickets at average price less than 500 USD. Find the best seats and record the total price in google keep notes.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw overall background (dominant white)
draw.rectangle([(0, 0), (1440, 2960)], fill=(255, 255, 255))

# Colors
status_bar_bg = (246, 247, 248)       # very light gray for status bar
header_divider = (230, 230, 230)      # subtle divider
card_border = (240, 240, 240)         # card border / subtle background
thumbnail_bg = (237, 243, 250)        # light bluish thumbnail background
section_divider = (242, 242, 242)     # light section separator
shadow_color = (0, 0, 0, 18)          # transparent shadow if needed (not used with RGBA here)

# Status bar area (top)
status_bar_h = 96
draw.rectangle([(0, 0), (1440, status_bar_h)], fill=status_bar_bg)

# Main header area (title + controls). Keep it white but add a subtle bottom divider/shadow.
header_top = status_bar_h
header_bottom = 336
draw.rectangle([(0, header_top), (1440, header_bottom)], fill=(255, 255, 255))
# thin divider line under header
draw.line([(24, header_bottom), (1440 - 24, header_bottom)], fill=header_divider, width=1)

# Subtle horizontal separator above listings area (near info text area)
info_sep_y = 680
draw.line([(24, info_sep_y), (1440 - 24, info_sep_y)], fill=section_divider, width=1)

# Listing cards (two large rows). Use subtle off-white card background and thin separators.
first_card_y = 739
card_height = 455
second_card_y = first_card_y + card_height + 0  # matches screenshot spacing

card_margin_x = 24
card_radius = 12

# Card 1 background
draw.rounded_rectangle(
    [(card_margin_x, first_card_y), (1440 - card_margin_x, first_card_y + card_height)],
    radius=card_radius,
    fill=(255, 255, 255),
    outline=card_border,
    width=1
)

# Card 2 background
draw.rounded_rectangle(
    [(card_margin_x, second_card_y), (1440 - card_margin_x, second_card_y + card_height)],
    radius=card_radius,
    fill=(255, 255, 255),
    outline=card_border,
    width=1
)

# Thin separator line between the two listing cards (exact seam)
sep_y = second_card_y
draw.line([(24, sep_y), (1440 - 24, sep_y)], fill=section_divider, width=1)

# Draw thumbnail placeholders on the left for each listing (rounded bluish rectangles).
thumb_x = 48
thumb_w = 240
thumb_h = 200
thumb_radius = 18

# Center thumbnails vertically within each card (approx)
thumb1_y = first_card_y + 36
thumb2_y = second_card_y + 36

draw.rounded_rectangle(
    [(thumb_x, thumb1_y), (thumb_x + thumb_w, thumb1_y + thumb_h)],
    radius=thumb_radius,
    fill=thumbnail_bg,
    outline=(215, 220, 225),
    width=2
)

draw.rounded_rectangle(
    [(thumb_x, thumb2_y), (thumb_x + thumb_w, thumb2_y + thumb_h)],
    radius=thumb_radius,
    fill=thumbnail_bg,
    outline=(215, 220, 225),
    width=2
)

# Add subtle dividing rules inside each card to roughly match structure (not drawing any icons/text)
# a light horizontal guide under the thumbnail/text area
inner_sep_offset = 260
draw.line(
    [(card_margin_x + 12, first_card_y + inner_sep_offset), (1440 - card_margin_x - 12, first_card_y + inner_sep_offset)],
    fill=section_divider,
    width=1
)
draw.line(
    [(card_margin_x + 12, second_card_y + inner_sep_offset), (1440 - card_margin_x - 12, second_card_y + inner_sep_offset)],
    fill=section_divider,
    width=1
)

# Top-of-content subtle divider (below "2 Listings" / controls area)
draw.line([(24, 520), (1440 - 24, 520)], fill=section_divider, width=1)

# Bottom area left intentionally blank (avoid drawing the floating "Back to map" button)
# but add a faint large-area shadow band near bottom to match screenshot's depth (very subtle)
shadow_band_top = 2680
draw.rectangle([(0, shadow_band_top), (1440, 2960)], fill=(255, 255, 255))

# End of background/structure drawing

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_14_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-17/00_icon_Include_fees.png
try:
    _c0 = get_crop(0, 335, 108)
    canvas.paste(_c0, (543, 264), _c0)
except Exception:
    pass
layout["Include_fees"] = [543, 264, 878, 372]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_14_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-17/01_icon_7.3.png
try:
    _c1 = get_crop(1, 1440, 455)
    canvas.paste(_c1, (0, 739), _c1)
except Exception:
    pass
layout["7.3"] = [0, 739, 1440, 1194]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_14_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-17/02_icon_Best_seats.png
try:
    _c2 = get_crop(2, 303, 108)
    canvas.paste(_c2, (914, 264), _c2)
except Exception:
    pass
layout["Best_seats"] = [914, 264, 1217, 372]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_14_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-17/03_icon_3_tickets.png
try:
    _c3 = get_crop(3, 267, 108)
    canvas.paste(_c3, (240, 264), _c3)
except Exception:
    pass
layout["3_tickets"] = [240, 264, 507, 372]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_14_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-17/04_icon_Ili.png
try:
    _c4 = get_crop(4, 156, 108)
    canvas.paste(_c4, (48, 264), _c4)
except Exception:
    pass
layout["Ili"] = [48, 264, 204, 372]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_14_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-17/05_icon_7.7.png
try:
    _c5 = get_crop(5, 1440, 455)
    canvas.paste(_c5, (0, 1194), _c5)
except Exception:
    pass
layout["7.7"] = [0, 1194, 1440, 1649]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_14_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-17/06_icon_Low_pr.png
try:
    _c6 = get_crop(6, 187, 108)
    canvas.paste(_c6, (1253, 264), _c6)
except Exception:
    pass
layout["Low_pr"] = [1253, 264, 1440, 372]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_14_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-17/07_icon_Back_to_map.png
try:
    _c7 = get_crop(7, 523, 144)
    canvas.paste(_c7, (458, 2756), _c7)
except Exception:
    pass
layout["Back_to_map"] = [458, 2756, 981, 2900]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_14_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-17/08_icon_Low_pr.png
try:
    _c8 = get_crop(8, 156, 156)
    canvas.paste(_c8, (1284, 96), _c8)
except Exception:
    pass
layout["Low_pr"] = [1284, 96, 1440, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_14_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-17/09_icon_GEK.png
try:
    _c9 = get_crop(9, 58, 58)
    canvas.paste(_c9, (244, 3), _c9)
except Exception:
    pass
layout["GEK"] = [244, 3, 302, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_14_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-17/10_icon_Wy.png
try:
    _c10 = get_crop(10, 55, 59)
    canvas.paste(_c10, (181, 2), _c10)
except Exception:
    pass
layout["Wy"] = [181, 2, 236, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_14_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-17/11_icon_Great_deal.png
try:
    _c11 = get_crop(11, 1440, 455)
    canvas.paste(_c11, (0, 739), _c11)
except Exception:
    pass
layout["Great_deal"] = [0, 739, 1440, 1194]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_14_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-17/12_icon_Wy.png
try:
    _c12 = get_crop(12, 60, 62)
    canvas.paste(_c12, (112, 1), _c12)
except Exception:
    pass
layout["Wy"] = [112, 1, 172, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_14_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-17/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 53, 67)
    canvas.paste(_c13, (1151, 0), _c13)
except Exception:
    pass
layout["icon_13"] = [1151, 0, 1204, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_14_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-17/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 102, 63)
    canvas.paste(_c14, (1212, 1), _c14)
except Exception:
    pass
layout["icon_14"] = [1212, 1, 1314, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_14_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-17/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 52, 58)
    canvas.paste(_c15, (1320, 2), _c15)
except Exception:
    pass
layout["icon_15"] = [1320, 2, 1372, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_14_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-17/16_icon_Sort_by.png
try:
    _c16 = get_crop(16, 455, 144)
    canvas.paste(_c16, (961, 373), _c16)
except Exception:
    pass
layout["Sort_by"] = [961, 373, 1416, 517]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_14_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-17/17_icon_7.42.png
try:
    _c17 = get_crop(17, 99, 64)
    canvas.paste(_c17, (10, 0), _c17)
except Exception:
    pass
layout["7.42"] = [10, 0, 109, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_14_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-17/18_icon_Ili.png
try:
    _c18 = get_crop(18, 156, 156)
    canvas.paste(_c18, (0, 96), _c18)
except Exception:
    pass
layout["Ili"] = [0, 96, 156, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_14_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-17/19_icon_Listings.png
try:
    _c19 = get_crop(19, 156, 108)
    canvas.paste(_c19, (48, 264), _c19)
except Exception:
    pass
layout["Listings"] = [48, 264, 204, 372]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_14_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-17/20_icon_Great_deal.png
try:
    _c20 = get_crop(20, 1440, 455)
    canvas.paste(_c20, (0, 1194), _c20)
except Exception:
    pass
layout["Great_deal"] = [0, 1194, 1440, 1649]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_14_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-17/21_text_The_Lion_King.png
try:
    _c21 = get_crop(21, 496, 156)
    canvas.paste(_c21, (156, 96), _c21)
except Exception:
    pass
layout["The_Lion_King"] = [156, 96, 652, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_14_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-17/22_text_New_York.png
try:
    _c22 = get_crop(22, 496, 156)
    canvas.paste(_c22, (156, 96), _c22)
except Exception:
    pass
layout["New_York"] = [156, 96, 652, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_14_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-17/23_text_Fri_Mar_22_8_PM.png
try:
    _c23 = get_crop(23, 496, 156)
    canvas.paste(_c23, (156, 96), _c23)
except Exception:
    pass
layout["Fri,_Mar_22,8_PM"] = [156, 96, 652, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_14_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-17/24_text_We_sell_resale_tickets._Resale_tickets_m.png
try:
    _c24 = get_crop(24, 335, 108)
    canvas.paste(_c24, (543, 264), _c24)
except Exception:
    pass
layout["€_We_sell_resale_tickets."] = [543, 264, 878, 372]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_14_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-17/25_text_face_value.png
try:
    _c25 = get_crop(25, 218, 43)
    canvas.paste(_c25, (57, 639), _c25)
except Exception:
    pass
layout["face_value:"] = [57, 639, 275, 682]
