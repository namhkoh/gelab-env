# page_id: page_seatgeek_03681efe5b614d869915a8728b755f3d_10
# screenshot: 2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-13.png
# step_index: 10/10
# task: Open SeatGeek. Search "Metropolitan Opera". Find the next available show. Filter by "best seats". What section are they in for the lowest price tickets?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Background/base fills and structural UI elements for the mobile UI page
# Assumes available variables: canvas (PIL.Image), draw (PIL.ImageDraw), font_sm, font_md, font_lg, font_xl

# Colors
bg = (243, 245, 246)        # page background (very light gray)
status_bg = (230, 231, 233) # top status bar
muted = (236, 238, 240)     # slightly muted panel
card_bg = (255, 255, 255)   # white card
divider = (219, 221, 223)   # light divider
shadow = (216, 218, 220)    # soft shadow for elevation
thumb_bg = (242, 244, 245)  # thumbnail background

W, H = canvas.size

# Fill overall background
draw.rectangle((0, 0, W, H), fill=bg)

# Status bar area (top)
status_h = 88
draw.rectangle((0, 0, W, status_h), fill=status_bg)
# subtle bottom hairline under status
draw.line((0, status_h - 1, W, status_h - 1), fill=divider, width=1)

# Header / Search pill background (rounded pill behind title & icons)
header_left, header_top = 48, 120
header_right, header_bottom = 1392, 276
header_radius = 80
# shadow behind pill
draw.rounded_rectangle(
    (header_left, header_top - 6, header_right, header_bottom + 6),
    radius=header_radius,
    fill=shadow
)
# pill itself
draw.rounded_rectangle(
    (header_left, header_top, header_right, header_bottom),
    radius=header_radius,
    fill=card_bg,
    outline=divider,
    width=1
)

# Divider below header area (separates header from filters / content)
draw.line((32, header_bottom + 12, W - 32, header_bottom + 12), fill=divider, width=1)

# Main seatmap/content background block
# This is a large soft panel where the venue diagrams live; keep it subtle so pasted diagrams sit on it.
content_left, content_top = 48, header_bottom + 36  # start below chips area
content_right, content_bottom = 1392, 1840
content_radius = 24
# background fill for content area
draw.rounded_rectangle(
    (content_left, content_top, content_right, content_bottom),
    radius=content_radius,
    fill=muted,
    outline=(230,230,232),
    width=1
)

# Add soft inner padding top shadow to emphasize separation (thin)
draw.line((content_left + 8, content_top + 4, content_right - 8, content_top + 4), fill=(238,240,241), width=1)

# Listings header card (the white rounded sheet that contains "16 Listings" and sort control)
list_header_top = 1960
list_header_bottom = 2068
list_header_left = 24
list_header_right = W - 24
list_header_radius = 24
# shadow behind header
draw.rounded_rectangle(
    (list_header_left, list_header_top + 4, list_header_right, list_header_bottom + 8),
    radius=list_header_radius,
    fill=shadow
)
# header card
draw.rounded_rectangle(
    (list_header_left, list_header_top, list_header_right, list_header_bottom),
    radius=list_header_radius,
    fill=card_bg,
    outline=divider,
    width=1
)
# horizontal divider under header to separate from listing content
divider_y = list_header_bottom + 12
draw.line((list_header_left + 8, divider_y, list_header_right - 8, divider_y), fill=divider, width=1)

# Disclaimer/info panel area (subtle, below header)
disclaimer_top = divider_y + 24
disclaimer_bottom = disclaimer_top + 120
draw.rectangle((list_header_left, disclaimer_top, list_header_right, disclaimer_bottom), fill=card_bg)
# light left padding marker line (visual structure, not text)
draw.line((list_header_left + 20, disclaimer_top + 16, list_header_left + 20, disclaimer_bottom - 16), fill=(248,249,250), width=8)

# First listing card (primary ticket card)
card1_top = disclaimer_bottom + 24
card1_bottom = card1_top + 480
card_left = 24
card_right = W - 24
card_radius = 20
# shadow
draw.rounded_rectangle(
    (card_left, card1_top + 6, card_right, card1_bottom + 10),
    radius=card_radius,
    fill=shadow
)
# card background
draw.rounded_rectangle(
    (card_left, card1_top, card_right, card1_bottom),
    radius=card_radius,
    fill=card_bg,
    outline=divider,
    width=1
)
# small thumbnail background on left inside card (rounded square)
thumb_margin = 36
thumb_size = 220
thumb_x0 = card_left + thumb_margin
thumb_y0 = card1_top + thumb_margin
thumb_x1 = thumb_x0 + thumb_size
thumb_y1 = thumb_y0 + thumb_size
draw.rounded_rectangle((thumb_x0, thumb_y0, thumb_x1, thumb_y1), radius=18, fill=thumb_bg, outline=(230,232,234), width=1)

# Subtle vertical divider between thumbnail and text area
text_area_x = thumb_x1 + 28
draw.line((text_area_x, card1_top + 24, text_area_x, card1_bottom - 24), fill=(248,249,250), width=1)

# Second listing card (partial, lower on screen)
card2_top = card1_bottom + 8
card2_bottom = H - 24
# shadow
draw.rounded_rectangle(
    (card_left, card2_top + 6, card_right, card2_bottom + 6),
    radius=card_radius,
    fill=shadow
)
draw.rounded_rectangle(
    (card_left, card2_top, card_right, card2_bottom),
    radius=card_radius,
    fill=card_bg,
    outline=divider,
    width=1
)
# thumbnail for second card
thumb2_x0 = card_left + thumb_margin
thumb2_y0 = card2_top + thumb_margin
thumb2_x1 = thumb2_x0 + 180
thumb2_y1 = thumb2_y0 + 180
draw.rounded_rectangle((thumb2_x0, thumb2_y0, thumb2_x1, thumb2_y1), radius=16, fill=thumb_bg, outline=(230,232,234), width=1)

# separators between list cards (thin)
sep_y = card1_bottom + 4
draw.line((card_left + 12, sep_y, card_right - 12, sep_y), fill=divider, width=1)

# Bottom safe area bar (soft)
safe_bar_h = 36
draw.rectangle((0, H - safe_bar_h, W, H), fill=bg)

# Final subtle accents: small rounded corner indicators at major panels (purely structural)
# top of content panel small corner radii visual
draw.rectangle((content_left, content_top - 8, content_left + 6, content_top), fill=(240,241,242))
draw.rectangle((content_right - 6, content_top - 8, content_right, content_top), fill=(240,241,242))

# Note: No icons or text are drawn here. Only background, cards, separators and safe structural elements.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_10_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-13/00_icon_Include.png
try:
    _c0 = get_crop(0, 335, 108)
    canvas.paste(_c0, (544, 312), _c0)
except Exception:
    pass
layout["Include"] = [544, 312, 879, 420]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_10_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-13/01_icon_Best_seats.png
try:
    _c1 = get_crop(1, 303, 108)
    canvas.paste(_c1, (915, 312), _c1)
except Exception:
    pass
layout["Best_seats"] = [915, 312, 1218, 420]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_10_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-13/02_icon_Quantity.png
try:
    _c2 = get_crop(2, 268, 108)
    canvas.paste(_c2, (240, 312), _c2)
except Exception:
    pass
layout["Quantity"] = [240, 312, 508, 420]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_10_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-13/03_icon_Low_pri.png
try:
    _c3 = get_crop(3, 186, 108)
    canvas.paste(_c3, (1254, 312), _c3)
except Exception:
    pass
layout["Low_pri"] = [1254, 312, 1440, 420]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_10_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-13/04_icon_Tit.png
try:
    _c4 = get_crop(4, 156, 108)
    canvas.paste(_c4, (48, 312), _c4)
except Exception:
    pass
layout["Tit"] = [48, 312, 204, 420]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_10_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-13/05_icon_0y.png
try:
    _c5 = get_crop(5, 1440, 455)
    canvas.paste(_c5, (0, 2355), _c5)
except Exception:
    pass
layout["0y"] = [0, 2355, 1440, 2810]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_10_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-13/06_icon_Include.png
try:
    _c6 = get_crop(6, 1344, 156)
    canvas.paste(_c6, (48, 120), _c6)
except Exception:
    pass
layout["Include"] = [48, 120, 1392, 276]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_10_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-13/07_icon_El_Nino.png
try:
    _c7 = get_crop(7, 56, 61)
    canvas.paste(_c7, (313, 3), _c7)
except Exception:
    pass
layout["El_Nino"] = [313, 3, 369, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_10_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-13/08_icon_7_59_my.png
try:
    _c8 = get_crop(8, 69, 64)
    canvas.paste(_c8, (110, 0), _c8)
except Exception:
    pass
layout["7:59_my"] = [110, 0, 179, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_10_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-13/09_icon_El_Nino.png
try:
    _c9 = get_crop(9, 61, 61)
    canvas.paste(_c9, (243, 2), _c9)
except Exception:
    pass
layout["El_Nino"] = [243, 2, 304, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_10_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-13/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 51, 65)
    canvas.paste(_c10, (1152, 1), _c10)
except Exception:
    pass
layout["icon_10"] = [1152, 1, 1203, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_10_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-13/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 102, 63)
    canvas.paste(_c11, (1213, 1), _c11)
except Exception:
    pass
layout["icon_11"] = [1213, 1, 1315, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_10_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-13/12_icon_7_59_my.png
try:
    _c12 = get_crop(12, 55, 59)
    canvas.paste(_c12, (182, 2), _c12)
except Exception:
    pass
layout["7:59_my"] = [182, 2, 237, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_10_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-13/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 52, 57)
    canvas.paste(_c13, (1321, 3), _c13)
except Exception:
    pass
layout["icon_13"] = [1321, 3, 1373, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_10_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-13/14_icon_New_York.png
try:
    _c14 = get_crop(14, 49, 63)
    canvas.paste(_c14, (383, 1), _c14)
except Exception:
    pass
layout["New_York"] = [383, 1, 432, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_10_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-13/15_icon_Amazing_deal.png
try:
    _c15 = get_crop(15, 1440, 455)
    canvas.paste(_c15, (0, 2355), _c15)
except Exception:
    pass
layout["Amazing_deal"] = [0, 2355, 1440, 2810]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_10_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-13/16_icon_Sort_by_price.png
try:
    _c16 = get_crop(16, 455, 144)
    canvas.paste(_c16, (961, 1989), _c16)
except Exception:
    pass
layout["Sort_by_price"] = [961, 1989, 1416, 2133]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_10_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-13/17_icon_Low_pri.png
try:
    _c17 = get_crop(17, 156, 156)
    canvas.paste(_c17, (1236, 120), _c17)
except Exception:
    pass
layout["Low_pri"] = [1236, 120, 1392, 276]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_10_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-13/18_icon_0y.png
try:
    _c18 = get_crop(18, 382, 106)
    canvas.paste(_c18, (52, 2854), _c18)
except Exception:
    pass
layout["0y"] = [52, 2854, 434, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_10_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-13/19_text_STAGE.png
try:
    _c19 = get_crop(19, 42, 16)
    canvas.paste(_c19, (470, 611), _c19)
except Exception:
    pass
layout["STAGE"] = [470, 611, 512, 627]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_10_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-13/20_text_ORCHESTRA_PIT.png
try:
    _c20 = get_crop(20, 136, 25)
    canvas.paste(_c20, (421, 684), _c20)
except Exception:
    pass
layout["ORCHESTRA_PIT"] = [421, 684, 557, 709]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_10_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-13/21_text_ORCH_L.png
try:
    _c21 = get_crop(21, 87, 27)
    canvas.paste(_c21, (354, 821), _c21)
except Exception:
    pass
layout["ORCH_L"] = [354, 821, 441, 848]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_10_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-13/22_text_49.png
try:
    _c22 = get_crop(22, 36, 30)
    canvas.paste(_c22, (844, 920), _c22)
except Exception:
    pass
layout["49"] = [844, 920, 880, 950]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_10_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-13/23_text_18.png
try:
    _c23 = get_crop(23, 34, 29)
    canvas.paste(_c23, (1057, 911), _c23)
except Exception:
    pass
layout["18"] = [1057, 911, 1091, 940]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_10_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-13/24_text_27.png
try:
    _c24 = get_crop(24, 36, 27)
    canvas.paste(_c24, (923, 941), _c24)
except Exception:
    pass
layout["27"] = [923, 941, 959, 968]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_10_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-13/25_text_26.png
try:
    _c25 = get_crop(25, 34, 27)
    canvas.paste(_c25, (983, 939), _c25)
except Exception:
    pass
layout["26"] = [983, 939, 1017, 966]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_10_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-13/26_text_LEVEL_1_ORCHESTRA.png
try:
    _c26 = get_crop(26, 173, 25)
    canvas.paste(_c26, (402, 1017), _c26)
except Exception:
    pass
layout["LEVEL_1_ORCHESTRA"] = [402, 1017, 575, 1042]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_10_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-13/27_text_LEVEL_2_PARTERRE.png
try:
    _c27 = get_crop(27, 159, 25)
    canvas.paste(_c27, (879, 1017), _c27)
except Exception:
    pass
layout["LEVEL_2_PARTERRE"] = [879, 1017, 1038, 1042]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_10_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-13/28_text_30.png
try:
    _c28 = get_crop(28, 32, 27)
    canvas.paste(_c28, (615, 1126), _c28)
except Exception:
    pass
layout["30"] = [615, 1126, 647, 1153]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_10_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-13/29_text_33.png
try:
    _c29 = get_crop(29, 31, 27)
    canvas.paste(_c29, (331, 1149), _c29)
except Exception:
    pass
layout["33"] = [331, 1149, 362, 1176]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_10_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-13/30_text_ROOM.png
try:
    _c30 = get_crop(30, 64, 27)
    canvas.paste(_c30, (950, 1367), _c30)
except Exception:
    pass
layout["~ROOM"] = [950, 1367, 1014, 1394]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_10_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-13/31_text_BALC_L.png
try:
    _c31 = get_crop(31, 80, 27)
    canvas.paste(_c31, (342, 1739), _c31)
except Exception:
    pass
layout["BALC_L"] = [342, 1739, 422, 1766]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_10_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-13/32_text_BALC_R.png
try:
    _c32 = get_crop(32, 83, 27)
    canvas.paste(_c32, (557, 1739), _c32)
except Exception:
    pass
layout["BALC_R"] = [557, 1739, 640, 1766]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_10_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-13/33_text_LEVEL_5_BALCONY.png
try:
    _c33 = get_crop(33, 153, 20)
    canvas.paste(_c33, (412, 1844), _c33)
except Exception:
    pass
layout["LEVEL_5_BALCONY"] = [412, 1844, 565, 1864]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_10_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-13/34_text_16_Listings.png
try:
    _c34 = get_crop(34, 291, 81)
    canvas.paste(_c34, (54, 2024), _c34)
except Exception:
    pass
layout["16_Listings"] = [54, 2024, 345, 2105]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_10_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-13/35_text_We_sell_resale_tickets._Resale_tickets_m.png
try:
    _c35 = get_crop(35, 1440, 455)
    canvas.paste(_c35, (0, 2355), _c35)
except Exception:
    pass
layout["€_We_sell_resale_tickets."] = [0, 2355, 1440, 2810]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_10_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-13/36_text_face_value.png
try:
    _c36 = get_crop(36, 218, 43)
    canvas.paste(_c36, (57, 2256), _c36)
except Exception:
    pass
layout["face_value:"] = [57, 2256, 275, 2299]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_10_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-13/37_text_S623_each.png
try:
    _c37 = get_crop(37, 273, 66)
    canvas.paste(_c37, (485, 2862), _c37)
except Exception:
    pass
layout["S623_each"] = [485, 2862, 758, 2928]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_10_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-13/38_text_STANDING.png
try:
    _c38 = get_crop(38, 104, 46)
    canvas.paste(_c38, (538, 933), _c38)
except Exception:
    pass
layout["STANDING"] = [538, 933, 642, 979]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_10_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-13/39_text_STANDING.png
try:
    _c39 = get_crop(39, 104, 45)
    canvas.paste(_c39, (338, 935), _c39)
except Exception:
    pass
layout["STANDING"] = [338, 935, 442, 980]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_10_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-13/40_text_STANDING.png
try:
    _c40 = get_crop(40, 101, 66)
    canvas.paste(_c40, (790, 1300), _c40)
except Exception:
    pass
layout["STANDING"] = [790, 1300, 891, 1366]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_10_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-13/41_text_STANDING.png
try:
    _c41 = get_crop(41, 103, 60)
    canvas.paste(_c41, (322, 1317), _c41)
except Exception:
    pass
layout["STANDING"] = [322, 1317, 425, 1377]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_10_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-13/42_text_STANDING.png
try:
    _c42 = get_crop(42, 104, 56)
    canvas.paste(_c42, (554, 1319), _c42)
except Exception:
    pass
layout["~STANDING"] = [554, 1319, 658, 1375]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_10_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-13/43_text_STANDING.png
try:
    _c43 = get_crop(43, 103, 51)
    canvas.paste(_c43, (1001, 1765), _c43)
except Exception:
    pass
layout["~STANDING"] = [1001, 1765, 1104, 1816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_10_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-13/44_text_STANDING.png
try:
    _c44 = get_crop(44, 105, 51)
    canvas.paste(_c44, (811, 1765), _c44)
except Exception:
    pass
layout["STANDING"] = [811, 1765, 916, 1816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_10_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-13/45_clickable_Back.png
try:
    _c45 = get_crop(45, 156, 156)
    canvas.paste(_c45, (48, 120), _c45)
except Exception:
    pass
layout["Back"] = [48, 120, 204, 276]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_10_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-13/46_clickable_El_Nino_-_New_York.png
try:
    _c46 = get_crop(46, 363, 156)
    canvas.paste(_c46, (204, 120), _c46)
except Exception:
    pass
layout["El_Nino_-_New_York"] = [204, 120, 567, 276]
