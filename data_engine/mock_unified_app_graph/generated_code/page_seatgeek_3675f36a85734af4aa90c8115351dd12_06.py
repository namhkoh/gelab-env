# page_id: page_seatgeek_3675f36a85734af4aa90c8115351dd12_06
# screenshot: 2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-9.png
# step_index: 6/9
# task: Open SeatGeek. Search "The Fonda Theatre". Select the top popular event and track it. What is the lowest price?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Background and structure drawing for the mobile UI page (PIL)
# Assumes variables provided: canvas (PIL Image 1440x2960 RGB), draw (ImageDraw), fonts: font_sm, font_md, font_lg, font_xl

# 1) Fill overall background with the dominant very light bluish-gray
draw.rectangle([(0, 0), (1440, 2960)], fill="#eef1f4")

# 2) Status bar area at top (~88px) — slightly darker than background
status_h = 88
draw.rectangle([(0, 0), (1440, status_h)], fill="#e5e7ea")

# subtle bottom divider under status bar
draw.line([(0, status_h), (1440, status_h)], fill="#d6d8da", width=1)

# 3) Light shadow / backdrop behind the main header area (keeps header contents readable)
# This intentionally sits behind the header (but offset so it doesn't exactly duplicate header elements)
header_back_top = status_h + 10
header_back_bottom = header_back_top + 200
draw.rounded_rectangle(
    [(32, header_back_top), (1408, header_back_bottom)],
    radius=40,
    fill="#ffffff",
    outline="#ececec",
    width=1
)

# a very subtle shadow line below the header backdrop
draw.line(
    [(32, header_back_bottom + 1), (1408, header_back_bottom + 1)],
    fill="#e6e7e9",
    width=1
)

# 4) Central seating-map card: outer card with border and light inner area
# Card positioned centered in upper/mid region of the canvas
card_left = 200
card_right = 1240
card_top = 560
card_bottom = 1720
card_radius = 18

# Outer border (drop-shadow effect): a faint darker stroke around the card
draw.rounded_rectangle(
    [(card_left - 6, card_top - 6), (card_right + 6, card_bottom + 6)],
    radius=card_radius + 6,
    fill="#000000",
    outline=None
)
# Overlay a semi-transparent simulation by drawing a larger very-light rectangle to mimic subtle shadow wash
draw.rounded_rectangle(
    [(card_left - 6, card_top - 6), (card_right + 6, card_bottom + 6)],
    radius=card_radius + 6,
    fill="#00000000",
    outline=None
)

# Main card (white) with a light gray internal border to match the map frame
draw.rounded_rectangle(
    [(card_left, card_top), (card_right, card_bottom)],
    radius=card_radius,
    fill="#ffffff",
    outline="#cfcfcf",
    width=6
)

# Inner inset area inside the card for map content (keeps a margin so pasted map/icons sit on top)
inset_margin = 28
draw.rectangle(
    [(card_left + inset_margin, card_top + inset_margin), (card_right - inset_margin, card_bottom - inset_margin)],
    fill="#f5f6f7",
    outline=None
)

# 5) Listings container at bottom: full-width rounded panel with white background and top rounded corners
list_top = 1960
list_radius = 40
draw.rounded_rectangle(
    [(0, list_top), (1440, 2960)],
    radius=list_radius,
    fill="#ffffff",
    outline="#e6e6e6",
    width=1
)

# Subtle top shadow line to separate content area from map area
draw.line([(16, list_top + 1), (1424, list_top + 1)], fill="#e0e0e0", width=2)

# 6) Header row inside listings (left label + right sort control area)
# Draw a faint divider area for the header region (no text/icons - they will be pasted)
list_header_h = 120
draw.rectangle(
    [(0, list_top), (1440, list_top + list_header_h)],
    fill="#ffffff",
    outline=None
)
# thin divider under the listings header
draw.line([(24, list_top + list_header_h), (1416, list_top + list_header_h)], fill="#ebecec", width=1)

# 7) Row separators for two list items (visual structure only)
# First listing row area (illustrative separators only)
first_row_top = list_top + list_header_h + 28
row_height = 220
sep_y1 = first_row_top + row_height
sep_y2 = sep_y1 + row_height + 24

# separators between listing rows
draw.line([(24, sep_y1), (1416, sep_y1)], fill="#efefef", width=1)
draw.line([(24, sep_y2), (1416, sep_y2)], fill="#efefef", width=1)

# 8) Left-side thumbnail background boxes (light rounded squares where listing thumbnails will be pasted)
thumb_w = 210
thumb_h = 170
thumb_x = 48
thumb_y1 = first_row_top + 20
thumb_y2 = first_row_top + 20 + row_height + 24

draw.rounded_rectangle(
    [(thumb_x, thumb_y1), (thumb_x + thumb_w, thumb_y1 + thumb_h)],
    radius=18,
    fill="#f0f2f6",
    outline="#e0e2e6",
    width=2
)
draw.rounded_rectangle(
    [(thumb_x, thumb_y2), (thumb_x + thumb_w, thumb_y2 + thumb_h)],
    radius=18,
    fill="#f0f2f6",
    outline="#e0e2e6",
    width=2
)

# 9) Right side small divider hint for the sort control (visual only)
# a small vertical separator near the right edge of the listing header (the actual sort icon will be pasted)
draw.line([(1080, list_top + 24), (1080, list_top + list_header_h - 24)], fill="#f0f0f0", width=1)

# End of structural/background drawing.
# All actual icons/text/buttons will be pasted on top of these shapes (do not redraw those elements here).

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_06_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-9/00_icon_Include_fees.png
try:
    _c0 = get_crop(0, 335, 108)
    canvas.paste(_c0, (544, 312), _c0)
except Exception:
    pass
layout["Include_fees"] = [544, 312, 879, 420]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_06_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-9/01_icon_2_Listings.png
try:
    _c1 = get_crop(1, 1440, 455)
    canvas.paste(_c1, (0, 2134), _c1)
except Exception:
    pass
layout["2_Listings"] = [0, 2134, 1440, 2589]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_06_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-9/02_icon_STAGE.png
try:
    _c2 = get_crop(2, 603, 371)
    canvas.paste(_c2, (415, 950), _c2)
except Exception:
    pass
layout["STAGE"] = [415, 950, 1018, 1321]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_06_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-9/03_icon_Best_seats.png
try:
    _c3 = get_crop(3, 303, 108)
    canvas.paste(_c3, (915, 312), _c3)
except Exception:
    pass
layout["Best_seats"] = [915, 312, 1218, 420]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_06_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-9/04_icon_9.1.png
try:
    _c4 = get_crop(4, 1440, 371)
    canvas.paste(_c4, (0, 2589), _c4)
except Exception:
    pass
layout["9.1"] = [0, 2589, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_06_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-9/05_icon_Quantity.png
try:
    _c5 = get_crop(5, 268, 108)
    canvas.paste(_c5, (240, 312), _c5)
except Exception:
    pass
layout["Quantity"] = [240, 312, 508, 420]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_06_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-9/06_icon_Tit.png
try:
    _c6 = get_crop(6, 156, 108)
    canvas.paste(_c6, (48, 312), _c6)
except Exception:
    pass
layout["Tit"] = [48, 312, 204, 420]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_06_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-9/07_icon_Mkgee.png
try:
    _c7 = get_crop(7, 65, 63)
    canvas.paste(_c7, (240, 2), _c7)
except Exception:
    pass
layout["Mkgee"] = [240, 2, 305, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_06_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-9/08_icon_Mkgee.png
try:
    _c8 = get_crop(8, 62, 64)
    canvas.paste(_c8, (310, 2), _c8)
except Exception:
    pass
layout["Mkgee"] = [310, 2, 372, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_06_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-9/09_icon_GA_BALCONY.png
try:
    _c9 = get_crop(9, 226, 232)
    canvas.paste(_c9, (605, 1361), _c9)
except Exception:
    pass
layout["GA_BALCONY"] = [605, 1361, 831, 1593]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_06_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-9/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 102, 63)
    canvas.paste(_c10, (1213, 1), _c10)
except Exception:
    pass
layout["icon_10"] = [1213, 1, 1315, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_06_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-9/11_icon_8.11_Wy.png
try:
    _c11 = get_crop(11, 69, 64)
    canvas.paste(_c11, (109, 0), _c11)
except Exception:
    pass
layout["8.11_Wy"] = [109, 0, 178, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_06_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-9/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 51, 64)
    canvas.paste(_c12, (1152, 1), _c12)
except Exception:
    pass
layout["icon_12"] = [1152, 1, 1203, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_06_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-9/13_icon_8.11_Wy.png
try:
    _c13 = get_crop(13, 54, 62)
    canvas.paste(_c13, (182, 1), _c13)
except Exception:
    pass
layout["8.11_Wy"] = [182, 1, 236, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_06_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-9/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 53, 57)
    canvas.paste(_c14, (1320, 3), _c14)
except Exception:
    pass
layout["icon_14"] = [1320, 3, 1373, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_06_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-9/15_icon_STAGE.png
try:
    _c15 = get_crop(15, 412, 221)
    canvas.paste(_c15, (511, 693), _c15)
except Exception:
    pass
layout["STAGE"] = [511, 693, 923, 914]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_06_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-9/16_icon_Include_fees.png
try:
    _c16 = get_crop(16, 1344, 156)
    canvas.paste(_c16, (48, 120), _c16)
except Exception:
    pass
layout["Include_fees"] = [48, 120, 1392, 276]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_06_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-9/17_icon_Amazing_deal.png
try:
    _c17 = get_crop(17, 1440, 455)
    canvas.paste(_c17, (0, 2134), _c17)
except Exception:
    pass
layout["Amazing_deal"] = [0, 2134, 1440, 2589]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_06_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-9/18_icon_icon_18.png
try:
    _c18 = get_crop(18, 50, 65)
    canvas.paste(_c18, (383, 1), _c18)
except Exception:
    pass
layout["icon_18"] = [383, 1, 433, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_06_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-9/19_icon_Info.png
try:
    _c19 = get_crop(19, 156, 156)
    canvas.paste(_c19, (1236, 120), _c19)
except Exception:
    pass
layout["Info"] = [1236, 120, 1392, 276]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_06_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-9/20_icon_Sort_by_price.png
try:
    _c20 = get_crop(20, 455, 144)
    canvas.paste(_c20, (961, 1989), _c20)
except Exception:
    pass
layout["Sort_by_price"] = [961, 1989, 1416, 2133]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_06_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-9/21_text_2_Listings.png
try:
    _c21 = get_crop(21, 260, 68)
    canvas.paste(_c21, (57, 2032), _c21)
except Exception:
    pass
layout["2_Listings"] = [57, 2032, 317, 2100]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_06_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-9/22_text_S162_each.png
try:
    _c22 = get_crop(22, 1440, 371)
    canvas.paste(_c22, (0, 2589), _c22)
except Exception:
    pass
layout["S162_each"] = [0, 2589, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_06_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-9/23_text_Price_includes_fees.png
try:
    _c23 = get_crop(23, 1440, 371)
    canvas.paste(_c23, (0, 2589), _c23)
except Exception:
    pass
layout["Price_includes_fees"] = [0, 2589, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_06_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-9/24_text_9.1.png
try:
    _c24 = get_crop(24, 41, 31)
    canvas.paste(_c24, (502, 2810), _c24)
except Exception:
    pass
layout["9.1"] = [502, 2810, 543, 2841]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_06_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-9/25_text_Amazing_deal.png
try:
    _c25 = get_crop(25, 1440, 371)
    canvas.paste(_c25, (0, 2589), _c25)
except Exception:
    pass
layout["Amazing_deal"] = [0, 2589, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_06_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-9/26_text_1-2_tickets.png
try:
    _c26 = get_crop(26, 216, 48)
    canvas.paste(_c26, (488, 2872), _c26)
except Exception:
    pass
layout["1-2_tickets"] = [488, 2872, 704, 2920]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_06_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-9/27_clickable_Back.png
try:
    _c27 = get_crop(27, 156, 156)
    canvas.paste(_c27, (48, 120), _c27)
except Exception:
    pass
layout["Back"] = [48, 120, 204, 276]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_06_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-9/28_clickable_Mk.gee.png
try:
    _c28 = get_crop(28, 326, 156)
    canvas.paste(_c28, (204, 120), _c28)
except Exception:
    pass
layout["Mk.gee"] = [204, 120, 530, 276]
