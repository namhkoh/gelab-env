# page_id: page_seatgeek_3675f36a85734af4aa90c8115351dd12_09
# screenshot: 2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-12.png
# step_index: 9/9
# task: Open SeatGeek. Search "The Fonda Theatre". Select the top popular event and track it. What is the lowest price?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: fallback_compose
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Paint UI background and structural elements for the provided canvas.
# Uses: canvas (PIL Image 1440x2960), draw (ImageDraw), font_sm/font_md/font_lg/font_xl

w, h = canvas.size

# Colors (approximate from screenshot)
bg = (239, 242, 245)         # main page background (very light bluish gray)
status_bar = (224, 226, 229) # top status bar slightly darker
header_strip = (246, 247, 249) # header area behind the pill/search
card_border = (189, 192, 196)  # subtle grey border for map/card
card_shadow = (212, 215, 218)  # shadow for cards
white = (255, 255, 255)
divider = (227, 230, 233)

# Fill overall background
draw.rectangle([(0, 0), (w, h)], fill=bg)

# Status bar area (top ~80px)
status_h = 80
draw.rectangle([(0, 0), (w, status_h)], fill=status_bar)

# Header area (below status bar)
header_y0 = status_h
header_y1 = 260
draw.rectangle([(0, header_y0), (w, header_y1)], fill=header_strip)

# Subtle bottom divider under header
draw.line([(24, header_y1), (w-24, header_y1)], fill=divider, width=1)

# Central seating-map container (outer frame + inner canvas)
map_w, map_h = 640, 900
map_x0 = (w - map_w) // 2
map_y0 = 360
map_x1 = map_x0 + map_w
map_y1 = map_y0 + map_h

# Outer border/shadow for map card
outer_pad = 10
draw.rounded_rectangle(
    [(map_x0-outer_pad, map_y0-outer_pad, map_x1+outer_pad, map_y1+outer_pad)],
    radius=14,
    fill=card_shadow
)

# White map card with grey border (frame)
draw.rounded_rectangle(
    [(map_x0, map_y0, map_x1, map_y1)],
    radius=12,
    fill=white,
    outline=card_border,
    width=6
)

# Inner content panel inside the map frame (slightly inset)
inset = 24
draw.rectangle(
    [(map_x0 + inset, map_y0 + inset, map_x1 - inset, map_y1 - inset)],
    fill=(247, 248, 250)
)

# A thin inner divider near the top of the map card (suggests stage area above content)
draw.line(
    [(map_x0 + 20, map_y0 + 120), (map_x1 - 20, map_y0 + 120)],
    fill=divider,
    width=2
)

# Bottom listings area card (rounded white panel)
list_top = 2020
list_radius = 36
draw.rounded_rectangle(
    [(16, list_top, w-16, h-16)],
    radius=list_radius,
    fill=white
)

# Shadow under listings panel (subtle)
draw.rectangle([(16, list_top-6, w-16, list_top)], fill=card_shadow)

# Small top divider inside listings card to separate header and content
draw.line([(32, list_top+96), (w-32, list_top+96)], fill=divider, width=1)

# Draw separators for two listing rows (visual structure only)
row_height = 220
first_row_y = list_top + 120
second_row_y = first_row_y + row_height
draw.line([(32, first_row_y + row_height), (w-32, first_row_y + row_height)], fill=divider, width=1)
draw.line([(32, first_row_y + 2*row_height), (w-32, first_row_y + 2*row_height)], fill=divider, width=1)

# Left thumbnail placeholders (rounded squares) for two listings - outline only (no icon or content)
thumb_w = 210
thumb_h = 140
thumb_rx = 20
thumb_x = 56
thumb_y1 = first_row_y + 20
thumb_y2 = second_row_y + 20

draw.rounded_rectangle([(thumb_x, thumb_y1, thumb_x+thumb_w, thumb_y1+thumb_h)], radius=12, fill=(242,244,247))
draw.rounded_rectangle([(thumb_x, thumb_y2, thumb_x+thumb_w, thumb_y2+thumb_h)], radius=12, fill=(242,244,247))

# Right-side thin vertical divider for listing header area (structure only)
header_div_x = w - 180
draw.line([(header_div_x, list_top + 28), (header_div_x, list_top + 140)], fill=divider, width=1)

# Small horizontal separators above the listing items to give a card-like effect
draw.line([(32, first_row_y - 12), (w-32, first_row_y - 12)], fill=divider, width=1)

# Small rounded indicator bar under the header pill area (structure hint, not a duplicate of pill)
pill_hint_y = 200
draw.rounded_rectangle([(32, pill_hint_y, w-32, pill_hint_y+6)], radius=3, fill=divider)

# Vertical spacing guide lines (very faint) to suggest layout columns (for pasting content alignment)
col_x1 = 56 + thumb_w + 36
draw.line([(col_x1, header_y1+12), (col_x1, list_top-12)], fill=(245,246,247), width=1)

# Top-to-map separator (thin)
map_sep_y = map_y0 - 34
draw.line([(32, map_sep_y), (w-32, map_sep_y)], fill=divider, width=1)

# Give a subtle rounded corner highlight to the map card top (visual structure)
draw.arc([map_x0, map_y0, map_x0+40, map_y0+40], start=180, end=270, fill=card_border, width=2)
draw.arc([map_x1-40, map_y0, map_x1, map_y0+40], start=270, end=360, fill=card_border, width=2)

# Final subtle bottom edge divider above the very bottom of screen
draw.line([(24, h-120), (w-24, h-120)], fill=divider, width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_09_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-12/00_icon_Include_fees.png
try:
    _c0 = get_crop(0, 335, 108)
    canvas.paste(_c0, (544, 312), _c0)
except Exception:
    pass
layout["Include_fees"] = [544, 312, 879, 420]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_09_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-12/01_icon_2_Listings.png
try:
    _c1 = get_crop(1, 1440, 455)
    canvas.paste(_c1, (0, 2134), _c1)
except Exception:
    pass
layout["2_Listings"] = [0, 2134, 1440, 2589]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_09_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-12/02_icon_STAGE.png
try:
    _c2 = get_crop(2, 603, 371)
    canvas.paste(_c2, (415, 950), _c2)
except Exception:
    pass
layout["STAGE"] = [415, 950, 1018, 1321]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_09_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-12/03_icon_Best_seats.png
try:
    _c3 = get_crop(3, 303, 108)
    canvas.paste(_c3, (915, 312), _c3)
except Exception:
    pass
layout["Best_seats"] = [915, 312, 1218, 420]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_09_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-12/04_icon_9.1.png
try:
    _c4 = get_crop(4, 1440, 371)
    canvas.paste(_c4, (0, 2589), _c4)
except Exception:
    pass
layout["9.1"] = [0, 2589, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_09_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-12/05_icon_Quantity.png
try:
    _c5 = get_crop(5, 268, 108)
    canvas.paste(_c5, (240, 312), _c5)
except Exception:
    pass
layout["Quantity"] = [240, 312, 508, 420]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_09_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-12/06_icon_Tit.png
try:
    _c6 = get_crop(6, 156, 108)
    canvas.paste(_c6, (48, 312), _c6)
except Exception:
    pass
layout["Tit"] = [48, 312, 204, 420]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_09_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-12/07_icon_GA_BALCONY.png
try:
    _c7 = get_crop(7, 226, 232)
    canvas.paste(_c7, (605, 1361), _c7)
except Exception:
    pass
layout["GA_BALCONY"] = [605, 1361, 831, 1593]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_09_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-12/08_icon_Mkgee.png
try:
    _c8 = get_crop(8, 65, 63)
    canvas.paste(_c8, (240, 2), _c8)
except Exception:
    pass
layout["Mkgee"] = [240, 2, 305, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_09_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-12/09_icon_Mkgee.png
try:
    _c9 = get_crop(9, 62, 64)
    canvas.paste(_c9, (310, 2), _c9)
except Exception:
    pass
layout["Mkgee"] = [310, 2, 372, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_09_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-12/10_icon_8.12_Wy.png
try:
    _c10 = get_crop(10, 66, 63)
    canvas.paste(_c10, (112, 1), _c10)
except Exception:
    pass
layout["8.12_Wy"] = [112, 1, 178, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_09_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-12/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 102, 63)
    canvas.paste(_c11, (1213, 1), _c11)
except Exception:
    pass
layout["icon_11"] = [1213, 1, 1315, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_09_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-12/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 51, 64)
    canvas.paste(_c12, (1152, 1), _c12)
except Exception:
    pass
layout["icon_12"] = [1152, 1, 1203, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_09_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-12/13_icon_8.12_Wy.png
try:
    _c13 = get_crop(13, 53, 61)
    canvas.paste(_c13, (182, 2), _c13)
except Exception:
    pass
layout["8.12_Wy"] = [182, 2, 235, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_09_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-12/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 53, 57)
    canvas.paste(_c14, (1320, 3), _c14)
except Exception:
    pass
layout["icon_14"] = [1320, 3, 1373, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_09_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-12/15_icon_STAGE.png
try:
    _c15 = get_crop(15, 412, 221)
    canvas.paste(_c15, (511, 693), _c15)
except Exception:
    pass
layout["STAGE"] = [511, 693, 923, 914]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_09_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-12/16_icon_Include_fees.png
try:
    _c16 = get_crop(16, 1344, 156)
    canvas.paste(_c16, (48, 120), _c16)
except Exception:
    pass
layout["Include_fees"] = [48, 120, 1392, 276]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_09_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-12/17_icon_Amazing_deal.png
try:
    _c17 = get_crop(17, 1440, 455)
    canvas.paste(_c17, (0, 2134), _c17)
except Exception:
    pass
layout["Amazing_deal"] = [0, 2134, 1440, 2589]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_09_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-12/18_icon_icon_18.png
try:
    _c18 = get_crop(18, 50, 65)
    canvas.paste(_c18, (383, 1), _c18)
except Exception:
    pass
layout["icon_18"] = [383, 1, 433, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_09_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-12/19_icon_Info.png
try:
    _c19 = get_crop(19, 156, 156)
    canvas.paste(_c19, (1236, 120), _c19)
except Exception:
    pass
layout["Info"] = [1236, 120, 1392, 276]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_09_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-12/20_icon_Sort_by_price.png
try:
    _c20 = get_crop(20, 455, 144)
    canvas.paste(_c20, (961, 1989), _c20)
except Exception:
    pass
layout["Sort_by_price"] = [961, 1989, 1416, 2133]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_09_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-12/21_text_2_Listings.png
try:
    _c21 = get_crop(21, 260, 68)
    canvas.paste(_c21, (57, 2032), _c21)
except Exception:
    pass
layout["2_Listings"] = [57, 2032, 317, 2100]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_09_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-12/22_text_S162_each.png
try:
    _c22 = get_crop(22, 1440, 371)
    canvas.paste(_c22, (0, 2589), _c22)
except Exception:
    pass
layout["S162_each"] = [0, 2589, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_09_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-12/23_text_Price_includes_fees.png
try:
    _c23 = get_crop(23, 1440, 371)
    canvas.paste(_c23, (0, 2589), _c23)
except Exception:
    pass
layout["Price_includes_fees"] = [0, 2589, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_09_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-12/24_text_9.1.png
try:
    _c24 = get_crop(24, 41, 31)
    canvas.paste(_c24, (502, 2810), _c24)
except Exception:
    pass
layout["9.1"] = [502, 2810, 543, 2841]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_09_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-12/25_text_Amazing_deal.png
try:
    _c25 = get_crop(25, 1440, 371)
    canvas.paste(_c25, (0, 2589), _c25)
except Exception:
    pass
layout["Amazing_deal"] = [0, 2589, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_09_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-12/26_text_1-2_tickets.png
try:
    _c26 = get_crop(26, 216, 48)
    canvas.paste(_c26, (488, 2872), _c26)
except Exception:
    pass
layout["1-2_tickets"] = [488, 2872, 704, 2920]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_09_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-12/27_clickable_Back.png
try:
    _c27 = get_crop(27, 156, 156)
    canvas.paste(_c27, (48, 120), _c27)
except Exception:
    pass
layout["Back"] = [48, 120, 204, 276]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_09_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-12/28_clickable_Mk.gee.png
try:
    _c28 = get_crop(28, 326, 156)
    canvas.paste(_c28, (204, 120), _c28)
except Exception:
    pass
layout["Mk.gee"] = [204, 120, 530, 276]
