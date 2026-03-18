# page_id: page_seatgeek_42f99a380cf348fea73bd58c0e4ec8db_09
# screenshot: 2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-12.png
# step_index: 9/14
# task: Open SeatGeek and search for the broadway show "lion king" on March 22. I need 3 tickets at average price less than 500 USD. Find the best seats and record the total price in google keep notes.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Background and structural elements for the mobile UI page
# Assumes variables provided: canvas (PIL.Image), draw (PIL.ImageDraw), font_sm, font_md, font_lg, font_xl

# Colors (matching the screenshot's soft bluish-gray theme)
BG = (240, 243, 246)          # main page background
STATUS_BG = (226, 229, 232)   # top status bar
HEADER_CARD = (255, 255, 255) # white header/search pill
SUBTLE = (246, 247, 248)      # subtle card background
BORDER = (210, 214, 218)      # light border lines
DIVIDER = (224, 226, 228)     # divider lines
SHADOW = (220, 224, 227)      # soft shadow

W, H = canvas.size

# Fill overall background
draw.rectangle([(0, 0), (W, H)], fill=BG)

# Status bar area ~ top 72px (slightly darker than page bg)
status_height = 72
draw.rectangle([(0, 0), (W, status_height)], fill=STATUS_BG)

# Subtle bottom edge shadow under status bar
draw.line([(0, status_height), (W, status_height)], fill=DIVIDER, width=1)

# Header/search pill (rounded) centered near top; leave inner area for icons/text (do not draw text/icons)
header_left = 36
header_top = 76
header_right = W - 36
header_bottom = 200
header_radius = 60
# Shadow for the header pill (soft offset)
shadow_offset = 6
draw.rounded_rectangle(
    [header_left, header_top + shadow_offset, header_right, header_bottom + shadow_offset],
    radius=header_radius,
    fill=SHADOW
)
draw.rounded_rectangle(
    [header_left, header_top, header_right, header_bottom],
    radius=header_radius,
    fill=HEADER_CARD,
    outline=BORDER,
    width=1
)

# Row area for filter chips (pills) - background influence only (no chips drawn)
# We'll provide a subtle band to suggest the region behind chips.
filters_top = header_bottom + 28
filters_bottom = filters_top + 120
band_margin = 48
draw.rectangle([(band_margin, filters_top), (W - band_margin, filters_bottom)], fill=BG)
# faint dividing line below filters
draw.line([(band_margin, filters_bottom + 8), (W - band_margin, filters_bottom + 8)], fill=DIVIDER, width=1)

# Seating/map container card (large centered rounded card)
map_left = 200
map_top = filters_bottom + 36
map_right = W - 200
map_bottom = map_top + 920
map_radius = 18
# subtle drop shadow (larger, behind)
draw.rounded_rectangle(
    [map_left, map_top + 12, map_right, map_bottom + 12],
    radius=map_radius,
    fill=SHADOW
)
# main map card
draw.rounded_rectangle(
    [map_left, map_top, map_right, map_bottom],
    radius=map_radius,
    fill=SUBTLE,
    outline=BORDER,
    width=2
)

# Inner subtle inset border to give depth (do not draw any icons/text inside)
inset_padding = 18
draw.rounded_rectangle(
    [map_left + inset_padding, map_top + inset_padding, map_right - inset_padding, map_bottom - inset_padding],
    radius=12,
    outline=(235,237,239),
    width=1
)

# Decorative dashed outline boxes to the left/right of the map (to mimic layout accents)
# Draw as faint dotted rectangles (do not overlap or duplicate detected icons content)
dash_box_w = 220
dash_box_h = 320
# left dashed box
lx = map_left - 120
ly = map_top + 120
rx = lx + dash_box_w
ry = ly + dash_box_h
# draw dotted rectangle manually
dot_spacing = 12
for x in range(lx, rx, dot_spacing):
    draw.point((x, ly), fill=DIVIDER)
    draw.point((x, ry), fill=DIVIDER)
for y in range(ly, ry, dot_spacing):
    draw.point((lx, y), fill=DIVIDER)
    draw.point((rx, y), fill=DIVIDER)

# right dashed box mirrored
rx2 = map_right + 120
lx2 = rx2 - dash_box_w
ry2 = ly + dash_box_h
for x in range(lx2, rx2, dot_spacing):
    draw.point((x, ly), fill=DIVIDER)
    draw.point((x, ry2), fill=DIVIDER)
for y in range(ly, ry2, dot_spacing):
    draw.point((lx2, y), fill=DIVIDER)
    draw.point((rx2, y), fill=DIVIDER)

# Listings container card anchored near bottom (rounded top corners)
list_top = map_bottom + 80
list_left = 0
list_right = W
list_bottom = H
list_radius = 36
# shadow for the listings card
draw.rounded_rectangle([list_left, list_top + 8, list_right, list_bottom + 8], radius=list_radius, fill=SHADOW)
draw.rounded_rectangle([list_left, list_top, list_right, list_bottom], radius=list_radius, fill=HEADER_CARD, outline=BORDER, width=1)

# Divider under listings header area (reserve top portion for "36 Listings" etc; do not draw those labels)
header_area_height = 140
draw.line([(24, list_top + header_area_height), (W - 24, list_top + header_area_height)], fill=DIVIDER, width=1)

# A few separators for the listing items (do not draw thumbnails/text; just structural dividers)
first_item_top = list_top + header_area_height + 28
item_height = 260
sep_y = first_item_top + item_height
# draw two item separators to indicate multiple listings
draw.line([(24, sep_y), (W - 24, sep_y)], fill=DIVIDER, width=1)
draw.line([(24, sep_y + item_height), (W - 24, sep_y + item_height)], fill=DIVIDER, width=1)

# Left thumbnail rounded backgrounds for list items (structural only, no thumbnails)
thumb_size = (200, 200)
thumb_x = 48
thumb_y = first_item_top + 16
# draw a muted rounded rectangle placeholder for each item (these will be replaced by pasted thumbnails)
for i in range(2):
    y0 = thumb_y + i * item_height
    x0 = thumb_x
    x1 = x0 + thumb_size[0]
    y1 = y0 + thumb_size[1]
    draw.rounded_rectangle([x0, y0, x1, y1], radius=18, fill=SUBTLE, outline=BORDER)

# Small subtle vertical separators and accent lines inside the listings area to reinforce structure
# Right aligned vertical line for sort control area (structural only)
sort_x = W - 280
sort_y0 = list_top + 28
sort_y1 = list_top + header_area_height - 20
draw.line([(sort_x, sort_y0), (sort_x, sort_y1)], fill=DIVIDER, width=1)

# Final subtle top shadow line for the whole listings area
draw.line([(0, list_top), (W, list_top)], fill=DIVIDER, width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_09_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-12/00_icon_Include_fees.png
try:
    _c0 = get_crop(0, 335, 108)
    canvas.paste(_c0, (543, 312), _c0)
except Exception:
    pass
layout["Include_fees"] = [543, 312, 878, 420]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_09_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-12/01_icon_3_tickets.png
try:
    _c1 = get_crop(1, 267, 108)
    canvas.paste(_c1, (240, 312), _c1)
except Exception:
    pass
layout["3_tickets"] = [240, 312, 507, 420]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_09_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-12/02_icon_Best_seats.png
try:
    _c2 = get_crop(2, 303, 108)
    canvas.paste(_c2, (914, 312), _c2)
except Exception:
    pass
layout["Best_seats"] = [914, 312, 1217, 420]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_09_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-12/03_icon_STAGE.png
try:
    _c3 = get_crop(3, 528, 268)
    canvas.paste(_c3, (458, 638), _c3)
except Exception:
    pass
layout["STAGE"] = [458, 638, 986, 906]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_09_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-12/04_icon_Tit.png
try:
    _c4 = get_crop(4, 156, 108)
    canvas.paste(_c4, (48, 312), _c4)
except Exception:
    pass
layout["Tit"] = [48, 312, 204, 420]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_09_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-12/05_icon_Low_pr.png
try:
    _c5 = get_crop(5, 187, 108)
    canvas.paste(_c5, (1253, 312), _c5)
except Exception:
    pass
layout["Low_pr"] = [1253, 312, 1440, 420]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_09_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-12/06_icon_7.3.png
try:
    _c6 = get_crop(6, 1440, 455)
    canvas.paste(_c6, (0, 2355), _c6)
except Exception:
    pass
layout["7.3"] = [0, 2355, 1440, 2810]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_09_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-12/07_icon_New_York.png
try:
    _c7 = get_crop(7, 496, 156)
    canvas.paste(_c7, (204, 120), _c7)
except Exception:
    pass
layout["New_York"] = [204, 120, 700, 276]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_09_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-12/08_icon_GEK.png
try:
    _c8 = get_crop(8, 61, 60)
    canvas.paste(_c8, (244, 1), _c8)
except Exception:
    pass
layout["GEK"] = [244, 1, 305, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_09_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-12/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 51, 65)
    canvas.paste(_c9, (1152, 1), _c9)
except Exception:
    pass
layout["icon_9"] = [1152, 1, 1203, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_09_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-12/10_icon_0.png
try:
    _c10 = get_crop(10, 104, 63)
    canvas.paste(_c10, (1212, 1), _c10)
except Exception:
    pass
layout["0#"] = [1212, 1, 1316, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_09_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-12/11_icon_my.png
try:
    _c11 = get_crop(11, 63, 63)
    canvas.paste(_c11, (110, 0), _c11)
except Exception:
    pass
layout["my"] = [110, 0, 173, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_09_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-12/12_icon_my.png
try:
    _c12 = get_crop(12, 55, 62)
    canvas.paste(_c12, (181, 0), _c12)
except Exception:
    pass
layout["my"] = [181, 0, 236, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_09_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-12/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 53, 57)
    canvas.paste(_c13, (1320, 3), _c13)
except Exception:
    pass
layout["icon_13"] = [1320, 3, 1373, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_09_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-12/14_icon_0.png
try:
    _c14 = get_crop(14, 156, 156)
    canvas.paste(_c14, (1236, 120), _c14)
except Exception:
    pass
layout["0#"] = [1236, 120, 1392, 276]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_09_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-12/15_icon_Great_deal.png
try:
    _c15 = get_crop(15, 1440, 455)
    canvas.paste(_c15, (0, 2355), _c15)
except Exception:
    pass
layout["Great_deal"] = [0, 2355, 1440, 2810]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_09_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-12/16_icon_7.41.png
try:
    _c16 = get_crop(16, 98, 64)
    canvas.paste(_c16, (7, 0), _c16)
except Exception:
    pass
layout["7.41"] = [7, 0, 105, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_09_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-12/17_icon_BOXL.png
try:
    _c17 = get_crop(17, 455, 144)
    canvas.paste(_c17, (961, 1989), _c17)
except Exception:
    pass
layout["BOXL"] = [961, 1989, 1416, 2133]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_09_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-12/18_text_ORCHESTRA.png
try:
    _c18 = get_crop(18, 138, 29)
    canvas.paste(_c18, (650, 997), _c18)
except Exception:
    pass
layout["ORCHESTRA"] = [650, 997, 788, 1026]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_09_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-12/19_text_BOXL.png
try:
    _c19 = get_crop(19, 71, 27)
    canvas.paste(_c19, (326, 1420), _c19)
except Exception:
    pass
layout["BOXL"] = [326, 1420, 397, 1447]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_09_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-12/20_text_BOX_R.png
try:
    _c20 = get_crop(20, 73, 25)
    canvas.paste(_c20, (1043, 1422), _c20)
except Exception:
    pass
layout["BOX_R"] = [1043, 1422, 1116, 1447]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_09_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-12/21_text_36_Listings.png
try:
    _c21 = get_crop(21, 300, 77)
    canvas.paste(_c21, (56, 2028), _c21)
except Exception:
    pass
layout["36_Listings"] = [56, 2028, 356, 2105]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_09_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-12/22_text_Sort_by_price.png
try:
    _c22 = get_crop(22, 455, 144)
    canvas.paste(_c22, (961, 1989), _c22)
except Exception:
    pass
layout["Sort_by_price"] = [961, 1989, 1416, 2133]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_09_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-12/23_text_We_sell_resale_tickets._Resale_tickets_m.png
try:
    _c23 = get_crop(23, 1440, 455)
    canvas.paste(_c23, (0, 2355), _c23)
except Exception:
    pass
layout["€_We_sell_resale_tickets."] = [0, 2355, 1440, 2810]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_09_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-12/24_text_face_value.png
try:
    _c24 = get_crop(24, 218, 43)
    canvas.paste(_c24, (57, 2256), _c24)
except Exception:
    pass
layout["face_value:"] = [57, 2256, 275, 2299]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_09_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-12/25_text_S266_each.png
try:
    _c25 = get_crop(25, 277, 65)
    canvas.paste(_c25, (485, 2862), _c25)
except Exception:
    pass
layout["S266_each"] = [485, 2862, 762, 2927]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_09_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-12/26_clickable_Back.png
try:
    _c26 = get_crop(26, 156, 156)
    canvas.paste(_c26, (48, 120), _c26)
except Exception:
    pass
layout["Back"] = [48, 120, 204, 276]
