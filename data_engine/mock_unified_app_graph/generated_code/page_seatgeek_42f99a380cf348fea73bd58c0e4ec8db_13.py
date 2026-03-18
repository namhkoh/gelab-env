# page_id: page_seatgeek_42f99a380cf348fea73bd58c0e4ec8db_13
# screenshot: 2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-16.png
# step_index: 13/14
# task: Open SeatGeek and search for the broadway show "lion king" on March 22. I need 3 tickets at average price less than 500 USD. Find the best seats and record the total price in google keep notes.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Background
bg_color = (240, 243, 246)  # light neutral bluish-gray
draw.rectangle([(0, 0), canvas.size], fill=bg_color)

# Status bar area (top ~64px)
status_h = 64
status_color = (225, 229, 232)  # slightly darker than background
draw.rectangle([(0, 0), (1440, status_h)], fill=status_color)
# subtle divider under status bar
draw.line([(0, status_h), (1440, status_h)], fill=(208, 213, 217), width=1)

# Header / toolbar background (rounded pill behind title area)
header_box = (40, 72, 1400, 168)
header_shadow_offset = 6
# shadow
draw.rounded_rectangle(
    [header_box[0], header_box[1] + header_shadow_offset,
     header_box[2], header_box[3] + header_shadow_offset],
    radius=48, fill=(215, 219, 223))
# white header pill
draw.rounded_rectangle(header_box, radius=48, fill=(255, 255, 255))
# thin divider line at bottom of header
draw.line([(header_box[0] + 8, header_box[3]), (header_box[2] - 8, header_box[3])],
          fill=(235, 238, 240), width=1)

# Map / seating card background (big centered card)
map_box = (200, 300, 1240, 1120)
map_shadow_offset = 10
# shadow behind map card
draw.rounded_rectangle(
    [map_box[0], map_box[1] + map_shadow_offset, map_box[2], map_box[3] + map_shadow_offset],
    radius=14, fill=(210, 215, 219))
# map card background (very light)
draw.rounded_rectangle(map_box, radius=14, fill=(250, 250, 252))
# inner thin outline around the map card to match screenshot style
outline_inset = 6
draw.rounded_rectangle(
    [map_box[0] + outline_inset, map_box[1] + outline_inset,
     map_box[2] - outline_inset, map_box[3] - outline_inset],
    radius=10, outline=(200, 204, 208), width=6)

# Optional dashed guide rectangles left and right of map (subtle, as in screenshot)
dash_color = (203, 207, 210)
# left dashed vertical
lx1, ly1, lx2, ly2 = 120, map_box[1] + 120, 180, map_box[3] - 120
x = lx1
dash_len = 8
gap = 6
# draw dashed rectangle (left)
x0, y0, x1, y1 = lx1, ly1, lx2, ly2
# top
cx = x0
while cx < x1:
    draw.line([(cx, y0), (min(cx + dash_len, x1), y0)], fill=dash_color, width=1)
    cx += dash_len + gap
# right
cy = y0
while cy < y1:
    draw.line([(x1, cy), (x1, min(cy + dash_len, y1))], fill=dash_color, width=1)
    cy += dash_len + gap
# bottom
cx = x1
while cx > x0:
    draw.line([(max(cx - dash_len, x0), y1), (cx, y1)], fill=dash_color, width=1)
    cx -= dash_len + gap
# left
cy = y1
while cy > y0:
    draw.line([(x0, max(cy - dash_len, y0)), (x0, cy)], fill=dash_color, width=1)
    cy -= dash_len + gap

# right dashed rectangle mirrored
rx1, ry1, rx2, ry2 = 1260, map_box[1] + 120, 1320, map_box[3] - 120
# top
cx = rx1
while cx < rx2:
    draw.line([(cx, ry1), (min(cx + dash_len, rx2), ry1)], fill=dash_color, width=1)
    cx += dash_len + gap
# right
cy = ry1
while cy < ry2:
    draw.line([(rx2, cy), (rx2, min(cy + dash_len, ry2))], fill=dash_color, width=1)
    cy += dash_len + gap
# bottom
cx = rx2
while cx > rx1:
    draw.line([(max(cx - dash_len, rx1), ry2), (cx, ry2)], fill=dash_color, width=1)
    cx -= dash_len + gap
# left
cy = ry2
while cy > ry1:
    draw.line([(rx1, max(cy - dash_len, ry1)), (rx1, cy)], fill=dash_color, width=1)
    cy -= dash_len + gap

# Lower "listings" card background (white sheet with rounded top corners)
list_box = (0, 1920, 1440, 2960)
# subtle shadow above
draw.rectangle([ (0, 1912), (1440, 1920) ], fill=(220, 224, 227))
draw.rounded_rectangle(list_box, radius=32, fill=(255, 255, 255))
# top divider line for listings header
draw.line([(24, 1988), (1416, 1988)], fill=(232, 235, 237), width=2)

# Section separator lines for listing items (subtle)
sep_color = (240, 241, 243)
# approximate positions for the two main listing rows seen in screenshot
draw.line([(24, 2360), (1416, 2360)], fill=sep_color, width=1)
draw.line([(24, 2720), (1416, 2720)], fill=sep_color, width=1)

# Small accent bar behind sorting control area (right side of listings header)
sort_box = (980, 1950, 1400, 2008)
draw.rounded_rectangle(sort_box, radius=14, fill=(255, 255, 255))
draw.rectangle([(980, 1950), (1400, 1964)], fill=(255,255,255))  # keep top area crisp
# small chevron separator at header right (visual only, not a real icon)
draw.line([(960, 1964), (960, 1996)], fill=(235, 238, 240), width=1)

# Final subtle vignette at bottom to anchor page
draw.rectangle([(0, 2900), (1440, 2960)], fill=(248, 249, 250))

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_13_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-16/00_icon_Include_fees.png
try:
    _c0 = get_crop(0, 335, 108)
    canvas.paste(_c0, (543, 312), _c0)
except Exception:
    pass
layout["Include_fees"] = [543, 312, 878, 420]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_13_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-16/01_icon_Best_seats.png
try:
    _c1 = get_crop(1, 303, 108)
    canvas.paste(_c1, (914, 312), _c1)
except Exception:
    pass
layout["Best_seats"] = [914, 312, 1217, 420]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_13_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-16/02_icon_3_tickets.png
try:
    _c2 = get_crop(2, 267, 108)
    canvas.paste(_c2, (240, 312), _c2)
except Exception:
    pass
layout["3_tickets"] = [240, 312, 507, 420]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_13_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-16/03_icon_STAGE.png
try:
    _c3 = get_crop(3, 528, 268)
    canvas.paste(_c3, (458, 638), _c3)
except Exception:
    pass
layout["STAGE"] = [458, 638, 986, 906]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_13_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-16/04_icon_Low_pr.png
try:
    _c4 = get_crop(4, 187, 108)
    canvas.paste(_c4, (1253, 312), _c4)
except Exception:
    pass
layout["Low_pr"] = [1253, 312, 1440, 420]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_13_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-16/05_icon_Tit.png
try:
    _c5 = get_crop(5, 156, 108)
    canvas.paste(_c5, (48, 312), _c5)
except Exception:
    pass
layout["Tit"] = [48, 312, 204, 420]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_13_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-16/06_icon_7.3.png
try:
    _c6 = get_crop(6, 1440, 455)
    canvas.paste(_c6, (0, 2355), _c6)
except Exception:
    pass
layout["7.3"] = [0, 2355, 1440, 2810]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_13_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-16/07_icon_New_York.png
try:
    _c7 = get_crop(7, 496, 156)
    canvas.paste(_c7, (204, 120), _c7)
except Exception:
    pass
layout["New_York"] = [204, 120, 700, 276]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_13_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-16/08_icon_GEK.png
try:
    _c8 = get_crop(8, 60, 60)
    canvas.paste(_c8, (244, 1), _c8)
except Exception:
    pass
layout["GEK"] = [244, 1, 304, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_13_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-16/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 105, 64)
    canvas.paste(_c9, (1212, 0), _c9)
except Exception:
    pass
layout["icon_9"] = [1212, 0, 1317, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_13_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-16/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 51, 66)
    canvas.paste(_c10, (1152, 1), _c10)
except Exception:
    pass
layout["icon_10"] = [1152, 1, 1203, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_13_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-16/11_icon_Wy.png
try:
    _c11 = get_crop(11, 62, 63)
    canvas.paste(_c11, (111, 1), _c11)
except Exception:
    pass
layout["Wy"] = [111, 1, 173, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_13_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-16/12_icon_Wy.png
try:
    _c12 = get_crop(12, 57, 61)
    canvas.paste(_c12, (180, 1), _c12)
except Exception:
    pass
layout["Wy"] = [180, 1, 237, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_13_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-16/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 54, 61)
    canvas.paste(_c13, (1319, 1), _c13)
except Exception:
    pass
layout["icon_13"] = [1319, 1, 1373, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_13_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-16/14_icon_Great_deal.png
try:
    _c14 = get_crop(14, 1440, 455)
    canvas.paste(_c14, (0, 2355), _c14)
except Exception:
    pass
layout["Great_deal"] = [0, 2355, 1440, 2810]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_13_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-16/15_icon_7.42.png
try:
    _c15 = get_crop(15, 99, 64)
    canvas.paste(_c15, (8, 0), _c15)
except Exception:
    pass
layout["7.42"] = [8, 0, 107, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_13_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-16/16_icon_Low_pr.png
try:
    _c16 = get_crop(16, 156, 156)
    canvas.paste(_c16, (1236, 120), _c16)
except Exception:
    pass
layout["Low_pr"] = [1236, 120, 1392, 276]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_13_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-16/17_icon_MEZZANINE.png
try:
    _c17 = get_crop(17, 455, 144)
    canvas.paste(_c17, (961, 1989), _c17)
except Exception:
    pass
layout["MEZZANINE"] = [961, 1989, 1416, 2133]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_13_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-16/18_icon_S434_each.png
try:
    _c18 = get_crop(18, 384, 106)
    canvas.paste(_c18, (52, 2854), _c18)
except Exception:
    pass
layout["S434_each"] = [52, 2854, 436, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_13_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-16/19_text_ORCHESTRA.png
try:
    _c19 = get_crop(19, 138, 29)
    canvas.paste(_c19, (650, 997), _c19)
except Exception:
    pass
layout["ORCHESTRA"] = [650, 997, 788, 1026]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_13_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-16/20_text_BOXL.png
try:
    _c20 = get_crop(20, 71, 27)
    canvas.paste(_c20, (326, 1420), _c20)
except Exception:
    pass
layout["BOXL"] = [326, 1420, 397, 1447]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_13_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-16/21_text_BOX_R.png
try:
    _c21 = get_crop(21, 73, 25)
    canvas.paste(_c21, (1043, 1422), _c21)
except Exception:
    pass
layout["BOX_R"] = [1043, 1422, 1116, 1447]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_13_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-16/22_text_2_Listings.png
try:
    _c22 = get_crop(22, 260, 68)
    canvas.paste(_c22, (57, 2032), _c22)
except Exception:
    pass
layout["2_Listings"] = [57, 2032, 317, 2100]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_13_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-16/23_text_Sort_by_price.png
try:
    _c23 = get_crop(23, 455, 144)
    canvas.paste(_c23, (961, 1989), _c23)
except Exception:
    pass
layout["Sort_by_price"] = [961, 1989, 1416, 2133]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_13_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-16/24_text_We_sell_resale_tickets._Resale_tickets_m.png
try:
    _c24 = get_crop(24, 1440, 455)
    canvas.paste(_c24, (0, 2355), _c24)
except Exception:
    pass
layout["€_We_sell_resale_tickets."] = [0, 2355, 1440, 2810]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_13_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-16/25_text_face_value.png
try:
    _c25 = get_crop(25, 218, 43)
    canvas.paste(_c25, (57, 2256), _c25)
except Exception:
    pass
layout["face_value:"] = [57, 2256, 275, 2299]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_13_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-16/26_text_S434_each.png
try:
    _c26 = get_crop(26, 276, 61)
    canvas.paste(_c26, (485, 2862), _c26)
except Exception:
    pass
layout["S434_each"] = [485, 2862, 761, 2923]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_13_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-16/27_clickable_Back.png
try:
    _c27 = get_crop(27, 156, 156)
    canvas.paste(_c27, (48, 120), _c27)
except Exception:
    pass
layout["Back"] = [48, 120, 204, 276]
