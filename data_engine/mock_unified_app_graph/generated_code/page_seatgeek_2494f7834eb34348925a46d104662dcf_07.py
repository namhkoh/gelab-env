# page_id: page_seatgeek_2494f7834eb34348925a46d104662dcf_07
# screenshot: 2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-10.png
# step_index: 7/9
# task: Open SeatGeek. Search for "Book of Mormon". Add the show to favorite. Select date April 26. Set the ticket number to 2 and proceed. What is the lowest price for each ticket?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Canvas and draw are pre-provided. This script draws the background, top status area,
# header pill, seating-map cards, and the bottom content card with separators/shadows.
# Do not draw any icons or text -- those will be pasted separately.

canvas_w, canvas_h = canvas.size

# Colors
bg_color = "#F2F4F6"        # overall app background (light cool grey)
status_color = "#E8EAEC"    # status bar background
divider_color = "#DBDDE0"   # thin divider line
card_shadow = "#E6E8E9"     # subtle shadow for elevated elements
card_fill = "#FFFFFF"       # white cards
card_border = "#D0D2D3"     # card outlines / subtle borders
muted_fill = "#F8F9FA"      # slightly muted panel fill

# Fill background
draw.rectangle((0, 0, canvas_w, canvas_h), fill=bg_color)

# Top status bar (do NOT draw icons/text)
status_h = 70
draw.rectangle((0, 0, canvas_w, status_h), fill=status_color)
draw.line((0, status_h, canvas_w, status_h), fill=divider_color, width=1)

# Header / toolbar pill (rounded) - matches detected pill position and size
hdr_x, hdr_y = 48, 120
hdr_w, hdr_h = 1344, 156
hdr_radius = 78

# shadow for header pill
shadow_offset = 8
draw.rounded_rectangle(
    (hdr_x, hdr_y + shadow_offset, hdr_x + hdr_w, hdr_y + hdr_h + shadow_offset),
    radius=hdr_radius,
    fill=card_shadow,
    outline=None
)

# header pill background and subtle border
draw.rounded_rectangle(
    (hdr_x, hdr_y, hdr_x + hdr_w, hdr_y + hdr_h),
    radius=hdr_radius,
    fill=card_fill,
    outline=card_border,
    width=1
)

# Horizontal divider under filter area (chips area is auto-pasted so just a subtle guide)
filter_div_y = hdr_y + hdr_h + 32  # a bit below the pill where chips live
draw.line((48, filter_div_y, canvas_w - 48, filter_div_y), fill=divider_color, width=1)

# Seating map area (two stacked cards: orchestra map and mezzanine map)
map_center_x = canvas_w // 2

# Orchestra map card
seat_w, seat_h = 560, 460
seat_x = map_center_x - seat_w // 2
seat_y = filter_div_y + 60
seat_radius = 12

# subtle shadow
draw.rounded_rectangle(
    (seat_x, seat_y + 8, seat_x + seat_w, seat_y + seat_h + 8),
    radius=seat_radius,
    fill=card_shadow,
    outline=None
)
# main map card
draw.rounded_rectangle(
    (seat_x, seat_y, seat_x + seat_w, seat_y + seat_h),
    radius=seat_radius,
    fill=card_fill,
    outline=card_border,
    width=1
)
# stage/top area inset
stage_h = 110
draw.rounded_rectangle(
    (seat_x + 18, seat_y + 18, seat_x + seat_w - 18, seat_y + 18 + stage_h),
    radius=8,
    fill=muted_fill,
    outline=card_border,
    width=1
)

# Mezzanine map card below
mezz_h = 360
mezz_gap = 48
mezz_x = seat_x
mezz_y = seat_y + seat_h + mezz_gap
mezz_radius = 12

# shadow
draw.rounded_rectangle(
    (mezz_x, mezz_y + 8, mezz_x + seat_w, mezz_y + mezz_h + 8),
    radius=mezz_radius,
    fill=card_shadow,
    outline=None
)
draw.rounded_rectangle(
    (mezz_x, mezz_y, mezz_x + seat_w, mezz_y + mezz_h),
    radius=mezz_radius,
    fill=card_fill,
    outline=card_border,
    width=1
)

# Light dashed bounding guides on sides of the seating area (subtle, non-intrusive)
dash_color = "#E0E0E0"
dash_len = 8
# left vertical dashed
lx = seat_x - 42
for y in range(seat_y, mezz_y + mezz_h, dash_len * 2):
    draw.line((lx, y, lx, min(y + dash_len, mezz_y + mezz_h)), fill=dash_color, width=1)
# right vertical dashed
rx = seat_x + seat_w + 42
for y in range(seat_y, mezz_y + mezz_h, dash_len * 2):
    draw.line((rx, y, rx, min(y + dash_len, mezz_y + mezz_h)), fill=dash_color, width=1)

# Bottom content card (Box office & resale area)
bottom_top = mezz_y + mezz_h + 80
bottom_margin = 24
bottom_radius = 32

# shadow for bottom sheet
draw.rounded_rectangle(
    (bottom_margin, bottom_top + 10, canvas_w - bottom_margin, canvas_h - 24),
    radius=bottom_radius,
    fill=card_shadow,
    outline=None
)

# main bottom card
draw.rounded_rectangle(
    (bottom_margin, bottom_top, canvas_w - bottom_margin, canvas_h - 24),
    radius=bottom_radius,
    fill=card_fill,
    outline=card_border,
    width=1
)

# top separator line inside bottom card (section header divider)
section_header_h = bottom_top + 72
draw.line((bottom_margin + 24, section_header_h, canvas_w - bottom_margin - 24, section_header_h),
          fill=divider_color, width=1)

# subtle horizontal rule above content area (to mirror app's UI)
draw.line((bottom_margin + 12, bottom_top + 6, canvas_w - bottom_margin - 12, bottom_top + 6),
          fill=muted_fill, width=1)

# small decorative pill backgrounds for where badges/labels might appear (do NOT draw labels/text)
# These are just background shapes to match the original layout without duplicating text/icons.
pill_w, pill_h = 140, 38
pill_x = bottom_margin + 24
pill_y = section_header_h + 140
draw.rounded_rectangle(
    (pill_x, pill_y, pill_x + pill_w, pill_y + pill_h),
    radius=10,
    fill="#FFF5EB",  # subtle warm pill background (no text)
    outline="#F2D6C1",
    width=1
)

# another neutral rectangular preview placeholder (thumbnail background)
thumb_w, thumb_h = 260, 170
thumb_x = bottom_margin + 24
thumb_y = section_header_h + 28
draw.rectangle((thumb_x, thumb_y, thumb_x + thumb_w, thumb_y + thumb_h), fill=muted_fill, outline=card_border)

# final subtle bottom divider near very bottom edge
draw.line((0, canvas_h - 24, canvas_w, canvas_h - 24), fill=divider_color, width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_07_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-10/00_icon_Include_fees.png
try:
    _c0 = get_crop(0, 335, 108)
    canvas.paste(_c0, (544, 312), _c0)
except Exception:
    pass
layout["Include_fees"] = [544, 312, 879, 420]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_07_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-10/01_icon_Hide_resale.png
try:
    _c1 = get_crop(1, 315, 108)
    canvas.paste(_c1, (915, 312), _c1)
except Exception:
    pass
layout["Hide_resale"] = [915, 312, 1230, 420]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_07_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-10/02_icon_below_face_value.png
try:
    _c2 = get_crop(2, 1440, 588)
    canvas.paste(_c2, (0, 2355), _c2)
except Exception:
    pass
layout["below_face_value"] = [0, 2355, 1440, 2943]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_07_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-10/03_icon_Quantity.png
try:
    _c3 = get_crop(3, 268, 108)
    canvas.paste(_c3, (240, 312), _c3)
except Exception:
    pass
layout["Quantity"] = [240, 312, 508, 420]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_07_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-10/04_icon_Tit.png
try:
    _c4 = get_crop(4, 156, 108)
    canvas.paste(_c4, (48, 312), _c4)
except Exception:
    pass
layout["Tit"] = [48, 312, 204, 420]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_07_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-10/05_icon_Acces.png
try:
    _c5 = get_crop(5, 174, 108)
    canvas.paste(_c5, (1266, 312), _c5)
except Exception:
    pass
layout["Acces="] = [1266, 312, 1440, 420]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_07_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-10/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 51, 65)
    canvas.paste(_c6, (1152, 1), _c6)
except Exception:
    pass
layout["icon_6"] = [1152, 1, 1203, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_07_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-10/07_icon_Wy.png
try:
    _c7 = get_crop(7, 71, 65)
    canvas.paste(_c7, (108, 0), _c7)
except Exception:
    pass
layout["Wy"] = [108, 0, 179, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_07_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-10/08_icon_Wy.png
try:
    _c8 = get_crop(8, 55, 60)
    canvas.paste(_c8, (182, 2), _c8)
except Exception:
    pass
layout["Wy"] = [182, 2, 237, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_07_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-10/09_icon_Wy.png
try:
    _c9 = get_crop(9, 65, 60)
    canvas.paste(_c9, (242, 3), _c9)
except Exception:
    pass
layout["Wy"] = [242, 3, 307, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_07_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-10/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 102, 63)
    canvas.paste(_c10, (1213, 1), _c10)
except Exception:
    pass
layout["icon_10"] = [1213, 1, 1315, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_07_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-10/11_icon_Acces.png
try:
    _c11 = get_crop(11, 156, 156)
    canvas.paste(_c11, (1236, 120), _c11)
except Exception:
    pass
layout["Acces="] = [1236, 120, 1392, 276]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_07_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-10/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 59, 60)
    canvas.paste(_c12, (314, 3), _c12)
except Exception:
    pass
layout["icon_12"] = [314, 3, 373, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_07_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-10/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 52, 57)
    canvas.paste(_c13, (1320, 3), _c13)
except Exception:
    pass
layout["icon_13"] = [1320, 3, 1372, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_07_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-10/14_icon_Include_fees.png
try:
    _c14 = get_crop(14, 1344, 156)
    canvas.paste(_c14, (48, 120), _c14)
except Exception:
    pass
layout["Include_fees"] = [48, 120, 1392, 276]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_07_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-10/15_icon_The_Book_of_Mormon.png
try:
    _c15 = get_crop(15, 51, 62)
    canvas.paste(_c15, (382, 1), _c15)
except Exception:
    pass
layout["The_Book_of_Mormon"] = [382, 1, 433, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_07_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-10/16_icon_Sort_by_price.png
try:
    _c16 = get_crop(16, 455, 144)
    canvas.paste(_c16, (961, 1989), _c16)
except Exception:
    pass
layout["Sort_by_price"] = [961, 1989, 1416, 2133]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_07_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-10/17_icon_Amazing_deal.png
try:
    _c17 = get_crop(17, 1440, 588)
    canvas.paste(_c17, (0, 2355), _c17)
except Exception:
    pass
layout["Amazing_deal"] = [0, 2355, 1440, 2943]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_07_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-10/18_icon_6.51.png
try:
    _c18 = get_crop(18, 100, 65)
    canvas.paste(_c18, (5, 0), _c18)
except Exception:
    pass
layout["6.51"] = [5, 0, 105, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_07_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-10/19_icon_Oy_ANane_Atthebox_Orrce.png
try:
    _c19 = get_crop(19, 335, 108)
    canvas.paste(_c19, (544, 312), _c19)
except Exception:
    pass
layout["Oy_ANane_Atthebox_Orrce"] = [544, 312, 879, 420]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_07_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-10/20_icon_Limited_View.png
try:
    _c20 = get_crop(20, 1440, 588)
    canvas.paste(_c20, (0, 2355), _c20)
except Exception:
    pass
layout["Limited_View"] = [0, 2355, 1440, 2943]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_07_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-10/21_text_Box_office_resale.png
try:
    _c21 = get_crop(21, 489, 54)
    canvas.paste(_c21, (58, 2033), _c21)
except Exception:
    pass
layout["Box_office_&_resale"] = [58, 2033, 547, 2087]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_07_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-10/22_text_We_sell_box_office_and_resale_tickets._R.png
try:
    _c22 = get_crop(22, 1440, 588)
    canvas.paste(_c22, (0, 2355), _c22)
except Exception:
    pass
layout["We_sell_box_office_and_€_"] = [0, 2355, 1440, 2943]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_07_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-10/23_text_below_face_value.png
try:
    _c23 = get_crop(23, 350, 54)
    canvas.paste(_c23, (56, 2250), _c23)
except Exception:
    pass
layout["below_face_value"] = [56, 2250, 406, 2304]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_07_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-10/24_clickable_Back.png
try:
    _c24 = get_crop(24, 156, 156)
    canvas.paste(_c24, (48, 120), _c24)
except Exception:
    pass
layout["Back"] = [48, 120, 204, 276]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_07_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-10/25_clickable_The_Book_of_Mormon.png
try:
    _c25 = get_crop(25, 413, 156)
    canvas.paste(_c25, (204, 120), _c25)
except Exception:
    pass
layout["The_Book_of_Mormon"] = [204, 120, 617, 276]
