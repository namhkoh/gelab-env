# page_id: page_seatgeek_42f99a380cf348fea73bd58c0e4ec8db_10
# screenshot: 2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-13.png
# step_index: 10/14
# task: Open SeatGeek and search for the broadway show "lion king" on March 22. I need 3 tickets at average price less than 500 USD. Find the best seats and record the total price in google keep notes.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Background and structural elements for the SeatGeek-style UI (PIL drawing)
# Uses provided variables: canvas (PIL.Image), draw (ImageDraw), font_sm, font_md, font_lg, font_xl

# Canvas dimensions (for reference): 1440x2960

# Colors
bg = "#eef1f4"            # page background (light bluish gray)
status_bar = "#e6e8ea"    # top status bar
header_shadow = "#d7d9db" # shadow under header pill
header_fill = "#ffffff"   # header pill fill
card_border = "#c9cbd0"   # subtle card border
card_fill = "#f7f8f9"     # card inner fill
map_card_border = "#bfc2c6"
map_card_fill = "#f5f6f7"
list_card_fill = "#ffffff"
divider = "#e6e7e9"
separator = "#e9eaec"

w, h = canvas.size

# Fill overall background
draw.rectangle([(0, 0), (w, h)], fill=bg)

# Status bar area at top (~72px)
status_h = 72
draw.rectangle([(0, 0), (w, status_h)], fill=status_bar)

# Header / toolbar pill (rounded) centered under status bar
header_left = 40
header_right = w - 40
header_top = status_h + 12
header_bottom = header_top + 112
header_radius = 56

# Shadow for header pill (slight downward offset)
shadow_offset = 6
draw.rounded_rectangle(
    [(header_left, header_top + shadow_offset),
     (header_right, header_bottom + shadow_offset)],
    radius=header_radius,
    fill=header_shadow
)

# Header pill (white)
draw.rounded_rectangle(
    [(header_left, header_top),
     (header_right, header_bottom)],
    radius=header_radius,
    fill=header_fill
)

# Thin divider line under header region
divider_y = header_bottom + 18
draw.line([(20, divider_y), (w - 20, divider_y)], fill=divider, width=1)

# Filter area background (subtle band behind chips) - do not draw chip shapes themselves
filter_band_top = divider_y + 16
filter_band_bottom = filter_band_top + 160
draw.rectangle([(0, filter_band_top), (w, filter_band_bottom)], fill=bg)

# Main seating-map card area (large centered rounded card)
map_left = 240
map_right = w - 240
map_top = filter_band_bottom + 20
map_bottom = map_top + 1160  # tall area for map + mezzanine visualization
map_radius = 22

# Outer border (slightly darker)
draw.rounded_rectangle(
    [(map_left - 8, map_top - 8), (map_right + 8, map_bottom + 8)],
    radius=map_radius + 8,
    fill=map_card_border
)

# Inner card fill (light)
draw.rounded_rectangle(
    [(map_left, map_top), (map_right, map_bottom)],
    radius=map_radius,
    fill=map_card_fill
)

# Subtle inner inset to suggest a framed seating diagram (do NOT draw detailed seating or stage)
inset = 28
draw.rounded_rectangle(
    [(map_left + inset, map_top + inset), (map_right - inset, map_bottom - inset)],
    radius=12,
    fill=card_fill
)

# Light dashed-like guide lines (non-intrusive) to separate upper/lower portions of the map card
# (drawn as faint solid lines to avoid complex dash support)
upper_sep_y = map_top + 420
lower_sep_y = map_top + 820
draw.line([(map_left + 20, upper_sep_y), (map_right - 20, upper_sep_y)], fill=separator, width=1)
draw.line([(map_left + 20, lower_sep_y), (map_right - 20, lower_sep_y)], fill=separator, width=1)

# Mezzanine background block (sub-card inside the map area) - wide, shallow rounded rect
mezz_left = map_left + 60
mezz_right = map_right - 60
mezz_top = lower_sep_y + 60
mezz_bottom = mezz_top + 180
mezz_radius = 14
draw.rounded_rectangle(
    [(mezz_left - 4, mezz_top - 4), (mezz_right + 4, mezz_bottom + 4)],
    radius=mezz_radius + 4,
    fill=map_card_border
)
draw.rounded_rectangle(
    [(mezz_left, mezz_top), (mezz_right, mezz_bottom)],
    radius=mezz_radius,
    fill=card_fill
)

# Listings section card at bottom (rounded top corners)
list_card_top = map_bottom + 60
list_card_left = 20
list_card_right = w - 20
list_card_bottom = h  # extend to bottom
list_card_radius = 36

# Shadow behind list card
draw.rounded_rectangle(
    [(list_card_left + 4, list_card_top + 8), (list_card_right - 4, list_card_top + 8 + 220)],
    radius=list_card_radius,
    fill="#e6e7e9"
)

# White list card (rounded top)
draw.pieslice = getattr(draw, "pieslice", None)  # ensure no lint issues; not used further
draw.rounded_rectangle(
    [(list_card_left, list_card_top), (list_card_right, list_card_bottom)],
    radius=list_card_radius,
    fill=list_card_fill
)

# Divider under the list header area
list_header_divider_y = list_card_top + 120
draw.line([(list_card_left + 24, list_header_divider_y), (list_card_right - 24, list_header_divider_y)], fill=divider, width=1)

# Faint separators for a few listing rows (background only, not drawing any text or thumbnails)
row_height = 280
first_row_top = list_header_divider_y + 48
for i in range(3):
    top = first_row_top + i * (row_height + 20)
    # subtle card background for each listing preview
    cell_left = list_card_left + 24
    cell_right = list_card_right - 24
    cell_bottom = top + row_height
    draw.rounded_rectangle(
        [(cell_left, top), (cell_right, cell_bottom)],
        radius=18,
        fill="#fafbfc"
    )
    # separator line below each listing cell
    draw.line([(cell_left + 12, cell_bottom + 18), (cell_right - 12, cell_bottom + 18)], fill=separator, width=1)

# Final thin bottom safety margin line
draw.line([(0, h - 1), (w, h - 1)], fill=divider, width=1)

# End of structural drawing.
# Note: All icons, text, and interactive elements will be pasted on top of these backgrounds;
# this code intentionally does not draw any of those detected elements.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_10_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-13/00_icon_Include_fees.png
try:
    _c0 = get_crop(0, 335, 108)
    canvas.paste(_c0, (543, 312), _c0)
except Exception:
    pass
layout["Include_fees"] = [543, 312, 878, 420]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_10_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-13/01_icon_Best_seats.png
try:
    _c1 = get_crop(1, 303, 108)
    canvas.paste(_c1, (914, 312), _c1)
except Exception:
    pass
layout["Best_seats"] = [914, 312, 1217, 420]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_10_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-13/02_icon_3_tickets.png
try:
    _c2 = get_crop(2, 267, 108)
    canvas.paste(_c2, (240, 312), _c2)
except Exception:
    pass
layout["3_tickets"] = [240, 312, 507, 420]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_10_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-13/03_icon_STAGE.png
try:
    _c3 = get_crop(3, 528, 267)
    canvas.paste(_c3, (458, 638), _c3)
except Exception:
    pass
layout["STAGE"] = [458, 638, 986, 905]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_10_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-13/04_icon_Low_pr.png
try:
    _c4 = get_crop(4, 187, 108)
    canvas.paste(_c4, (1253, 312), _c4)
except Exception:
    pass
layout["Low_pr"] = [1253, 312, 1440, 420]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_10_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-13/05_icon_Tit.png
try:
    _c5 = get_crop(5, 156, 108)
    canvas.paste(_c5, (48, 312), _c5)
except Exception:
    pass
layout["Tit"] = [48, 312, 204, 420]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_10_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-13/06_icon_7.3.png
try:
    _c6 = get_crop(6, 1440, 455)
    canvas.paste(_c6, (0, 2355), _c6)
except Exception:
    pass
layout["7.3"] = [0, 2355, 1440, 2810]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_10_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-13/07_icon_New_York.png
try:
    _c7 = get_crop(7, 496, 156)
    canvas.paste(_c7, (204, 120), _c7)
except Exception:
    pass
layout["New_York"] = [204, 120, 700, 276]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_10_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-13/08_icon_GEK.png
try:
    _c8 = get_crop(8, 60, 60)
    canvas.paste(_c8, (244, 1), _c8)
except Exception:
    pass
layout["GEK"] = [244, 1, 304, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_10_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-13/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 105, 64)
    canvas.paste(_c9, (1212, 0), _c9)
except Exception:
    pass
layout["icon_9"] = [1212, 0, 1317, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_10_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-13/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 51, 66)
    canvas.paste(_c10, (1152, 1), _c10)
except Exception:
    pass
layout["icon_10"] = [1152, 1, 1203, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_10_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-13/11_icon_my.png
try:
    _c11 = get_crop(11, 64, 64)
    canvas.paste(_c11, (109, 0), _c11)
except Exception:
    pass
layout["my"] = [109, 0, 173, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_10_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-13/12_icon_my.png
try:
    _c12 = get_crop(12, 57, 62)
    canvas.paste(_c12, (180, 0), _c12)
except Exception:
    pass
layout["my"] = [180, 0, 237, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_10_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-13/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 54, 61)
    canvas.paste(_c13, (1319, 1), _c13)
except Exception:
    pass
layout["icon_13"] = [1319, 1, 1373, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_10_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-13/14_icon_Great_deal.png
try:
    _c14 = get_crop(14, 1440, 455)
    canvas.paste(_c14, (0, 2355), _c14)
except Exception:
    pass
layout["Great_deal"] = [0, 2355, 1440, 2810]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_10_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-13/15_icon_Low_pr.png
try:
    _c15 = get_crop(15, 156, 156)
    canvas.paste(_c15, (1236, 120), _c15)
except Exception:
    pass
layout["Low_pr"] = [1236, 120, 1392, 276]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_10_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-13/16_icon_7.41.png
try:
    _c16 = get_crop(16, 98, 65)
    canvas.paste(_c16, (7, 0), _c16)
except Exception:
    pass
layout["7.41"] = [7, 0, 105, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_10_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-13/17_icon_MEZZANINE.png
try:
    _c17 = get_crop(17, 455, 144)
    canvas.paste(_c17, (961, 1989), _c17)
except Exception:
    pass
layout["MEZZANINE"] = [961, 1989, 1416, 2133]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_10_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-13/18_icon_S434_each.png
try:
    _c18 = get_crop(18, 384, 106)
    canvas.paste(_c18, (52, 2854), _c18)
except Exception:
    pass
layout["S434_each"] = [52, 2854, 436, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_10_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-13/19_text_ORCHESTRA.png
try:
    _c19 = get_crop(19, 138, 29)
    canvas.paste(_c19, (650, 997), _c19)
except Exception:
    pass
layout["ORCHESTRA"] = [650, 997, 788, 1026]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_10_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-13/20_text_BOXL.png
try:
    _c20 = get_crop(20, 71, 27)
    canvas.paste(_c20, (326, 1420), _c20)
except Exception:
    pass
layout["BOXL"] = [326, 1420, 397, 1447]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_10_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-13/21_text_BOX_R.png
try:
    _c21 = get_crop(21, 73, 25)
    canvas.paste(_c21, (1043, 1422), _c21)
except Exception:
    pass
layout["BOX_R"] = [1043, 1422, 1116, 1447]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_10_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-13/22_text_Listings.png
try:
    _c22 = get_crop(22, 216, 68)
    canvas.paste(_c22, (101, 2032), _c22)
except Exception:
    pass
layout["Listings"] = [101, 2032, 317, 2100]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_10_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-13/23_text_Sort_by_price.png
try:
    _c23 = get_crop(23, 455, 144)
    canvas.paste(_c23, (961, 1989), _c23)
except Exception:
    pass
layout["Sort_by_price"] = [961, 1989, 1416, 2133]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_10_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-13/24_text_We_sell_resale_tickets._Resale_tickets_m.png
try:
    _c24 = get_crop(24, 1440, 455)
    canvas.paste(_c24, (0, 2355), _c24)
except Exception:
    pass
layout["€_We_sell_resale_tickets."] = [0, 2355, 1440, 2810]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_10_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-13/25_text_face_value.png
try:
    _c25 = get_crop(25, 218, 43)
    canvas.paste(_c25, (57, 2256), _c25)
except Exception:
    pass
layout["face_value:"] = [57, 2256, 275, 2299]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_10_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-13/26_text_S434_each.png
try:
    _c26 = get_crop(26, 276, 61)
    canvas.paste(_c26, (485, 2862), _c26)
except Exception:
    pass
layout["S434_each"] = [485, 2862, 761, 2923]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_10_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-13/27_clickable_Back.png
try:
    _c27 = get_crop(27, 156, 156)
    canvas.paste(_c27, (48, 120), _c27)
except Exception:
    pass
layout["Back"] = [48, 120, 204, 276]
