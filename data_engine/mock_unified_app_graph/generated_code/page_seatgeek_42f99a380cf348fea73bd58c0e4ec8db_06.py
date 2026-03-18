# page_id: page_seatgeek_42f99a380cf348fea73bd58c0e4ec8db_06
# screenshot: 2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-9.png
# step_index: 6/14
# task: Open SeatGeek and search for the broadway show "lion king" on March 22. I need 3 tickets at average price less than 500 USD. Find the best seats and record the total price in google keep notes.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Fill overall background with the app's light bluish-gray canvas color
draw.rectangle([(0, 0), (1440, 2960)], fill=(241, 244, 246))

# Status bar area (top strip) - slightly darker than page background to emulate OS status bar
status_h = 60
draw.rectangle([(0, 0), (1440, status_h)], fill=(218, 222, 225))

# Thin divider under status bar for subtle separation
draw.line([(0, status_h), (1440, status_h)], fill=(210, 213, 216), width=1)

# Header / Title pill background (rounded) centered under status bar
hdr_x1, hdr_y1 = 48, 120
hdr_x2, hdr_y2 = hdr_x1 + 1344, hdr_y1 + 156
# subtle shadow behind header pill
shadow_offset = 6
draw.rounded_rectangle(
    [(hdr_x1, hdr_y1 + shadow_offset), (hdr_x2, hdr_y2 + shadow_offset)],
    radius=78,
    fill=(226, 228, 231),
)
# header pill (white)
draw.rounded_rectangle(
    [(hdr_x1, hdr_y1), (hdr_x2, hdr_y2)],
    radius=78,
    fill=(255, 255, 255),
    outline=(235, 238, 241),
    width=1,
)

# Big seat-map / venue container background (light card with thin border)
map_x1, map_y1 = 180, 420
map_x2, map_y2 = 1260, 1520
draw.rounded_rectangle(
    [(map_x1 - 6, map_y1 - 6), (map_x2 + 6, map_y2 + 6)],
    radius=28,
    fill=(226, 229, 232),
)
draw.rounded_rectangle(
    [(map_x1, map_y1), (map_x2, map_y2)],
    radius=22,
    fill=(247, 248, 249),
    outline=(200, 203, 207),
    width=4,
)

# Optional subtle dashed guideline boxes at the sides of the map to hint layout (very subtle)
# (drawn faintly so as not to conflict with pasted exact map content)
left_guide_box = [(120, map_y1 + 120), (180, map_y2 - 120)]
right_guide_box = [(1260, map_y1 + 120), (1320, map_y2 - 120)]
draw.rectangle(left_guide_box, outline=(230, 232, 235), width=1)
draw.rectangle(right_guide_box, outline=(230, 232, 235), width=1)

# Listings container (white sheet) occupying bottom portion with rounded top corners
list_x1, list_y1 = 0, 1880
list_x2, list_y2 = 1440, 2960
# subtle shadow above the listings container
draw.rectangle([(0, list_y1 - 6), (1440, list_y1)], fill=(230, 232, 234))
draw.rounded_rectangle(
    [(list_x1, list_y1), (list_x2, list_y2)],
    radius=32,
    fill=(255, 255, 255),
    outline=(235, 238, 241),
    width=1,
)

# A light top divider inside the listings card to mark the header area
draw.line([(28, list_y1 + 140), (1412, list_y1 + 140)], fill=(226, 229, 231), width=1)

# Separator lines between listing items (positions chosen to align visually without drawing any icon/text)
# First item separator
draw.line([(20, list_y1 + 320), (1420, list_y1 + 320)], fill=(236, 238, 240), width=1)
# Second item separator
draw.line([(20, list_y1 + 760), (1420, list_y1 + 760)], fill=(236, 238, 240), width=1)

# Subtle section divider above the "We sell resale tickets..." explanatory text area
draw.line([(28, list_y1 + 80), (1412, list_y1 + 80)], fill=(235, 238, 240), width=1)

# Soft rounded background behind each listing row area (very light) to give card feel
row_pad_x1 = 28
row_pad_x2 = 1412
row_heights = [list_y1 + 160, list_y1 + 600, list_y1 + 1040]
for idx, row_top in enumerate(row_heights):
    row_bottom = row_top + 360
    # Keep these very faint so pasted listing content remains dominant
    draw.rounded_rectangle(
        [(row_pad_x1, row_top), (row_pad_x2, row_bottom)],
        radius=14,
        fill=(250, 251, 252),
        outline=None,
    )

# Small visual affordance: a faint circular drag-handle at top center of listings card
handle_cx = 720
handle_cy = list_y1 + 18
draw.ellipse([(handle_cx - 28, handle_cy - 6), (handle_cx + 28, handle_cy + 6)], fill=(242, 244, 246), outline=(230, 232, 235))

# Final subtle vignette/shadow around the main card areas to lift them off the page lightly
# (drawn as semi-solid rectangles to emulate soft shadows)
draw.rectangle([(20, hdr_y2 + 8), (1420, hdr_y2 + 12)], fill=(236, 238, 240))
draw.rectangle([(map_x1 - 8, map_y2 + 8), (map_x2 + 8, map_y2 + 12)], fill=(232, 234, 236))
draw.rectangle([(0, list_y1 - 4), (1440, list_y1)], fill=(228, 230, 233))

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_06_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-9/00_icon_Include_fees.png
try:
    _c0 = get_crop(0, 335, 108)
    canvas.paste(_c0, (544, 312), _c0)
except Exception:
    pass
layout["Include_fees"] = [544, 312, 879, 420]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_06_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-9/01_icon_Best_seats.png
try:
    _c1 = get_crop(1, 303, 108)
    canvas.paste(_c1, (915, 312), _c1)
except Exception:
    pass
layout["Best_seats"] = [915, 312, 1218, 420]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_06_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-9/02_icon_Quantity.png
try:
    _c2 = get_crop(2, 268, 108)
    canvas.paste(_c2, (240, 312), _c2)
except Exception:
    pass
layout["Quantity"] = [240, 312, 508, 420]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_06_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-9/03_icon_STAGE.png
try:
    _c3 = get_crop(3, 530, 268)
    canvas.paste(_c3, (457, 638), _c3)
except Exception:
    pass
layout["STAGE"] = [457, 638, 987, 906]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_06_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-9/04_icon_Tit.png
try:
    _c4 = get_crop(4, 156, 108)
    canvas.paste(_c4, (48, 312), _c4)
except Exception:
    pass
layout["Tit"] = [48, 312, 204, 420]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_06_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-9/05_icon_10.0.png
try:
    _c5 = get_crop(5, 1440, 455)
    canvas.paste(_c5, (0, 2355), _c5)
except Exception:
    pass
layout["10.0"] = [0, 2355, 1440, 2810]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_06_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-9/06_icon_Low_pri.png
try:
    _c6 = get_crop(6, 186, 108)
    canvas.paste(_c6, (1254, 312), _c6)
except Exception:
    pass
layout["Low_pri"] = [1254, 312, 1440, 420]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_06_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-9/07_icon_New_York.png
try:
    _c7 = get_crop(7, 1344, 156)
    canvas.paste(_c7, (48, 120), _c7)
except Exception:
    pass
layout["New_York"] = [48, 120, 1392, 276]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_06_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-9/08_icon_GEK.png
try:
    _c8 = get_crop(8, 61, 60)
    canvas.paste(_c8, (244, 1), _c8)
except Exception:
    pass
layout["GEK"] = [244, 1, 305, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_06_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-9/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 51, 65)
    canvas.paste(_c9, (1152, 1), _c9)
except Exception:
    pass
layout["icon_9"] = [1152, 1, 1203, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_06_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-9/10_icon_0.png
try:
    _c10 = get_crop(10, 102, 63)
    canvas.paste(_c10, (1213, 1), _c10)
except Exception:
    pass
layout["0#"] = [1213, 1, 1315, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_06_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-9/11_icon_my.png
try:
    _c11 = get_crop(11, 63, 63)
    canvas.paste(_c11, (110, 0), _c11)
except Exception:
    pass
layout["my"] = [110, 0, 173, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_06_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-9/12_icon_my.png
try:
    _c12 = get_crop(12, 55, 61)
    canvas.paste(_c12, (182, 0), _c12)
except Exception:
    pass
layout["my"] = [182, 0, 237, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_06_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-9/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 53, 57)
    canvas.paste(_c13, (1320, 3), _c13)
except Exception:
    pass
layout["icon_13"] = [1320, 3, 1373, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_06_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-9/14_icon_0.png
try:
    _c14 = get_crop(14, 156, 156)
    canvas.paste(_c14, (1236, 120), _c14)
except Exception:
    pass
layout["0#"] = [1236, 120, 1392, 276]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_06_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-9/15_icon_Amazing_deal.png
try:
    _c15 = get_crop(15, 1440, 455)
    canvas.paste(_c15, (0, 2355), _c15)
except Exception:
    pass
layout["Amazing_deal"] = [0, 2355, 1440, 2810]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_06_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-9/16_icon_7.41.png
try:
    _c16 = get_crop(16, 98, 64)
    canvas.paste(_c16, (6, 0), _c16)
except Exception:
    pass
layout["7.41"] = [6, 0, 104, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_06_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-9/17_icon_Sort_by_price.png
try:
    _c17 = get_crop(17, 455, 144)
    canvas.paste(_c17, (961, 1989), _c17)
except Exception:
    pass
layout["Sort_by_price"] = [961, 1989, 1416, 2133]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_06_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-9/18_icon_BOXL.png
try:
    _c18 = get_crop(18, 455, 144)
    canvas.paste(_c18, (961, 1989), _c18)
except Exception:
    pass
layout["BOXL"] = [961, 1989, 1416, 2133]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_06_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-9/19_text_ORCHESTRA.png
try:
    _c19 = get_crop(19, 138, 29)
    canvas.paste(_c19, (650, 997), _c19)
except Exception:
    pass
layout["ORCHESTRA"] = [650, 997, 788, 1026]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_06_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-9/20_text_BOXL.png
try:
    _c20 = get_crop(20, 71, 27)
    canvas.paste(_c20, (326, 1420), _c20)
except Exception:
    pass
layout["BOXL"] = [326, 1420, 397, 1447]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_06_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-9/21_text_BOX_R.png
try:
    _c21 = get_crop(21, 73, 25)
    canvas.paste(_c21, (1043, 1422), _c21)
except Exception:
    pass
layout["BOX_R"] = [1043, 1422, 1116, 1447]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_06_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-9/22_text_106_Listings.png
try:
    _c22 = get_crop(22, 326, 72)
    canvas.paste(_c22, (54, 2029), _c22)
except Exception:
    pass
layout["106_Listings"] = [54, 2029, 380, 2101]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_06_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-9/23_text_We_sell_resale_tickets._Resale_tickets_m.png
try:
    _c23 = get_crop(23, 1440, 455)
    canvas.paste(_c23, (0, 2355), _c23)
except Exception:
    pass
layout["€_We_sell_resale_tickets."] = [0, 2355, 1440, 2810]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_06_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-9/24_text_face_value.png
try:
    _c24 = get_crop(24, 218, 43)
    canvas.paste(_c24, (57, 2256), _c24)
except Exception:
    pass
layout["face_value:"] = [57, 2256, 275, 2299]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_06_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-9/25_text_S212_each.png
try:
    _c25 = get_crop(25, 257, 63)
    canvas.paste(_c25, (485, 2862), _c25)
except Exception:
    pass
layout["S212_each"] = [485, 2862, 742, 2925]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_06_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-9/26_clickable_Back.png
try:
    _c26 = get_crop(26, 156, 156)
    canvas.paste(_c26, (48, 120), _c26)
except Exception:
    pass
layout["Back"] = [48, 120, 204, 276]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_06_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-9/27_clickable_The_Lion_King_-_New_York.png
try:
    _c27 = get_crop(27, 496, 156)
    canvas.paste(_c27, (204, 120), _c27)
except Exception:
    pass
layout["The_Lion_King_-_New_York"] = [204, 120, 700, 276]
