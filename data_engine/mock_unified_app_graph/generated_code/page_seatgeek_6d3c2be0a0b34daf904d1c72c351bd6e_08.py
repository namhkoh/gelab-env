# page_id: page_seatgeek_6d3c2be0a0b34daf904d1c72c351bd6e_08
# screenshot: 2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-11.png
# step_index: 8/9
# task: Open SeatGeek. Look up "Phoenix Suns" tickets for next upcoming event. Which section are tickets in?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Background and UI structure for the provided canvas.
# Uses provided variables: canvas (PIL Image), draw (ImageDraw), font_* (unused here).

# Colors
bg = (244, 246, 248)            # overall app background (very light gray-blue)
status_bar = (232, 234, 236)    # status bar top
header_shadow = (215, 219, 223) # shadow behind header pill
header_fill = (255, 255, 255)   # header / toolbar background
chip_area_shadow = (235, 237, 239)
arena_outer = (220, 223, 226)   # border of arena card
arena_inner = (247, 248, 249)   # inner arena background
list_card_fill = (255, 255, 255) # listings card background
divider = (220, 223, 226)

W, H = canvas.size

# 1) Fill overall background
draw.rectangle([(0, 0), (W, H)], fill=bg)

# 2) Status bar area (top ~64px)
status_h = 64
draw.rectangle([(0, 0), (W, status_h)], fill=status_bar)

# subtle bottom hairline under status bar
draw.line([(0, status_h-1), (W, status_h-1)], fill=divider, width=1)

# 3) Header/toolbar background (rounded pill area)
# Use the detected header bounds (48,120) size (1344x156)
hdr_left, hdr_top = 48, 120
hdr_w, hdr_h = 1344, 156
hdr_box = (hdr_left, hdr_top, hdr_left + hdr_w, hdr_top + hdr_h)
hdr_radius = 80

# Header shadow (slightly offset)
shadow_offset = 6
shadow_box = (hdr_box[0], hdr_box[1] + shadow_offset, hdr_box[2], hdr_box[3] + shadow_offset)
draw.rounded_rectangle(shadow_box, radius=hdr_radius, fill=header_shadow)

# Header main rounded rectangle
draw.rounded_rectangle(hdr_box, radius=hdr_radius, fill=header_fill, outline=divider, width=1)

# subtle inner divider line across header (to hint at info icon separator on the right)
sep_x = hdr_box[2] - 120
draw.line([(sep_x, hdr_box[1] + 16), (sep_x, hdr_box[3] - 16)], fill=divider, width=1)

# 4) Chip/filters row background hint
# Rather than drawing individual chips (they will be pasted), draw a faint horizontal band/shadow
chips_y_center = 312 + 108//2  # from detected chips row roughly
band_h = 120
band_box = (48, chips_y_center - band_h//2, W - 48, chips_y_center + band_h//2)
draw.rounded_rectangle(band_box, radius=60, fill=chip_area_shadow)

# Remove direct overlap with header by drawing band only below header
# (Ensure the band is slightly lighter so chips pasted will be visible)
draw.rectangle([(band_box[0], hdr_box[3]), (band_box[2], band_box[3])], fill=chip_area_shadow)

# 5) Arena seating map card background
# Large rounded card centered under chips. Keep it subtle and light so map details pasted on top will show.
arena_left = 80
arena_right = W - 80
arena_top = hdr_box[3] + 40
arena_bottom = 1550  # approximate bottom of arena area
arena_radius = 28

# Outer border to simulate thin frame
draw.rounded_rectangle(
    (arena_left - 6, arena_top - 6, arena_right + 6, arena_bottom + 6),
    radius=arena_radius + 6,
    fill=arena_outer
)

# Inner fill
draw.rounded_rectangle((arena_left, arena_top, arena_right, arena_bottom), radius=arena_radius, fill=arena_inner, outline=arena_outer, width=2)

# Add a subtle inner highlight top edge
highlight_y = arena_top + 6
draw.line([(arena_left + 12, highlight_y), (arena_right - 12, highlight_y)], fill=(250,250,251), width=1)

# 6) Separator line between arena and listings (to emphasize the transition)
sep_y = arena_bottom + 18
draw.line([(48, sep_y), (W - 48, sep_y)], fill=divider, width=1)

# 7) Listings card background (bottom rounded sheet)
# Use a full-width rounded white sheet starting a little below the arena to hold the listings.
list_top = sep_y + 28
list_bottom = H  # full bottom
list_radius = 36

# Draw drop shadow by drawing a slightly darker band above the card
shadow_box2 = (12, list_top + 2, W - 12, list_top + 18)
draw.rectangle(shadow_box2, fill=header_shadow)

# Main listings card
draw.rounded_rectangle((0, list_top, W, list_bottom), radius=list_radius, fill=list_card_fill, outline=divider, width=1)

# 8) Listings header separator / small divider under title area
# The detected "407 Listings" label sits inside this card near y ~2029; draw a subtle divider under the header region
header_div_y = list_top + 60
draw.line([(24, header_div_y), (W - 24, header_div_y)], fill=divider, width=1)

# 9) Item separators inside listings card
# Draw two separators to suggest separate list rows (details/icons will be pasted on top).
row1_y = header_div_y + 140
row2_y = row1_y + 260
draw.line([(24, row1_y), (W - 24, row1_y)], fill=divider, width=1)
draw.line([(24, row2_y), (W - 24, row2_y)], fill=divider, width=1)

# 10) Small subtle rounded thumbnail backgrounds on left side of list rows
# These are just for structure/background; actual images/icons will be pasted on top at exact positions.
thumb_w, thumb_h = 220, 150
thumb_radius = 16
thumb_x = 48
thumb1_y = header_div_y + 20
thumb2_y = row1_y + 20

# Light gray placeholders (under images that will be pasted)
draw.rounded_rectangle((thumb_x, thumb1_y, thumb_x + thumb_w, thumb1_y + thumb_h), radius=thumb_radius, fill=(245,246,247))
draw.rounded_rectangle((thumb_x, thumb2_y, thumb_x + thumb_w, thumb2_y + thumb_h), radius=thumb_radius, fill=(245,246,247))

# 11) Right-side subtle accent bar near top-right of listings card (where "Sort by deal" controls are)
# Do not draw the actual sort control; just hint with a tiny underline and light marker
sort_hint_x1 = W - 280
sort_hint_y = header_div_y - 18
draw.line([(sort_hint_x1, sort_hint_y), (W - 48, sort_hint_y)], fill=divider, width=1)
draw.ellipse([(W - 72, sort_hint_y - 10), (W - 52, sort_hint_y + 10)], fill=header_fill, outline=divider)

# 12) Final soft vignette at card bottom (tiny)
vignette_top = list_bottom - 40
draw.rectangle([(0, vignette_top), (W, list_bottom)], fill=(255,255,255,0))

# Finished structural drawing. The actual icons/text will be pasted on top at their detected positions.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_08_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-11/00_icon_Include_fees.png
try:
    _c0 = get_crop(0, 335, 108)
    canvas.paste(_c0, (544, 312), _c0)
except Exception:
    pass
layout["Include_fees"] = [544, 312, 879, 420]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_08_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-11/01_icon_Courtside.png
try:
    _c1 = get_crop(1, 286, 108)
    canvas.paste(_c1, (915, 312), _c1)
except Exception:
    pass
layout["Courtside"] = [915, 312, 1201, 420]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_08_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-11/02_icon_Nonutacr.png
try:
    _c2 = get_crop(2, 1440, 371)
    canvas.paste(_c2, (0, 2589), _c2)
except Exception:
    pass
layout["Nonutacr"] = [0, 2589, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_08_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-11/03_icon_Quantity.png
try:
    _c3 = get_crop(3, 268, 108)
    canvas.paste(_c3, (240, 312), _c3)
except Exception:
    pass
layout["Quantity"] = [240, 312, 508, 420]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_08_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-11/04_icon_Tit.png
try:
    _c4 = get_crop(4, 156, 108)
    canvas.paste(_c4, (48, 312), _c4)
except Exception:
    pass
layout["Tit"] = [48, 312, 204, 420]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_08_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-11/05_icon_Center.png
try:
    _c5 = get_crop(5, 203, 108)
    canvas.paste(_c5, (1237, 312), _c5)
except Exception:
    pass
layout["Center"] = [1237, 312, 1440, 420]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_08_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-11/06_icon_407_Listings.png
try:
    _c6 = get_crop(6, 1440, 455)
    canvas.paste(_c6, (0, 2134), _c6)
except Exception:
    pass
layout["407_Listings"] = [0, 2134, 1440, 2589]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_08_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-11/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 58, 61)
    canvas.paste(_c7, (312, 3), _c7)
except Exception:
    pass
layout["icon_7"] = [312, 3, 370, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_08_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-11/08_icon_7_07_my.png
try:
    _c8 = get_crop(8, 66, 62)
    canvas.paste(_c8, (111, 1), _c8)
except Exception:
    pass
layout["7:07_my"] = [111, 1, 177, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_08_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-11/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 62, 60)
    canvas.paste(_c9, (242, 3), _c9)
except Exception:
    pass
layout["icon_9"] = [242, 3, 304, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_08_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-11/10_icon_W_Conf_Ist_Rnd_Suns_at_Timberwolves_Gm_2.png
try:
    _c10 = get_crop(10, 1344, 156)
    canvas.paste(_c10, (48, 120), _c10)
except Exception:
    pass
layout["W_Conf_Ist_Rnd:_Suns_at_T"] = [48, 120, 1392, 276]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_08_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-11/11_icon_7_07_my.png
try:
    _c11 = get_crop(11, 53, 60)
    canvas.paste(_c11, (182, 2), _c11)
except Exception:
    pass
layout["7:07_my"] = [182, 2, 235, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_08_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-11/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 50, 65)
    canvas.paste(_c12, (1153, 1), _c12)
except Exception:
    pass
layout["icon_12"] = [1153, 1, 1203, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_08_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-11/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 52, 57)
    canvas.paste(_c13, (1320, 3), _c13)
except Exception:
    pass
layout["icon_13"] = [1320, 3, 1372, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_08_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-11/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 104, 62)
    canvas.paste(_c14, (1212, 1), _c14)
except Exception:
    pass
layout["icon_14"] = [1212, 1, 1316, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_08_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-11/15_icon_Amazing_deal.png
try:
    _c15 = get_crop(15, 1440, 455)
    canvas.paste(_c15, (0, 2134), _c15)
except Exception:
    pass
layout["Amazing_deal"] = [0, 2134, 1440, 2589]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_08_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-11/16_icon_icon_16.png
try:
    _c16 = get_crop(16, 49, 63)
    canvas.paste(_c16, (383, 2), _c16)
except Exception:
    pass
layout["icon_16"] = [383, 2, 432, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_08_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-11/17_icon_Center.png
try:
    _c17 = get_crop(17, 156, 156)
    canvas.paste(_c17, (1236, 120), _c17)
except Exception:
    pass
layout["Center"] = [1236, 120, 1392, 276]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_08_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-11/18_icon_Sort_by_deal.png
try:
    _c18 = get_crop(18, 440, 144)
    canvas.paste(_c18, (976, 1989), _c18)
except Exception:
    pass
layout["Sort_by_deal"] = [976, 1989, 1416, 2133]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_08_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-11/19_text_-228.png
try:
    _c19 = get_crop(19, 48, 27)
    canvas.paste(_c19, (467, 858), _c19)
except Exception:
    pass
layout["-228"] = [467, 858, 515, 885]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_08_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-11/20_text_-230.png
try:
    _c20 = get_crop(20, 48, 27)
    canvas.paste(_c20, (615, 858), _c20)
except Exception:
    pass
layout["-230"] = [615, 858, 663, 885]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_08_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-11/21_text_232.png
try:
    _c21 = get_crop(21, 48, 30)
    canvas.paste(_c21, (772, 855), _c21)
except Exception:
    pass
layout["232"] = [772, 855, 820, 885]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_08_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-11/22_text_234.png
try:
    _c22 = get_crop(22, 48, 27)
    canvas.paste(_c22, (920, 858), _c22)
except Exception:
    pass
layout["234"] = [920, 858, 968, 885]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_08_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-11/23_text_226.png
try:
    _c23 = get_crop(23, 48, 29)
    canvas.paste(_c23, (361, 886), _c23)
except Exception:
    pass
layout["226"] = [361, 886, 409, 915]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_08_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-11/24_text_236.png
try:
    _c24 = get_crop(24, 46, 29)
    canvas.paste(_c24, (1031, 886), _c24)
except Exception:
    pass
layout["236"] = [1031, 886, 1077, 915]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_08_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-11/25_text_B32.png
try:
    _c25 = get_crop(25, 48, 25)
    canvas.paste(_c25, (476, 902), _c25)
except Exception:
    pass
layout["B32"] = [476, 902, 524, 927]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_08_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-11/26_text_5401.png
try:
    _c26 = get_crop(26, 51, 28)
    canvas.paste(_c26, (647, 899), _c26)
except Exception:
    pass
layout["5401"] = [647, 899, 698, 927]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_08_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-11/27_text_541.png
try:
    _c27 = get_crop(27, 45, 28)
    canvas.paste(_c27, (724, 899), _c27)
except Exception:
    pass
layout["541"] = [724, 899, 769, 927]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_08_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-11/28_text_B5O.png
try:
    _c28 = get_crop(28, 51, 28)
    canvas.paste(_c28, (781, 899), _c28)
except Exception:
    pass
layout["B5O_"] = [781, 899, 832, 927]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_08_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-11/29_text_B5Z.png
try:
    _c29 = get_crop(29, 45, 25)
    canvas.paste(_c29, (916, 902), _c29)
except Exception:
    pass
layout["B5Z"] = [916, 902, 961, 927]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_08_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-11/30_text_129.png
try:
    _c30 = get_crop(30, 46, 27)
    canvas.paste(_c30, (543, 939), _c30)
except Exception:
    pass
layout["129"] = [543, 939, 589, 966]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_08_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-11/31_text_133.png
try:
    _c31 = get_crop(31, 46, 29)
    canvas.paste(_c31, (846, 939), _c31)
except Exception:
    pass
layout["133"] = [846, 939, 892, 968]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_08_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-11/32_text_523.png
try:
    _c32 = get_crop(32, 45, 27)
    canvas.paste(_c32, (347, 962), _c32)
except Exception:
    pass
layout["523"] = [347, 962, 392, 989]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_08_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-11/33_text_562.png
try:
    _c33 = get_crop(33, 48, 31)
    canvas.paste(_c33, (1044, 961), _c33)
except Exception:
    pass
layout["562"] = [1044, 961, 1092, 992]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_08_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-11/34_text_224.png
try:
    _c34 = get_crop(34, 48, 28)
    canvas.paste(_c34, (245, 1003), _c34)
except Exception:
    pass
layout["224"] = [245, 1003, 293, 1031]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_08_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-11/35_text_238.png
try:
    _c35 = get_crop(35, 45, 30)
    canvas.paste(_c35, (1147, 1001), _c35)
except Exception:
    pass
layout["238"] = [1147, 1001, 1192, 1031]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_08_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-11/36_text_124.png
try:
    _c36 = get_crop(36, 48, 27)
    canvas.paste(_c36, (312, 1031), _c36)
except Exception:
    pass
layout["124"] = [312, 1031, 360, 1058]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_08_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-11/37_text_138.png
try:
    _c37 = get_crop(37, 46, 27)
    canvas.paste(_c37, (1082, 1031), _c37)
except Exception:
    pass
layout["138"] = [1082, 1031, 1128, 1058]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_08_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-11/38_text_124.png
try:
    _c38 = get_crop(38, 46, 27)
    canvas.paste(_c38, (372, 1054), _c38)
except Exception:
    pass
layout["124"] = [372, 1054, 418, 1081]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_08_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-11/39_text_138.png
try:
    _c39 = get_crop(39, 46, 27)
    canvas.paste(_c39, (1029, 1050), _c39)
except Exception:
    pass
layout["138"] = [1029, 1050, 1075, 1077]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_08_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-11/40_text_S71.png
try:
    _c40 = get_crop(40, 43, 30)
    canvas.paste(_c40, (1133, 1068), _c40)
except Exception:
    pass
layout["S71"] = [1133, 1068, 1176, 1098]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_08_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-11/41_text_222.png
try:
    _c41 = get_crop(41, 48, 27)
    canvas.paste(_c41, (139, 1117), _c41)
except Exception:
    pass
layout["222"] = [139, 1117, 187, 1144]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_08_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-11/42_text_222.png
try:
    _c42 = get_crop(42, 46, 27)
    canvas.paste(_c42, (215, 1117), _c42)
except Exception:
    pass
layout["222"] = [215, 1117, 261, 1144]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_08_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-11/43_text_VISITORS.png
try:
    _c43 = get_crop(43, 80, 21)
    canvas.paste(_c43, (570, 1115), _c43)
except Exception:
    pass
layout["VISITORS"] = [570, 1115, 650, 1136]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_08_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-11/44_text_SCORERS.png
try:
    _c44 = get_crop(44, 83, 21)
    canvas.paste(_c44, (678, 1115), _c44)
except Exception:
    pass
layout["SCORERS"] = [678, 1115, 761, 1136]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_08_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-11/45_text_240.png
try:
    _c45 = get_crop(45, 48, 27)
    canvas.paste(_c45, (1177, 1117), _c45)
except Exception:
    pass
layout["240"] = [1177, 1117, 1225, 1144]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_08_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-11/46_text_240.png
try:
    _c46 = get_crop(46, 48, 30)
    canvas.paste(_c46, (1251, 1114), _c46)
except Exception:
    pass
layout["240"] = [1251, 1114, 1299, 1144]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_08_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-11/47_text_CS.png
try:
    _c47 = get_crop(47, 34, 27)
    canvas.paste(_c47, (539, 1165), _c47)
except Exception:
    pass
layout["CS"] = [539, 1165, 573, 1192]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_08_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-11/48_text_CS.png
try:
    _c48 = get_crop(48, 35, 27)
    canvas.paste(_c48, (855, 1161), _c48)
except Exception:
    pass
layout["CS"] = [855, 1161, 890, 1188]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_08_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-11/49_text_221.png
try:
    _c49 = get_crop(49, 41, 27)
    canvas.paste(_c49, (199, 1193), _c49)
except Exception:
    pass
layout["221"] = [199, 1193, 240, 1220]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_08_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-11/50_text_TI8.png
try:
    _c50 = get_crop(50, 39, 25)
    canvas.paste(_c50, (263, 1179), _c50)
except Exception:
    pass
layout["TI8"] = [263, 1179, 302, 1204]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_08_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-11/51_text_S74.png
try:
    _c51 = get_crop(51, 48, 29)
    canvas.paste(_c51, (1133, 1177), _c51)
except Exception:
    pass
layout["S74"] = [1133, 1177, 1181, 1206]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_08_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-11/52_text_201.png
try:
    _c52 = get_crop(52, 46, 27)
    canvas.paste(_c52, (1193, 1193), _c52)
except Exception:
    pass
layout["201"] = [1193, 1193, 1239, 1220]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_08_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-11/53_text_220.png
try:
    _c53 = get_crop(53, 48, 28)
    canvas.paste(_c53, (215, 1269), _c53)
except Exception:
    pass
layout["220"] = [215, 1269, 263, 1297]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_08_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-11/54_text_CS5.png
try:
    _c54 = get_crop(54, 44, 21)
    canvas.paste(_c54, (648, 1284), _c54)
except Exception:
    pass
layout["CS5"] = [648, 1284, 692, 1305]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_08_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-11/55_text_202.png
try:
    _c55 = get_crop(55, 46, 30)
    canvas.paste(_c55, (1177, 1267), _c55)
except Exception:
    pass
layout["202"] = [1177, 1267, 1223, 1297]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_08_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-11/56_text_113.png
try:
    _c56 = get_crop(56, 46, 27)
    canvas.paste(_c56, (550, 1325), _c56)
except Exception:
    pass
layout["113"] = [550, 1325, 596, 1352]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_08_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-11/57_text_219.png
try:
    _c57 = get_crop(57, 48, 27)
    canvas.paste(_c57, (111, 1346), _c57)
except Exception:
    pass
layout["219"] = [111, 1346, 159, 1373]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_08_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-11/58_text_118.png
try:
    _c58 = get_crop(58, 47, 27)
    canvas.paste(_c58, (294, 1339), _c58)
except Exception:
    pass
layout["118"] = [294, 1339, 341, 1366]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_08_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-11/59_text_104.png
try:
    _c59 = get_crop(59, 46, 27)
    canvas.paste(_c59, (1098, 1341), _c59)
except Exception:
    pass
layout["104"] = [1098, 1341, 1144, 1368]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_08_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-11/60_text_203.png
try:
    _c60 = get_crop(60, 46, 28)
    canvas.paste(_c60, (1281, 1343), _c60)
except Exception:
    pass
layout["203"] = [1281, 1343, 1327, 1371]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_08_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-11/61_text_218.png
try:
    _c61 = get_crop(61, 48, 30)
    canvas.paste(_c61, (245, 1387), _c61)
except Exception:
    pass
layout["218"] = [245, 1387, 293, 1417]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_08_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-11/62_text_204.png
try:
    _c62 = get_crop(62, 48, 27)
    canvas.paste(_c62, (1147, 1390), _c62)
except Exception:
    pass
layout["204"] = [1147, 1390, 1195, 1417]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_08_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-11/63_text_522.png
try:
    _c63 = get_crop(63, 48, 29)
    canvas.paste(_c63, (347, 1429), _c63)
except Exception:
    pass
layout["522"] = [347, 1429, 395, 1458]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_08_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-11/64_text_217.png
try:
    _c64 = get_crop(64, 45, 28)
    canvas.paste(_c64, (287, 1454), _c64)
except Exception:
    pass
layout["217"] = [287, 1454, 332, 1482]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_08_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-11/65_text_CHAIRMANS.png
try:
    _c65 = get_crop(65, 108, 25)
    canvas.paste(_c65, (666, 1482), _c65)
except Exception:
    pass
layout["CHAIRMANS"] = [666, 1482, 774, 1507]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_08_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-11/66_text_216.png
try:
    _c66 = get_crop(66, 48, 27)
    canvas.paste(_c66, (361, 1505), _c66)
except Exception:
    pass
layout["216"] = [361, 1505, 409, 1532]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_08_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-11/67_text_C1Z.png
try:
    _c67 = get_crop(67, 48, 28)
    canvas.paste(_c67, (448, 1491), _c67)
except Exception:
    pass
layout["C1Z"] = [448, 1491, 496, 1519]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_08_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-11/68_text_C15.png
try:
    _c68 = get_crop(68, 48, 28)
    canvas.paste(_c68, (525, 1491), _c68)
except Exception:
    pass
layout["C15"] = [525, 1491, 573, 1519]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_08_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-11/69_text_C13.png
try:
    _c69 = get_crop(69, 48, 28)
    canvas.paste(_c69, (601, 1491), _c69)
except Exception:
    pass
layout["C13"] = [601, 1491, 649, 1519]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_08_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-11/70_text_206.png
try:
    _c70 = get_crop(70, 48, 27)
    canvas.paste(_c70, (1031, 1505), _c70)
except Exception:
    pass
layout["206"] = [1031, 1505, 1079, 1532]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_08_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-11/71_text_214.png
try:
    _c71 = get_crop(71, 48, 27)
    canvas.paste(_c71, (472, 1535), _c71)
except Exception:
    pass
layout["214"] = [472, 1535, 520, 1562]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_08_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-11/72_text_208.png
try:
    _c72 = get_crop(72, 48, 27)
    canvas.paste(_c72, (927, 1535), _c72)
except Exception:
    pass
layout["208"] = [927, 1535, 975, 1562]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_08_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-11/73_text_213.png
try:
    _c73 = get_crop(73, 48, 29)
    canvas.paste(_c73, (541, 1552), _c73)
except Exception:
    pass
layout["213"] = [541, 1552, 589, 1581]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_08_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-11/74_text_407_Listings.png
try:
    _c74 = get_crop(74, 331, 72)
    canvas.paste(_c74, (56, 2029), _c74)
except Exception:
    pass
layout["407_Listings"] = [56, 2029, 387, 2101]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_08_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-11/75_text_S239_each.png
try:
    _c75 = get_crop(75, 1440, 371)
    canvas.paste(_c75, (0, 2589), _c75)
except Exception:
    pass
layout["S239_each"] = [0, 2589, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_08_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-11/76_text_Price_includes_fees.png
try:
    _c76 = get_crop(76, 1440, 371)
    canvas.paste(_c76, (0, 2589), _c76)
except Exception:
    pass
layout["Price_includes_fees"] = [0, 2589, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_08_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-11/77_text_9.8.png
try:
    _c77 = get_crop(77, 50, 31)
    canvas.paste(_c77, (502, 2810), _c77)
except Exception:
    pass
layout["9.8"] = [502, 2810, 552, 2841]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_08_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-11/78_text_Amazing_deal.png
try:
    _c78 = get_crop(78, 1440, 371)
    canvas.paste(_c78, (0, 2589), _c78)
except Exception:
    pass
layout["Amazing_deal"] = [0, 2589, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_08_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-11/79_text_1-4_tickets.png
try:
    _c79 = get_crop(79, 221, 50)
    canvas.paste(_c79, (488, 2870), _c79)
except Exception:
    pass
layout["1-4_tickets"] = [488, 2870, 709, 2920]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_08_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-11/80_text_40.png
try:
    _c80 = get_crop(80, 69, 13)
    canvas.paste(_c80, (655, 2945), _c80)
except Exception:
    pass
layout["40"] = [655, 2945, 724, 2958]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_08_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-11/81_clickable_Back.png
try:
    _c81 = get_crop(81, 156, 156)
    canvas.paste(_c81, (48, 120), _c81)
except Exception:
    pass
layout["Back"] = [48, 120, 204, 276]
