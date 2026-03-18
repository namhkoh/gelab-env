# page_id: page_seatgeek_6d3c2be0a0b34daf904d1c72c351bd6e_06
# screenshot: 2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-9.png
# step_index: 6/9
# task: Open SeatGeek. Look up "Phoenix Suns" tickets for next upcoming event. Which section are tickets in?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Background and structural UI drawing for SeatGeek-like page
# Uses provided variables: canvas (PIL Image), draw (ImageDraw), font_sm, font_md, font_lg, font_xl

W, H = canvas.size

# Colors
bg_color = (241, 243, 245)         # overall page background (very light gray)
status_bar_color = (226, 228, 230) # status bar slightly darker
header_shadow = (210, 213, 216)    # subtle shadow for header pill
header_bg = (255, 255, 255)        # header pill white
header_border = (216, 219, 222)
chip_section_bg = bg_color
map_outer_bg = (246, 247, 248)     # slightly different panel for map container
map_border = (200, 204, 207)
map_inner_bg = (255, 255, 255)
list_sheet_bg = (255, 255, 255)
divider = (225, 228, 230)
item_card_border = (235, 237, 239)
muted_gray = (245, 246, 247)

# Fill overall background
draw.rectangle([0, 0, W, H], fill=bg_color)

# Status bar (top area) ~ 0..70
status_h = 70
draw.rectangle([0, 0, W, status_h], fill=status_bar_color)

# subtle bottom hairline for status bar
draw.line([(0, status_h - 1), (W, status_h - 1)], fill=divider, width=1)

# Header / title pill (rounded) - use detected header position as guideline
hdr_left, hdr_top = 48, 120
hdr_w, hdr_h = 1344, 156
hdr_right = hdr_left + hdr_w
hdr_bottom = hdr_top + hdr_h
hdr_radius = 40

# Header shadow (slightly offset darker rounded box to simulate drop shadow)
shadow_offset = 6
draw.rounded_rectangle(
    [hdr_left, hdr_top + shadow_offset, hdr_right, hdr_bottom + shadow_offset],
    radius=hdr_radius,
    fill=header_shadow
)

# Header main white pill
draw.rounded_rectangle(
    [hdr_left, hdr_top, hdr_right, hdr_bottom],
    radius=hdr_radius,
    fill=header_bg,
    outline=header_border,
    width=1
)

# Small divider under header pill
draw.line([(hdr_left + 8, hdr_bottom + 18), (hdr_right - 8, hdr_bottom + 18)], fill=divider, width=1)

# Chips / filters area sits below header; draw a subtle background band (no chips themselves)
chips_band_top = hdr_bottom + 26
chips_band_bottom = chips_band_top + 160
draw.rectangle([0, chips_band_top, W, chips_band_bottom], fill=chip_section_bg)

# Light horizontal separator under chips
draw.line([(32, chips_band_bottom - 6), (W - 32, chips_band_bottom - 6)], fill=divider, width=1)

# Main seating map container (centered)
map_margin_x = 80
map_top = chips_band_bottom + 24
map_bottom = 1540
map_left = map_margin_x
map_right = W - map_margin_x
map_radius = 24

# Outer container (subtle panel)
draw.rounded_rectangle([map_left, map_top, map_right, map_bottom], radius=map_radius, fill=map_outer_bg, outline=map_border, width=2)

# Inner white canvas where the arena map sits
inner_pad = 20
draw.rounded_rectangle([map_left + inner_pad, map_top + inner_pad, map_right - inner_pad, map_bottom - inner_pad],
                       radius=18, fill=map_inner_bg)

# Add subtle inner divider lines to visually separate map header area from map content
map_header_y = map_top + inner_pad + 72
draw.line([(map_left + inner_pad + 6, map_header_y), (map_right - inner_pad - 6, map_header_y)], fill=divider, width=1)

# Bottom listing sheet (rounded top corners) - anchored near bottom
sheet_top = 1880
sheet_radius = 28
draw.rounded_rectangle([0, sheet_top, W, H], radius=sheet_radius, fill=list_sheet_bg)

# Sheet top shadow (simple thin band)
draw.line([(12, sheet_top + 2), (W - 12, sheet_top + 2)], fill=header_shadow, width=1)

# Section header divider inside sheet (for "407 Listings" header area)
hdr_inside_top = sheet_top + 32
draw.line([(24, hdr_inside_top + 88), (W - 24, hdr_inside_top + 88)], fill=divider, width=1)

# Two list item card backgrounds (rounded rectangles)
item_left = 24
item_right = W - 24
item_h = 240
first_item_top = hdr_inside_top + 24
second_item_top = first_item_top + item_h + 28

card_radius = 18
# First item card
draw.rounded_rectangle([item_left, first_item_top, item_right, first_item_top + item_h],
                       radius=card_radius, fill=list_sheet_bg, outline=item_card_border, width=1)
# Second item card
draw.rounded_rectangle([item_left, second_item_top, item_right, second_item_top + item_h],
                       radius=card_radius, fill=list_sheet_bg, outline=item_card_border, width=1)

# Thumbnails area (left side of each card) - light neutral rects as placeholders (images will be pasted on top)
thumb_w = 320
thumb_h = item_h - 28
thumb_rx = 12
thumb_radius = 12
# First thumbnail background
draw.rounded_rectangle([item_left + thumb_rx, first_item_top + 14, item_left + thumb_rx + thumb_w, first_item_top + 14 + thumb_h],
                       radius=thumb_radius, fill=muted_gray)
# Second thumbnail background
draw.rounded_rectangle([item_left + thumb_rx, second_item_top + 14, item_left + thumb_rx + thumb_w, second_item_top + 14 + thumb_h],
                       radius=thumb_radius, fill=muted_gray)

# Vertical divider between thumbnail and content within each card
vd_x = item_left + thumb_rx + thumb_w + 24
draw.line([(vd_x, first_item_top + 12), (vd_x, first_item_top + item_h - 12)], fill=muted_gray, width=1)
draw.line([(vd_x, second_item_top + 12), (vd_x, second_item_top + item_h - 12)], fill=muted_gray, width=1)

# Light separators between list items
sep_y = second_item_top + item_h + 20
draw.line([(24, sep_y), (W - 24, sep_y)], fill=divider, width=1)

# Final subtle bottom padding line
draw.line([(0, H - 1), (W, H - 1)], fill=header_border, width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_06_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-9/00_icon_Include_fees.png
try:
    _c0 = get_crop(0, 335, 108)
    canvas.paste(_c0, (544, 312), _c0)
except Exception:
    pass
layout["Include_fees"] = [544, 312, 879, 420]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_06_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-9/01_icon_Courtside.png
try:
    _c1 = get_crop(1, 286, 108)
    canvas.paste(_c1, (915, 312), _c1)
except Exception:
    pass
layout["Courtside"] = [915, 312, 1201, 420]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_06_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-9/02_icon_Quantity.png
try:
    _c2 = get_crop(2, 268, 108)
    canvas.paste(_c2, (240, 312), _c2)
except Exception:
    pass
layout["Quantity"] = [240, 312, 508, 420]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_06_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-9/03_icon_Tit.png
try:
    _c3 = get_crop(3, 156, 108)
    canvas.paste(_c3, (48, 312), _c3)
except Exception:
    pass
layout["Tit"] = [48, 312, 204, 420]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_06_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-9/04_icon_9.0.png
try:
    _c4 = get_crop(4, 1440, 371)
    canvas.paste(_c4, (0, 2589), _c4)
except Exception:
    pass
layout["9.0"] = [0, 2589, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_06_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-9/05_icon_Center.png
try:
    _c5 = get_crop(5, 203, 108)
    canvas.paste(_c5, (1237, 312), _c5)
except Exception:
    pass
layout["Center"] = [1237, 312, 1440, 420]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_06_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-9/06_icon_W_Conf_Ist_Rnd_Suns_at_Timberwolves_Gm_2.png
try:
    _c6 = get_crop(6, 1344, 156)
    canvas.paste(_c6, (48, 120), _c6)
except Exception:
    pass
layout["W_Conf_Ist_Rnd:_Suns_at_T"] = [48, 120, 1392, 276]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_06_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-9/07_icon_407_Listings.png
try:
    _c7 = get_crop(7, 1440, 455)
    canvas.paste(_c7, (0, 2134), _c7)
except Exception:
    pass
layout["407_Listings"] = [0, 2134, 1440, 2589]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_06_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-9/08_icon_Great_deal.png
try:
    _c8 = get_crop(8, 1440, 455)
    canvas.paste(_c8, (0, 2134), _c8)
except Exception:
    pass
layout["Great_deal"] = [0, 2134, 1440, 2589]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_06_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-9/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 58, 62)
    canvas.paste(_c9, (312, 3), _c9)
except Exception:
    pass
layout["icon_9"] = [312, 3, 370, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_06_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-9/10_icon_7_07_my.png
try:
    _c10 = get_crop(10, 68, 62)
    canvas.paste(_c10, (110, 1), _c10)
except Exception:
    pass
layout["7:07_my"] = [110, 1, 178, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_06_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-9/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 62, 61)
    canvas.paste(_c11, (242, 2), _c11)
except Exception:
    pass
layout["icon_11"] = [242, 2, 304, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_06_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-9/12_icon_7_07_my.png
try:
    _c12 = get_crop(12, 53, 60)
    canvas.paste(_c12, (182, 2), _c12)
except Exception:
    pass
layout["7:07_my"] = [182, 2, 235, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_06_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-9/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 53, 65)
    canvas.paste(_c13, (1152, 1), _c13)
except Exception:
    pass
layout["icon_13"] = [1152, 1, 1205, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_06_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-9/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 53, 57)
    canvas.paste(_c14, (1320, 3), _c14)
except Exception:
    pass
layout["icon_14"] = [1320, 3, 1373, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_06_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-9/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 105, 62)
    canvas.paste(_c15, (1212, 1), _c15)
except Exception:
    pass
layout["icon_15"] = [1212, 1, 1317, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_06_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-9/16_icon_Sort_by_price.png
try:
    _c16 = get_crop(16, 455, 144)
    canvas.paste(_c16, (961, 1989), _c16)
except Exception:
    pass
layout["Sort_by_price"] = [961, 1989, 1416, 2133]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_06_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-9/17_icon_icon_17.png
try:
    _c17 = get_crop(17, 50, 64)
    canvas.paste(_c17, (382, 1), _c17)
except Exception:
    pass
layout["icon_17"] = [382, 1, 432, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_06_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-9/18_icon_Center.png
try:
    _c18 = get_crop(18, 156, 156)
    canvas.paste(_c18, (1236, 120), _c18)
except Exception:
    pass
layout["Center"] = [1236, 120, 1392, 276]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_06_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-9/19_text_-228.png
try:
    _c19 = get_crop(19, 48, 27)
    canvas.paste(_c19, (467, 858), _c19)
except Exception:
    pass
layout["-228"] = [467, 858, 515, 885]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_06_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-9/20_text_-230.png
try:
    _c20 = get_crop(20, 50, 27)
    canvas.paste(_c20, (615, 858), _c20)
except Exception:
    pass
layout["-230"] = [615, 858, 665, 885]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_06_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-9/21_text_-232.png
try:
    _c21 = get_crop(21, 50, 27)
    canvas.paste(_c21, (770, 858), _c21)
except Exception:
    pass
layout["-232"] = [770, 858, 820, 885]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_06_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-9/22_text_234.png
try:
    _c22 = get_crop(22, 48, 27)
    canvas.paste(_c22, (920, 858), _c22)
except Exception:
    pass
layout["234"] = [920, 858, 968, 885]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_06_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-9/23_text_226.png
try:
    _c23 = get_crop(23, 48, 29)
    canvas.paste(_c23, (361, 886), _c23)
except Exception:
    pass
layout["226"] = [361, 886, 409, 915]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_06_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-9/24_text_236.png
try:
    _c24 = get_crop(24, 48, 29)
    canvas.paste(_c24, (1029, 886), _c24)
except Exception:
    pass
layout["236"] = [1029, 886, 1077, 915]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_06_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-9/25_text_B32.png
try:
    _c25 = get_crop(25, 48, 25)
    canvas.paste(_c25, (476, 902), _c25)
except Exception:
    pass
layout["B32"] = [476, 902, 524, 927]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_06_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-9/26_text_S40.png
try:
    _c26 = get_crop(26, 51, 30)
    canvas.paste(_c26, (647, 899), _c26)
except Exception:
    pass
layout["S40"] = [647, 899, 698, 929]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_06_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-9/27_text_541.png
try:
    _c27 = get_crop(27, 45, 28)
    canvas.paste(_c27, (724, 899), _c27)
except Exception:
    pass
layout["541"] = [724, 899, 769, 927]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_06_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-9/28_text_B5O.png
try:
    _c28 = get_crop(28, 51, 28)
    canvas.paste(_c28, (781, 899), _c28)
except Exception:
    pass
layout["B5O_"] = [781, 899, 832, 927]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_06_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-9/29_text_B5Z.png
try:
    _c29 = get_crop(29, 45, 25)
    canvas.paste(_c29, (916, 902), _c29)
except Exception:
    pass
layout["B5Z"] = [916, 902, 961, 927]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_06_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-9/30_text_129.png
try:
    _c30 = get_crop(30, 48, 29)
    canvas.paste(_c30, (541, 939), _c30)
except Exception:
    pass
layout["129"] = [541, 939, 589, 968]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_06_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-9/31_text_133.png
try:
    _c31 = get_crop(31, 46, 29)
    canvas.paste(_c31, (846, 939), _c31)
except Exception:
    pass
layout["133"] = [846, 939, 892, 968]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_06_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-9/32_text_523.png
try:
    _c32 = get_crop(32, 45, 27)
    canvas.paste(_c32, (347, 962), _c32)
except Exception:
    pass
layout["523"] = [347, 962, 392, 989]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_06_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-9/33_text_S62.png
try:
    _c33 = get_crop(33, 52, 32)
    canvas.paste(_c33, (1043, 961), _c33)
except Exception:
    pass
layout["S62"] = [1043, 961, 1095, 993]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_06_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-9/34_text_224.png
try:
    _c34 = get_crop(34, 48, 28)
    canvas.paste(_c34, (245, 1003), _c34)
except Exception:
    pass
layout["224"] = [245, 1003, 293, 1031]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_06_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-9/35_text_238.png
try:
    _c35 = get_crop(35, 45, 30)
    canvas.paste(_c35, (1147, 1001), _c35)
except Exception:
    pass
layout["238"] = [1147, 1001, 1192, 1031]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_06_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-9/36_text_124.png
try:
    _c36 = get_crop(36, 48, 27)
    canvas.paste(_c36, (312, 1031), _c36)
except Exception:
    pass
layout["124"] = [312, 1031, 360, 1058]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_06_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-9/37_text_138.png
try:
    _c37 = get_crop(37, 46, 27)
    canvas.paste(_c37, (1082, 1031), _c37)
except Exception:
    pass
layout["138"] = [1082, 1031, 1128, 1058]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_06_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-9/38_text_124.png
try:
    _c38 = get_crop(38, 46, 27)
    canvas.paste(_c38, (372, 1054), _c38)
except Exception:
    pass
layout["124"] = [372, 1054, 418, 1081]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_06_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-9/39_text_138.png
try:
    _c39 = get_crop(39, 46, 27)
    canvas.paste(_c39, (1029, 1050), _c39)
except Exception:
    pass
layout["138"] = [1029, 1050, 1075, 1077]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_06_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-9/40_text_S71.png
try:
    _c40 = get_crop(40, 43, 30)
    canvas.paste(_c40, (1133, 1068), _c40)
except Exception:
    pass
layout["S71"] = [1133, 1068, 1176, 1098]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_06_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-9/41_text_222.png
try:
    _c41 = get_crop(41, 48, 27)
    canvas.paste(_c41, (139, 1117), _c41)
except Exception:
    pass
layout["222"] = [139, 1117, 187, 1144]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_06_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-9/42_text_222.png
try:
    _c42 = get_crop(42, 46, 27)
    canvas.paste(_c42, (215, 1117), _c42)
except Exception:
    pass
layout["222"] = [215, 1117, 261, 1144]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_06_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-9/43_text_VISITORS.png
try:
    _c43 = get_crop(43, 80, 21)
    canvas.paste(_c43, (570, 1115), _c43)
except Exception:
    pass
layout["VISITORS"] = [570, 1115, 650, 1136]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_06_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-9/44_text_SCORERS.png
try:
    _c44 = get_crop(44, 85, 21)
    canvas.paste(_c44, (676, 1115), _c44)
except Exception:
    pass
layout["~SCORERS"] = [676, 1115, 761, 1136]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_06_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-9/45_text_240.png
try:
    _c45 = get_crop(45, 48, 27)
    canvas.paste(_c45, (1177, 1117), _c45)
except Exception:
    pass
layout["240"] = [1177, 1117, 1225, 1144]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_06_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-9/46_text_240.png
try:
    _c46 = get_crop(46, 48, 30)
    canvas.paste(_c46, (1251, 1114), _c46)
except Exception:
    pass
layout["240"] = [1251, 1114, 1299, 1144]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_06_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-9/47_text_CS.png
try:
    _c47 = get_crop(47, 36, 27)
    canvas.paste(_c47, (539, 1165), _c47)
except Exception:
    pass
layout["CS"] = [539, 1165, 575, 1192]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_06_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-9/48_text_CS.png
try:
    _c48 = get_crop(48, 35, 29)
    canvas.paste(_c48, (855, 1161), _c48)
except Exception:
    pass
layout["CS"] = [855, 1161, 890, 1190]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_06_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-9/49_text_221.png
try:
    _c49 = get_crop(49, 41, 27)
    canvas.paste(_c49, (199, 1193), _c49)
except Exception:
    pass
layout["221"] = [199, 1193, 240, 1220]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_06_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-9/50_text_TI8.png
try:
    _c50 = get_crop(50, 39, 25)
    canvas.paste(_c50, (263, 1179), _c50)
except Exception:
    pass
layout["TI8"] = [263, 1179, 302, 1204]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_06_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-9/51_text_S74.png
try:
    _c51 = get_crop(51, 48, 29)
    canvas.paste(_c51, (1133, 1177), _c51)
except Exception:
    pass
layout["S74"] = [1133, 1177, 1181, 1206]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_06_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-9/52_text_201.png
try:
    _c52 = get_crop(52, 46, 27)
    canvas.paste(_c52, (1193, 1193), _c52)
except Exception:
    pass
layout["201"] = [1193, 1193, 1239, 1220]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_06_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-9/53_text_220.png
try:
    _c53 = get_crop(53, 48, 28)
    canvas.paste(_c53, (215, 1269), _c53)
except Exception:
    pass
layout["220"] = [215, 1269, 263, 1297]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_06_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-9/54_text_CSS.png
try:
    _c54 = get_crop(54, 50, 27)
    canvas.paste(_c54, (643, 1279), _c54)
except Exception:
    pass
layout["CSS"] = [643, 1279, 693, 1306]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_06_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-9/55_text_202.png
try:
    _c55 = get_crop(55, 46, 30)
    canvas.paste(_c55, (1177, 1267), _c55)
except Exception:
    pass
layout["202"] = [1177, 1267, 1223, 1297]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_06_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-9/56_text_113.png
try:
    _c56 = get_crop(56, 46, 27)
    canvas.paste(_c56, (550, 1325), _c56)
except Exception:
    pass
layout["113"] = [550, 1325, 596, 1352]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_06_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-9/57_text_219.png
try:
    _c57 = get_crop(57, 48, 27)
    canvas.paste(_c57, (111, 1346), _c57)
except Exception:
    pass
layout["219"] = [111, 1346, 159, 1373]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_06_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-9/58_text_118.png
try:
    _c58 = get_crop(58, 47, 27)
    canvas.paste(_c58, (294, 1339), _c58)
except Exception:
    pass
layout["118"] = [294, 1339, 341, 1366]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_06_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-9/59_text_104.png
try:
    _c59 = get_crop(59, 46, 27)
    canvas.paste(_c59, (1098, 1341), _c59)
except Exception:
    pass
layout["104"] = [1098, 1341, 1144, 1368]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_06_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-9/60_text_203.png
try:
    _c60 = get_crop(60, 46, 28)
    canvas.paste(_c60, (1281, 1343), _c60)
except Exception:
    pass
layout["203"] = [1281, 1343, 1327, 1371]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_06_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-9/61_text_218.png
try:
    _c61 = get_crop(61, 48, 30)
    canvas.paste(_c61, (245, 1387), _c61)
except Exception:
    pass
layout["218"] = [245, 1387, 293, 1417]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_06_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-9/62_text_204.png
try:
    _c62 = get_crop(62, 48, 27)
    canvas.paste(_c62, (1147, 1390), _c62)
except Exception:
    pass
layout["204"] = [1147, 1390, 1195, 1417]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_06_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-9/63_text_522.png
try:
    _c63 = get_crop(63, 48, 29)
    canvas.paste(_c63, (347, 1429), _c63)
except Exception:
    pass
layout["522"] = [347, 1429, 395, 1458]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_06_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-9/64_text_217.png
try:
    _c64 = get_crop(64, 45, 28)
    canvas.paste(_c64, (287, 1454), _c64)
except Exception:
    pass
layout["217"] = [287, 1454, 332, 1482]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_06_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-9/65_text_CHAIRMANS.png
try:
    _c65 = get_crop(65, 110, 25)
    canvas.paste(_c65, (664, 1482), _c65)
except Exception:
    pass
layout["~CHAIRMANS"] = [664, 1482, 774, 1507]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_06_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-9/66_text_216.png
try:
    _c66 = get_crop(66, 48, 27)
    canvas.paste(_c66, (361, 1505), _c66)
except Exception:
    pass
layout["216"] = [361, 1505, 409, 1532]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_06_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-9/67_text_C1Z.png
try:
    _c67 = get_crop(67, 48, 28)
    canvas.paste(_c67, (448, 1491), _c67)
except Exception:
    pass
layout["C1Z"] = [448, 1491, 496, 1519]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_06_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-9/68_text_C15.png
try:
    _c68 = get_crop(68, 48, 28)
    canvas.paste(_c68, (525, 1491), _c68)
except Exception:
    pass
layout["C15"] = [525, 1491, 573, 1519]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_06_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-9/69_text_C13.png
try:
    _c69 = get_crop(69, 48, 28)
    canvas.paste(_c69, (601, 1491), _c69)
except Exception:
    pass
layout["C13"] = [601, 1491, 649, 1519]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_06_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-9/70_text_C9.png
try:
    _c70 = get_crop(70, 34, 25)
    canvas.paste(_c70, (835, 1494), _c70)
except Exception:
    pass
layout["C9"] = [835, 1494, 869, 1519]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_06_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-9/71_text_206.png
try:
    _c71 = get_crop(71, 48, 27)
    canvas.paste(_c71, (1031, 1505), _c71)
except Exception:
    pass
layout["206"] = [1031, 1505, 1079, 1532]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_06_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-9/72_text_214.png
try:
    _c72 = get_crop(72, 48, 27)
    canvas.paste(_c72, (472, 1535), _c72)
except Exception:
    pass
layout["214"] = [472, 1535, 520, 1562]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_06_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-9/73_text_208.png
try:
    _c73 = get_crop(73, 48, 27)
    canvas.paste(_c73, (927, 1535), _c73)
except Exception:
    pass
layout["208"] = [927, 1535, 975, 1562]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_06_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-9/74_text_213.png
try:
    _c74 = get_crop(74, 48, 29)
    canvas.paste(_c74, (541, 1552), _c74)
except Exception:
    pass
layout["213"] = [541, 1552, 589, 1581]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_06_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-9/75_text_407_Listings.png
try:
    _c75 = get_crop(75, 333, 78)
    canvas.paste(_c75, (56, 2026), _c75)
except Exception:
    pass
layout["407_Listings"] = [56, 2026, 389, 2104]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_06_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-9/76_text_S117_each.png
try:
    _c76 = get_crop(76, 1440, 371)
    canvas.paste(_c76, (0, 2589), _c76)
except Exception:
    pass
layout["S117_each"] = [0, 2589, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_06_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-9/77_text_Price_includes_fees.png
try:
    _c77 = get_crop(77, 1440, 371)
    canvas.paste(_c77, (0, 2589), _c77)
except Exception:
    pass
layout["Price_includes_fees"] = [0, 2589, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_06_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-9/78_text_9.0.png
try:
    _c78 = get_crop(78, 54, 36)
    canvas.paste(_c78, (501, 2809), _c78)
except Exception:
    pass
layout["9.0"] = [501, 2809, 555, 2845]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_06_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-9/79_text_Amazing_deal.png
try:
    _c79 = get_crop(79, 1440, 371)
    canvas.paste(_c79, (0, 2589), _c79)
except Exception:
    pass
layout["Amazing_deal"] = [0, 2589, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_06_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-9/80_text_2_tickets.png
try:
    _c80 = get_crop(80, 180, 43)
    canvas.paste(_c80, (489, 2876), _c80)
except Exception:
    pass
layout["2_tickets"] = [489, 2876, 669, 2919]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_06_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-9/81_clickable_Back.png
try:
    _c81 = get_crop(81, 156, 156)
    canvas.paste(_c81, (48, 120), _c81)
except Exception:
    pass
layout["Back"] = [48, 120, 204, 276]
