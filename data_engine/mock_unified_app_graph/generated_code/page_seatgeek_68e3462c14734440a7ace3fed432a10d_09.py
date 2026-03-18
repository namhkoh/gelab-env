# page_id: page_seatgeek_68e3462c14734440a7ace3fed432a10d_09
# screenshot: 2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-12.png
# step_index: 9/13
# task: Open SeatGeek and change the current location to Los Angeles. Then find the first concert show and track its performer.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw background and structural UI elements for the provided canvas
# Available variables: canvas (PIL.Image), draw (PIL.ImageDraw), font_sm, font_md, font_lg, font_xl

W, H = canvas.size

# Colors
bg = (242, 244, 246)          # page background (light cool gray)
status_bar_bg = (230, 232, 234)
header_shadow = (215, 219, 222)
card_bg = (255, 255, 255)
muted_divider = (220, 223, 226)
chip_bg = (255, 255, 255)
chip_border = (227, 230, 232)
chip_selected = (20, 20, 20)  # selected chip (dark)
map_outer = (236, 238, 240)
map_border = (180, 184, 188)
stage_fill = (84, 88, 92)
listing_card_divider = (236, 238, 240)

# Fill overall background
draw.rectangle([(0,0),(W,H)], fill=bg)

# Status bar area (top ~64px)
status_h = 64
draw.rectangle([(0,0),(W,status_h)], fill=status_bar_bg)
# subtle bottom divider of status bar
draw.line([(0,status_h-1),(W,status_h-1)], fill=muted_divider, width=1)

# Header pill (rounded) with soft shadow
header_top = 100
header_bottom = 260
header_left = 28
header_right = W - 28
header_radius = 84
# shadow (slightly offset)
draw.rounded_rectangle(
    [(header_left, header_top+6), (header_right, header_bottom+6)],
    radius=header_radius, fill=header_shadow
)
# white pill
draw.rounded_rectangle(
    [(header_left, header_top), (header_right, header_bottom)],
    radius=header_radius, fill=card_bg, outline=muted_divider, width=1
)

# Filter chips row background area (just ensure spacing & separators)
chips_y = 312
# Chips given positions in detection - draw their background pills only (no icons/text)
chips = [
    (240, 312, 240+268, 312+108),   # Quantity
    (544, 312, 544+335, 312+108),   # Include fees (selected)
    (915, 312, 915+303, 312+108),   # Best seats
    (1254,312,1254+186,312+108)     # Low pri
]
for idx, (x1,y1,x2,y2) in enumerate(chips):
    r = (y2-y1)//2
    if idx == 1:
        # selected chip darker pill (draw darker background)
        draw.rounded_rectangle([(x1,y1),(x2,y2)], radius=r, fill=chip_selected)
    else:
        draw.rounded_rectangle([(x1,y1),(x2,y2)], radius=r, fill=chip_bg, outline=chip_border, width=1)

# Large seating map area - outer subtle container and inner map
map_left, map_top = 120, 360
map_right, map_bottom = W-120, 1760
# outer soft background
draw.ellipse([(map_left, map_top), (map_right, map_bottom)], fill=map_outer)
# inner white map (seat map background) with border
inner_margin = 30
draw.ellipse(
    [(map_left+inner_margin, map_top+inner_margin), (map_right-inner_margin, map_bottom-inner_margin)],
    fill=card_bg, outline=map_border, width=6
)
# Stage trapezoid at top of map (background only)
stage_w = 360
stage_h = 120
sx = (W//2) - stage_w//2
sy = map_top + 12
draw.polygon([
    (sx, sy+stage_h),
    (sx + stage_w, sy+stage_h),
    (sx + stage_w*0.85, sy + 18),
    (sx + stage_w*0.15, sy + 18),
], fill=stage_fill, outline=map_border)

# Slight inner "rim" around map to emulate the map outline (two-tone)
rim_offset = 12
draw.ellipse(
    [(map_left+rim_offset, map_top+rim_offset), (map_right-rim_offset, map_bottom-rim_offset)],
    outline=(230,231,233), width=4
)

# Listings container (bottom area) as large rounded white card
listings_top = 1980
listings_left = 20
listings_right = W-20
listings_bottom = H - 20
listings_radius = 28
# subtle shadow above listings
draw.rectangle([(listings_left, listings_top-6),(listings_right, listings_top)], fill=header_shadow)
draw.rounded_rectangle(
    [(listings_left, listings_top),(listings_right, listings_bottom)],
    radius=listings_radius, fill=card_bg, outline=muted_divider, width=1
)

# Title area within listings (divider line under title)
title_divider_y = listings_top + 84
draw.line([(listings_left+24, title_divider_y),(listings_right-24, title_divider_y)], fill=listing_card_divider, width=1)

# Draw two listing card backgrounds and separators (no text/icons)
card_margin_x = listings_left + 24
card_w = listings_right - 24 - card_margin_x
# Approximate heights for two listing items
first_card_top = title_divider_y + 28
first_card_bottom = first_card_top + 360
second_card_top = first_card_bottom + 24
second_card_bottom = second_card_top + 360

# Card background rectangles (rounded)
card_radius = 16
draw.rounded_rectangle([(card_margin_x, first_card_top),(listings_right-24, first_card_bottom)],
                       radius=card_radius, fill=card_bg, outline=listing_card_divider, width=1)
draw.rounded_rectangle([(card_margin_x, second_card_top),(listings_right-24, second_card_bottom)],
                       radius=card_radius, fill=card_bg, outline=listing_card_divider, width=1)

# Thin separators between cards and below
sep_y1 = first_card_bottom + 12
draw.line([(card_margin_x, sep_y1),(listings_right-24, sep_y1)], fill=listing_card_divider, width=1)
sep_y2 = second_card_bottom + 12
draw.line([(card_margin_x, sep_y2),(listings_right-24, sep_y2)], fill=listing_card_divider, width=1)

# Small left thumbnails background boxes inside each card (to reserve space)
thumb_w, thumb_h = 210, 160
thumb_radius = 12
thumb_x = card_margin_x + 12
thumb_y1 = first_card_top + 18
thumb_y2 = second_card_top + 18
draw.rounded_rectangle([(thumb_x, thumb_y1),(thumb_x+thumb_w, thumb_y1+thumb_h)],
                       radius=thumb_radius, fill=map_outer, outline=muted_divider)
draw.rounded_rectangle([(thumb_x, thumb_y2),(thumb_x+thumb_w, thumb_y2+thumb_h)],
                       radius=thumb_radius, fill=map_outer, outline=muted_divider)

# Right-side small divider for sort control area near top of listings (visual only)
sort_area_right = listings_right - 36
sort_area_left = sort_area_right - 260
sort_area_top = listings_top + 18
sort_area_bottom = sort_area_top + 60
draw.rounded_rectangle([(sort_area_left, sort_area_top),(sort_area_right, sort_area_bottom)],
                       radius=30, fill=card_bg, outline=muted_divider)

# Additional subtle horizontal separators across page for structure
for y in [header_bottom+24, map_bottom+24, title_divider_y-8]:
    draw.line([(36,y),(W-36,y)], fill=listing_card_divider, width=1)

# Small decorative bottom shadow for listings container
draw.rectangle([(listings_left, listings_bottom-6),(listings_right, listings_bottom)], fill=(240,242,244))

# Done drawing structural background elements

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_09_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-12/00_icon_Include.png
try:
    _c0 = get_crop(0, 335, 108)
    canvas.paste(_c0, (544, 312), _c0)
except Exception:
    pass
layout["Include"] = [544, 312, 879, 420]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_09_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-12/01_icon_8.8.png
try:
    _c1 = get_crop(1, 1440, 371)
    canvas.paste(_c1, (0, 2589), _c1)
except Exception:
    pass
layout["8.8"] = [0, 2589, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_09_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-12/02_icon_Best_seats.png
try:
    _c2 = get_crop(2, 303, 108)
    canvas.paste(_c2, (915, 312), _c2)
except Exception:
    pass
layout["Best_seats"] = [915, 312, 1218, 420]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_09_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-12/03_icon_Quantity.png
try:
    _c3 = get_crop(3, 268, 108)
    canvas.paste(_c3, (240, 312), _c3)
except Exception:
    pass
layout["Quantity"] = [240, 312, 508, 420]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_09_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-12/04_icon_Tit.png
try:
    _c4 = get_crop(4, 156, 108)
    canvas.paste(_c4, (48, 312), _c4)
except Exception:
    pass
layout["Tit"] = [48, 312, 204, 420]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_09_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-12/05_icon_2462_Listings.png
try:
    _c5 = get_crop(5, 1440, 455)
    canvas.paste(_c5, (0, 2134), _c5)
except Exception:
    pass
layout["2462_Listings"] = [0, 2134, 1440, 2589]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_09_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-12/06_icon_Low_pri.png
try:
    _c6 = get_crop(6, 186, 108)
    canvas.paste(_c6, (1254, 312), _c6)
except Exception:
    pass
layout["Low_pri"] = [1254, 312, 1440, 420]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_09_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-12/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 53, 65)
    canvas.paste(_c7, (1152, 1), _c7)
except Exception:
    pass
layout["icon_7"] = [1152, 1, 1205, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_09_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-12/08_icon_Low_pri.png
try:
    _c8 = get_crop(8, 156, 156)
    canvas.paste(_c8, (1236, 120), _c8)
except Exception:
    pass
layout["Low_pri"] = [1236, 120, 1392, 276]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_09_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-12/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 104, 62)
    canvas.paste(_c9, (1211, 1), _c9)
except Exception:
    pass
layout["icon_9"] = [1211, 1, 1315, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_09_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-12/10_icon_Wy.png
try:
    _c10 = get_crop(10, 62, 62)
    canvas.paste(_c10, (110, 1), _c10)
except Exception:
    pass
layout["Wy"] = [110, 1, 172, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_09_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-12/11_icon_Wy.png
try:
    _c11 = get_crop(11, 57, 61)
    canvas.paste(_c11, (181, 1), _c11)
except Exception:
    pass
layout["Wy"] = [181, 1, 238, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_09_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-12/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 54, 57)
    canvas.paste(_c12, (1320, 3), _c12)
except Exception:
    pass
layout["icon_12"] = [1320, 3, 1374, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_09_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-12/13_icon_Keep_the_Party_Going_A_Tribute_to_Jimmy_.png
try:
    _c13 = get_crop(13, 955, 156)
    canvas.paste(_c13, (204, 120), _c13)
except Exception:
    pass
layout["Keep_the_Party_Going:_A_T"] = [204, 120, 1159, 276]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_09_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-12/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 52, 58)
    canvas.paste(_c14, (315, 3), _c14)
except Exception:
    pass
layout["icon_14"] = [315, 3, 367, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_09_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-12/15_icon_Wy.png
try:
    _c15 = get_crop(15, 53, 59)
    canvas.paste(_c15, (248, 2), _c15)
except Exception:
    pass
layout["Wy"] = [248, 2, 301, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_09_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-12/16_icon_Sort_by_price.png
try:
    _c16 = get_crop(16, 455, 144)
    canvas.paste(_c16, (961, 1989), _c16)
except Exception:
    pass
layout["Sort_by_price"] = [961, 1989, 1416, 2133]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_09_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-12/17_icon_icon_17.png
try:
    _c17 = get_crop(17, 50, 62)
    canvas.paste(_c17, (382, 0), _c17)
except Exception:
    pass
layout["icon_17"] = [382, 0, 432, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_09_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-12/18_icon_8.31.png
try:
    _c18 = get_crop(18, 103, 64)
    canvas.paste(_c18, (4, 0), _c18)
except Exception:
    pass
layout["8.31"] = [4, 0, 107, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_09_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-12/19_icon_Amazing_deal.png
try:
    _c19 = get_crop(19, 1440, 455)
    canvas.paste(_c19, (0, 2134), _c19)
except Exception:
    pass
layout["Amazing_deal"] = [0, 2134, 1440, 2589]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_09_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-12/20_text_STAGE.png
try:
    _c20 = get_crop(20, 42, 16)
    canvas.paste(_c20, (699, 674), _c20)
except Exception:
    pass
layout["STAGE"] = [699, 674, 741, 690]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_09_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-12/21_text_K1.png
try:
    _c21 = get_crop(21, 32, 27)
    canvas.paste(_c21, (467, 1184), _c21)
except Exception:
    pass
layout["K1"] = [467, 1184, 499, 1211]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_09_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-12/22_text_F1.png
try:
    _c22 = get_crop(22, 29, 27)
    canvas.paste(_c22, (939, 1184), _c22)
except Exception:
    pass
layout["F1"] = [939, 1184, 968, 1211]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_09_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-12/23_text_J2.png
try:
    _c23 = get_crop(23, 32, 30)
    canvas.paste(_c23, (548, 1209), _c23)
except Exception:
    pass
layout["J2"] = [548, 1209, 580, 1239]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_09_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-12/24_text_G2.png
try:
    _c24 = get_crop(24, 37, 27)
    canvas.paste(_c24, (855, 1209), _c24)
except Exception:
    pass
layout["G2"] = [855, 1209, 892, 1236]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_09_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-12/25_text_U1.png
try:
    _c25 = get_crop(25, 36, 29)
    canvas.paste(_c25, (465, 1628), _c25)
except Exception:
    pass
layout["U1"] = [465, 1628, 501, 1657]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_09_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-12/26_text_Q1.png
try:
    _c26 = get_crop(26, 34, 27)
    canvas.paste(_c26, (936, 1630), _c26)
except Exception:
    pass
layout["Q1"] = [936, 1630, 970, 1657]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_09_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-12/27_text_2462_Listings.png
try:
    _c27 = get_crop(27, 367, 77)
    canvas.paste(_c27, (54, 2027), _c27)
except Exception:
    pass
layout["2462_Listings"] = [54, 2027, 421, 2104]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_09_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-12/28_text_S249_each.png
try:
    _c28 = get_crop(28, 1440, 371)
    canvas.paste(_c28, (0, 2589), _c28)
except Exception:
    pass
layout["S249_each"] = [0, 2589, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_09_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-12/29_text_Price_includes_fees.png
try:
    _c29 = get_crop(29, 1440, 371)
    canvas.paste(_c29, (0, 2589), _c29)
except Exception:
    pass
layout["Price_includes_fees"] = [0, 2589, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_09_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-12/30_text_8.8.png
try:
    _c30 = get_crop(30, 50, 29)
    canvas.paste(_c30, (502, 2812), _c30)
except Exception:
    pass
layout["8.8"] = [502, 2812, 552, 2841]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_09_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-12/31_text_Amazing_deal.png
try:
    _c31 = get_crop(31, 1440, 371)
    canvas.paste(_c31, (0, 2589), _c31)
except Exception:
    pass
layout["Amazing_deal"] = [0, 2589, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_09_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-12/32_text_2_tickets.png
try:
    _c32 = get_crop(32, 180, 43)
    canvas.paste(_c32, (489, 2876), _c32)
except Exception:
    pass
layout["2_tickets"] = [489, 2876, 669, 2919]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_09_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-12/33_text_ELECTRIC.png
try:
    _c33 = get_crop(33, 88, 36)
    canvas.paste(_c33, (605, 1202), _c33)
except Exception:
    pass
layout["ELECTRIC"] = [605, 1202, 693, 1238]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_09_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-12/34_text_AUDIO.png
try:
    _c34 = get_crop(34, 61, 25)
    canvas.paste(_c34, (759, 1209), _c34)
except Exception:
    pass
layout["AUDIO"] = [759, 1209, 820, 1234]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_09_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-12/35_clickable_Back.png
try:
    _c35 = get_crop(35, 156, 156)
    canvas.paste(_c35, (48, 120), _c35)
except Exception:
    pass
layout["Back"] = [48, 120, 204, 276]
