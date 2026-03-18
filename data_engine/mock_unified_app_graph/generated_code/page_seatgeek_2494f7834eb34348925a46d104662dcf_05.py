# page_id: page_seatgeek_2494f7834eb34348925a46d104662dcf_05
# screenshot: 2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-8.png
# step_index: 5/9
# task: Open SeatGeek. Search for "Book of Mormon". Add the show to favorite. Select date April 26. Set the ticket number to 2 and proceed. What is the lowest price for each ticket?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: fallback_compose
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Top status bar
draw.rectangle([(0, 0), (1440, 88)], fill=(30, 30, 30))

# thin divider under status bar
draw.line([(0, 88), (1440, 88)], fill=(220, 220, 220), width=1)

# Hero banner background (soft off-white to warm)
draw.rectangle([(0, 88), (1440, 520)], fill=(253, 250, 243))

# Soft radial-ish spotlight (large pale ellipse behind main hero image)
spot_bbox = (-300, -200, 1740, 860)
draw.ellipse(spot_bbox, fill=(255, 252, 244))

# Decorative gold toothed edges on left and right of hero banner
gold_light = (224, 181, 63)
gold_dark = (183, 137, 41)
left_x = 0
right_x = 1440
y_top = 88
y_bottom = 520
# draw triangular teeth repeating down the left edge
tooth_height = 32
tooth_width = 140
y = y_top
toggle = True
while y < y_bottom + tooth_height:
    p1 = (left_x, y)
    p2 = (left_x + tooth_width, y + tooth_height/2)
    p3 = (left_x, y + tooth_height)
    draw.polygon([p1, p2, p3], fill=gold_light if toggle else gold_dark)
    y += tooth_height
    toggle = not toggle

# right edge mirrored
y = y_top
toggle = False
while y < y_bottom + tooth_height:
    p1 = (right_x, y)
    p2 = (right_x - tooth_width, y + tooth_height/2)
    p3 = (right_x, y + tooth_height)
    draw.polygon([p1, p2, p3], fill=gold_light if toggle else gold_dark)
    y += tooth_height
    toggle = not toggle

# subtle horizontal gradient band near bottom of hero area
for i in range(12):
    alpha_shade = 245 - i*6
    shade = (alpha_shade, alpha_shade, alpha_shade)
    y_line = 480 + i
    draw.line([(0, y_line), (1440, y_line)], fill=shade)

# Card container under hero (title area) - rounded white card with soft shadow
card_x0, card_y0 = 40, 520
card_x1, card_y1 = 1400, 760
# shadow
for s in range(1, 6):
    shadow_bbox = [card_x0 + s, card_y0 + s, card_x1 + s, card_y1 + s]
    shade = 245 + s  # progressively lighter
    draw.rounded_rectangle(shadow_bbox, radius=18, fill=(shade, shade, shade))
# white card
draw.rounded_rectangle([card_x0, card_y0, card_x1, card_y1], radius=18, fill=(255, 255, 255))

# small divider under the card
draw.line([(card_x0 + 20, card_y1 + 1), (card_x1 - 20, card_y1 + 1)], fill=(230, 230, 230), width=1)

# Main content panel background (list area)
panel_top = card_y1 + 24
panel_left = 0
panel_right = 1440
panel_bottom = 2960
draw.rectangle([(panel_left, panel_top), (panel_right, panel_bottom)], fill=(250, 250, 250))

# Section header backgrounds (subtle white blocks to anchor groups)
# "New York, NY" section header band area
hdr1_y = panel_top + 20
draw.rectangle([(40, hdr1_y), (1400, hdr1_y + 120)], fill=(255, 255, 255))
# subtle bottom border for header
draw.line([(40, hdr1_y + 120), (1400, hdr1_y + 120)], fill=(235, 235, 235), width=1)

# List area block behind date rows (a large white card to host list items)
list_block_y0 = hdr1_y + 140
list_block_y1 = list_block_y0 + 1600
draw.rounded_rectangle([40, list_block_y0, 1400, list_block_y1], radius=12, fill=(255, 255, 255))

# Add faint separators within the list block to suggest row structure (do not draw pills or icons)
sep_y = list_block_y0 + 180
while sep_y < list_block_y1 - 40:
    draw.line([(80, sep_y), (1360, sep_y)], fill=(245, 245, 245), width=1)
    sep_y += 293  # roughly align with detected row heights without drawing item content

# "All Shows" header area below the list block
all_shows_y = list_block_y1 + 40
draw.rectangle([(40, all_shows_y), (1400, all_shows_y + 80)], fill=(250, 250, 250))
# subtle underline for the All Shows header
draw.line([(40, all_shows_y + 80), (1400, all_shows_y + 80)], fill=(230, 230, 230), width=1)

# Second list block for "All Shows"
all_list_y0 = all_shows_y + 110
all_list_y1 = all_list_y0 + 1100
draw.rounded_rectangle([40, all_list_y0, 1400, all_list_y1], radius=12, fill=(255, 255, 255))
# separators for this list
sep_y = all_list_y0 + 160
while sep_y < all_list_y1 - 40:
    draw.line([(80, sep_y), (1360, sep_y)], fill=(245, 245, 245), width=1)
    sep_y += 293

# subtle footer area at bottom
footer_top = all_list_y1 + 40
draw.rectangle([(0, footer_top), (1440, 2960)], fill=(250, 250, 250))
draw.line([(40, footer_top), (1400, footer_top)], fill=(230, 230, 230), width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_05_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-8/00_icon_Eugene_0_Neill_Theatre.png
try:
    _c0 = get_crop(0, 1440, 293)
    canvas.paste(_c0, (0, 1279), _c0)
except Exception:
    pass
layout["Eugene_0'Neill_Theatre"] = [0, 1279, 1440, 1572]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_05_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-8/01_icon_24.png
try:
    _c1 = get_crop(1, 1440, 293)
    canvas.paste(_c1, (0, 1572), _c1)
except Exception:
    pass
layout["24"] = [0, 1572, 1440, 1865]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_05_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-8/02_icon_Eugene_0_Neill_Theatre.png
try:
    _c2 = get_crop(2, 1440, 293)
    canvas.paste(_c2, (0, 1572), _c2)
except Exception:
    pass
layout["Eugene_0'Neill_Theatre"] = [0, 1572, 1440, 1865]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_05_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-8/03_icon_23.png
try:
    _c3 = get_crop(3, 1440, 293)
    canvas.paste(_c3, (0, 1279), _c3)
except Exception:
    pass
layout["23"] = [0, 1279, 1440, 1572]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_05_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-8/04_icon_25.png
try:
    _c4 = get_crop(4, 1440, 293)
    canvas.paste(_c4, (0, 1865), _c4)
except Exception:
    pass
layout["25"] = [0, 1865, 1440, 2158]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_05_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-8/05_icon_23.png
try:
    _c5 = get_crop(5, 1440, 293)
    canvas.paste(_c5, (0, 2596), _c5)
except Exception:
    pass
layout["23"] = [0, 2596, 1440, 2889]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_05_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-8/06_icon_MORM.png
try:
    _c6 = get_crop(6, 1440, 126)
    canvas.paste(_c6, (0, 933), _c6)
except Exception:
    pass
layout["MORM"] = [0, 933, 1440, 1059]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_05_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-8/07_icon_Eugene_0_Neill_Theatre.png
try:
    _c7 = get_crop(7, 1440, 293)
    canvas.paste(_c7, (0, 1865), _c7)
except Exception:
    pass
layout["Eugene_0'Neill_Theatre"] = [0, 1865, 1440, 2158]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_05_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-8/08_icon_6.50.png
try:
    _c8 = get_crop(8, 144, 144)
    canvas.paste(_c8, (36, 84), _c8)
except Exception:
    pass
layout["6.50"] = [36, 84, 180, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_05_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-8/09_icon_26.png
try:
    _c9 = get_crop(9, 1440, 293)
    canvas.paste(_c9, (0, 2158), _c9)
except Exception:
    pass
layout["26"] = [0, 2158, 1440, 2451]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_05_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-8/10_icon_Track_this_performer.png
try:
    _c10 = get_crop(10, 144, 144)
    canvas.paste(_c10, (1104, 84), _c10)
except Exception:
    pass
layout["Track_this_performer"] = [1104, 84, 1248, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_05_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-8/11_icon_New_York_NY.png
try:
    _c11 = get_crop(11, 1440, 293)
    canvas.paste(_c11, (0, 2596), _c11)
except Exception:
    pass
layout["New_York,_NY"] = [0, 2596, 1440, 2889]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_05_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-8/12_icon_Share_this_performer.png
try:
    _c12 = get_crop(12, 144, 144)
    canvas.paste(_c12, (1260, 84), _c12)
except Exception:
    pass
layout["Share_this_performer"] = [1260, 84, 1404, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_05_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-8/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 103, 116)
    canvas.paste(_c13, (1297, 945), _c13)
except Exception:
    pass
layout["icon_13"] = [1297, 945, 1400, 1061]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_05_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-8/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 105, 88)
    canvas.paste(_c14, (1118, 0), _c14)
except Exception:
    pass
layout["icon_14"] = [1118, 0, 1223, 88]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_05_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-8/15_icon_Eugene_0_Neill_Theatre.png
try:
    _c15 = get_crop(15, 1440, 293)
    canvas.paste(_c15, (0, 2158), _c15)
except Exception:
    pass
layout["Eugene_0'Neill_Theatre"] = [0, 2158, 1440, 2451]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_05_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-8/16_icon_icon_16.png
try:
    _c16 = get_crop(16, 56, 63)
    canvas.paste(_c16, (1319, 968), _c16)
except Exception:
    pass
layout["icon_16"] = [1319, 968, 1375, 1031]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_05_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-8/17_text_6.50.png
try:
    _c17 = get_crop(17, 89, 41)
    canvas.paste(_c17, (22, 17), _c17)
except Exception:
    pass
layout["6.50"] = [22, 17, 111, 58]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_05_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-8/18_text_Wv.png
try:
    _c18 = get_crop(18, 47, 43)
    canvas.paste(_c18, (124, 15), _c18)
except Exception:
    pass
layout["Wv"] = [124, 15, 171, 58]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_05_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-8/19_text_New_York_NY.png
try:
    _c19 = get_crop(19, 352, 65)
    canvas.paste(_c19, (55, 1177), _c19)
except Exception:
    pass
layout["New_York,_NY"] = [55, 1177, 407, 1242]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_05_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-8/20_text_AII_Shows.png
try:
    _c20 = get_crop(20, 249, 55)
    canvas.paste(_c20, (60, 2495), _c20)
except Exception:
    pass
layout["AII_Shows"] = [60, 2495, 309, 2550]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_05_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-8/21_text_The_Rook_Of_Mormon.png
try:
    _c21 = get_crop(21, 1440, 293)
    canvas.paste(_c21, (0, 2596), _c21)
except Exception:
    pass
layout["The_Rook_Of_Mormon"] = [0, 2596, 1440, 2889]
