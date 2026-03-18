# page_id: page_seatgeek_03681efe5b614d869915a8728b755f3d_05
# screenshot: 2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-8.png
# step_index: 5/10
# task: Open SeatGeek. Search "Metropolitan Opera". Find the next available show. Filter by "best seats". What section are they in for the lowest price tickets?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw background and structural elements for the UI page (no text/icons)
# Available variables: canvas (PIL Image 1440x2960 RGB), draw (ImageDraw), font_sm, font_md, font_lg, font_xl

W, H = canvas.size

# Overall page background (slightly warm off-white)
draw.rectangle([(0, 0), (W, H)], fill=(250, 250, 251))

# Status bar (top) - dark area for time/signal icons
status_h = 72
draw.rectangle([(0, 0), (W, status_h)], fill=(38, 38, 38))

# Subtle top divider under status bar
draw.line([(0, status_h), (W, status_h)], fill=(0, 0, 0, 16), width=1)

# Map/header image area (under status bar) - pale map-like background
map_top = status_h
map_bottom = 560
draw.rectangle([(0, map_top), (W, map_bottom)], fill=(232, 244, 255))

# Soft curved bottom edge for map area using a large rounded rectangle mask effect
# Draw a white rounded card that overlaps bottom of map to create the card-over-map look
card_top = map_bottom - 40
card_left = 20
card_right = W - 20
card_bottom = 920
card_radius = 28

# Shadow for the card (subtle)
shadow_offset = 6
draw.rounded_rectangle(
    [(card_left, card_top + shadow_offset), (card_right, card_bottom + shadow_offset)],
    radius=card_radius,
    fill=(235, 235, 235)
)

# White card that contains the location header (text/icons will be pasted later)
draw.rounded_rectangle(
    [(card_left, card_top), (card_right, card_bottom)],
    radius=card_radius,
    fill=(255, 255, 255)
)

# Thin divider under the header card
div_y = card_bottom + 22
draw.line([(24, div_y), (W - 24, div_y)], fill=(230, 230, 230), width=1)

# "Popular events" section area background (keeps content readable)
pop_section_top = div_y + 36
pop_section_bottom = pop_section_top + 360
draw.rectangle([(0, pop_section_top), (W, pop_section_bottom)], fill=(250, 250, 251))

# Draw rounded placeholders behind the three event thumbnails (they will be overlaid)
# Using positions and sizes from detected elements: (x, y, w, h)
event_thumbs = [
    (48, 1273, 462, 519),    # left thumbnail placeholder
    (546, 1273, 462, 519),   # middle thumbnail placeholder
    (1044, 1273, 396, 533)   # right thumbnail placeholder
]
for (x, y, w, h) in event_thumbs:
    # Slight rounded light background for each thumbnail card
    rx = 24
    draw.rounded_rectangle(
        [(x, y), (x + w, y + h)],
        radius=18,
        fill=(250, 250, 250),
        outline=(235, 235, 235),
        width=1
    )
    # Subtle inner shading bar at bottom of each card to imply image area (very subtle)
    shade_h = 28
    draw.rectangle([(x + 12, y + h - shade_h - 12), (x + w - 12, y + h - 12)], fill=(245, 245, 245))

# Separator line under the popular events row
sep_y = pop_section_top + 220
draw.line([(24, sep_y), (W - 24, sep_y)], fill=(235, 235, 235), width=1)

# Seating charts section card area
seating_top = sep_y + 36
seating_bottom = seating_top + 420
draw.rectangle([(0, seating_top), (W, seating_bottom)], fill=(250, 250, 251))

# Draw three seating chart placeholders (positions from detected icons at y ~2049)
seating_thumbs = [
    (48, 2049, 462, 437),
    (546, 2049, 462, 437),
    (1044, 2049, 396, 437)
]
for (x, y, w, h) in seating_thumbs:
    draw.rounded_rectangle(
        [(x, y), (x + w, y + h)],
        radius=16,
        fill=(248, 248, 249),
        outline=(230, 230, 230),
        width=1
    )

# Thin divider under seating charts
seating_div_y = seating_bottom + 8
draw.line([(24, seating_div_y), (W - 24, seating_div_y)], fill=(230, 230, 230), width=1)

# "All events" list area background (keeps subtle section separation)
all_events_top = seating_div_y + 28
all_events_bottom = H - 160
draw.rectangle([(0, all_events_top), (W, all_events_bottom)], fill=(255, 255, 255))

# Draw faint horizontal separators for a few sample list rows (structural only)
list_row_height = 110
row_x1 = 24
row_x2 = W - 24
for i in range(4):
    ry = all_events_top + i * list_row_height
    # light divider between rows
    draw.line([(row_x1, ry + list_row_height), (row_x2, ry + list_row_height)], fill=(240, 240, 240), width=1)

# Floating small toolbar/button background at right side of header (no icons drawn)
toolbar_w = 86
toolbar_h = 86
toolbar_x = W - toolbar_w - 28
toolbar_y = status_h + 10
draw.rounded_rectangle([(toolbar_x, toolbar_y), (toolbar_x + toolbar_w, toolbar_y + toolbar_h)], radius=18, fill=(255, 255, 255), outline=(220, 220, 220))

# Subtle left back arrow background in map area (no arrow icon)
back_w = 86
back_h = 86
back_x = 28
back_y = status_h + 10
draw.rounded_rectangle([(back_x, back_y), (back_x + back_w, back_y + back_h)], radius=18, fill=(255, 255, 255), outline=(220, 220, 220))

# Final faint bottom shadow to ground the page
bottom_shadow_h = 18
draw.rectangle([(0, H - bottom_shadow_h), (W, H)], fill=(245, 245, 246))

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_05_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-8/00_icon_08.png
try:
    _c0 = get_crop(0, 462, 437)
    canvas.paste(_c0, (48, 2049), _c0)
except Exception:
    pass
layout["08"] = [48, 2049, 510, 2486]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_05_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-8/01_icon_S93.png
try:
    _c1 = get_crop(1, 462, 519)
    canvas.paste(_c1, (546, 1273), _c1)
except Exception:
    pass
layout["S93+"] = [546, 1273, 1008, 1792]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_05_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-8/02_icon_38.png
try:
    _c2 = get_crop(2, 462, 437)
    canvas.paste(_c2, (546, 2049), _c2)
except Exception:
    pass
layout["38"] = [546, 2049, 1008, 2486]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_05_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-8/03_icon_S82.png
try:
    _c3 = get_crop(3, 462, 519)
    canvas.paste(_c3, (48, 1273), _c3)
except Exception:
    pass
layout["S82+"] = [48, 1273, 510, 1792]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_05_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-8/04_icon_S348.png
try:
    _c4 = get_crop(4, 396, 533)
    canvas.paste(_c4, (1044, 1273), _c4)
except Exception:
    pass
layout["S348+"] = [1044, 1273, 1440, 1806]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_05_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-8/05_icon_See_more_options.png
try:
    _c5 = get_crop(5, 204, 174)
    canvas.paste(_c5, (1236, 806), _c5)
except Exception:
    pass
layout["See_more_options"] = [1236, 806, 1440, 980]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_05_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-8/06_icon_E1O6th.png
try:
    _c6 = get_crop(6, 44, 59)
    canvas.paste(_c6, (1327, 4), _c6)
except Exception:
    pass
layout["E1O6th"] = [1327, 4, 1371, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_05_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-8/07_icon_88.png
try:
    _c7 = get_crop(7, 396, 437)
    canvas.paste(_c7, (1044, 2049), _c7)
except Exception:
    pass
layout["88"] = [1044, 2049, 1440, 2486]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_05_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-8/08_icon_Romeo_et_Juliette.png
try:
    _c8 = get_crop(8, 396, 437)
    canvas.paste(_c8, (1044, 2049), _c8)
except Exception:
    pass
layout["Romeo_et_Juliette"] = [1044, 2049, 1440, 2486]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_05_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-8/09_icon_Apr.png
try:
    _c9 = get_crop(9, 462, 519)
    canvas.paste(_c9, (546, 1273), _c9)
except Exception:
    pass
layout["Apr"] = [546, 1273, 1008, 1792]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_05_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-8/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 42, 72)
    canvas.paste(_c10, (1398, 1304), _c10)
except Exception:
    pass
layout["icon_10"] = [1398, 1304, 1440, 1376]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_05_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-8/11_icon_27_8_PM.png
try:
    _c11 = get_crop(11, 462, 519)
    canvas.paste(_c11, (48, 1273), _c11)
except Exception:
    pass
layout["27,8_PM"] = [48, 1273, 510, 1792]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_05_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-8/12_text_7.57_Wy.png
try:
    _c12 = get_crop(12, 151, 45)
    canvas.paste(_c12, (20, 13), _c12)
except Exception:
    pass
layout["7.57_Wy"] = [20, 13, 171, 58]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_05_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-8/13_text_St.png
try:
    _c13 = get_crop(13, 28, 28)
    canvas.paste(_c13, (1188, 233), _c13)
except Exception:
    pass
layout["St"] = [1188, 233, 1216, 261]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_05_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-8/14_text_St.png
try:
    _c14 = get_crop(14, 27, 30)
    canvas.paste(_c14, (217, 365), _c14)
except Exception:
    pass
layout["St"] = [217, 365, 244, 395]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_05_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-8/15_text_St.png
try:
    _c15 = get_crop(15, 30, 27)
    canvas.paste(_c15, (670, 509), _c15)
except Exception:
    pass
layout["St"] = [670, 509, 700, 536]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_05_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-8/16_text_St.png
try:
    _c16 = get_crop(16, 29, 27)
    canvas.paste(_c16, (1089, 527), _c16)
except Exception:
    pass
layout["St"] = [1089, 527, 1118, 554]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_05_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-8/17_text_St.png
try:
    _c17 = get_crop(17, 29, 27)
    canvas.paste(_c17, (835, 571), _c17)
except Exception:
    pass
layout["St"] = [835, 571, 864, 598]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_05_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-8/18_text_495.png
try:
    _c18 = get_crop(18, 41, 25)
    canvas.paste(_c18, (448, 659), _c18)
except Exception:
    pass
layout["495"] = [448, 659, 489, 684]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_05_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-8/19_text_St.png
try:
    _c19 = get_crop(19, 29, 27)
    canvas.paste(_c19, (583, 675), _c19)
except Exception:
    pass
layout["St"] = [583, 675, 612, 702]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_05_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-8/20_text_St.png
try:
    _c20 = get_crop(20, 27, 30)
    canvas.paste(_c20, (957, 668), _c20)
except Exception:
    pass
layout["St"] = [957, 668, 984, 698]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_05_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-8/21_text_09.png
try:
    _c21 = get_crop(21, 21, 23)
    canvas.paste(_c21, (824, 741), _c21)
except Exception:
    pass
layout["09"] = [824, 741, 845, 764]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_05_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-8/22_text_Metropolitan_Opera.png
try:
    _c22 = get_crop(22, 570, 79)
    canvas.paste(_c22, (46, 858), _c22)
except Exception:
    pass
layout["Metropolitan_Opera"] = [46, 858, 616, 937]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_05_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-8/23_text_New_York_NY.png
try:
    _c23 = get_crop(23, 304, 57)
    canvas.paste(_c23, (42, 942), _c23)
except Exception:
    pass
layout["New_York,_NY"] = [42, 942, 346, 999]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_05_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-8/24_text_Popular_events.png
try:
    _c24 = get_crop(24, 72, 72)
    canvas.paste(_c24, (408, 1297), _c24)
except Exception:
    pass
layout["Popular_events"] = [408, 1297, 480, 1369]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_05_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-8/25_text_Madame_Butterfly.png
try:
    _c25 = get_crop(25, 396, 533)
    canvas.paste(_c25, (1044, 1273), _c25)
except Exception:
    pass
layout["Madame_Butterfly"] = [1044, 1273, 1440, 1806]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_05_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-8/26_text_Sat.png
try:
    _c26 = get_crop(26, 91, 49)
    canvas.paste(_c26, (44, 1698), _c26)
except Exception:
    pass
layout["Sat,"] = [44, 1698, 135, 1747]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_05_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-8/27_text_27_8_PM.png
try:
    _c27 = get_crop(27, 174, 48)
    canvas.paste(_c27, (213, 1695), _c27)
except Exception:
    pass
layout["27,8_PM"] = [213, 1695, 387, 1743]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_05_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-8/28_text_Thu.png
try:
    _c28 = get_crop(28, 99, 51)
    canvas.paste(_c28, (540, 1695), _c28)
except Exception:
    pass
layout["Thu,"] = [540, 1695, 639, 1746]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_05_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-8/29_text_25_7.30_PM.png
try:
    _c29 = get_crop(29, 237, 50)
    canvas.paste(_c29, (719, 1695), _c29)
except Exception:
    pass
layout["25,7.30_PM"] = [719, 1695, 956, 1745]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_05_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-8/30_text_New_York.png
try:
    _c30 = get_crop(30, 205, 48)
    canvas.paste(_c30, (1043, 1688), _c30)
except Exception:
    pass
layout["New_York"] = [1043, 1688, 1248, 1736]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_05_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-8/31_text_Fri_Apr_26_8_PM.png
try:
    _c31 = get_crop(31, 396, 533)
    canvas.paste(_c31, (1044, 1273), _c31)
except Exception:
    pass
layout["Fri,_Apr_26,8_PM"] = [1044, 1273, 1440, 1806]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_05_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-8/32_text_Seating_charts.png
try:
    _c32 = get_crop(32, 391, 77)
    canvas.paste(_c32, (38, 1919), _c32)
except Exception:
    pass
layout["Seating_charts"] = [38, 1919, 429, 1996]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_05_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-8/33_text_El_Nino.png
try:
    _c33 = get_crop(33, 153, 52)
    canvas.paste(_c33, (46, 2422), _c33)
except Exception:
    pass
layout["El_Nino"] = [46, 2422, 199, 2474]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_05_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-8/34_text_American_Ballet_The..png
try:
    _c34 = get_crop(34, 462, 437)
    canvas.paste(_c34, (546, 2049), _c34)
except Exception:
    pass
layout["American_Ballet_The."] = [546, 2049, 1008, 2486]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_05_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-8/35_text_All_events.png
try:
    _c35 = get_crop(35, 256, 57)
    canvas.paste(_c35, (46, 2604), _c35)
except Exception:
    pass
layout["All_events"] = [46, 2604, 302, 2661]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_05_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-8/36_text_Tomorrow.png
try:
    _c36 = get_crop(36, 223, 55)
    canvas.paste(_c36, (43, 2749), _c36)
except Exception:
    pass
layout["Tomorrow"] = [43, 2749, 266, 2804]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_05_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-8/37_text_El_Nino.png
try:
    _c37 = get_crop(37, 157, 50)
    canvas.paste(_c37, (349, 2748), _c37)
except Exception:
    pass
layout["El_Nino"] = [349, 2748, 506, 2798]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_05_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-8/38_text_New_York.png
try:
    _c38 = get_crop(38, 1440, 241)
    canvas.paste(_c38, (0, 2687), _c38)
except Exception:
    pass
layout["New_York"] = [0, 2687, 1440, 2928]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_05_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-8/39_text_Tue_8_PM.png
try:
    _c39 = get_crop(39, 200, 50)
    canvas.paste(_c39, (42, 2822), _c39)
except Exception:
    pass
layout["Tue,_8_PM"] = [42, 2822, 242, 2872]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_05_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-8/40_text_S82.png
try:
    _c40 = get_crop(40, 89, 52)
    canvas.paste(_c40, (345, 2817), _c40)
except Exception:
    pass
layout["S82"] = [345, 2817, 434, 2869]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_05_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-8/41_text_Metropolitan_Opera.png
try:
    _c41 = get_crop(41, 1440, 241)
    canvas.paste(_c41, (0, 2687), _c41)
except Exception:
    pass
layout["Metropolitan_Opera"] = [0, 2687, 1440, 2928]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_05_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-8/42_text_New_York_NY.png
try:
    _c42 = get_crop(42, 269, 50)
    canvas.paste(_c42, (893, 2822), _c42)
except Exception:
    pass
layout["New_York,_NY"] = [893, 2822, 1162, 2872]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_05_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-8/43_text_Kenne.png
try:
    _c43 = get_crop(43, 76, 70)
    canvas.paste(_c43, (220, 53), _c43)
except Exception:
    pass
layout["Kenne"] = [220, 53, 296, 123]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_05_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-8/44_text_E1O6th.png
try:
    _c44 = get_crop(44, 91, 68)
    canvas.paste(_c44, (1184, 83), _c44)
except Exception:
    pass
layout["E1O6th"] = [1184, 83, 1275, 151]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_05_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-8/45_text_S1st_St.png
try:
    _c45 = get_crop(45, 144, 144)
    canvas.paste(_c45, (36, 84), _c45)
except Exception:
    pass
layout["~S1st_St"] = [36, 84, 180, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_05_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-8/46_text_Broadwa.png
try:
    _c46 = get_crop(46, 96, 81)
    canvas.paste(_c46, (1354, 533), _c46)
except Exception:
    pass
layout["Broadwa"] = [1354, 533, 1450, 614]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_05_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-8/47_clickable_Click_to_search_for_events_or_informatio.png
try:
    _c47 = get_crop(47, 1440, 704)
    canvas.paste(_c47, (0, 72), _c47)
except Exception:
    pass
layout["Click_to_search_for_event"] = [0, 72, 1440, 776]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_05_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-8/48_clickable_Tracking.png
try:
    _c48 = get_crop(48, 144, 144)
    canvas.paste(_c48, (1260, 84), _c48)
except Exception:
    pass
layout["Tracking"] = [1260, 84, 1404, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_05_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-8/49_clickable_Tracking.png
try:
    _c49 = get_crop(49, 72, 72)
    canvas.paste(_c49, (906, 1297), _c49)
except Exception:
    pass
layout["Tracking"] = [906, 1297, 978, 1369]
