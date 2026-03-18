# page_id: page_seatgeek_1e6c9e893d9e4bc99959744188677162_05
# screenshot: 2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-8.png
# step_index: 5/8
# task: Open SeatGeek. Search "Radio City Music Hall" and then add the venue to favorite. Who are the performers of the top recommended event?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw background and structural UI elements for the canvas using PIL's ImageDraw API.
# Available variables: canvas (PIL Image 1440x2960), draw (ImageDraw), font_sm, font_md, font_lg, font_xl

# Overall background (match screenshot dominant very light color)
draw.rectangle([0, 0, 1440, 2960], fill=(250, 250, 250))

# Status bar area at top (~72px tall)
status_h = 72
draw.rectangle([0, 0, 1440, status_h], fill=(235, 235, 235))
# thin bottom divider under status bar
draw.line([(0, status_h), (1440, status_h)], fill=(215, 215, 215), width=1)

# Map area exists at (0,72)->height 704 per detection; do NOT draw map content.
map_top = status_h
map_h = 704
map_bottom = map_top + map_h
# Add a subtle divider under the map (separates map from header)
draw.line([(0, map_bottom), (1440, map_bottom)], fill=(220, 220, 220), width=1)

# Header card (white background) directly beneath map (title + location area)
header_top = map_bottom + 12
header_bottom = header_top + 140
draw.rounded_rectangle([24, header_top, 1440 - 24, header_bottom], radius=6, fill=(255, 255, 255), outline=None)
# subtle top shadow line for header card
draw.line([(24, header_top), (1440 - 24, header_top)], fill=(245, 245, 245), width=1)
# subtle bottom divider under header
draw.line([(24, header_bottom), (1440 - 24, header_bottom)], fill=(230, 230, 230), width=1)

# Popular events section background block (white)
popular_top = header_bottom + 24
popular_bottom = popular_top + 360
draw.rectangle([0, popular_top, 1440, popular_bottom], fill=(250, 250, 250))  # keep slight contrast with cards
# inner white card band for the heading and thumbnail area
draw.rectangle([24, popular_top, 1440 - 24, popular_bottom - 12], fill=(255, 255, 255))
# heading divider line below heading area (approx)
draw.line([(24 + 24, popular_top + 68), (1440 - 24 - 24, popular_top + 68)], fill=(235, 235, 235), width=1)

# Draw rounded card backgrounds behind the popular event thumbnails (do NOT draw the event images themselves).
thumbs = [
    (48, 1273, 48 + 462, 1273 + 533),   # left
    (546, 1273, 546 + 462, 1273 + 533), # middle
    (1044, 1273, 1044 + 396, 1273 + 519) # right
]
# We'll draw subtle rounded white cards with light shadow/border behind each thumbnail area.
for (l, t, r, b) in thumbs:
    pad = 12
    rect = [l - pad, t - pad, r + pad, b + pad]
    # card background
    draw.rounded_rectangle(rect, radius=20, fill=(255, 255, 255))
    # light border to separate from page
    draw.rounded_rectangle(rect, radius=20, outline=(235, 235, 235), width=1)

# Seating charts section background + heading divider
seating_top = 1960
seating_bottom = seating_top + 520
# big white band
draw.rectangle([0, seating_top, 1440, seating_bottom], fill=(250, 250, 250))
draw.rectangle([24, seating_top + 12, 1440 - 24, seating_bottom - 24], fill=(255, 255, 255))
# heading divider line
draw.line([(24 + 24, seating_top + 72), (1440 - 24 - 24, seating_top + 72)], fill=(235, 235, 235), width=1)

# Seating chart small cards (rounded grey tiles) behind the thumbnails
seating_cards = [
    (48, 2049, 48 + 462, 2049 + 437),
    (546, 2049, 546 + 462, 2049 + 437),
    (1044, 2049, 1044 + 396, 2049 + 437)
]
for (l, t, r, b) in seating_cards:
    pad = 8
    rect = [l - pad, t - pad, r + pad, b + pad]
    # light grey tile background
    draw.rounded_rectangle(rect, radius=12, fill=(250, 250, 250), outline=(235, 235, 235))
    # inner soft rectangle to mimic seating tile area (still keep empty for pasted image)
    inner = [rect[0] + 8, rect[1] + 8, rect[2] - 8, rect[3] - 8]
    draw.rounded_rectangle(inner, radius=10, fill=(245, 245, 245))

# Separator line between seating charts and All events section
sep_y = seating_bottom + 12
draw.line([(24, sep_y), (1440 - 24, sep_y)], fill=(220, 220, 220), width=1)

# All events section background area at bottom
all_top = sep_y + 24
all_bottom = 2960 - 40
draw.rectangle([0, all_top, 1440, all_bottom], fill=(250, 250, 250))
draw.rectangle([24, all_top, 1440 - 24, all_bottom], fill=(255, 255, 255))

# small horizontal separators inside All events area (repeat separators to suggest list structure)
list_y = all_top + 120
for i in range(4):
    draw.line([(24 + 24, list_y + i * 140), (1440 - 24 - 24, list_y + i * 140)], fill=(240, 240, 240), width=1)

# Light right edge scroll indicator area (thin)
draw.rectangle([1432, map_bottom + 8, 1440, map_bottom + 2000], fill=(250, 250, 250))

# Final subtle footer divider
draw.line([(24, all_bottom - 12), (1440 - 24, all_bottom - 12)], fill=(230, 230, 230), width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_05_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-8/00_icon_Jacob_Collier_with_Ki.png
try:
    _c0 = get_crop(0, 462, 437)
    canvas.paste(_c0, (546, 2049), _c0)
except Exception:
    pass
layout["Jacob_Collier_with_Ki_"] = [546, 2049, 1008, 2486]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_05_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-8/01_icon_Seating_charts.png
try:
    _c1 = get_crop(1, 462, 437)
    canvas.paste(_c1, (48, 2049), _c1)
except Exception:
    pass
layout["Seating_charts"] = [48, 2049, 510, 2486]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_05_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-8/02_icon_S186.png
try:
    _c2 = get_crop(2, 462, 533)
    canvas.paste(_c2, (48, 1273), _c2)
except Exception:
    pass
layout["S186+"] = [48, 1273, 510, 1806]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_05_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-8/03_icon_S186.png
try:
    _c3 = get_crop(3, 462, 533)
    canvas.paste(_c3, (546, 1273), _c3)
except Exception:
    pass
layout["S186+"] = [546, 1273, 1008, 1806]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_05_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-8/04_icon_The_Black_Crowes.png
try:
    _c4 = get_crop(4, 396, 437)
    canvas.paste(_c4, (1044, 2049), _c4)
except Exception:
    pass
layout["The_Black_Crowes"] = [1044, 2049, 1440, 2486]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_05_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-8/05_icon_43rd.png
try:
    _c5 = get_crop(5, 144, 144)
    canvas.paste(_c5, (36, 84), _c5)
except Exception:
    pass
layout["43rd'"] = [36, 84, 180, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_05_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-8/06_icon_S134.png
try:
    _c6 = get_crop(6, 396, 519)
    canvas.paste(_c6, (1044, 1273), _c6)
except Exception:
    pass
layout["S134+"] = [1044, 1273, 1440, 1792]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_05_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-8/07_icon_495.png
try:
    _c7 = get_crop(7, 204, 174)
    canvas.paste(_c7, (1236, 806), _c7)
except Exception:
    pass
layout["495"] = [1236, 806, 1440, 980]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_05_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-8/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 46, 62)
    canvas.paste(_c8, (1327, 3), _c8)
except Exception:
    pass
layout["icon_8"] = [1327, 3, 1373, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_05_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-8/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 43, 62)
    canvas.paste(_c9, (1158, 6), _c9)
except Exception:
    pass
layout["icon_9"] = [1158, 6, 1201, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_05_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-8/10_icon_8.32_my.png
try:
    _c10 = get_crop(10, 144, 144)
    canvas.paste(_c10, (36, 84), _c10)
except Exception:
    pass
layout["8.32_my"] = [36, 84, 180, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_05_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-8/11_icon_The_Black_Crowes.png
try:
    _c11 = get_crop(11, 396, 437)
    canvas.paste(_c11, (1044, 2049), _c11)
except Exception:
    pass
layout["The_Black_Crowes"] = [1044, 2049, 1440, 2486]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_05_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-8/12_text_St.png
try:
    _c12 = get_crop(12, 27, 27)
    canvas.paste(_c12, (617, 261), _c12)
except Exception:
    pass
layout["St"] = [617, 261, 644, 288]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_05_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-8/13_text_St.png
try:
    _c13 = get_crop(13, 29, 27)
    canvas.paste(_c13, (1036, 280), _c13)
except Exception:
    pass
layout["St"] = [1036, 280, 1065, 307]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_05_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-8/14_text_St.png
try:
    _c14 = get_crop(14, 30, 27)
    canvas.paste(_c14, (781, 324), _c14)
except Exception:
    pass
layout["St"] = [781, 324, 811, 351]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_05_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-8/15_text_495.png
try:
    _c15 = get_crop(15, 41, 25)
    canvas.paste(_c15, (395, 411), _c15)
except Exception:
    pass
layout["495"] = [395, 411, 436, 436]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_05_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-8/16_text_St.png
try:
    _c16 = get_crop(16, 28, 27)
    canvas.paste(_c16, (529, 428), _c16)
except Exception:
    pass
layout["St"] = [529, 428, 557, 455]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_05_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-8/17_text_St.png
try:
    _c17 = get_crop(17, 27, 29)
    canvas.paste(_c17, (904, 421), _c17)
except Exception:
    pass
layout["St"] = [904, 421, 931, 450]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_05_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-8/18_text_25A.png
try:
    _c18 = get_crop(18, 44, 25)
    canvas.paste(_c18, (1320, 548), _c18)
except Exception:
    pass
layout["(25A"] = [1320, 548, 1364, 573]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_05_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-8/19_text_Ave.png
try:
    _c19 = get_crop(19, 46, 30)
    canvas.paste(_c19, (1385, 714), _c19)
except Exception:
    pass
layout["Ave"] = [1385, 714, 1431, 744]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_05_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-8/20_text_495.png
try:
    _c20 = get_crop(20, 37, 19)
    canvas.paste(_c20, (1173, 752), _c20)
except Exception:
    pass
layout["495"] = [1173, 752, 1210, 771]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_05_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-8/21_text_Radio_City_Music_Hall.png
try:
    _c21 = get_crop(21, 72, 72)
    canvas.paste(_c21, (408, 1297), _c21)
except Exception:
    pass
layout["Radio_City_Music_Hall"] = [408, 1297, 480, 1369]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_05_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-8/22_text_New_York_NY.png
try:
    _c22 = get_crop(22, 304, 57)
    canvas.paste(_c22, (42, 942), _c22)
except Exception:
    pass
layout["New_York,_NY"] = [42, 942, 346, 999]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_05_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-8/23_text_Popular_events.png
try:
    _c23 = get_crop(23, 72, 72)
    canvas.paste(_c23, (408, 1297), _c23)
except Exception:
    pass
layout["Popular_events"] = [408, 1297, 480, 1369]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_05_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-8/24_text_Laufey_with_Wasia.png
try:
    _c24 = get_crop(24, 462, 533)
    canvas.paste(_c24, (48, 1273), _c24)
except Exception:
    pass
layout["Laufey_with_Wasia"] = [48, 1273, 510, 1806]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_05_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-8/25_text_Laufey_with_Wasia.png
try:
    _c25 = get_crop(25, 462, 533)
    canvas.paste(_c25, (546, 1273), _c25)
except Exception:
    pass
layout["Laufey_with_Wasia"] = [546, 1273, 1008, 1806]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_05_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-8/26_text_The_Black_Crowes.png
try:
    _c26 = get_crop(26, 396, 519)
    canvas.paste(_c26, (1044, 1273), _c26)
except Exception:
    pass
layout["The_Black_Crowes"] = [1044, 1273, 1440, 1792]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_05_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-8/27_text_Project.png
try:
    _c27 = get_crop(27, 165, 55)
    canvas.paste(_c27, (42, 1688), _c27)
except Exception:
    pass
layout["Project"] = [42, 1688, 207, 1743]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_05_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-8/28_text_Project.png
try:
    _c28 = get_crop(28, 163, 57)
    canvas.paste(_c28, (541, 1686), _c28)
except Exception:
    pass
layout["Project"] = [541, 1686, 704, 1743]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_05_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-8/29_text_Sat_Apr_27_8_PM.png
try:
    _c29 = get_crop(29, 396, 519)
    canvas.paste(_c29, (1044, 1273), _c29)
except Exception:
    pass
layout["Sat,_Apr_27,8_PM"] = [1044, 1273, 1440, 1792]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_05_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-8/30_text_Sat.png
try:
    _c30 = get_crop(30, 87, 45)
    canvas.paste(_c30, (45, 1759), _c30)
except Exception:
    pass
layout["Sat,"] = [45, 1759, 132, 1804]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_05_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-8/31_text_4_8_PM.png
try:
    _c31 = get_crop(31, 154, 45)
    canvas.paste(_c31, (230, 1759), _c31)
except Exception:
    pass
layout["4,8_PM"] = [230, 1759, 384, 1804]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_05_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-8/32_text_Fri.png
try:
    _c32 = get_crop(32, 69, 45)
    canvas.paste(_c32, (542, 1759), _c32)
except Exception:
    pass
layout["Fri,"] = [542, 1759, 611, 1804]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_05_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-8/33_text_3_8_PM.png
try:
    _c33 = get_crop(33, 154, 45)
    canvas.paste(_c33, (706, 1759), _c33)
except Exception:
    pass
layout["3,8_PM"] = [706, 1759, 860, 1804]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_05_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-8/34_text_Seating_charts.png
try:
    _c34 = get_crop(34, 390, 76)
    canvas.paste(_c34, (39, 1920), _c34)
except Exception:
    pass
layout["Seating_charts"] = [39, 1920, 429, 1996]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_05_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-8/35_text_girl_in_red.png
try:
    _c35 = get_crop(35, 209, 61)
    canvas.paste(_c35, (41, 2420), _c35)
except Exception:
    pass
layout["girl_in_red"] = [41, 2420, 250, 2481]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_05_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-8/36_text_Jacob_Collier_with_Ki.png
try:
    _c36 = get_crop(36, 462, 437)
    canvas.paste(_c36, (546, 2049), _c36)
except Exception:
    pass
layout["Jacob_Collier_with_Ki_"] = [546, 2049, 1008, 2486]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_05_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-8/37_text_All_events.png
try:
    _c37 = get_crop(37, 258, 60)
    canvas.paste(_c37, (45, 2603), _c37)
except Exception:
    pass
layout["All_events"] = [45, 2603, 303, 2663]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_05_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-8/38_text_Apr_24.png
try:
    _c38 = get_crop(38, 151, 55)
    canvas.paste(_c38, (44, 2747), _c38)
except Exception:
    pass
layout["Apr_24"] = [44, 2747, 195, 2802]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_05_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-8/39_text_girl_in_red.png
try:
    _c39 = get_crop(39, 209, 57)
    canvas.paste(_c39, (345, 2747), _c39)
except Exception:
    pass
layout["girl_in_red"] = [345, 2747, 554, 2804]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_05_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-8/40_text_Wed.png
try:
    _c40 = get_crop(40, 103, 50)
    canvas.paste(_c40, (44, 2819), _c40)
except Exception:
    pass
layout["Wed,"] = [44, 2819, 147, 2869]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_05_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-8/41_text_8_PM.png
try:
    _c41 = get_crop(41, 104, 43)
    canvas.paste(_c41, (158, 2822), _c41)
except Exception:
    pass
layout["8_PM"] = [158, 2822, 262, 2865]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_05_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-8/42_text_S22.png
try:
    _c42 = get_crop(42, 89, 52)
    canvas.paste(_c42, (345, 2817), _c42)
except Exception:
    pass
layout["S22"] = [345, 2817, 434, 2869]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_05_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-8/43_text_Radio_City_Music_Hall.png
try:
    _c43 = get_crop(43, 1440, 241)
    canvas.paste(_c43, (0, 2687), _c43)
except Exception:
    pass
layout["Radio_City_Music_Hall"] = [0, 2687, 1440, 2928]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_05_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-8/44_text_New_York_NY.png
try:
    _c44 = get_crop(44, 266, 50)
    canvas.paste(_c44, (919, 2822), _c44)
except Exception:
    pass
layout["New_York,_NY"] = [919, 2822, 1185, 2872]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_05_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-8/45_text_Broadway.png
try:
    _c45 = get_crop(45, 106, 91)
    canvas.paste(_c45, (1301, 285), _c45)
except Exception:
    pass
layout["Broadway"] = [1301, 285, 1407, 376]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_05_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-8/46_text_W_14th.png
try:
    _c46 = get_crop(46, 85, 66)
    canvas.paste(_c46, (351, 719), _c46)
except Exception:
    pass
layout["W_14th"] = [351, 719, 436, 785]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_05_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-8/47_clickable_Click_to_view_the_location_of_Radio_City.png
try:
    _c47 = get_crop(47, 1440, 704)
    canvas.paste(_c47, (0, 72), _c47)
except Exception:
    pass
layout["Click_to_view_the_locatio"] = [0, 72, 1440, 776]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_05_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-8/48_clickable_Tracking.png
try:
    _c48 = get_crop(48, 144, 144)
    canvas.paste(_c48, (1260, 84), _c48)
except Exception:
    pass
layout["Tracking"] = [1260, 84, 1404, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_05_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-8/49_clickable_Tracking.png
try:
    _c49 = get_crop(49, 72, 72)
    canvas.paste(_c49, (906, 1297), _c49)
except Exception:
    pass
layout["Tracking"] = [906, 1297, 978, 1369]
