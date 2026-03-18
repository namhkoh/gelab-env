# page_id: page_seatgeek_1e6c9e893d9e4bc99959744188677162_06
# screenshot: 2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-9.png
# step_index: 6/8
# task: Open SeatGeek. Search "Radio City Music Hall" and then add the venue to favorite. Who are the performers of the top recommended event?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw background and structural elements for the UI mockup.
# Assumes variables provided in environment:
# - canvas: PIL.Image (1440x2960 RGB)
# - draw: PIL.ImageDraw.Draw for canvas
# - font_sm, font_md, font_lg, font_xl available (not used for drawing text here)

# Colors
BG = (250, 250, 250)            # page background
STATUS_BG = (242, 242, 242)     # status bar background
DIVIDER = (226, 226, 226)       # thin divider lines
CARD_BG = (247, 249, 250)       # card backgrounds behind thumbnails
CARD_BG_ALT = (245, 246, 247)   # alternate card background
SECTION_BG = (255, 255, 255)    # white section backgrounds
SHADOW = (235, 235, 235)        # very light shadow

# Helper to draw a rounded rectangle with optional simple shadow
def draw_card(x0, y0, x1, y1, radius=24, fill=CARD_BG, shadow=True):
    if shadow:
        # simple shadow as a light rectangle offset downward-right
        sx0, sy0, sx1, sy1 = x0+4, y0+4, x1+4, y1+4
        draw.rounded_rectangle([sx0, sy0, sx1, sy1], radius=radius, fill=SHADOW)
    draw.rounded_rectangle([x0, y0, x1, y1], radius=radius, fill=fill)

# Fill overall background
draw.rectangle([0, 0, 1440, 2960], fill=BG)

# Status bar (top)
STATUS_H = 72
draw.rectangle([0, 0, 1440, STATUS_H], fill=STATUS_BG)
# subtle bottom divider for status bar
draw.line([0, STATUS_H-1, 1440, STATUS_H-1], fill=DIVIDER, width=1)

# Map area placeholder background (the real map will be pasted over this area).
# The live element covering this area is detected as clickable (pos y=72 height=704),
# so keep this neutral and unobtrusive.
MAP_TOP = STATUS_H
MAP_BOTTOM = MAP_TOP + 704
# Slightly off-white so the pasted map will sit on a clean base
draw.rectangle([0, MAP_TOP, 1440, MAP_BOTTOM], fill=(255,255,255))

# Title/header area below the map (venue title, small menu). Keep white.
TITLE_TOP = MAP_BOTTOM
TITLE_BOTTOM = TITLE_TOP + 160
draw.rectangle([0, TITLE_TOP, 1440, TITLE_BOTTOM], fill=SECTION_BG)
# divider under title
draw.line([48, TITLE_BOTTOM, 1440-48, TITLE_BOTTOM], fill=DIVIDER, width=1)

# "Popular events" section area -- white background
POP_TOP = TITLE_BOTTOM + 24
POP_BOTTOM = POP_TOP + 520
draw.rectangle([0, POP_TOP, 1440, POP_BOTTOM], fill=SECTION_BG)

# Divider above seating charts (between popular events and seating charts)
SEATING_DIV_Y = POP_BOTTOM - 40
draw.line([48, SEATING_DIV_Y, 1440-48, SEATING_DIV_Y], fill=DIVIDER, width=1)

# Draw rounded card backgrounds for the three event thumbnails (positions based on detected icons).
event_icon_boxes = [
    (48, 1273, 48+462, 1273+533),
    (546, 1273, 546+462, 1273+533),
    (1044, 1273, 1044+396, 1273+519)
]
for i, (x0, y0, x1, y1) in enumerate(event_icon_boxes):
    # Draw a slightly larger background card behind the actual thumbnail
    pad = 8
    draw_card(x0-pad, y0-pad, x1+pad, y1+pad, radius=22, fill=CARD_BG)

# Horizontal separator below popular events (subtle)
sep_y = POP_BOTTOM
draw.line([48, sep_y, 1440-48, sep_y], fill=DIVIDER, width=1)

# Seating charts section background
SEAT_TOP = sep_y + 24
SEAT_BOTTOM = SEAT_TOP + 420
draw.rectangle([0, SEAT_TOP, 1440, SEAT_BOTTOM], fill=SECTION_BG)

# Draw seating chart card backgrounds (three small rounded boxes)
seating_icon_boxes = [
    (48, 2049, 48+462, 2049+437),
    (546, 2049, 546+462, 2049+437),
    (1044, 2049, 1044+396, 2049+437)
]
for (x0, y0, x1, y1) in seating_icon_boxes:
    pad = 8
    draw_card(x0-pad, y0-pad, x1+pad, y1+pad, radius=16, fill=CARD_BG_ALT)

# Separator line under seating charts before "All events"
ALL_EVENTS_DIV_Y = SEAT_BOTTOM + 40
draw.line([48, ALL_EVENTS_DIV_Y, 1440-48, ALL_EVENTS_DIV_Y], fill=DIVIDER, width=1)

# "All events" area background (keep white)
ALL_TOP = ALL_EVENTS_DIV_Y + 24
# Draw a subtle full-width band where the upcoming list sits
draw.rectangle([0, ALL_TOP, 1440, 2960], fill=SECTION_BG)

# Additional subtle section separators to mirror UI structure
# Small thin lines separating list rows area (three sample lines)
list_start = ALL_TOP + 40
for i in range(4):
    y = list_start + i * 160
    draw.line([48, y, 1440-48, y], fill=(240,240,240), width=1)

# Final subtle bottom divider
draw.line([0, 2959, 1440, 2959], fill=DIVIDER, width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_06_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-9/00_icon_Jacob_Collier_with_Ki.png
try:
    _c0 = get_crop(0, 462, 437)
    canvas.paste(_c0, (546, 2049), _c0)
except Exception:
    pass
layout["Jacob_Collier_with_Ki_"] = [546, 2049, 1008, 2486]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_06_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-9/01_icon_Seating_charts.png
try:
    _c1 = get_crop(1, 462, 437)
    canvas.paste(_c1, (48, 2049), _c1)
except Exception:
    pass
layout["Seating_charts"] = [48, 2049, 510, 2486]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_06_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-9/02_icon_S186.png
try:
    _c2 = get_crop(2, 462, 533)
    canvas.paste(_c2, (48, 1273), _c2)
except Exception:
    pass
layout["S186+"] = [48, 1273, 510, 1806]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_06_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-9/03_icon_S186.png
try:
    _c3 = get_crop(3, 462, 533)
    canvas.paste(_c3, (546, 1273), _c3)
except Exception:
    pass
layout["S186+"] = [546, 1273, 1008, 1806]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_06_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-9/04_icon_The_Black_Crowes.png
try:
    _c4 = get_crop(4, 396, 437)
    canvas.paste(_c4, (1044, 2049), _c4)
except Exception:
    pass
layout["The_Black_Crowes"] = [1044, 2049, 1440, 2486]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_06_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-9/05_icon_S134.png
try:
    _c5 = get_crop(5, 396, 519)
    canvas.paste(_c5, (1044, 1273), _c5)
except Exception:
    pass
layout["S134+"] = [1044, 1273, 1440, 1792]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_06_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-9/06_icon_43rd.png
try:
    _c6 = get_crop(6, 144, 144)
    canvas.paste(_c6, (36, 84), _c6)
except Exception:
    pass
layout["43rd'"] = [36, 84, 180, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_06_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-9/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 45, 62)
    canvas.paste(_c7, (1327, 3), _c7)
except Exception:
    pass
layout["icon_7"] = [1327, 3, 1372, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_06_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-9/08_icon_495.png
try:
    _c8 = get_crop(8, 204, 174)
    canvas.paste(_c8, (1236, 806), _c8)
except Exception:
    pass
layout["495"] = [1236, 806, 1440, 980]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_06_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-9/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 43, 63)
    canvas.paste(_c9, (1158, 5), _c9)
except Exception:
    pass
layout["icon_9"] = [1158, 5, 1201, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_06_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-9/10_icon_The_Black_Crowes.png
try:
    _c10 = get_crop(10, 396, 437)
    canvas.paste(_c10, (1044, 2049), _c10)
except Exception:
    pass
layout["The_Black_Crowes"] = [1044, 2049, 1440, 2486]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_06_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-9/11_text_8.33_my.png
try:
    _c11 = get_crop(11, 156, 52)
    canvas.paste(_c11, (16, 9), _c11)
except Exception:
    pass
layout["8.33_my"] = [16, 9, 172, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_06_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-9/12_text_St.png
try:
    _c12 = get_crop(12, 27, 27)
    canvas.paste(_c12, (617, 261), _c12)
except Exception:
    pass
layout["St"] = [617, 261, 644, 288]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_06_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-9/13_text_St.png
try:
    _c13 = get_crop(13, 29, 27)
    canvas.paste(_c13, (1036, 280), _c13)
except Exception:
    pass
layout["St"] = [1036, 280, 1065, 307]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_06_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-9/14_text_St.png
try:
    _c14 = get_crop(14, 30, 27)
    canvas.paste(_c14, (781, 324), _c14)
except Exception:
    pass
layout["St"] = [781, 324, 811, 351]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_06_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-9/15_text_495.png
try:
    _c15 = get_crop(15, 41, 25)
    canvas.paste(_c15, (395, 411), _c15)
except Exception:
    pass
layout["495"] = [395, 411, 436, 436]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_06_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-9/16_text_St.png
try:
    _c16 = get_crop(16, 28, 27)
    canvas.paste(_c16, (529, 428), _c16)
except Exception:
    pass
layout["St"] = [529, 428, 557, 455]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_06_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-9/17_text_St.png
try:
    _c17 = get_crop(17, 27, 29)
    canvas.paste(_c17, (904, 421), _c17)
except Exception:
    pass
layout["St"] = [904, 421, 931, 450]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_06_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-9/18_text_25A.png
try:
    _c18 = get_crop(18, 44, 25)
    canvas.paste(_c18, (1320, 548), _c18)
except Exception:
    pass
layout["(25A"] = [1320, 548, 1364, 573]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_06_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-9/19_text_Ave.png
try:
    _c19 = get_crop(19, 46, 30)
    canvas.paste(_c19, (1385, 714), _c19)
except Exception:
    pass
layout["Ave"] = [1385, 714, 1431, 744]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_06_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-9/20_text_495.png
try:
    _c20 = get_crop(20, 37, 19)
    canvas.paste(_c20, (1173, 752), _c20)
except Exception:
    pass
layout["495"] = [1173, 752, 1210, 771]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_06_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-9/21_text_Radio_City_Music_Hall.png
try:
    _c21 = get_crop(21, 72, 72)
    canvas.paste(_c21, (408, 1297), _c21)
except Exception:
    pass
layout["Radio_City_Music_Hall"] = [408, 1297, 480, 1369]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_06_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-9/22_text_New_York_NY.png
try:
    _c22 = get_crop(22, 304, 57)
    canvas.paste(_c22, (42, 942), _c22)
except Exception:
    pass
layout["New_York,_NY"] = [42, 942, 346, 999]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_06_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-9/23_text_Popular_events.png
try:
    _c23 = get_crop(23, 72, 72)
    canvas.paste(_c23, (408, 1297), _c23)
except Exception:
    pass
layout["Popular_events"] = [408, 1297, 480, 1369]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_06_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-9/24_text_Laufey_with_Wasia.png
try:
    _c24 = get_crop(24, 462, 533)
    canvas.paste(_c24, (48, 1273), _c24)
except Exception:
    pass
layout["Laufey_with_Wasia"] = [48, 1273, 510, 1806]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_06_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-9/25_text_Laufey_with_Wasia.png
try:
    _c25 = get_crop(25, 462, 533)
    canvas.paste(_c25, (546, 1273), _c25)
except Exception:
    pass
layout["Laufey_with_Wasia"] = [546, 1273, 1008, 1806]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_06_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-9/26_text_The_Black_Crowes.png
try:
    _c26 = get_crop(26, 396, 519)
    canvas.paste(_c26, (1044, 1273), _c26)
except Exception:
    pass
layout["The_Black_Crowes"] = [1044, 1273, 1440, 1792]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_06_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-9/27_text_Project.png
try:
    _c27 = get_crop(27, 165, 55)
    canvas.paste(_c27, (42, 1688), _c27)
except Exception:
    pass
layout["Project"] = [42, 1688, 207, 1743]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_06_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-9/28_text_Project.png
try:
    _c28 = get_crop(28, 163, 57)
    canvas.paste(_c28, (541, 1686), _c28)
except Exception:
    pass
layout["Project"] = [541, 1686, 704, 1743]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_06_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-9/29_text_Sat_Apr_27_8_PM.png
try:
    _c29 = get_crop(29, 396, 519)
    canvas.paste(_c29, (1044, 1273), _c29)
except Exception:
    pass
layout["Sat,_Apr_27,8_PM"] = [1044, 1273, 1440, 1792]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_06_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-9/30_text_Sat.png
try:
    _c30 = get_crop(30, 87, 45)
    canvas.paste(_c30, (45, 1759), _c30)
except Exception:
    pass
layout["Sat,"] = [45, 1759, 132, 1804]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_06_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-9/31_text_4_8_PM.png
try:
    _c31 = get_crop(31, 154, 45)
    canvas.paste(_c31, (230, 1759), _c31)
except Exception:
    pass
layout["4,8_PM"] = [230, 1759, 384, 1804]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_06_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-9/32_text_Fri.png
try:
    _c32 = get_crop(32, 69, 45)
    canvas.paste(_c32, (542, 1759), _c32)
except Exception:
    pass
layout["Fri,"] = [542, 1759, 611, 1804]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_06_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-9/33_text_3_8_PM.png
try:
    _c33 = get_crop(33, 154, 45)
    canvas.paste(_c33, (706, 1759), _c33)
except Exception:
    pass
layout["3,8_PM"] = [706, 1759, 860, 1804]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_06_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-9/34_text_Seating_charts.png
try:
    _c34 = get_crop(34, 390, 76)
    canvas.paste(_c34, (39, 1920), _c34)
except Exception:
    pass
layout["Seating_charts"] = [39, 1920, 429, 1996]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_06_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-9/35_text_girl_in_red.png
try:
    _c35 = get_crop(35, 209, 61)
    canvas.paste(_c35, (41, 2420), _c35)
except Exception:
    pass
layout["girl_in_red"] = [41, 2420, 250, 2481]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_06_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-9/36_text_Jacob_Collier_with_Ki.png
try:
    _c36 = get_crop(36, 462, 437)
    canvas.paste(_c36, (546, 2049), _c36)
except Exception:
    pass
layout["Jacob_Collier_with_Ki_"] = [546, 2049, 1008, 2486]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_06_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-9/37_text_All_events.png
try:
    _c37 = get_crop(37, 258, 60)
    canvas.paste(_c37, (45, 2603), _c37)
except Exception:
    pass
layout["All_events"] = [45, 2603, 303, 2663]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_06_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-9/38_text_Apr_24.png
try:
    _c38 = get_crop(38, 151, 55)
    canvas.paste(_c38, (44, 2747), _c38)
except Exception:
    pass
layout["Apr_24"] = [44, 2747, 195, 2802]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_06_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-9/39_text_girl_in_red.png
try:
    _c39 = get_crop(39, 209, 57)
    canvas.paste(_c39, (345, 2747), _c39)
except Exception:
    pass
layout["girl_in_red"] = [345, 2747, 554, 2804]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_06_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-9/40_text_Wed.png
try:
    _c40 = get_crop(40, 103, 50)
    canvas.paste(_c40, (44, 2819), _c40)
except Exception:
    pass
layout["Wed,"] = [44, 2819, 147, 2869]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_06_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-9/41_text_8_PM.png
try:
    _c41 = get_crop(41, 104, 43)
    canvas.paste(_c41, (158, 2822), _c41)
except Exception:
    pass
layout["8_PM"] = [158, 2822, 262, 2865]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_06_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-9/42_text_S22.png
try:
    _c42 = get_crop(42, 89, 52)
    canvas.paste(_c42, (345, 2817), _c42)
except Exception:
    pass
layout["S22"] = [345, 2817, 434, 2869]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_06_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-9/43_text_Radio_City_Music_Hall.png
try:
    _c43 = get_crop(43, 1440, 241)
    canvas.paste(_c43, (0, 2687), _c43)
except Exception:
    pass
layout["Radio_City_Music_Hall"] = [0, 2687, 1440, 2928]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_06_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-9/44_text_New_York_NY.png
try:
    _c44 = get_crop(44, 266, 50)
    canvas.paste(_c44, (919, 2822), _c44)
except Exception:
    pass
layout["New_York,_NY"] = [919, 2822, 1185, 2872]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_06_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-9/45_text_Broadway.png
try:
    _c45 = get_crop(45, 107, 91)
    canvas.paste(_c45, (1300, 285), _c45)
except Exception:
    pass
layout["Broadway"] = [1300, 285, 1407, 376]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_06_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-9/46_text_W_14th.png
try:
    _c46 = get_crop(46, 85, 66)
    canvas.paste(_c46, (351, 719), _c46)
except Exception:
    pass
layout["W_14th"] = [351, 719, 436, 785]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_06_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-9/47_clickable_Click_to_open_the_location_of_Radio_City.png
try:
    _c47 = get_crop(47, 1440, 704)
    canvas.paste(_c47, (0, 72), _c47)
except Exception:
    pass
layout["Click_to_open_the_locatio"] = [0, 72, 1440, 776]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_06_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-9/48_clickable_Tracking.png
try:
    _c48 = get_crop(48, 144, 144)
    canvas.paste(_c48, (1260, 84), _c48)
except Exception:
    pass
layout["Tracking"] = [1260, 84, 1404, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_06_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-9/49_clickable_Tracking.png
try:
    _c49 = get_crop(49, 72, 72)
    canvas.paste(_c49, (906, 1297), _c49)
except Exception:
    pass
layout["Tracking"] = [906, 1297, 978, 1369]
