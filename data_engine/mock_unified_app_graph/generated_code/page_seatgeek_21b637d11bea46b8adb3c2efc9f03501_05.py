# page_id: page_seatgeek_21b637d11bea46b8adb3c2efc9f03501_05
# screenshot: 2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-8.png
# step_index: 5/10
# task: Open SeatGeek and find the soonest upcoming NBA game in New York with "Nets", record the cheapest price in google keep notes.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw background and structural UI elements for the provided canvas.
# Available variables: canvas (PIL Image), draw (ImageDraw), font_sm, font_md, font_lg, font_xl

width, height = canvas.size

# Colors
bg_offwhite = (249, 249, 249)      # page background
status_bar_color = (10, 10, 10)    # dark status bar
hero_dark_1 = (28, 28, 28)         # hero image placeholder shades
hero_dark_2 = (22, 22, 22)
hero_dark_3 = (18, 18, 18)
card_white = (255, 255, 255)
panel_light = (244, 244, 246)      # very light gray panels
separator = (224, 224, 224)        # subtle separators
shadow = (230, 230, 230)

# Fill overall background
draw.rectangle([(0, 0), (width, height)], fill=bg_offwhite)

# Status bar (top area)
status_h = 96  # slightly taller to accommodate various device indicators
draw.rectangle([(0, 0), (width, status_h)], fill=status_bar_color)

# Hero image placeholder (dark area under status bar)
hero_top = status_h
hero_bottom = 560
# layered rectangles to simulate a subtle vertical gradient / photo placeholder
draw.rectangle([(0, hero_top), (width, hero_top + int((hero_bottom - hero_top) * 0.33))], fill=hero_dark_1)
draw.rectangle([(0, hero_top + int((hero_bottom - hero_top) * 0.33)), (width, hero_top + int((hero_bottom - hero_top) * 0.66))], fill=hero_dark_2)
draw.rectangle([(0, hero_top + int((hero_bottom - hero_top) * 0.66)), (width, hero_bottom)], fill=hero_dark_3)

# Hero bottom overlay: slight fade to white for smooth transition to content
fade_top = hero_bottom - 40
fade_steps = 8
for i in range(fade_steps):
    # compute color interpolating from hero_dark_3 to bg_offwhite
    t = (i + 1) / float(fade_steps + 1)
    r = int(hero_dark_3[0] * (1 - t) + bg_offwhite[0] * t)
    g = int(hero_dark_3[1] * (1 - t) + bg_offwhite[1] * t)
    b = int(hero_dark_3[2] * (1 - t) + bg_offwhite[2] * t)
    y0 = fade_top + int((i / fade_steps) * 40)
    y1 = fade_top + int(((i + 1) / fade_steps) * 40)
    draw.rectangle([(0, y0), (width, y1)], fill=(r, g, b))

# Main title card (rounded white card overlapping hero)
title_card_left = 36
title_card_right = width - 36
title_card_top = hero_bottom - 36
title_card_bottom = 920
radius = 20

# subtle shadow behind the card
shadow_rect = (title_card_left + 6, title_card_top + 10, title_card_right + 6, title_card_bottom + 10)
draw.rounded_rectangle(shadow_rect, radius=radius, fill=shadow)

# white card
draw.rounded_rectangle((title_card_left, title_card_top, title_card_right, title_card_bottom),
                       radius=radius, fill=card_white)

# Divider under title card separating it from little guarantee row
divider_y = title_card_bottom + 6
draw.line([(title_card_left, divider_y), (title_card_right, divider_y)], fill=separator, width=1)

# "Protected by our Buyer Guarantee" panel background area (light panel)
protected_top = divider_y + 20
protected_h = 126
protected_left = 0
protected_right = width
protected_bottom = protected_top + protected_h
draw.rectangle([(protected_left, protected_top), (protected_right, protected_bottom)], fill=card_white)

# subtle horizontal separators above and below the protected panel
draw.line([(protected_left + 24, protected_top), (protected_right - 24, protected_top)], fill=separator, width=1)
draw.line([(protected_left + 24, protected_bottom), (protected_right - 24, protected_bottom)], fill=separator, width=1)

# Section container background (main list area) - keep it white
list_start = protected_bottom + 30
list_end = height - 120
list_pad_h = 24
draw.rectangle([(0, list_start - list_pad_h), (width, list_end + list_pad_h)], fill=bg_offwhite)

# Section header card background for "New York, NY" area (just background strip)
section_header_top = list_start
section_header_h = 120
draw.rectangle([(36, section_header_top), (width - 36, section_header_top + section_header_h)], fill=bg_offwhite)

# Draw separators and row card backgrounds for each detected row area.
# Detected row top positions (from detection): 1279, 1572, 1865, 2158, 2596 (each height 293)
row_positions = [1279, 1572, 1865, 2158, 2596]
row_height = 293
row_left = 24
row_right = width - 24
row_radius = 18

for idx, rtop in enumerate(row_positions):
    rbottom = rtop + row_height
    # faint background to separate rows from page (very subtle)
    # use alternate light panel for grouped rows (but keep minimal to avoid drawing content)
    draw.rectangle([(row_left, rtop + 10), (row_right, rbottom - 12)], fill=card_white)
    # very subtle top divider line
    draw.line([(row_left + 10, rtop + 10), (row_right - 10, rtop + 10)], fill=separator, width=1)
    # very subtle bottom divider
    draw.line([(row_left + 10, rbottom - 12), (row_right - 10, rbottom - 12)], fill=separator, width=1)
    # light shadow at the bottom of each row block
    draw.rectangle([(row_left + 8, rbottom - 6), (row_right - 8, rbottom - 4)], fill=(245, 245, 247))

# Section title "All Games" area background strip (below the first group)
all_games_top = 2480
all_games_height = 120
draw.rectangle([(36, all_games_top), (width - 36, all_games_top + all_games_height)], fill=bg_offwhite)
draw.line([(36, all_games_top + all_games_height), (width - 36, all_games_top + all_games_height)], fill=separator, width=1)

# Bottom safe area (subtle)
bottom_h = 80
draw.rectangle([(0, height - bottom_h), (width, height)], fill=bg_offwhite)
draw.line([(36, height - bottom_h), (width - 36, height - bottom_h)], fill=separator, width=1)

# Thin global separators between major sections
# between title card and protected panel
draw.line([(36, title_card_bottom + 12), (width - 36, title_card_bottom + 12)], fill=separator, width=1)
# between protected panel and list start
draw.line([(36, protected_bottom + 12), (width - 36, protected_bottom + 12)], fill=separator, width=1)

# End of structural/background drawing.
# (Actual icons, texts and interactive elements will be layered on top later.)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_05_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-8/00_icon_Track_this_performer.png
try:
    _c0 = get_crop(0, 144, 144)
    canvas.paste(_c0, (1104, 84), _c0)
except Exception:
    pass
layout["Track_this_performer"] = [1104, 84, 1248, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_05_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-8/01_icon_Share_this_performer.png
try:
    _c1 = get_crop(1, 144, 144)
    canvas.paste(_c1, (1260, 84), _c1)
except Exception:
    pass
layout["Share_this_performer"] = [1260, 84, 1404, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_05_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-8/02_icon_21.png
try:
    _c2 = get_crop(2, 1440, 293)
    canvas.paste(_c2, (0, 2596), _c2)
except Exception:
    pass
layout["21"] = [0, 2596, 1440, 2889]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_05_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-8/03_icon_31.png
try:
    _c3 = get_crop(3, 1440, 293)
    canvas.paste(_c3, (0, 1865), _c3)
except Exception:
    pass
layout["31"] = [0, 1865, 1440, 2158]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_05_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-8/04_icon_03.png
try:
    _c4 = get_crop(4, 1440, 293)
    canvas.paste(_c4, (0, 2158), _c4)
except Exception:
    pass
layout["03"] = [0, 2158, 1440, 2451]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_05_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-8/05_icon_29.png
try:
    _c5 = get_crop(5, 1440, 293)
    canvas.paste(_c5, (0, 1572), _c5)
except Exception:
    pass
layout["29"] = [0, 1572, 1440, 1865]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_05_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-8/06_icon_Barclays_Center.png
try:
    _c6 = get_crop(6, 1440, 293)
    canvas.paste(_c6, (0, 1572), _c6)
except Exception:
    pass
layout["Barclays_Center"] = [0, 1572, 1440, 1865]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_05_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-8/07_icon_23.png
try:
    _c7 = get_crop(7, 1440, 293)
    canvas.paste(_c7, (0, 1279), _c7)
except Exception:
    pass
layout["23"] = [0, 1279, 1440, 1572]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_05_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-8/08_icon_Madison_Square_Garden.png
try:
    _c8 = get_crop(8, 1440, 293)
    canvas.paste(_c8, (0, 1279), _c8)
except Exception:
    pass
layout["Madison_Square_Garden"] = [0, 1279, 1440, 1572]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_05_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-8/09_icon_Protected_by_our_Buyer_Guarantee.png
try:
    _c9 = get_crop(9, 1440, 126)
    canvas.paste(_c9, (0, 933), _c9)
except Exception:
    pass
layout["Protected_by_our_Buyer_Gu"] = [0, 933, 1440, 1059]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_05_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-8/10_icon_6.37.png
try:
    _c10 = get_crop(10, 144, 144)
    canvas.paste(_c10, (36, 84), _c10)
except Exception:
    pass
layout["6.37"] = [36, 84, 180, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_05_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-8/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 111, 64)
    canvas.paste(_c11, (1203, 0), _c11)
except Exception:
    pass
layout["icon_11"] = [1203, 0, 1314, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_05_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-8/12_icon_Barclays_Center.png
try:
    _c12 = get_crop(12, 1440, 293)
    canvas.paste(_c12, (0, 1865), _c12)
except Exception:
    pass
layout["Barclays_Center"] = [0, 1865, 1440, 2158]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_05_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-8/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 62, 67)
    canvas.paste(_c13, (1140, 0), _c13)
except Exception:
    pass
layout["icon_13"] = [1140, 0, 1202, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_05_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-8/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 68, 60)
    canvas.paste(_c14, (1320, 0), _c14)
except Exception:
    pass
layout["icon_14"] = [1320, 0, 1388, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_05_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-8/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 90, 105)
    canvas.paste(_c15, (1303, 951), _c15)
except Exception:
    pass
layout["icon_15"] = [1303, 951, 1393, 1056]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_05_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-8/16_icon_6.37.png
try:
    _c16 = get_crop(16, 105, 62)
    canvas.paste(_c16, (14, 4), _c16)
except Exception:
    pass
layout["6.37"] = [14, 4, 119, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_05_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-8/17_icon_GEET.png
try:
    _c17 = get_crop(17, 60, 64)
    canvas.paste(_c17, (176, 4), _c17)
except Exception:
    pass
layout["GEET"] = [176, 4, 236, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_05_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-8/18_icon_Milwaukee_WI.png
try:
    _c18 = get_crop(18, 1440, 293)
    canvas.paste(_c18, (0, 2596), _c18)
except Exception:
    pass
layout["Milwaukee,_WI"] = [0, 2596, 1440, 2889]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_05_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-8/19_icon_GEET.png
try:
    _c19 = get_crop(19, 58, 61)
    canvas.paste(_c19, (115, 5), _c19)
except Exception:
    pass
layout["GEET"] = [115, 5, 173, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_05_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-8/20_icon_Los_Angeles_Lakers_at_Brooklyn_Nets.png
try:
    _c20 = get_crop(20, 1440, 293)
    canvas.paste(_c20, (0, 1865), _c20)
except Exception:
    pass
layout["Los_Angeles_Lakers_at_Bro"] = [0, 1865, 1440, 2158]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_05_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-8/21_icon_7.30_PM_._Barclays_Center.png
try:
    _c21 = get_crop(21, 1440, 293)
    canvas.paste(_c21, (0, 2158), _c21)
except Exception:
    pass
layout["7.30_PM_._Barclays_Center"] = [0, 2158, 1440, 2451]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_05_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-8/22_text_TOMER.png
try:
    _c22 = get_crop(22, 58, 28)
    canvas.paste(_c22, (492, 41), _c22)
except Exception:
    pass
layout["TOMER"] = [492, 41, 550, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_05_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-8/23_text_BARCLAYS_CENTER.png
try:
    _c23 = get_crop(23, 203, 32)
    canvas.paste(_c23, (587, 37), _c23)
except Exception:
    pass
layout["BARCLAYS_CENTER"] = [587, 37, 790, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_05_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-8/24_text_New_York_NY.png
try:
    _c24 = get_crop(24, 350, 60)
    canvas.paste(_c24, (57, 1179), _c24)
except Exception:
    pass
layout["New_York,_NY"] = [57, 1179, 407, 1239]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_05_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-8/25_text_Rrooklvn.png
try:
    _c25 = get_crop(25, 201, 29)
    canvas.paste(_c25, (319, 2930), _c25)
except Exception:
    pass
layout["Rrooklvn"] = [319, 2930, 520, 2959]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_05_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-8/26_text_Netc_at_New.png
try:
    _c26 = get_crop(26, 275, 29)
    canvas.paste(_c26, (529, 2930), _c26)
except Exception:
    pass
layout["Netc_at_New"] = [529, 2930, 804, 2959]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_05_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-8/27_text_Vork.png
try:
    _c27 = get_crop(27, 101, 29)
    canvas.paste(_c27, (814, 2930), _c27)
except Exception:
    pass
layout["Vork"] = [814, 2930, 915, 2959]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_05_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-8/28_text_Knicke.png
try:
    _c28 = get_crop(28, 154, 32)
    canvas.paste(_c28, (925, 2927), _c28)
except Exception:
    pass
layout["Knicke"] = [925, 2927, 1079, 2959]
