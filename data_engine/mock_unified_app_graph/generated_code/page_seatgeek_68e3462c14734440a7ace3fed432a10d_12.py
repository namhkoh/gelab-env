# page_id: page_seatgeek_68e3462c14734440a7ace3fed432a10d_12
# screenshot: 2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-15.png
# step_index: 12/13
# task: Open SeatGeek and change the current location to Los Angeles. Then find the first concert show and track its performer.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# canvas: PIL Image (1440x2960 RGB)
# draw: PIL ImageDraw object
# font_sm, font_md, font_lg, font_xl available

w, h = canvas.size

# Colors
bg_offwhite = (250, 250, 250)
hero_black = (10, 10, 10)
card_white = (255, 255, 255)
shadow_gray = (230, 230, 230)
divider_gray = (224, 224, 224)

# Fill overall background (dominant color)
draw.rectangle([(0, 0), (w, h)], fill=bg_offwhite)

# Status bar area (dark)
status_h = 96
draw.rectangle([(0, 0), (w, status_h)], fill=hero_black)

# Large hero/banner area at top (image background)
hero_bottom = 720
draw.rectangle([(0, 0), (w, hero_bottom)], fill=hero_black)

# Divider line under the hero area
divider_y = hero_bottom - 8
draw.line([(48, divider_y), (w - 48, divider_y)], fill=divider_gray, width=1)

# Main content card overlapping hero (white rounded rectangle with subtle shadow)
main_card_top = hero_bottom - 56
main_card_left = 24
main_card_right = w - 24
main_card_bottom = 1160
card_radius = 20

# shadow
shadow_offset = 8
draw.rounded_rectangle(
    [(main_card_left + shadow_offset, main_card_top + shadow_offset),
     (main_card_right + shadow_offset, main_card_bottom + shadow_offset)],
    radius=card_radius, fill=shadow_gray
)
# card
draw.rounded_rectangle(
    [(main_card_left, main_card_top), (main_card_right, main_card_bottom)],
    radius=card_radius, fill=card_white
)

# Thin divider under title area inside main card
title_divider_y = main_card_top + 140
draw.line([(main_card_left + 24, title_divider_y), (main_card_right - 24, title_divider_y)], fill=divider_gray, width=1)

# Section card: "Los Angeles, CA" event card background
event1_top = main_card_bottom + 40
event_card_h = 300
event_card_left = 24
event_card_right = w - 24
event1_bottom = event1_top + event_card_h
event_radius = 16

# subtle shadow behind event card
draw.rounded_rectangle(
    [(event_card_left + 6, event1_top + 6), (event_card_right + 6, event1_bottom + 6)],
    radius=event_radius, fill=shadow_gray
)
draw.rounded_rectangle(
    [(event_card_left, event1_top), (event_card_right, event1_bottom)],
    radius=event_radius, fill=card_white
)

# Separator line below first event card
sep_y1 = event1_bottom + 34
draw.line([(36, sep_y1), (w - 36, sep_y1)], fill=divider_gray, width=1)

# Section header for "All Shows" area (no text, just spacing background)
all_shows_top = sep_y1 + 36
# light background band to denote section
band_h = 96
draw.rectangle([(0, all_shows_top), (w, all_shows_top + band_h)], fill=bg_offwhite)

# Second event card under "All Shows"
event2_top = all_shows_top + band_h + 24
event2_bottom = event2_top + event_card_h
# shadow
draw.rounded_rectangle(
    [(event_card_left + 6, event2_top + 6), (event_card_right + 6, event2_bottom + 6)],
    radius=event_radius, fill=shadow_gray
)
draw.rounded_rectangle(
    [(event_card_left, event2_top), (event_card_right, event2_bottom)],
    radius=event_radius, fill=card_white
)

# Final subtle bottom divider
final_div_y = event2_bottom + 40
draw.line([(36, final_div_y), (w - 36, final_div_y)], fill=divider_gray, width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_12_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-15/00_icon_Track_this_performer.png
try:
    _c0 = get_crop(0, 144, 144)
    canvas.paste(_c0, (1104, 84), _c0)
except Exception:
    pass
layout["Track_this_performer"] = [1104, 84, 1248, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_12_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-15/01_icon_Share_this_performer.png
try:
    _c1 = get_crop(1, 144, 144)
    canvas.paste(_c1, (1260, 84), _c1)
except Exception:
    pass
layout["Share_this_performer"] = [1260, 84, 1404, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_12_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-15/02_icon_Hollywood_Bowl.png
try:
    _c2 = get_crop(2, 1440, 367)
    canvas.paste(_c2, (0, 1366), _c2)
except Exception:
    pass
layout["Hollywood_Bowl"] = [0, 1366, 1440, 1733]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_12_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-15/03_icon_11.png
try:
    _c3 = get_crop(3, 1440, 367)
    canvas.paste(_c3, (0, 1953), _c3)
except Exception:
    pass
layout["11"] = [0, 1953, 1440, 2320]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_12_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-15/04_icon_8331.png
try:
    _c4 = get_crop(4, 144, 144)
    canvas.paste(_c4, (36, 84), _c4)
except Exception:
    pass
layout["8331"] = [36, 84, 180, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_12_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-15/05_icon_11.png
try:
    _c5 = get_crop(5, 1440, 367)
    canvas.paste(_c5, (0, 1366), _c5)
except Exception:
    pass
layout["11"] = [0, 1366, 1440, 1733]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_12_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-15/06_icon_8331.png
try:
    _c6 = get_crop(6, 52, 56)
    canvas.paste(_c6, (117, 7), _c6)
except Exception:
    pass
layout["8331"] = [117, 7, 169, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_12_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-15/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 56, 53)
    canvas.paste(_c7, (180, 8), _c7)
except Exception:
    pass
layout["icon_7"] = [180, 8, 236, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_12_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-15/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 52, 52)
    canvas.paste(_c8, (315, 10), _c8)
except Exception:
    pass
layout["icon_8"] = [315, 10, 367, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_12_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-15/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 55, 51)
    canvas.paste(_c9, (246, 10), _c9)
except Exception:
    pass
layout["icon_9"] = [246, 10, 301, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_12_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-15/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 50, 61)
    canvas.paste(_c10, (1153, 6), _c10)
except Exception:
    pass
layout["icon_10"] = [1153, 6, 1203, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_12_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-15/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 100, 67)
    canvas.paste(_c11, (1217, 3), _c11)
except Exception:
    pass
layout["icon_11"] = [1217, 3, 1317, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_12_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-15/12_icon_Keep_the_Party_Going_A_Tribute_to_Jimmy.png
try:
    _c12 = get_crop(12, 1440, 126)
    canvas.paste(_c12, (0, 1020), _c12)
except Exception:
    pass
layout["Keep_the_Party_Going:_A_T"] = [0, 1020, 1440, 1146]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_12_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-15/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 54, 69)
    canvas.paste(_c13, (1319, 2), _c13)
except Exception:
    pass
layout["icon_13"] = [1319, 2, 1373, 71]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_12_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-15/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 52, 57)
    canvas.paste(_c14, (382, 6), _c14)
except Exception:
    pass
layout["icon_14"] = [382, 6, 434, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_12_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-15/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 76, 99)
    canvas.paste(_c15, (1309, 1040), _c15)
except Exception:
    pass
layout["icon_15"] = [1309, 1040, 1385, 1139]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_12_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-15/16_text_8331.png
try:
    _c16 = get_crop(16, 92, 49)
    canvas.paste(_c16, (16, 12), _c16)
except Exception:
    pass
layout["8331"] = [16, 12, 108, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_12_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-15/17_text_Los_Angeles_CA.png
try:
    _c17 = get_crop(17, 422, 73)
    canvas.paste(_c17, (54, 1262), _c17)
except Exception:
    pass
layout["Los_Angeles,_CA"] = [54, 1262, 476, 1335]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_12_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-15/18_text_AII_Shows.png
try:
    _c18 = get_crop(18, 249, 55)
    canvas.paste(_c18, (60, 1852), _c18)
except Exception:
    pass
layout["AII_Shows"] = [60, 1852, 309, 1907]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_12_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-15/19_text_Keep_the_Party_Going_A_Tribute_to_Jimmy.png
try:
    _c19 = get_crop(19, 1440, 367)
    canvas.paste(_c19, (0, 1953), _c19)
except Exception:
    pass
layout["Keep_the_Party_Going:_A_T"] = [0, 1953, 1440, 2320]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_12_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-15/20_text_Buffett.png
try:
    _c20 = get_crop(20, 165, 52)
    canvas.paste(_c20, (315, 2061), _c20)
except Exception:
    pass
layout["Buffett"] = [315, 2061, 480, 2113]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_12_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-15/21_text_7.00_PM.png
try:
    _c21 = get_crop(21, 179, 49)
    canvas.paste(_c21, (312, 2144), _c21)
except Exception:
    pass
layout["7.00_PM"] = [312, 2144, 491, 2193]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_12_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-15/22_text_Los_Angeles_CA.png
try:
    _c22 = get_crop(22, 1440, 367)
    canvas.paste(_c22, (0, 1953), _c22)
except Exception:
    pass
layout["Los_Angeles,_CA"] = [0, 1953, 1440, 2320]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_12_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-15/23_text_From_S251.png
try:
    _c23 = get_crop(23, 224, 57)
    canvas.paste(_c23, (314, 2229), _c23)
except Exception:
    pass
layout["From_S251"] = [314, 2229, 538, 2286]
