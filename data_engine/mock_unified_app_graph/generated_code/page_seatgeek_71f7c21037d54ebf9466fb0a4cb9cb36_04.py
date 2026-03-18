# page_id: page_seatgeek_71f7c21037d54ebf9466fb0a4cb9cb36_04
# screenshot: 2024_4_22_19_2_71f7c21037d54ebf9466fb0a4cb9cb36-7.png
# step_index: 4/4
# task: Open SeatGeek. Search for concerts in "New York City". Filter by "pop" genre. What is the second recommendation?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Background fill
draw.rectangle((0, 0, 1440, 2960), fill=(246, 246, 246))  # overall app background (very light gray)

# Status bar area (top ~72px)
status_h = 72
draw.rectangle((0, 0, 1440, status_h), fill=(240, 240, 240))  # subtle gray status bar background
# thin bottom divider under status bar
draw.line((0, status_h, 1440, status_h), fill=(220, 220, 220), width=1)

# Header / toolbar area
header_top = status_h
header_bottom = 168
draw.rectangle((0, header_top, 1440, header_bottom), fill=(255, 255, 255))  # header background (white)
# bottom divider line for header
draw.line((24, header_bottom, 1416, header_bottom), fill=(225, 225, 225), width=2)

# Filter "pills" container background (single rounded container behind pill chips)
pills_top = 300
pills_bottom = 370
pills_left = 24
pills_right = 1416
draw.rounded_rectangle((pills_left, pills_top, pills_right, pills_bottom),
                       radius=28,
                       fill=(255, 255, 255),
                       outline=(225, 225, 225),
                       width=2)
# subtle separator below pills
draw.line((24, pills_bottom + 16, 1416, pills_bottom + 16), fill=(235, 235, 235), width=1)

# Event card backgrounds (rounded white cards behind event details)
# First event card (around area where "Track"/"Share" buttons will be pasted)
card_margin = 24
first_card_top = 1320
first_card_bottom = 1506
draw.rounded_rectangle((card_margin, first_card_top, 1440 - card_margin, first_card_bottom),
                       radius=8,
                       fill=(255, 255, 255),
                       outline=(235, 235, 235),
                       width=1)
# subtle top divider inside first card (to suggest separation from image above)
draw.line((card_margin + 12, first_card_top + 72, 1440 - card_margin - 12, first_card_top + 72),
          fill=(240, 240, 240), width=1)

# Divider between feed items
divider_y = first_card_bottom + 16
draw.line((0, divider_y, 1440, divider_y), fill=(230, 230, 230), width=12)  # thick faint gap between feed sections

# Second event card (lower in the feed)
second_card_top = 2440
second_card_bottom = 2626
draw.rounded_rectangle((card_margin, second_card_top, 1440 - card_margin, second_card_bottom),
                       radius=8,
                       fill=(255, 255, 255),
                       outline=(235, 235, 235),
                       width=1)
# subtle top divider inside second card
draw.line((card_margin + 12, second_card_top + 72, 1440 - card_margin - 12, second_card_top + 72),
          fill=(240, 240, 240), width=1)

# Thin separators and subtle rules across the page
# Under header area (for the title/subtitle region)
draw.line((24, 216, 1416, 216), fill=(245, 245, 245), width=1)

# section separators (below each card)
draw.line((24, first_card_bottom + 6, 1416, first_card_bottom + 6), fill=(230, 230, 230), width=1)
draw.line((24, second_card_bottom + 6, 1416, second_card_bottom + 6), fill=(230, 230, 230), width=1)

# Light inset content background strips to suggest image/banner placeholders without drawing actual content.
# These are subtle and intentionally do NOT overlap the exact detected image bounding boxes.
# Upper subtle band (behind where the top image sits, but inset to avoid duplicating detected image content)
draw.rectangle((48, 420, 1392, 520), fill=(250, 246, 255))  # faint tint hint (very light purple tint)
# Lower subtle band (between the two large image areas)
draw.rectangle((48, 1588, 1392, 1688), fill=(243, 249, 255))  # faint light blue tint

# Final thin bottom padding line
draw.line((0, 2956, 1440, 2956), fill=(240, 240, 240), width=8)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/71f7c21037d54ebf9466fb0a4cb9cb36/step_04_2024_4_22_19_2_71f7c21037d54ebf9466fb0a4cb9cb36-7/00_icon_Alternative.png
try:
    _c0 = get_crop(0, 311, 97)
    canvas.paste(_c0, (240, 335), _c0)
except Exception:
    pass
layout["Alternative"] = [240, 335, 551, 432]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/71f7c21037d54ebf9466fb0a4cb9cb36/step_04_2024_4_22_19_2_71f7c21037d54ebf9466fb0a4cb9cb36-7/01_icon_Hip-Hop.png
try:
    _c1 = get_crop(1, 264, 97)
    canvas.paste(_c1, (775, 335), _c1)
except Exception:
    pass
layout["Hip-Hop"] = [775, 335, 1039, 432]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/71f7c21037d54ebf9466fb0a4cb9cb36/step_04_2024_4_22_19_2_71f7c21037d54ebf9466fb0a4cb9cb36-7/02_icon_Rock.png
try:
    _c2 = get_crop(2, 195, 97)
    canvas.paste(_c2, (21, 335), _c2)
except Exception:
    pass
layout["Rock"] = [21, 335, 216, 432]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/71f7c21037d54ebf9466fb0a4cb9cb36/step_04_2024_4_22_19_2_71f7c21037d54ebf9466fb0a4cb9cb36-7/03_icon_Rap.png
try:
    _c3 = get_crop(3, 169, 97)
    canvas.paste(_c3, (1271, 335), _c3)
except Exception:
    pass
layout["Rap"] = [1271, 335, 1440, 432]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/71f7c21037d54ebf9466fb0a4cb9cb36/step_04_2024_4_22_19_2_71f7c21037d54ebf9466fb0a4cb9cb36-7/04_icon_Folk.png
try:
    _c4 = get_crop(4, 176, 97)
    canvas.paste(_c4, (575, 335), _c4)
except Exception:
    pass
layout["Folk"] = [575, 335, 751, 432]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/71f7c21037d54ebf9466fb0a4cb9cb36/step_04_2024_4_22_19_2_71f7c21037d54ebf9466fb0a4cb9cb36-7/05_icon_Rnb.png
try:
    _c5 = get_crop(5, 175, 97)
    canvas.paste(_c5, (1063, 335), _c5)
except Exception:
    pass
layout["Rnb"] = [1063, 335, 1238, 432]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/71f7c21037d54ebf9466fb0a4cb9cb36/step_04_2024_4_22_19_2_71f7c21037d54ebf9466fb0a4cb9cb36-7/06_icon_Track.png
try:
    _c6 = get_crop(6, 267, 185)
    canvas.paste(_c6, (0, 1382), _c6)
except Exception:
    pass
layout["Track"] = [0, 1382, 267, 1567]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/71f7c21037d54ebf9466fb0a4cb9cb36/step_04_2024_4_22_19_2_71f7c21037d54ebf9466fb0a4cb9cb36-7/07_icon_Track.png
try:
    _c7 = get_crop(7, 267, 185)
    canvas.paste(_c7, (0, 2517), _c7)
except Exception:
    pass
layout["Track"] = [0, 2517, 267, 2702]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/71f7c21037d54ebf9466fb0a4cb9cb36/step_04_2024_4_22_19_2_71f7c21037d54ebf9466fb0a4cb9cb36-7/08_icon_Share.png
try:
    _c8 = get_crop(8, 248, 162)
    canvas.paste(_c8, (267, 2533), _c8)
except Exception:
    pass
layout["Share"] = [267, 2533, 515, 2695]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/71f7c21037d54ebf9466fb0a4cb9cb36/step_04_2024_4_22_19_2_71f7c21037d54ebf9466fb0a4cb9cb36-7/09_icon_Share.png
try:
    _c9 = get_crop(9, 248, 162)
    canvas.paste(_c9, (267, 1398), _c9)
except Exception:
    pass
layout["Share"] = [267, 1398, 515, 1560]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/71f7c21037d54ebf9466fb0a4cb9cb36/step_04_2024_4_22_19_2_71f7c21037d54ebf9466fb0a4cb9cb36-7/10_icon_884.png
try:
    _c10 = get_crop(10, 144, 240)
    canvas.paste(_c10, (1260, 72), _c10)
except Exception:
    pass
layout["884"] = [1260, 72, 1404, 312]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/71f7c21037d54ebf9466fb0a4cb9cb36/step_04_2024_4_22_19_2_71f7c21037d54ebf9466fb0a4cb9cb36-7/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 54, 56)
    canvas.paste(_c11, (314, 7), _c11)
except Exception:
    pass
layout["icon_11"] = [314, 7, 368, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/71f7c21037d54ebf9466fb0a4cb9cb36/step_04_2024_4_22_19_2_71f7c21037d54ebf9466fb0a4cb9cb36-7/12_icon_Camden_NJ.png
try:
    _c12 = get_crop(12, 1440, 1135)
    canvas.paste(_c12, (0, 1591), _c12)
except Exception:
    pass
layout["Camden,_NJ"] = [0, 1591, 1440, 2726]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/71f7c21037d54ebf9466fb0a4cb9cb36/step_04_2024_4_22_19_2_71f7c21037d54ebf9466fb0a4cb9cb36-7/13_icon_Pop.png
try:
    _c13 = get_crop(13, 61, 56)
    canvas.paste(_c13, (242, 7), _c13)
except Exception:
    pass
layout["Pop"] = [242, 7, 303, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/71f7c21037d54ebf9466fb0a4cb9cb36/step_04_2024_4_22_19_2_71f7c21037d54ebf9466fb0a4cb9cb36-7/14_icon_7.03_my.png
try:
    _c14 = get_crop(14, 144, 240)
    canvas.paste(_c14, (0, 72), _c14)
except Exception:
    pass
layout["7.03_my"] = [0, 72, 144, 312]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/71f7c21037d54ebf9466fb0a4cb9cb36/step_04_2024_4_22_19_2_71f7c21037d54ebf9466fb0a4cb9cb36-7/15_icon_7.03_my.png
try:
    _c15 = get_crop(15, 58, 59)
    canvas.paste(_c15, (113, 3), _c15)
except Exception:
    pass
layout["7.03_my"] = [113, 3, 171, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/71f7c21037d54ebf9466fb0a4cb9cb36/step_04_2024_4_22_19_2_71f7c21037d54ebf9466fb0a4cb9cb36-7/16_icon_icon_16.png
try:
    _c16 = get_crop(16, 48, 65)
    canvas.paste(_c16, (1154, 1), _c16)
except Exception:
    pass
layout["icon_16"] = [1154, 1, 1202, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/71f7c21037d54ebf9466fb0a4cb9cb36/step_04_2024_4_22_19_2_71f7c21037d54ebf9466fb0a4cb9cb36-7/17_icon_884.png
try:
    _c17 = get_crop(17, 94, 61)
    canvas.paste(_c17, (1216, 2), _c17)
except Exception:
    pass
layout["884"] = [1216, 2, 1310, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/71f7c21037d54ebf9466fb0a4cb9cb36/step_04_2024_4_22_19_2_71f7c21037d54ebf9466fb0a4cb9cb36-7/18_icon_7.03_my.png
try:
    _c18 = get_crop(18, 49, 57)
    canvas.paste(_c18, (184, 5), _c18)
except Exception:
    pass
layout["7.03_my"] = [184, 5, 233, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/71f7c21037d54ebf9466fb0a4cb9cb36/step_04_2024_4_22_19_2_71f7c21037d54ebf9466fb0a4cb9cb36-7/19_icon_icon_19.png
try:
    _c19 = get_crop(19, 49, 56)
    canvas.paste(_c19, (1320, 5), _c19)
except Exception:
    pass
layout["icon_19"] = [1320, 5, 1369, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/71f7c21037d54ebf9466fb0a4cb9cb36/step_04_2024_4_22_19_2_71f7c21037d54ebf9466fb0a4cb9cb36-7/20_icon_icon_20.png
try:
    _c20 = get_crop(20, 44, 58)
    canvas.paste(_c20, (385, 5), _c20)
except Exception:
    pass
layout["icon_20"] = [385, 5, 429, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/71f7c21037d54ebf9466fb0a4cb9cb36/step_04_2024_4_22_19_2_71f7c21037d54ebf9466fb0a4cb9cb36-7/21_icon_New_York_NY.png
try:
    _c21 = get_crop(21, 1440, 1135)
    canvas.paste(_c21, (0, 456), _c21)
except Exception:
    pass
layout["New_York,_NY"] = [0, 456, 1440, 1591]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/71f7c21037d54ebf9466fb0a4cb9cb36/step_04_2024_4_22_19_2_71f7c21037d54ebf9466fb0a4cb9cb36-7/22_text_New_York_NY.png
try:
    _c22 = get_crop(22, 311, 97)
    canvas.paste(_c22, (240, 335), _c22)
except Exception:
    pass
layout["New_York,_NY"] = [240, 335, 551, 432]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/71f7c21037d54ebf9466fb0a4cb9cb36/step_04_2024_4_22_19_2_71f7c21037d54ebf9466fb0a4cb9cb36-7/23_text_date.png
try:
    _c23 = get_crop(23, 117, 52)
    canvas.paste(_c23, (606, 208), _c23)
except Exception:
    pass
layout["date"] = [606, 208, 723, 260]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/71f7c21037d54ebf9466fb0a4cb9cb36/step_04_2024_4_22_19_2_71f7c21037d54ebf9466fb0a4cb9cb36-7/24_text_Hozier.png
try:
    _c24 = get_crop(24, 167, 60)
    canvas.paste(_c24, (42, 2336), _c24)
except Exception:
    pass
layout["Hozier"] = [42, 2336, 209, 2396]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/71f7c21037d54ebf9466fb0a4cb9cb36/step_04_2024_4_22_19_2_71f7c21037d54ebf9466fb0a4cb9cb36-7/25_text_Sat.png
try:
    _c25 = get_crop(25, 99, 54)
    canvas.paste(_c25, (43, 2413), _c25)
except Exception:
    pass
layout["Sat,"] = [43, 2413, 142, 2467]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/71f7c21037d54ebf9466fb0a4cb9cb36/step_04_2024_4_22_19_2_71f7c21037d54ebf9466fb0a4cb9cb36-7/26_text_25_8_PM.png
try:
    _c26 = get_crop(26, 191, 52)
    canvas.paste(_c26, (238, 2412), _c26)
except Exception:
    pass
layout["25,_8_PM"] = [238, 2412, 429, 2464]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/71f7c21037d54ebf9466fb0a4cb9cb36/step_04_2024_4_22_19_2_71f7c21037d54ebf9466fb0a4cb9cb36-7/27_text_Camden_NJ.png
try:
    _c27 = get_crop(27, 265, 60)
    canvas.paste(_c27, (452, 2408), _c27)
except Exception:
    pass
layout["Camden,_NJ"] = [452, 2408, 717, 2468]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/71f7c21037d54ebf9466fb0a4cb9cb36/step_04_2024_4_22_19_2_71f7c21037d54ebf9466fb0a4cb9cb36-7/28_text_Freedom_Mortgage_Pavilion.png
try:
    _c28 = get_crop(28, 1440, 1135)
    canvas.paste(_c28, (0, 1591), _c28)
except Exception:
    pass
layout["Freedom_Mortgage_Pavilion"] = [0, 1591, 1440, 2726]
