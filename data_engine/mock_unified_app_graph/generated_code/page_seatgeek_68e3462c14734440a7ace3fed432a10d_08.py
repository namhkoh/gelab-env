# page_id: page_seatgeek_68e3462c14734440a7ace3fed432a10d_08
# screenshot: 2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-11.png
# step_index: 8/13
# task: Open SeatGeek and change the current location to Los Angeles. Then find the first concert show and track its performer.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Top status bar
draw.rectangle([(0, 0), (1440, 80)], fill="#efefef")

# Header / toolbar background
draw.rectangle([(0, 80), (1440, 240)], fill="#ffffff")
# subtle bottom divider for header
draw.line([(24, 240), (1416, 240)], fill="#e6e6e6", width=2)

# Genre/pills container background (keeps same white but with subtle top divider)
draw.rectangle([(0, 240), (1440, 360)], fill="#ffffff")
draw.line([(24, 360), (1416, 360)], fill="#f0f0f0", width=1)

# Spacer area between pills and first big image (keeps clean white)
draw.rectangle([(0, 360), (1440, 456)], fill="#ffffff")
draw.line([(0, 456), (1440, 456)], fill="#e6e6e6", width=2)

# Divider at the bottom of the first large content block (image/post area starts at y=456 and is a detected element,
# so we only draw the dividing line that separates that area from following content)
draw.line([(0, 1591), (1440, 1591)], fill="#e6e6e6", width=2)

# Divider at the bottom of the second large content block (detected element area ends here)
draw.line([(0, 2726), (1440, 2726)], fill="#e6e6e6", width=2)

# Content card background strips above/below large detected image areas
# (we avoid drawing inside detected large image rectangles; these are just white card areas outside them)
# Top card strip (above first image) - subtle rounded rectangle at top-left area for visual structure
draw.rounded_rectangle([(20, 300), (1420, 440)], radius=12, fill="#ffffff", outline="#e9e9e9", width=1)

# Bottom card strip (below the second detected large area)
draw.rounded_rectangle([(20, 2736), (1420, 2890)], radius=12, fill="#ffffff", outline="#e9e9e9", width=1)

# Subtle shadows / separators for list sections (thin lines)
for y in (120, 200, 340, 520, 1000, 1400, 2200):
    # Only draw separators in safe zones (avoid overlapping the two big detected image blocks)
    if not (456 <= y <= 2726):
        draw.line([(24, y), (1416, y)], fill="#f2f2f2", width=1)

# Left and right safe edge gutters (visual guide)
draw.rectangle([(0, 0), (24, 2960)], fill="#ffffff")
draw.rectangle([(1416, 0), (1440, 2960)], fill="#ffffff")

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_08_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-11/00_icon_Rnb.png
try:
    _c0 = get_crop(0, 171, 97)
    canvas.paste(_c0, (1269, 335), _c0)
except Exception:
    pass
layout["Rnb"] = [1269, 335, 1440, 432]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_08_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-11/01_icon_Alternative.png
try:
    _c1 = get_crop(1, 311, 97)
    canvas.paste(_c1, (437, 335), _c1)
except Exception:
    pass
layout["Alternative"] = [437, 335, 748, 432]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_08_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-11/02_icon_Folk.png
try:
    _c2 = get_crop(2, 176, 97)
    canvas.paste(_c2, (772, 335), _c2)
except Exception:
    pass
layout["Folk"] = [772, 335, 948, 432]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_08_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-11/03_icon_Hip-Hop.png
try:
    _c3 = get_crop(3, 264, 97)
    canvas.paste(_c3, (972, 335), _c3)
except Exception:
    pass
layout["Hip-Hop"] = [972, 335, 1236, 432]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_08_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-11/04_icon_Pop.png
try:
    _c4 = get_crop(4, 173, 97)
    canvas.paste(_c4, (21, 335), _c4)
except Exception:
    pass
layout["Pop"] = [21, 335, 194, 432]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_08_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-11/05_icon_Rock.png
try:
    _c5 = get_crop(5, 195, 97)
    canvas.paste(_c5, (218, 335), _c5)
except Exception:
    pass
layout["Rock"] = [218, 335, 413, 432]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_08_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-11/06_icon_Track.png
try:
    _c6 = get_crop(6, 267, 185)
    canvas.paste(_c6, (0, 1382), _c6)
except Exception:
    pass
layout["Track"] = [0, 1382, 267, 1567]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_08_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-11/07_icon_Track.png
try:
    _c7 = get_crop(7, 267, 185)
    canvas.paste(_c7, (0, 2517), _c7)
except Exception:
    pass
layout["Track"] = [0, 2517, 267, 2702]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_08_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-11/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 50, 53)
    canvas.paste(_c8, (316, 7), _c8)
except Exception:
    pass
layout["icon_8"] = [316, 7, 366, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_08_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-11/09_icon_Share.png
try:
    _c9 = get_crop(9, 248, 162)
    canvas.paste(_c9, (267, 1398), _c9)
except Exception:
    pass
layout["Share"] = [267, 1398, 515, 1560]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_08_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-11/10_icon_Share.png
try:
    _c10 = get_crop(10, 248, 162)
    canvas.paste(_c10, (267, 2533), _c10)
except Exception:
    pass
layout["Share"] = [267, 2533, 515, 2695]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_08_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-11/11_icon_Keep_the_Party_Going_A_Tribute_to_Jimmy_.png
try:
    _c11 = get_crop(11, 1440, 1135)
    canvas.paste(_c11, (0, 456), _c11)
except Exception:
    pass
layout["Keep_the_Party_Going:_A_T"] = [0, 456, 1440, 1591]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_08_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-11/12_icon_884.png
try:
    _c12 = get_crop(12, 144, 240)
    canvas.paste(_c12, (1260, 72), _c12)
except Exception:
    pass
layout["884"] = [1260, 72, 1404, 312]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_08_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-11/13_icon_8.31_my.png
try:
    _c13 = get_crop(13, 57, 59)
    canvas.paste(_c13, (112, 4), _c13)
except Exception:
    pass
layout["8.31_my"] = [112, 4, 169, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_08_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-11/14_icon_Concerts.png
try:
    _c14 = get_crop(14, 53, 56)
    canvas.paste(_c14, (247, 6), _c14)
except Exception:
    pass
layout["Concerts"] = [247, 6, 300, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_08_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-11/15_icon_8.31_my.png
try:
    _c15 = get_crop(15, 57, 59)
    canvas.paste(_c15, (181, 3), _c15)
except Exception:
    pass
layout["8.31_my"] = [181, 3, 238, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_08_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-11/16_icon_8.31_my.png
try:
    _c16 = get_crop(16, 144, 240)
    canvas.paste(_c16, (0, 72), _c16)
except Exception:
    pass
layout["8.31_my"] = [0, 72, 144, 312]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_08_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-11/17_icon_icon_17.png
try:
    _c17 = get_crop(17, 48, 56)
    canvas.paste(_c17, (1321, 4), _c17)
except Exception:
    pass
layout["icon_17"] = [1321, 4, 1369, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_08_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-11/18_icon_icon_18.png
try:
    _c18 = get_crop(18, 48, 65)
    canvas.paste(_c18, (1154, 1), _c18)
except Exception:
    pass
layout["icon_18"] = [1154, 1, 1202, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_08_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-11/19_icon_884.png
try:
    _c19 = get_crop(19, 74, 62)
    canvas.paste(_c19, (1214, 2), _c19)
except Exception:
    pass
layout["884"] = [1214, 2, 1288, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_08_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-11/20_icon_icon_20.png
try:
    _c20 = get_crop(20, 45, 58)
    canvas.paste(_c20, (384, 3), _c20)
except Exception:
    pass
layout["icon_20"] = [384, 3, 429, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_08_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-11/21_icon_icon_21.png
try:
    _c21 = get_crop(21, 46, 58)
    canvas.paste(_c21, (1270, 4), _c21)
except Exception:
    pass
layout["icon_21"] = [1270, 4, 1316, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_08_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-11/22_text_Concerts.png
try:
    _c22 = get_crop(22, 267, 63)
    canvas.paste(_c22, (186, 133), _c22)
except Exception:
    pass
layout["Concerts"] = [186, 133, 453, 196]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_08_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-11/23_text_Los_Angeles_CA.png
try:
    _c23 = get_crop(23, 195, 97)
    canvas.paste(_c23, (218, 335), _c23)
except Exception:
    pass
layout["Los_Angeles,_CA"] = [218, 335, 413, 432]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_08_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-11/24_text_date.png
try:
    _c24 = get_crop(24, 116, 52)
    canvas.paste(_c24, (671, 208), _c24)
except Exception:
    pass
layout["date"] = [671, 208, 787, 260]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_08_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-11/25_text_Keep_the_Party_Going_A_Tribute_to_Jimmy_.png
try:
    _c25 = get_crop(25, 1440, 1135)
    canvas.paste(_c25, (0, 456), _c25)
except Exception:
    pass
layout["Keep_the_Party_Going:_A_T"] = [0, 456, 1440, 1591]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_08_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-11/26_text_Thu_Apr_11_7_PM.png
try:
    _c26 = get_crop(26, 267, 185)
    canvas.paste(_c26, (0, 1382), _c26)
except Exception:
    pass
layout["Thu,_Apr_11,7_PM"] = [0, 1382, 267, 1567]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_08_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-11/27_text_Los_Angeles_CA.png
try:
    _c27 = get_crop(27, 346, 57)
    canvas.paste(_c27, (421, 1279), _c27)
except Exception:
    pass
layout["Los_Angeles,_CA"] = [421, 1279, 767, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_08_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-11/28_text_Hollywood_Bowl.png
try:
    _c28 = get_crop(28, 341, 60)
    canvas.paste(_c28, (790, 1275), _c28)
except Exception:
    pass
layout["Hollywood_Bowl"] = [790, 1275, 1131, 1335]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_08_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-11/29_text_S261.png
try:
    _c29 = get_crop(29, 135, 52)
    canvas.paste(_c29, (63, 2190), _c29)
except Exception:
    pass
layout["S261+"] = [63, 2190, 198, 2242]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_08_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-11/30_text_Billy_Joel_and_Sting.png
try:
    _c30 = get_crop(30, 248, 162)
    canvas.paste(_c30, (267, 2533), _c30)
except Exception:
    pass
layout["Billy_Joel_and_Sting"] = [267, 2533, 515, 2695]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_08_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-11/31_text_Sat_Apr_13.png
try:
    _c31 = get_crop(31, 267, 185)
    canvas.paste(_c31, (0, 2517), _c31)
except Exception:
    pass
layout["Sat,_Apr_13,"] = [0, 2517, 267, 2702]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_08_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-11/32_text_7_PM.png
try:
    _c32 = get_crop(32, 112, 48)
    canvas.paste(_c32, (294, 2414), _c32)
except Exception:
    pass
layout["7_PM"] = [294, 2414, 406, 2462]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_08_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-11/33_text_San_Diego_CA.png
try:
    _c33 = get_crop(33, 315, 72)
    canvas.paste(_c33, (424, 2406), _c33)
except Exception:
    pass
layout["San_Diego,_CA"] = [424, 2406, 739, 2478]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_08_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-11/34_text_Petco_Park.png
try:
    _c34 = get_crop(34, 235, 48)
    canvas.paste(_c34, (763, 2414), _c34)
except Exception:
    pass
layout["Petco_Park"] = [763, 2414, 998, 2462]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_08_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-11/35_clickable_261.png
try:
    _c35 = get_crop(35, 1440, 1135)
    canvas.paste(_c35, (0, 1591), _c35)
except Exception:
    pass
layout["$261+"] = [0, 1591, 1440, 2726]
