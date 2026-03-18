# page_id: page_seatgeek_68e3462c14734440a7ace3fed432a10d_13
# screenshot: 2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-16.png
# step_index: 13/13
# task: Open SeatGeek and change the current location to Los Angeles. Then find the first concert show and track its performer.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw UI background and structure for a 1440x2960 canvas using provided `canvas` and `draw`.

# Overall background
draw.rectangle([(0, 0), (1440, 2960)], fill="#FBFBFB")  # very light off-white page background

# Top hero / artwork area (dark banner)
hero_height = 520
draw.rectangle([(0, 0), (1440, hero_height)], fill="#0B0B0B")  # deep black banner behind the hero illustration

# Status bar area (top ~50-88px). Keep it same dark tone so icons pasted later sit on it.
status_height = 88
draw.rectangle([(0, 0), (1440, status_height)], fill="#000000")

# Subtle top overlay gradient imitation (simple darker band) to help separation from hero art
# (draw a slightly translucent-looking darker band by just drawing a slightly different black)
draw.rectangle([(0, status_height), (1440, status_height + 6)], fill="#070707")

# Header/content divider under hero area
divider_y = hero_height
draw.line([(40, divider_y), (1400, divider_y)], fill="#E8E8E8", width=1)

# Title header background area (behind the title text block that will be pasted)
title_block_top = 980
title_block_bottom = 1140
draw.rounded_rectangle(
    [(24, title_block_top), (1440 - 24, title_block_bottom)],
    radius=6,
    fill="#FFFFFF",
    outline=None
)

# Thin divider below title block
draw.line([(24, title_block_bottom + 1), (1440 - 24, title_block_bottom + 1)], fill="#E6E6E6", width=1)

# "Los Angeles, CA" section card background (first listing area)
first_card_top = 1200
first_card_bottom = 1560
# subtle shadow
draw.rectangle([(24, first_card_top + 8), (1440 - 24, first_card_bottom + 10)], fill="#F2F2F2")
# main card
draw.rounded_rectangle(
    [(24, first_card_top), (1440 - 24, first_card_bottom)],
    radius=10,
    fill="#FFFFFF",
    outline=None
)

# Divider above first listing (separates header/title from listings)
draw.line([(24, first_card_top - 1), (1440 - 24, first_card_top - 1)], fill="#F0F0F0", width=1)

# Draw a subtle horizontal separator inside the first card to hint at sections (but avoid drawing icons/text)
inner_sep_y = first_card_top + 160
draw.line([(40, inner_sep_y), (1440 - 40, inner_sep_y)], fill="#F3F3F3", width=1)

# "All Shows" section header area separation
all_shows_top = 1680
draw.line([(24, all_shows_top - 40), (1440 - 24, all_shows_top - 40)], fill="#F0F0F0", width=1)

# Second listing card background (All Shows list item)
second_card_top = 1720
second_card_bottom = 2050
# subtle shadow
draw.rectangle([(24, second_card_top + 8), (1440 - 24, second_card_bottom + 10)], fill="#F7F7F7")
# main card
draw.rounded_rectangle(
    [(24, second_card_top), (1440 - 24, second_card_bottom)],
    radius=10,
    fill="#FFFFFF",
    outline=None
)

# Dividers corresponding to detected layout rows (do not draw icons/text)
# Divider lines at the top edges where lists begin (matching detected positions)
draw.line([(0, 1366), (1440, 1366)], fill="#ECECEC", width=1)
draw.line([(0, 1953), (1440, 1953)], fill="#ECECEC", width=1)

# Bottom area remains empty (light background). Add a faint footer guideline near bottom of content area.
draw.line([(24, 2200), (1440 - 24, 2200)], fill="#F5F5F5", width=1)

# Add subtle left gutter shadow to emphasize cards (vertical faint line)
draw.line([(24, 920), (24, 2200)], fill="#F4F4F4", width=2)
draw.line([(1440 - 24, 920), (1440 - 24, 2200)], fill="#F4F4F4", width=2)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_13_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-16/00_icon_Track_this_performer.png
try:
    _c0 = get_crop(0, 144, 144)
    canvas.paste(_c0, (1104, 84), _c0)
except Exception:
    pass
layout["Track_this_performer"] = [1104, 84, 1248, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_13_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-16/01_icon_Share_this_performer.png
try:
    _c1 = get_crop(1, 144, 144)
    canvas.paste(_c1, (1260, 84), _c1)
except Exception:
    pass
layout["Share_this_performer"] = [1260, 84, 1404, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_13_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-16/02_icon_Hollywood_Bowl.png
try:
    _c2 = get_crop(2, 1440, 367)
    canvas.paste(_c2, (0, 1366), _c2)
except Exception:
    pass
layout["Hollywood_Bowl"] = [0, 1366, 1440, 1733]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_13_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-16/03_icon_11.png
try:
    _c3 = get_crop(3, 1440, 367)
    canvas.paste(_c3, (0, 1953), _c3)
except Exception:
    pass
layout["11"] = [0, 1953, 1440, 2320]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_13_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-16/04_icon_8331.png
try:
    _c4 = get_crop(4, 144, 144)
    canvas.paste(_c4, (36, 84), _c4)
except Exception:
    pass
layout["8331"] = [36, 84, 180, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_13_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-16/05_icon_11.png
try:
    _c5 = get_crop(5, 1440, 367)
    canvas.paste(_c5, (0, 1366), _c5)
except Exception:
    pass
layout["11"] = [0, 1366, 1440, 1733]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_13_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-16/06_icon_8331.png
try:
    _c6 = get_crop(6, 52, 56)
    canvas.paste(_c6, (117, 7), _c6)
except Exception:
    pass
layout["8331"] = [117, 7, 169, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_13_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-16/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 57, 53)
    canvas.paste(_c7, (180, 8), _c7)
except Exception:
    pass
layout["icon_7"] = [180, 8, 237, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_13_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-16/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 52, 52)
    canvas.paste(_c8, (315, 10), _c8)
except Exception:
    pass
layout["icon_8"] = [315, 10, 367, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_13_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-16/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 55, 52)
    canvas.paste(_c9, (246, 9), _c9)
except Exception:
    pass
layout["icon_9"] = [246, 9, 301, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_13_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-16/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 51, 62)
    canvas.paste(_c10, (1152, 6), _c10)
except Exception:
    pass
layout["icon_10"] = [1152, 6, 1203, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_13_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-16/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 100, 68)
    canvas.paste(_c11, (1217, 3), _c11)
except Exception:
    pass
layout["icon_11"] = [1217, 3, 1317, 71]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_13_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-16/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 54, 70)
    canvas.paste(_c12, (1319, 2), _c12)
except Exception:
    pass
layout["icon_12"] = [1319, 2, 1373, 72]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_13_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-16/13_icon_Keep_the_Party_Going_A_Tribute_to_Jimmy.png
try:
    _c13 = get_crop(13, 1440, 126)
    canvas.paste(_c13, (0, 1020), _c13)
except Exception:
    pass
layout["Keep_the_Party_Going:_A_T"] = [0, 1020, 1440, 1146]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_13_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-16/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 52, 58)
    canvas.paste(_c14, (382, 6), _c14)
except Exception:
    pass
layout["icon_14"] = [382, 6, 434, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_13_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-16/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 76, 99)
    canvas.paste(_c15, (1309, 1040), _c15)
except Exception:
    pass
layout["icon_15"] = [1309, 1040, 1385, 1139]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_13_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-16/16_text_8331.png
try:
    _c16 = get_crop(16, 92, 49)
    canvas.paste(_c16, (16, 12), _c16)
except Exception:
    pass
layout["8331"] = [16, 12, 108, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_13_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-16/17_text_Los_Angeles_CA.png
try:
    _c17 = get_crop(17, 422, 73)
    canvas.paste(_c17, (54, 1262), _c17)
except Exception:
    pass
layout["Los_Angeles,_CA"] = [54, 1262, 476, 1335]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_13_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-16/18_text_AII_Shows.png
try:
    _c18 = get_crop(18, 249, 55)
    canvas.paste(_c18, (60, 1852), _c18)
except Exception:
    pass
layout["AII_Shows"] = [60, 1852, 309, 1907]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_13_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-16/19_text_Keep_the_Party_Going_A_Tribute_to_Jimmy.png
try:
    _c19 = get_crop(19, 1440, 367)
    canvas.paste(_c19, (0, 1953), _c19)
except Exception:
    pass
layout["Keep_the_Party_Going:_A_T"] = [0, 1953, 1440, 2320]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_13_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-16/20_text_Buffett.png
try:
    _c20 = get_crop(20, 165, 52)
    canvas.paste(_c20, (315, 2061), _c20)
except Exception:
    pass
layout["Buffett"] = [315, 2061, 480, 2113]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_13_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-16/21_text_7.00_PM.png
try:
    _c21 = get_crop(21, 179, 49)
    canvas.paste(_c21, (312, 2144), _c21)
except Exception:
    pass
layout["7.00_PM"] = [312, 2144, 491, 2193]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_13_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-16/22_text_Los_Angeles_CA.png
try:
    _c22 = get_crop(22, 1440, 367)
    canvas.paste(_c22, (0, 1953), _c22)
except Exception:
    pass
layout["Los_Angeles,_CA"] = [0, 1953, 1440, 2320]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_13_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-16/23_text_From_S251.png
try:
    _c23 = get_crop(23, 224, 57)
    canvas.paste(_c23, (314, 2229), _c23)
except Exception:
    pass
layout["From_S251"] = [314, 2229, 538, 2286]
