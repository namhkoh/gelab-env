# page_id: page_seatgeek_2d936b7d0af74c20ba6cd92218729838_05
# screenshot: 2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-8.png
# step_index: 5/12
# task: Open SeatGeek. Track "Los Angeles Clippers" and "Golden State Warriors".
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Top status bar
status_h = 96
draw.rectangle([(0, 0), (1440, status_h)], fill=(22, 22, 22))

# Subtle top gradient overlay (to sit over hero image area)
# Draw a few translucent-like bands using slightly different grays to simulate a gradient
for i, y in enumerate(range(status_h, status_h + 220, 40)):
    shade = 230 - i * 6
    draw.rectangle([(0, y), (1440, y + 40)], fill=(shade, shade, shade))

# Title card (rounded) that sits below the hero image
title_card_y0 = 560
title_card_y1 = 980
draw.rounded_rectangle(
    [(36, title_card_y0), (1404, title_card_y1)],
    radius=24,
    fill=(255, 255, 255),
    outline=(230, 230, 230),
    width=1
)

# Subtle divider under the buyer guarantee row (approximate)
divider_y = 1060
draw.line([(36, divider_y), (1404, divider_y)], fill=(240, 240, 240), width=1)

# "No Games near ..." card area (rounded)
no_games_y0 = 1220
no_games_y1 = 1500
draw.rounded_rectangle(
    [(36, no_games_y0), (1404, no_games_y1)],
    radius=16,
    fill=(255, 255, 255),
    outline=(235, 235, 235),
    width=1
)

# Separator line below the No Games card
draw.line([(36, no_games_y1 + 20), (1404, no_games_y1 + 20)], fill=(243, 243, 243), width=1)

# "All Games" header background separation (just a subtle band)
all_games_header_y = 1620
draw.rectangle([(0, all_games_header_y - 8), (1440, all_games_header_y + 8)], fill=(250, 250, 250))

# List container background (light)
list_y0 = all_games_header_y
list_y1 = 2960
draw.rectangle([(0, list_y0), (1440, list_y1)], fill=(255, 255, 255))

# Draw separators between list items using detected item top positions
# Items detected at tops: 1785, 2152, 2519, (each item height ~367)
item_tops = [1785, 2152, 2519, 2886]
for y in item_tops:
    # thin subtle divider
    draw.line([(36, y - 1), (1404, y - 1)], fill=(242, 242, 242), width=1)
    # slight shadow below divider to give separation
    draw.line([(36, y + 2), (1404, y + 2)], fill=(250, 250, 250), width=1)

# Left gutter subtle guide (vertical subtle line to separate date column from content)
draw.line([(160, list_y0 + 20), (160, list_y1)], fill=(248, 248, 248), width=1)

# Bottom padding separator
draw.line([(0, 2940), (1440, 2940)], fill=(240, 240, 240), width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_05_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-8/00_icon_Track_Now.png
try:
    _c0 = get_crop(0, 337, 153)
    canvas.paste(_c0, (60, 1376), _c0)
except Exception:
    pass
layout["Track_Now"] = [60, 1376, 397, 1529]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_05_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-8/01_icon_Los_Angeles_CA.png
try:
    _c1 = get_crop(1, 1440, 367)
    canvas.paste(_c1, (0, 2152), _c1)
except Exception:
    pass
layout["Los_Angeles,_CA"] = [0, 2152, 1440, 2519]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_05_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-8/02_icon_Los_Angeles_CA.png
try:
    _c2 = get_crop(2, 1440, 367)
    canvas.paste(_c2, (0, 1785), _c2)
except Exception:
    pass
layout["Los_Angeles,_CA"] = [0, 1785, 1440, 2152]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_05_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-8/03_icon_23.png
try:
    _c3 = get_crop(3, 1440, 367)
    canvas.paste(_c3, (0, 2152), _c3)
except Exception:
    pass
layout["23"] = [0, 2152, 1440, 2519]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_05_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-8/04_icon_Share_this_performer.png
try:
    _c4 = get_crop(4, 144, 144)
    canvas.paste(_c4, (1260, 84), _c4)
except Exception:
    pass
layout["Share_this_performer"] = [1260, 84, 1404, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_05_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-8/05_icon_26.png
try:
    _c5 = get_crop(5, 1440, 367)
    canvas.paste(_c5, (0, 2519), _c5)
except Exception:
    pass
layout["26"] = [0, 2519, 1440, 2886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_05_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-8/06_icon_Track_this_performer.png
try:
    _c6 = get_crop(6, 144, 144)
    canvas.paste(_c6, (1104, 84), _c6)
except Exception:
    pass
layout["Track_this_performer"] = [1104, 84, 1248, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_05_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-8/07_icon_6.53_W.png
try:
    _c7 = get_crop(7, 144, 144)
    canvas.paste(_c7, (36, 84), _c7)
except Exception:
    pass
layout["6.53_W"] = [36, 84, 180, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_05_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-8/08_icon_Western_Conference_First_Round_LA_Clippe.png
try:
    _c8 = get_crop(8, 1440, 367)
    canvas.paste(_c8, (0, 2519), _c8)
except Exception:
    pass
layout["Western_Conference_First_"] = [0, 2519, 1440, 2886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_05_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-8/09_icon_21.png
try:
    _c9 = get_crop(9, 1440, 367)
    canvas.paste(_c9, (0, 1785), _c9)
except Exception:
    pass
layout["21"] = [0, 1785, 1440, 2152]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_05_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-8/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 54, 64)
    canvas.paste(_c10, (1150, 3), _c10)
except Exception:
    pass
layout["icon_10"] = [1150, 3, 1204, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_05_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-8/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 86, 108)
    canvas.paste(_c11, (1305, 950), _c11)
except Exception:
    pass
layout["icon_11"] = [1305, 950, 1391, 1058]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_05_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-8/12_icon_6.53_W.png
try:
    _c12 = get_crop(12, 144, 144)
    canvas.paste(_c12, (36, 84), _c12)
except Exception:
    pass
layout["6.53_W"] = [36, 84, 180, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_05_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-8/13_icon_Protected_by_our_Buyer_Guarantee.png
try:
    _c13 = get_crop(13, 1440, 126)
    canvas.paste(_c13, (0, 933), _c13)
except Exception:
    pass
layout["Protected_by_our_Buyer_Gu"] = [0, 933, 1440, 1059]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_05_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-8/14_icon_Fri.png
try:
    _c14 = get_crop(14, 211, 54)
    canvas.paste(_c14, (56, 2906), _c14)
except Exception:
    pass
layout["Fri"] = [56, 2906, 267, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_05_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-8/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 110, 67)
    canvas.paste(_c15, (1215, 2), _c15)
except Exception:
    pass
layout["icon_15"] = [1215, 2, 1325, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_05_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-8/16_icon_6.53_W.png
try:
    _c16 = get_crop(16, 58, 65)
    canvas.paste(_c16, (181, 1), _c16)
except Exception:
    pass
layout["6.53_W"] = [181, 1, 239, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_05_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-8/17_icon_Western_Conference_First_Round_Dallas.png
try:
    _c17 = get_crop(17, 1440, 367)
    canvas.paste(_c17, (0, 1785), _c17)
except Exception:
    pass
layout["Western_Conference_First_"] = [0, 1785, 1440, 2152]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_05_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-8/18_text_No_Games_near_New_York_NY.png
try:
    _c18 = get_crop(18, 337, 153)
    canvas.paste(_c18, (60, 1376), _c18)
except Exception:
    pass
layout["No_Games_near_New_York,_N"] = [60, 1376, 397, 1529]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_05_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-8/19_text_Track_Los_Angeles_Clippers_for_event_upd.png
try:
    _c19 = get_crop(19, 337, 153)
    canvas.paste(_c19, (60, 1376), _c19)
except Exception:
    pass
layout["Track_Los_Angeles_Clipper"] = [60, 1376, 397, 1529]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_05_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-8/20_text_AIl_Games.png
try:
    _c20 = get_crop(20, 268, 60)
    canvas.paste(_c20, (59, 1682), _c20)
except Exception:
    pass
layout["AIl_Games"] = [59, 1682, 327, 1742]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_05_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-8/21_text_Western_Conference_First_Round_LA_Clinne.png
try:
    _c21 = get_crop(21, 1440, 367)
    canvas.paste(_c21, (0, 2519), _c21)
except Exception:
    pass
layout["Western_Conference_First_"] = [0, 2519, 1440, 2886]
