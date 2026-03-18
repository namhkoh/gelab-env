# page_id: page_seatgeek_094b5cdb02e246858451240263e6ef7f_01
# screenshot: 2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-4.png
# step_index: 1/9
# task: Open SeatGeek. Find the soonest upcoming NBA game in Boston with "Celtics". What is the highest price for each ticket?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Paint overall background
draw.rectangle((0, 0, 1440, 2960), fill="#FBFBFB")

# Status bar (top area)
status_h = 80
draw.rectangle((0, 0, 1440, status_h), fill="#F2F2F2")
# subtle bottom divider under status bar
draw.line((24, status_h - 1, 1440 - 24, status_h - 1), fill="#E6E6E6", width=1)

# Header area (under status bar) - keep it clean/white with subtle bottom divider
header_top = status_h
header_bottom = 160
draw.rectangle((0, header_top, 1440, header_bottom), fill="#FFFFFF")
draw.line((24, header_bottom - 1, 1440 - 24, header_bottom - 1), fill="#EAEAEA", width=1)

# "Just for you" section card background (rounded panel behind the horizontal cards)
jf_x1, jf_x2 = 36, 1440 - 36
jf_y1, jf_y2 = 1360, 2010
jf_radius = 28
# shadow
draw.rounded_rectangle((jf_x1 + 0, jf_y1 + 8, jf_x2 + 0, jf_y2 + 8), radius=jf_radius, fill="#F2F2F2")
# main card
draw.rounded_rectangle((jf_x1, jf_y1, jf_x2, jf_y2), radius=jf_radius, fill="#FFFFFF")

# subtle divider above trending section
trending_top = jf_y2 + 24
draw.line((24, trending_top, 1440 - 24, trending_top), fill="#E9E9E9", width=1)

# Trending events list background (panel)
tr_x1, tr_x2 = 0, 1440
tr_y1, tr_y2 = trending_top + 24, 2760
# very subtle off-white panel so rows sit on a slight surface
draw.rectangle((tr_x1, tr_y1, tr_x2, tr_y2), fill="#FBFBFB")
# draw separators for list rows (approx where detected rows will be pasted); keep them light
row_seps = [2196, 2432, 2668]  # approximate Y coordinates for separators
for y in row_seps:
    draw.line((24, y, 1440 - 24, y), fill="#EFEFEF", width=1)

# Add faint left numeric badges background (decorative dots behind ranking numbers)
# These are only backgrounds (do not draw any numbers/icons)
badge_radius = 56
badge_centers = [(72, 2140), (72, 2376), (72, 2612)]
for cx, cy in badge_centers:
    # dotted circle - approximate by a soft filled circle
    draw.ellipse((cx - badge_radius, cy - badge_radius, cx + badge_radius, cy + badge_radius), fill="#FFF6F4")

# Bottom navigation bar background
nav_top = 2792
nav_bottom = 2960
# top divider/shadow
draw.line((0, nav_top, 1440, nav_top), fill="#E7E7E7", width=1)
# nav background
draw.rectangle((0, nav_top, 1440, nav_bottom), fill="#FFFFFF")
# slight inner top highlight
draw.line((36, nav_top + 2, 1440 - 36, nav_top + 2), fill="#FBFBFB", width=1)

# Decorative left and right edge vertical strips to match screenshot margins
# (very subtle, not duplicating any icons/text)
draw.rectangle((0, header_bottom, 24, 2400), fill="#FBFBFB")
draw.rectangle((1440 - 24, header_bottom, 1440, 2400), fill="#FBFBFB")

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_01_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-4/00_icon_Clippers.png
try:
    _c0 = get_crop(0, 1344, 840)
    canvas.paste(_c0, (48, 360), _c0)
except Exception:
    pass
layout["Clippers"] = [48, 360, 1392, 1200]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_01_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-4/01_icon_Dodger_Stadium.png
try:
    _c1 = get_crop(1, 1309, 236)
    canvas.paste(_c1, (0, 2197), _c1)
except Exception:
    pass
layout["Dodger_Stadium"] = [0, 2197, 1309, 2433]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_01_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-4/02_icon_View_all.png
try:
    _c2 = get_crop(2, 98, 149)
    canvas.paste(_c2, (1342, 2244), _c2)
except Exception:
    pass
layout["View_all"] = [1342, 2244, 1440, 2393]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_01_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-4/03_icon_S53.png
try:
    _c3 = get_crop(3, 462, 519)
    canvas.paste(_c3, (48, 1431), _c3)
except Exception:
    pass
layout["S53+"] = [48, 1431, 510, 1950]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_01_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-4/04_icon_icon_4.png
try:
    _c4 = get_crop(4, 104, 148)
    canvas.paste(_c4, (1336, 2481), _c4)
except Exception:
    pass
layout["icon_4"] = [1336, 2481, 1440, 2629]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_01_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-4/05_icon_Angel_Stadium_of_Anaheim.png
try:
    _c5 = get_crop(5, 1309, 236)
    canvas.paste(_c5, (0, 2433), _c5)
except Exception:
    pass
layout["Angel_Stadium_of_Anaheim"] = [0, 2433, 1309, 2669]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_01_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-4/06_icon_Los_Angeles_CA.png
try:
    _c6 = get_crop(6, 64, 57)
    canvas.paste(_c6, (242, 5), _c6)
except Exception:
    pass
layout["Los_Angeles,_CA"] = [242, 5, 306, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_01_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-4/07_icon_4.59_my.png
try:
    _c7 = get_crop(7, 55, 57)
    canvas.paste(_c7, (114, 5), _c7)
except Exception:
    pass
layout["4.59_my"] = [114, 5, 169, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_01_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-4/08_icon_888.png
try:
    _c8 = get_crop(8, 144, 240)
    canvas.paste(_c8, (1260, 72), _c8)
except Exception:
    pass
layout["888"] = [1260, 72, 1404, 312]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_01_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-4/09_icon_888.png
try:
    _c9 = get_crop(9, 98, 64)
    canvas.paste(_c9, (1215, 0), _c9)
except Exception:
    pass
layout["888"] = [1215, 0, 1313, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_01_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-4/10_icon_4.59_my.png
try:
    _c10 = get_crop(10, 50, 56)
    canvas.paste(_c10, (184, 6), _c10)
except Exception:
    pass
layout["4.59_my"] = [184, 6, 234, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_01_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-4/11_icon_W_Conf_Ist_Rnd_Mavericks_at_Clippers.png
try:
    _c11 = get_crop(11, 288, 168)
    canvas.paste(_c11, (864, 2792), _c11)
except Exception:
    pass
layout["W_Conf_Ist_Rnd:_Mavericks"] = [864, 2792, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_01_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-4/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 52, 63)
    canvas.paste(_c12, (1319, 2), _c12)
except Exception:
    pass
layout["icon_12"] = [1319, 2, 1371, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_01_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-4/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 46, 66)
    canvas.paste(_c13, (1154, 0), _c13)
except Exception:
    pass
layout["icon_13"] = [1154, 0, 1200, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_01_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-4/14_icon_S236.png
try:
    _c14 = get_crop(14, 462, 533)
    canvas.paste(_c14, (546, 1431), _c14)
except Exception:
    pass
layout["S236+"] = [546, 1431, 1008, 1964]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_01_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-4/15_icon_May.png
try:
    _c15 = get_crop(15, 264, 183)
    canvas.paste(_c15, (1176, 2014), _c15)
except Exception:
    pass
layout["May"] = [1176, 2014, 1440, 2197]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_01_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-4/16_icon_W_Conf_Ist_Rnd_Mavericks_at_Clippers.png
try:
    _c16 = get_crop(16, 288, 168)
    canvas.paste(_c16, (288, 2792), _c16)
except Exception:
    pass
layout["W_Conf_Ist_Rnd:_Mavericks"] = [288, 2792, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_01_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-4/17_icon_icon_17.png
try:
    _c17 = get_crop(17, 101, 116)
    canvas.paste(_c17, (1339, 2707), _c17)
except Exception:
    pass
layout["icon_17"] = [1339, 2707, 1440, 2823]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_01_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-4/18_icon_Browse.png
try:
    _c18 = get_crop(18, 288, 162)
    canvas.paste(_c18, (0, 2792), _c18)
except Exception:
    pass
layout["Browse"] = [0, 2792, 288, 2954]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_01_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-4/19_icon_S236.png
try:
    _c19 = get_crop(19, 462, 533)
    canvas.paste(_c19, (546, 1431), _c19)
except Exception:
    pass
layout["S236+"] = [546, 1431, 1008, 1964]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_01_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-4/20_icon_icon_20.png
try:
    _c20 = get_crop(20, 53, 57)
    canvas.paste(_c20, (316, 6), _c20)
except Exception:
    pass
layout["icon_20"] = [316, 6, 369, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_01_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-4/21_icon_Account.png
try:
    _c21 = get_crop(21, 288, 168)
    canvas.paste(_c21, (1152, 2792), _c21)
except Exception:
    pass
layout["Account"] = [1152, 2792, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_01_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-4/22_icon_icon_22.png
try:
    _c22 = get_crop(22, 116, 127)
    canvas.paste(_c22, (1138, 2495), _c22)
except Exception:
    pass
layout["icon_22"] = [1138, 2495, 1254, 2622]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_01_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-4/23_icon_W_Conf_Ist_Rnd_Mavericks_at_Clippers.png
try:
    _c23 = get_crop(23, 288, 168)
    canvas.paste(_c23, (576, 2792), _c23)
except Exception:
    pass
layout["W_Conf_Ist_Rnd:_Mavericks"] = [576, 2792, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_01_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-4/24_text_Los_Angeles_CA.png
try:
    _c24 = get_crop(24, 459, 80)
    canvas.paste(_c24, (44, 132), _c24)
except Exception:
    pass
layout["Los_Angeles,_CA"] = [44, 132, 503, 212]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_01_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-4/25_text_Just_for_you.png
try:
    _c25 = get_crop(25, 306, 66)
    canvas.paste(_c25, (38, 1310), _c25)
except Exception:
    pass
layout["Just_for_you"] = [38, 1310, 344, 1376]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_01_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-4/26_text_View_all.png
try:
    _c26 = get_crop(26, 264, 183)
    canvas.paste(_c26, (1176, 1248), _c26)
except Exception:
    pass
layout["View_all"] = [1176, 1248, 1440, 1431]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_01_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-4/27_text_W_Conf_Ist_Rnd_Mavericks_at_Clippers.png
try:
    _c27 = get_crop(27, 288, 168)
    canvas.paste(_c27, (576, 2792), _c27)
except Exception:
    pass
layout["W_Conf_Ist_Rnd:_Mavericks"] = [576, 2792, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_01_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-4/28_clickable_Tracking.png
try:
    _c28 = get_crop(28, 72, 72)
    canvas.paste(_c28, (408, 1455), _c28)
except Exception:
    pass
layout["Tracking"] = [408, 1455, 480, 1527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_01_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-4/29_clickable_Tracking.png
try:
    _c29 = get_crop(29, 72, 72)
    canvas.paste(_c29, (906, 1455), _c29)
except Exception:
    pass
layout["Tracking"] = [906, 1455, 978, 1527]
