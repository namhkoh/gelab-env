# page_id: page_seatgeek_1e6c9e893d9e4bc99959744188677162_01
# screenshot: 2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-4.png
# step_index: 1/8
# task: Open SeatGeek. Search "Radio City Music Hall" and then add the venue to favorite. Who are the performers of the top recommended event?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Background fill (slightly warm white)
draw.rectangle((0, 0, 1440, 2960), fill="#FBFBFB")

# Status bar area (top ~72px) - subtle gray background (do not draw icons)
status_h = 72
draw.rectangle((0, 0, 1440, status_h), fill="#F2F3F5")
# thin separator under status bar
draw.line((20, status_h - 1, 1420, status_h - 1), fill="#E6E6E8", width=1)

# Header area (location + date) - white panel with bottom divider
header_top = status_h
header_bottom = 192
draw.rectangle((0, header_top, 1440, header_bottom), fill="#FFFFFF")
draw.line((24, header_bottom, 1416, header_bottom), fill="#E8E8E8", width=1)

# Big hero card region is intentionally left blank (content will be pasted).
# But add a subtle faint drop shadow line under where that hero card would sit
# Hero card detected at y=360 with height 840 => bottom ~1200
hero_bottom = 360 + 840
shadow_y = hero_bottom + 6
draw.line((24, shadow_y, 1416, shadow_y), fill="#F0F1F3", width=2)

# Divider directly under hero card to separate sections
draw.line((24, hero_bottom + 12, 1416, hero_bottom + 12), fill="#E9E9EA", width=1)

# "Just for you" section container (rounded white card background)
just_for_you_top = hero_bottom + 24   # ~1224
just_for_you_bottom = 1700
draw.rounded_rectangle(
    (24, just_for_you_top, 1416, just_for_you_bottom),
    radius=12,
    fill="#FFFFFF",
    outline="#F0F0F1",
    width=1
)

# Subtle inner divider between "Just for you" card contents (so pasted items sit on top)
draw.line((40, just_for_you_top + 120, 1400, just_for_you_top + 120), fill="#F2F2F3", width=1)

# Separator between sections
sep_y = just_for_you_bottom + 24
draw.line((24, sep_y, 1416, sep_y), fill="#EAEAEA", width=1)

# "Trending events" section header area left blank for text, draw container below for list
trending_top = sep_y + 24   # roughly where "Trending events" sits
trending_bottom = 2680
draw.rounded_rectangle(
    (24, trending_top, 1416, trending_bottom),
    radius=12,
    fill="#FFFFFF",
    outline="#F0F0F1",
    width=1
)

# Horizontal separators for list rows inside trending area (light)
row_start_x = 40
row_end_x = 1400
# approximate three rows visible in screenshot; draw separators at approximate positions
row_y_positions = [trending_top + 120, trending_top + 240, trending_top + 360, trending_top + 480]
for y in row_y_positions:
    draw.line((row_start_x, y, row_end_x, y), fill="#F2F2F3", width=1)

# Slight shadow line above trending container to separate from content above
draw.line((24, trending_top - 6, 1416, trending_top - 6), fill="#F3F4F5", width=2)

# Bottom navigation bar background (do not draw icons)
nav_top = 2792
draw.rectangle((0, nav_top, 1440, 2960), fill="#FFFFFF")
# top divider for nav bar
draw.line((24, nav_top, 1416, nav_top), fill="#E7E7E8", width=1)

# Small subtle left and right padding accent lines to mimic app chrome (decorative only)
draw.line((12, status_h + 8, 12, 200), fill="#F5F5F6", width=2)
draw.line((1428, status_h + 8, 1428, 200), fill="#F5F5F6", width=2)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_01_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-4/00_icon_Globe_Life_Field.png
try:
    _c0 = get_crop(0, 1309, 236)
    canvas.paste(_c0, (0, 2197), _c0)
except Exception:
    pass
layout["Globe_Life_Field"] = [0, 2197, 1309, 2433]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_01_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-4/01_icon_Mavericks.png
try:
    _c1 = get_crop(1, 1344, 840)
    canvas.paste(_c1, (48, 360), _c1)
except Exception:
    pass
layout["Mavericks"] = [48, 360, 1392, 1200]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_01_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-4/02_icon_View_all.png
try:
    _c2 = get_crop(2, 95, 148)
    canvas.paste(_c2, (1345, 2244), _c2)
except Exception:
    pass
layout["View_all"] = [1345, 2244, 1440, 2392]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_01_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-4/03_icon_9_8_PM.png
try:
    _c3 = get_crop(3, 462, 533)
    canvas.paste(_c3, (48, 1431), _c3)
except Exception:
    pass
layout["9,8_PM"] = [48, 1431, 510, 1964]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_01_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-4/04_icon_icon_4.png
try:
    _c4 = get_crop(4, 102, 147)
    canvas.paste(_c4, (1338, 2481), _c4)
except Exception:
    pass
layout["icon_4"] = [1338, 2481, 1440, 2628]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_01_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-4/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 63, 58)
    canvas.paste(_c5, (242, 5), _c5)
except Exception:
    pass
layout["icon_5"] = [242, 5, 305, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_01_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-4/06_icon_888.png
try:
    _c6 = get_crop(6, 101, 64)
    canvas.paste(_c6, (1213, 1), _c6)
except Exception:
    pass
layout["888"] = [1213, 1, 1314, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_01_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-4/07_icon_Tracking.png
try:
    _c7 = get_crop(7, 288, 168)
    canvas.paste(_c7, (864, 2792), _c7)
except Exception:
    pass
layout["Tracking"] = [864, 2792, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_01_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-4/08_icon_American_Airlines_Center.png
try:
    _c8 = get_crop(8, 1309, 236)
    canvas.paste(_c8, (0, 2433), _c8)
except Exception:
    pass
layout["American_Airlines_Center"] = [0, 2433, 1309, 2669]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_01_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-4/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 55, 58)
    canvas.paste(_c9, (314, 5), _c9)
except Exception:
    pass
layout["icon_9"] = [314, 5, 369, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_01_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-4/10_icon_8.32_Wy.png
try:
    _c10 = get_crop(10, 56, 58)
    canvas.paste(_c10, (113, 4), _c10)
except Exception:
    pass
layout["8.32_Wy"] = [113, 4, 169, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_01_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-4/11_icon_888.png
try:
    _c11 = get_crop(11, 144, 240)
    canvas.paste(_c11, (1260, 72), _c11)
except Exception:
    pass
layout["888"] = [1260, 72, 1404, 312]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_01_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-4/12_icon_8.32_Wy.png
try:
    _c12 = get_crop(12, 48, 57)
    canvas.paste(_c12, (184, 5), _c12)
except Exception:
    pass
layout["8.32_Wy"] = [184, 5, 232, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_01_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-4/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 52, 63)
    canvas.paste(_c13, (1319, 2), _c13)
except Exception:
    pass
layout["icon_13"] = [1319, 2, 1371, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_01_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-4/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 47, 65)
    canvas.paste(_c14, (1154, 1), _c14)
except Exception:
    pass
layout["icon_14"] = [1154, 1, 1201, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_01_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-4/15_icon_S159.png
try:
    _c15 = get_crop(15, 462, 533)
    canvas.paste(_c15, (48, 1431), _c15)
except Exception:
    pass
layout["S159"] = [48, 1431, 510, 1964]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_01_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-4/16_icon_Account.png
try:
    _c16 = get_crop(16, 288, 168)
    canvas.paste(_c16, (1152, 2792), _c16)
except Exception:
    pass
layout["Account"] = [1152, 2792, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_01_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-4/17_icon_icon_17.png
try:
    _c17 = get_crop(17, 115, 130)
    canvas.paste(_c17, (1141, 2490), _c17)
except Exception:
    pass
layout["icon_17"] = [1141, 2490, 1256, 2620]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_01_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-4/18_icon_W_Conf_Ist_Rnd_Clippers_at_Mavericks.png
try:
    _c18 = get_crop(18, 288, 168)
    canvas.paste(_c18, (288, 2792), _c18)
except Exception:
    pass
layout["W_Conf_Ist_Rnd:_Clippers_"] = [288, 2792, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_01_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-4/19_icon_Browse.png
try:
    _c19 = get_crop(19, 288, 162)
    canvas.paste(_c19, (0, 2792), _c19)
except Exception:
    pass
layout["Browse"] = [0, 2792, 288, 2954]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_01_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-4/20_icon_icon_20.png
try:
    _c20 = get_crop(20, 98, 111)
    canvas.paste(_c20, (1342, 2708), _c20)
except Exception:
    pass
layout["icon_20"] = [1342, 2708, 1440, 2819]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_01_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-4/21_icon_W_Conf_Ist_Rnd_Clippers_at_Mavericks.png
try:
    _c21 = get_crop(21, 288, 168)
    canvas.paste(_c21, (576, 2792), _c21)
except Exception:
    pass
layout["W_Conf_Ist_Rnd:_Clippers_"] = [576, 2792, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_01_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-4/22_text_Dallas_TX.png
try:
    _c22 = get_crop(22, 295, 76)
    canvas.paste(_c22, (41, 129), _c22)
except Exception:
    pass
layout["Dallas,_TX"] = [41, 129, 336, 205]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_01_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-4/23_text_date.png
try:
    _c23 = get_crop(23, 114, 52)
    canvas.paste(_c23, (137, 208), _c23)
except Exception:
    pass
layout["date"] = [137, 208, 251, 260]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_01_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-4/24_text_Just_for_you.png
try:
    _c24 = get_crop(24, 306, 66)
    canvas.paste(_c24, (38, 1310), _c24)
except Exception:
    pass
layout["Just_for_you"] = [38, 1310, 344, 1376]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_01_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-4/25_text_View_all.png
try:
    _c25 = get_crop(25, 264, 183)
    canvas.paste(_c25, (1176, 1248), _c25)
except Exception:
    pass
layout["View_all"] = [1176, 1248, 1440, 1431]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_01_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-4/26_text_Trending_events.png
try:
    _c26 = get_crop(26, 423, 81)
    canvas.paste(_c26, (38, 2068), _c26)
except Exception:
    pass
layout["Trending_events"] = [38, 2068, 461, 2149]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_01_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-4/27_text_View_all.png
try:
    _c27 = get_crop(27, 264, 183)
    canvas.paste(_c27, (1176, 2014), _c27)
except Exception:
    pass
layout["View_all"] = [1176, 2014, 1440, 2197]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_01_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-4/28_text_W_Conf_Ist_Rnd_Clippers_at_Mavericks.png
try:
    _c28 = get_crop(28, 288, 168)
    canvas.paste(_c28, (576, 2792), _c28)
except Exception:
    pass
layout["W_Conf_Ist_Rnd:_Clippers_"] = [576, 2792, 864, 2960]
