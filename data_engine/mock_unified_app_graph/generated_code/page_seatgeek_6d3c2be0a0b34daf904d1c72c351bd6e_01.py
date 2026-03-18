# page_id: page_seatgeek_6d3c2be0a0b34daf904d1c72c351bd6e_01
# screenshot: 2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-4.png
# step_index: 1/9
# task: Open SeatGeek. Look up "Phoenix Suns" tickets for next upcoming event. Which section are tickets in?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Background base
draw.rectangle([(0, 0), (1440, 2960)], fill=(250, 250, 250))

# Status bar (top ~72px)
status_h = 72
draw.rectangle([(0, 0), (1440, status_h)], fill=(238, 238, 238))
# subtle bottom divider under status bar
draw.line([(0, status_h), (1440, status_h)], fill=(225, 225, 225), width=1)

# Header / toolbar area (below status)
header_top = status_h
header_h = 88
draw.rectangle([(0, header_top), (1440, header_top + header_h)], fill=(255, 255, 255))
# header bottom divider
draw.line([(24, header_top + header_h), (1440 - 24, header_top + header_h)], fill=(235, 235, 235), width=1)

# Big page spacing (leave hero area untouched) - subtle shadow under hero area region
# Hero card in screenshot is at y ~360..1200 (detected elsewhere) - don't draw over it.
# Draw a soft, wide horizontal divider under hero area to separate from sections.
hero_bottom = 1200  # avoid drawing into hero
draw.line([(24, hero_bottom + 8), (1440 - 24, hero_bottom + 8)], fill=(240, 240, 240), width=1)

# "Just for you" section background card (rounded rectangle)
just_card_top = 1280
just_card_bottom = 2000
card_pad = 24
card_coords = (card_pad, just_card_top, 1440 - card_pad, just_card_bottom)
draw.rounded_rectangle(card_coords, radius=20, fill=(255, 255, 255), outline=None)

# subtle shadow line under the "Just for you" card
draw.line([(card_pad + 8, just_card_bottom), (1440 - card_pad - 8, just_card_bottom)], fill=(235, 235, 235), width=1)

# "Trending events" list container background
trending_top = just_card_bottom + 24
trending_bottom = 2680
trend_pad = 24
trend_coords = (trend_pad, trending_top, 1440 - trend_pad, trending_bottom)
draw.rounded_rectangle(trend_coords, radius=14, fill=(255, 255, 255), outline=None)

# separators between trending items (aligned to detected item positions)
# Using exact y positions based on detected crops:
first_item_y = 2183
second_item_y = 2419
third_item_bottom = 2655  # approximate end of third item
# draw separators across the interior of trending container
sep_left = trend_pad + 16
sep_right = 1440 - trend_pad - 16
# Separator before first item (subtle)
draw.line([(sep_left, first_item_y - 24), (sep_right, first_item_y - 24)], fill=(245, 245, 245), width=1)
# Separator between item1 and item2
draw.line([(sep_left, second_item_y), (sep_right, second_item_y)], fill=(235, 235, 235), width=1)
# Separator between item2 and item3
draw.line([(sep_left, third_item_bottom), (sep_right, third_item_bottom)], fill=(235, 235, 235), width=1)

# Light left inset area for trending item icons (visual structure only)
# (Do not draw icons or text; just guide boxes/backgrounds)
icon_box_w = 96
icon_box_h = 96
icon_x = trend_pad + 24
# place faint rounded rect placeholders for three items (background only)
draw.rounded_rectangle((icon_x, first_item_y - 36, icon_x + icon_box_w, first_item_y + icon_box_h - 36),
                       radius=12, fill=(252, 240, 240))
draw.rounded_rectangle((icon_x, second_item_y - 36, icon_x + icon_box_w, second_item_y + icon_box_h - 36),
                       radius=12, fill=(252, 240, 240))
draw.rounded_rectangle((icon_x, third_item_bottom - 236, icon_x + icon_box_w, third_item_bottom - 140),
                       radius=12, fill=(252, 240, 240))

# subtle right-side badges background (do not draw numbers or overlays)
badge_w = 96
badge_x = 1440 - trend_pad - badge_w - 16
# faint circular backgrounds to indicate rank badges (structure only)
draw.ellipse((badge_x, first_item_y - 20, badge_x + badge_w, first_item_y + badge_w - 20), fill=(255, 244, 244))
draw.ellipse((badge_x, second_item_y - 20, badge_x + badge_w, second_item_y + badge_w - 20), fill=(255, 244, 244))
draw.ellipse((badge_x, third_item_bottom - 236, badge_x + badge_w, third_item_bottom - 140), fill=(255, 244, 244))

# Bottom navigation bar background (leave icons to be pasted on top)
nav_top = 2792
nav_bottom = 2960
draw.rectangle([(0, nav_top), (1440, nav_bottom)], fill=(255, 255, 255))
# top divider for nav bar
draw.line([(24, nav_top), (1440 - 24, nav_top)], fill=(235, 235, 235), width=1)

# small accent: left floating filter pill at header right (background only, no icon)
pill_w = 64
pill_h = 40
pill_x = 1440 - 24 - pill_w
pill_y = header_top + (header_h - pill_h) // 2
draw.rounded_rectangle((pill_x, pill_y, pill_x + pill_w, pill_y + pill_h), radius=10, fill=(255, 255, 255), outline=(230,230,230))

# final subtle vignette shadow under major sections to add depth (very light)
draw.rectangle([(0, hero_bottom + 2), (1440, hero_bottom + 6)], fill=(245, 245, 245))
draw.rectangle([(0, just_card_bottom + 2), (1440, just_card_bottom + 6)], fill=(245, 245, 245))

# NOTE: Do not draw any icons/text that correspond to detected elements.
# This file only establishes background, containers, dividers and structure.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_01_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-4/00_icon_Knicks.png
try:
    _c0 = get_crop(0, 1344, 840)
    canvas.paste(_c0, (48, 360), _c0)
except Exception:
    pass
layout["Knicks"] = [48, 360, 1392, 1200]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_01_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-4/01_icon_BOOK_OF.png
try:
    _c1 = get_crop(1, 462, 519)
    canvas.paste(_c1, (48, 1431), _c1)
except Exception:
    pass
layout["BOOK_OF"] = [48, 1431, 510, 1950]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_01_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-4/02_icon_August_Wilson_Theatre.png
try:
    _c2 = get_crop(2, 1309, 236)
    canvas.paste(_c2, (0, 2183), _c2)
except Exception:
    pass
layout["August_Wilson_Theatre"] = [0, 2183, 1309, 2419]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_01_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-4/03_icon_Yankee_Stadium.png
try:
    _c3 = get_crop(3, 1309, 236)
    canvas.paste(_c3, (0, 2419), _c3)
except Exception:
    pass
layout["Yankee_Stadium"] = [0, 2419, 1309, 2655]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_01_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-4/04_icon_S116.png
try:
    _c4 = get_crop(4, 396, 519)
    canvas.paste(_c4, (1044, 1431), _c4)
except Exception:
    pass
layout["S116+"] = [1044, 1431, 1440, 1950]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_01_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-4/05_icon_S94.png
try:
    _c5 = get_crop(5, 462, 519)
    canvas.paste(_c5, (546, 1431), _c5)
except Exception:
    pass
layout["S94+"] = [546, 1431, 1008, 1950]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_01_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-4/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 99, 152)
    canvas.paste(_c6, (1341, 2464), _c6)
except Exception:
    pass
layout["icon_6"] = [1341, 2464, 1440, 2616]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_01_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-4/07_icon_View_all.png
try:
    _c7 = get_crop(7, 98, 149)
    canvas.paste(_c7, (1342, 2228), _c7)
except Exception:
    pass
layout["View_all"] = [1342, 2228, 1440, 2377]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_01_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-4/08_icon_New_York_NY.png
try:
    _c8 = get_crop(8, 61, 58)
    canvas.paste(_c8, (243, 5), _c8)
except Exception:
    pass
layout["New_York,_NY"] = [243, 5, 304, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_01_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-4/09_icon_May.png
try:
    _c9 = get_crop(9, 264, 183)
    canvas.paste(_c9, (1176, 2000), _c9)
except Exception:
    pass
layout["May"] = [1176, 2000, 1440, 2183]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_01_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-4/10_icon_888.png
try:
    _c10 = get_crop(10, 99, 63)
    canvas.paste(_c10, (1214, 1), _c10)
except Exception:
    pass
layout["888"] = [1214, 1, 1313, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_01_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-4/11_icon_7_06_my.png
try:
    _c11 = get_crop(11, 55, 58)
    canvas.paste(_c11, (114, 4), _c11)
except Exception:
    pass
layout["7:06_my"] = [114, 4, 169, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_01_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-4/12_icon_E_Conf_Ist_Rnd_76ers_at_Knicks_Gm_2_H.png
try:
    _c12 = get_crop(12, 288, 168)
    canvas.paste(_c12, (864, 2792), _c12)
except Exception:
    pass
layout["E_Conf_Ist_Rnd:_76ers_at_"] = [864, 2792, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_01_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-4/13_icon_888.png
try:
    _c13 = get_crop(13, 144, 240)
    canvas.paste(_c13, (1260, 72), _c13)
except Exception:
    pass
layout["888"] = [1260, 72, 1404, 312]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_01_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-4/14_icon_7_06_my.png
try:
    _c14 = get_crop(14, 47, 57)
    canvas.paste(_c14, (185, 5), _c14)
except Exception:
    pass
layout["7:06_my"] = [185, 5, 232, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_01_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-4/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 50, 63)
    canvas.paste(_c15, (1320, 2), _c15)
except Exception:
    pass
layout["icon_15"] = [1320, 2, 1370, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_01_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-4/16_icon_E_Conf_Ist_Rnd_76ers_at_Knicks_Gm_2_H.png
try:
    _c16 = get_crop(16, 288, 168)
    canvas.paste(_c16, (288, 2792), _c16)
except Exception:
    pass
layout["E_Conf_Ist_Rnd:_76ers_at_"] = [288, 2792, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_01_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-4/17_icon_E_Conf_Ist_Rnd_76ers_at_Knicks_Gm_2_H.png
try:
    _c17 = get_crop(17, 288, 168)
    canvas.paste(_c17, (576, 2792), _c17)
except Exception:
    pass
layout["E_Conf_Ist_Rnd:_76ers_at_"] = [576, 2792, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_01_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-4/18_icon_icon_18.png
try:
    _c18 = get_crop(18, 54, 59)
    canvas.paste(_c18, (314, 5), _c18)
except Exception:
    pass
layout["icon_18"] = [314, 5, 368, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_01_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-4/19_icon_icon_19.png
try:
    _c19 = get_crop(19, 46, 64)
    canvas.paste(_c19, (1154, 1), _c19)
except Exception:
    pass
layout["icon_19"] = [1154, 1, 1200, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_01_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-4/20_icon_icon_20.png
try:
    _c20 = get_crop(20, 99, 119)
    canvas.paste(_c20, (1341, 2698), _c20)
except Exception:
    pass
layout["icon_20"] = [1341, 2698, 1440, 2817]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_01_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-4/21_icon_Browse.png
try:
    _c21 = get_crop(21, 288, 162)
    canvas.paste(_c21, (0, 2792), _c21)
except Exception:
    pass
layout["Browse"] = [0, 2792, 288, 2954]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_01_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-4/22_icon_Account.png
try:
    _c22 = get_crop(22, 288, 168)
    canvas.paste(_c22, (1152, 2792), _c22)
except Exception:
    pass
layout["Account"] = [1152, 2792, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_01_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-4/23_icon_Andrew_Schulz.png
try:
    _c23 = get_crop(23, 462, 519)
    canvas.paste(_c23, (546, 1431), _c23)
except Exception:
    pass
layout["Andrew_Schulz"] = [546, 1431, 1008, 1950]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_01_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-4/24_icon_icon_24.png
try:
    _c24 = get_crop(24, 116, 127)
    canvas.paste(_c24, (1138, 2484), _c24)
except Exception:
    pass
layout["icon_24"] = [1138, 2484, 1254, 2611]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_01_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-4/25_icon_New_York_NY.png
try:
    _c25 = get_crop(25, 390, 86)
    canvas.paste(_c25, (40, 119), _c25)
except Exception:
    pass
layout["New_York,_NY"] = [40, 119, 430, 205]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_01_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-4/26_icon_The.png
try:
    _c26 = get_crop(26, 91, 102)
    canvas.paste(_c26, (36, 1427), _c26)
except Exception:
    pass
layout["The"] = [36, 1427, 127, 1529]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_01_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-4/27_text_date.png
try:
    _c27 = get_crop(27, 114, 52)
    canvas.paste(_c27, (137, 208), _c27)
except Exception:
    pass
layout["date"] = [137, 208, 251, 260]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_01_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-4/28_text_Just_for_you.png
try:
    _c28 = get_crop(28, 306, 66)
    canvas.paste(_c28, (38, 1310), _c28)
except Exception:
    pass
layout["Just_for_you"] = [38, 1310, 344, 1376]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_01_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-4/29_text_View_all.png
try:
    _c29 = get_crop(29, 264, 183)
    canvas.paste(_c29, (1176, 1248), _c29)
except Exception:
    pass
layout["View_all"] = [1176, 1248, 1440, 1431]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_01_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-4/30_text_E_Conf_Ist_Rnd_76ers_at_Knicks_Gm_2_H.png
try:
    _c30 = get_crop(30, 288, 168)
    canvas.paste(_c30, (576, 2792), _c30)
except Exception:
    pass
layout["E_Conf_Ist_Rnd:_76ers_at_"] = [576, 2792, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_01_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-4/31_clickable_Tracking.png
try:
    _c31 = get_crop(31, 72, 72)
    canvas.paste(_c31, (408, 1455), _c31)
except Exception:
    pass
layout["Tracking"] = [408, 1455, 480, 1527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_01_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-4/32_clickable_Tracking.png
try:
    _c32 = get_crop(32, 72, 72)
    canvas.paste(_c32, (906, 1455), _c32)
except Exception:
    pass
layout["Tracking"] = [906, 1455, 978, 1527]
