# page_id: page_seatgeek_3297e3a54de8487d8c2b67798919ed2f_01
# screenshot: 2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-4.png
# step_index: 1/11
# task: Open SeatGeek. Search "Comedy Show in Los Angeles". Find the top recommendation. When is the event?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw overall background (dominant color: white)
draw.rectangle((0, 0, 1440, 2960), fill="#ffffff")

# Status bar area (light gray)
status_h = 64
draw.rectangle((0, 0, 1440, status_h), fill="#f3f4f5")

# Header / toolbar area (white) and bottom divider
header_top = status_h
header_bottom = 240
draw.rectangle((0, header_top, 1440, header_bottom), fill="#ffffff")
draw.line((24, header_bottom, 1440-24, header_bottom), fill="#e6e6e6", width=2)

# Right-side vertical accent for the hero carousel (keeps outside hero card bbox)
# Hero card bbox: (48,360)-(1392,1200) so draw accent just at the far right edge
draw.rounded_rectangle((1396, 360, 1440, 1200), radius=6, fill="#c89b2b")

# Subtle content-area shadow under header to separate header from content
draw.line((24, 260, 1440-24, 260), fill="#f0f0f0", width=1)

# "Just for you" section background (card container area)
just_top = 1200
just_bottom = 1998  # end of the horizontal scroll area region
draw.rounded_rectangle((24, just_top, 1440-24, just_bottom), radius=12, fill="#ffffff", outline=None)

# Divider below the horizontal "Just for you" carousel area
draw.line((24, just_bottom + 6, 1440-24, just_bottom + 6), fill="#ededed", width=2)

# Trending events container background
trending_top = just_bottom + 32
trending_bottom = 2720
draw.rounded_rectangle((24, trending_top, 1440-24, trending_bottom), radius=12, fill="#ffffff", outline=None)

# Horizontal separators between trending event rows (light separators)
sep_x0 = 32
sep_x1 = 1440 - 32
# approximate separators that align with rows (kept subtle)
draw.line((sep_x0, 2100, sep_x1, 2100), fill="#f1f1f1", width=2)
draw.line((sep_x0, 2330, sep_x1, 2330), fill="#f1f1f1", width=2)
draw.line((sep_x0, 2560, sep_x1, 2560), fill="#f1f1f1", width=2)

# Thin left padding guide lines (very subtle) for content alignment (not text/icons)
draw.line((40, header_bottom + 12, 40, trending_bottom - 12), fill="#ffffff", width=1)
draw.line((1400, header_bottom + 12, 1400, trending_bottom - 12), fill="#ffffff", width=1)

# Bottom navigation bar background and top border shadow
nav_top = 2792
nav_bottom = 2960
draw.rectangle((0, nav_top, 1440, nav_bottom), fill="#ffffff")
draw.line((24, nav_top, 1440-24, nav_top), fill="#e8e8e8", width=2)
# slight top shadow above nav (soft)
for i, a in enumerate([1, 2, 4]):
    alpha_y = nav_top - (6 + i*2)
    if alpha_y > 0:
        draw.line((24, alpha_y, 1440-24, alpha_y), fill="#f6f6f6", width=1)

# Subtle overall vignette edges (very light) to match screenshot depth (not covering content)
edge_strip = 18
draw.rectangle((0, 0, edge_strip, 2960), fill="#ffffff")
draw.rectangle((1440-edge_strip, 0, 1440, 2960), fill="#ffffff")
draw.rectangle((0, 0, 1440, edge_strip), fill="#ffffff")
draw.rectangle((0, 2960-edge_strip, 1440, 2960), fill="#ffffff")

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_01_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-4/00_icon_Knicks.png
try:
    _c0 = get_crop(0, 1344, 840)
    canvas.paste(_c0, (48, 360), _c0)
except Exception:
    pass
layout["Knicks"] = [48, 360, 1392, 1200]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_01_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-4/01_icon_BOOK_OF.png
try:
    _c1 = get_crop(1, 462, 519)
    canvas.paste(_c1, (48, 1431), _c1)
except Exception:
    pass
layout["BOOK_OF"] = [48, 1431, 510, 1950]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_01_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-4/02_icon_August_Wilson_Theatre.png
try:
    _c2 = get_crop(2, 1309, 236)
    canvas.paste(_c2, (0, 2183), _c2)
except Exception:
    pass
layout["August_Wilson_Theatre"] = [0, 2183, 1309, 2419]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_01_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-4/03_icon_Yankee_Stadium.png
try:
    _c3 = get_crop(3, 1309, 236)
    canvas.paste(_c3, (0, 2419), _c3)
except Exception:
    pass
layout["Yankee_Stadium"] = [0, 2419, 1309, 2655]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_01_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-4/04_icon_S116.png
try:
    _c4 = get_crop(4, 396, 519)
    canvas.paste(_c4, (1044, 1431), _c4)
except Exception:
    pass
layout["S116+"] = [1044, 1431, 1440, 1950]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_01_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-4/05_icon_S94.png
try:
    _c5 = get_crop(5, 462, 519)
    canvas.paste(_c5, (546, 1431), _c5)
except Exception:
    pass
layout["S94+"] = [546, 1431, 1008, 1950]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_01_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-4/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 99, 152)
    canvas.paste(_c6, (1341, 2464), _c6)
except Exception:
    pass
layout["icon_6"] = [1341, 2464, 1440, 2616]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_01_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-4/07_icon_View_all.png
try:
    _c7 = get_crop(7, 98, 149)
    canvas.paste(_c7, (1342, 2228), _c7)
except Exception:
    pass
layout["View_all"] = [1342, 2228, 1440, 2377]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_01_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-4/08_icon_New_York_NY.png
try:
    _c8 = get_crop(8, 61, 58)
    canvas.paste(_c8, (243, 5), _c8)
except Exception:
    pass
layout["New_York,_NY"] = [243, 5, 304, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_01_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-4/09_icon_May.png
try:
    _c9 = get_crop(9, 264, 183)
    canvas.paste(_c9, (1176, 2000), _c9)
except Exception:
    pass
layout["May"] = [1176, 2000, 1440, 2183]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_01_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-4/10_icon_888.png
try:
    _c10 = get_crop(10, 99, 63)
    canvas.paste(_c10, (1214, 1), _c10)
except Exception:
    pass
layout["888"] = [1214, 1, 1313, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_01_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-4/11_icon_7_09_my.png
try:
    _c11 = get_crop(11, 54, 58)
    canvas.paste(_c11, (115, 4), _c11)
except Exception:
    pass
layout["7:09_my"] = [115, 4, 169, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_01_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-4/12_icon_E_Conf_Ist_Rnd_76ers_at_Knicks_Gm_2_H.png
try:
    _c12 = get_crop(12, 288, 168)
    canvas.paste(_c12, (864, 2792), _c12)
except Exception:
    pass
layout["E_Conf_Ist_Rnd:_76ers_at_"] = [864, 2792, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_01_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-4/13_icon_888.png
try:
    _c13 = get_crop(13, 144, 240)
    canvas.paste(_c13, (1260, 72), _c13)
except Exception:
    pass
layout["888"] = [1260, 72, 1404, 312]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_01_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-4/14_icon_7_09_my.png
try:
    _c14 = get_crop(14, 47, 57)
    canvas.paste(_c14, (185, 5), _c14)
except Exception:
    pass
layout["7:09_my"] = [185, 5, 232, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_01_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-4/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 50, 63)
    canvas.paste(_c15, (1320, 2), _c15)
except Exception:
    pass
layout["icon_15"] = [1320, 2, 1370, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_01_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-4/16_icon_E_Conf_Ist_Rnd_76ers_at_Knicks_Gm_2_H.png
try:
    _c16 = get_crop(16, 288, 168)
    canvas.paste(_c16, (288, 2792), _c16)
except Exception:
    pass
layout["E_Conf_Ist_Rnd:_76ers_at_"] = [288, 2792, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_01_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-4/17_icon_E_Conf_Ist_Rnd_76ers_at_Knicks_Gm_2_H.png
try:
    _c17 = get_crop(17, 288, 168)
    canvas.paste(_c17, (576, 2792), _c17)
except Exception:
    pass
layout["E_Conf_Ist_Rnd:_76ers_at_"] = [576, 2792, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_01_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-4/18_icon_icon_18.png
try:
    _c18 = get_crop(18, 54, 59)
    canvas.paste(_c18, (314, 5), _c18)
except Exception:
    pass
layout["icon_18"] = [314, 5, 368, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_01_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-4/19_icon_icon_19.png
try:
    _c19 = get_crop(19, 46, 64)
    canvas.paste(_c19, (1154, 1), _c19)
except Exception:
    pass
layout["icon_19"] = [1154, 1, 1200, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_01_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-4/20_icon_icon_20.png
try:
    _c20 = get_crop(20, 99, 119)
    canvas.paste(_c20, (1341, 2698), _c20)
except Exception:
    pass
layout["icon_20"] = [1341, 2698, 1440, 2817]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_01_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-4/21_icon_Browse.png
try:
    _c21 = get_crop(21, 288, 162)
    canvas.paste(_c21, (0, 2792), _c21)
except Exception:
    pass
layout["Browse"] = [0, 2792, 288, 2954]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_01_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-4/22_icon_Account.png
try:
    _c22 = get_crop(22, 288, 168)
    canvas.paste(_c22, (1152, 2792), _c22)
except Exception:
    pass
layout["Account"] = [1152, 2792, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_01_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-4/23_icon_Andrew_Schulz.png
try:
    _c23 = get_crop(23, 462, 519)
    canvas.paste(_c23, (546, 1431), _c23)
except Exception:
    pass
layout["Andrew_Schulz"] = [546, 1431, 1008, 1950]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_01_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-4/24_icon_icon_24.png
try:
    _c24 = get_crop(24, 116, 127)
    canvas.paste(_c24, (1138, 2484), _c24)
except Exception:
    pass
layout["icon_24"] = [1138, 2484, 1254, 2611]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_01_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-4/25_icon_New_York_NY.png
try:
    _c25 = get_crop(25, 390, 86)
    canvas.paste(_c25, (40, 119), _c25)
except Exception:
    pass
layout["New_York,_NY"] = [40, 119, 430, 205]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_01_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-4/26_icon_The.png
try:
    _c26 = get_crop(26, 91, 102)
    canvas.paste(_c26, (36, 1427), _c26)
except Exception:
    pass
layout["The"] = [36, 1427, 127, 1529]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_01_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-4/27_text_date.png
try:
    _c27 = get_crop(27, 114, 52)
    canvas.paste(_c27, (137, 208), _c27)
except Exception:
    pass
layout["date"] = [137, 208, 251, 260]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_01_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-4/28_text_Just_for_you.png
try:
    _c28 = get_crop(28, 306, 66)
    canvas.paste(_c28, (38, 1310), _c28)
except Exception:
    pass
layout["Just_for_you"] = [38, 1310, 344, 1376]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_01_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-4/29_text_View_all.png
try:
    _c29 = get_crop(29, 264, 183)
    canvas.paste(_c29, (1176, 1248), _c29)
except Exception:
    pass
layout["View_all"] = [1176, 1248, 1440, 1431]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_01_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-4/30_text_E_Conf_Ist_Rnd_76ers_at_Knicks_Gm_2_H.png
try:
    _c30 = get_crop(30, 288, 168)
    canvas.paste(_c30, (576, 2792), _c30)
except Exception:
    pass
layout["E_Conf_Ist_Rnd:_76ers_at_"] = [576, 2792, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_01_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-4/31_clickable_Tracking.png
try:
    _c31 = get_crop(31, 72, 72)
    canvas.paste(_c31, (408, 1455), _c31)
except Exception:
    pass
layout["Tracking"] = [408, 1455, 480, 1527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_01_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-4/32_clickable_Tracking.png
try:
    _c32 = get_crop(32, 72, 72)
    canvas.paste(_c32, (906, 1455), _c32)
except Exception:
    pass
layout["Tracking"] = [906, 1455, 978, 1527]
