# page_id: page_seatgeek_49ceba1342134bb89f14fac27abc2dcd_01
# screenshot: 2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-4.png
# step_index: 1/12
# task: Open SeatGeek. Track "New York Yankees", "Boston Red Sox".
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Top status bar
draw.rectangle([(0, 0), (1440, 80)], fill="#F2F3F5")  # subtle gray status bar
draw.line([(0, 80), (1440, 80)], fill="#E6E7E9", width=1)

# Header area (location / filters)
draw.rectangle([(0, 80), (1440, 220)], fill="#FFFFFF")
draw.line([(24, 220), (1416, 220)], fill="#E6E7E9", width=1)

# Reserve space below hero card with a subtle divider (hero card itself is detected and will be pasted; do not draw it)
draw.line([(24, 1208), (1416, 1208)], fill="#ECEFF1", width=1)

# "Just for you" section background card (rounded)
just_card = (24, 1220, 1416, 1620)
draw.rounded_rectangle(just_card, radius=16, fill="#FFFFFF", outline="#EAECEF", width=1)

# Soft shadow under the "Just for you" card
draw.rectangle([(just_card[0]+6, just_card[3]+2), (just_card[2]-6, just_card[3]+4)], fill="#F6F7F8")

# Divider between "Just for you" and next section
draw.line([(24, 1640), (1416, 1640)], fill="#F0F1F3", width=1)

# "Trending events" section background card (rounded)
trending_card = (24, 1880, 1416, 2600)
draw.rounded_rectangle(trending_card, radius=16, fill="#FFFFFF", outline="#EAECEF", width=1)

# Subtle separators inside trending card to suggest list rows (only structural lines, no text)
row_y = 2020
for i in range(3):
    draw.line([(48, row_y + i * 200), (1392, row_y + i * 200)], fill="#F1F2F4", width=1)

# Right-hand subtle gradient accent strip (to mirror UI visual balance, not an icon or text)
accent_strip = (1390, 1880, 1440, 2600)
draw.rectangle(accent_strip, fill="#FFFFFF")

# Bottom navigation bar background
draw.rectangle([(0, 2792), (1440, 2960)], fill="#FFFFFF")
draw.line([(0, 2792), (1440, 2792)], fill="#E6E7E9", width=1)

# Small top shadow for navigation bar
draw.rectangle([(0, 2776), (1440, 2792)], fill="#FBFBFB")

# Left and right page margins accent lines to frame content
draw.line([(24, 220), (24, 2600)], fill="#FFFFFF", width=1)
draw.line([(1416, 220), (1416, 2600)], fill="#FFFFFF", width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_01_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-4/00_icon_Globe_Life_Field.png
try:
    _c0 = get_crop(0, 1309, 236)
    canvas.paste(_c0, (0, 2197), _c0)
except Exception:
    pass
layout["Globe_Life_Field"] = [0, 2197, 1309, 2433]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_01_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-4/01_icon_Mavericks.png
try:
    _c1 = get_crop(1, 1344, 840)
    canvas.paste(_c1, (48, 360), _c1)
except Exception:
    pass
layout["Mavericks"] = [48, 360, 1392, 1200]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_01_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-4/02_icon_View_all.png
try:
    _c2 = get_crop(2, 95, 148)
    canvas.paste(_c2, (1345, 2244), _c2)
except Exception:
    pass
layout["View_all"] = [1345, 2244, 1440, 2392]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_01_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-4/03_icon_9_8_PM.png
try:
    _c3 = get_crop(3, 462, 533)
    canvas.paste(_c3, (48, 1431), _c3)
except Exception:
    pass
layout["9,8_PM"] = [48, 1431, 510, 1964]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_01_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-4/04_icon_icon_4.png
try:
    _c4 = get_crop(4, 102, 147)
    canvas.paste(_c4, (1338, 2481), _c4)
except Exception:
    pass
layout["icon_4"] = [1338, 2481, 1440, 2628]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_01_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-4/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 63, 58)
    canvas.paste(_c5, (242, 5), _c5)
except Exception:
    pass
layout["icon_5"] = [242, 5, 305, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_01_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-4/06_icon_American_Airlines_Center.png
try:
    _c6 = get_crop(6, 1309, 236)
    canvas.paste(_c6, (0, 2433), _c6)
except Exception:
    pass
layout["American_Airlines_Center"] = [0, 2433, 1309, 2669]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_01_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-4/07_icon_888.png
try:
    _c7 = get_crop(7, 101, 64)
    canvas.paste(_c7, (1213, 1), _c7)
except Exception:
    pass
layout["888"] = [1213, 1, 1314, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_01_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-4/08_icon_Tracking.png
try:
    _c8 = get_crop(8, 288, 168)
    canvas.paste(_c8, (864, 2792), _c8)
except Exception:
    pass
layout["Tracking"] = [864, 2792, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_01_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-4/09_icon_8.34_Wy.png
try:
    _c9 = get_crop(9, 56, 58)
    canvas.paste(_c9, (114, 4), _c9)
except Exception:
    pass
layout["8.34_Wy"] = [114, 4, 170, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_01_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-4/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 55, 58)
    canvas.paste(_c10, (314, 5), _c10)
except Exception:
    pass
layout["icon_10"] = [314, 5, 369, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_01_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-4/11_icon_888.png
try:
    _c11 = get_crop(11, 144, 240)
    canvas.paste(_c11, (1260, 72), _c11)
except Exception:
    pass
layout["888"] = [1260, 72, 1404, 312]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_01_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-4/12_icon_8.34_Wy.png
try:
    _c12 = get_crop(12, 48, 58)
    canvas.paste(_c12, (184, 4), _c12)
except Exception:
    pass
layout["8.34_Wy"] = [184, 4, 232, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_01_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-4/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 52, 63)
    canvas.paste(_c13, (1319, 2), _c13)
except Exception:
    pass
layout["icon_13"] = [1319, 2, 1371, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_01_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-4/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 47, 65)
    canvas.paste(_c14, (1154, 1), _c14)
except Exception:
    pass
layout["icon_14"] = [1154, 1, 1201, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_01_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-4/15_icon_S159.png
try:
    _c15 = get_crop(15, 462, 533)
    canvas.paste(_c15, (48, 1431), _c15)
except Exception:
    pass
layout["S159"] = [48, 1431, 510, 1964]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_01_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-4/16_icon_Account.png
try:
    _c16 = get_crop(16, 288, 168)
    canvas.paste(_c16, (1152, 2792), _c16)
except Exception:
    pass
layout["Account"] = [1152, 2792, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_01_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-4/17_icon_icon_17.png
try:
    _c17 = get_crop(17, 115, 130)
    canvas.paste(_c17, (1141, 2490), _c17)
except Exception:
    pass
layout["icon_17"] = [1141, 2490, 1256, 2620]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_01_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-4/18_icon_W_Conf_Ist_Rnd_Clippers_at_Mavericks.png
try:
    _c18 = get_crop(18, 288, 168)
    canvas.paste(_c18, (288, 2792), _c18)
except Exception:
    pass
layout["W_Conf_Ist_Rnd:_Clippers_"] = [288, 2792, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_01_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-4/19_icon_Browse.png
try:
    _c19 = get_crop(19, 288, 162)
    canvas.paste(_c19, (0, 2792), _c19)
except Exception:
    pass
layout["Browse"] = [0, 2792, 288, 2954]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_01_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-4/20_icon_icon_20.png
try:
    _c20 = get_crop(20, 98, 111)
    canvas.paste(_c20, (1342, 2708), _c20)
except Exception:
    pass
layout["icon_20"] = [1342, 2708, 1440, 2819]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_01_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-4/21_icon_W_Conf_Ist_Rnd_Clippers_at_Mavericks.png
try:
    _c21 = get_crop(21, 288, 168)
    canvas.paste(_c21, (576, 2792), _c21)
except Exception:
    pass
layout["W_Conf_Ist_Rnd:_Clippers_"] = [576, 2792, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_01_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-4/22_icon_8.34_Wy.png
try:
    _c22 = get_crop(22, 94, 60)
    canvas.paste(_c22, (14, 2), _c22)
except Exception:
    pass
layout["8.34_Wy"] = [14, 2, 108, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_01_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-4/23_text_Dallas_TX.png
try:
    _c23 = get_crop(23, 295, 76)
    canvas.paste(_c23, (41, 129), _c23)
except Exception:
    pass
layout["Dallas,_TX"] = [41, 129, 336, 205]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_01_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-4/24_text_date.png
try:
    _c24 = get_crop(24, 114, 52)
    canvas.paste(_c24, (137, 208), _c24)
except Exception:
    pass
layout["date"] = [137, 208, 251, 260]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_01_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-4/25_text_Just_for_you.png
try:
    _c25 = get_crop(25, 306, 66)
    canvas.paste(_c25, (38, 1310), _c25)
except Exception:
    pass
layout["Just_for_you"] = [38, 1310, 344, 1376]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_01_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-4/26_text_View_all.png
try:
    _c26 = get_crop(26, 264, 183)
    canvas.paste(_c26, (1176, 1248), _c26)
except Exception:
    pass
layout["View_all"] = [1176, 1248, 1440, 1431]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_01_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-4/27_text_Trending_events.png
try:
    _c27 = get_crop(27, 423, 81)
    canvas.paste(_c27, (38, 2068), _c27)
except Exception:
    pass
layout["Trending_events"] = [38, 2068, 461, 2149]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_01_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-4/28_text_View_all.png
try:
    _c28 = get_crop(28, 264, 183)
    canvas.paste(_c28, (1176, 2014), _c28)
except Exception:
    pass
layout["View_all"] = [1176, 2014, 1440, 2197]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_01_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-4/29_text_W_Conf_Ist_Rnd_Clippers_at_Mavericks.png
try:
    _c29 = get_crop(29, 288, 168)
    canvas.paste(_c29, (576, 2792), _c29)
except Exception:
    pass
layout["W_Conf_Ist_Rnd:_Clippers_"] = [576, 2792, 864, 2960]
