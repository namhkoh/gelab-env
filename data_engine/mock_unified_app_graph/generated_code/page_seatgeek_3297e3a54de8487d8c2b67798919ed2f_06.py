# page_id: page_seatgeek_3297e3a54de8487d8c2b67798919ed2f_06
# screenshot: 2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-9.png
# step_index: 6/11
# task: Open SeatGeek. Search "Comedy Show in Los Angeles". Find the top recommendation. When is the event?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw overall background (slightly warm white)
draw.rectangle([(0, 0), (1440, 2960)], fill="#FBFBFB")

# Status bar at top (~72px)
status_h = 72
draw.rectangle([(0, 0), (1440, status_h)], fill="#EFEFF1")
# subtle divider under status bar
draw.line([(0, status_h), (1440, status_h)], fill="#E3E3E6", width=1)

# Header / toolbar area (title + close icon area)
header_top = status_h
header_bottom = 260
draw.rectangle([(0, header_top), (1440, header_bottom)], fill="#FFFFFF")
# bottom divider under header
draw.line([(24, header_bottom), (1440 - 24, header_bottom)], fill="#E6E6E8", width=1)

# Location input card background (rounded rect)
loc_card_x0 = 40
loc_card_y0 = 320
loc_card_x1 = 1400
loc_card_y1 = 430
draw.rounded_rectangle([(loc_card_x0, loc_card_y0), (loc_card_x1, loc_card_y1)],
                       radius=14, fill="#FFFFFF", outline="#E7E7E9", width=1)

# Date selector rounded card (the large tabbed area "Today/Tomorrow/Weekend")
date_card_x0 = 48
date_card_y0 = 505
date_card_w = 1344
date_card_h = 153
date_card_x1 = date_card_x0 + date_card_w
date_card_y1 = date_card_y0 + date_card_h
draw.rounded_rectangle([(date_card_x0, date_card_y0), (date_card_x1, date_card_y1)],
                       radius=12, fill="#FFFFFF", outline="#E6E6E8", width=1)
# thin inner divider line inside the date card (as in UI)
inner_div_y = date_card_y0 + int(date_card_h * 0.55)
draw.line([(date_card_x0 + 24, inner_div_y), (date_card_x1 - 24, inner_div_y)], fill="#EDEDED", width=1)

# A faint separator line above the block of category cards
cat_sep_y = 980
draw.line([(24, cat_sep_y), (1440 - 24, cat_sep_y)], fill="#E9E9EB", width=1)

# Category card backgrounds (three rounded dark cards across)
# Left card (Sports)
left_card = (42, 1160, 42 + 440, 1160 + 320)
draw.rounded_rectangle([left_card[0:2], left_card[2:4]], radius=18, fill="#161616")

# Middle card (Concerts)
middle_card = (474, 1100, 474 + 492, 1100 + 320)
draw.rounded_rectangle([middle_card[0:2], middle_card[2:4]], radius=18, fill="#151515")

# Right card (Broadway)
right_card = (1036, 1161, 1036 + 404, 1161 + 320)
draw.rounded_rectangle([right_card[0:2], right_card[2:4]], radius=18, fill="#141414")

# Light horizontal separator below the category cards
draw.line([(24, 1520), (1440 - 24, 1520)], fill="#E7E7E9", width=1)

# "Just announced" section area background (subtle off-white)
just_ann_y0 = 1530
just_ann_y1 = 1750
draw.rectangle([(0, just_ann_y0), (1440, just_ann_y1)], fill="#FBFBFB")
# small rounded thumbnail background for the just announced item
ja_thumb = (42, 1740, 42 + 320, 1740 + 200)
draw.rounded_rectangle([ja_thumb[0:2], ja_thumb[2:4]], radius=14, fill="#F0F6FF")

# Separator line under the just announced content
sep2_y = 1870
draw.line([(24, sep2_y), (1440 - 24, sep2_y)], fill="#E8E8EA", width=1)

# "Sports" section header background remains canvas; add subtle band behind carousel strip
carousel_band_y0 = 1900
carousel_band_y1 = 2300
draw.rectangle([(0, carousel_band_y0), (1440, carousel_band_y1)], fill="#FAFAFA")
# placeholder rounded cards for the sports thumbnails row (backgrounds only)
thumb_w = 360
thumb_h = 240
thumb_pad = 36
x = 42
for i in range(4):
    rx0 = x + i * (thumb_w + 16)
    rx1 = rx0 + thumb_w
    ry0 = carousel_band_y0 + 64
    ry1 = ry0 + thumb_h
    draw.rounded_rectangle([(rx0, ry0), (rx1, ry1)], radius=12, fill="#FFFFFF", outline="#E6E6E8", width=1)

# Bottom navigation bar background (white) and top divider + subtle shadow
nav_top = 2792
nav_bottom = 2960
draw.rectangle([(0, nav_top), (1440, nav_bottom)], fill="#FFFFFF")
# top divider line
draw.line([(0, nav_top), (1440, nav_top)], fill="#E6E6E8", width=1)
# soft shadow above nav (thin darker band)
for i, alpha_col in enumerate([230, 225, 220]):
    y = nav_top - (i + 1) * 2
    # draw progressively darker thin lines to imply shadow
    draw.line([(0, y), (1440, y)], fill=(220 - i*6, 220 - i*6, 220 - i*6), width=1)

# Final subtle overall vertical separators between major sections
draw.line([(24, 260), (24, 2800)], fill="#FFFFFF", width=1)  # left padding column (visual guide)
draw.line([(1440 - 24, 260), (1440 - 24, 2800)], fill="#FFFFFF", width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_06_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-9/00_icon_Broadway.png
try:
    _c0 = get_crop(0, 404, 317)
    canvas.paste(_c0, (1036, 1261), _c0)
except Exception:
    pass
layout["Broadway"] = [1036, 1261, 1440, 1578]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_06_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-9/01_icon_Concerts.png
try:
    _c1 = get_crop(1, 492, 149)
    canvas.paste(_c1, (474, 1052), _c1)
except Exception:
    pass
layout["Concerts"] = [474, 1052, 966, 1201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_06_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-9/02_icon_Sports.png
try:
    _c2 = get_crop(2, 471, 321)
    canvas.paste(_c2, (42, 1260), _c2)
except Exception:
    pass
layout["Sports"] = [42, 1260, 513, 1581]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_06_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-9/03_icon_Tomorrow.png
try:
    _c3 = get_crop(3, 1344, 153)
    canvas.paste(_c3, (48, 505), _c3)
except Exception:
    pass
layout["Tomorrow"] = [48, 505, 1392, 658]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_06_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-9/04_icon_Los_Angeles_CA.png
try:
    _c4 = get_crop(4, 61, 57)
    canvas.paste(_c4, (243, 6), _c4)
except Exception:
    pass
layout["Los_Angeles,_CA"] = [243, 6, 304, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_06_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-9/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 49, 68)
    canvas.paste(_c5, (1153, 1), _c5)
except Exception:
    pass
layout["icon_5"] = [1153, 1, 1202, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_06_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-9/06_icon_7_10_W.png
try:
    _c6 = get_crop(6, 46, 55)
    canvas.paste(_c6, (186, 7), _c6)
except Exception:
    pass
layout["7:10_W"] = [186, 7, 232, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_06_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-9/07_icon_Tracking.png
try:
    _c7 = get_crop(7, 288, 168)
    canvas.paste(_c7, (864, 2792), _c7)
except Exception:
    pass
layout["Tracking"] = [864, 2792, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_06_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-9/08_icon_7_10_W.png
try:
    _c8 = get_crop(8, 55, 58)
    canvas.paste(_c8, (115, 4), _c8)
except Exception:
    pass
layout["7:10_W"] = [115, 4, 170, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_06_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-9/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 56, 55)
    canvas.paste(_c9, (313, 8), _c9)
except Exception:
    pass
layout["icon_9"] = [313, 8, 369, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_06_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-9/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 51, 63)
    canvas.paste(_c10, (1321, 2), _c10)
except Exception:
    pass
layout["icon_10"] = [1321, 2, 1372, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_06_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-9/11_icon_IA.png
try:
    _c11 = get_crop(11, 288, 162)
    canvas.paste(_c11, (0, 2792), _c11)
except Exception:
    pass
layout["IA"] = [0, 2792, 288, 2954]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_06_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-9/12_icon_Tickets.png
try:
    _c12 = get_crop(12, 288, 168)
    canvas.paste(_c12, (576, 2792), _c12)
except Exception:
    pass
layout["Tickets"] = [576, 2792, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_06_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-9/13_icon_IA.png
try:
    _c13 = get_crop(13, 288, 168)
    canvas.paste(_c13, (288, 2792), _c13)
except Exception:
    pass
layout["IA"] = [288, 2792, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_06_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-9/14_icon_Account.png
try:
    _c14 = get_crop(14, 288, 168)
    canvas.paste(_c14, (1152, 2792), _c14)
except Exception:
    pass
layout["Account"] = [1152, 2792, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_06_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-9/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 104, 65)
    canvas.paste(_c15, (1213, 1), _c15)
except Exception:
    pass
layout["icon_15"] = [1213, 1, 1317, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_06_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-9/16_icon_Close.png
try:
    _c16 = get_crop(16, 144, 240)
    canvas.paste(_c16, (1260, 72), _c16)
except Exception:
    pass
layout["Close"] = [1260, 72, 1404, 312]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_06_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-9/17_icon_Today.png
try:
    _c17 = get_crop(17, 448, 149)
    canvas.paste(_c17, (48, 901), _c17)
except Exception:
    pass
layout["Today"] = [48, 901, 496, 1050]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_06_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-9/18_text_Los_Angeles_CA.png
try:
    _c18 = get_crop(18, 458, 80)
    canvas.paste(_c18, (42, 132), _c18)
except Exception:
    pass
layout["Los_Angeles,_CA"] = [42, 132, 500, 212]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_06_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-9/19_text_Location.png
try:
    _c19 = get_crop(19, 235, 54)
    canvas.paste(_c19, (44, 382), _c19)
except Exception:
    pass
layout["Location"] = [44, 382, 279, 436]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_06_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-9/20_text_Date.png
try:
    _c20 = get_crop(20, 140, 60)
    canvas.paste(_c20, (42, 775), _c20)
except Exception:
    pass
layout["Date"] = [42, 775, 182, 835]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_06_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-9/21_text_Clear.png
try:
    _c21 = get_crop(21, 264, 149)
    canvas.paste(_c21, (1176, 730), _c21)
except Exception:
    pass
layout["Clear"] = [1176, 730, 1440, 879]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_06_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-9/22_text_Tomorrow.png
try:
    _c22 = get_crop(22, 448, 149)
    canvas.paste(_c22, (496, 901), _c22)
except Exception:
    pass
layout["Tomorrow"] = [496, 901, 944, 1050]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_06_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-9/23_text_Weekend.png
try:
    _c23 = get_crop(23, 448, 149)
    canvas.paste(_c23, (944, 901), _c23)
except Exception:
    pass
layout["Weekend"] = [944, 901, 1392, 1050]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_06_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-9/24_text_Set_custom_date.png
try:
    _c24 = get_crop(24, 492, 149)
    canvas.paste(_c24, (474, 1052), _c24)
except Exception:
    pass
layout["Set_custom_date"] = [474, 1052, 966, 1201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_06_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-9/25_text_Just_announced.png
try:
    _c25 = get_crop(25, 412, 54)
    canvas.paste(_c25, (42, 1691), _c25)
except Exception:
    pass
layout["Just_announced"] = [42, 1691, 454, 1745]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_06_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-9/26_text_View_all.png
try:
    _c26 = get_crop(26, 165, 43)
    canvas.paste(_c26, (1227, 1699), _c26)
except Exception:
    pass
layout["View_all"] = [1227, 1699, 1392, 1742]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_06_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-9/27_text_S52.png
try:
    _c27 = get_crop(27, 114, 52)
    canvas.paste(_c27, (95, 2037), _c27)
except Exception:
    pass
layout["S52+"] = [95, 2037, 209, 2089]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_06_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-9/28_text_Andrew_Schulz.png
try:
    _c28 = get_crop(28, 321, 52)
    canvas.paste(_c28, (46, 2162), _c28)
except Exception:
    pass
layout["Andrew_Schulz"] = [46, 2162, 367, 2214]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_06_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-9/29_text_Thu.png
try:
    _c29 = get_crop(29, 92, 45)
    canvas.paste(_c29, (45, 2235), _c29)
except Exception:
    pass
layout["Thu,"] = [45, 2235, 137, 2280]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_06_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-9/30_text_9_7.30_PM.png
try:
    _c30 = get_crop(30, 214, 49)
    canvas.paste(_c30, (234, 2232), _c30)
except Exception:
    pass
layout["9,7.30_PM"] = [234, 2232, 448, 2281]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_06_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-9/31_text_Sports.png
try:
    _c31 = get_crop(31, 179, 68)
    canvas.paste(_c31, (41, 2446), _c31)
except Exception:
    pass
layout["Sports"] = [41, 2446, 220, 2514]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_06_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-9/32_text_View_all.png
try:
    _c32 = get_crop(32, 170, 49)
    canvas.paste(_c32, (1223, 2447), _c32)
except Exception:
    pass
layout["View_all"] = [1223, 2447, 1393, 2496]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_06_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-9/33_text_IA.png
try:
    _c33 = get_crop(33, 117, 140)
    canvas.paste(_c33, (67, 2648), _c33)
except Exception:
    pass
layout["IA"] = [67, 2648, 184, 2788]
