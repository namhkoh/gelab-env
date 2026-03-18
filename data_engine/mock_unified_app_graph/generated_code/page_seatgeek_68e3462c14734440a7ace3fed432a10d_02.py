# page_id: page_seatgeek_68e3462c14734440a7ace3fed432a10d_02
# screenshot: 2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-5.png
# step_index: 2/13
# task: Open SeatGeek and change the current location to Los Angeles. Then find the first concert show and track its performer.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw UI background and structural elements for the provided canvas using PIL draw object.
# Available variables: canvas (PIL Image 1440x2960), draw (ImageDraw)
# Fonts are available but not used per instructions.

# Canvas base fill (dominant color: white)
draw.rectangle((0, 0, 1440, 2960), fill=(255, 255, 255))

# Status bar area (top ~96px) - light gray background
status_h = 96
draw.rectangle((0, 0, 1440, status_h), fill=(245, 245, 245))

# Header / toolbar area below status bar
header_top = status_h
header_bottom = 220
draw.rectangle((0, header_top, 1440, header_bottom), fill=(255, 255, 255))
# Header bottom divider line
draw.line((24, header_bottom, 1440 - 24, header_bottom), fill=(230, 230, 230), width=2)

# "Location" input card (rounded white box with subtle border)
loc_card_left = 48
loc_card_right = 1392
loc_card_top = 460
loc_card_bottom = 540
loc_radius = 20
draw.rounded_rectangle(
    (loc_card_left, loc_card_top, loc_card_right, loc_card_bottom),
    radius=loc_radius,
    fill=(255, 255, 255),
    outline=(235, 235, 235),
    width=2
)

# Thin divider under the location card area
draw.line((24, loc_card_bottom + 36, 1440 - 24, loc_card_bottom + 36), fill=(240, 240, 240), width=1)

# Date card (larger rounded rectangle containing date tabs)
date_card_top = 860
date_card_bottom = 1052
date_card_left = 48
date_card_right = 1392
date_radius = 20
draw.rounded_rectangle(
    (date_card_left, date_card_top, date_card_right, date_card_bottom),
    radius=date_radius,
    fill=(255, 255, 255),
    outline=(235, 235, 235),
    width=2
)
# Inner divider within date card (to separate tabs from "Set custom date" area)
inner_div_y = date_card_top + int((date_card_bottom - date_card_top) * 0.45)
draw.line((date_card_left + 24, inner_div_y, date_card_right - 24, inner_div_y), fill=(240, 240, 240), width=2)

# Section separator line between the filter area and the content area
content_sep_y = 1310
draw.line((24, content_sep_y, 1440 - 24, content_sep_y), fill=(220, 220, 220), width=1)

# Dark content overlay for "Just for you" and card gallery area
# This provides a dim background behind the thumbnail cards and trending list.
overlay_top = content_sep_y
overlay_bottom = 2792  # leaving bottom nav area intact
draw.rectangle((0, overlay_top, 1440, overlay_bottom), fill=(44, 44, 44))

# Add a slight translucent top band to emulate soft shadow (single darker strip)
draw.rectangle((0, overlay_top, 1440, overlay_top + 6), fill=(34, 34, 34))

# Within the dark overlay, draw subtle rounded card container behind the "Just for you" thumbnails area
thumbs_container_top = overlay_top + 24
thumbs_container_bottom = thumbs_container_top + 210
draw.rounded_rectangle(
    (24, thumbs_container_top, 1440 - 24, thumbs_container_bottom),
    radius=18,
    fill=(60, 60, 60),
    outline=(70, 70, 70),
    width=1
)

# Row separators for the "Trending events" list within overlay
# Draw a few horizontal separators to structure the list rows (positions approximate)
trending_start_y = thumbs_container_bottom + 40
row_height = 120
for i in range(4):
    y = trending_start_y + i * row_height
    draw.line((24, y, 1440 - 24, y), fill=(70, 70, 70), width=1)

# Bottom navigation bar area (leave icons to be pasted; draw background and top border)
bottom_nav_top = 2792
bottom_nav_bottom = 2960
draw.rectangle((0, bottom_nav_top, 1440, bottom_nav_bottom), fill=(255, 255, 255))
# Top border/shadow for bottom nav
draw.line((0, bottom_nav_top, 1440, bottom_nav_top), fill=(230, 230, 230), width=2)

# Small subtle shadow under header to separate from content
draw.line((0, header_bottom + 2, 1440, header_bottom + 2), fill=(245, 245, 245), width=1)

# Add faint left/right inset guides (visual structure) — very light, so they act as background scaffolding
draw.line((48, header_bottom + 6, 48, bottom_nav_top - 6), fill=(250, 250, 250), width=1)
draw.line((1392, header_bottom + 6, 1392, bottom_nav_top - 6), fill=(250, 250, 250), width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_02_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-5/00_icon_Tomorrow.png
try:
    _c0 = get_crop(0, 1344, 153)
    canvas.paste(_c0, (48, 505), _c0)
except Exception:
    pass
layout["Tomorrow"] = [48, 505, 1392, 658]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_02_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-5/01_icon_icon_1.png
try:
    _c1 = get_crop(1, 99, 143)
    canvas.paste(_c1, (1341, 2469), _c1)
except Exception:
    pass
layout["icon_1"] = [1341, 2469, 1440, 2612]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_02_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-5/02_icon_icon_2.png
try:
    _c2 = get_crop(2, 47, 69)
    canvas.paste(_c2, (1153, 0), _c2)
except Exception:
    pass
layout["icon_2"] = [1153, 0, 1200, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_02_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-5/03_icon_View_all.png
try:
    _c3 = get_crop(3, 95, 144)
    canvas.paste(_c3, (1345, 2229), _c3)
except Exception:
    pass
layout["View_all"] = [1345, 2229, 1440, 2373]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_02_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-5/04_icon_8.30_my.png
try:
    _c4 = get_crop(4, 57, 58)
    canvas.paste(_c4, (181, 4), _c4)
except Exception:
    pass
layout["8.30_my"] = [181, 4, 238, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_02_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-5/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 100, 68)
    canvas.paste(_c5, (1214, 0), _c5)
except Exception:
    pass
layout["icon_5"] = [1214, 0, 1314, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_02_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-5/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 52, 65)
    canvas.paste(_c6, (1320, 1), _c6)
except Exception:
    pass
layout["icon_6"] = [1320, 1, 1372, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_02_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-5/07_icon_8.30_my.png
try:
    _c7 = get_crop(7, 52, 59)
    canvas.paste(_c7, (116, 3), _c7)
except Exception:
    pass
layout["8.30_my"] = [116, 3, 168, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_02_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-5/08_icon_S86.png
try:
    _c8 = get_crop(8, 472, 546)
    canvas.paste(_c8, (538, 1431), _c8)
except Exception:
    pass
layout["S86+"] = [538, 1431, 1010, 1977]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_02_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-5/09_icon_St._James_Theatre.png
try:
    _c9 = get_crop(9, 288, 168)
    canvas.paste(_c9, (576, 2792), _c9)
except Exception:
    pass
layout["St._James_Theatre"] = [576, 2792, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_02_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-5/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 52, 55)
    canvas.paste(_c10, (315, 6), _c10)
except Exception:
    pass
layout["icon_10"] = [315, 6, 367, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_02_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-5/11_icon_New_York_NY.png
try:
    _c11 = get_crop(11, 52, 56)
    canvas.paste(_c11, (247, 6), _c11)
except Exception:
    pass
layout["New_York,_NY"] = [247, 6, 299, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_02_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-5/12_icon_S87.png
try:
    _c12 = get_crop(12, 406, 312)
    canvas.paste(_c12, (1034, 1433), _c12)
except Exception:
    pass
layout["S87+"] = [1034, 1433, 1440, 1745]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_02_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-5/13_icon_S2_4_D.png
try:
    _c13 = get_crop(13, 107, 120)
    canvas.paste(_c13, (1140, 2484), _c13)
except Exception:
    pass
layout["S2_(#4_D="] = [1140, 2484, 1247, 2604]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_02_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-5/14_icon_Browse.png
try:
    _c14 = get_crop(14, 288, 162)
    canvas.paste(_c14, (0, 2792), _c14)
except Exception:
    pass
layout["Browse"] = [0, 2792, 288, 2954]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_02_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-5/15_icon_Close.png
try:
    _c15 = get_crop(15, 144, 240)
    canvas.paste(_c15, (1260, 72), _c15)
except Exception:
    pass
layout["Close"] = [1260, 72, 1404, 312]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_02_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-5/16_icon_Barclays_Center.png
try:
    _c16 = get_crop(16, 288, 168)
    canvas.paste(_c16, (576, 2792), _c16)
except Exception:
    pass
layout["Barclays_Center"] = [576, 2792, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_02_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-5/17_icon_Tracking.png
try:
    _c17 = get_crop(17, 288, 168)
    canvas.paste(_c17, (864, 2792), _c17)
except Exception:
    pass
layout["Tracking"] = [864, 2792, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_02_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-5/18_icon_Hadestown.png
try:
    _c18 = get_crop(18, 288, 168)
    canvas.paste(_c18, (288, 2792), _c18)
except Exception:
    pass
layout["Hadestown"] = [288, 2792, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_02_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-5/19_icon_Account.png
try:
    _c19 = get_crop(19, 288, 168)
    canvas.paste(_c19, (1152, 2792), _c19)
except Exception:
    pass
layout["Account"] = [1152, 2792, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_02_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-5/20_text_New_York_NY.png
try:
    _c20 = get_crop(20, 382, 68)
    canvas.paste(_c20, (48, 133), _c20)
except Exception:
    pass
layout["New_York,_NY"] = [48, 133, 430, 201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_02_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-5/21_text_date.png
try:
    _c21 = get_crop(21, 117, 52)
    canvas.paste(_c21, (134, 208), _c21)
except Exception:
    pass
layout["date"] = [134, 208, 251, 260]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_02_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-5/22_text_Location.png
try:
    _c22 = get_crop(22, 235, 54)
    canvas.paste(_c22, (44, 382), _c22)
except Exception:
    pass
layout["Location"] = [44, 382, 279, 436]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_02_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-5/23_text_Date.png
try:
    _c23 = get_crop(23, 140, 60)
    canvas.paste(_c23, (42, 775), _c23)
except Exception:
    pass
layout["Date"] = [42, 775, 182, 835]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_02_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-5/24_text_Clear.png
try:
    _c24 = get_crop(24, 264, 149)
    canvas.paste(_c24, (1176, 730), _c24)
except Exception:
    pass
layout["Clear"] = [1176, 730, 1440, 879]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_02_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-5/25_text_Today.png
try:
    _c25 = get_crop(25, 448, 149)
    canvas.paste(_c25, (48, 901), _c25)
except Exception:
    pass
layout["Today"] = [48, 901, 496, 1050]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_02_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-5/26_text_Tomorrow.png
try:
    _c26 = get_crop(26, 448, 149)
    canvas.paste(_c26, (496, 901), _c26)
except Exception:
    pass
layout["Tomorrow"] = [496, 901, 944, 1050]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_02_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-5/27_text_Weekend.png
try:
    _c27 = get_crop(27, 448, 149)
    canvas.paste(_c27, (944, 901), _c27)
except Exception:
    pass
layout["Weekend"] = [944, 901, 1392, 1050]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_02_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-5/28_text_Set_custom_date.png
try:
    _c28 = get_crop(28, 492, 149)
    canvas.paste(_c28, (474, 1052), _c28)
except Exception:
    pass
layout["Set_custom_date"] = [474, 1052, 966, 1201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_02_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-5/29_text_Just_for_you.png
try:
    _c29 = get_crop(29, 306, 66)
    canvas.paste(_c29, (38, 1310), _c29)
except Exception:
    pass
layout["Just_for_you"] = [38, 1310, 344, 1376]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_02_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-5/30_text_View_all.png
try:
    _c30 = get_crop(30, 170, 49)
    canvas.paste(_c30, (1223, 1314), _c30)
except Exception:
    pass
layout["View_all"] = [1223, 1314, 1393, 1363]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_02_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-5/31_text_BARCLAYS_CEHTE.png
try:
    _c31 = get_crop(31, 81, 19)
    canvas.paste(_c31, (227, 1443), _c31)
except Exception:
    pass
layout["BARCLAYS_CEHTE"] = [227, 1443, 308, 1462]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_02_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-5/32_text_S243.png
try:
    _c32 = get_crop(32, 136, 45)
    canvas.paste(_c32, (98, 1664), _c32)
except Exception:
    pass
layout["S243+"] = [98, 1664, 234, 1709]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_02_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-5/33_text_Lakers_at_Nets.png
try:
    _c33 = get_crop(33, 320, 50)
    canvas.paste(_c33, (42, 1785), _c33)
except Exception:
    pass
layout["Lakers_at_Nets"] = [42, 1785, 362, 1835]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_02_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-5/34_text_Matt_Rife.png
try:
    _c34 = get_crop(34, 204, 50)
    canvas.paste(_c34, (1041, 1785), _c34)
except Exception:
    pass
layout["Matt_Rife"] = [1041, 1785, 1245, 1835]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_02_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-5/35_text_Sun_Mar_31_6_PM.png
try:
    _c35 = get_crop(35, 359, 52)
    canvas.paste(_c35, (42, 1853), _c35)
except Exception:
    pass
layout["Sun,_Mar_31,_6_PM"] = [42, 1853, 401, 1905]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_02_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-5/36_text_Sat.png
try:
    _c36 = get_crop(36, 94, 51)
    canvas.paste(_c36, (1037, 1854), _c36)
except Exception:
    pass
layout["Sat,"] = [1037, 1854, 1131, 1905]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_02_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-5/37_text_27_7_PM.png
try:
    _c37 = get_crop(37, 174, 48)
    canvas.paste(_c37, (1210, 1852), _c37)
except Exception:
    pass
layout["27,7_PM"] = [1210, 1852, 1384, 1900]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_02_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-5/38_text_Trending_events.png
try:
    _c38 = get_crop(38, 424, 81)
    canvas.paste(_c38, (38, 2054), _c38)
except Exception:
    pass
layout["Trending_events"] = [38, 2054, 462, 2135]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_02_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-5/39_text_View_all.png
try:
    _c39 = get_crop(39, 165, 43)
    canvas.paste(_c39, (1227, 2071), _c39)
except Exception:
    pass
layout["View_all"] = [1227, 2071, 1392, 2114]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_02_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-5/40_text_NCAA_M_Basketball_Brooklyn.png
try:
    _c40 = get_crop(40, 288, 168)
    canvas.paste(_c40, (288, 2792), _c40)
except Exception:
    pass
layout["NCAA_M_Basketball_Brookly"] = [288, 2792, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_02_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-5/41_text_S2_4_D.png
try:
    _c41 = get_crop(41, 223, 49)
    canvas.paste(_c41, (884, 2475), _c41)
except Exception:
    pass
layout["S2_(#4_D="] = [884, 2475, 1107, 2524]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_02_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-5/42_text_Mar_22.png
try:
    _c42 = get_crop(42, 155, 52)
    canvas.paste(_c42, (232, 2549), _c42)
except Exception:
    pass
layout["Mar_22"] = [232, 2549, 387, 2601]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_02_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-5/43_text_Barclays_Center.png
try:
    _c43 = get_crop(43, 351, 65)
    canvas.paste(_c43, (407, 2543), _c43)
except Exception:
    pass
layout["Barclays_Center"] = [407, 2543, 758, 2608]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_02_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-5/44_text_Hadestown.png
try:
    _c44 = get_crop(44, 288, 168)
    canvas.paste(_c44, (288, 2792), _c44)
except Exception:
    pass
layout["Hadestown"] = [288, 2792, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_02_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-5/45_clickable_Location.png
try:
    _c45 = get_crop(45, 1440, 937)
    canvas.paste(_c45, (0, 312), _c45)
except Exception:
    pass
layout["Location"] = [0, 312, 1440, 1249]
