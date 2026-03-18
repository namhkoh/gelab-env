# page_id: page_eventbrite_31528ae16c264b1b98bd4e1f25c5d9e5_07
# screenshot: 2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-9.png
# step_index: 7/11
# task: Open Eventbrite. Search 'Fitness'. Filter for free events. Browse and select any 'Yoga' event. Note the location.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw overall background
draw.rectangle((0, 0, 1440, 2960), fill="#FAFAFC")  # subtle off-white canvas background

# Status bar area (top)
status_h = 96
draw.rectangle((0, 0, 1440, status_h), fill="#BDBDBF")  # light gray status bar

# Header / toolbar area under status bar
header_top = status_h
header_bottom = 280
draw.rectangle((0, header_top, 1440, header_bottom), fill="#FFFFFF")  # white header background
# thin divider under header
draw.line((48, header_bottom, 1440-48, header_bottom), fill="#E6E6EA", width=2)

# Filter/chips band background (subtle)
chips_top = 360
chips_bottom = 520
draw.rectangle((48, chips_top, 1440-48, chips_bottom), fill="#FFFFFF")  # keep it white but separated
# soft shadow under chips band
draw.line((48, chips_bottom, 1440-48, chips_bottom), fill="#F0F2F5", width=3)

# Thin separator above the event list
sep_y = 560
draw.line((24, sep_y, 1440-24, sep_y), fill="#EFEFF2", width=1)

# First event card container (behind the large banner image)
# Detected image at (48,676) size (1344x1175); draw a subtle card slightly larger as background container
card1_x0 = 48 - 8
card1_y0 = 676 - 8
card1_x1 = 48 + 1344 + 8
card1_y1 = 676 + 1175 + 8
draw.rounded_rectangle((card1_x0, card1_y0, card1_x1, card1_y1),
                       radius=24, fill="#FFFFFF", outline="#E8E9EC", width=1)
# drop shadow (very subtle) below card1
shadow_y0 = card1_y1 + 6
draw.rectangle((card1_x0+6, shadow_y0, card1_x1-6, shadow_y0+4), fill="#F2F4F7")

# Space/spacing line between card image and title area (visual divider)
# (Keep light so it doesn't interfere with pasted text/icons)
divider1_y = card1_y0 + 420
draw.line((card1_x0+16, divider1_y, card1_x1-16, divider1_y), fill="#F3F4F6", width=1)

# Second event card container (for subsequent event image/content)
# Detected second event area at (48,1899) size (1344x917)
card2_x0 = 48 - 8
card2_y0 = 1899 - 8
card2_x1 = 48 + 1344 + 8
card2_y1 = 1899 + 917 + 8
draw.rounded_rectangle((card2_x0, card2_y0, card2_x1, card2_y1),
                       radius=20, fill="#FFFFFF", outline="#E8E9EC", width=1)
# subtle shadow below second card
shadow2_y0 = card2_y1 + 6
draw.rectangle((card2_x0+6, shadow2_y0, card2_x1-6, shadow2_y0+4), fill="#F2F4F7")

# Horizontal separators between list items (subtle)
sep_lines = [card1_y1 + 24, card2_y1 + 24]
for y in sep_lines:
    draw.line((24, y, 1440-24, y), fill="#F0F1F4", width=1)

# Bottom navigation bar background and top divider
nav_h = 100
nav_y0 = 2960 - nav_h
draw.rectangle((0, nav_y0, 1440, 2960), fill="#FFFFFF")
draw.line((24, nav_y0, 1440-24, nav_y0), fill="#E6E7EA", width=2)

# Left and right margins subtle vertical guides (do not draw icons/text)
draw.line((48, 0, 48, 2960), fill="#FFFFFF", width=1)   # left content margin (invisible - keeps alignment)
draw.line((1440-48, 0, 1440-48, 2960), fill="#FFFFFF", width=1)

# Decorative faint background band behind the event list to add depth
band_top = 620
band_bottom = 2200
draw.rectangle((0, band_top, 1440, band_bottom), fill="#FFFFFF")

# Final top shadow under status+header to ground the UI
draw.line((0, header_bottom+1, 1440, header_bottom+1), fill="#EDEFF2", width=2)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_07_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-9/00_icon_Music.png
try:
    _c0 = get_crop(0, 187, 103)
    canvas.paste(_c0, (850, 410), _c0)
except Exception:
    pass
layout["Music"] = [850, 410, 1037, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_07_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-9/01_icon_Anytime.png
try:
    _c1 = get_crop(1, 400, 103)
    canvas.paste(_c1, (438, 410), _c1)
except Exception:
    pass
layout["Anytime"] = [438, 410, 838, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_07_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-9/02_icon_Business.png
try:
    _c2 = get_crop(2, 241, 103)
    canvas.paste(_c2, (1049, 410), _c2)
except Exception:
    pass
layout["Business"] = [1049, 410, 1290, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_07_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-9/03_icon_1_Filter.png
try:
    _c3 = get_crop(3, 372, 103)
    canvas.paste(_c3, (54, 410), _c3)
except Exception:
    pass
layout["1_Filter"] = [54, 410, 426, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_07_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-9/04_icon_Fo.png
try:
    _c4 = get_crop(4, 139, 111)
    canvas.paste(_c4, (1295, 406), _c4)
except Exception:
    pass
layout["Fo("] = [1295, 406, 1434, 517]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_07_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-9/05_icon_Favorite_button.png
try:
    _c5 = get_crop(5, 144, 144)
    canvas.paste(_c5, (1092, 1192), _c5)
except Exception:
    pass
layout["Favorite_button"] = [1092, 1192, 1236, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_07_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-9/06_icon_Overflow_menu_button.png
try:
    _c6 = get_crop(6, 144, 144)
    canvas.paste(_c6, (1236, 1192), _c6)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 1192, 1380, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_07_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-9/07_icon_Fo.png
try:
    _c7 = get_crop(7, 144, 144)
    canvas.paste(_c7, (1248, 96), _c7)
except Exception:
    pass
layout["Fo("] = [1248, 96, 1392, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_07_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-9/08_icon_7.55.png
try:
    _c8 = get_crop(8, 126, 116)
    canvas.paste(_c8, (53, 113), _c8)
except Exception:
    pass
layout["7.55"] = [53, 113, 179, 229]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_07_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-9/09_icon_7.55.png
try:
    _c9 = get_crop(9, 62, 64)
    canvas.paste(_c9, (179, 0), _c9)
except Exception:
    pass
layout["7.55"] = [179, 0, 241, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_07_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-9/10_icon_Fitness.png
try:
    _c10 = get_crop(10, 70, 64)
    canvas.paste(_c10, (306, 0), _c10)
except Exception:
    pass
layout["Fitness"] = [306, 0, 376, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_07_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-9/11_icon_Fitness.png
try:
    _c11 = get_crop(11, 53, 64)
    canvas.paste(_c11, (248, 0), _c11)
except Exception:
    pass
layout["Fitness"] = [248, 0, 301, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_07_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-9/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 103, 61)
    canvas.paste(_c12, (1206, 0), _c12)
except Exception:
    pass
layout["icon_12"] = [1206, 0, 1309, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_07_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-9/13_icon_7.55.png
try:
    _c13 = get_crop(13, 61, 65)
    canvas.paste(_c13, (114, 0), _c13)
except Exception:
    pass
layout["7.55"] = [114, 0, 175, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_07_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-9/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 66, 60)
    canvas.paste(_c14, (1317, 0), _c14)
except Exception:
    pass
layout["icon_14"] = [1317, 0, 1383, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_07_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-9/15_icon_Fitness.png
try:
    _c15 = get_crop(15, 1344, 191)
    canvas.paste(_c15, (48, 72), _c15)
except Exception:
    pass
layout["Fitness"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_07_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-9/16_icon_Favorite_button.png
try:
    _c16 = get_crop(16, 144, 144)
    canvas.paste(_c16, (1092, 2415), _c16)
except Exception:
    pass
layout["Favorite_button"] = [1092, 2415, 1236, 2559]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_07_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-9/17_icon_8.15_PM_EDT.png
try:
    _c17 = get_crop(17, 288, 156)
    canvas.paste(_c17, (288, 2804), _c17)
except Exception:
    pass
layout["8.15_PM_EDT"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_07_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-9/18_icon_Empowering_Wisdom_Parenting_Circle.png
try:
    _c18 = get_crop(18, 288, 156)
    canvas.paste(_c18, (864, 2804), _c18)
except Exception:
    pass
layout["Empowering_Wisdom_Parenti"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_07_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-9/19_icon_icon_19.png
try:
    _c19 = get_crop(19, 52, 62)
    canvas.paste(_c19, (383, 2), _c19)
except Exception:
    pass
layout["icon_19"] = [383, 2, 435, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_07_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-9/20_icon_Tue_Apr_30.png
try:
    _c20 = get_crop(20, 288, 156)
    canvas.paste(_c20, (0, 2804), _c20)
except Exception:
    pass
layout["Tue,_Apr_30"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_07_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-9/21_icon_San_Francisco.png
try:
    _c21 = get_crop(21, 536, 144)
    canvas.paste(_c21, (0, 259), _c21)
except Exception:
    pass
layout["San_Francisco"] = [0, 259, 536, 403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_07_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-9/22_icon_Overflow_menu_button.png
try:
    _c22 = get_crop(22, 144, 144)
    canvas.paste(_c22, (1236, 2415), _c22)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 2415, 1380, 2559]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_07_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-9/23_icon_Promoted.png
try:
    _c23 = get_crop(23, 44, 59)
    canvas.paste(_c23, (284, 1748), _c23)
except Exception:
    pass
layout["Promoted"] = [284, 1748, 328, 1807]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_07_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-9/24_icon_Empowering_Wisdom_Parenting_Circle.png
try:
    _c24 = get_crop(24, 288, 156)
    canvas.paste(_c24, (576, 2804), _c24)
except Exception:
    pass
layout["Empowering_Wisdom_Parenti"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_07_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-9/25_icon_Empowering_Wisdom_Parenting_Circle.png
try:
    _c25 = get_crop(25, 288, 156)
    canvas.paste(_c25, (1152, 2804), _c25)
except Exception:
    pass
layout["Empowering_Wisdom_Parenti"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_07_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-9/26_icon_SATURDAY_APRIL_27_9_A.png
try:
    _c26 = get_crop(26, 1344, 1175)
    canvas.paste(_c26, (48, 676), _c26)
except Exception:
    pass
layout["SATURDAY,APRIL_27|9_A"] = [48, 676, 1392, 1851]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_07_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-9/27_text_7.55.png
try:
    _c27 = get_crop(27, 92, 43)
    canvas.paste(_c27, (22, 17), _c27)
except Exception:
    pass
layout["7.55"] = [22, 17, 114, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_07_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-9/28_text_473_events.png
try:
    _c28 = get_crop(28, 372, 103)
    canvas.paste(_c28, (54, 410), _c28)
except Exception:
    pass
layout["473_events"] = [54, 410, 426, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_07_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-9/29_text_MASTERING.png
try:
    _c29 = get_crop(29, 1344, 1175)
    canvas.paste(_c29, (48, 676), _c29)
except Exception:
    pass
layout["MASTERING"] = [48, 676, 1392, 1851]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_07_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-9/30_text_Sat.png
try:
    _c30 = get_crop(30, 90, 53)
    canvas.paste(_c30, (90, 1619), _c30)
except Exception:
    pass
layout["Sat,"] = [90, 1619, 180, 1672]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_07_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-9/31_text_27.png
try:
    _c31 = get_crop(31, 64, 43)
    canvas.paste(_c31, (253, 1622), _c31)
except Exception:
    pass
layout["27"] = [253, 1622, 317, 1665]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_07_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-9/32_text_9_00_AM_EDT.png
try:
    _c32 = get_crop(32, 254, 45)
    canvas.paste(_c32, (334, 1620), _c32)
except Exception:
    pass
layout["9:00_AM_EDT"] = [334, 1620, 588, 1665]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_07_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-9/33_text_Online.png
try:
    _c33 = get_crop(33, 129, 45)
    canvas.paste(_c33, (91, 1687), _c33)
except Exception:
    pass
layout["Online"] = [91, 1687, 220, 1732]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_07_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-9/34_text_Free.png
try:
    _c34 = get_crop(34, 80, 39)
    canvas.paste(_c34, (117, 2614), _c34)
except Exception:
    pass
layout["Free"] = [117, 2614, 197, 2653]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_07_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-9/35_text_Empowering_Wisdom_Parenting_Circle.png
try:
    _c35 = get_crop(35, 1344, 917)
    canvas.paste(_c35, (48, 1899), _c35)
except Exception:
    pass
layout["Empowering_Wisdom_Parenti"] = [48, 1899, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_07_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-9/36_text_Tue_Apr_30.png
try:
    _c36 = get_crop(36, 288, 156)
    canvas.paste(_c36, (0, 2804), _c36)
except Exception:
    pass
layout["Tue,_Apr_30"] = [0, 2804, 288, 2960]
