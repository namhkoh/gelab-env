# page_id: page_eventbrite_f01eaa41f6284da09deb7ced3e4eea4e_06
# screenshot: 2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-8.png
# step_index: 6/11
# task: Open Eventbrite. Check out 'Sports' events. Apply filters for events happening this week. Select the first event. Check similar events and add the first similar event to favorite.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Background fill
draw.rectangle([0, 0, 1440, 2960], fill="#f6f7fb")

# Status bar (top area)
status_h = 96
draw.rectangle([0, 0, 1440, status_h], fill="#bdbdbd")
# subtle divider under status bar
draw.line([0, status_h, 1440, status_h], fill="#b3b3b3", width=1)

# Header / toolbar area
header_top = status_h
header_bottom = 220
draw.rectangle([0, header_top, 1440, header_bottom], fill="#ffffff")
# header bottom divider
draw.line([48, header_bottom, 1392, header_bottom], fill="#e6e9ee", width=1)

# Subtle divider under filter row area (avoid drawing chips/icons themselves)
filter_divider_y = 520
draw.line([48, filter_divider_y, 1392, filter_divider_y], fill="#eceff3", width=1)

# Card 1 (rounded background + subtle shadow)
card1_left, card1_top = 48, 640
card1_right, card1_bottom = 1392, 1816
card_radius = 28

# shadow for card1
shadow_offset = 8
draw.rounded_rectangle(
    [card1_left + shadow_offset, card1_top + shadow_offset, card1_right + shadow_offset, card1_bottom + shadow_offset],
    radius=card_radius,
    fill="#e9edf2"
)
# white card surface
draw.rounded_rectangle([card1_left, card1_top, card1_right, card1_bottom], radius=card_radius, fill="#ffffff")

# Separator line between cards (light)
sep_y = card1_bottom + 16
draw.line([48, sep_y, 1392, sep_y], fill="#f0f2f5", width=1)

# Card 2 (rounded background + subtle shadow)
card2_left, card2_top = 48, 1760
card2_right, card2_bottom = 1392, 2816
# shadow for card2
draw.rounded_rectangle(
    [card2_left + shadow_offset, card2_top + shadow_offset, card2_right + shadow_offset, card2_bottom + shadow_offset],
    radius=card_radius,
    fill="#e9edf2"
)
# white card surface
draw.rounded_rectangle([card2_left, card2_top, card2_right, card2_bottom], radius=card_radius, fill="#ffffff")

# Subtle inner dividers for card content areas (do not draw text/icons)
# e.g., divider under image areas (approximate)
img1_bottom_approx = 1784  # approximate image bottom from detected crops
draw.line([card1_left + 16, img1_bottom_approx + 8, card1_right - 16, img1_bottom_approx + 8], fill="#f3f5f8", width=1)

img2_bottom_approx = 2816  # approximate second image bottom (keeps consistency)
draw.line([card2_left + 16, img2_bottom_approx - 160, card2_right - 16, img2_bottom_approx - 160], fill="#f3f5f8", width=1)

# Bottom navigation bar background and top divider
nav_top = 2720
draw.rectangle([0, nav_top, 1440, 2960], fill="#ffffff")
draw.line([0, nav_top, 1440, nav_top], fill="#e6e9ee", width=1)

# Small accent line under header title area (for visual balance)
draw.line([48, header_bottom - 8, 200, header_bottom - 8], fill="#dfe6ff", width=4)

# End of UI structural drawing.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_06_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-8/00_icon_Music.png
try:
    _c0 = get_crop(0, 187, 103)
    canvas.paste(_c0, (882, 410), _c0)
except Exception:
    pass
layout["Music"] = [882, 410, 1069, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_06_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-8/01_icon_This_Week.png
try:
    _c1 = get_crop(1, 432, 103)
    canvas.paste(_c1, (438, 410), _c1)
except Exception:
    pass
layout["This_Week"] = [438, 410, 870, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_06_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-8/02_icon_Business.png
try:
    _c2 = get_crop(2, 241, 103)
    canvas.paste(_c2, (1081, 410), _c2)
except Exception:
    pass
layout["Business"] = [1081, 410, 1322, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_06_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-8/03_icon_1_Filter.png
try:
    _c3 = get_crop(3, 372, 103)
    canvas.paste(_c3, (54, 410), _c3)
except Exception:
    pass
layout["1_Filter"] = [54, 410, 426, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_06_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-8/04_icon_Favorite_button.png
try:
    _c4 = get_crop(4, 144, 144)
    canvas.paste(_c4, (1092, 2348), _c4)
except Exception:
    pass
layout["Favorite_button"] = [1092, 2348, 1236, 2492]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_06_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-8/05_icon_Favorite_button.png
try:
    _c5 = get_crop(5, 144, 144)
    canvas.paste(_c5, (1092, 1192), _c5)
except Exception:
    pass
layout["Favorite_button"] = [1092, 1192, 1236, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_06_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-8/06_icon_Backpacking_Clinic_w_Sports_Basement.png
try:
    _c6 = get_crop(6, 1344, 1108)
    canvas.paste(_c6, (48, 676), _c6)
except Exception:
    pass
layout["Backpacking_Clinic_w__Spo"] = [48, 676, 1392, 1784]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_06_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-8/07_icon_Business.png
try:
    _c7 = get_crop(7, 93, 109)
    canvas.paste(_c7, (1329, 408), _c7)
except Exception:
    pass
layout["Business"] = [1329, 408, 1422, 517]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_06_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-8/08_icon_Overflow_menu_button.png
try:
    _c8 = get_crop(8, 144, 144)
    canvas.paste(_c8, (1236, 2348), _c8)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 2348, 1380, 2492]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_06_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-8/09_icon_Overflow_menu_button.png
try:
    _c9 = get_crop(9, 144, 144)
    canvas.paste(_c9, (1236, 1192), _c9)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 1192, 1380, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_06_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-8/10_icon_Kids_and_Teens_Ballroom_Dance_classes.png
try:
    _c10 = get_crop(10, 1344, 984)
    canvas.paste(_c10, (48, 1832), _c10)
except Exception:
    pass
layout["Kids_and_Teens_Ballroom_D"] = [48, 1832, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_06_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-8/11_icon_Sports.png
try:
    _c11 = get_crop(11, 1344, 191)
    canvas.paste(_c11, (48, 72), _c11)
except Exception:
    pass
layout["Sports"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_06_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-8/12_icon_4.36.png
try:
    _c12 = get_crop(12, 120, 111)
    canvas.paste(_c12, (57, 114), _c12)
except Exception:
    pass
layout["4.36"] = [57, 114, 177, 225]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_06_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-8/13_icon_Close_current_screen.png
try:
    _c13 = get_crop(13, 144, 144)
    canvas.paste(_c13, (1248, 96), _c13)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 96, 1392, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_06_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-8/14_icon_4.36.png
try:
    _c14 = get_crop(14, 58, 63)
    canvas.paste(_c14, (181, 1), _c14)
except Exception:
    pass
layout["4.36"] = [181, 1, 239, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_06_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-8/15_icon_Sports.png
try:
    _c15 = get_crop(15, 66, 63)
    canvas.paste(_c15, (308, 1), _c15)
except Exception:
    pass
layout["Sports"] = [308, 1, 374, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_06_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-8/16_icon_4.36.png
try:
    _c16 = get_crop(16, 59, 64)
    canvas.paste(_c16, (115, 0), _c16)
except Exception:
    pass
layout["4.36"] = [115, 0, 174, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_06_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-8/17_icon_Sports.png
try:
    _c17 = get_crop(17, 51, 63)
    canvas.paste(_c17, (247, 1), _c17)
except Exception:
    pass
layout["Sports"] = [247, 1, 298, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_06_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-8/18_icon_icon_18.png
try:
    _c18 = get_crop(18, 101, 62)
    canvas.paste(_c18, (1209, 0), _c18)
except Exception:
    pass
layout["icon_18"] = [1209, 0, 1310, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_06_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-8/19_icon_San_Francisco.png
try:
    _c19 = get_crop(19, 536, 144)
    canvas.paste(_c19, (0, 259), _c19)
except Exception:
    pass
layout["San_Francisco"] = [0, 259, 536, 403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_06_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-8/20_icon_icon_20.png
try:
    _c20 = get_crop(20, 57, 62)
    canvas.paste(_c20, (1317, 0), _c20)
except Exception:
    pass
layout["icon_20"] = [1317, 0, 1374, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_06_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-8/21_icon_7_00_PM_PDT.png
try:
    _c21 = get_crop(21, 288, 156)
    canvas.paste(_c21, (288, 2804), _c21)
except Exception:
    pass
layout["7:00_PM_PDT"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_06_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-8/22_icon_icon_22.png
try:
    _c22 = get_crop(22, 49, 62)
    canvas.paste(_c22, (384, 2), _c22)
except Exception:
    pass
layout["icon_22"] = [384, 2, 433, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_06_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-8/23_icon_icon_23.png
try:
    _c23 = get_crop(23, 90, 75)
    canvas.paste(_c23, (1271, 1827), _c23)
except Exception:
    pass
layout["icon_23"] = [1271, 1827, 1361, 1902]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_06_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-8/24_icon_Kids_and_Teens_Ballroom_Dance_classes.png
try:
    _c24 = get_crop(24, 288, 156)
    canvas.paste(_c24, (576, 2804), _c24)
except Exception:
    pass
layout["Kids_and_Teens_Ballroom_D"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_06_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-8/25_icon_Backpacking_Clinic_w_Sports_Basement.png
try:
    _c25 = get_crop(25, 1344, 1108)
    canvas.paste(_c25, (48, 676), _c25)
except Exception:
    pass
layout["Backpacking_Clinic_w__Spo"] = [48, 676, 1392, 1784]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_06_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-8/26_icon_Wed_Apr_24.png
try:
    _c26 = get_crop(26, 288, 156)
    canvas.paste(_c26, (0, 2804), _c26)
except Exception:
    pass
layout["Wed,_Apr_24"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_06_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-8/27_text_4.36.png
try:
    _c27 = get_crop(27, 92, 43)
    canvas.paste(_c27, (22, 17), _c27)
except Exception:
    pass
layout["4.36"] = [22, 17, 114, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_06_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-8/28_text_347_events.png
try:
    _c28 = get_crop(28, 372, 103)
    canvas.paste(_c28, (54, 410), _c28)
except Exception:
    pass
layout["347_events"] = [54, 410, 426, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_06_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-8/29_text_Wed_Apr_24.png
try:
    _c29 = get_crop(29, 288, 156)
    canvas.paste(_c29, (0, 2804), _c29)
except Exception:
    pass
layout["Wed,_Apr_24"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_06_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-8/30_text_7_00_PM_PDT.png
try:
    _c30 = get_crop(30, 288, 156)
    canvas.paste(_c30, (288, 2804), _c30)
except Exception:
    pass
layout["7:00_PM_PDT"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_06_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-8/31_clickable_Tickets.png
try:
    _c31 = get_crop(31, 288, 156)
    canvas.paste(_c31, (864, 2804), _c31)
except Exception:
    pass
layout["Tickets"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_06_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-8/32_clickable_More.png
try:
    _c32 = get_crop(32, 288, 156)
    canvas.paste(_c32, (1152, 2804), _c32)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]
