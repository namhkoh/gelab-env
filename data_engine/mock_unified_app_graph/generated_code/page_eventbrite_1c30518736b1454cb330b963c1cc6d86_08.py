# page_id: page_eventbrite_1c30518736b1454cb330b963c1cc6d86_08
# screenshot: 2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-10.png
# step_index: 8/9
# task: Open Eventbrite. Search for "Open Mic Nights". Filter the results to only include free events. Select the first non-promoted event in the list - what"s the location of that event?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw overall background (canvas already white, but ensure consistent fill)
draw.rectangle([(0, 0), (1440, 2960)], fill="#FFFFFF")

# Status bar at top (~96px high) - light gray
status_bar_height = 96
draw.rectangle([(0, 0), (1440, status_bar_height)], fill="#CFCFCF")

# Header / toolbar area (under status bar) - keep white but add subtle bottom divider
header_top = status_bar_height
header_bottom = 256
draw.rectangle([(0, header_top), (1440, header_bottom)], fill="#FFFFFF")
draw.line([(48, header_bottom), (1392, header_bottom)], fill="#E6E6E6", width=2)

# Filter row area under header - white, with faint shadow line under filter chips
filter_top = header_bottom
filter_bottom = 336
draw.rectangle([(0, filter_top), (1440, filter_bottom)], fill="#FFFFFF")
draw.line([(48, filter_bottom), (1392, filter_bottom)], fill="#F0F0F0", width=1)

# Content list area background (slight off-white band to separate from header)
content_top = filter_bottom
content_bg_bottom = 2760  # leave room for bottom nav
draw.rectangle([(0, content_top), (1440, content_bg_bottom)], fill="#FCFCFD")

# Define common card geometry (left/right margins at 48)
left = 48
right = 1392
card_radius = 18

# First event card - image banner background (rounded)
first_image_top = 364
first_image_bottom = 560
draw.rounded_rectangle([(left, first_image_top), (right, first_image_bottom)],
                       radius=12, fill="#FFF5EE", outline=None)

# White detail card below first image (card body background)
first_detail_top = first_image_bottom + 18
first_detail_bottom = first_detail_top + 180
# Draw a subtle shadow rectangle behind the card
draw.rectangle([(left+2, first_detail_top+6), (right+2, first_detail_bottom+6)], fill="#EFEFF1")
draw.rounded_rectangle([(left, first_detail_top), (right, first_detail_bottom)],
                       radius=12, fill="#FFFFFF", outline="#EFEFEF", width=1)

# Separator line between list items
sep_y = first_detail_bottom + 22
draw.line([(48, sep_y), (1392, sep_y)], fill="#F2F2F4", width=1)

# Second event card - larger image banner (darker background to hint an image)
second_image_top = sep_y + 24
second_image_bottom = second_image_top + 320
draw.rounded_rectangle([(left, second_image_top), (right, second_image_bottom)],
                       radius=12, fill="#3E2A20", outline=None)

# White detail card below second image
second_detail_top = second_image_bottom + 18
second_detail_bottom = second_detail_top + 160
draw.rectangle([(left+2, second_detail_top+6), (right+2, second_detail_bottom+6)], fill="#EFEFF1")
draw.rounded_rectangle([(left, second_detail_top), (right, second_detail_bottom)],
                       radius=12, fill="#FFFFFF", outline="#EFEFEF", width=1)

# Another separator for the upcoming list
sep2_y = second_detail_bottom + 22
draw.line([(48, sep2_y), (1392, sep2_y)], fill="#F2F2F4", width=1)

# Third image placeholder (partially visible further down)
third_image_top = sep2_y + 28
third_image_bottom = third_image_top + 320
draw.rounded_rectangle([(left, third_image_top), (right, third_image_bottom)],
                       radius=12, fill="#000000", outline=None)

# Cards often have small rounded tag backgrounds (e.g., "Free" badge) — do not draw actual text.
# Instead, draw neutral translucent pill backgrounds where badges appear to provide structure,
# but avoid overlapping detected icon positions. Place them near left margin within detail areas.
badge_w = 78
badge_h = 36
badge_radius = 10
# First detail badge area (empty pill only)
badge1_x = left + 8
badge1_y = first_detail_top + 12
draw.rounded_rectangle([(badge1_x, badge1_y), (badge1_x + badge_w, badge1_y + badge_h)],
                       radius=badge_radius, fill="#EDF7EE", outline=None)

# Second detail badge area
badge2_x = left + 8
badge2_y = second_detail_top + 12
draw.rounded_rectangle([(badge2_x, badge2_y), (badge2_x + badge_w, badge2_y + badge_h)],
                       radius=badge_radius, fill="#EDF7EE", outline=None)

# Light horizontal separators within detail cards to indicate grouping (no text)
draw.line([(left+24, first_detail_top + 64), (right-24, first_detail_top + 64)], fill="#F6F6F7", width=1)
draw.line([(left+24, second_detail_top + 64), (right-24, second_detail_top + 64)], fill="#F6F6F7", width=1)

# Bottom navigation bar area (given detected nav icons at y=2804 height 156)
nav_top = 2804
nav_bottom = 2960
draw.rectangle([(0, nav_top), (1440, nav_bottom)], fill="#FFFFFF")
# Top divider for nav
draw.line([(0, nav_top), (1440, nav_top)], fill="#E6E6E6", width=2)

# Subtle left and right screen edges shadow for depth
edge_shadow_width = 12
# left edge
draw.rectangle([(0, header_bottom), (edge_shadow_width, nav_bottom)], fill="#FBFBFC")
# right edge
draw.rectangle([(1440-edge_shadow_width, header_bottom), (1440, nav_bottom)], fill="#FBFBFC")

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_08_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-10/00_icon_Music.png
try:
    _c0 = get_crop(0, 198, 110)
    canvas.paste(_c0, (843, 406), _c0)
except Exception:
    pass
layout["Music"] = [843, 406, 1041, 516]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_08_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-10/01_icon_Business.png
try:
    _c1 = get_crop(1, 251, 111)
    canvas.paste(_c1, (1042, 405), _c1)
except Exception:
    pass
layout["Business"] = [1042, 405, 1293, 516]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_08_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-10/02_icon_Anytime.png
try:
    _c2 = get_crop(2, 1344, 191)
    canvas.paste(_c2, (48, 72), _c2)
except Exception:
    pass
layout["Anytime"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_08_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-10/03_icon_1_Filter.png
try:
    _c3 = get_crop(3, 492, 144)
    canvas.paste(_c3, (0, 259), _c3)
except Exception:
    pass
layout["1_Filter"] = [0, 259, 492, 403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_08_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-10/04_icon_Fo.png
try:
    _c4 = get_crop(4, 139, 108)
    canvas.paste(_c4, (1296, 407), _c4)
except Exception:
    pass
layout["Fo("] = [1296, 407, 1435, 515]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_08_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-10/05_icon_Iigiht.png
try:
    _c5 = get_crop(5, 144, 144)
    canvas.paste(_c5, (1092, 1942), _c5)
except Exception:
    pass
layout["Iigiht"] = [1092, 1942, 1236, 2086]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_08_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-10/06_icon_Overflow_menu_button.png
try:
    _c6 = get_crop(6, 144, 144)
    canvas.paste(_c6, (1236, 1942), _c6)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 1942, 1380, 2086]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_08_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-10/07_icon_Overflow_menu_button.png
try:
    _c7 = get_crop(7, 144, 144)
    canvas.paste(_c7, (1236, 719), _c7)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 719, 1380, 863]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_08_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-10/08_icon_480653.png
try:
    _c8 = get_crop(8, 144, 144)
    canvas.paste(_c8, (1092, 719), _c8)
except Exception:
    pass
layout["480653"] = [1092, 719, 1236, 863]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_08_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-10/09_icon_4.54.png
try:
    _c9 = get_crop(9, 113, 109)
    canvas.paste(_c9, (60, 115), _c9)
except Exception:
    pass
layout["4.54"] = [60, 115, 173, 224]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_08_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-10/10_icon_Close_current_screen.png
try:
    _c10 = get_crop(10, 144, 144)
    canvas.paste(_c10, (1248, 96), _c10)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 96, 1392, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_08_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-10/11_icon_4.54.png
try:
    _c11 = get_crop(11, 55, 63)
    canvas.paste(_c11, (116, 1), _c11)
except Exception:
    pass
layout["4.54"] = [116, 1, 171, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_08_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-10/12_icon_Open_Mic_Night.png
try:
    _c12 = get_crop(12, 1344, 191)
    canvas.paste(_c12, (48, 72), _c12)
except Exception:
    pass
layout["Open_Mic_Night"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_08_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-10/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 61, 61)
    canvas.paste(_c13, (310, 1), _c13)
except Exception:
    pass
layout["icon_13"] = [310, 1, 371, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_08_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-10/14_icon_4.54.png
try:
    _c14 = get_crop(14, 55, 62)
    canvas.paste(_c14, (183, 1), _c14)
except Exception:
    pass
layout["4.54"] = [183, 1, 238, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_08_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-10/15_icon_Leading_Role.png
try:
    _c15 = get_crop(15, 42, 63)
    canvas.paste(_c15, (284, 1273), _c15)
except Exception:
    pass
layout["Leading_Role"] = [284, 1273, 326, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_08_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-10/16_icon_icon_16.png
try:
    _c16 = get_crop(16, 51, 62)
    canvas.paste(_c16, (247, 1), _c16)
except Exception:
    pass
layout["icon_16"] = [247, 1, 298, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_08_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-10/17_icon_icon_17.png
try:
    _c17 = get_crop(17, 72, 61)
    canvas.paste(_c17, (1209, 0), _c17)
except Exception:
    pass
layout["icon_17"] = [1209, 0, 1281, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_08_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-10/18_icon_icon_18.png
try:
    _c18 = get_crop(18, 58, 61)
    canvas.paste(_c18, (1317, 0), _c18)
except Exception:
    pass
layout["icon_18"] = [1317, 0, 1375, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_08_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-10/19_icon_Anytime.png
try:
    _c19 = get_crop(19, 1344, 853)
    canvas.paste(_c19, (48, 525), _c19)
except Exception:
    pass
layout["Anytime"] = [48, 525, 1392, 1378]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_08_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-10/20_icon_Tickets.png
try:
    _c20 = get_crop(20, 288, 156)
    canvas.paste(_c20, (864, 2804), _c20)
except Exception:
    pass
layout["Tickets"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_08_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-10/21_icon_4.54.png
try:
    _c21 = get_crop(21, 93, 64)
    canvas.paste(_c21, (14, 0), _c21)
except Exception:
    pass
layout["4.54"] = [14, 0, 107, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_08_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-10/22_icon_icon_22.png
try:
    _c22 = get_crop(22, 62, 62)
    canvas.paste(_c22, (1251, 0), _c22)
except Exception:
    pass
layout["icon_22"] = [1251, 0, 1313, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_08_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-10/23_icon_Los_Angeles.png
try:
    _c23 = get_crop(23, 492, 144)
    canvas.paste(_c23, (0, 259), _c23)
except Exception:
    pass
layout["Los_Angeles"] = [0, 259, 492, 403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_08_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-10/24_icon_OPEN_IIIC.png
try:
    _c24 = get_crop(24, 1344, 1029)
    canvas.paste(_c24, (48, 1426), _c24)
except Exception:
    pass
layout["OPEN_IIIC"] = [48, 1426, 1392, 2455]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_08_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-10/25_icon_icon_25.png
try:
    _c25 = get_crop(25, 45, 59)
    canvas.paste(_c25, (384, 3), _c25)
except Exception:
    pass
layout["icon_25"] = [384, 3, 429, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_08_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-10/26_icon_Search_events.png
try:
    _c26 = get_crop(26, 288, 156)
    canvas.paste(_c26, (288, 2804), _c26)
except Exception:
    pass
layout["Search_events"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_08_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-10/27_icon_Home.png
try:
    _c27 = get_crop(27, 288, 156)
    canvas.paste(_c27, (0, 2804), _c27)
except Exception:
    pass
layout["Home"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_08_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-10/28_icon_Center_South.png
try:
    _c28 = get_crop(28, 1344, 1029)
    canvas.paste(_c28, (48, 1426), _c28)
except Exception:
    pass
layout["Center_South"] = [48, 1426, 1392, 2455]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_08_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-10/29_icon_More.png
try:
    _c29 = get_crop(29, 288, 156)
    canvas.paste(_c29, (1152, 2804), _c29)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_08_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-10/30_icon_Leading_Role_Store_Opening_with_Strawber.png
try:
    _c30 = get_crop(30, 1344, 853)
    canvas.paste(_c30, (48, 525), _c30)
except Exception:
    pass
layout["Leading_Role_Store_Openin"] = [48, 525, 1392, 1378]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_08_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-10/31_icon_T_INSIDE_WATS.png
try:
    _c31 = get_crop(31, 288, 156)
    canvas.paste(_c31, (576, 2804), _c31)
except Exception:
    pass
layout["T_INSIDE_WATS"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_08_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-10/32_icon_Fire_Open_Mic_Nights.png
try:
    _c32 = get_crop(32, 590, 84)
    canvas.paste(_c32, (88, 2194), _c32)
except Exception:
    pass
layout["Fire_Open_Mic_Nights!"] = [88, 2194, 678, 2278]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_08_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-10/33_text_T_INSIDE_WATS.png
try:
    _c33 = get_crop(33, 288, 156)
    canvas.paste(_c33, (576, 2804), _c33)
except Exception:
    pass
layout["T_INSIDE_WATS"] = [576, 2804, 864, 2960]
