# page_id: page_eventbrite_9fdb2ee43d5a49adac5304bdd5dacfc2_01
# screenshot: 2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-3.png
# step_index: 1/8
# task: Open Eventbrite. Look up 'Pet' events. Filter by events happening this weekend. Select the third non-promoted event from the results - how much are the tickets for the event?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Background
draw.rectangle([(0, 0), (1440, 2960)], fill="#FBFBFD")  # overall subtle off-white background

# Status bar (top ~50px)
status_h = 80
draw.rectangle([(0, 0), (1440, status_h)], fill="#CFCFCF")  # light grey status bar

# Header area (below status bar) with subtle bottom divider
header_top = status_h
header_bottom = 220
draw.rectangle([(0, header_top), (1440, header_bottom)], fill="#FFFFFF")
draw.line([(32, header_bottom), (1440-32, header_bottom)], fill="#E9E7EE", width=2)

# Main content area subtle top padding shadow
draw.rectangle([(0, header_bottom), (1440, header_bottom+8)], fill="#F6F5F8")

# Card list backgrounds (use detected title positions as anchors)
card_x0 = 48
card_x1 = 48 + 1344  # matches detected width
card_w = card_x1 - card_x0
card_h = 396
card_positions_y = [490, 886, 1282, 1678, 2074, 2525]

for y in card_positions_y:
    # shadow
    shadow_offset = 6
    shadow_bbox = [card_x0 + 4, y + 8, card_x1 + 4, y + card_h + 8]
    draw.rounded_rectangle(shadow_bbox, radius=18, fill="#ECE9ED")
    # card background
    card_bbox = [card_x0, y, card_x1, y + card_h]
    draw.rounded_rectangle(card_bbox, radius=18, fill="#FFFFFF")
    # left thumbnail placeholder (content background for image)
    thumb_margin = 18
    thumb_x0 = card_x0 + thumb_margin
    thumb_x1 = thumb_x0 + 170
    thumb_y0 = y + thumb_margin
    thumb_y1 = thumb_y0 + 110
    draw.rounded_rectangle([(thumb_x0, thumb_y0), (thumb_x1, thumb_y1)], radius=8, fill="#E9EEF3")
    # small neutral strip behind title area (visual grouping, not text)
    strip_x0 = thumb_x1 + 20
    strip_x1 = card_x1 - 36
    strip_y0 = y + 22
    strip_y1 = y + 22 + 6
    draw.rectangle([(strip_x0, strip_y0), (strip_x1, strip_y1)], fill="#F5F4F7")
    # light divider at bottom of card area (separates cards)
    sep_y = y + card_h + 6
    draw.line([(card_x0 + 8, sep_y), (card_x1 - 8, sep_y)], fill="#F0EEF2", width=1)

# Additional subtle section separator above the list title area
title_sep_y = 460
draw.line([(32, title_sep_y), (1440-32, title_sep_y)], fill="#F0EDF2", width=1)

# Floating content area background example (for location dropdown area near lower content)
# Draw a soft rounded pill background where the location selector will be placed (background only)
loc_bbox = [420, 2490, 420 + 520, 2490 + 96]  # placed near detected location area but simple background only
draw.rounded_rectangle(loc_bbox, radius=48, fill="#FFFFFF", outline="#E8E6EA", width=1)

# Bottom navigation bar background and top divider (space reserved for nav icons)
nav_top = 2804
draw.rectangle([(0, nav_top), (1440, 2960)], fill="#FFFFFF")
draw.line([(24, nav_top), (1440-24, nav_top)], fill="#EAE7EB", width=2)
# subtle shadow above nav
draw.rectangle([(0, nav_top-6), (1440, nav_top)], fill="#F6F5F8")

# End of drawing - leave all icons/text areas blank for pasted crops

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_01_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-3/00_icon_Search_events.png
try:
    _c0 = get_crop(0, 1179, 144)
    canvas.paste(_c0, (195, 93), _c0)
except Exception:
    pass
layout["Search_events"] = [195, 93, 1374, 237]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_01_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-3/01_icon_Washington.png
try:
    _c1 = get_crop(1, 454, 117)
    canvas.paste(_c1, (493, 2651), _c1)
except Exception:
    pass
layout["Washington"] = [493, 2651, 947, 2768]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_01_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-3/02_icon_Workshop_on_Grief_and_Remembrance.png
try:
    _c2 = get_crop(2, 1344, 396)
    canvas.paste(_c2, (48, 886), _c2)
except Exception:
    pass
layout["Workshop_on_Grief_and_Rem"] = [48, 886, 1392, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_01_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-3/03_icon_Overflow_menu_button.png
try:
    _c3 = get_crop(3, 144, 123)
    canvas.paste(_c3, (1284, 1555), _c3)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 1555, 1428, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_01_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-3/04_icon_Favorite_button.png
try:
    _c4 = get_crop(4, 144, 123)
    canvas.paste(_c4, (1140, 1159), _c4)
except Exception:
    pass
layout["Favorite_button"] = [1140, 1159, 1284, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_01_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-3/05_icon_Favorite_button.png
try:
    _c5 = get_crop(5, 144, 139)
    canvas.paste(_c5, (1140, 747), _c5)
except Exception:
    pass
layout["Favorite_button"] = [1140, 747, 1284, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_01_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-3/06_icon_Favorite_button.png
try:
    _c6 = get_crop(6, 144, 123)
    canvas.paste(_c6, (1140, 1555), _c6)
except Exception:
    pass
layout["Favorite_button"] = [1140, 1555, 1284, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_01_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-3/07_icon_Rockvin.png
try:
    _c7 = get_crop(7, 288, 156)
    canvas.paste(_c7, (288, 2804), _c7)
except Exception:
    pass
layout["Rockvin"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_01_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-3/08_icon_Home.png
try:
    _c8 = get_crop(8, 288, 156)
    canvas.paste(_c8, (0, 2804), _c8)
except Exception:
    pass
layout["Home"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_01_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-3/09_icon_Overflow_menu_button.png
try:
    _c9 = get_crop(9, 144, 139)
    canvas.paste(_c9, (1284, 2331), _c9)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 2331, 1428, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_01_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-3/10_icon_Favorite_button.png
try:
    _c10 = get_crop(10, 144, 139)
    canvas.paste(_c10, (1140, 2331), _c10)
except Exception:
    pass
layout["Favorite_button"] = [1140, 2331, 1284, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_01_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-3/11_icon_Overflow_menu_button.png
try:
    _c11 = get_crop(11, 144, 139)
    canvas.paste(_c11, (1284, 1935), _c11)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 1935, 1428, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_01_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-3/12_icon_Wendt_Center_For_Loss_and_Healing.png
try:
    _c12 = get_crop(12, 1344, 396)
    canvas.paste(_c12, (48, 1282), _c12)
except Exception:
    pass
layout["Wendt_Center_For_Loss_and"] = [48, 1282, 1392, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_01_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-3/13_icon_Overflow_menu_button.png
try:
    _c13 = get_crop(13, 144, 139)
    canvas.paste(_c13, (1284, 747), _c13)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 747, 1428, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_01_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-3/14_icon_Favorite_button.png
try:
    _c14 = get_crop(14, 144, 139)
    canvas.paste(_c14, (1140, 1935), _c14)
except Exception:
    pass
layout["Favorite_button"] = [1140, 1935, 1284, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_01_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-3/15_icon_Yoga_for_Grief_Loss.png
try:
    _c15 = get_crop(15, 1344, 396)
    canvas.paste(_c15, (48, 1678), _c15)
except Exception:
    pass
layout["Yoga_for_Grief_&_Loss"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_01_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-3/16_icon_4.47.png
try:
    _c16 = get_crop(16, 57, 61)
    canvas.paste(_c16, (182, 2), _c16)
except Exception:
    pass
layout["4.47"] = [182, 2, 239, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_01_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-3/17_icon_icon_17.png
try:
    _c17 = get_crop(17, 60, 59)
    canvas.paste(_c17, (312, 3), _c17)
except Exception:
    pass
layout["icon_17"] = [312, 3, 372, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_01_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-3/18_icon_Sun_Oct_6.png
try:
    _c18 = get_crop(18, 1344, 396)
    canvas.paste(_c18, (48, 2074), _c18)
except Exception:
    pass
layout["Sun,_Oct_6"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_01_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-3/19_icon_Break_into_Tech_Social_Assembly.png
try:
    _c19 = get_crop(19, 1344, 396)
    canvas.paste(_c19, (48, 490), _c19)
except Exception:
    pass
layout["Break_into_Tech_Social_@_"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_01_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-3/20_icon_4.47.png
try:
    _c20 = get_crop(20, 98, 94)
    canvas.paste(_c20, (43, 123), _c20)
except Exception:
    pass
layout["4.47"] = [43, 123, 141, 217]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_01_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-3/21_icon_icon_21.png
try:
    _c21 = get_crop(21, 51, 60)
    canvas.paste(_c21, (248, 2), _c21)
except Exception:
    pass
layout["icon_21"] = [248, 2, 299, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_01_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-3/22_icon_Overflow_menu_button.png
try:
    _c22 = get_crop(22, 144, 123)
    canvas.paste(_c22, (1284, 1159), _c22)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 1159, 1428, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_01_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-3/23_icon_Pain.png
try:
    _c23 = get_crop(23, 1344, 396)
    canvas.paste(_c23, (48, 1282), _c23)
except Exception:
    pass
layout["Pain"] = [48, 1282, 1392, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_01_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-3/24_icon_icon_24.png
try:
    _c24 = get_crop(24, 47, 52)
    canvas.paste(_c24, (1321, 7), _c24)
except Exception:
    pass
layout["icon_24"] = [1321, 7, 1368, 59]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_01_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-3/25_icon_Yoga_for_Grief_Loss.png
try:
    _c25 = get_crop(25, 1344, 396)
    canvas.paste(_c25, (48, 2074), _c25)
except Exception:
    pass
layout["Yoga_for_Grief_&_Loss"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_01_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-3/26_icon_4.47.png
try:
    _c26 = get_crop(26, 59, 61)
    canvas.paste(_c26, (114, 2), _c26)
except Exception:
    pass
layout["4.47"] = [114, 2, 173, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_01_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-3/27_icon_icon_27.png
try:
    _c27 = get_crop(27, 60, 57)
    canvas.paste(_c27, (1213, 5), _c27)
except Exception:
    pass
layout["icon_27"] = [1213, 5, 1273, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_01_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-3/28_icon_rtist_Romare_Beardens_collage_Tomorrow.png
try:
    _c28 = get_crop(28, 1344, 396)
    canvas.paste(_c28, (48, 886), _c28)
except Exception:
    pass
layout["rtist_Romare_Beardens_col"] = [48, 886, 1392, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_01_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-3/29_icon_icon_29.png
try:
    _c29 = get_crop(29, 44, 56)
    canvas.paste(_c29, (385, 7), _c29)
except Exception:
    pass
layout["icon_29"] = [385, 7, 429, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_01_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-3/30_icon_icon_30.png
try:
    _c30 = get_crop(30, 61, 55)
    canvas.paste(_c30, (1252, 6), _c30)
except Exception:
    pass
layout["icon_30"] = [1252, 6, 1313, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_01_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-3/31_icon_Tickets.png
try:
    _c31 = get_crop(31, 288, 156)
    canvas.paste(_c31, (864, 2804), _c31)
except Exception:
    pass
layout["Tickets"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_01_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-3/32_icon_4.47.png
try:
    _c32 = get_crop(32, 91, 59)
    canvas.paste(_c32, (15, 4), _c32)
except Exception:
    pass
layout["4.47"] = [15, 4, 106, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_01_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-3/33_icon_I_O0AM_EDT.png
try:
    _c33 = get_crop(33, 1344, 396)
    canvas.paste(_c33, (48, 1678), _c33)
except Exception:
    pass
layout["I:O0AM_EDT"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_01_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-3/34_icon_1I_O0AM_EDT.png
try:
    _c34 = get_crop(34, 1344, 396)
    canvas.paste(_c34, (48, 2074), _c34)
except Exception:
    pass
layout["1I:O0AM_EDT"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_01_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-3/35_text_More_events_you_II_love.png
try:
    _c35 = get_crop(35, 1344, 396)
    canvas.paste(_c35, (48, 490), _c35)
except Exception:
    pass
layout["More_events_you'II_love"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_01_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-3/36_text_Tue_Apr_30.png
try:
    _c36 = get_crop(36, 200, 43)
    canvas.paste(_c36, (390, 2525), _c36)
except Exception:
    pass
layout["Tue,_Apr_30"] = [390, 2525, 590, 2568]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_01_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-3/37_text_7_00_PM_EDT.png
try:
    _c37 = get_crop(37, 1344, 346)
    canvas.paste(_c37, (48, 2470), _c37)
except Exception:
    pass
layout["7:00_PM_EDT"] = [48, 2470, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_01_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-3/38_text_Free.png
try:
    _c38 = get_crop(38, 78, 38)
    canvas.paste(_c38, (274, 2561), _c38)
except Exception:
    pass
layout["Free"] = [274, 2561, 352, 2599]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_01_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-3/39_text_8_341_creator_followers.png
try:
    _c39 = get_crop(39, 288, 156)
    canvas.paste(_c39, (576, 2804), _c39)
except Exception:
    pass
layout["8_341_creator_followers"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_01_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-3/40_clickable_More.png
try:
    _c40 = get_crop(40, 288, 156)
    canvas.paste(_c40, (1152, 2804), _c40)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]
