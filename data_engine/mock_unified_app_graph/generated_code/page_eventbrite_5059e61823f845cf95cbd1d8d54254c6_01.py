# page_id: page_eventbrite_5059e61823f845cf95cbd1d8d54254c6_01
# screenshot: 2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-3.png
# step_index: 1/19
# task: Open Eventbrite. Look for 'Education' in Los Angeles happening on May 4. Filter to show only free events. How many events are posted?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Top status bar
draw.rectangle([(0, 0), (1440, 56)], fill=(111, 115, 118))  # dark status bar

# Search toolbar background (rounded light field behind the search area)
toolbar_left = 48
toolbar_top = 72
toolbar_w = 1344
toolbar_h = 144
toolbar_rect = (toolbar_left, toolbar_top, toolbar_left + toolbar_w, toolbar_top + toolbar_h)
draw.rounded_rectangle(toolbar_rect, radius=36, fill=(245, 246, 248), outline=(230, 231, 235), width=1)

# Subtle divider under toolbar
draw.line([(48, toolbar_top + toolbar_h + 8), (1440 - 48, toolbar_top + toolbar_h + 8)], fill=(236, 237, 240), width=1)

# Card backgrounds for event rows (rounded white cards with light outline/shadow)
card_positions = [
    (48, 490, 48 + 1344, 490 + 396),
    (48, 886, 48 + 1344, 886 + 396),
    (48, 1282, 48 + 1344, 1282 + 396),
    (48, 1678, 48 + 1344, 1678 + 396),
    (48, 2074, 48 + 1344, 2074 + 396),
    (48, 2470, 48 + 1344, 2470 + 346),
]

for (x1, y1, x2, y2) in card_positions:
    # light shadow block behind card (subtle)
    shadow_offset = 6
    shadow_box = (x1 + shadow_offset, y1 + shadow_offset, x2 + shadow_offset, y2 + shadow_offset)
    draw.rounded_rectangle(shadow_box, radius=14, fill=(250, 250, 251))
    # card itself
    draw.rounded_rectangle((x1, y1, x2, y2), radius=12, fill=(255, 255, 255), outline=(236, 237, 240), width=1)

    # separator line at bottom of each card (subtle)
    draw.line([(x1 + 12, y2 - 1), (x2 - 12, y2 - 1)], fill=(244, 244, 245), width=1)

# Large section title area background (behind "More events you'll love" header)
title_area_top = toolbar_top + toolbar_h + 24
title_area_bottom = 420
draw.rectangle([(48, title_area_top), (48 + 1344, title_area_bottom)], fill=(255, 255, 255))  # keep white but separate area
draw.line([(48, title_area_bottom), (48 + 1344, title_area_bottom)], fill=(244, 244, 245), width=1)

# Floating location pill shadow area behind expected floating control (do not draw the control itself)
pill_shadow_center_x = 720
pill_shadow_center_y = 2651
pill_w, pill_h = 405, 117
pill_box = (pill_shadow_center_x - pill_w//2, pill_shadow_center_y - pill_h//2,
            pill_shadow_center_x + pill_w//2, pill_shadow_center_y + pill_h//2)
# subtle shadow (only background shadow, not the pill content)
draw.rounded_rectangle((pill_box[0]+6, pill_box[1]+8, pill_box[2]+6, pill_box[3]+8), radius=40, fill=(250,250,251))

# Bottom navigation bar area
nav_top = 2804
nav_height = 156
draw.rectangle([(0, nav_top), (1440, nav_top + nav_height)], fill=(255, 255, 255))
# top divider for nav
draw.line([(0, nav_top), (1440, nav_top)], fill=(235, 236, 238), width=1)

# Subtle left and right safe-area guide lines (not visible UI, just structural guides)
# left margin vertical guide (very faint)
draw.line([(48, 56), (48, 2804)], fill=(250, 250, 251), width=1)
draw.line([(1440-48, 56), (1440-48, 2804)], fill=(250, 250, 251), width=1)

# End of structural/background drawing

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_01_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-3/00_icon_City.png
try:
    _c0 = get_crop(0, 1344, 396)
    canvas.paste(_c0, (48, 886), _c0)
except Exception:
    pass
layout["City,"] = [48, 886, 1392, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_01_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-3/01_icon_Q_Search_events.png
try:
    _c1 = get_crop(1, 1179, 144)
    canvas.paste(_c1, (195, 93), _c1)
except Exception:
    pass
layout["Q_Search_events"] = [195, 93, 1374, 237]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_01_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-3/02_icon_New_York.png
try:
    _c2 = get_crop(2, 144, 139)
    canvas.paste(_c2, (1140, 747), _c2)
except Exception:
    pass
layout["New_York"] = [1140, 747, 1284, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_01_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-3/03_icon_Conference_Connections.png
try:
    _c3 = get_crop(3, 144, 139)
    canvas.paste(_c3, (1140, 1935), _c3)
except Exception:
    pass
layout["Conference_Connections"] = [1140, 1935, 1284, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_01_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-3/04_icon_Free.png
try:
    _c4 = get_crop(4, 1344, 396)
    canvas.paste(_c4, (48, 1282), _c4)
except Exception:
    pass
layout["Free"] = [48, 1282, 1392, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_01_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-3/05_icon_Conference_Connections.png
try:
    _c5 = get_crop(5, 144, 123)
    canvas.paste(_c5, (1140, 2347), _c5)
except Exception:
    pass
layout["Conference_Connections"] = [1140, 2347, 1284, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_01_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-3/06_icon_VOSCHINO.png
try:
    _c6 = get_crop(6, 1344, 396)
    canvas.paste(_c6, (48, 1678), _c6)
except Exception:
    pass
layout["VOSCHINO"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_01_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-3/07_icon_New_York.png
try:
    _c7 = get_crop(7, 144, 123)
    canvas.paste(_c7, (1140, 1159), _c7)
except Exception:
    pass
layout["New_York"] = [1140, 1159, 1284, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_01_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-3/08_icon_New_York.png
try:
    _c8 = get_crop(8, 144, 123)
    canvas.paste(_c8, (1284, 1159), _c8)
except Exception:
    pass
layout["New_York"] = [1284, 1159, 1428, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_01_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-3/09_icon_New_York.png
try:
    _c9 = get_crop(9, 144, 139)
    canvas.paste(_c9, (1284, 747), _c9)
except Exception:
    pass
layout["New_York"] = [1284, 747, 1428, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_01_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-3/10_icon_Overflow_menu_button.png
try:
    _c10 = get_crop(10, 144, 123)
    canvas.paste(_c10, (1284, 2347), _c10)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 2347, 1428, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_01_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-3/11_icon_Union_H.png
try:
    _c11 = get_crop(11, 288, 156)
    canvas.paste(_c11, (288, 2804), _c11)
except Exception:
    pass
layout["Union_H"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_01_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-3/12_icon_Good_Afternoon_New_York.png
try:
    _c12 = get_crop(12, 1344, 396)
    canvas.paste(_c12, (48, 1678), _c12)
except Exception:
    pass
layout["Good_Afternoon_New_York"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_01_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-3/13_icon_New_York.png
try:
    _c13 = get_crop(13, 405, 117)
    canvas.paste(_c13, (518, 2651), _c13)
except Exception:
    pass
layout["New_York"] = [518, 2651, 923, 2768]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_01_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-3/14_icon_Overflow_menu_button.png
try:
    _c14 = get_crop(14, 144, 139)
    canvas.paste(_c14, (1284, 1935), _c14)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 1935, 1428, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_01_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-3/15_icon_Overflow_menu_button.png
try:
    _c15 = get_crop(15, 144, 123)
    canvas.paste(_c15, (1284, 1555), _c15)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 1555, 1428, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_01_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-3/16_icon_7.34.png
try:
    _c16 = get_crop(16, 55, 61)
    canvas.paste(_c16, (183, 2), _c16)
except Exception:
    pass
layout["7.34"] = [183, 2, 238, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_01_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-3/17_icon_Medical_Hair_Loss_Therapy_Training.png
try:
    _c17 = get_crop(17, 1344, 396)
    canvas.paste(_c17, (48, 886), _c17)
except Exception:
    pass
layout["Medical_Hair_Loss_Therapy"] = [48, 886, 1392, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_01_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-3/18_icon_free_Grief_and_Loss_Healing_Circle.png
try:
    _c18 = get_crop(18, 1344, 396)
    canvas.paste(_c18, (48, 490), _c18)
except Exception:
    pass
layout["(free)_Grief_and_Loss_Hea"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_01_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-3/19_icon_139_creator_followers.png
try:
    _c19 = get_crop(19, 1344, 396)
    canvas.paste(_c19, (48, 1282), _c19)
except Exception:
    pass
layout["139_creator_followers"] = [48, 1282, 1392, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_01_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-3/20_icon_Favorite_button.png
try:
    _c20 = get_crop(20, 144, 123)
    canvas.paste(_c20, (1140, 1555), _c20)
except Exception:
    pass
layout["Favorite_button"] = [1140, 1555, 1284, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_01_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-3/21_icon_Home.png
try:
    _c21 = get_crop(21, 288, 156)
    canvas.paste(_c21, (0, 2804), _c21)
except Exception:
    pass
layout["Home"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_01_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-3/22_icon_icon_22.png
try:
    _c22 = get_crop(22, 60, 58)
    canvas.paste(_c22, (312, 3), _c22)
except Exception:
    pass
layout["icon_22"] = [312, 3, 372, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_01_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-3/23_icon_7.34.png
try:
    _c23 = get_crop(23, 100, 97)
    canvas.paste(_c23, (43, 123), _c23)
except Exception:
    pass
layout["7.34"] = [43, 123, 143, 220]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_01_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-3/24_icon_Free.png
try:
    _c24 = get_crop(24, 1344, 396)
    canvas.paste(_c24, (48, 490), _c24)
except Exception:
    pass
layout["Free"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_01_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-3/25_icon_icon_25.png
try:
    _c25 = get_crop(25, 50, 59)
    canvas.paste(_c25, (248, 3), _c25)
except Exception:
    pass
layout["icon_25"] = [248, 3, 298, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_01_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-3/26_icon_icon_26.png
try:
    _c26 = get_crop(26, 47, 52)
    canvas.paste(_c26, (1321, 8), _c26)
except Exception:
    pass
layout["icon_26"] = [1321, 8, 1368, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_01_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-3/27_icon_7.34.png
try:
    _c27 = get_crop(27, 58, 60)
    canvas.paste(_c27, (115, 2), _c27)
except Exception:
    pass
layout["7.34"] = [115, 2, 173, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_01_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-3/28_icon_icon_28.png
try:
    _c28 = get_crop(28, 58, 57)
    canvas.paste(_c28, (1212, 5), _c28)
except Exception:
    pass
layout["icon_28"] = [1212, 5, 1270, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_01_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-3/29_icon_Free.png
try:
    _c29 = get_crop(29, 130, 74)
    canvas.paste(_c29, (244, 560), _c29)
except Exception:
    pass
layout["Free"] = [244, 560, 374, 634]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_01_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-3/30_icon_icon_30.png
try:
    _c30 = get_crop(30, 41, 54)
    canvas.paste(_c30, (1272, 7), _c30)
except Exception:
    pass
layout["icon_30"] = [1272, 7, 1313, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_01_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-3/31_icon_Q_Search_events.png
try:
    _c31 = get_crop(31, 44, 56)
    canvas.paste(_c31, (385, 6), _c31)
except Exception:
    pass
layout["Q_Search_events"] = [385, 6, 429, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_01_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-3/32_icon_Free.png
try:
    _c32 = get_crop(32, 127, 75)
    canvas.paste(_c32, (245, 2540), _c32)
except Exception:
    pass
layout["Free"] = [245, 2540, 372, 2615]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_01_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-3/33_icon_free_Grief_and_Loss_Healing_Circle.png
try:
    _c33 = get_crop(33, 1344, 396)
    canvas.paste(_c33, (48, 490), _c33)
except Exception:
    pass
layout["(free)_Grief_and_Loss_Hea"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_01_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-3/34_icon_8_1646_creator_followers.png
try:
    _c34 = get_crop(34, 1344, 396)
    canvas.paste(_c34, (48, 886), _c34)
except Exception:
    pass
layout["8_1646_creator_followers"] = [48, 886, 1392, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_01_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-3/35_icon_8_7107_creator_followers.png
try:
    _c35 = get_crop(35, 1344, 396)
    canvas.paste(_c35, (48, 2074), _c35)
except Exception:
    pass
layout["8_7107_creator_followers"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_01_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-3/36_icon_7.34.png
try:
    _c36 = get_crop(36, 91, 58)
    canvas.paste(_c36, (16, 4), _c36)
except Exception:
    pass
layout["7.34"] = [16, 4, 107, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_01_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-3/37_icon_City.png
try:
    _c37 = get_crop(37, 1344, 396)
    canvas.paste(_c37, (48, 2074), _c37)
except Exception:
    pass
layout["City"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_01_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-3/38_icon_Sat_May_4_._11_59_PM_EDT.png
try:
    _c38 = get_crop(38, 1344, 346)
    canvas.paste(_c38, (48, 2470), _c38)
except Exception:
    pass
layout["Sat;_May_4_._11:59_PM_EDT"] = [48, 2470, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_01_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-3/39_text_More_events_you_II_love.png
try:
    _c39 = get_crop(39, 1344, 396)
    canvas.paste(_c39, (48, 490), _c39)
except Exception:
    pass
layout["More_events_you'II_love"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_01_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-3/40_text_vLt.png
try:
    _c40 = get_crop(40, 59, 49)
    canvas.paste(_c40, (42, 2535), _c40)
except Exception:
    pass
layout["vLt"] = [42, 2535, 101, 2584]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_01_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-3/41_clickable_Favorites.png
try:
    _c41 = get_crop(41, 288, 156)
    canvas.paste(_c41, (576, 2804), _c41)
except Exception:
    pass
layout["Favorites"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_01_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-3/42_clickable_Tickets.png
try:
    _c42 = get_crop(42, 288, 156)
    canvas.paste(_c42, (864, 2804), _c42)
except Exception:
    pass
layout["Tickets"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_01_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-3/43_clickable_More.png
try:
    _c43 = get_crop(43, 288, 156)
    canvas.paste(_c43, (1152, 2804), _c43)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]
