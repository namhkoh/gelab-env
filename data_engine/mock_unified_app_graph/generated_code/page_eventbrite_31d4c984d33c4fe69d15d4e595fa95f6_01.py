# page_id: page_eventbrite_31d4c984d33c4fe69d15d4e595fa95f6_01
# screenshot: 2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-3.png
# step_index: 1/14
# task: Open Eventbrite. Look for 'community events' in 'Chicago'. Select the first event happening tomorrow that is not promoted. Check if they have an option for 'refund policy'.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# fill overall background (dominant page color - off-white)
draw.rectangle([(0, 0), (1440, 2960)], fill="#FFFFFF")

# status bar area (top ~80px) - light gray background
draw.rectangle([(0, 0), (1440, 80)], fill="#D6D6D6")

# subtle divider under status bar
draw.line([(0, 80), (1440, 80)], fill="#CFCFCF", width=1)

# header/toolbar background (area behind the search bar and logo)
header_top = 80
header_bottom = 240
draw.rectangle([(0, header_top), (1440, header_bottom)], fill="#FFFFFF")
# header bottom divider
draw.line([(24, header_bottom), (1440-24, header_bottom)], fill="#EFEFF2", width=1)

# Card-like section containers for each list item (rounded rects)
card_x1 = 48
card_x2 = 1392
card_width = card_x2 - card_x1
card_height = 396
card_radius = 12

card_tops = [490, 886, 1282, 1678, 2074]
for y in card_tops:
    # subtle background for card
    draw.rounded_rectangle(
        [(card_x1, y), (card_x2, y + card_height)],
        radius=card_radius,
        fill="#FFFFFF",
        outline="#EFEFF4",
        width=1
    )
    # very light bottom shadow line to separate cards
    shadow_y = y + card_height
    draw.line([(card_x1 + 8, shadow_y), (card_x2 - 8, shadow_y)], fill="#F6F6F8", width=2)

# separators between sections (extra faint horizontal rules)
separators = [card_tops[0] - 24, card_tops[1] - 24, card_tops[2] - 24, card_tops[3] - 24, card_tops[4] - 24]
for sy in separators:
    draw.line([(48, sy), (1392, sy)], fill="#FBFBFC", width=1)

# subtle left page margin guideline (visual rhythm, not an icon)
draw.line([(48, 300), (48, 2600)], fill="#FBFBFC", width=1)

# bottom navigation area background (nav bar)
nav_top = 2804
nav_bottom = 2960
draw.rectangle([(0, nav_top), (1440, nav_bottom)], fill="#FFFFFF")
# top divider for nav bar
draw.line([(0, nav_top), (1440, nav_top)], fill="#EDECEC", width=1)

# a soft floating card background behind the center-bottom "Find" pill area
# (kept neutral and abstract; actual pill icon/text will be pasted on top)
floating_w = 760
floating_h = 120
floating_x = (1440 - floating_w) // 2
floating_y = 2320
draw.rounded_rectangle(
    [(floating_x, floating_y), (floating_x + floating_w, floating_y + floating_h)],
    radius=48,
    fill="#FFFFFF",
    outline="#E9E9EA",
    width=1
)
# small shadow under the floating card
draw.line([(floating_x + 12, floating_y + floating_h), (floating_x + floating_w - 12, floating_y + floating_h)], fill="#F2F2F4", width=3)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_01_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-3/00_icon_YG.png
try:
    _c0 = get_crop(0, 1344, 396)
    canvas.paste(_c0, (48, 2074), _c0)
except Exception:
    pass
layout["YG"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_01_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-3/01_icon_Online.png
try:
    _c1 = get_crop(1, 1344, 396)
    canvas.paste(_c1, (48, 490), _c1)
except Exception:
    pass
layout["Online"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_01_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-3/02_icon_Online.png
try:
    _c2 = get_crop(2, 1344, 396)
    canvas.paste(_c2, (48, 1678), _c2)
except Exception:
    pass
layout["Online"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_01_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-3/03_icon_Search_events.png
try:
    _c3 = get_crop(3, 1179, 144)
    canvas.paste(_c3, (195, 93), _c3)
except Exception:
    pass
layout["Search_events"] = [195, 93, 1374, 237]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_01_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-3/04_icon_Or.png
try:
    _c4 = get_crop(4, 288, 156)
    canvas.paste(_c4, (288, 2804), _c4)
except Exception:
    pass
layout["Or,"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_01_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-3/05_icon_8_1252_creator_followers.png
try:
    _c5 = get_crop(5, 1344, 396)
    canvas.paste(_c5, (48, 1282), _c5)
except Exception:
    pass
layout["8_1252_creator_followers"] = [48, 1282, 1392, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_01_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-3/06_icon_Loss.png
try:
    _c6 = get_crop(6, 144, 125)
    canvas.paste(_c6, (1140, 2345), _c6)
except Exception:
    pass
layout["Loss"] = [1140, 2345, 1284, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_01_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-3/07_icon_Favorite_button.png
try:
    _c7 = get_crop(7, 144, 125)
    canvas.paste(_c7, (1140, 1949), _c7)
except Exception:
    pass
layout["Favorite_button"] = [1140, 1949, 1284, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_01_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-3/08_icon_8.06.png
try:
    _c8 = get_crop(8, 114, 106)
    canvas.paste(_c8, (34, 118), _c8)
except Exception:
    pass
layout["8.06"] = [34, 118, 148, 224]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_01_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-3/09_icon_Overflow_menu_button.png
try:
    _c9 = get_crop(9, 144, 125)
    canvas.paste(_c9, (1284, 2345), _c9)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 2345, 1428, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_01_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-3/10_icon_Home.png
try:
    _c10 = get_crop(10, 288, 156)
    canvas.paste(_c10, (0, 2804), _c10)
except Exception:
    pass
layout["Home"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_01_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-3/11_icon_Overflow_menu_button.png
try:
    _c11 = get_crop(11, 144, 139)
    canvas.paste(_c11, (1284, 747), _c11)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 747, 1428, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_01_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-3/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 61, 59)
    canvas.paste(_c12, (311, 3), _c12)
except Exception:
    pass
layout["icon_12"] = [311, 3, 372, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_01_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-3/13_icon_Overflow_menu_button.png
try:
    _c13 = get_crop(13, 144, 139)
    canvas.paste(_c13, (1284, 1143), _c13)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 1143, 1428, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_01_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-3/14_icon_Overflow_menu_button.png
try:
    _c14 = get_crop(14, 144, 125)
    canvas.paste(_c14, (1284, 1949), _c14)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 1949, 1428, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_01_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-3/15_icon_Overflow_menu_button.png
try:
    _c15 = get_crop(15, 144, 139)
    canvas.paste(_c15, (1284, 1539), _c15)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 1539, 1428, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_01_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-3/16_icon_8.06.png
try:
    _c16 = get_crop(16, 55, 60)
    canvas.paste(_c16, (183, 3), _c16)
except Exception:
    pass
layout["8.06"] = [183, 3, 238, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_01_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-3/17_icon_icon_17.png
try:
    _c17 = get_crop(17, 51, 60)
    canvas.paste(_c17, (248, 2), _c17)
except Exception:
    pass
layout["icon_17"] = [248, 2, 299, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_01_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-3/18_icon_Favorite_button.png
try:
    _c18 = get_crop(18, 144, 139)
    canvas.paste(_c18, (1140, 747), _c18)
except Exception:
    pass
layout["Favorite_button"] = [1140, 747, 1284, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_01_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-3/19_icon_1252_creator_followers.png
try:
    _c19 = get_crop(19, 1344, 396)
    canvas.paste(_c19, (48, 886), _c19)
except Exception:
    pass
layout["1252_creator_followers"] = [48, 886, 1392, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_01_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-3/20_icon_Working_with_Grief_and_Loss.png
try:
    _c20 = get_crop(20, 1344, 396)
    canvas.paste(_c20, (48, 490), _c20)
except Exception:
    pass
layout["Working_with_Grief_and_Lo"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_01_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-3/21_icon_icon_21.png
try:
    _c21 = get_crop(21, 48, 55)
    canvas.paste(_c21, (1321, 6), _c21)
except Exception:
    pass
layout["icon_21"] = [1321, 6, 1369, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_01_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-3/22_icon_Favorite_button.png
try:
    _c22 = get_crop(22, 144, 139)
    canvas.paste(_c22, (1140, 1143), _c22)
except Exception:
    pass
layout["Favorite_button"] = [1140, 1143, 1284, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_01_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-3/23_icon_icon_23.png
try:
    _c23 = get_crop(23, 94, 61)
    canvas.paste(_c23, (1211, 2), _c23)
except Exception:
    pass
layout["icon_23"] = [1211, 2, 1305, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_01_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-3/24_icon_5.00AM_EST.png
try:
    _c24 = get_crop(24, 1344, 396)
    canvas.paste(_c24, (48, 1678), _c24)
except Exception:
    pass
layout["5.00AM_EST"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_01_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-3/25_icon_Online_events.png
try:
    _c25 = get_crop(25, 288, 156)
    canvas.paste(_c25, (576, 2804), _c25)
except Exception:
    pass
layout["Online_events"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_01_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-3/26_icon_8.06.png
try:
    _c26 = get_crop(26, 56, 62)
    canvas.paste(_c26, (116, 2), _c26)
except Exception:
    pass
layout["8.06"] = [116, 2, 172, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_01_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-3/27_icon_Favorite_button.png
try:
    _c27 = get_crop(27, 144, 139)
    canvas.paste(_c27, (1140, 1539), _c27)
except Exception:
    pass
layout["Favorite_button"] = [1140, 1539, 1284, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_01_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-3/28_icon_icon_28.png
try:
    _c28 = get_crop(28, 44, 57)
    canvas.paste(_c28, (385, 6), _c28)
except Exception:
    pass
layout["icon_28"] = [385, 6, 429, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_01_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-3/29_icon_Loss.png
try:
    _c29 = get_crop(29, 288, 156)
    canvas.paste(_c29, (864, 2804), _c29)
except Exception:
    pass
layout["Loss"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_01_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-3/30_icon_Online.png
try:
    _c30 = get_crop(30, 112, 54)
    canvas.paste(_c30, (390, 703), _c30)
except Exception:
    pass
layout["Online"] = [390, 703, 502, 757]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_01_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-3/31_icon_Ur.png
try:
    _c31 = get_crop(31, 59, 59)
    canvas.paste(_c31, (388, 2641), _c31)
except Exception:
    pass
layout["Ur"] = [388, 2641, 447, 2700]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_01_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-3/32_icon_Online_events.png
try:
    _c32 = get_crop(32, 586, 117)
    canvas.paste(_c32, (427, 2651), _c32)
except Exception:
    pass
layout["Online_events"] = [427, 2651, 1013, 2768]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_01_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-3/33_icon_Understanding_your_Grief_and_Loss.png
try:
    _c33 = get_crop(33, 1344, 396)
    canvas.paste(_c33, (48, 2074), _c33)
except Exception:
    pass
layout["Understanding_your_Grief_"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_01_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-3/34_text_8.06.png
try:
    _c34 = get_crop(34, 91, 43)
    canvas.paste(_c34, (20, 17), _c34)
except Exception:
    pass
layout["8.06"] = [20, 17, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_01_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-3/35_text_More_events_you_II_love.png
try:
    _c35 = get_crop(35, 1344, 396)
    canvas.paste(_c35, (48, 490), _c35)
except Exception:
    pass
layout["More_events_you'II_love"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_01_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-3/36_text_Sat.png
try:
    _c36 = get_crop(36, 77, 45)
    canvas.paste(_c36, (390, 2583), _c36)
except Exception:
    pass
layout["Sat,"] = [390, 2583, 467, 2628]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_01_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-3/37_text_5.00_AM_EDT.png
try:
    _c37 = get_crop(37, 1344, 346)
    canvas.paste(_c37, (48, 2470), _c37)
except Exception:
    pass
layout["5.00_AM_EDT"] = [48, 2470, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_01_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-3/38_text_Loss.png
try:
    _c38 = get_crop(38, 110, 57)
    canvas.paste(_c38, (1031, 2646), _c38)
except Exception:
    pass
layout["Loss"] = [1031, 2646, 1141, 2703]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_01_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-3/39_clickable_More.png
try:
    _c39 = get_crop(39, 288, 156)
    canvas.paste(_c39, (1152, 2804), _c39)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]
