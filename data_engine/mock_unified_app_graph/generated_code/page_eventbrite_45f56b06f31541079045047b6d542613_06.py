# page_id: page_eventbrite_45f56b06f31541079045047b6d542613_06
# screenshot: 2024_4_23_19_27_45f56b06f31541079045047b6d542613-8.png
# step_index: 6/21
# task: Open Eventbrite. Search events 'Yoga session' in New York. Filter free events and set date from May 3 to May 6. What is the location of the first promoted event?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Fill overall background (dominant off-white)
canvas.paste((253,252,255), (0, 0, canvas.size[0], canvas.size[1]))

# Top status bar (approx 60px high) - light gray
status_bar_height = 60
draw.rectangle([(0, 0), (canvas.size[0], status_bar_height)], fill=(200, 200, 200))

# Header area below status bar
header_top = status_bar_height
header_bottom = 180
draw.rectangle([(0, header_top), (canvas.size[0], header_bottom)], fill=(255, 255, 255))

# Divider under header
draw.line([(24, header_bottom), (canvas.size[0]-24, header_bottom)], fill=(235,235,240), width=1)

# Bottom navigation bar background and subtle top divider/shadow
nav_top = 2800
draw.rectangle([(0, nav_top), (canvas.size[0], canvas.size[1])], fill=(255,255,255))
# subtle top shadow line
draw.line([(0, nav_top), (canvas.size[0], nav_top)], fill=(230,230,235), width=2)

# Card background style
card_x = 48
card_w = 1344
card_x2 = card_x + card_w
card_radius = 18
card_shadow_offset = 8

card_rows = [
    (490, 396),
    (886, 396),
    (1282, 396),
    (1678, 396),
    (2074, 396),
    (2470, 346),
]

# Draw card shadows and card fills (rounded rectangles)
for y, h in card_rows:
    # shadow
    shadow_box = [card_x, y + card_shadow_offset, card_x2, y + h + card_shadow_offset]
    # slightly purplish/gray shadow to mimic screenshot subtle shadow
    draw.rounded_rectangle(shadow_box, radius=card_radius, fill=(235,230,242))
    # main card (white)
    card_box = [card_x, y, card_x2, y + h]
    draw.rounded_rectangle(card_box, radius=card_radius, fill=(255,255,255))
    # subtle inner divider near bottom of each card area (not overlapping content)
    divider_y = y + h + 8
    draw.line([(card_x+12, divider_y), (card_x2-12, divider_y)], fill=(245,244,247), width=1)

# Large content/banner area near the very top content region (behind title area)
# (a faint tinted block under the "More events you'll love" heading)
banner_top = 220
banner_bottom = 360
draw.rectangle([(24, banner_top), (canvas.size[0]-24, banner_bottom)], fill=(255,255,255,0))

# Subtle full-page vertical divider (very faint) to guide layout (centered gutter)
gutter_x = 48
draw.line([(gutter_x, header_bottom+8), (gutter_x, canvas.size[1]-120)], fill=(250,250,252), width=1)

# Top area shading below status bar to indicate elevation (very subtle)
draw.line([(0, status_bar_height), (canvas.size[0], status_bar_height)], fill=(220,220,225), width=1)
draw.line([(0, status_bar_height+1), (canvas.size[0], status_bar_height+1)], fill=(245,245,247), width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_06_2024_4_23_19_27_45f56b06f31541079045047b6d542613-8/00_icon_New_York.png
try:
    _c0 = get_crop(0, 405, 117)
    canvas.paste(_c0, (518, 2651), _c0)
except Exception:
    pass
layout["New_York"] = [518, 2651, 923, 2768]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_06_2024_4_23_19_27_45f56b06f31541079045047b6d542613-8/01_icon_City.png
try:
    _c1 = get_crop(1, 1344, 396)
    canvas.paste(_c1, (48, 886), _c1)
except Exception:
    pass
layout["City,"] = [48, 886, 1392, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_06_2024_4_23_19_27_45f56b06f31541079045047b6d542613-8/02_icon_Search_events.png
try:
    _c2 = get_crop(2, 1179, 144)
    canvas.paste(_c2, (195, 93), _c2)
except Exception:
    pass
layout["Search_events"] = [195, 93, 1374, 237]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_06_2024_4_23_19_27_45f56b06f31541079045047b6d542613-8/03_icon_New_York.png
try:
    _c3 = get_crop(3, 144, 139)
    canvas.paste(_c3, (1140, 747), _c3)
except Exception:
    pass
layout["New_York"] = [1140, 747, 1284, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_06_2024_4_23_19_27_45f56b06f31541079045047b6d542613-8/04_icon_Conference_Connections.png
try:
    _c4 = get_crop(4, 144, 139)
    canvas.paste(_c4, (1140, 1935), _c4)
except Exception:
    pass
layout["Conference_Connections"] = [1140, 1935, 1284, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_06_2024_4_23_19_27_45f56b06f31541079045047b6d542613-8/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 49, 65)
    canvas.paste(_c5, (1153, 2), _c5)
except Exception:
    pass
layout["icon_5"] = [1153, 2, 1202, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_06_2024_4_23_19_27_45f56b06f31541079045047b6d542613-8/06_icon_New_York.png
try:
    _c6 = get_crop(6, 144, 123)
    canvas.paste(_c6, (1284, 1159), _c6)
except Exception:
    pass
layout["New_York"] = [1284, 1159, 1428, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_06_2024_4_23_19_27_45f56b06f31541079045047b6d542613-8/07_icon_Conference_Connections.png
try:
    _c7 = get_crop(7, 144, 123)
    canvas.paste(_c7, (1140, 2347), _c7)
except Exception:
    pass
layout["Conference_Connections"] = [1140, 2347, 1284, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_06_2024_4_23_19_27_45f56b06f31541079045047b6d542613-8/08_icon_New_York.png
try:
    _c8 = get_crop(8, 144, 123)
    canvas.paste(_c8, (1140, 1159), _c8)
except Exception:
    pass
layout["New_York"] = [1140, 1159, 1284, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_06_2024_4_23_19_27_45f56b06f31541079045047b6d542613-8/09_icon_New_York.png
try:
    _c9 = get_crop(9, 144, 139)
    canvas.paste(_c9, (1284, 747), _c9)
except Exception:
    pass
layout["New_York"] = [1284, 747, 1428, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_06_2024_4_23_19_27_45f56b06f31541079045047b6d542613-8/10_icon_Good_Afternoon_New_York.png
try:
    _c10 = get_crop(10, 1344, 396)
    canvas.paste(_c10, (48, 1678), _c10)
except Exception:
    pass
layout["Good_Afternoon_New_York"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_06_2024_4_23_19_27_45f56b06f31541079045047b6d542613-8/11_icon_Overflow_menu_button.png
try:
    _c11 = get_crop(11, 144, 123)
    canvas.paste(_c11, (1284, 2347), _c11)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 2347, 1428, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_06_2024_4_23_19_27_45f56b06f31541079045047b6d542613-8/12_icon_Pier_36.png
try:
    _c12 = get_crop(12, 288, 156)
    canvas.paste(_c12, (288, 2804), _c12)
except Exception:
    pass
layout["Pier_36"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_06_2024_4_23_19_27_45f56b06f31541079045047b6d542613-8/13_icon_Overflow_menu_button.png
try:
    _c13 = get_crop(13, 144, 139)
    canvas.paste(_c13, (1284, 1935), _c13)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 1935, 1428, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_06_2024_4_23_19_27_45f56b06f31541079045047b6d542613-8/14_icon_Overflow_menu_button.png
try:
    _c14 = get_crop(14, 144, 123)
    canvas.paste(_c14, (1284, 1555), _c14)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 1555, 1428, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_06_2024_4_23_19_27_45f56b06f31541079045047b6d542613-8/15_icon_139_creator_followers.png
try:
    _c15 = get_crop(15, 1344, 396)
    canvas.paste(_c15, (48, 1282), _c15)
except Exception:
    pass
layout["139_creator_followers"] = [48, 1282, 1392, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_06_2024_4_23_19_27_45f56b06f31541079045047b6d542613-8/16_icon_Medical_Hair_Loss_Therapy_Training.png
try:
    _c16 = get_crop(16, 1344, 396)
    canvas.paste(_c16, (48, 886), _c16)
except Exception:
    pass
layout["Medical_Hair_Loss_Therapy"] = [48, 886, 1392, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_06_2024_4_23_19_27_45f56b06f31541079045047b6d542613-8/17_icon_free_Grief_and_Loss_Healing_Circle.png
try:
    _c17 = get_crop(17, 1344, 396)
    canvas.paste(_c17, (48, 490), _c17)
except Exception:
    pass
layout["(free)_Grief_and_Loss_Hea"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_06_2024_4_23_19_27_45f56b06f31541079045047b6d542613-8/18_icon_Favorite_button.png
try:
    _c18 = get_crop(18, 144, 123)
    canvas.paste(_c18, (1140, 1555), _c18)
except Exception:
    pass
layout["Favorite_button"] = [1140, 1555, 1284, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_06_2024_4_23_19_27_45f56b06f31541079045047b6d542613-8/19_icon_icon_19.png
try:
    _c19 = get_crop(19, 97, 60)
    canvas.paste(_c19, (1216, 3), _c19)
except Exception:
    pass
layout["icon_19"] = [1216, 3, 1313, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_06_2024_4_23_19_27_45f56b06f31541079045047b6d542613-8/20_icon_ACHT_PART.png
try:
    _c20 = get_crop(20, 288, 156)
    canvas.paste(_c20, (0, 2804), _c20)
except Exception:
    pass
layout["'ACHT_PART"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_06_2024_4_23_19_27_45f56b06f31541079045047b6d542613-8/21_icon_VOSCHINO.png
try:
    _c21 = get_crop(21, 1344, 396)
    canvas.paste(_c21, (48, 1678), _c21)
except Exception:
    pass
layout["VOSCHINO"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_06_2024_4_23_19_27_45f56b06f31541079045047b6d542613-8/22_icon_Free.png
try:
    _c22 = get_crop(22, 1344, 396)
    canvas.paste(_c22, (48, 1282), _c22)
except Exception:
    pass
layout["Free"] = [48, 1282, 1392, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_06_2024_4_23_19_27_45f56b06f31541079045047b6d542613-8/23_icon_7.28.png
try:
    _c23 = get_crop(23, 101, 97)
    canvas.paste(_c23, (42, 123), _c23)
except Exception:
    pass
layout["7.28"] = [42, 123, 143, 220]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_06_2024_4_23_19_27_45f56b06f31541079045047b6d542613-8/24_icon_icon_24.png
try:
    _c24 = get_crop(24, 60, 58)
    canvas.paste(_c24, (312, 3), _c24)
except Exception:
    pass
layout["icon_24"] = [312, 3, 372, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_06_2024_4_23_19_27_45f56b06f31541079045047b6d542613-8/25_icon_7.28.png
try:
    _c25 = get_crop(25, 55, 60)
    canvas.paste(_c25, (183, 2), _c25)
except Exception:
    pass
layout["7.28"] = [183, 2, 238, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_06_2024_4_23_19_27_45f56b06f31541079045047b6d542613-8/26_icon_icon_26.png
try:
    _c26 = get_crop(26, 50, 60)
    canvas.paste(_c26, (248, 2), _c26)
except Exception:
    pass
layout["icon_26"] = [248, 2, 298, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_06_2024_4_23_19_27_45f56b06f31541079045047b6d542613-8/27_icon_icon_27.png
try:
    _c27 = get_crop(27, 48, 53)
    canvas.paste(_c27, (1321, 7), _c27)
except Exception:
    pass
layout["icon_27"] = [1321, 7, 1369, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_06_2024_4_23_19_27_45f56b06f31541079045047b6d542613-8/28_icon_Free.png
try:
    _c28 = get_crop(28, 1344, 396)
    canvas.paste(_c28, (48, 490), _c28)
except Exception:
    pass
layout["Free"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_06_2024_4_23_19_27_45f56b06f31541079045047b6d542613-8/29_icon_7.28.png
try:
    _c29 = get_crop(29, 59, 61)
    canvas.paste(_c29, (115, 2), _c29)
except Exception:
    pass
layout["7.28"] = [115, 2, 174, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_06_2024_4_23_19_27_45f56b06f31541079045047b6d542613-8/30_icon_Free.png
try:
    _c30 = get_crop(30, 130, 74)
    canvas.paste(_c30, (244, 560), _c30)
except Exception:
    pass
layout["Free"] = [244, 560, 374, 634]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_06_2024_4_23_19_27_45f56b06f31541079045047b6d542613-8/31_icon_icon_31.png
try:
    _c31 = get_crop(31, 44, 57)
    canvas.paste(_c31, (385, 6), _c31)
except Exception:
    pass
layout["icon_31"] = [385, 6, 429, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_06_2024_4_23_19_27_45f56b06f31541079045047b6d542613-8/32_icon_8_1646_creator_followers.png
try:
    _c32 = get_crop(32, 1344, 396)
    canvas.paste(_c32, (48, 886), _c32)
except Exception:
    pass
layout["8_1646_creator_followers"] = [48, 886, 1392, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_06_2024_4_23_19_27_45f56b06f31541079045047b6d542613-8/33_icon_free_Grief_and_Loss_Healing_Circle.png
try:
    _c33 = get_crop(33, 1344, 396)
    canvas.paste(_c33, (48, 490), _c33)
except Exception:
    pass
layout["(free)_Grief_and_Loss_Hea"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_06_2024_4_23_19_27_45f56b06f31541079045047b6d542613-8/34_icon_City.png
try:
    _c34 = get_crop(34, 1344, 396)
    canvas.paste(_c34, (48, 2074), _c34)
except Exception:
    pass
layout["City"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_06_2024_4_23_19_27_45f56b06f31541079045047b6d542613-8/35_icon_Tickets.png
try:
    _c35 = get_crop(35, 288, 156)
    canvas.paste(_c35, (864, 2804), _c35)
except Exception:
    pass
layout["Tickets"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_06_2024_4_23_19_27_45f56b06f31541079045047b6d542613-8/36_icon_HIPHOP_Dancehall_yacht_party_NEW_YORK.png
try:
    _c36 = get_crop(36, 1344, 346)
    canvas.paste(_c36, (48, 2470), _c36)
except Exception:
    pass
layout["HIPHOP_Dancehall_yacht_pa"] = [48, 2470, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_06_2024_4_23_19_27_45f56b06f31541079045047b6d542613-8/37_text_7.28.png
try:
    _c37 = get_crop(37, 91, 45)
    canvas.paste(_c37, (20, 15), _c37)
except Exception:
    pass
layout["7.28"] = [20, 15, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_06_2024_4_23_19_27_45f56b06f31541079045047b6d542613-8/38_text_More_events_you_II_love.png
try:
    _c38 = get_crop(38, 1344, 396)
    canvas.paste(_c38, (48, 490), _c38)
except Exception:
    pass
layout["More_events_you'II_love"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_06_2024_4_23_19_27_45f56b06f31541079045047b6d542613-8/39_text_Sat_Apr_27.png
try:
    _c39 = get_crop(39, 195, 43)
    canvas.paste(_c39, (390, 2525), _c39)
except Exception:
    pass
layout["Sat,_Apr_27"] = [390, 2525, 585, 2568]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_06_2024_4_23_19_27_45f56b06f31541079045047b6d542613-8/40_text_1_00_PM_EDT.png
try:
    _c40 = get_crop(40, 1344, 346)
    canvas.paste(_c40, (48, 2470), _c40)
except Exception:
    pass
layout["1:00_PM_EDT"] = [48, 2470, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_06_2024_4_23_19_27_45f56b06f31541079045047b6d542613-8/41_text_PPHc.png
try:
    _c41 = get_crop(41, 1344, 346)
    canvas.paste(_c41, (48, 2470), _c41)
except Exception:
    pass
layout["PPHc"] = [48, 2470, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_06_2024_4_23_19_27_45f56b06f31541079045047b6d542613-8/42_text_ACHT_PART.png
try:
    _c42 = get_crop(42, 288, 156)
    canvas.paste(_c42, (0, 2804), _c42)
except Exception:
    pass
layout["'ACHT_PART"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_06_2024_4_23_19_27_45f56b06f31541079045047b6d542613-8/43_clickable_Favorites.png
try:
    _c43 = get_crop(43, 288, 156)
    canvas.paste(_c43, (576, 2804), _c43)
except Exception:
    pass
layout["Favorites"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_06_2024_4_23_19_27_45f56b06f31541079045047b6d542613-8/44_clickable_More.png
try:
    _c44 = get_crop(44, 288, 156)
    canvas.paste(_c44, (1152, 2804), _c44)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]
