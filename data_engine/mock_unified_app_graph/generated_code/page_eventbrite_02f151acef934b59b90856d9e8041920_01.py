# page_id: page_eventbrite_02f151acef934b59b90856d9e8041920_01
# screenshot: 2024_4_24_17_24_02f151acef934b59b90856d9e8041920-3.png
# step_index: 1/11
# task: Open Eventbrite. Check the "Tech" events happening this month. Open the first event and check its date and time.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw UI background and structural elements for the mobile Eventbrite-like page
# Available objects: canvas (PIL Image 1440x2960), draw (ImageDraw), fonts: font_sm, font_md, font_lg, font_xl

# Base background
draw.rectangle((0, 0, 1440, 2960), fill="#FFFFFF")

# Status bar area (top ~50px)
status_h = 50
draw.rectangle((0, 0, 1440, status_h), fill="#CFCFCF")

# Header / search bar area
# Draw a subtle background band behind the header region
header_top = status_h
header_bottom = 240
draw.rectangle((0, header_top, 1440, header_bottom), fill="#FFFFFF")

# Search field background (rounded)
search_left = 195
search_top = 70
search_right = search_left + 1179
search_bottom = search_top + 144
search_radius = 72
draw.rounded_rectangle(
    (search_left, search_top, search_right, search_bottom),
    radius=search_radius,
    fill="#F7F7F9",
    outline="#E6E6EB",
    width=2
)

# Thin divider under header area
draw.line((48, header_bottom - 10, 1392, header_bottom - 10), fill="#E9E9EF", width=1)

# Event cards: draw subtle card backgrounds and soft shadows for each row group
card_x1 = 48
card_x2 = 1392
card_height = 396
card_radius = 12

row_tops = [490, 886, 1282, 1678, 2074, 2470]
for y in row_tops:
    # Soft shadow (very light, offset down/right)
    shadow_box = (card_x1 + 6, y + 8, card_x2 + 6, y + card_height + 8)
    draw.rectangle(shadow_box, fill="#F4F4F6")
    # Card background (rounded)
    card_box = (card_x1, y, card_x2, y + card_height)
    draw.rounded_rectangle(card_box, radius=card_radius, fill="#FFFFFF")

    # Top separator line for clarity between stacked cards
    draw.line((card_x1 + 6, y, card_x2 - 6, y), fill="#F0F0F2", width=1)

# Additional faint separators between rows (across full content width)
separator_positions = [490, 886, 1282, 1678, 2074, 2470, 2804]
for sy in separator_positions:
    draw.line((36, sy - 6, 1404, sy - 6), fill="#EFEFF2", width=1)

# Bottom navigation bar background and top divider
nav_top = 2804
draw.rectangle((0, nav_top, 1440, 2960), fill="#FFFFFF")
draw.line((0, nav_top, 1440, nav_top), fill="#E9E9EF", width=2)

# Subtle left and right edge padding shadows (page edges)
draw.rectangle((0, status_h, 12, 2960 - 160), fill="#FFFFFF")
draw.rectangle((1428, status_h, 1440, 2960 - 160), fill="#FFFFFF")

# Light content-area banner strip (for any colored section backgrounds)
# (kept subtle and generic so it does not duplicate any specific content)
banner_y = 2420
draw.rectangle((48, banner_y, 1392, banner_y + 56), fill="#FFFFFF")

# Final hairline at very top to separate status bar and header
draw.line((0, status_h, 1440, status_h), fill="#BDBDC1", width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_01_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-3/00_icon_Online.png
try:
    _c0 = get_crop(0, 1344, 396)
    canvas.paste(_c0, (48, 490), _c0)
except Exception:
    pass
layout["Online"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_01_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-3/01_icon_Online.png
try:
    _c1 = get_crop(1, 1344, 396)
    canvas.paste(_c1, (48, 2074), _c1)
except Exception:
    pass
layout["Online"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_01_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-3/02_icon_Q_Search_events.png
try:
    _c2 = get_crop(2, 1179, 144)
    canvas.paste(_c2, (195, 93), _c2)
except Exception:
    pass
layout["Q_Search_events"] = [195, 93, 1374, 237]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_01_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-3/03_icon_Online.png
try:
    _c3 = get_crop(3, 1344, 396)
    canvas.paste(_c3, (48, 1678), _c3)
except Exception:
    pass
layout["Online"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_01_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-3/04_icon_OU7_LIGQLUI.png
try:
    _c4 = get_crop(4, 288, 156)
    canvas.paste(_c4, (288, 2804), _c4)
except Exception:
    pass
layout["OU7_LIGQLUI"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_01_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-3/05_icon_Favorite_button.png
try:
    _c5 = get_crop(5, 144, 125)
    canvas.paste(_c5, (1140, 1949), _c5)
except Exception:
    pass
layout["Favorite_button"] = [1140, 1949, 1284, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_01_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-3/06_icon_Home.png
try:
    _c6 = get_crop(6, 288, 156)
    canvas.paste(_c6, (0, 2804), _c6)
except Exception:
    pass
layout["Home"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_01_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-3/07_icon_Favorite_button.png
try:
    _c7 = get_crop(7, 144, 125)
    canvas.paste(_c7, (1140, 2345), _c7)
except Exception:
    pass
layout["Favorite_button"] = [1140, 2345, 1284, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_01_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-3/08_icon_5.24.png
try:
    _c8 = get_crop(8, 108, 102)
    canvas.paste(_c8, (38, 120), _c8)
except Exception:
    pass
layout["5.24"] = [38, 120, 146, 222]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_01_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-3/09_icon_Overflow_menu_button.png
try:
    _c9 = get_crop(9, 144, 125)
    canvas.paste(_c9, (1284, 2345), _c9)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 2345, 1428, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_01_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-3/10_icon_Overflow_menu_button.png
try:
    _c10 = get_crop(10, 144, 139)
    canvas.paste(_c10, (1284, 1539), _c10)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 1539, 1428, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_01_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-3/11_icon_5.24.png
try:
    _c11 = get_crop(11, 54, 61)
    canvas.paste(_c11, (184, 2), _c11)
except Exception:
    pass
layout["5.24"] = [184, 2, 238, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_01_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-3/12_icon_Overflow_menu_button.png
try:
    _c12 = get_crop(12, 144, 139)
    canvas.paste(_c12, (1284, 747), _c12)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 747, 1428, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_01_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-3/13_icon_Overflow_menu_button.png
try:
    _c13 = get_crop(13, 144, 139)
    canvas.paste(_c13, (1284, 1143), _c13)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 1143, 1428, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_01_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-3/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 60, 59)
    canvas.paste(_c14, (312, 3), _c14)
except Exception:
    pass
layout["icon_14"] = [312, 3, 372, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_01_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-3/15_icon_Overflow_menu_button.png
try:
    _c15 = get_crop(15, 144, 125)
    canvas.paste(_c15, (1284, 1949), _c15)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 1949, 1428, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_01_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-3/16_icon_Art_for_Grief_and_Loss.png
try:
    _c16 = get_crop(16, 1344, 396)
    canvas.paste(_c16, (48, 1282), _c16)
except Exception:
    pass
layout["Art_for_Grief_and_Loss"] = [48, 1282, 1392, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_01_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-3/17_icon_TomU_5i0.png
try:
    _c17 = get_crop(17, 586, 117)
    canvas.paste(_c17, (427, 2651), _c17)
except Exception:
    pass
layout["TomU'5i0"] = [427, 2651, 1013, 2768]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_01_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-3/18_icon_icon_18.png
try:
    _c18 = get_crop(18, 50, 59)
    canvas.paste(_c18, (248, 3), _c18)
except Exception:
    pass
layout["icon_18"] = [248, 3, 298, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_01_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-3/19_icon_Favorite_button.png
try:
    _c19 = get_crop(19, 144, 139)
    canvas.paste(_c19, (1140, 747), _c19)
except Exception:
    pass
layout["Favorite_button"] = [1140, 747, 1284, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_01_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-3/20_icon_Senaratinn_Grief_and_Loss.png
try:
    _c20 = get_crop(20, 288, 156)
    canvas.paste(_c20, (864, 2804), _c20)
except Exception:
    pass
layout["Senaratinn_Grief;_and_Los"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_01_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-3/21_icon_icon_21.png
try:
    _c21 = get_crop(21, 48, 53)
    canvas.paste(_c21, (1321, 7), _c21)
except Exception:
    pass
layout["icon_21"] = [1321, 7, 1369, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_01_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-3/22_icon_Working_with_Grief_and_Loss.png
try:
    _c22 = get_crop(22, 1344, 396)
    canvas.paste(_c22, (48, 490), _c22)
except Exception:
    pass
layout["Working_with_Grief_and_Lo"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_01_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-3/23_icon_Favorite_button.png
try:
    _c23 = get_crop(23, 144, 139)
    canvas.paste(_c23, (1140, 1143), _c23)
except Exception:
    pass
layout["Favorite_button"] = [1140, 1143, 1284, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_01_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-3/24_icon_Favorite_button.png
try:
    _c24 = get_crop(24, 144, 139)
    canvas.paste(_c24, (1140, 1539), _c24)
except Exception:
    pass
layout["Favorite_button"] = [1140, 1539, 1284, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_01_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-3/25_icon_5.24.png
try:
    _c25 = get_crop(25, 58, 60)
    canvas.paste(_c25, (115, 3), _c25)
except Exception:
    pass
layout["5.24"] = [115, 3, 173, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_01_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-3/26_icon_icon_26.png
try:
    _c26 = get_crop(26, 68, 60)
    canvas.paste(_c26, (1212, 3), _c26)
except Exception:
    pass
layout["icon_26"] = [1212, 3, 1280, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_01_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-3/27_icon_S_00_AM_EDT.png
try:
    _c27 = get_crop(27, 1344, 396)
    canvas.paste(_c27, (48, 1678), _c27)
except Exception:
    pass
layout["S:00_AM_EDT"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_01_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-3/28_icon_5_O0_AM_EDT.png
try:
    _c28 = get_crop(28, 1344, 396)
    canvas.paste(_c28, (48, 2074), _c28)
except Exception:
    pass
layout["5:O0_AM_EDT"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_01_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-3/29_icon_Online.png
try:
    _c29 = get_crop(29, 112, 53)
    canvas.paste(_c29, (390, 1496), _c29)
except Exception:
    pass
layout["Online"] = [390, 1496, 502, 1549]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_01_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-3/30_icon_Q_Search_events.png
try:
    _c30 = get_crop(30, 44, 56)
    canvas.paste(_c30, (385, 7), _c30)
except Exception:
    pass
layout["Q_Search_events"] = [385, 7, 429, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_01_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-3/31_icon_Online.png
try:
    _c31 = get_crop(31, 112, 54)
    canvas.paste(_c31, (390, 703), _c31)
except Exception:
    pass
layout["Online"] = [390, 703, 502, 757]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_01_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-3/32_icon_Understanding_Grief_and_Loss.png
try:
    _c32 = get_crop(32, 1344, 396)
    canvas.paste(_c32, (48, 886), _c32)
except Exception:
    pass
layout["Understanding_Grief_and_L"] = [48, 886, 1392, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_01_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-3/33_icon_icon_33.png
try:
    _c33 = get_crop(33, 42, 56)
    canvas.paste(_c33, (1272, 5), _c33)
except Exception:
    pass
layout["icon_33"] = [1272, 5, 1314, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_01_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-3/34_icon_suppoloyed_Orilee_herapeeticrarard_Outh_.png
try:
    _c34 = get_crop(34, 1344, 396)
    canvas.paste(_c34, (48, 1282), _c34)
except Exception:
    pass
layout["suppoloyed_Orilee__herape"] = [48, 1282, 1392, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_01_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-3/35_icon_TomU_5i0.png
try:
    _c35 = get_crop(35, 288, 156)
    canvas.paste(_c35, (576, 2804), _c35)
except Exception:
    pass
layout["TomU'5i0"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_01_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-3/36_text_5.24.png
try:
    _c36 = get_crop(36, 92, 43)
    canvas.paste(_c36, (22, 17), _c36)
except Exception:
    pass
layout["5.24"] = [22, 17, 114, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_01_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-3/37_text_More_events_you_II_love.png
try:
    _c37 = get_crop(37, 1344, 396)
    canvas.paste(_c37, (48, 490), _c37)
except Exception:
    pass
layout["More_events_you'II_love"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_01_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-3/38_text_Thu_May_2_11_00_AM_EDT.png
try:
    _c38 = get_crop(38, 1344, 346)
    canvas.paste(_c38, (48, 2470), _c38)
except Exception:
    pass
layout["Thu,_May_2_+_11:00_AM_EDT"] = [48, 2470, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_01_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-3/39_clickable_More.png
try:
    _c39 = get_crop(39, 288, 156)
    canvas.paste(_c39, (1152, 2804), _c39)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]
