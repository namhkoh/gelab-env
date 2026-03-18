# page_id: page_eventbrite_b45cca13f24546f9824a1ca2aab19c63_01
# screenshot: 2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-3.png
# step_index: 1/11
# task: Open Eventbrite. Search for "Art". Filter for events in New York. Select first recommended event. Save it to wishlist. What is the duration of the event?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw background and structural UI elements for the page
# Available variables: canvas (PIL Image), draw (PIL ImageDraw), font_sm/font_md/font_lg/font_xl

# Colors
BG = "#FFFFFF"
STATUS_BG = "#E9EAED"        # status bar background
HEADER_BG = "#FFFFFF"        # header / toolbar background
DIVIDER = "#E6E6E9"          # subtle dividers
CARD_SHADOW = "#F4F5F7"      # card shadow / lift
CARD_BG = "#FFFFFF"          # card background
BOTTOM_BG = "#FFFFFF"        # bottom nav background
BOTTOM_DIV = "#E6E6E9"

W = canvas.width
H = canvas.height

# Fill full background
draw.rectangle([(0, 0), (W, H)], fill=BG)

# Status bar area (top ~80px)
status_h = 80
draw.rectangle([(0, 0), (W, status_h)], fill=STATUS_BG)

# Header/toolbar area under status bar
header_top = status_h
header_bottom = 168
draw.rectangle([(0, header_top), (W, header_bottom)], fill=HEADER_BG)
# header bottom divider
draw.line([(24, header_bottom), (W-24, header_bottom)], fill=DIVIDER, width=1)

# Page title area (left intentionally blank for text to be pasted on top)
# Add a faint underline where the title section meets the list
title_bottom = 300
draw.line([(48, title_bottom), (W-48, title_bottom)], fill=DIVIDER, width=1)

# Event list card positions (match detected list item tops)
card_x0 = 48
card_x1 = W - 48
card_h = 396
card_tops = [490, 886, 1282, 1678, 2074, 2470]

for y in card_tops:
    # subtle shadow offset below each card
    shadow_offset = 6
    draw.rounded_rectangle(
        [(card_x0 + shadow_offset, y + shadow_offset),
         (card_x1 + shadow_offset, y + card_h + shadow_offset)],
        radius=14, fill=CARD_SHADOW, outline=None
    )
    # main card background (rounded)
    draw.rounded_rectangle(
        [(card_x0, y), (card_x1, y + card_h)],
        radius=12, fill=CARD_BG, outline=DIVIDER, width=1
    )
    # subtle separator line above each card (helps visually separate cards)
    draw.line([(card_x0 + 8, y - 18), (card_x1 - 8, y - 18)], fill=DIVIDER, width=1)

# Bottom navigation bar background and divider (approx area at bottom)
bottom_nav_top = 2804
draw.rectangle([(0, bottom_nav_top), (W, H)], fill=BOTTOM_BG)
draw.line([(0, bottom_nav_top), (W, bottom_nav_top)], fill=BOTTOM_DIV, width=1)

# Add faint horizontal separators between major sections further up the page
# (e.g., after a few cards to indicate grouping)
draw.line([(48, 1282 - 28), (W - 48, 1282 - 28)], fill=DIVIDER, width=1)
draw.line([(48, 2074 - 28), (W - 48, 2074 - 28)], fill=DIVIDER, width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_01_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-3/00_icon_YG.png
try:
    _c0 = get_crop(0, 1344, 396)
    canvas.paste(_c0, (48, 2074), _c0)
except Exception:
    pass
layout["YG"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_01_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-3/01_icon_Online.png
try:
    _c1 = get_crop(1, 1344, 396)
    canvas.paste(_c1, (48, 490), _c1)
except Exception:
    pass
layout["Online"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_01_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-3/02_icon_Online.png
try:
    _c2 = get_crop(2, 1344, 396)
    canvas.paste(_c2, (48, 1678), _c2)
except Exception:
    pass
layout["Online"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_01_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-3/03_icon_Q_Search_events.png
try:
    _c3 = get_crop(3, 1179, 144)
    canvas.paste(_c3, (195, 93), _c3)
except Exception:
    pass
layout["Q_Search_events"] = [195, 93, 1374, 237]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_01_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-3/04_icon_Or.png
try:
    _c4 = get_crop(4, 288, 156)
    canvas.paste(_c4, (288, 2804), _c4)
except Exception:
    pass
layout["Or,"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_01_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-3/05_icon_Understanding_Grief_and_Loss.png
try:
    _c5 = get_crop(5, 1344, 396)
    canvas.paste(_c5, (48, 1282), _c5)
except Exception:
    pass
layout["Understanding_Grief_and_L"] = [48, 1282, 1392, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_01_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-3/06_icon_Loss.png
try:
    _c6 = get_crop(6, 144, 125)
    canvas.paste(_c6, (1140, 2345), _c6)
except Exception:
    pass
layout["Loss"] = [1140, 2345, 1284, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_01_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-3/07_icon_Favorite_button.png
try:
    _c7 = get_crop(7, 144, 125)
    canvas.paste(_c7, (1140, 1949), _c7)
except Exception:
    pass
layout["Favorite_button"] = [1140, 1949, 1284, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_01_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-3/08_icon_7.04.png
try:
    _c8 = get_crop(8, 113, 106)
    canvas.paste(_c8, (35, 118), _c8)
except Exception:
    pass
layout["7.04"] = [35, 118, 148, 224]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_01_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-3/09_icon_Overflow_menu_button.png
try:
    _c9 = get_crop(9, 144, 125)
    canvas.paste(_c9, (1284, 2345), _c9)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 2345, 1428, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_01_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-3/10_icon_Home.png
try:
    _c10 = get_crop(10, 288, 156)
    canvas.paste(_c10, (0, 2804), _c10)
except Exception:
    pass
layout["Home"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_01_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-3/11_icon_Overflow_menu_button.png
try:
    _c11 = get_crop(11, 144, 139)
    canvas.paste(_c11, (1284, 747), _c11)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 747, 1428, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_01_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-3/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 62, 59)
    canvas.paste(_c12, (311, 3), _c12)
except Exception:
    pass
layout["icon_12"] = [311, 3, 373, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_01_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-3/13_icon_Overflow_menu_button.png
try:
    _c13 = get_crop(13, 144, 139)
    canvas.paste(_c13, (1284, 1143), _c13)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 1143, 1428, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_01_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-3/14_icon_7.04.png
try:
    _c14 = get_crop(14, 55, 60)
    canvas.paste(_c14, (183, 3), _c14)
except Exception:
    pass
layout["7.04"] = [183, 3, 238, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_01_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-3/15_icon_Overflow_menu_button.png
try:
    _c15 = get_crop(15, 144, 125)
    canvas.paste(_c15, (1284, 1949), _c15)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 1949, 1428, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_01_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-3/16_icon_Overflow_menu_button.png
try:
    _c16 = get_crop(16, 144, 139)
    canvas.paste(_c16, (1284, 1539), _c16)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 1539, 1428, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_01_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-3/17_icon_icon_17.png
try:
    _c17 = get_crop(17, 51, 61)
    canvas.paste(_c17, (248, 2), _c17)
except Exception:
    pass
layout["icon_17"] = [248, 2, 299, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_01_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-3/18_icon_Favorite_button.png
try:
    _c18 = get_crop(18, 144, 139)
    canvas.paste(_c18, (1140, 747), _c18)
except Exception:
    pass
layout["Favorite_button"] = [1140, 747, 1284, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_01_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-3/19_icon_1252_creator_followers.png
try:
    _c19 = get_crop(19, 1344, 396)
    canvas.paste(_c19, (48, 886), _c19)
except Exception:
    pass
layout["1252_creator_followers"] = [48, 886, 1392, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_01_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-3/20_icon_Working_with_Grief_and_Loss.png
try:
    _c20 = get_crop(20, 1344, 396)
    canvas.paste(_c20, (48, 490), _c20)
except Exception:
    pass
layout["Working_with_Grief_and_Lo"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_01_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-3/21_icon_icon_21.png
try:
    _c21 = get_crop(21, 48, 55)
    canvas.paste(_c21, (1321, 6), _c21)
except Exception:
    pass
layout["icon_21"] = [1321, 6, 1369, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_01_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-3/22_icon_Favorite_button.png
try:
    _c22 = get_crop(22, 144, 139)
    canvas.paste(_c22, (1140, 1143), _c22)
except Exception:
    pass
layout["Favorite_button"] = [1140, 1143, 1284, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_01_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-3/23_icon_icon_23.png
try:
    _c23 = get_crop(23, 95, 62)
    canvas.paste(_c23, (1211, 2), _c23)
except Exception:
    pass
layout["icon_23"] = [1211, 2, 1306, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_01_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-3/24_icon_5.00AM_EST.png
try:
    _c24 = get_crop(24, 1344, 396)
    canvas.paste(_c24, (48, 1678), _c24)
except Exception:
    pass
layout["5.00AM_EST"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_01_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-3/25_icon_Online_events.png
try:
    _c25 = get_crop(25, 288, 156)
    canvas.paste(_c25, (576, 2804), _c25)
except Exception:
    pass
layout["Online_events"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_01_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-3/26_icon_Favorite_button.png
try:
    _c26 = get_crop(26, 144, 139)
    canvas.paste(_c26, (1140, 1539), _c26)
except Exception:
    pass
layout["Favorite_button"] = [1140, 1539, 1284, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_01_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-3/27_icon_7.04.png
try:
    _c27 = get_crop(27, 56, 61)
    canvas.paste(_c27, (116, 2), _c27)
except Exception:
    pass
layout["7.04"] = [116, 2, 172, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_01_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-3/28_icon_Q_Search_events.png
try:
    _c28 = get_crop(28, 44, 57)
    canvas.paste(_c28, (385, 6), _c28)
except Exception:
    pass
layout["Q_Search_events"] = [385, 6, 429, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_01_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-3/29_icon_Loss.png
try:
    _c29 = get_crop(29, 288, 156)
    canvas.paste(_c29, (864, 2804), _c29)
except Exception:
    pass
layout["Loss"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_01_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-3/30_icon_Online.png
try:
    _c30 = get_crop(30, 112, 54)
    canvas.paste(_c30, (390, 703), _c30)
except Exception:
    pass
layout["Online"] = [390, 703, 502, 757]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_01_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-3/31_icon_Ur.png
try:
    _c31 = get_crop(31, 59, 59)
    canvas.paste(_c31, (388, 2641), _c31)
except Exception:
    pass
layout["Ur"] = [388, 2641, 447, 2700]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_01_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-3/32_icon_Online_events.png
try:
    _c32 = get_crop(32, 586, 117)
    canvas.paste(_c32, (427, 2651), _c32)
except Exception:
    pass
layout["Online_events"] = [427, 2651, 1013, 2768]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_01_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-3/33_icon_Understanding_your_Grief_and_Loss.png
try:
    _c33 = get_crop(33, 1344, 396)
    canvas.paste(_c33, (48, 2074), _c33)
except Exception:
    pass
layout["Understanding_your_Grief_"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_01_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-3/34_icon_Grief_and.png
try:
    _c34 = get_crop(34, 1344, 396)
    canvas.paste(_c34, (48, 1282), _c34)
except Exception:
    pass
layout["Grief_and"] = [48, 1282, 1392, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_01_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-3/35_icon_Grief_and.png
try:
    _c35 = get_crop(35, 1344, 396)
    canvas.paste(_c35, (48, 886), _c35)
except Exception:
    pass
layout["Grief_and"] = [48, 886, 1392, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_01_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-3/36_text_7.04.png
try:
    _c36 = get_crop(36, 92, 41)
    canvas.paste(_c36, (22, 17), _c36)
except Exception:
    pass
layout["7.04"] = [22, 17, 114, 58]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_01_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-3/37_text_More_events_you_II_love.png
try:
    _c37 = get_crop(37, 1344, 396)
    canvas.paste(_c37, (48, 490), _c37)
except Exception:
    pass
layout["More_events_you'II_love"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_01_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-3/38_text_Sat.png
try:
    _c38 = get_crop(38, 77, 45)
    canvas.paste(_c38, (390, 2583), _c38)
except Exception:
    pass
layout["Sat,"] = [390, 2583, 467, 2628]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_01_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-3/39_text_5.00_AM_EDT.png
try:
    _c39 = get_crop(39, 1344, 346)
    canvas.paste(_c39, (48, 2470), _c39)
except Exception:
    pass
layout["5.00_AM_EDT"] = [48, 2470, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_01_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-3/40_text_Loss.png
try:
    _c40 = get_crop(40, 110, 57)
    canvas.paste(_c40, (1031, 2646), _c40)
except Exception:
    pass
layout["Loss"] = [1031, 2646, 1141, 2703]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_01_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-3/41_clickable_More.png
try:
    _c41 = get_crop(41, 288, 156)
    canvas.paste(_c41, (1152, 2804), _c41)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]
