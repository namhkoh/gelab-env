# page_id: page_eventbrite_b28077cf24f341ff9de826ac8bd7fb2b_01
# screenshot: 2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-3.png
# step_index: 1/16
# task: Open Eventbrite. Explore 'Wellness' events in Washington. Filter to only show free events. Add the first non-promoted event to favorite and follow the organizer.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# draw overall background
draw.rectangle((0, 0, 1440, 2960), fill=(250, 250, 252))  # subtle off-white page background

# status bar area (top)
status_h = 56
draw.rectangle((0, 0, 1440, status_h), fill=(200, 200, 200))  # status bar background
# subtle divider under status bar
draw.line((0, status_h, 1440, status_h), fill=(225, 225, 227), width=1)

# search bar / toolbar area (rounded search background only - icons/text will be pasted later)
search_x, search_y = 195, 93
search_w, search_h = 1179, 144
search_bbox = (search_x, search_y, search_x + search_w, search_y + search_h)
draw.rounded_rectangle(search_bbox, radius=72, fill=(249, 249, 250), outline=(224, 224, 228), width=2)

# subtle divider under toolbar/search area
divider_y = search_y + search_h + 18
draw.line((48, divider_y, 1392, divider_y), fill=(238, 238, 241), width=1)

# Section card backgrounds: positions based on detected rows (x=48, width=1344)
card_x = 48
card_w = 1344
card_positions = [490, 886, 1282, 1678, 2074, 2470]  # y positions for list items / cards
card_heights = {490:396, 886:396, 1282:396, 1678:396, 2074:396, 2470:346}

for y in card_positions:
    h = card_heights.get(y, 396)
    # subtle drop shadow (very light)
    shadow_bbox = (card_x + 3, y + 6, card_x + card_w + 3, y + h + 6)
    draw.rounded_rectangle(shadow_bbox, radius=14, fill=(245, 245, 247), outline=None)
    # white card background
    card_bbox = (card_x, y, card_x + card_w, y + h)
    draw.rounded_rectangle(card_bbox, radius=12, fill=(255, 255, 255), outline=(240, 240, 245), width=1)
    # subtle separator line under card (within content margin)
    sep_y = y + h + 14
    draw.line((card_x + 8, sep_y, card_x + card_w - 8, sep_y), fill=(243, 243, 245), width=1)

# content area band for large title area (behind "More events you'll love" - don't draw text)
title_band_y = 240
title_band_h = 120
draw.rectangle((48, title_band_y, 1392, title_band_y + title_band_h), fill=(250, 250, 252))

# bottom navigation bar background
bottom_nav_y = 2804
draw.rectangle((0, bottom_nav_y, 1440, 2960), fill=(255, 255, 255))
# top divider for navigation bar
draw.line((0, bottom_nav_y, 1440, bottom_nav_y), fill=(230, 230, 233), width=1)

# subtle left/right page margins lines (very faint) to anchor layout edges
draw.line((48, 0, 48, 2960), fill=(250, 250, 252), width=1)
draw.line((1392, 0, 1392, 2960), fill=(250, 250, 252), width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_01_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-3/00_icon_Online.png
try:
    _c0 = get_crop(0, 1344, 396)
    canvas.paste(_c0, (48, 490), _c0)
except Exception:
    pass
layout["Online"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_01_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-3/01_icon_Online.png
try:
    _c1 = get_crop(1, 1344, 396)
    canvas.paste(_c1, (48, 2074), _c1)
except Exception:
    pass
layout["Online"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_01_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-3/02_icon_Search_events.png
try:
    _c2 = get_crop(2, 1179, 144)
    canvas.paste(_c2, (195, 93), _c2)
except Exception:
    pass
layout["Search_events"] = [195, 93, 1374, 237]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_01_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-3/03_icon_Online.png
try:
    _c3 = get_crop(3, 1344, 396)
    canvas.paste(_c3, (48, 1678), _c3)
except Exception:
    pass
layout["Online"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_01_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-3/04_icon_OU7_LIGQLUI.png
try:
    _c4 = get_crop(4, 288, 156)
    canvas.paste(_c4, (288, 2804), _c4)
except Exception:
    pass
layout["OU7_LIGQLUI"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_01_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-3/05_icon_Favorite_button.png
try:
    _c5 = get_crop(5, 144, 125)
    canvas.paste(_c5, (1140, 1949), _c5)
except Exception:
    pass
layout["Favorite_button"] = [1140, 1949, 1284, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_01_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-3/06_icon_Home.png
try:
    _c6 = get_crop(6, 288, 156)
    canvas.paste(_c6, (0, 2804), _c6)
except Exception:
    pass
layout["Home"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_01_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-3/07_icon_Favorite_button.png
try:
    _c7 = get_crop(7, 144, 125)
    canvas.paste(_c7, (1140, 2345), _c7)
except Exception:
    pass
layout["Favorite_button"] = [1140, 2345, 1284, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_01_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-3/08_icon_4.44.png
try:
    _c8 = get_crop(8, 106, 101)
    canvas.paste(_c8, (39, 121), _c8)
except Exception:
    pass
layout["4.44"] = [39, 121, 145, 222]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_01_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-3/09_icon_Overflow_menu_button.png
try:
    _c9 = get_crop(9, 144, 125)
    canvas.paste(_c9, (1284, 2345), _c9)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 2345, 1428, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_01_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-3/10_icon_Overflow_menu_button.png
try:
    _c10 = get_crop(10, 144, 139)
    canvas.paste(_c10, (1284, 1539), _c10)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 1539, 1428, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_01_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-3/11_icon_4.44.png
try:
    _c11 = get_crop(11, 54, 61)
    canvas.paste(_c11, (184, 2), _c11)
except Exception:
    pass
layout["4.44"] = [184, 2, 238, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_01_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-3/12_icon_Overflow_menu_button.png
try:
    _c12 = get_crop(12, 144, 139)
    canvas.paste(_c12, (1284, 747), _c12)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 747, 1428, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_01_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-3/13_icon_Overflow_menu_button.png
try:
    _c13 = get_crop(13, 144, 139)
    canvas.paste(_c13, (1284, 1143), _c13)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 1143, 1428, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_01_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-3/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 60, 59)
    canvas.paste(_c14, (312, 3), _c14)
except Exception:
    pass
layout["icon_14"] = [312, 3, 372, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_01_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-3/15_icon_Overflow_menu_button.png
try:
    _c15 = get_crop(15, 144, 125)
    canvas.paste(_c15, (1284, 1949), _c15)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 1949, 1428, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_01_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-3/16_icon_Art_for_Grief_and_Loss.png
try:
    _c16 = get_crop(16, 1344, 396)
    canvas.paste(_c16, (48, 1282), _c16)
except Exception:
    pass
layout["Art_for_Grief_and_Loss"] = [48, 1282, 1392, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_01_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-3/17_icon_TomU_5i0.png
try:
    _c17 = get_crop(17, 586, 117)
    canvas.paste(_c17, (427, 2651), _c17)
except Exception:
    pass
layout["TomU'5i0"] = [427, 2651, 1013, 2768]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_01_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-3/18_icon_icon_18.png
try:
    _c18 = get_crop(18, 50, 59)
    canvas.paste(_c18, (248, 3), _c18)
except Exception:
    pass
layout["icon_18"] = [248, 3, 298, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_01_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-3/19_icon_Favorite_button.png
try:
    _c19 = get_crop(19, 144, 139)
    canvas.paste(_c19, (1140, 747), _c19)
except Exception:
    pass
layout["Favorite_button"] = [1140, 747, 1284, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_01_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-3/20_icon_Senaratinn_Grief_and_Loss.png
try:
    _c20 = get_crop(20, 288, 156)
    canvas.paste(_c20, (864, 2804), _c20)
except Exception:
    pass
layout["Senaratinn_Grief;_and_Los"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_01_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-3/21_icon_icon_21.png
try:
    _c21 = get_crop(21, 48, 53)
    canvas.paste(_c21, (1321, 7), _c21)
except Exception:
    pass
layout["icon_21"] = [1321, 7, 1369, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_01_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-3/22_icon_Working_with_Grief_and_Loss.png
try:
    _c22 = get_crop(22, 1344, 396)
    canvas.paste(_c22, (48, 490), _c22)
except Exception:
    pass
layout["Working_with_Grief_and_Lo"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_01_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-3/23_icon_Favorite_button.png
try:
    _c23 = get_crop(23, 144, 139)
    canvas.paste(_c23, (1140, 1143), _c23)
except Exception:
    pass
layout["Favorite_button"] = [1140, 1143, 1284, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_01_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-3/24_icon_Favorite_button.png
try:
    _c24 = get_crop(24, 144, 139)
    canvas.paste(_c24, (1140, 1539), _c24)
except Exception:
    pass
layout["Favorite_button"] = [1140, 1539, 1284, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_01_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-3/25_icon_4.44.png
try:
    _c25 = get_crop(25, 57, 60)
    canvas.paste(_c25, (115, 3), _c25)
except Exception:
    pass
layout["4.44"] = [115, 3, 172, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_01_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-3/26_icon_icon_26.png
try:
    _c26 = get_crop(26, 67, 60)
    canvas.paste(_c26, (1212, 3), _c26)
except Exception:
    pass
layout["icon_26"] = [1212, 3, 1279, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_01_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-3/27_icon_S_00_AM_EDT.png
try:
    _c27 = get_crop(27, 1344, 396)
    canvas.paste(_c27, (48, 1678), _c27)
except Exception:
    pass
layout["S:00_AM_EDT"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_01_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-3/28_icon_5_O0_AM_EDT.png
try:
    _c28 = get_crop(28, 1344, 396)
    canvas.paste(_c28, (48, 2074), _c28)
except Exception:
    pass
layout["5:O0_AM_EDT"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_01_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-3/29_icon_Online.png
try:
    _c29 = get_crop(29, 112, 53)
    canvas.paste(_c29, (390, 1496), _c29)
except Exception:
    pass
layout["Online"] = [390, 1496, 502, 1549]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_01_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-3/30_icon_Online.png
try:
    _c30 = get_crop(30, 112, 54)
    canvas.paste(_c30, (390, 703), _c30)
except Exception:
    pass
layout["Online"] = [390, 703, 502, 757]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_01_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-3/31_icon_icon_31.png
try:
    _c31 = get_crop(31, 44, 56)
    canvas.paste(_c31, (385, 7), _c31)
except Exception:
    pass
layout["icon_31"] = [385, 7, 429, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_01_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-3/32_icon_Understanding_Grief_and_Loss.png
try:
    _c32 = get_crop(32, 1344, 396)
    canvas.paste(_c32, (48, 886), _c32)
except Exception:
    pass
layout["Understanding_Grief_and_L"] = [48, 886, 1392, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_01_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-3/33_icon_icon_33.png
try:
    _c33 = get_crop(33, 42, 56)
    canvas.paste(_c33, (1272, 5), _c33)
except Exception:
    pass
layout["icon_33"] = [1272, 5, 1314, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_01_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-3/34_icon_4.44.png
try:
    _c34 = get_crop(34, 90, 60)
    canvas.paste(_c34, (16, 4), _c34)
except Exception:
    pass
layout["4.44"] = [16, 4, 106, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_01_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-3/35_icon_suppoloyed_Orilee_herapeeticrarard_Outh_.png
try:
    _c35 = get_crop(35, 1344, 396)
    canvas.paste(_c35, (48, 1282), _c35)
except Exception:
    pass
layout["suppoloyed_Orilee__herape"] = [48, 1282, 1392, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_01_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-3/36_icon_TomU_5i0.png
try:
    _c36 = get_crop(36, 288, 156)
    canvas.paste(_c36, (576, 2804), _c36)
except Exception:
    pass
layout["TomU'5i0"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_01_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-3/37_text_More_events_you_II_love.png
try:
    _c37 = get_crop(37, 1344, 396)
    canvas.paste(_c37, (48, 490), _c37)
except Exception:
    pass
layout["More_events_you'II_love"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_01_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-3/38_text_Thu_May_2_11_00_AM_EDT.png
try:
    _c38 = get_crop(38, 1344, 346)
    canvas.paste(_c38, (48, 2470), _c38)
except Exception:
    pass
layout["Thu,_May_2_+_11:00_AM_EDT"] = [48, 2470, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_01_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-3/39_clickable_More.png
try:
    _c39 = get_crop(39, 288, 156)
    canvas.paste(_c39, (1152, 2804), _c39)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]
