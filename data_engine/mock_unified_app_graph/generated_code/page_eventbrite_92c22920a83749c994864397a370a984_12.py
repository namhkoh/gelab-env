# page_id: page_eventbrite_92c22920a83749c994864397a370a984_12
# screenshot: 2024_4_24_16_59_92c22920a83749c994864397a370a984-14.png
# step_index: 12/13
# task: Open Eventbrite. Set the city to "Chicago". Select the "Sports" category and view the recommended events. See the date of the first non-promoted event.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Top status bar
draw.rectangle([(0, 0), (1440, 80)], fill="#BDBDBD")

# Header / search area background
draw.rectangle([(0, 80), (1440, 240)], fill="#FFFFFF")
# subtle divider under header
draw.line([(48, 240), (1392, 240)], fill="#E6E6E6", width=2)

# Light page background
draw.rectangle([(0, 240), (1440, 2960)], fill="#FAFAFB")

# Large card background for the first event block (rounded)
card1_coords = (48, 420, 1392, 1180)
draw.rounded_rectangle(card1_coords, radius=28, fill="#FFFFFF", outline="#E7E7EA", width=1)

# Subtle dark banner area behind the promoted image region (visual content area background only)
# This represents a dark image band / content area; actual image will be pasted on top.
draw.rounded_rectangle((48, 880, 1392, 1280), radius=20, fill="#2C0A0A", outline=None)

# Separator between first and second event card
draw.line([(48, 1188), (1392, 1188)], fill="#F0F0F2", width=1)

# Large card background for the second event block (rounded)
card2_coords = (48, 1220, 1392, 2120)
draw.rounded_rectangle(card2_coords, radius=28, fill="#FFFFFF", outline="#E7E7EA", width=1)

# Image content background area for the second event (dark/photographic placeholder)
draw.rounded_rectangle((48, 1520, 1392, 2020), radius=20, fill="#5A3B2A", outline=None)

# Thin divider lines for section separation
draw.line([(48, 2128), (1392, 2128)], fill="#EFEFF1", width=1)
draw.line([(48, 2804), (1392, 2804)], fill="#E6E6E9", width=2)

# Bottom navigation bar background
draw.rectangle([(0, 2804), (1440, 2960)], fill="#FFFFFF")
# slight top shadow line for nav
draw.line([(0, 2804), (1440, 2804)], fill="#E6E6E6", width=2)

# Small subtle left and right page margins as faint vertical lines
draw.line([(48, 240), (48, 2804)], fill="#F4F4F6", width=1)
draw.line([(1392, 240), (1392, 2804)], fill="#F4F4F6", width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_12_2024_4_24_16_59_92c22920a83749c994864397a370a984-14/00_icon_Sports_Fitness.png
try:
    _c0 = get_crop(0, 390, 113)
    canvas.paste(_c0, (843, 406), _c0)
except Exception:
    pass
layout["Sports_&_Fitness"] = [843, 406, 1233, 519]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_12_2024_4_24_16_59_92c22920a83749c994864397a370a984-14/01_icon_Anytime.png
try:
    _c1 = get_crop(1, 1344, 364)
    canvas.paste(_c1, (48, 525), _c1)
except Exception:
    pass
layout["Anytime"] = [48, 525, 1392, 889]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_12_2024_4_24_16_59_92c22920a83749c994864397a370a984-14/02_icon_1_Filter.png
try:
    _c2 = get_crop(2, 417, 144)
    canvas.paste(_c2, (0, 259), _c2)
except Exception:
    pass
layout["1_Filter"] = [0, 259, 417, 403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_12_2024_4_24_16_59_92c22920a83749c994864397a370a984-14/03_icon_CHICAGO.png
try:
    _c3 = get_crop(3, 144, 144)
    canvas.paste(_c3, (1092, 1453), _c3)
except Exception:
    pass
layout["CHICAGO"] = [1092, 1453, 1236, 1597]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_12_2024_4_24_16_59_92c22920a83749c994864397a370a984-14/04_icon_MayW.png
try:
    _c4 = get_crop(4, 144, 144)
    canvas.paste(_c4, (1236, 1453), _c4)
except Exception:
    pass
layout["MayW"] = [1236, 1453, 1380, 1597]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_12_2024_4_24_16_59_92c22920a83749c994864397a370a984-14/05_icon_Favorite_button.png
try:
    _c5 = get_crop(5, 144, 144)
    canvas.paste(_c5, (1092, 2549), _c5)
except Exception:
    pass
layout["Favorite_button"] = [1092, 2549, 1236, 2693]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_12_2024_4_24_16_59_92c22920a83749c994864397a370a984-14/06_icon_Overflow_menu_button.png
try:
    _c6 = get_crop(6, 144, 144)
    canvas.paste(_c6, (1236, 2549), _c6)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 2549, 1380, 2693]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_12_2024_4_24_16_59_92c22920a83749c994864397a370a984-14/07_icon_Few_tickets_left.png
try:
    _c7 = get_crop(7, 288, 156)
    canvas.paste(_c7, (0, 2804), _c7)
except Exception:
    pass
layout["Few_tickets_left"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_12_2024_4_24_16_59_92c22920a83749c994864397a370a984-14/08_icon_5.01.png
try:
    _c8 = get_crop(8, 124, 116)
    canvas.paste(_c8, (55, 113), _c8)
except Exception:
    pass
layout["5.01"] = [55, 113, 179, 229]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_12_2024_4_24_16_59_92c22920a83749c994864397a370a984-14/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 52, 66)
    canvas.paste(_c9, (1152, 0), _c9)
except Exception:
    pass
layout["icon_9"] = [1152, 0, 1204, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_12_2024_4_24_16_59_92c22920a83749c994864397a370a984-14/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 62, 62)
    canvas.paste(_c10, (1213, 0), _c10)
except Exception:
    pass
layout["icon_10"] = [1213, 0, 1275, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_12_2024_4_24_16_59_92c22920a83749c994864397a370a984-14/11_icon_5.01.png
try:
    _c11 = get_crop(11, 59, 63)
    canvas.paste(_c11, (181, 0), _c11)
except Exception:
    pass
layout["5.01"] = [181, 0, 240, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_12_2024_4_24_16_59_92c22920a83749c994864397a370a984-14/12_icon_Search_forae.png
try:
    _c12 = get_crop(12, 68, 62)
    canvas.paste(_c12, (307, 1), _c12)
except Exception:
    pass
layout["Search_forae"] = [307, 1, 375, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_12_2024_4_24_16_59_92c22920a83749c994864397a370a984-14/13_icon_VcHP.png
try:
    _c13 = get_crop(13, 1344, 1048)
    canvas.paste(_c13, (48, 937), _c13)
except Exception:
    pass
layout["~VcHP"] = [48, 937, 1392, 1985]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_12_2024_4_24_16_59_92c22920a83749c994864397a370a984-14/14_icon_Search_forae.png
try:
    _c14 = get_crop(14, 1344, 191)
    canvas.paste(_c14, (48, 72), _c14)
except Exception:
    pass
layout["Search_forae"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_12_2024_4_24_16_59_92c22920a83749c994864397a370a984-14/15_icon_5.01.png
try:
    _c15 = get_crop(15, 60, 64)
    canvas.paste(_c15, (114, 0), _c15)
except Exception:
    pass
layout["5.01"] = [114, 0, 174, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_12_2024_4_24_16_59_92c22920a83749c994864397a370a984-14/16_icon_icon_16.png
try:
    _c16 = get_crop(16, 53, 62)
    canvas.paste(_c16, (246, 1), _c16)
except Exception:
    pass
layout["icon_16"] = [246, 1, 299, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_12_2024_4_24_16_59_92c22920a83749c994864397a370a984-14/17_icon_icon_17.png
try:
    _c17 = get_crop(17, 57, 59)
    canvas.paste(_c17, (1318, 0), _c17)
except Exception:
    pass
layout["icon_17"] = [1318, 0, 1375, 59]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_12_2024_4_24_16_59_92c22920a83749c994864397a370a984-14/18_icon_Promoted.png
try:
    _c18 = get_crop(18, 268, 67)
    canvas.paste(_c18, (60, 780), _c18)
except Exception:
    pass
layout["Promoted"] = [60, 780, 328, 847]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_12_2024_4_24_16_59_92c22920a83749c994864397a370a984-14/19_icon_Few_tickets_left.png
try:
    _c19 = get_crop(19, 370, 85)
    canvas.paste(_c19, (86, 1629), _c19)
except Exception:
    pass
layout["Few_tickets_left"] = [86, 1629, 456, 1714]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_12_2024_4_24_16_59_92c22920a83749c994864397a370a984-14/20_icon_Few_tickets_left.png
try:
    _c20 = get_crop(20, 288, 156)
    canvas.paste(_c20, (576, 2804), _c20)
except Exception:
    pass
layout["Few_tickets_left"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_12_2024_4_24_16_59_92c22920a83749c994864397a370a984-14/21_icon_Few_tickets_left.png
try:
    _c21 = get_crop(21, 288, 156)
    canvas.paste(_c21, (288, 2804), _c21)
except Exception:
    pass
layout["Few_tickets_left"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_12_2024_4_24_16_59_92c22920a83749c994864397a370a984-14/22_icon_Chicago.png
try:
    _c22 = get_crop(22, 417, 144)
    canvas.paste(_c22, (0, 259), _c22)
except Exception:
    pass
layout["Chicago"] = [0, 259, 417, 403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_12_2024_4_24_16_59_92c22920a83749c994864397a370a984-14/23_icon_Few_tickets_left.png
try:
    _c23 = get_crop(23, 288, 156)
    canvas.paste(_c23, (0, 2804), _c23)
except Exception:
    pass
layout["Few_tickets_left"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_12_2024_4_24_16_59_92c22920a83749c994864397a370a984-14/24_icon_More.png
try:
    _c24 = get_crop(24, 288, 156)
    canvas.paste(_c24, (1152, 2804), _c24)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_12_2024_4_24_16_59_92c22920a83749c994864397a370a984-14/25_icon_Tickets.png
try:
    _c25 = get_crop(25, 288, 156)
    canvas.paste(_c25, (864, 2804), _c25)
except Exception:
    pass
layout["Tickets"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_12_2024_4_24_16_59_92c22920a83749c994864397a370a984-14/26_icon_icon_26.png
try:
    _c26 = get_crop(26, 40, 61)
    canvas.paste(_c26, (1274, 0), _c26)
except Exception:
    pass
layout["icon_26"] = [1274, 0, 1314, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_12_2024_4_24_16_59_92c22920a83749c994864397a370a984-14/27_icon_Search_forae.png
try:
    _c27 = get_crop(27, 49, 61)
    canvas.paste(_c27, (384, 2), _c27)
except Exception:
    pass
layout["Search_forae"] = [384, 2, 433, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_12_2024_4_24_16_59_92c22920a83749c994864397a370a984-14/28_icon_MLW_AZTECA_LUCHA_Triller_TV_PPV.png
try:
    _c28 = get_crop(28, 1344, 1048)
    canvas.paste(_c28, (48, 937), _c28)
except Exception:
    pass
layout["MLW:_AZTECA_LUCHA_(Trille"] = [48, 937, 1392, 1985]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_12_2024_4_24_16_59_92c22920a83749c994864397a370a984-14/29_text_5.01.png
try:
    _c29 = get_crop(29, 87, 43)
    canvas.paste(_c29, (22, 17), _c29)
except Exception:
    pass
layout["5.01"] = [22, 17, 109, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_12_2024_4_24_16_59_92c22920a83749c994864397a370a984-14/30_text_Basics_of_Roller_Skating_balance_power.png
try:
    _c30 = get_crop(30, 1344, 364)
    canvas.paste(_c30, (48, 525), _c30)
except Exception:
    pass
layout["Basics_of_Roller_Skating_"] = [48, 525, 1392, 889]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_12_2024_4_24_16_59_92c22920a83749c994864397a370a984-14/31_text_Mon.png
try:
    _c31 = get_crop(31, 109, 52)
    canvas.paste(_c31, (93, 659), _c31)
except Exception:
    pass
layout["Mon,"] = [93, 659, 202, 711]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_12_2024_4_24_16_59_92c22920a83749c994864397a370a984-14/32_text_6_._7_00_PM_EDT.png
try:
    _c32 = get_crop(32, 307, 45)
    canvas.paste(_c32, (288, 658), _c32)
except Exception:
    pass
layout["6_._7:00_PM_EDT"] = [288, 658, 595, 703]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_12_2024_4_24_16_59_92c22920a83749c994864397a370a984-14/33_text_Online.png
try:
    _c33 = get_crop(33, 126, 43)
    canvas.paste(_c33, (94, 727), _c33)
except Exception:
    pass
layout["Online"] = [94, 727, 220, 770]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_12_2024_4_24_16_59_92c22920a83749c994864397a370a984-14/34_text_ZTEC.png
try:
    _c34 = get_crop(34, 210, 109)
    canvas.paste(_c34, (719, 1012), _c34)
except Exception:
    pass
layout["(ZTEC"] = [719, 1012, 929, 1121]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_12_2024_4_24_16_59_92c22920a83749c994864397a370a984-14/35_clickable_Event_s_image.png
try:
    _c35 = get_crop(35, 1344, 783)
    canvas.paste(_c35, (48, 2033), _c35)
except Exception:
    pass
layout["Event's_image"] = [48, 2033, 1392, 2816]
