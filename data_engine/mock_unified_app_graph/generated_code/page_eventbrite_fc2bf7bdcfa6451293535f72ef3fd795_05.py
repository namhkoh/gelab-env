# page_id: page_eventbrite_fc2bf7bdcfa6451293535f72ef3fd795_05
# screenshot: 2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-7.png
# step_index: 5/8
# task: Open Eventbrite. Search for events by 'Music' under online events. Choose the second event in the list. Get the event's duration information.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Top-level background
draw.rectangle((0, 0, 1440, 2960), fill="#FBFBFD")

# Status bar area (top ~88px)
status_h = 88
draw.rectangle((0, 0, 1440, status_h), fill="#BDBDBF")

# Subtle top overlay line to mimic Android status bar bottom edge
draw.line((0, status_h - 1, 1440, status_h - 1), fill="#9E9EA0", width=1)

# Header / search area background (below status bar)
header_top = status_h
header_bottom = 264
draw.rectangle((0, header_top, 1440, header_bottom), fill="#F6F7FA")

# Thin divider under header/search area
draw.line((48, header_bottom, 1392, header_bottom), fill="#E6E6EA", width=2)

# Secondary divider between filters and list (approx)
filters_div_y = 520
draw.line((48, filters_div_y, 1392, filters_div_y), fill="#F0F0F3", width=1)

# Card 1 (first event) background + subtle shadow
card1_x0, card1_y0 = 36, 652
card1_x1, card1_y1 = 1404, 1796
# shadow
draw.rectangle((card1_x0 + 8, card1_y0 + 8, card1_x1 + 8, card1_y1 + 8), fill="#ECECF1")
# card background (rounded)
try:
    draw.rounded_rectangle((card1_x0, card1_y0, card1_x1, card1_y1), radius=20, fill="#FFFFFF", outline="#E8E8EB", width=1)
except Exception:
    # fallback if rounded_rectangle not available
    draw.rectangle((card1_x0, card1_y0, card1_x1, card1_y1), fill="#FFFFFF", outline="#E8E8EB")

# Separator line under card1 area (to separate the title block from next card area)
draw.line((48, card1_y1 + 12, 1392, card1_y1 + 12), fill="#ECECF1", width=1)

# Card 2 (second event) background + subtle shadow
card2_x0, card2_y0 = 36, card1_y1 + 40
card2_x1, card2_y1 = 1404, 2856
# clamp to canvas bottom
if card2_y1 > 2950:
    card2_y1 = 2950
# shadow
draw.rectangle((card2_x0 + 8, card2_y0 + 8, card2_x1 + 8, card2_y1 + 8), fill="#ECECF1")
# card background (rounded)
try:
    draw.rounded_rectangle((card2_x0, card2_y0, card2_x1, card2_y1), radius=20, fill="#FFFFFF", outline="#E8E8EB", width=1)
except Exception:
    draw.rectangle((card2_x0, card2_y0, card2_x1, card2_y1), fill="#FFFFFF", outline="#E8E8EB")

# Dark banner area hint behind first card's image region (so pasted image looks on darker strip)
banner_h = 180
banner_y = card1_y0
draw.rectangle((card1_x0, banner_y, card1_x1, banner_y + banner_h), fill="#EEF6FB")

# Small divider lines between list items (fine separators)
sep_y_positions = [card1_y1 + 6, card2_y1 + 6]
for y in sep_y_positions:
    if y < 2960:
        draw.line((48, y, 1392, y), fill="#F3F3F6", width=1)

# Bottom navigation background area and top divider
nav_top = 2836
if nav_top < 2600:
    nav_top = 2836
draw.line((0, nav_top, 1440, nav_top), fill="#E6E6EA", width=1)
draw.rectangle((0, nav_top, 1440, 2960), fill="#FFFFFF")

# small subtle left margin rule near list for visual structure
draw.line((48, header_bottom + 4, 48, nav_top - 24), fill="#F4F4F6", width=1)
draw.line((1392, header_bottom + 4, 1392, nav_top - 24), fill="#F4F4F6", width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_05_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-7/00_icon_Anytime.png
try:
    _c0 = get_crop(0, 400, 135)
    canvas.paste(_c0, (438, 390), _c0)
except Exception:
    pass
layout["Anytime"] = [438, 390, 838, 525]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_05_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-7/01_icon_Music.png
try:
    _c1 = get_crop(1, 187, 135)
    canvas.paste(_c1, (850, 390), _c1)
except Exception:
    pass
layout["Music"] = [850, 390, 1037, 525]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_05_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-7/02_icon_1_Filter.png
try:
    _c2 = get_crop(2, 372, 135)
    canvas.paste(_c2, (54, 390), _c2)
except Exception:
    pass
layout["1_Filter"] = [54, 390, 426, 525]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_05_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-7/03_icon_KhB.png
try:
    _c3 = get_crop(3, 144, 144)
    canvas.paste(_c3, (1092, 1192), _c3)
except Exception:
    pass
layout["KhB"] = [1092, 1192, 1236, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_05_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-7/04_icon_KhB.png
try:
    _c4 = get_crop(4, 144, 144)
    canvas.paste(_c4, (1236, 1192), _c4)
except Exception:
    pass
layout["KhB"] = [1236, 1192, 1380, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_05_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-7/05_icon_8.04.png
try:
    _c5 = get_crop(5, 123, 113)
    canvas.paste(_c5, (55, 115), _c5)
except Exception:
    pass
layout["8.04"] = [55, 115, 178, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_05_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-7/06_icon_8.04.png
try:
    _c6 = get_crop(6, 61, 64)
    canvas.paste(_c6, (180, 0), _c6)
except Exception:
    pass
layout["8.04"] = [180, 0, 241, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_05_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-7/07_icon_Search_forae.png
try:
    _c7 = get_crop(7, 68, 63)
    canvas.paste(_c7, (307, 0), _c7)
except Exception:
    pass
layout["Search_forae"] = [307, 0, 375, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_05_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-7/08_icon_PARTY.png
try:
    _c8 = get_crop(8, 1344, 1091)
    canvas.paste(_c8, (48, 676), _c8)
except Exception:
    pass
layout["PARTY"] = [48, 676, 1392, 1767]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_05_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-7/09_icon_8.04.png
try:
    _c9 = get_crop(9, 61, 65)
    canvas.paste(_c9, (114, 0), _c9)
except Exception:
    pass
layout["8.04"] = [114, 0, 175, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_05_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-7/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 54, 64)
    canvas.paste(_c10, (246, 0), _c10)
except Exception:
    pass
layout["icon_10"] = [246, 0, 300, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_05_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-7/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 80, 60)
    canvas.paste(_c11, (1207, 0), _c11)
except Exception:
    pass
layout["icon_11"] = [1207, 0, 1287, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_05_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-7/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 64, 59)
    canvas.paste(_c12, (1317, 0), _c12)
except Exception:
    pass
layout["icon_12"] = [1317, 0, 1381, 59]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_05_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-7/13_icon_Overflow_menu_button.png
try:
    _c13 = get_crop(13, 144, 144)
    canvas.paste(_c13, (1236, 2331), _c13)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 2331, 1380, 2475]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_05_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-7/14_icon_Search_forae.png
try:
    _c14 = get_crop(14, 1344, 191)
    canvas.paste(_c14, (48, 72), _c14)
except Exception:
    pass
layout["Search_forae"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_05_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-7/15_icon_Favorite_button.png
try:
    _c15 = get_crop(15, 144, 144)
    canvas.paste(_c15, (1092, 2331), _c15)
except Exception:
    pass
layout["Favorite_button"] = [1092, 2331, 1236, 2475]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_05_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-7/16_icon_6988_Beach_Blvd_b_204.png
try:
    _c16 = get_crop(16, 288, 156)
    canvas.paste(_c16, (288, 2804), _c16)
except Exception:
    pass
layout["6988_Beach_Blvd_b_204"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_05_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-7/17_icon_Search_forae.png
try:
    _c17 = get_crop(17, 51, 61)
    canvas.paste(_c17, (383, 2), _c17)
except Exception:
    pass
layout["Search_forae"] = [383, 2, 434, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_05_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-7/18_icon_The_Melrose_House.png
try:
    _c18 = get_crop(18, 44, 60)
    canvas.paste(_c18, (284, 1662), _c18)
except Exception:
    pass
layout["The_Melrose_House"] = [284, 1662, 328, 1722]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_05_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-7/19_icon_Los_Angeles.png
try:
    _c19 = get_crop(19, 492, 144)
    canvas.paste(_c19, (0, 259), _c19)
except Exception:
    pass
layout["Los_Angeles"] = [0, 259, 492, 403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_05_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-7/20_icon_10_00_AM_PDT.png
try:
    _c20 = get_crop(20, 288, 156)
    canvas.paste(_c20, (576, 2804), _c20)
except Exception:
    pass
layout["10:00_AM_PDT"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_05_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-7/21_icon_D_FESTA_LOS_ANGELES_THE_KPOP.png
try:
    _c21 = get_crop(21, 288, 156)
    canvas.paste(_c21, (864, 2804), _c21)
except Exception:
    pass
layout["D'FESTA_LOS_ANGELES_(_THE"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_05_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-7/22_icon_More.png
try:
    _c22 = get_crop(22, 288, 156)
    canvas.paste(_c22, (1152, 2804), _c22)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_05_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-7/23_icon_Sat_Jun_22.png
try:
    _c23 = get_crop(23, 288, 156)
    canvas.paste(_c23, (0, 2804), _c23)
except Exception:
    pass
layout["Sat,_Jun_22"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_05_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-7/24_icon_icon_24.png
try:
    _c24 = get_crop(24, 40, 61)
    canvas.paste(_c24, (1274, 0), _c24)
except Exception:
    pass
layout["icon_24"] = [1274, 0, 1314, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_05_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-7/25_icon_icon_25.png
try:
    _c25 = get_crop(25, 96, 81)
    canvas.paste(_c25, (1004, 2418), _c25)
except Exception:
    pass
layout["icon_25"] = [1004, 2418, 1100, 2499]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_05_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-7/26_icon_Tickets.png
try:
    _c26 = get_crop(26, 288, 156)
    canvas.paste(_c26, (864, 2804), _c26)
except Exception:
    pass
layout["Tickets"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_05_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-7/27_text_8.04.png
try:
    _c27 = get_crop(27, 94, 45)
    canvas.paste(_c27, (20, 15), _c27)
except Exception:
    pass
layout["8.04"] = [20, 15, 114, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_05_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-7/28_text_1_474events.png
try:
    _c28 = get_crop(28, 372, 135)
    canvas.paste(_c28, (54, 390), _c28)
except Exception:
    pass
layout["1,474events"] = [54, 390, 426, 525]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_05_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-7/29_text_Luv_Language_Day_Party_A_Modern_RgB_Vibe.png
try:
    _c29 = get_crop(29, 1344, 1091)
    canvas.paste(_c29, (48, 676), _c29)
except Exception:
    pass
layout["Luv_Language_Day_Party:_A"] = [48, 676, 1392, 1767]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_05_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-7/30_text_Sun.png
try:
    _c30 = get_crop(30, 101, 54)
    canvas.paste(_c30, (90, 1534), _c30)
except Exception:
    pass
layout["Sun,"] = [90, 1534, 191, 1588]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_05_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-7/31_text_28.png
try:
    _c31 = get_crop(31, 64, 45)
    canvas.paste(_c31, (262, 1537), _c31)
except Exception:
    pass
layout["28"] = [262, 1537, 326, 1582]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_05_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-7/32_text_4.00_PM_PDT.png
try:
    _c32 = get_crop(32, 253, 45)
    canvas.paste(_c32, (346, 1537), _c32)
except Exception:
    pass
layout["4.00_PM_PDT"] = [346, 1537, 599, 1582]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_05_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-7/33_text_The_Melrose_House.png
try:
    _c33 = get_crop(33, 359, 43)
    canvas.paste(_c33, (94, 1606), _c33)
except Exception:
    pass
layout["The_Melrose_House"] = [94, 1606, 453, 1649]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_05_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-7/34_text_DFESTA.png
try:
    _c34 = get_crop(34, 433, 102)
    canvas.paste(_c34, (72, 1820), _c34)
except Exception:
    pass
layout["DFESTA'"] = [72, 1820, 505, 1922]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_05_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-7/35_text_0_6_._2_1_._2_4.png
try:
    _c35 = get_crop(35, 341, 72)
    canvas.paste(_c35, (592, 1841), _c35)
except Exception:
    pass
layout["0_6_._2_1_._2_4"] = [592, 1841, 933, 1913]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_05_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-7/36_text_0_8_._1_8_._2_4.png
try:
    _c36 = get_crop(36, 340, 72)
    canvas.paste(_c36, (1023, 1841), _c36)
except Exception:
    pass
layout["0_8_._1_8_._2_4"] = [1023, 1841, 1363, 1913]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_05_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-7/37_text_CA.png
try:
    _c37 = get_crop(37, 64, 27)
    canvas.paste(_c37, (88, 2449), _c37)
except Exception:
    pass
layout["CA_"] = [88, 2449, 152, 2476]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_05_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-7/38_text_t.png
try:
    _c38 = get_crop(38, 71, 27)
    canvas.paste(_c38, (252, 2449), _c38)
except Exception:
    pass
layout["t+"] = [252, 2449, 323, 2476]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_05_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-7/39_text_D_FESTA_LOS_ANGELES_THE_KPOP.png
try:
    _c39 = get_crop(39, 1344, 1001)
    canvas.paste(_c39, (48, 1815), _c39)
except Exception:
    pass
layout["D'FESTA_LOS_ANGELES_(_THE"] = [48, 1815, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_05_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-7/40_text_EXPERIENCE.png
try:
    _c40 = get_crop(40, 380, 72)
    canvas.paste(_c40, (93, 2588), _c40)
except Exception:
    pass
layout["EXPERIENCE)"] = [93, 2588, 473, 2660]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_05_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-7/41_text_Sat_Jun_22.png
try:
    _c41 = get_crop(41, 219, 54)
    canvas.paste(_c41, (90, 2673), _c41)
except Exception:
    pass
layout["Sat,_Jun_22"] = [90, 2673, 309, 2727]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_05_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-7/42_text_10_00_AM_PDT.png
try:
    _c42 = get_crop(42, 288, 156)
    canvas.paste(_c42, (288, 2804), _c42)
except Exception:
    pass
layout["10:00_AM_PDT"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_05_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-7/43_text_6988_Beach_Blvd_b_204.png
try:
    _c43 = get_crop(43, 288, 156)
    canvas.paste(_c43, (288, 2804), _c43)
except Exception:
    pass
layout["6988_Beach_Blvd_b_204"] = [288, 2804, 576, 2960]
