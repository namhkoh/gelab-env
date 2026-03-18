# page_id: page_eventbrite_92c22920a83749c994864397a370a984_06
# screenshot: 2024_4_24_16_59_92c22920a83749c994864397a370a984-8.png
# step_index: 6/13
# task: Open Eventbrite. Set the city to "Chicago". Select the "Sports" category and view the recommended events. See the date of the first non-promoted event.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw overall background (canvas already white, but fill to ensure consistency)
draw.rectangle((0, 0, 1440, 2960), fill="#ffffff")

# Top status bar area (approx ~80px tall)
status_h = 80
draw.rectangle((0, 0, 1440, status_h), fill="#d0d0d0")
# subtle divider under status bar
draw.line((0, status_h, 1440, status_h), fill="#c0c0c0", width=1)

# Header / toolbar region (beneath status bar)
header_top = status_h
header_bottom = 220
draw.rectangle((0, header_top, 1440, header_bottom), fill="#ffffff")
# faint divider line under header
draw.line((24, header_bottom, 1440-24, header_bottom), fill="#efecec", width=2)

# Left vertical accent padding (matches app padding column)
draw.rectangle((0, header_bottom + 20, 48, 2800), fill="#fafafa")

# Draw subtle section separators across content area
separator_color = "#f0eef2"
separator_positions = [460, 842, 1220, 1608, 2004, 2400, 2788]
for y in separator_positions:
    draw.line((24, y, 1440-24, y), fill=separator_color, width=2)

# Draw rounded card backgrounds for each event list item (structural only)
# These are background containers behind each group; actual images/text/icons will be pasted on top.
card_fill = "#ffffff"
card_shadow = "#f3f3f5"
card_outline = "#eeeaf0"
card_x1 = 48 - 8  # slight inset to show padding
card_x2 = 1440 - 48 + 8
card_radius = 18

event_tops = [490, 886, 1282, 1678, 2074, 2470]
for top in event_tops:
    top_rect = top - 8
    bottom_rect = top + 396 + 8
    # shadow (slightly offset and faint)
    draw.rounded_rectangle((card_x1+6, top_rect+6, card_x2+6, bottom_rect+6), radius=card_radius, fill=card_shadow, outline=None)
    # main card background
    draw.rounded_rectangle((card_x1, top_rect, card_x2, bottom_rect), radius=card_radius, fill=card_fill, outline=card_outline)

# Content banner / large dark image area near mid-page (structural background)
# (Use a subtle muted block where featured image areas appear; images will be pasted on top)
banner_y = 2470
draw.rectangle((24, banner_y - 40, 1440-24, banner_y + 260), fill="#fbfbfc")

# Floating bottom navigation background and top divider
nav_top = 2740
draw.rectangle((0, nav_top, 1440, 2960), fill="#ffffff")
draw.line((0, nav_top, 1440, nav_top), fill="#e6e3e8", width=2)
# subtle nav shadow (above the nav bar)
draw.rectangle((0, nav_top-6, 1440, nav_top), fill="#fbfbfb")

# Small decorative left gutter dots to indicate list flow (purely structural)
for i, top in enumerate(event_tops):
    cx = 32
    cy = top + 40
    r = 6
    draw.ellipse((cx-r, cy-r, cx+r, cy+r), fill="#e7e1ea")

# Final faint bottom edge line
draw.line((24, 2958, 1440-24, 2958), fill="#f1eff2", width=2)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_06_2024_4_24_16_59_92c22920a83749c994864397a370a984-8/00_icon_Chicago.png
try:
    _c0 = get_crop(0, 388, 117)
    canvas.paste(_c0, (526, 2651), _c0)
except Exception:
    pass
layout["Chicago"] = [526, 2651, 914, 2768]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_06_2024_4_24_16_59_92c22920a83749c994864397a370a984-8/01_icon_Cy1o6.png
try:
    _c1 = get_crop(1, 1344, 396)
    canvas.paste(_c1, (48, 490), _c1)
except Exception:
    pass
layout["Cy1o6"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_06_2024_4_24_16_59_92c22920a83749c994864397a370a984-8/02_icon_Search_events.png
try:
    _c2 = get_crop(2, 1179, 144)
    canvas.paste(_c2, (195, 93), _c2)
except Exception:
    pass
layout["Search_events"] = [195, 93, 1374, 237]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_06_2024_4_24_16_59_92c22920a83749c994864397a370a984-8/03_icon_Okstore.png
try:
    _c3 = get_crop(3, 1344, 396)
    canvas.paste(_c3, (48, 1678), _c3)
except Exception:
    pass
layout["Okstore"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_06_2024_4_24_16_59_92c22920a83749c994864397a370a984-8/04_icon_at_Goblin_Market.png
try:
    _c4 = get_crop(4, 144, 139)
    canvas.paste(_c4, (1140, 1935), _c4)
except Exception:
    pass
layout["at_Goblin_Market"] = [1140, 1935, 1284, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_06_2024_4_24_16_59_92c22920a83749c994864397a370a984-8/05_icon_Favorite_button.png
try:
    _c5 = get_crop(5, 144, 109)
    canvas.paste(_c5, (1140, 2361), _c5)
except Exception:
    pass
layout["Favorite_button"] = [1140, 2361, 1284, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_06_2024_4_24_16_59_92c22920a83749c994864397a370a984-8/06_icon_I_00_PM_CDT.png
try:
    _c6 = get_crop(6, 1344, 396)
    canvas.paste(_c6, (48, 1282), _c6)
except Exception:
    pass
layout["I:00_PM_CDT"] = [48, 1282, 1392, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_06_2024_4_24_16_59_92c22920a83749c994864397a370a984-8/07_icon_Favorite_button.png
try:
    _c7 = get_crop(7, 144, 139)
    canvas.paste(_c7, (1140, 1539), _c7)
except Exception:
    pass
layout["Favorite_button"] = [1140, 1539, 1284, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_06_2024_4_24_16_59_92c22920a83749c994864397a370a984-8/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 49, 65)
    canvas.paste(_c8, (1153, 2), _c8)
except Exception:
    pass
layout["icon_8"] = [1153, 2, 1202, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_06_2024_4_24_16_59_92c22920a83749c994864397a370a984-8/09_icon_Overflow_menu_button.png
try:
    _c9 = get_crop(9, 144, 109)
    canvas.paste(_c9, (1284, 2361), _c9)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 2361, 1428, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_06_2024_4_24_16_59_92c22920a83749c994864397a370a984-8/10_icon_Overflow_menu_button.png
try:
    _c10 = get_crop(10, 144, 139)
    canvas.paste(_c10, (1284, 1935), _c10)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 1935, 1428, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_06_2024_4_24_16_59_92c22920a83749c994864397a370a984-8/11_icon_Favorite_button.png
try:
    _c11 = get_crop(11, 144, 125)
    canvas.paste(_c11, (1140, 1157), _c11)
except Exception:
    pass
layout["Favorite_button"] = [1140, 1157, 1284, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_06_2024_4_24_16_59_92c22920a83749c994864397a370a984-8/12_icon_Hilton_Ch.png
try:
    _c12 = get_crop(12, 288, 156)
    canvas.paste(_c12, (288, 2804), _c12)
except Exception:
    pass
layout["Hilton_Ch"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_06_2024_4_24_16_59_92c22920a83749c994864397a370a984-8/13_icon_Overflow_menu_button.png
try:
    _c13 = get_crop(13, 144, 125)
    canvas.paste(_c13, (1284, 1157), _c13)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 1157, 1428, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_06_2024_4_24_16_59_92c22920a83749c994864397a370a984-8/14_icon_60615.png
try:
    _c14 = get_crop(14, 1344, 396)
    canvas.paste(_c14, (48, 2074), _c14)
except Exception:
    pass
layout["60615"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_06_2024_4_24_16_59_92c22920a83749c994864397a370a984-8/15_icon_Favorite_button.png
try:
    _c15 = get_crop(15, 144, 139)
    canvas.paste(_c15, (1140, 747), _c15)
except Exception:
    pass
layout["Favorite_button"] = [1140, 747, 1284, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_06_2024_4_24_16_59_92c22920a83749c994864397a370a984-8/16_icon_Overflow_menu_button.png
try:
    _c16 = get_crop(16, 144, 139)
    canvas.paste(_c16, (1284, 747), _c16)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 747, 1428, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_06_2024_4_24_16_59_92c22920a83749c994864397a370a984-8/17_icon_5.00.png
try:
    _c17 = get_crop(17, 115, 106)
    canvas.paste(_c17, (34, 118), _c17)
except Exception:
    pass
layout["5.00"] = [34, 118, 149, 224]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_06_2024_4_24_16_59_92c22920a83749c994864397a370a984-8/18_icon_Overflow_menu_button.png
try:
    _c18 = get_crop(18, 144, 139)
    canvas.paste(_c18, (1284, 1539), _c18)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 1539, 1428, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_06_2024_4_24_16_59_92c22920a83749c994864397a370a984-8/19_icon_icon_19.png
try:
    _c19 = get_crop(19, 61, 59)
    canvas.paste(_c19, (311, 3), _c19)
except Exception:
    pass
layout["icon_19"] = [311, 3, 372, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_06_2024_4_24_16_59_92c22920a83749c994864397a370a984-8/20_icon_icon_20.png
try:
    _c20 = get_crop(20, 98, 61)
    canvas.paste(_c20, (1215, 2), _c20)
except Exception:
    pass
layout["icon_20"] = [1215, 2, 1313, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_06_2024_4_24_16_59_92c22920a83749c994864397a370a984-8/21_icon_5.00.png
try:
    _c21 = get_crop(21, 57, 60)
    canvas.paste(_c21, (182, 2), _c21)
except Exception:
    pass
layout["5.00"] = [182, 2, 239, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_06_2024_4_24_16_59_92c22920a83749c994864397a370a984-8/22_icon_uaO_Iaimg.png
try:
    _c22 = get_crop(22, 1344, 396)
    canvas.paste(_c22, (48, 886), _c22)
except Exception:
    pass
layout["#uaO_{Iaimg"] = [48, 886, 1392, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_06_2024_4_24_16_59_92c22920a83749c994864397a370a984-8/23_icon_icon_23.png
try:
    _c23 = get_crop(23, 51, 60)
    canvas.paste(_c23, (247, 2), _c23)
except Exception:
    pass
layout["icon_23"] = [247, 2, 298, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_06_2024_4_24_16_59_92c22920a83749c994864397a370a984-8/24_icon_Day.png
try:
    _c24 = get_crop(24, 1344, 396)
    canvas.paste(_c24, (48, 1678), _c24)
except Exception:
    pass
layout["Day"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_06_2024_4_24_16_59_92c22920a83749c994864397a370a984-8/25_icon_icon_25.png
try:
    _c25 = get_crop(25, 48, 53)
    canvas.paste(_c25, (1321, 7), _c25)
except Exception:
    pass
layout["icon_25"] = [1321, 7, 1369, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_06_2024_4_24_16_59_92c22920a83749c994864397a370a984-8/26_icon_7940_Wolcott_Ave_apt_2_Chicago_IL_USA.png
try:
    _c26 = get_crop(26, 1344, 396)
    canvas.paste(_c26, (48, 886), _c26)
except Exception:
    pass
layout["7940_$_Wolcott_Ave_apt_2,"] = [48, 886, 1392, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_06_2024_4_24_16_59_92c22920a83749c994864397a370a984-8/27_icon_5.00.png
try:
    _c27 = get_crop(27, 57, 61)
    canvas.paste(_c27, (116, 2), _c27)
except Exception:
    pass
layout["5.00"] = [116, 2, 173, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_06_2024_4_24_16_59_92c22920a83749c994864397a370a984-8/28_icon_ON.png
try:
    _c28 = get_crop(28, 1344, 396)
    canvas.paste(_c28, (48, 1282), _c28)
except Exception:
    pass
layout["ON"] = [48, 1282, 1392, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_06_2024_4_24_16_59_92c22920a83749c994864397a370a984-8/29_icon_INE_S_PIRITSEXP.png
try:
    _c29 = get_crop(29, 288, 156)
    canvas.paste(_c29, (0, 2804), _c29)
except Exception:
    pass
layout["INE_S_PIRITSEXP("] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_06_2024_4_24_16_59_92c22920a83749c994864397a370a984-8/30_icon_1O_00_AM_CDT.png
try:
    _c30 = get_crop(30, 1344, 396)
    canvas.paste(_c30, (48, 2074), _c30)
except Exception:
    pass
layout["1O:00_AM_CDT"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_06_2024_4_24_16_59_92c22920a83749c994864397a370a984-8/31_icon_icon_31.png
try:
    _c31 = get_crop(31, 44, 57)
    canvas.paste(_c31, (385, 6), _c31)
except Exception:
    pass
layout["icon_31"] = [385, 6, 429, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_06_2024_4_24_16_59_92c22920a83749c994864397a370a984-8/32_icon_73_creator_followers.png
try:
    _c32 = get_crop(32, 1344, 396)
    canvas.paste(_c32, (48, 490), _c32)
except Exception:
    pass
layout["73_creator_followers"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_06_2024_4_24_16_59_92c22920a83749c994864397a370a984-8/33_icon_9_298_creator_followers.png
try:
    _c33 = get_crop(33, 288, 156)
    canvas.paste(_c33, (576, 2804), _c33)
except Exception:
    pass
layout["9_298_creator_followers"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_06_2024_4_24_16_59_92c22920a83749c994864397a370a984-8/34_text_5.00.png
try:
    _c34 = get_crop(34, 89, 43)
    canvas.paste(_c34, (22, 17), _c34)
except Exception:
    pass
layout["5.00"] = [22, 17, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_06_2024_4_24_16_59_92c22920a83749c994864397a370a984-8/35_text_More_events_you_II_love.png
try:
    _c35 = get_crop(35, 1344, 396)
    canvas.paste(_c35, (48, 490), _c35)
except Exception:
    pass
layout["More_events_you'II_love"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_06_2024_4_24_16_59_92c22920a83749c994864397a370a984-8/36_text_Tue.png
try:
    _c36 = get_crop(36, 84, 43)
    canvas.paste(_c36, (390, 2525), _c36)
except Exception:
    pass
layout["Tue,"] = [390, 2525, 474, 2568]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_06_2024_4_24_16_59_92c22920a83749c994864397a370a984-8/37_text_24_-.png
try:
    _c37 = get_crop(37, 59, 36)
    canvas.paste(_c37, (540, 2526), _c37)
except Exception:
    pass
layout["24_-"] = [540, 2526, 599, 2562]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_06_2024_4_24_16_59_92c22920a83749c994864397a370a984-8/38_text_4.30_PM_CDT.png
try:
    _c38 = get_crop(38, 1344, 346)
    canvas.paste(_c38, (48, 2470), _c38)
except Exception:
    pass
layout["4.30_PM_CDT"] = [48, 2470, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_06_2024_4_24_16_59_92c22920a83749c994864397a370a984-8/39_text_NDEPEN.png
try:
    _c39 = get_crop(39, 327, 97)
    canvas.paste(_c39, (35, 2570), _c39)
except Exception:
    pass
layout["NDEPEN"] = [35, 2570, 362, 2667]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_06_2024_4_24_16_59_92c22920a83749c994864397a370a984-8/40_text_SB.png
try:
    _c40 = get_crop(40, 55, 31)
    canvas.paste(_c40, (231, 2715), _c40)
except Exception:
    pass
layout["(SB"] = [231, 2715, 286, 2746]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_06_2024_4_24_16_59_92c22920a83749c994864397a370a984-8/41_text_INE_S_PIRITSEXP.png
try:
    _c41 = get_crop(41, 288, 156)
    canvas.paste(_c41, (0, 2804), _c41)
except Exception:
    pass
layout["INE_S_PIRITSEXP("] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_06_2024_4_24_16_59_92c22920a83749c994864397a370a984-8/42_clickable_Tickets.png
try:
    _c42 = get_crop(42, 288, 156)
    canvas.paste(_c42, (864, 2804), _c42)
except Exception:
    pass
layout["Tickets"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_06_2024_4_24_16_59_92c22920a83749c994864397a370a984-8/43_clickable_More.png
try:
    _c43 = get_crop(43, 288, 156)
    canvas.paste(_c43, (1152, 2804), _c43)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]
