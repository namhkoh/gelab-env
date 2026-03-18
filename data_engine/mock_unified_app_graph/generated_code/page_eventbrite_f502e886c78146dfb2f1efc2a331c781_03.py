# page_id: page_eventbrite_f502e886c78146dfb2f1efc2a331c781_03
# screenshot: 2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-5.png
# step_index: 3/18
# task: Open Eventbrite. Search for 'music festival' in San Francisco. Set date available from April 30 to May 3. How many events are listed?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# paint overall background
draw.rectangle((0, 0, 1440, 2960), fill=(255, 255, 255))

# status bar (top area)
status_h = 72
draw.rectangle((0, 0, 1440, status_h), fill=(205, 205, 205))

# subtle divider under status bar
draw.line((0, status_h, 1440, status_h), fill=(190, 190, 190), width=1)

# large search/header area background (behind detected search widgets)
search_x = 48
search_y = 72
search_w = 1344
search_h = 191
search_rect = (search_x, search_y, search_x + search_w, search_y + search_h)
draw.rounded_rectangle(search_rect, radius=12, fill=(250, 250, 252), outline=None)

# prominent blue underline under header/search area (thin accent line)
blue_y = search_y + 80
draw.rectangle((search_x, blue_y, search_x + search_w, blue_y + 4), fill=(34, 84, 255))

# card container behind the list of recent items (subtle white card with faint border)
list_top = 360
list_bottom = 1840
card_rect = (36, list_top, 1404, list_bottom)
draw.rounded_rectangle(card_rect, radius=10, fill=(255, 255, 255), outline=(235, 235, 240), width=1)

# draw horizontal separators for each row (the rows are 144px tall at the detected positions)
row_positions = [390, 534, 678, 822, 966, 1110, 1254, 1398, 1542, 1686]
left_inset = 48
right_inset = 1392
for y in row_positions:
    # bottom divider of each row
    draw.line((left_inset, y + 144, right_inset, y + 144), fill=(235, 235, 240), width=1)

# subtle vertical guide/margin line (decorative, not overlapping content)
guide_x = 48
draw.line((guide_x, list_top, guide_x, list_bottom), fill=(245, 245, 247), width=1)

# draw a faint divider under the "Recent" header area (near where the header text sits)
recent_div_y = 350
draw.line((48, recent_div_y, 1392, recent_div_y), fill=(240, 240, 245), width=1)

# bottom navigation background and top divider
nav_top = 2804
draw.rectangle((0, nav_top - 4, 1440, 2960), fill=(250, 250, 252))
draw.line((0, nav_top - 4, 1440, nav_top - 4), fill=(220, 220, 225), width=1)

# subtle shadow line above bottom nav for separation
draw.line((0, nav_top - 8, 1440, nav_top - 8), fill=(245, 245, 247), width=1)

# final subtle left/right padding guides at extreme edges (very light)
draw.line((16, 0, 16, 2960), fill=(255, 255, 255), width=1)
draw.line((1424, 0, 1424, 2960), fill=(255, 255, 255), width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_03_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-5/00_icon_7.18.png
try:
    _c0 = get_crop(0, 58, 63)
    canvas.paste(_c0, (115, 1), _c0)
except Exception:
    pass
layout["7.18"] = [115, 1, 173, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_03_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-5/01_icon_7.18.png
try:
    _c1 = get_crop(1, 58, 63)
    canvas.paste(_c1, (181, 0), _c1)
except Exception:
    pass
layout["7.18"] = [181, 0, 239, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_03_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-5/02_icon_Search_forae.png
try:
    _c2 = get_crop(2, 63, 64)
    canvas.paste(_c2, (309, 1), _c2)
except Exception:
    pass
layout["Search_forae"] = [309, 1, 372, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_03_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-5/03_icon_icon_3.png
try:
    _c3 = get_crop(3, 49, 61)
    canvas.paste(_c3, (249, 2), _c3)
except Exception:
    pass
layout["icon_3"] = [249, 2, 298, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_03_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-5/04_icon_Cancel.png
try:
    _c4 = get_crop(4, 98, 62)
    canvas.paste(_c4, (1212, 0), _c4)
except Exception:
    pass
layout["Cancel"] = [1212, 0, 1310, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_03_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-5/05_icon_Cancel.png
try:
    _c5 = get_crop(5, 58, 62)
    canvas.paste(_c5, (1316, 0), _c5)
except Exception:
    pass
layout["Cancel"] = [1316, 0, 1374, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_03_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-5/06_icon_Education.png
try:
    _c6 = get_crop(6, 1344, 144)
    canvas.paste(_c6, (48, 534), _c6)
except Exception:
    pass
layout["Education"] = [48, 534, 1392, 678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_03_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-5/07_icon_Cancel.png
try:
    _c7 = get_crop(7, 149, 144)
    canvas.paste(_c7, (1243, 97), _c7)
except Exception:
    pass
layout["Cancel"] = [1243, 97, 1392, 241]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_03_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-5/08_icon_Favorites.png
try:
    _c8 = get_crop(8, 288, 156)
    canvas.paste(_c8, (576, 2804), _c8)
except Exception:
    pass
layout["Favorites"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_03_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-5/09_icon_Search_forae.png
try:
    _c9 = get_crop(9, 1344, 144)
    canvas.paste(_c9, (48, 390), _c9)
except Exception:
    pass
layout["Search_forae"] = [48, 390, 1392, 534]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_03_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-5/10_icon_7.18.png
try:
    _c10 = get_crop(10, 127, 114)
    canvas.paste(_c10, (53, 114), _c10)
except Exception:
    pass
layout["7.18"] = [53, 114, 180, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_03_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-5/11_icon_Close_current_screen.png
try:
    _c11 = get_crop(11, 144, 144)
    canvas.paste(_c11, (1248, 1254), _c11)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 1254, 1392, 1398]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_03_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-5/12_icon_Close_current_screen.png
try:
    _c12 = get_crop(12, 144, 144)
    canvas.paste(_c12, (1248, 822), _c12)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 822, 1392, 966]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_03_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-5/13_icon_Close_current_screen.png
try:
    _c13 = get_crop(13, 144, 144)
    canvas.paste(_c13, (1248, 1398), _c13)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 1398, 1392, 1542]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_03_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-5/14_icon_Basketball.png
try:
    _c14 = get_crop(14, 1344, 144)
    canvas.paste(_c14, (48, 1398), _c14)
except Exception:
    pass
layout["Basketball"] = [48, 1398, 1392, 1542]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_03_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-5/15_icon_Tickets.png
try:
    _c15 = get_crop(15, 288, 156)
    canvas.paste(_c15, (864, 2804), _c15)
except Exception:
    pass
layout["Tickets"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_03_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-5/16_icon_Close_current_screen.png
try:
    _c16 = get_crop(16, 144, 144)
    canvas.paste(_c16, (1248, 678), _c16)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 678, 1392, 822]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_03_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-5/17_icon_Close_current_screen.png
try:
    _c17 = get_crop(17, 144, 144)
    canvas.paste(_c17, (1248, 534), _c17)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 534, 1392, 678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_03_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-5/18_icon_Science_Tech.png
try:
    _c18 = get_crop(18, 1344, 144)
    canvas.paste(_c18, (48, 1254), _c18)
except Exception:
    pass
layout["Science_&_Tech"] = [48, 1254, 1392, 1398]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_03_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-5/19_icon_Close_current_screen.png
try:
    _c19 = get_crop(19, 144, 144)
    canvas.paste(_c19, (1248, 1110), _c19)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 1110, 1392, 1254]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_03_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-5/20_icon_Close_current_screen.png
try:
    _c20 = get_crop(20, 144, 144)
    canvas.paste(_c20, (1248, 1686), _c20)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 1686, 1392, 1830]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_03_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-5/21_icon_Close_current_screen.png
try:
    _c21 = get_crop(21, 144, 144)
    canvas.paste(_c21, (1248, 1542), _c21)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 1542, 1392, 1686]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_03_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-5/22_icon_Home.png
try:
    _c22 = get_crop(22, 288, 156)
    canvas.paste(_c22, (0, 2804), _c22)
except Exception:
    pass
layout["Home"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_03_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-5/23_icon_Education.png
try:
    _c23 = get_crop(23, 1344, 144)
    canvas.paste(_c23, (48, 678), _c23)
except Exception:
    pass
layout["Education"] = [48, 678, 1392, 822]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_03_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-5/24_icon_Search_forae.png
try:
    _c24 = get_crop(24, 48, 65)
    canvas.paste(_c24, (383, 2), _c24)
except Exception:
    pass
layout["Search_forae"] = [383, 2, 431, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_03_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-5/25_icon_Cancel.png
try:
    _c25 = get_crop(25, 144, 144)
    canvas.paste(_c25, (1248, 390), _c25)
except Exception:
    pass
layout["Cancel"] = [1248, 390, 1392, 534]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_03_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-5/26_icon_Search_events.png
try:
    _c26 = get_crop(26, 288, 156)
    canvas.paste(_c26, (288, 2804), _c26)
except Exception:
    pass
layout["Search_events"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_03_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-5/27_icon_More.png
try:
    _c27 = get_crop(27, 288, 156)
    canvas.paste(_c27, (1152, 2804), _c27)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_03_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-5/28_icon_Close_current_screen.png
try:
    _c28 = get_crop(28, 144, 144)
    canvas.paste(_c28, (1248, 966), _c28)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 966, 1392, 1110]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_03_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-5/29_icon_Search_forae.png
try:
    _c29 = get_crop(29, 1344, 191)
    canvas.paste(_c29, (48, 72), _c29)
except Exception:
    pass
layout["Search_forae"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_03_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-5/30_icon_Exhibition.png
try:
    _c30 = get_crop(30, 1344, 144)
    canvas.paste(_c30, (48, 1542), _c30)
except Exception:
    pass
layout["Exhibition"] = [48, 1542, 1392, 1686]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_03_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-5/31_icon_Science_Tech.png
try:
    _c31 = get_crop(31, 1344, 144)
    canvas.paste(_c31, (48, 1110), _c31)
except Exception:
    pass
layout["Science_&_Tech"] = [48, 1110, 1392, 1254]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_03_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-5/32_icon_Food_Drink.png
try:
    _c32 = get_crop(32, 1344, 144)
    canvas.paste(_c32, (48, 966), _c32)
except Exception:
    pass
layout["Food_&_Drink"] = [48, 966, 1392, 1110]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_03_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-5/33_icon_Festival.png
try:
    _c33 = get_crop(33, 116, 129)
    canvas.paste(_c33, (26, 1697), _c33)
except Exception:
    pass
layout["Festival"] = [26, 1697, 142, 1826]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_03_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-5/34_icon_7.18.png
try:
    _c34 = get_crop(34, 95, 63)
    canvas.paste(_c34, (13, 1), _c34)
except Exception:
    pass
layout["7.18"] = [13, 1, 108, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_03_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-5/35_text_Recent.png
try:
    _c35 = get_crop(35, 200, 56)
    canvas.paste(_c35, (46, 301), _c35)
except Exception:
    pass
layout["Recent"] = [46, 301, 246, 357]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_03_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-5/36_text_Music.png
try:
    _c36 = get_crop(36, 124, 53)
    canvas.paste(_c36, (163, 871), _c36)
except Exception:
    pass
layout["Music"] = [163, 871, 287, 924]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_03_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-5/37_text_Festival.png
try:
    _c37 = get_crop(37, 154, 45)
    canvas.paste(_c37, (163, 1738), _c37)
except Exception:
    pass
layout["Festival"] = [163, 1738, 317, 1783]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_03_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-5/38_clickable_Music.png
try:
    _c38 = get_crop(38, 1344, 144)
    canvas.paste(_c38, (48, 822), _c38)
except Exception:
    pass
layout["Music"] = [48, 822, 1392, 966]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_03_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-5/39_clickable_Festival.png
try:
    _c39 = get_crop(39, 1344, 144)
    canvas.paste(_c39, (48, 1686), _c39)
except Exception:
    pass
layout["Festival"] = [48, 1686, 1392, 1830]
