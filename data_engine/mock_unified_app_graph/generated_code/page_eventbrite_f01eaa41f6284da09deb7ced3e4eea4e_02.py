# page_id: page_eventbrite_f01eaa41f6284da09deb7ced3e4eea4e_02
# screenshot: 2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-4.png
# step_index: 2/11
# task: Open Eventbrite. Check out 'Sports' events. Apply filters for events happening this week. Select the first event. Check similar events and add the first similar event to favorite.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Top-level background (dominant white)
draw.rectangle([(0, 0), (1440, 2960)], fill="#ffffff")

# Status bar (top area)
status_bar_height = 72
draw.rectangle([(0, 0), (1440, status_bar_height)], fill="#cfcfcf")

# Search/header area background (behind the detected search elements)
search_x0, search_y0 = 48, status_bar_height
search_w, search_h = 1344, 191
search_rect = (search_x0, search_y0, search_x0 + search_w, search_y0 + search_h)
draw.rounded_rectangle(search_rect, radius=8, fill="#ffffff", outline=None)

# Blue underline / divider under the search field (visual separator)
underline_y = search_y0 + int(search_h * 0.75)
draw.line([(search_x0, underline_y), (search_x0 + search_w, underline_y)], fill="#2f55d9", width=4)

# Subtle divider under the header area
draw.line([(24, underline_y + 18), (1440 - 24, underline_y + 18)], fill="#e6e6e9", width=1)

# Large content panel behind the list of "Recent" items (soft off-white card)
panel_x0, panel_y0 = 24, underline_y + 36
panel_x1, panel_y1 = 1440 - 24, 1800
draw.rounded_rectangle((panel_x0, panel_y0, panel_x1, panel_y1), radius=12, fill="#fbfbfd", outline=None)

# Section separators for list groups (thin lines to hint structure)
separator_color = "#efeef1"
# Use the detected row Y positions as guiding separators (approximate)
row_tops = [390, 534, 678, 822, 966, 1110, 1254, 1398, 1542, 1686]
for y in row_tops:
    draw.line([(panel_x0 + 16, y), (panel_x1 - 16, y)], fill=separator_color, width=1)

# Slight left margin guide (vertical faint line to group content area)
draw.line([(panel_x0 + 96, panel_y0 + 8), (panel_x0 + 96, panel_y1 - 8)], fill="#f3f3f5", width=1)

# Bottom navigation bar background and top divider
nav_top = 2804
draw.line([(0, nav_top), (1440, nav_top)], fill="#e6e6e9", width=2)
draw.rectangle([(0, nav_top), (1440, 2960)], fill="#ffffff")

# Subtle shadow above bottom nav to lift it visually
shadow_y0 = nav_top - 6
for i, alpha in enumerate([60, 40, 20], start=0):
    y = shadow_y0 + i
    draw.line([(0, y), (1440, y)], fill="#f0f0f0", width=1)

# Small divider near very bottom (edge accent)
draw.line([(24, 2956), (1440 - 24, 2956)], fill="#f6f6f8", width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_02_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-4/00_icon_4.35.png
try:
    _c0 = get_crop(0, 60, 64)
    canvas.paste(_c0, (114, 1), _c0)
except Exception:
    pass
layout["4.35"] = [114, 1, 174, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_02_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-4/01_icon_4.35.png
try:
    _c1 = get_crop(1, 58, 63)
    canvas.paste(_c1, (181, 1), _c1)
except Exception:
    pass
layout["4.35"] = [181, 1, 239, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_02_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-4/02_icon_Search_for_-..png
try:
    _c2 = get_crop(2, 64, 63)
    canvas.paste(_c2, (309, 2), _c2)
except Exception:
    pass
layout["(Search_for:-."] = [309, 2, 373, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_02_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-4/03_icon_icon_3.png
try:
    _c3 = get_crop(3, 49, 62)
    canvas.paste(_c3, (249, 2), _c3)
except Exception:
    pass
layout["icon_3"] = [249, 2, 298, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_02_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-4/04_icon_Cancel.png
try:
    _c4 = get_crop(4, 149, 144)
    canvas.paste(_c4, (1243, 97), _c4)
except Exception:
    pass
layout["Cancel"] = [1243, 97, 1392, 241]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_02_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-4/05_icon_community_events.png
try:
    _c5 = get_crop(5, 1344, 144)
    canvas.paste(_c5, (48, 534), _c5)
except Exception:
    pass
layout["community_events"] = [48, 534, 1392, 678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_02_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-4/06_icon_Cancel.png
try:
    _c6 = get_crop(6, 57, 62)
    canvas.paste(_c6, (1316, 0), _c6)
except Exception:
    pass
layout["Cancel"] = [1316, 0, 1373, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_02_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-4/07_icon_Cancel.png
try:
    _c7 = get_crop(7, 99, 62)
    canvas.paste(_c7, (1212, 0), _c7)
except Exception:
    pass
layout["Cancel"] = [1212, 0, 1311, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_02_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-4/08_icon_community_events.png
try:
    _c8 = get_crop(8, 1344, 144)
    canvas.paste(_c8, (48, 390), _c8)
except Exception:
    pass
layout["community_events"] = [48, 390, 1392, 534]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_02_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-4/09_icon_4.35.png
try:
    _c9 = get_crop(9, 123, 109)
    canvas.paste(_c9, (54, 114), _c9)
except Exception:
    pass
layout["4.35"] = [54, 114, 177, 223]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_02_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-4/10_icon_Tickets.png
try:
    _c10 = get_crop(10, 288, 156)
    canvas.paste(_c10, (864, 2804), _c10)
except Exception:
    pass
layout["Tickets"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_02_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-4/11_icon_Close_current_screen.png
try:
    _c11 = get_crop(11, 144, 144)
    canvas.paste(_c11, (1248, 822), _c11)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 822, 1392, 966]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_02_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-4/12_icon_Music_Festival.png
try:
    _c12 = get_crop(12, 112, 129)
    canvas.paste(_c12, (27, 1697), _c12)
except Exception:
    pass
layout["Music_Festival"] = [27, 1697, 139, 1826]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_02_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-4/13_icon_community_events.png
try:
    _c13 = get_crop(13, 1344, 144)
    canvas.paste(_c13, (48, 678), _c13)
except Exception:
    pass
layout["community_events"] = [48, 678, 1392, 822]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_02_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-4/14_icon_Close_current_screen.png
try:
    _c14 = get_crop(14, 144, 144)
    canvas.paste(_c14, (1248, 534), _c14)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 534, 1392, 678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_02_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-4/15_icon_Close_current_screen.png
try:
    _c15 = get_crop(15, 144, 144)
    canvas.paste(_c15, (1248, 678), _c15)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 678, 1392, 822]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_02_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-4/16_icon_Close_current_screen.png
try:
    _c16 = get_crop(16, 144, 144)
    canvas.paste(_c16, (1248, 1398), _c16)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 1398, 1392, 1542]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_02_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-4/17_icon_Favorites.png
try:
    _c17 = get_crop(17, 288, 156)
    canvas.paste(_c17, (576, 2804), _c17)
except Exception:
    pass
layout["Favorites"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_02_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-4/18_icon_Close_current_screen.png
try:
    _c18 = get_crop(18, 144, 144)
    canvas.paste(_c18, (1248, 1686), _c18)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 1686, 1392, 1830]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_02_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-4/19_icon_Close_current_screen.png
try:
    _c19 = get_crop(19, 144, 144)
    canvas.paste(_c19, (1248, 1254), _c19)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 1254, 1392, 1398]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_02_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-4/20_icon_Close_current_screen.png
try:
    _c20 = get_crop(20, 144, 144)
    canvas.paste(_c20, (1248, 1110), _c20)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 1110, 1392, 1254]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_02_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-4/21_icon_Close_current_screen.png
try:
    _c21 = get_crop(21, 144, 144)
    canvas.paste(_c21, (1248, 1542), _c21)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 1542, 1392, 1686]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_02_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-4/22_icon_Cancel.png
try:
    _c22 = get_crop(22, 144, 144)
    canvas.paste(_c22, (1248, 390), _c22)
except Exception:
    pass
layout["Cancel"] = [1248, 390, 1392, 534]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_02_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-4/23_icon_Yoga_session.png
try:
    _c23 = get_crop(23, 1344, 144)
    canvas.paste(_c23, (48, 1398), _c23)
except Exception:
    pass
layout["Yoga_session"] = [48, 1398, 1392, 1542]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_02_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-4/24_icon_Close_current_screen.png
try:
    _c24 = get_crop(24, 144, 144)
    canvas.paste(_c24, (1248, 966), _c24)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 966, 1392, 1110]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_02_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-4/25_icon_Search_for_-..png
try:
    _c25 = get_crop(25, 1344, 191)
    canvas.paste(_c25, (48, 72), _c25)
except Exception:
    pass
layout["(Search_for:-."] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_02_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-4/26_icon_Home.png
try:
    _c26 = get_crop(26, 288, 156)
    canvas.paste(_c26, (0, 2804), _c26)
except Exception:
    pass
layout["Home"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_02_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-4/27_icon_Search_events.png
try:
    _c27 = get_crop(27, 288, 156)
    canvas.paste(_c27, (288, 2804), _c27)
except Exception:
    pass
layout["Search_events"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_02_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-4/28_icon_4.35.png
try:
    _c28 = get_crop(28, 95, 62)
    canvas.paste(_c28, (13, 2), _c28)
except Exception:
    pass
layout["4.35"] = [13, 2, 108, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_02_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-4/29_icon_Coding_Workshop.png
try:
    _c29 = get_crop(29, 1344, 144)
    canvas.paste(_c29, (48, 1542), _c29)
except Exception:
    pass
layout["Coding_Workshop"] = [48, 1542, 1392, 1686]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_02_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-4/30_icon_More.png
try:
    _c30 = get_crop(30, 288, 156)
    canvas.paste(_c30, (1152, 2804), _c30)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_02_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-4/31_icon_community_events.png
try:
    _c31 = get_crop(31, 1344, 144)
    canvas.paste(_c31, (48, 822), _c31)
except Exception:
    pass
layout["community_events"] = [48, 822, 1392, 966]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_02_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-4/32_icon_Search_for_-..png
try:
    _c32 = get_crop(32, 47, 65)
    canvas.paste(_c32, (383, 2), _c32)
except Exception:
    pass
layout["(Search_for:-."] = [383, 2, 430, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_02_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-4/33_icon_Food_and_Drink.png
try:
    _c33 = get_crop(33, 1344, 144)
    canvas.paste(_c33, (48, 1254), _c33)
except Exception:
    pass
layout["Food_and_Drink"] = [48, 1254, 1392, 1398]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_02_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-4/34_text_Art.png
try:
    _c34 = get_crop(34, 64, 43)
    canvas.paste(_c34, (165, 1019), _c34)
except Exception:
    pass
layout["Art"] = [165, 1019, 229, 1062]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_02_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-4/35_text_Food_and_Drink.png
try:
    _c35 = get_crop(35, 286, 49)
    canvas.paste(_c35, (164, 1159), _c35)
except Exception:
    pass
layout["Food_and_Drink"] = [164, 1159, 450, 1208]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_02_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-4/36_text_Music_Festival.png
try:
    _c36 = get_crop(36, 273, 52)
    canvas.paste(_c36, (163, 1734), _c36)
except Exception:
    pass
layout["Music_Festival"] = [163, 1734, 436, 1786]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_02_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-4/37_clickable_Art.png
try:
    _c37 = get_crop(37, 1344, 144)
    canvas.paste(_c37, (48, 966), _c37)
except Exception:
    pass
layout["Art"] = [48, 966, 1392, 1110]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_02_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-4/38_clickable_Food_and_Drink.png
try:
    _c38 = get_crop(38, 1344, 144)
    canvas.paste(_c38, (48, 1110), _c38)
except Exception:
    pass
layout["Food_and_Drink"] = [48, 1110, 1392, 1254]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_02_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-4/39_clickable_Music_Festival.png
try:
    _c39 = get_crop(39, 1344, 144)
    canvas.paste(_c39, (48, 1686), _c39)
except Exception:
    pass
layout["Music_Festival"] = [48, 1686, 1392, 1830]
