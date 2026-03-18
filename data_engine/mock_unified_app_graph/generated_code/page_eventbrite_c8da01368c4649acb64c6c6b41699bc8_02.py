# page_id: page_eventbrite_c8da01368c4649acb64c6c6b41699bc8_02
# screenshot: 2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-4.png
# step_index: 2/13
# task: Open Eventbrite. Look up "Animal" events. Filter by events happening next week. Select the first event - who is the organizer?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# draw background and structural elements for the UI
draw.rectangle((0, 0, 1440, 2960), fill="#FFFFFF")  # base background

# Status bar (top)
status_h = 72
draw.rectangle((0, 0, 1440, status_h), fill="#cfcfcf")  # light gray status bar
draw.line((0, status_h-1, 1440, status_h-1), fill="#bdbdbd", width=1)

# Header / search area
header_top = status_h
header_bottom = 160
draw.rectangle((0, header_top, 1440, header_bottom), fill="#FFFFFF")
# subtle divider under header
draw.line((0, header_bottom, 1440, header_bottom), fill="#ececf2", width=1)
# prominent blue underline for search field (aligned with page content margins)
underline_y = 140
draw.line((48, underline_y, 1392, underline_y), fill="#2b56ff", width=4)

# Large section card background (rounded) for the list of recent items / categories
card_left = 48
card_right = 1392
card_top = 260
card_bottom = 1760
draw.rounded_rectangle((card_left, card_top, card_right, card_bottom),
                       radius=16, fill="#fbfbfd", outline="#f0f0f5", width=1)

# Row separators inside the card (do not draw icons/text)
separator_color = "#ececf2"
row_ys = [534, 678, 822, 966, 1110, 1254, 1398, 1542, 1698]
for y in row_ys:
    draw.line((card_left + 24, y, card_right - 24, y), fill=separator_color, width=1)

# Thin left/right inner guides (very subtle)
draw.line((card_left, card_top+8, card_left, card_bottom-8), fill="#fafafb", width=1)
draw.line((card_right, card_top+8, card_right, card_bottom-8), fill="#fafafb", width=1)

# Additional subtle separators near top of content (under "Recent" area)
draw.line((card_left + 12, card_top + 20, card_right - 12, card_top + 20), fill="#f3f3f6", width=1)

# Bottom navigation background and top divider
nav_top = 2804
draw.rectangle((0, nav_top, 1440, 2960), fill="#FFFFFF")
draw.line((0, nav_top, 1440, nav_top), fill="#e6e6ee", width=2)
# tiny top shadow band
draw.rectangle((0, nav_top-6, 1440, nav_top), fill="#f7f7fa")

# subtle left and right page edge lines
draw.line((0, 0, 0, 2960), fill="#ffffff", width=1)
draw.line((1439, 0, 1439, 2960), fill="#ffffff", width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_02_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-4/00_icon_5.15.png
try:
    _c0 = get_crop(0, 58, 64)
    canvas.paste(_c0, (115, 1), _c0)
except Exception:
    pass
layout["5.15"] = [115, 1, 173, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_02_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-4/01_icon_5.15.png
try:
    _c1 = get_crop(1, 58, 62)
    canvas.paste(_c1, (181, 1), _c1)
except Exception:
    pass
layout["5.15"] = [181, 1, 239, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_02_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-4/02_icon_Search_forae.png
try:
    _c2 = get_crop(2, 63, 62)
    canvas.paste(_c2, (309, 2), _c2)
except Exception:
    pass
layout["Search_forae"] = [309, 2, 372, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_02_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-4/03_icon_icon_3.png
try:
    _c3 = get_crop(3, 49, 61)
    canvas.paste(_c3, (249, 2), _c3)
except Exception:
    pass
layout["icon_3"] = [249, 2, 298, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_02_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-4/04_icon_Cancel.png
try:
    _c4 = get_crop(4, 149, 144)
    canvas.paste(_c4, (1243, 97), _c4)
except Exception:
    pass
layout["Cancel"] = [1243, 97, 1392, 241]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_02_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-4/05_icon_Cancel.png
try:
    _c5 = get_crop(5, 57, 62)
    canvas.paste(_c5, (1316, 0), _c5)
except Exception:
    pass
layout["Cancel"] = [1316, 0, 1373, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_02_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-4/06_icon_Cancel.png
try:
    _c6 = get_crop(6, 97, 62)
    canvas.paste(_c6, (1212, 0), _c6)
except Exception:
    pass
layout["Cancel"] = [1212, 0, 1309, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_02_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-4/07_icon_Open_Mic_Night.png
try:
    _c7 = get_crop(7, 1344, 144)
    canvas.paste(_c7, (48, 534), _c7)
except Exception:
    pass
layout["Open_Mic_Night"] = [48, 534, 1392, 678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_02_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-4/08_icon_Outdoor.png
try:
    _c8 = get_crop(8, 114, 128)
    canvas.paste(_c8, (26, 1698), _c8)
except Exception:
    pass
layout["Outdoor"] = [26, 1698, 140, 1826]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_02_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-4/09_icon_5.15.png
try:
    _c9 = get_crop(9, 124, 113)
    canvas.paste(_c9, (55, 115), _c9)
except Exception:
    pass
layout["5.15"] = [55, 115, 179, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_02_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-4/10_icon_Tickets.png
try:
    _c10 = get_crop(10, 288, 156)
    canvas.paste(_c10, (864, 2804), _c10)
except Exception:
    pass
layout["Tickets"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_02_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-4/11_icon_Close_current_screen.png
try:
    _c11 = get_crop(11, 144, 144)
    canvas.paste(_c11, (1248, 822), _c11)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 822, 1392, 966]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_02_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-4/12_icon_Close_current_screen.png
try:
    _c12 = get_crop(12, 144, 144)
    canvas.paste(_c12, (1248, 534), _c12)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 534, 1392, 678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_02_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-4/13_icon_Close_current_screen.png
try:
    _c13 = get_crop(13, 144, 144)
    canvas.paste(_c13, (1248, 1686), _c13)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 1686, 1392, 1830]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_02_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-4/14_icon_Close_current_screen.png
try:
    _c14 = get_crop(14, 144, 144)
    canvas.paste(_c14, (1248, 678), _c14)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 678, 1392, 822]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_02_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-4/15_icon_community_events.png
try:
    _c15 = get_crop(15, 1344, 144)
    canvas.paste(_c15, (48, 1542), _c15)
except Exception:
    pass
layout["community_events"] = [48, 1542, 1392, 1686]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_02_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-4/16_icon_Close_current_screen.png
try:
    _c16 = get_crop(16, 144, 144)
    canvas.paste(_c16, (1248, 1254), _c16)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 1254, 1392, 1398]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_02_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-4/17_icon_Close_current_screen.png
try:
    _c17 = get_crop(17, 144, 144)
    canvas.paste(_c17, (1248, 1110), _c17)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 1110, 1392, 1254]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_02_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-4/18_icon_Close_current_screen.png
try:
    _c18 = get_crop(18, 144, 144)
    canvas.paste(_c18, (1248, 1398), _c18)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 1398, 1392, 1542]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_02_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-4/19_icon_Cancel.png
try:
    _c19 = get_crop(19, 144, 144)
    canvas.paste(_c19, (1248, 390), _c19)
except Exception:
    pass
layout["Cancel"] = [1248, 390, 1392, 534]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_02_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-4/20_icon_Close_current_screen.png
try:
    _c20 = get_crop(20, 144, 144)
    canvas.paste(_c20, (1248, 1542), _c20)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 1542, 1392, 1686]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_02_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-4/21_icon_Search_forae.png
try:
    _c21 = get_crop(21, 1344, 144)
    canvas.paste(_c21, (48, 390), _c21)
except Exception:
    pass
layout["Search_forae"] = [48, 390, 1392, 534]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_02_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-4/22_icon_Favorites.png
try:
    _c22 = get_crop(22, 288, 156)
    canvas.paste(_c22, (576, 2804), _c22)
except Exception:
    pass
layout["Favorites"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_02_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-4/23_icon_Close_current_screen.png
try:
    _c23 = get_crop(23, 144, 144)
    canvas.paste(_c23, (1248, 966), _c23)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 966, 1392, 1110]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_02_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-4/24_icon_Search_events.png
try:
    _c24 = get_crop(24, 288, 156)
    canvas.paste(_c24, (288, 2804), _c24)
except Exception:
    pass
layout["Search_events"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_02_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-4/25_icon_Open_Mic_Night.png
try:
    _c25 = get_crop(25, 1344, 144)
    canvas.paste(_c25, (48, 678), _c25)
except Exception:
    pass
layout["Open_Mic_Night"] = [48, 678, 1392, 822]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_02_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-4/26_icon_Home.png
try:
    _c26 = get_crop(26, 288, 156)
    canvas.paste(_c26, (0, 2804), _c26)
except Exception:
    pass
layout["Home"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_02_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-4/27_icon_community_events.png
try:
    _c27 = get_crop(27, 1344, 144)
    canvas.paste(_c27, (48, 1398), _c27)
except Exception:
    pass
layout["community_events"] = [48, 1398, 1392, 1542]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_02_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-4/28_icon_Search_forae.png
try:
    _c28 = get_crop(28, 47, 64)
    canvas.paste(_c28, (383, 2), _c28)
except Exception:
    pass
layout["Search_forae"] = [383, 2, 430, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_02_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-4/29_icon_Search_forae.png
try:
    _c29 = get_crop(29, 1344, 191)
    canvas.paste(_c29, (48, 72), _c29)
except Exception:
    pass
layout["Search_forae"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_02_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-4/30_icon_Cooking.png
try:
    _c30 = get_crop(30, 1344, 144)
    canvas.paste(_c30, (48, 1254), _c30)
except Exception:
    pass
layout["Cooking"] = [48, 1254, 1392, 1398]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_02_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-4/31_icon_Wellness.png
try:
    _c31 = get_crop(31, 1344, 144)
    canvas.paste(_c31, (48, 1110), _c31)
except Exception:
    pass
layout["Wellness"] = [48, 1110, 1392, 1254]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_02_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-4/32_icon_More.png
try:
    _c32 = get_crop(32, 288, 156)
    canvas.paste(_c32, (1152, 2804), _c32)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_02_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-4/33_text_5.15.png
try:
    _c33 = get_crop(33, 92, 43)
    canvas.paste(_c33, (22, 17), _c33)
except Exception:
    pass
layout["5.15"] = [22, 17, 114, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_02_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-4/34_text_Recent.png
try:
    _c34 = get_crop(34, 200, 56)
    canvas.paste(_c34, (46, 301), _c34)
except Exception:
    pass
layout["Recent"] = [46, 301, 246, 357]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_02_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-4/35_text_Photography.png
try:
    _c35 = get_crop(35, 250, 61)
    canvas.paste(_c35, (160, 870), _c35)
except Exception:
    pass
layout["Photography"] = [160, 870, 410, 931]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_02_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-4/36_text_Pets.png
try:
    _c36 = get_crop(36, 96, 54)
    canvas.paste(_c36, (162, 1016), _c36)
except Exception:
    pass
layout["Pets"] = [162, 1016, 258, 1070]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_02_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-4/37_text_community_events.png
try:
    _c37 = get_crop(37, 1344, 144)
    canvas.paste(_c37, (48, 1686), _c37)
except Exception:
    pass
layout["community_events"] = [48, 1686, 1392, 1830]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_02_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-4/38_clickable_Photography.png
try:
    _c38 = get_crop(38, 1344, 144)
    canvas.paste(_c38, (48, 822), _c38)
except Exception:
    pass
layout["Photography"] = [48, 822, 1392, 966]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_02_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-4/39_clickable_Pets.png
try:
    _c39 = get_crop(39, 1344, 144)
    canvas.paste(_c39, (48, 966), _c39)
except Exception:
    pass
layout["Pets"] = [48, 966, 1392, 1110]
