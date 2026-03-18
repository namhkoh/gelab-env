# page_id: page_eventbrite_4fbf805fbd914a178f72f68b0bc03f81_06
# screenshot: 2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-8.png
# step_index: 6/10
# task: Open Eventbrite. Explore "Education" events. Apply filters for events happening tomorrow. From the list, select the third event and check out its description.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw overall background
draw.rectangle((0, 0, 1440, 2960), fill="#F6F7FB")

# Status bar (top area)
status_h = 96
draw.rectangle((0, 0, 1440, status_h), fill="#9A9EA0")

# Header / toolbar area below status bar
header_top = status_h
header_h = 140
draw.rectangle((0, header_top, 1440, header_top + header_h), fill="#FFFFFF")
# Thin divider under header
divider_y = header_top + header_h
draw.line((40, divider_y, 1400, divider_y), fill="#E3E6EA", width=2)

# Location / filters row area (kept subtle, content will be pasted on top)
filters_top = divider_y
filters_h = 174  # stretches down to where filter pills are pasted (~y=410)
draw.rectangle((0, filters_top, 1440, filters_top + filters_h), fill="#FFFFFF")
# Bottom separator for filters area
filters_bottom = filters_top + filters_h
draw.line((40, filters_bottom, 1400, filters_bottom), fill="#E9ECF0", width=2)

# Function to draw card with subtle shadow and rounded corners
def draw_card(x, y, w, h, radius=28, fill="#FFFFFF", shadow_color="#EDF1F4"):
    # shadow (slightly offset)
    shadow_offset = 8
    sx0, sy0, sx1, sy1 = x, y + shadow_offset, x + w, y + h + shadow_offset
    try:
        draw.rounded_rectangle((sx0, sy0, sx1, sy1), radius=radius, fill=shadow_color)
    except TypeError:
        # older Pillow fallback: draw.rectangle shadow
        draw.rectangle((sx0, sy0, sx1, sy1), fill=shadow_color)
    # main card
    try:
        draw.rounded_rectangle((x, y, x + w, y + h), radius=radius, fill=fill, outline="#E6E9EE", width=1)
    except TypeError:
        draw.rectangle((x, y, x + w, y + h), fill=fill, outline="#E6E9EE")

# First large event card (background/frame only)
card1_x, card1_y = 48, 676
card1_w, card1_h = 1344, 1175
draw_card(card1_x, card1_y, card1_w, card1_h, radius=20)

# Small separator/gutter below first card (visual)
gut_y = card1_y + card1_h + 28
draw.line((64, gut_y, 1376, gut_y), fill="#F0F2F5", width=1)

# Second large event card / content area (background/frame only)
card2_x, card2_y = 48, 1899
card2_w, card2_h = 1344, 917
draw_card(card2_x, card2_y, card2_w, card2_h, radius=20)

# Top of page subtle headline area separator (where "244 events" appears)
headline_y = filters_bottom + 20
draw.line((48, headline_y + 96, 1392, headline_y + 96), fill="#F2F4F6", width=1)

# Bottom navigation bar background and top divider
nav_height = 156  # matches detected bottom element heights
nav_top = 2960 - nav_height
draw.rectangle((0, nav_top, 1440, 2960), fill="#FFFFFF")
draw.line((24, nav_top, 1416, nav_top), fill="#E3E6EA", width=2)

# Safe area padding/shadow above navigation for depth
draw.line((24, nav_top + 2, 1416, nav_top + 2), fill="#F5F7F9", width=1)

# Additional subtle separators for content sections
# Separator between first card and promoted tag area (approx)
sep_y1 = card1_y + card1_h - 160
draw.line((64, sep_y1, 1376, sep_y1), fill="#F6F7F9", width=1)

# Small horizontal rule above second card
sep_y2 = card2_y - 24
draw.line((64, sep_y2, 1376, sep_y2), fill="#F6F7F9", width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_06_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-8/00_icon_Tomorrow.png
try:
    _c0 = get_crop(0, 432, 103)
    canvas.paste(_c0, (438, 410), _c0)
except Exception:
    pass
layout["Tomorrow"] = [438, 410, 870, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_06_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-8/01_icon_Music.png
try:
    _c1 = get_crop(1, 187, 103)
    canvas.paste(_c1, (882, 410), _c1)
except Exception:
    pass
layout["Music"] = [882, 410, 1069, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_06_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-8/02_icon_Business.png
try:
    _c2 = get_crop(2, 241, 103)
    canvas.paste(_c2, (1081, 410), _c2)
except Exception:
    pass
layout["Business"] = [1081, 410, 1322, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_06_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-8/03_icon_1_Filter.png
try:
    _c3 = get_crop(3, 372, 103)
    canvas.paste(_c3, (54, 410), _c3)
except Exception:
    pass
layout["1_Filter"] = [54, 410, 426, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_06_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-8/04_icon_Favorite_button.png
try:
    _c4 = get_crop(4, 144, 144)
    canvas.paste(_c4, (1092, 2415), _c4)
except Exception:
    pass
layout["Favorite_button"] = [1092, 2415, 1236, 2559]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_06_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-8/05_icon_Overflow_menu_button.png
try:
    _c5 = get_crop(5, 144, 144)
    canvas.paste(_c5, (1236, 2415), _c5)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 2415, 1380, 2559]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_06_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-8/06_icon_Business.png
try:
    _c6 = get_crop(6, 93, 109)
    canvas.paste(_c6, (1329, 408), _c6)
except Exception:
    pass
layout["Business"] = [1329, 408, 1422, 517]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_06_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-8/07_icon_Favorite_button.png
try:
    _c7 = get_crop(7, 144, 144)
    canvas.paste(_c7, (1092, 1192), _c7)
except Exception:
    pass
layout["Favorite_button"] = [1092, 1192, 1236, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_06_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-8/08_icon_Overflow_menu_button.png
try:
    _c8 = get_crop(8, 144, 144)
    canvas.paste(_c8, (1236, 1192), _c8)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 1192, 1380, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_06_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-8/09_icon_The_Goran_Technique_How_To_Deal_With.png
try:
    _c9 = get_crop(9, 1344, 1175)
    canvas.paste(_c9, (48, 676), _c9)
except Exception:
    pass
layout["The_Goran_Technique:_How_"] = [48, 676, 1392, 1851]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_06_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-8/10_icon_Close_current_screen.png
try:
    _c10 = get_crop(10, 144, 144)
    canvas.paste(_c10, (1248, 96), _c10)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 96, 1392, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_06_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-8/11_icon_Education.png
try:
    _c11 = get_crop(11, 65, 62)
    canvas.paste(_c11, (308, 1), _c11)
except Exception:
    pass
layout["Education"] = [308, 1, 373, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_06_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-8/12_icon_4.56.png
try:
    _c12 = get_crop(12, 120, 114)
    canvas.paste(_c12, (56, 114), _c12)
except Exception:
    pass
layout["4.56"] = [56, 114, 176, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_06_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-8/13_icon_4.56.png
try:
    _c13 = get_crop(13, 58, 63)
    canvas.paste(_c13, (181, 1), _c13)
except Exception:
    pass
layout["4.56"] = [181, 1, 239, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_06_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-8/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 101, 62)
    canvas.paste(_c14, (1209, 0), _c14)
except Exception:
    pass
layout["icon_14"] = [1209, 0, 1310, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_06_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-8/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 49, 61)
    canvas.paste(_c15, (250, 1), _c15)
except Exception:
    pass
layout["icon_15"] = [250, 1, 299, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_06_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-8/16_icon_4.56.png
try:
    _c16 = get_crop(16, 58, 64)
    canvas.paste(_c16, (115, 0), _c16)
except Exception:
    pass
layout["4.56"] = [115, 0, 173, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_06_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-8/17_icon_Los_Angeles.png
try:
    _c17 = get_crop(17, 492, 144)
    canvas.paste(_c17, (0, 259), _c17)
except Exception:
    pass
layout["Los_Angeles"] = [0, 259, 492, 403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_06_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-8/18_icon_icon_18.png
try:
    _c18 = get_crop(18, 58, 62)
    canvas.paste(_c18, (1317, 0), _c18)
except Exception:
    pass
layout["icon_18"] = [1317, 0, 1375, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_06_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-8/19_icon_Education.png
try:
    _c19 = get_crop(19, 1344, 191)
    canvas.paste(_c19, (48, 72), _c19)
except Exception:
    pass
layout["Education"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_06_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-8/20_icon_Education.png
try:
    _c20 = get_crop(20, 48, 61)
    canvas.paste(_c20, (384, 2), _c20)
except Exception:
    pass
layout["Education"] = [384, 2, 432, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_06_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-8/21_icon_God.png
try:
    _c21 = get_crop(21, 1344, 917)
    canvas.paste(_c21, (48, 1899), _c21)
except Exception:
    pass
layout["God"] = [48, 1899, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_06_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-8/22_icon_IAI_EL.png
try:
    _c22 = get_crop(22, 288, 156)
    canvas.paste(_c22, (288, 2804), _c22)
except Exception:
    pass
layout["IAI_EL_"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_06_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-8/23_icon_IAI_EL.png
try:
    _c23 = get_crop(23, 288, 156)
    canvas.paste(_c23, (576, 2804), _c23)
except Exception:
    pass
layout["IAI_EL_"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_06_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-8/24_icon_Promoted.png
try:
    _c24 = get_crop(24, 245, 64)
    canvas.paste(_c24, (82, 1744), _c24)
except Exception:
    pass
layout["Promoted"] = [82, 1744, 327, 1808]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_06_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-8/25_icon_An_Intro_to_Voice_Overs.png
try:
    _c25 = get_crop(25, 288, 156)
    canvas.paste(_c25, (1152, 2804), _c25)
except Exception:
    pass
layout["An_Intro_to_Voice_Overs"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_06_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-8/26_icon_An_Intro_to_Voice_Overs.png
try:
    _c26 = get_crop(26, 288, 156)
    canvas.paste(_c26, (864, 2804), _c26)
except Exception:
    pass
layout["An_Intro_to_Voice_Overs"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_06_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-8/27_icon_An_Intro_to_Voice_Overs.png
try:
    _c27 = get_crop(27, 288, 156)
    canvas.paste(_c27, (864, 2804), _c27)
except Exception:
    pass
layout["An_Intro_to_Voice_Overs"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_06_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-8/28_icon_Ticket_sales_end_soon.png
try:
    _c28 = get_crop(28, 288, 156)
    canvas.paste(_c28, (288, 2804), _c28)
except Exception:
    pass
layout["Ticket_sales_end_soon"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_06_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-8/29_text_4.56.png
try:
    _c29 = get_crop(29, 89, 43)
    canvas.paste(_c29, (22, 17), _c29)
except Exception:
    pass
layout["4.56"] = [22, 17, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_06_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-8/30_text_244_events.png
try:
    _c30 = get_crop(30, 372, 103)
    canvas.paste(_c30, (54, 410), _c30)
except Exception:
    pass
layout["244_events"] = [54, 410, 426, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_06_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-8/31_text_Online.png
try:
    _c31 = get_crop(31, 129, 45)
    canvas.paste(_c31, (91, 1687), _c31)
except Exception:
    pass
layout["Online"] = [91, 1687, 220, 1732]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_06_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-8/32_text_IAI_EL.png
try:
    _c32 = get_crop(32, 288, 156)
    canvas.paste(_c32, (288, 2804), _c32)
except Exception:
    pass
layout["IAI_EL_"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_06_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-8/33_clickable_Home.png
try:
    _c33 = get_crop(33, 288, 156)
    canvas.paste(_c33, (0, 2804), _c33)
except Exception:
    pass
layout["Home"] = [0, 2804, 288, 2960]
