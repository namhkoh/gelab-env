# page_id: page_eventbrite_9fdb2ee43d5a49adac5304bdd5dacfc2_07
# screenshot: 2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-9.png
# step_index: 7/8
# task: Open Eventbrite. Look up 'Pet' events. Filter by events happening this weekend. Select the third non-promoted event from the results - how much are the tickets for the event?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Canvas and draw are provided (1440x2960 RGB). Fonts: font_sm, font_md, font_lg, font_xl
# This script draws only background and structural chrome (status bar, headers, card backgrounds, separators).

# Colors
bg_color = "#FBFBFD"          # very light page background
status_bar_color = "#CFCFCF"  # top status bar grey
header_bg = "#FFFFFF"         # header white
divider = "#E6E7EB"           # subtle divider
card_shadow = "#F0F1F5"       # shadow under cards
card_bg = "#FFFFFF"           # card main background
card_border = "#E9E9EF"       # card border

w, h = canvas.size

# Fill overall background
draw.rectangle((0, 0, w, h), fill=bg_color)

# Status bar (top area - slightly darker)
status_h = 96
draw.rectangle((0, 0, w, status_h), fill=status_bar_color)

# Header / toolbar area below status bar
header_top = status_h
header_bottom = 190
draw.rectangle((0, header_top, w, header_bottom), fill=header_bg)
# header bottom divider
draw.line((48, header_bottom, w-48, header_bottom), fill=divider, width=1)

# Thin separator under filter/search row (approx)
filter_sep_y = 256
draw.line((48, filter_sep_y, w-48, filter_sep_y), fill=divider, width=1)

# Define card container positions (aligned to detected UI margins)
card_x1 = 48
card_x2 = w - 48

# First event card - spans image + text block (do not draw internal content)
card1_y1 = 200
card1_y2 = 1343
radius = 24
# shadow
draw.rounded_rectangle((card_x1+6, card1_y1+8, card_x2+6, card1_y2+8), radius=radius+2, fill=card_shadow, outline=None)
# card background with subtle border
draw.rounded_rectangle((card_x1, card1_y1, card_x2, card1_y2), radius=radius, fill=card_bg, outline=card_border, width=1)

# Divider between cards
draw.line((card_x1+8, card1_y2+20, card_x2-8, card1_y2+20), fill=divider, width=1)

# Second event card
card2_y1 = 1391
card2_y2 = 2499
# shadow
draw.rounded_rectangle((card_x1+6, card2_y1+8, card_x2+6, card2_y2+8), radius=radius+2, fill=card_shadow, outline=None)
# card background
draw.rounded_rectangle((card_x1, card2_y1, card_x2, card2_y2), radius=radius, fill=card_bg, outline=card_border, width=1)

# Separator line above bottom navigation (so nav stands out)
nav_top_y = 2804
draw.line((0, nav_top_y, w, nav_top_y), fill=divider, width=1)

# Light horizontal guides for content rhythm (subtle)
# small separators to structure content areas (do not add text/icons)
for y in (card1_y1 + 320, card1_y1 + 640, card2_y1 + 320, card2_y1 + 640):
    if y < nav_top_y - 120:
        draw.line((card_x1+12, y, card_x2-12, y), fill="#F4F5F7", width=1)

# Top-left search strip background (behind search area, not drawing icons/text)
search_strip_top = header_bottom + 8
search_strip_bottom = search_strip_top + 86
draw.rectangle((48, search_strip_top, card_x2, search_strip_bottom), fill=card_bg, outline=card_border)

# Slight inset shadow under the search strip
draw.line((48, search_strip_bottom+1, card_x2, search_strip_bottom+1), fill="#F0F1F5", width=2)

# End of structural drawing.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_07_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-9/00_icon_Music.png
try:
    _c0 = get_crop(0, 196, 110)
    canvas.paste(_c0, (947, 406), _c0)
except Exception:
    pass
layout["Music"] = [947, 406, 1143, 516]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_07_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-9/01_icon_This_Weekend.png
try:
    _c1 = get_crop(1, 1344, 191)
    canvas.paste(_c1, (48, 72), _c1)
except Exception:
    pass
layout["This_Weekend"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_07_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-9/02_icon_Business.png
try:
    _c2 = get_crop(2, 250, 112)
    canvas.paste(_c2, (1145, 406), _c2)
except Exception:
    pass
layout["Business"] = [1145, 406, 1395, 518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_07_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-9/03_icon_Favorite_button.png
try:
    _c3 = get_crop(3, 144, 144)
    canvas.paste(_c3, (1092, 811), _c3)
except Exception:
    pass
layout["Favorite_button"] = [1092, 811, 1236, 955]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_07_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-9/04_icon_1_Filter.png
try:
    _c4 = get_crop(4, 493, 144)
    canvas.paste(_c4, (0, 259), _c4)
except Exception:
    pass
layout["1_Filter"] = [0, 259, 493, 403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_07_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-9/05_icon_Overflow_menu_button.png
try:
    _c5 = get_crop(5, 144, 144)
    canvas.paste(_c5, (1236, 811), _c5)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 811, 1380, 955]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_07_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-9/06_icon_Paint_Your_Pet_Glen_Burnie_Bubba_s_33_wi.png
try:
    _c6 = get_crop(6, 1344, 1108)
    canvas.paste(_c6, (48, 1391), _c6)
except Exception:
    pass
layout["Paint_Your_Pet!_Glen_Burn"] = [48, 1391, 1392, 2499]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_07_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-9/07_icon_Favorite_button.png
try:
    _c7 = get_crop(7, 144, 144)
    canvas.paste(_c7, (1092, 1907), _c7)
except Exception:
    pass
layout["Favorite_button"] = [1092, 1907, 1236, 2051]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_07_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-9/08_icon_Overflow_menu_button.png
try:
    _c8 = get_crop(8, 144, 144)
    canvas.paste(_c8, (1236, 1907), _c8)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 1907, 1380, 2051]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_07_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-9/09_icon_Sales_ended.png
try:
    _c9 = get_crop(9, 252, 81)
    canvas.paste(_c9, (91, 2082), _c9)
except Exception:
    pass
layout["Sales_ended"] = [91, 2082, 343, 2163]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_07_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-9/10_icon_Washington.png
try:
    _c10 = get_crop(10, 493, 144)
    canvas.paste(_c10, (0, 259), _c10)
except Exception:
    pass
layout["Washington"] = [0, 259, 493, 403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_07_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-9/11_icon_Close_current_screen.png
try:
    _c11 = get_crop(11, 144, 144)
    canvas.paste(_c11, (1248, 96), _c11)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 96, 1392, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_07_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-9/12_icon_4.48.png
try:
    _c12 = get_crop(12, 55, 65)
    canvas.paste(_c12, (116, 1), _c12)
except Exception:
    pass
layout["4.48"] = [116, 1, 171, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_07_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-9/13_icon_4.48.png
try:
    _c13 = get_crop(13, 110, 109)
    canvas.paste(_c13, (62, 116), _c13)
except Exception:
    pass
layout["4.48"] = [62, 116, 172, 225]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_07_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-9/14_icon_April_Community_Tours_Meet_the_Animals.png
try:
    _c14 = get_crop(14, 1344, 818)
    canvas.paste(_c14, (48, 525), _c14)
except Exception:
    pass
layout["April_Community_Tours!_Me"] = [48, 525, 1392, 1343]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_07_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-9/15_icon_4.48.png
try:
    _c15 = get_crop(15, 54, 64)
    canvas.paste(_c15, (183, 1), _c15)
except Exception:
    pass
layout["4.48"] = [183, 1, 237, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_07_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-9/16_icon_Pets.png
try:
    _c16 = get_crop(16, 62, 63)
    canvas.paste(_c16, (310, 1), _c16)
except Exception:
    pass
layout["Pets"] = [310, 1, 372, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_07_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-9/17_icon_Pets.png
try:
    _c17 = get_crop(17, 45, 63)
    canvas.paste(_c17, (251, 1), _c17)
except Exception:
    pass
layout["Pets"] = [251, 1, 296, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_07_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-9/18_icon_icon_18.png
try:
    _c18 = get_crop(18, 70, 62)
    canvas.paste(_c18, (1210, 0), _c18)
except Exception:
    pass
layout["icon_18"] = [1210, 0, 1280, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_07_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-9/19_icon_icon_19.png
try:
    _c19 = get_crop(19, 53, 62)
    canvas.paste(_c19, (1318, 0), _c19)
except Exception:
    pass
layout["icon_19"] = [1318, 0, 1371, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_07_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-9/20_icon_Tickets.png
try:
    _c20 = get_crop(20, 288, 156)
    canvas.paste(_c20, (864, 2804), _c20)
except Exception:
    pass
layout["Tickets"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_07_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-9/21_icon_Pets.png
try:
    _c21 = get_crop(21, 1344, 191)
    canvas.paste(_c21, (48, 72), _c21)
except Exception:
    pass
layout["Pets"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_07_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-9/22_icon_Search_events.png
try:
    _c22 = get_crop(22, 288, 156)
    canvas.paste(_c22, (288, 2804), _c22)
except Exception:
    pass
layout["Search_events"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_07_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-9/23_icon_icon_23.png
try:
    _c23 = get_crop(23, 56, 63)
    canvas.paste(_c23, (1257, 0), _c23)
except Exception:
    pass
layout["icon_23"] = [1257, 0, 1313, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_07_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-9/24_icon_4.48.png
try:
    _c24 = get_crop(24, 92, 64)
    canvas.paste(_c24, (15, 0), _c24)
except Exception:
    pass
layout["4.48"] = [15, 0, 107, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_07_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-9/25_icon_More.png
try:
    _c25 = get_crop(25, 288, 156)
    canvas.paste(_c25, (1152, 2804), _c25)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_07_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-9/26_icon_Paint_Your_Pet_Glen_Burnie_Bubba_s_33_wi.png
try:
    _c26 = get_crop(26, 1344, 1108)
    canvas.paste(_c26, (48, 1391), _c26)
except Exception:
    pass
layout["Paint_Your_Pet!_Glen_Burn"] = [48, 1391, 1392, 2499]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_07_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-9/27_icon_icon_27.png
try:
    _c27 = get_crop(27, 45, 62)
    canvas.paste(_c27, (385, 2), _c27)
except Exception:
    pass
layout["icon_27"] = [385, 2, 430, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_07_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-9/28_icon_Home.png
try:
    _c28 = get_crop(28, 288, 156)
    canvas.paste(_c28, (0, 2804), _c28)
except Exception:
    pass
layout["Home"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_07_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-9/29_text_Bubba_s_33.png
try:
    _c29 = get_crop(29, 211, 45)
    canvas.paste(_c29, (94, 2402), _c29)
except Exception:
    pass
layout["Bubba's_33"] = [94, 2402, 305, 2447]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_07_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-9/30_clickable_Favorites.png
try:
    _c30 = get_crop(30, 288, 156)
    canvas.paste(_c30, (576, 2804), _c30)
except Exception:
    pass
layout["Favorites"] = [576, 2804, 864, 2960]
