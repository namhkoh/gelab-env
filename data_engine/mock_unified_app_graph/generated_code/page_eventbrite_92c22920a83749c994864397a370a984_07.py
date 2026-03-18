# page_id: page_eventbrite_92c22920a83749c994864397a370a984_07
# screenshot: 2024_4_24_16_59_92c22920a83749c994864397a370a984-9.png
# step_index: 7/13
# task: Open Eventbrite. Set the city to "Chicago". Select the "Sports" category and view the recommended events. See the date of the first non-promoted event.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Background and structural UI elements for Eventbrite-like page
bg = "#F5F6F8"
status_bar = "#BDBDBD"
header_bg = "#FFFFFF"
divider = "#E6E6E9"
card_shadow = "#E9EDF2"
card_bg = "#FFFFFF"
nav_bg = "#FFFFFF"
nav_border = "#E6E6E9"

# Fill overall background
draw.rectangle([(0, 0), (1440, 2960)], fill=bg)

# Status bar (top)
draw.rectangle([(0, 0), (1440, 72)], fill=status_bar)

# Header / Search area background (behind detected search crop at ~y=72)
search_area = (48, 72, 1392, 248)
# shadow for header card
draw.rectangle([(search_area[0]-4, search_area[1]-4), (search_area[2]+4, search_area[3]+6)], fill=card_shadow)
# main search background
try:
    draw.rounded_rectangle(search_area, radius=18, fill=header_bg, outline=divider, width=1)
except AttributeError:
    draw.rectangle(search_area, fill=header_bg, outline=divider)

# Thin divider under header / search
draw.line([(32, 256), (1408, 256)], fill=divider, width=2)

# Section title divider (approx where "10,000 events" sits) - subtle top spacing line
draw.line([(48, 336), (1392, 336)], fill=divider, width=1)

# First event card background (behind large event image area)
card1_x0, card1_y0 = 48, 676
card1_w, card1_h = 1344, 1096
card1_rect = (card1_x0, card1_y0, card1_x0 + card1_w, card1_y0 + card1_h)
# shadow
draw.rectangle([(card1_rect[0]-6, card1_rect[1]-6), (card1_rect[2]+6, card1_rect[3]+8)], fill=card_shadow)
# card white background with rounded corners
try:
    draw.rounded_rectangle(card1_rect, radius=24, fill=card_bg, outline=divider, width=1)
except AttributeError:
    draw.rectangle(card1_rect, fill=card_bg, outline=divider)

# Small separator under first card (between cards and metadata)
draw.line([(48, card1_rect[3] + 28), (1392, card1_rect[3] + 28)], fill=divider, width=1)

# Second event/promoted card background (behind detected large promo image)
card2_x0, card2_y0 = 48, 1820
card2_w, card2_h = 1344, 996
card2_rect = (card2_x0, card2_y0, card2_x0 + card2_w, card2_y0 + card2_h)
# shadow
draw.rectangle([(card2_rect[0]-6, card2_rect[1]-6), (card2_rect[2]+6, card2_rect[3]+8)], fill=card_shadow)
# card background
try:
    draw.rounded_rectangle(card2_rect, radius=20, fill=card_bg, outline=divider, width=1)
except AttributeError:
    draw.rectangle(card2_rect, fill=card_bg, outline=divider)

# Light section divider above bottom navigation
draw.line([(32, 2808), (1408, 2808)], fill=divider, width=2)

# Bottom navigation bar background
nav_y0 = 2810
draw.rectangle([(0, nav_y0), (1440, 2960)], fill=nav_bg)
# top border of nav
draw.line([(0, nav_y0), (1440, nav_y0)], fill=nav_border, width=2)

# Floating action area shadows for right-side action buttons (visual background behind detected favorite/overflow icons)
# Top action cluster (near first card)
fa_center_x, fa_center_y = 1170, 1192  # center-ish for two stacked action icons (detected icons will be pasted)
# subtle circular shadow background behind icons (do not draw icon glyphs)
draw.ellipse([(fa_center_x-92, fa_center_y-92), (fa_center_x+92, fa_center_y+92)], fill=card_shadow)
# Lower action cluster (near second card)
fa2_center_x, fa2_center_y = 1170, 2336
draw.ellipse([(fa2_center_x-92, fa2_center_y-92), (fa2_center_x+92, fa2_center_y+92)], fill=card_shadow)

# Overall minor horizontal separators to structure list (between likely list items)
for y in (520, 920, 1440, 1680, 2160, 2560):
    draw.line([(48, y), (1392, y)], fill=divider, width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_07_2024_4_24_16_59_92c22920a83749c994864397a370a984-9/00_icon_Music.png
try:
    _c0 = get_crop(0, 187, 103)
    canvas.paste(_c0, (837, 410), _c0)
except Exception:
    pass
layout["Music"] = [837, 410, 1024, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_07_2024_4_24_16_59_92c22920a83749c994864397a370a984-9/01_icon_Business.png
try:
    _c1 = get_crop(1, 241, 103)
    canvas.paste(_c1, (1036, 410), _c1)
except Exception:
    pass
layout["Business"] = [1036, 410, 1277, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_07_2024_4_24_16_59_92c22920a83749c994864397a370a984-9/02_icon_Anytime.png
try:
    _c2 = get_crop(2, 400, 103)
    canvas.paste(_c2, (425, 410), _c2)
except Exception:
    pass
layout["Anytime"] = [425, 410, 825, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_07_2024_4_24_16_59_92c22920a83749c994864397a370a984-9/03_icon_Filters.png
try:
    _c3 = get_crop(3, 359, 103)
    canvas.paste(_c3, (54, 410), _c3)
except Exception:
    pass
layout["Filters"] = [54, 410, 413, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_07_2024_4_24_16_59_92c22920a83749c994864397a370a984-9/04_icon_Favorite_button.png
try:
    _c4 = get_crop(4, 144, 144)
    canvas.paste(_c4, (1092, 2336), _c4)
except Exception:
    pass
layout["Favorite_button"] = [1092, 2336, 1236, 2480]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_07_2024_4_24_16_59_92c22920a83749c994864397a370a984-9/05_icon_Favorite_button.png
try:
    _c5 = get_crop(5, 144, 144)
    canvas.paste(_c5, (1092, 1192), _c5)
except Exception:
    pass
layout["Favorite_button"] = [1092, 1192, 1236, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_07_2024_4_24_16_59_92c22920a83749c994864397a370a984-9/06_icon_Foo.png
try:
    _c6 = get_crop(6, 148, 110)
    canvas.paste(_c6, (1282, 406), _c6)
except Exception:
    pass
layout["Foo"] = [1282, 406, 1430, 516]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_07_2024_4_24_16_59_92c22920a83749c994864397a370a984-9/07_icon_Overflow_menu_button.png
try:
    _c7 = get_crop(7, 144, 144)
    canvas.paste(_c7, (1236, 2336), _c7)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 2336, 1380, 2480]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_07_2024_4_24_16_59_92c22920a83749c994864397a370a984-9/08_icon_Overflow_menu_button.png
try:
    _c8 = get_crop(8, 144, 144)
    canvas.paste(_c8, (1236, 1192), _c8)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 1192, 1380, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_07_2024_4_24_16_59_92c22920a83749c994864397a370a984-9/09_icon_5.00.png
try:
    _c9 = get_crop(9, 118, 110)
    canvas.paste(_c9, (59, 117), _c9)
except Exception:
    pass
layout["5.00"] = [59, 117, 177, 227]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_07_2024_4_24_16_59_92c22920a83749c994864397a370a984-9/10_icon_Search_forae.png
try:
    _c10 = get_crop(10, 1344, 191)
    canvas.paste(_c10, (48, 72), _c10)
except Exception:
    pass
layout["Search_forae"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_07_2024_4_24_16_59_92c22920a83749c994864397a370a984-9/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 53, 65)
    canvas.paste(_c11, (1152, 0), _c11)
except Exception:
    pass
layout["icon_11"] = [1152, 0, 1205, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_07_2024_4_24_16_59_92c22920a83749c994864397a370a984-9/12_icon_Search_forae.png
try:
    _c12 = get_crop(12, 67, 62)
    canvas.paste(_c12, (308, 1), _c12)
except Exception:
    pass
layout["Search_forae"] = [308, 1, 375, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_07_2024_4_24_16_59_92c22920a83749c994864397a370a984-9/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 71, 62)
    canvas.paste(_c13, (1212, 0), _c13)
except Exception:
    pass
layout["icon_13"] = [1212, 0, 1283, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_07_2024_4_24_16_59_92c22920a83749c994864397a370a984-9/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 51, 62)
    canvas.paste(_c14, (249, 1), _c14)
except Exception:
    pass
layout["icon_14"] = [249, 1, 300, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_07_2024_4_24_16_59_92c22920a83749c994864397a370a984-9/15_icon_5.00.png
try:
    _c15 = get_crop(15, 60, 63)
    canvas.paste(_c15, (181, 0), _c15)
except Exception:
    pass
layout["5.00"] = [181, 0, 241, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_07_2024_4_24_16_59_92c22920a83749c994864397a370a984-9/16_icon_5.00.png
try:
    _c16 = get_crop(16, 59, 66)
    canvas.paste(_c16, (115, 0), _c16)
except Exception:
    pass
layout["5.00"] = [115, 0, 174, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_07_2024_4_24_16_59_92c22920a83749c994864397a370a984-9/17_icon_Chicago.png
try:
    _c17 = get_crop(17, 417, 144)
    canvas.paste(_c17, (0, 259), _c17)
except Exception:
    pass
layout["Chicago"] = [0, 259, 417, 403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_07_2024_4_24_16_59_92c22920a83749c994864397a370a984-9/18_icon_icon_18.png
try:
    _c18 = get_crop(18, 57, 59)
    canvas.paste(_c18, (1318, 0), _c18)
except Exception:
    pass
layout["icon_18"] = [1318, 0, 1375, 59]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_07_2024_4_24_16_59_92c22920a83749c994864397a370a984-9/19_icon_prevention.png
try:
    _c19 = get_crop(19, 100, 120)
    canvas.paste(_c19, (51, 2278), _c19)
except Exception:
    pass
layout["prevention"] = [51, 2278, 151, 2398]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_07_2024_4_24_16_59_92c22920a83749c994864397a370a984-9/20_icon_Powered_by_FLASHTRANSACT.png
try:
    _c20 = get_crop(20, 1344, 1096)
    canvas.paste(_c20, (48, 676), _c20)
except Exception:
    pass
layout["Powered_by:_FLASHTRANSACT"] = [48, 676, 1392, 1772]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_07_2024_4_24_16_59_92c22920a83749c994864397a370a984-9/21_icon_Search_forae.png
try:
    _c21 = get_crop(21, 52, 61)
    canvas.paste(_c21, (383, 2), _c21)
except Exception:
    pass
layout["Search_forae"] = [383, 2, 435, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_07_2024_4_24_16_59_92c22920a83749c994864397a370a984-9/22_icon_Fri_Apr_26_._1_OO_AM_CDT.png
try:
    _c22 = get_crop(22, 288, 156)
    canvas.paste(_c22, (288, 2804), _c22)
except Exception:
    pass
layout["Fri,_Apr_26_._1:OO_AM_CDT"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_07_2024_4_24_16_59_92c22920a83749c994864397a370a984-9/23_icon_Breathe_Well_Free_Lung_Health_Screening.png
try:
    _c23 = get_crop(23, 288, 156)
    canvas.paste(_c23, (864, 2804), _c23)
except Exception:
    pass
layout["Breathe_Well:_Free_Lung_H"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_07_2024_4_24_16_59_92c22920a83749c994864397a370a984-9/24_icon_Breathe_Well_Free_Lung_Health_Screening.png
try:
    _c24 = get_crop(24, 288, 156)
    canvas.paste(_c24, (1152, 2804), _c24)
except Exception:
    pass
layout["Breathe_Well:_Free_Lung_H"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_07_2024_4_24_16_59_92c22920a83749c994864397a370a984-9/25_icon_Event.png
try:
    _c25 = get_crop(25, 288, 156)
    canvas.paste(_c25, (0, 2804), _c25)
except Exception:
    pass
layout["Event"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_07_2024_4_24_16_59_92c22920a83749c994864397a370a984-9/26_icon_icon_26.png
try:
    _c26 = get_crop(26, 41, 61)
    canvas.paste(_c26, (1273, 0), _c26)
except Exception:
    pass
layout["icon_26"] = [1273, 0, 1314, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_07_2024_4_24_16_59_92c22920a83749c994864397a370a984-9/27_icon_Breathe_Well_Free_Lung_Health_Screening.png
try:
    _c27 = get_crop(27, 288, 156)
    canvas.paste(_c27, (576, 2804), _c27)
except Exception:
    pass
layout["Breathe_Well:_Free_Lung_H"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_07_2024_4_24_16_59_92c22920a83749c994864397a370a984-9/28_icon_Breathe_Well_Free_Lung_Health_Screening.png
try:
    _c28 = get_crop(28, 1344, 996)
    canvas.paste(_c28, (48, 1820), _c28)
except Exception:
    pass
layout["Breathe_Well:_Free_Lung_H"] = [48, 1820, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_07_2024_4_24_16_59_92c22920a83749c994864397a370a984-9/29_icon_Promoted.png
try:
    _c29 = get_crop(29, 244, 63)
    canvas.paste(_c29, (83, 1667), _c29)
except Exception:
    pass
layout["Promoted"] = [83, 1667, 327, 1730]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_07_2024_4_24_16_59_92c22920a83749c994864397a370a984-9/30_text_5.00.png
try:
    _c30 = get_crop(30, 91, 45)
    canvas.paste(_c30, (20, 15), _c30)
except Exception:
    pass
layout["5.00"] = [20, 15, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_07_2024_4_24_16_59_92c22920a83749c994864397a370a984-9/31_text_10_000_events.png
try:
    _c31 = get_crop(31, 359, 103)
    canvas.paste(_c31, (54, 410), _c31)
except Exception:
    pass
layout["10,000_events"] = [54, 410, 413, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_07_2024_4_24_16_59_92c22920a83749c994864397a370a984-9/32_text_GenHarp.png
try:
    _c32 = get_crop(32, 214, 69)
    canvas.paste(_c32, (234, 1875), _c32)
except Exception:
    pass
layout["GenHarp"] = [234, 1875, 448, 1944]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_07_2024_4_24_16_59_92c22920a83749c994864397a370a984-9/33_text_REGISTER_NOW.png
try:
    _c33 = get_crop(33, 256, 39)
    canvas.paste(_c33, (928, 1897), _c33)
except Exception:
    pass
layout["REGISTER_NOW"] = [928, 1897, 1184, 1936]
