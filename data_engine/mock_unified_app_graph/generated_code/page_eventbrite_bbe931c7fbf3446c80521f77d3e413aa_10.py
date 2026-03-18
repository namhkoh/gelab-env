# page_id: page_eventbrite_bbe931c7fbf3446c80521f77d3e413aa_10
# screenshot: 2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-12.png
# step_index: 10/20
# task: Open Eventbrite. Search free events in Los Angeles. Select the first one. Follow the organizer. Read more about the event. Add it to Favorites.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: fallback_compose
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Paint the overall background
bg_color = "#F6F7FB"         # soft off-white dominant background
draw.rectangle((0, 0, 1440, 2960), fill=bg_color)

# Status bar area (top)
status_h = 72
status_color = "#9CA0A4"     # slightly darker grey for status bar
draw.rectangle((0, 0, 1440, status_h), fill=status_color)
# subtle bottom divider for status bar
draw.line((0, status_h - 1, 1440, status_h - 1), fill="#BFC4C7", width=1)

# Search/header area (rounded white card behind search field)
search_x0, search_y0 = 48, 72
search_w, search_h = 1344, 191
search_box = (search_x0, search_y0, search_x0 + search_w, search_y0 + search_h)
draw.rounded_rectangle(search_box, radius=14, fill="#FFFFFF", outline=None)
# divider under the search area
draw.line((search_x0, search_y0 + search_h + 6, search_x0 + search_w, search_y0 + search_h + 6), fill="#E6EAEE", width=1)

# Thin section divider below filter chips area (between filters and results)
# Provide a subtle separator across content width (respecting left/right margins)
filters_div_y = 480
draw.line((48, filters_div_y, 1392, filters_div_y), fill="#ECEFF2", width=1)

# First event card background with a soft shadow
card1_x0, card1_y0 = 48, 676
card1_w, card1_h = 1344, 1175
card1_box = (card1_x0, card1_y0, card1_x0 + card1_w, card1_y0 + card1_h)

# shadow (simulated by a slightly offset pale rectangle)
shadow_offset = 8
shadow_box = (card1_box[0] + shadow_offset, card1_box[1] + shadow_offset,
              card1_box[2] + shadow_offset, card1_box[3] + shadow_offset)
draw.rounded_rectangle(shadow_box, radius=28, fill="#E8ECEF")

# main card
draw.rounded_rectangle(card1_box, radius=24, fill="#FFFFFF", outline="#E6E9EC", width=1)

# Image/content background inside first card (image will be pasted on top)
img1_margin = 16
img1_box = (card1_x0 + img1_margin, card1_y0 + img1_margin,
            card1_x0 + card1_w - img1_margin, card1_y0 + img1_h - img1_margin - 360)
draw.rectangle(img1_box, fill="#F1F3F5", outline=None, width=0)

# subtle inner divider under the image area in the card
img1_div_y = img1_box[3] + 12
draw.line((card1_x0 + 12, img1_div_y, card1_x0 + card1_w - 12, img1_div_y), fill="#F0F2F4", width=1)

# Second event card (lower) with shadow
card2_x0, card2_y0 = 48, 1899
card2_w, card2_h = 1344, 917
card2_box = (card2_x0, card2_y0, card2_x0 + card2_w, card2_y0 + card2_h)

# shadow for second card
shadow_box2 = (card2_box[0] + shadow_offset, card2_box[1] + shadow_offset,
               card2_box[2] + shadow_offset, card2_box[3] + shadow_offset)
draw.rounded_rectangle(shadow_box2, radius=24, fill="#E8ECEF")

# main second card
draw.rounded_rectangle(card2_box, radius=20, fill="#FFFFFF", outline="#E6E9EC", width=1)

# Image/content background inside second card (space for event image)
img2_margin = 16
img2_box = (card2_x0 + img2_margin, card2_y0 + img2_margin,
            card2_x0 + card2_w - img2_margin, card2_y0 + int(card2_h * 0.45))
draw.rectangle(img2_box, fill="#EFEFF1", outline=None)

# divider under second card's image
img2_div_y = img2_box[3] + 12
draw.line((card2_x0 + 12, img2_div_y, card2_x0 + card2_w - 12, img2_div_y), fill="#F0F2F4", width=1)

# Global subtle horizontal separators to break sections
sep_y_positions = [img1_div_y + 220, card2_box[3] + 30]
for y in sep_y_positions:
    if 0 < y < 2960:
        draw.line((48, y, 1392, y), fill="#F3F5F7", width=1)

# Bottom navigation bar background and top divider
nav_h = 160
nav_y0 = 2960 - nav_h
draw.rectangle((0, nav_y0, 1440, 2960), fill="#FFFFFF")
# top divider of nav
draw.line((0, nav_y0, 1440, nav_y0), fill="#E6E9EC", width=1)

# Small indicator bar above bottom nav (subtle)
indicator_y = nav_y0 + 12
draw.line((72, indicator_y, 1368, indicator_y), fill="#FA7C42", width=3)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_10_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-12/00_icon_Music.png
try:
    _c0 = get_crop(0, 187, 103)
    canvas.paste(_c0, (850, 410), _c0)
except Exception:
    pass
layout["Music"] = [850, 410, 1037, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_10_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-12/01_icon_Anytime.png
try:
    _c1 = get_crop(1, 400, 103)
    canvas.paste(_c1, (438, 410), _c1)
except Exception:
    pass
layout["Anytime"] = [438, 410, 838, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_10_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-12/02_icon_Business.png
try:
    _c2 = get_crop(2, 241, 103)
    canvas.paste(_c2, (1049, 410), _c2)
except Exception:
    pass
layout["Business"] = [1049, 410, 1290, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_10_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-12/03_icon_1_Filter.png
try:
    _c3 = get_crop(3, 372, 103)
    canvas.paste(_c3, (54, 410), _c3)
except Exception:
    pass
layout["1_Filter"] = [54, 410, 426, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_10_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-12/04_icon_Fo.png
try:
    _c4 = get_crop(4, 137, 111)
    canvas.paste(_c4, (1295, 406), _c4)
except Exception:
    pass
layout["Fo("] = [1295, 406, 1432, 517]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_10_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-12/05_icon_7.png
try:
    _c5 = get_crop(5, 144, 144)
    canvas.paste(_c5, (1092, 2415), _c5)
except Exception:
    pass
layout["7_`"] = [1092, 2415, 1236, 2559]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_10_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-12/06_icon_Favorite_button.png
try:
    _c6 = get_crop(6, 144, 144)
    canvas.paste(_c6, (1092, 1192), _c6)
except Exception:
    pass
layout["Favorite_button"] = [1092, 1192, 1236, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_10_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-12/07_icon_Overflow_menu_button.png
try:
    _c7 = get_crop(7, 144, 144)
    canvas.paste(_c7, (1236, 1192), _c7)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 1192, 1380, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_10_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-12/08_icon_Minorities_Building_Wealth_with_Franchis.png
try:
    _c8 = get_crop(8, 1344, 1175)
    canvas.paste(_c8, (48, 676), _c8)
except Exception:
    pass
layout["Minorities_Building_Wealt"] = [48, 676, 1392, 1851]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_10_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-12/09_icon_7.png
try:
    _c9 = get_crop(9, 144, 144)
    canvas.paste(_c9, (1236, 2415), _c9)
except Exception:
    pass
layout["7_`"] = [1236, 2415, 1380, 2559]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_10_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-12/10_icon_9.12.png
try:
    _c10 = get_crop(10, 121, 111)
    canvas.paste(_c10, (57, 116), _c10)
except Exception:
    pass
layout["9.12"] = [57, 116, 178, 227]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_10_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-12/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 57, 62)
    canvas.paste(_c11, (246, 1), _c11)
except Exception:
    pass
layout["icon_11"] = [246, 1, 303, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_10_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-12/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 53, 65)
    canvas.paste(_c12, (1151, 0), _c12)
except Exception:
    pass
layout["icon_12"] = [1151, 0, 1204, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_10_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-12/13_icon_Search_forae.png
try:
    _c13 = get_crop(13, 61, 63)
    canvas.paste(_c13, (311, 1), _c13)
except Exception:
    pass
layout["Search_forae"] = [311, 1, 372, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_10_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-12/14_icon_9.12.png
try:
    _c14 = get_crop(14, 55, 62)
    canvas.paste(_c14, (182, 0), _c14)
except Exception:
    pass
layout["9.12"] = [182, 0, 237, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_10_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-12/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 72, 62)
    canvas.paste(_c15, (1212, 0), _c15)
except Exception:
    pass
layout["icon_15"] = [1212, 0, 1284, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_10_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-12/16_icon_icon_16.png
try:
    _c16 = get_crop(16, 59, 59)
    canvas.paste(_c16, (1317, 0), _c16)
except Exception:
    pass
layout["icon_16"] = [1317, 0, 1376, 59]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_10_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-12/17_icon_WEDDING_EXPO.png
try:
    _c17 = get_crop(17, 1344, 917)
    canvas.paste(_c17, (48, 1899), _c17)
except Exception:
    pass
layout["WEDDING_EXPO"] = [48, 1899, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_10_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-12/18_icon_9.12.png
try:
    _c18 = get_crop(18, 56, 64)
    canvas.paste(_c18, (115, 0), _c18)
except Exception:
    pass
layout["9.12"] = [115, 0, 171, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_10_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-12/19_icon_Search_forae.png
try:
    _c19 = get_crop(19, 1344, 191)
    canvas.paste(_c19, (48, 72), _c19)
except Exception:
    pass
layout["Search_forae"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_10_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-12/20_icon_2024.png
try:
    _c20 = get_crop(20, 288, 156)
    canvas.paste(_c20, (288, 2804), _c20)
except Exception:
    pass
layout["2024"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_10_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-12/21_icon_Los_Angeles.png
try:
    _c21 = get_crop(21, 492, 144)
    canvas.paste(_c21, (0, 259), _c21)
except Exception:
    pass
layout["Los_Angeles"] = [0, 259, 492, 403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_10_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-12/22_icon_Search_forae.png
try:
    _c22 = get_crop(22, 50, 61)
    canvas.paste(_c22, (383, 2), _c22)
except Exception:
    pass
layout["Search_forae"] = [383, 2, 433, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_10_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-12/23_icon_Promoted.png
try:
    _c23 = get_crop(23, 241, 64)
    canvas.paste(_c23, (85, 1744), _c23)
except Exception:
    pass
layout["Promoted"] = [85, 1744, 326, 1808]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_10_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-12/24_icon_Reptacular_Ranch_Wedding_Expo_April_Zth.png
try:
    _c24 = get_crop(24, 288, 156)
    canvas.paste(_c24, (576, 2804), _c24)
except Exception:
    pass
layout["Reptacular_Ranch_Wedding_"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_10_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-12/25_icon_Reptacular_Ranch_Wedding_Expo_April_Zth.png
try:
    _c25 = get_crop(25, 288, 156)
    canvas.paste(_c25, (864, 2804), _c25)
except Exception:
    pass
layout["Reptacular_Ranch_Wedding_"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_10_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-12/26_icon_7.png
try:
    _c26 = get_crop(26, 288, 156)
    canvas.paste(_c26, (1152, 2804), _c26)
except Exception:
    pass
layout["7_`"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_10_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-12/27_icon_2024.png
try:
    _c27 = get_crop(27, 288, 156)
    canvas.paste(_c27, (0, 2804), _c27)
except Exception:
    pass
layout["2024"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_10_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-12/28_icon_icon_28.png
try:
    _c28 = get_crop(28, 41, 61)
    canvas.paste(_c28, (1273, 0), _c28)
except Exception:
    pass
layout["icon_28"] = [1273, 0, 1314, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_10_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-12/29_text_9.12.png
try:
    _c29 = get_crop(29, 91, 43)
    canvas.paste(_c29, (20, 17), _c29)
except Exception:
    pass
layout["9.12"] = [20, 17, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_10_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-12/30_text_9_828_events.png
try:
    _c30 = get_crop(30, 372, 103)
    canvas.paste(_c30, (54, 410), _c30)
except Exception:
    pass
layout["9,828_events"] = [54, 410, 426, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_10_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-12/31_text_Los_Angeles_Convention_Center.png
try:
    _c31 = get_crop(31, 587, 60)
    canvas.paste(_c31, (92, 1684), _c31)
except Exception:
    pass
layout["Los_Angeles_Convention_Ce"] = [92, 1684, 679, 1744]
