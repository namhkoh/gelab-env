# page_id: page_eventbrite_1a166da440f24e2e9152f2c0e40eb7aa_14
# screenshot: 2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-16.png
# step_index: 14/16
# task: Open Eventbrite. Check "Sports" category. Filter events happening next month. Add the first event to your wishlist.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw overall background
draw.rectangle((0, 0, 1440, 2960), fill="#f6f7fb")

# Status bar area (top ~72px)
status_h = 72
draw.rectangle((0, 0, 1440, status_h), fill="#cfcfcf")
# subtle bottom divider of status bar
draw.line((0, status_h-1, 1440, status_h-1), fill="#bdbdbd", width=1)

# Search/header area background (behind the search field)
search_x0, search_y0 = 48, 72
search_w, search_h = 1344, 191
search_x1, search_y1 = search_x0 + search_w, search_y0 + search_h
draw.rounded_rectangle((search_x0, search_y0, search_x1, search_y1),
                       radius=20, fill="#ffffff", outline=None)

# Thin divider below header/search
divider_y = search_y1 + 10
draw.line((48, divider_y, 1392, divider_y), fill="#e6e8eb", width=2)

# Light background band where filter chips sit (do not draw chips themselves)
filters_band_y0 = divider_y + 18
filters_band_y1 = filters_band_y0 + 120
draw.rectangle((48, filters_band_y0, 1392, filters_band_y1), fill="#f6f7fb")

# Thin subtle separator under filter area
draw.line((48, filters_band_y1 + 6, 1392, filters_band_y1 + 6), fill="#eceff2", width=1)

# First event card background with shadow
card1_x0, card1_y0 = 48, 676
card1_w, card1_h = 1344, 1012
card1_x1, card1_y1 = card1_x0 + card1_w, card1_y0 + card1_h

# shadow (offset)
shadow_offset = 10
draw.rounded_rectangle((card1_x0 + shadow_offset, card1_y0 + shadow_offset,
                        card1_x1 + shadow_offset, card1_y1 + shadow_offset),
                       radius=28, fill="#e9ebee")

# card surface
draw.rounded_rectangle((card1_x0, card1_y0, card1_x1, card1_y1),
                       radius=28, fill="#ffffff", outline="#f0f2f4", width=1)

# image placeholder region inside first card (top area)
img_pad = 24
img_top_h = 360
draw.rounded_rectangle((card1_x0 + img_pad, card1_y0 + img_pad,
                        card1_x1 - img_pad, card1_y0 + img_pad + img_top_h),
                       radius=18, fill="#dfe6ea")

# subtle divider below image area inside card
draw.line((card1_x0 + img_pad, card1_y0 + img_pad + img_top_h + 12,
           card1_x1 - img_pad, card1_y0 + img_pad + img_top_h + 12),
          fill="#f1f3f5", width=1)

# Second event card background with shadow
card2_x0, card2_y0 = 48, 1736
card2_w, card2_h = 1344, 1024
card2_x1, card2_y1 = card2_x0 + card2_w, card2_y0 + card2_h

# shadow
draw.rounded_rectangle((card2_x0 + shadow_offset, card2_y0 + shadow_offset,
                        card2_x1 + shadow_offset, card2_y1 + shadow_offset),
                       radius=28, fill="#e9ebee")

# card surface
draw.rounded_rectangle((card2_x0, card2_y0, card2_x1, card2_y1),
                       radius=28, fill="#ffffff", outline="#f0f2f4", width=1)

# second card image placeholder (smaller banner area)
img2_pad = 22
img2_h = 240
draw.rounded_rectangle((card2_x0 + img2_pad, card2_y0 + img2_pad,
                        card2_x1 - img2_pad, card2_y0 + img2_pad + img2_h),
                       radius=14, fill="#f3f6f8")

# subtle content separators between cards and content areas
sep_x0 = 48
sep_x1 = 1392
draw.line((sep_x0, card1_y1 + 20, sep_x1, card1_y1 + 20), fill="#f2f4f6", width=1)
draw.line((sep_x0, card2_y1 + 8, sep_x1, card2_y1 + 8), fill="#f2f4f6", width=1)

# Bottom navigation bar background (do not draw icons)
nav_y0 = 2804
nav_h = 156
draw.rectangle((0, nav_y0, 1440, nav_y0 + nav_h), fill="#ffffff")
# top border of nav
draw.line((0, nav_y0, 1440, nav_y0), fill="#e6e8eb", width=2)

# small top shadow under content area for separation from nav
draw.line((0, nav_y0 - 8, 1440, nav_y0 - 8), fill="#f0f2f4", width=1)

# add a faint left margin guide (visual only, very faint)
draw.line((48, status_h + 8, 48, nav_y0 - 8), fill="#fbfdfe", width=2)
draw.line((1392, status_h + 8, 1392, nav_y0 - 8), fill="#fbfdfe", width=2)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_14_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-16/00_icon_31_2024.png
try:
    _c0 = get_crop(0, 584, 103)
    canvas.paste(_c0, (458, 410), _c0)
except Exception:
    pass
layout["31,_2024"] = [458, 410, 1042, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_14_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-16/01_icon_2_Filters.png
try:
    _c1 = get_crop(1, 392, 103)
    canvas.paste(_c1, (54, 410), _c1)
except Exception:
    pass
layout["2_Filters"] = [54, 410, 446, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_14_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-16/02_icon_Sports_Fitness.png
try:
    _c2 = get_crop(2, 338, 103)
    canvas.paste(_c2, (1054, 410), _c2)
except Exception:
    pass
layout["Sports_&_Fitness"] = [1054, 410, 1392, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_14_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-16/03_icon_Favorite_button.png
try:
    _c3 = get_crop(3, 144, 144)
    canvas.paste(_c3, (1092, 1192), _c3)
except Exception:
    pass
layout["Favorite_button"] = [1092, 1192, 1236, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_14_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-16/04_icon_Overflow_menu_button.png
try:
    _c4 = get_crop(4, 144, 144)
    canvas.paste(_c4, (1236, 1192), _c4)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 1192, 1380, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_14_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-16/05_icon_sportirela_D.png
try:
    _c5 = get_crop(5, 144, 144)
    canvas.paste(_c5, (1092, 2252), _c5)
except Exception:
    pass
layout["sportirela'D"] = [1092, 2252, 1236, 2396]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_14_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-16/06_icon_5.31.png
try:
    _c6 = get_crop(6, 61, 67)
    canvas.paste(_c6, (113, 0), _c6)
except Exception:
    pass
layout["5.31"] = [113, 0, 174, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_14_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-16/07_icon_Search_forae.png
try:
    _c7 = get_crop(7, 66, 63)
    canvas.paste(_c7, (308, 1), _c7)
except Exception:
    pass
layout["Search_forae"] = [308, 1, 374, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_14_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-16/08_icon_5.31.png
try:
    _c8 = get_crop(8, 117, 113)
    canvas.paste(_c8, (59, 114), _c8)
except Exception:
    pass
layout["5.31"] = [59, 114, 176, 227]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_14_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-16/09_icon_5.31.png
try:
    _c9 = get_crop(9, 60, 65)
    canvas.paste(_c9, (180, 0), _c9)
except Exception:
    pass
layout["5.31"] = [180, 0, 240, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_14_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-16/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 52, 64)
    canvas.paste(_c10, (247, 1), _c10)
except Exception:
    pass
layout["icon_10"] = [247, 1, 299, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_14_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-16/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 56, 62)
    canvas.paste(_c11, (1317, 0), _c11)
except Exception:
    pass
layout["icon_11"] = [1317, 0, 1373, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_14_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-16/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 63, 62)
    canvas.paste(_c12, (1211, 0), _c12)
except Exception:
    pass
layout["icon_12"] = [1211, 0, 1274, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_14_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-16/13_icon_Basics_of_Roller_Skating_balance_power.png
try:
    _c13 = get_crop(13, 1344, 1012)
    canvas.paste(_c13, (48, 676), _c13)
except Exception:
    pass
layout["Basics_of_Roller_Skating_"] = [48, 676, 1392, 1688]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_14_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-16/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 41, 62)
    canvas.paste(_c14, (1272, 0), _c14)
except Exception:
    pass
layout["icon_14"] = [1272, 0, 1313, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_14_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-16/15_icon_14._1_30_PM_EDT.png
try:
    _c15 = get_crop(15, 288, 156)
    canvas.paste(_c15, (576, 2804), _c15)
except Exception:
    pass
layout["14._1:30_PM_EDT"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_14_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-16/16_icon_Search_forae.png
try:
    _c16 = get_crop(16, 47, 64)
    canvas.paste(_c16, (384, 1), _c16)
except Exception:
    pass
layout["Search_forae"] = [384, 1, 431, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_14_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-16/17_icon_Overflow_menu_button.png
try:
    _c17 = get_crop(17, 144, 144)
    canvas.paste(_c17, (1236, 2252), _c17)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 2252, 1380, 2396]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_14_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-16/18_icon_Online.png
try:
    _c18 = get_crop(18, 377, 144)
    canvas.paste(_c18, (0, 259), _c18)
except Exception:
    pass
layout["Online"] = [0, 259, 377, 403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_14_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-16/19_icon_Search_forae.png
try:
    _c19 = get_crop(19, 1344, 191)
    canvas.paste(_c19, (48, 72), _c19)
except Exception:
    pass
layout["Search_forae"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_14_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-16/20_icon_14._1_30_PM_EDT.png
try:
    _c20 = get_crop(20, 288, 156)
    canvas.paste(_c20, (288, 2804), _c20)
except Exception:
    pass
layout["14._1:30_PM_EDT"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_14_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-16/21_icon_5.31.png
try:
    _c21 = get_crop(21, 94, 64)
    canvas.paste(_c21, (12, 1), _c21)
except Exception:
    pass
layout["5.31"] = [12, 1, 106, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_14_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-16/22_icon_Promoted.png
try:
    _c22 = get_crop(22, 38, 54)
    canvas.paste(_c22, (286, 1587), _c22)
except Exception:
    pass
layout["Promoted_@"] = [286, 1587, 324, 1641]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_14_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-16/23_icon_Tickets.png
try:
    _c23 = get_crop(23, 288, 156)
    canvas.paste(_c23, (864, 2804), _c23)
except Exception:
    pass
layout["Tickets"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_14_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-16/24_icon_Galway_Sports_Partnership_s_Online.png
try:
    _c24 = get_crop(24, 288, 156)
    canvas.paste(_c24, (864, 2804), _c24)
except Exception:
    pass
layout["Galway_Sports_Partnership"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_14_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-16/25_text_382_events.png
try:
    _c25 = get_crop(25, 392, 103)
    canvas.paste(_c25, (54, 410), _c25)
except Exception:
    pass
layout["382_events"] = [54, 410, 446, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_14_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-16/26_text_0uus.png
try:
    _c26 = get_crop(26, 48, 18)
    canvas.paste(_c26, (1146, 1779), _c26)
except Exception:
    pass
layout["0uus"] = [1146, 1779, 1194, 1797]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_14_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-16/27_text_Safeguarding_1.png
try:
    _c27 = get_crop(27, 1344, 1024)
    canvas.paste(_c27, (48, 1736), _c27)
except Exception:
    pass
layout["Safeguarding_1"] = [48, 1736, 1392, 2760]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_14_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-16/28_text_SPORT.png
try:
    _c28 = get_crop(28, 283, 90)
    canvas.paste(_c28, (501, 1970), _c28)
except Exception:
    pass
layout["SPORT"] = [501, 1970, 784, 2060]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_14_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-16/29_text_Cht.png
try:
    _c29 = get_crop(29, 37, 19)
    canvas.paste(_c29, (1136, 2017), _c29)
except Exception:
    pass
layout["Cht"] = [1136, 2017, 1173, 2036]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_14_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-16/30_text_Scrt_ORdt.png
try:
    _c30 = get_crop(30, 161, 27)
    canvas.paste(_c30, (1101, 2070), _c30)
except Exception:
    pass
layout["Scrt_ORdt"] = [1101, 2070, 1262, 2097]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_14_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-16/31_text_IRELAND.png
try:
    _c31 = get_crop(31, 1344, 1024)
    canvas.paste(_c31, (48, 1736), _c31)
except Exception:
    pass
layout["IRELAND"] = [48, 1736, 1392, 2760]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_14_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-16/32_text_sportirela_D.png
try:
    _c32 = get_crop(32, 166, 27)
    canvas.paste(_c32, (1098, 2097), _c32)
except Exception:
    pass
layout["sportirela'D"] = [1098, 2097, 1264, 2124]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_14_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-16/33_text_ETHICS.png
try:
    _c33 = get_crop(33, 162, 57)
    canvas.paste(_c33, (508, 2153), _c33)
except Exception:
    pass
layout["ETHICS"] = [508, 2153, 670, 2210]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_14_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-16/34_text_Galway_Sports_Partnership_s_Online.png
try:
    _c34 = get_crop(34, 1344, 1024)
    canvas.paste(_c34, (48, 1736), _c34)
except Exception:
    pass
layout["Galway_Sports_Partnership"] = [48, 1736, 1392, 2760]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_14_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-16/35_text_Tue.png
try:
    _c35 = get_crop(35, 94, 53)
    canvas.paste(_c35, (90, 2595), _c35)
except Exception:
    pass
layout["Tue,"] = [90, 2595, 184, 2648]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_14_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-16/36_text_14._1_30_PM_EDT.png
try:
    _c36 = get_crop(36, 331, 50)
    canvas.paste(_c36, (271, 2594), _c36)
except Exception:
    pass
layout["14._1:30_PM_EDT"] = [271, 2594, 602, 2644]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_14_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-16/37_text_Online.png
try:
    _c37 = get_crop(37, 126, 43)
    canvas.paste(_c37, (94, 2665), _c37)
except Exception:
    pass
layout["Online"] = [94, 2665, 220, 2708]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_14_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-16/38_clickable_Home.png
try:
    _c38 = get_crop(38, 288, 156)
    canvas.paste(_c38, (0, 2804), _c38)
except Exception:
    pass
layout["Home"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_14_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-16/39_clickable_More.png
try:
    _c39 = get_crop(39, 288, 156)
    canvas.paste(_c39, (1152, 2804), _c39)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]
