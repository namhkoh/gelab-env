# page_id: page_eventbrite_997cbfb071cf458f82ccaba5bddfc7b8_04
# screenshot: 2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-6.png
# step_index: 4/15
# task: Open Eventbrite. Search Online free events. Select the first one. Follow the organizer. Read more about the event. Add it to Favorites.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Paint overall background
draw.rectangle([0, 0, 1440, 2960], fill=(247, 249, 251))  # very light off-white/blue tint

# Status bar (top)
status_h =  Fifty = 50
status_h = 50
draw.rectangle([0, 0, 1440, status_h], fill=(189, 189, 189))  # muted gray status bar

# Top header/search area background
header_top = status_h
header_bottom = 180
draw.rectangle([0, header_top, 1440, header_bottom], fill=(255, 255, 255))  # white search area

# Subtle divider under search area
draw.line([(48, header_bottom), (1440-48, header_bottom)], fill=(220, 220, 225), width=2)

# Thin separator below filters row (approx where chips area sits)
filters_sep_y = 460
draw.line([(48, filters_sep_y), (1440-48, filters_sep_y)], fill=(230, 230, 235), width=1)

# Shadow/backdrop for first content card (slight offset darker rectangle)
card1_x1, card1_y1 = 48, 676
card1_w, card1_h = 1344, 1175
card1_x2, card1_y2 = card1_x1 + card1_w, card1_y1 + card1_h
# faux shadow
draw.rectangle([card1_x1+6, card1_y1+8, card1_x2+6, card1_y2+8], fill=(236, 238, 241))
# card background (rounded)
draw.rounded_rectangle([card1_x1, card1_y1, card1_x2, card1_y2],
                       radius=28, fill=(255, 255, 255), outline=(220, 223, 227), width=1)

# Image/banner background area portion inside first card (top band)
banner_h = 360
draw.rounded_rectangle([card1_x1+12, card1_y1+12, card1_x2-12, card1_y1+12+banner_h],
                       radius=18, fill=(27, 27, 27), outline=None)

# Separator line between banner and content area in first card
draw.line([(card1_x1+24, card1_y1+12+banner_h+8), (card1_x2-24, card1_y1+12+banner_h+8)],
          fill=(238, 238, 240), width=1)

# Small tag background area (e.g., where "Free" pill sits) - draw淡 background rectangle but DO NOT draw text/pill
# Place it below banner on left
draw.rounded_rectangle([card1_x1+24, card1_y1+12+banner_h+24, card1_x1+140, card1_y1+12+banner_h+24+44],
                       radius=12, fill=(236, 245, 241), outline=None)

# Shadow/backdrop for second content card
card2_x1, card2_y1 = 48, 1899
card2_w, card2_h = 1344, 917
card2_x2, card2_y2 = card2_x1 + card2_w, card2_y1 + card2_h
draw.rectangle([card2_x1+6, card2_y1+8, card2_x2+6, card2_y2+8], fill=(236, 238, 241))
draw.rounded_rectangle([card2_x1, card2_y1, card2_x2, card2_y2],
                       radius=28, fill=(255, 255, 255), outline=(220, 223, 227), width=1)

# Large image area inside second card (top portion)
banner2_h = 360
draw.rounded_rectangle([card2_x1+12, card2_y1+12, card2_x2-12, card2_y1+12+banner2_h],
                       radius=18, fill=(60, 68, 78), outline=None)

# "Free" pill background area for second card (do not draw the label)
draw.rounded_rectangle([card2_x1+24, card2_y1+12+banner2_h+24, card2_x1+160, card2_y1+12+banner2_h+24+48],
                       radius=12, fill=(236, 245, 241), outline=None)

# Divider between cards and subsequent list area
draw.line([(48, card2_y2 + 8), (1440-48, card2_y2 + 8)], fill=(225, 225, 230), width=2)

# Bottom navigation bar background
nav_h = 120
nav_top = 2960 - nav_h
draw.rectangle([0, nav_top, 1440, 2960], fill=(255, 255, 255))
# top hairline for nav
draw.line([(0, nav_top), (1440, nav_top)], fill=(220, 220, 225), width=2)

# Small indicator row above nav (subtle)
indicator_y = nav_top - 28
draw.line([(48, indicator_y), (1440-48, indicator_y)], fill=(245, 245, 247), width=1)

# Section header background for the "10,000 events" heading row area (leave text to be pasted)
heading_area_y1 = 336
heading_area_y2 = 420
draw.rectangle([48, heading_area_y1, 1440-48, heading_area_y2], fill=(247, 249, 251))
draw.line([(48, heading_area_y2), (1440-48, heading_area_y2)], fill=(235, 235, 238), width=1)

# Additional subtle vertical guides/padding markers (not UI elements, but help structure)
# left content guide
draw.line([(48, header_bottom+8), (48, 2800)], fill=(250, 251, 252), width=1)
# right content guide
draw.line([(1440-48, header_bottom+8), (1440-48, 2800)], fill=(250, 251, 252), width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_04_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-6/00_icon_Music.png
try:
    _c0 = get_crop(0, 187, 103)
    canvas.paste(_c0, (837, 410), _c0)
except Exception:
    pass
layout["Music"] = [837, 410, 1024, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_04_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-6/01_icon_Business.png
try:
    _c1 = get_crop(1, 241, 103)
    canvas.paste(_c1, (1036, 410), _c1)
except Exception:
    pass
layout["Business"] = [1036, 410, 1277, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_04_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-6/02_icon_Anytime.png
try:
    _c2 = get_crop(2, 400, 103)
    canvas.paste(_c2, (425, 410), _c2)
except Exception:
    pass
layout["Anytime"] = [425, 410, 825, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_04_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-6/03_icon_Filters.png
try:
    _c3 = get_crop(3, 359, 103)
    canvas.paste(_c3, (54, 410), _c3)
except Exception:
    pass
layout["Filters"] = [54, 410, 413, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_04_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-6/04_icon_Foo.png
try:
    _c4 = get_crop(4, 138, 110)
    canvas.paste(_c4, (1284, 406), _c4)
except Exception:
    pass
layout["Foo"] = [1284, 406, 1422, 516]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_04_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-6/05_icon_EcoMmcR.png
try:
    _c5 = get_crop(5, 144, 144)
    canvas.paste(_c5, (1092, 1192), _c5)
except Exception:
    pass
layout["EcoMmcR"] = [1092, 1192, 1236, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_04_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-6/06_icon_EcoMmcR.png
try:
    _c6 = get_crop(6, 144, 144)
    canvas.paste(_c6, (1236, 1192), _c6)
except Exception:
    pass
layout["EcoMmcR"] = [1236, 1192, 1380, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_04_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-6/07_icon_9.15.png
try:
    _c7 = get_crop(7, 129, 118)
    canvas.paste(_c7, (53, 112), _c7)
except Exception:
    pass
layout["9.15"] = [53, 112, 182, 230]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_04_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-6/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 57, 61)
    canvas.paste(_c8, (246, 1), _c8)
except Exception:
    pass
layout["icon_8"] = [246, 1, 303, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_04_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-6/09_icon_Online.png
try:
    _c9 = get_crop(9, 377, 144)
    canvas.paste(_c9, (0, 259), _c9)
except Exception:
    pass
layout["Online"] = [0, 259, 377, 403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_04_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-6/10_icon_9.15.png
try:
    _c10 = get_crop(10, 55, 62)
    canvas.paste(_c10, (182, 0), _c10)
except Exception:
    pass
layout["9.15"] = [182, 0, 237, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_04_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-6/11_icon_Overflow_menu_button.png
try:
    _c11 = get_crop(11, 144, 144)
    canvas.paste(_c11, (1236, 2415), _c11)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 2415, 1380, 2559]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_04_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-6/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 84, 60)
    canvas.paste(_c12, (1209, 0), _c12)
except Exception:
    pass
layout["icon_12"] = [1209, 0, 1293, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_04_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-6/13_icon_Search_forae.png
try:
    _c13 = get_crop(13, 61, 62)
    canvas.paste(_c13, (311, 1), _c13)
except Exception:
    pass
layout["Search_forae"] = [311, 1, 372, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_04_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-6/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 61, 60)
    canvas.paste(_c14, (1316, 0), _c14)
except Exception:
    pass
layout["icon_14"] = [1316, 0, 1377, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_04_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-6/15_icon_Search_forae.png
try:
    _c15 = get_crop(15, 1344, 191)
    canvas.paste(_c15, (48, 72), _c15)
except Exception:
    pass
layout["Search_forae"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_04_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-6/16_icon_EcoMMERCE_TracK.png
try:
    _c16 = get_crop(16, 1344, 1175)
    canvas.paste(_c16, (48, 676), _c16)
except Exception:
    pass
layout["EcoMMERCE_TracK"] = [48, 676, 1392, 1851]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_04_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-6/17_icon_9.15.png
try:
    _c17 = get_crop(17, 55, 64)
    canvas.paste(_c17, (116, 0), _c17)
except Exception:
    pass
layout["9.15"] = [116, 0, 171, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_04_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-6/18_icon_5-17.png
try:
    _c18 = get_crop(18, 288, 156)
    canvas.paste(_c18, (288, 2804), _c18)
except Exception:
    pass
layout["5-17)"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_04_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-6/19_icon_Favorite_button.png
try:
    _c19 = get_crop(19, 144, 144)
    canvas.paste(_c19, (1092, 2415), _c19)
except Exception:
    pass
layout["Favorite_button"] = [1092, 2415, 1236, 2559]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_04_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-6/20_icon_Search_forae.png
try:
    _c20 = get_crop(20, 50, 61)
    canvas.paste(_c20, (383, 2), _c20)
except Exception:
    pass
layout["Search_forae"] = [383, 2, 433, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_04_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-6/21_icon_Introduction_to_Intuition_Process_Kids_a.png
try:
    _c21 = get_crop(21, 288, 156)
    canvas.paste(_c21, (864, 2804), _c21)
except Exception:
    pass
layout["Introduction_to_Intuition"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_04_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-6/22_icon_Introduction_to_Intuition_Process_Kids_a.png
try:
    _c22 = get_crop(22, 288, 156)
    canvas.paste(_c22, (1152, 2804), _c22)
except Exception:
    pass
layout["Introduction_to_Intuition"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_04_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-6/23_icon_Introduction_to_Intuition_Process_Kids_a.png
try:
    _c23 = get_crop(23, 288, 156)
    canvas.paste(_c23, (576, 2804), _c23)
except Exception:
    pass
layout["Introduction_to_Intuition"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_04_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-6/24_icon_icon_24.png
try:
    _c24 = get_crop(24, 39, 60)
    canvas.paste(_c24, (1275, 0), _c24)
except Exception:
    pass
layout["icon_24"] = [1275, 0, 1314, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_04_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-6/25_icon_5-17.png
try:
    _c25 = get_crop(25, 288, 156)
    canvas.paste(_c25, (0, 2804), _c25)
except Exception:
    pass
layout["5-17)"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_04_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-6/26_text_9.15.png
try:
    _c26 = get_crop(26, 94, 43)
    canvas.paste(_c26, (20, 17), _c26)
except Exception:
    pass
layout["9.15"] = [20, 17, 114, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_04_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-6/27_text_10_000_events.png
try:
    _c27 = get_crop(27, 359, 103)
    canvas.paste(_c27, (54, 410), _c27)
except Exception:
    pass
layout["10,000_events"] = [54, 410, 413, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_04_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-6/28_text_MERcE_TrACK_JUNI.png
try:
    _c28 = get_crop(28, 307, 39)
    canvas.paste(_c28, (114, 704), _c28)
except Exception:
    pass
layout["MERcE_TrACK_JUNI"] = [114, 704, 421, 743]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_04_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-6/29_text_Ecommerce_TracK_JUNI.png
try:
    _c29 = get_crop(29, 400, 103)
    canvas.paste(_c29, (425, 410), _c29)
except Exception:
    pass
layout["Ecommerce_TracK_JUNI"] = [425, 410, 825, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_04_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-6/30_text_Ecommercetrack.png
try:
    _c30 = get_crop(30, 263, 30)
    canvas.paste(_c30, (849, 707), _c30)
except Exception:
    pass
layout["Ecommercetrack"] = [849, 707, 1112, 737]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_04_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-6/31_text_JUNI.png
try:
    _c31 = get_crop(31, 122, 39)
    canvas.paste(_c31, (1125, 704), _c31)
except Exception:
    pass
layout["JUNI"] = [1125, 704, 1247, 743]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_04_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-6/32_text_FcO.png
try:
    _c32 = get_crop(32, 64, 30)
    canvas.paste(_c32, (1260, 707), _c32)
except Exception:
    pass
layout["FcO"] = [1260, 707, 1324, 737]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_04_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-6/33_text_EZRA.png
try:
    _c33 = get_crop(33, 270, 101)
    canvas.paste(_c33, (689, 864), _c33)
except Exception:
    pass
layout["EZRA"] = [689, 864, 959, 965]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_04_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-6/34_text_Online.png
try:
    _c34 = get_crop(34, 129, 45)
    canvas.paste(_c34, (91, 1687), _c34)
except Exception:
    pass
layout["Online"] = [91, 1687, 220, 1732]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_04_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-6/35_text_Promoted.png
try:
    _c35 = get_crop(35, 144, 144)
    canvas.paste(_c35, (234, 1704), _c35)
except Exception:
    pass
layout["Promoted"] = [234, 1704, 378, 1848]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_04_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-6/36_text_muuomrocess.png
try:
    _c36 = get_crop(36, 1344, 917)
    canvas.paste(_c36, (48, 1899), _c36)
except Exception:
    pass
layout["muuomrocess"] = [48, 1899, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_04_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-6/37_text_THBART_OF_LivING.png
try:
    _c37 = get_crop(37, 251, 29)
    canvas.paste(_c37, (65, 1961), _c37)
except Exception:
    pass
layout["THBART_OF_LivING"] = [65, 1961, 316, 1990]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_04_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-6/38_text_Unlock_your_child_s_full_potentiall.png
try:
    _c38 = get_crop(38, 1344, 917)
    canvas.paste(_c38, (48, 1899), _c38)
except Exception:
    pass
layout["Unlock_your_child's_full_"] = [48, 1899, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_04_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-6/39_text_Get_the.png
try:
    _c39 = get_crop(39, 299, 76)
    canvas.paste(_c39, (218, 2244), _c39)
except Exception:
    pass
layout["Get_the"] = [218, 2244, 517, 2320]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_04_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-6/40_text_RIGHT_THOUGHT.png
try:
    _c40 = get_crop(40, 1344, 917)
    canvas.paste(_c40, (48, 1899), _c40)
except Exception:
    pass
layout["RIGHT_THOUGHT"] = [48, 1899, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_04_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-6/41_text_at_the.png
try:
    _c41 = get_crop(41, 246, 77)
    canvas.paste(_c41, (243, 2426), _c41)
except Exception:
    pass
layout["at_the"] = [243, 2426, 489, 2503]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_04_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-6/42_text_RICHT_TTME.png
try:
    _c42 = get_crop(42, 288, 156)
    canvas.paste(_c42, (288, 2804), _c42)
except Exception:
    pass
layout["RICHT_TTME"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_04_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-6/43_text_Free.png
try:
    _c43 = get_crop(43, 80, 39)
    canvas.paste(_c43, (117, 2614), _c43)
except Exception:
    pass
layout["Free"] = [117, 2614, 197, 2653]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_04_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-6/44_text_Introduction_to_Intuition_Process_Kids_a.png
try:
    _c44 = get_crop(44, 1344, 917)
    canvas.paste(_c44, (48, 1899), _c44)
except Exception:
    pass
layout["Introduction_to_Intuition"] = [48, 1899, 1392, 2816]
