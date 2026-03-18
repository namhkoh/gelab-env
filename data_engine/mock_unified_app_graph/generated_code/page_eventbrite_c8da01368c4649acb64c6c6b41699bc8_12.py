# page_id: page_eventbrite_c8da01368c4649acb64c6c6b41699bc8_12
# screenshot: 2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-14.png
# step_index: 12/13
# task: Open Eventbrite. Look up "Animal" events. Filter by events happening next week. Select the first event - who is the organizer?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Paint overall background (slightly warm off-white like the app background)
draw.rectangle([0, 0, 1440, 2960], fill="#FBFBFC")

# Top status bar area (approx ~96px tall)
status_h = 96
draw.rectangle([0, 0, 1440, status_h], fill="#BDBDBD")  # muted gray status bar

# Subtle divider under status bar (a hairline)
draw.line([(24, status_h), (1416, status_h)], fill="#E0E0E0", width=1)

# Header / toolbar area (white background beneath status bar)
header_top = status_h
header_bottom = 220
draw.rectangle([0, header_top, 1440, header_bottom], fill="#FFFFFF")

# Thin divider under the header
draw.line([(24, header_bottom), (1416, header_bottom)], fill="#E6E6E6", width=1)

# Content area main background (slightly different tint to separate from header)
content_top = header_bottom + 16
draw.rectangle([0, content_top, 1440, 2960], fill="#FBFBFC")

# First event card container (rounded rectangle) - acts as background for first event group
card_margin_x = 48
card1_top = 320
card1_bottom = 1240
card1_coords = [card_margin_x, card1_top, 1440 - card_margin_x, card1_bottom]
draw.rounded_rectangle(card1_coords, radius=20, fill="#FFFFFF", outline="#E8E8E8", width=1)

# Light inner background band to suggest the image area within the first card (no content drawn)
# Keep it subtle and not a duplicate of detected image content: fill very light cream
image_band_top = card1_top + 24
image_band_bottom = image_band_top + 420
draw.rectangle([card_margin_x + 12, image_band_top, 1440 - card_margin_x - 12, image_band_bottom], fill="#FFF9F4")

# Divider line separating image area from details in first card
draw.line([(card_margin_x + 12, image_band_bottom + 18), (1440 - card_margin_x - 12, image_band_bottom + 18)], fill="#F0F0F0", width=1)

# Second event card container further down the list
card2_top = 1460
card2_bottom = 2380
card2_coords = [card_margin_x, card2_top, 1440 - card_margin_x, card2_bottom]
draw.rounded_rectangle(card2_coords, radius=20, fill="#FFFFFF", outline="#E8E8E8", width=1)

# Light banner background inside second card (placeholder behind large image area)
sec_image_top = card2_top + 24
sec_image_bottom = sec_image_top + 560
draw.rectangle([card_margin_x + 12, sec_image_top, 1440 - card_margin_x - 12, sec_image_bottom], fill="#FFFFFF")

# Subtle separators between list items (outside card containers)
sep_y = card1_bottom + 24
draw.line([(24, sep_y), (1416, sep_y)], fill="#F4F4F6", width=1)
sep2_y = card2_bottom + 24
draw.line([(24, sep2_y), (1416, sep2_y)], fill="#F4F4F6", width=1)

# Bottom navigation bar area
nav_h = 120
nav_top = 2960 - nav_h
draw.rectangle([0, nav_top, 1440, 2960], fill="#FFFFFF")
# top border of nav
draw.line([(0, nav_top), (1440, nav_top)], fill="#E9E9EB", width=1)

# Small floating dividers / visual guides for list spacing (do not draw any icons or text)
# gentle left gutter guide
draw.line([(card_margin_x, content_top + 8), (card_margin_x, sep2_y)], fill="#FAFAFA", width=1)
# gentle right gutter guide
draw.line([(1440 - card_margin_x, content_top + 8), (1440 - card_margin_x, sep2_y)], fill="#FAFAFA", width=1)

# Subtle accent band at the very top of the content area (below header) to visually anchor filters
accent_top = header_bottom + 6
accent_bottom = accent_top + 8
draw.rectangle([24, accent_top, 1416, accent_bottom], fill="#F5F7FF")

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_12_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-14/00_icon_Music.png
try:
    _c0 = get_crop(0, 187, 103)
    canvas.paste(_c0, (1111, 410), _c0)
except Exception:
    pass
layout["Music"] = [1111, 410, 1298, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_12_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-14/01_icon_Apr_28_-_May_04_2024.png
try:
    _c1 = get_crop(1, 661, 103)
    canvas.paste(_c1, (438, 410), _c1)
except Exception:
    pass
layout["Apr_28_-_May_04,_2024"] = [438, 410, 1099, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_12_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-14/02_icon_1_Filter.png
try:
    _c2 = get_crop(2, 372, 103)
    canvas.paste(_c2, (54, 410), _c2)
except Exception:
    pass
layout["1_Filter"] = [54, 410, 426, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_12_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-14/03_icon_Laultutn.png
try:
    _c3 = get_crop(3, 144, 144)
    canvas.paste(_c3, (1092, 1192), _c3)
except Exception:
    pass
layout["Laultutn"] = [1092, 1192, 1236, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_12_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-14/04_icon_Overflow_menu_button.png
try:
    _c4 = get_crop(4, 144, 144)
    canvas.paste(_c4, (1236, 1192), _c4)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 1192, 1380, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_12_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-14/05_icon_Close_current_screen.png
try:
    _c5 = get_crop(5, 144, 144)
    canvas.paste(_c5, (1248, 96), _c5)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 96, 1392, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_12_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-14/06_icon_Nnnual.png
try:
    _c6 = get_crop(6, 1344, 1108)
    canvas.paste(_c6, (48, 676), _c6)
except Exception:
    pass
layout["Nnnual"] = [48, 676, 1392, 1784]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_12_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-14/07_icon_Bu.png
try:
    _c7 = get_crop(7, 118, 111)
    canvas.paste(_c7, (1305, 406), _c7)
except Exception:
    pass
layout["Bu"] = [1305, 406, 1423, 517]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_12_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-14/08_icon_Animal.png
try:
    _c8 = get_crop(8, 61, 62)
    canvas.paste(_c8, (311, 1), _c8)
except Exception:
    pass
layout["Animal"] = [311, 1, 372, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_12_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-14/09_icon_5.15.png
try:
    _c9 = get_crop(9, 55, 63)
    canvas.paste(_c9, (182, 1), _c9)
except Exception:
    pass
layout["5.15"] = [182, 1, 237, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_12_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-14/10_icon_5.15.png
try:
    _c10 = get_crop(10, 55, 64)
    canvas.paste(_c10, (117, 1), _c10)
except Exception:
    pass
layout["5.15"] = [117, 1, 172, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_12_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-14/11_icon_5.15.png
try:
    _c11 = get_crop(11, 117, 114)
    canvas.paste(_c11, (57, 114), _c11)
except Exception:
    pass
layout["5.15"] = [57, 114, 174, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_12_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-14/12_icon_Animal.png
try:
    _c12 = get_crop(12, 48, 62)
    canvas.paste(_c12, (250, 1), _c12)
except Exception:
    pass
layout["Animal"] = [250, 1, 298, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_12_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-14/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 99, 62)
    canvas.paste(_c13, (1210, 0), _c13)
except Exception:
    pass
layout["icon_13"] = [1210, 0, 1309, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_12_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-14/14_icon_CATS_v.png
try:
    _c14 = get_crop(14, 1344, 984)
    canvas.paste(_c14, (48, 1832), _c14)
except Exception:
    pass
layout["CATS_v"] = [48, 1832, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_12_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-14/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 54, 63)
    canvas.paste(_c15, (1318, 0), _c15)
except Exception:
    pass
layout["icon_15"] = [1318, 0, 1372, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_12_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-14/16_icon_icon_16.png
try:
    _c16 = get_crop(16, 46, 60)
    canvas.paste(_c16, (385, 3), _c16)
except Exception:
    pass
layout["icon_16"] = [385, 3, 431, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_12_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-14/17_icon_Animal.png
try:
    _c17 = get_crop(17, 1344, 191)
    canvas.paste(_c17, (48, 72), _c17)
except Exception:
    pass
layout["Animal"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_12_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-14/18_icon_Chicago.png
try:
    _c18 = get_crop(18, 417, 144)
    canvas.paste(_c18, (0, 259), _c18)
except Exception:
    pass
layout["Chicago"] = [0, 259, 417, 403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_12_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-14/19_icon_Mau_11.0_AM_CDt.png
try:
    _c19 = get_crop(19, 288, 156)
    canvas.paste(_c19, (288, 2804), _c19)
except Exception:
    pass
layout["Mau^_,_11.0_AM_CDt"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_12_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-14/20_icon_C.A.R.E._First_Saturday_Adoption_Event_a.png
try:
    _c20 = get_crop(20, 288, 156)
    canvas.paste(_c20, (576, 2804), _c20)
except Exception:
    pass
layout["C.A.R.E._First_Saturday_A"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_12_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-14/21_icon_Cat.png
try:
    _c21 = get_crop(21, 288, 156)
    canvas.paste(_c21, (0, 2804), _c21)
except Exception:
    pass
layout["Cat"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_12_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-14/22_icon_C.A.R.E._First_Saturday_Adoption_Event_a.png
try:
    _c22 = get_crop(22, 288, 156)
    canvas.paste(_c22, (864, 2804), _c22)
except Exception:
    pass
layout["C.A.R.E._First_Saturday_A"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_12_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-14/23_icon_Free.png
try:
    _c23 = get_crop(23, 128, 78)
    canvas.paste(_c23, (91, 2524), _c23)
except Exception:
    pass
layout["Free"] = [91, 2524, 219, 2602]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_12_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-14/24_icon_Tickets.png
try:
    _c24 = get_crop(24, 288, 156)
    canvas.paste(_c24, (864, 2804), _c24)
except Exception:
    pass
layout["Tickets"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_12_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-14/25_icon_More.png
try:
    _c25 = get_crop(25, 288, 156)
    canvas.paste(_c25, (1152, 2804), _c25)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_12_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-14/26_text_5.15.png
try:
    _c26 = get_crop(26, 92, 43)
    canvas.paste(_c26, (22, 17), _c26)
except Exception:
    pass
layout["5.15"] = [22, 17, 114, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_12_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-14/27_text_22_events.png
try:
    _c27 = get_crop(27, 372, 103)
    canvas.paste(_c27, (54, 410), _c27)
except Exception:
    pass
layout["22_events"] = [54, 410, 426, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_12_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-14/28_text_SuNDAT_ApRLIBTAN_JeM.png
try:
    _c28 = get_crop(28, 372, 103)
    canvas.paste(_c28, (54, 410), _c28)
except Exception:
    pass
layout["SuNDAT_ApRLIBTAN_JeM"] = [54, 410, 426, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_12_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-14/29_text_Animu_Cuze_849uo.png
try:
    _c29 = get_crop(29, 305, 40)
    canvas.paste(_c29, (535, 777), _c29)
except Exception:
    pass
layout["Animu_|_Cuze_[849uo'"] = [535, 777, 840, 817]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_12_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-14/30_text_The_Oak_Park_Country_Club.png
try:
    _c30 = get_crop(30, 503, 55)
    canvas.paste(_c30, (90, 1686), _c30)
except Exception:
    pass
layout["The_Oak_Park_Country_Club"] = [90, 1686, 593, 1741]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_12_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-14/31_text_Mau_11.0_AM_CDt.png
try:
    _c31 = get_crop(31, 288, 156)
    canvas.paste(_c31, (288, 2804), _c31)
except Exception:
    pass
layout["Mau^_,_11.0_AM_CDt"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_12_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-14/32_text_Cat.png
try:
    _c32 = get_crop(32, 71, 38)
    canvas.paste(_c32, (92, 2774), _c32)
except Exception:
    pass
layout["Cat"] = [92, 2774, 163, 2812]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_12_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-14/33_clickable_Favorite_button.png
try:
    _c33 = get_crop(33, 144, 144)
    canvas.paste(_c33, (1092, 2348), _c33)
except Exception:
    pass
layout["Favorite_button"] = [1092, 2348, 1236, 2492]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_12_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-14/34_clickable_Overflow_menu_button.png
try:
    _c34 = get_crop(34, 144, 144)
    canvas.paste(_c34, (1236, 2348), _c34)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 2348, 1380, 2492]
