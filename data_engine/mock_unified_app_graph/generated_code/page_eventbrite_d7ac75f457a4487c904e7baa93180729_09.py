# page_id: page_eventbrite_d7ac75f457a4487c904e7baa93180729_09
# screenshot: 2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-11.png
# step_index: 9/11
# task: Open Eventbrite. Search for 'Cooking' classes. Filter to only show free events that occur in the weekend. Select the first event and proceed to checkout.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Fill overall background with a very light off-white (dominant canvas tone)
draw.rectangle([(0, 0), (1440, 2960)], fill="#F7F9FB")

# Status bar area (top ~96px) - subtle warm gray to match screenshot status bar
draw.rectangle([(0, 0), (1440, 96)], fill="#D0D0D0")

# Header / toolbar area under status bar (search/header band)
header_top = 96
header_bottom = 360
draw.rectangle([(0, header_top), (1440, header_bottom)], fill="#FFFFFF")
# subtle bottom divider under header
draw.line([(48, header_bottom), (1392, header_bottom)], fill="#E6E6E9", width=2)

# Thin accent divider under the filter row area (separates header/filters from content)
filters_div_y = 480
draw.line([(24, filters_div_y), (1416, filters_div_y)], fill="#ECEFF1", width=1)

# First event image placeholder (rounded card background)
first_image_box = (48, 520, 1392, 840)
draw.rounded_rectangle(first_image_box, radius=24, fill="#3F8F66", outline="#2E6C4F", width=2)

# First event details card background (white card under image)
first_details_box = (48, 860, 1392, 1060)
draw.rounded_rectangle(first_details_box, radius=12, fill="#FFFFFF", outline="#ECEFF2", width=1)

# Separator line after first event details
draw.line([(48, 1068), (1392, 1068)], fill="#F0F2F4", width=1)

# Second event image placeholder (rounded card background with soft pink tone)
second_image_box = (48, 1160, 1392, 1560)
draw.rounded_rectangle(second_image_box, radius=24, fill="#FFDCE6", outline="#E6B5C6", width=2)

# Second event details card background
second_details_box = (48, 1580, 1392, 1840)
draw.rounded_rectangle(second_details_box, radius=12, fill="#FFFFFF", outline="#ECEFF2", width=1)

# Separator line after second event details
draw.line([(48, 1848), (1392, 1848)], fill="#F0F2F4", width=1)

# Additional subtle section backgrounds for lower-listed items (stack illusion)
# small alternating band to suggest grouping of other list items
y = 1888
for i in range(3):
    band_top = y + i * 220
    band_bottom = band_top + 180
    draw.rectangle([(48, band_top), (1392, band_bottom)], fill="#FFFFFF" if i % 2 == 0 else "#FBFCFD", outline="#F0F2F4")

# Content area vertical separators (thin)
draw.line([(48, 520), (48, 1848)], fill="#FFFFFF", width=1)   # left boundary accent
draw.line([(1392, 520), (1392, 1848)], fill="#FFFFFF", width=1) # right boundary accent

# Soft shadow-like strokes behind key cards to suggest elevation (subtle structural effect)
shadow_offset = 6
draw.rounded_rectangle(
    (first_image_box[0]+shadow_offset, first_image_box[1]+shadow_offset,
     first_image_box[2]+shadow_offset, first_image_box[3]+shadow_offset),
    radius=24, fill=None, outline="#E8ECEB", width=1
)
draw.rounded_rectangle(
    (second_image_box[0]+shadow_offset, second_image_box[1]+shadow_offset,
     second_image_box[2]+shadow_offset, second_image_box[3]+shadow_offset),
    radius=24, fill=None, outline="#E8ECEB", width=1
)

# Bottom navigation bar background and top divider (do not draw icons)
nav_top = 2820
draw.rectangle([(0, nav_top), (1440, 2960)], fill="#FFFFFF")
draw.line([(24, nav_top), (1416, nav_top)], fill="#E6E9EB", width=2)

# Subtle left gutter vertical guideline to visually separate content from screen edge
draw.line([(24, 360), (24, 2760)], fill="#FAFBFC", width=1)

# Final light right gutter line for balance
draw.line([(1416, 360), (1416, 2760)], fill="#FAFBFC", width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_09_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-11/00_icon_Music.png
try:
    _c0 = get_crop(0, 187, 103)
    canvas.paste(_c0, (974, 410), _c0)
except Exception:
    pass
layout["Music"] = [974, 410, 1161, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_09_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-11/01_icon_This_Weekend.png
try:
    _c1 = get_crop(1, 504, 103)
    canvas.paste(_c1, (458, 410), _c1)
except Exception:
    pass
layout["This_Weekend"] = [458, 410, 962, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_09_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-11/02_icon_Business.png
try:
    _c2 = get_crop(2, 219, 103)
    canvas.paste(_c2, (1173, 410), _c2)
except Exception:
    pass
layout["Business"] = [1173, 410, 1392, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_09_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-11/03_icon_2_Filters.png
try:
    _c3 = get_crop(3, 392, 103)
    canvas.paste(_c3, (54, 410), _c3)
except Exception:
    pass
layout["2_Filters"] = [54, 410, 446, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_09_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-11/04_icon_San_Hateo.png
try:
    _c4 = get_crop(4, 144, 144)
    canvas.paste(_c4, (1092, 1192), _c4)
except Exception:
    pass
layout["San_Hateo"] = [1092, 1192, 1236, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_09_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-11/05_icon_City.png
try:
    _c5 = get_crop(5, 144, 144)
    canvas.paste(_c5, (1236, 1192), _c5)
except Exception:
    pass
layout["City"] = [1236, 1192, 1380, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_09_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-11/06_icon_Close_current_screen.png
try:
    _c6 = get_crop(6, 144, 144)
    canvas.paste(_c6, (1248, 96), _c6)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 96, 1392, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_09_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-11/07_icon_Cooking.png
try:
    _c7 = get_crop(7, 65, 63)
    canvas.paste(_c7, (308, 1), _c7)
except Exception:
    pass
layout["Cooking"] = [308, 1, 373, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_09_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-11/08_icon_Cooking.png
try:
    _c8 = get_crop(8, 1344, 191)
    canvas.paste(_c8, (48, 72), _c8)
except Exception:
    pass
layout["Cooking"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_09_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-11/09_icon_4.39.png
try:
    _c9 = get_crop(9, 59, 63)
    canvas.paste(_c9, (181, 1), _c9)
except Exception:
    pass
layout["4.39"] = [181, 1, 240, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_09_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-11/10_icon_4.39.png
try:
    _c10 = get_crop(10, 58, 65)
    canvas.paste(_c10, (115, 0), _c10)
except Exception:
    pass
layout["4.39"] = [115, 0, 173, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_09_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-11/11_icon_4.39.png
try:
    _c11 = get_crop(11, 117, 113)
    canvas.paste(_c11, (58, 114), _c11)
except Exception:
    pass
layout["4.39"] = [58, 114, 175, 227]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_09_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-11/12_icon_Favorite_button.png
try:
    _c12 = get_crop(12, 144, 144)
    canvas.paste(_c12, (1092, 2348), _c12)
except Exception:
    pass
layout["Favorite_button"] = [1092, 2348, 1236, 2492]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_09_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-11/13_icon_Cooking.png
try:
    _c13 = get_crop(13, 52, 64)
    canvas.paste(_c13, (247, 0), _c13)
except Exception:
    pass
layout["Cooking"] = [247, 0, 299, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_09_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-11/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 88, 63)
    canvas.paste(_c14, (1210, 0), _c14)
except Exception:
    pass
layout["icon_14"] = [1210, 0, 1298, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_09_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-11/15_icon_SA_TURDAY_APRIL_27_-_IIAM_-_2PM.png
try:
    _c15 = get_crop(15, 1344, 1108)
    canvas.paste(_c15, (48, 676), _c15)
except Exception:
    pass
layout["SA_TURDAY,_APRIL_27_-_IIA"] = [48, 676, 1392, 1784]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_09_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-11/16_icon_icon_16.png
try:
    _c16 = get_crop(16, 55, 63)
    canvas.paste(_c16, (1318, 0), _c16)
except Exception:
    pass
layout["icon_16"] = [1318, 0, 1373, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_09_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-11/17_icon_Sun_Apr_28_1_00_AM_PDT.png
try:
    _c17 = get_crop(17, 288, 156)
    canvas.paste(_c17, (288, 2804), _c17)
except Exception:
    pass
layout["Sun,_Apr_28_+_1:00_AM_PDT"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_09_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-11/18_icon_Cooking.png
try:
    _c18 = get_crop(18, 47, 63)
    canvas.paste(_c18, (384, 1), _c18)
except Exception:
    pass
layout["Cooking"] = [384, 1, 431, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_09_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-11/19_icon_Overflow_menu_button.png
try:
    _c19 = get_crop(19, 144, 144)
    canvas.paste(_c19, (1236, 2348), _c19)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 2348, 1380, 2492]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_09_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-11/20_icon_San_Francisco.png
try:
    _c20 = get_crop(20, 536, 144)
    canvas.paste(_c20, (0, 259), _c20)
except Exception:
    pass
layout["San_Francisco"] = [0, 259, 536, 403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_09_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-11/21_icon_4.39.png
try:
    _c21 = get_crop(21, 93, 64)
    canvas.paste(_c21, (12, 0), _c21)
except Exception:
    pass
layout["4.39"] = [12, 0, 105, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_09_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-11/22_icon_icon_22.png
try:
    _c22 = get_crop(22, 42, 64)
    canvas.paste(_c22, (1273, 0), _c22)
except Exception:
    pass
layout["icon_22"] = [1273, 0, 1315, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_09_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-11/23_icon_Favorites.png
try:
    _c23 = get_crop(23, 288, 156)
    canvas.paste(_c23, (576, 2804), _c23)
except Exception:
    pass
layout["Favorites"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_09_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-11/24_icon_Free.png
try:
    _c24 = get_crop(24, 125, 79)
    canvas.paste(_c24, (91, 2524), _c24)
except Exception:
    pass
layout["Free"] = [91, 2524, 216, 2603]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_09_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-11/25_icon_FREE_in_Hilton_Concord.png
try:
    _c25 = get_crop(25, 288, 156)
    canvas.paste(_c25, (864, 2804), _c25)
except Exception:
    pass
layout["FREE_in_Hilton_Concord"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_09_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-11/26_icon_Hilton_Concord.png
try:
    _c26 = get_crop(26, 288, 156)
    canvas.paste(_c26, (0, 2804), _c26)
except Exception:
    pass
layout["Hilton_Concord"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_09_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-11/27_icon_B_0_UtiquE.png
try:
    _c27 = get_crop(27, 1344, 984)
    canvas.paste(_c27, (48, 1832), _c27)
except Exception:
    pass
layout["B_0_UtiquE"] = [48, 1832, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_09_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-11/28_icon_FREE_in_Hilton_Concord.png
try:
    _c28 = get_crop(28, 288, 156)
    canvas.paste(_c28, (864, 2804), _c28)
except Exception:
    pass
layout["FREE_in_Hilton_Concord"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_09_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-11/29_icon_More.png
try:
    _c29 = get_crop(29, 288, 156)
    canvas.paste(_c29, (1152, 2804), _c29)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_09_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-11/30_text_11_events.png
try:
    _c30 = get_crop(30, 392, 103)
    canvas.paste(_c30, (54, 410), _c30)
except Exception:
    pass
layout["11_events"] = [54, 410, 446, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_09_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-11/31_text_Free.png
try:
    _c31 = get_crop(31, 80, 38)
    canvas.paste(_c31, (117, 1391), _c31)
except Exception:
    pass
layout["Free"] = [117, 1391, 197, 1429]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_09_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-11/32_text_Earth.png
try:
    _c32 = get_crop(32, 155, 63)
    canvas.paste(_c32, (92, 1456), _c32)
except Exception:
    pass
layout["Earth"] = [92, 1456, 247, 1519]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_09_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-11/33_text_Every_Day_Faire_At_Foster.png
try:
    _c33 = get_crop(33, 1344, 1108)
    canvas.paste(_c33, (48, 676), _c33)
except Exception:
    pass
layout["Every_Day_Faire_At_Foster"] = [48, 676, 1392, 1784]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_09_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-11/34_text_Library.png
try:
    _c34 = get_crop(34, 203, 77)
    canvas.paste(_c34, (89, 1532), _c34)
except Exception:
    pass
layout["Library"] = [89, 1532, 292, 1609]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_09_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-11/35_text_Sat.png
try:
    _c35 = get_crop(35, 89, 52)
    canvas.paste(_c35, (90, 1620), _c35)
except Exception:
    pass
layout["Sat,"] = [90, 1620, 179, 1672]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_09_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-11/36_text_27.png
try:
    _c36 = get_crop(36, 66, 43)
    canvas.paste(_c36, (253, 1622), _c36)
except Exception:
    pass
layout["27"] = [253, 1622, 319, 1665]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_09_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-11/37_text_1I_00_AM_PDT.png
try:
    _c37 = get_crop(37, 276, 45)
    canvas.paste(_c37, (339, 1620), _c37)
except Exception:
    pass
layout["1I:00_AM_PDT"] = [339, 1620, 615, 1665]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_09_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-11/38_text_Foster.png
try:
    _c38 = get_crop(38, 126, 43)
    canvas.paste(_c38, (94, 1689), _c38)
except Exception:
    pass
layout["Foster"] = [94, 1689, 220, 1732]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_09_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-11/39_text_Library.png
try:
    _c39 = get_crop(39, 142, 56)
    canvas.paste(_c39, (296, 1685), _c39)
except Exception:
    pass
layout["Library"] = [296, 1685, 438, 1741]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_09_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-11/40_text_Hilton_Concord.png
try:
    _c40 = get_crop(40, 288, 156)
    canvas.paste(_c40, (0, 2804), _c40)
except Exception:
    pass
layout["Hilton_Concord"] = [0, 2804, 288, 2960]
