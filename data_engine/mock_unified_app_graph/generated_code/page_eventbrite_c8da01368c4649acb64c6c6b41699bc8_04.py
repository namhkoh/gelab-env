# page_id: page_eventbrite_c8da01368c4649acb64c6c6b41699bc8_04
# screenshot: 2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-6.png
# step_index: 4/13
# task: Open Eventbrite. Look up "Animal" events. Filter by events happening next week. Select the first event - who is the organizer?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw background and UI structure for the Event list page

# Colors
STATUS_BAR = "#bdbdbd"
DIVIDER = "#e6e6e6"
CARD_SHADOW = "#efefef"
CARD_BG = "#ffffff"
IMAGE_BG = "#f2f6fb"
SECOND_IMAGE_BG = "#fbfcfe"
BOTTOM_BAR_BG = "#ffffff"

# Status bar (top ~50px)
draw.rectangle((0, 0, 1440, 56), fill=STATUS_BAR)

# Header area (keeps white background, draw subtle bottom divider)
draw.line((40, 220, 1400, 220), fill=DIVIDER, width=2)

# Thin subtle divider under search/header area (closer to where chips begin)
draw.line((40, 260, 1400, 260), fill=DIVIDER, width=1)

# First event card (rounded card background + shadow)
# Using the detected primary event region as a guide (pos approx 48,676 size 1344x1115)
card1_left, card1_top = 48, 676
card1_right, card1_bottom = card1_left + 1344, card1_top + 1115

# Shadow (offset slightly down-right)
shadow_offset = 12
draw.rounded_rectangle(
    (card1_left + shadow_offset, card1_top + shadow_offset, card1_right + shadow_offset, card1_bottom + shadow_offset),
    radius=26, fill=CARD_SHADOW
)

# Card background
draw.rounded_rectangle(
    (card1_left, card1_top, card1_right, card1_bottom),
    radius=22, fill=CARD_BG
)

# Image area background inside the first card (top portion) so images pasted look anchored
image_area_height1 = int((card1_bottom - card1_top) * 0.42)
draw.rounded_rectangle(
    (card1_left + 16, card1_top + 16, card1_right - 16, card1_top + 16 + image_area_height1),
    radius=14, fill=IMAGE_BG
)

# Separator line between first card content and following content
sep_y = card1_bottom + 18
draw.line((48, sep_y, 1392, sep_y), fill=DIVIDER, width=1)

# Second event card (rounded card background + shadow)
# Detected second event region approx pos (48,1839) size 1344x945
card2_left, card2_top = 48, 1839
card2_right, card2_bottom = card2_left + 1344, card2_top + 945

# Shadow
draw.rounded_rectangle(
    (card2_left + shadow_offset, card2_top + shadow_offset, card2_right + shadow_offset, card2_bottom + shadow_offset),
    radius=26, fill=CARD_SHADOW
)

# Card background
draw.rounded_rectangle(
    (card2_left, card2_top, card2_right, card2_bottom),
    radius=22, fill=CARD_BG
)

# Image area background inside the second card (top portion)
image_area_height2 = int((card2_bottom - card2_top) * 0.45)
draw.rounded_rectangle(
    (card2_left + 16, card2_top + 16, card2_right - 16, card2_top + 16 + image_area_height2),
    radius=14, fill=SECOND_IMAGE_BG
)

# Subtle dividing line above the bottom navigation
nav_top = 2804
draw.line((24, nav_top, 1416, nav_top), fill=DIVIDER, width=2)

# Bottom navigation bar background
draw.rectangle((0, nav_top, 1440, 2960), fill=BOTTOM_BAR_BG)

# Slight top shadow on nav bar to lift it
draw.line((0, nav_top + 2, 1440, nav_top + 2), fill="#f0f0f0", width=1)

# Additional subtle full-width separators for visual grouping
# Under header area (another faint line)
draw.line((24, 340, 1416, 340), fill="#f3f3f3", width=1)

# Add left and right margins' vertical guides (very faint) to visually align content blocks
draw.line((44, 220, 44, nav_top - 24), fill="#fafafa", width=1)
draw.line((1396, 220, 1396, nav_top - 24), fill="#fafafa", width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_04_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-6/00_icon_Music.png
try:
    _c0 = get_crop(0, 187, 103)
    canvas.paste(_c0, (837, 410), _c0)
except Exception:
    pass
layout["Music"] = [837, 410, 1024, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_04_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-6/01_icon_Business.png
try:
    _c1 = get_crop(1, 241, 103)
    canvas.paste(_c1, (1036, 410), _c1)
except Exception:
    pass
layout["Business"] = [1036, 410, 1277, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_04_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-6/02_icon_Anytime.png
try:
    _c2 = get_crop(2, 400, 103)
    canvas.paste(_c2, (425, 410), _c2)
except Exception:
    pass
layout["Anytime"] = [425, 410, 825, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_04_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-6/03_icon_Filters.png
try:
    _c3 = get_crop(3, 359, 103)
    canvas.paste(_c3, (54, 410), _c3)
except Exception:
    pass
layout["Filters"] = [54, 410, 413, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_04_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-6/04_icon_Foo.png
try:
    _c4 = get_crop(4, 150, 110)
    canvas.paste(_c4, (1282, 406), _c4)
except Exception:
    pass
layout["Foo"] = [1282, 406, 1432, 516]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_04_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-6/05_icon_CONSORTIUM.png
try:
    _c5 = get_crop(5, 144, 144)
    canvas.paste(_c5, (1092, 1192), _c5)
except Exception:
    pass
layout["CONSORTIUM"] = [1092, 1192, 1236, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_04_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-6/06_icon_Foo.png
try:
    _c6 = get_crop(6, 144, 144)
    canvas.paste(_c6, (1248, 96), _c6)
except Exception:
    pass
layout["Foo"] = [1248, 96, 1392, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_04_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-6/07_icon_5.15.png
try:
    _c7 = get_crop(7, 118, 112)
    canvas.paste(_c7, (57, 115), _c7)
except Exception:
    pass
layout["5.15"] = [57, 115, 175, 227]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_04_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-6/08_icon_Minas_Gerais_Krustallos_Glen_Ellyn_Road_.png
try:
    _c8 = get_crop(8, 288, 156)
    canvas.paste(_c8, (576, 2804), _c8)
except Exception:
    pass
layout["Minas_Gerais_Krustallos,_"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_04_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-6/09_icon_5.15.png
try:
    _c9 = get_crop(9, 61, 65)
    canvas.paste(_c9, (180, 0), _c9)
except Exception:
    pass
layout["5.15"] = [180, 0, 241, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_04_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-6/10_icon_Animal.png
try:
    _c10 = get_crop(10, 68, 64)
    canvas.paste(_c10, (308, 0), _c10)
except Exception:
    pass
layout["Animal"] = [308, 0, 376, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_04_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-6/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 106, 61)
    canvas.paste(_c11, (1205, 0), _c11)
except Exception:
    pass
layout["icon_11"] = [1205, 0, 1311, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_04_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-6/12_icon_Animal.png
try:
    _c12 = get_crop(12, 54, 65)
    canvas.paste(_c12, (246, 0), _c12)
except Exception:
    pass
layout["Animal"] = [246, 0, 300, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_04_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-6/13_icon_5.15.png
try:
    _c13 = get_crop(13, 62, 66)
    canvas.paste(_c13, (114, 0), _c13)
except Exception:
    pass
layout["5.15"] = [114, 0, 176, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_04_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-6/14_icon_Chicago.png
try:
    _c14 = get_crop(14, 417, 144)
    canvas.paste(_c14, (0, 259), _c14)
except Exception:
    pass
layout["Chicago"] = [0, 259, 417, 403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_04_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-6/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 59, 61)
    canvas.paste(_c15, (1318, 0), _c15)
except Exception:
    pass
layout["icon_15"] = [1318, 0, 1377, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_04_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-6/16_icon_Minas_Gerais_Krustallos_Glen_Ellyn_Road_.png
try:
    _c16 = get_crop(16, 288, 156)
    canvas.paste(_c16, (864, 2804), _c16)
except Exception:
    pass
layout["Minas_Gerais_Krustallos,_"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_04_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-6/17_icon_Memouea.png
try:
    _c17 = get_crop(17, 1344, 1115)
    canvas.paste(_c17, (48, 676), _c17)
except Exception:
    pass
layout["Memouea"] = [48, 676, 1392, 1791]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_04_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-6/18_icon_Minas_Gerais_Krustallos_Glen_Ellyn_Road_.png
try:
    _c18 = get_crop(18, 288, 156)
    canvas.paste(_c18, (1152, 2804), _c18)
except Exception:
    pass
layout["Minas_Gerais_Krustallos,_"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_04_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-6/19_icon_Animal.png
try:
    _c19 = get_crop(19, 1344, 191)
    canvas.paste(_c19, (48, 72), _c19)
except Exception:
    pass
layout["Animal"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_04_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-6/20_icon_Favorite_button.png
try:
    _c20 = get_crop(20, 144, 144)
    canvas.paste(_c20, (1092, 2355), _c20)
except Exception:
    pass
layout["Favorite_button"] = [1092, 2355, 1236, 2499]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_04_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-6/21_icon_icon_21.png
try:
    _c21 = get_crop(21, 50, 61)
    canvas.paste(_c21, (384, 3), _c21)
except Exception:
    pass
layout["icon_21"] = [384, 3, 434, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_04_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-6/22_icon_Discover_your_mystical_Power_Animals.png
try:
    _c22 = get_crop(22, 1344, 945)
    canvas.paste(_c22, (48, 1839), _c22)
except Exception:
    pass
layout["Discover_your_mystical_Po"] = [48, 1839, 1392, 2784]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_04_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-6/23_icon_CONSORTIUM.png
try:
    _c23 = get_crop(23, 144, 144)
    canvas.paste(_c23, (1236, 1192), _c23)
except Exception:
    pass
layout["CONSORTIUM"] = [1236, 1192, 1380, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_04_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-6/24_icon_tickets_left.png
try:
    _c24 = get_crop(24, 367, 85)
    canvas.paste(_c24, (89, 1370), _c24)
except Exception:
    pass
layout["tickets_left"] = [89, 1370, 456, 1455]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_04_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-6/25_icon_7_00_PM_CDT.png
try:
    _c25 = get_crop(25, 288, 156)
    canvas.paste(_c25, (288, 2804), _c25)
except Exception:
    pass
layout["7:00_PM_CDT"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_04_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-6/26_text_5.15.png
try:
    _c26 = get_crop(26, 92, 43)
    canvas.paste(_c26, (22, 17), _c26)
except Exception:
    pass
layout["5.15"] = [22, 17, 114, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_04_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-6/27_text_179_events.png
try:
    _c27 = get_crop(27, 359, 103)
    canvas.paste(_c27, (54, 410), _c27)
except Exception:
    pass
layout["179_events"] = [54, 410, 413, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_04_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-6/28_text_Animal_Welfare_Music_and_Memories_Night.png
try:
    _c28 = get_crop(28, 1344, 1115)
    canvas.paste(_c28, (48, 676), _c28)
except Exception:
    pass
layout["Animal_Welfare_Music_and_"] = [48, 676, 1392, 1791]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_04_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-6/29_text_Wed_Apr_24.png
try:
    _c29 = get_crop(29, 246, 50)
    canvas.paste(_c29, (95, 1561), _c29)
except Exception:
    pass
layout["Wed,_Apr_24"] = [95, 1561, 341, 1611]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_04_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-6/30_text_7.30_PM_EDT.png
try:
    _c30 = get_crop(30, 251, 45)
    canvas.paste(_c30, (357, 1560), _c30)
except Exception:
    pass
layout["7.30_PM_EDT"] = [357, 1560, 608, 1605]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_04_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-6/31_text_Online.png
try:
    _c31 = get_crop(31, 126, 43)
    canvas.paste(_c31, (94, 1629), _c31)
except Exception:
    pass
layout["Online"] = [94, 1629, 220, 1672]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_04_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-6/32_text_Promoted.png
try:
    _c32 = get_crop(32, 193, 40)
    canvas.paste(_c32, (94, 1699), _c32)
except Exception:
    pass
layout["Promoted"] = [94, 1699, 287, 1739]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_04_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-6/33_clickable_Overflow_menu_button.png
try:
    _c33 = get_crop(33, 144, 144)
    canvas.paste(_c33, (1236, 2355), _c33)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 2355, 1380, 2499]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_04_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-6/34_clickable_Home.png
try:
    _c34 = get_crop(34, 288, 156)
    canvas.paste(_c34, (0, 2804), _c34)
except Exception:
    pass
layout["Home"] = [0, 2804, 288, 2960]
