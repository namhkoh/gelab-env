# page_id: page_eventbrite_c36ef2ec281b4d40839b2bf8f321a58b_08
# screenshot: 2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-10.png
# step_index: 8/8
# task: Open Eventbrite. Set the city to "Chicago". Select the "Fashion" category and view the recommended events. See the date of the first play and its venue.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Canvas and draw are provided. Fonts are available but not used for drawing text here.
w, h = canvas.size

# Colors
bg_color = (247, 248, 250)        # very light off-white page background
status_bar_color = (158, 158, 158) # status bar gray
header_bg = (255, 255, 255)       # white header/search area
divider_color = (221, 224, 228)   # subtle divider
card_bg = (255, 255, 255)         # card background (white)
card_shadow = (232, 234, 237)     # subtle shadow for cards
nav_bg = (255, 255, 255)          # bottom navigation bar background
nav_divider = (214, 216, 219)     # nav top divider

# Fill overall background
draw.rectangle([(0, 0), (w, h)], fill=bg_color)

# Status bar area (~72px high)
status_h = 72
draw.rectangle([(0, 0), (w, status_h)], fill=status_bar_color)

# Header / Search area (beneath status bar)
header_top = status_h
header_bottom = 220
draw.rectangle([(0, header_top), (w, header_bottom)], fill=header_bg)
# subtle bottom divider under search/header
draw.line([(24, header_bottom), (w-24, header_bottom)], fill=divider_color, width=2)

# Filters row divider (under filter pills area)
filters_divider_y = 480
draw.line([(24, filters_divider_y), (w-24, filters_divider_y)], fill=divider_color, width=1)

# Draw two main content cards (containers) with rounded corners and subtle shadows.
# Card 1 (first event) - positioned to align with detected event content blocks
card1_x0, card1_y0 = 36, 620
card1_x1, card1_y1 = w - 36, 1440
radius = 20
# shadow
shadow_offset = 8
draw.rounded_rectangle(
    [(card1_x0, card1_y0 + shadow_offset), (card1_x1, card1_y1 + shadow_offset)],
    radius=radius, fill=card_shadow
)
# card body
draw.rounded_rectangle(
    [(card1_x0, card1_y0), (card1_x1, card1_y1)],
    radius=radius, fill=card_bg
)
# subtle border line at bottom of card1
draw.line([(card1_x0 + 12, card1_y1), (card1_x1 - 12, card1_y1)], fill=divider_color, width=1)

# Card 2 (second event)
card2_x0, card2_y0 = 36, 1760
card2_x1, card2_y1 = w - 36, 2560
# shadow
draw.rounded_rectangle(
    [(card2_x0, card2_y0 + shadow_offset), (card2_x1, card2_y1 + shadow_offset)],
    radius=radius, fill=card_shadow
)
# card body
draw.rounded_rectangle(
    [(card2_x0, card2_y0), (card2_x1, card2_y1)],
    radius=radius, fill=card_bg
)
# subtle border line at bottom of card2
draw.line([(card2_x0 + 12, card2_y1), (card2_x1 - 12, card2_y1)], fill=divider_color, width=1)

# Lightweight separators between stacked content sections
sep_x0 = 24
sep_x1 = w - 24
# a separator just above the first card (content heading area)
draw.line([(sep_x0, card1_y0 - 28), (sep_x1, card1_y0 - 28)], fill=divider_color, width=1)
# separator between card1 and card2
draw.line([(sep_x0, card1_y1 + 24), (sep_x1, card1_y1 + 24)], fill=divider_color, width=1)

# Bottom navigation bar background (reserve area for clickable nav)
nav_top = 2804
draw.rectangle([(0, nav_top), (w, h)], fill=nav_bg)
# top divider for nav
draw.line([(0, nav_top), (w, nav_top)], fill=nav_divider, width=2)

# Small visual indicator bars on cards to imply image/content areas (abstract only)
# For first card: a subtle darker band at the top area to suggest hero image placement (not an image)
band_h = 220
band_margin = 28
band_color = (245, 245, 246)
draw.rectangle([(card1_x0 + band_margin, card1_y0 + band_margin),
                (card1_x1 - band_margin, card1_y0 + band_margin + band_h)],
               fill=band_color)

# For second card: similar band for a hero thumbnail area
draw.rectangle([(card2_x0 + band_margin, card2_y0 + band_margin),
                (card2_x1 - band_margin, card2_y0 + band_margin + band_h)],
               fill=band_color)

# subtle rounded corner accent on page edges (thin stroke)
edge_stroke = (238, 239, 241)
draw.rounded_rectangle([(8, 8), (w-8, h-8)], radius=24, outline=edge_stroke, width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_08_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-10/00_icon_Fashion.png
try:
    _c0 = get_crop(0, 220, 135)
    canvas.paste(_c0, (850, 390), _c0)
except Exception:
    pass
layout["Fashion"] = [850, 390, 1070, 525]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_08_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-10/01_icon_Anytime.png
try:
    _c1 = get_crop(1, 400, 135)
    canvas.paste(_c1, (438, 390), _c1)
except Exception:
    pass
layout["Anytime"] = [438, 390, 838, 525]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_08_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-10/02_icon_1_Filter.png
try:
    _c2 = get_crop(2, 372, 135)
    canvas.paste(_c2, (54, 390), _c2)
except Exception:
    pass
layout["1_Filter"] = [54, 390, 426, 525]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_08_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-10/03_icon_Overflow_menu_button.png
try:
    _c3 = get_crop(3, 144, 144)
    canvas.paste(_c3, (1236, 2331), _c3)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 2331, 1380, 2475]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_08_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-10/04_icon_Favorite_button.png
try:
    _c4 = get_crop(4, 144, 144)
    canvas.paste(_c4, (1092, 2331), _c4)
except Exception:
    pass
layout["Favorite_button"] = [1092, 2331, 1236, 2475]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_08_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-10/05_icon_Wite.png
try:
    _c5 = get_crop(5, 144, 144)
    canvas.paste(_c5, (1092, 1192), _c5)
except Exception:
    pass
layout["Wite"] = [1092, 1192, 1236, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_08_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-10/06_icon_5.13.png
try:
    _c6 = get_crop(6, 121, 112)
    canvas.paste(_c6, (56, 115), _c6)
except Exception:
    pass
layout["5.13"] = [56, 115, 177, 227]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_08_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-10/07_icon_Search_forae.png
try:
    _c7 = get_crop(7, 67, 63)
    canvas.paste(_c7, (308, 0), _c7)
except Exception:
    pass
layout["Search_forae"] = [308, 0, 375, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_08_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-10/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 55, 64)
    canvas.paste(_c8, (246, 0), _c8)
except Exception:
    pass
layout["icon_8"] = [246, 0, 301, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_08_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-10/09_icon_5.13.png
try:
    _c9 = get_crop(9, 60, 63)
    canvas.paste(_c9, (181, 0), _c9)
except Exception:
    pass
layout["5.13"] = [181, 0, 241, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_08_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-10/10_icon_Reosier.png
try:
    _c10 = get_crop(10, 144, 144)
    canvas.paste(_c10, (1236, 1192), _c10)
except Exception:
    pass
layout["Reosier"] = [1236, 1192, 1380, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_08_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-10/11_icon_Visj.png
try:
    _c11 = get_crop(11, 1344, 1091)
    canvas.paste(_c11, (48, 676), _c11)
except Exception:
    pass
layout["Visj"] = [48, 676, 1392, 1767]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_08_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-10/12_icon_5.13.png
try:
    _c12 = get_crop(12, 61, 65)
    canvas.paste(_c12, (114, 0), _c12)
except Exception:
    pass
layout["5.13"] = [114, 0, 175, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_08_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-10/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 62, 59)
    canvas.paste(_c13, (1318, 0), _c13)
except Exception:
    pass
layout["icon_13"] = [1318, 0, 1380, 59]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_08_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-10/14_icon_Financial_Literacy_for_Beauty_Profession.png
try:
    _c14 = get_crop(14, 288, 156)
    canvas.paste(_c14, (576, 2804), _c14)
except Exception:
    pass
layout["Financial_Literacy_for_Be"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_08_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-10/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 72, 60)
    canvas.paste(_c15, (1207, 0), _c15)
except Exception:
    pass
layout["icon_15"] = [1207, 0, 1279, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_08_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-10/16_icon_Search_forae.png
try:
    _c16 = get_crop(16, 1344, 191)
    canvas.paste(_c16, (48, 72), _c16)
except Exception:
    pass
layout["Search_forae"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_08_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-10/17_icon_Search_forae.png
try:
    _c17 = get_crop(17, 51, 61)
    canvas.paste(_c17, (383, 2), _c17)
except Exception:
    pass
layout["Search_forae"] = [383, 2, 434, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_08_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-10/18_icon_icon_18.png
try:
    _c18 = get_crop(18, 62, 55)
    canvas.paste(_c18, (35, 1961), _c18)
except Exception:
    pass
layout["icon_18"] = [35, 1961, 97, 2016]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_08_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-10/19_icon_Financial_Literacy_for_Beauty_Profession.png
try:
    _c19 = get_crop(19, 1344, 1001)
    canvas.paste(_c19, (48, 1815), _c19)
except Exception:
    pass
layout["Financial_Literacy_for_Be"] = [48, 1815, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_08_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-10/20_icon_Financial_Literacy_for_Beauty_Profession.png
try:
    _c20 = get_crop(20, 288, 156)
    canvas.paste(_c20, (864, 2804), _c20)
except Exception:
    pass
layout["Financial_Literacy_for_Be"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_08_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-10/21_icon_Pow.png
try:
    _c21 = get_crop(21, 268, 245)
    canvas.paste(_c21, (1054, 1818), _c21)
except Exception:
    pass
layout["Pow"] = [1054, 1818, 1322, 2063]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_08_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-10/22_icon_Chicago.png
try:
    _c22 = get_crop(22, 417, 144)
    canvas.paste(_c22, (0, 259), _c22)
except Exception:
    pass
layout["Chicago"] = [0, 259, 417, 403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_08_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-10/23_icon_Promoted.png
try:
    _c23 = get_crop(23, 43, 60)
    canvas.paste(_c23, (285, 1661), _c23)
except Exception:
    pass
layout["Promoted"] = [285, 1661, 328, 1721]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_08_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-10/24_icon_7_00_PM_EDT.png
try:
    _c24 = get_crop(24, 288, 156)
    canvas.paste(_c24, (288, 2804), _c24)
except Exception:
    pass
layout["7:00_PM_EDT"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_08_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-10/25_icon_icon_25.png
try:
    _c25 = get_crop(25, 62, 55)
    canvas.paste(_c25, (34, 2012), _c25)
except Exception:
    pass
layout["icon_25"] = [34, 2012, 96, 2067]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_08_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-10/26_icon_More.png
try:
    _c26 = get_crop(26, 288, 156)
    canvas.paste(_c26, (1152, 2804), _c26)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_08_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-10/27_icon_Pow.png
try:
    _c27 = get_crop(27, 57, 58)
    canvas.paste(_c27, (1314, 1918), _c27)
except Exception:
    pass
layout["Pow"] = [1314, 1918, 1371, 1976]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_08_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-10/28_icon_icon_28.png
try:
    _c28 = get_crop(28, 40, 59)
    canvas.paste(_c28, (1274, 1), _c28)
except Exception:
    pass
layout["icon_28"] = [1274, 1, 1314, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_08_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-10/29_icon_icon_29.png
try:
    _c29 = get_crop(29, 65, 49)
    canvas.paste(_c29, (33, 1915), _c29)
except Exception:
    pass
layout["icon_29"] = [33, 1915, 98, 1964]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_08_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-10/30_text_5.13.png
try:
    _c30 = get_crop(30, 89, 43)
    canvas.paste(_c30, (22, 17), _c30)
except Exception:
    pass
layout["5.13"] = [22, 17, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_08_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-10/31_text_325_events.png
try:
    _c31 = get_crop(31, 372, 135)
    canvas.paste(_c31, (54, 390), _c31)
except Exception:
    pass
layout["325_events"] = [54, 390, 426, 525]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_08_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-10/32_text_TIIL.png
try:
    _c32 = get_crop(32, 86, 48)
    canvas.paste(_c32, (70, 696), _c32)
except Exception:
    pass
layout["TIIL"] = [70, 696, 156, 744]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_08_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-10/33_text_FashionBar_The_Showroom_Consulting.png
try:
    _c33 = get_crop(33, 1344, 1091)
    canvas.paste(_c33, (48, 676), _c33)
except Exception:
    pass
layout["FashionBar_The_Showroom_&"] = [48, 676, 1392, 1767]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_08_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-10/34_text_FREEDOM.png
try:
    _c34 = get_crop(34, 559, 109)
    canvas.paste(_c34, (99, 1958), _c34)
except Exception:
    pass
layout["FREEDOM"] = [99, 1958, 658, 2067]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_08_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-10/35_text_Mon_Apr_29.png
try:
    _c35 = get_crop(35, 244, 54)
    canvas.paste(_c35, (93, 2678), _c35)
except Exception:
    pass
layout["Mon,_Apr_29"] = [93, 2678, 337, 2732]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_08_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-10/36_text_7_00_PM_EDT.png
try:
    _c36 = get_crop(36, 288, 156)
    canvas.paste(_c36, (288, 2804), _c36)
except Exception:
    pass
layout["7:00_PM_EDT"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_08_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-10/37_text_Online.png
try:
    _c37 = get_crop(37, 133, 54)
    canvas.paste(_c37, (89, 2744), _c37)
except Exception:
    pass
layout["Online"] = [89, 2744, 222, 2798]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_08_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-10/38_clickable_Home.png
try:
    _c38 = get_crop(38, 288, 156)
    canvas.paste(_c38, (0, 2804), _c38)
except Exception:
    pass
layout["Home"] = [0, 2804, 288, 2960]
