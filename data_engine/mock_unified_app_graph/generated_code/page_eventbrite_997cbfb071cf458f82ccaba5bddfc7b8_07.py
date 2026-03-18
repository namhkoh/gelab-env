# page_id: page_eventbrite_997cbfb071cf458f82ccaba5bddfc7b8_07
# screenshot: 2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-9.png
# step_index: 7/15
# task: Open Eventbrite. Search Online free events. Select the first one. Follow the organizer. Read more about the event. Add it to Favorites.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Page background
page_bg = "#f6f7f9"
draw.rectangle((0, 0, 1440, 2960), fill=page_bg)

# Status bar (top)
status_h = 72
status_color = "#9e9e9e"
draw.rectangle((0, 0, 1440, status_h), fill=status_color)

# Header / Search area card background
search_left, search_top = 48, 72
search_right, search_bottom = 1392, 72 + 191  # matches detected search area height
search_bg = "#ffffff"
search_border = "#e6e8eb"
draw.rounded_rectangle((search_left, search_top, search_right, search_bottom),
                       radius=18, fill=search_bg, outline=search_border, width=1)

# Divider under search area
divider_color = "#e0e1e4"
draw.line((search_left + 8, search_bottom + 6, search_right - 8, search_bottom + 6), fill=divider_color, width=2)

# Thin subtle shadow below header area
draw.rectangle((search_left + 6, search_bottom + 8, search_right - 6, search_bottom + 10), fill="#f0f1f3")

# Divider under filters row (approx)
filters_div_y = 420
draw.line((search_left, filters_div_y, search_right, filters_div_y), fill=divider_color, width=2)

# "Content list" background area (keeps a subtle tint)
content_area_top = filters_div_y + 24
draw.rectangle((0, content_area_top, 1440, 2760), fill=page_bg)

# First event card (rounded white card with subtle drop shadow)
card_margin_h = 48
card_left = 48
card_right = 1440 - card_margin_h
card1_top = 440
card1_bottom = 920
card_radius = 20

# Shadow
shadow_color = "#e9eef5"
draw.rounded_rectangle((card_left + 8, card1_top + 10, card_right + 8, card1_bottom + 10),
                       radius=card_radius, fill=shadow_color)

# Card background
card_bg = "#ffffff"
draw.rounded_rectangle((card_left, card1_top, card_right, card1_bottom),
                       radius=card_radius, fill=card_bg)

# Image/frame area inside first card (background frame only, actual image will be pasted)
img1_left = card_left + 16
img1_top = card1_top + 16
img1_right = card_right - 16
img1_height = 240
img1_bottom = img1_top + img1_height
img_frame_color = "#f3f5f7"  # neutral placeholder behind pasted image
draw.rounded_rectangle((img1_left, img1_top, img1_right, img1_bottom),
                       radius=14, fill=img_frame_color)

# Subtle divider below image within the card
draw.line((img1_left, img1_bottom + 12, img1_right, img1_bottom + 12), fill="#efeff1", width=1)

# Space reserved for metadata area (no text drawn)
meta_top = img1_bottom + 24
meta_bottom = card1_bottom - 24
# Draw faint blocks to suggest content groupings (no text)
draw.rectangle((img1_left, meta_top, img1_right, meta_top + 2), fill="#f7f8fa")
draw.rectangle((img1_left, meta_top + 52, img1_right, meta_top + 54), fill="#f7f8fa")

# Separator between events
sep_y = card1_bottom + 32
draw.line((48, sep_y, 1392, sep_y), fill=divider_color, width=1)

# Second event card (rounded white card with subtle drop shadow)
card2_top = sep_y + 32
card2_bottom = card2_top + 420
# Shadow
draw.rounded_rectangle((card_left + 8, card2_top + 10, card_right + 8, card2_bottom + 10),
                       radius=card_radius, fill=shadow_color)
# Card background
draw.rounded_rectangle((card_left, card2_top, card_right, card2_bottom),
                       radius=card_radius, fill=card_bg)
# Image/frame area inside second card (background only)
img2_left = card_left + 16
img2_top = card2_top + 16
img2_right = card_right - 16
img2_height = 300
img2_bottom = img2_top + img2_height
img2_frame_color = "#0f0f0f"  # dark placeholder behind pasted image (matches darker media area)
draw.rounded_rectangle((img2_left, img2_top, img2_right, img2_bottom),
                       radius=14, fill=img2_frame_color)

# Small decorative horizontal rule below second image
draw.line((img2_left, img2_bottom + 14, img2_right, img2_bottom + 14), fill="#efeff1", width=1)

# Footer / bottom navigation background (leave icons to be pasted)
nav_top = 2804
nav_color = "#ffffff"
draw.rectangle((0, nav_top, 1440, 2960), fill=nav_color)
# Top border for nav
draw.line((0, nav_top, 1440, nav_top), fill="#e6e8eb", width=2)
# Slight shadow above nav
draw.rectangle((0, nav_top - 6, 1440, nav_top - 2), fill="#f2f3f5")

# Left and right side margins subtle vertical guides (not content)
side_guide_color = "#f3f4f5"
draw.rectangle((0, 0, 24, 2960), fill=side_guide_color)
draw.rectangle((1440 - 24, 0, 1440, 2960), fill=side_guide_color)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_07_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-9/00_icon_Music.png
try:
    _c0 = get_crop(0, 187, 103)
    canvas.paste(_c0, (850, 410), _c0)
except Exception:
    pass
layout["Music"] = [850, 410, 1037, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_07_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-9/01_icon_Anytime.png
try:
    _c1 = get_crop(1, 400, 103)
    canvas.paste(_c1, (438, 410), _c1)
except Exception:
    pass
layout["Anytime"] = [438, 410, 838, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_07_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-9/02_icon_Business.png
try:
    _c2 = get_crop(2, 241, 103)
    canvas.paste(_c2, (1049, 410), _c2)
except Exception:
    pass
layout["Business"] = [1049, 410, 1290, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_07_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-9/03_icon_1_Filter.png
try:
    _c3 = get_crop(3, 372, 103)
    canvas.paste(_c3, (54, 410), _c3)
except Exception:
    pass
layout["1_Filter"] = [54, 410, 426, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_07_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-9/04_icon_Fo.png
try:
    _c4 = get_crop(4, 133, 111)
    canvas.paste(_c4, (1296, 406), _c4)
except Exception:
    pass
layout["Fo("] = [1296, 406, 1429, 517]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_07_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-9/05_icon_EcOMMER.png
try:
    _c5 = get_crop(5, 144, 144)
    canvas.paste(_c5, (1092, 2415), _c5)
except Exception:
    pass
layout["EcOMMER"] = [1092, 2415, 1236, 2559]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_07_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-9/06_icon_WEDNESDAY_6_PM.png
try:
    _c6 = get_crop(6, 144, 144)
    canvas.paste(_c6, (1092, 1192), _c6)
except Exception:
    pass
layout["WEDNESDAY_6_PM"] = [1092, 1192, 1236, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_07_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-9/07_icon_9.15.png
try:
    _c7 = get_crop(7, 125, 115)
    canvas.paste(_c7, (55, 113), _c7)
except Exception:
    pass
layout["9.15"] = [55, 113, 180, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_07_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-9/08_icon_EcOMMER.png
try:
    _c8 = get_crop(8, 144, 144)
    canvas.paste(_c8, (1236, 2415), _c8)
except Exception:
    pass
layout["EcOMMER"] = [1236, 2415, 1380, 2559]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_07_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-9/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 55, 61)
    canvas.paste(_c9, (247, 1), _c9)
except Exception:
    pass
layout["icon_9"] = [247, 1, 302, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_07_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-9/10_icon_WEDNESDAY_6_PM.png
try:
    _c10 = get_crop(10, 144, 144)
    canvas.paste(_c10, (1236, 1192), _c10)
except Exception:
    pass
layout["WEDNESDAY_6_PM"] = [1236, 1192, 1380, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_07_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-9/11_icon_Search_forae.png
try:
    _c11 = get_crop(11, 59, 62)
    canvas.paste(_c11, (312, 1), _c11)
except Exception:
    pass
layout["Search_forae"] = [312, 1, 371, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_07_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-9/12_icon_9.15.png
try:
    _c12 = get_crop(12, 55, 62)
    canvas.paste(_c12, (182, 0), _c12)
except Exception:
    pass
layout["9.15"] = [182, 0, 237, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_07_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-9/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 61, 60)
    canvas.paste(_c13, (1316, 0), _c13)
except Exception:
    pass
layout["icon_13"] = [1316, 0, 1377, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_07_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-9/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 66, 61)
    canvas.paste(_c14, (1209, 0), _c14)
except Exception:
    pass
layout["icon_14"] = [1209, 0, 1275, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_07_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-9/15_icon_9.15.png
try:
    _c15 = get_crop(15, 56, 64)
    canvas.paste(_c15, (115, 0), _c15)
except Exception:
    pass
layout["9.15"] = [115, 0, 171, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_07_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-9/16_icon_2024.png
try:
    _c16 = get_crop(16, 288, 156)
    canvas.paste(_c16, (288, 2804), _c16)
except Exception:
    pass
layout["2024"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_07_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-9/17_icon_Search_forae.png
try:
    _c17 = get_crop(17, 50, 61)
    canvas.paste(_c17, (383, 2), _c17)
except Exception:
    pass
layout["Search_forae"] = [383, 2, 433, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_07_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-9/18_icon_icon_18.png
try:
    _c18 = get_crop(18, 68, 60)
    canvas.paste(_c18, (1245, 0), _c18)
except Exception:
    pass
layout["icon_18"] = [1245, 0, 1313, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_07_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-9/19_icon_Search_forae.png
try:
    _c19 = get_crop(19, 1344, 191)
    canvas.paste(_c19, (48, 72), _c19)
except Exception:
    pass
layout["Search_forae"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_07_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-9/20_icon_Building_a_7_Figure_Ecommerce_Business_i.png
try:
    _c20 = get_crop(20, 288, 156)
    canvas.paste(_c20, (576, 2804), _c20)
except Exception:
    pass
layout["Building_a_7_Figure_Ecomm"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_07_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-9/21_icon_Online.png
try:
    _c21 = get_crop(21, 377, 144)
    canvas.paste(_c21, (0, 259), _c21)
except Exception:
    pass
layout["Online"] = [0, 259, 377, 403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_07_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-9/22_icon_More.png
try:
    _c22 = get_crop(22, 288, 156)
    canvas.paste(_c22, (1152, 2804), _c22)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_07_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-9/23_icon_Active_Military_Veterans_VA_Homebuyer.png
try:
    _c23 = get_crop(23, 1344, 1175)
    canvas.paste(_c23, (48, 676), _c23)
except Exception:
    pass
layout["Active_Military_&_Veteran"] = [48, 676, 1392, 1851]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_07_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-9/24_icon_2024.png
try:
    _c24 = get_crop(24, 288, 156)
    canvas.paste(_c24, (0, 2804), _c24)
except Exception:
    pass
layout["2024"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_07_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-9/25_icon_Promoted.png
try:
    _c25 = get_crop(25, 242, 66)
    canvas.paste(_c25, (86, 1742), _c25)
except Exception:
    pass
layout["Promoted"] = [86, 1742, 328, 1808]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_07_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-9/26_text_9.15.png
try:
    _c26 = get_crop(26, 94, 43)
    canvas.paste(_c26, (20, 17), _c26)
except Exception:
    pass
layout["9.15"] = [20, 17, 114, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_07_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-9/27_text_10_000_events.png
try:
    _c27 = get_crop(27, 372, 103)
    canvas.paste(_c27, (54, 410), _c27)
except Exception:
    pass
layout["10,000_events"] = [54, 410, 426, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_07_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-9/28_text_ACTIVE_MILITARY_VETERANS.png
try:
    _c28 = get_crop(28, 372, 103)
    canvas.paste(_c28, (54, 410), _c28)
except Exception:
    pass
layout["ACTIVE_MILITARY_&_VETERAN"] = [54, 410, 426, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_07_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-9/29_text_Online.png
try:
    _c29 = get_crop(29, 129, 45)
    canvas.paste(_c29, (91, 1687), _c29)
except Exception:
    pass
layout["Online"] = [91, 1687, 220, 1732]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_07_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-9/30_text_IERCE_TraCKJUNI.png
try:
    _c30 = get_crop(30, 307, 41)
    canvas.paste(_c30, (114, 1925), _c30)
except Exception:
    pass
layout["IERCE_TraCKJUNI"] = [114, 1925, 421, 1966]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_07_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-9/31_text_Ecommerce_TracK_JUNI.png
try:
    _c31 = get_crop(31, 397, 39)
    canvas.paste(_c31, (436, 1927), _c31)
except Exception:
    pass
layout["Ecommerce_TracK_JUNI"] = [436, 1927, 833, 1966]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_07_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-9/32_text_Ecommerce_TracK_JUNI.png
try:
    _c32 = get_crop(32, 399, 39)
    canvas.paste(_c32, (848, 1927), _c32)
except Exception:
    pass
layout["Ecommerce_TracK__JUNI"] = [848, 1927, 1247, 1966]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_07_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-9/33_text_Eco.png
try:
    _c33 = get_crop(33, 64, 29)
    canvas.paste(_c33, (1260, 1931), _c33)
except Exception:
    pass
layout["Eco"] = [1260, 1931, 1324, 1960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_07_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-9/34_text_EZRA.png
try:
    _c34 = get_crop(34, 268, 97)
    canvas.paste(_c34, (690, 2089), _c34)
except Exception:
    pass
layout["EZRA"] = [690, 2089, 958, 2186]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_07_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-9/35_text_CEo_BoOmIby_CINdy_Joseph.png
try:
    _c35 = get_crop(35, 1344, 917)
    canvas.paste(_c35, (48, 1899), _c35)
except Exception:
    pass
layout["CEo,_BoOmIby_CINdy_Joseph"] = [48, 1899, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_07_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-9/36_text_Zipify_Apps.png
try:
    _c36 = get_crop(36, 1344, 917)
    canvas.paste(_c36, (48, 1899), _c36)
except Exception:
    pass
layout["{_Zipify_Apps"] = [48, 1899, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_07_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-9/37_text_ECOMMERCE_TracK_JUNI.png
try:
    _c37 = get_crop(37, 397, 38)
    canvas.paste(_c37, (154, 2501), _c37)
except Exception:
    pass
layout["ECOMMERCE_TracK_JUNI"] = [154, 2501, 551, 2539]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_07_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-9/38_text_EcoMMERCE_TracK.png
try:
    _c38 = get_crop(38, 1344, 917)
    canvas.paste(_c38, (48, 1899), _c38)
except Exception:
    pass
layout["EcoMMERCE_TracK"] = [48, 1899, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_07_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-9/39_text_JUNI.png
try:
    _c39 = get_crop(39, 119, 38)
    canvas.paste(_c39, (843, 2501), _c39)
except Exception:
    pass
layout["JUNI"] = [843, 2501, 962, 2539]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_07_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-9/40_text_Free.png
try:
    _c40 = get_crop(40, 80, 39)
    canvas.paste(_c40, (117, 2614), _c40)
except Exception:
    pass
layout["Free"] = [117, 2614, 197, 2653]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_07_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-9/41_text_Building_a_7_Figure_Ecommerce_Business_i.png
try:
    _c41 = get_crop(41, 1344, 917)
    canvas.paste(_c41, (48, 1899), _c41)
except Exception:
    pass
layout["Building_a_7_Figure_Ecomm"] = [48, 1899, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_07_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-9/42_text_2024.png
try:
    _c42 = get_crop(42, 288, 156)
    canvas.paste(_c42, (0, 2804), _c42)
except Exception:
    pass
layout["2024"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_07_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-9/43_clickable_Tickets.png
try:
    _c43 = get_crop(43, 288, 156)
    canvas.paste(_c43, (864, 2804), _c43)
except Exception:
    pass
layout["Tickets"] = [864, 2804, 1152, 2960]
