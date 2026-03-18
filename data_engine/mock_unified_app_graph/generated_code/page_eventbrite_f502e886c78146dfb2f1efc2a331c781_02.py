# page_id: page_eventbrite_f502e886c78146dfb2f1efc2a331c781_02
# screenshot: 2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-4.png
# step_index: 2/18
# task: Open Eventbrite. Search for 'music festival' in San Francisco. Set date available from April 30 to May 3. How many events are listed?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Background fill
bg_color = "#F6F7F9"
status_color = "#BDBDBD"
header_bg = "#FFFFFF"
divider_color = "#E1E1E1"
card_shadow = "#E9E9EA"
card_bg = "#FFFFFF"
image_placeholder_light = "#F0F1F3"
image_placeholder_dark = "#111214"
bottom_nav_bg = "#FFFFFF"

# Fill whole canvas
draw.rectangle([(0, 0), (1440, 2960)], fill=bg_color)

# Status bar area (top ~72px)
draw.rectangle([(0, 0), (1440, 72)], fill=status_color)

# Header / search area background (below status bar)
draw.rectangle([(0, 72), (1440, 420)], fill=header_bg)

# Thin divider under header / search area
draw.line([(40, 420), (1400, 420)], fill=divider_color, width=2)

# Another subtle divider above content area
draw.line([(40, 520), (1400, 520)], fill=divider_color, width=1)

# First event card (rounded) with subtle shadow behind it
card1_x0, card1_y0 = 32, 620
card1_x1, card1_y1 = 1408, 1900
# shadow
draw.rounded_rectangle(
    [(card1_x0 + 8, card1_y0 + 8), (card1_x1 + 8, card1_y1 + 8)],
    radius=22,
    fill=card_shadow
)
# card background
draw.rounded_rectangle(
    [(card1_x0, card1_y0), (card1_x1, card1_y1)],
    radius=20,
    fill=card_bg
)

# Background for the large event image (first)
img1_x0, img1_y0 = 48, 676
img1_x1, img1_y1 = 1392, 1851  # matches detected image crop extents
draw.rounded_rectangle(
    [(img1_x0, img1_y0), (img1_x1, img1_y1)],
    radius=14,
    fill=image_placeholder_light
)

# Separator between cards
sep_y = 1848
draw.line([(40, sep_y), (1400, sep_y)], fill=divider_color, width=2)

# Second event card (rounded) with shadow
card2_x0, card2_y0 = 32, 1840
card2_x1, card2_y1 = 1408, 2868
draw.rounded_rectangle(
    [(card2_x0 + 8, card2_y0 + 8), (card2_x1 + 8, card2_y1 + 8)],
    radius=22,
    fill=card_shadow
)
draw.rounded_rectangle(
    [(card2_x0, card2_y0), (card2_x1, card2_y1)],
    radius=20,
    fill=card_bg
)

# Background for the large event image (second) - darker banner area
img2_x0, img2_y0 = 48, 1899
img2_x1, img2_y1 = 1392, 2816  # matches detected image crop extents
draw.rounded_rectangle(
    [(img2_x0, img2_y0), (img2_x1, img2_y1)],
    radius=14,
    fill=image_placeholder_dark
)

# Thin divider above bottom navigation
nav_top = 2720
draw.line([(0, nav_top), (1440, nav_top)], fill=divider_color, width=2)

# Bottom navigation bar background
draw.rectangle([(0, nav_top), (1440, 2960)], fill=bottom_nav_bg)

# Subtle top shadow on bottom nav
draw.line([(0, nav_top + 1), (1440, nav_top + 1)], fill="#F2F2F3", width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_02_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-4/00_icon_Music.png
try:
    _c0 = get_crop(0, 187, 103)
    canvas.paste(_c0, (837, 410), _c0)
except Exception:
    pass
layout["Music"] = [837, 410, 1024, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_02_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-4/01_icon_Business.png
try:
    _c1 = get_crop(1, 241, 103)
    canvas.paste(_c1, (1036, 410), _c1)
except Exception:
    pass
layout["Business"] = [1036, 410, 1277, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_02_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-4/02_icon_Anytime.png
try:
    _c2 = get_crop(2, 400, 103)
    canvas.paste(_c2, (425, 410), _c2)
except Exception:
    pass
layout["Anytime"] = [425, 410, 825, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_02_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-4/03_icon_Filters.png
try:
    _c3 = get_crop(3, 359, 103)
    canvas.paste(_c3, (54, 410), _c3)
except Exception:
    pass
layout["Filters"] = [54, 410, 413, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_02_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-4/04_icon_Foo.png
try:
    _c4 = get_crop(4, 149, 110)
    canvas.paste(_c4, (1283, 406), _c4)
except Exception:
    pass
layout["Foo"] = [1283, 406, 1432, 516]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_02_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-4/05_icon_Favorite_button.png
try:
    _c5 = get_crop(5, 144, 144)
    canvas.paste(_c5, (1092, 1192), _c5)
except Exception:
    pass
layout["Favorite_button"] = [1092, 1192, 1236, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_02_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-4/06_icon_JNDRE_Roto.png
try:
    _c6 = get_crop(6, 144, 144)
    canvas.paste(_c6, (1092, 2415), _c6)
except Exception:
    pass
layout["JNDRE_Roto"] = [1092, 2415, 1236, 2559]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_02_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-4/07_icon_Overflow_menu_button.png
try:
    _c7 = get_crop(7, 144, 144)
    canvas.paste(_c7, (1236, 1192), _c7)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 1192, 1380, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_02_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-4/08_icon_JNDRE_Roto.png
try:
    _c8 = get_crop(8, 144, 144)
    canvas.paste(_c8, (1236, 2415), _c8)
except Exception:
    pass
layout["JNDRE_Roto"] = [1236, 2415, 1380, 2559]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_02_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-4/09_icon_Tom_Everhart_at_the_Grand_Opening_of_Cho.png
try:
    _c9 = get_crop(9, 1344, 1175)
    canvas.paste(_c9, (48, 676), _c9)
except Exception:
    pass
layout["Tom_Everhart_at_the_Grand"] = [48, 676, 1392, 1851]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_02_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-4/10_icon_7.18.png
try:
    _c10 = get_crop(10, 125, 111)
    canvas.paste(_c10, (55, 117), _c10)
except Exception:
    pass
layout["7.18"] = [55, 117, 180, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_02_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-4/11_icon_Search_forae.png
try:
    _c11 = get_crop(11, 1344, 191)
    canvas.paste(_c11, (48, 72), _c11)
except Exception:
    pass
layout["Search_forae"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_02_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-4/12_icon_Search_forae.png
try:
    _c12 = get_crop(12, 69, 63)
    canvas.paste(_c12, (307, 0), _c12)
except Exception:
    pass
layout["Search_forae"] = [307, 0, 376, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_02_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-4/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 54, 64)
    canvas.paste(_c13, (246, 0), _c13)
except Exception:
    pass
layout["icon_13"] = [246, 0, 300, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_02_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-4/14_icon_7.18.png
try:
    _c14 = get_crop(14, 61, 63)
    canvas.paste(_c14, (181, 0), _c14)
except Exception:
    pass
layout["7.18"] = [181, 0, 242, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_02_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-4/15_icon_7.18.png
try:
    _c15 = get_crop(15, 62, 65)
    canvas.paste(_c15, (114, 0), _c15)
except Exception:
    pass
layout["7.18"] = [114, 0, 176, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_02_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-4/16_icon_icon_16.png
try:
    _c16 = get_crop(16, 86, 60)
    canvas.paste(_c16, (1207, 0), _c16)
except Exception:
    pass
layout["icon_16"] = [1207, 0, 1293, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_02_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-4/17_icon_Sun_Apr_28_._5.00_PM_PDT.png
try:
    _c17 = get_crop(17, 288, 156)
    canvas.paste(_c17, (576, 2804), _c17)
except Exception:
    pass
layout["Sun,_Apr_28_._5.00_PM_PDT"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_02_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-4/18_icon_icon_18.png
try:
    _c18 = get_crop(18, 64, 59)
    canvas.paste(_c18, (1315, 0), _c18)
except Exception:
    pass
layout["icon_18"] = [1315, 0, 1379, 59]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_02_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-4/19_icon_Los_Angeles.png
try:
    _c19 = get_crop(19, 492, 144)
    canvas.paste(_c19, (0, 259), _c19)
except Exception:
    pass
layout["Los_Angeles"] = [0, 259, 492, 403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_02_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-4/20_icon_Search_forae.png
try:
    _c20 = get_crop(20, 52, 61)
    canvas.paste(_c20, (383, 2), _c20)
except Exception:
    pass
layout["Search_forae"] = [383, 2, 435, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_02_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-4/21_icon_Iet.png
try:
    _c21 = get_crop(21, 288, 156)
    canvas.paste(_c21, (864, 2804), _c21)
except Exception:
    pass
layout["Iet"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_02_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-4/22_icon_Regal_LA_Live.png
try:
    _c22 = get_crop(22, 288, 156)
    canvas.paste(_c22, (288, 2804), _c22)
except Exception:
    pass
layout["Regal_LA_Live"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_02_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-4/23_icon_JNDRE_Roto.png
try:
    _c23 = get_crop(23, 148, 153)
    canvas.paste(_c23, (956, 2393), _c23)
except Exception:
    pass
layout["JNDRE_Roto"] = [956, 2393, 1104, 2546]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_02_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-4/24_icon_Iet.png
try:
    _c24 = get_crop(24, 288, 156)
    canvas.paste(_c24, (1152, 2804), _c24)
except Exception:
    pass
layout["Iet"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_02_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-4/25_icon_Tom_Everhart_at_the_Grand_Opening_of_Cho.png
try:
    _c25 = get_crop(25, 1344, 1175)
    canvas.paste(_c25, (48, 676), _c25)
except Exception:
    pass
layout["Tom_Everhart_at_the_Grand"] = [48, 676, 1392, 1851]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_02_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-4/26_icon_Promoted.png
try:
    _c26 = get_crop(26, 242, 66)
    canvas.paste(_c26, (85, 1744), _c26)
except Exception:
    pass
layout["Promoted"] = [85, 1744, 327, 1810]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_02_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-4/27_icon_icon_27.png
try:
    _c27 = get_crop(27, 39, 60)
    canvas.paste(_c27, (1275, 0), _c27)
except Exception:
    pass
layout["icon_27"] = [1275, 0, 1314, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_02_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-4/28_icon_Regal_LA_Live.png
try:
    _c28 = get_crop(28, 288, 156)
    canvas.paste(_c28, (0, 2804), _c28)
except Exception:
    pass
layout["Regal_LA_Live"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_02_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-4/29_icon_InTERNATIONAL.png
try:
    _c29 = get_crop(29, 1344, 917)
    canvas.paste(_c29, (48, 1899), _c29)
except Exception:
    pass
layout["InTERNATIONAL"] = [48, 1899, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_02_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-4/30_icon_Beyond_Hollywood_Int_I_Film_Festival_202.png
try:
    _c30 = get_crop(30, 1344, 917)
    canvas.paste(_c30, (48, 1899), _c30)
except Exception:
    pass
layout["Beyond_Hollywood_Int'I_Fi"] = [48, 1899, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_02_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-4/31_icon_7.18.png
try:
    _c31 = get_crop(31, 135, 64)
    canvas.paste(_c31, (6, 0), _c31)
except Exception:
    pass
layout["7.18"] = [6, 0, 141, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_02_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-4/32_text_10_000_events.png
try:
    _c32 = get_crop(32, 359, 103)
    canvas.paste(_c32, (54, 410), _c32)
except Exception:
    pass
layout["10,000_events"] = [54, 410, 413, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_02_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-4/33_text_313_N_Beverly_Dr.png
try:
    _c33 = get_crop(33, 323, 55)
    canvas.paste(_c33, (90, 1686), _c33)
except Exception:
    pass
layout["313_N_Beverly_Dr"] = [90, 1686, 413, 1741]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_02_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-4/34_text_Thu_Apr_25.png
try:
    _c34 = get_crop(34, 230, 52)
    canvas.paste(_c34, (93, 2680), _c34)
except Exception:
    pass
layout["Thu,_Apr_25"] = [93, 2680, 323, 2732]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_02_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-4/35_text_Sun_Apr_28_._5.00_PM_PDT.png
try:
    _c35 = get_crop(35, 1344, 917)
    canvas.paste(_c35, (48, 1899), _c35)
except Exception:
    pass
layout["Sun,_Apr_28_._5.00_PM_PDT"] = [48, 1899, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_02_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-4/36_text_Regal_LA_Live.png
try:
    _c36 = get_crop(36, 288, 156)
    canvas.paste(_c36, (0, 2804), _c36)
except Exception:
    pass
layout["Regal_LA_Live"] = [0, 2804, 288, 2960]
