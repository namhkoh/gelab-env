# page_id: page_eventbrite_182a9ea236924e9eb43ccbe82fd1506e_07
# screenshot: 2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-9.png
# step_index: 7/13
# task: Open Eventbrite. Set time to tomorrow. Clear all search filters. Select the third one in New York. Record its location and time in Google Keep Notes. Add it to Favorites.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw status bar
status_h = 72
draw.rectangle([(0, 0), (1440, status_h)], fill="#bfbfbf")  # top status bar (gray)

# Subtle top hairline
draw.line([(0, status_h), (1440, status_h)], fill="#9a9a9a", width=1)

# Header / search area background (leave content area white but add a faint divider)
header_top = status_h
header_bottom = 232
draw.rectangle([(0, header_top), (1440, header_bottom)], fill="#ffffff")
draw.line([(48, header_bottom), (1392, header_bottom)], fill="#e0e0e0", width=2)

# Filter / chips area background band (very subtle)
filters_top = header_bottom + 12
filters_bottom = filters_top + 120
draw.rectangle([(0, filters_top), (1440, filters_bottom)], fill="#ffffff")
draw.line([(48, filters_bottom), (1392, filters_bottom)], fill="#ececec", width=1)

# Main content card 1 (large white card with soft shadow)
card_x0, card_x1 = 48, 1392
card1_y0, card1_y1 = 300, 1440
shadow_offset = 8
# shadow
draw.rounded_rectangle(
    [(card_x0 + shadow_offset, card1_y0 + shadow_offset), (card_x1 + shadow_offset, card1_y1 + shadow_offset)],
    radius=26,
    fill="#f5f5f5",
)
# card
draw.rounded_rectangle([(card_x0, card1_y0), (card_x1, card1_y1)], radius=24, fill="#ffffff", outline="#e6e6e6", width=1)

# Separator line under first card
sep_y = card1_y1 + 16
draw.line([(48, sep_y), (1392, sep_y)], fill="#ececec", width=1)

# Image/content card area (dark backdrop behind image region)
img_card_y0, img_card_y1 = 1680, 2680
# subtle shadow
draw.rounded_rectangle(
    [(card_x0 + shadow_offset, img_card_y0 + shadow_offset), (card_x1 + shadow_offset, img_card_y1 + shadow_offset)],
    radius=20,
    fill="#f5f5f5",
)
# dark background area for media
draw.rounded_rectangle([(card_x0, img_card_y0), (card_x1, img_card_y1)], radius=20, fill="#111111", outline="#e6e6e6", width=1)

# Small light card for promoted/label area under media (background band)
band_y0 = img_card_y1 + 32
band_y1 = band_y0 + 160
draw.rectangle([(card_x0, band_y0), (card_x1, band_y1)], fill="#ffffff")
draw.line([(card_x0, band_y1), (card_x1, band_y1)], fill="#ececec", width=1)

# Bottom navigation bar background
nav_top = 2820
nav_bottom = 2960
draw.rectangle([(0, nav_top), (1440, nav_bottom)], fill="#ffffff")
draw.line([(0, nav_top), (1440, nav_top)], fill="#e0e0e0", width=2)

# Additional subtle separators for content flow
draw.line([(48, 1600), (1392, 1600)], fill="#f0f0f0", width=1)
draw.line([(48, 1520), (1392, 1520)], fill="#f7f7f7", width=1)

# Gentle vignette/shadow near cards to lift them (soft rectangular strokes)
# top card inner top shadow
draw.line([(card_x0 + 8, card1_y0 + 2), (card_x1 - 8, card1_y0 + 2)], fill="#fbfbfb", width=2)
# image card inner top highlight
draw.line([(card_x0 + 8, img_card_y0 + 2), (card_x1 - 8, img_card_y0 + 2)], fill="#0b0b0b", width=2)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_07_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-9/00_icon_Tomorrow.png
try:
    _c0 = get_crop(0, 432, 103)
    canvas.paste(_c0, (438, 410), _c0)
except Exception:
    pass
layout["Tomorrow"] = [438, 410, 870, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_07_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-9/01_icon_Music.png
try:
    _c1 = get_crop(1, 187, 103)
    canvas.paste(_c1, (882, 410), _c1)
except Exception:
    pass
layout["Music"] = [882, 410, 1069, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_07_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-9/02_icon_1_Filter.png
try:
    _c2 = get_crop(2, 372, 103)
    canvas.paste(_c2, (54, 410), _c2)
except Exception:
    pass
layout["1_Filter"] = [54, 410, 426, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_07_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-9/03_icon_Business.png
try:
    _c3 = get_crop(3, 241, 103)
    canvas.paste(_c3, (1081, 410), _c3)
except Exception:
    pass
layout["Business"] = [1081, 410, 1322, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_07_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-9/04_icon_Business.png
try:
    _c4 = get_crop(4, 100, 109)
    canvas.paste(_c4, (1328, 408), _c4)
except Exception:
    pass
layout["Business"] = [1328, 408, 1428, 517]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_07_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-9/05_icon_Favorite_button.png
try:
    _c5 = get_crop(5, 144, 144)
    canvas.paste(_c5, (1092, 2355), _c5)
except Exception:
    pass
layout["Favorite_button"] = [1092, 2355, 1236, 2499]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_07_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-9/06_icon_Favorite_button.png
try:
    _c6 = get_crop(6, 144, 144)
    canvas.paste(_c6, (1092, 1192), _c6)
except Exception:
    pass
layout["Favorite_button"] = [1092, 1192, 1236, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_07_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-9/07_icon_9.32.png
try:
    _c7 = get_crop(7, 118, 112)
    canvas.paste(_c7, (58, 114), _c7)
except Exception:
    pass
layout["9.32"] = [58, 114, 176, 226]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_07_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-9/08_icon_Overflow_menu_button.png
try:
    _c8 = get_crop(8, 144, 144)
    canvas.paste(_c8, (1236, 2355), _c8)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 2355, 1380, 2499]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_07_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-9/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 56, 62)
    canvas.paste(_c9, (247, 1), _c9)
except Exception:
    pass
layout["icon_9"] = [247, 1, 303, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_07_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-9/10_icon_9.32.png
try:
    _c10 = get_crop(10, 57, 63)
    canvas.paste(_c10, (180, 1), _c10)
except Exception:
    pass
layout["9.32"] = [180, 1, 237, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_07_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-9/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 89, 61)
    canvas.paste(_c11, (1208, 0), _c11)
except Exception:
    pass
layout["icon_11"] = [1208, 0, 1297, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_07_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-9/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 62, 61)
    canvas.paste(_c12, (1315, 0), _c12)
except Exception:
    pass
layout["icon_12"] = [1315, 0, 1377, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_07_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-9/13_icon_Search_forae.png
try:
    _c13 = get_crop(13, 57, 64)
    canvas.paste(_c13, (313, 0), _c13)
except Exception:
    pass
layout["Search_forae"] = [313, 0, 370, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_07_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-9/14_icon_9.32.png
try:
    _c14 = get_crop(14, 56, 65)
    canvas.paste(_c14, (114, 0), _c14)
except Exception:
    pass
layout["9.32"] = [114, 0, 170, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_07_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-9/15_icon_TheRealReal.png
try:
    _c15 = get_crop(15, 1344, 977)
    canvas.paste(_c15, (48, 1839), _c15)
except Exception:
    pass
layout["TheRealReal"] = [48, 1839, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_07_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-9/16_icon_Search_forae.png
try:
    _c16 = get_crop(16, 1344, 191)
    canvas.paste(_c16, (48, 72), _c16)
except Exception:
    pass
layout["Search_forae"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_07_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-9/17_icon_Fashion.png
try:
    _c17 = get_crop(17, 288, 156)
    canvas.paste(_c17, (288, 2804), _c17)
except Exception:
    pass
layout["Fashion"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_07_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-9/18_icon_Overflow_menu_button.png
try:
    _c18 = get_crop(18, 144, 144)
    canvas.paste(_c18, (1236, 1192), _c18)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 1192, 1380, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_07_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-9/19_icon_FOR.png
try:
    _c19 = get_crop(19, 1344, 1115)
    canvas.paste(_c19, (48, 676), _c19)
except Exception:
    pass
layout["FOR"] = [48, 676, 1392, 1791]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_07_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-9/20_icon_Going_fast.png
try:
    _c20 = get_crop(20, 275, 97)
    canvas.paste(_c20, (89, 2531), _c20)
except Exception:
    pass
layout["Going_fast"] = [89, 2531, 364, 2628]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_07_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-9/21_icon_Fashion.png
try:
    _c21 = get_crop(21, 288, 156)
    canvas.paste(_c21, (0, 2804), _c21)
except Exception:
    pass
layout["Fashion"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_07_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-9/22_icon_WBONY_TheRealReal_Sustainability_in.png
try:
    _c22 = get_crop(22, 288, 156)
    canvas.paste(_c22, (576, 2804), _c22)
except Exception:
    pass
layout["WBONY_&_TheRealReal_Susta"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_07_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-9/23_icon_Search_forae.png
try:
    _c23 = get_crop(23, 48, 62)
    canvas.paste(_c23, (383, 1), _c23)
except Exception:
    pass
layout["Search_forae"] = [383, 1, 431, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_07_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-9/24_icon_New_York.png
try:
    _c24 = get_crop(24, 434, 144)
    canvas.paste(_c24, (0, 259), _c24)
except Exception:
    pass
layout["New_York"] = [0, 259, 434, 403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_07_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-9/25_icon_WBONY_TheRealReal_Sustainability_in.png
try:
    _c25 = get_crop(25, 288, 156)
    canvas.paste(_c25, (864, 2804), _c25)
except Exception:
    pass
layout["WBONY_&_TheRealReal_Susta"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_07_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-9/26_icon_More.png
try:
    _c26 = get_crop(26, 288, 156)
    canvas.paste(_c26, (1152, 2804), _c26)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_07_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-9/27_icon_icon_27.png
try:
    _c27 = get_crop(27, 39, 61)
    canvas.paste(_c27, (1275, 0), _c27)
except Exception:
    pass
layout["icon_27"] = [1275, 0, 1314, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_07_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-9/28_icon_7_00_PM_EDT.png
try:
    _c28 = get_crop(28, 1344, 1115)
    canvas.paste(_c28, (48, 676), _c28)
except Exception:
    pass
layout["7:00_PM_EDT"] = [48, 676, 1392, 1791]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_07_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-9/29_icon_Oulala_Cafe_and_Lounge.png
try:
    _c29 = get_crop(29, 48, 59)
    canvas.paste(_c29, (280, 1690), _c29)
except Exception:
    pass
layout["Oulala_Cafe_and_Lounge"] = [280, 1690, 328, 1749]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_07_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-9/30_text_9.32.png
try:
    _c30 = get_crop(30, 96, 49)
    canvas.paste(_c30, (16, 12), _c30)
except Exception:
    pass
layout["9.32"] = [16, 12, 112, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_07_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-9/31_text_2_412_events.png
try:
    _c31 = get_crop(31, 372, 103)
    canvas.paste(_c31, (54, 410), _c31)
except Exception:
    pass
layout["2,412_events"] = [54, 410, 426, 513]
