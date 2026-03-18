# page_id: page_eventbrite_182a9ea236924e9eb43ccbe82fd1506e_09
# screenshot: 2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-11.png
# step_index: 9/13
# task: Open Eventbrite. Set time to tomorrow. Clear all search filters. Select the third one in New York. Record its location and time in Google Keep Notes. Add it to Favorites.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw overall background (keep white as dominant color)
draw.rectangle([(0, 0), (1440, 2960)], fill="#FFFFFF")

# Status bar area (top ~72px) - light grey background
status_h = 72
draw.rectangle([(0, 0), (1440, status_h)], fill="#E6E6E6")
# subtle bottom border for status bar
draw.line([(0, status_h-1), (1440, status_h-1)], fill="#D0D0D0", width=1)

# Header / Search area background (spans under status bar)
search_top = status_h
search_height = 191  # matches the detected search crop height
draw.rectangle([(0, search_top), (1440, search_top + search_height)], fill="#FFFFFF")
# Divider below search area
divider_y = search_top + search_height
draw.line([(48, divider_y), (1392, divider_y)], fill="#E6E6E6", width=2)

# Filter chips row background area (subtle area to give structure)
filters_top = divider_y + 36
filters_bottom = filters_top + 140  # approximate area for chips and counts
# keep it white but provide a faint top and bottom separator to define the row
draw.line([(48, filters_top), (1392, filters_top)], fill="#F0F0F0", width=1)
draw.line([(48, filters_bottom), (1392, filters_bottom)], fill="#F0F0F0", width=1)

# Large first event card background (rounded rectangle)
card1_left = 48
card1_right = 48 + 1344
card1_top = filters_bottom + 24
card1_bottom = 1700  # leave space for second card below
card1_radius = 20
draw.rounded_rectangle(
    [(card1_left, card1_top), (card1_right, card1_bottom)],
    radius=card1_radius,
    fill="#FBFBFC",
    outline="#ECECEC",
    width=1
)

# Inside first card: area reserved for the event artwork (dark placeholder background)
# The artwork occupies the upper-mid portion of the card; we'll place a wide dark rectangle as the content background.
art1_top = card1_top + 40
art1_height = 440
art1_left = card1_left + 40
art1_right = card1_right - 40
art1_bottom = art1_top + art1_height
draw.rounded_rectangle(
    [(art1_left, art1_top), (art1_right, art1_bottom)],
    radius=16,
    fill="#EFEFF3",  # light neutral placeholder so real artwork can be pasted over
    outline="#E0E0E3",
    width=1
)

# Tag/badge background placeholder area inside first card (do not draw text)
badge_left = art1_left
badge_top = art1_bottom + 28
badge_width = 220
badge_height = 56
draw.rounded_rectangle(
    [(badge_left, badge_top), (badge_left + badge_width, badge_top + badge_height)],
    radius=28,
    fill="#F3EEF7"
)

# Separator line under the first card area (to separate from next content)
sep_y = card1_bottom + 20
draw.line([(48, sep_y), (1392, sep_y)], fill="#E6E6E6", width=1)

# Second event image card background (exact detected image area)
# Detected element: pos=(48,1839) size=1344x977
card2_left = 48
card2_top = 1839
card2_right = card2_left + 1344
card2_bottom = card2_top + 977
card2_radius = 16
# Draw an image placeholder background (darker to indicate image area)
draw.rounded_rectangle(
    [(card2_left, card2_top), (card2_right, card2_bottom)],
    radius=card2_radius,
    fill="#EDEDEF",
    outline="#E0E0E2",
    width=1
)

# A small light tag above the text block under card2 (placeholder background for "Going fast" badge)
badge2_left = card2_left + 12
badge2_top = card2_bottom + 22
badge2_w = 180
badge2_h = 52
draw.rounded_rectangle(
    [(badge2_left, badge2_top), (badge2_left + badge2_w, badge2_top + badge2_h)],
    radius=26,
    fill="#F7EEF2"
)

# Content area under second card: neutral background for text blocks
text_block_top = badge2_top + badge2_h + 18
text_block_left = card2_left
text_block_right = card2_right
text_block_bottom = text_block_top + 220
# Keep it white but draw a faint top divider
draw.rectangle([(text_block_left, text_block_top), (text_block_right, text_block_bottom)], fill="#FFFFFF")
draw.line([(text_block_left, text_block_top), (text_block_right, text_block_top)], fill="#F0F0F0", width=1)

# Global separators between major sections
for y in (divider_y, card1_bottom + 20, card2_bottom + 18):
    draw.line([(48, y), (1392, y)], fill="#F0F0F0", width=1)

# Bottom navigation bar area
nav_height = 120
nav_top = 2960 - nav_height
draw.rectangle([(0, nav_top), (1440, 2960)], fill="#FFFFFF")
# top border/shadow of bottom nav
draw.line([(0, nav_top), (1440, nav_top)], fill="#E6E6E6", width=2)

# Small home section background under nav icons (to visually anchor nav)
home_bar_h = 6
draw.rectangle([(640, nav_top + 12), (800, nav_top + 12 + home_bar_h)], fill="#FFEDE6")

# Final subtle outer frame (very light) to finish layout
draw.rectangle([(0, 0), (1439, 2959)], outline="#FFFFFF", width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_09_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-11/00_icon_Tomorrow.png
try:
    _c0 = get_crop(0, 432, 103)
    canvas.paste(_c0, (438, 410), _c0)
except Exception:
    pass
layout["Tomorrow"] = [438, 410, 870, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_09_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-11/01_icon_Music.png
try:
    _c1 = get_crop(1, 187, 103)
    canvas.paste(_c1, (882, 410), _c1)
except Exception:
    pass
layout["Music"] = [882, 410, 1069, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_09_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-11/02_icon_1_Filter.png
try:
    _c2 = get_crop(2, 372, 103)
    canvas.paste(_c2, (54, 410), _c2)
except Exception:
    pass
layout["1_Filter"] = [54, 410, 426, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_09_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-11/03_icon_Business.png
try:
    _c3 = get_crop(3, 241, 103)
    canvas.paste(_c3, (1081, 410), _c3)
except Exception:
    pass
layout["Business"] = [1081, 410, 1322, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_09_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-11/04_icon_Business.png
try:
    _c4 = get_crop(4, 100, 109)
    canvas.paste(_c4, (1328, 408), _c4)
except Exception:
    pass
layout["Business"] = [1328, 408, 1428, 517]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_09_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-11/05_icon_Favorite_button.png
try:
    _c5 = get_crop(5, 144, 144)
    canvas.paste(_c5, (1092, 2355), _c5)
except Exception:
    pass
layout["Favorite_button"] = [1092, 2355, 1236, 2499]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_09_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-11/06_icon_Favorite_button.png
try:
    _c6 = get_crop(6, 144, 144)
    canvas.paste(_c6, (1092, 1192), _c6)
except Exception:
    pass
layout["Favorite_button"] = [1092, 1192, 1236, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_09_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-11/07_icon_9.32.png
try:
    _c7 = get_crop(7, 118, 112)
    canvas.paste(_c7, (58, 114), _c7)
except Exception:
    pass
layout["9.32"] = [58, 114, 176, 226]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_09_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-11/08_icon_Overflow_menu_button.png
try:
    _c8 = get_crop(8, 144, 144)
    canvas.paste(_c8, (1236, 2355), _c8)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 2355, 1380, 2499]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_09_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-11/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 56, 62)
    canvas.paste(_c9, (247, 1), _c9)
except Exception:
    pass
layout["icon_9"] = [247, 1, 303, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_09_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-11/10_icon_9.32.png
try:
    _c10 = get_crop(10, 57, 63)
    canvas.paste(_c10, (180, 1), _c10)
except Exception:
    pass
layout["9.32"] = [180, 1, 237, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_09_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-11/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 89, 61)
    canvas.paste(_c11, (1208, 0), _c11)
except Exception:
    pass
layout["icon_11"] = [1208, 0, 1297, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_09_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-11/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 62, 61)
    canvas.paste(_c12, (1315, 0), _c12)
except Exception:
    pass
layout["icon_12"] = [1315, 0, 1377, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_09_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-11/13_icon_Search_forae.png
try:
    _c13 = get_crop(13, 57, 64)
    canvas.paste(_c13, (313, 0), _c13)
except Exception:
    pass
layout["Search_forae"] = [313, 0, 370, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_09_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-11/14_icon_9.32.png
try:
    _c14 = get_crop(14, 56, 65)
    canvas.paste(_c14, (114, 0), _c14)
except Exception:
    pass
layout["9.32"] = [114, 0, 170, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_09_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-11/15_icon_TheRealReal.png
try:
    _c15 = get_crop(15, 1344, 977)
    canvas.paste(_c15, (48, 1839), _c15)
except Exception:
    pass
layout["TheRealReal"] = [48, 1839, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_09_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-11/16_icon_Search_forae.png
try:
    _c16 = get_crop(16, 1344, 191)
    canvas.paste(_c16, (48, 72), _c16)
except Exception:
    pass
layout["Search_forae"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_09_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-11/17_icon_Fashion.png
try:
    _c17 = get_crop(17, 288, 156)
    canvas.paste(_c17, (288, 2804), _c17)
except Exception:
    pass
layout["Fashion"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_09_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-11/18_icon_Overflow_menu_button.png
try:
    _c18 = get_crop(18, 144, 144)
    canvas.paste(_c18, (1236, 1192), _c18)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 1192, 1380, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_09_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-11/19_icon_FOR.png
try:
    _c19 = get_crop(19, 1344, 1115)
    canvas.paste(_c19, (48, 676), _c19)
except Exception:
    pass
layout["FOR"] = [48, 676, 1392, 1791]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_09_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-11/20_icon_Going_fast.png
try:
    _c20 = get_crop(20, 275, 97)
    canvas.paste(_c20, (89, 2531), _c20)
except Exception:
    pass
layout["Going_fast"] = [89, 2531, 364, 2628]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_09_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-11/21_icon_Fashion.png
try:
    _c21 = get_crop(21, 288, 156)
    canvas.paste(_c21, (0, 2804), _c21)
except Exception:
    pass
layout["Fashion"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_09_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-11/22_icon_WBONY_TheRealReal_Sustainability_in.png
try:
    _c22 = get_crop(22, 288, 156)
    canvas.paste(_c22, (576, 2804), _c22)
except Exception:
    pass
layout["WBONY_&_TheRealReal_Susta"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_09_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-11/23_icon_Search_forae.png
try:
    _c23 = get_crop(23, 48, 62)
    canvas.paste(_c23, (383, 1), _c23)
except Exception:
    pass
layout["Search_forae"] = [383, 1, 431, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_09_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-11/24_icon_New_York.png
try:
    _c24 = get_crop(24, 434, 144)
    canvas.paste(_c24, (0, 259), _c24)
except Exception:
    pass
layout["New_York"] = [0, 259, 434, 403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_09_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-11/25_icon_WBONY_TheRealReal_Sustainability_in.png
try:
    _c25 = get_crop(25, 288, 156)
    canvas.paste(_c25, (864, 2804), _c25)
except Exception:
    pass
layout["WBONY_&_TheRealReal_Susta"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_09_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-11/26_icon_More.png
try:
    _c26 = get_crop(26, 288, 156)
    canvas.paste(_c26, (1152, 2804), _c26)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_09_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-11/27_icon_icon_27.png
try:
    _c27 = get_crop(27, 39, 61)
    canvas.paste(_c27, (1275, 0), _c27)
except Exception:
    pass
layout["icon_27"] = [1275, 0, 1314, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_09_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-11/28_icon_7_00_PM_EDT.png
try:
    _c28 = get_crop(28, 1344, 1115)
    canvas.paste(_c28, (48, 676), _c28)
except Exception:
    pass
layout["7:00_PM_EDT"] = [48, 676, 1392, 1791]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_09_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-11/29_icon_Oulala_Cafe_and_Lounge.png
try:
    _c29 = get_crop(29, 48, 59)
    canvas.paste(_c29, (280, 1690), _c29)
except Exception:
    pass
layout["Oulala_Cafe_and_Lounge"] = [280, 1690, 328, 1749]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_09_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-11/30_text_9.32.png
try:
    _c30 = get_crop(30, 96, 49)
    canvas.paste(_c30, (16, 12), _c30)
except Exception:
    pass
layout["9.32"] = [16, 12, 112, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_09_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-11/31_text_2_412_events.png
try:
    _c31 = get_crop(31, 372, 103)
    canvas.paste(_c31, (54, 410), _c31)
except Exception:
    pass
layout["2,412_events"] = [54, 410, 426, 513]
