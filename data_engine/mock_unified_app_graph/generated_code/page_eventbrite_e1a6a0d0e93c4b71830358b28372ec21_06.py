# page_id: page_eventbrite_e1a6a0d0e93c4b71830358b28372ec21_06
# screenshot: 2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-8.png
# step_index: 6/9
# task: Open Eventbrite. Search for "Language Learning". Filter only online events. Note how many events are available for "Spanish".
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Top-level background fill (match dominant off-white of screenshot)
draw.rectangle((0, 0, 1440, 2960), fill=(250, 250, 252))

# Status bar (top ~60px) - light grey
status_h = 60
draw.rectangle((0, 0, 1440, status_h), fill=(190, 190, 190))

# Subtle divider under status bar
draw.line((0, status_h, 1440, status_h), fill=(210, 210, 210), width=1)

# Header area (search/title bar) - white with bottom divider
header_y0 = status_h
header_y1 = 150
draw.rectangle((0, header_y0, 1440, header_y1), fill=(255, 255, 255))
draw.line((24, header_y1, 1440-24, header_y1), fill=(225, 225, 225), width=1)

# Light background band behind the filter chips row (very pale blue)
# Chips themselves will be pasted on top at y ~ 410, so this acts as a backdrop
chips_band_y0 = 372
chips_band_y1 = 472
draw.rectangle((0, chips_band_y0, 1440, chips_band_y1), fill=(245, 251, 255))

# Main content area subtle horizontal rule under filters/title
draw.line((24, chips_band_y1 + 6, 1440-24, chips_band_y1 + 6), fill=(235, 235, 238), width=1)

# First event card: rounded white card with soft shadow
card1_x0 = 36
card1_x1 = 1440 - 36
card1_y0 = 640
card1_y1 = 1888
radius = 28

# Shadow (slightly offset and soft color)
shadow_offset = 8
draw.rounded_rectangle(
    (card1_x0 + shadow_offset, card1_y0 + shadow_offset, card1_x1 + shadow_offset, card1_y1 + shadow_offset),
    radius=radius, fill=(230, 230, 235)
)

# Card background
draw.rounded_rectangle((card1_x0, card1_y0, card1_x1, card1_y1), radius=radius, fill=(255, 255, 255), outline=(235, 235, 240))

# Thin divider/edge inside card near top (to separate image area from meta)
draw.line((card1_x0 + 24, card1_y0 + 420, card1_x1 - 24, card1_y0 + 420), fill=(245, 245, 247), width=1)

# Separator between first and second card (a faint line)
sep_y = card1_y1 + 8
draw.line((24, sep_y, 1440-24, sep_y), fill=(232, 232, 235), width=1)

# Second event card: rounded white card with soft shadow
card2_x0 = 36
card2_x1 = 1440 - 36
card2_y0 = card1_y1 + 32
card2_y1 = 2820
draw.rounded_rectangle(
    (card2_x0 + shadow_offset, card2_y0 + shadow_offset, card2_x1 + shadow_offset, card2_y1 + shadow_offset),
    radius=radius, fill=(230, 230, 235)
)
draw.rounded_rectangle((card2_x0, card2_y0, card2_x1, card2_y1), radius=radius, fill=(255, 255, 255), outline=(235, 235, 240))

# Subtle horizontal separators within second card (for image/meta spacing)
draw.line((card2_x0 + 24, card2_y0 + 300, card2_x1 - 24, card2_y0 + 300), fill=(245, 245, 247), width=1)

# Floating content area accent: a pale green small pill background location example (for "Free" badge background)
# NOTE: Do not draw any text; only the subtle rounded rect backdrop where the badge sits.
badge_x0 = 96
badge_y0 = 1360
badge_x1 = badge_x0 + 120
badge_y1 = badge_y0 + 46
draw.rounded_rectangle((badge_x0, badge_y0, badge_x1, badge_y1), radius=12, fill=(236, 247, 238), outline=(215, 232, 218))

# Another badge backdrop lower on page
badge2_x0 = 96
badge2_y0 = 2608
badge2_x1 = badge2_x0 + 120
badge2_y1 = badge2_y0 + 48
draw.rounded_rectangle((badge2_x0, badge2_y0, badge2_x1, badge2_y1), radius=12, fill=(236, 247, 238), outline=(215, 232, 218))

# Bottom navigation bar background (separate from content) - keep above extreme bottom so content icons can be pasted
nav_h = 140
nav_y0 = 2960 - nav_h
draw.rectangle((0, nav_y0, 1440, 2960), fill=(255, 255, 255))
# Top divider for nav bar
draw.line((24, nav_y0, 1440-24, nav_y0), fill=(225, 225, 228), width=1)

# Slight inner shadow at very bottom to anchor the nav
draw.line((24, 2960 - 1, 1440 - 24, 2960 - 1), fill=(240, 240, 242), width=1)

# Small left and right page margins subtle guides (not content)
draw.line((24, header_y0, 24, 2960 - nav_h), fill=(250, 250, 251), width=1)
draw.line((1440-24, header_y0, 1440-24, 2960 - nav_h), fill=(250, 250, 251), width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_06_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-8/00_icon_Music.png
try:
    _c0 = get_crop(0, 187, 103)
    canvas.paste(_c0, (837, 410), _c0)
except Exception:
    pass
layout["Music"] = [837, 410, 1024, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_06_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-8/01_icon_Anytime.png
try:
    _c1 = get_crop(1, 400, 103)
    canvas.paste(_c1, (425, 410), _c1)
except Exception:
    pass
layout["Anytime"] = [425, 410, 825, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_06_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-8/02_icon_Business.png
try:
    _c2 = get_crop(2, 241, 103)
    canvas.paste(_c2, (1036, 410), _c2)
except Exception:
    pass
layout["Business"] = [1036, 410, 1277, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_06_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-8/03_icon_Filters.png
try:
    _c3 = get_crop(3, 359, 103)
    canvas.paste(_c3, (54, 410), _c3)
except Exception:
    pass
layout["Filters"] = [54, 410, 413, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_06_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-8/04_icon_Foo.png
try:
    _c4 = get_crop(4, 139, 110)
    canvas.paste(_c4, (1284, 406), _c4)
except Exception:
    pass
layout["Foo"] = [1284, 406, 1423, 516]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_06_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-8/05_icon_EcoMmcR.png
try:
    _c5 = get_crop(5, 144, 144)
    canvas.paste(_c5, (1092, 1192), _c5)
except Exception:
    pass
layout["EcoMmcR"] = [1092, 1192, 1236, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_06_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-8/06_icon_EcoMmcR.png
try:
    _c6 = get_crop(6, 144, 144)
    canvas.paste(_c6, (1236, 1192), _c6)
except Exception:
    pass
layout["EcoMmcR"] = [1236, 1192, 1380, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_06_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-8/07_icon_Close_current_screen.png
try:
    _c7 = get_crop(7, 144, 144)
    canvas.paste(_c7, (1248, 96), _c7)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 96, 1392, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_06_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-8/08_icon_0_Zipify_Apps.png
try:
    _c8 = get_crop(8, 1344, 1175)
    canvas.paste(_c8, (48, 676), _c8)
except Exception:
    pass
layout["0_Zipify_Apps"] = [48, 676, 1392, 1851]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_06_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-8/09_icon_5.18.png
try:
    _c9 = get_crop(9, 130, 121)
    canvas.paste(_c9, (52, 110), _c9)
except Exception:
    pass
layout["5.18"] = [52, 110, 182, 231]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_06_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-8/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 103, 61)
    canvas.paste(_c10, (1208, 0), _c10)
except Exception:
    pass
layout["icon_10"] = [1208, 0, 1311, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_06_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-8/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 64, 61)
    canvas.paste(_c11, (309, 1), _c11)
except Exception:
    pass
layout["icon_11"] = [309, 1, 373, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_06_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-8/12_icon_5.18.png
try:
    _c12 = get_crop(12, 58, 63)
    canvas.paste(_c12, (182, 0), _c12)
except Exception:
    pass
layout["5.18"] = [182, 0, 240, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_06_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-8/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 53, 62)
    canvas.paste(_c13, (246, 1), _c13)
except Exception:
    pass
layout["icon_13"] = [246, 1, 299, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_06_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-8/14_icon_Language_Learning.png
try:
    _c14 = get_crop(14, 1344, 191)
    canvas.paste(_c14, (48, 72), _c14)
except Exception:
    pass
layout["Language_Learning"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_06_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-8/15_icon_5.18.png
try:
    _c15 = get_crop(15, 58, 64)
    canvas.paste(_c15, (116, 0), _c15)
except Exception:
    pass
layout["5.18"] = [116, 0, 174, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_06_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-8/16_icon_icon_16.png
try:
    _c16 = get_crop(16, 60, 61)
    canvas.paste(_c16, (1318, 0), _c16)
except Exception:
    pass
layout["icon_16"] = [1318, 0, 1378, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_06_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-8/17_icon_Online.png
try:
    _c17 = get_crop(17, 377, 144)
    canvas.paste(_c17, (0, 259), _c17)
except Exception:
    pass
layout["Online"] = [0, 259, 377, 403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_06_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-8/18_icon_3_00_PM_EDT.png
try:
    _c18 = get_crop(18, 288, 156)
    canvas.paste(_c18, (288, 2804), _c18)
except Exception:
    pass
layout["3:00_PM_EDT"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_06_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-8/19_icon_icon_19.png
try:
    _c19 = get_crop(19, 49, 60)
    canvas.paste(_c19, (384, 3), _c19)
except Exception:
    pass
layout["icon_19"] = [384, 3, 433, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_06_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-8/20_icon_Algebra.png
try:
    _c20 = get_crop(20, 144, 144)
    canvas.paste(_c20, (1092, 2415), _c20)
except Exception:
    pass
layout["Algebra"] = [1092, 2415, 1236, 2559]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_06_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-8/21_icon_Algebra.png
try:
    _c21 = get_crop(21, 144, 144)
    canvas.paste(_c21, (1236, 2415), _c21)
except Exception:
    pass
layout["Algebra"] = [1236, 2415, 1380, 2559]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_06_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-8/22_icon_Promoted.png
try:
    _c22 = get_crop(22, 241, 64)
    canvas.paste(_c22, (86, 1743), _c22)
except Exception:
    pass
layout["Promoted"] = [86, 1743, 327, 1807]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_06_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-8/23_icon_Tue_Apr_30.png
try:
    _c23 = get_crop(23, 288, 156)
    canvas.paste(_c23, (0, 2804), _c23)
except Exception:
    pass
layout["Tue,_Apr_30"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_06_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-8/24_icon_Limits.png
try:
    _c24 = get_crop(24, 288, 156)
    canvas.paste(_c24, (864, 2804), _c24)
except Exception:
    pass
layout["'Limits"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_06_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-8/25_icon_Promoted.png
try:
    _c25 = get_crop(25, 42, 58)
    canvas.paste(_c25, (284, 1748), _c25)
except Exception:
    pass
layout["Promoted"] = [284, 1748, 326, 1806]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_06_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-8/26_icon_Learning_Calculus_the_easy_way.png
try:
    _c26 = get_crop(26, 288, 156)
    canvas.paste(_c26, (576, 2804), _c26)
except Exception:
    pass
layout["Learning_Calculus_the_eas"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_06_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-8/27_icon_5.18.png
try:
    _c27 = get_crop(27, 162, 64)
    canvas.paste(_c27, (5, 0), _c27)
except Exception:
    pass
layout["5.18"] = [5, 0, 167, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_06_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-8/28_text_10_000_events.png
try:
    _c28 = get_crop(28, 359, 103)
    canvas.paste(_c28, (54, 410), _c28)
except Exception:
    pass
layout["10,000_events"] = [54, 410, 413, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_06_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-8/29_text_Free.png
try:
    _c29 = get_crop(29, 80, 38)
    canvas.paste(_c29, (117, 1391), _c29)
except Exception:
    pass
layout["Free"] = [117, 1391, 197, 1429]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_06_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-8/30_text_ADVERTISING_MASTERCLASS_BUILDING_AN.png
try:
    _c30 = get_crop(30, 1344, 1175)
    canvas.paste(_c30, (48, 676), _c30)
except Exception:
    pass
layout["ADVERTISING_MASTERCLASS:_"] = [48, 676, 1392, 1851]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_06_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-8/31_text_EIGHT_FIGURE_BUSINESS.png
try:
    _c31 = get_crop(31, 1344, 1175)
    canvas.paste(_c31, (48, 676), _c31)
except Exception:
    pass
layout["EIGHT_FIGURE_BUSINESS"] = [48, 676, 1392, 1851]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_06_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-8/32_text_Wed.png
try:
    _c32 = get_crop(32, 107, 52)
    canvas.paste(_c32, (93, 1619), _c32)
except Exception:
    pass
layout["Wed,"] = [93, 1619, 200, 1671]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_06_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-8/33_text_24.png
try:
    _c33 = get_crop(33, 64, 43)
    canvas.paste(_c33, (276, 1622), _c33)
except Exception:
    pass
layout["24"] = [276, 1622, 340, 1665]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_06_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-8/34_text_12_00_PM_EDT.png
try:
    _c34 = get_crop(34, 277, 48)
    canvas.paste(_c34, (359, 1619), _c34)
except Exception:
    pass
layout["12:00_PM_EDT"] = [359, 1619, 636, 1667]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_06_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-8/35_text_Online.png
try:
    _c35 = get_crop(35, 129, 45)
    canvas.paste(_c35, (91, 1687), _c35)
except Exception:
    pass
layout["Online"] = [91, 1687, 220, 1732]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_06_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-8/36_text_JC.png
try:
    _c36 = get_crop(36, 92, 41)
    canvas.paste(_c36, (986, 1976), _c36)
except Exception:
    pass
layout["JC"] = [986, 1976, 1078, 2017]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_06_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-8/37_text_Jns.png
try:
    _c37 = get_crop(37, 90, 43)
    canvas.paste(_c37, (861, 2004), _c37)
except Exception:
    pass
layout["Jns"] = [861, 2004, 951, 2047]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_06_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-8/38_text_Ecavtr.png
try:
    _c38 = get_crop(38, 308, 134)
    canvas.paste(_c38, (146, 2064), _c38)
except Exception:
    pass
layout["Ecavtr;'"] = [146, 2064, 454, 2198]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_06_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-8/39_text_AP.png
try:
    _c39 = get_crop(39, 241, 230)
    canvas.paste(_c39, (1116, 1988), _c39)
except Exception:
    pass
layout["AP"] = [1116, 1988, 1357, 2218]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_06_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-8/40_text_Lkri.png
try:
    _c40 = get_crop(40, 153, 53)
    canvas.paste(_c40, (943, 2164), _c40)
except Exception:
    pass
layout["Lkri"] = [943, 2164, 1096, 2217]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_06_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-8/41_text_Uim.png
try:
    _c41 = get_crop(41, 101, 60)
    canvas.paste(_c41, (241, 2199), _c41)
except Exception:
    pass
layout["Uim"] = [241, 2199, 342, 2259]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_06_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-8/42_text_Grwhs_Smq.png
try:
    _c42 = get_crop(42, 1344, 917)
    canvas.paste(_c42, (48, 1899), _c42)
except Exception:
    pass
layout["Grwhs_Smq"] = [48, 1899, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_06_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-8/43_text_Free.png
try:
    _c43 = get_crop(43, 80, 39)
    canvas.paste(_c43, (117, 2614), _c43)
except Exception:
    pass
layout["Free"] = [117, 2614, 197, 2653]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_06_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-8/44_text_Learning_Calculus_the_easy_way.png
try:
    _c44 = get_crop(44, 1344, 917)
    canvas.paste(_c44, (48, 1899), _c44)
except Exception:
    pass
layout["Learning_Calculus_the_eas"] = [48, 1899, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_06_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-8/45_text_Tue_Apr_30.png
try:
    _c45 = get_crop(45, 288, 156)
    canvas.paste(_c45, (0, 2804), _c45)
except Exception:
    pass
layout["Tue,_Apr_30"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_06_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-8/46_clickable_More.png
try:
    _c46 = get_crop(46, 288, 156)
    canvas.paste(_c46, (1152, 2804), _c46)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]
