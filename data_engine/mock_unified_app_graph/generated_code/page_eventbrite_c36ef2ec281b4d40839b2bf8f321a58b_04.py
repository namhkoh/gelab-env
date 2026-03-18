# page_id: page_eventbrite_c36ef2ec281b4d40839b2bf8f321a58b_04
# screenshot: 2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-6.png
# step_index: 4/8
# task: Open Eventbrite. Set the city to "Chicago". Select the "Fashion" category and view the recommended events. See the date of the first play and its venue.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw UI background and structural elements for a 1440x2960 mobile UI canvas.
# Available variables:
# - canvas: PIL Image (1440x2960 RGB, initially white)
# - draw: PIL ImageDraw object
# - font_sm, font_md, font_lg, font_xl

w, h = canvas.size

# Overall background (very light cool gray)
draw.rectangle((0, 0, w, h), fill="#F6F7F9")

# Status bar (top area ~64px) - muted gray
status_h = 64
draw.rectangle((0, 0, w, status_h), fill="#BDBEC0")

# Top toolbar / header area below status bar (~64..140)
toolbar_top = status_h
toolbar_bottom = 140
draw.rectangle((0, toolbar_top, w, toolbar_bottom), fill="#FFFFFF")

# Subtle bottom divider under header
draw.line((32, toolbar_bottom, w - 32, toolbar_bottom), fill="#E6E7EA", width=2)

# Light horizontal separator where filter chips area sits (beneath search)
chips_separator_y = 360
draw.line((24, chips_separator_y, w - 24, chips_separator_y), fill="#ECEFF2", width=1)

# First event card area (white card with shadow)
card1_x1, card1_x2 = 32, w - 32
card1_y1, card1_y2 = 600, 1960
card_corner = 28
# Shadow
draw.rounded_rectangle(
    (card1_x1 + 6, card1_y1 + 8, card1_x2 + 6, card1_y2 + 8),
    radius=card_corner,
    fill="#ECEFF3",
    outline=None
)
# Card background
draw.rounded_rectangle(
    (card1_x1, card1_y1, card1_x2, card1_y2),
    radius=card_corner,
    fill="#FFFFFF",
    outline="#E7E8EB"
)

# Image/hero background area for first card (placeholder background only)
# Detected image crop will be pasted on top; this provides the underlying card image background.
img1_x, img1_y = 48, 676
img1_w, img1_h = 1344, 1175
img1_box = (img1_x, img1_y, img1_x + img1_w, img1_y + img1_h)
img_radius = 20
draw.rounded_rectangle(img1_box, radius=img_radius, fill="#F2F6F8", outline="#E1E6EA")

# Separator below first card content area
sep_y = img1_y + img1_h + 24
draw.line((48, sep_y, w - 48, sep_y), fill="#F0F1F3", width=1)

# Second promoted/banner card (white card with subtle shadow)
card2_x1, card2_x2 = 32, w - 32
card2_y1, card2_y2 = 1840, 2870
card2_corner = 20
# Shadow
draw.rounded_rectangle(
    (card2_x1 + 6, card2_y1 + 6, card2_x2 + 6, card2_y2 + 6),
    radius=card2_corner,
    fill="#F0F2F4"
)
# Card background
draw.rounded_rectangle(
    (card2_x1, card2_y1, card2_x2, card2_y2),
    radius=card2_corner,
    fill="#FFFFFF",
    outline="#E9EAED"
)

# Banner/image placeholder for second card (background behind pasted banner)
img2_x, img2_y = 48, 1899
img2_w, img2_h = 1344, 917
img2_box = (img2_x, img2_y, img2_x + img2_w, img2_y + img2_h)
draw.rounded_rectangle(img2_box, radius=18, fill="#FFF8EE", outline="#EDE7E2")

# Thin divider lines between major sections
draw.line((24, 1200, w - 24, 1200), fill="#F3F4F6", width=1)
draw.line((24, 1800, w - 24, 1800), fill="#F3F4F6", width=1)

# Bottom navigation bar area
nav_h = 130
nav_y1 = h - nav_h
# Top divider
draw.line((0, nav_y1, w, nav_y1), fill="#E6E7EA", width=2)
# Nav background
draw.rectangle((0, nav_y1, w, h), fill="#FFFFFF")
# Subtle active pill background at center (structural only; icons will be pasted on top)
pill_w, pill_h = 84, 84
pill_x = w // 2 - pill_w // 2
pill_y = nav_y1 + 18
draw.ellipse((pill_x - 6, pill_y - 6, pill_x + pill_w + 6, pill_y + pill_h + 6), fill="#FFF2EE")
draw.ellipse((pill_x, pill_y, pill_x + pill_w, pill_y + pill_h), fill="#FFFFFF", outline="#F0A05A")

# Small left/right subtle shadows for card stack feel
draw.rectangle((24, 520, w - 24, 540), fill="#F7F8F9")
draw.rectangle((24, 1720, w - 24, 1740), fill="#F7F8F9")

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_04_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-6/00_icon_Music.png
try:
    _c0 = get_crop(0, 187, 103)
    canvas.paste(_c0, (837, 410), _c0)
except Exception:
    pass
layout["Music"] = [837, 410, 1024, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_04_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-6/01_icon_Business.png
try:
    _c1 = get_crop(1, 241, 103)
    canvas.paste(_c1, (1036, 410), _c1)
except Exception:
    pass
layout["Business"] = [1036, 410, 1277, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_04_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-6/02_icon_Anytime.png
try:
    _c2 = get_crop(2, 400, 103)
    canvas.paste(_c2, (425, 410), _c2)
except Exception:
    pass
layout["Anytime"] = [425, 410, 825, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_04_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-6/03_icon_Filters.png
try:
    _c3 = get_crop(3, 359, 103)
    canvas.paste(_c3, (54, 410), _c3)
except Exception:
    pass
layout["Filters"] = [54, 410, 413, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_04_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-6/04_icon_Foo.png
try:
    _c4 = get_crop(4, 150, 110)
    canvas.paste(_c4, (1282, 406), _c4)
except Exception:
    pass
layout["Foo"] = [1282, 406, 1432, 516]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_04_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-6/05_icon_Favorite_button.png
try:
    _c5 = get_crop(5, 144, 144)
    canvas.paste(_c5, (1092, 1192), _c5)
except Exception:
    pass
layout["Favorite_button"] = [1092, 1192, 1236, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_04_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-6/06_icon_Favorite_button.png
try:
    _c6 = get_crop(6, 144, 144)
    canvas.paste(_c6, (1092, 2415), _c6)
except Exception:
    pass
layout["Favorite_button"] = [1092, 2415, 1236, 2559]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_04_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-6/07_icon_Overflow_menu_button.png
try:
    _c7 = get_crop(7, 144, 144)
    canvas.paste(_c7, (1236, 1192), _c7)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 1192, 1380, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_04_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-6/08_icon_Overflow_menu_button.png
try:
    _c8 = get_crop(8, 144, 144)
    canvas.paste(_c8, (1236, 2415), _c8)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 2415, 1380, 2559]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_04_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-6/09_icon_5.12.png
try:
    _c9 = get_crop(9, 119, 110)
    canvas.paste(_c9, (59, 117), _c9)
except Exception:
    pass
layout["5.12"] = [59, 117, 178, 227]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_04_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-6/10_icon_Search_forae.png
try:
    _c10 = get_crop(10, 1344, 191)
    canvas.paste(_c10, (48, 72), _c10)
except Exception:
    pass
layout["Search_forae"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_04_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-6/11_icon_Pretty_In_Pink_Daytime_Yacht_Event.png
try:
    _c11 = get_crop(11, 1344, 1175)
    canvas.paste(_c11, (48, 676), _c11)
except Exception:
    pass
layout["Pretty_In_Pink_(Daytime)_"] = [48, 676, 1392, 1851]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_04_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-6/12_icon_Search_forae.png
try:
    _c12 = get_crop(12, 69, 62)
    canvas.paste(_c12, (307, 1), _c12)
except Exception:
    pass
layout["Search_forae"] = [307, 1, 376, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_04_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-6/13_icon_5.12.png
try:
    _c13 = get_crop(13, 60, 63)
    canvas.paste(_c13, (181, 0), _c13)
except Exception:
    pass
layout["5.12"] = [181, 0, 241, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_04_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-6/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 52, 62)
    canvas.paste(_c14, (248, 1), _c14)
except Exception:
    pass
layout["icon_14"] = [248, 1, 300, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_04_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-6/15_icon_Chicago.png
try:
    _c15 = get_crop(15, 417, 144)
    canvas.paste(_c15, (0, 259), _c15)
except Exception:
    pass
layout["Chicago"] = [0, 259, 417, 403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_04_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-6/16_icon_icon_16.png
try:
    _c16 = get_crop(16, 89, 60)
    canvas.paste(_c16, (1208, 0), _c16)
except Exception:
    pass
layout["icon_16"] = [1208, 0, 1297, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_04_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-6/17_icon_5.12.png
try:
    _c17 = get_crop(17, 60, 65)
    canvas.paste(_c17, (115, 0), _c17)
except Exception:
    pass
layout["5.12"] = [115, 0, 175, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_04_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-6/18_icon_icon_18.png
try:
    _c18 = get_crop(18, 62, 59)
    canvas.paste(_c18, (1316, 0), _c18)
except Exception:
    pass
layout["icon_18"] = [1316, 0, 1378, 59]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_04_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-6/19_icon_Event.png
try:
    _c19 = get_crop(19, 288, 156)
    canvas.paste(_c19, (288, 2804), _c19)
except Exception:
    pass
layout["Event"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_04_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-6/20_icon_Search_forae.png
try:
    _c20 = get_crop(20, 52, 61)
    canvas.paste(_c20, (383, 2), _c20)
except Exception:
    pass
layout["Search_forae"] = [383, 2, 435, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_04_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-6/21_icon_prevention.png
try:
    _c21 = get_crop(21, 101, 125)
    canvas.paste(_c21, (50, 2357), _c21)
except Exception:
    pass
layout["prevention"] = [50, 2357, 151, 2482]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_04_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-6/22_icon_Breathe_Well_Free_Lung_Health_Screening.png
try:
    _c22 = get_crop(22, 288, 156)
    canvas.paste(_c22, (1152, 2804), _c22)
except Exception:
    pass
layout["Breathe_Well:_Free_Lung_H"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_04_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-6/23_icon_REGISTER_NOW.png
try:
    _c23 = get_crop(23, 128, 94)
    canvas.paste(_c23, (799, 1946), _c23)
except Exception:
    pass
layout["REGISTER_NOW"] = [799, 1946, 927, 2040]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_04_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-6/24_icon_Anita_Dee_Yacht_Charters.png
try:
    _c24 = get_crop(24, 46, 59)
    canvas.paste(_c24, (281, 1748), _c24)
except Exception:
    pass
layout["Anita_Dee_Yacht_Charters"] = [281, 1748, 327, 1807]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_04_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-6/25_icon_Breathe_Well_Free_Lung_Health_Screening.png
try:
    _c25 = get_crop(25, 288, 156)
    canvas.paste(_c25, (864, 2804), _c25)
except Exception:
    pass
layout["Breathe_Well:_Free_Lung_H"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_04_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-6/26_icon_Breathe_Well_Free_Lung_Health_Screening.png
try:
    _c26 = get_crop(26, 288, 156)
    canvas.paste(_c26, (576, 2804), _c26)
except Exception:
    pass
layout["Breathe_Well:_Free_Lung_H"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_04_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-6/27_icon_icon_27.png
try:
    _c27 = get_crop(27, 40, 61)
    canvas.paste(_c27, (1274, 0), _c27)
except Exception:
    pass
layout["icon_27"] = [1274, 0, 1314, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_04_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-6/28_icon_Event.png
try:
    _c28 = get_crop(28, 288, 156)
    canvas.paste(_c28, (0, 2804), _c28)
except Exception:
    pass
layout["Event"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_04_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-6/29_icon_Promoted.png
try:
    _c29 = get_crop(29, 246, 62)
    canvas.paste(_c29, (83, 1746), _c29)
except Exception:
    pass
layout["Promoted"] = [83, 1746, 329, 1808]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_04_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-6/30_icon_Breathe_Well_Free_Lung_Health_Screening.png
try:
    _c30 = get_crop(30, 1344, 917)
    canvas.paste(_c30, (48, 1899), _c30)
except Exception:
    pass
layout["Breathe_Well:_Free_Lung_H"] = [48, 1899, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_04_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-6/31_text_5.12.png
try:
    _c31 = get_crop(31, 89, 43)
    canvas.paste(_c31, (22, 17), _c31)
except Exception:
    pass
layout["5.12"] = [22, 17, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_04_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-6/32_text_10_000_events.png
try:
    _c32 = get_crop(32, 359, 103)
    canvas.paste(_c32, (54, 410), _c32)
except Exception:
    pass
layout["10,000_events"] = [54, 410, 413, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_04_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-6/33_text_Anita_Dee_Yacht_Charters.png
try:
    _c33 = get_crop(33, 472, 50)
    canvas.paste(_c33, (91, 1686), _c33)
except Exception:
    pass
layout["Anita_Dee_Yacht_Charters"] = [91, 1686, 563, 1736]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_04_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-6/34_text_GenHarp.png
try:
    _c34 = get_crop(34, 212, 68)
    canvas.paste(_c34, (236, 1954), _c34)
except Exception:
    pass
layout["GenHarp"] = [236, 1954, 448, 2022]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_04_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-6/35_text_REGISTER_NOW.png
try:
    _c35 = get_crop(35, 256, 41)
    canvas.paste(_c35, (928, 1976), _c35)
except Exception:
    pass
layout["REGISTER_NOW"] = [928, 1976, 1184, 2017]
