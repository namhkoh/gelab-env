# page_id: page_eventbrite_f502e886c78146dfb2f1efc2a331c781_05
# screenshot: 2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-7.png
# step_index: 5/18
# task: Open Eventbrite. Search for 'music festival' in San Francisco. Set date available from April 30 to May 3. How many events are listed?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Canvas and draw are provided. Fonts: font_sm, font_md, font_lg, font_xl

w, h = canvas.size

# Background fill (dominant off-white)
draw.rectangle([(0, 0), (w, h)], fill="#F6F7F9")

# Status bar area at top (~50px). Use slightly darker gray to match screenshot status bar.
status_h = 96
draw.rectangle([(0, 0), (w, status_h)], fill="#CFCFCF")

# Header / toolbar area (white) with bottom divider
header_top = status_h
header_h = 140
draw.rectangle([(0, header_top), (w, header_top + header_h)], fill="#FFFFFF")
# subtle divider line under header
divider_y = header_top + header_h
draw.line([(48, divider_y), (w - 48, divider_y)], fill="#E6E6E6", width=2)

# Main content margin
margin_x = 48

# First big content card background (rounded rectangle) with soft shadow
card1_top = 660
card1_width = w - 2 * margin_x + 24  # slightly wider shadow/backdrop
card1_height = 1194 + 48
card1_x0 = margin_x - 12
card1_y0 = card1_top
card1_x1 = card1_x0 + card1_width
card1_y1 = card1_y0 + card1_height

# Shadow (offset darker rounded rect)
draw.rounded_rectangle(
    [(card1_x0 + 6, card1_y0 + 8), (card1_x1 + 6, card1_y1 + 8)],
    radius=30,
    fill="#E9E9EA"
)
# Card background
draw.rounded_rectangle(
    [(card1_x0, card1_y0), (card1_x1, card1_y1)],
    radius=30,
    fill="#FFFFFF"
)

# Separator under first card (light)
sep1_y = card1_y1 + 28
draw.line([(margin_x, sep1_y), (w - margin_x, sep1_y)], fill="#F0F1F3", width=1)

# Second content card background (rounded rectangle) with shadow
card2_top = sep1_y + 40
card2_height = 620
card2_x0 = margin_x - 12
card2_y0 = card2_top
card2_x1 = w - margin_x + 12
card2_y1 = card2_y0 + card2_height

draw.rounded_rectangle(
    [(card2_x0 + 6, card2_y0 + 8), (card2_x1 + 6, card2_y1 + 8)],
    radius=24,
    fill="#E9E9EA"
)
draw.rounded_rectangle(
    [(card2_x0, card2_y0), (card2_x1, card2_y1)],
    radius=24,
    fill="#FFFFFF"
)

# Subtle horizontal separator to indicate content group boundaries further down
sep2_y = card2_y1 + 36
draw.line([(margin_x, sep2_y), (w - margin_x, sep2_y)], fill="#EFEFF1", width=2)

# Bottom navigation bar background and top divider
nav_h = 120
nav_top = h - nav_h
draw.rectangle([(0, nav_top), (w, h)], fill="#FFFFFF")
draw.line([(0, nav_top), (w, nav_top)], fill="#E6E6E6", width=2)

# Small top shadow on content area under header to give subtle depth
shadow_strip_top = divider_y
shadow_strip_bottom = divider_y + 6
draw.rectangle([(0, shadow_strip_top), (w, shadow_strip_bottom)], fill="#F2F3F5")

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_05_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-7/00_icon_Music.png
try:
    _c0 = get_crop(0, 187, 103)
    canvas.paste(_c0, (837, 410), _c0)
except Exception:
    pass
layout["Music"] = [837, 410, 1024, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_05_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-7/01_icon_Business.png
try:
    _c1 = get_crop(1, 241, 103)
    canvas.paste(_c1, (1036, 410), _c1)
except Exception:
    pass
layout["Business"] = [1036, 410, 1277, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_05_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-7/02_icon_Anytime.png
try:
    _c2 = get_crop(2, 400, 103)
    canvas.paste(_c2, (425, 410), _c2)
except Exception:
    pass
layout["Anytime"] = [425, 410, 825, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_05_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-7/03_icon_Filters.png
try:
    _c3 = get_crop(3, 359, 103)
    canvas.paste(_c3, (54, 410), _c3)
except Exception:
    pass
layout["Filters"] = [54, 410, 413, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_05_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-7/04_icon_Favorite_button.png
try:
    _c4 = get_crop(4, 144, 144)
    canvas.paste(_c4, (1092, 1192), _c4)
except Exception:
    pass
layout["Favorite_button"] = [1092, 1192, 1236, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_05_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-7/05_icon_NETELIX.png
try:
    _c5 = get_crop(5, 144, 144)
    canvas.paste(_c5, (1092, 2434), _c5)
except Exception:
    pass
layout["NETELIX"] = [1092, 2434, 1236, 2578]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_05_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-7/06_icon_Foo.png
try:
    _c6 = get_crop(6, 150, 110)
    canvas.paste(_c6, (1282, 406), _c6)
except Exception:
    pass
layout["Foo"] = [1282, 406, 1432, 516]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_05_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-7/07_icon_NETELIX.png
try:
    _c7 = get_crop(7, 144, 144)
    canvas.paste(_c7, (1236, 2434), _c7)
except Exception:
    pass
layout["NETELIX"] = [1236, 2434, 1380, 2578]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_05_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-7/08_icon_Overflow_menu_button.png
try:
    _c8 = get_crop(8, 144, 144)
    canvas.paste(_c8, (1236, 1192), _c8)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 1192, 1380, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_05_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-7/09_icon_Foo.png
try:
    _c9 = get_crop(9, 144, 144)
    canvas.paste(_c9, (1248, 96), _c9)
except Exception:
    pass
layout["Foo"] = [1248, 96, 1392, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_05_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-7/10_icon_7.18.png
try:
    _c10 = get_crop(10, 125, 116)
    canvas.paste(_c10, (54, 113), _c10)
except Exception:
    pass
layout["7.18"] = [54, 113, 179, 229]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_05_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-7/11_icon_Just_addedl.png
try:
    _c11 = get_crop(11, 313, 123)
    canvas.paste(_c11, (96, 2582), _c11)
except Exception:
    pass
layout["Just_addedl"] = [96, 2582, 409, 2705]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_05_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-7/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 64, 62)
    canvas.paste(_c12, (309, 0), _c12)
except Exception:
    pass
layout["icon_12"] = [309, 0, 373, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_05_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-7/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 104, 61)
    canvas.paste(_c13, (1207, 0), _c13)
except Exception:
    pass
layout["icon_13"] = [1207, 0, 1311, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_05_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-7/14_icon_7.18.png
try:
    _c14 = get_crop(14, 58, 62)
    canvas.paste(_c14, (182, 0), _c14)
except Exception:
    pass
layout["7.18"] = [182, 0, 240, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_05_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-7/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 49, 61)
    canvas.paste(_c15, (250, 1), _c15)
except Exception:
    pass
layout["icon_15"] = [250, 1, 299, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_05_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-7/16_icon_7.18.png
try:
    _c16 = get_crop(16, 59, 63)
    canvas.paste(_c16, (115, 0), _c16)
except Exception:
    pass
layout["7.18"] = [115, 0, 174, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_05_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-7/17_icon_Los_Angeles.png
try:
    _c17 = get_crop(17, 492, 144)
    canvas.paste(_c17, (0, 259), _c17)
except Exception:
    pass
layout["Los_Angeles"] = [0, 259, 492, 403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_05_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-7/18_icon_icon_18.png
try:
    _c18 = get_crop(18, 59, 61)
    canvas.paste(_c18, (1318, 0), _c18)
except Exception:
    pass
layout["icon_18"] = [1318, 0, 1377, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_05_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-7/19_icon_Netflix_Is_A_Joke_Fest.png
try:
    _c19 = get_crop(19, 288, 156)
    canvas.paste(_c19, (288, 2804), _c19)
except Exception:
    pass
layout["Netflix_Is_A_Joke_Fest"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_05_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-7/20_icon_LLS_Orchestra_at_the_Million_Dollar_Thea.png
try:
    _c20 = get_crop(20, 1344, 1194)
    canvas.paste(_c20, (48, 676), _c20)
except Exception:
    pass
layout["LLS_Orchestra_at_the_Mill"] = [48, 676, 1392, 1870]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_05_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-7/21_icon_Music_Festival.png
try:
    _c21 = get_crop(21, 1344, 191)
    canvas.paste(_c21, (48, 72), _c21)
except Exception:
    pass
layout["Music_Festival"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_05_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-7/22_icon_LLS_Orchestra_at_the_Million_Dollar_Thea.png
try:
    _c22 = get_crop(22, 1344, 1194)
    canvas.paste(_c22, (48, 676), _c22)
except Exception:
    pass
layout["LLS_Orchestra_at_the_Mill"] = [48, 676, 1392, 1870]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_05_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-7/23_icon_Music_Festival.png
try:
    _c23 = get_crop(23, 49, 60)
    canvas.paste(_c23, (384, 3), _c23)
except Exception:
    pass
layout["Music_Festival"] = [384, 3, 433, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_05_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-7/24_icon_Seinfeld_Gaffigan.png
try:
    _c24 = get_crop(24, 288, 156)
    canvas.paste(_c24, (576, 2804), _c24)
except Exception:
    pass
layout["Seinfeld,_Gaffigan,"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_05_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-7/25_icon_Just_addedl.png
try:
    _c25 = get_crop(25, 288, 156)
    canvas.paste(_c25, (0, 2804), _c25)
except Exception:
    pass
layout["Just_addedl"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_05_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-7/26_icon_Seinfeld_Gaffigan.png
try:
    _c26 = get_crop(26, 288, 156)
    canvas.paste(_c26, (864, 2804), _c26)
except Exception:
    pass
layout["Seinfeld,_Gaffigan,"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_05_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-7/27_icon_LLS_Orchestra_at_the_Million_Dollar_Thea.png
try:
    _c27 = get_crop(27, 1344, 1194)
    canvas.paste(_c27, (48, 676), _c27)
except Exception:
    pass
layout["LLS_Orchestra_at_the_Mill"] = [48, 676, 1392, 1870]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_05_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-7/28_icon_Seinfeld_Gaffigan.png
try:
    _c28 = get_crop(28, 288, 156)
    canvas.paste(_c28, (1152, 2804), _c28)
except Exception:
    pass
layout["Seinfeld,_Gaffigan,"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_05_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-7/29_icon_LLS_Orchestra_at_the_Million_Dollar_Thea.png
try:
    _c29 = get_crop(29, 1344, 1194)
    canvas.paste(_c29, (48, 676), _c29)
except Exception:
    pass
layout["LLS_Orchestra_at_the_Mill"] = [48, 676, 1392, 1870]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_05_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-7/30_icon_7.18.png
try:
    _c30 = get_crop(30, 126, 63)
    canvas.paste(_c30, (9, 0), _c30)
except Exception:
    pass
layout["7.18"] = [9, 0, 135, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_05_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-7/31_icon_Million_Dollar_Theater.png
try:
    _c31 = get_crop(31, 45, 59)
    canvas.paste(_c31, (283, 1766), _c31)
except Exception:
    pass
layout["Million_Dollar_Theater"] = [283, 1766, 328, 1825]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_05_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-7/32_icon_Promoted.png
try:
    _c32 = get_crop(32, 247, 63)
    canvas.paste(_c32, (84, 1764), _c32)
except Exception:
    pass
layout["Promoted"] = [84, 1764, 331, 1827]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_05_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-7/33_text_6_446_events.png
try:
    _c33 = get_crop(33, 359, 103)
    canvas.paste(_c33, (54, 410), _c33)
except Exception:
    pass
layout["6,446_events"] = [54, 410, 413, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_05_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-7/34_text_FSA_Joxe.png
try:
    _c34 = get_crop(34, 1344, 898)
    canvas.paste(_c34, (48, 1918), _c34)
except Exception:
    pass
layout["FSA_Joxe:"] = [48, 1918, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_05_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-7/35_text_THE_BIGGESI.png
try:
    _c35 = get_crop(35, 431, 133)
    canvas.paste(_c35, (805, 1972), _c35)
except Exception:
    pass
layout["THE_BIGGESI"] = [805, 1972, 1236, 2105]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_05_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-7/36_text_Probabiy.png
try:
    _c36 = get_crop(36, 177, 62)
    canvas.paste(_c36, (817, 2185), _c36)
except Exception:
    pass
layout["'Probabiy"] = [817, 2185, 994, 2247]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_05_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-7/37_text_FESTIVAL.png
try:
    _c37 = get_crop(37, 1344, 898)
    canvas.paste(_c37, (48, 1918), _c37)
except Exception:
    pass
layout["FESTIVAL"] = [48, 1918, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_05_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-7/38_text_NETELIX.png
try:
    _c38 = get_crop(38, 118, 49)
    canvas.paste(_c38, (1013, 2335), _c38)
except Exception:
    pass
layout["NETELIX"] = [1013, 2335, 1131, 2384]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_05_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-7/39_text_Netflix_Is_A_Joke_Fest.png
try:
    _c39 = get_crop(39, 288, 156)
    canvas.paste(_c39, (288, 2804), _c39)
except Exception:
    pass
layout["Netflix_Is_A_Joke_Fest"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_05_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-7/40_text_Seinfeld_Gaffigan.png
try:
    _c40 = get_crop(40, 288, 156)
    canvas.paste(_c40, (864, 2804), _c40)
except Exception:
    pass
layout["Seinfeld,_Gaffigan,"] = [864, 2804, 1152, 2960]
