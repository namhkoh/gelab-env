# page_id: page_eventbrite_c7c81d1bf6744774b99294e9f124dda3_04
# screenshot: 2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-6.png
# step_index: 4/10
# task: Open Eventbrite. Search for "Fitness". Select the events in the location "Chicago". What is the price of the first event in listing?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Top status bar
status_bar_h = 96
draw.rectangle((0, 0, 1440, status_bar_h), fill=(189, 189, 189))  # light grey status bar
draw.line((0, status_bar_h, 1440, status_bar_h), fill=(160, 160, 160), width=1)

# Header / toolbar area (white background)
header_top = status_bar_h
header_bottom = 260
draw.rectangle((0, header_top, 1440, header_bottom), fill=(255, 255, 255))
# subtle bottom divider under header
draw.line((32, header_bottom, 1408, header_bottom), fill=(220, 220, 220), width=2)

# Filter chip area background (keep it white but add subtle divider under it)
filters_top = header_bottom
filters_bottom = 480
draw.rectangle((0, filters_top, 1440, filters_bottom), fill=(255, 255, 255))
draw.line((32, filters_bottom, 1408, filters_bottom), fill=(235, 235, 235), width=1)

# Main content background (slightly warm white)
draw.rectangle((0, filters_bottom, 1440, 2804), fill=(250, 251, 252))

# Event card 1 background and shadow
card_margin_x = 48
card_right = 1440 - card_margin_x
card1_top = 620
card1_bottom = 1188
shadow_offset = 10
# shadow
draw.rounded_rectangle(
    (card_margin_x + shadow_offset, card1_top + shadow_offset, card_right + shadow_offset, card1_bottom + shadow_offset),
    radius=28,
    fill=(240, 240, 242)
)
# card body
draw.rounded_rectangle(
    (card_margin_x, card1_top, card_right, card1_bottom),
    radius=28,
    fill=(255, 255, 255),
    outline=(232, 232, 235),
    width=1
)
# image background area inside card 1 (behind the event image)
img1_pad = 16
img1_top = card1_top + img1_pad
img1_bottom = img1_top + 420
draw.rectangle((card_margin_x + img1_pad, img1_top, card_right - img1_pad, img1_bottom), fill=(240, 240, 240))

# subtle divider between image and text in card 1
draw.line((card_margin_x + 12, img1_bottom + 18, card_right - 12, img1_bottom + 18), fill=(245, 245, 246), width=1)

# Event card 1 text area background (keeps white but slightly offset to separate)
text_area_top = img1_bottom + 24
text_area_bottom = card1_bottom - 20
draw.rectangle((card_margin_x + 12, text_area_top, card_right - 12, text_area_bottom), fill=(255, 255, 255))

# Card 2 background and shadow (lower promoted/second listing)
card2_top = 1488
card2_bottom = 2248
# shadow
draw.rounded_rectangle(
    (card_margin_x + shadow_offset, card2_top + shadow_offset, card_right + shadow_offset, card2_bottom + shadow_offset),
    radius=28,
    fill=(240, 240, 242)
)
# card body
draw.rounded_rectangle(
    (card_margin_x, card2_top, card_right, card2_bottom),
    radius=28,
    fill=(255, 255, 255),
    outline=(232, 232, 235),
    width=1
)
# decorative banner/background for the second card's image area (keeps distinct but not duplicating content)
img2_pad = 16
img2_top = card2_top + img2_pad
img2_bottom = img2_top + 360
# soft teal band behind the event image
draw.rectangle((card_margin_x + img2_pad, img2_top, card_right - img2_pad, img2_bottom), fill=(222, 247, 243))

# subtle divider between image and text in card 2
draw.line((card_margin_x + 12, img2_bottom + 18, card_right - 12, img2_bottom + 18), fill=(245, 245, 246), width=1)

# Card 2 text area
text2_top = img2_bottom + 24
text2_bottom = card2_bottom - 20
draw.rectangle((card_margin_x + 12, text2_top, card_right - 12, text2_bottom), fill=(255, 255, 255))

# Small section separators near promoted label area
sep_y = 1560
draw.line((32, sep_y, 1408, sep_y), fill=(245, 245, 246), width=1)
draw.line((32, sep_y + 160, 1408, sep_y + 160), fill=(245, 245, 246), width=1)

# Bottom navigation bar background and top border
nav_top = 2804
nav_bottom = 2960
draw.rectangle((0, nav_top, 1440, nav_bottom), fill=(255, 255, 255))
draw.line((0, nav_top, 1440, nav_top), fill=(220, 220, 220), width=2)

# Light left and right page margins vertical guide lines (subtle)
draw.line((32, 0, 32, 2960), fill=(250, 250, 251), width=1)
draw.line((1408, 0, 1408, 2960), fill=(250, 250, 251), width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_04_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-6/00_icon_Music.png
try:
    _c0 = get_crop(0, 187, 103)
    canvas.paste(_c0, (837, 410), _c0)
except Exception:
    pass
layout["Music"] = [837, 410, 1024, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_04_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-6/01_icon_Business.png
try:
    _c1 = get_crop(1, 241, 103)
    canvas.paste(_c1, (1036, 410), _c1)
except Exception:
    pass
layout["Business"] = [1036, 410, 1277, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_04_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-6/02_icon_Anytime.png
try:
    _c2 = get_crop(2, 400, 103)
    canvas.paste(_c2, (425, 410), _c2)
except Exception:
    pass
layout["Anytime"] = [425, 410, 825, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_04_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-6/03_icon_Filters.png
try:
    _c3 = get_crop(3, 359, 103)
    canvas.paste(_c3, (54, 410), _c3)
except Exception:
    pass
layout["Filters"] = [54, 410, 413, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_04_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-6/04_icon_Foo.png
try:
    _c4 = get_crop(4, 149, 110)
    canvas.paste(_c4, (1282, 406), _c4)
except Exception:
    pass
layout["Foo"] = [1282, 406, 1431, 516]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_04_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-6/05_icon_EKEL.png
try:
    _c5 = get_crop(5, 144, 144)
    canvas.paste(_c5, (1092, 2252), _c5)
except Exception:
    pass
layout["EKEL"] = [1092, 2252, 1236, 2396]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_04_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-6/06_icon_Overflow_menu_button.png
try:
    _c6 = get_crop(6, 144, 144)
    canvas.paste(_c6, (1236, 2252), _c6)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 2252, 1380, 2396]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_04_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-6/07_icon_Favorite_button.png
try:
    _c7 = get_crop(7, 144, 144)
    canvas.paste(_c7, (1092, 1192), _c7)
except Exception:
    pass
layout["Favorite_button"] = [1092, 1192, 1236, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_04_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-6/08_icon_Overflow_menu_button.png
try:
    _c8 = get_crop(8, 144, 144)
    canvas.paste(_c8, (1236, 1192), _c8)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 1192, 1380, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_04_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-6/09_icon_Foo.png
try:
    _c9 = get_crop(9, 144, 144)
    canvas.paste(_c9, (1248, 96), _c9)
except Exception:
    pass
layout["Foo"] = [1248, 96, 1392, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_04_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-6/10_icon_7.09.png
try:
    _c10 = get_crop(10, 125, 116)
    canvas.paste(_c10, (53, 112), _c10)
except Exception:
    pass
layout["7.09"] = [53, 112, 178, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_04_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-6/11_icon_Fitness.png
try:
    _c11 = get_crop(11, 68, 64)
    canvas.paste(_c11, (308, 0), _c11)
except Exception:
    pass
layout["Fitness"] = [308, 0, 376, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_04_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-6/12_icon_Fitness.png
try:
    _c12 = get_crop(12, 54, 65)
    canvas.paste(_c12, (246, 0), _c12)
except Exception:
    pass
layout["Fitness"] = [246, 0, 300, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_04_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-6/13_icon_7.09.png
try:
    _c13 = get_crop(13, 60, 64)
    canvas.paste(_c13, (181, 0), _c13)
except Exception:
    pass
layout["7.09"] = [181, 0, 241, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_04_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-6/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 105, 61)
    canvas.paste(_c14, (1204, 0), _c14)
except Exception:
    pass
layout["icon_14"] = [1204, 0, 1309, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_04_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-6/15_icon_7.09.png
try:
    _c15 = get_crop(15, 61, 65)
    canvas.paste(_c15, (114, 0), _c15)
except Exception:
    pass
layout["7.09"] = [114, 0, 175, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_04_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-6/16_icon_icon_16.png
try:
    _c16 = get_crop(16, 65, 60)
    canvas.paste(_c16, (1317, 0), _c16)
except Exception:
    pass
layout["icon_16"] = [1317, 0, 1382, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_04_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-6/17_icon_Promoted.png
try:
    _c17 = get_crop(17, 282, 69)
    canvas.paste(_c17, (50, 1579), _c17)
except Exception:
    pass
layout["Promoted"] = [50, 1579, 332, 1648]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_04_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-6/18_icon_Mindhappy_Party.png
try:
    _c18 = get_crop(18, 1344, 1012)
    canvas.paste(_c18, (48, 676), _c18)
except Exception:
    pass
layout["Mindhappy_Party"] = [48, 676, 1392, 1688]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_04_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-6/19_icon_New_York.png
try:
    _c19 = get_crop(19, 434, 144)
    canvas.paste(_c19, (0, 259), _c19)
except Exception:
    pass
layout["New_York"] = [0, 259, 434, 403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_04_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-6/20_icon_Day.png
try:
    _c20 = get_crop(20, 288, 156)
    canvas.paste(_c20, (576, 2804), _c20)
except Exception:
    pass
layout["Day"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_04_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-6/21_icon_icon_21.png
try:
    _c21 = get_crop(21, 51, 62)
    canvas.paste(_c21, (384, 2), _c21)
except Exception:
    pass
layout["icon_21"] = [384, 2, 435, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_04_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-6/22_icon_Fitness.png
try:
    _c22 = get_crop(22, 1344, 191)
    canvas.paste(_c22, (48, 72), _c22)
except Exception:
    pass
layout["Fitness"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_04_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-6/23_icon_Albee_Square.png
try:
    _c23 = get_crop(23, 44, 55)
    canvas.paste(_c23, (285, 2729), _c23)
except Exception:
    pass
layout["Albee_Square"] = [285, 2729, 329, 2784]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_04_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-6/24_icon_Day.png
try:
    _c24 = get_crop(24, 288, 156)
    canvas.paste(_c24, (864, 2804), _c24)
except Exception:
    pass
layout["Day"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_04_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-6/25_icon_Day.png
try:
    _c25 = get_crop(25, 288, 156)
    canvas.paste(_c25, (1152, 2804), _c25)
except Exception:
    pass
layout["Day"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_04_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-6/26_icon_11.30AM_EDT.png
try:
    _c26 = get_crop(26, 288, 156)
    canvas.paste(_c26, (288, 2804), _c26)
except Exception:
    pass
layout["11.30AM_EDT"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_04_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-6/27_icon_Pce_MoNR.png
try:
    _c27 = get_crop(27, 1344, 1080)
    canvas.paste(_c27, (48, 1736), _c27)
except Exception:
    pass
layout["[Pce:_{MoNR"] = [48, 1736, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_04_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-6/28_text_7.09.png
try:
    _c28 = get_crop(28, 91, 45)
    canvas.paste(_c28, (20, 15), _c28)
except Exception:
    pass
layout["7.09"] = [20, 15, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_04_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-6/29_text_1_873_events.png
try:
    _c29 = get_crop(29, 359, 103)
    canvas.paste(_c29, (54, 410), _c29)
except Exception:
    pass
layout["1,873_events"] = [54, 410, 413, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_04_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-6/30_text_Explore.png
try:
    _c30 = get_crop(30, 174, 61)
    canvas.paste(_c30, (182, 1754), _c30)
except Exception:
    pass
layout["Explore"] = [182, 1754, 356, 1815]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_04_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-6/31_text_Schools.png
try:
    _c31 = get_crop(31, 179, 52)
    canvas.paste(_c31, (183, 1818), _c31)
except Exception:
    pass
layout["Schools"] = [183, 1818, 362, 1870]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_04_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-6/32_clickable_Home.png
try:
    _c32 = get_crop(32, 288, 156)
    canvas.paste(_c32, (0, 2804), _c32)
except Exception:
    pass
layout["Home"] = [0, 2804, 288, 2960]
