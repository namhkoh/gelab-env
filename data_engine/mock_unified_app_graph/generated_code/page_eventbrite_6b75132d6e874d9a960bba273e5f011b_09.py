# page_id: page_eventbrite_6b75132d6e874d9a960bba273e5f011b_09
# screenshot: 2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-11.png
# step_index: 9/11
# task: Open Eventbrite. Set the city to 'San Francisco'. Search 'Outdoor'. Select an event starting after 5 PM. Check the ticket price.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# fill overall background (very light warm gray)
draw.rectangle([(0, 0), (1440, 2960)], fill=(248, 249, 250))

# STATUS BAR (top area)
status_h = 96
draw.rectangle([(0, 0), (1440, status_h)], fill=(169, 169, 169))  # muted gray status bar
# subtle bottom line under status bar
draw.line([(24, status_h-1), (1440-24, status_h-1)], fill=(200, 200, 200), width=1)

# HEADER / TOOLBAR area (search/title area)
header_top = status_h
header_bottom = status_h + 112
draw.rectangle([(0, header_top), (1440, header_bottom)], fill=(255, 255, 255))
# bottom divider under header
draw.line([(48, header_bottom), (1440-48, header_bottom)], fill=(230, 230, 230), width=2)

# Subtle underline for filter/search area (separating toolbar from content)
filters_sep_y = header_bottom + 120
draw.line([(48, filters_sep_y), (1440-48, filters_sep_y)], fill=(240, 240, 240), width=1)

# MAIN CONTENT MARGINS
content_left = 48
content_right = 1440 - 48

# First event card background (rounded white card with subtle shadow)
card1_top = header_bottom + 160
card1_width = content_right - content_left
card1_height = 1150
card1_bbox = (content_left, card1_top, content_left + card1_width, card1_top + card1_height)

# shadow for card1
shadow_offset = 8
shadow_bbox = (card1_bbox[0] + shadow_offset, card1_bbox[1] + shadow_offset,
               card1_bbox[2] + shadow_offset, card1_bbox[3] + shadow_offset)
draw.rounded_rectangle(shadow_bbox, radius=20, fill=(230, 230, 230))

# main card1
draw.rounded_rectangle(card1_bbox, radius=20, fill=(255, 255, 255))

# image/background area inside card1 (top portion where photo will be pasted)
img1_h = 420
img1_bbox = (card1_bbox[0] + 16, card1_bbox[1] + 16, card1_bbox[2] - 16, card1_bbox[1] + 16 + img1_h)
draw.rounded_rectangle(img1_bbox, radius=14, fill=(245, 246, 247))

# subtle divider under card1 content area
divider1_y = card1_bbox[1] + img1_h + 36
draw.line([(card1_bbox[0] + 16, divider1_y), (card1_bbox[2] - 16, divider1_y)], fill=(240, 240, 240), width=1)

# Second event card background (rounded white card with subtle shadow)
card2_top = card1_bbox[1] + card1_height + 32
card2_height = 1080
card2_bbox = (content_left, card2_top, content_left + card1_width, card2_top + card2_height)

# shadow for card2
shadow2_bbox = (card2_bbox[0] + shadow_offset, card2_bbox[1] + shadow_offset,
                card2_bbox[2] + shadow_offset, card2_bbox[3] + shadow_offset)
draw.rounded_rectangle(shadow2_bbox, radius=20, fill=(230, 230, 230))

# main card2
draw.rounded_rectangle(card2_bbox, radius=20, fill=(255, 255, 255))

# image/background area inside card2 (top portion where photo will be pasted)
img2_h = 420
img2_bbox = (card2_bbox[0] + 16, card2_bbox[1] + 16, card2_bbox[2] - 16, card2_bbox[1] + 16 + img2_h)
draw.rounded_rectangle(img2_bbox, radius=14, fill=(245, 246, 247))

# divider under card2 image area
divider2_y = card2_bbox[1] + img2_h + 36
draw.line([(card2_bbox[0] + 16, divider2_y), (card2_bbox[2] - 16, divider2_y)], fill=(240, 240, 240), width=1)

# Light background band behind the list count / small section header (near top of content)
band_top = header_bottom + 24
band_bottom = band_top + 64
draw.rectangle([(content_left, band_top), (content_right, band_bottom)], fill=(255, 255, 255))
draw.line([(content_left, band_bottom), (content_right, band_bottom)], fill=(235, 235, 235), width=1)

# Thin separators between content groups
sep_y = card1_bbox[1] - 40
draw.line([(content_left, sep_y), (content_right, sep_y)], fill=(245, 245, 245), width=1)
sep_y2 = card2_bbox[1] - 40
draw.line([(content_left, sep_y2), (content_right, sep_y2)], fill=(245, 245, 245), width=1)

# BOTTOM NAV BAR background (reserve bottom area; icons will be pasted on top)
nav_top = 2804
nav_bottom = 2960
draw.rectangle([(0, nav_top), (1440, nav_bottom)], fill=(255, 255, 255))
# top divider for nav bar
draw.line([(24, nav_top), (1440-24, nav_top)], fill=(230, 230, 230), width=2)

# subtle shadow above nav bar
draw.line([(24, nav_top+2), (1440-24, nav_top+2)], fill=(245, 245, 245), width=1)

# final fine border around content area (very light)
draw.rectangle([(content_left-8, header_bottom+8), (content_right+8, nav_top-8)], outline=(245, 245, 246), width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_09_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-11/00_icon_Music.png
try:
    _c0 = get_crop(0, 187, 103)
    canvas.paste(_c0, (837, 410), _c0)
except Exception:
    pass
layout["Music"] = [837, 410, 1024, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_09_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-11/01_icon_Business.png
try:
    _c1 = get_crop(1, 241, 103)
    canvas.paste(_c1, (1036, 410), _c1)
except Exception:
    pass
layout["Business"] = [1036, 410, 1277, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_09_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-11/02_icon_Anytime.png
try:
    _c2 = get_crop(2, 400, 103)
    canvas.paste(_c2, (425, 410), _c2)
except Exception:
    pass
layout["Anytime"] = [425, 410, 825, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_09_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-11/03_icon_Filters.png
try:
    _c3 = get_crop(3, 359, 103)
    canvas.paste(_c3, (54, 410), _c3)
except Exception:
    pass
layout["Filters"] = [54, 410, 413, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_09_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-11/04_icon_Favorite_button.png
try:
    _c4 = get_crop(4, 144, 144)
    canvas.paste(_c4, (1092, 2269), _c4)
except Exception:
    pass
layout["Favorite_button"] = [1092, 2269, 1236, 2413]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_09_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-11/05_icon_Foo.png
try:
    _c5 = get_crop(5, 150, 110)
    canvas.paste(_c5, (1282, 406), _c5)
except Exception:
    pass
layout["Foo"] = [1282, 406, 1432, 516]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_09_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-11/06_icon_Favorite_button.png
try:
    _c6 = get_crop(6, 144, 144)
    canvas.paste(_c6, (1092, 1192), _c6)
except Exception:
    pass
layout["Favorite_button"] = [1092, 1192, 1236, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_09_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-11/07_icon_Overflow_menu_button.png
try:
    _c7 = get_crop(7, 144, 144)
    canvas.paste(_c7, (1236, 2269), _c7)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 2269, 1380, 2413]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_09_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-11/08_icon_Overflow_menu_button.png
try:
    _c8 = get_crop(8, 144, 144)
    canvas.paste(_c8, (1236, 1192), _c8)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 1192, 1380, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_09_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-11/09_icon_Outdoor.png
try:
    _c9 = get_crop(9, 1344, 191)
    canvas.paste(_c9, (48, 72), _c9)
except Exception:
    pass
layout["Outdoor"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_09_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-11/10_icon_Foo.png
try:
    _c10 = get_crop(10, 144, 144)
    canvas.paste(_c10, (1248, 96), _c10)
except Exception:
    pass
layout["Foo"] = [1248, 96, 1392, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_09_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-11/11_icon_8.11.png
try:
    _c11 = get_crop(11, 121, 112)
    canvas.paste(_c11, (56, 115), _c11)
except Exception:
    pass
layout["8.11"] = [56, 115, 177, 227]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_09_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-11/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 54, 65)
    canvas.paste(_c12, (1151, 0), _c12)
except Exception:
    pass
layout["icon_12"] = [1151, 0, 1205, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_09_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-11/13_icon_Nor_Cal_Outdoor_Academy.png
try:
    _c13 = get_crop(13, 1344, 1029)
    canvas.paste(_c13, (48, 676), _c13)
except Exception:
    pass
layout["Nor_Cal_Outdoor_Academy"] = [48, 676, 1392, 1705]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_09_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-11/14_icon_Outdoor.png
try:
    _c14 = get_crop(14, 68, 63)
    canvas.paste(_c14, (308, 0), _c14)
except Exception:
    pass
layout["Outdoor"] = [308, 0, 376, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_09_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-11/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 100, 63)
    canvas.paste(_c15, (1211, 0), _c15)
except Exception:
    pass
layout["icon_15"] = [1211, 0, 1311, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_09_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-11/16_icon_2_._6_00_PM_PDT.png
try:
    _c16 = get_crop(16, 1344, 1029)
    canvas.paste(_c16, (48, 1753), _c16)
except Exception:
    pass
layout["2_._6:00_PM_PDT"] = [48, 1753, 1392, 2782]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_09_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-11/17_icon_8.11.png
try:
    _c17 = get_crop(17, 58, 63)
    canvas.paste(_c17, (182, 0), _c17)
except Exception:
    pass
layout["8.11"] = [182, 0, 240, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_09_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-11/18_icon_8.11.png
try:
    _c18 = get_crop(18, 63, 64)
    canvas.paste(_c18, (112, 0), _c18)
except Exception:
    pass
layout["8.11"] = [112, 0, 175, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_09_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-11/19_icon_Outdoor.png
try:
    _c19 = get_crop(19, 51, 63)
    canvas.paste(_c19, (247, 1), _c19)
except Exception:
    pass
layout["Outdoor"] = [247, 1, 298, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_09_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-11/20_icon_icon_20.png
try:
    _c20 = get_crop(20, 57, 61)
    canvas.paste(_c20, (1318, 0), _c20)
except Exception:
    pass
layout["icon_20"] = [1318, 0, 1375, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_09_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-11/21_icon_San_Francisco.png
try:
    _c21 = get_crop(21, 536, 144)
    canvas.paste(_c21, (0, 259), _c21)
except Exception:
    pass
layout["San_Francisco"] = [0, 259, 536, 403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_09_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-11/22_icon_Outdoor.png
try:
    _c22 = get_crop(22, 50, 62)
    canvas.paste(_c22, (384, 2), _c22)
except Exception:
    pass
layout["Outdoor"] = [384, 2, 434, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_09_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-11/23_icon_2_._6_00_PM_PDT.png
try:
    _c23 = get_crop(23, 288, 156)
    canvas.paste(_c23, (576, 2804), _c23)
except Exception:
    pass
layout["2_._6:00_PM_PDT"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_09_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-11/24_icon_2_._6_00_PM_PDT.png
try:
    _c24 = get_crop(24, 288, 156)
    canvas.paste(_c24, (864, 2804), _c24)
except Exception:
    pass
layout["2_._6:00_PM_PDT"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_09_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-11/25_icon_More.png
try:
    _c25 = get_crop(25, 288, 156)
    canvas.paste(_c25, (1152, 2804), _c25)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_09_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-11/26_icon_8.11.png
try:
    _c26 = get_crop(26, 98, 63)
    canvas.paste(_c26, (9, 0), _c26)
except Exception:
    pass
layout["8.11"] = [9, 0, 107, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_09_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-11/27_icon_Jackson_Playground.png
try:
    _c27 = get_crop(27, 288, 156)
    canvas.paste(_c27, (288, 2804), _c27)
except Exception:
    pass
layout["Jackson_Playground"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_09_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-11/28_text_1_178_events.png
try:
    _c28 = get_crop(28, 359, 103)
    canvas.paste(_c28, (54, 410), _c28)
except Exception:
    pass
layout["1,178_events"] = [54, 410, 413, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_09_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-11/29_clickable_Home.png
try:
    _c29 = get_crop(29, 288, 156)
    canvas.paste(_c29, (0, 2804), _c29)
except Exception:
    pass
layout["Home"] = [0, 2804, 288, 2960]
