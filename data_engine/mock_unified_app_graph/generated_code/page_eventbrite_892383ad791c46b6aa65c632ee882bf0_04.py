# page_id: page_eventbrite_892383ad791c46b6aa65c632ee882bf0_04
# screenshot: 2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-6.png
# step_index: 4/12
# task: Open Eventbrite. Search for online "Music" events happening next weekend.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Status bar
status_h = 72
draw.rectangle([(0, 0), (1440, status_h)], fill=(206, 206, 206))

# Header / toolbar background
header_top = status_h
header_h = 128
draw.rectangle([(0, header_top), (1440, header_top + header_h)], fill=(255, 255, 255))
# subtle bottom divider under header
draw.line([(24, header_top + header_h), (1440 - 24, header_top + header_h)], fill=(225, 225, 225), width=2)

# Thin divider below header area (separates toolbar from filters/location)
divider_y = header_top + header_h + 56
draw.line([(24, divider_y), (1440 - 24, divider_y)], fill=(240, 240, 240), width=1)

# Filters area background (keeps it visually separate, but do not draw pills/icons/text)
filters_area_top = divider_y
filters_area_h = 120
draw.rectangle([(0, filters_area_top), (1440, filters_area_top + filters_area_h)], fill=(255, 255, 255))
# faint bottom border under filters
draw.line([(24, filters_area_top + filters_area_h), (1440 - 24, filters_area_top + filters_area_h)], fill=(235, 235, 235), width=1)

# "10,000 events" area separator (space for count/title) - keep clean white with divider below
events_header_top = filters_area_top + filters_area_h + 24
draw.rectangle([(0, events_header_top), (1440, events_header_top + 80)], fill=(255, 255, 255))
draw.line([(24, events_header_top + 80), (1440 - 24, events_header_top + 80)], fill=(245, 245, 245), width=1)

# Event card 1 container (rounded rectangle background + subtle border)
card1_x = 48
card1_y = 676
card1_w = 1344
card1_h = 420  # card background for image + details area
card1_bbox = [card1_x - 8, card1_y - 8, card1_x + card1_w + 8, card1_y + card1_h + 8]
draw.rounded_rectangle(card1_bbox, radius=24, fill=(255, 255, 255), outline=(235, 235, 235), width=1)

# Light underline to separate image area from text within card (approximate; will be covered by pasted content)
img_div_y = card1_y + 320
draw.line([(card1_x + 12, img_div_y), (card1_x + card1_w - 12, img_div_y)], fill=(245, 245, 245), width=1)

# Event card 2 container (lower on the page)
card2_x = 48
card2_y = card1_y + card1_h + 60
card2_w = 1344
card2_h = 420
card2_bbox = [card2_x - 8, card2_y - 8, card2_x + card2_w + 8, card2_y + card2_h + 8]
draw.rounded_rectangle(card2_bbox, radius=24, fill=(255, 255, 255), outline=(235, 235, 235), width=1)

# Divider between cards and subsequent content
sep_y = card2_y + card2_h + 24
draw.line([(24, sep_y), (1440 - 24, sep_y)], fill=(240, 240, 240), width=1)

# Large content band background (for any wider banners or promoted area)
band_top = sep_y + 24
band_h = 200
draw.rectangle([(0, band_top), (1440, band_top + band_h)], fill=(250, 250, 250))
draw.line([(24, band_top + band_h), (1440 - 24, band_top + band_h)], fill=(245, 245, 245), width=1)

# Bottom navigation bar background and top divider
nav_h = 120
nav_top = 2960 - nav_h
draw.line([(0, nav_top), (1440, nav_top)], fill=(225, 225, 225), width=2)
draw.rectangle([(0, nav_top), (1440, 2960)], fill=(255, 255, 255))

# Subtle shadow lines beneath cards to lift them slightly
draw.line([(card1_bbox[0] + 6, card1_bbox[3] + 2), (card1_bbox[2] - 6, card1_bbox[3] + 2)], fill=(240, 240, 240), width=2)
draw.line([(card2_bbox[0] + 6, card2_bbox[3] + 2), (card2_bbox[2] - 6, card2_bbox[3] + 2)], fill=(240, 240, 240), width=2)

# Light left margin vertical guide (purely visual structure)
draw.line([(48, header_top + 8), (48, 2800)], fill=(248, 248, 248), width=1)
draw.line([(1440 - 48, header_top + 8), (1440 - 48, 2800)], fill=(248, 248, 248), width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_04_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-6/00_icon_Music.png
try:
    _c0 = get_crop(0, 187, 103)
    canvas.paste(_c0, (837, 410), _c0)
except Exception:
    pass
layout["Music"] = [837, 410, 1024, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_04_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-6/01_icon_Business.png
try:
    _c1 = get_crop(1, 241, 103)
    canvas.paste(_c1, (1036, 410), _c1)
except Exception:
    pass
layout["Business"] = [1036, 410, 1277, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_04_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-6/02_icon_Anytime.png
try:
    _c2 = get_crop(2, 400, 103)
    canvas.paste(_c2, (425, 410), _c2)
except Exception:
    pass
layout["Anytime"] = [425, 410, 825, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_04_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-6/03_icon_Filters.png
try:
    _c3 = get_crop(3, 359, 103)
    canvas.paste(_c3, (54, 410), _c3)
except Exception:
    pass
layout["Filters"] = [54, 410, 413, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_04_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-6/04_icon_Favorite_button.png
try:
    _c4 = get_crop(4, 144, 144)
    canvas.paste(_c4, (1092, 2415), _c4)
except Exception:
    pass
layout["Favorite_button"] = [1092, 2415, 1236, 2559]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_04_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-6/05_icon_Foo.png
try:
    _c5 = get_crop(5, 147, 110)
    canvas.paste(_c5, (1283, 406), _c5)
except Exception:
    pass
layout["Foo"] = [1283, 406, 1430, 516]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_04_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-6/06_icon_Favorite_button.png
try:
    _c6 = get_crop(6, 144, 144)
    canvas.paste(_c6, (1092, 1192), _c6)
except Exception:
    pass
layout["Favorite_button"] = [1092, 1192, 1236, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_04_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-6/07_icon_Overflow_menu_button.png
try:
    _c7 = get_crop(7, 144, 144)
    canvas.paste(_c7, (1236, 2415), _c7)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 2415, 1380, 2559]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_04_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-6/08_icon_Overflow_menu_button.png
try:
    _c8 = get_crop(8, 144, 144)
    canvas.paste(_c8, (1236, 1192), _c8)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 1192, 1380, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_04_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-6/09_icon_TO_SKYROCKET_SITE.png
try:
    _c9 = get_crop(9, 1344, 1175)
    canvas.paste(_c9, (48, 676), _c9)
except Exception:
    pass
layout["TO_SKYROCKET_SITE"] = [48, 676, 1392, 1851]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_04_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-6/10_icon_Foo.png
try:
    _c10 = get_crop(10, 144, 144)
    canvas.paste(_c10, (1248, 96), _c10)
except Exception:
    pass
layout["Foo"] = [1248, 96, 1392, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_04_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-6/11_icon_5.22.png
try:
    _c11 = get_crop(11, 61, 65)
    canvas.paste(_c11, (180, 0), _c11)
except Exception:
    pass
layout["5.22"] = [180, 0, 241, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_04_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-6/12_icon_Music.png
try:
    _c12 = get_crop(12, 68, 64)
    canvas.paste(_c12, (307, 0), _c12)
except Exception:
    pass
layout["Music"] = [307, 0, 375, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_04_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-6/13_icon_Music.png
try:
    _c13 = get_crop(13, 54, 65)
    canvas.paste(_c13, (246, 0), _c13)
except Exception:
    pass
layout["Music"] = [246, 0, 300, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_04_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-6/14_icon_5.22.png
try:
    _c14 = get_crop(14, 128, 117)
    canvas.paste(_c14, (52, 112), _c14)
except Exception:
    pass
layout["5.22"] = [52, 112, 180, 229]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_04_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-6/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 105, 61)
    canvas.paste(_c15, (1205, 0), _c15)
except Exception:
    pass
layout["icon_15"] = [1205, 0, 1310, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_04_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-6/16_icon_5.22.png
try:
    _c16 = get_crop(16, 60, 65)
    canvas.paste(_c16, (115, 0), _c16)
except Exception:
    pass
layout["5.22"] = [115, 0, 175, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_04_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-6/17_icon_icon_17.png
try:
    _c17 = get_crop(17, 60, 61)
    canvas.paste(_c17, (1318, 0), _c17)
except Exception:
    pass
layout["icon_17"] = [1318, 0, 1378, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_04_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-6/18_icon_Online.png
try:
    _c18 = get_crop(18, 377, 144)
    canvas.paste(_c18, (0, 259), _c18)
except Exception:
    pass
layout["Online"] = [0, 259, 377, 403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_04_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-6/19_icon_Unlock_the_Mysteries_of_Magic_and_Witchc.png
try:
    _c19 = get_crop(19, 1344, 917)
    canvas.paste(_c19, (48, 1899), _c19)
except Exception:
    pass
layout["Unlock_the_Mysteries_of_M"] = [48, 1899, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_04_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-6/20_icon_Unlock_the_Mysteries_of_Magic_and_Witchc.png
try:
    _c20 = get_crop(20, 288, 156)
    canvas.paste(_c20, (576, 2804), _c20)
except Exception:
    pass
layout["Unlock_the_Mysteries_of_M"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_04_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-6/21_icon_8.30_PM_EDT.png
try:
    _c21 = get_crop(21, 288, 156)
    canvas.paste(_c21, (288, 2804), _c21)
except Exception:
    pass
layout["8.30_PM_EDT"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_04_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-6/22_icon_icon_22.png
try:
    _c22 = get_crop(22, 50, 62)
    canvas.paste(_c22, (384, 2), _c22)
except Exception:
    pass
layout["icon_22"] = [384, 2, 434, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_04_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-6/23_icon_Unlock_the_Mysteries_of_Magic_and_Witchc.png
try:
    _c23 = get_crop(23, 288, 156)
    canvas.paste(_c23, (864, 2804), _c23)
except Exception:
    pass
layout["Unlock_the_Mysteries_of_M"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_04_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-6/24_icon_Music.png
try:
    _c24 = get_crop(24, 1344, 191)
    canvas.paste(_c24, (48, 72), _c24)
except Exception:
    pass
layout["Music"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_04_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-6/25_icon_SEO_MASTERY_HOW_TO_SKYROCKET_SITE.png
try:
    _c25 = get_crop(25, 1344, 1175)
    canvas.paste(_c25, (48, 676), _c25)
except Exception:
    pass
layout["SEO_MASTERY:_HOW_TO_SKYRO"] = [48, 676, 1392, 1851]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_04_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-6/26_icon_More.png
try:
    _c26 = get_crop(26, 288, 156)
    canvas.paste(_c26, (1152, 2804), _c26)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_04_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-6/27_icon_Promoted.png
try:
    _c27 = get_crop(27, 247, 63)
    canvas.paste(_c27, (83, 1745), _c27)
except Exception:
    pass
layout["Promoted"] = [83, 1745, 330, 1808]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_04_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-6/28_icon_5.22.png
try:
    _c28 = get_crop(28, 110, 64)
    canvas.paste(_c28, (11, 0), _c28)
except Exception:
    pass
layout["5.22"] = [11, 0, 121, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_04_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-6/29_icon_Sun_May_19.png
try:
    _c29 = get_crop(29, 288, 156)
    canvas.paste(_c29, (0, 2804), _c29)
except Exception:
    pass
layout["Sun,_May_19"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_04_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-6/30_icon_Promoted.png
try:
    _c30 = get_crop(30, 45, 59)
    canvas.paste(_c30, (283, 1747), _c30)
except Exception:
    pass
layout["Promoted"] = [283, 1747, 328, 1806]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_04_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-6/31_text_10_000_events.png
try:
    _c31 = get_crop(31, 359, 103)
    canvas.paste(_c31, (54, 410), _c31)
except Exception:
    pass
layout["10,000_events"] = [54, 410, 413, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_04_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-6/32_text_Online.png
try:
    _c32 = get_crop(32, 129, 45)
    canvas.paste(_c32, (91, 1687), _c32)
except Exception:
    pass
layout["Online"] = [91, 1687, 220, 1732]
