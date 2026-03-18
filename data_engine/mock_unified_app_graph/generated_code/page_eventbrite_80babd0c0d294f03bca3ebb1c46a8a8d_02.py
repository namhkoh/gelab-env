# page_id: page_eventbrite_80babd0c0d294f03bca3ebb1c46a8a8d_02
# screenshot: 2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-4.png
# step_index: 2/8
# task: Open Eventbrite. Search Art event in New York. Select the second one. Record its location and time in Google Keep Notes. Follow the organizer.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw background and structural elements for the UI page using provided `canvas` and `draw`.

# Canvas assumed: 1440x2960 RGB, white by default.
# Colors
bg_offwhite = (250, 250, 252)
status_bar_gray = (189, 189, 189)
divider_gray = (226, 227, 230)
card_white = (255, 255, 255)
card_border = (236, 237, 240)
chip_band = (239, 249, 255)
nav_top_divider = (222, 223, 226)

# Fill overall background with the dominant off-white
draw.rectangle((0, 0, 1440, 2960), fill=bg_offwhite)

# Status bar area (top)
status_h = 96
draw.rectangle((0, 0, 1440, status_h), fill=status_bar_gray)

# Header/Search area (below status)
header_top = status_h
header_bottom = 220
# subtle white header panel (slightly brighter than bg)
draw.rectangle((0, header_top, 1440, header_bottom), fill=card_white)
# bottom divider under header
draw.line((48, header_bottom, 1392, header_bottom), fill=divider_gray, width=2)

# Light band behind location / filter chips area (keeps chips visible but doesn't redraw them)
chips_band_top = 320
chips_band_bottom = 520
draw.rectangle((0, chips_band_top, 1440, chips_band_bottom), fill=chip_band)
# subtle top divider for chips band
draw.line((48, chips_band_top, 1392, chips_band_top), fill=divider_gray, width=1)
# subtle bottom divider for chips band
draw.line((48, chips_band_bottom, 1392, chips_band_bottom), fill=divider_gray, width=1)

# Large section header divider (for the "10,000 events" heading area)
section_div_y = 560
draw.line((48, section_div_y, 1392, section_div_y), fill=divider_gray, width=2)

# Event card 1 background (rounded rectangle)
card1_x0, card1_y0 = 48, 676
card1_w, card1_h = 1344, 1096
card1_x1, card1_y1 = card1_x0 + card1_w, card1_y0 + card1_h
draw.rounded_rectangle((card1_x0, card1_y0, card1_x1, card1_y1),
                       radius=24, fill=card_white, outline=card_border, width=2)

# Divider line between image area and text within card1 (subtle, across card)
# place divider roughly 420px from card top (visual divider for image/content)
inner_div_y1 = card1_y0 + 420
draw.line((card1_x0 + 20, inner_div_y1, card1_x1 - 20, inner_div_y1), fill=divider_gray, width=1)

# Small subtle shadow below card1 (simple darker line)
draw.line((card1_x0 + 6, card1_y1 + 2, card1_x1 - 6, card1_y1 + 2), fill=card_border, width=2)

# Event card 2 background (rounded rectangle)
card2_x0, card2_y0 = 48, 1820
card2_w, card2_h = 1344, 996
card2_x1, card2_y1 = card2_x0 + card2_w, card2_y0 + card2_h
draw.rounded_rectangle((card2_x0, card2_y0, card2_x1, card2_y1),
                       radius=24, fill=card_white, outline=card_border, width=2)

# Divider inside card2 (for image vs text strip)
inner_div_y2 = card2_y0 + 360
draw.line((card2_x0 + 20, inner_div_y2, card2_x1 - 20, inner_div_y2), fill=divider_gray, width=1)

# Subtle shadow under card2
draw.line((card2_x0 + 6, card2_y1 + 2, card2_x1 - 6, card2_y1 + 2), fill=card_border, width=2)

# Horizontal separators between list sections (across full content width, with side padding)
sep_x0, sep_x1 = 48, 1392
separators = [section_div_y + 300, card1_y1 + 40, card2_y1 + 40]
for y in separators:
    if 0 < y < 2960:
        draw.line((sep_x0, y, sep_x1, y), fill=divider_gray, width=1)

# Bottom navigation bar background and top divider
nav_h = 120
nav_top = 2960 - nav_h
draw.rectangle((0, nav_top, 1440, 2960), fill=card_white)
draw.line((0, nav_top, 1440, nav_top), fill=nav_top_divider, width=2)

# Small indicator bar above nav for safe separation
draw.line((48, nav_top - 12, 1392, nav_top - 12), fill=divider_gray, width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_02_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-4/00_icon_Music.png
try:
    _c0 = get_crop(0, 187, 103)
    canvas.paste(_c0, (837, 410), _c0)
except Exception:
    pass
layout["Music"] = [837, 410, 1024, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_02_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-4/01_icon_Business.png
try:
    _c1 = get_crop(1, 241, 103)
    canvas.paste(_c1, (1036, 410), _c1)
except Exception:
    pass
layout["Business"] = [1036, 410, 1277, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_02_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-4/02_icon_Anytime.png
try:
    _c2 = get_crop(2, 400, 103)
    canvas.paste(_c2, (425, 410), _c2)
except Exception:
    pass
layout["Anytime"] = [425, 410, 825, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_02_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-4/03_icon_Filters.png
try:
    _c3 = get_crop(3, 359, 103)
    canvas.paste(_c3, (54, 410), _c3)
except Exception:
    pass
layout["Filters"] = [54, 410, 413, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_02_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-4/04_icon_Foo.png
try:
    _c4 = get_crop(4, 154, 110)
    canvas.paste(_c4, (1282, 406), _c4)
except Exception:
    pass
layout["Foo"] = [1282, 406, 1436, 516]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_02_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-4/05_icon_IMETHOD.png
try:
    _c5 = get_crop(5, 144, 144)
    canvas.paste(_c5, (1092, 2336), _c5)
except Exception:
    pass
layout["IMETHOD"] = [1092, 2336, 1236, 2480]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_02_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-4/06_icon_IMETHOD.png
try:
    _c6 = get_crop(6, 144, 144)
    canvas.paste(_c6, (1236, 2336), _c6)
except Exception:
    pass
layout["IMETHOD"] = [1236, 2336, 1380, 2480]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_02_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-4/07_icon_9.25.png
try:
    _c7 = get_crop(7, 128, 116)
    canvas.paste(_c7, (54, 114), _c7)
except Exception:
    pass
layout["9.25"] = [54, 114, 182, 230]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_02_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-4/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 56, 61)
    canvas.paste(_c8, (247, 1), _c8)
except Exception:
    pass
layout["icon_8"] = [247, 1, 303, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_02_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-4/09_icon_Search_forae.png
try:
    _c9 = get_crop(9, 63, 63)
    canvas.paste(_c9, (311, 1), _c9)
except Exception:
    pass
layout["Search_forae"] = [311, 1, 374, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_02_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-4/10_icon_New_York.png
try:
    _c10 = get_crop(10, 434, 144)
    canvas.paste(_c10, (0, 259), _c10)
except Exception:
    pass
layout["New_York"] = [0, 259, 434, 403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_02_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-4/11_icon_9.25.png
try:
    _c11 = get_crop(11, 55, 62)
    canvas.paste(_c11, (182, 0), _c11)
except Exception:
    pass
layout["9.25"] = [182, 0, 237, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_02_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-4/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 102, 60)
    canvas.paste(_c12, (1205, 0), _c12)
except Exception:
    pass
layout["icon_12"] = [1205, 0, 1307, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_02_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-4/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 67, 59)
    canvas.paste(_c13, (1314, 0), _c13)
except Exception:
    pass
layout["icon_13"] = [1314, 0, 1381, 59]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_02_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-4/14_icon_Overflow_menu_button.png
try:
    _c14 = get_crop(14, 144, 144)
    canvas.paste(_c14, (1236, 1192), _c14)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 1192, 1380, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_02_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-4/15_icon_9.25.png
try:
    _c15 = get_crop(15, 59, 64)
    canvas.paste(_c15, (114, 0), _c15)
except Exception:
    pass
layout["9.25"] = [114, 0, 173, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_02_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-4/16_icon_GLABING_BLOOM_SOUND_COLLECTIVE.png
try:
    _c16 = get_crop(16, 1344, 996)
    canvas.paste(_c16, (48, 1820), _c16)
except Exception:
    pass
layout["GLABING;_BLOOM_SOUND_COLL"] = [48, 1820, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_02_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-4/17_icon_Search_forae.png
try:
    _c17 = get_crop(17, 51, 61)
    canvas.paste(_c17, (383, 2), _c17)
except Exception:
    pass
layout["Search_forae"] = [383, 2, 434, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_02_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-4/18_icon_Search_forae.png
try:
    _c18 = get_crop(18, 1344, 191)
    canvas.paste(_c18, (48, 72), _c18)
except Exception:
    pass
layout["Search_forae"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_02_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-4/19_icon_The_Snace_at_Irondale.png
try:
    _c19 = get_crop(19, 288, 156)
    canvas.paste(_c19, (288, 2804), _c19)
except Exception:
    pass
layout["The_Snace_at_Irondale"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_02_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-4/20_icon_slO_2Lo.png
try:
    _c20 = get_crop(20, 288, 156)
    canvas.paste(_c20, (864, 2804), _c20)
except Exception:
    pass
layout["slO_2Lo"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_02_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-4/21_icon_Favorite_button.png
try:
    _c21 = get_crop(21, 144, 144)
    canvas.paste(_c21, (1092, 1192), _c21)
except Exception:
    pass
layout["Favorite_button"] = [1092, 1192, 1236, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_02_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-4/22_icon_Promoted.png
try:
    _c22 = get_crop(22, 144, 144)
    canvas.paste(_c22, (234, 1625), _c22)
except Exception:
    pass
layout["Promoted"] = [234, 1625, 378, 1769]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_02_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-4/23_icon_Tequila_Artistic_Transformation.png
try:
    _c23 = get_crop(23, 1344, 1096)
    canvas.paste(_c23, (48, 676), _c23)
except Exception:
    pass
layout["Tequila_&_Artistic_Transf"] = [48, 676, 1392, 1772]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_02_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-4/24_icon_slO_2Lo.png
try:
    _c24 = get_crop(24, 288, 156)
    canvas.paste(_c24, (1152, 2804), _c24)
except Exception:
    pass
layout["slO_2Lo"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_02_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-4/25_icon_6.30_PM_EDT.png
try:
    _c25 = get_crop(25, 288, 156)
    canvas.paste(_c25, (576, 2804), _c25)
except Exception:
    pass
layout["6.30_PM_EDT"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_02_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-4/26_icon_Anytime.png
try:
    _c26 = get_crop(26, 210, 292)
    canvas.paste(_c26, (477, 670), _c26)
except Exception:
    pass
layout["Anytime"] = [477, 670, 687, 962]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_02_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-4/27_icon_10_000_events.png
try:
    _c27 = get_crop(27, 214, 295)
    canvas.paste(_c27, (217, 669), _c27)
except Exception:
    pass
layout["10,000_events"] = [217, 669, 431, 964]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_02_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-4/28_icon_Wed_Mar_20.png
try:
    _c28 = get_crop(28, 288, 156)
    canvas.paste(_c28, (0, 2804), _c28)
except Exception:
    pass
layout["Wed,_Mar_20"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_02_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-4/29_text_9.25.png
try:
    _c29 = get_crop(29, 94, 45)
    canvas.paste(_c29, (20, 15), _c29)
except Exception:
    pass
layout["9.25"] = [20, 15, 114, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_02_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-4/30_text_10_000_events.png
try:
    _c30 = get_crop(30, 359, 103)
    canvas.paste(_c30, (54, 410), _c30)
except Exception:
    pass
layout["10,000_events"] = [54, 410, 413, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_02_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-4/31_text_3.20.24.png
try:
    _c31 = get_crop(31, 172, 40)
    canvas.paste(_c31, (649, 1819), _c31)
except Exception:
    pass
layout["3.20.24"] = [649, 1819, 821, 1859]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_02_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-4/32_text_Wed_Mar_20.png
try:
    _c32 = get_crop(32, 288, 156)
    canvas.paste(_c32, (0, 2804), _c32)
except Exception:
    pass
layout["Wed,_Mar_20"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_02_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-4/33_text_6.30_PM_EDT.png
try:
    _c33 = get_crop(33, 288, 156)
    canvas.paste(_c33, (288, 2804), _c33)
except Exception:
    pass
layout["6.30_PM_EDT"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_02_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-4/34_text_The_Snace_at_Irondale.png
try:
    _c34 = get_crop(34, 288, 156)
    canvas.paste(_c34, (288, 2804), _c34)
except Exception:
    pass
layout["The_Snace_at_Irondale"] = [288, 2804, 576, 2960]
