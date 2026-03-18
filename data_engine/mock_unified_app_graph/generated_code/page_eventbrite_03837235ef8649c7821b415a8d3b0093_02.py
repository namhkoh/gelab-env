# page_id: page_eventbrite_03837235ef8649c7821b415a8d3b0093_02
# screenshot: 2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-4.png
# step_index: 2/8
# task: Open Eventbrite. Locate the 'Conference' category. Filter the results to only show virtual events. Choose the first event from the results. What is the duration of this event?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw UI background and structural elements for the Eventbrite-like mobile UI.
# Available variables:
# - canvas: PIL Image (1440x2960 RGB)
# - draw: PIL ImageDraw object
# - font_sm, font_md, font_lg, font_xl

# Top status bar
status_bar_h = 72
draw.rectangle([(0, 0), (1440, status_bar_h)], fill="#2f3136")

# Header / search area background (large search input region)
search_top = status_bar_h
search_h = 120
search_pad_h = 24
draw.rectangle([(0, search_top), (1440, search_top + search_h)], fill="#fbfbfd")

# Subtle bottom divider under search area
divider_y = search_top + search_h
draw.line([(48, divider_y), (1392, divider_y)], fill="#e0e0e6", width=2)

# Filter pills row - draw only a faint guiding background strip (do NOT draw pill content)
filters_row_top = divider_y + 24
filters_row_h = 120
# Keep it very subtle so the actual pill shapes (which will be pasted) remain prominent.
draw.rectangle([(0, filters_row_top), (1440, filters_row_top + filters_row_h)], fill="#ffffff")

# Divider under filters/pills
filters_div_y = filters_row_top + filters_row_h + 8
draw.line([(48, filters_div_y), (1392, filters_div_y)], fill="#ecebf0", width=1)

# Main background (ensure a slightly warm-white overall canvas tone)
draw.rectangle([(0, filters_div_y + 1), (1440, 2960)], fill="#fbfbfd")

# Render first event card background and image container (rounded)
card_margin_x = 48
first_image_top = filters_div_y + 40
first_image_bottom = first_image_top + 400
card_radius = 24

# subtle shadow for first card (a faint offset darker rect)
shadow_offset = 8
draw.rounded_rectangle(
    [(card_margin_x + shadow_offset, first_image_top + shadow_offset),
     (1440 - card_margin_x + shadow_offset - 48, first_image_bottom + shadow_offset)],
    radius=card_radius, fill="#e9e9ee"
)

# image background (dark area representing where image will be pasted)
draw.rounded_rectangle(
    [(card_margin_x, first_image_top), (1440 - card_margin_x - 48, first_image_bottom)],
    radius=card_radius, fill="#171717"
)

# Content block under first image (white card area for title / meta)
first_content_top = first_image_bottom + 24
first_content_bottom = first_content_top + 220
draw.rounded_rectangle(
    [(card_margin_x, first_content_top), (1440 - card_margin_x - 48, first_content_bottom)],
    radius=14, fill="#ffffff"
)
# subtle top divider on content card
draw.line([(card_margin_x + 24, first_content_top + 10), (1440 - card_margin_x - 48 - 24, first_content_top + 10)], fill="#f0eff4", width=1)

# Separator between first and second event
sep_y = first_content_bottom + 32
draw.line([(48, sep_y), (1392, sep_y)], fill="#ececf2", width=1)

# Second event image container (rounded)
second_image_top = sep_y + 24
second_image_bottom = second_image_top + 360

draw.rounded_rectangle(
    [(card_margin_x + shadow_offset, second_image_top + shadow_offset),
     (1440 - card_margin_x + shadow_offset - 48, second_image_bottom + shadow_offset)],
    radius=card_radius, fill="#e9e9ee"
)

draw.rounded_rectangle(
    [(card_margin_x, second_image_top), (1440 - card_margin_x - 48, second_image_bottom)],
    radius=card_radius, fill="#171717"
)

# Content block under second image
second_content_top = second_image_bottom + 24
second_content_bottom = second_content_top + 220
draw.rounded_rectangle(
    [(card_margin_x, second_content_top), (1440 - card_margin_x - 48, second_content_bottom)],
    radius=14, fill="#ffffff"
)
draw.line([(card_margin_x + 24, second_content_top + 10), (1440 - card_margin_x - 48 - 24, second_content_top + 10)], fill="#f0eff4", width=1)

# Large section header "10,000 events" area - draw only the spacing and small decorative underline (no text)
events_header_top = filters_div_y + 4
draw.line([(48, events_header_top + 64), (1392, events_header_top + 64)], fill="#efeef3", width=1)

# Bottom navigation bar area - background and top divider
nav_h = 120
nav_top = 2960 - nav_h
draw.rectangle([(0, nav_top), (1440, 2960)], fill="#ffffff")
draw.line([(48, nav_top), (1392, nav_top)], fill="#e6e6eb", width=1)

# Small central indicator area on nav (subtle)
indicator_radius = 4
draw.ellipse([(720 - indicator_radius, nav_top + 12 - indicator_radius), (720 + indicator_radius, nav_top + 12 + indicator_radius)], fill="#ff6a00")

# Final subtle overall left/right content guide lines (margins)
draw.line([(card_margin_x, status_bar_h), (card_margin_x, 2960 - nav_h)], fill="#ffffff00")  # invisible guide (no-op)
draw.line([(1440 - card_margin_x - 48, status_bar_h), (1440 - card_margin_x - 48, 2960 - nav_h)], fill="#ffffff00")

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_02_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-4/00_icon_Music.png
try:
    _c0 = get_crop(0, 187, 103)
    canvas.paste(_c0, (837, 410), _c0)
except Exception:
    pass
layout["Music"] = [837, 410, 1024, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_02_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-4/01_icon_Business.png
try:
    _c1 = get_crop(1, 241, 103)
    canvas.paste(_c1, (1036, 410), _c1)
except Exception:
    pass
layout["Business"] = [1036, 410, 1277, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_02_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-4/02_icon_Anytime.png
try:
    _c2 = get_crop(2, 400, 103)
    canvas.paste(_c2, (425, 410), _c2)
except Exception:
    pass
layout["Anytime"] = [425, 410, 825, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_02_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-4/03_icon_Filters.png
try:
    _c3 = get_crop(3, 359, 103)
    canvas.paste(_c3, (54, 410), _c3)
except Exception:
    pass
layout["Filters"] = [54, 410, 413, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_02_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-4/04_icon_Favorite_button.png
try:
    _c4 = get_crop(4, 144, 144)
    canvas.paste(_c4, (1092, 2415), _c4)
except Exception:
    pass
layout["Favorite_button"] = [1092, 2415, 1236, 2559]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_02_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-4/05_icon_Overflow_menu_button.png
try:
    _c5 = get_crop(5, 144, 144)
    canvas.paste(_c5, (1236, 2415), _c5)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 2415, 1380, 2559]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_02_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-4/06_icon_Foo.png
try:
    _c6 = get_crop(6, 132, 110)
    canvas.paste(_c6, (1284, 406), _c6)
except Exception:
    pass
layout["Foo"] = [1284, 406, 1416, 516]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_02_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-4/07_icon_EcoMmcR.png
try:
    _c7 = get_crop(7, 144, 144)
    canvas.paste(_c7, (1092, 1192), _c7)
except Exception:
    pass
layout["EcoMmcR"] = [1092, 1192, 1236, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_02_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-4/08_icon_EcoMmcR.png
try:
    _c8 = get_crop(8, 144, 144)
    canvas.paste(_c8, (1236, 1192), _c8)
except Exception:
    pass
layout["EcoMmcR"] = [1236, 1192, 1380, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_02_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-4/09_icon_4.41.png
try:
    _c9 = get_crop(9, 128, 115)
    canvas.paste(_c9, (54, 114), _c9)
except Exception:
    pass
layout["4.41"] = [54, 114, 182, 229]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_02_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-4/10_icon_0_Zipify_Apps.png
try:
    _c10 = get_crop(10, 1344, 1175)
    canvas.paste(_c10, (48, 676), _c10)
except Exception:
    pass
layout["0_Zipify_Apps"] = [48, 676, 1392, 1851]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_02_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-4/11_icon_Search_forae.png
try:
    _c11 = get_crop(11, 67, 61)
    canvas.paste(_c11, (308, 1), _c11)
except Exception:
    pass
layout["Search_forae"] = [308, 1, 375, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_02_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-4/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 54, 62)
    canvas.paste(_c12, (246, 1), _c12)
except Exception:
    pass
layout["icon_12"] = [246, 1, 300, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_02_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-4/13_icon_4.41.png
try:
    _c13 = get_crop(13, 61, 64)
    canvas.paste(_c13, (113, 0), _c13)
except Exception:
    pass
layout["4.41"] = [113, 0, 174, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_02_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-4/14_icon_4.41.png
try:
    _c14 = get_crop(14, 59, 63)
    canvas.paste(_c14, (182, 0), _c14)
except Exception:
    pass
layout["4.41"] = [182, 0, 241, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_02_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-4/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 76, 60)
    canvas.paste(_c15, (1209, 0), _c15)
except Exception:
    pass
layout["icon_15"] = [1209, 0, 1285, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_02_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-4/16_icon_icon_16.png
try:
    _c16 = get_crop(16, 63, 59)
    canvas.paste(_c16, (1316, 0), _c16)
except Exception:
    pass
layout["icon_16"] = [1316, 0, 1379, 59]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_02_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-4/17_icon_Search_forae.png
try:
    _c17 = get_crop(17, 1344, 191)
    canvas.paste(_c17, (48, 72), _c17)
except Exception:
    pass
layout["Search_forae"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_02_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-4/18_icon_San_Francisco.png
try:
    _c18 = get_crop(18, 536, 144)
    canvas.paste(_c18, (0, 259), _c18)
except Exception:
    pass
layout["San_Francisco"] = [0, 259, 536, 403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_02_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-4/19_icon_How_To_Start_a_Business_Wholesaling_Real.png
try:
    _c19 = get_crop(19, 1344, 917)
    canvas.paste(_c19, (48, 1899), _c19)
except Exception:
    pass
layout["How_To_Start_a_Business,_"] = [48, 1899, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_02_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-4/20_icon_28_._7_00_PM_EDT.png
try:
    _c20 = get_crop(20, 288, 156)
    canvas.paste(_c20, (288, 2804), _c20)
except Exception:
    pass
layout["28_._7:00_PM_EDT"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_02_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-4/21_icon_Search_forae.png
try:
    _c21 = get_crop(21, 52, 60)
    canvas.paste(_c21, (383, 3), _c21)
except Exception:
    pass
layout["Search_forae"] = [383, 3, 435, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_02_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-4/22_icon_4.41.png
try:
    _c22 = get_crop(22, 92, 63)
    canvas.paste(_c22, (11, 0), _c22)
except Exception:
    pass
layout["4.41"] = [11, 0, 103, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_02_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-4/23_icon_Promoted.png
try:
    _c23 = get_crop(23, 240, 66)
    canvas.paste(_c23, (87, 1743), _c23)
except Exception:
    pass
layout["Promoted"] = [87, 1743, 327, 1809]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_02_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-4/24_icon_How_To_Start_a_Business_Wholesaling_Real.png
try:
    _c24 = get_crop(24, 288, 156)
    canvas.paste(_c24, (864, 2804), _c24)
except Exception:
    pass
layout["How_To_Start_a_Business,_"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_02_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-4/25_icon_How_To_Start_a_Business_Wholesaling_Real.png
try:
    _c25 = get_crop(25, 288, 156)
    canvas.paste(_c25, (576, 2804), _c25)
except Exception:
    pass
layout["How_To_Start_a_Business,_"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_02_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-4/26_icon_icon_26.png
try:
    _c26 = get_crop(26, 39, 60)
    canvas.paste(_c26, (1275, 0), _c26)
except Exception:
    pass
layout["icon_26"] = [1275, 0, 1314, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_02_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-4/27_icon_Sun.png
try:
    _c27 = get_crop(27, 288, 156)
    canvas.paste(_c27, (0, 2804), _c27)
except Exception:
    pass
layout["Sun,"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_02_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-4/28_icon_More.png
try:
    _c28 = get_crop(28, 288, 156)
    canvas.paste(_c28, (1152, 2804), _c28)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_02_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-4/29_text_10_000_events.png
try:
    _c29 = get_crop(29, 359, 103)
    canvas.paste(_c29, (54, 410), _c29)
except Exception:
    pass
layout["10,000_events"] = [54, 410, 413, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_02_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-4/30_text_Free.png
try:
    _c30 = get_crop(30, 80, 38)
    canvas.paste(_c30, (117, 1391), _c30)
except Exception:
    pass
layout["Free"] = [117, 1391, 197, 1429]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_02_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-4/31_text_ADVERTISING_MASTERCLASS_BUILDING_AN.png
try:
    _c31 = get_crop(31, 1344, 1175)
    canvas.paste(_c31, (48, 676), _c31)
except Exception:
    pass
layout["ADVERTISING_MASTERCLASS:_"] = [48, 676, 1392, 1851]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_02_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-4/32_text_EIGHT_FIGURE_BUSINESS.png
try:
    _c32 = get_crop(32, 1344, 1175)
    canvas.paste(_c32, (48, 676), _c32)
except Exception:
    pass
layout["EIGHT_FIGURE_BUSINESS"] = [48, 676, 1392, 1851]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_02_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-4/33_text_Wed.png
try:
    _c33 = get_crop(33, 107, 50)
    canvas.paste(_c33, (93, 1619), _c33)
except Exception:
    pass
layout["Wed,"] = [93, 1619, 200, 1669]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_02_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-4/34_text_24.png
try:
    _c34 = get_crop(34, 64, 43)
    canvas.paste(_c34, (276, 1622), _c34)
except Exception:
    pass
layout["24"] = [276, 1622, 340, 1665]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_02_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-4/35_text_12_00_PM_EDT.png
try:
    _c35 = get_crop(35, 277, 48)
    canvas.paste(_c35, (359, 1619), _c35)
except Exception:
    pass
layout["12:00_PM_EDT"] = [359, 1619, 636, 1667]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_02_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-4/36_text_Online.png
try:
    _c36 = get_crop(36, 129, 45)
    canvas.paste(_c36, (91, 1687), _c36)
except Exception:
    pass
layout["Online"] = [91, 1687, 220, 1732]
