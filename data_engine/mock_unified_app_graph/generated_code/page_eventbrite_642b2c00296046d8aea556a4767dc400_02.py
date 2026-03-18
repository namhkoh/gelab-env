# page_id: page_eventbrite_642b2c00296046d8aea556a4767dc400_02
# screenshot: 2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-4.png
# step_index: 2/12
# task: Open Eventbrite. Search free events in New York. Select the first one. Follow the organizer. Read more about the event. Add it to Favorites.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw background and structural elements for the mobile UI page
# Available variables: canvas (PIL Image 1440x2960 RGB), draw (ImageDraw), font_sm, font_md, font_lg, font_xl

w, h = canvas.size

# Colors
bg_color = "#FBFBFD"            # page background (very light)
status_color = "#9AA0A6"        # top status bar (muted gray)
header_color = "#F5F6FA"        # search/header background (very pale lavender)
divider_color = "#D7D7DB"       # subtle divider lines
card_outline = "#E6E6E9"        # card border
card_shadow = "#EFEFF2"         # card shadow / lift
card_bg = "#FFFFFF"             # card background
nav_bg = "#FFFFFF"              # bottom nav background (white)
nav_div = "#E9E9EB"             # nav divider

# Dimensions
status_h = 96                   # approximate status bar height
header_h = 160                  # header/search area height (below status bar)
header_top = status_h
header_bottom = status_h + header_h

# Fill overall background
draw.rectangle([(0, 0), (w, h)], fill=bg_color)

# Status bar
draw.rectangle([(0, 0), (w, status_h)], fill=status_color)

# Header/search background (below status bar)
draw.rectangle([(0, header_top), (w, header_bottom)], fill=header_color)

# Thin divider under header
draw.line([(48, header_bottom + 6), (w - 48, header_bottom + 6)], fill=divider_color, width=2)

# Subtle horizontal separator under the filter area (approx near pills row)
filters_sep_y = header_bottom + 210
draw.line([(48, filters_sep_y), (w - 48, filters_sep_y)], fill=divider_color, width=1)

# Card 1: "Tequila & Artistic Transformation" area background
card1_x0 = 48
card1_y0 = 660
card1_w = 1344
card1_h = 1096
card1_x1 = card1_x0 + card1_w
card1_y1 = card1_y0 + card1_h
card_radius = 22

# Shadow for card1 (subtle offset)
draw.rounded_rectangle([(card1_x0 + 6, card1_y0 + 8), (card1_x1 + 6, card1_y1 + 8)],
                       radius=card_radius, fill=card_shadow, outline=None)

# Card1 background and outline
draw.rounded_rectangle([(card1_x0, card1_y0), (card1_x1, card1_y1)],
                       radius=card_radius, fill=card_bg, outline=card_outline, width=1)

# Separator above Card1 (thin)
draw.line([(48, card1_y0 - 18), (w - 48, card1_y0 - 18)], fill=divider_color, width=1)

# Card 2: Large event poster area background
card2_x0 = 48
card2_y0 = 1820
card2_w = 1344
card2_h = 996
card2_x1 = card2_x0 + card2_w
card2_y1 = card2_y0 + card2_h

# Shadow for card2
draw.rounded_rectangle([(card2_x0 + 6, card2_y0 + 8), (card2_x1 + 6, card2_y1 + 8)],
                       radius=card_radius, fill=card_shadow, outline=None)

# Card2 background and outline
draw.rounded_rectangle([(card2_x0, card2_y0), (card2_x1, card2_y1)],
                       radius=card_radius, fill=card_bg, outline=card_outline, width=1)

# Separator above Card2 (thin)
draw.line([(48, card2_y0 - 18), (w - 48, card2_y0 - 18)], fill=divider_color, width=1)

# Content band behind the image carousel (subtle)
carousel_y0 = header_bottom + 40
carousel_y1 = card1_y0 - 40
if carousel_y1 > carousel_y0 + 20:
    band_x0 = 48
    band_x1 = w - 48
    band_radius = 12
    draw.rounded_rectangle([(band_x0, carousel_y0), (band_x1, carousel_y1)],
                           radius=band_radius, fill=bg_color, outline=divider_color, width=1)

# Thin rule separating major content sections
draw.line([(48, card1_y1 + 18), (w - 48, card1_y1 + 18)], fill=divider_color, width=1)

# Bottom navigation bar background and top divider
nav_h = 110
nav_y0 = h - nav_h
draw.rectangle([(0, nav_y0), (w, h)], fill=nav_bg)
draw.line([(0, nav_y0), (w, nav_y0)], fill=nav_div, width=2)

# Small center highlight under search (subtle)
search_underline_y = header_top + 112
draw.line([(120, search_underline_y), (w - 120, search_underline_y)], fill=divider_color, width=1)

# Add subtle page-wide vertical gutters (left and right margins)
gutter_width = 48
draw.line([(gutter_width, 0), (gutter_width, h)], fill=bg_color, width=1)
draw.line([(w - gutter_width, 0), (w - gutter_width, h)], fill=bg_color, width=1)

# Final top and bottom edge subtle strokes to help separate status/header and content
draw.line([(0, status_h), (w, status_h)], fill=divider_color, width=1)
draw.line([(0, h - 1), (w, h - 1)], fill=divider_color, width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_02_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-4/00_icon_Music.png
try:
    _c0 = get_crop(0, 187, 103)
    canvas.paste(_c0, (837, 410), _c0)
except Exception:
    pass
layout["Music"] = [837, 410, 1024, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_02_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-4/01_icon_Business.png
try:
    _c1 = get_crop(1, 241, 103)
    canvas.paste(_c1, (1036, 410), _c1)
except Exception:
    pass
layout["Business"] = [1036, 410, 1277, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_02_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-4/02_icon_Anytime.png
try:
    _c2 = get_crop(2, 400, 103)
    canvas.paste(_c2, (425, 410), _c2)
except Exception:
    pass
layout["Anytime"] = [425, 410, 825, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_02_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-4/03_icon_Filters.png
try:
    _c3 = get_crop(3, 359, 103)
    canvas.paste(_c3, (54, 410), _c3)
except Exception:
    pass
layout["Filters"] = [54, 410, 413, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_02_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-4/04_icon_Foo.png
try:
    _c4 = get_crop(4, 154, 110)
    canvas.paste(_c4, (1282, 406), _c4)
except Exception:
    pass
layout["Foo"] = [1282, 406, 1436, 516]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_02_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-4/05_icon_IMETHOD.png
try:
    _c5 = get_crop(5, 144, 144)
    canvas.paste(_c5, (1092, 2336), _c5)
except Exception:
    pass
layout["IMETHOD"] = [1092, 2336, 1236, 2480]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_02_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-4/06_icon_IMETHOD.png
try:
    _c6 = get_crop(6, 144, 144)
    canvas.paste(_c6, (1236, 2336), _c6)
except Exception:
    pass
layout["IMETHOD"] = [1236, 2336, 1380, 2480]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_02_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-4/07_icon_9.09.png
try:
    _c7 = get_crop(7, 128, 116)
    canvas.paste(_c7, (54, 114), _c7)
except Exception:
    pass
layout["9.09"] = [54, 114, 182, 230]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_02_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-4/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 56, 61)
    canvas.paste(_c8, (247, 1), _c8)
except Exception:
    pass
layout["icon_8"] = [247, 1, 303, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_02_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-4/09_icon_Search_forae.png
try:
    _c9 = get_crop(9, 63, 63)
    canvas.paste(_c9, (311, 1), _c9)
except Exception:
    pass
layout["Search_forae"] = [311, 1, 374, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_02_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-4/10_icon_New_York.png
try:
    _c10 = get_crop(10, 434, 144)
    canvas.paste(_c10, (0, 259), _c10)
except Exception:
    pass
layout["New_York"] = [0, 259, 434, 403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_02_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-4/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 102, 60)
    canvas.paste(_c11, (1205, 0), _c11)
except Exception:
    pass
layout["icon_11"] = [1205, 0, 1307, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_02_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-4/12_icon_9.09.png
try:
    _c12 = get_crop(12, 55, 62)
    canvas.paste(_c12, (182, 0), _c12)
except Exception:
    pass
layout["9.09"] = [182, 0, 237, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_02_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-4/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 66, 59)
    canvas.paste(_c13, (1314, 0), _c13)
except Exception:
    pass
layout["icon_13"] = [1314, 0, 1380, 59]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_02_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-4/14_icon_Overflow_menu_button.png
try:
    _c14 = get_crop(14, 144, 144)
    canvas.paste(_c14, (1236, 1192), _c14)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 1192, 1380, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_02_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-4/15_icon_GLABING_BLOOM_SOUND_COLLECTIVE.png
try:
    _c15 = get_crop(15, 1344, 996)
    canvas.paste(_c15, (48, 1820), _c15)
except Exception:
    pass
layout["GLABING;_BLOOM_SOUND_COLL"] = [48, 1820, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_02_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-4/16_icon_9.09.png
try:
    _c16 = get_crop(16, 59, 64)
    canvas.paste(_c16, (113, 0), _c16)
except Exception:
    pass
layout["9.09"] = [113, 0, 172, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_02_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-4/17_icon_Search_forae.png
try:
    _c17 = get_crop(17, 51, 61)
    canvas.paste(_c17, (383, 2), _c17)
except Exception:
    pass
layout["Search_forae"] = [383, 2, 434, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_02_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-4/18_icon_Search_forae.png
try:
    _c18 = get_crop(18, 1344, 191)
    canvas.paste(_c18, (48, 72), _c18)
except Exception:
    pass
layout["Search_forae"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_02_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-4/19_icon_The_Snace_at_Irondale.png
try:
    _c19 = get_crop(19, 288, 156)
    canvas.paste(_c19, (288, 2804), _c19)
except Exception:
    pass
layout["The_Snace_at_Irondale"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_02_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-4/20_icon_slO_2Lo.png
try:
    _c20 = get_crop(20, 288, 156)
    canvas.paste(_c20, (864, 2804), _c20)
except Exception:
    pass
layout["slO_2Lo"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_02_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-4/21_icon_Favorite_button.png
try:
    _c21 = get_crop(21, 144, 144)
    canvas.paste(_c21, (1092, 1192), _c21)
except Exception:
    pass
layout["Favorite_button"] = [1092, 1192, 1236, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_02_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-4/22_icon_Promoted.png
try:
    _c22 = get_crop(22, 244, 66)
    canvas.paste(_c22, (84, 1665), _c22)
except Exception:
    pass
layout["Promoted"] = [84, 1665, 328, 1731]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_02_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-4/23_icon_Tequila_Artistic_Transformation.png
try:
    _c23 = get_crop(23, 1344, 1096)
    canvas.paste(_c23, (48, 676), _c23)
except Exception:
    pass
layout["Tequila_&_Artistic_Transf"] = [48, 676, 1392, 1772]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_02_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-4/24_icon_slO_2Lo.png
try:
    _c24 = get_crop(24, 288, 156)
    canvas.paste(_c24, (1152, 2804), _c24)
except Exception:
    pass
layout["slO_2Lo"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_02_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-4/25_icon_6.30_PM_EDT.png
try:
    _c25 = get_crop(25, 288, 156)
    canvas.paste(_c25, (576, 2804), _c25)
except Exception:
    pass
layout["6.30_PM_EDT"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_02_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-4/26_icon_Anytime.png
try:
    _c26 = get_crop(26, 210, 292)
    canvas.paste(_c26, (477, 670), _c26)
except Exception:
    pass
layout["Anytime"] = [477, 670, 687, 962]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_02_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-4/27_icon_10_000_events.png
try:
    _c27 = get_crop(27, 214, 295)
    canvas.paste(_c27, (217, 669), _c27)
except Exception:
    pass
layout["10,000_events"] = [217, 669, 431, 964]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_02_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-4/28_icon_Wed_Mar_20.png
try:
    _c28 = get_crop(28, 288, 156)
    canvas.paste(_c28, (0, 2804), _c28)
except Exception:
    pass
layout["Wed,_Mar_20"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_02_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-4/29_text_9.09.png
try:
    _c29 = get_crop(29, 94, 45)
    canvas.paste(_c29, (17, 15), _c29)
except Exception:
    pass
layout["9.09"] = [17, 15, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_02_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-4/30_text_10_000_events.png
try:
    _c30 = get_crop(30, 359, 103)
    canvas.paste(_c30, (54, 410), _c30)
except Exception:
    pass
layout["10,000_events"] = [54, 410, 413, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_02_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-4/31_text_3.20.24.png
try:
    _c31 = get_crop(31, 172, 40)
    canvas.paste(_c31, (649, 1819), _c31)
except Exception:
    pass
layout["3.20.24"] = [649, 1819, 821, 1859]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_02_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-4/32_text_Wed_Mar_20.png
try:
    _c32 = get_crop(32, 288, 156)
    canvas.paste(_c32, (0, 2804), _c32)
except Exception:
    pass
layout["Wed,_Mar_20"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_02_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-4/33_text_6.30_PM_EDT.png
try:
    _c33 = get_crop(33, 288, 156)
    canvas.paste(_c33, (288, 2804), _c33)
except Exception:
    pass
layout["6.30_PM_EDT"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_02_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-4/34_text_The_Snace_at_Irondale.png
try:
    _c34 = get_crop(34, 288, 156)
    canvas.paste(_c34, (288, 2804), _c34)
except Exception:
    pass
layout["The_Snace_at_Irondale"] = [288, 2804, 576, 2960]
