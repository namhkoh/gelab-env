# page_id: page_eventbrite_97cc452004d64d67bb4c5b51ffb8f6d4_02
# screenshot: 2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-4.png
# step_index: 2/7
# task: Open Eventbrite. Search Business event. Select the first one that is not promoted. Share to google keep notes.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Paint overall background
bg_color = (249, 250, 252)  # off-white dominant background
draw.rectangle([(0, 0), canvas.size], fill=bg_color)

# Status bar (top area)
status_h = 96
status_color = (189, 189, 189)  # muted gray status bar
draw.rectangle([(0, 0), (canvas.size[0], status_h)], fill=status_color)

# Header area (behind the search/header content)
header_y0 = status_h
header_y1 = 240
header_bg = (255, 255, 255)  # clean white header behind search elements
draw.rectangle([(0, header_y0), (canvas.size[0], header_y1)], fill=header_bg)

# Subtle bottom divider under header
divider_color = (225, 226, 230)
draw.line([(48, header_y1), (canvas.size[0]-48, header_y1)], fill=divider_color, width=2)

# Section: chips / filters area background strip (light tint)
chips_strip_y0 = 340
chips_strip_y1 = 460
chips_strip_color = (247, 250, 252)  # slightly different light tint
draw.rectangle([(0, chips_strip_y0), (canvas.size[0], chips_strip_y1)], fill=chips_strip_color)

# Big carousel / hero area background (behind the image row of bottles)
hero_y0 = 240
hero_y1 = 540
hero_bg = (255, 255, 255)  # keep white so pasted images show naturally
# Add a subtle half-height shadow band below hero to separate from list
draw.rectangle([(48, hero_y0), (canvas.size[0]-48, hero_y1)], fill=hero_bg)
draw.line([(48, hero_y1+8), (canvas.size[0]-48, hero_y1+8)], fill=divider_color, width=2)

# "10,000 events" heading area background (just the strip)
events_heading_y0 = hero_y1 + 28
events_heading_y1 = events_heading_y0 + 60
draw.rectangle([(48, events_heading_y0), (canvas.size[0]-48, events_heading_y1)], fill=bg_color)

# Separator under heading
draw.line([(48, events_heading_y1+16), (canvas.size[0]-48, events_heading_y1+16)], fill=divider_color, width=1)

# First event card container (rounded rectangle)
card_margin_x = 48
card1_y0 = events_heading_y1 + 36
card1_y1 = card1_y0 + 260
card_bg = (255, 255, 255)
card_shadow = (235, 236, 240)
# shadow
draw.rounded_rectangle([(card_margin_x+4, card1_y0+6), (canvas.size[0]-card_margin_x+4, card1_y1+6)],
                       radius=16, fill=card_shadow)
# card body
draw.rounded_rectangle([(card_margin_x, card1_y0), (canvas.size[0]-card_margin_x, card1_y1)],
                       radius=16, fill=card_bg)

# Thin divider between card title area and image area (within first card)
draw.line([(card_margin_x+20, card1_y1 - 6), (canvas.size[0]-card_margin_x-20, card1_y1 - 6)], fill=divider_color, width=1)

# Large image banner placeholder for featured event (dark rounded rect)
banner_y0 = card1_y1 + 36
banner_y1 = banner_y0 + 320
banner_x0 = 48
banner_x1 = canvas.size[0] - 48
banner_bg = (36, 38, 45)  # deep desaturated background for image area
banner_shadow = (28, 30, 36)
# shadow behind banner
draw.rounded_rectangle([(banner_x0+4, banner_y0+8), (banner_x1+4, banner_y1+8)], radius=20, fill=banner_shadow)
# banner body
draw.rounded_rectangle([(banner_x0, banner_y0), (banner_x1, banner_y1)], radius=20, fill=banner_bg)

# Small separator line below banner
draw.line([(48, banner_y1+28), (canvas.size[0]-48, banner_y1+28)], fill=divider_color, width=1)

# Second event card area (white card background)
card2_y0 = banner_y1 + 56
card2_y1 = card2_y0 + 300
# shadow
draw.rounded_rectangle([(card_margin_x+4, card2_y0+6), (canvas.size[0]-card_margin_x+4, card2_y1+6)],
                       radius=16, fill=card_shadow)
# card
draw.rounded_rectangle([(card_margin_x, card2_y0), (canvas.size[0]-card_margin_x, card2_y1)],
                       radius=16, fill=card_bg)

# Subtle separators between list items further down
sep_y = card2_y1 + 32
for i in range(3):
    draw.line([(48, sep_y + i*180), (canvas.size[0]-48, sep_y + i*180)], fill=divider_color, width=1)

# Large content block further down (a darker section background for promoted/featured area)
featured_y0 = 1700
featured_y1 = featured_y0 + 380
featured_margin = 48
featured_bg = (250, 250, 252)
draw.rectangle([(0, featured_y0), (canvas.size[0], featured_y1)], fill=featured_bg)
# inside a rounded dark pane for a featured image
draw.rounded_rectangle([(featured_margin, featured_y0+20), (canvas.size[0]-featured_margin, featured_y1-20)],
                       radius=18, fill=(30,32,36))

# Bottom navigation bar
nav_h = 150
nav_y0 = canvas.size[1] - nav_h
nav_y1 = canvas.size[1]
nav_bg = (255, 255, 255)
nav_border = (220, 221, 224)
draw.rectangle([(0, nav_y0), (canvas.size[0], nav_y1)], fill=nav_bg)
# top border of nav
draw.line([(0, nav_y0), (canvas.size[0], nav_y0)], fill=nav_border, width=2)

# Soft horizontal separators near bottom content to visually group items
for y in (card1_y1 + 20, banner_y1 + 20, card2_y1 + 20, featured_y1 + 10):
    draw.line([(48, y), (canvas.size[0]-48, y)], fill=divider_color, width=1)

# subtle left gutter rule for content column
gutter_x = 48
draw.line([(gutter_x, status_h), (gutter_x, canvas.size[1]-nav_h-20)], fill=(245,245,247), width=2)

# subtle right gutter rule
gutter_rx = canvas.size[0] - 48
draw.line([(gutter_rx, status_h), (gutter_rx, canvas.size[1]-nav_h-20)], fill=(245,245,247), width=2)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_02_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-4/00_icon_Music.png
try:
    _c0 = get_crop(0, 187, 103)
    canvas.paste(_c0, (837, 410), _c0)
except Exception:
    pass
layout["Music"] = [837, 410, 1024, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_02_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-4/01_icon_Business.png
try:
    _c1 = get_crop(1, 241, 103)
    canvas.paste(_c1, (1036, 410), _c1)
except Exception:
    pass
layout["Business"] = [1036, 410, 1277, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_02_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-4/02_icon_Anytime.png
try:
    _c2 = get_crop(2, 400, 103)
    canvas.paste(_c2, (425, 410), _c2)
except Exception:
    pass
layout["Anytime"] = [425, 410, 825, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_02_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-4/03_icon_Filters.png
try:
    _c3 = get_crop(3, 359, 103)
    canvas.paste(_c3, (54, 410), _c3)
except Exception:
    pass
layout["Filters"] = [54, 410, 413, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_02_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-4/04_icon_Foo.png
try:
    _c4 = get_crop(4, 154, 110)
    canvas.paste(_c4, (1282, 406), _c4)
except Exception:
    pass
layout["Foo"] = [1282, 406, 1436, 516]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_02_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-4/05_icon_IMETHOD.png
try:
    _c5 = get_crop(5, 144, 144)
    canvas.paste(_c5, (1092, 2336), _c5)
except Exception:
    pass
layout["IMETHOD"] = [1092, 2336, 1236, 2480]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_02_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-4/06_icon_IMETHOD.png
try:
    _c6 = get_crop(6, 144, 144)
    canvas.paste(_c6, (1236, 2336), _c6)
except Exception:
    pass
layout["IMETHOD"] = [1236, 2336, 1380, 2480]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_02_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-4/07_icon_9.39.png
try:
    _c7 = get_crop(7, 128, 116)
    canvas.paste(_c7, (54, 114), _c7)
except Exception:
    pass
layout["9.39"] = [54, 114, 182, 230]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_02_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-4/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 56, 61)
    canvas.paste(_c8, (247, 1), _c8)
except Exception:
    pass
layout["icon_8"] = [247, 1, 303, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_02_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-4/09_icon_Search_forae.png
try:
    _c9 = get_crop(9, 63, 63)
    canvas.paste(_c9, (311, 1), _c9)
except Exception:
    pass
layout["Search_forae"] = [311, 1, 374, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_02_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-4/10_icon_9.39.png
try:
    _c10 = get_crop(10, 56, 62)
    canvas.paste(_c10, (182, 0), _c10)
except Exception:
    pass
layout["9.39"] = [182, 0, 238, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_02_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-4/11_icon_New_York.png
try:
    _c11 = get_crop(11, 434, 144)
    canvas.paste(_c11, (0, 259), _c11)
except Exception:
    pass
layout["New_York"] = [0, 259, 434, 403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_02_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-4/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 102, 60)
    canvas.paste(_c12, (1205, 0), _c12)
except Exception:
    pass
layout["icon_12"] = [1205, 0, 1307, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_02_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-4/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 67, 59)
    canvas.paste(_c13, (1314, 0), _c13)
except Exception:
    pass
layout["icon_13"] = [1314, 0, 1381, 59]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_02_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-4/14_icon_Overflow_menu_button.png
try:
    _c14 = get_crop(14, 144, 144)
    canvas.paste(_c14, (1236, 1192), _c14)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 1192, 1380, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_02_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-4/15_icon_9.39.png
try:
    _c15 = get_crop(15, 60, 64)
    canvas.paste(_c15, (113, 0), _c15)
except Exception:
    pass
layout["9.39"] = [113, 0, 173, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_02_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-4/16_icon_GLABING_BLOOM_SOUND_COLLECTIVE.png
try:
    _c16 = get_crop(16, 1344, 996)
    canvas.paste(_c16, (48, 1820), _c16)
except Exception:
    pass
layout["GLABING;_BLOOM_SOUND_COLL"] = [48, 1820, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_02_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-4/17_icon_Search_forae.png
try:
    _c17 = get_crop(17, 51, 61)
    canvas.paste(_c17, (383, 2), _c17)
except Exception:
    pass
layout["Search_forae"] = [383, 2, 434, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_02_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-4/18_icon_Search_forae.png
try:
    _c18 = get_crop(18, 1344, 191)
    canvas.paste(_c18, (48, 72), _c18)
except Exception:
    pass
layout["Search_forae"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_02_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-4/19_icon_The_Snace_at_Irondale.png
try:
    _c19 = get_crop(19, 288, 156)
    canvas.paste(_c19, (288, 2804), _c19)
except Exception:
    pass
layout["The_Snace_at_Irondale"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_02_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-4/20_icon_slO_2Lo.png
try:
    _c20 = get_crop(20, 288, 156)
    canvas.paste(_c20, (864, 2804), _c20)
except Exception:
    pass
layout["slO_2Lo"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_02_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-4/21_icon_Favorite_button.png
try:
    _c21 = get_crop(21, 144, 144)
    canvas.paste(_c21, (1092, 1192), _c21)
except Exception:
    pass
layout["Favorite_button"] = [1092, 1192, 1236, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_02_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-4/22_icon_Promoted.png
try:
    _c22 = get_crop(22, 144, 144)
    canvas.paste(_c22, (234, 1625), _c22)
except Exception:
    pass
layout["Promoted"] = [234, 1625, 378, 1769]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_02_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-4/23_icon_Tequila_Artistic_Transformation.png
try:
    _c23 = get_crop(23, 1344, 1096)
    canvas.paste(_c23, (48, 676), _c23)
except Exception:
    pass
layout["Tequila_&_Artistic_Transf"] = [48, 676, 1392, 1772]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_02_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-4/24_icon_slO_2Lo.png
try:
    _c24 = get_crop(24, 288, 156)
    canvas.paste(_c24, (1152, 2804), _c24)
except Exception:
    pass
layout["slO_2Lo"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_02_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-4/25_icon_6.30_PM_EDT.png
try:
    _c25 = get_crop(25, 288, 156)
    canvas.paste(_c25, (576, 2804), _c25)
except Exception:
    pass
layout["6.30_PM_EDT"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_02_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-4/26_icon_10_000_events.png
try:
    _c26 = get_crop(26, 214, 295)
    canvas.paste(_c26, (217, 669), _c26)
except Exception:
    pass
layout["10,000_events"] = [217, 669, 431, 964]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_02_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-4/27_icon_Anytime.png
try:
    _c27 = get_crop(27, 210, 292)
    canvas.paste(_c27, (477, 670), _c27)
except Exception:
    pass
layout["Anytime"] = [477, 670, 687, 962]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_02_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-4/28_icon_Wed_Mar_20.png
try:
    _c28 = get_crop(28, 288, 156)
    canvas.paste(_c28, (0, 2804), _c28)
except Exception:
    pass
layout["Wed,_Mar_20"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_02_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-4/29_text_9.39.png
try:
    _c29 = get_crop(29, 94, 45)
    canvas.paste(_c29, (17, 15), _c29)
except Exception:
    pass
layout["9.39"] = [17, 15, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_02_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-4/30_text_10_000_events.png
try:
    _c30 = get_crop(30, 359, 103)
    canvas.paste(_c30, (54, 410), _c30)
except Exception:
    pass
layout["10,000_events"] = [54, 410, 413, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_02_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-4/31_text_3.20.24.png
try:
    _c31 = get_crop(31, 172, 40)
    canvas.paste(_c31, (649, 1819), _c31)
except Exception:
    pass
layout["3.20.24"] = [649, 1819, 821, 1859]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_02_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-4/32_text_Wed_Mar_20.png
try:
    _c32 = get_crop(32, 288, 156)
    canvas.paste(_c32, (0, 2804), _c32)
except Exception:
    pass
layout["Wed,_Mar_20"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_02_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-4/33_text_6.30_PM_EDT.png
try:
    _c33 = get_crop(33, 288, 156)
    canvas.paste(_c33, (288, 2804), _c33)
except Exception:
    pass
layout["6.30_PM_EDT"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_02_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-4/34_text_The_Snace_at_Irondale.png
try:
    _c34 = get_crop(34, 288, 156)
    canvas.paste(_c34, (288, 2804), _c34)
except Exception:
    pass
layout["The_Snace_at_Irondale"] = [288, 2804, 576, 2960]
