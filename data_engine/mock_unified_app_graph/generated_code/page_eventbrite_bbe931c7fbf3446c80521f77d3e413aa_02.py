# page_id: page_eventbrite_bbe931c7fbf3446c80521f77d3e413aa_02
# screenshot: 2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-4.png
# step_index: 2/20
# task: Open Eventbrite. Search free events in Los Angeles. Select the first one. Follow the organizer. Read more about the event. Add it to Favorites.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# fill overall background (dominant light warm gray from screenshot)
bg_color = (246, 244, 246)  # very light warm gray
draw.rectangle([(0, 0), canvas.size], fill=bg_color)

# STATUS BAR (top ~56px) - darker translucent strip
status_bar_h = 56
status_color = (154, 156, 158)  # muted gray similar to status bar
draw.rectangle([(0, 0), (canvas.size[0], status_bar_h)], fill=status_color)

# subtle top gradient highlight under status bar for depth
for i in range(8):
    alpha = int(8 - i)
    y = status_bar_h + i
    shade = (246 - i, 244 - i, 246 - i)
    draw.line([(0, y), (canvas.size[0], y)], fill=shade)

# HEADER / SEARCH AREA background (large pale area for search field)
header_top = status_bar_h
header_bottom = 200
header_bg = (252, 251, 252)  # slightly different pale white
draw.rectangle([(0, header_top), (canvas.size[0], header_bottom)], fill=header_bg)

# thin divider under header/search area
divider_color = (220, 218, 222)
draw.line([(40, header_bottom), (canvas.size[0] - 40, header_bottom)], fill=divider_color, width=2)

# Section: Filter chips row background band (do not draw chips themselves)
filters_band_top = 360
filters_band_bottom = 460
filters_bg = (249, 251, 254)  # very pale bluish band where filter pills sit
draw.rectangle([(0, filters_band_top), (canvas.size[0], filters_band_bottom)], fill=filters_bg)

# subtle shadow line under filters band
draw.line([(40, filters_band_bottom), (canvas.size[0] - 40, filters_band_bottom)], fill=divider_color, width=1)

# Big "carousel" image area background (just the card behind the row of images)
carousel_top = 520
carousel_bottom = 820
carousel_margin = 48
carousel_bg = (255, 255, 255)  # white card behind carousel images
draw.rounded_rectangle([(carousel_margin, carousel_top), (canvas.size[0] - carousel_margin, carousel_bottom)],
                       radius=12, fill=carousel_bg, outline=None)

# thin separators around the carousel card
draw.line([(carousel_margin, carousel_bottom + 8), (canvas.size[0] - carousel_margin, carousel_bottom + 8)],
          fill=divider_color, width=1)

# Event list container (full-width subtle card background for list)
list_top = carousel_bottom + 28
list_margin_lr = 24
list_bg = (255, 255, 255)
draw.rectangle([(list_margin_lr, list_top), (canvas.size[0] - list_margin_lr, canvas.size[1] - 140)],
               fill=list_bg)

# add subtle inner separators to indicate list rows (do not draw any text/images)
row_height = 360
y = list_top + 24
sep_color = (235, 233, 238)
while y < canvas.size[1] - 200:
    # draw a rounded card background for each event row
    card_left = list_margin_lr + 24
    card_right = canvas.size[0] - list_margin_lr - 24
    card_top = y
    card_bottom = min(y + row_height - 20, canvas.size[1] - 200)
    draw.rounded_rectangle([(card_left, card_top), (card_right, card_bottom)], radius=18,
                           fill=(255, 255, 255), outline=None)
    # thin divider under each card
    draw.line([(card_left + 12, card_bottom + 10), (card_right - 12, card_bottom + 10)], fill=sep_color, width=1)
    y += row_height

# Promoted badge background placeholder (small rounded rectangle behind where badge text will paste)
# Position intentionally near where detected badge appears but only a neutral background
promo_x = 60
promo_y = 640
promo_w = 96
promo_h = 44
draw.rounded_rectangle([(promo_x, promo_y), (promo_x + promo_w, promo_y + promo_h)], radius=10,
                       fill=(245, 249, 247))

# Large image card background (for the prominent event image lower on the page)
big_img_top = 1760
big_img_left = 48
big_img_right = canvas.size[0] - 48
big_img_bottom = big_img_top + 420
draw.rounded_rectangle([(big_img_left, big_img_top), (big_img_right, big_img_bottom)],
                       radius=16, fill=(34, 34, 34))  # dark placeholder background behind event artwork

# add a light border around the big image card
draw.rounded_rectangle([(big_img_left, big_img_top), (big_img_right, big_img_bottom)],
                       radius=16, outline=(205, 203, 208), width=1)

# small floating circle buttons background (neutral discs where heart/share buttons will overlay)
# positions based on general layout but only neutral circular backgrounds (no icons)
circle1_center = (big_img_right - 92, big_img_top + 40)
circle2_center = (big_img_right - 44, big_img_top + 40)
for cx, cy in [circle1_center, circle2_center]:
    r = 34
    draw.ellipse([(cx - r, cy - r), (cx + r, cy + r)], fill=(255, 255, 255), outline=(220, 218, 222))

# bottom navigation bar background
nav_top = canvas.size[1] - 112
nav_bottom = canvas.size[1]
nav_bg = (255, 255, 255)
draw.rectangle([(0, nav_top), (canvas.size[0], nav_bottom)], fill=nav_bg)

# top divider for nav
draw.line([(0, nav_top), (canvas.size[0], nav_top)], fill=divider_color, width=1)

# active tab indicator (small orange dot above center icon area) - neutral simple accent
indicator_x = canvas.size[0] // 2
indicator_y = nav_top + 18
draw.ellipse([(indicator_x - 6, indicator_y - 6), (indicator_x + 6, indicator_y + 6)], fill=(237, 102, 54))

# subtle overall bottom shadow to lift nav
for i in range(6):
    y = nav_top - i - 1
    alpha = int(6 - i)
    shade = (240 - i*2, 239 - i*2, 241 - i*2)
    draw.line([(0, y), (canvas.size[0], y)], fill=shade)

# final subtle vertical padding lines to frame content edges
edge_line_color = (247, 246, 248)
draw.line([(48, header_bottom + 8), (48, canvas.size[1] - 140)], fill=edge_line_color)
draw.line([(canvas.size[0] - 48, header_bottom + 8), (canvas.size[0] - 48, canvas.size[1] - 140)],
          fill=edge_line_color)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_02_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-4/00_icon_Music.png
try:
    _c0 = get_crop(0, 187, 103)
    canvas.paste(_c0, (837, 410), _c0)
except Exception:
    pass
layout["Music"] = [837, 410, 1024, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_02_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-4/01_icon_Business.png
try:
    _c1 = get_crop(1, 241, 103)
    canvas.paste(_c1, (1036, 410), _c1)
except Exception:
    pass
layout["Business"] = [1036, 410, 1277, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_02_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-4/02_icon_Anytime.png
try:
    _c2 = get_crop(2, 400, 103)
    canvas.paste(_c2, (425, 410), _c2)
except Exception:
    pass
layout["Anytime"] = [425, 410, 825, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_02_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-4/03_icon_Filters.png
try:
    _c3 = get_crop(3, 359, 103)
    canvas.paste(_c3, (54, 410), _c3)
except Exception:
    pass
layout["Filters"] = [54, 410, 413, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_02_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-4/04_icon_Foo.png
try:
    _c4 = get_crop(4, 154, 110)
    canvas.paste(_c4, (1282, 406), _c4)
except Exception:
    pass
layout["Foo"] = [1282, 406, 1436, 516]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_02_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-4/05_icon_IMETHOD.png
try:
    _c5 = get_crop(5, 144, 144)
    canvas.paste(_c5, (1092, 2336), _c5)
except Exception:
    pass
layout["IMETHOD"] = [1092, 2336, 1236, 2480]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_02_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-4/06_icon_IMETHOD.png
try:
    _c6 = get_crop(6, 144, 144)
    canvas.paste(_c6, (1236, 2336), _c6)
except Exception:
    pass
layout["IMETHOD"] = [1236, 2336, 1380, 2480]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_02_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-4/07_icon_9.11.png
try:
    _c7 = get_crop(7, 129, 116)
    canvas.paste(_c7, (54, 114), _c7)
except Exception:
    pass
layout["9.11"] = [54, 114, 183, 230]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_02_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-4/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 57, 61)
    canvas.paste(_c8, (246, 1), _c8)
except Exception:
    pass
layout["icon_8"] = [246, 1, 303, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_02_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-4/09_icon_Search_forae.png
try:
    _c9 = get_crop(9, 63, 63)
    canvas.paste(_c9, (311, 1), _c9)
except Exception:
    pass
layout["Search_forae"] = [311, 1, 374, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_02_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-4/10_icon_9.11.png
try:
    _c10 = get_crop(10, 55, 62)
    canvas.paste(_c10, (182, 0), _c10)
except Exception:
    pass
layout["9.11"] = [182, 0, 237, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_02_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-4/11_icon_New_York.png
try:
    _c11 = get_crop(11, 434, 144)
    canvas.paste(_c11, (0, 259), _c11)
except Exception:
    pass
layout["New_York"] = [0, 259, 434, 403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_02_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-4/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 101, 60)
    canvas.paste(_c12, (1206, 0), _c12)
except Exception:
    pass
layout["icon_12"] = [1206, 0, 1307, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_02_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-4/13_icon_9.11.png
try:
    _c13 = get_crop(13, 60, 64)
    canvas.paste(_c13, (112, 0), _c13)
except Exception:
    pass
layout["9.11"] = [112, 0, 172, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_02_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-4/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 66, 59)
    canvas.paste(_c14, (1314, 0), _c14)
except Exception:
    pass
layout["icon_14"] = [1314, 0, 1380, 59]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_02_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-4/15_icon_Overflow_menu_button.png
try:
    _c15 = get_crop(15, 144, 144)
    canvas.paste(_c15, (1236, 1192), _c15)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 1192, 1380, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_02_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-4/16_icon_GLABING_BLOOM_SOUND_COLLECTIVE.png
try:
    _c16 = get_crop(16, 1344, 996)
    canvas.paste(_c16, (48, 1820), _c16)
except Exception:
    pass
layout["GLABING;_BLOOM_SOUND_COLL"] = [48, 1820, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_02_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-4/17_icon_Search_forae.png
try:
    _c17 = get_crop(17, 51, 62)
    canvas.paste(_c17, (383, 2), _c17)
except Exception:
    pass
layout["Search_forae"] = [383, 2, 434, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_02_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-4/18_icon_The_Snace_at_Irondale.png
try:
    _c18 = get_crop(18, 288, 156)
    canvas.paste(_c18, (288, 2804), _c18)
except Exception:
    pass
layout["The_Snace_at_Irondale"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_02_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-4/19_icon_Search_forae.png
try:
    _c19 = get_crop(19, 1344, 191)
    canvas.paste(_c19, (48, 72), _c19)
except Exception:
    pass
layout["Search_forae"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_02_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-4/20_icon_slO_2Lo.png
try:
    _c20 = get_crop(20, 288, 156)
    canvas.paste(_c20, (864, 2804), _c20)
except Exception:
    pass
layout["slO_2Lo"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_02_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-4/21_icon_Favorite_button.png
try:
    _c21 = get_crop(21, 144, 144)
    canvas.paste(_c21, (1092, 1192), _c21)
except Exception:
    pass
layout["Favorite_button"] = [1092, 1192, 1236, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_02_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-4/22_icon_Promoted.png
try:
    _c22 = get_crop(22, 244, 66)
    canvas.paste(_c22, (84, 1665), _c22)
except Exception:
    pass
layout["Promoted"] = [84, 1665, 328, 1731]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_02_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-4/23_icon_Tequila_Artistic_Transformation.png
try:
    _c23 = get_crop(23, 1344, 1096)
    canvas.paste(_c23, (48, 676), _c23)
except Exception:
    pass
layout["Tequila_&_Artistic_Transf"] = [48, 676, 1392, 1772]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_02_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-4/24_icon_slO_2Lo.png
try:
    _c24 = get_crop(24, 288, 156)
    canvas.paste(_c24, (1152, 2804), _c24)
except Exception:
    pass
layout["slO_2Lo"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_02_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-4/25_icon_6.30_PM_EDT.png
try:
    _c25 = get_crop(25, 288, 156)
    canvas.paste(_c25, (576, 2804), _c25)
except Exception:
    pass
layout["6.30_PM_EDT"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_02_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-4/26_icon_Anytime.png
try:
    _c26 = get_crop(26, 210, 292)
    canvas.paste(_c26, (477, 670), _c26)
except Exception:
    pass
layout["Anytime"] = [477, 670, 687, 962]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_02_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-4/27_icon_10_000_events.png
try:
    _c27 = get_crop(27, 213, 295)
    canvas.paste(_c27, (217, 669), _c27)
except Exception:
    pass
layout["10,000_events"] = [217, 669, 430, 964]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_02_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-4/28_text_9.11.png
try:
    _c28 = get_crop(28, 89, 43)
    canvas.paste(_c28, (20, 17), _c28)
except Exception:
    pass
layout["9.11"] = [20, 17, 109, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_02_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-4/29_text_10_000_events.png
try:
    _c29 = get_crop(29, 359, 103)
    canvas.paste(_c29, (54, 410), _c29)
except Exception:
    pass
layout["10,000_events"] = [54, 410, 413, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_02_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-4/30_text_3.20.24.png
try:
    _c30 = get_crop(30, 172, 40)
    canvas.paste(_c30, (649, 1819), _c30)
except Exception:
    pass
layout["3.20.24"] = [649, 1819, 821, 1859]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_02_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-4/31_text_Wed_Mar_20.png
try:
    _c31 = get_crop(31, 288, 156)
    canvas.paste(_c31, (0, 2804), _c31)
except Exception:
    pass
layout["Wed,_Mar_20"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_02_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-4/32_text_6.30_PM_EDT.png
try:
    _c32 = get_crop(32, 288, 156)
    canvas.paste(_c32, (288, 2804), _c32)
except Exception:
    pass
layout["6.30_PM_EDT"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_02_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-4/33_text_The_Snace_at_Irondale.png
try:
    _c33 = get_crop(33, 288, 156)
    canvas.paste(_c33, (288, 2804), _c33)
except Exception:
    pass
layout["The_Snace_at_Irondale"] = [288, 2804, 576, 2960]
