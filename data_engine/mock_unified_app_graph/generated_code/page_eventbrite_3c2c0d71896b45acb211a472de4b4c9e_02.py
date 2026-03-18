# page_id: page_eventbrite_3c2c0d71896b45acb211a472de4b4c9e_02
# screenshot: 2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-4.png
# step_index: 2/15
# task: Open Eventbrite. Search free Health event in Los Angeles. Select the first one that is not promoted. Follow the organizer. Share to google keep notes.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw background and structural UI elements for Eventbrite-like mobile page
# Uses provided variables: canvas (PIL Image), draw (ImageDraw), font_sm, font_md, font_lg, font_xl

# Colors
bg_color = (248, 249, 250)        # overall page background (very light)
status_bar_color = (116, 119, 123) # dark gray status bar
search_bg = (245, 247, 249)       # search field background
chip_bg = (226, 241, 252)         # pale blue chip band (not drawing chips themselves)
divider = (224, 226, 229)         # subtle divider lines
card_bg = (255, 255, 255)         # white card background
image_placeholder = (28, 30, 33)  # dark image area placeholder
light_placeholder = (243, 244, 246) # light image/card placeholder
bottom_bar = (255, 255, 255)
accent_shadow = (236, 238, 240)

w, h = canvas.size

# Fill overall background
draw.rectangle([(0,0),(w,h)], fill=bg_color)

# Status bar (top)
status_h = 110
draw.rectangle([(0,0),(w,status_h)], fill=status_bar_color)

# Thin subtle top overlay to simulate slight gradient highlight (very faint)
draw.line([(0,status_h-1),(w,status_h-1)], fill=(190,192,195), width=1)

# Search/header area below status bar
search_top = status_h + 18
search_left = 48
search_right = w - 48
search_height = 120
search_box = (search_left, search_top, search_right, search_top + search_height)
# search background rounded rectangle
try:
    draw.rounded_rectangle(search_box, radius=28, fill=search_bg, outline=divider, width=1)
except Exception:
    draw.rectangle(search_box, fill=search_bg, outline=divider)

# Underline divider below search area
divider_y = search_top + search_height + 24
draw.line([(48, divider_y), (w-48, divider_y)], fill=divider, width=2)

# Filter chips row area (we draw a faint band behind chips, but not the chips themselves)
chips_band_top = divider_y + 18
chips_band_bottom = chips_band_top + 150
draw.rectangle([(0, chips_band_top), (w, chips_band_bottom)], fill=bg_color)
# draw a faint horizontal rule under chips
draw.line([(48, chips_band_bottom-2), (w-48, chips_band_bottom-2)], fill=divider, width=1)

# "10,000 events" heading area - keep background consistent, add subtle spacing line
heading_sep_y = chips_band_bottom + 40
draw.line([(48, heading_sep_y), (w-48, heading_sep_y)], fill=divider, width=1)

# Large horizontal image/gallery placeholder under the heading (bottle images area)
gallery_top = heading_sep_y + 20
gallery_left = 48
gallery_right = w - 48
gallery_height = 180
gallery_box = (gallery_left, gallery_top, gallery_right, gallery_top + gallery_height)
# light placeholder with rounded corners
try:
    draw.rounded_rectangle(gallery_box, radius=12, fill=light_placeholder, outline=divider, width=1)
except Exception:
    draw.rectangle(gallery_box, fill=light_placeholder, outline=divider)
# faint top and bottom separators to emulate divider lines adjacent to gallery
draw.line([(48, gallery_top-16), (w-48, gallery_top-16)], fill=divider, width=1)
draw.line([(48, gallery_top + gallery_height + 16), (w-48, gallery_top + gallery_height + 16)], fill=divider, width=1)

# Thin separator before first event card
first_card_top = gallery_top + gallery_height + 40

# First event card background (title + meta area). We draw the card panel background only.
card_margin_x = 48
card_width = w - card_margin_x*2
card1_top = first_card_top
card1_bottom = card1_top + 220
card1_box = (card_margin_x, card1_top, card_margin_x + card_width, card1_bottom)
# subtle white card with light border
try:
    draw.rounded_rectangle(card1_box, radius=10, fill=card_bg, outline=accent_shadow, width=1)
except Exception:
    draw.rectangle(card1_box, fill=card_bg, outline=accent_shadow)
# horizontal divider inside card to separate header text area from image tile area
draw.line([(card_margin_x+24, card1_top+140), (card_margin_x + card_width - 24, card1_top+140)], fill=divider, width=1)

# Large image card placeholder for the next event (colorful banner). We draw only the dark/image background area.
# Positioned lower on the page
banner_top = card1_bottom + 28
banner_left = 48
banner_right = w - 48
banner_height = 360
banner_box = (banner_left, banner_top, banner_right, banner_top + banner_height)
try:
    draw.rounded_rectangle(banner_box, radius=18, fill=image_placeholder, outline=divider, width=1)
except Exception:
    draw.rectangle(banner_box, fill=image_placeholder, outline=divider)

# overlay a subtle rounded white mask at bottom of banner to hint rounded radius effect (not content)
mask_height = 10
draw.rectangle([(banner_left, banner_top + banner_height - mask_height), (banner_right, banner_top + banner_height)], fill=(20,22,24))

# Small badge area above subsequent event title (don't draw text; just area background for badge)
badge_top = banner_top + banner_height + 18
badge_box = (card_margin_x+12, badge_top, card_margin_x+12+140, badge_top+44)
try:
    draw.rounded_rectangle(badge_box, radius=12, fill=(245,241,246), outline=divider, width=1)
except Exception:
    draw.rectangle(badge_box, fill=(245,241,246), outline=divider)

# Second event card title/meta background area below banner
card2_top = badge_top + 64
card2_box = (card_margin_x, card2_top, card_margin_x + card_width, card2_top + 180)
try:
    draw.rounded_rectangle(card2_box, radius=10, fill=card_bg, outline=accent_shadow, width=1)
except Exception:
    draw.rectangle(card2_box, fill=card_bg, outline=accent_shadow)

# Separator line between list items further down
sep_y = card2_top + 190
draw.line([(48, sep_y), (w-48, sep_y)], fill=divider, width=1)

# Large image grid placeholder further down (to represent event list continuing)
grid_top = sep_y + 20
grid_box = (48, grid_top, w-48, grid_top + 420)
try:
    draw.rounded_rectangle(grid_box, radius=12, fill=light_placeholder, outline=divider, width=1)
except Exception:
    draw.rectangle(grid_box, fill=light_placeholder, outline=divider)

# Bottom navigation bar area
bottom_h = 120
draw.rectangle([(0, h - bottom_h), (w, h)], fill=bottom_bar)
# top border for bottom bar
draw.line([(0, h - bottom_h), (w, h - bottom_h)], fill=divider, width=1)

# small indicator area centered above bottom bar for selection highlight (do not draw icons)
indicator_w = 120
indicator_h = 6
ind_x1 = (w - indicator_w)//2
ind_x2 = ind_x1 + indicator_w
ind_y = h - bottom_h - 14
draw.rounded_rectangle((ind_x1, ind_y, ind_x2, ind_y + indicator_h), radius=3, fill=(242, 101, 34))

# Left and right subtle page edge decorations (very light)
draw.line([(24, search_top+search_height+6), (24, h - bottom_h - 6)], fill=(250,250,251), width=1)
draw.line([(w-24, search_top+search_height+6), (w-24, h - bottom_h - 6)], fill=(250,250,251), width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_02_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-4/00_icon_Music.png
try:
    _c0 = get_crop(0, 187, 103)
    canvas.paste(_c0, (837, 410), _c0)
except Exception:
    pass
layout["Music"] = [837, 410, 1024, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_02_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-4/01_icon_Business.png
try:
    _c1 = get_crop(1, 241, 103)
    canvas.paste(_c1, (1036, 410), _c1)
except Exception:
    pass
layout["Business"] = [1036, 410, 1277, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_02_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-4/02_icon_Anytime.png
try:
    _c2 = get_crop(2, 400, 103)
    canvas.paste(_c2, (425, 410), _c2)
except Exception:
    pass
layout["Anytime"] = [425, 410, 825, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_02_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-4/03_icon_Filters.png
try:
    _c3 = get_crop(3, 359, 103)
    canvas.paste(_c3, (54, 410), _c3)
except Exception:
    pass
layout["Filters"] = [54, 410, 413, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_02_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-4/04_icon_Foo.png
try:
    _c4 = get_crop(4, 154, 110)
    canvas.paste(_c4, (1282, 406), _c4)
except Exception:
    pass
layout["Foo"] = [1282, 406, 1436, 516]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_02_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-4/05_icon_IMETHOD.png
try:
    _c5 = get_crop(5, 144, 144)
    canvas.paste(_c5, (1092, 2336), _c5)
except Exception:
    pass
layout["IMETHOD"] = [1092, 2336, 1236, 2480]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_02_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-4/06_icon_IMETHOD.png
try:
    _c6 = get_crop(6, 144, 144)
    canvas.paste(_c6, (1236, 2336), _c6)
except Exception:
    pass
layout["IMETHOD"] = [1236, 2336, 1380, 2480]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_02_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-4/07_icon_9.41.png
try:
    _c7 = get_crop(7, 128, 116)
    canvas.paste(_c7, (54, 114), _c7)
except Exception:
    pass
layout["9.41"] = [54, 114, 182, 230]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_02_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-4/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 56, 61)
    canvas.paste(_c8, (247, 1), _c8)
except Exception:
    pass
layout["icon_8"] = [247, 1, 303, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_02_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-4/09_icon_Search_forae.png
try:
    _c9 = get_crop(9, 63, 63)
    canvas.paste(_c9, (311, 1), _c9)
except Exception:
    pass
layout["Search_forae"] = [311, 1, 374, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_02_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-4/10_icon_New_York.png
try:
    _c10 = get_crop(10, 434, 144)
    canvas.paste(_c10, (0, 259), _c10)
except Exception:
    pass
layout["New_York"] = [0, 259, 434, 403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_02_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-4/11_icon_9.41.png
try:
    _c11 = get_crop(11, 56, 62)
    canvas.paste(_c11, (182, 0), _c11)
except Exception:
    pass
layout["9.41"] = [182, 0, 238, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_02_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-4/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 102, 60)
    canvas.paste(_c12, (1205, 0), _c12)
except Exception:
    pass
layout["icon_12"] = [1205, 0, 1307, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_02_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-4/13_icon_9.41.png
try:
    _c13 = get_crop(13, 61, 64)
    canvas.paste(_c13, (112, 0), _c13)
except Exception:
    pass
layout["9.41"] = [112, 0, 173, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_02_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-4/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 66, 59)
    canvas.paste(_c14, (1314, 0), _c14)
except Exception:
    pass
layout["icon_14"] = [1314, 0, 1380, 59]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_02_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-4/15_icon_Overflow_menu_button.png
try:
    _c15 = get_crop(15, 144, 144)
    canvas.paste(_c15, (1236, 1192), _c15)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 1192, 1380, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_02_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-4/16_icon_GLABING_BLOOM_SOUND_COLLECTIVE.png
try:
    _c16 = get_crop(16, 1344, 996)
    canvas.paste(_c16, (48, 1820), _c16)
except Exception:
    pass
layout["GLABING;_BLOOM_SOUND_COLL"] = [48, 1820, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_02_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-4/17_icon_Search_forae.png
try:
    _c17 = get_crop(17, 51, 62)
    canvas.paste(_c17, (383, 2), _c17)
except Exception:
    pass
layout["Search_forae"] = [383, 2, 434, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_02_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-4/18_icon_The_Snace_at_Irondale.png
try:
    _c18 = get_crop(18, 288, 156)
    canvas.paste(_c18, (288, 2804), _c18)
except Exception:
    pass
layout["The_Snace_at_Irondale"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_02_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-4/19_icon_Search_forae.png
try:
    _c19 = get_crop(19, 1344, 191)
    canvas.paste(_c19, (48, 72), _c19)
except Exception:
    pass
layout["Search_forae"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_02_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-4/20_icon_slO_2Lo.png
try:
    _c20 = get_crop(20, 288, 156)
    canvas.paste(_c20, (864, 2804), _c20)
except Exception:
    pass
layout["slO_2Lo"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_02_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-4/21_icon_Favorite_button.png
try:
    _c21 = get_crop(21, 144, 144)
    canvas.paste(_c21, (1092, 1192), _c21)
except Exception:
    pass
layout["Favorite_button"] = [1092, 1192, 1236, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_02_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-4/22_icon_Promoted.png
try:
    _c22 = get_crop(22, 244, 66)
    canvas.paste(_c22, (84, 1665), _c22)
except Exception:
    pass
layout["Promoted"] = [84, 1665, 328, 1731]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_02_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-4/23_icon_Tequila_Artistic_Transformation.png
try:
    _c23 = get_crop(23, 1344, 1096)
    canvas.paste(_c23, (48, 676), _c23)
except Exception:
    pass
layout["Tequila_&_Artistic_Transf"] = [48, 676, 1392, 1772]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_02_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-4/24_icon_slO_2Lo.png
try:
    _c24 = get_crop(24, 288, 156)
    canvas.paste(_c24, (1152, 2804), _c24)
except Exception:
    pass
layout["slO_2Lo"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_02_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-4/25_icon_6.30_PM_EDT.png
try:
    _c25 = get_crop(25, 288, 156)
    canvas.paste(_c25, (576, 2804), _c25)
except Exception:
    pass
layout["6.30_PM_EDT"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_02_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-4/26_icon_Anytime.png
try:
    _c26 = get_crop(26, 210, 292)
    canvas.paste(_c26, (477, 670), _c26)
except Exception:
    pass
layout["Anytime"] = [477, 670, 687, 962]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_02_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-4/27_icon_10_000_events.png
try:
    _c27 = get_crop(27, 213, 295)
    canvas.paste(_c27, (217, 669), _c27)
except Exception:
    pass
layout["10,000_events"] = [217, 669, 430, 964]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_02_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-4/28_icon_Wed_Mar_20.png
try:
    _c28 = get_crop(28, 288, 156)
    canvas.paste(_c28, (0, 2804), _c28)
except Exception:
    pass
layout["Wed,_Mar_20"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_02_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-4/29_text_9.41.png
try:
    _c29 = get_crop(29, 89, 43)
    canvas.paste(_c29, (20, 15), _c29)
except Exception:
    pass
layout["9.41"] = [20, 15, 109, 58]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_02_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-4/30_text_10_000_events.png
try:
    _c30 = get_crop(30, 359, 103)
    canvas.paste(_c30, (54, 410), _c30)
except Exception:
    pass
layout["10,000_events"] = [54, 410, 413, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_02_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-4/31_text_3.20.24.png
try:
    _c31 = get_crop(31, 172, 40)
    canvas.paste(_c31, (649, 1819), _c31)
except Exception:
    pass
layout["3.20.24"] = [649, 1819, 821, 1859]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_02_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-4/32_text_Wed_Mar_20.png
try:
    _c32 = get_crop(32, 288, 156)
    canvas.paste(_c32, (0, 2804), _c32)
except Exception:
    pass
layout["Wed,_Mar_20"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_02_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-4/33_text_6.30_PM_EDT.png
try:
    _c33 = get_crop(33, 288, 156)
    canvas.paste(_c33, (288, 2804), _c33)
except Exception:
    pass
layout["6.30_PM_EDT"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_02_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-4/34_text_The_Snace_at_Irondale.png
try:
    _c34 = get_crop(34, 288, 156)
    canvas.paste(_c34, (288, 2804), _c34)
except Exception:
    pass
layout["The_Snace_at_Irondale"] = [288, 2804, 576, 2960]
