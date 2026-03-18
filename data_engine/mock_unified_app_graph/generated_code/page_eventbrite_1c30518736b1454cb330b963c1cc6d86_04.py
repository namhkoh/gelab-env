# page_id: page_eventbrite_1c30518736b1454cb330b963c1cc6d86_04
# screenshot: 2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-6.png
# step_index: 4/9
# task: Open Eventbrite. Search for "Open Mic Nights". Filter the results to only include free events. Select the first non-promoted event in the list - what"s the location of that event?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw UI background and structure for Eventbrite-like page
# Available variables: canvas (PIL Image 1440x2960), draw (ImageDraw), font_sm, font_md, font_lg, font_xl

# Colors
status_bar_color = (200, 200, 200)      # light grey for status bar
page_bg = (255, 255, 255)               # white background
divider = (225, 225, 225)               # light divider lines
card_bg = (255, 255, 255)               # white cards
card_outline = (230, 230, 230)          # subtle card border
image_placeholder = (235, 230, 225)     # warm neutral for image placeholders
muted_section = (245, 246, 250)         # very light tint for some section backgrounds
nav_bar_bg = (255, 255, 255)            # bottom nav background
nav_divider = (240, 240, 240)

W, H = canvas.size

# Fill overall background (canvas initially white, but set explicitly)
draw.rectangle([(0,0),(W,H)], fill=page_bg)

# Status bar (top area)
status_h = 84
draw.rectangle([(0,0),(W,status_h)], fill=status_bar_color)

# Header / toolbar area below status bar
header_top = status_h
header_bottom = 260
draw.rectangle([(0, header_top), (W, header_bottom)], fill=page_bg)

# Thin divider under header
draw.line([(48, header_bottom+2), (W-48, header_bottom+2)], fill=divider, width=2)

# Location / filter strip background subtle band (behind chips)
filter_band_top = 352
filter_band_bottom = 470
draw.rectangle([(0, filter_band_top), (W, filter_band_bottom)], fill=page_bg)

# Divider below filter chips area
draw.line([(48, filter_band_bottom+6), (W-48, filter_band_bottom+6)], fill=divider, width=1)

# First event card container (rounded rect)
card1_x0, card1_y0 = 48, 596
card1_x1, card1_y1 = W-48, 1236
card_corner = 28
# subtle drop shadow (a faint gray rectangle behind)
shadow_offset = 8
draw.rectangle([(card1_x0+shadow_offset, card1_y0+shadow_offset),
                (card1_x1+shadow_offset, card1_y1+shadow_offset)], fill=(245,245,245))
# main card
draw.rounded_rectangle([(card1_x0, card1_y0), (card1_x1, card1_y1)],
                       radius=card_corner, fill=card_bg, outline=card_outline, width=1)

# Image area placeholder inside card (inset so actual pasted image overlays cleanly)
img_pad = 24
img_x0 = card1_x0 + img_pad
img_y0 = card1_y0 + img_pad
img_x1 = card1_x1 - img_pad
img_y1 = img_y0 + 360  # approximate image height region
draw.rounded_rectangle([(img_x0, img_y0), (img_x1, img_y1)], radius=16,
                       fill=image_placeholder, outline=(220,220,220))

# Badge/background strip under image (for small badges like "Ticket sales end soon") - very light tint
badge_h = 40
badge_y = img_y1 + 12
draw.rounded_rectangle([(img_x0, badge_y), (img_x0+220, badge_y+badge_h)], radius=12,
                       fill=(241,236,246), outline=None)

# Divider between image and textual content inside card
text_div_y = img_y1 + 80
draw.line([(img_x0, text_div_y), (img_x1, text_div_y)], fill=(250,250,250), width=1)

# Second event card container further down
card2_x0, card2_y0 = 48, 1500
card2_x1, card2_y1 = W-48, 2170
# shadow
draw.rectangle([(card2_x0+shadow_offset, card2_y0+shadow_offset),
                (card2_x1+shadow_offset, card2_y1+shadow_offset)], fill=(245,245,245))
draw.rounded_rectangle([(card2_x0, card2_y0), (card2_x1, card2_y1)],
                       radius=24, fill=card_bg, outline=card_outline, width=1)

# Right-side image thumbnail placeholder inside second card (mimicking layout)
thumb_w = 480
thumb_pad = 28
thumb_x1 = card2_x1 - thumb_pad
thumb_x0 = thumb_x1 - thumb_w
thumb_y0 = card2_y0 + thumb_pad
thumb_y1 = thumb_y0 + 360
draw.rounded_rectangle([(thumb_x0, thumb_y0), (thumb_x1, thumb_y1)], radius=16,
                       fill=image_placeholder, outline=(220,220,220))

# Left-side textual area background (light muted band to separate)
left_area_x0 = card2_x0 + thumb_pad
left_area_x1 = thumb_x0 - 20
left_area_y0 = thumb_y0
left_area_y1 = thumb_y1
draw.rectangle([(left_area_x0, left_area_y0), (left_area_x1, left_area_y1)], fill=card_bg)

# Small "Free" pill background (do not draw text) - placed under left area
pill_w, pill_h = 72, 36
pill_x = left_area_x0
pill_y = left_area_y1 + 18
draw.rounded_rectangle([(pill_x, pill_y), (pill_x+pill_w, pill_y+pill_h)], radius=10, fill=(225,236,233))

# Horizontal separator between list items
sep_y = card2_y1 + 26
draw.line([(48, sep_y), (W-48, sep_y)], fill=divider, width=1)

# Large content area banner further down (muted background behind rows of small cards)
banner_top = 2300
banner_bottom = 2680
draw.rectangle([(48, banner_top), (W-48, banner_bottom)], fill=muted_section)
draw.line([(48, banner_top), (W-48, banner_top)], fill=divider, width=1)

# Bottom navigation bar background and divider
nav_h = 100
nav_y0 = H - nav_h
draw.line([(0, nav_y0), (W, nav_y0)], fill=nav_divider, width=1)
draw.rectangle([(0, nav_y0), (W, H)], fill=nav_bar_bg)

# subtle top shadow for nav
draw.rectangle([(0, nav_y0-4), (W, nav_y0)], fill=(250,250,250))

# Small center pill indicator area on nav (just a soft horizontal line to suggest active area)
indicator_w = 120
indicator_h = 6
ind_x0 = (W - indicator_w)//2
ind_x1 = ind_x0 + indicator_w
ind_y = nav_y0 + 18
draw.rounded_rectangle([(ind_x0, ind_y), (ind_x1, ind_y + indicator_h)], radius=3, fill=(255,140,30))

# Final subtle vertical padding lines to frame content edges
edge_x = 48
draw.line([(edge_x, header_bottom+8), (edge_x, H - nav_h - 8)], fill=(250,250,250), width=1)
draw.line([(W-edge_x, header_bottom+8), (W-edge_x, H - nav_h - 8)], fill=(250,250,250), width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_04_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-6/00_icon_Music.png
try:
    _c0 = get_crop(0, 187, 103)
    canvas.paste(_c0, (837, 410), _c0)
except Exception:
    pass
layout["Music"] = [837, 410, 1024, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_04_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-6/01_icon_Business.png
try:
    _c1 = get_crop(1, 241, 103)
    canvas.paste(_c1, (1036, 410), _c1)
except Exception:
    pass
layout["Business"] = [1036, 410, 1277, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_04_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-6/02_icon_Anytime.png
try:
    _c2 = get_crop(2, 400, 103)
    canvas.paste(_c2, (425, 410), _c2)
except Exception:
    pass
layout["Anytime"] = [425, 410, 825, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_04_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-6/03_icon_Filters.png
try:
    _c3 = get_crop(3, 359, 103)
    canvas.paste(_c3, (54, 410), _c3)
except Exception:
    pass
layout["Filters"] = [54, 410, 413, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_04_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-6/04_icon_Favorite_button.png
try:
    _c4 = get_crop(4, 144, 144)
    canvas.paste(_c4, (1092, 2434), _c4)
except Exception:
    pass
layout["Favorite_button"] = [1092, 2434, 1236, 2578]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_04_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-6/05_icon_Favorite_button.png
try:
    _c5 = get_crop(5, 144, 144)
    canvas.paste(_c5, (1092, 1192), _c5)
except Exception:
    pass
layout["Favorite_button"] = [1092, 1192, 1236, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_04_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-6/06_icon_Foo.png
try:
    _c6 = get_crop(6, 149, 110)
    canvas.paste(_c6, (1283, 406), _c6)
except Exception:
    pass
layout["Foo"] = [1283, 406, 1432, 516]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_04_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-6/07_icon_Overflow_menu_button.png
try:
    _c7 = get_crop(7, 144, 144)
    canvas.paste(_c7, (1236, 1192), _c7)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 1192, 1380, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_04_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-6/08_icon_Overflow_menu_button.png
try:
    _c8 = get_crop(8, 144, 144)
    canvas.paste(_c8, (1236, 2434), _c8)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 2434, 1380, 2578]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_04_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-6/09_icon_4.53.png
try:
    _c9 = get_crop(9, 123, 112)
    canvas.paste(_c9, (56, 115), _c9)
except Exception:
    pass
layout["4.53"] = [56, 115, 179, 227]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_04_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-6/10_icon_Foo.png
try:
    _c10 = get_crop(10, 144, 144)
    canvas.paste(_c10, (1248, 96), _c10)
except Exception:
    pass
layout["Foo"] = [1248, 96, 1392, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_04_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-6/11_icon_Open_Mic_Night.png
try:
    _c11 = get_crop(11, 1344, 191)
    canvas.paste(_c11, (48, 72), _c11)
except Exception:
    pass
layout["Open_Mic_Night"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_04_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-6/12_icon_LLS_Orchestra_at_the_Million_Dollar_Thea.png
try:
    _c12 = get_crop(12, 1344, 1194)
    canvas.paste(_c12, (48, 676), _c12)
except Exception:
    pass
layout["LLS_Orchestra_at_the_Mill"] = [48, 676, 1392, 1870]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_04_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-6/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 103, 61)
    canvas.paste(_c13, (1207, 0), _c13)
except Exception:
    pass
layout["icon_13"] = [1207, 0, 1310, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_04_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-6/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 66, 61)
    canvas.paste(_c14, (308, 1), _c14)
except Exception:
    pass
layout["icon_14"] = [308, 1, 374, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_04_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-6/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 51, 62)
    canvas.paste(_c15, (247, 1), _c15)
except Exception:
    pass
layout["icon_15"] = [247, 1, 298, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_04_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-6/16_icon_4.53.png
try:
    _c16 = get_crop(16, 58, 63)
    canvas.paste(_c16, (182, 0), _c16)
except Exception:
    pass
layout["4.53"] = [182, 0, 240, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_04_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-6/17_icon_4.53.png
try:
    _c17 = get_crop(17, 57, 64)
    canvas.paste(_c17, (116, 0), _c17)
except Exception:
    pass
layout["4.53"] = [116, 0, 173, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_04_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-6/18_icon_LLS_Orchestra_at_the_Million_Dollar_Thea.png
try:
    _c18 = get_crop(18, 1344, 1194)
    canvas.paste(_c18, (48, 676), _c18)
except Exception:
    pass
layout["LLS_Orchestra_at_the_Mill"] = [48, 676, 1392, 1870]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_04_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-6/19_icon_icon_19.png
try:
    _c19 = get_crop(19, 60, 61)
    canvas.paste(_c19, (1318, 0), _c19)
except Exception:
    pass
layout["icon_19"] = [1318, 0, 1378, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_04_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-6/20_icon_Los_Angeles.png
try:
    _c20 = get_crop(20, 492, 144)
    canvas.paste(_c20, (0, 259), _c20)
except Exception:
    pass
layout["Los_Angeles"] = [0, 259, 492, 403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_04_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-6/21_icon_icon_21.png
try:
    _c21 = get_crop(21, 48, 60)
    canvas.paste(_c21, (384, 3), _c21)
except Exception:
    pass
layout["icon_21"] = [384, 3, 432, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_04_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-6/22_icon_D.png
try:
    _c22 = get_crop(22, 288, 156)
    canvas.paste(_c22, (288, 2804), _c22)
except Exception:
    pass
layout["D_"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_04_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-6/23_icon_6pm.png
try:
    _c23 = get_crop(23, 1344, 898)
    canvas.paste(_c23, (48, 1918), _c23)
except Exception:
    pass
layout["6pm"] = [48, 1918, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_04_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-6/24_icon_Million_Dollar_Theater.png
try:
    _c24 = get_crop(24, 44, 59)
    canvas.paste(_c24, (283, 1766), _c24)
except Exception:
    pass
layout["Million_Dollar_Theater"] = [283, 1766, 327, 1825]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_04_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-6/25_icon_Tom_Everhart_at_the_Grand_Opening_of_Cho.png
try:
    _c25 = get_crop(25, 288, 156)
    canvas.paste(_c25, (864, 2804), _c25)
except Exception:
    pass
layout["Tom_Everhart_at_the_Grand"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_04_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-6/26_icon_Tom_Everhart_at_the_Grand_Opening_of_Cho.png
try:
    _c26 = get_crop(26, 288, 156)
    canvas.paste(_c26, (1152, 2804), _c26)
except Exception:
    pass
layout["Tom_Everhart_at_the_Grand"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_04_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-6/27_icon_Ie.png
try:
    _c27 = get_crop(27, 288, 156)
    canvas.paste(_c27, (576, 2804), _c27)
except Exception:
    pass
layout["Ie"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_04_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-6/28_icon_LLS_Orchestra_at_the_Million_Dollar_Thea.png
try:
    _c28 = get_crop(28, 1344, 1194)
    canvas.paste(_c28, (48, 676), _c28)
except Exception:
    pass
layout["LLS_Orchestra_at_the_Mill"] = [48, 676, 1392, 1870]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_04_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-6/29_icon_4.53.png
try:
    _c29 = get_crop(29, 122, 63)
    canvas.paste(_c29, (9, 0), _c29)
except Exception:
    pass
layout["4.53"] = [9, 0, 131, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_04_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-6/30_icon_Free.png
try:
    _c30 = get_crop(30, 288, 156)
    canvas.paste(_c30, (0, 2804), _c30)
except Exception:
    pass
layout["Free"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_04_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-6/31_icon_Tickets.png
try:
    _c31 = get_crop(31, 288, 156)
    canvas.paste(_c31, (864, 2804), _c31)
except Exception:
    pass
layout["Tickets"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_04_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-6/32_text_2_681_events.png
try:
    _c32 = get_crop(32, 359, 103)
    canvas.paste(_c32, (54, 410), _c32)
except Exception:
    pass
layout["2,681_events"] = [54, 410, 413, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_04_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-6/33_text_Million_Dollar_Theater.png
try:
    _c33 = get_crop(33, 404, 55)
    canvas.paste(_c33, (93, 1704), _c33)
except Exception:
    pass
layout["Million_Dollar_Theater"] = [93, 1704, 497, 1759]
