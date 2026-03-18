# page_id: page_eventbrite_892383ad791c46b6aa65c632ee882bf0_12
# screenshot: 2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-14.png
# step_index: 12/12
# task: Open Eventbrite. Search for online "Music" events happening next weekend.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw background and structural UI elements for the Eventbrite-like mobile UI.
# Available objects: canvas (PIL.Image), draw (PIL.ImageDraw), font_sm, font_md, font_lg, font_xl

W, H = canvas.size

# Colors
status_bar_color = (200, 200, 200)      # light gray status bar
toolbar_bg = (255, 255, 255)           # white toolbar/header
divider_color = (226, 228, 230)        # very light gray divider lines
chip_area_bg = (250, 251, 253)         # slight off-white for chip/filter area
card_bg = (255, 255, 255)              # card background (white)
card_shadow = (241, 242, 244)          # subtle shadow behind cards
content_bg = (255, 255, 255)           # main background (white)
bottom_nav_bg = (255, 255, 255)        # bottom navigation bar background
muted_line = (238, 239, 240)

# Layout sizes (approximate, matching screenshot proportions)
status_bar_h = 96
toolbar_h = 116
filter_area_top = status_bar_h + toolbar_h  # starting y for filter area
filter_area_h = 200
filter_area_bottom = filter_area_top + filter_area_h

# Fill overall canvas with content background
draw.rectangle([(0, 0), (W, H)], fill=content_bg)

# Status bar (top strip where time/signal icons live)
draw.rectangle([(0, 0), (W, status_bar_h)], fill=status_bar_color)

# Toolbar / header background (search bar area; icons/text will be pasted on top)
toolbar_y0 = status_bar_h
toolbar_y1 = status_bar_h + toolbar_h
draw.rectangle([(0, toolbar_y0), (W, toolbar_y1)], fill=toolbar_bg)

# Thin divider under toolbar
draw.line([(48, toolbar_y1), (W-48, toolbar_y1)], fill=divider_color, width=2)

# Filter/chips area background (subtle)
draw.rectangle([(0, toolbar_y1), (W, filter_area_bottom)], fill=chip_area_bg)

# Subtle bottom border under chips row
draw.line([(0, filter_area_bottom), (W, filter_area_bottom)], fill=divider_color, width=1)

# "10,000 events" area spacing - leave text blank; draw a subtle horizontal padding guide
title_area_top = filter_area_bottom + 18
title_area_bottom = title_area_top + 64
# small divider under title
draw.line([(48, title_area_bottom), (W-48, title_area_bottom)], fill=muted_line, width=1)

# Card containers / image placeholders (ONLY draw backgrounds and shadows, not images)
# First event card: image at (48,676) size 1344x1012 per detections -> compute container rect slightly larger
img1_x = 48
img1_y = 676
img1_w = 1344
img1_h = 1012
card1_left = img1_x - 12
card1_top = img1_y - 28
card1_right = img1_x + img1_w + 12
card1_bottom = img1_y + img1_h + 80  # give room for title area below image

# Draw a soft shadow below the card (subtle rectangle offset)
shadow_offset = 10
draw.rounded_rectangle(
    [(card1_left+shadow_offset, card1_top+shadow_offset), (card1_right+shadow_offset, card1_bottom+shadow_offset)],
    radius=28, fill=card_shadow)

# Draw the card background (rounded)
draw.rounded_rectangle([(card1_left, card1_top), (card1_right, card1_bottom)], radius=28, fill=card_bg, outline=divider_color, width=1)

# Provide an inner rounded rect where the image will appear (so pasted image aligns above it) - keep same shape but do not draw content
image1_placeholder = (img1_x, img1_y, img1_x + img1_w, img1_y + img1_h)
# a subtle inner stroke to frame where the image will be pasted (very light)
draw.rounded_rectangle([ (image1_placeholder[0], image1_placeholder[1]), (image1_placeholder[2], image1_placeholder[3]) ], radius=20, outline=(245,245,246), width=1)

# Small divider / spacing between first card and next content (below title area)
after_card1_y = card1_bottom + 28
draw.line([(48, after_card1_y), (W-48, after_card1_y)], fill=muted_line, width=1)

# Second event card: image at (48,1736) size 1344x1080 per detections
img2_x = 48
img2_y = 1736
img2_w = 1344
img2_h = 1080
card2_left = img2_x - 12
card2_top = img2_y - 28
card2_right = img2_x + img2_w + 12
card2_bottom = img2_y + img2_h + 80

# Ensure we don't draw over the bottom-most navigation area: bottom nav top is approx 2804 (detections). Clip bottom if needed.
bottom_nav_top = 2804
if card2_bottom > bottom_nav_top - 12:
    card2_bottom = bottom_nav_top - 12

# Shadow for second card
draw.rounded_rectangle(
    [(card2_left+shadow_offset, card2_top+shadow_offset), (card2_right+shadow_offset, card2_bottom+shadow_offset)],
    radius=28, fill=card_shadow)

# Card bg for second card
draw.rounded_rectangle([(card2_left, card2_top), (card2_right, card2_bottom)], radius=28, fill=card_bg, outline=divider_color, width=1)

# Inner rounded rect marking image placement
draw.rounded_rectangle([(img2_x, img2_y), (img2_x + img2_w, img2_y + img2_h)], radius=20, outline=(245,245,246), width=1)

# Separators and small dividers for content areas
# Light divider above bottom navigation
draw.line([(0, bottom_nav_top), (W, bottom_nav_top)], fill=divider_color, width=2)

# Bottom navigation background
nav_h = H - bottom_nav_top
draw.rectangle([(0, bottom_nav_top), (W, H)], fill=bottom_nav_bg)

# Subtle shadow line at top of nav
draw.line([(0, bottom_nav_top), (W, bottom_nav_top)], fill=(230,231,233), width=1)

# Left and right safe margins lines for the list content (visual guide lines)
left_margin = 48
right_margin = W - 48
draw.line([(left_margin, 0), (left_margin, H)], fill=(255,255,255,0))
draw.line([(right_margin, 0), (right_margin, H)], fill=(255,255,255,0))

# Final small UI polish: faint horizontal rules separating list items (between card1 title area and next image)
# Place a couple of faint separators to match screenshot rhythm
sep1_y = card1_bottom + 44
draw.line([(left_margin, sep1_y), (right_margin, sep1_y)], fill=muted_line, width=1)

sep2_y = img2_y + img2_h + 44
if sep2_y < bottom_nav_top - 8:
    draw.line([(left_margin, sep2_y), (right_margin, sep2_y)], fill=muted_line, width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_12_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-14/00_icon_Music.png
try:
    _c0 = get_crop(0, 187, 103)
    canvas.paste(_c0, (1111, 410), _c0)
except Exception:
    pass
layout["Music"] = [1111, 410, 1298, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_12_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-14/01_icon_Apr_28_-_May_04_2024.png
try:
    _c1 = get_crop(1, 661, 103)
    canvas.paste(_c1, (438, 410), _c1)
except Exception:
    pass
layout["Apr_28_-_May_04,_2024"] = [438, 410, 1099, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_12_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-14/02_icon_1_Filter.png
try:
    _c2 = get_crop(2, 372, 103)
    canvas.paste(_c2, (54, 410), _c2)
except Exception:
    pass
layout["1_Filter"] = [54, 410, 426, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_12_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-14/03_icon_Favorite_button.png
try:
    _c3 = get_crop(3, 144, 144)
    canvas.paste(_c3, (1092, 1192), _c3)
except Exception:
    pass
layout["Favorite_button"] = [1092, 1192, 1236, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_12_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-14/04_icon_Favorite_button.png
try:
    _c4 = get_crop(4, 144, 144)
    canvas.paste(_c4, (1092, 2252), _c4)
except Exception:
    pass
layout["Favorite_button"] = [1092, 2252, 1236, 2396]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_12_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-14/05_icon_Overflow_menu_button.png
try:
    _c5 = get_crop(5, 144, 144)
    canvas.paste(_c5, (1236, 2252), _c5)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 2252, 1380, 2396]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_12_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-14/06_icon_Tune_in_and_Turn_it_UPI_Fun_music_trivia.png
try:
    _c6 = get_crop(6, 1344, 1080)
    canvas.paste(_c6, (48, 1736), _c6)
except Exception:
    pass
layout["Tune_in_and_Turn_it_UPI_F"] = [48, 1736, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_12_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-14/07_icon_Overflow_menu_button.png
try:
    _c7 = get_crop(7, 144, 144)
    canvas.paste(_c7, (1236, 1192), _c7)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 1192, 1380, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_12_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-14/08_icon_One_Night_in_Memphis_Tickets.png
try:
    _c8 = get_crop(8, 1344, 1012)
    canvas.paste(_c8, (48, 676), _c8)
except Exception:
    pass
layout["One_Night_in_Memphis_Tick"] = [48, 676, 1392, 1688]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_12_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-14/09_icon_Close_current_screen.png
try:
    _c9 = get_crop(9, 144, 144)
    canvas.paste(_c9, (1248, 96), _c9)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 96, 1392, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_12_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-14/10_icon_Music.png
try:
    _c10 = get_crop(10, 67, 63)
    canvas.paste(_c10, (308, 1), _c10)
except Exception:
    pass
layout["Music"] = [308, 1, 375, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_12_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-14/11_icon_celebrating_the_music_of_Taylor_Swift.png
try:
    _c11 = get_crop(11, 288, 156)
    canvas.paste(_c11, (576, 2804), _c11)
except Exception:
    pass
layout["celebrating_the_music_of_"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_12_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-14/12_icon_5.23.png
try:
    _c12 = get_crop(12, 58, 65)
    canvas.paste(_c12, (115, 0), _c12)
except Exception:
    pass
layout["5.23"] = [115, 0, 173, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_12_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-14/13_icon_5.23.png
try:
    _c13 = get_crop(13, 58, 64)
    canvas.paste(_c13, (182, 0), _c13)
except Exception:
    pass
layout["5.23"] = [182, 0, 240, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_12_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-14/14_icon_Bu.png
try:
    _c14 = get_crop(14, 101, 111)
    canvas.paste(_c14, (1306, 406), _c14)
except Exception:
    pass
layout["Bu"] = [1306, 406, 1407, 517]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_12_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-14/15_icon_Music.png
try:
    _c15 = get_crop(15, 51, 64)
    canvas.paste(_c15, (247, 1), _c15)
except Exception:
    pass
layout["Music"] = [247, 1, 298, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_12_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-14/16_icon_5.23.png
try:
    _c16 = get_crop(16, 126, 117)
    canvas.paste(_c16, (53, 112), _c16)
except Exception:
    pass
layout["5.23"] = [53, 112, 179, 229]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_12_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-14/17_icon_icon_17.png
try:
    _c17 = get_crop(17, 102, 62)
    canvas.paste(_c17, (1208, 0), _c17)
except Exception:
    pass
layout["icon_17"] = [1208, 0, 1310, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_12_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-14/18_icon_icon_18.png
try:
    _c18 = get_crop(18, 57, 63)
    canvas.paste(_c18, (1317, 0), _c18)
except Exception:
    pass
layout["icon_18"] = [1317, 0, 1374, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_12_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-14/19_icon_Music.png
try:
    _c19 = get_crop(19, 1344, 191)
    canvas.paste(_c19, (48, 72), _c19)
except Exception:
    pass
layout["Music"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_12_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-14/20_icon_icon_20.png
try:
    _c20 = get_crop(20, 49, 63)
    canvas.paste(_c20, (384, 2), _c20)
except Exception:
    pass
layout["icon_20"] = [384, 2, 433, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_12_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-14/21_icon_5.23.png
try:
    _c21 = get_crop(21, 94, 65)
    canvas.paste(_c21, (12, 0), _c21)
except Exception:
    pass
layout["5.23"] = [12, 0, 106, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_12_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-14/22_icon_5.30_PM_EDT.png
try:
    _c22 = get_crop(22, 288, 156)
    canvas.paste(_c22, (288, 2804), _c22)
except Exception:
    pass
layout["5.30_PM_EDT"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_12_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-14/23_icon_Online.png
try:
    _c23 = get_crop(23, 377, 144)
    canvas.paste(_c23, (0, 259), _c23)
except Exception:
    pass
layout["Online"] = [0, 259, 377, 403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_12_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-14/24_icon_celebrating_the_music_of_Taylor_Swift.png
try:
    _c24 = get_crop(24, 288, 156)
    canvas.paste(_c24, (864, 2804), _c24)
except Exception:
    pass
layout["celebrating_the_music_of_"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_12_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-14/25_text_10_000_events.png
try:
    _c25 = get_crop(25, 372, 103)
    canvas.paste(_c25, (54, 410), _c25)
except Exception:
    pass
layout["10,000_events"] = [54, 410, 426, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_12_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-14/26_text_Online.png
try:
    _c26 = get_crop(26, 131, 50)
    canvas.paste(_c26, (90, 2745), _c26)
except Exception:
    pass
layout["Online"] = [90, 2745, 221, 2795]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_12_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-14/27_clickable_Home.png
try:
    _c27 = get_crop(27, 288, 156)
    canvas.paste(_c27, (0, 2804), _c27)
except Exception:
    pass
layout["Home"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_12_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-14/28_clickable_More.png
try:
    _c28 = get_crop(28, 288, 156)
    canvas.paste(_c28, (1152, 2804), _c28)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]
