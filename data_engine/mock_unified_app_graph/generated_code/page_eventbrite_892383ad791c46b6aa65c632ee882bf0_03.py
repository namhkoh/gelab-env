# page_id: page_eventbrite_892383ad791c46b6aa65c632ee882bf0_03
# screenshot: 2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-5.png
# step_index: 3/12
# task: Open Eventbrite. Search for online "Music" events happening next weekend.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw background and structural UI elements for the mobile UI page.
# Available variables: canvas (PIL Image 1440x2960), draw (ImageDraw), font_sm, font_md, font_lg, font_xl

W, H = canvas.size

# Colors
status_bar_color = (189, 189, 189)   # light gray for status bar
header_accent_blue = (45, 91, 227)   # strong blue accent (underline)
divider_light = (230, 230, 230)      # very light gray for separators
card_shadow = (236, 236, 236)        # subtle shadow color
card_border = (235, 235, 235)        # card border
bottom_nav_bg = (250, 250, 250)      # very light for bottom nav

# 1) Status bar (top ~56px)
status_h = 56
draw.rectangle([0, 0, W, status_h], fill=status_bar_color)

# 2) Header / toolbar area (below status bar)
header_top = status_h
header_height = 112  # header area height (approx)
header_bottom = header_top + header_height
# keep header background white (canvas is white) but draw subtle divider and accent
draw.line([(24, header_bottom - 6), (W - 24, header_bottom - 6)], fill=header_accent_blue, width=6)
# thin divider under header
draw.line([(24, header_bottom), (W - 24, header_bottom)], fill=divider_light, width=2)

# 3) Section separators and subtle dividers
# Divider under the "Popular" block (approx location)
popular_div_y = 360
draw.line([(24, popular_div_y), (W - 24, popular_div_y)], fill=divider_light, width=2)

# Light divider between category list and events (approx)
events_div_y = 1020
draw.line([(24, events_div_y), (W - 24, events_div_y)], fill=divider_light, width=1)

# 4) Event card background rounded rectangles (behind detected event groups)
# Using the detected event bounding boxes positions (left=48, width=1344, height=396) stacked vertically
card_left = 48
card_width = 1344
card_height = 396
card_right = card_left + card_width

card_ys = [1117, 1513, 1909, 2305]  # y positions from detected elements
for y in card_ys:
    x0 = card_left
    y0 = y
    x1 = card_right
    y1 = y + card_height

    # Draw subtle shadow offset
    shadow_offset = 6
    draw.rounded_rectangle([x0 + shadow_offset, y0 + shadow_offset, x1 + shadow_offset, y1 + shadow_offset],
                           radius=12, fill=card_shadow, outline=None)

    # Draw white card background with light border
    draw.rounded_rectangle([x0, y0, x1, y1],
                           radius=12, fill=(255, 255, 255), outline=card_border, width=1)

    # subtle separator under each card (tiny gap and a hairline)
    draw.line([(x0 + 8, y1 + 12), (x1 - 8, y1 + 12)], fill=divider_light, width=1)

# 5) Content area backgrounds: a faint, wide band behind the main list (gives depth)
band_top = 1040
band_bottom = 2720
draw.rectangle([24, band_top, W - 24, band_bottom], fill=(255, 255, 255))  # keep white but frame with subtle border
draw.rectangle([24, band_top, W - 24, band_bottom], outline=divider_light, width=1)

# 6) Bottom navigation bar background and top divider
bottom_nav_top = 2804
draw.rectangle([0, bottom_nav_top, W, H], fill=bottom_nav_bg)
draw.line([(24, bottom_nav_top), (W - 24, bottom_nav_top)], fill=divider_light, width=1)

# 7) Additional subtle UI separators near the top search area
# thin line under the search area (reinforce header underline)
draw.line([(24, header_bottom + 8), (W - 24, header_bottom + 8)], fill=divider_light, width=1)

# (No text or icons are drawn - those are pasted separately)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_03_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-5/00_icon_MAY_2.png
try:
    _c0 = get_crop(0, 1344, 396)
    canvas.paste(_c0, (48, 2305), _c0)
except Exception:
    pass
layout["MAY_2"] = [48, 2305, 1392, 2701]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_03_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-5/01_icon_Isic_workshop_with.png
try:
    _c1 = get_crop(1, 1344, 396)
    canvas.paste(_c1, (48, 1909), _c1)
except Exception:
    pass
layout["Isic_workshop_with"] = [48, 1909, 1392, 2305]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_03_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-5/02_icon_8_846_creator_followers.png
try:
    _c2 = get_crop(2, 1344, 396)
    canvas.paste(_c2, (48, 1909), _c2)
except Exception:
    pass
layout["8_846_creator_followers"] = [48, 1909, 1392, 2305]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_03_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-5/03_icon_Online.png
try:
    _c3 = get_crop(3, 111, 49)
    canvas.paste(_c3, (390, 1751), _c3)
except Exception:
    pass
layout["Online"] = [390, 1751, 501, 1800]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_03_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-5/04_icon_ONLINE_Colour_the_Music.png
try:
    _c4 = get_crop(4, 1344, 396)
    canvas.paste(_c4, (48, 1513), _c4)
except Exception:
    pass
layout["ONLINE_Colour_the_Music"] = [48, 1513, 1392, 1909]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_03_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-5/05_icon_5.22.png
try:
    _c5 = get_crop(5, 131, 111)
    canvas.paste(_c5, (50, 114), _c5)
except Exception:
    pass
layout["5.22"] = [50, 114, 181, 225]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_03_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-5/06_icon_Music.png
try:
    _c6 = get_crop(6, 59, 62)
    canvas.paste(_c6, (311, 2), _c6)
except Exception:
    pass
layout["Music"] = [311, 2, 370, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_03_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-5/07_icon_Music.png
try:
    _c7 = get_crop(7, 1344, 191)
    canvas.paste(_c7, (48, 72), _c7)
except Exception:
    pass
layout["Music"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_03_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-5/08_icon_5.22.png
try:
    _c8 = get_crop(8, 56, 62)
    canvas.paste(_c8, (182, 1), _c8)
except Exception:
    pass
layout["5.22"] = [182, 1, 238, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_03_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-5/09_icon_5.22.png
try:
    _c9 = get_crop(9, 58, 64)
    canvas.paste(_c9, (115, 1), _c9)
except Exception:
    pass
layout["5.22"] = [115, 1, 173, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_03_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-5/10_icon_Online.png
try:
    _c10 = get_crop(10, 113, 51)
    canvas.paste(_c10, (390, 2147), _c10)
except Exception:
    pass
layout["Online"] = [390, 2147, 503, 2198]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_03_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-5/11_icon_9_00_AM_EDT.png
try:
    _c11 = get_crop(11, 1344, 396)
    canvas.paste(_c11, (48, 1117), _c11)
except Exception:
    pass
layout["9:00_AM_EDT"] = [48, 1117, 1392, 1513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_03_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-5/12_icon_Music.png
try:
    _c12 = get_crop(12, 46, 58)
    canvas.paste(_c12, (251, 4), _c12)
except Exception:
    pass
layout["Music"] = [251, 4, 297, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_03_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-5/13_icon_Wed_May_8_1_30_PM_EDT.png
try:
    _c13 = get_crop(13, 288, 156)
    canvas.paste(_c13, (288, 2804), _c13)
except Exception:
    pass
layout["Wed,_May_8_+_1:30_PM_EDT"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_03_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-5/14_icon_Wed_May_8_1_30_PM_EDT.png
try:
    _c14 = get_crop(14, 288, 156)
    canvas.paste(_c14, (576, 2804), _c14)
except Exception:
    pass
layout["Wed,_May_8_+_1:30_PM_EDT"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_03_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-5/15_icon_Online.png
try:
    _c15 = get_crop(15, 1344, 396)
    canvas.paste(_c15, (48, 1513), _c15)
except Exception:
    pass
layout["Online"] = [48, 1513, 1392, 1909]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_03_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-5/16_icon_Cancel.png
try:
    _c16 = get_crop(16, 97, 63)
    canvas.paste(_c16, (1214, 0), _c16)
except Exception:
    pass
layout["Cancel"] = [1214, 0, 1311, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_03_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-5/17_icon_Online.png
try:
    _c17 = get_crop(17, 112, 51)
    canvas.paste(_c17, (390, 2543), _c17)
except Exception:
    pass
layout["Online"] = [390, 2543, 502, 2594]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_03_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-5/18_icon_Online.png
try:
    _c18 = get_crop(18, 112, 49)
    canvas.paste(_c18, (391, 1354), _c18)
except Exception:
    pass
layout["Online"] = [391, 1354, 503, 1403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_03_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-5/19_icon_Cancel.png
try:
    _c19 = get_crop(19, 49, 61)
    canvas.paste(_c19, (1322, 1), _c19)
except Exception:
    pass
layout["Cancel"] = [1322, 1, 1371, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_03_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-5/20_icon_Tickets.png
try:
    _c20 = get_crop(20, 288, 156)
    canvas.paste(_c20, (864, 2804), _c20)
except Exception:
    pass
layout["Tickets"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_03_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-5/21_icon_Cancel.png
try:
    _c21 = get_crop(21, 144, 144)
    canvas.paste(_c21, (1099, 96), _c21)
except Exception:
    pass
layout["Cancel"] = [1099, 96, 1243, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_03_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-5/22_icon_5.22.png
try:
    _c22 = get_crop(22, 93, 62)
    canvas.paste(_c22, (16, 1), _c22)
except Exception:
    pass
layout["5.22"] = [16, 1, 109, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_03_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-5/23_icon_latin_music.png
try:
    _c23 = get_crop(23, 1344, 144)
    canvas.paste(_c23, (48, 858), _c23)
except Exception:
    pass
layout["latin_music"] = [48, 858, 1392, 1002]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_03_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-5/24_icon_live_music.png
try:
    _c24 = get_crop(24, 96, 94)
    canvas.paste(_c24, (31, 529), _c24)
except Exception:
    pass
layout["live_music"] = [31, 529, 127, 623]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_03_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-5/25_icon_8_20_creator_followers.png
try:
    _c25 = get_crop(25, 1344, 396)
    canvas.paste(_c25, (48, 2305), _c25)
except Exception:
    pass
layout["8_20_creator_followers"] = [48, 2305, 1392, 2701]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_03_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-5/26_icon_Thu.png
try:
    _c26 = get_crop(26, 1344, 396)
    canvas.paste(_c26, (48, 1117), _c26)
except Exception:
    pass
layout["Thu,"] = [48, 1117, 1392, 1513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_03_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-5/27_icon_Home.png
try:
    _c27 = get_crop(27, 288, 156)
    canvas.paste(_c27, (0, 2804), _c27)
except Exception:
    pass
layout["Home"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_03_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-5/28_icon_classical_music.png
try:
    _c28 = get_crop(28, 1344, 120)
    canvas.paste(_c28, (48, 738), _c28)
except Exception:
    pass
layout["classical_music"] = [48, 738, 1392, 858]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_03_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-5/29_icon_More.png
try:
    _c29 = get_crop(29, 288, 156)
    canvas.paste(_c29, (1152, 2804), _c29)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_03_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-5/30_icon_Cancel.png
try:
    _c30 = get_crop(30, 149, 144)
    canvas.paste(_c30, (1243, 97), _c30)
except Exception:
    pass
layout["Cancel"] = [1243, 97, 1392, 241]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_03_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-5/31_icon_latin_music.png
try:
    _c31 = get_crop(31, 90, 94)
    canvas.paste(_c31, (34, 768), _c31)
except Exception:
    pass
layout["latin_music"] = [34, 768, 124, 862]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_03_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-5/32_icon_Events.png
try:
    _c32 = get_crop(32, 87, 83)
    canvas.paste(_c32, (36, 893), _c32)
except Exception:
    pass
layout["Events"] = [36, 893, 123, 976]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_03_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-5/33_text_Popular.png
try:
    _c33 = get_crop(33, 221, 78)
    canvas.paste(_c33, (44, 298), _c33)
except Exception:
    pass
layout["Popular"] = [44, 298, 265, 376]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_03_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-5/34_text_live_music.png
try:
    _c34 = get_crop(34, 193, 48)
    canvas.paste(_c34, (162, 430), _c34)
except Exception:
    pass
layout["live_music"] = [162, 430, 355, 478]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_03_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-5/35_text_music_concerts.png
try:
    _c35 = get_crop(35, 1344, 120)
    canvas.paste(_c35, (48, 498), _c35)
except Exception:
    pass
layout["music_concerts"] = [48, 498, 1392, 618]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_03_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-5/36_text_techno_music.png
try:
    _c36 = get_crop(36, 256, 43)
    canvas.paste(_c36, (163, 672), _c36)
except Exception:
    pass
layout["techno_music"] = [163, 672, 419, 715]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_03_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-5/37_text_Events.png
try:
    _c37 = get_crop(37, 191, 61)
    canvas.paste(_c37, (45, 1026), _c37)
except Exception:
    pass
layout["Events"] = [45, 1026, 236, 1087]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_03_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-5/38_text_Wed_May_8_1_30_PM_EDT.png
try:
    _c38 = get_crop(38, 288, 156)
    canvas.paste(_c38, (576, 2804), _c38)
except Exception:
    pass
layout["Wed,_May_8_+_1:30_PM_EDT"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_03_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-5/39_clickable_live_music.png
try:
    _c39 = get_crop(39, 1344, 120)
    canvas.paste(_c39, (48, 378), _c39)
except Exception:
    pass
layout["live_music"] = [48, 378, 1392, 498]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_03_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-5/40_clickable_techno_music.png
try:
    _c40 = get_crop(40, 1344, 120)
    canvas.paste(_c40, (48, 618), _c40)
except Exception:
    pass
layout["techno_music"] = [48, 618, 1392, 738]
