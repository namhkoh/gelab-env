# page_id: page_eventbrite_a9f633a394e74f78843aa30bd2792346_09
# screenshot: 2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-11.png
# step_index: 9/18
# task: Open Eventbrite. Set the city to "Los Angeles". Look for Photography workshops happening next week. What is the price of the tickets for first non-promoted event?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw background and structural UI elements for the mobile Eventbrite page.
# Assumes variables provided: canvas (PIL.Image), draw (PIL.ImageDraw), font_sm, font_md, font_lg, font_xl

# Canvas size
W, H = canvas.size  # expected 1440x2960

# Colors (selected to match screenshot tones)
bg_color = (255, 255, 255)            # main page background (white)
status_bar_color = (189, 189, 189)    # top status bar grey
search_bg = (241, 247, 251)           # pale blue search background
divider_color = (226, 229, 234)       # subtle divider lines
card_shadow = (231, 234, 238)         # card shadow
card_bg = (255, 255, 255)             # card white
image_placeholder = (28, 28, 30)      # dark image placeholder/background
muted_panel = (248, 250, 252)         # light panel color
bottom_nav_border = (220, 223, 227)   # top border for bottom nav

# Fill overall background (canvas already white, but explicit to be safe)
draw.rectangle((0, 0, W, H), fill=bg_color)

# Status bar area (approx ~64px tall)
status_h = 64
draw.rectangle((0, 0, W, status_h), fill=status_bar_color)

# Header area (below status bar) with subtle divider
header_top = status_h
header_bottom = 160
draw.rectangle((0, header_top, W, header_bottom), fill=bg_color)
# divider under header
draw.line((40, header_bottom, W-40, header_bottom), fill=divider_color, width=2)

# Search bar / search row background (rounded pill)
search_left = 40
search_right = W - 40
search_top = header_top + 22
search_bottom = header_top + 86
try:
    draw.rounded_rectangle((search_left, search_top, search_right, search_bottom),
                           radius=30, fill=search_bg, outline=None)
except Exception:
    # fallback if rounded_rectangle not available
    draw.rectangle((search_left, search_top, search_right, search_bottom), fill=search_bg)

# Filter chips area separator (thin line below chips)
chips_line_y = header_bottom + 120
draw.line((40, chips_line_y, W-40, chips_line_y), fill=divider_color, width=1)

# Main content area padding
content_left = 48
content_right = W - 48

# First event card (rounded white card with subtle drop shadow)
card1_top = header_bottom + 60
card1_bottom = card1_top + 540
card_radius = 28

# shadow
try:
    draw.rounded_rectangle((content_left, card1_top + 8, content_right, card1_bottom + 8),
                           radius=card_radius, fill=card_shadow)
except Exception:
    draw.rectangle((content_left, card1_top + 8, content_right, card1_bottom + 8), fill=card_shadow)

# card background
try:
    draw.rounded_rectangle((content_left, card1_top, content_right, card1_bottom),
                           radius=card_radius, fill=card_bg, outline=divider_color, width=1)
except Exception:
    draw.rectangle((content_left, card1_top, content_right, card1_bottom), fill=card_bg)

# Image area placeholder inside card (rounded)
img_margin = 24
img_top = card1_top + img_margin
img_bottom = img_top + 320
try:
    draw.rounded_rectangle((content_left + img_margin, img_top, content_right - img_margin, img_bottom),
                           radius=16, fill=image_placeholder)
except Exception:
    draw.rectangle((content_left + img_margin, img_top, content_right - img_margin, img_bottom),
                   fill=image_placeholder)

# Separator line between image and metadata area
meta_sep_y = img_bottom + 20
draw.line((content_left + 20, meta_sep_y, content_right - 20, meta_sep_y),
          fill=(242,242,245), width=1)

# Second event card (below first, similar style)
card2_top = card1_bottom + 120
card2_bottom = card2_top + 540

# shadow
try:
    draw.rounded_rectangle((content_left, card2_top + 8, content_right, card2_bottom + 8),
                           radius=card_radius, fill=card_shadow)
except Exception:
    draw.rectangle((content_left, card2_top + 8, content_right, card2_bottom + 8), fill=card_shadow)

# card background
try:
    draw.rounded_rectangle((content_left, card2_top, content_right, card2_bottom),
                           radius=card_radius, fill=card_bg, outline=divider_color, width=1)
except Exception:
    draw.rectangle((content_left, card2_top, content_right, card2_bottom), fill=card_bg)

# Image area placeholder for second card
img2_top = card2_top + img_margin
img2_bottom = img2_top + 320
try:
    draw.rounded_rectangle((content_left + img_margin, img2_top, content_right - img_margin, img2_bottom),
                           radius=16, fill=(20,20,20))
except Exception:
    draw.rectangle((content_left + img_margin, img2_top, content_right - img_margin, img2_bottom),
                   fill=(20,20,20))

# Thin divider line between the two cards area
draw.line((content_left, card1_bottom + 60, content_right, card1_bottom + 60), fill=divider_color, width=1)

# A muted small background strip under the filters area (to visually separate chips/filters)
muted_strip_top = header_bottom + 16
muted_strip_bottom = muted_strip_top + 48
draw.rectangle((40, muted_strip_top, W-40, muted_strip_bottom), fill=muted_panel)

# Bottom navigation area: white bar with top border (keeps icons pasted on top)
nav_h = 120
nav_top = H - nav_h
draw.rectangle((0, nav_top, W, H), fill=card_bg)
# top border for nav
draw.line((0, nav_top, W, nav_top), fill=bottom_nav_border, width=2)

# Small center notch-like divider above bottom nav (visual)
draw.line((120, nav_top - 12, W - 120, nav_top - 12), fill=(248,248,250), width=1)

# Light left margin guide lines (subtle) to structure content column (not visible prominently)
draw.line((content_left, header_bottom, content_left, H - nav_h), fill=(255,255,255,0))
draw.line((content_right, header_bottom, content_right, H - nav_h), fill=(255,255,255,0))

# End of structural drawing.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_09_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-11/00_icon_Music.png
try:
    _c0 = get_crop(0, 187, 103)
    canvas.paste(_c0, (837, 410), _c0)
except Exception:
    pass
layout["Music"] = [837, 410, 1024, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_09_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-11/01_icon_Business.png
try:
    _c1 = get_crop(1, 241, 103)
    canvas.paste(_c1, (1036, 410), _c1)
except Exception:
    pass
layout["Business"] = [1036, 410, 1277, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_09_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-11/02_icon_Anytime.png
try:
    _c2 = get_crop(2, 400, 103)
    canvas.paste(_c2, (425, 410), _c2)
except Exception:
    pass
layout["Anytime"] = [425, 410, 825, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_09_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-11/03_icon_Filters.png
try:
    _c3 = get_crop(3, 359, 103)
    canvas.paste(_c3, (54, 410), _c3)
except Exception:
    pass
layout["Filters"] = [54, 410, 413, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_09_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-11/04_icon_Favorite_button.png
try:
    _c4 = get_crop(4, 144, 144)
    canvas.paste(_c4, (1092, 2434), _c4)
except Exception:
    pass
layout["Favorite_button"] = [1092, 2434, 1236, 2578]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_09_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-11/05_icon_Comp_Cards.png
try:
    _c5 = get_crop(5, 144, 144)
    canvas.paste(_c5, (1092, 1192), _c5)
except Exception:
    pass
layout["Comp_Cards_|"] = [1092, 1192, 1236, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_09_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-11/06_icon_Foo.png
try:
    _c6 = get_crop(6, 149, 110)
    canvas.paste(_c6, (1282, 406), _c6)
except Exception:
    pass
layout["Foo"] = [1282, 406, 1431, 516]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_09_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-11/07_icon_Overflow_menu_button.png
try:
    _c7 = get_crop(7, 144, 144)
    canvas.paste(_c7, (1236, 2434), _c7)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 2434, 1380, 2578]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_09_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-11/08_icon_Overflow_menu_button.png
try:
    _c8 = get_crop(8, 144, 144)
    canvas.paste(_c8, (1236, 1192), _c8)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 1192, 1380, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_09_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-11/09_icon_Foo.png
try:
    _c9 = get_crop(9, 144, 144)
    canvas.paste(_c9, (1248, 96), _c9)
except Exception:
    pass
layout["Foo"] = [1248, 96, 1392, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_09_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-11/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 54, 64)
    canvas.paste(_c10, (1151, 1), _c10)
except Exception:
    pass
layout["icon_10"] = [1151, 1, 1205, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_09_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-11/11_icon_4.50.png
try:
    _c11 = get_crop(11, 117, 110)
    canvas.paste(_c11, (58, 116), _c11)
except Exception:
    pass
layout["4.50"] = [58, 116, 175, 226]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_09_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-11/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 101, 63)
    canvas.paste(_c12, (1211, 0), _c12)
except Exception:
    pass
layout["icon_12"] = [1211, 0, 1312, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_09_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-11/13_icon_Photography.png
try:
    _c13 = get_crop(13, 68, 63)
    canvas.paste(_c13, (308, 0), _c13)
except Exception:
    pass
layout["Photography"] = [308, 0, 376, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_09_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-11/14_icon_Photography.png
try:
    _c14 = get_crop(14, 1344, 191)
    canvas.paste(_c14, (48, 72), _c14)
except Exception:
    pass
layout["Photography"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_09_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-11/15_icon_4.50.png
try:
    _c15 = get_crop(15, 60, 63)
    canvas.paste(_c15, (181, 0), _c15)
except Exception:
    pass
layout["4.50"] = [181, 0, 241, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_09_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-11/16_icon_icon_16.png
try:
    _c16 = get_crop(16, 51, 62)
    canvas.paste(_c16, (249, 1), _c16)
except Exception:
    pass
layout["icon_16"] = [249, 1, 300, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_09_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-11/17_icon_4.50.png
try:
    _c17 = get_crop(17, 61, 65)
    canvas.paste(_c17, (114, 0), _c17)
except Exception:
    pass
layout["4.50"] = [114, 0, 175, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_09_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-11/18_icon_icon_18.png
try:
    _c18 = get_crop(18, 55, 61)
    canvas.paste(_c18, (1319, 0), _c18)
except Exception:
    pass
layout["icon_18"] = [1319, 0, 1374, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_09_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-11/19_icon_Digitals.png
try:
    _c19 = get_crop(19, 1344, 1194)
    canvas.paste(_c19, (48, 676), _c19)
except Exception:
    pass
layout["Digitals"] = [48, 676, 1392, 1870]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_09_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-11/20_icon_Los_Angeles.png
try:
    _c20 = get_crop(20, 492, 144)
    canvas.paste(_c20, (0, 259), _c20)
except Exception:
    pass
layout["Los_Angeles"] = [0, 259, 492, 403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_09_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-11/21_icon_Photography.png
try:
    _c21 = get_crop(21, 50, 60)
    canvas.paste(_c21, (384, 3), _c21)
except Exception:
    pass
layout["Photography"] = [384, 3, 434, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_09_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-11/22_icon_0..png
try:
    _c22 = get_crop(22, 288, 156)
    canvas.paste(_c22, (288, 2804), _c22)
except Exception:
    pass
layout["0."] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_09_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-11/23_icon_Learning_to_See_Light_with_Jim_Sullivan.png
try:
    _c23 = get_crop(23, 288, 156)
    canvas.paste(_c23, (864, 2804), _c23)
except Exception:
    pass
layout["Learning_to_See_Light_wit"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_09_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-11/24_icon_Promoted.png
try:
    _c24 = get_crop(24, 249, 65)
    canvas.paste(_c24, (83, 1763), _c24)
except Exception:
    pass
layout["Promoted"] = [83, 1763, 332, 1828]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_09_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-11/25_icon_Digitals.png
try:
    _c25 = get_crop(25, 1344, 1194)
    canvas.paste(_c25, (48, 676), _c25)
except Exception:
    pass
layout["Digitals"] = [48, 676, 1392, 1870]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_09_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-11/26_icon_Learning_to_See_Light_with_Jim_Sullivan.png
try:
    _c26 = get_crop(26, 288, 156)
    canvas.paste(_c26, (1152, 2804), _c26)
except Exception:
    pass
layout["Learning_to_See_Light_wit"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_09_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-11/27_icon_on_food_lifestyle_photography.png
try:
    _c27 = get_crop(27, 1344, 898)
    canvas.paste(_c27, (48, 1918), _c27)
except Exception:
    pass
layout["on_food_&_lifestyle_photo"] = [48, 1918, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_09_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-11/28_icon_2715_Main_St.png
try:
    _c28 = get_crop(28, 259, 65)
    canvas.paste(_c28, (91, 1696), _c28)
except Exception:
    pass
layout["2715_Main_St"] = [91, 1696, 350, 1761]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_09_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-11/29_icon_Learning_to_See_Light_with_Jim_Sullivan.png
try:
    _c29 = get_crop(29, 288, 156)
    canvas.paste(_c29, (576, 2804), _c29)
except Exception:
    pass
layout["Learning_to_See_Light_wit"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_09_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-11/30_icon_IAIA.png
try:
    _c30 = get_crop(30, 288, 156)
    canvas.paste(_c30, (0, 2804), _c30)
except Exception:
    pass
layout["IAIA"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_09_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-11/31_text_4.50.png
try:
    _c31 = get_crop(31, 89, 43)
    canvas.paste(_c31, (22, 17), _c31)
except Exception:
    pass
layout["4.50"] = [22, 17, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_09_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-11/32_text_470_events.png
try:
    _c32 = get_crop(32, 359, 103)
    canvas.paste(_c32, (54, 410), _c32)
except Exception:
    pass
layout["470_events"] = [54, 410, 413, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_09_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-11/33_text_Thu_Apr_25.png
try:
    _c33 = get_crop(33, 232, 55)
    canvas.paste(_c33, (93, 1637), _c33)
except Exception:
    pass
layout["Thu,_Apr_25"] = [93, 1637, 325, 1692]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_09_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-11/34_text_1.45_PM_PDT.png
try:
    _c34 = get_crop(34, 249, 48)
    canvas.paste(_c34, (347, 1637), _c34)
except Exception:
    pass
layout["1.45_PM_PDT"] = [347, 1637, 596, 1685]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_09_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-11/35_text_IAIA.png
try:
    _c35 = get_crop(35, 85, 21)
    canvas.paste(_c35, (96, 2787), _c35)
except Exception:
    pass
layout["IAIA"] = [96, 2787, 181, 2808]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_09_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-11/36_text_A.png
try:
    _c36 = get_crop(36, 45, 25)
    canvas.paste(_c36, (199, 2784), _c36)
except Exception:
    pass
layout["^A__"] = [199, 2784, 244, 2809]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_09_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-11/37_text_2q.png
try:
    _c37 = get_crop(37, 53, 25)
    canvas.paste(_c37, (291, 2784), _c37)
except Exception:
    pass
layout["2q"] = [291, 2784, 344, 2809]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_09_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-11/38_text_0..png
try:
    _c38 = get_crop(38, 90, 27)
    canvas.paste(_c38, (374, 2784), _c38)
except Exception:
    pass
layout["0."] = [374, 2784, 464, 2811]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_09_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-11/39_text_DiA_rnt.png
try:
    _c39 = get_crop(39, 145, 25)
    canvas.paste(_c39, (472, 2784), _c39)
except Exception:
    pass
layout["DiA_rnt"] = [472, 2784, 617, 2809]
