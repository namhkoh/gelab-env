# page_id: page_eventbrite_4fbf805fbd914a178f72f68b0bc03f81_04
# screenshot: 2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-6.png
# step_index: 4/10
# task: Open Eventbrite. Explore "Education" events. Apply filters for events happening tomorrow. From the list, select the third event and check out its description.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Background and structural UI elements for Eventbrite-like mobile page
# Uses provided variables: canvas (PIL Image), draw (ImageDraw), font_sm, font_md, font_lg, font_xl

W, H = canvas.size

# Colors
status_bar_color = "#9e9e9e"      # top status bar (grey)
page_bg = "#ffffff"               # page background (white)
divider = "#e6e6e6"               # thin separators
card_shadow = "#e9e9ea"           # card shadow
card_bg = "#ffffff"               # card fill
muted_bg = "#fbfcfd"              # slightly off-white for large background bands

# Fill overall background (canvas starts white, but ensure uniform)
draw.rectangle([0, 0, W, H], fill=page_bg)

# Status bar (approx 56px high)
status_h = 56
draw.rectangle([0, 0, W, status_h], fill=status_bar_color)

# Header area (search / title area)
header_top = status_h
header_h = 140
draw.rectangle([0, header_top, W, header_top + header_h], fill=page_bg)
# header bottom divider
draw.line([48, header_top + header_h, W - 48, header_top + header_h], fill=divider, width=2)

# Location / filters band (keeps a clean white background)
loc_top = header_top + header_h
loc_h = 80
draw.rectangle([0, loc_top, W, loc_top + loc_h], fill=page_bg)
draw.line([48, loc_top + loc_h, W - 48, loc_top + loc_h], fill=divider, width=1)

# Filters / chips area (leave chips themselves to be pasted; draw subtle background band)
filters_top = loc_top + loc_h
filters_h = 110
draw.rectangle([0, filters_top, W, filters_top + filters_h], fill=page_bg)
# divider under filters area
draw.line([48, filters_top + filters_h, W - 48, filters_top + filters_h], fill=divider, width=1)

# Content area background (slightly warmer white to subtly separate from header)
content_top = filters_top + filters_h + 12
draw.rectangle([0, content_top, W, H - 320], fill=page_bg)

# Card container settings
card_x = 48
card_w = W - 2 * card_x
corner_radius = 28

# First large listing card (container for first event)
card1_top = content_top + 60
card1_h = 1400  # tall to accommodate image + details; event image & content will be pasted on top
# shadow
draw.rounded_rectangle([card_x + 6, card1_top + 8, card_x + card_w + 6, card1_top + card1_h + 8],
                       radius=corner_radius + 2, fill=card_shadow)
# card background
draw.rounded_rectangle([card_x, card1_top, card_x + card_w, card1_top + card1_h],
                       radius=corner_radius, fill=card_bg)
# subtle inner top divider (separates potential header inside card)
draw.line([card_x + 24, card1_top + 180, card_x + card_w - 24, card1_top + 180], fill=divider, width=1)

# Second listing card (next event)
card2_top = card1_top + card1_h + 28
card2_h = 980
# shadow
draw.rounded_rectangle([card_x + 6, card2_top + 8, card_x + card_w + 6, card2_top + card2_h + 8],
                       radius=corner_radius + 2, fill=card_shadow)
# card background
draw.rounded_rectangle([card_x, card2_top, card_x + card_w, card2_top + card2_h],
                       radius=corner_radius, fill=card_bg)
# inner divider for promoted/tag area (do not draw the tag itself)
draw.line([card_x + 24, card2_top + 160, card_x + card_w - 24, card2_top + 160], fill=divider, width=1)

# Large horizontal separator between major sections (below cards)
sep_y = card2_top + card2_h + 18
draw.line([24, sep_y, W - 24, sep_y], fill=divider, width=2)

# Floating content band for promoted/featured area near middle (subtle background, do not draw text)
featured_top = card1_top + 420
featured_h = 140
draw.rectangle([card_x + 18, featured_top, card_x + card_w - 18, featured_top + featured_h], fill=muted_bg, outline=None)
# small top and bottom hairlines
draw.line([card_x + 18, featured_top, card_x + card_w - 18, featured_top], fill=divider, width=1)
draw.line([card_x + 18, featured_top + featured_h, card_x + card_w - 18, featured_top + featured_h], fill=divider, width=1)

# Bottom navigation bar area
nav_h = 156
nav_top = H - nav_h
draw.rectangle([0, nav_top, W, H], fill=page_bg)
# top border for nav
draw.line([0, nav_top, W, nav_top], fill=divider, width=2)
# subtle notch background behind center area (to hint the active icon area)
center_notch_w = 220
center_x = W // 2
draw.rectangle([center_x - center_notch_w//2, nav_top + 12, center_x + center_notch_w//2, nav_top + nav_h - 24],
               fill=muted_bg, outline=None)

# Small bottom plate line to ground the UI
draw.line([24, H - 2, W - 24, H - 2], fill=divider, width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_04_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-6/00_icon_Music.png
try:
    _c0 = get_crop(0, 187, 103)
    canvas.paste(_c0, (837, 410), _c0)
except Exception:
    pass
layout["Music"] = [837, 410, 1024, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_04_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-6/01_icon_Business.png
try:
    _c1 = get_crop(1, 241, 103)
    canvas.paste(_c1, (1036, 410), _c1)
except Exception:
    pass
layout["Business"] = [1036, 410, 1277, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_04_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-6/02_icon_Anytime.png
try:
    _c2 = get_crop(2, 400, 103)
    canvas.paste(_c2, (425, 410), _c2)
except Exception:
    pass
layout["Anytime"] = [425, 410, 825, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_04_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-6/03_icon_Filters.png
try:
    _c3 = get_crop(3, 359, 103)
    canvas.paste(_c3, (54, 410), _c3)
except Exception:
    pass
layout["Filters"] = [54, 410, 413, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_04_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-6/04_icon_Foo.png
try:
    _c4 = get_crop(4, 149, 110)
    canvas.paste(_c4, (1283, 406), _c4)
except Exception:
    pass
layout["Foo"] = [1283, 406, 1432, 516]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_04_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-6/05_icon_Favorite_button.png
try:
    _c5 = get_crop(5, 144, 144)
    canvas.paste(_c5, (1092, 1192), _c5)
except Exception:
    pass
layout["Favorite_button"] = [1092, 1192, 1236, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_04_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-6/06_icon_Favorite_button.png
try:
    _c6 = get_crop(6, 144, 144)
    canvas.paste(_c6, (1092, 2415), _c6)
except Exception:
    pass
layout["Favorite_button"] = [1092, 2415, 1236, 2559]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_04_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-6/07_icon_Overflow_menu_button.png
try:
    _c7 = get_crop(7, 144, 144)
    canvas.paste(_c7, (1236, 1192), _c7)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 1192, 1380, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_04_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-6/08_icon_Overflow_menu_button.png
try:
    _c8 = get_crop(8, 144, 144)
    canvas.paste(_c8, (1236, 2415), _c8)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 2415, 1380, 2559]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_04_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-6/09_icon_Foo.png
try:
    _c9 = get_crop(9, 144, 144)
    canvas.paste(_c9, (1248, 96), _c9)
except Exception:
    pass
layout["Foo"] = [1248, 96, 1392, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_04_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-6/10_icon_Tom_Everhart_at_the_Grand_Opening_of_Cho.png
try:
    _c10 = get_crop(10, 1344, 1175)
    canvas.paste(_c10, (48, 676), _c10)
except Exception:
    pass
layout["Tom_Everhart_at_the_Grand"] = [48, 676, 1392, 1851]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_04_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-6/11_icon_4.56.png
try:
    _c11 = get_crop(11, 122, 113)
    canvas.paste(_c11, (55, 115), _c11)
except Exception:
    pass
layout["4.56"] = [55, 115, 177, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_04_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-6/12_icon_4.56.png
try:
    _c12 = get_crop(12, 61, 65)
    canvas.paste(_c12, (180, 0), _c12)
except Exception:
    pass
layout["4.56"] = [180, 0, 241, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_04_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-6/13_icon_Education.png
try:
    _c13 = get_crop(13, 68, 64)
    canvas.paste(_c13, (308, 0), _c13)
except Exception:
    pass
layout["Education"] = [308, 0, 376, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_04_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-6/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 54, 64)
    canvas.paste(_c14, (246, 0), _c14)
except Exception:
    pass
layout["icon_14"] = [246, 0, 300, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_04_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-6/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 105, 61)
    canvas.paste(_c15, (1204, 0), _c15)
except Exception:
    pass
layout["icon_15"] = [1204, 0, 1309, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_04_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-6/16_icon_Education.png
try:
    _c16 = get_crop(16, 1344, 191)
    canvas.paste(_c16, (48, 72), _c16)
except Exception:
    pass
layout["Education"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_04_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-6/17_icon_4.56.png
try:
    _c17 = get_crop(17, 60, 66)
    canvas.paste(_c17, (115, 0), _c17)
except Exception:
    pass
layout["4.56"] = [115, 0, 175, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_04_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-6/18_icon_icon_18.png
try:
    _c18 = get_crop(18, 64, 60)
    canvas.paste(_c18, (1317, 0), _c18)
except Exception:
    pass
layout["icon_18"] = [1317, 0, 1381, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_04_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-6/19_icon_Los_Angeles.png
try:
    _c19 = get_crop(19, 492, 144)
    canvas.paste(_c19, (0, 259), _c19)
except Exception:
    pass
layout["Los_Angeles"] = [0, 259, 492, 403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_04_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-6/20_icon_The_Vermont_Hollywood.png
try:
    _c20 = get_crop(20, 288, 156)
    canvas.paste(_c20, (288, 2804), _c20)
except Exception:
    pass
layout["The_Vermont_Hollywood"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_04_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-6/21_icon_Education.png
try:
    _c21 = get_crop(21, 50, 62)
    canvas.paste(_c21, (384, 2), _c21)
except Exception:
    pass
layout["Education"] = [384, 2, 434, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_04_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-6/22_icon_Afro_Ball_A_Celebration_of_Excellence.png
try:
    _c22 = get_crop(22, 288, 156)
    canvas.paste(_c22, (576, 2804), _c22)
except Exception:
    pass
layout["Afro_Ball:_A_Celebration_"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_04_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-6/23_icon_Afro_Ball_A_Celebration_of_Excellence.png
try:
    _c23 = get_crop(23, 288, 156)
    canvas.paste(_c23, (864, 2804), _c23)
except Exception:
    pass
layout["Afro_Ball:_A_Celebration_"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_04_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-6/24_icon_Afro_Ball_A_Celebration_of_Excellence.png
try:
    _c24 = get_crop(24, 1344, 917)
    canvas.paste(_c24, (48, 1899), _c24)
except Exception:
    pass
layout["Afro_Ball:_A_Celebration_"] = [48, 1899, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_04_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-6/25_icon_Promoted.png
try:
    _c25 = get_crop(25, 242, 67)
    canvas.paste(_c25, (85, 1744), _c25)
except Exception:
    pass
layout["Promoted"] = [85, 1744, 327, 1811]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_04_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-6/26_icon_Tom_Everhart_at_the_Grand_Opening_of_Cho.png
try:
    _c26 = get_crop(26, 1344, 1175)
    canvas.paste(_c26, (48, 676), _c26)
except Exception:
    pass
layout["Tom_Everhart_at_the_Grand"] = [48, 676, 1392, 1851]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_04_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-6/27_icon_More.png
try:
    _c27 = get_crop(27, 288, 156)
    canvas.paste(_c27, (1152, 2804), _c27)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_04_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-6/28_text_4.56.png
try:
    _c28 = get_crop(28, 89, 43)
    canvas.paste(_c28, (22, 17), _c28)
except Exception:
    pass
layout["4.56"] = [22, 17, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_04_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-6/29_text_2_837_events.png
try:
    _c29 = get_crop(29, 359, 103)
    canvas.paste(_c29, (54, 410), _c29)
except Exception:
    pass
layout["2,837_events"] = [54, 410, 413, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_04_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-6/30_text_313_N_Beverly_Dr.png
try:
    _c30 = get_crop(30, 323, 55)
    canvas.paste(_c30, (90, 1686), _c30)
except Exception:
    pass
layout["313_N_Beverly_Dr"] = [90, 1686, 413, 1741]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_04_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-6/31_clickable_Home.png
try:
    _c31 = get_crop(31, 288, 156)
    canvas.paste(_c31, (0, 2804), _c31)
except Exception:
    pass
layout["Home"] = [0, 2804, 288, 2960]
