# page_id: page_eventbrite_d7ac75f457a4487c904e7baa93180729_07
# screenshot: 2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-9.png
# step_index: 7/11
# task: Open Eventbrite. Search for 'Cooking' classes. Filter to only show free events that occur in the weekend. Select the first event and proceed to checkout.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Canvas and draw are provided. This script paints the background, status bar,
# header dividers, section card backgrounds (rounded cards), separators, and bottom nav background.

# Colors
bg_color = (250, 251, 252)        # very light page background
status_bar_color = (200, 200, 200) # light grey status bar
header_bg = (255, 255, 255)       # white header area
divider_color = (226, 231, 235)   # subtle divider lines
card_shadow = (235, 239, 243)     # shadow color for cards
card_bg = (255, 255, 255)         # card background (white)
card_outline = (235, 238, 241)    # slight outline for cards
bottom_border = (215, 219, 223)   # top border of bottom nav
bottom_bg = (255, 255, 255)       # bottom nav background

W, H = canvas.size

# Fill overall background
draw.rectangle([0, 0, W, H], fill=bg_color)

# Status bar (top ~96px)
status_h = 96
draw.rectangle([0, 0, W, status_h], fill=status_bar_color)

# Header / search area below status bar (~96-176)
header_y0 = status_h
header_y1 = status_h + 80
draw.rectangle([0, header_y0, W, header_y1], fill=header_bg)
# header bottom divider
draw.line([(48, header_y1), (W-48, header_y1)], fill=divider_color, width=2)

# Location/filter strip area (below header)
loc_y0 = header_y1 + 8
loc_y1 = loc_y0 + 160
draw.rectangle([0, loc_y0, W, loc_y1], fill=header_bg)
# subtle divider under location/filters
draw.line([(48, loc_y1), (W-48, loc_y1)], fill=divider_color, width=2)

# "Results" divider line (separates controls from content)
results_div_y = loc_y1 + 80
draw.line([(48, results_div_y), (W-48, results_div_y)], fill=divider_color, width=1)

# Card backgrounds for the two large event tiles
cards = [
    # first large event card (top event image area)
    (48, 676, 48 + 1344, 676 + 1096),
    # second event card (lower)
    (48, 1820, 48 + 1344, 1820 + 996),
]

for (x0, y0, x1, y1) in cards:
    w = x1 - x0
    h = y1 - y0
    radius = 24

    # shadow (offset down-right slightly)
    shadow_offset = 8
    draw.rounded_rectangle(
        [x0 + shadow_offset, y0 + shadow_offset, x1 + shadow_offset, y1 + shadow_offset],
        radius=radius,
        fill=card_shadow
    )

    # main card background
    draw.rounded_rectangle([x0, y0, x1, y1], radius=radius, fill=card_bg, outline=card_outline, width=1)

    # subtle inner top divider to separate image area from metadata area (if any)
    inner_div_y = y0 + int(h * 0.62)
    draw.line([(x0 + 16, inner_div_y), (x1 - 16, inner_div_y)], fill=(245, 247, 249), width=1)

# Additional subtle separators between content sections
sep_positions = [
    results_div_y + 220,  # a gentle separator a bit below results header
    1640,                 # between first card region and following content
]
for y in sep_positions:
    draw.line([(48, y), (W-48, y)], fill=divider_color, width=1)

# Bottom navigation bar background and top border
nav_h = 120
nav_y0 = H - nav_h
draw.rectangle([0, nav_y0, W, H], fill=bottom_bg)
draw.line([(0, nav_y0), (W, nav_y0)], fill=bottom_border, width=2)

# Small top-of-page thin hairline under status bar for separation
draw.line([(0, status_h), (W, status_h)], fill=divider_color, width=1)

# End of background/structure painting

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_07_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-9/00_icon_Music.png
try:
    _c0 = get_crop(0, 187, 103)
    canvas.paste(_c0, (850, 410), _c0)
except Exception:
    pass
layout["Music"] = [850, 410, 1037, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_07_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-9/01_icon_Anytime.png
try:
    _c1 = get_crop(1, 400, 103)
    canvas.paste(_c1, (438, 410), _c1)
except Exception:
    pass
layout["Anytime"] = [438, 410, 838, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_07_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-9/02_icon_Business.png
try:
    _c2 = get_crop(2, 241, 103)
    canvas.paste(_c2, (1049, 410), _c2)
except Exception:
    pass
layout["Business"] = [1049, 410, 1290, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_07_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-9/03_icon_1_Filter.png
try:
    _c3 = get_crop(3, 372, 103)
    canvas.paste(_c3, (54, 410), _c3)
except Exception:
    pass
layout["1_Filter"] = [54, 410, 426, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_07_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-9/04_icon_Favorite_button.png
try:
    _c4 = get_crop(4, 144, 144)
    canvas.paste(_c4, (1092, 1192), _c4)
except Exception:
    pass
layout["Favorite_button"] = [1092, 1192, 1236, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_07_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-9/05_icon_Favorite_button.png
try:
    _c5 = get_crop(5, 144, 144)
    canvas.paste(_c5, (1092, 2336), _c5)
except Exception:
    pass
layout["Favorite_button"] = [1092, 2336, 1236, 2480]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_07_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-9/06_icon_Fo.png
try:
    _c6 = get_crop(6, 140, 111)
    canvas.paste(_c6, (1294, 406), _c6)
except Exception:
    pass
layout["Fo("] = [1294, 406, 1434, 517]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_07_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-9/07_icon_Overflow_menu_button.png
try:
    _c7 = get_crop(7, 144, 144)
    canvas.paste(_c7, (1236, 1192), _c7)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 1192, 1380, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_07_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-9/08_icon_Overflow_menu_button.png
try:
    _c8 = get_crop(8, 144, 144)
    canvas.paste(_c8, (1236, 2336), _c8)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 2336, 1380, 2480]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_07_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-9/09_icon_Cooking.png
try:
    _c9 = get_crop(9, 1344, 191)
    canvas.paste(_c9, (48, 72), _c9)
except Exception:
    pass
layout["Cooking"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_07_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-9/10_icon_4.39.png
try:
    _c10 = get_crop(10, 119, 110)
    canvas.paste(_c10, (57, 116), _c10)
except Exception:
    pass
layout["4.39"] = [57, 116, 176, 226]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_07_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-9/11_icon_Close_current_screen.png
try:
    _c11 = get_crop(11, 144, 144)
    canvas.paste(_c11, (1248, 96), _c11)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 96, 1392, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_07_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-9/12_icon_Cooking.png
try:
    _c12 = get_crop(12, 65, 62)
    canvas.paste(_c12, (308, 1), _c12)
except Exception:
    pass
layout["Cooking"] = [308, 1, 373, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_07_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-9/13_icon_4.39.png
try:
    _c13 = get_crop(13, 58, 62)
    canvas.paste(_c13, (181, 1), _c13)
except Exception:
    pass
layout["4.39"] = [181, 1, 239, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_07_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-9/14_icon_4.39.png
try:
    _c14 = get_crop(14, 57, 64)
    canvas.paste(_c14, (115, 0), _c14)
except Exception:
    pass
layout["4.39"] = [115, 0, 172, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_07_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-9/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 49, 61)
    canvas.paste(_c15, (250, 1), _c15)
except Exception:
    pass
layout["icon_15"] = [250, 1, 299, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_07_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-9/16_icon_icon_16.png
try:
    _c16 = get_crop(16, 101, 61)
    canvas.paste(_c16, (1209, 0), _c16)
except Exception:
    pass
layout["icon_16"] = [1209, 0, 1310, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_07_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-9/17_icon_icon_17.png
try:
    _c17 = get_crop(17, 58, 62)
    canvas.paste(_c17, (1317, 0), _c17)
except Exception:
    pass
layout["icon_17"] = [1317, 0, 1375, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_07_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-9/18_icon_Thu_Mav_16_._4_00_PM_PDT.png
try:
    _c18 = get_crop(18, 288, 156)
    canvas.paste(_c18, (288, 2804), _c18)
except Exception:
    pass
layout["Thu,_Mav_16_._4:00_PM_PDT"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_07_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-9/19_icon_Sizzling_Summer_Flavors_Master_the_Poten.png
try:
    _c19 = get_crop(19, 1344, 996)
    canvas.paste(_c19, (48, 1820), _c19)
except Exception:
    pass
layout["Sizzling_Summer_Flavors:_"] = [48, 1820, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_07_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-9/20_icon_Separating_the_Al_hype_from_the_real_val.png
try:
    _c20 = get_crop(20, 1344, 1096)
    canvas.paste(_c20, (48, 676), _c20)
except Exception:
    pass
layout["Separating_the_Al_hype_fr"] = [48, 676, 1392, 1772]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_07_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-9/21_icon_icon_21.png
try:
    _c21 = get_crop(21, 48, 61)
    canvas.paste(_c21, (384, 2), _c21)
except Exception:
    pass
layout["icon_21"] = [384, 2, 432, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_07_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-9/22_icon_San_Francisco.png
try:
    _c22 = get_crop(22, 536, 144)
    canvas.paste(_c22, (0, 259), _c22)
except Exception:
    pass
layout["San_Francisco"] = [0, 259, 536, 403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_07_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-9/23_icon_Sizzling_Summer_Flavors_Master_the_Poten.png
try:
    _c23 = get_crop(23, 288, 156)
    canvas.paste(_c23, (864, 2804), _c23)
except Exception:
    pass
layout["Sizzling_Summer_Flavors:_"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_07_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-9/24_icon_4.39.png
try:
    _c24 = get_crop(24, 145, 63)
    canvas.paste(_c24, (11, 0), _c24)
except Exception:
    pass
layout["4.39"] = [11, 0, 156, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_07_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-9/25_icon_of_Induction.png
try:
    _c25 = get_crop(25, 288, 156)
    canvas.paste(_c25, (0, 2804), _c25)
except Exception:
    pass
layout["of_Induction"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_07_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-9/26_icon_More.png
try:
    _c26 = get_crop(26, 288, 156)
    canvas.paste(_c26, (1152, 2804), _c26)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_07_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-9/27_icon_Cooking.png
try:
    _c27 = get_crop(27, 288, 156)
    canvas.paste(_c27, (576, 2804), _c27)
except Exception:
    pass
layout["Cooking"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_07_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-9/28_icon_Free.png
try:
    _c28 = get_crop(28, 128, 76)
    canvas.paste(_c28, (91, 2514), _c28)
except Exception:
    pass
layout["Free"] = [91, 2514, 219, 2590]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_07_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-9/29_text_37_events.png
try:
    _c29 = get_crop(29, 372, 103)
    canvas.paste(_c29, (54, 410), _c29)
except Exception:
    pass
layout["37_events"] = [54, 410, 426, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_07_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-9/30_text_Paaer.png
try:
    _c30 = get_crop(30, 55, 16)
    canvas.paste(_c30, (84, 741), _c30)
except Exception:
    pass
layout["Paaer"] = [84, 741, 139, 757]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_07_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-9/31_text_Sizzling_Summer.png
try:
    _c31 = get_crop(31, 435, 81)
    canvas.paste(_c31, (102, 1878), _c31)
except Exception:
    pass
layout["Sizzling_Summer"] = [102, 1878, 537, 1959]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_07_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-9/32_text_Flavors_Master_the.png
try:
    _c32 = get_crop(32, 498, 56)
    canvas.paste(_c32, (109, 1959), _c32)
except Exception:
    pass
layout["Flavors:_Master_the"] = [109, 1959, 607, 2015]
