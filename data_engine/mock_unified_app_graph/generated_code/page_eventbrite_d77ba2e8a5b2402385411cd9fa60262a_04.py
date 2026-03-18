# page_id: page_eventbrite_d77ba2e8a5b2402385411cd9fa60262a_04
# screenshot: 2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-6.png
# step_index: 4/8
# task: Open Eventbrite. Search for "Music". Filter only free events. Choose the first event. When is the date and timing?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Background and structural layout for Eventbrite "Music" search page
# Assumes variables `canvas` (1440x2960 RGB Image) and `draw` (ImageDraw) exist.

w, h = 1440, 2960

# Colors
bg_color = (250, 250, 252)          # very light off-white background
status_bar_color = (160, 160, 160)  # muted gray status bar
header_bg = (255, 255, 255)         # white header
divider = (226, 228, 231)           # light divider lines
card_bg = (255, 255, 255)           # card background (white)
card_container = (245, 247, 250)    # subtle card container fill
image_placeholder = (36, 36, 40)    # dark image container (backdrop behind images)
bottom_nav_bg = (255, 255, 255)     # bottom nav background
subtle_shadow = (230, 232, 235)     # used for subtle shadows / separators

# Fill canvas background
draw.rectangle([(0, 0), (w, h)], fill=bg_color)

# Status bar (top area) - keep icons drawn on top by other layers
status_h = 88
draw.rectangle([(0, 0), (w, status_h)], fill=status_bar_color)

# Header / search area
header_top = status_h
header_bottom = header_top + 140
draw.rectangle([(0, header_top), (w, header_bottom)], fill=header_bg)

# Subtle bottom divider under header
draw.line([(48, header_bottom), (w-48, header_bottom)], fill=divider, width=2)

# Location row separator (below chips area)
# Chips are not drawn; we only add a faint divider where layout separates
chips_div_y = 398
draw.line([(48, chips_div_y), (w-48, chips_div_y)], fill=divider, width=1)

# Large content area / list background (leave white, but add subtle vertical padding)
content_left = 40
content_right = w - 40

# First event card container (rounded rectangle behind the image & details)
# Image crop will be pasted on top at (48,676) 1344x1175; container slightly larger for corners & shadow.
card1_top = 660
card1_bottom = 1860
card_radius = 28
draw.rounded_rectangle(
    [(content_left, card1_top), (content_right, card1_bottom)],
    radius=card_radius,
    fill=card_container,
    outline=None
)

# Add a darker inner rounded rect where the image sits to emulate image area backdrop
img1_x0, img1_y0 = 48, 676
img1_x1 = img1_x0 + 1344
img1_y1 = img1_y0 + 1175
img_radius = 20
draw.rounded_rectangle(
    [(img1_x0-4, img1_y0-4), (img1_x1+4, img1_y1+4)],
    radius=img_radius,
    fill=image_placeholder,
    outline=None
)

# Event details card (white surface under first image)
details1_top = img1_y1 + 24
details1_bottom = details1_top + 160
draw.rectangle(
    [(content_left + 8, details1_top), (content_right - 8, details1_bottom)],
    fill=card_bg,
    outline=None
)
# subtle divider under details
draw.line([(content_left + 8, details1_bottom), (content_right - 8, details1_bottom)], fill=subtle_shadow, width=1)

# Second event card container (rounded rectangle for next image)
# Second image crop at (48,1899) size 1344x917
img2_x0, img2_y0 = 48, 1899
img2_x1 = img2_x0 + 1344
img2_y1 = img2_y0 + 917
draw.rounded_rectangle(
    [(content_left, img2_y0 - 16), (content_right, img2_y1 + 24)],
    radius=24,
    fill=card_container,
    outline=None
)
# Dark backdrop for second image area
draw.rounded_rectangle(
    [(img2_x0-4, img2_y0-4), (img2_x1+4, img2_y1+4)],
    radius=18,
    fill=image_placeholder,
    outline=None
)

# Separator lines between major sections
sep_y = img2_y1 + 64
draw.line([(48, sep_y), (w-48, sep_y)], fill=divider, width=1)

# Bottom navigation bar background and top divider
bottom_nav_h = 156
bottom_nav_top = h - bottom_nav_h
draw.rectangle([(0, bottom_nav_top), (w, h)], fill=bottom_nav_bg)
# top divider for bottom nav
draw.line([(0, bottom_nav_top), (w, bottom_nav_top)], fill=divider, width=2)

# Soft shadows under major cards to lift them slightly
shadow_strip_h = 8
# shadow under first card container
draw.rectangle([(content_left, card1_bottom), (content_right, card1_bottom + shadow_strip_h)], fill=(240,240,242))
# shadow under second card container
draw.rectangle([(content_left, img2_y1 + 24), (content_right, img2_y1 + 24 + shadow_strip_h)], fill=(240,240,242))

# Left margin vertical rule near the location pin area (subtle visual guide)
draw.line([(48, header_bottom + 24), (48, header_bottom + 120)], fill=subtle_shadow, width=2)

# Small decorative horizontal rules to indicate grouping (not text)
group_y_positions = [208, 320, 560]
for gy in group_y_positions:
    draw.line([(48, gy), (w-48, gy)], fill=(245,245,247), width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_04_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-6/00_icon_Music.png
try:
    _c0 = get_crop(0, 187, 103)
    canvas.paste(_c0, (837, 410), _c0)
except Exception:
    pass
layout["Music"] = [837, 410, 1024, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_04_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-6/01_icon_Business.png
try:
    _c1 = get_crop(1, 241, 103)
    canvas.paste(_c1, (1036, 410), _c1)
except Exception:
    pass
layout["Business"] = [1036, 410, 1277, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_04_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-6/02_icon_Anytime.png
try:
    _c2 = get_crop(2, 400, 103)
    canvas.paste(_c2, (425, 410), _c2)
except Exception:
    pass
layout["Anytime"] = [425, 410, 825, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_04_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-6/03_icon_Filters.png
try:
    _c3 = get_crop(3, 359, 103)
    canvas.paste(_c3, (54, 410), _c3)
except Exception:
    pass
layout["Filters"] = [54, 410, 413, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_04_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-6/04_icon_Favorite_button.png
try:
    _c4 = get_crop(4, 144, 144)
    canvas.paste(_c4, (1092, 1192), _c4)
except Exception:
    pass
layout["Favorite_button"] = [1092, 1192, 1236, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_04_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-6/05_icon_Foo.png
try:
    _c5 = get_crop(5, 147, 110)
    canvas.paste(_c5, (1283, 406), _c5)
except Exception:
    pass
layout["Foo"] = [1283, 406, 1430, 516]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_04_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-6/06_icon_Overflow_menu_button.png
try:
    _c6 = get_crop(6, 144, 144)
    canvas.paste(_c6, (1236, 1192), _c6)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 1192, 1380, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_04_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-6/07_icon_Interactive_Live_Music_and_Jam_Session_a.png
try:
    _c7 = get_crop(7, 1344, 1175)
    canvas.paste(_c7, (48, 676), _c7)
except Exception:
    pass
layout["Interactive_Live_Music_an"] = [48, 676, 1392, 1851]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_04_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-6/08_icon_S0.png
try:
    _c8 = get_crop(8, 144, 144)
    canvas.paste(_c8, (1236, 2415), _c8)
except Exception:
    pass
layout["S0"] = [1236, 2415, 1380, 2559]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_04_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-6/09_icon_Foo.png
try:
    _c9 = get_crop(9, 144, 144)
    canvas.paste(_c9, (1248, 96), _c9)
except Exception:
    pass
layout["Foo"] = [1248, 96, 1392, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_04_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-6/10_icon_S0.png
try:
    _c10 = get_crop(10, 144, 144)
    canvas.paste(_c10, (1092, 2415), _c10)
except Exception:
    pass
layout["S0"] = [1092, 2415, 1236, 2559]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_04_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-6/11_icon_6.48.png
try:
    _c11 = get_crop(11, 127, 116)
    canvas.paste(_c11, (53, 112), _c11)
except Exception:
    pass
layout["6.48"] = [53, 112, 180, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_04_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-6/12_icon_6.48.png
try:
    _c12 = get_crop(12, 61, 65)
    canvas.paste(_c12, (180, 0), _c12)
except Exception:
    pass
layout["6.48"] = [180, 0, 241, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_04_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-6/13_icon_Music.png
try:
    _c13 = get_crop(13, 68, 65)
    canvas.paste(_c13, (307, 0), _c13)
except Exception:
    pass
layout["Music"] = [307, 0, 375, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_04_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-6/14_icon_Music.png
try:
    _c14 = get_crop(14, 54, 65)
    canvas.paste(_c14, (246, 0), _c14)
except Exception:
    pass
layout["Music"] = [246, 0, 300, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_04_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-6/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 105, 61)
    canvas.paste(_c15, (1205, 0), _c15)
except Exception:
    pass
layout["icon_15"] = [1205, 0, 1310, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_04_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-6/16_icon_6.48.png
try:
    _c16 = get_crop(16, 61, 66)
    canvas.paste(_c16, (114, 0), _c16)
except Exception:
    pass
layout["6.48"] = [114, 0, 175, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_04_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-6/17_icon_icon_17.png
try:
    _c17 = get_crop(17, 63, 60)
    canvas.paste(_c17, (1318, 0), _c17)
except Exception:
    pass
layout["icon_17"] = [1318, 0, 1381, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_04_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-6/18_icon_New_York.png
try:
    _c18 = get_crop(18, 434, 144)
    canvas.paste(_c18, (0, 259), _c18)
except Exception:
    pass
layout["New_York"] = [0, 259, 434, 403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_04_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-6/19_icon_12_00_PM_EDT.png
try:
    _c19 = get_crop(19, 288, 156)
    canvas.paste(_c19, (288, 2804), _c19)
except Exception:
    pass
layout["12:00_PM_EDT"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_04_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-6/20_icon_icon_20.png
try:
    _c20 = get_crop(20, 51, 62)
    canvas.paste(_c20, (384, 2), _c20)
except Exception:
    pass
layout["icon_20"] = [384, 2, 435, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_04_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-6/21_icon_Music.png
try:
    _c21 = get_crop(21, 1344, 191)
    canvas.paste(_c21, (48, 72), _c21)
except Exception:
    pass
layout["Music"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_04_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-6/22_icon_Promoted.png
try:
    _c22 = get_crop(22, 265, 68)
    canvas.paste(_c22, (64, 1742), _c22)
except Exception:
    pass
layout["Promoted"] = [64, 1742, 329, 1810]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_04_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-6/23_icon_Brooklyn.png
try:
    _c23 = get_crop(23, 1344, 1175)
    canvas.paste(_c23, (48, 676), _c23)
except Exception:
    pass
layout["Brooklyn"] = [48, 676, 1392, 1851]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_04_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-6/24_icon_12_00_PM_EDT.png
try:
    _c24 = get_crop(24, 288, 156)
    canvas.paste(_c24, (576, 2804), _c24)
except Exception:
    pass
layout["12:00_PM_EDT"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_04_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-6/25_icon_JAk.png
try:
    _c25 = get_crop(25, 1344, 917)
    canvas.paste(_c25, (48, 1899), _c25)
except Exception:
    pass
layout["JAk"] = [48, 1899, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_04_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-6/26_icon_pinkFROG_cafe.png
try:
    _c26 = get_crop(26, 284, 61)
    canvas.paste(_c26, (89, 1678), _c26)
except Exception:
    pass
layout["pinkFROG_cafe"] = [89, 1678, 373, 1739]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_04_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-6/27_icon_Sat.png
try:
    _c27 = get_crop(27, 288, 156)
    canvas.paste(_c27, (0, 2804), _c27)
except Exception:
    pass
layout["Sat,"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_04_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-6/28_icon_MARQUIS_DEMDNL.png
try:
    _c28 = get_crop(28, 288, 156)
    canvas.paste(_c28, (864, 2804), _c28)
except Exception:
    pass
layout["MARQUIS_DEMDNL:"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_04_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-6/29_text_6.48.png
try:
    _c29 = get_crop(29, 89, 43)
    canvas.paste(_c29, (22, 15), _c29)
except Exception:
    pass
layout["6.48"] = [22, 15, 111, 58]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_04_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-6/30_text_9_180_events.png
try:
    _c30 = get_crop(30, 359, 103)
    canvas.paste(_c30, (54, 410), _c30)
except Exception:
    pass
layout["9,180_events"] = [54, 410, 413, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_04_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-6/31_clickable_More.png
try:
    _c31 = get_crop(31, 288, 156)
    canvas.paste(_c31, (1152, 2804), _c31)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]
