# page_id: page_eventbrite_6b75132d6e874d9a960bba273e5f011b_01
# screenshot: 2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-3.png
# step_index: 1/11
# task: Open Eventbrite. Set the city to 'San Francisco'. Search 'Outdoor'. Select an event starting after 5 PM. Check the ticket price.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw background and structural UI elements for Eventbrite-like mobile page
# Uses provided canvas (1440x2960) and draw (ImageDraw)

# Colors
bg = "#ffffff"                 # page background (dominant white)
status_bar = "#cfcfcf"         # light gray status bar
header_bg = "#ffffff"          # header/toolbar background
card_bg = "#ffffff"            # card background (white)
card_shadow = "#e9e9ea"        # subtle shadow for cards
divider = "#efeff2"            # light divider line
bottom_bar = "#ffffff"         # bottom navigation background
accent_pulse = "#f5f3f8"       # very light purple tint for section backdrop

# Fill full background
draw.rectangle([(0, 0), (1440, 2960)], fill=bg)

# Status bar (top ~56px)
draw.rectangle([(0, 0), (1440, 56)], fill=status_bar)

# Header area below status bar (toolbar zone)
header_top = 56
header_bottom = 160
# Draw header background strips on left and right to avoid overlapping the central search widget area
# (search widget will be pasted later at x=195..1374, y=93..237)
search_box = (195, 93, 195 + 1179, 93 + 144)
# Left header slice
draw.rectangle([(0, header_top), (search_box[0], header_bottom)], fill=header_bg)
# Right header slice
draw.rectangle([(search_box[2], header_top), (1440, header_bottom)], fill=header_bg)
# Subtle bottom divider for header
draw.line([(0, header_bottom), (1440, header_bottom)], fill=divider, width=1)

# Section background (slight tint behind the large "More events you'll love" area)
section_y_top = header_bottom + 40
section_y_bottom = section_y_top + 220
draw.rectangle([(0, section_y_top), (1440, section_y_bottom)], fill=accent_pulse)

# Positions for event list card groups (as detected clusters). We'll draw rounded card backgrounds and light shadows.
card_positions = [
    (48, 490, 48 + 1344, 490 + 396),
    (48, 1282, 48 + 1344, 1282 + 396),
    (48, 1678, 48 + 1344, 1678 + 396),
    (48, 2074, 48 + 1344, 2074 + 396),
    (48, 2470, 48 + 1344, 2470 + 346)  # last one slightly shorter (as in detection)
]

for (x1, y1, x2, y2) in card_positions:
    # Shadow (slightly offset)
    shadow_box = (x1 + 6, y1 + 8, x2 + 6, y2 + 8)
    draw.rounded_rectangle(shadow_box, radius=14, fill=card_shadow)
    # Card background
    draw.rounded_rectangle((x1, y1, x2, y2), radius=12, fill=card_bg, outline=None)

    # Divider line under each card to separate from following content
    div_y = y2 + 20
    draw.line([(x1 + 12, div_y), (x2 - 12, div_y)], fill=divider, width=1)

# Additional thin separators between major content sections
separator_positions = [section_y_bottom + 20, 430, 1200, 1600, 2000, 2400]
for sy in separator_positions:
    draw.line([(48, sy), (1392, sy)], fill=divider, width=1)

# Bottom navigation bar background with top divider
bottom_h = 120
draw.rectangle([(0, 2960 - bottom_h), (1440, 2960)], fill=bottom_bar)
draw.line([(0, 2960 - bottom_h), (1440, 2960 - bottom_h)], fill=divider, width=1)

# Floating subtle card holder near the lower center (background only, actual location pill will be pasted on top)
# Place it slightly above bottom nav; ensure it does not draw text/icons
floating_box = (420, 2560, 1020, 2660)
draw.rounded_rectangle(floating_box, radius=36, fill=card_bg, outline=divider)

# Small left gutter vertical rule to define content column (visual structure only)
draw.line([(48, header_bottom + 10), (48, 2960 - bottom_h - 10)], fill=divider, width=1)

# Final subtle overall vignette lines between list items to improve separation (vertical spacing cues)
y = 600
while y < 2500:
    draw.line([(1200, y), (1392, y)], fill=divider, width=1)
    y += 360

# Note: No text, icons, or interactive elements are drawn. Detected UI elements (search, icons, texts, buttons)
# will be pasted on top at their exact positions.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_01_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-3/00_icon_ripg_-_LeaTG_Atans.png
try:
    _c0 = get_crop(0, 1344, 396)
    canvas.paste(_c0, (48, 2074), _c0)
except Exception:
    pass
layout["ripg_-_LeaTG_Atans"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_01_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-3/01_icon_EYPCG.png
try:
    _c1 = get_crop(1, 1344, 396)
    canvas.paste(_c1, (48, 1678), _c1)
except Exception:
    pass
layout["EYPCG"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_01_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-3/02_icon_Chicago.png
try:
    _c2 = get_crop(2, 388, 117)
    canvas.paste(_c2, (526, 2651), _c2)
except Exception:
    pass
layout["Chicago"] = [526, 2651, 914, 2768]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_01_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-3/03_icon_iokstore.png
try:
    _c3 = get_crop(3, 1344, 396)
    canvas.paste(_c3, (48, 1282), _c3)
except Exception:
    pass
layout["iokstore"] = [48, 1282, 1392, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_01_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-3/04_icon_Search_events.png
try:
    _c4 = get_crop(4, 1179, 144)
    canvas.paste(_c4, (195, 93), _c4)
except Exception:
    pass
layout["Search_events"] = [195, 93, 1374, 237]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_01_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-3/05_icon_Sat_Oct_19.png
try:
    _c5 = get_crop(5, 1344, 396)
    canvas.paste(_c5, (48, 490), _c5)
except Exception:
    pass
layout["Sat,_Oct_19"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_01_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-3/06_icon_Dovetail_Brewery.png
try:
    _c6 = get_crop(6, 144, 139)
    canvas.paste(_c6, (1140, 1935), _c6)
except Exception:
    pass
layout["Dovetail_Brewery"] = [1140, 1935, 1284, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_01_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-3/07_icon_Favorite_button.png
try:
    _c7 = get_crop(7, 144, 123)
    canvas.paste(_c7, (1140, 2347), _c7)
except Exception:
    pass
layout["Favorite_button"] = [1140, 2347, 1284, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_01_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-3/08_icon_Favorite_button.png
try:
    _c8 = get_crop(8, 144, 139)
    canvas.paste(_c8, (1140, 1539), _c8)
except Exception:
    pass
layout["Favorite_button"] = [1140, 1539, 1284, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_01_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-3/09_icon_7940_Wolcott_Ave_apt_2_Chicago_IL_USA.png
try:
    _c9 = get_crop(9, 1344, 396)
    canvas.paste(_c9, (48, 490), _c9)
except Exception:
    pass
layout["7940_$_Wolcott_Ave_apt_2,"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_01_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-3/10_icon_Overflow_menu_button.png
try:
    _c10 = get_crop(10, 144, 139)
    canvas.paste(_c10, (1284, 1935), _c10)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 1935, 1428, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_01_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-3/11_icon_Overflow_menu_button.png
try:
    _c11 = get_crop(11, 144, 123)
    canvas.paste(_c11, (1284, 2347), _c11)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 2347, 1428, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_01_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-3/12_icon_Favorite_button.png
try:
    _c12 = get_crop(12, 144, 125)
    canvas.paste(_c12, (1140, 761), _c12)
except Exception:
    pass
layout["Favorite_button"] = [1140, 761, 1284, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_01_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-3/13_icon_Favorite_button.png
try:
    _c13 = get_crop(13, 144, 139)
    canvas.paste(_c13, (1140, 1143), _c13)
except Exception:
    pass
layout["Favorite_button"] = [1140, 1143, 1284, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_01_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-3/14_icon_Napervili.png
try:
    _c14 = get_crop(14, 288, 156)
    canvas.paste(_c14, (288, 2804), _c14)
except Exception:
    pass
layout["Napervili"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_01_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-3/15_icon_Overflow_menu_button.png
try:
    _c15 = get_crop(15, 144, 125)
    canvas.paste(_c15, (1284, 761), _c15)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 761, 1428, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_01_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-3/16_icon_Overflow_menu_button.png
try:
    _c16 = get_crop(16, 144, 139)
    canvas.paste(_c16, (1284, 1539), _c16)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 1539, 1428, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_01_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-3/17_icon_I_00_PM_CDT.png
try:
    _c17 = get_crop(17, 1344, 396)
    canvas.paste(_c17, (48, 886), _c17)
except Exception:
    pass
layout["I:00_PM_CDT"] = [48, 886, 1392, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_01_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-3/18_icon_8.11.png
try:
    _c18 = get_crop(18, 54, 60)
    canvas.paste(_c18, (184, 2), _c18)
except Exception:
    pass
layout["8.11"] = [184, 2, 238, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_01_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-3/19_icon_8.11.png
try:
    _c19 = get_crop(19, 102, 99)
    canvas.paste(_c19, (41, 122), _c19)
except Exception:
    pass
layout["8.11"] = [41, 122, 143, 221]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_01_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-3/20_icon_mhroucuntc.png
try:
    _c20 = get_crop(20, 288, 156)
    canvas.paste(_c20, (0, 2804), _c20)
except Exception:
    pass
layout["mhroucuntc"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_01_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-3/21_icon_icon_21.png
try:
    _c21 = get_crop(21, 60, 58)
    canvas.paste(_c21, (312, 3), _c21)
except Exception:
    pass
layout["icon_21"] = [312, 3, 372, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_01_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-3/22_icon_Overflow_menu_button.png
try:
    _c22 = get_crop(22, 144, 139)
    canvas.paste(_c22, (1284, 1143), _c22)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 1143, 1428, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_01_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-3/23_icon_ON.png
try:
    _c23 = get_crop(23, 1344, 396)
    canvas.paste(_c23, (48, 886), _c23)
except Exception:
    pass
layout["ON"] = [48, 886, 1392, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_01_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-3/24_icon_Planting_Seeds_bilingual.png
try:
    _c24 = get_crop(24, 1344, 396)
    canvas.paste(_c24, (48, 2074), _c24)
except Exception:
    pass
layout["Planting_Seeds_(bilingual"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_01_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-3/25_icon_8.11.png
try:
    _c25 = get_crop(25, 58, 59)
    canvas.paste(_c25, (114, 3), _c25)
except Exception:
    pass
layout["8.11"] = [114, 3, 172, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_01_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-3/26_icon_icon_26.png
try:
    _c26 = get_crop(26, 50, 59)
    canvas.paste(_c26, (248, 2), _c26)
except Exception:
    pass
layout["icon_26"] = [248, 2, 298, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_01_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-3/27_icon_icon_27.png
try:
    _c27 = get_crop(27, 48, 53)
    canvas.paste(_c27, (1321, 7), _c27)
except Exception:
    pass
layout["icon_27"] = [1321, 7, 1369, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_01_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-3/28_icon_Indie_Bookstore_Day_at_Goblin_Market.png
try:
    _c28 = get_crop(28, 1344, 396)
    canvas.paste(_c28, (48, 1282), _c28)
except Exception:
    pass
layout["Indie_Bookstore_Day_at_Go"] = [48, 1282, 1392, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_01_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-3/29_icon_Grief_Ren.png
try:
    _c29 = get_crop(29, 1344, 346)
    canvas.paste(_c29, (48, 2470), _c29)
except Exception:
    pass
layout["Grief_Ren"] = [48, 2470, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_01_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-3/30_icon_icon_30.png
try:
    _c30 = get_crop(30, 56, 57)
    canvas.paste(_c30, (1213, 5), _c30)
except Exception:
    pass
layout["icon_30"] = [1213, 5, 1269, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_01_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-3/31_icon_icon_31.png
try:
    _c31 = get_crop(31, 41, 55)
    canvas.paste(_c31, (1272, 6), _c31)
except Exception:
    pass
layout["icon_31"] = [1272, 6, 1313, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_01_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-3/32_icon_icon_32.png
try:
    _c32 = get_crop(32, 44, 55)
    canvas.paste(_c32, (385, 7), _c32)
except Exception:
    pass
layout["icon_32"] = [385, 7, 429, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_01_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-3/33_icon_Self-Love_in_Nature_Releasing_Grief_thro.png
try:
    _c33 = get_crop(33, 1344, 396)
    canvas.paste(_c33, (48, 2074), _c33)
except Exception:
    pass
layout["Self-Love_in_Nature:_Rele"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_01_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-3/34_icon_6_00_PM_CDT.png
try:
    _c34 = get_crop(34, 1344, 396)
    canvas.paste(_c34, (48, 1678), _c34)
except Exception:
    pass
layout["6:00_PM_CDT"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_01_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-3/35_icon_6_00_PM_CDT.png
try:
    _c35 = get_crop(35, 1344, 396)
    canvas.paste(_c35, (48, 1678), _c35)
except Exception:
    pass
layout["6:00_PM_CDT"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_01_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-3/36_icon_Tickets.png
try:
    _c36 = get_crop(36, 288, 156)
    canvas.paste(_c36, (864, 2804), _c36)
except Exception:
    pass
layout["Tickets"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_01_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-3/37_icon_Dovetail_Brewery.png
try:
    _c37 = get_crop(37, 1344, 396)
    canvas.paste(_c37, (48, 1678), _c37)
except Exception:
    pass
layout["Dovetail_Brewery"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_01_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-3/38_text_8.11.png
try:
    _c38 = get_crop(38, 89, 41)
    canvas.paste(_c38, (20, 17), _c38)
except Exception:
    pass
layout["8.11"] = [20, 17, 109, 58]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_01_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-3/39_text_More_events_you_II_love.png
try:
    _c39 = get_crop(39, 1344, 396)
    canvas.paste(_c39, (48, 490), _c39)
except Exception:
    pass
layout["More_events_you'II_love"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_01_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-3/40_text_Tue_Apr_23.png
try:
    _c40 = get_crop(40, 200, 43)
    canvas.paste(_c40, (390, 2525), _c40)
except Exception:
    pass
layout["Tue,_Apr_23"] = [390, 2525, 590, 2568]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_01_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-3/41_text_7_00_PM_CDT.png
try:
    _c41 = get_crop(41, 1344, 346)
    canvas.paste(_c41, (48, 2470), _c41)
except Exception:
    pass
layout["7:00_PM_CDT"] = [48, 2470, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_01_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-3/42_clickable_Favorites.png
try:
    _c42 = get_crop(42, 288, 156)
    canvas.paste(_c42, (576, 2804), _c42)
except Exception:
    pass
layout["Favorites"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_01_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-3/43_clickable_More.png
try:
    _c43 = get_crop(43, 288, 156)
    canvas.paste(_c43, (1152, 2804), _c43)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]
