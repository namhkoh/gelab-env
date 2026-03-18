# page_id: page_eventbrite_6b75132d6e874d9a960bba273e5f011b_07
# screenshot: 2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-9.png
# step_index: 7/11
# task: Open Eventbrite. Set the city to 'San Francisco'. Search 'Outdoor'. Select an event starting after 5 PM. Check the ticket price.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw UI background and structural elements for the mobile page
# Uses provided variables: canvas (PIL Image), draw (ImageDraw)

# Colors
WHITE = (255, 255, 255)
STATUS_BAR_GRAY = (189, 189, 189)    # top status bar
HEADER_BLUE = (45, 91, 240)          # bright blue underline for search
DIVIDER_GRAY = (220, 220, 225)       # subtle dividers
CARD_BG = (250, 250, 251)            # very light card background
NAV_TOP_BORDER = (233, 233, 235)     # nav bar top divider

w, h = canvas.size

# Full background (ensure consistent base)
draw.rectangle((0, 0, w, h), fill=WHITE)

# Status bar area (top ~84px)
status_h = 84
draw.rectangle((0, 0, w, status_h), fill=STATUS_BAR_GRAY)

# Header / search area (directly under status bar)
header_top = status_h
header_bottom = 200
draw.rectangle((0, header_top, w, header_bottom), fill=WHITE)

# Blue search underline (match search field width margin seen in screenshot)
search_left = 48
search_right = w - 48
underline_y = header_bottom
underline_thickness = 6
draw.rectangle((search_left, underline_y, search_right, underline_y + underline_thickness), fill=HEADER_BLUE)

# Thin divider under the blue underline for subtle separation
draw.line((search_left, underline_y + underline_thickness + 2, search_right, underline_y + underline_thickness + 2),
          fill=DIVIDER_GRAY, width=1)

# "Recent" heading area - leave text out, but provide a subtle background/buffer
recent_top = underline_y + underline_thickness + 24
recent_bottom = recent_top + 64
draw.rectangle((search_left, recent_top, search_right, recent_bottom), fill=WHITE)

# Rounded card/background for the list group (subtle very-light background)
list_card_top = recent_bottom + 24
list_card_bottom = 1780
card_left = 36
card_right = w - 36
draw.rounded_rectangle((card_left, list_card_top, card_right, list_card_bottom),
                       radius=10, fill=CARD_BG)

# Separator lines between list items (positions inferred from UI layout)
# These lines run the same width as the content area (match the card margins)
item_y_positions = [390, 534, 678, 822, 966, 1110, 1254, 1398, 1542, 1686]
for y in item_y_positions:
    # only draw separators that lie inside the list card region
    if list_card_top < y < list_card_bottom:
        draw.line((search_left, y, search_right, y), fill=DIVIDER_GRAY, width=1)

# Additional subtle vertical left guide (content inset) - not text or icons
content_inset_x = 120
draw.line((content_inset_x, list_card_top + 12, content_inset_x, list_card_bottom - 12),
          fill=(245, 245, 246), width=1)

# Bottom navigation bar background and top divider
nav_top = 2804
draw.rectangle((0, nav_top, w, h), fill=WHITE)
draw.line((0, nav_top, w, nav_top), fill=NAV_TOP_BORDER, width=2)

# Slight shadow above nav to separate it from content
shadow_y = nav_top - 6
draw.line((0, shadow_y, w, shadow_y), fill=(245, 245, 246), width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_07_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-9/00_icon_8.11.png
try:
    _c0 = get_crop(0, 58, 63)
    canvas.paste(_c0, (114, 2), _c0)
except Exception:
    pass
layout["8.11"] = [114, 2, 172, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_07_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-9/01_icon_8.11.png
try:
    _c1 = get_crop(1, 57, 61)
    canvas.paste(_c1, (181, 2), _c1)
except Exception:
    pass
layout["8.11"] = [181, 2, 238, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_07_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-9/02_icon_icon_2.png
try:
    _c2 = get_crop(2, 49, 67)
    canvas.paste(_c2, (1154, 1), _c2)
except Exception:
    pass
layout["icon_2"] = [1154, 1, 1203, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_07_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-9/03_icon_Search_for__..png
try:
    _c3 = get_crop(3, 62, 62)
    canvas.paste(_c3, (310, 2), _c3)
except Exception:
    pass
layout["[Search_for__."] = [310, 2, 372, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_07_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-9/04_icon_icon_4.png
try:
    _c4 = get_crop(4, 46, 60)
    canvas.paste(_c4, (251, 3), _c4)
except Exception:
    pass
layout["icon_4"] = [251, 3, 297, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_07_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-9/05_icon_Cancel.png
try:
    _c5 = get_crop(5, 97, 65)
    canvas.paste(_c5, (1212, 1), _c5)
except Exception:
    pass
layout["Cancel"] = [1212, 1, 1309, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_07_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-9/06_icon_Cancel.png
try:
    _c6 = get_crop(6, 149, 144)
    canvas.paste(_c6, (1243, 97), _c6)
except Exception:
    pass
layout["Cancel"] = [1243, 97, 1392, 241]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_07_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-9/07_icon_Cancel.png
try:
    _c7 = get_crop(7, 54, 62)
    canvas.paste(_c7, (1318, 1), _c7)
except Exception:
    pass
layout["Cancel"] = [1318, 1, 1372, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_07_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-9/08_icon_community_events.png
try:
    _c8 = get_crop(8, 1344, 144)
    canvas.paste(_c8, (48, 534), _c8)
except Exception:
    pass
layout["community_events"] = [48, 534, 1392, 678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_07_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-9/09_icon_8.11.png
try:
    _c9 = get_crop(9, 126, 107)
    canvas.paste(_c9, (49, 116), _c9)
except Exception:
    pass
layout["8.11"] = [49, 116, 175, 223]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_07_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-9/10_icon_Close_current_screen.png
try:
    _c10 = get_crop(10, 144, 144)
    canvas.paste(_c10, (1248, 822), _c10)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 822, 1392, 966]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_07_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-9/11_icon_Tickets.png
try:
    _c11 = get_crop(11, 288, 156)
    canvas.paste(_c11, (864, 2804), _c11)
except Exception:
    pass
layout["Tickets"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_07_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-9/12_icon_Close_current_screen.png
try:
    _c12 = get_crop(12, 144, 144)
    canvas.paste(_c12, (1248, 534), _c12)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 534, 1392, 678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_07_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-9/13_icon_Close_current_screen.png
try:
    _c13 = get_crop(13, 144, 144)
    canvas.paste(_c13, (1248, 678), _c13)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 678, 1392, 822]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_07_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-9/14_icon_Close_current_screen.png
try:
    _c14 = get_crop(14, 144, 144)
    canvas.paste(_c14, (1248, 1254), _c14)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 1254, 1392, 1398]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_07_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-9/15_icon_Close_current_screen.png
try:
    _c15 = get_crop(15, 144, 144)
    canvas.paste(_c15, (1248, 1398), _c15)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 1398, 1392, 1542]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_07_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-9/16_icon_Close_current_screen.png
try:
    _c16 = get_crop(16, 144, 144)
    canvas.paste(_c16, (1248, 1686), _c16)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 1686, 1392, 1830]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_07_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-9/17_icon_Close_current_screen.png
try:
    _c17 = get_crop(17, 144, 144)
    canvas.paste(_c17, (1248, 1110), _c17)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 1110, 1392, 1254]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_07_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-9/18_icon_Coding_Workshop.png
try:
    _c18 = get_crop(18, 1344, 144)
    canvas.paste(_c18, (48, 1542), _c18)
except Exception:
    pass
layout["Coding_Workshop"] = [48, 1542, 1392, 1686]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_07_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-9/19_icon_Search_for__..png
try:
    _c19 = get_crop(19, 1344, 191)
    canvas.paste(_c19, (48, 72), _c19)
except Exception:
    pass
layout["[Search_for__."] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_07_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-9/20_icon_Close_current_screen.png
try:
    _c20 = get_crop(20, 144, 144)
    canvas.paste(_c20, (1248, 1542), _c20)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 1542, 1392, 1686]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_07_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-9/21_icon_Cancel.png
try:
    _c21 = get_crop(21, 144, 144)
    canvas.paste(_c21, (1248, 390), _c21)
except Exception:
    pass
layout["Cancel"] = [1248, 390, 1392, 534]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_07_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-9/22_icon_Music.png
try:
    _c22 = get_crop(22, 116, 131)
    canvas.paste(_c22, (26, 1696), _c22)
except Exception:
    pass
layout["Music"] = [26, 1696, 142, 1827]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_07_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-9/23_icon_Close_current_screen.png
try:
    _c23 = get_crop(23, 144, 144)
    canvas.paste(_c23, (1248, 966), _c23)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 966, 1392, 1110]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_07_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-9/24_icon_Favorites.png
try:
    _c24 = get_crop(24, 288, 156)
    canvas.paste(_c24, (576, 2804), _c24)
except Exception:
    pass
layout["Favorites"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_07_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-9/25_icon_community_events.png
try:
    _c25 = get_crop(25, 1344, 144)
    canvas.paste(_c25, (48, 390), _c25)
except Exception:
    pass
layout["community_events"] = [48, 390, 1392, 534]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_07_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-9/26_icon_Search_events.png
try:
    _c26 = get_crop(26, 288, 156)
    canvas.paste(_c26, (288, 2804), _c26)
except Exception:
    pass
layout["Search_events"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_07_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-9/27_icon_Coding_Workshop.png
try:
    _c27 = get_crop(27, 1344, 144)
    canvas.paste(_c27, (48, 1398), _c27)
except Exception:
    pass
layout["Coding_Workshop"] = [48, 1398, 1392, 1542]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_07_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-9/28_icon_Fitness.png
try:
    _c28 = get_crop(28, 1344, 144)
    canvas.paste(_c28, (48, 678), _c28)
except Exception:
    pass
layout["Fitness"] = [48, 678, 1392, 822]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_07_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-9/29_icon_Home.png
try:
    _c29 = get_crop(29, 288, 156)
    canvas.paste(_c29, (0, 2804), _c29)
except Exception:
    pass
layout["Home"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_07_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-9/30_icon_More.png
try:
    _c30 = get_crop(30, 288, 156)
    canvas.paste(_c30, (1152, 2804), _c30)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_07_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-9/31_icon_Search_for__..png
try:
    _c31 = get_crop(31, 47, 62)
    canvas.paste(_c31, (383, 3), _c31)
except Exception:
    pass
layout["[Search_for__."] = [383, 3, 430, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_07_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-9/32_icon_Art.png
try:
    _c32 = get_crop(32, 1344, 144)
    canvas.paste(_c32, (48, 822), _c32)
except Exception:
    pass
layout["Art"] = [48, 822, 1392, 966]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_07_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-9/33_icon_Coding_Workshop.png
try:
    _c33 = get_crop(33, 1344, 144)
    canvas.paste(_c33, (48, 1254), _c33)
except Exception:
    pass
layout["Coding_Workshop"] = [48, 1254, 1392, 1398]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_07_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-9/34_text_8.11.png
try:
    _c34 = get_crop(34, 89, 41)
    canvas.paste(_c34, (20, 17), _c34)
except Exception:
    pass
layout["8.11"] = [20, 17, 109, 58]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_07_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-9/35_text_Food_and_Drink.png
try:
    _c35 = get_crop(35, 290, 51)
    canvas.paste(_c35, (162, 1015), _c35)
except Exception:
    pass
layout["Food_and_Drink"] = [162, 1015, 452, 1066]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_07_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-9/36_text_Education.png
try:
    _c36 = get_crop(36, 197, 55)
    canvas.paste(_c36, (161, 1158), _c36)
except Exception:
    pass
layout["Education"] = [161, 1158, 358, 1213]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_07_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-9/37_text_Music.png
try:
    _c37 = get_crop(37, 120, 45)
    canvas.paste(_c37, (165, 1738), _c37)
except Exception:
    pass
layout["Music"] = [165, 1738, 285, 1783]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_07_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-9/38_clickable_Food_and_Drink.png
try:
    _c38 = get_crop(38, 1344, 144)
    canvas.paste(_c38, (48, 966), _c38)
except Exception:
    pass
layout["Food_and_Drink"] = [48, 966, 1392, 1110]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_07_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-9/39_clickable_Education.png
try:
    _c39 = get_crop(39, 1344, 144)
    canvas.paste(_c39, (48, 1110), _c39)
except Exception:
    pass
layout["Education"] = [48, 1110, 1392, 1254]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_07_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-9/40_clickable_Music.png
try:
    _c40 = get_crop(40, 1344, 144)
    canvas.paste(_c40, (48, 1686), _c40)
except Exception:
    pass
layout["Music"] = [48, 1686, 1392, 1830]
