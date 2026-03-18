# page_id: page_eventbrite_d7ac75f457a4487c904e7baa93180729_02
# screenshot: 2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-4.png
# step_index: 2/11
# task: Open Eventbrite. Search for 'Cooking' classes. Filter to only show free events that occur in the weekend. Select the first event and proceed to checkout.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw structural background for the mobile UI page

# Colors
status_bar_color = (189, 189, 189)       # light grey for status bar
toolbar_divider_color = (38, 85, 255)    # bright blue underline for search field
subtle_divider = (228, 228, 233)         # soft separator lines
card_bg = (250, 250, 251)                # very subtle off-white for grouped card areas
nav_top_border = (230, 230, 235)         # top border of bottom nav

w, h = canvas.size

# 1) Status bar area (top ~56px)
status_h = 56
draw.rectangle([(0, 0), (w, status_h)], fill=status_bar_color)

# subtle bottom shadow line under status bar
draw.line([(0, status_h), (w, status_h)], fill=subtle_divider, width=1)

# 2) Toolbar / Search area background (just below status bar)
# Keep a white toolbar but provide a prominent blue underline for the active search
toolbar_top = status_h
toolbar_bottom = 170
draw.rectangle([(0, toolbar_top), (w, toolbar_bottom)], fill=(255, 255, 255))

# Blue underline roughly aligned with search input margins (left at x=48)
underline_y = 146
draw.line([(48, underline_y), (w-48, underline_y)], fill=toolbar_divider_color, width=4)

# a thin light divider under the toolbar for separation
draw.line([(0, toolbar_bottom), (w, toolbar_bottom)], fill=subtle_divider, width=1)

# 3) Section card background for the main "Recent" / list area
# A very subtle rounded card that groups the recent items (leaving space for icons/text to be pasted)
card_left = 24
card_right = w - 24
card_top = 480
card_bottom = 1820
draw.rounded_rectangle([(card_left, card_top), (card_right, card_bottom)], radius=12, fill=card_bg, outline=None)

# subtle shadow line at top of card
draw.line([(card_left+4, card_top+1), (card_right-4, card_top+1)], fill=subtle_divider, width=1)
draw.line([(card_left+4, card_top+2), (card_right-4, card_top+2)], fill=(245,245,247), width=1)

# 4) Horizontal separators between list rows (use positions inferred from detected rows)
# We draw thin separators across the card region leaving small insets on left/right for nicer look
sep_x0 = 48
sep_x1 = w - 48
separator_positions = [
    # Based on detected clickable rows and their heights: draw separators at bottom edges
    # recent blocks near y: 534,678,822 (their bottoms)
    677, 821, 965,
    # main clickable rows bottoms: 1109,1253,1397,1541,1685,1829
    1109, 1253, 1397, 1541, 1685, 1829
]
for y in separator_positions:
    # only draw separators that fall within the card bounds
    if card_top < y < card_bottom:
        draw.line([(sep_x0, y), (sep_x1, y)], fill=subtle_divider, width=1)

# 5) Left-side faint vertical guideline where list content starts (visual alignment helper)
draw.line([(48, card_top+8), (48, card_bottom-8)], fill=(245,245,247), width=1)

# 6) Bottom navigation area (persistent footer)
nav_top = 2804
nav_bottom = h
draw.rectangle([(0, nav_top), (w, nav_bottom)], fill=(255, 255, 255))
# top border for nav
draw.line([(0, nav_top), (w, nav_top)], fill=nav_top_border, width=1)

# small rounded background behind central area of nav to subtly ground icons (no icons drawn)
nav_center_bg_left = 0
nav_center_bg_right = w
draw.rectangle([(nav_center_bg_left, nav_top), (nav_center_bg_right, nav_top+2)], fill=nav_top_border)

# 7) Right-side column subtle separators (for the 'close' icon column)
# draw faint vertical rule where the close icons appear (visual alignment only)
close_col_x = 1248
draw.line([(close_col_x, card_top), (close_col_x, card_bottom)], fill=(245,245,247), width=1)

# 8) Final subtle overall vignette: slight lower-right corner soft overlay to mimic screenshot depth
# (very light, barely visible)
overlay_color = (255, 255, 255, 16)
# Use a few translucent rectangles to simulate softness (canvas is RGB; emulate with very light strokes)
for i, offset in enumerate(range(0, 40, 10)):
    alpha_shade = 255 - (i * 10)
    color = (255, 255, 255)
    # draw faint lines to soften corner
    draw.line([(w-1-offset, nav_top+offset), (w-1, nav_top+offset)], fill=(250,250,250), width=1)

# Note: All interactive elements, icons and text will be pasted on top at their detected positions.
# This drawing provides only background, dividers and card groupings.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_02_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-4/00_icon_4.38.png
try:
    _c0 = get_crop(0, 60, 63)
    canvas.paste(_c0, (114, 1), _c0)
except Exception:
    pass
layout["4.38"] = [114, 1, 174, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_02_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-4/01_icon_4.38.png
try:
    _c1 = get_crop(1, 58, 62)
    canvas.paste(_c1, (181, 1), _c1)
except Exception:
    pass
layout["4.38"] = [181, 1, 239, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_02_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-4/02_icon_Search_for_-..png
try:
    _c2 = get_crop(2, 64, 63)
    canvas.paste(_c2, (309, 2), _c2)
except Exception:
    pass
layout["(Search_for:-."] = [309, 2, 373, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_02_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-4/03_icon_icon_3.png
try:
    _c3 = get_crop(3, 49, 62)
    canvas.paste(_c3, (249, 2), _c3)
except Exception:
    pass
layout["icon_3"] = [249, 2, 298, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_02_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-4/04_icon_Cancel.png
try:
    _c4 = get_crop(4, 149, 144)
    canvas.paste(_c4, (1243, 97), _c4)
except Exception:
    pass
layout["Cancel"] = [1243, 97, 1392, 241]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_02_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-4/05_icon_Cancel.png
try:
    _c5 = get_crop(5, 99, 62)
    canvas.paste(_c5, (1212, 0), _c5)
except Exception:
    pass
layout["Cancel"] = [1212, 0, 1311, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_02_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-4/06_icon_Cancel.png
try:
    _c6 = get_crop(6, 58, 62)
    canvas.paste(_c6, (1316, 0), _c6)
except Exception:
    pass
layout["Cancel"] = [1316, 0, 1374, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_02_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-4/07_icon_community_events.png
try:
    _c7 = get_crop(7, 1344, 144)
    canvas.paste(_c7, (48, 534), _c7)
except Exception:
    pass
layout["community_events"] = [48, 534, 1392, 678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_02_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-4/08_icon_4.38.png
try:
    _c8 = get_crop(8, 127, 110)
    canvas.paste(_c8, (52, 113), _c8)
except Exception:
    pass
layout["4.38"] = [52, 113, 179, 223]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_02_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-4/09_icon_Tickets.png
try:
    _c9 = get_crop(9, 288, 156)
    canvas.paste(_c9, (864, 2804), _c9)
except Exception:
    pass
layout["Tickets"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_02_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-4/10_icon_community_events.png
try:
    _c10 = get_crop(10, 1344, 144)
    canvas.paste(_c10, (48, 678), _c10)
except Exception:
    pass
layout["community_events"] = [48, 678, 1392, 822]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_02_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-4/11_icon_Close_current_screen.png
try:
    _c11 = get_crop(11, 144, 144)
    canvas.paste(_c11, (1248, 822), _c11)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 822, 1392, 966]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_02_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-4/12_icon_Close_current_screen.png
try:
    _c12 = get_crop(12, 144, 144)
    canvas.paste(_c12, (1248, 534), _c12)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 534, 1392, 678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_02_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-4/13_icon_Close_current_screen.png
try:
    _c13 = get_crop(13, 144, 144)
    canvas.paste(_c13, (1248, 1254), _c13)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 1254, 1392, 1398]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_02_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-4/14_icon_Close_current_screen.png
try:
    _c14 = get_crop(14, 144, 144)
    canvas.paste(_c14, (1248, 678), _c14)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 678, 1392, 822]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_02_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-4/15_icon_Search_for_-..png
try:
    _c15 = get_crop(15, 1344, 144)
    canvas.paste(_c15, (48, 390), _c15)
except Exception:
    pass
layout["(Search_for:-."] = [48, 390, 1392, 534]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_02_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-4/16_icon_Close_current_screen.png
try:
    _c16 = get_crop(16, 144, 144)
    canvas.paste(_c16, (1248, 1398), _c16)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 1398, 1392, 1542]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_02_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-4/17_icon_Search_for_-..png
try:
    _c17 = get_crop(17, 1344, 191)
    canvas.paste(_c17, (48, 72), _c17)
except Exception:
    pass
layout["(Search_for:-."] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_02_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-4/18_icon_Close_current_screen.png
try:
    _c18 = get_crop(18, 144, 144)
    canvas.paste(_c18, (1248, 1110), _c18)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 1110, 1392, 1254]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_02_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-4/19_icon_Close_current_screen.png
try:
    _c19 = get_crop(19, 144, 144)
    canvas.paste(_c19, (1248, 1686), _c19)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 1686, 1392, 1830]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_02_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-4/20_icon_session.png
try:
    _c20 = get_crop(20, 1344, 144)
    canvas.paste(_c20, (48, 1542), _c20)
except Exception:
    pass
layout["session"] = [48, 1542, 1392, 1686]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_02_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-4/21_icon_Favorites.png
try:
    _c21 = get_crop(21, 288, 156)
    canvas.paste(_c21, (576, 2804), _c21)
except Exception:
    pass
layout["Favorites"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_02_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-4/22_icon_Close_current_screen.png
try:
    _c22 = get_crop(22, 144, 144)
    canvas.paste(_c22, (1248, 1542), _c22)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 1542, 1392, 1686]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_02_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-4/23_icon_Yoga.png
try:
    _c23 = get_crop(23, 115, 129)
    canvas.paste(_c23, (26, 1697), _c23)
except Exception:
    pass
layout["Yoga"] = [26, 1697, 141, 1826]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_02_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-4/24_icon_Cancel.png
try:
    _c24 = get_crop(24, 144, 144)
    canvas.paste(_c24, (1248, 390), _c24)
except Exception:
    pass
layout["Cancel"] = [1248, 390, 1392, 534]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_02_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-4/25_icon_Home.png
try:
    _c25 = get_crop(25, 288, 156)
    canvas.paste(_c25, (0, 2804), _c25)
except Exception:
    pass
layout["Home"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_02_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-4/26_icon_Search_events.png
try:
    _c26 = get_crop(26, 288, 156)
    canvas.paste(_c26, (288, 2804), _c26)
except Exception:
    pass
layout["Search_events"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_02_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-4/27_icon_4.38.png
try:
    _c27 = get_crop(27, 95, 62)
    canvas.paste(_c27, (13, 2), _c27)
except Exception:
    pass
layout["4.38"] = [13, 2, 108, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_02_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-4/28_icon_Close_current_screen.png
try:
    _c28 = get_crop(28, 144, 144)
    canvas.paste(_c28, (1248, 966), _c28)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 966, 1392, 1110]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_02_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-4/29_icon_community_events.png
try:
    _c29 = get_crop(29, 1344, 144)
    canvas.paste(_c29, (48, 822), _c29)
except Exception:
    pass
layout["community_events"] = [48, 822, 1392, 966]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_02_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-4/30_icon_More.png
try:
    _c30 = get_crop(30, 288, 156)
    canvas.paste(_c30, (1152, 2804), _c30)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_02_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-4/31_icon_Search_for_-..png
try:
    _c31 = get_crop(31, 47, 65)
    canvas.paste(_c31, (383, 2), _c31)
except Exception:
    pass
layout["(Search_for:-."] = [383, 2, 430, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_02_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-4/32_text_Fitness.png
try:
    _c32 = get_crop(32, 145, 51)
    canvas.paste(_c32, (163, 1017), _c32)
except Exception:
    pass
layout["Fitness"] = [163, 1017, 308, 1068]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_02_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-4/33_text_Art.png
try:
    _c33 = get_crop(33, 67, 45)
    canvas.paste(_c33, (164, 1163), _c33)
except Exception:
    pass
layout["Art"] = [164, 1163, 231, 1208]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_02_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-4/34_text_Food_and_Drink.png
try:
    _c34 = get_crop(34, 288, 50)
    canvas.paste(_c34, (164, 1302), _c34)
except Exception:
    pass
layout["Food_and_Drink"] = [164, 1302, 452, 1352]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_02_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-4/35_text_Education.png
try:
    _c35 = get_crop(35, 197, 53)
    canvas.paste(_c35, (161, 1447), _c35)
except Exception:
    pass
layout["Education"] = [161, 1447, 358, 1500]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_02_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-4/36_text_Coding_Workshop.png
try:
    _c36 = get_crop(36, 1344, 144)
    canvas.paste(_c36, (48, 1686), _c36)
except Exception:
    pass
layout["Coding_Workshop"] = [48, 1686, 1392, 1830]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_02_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-4/37_clickable_Fitness.png
try:
    _c37 = get_crop(37, 1344, 144)
    canvas.paste(_c37, (48, 966), _c37)
except Exception:
    pass
layout["Fitness"] = [48, 966, 1392, 1110]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_02_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-4/38_clickable_Art.png
try:
    _c38 = get_crop(38, 1344, 144)
    canvas.paste(_c38, (48, 1110), _c38)
except Exception:
    pass
layout["Art"] = [48, 1110, 1392, 1254]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_02_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-4/39_clickable_Food_and_Drink.png
try:
    _c39 = get_crop(39, 1344, 144)
    canvas.paste(_c39, (48, 1254), _c39)
except Exception:
    pass
layout["Food_and_Drink"] = [48, 1254, 1392, 1398]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_02_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-4/40_clickable_Education.png
try:
    _c40 = get_crop(40, 1344, 144)
    canvas.paste(_c40, (48, 1398), _c40)
except Exception:
    pass
layout["Education"] = [48, 1398, 1392, 1542]
