# page_id: page_eventbrite_d77ba2e8a5b2402385411cd9fa60262a_02
# screenshot: 2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-4.png
# step_index: 2/8
# task: Open Eventbrite. Search for "Music". Filter only free events. Choose the first event. When is the date and timing?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw background and structural UI elements for the provided canvas
# Available variables: canvas (PIL Image), draw (ImageDraw), font_sm, font_md, font_lg, font_xl

W, H = canvas.size

# Colors
bg_color = (250, 250, 250)         # page background (very light)
status_bar_color = (197, 197, 197) # top status bar grey
header_bg = (255, 255, 255)        # header/search area white
accent_blue = (46, 108, 255)       # search underline / accent
card_bg = (255, 255, 255)          # card/list background (white)
divider = (230, 230, 235)          # subtle divider lines
nav_top_border = (225, 225, 230)   # top border above bottom nav
nav_bg = (255, 255, 255)           # bottom nav background

# Clear/fill full canvas
draw.rectangle((0, 0, W, H), fill=bg_color)

# Status bar (area at very top)
status_h = 80  # status bar height in px (approx)
draw.rectangle((0, 0, W, status_h), fill=status_bar_color)

# Header / Search area (rounded rect spanning most width)
# Positioned below status bar with some top padding
header_top = status_h + 10
header_bottom = header_top + 140
header_left = 48
header_right = W - 48
header_radius = 10
# subtle drop-shadow line above header (very light)
draw.line((header_left, header_top - 6, header_right, header_top - 6), fill=(245,245,247), width=1)
# header background
try:
    draw.rounded_rectangle((header_left, header_top, header_right, header_bottom),
                           radius=header_radius, fill=header_bg, outline=None)
except Exception:
    # fallback if rounded_rectangle not available
    draw.rectangle((header_left, header_top, header_right, header_bottom), fill=header_bg)

# Blue underline for the search field (thin accent line)
underline_y = header_bottom - 6
draw.rectangle((header_left, underline_y, header_right, underline_y + 6), fill=accent_blue)

# Subtle divider under the header to separate from content
draw.line((24, header_bottom + 6, W - 24, header_bottom + 6), fill=divider, width=1)

# Content area card (main list region) - large rounded rectangle
content_top = header_bottom + 24
content_bottom = H - 180  # leave space for bottom nav
content_left = 24
content_right = W - 24
content_radius = 8
try:
    draw.rounded_rectangle((content_left, content_top, content_right, content_bottom),
                           radius=content_radius, fill=card_bg, outline=None)
except Exception:
    draw.rectangle((content_left, content_top, content_right, content_bottom), fill=card_bg)

# Section header divider (e.g., "Recent" area) - leave an area at top of content card
section_title_y = content_top + 32
# subtle horizontal line under section title area
draw.line((content_left + 24, section_title_y + 48, content_right - 24, section_title_y + 48),
          fill=divider, width=1)

# Separator lines for list items (do not draw icons/text)
# Use detected-ish y positions from the UI to place separators between rows
# These lines are to separate rows visually; they are light and thin.
item_separators = [
    534, 678, 822, 966, 1110, 1254, 1398, 1542, 1686
]
# Only draw separators that fall inside the content card bounds
for y in item_separators:
    if content_top + 10 < y < content_bottom - 10:
        draw.line((content_left + 24, y, content_right - 24, y), fill=divider, width=1)

# Add a faint vertical guide to indicate left content inset (not icons/text)
left_inset_x = content_left + 110
draw.line((left_inset_x, content_top + 16, left_inset_x, content_bottom - 16),
          fill=(248,248,249), width=1)

# Bottom navigation bar background and top border
nav_top = H - 156
draw.rectangle((0, nav_top, W, H), fill=nav_bg)
# top border line for nav
draw.line((0, nav_top, W, nav_top), fill=nav_top_border, width=1)

# Additional subtle top-of-screen thin border (very light)
draw.line((0, status_h, W, status_h), fill=(235,235,238), width=1)

# End of structural drawing

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_02_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-4/00_icon_6.48.png
try:
    _c0 = get_crop(0, 60, 64)
    canvas.paste(_c0, (114, 1), _c0)
except Exception:
    pass
layout["6.48"] = [114, 1, 174, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_02_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-4/01_icon_6.48.png
try:
    _c1 = get_crop(1, 58, 62)
    canvas.paste(_c1, (181, 1), _c1)
except Exception:
    pass
layout["6.48"] = [181, 1, 239, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_02_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-4/02_icon_Search_forae.png
try:
    _c2 = get_crop(2, 64, 64)
    canvas.paste(_c2, (309, 1), _c2)
except Exception:
    pass
layout["Search_forae"] = [309, 1, 373, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_02_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-4/03_icon_icon_3.png
try:
    _c3 = get_crop(3, 50, 62)
    canvas.paste(_c3, (248, 2), _c3)
except Exception:
    pass
layout["icon_3"] = [248, 2, 298, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_02_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-4/04_icon_Science_Tech.png
try:
    _c4 = get_crop(4, 1344, 144)
    canvas.paste(_c4, (48, 534), _c4)
except Exception:
    pass
layout["Science_&_Tech"] = [48, 534, 1392, 678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_02_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-4/05_icon_Cancel.png
try:
    _c5 = get_crop(5, 58, 62)
    canvas.paste(_c5, (1316, 0), _c5)
except Exception:
    pass
layout["Cancel"] = [1316, 0, 1374, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_02_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-4/06_icon_Cancel.png
try:
    _c6 = get_crop(6, 99, 62)
    canvas.paste(_c6, (1212, 0), _c6)
except Exception:
    pass
layout["Cancel"] = [1212, 0, 1311, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_02_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-4/07_icon_Cancel.png
try:
    _c7 = get_crop(7, 149, 144)
    canvas.paste(_c7, (1243, 97), _c7)
except Exception:
    pass
layout["Cancel"] = [1243, 97, 1392, 241]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_02_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-4/08_icon_Science_Tech.png
try:
    _c8 = get_crop(8, 1344, 144)
    canvas.paste(_c8, (48, 678), _c8)
except Exception:
    pass
layout["Science_&_Tech"] = [48, 678, 1392, 822]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_02_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-4/09_icon_Search_forae.png
try:
    _c9 = get_crop(9, 1344, 144)
    canvas.paste(_c9, (48, 390), _c9)
except Exception:
    pass
layout["Search_forae"] = [48, 390, 1392, 534]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_02_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-4/10_icon_Close_current_screen.png
try:
    _c10 = get_crop(10, 144, 144)
    canvas.paste(_c10, (1248, 822), _c10)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 822, 1392, 966]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_02_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-4/11_icon_Favorites.png
try:
    _c11 = get_crop(11, 288, 156)
    canvas.paste(_c11, (576, 2804), _c11)
except Exception:
    pass
layout["Favorites"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_02_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-4/12_icon_6.48.png
try:
    _c12 = get_crop(12, 125, 113)
    canvas.paste(_c12, (54, 115), _c12)
except Exception:
    pass
layout["6.48"] = [54, 115, 179, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_02_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-4/13_icon_Close_current_screen.png
try:
    _c13 = get_crop(13, 144, 144)
    canvas.paste(_c13, (1248, 1254), _c13)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 1254, 1392, 1398]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_02_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-4/14_icon_Close_current_screen.png
try:
    _c14 = get_crop(14, 144, 144)
    canvas.paste(_c14, (1248, 1398), _c14)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 1398, 1392, 1542]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_02_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-4/15_icon_Close_current_screen.png
try:
    _c15 = get_crop(15, 144, 144)
    canvas.paste(_c15, (1248, 534), _c15)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 534, 1392, 678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_02_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-4/16_icon_Close_current_screen.png
try:
    _c16 = get_crop(16, 144, 144)
    canvas.paste(_c16, (1248, 678), _c16)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 678, 1392, 822]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_02_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-4/17_icon_Close_current_screen.png
try:
    _c17 = get_crop(17, 144, 144)
    canvas.paste(_c17, (1248, 1110), _c17)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 1110, 1392, 1254]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_02_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-4/18_icon_Cancel.png
try:
    _c18 = get_crop(18, 144, 144)
    canvas.paste(_c18, (1248, 390), _c18)
except Exception:
    pass
layout["Cancel"] = [1248, 390, 1392, 534]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_02_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-4/19_icon_Talkshow.png
try:
    _c19 = get_crop(19, 1344, 144)
    canvas.paste(_c19, (48, 1398), _c19)
except Exception:
    pass
layout["Talkshow"] = [48, 1398, 1392, 1542]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_02_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-4/20_icon_Tickets.png
try:
    _c20 = get_crop(20, 288, 156)
    canvas.paste(_c20, (864, 2804), _c20)
except Exception:
    pass
layout["Tickets"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_02_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-4/21_icon_Close_current_screen.png
try:
    _c21 = get_crop(21, 144, 144)
    canvas.paste(_c21, (1248, 1686), _c21)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 1686, 1392, 1830]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_02_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-4/22_icon_Search_forae.png
try:
    _c22 = get_crop(22, 48, 65)
    canvas.paste(_c22, (383, 2), _c22)
except Exception:
    pass
layout["Search_forae"] = [383, 2, 431, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_02_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-4/23_icon_Close_current_screen.png
try:
    _c23 = get_crop(23, 144, 144)
    canvas.paste(_c23, (1248, 1542), _c23)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 1542, 1392, 1686]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_02_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-4/24_icon_Basketball.png
try:
    _c24 = get_crop(24, 1344, 144)
    canvas.paste(_c24, (48, 822), _c24)
except Exception:
    pass
layout["Basketball"] = [48, 822, 1392, 966]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_02_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-4/25_icon_Close_current_screen.png
try:
    _c25 = get_crop(25, 144, 144)
    canvas.paste(_c25, (1248, 966), _c25)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 966, 1392, 1110]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_02_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-4/26_icon_Home.png
try:
    _c26 = get_crop(26, 288, 156)
    canvas.paste(_c26, (0, 2804), _c26)
except Exception:
    pass
layout["Home"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_02_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-4/27_icon_Search_events.png
try:
    _c27 = get_crop(27, 288, 156)
    canvas.paste(_c27, (288, 2804), _c27)
except Exception:
    pass
layout["Search_events"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_02_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-4/28_icon_Search_forae.png
try:
    _c28 = get_crop(28, 1344, 191)
    canvas.paste(_c28, (48, 72), _c28)
except Exception:
    pass
layout["Search_forae"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_02_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-4/29_icon_Broadway.png
try:
    _c29 = get_crop(29, 1344, 144)
    canvas.paste(_c29, (48, 1542), _c29)
except Exception:
    pass
layout["Broadway"] = [48, 1542, 1392, 1686]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_02_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-4/30_icon_More.png
try:
    _c30 = get_crop(30, 288, 156)
    canvas.paste(_c30, (1152, 2804), _c30)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_02_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-4/31_icon_Taylor_Swift.png
try:
    _c31 = get_crop(31, 1344, 144)
    canvas.paste(_c31, (48, 1254), _c31)
except Exception:
    pass
layout["Taylor_Swift"] = [48, 1254, 1392, 1398]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_02_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-4/32_icon_Exhibition.png
try:
    _c32 = get_crop(32, 1344, 144)
    canvas.paste(_c32, (48, 966), _c32)
except Exception:
    pass
layout["Exhibition"] = [48, 966, 1392, 1110]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_02_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-4/33_icon_Football.png
try:
    _c33 = get_crop(33, 118, 130)
    canvas.paste(_c33, (25, 1696), _c33)
except Exception:
    pass
layout["Football"] = [25, 1696, 143, 1826]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_02_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-4/34_icon_6.48.png
try:
    _c34 = get_crop(34, 92, 63)
    canvas.paste(_c34, (15, 2), _c34)
except Exception:
    pass
layout["6.48"] = [15, 2, 107, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_02_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-4/35_text_Recent.png
try:
    _c35 = get_crop(35, 200, 56)
    canvas.paste(_c35, (46, 301), _c35)
except Exception:
    pass
layout["Recent"] = [46, 301, 246, 357]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_02_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-4/36_text_Festival.png
try:
    _c36 = get_crop(36, 154, 45)
    canvas.paste(_c36, (163, 1162), _c36)
except Exception:
    pass
layout["Festival"] = [163, 1162, 317, 1207]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_02_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-4/37_text_Football.png
try:
    _c37 = get_crop(37, 159, 43)
    canvas.paste(_c37, (165, 1740), _c37)
except Exception:
    pass
layout["Football"] = [165, 1740, 324, 1783]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_02_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-4/38_clickable_Festival.png
try:
    _c38 = get_crop(38, 1344, 144)
    canvas.paste(_c38, (48, 1110), _c38)
except Exception:
    pass
layout["Festival"] = [48, 1110, 1392, 1254]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_02_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-4/39_clickable_Football.png
try:
    _c39 = get_crop(39, 1344, 144)
    canvas.paste(_c39, (48, 1686), _c39)
except Exception:
    pass
layout["Football"] = [48, 1686, 1392, 1830]
