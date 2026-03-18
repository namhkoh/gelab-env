# page_id: page_eventbrite_31528ae16c264b1b98bd4e1f25c5d9e5_02
# screenshot: 2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-4.png
# step_index: 2/11
# task: Open Eventbrite. Search 'Fitness'. Filter for free events. Browse and select any 'Yoga' event. Note the location.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw UI background and structural elements for the mobile page
# Available: canvas (PIL Image 1440x2960), draw (ImageDraw), font_sm, font_md, font_lg, font_xl

# Colors
BG_WHITE = (255, 255, 255)
STATUS_GRAY = (200, 200, 200)        # status bar background
HEADER_DIVIDER_BLUE = (38, 81, 255)  # accent underline under search
LIGHT_DIVIDER = (235, 235, 240)      # subtle section dividers
NAV_TOP_DIVIDER = (225, 225, 230)    # nav bar top border

# Clear canvas / ensure background
draw.rectangle((0, 0, 1440, 2960), fill=BG_WHITE)

# Status bar (top area)
STATUS_H = 50
draw.rectangle((0, 0, 1440, STATUS_H), fill=STATUS_GRAY)

# Header area (search area background) — keep white but ensure separation from status bar
HEADER_TOP = STATUS_H
HEADER_BOTTOM = 260
draw.rectangle((0, HEADER_TOP, 1440, HEADER_BOTTOM), fill=BG_WHITE)

# Accent underline for the search/header area (thin blue line across inner margins)
UNDERLINE_Y = 150
UNDERLINE_X0 = 48
UNDERLINE_X1 = 1440 - 48
draw.line((UNDERLINE_X0, UNDERLINE_Y, UNDERLINE_X1, UNDERLINE_Y), fill=HEADER_DIVIDER_BLUE, width=4)

# subtle light divider under the header block
draw.line((UNDERLINE_X0, UNDERLINE_Y + 10, UNDERLINE_X1, UNDERLINE_Y + 10), fill=LIGHT_DIVIDER, width=1)

# Section separator between header and content
draw.line((24, 280, 1440 - 24, 280), fill=LIGHT_DIVIDER, width=1)

# Row separators for the recent/list items (light thin lines across content area with left/right padding)
row_y_positions = [
    534, 678, 822, 966, 1110, 1254, 1398, 1542, 1686
]
for y in row_y_positions:
    draw.line((UNDERLINE_X0, y, UNDERLINE_X1, y), fill=LIGHT_DIVIDER, width=1)

# Subtle section divider near the "Recent" title area (to structure spacing, not text)
draw.line((UNDERLINE_X0, 360, UNDERLINE_X1, 360), fill=LIGHT_DIVIDER, width=1)

# Bottom navigation bar background and top divider
NAV_TOP = 2804
draw.line((0, NAV_TOP, 1440, NAV_TOP), fill=NAV_TOP_DIVIDER, width=2)
draw.rectangle((0, NAV_TOP, 1440, 2960), fill=BG_WHITE)

# Slight shadow band above nav to give depth
shadow_y0 = NAV_TOP - 8
shadow_y1 = NAV_TOP
draw.rectangle((0, shadow_y0, 1440, shadow_y1), fill=(248, 248, 250))

# Optional subtle left/right page margins guide (very faint) to align content blocks
margin_color = (250, 250, 251)
draw.line((48, 0, 48, 2960), fill=margin_color, width=1)
draw.line((1440 - 48, 0, 1440 - 48, 2960), fill=margin_color, width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_02_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-4/00_icon_7.54.png
try:
    _c0 = get_crop(0, 59, 63)
    canvas.paste(_c0, (115, 1), _c0)
except Exception:
    pass
layout["7.54"] = [115, 1, 174, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_02_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-4/01_icon_7.54.png
try:
    _c1 = get_crop(1, 58, 62)
    canvas.paste(_c1, (181, 1), _c1)
except Exception:
    pass
layout["7.54"] = [181, 1, 239, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_02_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-4/02_icon_Search_forae.png
try:
    _c2 = get_crop(2, 63, 64)
    canvas.paste(_c2, (309, 1), _c2)
except Exception:
    pass
layout["Search_forae"] = [309, 1, 372, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_02_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-4/03_icon_icon_3.png
try:
    _c3 = get_crop(3, 49, 62)
    canvas.paste(_c3, (249, 2), _c3)
except Exception:
    pass
layout["icon_3"] = [249, 2, 298, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_02_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-4/04_icon_Cancel.png
try:
    _c4 = get_crop(4, 56, 63)
    canvas.paste(_c4, (1317, 0), _c4)
except Exception:
    pass
layout["Cancel"] = [1317, 0, 1373, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_02_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-4/05_icon_Cancel.png
try:
    _c5 = get_crop(5, 149, 144)
    canvas.paste(_c5, (1243, 97), _c5)
except Exception:
    pass
layout["Cancel"] = [1243, 97, 1392, 241]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_02_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-4/06_icon_Food_and_Drink.png
try:
    _c6 = get_crop(6, 1344, 144)
    canvas.paste(_c6, (48, 534), _c6)
except Exception:
    pass
layout["Food_and_Drink"] = [48, 534, 1392, 678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_02_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-4/07_icon_Cancel.png
try:
    _c7 = get_crop(7, 98, 63)
    canvas.paste(_c7, (1212, 0), _c7)
except Exception:
    pass
layout["Cancel"] = [1212, 0, 1310, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_02_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-4/08_icon_Search_forae.png
try:
    _c8 = get_crop(8, 1344, 144)
    canvas.paste(_c8, (48, 390), _c8)
except Exception:
    pass
layout["Search_forae"] = [48, 390, 1392, 534]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_02_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-4/09_icon_7.54.png
try:
    _c9 = get_crop(9, 122, 114)
    canvas.paste(_c9, (56, 115), _c9)
except Exception:
    pass
layout["7.54"] = [56, 115, 178, 229]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_02_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-4/10_icon_7.54.png
try:
    _c10 = get_crop(10, 93, 63)
    canvas.paste(_c10, (14, 1), _c10)
except Exception:
    pass
layout["7.54"] = [14, 1, 107, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_02_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-4/11_icon_Close_current_screen.png
try:
    _c11 = get_crop(11, 144, 144)
    canvas.paste(_c11, (1248, 822), _c11)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 822, 1392, 966]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_02_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-4/12_icon_Tickets.png
try:
    _c12 = get_crop(12, 288, 156)
    canvas.paste(_c12, (864, 2804), _c12)
except Exception:
    pass
layout["Tickets"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_02_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-4/13_icon_Close_current_screen.png
try:
    _c13 = get_crop(13, 144, 144)
    canvas.paste(_c13, (1248, 534), _c13)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 534, 1392, 678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_02_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-4/14_icon_Close_current_screen.png
try:
    _c14 = get_crop(14, 144, 144)
    canvas.paste(_c14, (1248, 678), _c14)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 678, 1392, 822]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_02_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-4/15_icon_Close_current_screen.png
try:
    _c15 = get_crop(15, 144, 144)
    canvas.paste(_c15, (1248, 1686), _c15)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 1686, 1392, 1830]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_02_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-4/16_icon_Close_current_screen.png
try:
    _c16 = get_crop(16, 144, 144)
    canvas.paste(_c16, (1248, 1398), _c16)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 1398, 1392, 1542]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_02_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-4/17_icon_Close_current_screen.png
try:
    _c17 = get_crop(17, 144, 144)
    canvas.paste(_c17, (1248, 1254), _c17)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 1254, 1392, 1398]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_02_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-4/18_icon_Close_current_screen.png
try:
    _c18 = get_crop(18, 144, 144)
    canvas.paste(_c18, (1248, 1542), _c18)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 1542, 1392, 1686]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_02_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-4/19_icon_Search_events.png
try:
    _c19 = get_crop(19, 288, 156)
    canvas.paste(_c19, (288, 2804), _c19)
except Exception:
    pass
layout["Search_events"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_02_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-4/20_icon_Close_current_screen.png
try:
    _c20 = get_crop(20, 144, 144)
    canvas.paste(_c20, (1248, 1110), _c20)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 1110, 1392, 1254]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_02_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-4/21_icon_Food_Drink.png
try:
    _c21 = get_crop(21, 1344, 144)
    canvas.paste(_c21, (48, 1542), _c21)
except Exception:
    pass
layout["Food_&_Drink"] = [48, 1542, 1392, 1686]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_02_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-4/22_icon_Favorites.png
try:
    _c22 = get_crop(22, 288, 156)
    canvas.paste(_c22, (576, 2804), _c22)
except Exception:
    pass
layout["Favorites"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_02_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-4/23_icon_Home.png
try:
    _c23 = get_crop(23, 288, 156)
    canvas.paste(_c23, (0, 2804), _c23)
except Exception:
    pass
layout["Home"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_02_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-4/24_icon_Search_forae.png
try:
    _c24 = get_crop(24, 1344, 191)
    canvas.paste(_c24, (48, 72), _c24)
except Exception:
    pass
layout["Search_forae"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_02_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-4/25_icon_Cancel.png
try:
    _c25 = get_crop(25, 144, 144)
    canvas.paste(_c25, (1248, 390), _c25)
except Exception:
    pass
layout["Cancel"] = [1248, 390, 1392, 534]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_02_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-4/26_icon_Education.png
try:
    _c26 = get_crop(26, 1344, 144)
    canvas.paste(_c26, (48, 678), _c26)
except Exception:
    pass
layout["Education"] = [48, 678, 1392, 822]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_02_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-4/27_icon_Search_forae.png
try:
    _c27 = get_crop(27, 47, 65)
    canvas.paste(_c27, (383, 2), _c27)
except Exception:
    pass
layout["Search_forae"] = [383, 2, 430, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_02_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-4/28_icon_Close_current_screen.png
try:
    _c28 = get_crop(28, 144, 144)
    canvas.paste(_c28, (1248, 966), _c28)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 966, 1392, 1110]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_02_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-4/29_icon_Food_Drink.png
try:
    _c29 = get_crop(29, 1344, 144)
    canvas.paste(_c29, (48, 1398), _c29)
except Exception:
    pass
layout["Food_&_Drink"] = [48, 1398, 1392, 1542]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_02_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-4/30_icon_session.png
try:
    _c30 = get_crop(30, 1344, 144)
    canvas.paste(_c30, (48, 822), _c30)
except Exception:
    pass
layout["session"] = [48, 822, 1392, 966]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_02_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-4/31_icon_Science_Tech.png
try:
    _c31 = get_crop(31, 113, 130)
    canvas.paste(_c31, (27, 1697), _c31)
except Exception:
    pass
layout["Science_&_Tech"] = [27, 1697, 140, 1827]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_02_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-4/32_icon_More.png
try:
    _c32 = get_crop(32, 288, 156)
    canvas.paste(_c32, (1152, 2804), _c32)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_02_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-4/33_icon_Coding_Workshop.png
try:
    _c33 = get_crop(33, 1344, 144)
    canvas.paste(_c33, (48, 966), _c33)
except Exception:
    pass
layout["Coding_Workshop"] = [48, 966, 1392, 1110]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_02_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-4/34_icon_Music_Festival.png
try:
    _c34 = get_crop(34, 1344, 144)
    canvas.paste(_c34, (48, 1254), _c34)
except Exception:
    pass
layout["Music_Festival"] = [48, 1254, 1392, 1398]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_02_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-4/35_icon_Coding_Workshop.png
try:
    _c35 = get_crop(35, 1344, 144)
    canvas.paste(_c35, (48, 1110), _c35)
except Exception:
    pass
layout["Coding_Workshop"] = [48, 1110, 1392, 1254]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_02_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-4/36_text_Recent.png
try:
    _c36 = get_crop(36, 200, 56)
    canvas.paste(_c36, (46, 301), _c36)
except Exception:
    pass
layout["Recent"] = [46, 301, 246, 357]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_02_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-4/37_text_Science_Tech.png
try:
    _c37 = get_crop(37, 292, 49)
    canvas.paste(_c37, (160, 1735), _c37)
except Exception:
    pass
layout["Science_&_Tech"] = [160, 1735, 452, 1784]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_02_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-4/38_clickable_Science_Tech.png
try:
    _c38 = get_crop(38, 1344, 144)
    canvas.paste(_c38, (48, 1686), _c38)
except Exception:
    pass
layout["Science_&_Tech"] = [48, 1686, 1392, 1830]
