# page_id: page_eventbrite_4fbf805fbd914a178f72f68b0bc03f81_02
# screenshot: 2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-4.png
# step_index: 2/10
# task: Open Eventbrite. Explore "Education" events. Apply filters for events happening tomorrow. From the list, select the third event and check out its description.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw UI background and structure for the provided canvas
# Assumes: canvas (PIL.Image), draw (PIL.ImageDraw), font_sm/font_md/font_lg/font_xl exist

# Colors
bg_color = (255, 255, 255)           # page background (dominant color)
status_bar_color = (158, 162, 166)   # muted grey for status bar
header_bg = (255, 255, 255)          # header area stays white
search_underline = (25, 118, 210)    # Eventbrite-like blue accent
divider_color = (233, 233, 239)      # subtle divider
row_bg = (250, 251, 252)             # very light row background
row_outline = (236, 236, 240)        # row outline / shadow
bottom_nav_bg = (255, 255, 255)      # bottom nav background
bottom_nav_border = (222, 222, 226)  # top border for bottom nav

W, H = canvas.size

# Full background (canvas is already white, but set explicitly)
draw.rectangle([(0, 0), (W, H)], fill=bg_color)

# Status bar area (top ~64px)
status_h = 64
draw.rectangle([(0, 0), (W, status_h)], fill=status_bar_color)

# Header / search area background (below status bar)
header_top = status_h
header_bottom = 160
draw.rectangle([(0, header_top), (W, header_bottom)], fill=header_bg)

# Blue underline for the search input (thin accent line)
underline_y = 132
underline_margin_x = 72
draw.line([(underline_margin_x, underline_y), (W - underline_margin_x, underline_y)],
          fill=search_underline, width=4)

# Light divider under header
draw.line([(0, header_bottom), (W, header_bottom)], fill=divider_color, width=1)

# Draw subtle rounded card backgrounds for each "Recent" list row
# Rows detected in the crop metadata: starting y positions and heights
row_x = 48
row_w = 1344
row_h = 144
row_positions = [390, 534, 678, 822, 966, 1110, 1254, 1398, 1542, 1696]

radius = 12
for y in row_positions:
    x1 = row_x
    y1 = y
    x2 = row_x + row_w
    y2 = y + row_h
    # Slight outline to separate rows from the page
    draw.rounded_rectangle([(x1, y1), (x2, y2)], radius=radius, fill=row_bg, outline=row_outline, width=1)
    # Subtle bottom divider (to emphasize row separation)
    draw.line([(x1 + 20, y2), (x2 - 20, y2)], fill=divider_color, width=1)

# Section separator line above the "Recent" rows (a bit above the first row)
sep_y = row_positions[0] - 40
draw.line([(48, sep_y), (W - 48, sep_y)], fill=divider_color, width=1)

# Bottom navigation bar background and top border
nav_h = 156
nav_top = H - nav_h
draw.rectangle([(0, nav_top), (W, H)], fill=bottom_nav_bg)
# top border line
draw.line([(0, nav_top), (W, nav_top)], fill=bottom_nav_border, width=2)
# subtle inner top shadow
draw.line([(0, nav_top + 2), (W, nav_top + 2)], fill=(245, 245, 247), width=1)

# A faint full-width bottom edge to give a grounded look
draw.line([(0, H - 1), (W, H - 1)], fill=(220, 220, 223), width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_02_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-4/00_icon_4.56.png
try:
    _c0 = get_crop(0, 58, 62)
    canvas.paste(_c0, (181, 1), _c0)
except Exception:
    pass
layout["4.56"] = [181, 1, 239, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_02_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-4/01_icon_4.56.png
try:
    _c1 = get_crop(1, 60, 64)
    canvas.paste(_c1, (114, 1), _c1)
except Exception:
    pass
layout["4.56"] = [114, 1, 174, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_02_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-4/02_icon_Search_forae.png
try:
    _c2 = get_crop(2, 63, 63)
    canvas.paste(_c2, (309, 2), _c2)
except Exception:
    pass
layout["Search_forae"] = [309, 2, 372, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_02_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-4/03_icon_icon_3.png
try:
    _c3 = get_crop(3, 49, 61)
    canvas.paste(_c3, (249, 2), _c3)
except Exception:
    pass
layout["icon_3"] = [249, 2, 298, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_02_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-4/04_icon_community_events.png
try:
    _c4 = get_crop(4, 1344, 144)
    canvas.paste(_c4, (48, 1398), _c4)
except Exception:
    pass
layout["community_events"] = [48, 1398, 1392, 1542]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_02_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-4/05_icon_Cancel.png
try:
    _c5 = get_crop(5, 56, 62)
    canvas.paste(_c5, (1317, 0), _c5)
except Exception:
    pass
layout["Cancel"] = [1317, 0, 1373, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_02_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-4/06_icon_community_events.png
try:
    _c6 = get_crop(6, 1344, 144)
    canvas.paste(_c6, (48, 1254), _c6)
except Exception:
    pass
layout["community_events"] = [48, 1254, 1392, 1398]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_02_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-4/07_icon_Open_Mic_Night.png
try:
    _c7 = get_crop(7, 1344, 144)
    canvas.paste(_c7, (48, 534), _c7)
except Exception:
    pass
layout["Open_Mic_Night"] = [48, 534, 1392, 678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_02_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-4/08_icon_community_events.png
try:
    _c8 = get_crop(8, 1344, 144)
    canvas.paste(_c8, (48, 1542), _c8)
except Exception:
    pass
layout["community_events"] = [48, 1542, 1392, 1686]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_02_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-4/09_icon_Cancel.png
try:
    _c9 = get_crop(9, 149, 144)
    canvas.paste(_c9, (1243, 97), _c9)
except Exception:
    pass
layout["Cancel"] = [1243, 97, 1392, 241]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_02_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-4/10_icon_Cancel.png
try:
    _c10 = get_crop(10, 97, 62)
    canvas.paste(_c10, (1212, 0), _c10)
except Exception:
    pass
layout["Cancel"] = [1212, 0, 1309, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_02_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-4/11_icon_4.56.png
try:
    _c11 = get_crop(11, 123, 111)
    canvas.paste(_c11, (55, 116), _c11)
except Exception:
    pass
layout["4.56"] = [55, 116, 178, 227]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_02_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-4/12_icon_Close_current_screen.png
try:
    _c12 = get_crop(12, 144, 144)
    canvas.paste(_c12, (1248, 822), _c12)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 822, 1392, 966]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_02_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-4/13_icon_Tickets.png
try:
    _c13 = get_crop(13, 288, 156)
    canvas.paste(_c13, (864, 2804), _c13)
except Exception:
    pass
layout["Tickets"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_02_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-4/14_icon_Open_Mic_Night.png
try:
    _c14 = get_crop(14, 1344, 144)
    canvas.paste(_c14, (48, 390), _c14)
except Exception:
    pass
layout["Open_Mic_Night"] = [48, 390, 1392, 534]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_02_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-4/15_icon_Favorites.png
try:
    _c15 = get_crop(15, 288, 156)
    canvas.paste(_c15, (576, 2804), _c15)
except Exception:
    pass
layout["Favorites"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_02_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-4/16_icon_Close_current_screen.png
try:
    _c16 = get_crop(16, 144, 144)
    canvas.paste(_c16, (1248, 678), _c16)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 678, 1392, 822]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_02_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-4/17_icon_Close_current_screen.png
try:
    _c17 = get_crop(17, 144, 144)
    canvas.paste(_c17, (1248, 534), _c17)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 534, 1392, 678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_02_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-4/18_icon_Close_current_screen.png
try:
    _c18 = get_crop(18, 144, 144)
    canvas.paste(_c18, (1248, 1254), _c18)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 1254, 1392, 1398]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_02_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-4/19_icon_Close_current_screen.png
try:
    _c19 = get_crop(19, 144, 144)
    canvas.paste(_c19, (1248, 1398), _c19)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 1398, 1392, 1542]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_02_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-4/20_icon_Close_current_screen.png
try:
    _c20 = get_crop(20, 144, 144)
    canvas.paste(_c20, (1248, 1110), _c20)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 1110, 1392, 1254]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_02_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-4/21_icon_Close_current_screen.png
try:
    _c21 = get_crop(21, 144, 144)
    canvas.paste(_c21, (1248, 1686), _c21)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 1686, 1392, 1830]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_02_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-4/22_icon_Photography.png
try:
    _c22 = get_crop(22, 1344, 144)
    canvas.paste(_c22, (48, 678), _c22)
except Exception:
    pass
layout["Photography"] = [48, 678, 1392, 822]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_02_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-4/23_icon_Close_current_screen.png
try:
    _c23 = get_crop(23, 144, 144)
    canvas.paste(_c23, (1248, 1542), _c23)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 1542, 1392, 1686]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_02_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-4/24_icon_Search_forae.png
try:
    _c24 = get_crop(24, 48, 64)
    canvas.paste(_c24, (383, 2), _c24)
except Exception:
    pass
layout["Search_forae"] = [383, 2, 431, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_02_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-4/25_icon_Cancel.png
try:
    _c25 = get_crop(25, 144, 144)
    canvas.paste(_c25, (1248, 390), _c25)
except Exception:
    pass
layout["Cancel"] = [1248, 390, 1392, 534]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_02_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-4/26_icon_Fitness.png
try:
    _c26 = get_crop(26, 118, 130)
    canvas.paste(_c26, (25, 1696), _c26)
except Exception:
    pass
layout["Fitness"] = [25, 1696, 143, 1826]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_02_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-4/27_icon_community_events.png
try:
    _c27 = get_crop(27, 1344, 144)
    canvas.paste(_c27, (48, 1110), _c27)
except Exception:
    pass
layout["community_events"] = [48, 1110, 1392, 1254]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_02_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-4/28_icon_Search_forae.png
try:
    _c28 = get_crop(28, 1344, 191)
    canvas.paste(_c28, (48, 72), _c28)
except Exception:
    pass
layout["Search_forae"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_02_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-4/29_icon_Close_current_screen.png
try:
    _c29 = get_crop(29, 144, 144)
    canvas.paste(_c29, (1248, 966), _c29)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 966, 1392, 1110]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_02_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-4/30_icon_Home.png
try:
    _c30 = get_crop(30, 288, 156)
    canvas.paste(_c30, (0, 2804), _c30)
except Exception:
    pass
layout["Home"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_02_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-4/31_icon_Search_events.png
try:
    _c31 = get_crop(31, 288, 156)
    canvas.paste(_c31, (288, 2804), _c31)
except Exception:
    pass
layout["Search_events"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_02_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-4/32_icon_Cooking.png
try:
    _c32 = get_crop(32, 1344, 144)
    canvas.paste(_c32, (48, 966), _c32)
except Exception:
    pass
layout["Cooking"] = [48, 966, 1392, 1110]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_02_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-4/33_icon_Wellness.png
try:
    _c33 = get_crop(33, 1344, 144)
    canvas.paste(_c33, (48, 822), _c33)
except Exception:
    pass
layout["Wellness"] = [48, 822, 1392, 966]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_02_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-4/34_icon_More.png
try:
    _c34 = get_crop(34, 288, 156)
    canvas.paste(_c34, (1152, 2804), _c34)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_02_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-4/35_icon_4.56.png
try:
    _c35 = get_crop(35, 94, 62)
    canvas.paste(_c35, (13, 2), _c35)
except Exception:
    pass
layout["4.56"] = [13, 2, 107, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_02_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-4/36_text_Recent.png
try:
    _c36 = get_crop(36, 203, 56)
    canvas.paste(_c36, (46, 301), _c36)
except Exception:
    pass
layout["Recent"] = [46, 301, 249, 357]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_02_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-4/37_text_Fitness.png
try:
    _c37 = get_crop(37, 140, 43)
    canvas.paste(_c37, (165, 1740), _c37)
except Exception:
    pass
layout["Fitness"] = [165, 1740, 305, 1783]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_02_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-4/38_clickable_Fitness.png
try:
    _c38 = get_crop(38, 1344, 144)
    canvas.paste(_c38, (48, 1686), _c38)
except Exception:
    pass
layout["Fitness"] = [48, 1686, 1392, 1830]
