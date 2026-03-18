# page_id: page_eventbrite_4fbf805fbd914a178f72f68b0bc03f81_07
# screenshot: 2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-9.png
# step_index: 7/10
# task: Open Eventbrite. Explore "Education" events. Apply filters for events happening tomorrow. From the list, select the third event and check out its description.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw background and structural UI elements for the Eventbrite-like screen.
# Uses provided `canvas` (PIL Image) and `draw` (ImageDraw) objects.

# Colors
status_bar_color = "#d7d7d7"   # light gray status bar
header_bg = "#ffffff"          # header background (white)
divider = "#e6e6e6"            # subtle dividers
card_shadow = "#e9e9e9"        # card shadow color
card_fill = "#ffffff"          # card fill
card_border = "#f2f2f2"        # light card border
nav_bg = "#ffffff"             # bottom nav background
page_bg = "#ffffff"            # page background (canvas already white)

W, H = canvas.size

# 1) Status bar area at the very top (~56px)
status_h = 56
draw.rectangle([0, 0, W, status_h], fill=status_bar_color)

# 2) Header / toolbar background (below status bar)
header_top = status_h
header_bottom = 160
draw.rectangle([0, header_top, W, header_bottom], fill=header_bg)

# Header bottom divider line
draw.line([(24, header_bottom), (W-24, header_bottom)], fill=divider, width=2)

# 3) Subtle section divider under filter/header area (separates filters from list)
# Position chosen to sit below expected filter pills (safe to draw background only)
filter_div_y = 320
draw.line([(24, filter_div_y), (W-24, filter_div_y)], fill=divider, width=1)

# 4) Event list "card" containers (rounded rectangles with soft shadows)
card_x0 = 48
card_x1 = W - 48
corner_radius = 22

# Card 1 (top event container)
card1_y0 = 340
card1_y1 = 820
# shadow
draw.rounded_rectangle([card_x0, card1_y0 + 8, card_x1, card1_y1 + 8], radius=corner_radius, fill=card_shadow)
# main card
draw.rounded_rectangle([card_x0, card1_y0, card_x1, card1_y1], radius=corner_radius, fill=card_fill, outline=card_border, width=1)
# separator below card1
draw.line([(card_x0+8, card1_y1+14), (card_x1-8, card1_y1+14)], fill=divider, width=1)

# Card 2 (second event container)
card2_y0 = 900
card2_y1 = 1400
draw.rounded_rectangle([card_x0, card2_y0 + 8, card_x1, card2_y1 + 8], radius=corner_radius, fill=card_shadow)
draw.rounded_rectangle([card_x0, card2_y0, card_x1, card2_y1], radius=corner_radius, fill=card_fill, outline=card_border, width=1)
draw.line([(card_x0+8, card2_y1+14), (card_x1-8, card2_y1+14)], fill=divider, width=1)

# Card 3 (third event container / featured content)
card3_y0 = 1480
card3_y1 = 2200
draw.rounded_rectangle([card_x0, card3_y0 + 8, card_x1, card3_y1 + 8], radius=corner_radius, fill=card_shadow)
draw.rounded_rectangle([card_x0, card3_y0, card_x1, card3_y1], radius=corner_radius, fill=card_fill, outline=card_border, width=1)
draw.line([(card_x0+8, card3_y1+14), (card_x1-8, card3_y1+14)], fill=divider, width=1)

# 5) Subtle horizontal separators for content sections further down
sep_positions = [card1_y1 + 80, card2_y1 + 80, card3_y1 + 80]
for y in sep_positions:
    if y < H - 200:
        draw.line([(24, y), (W-24, y)], fill="#f3f3f3", width=1)

# 6) Bottom navigation background area and top divider
nav_top = 2804
draw.line([(24, nav_top), (W-24, nav_top)], fill=divider, width=2)
draw.rectangle([0, nav_top, W, H], fill=nav_bg)

# 7) Light top shadow for the page (very subtle)
draw.line([(0, status_h), (W, status_h)], fill="#eeeeee", width=1)

# NOTE: All icons, buttons, text, and image content are intentionally NOT drawn here.
# This file only lays out the background, cards, dividers, and structural elements.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_07_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-9/00_icon_Tomorrow.png
try:
    _c0 = get_crop(0, 1344, 191)
    canvas.paste(_c0, (48, 72), _c0)
except Exception:
    pass
layout["Tomorrow"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_07_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-9/01_icon_Music.png
try:
    _c1 = get_crop(1, 197, 111)
    canvas.paste(_c1, (875, 406), _c1)
except Exception:
    pass
layout["Music"] = [875, 406, 1072, 517]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_07_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-9/02_icon_Business.png
try:
    _c2 = get_crop(2, 250, 112)
    canvas.paste(_c2, (1074, 406), _c2)
except Exception:
    pass
layout["Business"] = [1074, 406, 1324, 518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_07_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-9/03_icon_1_Filter.png
try:
    _c3 = get_crop(3, 492, 144)
    canvas.paste(_c3, (0, 259), _c3)
except Exception:
    pass
layout["1_Filter"] = [0, 259, 492, 403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_07_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-9/04_icon_Business.png
try:
    _c4 = get_crop(4, 144, 144)
    canvas.paste(_c4, (1092, 578), _c4)
except Exception:
    pass
layout["Business"] = [1092, 578, 1236, 722]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_07_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-9/05_icon_Business.png
try:
    _c5 = get_crop(5, 144, 144)
    canvas.paste(_c5, (1236, 578), _c5)
except Exception:
    pass
layout["Business"] = [1236, 578, 1380, 722]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_07_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-9/06_icon_REAL_ESTATE_your.png
try:
    _c6 = get_crop(6, 144, 144)
    canvas.paste(_c6, (1092, 1820), _c6)
except Exception:
    pass
layout["REAL_ESTATE_your"] = [1092, 1820, 1236, 1964]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_07_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-9/07_icon_WIN_by_making.png
try:
    _c7 = get_crop(7, 144, 144)
    canvas.paste(_c7, (1236, 1820), _c7)
except Exception:
    pass
layout["WIN_by_making"] = [1236, 1820, 1380, 1964]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_07_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-9/08_icon_Business.png
try:
    _c8 = get_crop(8, 90, 108)
    canvas.paste(_c8, (1329, 407), _c8)
except Exception:
    pass
layout["Business"] = [1329, 407, 1419, 515]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_07_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-9/09_icon_Close_current_screen.png
try:
    _c9 = get_crop(9, 144, 144)
    canvas.paste(_c9, (1248, 96), _c9)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 96, 1392, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_07_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-9/10_icon_Education.png
try:
    _c10 = get_crop(10, 59, 63)
    canvas.paste(_c10, (311, 0), _c10)
except Exception:
    pass
layout["Education"] = [311, 0, 370, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_07_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-9/11_icon_4.56.png
try:
    _c11 = get_crop(11, 54, 65)
    canvas.paste(_c11, (117, 0), _c11)
except Exception:
    pass
layout["4.56"] = [117, 0, 171, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_07_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-9/12_icon_4.56.png
try:
    _c12 = get_crop(12, 54, 63)
    canvas.paste(_c12, (183, 1), _c12)
except Exception:
    pass
layout["4.56"] = [183, 1, 237, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_07_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-9/13_icon_4.56.png
try:
    _c13 = get_crop(13, 111, 110)
    canvas.paste(_c13, (61, 115), _c13)
except Exception:
    pass
layout["4.56"] = [61, 115, 172, 225]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_07_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-9/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 47, 62)
    canvas.paste(_c14, (250, 1), _c14)
except Exception:
    pass
layout["icon_14"] = [250, 1, 297, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_07_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-9/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 61, 62)
    canvas.paste(_c15, (1212, 0), _c15)
except Exception:
    pass
layout["icon_15"] = [1212, 0, 1273, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_07_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-9/16_icon_icon_16.png
try:
    _c16 = get_crop(16, 53, 62)
    canvas.paste(_c16, (1318, 0), _c16)
except Exception:
    pass
layout["icon_16"] = [1318, 0, 1371, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_07_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-9/17_icon_Tickets.png
try:
    _c17 = get_crop(17, 288, 156)
    canvas.paste(_c17, (864, 2804), _c17)
except Exception:
    pass
layout["Tickets"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_07_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-9/18_icon_Los_Angeles.png
try:
    _c18 = get_crop(18, 492, 144)
    canvas.paste(_c18, (0, 259), _c18)
except Exception:
    pass
layout["Los_Angeles"] = [0, 259, 492, 403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_07_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-9/19_icon_icon_19.png
try:
    _c19 = get_crop(19, 40, 62)
    canvas.paste(_c19, (1273, 0), _c19)
except Exception:
    pass
layout["icon_19"] = [1273, 0, 1313, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_07_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-9/20_icon_Education.png
try:
    _c20 = get_crop(20, 1344, 191)
    canvas.paste(_c20, (48, 72), _c20)
except Exception:
    pass
layout["Education"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_07_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-9/21_icon_Education.png
try:
    _c21 = get_crop(21, 44, 61)
    canvas.paste(_c21, (385, 2), _c21)
except Exception:
    pass
layout["Education"] = [385, 2, 429, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_07_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-9/22_icon_Abg.png
try:
    _c22 = get_crop(22, 288, 156)
    canvas.paste(_c22, (288, 2804), _c22)
except Exception:
    pass
layout["Abg"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_07_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-9/23_icon_Free.png
try:
    _c23 = get_crop(23, 127, 76)
    canvas.paste(_c23, (91, 1998), _c23)
except Exception:
    pass
layout["Free"] = [91, 1998, 218, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_07_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-9/24_icon_Apr_25_-_Sat_May_25_._6.30_PM_EDT.png
try:
    _c24 = get_crop(24, 1344, 731)
    canvas.paste(_c24, (48, 525), _c24)
except Exception:
    pass
layout["Apr_25_-_Sat,_May_25_._6."] = [48, 525, 1392, 1256]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_07_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-9/25_icon_More.png
try:
    _c25 = get_crop(25, 288, 156)
    canvas.paste(_c25, (1152, 2804), _c25)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_07_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-9/26_icon_Home.png
try:
    _c26 = get_crop(26, 288, 156)
    canvas.paste(_c26, (0, 2804), _c26)
except Exception:
    pass
layout["Home"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_07_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-9/27_icon_Introduction_To_Our_Nationwide_Community.png
try:
    _c27 = get_crop(27, 1344, 1108)
    canvas.paste(_c27, (48, 1304), _c27)
except Exception:
    pass
layout["Introduction_To_Our_Natio"] = [48, 1304, 1392, 2412]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_07_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-9/28_icon_ONE.png
try:
    _c28 = get_crop(28, 288, 156)
    canvas.paste(_c28, (576, 2804), _c28)
except Exception:
    pass
layout["ONE"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_07_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-9/29_icon_4.56.png
try:
    _c29 = get_crop(29, 92, 64)
    canvas.paste(_c29, (15, 0), _c29)
except Exception:
    pass
layout["4.56"] = [15, 0, 107, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_07_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-9/30_icon_CD_E4FL.png
try:
    _c30 = get_crop(30, 1344, 1108)
    canvas.paste(_c30, (48, 1304), _c30)
except Exception:
    pass
layout["CD_E4FL"] = [48, 1304, 1392, 2412]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_07_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-9/31_icon_Promoted.png
try:
    _c31 = get_crop(31, 41, 61)
    canvas.paste(_c31, (284, 1151), _c31)
except Exception:
    pass
layout["Promoted"] = [284, 1151, 325, 1212]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_07_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-9/32_icon_Promoted.png
try:
    _c32 = get_crop(32, 240, 66)
    canvas.paste(_c32, (86, 1147), _c32)
except Exception:
    pass
layout["Promoted"] = [86, 1147, 326, 1213]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_07_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-9/33_text_Ticket_sales_end_soon.png
try:
    _c33 = get_crop(33, 415, 51)
    canvas.paste(_c33, (125, 775), _c33)
except Exception:
    pass
layout["Ticket_sales_end_soon"] = [125, 775, 540, 826]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_07_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-9/34_text_NCOME.png
try:
    _c34 = get_crop(34, 158, 40)
    canvas.paste(_c34, (168, 1419), _c34)
except Exception:
    pass
layout["NCOME"] = [168, 1419, 326, 1459]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_07_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-9/35_text_REAL_ESTATE.png
try:
    _c35 = get_crop(35, 504, 68)
    canvas.paste(_c35, (469, 1391), _c35)
except Exception:
    pass
layout["REAL_ESTATE"] = [469, 1391, 973, 1459]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_07_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-9/36_text_Discover.png
try:
    _c36 = get_crop(36, 267, 61)
    canvas.paste(_c36, (1049, 1410), _c36)
except Exception:
    pass
layout["Discover"] = [1049, 1410, 1316, 1471]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_07_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-9/37_text_IS.png
try:
    _c37 = get_crop(37, 84, 63)
    canvas.paste(_c37, (857, 1484), _c37)
except Exception:
    pass
layout["IS"] = [857, 1484, 941, 1547]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_07_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-9/38_text_The.png
try:
    _c38 = get_crop(38, 1344, 356)
    canvas.paste(_c38, (48, 2460), _c38)
except Exception:
    pass
layout["The"] = [48, 2460, 1392, 2816]
