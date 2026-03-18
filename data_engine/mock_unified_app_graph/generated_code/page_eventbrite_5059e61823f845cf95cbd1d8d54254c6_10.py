# page_id: page_eventbrite_5059e61823f845cf95cbd1d8d54254c6_10
# screenshot: 2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-12.png
# step_index: 10/19
# task: Open Eventbrite. Look for 'Education' in Los Angeles happening on May 4. Filter to show only free events. How many events are posted?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Canvas and drawing objects are provided: canvas (PIL Image), draw (ImageDraw), fonts: font_sm, font_md, font_lg, font_xl
w, h = canvas.size

# Colors
bg_color = "#fbfcfe"           # page background (very light)
status_color = "#8f9498"       # status bar gray
header_bg = "#ffffff"          # header background (white)
divider = "#e6e7eb"            # subtle divider lines
card_shadow = "#e9edf3"        # shadow behind cards
card_bg = "#ffffff"            # card background
section_bg = "#f5f8fb"         # subtle section background (for filter row)
nav_bg = "#ffffff"             # bottom navigation background

# Fill overall background
draw.rectangle([(0, 0), (w, h)], fill=bg_color)

# Status bar (top area)
status_h = 96
draw.rectangle([(0, 0), (w, status_h)], fill=status_color)

# Header / toolbar area (below status)
header_top = status_h
header_h = 140
draw.rectangle([(0, header_top), (w, header_top + header_h)], fill=header_bg)

# Header bottom divider
draw.line([(40, header_top + header_h), (w - 40, header_top + header_h)], fill=divider, width=2)

# Filters/Chips strip background (behind the chips; chips themselves will be pasted separately)
filters_top = header_top + header_h + 16
filters_bottom = filters_top + 120
draw.rectangle([(0, filters_top), (w, filters_bottom)], fill=section_bg)
# subtle top and bottom separators for filter area
draw.line([(40, filters_top), (w - 40, filters_top)], fill=divider, width=1)
draw.line([(40, filters_bottom), (w - 40, filters_bottom)], fill=divider, width=1)

# Main content area - list of event cards
margin_x = 48
card_radius = 28

# First event card (background rounded rectangle and subtle shadow)
card1_x0 = margin_x
card1_x1 = w - margin_x
card1_y0 = 600
card1_y1 = 1888
# shadow
draw.rounded_rectangle([(card1_x0 + 6, card1_y0 + 8), (card1_x1 + 6, card1_y1 + 8)], radius=card_radius, fill=card_shadow)
# card
draw.rounded_rectangle([(card1_x0, card1_y0), (card1_x1, card1_y1)], radius=card_radius, fill=card_bg, outline=divider, width=1)

# Subtle internal separator inside first card (to separate image area from details)
sep_y1 = card1_y0 + int((card1_y1 - card1_y0) * 0.62)
draw.line([(card1_x0 + 28, sep_y1), (card1_x1 - 28, sep_y1)], fill=divider, width=1)

# Second event card (below the first)
card2_y0 = card1_y1 + 24
card2_y1 = card2_y0 + 680
# shadow
draw.rounded_rectangle([(card1_x0 + 6, card2_y0 + 8), (card1_x1 + 6, card2_y1 + 8)], radius=card_radius, fill=card_shadow)
# card
draw.rounded_rectangle([(card1_x0, card2_y0), (card1_x1, card2_y1)], radius=card_radius, fill=card_bg, outline=divider, width=1)

# Separator line between cards area and the rest of the content
draw.line([(margin_x, card2_y1 + 24), (w - margin_x, card2_y1 + 24)], fill=divider, width=1)

# Additional subtle content band behind the large images (gives feel of a gray image background)
image_band_top = card1_y0 + 12
image_band_bottom = sep_y1 - 12
draw.rectangle([(card1_x0 + 12, image_band_top), (card1_x1 - 12, image_band_bottom)], fill="#fcfcfd")

image2_band_top = card2_y0 + 12
image2_band_bottom = card2_y0 + int((card2_y1 - card2_y0) * 0.55)
draw.rectangle([(card1_x0 + 12, image2_band_top), (card1_x1 - 12, image2_band_bottom)], fill="#fcfcfd")

# Floating section label background (e.g., where "2,810 events" appears) - just the background block (text will be pasted)
label_block_x0 = margin_x
label_block_x1 = margin_x + 380
label_block_y0 = filters_bottom + 24
label_block_y1 = label_block_y0 + 80
draw.rectangle([(label_block_x0, label_block_y0), (label_block_x1, label_block_y1)], fill=header_bg)

# Bottom navigation bar background and top divider
nav_h = 120
nav_top = h - nav_h
draw.rectangle([(0, nav_top), (w, h)], fill=nav_bg)
draw.line([(0, nav_top), (w, nav_top)], fill=divider, width=2)

# Small top-left location pin strip background (behind "Los Angeles" area)
loc_strip_top = header_top + 28
loc_strip_bottom = loc_strip_top + 64
draw.rectangle([(margin_x, loc_strip_top), (margin_x + 460, loc_strip_bottom)], fill=header_bg)
draw.line([(margin_x, loc_strip_bottom), (margin_x + 460, loc_strip_bottom)], fill=divider, width=1)

# End of drawing. UI elements (icons, texts, images, buttons) will be pasted on top by the caller.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_10_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-12/00_icon_Music.png
try:
    _c0 = get_crop(0, 187, 103)
    canvas.paste(_c0, (837, 410), _c0)
except Exception:
    pass
layout["Music"] = [837, 410, 1024, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_10_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-12/01_icon_Business.png
try:
    _c1 = get_crop(1, 241, 103)
    canvas.paste(_c1, (1036, 410), _c1)
except Exception:
    pass
layout["Business"] = [1036, 410, 1277, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_10_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-12/02_icon_Anytime.png
try:
    _c2 = get_crop(2, 400, 103)
    canvas.paste(_c2, (425, 410), _c2)
except Exception:
    pass
layout["Anytime"] = [425, 410, 825, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_10_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-12/03_icon_Filters.png
try:
    _c3 = get_crop(3, 359, 103)
    canvas.paste(_c3, (54, 410), _c3)
except Exception:
    pass
layout["Filters"] = [54, 410, 413, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_10_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-12/04_icon_Foo.png
try:
    _c4 = get_crop(4, 149, 110)
    canvas.paste(_c4, (1283, 406), _c4)
except Exception:
    pass
layout["Foo"] = [1283, 406, 1432, 516]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_10_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-12/05_icon_Favorite_button.png
try:
    _c5 = get_crop(5, 144, 144)
    canvas.paste(_c5, (1092, 2415), _c5)
except Exception:
    pass
layout["Favorite_button"] = [1092, 2415, 1236, 2559]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_10_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-12/06_icon_Favorite_button.png
try:
    _c6 = get_crop(6, 144, 144)
    canvas.paste(_c6, (1092, 1192), _c6)
except Exception:
    pass
layout["Favorite_button"] = [1092, 1192, 1236, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_10_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-12/07_icon_Overflow_menu_button.png
try:
    _c7 = get_crop(7, 144, 144)
    canvas.paste(_c7, (1236, 1192), _c7)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 1192, 1380, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_10_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-12/08_icon_Overflow_menu_button.png
try:
    _c8 = get_crop(8, 144, 144)
    canvas.paste(_c8, (1236, 2415), _c8)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 2415, 1380, 2559]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_10_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-12/09_icon_Foo.png
try:
    _c9 = get_crop(9, 144, 144)
    canvas.paste(_c9, (1248, 96), _c9)
except Exception:
    pass
layout["Foo"] = [1248, 96, 1392, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_10_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-12/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 54, 65)
    canvas.paste(_c10, (1151, 0), _c10)
except Exception:
    pass
layout["icon_10"] = [1151, 0, 1205, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_10_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-12/11_icon_7.35.png
try:
    _c11 = get_crop(11, 122, 114)
    canvas.paste(_c11, (55, 114), _c11)
except Exception:
    pass
layout["7.35"] = [55, 114, 177, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_10_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-12/12_icon_Tom_Everhart_at_the_Grand_Opening_of_Cho.png
try:
    _c12 = get_crop(12, 1344, 1175)
    canvas.paste(_c12, (48, 676), _c12)
except Exception:
    pass
layout["Tom_Everhart_at_the_Grand"] = [48, 676, 1392, 1851]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_10_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-12/13_icon_Education.png
try:
    _c13 = get_crop(13, 68, 63)
    canvas.paste(_c13, (308, 0), _c13)
except Exception:
    pass
layout["Education"] = [308, 0, 376, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_10_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-12/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 101, 63)
    canvas.paste(_c14, (1211, 0), _c14)
except Exception:
    pass
layout["icon_14"] = [1211, 0, 1312, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_10_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-12/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 54, 64)
    canvas.paste(_c15, (246, 0), _c15)
except Exception:
    pass
layout["icon_15"] = [246, 0, 300, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_10_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-12/16_icon_7.35.png
try:
    _c16 = get_crop(16, 61, 63)
    canvas.paste(_c16, (181, 0), _c16)
except Exception:
    pass
layout["7.35"] = [181, 0, 242, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_10_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-12/17_icon_Education.png
try:
    _c17 = get_crop(17, 1344, 191)
    canvas.paste(_c17, (48, 72), _c17)
except Exception:
    pass
layout["Education"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_10_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-12/18_icon_icon_18.png
try:
    _c18 = get_crop(18, 57, 61)
    canvas.paste(_c18, (1318, 0), _c18)
except Exception:
    pass
layout["icon_18"] = [1318, 0, 1375, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_10_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-12/19_icon_7.35.png
try:
    _c19 = get_crop(19, 61, 65)
    canvas.paste(_c19, (115, 0), _c19)
except Exception:
    pass
layout["7.35"] = [115, 0, 176, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_10_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-12/20_icon_Los_Angeles.png
try:
    _c20 = get_crop(20, 492, 144)
    canvas.paste(_c20, (0, 259), _c20)
except Exception:
    pass
layout["Los_Angeles"] = [0, 259, 492, 403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_10_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-12/21_icon_Education.png
try:
    _c21 = get_crop(21, 51, 61)
    canvas.paste(_c21, (384, 2), _c21)
except Exception:
    pass
layout["Education"] = [384, 2, 435, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_10_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-12/22_icon_Sat_Apr_27_._I.OO_AM_PDT.png
try:
    _c22 = get_crop(22, 288, 156)
    canvas.paste(_c22, (288, 2804), _c22)
except Exception:
    pass
layout["Sat,_Apr_27_._I.OO_AM_PDT"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_10_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-12/23_icon_Promoted.png
try:
    _c23 = get_crop(23, 241, 67)
    canvas.paste(_c23, (85, 1744), _c23)
except Exception:
    pass
layout["Promoted"] = [85, 1744, 326, 1811]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_10_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-12/24_icon_Erlends.png
try:
    _c24 = get_crop(24, 288, 156)
    canvas.paste(_c24, (864, 2804), _c24)
except Exception:
    pass
layout["Erlends"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_10_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-12/25_icon_Tickets.png
try:
    _c25 = get_crop(25, 288, 156)
    canvas.paste(_c25, (864, 2804), _c25)
except Exception:
    pass
layout["Tickets"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_10_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-12/26_icon_Kitten_Shower.png
try:
    _c26 = get_crop(26, 288, 156)
    canvas.paste(_c26, (0, 2804), _c26)
except Exception:
    pass
layout["Kitten_Shower"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_10_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-12/27_icon_More.png
try:
    _c27 = get_crop(27, 288, 156)
    canvas.paste(_c27, (1152, 2804), _c27)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_10_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-12/28_icon_7.35.png
try:
    _c28 = get_crop(28, 153, 64)
    canvas.paste(_c28, (6, 0), _c28)
except Exception:
    pass
layout["7.35"] = [6, 0, 159, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_10_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-12/29_icon_Sat_Apr_27_._I.OO_AM_PDT.png
try:
    _c29 = get_crop(29, 288, 156)
    canvas.paste(_c29, (576, 2804), _c29)
except Exception:
    pass
layout["Sat,_Apr_27_._I.OO_AM_PDT"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_10_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-12/30_text_2_810events.png
try:
    _c30 = get_crop(30, 359, 103)
    canvas.paste(_c30, (54, 410), _c30)
except Exception:
    pass
layout["2,810events"] = [54, 410, 413, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_10_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-12/31_text_313_N_Beverly_Dr.png
try:
    _c31 = get_crop(31, 323, 55)
    canvas.paste(_c31, (90, 1686), _c31)
except Exception:
    pass
layout["313_N_Beverly_Dr"] = [90, 1686, 413, 1741]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_10_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-12/32_text_Ysoarkitten_showe.png
try:
    _c32 = get_crop(32, 1344, 917)
    canvas.paste(_c32, (48, 1899), _c32)
except Exception:
    pass
layout["Ysoarkitten_showe"] = [48, 1899, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_10_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-12/33_text_April_27.png
try:
    _c33 = get_crop(33, 188, 61)
    canvas.paste(_c33, (196, 2247), _c33)
except Exception:
    pass
layout["April_27"] = [196, 2247, 384, 2308]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_10_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-12/34_text_I1am-4pm.png
try:
    _c34 = get_crop(34, 1344, 917)
    canvas.paste(_c34, (48, 1899), _c34)
except Exception:
    pass
layout["I1am-4pm"] = [48, 1899, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_10_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-12/35_text_Dest_Fricnds_Pel_Adoplion_Center.png
try:
    _c35 = get_crop(35, 1344, 917)
    canvas.paste(_c35, (48, 1899), _c35)
except Exception:
    pass
layout["Dest_Fricnds_Pel_Adoplion"] = [48, 1899, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_10_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-12/36_text_1845_Pontius_Ave.png
try:
    _c36 = get_crop(36, 287, 39)
    canvas.paste(_c36, (290, 2387), _c36)
except Exception:
    pass
layout["1845_Pontius_Ave:"] = [290, 2387, 577, 2426]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_10_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-12/37_text_Los_Angeles._CA_90225.png
try:
    _c37 = get_crop(37, 367, 43)
    canvas.paste(_c37, (251, 2432), _c37)
except Exception:
    pass
layout["Los_Angeles._CA_90225"] = [251, 2432, 618, 2475]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_10_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-12/38_text_Best.png
try:
    _c38 = get_crop(38, 87, 39)
    canvas.paste(_c38, (380, 2510), _c38)
except Exception:
    pass
layout["Best"] = [380, 2510, 467, 2549]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_10_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-12/39_text_Free.png
try:
    _c39 = get_crop(39, 80, 39)
    canvas.paste(_c39, (117, 2614), _c39)
except Exception:
    pass
layout["Free"] = [117, 2614, 197, 2653]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_10_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-12/40_text_Kitten_Shower.png
try:
    _c40 = get_crop(40, 288, 156)
    canvas.paste(_c40, (288, 2804), _c40)
except Exception:
    pass
layout["Kitten_Shower"] = [288, 2804, 576, 2960]
