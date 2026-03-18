# page_id: page_eventbrite_02f151acef934b59b90856d9e8041920_10
# screenshot: 2024_4_24_17_24_02f151acef934b59b90856d9e8041920-12.png
# step_index: 10/11
# task: Open Eventbrite. Check the "Tech" events happening this month. Open the first event and check its date and time.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: fallback_compose
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw screen background and structural UI elements for the Eventbrite "Tech" feed mockup.
# Available variables: canvas (PIL Image), draw (ImageDraw), font_sm, font_md, font_lg, font_xl

# Clear / set base background (dominant white)
draw.rectangle([(0, 0), canvas.size], fill="#ffffff")

# Status bar (top area)
STATUS_H = 72
draw.rectangle([(0, 0), (1440, STATUS_H)], fill="#cfcfcf")

# Header / toolbar area below status bar
HEADER_TOP = STATUS_H
HEADER_H = 88
draw.rectangle([(0, HEADER_TOP), (1440, HEADER_TOP + HEADER_H)], fill="#ffffff")

# Search field background (structural only — no icons/text)
SEARCH_LEFT = 48
SEARCH_RIGHT = 1392
SEARCH_TOP = HEADER_TOP + 24
SEARCH_BOTTOM = SEARCH_TOP + 52
draw.rounded_rectangle(
    [(SEARCH_LEFT, SEARCH_TOP), (SEARCH_RIGHT, SEARCH_BOTTOM)],
    radius=20,
    fill="#fbfdff",
    outline="#e6e6e9",
    width=1
)

# Thin divider under header
DIV_Y = HEADER_TOP + HEADER_H
draw.line([(48, DIV_Y), (1392, DIV_Y)], fill="#e6e6e9", width=2)

# Subtle section divider (above filters area)
FILTERS_DIV_Y = 520
draw.line([(48, FILTERS_DIV_Y), (1392, FILTERS_DIV_Y)], fill="#f0f1f3", width=1)

# Event card 1 background + shadow (big rounded card containing image + meta)
CARD_MARGIN_X = 36
card1_top = 600
card1_bottom = 1880
card_radius = 28

# Shadow
shadow_offset = 10
draw.rounded_rectangle(
    [(CARD_MARGIN_X + 6, card1_top + shadow_offset, 1440 - CARD_MARGIN_X + 6, card1_bottom + shadow_offset)],
    radius=card_radius,
    fill="#efeff1"
)

# Card surface
draw.rounded_rectangle(
    [(CARD_MARGIN_X, card1_top), (1440 - CARD_MARGIN_X, card1_bottom)],
    radius=card_radius,
    fill="#ffffff",
    outline="#e8e8ea",
    width=1
)

# Content area background for the first event image (dark placeholder behind the image)
img1_left = 48
img1_top = 676
img1_right = img1_left + 1344
img1_bottom = img1_top + 1175
draw.rounded_rectangle(
    [(img1_left, img1_top), (img1_right, img1_bottom)],
    radius=22,
    fill="#111827"
)

# Event card 2 background + shadow
card2_top = 1888
card2_bottom = 2628
# Shadow
draw.rounded_rectangle(
    [(CARD_MARGIN_X + 6, card2_top + shadow_offset, 1440 - CARD_MARGIN_X + 6, card2_bottom + shadow_offset)],
    radius=card_radius,
    fill="#efeff1"
)
# Card surface
draw.rounded_rectangle(
    [(CARD_MARGIN_X, card2_top), (1440 - CARD_MARGIN_X, card2_bottom)],
    radius=card_radius,
    fill="#ffffff",
    outline="#e8e8ea",
    width=1
)

# Content area background for the second event image (darker placeholder)
img2_left = 48
# second image detected at y=1899 (title) and big image starts lower; use detected area around 1960..2440
img2_top = 1960
img2_right = img2_left + 1344
img2_bottom = img2_top + 480
draw.rounded_rectangle(
    [(img2_left, img2_top), (img2_right, img2_bottom)],
    radius=20,
    fill="#0b1220"
)

# Subtle horizontal separators between cards/content blocks
sep_x_start = 48
sep_x_end = 1392
draw.line([(sep_x_start, card1_bottom + 12), (sep_x_end, card1_bottom + 12)], fill="#f0f1f3", width=1)
draw.line([(sep_x_start, card2_bottom + 12), (sep_x_end, card2_bottom + 12)], fill="#f0f1f3", width=1)

# Bottom navigation bar area
NAV_H = 140
nav_top = canvas.size[1] - NAV_H
draw.rectangle([(0, nav_top), (1440, canvas.size[1])], fill="#ffffff")
# Top divider of nav
draw.line([(0, nav_top), (1440, nav_top)], fill="#e6e6e9", width=2)

# Small subtle page edge gutters (left/right) to match mobile layout spacing
# (visual guides only, very light)
draw.line([(24, 0), (24, canvas.size[1])], fill="#ffffff00", width=1)  # effectively invisible placeholder

# End of structural drawing.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_10_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-12/00_icon_Music.png
try:
    _c0 = get_crop(0, 187, 103)
    canvas.paste(_c0, (1021, 410), _c0)
except Exception:
    pass
layout["Music"] = [1021, 410, 1208, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_10_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-12/01_icon_Apr_24_-_30_2024.png
try:
    _c1 = get_crop(1, 571, 103)
    canvas.paste(_c1, (438, 410), _c1)
except Exception:
    pass
layout["Apr_24_-_30,_2024"] = [438, 410, 1009, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_10_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-12/02_icon_1_Filter.png
try:
    _c2 = get_crop(2, 372, 103)
    canvas.paste(_c2, (54, 410), _c2)
except Exception:
    pass
layout["1_Filter"] = [54, 410, 426, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_10_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-12/03_icon_Busine.png
try:
    _c3 = get_crop(3, 172, 103)
    canvas.paste(_c3, (1220, 410), _c3)
except Exception:
    pass
layout["Busine:"] = [1220, 410, 1392, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_10_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-12/04_icon_Icademy.png
try:
    _c4 = get_crop(4, 144, 144)
    canvas.paste(_c4, (1092, 1192), _c4)
except Exception:
    pass
layout["Icademy"] = [1092, 1192, 1236, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_10_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-12/05_icon_presehter.png
try:
    _c5 = get_crop(5, 144, 144)
    canvas.paste(_c5, (1236, 1192), _c5)
except Exception:
    pass
layout["presehter"] = [1236, 1192, 1380, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_10_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-12/06_icon_Favorite_button.png
try:
    _c6 = get_crop(6, 144, 144)
    canvas.paste(_c6, (1092, 2415), _c6)
except Exception:
    pass
layout["Favorite_button"] = [1092, 2415, 1236, 2559]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_10_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-12/07_icon_Overflow_menu_button.png
try:
    _c7 = get_crop(7, 144, 144)
    canvas.paste(_c7, (1236, 2415), _c7)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 2415, 1380, 2559]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_10_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-12/08_icon_Close_current_screen.png
try:
    _c8 = get_crop(8, 144, 144)
    canvas.paste(_c8, (1248, 96), _c8)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 96, 1392, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_10_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-12/09_icon_LIVE_08A.png
try:
    _c9 = get_crop(9, 1344, 1175)
    canvas.paste(_c9, (48, 676), _c9)
except Exception:
    pass
layout["LIVE_08A"] = [48, 676, 1392, 1851]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_10_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-12/10_icon_Tech.png
try:
    _c10 = get_crop(10, 66, 63)
    canvas.paste(_c10, (308, 1), _c10)
except Exception:
    pass
layout["Tech"] = [308, 1, 374, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_10_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-12/11_icon_5.25.png
try:
    _c11 = get_crop(11, 59, 65)
    canvas.paste(_c11, (181, 0), _c11)
except Exception:
    pass
layout["5.25"] = [181, 0, 240, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_10_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-12/12_icon_5.25.png
try:
    _c12 = get_crop(12, 59, 67)
    canvas.paste(_c12, (115, 0), _c12)
except Exception:
    pass
layout["5.25"] = [115, 0, 174, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_10_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-12/13_icon_Tech.png
try:
    _c13 = get_crop(13, 52, 65)
    canvas.paste(_c13, (247, 0), _c13)
except Exception:
    pass
layout["Tech"] = [247, 0, 299, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_10_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-12/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 97, 62)
    canvas.paste(_c14, (1209, 0), _c14)
except Exception:
    pass
layout["icon_14"] = [1209, 0, 1306, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_10_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-12/15_icon_5.25.png
try:
    _c15 = get_crop(15, 124, 116)
    canvas.paste(_c15, (55, 114), _c15)
except Exception:
    pass
layout["5.25"] = [55, 114, 179, 230]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_10_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-12/16_icon_icon_16.png
try:
    _c16 = get_crop(16, 55, 63)
    canvas.paste(_c16, (1318, 0), _c16)
except Exception:
    pass
layout["icon_16"] = [1318, 0, 1373, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_10_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-12/17_icon_Code_Connect_Your_Tech_Journey.png
try:
    _c17 = get_crop(17, 1344, 917)
    canvas.paste(_c17, (48, 1899), _c17)
except Exception:
    pass
layout["Code_&_Connect:_Your_Tech"] = [48, 1899, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_10_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-12/18_icon_Sun_Apr_28_._I_O0_PM_EDT.png
try:
    _c18 = get_crop(18, 288, 156)
    canvas.paste(_c18, (288, 2804), _c18)
except Exception:
    pass
layout["Sun,_Apr_28_._I:O0_PM_EDT"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_10_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-12/19_icon_Everything_You_Need_To_Know_About_Starti.png
try:
    _c19 = get_crop(19, 1344, 1175)
    canvas.paste(_c19, (48, 676), _c19)
except Exception:
    pass
layout["Everything_You_Need_To_Kn"] = [48, 676, 1392, 1851]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_10_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-12/20_icon_Tech.png
try:
    _c20 = get_crop(20, 1344, 191)
    canvas.paste(_c20, (48, 72), _c20)
except Exception:
    pass
layout["Tech"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_10_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-12/21_icon_Free.png
try:
    _c21 = get_crop(21, 127, 78)
    canvas.paste(_c21, (91, 2592), _c21)
except Exception:
    pass
layout["Free"] = [91, 2592, 218, 2670]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_10_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-12/22_icon_icon_22.png
try:
    _c22 = get_crop(22, 50, 63)
    canvas.paste(_c22, (383, 1), _c22)
except Exception:
    pass
layout["icon_22"] = [383, 1, 433, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_10_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-12/23_icon_Online.png
try:
    _c23 = get_crop(23, 377, 144)
    canvas.paste(_c23, (0, 259), _c23)
except Exception:
    pass
layout["Online"] = [0, 259, 377, 403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_10_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-12/24_icon_Sun_Apr_28_._I_O0_PM_EDT.png
try:
    _c24 = get_crop(24, 288, 156)
    canvas.paste(_c24, (0, 2804), _c24)
except Exception:
    pass
layout["Sun,_Apr_28_._I:O0_PM_EDT"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_10_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-12/25_icon_Code_Connect_Your_Tech_Journey.png
try:
    _c25 = get_crop(25, 288, 156)
    canvas.paste(_c25, (576, 2804), _c25)
except Exception:
    pass
layout["Code_&_Connect:_Your_Tech"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_10_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-12/26_icon_Promoted.png
try:
    _c26 = get_crop(26, 244, 62)
    canvas.paste(_c26, (85, 1746), _c26)
except Exception:
    pass
layout["Promoted"] = [85, 1746, 329, 1808]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_10_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-12/27_icon_Code_Connect_Your_Tech_Journey.png
try:
    _c27 = get_crop(27, 288, 156)
    canvas.paste(_c27, (864, 2804), _c27)
except Exception:
    pass
layout["Code_&_Connect:_Your_Tech"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_10_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-12/28_icon_5.25.png
try:
    _c28 = get_crop(28, 153, 63)
    canvas.paste(_c28, (11, 1), _c28)
except Exception:
    pass
layout["5.25"] = [11, 1, 164, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_10_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-12/29_icon_icon_29.png
try:
    _c29 = get_crop(29, 40, 60)
    canvas.paste(_c29, (1274, 2), _c29)
except Exception:
    pass
layout["icon_29"] = [1274, 2, 1314, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_10_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-12/30_text_10_000_events.png
try:
    _c30 = get_crop(30, 372, 103)
    canvas.paste(_c30, (54, 410), _c30)
except Exception:
    pass
layout["10,000_events"] = [54, 410, 426, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_10_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-12/31_text_Online.png
try:
    _c31 = get_crop(31, 126, 43)
    canvas.paste(_c31, (94, 1689), _c31)
except Exception:
    pass
layout["Online"] = [94, 1689, 220, 1732]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_10_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-12/32_clickable_More.png
try:
    _c32 = get_crop(32, 288, 156)
    canvas.paste(_c32, (1152, 2804), _c32)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]
