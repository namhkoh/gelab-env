# page_id: page_eventbrite_5f8371b476c64aeeb04a6d8281b87f4d_05
# screenshot: 2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-7.png
# step_index: 5/7
# task: Open Eventbrite. Search Science & Tech event. Select the first one that is not promoted. If it is free, add it to Favorites. If it is not free, record its price in Google Keep Notes.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Background fill
draw.rectangle([(0, 0), (1440, 2960)], fill="#f5f6f8")

# Status bar area (top)
status_h = 96
draw.rectangle([(0, 0), (1440, status_h)], fill="#d0d0d0")

# Subtle divider under status bar
draw.line([(24, status_h), (1416, status_h)], fill="#cfcfcf", width=1)

# Header / title area background
header_top = status_h
header_bottom = 220
draw.rectangle([(0, header_top), (1440, header_bottom)], fill="#ffffff")
# Header bottom divider
draw.line([(24, header_bottom), (1416, header_bottom)], fill="#e2e2e6", width=2)

# Location / filter row background (behind chips)
loc_row_top = 240
loc_row_bottom = 320
draw.rectangle([(0, loc_row_top), (1440, loc_row_bottom)], fill="#ffffff")
# Subtle separator under location/filter row
draw.line([(24, loc_row_bottom), (1416, loc_row_bottom)], fill="#efeff2", width=1)

# First event card container (rounded)
card1_margin_x = 36
card1_top = 500
card1_width = 1440 - 2 * card1_margin_x
card1_height = 1080
card1_bbox = [card1_margin_x, card1_top, card1_margin_x + card1_width, card1_top + card1_height]
draw.rounded_rectangle(card1_bbox, radius=24, fill="#ffffff", outline="#e6e6ea", width=1)

# subtle shadow line under first card
draw.line([(card1_bbox[0]+8, card1_bbox[3]+6), (card1_bbox[2]-8, card1_bbox[3]+6)], fill="#efeff4", width=2)

# Second event/ad card container (rounded)
card2_top = 1580
card2_margin_x = 36
card2_width = 1440 - 2 * card2_margin_x
card2_height = 1080
card2_bbox = [card2_margin_x, card2_top, card2_margin_x + card2_width, card2_top + card2_height]
draw.rounded_rectangle(card2_bbox, radius=24, fill="#ffffff", outline="#e6e6ea", width=1)

# subtle shadow line under second card
draw.line([(card2_bbox[0]+8, card2_bbox[3]+6), (card2_bbox[2]-8, card2_bbox[3]+6)], fill="#efeff4", width=2)

# Content separators between event cards and list items
sep_y = card1_bbox[3] + 40
draw.line([(24, sep_y), (1416, sep_y)], fill="#f0f0f3", width=1)

sep_y2 = card2_bbox[3] + 40
draw.line([(24, sep_y2), (1416, sep_y2)], fill="#f0f0f3", width=1)

# Bottom navigation bar background
nav_top = 2804
nav_bottom = 2960
draw.rectangle([(0, nav_top), (1440, nav_bottom)], fill="#ffffff")
# Top divider for nav bar
draw.line([(0, nav_top), (1440, nav_top)], fill="#e6e6ea", width=1)

# Add subtle rounded background blocks behind each nav item (three or five evenly spaced)
nav_item_w = 288
for i in range(5):
    x0 = i * nav_item_w
    x1 = x0 + nav_item_w
    # Slight inset and very light rounded background to anchor icons
    inset = 16
    draw.rounded_rectangle(
        [x0 + inset, nav_top + 16, x1 - inset, nav_bottom - 16],
        radius=14,
        fill="#ffffff",
        outline=None
    )

# Light left edge fade (subtle)
draw.rectangle([(0, 2200), (24, 2960)], fill="#f7f7f9")

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_05_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-7/00_icon_Music.png
try:
    _c0 = get_crop(0, 196, 111)
    canvas.paste(_c0, (829, 405), _c0)
except Exception:
    pass
layout["Music"] = [829, 405, 1025, 516]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_05_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-7/01_icon_Business.png
try:
    _c1 = get_crop(1, 251, 111)
    canvas.paste(_c1, (1029, 406), _c1)
except Exception:
    pass
layout["Business"] = [1029, 406, 1280, 517]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_05_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-7/02_icon_Anytime.png
try:
    _c2 = get_crop(2, 408, 112)
    canvas.paste(_c2, (418, 406), _c2)
except Exception:
    pass
layout["Anytime"] = [418, 406, 826, 518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_05_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-7/03_icon_Filters.png
try:
    _c3 = get_crop(3, 434, 144)
    canvas.paste(_c3, (0, 259), _c3)
except Exception:
    pass
layout["Filters"] = [0, 259, 434, 403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_05_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-7/04_icon_Foo.png
try:
    _c4 = get_crop(4, 150, 109)
    canvas.paste(_c4, (1283, 406), _c4)
except Exception:
    pass
layout["Foo"] = [1283, 406, 1433, 515]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_05_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-7/05_icon_Breakout.png
try:
    _c5 = get_crop(5, 144, 144)
    canvas.paste(_c5, (1092, 886), _c5)
except Exception:
    pass
layout["Breakout"] = [1092, 886, 1236, 1030]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_05_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-7/06_icon_Khoury-prospigtotontheastoctLIdu.png
try:
    _c6 = get_crop(6, 144, 144)
    canvas.paste(_c6, (1092, 2128), _c6)
except Exception:
    pass
layout["Khoury-prospigtotontheast"] = [1092, 2128, 1236, 2272]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_05_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-7/07_icon_Breakout.png
try:
    _c7 = get_crop(7, 144, 144)
    canvas.paste(_c7, (1236, 886), _c7)
except Exception:
    pass
layout["Breakout"] = [1236, 886, 1380, 1030]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_05_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-7/08_icon_Khoury-prospigtotontheastoctLIdu.png
try:
    _c8 = get_crop(8, 144, 144)
    canvas.paste(_c8, (1236, 2128), _c8)
except Exception:
    pass
layout["Khoury-prospigtotontheast"] = [1236, 2128, 1380, 2272]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_05_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-7/09_icon_IN_Conversation_Women_in_STEAM.png
try:
    _c9 = get_crop(9, 1344, 1039)
    canvas.paste(_c9, (48, 525), _c9)
except Exception:
    pass
layout["IN_Conversation:_Women_in"] = [48, 525, 1392, 1564]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_05_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-7/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 57, 62)
    canvas.paste(_c10, (246, 1), _c10)
except Exception:
    pass
layout["icon_10"] = [246, 1, 303, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_05_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-7/11_icon_9.37.png
try:
    _c11 = get_crop(11, 112, 113)
    canvas.paste(_c11, (61, 114), _c11)
except Exception:
    pass
layout["9.37"] = [61, 114, 173, 227]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_05_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-7/12_icon_Close_current_screen.png
try:
    _c12 = get_crop(12, 144, 144)
    canvas.paste(_c12, (1248, 96), _c12)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 96, 1392, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_05_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-7/13_icon_9.37.png
try:
    _c13 = get_crop(13, 54, 62)
    canvas.paste(_c13, (183, 0), _c13)
except Exception:
    pass
layout["9.37"] = [183, 0, 237, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_05_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-7/14_icon_New_York.png
try:
    _c14 = get_crop(14, 434, 144)
    canvas.paste(_c14, (0, 259), _c14)
except Exception:
    pass
layout["New_York"] = [0, 259, 434, 403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_05_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-7/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 57, 63)
    canvas.paste(_c15, (313, 1), _c15)
except Exception:
    pass
layout["icon_15"] = [313, 1, 370, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_05_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-7/16_icon_9.37.png
try:
    _c16 = get_crop(16, 57, 63)
    canvas.paste(_c16, (113, 1), _c16)
except Exception:
    pass
layout["9.37"] = [113, 1, 170, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_05_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-7/17_icon_inuschooLykhovay-coLLrQ.png
try:
    _c17 = get_crop(17, 1344, 1029)
    canvas.paste(_c17, (48, 1612), _c17)
except Exception:
    pass
layout["inuschooLykhovay-coLLrQ="] = [48, 1612, 1392, 2641]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_05_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-7/18_icon_icon_18.png
try:
    _c18 = get_crop(18, 101, 61)
    canvas.paste(_c18, (1208, 0), _c18)
except Exception:
    pass
layout["icon_18"] = [1208, 0, 1309, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_05_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-7/19_icon_Tickets.png
try:
    _c19 = get_crop(19, 288, 156)
    canvas.paste(_c19, (864, 2804), _c19)
except Exception:
    pass
layout["Tickets"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_05_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-7/20_icon_icon_20.png
try:
    _c20 = get_crop(20, 57, 61)
    canvas.paste(_c20, (1317, 0), _c20)
except Exception:
    pass
layout["icon_20"] = [1317, 0, 1374, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_05_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-7/21_icon_Science_Tech.png
try:
    _c21 = get_crop(21, 1344, 191)
    canvas.paste(_c21, (48, 72), _c21)
except Exception:
    pass
layout["Science_&_Tech"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_05_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-7/22_icon_Science_Tech.png
try:
    _c22 = get_crop(22, 46, 62)
    canvas.paste(_c22, (384, 1), _c22)
except Exception:
    pass
layout["Science_&_Tech"] = [384, 1, 430, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_05_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-7/23_icon_More.png
try:
    _c23 = get_crop(23, 288, 156)
    canvas.paste(_c23, (1152, 2804), _c23)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_05_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-7/24_icon_Suites_Fifth_Avenue.png
try:
    _c24 = get_crop(24, 288, 156)
    canvas.paste(_c24, (288, 2804), _c24)
except Exception:
    pass
layout["Suites_Fifth_Avenue"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_05_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-7/25_icon_Favorites.png
try:
    _c25 = get_crop(25, 288, 156)
    canvas.paste(_c25, (576, 2804), _c25)
except Exception:
    pass
layout["Favorites"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_05_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-7/26_icon_Free.png
try:
    _c26 = get_crop(26, 47, 56)
    canvas.paste(_c26, (143, 2233), _c26)
except Exception:
    pass
layout["Free"] = [143, 2233, 190, 2289]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_05_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-7/27_icon_IN_Conversation_Women_in_STEAM.png
try:
    _c27 = get_crop(27, 1344, 1039)
    canvas.paste(_c27, (48, 525), _c27)
except Exception:
    pass
layout["IN_Conversation:_Women_in"] = [48, 525, 1392, 1564]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_05_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-7/28_icon_Home.png
try:
    _c28 = get_crop(28, 288, 156)
    canvas.paste(_c28, (0, 2804), _c28)
except Exception:
    pass
layout["Home"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_05_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-7/29_icon_Jay.png
try:
    _c29 = get_crop(29, 288, 156)
    canvas.paste(_c29, (0, 2804), _c29)
except Exception:
    pass
layout["Jay"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_05_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-7/30_icon_9.37.png
try:
    _c30 = get_crop(30, 113, 62)
    canvas.paste(_c30, (14, 1), _c30)
except Exception:
    pass
layout["9.37"] = [14, 1, 127, 63]
