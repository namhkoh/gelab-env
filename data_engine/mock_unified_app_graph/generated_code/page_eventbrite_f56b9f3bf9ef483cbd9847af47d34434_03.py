# page_id: page_eventbrite_f56b9f3bf9ef483cbd9847af47d34434_03
# screenshot: 2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-5.png
# step_index: 3/8
# task: Open Eventbrite. Look up "Gardening" events. Filter by events happening this week. Select the first event from the results. Follow the organizer and where is the event?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Paint overall background
draw.rectangle([0, 0, 1440, 2960], fill=(255, 255, 255))

# Status bar (top ~50px) - darker grey strip for time/signal area
status_bar_height = 64
draw.rectangle([0, 0, 1440, status_bar_height], fill=(189, 189, 189))

# Header area (search toolbar background) directly below status bar
header_top = status_bar_height
header_bottom = 184
draw.rectangle([0, header_top, 1440, header_bottom], fill=(255, 255, 255))

# Thin blue underline under the search field (approx where the search underline sits)
underline_y = 168
underline_left = 48
underline_right = 1392
underline_thickness = 6
draw.rectangle([underline_left, underline_y - underline_thickness//2,
                underline_right, underline_y + underline_thickness//2],
               fill=(37, 82, 255))

# Subtle shadow / divider under header
draw.rectangle([0, header_bottom, 1440, header_bottom + 2], fill=(230, 230, 230))

# Section card background for the "Popular" list (light subtle card)
card_left = 36
card_top = 256
card_right = 1404
card_bottom = 952
card_radius = 14
# very light off-white to separate from canvas white
draw.rounded_rectangle([card_left, card_top, card_right, card_bottom],
                       radius=card_radius, fill=(250, 250, 252), outline=None)

# Section title area is inside the card; add a subtle divider under the title region
title_divider_y = card_top + 60
draw.line([(card_left + 12, title_divider_y), (card_right - 12, title_divider_y)],
          fill=(235, 235, 240), width=1)

# Separator lines between the list items (faint)
# Based on detected item y positions, place separators between rows
separators = [
    (card_top + 182),  # between first and second item
    (card_top + 302),
    (card_top + 422),
    (card_top + 542)
]
for y in separators:
    draw.line([(card_left + 12, y), (card_right - 12, y)], fill=(240, 240, 245), width=1)

# A faint bounding line around the card (very subtle)
draw.rounded_rectangle([card_left, card_top, card_right, card_bottom],
                       radius=card_radius, outline=(240, 240, 246), width=1)

# Large empty content area remains white (no drawings to avoid overlapping pasted text/icons)

# Bottom navigation bar background and top divider
nav_top = 2804
nav_bottom = 2960
draw.rectangle([0, nav_top, 1440, nav_bottom], fill=(255, 255, 255))
draw.line([(0, nav_top), (1440, nav_top)], fill=(224, 224, 230), width=2)

# Very light vertical edge guides at sides (subtle)
edge_strip_width = 12
draw.rectangle([0, 0, edge_strip_width, 2960], fill=(255, 255, 255))
draw.rectangle([1440 - edge_strip_width, 0, 1440, 2960], fill=(255, 255, 255))

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_03_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-5/00_icon_Gardening.png
try:
    _c0 = get_crop(0, 1344, 191)
    canvas.paste(_c0, (48, 72), _c0)
except Exception:
    pass
layout["Gardening]"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_03_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-5/01_icon_gardening_class.png
try:
    _c1 = get_crop(1, 104, 103)
    canvas.paste(_c1, (29, 883), _c1)
except Exception:
    pass
layout["gardening_class"] = [29, 883, 133, 986]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_03_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-5/02_icon_icon_2.png
try:
    _c2 = get_crop(2, 55, 58)
    canvas.paste(_c2, (313, 5), _c2)
except Exception:
    pass
layout["icon_2"] = [313, 5, 368, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_03_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-5/03_icon_icon_3.png
try:
    _c3 = get_crop(3, 42, 54)
    canvas.paste(_c3, (254, 7), _c3)
except Exception:
    pass
layout["icon_3"] = [254, 7, 296, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_03_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-5/04_icon_5.09.png
try:
    _c4 = get_crop(4, 51, 60)
    canvas.paste(_c4, (185, 3), _c4)
except Exception:
    pass
layout["5.09"] = [185, 3, 236, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_03_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-5/05_icon_5.09.png
try:
    _c5 = get_crop(5, 102, 97)
    canvas.paste(_c5, (67, 122), _c5)
except Exception:
    pass
layout["5.09"] = [67, 122, 169, 219]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_03_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-5/06_icon_5.09.png
try:
    _c6 = get_crop(6, 53, 60)
    canvas.paste(_c6, (117, 4), _c6)
except Exception:
    pass
layout["5.09"] = [117, 4, 170, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_03_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-5/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 95, 100)
    canvas.paste(_c7, (34, 766), _c7)
except Exception:
    pass
layout["icon_7"] = [34, 766, 129, 866]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_03_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-5/08_icon_Tickets.png
try:
    _c8 = get_crop(8, 288, 156)
    canvas.paste(_c8, (864, 2804), _c8)
except Exception:
    pass
layout["Tickets"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_03_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-5/09_icon_Cancel.png
try:
    _c9 = get_crop(9, 47, 61)
    canvas.paste(_c9, (1322, 2), _c9)
except Exception:
    pass
layout["Cancel"] = [1322, 2, 1369, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_03_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-5/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 95, 101)
    canvas.paste(_c10, (34, 644), _c10)
except Exception:
    pass
layout["icon_10"] = [34, 644, 129, 745]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_03_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-5/11_icon_Search_events.png
try:
    _c11 = get_crop(11, 288, 156)
    canvas.paste(_c11, (288, 2804), _c11)
except Exception:
    pass
layout["Search_events"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_03_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-5/12_icon_Cancel.png
try:
    _c12 = get_crop(12, 96, 64)
    canvas.paste(_c12, (1214, 0), _c12)
except Exception:
    pass
layout["Cancel"] = [1214, 0, 1310, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_03_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-5/13_icon_More.png
try:
    _c13 = get_crop(13, 288, 156)
    canvas.paste(_c13, (1152, 2804), _c13)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_03_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-5/14_icon_Cancel.png
try:
    _c14 = get_crop(14, 149, 144)
    canvas.paste(_c14, (1243, 97), _c14)
except Exception:
    pass
layout["Cancel"] = [1243, 97, 1392, 241]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_03_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-5/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 95, 98)
    canvas.paste(_c15, (33, 527), _c15)
except Exception:
    pass
layout["icon_15"] = [33, 527, 128, 625]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_03_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-5/16_icon_Cancel.png
try:
    _c16 = get_crop(16, 144, 144)
    canvas.paste(_c16, (1099, 96), _c16)
except Exception:
    pass
layout["Cancel"] = [1099, 96, 1243, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_03_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-5/17_icon_Favorites.png
try:
    _c17 = get_crop(17, 288, 156)
    canvas.paste(_c17, (576, 2804), _c17)
except Exception:
    pass
layout["Favorites"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_03_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-5/18_icon_Home.png
try:
    _c18 = get_crop(18, 288, 156)
    canvas.paste(_c18, (0, 2804), _c18)
except Exception:
    pass
layout["Home"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_03_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-5/19_icon_Gardening.png
try:
    _c19 = get_crop(19, 41, 60)
    canvas.paste(_c19, (387, 5), _c19)
except Exception:
    pass
layout["Gardening]"] = [387, 5, 428, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_03_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-5/20_icon_Popular.png
try:
    _c20 = get_crop(20, 98, 108)
    canvas.paste(_c20, (35, 402), _c20)
except Exception:
    pass
layout["Popular"] = [35, 402, 133, 510]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_03_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-5/21_text_5.09.png
try:
    _c21 = get_crop(21, 91, 45)
    canvas.paste(_c21, (20, 15), _c21)
except Exception:
    pass
layout["5.09"] = [20, 15, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_03_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-5/22_text_Popular.png
try:
    _c22 = get_crop(22, 224, 78)
    canvas.paste(_c22, (41, 298), _c22)
except Exception:
    pass
layout["Popular"] = [41, 298, 265, 376]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_03_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-5/23_text_gardening_classes.png
try:
    _c23 = get_crop(23, 1344, 120)
    canvas.paste(_c23, (48, 378), _c23)
except Exception:
    pass
layout["gardening_classes"] = [48, 378, 1392, 498]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_03_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-5/24_text_vegetable_gardening.png
try:
    _c24 = get_crop(24, 1344, 120)
    canvas.paste(_c24, (48, 498), _c24)
except Exception:
    pass
layout["vegetable_gardening"] = [48, 498, 1392, 618]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_03_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-5/25_text_container_gardening.png
try:
    _c25 = get_crop(25, 1344, 120)
    canvas.paste(_c25, (48, 618), _c25)
except Exception:
    pass
layout["container_gardening"] = [48, 618, 1392, 738]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_03_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-5/26_text_hydroponic_gardening.png
try:
    _c26 = get_crop(26, 1344, 120)
    canvas.paste(_c26, (48, 738), _c26)
except Exception:
    pass
layout["hydroponic_gardening"] = [48, 738, 1392, 858]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_03_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-5/27_text_gardening_class.png
try:
    _c27 = get_crop(27, 1344, 144)
    canvas.paste(_c27, (48, 858), _c27)
except Exception:
    pass
layout["gardening_class"] = [48, 858, 1392, 1002]
