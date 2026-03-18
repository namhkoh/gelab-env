# page_id: page_eventbrite_92c22920a83749c994864397a370a984_02
# screenshot: 2024_4_24_16_59_92c22920a83749c994864397a370a984-4.png
# step_index: 2/13
# task: Open Eventbrite. Set the city to "Chicago". Select the "Sports" category and view the recommended events. See the date of the first non-promoted event.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Top status bar
status_h = 72
draw.rectangle((0, 0, 1440, status_h), fill=(200, 200, 200))

# subtle bottom edge of status bar
draw.line((0, status_h-1, 1440, status_h-1), fill=(170, 170, 170), width=1)

# Main divider under header/title area
divider_y = 360
draw.line((48, divider_y, 1440-48, divider_y), fill=(205, 200, 215), width=2)

# Row divider between icon options and browsing section
row_div_y = 650
draw.line((36, row_div_y, 1440-36, row_div_y), fill=(245, 245, 247), width=1)

# Pale blue circular backgrounds for the two option icons ("Nearby" and "Online events")
left_circle_bbox = (76, 476, 164, 564)   # left option background
right_circle_bbox = (336, 476, 424, 564) # right option background
light_blue = (227, 245, 255)
draw.ellipse(left_circle_bbox, fill=light_blue)
draw.ellipse(right_circle_bbox, fill=light_blue)

# Slight inner highlight for those circles (subtle)
highlight_color = (245, 252, 255)
draw.ellipse((left_circle_bbox[0]+6, left_circle_bbox[1]+6, left_circle_bbox[2]-6, left_circle_bbox[3]-6), fill=highlight_color)
draw.ellipse((right_circle_bbox[0]+6, right_circle_bbox[1]+6, right_circle_bbox[2]-6, right_circle_bbox[3]-6), fill=highlight_color)

# Subtle card / grouping background for the "Browsing in" location block
browsing_card = (36, 720, 1404, 920)
draw.rounded_rectangle(browsing_card, radius=12, fill=(250, 250, 252), outline=None)

# Very light left edge guide line under the header (thin)
draw.line((48, divider_y+20, 48, row_div_y-20), fill=(248, 247, 250), width=1)

# Large whitespace area remains white (no drawing) to avoid duplicating content elements

# Small faint footer separator near top of browsing card for visual separation
draw.line((48, 780, 1404-48, 780), fill=(245, 245, 246), width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_02_2024_4_24_16_59_92c22920a83749c994864397a370a984-4/00_icon_icon_0.png
try:
    _c0 = get_crop(0, 103, 105)
    canvas.paste(_c0, (1290, 835), _c0)
except Exception:
    pass
layout["icon_0"] = [1290, 835, 1393, 940]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_02_2024_4_24_16_59_92c22920a83749c994864397a370a984-4/01_icon_icon_1.png
try:
    _c1 = get_crop(1, 60, 62)
    canvas.paste(_c1, (310, 3), _c1)
except Exception:
    pass
layout["icon_1"] = [310, 3, 370, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_02_2024_4_24_16_59_92c22920a83749c994864397a370a984-4/02_icon_4.59.png
try:
    _c2 = get_crop(2, 61, 65)
    canvas.paste(_c2, (179, 1), _c2)
except Exception:
    pass
layout["4.59"] = [179, 1, 240, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_02_2024_4_24_16_59_92c22920a83749c994864397a370a984-4/03_icon_4.59.png
try:
    _c3 = get_crop(3, 59, 65)
    canvas.paste(_c3, (115, 1), _c3)
except Exception:
    pass
layout["4.59"] = [115, 1, 174, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_02_2024_4_24_16_59_92c22920a83749c994864397a370a984-4/04_icon_4.59.png
try:
    _c4 = get_crop(4, 168, 168)
    canvas.paste(_c4, (0, 72), _c4)
except Exception:
    pass
layout["4.59"] = [0, 72, 168, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_02_2024_4_24_16_59_92c22920a83749c994864397a370a984-4/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 49, 58)
    canvas.paste(_c5, (249, 6), _c5)
except Exception:
    pass
layout["icon_5"] = [249, 6, 298, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_02_2024_4_24_16_59_92c22920a83749c994864397a370a984-4/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 60, 65)
    canvas.paste(_c6, (1213, 0), _c6)
except Exception:
    pass
layout["icon_6"] = [1213, 0, 1273, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_02_2024_4_24_16_59_92c22920a83749c994864397a370a984-4/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 45, 59)
    canvas.paste(_c7, (1323, 3), _c7)
except Exception:
    pass
layout["icon_7"] = [1323, 3, 1368, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_02_2024_4_24_16_59_92c22920a83749c994864397a370a984-4/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 43, 63)
    canvas.paste(_c8, (1270, 1), _c8)
except Exception:
    pass
layout["icon_8"] = [1270, 1, 1313, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_02_2024_4_24_16_59_92c22920a83749c994864397a370a984-4/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 52, 66)
    canvas.paste(_c9, (382, 1), _c9)
except Exception:
    pass
layout["icon_9"] = [382, 1, 434, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_02_2024_4_24_16_59_92c22920a83749c994864397a370a984-4/10_icon_4.59.png
try:
    _c10 = get_crop(10, 93, 63)
    canvas.paste(_c10, (14, 2), _c10)
except Exception:
    pass
layout["4.59"] = [14, 2, 107, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_02_2024_4_24_16_59_92c22920a83749c994864397a370a984-4/11_icon_Nearby.png
try:
    _c11 = get_crop(11, 415, 114)
    canvas.paste(_c11, (48, 465), _c11)
except Exception:
    pass
layout["Nearby"] = [48, 465, 463, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_02_2024_4_24_16_59_92c22920a83749c994864397a370a984-4/12_text_Find_events_in..png
try:
    _c12 = get_crop(12, 1344, 129)
    canvas.paste(_c12, (48, 264), _c12)
except Exception:
    pass
layout["Find_events_in._"] = [48, 264, 1392, 393]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_02_2024_4_24_16_59_92c22920a83749c994864397a370a984-4/13_text_Online_events.png
try:
    _c13 = get_crop(13, 452, 114)
    canvas.paste(_c13, (511, 465), _c13)
except Exception:
    pass
layout["Online_events"] = [511, 465, 963, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_02_2024_4_24_16_59_92c22920a83749c994864397a370a984-4/14_text_Virtual_attendance.png
try:
    _c14 = get_crop(14, 452, 114)
    canvas.paste(_c14, (511, 465), _c14)
except Exception:
    pass
layout["Virtual_attendance"] = [511, 465, 963, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_02_2024_4_24_16_59_92c22920a83749c994864397a370a984-4/15_text_Browsing_in.png
try:
    _c15 = get_crop(15, 228, 55)
    canvas.paste(_c15, (44, 742), _c15)
except Exception:
    pass
layout["Browsing_in"] = [44, 742, 272, 797]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_02_2024_4_24_16_59_92c22920a83749c994864397a370a984-4/16_text_Los_Angeles.png
try:
    _c16 = get_crop(16, 1440, 138)
    canvas.paste(_c16, (0, 816), _c16)
except Exception:
    pass
layout["Los_Angeles"] = [0, 816, 1440, 954]
