# page_id: page_eventbrite_92c22920a83749c994864397a370a984_03
# screenshot: 2024_4_24_16_59_92c22920a83749c994864397a370a984-5.png
# step_index: 3/13
# task: Open Eventbrite. Set the city to "Chicago". Select the "Sports" category and view the recommended events. See the date of the first non-promoted event.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw status bar background (top area)
draw.rectangle([(0, 0), (1440, 72)], fill=(189, 189, 189))

# Draw header/toolbar area (just under status bar)
draw.rectangle([(0, 72), (1440, 190)], fill=(255, 255, 255))
# subtle divider under the toolbar
draw.line([(32, 190), (1408, 190)], fill=(230, 230, 230), width=1)

# Prominent blue underline under the "Find events in..." input area
# (spans the content width, aligned to the left margin)
blue_line_y = 336
draw.rectangle([(48, blue_line_y - 3), (1392, blue_line_y + 3)], fill=(59, 92, 255))

# Light separators between sections
draw.line([(32, 420), (1408, 420)], fill=(241, 242, 245), width=1)
draw.line([(32, 620), (1408, 620)], fill=(246, 246, 248), width=1)

# Subtle background card for the "Browsing in" area (light, rounded)
card_bbox = (24, 720, 1416, 920)
try:
    draw.rounded_rectangle(card_bbox, radius=14, fill=(255, 255, 255), outline=(240, 241, 246), width=1)
except Exception:
    # fallback if rounded_rectangle not available
    draw.rectangle(card_bbox, fill=(255, 255, 255), outline=(240, 241, 246))

# Keep the remainder of the canvas white (no additional content drawn)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_03_2024_4_24_16_59_92c22920a83749c994864397a370a984-5/00_icon_icon_0.png
try:
    _c0 = get_crop(0, 103, 105)
    canvas.paste(_c0, (1290, 835), _c0)
except Exception:
    pass
layout["icon_0"] = [1290, 835, 1393, 940]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_03_2024_4_24_16_59_92c22920a83749c994864397a370a984-5/01_icon_4.59.png
try:
    _c1 = get_crop(1, 61, 65)
    canvas.paste(_c1, (179, 1), _c1)
except Exception:
    pass
layout["4.59"] = [179, 1, 240, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_03_2024_4_24_16_59_92c22920a83749c994864397a370a984-5/02_icon_4.59.png
try:
    _c2 = get_crop(2, 59, 65)
    canvas.paste(_c2, (115, 1), _c2)
except Exception:
    pass
layout["4.59"] = [115, 1, 174, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_03_2024_4_24_16_59_92c22920a83749c994864397a370a984-5/03_icon_icon_3.png
try:
    _c3 = get_crop(3, 61, 62)
    canvas.paste(_c3, (309, 3), _c3)
except Exception:
    pass
layout["icon_3"] = [309, 3, 370, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_03_2024_4_24_16_59_92c22920a83749c994864397a370a984-5/04_icon_4.59.png
try:
    _c4 = get_crop(4, 168, 168)
    canvas.paste(_c4, (0, 72), _c4)
except Exception:
    pass
layout["4.59"] = [0, 72, 168, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_03_2024_4_24_16_59_92c22920a83749c994864397a370a984-5/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 50, 59)
    canvas.paste(_c5, (248, 5), _c5)
except Exception:
    pass
layout["icon_5"] = [248, 5, 298, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_03_2024_4_24_16_59_92c22920a83749c994864397a370a984-5/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 64, 64)
    canvas.paste(_c6, (1212, 1), _c6)
except Exception:
    pass
layout["icon_6"] = [1212, 1, 1276, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_03_2024_4_24_16_59_92c22920a83749c994864397a370a984-5/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 46, 58)
    canvas.paste(_c7, (1322, 3), _c7)
except Exception:
    pass
layout["icon_7"] = [1322, 3, 1368, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_03_2024_4_24_16_59_92c22920a83749c994864397a370a984-5/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 48, 62)
    canvas.paste(_c8, (1265, 1), _c8)
except Exception:
    pass
layout["icon_8"] = [1265, 1, 1313, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_03_2024_4_24_16_59_92c22920a83749c994864397a370a984-5/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 52, 65)
    canvas.paste(_c9, (382, 1), _c9)
except Exception:
    pass
layout["icon_9"] = [382, 1, 434, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_03_2024_4_24_16_59_92c22920a83749c994864397a370a984-5/10_icon_4.59.png
try:
    _c10 = get_crop(10, 93, 63)
    canvas.paste(_c10, (14, 2), _c10)
except Exception:
    pass
layout["4.59"] = [14, 2, 107, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_03_2024_4_24_16_59_92c22920a83749c994864397a370a984-5/11_icon_Nearby.png
try:
    _c11 = get_crop(11, 415, 114)
    canvas.paste(_c11, (48, 465), _c11)
except Exception:
    pass
layout["Nearby"] = [48, 465, 463, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_03_2024_4_24_16_59_92c22920a83749c994864397a370a984-5/12_text_Find_events_in..png
try:
    _c12 = get_crop(12, 1344, 129)
    canvas.paste(_c12, (48, 264), _c12)
except Exception:
    pass
layout["Find_events_in._"] = [48, 264, 1392, 393]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_03_2024_4_24_16_59_92c22920a83749c994864397a370a984-5/13_text_Online_events.png
try:
    _c13 = get_crop(13, 452, 114)
    canvas.paste(_c13, (511, 465), _c13)
except Exception:
    pass
layout["Online_events"] = [511, 465, 963, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_03_2024_4_24_16_59_92c22920a83749c994864397a370a984-5/14_text_Virtual_attendance.png
try:
    _c14 = get_crop(14, 452, 114)
    canvas.paste(_c14, (511, 465), _c14)
except Exception:
    pass
layout["Virtual_attendance"] = [511, 465, 963, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_03_2024_4_24_16_59_92c22920a83749c994864397a370a984-5/15_text_Browsing_in.png
try:
    _c15 = get_crop(15, 228, 55)
    canvas.paste(_c15, (44, 742), _c15)
except Exception:
    pass
layout["Browsing_in"] = [44, 742, 272, 797]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_03_2024_4_24_16_59_92c22920a83749c994864397a370a984-5/16_text_Los_Angeles.png
try:
    _c16 = get_crop(16, 1440, 138)
    canvas.paste(_c16, (0, 816), _c16)
except Exception:
    pass
layout["Los_Angeles"] = [0, 816, 1440, 954]
