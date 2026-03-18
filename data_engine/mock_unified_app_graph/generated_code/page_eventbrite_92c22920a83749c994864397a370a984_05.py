# page_id: page_eventbrite_92c22920a83749c994864397a370a984_05
# screenshot: 2024_4_24_16_59_92c22920a83749c994864397a370a984-7.png
# step_index: 5/13
# task: Open Eventbrite. Set the city to "Chicago". Select the "Sports" category and view the recommended events. See the date of the first non-promoted event.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw overall background
draw.rectangle((0, 0, 1440, 2960), fill="#ffffff")

# Status bar (top area)
status_h = 72
draw.rectangle((0, 0, 1440, status_h), fill="#d0d0d0")

# Header area (search / input region)
header_top = status_h
header_bottom = 220
draw.rectangle((0, header_top, 1440, header_bottom), fill="#ffffff")
# Blue underline for the search input
underline_y = 200
draw.line((48, underline_y, 1392, underline_y), fill="#2b56ff", width=4)

# Subtle divider under the header / icons area (separates header from list)
divider_y = 640
draw.line((32, divider_y, 1408, divider_y), fill="#efeff2", width=1)

# "Found locations" list area: draw alternating row backgrounds and separators
row_height = 132
list_start_y = 840
rows = 10  # enough rows to cover the detected list items
for i in range(rows):
    y0 = list_start_y + i * row_height
    y1 = y0 + row_height
    # alternate very subtle banding (do not draw anything that looks like text/icons)
    if i % 2 == 1:
        draw.rectangle((0, y0, 1440, y1), fill="#fbfbfd")
    # top separator line for each row
    draw.line((32, y0, 1408, y0), fill="#f0f0f2", width=1)
# final bottom separator after the last row
draw.line((32, list_start_y + rows * row_height, 1408, list_start_y + rows * row_height), fill="#f0f0f2", width=1)

# Light shadow band behind the main content area for depth (subtle)
draw.rectangle((0, header_bottom, 1440, divider_y), fill="#ffffff")
draw.line((0, header_bottom, 1440, header_bottom), fill="#eeeeef", width=1)

# Footer safe area background (very subtle)
footer_h = 48
draw.rectangle((0, 2960 - footer_h, 1440, 2960), fill="#ffffff")

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_05_2024_4_24_16_59_92c22920a83749c994864397a370a984-7/00_icon_icon_0.png
try:
    _c0 = get_crop(0, 48, 69)
    canvas.paste(_c0, (1154, 0), _c0)
except Exception:
    pass
layout["icon_0"] = [1154, 0, 1202, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_05_2024_4_24_16_59_92c22920a83749c994864397a370a984-7/01_icon_icon_1.png
try:
    _c1 = get_crop(1, 98, 65)
    canvas.paste(_c1, (1214, 0), _c1)
except Exception:
    pass
layout["icon_1"] = [1214, 0, 1312, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_05_2024_4_24_16_59_92c22920a83749c994864397a370a984-7/02_icon_5.00.png
try:
    _c2 = get_crop(2, 62, 64)
    canvas.paste(_c2, (179, 1), _c2)
except Exception:
    pass
layout["5.00"] = [179, 1, 241, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_05_2024_4_24_16_59_92c22920a83749c994864397a370a984-7/03_icon_icon_3.png
try:
    _c3 = get_crop(3, 63, 63)
    canvas.paste(_c3, (308, 2), _c3)
except Exception:
    pass
layout["icon_3"] = [308, 2, 371, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_05_2024_4_24_16_59_92c22920a83749c994864397a370a984-7/04_icon_5.00.png
try:
    _c4 = get_crop(4, 168, 168)
    canvas.paste(_c4, (0, 72), _c4)
except Exception:
    pass
layout["5.00"] = [0, 72, 168, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_05_2024_4_24_16_59_92c22920a83749c994864397a370a984-7/05_icon_5.00.png
try:
    _c5 = get_crop(5, 60, 65)
    canvas.paste(_c5, (115, 1), _c5)
except Exception:
    pass
layout["5.00"] = [115, 1, 175, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_05_2024_4_24_16_59_92c22920a83749c994864397a370a984-7/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 52, 64)
    canvas.paste(_c6, (1319, 0), _c6)
except Exception:
    pass
layout["icon_6"] = [1319, 0, 1371, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_05_2024_4_24_16_59_92c22920a83749c994864397a370a984-7/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 51, 58)
    canvas.paste(_c7, (247, 5), _c7)
except Exception:
    pass
layout["icon_7"] = [247, 5, 298, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_05_2024_4_24_16_59_92c22920a83749c994864397a370a984-7/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 85, 96)
    canvas.paste(_c8, (1310, 286), _c8)
except Exception:
    pass
layout["icon_8"] = [1310, 286, 1395, 382]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_05_2024_4_24_16_59_92c22920a83749c994864397a370a984-7/09_icon_District_of_Columbia.png
try:
    _c9 = get_crop(9, 1440, 132)
    canvas.paste(_c9, (0, 1740), _c9)
except Exception:
    pass
layout["District_of_Columbia"] = [0, 1740, 1440, 1872]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_05_2024_4_24_16_59_92c22920a83749c994864397a370a984-7/10_icon_San_Francisco.png
try:
    _c10 = get_crop(10, 1440, 132)
    canvas.paste(_c10, (0, 840), _c10)
except Exception:
    pass
layout["San_Francisco"] = [0, 840, 1440, 972]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_05_2024_4_24_16_59_92c22920a83749c994864397a370a984-7/11_icon_Chicago.png
try:
    _c11 = get_crop(11, 1440, 132)
    canvas.paste(_c11, (0, 1380), _c11)
except Exception:
    pass
layout["Chicago"] = [0, 1380, 1440, 1512]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_05_2024_4_24_16_59_92c22920a83749c994864397a370a984-7/12_icon_United_Kingdom.png
try:
    _c12 = get_crop(12, 1440, 132)
    canvas.paste(_c12, (0, 2100), _c12)
except Exception:
    pass
layout["United_Kingdom"] = [0, 2100, 1440, 2232]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_05_2024_4_24_16_59_92c22920a83749c994864397a370a984-7/13_icon_District_of_Columbia.png
try:
    _c13 = get_crop(13, 1440, 132)
    canvas.paste(_c13, (0, 1560), _c13)
except Exception:
    pass
layout["District_of_Columbia"] = [0, 1560, 1440, 1692]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_05_2024_4_24_16_59_92c22920a83749c994864397a370a984-7/14_icon_Los_Angeles.png
try:
    _c14 = get_crop(14, 1440, 132)
    canvas.paste(_c14, (0, 1020), _c14)
except Exception:
    pass
layout["Los_Angeles"] = [0, 1020, 1440, 1152]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_05_2024_4_24_16_59_92c22920a83749c994864397a370a984-7/15_icon_Philadelphia.png
try:
    _c15 = get_crop(15, 1440, 132)
    canvas.paste(_c15, (0, 1920), _c15)
except Exception:
    pass
layout["Philadelphia"] = [0, 1920, 1440, 2052]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_05_2024_4_24_16_59_92c22920a83749c994864397a370a984-7/16_icon_Miami.png
try:
    _c16 = get_crop(16, 1440, 132)
    canvas.paste(_c16, (0, 1200), _c16)
except Exception:
    pass
layout["Miami"] = [0, 1200, 1440, 1332]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_05_2024_4_24_16_59_92c22920a83749c994864397a370a984-7/17_icon_icon_17.png
try:
    _c17 = get_crop(17, 53, 65)
    canvas.paste(_c17, (382, 1), _c17)
except Exception:
    pass
layout["icon_17"] = [382, 1, 435, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_05_2024_4_24_16_59_92c22920a83749c994864397a370a984-7/18_text_5.00.png
try:
    _c18 = get_crop(18, 91, 45)
    canvas.paste(_c18, (20, 15), _c18)
except Exception:
    pass
layout["5.00"] = [20, 15, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_05_2024_4_24_16_59_92c22920a83749c994864397a370a984-7/19_text_Chicago.png
try:
    _c19 = get_crop(19, 1344, 129)
    canvas.paste(_c19, (48, 264), _c19)
except Exception:
    pass
layout["Chicago"] = [48, 264, 1392, 393]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_05_2024_4_24_16_59_92c22920a83749c994864397a370a984-7/20_text_Nearby.png
try:
    _c20 = get_crop(20, 415, 114)
    canvas.paste(_c20, (48, 465), _c20)
except Exception:
    pass
layout["Nearby"] = [48, 465, 463, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_05_2024_4_24_16_59_92c22920a83749c994864397a370a984-7/21_text_Online_events.png
try:
    _c21 = get_crop(21, 452, 114)
    canvas.paste(_c21, (511, 465), _c21)
except Exception:
    pass
layout["Online_events"] = [511, 465, 963, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_05_2024_4_24_16_59_92c22920a83749c994864397a370a984-7/22_text_Current_location.png
try:
    _c22 = get_crop(22, 415, 114)
    canvas.paste(_c22, (48, 465), _c22)
except Exception:
    pass
layout["Current_location"] = [48, 465, 463, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_05_2024_4_24_16_59_92c22920a83749c994864397a370a984-7/23_text_Virtual_attendance.png
try:
    _c23 = get_crop(23, 452, 114)
    canvas.paste(_c23, (511, 465), _c23)
except Exception:
    pass
layout["Virtual_attendance"] = [511, 465, 963, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_05_2024_4_24_16_59_92c22920a83749c994864397a370a984-7/24_text_Found_locations.png
try:
    _c24 = get_crop(24, 311, 50)
    canvas.paste(_c24, (44, 740), _c24)
except Exception:
    pass
layout["Found_locations"] = [44, 740, 355, 790]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_05_2024_4_24_16_59_92c22920a83749c994864397a370a984-7/25_text_New_York.png
try:
    _c25 = get_crop(25, 212, 55)
    canvas.paste(_c25, (44, 2288), _c25)
except Exception:
    pass
layout["New_York"] = [44, 2288, 256, 2343]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_05_2024_4_24_16_59_92c22920a83749c994864397a370a984-7/26_text_New_York.png
try:
    _c26 = get_crop(26, 154, 38)
    canvas.paste(_c26, (47, 2353), _c26)
except Exception:
    pass
layout["New_York"] = [47, 2353, 201, 2391]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_05_2024_4_24_16_59_92c22920a83749c994864397a370a984-7/27_text_Atlanta.png
try:
    _c27 = get_crop(27, 163, 52)
    canvas.paste(_c27, (44, 2468), _c27)
except Exception:
    pass
layout["Atlanta"] = [44, 2468, 207, 2520]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_05_2024_4_24_16_59_92c22920a83749c994864397a370a984-7/28_text_Georgia.png
try:
    _c28 = get_crop(28, 133, 43)
    canvas.paste(_c28, (45, 2533), _c28)
except Exception:
    pass
layout["Georgia"] = [45, 2533, 178, 2576]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_05_2024_4_24_16_59_92c22920a83749c994864397a370a984-7/29_clickable_New_York.png
try:
    _c29 = get_crop(29, 1440, 132)
    canvas.paste(_c29, (0, 2280), _c29)
except Exception:
    pass
layout["New_York"] = [0, 2280, 1440, 2412]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_05_2024_4_24_16_59_92c22920a83749c994864397a370a984-7/30_clickable_Atlanta.png
try:
    _c30 = get_crop(30, 1440, 132)
    canvas.paste(_c30, (0, 2460), _c30)
except Exception:
    pass
layout["Atlanta"] = [0, 2460, 1440, 2592]
