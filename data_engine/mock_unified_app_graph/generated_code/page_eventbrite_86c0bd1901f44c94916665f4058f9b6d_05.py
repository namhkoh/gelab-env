# page_id: page_eventbrite_86c0bd1901f44c94916665f4058f9b6d_05
# screenshot: 2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-7.png
# step_index: 5/11
# task: Open Eventbrite. Set the city to Los Angeles. Select the 'Food & Drink' category. What's the date of the first event that is not promoted?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw background and UI structure for the mobile UI page.
# Uses provided variables: canvas (PIL Image) and draw (ImageDraw)

# Fill full background with white (dominant page color)
draw.rectangle([(0, 0), (1440, 2960)], fill="#FFFFFF")

# Status bar area (top)
status_h = 72
draw.rectangle([(0, 0), (1440, status_h)], fill="#D9D9D9")
# subtle bottom line of status bar
draw.line([(0, status_h), (1440, status_h)], fill="#CFCFCF", width=1)

# Header / search area below status bar
header_top = status_h
header_bottom = 240
draw.rectangle([(0, header_top), (1440, header_bottom)], fill="#FFFFFF")
# prominent blue underline for the search input area
blue_y = 200
draw.line([(48, blue_y), (1392, blue_y)], fill="#2D56FF", width=6)

# Sub-header area (where the "Nearby" and "Online events" controls live)
subheader_top = header_bottom
subheader_bottom = 520
draw.rectangle([(0, subheader_top), (1440, subheader_bottom)], fill="#FFFFFF")
# very light divider under subheader
draw.line([(48, subheader_bottom), (1392, subheader_bottom)], fill="#EFEFF2", width=1)

# Section title area ("Found locations" region) background (keeps page consistent)
found_top = 720
found_bottom = 780
draw.rectangle([(0, found_top), (1440, found_bottom)], fill="#FFFFFF")
# small divider above found locations area
draw.line([(48, found_top), (1392, found_top)], fill="#F2F2F4", width=1)

# Draw subtle separators between the full-width location rows.
# Use left/right padding to avoid overlapping the far edges like native UI.
separator_x1 = 48
separator_x2 = 1392
separator_ys = [840, 1020, 1200, 1380, 1560, 1740, 1920, 2100, 2280, 2460]
for y in separator_ys:
    draw.line([(separator_x1, y), (separator_x2, y)], fill="#F3F3F6", width=1)

# Add a faint right edge shadow line to give subtle depth to the content column
draw.line([(0, 2960), (0, 0)], fill="#FFFFFF", width=0)  # no-op to keep structure consistent

# Footer safe area background (bottom)
footer_top = 2860
draw.rectangle([(0, footer_top), (1440, 2960)], fill="#FFFFFF")

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_05_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-7/00_icon_icon_0.png
try:
    _c0 = get_crop(0, 48, 69)
    canvas.paste(_c0, (1154, 0), _c0)
except Exception:
    pass
layout["icon_0"] = [1154, 0, 1202, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_05_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-7/01_icon_icon_1.png
try:
    _c1 = get_crop(1, 98, 65)
    canvas.paste(_c1, (1214, 0), _c1)
except Exception:
    pass
layout["icon_1"] = [1214, 0, 1312, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_05_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-7/02_icon_7.13.png
try:
    _c2 = get_crop(2, 61, 63)
    canvas.paste(_c2, (179, 1), _c2)
except Exception:
    pass
layout["7.13"] = [179, 1, 240, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_05_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-7/03_icon_7.13.png
try:
    _c3 = get_crop(3, 59, 63)
    canvas.paste(_c3, (115, 1), _c3)
except Exception:
    pass
layout["7.13"] = [115, 1, 174, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_05_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-7/04_icon_icon_4.png
try:
    _c4 = get_crop(4, 61, 61)
    canvas.paste(_c4, (309, 3), _c4)
except Exception:
    pass
layout["icon_4"] = [309, 3, 370, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_05_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-7/05_icon_7.13.png
try:
    _c5 = get_crop(5, 168, 168)
    canvas.paste(_c5, (0, 72), _c5)
except Exception:
    pass
layout["7.13"] = [0, 72, 168, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_05_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-7/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 50, 58)
    canvas.paste(_c6, (248, 5), _c6)
except Exception:
    pass
layout["icon_6"] = [248, 5, 298, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_05_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-7/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 51, 63)
    canvas.paste(_c7, (1320, 0), _c7)
except Exception:
    pass
layout["icon_7"] = [1320, 0, 1371, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_05_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-7/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 86, 98)
    canvas.paste(_c8, (1310, 285), _c8)
except Exception:
    pass
layout["icon_8"] = [1310, 285, 1396, 383]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_05_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-7/09_icon_San_Francisco.png
try:
    _c9 = get_crop(9, 1440, 132)
    canvas.paste(_c9, (0, 840), _c9)
except Exception:
    pass
layout["San_Francisco"] = [0, 840, 1440, 972]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_05_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-7/10_icon_District_of_Columbia.png
try:
    _c10 = get_crop(10, 1440, 132)
    canvas.paste(_c10, (0, 1740), _c10)
except Exception:
    pass
layout["District_of_Columbia"] = [0, 1740, 1440, 1872]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_05_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-7/11_icon_Chicago.png
try:
    _c11 = get_crop(11, 1440, 132)
    canvas.paste(_c11, (0, 1380), _c11)
except Exception:
    pass
layout["Chicago"] = [0, 1380, 1440, 1512]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_05_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-7/12_icon_Los_Angeles.png
try:
    _c12 = get_crop(12, 1440, 132)
    canvas.paste(_c12, (0, 1020), _c12)
except Exception:
    pass
layout["Los_Angeles"] = [0, 1020, 1440, 1152]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_05_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-7/13_icon_United_Kingdom.png
try:
    _c13 = get_crop(13, 1440, 132)
    canvas.paste(_c13, (0, 2100), _c13)
except Exception:
    pass
layout["United_Kingdom"] = [0, 2100, 1440, 2232]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_05_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-7/14_icon_District_of_Columbia.png
try:
    _c14 = get_crop(14, 1440, 132)
    canvas.paste(_c14, (0, 1560), _c14)
except Exception:
    pass
layout["District_of_Columbia"] = [0, 1560, 1440, 1692]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_05_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-7/15_icon_Philadelphia.png
try:
    _c15 = get_crop(15, 1440, 132)
    canvas.paste(_c15, (0, 1920), _c15)
except Exception:
    pass
layout["Philadelphia"] = [0, 1920, 1440, 2052]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_05_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-7/16_icon_Miami.png
try:
    _c16 = get_crop(16, 1440, 132)
    canvas.paste(_c16, (0, 1200), _c16)
except Exception:
    pass
layout["Miami"] = [0, 1200, 1440, 1332]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_05_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-7/17_icon_7.13.png
try:
    _c17 = get_crop(17, 94, 64)
    canvas.paste(_c17, (13, 1), _c17)
except Exception:
    pass
layout["7.13"] = [13, 1, 107, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_05_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-7/18_icon_Nearby.png
try:
    _c18 = get_crop(18, 415, 114)
    canvas.paste(_c18, (48, 465), _c18)
except Exception:
    pass
layout["Nearby"] = [48, 465, 463, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_05_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-7/19_icon_icon_19.png
try:
    _c19 = get_crop(19, 52, 64)
    canvas.paste(_c19, (383, 1), _c19)
except Exception:
    pass
layout["icon_19"] = [383, 1, 435, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_05_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-7/20_text_Los_Angeles.png
try:
    _c20 = get_crop(20, 1344, 129)
    canvas.paste(_c20, (48, 264), _c20)
except Exception:
    pass
layout["Los_Angeles"] = [48, 264, 1392, 393]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_05_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-7/21_text_Online_events.png
try:
    _c21 = get_crop(21, 452, 114)
    canvas.paste(_c21, (511, 465), _c21)
except Exception:
    pass
layout["Online_events"] = [511, 465, 963, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_05_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-7/22_text_Virtual_attendance.png
try:
    _c22 = get_crop(22, 452, 114)
    canvas.paste(_c22, (511, 465), _c22)
except Exception:
    pass
layout["Virtual_attendance"] = [511, 465, 963, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_05_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-7/23_text_Found_locations.png
try:
    _c23 = get_crop(23, 311, 50)
    canvas.paste(_c23, (44, 740), _c23)
except Exception:
    pass
layout["Found_locations"] = [44, 740, 355, 790]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_05_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-7/24_text_New_York.png
try:
    _c24 = get_crop(24, 212, 55)
    canvas.paste(_c24, (44, 2288), _c24)
except Exception:
    pass
layout["New_York"] = [44, 2288, 256, 2343]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_05_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-7/25_text_New_York.png
try:
    _c25 = get_crop(25, 154, 38)
    canvas.paste(_c25, (47, 2353), _c25)
except Exception:
    pass
layout["New_York"] = [47, 2353, 201, 2391]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_05_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-7/26_text_Atlanta.png
try:
    _c26 = get_crop(26, 163, 52)
    canvas.paste(_c26, (44, 2468), _c26)
except Exception:
    pass
layout["Atlanta"] = [44, 2468, 207, 2520]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_05_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-7/27_text_Georgia.png
try:
    _c27 = get_crop(27, 133, 43)
    canvas.paste(_c27, (45, 2533), _c27)
except Exception:
    pass
layout["Georgia"] = [45, 2533, 178, 2576]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_05_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-7/28_clickable_New_York.png
try:
    _c28 = get_crop(28, 1440, 132)
    canvas.paste(_c28, (0, 2280), _c28)
except Exception:
    pass
layout["New_York"] = [0, 2280, 1440, 2412]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_05_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-7/29_clickable_Atlanta.png
try:
    _c29 = get_crop(29, 1440, 132)
    canvas.paste(_c29, (0, 2460), _c29)
except Exception:
    pass
layout["Atlanta"] = [0, 2460, 1440, 2592]
