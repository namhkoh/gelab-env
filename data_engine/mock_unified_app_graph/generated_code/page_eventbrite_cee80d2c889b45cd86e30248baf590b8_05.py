# page_id: page_eventbrite_cee80d2c889b45cd86e30248baf590b8_05
# screenshot: 2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-7.png
# step_index: 5/13
# task: Open Eventbrite. Search Food & Drink party event in New York. Select the first one that is not promoted. Follow the organizer. Share to google keep notes.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw background and structural elements for the UI page using provided canvas & draw

# Status bar (top area)
status_h = 72
draw.rectangle([(0, 0), (1440, status_h)], fill="#F5F6F8")  # subtle light grey status bar

# Header underline (blue thin rule under the page title area)
title_top_y = 264
title_h = 129
title_bottom = title_top_y + title_h
underline_y = title_bottom + 8
left_margin = 48
right_margin = 1440 - 48
draw.line([(left_margin, underline_y), (right_margin, underline_y)], fill="#2E5BF7", width=4)
# slight lighter hairline under the blue underline for subtle separation
draw.line([(left_margin, underline_y + 4), (right_margin, underline_y + 4)], fill="#E9EDFF", width=1)

# Large rounded card background for the "Found locations" list area
card_x0 = 20
card_x1 = 1440 - 20
card_y0 = 720
card_y1 = 2760
draw.rounded_rectangle([(card_x0, card_y0), (card_x1, card_y1)], radius=18, fill="#FFFFFF", outline="#EFEFF2", width=1)

# Subtle shadow band below card top to separate header from list (thin gradient-ish band simulated)
shadow_top = card_y0 + 2
for i, alpha in enumerate([0.06, 0.04, 0.03]):
    y = shadow_top + i * 3
    grey = (int(240 - i*2), int(240 - i*2), int(244 - i*2))
    draw.line([(card_x0 + 2, y), (card_x1 - 2, y)], fill=grey, width=1)

# Separator lines between list items (light hairlines)
item_tops = [840, 1020, 1200, 1380, 1560, 1740, 1920, 2100, 2280, 2460]
sep_left = 44
sep_right = 1396
sep_color = "#F0F0F4"
for t in item_tops:
    # place separator roughly at bottom of each list row
    y = t + 128
    if card_y0 < y < card_y1:
        draw.line([(sep_left, y), (sep_right, y)], fill=sep_color, width=1)

# Thin divider above the found-locations heading
heading_y = 740
draw.line([(left_margin, heading_y - 12), (right_margin, heading_y - 12)], fill="#FBFBFD", width=1)

# Subtle left alignment guide area (visual composition, not content)
left_column_x = 48
draw.line([(left_column_x, card_y0 + 12), (left_column_x, card_y1 - 12)], fill="#F7F7FA", width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_05_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-7/00_icon_icon_0.png
try:
    _c0 = get_crop(0, 46, 68)
    canvas.paste(_c0, (1155, 0), _c0)
except Exception:
    pass
layout["icon_0"] = [1155, 0, 1201, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_05_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-7/01_icon_icon_1.png
try:
    _c1 = get_crop(1, 97, 65)
    canvas.paste(_c1, (1215, 0), _c1)
except Exception:
    pass
layout["icon_1"] = [1215, 0, 1312, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_05_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-7/02_icon_9.44.png
try:
    _c2 = get_crop(2, 59, 63)
    canvas.paste(_c2, (178, 1), _c2)
except Exception:
    pass
layout["9.44"] = [178, 1, 237, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_05_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-7/03_icon_9.44.png
try:
    _c3 = get_crop(3, 54, 64)
    canvas.paste(_c3, (114, 1), _c3)
except Exception:
    pass
layout["9.44"] = [114, 1, 168, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_05_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-7/04_icon_9.44.png
try:
    _c4 = get_crop(4, 168, 168)
    canvas.paste(_c4, (0, 72), _c4)
except Exception:
    pass
layout["9.44"] = [0, 72, 168, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_05_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-7/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 84, 94)
    canvas.paste(_c5, (1311, 287), _c5)
except Exception:
    pass
layout["icon_5"] = [1311, 287, 1395, 381]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_05_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-7/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 50, 62)
    canvas.paste(_c6, (1320, 1), _c6)
except Exception:
    pass
layout["icon_6"] = [1320, 1, 1370, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_05_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-7/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 53, 63)
    canvas.paste(_c7, (315, 1), _c7)
except Exception:
    pass
layout["icon_7"] = [315, 1, 368, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_05_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-7/08_icon_San_Francisco.png
try:
    _c8 = get_crop(8, 1440, 132)
    canvas.paste(_c8, (0, 840), _c8)
except Exception:
    pass
layout["San_Francisco"] = [0, 840, 1440, 972]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_05_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-7/09_icon_District_of_Columbia.png
try:
    _c9 = get_crop(9, 1440, 132)
    canvas.paste(_c9, (0, 1740), _c9)
except Exception:
    pass
layout["District_of_Columbia"] = [0, 1740, 1440, 1872]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_05_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-7/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 55, 62)
    canvas.paste(_c10, (247, 2), _c10)
except Exception:
    pass
layout["icon_10"] = [247, 2, 302, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_05_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-7/11_icon_Chicago.png
try:
    _c11 = get_crop(11, 1440, 132)
    canvas.paste(_c11, (0, 1380), _c11)
except Exception:
    pass
layout["Chicago"] = [0, 1380, 1440, 1512]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_05_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-7/12_icon_Los_Angeles.png
try:
    _c12 = get_crop(12, 1440, 132)
    canvas.paste(_c12, (0, 1020), _c12)
except Exception:
    pass
layout["Los_Angeles"] = [0, 1020, 1440, 1152]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_05_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-7/13_icon_United_Kingdom.png
try:
    _c13 = get_crop(13, 1440, 132)
    canvas.paste(_c13, (0, 2100), _c13)
except Exception:
    pass
layout["United_Kingdom"] = [0, 2100, 1440, 2232]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_05_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-7/14_icon_Miami.png
try:
    _c14 = get_crop(14, 1440, 132)
    canvas.paste(_c14, (0, 1200), _c14)
except Exception:
    pass
layout["Miami"] = [0, 1200, 1440, 1332]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_05_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-7/15_icon_District_of_Columbia.png
try:
    _c15 = get_crop(15, 1440, 132)
    canvas.paste(_c15, (0, 1560), _c15)
except Exception:
    pass
layout["District_of_Columbia"] = [0, 1560, 1440, 1692]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_05_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-7/16_icon_Philadelphia.png
try:
    _c16 = get_crop(16, 1440, 132)
    canvas.paste(_c16, (0, 1920), _c16)
except Exception:
    pass
layout["Philadelphia"] = [0, 1920, 1440, 2052]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_05_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-7/17_icon_icon_17.png
try:
    _c17 = get_crop(17, 51, 63)
    canvas.paste(_c17, (381, 0), _c17)
except Exception:
    pass
layout["icon_17"] = [381, 0, 432, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_05_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-7/18_icon_District_of_Columbia.png
try:
    _c18 = get_crop(18, 1440, 132)
    canvas.paste(_c18, (0, 1560), _c18)
except Exception:
    pass
layout["District_of_Columbia"] = [0, 1560, 1440, 1692]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_05_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-7/19_text_9.44.png
try:
    _c19 = get_crop(19, 94, 43)
    canvas.paste(_c19, (20, 15), _c19)
except Exception:
    pass
layout["9.44"] = [20, 15, 114, 58]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_05_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-7/20_text_New_York.png
try:
    _c20 = get_crop(20, 1344, 129)
    canvas.paste(_c20, (48, 264), _c20)
except Exception:
    pass
layout["New_York"] = [48, 264, 1392, 393]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_05_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-7/21_text_Nearby.png
try:
    _c21 = get_crop(21, 415, 114)
    canvas.paste(_c21, (48, 465), _c21)
except Exception:
    pass
layout["Nearby"] = [48, 465, 463, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_05_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-7/22_text_Online_events.png
try:
    _c22 = get_crop(22, 452, 114)
    canvas.paste(_c22, (511, 465), _c22)
except Exception:
    pass
layout["Online_events"] = [511, 465, 963, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_05_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-7/23_text_Current_location.png
try:
    _c23 = get_crop(23, 415, 114)
    canvas.paste(_c23, (48, 465), _c23)
except Exception:
    pass
layout["Current_location"] = [48, 465, 463, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_05_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-7/24_text_Virtual_attendance.png
try:
    _c24 = get_crop(24, 452, 114)
    canvas.paste(_c24, (511, 465), _c24)
except Exception:
    pass
layout["Virtual_attendance"] = [511, 465, 963, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_05_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-7/25_text_Found_locations.png
try:
    _c25 = get_crop(25, 311, 50)
    canvas.paste(_c25, (44, 740), _c25)
except Exception:
    pass
layout["Found_locations"] = [44, 740, 355, 790]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_05_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-7/26_text_New_York.png
try:
    _c26 = get_crop(26, 212, 55)
    canvas.paste(_c26, (44, 2288), _c26)
except Exception:
    pass
layout["New_York"] = [44, 2288, 256, 2343]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_05_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-7/27_text_New_York.png
try:
    _c27 = get_crop(27, 154, 38)
    canvas.paste(_c27, (47, 2353), _c27)
except Exception:
    pass
layout["New_York"] = [47, 2353, 201, 2391]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_05_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-7/28_text_Atlanta.png
try:
    _c28 = get_crop(28, 163, 52)
    canvas.paste(_c28, (44, 2468), _c28)
except Exception:
    pass
layout["Atlanta"] = [44, 2468, 207, 2520]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_05_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-7/29_text_Georgia.png
try:
    _c29 = get_crop(29, 133, 43)
    canvas.paste(_c29, (45, 2533), _c29)
except Exception:
    pass
layout["Georgia"] = [45, 2533, 178, 2576]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_05_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-7/30_clickable_New_York.png
try:
    _c30 = get_crop(30, 1440, 132)
    canvas.paste(_c30, (0, 2280), _c30)
except Exception:
    pass
layout["New_York"] = [0, 2280, 1440, 2412]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_05_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-7/31_clickable_Atlanta.png
try:
    _c31 = get_crop(31, 1440, 132)
    canvas.paste(_c31, (0, 2460), _c31)
except Exception:
    pass
layout["Atlanta"] = [0, 2460, 1440, 2592]
