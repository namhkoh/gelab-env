# page_id: page_eventbrite_92c22920a83749c994864397a370a984_08
# screenshot: 2024_4_24_16_59_92c22920a83749c994864397a370a984-10.png
# step_index: 8/13
# task: Open Eventbrite. Set the city to "Chicago". Select the "Sports" category and view the recommended events. See the date of the first non-promoted event.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw background and UI structure for the Filters screen.
# Assumes `canvas` (1440x2960 RGB) and `draw` (ImageDraw) are provided.

# Full white background
draw.rectangle([(0, 0), (1440, 2960)], fill=(255, 255, 255))

# Top status bar (light grey)
status_h = 84
draw.rectangle([(0, 0), (1440, status_h)], fill=(224, 224, 224))

# Header area under status bar (keeps white but we add a subtle divider)
header_top = status_h
header_bottom = 160
draw.rectangle([(0, header_top), (1440, header_bottom)], fill=(255, 255, 255))
# subtle bottom divider under header
draw.line([(24, header_bottom), (1416, header_bottom)], fill=(235, 233, 239), width=2)

# Light separators between major sections (subtle)
sep_color = (245, 244, 247)
seps = [520, 964, 1410, 1613, 1936]  # approximate separators between groups
for y in seps:
    draw.line([(36, y), (1404, y)], fill=sep_color, width=1)

# "Sort by" rounded container (base background)
sort_y1 = 2024
sort_h = 144
sort_x1 = 48
sort_x2 = sort_x1 + 1344  # matches detected Apply filters width region
sort_y2 = sort_y1 + sort_h
# outer rounded container
draw.rounded_rectangle(
    [(sort_x1, sort_y1), (sort_x2, sort_y2)],
    radius=18,
    fill=(247, 246, 249),
    outline=(214, 210, 218),
    width=2
)
# left selected tab background (subtle white inset)
left_tab_x1 = 54
left_tab_x2 = left_tab_x1 + 660
draw.rounded_rectangle(
    [(left_tab_x1, sort_y1 + 4), (left_tab_x2, sort_y2 - 4)],
    radius=14,
    fill=(255, 255, 255),
    outline=(210, 206, 212),
    width=2
)
# right tab area (slightly more muted)
right_tab_x1 = left_tab_x2 + 12
right_tab_x2 = right_tab_x1 + 660
draw.rounded_rectangle(
    [(right_tab_x1, sort_y1 + 4), (right_tab_x2, sort_y2 - 4)],
    radius=14,
    fill=(241, 239, 244),
    outline=(214, 210, 218),
    width=1
)
# small dividing shadow between tabs
divider_x = left_tab_x2 + 6
draw.line([(divider_x, sort_y1 + 12), (divider_x, sort_y2 - 12)], fill=(230, 227, 233), width=3)

# "Only free events" area: subtle placeholder background for the switch row (no switch)
price_section_y = 1560
draw.rectangle([(36, price_section_y), (1404, price_section_y + 120)], fill=(255, 255, 255))  # keep white
# faint divider under the price area
draw.line([(36, price_section_y + 120), (1404, price_section_y + 120)], fill=(245, 244, 247), width=1)

# Bottom "Apply filters" button background (rounded, white with border)
btn_y1 = 2768
btn_h = 144
btn_x1 = 48
btn_x2 = btn_x1 + 1344
btn_y2 = btn_y1 + btn_h
# subtle shadow behind the button
shadow_color = (235, 233, 240)
draw.rectangle([(btn_x1 + 6, btn_y1 + 6), (btn_x2 + 6, btn_y2 + 6)], fill=shadow_color)
# main button
draw.rounded_rectangle(
    [(btn_x1, btn_y1), (btn_x2, btn_y2)],
    radius=12,
    fill=(255, 255, 255),
    outline=(150, 146, 158),
    width=4
)

# Additional subtle structure: faint large content area bands to guide layout
# top content band under header
draw.rectangle([(36, 180), (1404, 420)], fill=(255, 255, 255))
# languages/event types bands (just background spacing guides)
draw.rectangle([(36, 700), (1404, 980)], fill=(255, 255, 255))
draw.rectangle([(36, 1160), (1404, 1440)], fill=(255, 255, 255))

# Final thin margin outline to frame the screen (very subtle)
draw.rectangle([(8, 8), (1432, 2952)], outline=(248, 247, 249), width=2)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_08_2024_4_24_16_59_92c22920a83749c994864397a370a984-10/00_icon_Food_Drink.png
try:
    _c0 = get_crop(0, 312, 144)
    canvas.paste(_c0, (512, 383), _c0)
except Exception:
    pass
layout["Food_&_Drink"] = [512, 383, 824, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_08_2024_4_24_16_59_92c22920a83749c994864397a370a984-10/01_icon_Music.png
try:
    _c1 = get_crop(1, 187, 135)
    canvas.paste(_c1, (36, 383), _c1)
except Exception:
    pass
layout["Music"] = [36, 383, 223, 518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_08_2024_4_24_16_59_92c22920a83749c994864397a370a984-10/02_icon_Community.png
try:
    _c2 = get_crop(2, 294, 144)
    canvas.paste(_c2, (848, 383), _c2)
except Exception:
    pass
layout["Community"] = [848, 383, 1142, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_08_2024_4_24_16_59_92c22920a83749c994864397a370a984-10/03_icon_French.png
try:
    _c3 = get_crop(3, 205, 144)
    canvas.paste(_c3, (768, 1275), _c3)
except Exception:
    pass
layout["French"] = [768, 1275, 973, 1419]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_08_2024_4_24_16_59_92c22920a83749c994864397a370a984-10/04_icon_Business.png
try:
    _c4 = get_crop(4, 241, 135)
    canvas.paste(_c4, (247, 383), _c4)
except Exception:
    pass
layout["Business"] = [247, 383, 488, 518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_08_2024_4_24_16_59_92c22920a83749c994864397a370a984-10/05_icon_Spanish.png
try:
    _c5 = get_crop(5, 225, 144)
    canvas.paste(_c5, (519, 1275), _c5)
except Exception:
    pass
layout["Spanish"] = [519, 1275, 744, 1419]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_08_2024_4_24_16_59_92c22920a83749c994864397a370a984-10/06_icon_Expo.png
try:
    _c6 = get_crop(6, 167, 144)
    canvas.paste(_c6, (614, 829), _c6)
except Exception:
    pass
layout["Expo"] = [614, 829, 781, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_08_2024_4_24_16_59_92c22920a83749c994864397a370a984-10/07_icon_Italian.png
try:
    _c7 = get_crop(7, 191, 144)
    canvas.paste(_c7, (997, 1275), _c7)
except Exception:
    pass
layout["Italian"] = [997, 1275, 1188, 1419]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_08_2024_4_24_16_59_92c22920a83749c994864397a370a984-10/08_icon_Seminar.png
try:
    _c8 = get_crop(8, 232, 144)
    canvas.paste(_c8, (358, 829), _c8)
except Exception:
    pass
layout["Seminar"] = [358, 829, 590, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_08_2024_4_24_16_59_92c22920a83749c994864397a370a984-10/09_icon_Arts.png
try:
    _c9 = get_crop(9, 152, 144)
    canvas.paste(_c9, (1166, 383), _c9)
except Exception:
    pass
layout["Arts"] = [1166, 383, 1318, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_08_2024_4_24_16_59_92c22920a83749c994864397a370a984-10/10_icon_Convention.png
try:
    _c10 = get_crop(10, 293, 144)
    canvas.paste(_c10, (805, 829), _c10)
except Exception:
    pass
layout["Convention"] = [805, 829, 1098, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_08_2024_4_24_16_59_92c22920a83749c994864397a370a984-10/11_icon_Festival.png
try:
    _c11 = get_crop(11, 219, 144)
    canvas.paste(_c11, (1122, 829), _c11)
except Exception:
    pass
layout["Festival"] = [1122, 829, 1341, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_08_2024_4_24_16_59_92c22920a83749c994864397a370a984-10/12_icon_German.png
try:
    _c12 = get_crop(12, 225, 135)
    canvas.paste(_c12, (270, 1275), _c12)
except Exception:
    pass
layout["German"] = [270, 1275, 495, 1410]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_08_2024_4_24_16_59_92c22920a83749c994864397a370a984-10/13_icon_English.png
try:
    _c13 = get_crop(13, 210, 135)
    canvas.paste(_c13, (36, 1275), _c13)
except Exception:
    pass
layout["English"] = [36, 1275, 246, 1410]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_08_2024_4_24_16_59_92c22920a83749c994864397a370a984-10/14_icon_Conference.png
try:
    _c14 = get_crop(14, 298, 135)
    canvas.paste(_c14, (36, 829), _c14)
except Exception:
    pass
layout["Conference"] = [36, 829, 334, 964]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_08_2024_4_24_16_59_92c22920a83749c994864397a370a984-10/15_icon_Date.png
try:
    _c15 = get_crop(15, 660, 144)
    canvas.paste(_c15, (726, 2024), _c15)
except Exception:
    pass
layout["Date"] = [726, 2024, 1386, 2168]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_08_2024_4_24_16_59_92c22920a83749c994864397a370a984-10/16_icon_Relevance.png
try:
    _c16 = get_crop(16, 660, 144)
    canvas.paste(_c16, (54, 2024), _c16)
except Exception:
    pass
layout["Relevance"] = [54, 2024, 714, 2168]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_08_2024_4_24_16_59_92c22920a83749c994864397a370a984-10/17_icon_Clear_all.png
try:
    _c17 = get_crop(17, 51, 70)
    canvas.paste(_c17, (1153, 1), _c17)
except Exception:
    pass
layout["Clear_all"] = [1153, 1, 1204, 71]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_08_2024_4_24_16_59_92c22920a83749c994864397a370a984-10/18_icon_Apply_filters.png
try:
    _c18 = get_crop(18, 1344, 144)
    canvas.paste(_c18, (48, 2768), _c18)
except Exception:
    pass
layout["Apply_filters"] = [48, 2768, 1392, 2912]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_08_2024_4_24_16_59_92c22920a83749c994864397a370a984-10/19_icon_Clear_all.png
try:
    _c19 = get_crop(19, 100, 70)
    canvas.paste(_c19, (1211, 0), _c19)
except Exception:
    pass
layout["Clear_all"] = [1211, 0, 1311, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_08_2024_4_24_16_59_92c22920a83749c994864397a370a984-10/20_icon_5.00.png
try:
    _c20 = get_crop(20, 144, 144)
    canvas.paste(_c20, (12, 72), _c20)
except Exception:
    pass
layout["5.00"] = [12, 72, 156, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_08_2024_4_24_16_59_92c22920a83749c994864397a370a984-10/21_icon_5.00.png
try:
    _c21 = get_crop(21, 61, 64)
    canvas.paste(_c21, (179, 2), _c21)
except Exception:
    pass
layout["5.00"] = [179, 2, 240, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_08_2024_4_24_16_59_92c22920a83749c994864397a370a984-10/22_icon_icon_22.png
try:
    _c22 = get_crop(22, 66, 62)
    canvas.paste(_c22, (307, 3), _c22)
except Exception:
    pass
layout["icon_22"] = [307, 3, 373, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_08_2024_4_24_16_59_92c22920a83749c994864397a370a984-10/23_icon_5.00.png
try:
    _c23 = get_crop(23, 64, 66)
    canvas.paste(_c23, (112, 1), _c23)
except Exception:
    pass
layout["5.00"] = [112, 1, 176, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_08_2024_4_24_16_59_92c22920a83749c994864397a370a984-10/24_icon_Clear_all.png
try:
    _c24 = get_crop(24, 52, 69)
    canvas.paste(_c24, (1320, 0), _c24)
except Exception:
    pass
layout["Clear_all"] = [1320, 0, 1372, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_08_2024_4_24_16_59_92c22920a83749c994864397a370a984-10/25_icon_icon_25.png
try:
    _c25 = get_crop(25, 51, 61)
    canvas.paste(_c25, (248, 3), _c25)
except Exception:
    pass
layout["icon_25"] = [248, 3, 299, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_08_2024_4_24_16_59_92c22920a83749c994864397a370a984-10/26_icon_Clear_all.png
try:
    _c26 = get_crop(26, 178, 144)
    canvas.paste(_c26, (1214, 72), _c26)
except Exception:
    pass
layout["Clear_all"] = [1214, 72, 1392, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_08_2024_4_24_16_59_92c22920a83749c994864397a370a984-10/27_icon_Toggle_to_filter_only_free_events.png
try:
    _c27 = get_crop(27, 144, 144)
    canvas.paste(_c27, (1248, 1729), _c27)
except Exception:
    pass
layout["Toggle_to_filter_only_fre"] = [1248, 1729, 1392, 1873]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_08_2024_4_24_16_59_92c22920a83749c994864397a370a984-10/28_text_5.00.png
try:
    _c28 = get_crop(28, 91, 45)
    canvas.paste(_c28, (20, 15), _c28)
except Exception:
    pass
layout["5.00"] = [20, 15, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_08_2024_4_24_16_59_92c22920a83749c994864397a370a984-10/29_text_Filters.png
try:
    _c29 = get_crop(29, 180, 66)
    canvas.paste(_c29, (631, 116), _c29)
except Exception:
    pass
layout["Filters"] = [631, 116, 811, 182]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_08_2024_4_24_16_59_92c22920a83749c994864397a370a984-10/30_text_Categories.png
try:
    _c30 = get_crop(30, 187, 135)
    canvas.paste(_c30, (36, 383), _c30)
except Exception:
    pass
layout["Categories"] = [36, 383, 223, 518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_08_2024_4_24_16_59_92c22920a83749c994864397a370a984-10/31_text_Show_all_categories.png
try:
    _c31 = get_crop(31, 516, 144)
    canvas.paste(_c31, (0, 518), _c31)
except Exception:
    pass
layout["Show_all_categories"] = [0, 518, 516, 662]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_08_2024_4_24_16_59_92c22920a83749c994864397a370a984-10/32_text_Event_type.png
try:
    _c32 = get_crop(32, 298, 135)
    canvas.paste(_c32, (36, 829), _c32)
except Exception:
    pass
layout["Event_type"] = [36, 829, 334, 964]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_08_2024_4_24_16_59_92c22920a83749c994864397a370a984-10/33_text_Show_all_event_types.png
try:
    _c33 = get_crop(33, 535, 144)
    canvas.paste(_c33, (0, 964), _c33)
except Exception:
    pass
layout["Show_all_event_types"] = [0, 964, 535, 1108]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_08_2024_4_24_16_59_92c22920a83749c994864397a370a984-10/34_text_Languages.png
try:
    _c34 = get_crop(34, 210, 135)
    canvas.paste(_c34, (36, 1275), _c34)
except Exception:
    pass
layout["Languages"] = [36, 1275, 246, 1410]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_08_2024_4_24_16_59_92c22920a83749c994864397a370a984-10/35_text_Show_all_languages.png
try:
    _c35 = get_crop(35, 511, 144)
    canvas.paste(_c35, (0, 1410), _c35)
except Exception:
    pass
layout["Show_all_languages"] = [0, 1410, 511, 1554]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_08_2024_4_24_16_59_92c22920a83749c994864397a370a984-10/36_text_Price.png
try:
    _c36 = get_crop(36, 149, 63)
    canvas.paste(_c36, (45, 1613), _c36)
except Exception:
    pass
layout["Price"] = [45, 1613, 194, 1676]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_08_2024_4_24_16_59_92c22920a83749c994864397a370a984-10/37_text_Only_free_events.png
try:
    _c37 = get_crop(37, 660, 144)
    canvas.paste(_c37, (54, 2024), _c37)
except Exception:
    pass
layout["Only_free_events"] = [54, 2024, 714, 2168]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_08_2024_4_24_16_59_92c22920a83749c994864397a370a984-10/38_text_Sort_by.png
try:
    _c38 = get_crop(38, 206, 75)
    canvas.paste(_c38, (42, 1931), _c38)
except Exception:
    pass
layout["Sort_by"] = [42, 1931, 248, 2006]
