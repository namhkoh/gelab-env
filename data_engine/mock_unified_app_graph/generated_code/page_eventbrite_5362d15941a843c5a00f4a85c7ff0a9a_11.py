# page_id: page_eventbrite_5362d15941a843c5a00f4a85c7ff0a9a_11
# screenshot: 2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-13.png
# step_index: 11/12
# task: Open Eventbrite. Set the city to 'Los Angeles'. Search 'Business'. Filter 'French' speaking events. Add the first event to favorite.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw background and structural UI elements for the filters page
w, h = canvas.size

# Base background (dominant color - white)
draw.rectangle((0, 0, w, h), fill=(255, 255, 255))

# Status bar (top ~72px) - light gray to match screenshot status area
status_h = 72
draw.rectangle((0, 0, w, status_h), fill=(230, 230, 230))

# Header area under status bar (~72-160px) with subtle divider
header_top = status_h
header_bottom = 160
draw.rectangle((0, header_top, w, header_bottom), fill=(255, 255, 255))
# subtle divider line under header
draw.line((24, header_bottom, w - 24, header_bottom), fill=(225, 225, 230), width=1)

# Helper: rounded rect parameters
margin_x = 36
card_radius = 20

# Categories card background (rounded rectangle behind the category group)
cat_top = 180
cat_bottom = 580
draw.rounded_rectangle(
    (margin_x, cat_top, w - margin_x, cat_bottom),
    radius=card_radius,
    fill=(247, 253, 255),
    outline=(235, 240, 245)
)

# Event type card background
etype_top = 640
etype_bottom = 1030
draw.rounded_rectangle(
    (margin_x, etype_top, w - margin_x, etype_bottom),
    radius=card_radius,
    fill=(247, 253, 255),
    outline=(235, 240, 245)
)

# Languages card background
lang_top = 1120
lang_bottom = 1510
draw.rounded_rectangle(
    (margin_x, lang_top, w - margin_x, lang_bottom),
    radius=card_radius,
    fill=(247, 253, 255),
    outline=(235, 240, 245)
)

# Price area background (small card region for Price / Only free events)
price_top = 1560
price_bottom = 1760
draw.rounded_rectangle(
    (margin_x, price_top, w - margin_x, price_bottom),
    radius=18,
    fill=(255, 255, 255),
    outline=(235, 235, 238)
)

# Subtle separator lines between major sections (to structure the page)
sep_color = (230, 230, 235)
draw.line((margin_x, cat_bottom + 16, w - margin_x, cat_bottom + 16), fill=sep_color, width=1)
draw.line((margin_x, etype_bottom + 16, w - margin_x, etype_bottom + 16), fill=sep_color, width=1)
draw.line((margin_x, lang_bottom + 16, w - margin_x, lang_bottom + 16), fill=sep_color, width=1)

# Faint shadow strip above bottom "Apply filters" area (do not draw the button itself)
apply_area_top = 2680
draw.line((24, apply_area_top, w - 24, apply_area_top), fill=(235, 235, 240), width=2)

# Light background block for the "Sort by" control area (do not draw controls)
sort_top = 1900
sort_bottom = 2050
draw.rounded_rectangle(
    (margin_x, sort_top, w - margin_x, sort_bottom),
    radius=14,
    fill=(250, 250, 252),
    outline=(235, 235, 240)
)

# Small bottom padding area to match app spacing
bottom_padding_top = h - 220
draw.rectangle((0, bottom_padding_top, w, h), fill=(255, 255, 255))

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_11_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-13/00_icon_French.png
try:
    _c0 = get_crop(0, 205, 144)
    canvas.paste(_c0, (768, 1275), _c0)
except Exception:
    pass
layout["French"] = [768, 1275, 973, 1419]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_11_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-13/01_icon_Food_Drink.png
try:
    _c1 = get_crop(1, 312, 144)
    canvas.paste(_c1, (512, 383), _c1)
except Exception:
    pass
layout["Food_&_Drink"] = [512, 383, 824, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_11_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-13/02_icon_Music.png
try:
    _c2 = get_crop(2, 187, 135)
    canvas.paste(_c2, (36, 383), _c2)
except Exception:
    pass
layout["Music"] = [36, 383, 223, 518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_11_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-13/03_icon_Community.png
try:
    _c3 = get_crop(3, 294, 144)
    canvas.paste(_c3, (848, 383), _c3)
except Exception:
    pass
layout["Community"] = [848, 383, 1142, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_11_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-13/04_icon_Business.png
try:
    _c4 = get_crop(4, 241, 135)
    canvas.paste(_c4, (247, 383), _c4)
except Exception:
    pass
layout["Business"] = [247, 383, 488, 518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_11_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-13/05_icon_Spanish.png
try:
    _c5 = get_crop(5, 225, 144)
    canvas.paste(_c5, (519, 1275), _c5)
except Exception:
    pass
layout["Spanish"] = [519, 1275, 744, 1419]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_11_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-13/06_icon_Italian.png
try:
    _c6 = get_crop(6, 191, 144)
    canvas.paste(_c6, (997, 1275), _c6)
except Exception:
    pass
layout["Italian"] = [997, 1275, 1188, 1419]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_11_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-13/07_icon_Expo.png
try:
    _c7 = get_crop(7, 167, 144)
    canvas.paste(_c7, (614, 829), _c7)
except Exception:
    pass
layout["Expo"] = [614, 829, 781, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_11_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-13/08_icon_Seminar.png
try:
    _c8 = get_crop(8, 232, 144)
    canvas.paste(_c8, (358, 829), _c8)
except Exception:
    pass
layout["Seminar"] = [358, 829, 590, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_11_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-13/09_icon_Arts.png
try:
    _c9 = get_crop(9, 152, 144)
    canvas.paste(_c9, (1166, 383), _c9)
except Exception:
    pass
layout["Arts"] = [1166, 383, 1318, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_11_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-13/10_icon_Convention.png
try:
    _c10 = get_crop(10, 293, 144)
    canvas.paste(_c10, (805, 829), _c10)
except Exception:
    pass
layout["Convention"] = [805, 829, 1098, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_11_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-13/11_icon_Festival.png
try:
    _c11 = get_crop(11, 219, 144)
    canvas.paste(_c11, (1122, 829), _c11)
except Exception:
    pass
layout["Festival"] = [1122, 829, 1341, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_11_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-13/12_icon_German.png
try:
    _c12 = get_crop(12, 225, 135)
    canvas.paste(_c12, (270, 1275), _c12)
except Exception:
    pass
layout["German"] = [270, 1275, 495, 1410]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_11_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-13/13_icon_English.png
try:
    _c13 = get_crop(13, 210, 135)
    canvas.paste(_c13, (36, 1275), _c13)
except Exception:
    pass
layout["English"] = [36, 1275, 246, 1410]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_11_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-13/14_icon_Conference.png
try:
    _c14 = get_crop(14, 298, 135)
    canvas.paste(_c14, (36, 829), _c14)
except Exception:
    pass
layout["Conference"] = [36, 829, 334, 964]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_11_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-13/15_icon_Apply_filters_1.png
try:
    _c15 = get_crop(15, 1344, 144)
    canvas.paste(_c15, (48, 2768), _c15)
except Exception:
    pass
layout["Apply_filters_(1)"] = [48, 2768, 1392, 2912]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_11_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-13/16_icon_Date.png
try:
    _c16 = get_crop(16, 660, 144)
    canvas.paste(_c16, (726, 2024), _c16)
except Exception:
    pass
layout["Date"] = [726, 2024, 1386, 2168]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_11_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-13/17_icon_Relevance.png
try:
    _c17 = get_crop(17, 660, 144)
    canvas.paste(_c17, (54, 2024), _c17)
except Exception:
    pass
layout["Relevance"] = [54, 2024, 714, 2168]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_11_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-13/18_icon_Clear_all.png
try:
    _c18 = get_crop(18, 52, 68)
    canvas.paste(_c18, (1153, 1), _c18)
except Exception:
    pass
layout["Clear_all"] = [1153, 1, 1205, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_11_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-13/19_icon_8.02.png
try:
    _c19 = get_crop(19, 60, 63)
    canvas.paste(_c19, (180, 2), _c19)
except Exception:
    pass
layout["8.02"] = [180, 2, 240, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_11_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-13/20_icon_8.02.png
try:
    _c20 = get_crop(20, 144, 144)
    canvas.paste(_c20, (12, 72), _c20)
except Exception:
    pass
layout["8.02"] = [12, 72, 156, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_11_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-13/21_icon_Clear_all.png
try:
    _c21 = get_crop(21, 98, 68)
    canvas.paste(_c21, (1211, 1), _c21)
except Exception:
    pass
layout["Clear_all"] = [1211, 1, 1309, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_11_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-13/22_icon_icon_22.png
try:
    _c22 = get_crop(22, 64, 61)
    canvas.paste(_c22, (308, 4), _c22)
except Exception:
    pass
layout["icon_22"] = [308, 4, 372, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_11_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-13/23_icon_8.02.png
try:
    _c23 = get_crop(23, 63, 65)
    canvas.paste(_c23, (113, 1), _c23)
except Exception:
    pass
layout["8.02"] = [113, 1, 176, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_11_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-13/24_icon_Clear_all.png
try:
    _c24 = get_crop(24, 53, 67)
    canvas.paste(_c24, (1319, 0), _c24)
except Exception:
    pass
layout["Clear_all"] = [1319, 0, 1372, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_11_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-13/25_icon_icon_25.png
try:
    _c25 = get_crop(25, 50, 61)
    canvas.paste(_c25, (249, 3), _c25)
except Exception:
    pass
layout["icon_25"] = [249, 3, 299, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_11_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-13/26_icon_Clear_all.png
try:
    _c26 = get_crop(26, 178, 144)
    canvas.paste(_c26, (1214, 72), _c26)
except Exception:
    pass
layout["Clear_all"] = [1214, 72, 1392, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_11_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-13/27_icon_Toggle_to_filter_only_free_events.png
try:
    _c27 = get_crop(27, 144, 144)
    canvas.paste(_c27, (1248, 1729), _c27)
except Exception:
    pass
layout["Toggle_to_filter_only_fre"] = [1248, 1729, 1392, 1873]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_11_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-13/28_text_8.02.png
try:
    _c28 = get_crop(28, 91, 43)
    canvas.paste(_c28, (20, 17), _c28)
except Exception:
    pass
layout["8.02"] = [20, 17, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_11_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-13/29_text_Filters.png
try:
    _c29 = get_crop(29, 180, 66)
    canvas.paste(_c29, (631, 116), _c29)
except Exception:
    pass
layout["Filters"] = [631, 116, 811, 182]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_11_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-13/30_text_Categories.png
try:
    _c30 = get_crop(30, 187, 135)
    canvas.paste(_c30, (36, 383), _c30)
except Exception:
    pass
layout["Categories"] = [36, 383, 223, 518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_11_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-13/31_text_Show_all_categories.png
try:
    _c31 = get_crop(31, 516, 144)
    canvas.paste(_c31, (0, 518), _c31)
except Exception:
    pass
layout["Show_all_categories"] = [0, 518, 516, 662]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_11_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-13/32_text_Event_type.png
try:
    _c32 = get_crop(32, 298, 135)
    canvas.paste(_c32, (36, 829), _c32)
except Exception:
    pass
layout["Event_type"] = [36, 829, 334, 964]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_11_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-13/33_text_Show_all_event_types.png
try:
    _c33 = get_crop(33, 535, 144)
    canvas.paste(_c33, (0, 964), _c33)
except Exception:
    pass
layout["Show_all_event_types"] = [0, 964, 535, 1108]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_11_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-13/34_text_Languages.png
try:
    _c34 = get_crop(34, 210, 135)
    canvas.paste(_c34, (36, 1275), _c34)
except Exception:
    pass
layout["Languages"] = [36, 1275, 246, 1410]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_11_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-13/35_text_Show_all_languages.png
try:
    _c35 = get_crop(35, 511, 144)
    canvas.paste(_c35, (0, 1410), _c35)
except Exception:
    pass
layout["Show_all_languages"] = [0, 1410, 511, 1554]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_11_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-13/36_text_Price.png
try:
    _c36 = get_crop(36, 149, 63)
    canvas.paste(_c36, (45, 1613), _c36)
except Exception:
    pass
layout["Price"] = [45, 1613, 194, 1676]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_11_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-13/37_text_Only_free_events.png
try:
    _c37 = get_crop(37, 660, 144)
    canvas.paste(_c37, (54, 2024), _c37)
except Exception:
    pass
layout["Only_free_events"] = [54, 2024, 714, 2168]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_11_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-13/38_text_Sort_by.png
try:
    _c38 = get_crop(38, 206, 75)
    canvas.paste(_c38, (42, 1931), _c38)
except Exception:
    pass
layout["Sort_by"] = [42, 1931, 248, 2006]
