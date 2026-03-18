# page_id: page_eventbrite_3ce6196f48694f74bf7d05dc71840c63_06
# screenshot: 2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-8.png
# step_index: 6/9
# task: Open Eventbrite. Search for 'coding workshop'. Sort the results by date. Where is the location of the soonest event that is not promoted?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Top status bar
draw.rectangle([(0, 0), (1440, 80)], fill="#CFCFCF")

# Header area background (keeps it visually separate from content)
draw.rectangle([(0, 80), (1440, 168)], fill="#FFFFFF")
# subtle divider under header
draw.line([(24, 168), (1416, 168)], fill="#EFEFF1", width=2)

# Large content background (main white area already present, add subtle warm tint blocks for structure)
draw.rectangle([(24, 180), (1416, 2720)], fill="#FFFFFF")

# Category section card background (rounded)
draw.rounded_rectangle([(30, 200), (1410, 520)], radius=22, fill="#FFFFFF", outline="#F0F1F4", width=1)
# faint shadow line beneath card
draw.line([(36, 524), (1404, 524)], fill="#F3F4F6", width=2)

# Event type section card background (rounded)
draw.rounded_rectangle([(30, 650), (1410, 970)], radius=22, fill="#FFFFFF", outline="#F0F1F4", width=1)
draw.line([(36, 974), (1404, 974)], fill="#F3F4F6", width=2)

# Languages section card background (rounded)
draw.rounded_rectangle([(30, 1040), (1410, 1390)], radius=22, fill="#FFFFFF", outline="#F0F1F4", width=1)
draw.line([(36, 1394), (1404, 1394)], fill="#F3F4F6", width=2)

# Price / Toggle area container (subtle grouping)
draw.rounded_rectangle([(30, 1540), (1410, 1650)], radius=16, fill="#FFFFFF", outline="#F0F1F4", width=1)
draw.line([(36, 1654), (1404, 1654)], fill="#F3F4F6", width=2)

# Sort by control background (rounded pill with two segments)
sort_top = 1960
sort_bottom = 2108
left = 36
right = 1404
radius = 18
# overall container (slight border)
draw.rounded_rectangle([(left, sort_top), (right, sort_bottom)], radius=radius, fill="#FFFFFF", outline="#D9D6DD", width=2)
# left (selected) segment
mid = (left + right) // 2
draw.rounded_rectangle([(left+6, sort_top+6), (mid-4, sort_bottom-6)], radius=14, fill="#EAEAF0", outline=None)
# right (unselected) segment (just ensure a subtle separation, leave center border)
draw.rounded_rectangle([(mid+4, sort_top+6), (right-6, sort_bottom-6)], radius=14, fill="#FFFFFF", outline=None)
# thin separator between segments
draw.line([(mid, sort_top+8), (mid, sort_bottom-8)], fill="#E2DEE5", width=2)

# Large empty content area (visual bottom whitespace)
draw.rectangle([(24, 2120), (1416, 2660)], fill="#FFFFFF")

# Top and bottom safe-area separators near apply button area (do not draw button itself)
draw.line([(24, 2668), (1416, 2668)], fill="#E9E9EC", width=2)
draw.line([(24, 2752), (1416, 2752)], fill="#E9E9EC", width=2)

# Small decorative top-left back-area background accent (behind back icon only)
draw.ellipse([(-8, 88), (92, 196)], fill="#FFFFFF", outline=None)

# Subtle vertical rhythm lines to separate logical groups (very light, non-intrusive)
draw.line([(24, 520), (24, 2660)], fill="#FFFFFF", width=1)
draw.line([(1416, 520), (1416, 2660)], fill="#FFFFFF", width=1)

# Light page shadow at very bottom
draw.rectangle([(0, 2920), (1440, 2960)], fill="#FFFFFF")

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_06_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-8/00_icon_Food_Drink.png
try:
    _c0 = get_crop(0, 312, 144)
    canvas.paste(_c0, (512, 383), _c0)
except Exception:
    pass
layout["Food_&_Drink"] = [512, 383, 824, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_06_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-8/01_icon_Music.png
try:
    _c1 = get_crop(1, 187, 135)
    canvas.paste(_c1, (36, 383), _c1)
except Exception:
    pass
layout["Music"] = [36, 383, 223, 518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_06_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-8/02_icon_Community.png
try:
    _c2 = get_crop(2, 294, 144)
    canvas.paste(_c2, (848, 383), _c2)
except Exception:
    pass
layout["Community"] = [848, 383, 1142, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_06_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-8/03_icon_French.png
try:
    _c3 = get_crop(3, 205, 144)
    canvas.paste(_c3, (768, 1275), _c3)
except Exception:
    pass
layout["French"] = [768, 1275, 973, 1419]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_06_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-8/04_icon_Spanish.png
try:
    _c4 = get_crop(4, 225, 144)
    canvas.paste(_c4, (519, 1275), _c4)
except Exception:
    pass
layout["Spanish"] = [519, 1275, 744, 1419]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_06_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-8/05_icon_Business.png
try:
    _c5 = get_crop(5, 241, 135)
    canvas.paste(_c5, (247, 383), _c5)
except Exception:
    pass
layout["Business"] = [247, 383, 488, 518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_06_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-8/06_icon_Expo.png
try:
    _c6 = get_crop(6, 167, 144)
    canvas.paste(_c6, (614, 829), _c6)
except Exception:
    pass
layout["Expo"] = [614, 829, 781, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_06_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-8/07_icon_Seminar.png
try:
    _c7 = get_crop(7, 232, 144)
    canvas.paste(_c7, (358, 829), _c7)
except Exception:
    pass
layout["Seminar"] = [358, 829, 590, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_06_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-8/08_icon_Italian.png
try:
    _c8 = get_crop(8, 191, 144)
    canvas.paste(_c8, (997, 1275), _c8)
except Exception:
    pass
layout["Italian"] = [997, 1275, 1188, 1419]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_06_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-8/09_icon_Arts.png
try:
    _c9 = get_crop(9, 152, 144)
    canvas.paste(_c9, (1166, 383), _c9)
except Exception:
    pass
layout["Arts"] = [1166, 383, 1318, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_06_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-8/10_icon_Convention.png
try:
    _c10 = get_crop(10, 293, 144)
    canvas.paste(_c10, (805, 829), _c10)
except Exception:
    pass
layout["Convention"] = [805, 829, 1098, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_06_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-8/11_icon_Festival.png
try:
    _c11 = get_crop(11, 219, 144)
    canvas.paste(_c11, (1122, 829), _c11)
except Exception:
    pass
layout["Festival"] = [1122, 829, 1341, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_06_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-8/12_icon_German.png
try:
    _c12 = get_crop(12, 225, 135)
    canvas.paste(_c12, (270, 1275), _c12)
except Exception:
    pass
layout["German"] = [270, 1275, 495, 1410]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_06_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-8/13_icon_English.png
try:
    _c13 = get_crop(13, 210, 135)
    canvas.paste(_c13, (36, 1275), _c13)
except Exception:
    pass
layout["English"] = [36, 1275, 246, 1410]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_06_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-8/14_icon_Conference.png
try:
    _c14 = get_crop(14, 298, 135)
    canvas.paste(_c14, (36, 829), _c14)
except Exception:
    pass
layout["Conference"] = [36, 829, 334, 964]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_06_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-8/15_icon_Apply_filters_1.png
try:
    _c15 = get_crop(15, 1344, 144)
    canvas.paste(_c15, (48, 2768), _c15)
except Exception:
    pass
layout["Apply_filters_(1)"] = [48, 2768, 1392, 2912]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_06_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-8/16_icon_Date.png
try:
    _c16 = get_crop(16, 660, 144)
    canvas.paste(_c16, (726, 2024), _c16)
except Exception:
    pass
layout["Date"] = [726, 2024, 1386, 2168]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_06_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-8/17_icon_7.24.png
try:
    _c17 = get_crop(17, 60, 63)
    canvas.paste(_c17, (180, 2), _c17)
except Exception:
    pass
layout["7.24"] = [180, 2, 240, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_06_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-8/18_icon_7.24.png
try:
    _c18 = get_crop(18, 144, 144)
    canvas.paste(_c18, (12, 72), _c18)
except Exception:
    pass
layout["7.24"] = [12, 72, 156, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_06_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-8/19_icon_Relevance.png
try:
    _c19 = get_crop(19, 660, 144)
    canvas.paste(_c19, (54, 2024), _c19)
except Exception:
    pass
layout["Relevance"] = [54, 2024, 714, 2168]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_06_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-8/20_icon_7.24.png
try:
    _c20 = get_crop(20, 63, 65)
    canvas.paste(_c20, (112, 1), _c20)
except Exception:
    pass
layout["7.24"] = [112, 1, 175, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_06_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-8/21_icon_icon_21.png
try:
    _c21 = get_crop(21, 64, 62)
    canvas.paste(_c21, (308, 3), _c21)
except Exception:
    pass
layout["icon_21"] = [308, 3, 372, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_06_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-8/22_icon_Clear_all.png
try:
    _c22 = get_crop(22, 56, 67)
    canvas.paste(_c22, (1318, 0), _c22)
except Exception:
    pass
layout["Clear_all"] = [1318, 0, 1374, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_06_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-8/23_icon_Clear_all.png
try:
    _c23 = get_crop(23, 99, 65)
    canvas.paste(_c23, (1211, 0), _c23)
except Exception:
    pass
layout["Clear_all"] = [1211, 0, 1310, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_06_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-8/24_icon_icon_24.png
try:
    _c24 = get_crop(24, 51, 62)
    canvas.paste(_c24, (248, 2), _c24)
except Exception:
    pass
layout["icon_24"] = [248, 2, 299, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_06_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-8/25_icon_clickable_20.png
try:
    _c25 = get_crop(25, 144, 144)
    canvas.paste(_c25, (1248, 1729), _c25)
except Exception:
    pass
layout["clickable_20"] = [1248, 1729, 1392, 1873]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_06_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-8/26_icon_Clear_all.png
try:
    _c26 = get_crop(26, 178, 144)
    canvas.paste(_c26, (1214, 72), _c26)
except Exception:
    pass
layout["Clear_all"] = [1214, 72, 1392, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_06_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-8/27_icon_7.24.png
try:
    _c27 = get_crop(27, 101, 65)
    canvas.paste(_c27, (9, 0), _c27)
except Exception:
    pass
layout["7.24"] = [9, 0, 110, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_06_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-8/28_text_Filters.png
try:
    _c28 = get_crop(28, 180, 66)
    canvas.paste(_c28, (631, 116), _c28)
except Exception:
    pass
layout["Filters"] = [631, 116, 811, 182]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_06_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-8/29_text_Categories.png
try:
    _c29 = get_crop(29, 187, 135)
    canvas.paste(_c29, (36, 383), _c29)
except Exception:
    pass
layout["Categories"] = [36, 383, 223, 518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_06_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-8/30_text_Show_all_categories.png
try:
    _c30 = get_crop(30, 516, 144)
    canvas.paste(_c30, (0, 518), _c30)
except Exception:
    pass
layout["Show_all_categories"] = [0, 518, 516, 662]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_06_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-8/31_text_Event_type.png
try:
    _c31 = get_crop(31, 298, 135)
    canvas.paste(_c31, (36, 829), _c31)
except Exception:
    pass
layout["Event_type"] = [36, 829, 334, 964]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_06_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-8/32_text_Show_all_event_types.png
try:
    _c32 = get_crop(32, 535, 144)
    canvas.paste(_c32, (0, 964), _c32)
except Exception:
    pass
layout["Show_all_event_types"] = [0, 964, 535, 1108]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_06_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-8/33_text_Languages.png
try:
    _c33 = get_crop(33, 210, 135)
    canvas.paste(_c33, (36, 1275), _c33)
except Exception:
    pass
layout["Languages"] = [36, 1275, 246, 1410]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_06_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-8/34_text_Show_all_languages.png
try:
    _c34 = get_crop(34, 511, 144)
    canvas.paste(_c34, (0, 1410), _c34)
except Exception:
    pass
layout["Show_all_languages"] = [0, 1410, 511, 1554]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_06_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-8/35_text_Price.png
try:
    _c35 = get_crop(35, 149, 63)
    canvas.paste(_c35, (45, 1613), _c35)
except Exception:
    pass
layout["Price"] = [45, 1613, 194, 1676]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_06_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-8/36_text_Only_free_events.png
try:
    _c36 = get_crop(36, 660, 144)
    canvas.paste(_c36, (54, 2024), _c36)
except Exception:
    pass
layout["Only_free_events"] = [54, 2024, 714, 2168]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_06_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-8/37_text_Sort_by.png
try:
    _c37 = get_crop(37, 208, 76)
    canvas.paste(_c37, (40, 1930), _c37)
except Exception:
    pass
layout["Sort_by"] = [40, 1930, 248, 2006]
