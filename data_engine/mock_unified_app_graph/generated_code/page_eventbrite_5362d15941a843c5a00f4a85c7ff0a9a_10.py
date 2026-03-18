# page_id: page_eventbrite_5362d15941a843c5a00f4a85c7ff0a9a_10
# screenshot: 2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-12.png
# step_index: 10/12
# task: Open Eventbrite. Set the city to 'Los Angeles'. Search 'Business'. Filter 'French' speaking events. Add the first event to favorite.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw overall subtle off-white background
draw.rectangle([(0, 0), canvas.size], fill="#fbfcfe")

W, H = canvas.size

# Status bar (top area)
status_h = 88
draw.rectangle([(0, 0), (W, status_h)], fill="#cfcfcf")

# Header / toolbar area below status bar
header_top = status_h
header_h = 88
draw.rectangle([(0, header_top), (W, header_top + header_h)], fill="#ffffff")
# subtle divider under header
draw.line([(24, header_top + header_h), (W - 24, header_top + header_h)], fill="#e9e6ec", width=1)

# Function shortcut for rounded rects
def rr(x1, y1, x2, y2, r, fill=None, outline=None):
    draw.rounded_rectangle([(x1, y1), (x2, y2)], radius=r, fill=fill, outline=outline)

# Section card backgrounds (rounded) - avoid drawing over detected bottom action area
left_margin = 24
right_margin = W - 24

# Categories card area (behind chips) - subtle white card with faint border
categories_top = 220
categories_bottom = 560
rr(left_margin, categories_top, right_margin, categories_bottom, r=20, fill="#ffffff", outline="#efedf1")

# Event type card area
event_top = 680
event_bottom = 1040
rr(left_margin, event_top, right_margin, event_bottom, r=20, fill="#ffffff", outline="#efedf1")

# Languages card area
langs_top = 1120
langs_bottom = 1500
rr(left_margin, langs_top, right_margin, langs_bottom, r=20, fill="#ffffff", outline="#efedf1")

# Price / toggle area (smaller card to host price/toggle region)
price_top = 1560
price_bottom = 1710
rr(left_margin, price_top, right_margin, price_bottom, r=16, fill="#ffffff", outline="#efedf1")

# Light separators between logical sections (thin lines)
seps = [ (categories_bottom + 20), (event_bottom + 20), (langs_bottom + 20), (price_bottom + 40) ]
for y in seps:
    draw.line([(left_margin, y), (right_margin, y)], fill="#f0eef3", width=1)

# Draw a subtle container outline for "Sort by" area (but do not draw the inner tab controls themselves)
sort_container_top = 1880
sort_container_bottom = 2100
rr(left_margin, sort_container_top, right_margin, sort_container_bottom, r=18, fill="#ffffff", outline="#e6e0e9")

# Add a subtle inner divider above the sort container to separate from content
draw.line([(left_margin + 4, sort_container_top - 18), (right_margin - 4, sort_container_top - 18)], fill="#f1eef3", width=1)

# Add faint drop-shadows under cards to give structure (very subtle lines)
shadow_color = "#f3f2f6"
draw.line([(left_margin, categories_bottom + 2), (right_margin, categories_bottom + 2)], fill=shadow_color, width=2)
draw.line([(left_margin, event_bottom + 2), (right_margin, event_bottom + 2)], fill=shadow_color, width=2)
draw.line([(left_margin, langs_bottom + 2), (right_margin, langs_bottom + 2)], fill=shadow_color, width=2)

# Bottom safe area above the main action button (leave the Apply Filters area untouched)
bottom_safe_top = 2640
draw.rectangle([(0, bottom_safe_top), (W, H)], fill="#fbfcfe")

# Final thin top border for the very top of screen to match status area separation
draw.line([(0, status_h - 1), (W, status_h - 1)], fill="#bdbdbd", width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_10_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-12/00_icon_Food_Drink.png
try:
    _c0 = get_crop(0, 312, 144)
    canvas.paste(_c0, (512, 383), _c0)
except Exception:
    pass
layout["Food_&_Drink"] = [512, 383, 824, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_10_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-12/01_icon_Music.png
try:
    _c1 = get_crop(1, 187, 135)
    canvas.paste(_c1, (36, 383), _c1)
except Exception:
    pass
layout["Music"] = [36, 383, 223, 518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_10_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-12/02_icon_Community.png
try:
    _c2 = get_crop(2, 294, 144)
    canvas.paste(_c2, (848, 383), _c2)
except Exception:
    pass
layout["Community"] = [848, 383, 1142, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_10_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-12/03_icon_French.png
try:
    _c3 = get_crop(3, 205, 144)
    canvas.paste(_c3, (768, 1275), _c3)
except Exception:
    pass
layout["French"] = [768, 1275, 973, 1419]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_10_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-12/04_icon_Business.png
try:
    _c4 = get_crop(4, 241, 135)
    canvas.paste(_c4, (247, 383), _c4)
except Exception:
    pass
layout["Business"] = [247, 383, 488, 518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_10_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-12/05_icon_Spanish.png
try:
    _c5 = get_crop(5, 225, 144)
    canvas.paste(_c5, (519, 1275), _c5)
except Exception:
    pass
layout["Spanish"] = [519, 1275, 744, 1419]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_10_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-12/06_icon_Expo.png
try:
    _c6 = get_crop(6, 167, 144)
    canvas.paste(_c6, (614, 829), _c6)
except Exception:
    pass
layout["Expo"] = [614, 829, 781, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_10_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-12/07_icon_Italian.png
try:
    _c7 = get_crop(7, 191, 144)
    canvas.paste(_c7, (997, 1275), _c7)
except Exception:
    pass
layout["Italian"] = [997, 1275, 1188, 1419]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_10_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-12/08_icon_Seminar.png
try:
    _c8 = get_crop(8, 232, 144)
    canvas.paste(_c8, (358, 829), _c8)
except Exception:
    pass
layout["Seminar"] = [358, 829, 590, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_10_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-12/09_icon_Convention.png
try:
    _c9 = get_crop(9, 293, 144)
    canvas.paste(_c9, (805, 829), _c9)
except Exception:
    pass
layout["Convention"] = [805, 829, 1098, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_10_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-12/10_icon_Arts.png
try:
    _c10 = get_crop(10, 152, 144)
    canvas.paste(_c10, (1166, 383), _c10)
except Exception:
    pass
layout["Arts"] = [1166, 383, 1318, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_10_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-12/11_icon_Festival.png
try:
    _c11 = get_crop(11, 219, 144)
    canvas.paste(_c11, (1122, 829), _c11)
except Exception:
    pass
layout["Festival"] = [1122, 829, 1341, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_10_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-12/12_icon_German.png
try:
    _c12 = get_crop(12, 225, 135)
    canvas.paste(_c12, (270, 1275), _c12)
except Exception:
    pass
layout["German"] = [270, 1275, 495, 1410]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_10_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-12/13_icon_English.png
try:
    _c13 = get_crop(13, 210, 135)
    canvas.paste(_c13, (36, 1275), _c13)
except Exception:
    pass
layout["English"] = [36, 1275, 246, 1410]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_10_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-12/14_icon_Conference.png
try:
    _c14 = get_crop(14, 298, 135)
    canvas.paste(_c14, (36, 829), _c14)
except Exception:
    pass
layout["Conference"] = [36, 829, 334, 964]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_10_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-12/15_icon_Date.png
try:
    _c15 = get_crop(15, 660, 144)
    canvas.paste(_c15, (726, 2024), _c15)
except Exception:
    pass
layout["Date"] = [726, 2024, 1386, 2168]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_10_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-12/16_icon_Relevance.png
try:
    _c16 = get_crop(16, 660, 144)
    canvas.paste(_c16, (54, 2024), _c16)
except Exception:
    pass
layout["Relevance"] = [54, 2024, 714, 2168]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_10_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-12/17_icon_Clear_all.png
try:
    _c17 = get_crop(17, 51, 70)
    canvas.paste(_c17, (1153, 1), _c17)
except Exception:
    pass
layout["Clear_all"] = [1153, 1, 1204, 71]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_10_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-12/18_icon_Apply_filters.png
try:
    _c18 = get_crop(18, 1344, 144)
    canvas.paste(_c18, (48, 2768), _c18)
except Exception:
    pass
layout["Apply_filters"] = [48, 2768, 1392, 2912]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_10_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-12/19_icon_Clear_all.png
try:
    _c19 = get_crop(19, 100, 70)
    canvas.paste(_c19, (1211, 0), _c19)
except Exception:
    pass
layout["Clear_all"] = [1211, 0, 1311, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_10_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-12/20_icon_8.02.png
try:
    _c20 = get_crop(20, 144, 144)
    canvas.paste(_c20, (12, 72), _c20)
except Exception:
    pass
layout["8.02"] = [12, 72, 156, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_10_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-12/21_icon_8.02.png
try:
    _c21 = get_crop(21, 61, 63)
    canvas.paste(_c21, (179, 2), _c21)
except Exception:
    pass
layout["8.02"] = [179, 2, 240, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_10_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-12/22_icon_icon_22.png
try:
    _c22 = get_crop(22, 66, 62)
    canvas.paste(_c22, (307, 3), _c22)
except Exception:
    pass
layout["icon_22"] = [307, 3, 373, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_10_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-12/23_icon_8.02.png
try:
    _c23 = get_crop(23, 64, 65)
    canvas.paste(_c23, (112, 1), _c23)
except Exception:
    pass
layout["8.02"] = [112, 1, 176, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_10_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-12/24_icon_Clear_all.png
try:
    _c24 = get_crop(24, 52, 69)
    canvas.paste(_c24, (1320, 0), _c24)
except Exception:
    pass
layout["Clear_all"] = [1320, 0, 1372, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_10_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-12/25_icon_icon_25.png
try:
    _c25 = get_crop(25, 51, 61)
    canvas.paste(_c25, (248, 3), _c25)
except Exception:
    pass
layout["icon_25"] = [248, 3, 299, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_10_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-12/26_icon_Clear_all.png
try:
    _c26 = get_crop(26, 178, 144)
    canvas.paste(_c26, (1214, 72), _c26)
except Exception:
    pass
layout["Clear_all"] = [1214, 72, 1392, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_10_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-12/27_icon_clickable_20.png
try:
    _c27 = get_crop(27, 144, 144)
    canvas.paste(_c27, (1248, 1729), _c27)
except Exception:
    pass
layout["clickable_20"] = [1248, 1729, 1392, 1873]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_10_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-12/28_text_8.02.png
try:
    _c28 = get_crop(28, 91, 43)
    canvas.paste(_c28, (20, 17), _c28)
except Exception:
    pass
layout["8.02"] = [20, 17, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_10_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-12/29_text_Filters.png
try:
    _c29 = get_crop(29, 180, 66)
    canvas.paste(_c29, (631, 116), _c29)
except Exception:
    pass
layout["Filters"] = [631, 116, 811, 182]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_10_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-12/30_text_Categories.png
try:
    _c30 = get_crop(30, 187, 135)
    canvas.paste(_c30, (36, 383), _c30)
except Exception:
    pass
layout["Categories"] = [36, 383, 223, 518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_10_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-12/31_text_Show_all_categories.png
try:
    _c31 = get_crop(31, 516, 144)
    canvas.paste(_c31, (0, 518), _c31)
except Exception:
    pass
layout["Show_all_categories"] = [0, 518, 516, 662]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_10_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-12/32_text_Event_type.png
try:
    _c32 = get_crop(32, 298, 135)
    canvas.paste(_c32, (36, 829), _c32)
except Exception:
    pass
layout["Event_type"] = [36, 829, 334, 964]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_10_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-12/33_text_Show_all_event_types.png
try:
    _c33 = get_crop(33, 535, 144)
    canvas.paste(_c33, (0, 964), _c33)
except Exception:
    pass
layout["Show_all_event_types"] = [0, 964, 535, 1108]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_10_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-12/34_text_Languages.png
try:
    _c34 = get_crop(34, 210, 135)
    canvas.paste(_c34, (36, 1275), _c34)
except Exception:
    pass
layout["Languages"] = [36, 1275, 246, 1410]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_10_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-12/35_text_Show_all_languages.png
try:
    _c35 = get_crop(35, 511, 144)
    canvas.paste(_c35, (0, 1410), _c35)
except Exception:
    pass
layout["Show_all_languages"] = [0, 1410, 511, 1554]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_10_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-12/36_text_Price.png
try:
    _c36 = get_crop(36, 149, 63)
    canvas.paste(_c36, (45, 1613), _c36)
except Exception:
    pass
layout["Price"] = [45, 1613, 194, 1676]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_10_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-12/37_text_Only_free_events.png
try:
    _c37 = get_crop(37, 660, 144)
    canvas.paste(_c37, (54, 2024), _c37)
except Exception:
    pass
layout["Only_free_events"] = [54, 2024, 714, 2168]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_10_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-12/38_text_Sort_by.png
try:
    _c38 = get_crop(38, 206, 75)
    canvas.paste(_c38, (42, 1931), _c38)
except Exception:
    pass
layout["Sort_by"] = [42, 1931, 248, 2006]
