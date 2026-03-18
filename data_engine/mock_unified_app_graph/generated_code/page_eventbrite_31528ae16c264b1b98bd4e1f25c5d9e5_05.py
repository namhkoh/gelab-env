# page_id: page_eventbrite_31528ae16c264b1b98bd4e1f25c5d9e5_05
# screenshot: 2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-7.png
# step_index: 5/11
# task: Open Eventbrite. Search 'Fitness'. Filter for free events. Browse and select any 'Yoga' event. Note the location.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# draw overall background and structural elements for the UI page
# (variables provided by the environment: canvas (1440x2960 PIL Image), draw (ImageDraw), font_*)

# ensure base background is clean white
draw.rectangle([(0, 0), (1440, 2960)], fill="#FFFFFF")

# Status bar area at the very top (~72px high)
status_h = 72
draw.rectangle([(0, 0), (1440, status_h)], fill="#D0D2D4")  # light gray status bar

# Header / toolbar area below status bar
header_top = status_h
header_bottom = 160
draw.rectangle([(0, header_top), (1440, header_bottom)], fill="#FFFFFF")

# subtle header bottom divider / shadow
draw.line([(24, header_bottom), (1416, header_bottom)], fill="#EFEAF0", width=2)
draw.rectangle([(0, header_bottom-2), (1440, header_bottom+2)], fill=None, outline="#F6F3F7")

# Section separators between major groups (subtle, match soft purple/gray theme)
separators_y = [740, 1180, 1600, 1980]
for y in separators_y:
    draw.line([(36, y), (1404, y)], fill="#F0EDF2", width=1)

# Segmented control background for "Sort by" (rounded outer capsule)
seg_left = 48
seg_top = 2024
seg_right = 1392
seg_bottom = seg_top + 144
draw.rounded_rectangle(
    [(seg_left, seg_top), (seg_right, seg_bottom)],
    radius=18,
    fill="#F6F4F8",
    outline="#DCD9DE",
    width=2
)

# Slight inner shadow under segmented control to ground it
draw.line([(seg_left + 6, seg_bottom), (seg_right - 6, seg_bottom)], fill="#ECE9EE", width=3)

# Large bottom "Apply filters" bar background (rounded rectangle) with border
apply_left = 48
apply_top = 2768
apply_right = apply_left + 1344
apply_bottom = apply_top + 144
draw.rounded_rectangle(
    [(apply_left, apply_top), (apply_right, apply_bottom)],
    radius=12,
    fill="#FFFFFF",
    outline="#BFB9C3",
    width=4
)

# subtle top shadow for the bottom bar
draw.rectangle([(apply_left, apply_top-6), (apply_right, apply_top)], fill="#F3F0F4")

# Outer page left/right safe-area guidelines (very subtle vertical guides to frame layout)
draw.line([(24, header_bottom + 8), (24, apply_top - 12)], fill="#FFFFFF", width=1)
draw.line([(1416, header_bottom + 8), (1416, apply_top - 12)], fill="#FFFFFF", width=1)

# end of structural drawing

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_05_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-7/00_icon_Food_Drink.png
try:
    _c0 = get_crop(0, 312, 144)
    canvas.paste(_c0, (512, 383), _c0)
except Exception:
    pass
layout["Food_&_Drink"] = [512, 383, 824, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_05_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-7/01_icon_Music.png
try:
    _c1 = get_crop(1, 187, 135)
    canvas.paste(_c1, (36, 383), _c1)
except Exception:
    pass
layout["Music"] = [36, 383, 223, 518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_05_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-7/02_icon_Community.png
try:
    _c2 = get_crop(2, 294, 144)
    canvas.paste(_c2, (848, 383), _c2)
except Exception:
    pass
layout["Community"] = [848, 383, 1142, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_05_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-7/03_icon_French.png
try:
    _c3 = get_crop(3, 205, 144)
    canvas.paste(_c3, (768, 1275), _c3)
except Exception:
    pass
layout["French"] = [768, 1275, 973, 1419]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_05_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-7/04_icon_Business.png
try:
    _c4 = get_crop(4, 241, 135)
    canvas.paste(_c4, (247, 383), _c4)
except Exception:
    pass
layout["Business"] = [247, 383, 488, 518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_05_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-7/05_icon_Spanish.png
try:
    _c5 = get_crop(5, 225, 144)
    canvas.paste(_c5, (519, 1275), _c5)
except Exception:
    pass
layout["Spanish"] = [519, 1275, 744, 1419]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_05_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-7/06_icon_Expo.png
try:
    _c6 = get_crop(6, 167, 144)
    canvas.paste(_c6, (614, 829), _c6)
except Exception:
    pass
layout["Expo"] = [614, 829, 781, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_05_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-7/07_icon_Italian.png
try:
    _c7 = get_crop(7, 191, 144)
    canvas.paste(_c7, (997, 1275), _c7)
except Exception:
    pass
layout["Italian"] = [997, 1275, 1188, 1419]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_05_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-7/08_icon_Seminar.png
try:
    _c8 = get_crop(8, 232, 144)
    canvas.paste(_c8, (358, 829), _c8)
except Exception:
    pass
layout["Seminar"] = [358, 829, 590, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_05_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-7/09_icon_Arts.png
try:
    _c9 = get_crop(9, 152, 144)
    canvas.paste(_c9, (1166, 383), _c9)
except Exception:
    pass
layout["Arts"] = [1166, 383, 1318, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_05_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-7/10_icon_Convention.png
try:
    _c10 = get_crop(10, 293, 144)
    canvas.paste(_c10, (805, 829), _c10)
except Exception:
    pass
layout["Convention"] = [805, 829, 1098, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_05_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-7/11_icon_Festival.png
try:
    _c11 = get_crop(11, 219, 144)
    canvas.paste(_c11, (1122, 829), _c11)
except Exception:
    pass
layout["Festival"] = [1122, 829, 1341, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_05_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-7/12_icon_German.png
try:
    _c12 = get_crop(12, 225, 135)
    canvas.paste(_c12, (270, 1275), _c12)
except Exception:
    pass
layout["German"] = [270, 1275, 495, 1410]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_05_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-7/13_icon_English.png
try:
    _c13 = get_crop(13, 210, 135)
    canvas.paste(_c13, (36, 1275), _c13)
except Exception:
    pass
layout["English"] = [36, 1275, 246, 1410]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_05_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-7/14_icon_Conference.png
try:
    _c14 = get_crop(14, 298, 135)
    canvas.paste(_c14, (36, 829), _c14)
except Exception:
    pass
layout["Conference"] = [36, 829, 334, 964]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_05_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-7/15_icon_Date.png
try:
    _c15 = get_crop(15, 660, 144)
    canvas.paste(_c15, (726, 2024), _c15)
except Exception:
    pass
layout["Date"] = [726, 2024, 1386, 2168]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_05_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-7/16_icon_Relevance.png
try:
    _c16 = get_crop(16, 660, 144)
    canvas.paste(_c16, (54, 2024), _c16)
except Exception:
    pass
layout["Relevance"] = [54, 2024, 714, 2168]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_05_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-7/17_icon_Apply_filters.png
try:
    _c17 = get_crop(17, 1344, 144)
    canvas.paste(_c17, (48, 2768), _c17)
except Exception:
    pass
layout["Apply_filters"] = [48, 2768, 1392, 2912]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_05_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-7/18_icon_7.55.png
try:
    _c18 = get_crop(18, 61, 64)
    canvas.paste(_c18, (179, 2), _c18)
except Exception:
    pass
layout["7.55"] = [179, 2, 240, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_05_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-7/19_icon_7.55.png
try:
    _c19 = get_crop(19, 144, 144)
    canvas.paste(_c19, (12, 72), _c19)
except Exception:
    pass
layout["7.55"] = [12, 72, 156, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_05_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-7/20_icon_7.55.png
try:
    _c20 = get_crop(20, 65, 66)
    canvas.paste(_c20, (111, 1), _c20)
except Exception:
    pass
layout["7.55"] = [111, 1, 176, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_05_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-7/21_icon_icon_21.png
try:
    _c21 = get_crop(21, 66, 62)
    canvas.paste(_c21, (307, 3), _c21)
except Exception:
    pass
layout["icon_21"] = [307, 3, 373, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_05_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-7/22_icon_Clear_all.png
try:
    _c22 = get_crop(22, 55, 69)
    canvas.paste(_c22, (1319, 0), _c22)
except Exception:
    pass
layout["Clear_all"] = [1319, 0, 1374, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_05_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-7/23_icon_icon_23.png
try:
    _c23 = get_crop(23, 51, 62)
    canvas.paste(_c23, (248, 2), _c23)
except Exception:
    pass
layout["icon_23"] = [248, 2, 299, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_05_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-7/24_icon_Clear_all.png
try:
    _c24 = get_crop(24, 99, 70)
    canvas.paste(_c24, (1211, 0), _c24)
except Exception:
    pass
layout["Clear_all"] = [1211, 0, 1310, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_05_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-7/25_icon_Toggle_to_filter_only_free_events.png
try:
    _c25 = get_crop(25, 144, 144)
    canvas.paste(_c25, (1248, 1729), _c25)
except Exception:
    pass
layout["Toggle_to_filter_only_fre"] = [1248, 1729, 1392, 1873]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_05_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-7/26_icon_Clear_all.png
try:
    _c26 = get_crop(26, 178, 144)
    canvas.paste(_c26, (1214, 72), _c26)
except Exception:
    pass
layout["Clear_all"] = [1214, 72, 1392, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_05_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-7/27_text_Filters.png
try:
    _c27 = get_crop(27, 180, 66)
    canvas.paste(_c27, (631, 116), _c27)
except Exception:
    pass
layout["Filters"] = [631, 116, 811, 182]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_05_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-7/28_text_Categories.png
try:
    _c28 = get_crop(28, 187, 135)
    canvas.paste(_c28, (36, 383), _c28)
except Exception:
    pass
layout["Categories"] = [36, 383, 223, 518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_05_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-7/29_text_Show_all_categories.png
try:
    _c29 = get_crop(29, 516, 144)
    canvas.paste(_c29, (0, 518), _c29)
except Exception:
    pass
layout["Show_all_categories"] = [0, 518, 516, 662]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_05_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-7/30_text_Event_type.png
try:
    _c30 = get_crop(30, 298, 135)
    canvas.paste(_c30, (36, 829), _c30)
except Exception:
    pass
layout["Event_type"] = [36, 829, 334, 964]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_05_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-7/31_text_Show_all_event_types.png
try:
    _c31 = get_crop(31, 535, 144)
    canvas.paste(_c31, (0, 964), _c31)
except Exception:
    pass
layout["Show_all_event_types"] = [0, 964, 535, 1108]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_05_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-7/32_text_Languages.png
try:
    _c32 = get_crop(32, 210, 135)
    canvas.paste(_c32, (36, 1275), _c32)
except Exception:
    pass
layout["Languages"] = [36, 1275, 246, 1410]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_05_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-7/33_text_Show_all_languages.png
try:
    _c33 = get_crop(33, 511, 144)
    canvas.paste(_c33, (0, 1410), _c33)
except Exception:
    pass
layout["Show_all_languages"] = [0, 1410, 511, 1554]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_05_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-7/34_text_Price.png
try:
    _c34 = get_crop(34, 149, 63)
    canvas.paste(_c34, (45, 1613), _c34)
except Exception:
    pass
layout["Price"] = [45, 1613, 194, 1676]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_05_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-7/35_text_Only_free_events.png
try:
    _c35 = get_crop(35, 660, 144)
    canvas.paste(_c35, (54, 2024), _c35)
except Exception:
    pass
layout["Only_free_events"] = [54, 2024, 714, 2168]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_05_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-7/36_text_Sort_by.png
try:
    _c36 = get_crop(36, 206, 75)
    canvas.paste(_c36, (42, 1931), _c36)
except Exception:
    pass
layout["Sort_by"] = [42, 1931, 248, 2006]
