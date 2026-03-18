# page_id: page_eventbrite_d77ba2e8a5b2402385411cd9fa60262a_05
# screenshot: 2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-7.png
# step_index: 5/8
# task: Open Eventbrite. Search for "Music". Filter only free events. Choose the first event. When is the date and timing?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Top status bar
draw.rectangle([(0, 0), (1440, 96)], fill="#BDBDBD")

# Main background (slightly off-white to match screenshot)
draw.rectangle([(0, 96), (1440, 2960)], fill="#FBFBFD")

# Header area (toolbar) - keep it white and add subtle bottom divider
header_top = 96
header_bottom = 184
draw.rectangle([(0, header_top), (1440, header_bottom)], fill="#FFFFFF")
draw.line([(24, header_bottom), (1416, header_bottom)], fill="#E6E4EA", width=2)

# Subtle inner shadow under header (thin)
draw.line([(24, header_bottom + 2), (1416, header_bottom + 2)], fill="#F3F1F6", width=1)

# Section separators (faint lines between content groups)
separator_color = "#F1EEF4"
separators = [680, 1000, 1440, 1700, 1960]
for y in separators:
    draw.line([(36, y), (1404, y)], fill=separator_color, width=1)

# Segmented control background (Sort by control)
seg_x1 = 30
seg_x2 = 1410
seg_y1 = 2000
seg_y2 = 2176
seg_radius = 20
draw.rounded_rectangle([(seg_x1, seg_y1), (seg_x2, seg_y2)], radius=seg_radius, fill="#ECEAF0", outline="#CFCBD6", width=3)

# Left (selected) segment - rounded on left, square on right by overlaying seam
mid_x = seg_x1 + (seg_x2 - seg_x1) // 2
draw.rounded_rectangle([(seg_x1, seg_y1), (mid_x, seg_y2)], radius=seg_radius, fill="#FFFFFF", outline=None)
# Draw seam to square the inner right edge of the selected segment
draw.rectangle([(mid_x-1, seg_y1+2), (mid_x+1, seg_y2-2)], fill="#ECEAF0")

# Subtle shadow line under segmented control
draw.line([(seg_x1+6, seg_y2+2), (seg_x2-6, seg_y2+2)], fill="#D6D2DA", width=2)

# Light content-area card suggestions (rounded faint blocks behind groups)
card_fill = "#FFFFFF"
card_outline = "#F0EDF3"
# Categories / Event type / Languages block backgrounds (very subtle)
blocks = [
    (24, 240, 1416, 660),
    (24, 700, 1416, 1040),
    (24, 1080, 1416, 1480),
]
for b in blocks:
    draw.rounded_rectangle([ (b[0], b[1]), (b[2], b[3]) ], radius=12, fill=card_fill, outline=card_outline, width=1)

# Gentle left/right page margins shadow (very subtle)
draw.rectangle([(12, 184), (24, 2960)], fill="#FBFBFD")
draw.rectangle([(1416, 184), (1428, 2960)], fill="#FBFBFD")

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_05_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-7/00_icon_Food_Drink.png
try:
    _c0 = get_crop(0, 312, 144)
    canvas.paste(_c0, (512, 383), _c0)
except Exception:
    pass
layout["Food_&_Drink"] = [512, 383, 824, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_05_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-7/01_icon_Music.png
try:
    _c1 = get_crop(1, 187, 135)
    canvas.paste(_c1, (36, 383), _c1)
except Exception:
    pass
layout["Music"] = [36, 383, 223, 518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_05_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-7/02_icon_Community.png
try:
    _c2 = get_crop(2, 294, 144)
    canvas.paste(_c2, (848, 383), _c2)
except Exception:
    pass
layout["Community"] = [848, 383, 1142, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_05_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-7/03_icon_French.png
try:
    _c3 = get_crop(3, 205, 144)
    canvas.paste(_c3, (768, 1275), _c3)
except Exception:
    pass
layout["French"] = [768, 1275, 973, 1419]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_05_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-7/04_icon_Business.png
try:
    _c4 = get_crop(4, 241, 135)
    canvas.paste(_c4, (247, 383), _c4)
except Exception:
    pass
layout["Business"] = [247, 383, 488, 518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_05_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-7/05_icon_Spanish.png
try:
    _c5 = get_crop(5, 225, 144)
    canvas.paste(_c5, (519, 1275), _c5)
except Exception:
    pass
layout["Spanish"] = [519, 1275, 744, 1419]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_05_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-7/06_icon_Expo.png
try:
    _c6 = get_crop(6, 167, 144)
    canvas.paste(_c6, (614, 829), _c6)
except Exception:
    pass
layout["Expo"] = [614, 829, 781, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_05_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-7/07_icon_Italian.png
try:
    _c7 = get_crop(7, 191, 144)
    canvas.paste(_c7, (997, 1275), _c7)
except Exception:
    pass
layout["Italian"] = [997, 1275, 1188, 1419]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_05_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-7/08_icon_Arts.png
try:
    _c8 = get_crop(8, 152, 144)
    canvas.paste(_c8, (1166, 383), _c8)
except Exception:
    pass
layout["Arts"] = [1166, 383, 1318, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_05_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-7/09_icon_Seminar.png
try:
    _c9 = get_crop(9, 232, 144)
    canvas.paste(_c9, (358, 829), _c9)
except Exception:
    pass
layout["Seminar"] = [358, 829, 590, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_05_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-7/10_icon_Convention.png
try:
    _c10 = get_crop(10, 293, 144)
    canvas.paste(_c10, (805, 829), _c10)
except Exception:
    pass
layout["Convention"] = [805, 829, 1098, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_05_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-7/11_icon_Festival.png
try:
    _c11 = get_crop(11, 219, 144)
    canvas.paste(_c11, (1122, 829), _c11)
except Exception:
    pass
layout["Festival"] = [1122, 829, 1341, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_05_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-7/12_icon_German.png
try:
    _c12 = get_crop(12, 225, 135)
    canvas.paste(_c12, (270, 1275), _c12)
except Exception:
    pass
layout["German"] = [270, 1275, 495, 1410]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_05_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-7/13_icon_English.png
try:
    _c13 = get_crop(13, 210, 135)
    canvas.paste(_c13, (36, 1275), _c13)
except Exception:
    pass
layout["English"] = [36, 1275, 246, 1410]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_05_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-7/14_icon_Conference.png
try:
    _c14 = get_crop(14, 298, 135)
    canvas.paste(_c14, (36, 829), _c14)
except Exception:
    pass
layout["Conference"] = [36, 829, 334, 964]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_05_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-7/15_icon_Date.png
try:
    _c15 = get_crop(15, 660, 144)
    canvas.paste(_c15, (726, 2024), _c15)
except Exception:
    pass
layout["Date"] = [726, 2024, 1386, 2168]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_05_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-7/16_icon_Relevance.png
try:
    _c16 = get_crop(16, 660, 144)
    canvas.paste(_c16, (54, 2024), _c16)
except Exception:
    pass
layout["Relevance"] = [54, 2024, 714, 2168]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_05_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-7/17_icon_Apply_filters.png
try:
    _c17 = get_crop(17, 1344, 144)
    canvas.paste(_c17, (48, 2768), _c17)
except Exception:
    pass
layout["Apply_filters"] = [48, 2768, 1392, 2912]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_05_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-7/18_icon_6.49.png
try:
    _c18 = get_crop(18, 144, 144)
    canvas.paste(_c18, (12, 72), _c18)
except Exception:
    pass
layout["6.49"] = [12, 72, 156, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_05_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-7/19_icon_6.49.png
try:
    _c19 = get_crop(19, 61, 64)
    canvas.paste(_c19, (179, 2), _c19)
except Exception:
    pass
layout["6.49"] = [179, 2, 240, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_05_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-7/20_icon_6.49.png
try:
    _c20 = get_crop(20, 66, 66)
    canvas.paste(_c20, (110, 1), _c20)
except Exception:
    pass
layout["6.49"] = [110, 1, 176, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_05_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-7/21_icon_icon_21.png
try:
    _c21 = get_crop(21, 66, 62)
    canvas.paste(_c21, (307, 3), _c21)
except Exception:
    pass
layout["icon_21"] = [307, 3, 373, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_05_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-7/22_icon_Clear_all.png
try:
    _c22 = get_crop(22, 55, 69)
    canvas.paste(_c22, (1319, 0), _c22)
except Exception:
    pass
layout["Clear_all"] = [1319, 0, 1374, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_05_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-7/23_icon_icon_23.png
try:
    _c23 = get_crop(23, 51, 63)
    canvas.paste(_c23, (248, 2), _c23)
except Exception:
    pass
layout["icon_23"] = [248, 2, 299, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_05_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-7/24_icon_Clear_all.png
try:
    _c24 = get_crop(24, 99, 70)
    canvas.paste(_c24, (1211, 0), _c24)
except Exception:
    pass
layout["Clear_all"] = [1211, 0, 1310, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_05_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-7/25_icon_clickable_20.png
try:
    _c25 = get_crop(25, 144, 144)
    canvas.paste(_c25, (1248, 1729), _c25)
except Exception:
    pass
layout["clickable_20"] = [1248, 1729, 1392, 1873]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_05_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-7/26_icon_Clear_all.png
try:
    _c26 = get_crop(26, 178, 144)
    canvas.paste(_c26, (1214, 72), _c26)
except Exception:
    pass
layout["Clear_all"] = [1214, 72, 1392, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_05_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-7/27_text_Filters.png
try:
    _c27 = get_crop(27, 180, 66)
    canvas.paste(_c27, (631, 116), _c27)
except Exception:
    pass
layout["Filters"] = [631, 116, 811, 182]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_05_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-7/28_text_Categories.png
try:
    _c28 = get_crop(28, 187, 135)
    canvas.paste(_c28, (36, 383), _c28)
except Exception:
    pass
layout["Categories"] = [36, 383, 223, 518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_05_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-7/29_text_Show_all_categories.png
try:
    _c29 = get_crop(29, 516, 144)
    canvas.paste(_c29, (0, 518), _c29)
except Exception:
    pass
layout["Show_all_categories"] = [0, 518, 516, 662]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_05_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-7/30_text_Event_type.png
try:
    _c30 = get_crop(30, 298, 135)
    canvas.paste(_c30, (36, 829), _c30)
except Exception:
    pass
layout["Event_type"] = [36, 829, 334, 964]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_05_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-7/31_text_Show_all_event_types.png
try:
    _c31 = get_crop(31, 535, 144)
    canvas.paste(_c31, (0, 964), _c31)
except Exception:
    pass
layout["Show_all_event_types"] = [0, 964, 535, 1108]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_05_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-7/32_text_Languages.png
try:
    _c32 = get_crop(32, 210, 135)
    canvas.paste(_c32, (36, 1275), _c32)
except Exception:
    pass
layout["Languages"] = [36, 1275, 246, 1410]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_05_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-7/33_text_Show_all_languages.png
try:
    _c33 = get_crop(33, 511, 144)
    canvas.paste(_c33, (0, 1410), _c33)
except Exception:
    pass
layout["Show_all_languages"] = [0, 1410, 511, 1554]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_05_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-7/34_text_Price.png
try:
    _c34 = get_crop(34, 149, 63)
    canvas.paste(_c34, (45, 1613), _c34)
except Exception:
    pass
layout["Price"] = [45, 1613, 194, 1676]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_05_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-7/35_text_Only_free_events.png
try:
    _c35 = get_crop(35, 660, 144)
    canvas.paste(_c35, (54, 2024), _c35)
except Exception:
    pass
layout["Only_free_events"] = [54, 2024, 714, 2168]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_05_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-7/36_text_Sort_by.png
try:
    _c36 = get_crop(36, 206, 75)
    canvas.paste(_c36, (42, 1931), _c36)
except Exception:
    pass
layout["Sort_by"] = [42, 1931, 248, 2006]
