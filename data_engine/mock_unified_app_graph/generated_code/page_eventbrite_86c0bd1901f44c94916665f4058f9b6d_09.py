# page_id: page_eventbrite_86c0bd1901f44c94916665f4058f9b6d_09
# screenshot: 2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-11.png
# step_index: 9/11
# task: Open Eventbrite. Set the city to Los Angeles. Select the 'Food & Drink' category. What's the date of the first event that is not promoted?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw background and structural UI elements for the filters page

# Fill base background (page white)
draw.rectangle([(0, 0), canvas.size], fill="#FFFFFF")

# Status bar area (top subtle gray)
status_h = 80
draw.rectangle([(0, 0), (1440, status_h)], fill="#CFCFCF")

# Header / toolbar area (below status bar)
toolbar_top = status_h
toolbar_h = 84
toolbar_bottom = toolbar_top + toolbar_h
draw.rectangle([(0, toolbar_top), (1440, toolbar_bottom)], fill="#FFFFFF")

# Subtle bottom divider under toolbar
draw.line([(24, toolbar_bottom + 0.5), (1440 - 24, toolbar_bottom + 0.5)], fill="#E6E6F0", width=1)

# Section separators (subtle horizontal rules separating logical groups)
separators = [520, 964, 1410, 1610]
for y in separators:
    draw.line([(24, y + 0.5), (1440 - 24, y + 0.5)], fill="#F0F0F5", width=1)

# "Sort by" segmented control background and segments
seg_x0 = 36
seg_x1 = 1440 - 36
seg_y0 = 2004
seg_y1 = seg_y0 + 150
seg_radius = 12

# Outer rounded rect (subtle pale background + shadow)
draw.rounded_rectangle([(seg_x0, seg_y0), (seg_x1, seg_y1)], radius=seg_radius, fill="#ECEBF0")
# Inner left (selected) segment - lighter
left_margin = seg_x0 + 6
right_margin = seg_x0 + (seg_x1 - seg_x0) // 2 - 6
draw.rounded_rectangle([(left_margin, seg_y0 + 6), (right_margin, seg_y1 - 6)], radius=10, fill="#FFFFFF")
# Inner right (unselected) segment - slightly darker
left_unselected = right_margin + 2
right_unselected = seg_x1 - 6
draw.rounded_rectangle([(left_unselected, seg_y0 + 6), (right_unselected, seg_y1 - 6)], radius=10, fill="#E6E3EA")

# Very subtle inner divider between the two segments
divider_x = (seg_x0 + seg_x1) // 2
draw.line([(divider_x, seg_y0 + 10), (divider_x, seg_y1 - 10)], fill="#DDD9E0", width=1)

# Give segmented control a faint drop shadow line under it
draw.line([(seg_x0 + 4, seg_y1 + 4), (seg_x1 - 4, seg_y1 + 4)], fill="#E8E6EB", width=1)

# Content area subtle card banding behind the category/event type rows
# Draw light rounded panels to visually group the pill collections (but not drawing pills themselves)
group_boxes = [
    (24, 160, 1440 - 24, 560),   # Categories block area
    (24, 660, 1440 - 24, 1040),  # Event type block
    (24, 1120, 1440 - 24, 1480), # Languages block
]
for box in group_boxes:
    draw.rounded_rectangle([ (box[0], box[1]), (box[2], box[3]) ],
                           radius=14, outline=None, fill="#FFFFFF")

# Subtle separators/padding lines for Price and Only free events area
price_line_y = 1600
draw.line([(36, price_line_y), (1440 - 36, price_line_y)], fill="#F6F6F9", width=1)

# Footer top divider above the "Apply filters" control area (leave the actual button area for pasted element)
apply_divider_y = 2700
draw.line([(36, apply_divider_y), (1440 - 36, apply_divider_y)], fill="#E6E6F0", width=1)

# Small left rounded indicator next to header to suggest navigation area (background only)
nav_back_bg = [(24, toolbar_top + 20), (24 + 64, toolbar_top + 20 + 64)]
draw.rounded_rectangle(nav_back_bg, radius=12, fill="#FFFFFF", outline="#F0F0F5")

# End of structural drawing.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_09_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-11/00_icon_Food_Drink.png
try:
    _c0 = get_crop(0, 312, 144)
    canvas.paste(_c0, (512, 383), _c0)
except Exception:
    pass
layout["Food_&_Drink"] = [512, 383, 824, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_09_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-11/01_icon_Music.png
try:
    _c1 = get_crop(1, 187, 135)
    canvas.paste(_c1, (36, 383), _c1)
except Exception:
    pass
layout["Music"] = [36, 383, 223, 518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_09_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-11/02_icon_Community.png
try:
    _c2 = get_crop(2, 294, 144)
    canvas.paste(_c2, (848, 383), _c2)
except Exception:
    pass
layout["Community"] = [848, 383, 1142, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_09_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-11/03_icon_French.png
try:
    _c3 = get_crop(3, 205, 144)
    canvas.paste(_c3, (768, 1275), _c3)
except Exception:
    pass
layout["French"] = [768, 1275, 973, 1419]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_09_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-11/04_icon_Spanish.png
try:
    _c4 = get_crop(4, 225, 144)
    canvas.paste(_c4, (519, 1275), _c4)
except Exception:
    pass
layout["Spanish"] = [519, 1275, 744, 1419]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_09_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-11/05_icon_Business.png
try:
    _c5 = get_crop(5, 241, 135)
    canvas.paste(_c5, (247, 383), _c5)
except Exception:
    pass
layout["Business"] = [247, 383, 488, 518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_09_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-11/06_icon_Expo.png
try:
    _c6 = get_crop(6, 167, 144)
    canvas.paste(_c6, (614, 829), _c6)
except Exception:
    pass
layout["Expo"] = [614, 829, 781, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_09_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-11/07_icon_Italian.png
try:
    _c7 = get_crop(7, 191, 144)
    canvas.paste(_c7, (997, 1275), _c7)
except Exception:
    pass
layout["Italian"] = [997, 1275, 1188, 1419]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_09_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-11/08_icon_Seminar.png
try:
    _c8 = get_crop(8, 232, 144)
    canvas.paste(_c8, (358, 829), _c8)
except Exception:
    pass
layout["Seminar"] = [358, 829, 590, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_09_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-11/09_icon_Arts.png
try:
    _c9 = get_crop(9, 152, 144)
    canvas.paste(_c9, (1166, 383), _c9)
except Exception:
    pass
layout["Arts"] = [1166, 383, 1318, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_09_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-11/10_icon_Convention.png
try:
    _c10 = get_crop(10, 293, 144)
    canvas.paste(_c10, (805, 829), _c10)
except Exception:
    pass
layout["Convention"] = [805, 829, 1098, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_09_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-11/11_icon_Festival.png
try:
    _c11 = get_crop(11, 219, 144)
    canvas.paste(_c11, (1122, 829), _c11)
except Exception:
    pass
layout["Festival"] = [1122, 829, 1341, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_09_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-11/12_icon_English.png
try:
    _c12 = get_crop(12, 210, 135)
    canvas.paste(_c12, (36, 1275), _c12)
except Exception:
    pass
layout["English"] = [36, 1275, 246, 1410]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_09_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-11/13_icon_German.png
try:
    _c13 = get_crop(13, 225, 135)
    canvas.paste(_c13, (270, 1275), _c13)
except Exception:
    pass
layout["German"] = [270, 1275, 495, 1410]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_09_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-11/14_icon_Conference.png
try:
    _c14 = get_crop(14, 298, 135)
    canvas.paste(_c14, (36, 829), _c14)
except Exception:
    pass
layout["Conference"] = [36, 829, 334, 964]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_09_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-11/15_icon_Apply_filters_1.png
try:
    _c15 = get_crop(15, 1344, 144)
    canvas.paste(_c15, (48, 2768), _c15)
except Exception:
    pass
layout["Apply_filters_(1)"] = [48, 2768, 1392, 2912]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_09_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-11/16_icon_Date.png
try:
    _c16 = get_crop(16, 660, 144)
    canvas.paste(_c16, (726, 2024), _c16)
except Exception:
    pass
layout["Date"] = [726, 2024, 1386, 2168]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_09_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-11/17_icon_Relevance.png
try:
    _c17 = get_crop(17, 660, 144)
    canvas.paste(_c17, (54, 2024), _c17)
except Exception:
    pass
layout["Relevance"] = [54, 2024, 714, 2168]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_09_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-11/18_icon_Clear_all.png
try:
    _c18 = get_crop(18, 52, 68)
    canvas.paste(_c18, (1153, 1), _c18)
except Exception:
    pass
layout["Clear_all"] = [1153, 1, 1205, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_09_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-11/19_icon_7.14.png
try:
    _c19 = get_crop(19, 144, 144)
    canvas.paste(_c19, (12, 72), _c19)
except Exception:
    pass
layout["7.14"] = [12, 72, 156, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_09_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-11/20_icon_Clear_all.png
try:
    _c20 = get_crop(20, 97, 69)
    canvas.paste(_c20, (1211, 0), _c20)
except Exception:
    pass
layout["Clear_all"] = [1211, 0, 1308, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_09_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-11/21_icon_7.14.png
try:
    _c21 = get_crop(21, 60, 64)
    canvas.paste(_c21, (181, 1), _c21)
except Exception:
    pass
layout["7.14"] = [181, 1, 241, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_09_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-11/22_icon_icon_22.png
try:
    _c22 = get_crop(22, 64, 62)
    canvas.paste(_c22, (308, 3), _c22)
except Exception:
    pass
layout["icon_22"] = [308, 3, 372, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_09_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-11/23_icon_Clear_all.png
try:
    _c23 = get_crop(23, 53, 68)
    canvas.paste(_c23, (1319, 0), _c23)
except Exception:
    pass
layout["Clear_all"] = [1319, 0, 1372, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_09_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-11/24_icon_7.14.png
try:
    _c24 = get_crop(24, 63, 65)
    canvas.paste(_c24, (113, 1), _c24)
except Exception:
    pass
layout["7.14"] = [113, 1, 176, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_09_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-11/25_icon_Clear_all.png
try:
    _c25 = get_crop(25, 178, 144)
    canvas.paste(_c25, (1214, 72), _c25)
except Exception:
    pass
layout["Clear_all"] = [1214, 72, 1392, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_09_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-11/26_icon_icon_26.png
try:
    _c26 = get_crop(26, 51, 62)
    canvas.paste(_c26, (248, 2), _c26)
except Exception:
    pass
layout["icon_26"] = [248, 2, 299, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_09_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-11/27_icon_clickable_20.png
try:
    _c27 = get_crop(27, 144, 144)
    canvas.paste(_c27, (1248, 1729), _c27)
except Exception:
    pass
layout["clickable_20"] = [1248, 1729, 1392, 1873]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_09_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-11/28_text_Filters.png
try:
    _c28 = get_crop(28, 180, 66)
    canvas.paste(_c28, (631, 116), _c28)
except Exception:
    pass
layout["Filters"] = [631, 116, 811, 182]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_09_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-11/29_text_Categories.png
try:
    _c29 = get_crop(29, 187, 135)
    canvas.paste(_c29, (36, 383), _c29)
except Exception:
    pass
layout["Categories"] = [36, 383, 223, 518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_09_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-11/30_text_Show_all_categories.png
try:
    _c30 = get_crop(30, 516, 144)
    canvas.paste(_c30, (0, 518), _c30)
except Exception:
    pass
layout["Show_all_categories"] = [0, 518, 516, 662]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_09_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-11/31_text_Event_type.png
try:
    _c31 = get_crop(31, 298, 135)
    canvas.paste(_c31, (36, 829), _c31)
except Exception:
    pass
layout["Event_type"] = [36, 829, 334, 964]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_09_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-11/32_text_Show_all_event_types.png
try:
    _c32 = get_crop(32, 535, 144)
    canvas.paste(_c32, (0, 964), _c32)
except Exception:
    pass
layout["Show_all_event_types"] = [0, 964, 535, 1108]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_09_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-11/33_text_Languages.png
try:
    _c33 = get_crop(33, 210, 135)
    canvas.paste(_c33, (36, 1275), _c33)
except Exception:
    pass
layout["Languages"] = [36, 1275, 246, 1410]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_09_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-11/34_text_Show_all_languages.png
try:
    _c34 = get_crop(34, 511, 144)
    canvas.paste(_c34, (0, 1410), _c34)
except Exception:
    pass
layout["Show_all_languages"] = [0, 1410, 511, 1554]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_09_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-11/35_text_Price.png
try:
    _c35 = get_crop(35, 149, 63)
    canvas.paste(_c35, (45, 1613), _c35)
except Exception:
    pass
layout["Price"] = [45, 1613, 194, 1676]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_09_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-11/36_text_Only_free_events.png
try:
    _c36 = get_crop(36, 660, 144)
    canvas.paste(_c36, (54, 2024), _c36)
except Exception:
    pass
layout["Only_free_events"] = [54, 2024, 714, 2168]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_09_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-11/37_text_Sort_by.png
try:
    _c37 = get_crop(37, 206, 75)
    canvas.paste(_c37, (42, 1931), _c37)
except Exception:
    pass
layout["Sort_by"] = [42, 1931, 248, 2006]
