# page_id: page_eventbrite_b2798d8b10cc4118ab8cf6648f8a4077_04
# screenshot: 2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-6.png
# step_index: 4/12
# task: Open Eventbrite. Search Music event in New York. Select the first one. Record its location and time in Google Keep Notes. Add it to Favorites.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw UI background and structure for the given canvas
# Available: canvas (PIL Image 1440x2960), draw (ImageDraw), font_sm/font_md/font_lg/font_xl

# Base background (ensure clean white)
draw.rectangle((0, 0, 1440, 2960), fill="#FFFFFF")

# Status bar area (top)
status_h = 96
draw.rectangle((0, 0, 1440, status_h), fill="#E0E0E0")

# Header / toolbar area (below status bar)
header_top = status_h
header_bottom = 200
draw.rectangle((0, header_top, 1440, header_bottom), fill="#FFFFFF")
# subtle divider under header
draw.line((24, header_bottom, 1440 - 24, header_bottom), fill="#EDEAF0", width=2)

# Subtle vertical margins for content
left_margin = 48
right_margin = 1440 - 48

# Section separators (light lines between major groups)
separators = [580, 1030, 1480, 1790, 1968]
for y in separators:
    draw.line((left_margin, y, right_margin, y), fill="#F3F2F6", width=1)

# Segmented "Sort by" container background (rounded outer container)
seg_top = 2008
seg_bottom = 2168
seg_rect = (left_margin, seg_top, right_margin, seg_bottom)
draw.rounded_rectangle(seg_rect, radius=18, fill="#F6F4F8", outline="#DDD9E0", width=2)

# Add a subtle inner shadow beneath the segmented control to suggest elevation
shadow_top = seg_bottom
shadow_bottom = seg_bottom + 6
draw.rectangle((left_margin + 6, shadow_top, right_margin - 6, shadow_bottom), fill="#EFEAF0")

# "Only free events" toggle area: draw a faint separator box area (background hint)
toggle_top = 1860
toggle_bottom = 2060
toggle_rect = (left_margin, toggle_top, right_margin, toggle_bottom)
# very light background panel (keeps space for toggle switch and label which will be pasted later)
draw.rectangle(toggle_rect, fill="#FFFFFF")

# Main content area remains white; but draw faint grouping card backgrounds for visual structure:
# Categories/Event type/Languages sections get soft grouping shadows (behind chips)
group_boxes = [
    (left_margin - 8, 320, right_margin + 8, 560),   # Categories area
    (left_margin - 8, 760, right_margin + 8, 1030),  # Event type area
    (left_margin - 8, 1180, right_margin + 8, 1480), # Languages area
]
for box in group_boxes:
    # subtle very-light background to separate groups from the page
    draw.rounded_rectangle(box, radius=14, fill="#FFFFFF", outline="#F4F3F6", width=1)

# Price area (simple placeholder background to define the section region)
price_box = (left_margin - 8, 1560, right_margin + 8, 1820)
draw.rounded_rectangle(price_box, radius=12, fill="#FFFFFF", outline="#F4F3F6", width=1)

# Apply filters button background at bottom (rounded rectangle with border)
apply_top = 2768
apply_height = 144
apply_rect = (left_margin, apply_top, right_margin, apply_top + apply_height)
# slight shadow above button
draw.rectangle((left_margin + 6, apply_top - 6, right_margin - 6, apply_top - 2), fill="#EEE9EE")
# button background and border
draw.rounded_rectangle(apply_rect, radius=12, fill="#FFFFFF", outline="#BFB6C1", width=4)

# Bottom safe area shadow to separate from content
draw.rectangle((0, 2940, 1440, 2960), fill="#FFFFFF")

# Final thin separators near bottom to match screen structure
draw.line((left_margin, apply_top - 24, right_margin, apply_top - 24), fill="#F0EDF2", width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_04_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-6/00_icon_Music.png
try:
    _c0 = get_crop(0, 187, 135)
    canvas.paste(_c0, (36, 383), _c0)
except Exception:
    pass
layout["Music"] = [36, 383, 223, 518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_04_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-6/01_icon_Food_Drink.png
try:
    _c1 = get_crop(1, 312, 144)
    canvas.paste(_c1, (512, 383), _c1)
except Exception:
    pass
layout["Food_&_Drink"] = [512, 383, 824, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_04_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-6/02_icon_French.png
try:
    _c2 = get_crop(2, 205, 144)
    canvas.paste(_c2, (768, 1275), _c2)
except Exception:
    pass
layout["French"] = [768, 1275, 973, 1419]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_04_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-6/03_icon_Community.png
try:
    _c3 = get_crop(3, 294, 144)
    canvas.paste(_c3, (848, 383), _c3)
except Exception:
    pass
layout["Community"] = [848, 383, 1142, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_04_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-6/04_icon_Business.png
try:
    _c4 = get_crop(4, 241, 135)
    canvas.paste(_c4, (247, 383), _c4)
except Exception:
    pass
layout["Business"] = [247, 383, 488, 518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_04_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-6/05_icon_Spanish.png
try:
    _c5 = get_crop(5, 225, 144)
    canvas.paste(_c5, (519, 1275), _c5)
except Exception:
    pass
layout["Spanish"] = [519, 1275, 744, 1419]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_04_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-6/06_icon_Expo.png
try:
    _c6 = get_crop(6, 167, 144)
    canvas.paste(_c6, (614, 829), _c6)
except Exception:
    pass
layout["Expo"] = [614, 829, 781, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_04_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-6/07_icon_Seminar.png
try:
    _c7 = get_crop(7, 232, 144)
    canvas.paste(_c7, (358, 829), _c7)
except Exception:
    pass
layout["Seminar"] = [358, 829, 590, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_04_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-6/08_icon_Italian.png
try:
    _c8 = get_crop(8, 191, 144)
    canvas.paste(_c8, (997, 1275), _c8)
except Exception:
    pass
layout["Italian"] = [997, 1275, 1188, 1419]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_04_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-6/09_icon_Arts.png
try:
    _c9 = get_crop(9, 152, 144)
    canvas.paste(_c9, (1166, 383), _c9)
except Exception:
    pass
layout["Arts"] = [1166, 383, 1318, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_04_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-6/10_icon_Convention.png
try:
    _c10 = get_crop(10, 293, 144)
    canvas.paste(_c10, (805, 829), _c10)
except Exception:
    pass
layout["Convention"] = [805, 829, 1098, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_04_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-6/11_icon_Festival.png
try:
    _c11 = get_crop(11, 219, 144)
    canvas.paste(_c11, (1122, 829), _c11)
except Exception:
    pass
layout["Festival"] = [1122, 829, 1341, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_04_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-6/12_icon_German.png
try:
    _c12 = get_crop(12, 225, 135)
    canvas.paste(_c12, (270, 1275), _c12)
except Exception:
    pass
layout["German"] = [270, 1275, 495, 1410]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_04_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-6/13_icon_English.png
try:
    _c13 = get_crop(13, 210, 135)
    canvas.paste(_c13, (36, 1275), _c13)
except Exception:
    pass
layout["English"] = [36, 1275, 246, 1410]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_04_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-6/14_icon_Conference.png
try:
    _c14 = get_crop(14, 298, 135)
    canvas.paste(_c14, (36, 829), _c14)
except Exception:
    pass
layout["Conference"] = [36, 829, 334, 964]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_04_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-6/15_icon_Apply_filters_1.png
try:
    _c15 = get_crop(15, 1344, 144)
    canvas.paste(_c15, (48, 2768), _c15)
except Exception:
    pass
layout["Apply_filters_(1)"] = [48, 2768, 1392, 2912]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_04_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-6/16_icon_Date.png
try:
    _c16 = get_crop(16, 660, 144)
    canvas.paste(_c16, (726, 2024), _c16)
except Exception:
    pass
layout["Date"] = [726, 2024, 1386, 2168]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_04_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-6/17_icon_Relevance.png
try:
    _c17 = get_crop(17, 660, 144)
    canvas.paste(_c17, (54, 2024), _c17)
except Exception:
    pass
layout["Relevance"] = [54, 2024, 714, 2168]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_04_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-6/18_icon_9_18.png
try:
    _c18 = get_crop(18, 144, 144)
    canvas.paste(_c18, (12, 72), _c18)
except Exception:
    pass
layout["9:18"] = [12, 72, 156, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_04_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-6/19_icon_9_18.png
try:
    _c19 = get_crop(19, 64, 64)
    canvas.paste(_c19, (176, 1), _c19)
except Exception:
    pass
layout["9:18"] = [176, 1, 240, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_04_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-6/20_icon_Clear_all.png
try:
    _c20 = get_crop(20, 99, 65)
    canvas.paste(_c20, (1211, 0), _c20)
except Exception:
    pass
layout["Clear_all"] = [1211, 0, 1310, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_04_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-6/21_icon_Clear_all.png
try:
    _c21 = get_crop(21, 56, 67)
    canvas.paste(_c21, (1317, 0), _c21)
except Exception:
    pass
layout["Clear_all"] = [1317, 0, 1373, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_04_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-6/22_icon_9_18.png
try:
    _c22 = get_crop(22, 57, 64)
    canvas.paste(_c22, (114, 1), _c22)
except Exception:
    pass
layout["9:18"] = [114, 1, 171, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_04_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-6/23_icon_icon_23.png
try:
    _c23 = get_crop(23, 59, 63)
    canvas.paste(_c23, (245, 2), _c23)
except Exception:
    pass
layout["icon_23"] = [245, 2, 304, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_04_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-6/24_icon_Clear_all.png
try:
    _c24 = get_crop(24, 178, 144)
    canvas.paste(_c24, (1214, 72), _c24)
except Exception:
    pass
layout["Clear_all"] = [1214, 72, 1392, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_04_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-6/25_icon_icon_25.png
try:
    _c25 = get_crop(25, 55, 61)
    canvas.paste(_c25, (314, 3), _c25)
except Exception:
    pass
layout["icon_25"] = [314, 3, 369, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_04_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-6/26_icon_clickable_20.png
try:
    _c26 = get_crop(26, 144, 144)
    canvas.paste(_c26, (1248, 1729), _c26)
except Exception:
    pass
layout["clickable_20"] = [1248, 1729, 1392, 1873]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_04_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-6/27_text_9_18.png
try:
    _c27 = get_crop(27, 94, 45)
    canvas.paste(_c27, (17, 15), _c27)
except Exception:
    pass
layout["9:18"] = [17, 15, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_04_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-6/28_text_Filters.png
try:
    _c28 = get_crop(28, 180, 66)
    canvas.paste(_c28, (631, 116), _c28)
except Exception:
    pass
layout["Filters"] = [631, 116, 811, 182]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_04_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-6/29_text_Categories.png
try:
    _c29 = get_crop(29, 187, 135)
    canvas.paste(_c29, (36, 383), _c29)
except Exception:
    pass
layout["Categories"] = [36, 383, 223, 518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_04_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-6/30_text_Show_all_categories.png
try:
    _c30 = get_crop(30, 516, 144)
    canvas.paste(_c30, (0, 518), _c30)
except Exception:
    pass
layout["Show_all_categories"] = [0, 518, 516, 662]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_04_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-6/31_text_Event_type.png
try:
    _c31 = get_crop(31, 298, 135)
    canvas.paste(_c31, (36, 829), _c31)
except Exception:
    pass
layout["Event_type"] = [36, 829, 334, 964]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_04_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-6/32_text_Show_all_event_types.png
try:
    _c32 = get_crop(32, 535, 144)
    canvas.paste(_c32, (0, 964), _c32)
except Exception:
    pass
layout["Show_all_event_types"] = [0, 964, 535, 1108]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_04_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-6/33_text_Languages.png
try:
    _c33 = get_crop(33, 210, 135)
    canvas.paste(_c33, (36, 1275), _c33)
except Exception:
    pass
layout["Languages"] = [36, 1275, 246, 1410]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_04_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-6/34_text_Show_all_languages.png
try:
    _c34 = get_crop(34, 511, 144)
    canvas.paste(_c34, (0, 1410), _c34)
except Exception:
    pass
layout["Show_all_languages"] = [0, 1410, 511, 1554]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_04_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-6/35_text_Price.png
try:
    _c35 = get_crop(35, 149, 63)
    canvas.paste(_c35, (45, 1613), _c35)
except Exception:
    pass
layout["Price"] = [45, 1613, 194, 1676]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_04_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-6/36_text_Only_free_events.png
try:
    _c36 = get_crop(36, 660, 144)
    canvas.paste(_c36, (54, 2024), _c36)
except Exception:
    pass
layout["Only_free_events"] = [54, 2024, 714, 2168]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_04_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-6/37_text_Sort_by.png
try:
    _c37 = get_crop(37, 206, 75)
    canvas.paste(_c37, (42, 1931), _c37)
except Exception:
    pass
layout["Sort_by"] = [42, 1931, 248, 2006]
