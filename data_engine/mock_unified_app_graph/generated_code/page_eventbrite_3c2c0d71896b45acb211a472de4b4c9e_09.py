# page_id: page_eventbrite_3c2c0d71896b45acb211a472de4b4c9e_09
# screenshot: 2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-11.png
# step_index: 9/15
# task: Open Eventbrite. Search free Health event in Los Angeles. Select the first one that is not promoted. Follow the organizer. Share to google keep notes.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Fill overall background
bg_color = (250, 250, 252)  # very light off-white
draw.rectangle([(0, 0), canvas.size], fill=bg_color)

# Status bar area (top)
status_h = 96
status_color = (222, 222, 222)  # light grey status bar
draw.rectangle([(0, 0), (canvas.width, status_h)], fill=status_color)

# subtle bottom divider under status bar / header area
header_bottom = 170
divider_color = (228, 229, 233)
draw.line([(0, header_bottom), (canvas.width, header_bottom)], fill=divider_color, width=1)

# Slightly darker hairline under status bar
draw.line([(0, status_h-1), (canvas.width, status_h-1)], fill=(200,200,200), width=1)

# Section separators between major groups
sep_color = (236, 237, 240)
separator_positions = [580, 1040, 1480, 1700, 2160]
for y in separator_positions:
    draw.line([(40, y), (canvas.width-40, y)], fill=sep_color, width=1)

# Soft shadow band above the sticky bottom area (where "Apply filters" sits)
top_of_footer = 2720
shadow_color = (245, 246, 248)
draw.rectangle([(0, top_of_footer), (canvas.width, top_of_footer+8)], fill=shadow_color)

# Very subtle inner frame lines to suggest grouped content areas
frame_color = (245, 245, 247)
# small rounded card hint under "Sort by" area (background hint only, not drawing any controls/text)
sort_hint_top = 1900
sort_hint_bottom = 2070
draw.rounded_rectangle([(48, sort_hint_top), (canvas.width-48, sort_hint_bottom)], radius=12, fill=frame_color, outline=(238,238,241))

# gentle vignette at very bottom to ground the sticky button area
draw.rectangle([(0, canvas.height-12), (canvas.width, canvas.height)], fill=(245,245,247))

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_09_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-11/00_icon_Food_Drink.png
try:
    _c0 = get_crop(0, 312, 144)
    canvas.paste(_c0, (512, 383), _c0)
except Exception:
    pass
layout["Food_&_Drink"] = [512, 383, 824, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_09_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-11/01_icon_Music.png
try:
    _c1 = get_crop(1, 187, 135)
    canvas.paste(_c1, (36, 383), _c1)
except Exception:
    pass
layout["Music"] = [36, 383, 223, 518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_09_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-11/02_icon_Community.png
try:
    _c2 = get_crop(2, 294, 144)
    canvas.paste(_c2, (848, 383), _c2)
except Exception:
    pass
layout["Community"] = [848, 383, 1142, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_09_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-11/03_icon_French.png
try:
    _c3 = get_crop(3, 205, 144)
    canvas.paste(_c3, (768, 1275), _c3)
except Exception:
    pass
layout["French"] = [768, 1275, 973, 1419]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_09_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-11/04_icon_Spanish.png
try:
    _c4 = get_crop(4, 225, 144)
    canvas.paste(_c4, (519, 1275), _c4)
except Exception:
    pass
layout["Spanish"] = [519, 1275, 744, 1419]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_09_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-11/05_icon_Business.png
try:
    _c5 = get_crop(5, 241, 135)
    canvas.paste(_c5, (247, 383), _c5)
except Exception:
    pass
layout["Business"] = [247, 383, 488, 518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_09_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-11/06_icon_Expo.png
try:
    _c6 = get_crop(6, 167, 144)
    canvas.paste(_c6, (614, 829), _c6)
except Exception:
    pass
layout["Expo"] = [614, 829, 781, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_09_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-11/07_icon_Seminar.png
try:
    _c7 = get_crop(7, 232, 144)
    canvas.paste(_c7, (358, 829), _c7)
except Exception:
    pass
layout["Seminar"] = [358, 829, 590, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_09_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-11/08_icon_Italian.png
try:
    _c8 = get_crop(8, 191, 144)
    canvas.paste(_c8, (997, 1275), _c8)
except Exception:
    pass
layout["Italian"] = [997, 1275, 1188, 1419]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_09_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-11/09_icon_Arts.png
try:
    _c9 = get_crop(9, 152, 144)
    canvas.paste(_c9, (1166, 383), _c9)
except Exception:
    pass
layout["Arts"] = [1166, 383, 1318, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_09_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-11/10_icon_Convention.png
try:
    _c10 = get_crop(10, 293, 144)
    canvas.paste(_c10, (805, 829), _c10)
except Exception:
    pass
layout["Convention"] = [805, 829, 1098, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_09_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-11/11_icon_Festival.png
try:
    _c11 = get_crop(11, 219, 144)
    canvas.paste(_c11, (1122, 829), _c11)
except Exception:
    pass
layout["Festival"] = [1122, 829, 1341, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_09_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-11/12_icon_German.png
try:
    _c12 = get_crop(12, 225, 135)
    canvas.paste(_c12, (270, 1275), _c12)
except Exception:
    pass
layout["German"] = [270, 1275, 495, 1410]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_09_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-11/13_icon_English.png
try:
    _c13 = get_crop(13, 210, 135)
    canvas.paste(_c13, (36, 1275), _c13)
except Exception:
    pass
layout["English"] = [36, 1275, 246, 1410]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_09_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-11/14_icon_Conference.png
try:
    _c14 = get_crop(14, 298, 135)
    canvas.paste(_c14, (36, 829), _c14)
except Exception:
    pass
layout["Conference"] = [36, 829, 334, 964]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_09_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-11/15_icon_Apply_filters_1.png
try:
    _c15 = get_crop(15, 1344, 144)
    canvas.paste(_c15, (48, 2768), _c15)
except Exception:
    pass
layout["Apply_filters_(1)"] = [48, 2768, 1392, 2912]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_09_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-11/16_icon_Date.png
try:
    _c16 = get_crop(16, 660, 144)
    canvas.paste(_c16, (726, 2024), _c16)
except Exception:
    pass
layout["Date"] = [726, 2024, 1386, 2168]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_09_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-11/17_icon_Relevance.png
try:
    _c17 = get_crop(17, 660, 144)
    canvas.paste(_c17, (54, 2024), _c17)
except Exception:
    pass
layout["Relevance"] = [54, 2024, 714, 2168]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_09_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-11/18_icon_Clear_all.png
try:
    _c18 = get_crop(18, 51, 69)
    canvas.paste(_c18, (1153, 1), _c18)
except Exception:
    pass
layout["Clear_all"] = [1153, 1, 1204, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_09_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-11/19_icon_Clear_all.png
try:
    _c19 = get_crop(19, 99, 67)
    canvas.paste(_c19, (1211, 1), _c19)
except Exception:
    pass
layout["Clear_all"] = [1211, 1, 1310, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_09_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-11/20_icon_9.42.png
try:
    _c20 = get_crop(20, 144, 144)
    canvas.paste(_c20, (12, 72), _c20)
except Exception:
    pass
layout["9.42"] = [12, 72, 156, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_09_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-11/21_icon_9.42.png
try:
    _c21 = get_crop(21, 64, 63)
    canvas.paste(_c21, (176, 2), _c21)
except Exception:
    pass
layout["9.42"] = [176, 2, 240, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_09_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-11/22_icon_Clear_all.png
try:
    _c22 = get_crop(22, 53, 66)
    canvas.paste(_c22, (1319, 1), _c22)
except Exception:
    pass
layout["Clear_all"] = [1319, 1, 1372, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_09_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-11/23_icon_Clear_all.png
try:
    _c23 = get_crop(23, 178, 144)
    canvas.paste(_c23, (1214, 72), _c23)
except Exception:
    pass
layout["Clear_all"] = [1214, 72, 1392, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_09_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-11/24_icon_9.42.png
try:
    _c24 = get_crop(24, 56, 65)
    canvas.paste(_c24, (114, 1), _c24)
except Exception:
    pass
layout["9.42"] = [114, 1, 170, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_09_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-11/25_icon_icon_25.png
try:
    _c25 = get_crop(25, 59, 63)
    canvas.paste(_c25, (245, 2), _c25)
except Exception:
    pass
layout["icon_25"] = [245, 2, 304, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_09_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-11/26_icon_clickable_20.png
try:
    _c26 = get_crop(26, 144, 144)
    canvas.paste(_c26, (1248, 1729), _c26)
except Exception:
    pass
layout["clickable_20"] = [1248, 1729, 1392, 1873]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_09_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-11/27_icon_icon_27.png
try:
    _c27 = get_crop(27, 54, 61)
    canvas.paste(_c27, (314, 3), _c27)
except Exception:
    pass
layout["icon_27"] = [314, 3, 368, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_09_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-11/28_text_9.42.png
try:
    _c28 = get_crop(28, 91, 43)
    canvas.paste(_c28, (20, 15), _c28)
except Exception:
    pass
layout["9.42"] = [20, 15, 111, 58]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_09_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-11/29_text_Filters.png
try:
    _c29 = get_crop(29, 180, 66)
    canvas.paste(_c29, (631, 116), _c29)
except Exception:
    pass
layout["Filters"] = [631, 116, 811, 182]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_09_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-11/30_text_Categories.png
try:
    _c30 = get_crop(30, 187, 135)
    canvas.paste(_c30, (36, 383), _c30)
except Exception:
    pass
layout["Categories"] = [36, 383, 223, 518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_09_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-11/31_text_Show_all_categories.png
try:
    _c31 = get_crop(31, 516, 144)
    canvas.paste(_c31, (0, 518), _c31)
except Exception:
    pass
layout["Show_all_categories"] = [0, 518, 516, 662]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_09_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-11/32_text_Event_type.png
try:
    _c32 = get_crop(32, 298, 135)
    canvas.paste(_c32, (36, 829), _c32)
except Exception:
    pass
layout["Event_type"] = [36, 829, 334, 964]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_09_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-11/33_text_Show_all_event_types.png
try:
    _c33 = get_crop(33, 535, 144)
    canvas.paste(_c33, (0, 964), _c33)
except Exception:
    pass
layout["Show_all_event_types"] = [0, 964, 535, 1108]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_09_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-11/34_text_Languages.png
try:
    _c34 = get_crop(34, 210, 135)
    canvas.paste(_c34, (36, 1275), _c34)
except Exception:
    pass
layout["Languages"] = [36, 1275, 246, 1410]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_09_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-11/35_text_Show_all_languages.png
try:
    _c35 = get_crop(35, 511, 144)
    canvas.paste(_c35, (0, 1410), _c35)
except Exception:
    pass
layout["Show_all_languages"] = [0, 1410, 511, 1554]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_09_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-11/36_text_Price.png
try:
    _c36 = get_crop(36, 149, 63)
    canvas.paste(_c36, (45, 1613), _c36)
except Exception:
    pass
layout["Price"] = [45, 1613, 194, 1676]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_09_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-11/37_text_Only_free_events.png
try:
    _c37 = get_crop(37, 660, 144)
    canvas.paste(_c37, (54, 2024), _c37)
except Exception:
    pass
layout["Only_free_events"] = [54, 2024, 714, 2168]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_09_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-11/38_text_Sort_by.png
try:
    _c38 = get_crop(38, 206, 75)
    canvas.paste(_c38, (42, 1931), _c38)
except Exception:
    pass
layout["Sort_by"] = [42, 1931, 248, 2006]
