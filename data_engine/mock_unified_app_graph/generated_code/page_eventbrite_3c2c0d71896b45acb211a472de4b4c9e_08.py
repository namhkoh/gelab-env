# page_id: page_eventbrite_3c2c0d71896b45acb211a472de4b4c9e_08
# screenshot: 2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-10.png
# step_index: 8/15
# task: Open Eventbrite. Search free Health event in Los Angeles. Select the first one that is not promoted. Follow the organizer. Share to google keep notes.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw background fill (canvas already white, but ensure exact tone)
draw.rectangle([(0, 0), (1440, 2960)], fill="#ffffff")

# Status bar area (top ~72px) - light grey
draw.rectangle([(0, 0), (1440, 72)], fill="#d6d6d6")

# Header / toolbar area (below status bar)
header_top = 72
header_bottom = 180
draw.rectangle([(0, header_top), (1440, header_bottom)], fill="#ffffff")
# subtle divider/shadow under header
draw.line([(24, header_bottom), (1440-24, header_bottom)], fill="#e6e6ea", width=1)

# Section group soft cards (rounded backgrounds behind groups of chips)
card_left = 28
card_right = 1440 - 28
# Categories card (behind category chips)
cat_top = 240
cat_bottom = 680
draw.rounded_rectangle([(card_left, cat_top), (card_right, cat_bottom)],
                       radius=28, fill="#fbfdff", outline="#eceff3", width=1)
# faint shadow line under categories card
draw.line([(card_left+8, cat_bottom+2), (card_right-8, cat_bottom+2)], fill="#f0f1f4", width=2)

# Event type card
evt_top = 720
evt_bottom = 1040
draw.rounded_rectangle([(card_left, evt_top), (card_right, evt_bottom)],
                       radius=28, fill="#fbfdff", outline="#eceff3", width=1)
draw.line([(card_left+8, evt_bottom+2), (card_right-8, evt_bottom+2)], fill="#f0f1f4", width=2)

# Languages card
lang_top = 1220
lang_bottom = 1560
draw.rounded_rectangle([(card_left, lang_top), (card_right, lang_bottom)],
                       radius=28, fill="#fbfdff", outline="#eceff3", width=1)
draw.line([(card_left+8, lang_bottom+2), (card_right-8, lang_bottom+2)], fill="#f0f1f4", width=2)

# Price / toggle area separator (subtle)
sep_y = 1600
draw.line([(24, sep_y), (1440-24, sep_y)], fill="#f1f1f4", width=1)

# Sort by area background hint (light rounded container behind segmented control)
sort_top = 1900
sort_bottom = 2080
draw.rounded_rectangle([(48, sort_top), (1440-48, sort_bottom)],
                       radius=14, fill="#ffffff", outline="#e6e4ea", width=2)
# very light inner shadow to suggest elevation
draw.line([(50, sort_bottom), (1440-50, sort_bottom)], fill="#efecf1", width=3)

# Large pale bottom content area (keeps visual balance above apply button)
bottom_area_top = 2180
draw.rectangle([(0, bottom_area_top), (1440, 2768 - 20)], fill="#ffffff")

# Bottom safe area divider (above Apply filters button area)
draw.line([(24, 2768 - 40), (1440-24, 2768 - 40)], fill="#ece9ef", width=1)

# thin left/right margins guides (subtle)
draw.line([(24, header_bottom+12), (24, 2768 - 60)], fill="#ffffff", width=1)
draw.line([(1440-24, header_bottom+12), (1440-24, 2768 - 60)], fill="#ffffff", width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_08_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-10/00_icon_Food_Drink.png
try:
    _c0 = get_crop(0, 312, 144)
    canvas.paste(_c0, (512, 383), _c0)
except Exception:
    pass
layout["Food_&_Drink"] = [512, 383, 824, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_08_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-10/01_icon_Music.png
try:
    _c1 = get_crop(1, 187, 135)
    canvas.paste(_c1, (36, 383), _c1)
except Exception:
    pass
layout["Music"] = [36, 383, 223, 518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_08_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-10/02_icon_Community.png
try:
    _c2 = get_crop(2, 294, 144)
    canvas.paste(_c2, (848, 383), _c2)
except Exception:
    pass
layout["Community"] = [848, 383, 1142, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_08_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-10/03_icon_French.png
try:
    _c3 = get_crop(3, 205, 144)
    canvas.paste(_c3, (768, 1275), _c3)
except Exception:
    pass
layout["French"] = [768, 1275, 973, 1419]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_08_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-10/04_icon_Business.png
try:
    _c4 = get_crop(4, 241, 135)
    canvas.paste(_c4, (247, 383), _c4)
except Exception:
    pass
layout["Business"] = [247, 383, 488, 518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_08_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-10/05_icon_Spanish.png
try:
    _c5 = get_crop(5, 225, 144)
    canvas.paste(_c5, (519, 1275), _c5)
except Exception:
    pass
layout["Spanish"] = [519, 1275, 744, 1419]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_08_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-10/06_icon_Expo.png
try:
    _c6 = get_crop(6, 167, 144)
    canvas.paste(_c6, (614, 829), _c6)
except Exception:
    pass
layout["Expo"] = [614, 829, 781, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_08_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-10/07_icon_Seminar.png
try:
    _c7 = get_crop(7, 232, 144)
    canvas.paste(_c7, (358, 829), _c7)
except Exception:
    pass
layout["Seminar"] = [358, 829, 590, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_08_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-10/08_icon_Italian.png
try:
    _c8 = get_crop(8, 191, 144)
    canvas.paste(_c8, (997, 1275), _c8)
except Exception:
    pass
layout["Italian"] = [997, 1275, 1188, 1419]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_08_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-10/09_icon_Convention.png
try:
    _c9 = get_crop(9, 293, 144)
    canvas.paste(_c9, (805, 829), _c9)
except Exception:
    pass
layout["Convention"] = [805, 829, 1098, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_08_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-10/10_icon_Arts.png
try:
    _c10 = get_crop(10, 152, 144)
    canvas.paste(_c10, (1166, 383), _c10)
except Exception:
    pass
layout["Arts"] = [1166, 383, 1318, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_08_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-10/11_icon_German.png
try:
    _c11 = get_crop(11, 225, 135)
    canvas.paste(_c11, (270, 1275), _c11)
except Exception:
    pass
layout["German"] = [270, 1275, 495, 1410]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_08_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-10/12_icon_Festival.png
try:
    _c12 = get_crop(12, 219, 144)
    canvas.paste(_c12, (1122, 829), _c12)
except Exception:
    pass
layout["Festival"] = [1122, 829, 1341, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_08_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-10/13_icon_English.png
try:
    _c13 = get_crop(13, 210, 135)
    canvas.paste(_c13, (36, 1275), _c13)
except Exception:
    pass
layout["English"] = [36, 1275, 246, 1410]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_08_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-10/14_icon_Conference.png
try:
    _c14 = get_crop(14, 298, 135)
    canvas.paste(_c14, (36, 829), _c14)
except Exception:
    pass
layout["Conference"] = [36, 829, 334, 964]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_08_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-10/15_icon_Date.png
try:
    _c15 = get_crop(15, 660, 144)
    canvas.paste(_c15, (726, 2024), _c15)
except Exception:
    pass
layout["Date"] = [726, 2024, 1386, 2168]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_08_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-10/16_icon_Relevance.png
try:
    _c16 = get_crop(16, 660, 144)
    canvas.paste(_c16, (54, 2024), _c16)
except Exception:
    pass
layout["Relevance"] = [54, 2024, 714, 2168]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_08_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-10/17_icon_Clear_all.png
try:
    _c17 = get_crop(17, 51, 70)
    canvas.paste(_c17, (1153, 1), _c17)
except Exception:
    pass
layout["Clear_all"] = [1153, 1, 1204, 71]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_08_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-10/18_icon_Apply_filters.png
try:
    _c18 = get_crop(18, 1344, 144)
    canvas.paste(_c18, (48, 2768), _c18)
except Exception:
    pass
layout["Apply_filters"] = [48, 2768, 1392, 2912]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_08_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-10/19_icon_Clear_all.png
try:
    _c19 = get_crop(19, 100, 70)
    canvas.paste(_c19, (1211, 0), _c19)
except Exception:
    pass
layout["Clear_all"] = [1211, 0, 1311, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_08_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-10/20_icon_9.42.png
try:
    _c20 = get_crop(20, 144, 144)
    canvas.paste(_c20, (12, 72), _c20)
except Exception:
    pass
layout["9.42"] = [12, 72, 156, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_08_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-10/21_icon_9.42.png
try:
    _c21 = get_crop(21, 64, 63)
    canvas.paste(_c21, (176, 2), _c21)
except Exception:
    pass
layout["9.42"] = [176, 2, 240, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_08_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-10/22_icon_Clear_all.png
try:
    _c22 = get_crop(22, 52, 69)
    canvas.paste(_c22, (1320, 0), _c22)
except Exception:
    pass
layout["Clear_all"] = [1320, 0, 1372, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_08_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-10/23_icon_9.42.png
try:
    _c23 = get_crop(23, 57, 65)
    canvas.paste(_c23, (113, 1), _c23)
except Exception:
    pass
layout["9.42"] = [113, 1, 170, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_08_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-10/24_icon_icon_24.png
try:
    _c24 = get_crop(24, 59, 63)
    canvas.paste(_c24, (245, 2), _c24)
except Exception:
    pass
layout["icon_24"] = [245, 2, 304, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_08_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-10/25_icon_Clear_all.png
try:
    _c25 = get_crop(25, 178, 144)
    canvas.paste(_c25, (1214, 72), _c25)
except Exception:
    pass
layout["Clear_all"] = [1214, 72, 1392, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_08_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-10/26_icon_icon_26.png
try:
    _c26 = get_crop(26, 54, 61)
    canvas.paste(_c26, (314, 3), _c26)
except Exception:
    pass
layout["icon_26"] = [314, 3, 368, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_08_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-10/27_icon_clickable_20.png
try:
    _c27 = get_crop(27, 144, 144)
    canvas.paste(_c27, (1248, 1729), _c27)
except Exception:
    pass
layout["clickable_20"] = [1248, 1729, 1392, 1873]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_08_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-10/28_text_9.42.png
try:
    _c28 = get_crop(28, 91, 43)
    canvas.paste(_c28, (20, 15), _c28)
except Exception:
    pass
layout["9.42"] = [20, 15, 111, 58]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_08_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-10/29_text_Filters.png
try:
    _c29 = get_crop(29, 180, 66)
    canvas.paste(_c29, (631, 116), _c29)
except Exception:
    pass
layout["Filters"] = [631, 116, 811, 182]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_08_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-10/30_text_Categories.png
try:
    _c30 = get_crop(30, 187, 135)
    canvas.paste(_c30, (36, 383), _c30)
except Exception:
    pass
layout["Categories"] = [36, 383, 223, 518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_08_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-10/31_text_Show_all_categories.png
try:
    _c31 = get_crop(31, 516, 144)
    canvas.paste(_c31, (0, 518), _c31)
except Exception:
    pass
layout["Show_all_categories"] = [0, 518, 516, 662]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_08_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-10/32_text_Event_type.png
try:
    _c32 = get_crop(32, 298, 135)
    canvas.paste(_c32, (36, 829), _c32)
except Exception:
    pass
layout["Event_type"] = [36, 829, 334, 964]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_08_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-10/33_text_Show_all_event_types.png
try:
    _c33 = get_crop(33, 535, 144)
    canvas.paste(_c33, (0, 964), _c33)
except Exception:
    pass
layout["Show_all_event_types"] = [0, 964, 535, 1108]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_08_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-10/34_text_Languages.png
try:
    _c34 = get_crop(34, 210, 135)
    canvas.paste(_c34, (36, 1275), _c34)
except Exception:
    pass
layout["Languages"] = [36, 1275, 246, 1410]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_08_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-10/35_text_Show_all_languages.png
try:
    _c35 = get_crop(35, 511, 144)
    canvas.paste(_c35, (0, 1410), _c35)
except Exception:
    pass
layout["Show_all_languages"] = [0, 1410, 511, 1554]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_08_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-10/36_text_Price.png
try:
    _c36 = get_crop(36, 149, 63)
    canvas.paste(_c36, (45, 1613), _c36)
except Exception:
    pass
layout["Price"] = [45, 1613, 194, 1676]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_08_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-10/37_text_Only_free_events.png
try:
    _c37 = get_crop(37, 660, 144)
    canvas.paste(_c37, (54, 2024), _c37)
except Exception:
    pass
layout["Only_free_events"] = [54, 2024, 714, 2168]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_08_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-10/38_text_Sort_by.png
try:
    _c38 = get_crop(38, 206, 75)
    canvas.paste(_c38, (42, 1931), _c38)
except Exception:
    pass
layout["Sort_by"] = [42, 1931, 248, 2006]
