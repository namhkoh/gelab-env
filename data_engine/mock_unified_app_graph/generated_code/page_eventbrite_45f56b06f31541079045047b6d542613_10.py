# page_id: page_eventbrite_45f56b06f31541079045047b6d542613_10
# screenshot: 2024_4_23_19_27_45f56b06f31541079045047b6d542613-12.png
# step_index: 10/21
# task: Open Eventbrite. Search events 'Yoga session' in New York. Filter free events and set date from May 3 to May 6. What is the location of the first promoted event?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# fill overall background (slightly off-white to match app background)
draw.rectangle([(0, 0), (1440, 2960)], fill="#FBFCFE")

# status bar area (top)
status_h = 72
draw.rectangle([(0, 0), (1440, status_h)], fill="#C7C7C7")

# header area separator under the toolbar (subtle)
header_bottom = 180
draw.line([(24, header_bottom), (1440-24, header_bottom)], fill="#E6E6EA", width=1)

# Subtle shadow under header (very faint)
draw.line([(24, header_bottom+2), (1440-24, header_bottom+2)], fill="#F4F5F7", width=1)

# Section "cards" (rounded backgrounds behind groups)
# Categories card
draw.rounded_rectangle([(36, 170), (1404, 548)], radius=18, fill="#FBFDFF", outline="#ECEFF3", width=1)

# Event type card
draw.rounded_rectangle([(36, 700), (1404, 1028)], radius=18, fill="#FBFDFF", outline="#ECEFF3", width=1)

# Languages card
draw.rounded_rectangle([(36, 1160), (1404, 1488)], radius=18, fill="#FBFDFF", outline="#ECEFF3", width=1)

# Price / Free toggle area (card)
draw.rounded_rectangle([(36, 1520), (1404, 1880)], radius=18, fill="#FBFDFF", outline="#ECEFF3", width=1)

# Sort by area card (subtle background for the control region)
draw.rounded_rectangle([(36, 1888), (1404, 2060)], radius=14, fill="#FBFDFF", outline="#ECEFF3", width=1)

# Light separators between major sections (matching spacing from screenshot)
sep_color = "#F0F1F4"
draw.line([(24, 518), (1440-24, 518)], fill=sep_color, width=1)
draw.line([(24, 964), (1440-24, 964)], fill=sep_color, width=1)
draw.line([(24, 1410), (1440-24, 1410)], fill=sep_color, width=1)
draw.line([(24, 1613), (1440-24, 1613)], fill=sep_color, width=1)
draw.line([(24, 1931), (1440-24, 1931)], fill=sep_color, width=1)

# Very faint horizontal guide above bottom area (keeps bottom area distinct)
draw.line([(24, 2700), (1440-24, 2700)], fill="#F6F7FA", width=1)

# subtle rounded border frames around the grouped areas to add structure (thin)
outline_color = "#E8EAEE"
draw.rounded_rectangle([(30, 164), (1406, 556)], radius=20, outline=outline_color, width=1)
draw.rounded_rectangle([(30, 694), (1406, 1034)], radius=20, outline=outline_color, width=1)
draw.rounded_rectangle([(30, 1154), (1406, 1494)], radius=20, outline=outline_color, width=1)
draw.rounded_rectangle([(30, 1508), (1406, 1896)], radius=20, outline=outline_color, width=1)
draw.rounded_rectangle([(30, 1878), (1406, 2068)], radius=16, outline=outline_color, width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_10_2024_4_23_19_27_45f56b06f31541079045047b6d542613-12/00_icon_Food_Drink.png
try:
    _c0 = get_crop(0, 312, 144)
    canvas.paste(_c0, (512, 383), _c0)
except Exception:
    pass
layout["Food_&_Drink"] = [512, 383, 824, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_10_2024_4_23_19_27_45f56b06f31541079045047b6d542613-12/01_icon_Music.png
try:
    _c1 = get_crop(1, 187, 135)
    canvas.paste(_c1, (36, 383), _c1)
except Exception:
    pass
layout["Music"] = [36, 383, 223, 518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_10_2024_4_23_19_27_45f56b06f31541079045047b6d542613-12/02_icon_Community.png
try:
    _c2 = get_crop(2, 294, 144)
    canvas.paste(_c2, (848, 383), _c2)
except Exception:
    pass
layout["Community"] = [848, 383, 1142, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_10_2024_4_23_19_27_45f56b06f31541079045047b6d542613-12/03_icon_French.png
try:
    _c3 = get_crop(3, 205, 144)
    canvas.paste(_c3, (768, 1275), _c3)
except Exception:
    pass
layout["French"] = [768, 1275, 973, 1419]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_10_2024_4_23_19_27_45f56b06f31541079045047b6d542613-12/04_icon_Business.png
try:
    _c4 = get_crop(4, 241, 135)
    canvas.paste(_c4, (247, 383), _c4)
except Exception:
    pass
layout["Business"] = [247, 383, 488, 518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_10_2024_4_23_19_27_45f56b06f31541079045047b6d542613-12/05_icon_Spanish.png
try:
    _c5 = get_crop(5, 225, 144)
    canvas.paste(_c5, (519, 1275), _c5)
except Exception:
    pass
layout["Spanish"] = [519, 1275, 744, 1419]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_10_2024_4_23_19_27_45f56b06f31541079045047b6d542613-12/06_icon_Expo.png
try:
    _c6 = get_crop(6, 167, 144)
    canvas.paste(_c6, (614, 829), _c6)
except Exception:
    pass
layout["Expo"] = [614, 829, 781, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_10_2024_4_23_19_27_45f56b06f31541079045047b6d542613-12/07_icon_Italian.png
try:
    _c7 = get_crop(7, 191, 144)
    canvas.paste(_c7, (997, 1275), _c7)
except Exception:
    pass
layout["Italian"] = [997, 1275, 1188, 1419]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_10_2024_4_23_19_27_45f56b06f31541079045047b6d542613-12/08_icon_Seminar.png
try:
    _c8 = get_crop(8, 232, 144)
    canvas.paste(_c8, (358, 829), _c8)
except Exception:
    pass
layout["Seminar"] = [358, 829, 590, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_10_2024_4_23_19_27_45f56b06f31541079045047b6d542613-12/09_icon_Arts.png
try:
    _c9 = get_crop(9, 152, 144)
    canvas.paste(_c9, (1166, 383), _c9)
except Exception:
    pass
layout["Arts"] = [1166, 383, 1318, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_10_2024_4_23_19_27_45f56b06f31541079045047b6d542613-12/10_icon_Convention.png
try:
    _c10 = get_crop(10, 293, 144)
    canvas.paste(_c10, (805, 829), _c10)
except Exception:
    pass
layout["Convention"] = [805, 829, 1098, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_10_2024_4_23_19_27_45f56b06f31541079045047b6d542613-12/11_icon_Festival.png
try:
    _c11 = get_crop(11, 219, 144)
    canvas.paste(_c11, (1122, 829), _c11)
except Exception:
    pass
layout["Festival"] = [1122, 829, 1341, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_10_2024_4_23_19_27_45f56b06f31541079045047b6d542613-12/12_icon_German.png
try:
    _c12 = get_crop(12, 225, 135)
    canvas.paste(_c12, (270, 1275), _c12)
except Exception:
    pass
layout["German"] = [270, 1275, 495, 1410]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_10_2024_4_23_19_27_45f56b06f31541079045047b6d542613-12/13_icon_English.png
try:
    _c13 = get_crop(13, 210, 135)
    canvas.paste(_c13, (36, 1275), _c13)
except Exception:
    pass
layout["English"] = [36, 1275, 246, 1410]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_10_2024_4_23_19_27_45f56b06f31541079045047b6d542613-12/14_icon_Conference.png
try:
    _c14 = get_crop(14, 298, 135)
    canvas.paste(_c14, (36, 829), _c14)
except Exception:
    pass
layout["Conference"] = [36, 829, 334, 964]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_10_2024_4_23_19_27_45f56b06f31541079045047b6d542613-12/15_icon_Date.png
try:
    _c15 = get_crop(15, 660, 144)
    canvas.paste(_c15, (726, 2024), _c15)
except Exception:
    pass
layout["Date"] = [726, 2024, 1386, 2168]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_10_2024_4_23_19_27_45f56b06f31541079045047b6d542613-12/16_icon_Relevance.png
try:
    _c16 = get_crop(16, 660, 144)
    canvas.paste(_c16, (54, 2024), _c16)
except Exception:
    pass
layout["Relevance"] = [54, 2024, 714, 2168]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_10_2024_4_23_19_27_45f56b06f31541079045047b6d542613-12/17_icon_Clear_all.png
try:
    _c17 = get_crop(17, 51, 70)
    canvas.paste(_c17, (1153, 1), _c17)
except Exception:
    pass
layout["Clear_all"] = [1153, 1, 1204, 71]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_10_2024_4_23_19_27_45f56b06f31541079045047b6d542613-12/18_icon_Apply_filters.png
try:
    _c18 = get_crop(18, 1344, 144)
    canvas.paste(_c18, (48, 2768), _c18)
except Exception:
    pass
layout["Apply_filters"] = [48, 2768, 1392, 2912]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_10_2024_4_23_19_27_45f56b06f31541079045047b6d542613-12/19_icon_7.29.png
try:
    _c19 = get_crop(19, 61, 63)
    canvas.paste(_c19, (179, 2), _c19)
except Exception:
    pass
layout["7.29"] = [179, 2, 240, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_10_2024_4_23_19_27_45f56b06f31541079045047b6d542613-12/20_icon_Clear_all.png
try:
    _c20 = get_crop(20, 100, 70)
    canvas.paste(_c20, (1211, 0), _c20)
except Exception:
    pass
layout["Clear_all"] = [1211, 0, 1311, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_10_2024_4_23_19_27_45f56b06f31541079045047b6d542613-12/21_icon_7.29.png
try:
    _c21 = get_crop(21, 144, 144)
    canvas.paste(_c21, (12, 72), _c21)
except Exception:
    pass
layout["7.29"] = [12, 72, 156, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_10_2024_4_23_19_27_45f56b06f31541079045047b6d542613-12/22_icon_7.29.png
try:
    _c22 = get_crop(22, 65, 65)
    canvas.paste(_c22, (111, 1), _c22)
except Exception:
    pass
layout["7.29"] = [111, 1, 176, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_10_2024_4_23_19_27_45f56b06f31541079045047b6d542613-12/23_icon_icon_23.png
try:
    _c23 = get_crop(23, 66, 62)
    canvas.paste(_c23, (307, 3), _c23)
except Exception:
    pass
layout["icon_23"] = [307, 3, 373, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_10_2024_4_23_19_27_45f56b06f31541079045047b6d542613-12/24_icon_Clear_all.png
try:
    _c24 = get_crop(24, 52, 69)
    canvas.paste(_c24, (1320, 0), _c24)
except Exception:
    pass
layout["Clear_all"] = [1320, 0, 1372, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_10_2024_4_23_19_27_45f56b06f31541079045047b6d542613-12/25_icon_icon_25.png
try:
    _c25 = get_crop(25, 51, 62)
    canvas.paste(_c25, (248, 2), _c25)
except Exception:
    pass
layout["icon_25"] = [248, 2, 299, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_10_2024_4_23_19_27_45f56b06f31541079045047b6d542613-12/26_icon_Clear_all.png
try:
    _c26 = get_crop(26, 178, 144)
    canvas.paste(_c26, (1214, 72), _c26)
except Exception:
    pass
layout["Clear_all"] = [1214, 72, 1392, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_10_2024_4_23_19_27_45f56b06f31541079045047b6d542613-12/27_icon_clickable_20.png
try:
    _c27 = get_crop(27, 144, 144)
    canvas.paste(_c27, (1248, 1729), _c27)
except Exception:
    pass
layout["clickable_20"] = [1248, 1729, 1392, 1873]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_10_2024_4_23_19_27_45f56b06f31541079045047b6d542613-12/28_text_Filters.png
try:
    _c28 = get_crop(28, 180, 66)
    canvas.paste(_c28, (631, 116), _c28)
except Exception:
    pass
layout["Filters"] = [631, 116, 811, 182]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_10_2024_4_23_19_27_45f56b06f31541079045047b6d542613-12/29_text_Categories.png
try:
    _c29 = get_crop(29, 187, 135)
    canvas.paste(_c29, (36, 383), _c29)
except Exception:
    pass
layout["Categories"] = [36, 383, 223, 518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_10_2024_4_23_19_27_45f56b06f31541079045047b6d542613-12/30_text_Show_all_categories.png
try:
    _c30 = get_crop(30, 516, 144)
    canvas.paste(_c30, (0, 518), _c30)
except Exception:
    pass
layout["Show_all_categories"] = [0, 518, 516, 662]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_10_2024_4_23_19_27_45f56b06f31541079045047b6d542613-12/31_text_Event_type.png
try:
    _c31 = get_crop(31, 298, 135)
    canvas.paste(_c31, (36, 829), _c31)
except Exception:
    pass
layout["Event_type"] = [36, 829, 334, 964]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_10_2024_4_23_19_27_45f56b06f31541079045047b6d542613-12/32_text_Show_all_event_types.png
try:
    _c32 = get_crop(32, 535, 144)
    canvas.paste(_c32, (0, 964), _c32)
except Exception:
    pass
layout["Show_all_event_types"] = [0, 964, 535, 1108]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_10_2024_4_23_19_27_45f56b06f31541079045047b6d542613-12/33_text_Languages.png
try:
    _c33 = get_crop(33, 210, 135)
    canvas.paste(_c33, (36, 1275), _c33)
except Exception:
    pass
layout["Languages"] = [36, 1275, 246, 1410]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_10_2024_4_23_19_27_45f56b06f31541079045047b6d542613-12/34_text_Show_all_languages.png
try:
    _c34 = get_crop(34, 511, 144)
    canvas.paste(_c34, (0, 1410), _c34)
except Exception:
    pass
layout["Show_all_languages"] = [0, 1410, 511, 1554]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_10_2024_4_23_19_27_45f56b06f31541079045047b6d542613-12/35_text_Price.png
try:
    _c35 = get_crop(35, 149, 63)
    canvas.paste(_c35, (45, 1613), _c35)
except Exception:
    pass
layout["Price"] = [45, 1613, 194, 1676]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_10_2024_4_23_19_27_45f56b06f31541079045047b6d542613-12/36_text_Only_free_events.png
try:
    _c36 = get_crop(36, 660, 144)
    canvas.paste(_c36, (54, 2024), _c36)
except Exception:
    pass
layout["Only_free_events"] = [54, 2024, 714, 2168]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_10_2024_4_23_19_27_45f56b06f31541079045047b6d542613-12/37_text_Sort_by.png
try:
    _c37 = get_crop(37, 206, 75)
    canvas.paste(_c37, (42, 1931), _c37)
except Exception:
    pass
layout["Sort_by"] = [42, 1931, 248, 2006]
