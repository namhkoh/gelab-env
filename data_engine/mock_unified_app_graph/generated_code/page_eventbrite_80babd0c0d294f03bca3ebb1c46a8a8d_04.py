# page_id: page_eventbrite_80babd0c0d294f03bca3ebb1c46a8a8d_04
# screenshot: 2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-6.png
# step_index: 4/8
# task: Open Eventbrite. Search Art event in New York. Select the second one. Record its location and time in Google Keep Notes. Follow the organizer.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# draw overall background
draw.rectangle((0, 0, 1440, 2960), fill=(255, 255, 255))

# status bar (top area with system icons)
status_h = 72
draw.rectangle((0, 0, 1440, status_h), fill=(224, 224, 224))

# header / toolbar area below status bar
header_y0 = status_h
header_y1 = 156
draw.rectangle((0, header_y0, 1440, header_y1), fill=(255, 255, 255))
# subtle divider / shadow under header
draw.line((36, header_y1, 1404, header_y1), fill=(235, 235, 238), width=2)
draw.line((36, header_y1+2, 1404, header_y1+2), fill=(245, 245, 246), width=1)

# subtle large content separators between main sections
sep_color = (245, 245, 248)
seps = [700, 1020, 1460, 1820, 2720]
for y in seps:
    draw.line((36, y, 1404, y), fill=sep_color, width=1)

# light grouped background for the "Sort by" control area (so detected controls will be pasted on top)
sort_bg_bbox = (36, 1880, 1404, 2108)
draw.rounded_rectangle(sort_bg_bbox, radius=18, fill=(250, 250, 252), outline=(225, 221, 228), width=2)

# faint card-like background behind the major filter area (keeps center visually distinct)
filters_card_bbox = (24, 180, 1416, 2680)
draw.rounded_rectangle(filters_card_bbox, radius=8, fill=(255, 255, 255), outline=None)

# very subtle bottom separation above the sticky apply bar (apply bar will be pasted on top)
draw.line((36, 2720, 1404, 2720), fill=(236, 236, 240), width=2)

# small decorative left/right edge shadows to add depth (very subtle)
edge_shadow_color = (245, 245, 247)
draw.rectangle((0, header_y1, 12, 1400), fill=edge_shadow_color)
draw.rectangle((1428, header_y1, 1440, 1400), fill=edge_shadow_color)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_04_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-6/00_icon_Food_Drink.png
try:
    _c0 = get_crop(0, 312, 144)
    canvas.paste(_c0, (512, 383), _c0)
except Exception:
    pass
layout["Food_&_Drink"] = [512, 383, 824, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_04_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-6/01_icon_Music.png
try:
    _c1 = get_crop(1, 187, 135)
    canvas.paste(_c1, (36, 383), _c1)
except Exception:
    pass
layout["Music"] = [36, 383, 223, 518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_04_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-6/02_icon_Community.png
try:
    _c2 = get_crop(2, 294, 144)
    canvas.paste(_c2, (848, 383), _c2)
except Exception:
    pass
layout["Community"] = [848, 383, 1142, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_04_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-6/03_icon_French.png
try:
    _c3 = get_crop(3, 205, 144)
    canvas.paste(_c3, (768, 1275), _c3)
except Exception:
    pass
layout["French"] = [768, 1275, 973, 1419]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_04_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-6/04_icon_Spanish.png
try:
    _c4 = get_crop(4, 225, 144)
    canvas.paste(_c4, (519, 1275), _c4)
except Exception:
    pass
layout["Spanish"] = [519, 1275, 744, 1419]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_04_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-6/05_icon_Arts.png
try:
    _c5 = get_crop(5, 152, 144)
    canvas.paste(_c5, (1166, 383), _c5)
except Exception:
    pass
layout["Arts"] = [1166, 383, 1318, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_04_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-6/06_icon_Business.png
try:
    _c6 = get_crop(6, 241, 135)
    canvas.paste(_c6, (247, 383), _c6)
except Exception:
    pass
layout["Business"] = [247, 383, 488, 518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_04_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-6/07_icon_Expo.png
try:
    _c7 = get_crop(7, 167, 144)
    canvas.paste(_c7, (614, 829), _c7)
except Exception:
    pass
layout["Expo"] = [614, 829, 781, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_04_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-6/08_icon_Seminar.png
try:
    _c8 = get_crop(8, 232, 144)
    canvas.paste(_c8, (358, 829), _c8)
except Exception:
    pass
layout["Seminar"] = [358, 829, 590, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_04_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-6/09_icon_Italian.png
try:
    _c9 = get_crop(9, 191, 144)
    canvas.paste(_c9, (997, 1275), _c9)
except Exception:
    pass
layout["Italian"] = [997, 1275, 1188, 1419]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_04_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-6/10_icon_Convention.png
try:
    _c10 = get_crop(10, 293, 144)
    canvas.paste(_c10, (805, 829), _c10)
except Exception:
    pass
layout["Convention"] = [805, 829, 1098, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_04_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-6/11_icon_Festival.png
try:
    _c11 = get_crop(11, 219, 144)
    canvas.paste(_c11, (1122, 829), _c11)
except Exception:
    pass
layout["Festival"] = [1122, 829, 1341, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_04_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-6/12_icon_German.png
try:
    _c12 = get_crop(12, 225, 135)
    canvas.paste(_c12, (270, 1275), _c12)
except Exception:
    pass
layout["German"] = [270, 1275, 495, 1410]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_04_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-6/13_icon_English.png
try:
    _c13 = get_crop(13, 210, 135)
    canvas.paste(_c13, (36, 1275), _c13)
except Exception:
    pass
layout["English"] = [36, 1275, 246, 1410]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_04_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-6/14_icon_Conference.png
try:
    _c14 = get_crop(14, 298, 135)
    canvas.paste(_c14, (36, 829), _c14)
except Exception:
    pass
layout["Conference"] = [36, 829, 334, 964]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_04_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-6/15_icon_Apply_filters_1.png
try:
    _c15 = get_crop(15, 1344, 144)
    canvas.paste(_c15, (48, 2768), _c15)
except Exception:
    pass
layout["Apply_filters_(1)"] = [48, 2768, 1392, 2912]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_04_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-6/16_icon_Date.png
try:
    _c16 = get_crop(16, 660, 144)
    canvas.paste(_c16, (726, 2024), _c16)
except Exception:
    pass
layout["Date"] = [726, 2024, 1386, 2168]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_04_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-6/17_icon_Relevance.png
try:
    _c17 = get_crop(17, 660, 144)
    canvas.paste(_c17, (54, 2024), _c17)
except Exception:
    pass
layout["Relevance"] = [54, 2024, 714, 2168]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_04_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-6/18_icon_9.25.png
try:
    _c18 = get_crop(18, 144, 144)
    canvas.paste(_c18, (12, 72), _c18)
except Exception:
    pass
layout["9.25"] = [12, 72, 156, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_04_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-6/19_icon_9.25.png
try:
    _c19 = get_crop(19, 64, 63)
    canvas.paste(_c19, (176, 2), _c19)
except Exception:
    pass
layout["9.25"] = [176, 2, 240, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_04_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-6/20_icon_Clear_all.png
try:
    _c20 = get_crop(20, 99, 65)
    canvas.paste(_c20, (1211, 0), _c20)
except Exception:
    pass
layout["Clear_all"] = [1211, 0, 1310, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_04_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-6/21_icon_Clear_all.png
try:
    _c21 = get_crop(21, 56, 67)
    canvas.paste(_c21, (1317, 0), _c21)
except Exception:
    pass
layout["Clear_all"] = [1317, 0, 1373, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_04_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-6/22_icon_9.25.png
try:
    _c22 = get_crop(22, 57, 65)
    canvas.paste(_c22, (113, 1), _c22)
except Exception:
    pass
layout["9.25"] = [113, 1, 170, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_04_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-6/23_icon_icon_23.png
try:
    _c23 = get_crop(23, 59, 63)
    canvas.paste(_c23, (245, 2), _c23)
except Exception:
    pass
layout["icon_23"] = [245, 2, 304, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_04_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-6/24_icon_icon_24.png
try:
    _c24 = get_crop(24, 54, 61)
    canvas.paste(_c24, (314, 3), _c24)
except Exception:
    pass
layout["icon_24"] = [314, 3, 368, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_04_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-6/25_icon_Clear_all.png
try:
    _c25 = get_crop(25, 178, 144)
    canvas.paste(_c25, (1214, 72), _c25)
except Exception:
    pass
layout["Clear_all"] = [1214, 72, 1392, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_04_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-6/26_icon_clickable_20.png
try:
    _c26 = get_crop(26, 144, 144)
    canvas.paste(_c26, (1248, 1729), _c26)
except Exception:
    pass
layout["clickable_20"] = [1248, 1729, 1392, 1873]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_04_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-6/27_text_Filters.png
try:
    _c27 = get_crop(27, 180, 66)
    canvas.paste(_c27, (631, 116), _c27)
except Exception:
    pass
layout["Filters"] = [631, 116, 811, 182]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_04_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-6/28_text_Categories.png
try:
    _c28 = get_crop(28, 187, 135)
    canvas.paste(_c28, (36, 383), _c28)
except Exception:
    pass
layout["Categories"] = [36, 383, 223, 518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_04_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-6/29_text_Show_all_categories.png
try:
    _c29 = get_crop(29, 516, 144)
    canvas.paste(_c29, (0, 518), _c29)
except Exception:
    pass
layout["Show_all_categories"] = [0, 518, 516, 662]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_04_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-6/30_text_Event_type.png
try:
    _c30 = get_crop(30, 298, 135)
    canvas.paste(_c30, (36, 829), _c30)
except Exception:
    pass
layout["Event_type"] = [36, 829, 334, 964]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_04_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-6/31_text_Show_all_event_types.png
try:
    _c31 = get_crop(31, 535, 144)
    canvas.paste(_c31, (0, 964), _c31)
except Exception:
    pass
layout["Show_all_event_types"] = [0, 964, 535, 1108]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_04_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-6/32_text_Languages.png
try:
    _c32 = get_crop(32, 210, 135)
    canvas.paste(_c32, (36, 1275), _c32)
except Exception:
    pass
layout["Languages"] = [36, 1275, 246, 1410]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_04_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-6/33_text_Show_all_languages.png
try:
    _c33 = get_crop(33, 511, 144)
    canvas.paste(_c33, (0, 1410), _c33)
except Exception:
    pass
layout["Show_all_languages"] = [0, 1410, 511, 1554]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_04_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-6/34_text_Price.png
try:
    _c34 = get_crop(34, 149, 63)
    canvas.paste(_c34, (45, 1613), _c34)
except Exception:
    pass
layout["Price"] = [45, 1613, 194, 1676]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_04_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-6/35_text_Only_free_events.png
try:
    _c35 = get_crop(35, 660, 144)
    canvas.paste(_c35, (54, 2024), _c35)
except Exception:
    pass
layout["Only_free_events"] = [54, 2024, 714, 2168]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_04_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-6/36_text_Sort_by.png
try:
    _c36 = get_crop(36, 206, 75)
    canvas.paste(_c36, (42, 1931), _c36)
except Exception:
    pass
layout["Sort_by"] = [42, 1931, 248, 2006]
