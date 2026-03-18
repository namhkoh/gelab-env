# page_id: page_eventbrite_03837235ef8649c7821b415a8d3b0093_04
# screenshot: 2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-6.png
# step_index: 4/8
# task: Open Eventbrite. Locate the 'Conference' category. Filter the results to only show virtual events. Choose the first event from the results. What is the duration of this event?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw overall background (canvas already white, but ensure uniform fill)
draw.rectangle([(0, 0), (1440, 2960)], fill=(255, 255, 255))

# Top status bar / header areas
status_h = 120
header_h = 200

# Status bar (light neutral background)
draw.rectangle([(0, 0), (1440, status_h)], fill=(244, 245, 247))

# Header background (slightly different to separate from status)
draw.rectangle([(0, status_h), (1440, header_h)], fill=(255, 255, 255))

# Subtle bottom divider under header
draw.line([(24, header_h), (1416, header_h)], fill=(230, 231, 235), width=1)

# Section cards / group backgrounds (rounded rectangles)
card_fill = (250, 251, 253)
card_border = (232, 233, 236)
corner = 28

# Categories card (behind category pills)
draw.rounded_rectangle([(24, 260), (1416, 560)], radius=corner, fill=card_fill, outline=card_border, width=1)

# Event type card
draw.rounded_rectangle([(24, 700), (1416, 1000)], radius=corner, fill=card_fill, outline=card_border, width=1)

# Languages card
draw.rounded_rectangle([(24, 1120), (1416, 1440)], radius=corner, fill=card_fill, outline=card_border, width=1)

# Price / toggle card
draw.rounded_rectangle([(24, 1540), (1416, 1860)], radius=corner, fill=card_fill, outline=card_border, width=1)

# Sort-by card (background for segmented control)
seg_fill = (247, 248, 250)
seg_border = (220, 221, 225)
seg_top = 1920
seg_bottom = 2100
draw.rounded_rectangle([(48, seg_top), (1392, seg_bottom)], radius=20, fill=seg_fill, outline=seg_border, width=2)

# Subtle separators between major sections (light lines)
sep_color = (235, 236, 239)
draw.line([(24, 640), (1416, 640)], fill=sep_color, width=1)
draw.line([(24, 1080), (1416, 1080)], fill=sep_color, width=1)
draw.line([(24, 1500), (1416, 1500)], fill=sep_color, width=1)
draw.line([(24, 1900), (1416, 1900)], fill=sep_color, width=1)

# Soft shadow under header for depth
shadow_color = (240, 241, 243)
for i in range(4):
    draw.line([(24, header_h + i), (1416, header_h + i)], fill=shadow_color, width=1)

# Light page edge guides (very faint)
edge_color = (245, 246, 247)
draw.line([(24, header_h), (24, 2800)], fill=edge_color, width=1)
draw.line([(1416, header_h), (1416, 2800)], fill=edge_color, width=1)

# Bottom area: top divider above the final action button (do not draw the button itself)
apply_div_y = 2720
draw.line([(24, apply_div_y), (1416, apply_div_y)], fill=(229, 230, 234), width=2)

# Slight ambient vignette at very bottom to separate content from edge (subtle)
draw.rectangle([(0, 2860), (1440, 2960)], fill=(255, 255, 255, 10))

# Finished background + structural elements

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_04_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-6/00_icon_Food_Drink.png
try:
    _c0 = get_crop(0, 312, 144)
    canvas.paste(_c0, (512, 383), _c0)
except Exception:
    pass
layout["Food_&_Drink"] = [512, 383, 824, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_04_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-6/01_icon_Music.png
try:
    _c1 = get_crop(1, 187, 135)
    canvas.paste(_c1, (36, 383), _c1)
except Exception:
    pass
layout["Music"] = [36, 383, 223, 518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_04_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-6/02_icon_Conference.png
try:
    _c2 = get_crop(2, 298, 135)
    canvas.paste(_c2, (36, 829), _c2)
except Exception:
    pass
layout["Conference"] = [36, 829, 334, 964]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_04_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-6/03_icon_Community.png
try:
    _c3 = get_crop(3, 294, 144)
    canvas.paste(_c3, (848, 383), _c3)
except Exception:
    pass
layout["Community"] = [848, 383, 1142, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_04_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-6/04_icon_French.png
try:
    _c4 = get_crop(4, 205, 144)
    canvas.paste(_c4, (768, 1275), _c4)
except Exception:
    pass
layout["French"] = [768, 1275, 973, 1419]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_04_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-6/05_icon_Spanish.png
try:
    _c5 = get_crop(5, 225, 144)
    canvas.paste(_c5, (519, 1275), _c5)
except Exception:
    pass
layout["Spanish"] = [519, 1275, 744, 1419]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_04_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-6/06_icon_Business.png
try:
    _c6 = get_crop(6, 241, 135)
    canvas.paste(_c6, (247, 383), _c6)
except Exception:
    pass
layout["Business"] = [247, 383, 488, 518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_04_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-6/07_icon_Seminar.png
try:
    _c7 = get_crop(7, 232, 144)
    canvas.paste(_c7, (358, 829), _c7)
except Exception:
    pass
layout["Seminar"] = [358, 829, 590, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_04_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-6/08_icon_Expo.png
try:
    _c8 = get_crop(8, 167, 144)
    canvas.paste(_c8, (614, 829), _c8)
except Exception:
    pass
layout["Expo"] = [614, 829, 781, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_04_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-6/09_icon_Italian.png
try:
    _c9 = get_crop(9, 191, 144)
    canvas.paste(_c9, (997, 1275), _c9)
except Exception:
    pass
layout["Italian"] = [997, 1275, 1188, 1419]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_04_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-6/10_icon_Arts.png
try:
    _c10 = get_crop(10, 152, 144)
    canvas.paste(_c10, (1166, 383), _c10)
except Exception:
    pass
layout["Arts"] = [1166, 383, 1318, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_04_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-6/11_icon_Convention.png
try:
    _c11 = get_crop(11, 293, 144)
    canvas.paste(_c11, (805, 829), _c11)
except Exception:
    pass
layout["Convention"] = [805, 829, 1098, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_04_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-6/12_icon_Festival.png
try:
    _c12 = get_crop(12, 219, 144)
    canvas.paste(_c12, (1122, 829), _c12)
except Exception:
    pass
layout["Festival"] = [1122, 829, 1341, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_04_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-6/13_icon_English.png
try:
    _c13 = get_crop(13, 210, 135)
    canvas.paste(_c13, (36, 1275), _c13)
except Exception:
    pass
layout["English"] = [36, 1275, 246, 1410]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_04_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-6/14_icon_German.png
try:
    _c14 = get_crop(14, 225, 135)
    canvas.paste(_c14, (270, 1275), _c14)
except Exception:
    pass
layout["German"] = [270, 1275, 495, 1410]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_04_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-6/15_icon_Apply_filters_1.png
try:
    _c15 = get_crop(15, 1344, 144)
    canvas.paste(_c15, (48, 2768), _c15)
except Exception:
    pass
layout["Apply_filters_(1)"] = [48, 2768, 1392, 2912]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_04_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-6/16_icon_Date.png
try:
    _c16 = get_crop(16, 660, 144)
    canvas.paste(_c16, (726, 2024), _c16)
except Exception:
    pass
layout["Date"] = [726, 2024, 1386, 2168]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_04_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-6/17_icon_Relevance.png
try:
    _c17 = get_crop(17, 660, 144)
    canvas.paste(_c17, (54, 2024), _c17)
except Exception:
    pass
layout["Relevance"] = [54, 2024, 714, 2168]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_04_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-6/18_icon_4.41.png
try:
    _c18 = get_crop(18, 61, 65)
    canvas.paste(_c18, (179, 1), _c18)
except Exception:
    pass
layout["4.41"] = [179, 1, 240, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_04_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-6/19_icon_4.41.png
try:
    _c19 = get_crop(19, 144, 144)
    canvas.paste(_c19, (12, 72), _c19)
except Exception:
    pass
layout["4.41"] = [12, 72, 156, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_04_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-6/20_icon_4.41.png
try:
    _c20 = get_crop(20, 66, 66)
    canvas.paste(_c20, (110, 1), _c20)
except Exception:
    pass
layout["4.41"] = [110, 1, 176, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_04_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-6/21_icon_icon_21.png
try:
    _c21 = get_crop(21, 64, 62)
    canvas.paste(_c21, (308, 3), _c21)
except Exception:
    pass
layout["icon_21"] = [308, 3, 372, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_04_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-6/22_icon_Clear_all.png
try:
    _c22 = get_crop(22, 56, 67)
    canvas.paste(_c22, (1318, 0), _c22)
except Exception:
    pass
layout["Clear_all"] = [1318, 0, 1374, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_04_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-6/23_icon_Clear_all.png
try:
    _c23 = get_crop(23, 99, 65)
    canvas.paste(_c23, (1211, 0), _c23)
except Exception:
    pass
layout["Clear_all"] = [1211, 0, 1310, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_04_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-6/24_icon_icon_24.png
try:
    _c24 = get_crop(24, 51, 63)
    canvas.paste(_c24, (248, 2), _c24)
except Exception:
    pass
layout["icon_24"] = [248, 2, 299, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_04_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-6/25_icon_Toggle_to_filter_only_free_events.png
try:
    _c25 = get_crop(25, 144, 144)
    canvas.paste(_c25, (1248, 1729), _c25)
except Exception:
    pass
layout["Toggle_to_filter_only_fre"] = [1248, 1729, 1392, 1873]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_04_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-6/26_icon_Clear_all.png
try:
    _c26 = get_crop(26, 178, 144)
    canvas.paste(_c26, (1214, 72), _c26)
except Exception:
    pass
layout["Clear_all"] = [1214, 72, 1392, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_04_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-6/27_text_4.41.png
try:
    _c27 = get_crop(27, 87, 43)
    canvas.paste(_c27, (22, 15), _c27)
except Exception:
    pass
layout["4.41"] = [22, 15, 109, 58]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_04_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-6/28_text_Filters.png
try:
    _c28 = get_crop(28, 180, 66)
    canvas.paste(_c28, (631, 116), _c28)
except Exception:
    pass
layout["Filters"] = [631, 116, 811, 182]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_04_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-6/29_text_Categories.png
try:
    _c29 = get_crop(29, 187, 135)
    canvas.paste(_c29, (36, 383), _c29)
except Exception:
    pass
layout["Categories"] = [36, 383, 223, 518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_04_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-6/30_text_Show_all_categories.png
try:
    _c30 = get_crop(30, 516, 144)
    canvas.paste(_c30, (0, 518), _c30)
except Exception:
    pass
layout["Show_all_categories"] = [0, 518, 516, 662]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_04_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-6/31_text_Event_type.png
try:
    _c31 = get_crop(31, 298, 135)
    canvas.paste(_c31, (36, 829), _c31)
except Exception:
    pass
layout["Event_type"] = [36, 829, 334, 964]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_04_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-6/32_text_Show_all_event_types.png
try:
    _c32 = get_crop(32, 535, 144)
    canvas.paste(_c32, (0, 964), _c32)
except Exception:
    pass
layout["Show_all_event_types"] = [0, 964, 535, 1108]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_04_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-6/33_text_Languages.png
try:
    _c33 = get_crop(33, 210, 135)
    canvas.paste(_c33, (36, 1275), _c33)
except Exception:
    pass
layout["Languages"] = [36, 1275, 246, 1410]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_04_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-6/34_text_Show_all_languages.png
try:
    _c34 = get_crop(34, 511, 144)
    canvas.paste(_c34, (0, 1410), _c34)
except Exception:
    pass
layout["Show_all_languages"] = [0, 1410, 511, 1554]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_04_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-6/35_text_Price.png
try:
    _c35 = get_crop(35, 149, 63)
    canvas.paste(_c35, (45, 1613), _c35)
except Exception:
    pass
layout["Price"] = [45, 1613, 194, 1676]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_04_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-6/36_text_Only_free_events.png
try:
    _c36 = get_crop(36, 660, 144)
    canvas.paste(_c36, (54, 2024), _c36)
except Exception:
    pass
layout["Only_free_events"] = [54, 2024, 714, 2168]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_04_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-6/37_text_Sort_by.png
try:
    _c37 = get_crop(37, 206, 75)
    canvas.paste(_c37, (42, 1931), _c37)
except Exception:
    pass
layout["Sort_by"] = [42, 1931, 248, 2006]
