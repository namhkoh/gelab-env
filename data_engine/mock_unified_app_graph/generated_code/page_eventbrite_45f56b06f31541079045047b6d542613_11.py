# page_id: page_eventbrite_45f56b06f31541079045047b6d542613_11
# screenshot: 2024_4_23_19_27_45f56b06f31541079045047b6d542613-13.png
# step_index: 11/21
# task: Open Eventbrite. Search events 'Yoga session' in New York. Filter free events and set date from May 3 to May 6. What is the location of the first promoted event?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Background fill (dominant off-white)
draw.rectangle([(0, 0), (1440, 2960)], fill=(250, 250, 252))

# Status bar area (top ~72px)
status_h = 72
draw.rectangle([(0, 0), (1440, status_h)], fill=(190, 190, 190))

# Header / toolbar area below status bar
header_top = status_h
header_bottom = 160
draw.rectangle([(0, header_top), (1440, header_bottom)], fill=(255, 255, 255))
# subtle bottom divider / shadow for header
draw.line([(36, header_bottom), (1404, header_bottom)], fill=(230, 228, 235), width=1)
# faint thin line slightly above to simulate subtle elevation
draw.line([(36, header_bottom-2), (1404, header_bottom-2)], fill=(245, 245, 246), width=1)

# Utility: function to draw large rounded section backgrounds (subtle)
def section_bg(x1, y1, x2, y2, radius=18, fill=(245,245,247), outline=(235,233,238)):
    draw.rounded_rectangle([(x1, y1), (x2, y2)], radius=radius, fill=fill, outline=outline, width=1)

# Categories section background
section_bg(24, 180, 1416, 620, radius=20, fill=(255,255,255), outline=(245,244,246))
# Event type section background
section_bg(24, 660, 1416, 1040, radius=20, fill=(255,255,255), outline=(245,244,246))
# Languages section background
section_bg(24, 1160, 1416, 1500, radius=20, fill=(255,255,255), outline=(245,244,246))
# Price / Free toggle area background
section_bg(24, 1560, 1416, 1740, radius=16, fill=(255,255,255), outline=(245,244,246))
# Sort by area background (card behind segmented control)
section_bg(24, 1880, 1416, 2140, radius=18, fill=(248,247,250), outline=(235,233,238))

# Horizontal separators between logical groups (light)
seps = [620, 1040, 1500, 1740, 2140]
for y in seps:
    draw.line([(36, y), (1404, y)], fill=(240, 239, 242), width=1)

# Subtle left margin guide lines (very light) to mimic UI alignment (non-intrusive)
draw.line([(36, header_bottom+8), (36, 2700)], fill=(250,250,251), width=1)
draw.line([(1404, header_bottom+8), (1404, 2700)], fill=(250,250,251), width=1)

# Top of bottom action area shadow (above the "Apply filters" control which will be pasted)
apply_top = 2720
draw.rectangle([(24, apply_top-6), (1416, apply_top)], fill=(245,245,247))

# Final subtle overall vignette edges (very light) to match screenshot feel
edge_strip = 8
draw.rectangle([(0, 2960-edge_strip), (1440, 2960)], fill=(250,250,252))
draw.rectangle([(0, 0), (edge_strip, 2960)], fill=(250,250,252))
draw.rectangle([(1440-edge_strip, 0), (1440, 2960)], fill=(250,250,252))

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_11_2024_4_23_19_27_45f56b06f31541079045047b6d542613-13/00_icon_Food_Drink.png
try:
    _c0 = get_crop(0, 312, 144)
    canvas.paste(_c0, (512, 383), _c0)
except Exception:
    pass
layout["Food_&_Drink"] = [512, 383, 824, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_11_2024_4_23_19_27_45f56b06f31541079045047b6d542613-13/01_icon_Music.png
try:
    _c1 = get_crop(1, 187, 135)
    canvas.paste(_c1, (36, 383), _c1)
except Exception:
    pass
layout["Music"] = [36, 383, 223, 518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_11_2024_4_23_19_27_45f56b06f31541079045047b6d542613-13/02_icon_Community.png
try:
    _c2 = get_crop(2, 294, 144)
    canvas.paste(_c2, (848, 383), _c2)
except Exception:
    pass
layout["Community"] = [848, 383, 1142, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_11_2024_4_23_19_27_45f56b06f31541079045047b6d542613-13/03_icon_French.png
try:
    _c3 = get_crop(3, 205, 144)
    canvas.paste(_c3, (768, 1275), _c3)
except Exception:
    pass
layout["French"] = [768, 1275, 973, 1419]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_11_2024_4_23_19_27_45f56b06f31541079045047b6d542613-13/04_icon_Spanish.png
try:
    _c4 = get_crop(4, 225, 144)
    canvas.paste(_c4, (519, 1275), _c4)
except Exception:
    pass
layout["Spanish"] = [519, 1275, 744, 1419]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_11_2024_4_23_19_27_45f56b06f31541079045047b6d542613-13/05_icon_Business.png
try:
    _c5 = get_crop(5, 241, 135)
    canvas.paste(_c5, (247, 383), _c5)
except Exception:
    pass
layout["Business"] = [247, 383, 488, 518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_11_2024_4_23_19_27_45f56b06f31541079045047b6d542613-13/06_icon_Expo.png
try:
    _c6 = get_crop(6, 167, 144)
    canvas.paste(_c6, (614, 829), _c6)
except Exception:
    pass
layout["Expo"] = [614, 829, 781, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_11_2024_4_23_19_27_45f56b06f31541079045047b6d542613-13/07_icon_Italian.png
try:
    _c7 = get_crop(7, 191, 144)
    canvas.paste(_c7, (997, 1275), _c7)
except Exception:
    pass
layout["Italian"] = [997, 1275, 1188, 1419]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_11_2024_4_23_19_27_45f56b06f31541079045047b6d542613-13/08_icon_Seminar.png
try:
    _c8 = get_crop(8, 232, 144)
    canvas.paste(_c8, (358, 829), _c8)
except Exception:
    pass
layout["Seminar"] = [358, 829, 590, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_11_2024_4_23_19_27_45f56b06f31541079045047b6d542613-13/09_icon_Arts.png
try:
    _c9 = get_crop(9, 152, 144)
    canvas.paste(_c9, (1166, 383), _c9)
except Exception:
    pass
layout["Arts"] = [1166, 383, 1318, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_11_2024_4_23_19_27_45f56b06f31541079045047b6d542613-13/10_icon_Convention.png
try:
    _c10 = get_crop(10, 293, 144)
    canvas.paste(_c10, (805, 829), _c10)
except Exception:
    pass
layout["Convention"] = [805, 829, 1098, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_11_2024_4_23_19_27_45f56b06f31541079045047b6d542613-13/11_icon_Festival.png
try:
    _c11 = get_crop(11, 219, 144)
    canvas.paste(_c11, (1122, 829), _c11)
except Exception:
    pass
layout["Festival"] = [1122, 829, 1341, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_11_2024_4_23_19_27_45f56b06f31541079045047b6d542613-13/12_icon_English.png
try:
    _c12 = get_crop(12, 210, 135)
    canvas.paste(_c12, (36, 1275), _c12)
except Exception:
    pass
layout["English"] = [36, 1275, 246, 1410]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_11_2024_4_23_19_27_45f56b06f31541079045047b6d542613-13/13_icon_German.png
try:
    _c13 = get_crop(13, 225, 135)
    canvas.paste(_c13, (270, 1275), _c13)
except Exception:
    pass
layout["German"] = [270, 1275, 495, 1410]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_11_2024_4_23_19_27_45f56b06f31541079045047b6d542613-13/14_icon_Conference.png
try:
    _c14 = get_crop(14, 298, 135)
    canvas.paste(_c14, (36, 829), _c14)
except Exception:
    pass
layout["Conference"] = [36, 829, 334, 964]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_11_2024_4_23_19_27_45f56b06f31541079045047b6d542613-13/15_icon_Apply_filters_1.png
try:
    _c15 = get_crop(15, 1344, 144)
    canvas.paste(_c15, (48, 2768), _c15)
except Exception:
    pass
layout["Apply_filters_(1)"] = [48, 2768, 1392, 2912]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_11_2024_4_23_19_27_45f56b06f31541079045047b6d542613-13/16_icon_Date.png
try:
    _c16 = get_crop(16, 660, 144)
    canvas.paste(_c16, (726, 2024), _c16)
except Exception:
    pass
layout["Date"] = [726, 2024, 1386, 2168]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_11_2024_4_23_19_27_45f56b06f31541079045047b6d542613-13/17_icon_Relevance.png
try:
    _c17 = get_crop(17, 660, 144)
    canvas.paste(_c17, (54, 2024), _c17)
except Exception:
    pass
layout["Relevance"] = [54, 2024, 714, 2168]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_11_2024_4_23_19_27_45f56b06f31541079045047b6d542613-13/18_icon_Clear_all.png
try:
    _c18 = get_crop(18, 52, 68)
    canvas.paste(_c18, (1153, 1), _c18)
except Exception:
    pass
layout["Clear_all"] = [1153, 1, 1205, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_11_2024_4_23_19_27_45f56b06f31541079045047b6d542613-13/19_icon_7.29.png
try:
    _c19 = get_crop(19, 60, 63)
    canvas.paste(_c19, (180, 2), _c19)
except Exception:
    pass
layout["7.29"] = [180, 2, 240, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_11_2024_4_23_19_27_45f56b06f31541079045047b6d542613-13/20_icon_Clear_all.png
try:
    _c20 = get_crop(20, 99, 67)
    canvas.paste(_c20, (1211, 1), _c20)
except Exception:
    pass
layout["Clear_all"] = [1211, 1, 1310, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_11_2024_4_23_19_27_45f56b06f31541079045047b6d542613-13/21_icon_7.29.png
try:
    _c21 = get_crop(21, 144, 144)
    canvas.paste(_c21, (12, 72), _c21)
except Exception:
    pass
layout["7.29"] = [12, 72, 156, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_11_2024_4_23_19_27_45f56b06f31541079045047b6d542613-13/22_icon_7.29.png
try:
    _c22 = get_crop(22, 65, 65)
    canvas.paste(_c22, (111, 1), _c22)
except Exception:
    pass
layout["7.29"] = [111, 1, 176, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_11_2024_4_23_19_27_45f56b06f31541079045047b6d542613-13/23_icon_icon_23.png
try:
    _c23 = get_crop(23, 64, 62)
    canvas.paste(_c23, (308, 3), _c23)
except Exception:
    pass
layout["icon_23"] = [308, 3, 372, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_11_2024_4_23_19_27_45f56b06f31541079045047b6d542613-13/24_icon_Clear_all.png
try:
    _c24 = get_crop(24, 53, 66)
    canvas.paste(_c24, (1319, 1), _c24)
except Exception:
    pass
layout["Clear_all"] = [1319, 1, 1372, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_11_2024_4_23_19_27_45f56b06f31541079045047b6d542613-13/25_icon_icon_25.png
try:
    _c25 = get_crop(25, 51, 61)
    canvas.paste(_c25, (248, 3), _c25)
except Exception:
    pass
layout["icon_25"] = [248, 3, 299, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_11_2024_4_23_19_27_45f56b06f31541079045047b6d542613-13/26_icon_Clear_all.png
try:
    _c26 = get_crop(26, 178, 144)
    canvas.paste(_c26, (1214, 72), _c26)
except Exception:
    pass
layout["Clear_all"] = [1214, 72, 1392, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_11_2024_4_23_19_27_45f56b06f31541079045047b6d542613-13/27_icon_clickable_20.png
try:
    _c27 = get_crop(27, 144, 144)
    canvas.paste(_c27, (1248, 1729), _c27)
except Exception:
    pass
layout["clickable_20"] = [1248, 1729, 1392, 1873]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_11_2024_4_23_19_27_45f56b06f31541079045047b6d542613-13/28_text_Filters.png
try:
    _c28 = get_crop(28, 180, 66)
    canvas.paste(_c28, (631, 116), _c28)
except Exception:
    pass
layout["Filters"] = [631, 116, 811, 182]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_11_2024_4_23_19_27_45f56b06f31541079045047b6d542613-13/29_text_Categories.png
try:
    _c29 = get_crop(29, 187, 135)
    canvas.paste(_c29, (36, 383), _c29)
except Exception:
    pass
layout["Categories"] = [36, 383, 223, 518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_11_2024_4_23_19_27_45f56b06f31541079045047b6d542613-13/30_text_Show_all_categories.png
try:
    _c30 = get_crop(30, 516, 144)
    canvas.paste(_c30, (0, 518), _c30)
except Exception:
    pass
layout["Show_all_categories"] = [0, 518, 516, 662]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_11_2024_4_23_19_27_45f56b06f31541079045047b6d542613-13/31_text_Event_type.png
try:
    _c31 = get_crop(31, 298, 135)
    canvas.paste(_c31, (36, 829), _c31)
except Exception:
    pass
layout["Event_type"] = [36, 829, 334, 964]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_11_2024_4_23_19_27_45f56b06f31541079045047b6d542613-13/32_text_Show_all_event_types.png
try:
    _c32 = get_crop(32, 535, 144)
    canvas.paste(_c32, (0, 964), _c32)
except Exception:
    pass
layout["Show_all_event_types"] = [0, 964, 535, 1108]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_11_2024_4_23_19_27_45f56b06f31541079045047b6d542613-13/33_text_Languages.png
try:
    _c33 = get_crop(33, 210, 135)
    canvas.paste(_c33, (36, 1275), _c33)
except Exception:
    pass
layout["Languages"] = [36, 1275, 246, 1410]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_11_2024_4_23_19_27_45f56b06f31541079045047b6d542613-13/34_text_Show_all_languages.png
try:
    _c34 = get_crop(34, 511, 144)
    canvas.paste(_c34, (0, 1410), _c34)
except Exception:
    pass
layout["Show_all_languages"] = [0, 1410, 511, 1554]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_11_2024_4_23_19_27_45f56b06f31541079045047b6d542613-13/35_text_Price.png
try:
    _c35 = get_crop(35, 149, 63)
    canvas.paste(_c35, (45, 1613), _c35)
except Exception:
    pass
layout["Price"] = [45, 1613, 194, 1676]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_11_2024_4_23_19_27_45f56b06f31541079045047b6d542613-13/36_text_Only_free_events.png
try:
    _c36 = get_crop(36, 660, 144)
    canvas.paste(_c36, (54, 2024), _c36)
except Exception:
    pass
layout["Only_free_events"] = [54, 2024, 714, 2168]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_11_2024_4_23_19_27_45f56b06f31541079045047b6d542613-13/37_text_Sort_by.png
try:
    _c37 = get_crop(37, 206, 75)
    canvas.paste(_c37, (42, 1931), _c37)
except Exception:
    pass
layout["Sort_by"] = [42, 1931, 248, 2006]
