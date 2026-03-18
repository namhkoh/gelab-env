# page_id: page_eventbrite_c36ef2ec281b4d40839b2bf8f321a58b_05
# screenshot: 2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-7.png
# step_index: 5/8
# task: Open Eventbrite. Set the city to "Chicago". Select the "Fashion" category and view the recommended events. See the date of the first play and its venue.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw UI background and structural elements for the filters page
# Uses provided variables: canvas (PIL.Image), draw (PIL.ImageDraw), font_sm, font_md, font_lg, font_xl

# Fill overall background (canvas starts white, but ensure consistent fill)
draw.rectangle([(0, 0), (1440, 2960)], fill="#ffffff")

# Status bar at top (~72px high)
status_h = 72
draw.rectangle([(0, 0), (1440, status_h)], fill="#d0d0d4")

# Subtle divider under status bar
draw.line([(0, status_h), (1440, status_h)], fill="#c7c7cc", width=1)

# Header area (toolbar) background (keeps white but add faint bottom shadow)
header_top = status_h
header_bottom = 188
draw.rectangle([(0, header_top), (1440, header_bottom)], fill="#ffffff")
# header bottom shadow
draw.line([(0, header_bottom), (1440, header_bottom)], fill="#ebecee", width=2)

# Large content separators between major sections
# These are positioned to visually group Categories / Event type / Languages / Price / Sort sections
separator_color = "#efeff4"
# Under header, before Categories (leave some breathing room)
draw.line([(36, 300), (1404, 300)], fill=separator_color, width=1)
# Between Categories and Event type (below "Show all categories" area)
draw.line([(36, 740), (1404, 740)], fill=separator_color, width=1)
# Between Event type and Languages (below "Show all event types")
draw.line([(36, 1120), (1404, 1120)], fill=separator_color, width=1)
# Between Languages and Price (below "Show all languages")
draw.line([(36, 1570), (1404, 1570)], fill=separator_color, width=1)
# Above Sort by area
draw.line([(36, 1900), (1404, 1900)], fill=separator_color, width=1)

# Draw subtle rounded card background for the Sort control (segmented control)
# This is a grouped background behind the two segmented buttons ("Relevance" / "Date")
seg_x0, seg_y0 = 36, 1990
seg_x1, seg_y1 = 1404, 2168
draw.rounded_rectangle([(seg_x0, seg_y0), (seg_x1, seg_y1)],
                       radius=14, fill="#f7f7fb", outline="#e6e7ec", width=1)

# Add a faint inner shadow line to emphasize the selected segment area
draw.line([(seg_x0 + 12, seg_y0 + 4), (seg_x1 - 12, seg_y0 + 4)], fill="#f0f0f5", width=1)
draw.line([(seg_x0 + 12, seg_y1 - 4), (seg_x1 - 12, seg_y1 - 4)], fill="#f0f0f5", width=1)

# Draw a subtle long divider above the bottom action area (above Apply filters)
apply_top = 2768
draw.line([(24, apply_top - 80), (1416, apply_top - 80)], fill="#eeeef2", width=1)
# Slight shadow just above the apply button area to lift it visually
draw.rectangle([(24, apply_top - 76), (1416, apply_top - 74)], fill="#f3f3f6")

# Light footer/background band behind the bottom area (keeps white but adds faint rounded border)
footer_bg_y0 = apply_top - 96
footer_bg_y1 = 2960
draw.rounded_rectangle([(24, footer_bg_y0), (1416, footer_bg_y1 - 12)],
                       radius=12, outline="#e0e0e6", width=2, fill="#ffffff")

# Small decorative left/right margins for content (vertical rule subtle accents)
draw.line([(36, header_bottom + 12), (36, footer_bg_y0 - 12)], fill="#fbfbfd", width=1)
draw.line([(1404, header_bottom + 12), (1404, footer_bg_y0 - 12)], fill="#fbfbfd", width=1)

# End of structural/background drawing

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_05_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-7/00_icon_Food_Drink.png
try:
    _c0 = get_crop(0, 312, 144)
    canvas.paste(_c0, (512, 383), _c0)
except Exception:
    pass
layout["Food_&_Drink"] = [512, 383, 824, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_05_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-7/01_icon_Music.png
try:
    _c1 = get_crop(1, 187, 135)
    canvas.paste(_c1, (36, 383), _c1)
except Exception:
    pass
layout["Music"] = [36, 383, 223, 518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_05_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-7/02_icon_Community.png
try:
    _c2 = get_crop(2, 294, 144)
    canvas.paste(_c2, (848, 383), _c2)
except Exception:
    pass
layout["Community"] = [848, 383, 1142, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_05_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-7/03_icon_French.png
try:
    _c3 = get_crop(3, 205, 144)
    canvas.paste(_c3, (768, 1275), _c3)
except Exception:
    pass
layout["French"] = [768, 1275, 973, 1419]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_05_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-7/04_icon_Business.png
try:
    _c4 = get_crop(4, 241, 135)
    canvas.paste(_c4, (247, 383), _c4)
except Exception:
    pass
layout["Business"] = [247, 383, 488, 518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_05_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-7/05_icon_Spanish.png
try:
    _c5 = get_crop(5, 225, 144)
    canvas.paste(_c5, (519, 1275), _c5)
except Exception:
    pass
layout["Spanish"] = [519, 1275, 744, 1419]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_05_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-7/06_icon_Expo.png
try:
    _c6 = get_crop(6, 167, 144)
    canvas.paste(_c6, (614, 829), _c6)
except Exception:
    pass
layout["Expo"] = [614, 829, 781, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_05_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-7/07_icon_Italian.png
try:
    _c7 = get_crop(7, 191, 144)
    canvas.paste(_c7, (997, 1275), _c7)
except Exception:
    pass
layout["Italian"] = [997, 1275, 1188, 1419]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_05_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-7/08_icon_Seminar.png
try:
    _c8 = get_crop(8, 232, 144)
    canvas.paste(_c8, (358, 829), _c8)
except Exception:
    pass
layout["Seminar"] = [358, 829, 590, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_05_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-7/09_icon_Arts.png
try:
    _c9 = get_crop(9, 152, 144)
    canvas.paste(_c9, (1166, 383), _c9)
except Exception:
    pass
layout["Arts"] = [1166, 383, 1318, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_05_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-7/10_icon_Convention.png
try:
    _c10 = get_crop(10, 293, 144)
    canvas.paste(_c10, (805, 829), _c10)
except Exception:
    pass
layout["Convention"] = [805, 829, 1098, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_05_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-7/11_icon_Festival.png
try:
    _c11 = get_crop(11, 219, 144)
    canvas.paste(_c11, (1122, 829), _c11)
except Exception:
    pass
layout["Festival"] = [1122, 829, 1341, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_05_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-7/12_icon_German.png
try:
    _c12 = get_crop(12, 225, 135)
    canvas.paste(_c12, (270, 1275), _c12)
except Exception:
    pass
layout["German"] = [270, 1275, 495, 1410]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_05_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-7/13_icon_English.png
try:
    _c13 = get_crop(13, 210, 135)
    canvas.paste(_c13, (36, 1275), _c13)
except Exception:
    pass
layout["English"] = [36, 1275, 246, 1410]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_05_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-7/14_icon_Conference.png
try:
    _c14 = get_crop(14, 298, 135)
    canvas.paste(_c14, (36, 829), _c14)
except Exception:
    pass
layout["Conference"] = [36, 829, 334, 964]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_05_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-7/15_icon_Date.png
try:
    _c15 = get_crop(15, 660, 144)
    canvas.paste(_c15, (726, 2024), _c15)
except Exception:
    pass
layout["Date"] = [726, 2024, 1386, 2168]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_05_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-7/16_icon_Relevance.png
try:
    _c16 = get_crop(16, 660, 144)
    canvas.paste(_c16, (54, 2024), _c16)
except Exception:
    pass
layout["Relevance"] = [54, 2024, 714, 2168]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_05_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-7/17_icon_Apply_filters.png
try:
    _c17 = get_crop(17, 1344, 144)
    canvas.paste(_c17, (48, 2768), _c17)
except Exception:
    pass
layout["Apply_filters"] = [48, 2768, 1392, 2912]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_05_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-7/18_icon_5.12.png
try:
    _c18 = get_crop(18, 144, 144)
    canvas.paste(_c18, (12, 72), _c18)
except Exception:
    pass
layout["5.12"] = [12, 72, 156, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_05_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-7/19_icon_5.12.png
try:
    _c19 = get_crop(19, 60, 64)
    canvas.paste(_c19, (180, 1), _c19)
except Exception:
    pass
layout["5.12"] = [180, 1, 240, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_05_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-7/20_icon_icon_20.png
try:
    _c20 = get_crop(20, 64, 62)
    canvas.paste(_c20, (308, 3), _c20)
except Exception:
    pass
layout["icon_20"] = [308, 3, 372, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_05_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-7/21_icon_Clear_all.png
try:
    _c21 = get_crop(21, 55, 69)
    canvas.paste(_c21, (1319, 0), _c21)
except Exception:
    pass
layout["Clear_all"] = [1319, 0, 1374, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_05_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-7/22_icon_icon_22.png
try:
    _c22 = get_crop(22, 51, 62)
    canvas.paste(_c22, (248, 2), _c22)
except Exception:
    pass
layout["icon_22"] = [248, 2, 299, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_05_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-7/23_icon_Clear_all.png
try:
    _c23 = get_crop(23, 99, 70)
    canvas.paste(_c23, (1211, 0), _c23)
except Exception:
    pass
layout["Clear_all"] = [1211, 0, 1310, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_05_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-7/24_icon_5.12.png
try:
    _c24 = get_crop(24, 60, 66)
    canvas.paste(_c24, (116, 0), _c24)
except Exception:
    pass
layout["5.12"] = [116, 0, 176, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_05_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-7/25_icon_clickable_20.png
try:
    _c25 = get_crop(25, 144, 144)
    canvas.paste(_c25, (1248, 1729), _c25)
except Exception:
    pass
layout["clickable_20"] = [1248, 1729, 1392, 1873]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_05_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-7/26_icon_Clear_all.png
try:
    _c26 = get_crop(26, 178, 144)
    canvas.paste(_c26, (1214, 72), _c26)
except Exception:
    pass
layout["Clear_all"] = [1214, 72, 1392, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_05_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-7/27_text_5.12.png
try:
    _c27 = get_crop(27, 89, 43)
    canvas.paste(_c27, (22, 17), _c27)
except Exception:
    pass
layout["5.12"] = [22, 17, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_05_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-7/28_text_Filters.png
try:
    _c28 = get_crop(28, 180, 66)
    canvas.paste(_c28, (631, 116), _c28)
except Exception:
    pass
layout["Filters"] = [631, 116, 811, 182]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_05_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-7/29_text_Categories.png
try:
    _c29 = get_crop(29, 187, 135)
    canvas.paste(_c29, (36, 383), _c29)
except Exception:
    pass
layout["Categories"] = [36, 383, 223, 518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_05_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-7/30_text_Show_all_categories.png
try:
    _c30 = get_crop(30, 516, 144)
    canvas.paste(_c30, (0, 518), _c30)
except Exception:
    pass
layout["Show_all_categories"] = [0, 518, 516, 662]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_05_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-7/31_text_Event_type.png
try:
    _c31 = get_crop(31, 298, 135)
    canvas.paste(_c31, (36, 829), _c31)
except Exception:
    pass
layout["Event_type"] = [36, 829, 334, 964]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_05_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-7/32_text_Show_all_event_types.png
try:
    _c32 = get_crop(32, 535, 144)
    canvas.paste(_c32, (0, 964), _c32)
except Exception:
    pass
layout["Show_all_event_types"] = [0, 964, 535, 1108]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_05_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-7/33_text_Languages.png
try:
    _c33 = get_crop(33, 210, 135)
    canvas.paste(_c33, (36, 1275), _c33)
except Exception:
    pass
layout["Languages"] = [36, 1275, 246, 1410]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_05_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-7/34_text_Show_all_languages.png
try:
    _c34 = get_crop(34, 511, 144)
    canvas.paste(_c34, (0, 1410), _c34)
except Exception:
    pass
layout["Show_all_languages"] = [0, 1410, 511, 1554]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_05_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-7/35_text_Price.png
try:
    _c35 = get_crop(35, 149, 63)
    canvas.paste(_c35, (45, 1613), _c35)
except Exception:
    pass
layout["Price"] = [45, 1613, 194, 1676]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_05_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-7/36_text_Only_free_events.png
try:
    _c36 = get_crop(36, 660, 144)
    canvas.paste(_c36, (54, 2024), _c36)
except Exception:
    pass
layout["Only_free_events"] = [54, 2024, 714, 2168]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_05_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-7/37_text_Sort_by.png
try:
    _c37 = get_crop(37, 206, 75)
    canvas.paste(_c37, (42, 1931), _c37)
except Exception:
    pass
layout["Sort_by"] = [42, 1931, 248, 2006]
