# page_id: page_eventbrite_1c30518736b1454cb330b963c1cc6d86_06
# screenshot: 2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-8.png
# step_index: 6/9
# task: Open Eventbrite. Search for "Open Mic Nights". Filter the results to only include free events. Select the first non-promoted event in the list - what"s the location of that event?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw UI background and structure for a 1440x2960 canvas.
# Assumes variables provided: canvas (PIL.Image), draw (PIL.ImageDraw), font_sm, font_md, font_lg, font_xl

w, h = canvas.size

# Primary background (slightly off-white to match app background)
draw.rectangle([(0, 0), (w, h)], fill="#FBFBFD")

# Top status bar (approx ~72px tall)
status_h = 72
draw.rectangle([(0, 0), (w, status_h)], fill="#C9C9C9")

# Header / toolbar area beneath status bar (keeps the page background color, with subtle divider)
header_top = status_h
header_bottom = 176
draw.rectangle([(0, header_top), (w, header_bottom)], fill="#FBFBFD")
draw.line([(24, header_bottom), (w-24, header_bottom)], fill="#E6E6EA", width=2)

# Subtle horizontal separators between major groups (do not draw text or icons)
# Separator below Categories chips area (approx)
sep_y1 = 700
draw.line([(36, sep_y1), (w-36, sep_y1)], fill="#F0EFF2", width=1)

# Separator below Event type chips / Show all event types (approx)
sep_y2 = 1120
draw.line([(36, sep_y2), (w-36, sep_y2)], fill="#F0EFF2", width=1)

# Separator below Languages section (approx)
sep_y3 = 1500
draw.line([(36, sep_y3), (w-36, sep_y3)], fill="#F0EFF2", width=1)

# Price section area separator (light)
sep_y4 = 1700
draw.line([(36, sep_y4), (w-36, sep_y4)], fill="#F0EFF2", width=1)

# "Sort by" segmented control outer background (rounded container)
seg_x1 = 36
seg_x2 = w - 36
seg_y1 = 2024
seg_h = 144
seg_y2 = seg_y1 + seg_h
seg_radius = 18

# subtle drop shadow (simple thin shadow line)
shadow_y = seg_y2 + 6
draw.line([(seg_x1+6, shadow_y), (seg_x2-6, shadow_y)], fill="#E2DEE4", width=3)

# outer rounded container (the segmented control background)
draw.rounded_rectangle(
    [(seg_x1, seg_y1), (seg_x2, seg_y2)],
    radius=seg_radius,
    fill="#ECEAF0",
    outline="#D6D3DA",
    width=2
)

# Additional subtle separators / section dividers down the page to structure the layout
divider_positions = [880, 980, 1260, 1860]
for y in divider_positions:
    draw.line([(36, y), (w-36, y)], fill="#F7F6F9", width=1)

# Bottom safe area top divider (above the apply button region)
bottom_divider_y = 2690
draw.line([(24, bottom_divider_y), (w-24, bottom_divider_y)], fill="#E6E2E8", width=2)

# A faint rounded outline for page content bounds (very subtle)
content_pad = 24
draw.rounded_rectangle(
    [(content_pad, header_bottom+8), (w-content_pad, bottom_divider_y-8)],
    radius=12,
    outline="#F2F1F4",
    width=1
)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_06_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-8/00_icon_Food_Drink.png
try:
    _c0 = get_crop(0, 312, 144)
    canvas.paste(_c0, (512, 383), _c0)
except Exception:
    pass
layout["Food_&_Drink"] = [512, 383, 824, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_06_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-8/01_icon_Music.png
try:
    _c1 = get_crop(1, 187, 135)
    canvas.paste(_c1, (36, 383), _c1)
except Exception:
    pass
layout["Music"] = [36, 383, 223, 518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_06_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-8/02_icon_Community.png
try:
    _c2 = get_crop(2, 294, 144)
    canvas.paste(_c2, (848, 383), _c2)
except Exception:
    pass
layout["Community"] = [848, 383, 1142, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_06_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-8/03_icon_French.png
try:
    _c3 = get_crop(3, 205, 144)
    canvas.paste(_c3, (768, 1275), _c3)
except Exception:
    pass
layout["French"] = [768, 1275, 973, 1419]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_06_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-8/04_icon_Business.png
try:
    _c4 = get_crop(4, 241, 135)
    canvas.paste(_c4, (247, 383), _c4)
except Exception:
    pass
layout["Business"] = [247, 383, 488, 518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_06_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-8/05_icon_Spanish.png
try:
    _c5 = get_crop(5, 225, 144)
    canvas.paste(_c5, (519, 1275), _c5)
except Exception:
    pass
layout["Spanish"] = [519, 1275, 744, 1419]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_06_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-8/06_icon_Expo.png
try:
    _c6 = get_crop(6, 167, 144)
    canvas.paste(_c6, (614, 829), _c6)
except Exception:
    pass
layout["Expo"] = [614, 829, 781, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_06_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-8/07_icon_Italian.png
try:
    _c7 = get_crop(7, 191, 144)
    canvas.paste(_c7, (997, 1275), _c7)
except Exception:
    pass
layout["Italian"] = [997, 1275, 1188, 1419]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_06_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-8/08_icon_Seminar.png
try:
    _c8 = get_crop(8, 232, 144)
    canvas.paste(_c8, (358, 829), _c8)
except Exception:
    pass
layout["Seminar"] = [358, 829, 590, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_06_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-8/09_icon_Arts.png
try:
    _c9 = get_crop(9, 152, 144)
    canvas.paste(_c9, (1166, 383), _c9)
except Exception:
    pass
layout["Arts"] = [1166, 383, 1318, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_06_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-8/10_icon_Convention.png
try:
    _c10 = get_crop(10, 293, 144)
    canvas.paste(_c10, (805, 829), _c10)
except Exception:
    pass
layout["Convention"] = [805, 829, 1098, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_06_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-8/11_icon_Festival.png
try:
    _c11 = get_crop(11, 219, 144)
    canvas.paste(_c11, (1122, 829), _c11)
except Exception:
    pass
layout["Festival"] = [1122, 829, 1341, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_06_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-8/12_icon_German.png
try:
    _c12 = get_crop(12, 225, 135)
    canvas.paste(_c12, (270, 1275), _c12)
except Exception:
    pass
layout["German"] = [270, 1275, 495, 1410]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_06_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-8/13_icon_English.png
try:
    _c13 = get_crop(13, 210, 135)
    canvas.paste(_c13, (36, 1275), _c13)
except Exception:
    pass
layout["English"] = [36, 1275, 246, 1410]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_06_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-8/14_icon_Conference.png
try:
    _c14 = get_crop(14, 298, 135)
    canvas.paste(_c14, (36, 829), _c14)
except Exception:
    pass
layout["Conference"] = [36, 829, 334, 964]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_06_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-8/15_icon_Apply_filters_1.png
try:
    _c15 = get_crop(15, 1344, 144)
    canvas.paste(_c15, (48, 2768), _c15)
except Exception:
    pass
layout["Apply_filters_(1)"] = [48, 2768, 1392, 2912]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_06_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-8/16_icon_Date.png
try:
    _c16 = get_crop(16, 660, 144)
    canvas.paste(_c16, (726, 2024), _c16)
except Exception:
    pass
layout["Date"] = [726, 2024, 1386, 2168]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_06_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-8/17_icon_Relevance.png
try:
    _c17 = get_crop(17, 660, 144)
    canvas.paste(_c17, (54, 2024), _c17)
except Exception:
    pass
layout["Relevance"] = [54, 2024, 714, 2168]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_06_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-8/18_icon_4.54.png
try:
    _c18 = get_crop(18, 144, 144)
    canvas.paste(_c18, (12, 72), _c18)
except Exception:
    pass
layout["4.54"] = [12, 72, 156, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_06_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-8/19_icon_4.54.png
try:
    _c19 = get_crop(19, 61, 64)
    canvas.paste(_c19, (179, 2), _c19)
except Exception:
    pass
layout["4.54"] = [179, 2, 240, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_06_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-8/20_icon_icon_20.png
try:
    _c20 = get_crop(20, 64, 62)
    canvas.paste(_c20, (308, 3), _c20)
except Exception:
    pass
layout["icon_20"] = [308, 3, 372, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_06_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-8/21_icon_4.54.png
try:
    _c21 = get_crop(21, 64, 65)
    canvas.paste(_c21, (112, 1), _c21)
except Exception:
    pass
layout["4.54"] = [112, 1, 176, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_06_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-8/22_icon_Clear_all.png
try:
    _c22 = get_crop(22, 56, 67)
    canvas.paste(_c22, (1317, 0), _c22)
except Exception:
    pass
layout["Clear_all"] = [1317, 0, 1373, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_06_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-8/23_icon_Clear_all.png
try:
    _c23 = get_crop(23, 99, 65)
    canvas.paste(_c23, (1211, 0), _c23)
except Exception:
    pass
layout["Clear_all"] = [1211, 0, 1310, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_06_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-8/24_icon_icon_24.png
try:
    _c24 = get_crop(24, 51, 62)
    canvas.paste(_c24, (248, 2), _c24)
except Exception:
    pass
layout["icon_24"] = [248, 2, 299, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_06_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-8/25_icon_Toggle_to_filter_events_to_show_only_fre.png
try:
    _c25 = get_crop(25, 144, 144)
    canvas.paste(_c25, (1248, 1729), _c25)
except Exception:
    pass
layout["Toggle_to_filter_events_t"] = [1248, 1729, 1392, 1873]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_06_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-8/26_icon_Clear_all.png
try:
    _c26 = get_crop(26, 178, 144)
    canvas.paste(_c26, (1214, 72), _c26)
except Exception:
    pass
layout["Clear_all"] = [1214, 72, 1392, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_06_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-8/27_text_Filters.png
try:
    _c27 = get_crop(27, 180, 66)
    canvas.paste(_c27, (631, 116), _c27)
except Exception:
    pass
layout["Filters"] = [631, 116, 811, 182]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_06_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-8/28_text_Categories.png
try:
    _c28 = get_crop(28, 187, 135)
    canvas.paste(_c28, (36, 383), _c28)
except Exception:
    pass
layout["Categories"] = [36, 383, 223, 518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_06_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-8/29_text_Show_all_categories.png
try:
    _c29 = get_crop(29, 516, 144)
    canvas.paste(_c29, (0, 518), _c29)
except Exception:
    pass
layout["Show_all_categories"] = [0, 518, 516, 662]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_06_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-8/30_text_Event_type.png
try:
    _c30 = get_crop(30, 298, 135)
    canvas.paste(_c30, (36, 829), _c30)
except Exception:
    pass
layout["Event_type"] = [36, 829, 334, 964]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_06_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-8/31_text_Show_all_event_types.png
try:
    _c31 = get_crop(31, 535, 144)
    canvas.paste(_c31, (0, 964), _c31)
except Exception:
    pass
layout["Show_all_event_types"] = [0, 964, 535, 1108]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_06_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-8/32_text_Languages.png
try:
    _c32 = get_crop(32, 210, 135)
    canvas.paste(_c32, (36, 1275), _c32)
except Exception:
    pass
layout["Languages"] = [36, 1275, 246, 1410]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_06_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-8/33_text_Show_all_languages.png
try:
    _c33 = get_crop(33, 511, 144)
    canvas.paste(_c33, (0, 1410), _c33)
except Exception:
    pass
layout["Show_all_languages"] = [0, 1410, 511, 1554]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_06_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-8/34_text_Price.png
try:
    _c34 = get_crop(34, 149, 63)
    canvas.paste(_c34, (45, 1613), _c34)
except Exception:
    pass
layout["Price"] = [45, 1613, 194, 1676]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_06_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-8/35_text_Only_free_events.png
try:
    _c35 = get_crop(35, 660, 144)
    canvas.paste(_c35, (54, 2024), _c35)
except Exception:
    pass
layout["Only_free_events"] = [54, 2024, 714, 2168]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_06_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-8/36_text_Sort_by.png
try:
    _c36 = get_crop(36, 206, 75)
    canvas.paste(_c36, (42, 1931), _c36)
except Exception:
    pass
layout["Sort_by"] = [42, 1931, 248, 2006]
