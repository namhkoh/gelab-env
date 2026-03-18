# page_id: page_eventbrite_997cbfb071cf458f82ccaba5bddfc7b8_06
# screenshot: 2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-8.png
# step_index: 6/15
# task: Open Eventbrite. Search Online free events. Select the first one. Follow the organizer. Read more about the event. Add it to Favorites.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# draw UI background and structural elements for 1440x2960 canvas
# available variables: canvas (PIL Image), draw (ImageDraw), font_sm, font_md, font_lg, font_xl

w, h = canvas.size

# Colors
bg_color = "#fbfbfd"        # overall page background
status_color = "#bdbfc3"    # status bar
header_bg = "#ffffff"       # header background
divider_color = "#e6e7eb"   # light divider lines
card_fill = "#f7f8fb"       # subtle card background
card_border = "#e6e7eb"     # card border
seg_container = "#efeff4"   # segmented control background
shadow_color = "#ececf0"    # very light shadow for separation

# Clear background
draw.rectangle([(0, 0), (w, h)], fill=bg_color)

# Status bar (top)
status_h = 96
draw.rectangle([(0, 0), (w, status_h)], fill=status_color)

# Header / toolbar area
header_top = status_h
header_bottom = 176
draw.rectangle([(0, header_top), (w, header_bottom)], fill=header_bg)
# subtle bottom divider under header
draw.line([(24, header_bottom), (w-24, header_bottom)], fill=divider_color, width=2)

# Section card backgrounds (rounded rectangles behind groups)
# Categories card
cat_top = 320
cat_bottom = 560
cat_margin = 28
draw.rounded_rectangle(
    [(cat_margin, cat_top), (w - cat_margin, cat_bottom)],
    radius=20,
    fill=card_fill,
    outline=card_border,
    width=2
)

# Event type card
evt_top = 760
evt_bottom = 1004
draw.rounded_rectangle(
    [(cat_margin, evt_top), (w - cat_margin, evt_bottom)],
    radius=20,
    fill=card_fill,
    outline=card_border,
    width=2
)

# Languages card
lang_top = 1200
lang_bottom = 1440
draw.rounded_rectangle(
    [(cat_margin, lang_top), (w - cat_margin, lang_bottom)],
    radius=20,
    fill=card_fill,
    outline=card_border,
    width=2
)

# Price & toggle grouping card (subtle, spans down toward sort)
price_top = 1548
price_bottom = 2068
draw.rounded_rectangle(
    [(cat_margin, price_top), (w - cat_margin, price_bottom)],
    radius=16,
    fill=card_fill,
    outline=card_border,
    width=2
)

# Segmented 'Sort by' container background (behind the two selectable pills)
seg_left = 36
seg_top = 2008
seg_right = w - 36
seg_bottom = 2168
draw.rounded_rectangle(
    [(seg_left, seg_top), (seg_right, seg_bottom)],
    radius=18,
    fill=seg_container,
    outline=divider_color,
    width=2
)

# Large whitespace divider areas (subtle horizontal separators)
separator_ys = [cat_bottom + 20, evt_bottom + 20, lang_bottom + 20, price_bottom + 20, seg_bottom + 40]
for y in separator_ys:
    draw.line([(24, y), (w - 24, y)], fill=shadow_color, width=1)

# Top glow/shadow for bottom apply-filters area (to visually separate it)
apply_top = 2768
# a faint horizontal shadow line above the apply bar
draw.line([(16, apply_top - 16), (w - 16, apply_top - 16)], fill=shadow_color, width=6)
# a slightly darker thin divider
draw.line([(16, apply_top - 6), (w - 16, apply_top - 6)], fill=divider_color, width=2)

# Small left/right margins vertical guide (subtle)
draw.line([(24, header_bottom + 8), (24, h - 200)], fill=bg_color, width=1)
draw.line([(w-24, header_bottom + 8), (w-24, h - 200)], fill=bg_color, width=1)

# faint outer border around page content area (very subtle)
draw.rounded_rectangle(
    [(8, header_bottom + 6), (w - 8, apply_top - 20)],
    radius=8,
    outline=("#f0f0f3"),
    width=1
)

# End of background/structure drawing.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_06_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-8/00_icon_Food_Drink.png
try:
    _c0 = get_crop(0, 312, 144)
    canvas.paste(_c0, (512, 383), _c0)
except Exception:
    pass
layout["Food_&_Drink"] = [512, 383, 824, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_06_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-8/01_icon_Music.png
try:
    _c1 = get_crop(1, 187, 135)
    canvas.paste(_c1, (36, 383), _c1)
except Exception:
    pass
layout["Music"] = [36, 383, 223, 518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_06_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-8/02_icon_Community.png
try:
    _c2 = get_crop(2, 294, 144)
    canvas.paste(_c2, (848, 383), _c2)
except Exception:
    pass
layout["Community"] = [848, 383, 1142, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_06_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-8/03_icon_French.png
try:
    _c3 = get_crop(3, 205, 144)
    canvas.paste(_c3, (768, 1275), _c3)
except Exception:
    pass
layout["French"] = [768, 1275, 973, 1419]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_06_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-8/04_icon_Spanish.png
try:
    _c4 = get_crop(4, 225, 144)
    canvas.paste(_c4, (519, 1275), _c4)
except Exception:
    pass
layout["Spanish"] = [519, 1275, 744, 1419]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_06_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-8/05_icon_Business.png
try:
    _c5 = get_crop(5, 241, 135)
    canvas.paste(_c5, (247, 383), _c5)
except Exception:
    pass
layout["Business"] = [247, 383, 488, 518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_06_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-8/06_icon_Expo.png
try:
    _c6 = get_crop(6, 167, 144)
    canvas.paste(_c6, (614, 829), _c6)
except Exception:
    pass
layout["Expo"] = [614, 829, 781, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_06_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-8/07_icon_Seminar.png
try:
    _c7 = get_crop(7, 232, 144)
    canvas.paste(_c7, (358, 829), _c7)
except Exception:
    pass
layout["Seminar"] = [358, 829, 590, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_06_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-8/08_icon_Italian.png
try:
    _c8 = get_crop(8, 191, 144)
    canvas.paste(_c8, (997, 1275), _c8)
except Exception:
    pass
layout["Italian"] = [997, 1275, 1188, 1419]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_06_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-8/09_icon_Arts.png
try:
    _c9 = get_crop(9, 152, 144)
    canvas.paste(_c9, (1166, 383), _c9)
except Exception:
    pass
layout["Arts"] = [1166, 383, 1318, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_06_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-8/10_icon_Convention.png
try:
    _c10 = get_crop(10, 293, 144)
    canvas.paste(_c10, (805, 829), _c10)
except Exception:
    pass
layout["Convention"] = [805, 829, 1098, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_06_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-8/11_icon_Festival.png
try:
    _c11 = get_crop(11, 219, 144)
    canvas.paste(_c11, (1122, 829), _c11)
except Exception:
    pass
layout["Festival"] = [1122, 829, 1341, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_06_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-8/12_icon_German.png
try:
    _c12 = get_crop(12, 225, 135)
    canvas.paste(_c12, (270, 1275), _c12)
except Exception:
    pass
layout["German"] = [270, 1275, 495, 1410]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_06_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-8/13_icon_English.png
try:
    _c13 = get_crop(13, 210, 135)
    canvas.paste(_c13, (36, 1275), _c13)
except Exception:
    pass
layout["English"] = [36, 1275, 246, 1410]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_06_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-8/14_icon_Conference.png
try:
    _c14 = get_crop(14, 298, 135)
    canvas.paste(_c14, (36, 829), _c14)
except Exception:
    pass
layout["Conference"] = [36, 829, 334, 964]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_06_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-8/15_icon_Apply_filters_1.png
try:
    _c15 = get_crop(15, 1344, 144)
    canvas.paste(_c15, (48, 2768), _c15)
except Exception:
    pass
layout["Apply_filters_(1)"] = [48, 2768, 1392, 2912]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_06_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-8/16_icon_Date.png
try:
    _c16 = get_crop(16, 660, 144)
    canvas.paste(_c16, (726, 2024), _c16)
except Exception:
    pass
layout["Date"] = [726, 2024, 1386, 2168]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_06_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-8/17_icon_Relevance.png
try:
    _c17 = get_crop(17, 660, 144)
    canvas.paste(_c17, (54, 2024), _c17)
except Exception:
    pass
layout["Relevance"] = [54, 2024, 714, 2168]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_06_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-8/18_icon_9.15.png
try:
    _c18 = get_crop(18, 144, 144)
    canvas.paste(_c18, (12, 72), _c18)
except Exception:
    pass
layout["9.15"] = [12, 72, 156, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_06_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-8/19_icon_9.15.png
try:
    _c19 = get_crop(19, 65, 65)
    canvas.paste(_c19, (177, 1), _c19)
except Exception:
    pass
layout["9.15"] = [177, 1, 242, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_06_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-8/20_icon_Clear_all.png
try:
    _c20 = get_crop(20, 99, 65)
    canvas.paste(_c20, (1211, 0), _c20)
except Exception:
    pass
layout["Clear_all"] = [1211, 0, 1310, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_06_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-8/21_icon_Clear_all.png
try:
    _c21 = get_crop(21, 56, 67)
    canvas.paste(_c21, (1318, 0), _c21)
except Exception:
    pass
layout["Clear_all"] = [1318, 0, 1374, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_06_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-8/22_icon_9.15.png
try:
    _c22 = get_crop(22, 56, 64)
    canvas.paste(_c22, (115, 1), _c22)
except Exception:
    pass
layout["9.15"] = [115, 1, 171, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_06_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-8/23_icon_clickable_20.png
try:
    _c23 = get_crop(23, 144, 144)
    canvas.paste(_c23, (1248, 1729), _c23)
except Exception:
    pass
layout["clickable_20"] = [1248, 1729, 1392, 1873]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_06_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-8/24_icon_icon_24.png
try:
    _c24 = get_crop(24, 59, 62)
    canvas.paste(_c24, (245, 2), _c24)
except Exception:
    pass
layout["icon_24"] = [245, 2, 304, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_06_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-8/25_icon_Clear_all.png
try:
    _c25 = get_crop(25, 178, 144)
    canvas.paste(_c25, (1214, 72), _c25)
except Exception:
    pass
layout["Clear_all"] = [1214, 72, 1392, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_06_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-8/26_icon_icon_26.png
try:
    _c26 = get_crop(26, 54, 61)
    canvas.paste(_c26, (314, 3), _c26)
except Exception:
    pass
layout["icon_26"] = [314, 3, 368, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_06_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-8/27_text_9.15.png
try:
    _c27 = get_crop(27, 94, 43)
    canvas.paste(_c27, (20, 17), _c27)
except Exception:
    pass
layout["9.15"] = [20, 17, 114, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_06_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-8/28_text_Filters.png
try:
    _c28 = get_crop(28, 180, 66)
    canvas.paste(_c28, (631, 116), _c28)
except Exception:
    pass
layout["Filters"] = [631, 116, 811, 182]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_06_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-8/29_text_Categories.png
try:
    _c29 = get_crop(29, 187, 135)
    canvas.paste(_c29, (36, 383), _c29)
except Exception:
    pass
layout["Categories"] = [36, 383, 223, 518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_06_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-8/30_text_Show_all_categories.png
try:
    _c30 = get_crop(30, 516, 144)
    canvas.paste(_c30, (0, 518), _c30)
except Exception:
    pass
layout["Show_all_categories"] = [0, 518, 516, 662]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_06_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-8/31_text_Event_type.png
try:
    _c31 = get_crop(31, 298, 135)
    canvas.paste(_c31, (36, 829), _c31)
except Exception:
    pass
layout["Event_type"] = [36, 829, 334, 964]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_06_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-8/32_text_Show_all_event_types.png
try:
    _c32 = get_crop(32, 535, 144)
    canvas.paste(_c32, (0, 964), _c32)
except Exception:
    pass
layout["Show_all_event_types"] = [0, 964, 535, 1108]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_06_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-8/33_text_Languages.png
try:
    _c33 = get_crop(33, 210, 135)
    canvas.paste(_c33, (36, 1275), _c33)
except Exception:
    pass
layout["Languages"] = [36, 1275, 246, 1410]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_06_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-8/34_text_Show_all_languages.png
try:
    _c34 = get_crop(34, 511, 144)
    canvas.paste(_c34, (0, 1410), _c34)
except Exception:
    pass
layout["Show_all_languages"] = [0, 1410, 511, 1554]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_06_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-8/35_text_Price.png
try:
    _c35 = get_crop(35, 149, 63)
    canvas.paste(_c35, (45, 1613), _c35)
except Exception:
    pass
layout["Price"] = [45, 1613, 194, 1676]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_06_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-8/36_text_Only_free_events.png
try:
    _c36 = get_crop(36, 660, 144)
    canvas.paste(_c36, (54, 2024), _c36)
except Exception:
    pass
layout["Only_free_events"] = [54, 2024, 714, 2168]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_06_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-8/37_text_Sort_by.png
try:
    _c37 = get_crop(37, 206, 75)
    canvas.paste(_c37, (42, 1931), _c37)
except Exception:
    pass
layout["Sort_by"] = [42, 1931, 248, 2006]
