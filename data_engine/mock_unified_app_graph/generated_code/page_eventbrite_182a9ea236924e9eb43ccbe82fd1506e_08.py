# page_id: page_eventbrite_182a9ea236924e9eb43ccbe82fd1506e_08
# screenshot: 2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-10.png
# step_index: 8/13
# task: Open Eventbrite. Set time to tomorrow. Clear all search filters. Select the third one in New York. Record its location and time in Google Keep Notes. Add it to Favorites.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw background and UI structural elements for the Filters page
# (uses provided variables: canvas, draw, font_sm, font_md, font_lg, font_xl)

# Canvas size known: 1440x2960

# Colors
status_bar_color = "#c7c7c7"      # top status bar gray
header_divider = "#e9e7ee"        # subtle divider under header
page_bg = "#ffffff"               # main page background (canvas already white)
muted_divider = "#f0eef2"         # faint separators used between sections
sort_container_fill = "#f3f0f6"   # background for the sort segmented control
sort_border = "#dcd7df"
segment_selected = "#ffffff"
apply_btn_border = "#bfb8c1"
apply_btn_fill = "#ffffff"
shadow_color = (0, 0, 0, 18)      # RGBA for subtle shadow; will emulate using translucent rectangles

# 1) Status bar area (top ~72px)
status_h = 72
draw.rectangle([0, 0, 1440, status_h], fill=status_bar_color)

# 2) Header area (below status bar). Keep it white, add a bottom divider line.
header_top = status_h
header_bot = 168
draw.rectangle([0, header_top, 1440, header_bot], fill=page_bg)
# bottom divider
draw.line([(36, header_bot), (1404, header_bot)], fill=header_divider, width=2)

# 3) Subtle separators between logical sections
# Based on detected group vertical positions, add faint separators to structure content
separators = [
    660,   # below Categories area block
    1000,  # below Event type block
    1410,  # below Languages block
    1700,  # below Price / toggle area
    2300,  # upper content separator approaching sort area
]
for y in separators:
    draw.line([(36, y), (1404, y)], fill=muted_divider, width=1)

# 4) Sort-by segmented control container (rounded rect)
sort_x1 = 36
sort_y1 = 2016
sort_w = 1368
sort_h = 152
sort_x2 = sort_x1 + sort_w
sort_y2 = sort_y1 + sort_h
draw.rounded_rectangle([sort_x1, sort_y1, sort_x2, sort_y2],
                       radius=18, fill=sort_container_fill, outline=sort_border, width=2)

# subtle drop shadow under sort container (emulated with a faint rectangle)
shadow_y1 = sort_y2 + 4
shadow_y2 = shadow_y1 + 6
draw.rectangle([sort_x1 + 6, shadow_y1, sort_x2 - 6, shadow_y2], fill="#e9e6ea")

# 4a) Left selected segment background (rounded on left only)
seg_padding = 12
left_seg_x1 = sort_x1 + seg_padding + 6
left_seg_y1 = sort_y1 + seg_padding
left_seg_w = 660
left_seg_x2 = left_seg_x1 + left_seg_w
left_seg_y2 = sort_y2 - seg_padding
# Draw left segment white background with left rounded corners
draw.rounded_rectangle([left_seg_x1, left_seg_y1, left_seg_x2, left_seg_y2],
                       radius=14, fill=segment_selected, outline=None)

# 5) Apply filters button area at bottom (rounded rectangle with border)
apply_x1 = 48
apply_y1 = 2768
apply_w = 1344
apply_h = 144
apply_x2 = apply_x1 + apply_w
apply_y2 = apply_y1 + apply_h
# subtle shadow below the button
draw.rectangle([apply_x1 + 8, apply_y2 + 4, apply_x2 - 8, apply_y2 + 10], fill="#e7e4e8")
# button background and border
draw.rounded_rectangle([apply_x1, apply_y1, apply_x2, apply_y2],
                       radius=12, fill=apply_btn_fill, outline=apply_btn_border, width=3)

# 6) Large whitespace grouping cards (subtle pale blocks behind logical groups)
# These are soft backgrounds to hint grouping but won't conflict with pasted icons/text.
group_blocks = [
    # Categories group (top area where chips will be pasted on top)
    (24, 200, 1416, 660),
    # Event type group
    (24, 700, 1416, 1088),
    # Languages group
    (24, 1160, 1416, 1520),
    # Price / toggle area
    (24, 1520, 1416, 1888),
]
for (gx1, gy1, gx2, gy2) in group_blocks:
    # Very subtle off-white fill, low-contrast
    draw.rectangle([gx1, gy1, gx2, gy2], fill="#ffffff")

# 7) Fine vertical padding guides (very faint) to improve perceived structure
for y in [316, 452, 924, 1200, 1640]:
    draw.line([(60, y), (1400, y)], fill="#fbfbfb", width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_08_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-10/00_icon_Food_Drink.png
try:
    _c0 = get_crop(0, 312, 144)
    canvas.paste(_c0, (512, 383), _c0)
except Exception:
    pass
layout["Food_&_Drink"] = [512, 383, 824, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_08_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-10/01_icon_Music.png
try:
    _c1 = get_crop(1, 187, 135)
    canvas.paste(_c1, (36, 383), _c1)
except Exception:
    pass
layout["Music"] = [36, 383, 223, 518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_08_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-10/02_icon_Community.png
try:
    _c2 = get_crop(2, 294, 144)
    canvas.paste(_c2, (848, 383), _c2)
except Exception:
    pass
layout["Community"] = [848, 383, 1142, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_08_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-10/03_icon_French.png
try:
    _c3 = get_crop(3, 205, 144)
    canvas.paste(_c3, (768, 1275), _c3)
except Exception:
    pass
layout["French"] = [768, 1275, 973, 1419]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_08_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-10/04_icon_Business.png
try:
    _c4 = get_crop(4, 241, 135)
    canvas.paste(_c4, (247, 383), _c4)
except Exception:
    pass
layout["Business"] = [247, 383, 488, 518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_08_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-10/05_icon_Spanish.png
try:
    _c5 = get_crop(5, 225, 144)
    canvas.paste(_c5, (519, 1275), _c5)
except Exception:
    pass
layout["Spanish"] = [519, 1275, 744, 1419]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_08_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-10/06_icon_Expo.png
try:
    _c6 = get_crop(6, 167, 144)
    canvas.paste(_c6, (614, 829), _c6)
except Exception:
    pass
layout["Expo"] = [614, 829, 781, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_08_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-10/07_icon_Seminar.png
try:
    _c7 = get_crop(7, 232, 144)
    canvas.paste(_c7, (358, 829), _c7)
except Exception:
    pass
layout["Seminar"] = [358, 829, 590, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_08_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-10/08_icon_Italian.png
try:
    _c8 = get_crop(8, 191, 144)
    canvas.paste(_c8, (997, 1275), _c8)
except Exception:
    pass
layout["Italian"] = [997, 1275, 1188, 1419]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_08_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-10/09_icon_Arts.png
try:
    _c9 = get_crop(9, 152, 144)
    canvas.paste(_c9, (1166, 383), _c9)
except Exception:
    pass
layout["Arts"] = [1166, 383, 1318, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_08_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-10/10_icon_Convention.png
try:
    _c10 = get_crop(10, 293, 144)
    canvas.paste(_c10, (805, 829), _c10)
except Exception:
    pass
layout["Convention"] = [805, 829, 1098, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_08_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-10/11_icon_German.png
try:
    _c11 = get_crop(11, 225, 135)
    canvas.paste(_c11, (270, 1275), _c11)
except Exception:
    pass
layout["German"] = [270, 1275, 495, 1410]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_08_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-10/12_icon_Festival.png
try:
    _c12 = get_crop(12, 219, 144)
    canvas.paste(_c12, (1122, 829), _c12)
except Exception:
    pass
layout["Festival"] = [1122, 829, 1341, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_08_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-10/13_icon_English.png
try:
    _c13 = get_crop(13, 210, 135)
    canvas.paste(_c13, (36, 1275), _c13)
except Exception:
    pass
layout["English"] = [36, 1275, 246, 1410]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_08_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-10/14_icon_Conference.png
try:
    _c14 = get_crop(14, 298, 135)
    canvas.paste(_c14, (36, 829), _c14)
except Exception:
    pass
layout["Conference"] = [36, 829, 334, 964]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_08_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-10/15_icon_Date.png
try:
    _c15 = get_crop(15, 660, 144)
    canvas.paste(_c15, (726, 2024), _c15)
except Exception:
    pass
layout["Date"] = [726, 2024, 1386, 2168]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_08_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-10/16_icon_Relevance.png
try:
    _c16 = get_crop(16, 660, 144)
    canvas.paste(_c16, (54, 2024), _c16)
except Exception:
    pass
layout["Relevance"] = [54, 2024, 714, 2168]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_08_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-10/17_icon_Apply_filters.png
try:
    _c17 = get_crop(17, 1344, 144)
    canvas.paste(_c17, (48, 2768), _c17)
except Exception:
    pass
layout["Apply_filters"] = [48, 2768, 1392, 2912]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_08_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-10/18_icon_9.32.png
try:
    _c18 = get_crop(18, 144, 144)
    canvas.paste(_c18, (12, 72), _c18)
except Exception:
    pass
layout["9.32"] = [12, 72, 156, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_08_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-10/19_icon_9.32.png
try:
    _c19 = get_crop(19, 64, 63)
    canvas.paste(_c19, (176, 2), _c19)
except Exception:
    pass
layout["9.32"] = [176, 2, 240, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_08_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-10/20_icon_Clear_all.png
try:
    _c20 = get_crop(20, 55, 69)
    canvas.paste(_c20, (1319, 0), _c20)
except Exception:
    pass
layout["Clear_all"] = [1319, 0, 1374, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_08_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-10/21_icon_Clear_all.png
try:
    _c21 = get_crop(21, 99, 69)
    canvas.paste(_c21, (1211, 0), _c21)
except Exception:
    pass
layout["Clear_all"] = [1211, 0, 1310, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_08_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-10/22_icon_9.32.png
try:
    _c22 = get_crop(22, 57, 64)
    canvas.paste(_c22, (114, 1), _c22)
except Exception:
    pass
layout["9.32"] = [114, 1, 171, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_08_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-10/23_icon_icon_23.png
try:
    _c23 = get_crop(23, 59, 63)
    canvas.paste(_c23, (245, 2), _c23)
except Exception:
    pass
layout["icon_23"] = [245, 2, 304, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_08_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-10/24_icon_icon_24.png
try:
    _c24 = get_crop(24, 54, 61)
    canvas.paste(_c24, (314, 3), _c24)
except Exception:
    pass
layout["icon_24"] = [314, 3, 368, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_08_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-10/25_icon_clickable_20.png
try:
    _c25 = get_crop(25, 144, 144)
    canvas.paste(_c25, (1248, 1729), _c25)
except Exception:
    pass
layout["clickable_20"] = [1248, 1729, 1392, 1873]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_08_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-10/26_icon_Clear_all.png
try:
    _c26 = get_crop(26, 178, 144)
    canvas.paste(_c26, (1214, 72), _c26)
except Exception:
    pass
layout["Clear_all"] = [1214, 72, 1392, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_08_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-10/27_text_9.32.png
try:
    _c27 = get_crop(27, 96, 49)
    canvas.paste(_c27, (16, 12), _c27)
except Exception:
    pass
layout["9.32"] = [16, 12, 112, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_08_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-10/28_text_Filters.png
try:
    _c28 = get_crop(28, 180, 66)
    canvas.paste(_c28, (631, 116), _c28)
except Exception:
    pass
layout["Filters"] = [631, 116, 811, 182]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_08_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-10/29_text_Categories.png
try:
    _c29 = get_crop(29, 187, 135)
    canvas.paste(_c29, (36, 383), _c29)
except Exception:
    pass
layout["Categories"] = [36, 383, 223, 518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_08_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-10/30_text_Show_all_categories.png
try:
    _c30 = get_crop(30, 516, 144)
    canvas.paste(_c30, (0, 518), _c30)
except Exception:
    pass
layout["Show_all_categories"] = [0, 518, 516, 662]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_08_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-10/31_text_Event_type.png
try:
    _c31 = get_crop(31, 298, 135)
    canvas.paste(_c31, (36, 829), _c31)
except Exception:
    pass
layout["Event_type"] = [36, 829, 334, 964]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_08_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-10/32_text_Show_all_event_types.png
try:
    _c32 = get_crop(32, 535, 144)
    canvas.paste(_c32, (0, 964), _c32)
except Exception:
    pass
layout["Show_all_event_types"] = [0, 964, 535, 1108]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_08_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-10/33_text_Languages.png
try:
    _c33 = get_crop(33, 210, 135)
    canvas.paste(_c33, (36, 1275), _c33)
except Exception:
    pass
layout["Languages"] = [36, 1275, 246, 1410]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_08_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-10/34_text_Show_all_languages.png
try:
    _c34 = get_crop(34, 511, 144)
    canvas.paste(_c34, (0, 1410), _c34)
except Exception:
    pass
layout["Show_all_languages"] = [0, 1410, 511, 1554]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_08_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-10/35_text_Price.png
try:
    _c35 = get_crop(35, 149, 63)
    canvas.paste(_c35, (45, 1613), _c35)
except Exception:
    pass
layout["Price"] = [45, 1613, 194, 1676]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_08_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-10/36_text_Only_free_events.png
try:
    _c36 = get_crop(36, 660, 144)
    canvas.paste(_c36, (54, 2024), _c36)
except Exception:
    pass
layout["Only_free_events"] = [54, 2024, 714, 2168]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_08_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-10/37_text_Sort_by.png
try:
    _c37 = get_crop(37, 206, 75)
    canvas.paste(_c37, (42, 1931), _c37)
except Exception:
    pass
layout["Sort_by"] = [42, 1931, 248, 2006]
