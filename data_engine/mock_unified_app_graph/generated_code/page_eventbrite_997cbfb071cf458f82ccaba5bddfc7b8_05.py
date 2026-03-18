# page_id: page_eventbrite_997cbfb071cf458f82ccaba5bddfc7b8_05
# screenshot: 2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-7.png
# step_index: 5/15
# task: Open Eventbrite. Search Online free events. Select the first one. Follow the organizer. Read more about the event. Add it to Favorites.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Paint background and structural UI elements for the filters screen
# Uses provided variables: canvas (PIL Image), draw (ImageDraw), font_sm, font_md, font_lg, font_xl

# Colors
bg_color = (250, 251, 252)         # very light off-white background
status_bar_color = (193, 193, 193) # light gray status bar
header_divider = (231, 227, 235)   # very light purple/gray divider under header
section_divider = (238, 236, 240)  # subtle separators between sections
seg_bg = (245, 244, 246)           # segmented control inactive background
seg_inactive = (236, 232, 238)     # segment right inactive fill
seg_active = (255, 255, 255)       # left active segment (white)
button_border = (154, 147, 156)    # bottom apply button border color
button_shadow = (230, 228, 233)    # shadow above button
shadow_color = (220, 218, 224)     # subtle shadow for segmented control

W, H = canvas.size

# Fill overall background
draw.rectangle([(0, 0), (W, H)], fill=bg_color)

# Status bar (top)
status_h = 72
draw.rectangle([(0, 0), (W, status_h)], fill=status_bar_color)

# Header area below status bar (toolbar)
header_top = status_h
header_h = 86
draw.rectangle([(0, header_top), (W, header_top + header_h)], fill=(255, 255, 255))
# header bottom divider
draw.line([(24, header_top + header_h + 1), (W - 24, header_top + header_h + 1)], fill=header_divider, width=1)

# Thin separators between major filter sections
# Positions chosen to match typical spacing in the screenshot
separators = [
    560,  # after categories block
    920,  # after event type block
    1360, # after languages block
    1760, # after price/toggle area
    2190, # subtle line near sort control area
]
for y in separators:
    draw.line([(24, y), (W - 24, y)], fill=section_divider, width=1)

# Draw segmented "Sort by" control background with subtle shadow
seg_outer_left = 48
seg_outer_right = W - 48
seg_outer_top = 2018
seg_outer_bottom = seg_outer_top + 156  # slightly taller to accommodate shadow
seg_radius = 18

# shadow for segmented control
draw.rounded_rectangle(
    [(seg_outer_left + 2, seg_outer_top + 6), (seg_outer_right + 2, seg_outer_bottom + 6)],
    radius=seg_radius + 2,
    fill=shadow_color
)

# outer background (light)
draw.rounded_rectangle(
    [(seg_outer_left, seg_outer_top), (seg_outer_right, seg_outer_bottom)],
    radius=seg_radius,
    fill=seg_bg,
    outline=None
)

# draw left and right segments
seg_width = (seg_outer_right - seg_outer_left)
left_box = (seg_outer_left + 4, seg_outer_top + 4, seg_outer_left + seg_width//2 - 2, seg_outer_bottom - 4)
right_box = (seg_outer_left + seg_width//2 + 2, seg_outer_top + 4, seg_outer_right - 4, seg_outer_bottom - 4)

# left active (white)
draw.rounded_rectangle(left_box, radius=seg_radius - 6, fill=seg_active)
# right inactive
draw.rounded_rectangle(right_box, radius=seg_radius - 6, fill=seg_inactive)

# inner separator line between segments
mid_x = seg_outer_left + seg_width//2
draw.line([(mid_x, seg_outer_top + 6), (mid_x, seg_outer_bottom - 6)], fill=section_divider, width=1)

# Slight inner shadow under left active segment to emulate pressed effect
draw.line([(left_box[0]+6, left_box[3]-4), (left_box[2]-6, left_box[3]-4)], fill=shadow_color, width=1)

# Bottom "Apply filters" button background and border
btn_left = 48
btn_top = 2768
btn_right = W - 48
btn_bottom = btn_top + 144
btn_radius = 14

# subtle shadow above button for elevation
draw.rectangle([(btn_left + 2, btn_top - 4), (btn_right - 2, btn_top - 1)], fill=button_shadow)

# button fill (white to match screen)
draw.rounded_rectangle([(btn_left, btn_top), (btn_right, btn_bottom)], radius=btn_radius, fill=(255,255,255), outline=button_border, width=3)

# Add subtle horizontal separators above some groups to guide layout (visuals only)
# A faint line under the header title region
draw.line([(24, header_top + header_h + 28), (W - 24, header_top + header_h + 28)], fill=section_divider, width=1)

# Small card-like backgrounds behind larger content areas (subtle)
# For example behind the 'Price' toggle area (do not draw the toggle itself)
price_card_left = 24
price_card_right = W - 24
price_card_top = 1540
price_card_bottom = 1820
draw.rectangle([(price_card_left, price_card_top), (price_card_right, price_card_bottom)], fill=(250,250,251))

# Another subtle block behind language chips area
lang_block_top = 1220
lang_block_bottom = 1460
draw.rectangle([(24, lang_block_top), (W - 24, lang_block_bottom)], fill=(250,250,251))

# Very subtle vertical padding guides (not visible strongly)
for x in (24, W - 24):
    draw.line([(x, header_top), (x, H)], fill=(255,255,255), width=0)

# End of structural drawing

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_05_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-7/00_icon_Food_Drink.png
try:
    _c0 = get_crop(0, 312, 144)
    canvas.paste(_c0, (512, 383), _c0)
except Exception:
    pass
layout["Food_&_Drink"] = [512, 383, 824, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_05_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-7/01_icon_Music.png
try:
    _c1 = get_crop(1, 187, 135)
    canvas.paste(_c1, (36, 383), _c1)
except Exception:
    pass
layout["Music"] = [36, 383, 223, 518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_05_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-7/02_icon_Community.png
try:
    _c2 = get_crop(2, 294, 144)
    canvas.paste(_c2, (848, 383), _c2)
except Exception:
    pass
layout["Community"] = [848, 383, 1142, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_05_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-7/03_icon_French.png
try:
    _c3 = get_crop(3, 205, 144)
    canvas.paste(_c3, (768, 1275), _c3)
except Exception:
    pass
layout["French"] = [768, 1275, 973, 1419]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_05_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-7/04_icon_Business.png
try:
    _c4 = get_crop(4, 241, 135)
    canvas.paste(_c4, (247, 383), _c4)
except Exception:
    pass
layout["Business"] = [247, 383, 488, 518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_05_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-7/05_icon_Spanish.png
try:
    _c5 = get_crop(5, 225, 144)
    canvas.paste(_c5, (519, 1275), _c5)
except Exception:
    pass
layout["Spanish"] = [519, 1275, 744, 1419]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_05_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-7/06_icon_Expo.png
try:
    _c6 = get_crop(6, 167, 144)
    canvas.paste(_c6, (614, 829), _c6)
except Exception:
    pass
layout["Expo"] = [614, 829, 781, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_05_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-7/07_icon_Seminar.png
try:
    _c7 = get_crop(7, 232, 144)
    canvas.paste(_c7, (358, 829), _c7)
except Exception:
    pass
layout["Seminar"] = [358, 829, 590, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_05_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-7/08_icon_Italian.png
try:
    _c8 = get_crop(8, 191, 144)
    canvas.paste(_c8, (997, 1275), _c8)
except Exception:
    pass
layout["Italian"] = [997, 1275, 1188, 1419]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_05_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-7/09_icon_Arts.png
try:
    _c9 = get_crop(9, 152, 144)
    canvas.paste(_c9, (1166, 383), _c9)
except Exception:
    pass
layout["Arts"] = [1166, 383, 1318, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_05_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-7/10_icon_Convention.png
try:
    _c10 = get_crop(10, 293, 144)
    canvas.paste(_c10, (805, 829), _c10)
except Exception:
    pass
layout["Convention"] = [805, 829, 1098, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_05_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-7/11_icon_German.png
try:
    _c11 = get_crop(11, 225, 135)
    canvas.paste(_c11, (270, 1275), _c11)
except Exception:
    pass
layout["German"] = [270, 1275, 495, 1410]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_05_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-7/12_icon_Festival.png
try:
    _c12 = get_crop(12, 219, 144)
    canvas.paste(_c12, (1122, 829), _c12)
except Exception:
    pass
layout["Festival"] = [1122, 829, 1341, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_05_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-7/13_icon_English.png
try:
    _c13 = get_crop(13, 210, 135)
    canvas.paste(_c13, (36, 1275), _c13)
except Exception:
    pass
layout["English"] = [36, 1275, 246, 1410]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_05_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-7/14_icon_Conference.png
try:
    _c14 = get_crop(14, 298, 135)
    canvas.paste(_c14, (36, 829), _c14)
except Exception:
    pass
layout["Conference"] = [36, 829, 334, 964]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_05_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-7/15_icon_Date.png
try:
    _c15 = get_crop(15, 660, 144)
    canvas.paste(_c15, (726, 2024), _c15)
except Exception:
    pass
layout["Date"] = [726, 2024, 1386, 2168]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_05_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-7/16_icon_Relevance.png
try:
    _c16 = get_crop(16, 660, 144)
    canvas.paste(_c16, (54, 2024), _c16)
except Exception:
    pass
layout["Relevance"] = [54, 2024, 714, 2168]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_05_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-7/17_icon_Apply_filters.png
try:
    _c17 = get_crop(17, 1344, 144)
    canvas.paste(_c17, (48, 2768), _c17)
except Exception:
    pass
layout["Apply_filters"] = [48, 2768, 1392, 2912]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_05_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-7/18_icon_9.15.png
try:
    _c18 = get_crop(18, 144, 144)
    canvas.paste(_c18, (12, 72), _c18)
except Exception:
    pass
layout["9.15"] = [12, 72, 156, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_05_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-7/19_icon_9.15.png
try:
    _c19 = get_crop(19, 64, 64)
    canvas.paste(_c19, (176, 1), _c19)
except Exception:
    pass
layout["9.15"] = [176, 1, 240, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_05_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-7/20_icon_Clear_all.png
try:
    _c20 = get_crop(20, 55, 69)
    canvas.paste(_c20, (1319, 0), _c20)
except Exception:
    pass
layout["Clear_all"] = [1319, 0, 1374, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_05_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-7/21_icon_Clear_all.png
try:
    _c21 = get_crop(21, 99, 69)
    canvas.paste(_c21, (1211, 0), _c21)
except Exception:
    pass
layout["Clear_all"] = [1211, 0, 1310, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_05_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-7/22_icon_9.15.png
try:
    _c22 = get_crop(22, 56, 64)
    canvas.paste(_c22, (115, 1), _c22)
except Exception:
    pass
layout["9.15"] = [115, 1, 171, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_05_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-7/23_icon_icon_23.png
try:
    _c23 = get_crop(23, 59, 63)
    canvas.paste(_c23, (245, 2), _c23)
except Exception:
    pass
layout["icon_23"] = [245, 2, 304, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_05_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-7/24_icon_icon_24.png
try:
    _c24 = get_crop(24, 54, 61)
    canvas.paste(_c24, (314, 3), _c24)
except Exception:
    pass
layout["icon_24"] = [314, 3, 368, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_05_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-7/25_icon_clickable_20.png
try:
    _c25 = get_crop(25, 144, 144)
    canvas.paste(_c25, (1248, 1729), _c25)
except Exception:
    pass
layout["clickable_20"] = [1248, 1729, 1392, 1873]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_05_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-7/26_icon_Clear_all.png
try:
    _c26 = get_crop(26, 178, 144)
    canvas.paste(_c26, (1214, 72), _c26)
except Exception:
    pass
layout["Clear_all"] = [1214, 72, 1392, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_05_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-7/27_text_9.15.png
try:
    _c27 = get_crop(27, 94, 43)
    canvas.paste(_c27, (20, 17), _c27)
except Exception:
    pass
layout["9.15"] = [20, 17, 114, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_05_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-7/28_text_Filters.png
try:
    _c28 = get_crop(28, 180, 66)
    canvas.paste(_c28, (631, 116), _c28)
except Exception:
    pass
layout["Filters"] = [631, 116, 811, 182]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_05_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-7/29_text_Categories.png
try:
    _c29 = get_crop(29, 187, 135)
    canvas.paste(_c29, (36, 383), _c29)
except Exception:
    pass
layout["Categories"] = [36, 383, 223, 518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_05_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-7/30_text_Show_all_categories.png
try:
    _c30 = get_crop(30, 516, 144)
    canvas.paste(_c30, (0, 518), _c30)
except Exception:
    pass
layout["Show_all_categories"] = [0, 518, 516, 662]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_05_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-7/31_text_Event_type.png
try:
    _c31 = get_crop(31, 298, 135)
    canvas.paste(_c31, (36, 829), _c31)
except Exception:
    pass
layout["Event_type"] = [36, 829, 334, 964]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_05_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-7/32_text_Show_all_event_types.png
try:
    _c32 = get_crop(32, 535, 144)
    canvas.paste(_c32, (0, 964), _c32)
except Exception:
    pass
layout["Show_all_event_types"] = [0, 964, 535, 1108]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_05_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-7/33_text_Languages.png
try:
    _c33 = get_crop(33, 210, 135)
    canvas.paste(_c33, (36, 1275), _c33)
except Exception:
    pass
layout["Languages"] = [36, 1275, 246, 1410]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_05_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-7/34_text_Show_all_languages.png
try:
    _c34 = get_crop(34, 511, 144)
    canvas.paste(_c34, (0, 1410), _c34)
except Exception:
    pass
layout["Show_all_languages"] = [0, 1410, 511, 1554]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_05_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-7/35_text_Price.png
try:
    _c35 = get_crop(35, 149, 63)
    canvas.paste(_c35, (45, 1613), _c35)
except Exception:
    pass
layout["Price"] = [45, 1613, 194, 1676]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_05_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-7/36_text_Only_free_events.png
try:
    _c36 = get_crop(36, 660, 144)
    canvas.paste(_c36, (54, 2024), _c36)
except Exception:
    pass
layout["Only_free_events"] = [54, 2024, 714, 2168]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_05_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-7/37_text_Sort_by.png
try:
    _c37 = get_crop(37, 206, 75)
    canvas.paste(_c37, (42, 1931), _c37)
except Exception:
    pass
layout["Sort_by"] = [42, 1931, 248, 2006]
