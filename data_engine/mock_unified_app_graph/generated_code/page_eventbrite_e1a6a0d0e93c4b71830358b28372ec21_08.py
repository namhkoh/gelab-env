# page_id: page_eventbrite_e1a6a0d0e93c4b71830358b28372ec21_08
# screenshot: 2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-10.png
# step_index: 8/9
# task: Open Eventbrite. Search for "Language Learning". Filter only online events. Note how many events are available for "Spanish".
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Background and structural UI drawing for the provided canvas (1440x2960).
# Uses provided 'canvas' (PIL Image) and 'draw' (ImageDraw) objects.

# Fill overall background (dominant color: white)
draw.rectangle((0, 0, 1440, 2960), fill=(255, 255, 255))

# Status bar area (top ~72px) - light grey
status_h = 72
draw.rectangle((0, 0, 1440, status_h), fill=(189, 189, 189))

# Header / toolbar area under status bar (~72-160)
header_top = status_h
header_bottom = 160
draw.rectangle((0, header_top, 1440, header_bottom), fill=(255, 255, 255))

# Subtle divider under header
draw.line((24, header_bottom, 1416, header_bottom), fill=(230, 230, 235), width=2)

# Section card outlines (rounded rectangles) to group areas without drawing any icons/text.
# These are subtle outlines only (no fills) so they don't duplicate icon/text crops.
outline_color = (235, 236, 240)
outline_width = 2
radius = 18

# Categories group outline (around the category chips area)
cat_top = 260
cat_bottom = 560
draw.rounded_rectangle((28, cat_top, 1412, cat_bottom),
                       radius=radius, outline=outline_color, width=outline_width)

# Event type group outline
etype_top = 700
etype_bottom = 1040
draw.rounded_rectangle((28, etype_top, 1412, etype_bottom),
                       radius=radius, outline=outline_color, width=outline_width)

# Languages group outline
lang_top = 1130
lang_bottom = 1470
draw.rounded_rectangle((28, lang_top, 1412, lang_bottom),
                       radius=radius, outline=outline_color, width=outline_width)

# Price & Only free events area outline (taller to include toggle area)
price_top = 1540
price_bottom = 2110
draw.rounded_rectangle((28, price_top, 1412, price_bottom),
                       radius=16, outline=outline_color, width=outline_width)

# Sort by segmented control outer outline (subtle, only the outline)
seg_top = 1988
seg_bottom = 2108
seg_left = 36
seg_right = 1404
draw.rounded_rectangle((seg_left, seg_top, seg_right, seg_bottom),
                       radius=14, outline=(220, 221, 226), width=2)

# Subtle inner separator lines between major sections (light dividers)
divider_color = (242, 243, 245)
dividers_y = [
    240,   # just below header content area
    600,   # after categories
    1000,  # after event types
    1460,  # after languages
    1530,  # before price area
    1910,  # before sort area
    2736   # just above apply filters bar
]
for y in dividers_y:
    draw.line((24, y, 1416, y), fill=divider_color, width=1)

# Light drop shadows to add subtle separation under some outlines
shadow_color = (240, 241, 244, 120)
# small translucent rectangles to simulate shadow (use semi-opaque by blending onto canvas)
# Since we can't use new layers, draw thin soft shadows using slightly darker lines
for i, offset in enumerate([1, 2, 3]):
    alpha_shade = 8 + i * 8
    shade_color = (235 - i*3, 236 - i*3, 240 - i*3)
    # shadow under header divider
    draw.line((24, header_bottom + offset, 1416, header_bottom + offset), fill=shade_color, width=1)
    # shadow under apply-filters top area
    draw.line((24, 2736 + offset, 1416, 2736 + offset), fill=shade_color, width=1)

# Bottom 'apply filters' top divider (so pasted button appears separated)
draw.line((24, 2768, 1416, 2768), fill=(225, 226, 230), width=3)

# A faint large content-area background block (very subtle off-white) for the middle content area
# This helps visually structure the page without overlapping detected elements.
content_bg_top = header_bottom + 24
content_bg_bottom = 2680
draw.rectangle((12, content_bg_top, 1428, content_bg_bottom), fill=(255, 255, 255))

# Small left gutter accent line for visual rhythm (non-intrusive)
draw.line((24, content_bg_top + 20, 24, content_bg_bottom - 20), fill=(245, 246, 248), width=2)

# Final tiny highlight at very top edge of status bar to mimic device bezel
draw.line((0, status_h - 1, 1440, status_h - 1), fill=(200, 200, 200), width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_08_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-10/00_icon_Spanish.png
try:
    _c0 = get_crop(0, 225, 144)
    canvas.paste(_c0, (519, 1275), _c0)
except Exception:
    pass
layout["Spanish"] = [519, 1275, 744, 1419]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_08_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-10/01_icon_Food_Drink.png
try:
    _c1 = get_crop(1, 312, 144)
    canvas.paste(_c1, (512, 383), _c1)
except Exception:
    pass
layout["Food_&_Drink"] = [512, 383, 824, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_08_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-10/02_icon_Music.png
try:
    _c2 = get_crop(2, 187, 135)
    canvas.paste(_c2, (36, 383), _c2)
except Exception:
    pass
layout["Music"] = [36, 383, 223, 518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_08_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-10/03_icon_Community.png
try:
    _c3 = get_crop(3, 294, 144)
    canvas.paste(_c3, (848, 383), _c3)
except Exception:
    pass
layout["Community"] = [848, 383, 1142, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_08_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-10/04_icon_Business.png
try:
    _c4 = get_crop(4, 241, 135)
    canvas.paste(_c4, (247, 383), _c4)
except Exception:
    pass
layout["Business"] = [247, 383, 488, 518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_08_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-10/05_icon_French.png
try:
    _c5 = get_crop(5, 205, 144)
    canvas.paste(_c5, (768, 1275), _c5)
except Exception:
    pass
layout["French"] = [768, 1275, 973, 1419]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_08_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-10/06_icon_Expo.png
try:
    _c6 = get_crop(6, 167, 144)
    canvas.paste(_c6, (614, 829), _c6)
except Exception:
    pass
layout["Expo"] = [614, 829, 781, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_08_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-10/07_icon_Arts.png
try:
    _c7 = get_crop(7, 152, 144)
    canvas.paste(_c7, (1166, 383), _c7)
except Exception:
    pass
layout["Arts"] = [1166, 383, 1318, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_08_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-10/08_icon_Italian.png
try:
    _c8 = get_crop(8, 191, 144)
    canvas.paste(_c8, (997, 1275), _c8)
except Exception:
    pass
layout["Italian"] = [997, 1275, 1188, 1419]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_08_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-10/09_icon_German.png
try:
    _c9 = get_crop(9, 225, 135)
    canvas.paste(_c9, (270, 1275), _c9)
except Exception:
    pass
layout["German"] = [270, 1275, 495, 1410]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_08_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-10/10_icon_Seminar.png
try:
    _c10 = get_crop(10, 232, 144)
    canvas.paste(_c10, (358, 829), _c10)
except Exception:
    pass
layout["Seminar"] = [358, 829, 590, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_08_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-10/11_icon_Festival.png
try:
    _c11 = get_crop(11, 219, 144)
    canvas.paste(_c11, (1122, 829), _c11)
except Exception:
    pass
layout["Festival"] = [1122, 829, 1341, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_08_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-10/12_icon_Convention.png
try:
    _c12 = get_crop(12, 293, 144)
    canvas.paste(_c12, (805, 829), _c12)
except Exception:
    pass
layout["Convention"] = [805, 829, 1098, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_08_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-10/13_icon_English.png
try:
    _c13 = get_crop(13, 210, 135)
    canvas.paste(_c13, (36, 1275), _c13)
except Exception:
    pass
layout["English"] = [36, 1275, 246, 1410]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_08_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-10/14_icon_Conference.png
try:
    _c14 = get_crop(14, 298, 135)
    canvas.paste(_c14, (36, 829), _c14)
except Exception:
    pass
layout["Conference"] = [36, 829, 334, 964]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_08_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-10/15_icon_Apply_filters_1.png
try:
    _c15 = get_crop(15, 1344, 144)
    canvas.paste(_c15, (48, 2768), _c15)
except Exception:
    pass
layout["Apply_filters_(1)"] = [48, 2768, 1392, 2912]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_08_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-10/16_icon_Date.png
try:
    _c16 = get_crop(16, 660, 144)
    canvas.paste(_c16, (726, 2024), _c16)
except Exception:
    pass
layout["Date"] = [726, 2024, 1386, 2168]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_08_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-10/17_icon_Relevance.png
try:
    _c17 = get_crop(17, 660, 144)
    canvas.paste(_c17, (54, 2024), _c17)
except Exception:
    pass
layout["Relevance"] = [54, 2024, 714, 2168]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_08_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-10/18_icon_5.18.png
try:
    _c18 = get_crop(18, 144, 144)
    canvas.paste(_c18, (12, 72), _c18)
except Exception:
    pass
layout["5.18"] = [12, 72, 156, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_08_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-10/19_icon_5.18.png
try:
    _c19 = get_crop(19, 60, 64)
    canvas.paste(_c19, (180, 1), _c19)
except Exception:
    pass
layout["5.18"] = [180, 1, 240, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_08_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-10/20_icon_icon_20.png
try:
    _c20 = get_crop(20, 64, 62)
    canvas.paste(_c20, (308, 3), _c20)
except Exception:
    pass
layout["icon_20"] = [308, 3, 372, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_08_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-10/21_icon_5.18.png
try:
    _c21 = get_crop(21, 64, 65)
    canvas.paste(_c21, (112, 1), _c21)
except Exception:
    pass
layout["5.18"] = [112, 1, 176, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_08_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-10/22_icon_Clear_all.png
try:
    _c22 = get_crop(22, 56, 67)
    canvas.paste(_c22, (1318, 0), _c22)
except Exception:
    pass
layout["Clear_all"] = [1318, 0, 1374, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_08_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-10/23_icon_Clear_all.png
try:
    _c23 = get_crop(23, 99, 65)
    canvas.paste(_c23, (1211, 0), _c23)
except Exception:
    pass
layout["Clear_all"] = [1211, 0, 1310, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_08_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-10/24_icon_icon_24.png
try:
    _c24 = get_crop(24, 51, 61)
    canvas.paste(_c24, (248, 3), _c24)
except Exception:
    pass
layout["icon_24"] = [248, 3, 299, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_08_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-10/25_icon_clickable_20.png
try:
    _c25 = get_crop(25, 144, 144)
    canvas.paste(_c25, (1248, 1729), _c25)
except Exception:
    pass
layout["clickable_20"] = [1248, 1729, 1392, 1873]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_08_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-10/26_icon_Clear_all.png
try:
    _c26 = get_crop(26, 178, 144)
    canvas.paste(_c26, (1214, 72), _c26)
except Exception:
    pass
layout["Clear_all"] = [1214, 72, 1392, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_08_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-10/27_text_5.18.png
try:
    _c27 = get_crop(27, 91, 45)
    canvas.paste(_c27, (20, 15), _c27)
except Exception:
    pass
layout["5.18"] = [20, 15, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_08_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-10/28_text_Filters.png
try:
    _c28 = get_crop(28, 180, 66)
    canvas.paste(_c28, (631, 116), _c28)
except Exception:
    pass
layout["Filters"] = [631, 116, 811, 182]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_08_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-10/29_text_Categories.png
try:
    _c29 = get_crop(29, 187, 135)
    canvas.paste(_c29, (36, 383), _c29)
except Exception:
    pass
layout["Categories"] = [36, 383, 223, 518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_08_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-10/30_text_Show_all_categories.png
try:
    _c30 = get_crop(30, 516, 144)
    canvas.paste(_c30, (0, 518), _c30)
except Exception:
    pass
layout["Show_all_categories"] = [0, 518, 516, 662]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_08_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-10/31_text_Event_type.png
try:
    _c31 = get_crop(31, 298, 135)
    canvas.paste(_c31, (36, 829), _c31)
except Exception:
    pass
layout["Event_type"] = [36, 829, 334, 964]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_08_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-10/32_text_Show_all_event_types.png
try:
    _c32 = get_crop(32, 535, 144)
    canvas.paste(_c32, (0, 964), _c32)
except Exception:
    pass
layout["Show_all_event_types"] = [0, 964, 535, 1108]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_08_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-10/33_text_Languages.png
try:
    _c33 = get_crop(33, 210, 135)
    canvas.paste(_c33, (36, 1275), _c33)
except Exception:
    pass
layout["Languages"] = [36, 1275, 246, 1410]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_08_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-10/34_text_Show_all_languages.png
try:
    _c34 = get_crop(34, 511, 144)
    canvas.paste(_c34, (0, 1410), _c34)
except Exception:
    pass
layout["Show_all_languages"] = [0, 1410, 511, 1554]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_08_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-10/35_text_Price.png
try:
    _c35 = get_crop(35, 149, 63)
    canvas.paste(_c35, (45, 1613), _c35)
except Exception:
    pass
layout["Price"] = [45, 1613, 194, 1676]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_08_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-10/36_text_Only_free_events.png
try:
    _c36 = get_crop(36, 660, 144)
    canvas.paste(_c36, (54, 2024), _c36)
except Exception:
    pass
layout["Only_free_events"] = [54, 2024, 714, 2168]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_08_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-10/37_text_Sort_by.png
try:
    _c37 = get_crop(37, 206, 75)
    canvas.paste(_c37, (42, 1931), _c37)
except Exception:
    pass
layout["Sort_by"] = [42, 1931, 248, 2006]
