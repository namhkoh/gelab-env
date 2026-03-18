# page_id: page_eventbrite_86c0bd1901f44c94916665f4058f9b6d_08
# screenshot: 2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-10.png
# step_index: 8/11
# task: Open Eventbrite. Set the city to Los Angeles. Select the 'Food & Drink' category. What's the date of the first event that is not promoted?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Top status bar
status_h = 72
status_color = (185, 185, 185)        # muted gray status bar
draw.rectangle([(0, 0), (1440, status_h)], fill=status_color)

# Header / toolbar area (below status bar)
header_top = status_h
header_bottom = 152
header_bg = (255, 255, 255)           # white header
draw.rectangle([(0, header_top), (1440, header_bottom)], fill=header_bg)

# subtle header divider / shadow
divider_color = (230, 230, 235)
draw.line([(0, header_bottom), (1440, header_bottom)], fill=divider_color, width=2)

# Light overall background (canvas already white, but ensure uniform)
draw.rectangle([(0, header_bottom+1), (1440, 2960)], fill=(255,255,255))

# Section separators (thin subtle lines) — positioned to match visual grouping
sep_color = (242, 242, 245)
separators = [660, 1000, 1410, 1650, 2048]  # y positions between major sections
for y in separators:
    draw.line([(32, y), (1408, y)], fill=sep_color, width=2)

# Card-like subtle background blocks for grouped sections (rounded rectangles)
card_fill = (250, 250, 252)  # very light
card_outline = (240, 238, 244)

# Categories group background (large light area behind category chips)
draw.rounded_rectangle([(24, 260), (1416, 560)], radius=18, fill=None, outline=None)

# Event type group background
draw.rounded_rectangle([(24, 700), (1416, 1000)], radius=18, fill=None, outline=None)

# Languages group background
draw.rounded_rectangle([(24, 1100), (1416, 1410)], radius=18, fill=None, outline=None)

# Price / Toggle area background
draw.rounded_rectangle([(24, 1520), (1416, 1740)], radius=14, fill=None, outline=None)

# Sort-by segmented control container (subtle rounded pill background)
seg_top = 1988
seg_bottom = 2088
seg_left = 36
seg_right = 1404
seg_bg = (245, 243, 247)   # faint lavender/gray for segmented control background
seg_border = (220, 218, 225)
draw.rounded_rectangle([(seg_left, seg_top), (seg_right, seg_bottom)], radius=14, fill=seg_bg, outline=seg_border, width=2)

# Left and right segments visual separation (keep it subtle; labels will be pasted on top)
mid_x = (seg_left + seg_right) // 2
draw.line([(mid_x, seg_top+6), (mid_x, seg_bottom-6)], fill=seg_border, width=1)

# Subtle inner shadow under segmented control
shadow_y = seg_bottom + 6
draw.line([(seg_left+6, shadow_y), (seg_right-6, shadow_y)], fill=(245,245,247), width=2)

# Bottom "Apply filters" area: draw a faint container / shadow behind the actual button
apply_top = 2748
apply_bottom = 2920
apply_left = 40
apply_right = 1400
# faint shadow under the button area
draw.rectangle([(apply_left+4, apply_top+6), (apply_right+4, apply_bottom+6)], fill=(240,240,242))
# rounded border frame where button will sit (button will be pasted)
draw.rounded_rectangle([(apply_left, apply_top), (apply_right, apply_bottom)], radius=12, fill=None, outline=(200,198,204), width=3)

# Edge accents and subtle vertical separators for visual structure
accent_color = (245,245,247)
draw.line([(32, 220), (32, 2860)], fill=accent_color, width=2)
draw.line([(1408, 220), (1408, 2860)], fill=accent_color, width=2)

# A subtle large faint background block for the central content area (keeps visual balance)
draw.rectangle([(0, 2088), (1440, 2720)], fill=(255,255,255))

# top-left corner decorative rounded rectangle (background only, no icons/text)
draw.rounded_rectangle([(20, 20), (220, 68)], radius=12, fill=(230,230,230), outline=None)

# finalize minor divider lines to match app subtlety
for y in [188, 440, 760, 1140, 1560]:
    draw.line([(32, y), (1408, y)], fill=(248,248,249), width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_08_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-10/00_icon_Food_Drink.png
try:
    _c0 = get_crop(0, 312, 144)
    canvas.paste(_c0, (512, 383), _c0)
except Exception:
    pass
layout["Food_&_Drink"] = [512, 383, 824, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_08_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-10/01_icon_Music.png
try:
    _c1 = get_crop(1, 187, 135)
    canvas.paste(_c1, (36, 383), _c1)
except Exception:
    pass
layout["Music"] = [36, 383, 223, 518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_08_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-10/02_icon_Community.png
try:
    _c2 = get_crop(2, 294, 144)
    canvas.paste(_c2, (848, 383), _c2)
except Exception:
    pass
layout["Community"] = [848, 383, 1142, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_08_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-10/03_icon_French.png
try:
    _c3 = get_crop(3, 205, 144)
    canvas.paste(_c3, (768, 1275), _c3)
except Exception:
    pass
layout["French"] = [768, 1275, 973, 1419]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_08_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-10/04_icon_Business.png
try:
    _c4 = get_crop(4, 241, 135)
    canvas.paste(_c4, (247, 383), _c4)
except Exception:
    pass
layout["Business"] = [247, 383, 488, 518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_08_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-10/05_icon_Spanish.png
try:
    _c5 = get_crop(5, 225, 144)
    canvas.paste(_c5, (519, 1275), _c5)
except Exception:
    pass
layout["Spanish"] = [519, 1275, 744, 1419]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_08_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-10/06_icon_Expo.png
try:
    _c6 = get_crop(6, 167, 144)
    canvas.paste(_c6, (614, 829), _c6)
except Exception:
    pass
layout["Expo"] = [614, 829, 781, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_08_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-10/07_icon_Italian.png
try:
    _c7 = get_crop(7, 191, 144)
    canvas.paste(_c7, (997, 1275), _c7)
except Exception:
    pass
layout["Italian"] = [997, 1275, 1188, 1419]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_08_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-10/08_icon_Seminar.png
try:
    _c8 = get_crop(8, 232, 144)
    canvas.paste(_c8, (358, 829), _c8)
except Exception:
    pass
layout["Seminar"] = [358, 829, 590, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_08_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-10/09_icon_Arts.png
try:
    _c9 = get_crop(9, 152, 144)
    canvas.paste(_c9, (1166, 383), _c9)
except Exception:
    pass
layout["Arts"] = [1166, 383, 1318, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_08_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-10/10_icon_Convention.png
try:
    _c10 = get_crop(10, 293, 144)
    canvas.paste(_c10, (805, 829), _c10)
except Exception:
    pass
layout["Convention"] = [805, 829, 1098, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_08_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-10/11_icon_Festival.png
try:
    _c11 = get_crop(11, 219, 144)
    canvas.paste(_c11, (1122, 829), _c11)
except Exception:
    pass
layout["Festival"] = [1122, 829, 1341, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_08_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-10/12_icon_German.png
try:
    _c12 = get_crop(12, 225, 135)
    canvas.paste(_c12, (270, 1275), _c12)
except Exception:
    pass
layout["German"] = [270, 1275, 495, 1410]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_08_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-10/13_icon_English.png
try:
    _c13 = get_crop(13, 210, 135)
    canvas.paste(_c13, (36, 1275), _c13)
except Exception:
    pass
layout["English"] = [36, 1275, 246, 1410]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_08_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-10/14_icon_Conference.png
try:
    _c14 = get_crop(14, 298, 135)
    canvas.paste(_c14, (36, 829), _c14)
except Exception:
    pass
layout["Conference"] = [36, 829, 334, 964]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_08_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-10/15_icon_Date.png
try:
    _c15 = get_crop(15, 660, 144)
    canvas.paste(_c15, (726, 2024), _c15)
except Exception:
    pass
layout["Date"] = [726, 2024, 1386, 2168]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_08_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-10/16_icon_Relevance.png
try:
    _c16 = get_crop(16, 660, 144)
    canvas.paste(_c16, (54, 2024), _c16)
except Exception:
    pass
layout["Relevance"] = [54, 2024, 714, 2168]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_08_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-10/17_icon_Clear_all.png
try:
    _c17 = get_crop(17, 51, 70)
    canvas.paste(_c17, (1153, 1), _c17)
except Exception:
    pass
layout["Clear_all"] = [1153, 1, 1204, 71]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_08_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-10/18_icon_Apply_filters.png
try:
    _c18 = get_crop(18, 1344, 144)
    canvas.paste(_c18, (48, 2768), _c18)
except Exception:
    pass
layout["Apply_filters"] = [48, 2768, 1392, 2912]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_08_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-10/19_icon_Clear_all.png
try:
    _c19 = get_crop(19, 100, 70)
    canvas.paste(_c19, (1211, 0), _c19)
except Exception:
    pass
layout["Clear_all"] = [1211, 0, 1311, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_08_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-10/20_icon_7.13.png
try:
    _c20 = get_crop(20, 144, 144)
    canvas.paste(_c20, (12, 72), _c20)
except Exception:
    pass
layout["7.13"] = [12, 72, 156, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_08_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-10/21_icon_7.13.png
try:
    _c21 = get_crop(21, 61, 64)
    canvas.paste(_c21, (180, 1), _c21)
except Exception:
    pass
layout["7.13"] = [180, 1, 241, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_08_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-10/22_icon_icon_22.png
try:
    _c22 = get_crop(22, 64, 62)
    canvas.paste(_c22, (308, 3), _c22)
except Exception:
    pass
layout["icon_22"] = [308, 3, 372, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_08_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-10/23_icon_Clear_all.png
try:
    _c23 = get_crop(23, 52, 69)
    canvas.paste(_c23, (1320, 0), _c23)
except Exception:
    pass
layout["Clear_all"] = [1320, 0, 1372, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_08_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-10/24_icon_7.13.png
try:
    _c24 = get_crop(24, 64, 66)
    canvas.paste(_c24, (112, 0), _c24)
except Exception:
    pass
layout["7.13"] = [112, 0, 176, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_08_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-10/25_icon_icon_25.png
try:
    _c25 = get_crop(25, 51, 62)
    canvas.paste(_c25, (248, 2), _c25)
except Exception:
    pass
layout["icon_25"] = [248, 2, 299, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_08_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-10/26_icon_Clear_all.png
try:
    _c26 = get_crop(26, 178, 144)
    canvas.paste(_c26, (1214, 72), _c26)
except Exception:
    pass
layout["Clear_all"] = [1214, 72, 1392, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_08_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-10/27_icon_clickable_20.png
try:
    _c27 = get_crop(27, 144, 144)
    canvas.paste(_c27, (1248, 1729), _c27)
except Exception:
    pass
layout["clickable_20"] = [1248, 1729, 1392, 1873]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_08_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-10/28_text_7.13.png
try:
    _c28 = get_crop(28, 91, 45)
    canvas.paste(_c28, (20, 15), _c28)
except Exception:
    pass
layout["7.13"] = [20, 15, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_08_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-10/29_text_Filters.png
try:
    _c29 = get_crop(29, 180, 66)
    canvas.paste(_c29, (631, 116), _c29)
except Exception:
    pass
layout["Filters"] = [631, 116, 811, 182]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_08_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-10/30_text_Categories.png
try:
    _c30 = get_crop(30, 187, 135)
    canvas.paste(_c30, (36, 383), _c30)
except Exception:
    pass
layout["Categories"] = [36, 383, 223, 518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_08_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-10/31_text_Show_all_categories.png
try:
    _c31 = get_crop(31, 516, 144)
    canvas.paste(_c31, (0, 518), _c31)
except Exception:
    pass
layout["Show_all_categories"] = [0, 518, 516, 662]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_08_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-10/32_text_Event_type.png
try:
    _c32 = get_crop(32, 298, 135)
    canvas.paste(_c32, (36, 829), _c32)
except Exception:
    pass
layout["Event_type"] = [36, 829, 334, 964]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_08_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-10/33_text_Show_all_event_types.png
try:
    _c33 = get_crop(33, 535, 144)
    canvas.paste(_c33, (0, 964), _c33)
except Exception:
    pass
layout["Show_all_event_types"] = [0, 964, 535, 1108]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_08_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-10/34_text_Languages.png
try:
    _c34 = get_crop(34, 210, 135)
    canvas.paste(_c34, (36, 1275), _c34)
except Exception:
    pass
layout["Languages"] = [36, 1275, 246, 1410]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_08_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-10/35_text_Show_all_languages.png
try:
    _c35 = get_crop(35, 511, 144)
    canvas.paste(_c35, (0, 1410), _c35)
except Exception:
    pass
layout["Show_all_languages"] = [0, 1410, 511, 1554]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_08_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-10/36_text_Price.png
try:
    _c36 = get_crop(36, 149, 63)
    canvas.paste(_c36, (45, 1613), _c36)
except Exception:
    pass
layout["Price"] = [45, 1613, 194, 1676]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_08_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-10/37_text_Only_free_events.png
try:
    _c37 = get_crop(37, 660, 144)
    canvas.paste(_c37, (54, 2024), _c37)
except Exception:
    pass
layout["Only_free_events"] = [54, 2024, 714, 2168]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_08_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-10/38_text_Sort_by.png
try:
    _c38 = get_crop(38, 206, 75)
    canvas.paste(_c38, (42, 1931), _c38)
except Exception:
    pass
layout["Sort_by"] = [42, 1931, 248, 2006]
