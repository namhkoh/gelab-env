# page_id: page_eventbrite_bbe931c7fbf3446c80521f77d3e413aa_03
# screenshot: 2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-5.png
# step_index: 3/20
# task: Open Eventbrite. Search free events in Los Angeles. Select the first one. Follow the organizer. Read more about the event. Add it to Favorites.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw UI background, status bar, headers, section separators, segmented control, and bottom background.
w, h = canvas.size

# Colors
bg_color = (255, 255, 255)
status_bar_color = (238, 238, 238)        # light grey for status bar
header_divider = (232, 229, 238)          # subtle divider under header
section_sep = (245, 244, 247)             # very faint separators between sections
segmented_bg = (242, 240, 245)            # segmented control background
segment_border = (210, 205, 216)          # border for segmented control
selected_segment = (255, 255, 255)        # selected segment (white)
unselected_segment = (236, 234, 239)      # unselected/pressed segment shade
bottom_panel_bg = (250, 249, 252)         # faint bottom panel background / shadow

# Clear canvas (ensure base color)
draw.rectangle((0, 0, w, h), fill=bg_color)

# Status bar (top area)
status_h = 84
draw.rectangle((0, 0, w, status_h), fill=status_bar_color)

# Header / toolbar area (below status bar)
header_top = status_h
header_h = 84
header_bottom = header_top + header_h
draw.rectangle((0, header_top, w, header_bottom), fill=bg_color)
# Divider under header
draw.rectangle((0, header_bottom - 2, w, header_bottom), fill=header_divider)

# Section separators (faint horizontal rules between groups)
# Use positions roughly aligned with detected groups (categories, event types, languages, price, sort)
separators_y = [
    560,   # after categories / show all categories area
    915,   # after event types
    1350,  # after languages
    1700,  # after price / before sort area
    2320,  # above bottom / above apply filters area
]
for y in separators_y:
    draw.rectangle((36, y - 1, w - 36, y + 1), fill=section_sep)

# Segmented control for "Sort by" (two segments: Relevance / Date)
# Place it centered around detected sort area (~y=2024)
seg_left = 48
seg_right = w - 48
seg_top = 1978
seg_bottom = 2122
seg_radius = 18
# Outer container
draw.rounded_rectangle((seg_left, seg_top, seg_right, seg_bottom), radius=seg_radius, fill=segmented_bg, outline=segment_border, width=1)
# Left selected segment (slightly inset)
inner_pad = 8
left_seg_right = (seg_left + seg_right) // 2 - 6
draw.rounded_rectangle((seg_left + inner_pad, seg_top + inner_pad, left_seg_right, seg_bottom - inner_pad),
                       radius=14, fill=selected_segment, outline=segment_border, width=1)
# Right unselected segment (no rounded corners on inner left to blend)
right_seg_left = left_seg_right
draw.rectangle((right_seg_left, seg_top + inner_pad, seg_right - inner_pad, seg_bottom - inner_pad), fill=unselected_segment)
# subtle inner dividing line between segments
div_x = left_seg_right + 2
draw.rectangle((div_x - 1, seg_top + inner_pad + 6, div_x + 1, seg_bottom - inner_pad - 6), fill=segment_border)

# Add subtle shadow under segmented control (soft line)
shadow_y = seg_bottom + 8
draw.rectangle((seg_left + 4, shadow_y, seg_right - 4, shadow_y + 3), fill=(235, 233, 239))

# Bottom area background / shadow behind the "Apply filters" region
# Keep this subtle and slightly inset so the actual button pasted on top is not duplicated
bottom_panel_top = 2680
bottom_panel_radius = 14
draw.rounded_rectangle((24, bottom_panel_top, w - 24, h - 8), radius=bottom_panel_radius, fill=bottom_panel_bg, outline=(230, 227, 235), width=1)

# Small separation line above the bottom panel to ground it
draw.rectangle((24, bottom_panel_top - 2, w - 24, bottom_panel_top), fill=section_sep)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_03_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-5/00_icon_Food_Drink.png
try:
    _c0 = get_crop(0, 312, 144)
    canvas.paste(_c0, (512, 383), _c0)
except Exception:
    pass
layout["Food_&_Drink"] = [512, 383, 824, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_03_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-5/01_icon_Music.png
try:
    _c1 = get_crop(1, 187, 135)
    canvas.paste(_c1, (36, 383), _c1)
except Exception:
    pass
layout["Music"] = [36, 383, 223, 518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_03_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-5/02_icon_Community.png
try:
    _c2 = get_crop(2, 294, 144)
    canvas.paste(_c2, (848, 383), _c2)
except Exception:
    pass
layout["Community"] = [848, 383, 1142, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_03_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-5/03_icon_French.png
try:
    _c3 = get_crop(3, 205, 144)
    canvas.paste(_c3, (768, 1275), _c3)
except Exception:
    pass
layout["French"] = [768, 1275, 973, 1419]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_03_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-5/04_icon_Business.png
try:
    _c4 = get_crop(4, 241, 135)
    canvas.paste(_c4, (247, 383), _c4)
except Exception:
    pass
layout["Business"] = [247, 383, 488, 518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_03_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-5/05_icon_Spanish.png
try:
    _c5 = get_crop(5, 225, 144)
    canvas.paste(_c5, (519, 1275), _c5)
except Exception:
    pass
layout["Spanish"] = [519, 1275, 744, 1419]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_03_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-5/06_icon_Expo.png
try:
    _c6 = get_crop(6, 167, 144)
    canvas.paste(_c6, (614, 829), _c6)
except Exception:
    pass
layout["Expo"] = [614, 829, 781, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_03_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-5/07_icon_Seminar.png
try:
    _c7 = get_crop(7, 232, 144)
    canvas.paste(_c7, (358, 829), _c7)
except Exception:
    pass
layout["Seminar"] = [358, 829, 590, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_03_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-5/08_icon_Italian.png
try:
    _c8 = get_crop(8, 191, 144)
    canvas.paste(_c8, (997, 1275), _c8)
except Exception:
    pass
layout["Italian"] = [997, 1275, 1188, 1419]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_03_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-5/09_icon_Arts.png
try:
    _c9 = get_crop(9, 152, 144)
    canvas.paste(_c9, (1166, 383), _c9)
except Exception:
    pass
layout["Arts"] = [1166, 383, 1318, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_03_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-5/10_icon_Convention.png
try:
    _c10 = get_crop(10, 293, 144)
    canvas.paste(_c10, (805, 829), _c10)
except Exception:
    pass
layout["Convention"] = [805, 829, 1098, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_03_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-5/11_icon_German.png
try:
    _c11 = get_crop(11, 225, 135)
    canvas.paste(_c11, (270, 1275), _c11)
except Exception:
    pass
layout["German"] = [270, 1275, 495, 1410]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_03_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-5/12_icon_Festival.png
try:
    _c12 = get_crop(12, 219, 144)
    canvas.paste(_c12, (1122, 829), _c12)
except Exception:
    pass
layout["Festival"] = [1122, 829, 1341, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_03_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-5/13_icon_English.png
try:
    _c13 = get_crop(13, 210, 135)
    canvas.paste(_c13, (36, 1275), _c13)
except Exception:
    pass
layout["English"] = [36, 1275, 246, 1410]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_03_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-5/14_icon_Conference.png
try:
    _c14 = get_crop(14, 298, 135)
    canvas.paste(_c14, (36, 829), _c14)
except Exception:
    pass
layout["Conference"] = [36, 829, 334, 964]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_03_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-5/15_icon_Date.png
try:
    _c15 = get_crop(15, 660, 144)
    canvas.paste(_c15, (726, 2024), _c15)
except Exception:
    pass
layout["Date"] = [726, 2024, 1386, 2168]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_03_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-5/16_icon_Relevance.png
try:
    _c16 = get_crop(16, 660, 144)
    canvas.paste(_c16, (54, 2024), _c16)
except Exception:
    pass
layout["Relevance"] = [54, 2024, 714, 2168]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_03_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-5/17_icon_Apply_filters.png
try:
    _c17 = get_crop(17, 1344, 144)
    canvas.paste(_c17, (48, 2768), _c17)
except Exception:
    pass
layout["Apply_filters"] = [48, 2768, 1392, 2912]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_03_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-5/18_icon_9.11.png
try:
    _c18 = get_crop(18, 144, 144)
    canvas.paste(_c18, (12, 72), _c18)
except Exception:
    pass
layout["9.11"] = [12, 72, 156, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_03_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-5/19_icon_9.11.png
try:
    _c19 = get_crop(19, 64, 64)
    canvas.paste(_c19, (176, 1), _c19)
except Exception:
    pass
layout["9.11"] = [176, 1, 240, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_03_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-5/20_icon_9.11.png
try:
    _c20 = get_crop(20, 59, 65)
    canvas.paste(_c20, (112, 1), _c20)
except Exception:
    pass
layout["9.11"] = [112, 1, 171, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_03_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-5/21_icon_Clear_all.png
try:
    _c21 = get_crop(21, 55, 69)
    canvas.paste(_c21, (1319, 0), _c21)
except Exception:
    pass
layout["Clear_all"] = [1319, 0, 1374, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_03_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-5/22_icon_Clear_all.png
try:
    _c22 = get_crop(22, 99, 69)
    canvas.paste(_c22, (1211, 0), _c22)
except Exception:
    pass
layout["Clear_all"] = [1211, 0, 1310, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_03_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-5/23_icon_icon_23.png
try:
    _c23 = get_crop(23, 59, 63)
    canvas.paste(_c23, (245, 2), _c23)
except Exception:
    pass
layout["icon_23"] = [245, 2, 304, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_03_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-5/24_icon_icon_24.png
try:
    _c24 = get_crop(24, 54, 61)
    canvas.paste(_c24, (314, 3), _c24)
except Exception:
    pass
layout["icon_24"] = [314, 3, 368, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_03_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-5/25_icon_clickable_20.png
try:
    _c25 = get_crop(25, 144, 144)
    canvas.paste(_c25, (1248, 1729), _c25)
except Exception:
    pass
layout["clickable_20"] = [1248, 1729, 1392, 1873]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_03_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-5/26_icon_Clear_all.png
try:
    _c26 = get_crop(26, 178, 144)
    canvas.paste(_c26, (1214, 72), _c26)
except Exception:
    pass
layout["Clear_all"] = [1214, 72, 1392, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_03_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-5/27_text_9.11.png
try:
    _c27 = get_crop(27, 89, 43)
    canvas.paste(_c27, (20, 17), _c27)
except Exception:
    pass
layout["9.11"] = [20, 17, 109, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_03_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-5/28_text_Filters.png
try:
    _c28 = get_crop(28, 180, 66)
    canvas.paste(_c28, (631, 116), _c28)
except Exception:
    pass
layout["Filters"] = [631, 116, 811, 182]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_03_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-5/29_text_Categories.png
try:
    _c29 = get_crop(29, 187, 135)
    canvas.paste(_c29, (36, 383), _c29)
except Exception:
    pass
layout["Categories"] = [36, 383, 223, 518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_03_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-5/30_text_Show_all_categories.png
try:
    _c30 = get_crop(30, 516, 144)
    canvas.paste(_c30, (0, 518), _c30)
except Exception:
    pass
layout["Show_all_categories"] = [0, 518, 516, 662]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_03_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-5/31_text_Event_type.png
try:
    _c31 = get_crop(31, 298, 135)
    canvas.paste(_c31, (36, 829), _c31)
except Exception:
    pass
layout["Event_type"] = [36, 829, 334, 964]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_03_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-5/32_text_Show_all_event_types.png
try:
    _c32 = get_crop(32, 535, 144)
    canvas.paste(_c32, (0, 964), _c32)
except Exception:
    pass
layout["Show_all_event_types"] = [0, 964, 535, 1108]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_03_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-5/33_text_Languages.png
try:
    _c33 = get_crop(33, 210, 135)
    canvas.paste(_c33, (36, 1275), _c33)
except Exception:
    pass
layout["Languages"] = [36, 1275, 246, 1410]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_03_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-5/34_text_Show_all_languages.png
try:
    _c34 = get_crop(34, 511, 144)
    canvas.paste(_c34, (0, 1410), _c34)
except Exception:
    pass
layout["Show_all_languages"] = [0, 1410, 511, 1554]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_03_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-5/35_text_Price.png
try:
    _c35 = get_crop(35, 149, 63)
    canvas.paste(_c35, (45, 1613), _c35)
except Exception:
    pass
layout["Price"] = [45, 1613, 194, 1676]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_03_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-5/36_text_Only_free_events.png
try:
    _c36 = get_crop(36, 660, 144)
    canvas.paste(_c36, (54, 2024), _c36)
except Exception:
    pass
layout["Only_free_events"] = [54, 2024, 714, 2168]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_03_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-5/37_text_Sort_by.png
try:
    _c37 = get_crop(37, 206, 75)
    canvas.paste(_c37, (42, 1931), _c37)
except Exception:
    pass
layout["Sort_by"] = [42, 1931, 248, 2006]
