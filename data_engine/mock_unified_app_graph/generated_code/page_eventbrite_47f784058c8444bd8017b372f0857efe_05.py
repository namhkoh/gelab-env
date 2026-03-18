# page_id: page_eventbrite_47f784058c8444bd8017b372f0857efe_05
# screenshot: 2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-7.png
# step_index: 5/11
# task: Open Eventbrite. Explore local events scheduled for this weekend. Select the first event from the 'Science' category. Read details of the event.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# draw background and UI structure for filters page

# Full canvas is provided as `canvas` (1440x2960) and `draw` (ImageDraw)
w, h = canvas.size

# Colors
bg_color = (255, 255, 255)                 # main white background
status_bar_color = (227, 227, 227)         # light grey status bar
header_divider = (230, 226, 235)           # subtle divider/pale purple-grey
section_divider = (236, 234, 240)          # very light divider
muted_panel = (250, 250, 252)              # slightly off-white panels
segmented_bg = (244, 243, 247)             # segmented control background
segmented_selected = (241, 238, 244)       # selected segment (slightly darker)
bottom_button_fill = (255, 255, 255)       # apply filters button fill (white)
button_border = (190, 186, 197)            # border color for bottom button
shadow_color = (220, 217, 225, 90)         # translucent shadow (RGBA)

# Clear/fill background
draw.rectangle([(0,0),(w,h)], fill=bg_color)

# Status bar area (top)
status_h = 88
draw.rectangle([(0,0),(w,status_h)], fill=status_bar_color)

# Header / toolbar area (below status bar)
header_top = status_h
header_bottom = 180
draw.rectangle([(0, header_top), (w, header_bottom)], fill=bg_color)

# Thin divider under header
draw.line([(24, header_bottom), (w-24, header_bottom)], fill=header_divider, width=1)

# Section separators (approx positions based on layout)
# After "Categories" block
sep1_y = 520
draw.line([(24, sep1_y), (w-24, sep1_y)], fill=section_divider, width=1)

# After "Event type" block
sep2_y = 968
draw.line([(24, sep2_y), (w-24, sep2_y)], fill=section_divider, width=1)

# After "Languages" block
sep3_y = 1416
draw.line([(24, sep3_y), (w-24, sep3_y)], fill=section_divider, width=1)

# Light panel behind "Sort by" control area (subtle rounded panel)
sort_panel_top = 1900
sort_panel_bottom = 2096
panel_margin = 36
draw.rounded_rectangle(
    [(panel_margin, sort_panel_top), (w - panel_margin, sort_panel_bottom)],
    radius=14,
    fill=muted_panel,
    outline=section_divider,
    width=1
)

# Segmented control background inside the sort panel
seg_pad = 14
seg_x1 = panel_margin + seg_pad
seg_x2 = w - panel_margin - seg_pad
seg_y1 = sort_panel_top + 12
seg_y2 = sort_panel_bottom - 12
draw.rounded_rectangle(
    [(seg_x1, seg_y1), (seg_x2, seg_y2)],
    radius=12,
    fill=segmented_bg,
    outline=(220,216,226),
    width=1
)
# Left segment selected background (left half)
mid_x = (seg_x1 + seg_x2) // 2
draw.rounded_rectangle(
    [(seg_x1+2, seg_y1+2), (mid_x-2, seg_y2-2)],
    radius=10,
    fill=segmented_selected,
    outline=None
)

# Price / toggle area separator hint (a faint line to group controls)
price_sep_y = 1768
draw.line([(36, price_sep_y), (w-36, price_sep_y)], fill=section_divider, width=1)

# Subtle horizontal guides between sections for visual grouping
draw.line([(36, 1600), (w-36, 1600)], fill=section_divider, width=1)
draw.line([(36, 1840), (w-36, 1840)], fill=section_divider, width=1)

# Bottom Apply filters button area (rounded rect with border and subtle shadow)
btn_left = 48
btn_top = 2768
btn_width = 1344
btn_height = 144
btn_right = btn_left + btn_width
btn_bottom = btn_top + btn_height

# Draw drop shadow as a faint rectangle slightly below the button
shadow_offset = 8
# Use multiple translucent rectangles to simulate soft shadow if RGBA supported on canvas
try:
    # paste a small translucent shadow directly if canvas supports alpha by drawing onto it
    shadow = canvas.copy().convert("RGBA")
    sdraw = __import__("PIL").ImageDraw.Draw(shadow)
    sdraw.rounded_rectangle(
        [(btn_left, btn_top+shadow_offset), (btn_right, btn_bottom+shadow_offset)],
        radius=10,
        fill=(220,217,225,60)
    )
    canvas.paste(shadow, (0,0), shadow)
except Exception:
    # fallback: simple darker line under button
    draw.rectangle([(btn_left, btn_bottom+2), (btn_right, btn_bottom+6)], fill=(230,228,235))

# Button fill and border
draw.rounded_rectangle(
    [(btn_left, btn_top), (btn_right, btn_bottom)],
    radius=10,
    fill=bottom_button_fill,
    outline=button_border,
    width=3
)

# Header left back-arrow area: subtle touch target background (no icon)
back_touch_radius = 8
back_box = (24, header_top+16, 24+64, header_top+16+64)
# very light circular area for back arrow touch target background
draw.ellipse([(back_box[0], back_box[1]), (back_box[2], back_box[3])], fill=muted_panel)

# Header right clear all touch target background (no text/icon drawn)
right_box = (w - 24 - 64, header_top+16, w - 24, header_top+16+64)
draw.ellipse([(right_box[0], right_box[1]), (right_box[2], right_box[3])], fill=muted_panel)

# Large content area subtle vertical spacing lines to suggest sections (non-distracting)
for y in (260, 820, 1320, 1640, 1960):
    draw.line([(36, y), (w-36, y)], fill=(248,247,249), width=1)

# Finished structural drawing

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_05_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-7/00_icon_Food_Drink.png
try:
    _c0 = get_crop(0, 312, 144)
    canvas.paste(_c0, (512, 383), _c0)
except Exception:
    pass
layout["Food_&_Drink"] = [512, 383, 824, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_05_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-7/01_icon_Music.png
try:
    _c1 = get_crop(1, 187, 135)
    canvas.paste(_c1, (36, 383), _c1)
except Exception:
    pass
layout["Music"] = [36, 383, 223, 518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_05_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-7/02_icon_Community.png
try:
    _c2 = get_crop(2, 294, 144)
    canvas.paste(_c2, (848, 383), _c2)
except Exception:
    pass
layout["Community"] = [848, 383, 1142, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_05_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-7/03_icon_French.png
try:
    _c3 = get_crop(3, 205, 144)
    canvas.paste(_c3, (768, 1275), _c3)
except Exception:
    pass
layout["French"] = [768, 1275, 973, 1419]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_05_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-7/04_icon_Business.png
try:
    _c4 = get_crop(4, 241, 135)
    canvas.paste(_c4, (247, 383), _c4)
except Exception:
    pass
layout["Business"] = [247, 383, 488, 518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_05_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-7/05_icon_Spanish.png
try:
    _c5 = get_crop(5, 225, 144)
    canvas.paste(_c5, (519, 1275), _c5)
except Exception:
    pass
layout["Spanish"] = [519, 1275, 744, 1419]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_05_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-7/06_icon_Expo.png
try:
    _c6 = get_crop(6, 167, 144)
    canvas.paste(_c6, (614, 829), _c6)
except Exception:
    pass
layout["Expo"] = [614, 829, 781, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_05_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-7/07_icon_Italian.png
try:
    _c7 = get_crop(7, 191, 144)
    canvas.paste(_c7, (997, 1275), _c7)
except Exception:
    pass
layout["Italian"] = [997, 1275, 1188, 1419]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_05_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-7/08_icon_Seminar.png
try:
    _c8 = get_crop(8, 232, 144)
    canvas.paste(_c8, (358, 829), _c8)
except Exception:
    pass
layout["Seminar"] = [358, 829, 590, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_05_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-7/09_icon_Arts.png
try:
    _c9 = get_crop(9, 152, 144)
    canvas.paste(_c9, (1166, 383), _c9)
except Exception:
    pass
layout["Arts"] = [1166, 383, 1318, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_05_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-7/10_icon_Convention.png
try:
    _c10 = get_crop(10, 293, 144)
    canvas.paste(_c10, (805, 829), _c10)
except Exception:
    pass
layout["Convention"] = [805, 829, 1098, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_05_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-7/11_icon_Festival.png
try:
    _c11 = get_crop(11, 219, 144)
    canvas.paste(_c11, (1122, 829), _c11)
except Exception:
    pass
layout["Festival"] = [1122, 829, 1341, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_05_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-7/12_icon_German.png
try:
    _c12 = get_crop(12, 225, 135)
    canvas.paste(_c12, (270, 1275), _c12)
except Exception:
    pass
layout["German"] = [270, 1275, 495, 1410]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_05_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-7/13_icon_English.png
try:
    _c13 = get_crop(13, 210, 135)
    canvas.paste(_c13, (36, 1275), _c13)
except Exception:
    pass
layout["English"] = [36, 1275, 246, 1410]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_05_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-7/14_icon_Conference.png
try:
    _c14 = get_crop(14, 298, 135)
    canvas.paste(_c14, (36, 829), _c14)
except Exception:
    pass
layout["Conference"] = [36, 829, 334, 964]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_05_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-7/15_icon_Date.png
try:
    _c15 = get_crop(15, 660, 144)
    canvas.paste(_c15, (726, 2024), _c15)
except Exception:
    pass
layout["Date"] = [726, 2024, 1386, 2168]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_05_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-7/16_icon_Relevance.png
try:
    _c16 = get_crop(16, 660, 144)
    canvas.paste(_c16, (54, 2024), _c16)
except Exception:
    pass
layout["Relevance"] = [54, 2024, 714, 2168]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_05_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-7/17_icon_Apply_filters.png
try:
    _c17 = get_crop(17, 1344, 144)
    canvas.paste(_c17, (48, 2768), _c17)
except Exception:
    pass
layout["Apply_filters"] = [48, 2768, 1392, 2912]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_05_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-7/18_icon_7.58.png
try:
    _c18 = get_crop(18, 61, 63)
    canvas.paste(_c18, (179, 2), _c18)
except Exception:
    pass
layout["7.58"] = [179, 2, 240, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_05_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-7/19_icon_7.58.png
try:
    _c19 = get_crop(19, 144, 144)
    canvas.paste(_c19, (12, 72), _c19)
except Exception:
    pass
layout["7.58"] = [12, 72, 156, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_05_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-7/20_icon_7.58.png
try:
    _c20 = get_crop(20, 65, 65)
    canvas.paste(_c20, (111, 1), _c20)
except Exception:
    pass
layout["7.58"] = [111, 1, 176, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_05_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-7/21_icon_icon_21.png
try:
    _c21 = get_crop(21, 66, 62)
    canvas.paste(_c21, (307, 3), _c21)
except Exception:
    pass
layout["icon_21"] = [307, 3, 373, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_05_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-7/22_icon_Clear_all.png
try:
    _c22 = get_crop(22, 55, 69)
    canvas.paste(_c22, (1319, 0), _c22)
except Exception:
    pass
layout["Clear_all"] = [1319, 0, 1374, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_05_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-7/23_icon_icon_23.png
try:
    _c23 = get_crop(23, 51, 62)
    canvas.paste(_c23, (248, 2), _c23)
except Exception:
    pass
layout["icon_23"] = [248, 2, 299, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_05_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-7/24_icon_Clear_all.png
try:
    _c24 = get_crop(24, 99, 70)
    canvas.paste(_c24, (1211, 0), _c24)
except Exception:
    pass
layout["Clear_all"] = [1211, 0, 1310, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_05_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-7/25_icon_clickable_20.png
try:
    _c25 = get_crop(25, 144, 144)
    canvas.paste(_c25, (1248, 1729), _c25)
except Exception:
    pass
layout["clickable_20"] = [1248, 1729, 1392, 1873]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_05_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-7/26_icon_Clear_all.png
try:
    _c26 = get_crop(26, 178, 144)
    canvas.paste(_c26, (1214, 72), _c26)
except Exception:
    pass
layout["Clear_all"] = [1214, 72, 1392, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_05_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-7/27_text_7.58.png
try:
    _c27 = get_crop(27, 91, 45)
    canvas.paste(_c27, (20, 15), _c27)
except Exception:
    pass
layout["7.58"] = [20, 15, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_05_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-7/28_text_Filters.png
try:
    _c28 = get_crop(28, 180, 66)
    canvas.paste(_c28, (631, 116), _c28)
except Exception:
    pass
layout["Filters"] = [631, 116, 811, 182]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_05_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-7/29_text_Categories.png
try:
    _c29 = get_crop(29, 187, 135)
    canvas.paste(_c29, (36, 383), _c29)
except Exception:
    pass
layout["Categories"] = [36, 383, 223, 518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_05_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-7/30_text_Show_all_categories.png
try:
    _c30 = get_crop(30, 516, 144)
    canvas.paste(_c30, (0, 518), _c30)
except Exception:
    pass
layout["Show_all_categories"] = [0, 518, 516, 662]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_05_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-7/31_text_Event_type.png
try:
    _c31 = get_crop(31, 298, 135)
    canvas.paste(_c31, (36, 829), _c31)
except Exception:
    pass
layout["Event_type"] = [36, 829, 334, 964]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_05_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-7/32_text_Show_all_event_types.png
try:
    _c32 = get_crop(32, 535, 144)
    canvas.paste(_c32, (0, 964), _c32)
except Exception:
    pass
layout["Show_all_event_types"] = [0, 964, 535, 1108]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_05_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-7/33_text_Languages.png
try:
    _c33 = get_crop(33, 210, 135)
    canvas.paste(_c33, (36, 1275), _c33)
except Exception:
    pass
layout["Languages"] = [36, 1275, 246, 1410]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_05_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-7/34_text_Show_all_languages.png
try:
    _c34 = get_crop(34, 511, 144)
    canvas.paste(_c34, (0, 1410), _c34)
except Exception:
    pass
layout["Show_all_languages"] = [0, 1410, 511, 1554]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_05_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-7/35_text_Price.png
try:
    _c35 = get_crop(35, 149, 63)
    canvas.paste(_c35, (45, 1613), _c35)
except Exception:
    pass
layout["Price"] = [45, 1613, 194, 1676]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_05_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-7/36_text_Only_free_events.png
try:
    _c36 = get_crop(36, 660, 144)
    canvas.paste(_c36, (54, 2024), _c36)
except Exception:
    pass
layout["Only_free_events"] = [54, 2024, 714, 2168]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_05_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-7/37_text_Sort_by.png
try:
    _c37 = get_crop(37, 206, 75)
    canvas.paste(_c37, (42, 1931), _c37)
except Exception:
    pass
layout["Sort_by"] = [42, 1931, 248, 2006]
