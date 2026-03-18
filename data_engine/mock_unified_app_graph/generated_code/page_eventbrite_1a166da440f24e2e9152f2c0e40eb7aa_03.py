# page_id: page_eventbrite_1a166da440f24e2e9152f2c0e40eb7aa_03
# screenshot: 2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-5.png
# step_index: 3/16
# task: Open Eventbrite. Check "Sports" category. Filter events happening next month. Add the first event to your wishlist.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw background and structural elements for the UI page
w, h = canvas.size

# Colors
bg_color = (255, 255, 255)            # page background (dominant white)
status_bar_color = (190, 190, 190)    # top status bar grey
divider_color = (230, 232, 238)       # subtle divider lines
section_shadow = (246, 247, 250)      # very light card/shadow background
header_text_bg = (255, 255, 255)      # header area (keeps white)
bottom_shadow = (242, 243, 246)       # bottom apply area shadow
outline_color = (210, 209, 218)       # light outline for bottom shadow

# Fill overall background (canvas already white but ensure consistent)
draw.rectangle([(0, 0), (w, h)], fill=bg_color)

# Status bar (approx ~0-84px)
status_h = 84
draw.rectangle([(0, 0), (w, status_h)], fill=status_bar_color)

# Header / toolbar area (around the "Filters" title)
header_top = status_h
header_bottom = 168
draw.rectangle([(0, header_top), (w, header_bottom)], fill=header_text_bg)
# subtle bottom divider under header
draw.line([(24, header_bottom), (w - 24, header_bottom)], fill=divider_color, width=2)

# Section separators to visually group filter sections
# After "Show all categories"
sep_y_1 = 564
draw.line([(24, sep_y_1), (w - 24, sep_y_1)], fill=divider_color, width=1)

# After "Show all event types"
sep_y_2 = 1020
draw.line([(24, sep_y_2), (w - 24, sep_y_2)], fill=divider_color, width=1)

# After "Show all languages"
sep_y_3 = 1472
draw.line([(24, sep_y_3), (w - 24, sep_y_3)], fill=divider_color, width=1)

# Divider above the Sort & Price area to separate lower controls
sep_y_4 = 1860
draw.line([(24, sep_y_4), (w - 24, sep_y_4)], fill=divider_color, width=1)

# Very faint background blocks behind larger content groups (subtle, non-icon)
# These are soft backgrounds for grouping; kept light to avoid duplicating chips/buttons.
group_padding_lr = 24
# Top group background (covers Categories + Show all)
grp1_top = 300
grp1_bottom = 600
draw.rounded_rectangle(
    [(group_padding_lr, grp1_top), (w - group_padding_lr, grp1_bottom)],
    radius=16,
    fill=section_shadow,
    outline=None
)
# Middle group background (Event type + Show all)
grp2_top = 740
grp2_bottom = 1020
draw.rounded_rectangle(
    [(group_padding_lr, grp2_top), (w - group_padding_lr, grp2_bottom)],
    radius=16,
    fill=section_shadow,
    outline=None
)
# Languages group background
grp3_top = 1120
grp3_bottom = 1470
draw.rounded_rectangle(
    [(group_padding_lr, grp3_top), (w - group_padding_lr, grp3_bottom)],
    radius=16,
    fill=section_shadow,
    outline=None
)

# Price / toggle area subtle background block
grp4_top = 1540
grp4_bottom = 2050
draw.rounded_rectangle(
    [(group_padding_lr, grp4_top), (w - group_padding_lr, grp4_bottom)],
    radius=14,
    fill=bg_color,
    outline=divider_color
)

# Large bottom area shadow behind the "Apply filters" control (do not draw the button itself)
apply_top = 2738  # slightly above the detected button top to create a shadow/background
apply_bottom = h - 12
apply_left = 36
apply_right = w - 36
draw.rounded_rectangle(
    [(apply_left, apply_top), (apply_right, apply_bottom)],
    radius=18,
    fill=bottom_shadow,
    outline=outline_color
)

# Add a faint top divider above the bottom area to separate content from the apply region
draw.line([(24, apply_top - 18), (w - 24, apply_top - 18)], fill=divider_color, width=1)

# micro accents: faint vertical guides on left/right margins (very subtle)
draw.line([(group_padding_lr, header_bottom + 8), (group_padding_lr, apply_top - 24)], fill=divider_color, width=1)
draw.line([(w - group_padding_lr, header_bottom + 8), (w - group_padding_lr, apply_top - 24)], fill=divider_color, width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_03_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-5/00_icon_Food_Drink.png
try:
    _c0 = get_crop(0, 312, 144)
    canvas.paste(_c0, (512, 383), _c0)
except Exception:
    pass
layout["Food_&_Drink"] = [512, 383, 824, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_03_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-5/01_icon_Music.png
try:
    _c1 = get_crop(1, 187, 135)
    canvas.paste(_c1, (36, 383), _c1)
except Exception:
    pass
layout["Music"] = [36, 383, 223, 518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_03_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-5/02_icon_Community.png
try:
    _c2 = get_crop(2, 294, 144)
    canvas.paste(_c2, (848, 383), _c2)
except Exception:
    pass
layout["Community"] = [848, 383, 1142, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_03_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-5/03_icon_French.png
try:
    _c3 = get_crop(3, 205, 144)
    canvas.paste(_c3, (768, 1275), _c3)
except Exception:
    pass
layout["French"] = [768, 1275, 973, 1419]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_03_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-5/04_icon_Business.png
try:
    _c4 = get_crop(4, 241, 135)
    canvas.paste(_c4, (247, 383), _c4)
except Exception:
    pass
layout["Business"] = [247, 383, 488, 518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_03_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-5/05_icon_Spanish.png
try:
    _c5 = get_crop(5, 225, 144)
    canvas.paste(_c5, (519, 1275), _c5)
except Exception:
    pass
layout["Spanish"] = [519, 1275, 744, 1419]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_03_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-5/06_icon_Expo.png
try:
    _c6 = get_crop(6, 167, 144)
    canvas.paste(_c6, (614, 829), _c6)
except Exception:
    pass
layout["Expo"] = [614, 829, 781, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_03_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-5/07_icon_Italian.png
try:
    _c7 = get_crop(7, 191, 144)
    canvas.paste(_c7, (997, 1275), _c7)
except Exception:
    pass
layout["Italian"] = [997, 1275, 1188, 1419]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_03_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-5/08_icon_Arts.png
try:
    _c8 = get_crop(8, 152, 144)
    canvas.paste(_c8, (1166, 383), _c8)
except Exception:
    pass
layout["Arts"] = [1166, 383, 1318, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_03_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-5/09_icon_Seminar.png
try:
    _c9 = get_crop(9, 232, 144)
    canvas.paste(_c9, (358, 829), _c9)
except Exception:
    pass
layout["Seminar"] = [358, 829, 590, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_03_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-5/10_icon_Convention.png
try:
    _c10 = get_crop(10, 293, 144)
    canvas.paste(_c10, (805, 829), _c10)
except Exception:
    pass
layout["Convention"] = [805, 829, 1098, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_03_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-5/11_icon_Festival.png
try:
    _c11 = get_crop(11, 219, 144)
    canvas.paste(_c11, (1122, 829), _c11)
except Exception:
    pass
layout["Festival"] = [1122, 829, 1341, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_03_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-5/12_icon_German.png
try:
    _c12 = get_crop(12, 225, 135)
    canvas.paste(_c12, (270, 1275), _c12)
except Exception:
    pass
layout["German"] = [270, 1275, 495, 1410]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_03_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-5/13_icon_English.png
try:
    _c13 = get_crop(13, 210, 135)
    canvas.paste(_c13, (36, 1275), _c13)
except Exception:
    pass
layout["English"] = [36, 1275, 246, 1410]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_03_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-5/14_icon_Conference.png
try:
    _c14 = get_crop(14, 298, 135)
    canvas.paste(_c14, (36, 829), _c14)
except Exception:
    pass
layout["Conference"] = [36, 829, 334, 964]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_03_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-5/15_icon_Date.png
try:
    _c15 = get_crop(15, 660, 144)
    canvas.paste(_c15, (726, 2024), _c15)
except Exception:
    pass
layout["Date"] = [726, 2024, 1386, 2168]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_03_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-5/16_icon_Relevance.png
try:
    _c16 = get_crop(16, 660, 144)
    canvas.paste(_c16, (54, 2024), _c16)
except Exception:
    pass
layout["Relevance"] = [54, 2024, 714, 2168]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_03_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-5/17_icon_Apply_filters.png
try:
    _c17 = get_crop(17, 1344, 144)
    canvas.paste(_c17, (48, 2768), _c17)
except Exception:
    pass
layout["Apply_filters"] = [48, 2768, 1392, 2912]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_03_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-5/18_icon_5.31.png
try:
    _c18 = get_crop(18, 144, 144)
    canvas.paste(_c18, (12, 72), _c18)
except Exception:
    pass
layout["5.31"] = [12, 72, 156, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_03_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-5/19_icon_5.31.png
try:
    _c19 = get_crop(19, 61, 65)
    canvas.paste(_c19, (179, 1), _c19)
except Exception:
    pass
layout["5.31"] = [179, 1, 240, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_03_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-5/20_icon_5.31.png
try:
    _c20 = get_crop(20, 66, 66)
    canvas.paste(_c20, (110, 1), _c20)
except Exception:
    pass
layout["5.31"] = [110, 1, 176, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_03_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-5/21_icon_icon_21.png
try:
    _c21 = get_crop(21, 65, 62)
    canvas.paste(_c21, (308, 3), _c21)
except Exception:
    pass
layout["icon_21"] = [308, 3, 373, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_03_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-5/22_icon_Clear_all.png
try:
    _c22 = get_crop(22, 55, 69)
    canvas.paste(_c22, (1319, 0), _c22)
except Exception:
    pass
layout["Clear_all"] = [1319, 0, 1374, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_03_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-5/23_icon_icon_23.png
try:
    _c23 = get_crop(23, 51, 63)
    canvas.paste(_c23, (248, 2), _c23)
except Exception:
    pass
layout["icon_23"] = [248, 2, 299, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_03_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-5/24_icon_Clear_all.png
try:
    _c24 = get_crop(24, 99, 70)
    canvas.paste(_c24, (1211, 0), _c24)
except Exception:
    pass
layout["Clear_all"] = [1211, 0, 1310, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_03_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-5/25_icon_clickable_20.png
try:
    _c25 = get_crop(25, 144, 144)
    canvas.paste(_c25, (1248, 1729), _c25)
except Exception:
    pass
layout["clickable_20"] = [1248, 1729, 1392, 1873]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_03_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-5/26_icon_Clear_all.png
try:
    _c26 = get_crop(26, 178, 144)
    canvas.paste(_c26, (1214, 72), _c26)
except Exception:
    pass
layout["Clear_all"] = [1214, 72, 1392, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_03_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-5/27_text_5.31.png
try:
    _c27 = get_crop(27, 87, 45)
    canvas.paste(_c27, (20, 15), _c27)
except Exception:
    pass
layout["5.31"] = [20, 15, 107, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_03_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-5/28_text_Filters.png
try:
    _c28 = get_crop(28, 180, 66)
    canvas.paste(_c28, (631, 116), _c28)
except Exception:
    pass
layout["Filters"] = [631, 116, 811, 182]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_03_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-5/29_text_Categories.png
try:
    _c29 = get_crop(29, 187, 135)
    canvas.paste(_c29, (36, 383), _c29)
except Exception:
    pass
layout["Categories"] = [36, 383, 223, 518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_03_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-5/30_text_Show_all_categories.png
try:
    _c30 = get_crop(30, 516, 144)
    canvas.paste(_c30, (0, 518), _c30)
except Exception:
    pass
layout["Show_all_categories"] = [0, 518, 516, 662]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_03_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-5/31_text_Event_type.png
try:
    _c31 = get_crop(31, 298, 135)
    canvas.paste(_c31, (36, 829), _c31)
except Exception:
    pass
layout["Event_type"] = [36, 829, 334, 964]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_03_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-5/32_text_Show_all_event_types.png
try:
    _c32 = get_crop(32, 535, 144)
    canvas.paste(_c32, (0, 964), _c32)
except Exception:
    pass
layout["Show_all_event_types"] = [0, 964, 535, 1108]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_03_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-5/33_text_Languages.png
try:
    _c33 = get_crop(33, 210, 135)
    canvas.paste(_c33, (36, 1275), _c33)
except Exception:
    pass
layout["Languages"] = [36, 1275, 246, 1410]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_03_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-5/34_text_Show_all_languages.png
try:
    _c34 = get_crop(34, 511, 144)
    canvas.paste(_c34, (0, 1410), _c34)
except Exception:
    pass
layout["Show_all_languages"] = [0, 1410, 511, 1554]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_03_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-5/35_text_Price.png
try:
    _c35 = get_crop(35, 149, 63)
    canvas.paste(_c35, (45, 1613), _c35)
except Exception:
    pass
layout["Price"] = [45, 1613, 194, 1676]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_03_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-5/36_text_Only_free_events.png
try:
    _c36 = get_crop(36, 660, 144)
    canvas.paste(_c36, (54, 2024), _c36)
except Exception:
    pass
layout["Only_free_events"] = [54, 2024, 714, 2168]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_03_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-5/37_text_Sort_by.png
try:
    _c37 = get_crop(37, 206, 75)
    canvas.paste(_c37, (42, 1931), _c37)
except Exception:
    pass
layout["Sort_by"] = [42, 1931, 248, 2006]
