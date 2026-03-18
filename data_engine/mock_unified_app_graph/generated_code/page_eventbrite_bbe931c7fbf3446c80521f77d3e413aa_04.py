# page_id: page_eventbrite_bbe931c7fbf3446c80521f77d3e413aa_04
# screenshot: 2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-6.png
# step_index: 4/20
# task: Open Eventbrite. Search free events in Los Angeles. Select the first one. Follow the organizer. Read more about the event. Add it to Favorites.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Paint overall background
draw.rectangle((0, 0, 1440, 2960), fill="#FFFFFF")

# Status bar (top area)
status_h = 72
draw.rectangle((0, 0, 1440, status_h), fill="#E6E6E6")
# subtle bottom shadow of status bar
draw.rectangle((0, status_h - 1, 1440, status_h + 2), fill="#D9D9DB")

# Header / toolbar area (below status bar)
header_top = status_h
header_bottom = 152
draw.rectangle((0, header_top, 1440, header_bottom), fill="#FFFFFF")
# header bottom divider
draw.rectangle((0, header_bottom - 1, 1440, header_bottom + 1), fill="#ECECF1")

# Section separators (light thin lines between groups)
sep_color = "#F0F0F3"
separator_ys = [608, 1054, 1500, 1698]  # placed between major groups
for y in separator_ys:
    draw.rectangle((36, y, 1440 - 36, y + 1), fill=sep_color)

# Sort-by segmented control background (rounded outer container)
seg_x = 36
seg_y = 2024
seg_w = 1368
seg_h = 144
seg_box = (seg_x, seg_y, seg_x + seg_w, seg_y + seg_h)
draw.rounded_rectangle(seg_box, radius=18, fill="#F6F6F8", outline="#D6D4DA", width=2)

# Light inner top highlight on segmented control
draw.rectangle((seg_x + 2, seg_y + 2, seg_x + seg_w - 2, seg_y + 6), fill="#FFFFFF")

# Divider between the two segments (subtle)
mid_x = seg_x + seg_w // 2
draw.line((mid_x, seg_y + 6, mid_x, seg_y + seg_h - 6), fill="#E6E4E9", width=1)

# Bottom "Apply filters" floating button background and border (rounded)
apply_x = 48
apply_y = 2768
apply_w = 1344
apply_h = 144
# shadow
shadow_box = (apply_x + 6, apply_y + 6, apply_x + apply_w + 6, apply_y + apply_h + 6)
draw.rounded_rectangle(shadow_box, radius=14, fill="#EDEAF0")
# button background + border
apply_box = (apply_x, apply_y, apply_x + apply_w, apply_y + apply_h)
draw.rounded_rectangle(apply_box, radius=14, fill="#FFFFFF", outline="#CFCBD4", width=4)

# Additional subtle separators and background accents for large empty content area
# faint horizontal divider under header content region
draw.rectangle((36, 220, 1440 - 36, 222), fill="#FBFBFD")

# Light large area tint behind chip groups (very subtle, stays behind chips)
# Categories block background (subtle rectangular background pad)
draw.rectangle((24, 320, 1440 - 24, 660), fill="#FFFFFF")  # keep white but provide padding area
# Event type block
draw.rectangle((24, 760, 1440 - 24, 1100), fill="#FFFFFF")
# Languages block
draw.rectangle((24, 1196, 1440 - 24, 1528), fill="#FFFFFF")

# small bottom page shadow to separate big content area from bottom button
draw.rectangle((0, 2926, 1440, 2960), fill="#FFFFFF")

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_04_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-6/00_icon_Food_Drink.png
try:
    _c0 = get_crop(0, 312, 144)
    canvas.paste(_c0, (512, 383), _c0)
except Exception:
    pass
layout["Food_&_Drink"] = [512, 383, 824, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_04_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-6/01_icon_Music.png
try:
    _c1 = get_crop(1, 187, 135)
    canvas.paste(_c1, (36, 383), _c1)
except Exception:
    pass
layout["Music"] = [36, 383, 223, 518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_04_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-6/02_icon_Community.png
try:
    _c2 = get_crop(2, 294, 144)
    canvas.paste(_c2, (848, 383), _c2)
except Exception:
    pass
layout["Community"] = [848, 383, 1142, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_04_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-6/03_icon_French.png
try:
    _c3 = get_crop(3, 205, 144)
    canvas.paste(_c3, (768, 1275), _c3)
except Exception:
    pass
layout["French"] = [768, 1275, 973, 1419]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_04_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-6/04_icon_Spanish.png
try:
    _c4 = get_crop(4, 225, 144)
    canvas.paste(_c4, (519, 1275), _c4)
except Exception:
    pass
layout["Spanish"] = [519, 1275, 744, 1419]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_04_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-6/05_icon_Business.png
try:
    _c5 = get_crop(5, 241, 135)
    canvas.paste(_c5, (247, 383), _c5)
except Exception:
    pass
layout["Business"] = [247, 383, 488, 518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_04_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-6/06_icon_Expo.png
try:
    _c6 = get_crop(6, 167, 144)
    canvas.paste(_c6, (614, 829), _c6)
except Exception:
    pass
layout["Expo"] = [614, 829, 781, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_04_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-6/07_icon_Seminar.png
try:
    _c7 = get_crop(7, 232, 144)
    canvas.paste(_c7, (358, 829), _c7)
except Exception:
    pass
layout["Seminar"] = [358, 829, 590, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_04_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-6/08_icon_Italian.png
try:
    _c8 = get_crop(8, 191, 144)
    canvas.paste(_c8, (997, 1275), _c8)
except Exception:
    pass
layout["Italian"] = [997, 1275, 1188, 1419]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_04_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-6/09_icon_Arts.png
try:
    _c9 = get_crop(9, 152, 144)
    canvas.paste(_c9, (1166, 383), _c9)
except Exception:
    pass
layout["Arts"] = [1166, 383, 1318, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_04_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-6/10_icon_Convention.png
try:
    _c10 = get_crop(10, 293, 144)
    canvas.paste(_c10, (805, 829), _c10)
except Exception:
    pass
layout["Convention"] = [805, 829, 1098, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_04_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-6/11_icon_Festival.png
try:
    _c11 = get_crop(11, 219, 144)
    canvas.paste(_c11, (1122, 829), _c11)
except Exception:
    pass
layout["Festival"] = [1122, 829, 1341, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_04_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-6/12_icon_German.png
try:
    _c12 = get_crop(12, 225, 135)
    canvas.paste(_c12, (270, 1275), _c12)
except Exception:
    pass
layout["German"] = [270, 1275, 495, 1410]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_04_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-6/13_icon_English.png
try:
    _c13 = get_crop(13, 210, 135)
    canvas.paste(_c13, (36, 1275), _c13)
except Exception:
    pass
layout["English"] = [36, 1275, 246, 1410]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_04_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-6/14_icon_Conference.png
try:
    _c14 = get_crop(14, 298, 135)
    canvas.paste(_c14, (36, 829), _c14)
except Exception:
    pass
layout["Conference"] = [36, 829, 334, 964]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_04_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-6/15_icon_Apply_filters_1.png
try:
    _c15 = get_crop(15, 1344, 144)
    canvas.paste(_c15, (48, 2768), _c15)
except Exception:
    pass
layout["Apply_filters_(1)"] = [48, 2768, 1392, 2912]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_04_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-6/16_icon_Date.png
try:
    _c16 = get_crop(16, 660, 144)
    canvas.paste(_c16, (726, 2024), _c16)
except Exception:
    pass
layout["Date"] = [726, 2024, 1386, 2168]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_04_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-6/17_icon_Relevance.png
try:
    _c17 = get_crop(17, 660, 144)
    canvas.paste(_c17, (54, 2024), _c17)
except Exception:
    pass
layout["Relevance"] = [54, 2024, 714, 2168]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_04_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-6/18_icon_9.11.png
try:
    _c18 = get_crop(18, 144, 144)
    canvas.paste(_c18, (12, 72), _c18)
except Exception:
    pass
layout["9.11"] = [12, 72, 156, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_04_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-6/19_icon_9.11.png
try:
    _c19 = get_crop(19, 64, 63)
    canvas.paste(_c19, (176, 2), _c19)
except Exception:
    pass
layout["9.11"] = [176, 2, 240, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_04_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-6/20_icon_Clear_all.png
try:
    _c20 = get_crop(20, 99, 65)
    canvas.paste(_c20, (1211, 0), _c20)
except Exception:
    pass
layout["Clear_all"] = [1211, 0, 1310, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_04_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-6/21_icon_9.11.png
try:
    _c21 = get_crop(21, 58, 65)
    canvas.paste(_c21, (112, 1), _c21)
except Exception:
    pass
layout["9.11"] = [112, 1, 170, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_04_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-6/22_icon_Clear_all.png
try:
    _c22 = get_crop(22, 56, 67)
    canvas.paste(_c22, (1317, 0), _c22)
except Exception:
    pass
layout["Clear_all"] = [1317, 0, 1373, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_04_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-6/23_icon_icon_23.png
try:
    _c23 = get_crop(23, 59, 62)
    canvas.paste(_c23, (245, 2), _c23)
except Exception:
    pass
layout["icon_23"] = [245, 2, 304, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_04_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-6/24_icon_icon_24.png
try:
    _c24 = get_crop(24, 54, 61)
    canvas.paste(_c24, (314, 3), _c24)
except Exception:
    pass
layout["icon_24"] = [314, 3, 368, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_04_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-6/25_icon_clickable_20.png
try:
    _c25 = get_crop(25, 144, 144)
    canvas.paste(_c25, (1248, 1729), _c25)
except Exception:
    pass
layout["clickable_20"] = [1248, 1729, 1392, 1873]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_04_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-6/26_icon_Clear_all.png
try:
    _c26 = get_crop(26, 178, 144)
    canvas.paste(_c26, (1214, 72), _c26)
except Exception:
    pass
layout["Clear_all"] = [1214, 72, 1392, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_04_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-6/27_text_9.11.png
try:
    _c27 = get_crop(27, 89, 43)
    canvas.paste(_c27, (20, 17), _c27)
except Exception:
    pass
layout["9.11"] = [20, 17, 109, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_04_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-6/28_text_Filters.png
try:
    _c28 = get_crop(28, 180, 66)
    canvas.paste(_c28, (631, 116), _c28)
except Exception:
    pass
layout["Filters"] = [631, 116, 811, 182]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_04_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-6/29_text_Categories.png
try:
    _c29 = get_crop(29, 187, 135)
    canvas.paste(_c29, (36, 383), _c29)
except Exception:
    pass
layout["Categories"] = [36, 383, 223, 518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_04_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-6/30_text_Show_all_categories.png
try:
    _c30 = get_crop(30, 516, 144)
    canvas.paste(_c30, (0, 518), _c30)
except Exception:
    pass
layout["Show_all_categories"] = [0, 518, 516, 662]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_04_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-6/31_text_Event_type.png
try:
    _c31 = get_crop(31, 298, 135)
    canvas.paste(_c31, (36, 829), _c31)
except Exception:
    pass
layout["Event_type"] = [36, 829, 334, 964]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_04_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-6/32_text_Show_all_event_types.png
try:
    _c32 = get_crop(32, 535, 144)
    canvas.paste(_c32, (0, 964), _c32)
except Exception:
    pass
layout["Show_all_event_types"] = [0, 964, 535, 1108]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_04_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-6/33_text_Languages.png
try:
    _c33 = get_crop(33, 210, 135)
    canvas.paste(_c33, (36, 1275), _c33)
except Exception:
    pass
layout["Languages"] = [36, 1275, 246, 1410]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_04_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-6/34_text_Show_all_languages.png
try:
    _c34 = get_crop(34, 511, 144)
    canvas.paste(_c34, (0, 1410), _c34)
except Exception:
    pass
layout["Show_all_languages"] = [0, 1410, 511, 1554]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_04_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-6/35_text_Price.png
try:
    _c35 = get_crop(35, 149, 63)
    canvas.paste(_c35, (45, 1613), _c35)
except Exception:
    pass
layout["Price"] = [45, 1613, 194, 1676]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_04_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-6/36_text_Only_free_events.png
try:
    _c36 = get_crop(36, 660, 144)
    canvas.paste(_c36, (54, 2024), _c36)
except Exception:
    pass
layout["Only_free_events"] = [54, 2024, 714, 2168]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_04_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-6/37_text_Sort_by.png
try:
    _c37 = get_crop(37, 206, 75)
    canvas.paste(_c37, (42, 1931), _c37)
except Exception:
    pass
layout["Sort_by"] = [42, 1931, 248, 2006]
