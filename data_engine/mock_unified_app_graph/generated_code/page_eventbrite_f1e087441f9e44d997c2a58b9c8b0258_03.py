# page_id: page_eventbrite_f1e087441f9e44d997c2a58b9c8b0258_03
# screenshot: 2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-5.png
# step_index: 3/10
# task: Open Eventbrite. Find the 'Arts' category. Select events that are available for this weekend. From the results, open the first item and add it to favorite. Follow the organizer.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Background and structure drawing for the UI page (uses provided `canvas` and `draw`)

# Colors
status_color = (200, 200, 200)         # top status bar gray
header_div = (236, 234, 240)          # subtle header divider
page_bg = (255, 255, 255)             # main page white
group_bg = (244, 250, 255)            # very light bluish group background
group_outline = (226, 224, 230)       # subtle outline for groups
divider = (240, 239, 242)             # thin separators
sort_bg = (246, 244, 247)             # sort control background
sort_outline = (217, 212, 219)        # sort outline
bottom_area = (250, 250, 252)         # bottom safe area background
shadow_color = (230, 228, 233)        # faint shadows

W, H = canvas.size

# Fill canvas (dominant background)
draw.rectangle([(0, 0), (W, H)], fill=page_bg)

# Status bar (top)
status_h = 78
draw.rectangle([(0, 0), (W, status_h)], fill=status_color)
# subtle bottom divider under status
draw.rectangle([(0, status_h - 1), (W, status_h + 1)], fill=header_div)

# Header toolbar area (space for back arrow and title)
header_top = status_h
header_bottom = 170
draw.rectangle([(0, header_top), (W, header_bottom)], fill=page_bg)
# header bottom divider / subtle shadow
draw.rectangle([(0, header_bottom - 2), (W, header_bottom)], fill=header_div)

# Group/background cards for sections (rounded rectangles)
# Categories group (behind chips)
cat_box = (36, 240, W - 36, 620)
draw.rounded_rectangle(cat_box, radius=20, fill=group_bg, outline=group_outline, width=1)
# tiny shadow under categories card
draw.rectangle([(cat_box[0]+6, cat_box[3]), (cat_box[2]-6, cat_box[3]+3)], fill=shadow_color)

# Event type group
etype_box = (36, 720, W - 36, 1040)
draw.rounded_rectangle(etype_box, radius=20, fill=group_bg, outline=group_outline, width=1)
draw.rectangle([(etype_box[0]+6, etype_box[3]), (etype_box[2]-6, etype_box[3]+3)], fill=shadow_color)

# Languages group
lang_box = (36, 1140, W - 36, 1500)
draw.rounded_rectangle(lang_box, radius=20, fill=group_bg, outline=group_outline, width=1)
draw.rectangle([(lang_box[0]+6, lang_box[3]), (lang_box[2]-6, lang_box[3]+3)], fill=shadow_color)

# Price / "Only free events" area (light background to separate from sections)
price_box = (36, 1540, W - 36, 1860)
draw.rounded_rectangle(price_box, radius=16, fill=(255,255,255), outline=(245,243,246), width=1)
# faint divider under price area
draw.rectangle([(price_box[0]+8, price_box[3]), (price_box[2]-8, price_box[3]+2)], fill=divider)

# Separator lines between major sections (thin)
seps = [620 + 20, 1040 + 20, 1500 + 20, 1900]  # approximate y positions for separators
for y in seps:
    draw.line([(36, y), (W - 36, y)], fill=divider, width=1)

# Sort by segmented control container (background only)
sort_y_top = 2016
sort_y_bottom = 2088
sort_box = (36, sort_y_top, W - 36, sort_y_bottom)
draw.rounded_rectangle(sort_box, radius=14, fill=sort_bg, outline=sort_outline, width=1)
# inner slight shadow at bottom of sort box
draw.rectangle([(sort_box[0]+6, sort_box[3]), (sort_box[2]-6, sort_box[3]+3)], fill=shadow_color)

# Top area above bottom action: subtle dividing area and background
bottom_bg_top = 2640
draw.rectangle([(0, bottom_bg_top), (W, H)], fill=bottom_area)
# top divider for bottom area
draw.line([(24, bottom_bg_top), (W - 24, bottom_bg_top)], fill=divider, width=2)

# Apply filters area background (behind the button to provide context)
# We intentionally draw a soft background only; the actual button will be pasted on top.
apply_bg = (255, 255, 255)
apply_box = (24, 2696, W - 24, 2956)
draw.rounded_rectangle(apply_box, radius=12, fill=apply_bg, outline=group_outline, width=2)
# soft shadow under apply area
draw.rectangle([(apply_box[0]+6, apply_box[3]), (apply_box[2]-6, apply_box[3]+6)], fill=shadow_color)

# Final subtle touches: small vertical spacing lines to guide sections (left margin markers)
marker_x = 36
for y in [220, 700, 1120, 1540, 1930]:
    draw.line([(marker_x, y), (marker_x + 8, y)], fill=shadow_color, width=1)

# End of structural/background drawing.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_03_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-5/00_icon_Food_Drink.png
try:
    _c0 = get_crop(0, 312, 144)
    canvas.paste(_c0, (512, 383), _c0)
except Exception:
    pass
layout["Food_&_Drink"] = [512, 383, 824, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_03_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-5/01_icon_Music.png
try:
    _c1 = get_crop(1, 187, 135)
    canvas.paste(_c1, (36, 383), _c1)
except Exception:
    pass
layout["Music"] = [36, 383, 223, 518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_03_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-5/02_icon_Community.png
try:
    _c2 = get_crop(2, 294, 144)
    canvas.paste(_c2, (848, 383), _c2)
except Exception:
    pass
layout["Community"] = [848, 383, 1142, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_03_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-5/03_icon_French.png
try:
    _c3 = get_crop(3, 205, 144)
    canvas.paste(_c3, (768, 1275), _c3)
except Exception:
    pass
layout["French"] = [768, 1275, 973, 1419]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_03_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-5/04_icon_Business.png
try:
    _c4 = get_crop(4, 241, 135)
    canvas.paste(_c4, (247, 383), _c4)
except Exception:
    pass
layout["Business"] = [247, 383, 488, 518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_03_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-5/05_icon_Spanish.png
try:
    _c5 = get_crop(5, 225, 144)
    canvas.paste(_c5, (519, 1275), _c5)
except Exception:
    pass
layout["Spanish"] = [519, 1275, 744, 1419]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_03_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-5/06_icon_Expo.png
try:
    _c6 = get_crop(6, 167, 144)
    canvas.paste(_c6, (614, 829), _c6)
except Exception:
    pass
layout["Expo"] = [614, 829, 781, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_03_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-5/07_icon_Italian.png
try:
    _c7 = get_crop(7, 191, 144)
    canvas.paste(_c7, (997, 1275), _c7)
except Exception:
    pass
layout["Italian"] = [997, 1275, 1188, 1419]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_03_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-5/08_icon_Seminar.png
try:
    _c8 = get_crop(8, 232, 144)
    canvas.paste(_c8, (358, 829), _c8)
except Exception:
    pass
layout["Seminar"] = [358, 829, 590, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_03_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-5/09_icon_Arts.png
try:
    _c9 = get_crop(9, 152, 144)
    canvas.paste(_c9, (1166, 383), _c9)
except Exception:
    pass
layout["Arts"] = [1166, 383, 1318, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_03_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-5/10_icon_Convention.png
try:
    _c10 = get_crop(10, 293, 144)
    canvas.paste(_c10, (805, 829), _c10)
except Exception:
    pass
layout["Convention"] = [805, 829, 1098, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_03_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-5/11_icon_Festival.png
try:
    _c11 = get_crop(11, 219, 144)
    canvas.paste(_c11, (1122, 829), _c11)
except Exception:
    pass
layout["Festival"] = [1122, 829, 1341, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_03_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-5/12_icon_German.png
try:
    _c12 = get_crop(12, 225, 135)
    canvas.paste(_c12, (270, 1275), _c12)
except Exception:
    pass
layout["German"] = [270, 1275, 495, 1410]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_03_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-5/13_icon_English.png
try:
    _c13 = get_crop(13, 210, 135)
    canvas.paste(_c13, (36, 1275), _c13)
except Exception:
    pass
layout["English"] = [36, 1275, 246, 1410]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_03_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-5/14_icon_Conference.png
try:
    _c14 = get_crop(14, 298, 135)
    canvas.paste(_c14, (36, 829), _c14)
except Exception:
    pass
layout["Conference"] = [36, 829, 334, 964]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_03_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-5/15_icon_Date.png
try:
    _c15 = get_crop(15, 660, 144)
    canvas.paste(_c15, (726, 2024), _c15)
except Exception:
    pass
layout["Date"] = [726, 2024, 1386, 2168]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_03_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-5/16_icon_Relevance.png
try:
    _c16 = get_crop(16, 660, 144)
    canvas.paste(_c16, (54, 2024), _c16)
except Exception:
    pass
layout["Relevance"] = [54, 2024, 714, 2168]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_03_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-5/17_icon_Apply_filters.png
try:
    _c17 = get_crop(17, 1344, 144)
    canvas.paste(_c17, (48, 2768), _c17)
except Exception:
    pass
layout["Apply_filters"] = [48, 2768, 1392, 2912]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_03_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-5/18_icon_4.32.png
try:
    _c18 = get_crop(18, 144, 144)
    canvas.paste(_c18, (12, 72), _c18)
except Exception:
    pass
layout["4.32"] = [12, 72, 156, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_03_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-5/19_icon_4.32.png
try:
    _c19 = get_crop(19, 61, 65)
    canvas.paste(_c19, (179, 1), _c19)
except Exception:
    pass
layout["4.32"] = [179, 1, 240, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_03_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-5/20_icon_icon_20.png
try:
    _c20 = get_crop(20, 66, 62)
    canvas.paste(_c20, (307, 3), _c20)
except Exception:
    pass
layout["icon_20"] = [307, 3, 373, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_03_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-5/21_icon_4.32.png
try:
    _c21 = get_crop(21, 64, 65)
    canvas.paste(_c21, (112, 1), _c21)
except Exception:
    pass
layout["4.32"] = [112, 1, 176, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_03_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-5/22_icon_Clear_all.png
try:
    _c22 = get_crop(22, 55, 69)
    canvas.paste(_c22, (1319, 0), _c22)
except Exception:
    pass
layout["Clear_all"] = [1319, 0, 1374, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_03_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-5/23_icon_icon_23.png
try:
    _c23 = get_crop(23, 52, 63)
    canvas.paste(_c23, (248, 2), _c23)
except Exception:
    pass
layout["icon_23"] = [248, 2, 300, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_03_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-5/24_icon_Clear_all.png
try:
    _c24 = get_crop(24, 99, 70)
    canvas.paste(_c24, (1211, 0), _c24)
except Exception:
    pass
layout["Clear_all"] = [1211, 0, 1310, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_03_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-5/25_icon_clickable_20.png
try:
    _c25 = get_crop(25, 144, 144)
    canvas.paste(_c25, (1248, 1729), _c25)
except Exception:
    pass
layout["clickable_20"] = [1248, 1729, 1392, 1873]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_03_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-5/26_icon_Clear_all.png
try:
    _c26 = get_crop(26, 178, 144)
    canvas.paste(_c26, (1214, 72), _c26)
except Exception:
    pass
layout["Clear_all"] = [1214, 72, 1392, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_03_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-5/27_text_4.32.png
try:
    _c27 = get_crop(27, 89, 45)
    canvas.paste(_c27, (22, 15), _c27)
except Exception:
    pass
layout["4.32"] = [22, 15, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_03_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-5/28_text_Filters.png
try:
    _c28 = get_crop(28, 180, 66)
    canvas.paste(_c28, (631, 116), _c28)
except Exception:
    pass
layout["Filters"] = [631, 116, 811, 182]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_03_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-5/29_text_Categories.png
try:
    _c29 = get_crop(29, 187, 135)
    canvas.paste(_c29, (36, 383), _c29)
except Exception:
    pass
layout["Categories"] = [36, 383, 223, 518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_03_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-5/30_text_Show_all_categories.png
try:
    _c30 = get_crop(30, 516, 144)
    canvas.paste(_c30, (0, 518), _c30)
except Exception:
    pass
layout["Show_all_categories"] = [0, 518, 516, 662]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_03_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-5/31_text_Event_type.png
try:
    _c31 = get_crop(31, 298, 135)
    canvas.paste(_c31, (36, 829), _c31)
except Exception:
    pass
layout["Event_type"] = [36, 829, 334, 964]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_03_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-5/32_text_Show_all_event_types.png
try:
    _c32 = get_crop(32, 535, 144)
    canvas.paste(_c32, (0, 964), _c32)
except Exception:
    pass
layout["Show_all_event_types"] = [0, 964, 535, 1108]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_03_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-5/33_text_Languages.png
try:
    _c33 = get_crop(33, 210, 135)
    canvas.paste(_c33, (36, 1275), _c33)
except Exception:
    pass
layout["Languages"] = [36, 1275, 246, 1410]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_03_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-5/34_text_Show_all_languages.png
try:
    _c34 = get_crop(34, 511, 144)
    canvas.paste(_c34, (0, 1410), _c34)
except Exception:
    pass
layout["Show_all_languages"] = [0, 1410, 511, 1554]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_03_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-5/35_text_Price.png
try:
    _c35 = get_crop(35, 149, 63)
    canvas.paste(_c35, (45, 1613), _c35)
except Exception:
    pass
layout["Price"] = [45, 1613, 194, 1676]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_03_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-5/36_text_Only_free_events.png
try:
    _c36 = get_crop(36, 660, 144)
    canvas.paste(_c36, (54, 2024), _c36)
except Exception:
    pass
layout["Only_free_events"] = [54, 2024, 714, 2168]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_03_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-5/37_text_Sort_by.png
try:
    _c37 = get_crop(37, 206, 75)
    canvas.paste(_c37, (42, 1931), _c37)
except Exception:
    pass
layout["Sort_by"] = [42, 1931, 248, 2006]
