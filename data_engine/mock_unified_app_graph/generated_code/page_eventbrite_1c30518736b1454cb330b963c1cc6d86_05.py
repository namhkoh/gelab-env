# page_id: page_eventbrite_1c30518736b1454cb330b963c1cc6d86_05
# screenshot: 2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-7.png
# step_index: 5/9
# task: Open Eventbrite. Search for "Open Mic Nights". Filter the results to only include free events. Select the first non-promoted event in the list - what"s the location of that event?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw background and structural UI elements for the filters page

# Canvas is a provided PIL Image (1440x2960) and draw is an ImageDraw.Draw object.
# Fonts are available but not used for any text (text elements will be pasted separately).

# Colors
status_bar_color = (199, 199, 199)   # light grey for status bar
header_bg = (255, 255, 255)          # white header
page_bg = (255, 255, 255)            # main page background (white)
divider_color = (230, 228, 235)      # subtle divider
muted_shadow = (220, 216, 226)       # shadow/edge for controls
segment_off = (237, 235, 239)        # unselected segment bg
segment_on = (255, 255, 255)         # selected segment bg
apply_border = (150, 142, 160)       # border color for Apply filters button

# Clear full canvas to page background (in case not already)
draw.rectangle([(0, 0), canvas.size], fill=page_bg)

# Status bar (top area with system icons) - approximate height 72px
status_h = 72
draw.rectangle([(0, 0), (1440, status_h)], fill=status_bar_color)

# Header area (toolbar) - placed below status bar; title is centered but will be pasted later
header_top = status_h
header_bottom = 160
draw.rectangle([(0, header_top), (1440, header_bottom)], fill=header_bg)
# Subtle bottom divider for header
draw.line([(24, header_bottom), (1440 - 24, header_bottom)], fill=divider_color, width=1)

# Main content area - already white; add section separators to structure content
# Approximate separators under categories, event types, languages, price, and sort-by sections
separators_y = [540, 980, 1418, 1630, 1920]  # chosen to reflect the visual grouping
for y in separators_y:
    draw.line([(24, y), (1440 - 24, y)], fill=divider_color, width=1)

# Segmented control for "Sort by" (two segments: Relevance + Date)
# Use coordinates matching detected elements roughly (expand slightly for outline)
seg_x = 36
seg_y = 2024
seg_w = 1368
seg_h = 144
seg_radius = 18

# Outer rounded rect (slight shadow / border)
draw.rounded_rectangle(
    [(seg_x, seg_y), (seg_x + seg_w, seg_y + seg_h)],
    radius=seg_radius,
    fill=muted_shadow
)

# Inner background inset to create a border effect
inset = 6
draw.rounded_rectangle(
    [(seg_x + inset, seg_y + inset), (seg_x + seg_w - inset, seg_y + seg_h - inset)],
    radius=max(0, seg_radius - inset),
    fill=segment_off
)

# Left selected segment (slightly lighter)
left_w = (seg_w // 2) - 6
left_rect = [
    (seg_x + inset, seg_y + inset),
    (seg_x + inset + left_w, seg_y + seg_h - inset)
]
draw.rounded_rectangle(left_rect, radius=max(0, seg_radius - inset), fill=segment_on)

# Draw a subtle dividing line between segments (center)
center_x = seg_x + seg_w // 2
draw.line([(center_x, seg_y + inset), (center_x, seg_y + seg_h - inset)], fill=divider_color, width=1)

# Apply filters button at bottom (rounded rectangle with border)
btn_x = 48
btn_y = 2768
btn_w = 1344
btn_h = 144
btn_radius = 14

# Button background (white) with subtle border
draw.rounded_rectangle(
    [(btn_x, btn_y), (btn_x + btn_w, btn_y + btn_h)],
    radius=btn_radius,
    fill=(255, 255, 255),
    outline=apply_border,
    width=4
)

# Subtle inner highlight to mimic the inset look
inner_inset = 6
draw.rounded_rectangle(
    [(btn_x + inner_inset, btn_y + inner_inset), (btn_x + btn_w - inner_inset, btn_y + btn_h - inner_inset)],
    radius=max(0, btn_radius - inner_inset),
    fill=(255, 255, 255)
)

# Additional subtle horizontal padding lines to separate groups near top of content
# (helps visually group the filters without drawing any icons/text)
group_x0 = 24
group_x1 = 1440 - 24
group_lines = [
    (group_x0, 300, group_x1, 300),   # after categories heading area
    (group_x0, 760, group_x1, 760),   # after event type area
    (group_x0, 1230, group_x1, 1230), # after languages area
]
for x0, y, x1, _ in group_lines:
    draw.line([(x0, y), (x1, y)], fill=divider_color, width=1)

# Small top-left rounded corner accent on header to mimic native toolbar edge (decorative)
accent_w = 56
accent_h = 56
accent_bbox = (24, header_top + 12, 24 + accent_w, header_top + 12 + accent_h)
draw.rounded_rectangle(accent_bbox, radius=12, fill=header_bg, outline=divider_color, width=1)

# End of structural drawing. Text and icons will be pasted on top by the pipeline.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_05_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-7/00_icon_Food_Drink.png
try:
    _c0 = get_crop(0, 312, 144)
    canvas.paste(_c0, (512, 383), _c0)
except Exception:
    pass
layout["Food_&_Drink"] = [512, 383, 824, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_05_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-7/01_icon_Music.png
try:
    _c1 = get_crop(1, 187, 135)
    canvas.paste(_c1, (36, 383), _c1)
except Exception:
    pass
layout["Music"] = [36, 383, 223, 518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_05_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-7/02_icon_Community.png
try:
    _c2 = get_crop(2, 294, 144)
    canvas.paste(_c2, (848, 383), _c2)
except Exception:
    pass
layout["Community"] = [848, 383, 1142, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_05_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-7/03_icon_Business.png
try:
    _c3 = get_crop(3, 241, 135)
    canvas.paste(_c3, (247, 383), _c3)
except Exception:
    pass
layout["Business"] = [247, 383, 488, 518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_05_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-7/04_icon_French.png
try:
    _c4 = get_crop(4, 205, 144)
    canvas.paste(_c4, (768, 1275), _c4)
except Exception:
    pass
layout["French"] = [768, 1275, 973, 1419]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_05_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-7/05_icon_Spanish.png
try:
    _c5 = get_crop(5, 225, 144)
    canvas.paste(_c5, (519, 1275), _c5)
except Exception:
    pass
layout["Spanish"] = [519, 1275, 744, 1419]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_05_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-7/06_icon_Expo.png
try:
    _c6 = get_crop(6, 167, 144)
    canvas.paste(_c6, (614, 829), _c6)
except Exception:
    pass
layout["Expo"] = [614, 829, 781, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_05_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-7/07_icon_Arts.png
try:
    _c7 = get_crop(7, 152, 144)
    canvas.paste(_c7, (1166, 383), _c7)
except Exception:
    pass
layout["Arts"] = [1166, 383, 1318, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_05_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-7/08_icon_Italian.png
try:
    _c8 = get_crop(8, 191, 144)
    canvas.paste(_c8, (997, 1275), _c8)
except Exception:
    pass
layout["Italian"] = [997, 1275, 1188, 1419]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_05_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-7/09_icon_Seminar.png
try:
    _c9 = get_crop(9, 232, 144)
    canvas.paste(_c9, (358, 829), _c9)
except Exception:
    pass
layout["Seminar"] = [358, 829, 590, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_05_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-7/10_icon_Convention.png
try:
    _c10 = get_crop(10, 293, 144)
    canvas.paste(_c10, (805, 829), _c10)
except Exception:
    pass
layout["Convention"] = [805, 829, 1098, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_05_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-7/11_icon_Festival.png
try:
    _c11 = get_crop(11, 219, 144)
    canvas.paste(_c11, (1122, 829), _c11)
except Exception:
    pass
layout["Festival"] = [1122, 829, 1341, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_05_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-7/12_icon_German.png
try:
    _c12 = get_crop(12, 225, 135)
    canvas.paste(_c12, (270, 1275), _c12)
except Exception:
    pass
layout["German"] = [270, 1275, 495, 1410]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_05_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-7/13_icon_English.png
try:
    _c13 = get_crop(13, 210, 135)
    canvas.paste(_c13, (36, 1275), _c13)
except Exception:
    pass
layout["English"] = [36, 1275, 246, 1410]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_05_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-7/14_icon_Conference.png
try:
    _c14 = get_crop(14, 298, 135)
    canvas.paste(_c14, (36, 829), _c14)
except Exception:
    pass
layout["Conference"] = [36, 829, 334, 964]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_05_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-7/15_icon_Date.png
try:
    _c15 = get_crop(15, 660, 144)
    canvas.paste(_c15, (726, 2024), _c15)
except Exception:
    pass
layout["Date"] = [726, 2024, 1386, 2168]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_05_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-7/16_icon_Relevance.png
try:
    _c16 = get_crop(16, 660, 144)
    canvas.paste(_c16, (54, 2024), _c16)
except Exception:
    pass
layout["Relevance"] = [54, 2024, 714, 2168]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_05_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-7/17_icon_Apply_filters.png
try:
    _c17 = get_crop(17, 1344, 144)
    canvas.paste(_c17, (48, 2768), _c17)
except Exception:
    pass
layout["Apply_filters"] = [48, 2768, 1392, 2912]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_05_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-7/18_icon_4.54.png
try:
    _c18 = get_crop(18, 144, 144)
    canvas.paste(_c18, (12, 72), _c18)
except Exception:
    pass
layout["4.54"] = [12, 72, 156, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_05_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-7/19_icon_4.54.png
try:
    _c19 = get_crop(19, 61, 65)
    canvas.paste(_c19, (179, 1), _c19)
except Exception:
    pass
layout["4.54"] = [179, 1, 240, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_05_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-7/20_icon_icon_20.png
try:
    _c20 = get_crop(20, 66, 63)
    canvas.paste(_c20, (307, 2), _c20)
except Exception:
    pass
layout["icon_20"] = [307, 2, 373, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_05_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-7/21_icon_4.54.png
try:
    _c21 = get_crop(21, 64, 66)
    canvas.paste(_c21, (112, 1), _c21)
except Exception:
    pass
layout["4.54"] = [112, 1, 176, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_05_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-7/22_icon_Clear_all.png
try:
    _c22 = get_crop(22, 55, 69)
    canvas.paste(_c22, (1319, 0), _c22)
except Exception:
    pass
layout["Clear_all"] = [1319, 0, 1374, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_05_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-7/23_icon_icon_23.png
try:
    _c23 = get_crop(23, 52, 63)
    canvas.paste(_c23, (248, 2), _c23)
except Exception:
    pass
layout["icon_23"] = [248, 2, 300, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_05_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-7/24_icon_Clear_all.png
try:
    _c24 = get_crop(24, 99, 70)
    canvas.paste(_c24, (1211, 0), _c24)
except Exception:
    pass
layout["Clear_all"] = [1211, 0, 1310, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_05_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-7/25_icon_clickable_20.png
try:
    _c25 = get_crop(25, 144, 144)
    canvas.paste(_c25, (1248, 1729), _c25)
except Exception:
    pass
layout["clickable_20"] = [1248, 1729, 1392, 1873]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_05_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-7/26_icon_Clear_all.png
try:
    _c26 = get_crop(26, 178, 144)
    canvas.paste(_c26, (1214, 72), _c26)
except Exception:
    pass
layout["Clear_all"] = [1214, 72, 1392, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_05_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-7/27_text_Filters.png
try:
    _c27 = get_crop(27, 180, 66)
    canvas.paste(_c27, (631, 116), _c27)
except Exception:
    pass
layout["Filters"] = [631, 116, 811, 182]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_05_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-7/28_text_Categories.png
try:
    _c28 = get_crop(28, 187, 135)
    canvas.paste(_c28, (36, 383), _c28)
except Exception:
    pass
layout["Categories"] = [36, 383, 223, 518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_05_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-7/29_text_Show_all_categories.png
try:
    _c29 = get_crop(29, 516, 144)
    canvas.paste(_c29, (0, 518), _c29)
except Exception:
    pass
layout["Show_all_categories"] = [0, 518, 516, 662]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_05_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-7/30_text_Event_type.png
try:
    _c30 = get_crop(30, 298, 135)
    canvas.paste(_c30, (36, 829), _c30)
except Exception:
    pass
layout["Event_type"] = [36, 829, 334, 964]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_05_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-7/31_text_Show_all_event_types.png
try:
    _c31 = get_crop(31, 535, 144)
    canvas.paste(_c31, (0, 964), _c31)
except Exception:
    pass
layout["Show_all_event_types"] = [0, 964, 535, 1108]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_05_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-7/32_text_Languages.png
try:
    _c32 = get_crop(32, 210, 135)
    canvas.paste(_c32, (36, 1275), _c32)
except Exception:
    pass
layout["Languages"] = [36, 1275, 246, 1410]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_05_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-7/33_text_Show_all_languages.png
try:
    _c33 = get_crop(33, 511, 144)
    canvas.paste(_c33, (0, 1410), _c33)
except Exception:
    pass
layout["Show_all_languages"] = [0, 1410, 511, 1554]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_05_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-7/34_text_Price.png
try:
    _c34 = get_crop(34, 149, 63)
    canvas.paste(_c34, (45, 1613), _c34)
except Exception:
    pass
layout["Price"] = [45, 1613, 194, 1676]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_05_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-7/35_text_Only_free_events.png
try:
    _c35 = get_crop(35, 660, 144)
    canvas.paste(_c35, (54, 2024), _c35)
except Exception:
    pass
layout["Only_free_events"] = [54, 2024, 714, 2168]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_05_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-7/36_text_Sort_by.png
try:
    _c36 = get_crop(36, 206, 75)
    canvas.paste(_c36, (42, 1931), _c36)
except Exception:
    pass
layout["Sort_by"] = [42, 1931, 248, 2006]
