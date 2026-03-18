# page_id: page_eventbrite_d7ac75f457a4487c904e7baa93180729_05
# screenshot: 2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-7.png
# step_index: 5/11
# task: Open Eventbrite. Search for 'Cooking' classes. Filter to only show free events that occur in the weekend. Select the first event and proceed to checkout.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Background and structural layout for the Filters UI
# Uses provided canvas (1440x2960) and draw (ImageDraw)

# Colors
bg = (250, 250, 252)            # very light off-white background
status_bar = (190, 190, 190)    # status bar grey
header_bg = (255, 255, 255)     # header area (white)
divider = (230, 230, 235)       # subtle divider lines
muted = (245, 244, 248)         # muted panel fill
segment_fill = (244, 243, 247)  # segmented control background
segment_border = (215, 213, 221)
shadow = (220, 219, 224)

w, h = canvas.size

# Fill full background
draw.rectangle([(0, 0), (w, h)], fill=bg)

# Status bar area (top)
status_h = 72
draw.rectangle([(0, 0), (w, status_h)], fill=status_bar)

# Header area (below status bar) - keep it light and add divider
header_top = status_h
header_bottom = 160
draw.rectangle([(0, header_top), (w, header_bottom)], fill=header_bg)
# thin bottom divider under header
draw.line([(24, header_bottom), (w-24, header_bottom)], fill=divider, width=1)

# Main content left padding guide (visual structure only)
content_left = 36
content_right = w - 36

# Section separators - place between logical groups to match screenshot spacing
# After Categories area
sep1_y = 520
draw.line([(content_left, sep1_y), (content_right, sep1_y)], fill=divider, width=1)

# After Event type area
sep2_y = 980
draw.line([(content_left, sep2_y), (content_right, sep2_y)], fill=divider, width=1)

# After Languages area
sep3_y = 1415
draw.line([(content_left, sep3_y), (content_right, sep3_y)], fill=divider, width=1)

# After Price/Switch area (visual spacing)
sep4_y = 1800
draw.line([(content_left, sep4_y), (content_right, sep4_y)], fill=divider, width=1)

# "Sort by" segmented control background (rounded rectangle)
seg_top = 2016
seg_bottom = 2164
seg_left = 48
seg_right = w - 48
seg_radius = 14
# subtle shadow line above segment
draw.line([(seg_left, seg_top-6), (seg_right, seg_top-6)], fill=shadow, width=2)
# segment fill and border
draw.rounded_rectangle([(seg_left, seg_top), (seg_right, seg_bottom)],
                       radius=seg_radius, fill=segment_fill, outline=segment_border, width=2)

# Add subtle inner divider in the segmented control to hint two segments (no text)
# position roughly matching screenshot split (left ~ half+)
split_x = seg_left + int((seg_right - seg_left) * 0.5) + 20
draw.line([(split_x, seg_top+6), (split_x, seg_bottom-6)], fill=segment_border, width=1)

# Light content card behind filter groups (just broad panels, no text/chips)
# Draw faint rounded rectangles to suggest grouped areas
card_radius = 10
card_color = (255, 255, 255)  # white cards on the light background
card_outline = (240, 239, 242)

# Categories card area
cat_top = 200
cat_bottom = 520
draw.rounded_rectangle([(24, cat_top), (w-24, cat_bottom)],
                       radius=card_radius, fill=card_color, outline=card_outline, width=1)

# Event type card area
evt_top = 560
evt_bottom = 980
draw.rounded_rectangle([(24, evt_top), (w-24, evt_bottom)],
                       radius=card_radius, fill=card_color, outline=card_outline, width=1)

# Languages card area
lang_top = 1020
lang_bottom = 1415
draw.rounded_rectangle([(24, lang_top), (w-24, lang_bottom)],
                       radius=card_radius, fill=card_color, outline=card_outline, width=1)

# Price / Only free events area (no toggle drawn - leave space)
price_top = 1450
price_bottom = 1800
draw.rounded_rectangle([(24, price_top), (w-24, price_bottom)],
                       radius=card_radius, fill=card_color, outline=card_outline, width=1)

# Bottom safe-area separator above the persistent action area (visual only)
bottom_sep_y = 2680
draw.line([(24, bottom_sep_y), (w-24, bottom_sep_y)], fill=divider, width=2)

# Draw faint corner guides near edges to indicate actionable areas (subtle)
corner_guide_color = (245, 244, 247)
draw.rectangle([(12, header_bottom+12), (36, header_bottom+20)], fill=corner_guide_color)
draw.rectangle([(w-36, header_bottom+12), (w-12, header_bottom+20)], fill=corner_guide_color)

# Small shadow under header to separate from content
draw.line([(24, header_bottom+1), (w-24, header_bottom+1)], fill=shadow, width=1)

# Finished structural drawing

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_05_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-7/00_icon_Food_Drink.png
try:
    _c0 = get_crop(0, 312, 144)
    canvas.paste(_c0, (512, 383), _c0)
except Exception:
    pass
layout["Food_&_Drink"] = [512, 383, 824, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_05_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-7/01_icon_Music.png
try:
    _c1 = get_crop(1, 187, 135)
    canvas.paste(_c1, (36, 383), _c1)
except Exception:
    pass
layout["Music"] = [36, 383, 223, 518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_05_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-7/02_icon_Community.png
try:
    _c2 = get_crop(2, 294, 144)
    canvas.paste(_c2, (848, 383), _c2)
except Exception:
    pass
layout["Community"] = [848, 383, 1142, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_05_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-7/03_icon_French.png
try:
    _c3 = get_crop(3, 205, 144)
    canvas.paste(_c3, (768, 1275), _c3)
except Exception:
    pass
layout["French"] = [768, 1275, 973, 1419]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_05_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-7/04_icon_Business.png
try:
    _c4 = get_crop(4, 241, 135)
    canvas.paste(_c4, (247, 383), _c4)
except Exception:
    pass
layout["Business"] = [247, 383, 488, 518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_05_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-7/05_icon_Spanish.png
try:
    _c5 = get_crop(5, 225, 144)
    canvas.paste(_c5, (519, 1275), _c5)
except Exception:
    pass
layout["Spanish"] = [519, 1275, 744, 1419]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_05_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-7/06_icon_Expo.png
try:
    _c6 = get_crop(6, 167, 144)
    canvas.paste(_c6, (614, 829), _c6)
except Exception:
    pass
layout["Expo"] = [614, 829, 781, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_05_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-7/07_icon_Italian.png
try:
    _c7 = get_crop(7, 191, 144)
    canvas.paste(_c7, (997, 1275), _c7)
except Exception:
    pass
layout["Italian"] = [997, 1275, 1188, 1419]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_05_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-7/08_icon_Seminar.png
try:
    _c8 = get_crop(8, 232, 144)
    canvas.paste(_c8, (358, 829), _c8)
except Exception:
    pass
layout["Seminar"] = [358, 829, 590, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_05_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-7/09_icon_Arts.png
try:
    _c9 = get_crop(9, 152, 144)
    canvas.paste(_c9, (1166, 383), _c9)
except Exception:
    pass
layout["Arts"] = [1166, 383, 1318, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_05_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-7/10_icon_Convention.png
try:
    _c10 = get_crop(10, 293, 144)
    canvas.paste(_c10, (805, 829), _c10)
except Exception:
    pass
layout["Convention"] = [805, 829, 1098, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_05_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-7/11_icon_Festival.png
try:
    _c11 = get_crop(11, 219, 144)
    canvas.paste(_c11, (1122, 829), _c11)
except Exception:
    pass
layout["Festival"] = [1122, 829, 1341, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_05_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-7/12_icon_German.png
try:
    _c12 = get_crop(12, 225, 135)
    canvas.paste(_c12, (270, 1275), _c12)
except Exception:
    pass
layout["German"] = [270, 1275, 495, 1410]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_05_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-7/13_icon_English.png
try:
    _c13 = get_crop(13, 210, 135)
    canvas.paste(_c13, (36, 1275), _c13)
except Exception:
    pass
layout["English"] = [36, 1275, 246, 1410]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_05_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-7/14_icon_Conference.png
try:
    _c14 = get_crop(14, 298, 135)
    canvas.paste(_c14, (36, 829), _c14)
except Exception:
    pass
layout["Conference"] = [36, 829, 334, 964]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_05_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-7/15_icon_Date.png
try:
    _c15 = get_crop(15, 660, 144)
    canvas.paste(_c15, (726, 2024), _c15)
except Exception:
    pass
layout["Date"] = [726, 2024, 1386, 2168]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_05_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-7/16_icon_Relevance.png
try:
    _c16 = get_crop(16, 660, 144)
    canvas.paste(_c16, (54, 2024), _c16)
except Exception:
    pass
layout["Relevance"] = [54, 2024, 714, 2168]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_05_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-7/17_icon_Apply_filters.png
try:
    _c17 = get_crop(17, 1344, 144)
    canvas.paste(_c17, (48, 2768), _c17)
except Exception:
    pass
layout["Apply_filters"] = [48, 2768, 1392, 2912]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_05_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-7/18_icon_4.38.png
try:
    _c18 = get_crop(18, 144, 144)
    canvas.paste(_c18, (12, 72), _c18)
except Exception:
    pass
layout["4.38"] = [12, 72, 156, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_05_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-7/19_icon_4.38.png
try:
    _c19 = get_crop(19, 61, 65)
    canvas.paste(_c19, (179, 1), _c19)
except Exception:
    pass
layout["4.38"] = [179, 1, 240, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_05_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-7/20_icon_4.38.png
try:
    _c20 = get_crop(20, 65, 65)
    canvas.paste(_c20, (111, 1), _c20)
except Exception:
    pass
layout["4.38"] = [111, 1, 176, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_05_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-7/21_icon_icon_21.png
try:
    _c21 = get_crop(21, 66, 62)
    canvas.paste(_c21, (307, 3), _c21)
except Exception:
    pass
layout["icon_21"] = [307, 3, 373, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_05_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-7/22_icon_Clear_all.png
try:
    _c22 = get_crop(22, 55, 69)
    canvas.paste(_c22, (1319, 0), _c22)
except Exception:
    pass
layout["Clear_all"] = [1319, 0, 1374, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_05_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-7/23_icon_icon_23.png
try:
    _c23 = get_crop(23, 52, 63)
    canvas.paste(_c23, (248, 2), _c23)
except Exception:
    pass
layout["icon_23"] = [248, 2, 300, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_05_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-7/24_icon_Clear_all.png
try:
    _c24 = get_crop(24, 99, 70)
    canvas.paste(_c24, (1211, 0), _c24)
except Exception:
    pass
layout["Clear_all"] = [1211, 0, 1310, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_05_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-7/25_icon_Toggle_to_show_only_free_events..png
try:
    _c25 = get_crop(25, 144, 144)
    canvas.paste(_c25, (1248, 1729), _c25)
except Exception:
    pass
layout["Toggle_to_show_only_free_"] = [1248, 1729, 1392, 1873]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_05_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-7/26_icon_Clear_all.png
try:
    _c26 = get_crop(26, 178, 144)
    canvas.paste(_c26, (1214, 72), _c26)
except Exception:
    pass
layout["Clear_all"] = [1214, 72, 1392, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_05_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-7/27_text_4.38.png
try:
    _c27 = get_crop(27, 89, 45)
    canvas.paste(_c27, (22, 15), _c27)
except Exception:
    pass
layout["4.38"] = [22, 15, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_05_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-7/28_text_Filters.png
try:
    _c28 = get_crop(28, 180, 66)
    canvas.paste(_c28, (631, 116), _c28)
except Exception:
    pass
layout["Filters"] = [631, 116, 811, 182]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_05_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-7/29_text_Categories.png
try:
    _c29 = get_crop(29, 187, 135)
    canvas.paste(_c29, (36, 383), _c29)
except Exception:
    pass
layout["Categories"] = [36, 383, 223, 518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_05_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-7/30_text_Show_all_categories.png
try:
    _c30 = get_crop(30, 516, 144)
    canvas.paste(_c30, (0, 518), _c30)
except Exception:
    pass
layout["Show_all_categories"] = [0, 518, 516, 662]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_05_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-7/31_text_Event_type.png
try:
    _c31 = get_crop(31, 298, 135)
    canvas.paste(_c31, (36, 829), _c31)
except Exception:
    pass
layout["Event_type"] = [36, 829, 334, 964]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_05_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-7/32_text_Show_all_event_types.png
try:
    _c32 = get_crop(32, 535, 144)
    canvas.paste(_c32, (0, 964), _c32)
except Exception:
    pass
layout["Show_all_event_types"] = [0, 964, 535, 1108]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_05_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-7/33_text_Languages.png
try:
    _c33 = get_crop(33, 210, 135)
    canvas.paste(_c33, (36, 1275), _c33)
except Exception:
    pass
layout["Languages"] = [36, 1275, 246, 1410]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_05_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-7/34_text_Show_all_languages.png
try:
    _c34 = get_crop(34, 511, 144)
    canvas.paste(_c34, (0, 1410), _c34)
except Exception:
    pass
layout["Show_all_languages"] = [0, 1410, 511, 1554]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_05_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-7/35_text_Price.png
try:
    _c35 = get_crop(35, 149, 63)
    canvas.paste(_c35, (45, 1613), _c35)
except Exception:
    pass
layout["Price"] = [45, 1613, 194, 1676]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_05_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-7/36_text_Only_free_events.png
try:
    _c36 = get_crop(36, 660, 144)
    canvas.paste(_c36, (54, 2024), _c36)
except Exception:
    pass
layout["Only_free_events"] = [54, 2024, 714, 2168]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_05_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-7/37_text_Sort_by.png
try:
    _c37 = get_crop(37, 206, 75)
    canvas.paste(_c37, (42, 1931), _c37)
except Exception:
    pass
layout["Sort_by"] = [42, 1931, 248, 2006]
