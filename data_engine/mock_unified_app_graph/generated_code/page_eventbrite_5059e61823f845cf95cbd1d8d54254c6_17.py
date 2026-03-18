# page_id: page_eventbrite_5059e61823f845cf95cbd1d8d54254c6_17
# screenshot: 2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-19.png
# step_index: 17/19
# task: Open Eventbrite. Look for 'Education' in Los Angeles happening on May 4. Filter to show only free events. How many events are posted?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw UI background and structural elements for the filters page
# Available variables: canvas (PIL.Image), draw (PIL.ImageDraw), font_sm, font_md, font_lg, font_xl

w, h = canvas.size

# Colors (match visual style from screenshot)
bg_color = (255, 255, 255)            # main page background (white)
status_bar_color = (154, 160, 166)   # top status bar gray
divider_color = (236, 234, 240)      # thin dividers / subtle lines
muted_fill = (249, 249, 251)         # very light panel fills
panel_border = (212, 209, 216)       # subtle border for buttons/panels
seg_control_bg = (244, 242, 246)     # segmented control background
seg_selected = (230, 228, 233)       # selected segment fill
shadow_color = (230, 227, 234)       # faint shadow color

# 1) Overall background (in case canvas isn't pure white)
draw.rectangle([(0, 0), (w, h)], fill=bg_color)

# 2) Status bar area at top (approx 72px height)
status_h = 72
draw.rectangle([(0, 0), (w, status_h)], fill=status_bar_color)

# subtle top status bar bottom stroke (to separate from header)
draw.line([(0, status_h), (w, status_h)], fill=divider_color, width=1)

# 3) Header / toolbar area (below status bar)
header_top = status_h
header_bottom = 170
# Keep header background same as page (white) but add subtle drop shadow line
draw.rectangle([(0, header_top), (w, header_bottom)], fill=bg_color)
# Shadow / divider under header
draw.line([(24, header_bottom), (w-24, header_bottom)], fill=divider_color, width=2)

# 4) Section grouping backgrounds (rounded, faint) behind chip rows
# Categories chips area (behind icons that will be pasted)
cat_top = 320
cat_bottom = 540
draw.rounded_rectangle([(20, cat_top), (w-20, cat_bottom)], radius=22, fill=muted_fill, outline=None)

# Event type chips area
etype_top = 760
etype_bottom = 970
draw.rounded_rectangle([(20, etype_top), (w-20, etype_bottom)], radius=22, fill=muted_fill, outline=None)

# Languages chips area
lang_top = 1260
lang_bottom = 1470
draw.rounded_rectangle([(20, lang_top), (w-20, lang_bottom)], radius=22, fill=muted_fill, outline=None)

# 5) Price area separator & subtle band for "Only free events" region
price_top = 1580
price_bottom = 1720
# No full panel here in screenshot, but draw a faint divider and a subtle band area to group Price/Toggle
draw.line([(24, price_top), (w-24, price_top)], fill=divider_color, width=1)
draw.rounded_rectangle([(24, price_top+8), (w-24, price_bottom-8)], radius=12, fill=(255,255,255), outline=None)

# 6) Sort by segmented control background (rounded rectangle)
sort_top = 1948
sort_height = 116
seg_left = 48
seg_right = w - 48
seg_box = (seg_left, sort_top, seg_right, sort_top + sort_height)
draw.rounded_rectangle(seg_box, radius=14, fill=seg_control_bg, outline=panel_border)

# Draw subtle inner division line to separate segments (visual structure only)
mid_x = seg_left + (seg_right - seg_left) // 2
draw.line([(mid_x, sort_top+6), (mid_x, sort_top+sort_height-6)], fill=divider_color, width=1)

# Slight highlight for the left (selected) segment background (keeps text/icons for segments to be pasted)
left_seg = (seg_left+4, sort_top+4, mid_x-4, sort_top+sort_height-4)
draw.rounded_rectangle(left_seg, radius=10, fill=seg_selected, outline=None)

# 7) Separator lines between major sections for visual grouping
section_separators = [ (header_bottom+20), (cat_bottom+26), (etype_bottom+26), (lang_bottom+26), (price_bottom+26), (sort_top+sort_height+26) ]
for y in section_separators:
    draw.line([(24, y), (w-24, y)], fill=divider_color, width=1)

# 8) Bottom "Apply filters" button background and border (rounded)
# This is the main CTA background — the text and icon will be pasted on top.
apply_box = (48, 2768, w-48, 2768 + 144)
draw.rounded_rectangle(apply_box, radius=12, fill=(255,255,255), outline=panel_border, width=4)

# Add faint shadow under the apply button to lift it slightly
shadow_top = apply_box[1] + 6
draw.rounded_rectangle([(apply_box[0], shadow_top), (apply_box[2], apply_box[3]+6)], radius=12, outline=None, fill=None)
# subtle extra divider above button
draw.line([(apply_box[0], apply_box[1]-18), (apply_box[2], apply_box[1]-18)], fill=divider_color, width=1)

# 9) Small decorative accents: faint rounded borders around chip areas to help structure layout
# (These are only backgrounds/structure — the chip icons/labels will be pasted exactly on top.)
draw.rounded_rectangle([(40, cat_top+12), (w-40, cat_bottom-12)], radius=18, outline=(245,245,248), width=1)
draw.rounded_rectangle([(40, etype_top+12), (w-40, etype_bottom-12)], radius=18, outline=(245,245,248), width=1)
draw.rounded_rectangle([(40, lang_top+12), (w-40, lang_bottom-12)], radius=18, outline=(245,245,248), width=1)

# End of structural drawing. The actual icons, labels and interactive controls
# will be pasted on top at their detected positions.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_17_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-19/00_icon_Food_Drink.png
try:
    _c0 = get_crop(0, 312, 144)
    canvas.paste(_c0, (512, 383), _c0)
except Exception:
    pass
layout["Food_&_Drink"] = [512, 383, 824, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_17_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-19/01_icon_Music.png
try:
    _c1 = get_crop(1, 187, 135)
    canvas.paste(_c1, (36, 383), _c1)
except Exception:
    pass
layout["Music"] = [36, 383, 223, 518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_17_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-19/02_icon_Community.png
try:
    _c2 = get_crop(2, 294, 144)
    canvas.paste(_c2, (848, 383), _c2)
except Exception:
    pass
layout["Community"] = [848, 383, 1142, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_17_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-19/03_icon_French.png
try:
    _c3 = get_crop(3, 205, 144)
    canvas.paste(_c3, (768, 1275), _c3)
except Exception:
    pass
layout["French"] = [768, 1275, 973, 1419]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_17_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-19/04_icon_Business.png
try:
    _c4 = get_crop(4, 241, 135)
    canvas.paste(_c4, (247, 383), _c4)
except Exception:
    pass
layout["Business"] = [247, 383, 488, 518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_17_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-19/05_icon_Spanish.png
try:
    _c5 = get_crop(5, 225, 144)
    canvas.paste(_c5, (519, 1275), _c5)
except Exception:
    pass
layout["Spanish"] = [519, 1275, 744, 1419]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_17_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-19/06_icon_Expo.png
try:
    _c6 = get_crop(6, 167, 144)
    canvas.paste(_c6, (614, 829), _c6)
except Exception:
    pass
layout["Expo"] = [614, 829, 781, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_17_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-19/07_icon_Italian.png
try:
    _c7 = get_crop(7, 191, 144)
    canvas.paste(_c7, (997, 1275), _c7)
except Exception:
    pass
layout["Italian"] = [997, 1275, 1188, 1419]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_17_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-19/08_icon_Seminar.png
try:
    _c8 = get_crop(8, 232, 144)
    canvas.paste(_c8, (358, 829), _c8)
except Exception:
    pass
layout["Seminar"] = [358, 829, 590, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_17_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-19/09_icon_Arts.png
try:
    _c9 = get_crop(9, 152, 144)
    canvas.paste(_c9, (1166, 383), _c9)
except Exception:
    pass
layout["Arts"] = [1166, 383, 1318, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_17_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-19/10_icon_Convention.png
try:
    _c10 = get_crop(10, 293, 144)
    canvas.paste(_c10, (805, 829), _c10)
except Exception:
    pass
layout["Convention"] = [805, 829, 1098, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_17_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-19/11_icon_Festival.png
try:
    _c11 = get_crop(11, 219, 144)
    canvas.paste(_c11, (1122, 829), _c11)
except Exception:
    pass
layout["Festival"] = [1122, 829, 1341, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_17_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-19/12_icon_German.png
try:
    _c12 = get_crop(12, 225, 135)
    canvas.paste(_c12, (270, 1275), _c12)
except Exception:
    pass
layout["German"] = [270, 1275, 495, 1410]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_17_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-19/13_icon_English.png
try:
    _c13 = get_crop(13, 210, 135)
    canvas.paste(_c13, (36, 1275), _c13)
except Exception:
    pass
layout["English"] = [36, 1275, 246, 1410]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_17_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-19/14_icon_Conference.png
try:
    _c14 = get_crop(14, 298, 135)
    canvas.paste(_c14, (36, 829), _c14)
except Exception:
    pass
layout["Conference"] = [36, 829, 334, 964]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_17_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-19/15_icon_Date.png
try:
    _c15 = get_crop(15, 660, 144)
    canvas.paste(_c15, (726, 2024), _c15)
except Exception:
    pass
layout["Date"] = [726, 2024, 1386, 2168]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_17_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-19/16_icon_Relevance.png
try:
    _c16 = get_crop(16, 660, 144)
    canvas.paste(_c16, (54, 2024), _c16)
except Exception:
    pass
layout["Relevance"] = [54, 2024, 714, 2168]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_17_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-19/17_icon_Clear_all.png
try:
    _c17 = get_crop(17, 51, 70)
    canvas.paste(_c17, (1153, 1), _c17)
except Exception:
    pass
layout["Clear_all"] = [1153, 1, 1204, 71]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_17_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-19/18_icon_Apply_filters.png
try:
    _c18 = get_crop(18, 1344, 144)
    canvas.paste(_c18, (48, 2768), _c18)
except Exception:
    pass
layout["Apply_filters"] = [48, 2768, 1392, 2912]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_17_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-19/19_icon_Clear_all.png
try:
    _c19 = get_crop(19, 100, 70)
    canvas.paste(_c19, (1211, 0), _c19)
except Exception:
    pass
layout["Clear_all"] = [1211, 0, 1311, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_17_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-19/20_icon_7.36.png
try:
    _c20 = get_crop(20, 61, 63)
    canvas.paste(_c20, (179, 2), _c20)
except Exception:
    pass
layout["7.36"] = [179, 2, 240, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_17_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-19/21_icon_7.36.png
try:
    _c21 = get_crop(21, 144, 144)
    canvas.paste(_c21, (12, 72), _c21)
except Exception:
    pass
layout["7.36"] = [12, 72, 156, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_17_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-19/22_icon_icon_22.png
try:
    _c22 = get_crop(22, 66, 62)
    canvas.paste(_c22, (307, 3), _c22)
except Exception:
    pass
layout["icon_22"] = [307, 3, 373, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_17_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-19/23_icon_7.36.png
try:
    _c23 = get_crop(23, 65, 65)
    canvas.paste(_c23, (111, 1), _c23)
except Exception:
    pass
layout["7.36"] = [111, 1, 176, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_17_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-19/24_icon_Clear_all.png
try:
    _c24 = get_crop(24, 52, 69)
    canvas.paste(_c24, (1320, 0), _c24)
except Exception:
    pass
layout["Clear_all"] = [1320, 0, 1372, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_17_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-19/25_icon_icon_25.png
try:
    _c25 = get_crop(25, 51, 61)
    canvas.paste(_c25, (248, 3), _c25)
except Exception:
    pass
layout["icon_25"] = [248, 3, 299, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_17_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-19/26_icon_Clear_all.png
try:
    _c26 = get_crop(26, 178, 144)
    canvas.paste(_c26, (1214, 72), _c26)
except Exception:
    pass
layout["Clear_all"] = [1214, 72, 1392, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_17_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-19/27_icon_clickable_20.png
try:
    _c27 = get_crop(27, 144, 144)
    canvas.paste(_c27, (1248, 1729), _c27)
except Exception:
    pass
layout["clickable_20"] = [1248, 1729, 1392, 1873]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_17_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-19/28_text_Filters.png
try:
    _c28 = get_crop(28, 180, 66)
    canvas.paste(_c28, (631, 116), _c28)
except Exception:
    pass
layout["Filters"] = [631, 116, 811, 182]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_17_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-19/29_text_Categories.png
try:
    _c29 = get_crop(29, 187, 135)
    canvas.paste(_c29, (36, 383), _c29)
except Exception:
    pass
layout["Categories"] = [36, 383, 223, 518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_17_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-19/30_text_Show_all_categories.png
try:
    _c30 = get_crop(30, 516, 144)
    canvas.paste(_c30, (0, 518), _c30)
except Exception:
    pass
layout["Show_all_categories"] = [0, 518, 516, 662]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_17_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-19/31_text_Event_type.png
try:
    _c31 = get_crop(31, 298, 135)
    canvas.paste(_c31, (36, 829), _c31)
except Exception:
    pass
layout["Event_type"] = [36, 829, 334, 964]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_17_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-19/32_text_Show_all_event_types.png
try:
    _c32 = get_crop(32, 535, 144)
    canvas.paste(_c32, (0, 964), _c32)
except Exception:
    pass
layout["Show_all_event_types"] = [0, 964, 535, 1108]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_17_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-19/33_text_Languages.png
try:
    _c33 = get_crop(33, 210, 135)
    canvas.paste(_c33, (36, 1275), _c33)
except Exception:
    pass
layout["Languages"] = [36, 1275, 246, 1410]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_17_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-19/34_text_Show_all_languages.png
try:
    _c34 = get_crop(34, 511, 144)
    canvas.paste(_c34, (0, 1410), _c34)
except Exception:
    pass
layout["Show_all_languages"] = [0, 1410, 511, 1554]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_17_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-19/35_text_Price.png
try:
    _c35 = get_crop(35, 149, 63)
    canvas.paste(_c35, (45, 1613), _c35)
except Exception:
    pass
layout["Price"] = [45, 1613, 194, 1676]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_17_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-19/36_text_Only_free_events.png
try:
    _c36 = get_crop(36, 660, 144)
    canvas.paste(_c36, (54, 2024), _c36)
except Exception:
    pass
layout["Only_free_events"] = [54, 2024, 714, 2168]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_17_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-19/37_text_Sort_by.png
try:
    _c37 = get_crop(37, 206, 75)
    canvas.paste(_c37, (42, 1931), _c37)
except Exception:
    pass
layout["Sort_by"] = [42, 1931, 248, 2006]
