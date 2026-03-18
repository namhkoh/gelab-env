# page_id: page_eventbrite_3ce6196f48694f74bf7d05dc71840c63_05
# screenshot: 2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-7.png
# step_index: 5/9
# task: Open Eventbrite. Search for 'coding workshop'. Sort the results by date. Where is the location of the soonest event that is not promoted?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw background and structural UI elements for the provided canvas (1440x2960)
# Uses existing variables: canvas (PIL Image), draw (PIL.ImageDraw)

# Colors
bg_color = (250, 250, 252)        # very light off-white page background
status_bar_color = (205, 205, 205)  # light grey status bar
header_bg = (255, 255, 255)       # header white
divider_color = (230, 228, 235)   # soft grey divider
seg_bg = (243, 242, 246)          # segmented control background
seg_inactive = (236, 234, 239)    # inactive segment fill
seg_active = (255, 255, 255)      # active segment fill
btn_border = (197, 193, 201)      # apply button border
shadow = (220, 217, 224)          # subtle shadow

W, H = canvas.size

# 1) Full page background
draw.rectangle([0, 0, W, H], fill=bg_color)

# 2) Status bar at top (approx)
status_h = 96
draw.rectangle([0, 0, W, status_h], fill=status_bar_color)

# subtle top highlight and bottom shadow for status area
draw.line([(0, status_h-1), (W, status_h-1)], fill=(200,200,200))

# 3) Header / toolbar area below status bar
header_top = status_h
header_h = 120
header_bottom = header_top + header_h
draw.rectangle([0, header_top, W, header_bottom], fill=header_bg)

# bottom divider under header
draw.line([(24, header_bottom), (W-24, header_bottom)], fill=divider_color, width=1)

# 4) Section separators (thin dividers between content groups)
# Positions chosen to align with the typical layout spacing from the screenshot
separators_y = [520, 964, 1410, 1688]  # after categories, after event types, after languages, price area
for y in separators_y:
    draw.line([(24, y), (W-24, y)], fill=divider_color, width=1)

# 5) Subtle section group backgrounds (very faint rounded band behind group areas)
# Categories block band
draw.rounded_rectangle([18, 320, W-18, 540], radius=12, fill=(255,255,255,0), outline=None)
# Event type block band
draw.rounded_rectangle([18, 760, W-18, 980], radius=12, fill=(255,255,255,0), outline=None)
# Languages block band
draw.rounded_rectangle([18, 1208, W-18, 1438], radius=12, fill=(255,255,255,0), outline=None)

# 6) "Sort by" segmented control background
seg_x0 = 36
seg_y0 = 1996
seg_w = W - seg_x0*2
seg_h = 144
seg_radius = 14

# Draw subtle shadow below segmented control
draw.rounded_rectangle([seg_x0+2, seg_y0+6, seg_x0+seg_w+2, seg_y0+6+seg_h], radius=seg_radius, fill=shadow)

# Segmented control outer background
draw.rounded_rectangle([seg_x0, seg_y0, seg_x0+seg_w, seg_y0+seg_h], radius=seg_radius, fill=seg_bg, outline=divider_color)

# Inner segments: two halves (left=Relevance, right=Date)
half_w = seg_w // 2
left_rect = [seg_x0+6, seg_y0+6, seg_x0+half_w-6, seg_y0+seg_h-6]
right_rect = [seg_x0+half_w+6, seg_y0+6, seg_x0+seg_w-6, seg_y0+seg_h-6]

# Left segment (selected) - lighter
draw.rounded_rectangle(left_rect, radius=12, fill=seg_active, outline=None)
# Right segment (inactive) - slightly darker
draw.rounded_rectangle(right_rect, radius=12, fill=seg_inactive, outline=None)

# Add subtle inner separators (vertical)
sep_x = seg_x0 + half_w
draw.line([(sep_x, seg_y0+8), (sep_x, seg_y0+seg_h-8)], fill=divider_color)

# 7) Bottom "Apply filters" button background and border
btn_x0 = 48
btn_y0 = 2768
btn_w = 1344
btn_h = 144
btn_radius = 12

# Draw subtle shadow for button
draw.rounded_rectangle([btn_x0+2, btn_y0+6, btn_x0+btn_w+2, btn_y0+btn_h+6], radius=btn_radius, fill=shadow)

# Button background (white) with border
draw.rounded_rectangle([btn_x0, btn_y0, btn_x0+btn_w, btn_y0+btn_h], radius=btn_radius, fill=(255,255,255), outline=btn_border, width=4)

# 8) Page bottom padding line (subtle)
draw.line([(24, H-220), (W-24, H-220)], fill=divider_color, width=1)

# 9) Additional subtle horizontal guides to visually separate "Price" and "Only free events" area
draw.line([(36, 1628), (W-36, 1628)], fill=divider_color)
draw.line([(36, 1860), (W-36, 1860)], fill=(245,245,247))

# Note: Text, icons and the pill/tag buttons will be overlaid later at their exact detected positions.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_05_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-7/00_icon_Food_Drink.png
try:
    _c0 = get_crop(0, 312, 144)
    canvas.paste(_c0, (512, 383), _c0)
except Exception:
    pass
layout["Food_&_Drink"] = [512, 383, 824, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_05_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-7/01_icon_Music.png
try:
    _c1 = get_crop(1, 187, 135)
    canvas.paste(_c1, (36, 383), _c1)
except Exception:
    pass
layout["Music"] = [36, 383, 223, 518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_05_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-7/02_icon_Community.png
try:
    _c2 = get_crop(2, 294, 144)
    canvas.paste(_c2, (848, 383), _c2)
except Exception:
    pass
layout["Community"] = [848, 383, 1142, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_05_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-7/03_icon_French.png
try:
    _c3 = get_crop(3, 205, 144)
    canvas.paste(_c3, (768, 1275), _c3)
except Exception:
    pass
layout["French"] = [768, 1275, 973, 1419]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_05_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-7/04_icon_Business.png
try:
    _c4 = get_crop(4, 241, 135)
    canvas.paste(_c4, (247, 383), _c4)
except Exception:
    pass
layout["Business"] = [247, 383, 488, 518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_05_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-7/05_icon_Spanish.png
try:
    _c5 = get_crop(5, 225, 144)
    canvas.paste(_c5, (519, 1275), _c5)
except Exception:
    pass
layout["Spanish"] = [519, 1275, 744, 1419]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_05_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-7/06_icon_Expo.png
try:
    _c6 = get_crop(6, 167, 144)
    canvas.paste(_c6, (614, 829), _c6)
except Exception:
    pass
layout["Expo"] = [614, 829, 781, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_05_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-7/07_icon_Italian.png
try:
    _c7 = get_crop(7, 191, 144)
    canvas.paste(_c7, (997, 1275), _c7)
except Exception:
    pass
layout["Italian"] = [997, 1275, 1188, 1419]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_05_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-7/08_icon_Arts.png
try:
    _c8 = get_crop(8, 152, 144)
    canvas.paste(_c8, (1166, 383), _c8)
except Exception:
    pass
layout["Arts"] = [1166, 383, 1318, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_05_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-7/09_icon_Seminar.png
try:
    _c9 = get_crop(9, 232, 144)
    canvas.paste(_c9, (358, 829), _c9)
except Exception:
    pass
layout["Seminar"] = [358, 829, 590, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_05_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-7/10_icon_Convention.png
try:
    _c10 = get_crop(10, 293, 144)
    canvas.paste(_c10, (805, 829), _c10)
except Exception:
    pass
layout["Convention"] = [805, 829, 1098, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_05_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-7/11_icon_Festival.png
try:
    _c11 = get_crop(11, 219, 144)
    canvas.paste(_c11, (1122, 829), _c11)
except Exception:
    pass
layout["Festival"] = [1122, 829, 1341, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_05_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-7/12_icon_German.png
try:
    _c12 = get_crop(12, 225, 135)
    canvas.paste(_c12, (270, 1275), _c12)
except Exception:
    pass
layout["German"] = [270, 1275, 495, 1410]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_05_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-7/13_icon_English.png
try:
    _c13 = get_crop(13, 210, 135)
    canvas.paste(_c13, (36, 1275), _c13)
except Exception:
    pass
layout["English"] = [36, 1275, 246, 1410]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_05_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-7/14_icon_Conference.png
try:
    _c14 = get_crop(14, 298, 135)
    canvas.paste(_c14, (36, 829), _c14)
except Exception:
    pass
layout["Conference"] = [36, 829, 334, 964]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_05_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-7/15_icon_Date.png
try:
    _c15 = get_crop(15, 660, 144)
    canvas.paste(_c15, (726, 2024), _c15)
except Exception:
    pass
layout["Date"] = [726, 2024, 1386, 2168]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_05_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-7/16_icon_Relevance.png
try:
    _c16 = get_crop(16, 660, 144)
    canvas.paste(_c16, (54, 2024), _c16)
except Exception:
    pass
layout["Relevance"] = [54, 2024, 714, 2168]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_05_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-7/17_icon_Apply_filters.png
try:
    _c17 = get_crop(17, 1344, 144)
    canvas.paste(_c17, (48, 2768), _c17)
except Exception:
    pass
layout["Apply_filters"] = [48, 2768, 1392, 2912]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_05_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-7/18_icon_7.24.png
try:
    _c18 = get_crop(18, 144, 144)
    canvas.paste(_c18, (12, 72), _c18)
except Exception:
    pass
layout["7.24"] = [12, 72, 156, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_05_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-7/19_icon_7.24.png
try:
    _c19 = get_crop(19, 59, 64)
    canvas.paste(_c19, (181, 1), _c19)
except Exception:
    pass
layout["7.24"] = [181, 1, 240, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_05_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-7/20_icon_7.24.png
try:
    _c20 = get_crop(20, 65, 65)
    canvas.paste(_c20, (111, 1), _c20)
except Exception:
    pass
layout["7.24"] = [111, 1, 176, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_05_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-7/21_icon_icon_21.png
try:
    _c21 = get_crop(21, 65, 62)
    canvas.paste(_c21, (308, 3), _c21)
except Exception:
    pass
layout["icon_21"] = [308, 3, 373, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_05_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-7/22_icon_Clear_all.png
try:
    _c22 = get_crop(22, 55, 69)
    canvas.paste(_c22, (1319, 0), _c22)
except Exception:
    pass
layout["Clear_all"] = [1319, 0, 1374, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_05_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-7/23_icon_icon_23.png
try:
    _c23 = get_crop(23, 51, 62)
    canvas.paste(_c23, (248, 2), _c23)
except Exception:
    pass
layout["icon_23"] = [248, 2, 299, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_05_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-7/24_icon_Clear_all.png
try:
    _c24 = get_crop(24, 99, 70)
    canvas.paste(_c24, (1211, 0), _c24)
except Exception:
    pass
layout["Clear_all"] = [1211, 0, 1310, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_05_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-7/25_icon_clickable_20.png
try:
    _c25 = get_crop(25, 144, 144)
    canvas.paste(_c25, (1248, 1729), _c25)
except Exception:
    pass
layout["clickable_20"] = [1248, 1729, 1392, 1873]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_05_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-7/26_icon_Clear_all.png
try:
    _c26 = get_crop(26, 178, 144)
    canvas.paste(_c26, (1214, 72), _c26)
except Exception:
    pass
layout["Clear_all"] = [1214, 72, 1392, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_05_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-7/27_icon_7.24.png
try:
    _c27 = get_crop(27, 102, 65)
    canvas.paste(_c27, (8, 0), _c27)
except Exception:
    pass
layout["7.24"] = [8, 0, 110, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_05_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-7/28_text_Filters.png
try:
    _c28 = get_crop(28, 180, 66)
    canvas.paste(_c28, (631, 116), _c28)
except Exception:
    pass
layout["Filters"] = [631, 116, 811, 182]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_05_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-7/29_text_Categories.png
try:
    _c29 = get_crop(29, 187, 135)
    canvas.paste(_c29, (36, 383), _c29)
except Exception:
    pass
layout["Categories"] = [36, 383, 223, 518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_05_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-7/30_text_Show_all_categories.png
try:
    _c30 = get_crop(30, 516, 144)
    canvas.paste(_c30, (0, 518), _c30)
except Exception:
    pass
layout["Show_all_categories"] = [0, 518, 516, 662]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_05_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-7/31_text_Event_type.png
try:
    _c31 = get_crop(31, 298, 135)
    canvas.paste(_c31, (36, 829), _c31)
except Exception:
    pass
layout["Event_type"] = [36, 829, 334, 964]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_05_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-7/32_text_Show_all_event_types.png
try:
    _c32 = get_crop(32, 535, 144)
    canvas.paste(_c32, (0, 964), _c32)
except Exception:
    pass
layout["Show_all_event_types"] = [0, 964, 535, 1108]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_05_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-7/33_text_Languages.png
try:
    _c33 = get_crop(33, 210, 135)
    canvas.paste(_c33, (36, 1275), _c33)
except Exception:
    pass
layout["Languages"] = [36, 1275, 246, 1410]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_05_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-7/34_text_Show_all_languages.png
try:
    _c34 = get_crop(34, 511, 144)
    canvas.paste(_c34, (0, 1410), _c34)
except Exception:
    pass
layout["Show_all_languages"] = [0, 1410, 511, 1554]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_05_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-7/35_text_Price.png
try:
    _c35 = get_crop(35, 149, 63)
    canvas.paste(_c35, (45, 1613), _c35)
except Exception:
    pass
layout["Price"] = [45, 1613, 194, 1676]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_05_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-7/36_text_Only_free_events.png
try:
    _c36 = get_crop(36, 660, 144)
    canvas.paste(_c36, (54, 2024), _c36)
except Exception:
    pass
layout["Only_free_events"] = [54, 2024, 714, 2168]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_05_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-7/37_text_Sort_by.png
try:
    _c37 = get_crop(37, 206, 75)
    canvas.paste(_c37, (42, 1931), _c37)
except Exception:
    pass
layout["Sort_by"] = [42, 1931, 248, 2006]
