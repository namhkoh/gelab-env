# page_id: page_eventbrite_ee1eef38e6e94342b57a272493366950_03
# screenshot: 2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-5.png
# step_index: 3/10
# task: Open Eventbrite. Open "Fashion" category. Apply filter for free events. From the list, select the first non-promoted event and add it to your favorites.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw UI background and structural elements for the filters page
# (Uses provided variables: canvas (Image), draw (ImageDraw), font_sm, font_md, font_lg, font_xl)

W, H = canvas.size

# Colors
status_bar_color = "#CFCFCF"
divider_light = "#E9E9EE"
panel_bg = "#FBFCFE"
panel_border = "#E6E2E8"
seg_outer_fill = "#F5F3F7"
seg_outer_outline = "#D9D6DE"
apply_fill = "#FFFFFF"
apply_outline = "#BFB9C2"
shadow_color = (0, 0, 0, 24)  # translucent shadow (we'll emulate soft shadow with rectangles)
page_bg = "#FFFFFF"

# Ensure page background is solid white
draw.rectangle([0, 0, W, H], fill=page_bg)

# 1) Status bar area at very top (~0..96)
status_h = 96
draw.rectangle([0, 0, W, status_h], fill=status_bar_color)

# 2) Header / toolbar area (below status bar)
header_top = status_h
header_bottom = status_h + 96
draw.rectangle([0, header_top, W, header_bottom], fill=page_bg)
# subtle bottom divider under header
draw.line([24, header_bottom, W-24, header_bottom], fill=divider_light, width=2)

# 3) Section "cards" / group backgrounds (rounded rectangles behind groups of chips)
# Categories group (around y ~ 300..560)
cat_top = 300
cat_bottom = 560
cat_left = 36
cat_right = W - 36
draw.rounded_rectangle([cat_left, cat_top, cat_right, cat_bottom],
                       radius=20, fill=panel_bg, outline=panel_border, width=1)

# Event type group (around y ~ 760..980)
evt_top = 760
evt_bottom = 980
draw.rounded_rectangle([cat_left, evt_top, cat_right, evt_bottom],
                       radius=20, fill=panel_bg, outline=panel_border, width=1)

# Languages group (around y ~ 1190..1420)
lang_top = 1190
lang_bottom = 1420
draw.rounded_rectangle([cat_left, lang_top, cat_right, lang_bottom],
                       radius=20, fill=panel_bg, outline=panel_border, width=1)

# Price small group area (around y ~1560..1700) - subtle container for the Price label and toggle
price_top = 1560
price_bottom = 1710
price_left = 36
price_right = W - 36
draw.rounded_rectangle([price_left, price_top, price_right, price_bottom],
                       radius=16, fill=page_bg, outline=divider_light, width=1)

# 4) Separator lines between major sections to support visual grouping
sep_x1 = 36
sep_x2 = W - 36
# between categories and event type
draw.line([sep_x1, (cat_bottom + evt_top)//2, sep_x2, (cat_bottom + evt_top)//2], fill=divider_light, width=1)
# between event type and languages
draw.line([sep_x1, (evt_bottom + lang_top)//2, sep_x2, (evt_bottom + lang_top)//2], fill=divider_light, width=1)
# between languages and price
draw.line([sep_x1, (lang_bottom + price_top)//2, sep_x2, (lang_bottom + price_top)//2], fill=divider_light, width=1)

# 5) Sort by segmented control outer container (draw a subtle rounded container behind the two segments)
# The detected inner segments occupy roughly x=54..726 and x=726..1386 at y~2024..2168.
seg_outer_top = 1988
seg_outer_bottom = 2168
seg_outer_left = 36
seg_outer_right = W - 36
draw.rounded_rectangle([seg_outer_left, seg_outer_top, seg_outer_right, seg_outer_bottom],
                       radius=32, fill=seg_outer_fill, outline=seg_outer_outline, width=1)

# Add a faint inner shadow at the top of the segmented control
for i, offset in enumerate(range(0, 6)):
    alpha = int(12 - i*2)
    if alpha <= 0:
        break
    shade = (0, 0, 0, alpha)
    y = seg_outer_top + offset
    draw.line([seg_outer_left+6, y, seg_outer_right-6, y], fill=("#000000" if alpha>0 else "#000000"))

# 6) Large bottom "Apply filters" bar background (rounded rectangle with border)
apply_left = 48
apply_top = 2768
apply_right = W - 48
apply_bottom = apply_top + 144
draw.rounded_rectangle([apply_left, apply_top, apply_right, apply_bottom],
                       radius=18, fill=apply_fill, outline=apply_outline, width=4)

# subtle drop shadow under the apply bar (soft, by a few translucent rectangles)
shadow_y_start = apply_bottom
for i, dy in enumerate((2,5,9)):
    alpha = 24 - i*6
    if alpha <= 0:
        break
    shade_color = (0,0,0,alpha)
    # PIL ImageDraw doesn't support alpha directly on draw primitives, emulate with slightly darker lines
    draw.line([apply_left+6, shadow_y_start+dy, apply_right-6, shadow_y_start+dy], fill="#E6E6E6", width=1)

# 7) Minor decorative separators to emphasize groups (thin dotted/solid lines)
# Subtle horizontal guides near section titles (no text drawn)
draw.line([36, 240, W-36, 240], fill=divider_light, width=1)
draw.line([36, 700, W-36, 700], fill=divider_light, width=1)
draw.line([36, 1100, W-36, 1100], fill=divider_light, width=1)

# 8) Top-left small underline under status to separate from header (very subtle)
draw.line([0, status_h-1, W, status_h-1], fill="#CFCFCF", width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_03_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-5/00_icon_Food_Drink.png
try:
    _c0 = get_crop(0, 312, 144)
    canvas.paste(_c0, (512, 383), _c0)
except Exception:
    pass
layout["Food_&_Drink"] = [512, 383, 824, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_03_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-5/01_icon_Music.png
try:
    _c1 = get_crop(1, 187, 135)
    canvas.paste(_c1, (36, 383), _c1)
except Exception:
    pass
layout["Music"] = [36, 383, 223, 518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_03_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-5/02_icon_Community.png
try:
    _c2 = get_crop(2, 294, 144)
    canvas.paste(_c2, (848, 383), _c2)
except Exception:
    pass
layout["Community"] = [848, 383, 1142, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_03_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-5/03_icon_French.png
try:
    _c3 = get_crop(3, 205, 144)
    canvas.paste(_c3, (768, 1275), _c3)
except Exception:
    pass
layout["French"] = [768, 1275, 973, 1419]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_03_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-5/04_icon_Business.png
try:
    _c4 = get_crop(4, 241, 135)
    canvas.paste(_c4, (247, 383), _c4)
except Exception:
    pass
layout["Business"] = [247, 383, 488, 518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_03_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-5/05_icon_Spanish.png
try:
    _c5 = get_crop(5, 225, 144)
    canvas.paste(_c5, (519, 1275), _c5)
except Exception:
    pass
layout["Spanish"] = [519, 1275, 744, 1419]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_03_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-5/06_icon_Expo.png
try:
    _c6 = get_crop(6, 167, 144)
    canvas.paste(_c6, (614, 829), _c6)
except Exception:
    pass
layout["Expo"] = [614, 829, 781, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_03_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-5/07_icon_Italian.png
try:
    _c7 = get_crop(7, 191, 144)
    canvas.paste(_c7, (997, 1275), _c7)
except Exception:
    pass
layout["Italian"] = [997, 1275, 1188, 1419]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_03_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-5/08_icon_Arts.png
try:
    _c8 = get_crop(8, 152, 144)
    canvas.paste(_c8, (1166, 383), _c8)
except Exception:
    pass
layout["Arts"] = [1166, 383, 1318, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_03_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-5/09_icon_Seminar.png
try:
    _c9 = get_crop(9, 232, 144)
    canvas.paste(_c9, (358, 829), _c9)
except Exception:
    pass
layout["Seminar"] = [358, 829, 590, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_03_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-5/10_icon_Convention.png
try:
    _c10 = get_crop(10, 293, 144)
    canvas.paste(_c10, (805, 829), _c10)
except Exception:
    pass
layout["Convention"] = [805, 829, 1098, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_03_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-5/11_icon_Festival.png
try:
    _c11 = get_crop(11, 219, 144)
    canvas.paste(_c11, (1122, 829), _c11)
except Exception:
    pass
layout["Festival"] = [1122, 829, 1341, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_03_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-5/12_icon_German.png
try:
    _c12 = get_crop(12, 225, 135)
    canvas.paste(_c12, (270, 1275), _c12)
except Exception:
    pass
layout["German"] = [270, 1275, 495, 1410]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_03_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-5/13_icon_English.png
try:
    _c13 = get_crop(13, 210, 135)
    canvas.paste(_c13, (36, 1275), _c13)
except Exception:
    pass
layout["English"] = [36, 1275, 246, 1410]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_03_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-5/14_icon_Conference.png
try:
    _c14 = get_crop(14, 298, 135)
    canvas.paste(_c14, (36, 829), _c14)
except Exception:
    pass
layout["Conference"] = [36, 829, 334, 964]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_03_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-5/15_icon_Date.png
try:
    _c15 = get_crop(15, 660, 144)
    canvas.paste(_c15, (726, 2024), _c15)
except Exception:
    pass
layout["Date"] = [726, 2024, 1386, 2168]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_03_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-5/16_icon_Relevance.png
try:
    _c16 = get_crop(16, 660, 144)
    canvas.paste(_c16, (54, 2024), _c16)
except Exception:
    pass
layout["Relevance"] = [54, 2024, 714, 2168]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_03_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-5/17_icon_Apply_filters.png
try:
    _c17 = get_crop(17, 1344, 144)
    canvas.paste(_c17, (48, 2768), _c17)
except Exception:
    pass
layout["Apply_filters"] = [48, 2768, 1392, 2912]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_03_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-5/18_icon_5.27.png
try:
    _c18 = get_crop(18, 61, 64)
    canvas.paste(_c18, (179, 2), _c18)
except Exception:
    pass
layout["5.27"] = [179, 2, 240, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_03_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-5/19_icon_5.27.png
try:
    _c19 = get_crop(19, 144, 144)
    canvas.paste(_c19, (12, 72), _c19)
except Exception:
    pass
layout["5.27"] = [12, 72, 156, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_03_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-5/20_icon_5.27.png
try:
    _c20 = get_crop(20, 66, 65)
    canvas.paste(_c20, (110, 1), _c20)
except Exception:
    pass
layout["5.27"] = [110, 1, 176, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_03_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-5/21_icon_icon_21.png
try:
    _c21 = get_crop(21, 66, 62)
    canvas.paste(_c21, (307, 3), _c21)
except Exception:
    pass
layout["icon_21"] = [307, 3, 373, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_03_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-5/22_icon_Clear_all.png
try:
    _c22 = get_crop(22, 55, 69)
    canvas.paste(_c22, (1319, 0), _c22)
except Exception:
    pass
layout["Clear_all"] = [1319, 0, 1374, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_03_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-5/23_icon_icon_23.png
try:
    _c23 = get_crop(23, 51, 62)
    canvas.paste(_c23, (248, 2), _c23)
except Exception:
    pass
layout["icon_23"] = [248, 2, 299, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_03_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-5/24_icon_Clear_all.png
try:
    _c24 = get_crop(24, 99, 70)
    canvas.paste(_c24, (1211, 0), _c24)
except Exception:
    pass
layout["Clear_all"] = [1211, 0, 1310, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_03_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-5/25_icon_clickable_20.png
try:
    _c25 = get_crop(25, 144, 144)
    canvas.paste(_c25, (1248, 1729), _c25)
except Exception:
    pass
layout["clickable_20"] = [1248, 1729, 1392, 1873]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_03_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-5/26_icon_Clear_all.png
try:
    _c26 = get_crop(26, 178, 144)
    canvas.paste(_c26, (1214, 72), _c26)
except Exception:
    pass
layout["Clear_all"] = [1214, 72, 1392, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_03_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-5/27_text_5.27.png
try:
    _c27 = get_crop(27, 87, 43)
    canvas.paste(_c27, (22, 17), _c27)
except Exception:
    pass
layout["5.27"] = [22, 17, 109, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_03_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-5/28_text_Filters.png
try:
    _c28 = get_crop(28, 180, 66)
    canvas.paste(_c28, (631, 116), _c28)
except Exception:
    pass
layout["Filters"] = [631, 116, 811, 182]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_03_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-5/29_text_Categories.png
try:
    _c29 = get_crop(29, 187, 135)
    canvas.paste(_c29, (36, 383), _c29)
except Exception:
    pass
layout["Categories"] = [36, 383, 223, 518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_03_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-5/30_text_Show_all_categories.png
try:
    _c30 = get_crop(30, 516, 144)
    canvas.paste(_c30, (0, 518), _c30)
except Exception:
    pass
layout["Show_all_categories"] = [0, 518, 516, 662]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_03_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-5/31_text_Event_type.png
try:
    _c31 = get_crop(31, 298, 135)
    canvas.paste(_c31, (36, 829), _c31)
except Exception:
    pass
layout["Event_type"] = [36, 829, 334, 964]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_03_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-5/32_text_Show_all_event_types.png
try:
    _c32 = get_crop(32, 535, 144)
    canvas.paste(_c32, (0, 964), _c32)
except Exception:
    pass
layout["Show_all_event_types"] = [0, 964, 535, 1108]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_03_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-5/33_text_Languages.png
try:
    _c33 = get_crop(33, 210, 135)
    canvas.paste(_c33, (36, 1275), _c33)
except Exception:
    pass
layout["Languages"] = [36, 1275, 246, 1410]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_03_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-5/34_text_Show_all_languages.png
try:
    _c34 = get_crop(34, 511, 144)
    canvas.paste(_c34, (0, 1410), _c34)
except Exception:
    pass
layout["Show_all_languages"] = [0, 1410, 511, 1554]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_03_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-5/35_text_Price.png
try:
    _c35 = get_crop(35, 149, 63)
    canvas.paste(_c35, (45, 1613), _c35)
except Exception:
    pass
layout["Price"] = [45, 1613, 194, 1676]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_03_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-5/36_text_Only_free_events.png
try:
    _c36 = get_crop(36, 660, 144)
    canvas.paste(_c36, (54, 2024), _c36)
except Exception:
    pass
layout["Only_free_events"] = [54, 2024, 714, 2168]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_03_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-5/37_text_Sort_by.png
try:
    _c37 = get_crop(37, 206, 75)
    canvas.paste(_c37, (42, 1931), _c37)
except Exception:
    pass
layout["Sort_by"] = [42, 1931, 248, 2006]
