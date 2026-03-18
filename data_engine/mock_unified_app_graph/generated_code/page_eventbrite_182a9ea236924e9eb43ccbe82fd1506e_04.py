# page_id: page_eventbrite_182a9ea236924e9eb43ccbe82fd1506e_04
# screenshot: 2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-6.png
# step_index: 4/13
# task: Open Eventbrite. Set time to tomorrow. Clear all search filters. Select the third one in New York. Record its location and time in Google Keep Notes. Add it to Favorites.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw page background and structural UI elements for the provided canvas/draw objects.
# Assumes: canvas is a PIL.Image (1440x2960 RGB) and draw is an ImageDraw.Draw(canvas)

W, H = canvas.size

# Colors (chosen to match screenshot subtle tones)
bg_color = (255, 255, 255)            # main white background
status_bar_color = (200, 200, 200)    # light gray status bar
header_bg = (255, 255, 255)           # header white
divider = (230, 230, 238)             # soft divider line
section_bg = (250, 251, 255)          # very subtle off-white for section groups
section_border = (235, 233, 242)      # subtle border for section cards
shadow_color = (243, 241, 247)        # faint shadow for elevated elements
soft_separator = (245, 246, 249)      # soft horizontal separators

# Utility values
status_h = 96
header_h = 176
pad = 36

# 1) Fill background
draw.rectangle([(0, 0), (W, H)], fill=bg_color)

# 2) Status bar area (top ~96px)
draw.rectangle([(0, 0), (W, status_h)], fill=status_bar_color)

# 3) Header / toolbar background (below status bar)
draw.rectangle([(0, status_h), (W, header_h)], fill=header_bg)
# subtle bottom divider and shadow under header
draw.line([(0, header_h), (W, header_h)], fill=divider, width=1)
draw.line([(0, header_h+2), (W, header_h+2)], fill=shadow_color, width=1)

# 4) Section group rounded cards (do not draw individual chips/text/icons)
# Categories group (area that contains the category chips)
cat_top = 240
cat_bottom = 520
draw.rounded_rectangle([(pad, cat_top), (W - pad, cat_bottom)],
                       radius=18, fill=section_bg, outline=section_border, width=1)

# Event type group
etype_top = 640
etype_bottom = 980
draw.rounded_rectangle([(pad, etype_top), (W - pad, etype_bottom)],
                       radius=18, fill=section_bg, outline=section_border, width=1)

# Languages group
lang_top = 1080
lang_bottom = 1420
draw.rounded_rectangle([(pad, lang_top), (W - pad, lang_bottom)],
                       radius=18, fill=section_bg, outline=section_border, width=1)

# Price & toggle area (smaller card)
price_top = 1480
price_bottom = 1740
draw.rounded_rectangle([(pad, price_top), (W - pad, price_bottom)],
                       radius=14, fill=bg_color, outline=section_border, width=1)

# Sort-by area container (background behind segmented control)
sort_top = 1840
sort_bottom = 2100
draw.rounded_rectangle([(pad, sort_top), (W - pad, sort_bottom)],
                       radius=16, fill=section_bg, outline=section_border, width=1)

# 5) Horizontal soft separators between major sections (light)
separator_ys = [cat_top - 24, etype_top - 24, lang_top - 24, price_top - 24, sort_top - 24, 2400]
for y in separator_ys:
    draw.line([(pad, y), (W - pad, y)], fill=soft_separator, width=1)

# 6) Subtle top shadow for the bottom "Apply filters" area (leave actual button area clear)
apply_shadow_top = 2680
apply_shadow_bottom = 2764
draw.rectangle([(pad, apply_shadow_top), (W - pad, apply_shadow_bottom)], fill=shadow_color)

# 7) Bottom safe area (leave space for the bottom button which will be pasted on top)
bottom_safe_top = 2768
draw.rectangle([(0, bottom_safe_top), (W, H)], fill=bg_color)

# 8) Left alignment guide (visual structural element only, very faint)
draw.line([(pad, header_h + 8), (pad, H - pad)], fill=(245, 245, 247), width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_04_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-6/00_icon_Food_Drink.png
try:
    _c0 = get_crop(0, 312, 144)
    canvas.paste(_c0, (512, 383), _c0)
except Exception:
    pass
layout["Food_&_Drink"] = [512, 383, 824, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_04_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-6/01_icon_Music.png
try:
    _c1 = get_crop(1, 187, 135)
    canvas.paste(_c1, (36, 383), _c1)
except Exception:
    pass
layout["Music"] = [36, 383, 223, 518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_04_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-6/02_icon_Community.png
try:
    _c2 = get_crop(2, 294, 144)
    canvas.paste(_c2, (848, 383), _c2)
except Exception:
    pass
layout["Community"] = [848, 383, 1142, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_04_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-6/03_icon_Spanish.png
try:
    _c3 = get_crop(3, 225, 144)
    canvas.paste(_c3, (519, 1275), _c3)
except Exception:
    pass
layout["Spanish"] = [519, 1275, 744, 1419]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_04_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-6/04_icon_French.png
try:
    _c4 = get_crop(4, 205, 144)
    canvas.paste(_c4, (768, 1275), _c4)
except Exception:
    pass
layout["French"] = [768, 1275, 973, 1419]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_04_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-6/05_icon_Business.png
try:
    _c5 = get_crop(5, 241, 135)
    canvas.paste(_c5, (247, 383), _c5)
except Exception:
    pass
layout["Business"] = [247, 383, 488, 518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_04_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-6/06_icon_Expo.png
try:
    _c6 = get_crop(6, 167, 144)
    canvas.paste(_c6, (614, 829), _c6)
except Exception:
    pass
layout["Expo"] = [614, 829, 781, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_04_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-6/07_icon_Seminar.png
try:
    _c7 = get_crop(7, 232, 144)
    canvas.paste(_c7, (358, 829), _c7)
except Exception:
    pass
layout["Seminar"] = [358, 829, 590, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_04_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-6/08_icon_Italian.png
try:
    _c8 = get_crop(8, 191, 144)
    canvas.paste(_c8, (997, 1275), _c8)
except Exception:
    pass
layout["Italian"] = [997, 1275, 1188, 1419]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_04_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-6/09_icon_Arts.png
try:
    _c9 = get_crop(9, 152, 144)
    canvas.paste(_c9, (1166, 383), _c9)
except Exception:
    pass
layout["Arts"] = [1166, 383, 1318, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_04_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-6/10_icon_Convention.png
try:
    _c10 = get_crop(10, 293, 144)
    canvas.paste(_c10, (805, 829), _c10)
except Exception:
    pass
layout["Convention"] = [805, 829, 1098, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_04_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-6/11_icon_German.png
try:
    _c11 = get_crop(11, 225, 135)
    canvas.paste(_c11, (270, 1275), _c11)
except Exception:
    pass
layout["German"] = [270, 1275, 495, 1410]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_04_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-6/12_icon_Festival.png
try:
    _c12 = get_crop(12, 219, 144)
    canvas.paste(_c12, (1122, 829), _c12)
except Exception:
    pass
layout["Festival"] = [1122, 829, 1341, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_04_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-6/13_icon_English.png
try:
    _c13 = get_crop(13, 210, 135)
    canvas.paste(_c13, (36, 1275), _c13)
except Exception:
    pass
layout["English"] = [36, 1275, 246, 1410]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_04_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-6/14_icon_Conference.png
try:
    _c14 = get_crop(14, 298, 135)
    canvas.paste(_c14, (36, 829), _c14)
except Exception:
    pass
layout["Conference"] = [36, 829, 334, 964]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_04_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-6/15_icon_Apply_filters_1.png
try:
    _c15 = get_crop(15, 1344, 144)
    canvas.paste(_c15, (48, 2768), _c15)
except Exception:
    pass
layout["Apply_filters_(1)"] = [48, 2768, 1392, 2912]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_04_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-6/16_icon_Date.png
try:
    _c16 = get_crop(16, 660, 144)
    canvas.paste(_c16, (726, 2024), _c16)
except Exception:
    pass
layout["Date"] = [726, 2024, 1386, 2168]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_04_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-6/17_icon_Relevance.png
try:
    _c17 = get_crop(17, 660, 144)
    canvas.paste(_c17, (54, 2024), _c17)
except Exception:
    pass
layout["Relevance"] = [54, 2024, 714, 2168]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_04_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-6/18_icon_9.31.png
try:
    _c18 = get_crop(18, 65, 63)
    canvas.paste(_c18, (176, 2), _c18)
except Exception:
    pass
layout["9.31"] = [176, 2, 241, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_04_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-6/19_icon_9.31.png
try:
    _c19 = get_crop(19, 144, 144)
    canvas.paste(_c19, (12, 72), _c19)
except Exception:
    pass
layout["9.31"] = [12, 72, 156, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_04_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-6/20_icon_9.31.png
try:
    _c20 = get_crop(20, 61, 65)
    canvas.paste(_c20, (110, 1), _c20)
except Exception:
    pass
layout["9.31"] = [110, 1, 171, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_04_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-6/21_icon_Clear_all.png
try:
    _c21 = get_crop(21, 98, 65)
    canvas.paste(_c21, (1212, 0), _c21)
except Exception:
    pass
layout["Clear_all"] = [1212, 0, 1310, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_04_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-6/22_icon_Clear_all.png
try:
    _c22 = get_crop(22, 56, 67)
    canvas.paste(_c22, (1317, 0), _c22)
except Exception:
    pass
layout["Clear_all"] = [1317, 0, 1373, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_04_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-6/23_icon_icon_23.png
try:
    _c23 = get_crop(23, 59, 63)
    canvas.paste(_c23, (245, 2), _c23)
except Exception:
    pass
layout["icon_23"] = [245, 2, 304, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_04_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-6/24_icon_icon_24.png
try:
    _c24 = get_crop(24, 54, 61)
    canvas.paste(_c24, (314, 3), _c24)
except Exception:
    pass
layout["icon_24"] = [314, 3, 368, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_04_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-6/25_icon_clickable_20.png
try:
    _c25 = get_crop(25, 144, 144)
    canvas.paste(_c25, (1248, 1729), _c25)
except Exception:
    pass
layout["clickable_20"] = [1248, 1729, 1392, 1873]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_04_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-6/26_icon_Clear_all.png
try:
    _c26 = get_crop(26, 178, 144)
    canvas.paste(_c26, (1214, 72), _c26)
except Exception:
    pass
layout["Clear_all"] = [1214, 72, 1392, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_04_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-6/27_text_9.31.png
try:
    _c27 = get_crop(27, 89, 43)
    canvas.paste(_c27, (20, 17), _c27)
except Exception:
    pass
layout["9.31"] = [20, 17, 109, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_04_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-6/28_text_Filters.png
try:
    _c28 = get_crop(28, 180, 66)
    canvas.paste(_c28, (631, 116), _c28)
except Exception:
    pass
layout["Filters"] = [631, 116, 811, 182]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_04_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-6/29_text_Categories.png
try:
    _c29 = get_crop(29, 187, 135)
    canvas.paste(_c29, (36, 383), _c29)
except Exception:
    pass
layout["Categories"] = [36, 383, 223, 518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_04_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-6/30_text_Show_all_categories.png
try:
    _c30 = get_crop(30, 516, 144)
    canvas.paste(_c30, (0, 518), _c30)
except Exception:
    pass
layout["Show_all_categories"] = [0, 518, 516, 662]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_04_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-6/31_text_Event_type.png
try:
    _c31 = get_crop(31, 298, 135)
    canvas.paste(_c31, (36, 829), _c31)
except Exception:
    pass
layout["Event_type"] = [36, 829, 334, 964]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_04_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-6/32_text_Show_all_event_types.png
try:
    _c32 = get_crop(32, 535, 144)
    canvas.paste(_c32, (0, 964), _c32)
except Exception:
    pass
layout["Show_all_event_types"] = [0, 964, 535, 1108]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_04_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-6/33_text_Languages.png
try:
    _c33 = get_crop(33, 210, 135)
    canvas.paste(_c33, (36, 1275), _c33)
except Exception:
    pass
layout["Languages"] = [36, 1275, 246, 1410]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_04_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-6/34_text_Show_all_languages.png
try:
    _c34 = get_crop(34, 511, 144)
    canvas.paste(_c34, (0, 1410), _c34)
except Exception:
    pass
layout["Show_all_languages"] = [0, 1410, 511, 1554]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_04_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-6/35_text_Price.png
try:
    _c35 = get_crop(35, 149, 63)
    canvas.paste(_c35, (45, 1613), _c35)
except Exception:
    pass
layout["Price"] = [45, 1613, 194, 1676]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_04_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-6/36_text_Only_free_events.png
try:
    _c36 = get_crop(36, 660, 144)
    canvas.paste(_c36, (54, 2024), _c36)
except Exception:
    pass
layout["Only_free_events"] = [54, 2024, 714, 2168]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_04_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-6/37_text_Sort_by.png
try:
    _c37 = get_crop(37, 208, 76)
    canvas.paste(_c37, (40, 1930), _c37)
except Exception:
    pass
layout["Sort_by"] = [40, 1930, 248, 2006]
