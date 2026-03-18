# page_id: page_eventbrite_ee1eef38e6e94342b57a272493366950_05
# screenshot: 2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-7.png
# step_index: 5/10
# task: Open Eventbrite. Open "Fashion" category. Apply filter for free events. From the list, select the first non-promoted event and add it to your favorites.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw overall background (canvas already provided as white). We'll tint it very slightly to match the screenshot's neutral white.
bg_tint = (255, 255, 255)  # keep it white as the UI is predominantly white
draw.rectangle([(0, 0), canvas.size], fill=bg_tint)

# Status bar (top ~72px) - light gray strip
status_h = 72
status_color = (200, 200, 200)  # light gray
draw.rectangle([(0, 0), (canvas.width, status_h)], fill=status_color)

# Header / Toolbar area (below status bar). Keep it white but add a subtle bottom divider.
header_top = status_h
header_bottom = 168
header_color = (255, 255, 255)
draw.rectangle([(0, header_top), (canvas.width, header_bottom)], fill=header_color)

# Header bottom divider (fine line)
divider_color = (230, 232, 238)  # very light cool divider
draw.line([(24, header_bottom - 1), (canvas.width - 24, header_bottom - 1)], fill=divider_color, width=2)

# Section containers (rounded rectangles behind groups). Use very subtle fills and outlines so they don't compete with pasted content.
container_fill = (249, 251, 255)   # extremely subtle cool tint
container_outline = (235, 238, 245)  # subtle outline
corner_radius = 28

# Categories container (large block around the chips area)
cat_top = 220
cat_bottom = 1180
cat_left = 24
cat_right = canvas.width - 24
draw.rounded_rectangle([(cat_left, cat_top), (cat_right, cat_bottom)], radius=corner_radius,
                       fill=container_fill, outline=container_outline, width=1)

# Event type container
etype_top = 1380
etype_bottom = 1560
etype_left = 24
etype_right = canvas.width - 24
draw.rounded_rectangle([(etype_left, etype_top), (etype_right, etype_bottom)], radius=22,
                       fill=container_fill, outline=container_outline, width=1)

# Languages container
lang_top = 1820
lang_bottom = 1980
lang_left = 24
lang_right = canvas.width - 24
draw.rounded_rectangle([(lang_left, lang_top), (lang_right, lang_bottom)], radius=22,
                       fill=container_fill, outline=container_outline, width=1)

# Price / Only free events container (subtle area)
price_top = 2180
price_bottom = 2380
price_left = 24
price_right = canvas.width - 24
draw.rounded_rectangle([(price_left, price_top), (price_right, price_bottom)], radius=22,
                       fill=(255, 255, 255), outline=container_outline, width=1)

# Sort by area (thin container)
sort_top = 2520
sort_bottom = 2680
sort_left = 24
sort_right = canvas.width - 24
draw.rounded_rectangle([(sort_left, sort_top), (sort_right, sort_bottom)], radius=18,
                       fill=(255, 255, 255), outline=container_outline, width=1)

# Subtle separators between major sections (light lines)
sep_color = (242, 244, 249)
for y in (1160, 1360, 1800, 2160, 2500):
    draw.line([(36, y), (canvas.width - 36, y)], fill=sep_color, width=1)

# Add a faint shadow line under the last visible content area before the fixed bottom area (keeps visual separation)
shadow_y = 2720
draw.line([(24, shadow_y), (canvas.width - 24, shadow_y)], fill=(235, 235, 240), width=2)

# Decorative rounded corner highlights on containers (very subtle inner strokes)
inner_outline = (248, 249, 252)
draw.rounded_rectangle([(cat_left + 2, cat_top + 2), (cat_right - 2, cat_bottom - 2)], radius=corner_radius - 2,
                       outline=inner_outline, width=1)
draw.rounded_rectangle([(etype_left + 1, etype_top + 1), (etype_right - 1, etype_bottom - 1)], radius=20,
                       outline=inner_outline, width=1)
draw.rounded_rectangle([(lang_left + 1, lang_top + 1), (lang_right - 1, lang_bottom - 1)], radius=20,
                       outline=inner_outline, width=1)

# Note: No icons, text, buttons, or chips are drawn here. Detected UI elements will be pasted on top separately.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_05_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-7/00_icon_Fashion.png
try:
    _c0 = get_crop(0, 220, 144)
    canvas.paste(_c0, (1068, 764), _c0)
except Exception:
    pass
layout["Fashion"] = [1068, 764, 1288, 908]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_05_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-7/01_icon_Music.png
try:
    _c1 = get_crop(1, 187, 127)
    canvas.paste(_c1, (36, 383), _c1)
except Exception:
    pass
layout["Music"] = [36, 383, 223, 510]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_05_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-7/02_icon_Food_Drink.png
try:
    _c2 = get_crop(2, 312, 144)
    canvas.paste(_c2, (512, 383), _c2)
except Exception:
    pass
layout["Food_&_Drink"] = [512, 383, 824, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_05_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-7/03_icon_Health.png
try:
    _c3 = get_crop(3, 199, 144)
    canvas.paste(_c3, (777, 510), _c3)
except Exception:
    pass
layout["Health"] = [777, 510, 976, 654]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_05_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-7/04_icon_Business.png
try:
    _c4 = get_crop(4, 241, 144)
    canvas.paste(_c4, (247, 383), _c4)
except Exception:
    pass
layout["Business"] = [247, 383, 488, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_05_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-7/05_icon_Community.png
try:
    _c5 = get_crop(5, 294, 144)
    canvas.paste(_c5, (848, 383), _c5)
except Exception:
    pass
layout["Community"] = [848, 383, 1142, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_05_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-7/06_icon_Government.png
try:
    _c6 = get_crop(6, 310, 144)
    canvas.paste(_c6, (734, 764), _c6)
except Exception:
    pass
layout["Government"] = [734, 764, 1044, 908]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_05_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-7/07_icon_Expo.png
try:
    _c7 = get_crop(7, 167, 144)
    canvas.paste(_c7, (614, 1464), _c7)
except Exception:
    pass
layout["Expo"] = [614, 1464, 781, 1608]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_05_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-7/08_icon_Arts.png
try:
    _c8 = get_crop(8, 152, 127)
    canvas.paste(_c8, (1166, 383), _c8)
except Exception:
    pass
layout["Arts"] = [1166, 383, 1318, 510]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_05_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-7/09_icon_Holiday.png
try:
    _c9 = get_crop(9, 218, 127)
    canvas.paste(_c9, (492, 764), _c9)
except Exception:
    pass
layout["Holiday"] = [492, 764, 710, 891]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_05_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-7/10_icon_Spanish.png
try:
    _c10 = get_crop(10, 225, 144)
    canvas.paste(_c10, (519, 1910), _c10)
except Exception:
    pass
layout["Spanish"] = [519, 1910, 744, 2054]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_05_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-7/11_icon_Auto_Boat_Air.png
try:
    _c11 = get_crop(11, 369, 144)
    canvas.paste(_c11, (449, 891), _c11)
except Exception:
    pass
layout["Auto,_Boat_&_Air"] = [449, 891, 818, 1035]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_05_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-7/12_icon_Apply_filters_1.png
try:
    _c12 = get_crop(12, 1344, 144)
    canvas.paste(_c12, (48, 2768), _c12)
except Exception:
    pass
layout["Apply_filters_(1)"] = [48, 2768, 1392, 2912]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_05_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-7/13_icon_French.png
try:
    _c13 = get_crop(13, 205, 144)
    canvas.paste(_c13, (768, 1910), _c13)
except Exception:
    pass
layout["French"] = [768, 1910, 973, 2054]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_05_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-7/14_icon_Seminar.png
try:
    _c14 = get_crop(14, 232, 144)
    canvas.paste(_c14, (358, 1464), _c14)
except Exception:
    pass
layout["Seminar"] = [358, 1464, 590, 1608]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_05_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-7/15_icon_Spirituality.png
try:
    _c15 = get_crop(15, 282, 144)
    canvas.paste(_c15, (870, 637), _c15)
except Exception:
    pass
layout["Spirituality"] = [870, 637, 1152, 781]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_05_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-7/16_icon_Italian.png
try:
    _c16 = get_crop(16, 191, 144)
    canvas.paste(_c16, (997, 1910), _c16)
except Exception:
    pass
layout["Italian"] = [997, 1910, 1188, 2054]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_05_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-7/17_icon_Film_Media.png
try:
    _c17 = get_crop(17, 315, 127)
    canvas.paste(_c17, (36, 510), _c17)
except Exception:
    pass
layout["Film_&_Media"] = [36, 510, 351, 637]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_05_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-7/18_icon_Family_Education.png
try:
    _c18 = get_crop(18, 432, 144)
    canvas.paste(_c18, (36, 764), _c18)
except Exception:
    pass
layout["Family_&_Education"] = [36, 764, 468, 908]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_05_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-7/19_icon_Convention.png
try:
    _c19 = get_crop(19, 293, 144)
    canvas.paste(_c19, (805, 1464), _c19)
except Exception:
    pass
layout["Convention"] = [805, 1464, 1098, 1608]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_05_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-7/20_icon_Science_Tech.png
try:
    _c20 = get_crop(20, 361, 144)
    canvas.paste(_c20, (1000, 510), _c20)
except Exception:
    pass
layout["Science_&_Tech"] = [1000, 510, 1361, 654]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_05_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-7/21_icon_Festival.png
try:
    _c21 = get_crop(21, 219, 144)
    canvas.paste(_c21, (1122, 1464), _c21)
except Exception:
    pass
layout["Festival"] = [1122, 1464, 1341, 1608]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_05_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-7/22_icon_Sports_Fitness.png
try:
    _c22 = get_crop(22, 378, 144)
    canvas.paste(_c22, (375, 510), _c22)
except Exception:
    pass
layout["Sports_&_Fitness"] = [375, 510, 753, 654]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_05_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-7/23_icon_Home_Lifestyle.png
try:
    _c23 = get_crop(23, 389, 127)
    canvas.paste(_c23, (36, 891), _c23)
except Exception:
    pass
layout["Home_&_Lifestyle"] = [36, 891, 425, 1018]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_05_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-7/24_icon_Hobbies.png
try:
    _c24 = get_crop(24, 231, 144)
    canvas.paste(_c24, (842, 891), _c24)
except Exception:
    pass
layout["Hobbies"] = [842, 891, 1073, 1035]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_05_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-7/25_icon_Charity.png
try:
    _c25 = get_crop(25, 397, 144)
    canvas.paste(_c25, (449, 637), _c25)
except Exception:
    pass
layout["Charity"] = [449, 637, 846, 781]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_05_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-7/26_icon_German.png
try:
    _c26 = get_crop(26, 225, 135)
    canvas.paste(_c26, (270, 1910), _c26)
except Exception:
    pass
layout["German"] = [270, 1910, 495, 2045]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_05_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-7/27_icon_English.png
try:
    _c27 = get_crop(27, 210, 135)
    canvas.paste(_c27, (36, 1910), _c27)
except Exception:
    pass
layout["English"] = [36, 1910, 246, 2045]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_05_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-7/28_icon_Travel_Outdoor.png
try:
    _c28 = get_crop(28, 389, 127)
    canvas.paste(_c28, (36, 637), _c28)
except Exception:
    pass
layout["Travel_&_Outdoor"] = [36, 637, 425, 764]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_05_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-7/29_icon_Conference.png
try:
    _c29 = get_crop(29, 298, 135)
    canvas.paste(_c29, (36, 1464), _c29)
except Exception:
    pass
layout["Conference"] = [36, 1464, 334, 1599]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_05_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-7/30_icon_School_Activities.png
try:
    _c30 = get_crop(30, 392, 135)
    canvas.paste(_c30, (36, 1018), _c30)
except Exception:
    pass
layout["School_Activities"] = [36, 1018, 428, 1153]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_05_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-7/31_icon_5.27.png
try:
    _c31 = get_crop(31, 144, 144)
    canvas.paste(_c31, (12, 72), _c31)
except Exception:
    pass
layout["5.27"] = [12, 72, 156, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_05_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-7/32_icon_5.27.png
try:
    _c32 = get_crop(32, 64, 64)
    canvas.paste(_c32, (111, 1), _c32)
except Exception:
    pass
layout["5.27"] = [111, 1, 175, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_05_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-7/33_icon_5.27.png
try:
    _c33 = get_crop(33, 59, 63)
    canvas.paste(_c33, (181, 0), _c33)
except Exception:
    pass
layout["5.27"] = [181, 0, 240, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_05_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-7/34_icon_icon_34.png
try:
    _c34 = get_crop(34, 64, 60)
    canvas.paste(_c34, (308, 3), _c34)
except Exception:
    pass
layout["icon_34"] = [308, 3, 372, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_05_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-7/35_icon_Clear_all.png
try:
    _c35 = get_crop(35, 100, 62)
    canvas.paste(_c35, (1212, 0), _c35)
except Exception:
    pass
layout["Clear_all"] = [1212, 0, 1312, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_05_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-7/36_icon_Clear_all.png
try:
    _c36 = get_crop(36, 54, 64)
    canvas.paste(_c36, (1319, 0), _c36)
except Exception:
    pass
layout["Clear_all"] = [1319, 0, 1373, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_05_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-7/37_icon_icon_37.png
try:
    _c37 = get_crop(37, 51, 61)
    canvas.paste(_c37, (248, 2), _c37)
except Exception:
    pass
layout["icon_37"] = [248, 2, 299, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_05_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-7/38_icon_clickable_35.png
try:
    _c38 = get_crop(38, 144, 144)
    canvas.paste(_c38, (1248, 2364), _c38)
except Exception:
    pass
layout["clickable_35"] = [1248, 2364, 1392, 2508]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_05_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-7/39_icon_Clear_all.png
try:
    _c39 = get_crop(39, 178, 144)
    canvas.paste(_c39, (1214, 72), _c39)
except Exception:
    pass
layout["Clear_all"] = [1214, 72, 1392, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_05_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-7/40_icon_5.27.png
try:
    _c40 = get_crop(40, 103, 64)
    canvas.paste(_c40, (8, 0), _c40)
except Exception:
    pass
layout["5.27"] = [8, 0, 111, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_05_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-7/41_text_Filters.png
try:
    _c41 = get_crop(41, 180, 66)
    canvas.paste(_c41, (631, 116), _c41)
except Exception:
    pass
layout["Filters"] = [631, 116, 811, 182]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_05_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-7/42_text_Categories.png
try:
    _c42 = get_crop(42, 187, 127)
    canvas.paste(_c42, (36, 383), _c42)
except Exception:
    pass
layout["Categories"] = [36, 383, 223, 510]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_05_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-7/43_text_Show_less_categories.png
try:
    _c43 = get_crop(43, 550, 144)
    canvas.paste(_c43, (0, 1153), _c43)
except Exception:
    pass
layout["Show_less_categories"] = [0, 1153, 550, 1297]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_05_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-7/44_text_Event_type.png
try:
    _c44 = get_crop(44, 298, 135)
    canvas.paste(_c44, (36, 1464), _c44)
except Exception:
    pass
layout["Event_type"] = [36, 1464, 334, 1599]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_05_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-7/45_text_Show_all_event_types.png
try:
    _c45 = get_crop(45, 535, 144)
    canvas.paste(_c45, (0, 1599), _c45)
except Exception:
    pass
layout["Show_all_event_types"] = [0, 1599, 535, 1743]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_05_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-7/46_text_Languages.png
try:
    _c46 = get_crop(46, 210, 135)
    canvas.paste(_c46, (36, 1910), _c46)
except Exception:
    pass
layout["Languages"] = [36, 1910, 246, 2045]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_05_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-7/47_text_Show_all_languages.png
try:
    _c47 = get_crop(47, 511, 144)
    canvas.paste(_c47, (0, 2045), _c47)
except Exception:
    pass
layout["Show_all_languages"] = [0, 2045, 511, 2189]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_05_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-7/48_text_Price.png
try:
    _c48 = get_crop(48, 149, 63)
    canvas.paste(_c48, (45, 2249), _c48)
except Exception:
    pass
layout["Price"] = [45, 2249, 194, 2312]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_05_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-7/49_text_Only_free_events.png
try:
    _c49 = get_crop(49, 511, 144)
    canvas.paste(_c49, (0, 2045), _c49)
except Exception:
    pass
layout["Only_free_events"] = [0, 2045, 511, 2189]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_05_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-7/50_text_Sort_by.png
try:
    _c50 = get_crop(50, 206, 75)
    canvas.paste(_c50, (42, 2567), _c50)
except Exception:
    pass
layout["Sort_by"] = [42, 2567, 248, 2642]
