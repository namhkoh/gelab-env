# page_id: page_eventbrite_ee1eef38e6e94342b57a272493366950_04
# screenshot: 2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-6.png
# step_index: 4/10
# task: Open Eventbrite. Open "Fashion" category. Apply filter for free events. From the list, select the first non-promoted event and add it to your favorites.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Top status bar
draw.rectangle([(0, 0), (1440, 96)], fill=(200, 200, 200))

# Header area (toolbar) beneath status bar
header_top = 96
header_bottom = 184
draw.rectangle([(0, header_top), (1440, header_bottom)], fill=(255, 255, 255))

# Subtle bottom divider under header
draw.line([(24, header_bottom), (1416, header_bottom)], fill=(230, 230, 235), width=2)

# Background card for "Categories" group (rounded)
cat_card_top = 200
cat_card_bottom = 1150
draw.rounded_rectangle(
    [(32, cat_card_top), (1408, cat_card_bottom)],
    radius=28,
    fill=(249, 250, 252),
    outline=(235, 235, 240),
    width=1
)

# Divider between categories and next area ("Show less categories" region)
draw.line([(32+16, cat_card_bottom+18), (1408-16, cat_card_bottom+18)], fill=(240, 240, 245), width=1)

# Background card for "Event type" group
etype_top = 1188
etype_bottom = 1620
draw.rounded_rectangle(
    [(32, etype_top), (1408, etype_bottom)],
    radius=24,
    fill=(252, 252, 253),
    outline=(238, 238, 242),
    width=1
)

# Separator under event type
draw.line([(32+16, etype_bottom+18), (1408-16, etype_bottom+18)], fill=(240, 240, 245), width=1)

# Background card for "Languages" group
lang_top = 1648
lang_bottom = 2008
draw.rounded_rectangle(
    [(32, lang_top), (1408, lang_bottom)],
    radius=24,
    fill=(252, 253, 255),
    outline=(239, 239, 243),
    width=1
)

# Separator under languages
draw.line([(32+16, lang_bottom+18), (1408-16, lang_bottom+18)], fill=(240, 240, 245), width=1)

# Background card for "Price" and "Only free events" area
price_top = 2036
price_bottom = 2360
draw.rounded_rectangle(
    [(32, price_top), (1408, price_bottom)],
    radius=20,
    fill=(255, 255, 255),
    outline=(235, 235, 240),
    width=1
)

# Subtle horizontal divider within price area (to separate price label from toggle and following content)
draw.line([(80, price_top+80), (1360, price_top+80)], fill=(245, 245, 249), width=1)

# Background card for "Sort by" area above bottom controls
sort_top = 2388
sort_bottom = 2720
draw.rounded_rectangle(
    [(32, sort_top), (1408, sort_bottom)],
    radius=18,
    fill=(255, 255, 255),
    outline=(235, 235, 241),
    width=1
)

# Rounded shadow/strip for the bottom "Apply filters" container
footer_top = 2728
footer_bottom = 2920
# subtle shadow behind the footer
draw.rectangle([(32, footer_top+6), (1408, footer_bottom)], fill=(230, 230, 235))
# main footer background (slightly off-white)
draw.rounded_rectangle(
    [(48, footer_top), (1392, footer_top+144)],
    radius=12,
    fill=(255, 255, 255),
    outline=(180, 180, 190),
    width=3
)

# Light section separators across main content to match list grouping
separators_y = [360, 620, 900, 1150, 1464, 1910, 2249, 2567]
for y in separators_y:
    draw.line([(32, y), (1408, y)], fill=(245, 245, 248), width=1)

# subtle left edge guide and right edge guide for alignment (very faint)
draw.line([(32, 0), (32, 2960)], fill=(250, 250, 252), width=1)
draw.line([(1408, 0), (1408, 2960)], fill=(250, 250, 252), width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_04_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-6/00_icon_Food_Drink.png
try:
    _c0 = get_crop(0, 312, 144)
    canvas.paste(_c0, (512, 383), _c0)
except Exception:
    pass
layout["Food_&_Drink"] = [512, 383, 824, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_04_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-6/01_icon_Music.png
try:
    _c1 = get_crop(1, 187, 127)
    canvas.paste(_c1, (36, 383), _c1)
except Exception:
    pass
layout["Music"] = [36, 383, 223, 510]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_04_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-6/02_icon_Health.png
try:
    _c2 = get_crop(2, 199, 144)
    canvas.paste(_c2, (777, 510), _c2)
except Exception:
    pass
layout["Health"] = [777, 510, 976, 654]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_04_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-6/03_icon_Community.png
try:
    _c3 = get_crop(3, 294, 144)
    canvas.paste(_c3, (848, 383), _c3)
except Exception:
    pass
layout["Community"] = [848, 383, 1142, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_04_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-6/04_icon_Business.png
try:
    _c4 = get_crop(4, 241, 144)
    canvas.paste(_c4, (247, 383), _c4)
except Exception:
    pass
layout["Business"] = [247, 383, 488, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_04_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-6/05_icon_Government.png
try:
    _c5 = get_crop(5, 310, 144)
    canvas.paste(_c5, (734, 764), _c5)
except Exception:
    pass
layout["Government"] = [734, 764, 1044, 908]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_04_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-6/06_icon_Expo.png
try:
    _c6 = get_crop(6, 167, 144)
    canvas.paste(_c6, (614, 1464), _c6)
except Exception:
    pass
layout["Expo"] = [614, 1464, 781, 1608]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_04_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-6/07_icon_Auto_Boat_Air.png
try:
    _c7 = get_crop(7, 369, 144)
    canvas.paste(_c7, (449, 891), _c7)
except Exception:
    pass
layout["Auto,_Boat_&_Air"] = [449, 891, 818, 1035]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_04_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-6/08_icon_Holiday.png
try:
    _c8 = get_crop(8, 218, 127)
    canvas.paste(_c8, (492, 764), _c8)
except Exception:
    pass
layout["Holiday"] = [492, 764, 710, 891]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_04_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-6/09_icon_Spirituality.png
try:
    _c9 = get_crop(9, 282, 144)
    canvas.paste(_c9, (870, 637), _c9)
except Exception:
    pass
layout["Spirituality"] = [870, 637, 1152, 781]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_04_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-6/10_icon_Arts.png
try:
    _c10 = get_crop(10, 152, 127)
    canvas.paste(_c10, (1166, 383), _c10)
except Exception:
    pass
layout["Arts"] = [1166, 383, 1318, 510]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_04_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-6/11_icon_Spanish.png
try:
    _c11 = get_crop(11, 225, 144)
    canvas.paste(_c11, (519, 1910), _c11)
except Exception:
    pass
layout["Spanish"] = [519, 1910, 744, 2054]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_04_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-6/12_icon_Fashion.png
try:
    _c12 = get_crop(12, 220, 144)
    canvas.paste(_c12, (1068, 764), _c12)
except Exception:
    pass
layout["Fashion"] = [1068, 764, 1288, 908]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_04_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-6/13_icon_French.png
try:
    _c13 = get_crop(13, 205, 144)
    canvas.paste(_c13, (768, 1910), _c13)
except Exception:
    pass
layout["French"] = [768, 1910, 973, 2054]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_04_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-6/14_icon_Seminar.png
try:
    _c14 = get_crop(14, 232, 144)
    canvas.paste(_c14, (358, 1464), _c14)
except Exception:
    pass
layout["Seminar"] = [358, 1464, 590, 1608]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_04_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-6/15_icon_Film_Media.png
try:
    _c15 = get_crop(15, 315, 127)
    canvas.paste(_c15, (36, 510), _c15)
except Exception:
    pass
layout["Film_&_Media"] = [36, 510, 351, 637]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_04_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-6/16_icon_Italian.png
try:
    _c16 = get_crop(16, 191, 144)
    canvas.paste(_c16, (997, 1910), _c16)
except Exception:
    pass
layout["Italian"] = [997, 1910, 1188, 2054]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_04_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-6/17_icon_Family_Education.png
try:
    _c17 = get_crop(17, 432, 144)
    canvas.paste(_c17, (36, 764), _c17)
except Exception:
    pass
layout["Family_&_Education"] = [36, 764, 468, 908]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_04_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-6/18_icon_Convention.png
try:
    _c18 = get_crop(18, 293, 144)
    canvas.paste(_c18, (805, 1464), _c18)
except Exception:
    pass
layout["Convention"] = [805, 1464, 1098, 1608]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_04_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-6/19_icon_Science_Tech.png
try:
    _c19 = get_crop(19, 361, 144)
    canvas.paste(_c19, (1000, 510), _c19)
except Exception:
    pass
layout["Science_&_Tech"] = [1000, 510, 1361, 654]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_04_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-6/20_icon_Sports_Fitness.png
try:
    _c20 = get_crop(20, 378, 144)
    canvas.paste(_c20, (375, 510), _c20)
except Exception:
    pass
layout["Sports_&_Fitness"] = [375, 510, 753, 654]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_04_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-6/21_icon_Home_Lifestyle.png
try:
    _c21 = get_crop(21, 389, 127)
    canvas.paste(_c21, (36, 891), _c21)
except Exception:
    pass
layout["Home_&_Lifestyle"] = [36, 891, 425, 1018]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_04_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-6/22_icon_Charity.png
try:
    _c22 = get_crop(22, 397, 144)
    canvas.paste(_c22, (449, 637), _c22)
except Exception:
    pass
layout["Charity"] = [449, 637, 846, 781]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_04_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-6/23_icon_Festival.png
try:
    _c23 = get_crop(23, 219, 144)
    canvas.paste(_c23, (1122, 1464), _c23)
except Exception:
    pass
layout["Festival"] = [1122, 1464, 1341, 1608]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_04_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-6/24_icon_Hobbies.png
try:
    _c24 = get_crop(24, 231, 144)
    canvas.paste(_c24, (842, 891), _c24)
except Exception:
    pass
layout["Hobbies"] = [842, 891, 1073, 1035]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_04_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-6/25_icon_German.png
try:
    _c25 = get_crop(25, 225, 135)
    canvas.paste(_c25, (270, 1910), _c25)
except Exception:
    pass
layout["German"] = [270, 1910, 495, 2045]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_04_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-6/26_icon_English.png
try:
    _c26 = get_crop(26, 210, 135)
    canvas.paste(_c26, (36, 1910), _c26)
except Exception:
    pass
layout["English"] = [36, 1910, 246, 2045]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_04_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-6/27_icon_Travel_Outdoor.png
try:
    _c27 = get_crop(27, 389, 127)
    canvas.paste(_c27, (36, 637), _c27)
except Exception:
    pass
layout["Travel_&_Outdoor"] = [36, 637, 425, 764]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_04_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-6/28_icon_Conference.png
try:
    _c28 = get_crop(28, 298, 135)
    canvas.paste(_c28, (36, 1464), _c28)
except Exception:
    pass
layout["Conference"] = [36, 1464, 334, 1599]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_04_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-6/29_icon_School_Activities.png
try:
    _c29 = get_crop(29, 392, 135)
    canvas.paste(_c29, (36, 1018), _c29)
except Exception:
    pass
layout["School_Activities"] = [36, 1018, 428, 1153]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_04_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-6/30_icon_Apply_filters.png
try:
    _c30 = get_crop(30, 1344, 144)
    canvas.paste(_c30, (48, 2768), _c30)
except Exception:
    pass
layout["Apply_filters"] = [48, 2768, 1392, 2912]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_04_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-6/31_icon_5.27.png
try:
    _c31 = get_crop(31, 144, 144)
    canvas.paste(_c31, (12, 72), _c31)
except Exception:
    pass
layout["5.27"] = [12, 72, 156, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_04_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-6/32_icon_5.27.png
try:
    _c32 = get_crop(32, 65, 64)
    canvas.paste(_c32, (111, 1), _c32)
except Exception:
    pass
layout["5.27"] = [111, 1, 176, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_04_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-6/33_icon_5.27.png
try:
    _c33 = get_crop(33, 61, 63)
    canvas.paste(_c33, (180, 0), _c33)
except Exception:
    pass
layout["5.27"] = [180, 0, 241, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_04_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-6/34_icon_icon_34.png
try:
    _c34 = get_crop(34, 64, 61)
    canvas.paste(_c34, (308, 2), _c34)
except Exception:
    pass
layout["icon_34"] = [308, 2, 372, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_04_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-6/35_icon_Clear_all.png
try:
    _c35 = get_crop(35, 55, 66)
    canvas.paste(_c35, (1319, 0), _c35)
except Exception:
    pass
layout["Clear_all"] = [1319, 0, 1374, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_04_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-6/36_icon_Clear_all.png
try:
    _c36 = get_crop(36, 101, 64)
    canvas.paste(_c36, (1211, 0), _c36)
except Exception:
    pass
layout["Clear_all"] = [1211, 0, 1312, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_04_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-6/37_icon_icon_37.png
try:
    _c37 = get_crop(37, 51, 61)
    canvas.paste(_c37, (248, 2), _c37)
except Exception:
    pass
layout["icon_37"] = [248, 2, 299, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_04_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-6/38_icon_clickable_35.png
try:
    _c38 = get_crop(38, 144, 144)
    canvas.paste(_c38, (1248, 2364), _c38)
except Exception:
    pass
layout["clickable_35"] = [1248, 2364, 1392, 2508]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_04_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-6/39_icon_Clear_all.png
try:
    _c39 = get_crop(39, 178, 144)
    canvas.paste(_c39, (1214, 72), _c39)
except Exception:
    pass
layout["Clear_all"] = [1214, 72, 1392, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_04_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-6/40_text_5.27.png
try:
    _c40 = get_crop(40, 87, 43)
    canvas.paste(_c40, (22, 17), _c40)
except Exception:
    pass
layout["5.27"] = [22, 17, 109, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_04_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-6/41_text_Filters.png
try:
    _c41 = get_crop(41, 180, 66)
    canvas.paste(_c41, (631, 116), _c41)
except Exception:
    pass
layout["Filters"] = [631, 116, 811, 182]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_04_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-6/42_text_Categories.png
try:
    _c42 = get_crop(42, 187, 127)
    canvas.paste(_c42, (36, 383), _c42)
except Exception:
    pass
layout["Categories"] = [36, 383, 223, 510]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_04_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-6/43_text_Show_less_categories.png
try:
    _c43 = get_crop(43, 550, 144)
    canvas.paste(_c43, (0, 1153), _c43)
except Exception:
    pass
layout["Show_less_categories"] = [0, 1153, 550, 1297]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_04_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-6/44_text_Event_type.png
try:
    _c44 = get_crop(44, 298, 135)
    canvas.paste(_c44, (36, 1464), _c44)
except Exception:
    pass
layout["Event_type"] = [36, 1464, 334, 1599]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_04_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-6/45_text_Show_all_event_types.png
try:
    _c45 = get_crop(45, 535, 144)
    canvas.paste(_c45, (0, 1599), _c45)
except Exception:
    pass
layout["Show_all_event_types"] = [0, 1599, 535, 1743]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_04_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-6/46_text_Languages.png
try:
    _c46 = get_crop(46, 210, 135)
    canvas.paste(_c46, (36, 1910), _c46)
except Exception:
    pass
layout["Languages"] = [36, 1910, 246, 2045]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_04_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-6/47_text_Show_all_languages.png
try:
    _c47 = get_crop(47, 511, 144)
    canvas.paste(_c47, (0, 2045), _c47)
except Exception:
    pass
layout["Show_all_languages"] = [0, 2045, 511, 2189]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_04_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-6/48_text_Price.png
try:
    _c48 = get_crop(48, 149, 63)
    canvas.paste(_c48, (45, 2249), _c48)
except Exception:
    pass
layout["Price"] = [45, 2249, 194, 2312]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_04_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-6/49_text_Only_free_events.png
try:
    _c49 = get_crop(49, 511, 144)
    canvas.paste(_c49, (0, 2045), _c49)
except Exception:
    pass
layout["Only_free_events"] = [0, 2045, 511, 2189]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_04_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-6/50_text_Sort_by.png
try:
    _c50 = get_crop(50, 206, 75)
    canvas.paste(_c50, (42, 2567), _c50)
except Exception:
    pass
layout["Sort_by"] = [42, 2567, 248, 2642]
