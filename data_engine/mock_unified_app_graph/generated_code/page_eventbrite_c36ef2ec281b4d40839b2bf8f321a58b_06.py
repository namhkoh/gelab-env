# page_id: page_eventbrite_c36ef2ec281b4d40839b2bf8f321a58b_06
# screenshot: 2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-8.png
# step_index: 6/8
# task: Open Eventbrite. Set the city to "Chicago". Select the "Fashion" category and view the recommended events. See the date of the first play and its venue.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Paint UI structure on provided canvas and draw objects.
w, h = canvas.size

# Overall background (dominant color from screenshot)
draw.rectangle([(0, 0), (w, h)], fill="#FFFFFF")

# Status bar area (top ~88px) - a neutral light gray
status_h = 88
draw.rectangle([(0, 0), (w, status_h)], fill="#CFCFCF")

# Header / Toolbar area (below status bar)
header_top = status_h
header_bottom = 176
draw.rectangle([(0, header_top), (w, header_bottom)], fill="#FFFFFF")
# subtle bottom divider for header
draw.line([(24, header_bottom), (w - 24, header_bottom)], fill="#E6E7EA", width=2)

# Large subtle grouping background for "Categories" area (rounded rectangle)
categories_top = 220
categories_bottom = 1180
pad_x = 24
draw.rounded_rectangle(
    [(pad_x, categories_top), (w - pad_x, categories_bottom)],
    radius=20,
    fill="#F7FBFF",
    outline="#EEF4F9",
    width=1
)

# Divider below categories (thin line)
div1_y = categories_bottom + 24
draw.line([(24, div1_y), (w - 24, div1_y)], fill="#ECEFF2", width=1)

# Event type section background (rounded)
event_type_top = 1400
event_type_bottom = 1660
draw.rounded_rectangle(
    [(pad_x, event_type_top), (w - pad_x, event_type_bottom)],
    radius=18,
    fill="#F7FBFF",
    outline="#EEF4F9",
    width=1
)

# Divider below event type
div2_y = event_type_bottom + 20
draw.line([(24, div2_y), (w - 24, div2_y)], fill="#ECEFF2", width=1)

# Languages section background (rounded)
languages_top = 1860
languages_bottom = 2100
draw.rounded_rectangle(
    [(pad_x, languages_top), (w - pad_x, languages_bottom)],
    radius=18,
    fill="#F7FBFF",
    outline="#EEF4F9",
    width=1
)

# Divider below languages
div3_y = languages_bottom + 28
draw.line([(24, div3_y), (w - 24, div3_y)], fill="#ECEFF2", width=1)

# Price and "Only free events" area background (subtle)
price_top = 2220
price_bottom = 2520
draw.rounded_rectangle(
    [(pad_x, price_top), (w - pad_x, price_bottom)],
    radius=14,
    fill="#FFFFFF",
    outline="#F0F2F5",
    width=1
)

# Sub-divider lines inside price/sort area to suggest rows (no controls drawn)
row_y = price_top + 64
draw.line([(24, row_y), (w - 24, row_y)], fill="#F1F3F6", width=1)
row_y2 = row_y + 120
draw.line([(24, row_y2), (w - 24, row_y2)], fill="#F1F3F6", width=1)

# Bottom area: lightly indicate content area above the sticky "Apply filters" bar
# (but do not draw the actual button or text)
bottom_area_top = h - 220
draw.rectangle([(0, bottom_area_top), (w, h)], fill="#FFFFFF")
# subtle top border for the sticky region area
draw.line([(24, bottom_area_top), (w - 24, bottom_area_top)], fill="#E6E7EA", width=2)

# Additional subtle vertical rhythm lines (margins) to guide layout (very light)
draw.line([(24, header_bottom + 8), (24, h - 8)], fill="#F5F7F9", width=1)
draw.line([(w - 24, header_bottom + 8), (w - 24, h - 8)], fill="#F5F7F9", width=1)

# Small decorative shadow under header to separate from content
for i, alpha in enumerate([18, 14, 10], start=0):
    y = header_bottom + i
    shade = (230, 231, 234, alpha)
    # we can't rely on RGBA draw; simulate by drawing very light lines
    draw.line([(24, y), (w - 24, y)], fill="#F0F2F5", width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_06_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-8/00_icon_Food_Drink.png
try:
    _c0 = get_crop(0, 312, 144)
    canvas.paste(_c0, (512, 383), _c0)
except Exception:
    pass
layout["Food_&_Drink"] = [512, 383, 824, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_06_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-8/01_icon_Music.png
try:
    _c1 = get_crop(1, 187, 127)
    canvas.paste(_c1, (36, 383), _c1)
except Exception:
    pass
layout["Music"] = [36, 383, 223, 510]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_06_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-8/02_icon_Health.png
try:
    _c2 = get_crop(2, 199, 144)
    canvas.paste(_c2, (777, 510), _c2)
except Exception:
    pass
layout["Health"] = [777, 510, 976, 654]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_06_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-8/03_icon_Government.png
try:
    _c3 = get_crop(3, 310, 144)
    canvas.paste(_c3, (734, 764), _c3)
except Exception:
    pass
layout["Government"] = [734, 764, 1044, 908]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_06_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-8/04_icon_Community.png
try:
    _c4 = get_crop(4, 294, 144)
    canvas.paste(_c4, (848, 383), _c4)
except Exception:
    pass
layout["Community"] = [848, 383, 1142, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_06_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-8/05_icon_Business.png
try:
    _c5 = get_crop(5, 241, 144)
    canvas.paste(_c5, (247, 383), _c5)
except Exception:
    pass
layout["Business"] = [247, 383, 488, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_06_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-8/06_icon_Expo.png
try:
    _c6 = get_crop(6, 167, 144)
    canvas.paste(_c6, (614, 1464), _c6)
except Exception:
    pass
layout["Expo"] = [614, 1464, 781, 1608]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_06_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-8/07_icon_Auto_Boat_Air.png
try:
    _c7 = get_crop(7, 369, 144)
    canvas.paste(_c7, (449, 891), _c7)
except Exception:
    pass
layout["Auto,_Boat_&_Air"] = [449, 891, 818, 1035]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_06_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-8/08_icon_Holiday.png
try:
    _c8 = get_crop(8, 218, 127)
    canvas.paste(_c8, (492, 764), _c8)
except Exception:
    pass
layout["Holiday"] = [492, 764, 710, 891]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_06_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-8/09_icon_Spirituality.png
try:
    _c9 = get_crop(9, 282, 144)
    canvas.paste(_c9, (870, 637), _c9)
except Exception:
    pass
layout["Spirituality"] = [870, 637, 1152, 781]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_06_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-8/10_icon_Arts.png
try:
    _c10 = get_crop(10, 152, 127)
    canvas.paste(_c10, (1166, 383), _c10)
except Exception:
    pass
layout["Arts"] = [1166, 383, 1318, 510]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_06_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-8/11_icon_Spanish.png
try:
    _c11 = get_crop(11, 225, 144)
    canvas.paste(_c11, (519, 1910), _c11)
except Exception:
    pass
layout["Spanish"] = [519, 1910, 744, 2054]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_06_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-8/12_icon_French.png
try:
    _c12 = get_crop(12, 205, 144)
    canvas.paste(_c12, (768, 1910), _c12)
except Exception:
    pass
layout["French"] = [768, 1910, 973, 2054]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_06_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-8/13_icon_Fashion.png
try:
    _c13 = get_crop(13, 220, 144)
    canvas.paste(_c13, (1068, 764), _c13)
except Exception:
    pass
layout["Fashion"] = [1068, 764, 1288, 908]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_06_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-8/14_icon_Seminar.png
try:
    _c14 = get_crop(14, 232, 144)
    canvas.paste(_c14, (358, 1464), _c14)
except Exception:
    pass
layout["Seminar"] = [358, 1464, 590, 1608]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_06_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-8/15_icon_Film_Media.png
try:
    _c15 = get_crop(15, 315, 127)
    canvas.paste(_c15, (36, 510), _c15)
except Exception:
    pass
layout["Film_&_Media"] = [36, 510, 351, 637]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_06_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-8/16_icon_Italian.png
try:
    _c16 = get_crop(16, 191, 144)
    canvas.paste(_c16, (997, 1910), _c16)
except Exception:
    pass
layout["Italian"] = [997, 1910, 1188, 2054]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_06_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-8/17_icon_Family_Education.png
try:
    _c17 = get_crop(17, 432, 144)
    canvas.paste(_c17, (36, 764), _c17)
except Exception:
    pass
layout["Family_&_Education"] = [36, 764, 468, 908]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_06_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-8/18_icon_Convention.png
try:
    _c18 = get_crop(18, 293, 144)
    canvas.paste(_c18, (805, 1464), _c18)
except Exception:
    pass
layout["Convention"] = [805, 1464, 1098, 1608]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_06_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-8/19_icon_Science_Tech.png
try:
    _c19 = get_crop(19, 361, 144)
    canvas.paste(_c19, (1000, 510), _c19)
except Exception:
    pass
layout["Science_&_Tech"] = [1000, 510, 1361, 654]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_06_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-8/20_icon_Sports_Fitness.png
try:
    _c20 = get_crop(20, 378, 144)
    canvas.paste(_c20, (375, 510), _c20)
except Exception:
    pass
layout["Sports_&_Fitness"] = [375, 510, 753, 654]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_06_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-8/21_icon_Home_Lifestyle.png
try:
    _c21 = get_crop(21, 389, 127)
    canvas.paste(_c21, (36, 891), _c21)
except Exception:
    pass
layout["Home_&_Lifestyle"] = [36, 891, 425, 1018]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_06_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-8/22_icon_Charity.png
try:
    _c22 = get_crop(22, 397, 144)
    canvas.paste(_c22, (449, 637), _c22)
except Exception:
    pass
layout["Charity"] = [449, 637, 846, 781]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_06_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-8/23_icon_Festival.png
try:
    _c23 = get_crop(23, 219, 144)
    canvas.paste(_c23, (1122, 1464), _c23)
except Exception:
    pass
layout["Festival"] = [1122, 1464, 1341, 1608]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_06_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-8/24_icon_Hobbies.png
try:
    _c24 = get_crop(24, 231, 144)
    canvas.paste(_c24, (842, 891), _c24)
except Exception:
    pass
layout["Hobbies"] = [842, 891, 1073, 1035]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_06_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-8/25_icon_German.png
try:
    _c25 = get_crop(25, 225, 135)
    canvas.paste(_c25, (270, 1910), _c25)
except Exception:
    pass
layout["German"] = [270, 1910, 495, 2045]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_06_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-8/26_icon_English.png
try:
    _c26 = get_crop(26, 210, 135)
    canvas.paste(_c26, (36, 1910), _c26)
except Exception:
    pass
layout["English"] = [36, 1910, 246, 2045]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_06_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-8/27_icon_Travel_Outdoor.png
try:
    _c27 = get_crop(27, 389, 127)
    canvas.paste(_c27, (36, 637), _c27)
except Exception:
    pass
layout["Travel_&_Outdoor"] = [36, 637, 425, 764]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_06_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-8/28_icon_Conference.png
try:
    _c28 = get_crop(28, 298, 135)
    canvas.paste(_c28, (36, 1464), _c28)
except Exception:
    pass
layout["Conference"] = [36, 1464, 334, 1599]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_06_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-8/29_icon_School_Activities.png
try:
    _c29 = get_crop(29, 392, 135)
    canvas.paste(_c29, (36, 1018), _c29)
except Exception:
    pass
layout["School_Activities"] = [36, 1018, 428, 1153]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_06_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-8/30_icon_Apply_filters.png
try:
    _c30 = get_crop(30, 1344, 144)
    canvas.paste(_c30, (48, 2768), _c30)
except Exception:
    pass
layout["Apply_filters"] = [48, 2768, 1392, 2912]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_06_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-8/31_icon_5.13.png
try:
    _c31 = get_crop(31, 144, 144)
    canvas.paste(_c31, (12, 72), _c31)
except Exception:
    pass
layout["5.13"] = [12, 72, 156, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_06_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-8/32_icon_5.13.png
try:
    _c32 = get_crop(32, 64, 65)
    canvas.paste(_c32, (112, 0), _c32)
except Exception:
    pass
layout["5.13"] = [112, 0, 176, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_06_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-8/33_icon_5.13.png
try:
    _c33 = get_crop(33, 61, 63)
    canvas.paste(_c33, (180, 0), _c33)
except Exception:
    pass
layout["5.13"] = [180, 0, 241, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_06_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-8/34_icon_icon_34.png
try:
    _c34 = get_crop(34, 64, 61)
    canvas.paste(_c34, (308, 2), _c34)
except Exception:
    pass
layout["icon_34"] = [308, 2, 372, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_06_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-8/35_icon_Clear_all.png
try:
    _c35 = get_crop(35, 55, 66)
    canvas.paste(_c35, (1319, 0), _c35)
except Exception:
    pass
layout["Clear_all"] = [1319, 0, 1374, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_06_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-8/36_icon_Clear_all.png
try:
    _c36 = get_crop(36, 101, 64)
    canvas.paste(_c36, (1211, 0), _c36)
except Exception:
    pass
layout["Clear_all"] = [1211, 0, 1312, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_06_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-8/37_icon_icon_37.png
try:
    _c37 = get_crop(37, 51, 62)
    canvas.paste(_c37, (248, 1), _c37)
except Exception:
    pass
layout["icon_37"] = [248, 1, 299, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_06_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-8/38_icon_clickable_35.png
try:
    _c38 = get_crop(38, 144, 144)
    canvas.paste(_c38, (1248, 2364), _c38)
except Exception:
    pass
layout["clickable_35"] = [1248, 2364, 1392, 2508]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_06_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-8/39_icon_Clear_all.png
try:
    _c39 = get_crop(39, 178, 144)
    canvas.paste(_c39, (1214, 72), _c39)
except Exception:
    pass
layout["Clear_all"] = [1214, 72, 1392, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_06_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-8/40_text_5.13.png
try:
    _c40 = get_crop(40, 89, 43)
    canvas.paste(_c40, (22, 17), _c40)
except Exception:
    pass
layout["5.13"] = [22, 17, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_06_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-8/41_text_Filters.png
try:
    _c41 = get_crop(41, 180, 66)
    canvas.paste(_c41, (631, 116), _c41)
except Exception:
    pass
layout["Filters"] = [631, 116, 811, 182]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_06_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-8/42_text_Categories.png
try:
    _c42 = get_crop(42, 187, 127)
    canvas.paste(_c42, (36, 383), _c42)
except Exception:
    pass
layout["Categories"] = [36, 383, 223, 510]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_06_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-8/43_text_Show_less_categories.png
try:
    _c43 = get_crop(43, 550, 144)
    canvas.paste(_c43, (0, 1153), _c43)
except Exception:
    pass
layout["Show_less_categories"] = [0, 1153, 550, 1297]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_06_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-8/44_text_Event_type.png
try:
    _c44 = get_crop(44, 298, 135)
    canvas.paste(_c44, (36, 1464), _c44)
except Exception:
    pass
layout["Event_type"] = [36, 1464, 334, 1599]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_06_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-8/45_text_Show_all_event_types.png
try:
    _c45 = get_crop(45, 535, 144)
    canvas.paste(_c45, (0, 1599), _c45)
except Exception:
    pass
layout["Show_all_event_types"] = [0, 1599, 535, 1743]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_06_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-8/46_text_Languages.png
try:
    _c46 = get_crop(46, 210, 135)
    canvas.paste(_c46, (36, 1910), _c46)
except Exception:
    pass
layout["Languages"] = [36, 1910, 246, 2045]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_06_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-8/47_text_Show_all_languages.png
try:
    _c47 = get_crop(47, 511, 144)
    canvas.paste(_c47, (0, 2045), _c47)
except Exception:
    pass
layout["Show_all_languages"] = [0, 2045, 511, 2189]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_06_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-8/48_text_Price.png
try:
    _c48 = get_crop(48, 149, 63)
    canvas.paste(_c48, (45, 2249), _c48)
except Exception:
    pass
layout["Price"] = [45, 2249, 194, 2312]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_06_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-8/49_text_Only_free_events.png
try:
    _c49 = get_crop(49, 511, 144)
    canvas.paste(_c49, (0, 2045), _c49)
except Exception:
    pass
layout["Only_free_events"] = [0, 2045, 511, 2189]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_06_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-8/50_text_Sort_by.png
try:
    _c50 = get_crop(50, 206, 75)
    canvas.paste(_c50, (42, 2567), _c50)
except Exception:
    pass
layout["Sort_by"] = [42, 2567, 248, 2642]
