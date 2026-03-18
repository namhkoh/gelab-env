# page_id: page_eventbrite_3c2c0d71896b45acb211a472de4b4c9e_10
# screenshot: 2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-12.png
# step_index: 10/15
# task: Open Eventbrite. Search free Health event in Los Angeles. Select the first one that is not promoted. Follow the organizer. Share to google keep notes.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw overall subtle off-white background (canvas starts white; this slightly warms it)
draw.rectangle([(0, 0), (1440, 2960)], fill=(252, 252, 255))

# Status bar area at the very top (background only; icons/text are pasted on top)
status_h = 96
draw.rectangle([(0, 0), (1440, status_h)], fill=(220, 220, 220))

# Header / toolbar background (area behind the "Filters" title)
header_top = status_h
header_bottom = 176
draw.rectangle([(0, header_top), (1440, header_bottom)], fill=(255, 255, 255))

# Subtle divider/shadow under header
draw.line([(24, header_bottom), (1416, header_bottom)], fill=(230, 230, 235), width=2)

# --- Section background cards (rounded rectangles behind groups of elements) ---
# Categories card (large area behind many category chips)
cat_top = 260
cat_bottom = 1200
draw.rounded_rectangle([(24, cat_top), (1416, cat_bottom)],
                       radius=28, fill=(246, 250, 255), outline=(235, 240, 250), width=1)

# Event type card (small band behind the event-type chips)
etype_top = 1360
etype_bottom = 1640
draw.rounded_rectangle([(40, etype_top), (1400, etype_bottom)],
                       radius=24, fill=(250, 252, 255), outline=(235, 240, 250), width=1)

# Languages card
lang_top = 1820
lang_bottom = 2040
draw.rounded_rectangle([(40, lang_top), (1400, lang_bottom)],
                       radius=22, fill=(250, 253, 255), outline=(235, 240, 250), width=1)

# Price & sorting card area (background band above the bottom controls)
price_top = 2160
price_bottom = 2700  # keep this above the apply-filters button area (starts around 2768)
draw.rounded_rectangle([(36, price_top), (1404, price_bottom)],
                       radius=20, fill=(255, 255, 255), outline=(236, 238, 245), width=1)

# --- Separator lines between main sections ---
separators = [
    1153,  # below categories "Show less categories" area
    1599,  # below event type "Show all event types"
    2045,  # below languages "Show all languages"
    2249,  # near Price heading
    2567,  # near Sort by
    2704   # just above the apply filters area
]
for y in separators:
    draw.line([(36, y), (1404, y)], fill=(235, 238, 245), width=2)

# Thin inner dividers to give structure near top/bottom of cards
draw.line([(36, cat_top + 8), (1404, cat_top + 8)], fill=(245, 247, 250), width=1)
draw.line([(36, price_bottom - 8), (1404, price_bottom - 8)], fill=(245, 247, 250), width=1)

# Rounded corner highlights on the large cards (subtle strokes)
draw.rounded_rectangle([(24, cat_top), (1416, cat_bottom)], radius=28, outline=(245, 247, 250), width=1)
draw.rounded_rectangle([(40, etype_top), (1400, etype_bottom)], radius=22, outline=(245, 247, 250), width=1)
draw.rounded_rectangle([(40, lang_top), (1400, lang_bottom)], radius=20, outline=(245, 247, 250), width=1)
draw.rounded_rectangle([(36, price_top), (1404, price_bottom)], radius=20, outline=(245, 247, 250), width=1)

# Top-left back-area subtle accent (background only, do not draw the arrow)
back_accent_box = (24, header_top + 12, 120, header_bottom - 12)
draw.rounded_rectangle([back_accent_box[:2], back_accent_box[2:]], radius=12, fill=(250, 250, 255))

# Bottom safe-area divider (just above the apply filters button zone)
draw.line([(24, 2760), (1416, 2760)], fill=(230, 230, 235), width=3)

# Add faint vertical padding guides (visual structure only)
draw.line([(36, header_bottom+8), (36, price_bottom-8)], fill=(255, 255, 255), width=1)
draw.line([(1404, header_bottom+8), (1404, price_bottom-8)], fill=(255, 255, 255), width=1)

# Done drawing background and structural UI elements.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_10_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-12/00_icon_Music.png
try:
    _c0 = get_crop(0, 187, 127)
    canvas.paste(_c0, (36, 383), _c0)
except Exception:
    pass
layout["Music"] = [36, 383, 223, 510]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_10_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-12/01_icon_Food_Drink.png
try:
    _c1 = get_crop(1, 312, 144)
    canvas.paste(_c1, (512, 383), _c1)
except Exception:
    pass
layout["Food_&_Drink"] = [512, 383, 824, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_10_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-12/02_icon_Health.png
try:
    _c2 = get_crop(2, 199, 144)
    canvas.paste(_c2, (777, 510), _c2)
except Exception:
    pass
layout["Health"] = [777, 510, 976, 654]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_10_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-12/03_icon_Business.png
try:
    _c3 = get_crop(3, 241, 144)
    canvas.paste(_c3, (247, 383), _c3)
except Exception:
    pass
layout["Business"] = [247, 383, 488, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_10_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-12/04_icon_Community.png
try:
    _c4 = get_crop(4, 294, 144)
    canvas.paste(_c4, (848, 383), _c4)
except Exception:
    pass
layout["Community"] = [848, 383, 1142, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_10_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-12/05_icon_Government.png
try:
    _c5 = get_crop(5, 310, 144)
    canvas.paste(_c5, (734, 764), _c5)
except Exception:
    pass
layout["Government"] = [734, 764, 1044, 908]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_10_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-12/06_icon_Expo.png
try:
    _c6 = get_crop(6, 167, 144)
    canvas.paste(_c6, (614, 1464), _c6)
except Exception:
    pass
layout["Expo"] = [614, 1464, 781, 1608]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_10_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-12/07_icon_French.png
try:
    _c7 = get_crop(7, 205, 144)
    canvas.paste(_c7, (768, 1910), _c7)
except Exception:
    pass
layout["French"] = [768, 1910, 973, 2054]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_10_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-12/08_icon_Holiday.png
try:
    _c8 = get_crop(8, 218, 127)
    canvas.paste(_c8, (492, 764), _c8)
except Exception:
    pass
layout["Holiday"] = [492, 764, 710, 891]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_10_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-12/09_icon_Auto_Boat_Air.png
try:
    _c9 = get_crop(9, 369, 144)
    canvas.paste(_c9, (449, 891), _c9)
except Exception:
    pass
layout["Auto,_Boat_&_Air"] = [449, 891, 818, 1035]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_10_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-12/10_icon_Spanish.png
try:
    _c10 = get_crop(10, 225, 144)
    canvas.paste(_c10, (519, 1910), _c10)
except Exception:
    pass
layout["Spanish"] = [519, 1910, 744, 2054]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_10_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-12/11_icon_Fashion.png
try:
    _c11 = get_crop(11, 220, 144)
    canvas.paste(_c11, (1068, 764), _c11)
except Exception:
    pass
layout["Fashion"] = [1068, 764, 1288, 908]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_10_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-12/12_icon_Arts.png
try:
    _c12 = get_crop(12, 152, 127)
    canvas.paste(_c12, (1166, 383), _c12)
except Exception:
    pass
layout["Arts"] = [1166, 383, 1318, 510]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_10_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-12/13_icon_Italian.png
try:
    _c13 = get_crop(13, 191, 144)
    canvas.paste(_c13, (997, 1910), _c13)
except Exception:
    pass
layout["Italian"] = [997, 1910, 1188, 2054]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_10_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-12/14_icon_Seminar.png
try:
    _c14 = get_crop(14, 232, 144)
    canvas.paste(_c14, (358, 1464), _c14)
except Exception:
    pass
layout["Seminar"] = [358, 1464, 590, 1608]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_10_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-12/15_icon_Spirituality.png
try:
    _c15 = get_crop(15, 282, 144)
    canvas.paste(_c15, (870, 637), _c15)
except Exception:
    pass
layout["Spirituality"] = [870, 637, 1152, 781]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_10_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-12/16_icon_Film_Media.png
try:
    _c16 = get_crop(16, 315, 127)
    canvas.paste(_c16, (36, 510), _c16)
except Exception:
    pass
layout["Film_&_Media"] = [36, 510, 351, 637]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_10_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-12/17_icon_Apply_filters_1.png
try:
    _c17 = get_crop(17, 1344, 144)
    canvas.paste(_c17, (48, 2768), _c17)
except Exception:
    pass
layout["Apply_filters_(1)"] = [48, 2768, 1392, 2912]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_10_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-12/18_icon_Family_Education.png
try:
    _c18 = get_crop(18, 432, 144)
    canvas.paste(_c18, (36, 764), _c18)
except Exception:
    pass
layout["Family_&_Education"] = [36, 764, 468, 908]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_10_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-12/19_icon_Convention.png
try:
    _c19 = get_crop(19, 293, 144)
    canvas.paste(_c19, (805, 1464), _c19)
except Exception:
    pass
layout["Convention"] = [805, 1464, 1098, 1608]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_10_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-12/20_icon_Science_Tech.png
try:
    _c20 = get_crop(20, 361, 144)
    canvas.paste(_c20, (1000, 510), _c20)
except Exception:
    pass
layout["Science_&_Tech"] = [1000, 510, 1361, 654]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_10_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-12/21_icon_Sports_Fitness.png
try:
    _c21 = get_crop(21, 378, 144)
    canvas.paste(_c21, (375, 510), _c21)
except Exception:
    pass
layout["Sports_&_Fitness"] = [375, 510, 753, 654]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_10_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-12/22_icon_Festival.png
try:
    _c22 = get_crop(22, 219, 144)
    canvas.paste(_c22, (1122, 1464), _c22)
except Exception:
    pass
layout["Festival"] = [1122, 1464, 1341, 1608]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_10_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-12/23_icon_Home_Lifestyle.png
try:
    _c23 = get_crop(23, 389, 127)
    canvas.paste(_c23, (36, 891), _c23)
except Exception:
    pass
layout["Home_&_Lifestyle"] = [36, 891, 425, 1018]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_10_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-12/24_icon_Charity.png
try:
    _c24 = get_crop(24, 397, 144)
    canvas.paste(_c24, (449, 637), _c24)
except Exception:
    pass
layout["Charity"] = [449, 637, 846, 781]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_10_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-12/25_icon_Hobbies.png
try:
    _c25 = get_crop(25, 231, 144)
    canvas.paste(_c25, (842, 891), _c25)
except Exception:
    pass
layout["Hobbies"] = [842, 891, 1073, 1035]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_10_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-12/26_icon_German.png
try:
    _c26 = get_crop(26, 225, 135)
    canvas.paste(_c26, (270, 1910), _c26)
except Exception:
    pass
layout["German"] = [270, 1910, 495, 2045]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_10_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-12/27_icon_English.png
try:
    _c27 = get_crop(27, 210, 135)
    canvas.paste(_c27, (36, 1910), _c27)
except Exception:
    pass
layout["English"] = [36, 1910, 246, 2045]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_10_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-12/28_icon_Travel_Outdoor.png
try:
    _c28 = get_crop(28, 389, 127)
    canvas.paste(_c28, (36, 637), _c28)
except Exception:
    pass
layout["Travel_&_Outdoor"] = [36, 637, 425, 764]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_10_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-12/29_icon_Conference.png
try:
    _c29 = get_crop(29, 298, 135)
    canvas.paste(_c29, (36, 1464), _c29)
except Exception:
    pass
layout["Conference"] = [36, 1464, 334, 1599]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_10_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-12/30_icon_School_Activities.png
try:
    _c30 = get_crop(30, 392, 135)
    canvas.paste(_c30, (36, 1018), _c30)
except Exception:
    pass
layout["School_Activities"] = [36, 1018, 428, 1153]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_10_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-12/31_icon_Clear_all.png
try:
    _c31 = get_crop(31, 51, 67)
    canvas.paste(_c31, (1153, 1), _c31)
except Exception:
    pass
layout["Clear_all"] = [1153, 1, 1204, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_10_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-12/32_icon_Clear_all.png
try:
    _c32 = get_crop(32, 100, 64)
    canvas.paste(_c32, (1211, 1), _c32)
except Exception:
    pass
layout["Clear_all"] = [1211, 1, 1311, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_10_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-12/33_icon_9.42.png
try:
    _c33 = get_crop(33, 144, 144)
    canvas.paste(_c33, (12, 72), _c33)
except Exception:
    pass
layout["9.42"] = [12, 72, 156, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_10_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-12/34_icon_icon_34.png
try:
    _c34 = get_crop(34, 53, 60)
    canvas.paste(_c34, (315, 3), _c34)
except Exception:
    pass
layout["icon_34"] = [315, 3, 368, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_10_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-12/35_icon_Clear_all.png
try:
    _c35 = get_crop(35, 52, 63)
    canvas.paste(_c35, (1319, 1), _c35)
except Exception:
    pass
layout["Clear_all"] = [1319, 1, 1371, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_10_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-12/36_icon_9.42.png
try:
    _c36 = get_crop(36, 62, 62)
    canvas.paste(_c36, (178, 1), _c36)
except Exception:
    pass
layout["9.42"] = [178, 1, 240, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_10_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-12/37_icon_9.42.png
try:
    _c37 = get_crop(37, 55, 63)
    canvas.paste(_c37, (114, 1), _c37)
except Exception:
    pass
layout["9.42"] = [114, 1, 169, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_10_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-12/38_icon_icon_38.png
try:
    _c38 = get_crop(38, 58, 61)
    canvas.paste(_c38, (245, 2), _c38)
except Exception:
    pass
layout["icon_38"] = [245, 2, 303, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_10_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-12/39_icon_Clear_all.png
try:
    _c39 = get_crop(39, 178, 144)
    canvas.paste(_c39, (1214, 72), _c39)
except Exception:
    pass
layout["Clear_all"] = [1214, 72, 1392, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_10_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-12/40_icon_clickable_35.png
try:
    _c40 = get_crop(40, 144, 144)
    canvas.paste(_c40, (1248, 2364), _c40)
except Exception:
    pass
layout["clickable_35"] = [1248, 2364, 1392, 2508]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_10_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-12/41_text_9.42.png
try:
    _c41 = get_crop(41, 91, 43)
    canvas.paste(_c41, (20, 15), _c41)
except Exception:
    pass
layout["9.42"] = [20, 15, 111, 58]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_10_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-12/42_text_Filters.png
try:
    _c42 = get_crop(42, 180, 66)
    canvas.paste(_c42, (631, 116), _c42)
except Exception:
    pass
layout["Filters"] = [631, 116, 811, 182]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_10_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-12/43_text_Categories.png
try:
    _c43 = get_crop(43, 187, 127)
    canvas.paste(_c43, (36, 383), _c43)
except Exception:
    pass
layout["Categories"] = [36, 383, 223, 510]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_10_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-12/44_text_Show_less_categories.png
try:
    _c44 = get_crop(44, 550, 144)
    canvas.paste(_c44, (0, 1153), _c44)
except Exception:
    pass
layout["Show_less_categories"] = [0, 1153, 550, 1297]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_10_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-12/45_text_Event_type.png
try:
    _c45 = get_crop(45, 298, 135)
    canvas.paste(_c45, (36, 1464), _c45)
except Exception:
    pass
layout["Event_type"] = [36, 1464, 334, 1599]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_10_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-12/46_text_Show_all_event_types.png
try:
    _c46 = get_crop(46, 535, 144)
    canvas.paste(_c46, (0, 1599), _c46)
except Exception:
    pass
layout["Show_all_event_types"] = [0, 1599, 535, 1743]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_10_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-12/47_text_Languages.png
try:
    _c47 = get_crop(47, 210, 135)
    canvas.paste(_c47, (36, 1910), _c47)
except Exception:
    pass
layout["Languages"] = [36, 1910, 246, 2045]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_10_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-12/48_text_Show_all_languages.png
try:
    _c48 = get_crop(48, 511, 144)
    canvas.paste(_c48, (0, 2045), _c48)
except Exception:
    pass
layout["Show_all_languages"] = [0, 2045, 511, 2189]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_10_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-12/49_text_Price.png
try:
    _c49 = get_crop(49, 149, 63)
    canvas.paste(_c49, (45, 2249), _c49)
except Exception:
    pass
layout["Price"] = [45, 2249, 194, 2312]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_10_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-12/50_text_Only_free_events.png
try:
    _c50 = get_crop(50, 660, 61)
    canvas.paste(_c50, (54, 2659), _c50)
except Exception:
    pass
layout["Only_free_events"] = [54, 2659, 714, 2720]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_10_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-12/51_text_Sort_by.png
try:
    _c51 = get_crop(51, 206, 75)
    canvas.paste(_c51, (42, 2567), _c51)
except Exception:
    pass
layout["Sort_by"] = [42, 2567, 248, 2642]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_10_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-12/52_clickable_Date.png
try:
    _c52 = get_crop(52, 660, 61)
    canvas.paste(_c52, (726, 2659), _c52)
except Exception:
    pass
layout["Date"] = [726, 2659, 1386, 2720]
