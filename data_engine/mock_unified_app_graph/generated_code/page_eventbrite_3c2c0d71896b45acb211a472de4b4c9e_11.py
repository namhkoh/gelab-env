# page_id: page_eventbrite_3c2c0d71896b45acb211a472de4b4c9e_11
# screenshot: 2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-13.png
# step_index: 11/15
# task: Open Eventbrite. Search free Health event in Los Angeles. Select the first one that is not promoted. Follow the organizer. Share to google keep notes.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Canvas and draw are provided (1440x2960 RGB white)
# Draw overall subtle off-white background to match the app's canvas
bg = (250, 251, 253)          # Very light off-white
status_bar_bg = (209, 211, 213)  # light gray for status bar
header_divider = (232, 234, 238) # subtle divider
section_divider = (235, 236, 240) # section separators
panel_bg = (246, 247, 250)     # very light panel background
bottom_sheet = (245, 244, 248) # faint bottom sheet color

# Fill full canvas background
draw.rectangle([(0, 0), (1440, 2960)], fill=bg)

# Status bar area (top) - don't draw icons/text, only background
status_h = 96
draw.rectangle([(0, 0), (1440, status_h)], fill=status_bar_bg)

# Header / toolbar area below status bar
header_top = status_h
header_bottom = 168
draw.rectangle([(0, header_top), (1440, header_bottom)], fill=bg)
# subtle bottom divider under header
draw.line([(24, header_bottom), (1440-24, header_bottom)], fill=header_divider, width=1)

# Draw faint shadow under header (very subtle)
shadow_y = header_bottom + 1
draw.line([(24, shadow_y), (1440-24, shadow_y)], fill=(245,245,246), width=1)

# Section separators (thin lines) between the major filter groups
# Positions chosen to align with section boundaries in the UI
separators = [1280, 1620, 2060, 2260, 2640]  # y positions for separators
for y in separators:
    draw.line([(36, y), (1440-36, y)], fill=section_divider, width=1)

# Draw subtle rounded card backgrounds for some grouped areas (no icons/text)
# 1) A very light rounded area behind the "Categories" group
cat_box = (24, 320, 1440-24, 1220)
draw.rounded_rectangle(cat_box, radius=12, fill=panel_bg, outline=None)

# 2) A light rounded area behind the "Event type" row
etype_box = (24, 1400, 1440-24, 1560)
draw.rounded_rectangle(etype_box, radius=12, fill=panel_bg, outline=None)

# 3) A light rounded area behind the "Languages" row
lang_box = (24, 1860, 1440-24, 1960)
draw.rounded_rectangle(lang_box, radius=12, fill=panel_bg, outline=None)

# 4) A faint rounded sheet above the bottom action area (but do NOT overlap the apply-button area)
# Keep this sheet above y=2768 (apply filters top) to avoid drawing over the detected apply-button.
bottom_sheet_top = 2360
bottom_sheet_bottom = 2720  # stays above 2768
draw.rounded_rectangle((16, bottom_sheet_top, 1440-16, bottom_sheet_bottom), radius=18, fill=bottom_sheet, outline=(230,230,235))

# Add a subtle top divider for the bottom sheet to separate content
draw.line([(36, bottom_sheet_top+8), (1440-36, bottom_sheet_top+8)], fill=(235,235,239), width=1)

# Decorative inner divider lines (soft) to indicate grouping within long content
# Place a few faint vertical alignment guides (subtle)
for x in (36, 1440-36):
    draw.line([(x, header_bottom+12), (x, 2360)], fill=(250,250,251), width=1)

# Add a very subtle vignette top edge to the entire screen (soft)
draw.rectangle([(0, 0), (1440, 6)], fill=(248,248,249))

# End of drawing background and structural elements.
# (Do not draw any icons, buttons, or text — those will be pasted on top by the detector pipeline.)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_11_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-13/00_icon_Health.png
try:
    _c0 = get_crop(0, 199, 144)
    canvas.paste(_c0, (777, 510), _c0)
except Exception:
    pass
layout["Health"] = [777, 510, 976, 654]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_11_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-13/01_icon_Apply_filters_2.png
try:
    _c1 = get_crop(1, 1344, 144)
    canvas.paste(_c1, (48, 2768), _c1)
except Exception:
    pass
layout["Apply_filters_(2)"] = [48, 2768, 1392, 2912]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_11_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-13/02_icon_Food_Drink.png
try:
    _c2 = get_crop(2, 312, 144)
    canvas.paste(_c2, (512, 383), _c2)
except Exception:
    pass
layout["Food_&_Drink"] = [512, 383, 824, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_11_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-13/03_icon_Music.png
try:
    _c3 = get_crop(3, 187, 127)
    canvas.paste(_c3, (36, 383), _c3)
except Exception:
    pass
layout["Music"] = [36, 383, 223, 510]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_11_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-13/04_icon_Community.png
try:
    _c4 = get_crop(4, 294, 144)
    canvas.paste(_c4, (848, 383), _c4)
except Exception:
    pass
layout["Community"] = [848, 383, 1142, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_11_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-13/05_icon_Spirituality.png
try:
    _c5 = get_crop(5, 282, 144)
    canvas.paste(_c5, (870, 637), _c5)
except Exception:
    pass
layout["Spirituality"] = [870, 637, 1152, 781]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_11_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-13/06_icon_Government.png
try:
    _c6 = get_crop(6, 310, 144)
    canvas.paste(_c6, (734, 764), _c6)
except Exception:
    pass
layout["Government"] = [734, 764, 1044, 908]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_11_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-13/07_icon_Business.png
try:
    _c7 = get_crop(7, 241, 144)
    canvas.paste(_c7, (247, 383), _c7)
except Exception:
    pass
layout["Business"] = [247, 383, 488, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_11_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-13/08_icon_Expo.png
try:
    _c8 = get_crop(8, 167, 144)
    canvas.paste(_c8, (614, 1464), _c8)
except Exception:
    pass
layout["Expo"] = [614, 1464, 781, 1608]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_11_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-13/09_icon_Holiday.png
try:
    _c9 = get_crop(9, 218, 127)
    canvas.paste(_c9, (492, 764), _c9)
except Exception:
    pass
layout["Holiday"] = [492, 764, 710, 891]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_11_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-13/10_icon_Fashion.png
try:
    _c10 = get_crop(10, 220, 144)
    canvas.paste(_c10, (1068, 764), _c10)
except Exception:
    pass
layout["Fashion"] = [1068, 764, 1288, 908]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_11_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-13/11_icon_French.png
try:
    _c11 = get_crop(11, 205, 144)
    canvas.paste(_c11, (768, 1910), _c11)
except Exception:
    pass
layout["French"] = [768, 1910, 973, 2054]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_11_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-13/12_icon_Arts.png
try:
    _c12 = get_crop(12, 152, 127)
    canvas.paste(_c12, (1166, 383), _c12)
except Exception:
    pass
layout["Arts"] = [1166, 383, 1318, 510]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_11_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-13/13_icon_Spanish.png
try:
    _c13 = get_crop(13, 225, 144)
    canvas.paste(_c13, (519, 1910), _c13)
except Exception:
    pass
layout["Spanish"] = [519, 1910, 744, 2054]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_11_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-13/14_icon_Auto_Boat_Air.png
try:
    _c14 = get_crop(14, 369, 144)
    canvas.paste(_c14, (449, 891), _c14)
except Exception:
    pass
layout["Auto,_Boat_&_Air"] = [449, 891, 818, 1035]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_11_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-13/15_icon_Italian.png
try:
    _c15 = get_crop(15, 191, 144)
    canvas.paste(_c15, (997, 1910), _c15)
except Exception:
    pass
layout["Italian"] = [997, 1910, 1188, 2054]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_11_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-13/16_icon_Seminar.png
try:
    _c16 = get_crop(16, 232, 144)
    canvas.paste(_c16, (358, 1464), _c16)
except Exception:
    pass
layout["Seminar"] = [358, 1464, 590, 1608]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_11_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-13/17_icon_Film_Media.png
try:
    _c17 = get_crop(17, 315, 127)
    canvas.paste(_c17, (36, 510), _c17)
except Exception:
    pass
layout["Film_&_Media"] = [36, 510, 351, 637]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_11_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-13/18_icon_Family_Education.png
try:
    _c18 = get_crop(18, 432, 144)
    canvas.paste(_c18, (36, 764), _c18)
except Exception:
    pass
layout["Family_&_Education"] = [36, 764, 468, 908]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_11_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-13/19_icon_Convention.png
try:
    _c19 = get_crop(19, 293, 144)
    canvas.paste(_c19, (805, 1464), _c19)
except Exception:
    pass
layout["Convention"] = [805, 1464, 1098, 1608]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_11_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-13/20_icon_Science_Tech.png
try:
    _c20 = get_crop(20, 361, 144)
    canvas.paste(_c20, (1000, 510), _c20)
except Exception:
    pass
layout["Science_&_Tech"] = [1000, 510, 1361, 654]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_11_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-13/21_icon_Sports_Fitness.png
try:
    _c21 = get_crop(21, 378, 144)
    canvas.paste(_c21, (375, 510), _c21)
except Exception:
    pass
layout["Sports_&_Fitness"] = [375, 510, 753, 654]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_11_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-13/22_icon_Charity.png
try:
    _c22 = get_crop(22, 397, 144)
    canvas.paste(_c22, (449, 637), _c22)
except Exception:
    pass
layout["Charity"] = [449, 637, 846, 781]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_11_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-13/23_icon_Festival.png
try:
    _c23 = get_crop(23, 219, 144)
    canvas.paste(_c23, (1122, 1464), _c23)
except Exception:
    pass
layout["Festival"] = [1122, 1464, 1341, 1608]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_11_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-13/24_icon_Hobbies.png
try:
    _c24 = get_crop(24, 231, 144)
    canvas.paste(_c24, (842, 891), _c24)
except Exception:
    pass
layout["Hobbies"] = [842, 891, 1073, 1035]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_11_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-13/25_icon_Home_Lifestyle.png
try:
    _c25 = get_crop(25, 389, 127)
    canvas.paste(_c25, (36, 891), _c25)
except Exception:
    pass
layout["Home_&_Lifestyle"] = [36, 891, 425, 1018]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_11_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-13/26_icon_German.png
try:
    _c26 = get_crop(26, 225, 135)
    canvas.paste(_c26, (270, 1910), _c26)
except Exception:
    pass
layout["German"] = [270, 1910, 495, 2045]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_11_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-13/27_icon_English.png
try:
    _c27 = get_crop(27, 210, 135)
    canvas.paste(_c27, (36, 1910), _c27)
except Exception:
    pass
layout["English"] = [36, 1910, 246, 2045]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_11_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-13/28_icon_Travel_Outdoor.png
try:
    _c28 = get_crop(28, 389, 127)
    canvas.paste(_c28, (36, 637), _c28)
except Exception:
    pass
layout["Travel_&_Outdoor"] = [36, 637, 425, 764]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_11_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-13/29_icon_Conference.png
try:
    _c29 = get_crop(29, 298, 135)
    canvas.paste(_c29, (36, 1464), _c29)
except Exception:
    pass
layout["Conference"] = [36, 1464, 334, 1599]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_11_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-13/30_icon_School_Activities.png
try:
    _c30 = get_crop(30, 392, 135)
    canvas.paste(_c30, (36, 1018), _c30)
except Exception:
    pass
layout["School_Activities"] = [36, 1018, 428, 1153]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_11_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-13/31_icon_Clear_all.png
try:
    _c31 = get_crop(31, 52, 67)
    canvas.paste(_c31, (1153, 1), _c31)
except Exception:
    pass
layout["Clear_all"] = [1153, 1, 1205, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_11_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-13/32_icon_Clear_all.png
try:
    _c32 = get_crop(32, 99, 64)
    canvas.paste(_c32, (1211, 1), _c32)
except Exception:
    pass
layout["Clear_all"] = [1211, 1, 1310, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_11_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-13/33_icon_icon_33.png
try:
    _c33 = get_crop(33, 53, 60)
    canvas.paste(_c33, (315, 3), _c33)
except Exception:
    pass
layout["icon_33"] = [315, 3, 368, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_11_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-13/34_icon_9.42.png
try:
    _c34 = get_crop(34, 144, 144)
    canvas.paste(_c34, (12, 72), _c34)
except Exception:
    pass
layout["9.42"] = [12, 72, 156, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_11_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-13/35_icon_Clear_all.png
try:
    _c35 = get_crop(35, 53, 63)
    canvas.paste(_c35, (1319, 1), _c35)
except Exception:
    pass
layout["Clear_all"] = [1319, 1, 1372, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_11_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-13/36_icon_9.42.png
try:
    _c36 = get_crop(36, 62, 62)
    canvas.paste(_c36, (178, 1), _c36)
except Exception:
    pass
layout["9.42"] = [178, 1, 240, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_11_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-13/37_icon_9.42.png
try:
    _c37 = get_crop(37, 55, 63)
    canvas.paste(_c37, (114, 1), _c37)
except Exception:
    pass
layout["9.42"] = [114, 1, 169, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_11_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-13/38_icon_icon_38.png
try:
    _c38 = get_crop(38, 55, 60)
    canvas.paste(_c38, (247, 3), _c38)
except Exception:
    pass
layout["icon_38"] = [247, 3, 302, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_11_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-13/39_icon_Clear_all.png
try:
    _c39 = get_crop(39, 178, 144)
    canvas.paste(_c39, (1214, 72), _c39)
except Exception:
    pass
layout["Clear_all"] = [1214, 72, 1392, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_11_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-13/40_icon_clickable_35.png
try:
    _c40 = get_crop(40, 144, 144)
    canvas.paste(_c40, (1248, 2364), _c40)
except Exception:
    pass
layout["clickable_35"] = [1248, 2364, 1392, 2508]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_11_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-13/41_text_9.42.png
try:
    _c41 = get_crop(41, 91, 43)
    canvas.paste(_c41, (20, 15), _c41)
except Exception:
    pass
layout["9.42"] = [20, 15, 111, 58]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_11_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-13/42_text_Filters.png
try:
    _c42 = get_crop(42, 180, 66)
    canvas.paste(_c42, (631, 116), _c42)
except Exception:
    pass
layout["Filters"] = [631, 116, 811, 182]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_11_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-13/43_text_Categories.png
try:
    _c43 = get_crop(43, 187, 127)
    canvas.paste(_c43, (36, 383), _c43)
except Exception:
    pass
layout["Categories"] = [36, 383, 223, 510]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_11_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-13/44_text_Show_less_categories.png
try:
    _c44 = get_crop(44, 550, 144)
    canvas.paste(_c44, (0, 1153), _c44)
except Exception:
    pass
layout["Show_less_categories"] = [0, 1153, 550, 1297]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_11_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-13/45_text_Event_type.png
try:
    _c45 = get_crop(45, 298, 135)
    canvas.paste(_c45, (36, 1464), _c45)
except Exception:
    pass
layout["Event_type"] = [36, 1464, 334, 1599]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_11_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-13/46_text_Show_all_event_types.png
try:
    _c46 = get_crop(46, 535, 144)
    canvas.paste(_c46, (0, 1599), _c46)
except Exception:
    pass
layout["Show_all_event_types"] = [0, 1599, 535, 1743]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_11_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-13/47_text_Languages.png
try:
    _c47 = get_crop(47, 210, 135)
    canvas.paste(_c47, (36, 1910), _c47)
except Exception:
    pass
layout["Languages"] = [36, 1910, 246, 2045]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_11_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-13/48_text_Show_all_languages.png
try:
    _c48 = get_crop(48, 511, 144)
    canvas.paste(_c48, (0, 2045), _c48)
except Exception:
    pass
layout["Show_all_languages"] = [0, 2045, 511, 2189]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_11_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-13/49_text_Price.png
try:
    _c49 = get_crop(49, 149, 63)
    canvas.paste(_c49, (45, 2249), _c49)
except Exception:
    pass
layout["Price"] = [45, 2249, 194, 2312]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_11_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-13/50_text_Only_free_events.png
try:
    _c50 = get_crop(50, 660, 61)
    canvas.paste(_c50, (54, 2659), _c50)
except Exception:
    pass
layout["Only_free_events"] = [54, 2659, 714, 2720]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_11_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-13/51_text_Sort_by.png
try:
    _c51 = get_crop(51, 206, 75)
    canvas.paste(_c51, (42, 2567), _c51)
except Exception:
    pass
layout["Sort_by"] = [42, 2567, 248, 2642]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_11_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-13/52_clickable_Date.png
try:
    _c52 = get_crop(52, 660, 61)
    canvas.paste(_c52, (726, 2659), _c52)
except Exception:
    pass
layout["Date"] = [726, 2659, 1386, 2720]
