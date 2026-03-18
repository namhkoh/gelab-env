# page_id: page_eventbrite_92c22920a83749c994864397a370a984_10
# screenshot: 2024_4_24_16_59_92c22920a83749c994864397a370a984-12.png
# step_index: 10/13
# task: Open Eventbrite. Set the city to "Chicago". Select the "Sports" category and view the recommended events. See the date of the first non-promoted event.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw background and structural UI elements for the Filters screen
# Uses provided variables: canvas (PIL Image), draw (ImageDraw)

# Colors
bg_white = (255, 255, 255)
status_bar_gray = (198, 198, 198)
divider_light = (238, 238, 242)
divider_subtle = (245, 245, 247)
shadow_light = (243, 243, 246)
muted_gray = (230, 230, 235)

w, h = canvas.size

# Ensure full canvas background
draw.rectangle([0, 0, w, h], fill=bg_white)

# Status bar area (top)
status_h = 72
draw.rectangle([0, 0, w, status_h], fill=status_bar_gray)

# Header / toolbar area (below status bar)
toolbar_top = status_h
toolbar_h = 80
toolbar_bottom = toolbar_top + toolbar_h
draw.rectangle([0, toolbar_top, w, toolbar_bottom], fill=bg_white)

# Thin bottom divider under toolbar
draw.line([(24, toolbar_bottom), (w-24, toolbar_bottom)], fill=divider_light, width=1)

# Subtle horizontal separators between major sections
# Position separators roughly between the detected section blocks
separators = [1360, 1750, 2100, 2420]
for y in separators:
    draw.line([(36, y), (w-36, y)], fill=divider_subtle, width=1)

# Light background band behind the bottom area (floating apply bar area shadow)
# This provides the structural dorsal background without drawing the actual button
bottom_band_top = 2660
bottom_band_bottom = h
draw.rectangle([0, bottom_band_top, w, bottom_band_bottom], fill=shadow_light)

# Subtle rounded top shadow above the apply bar location
# Slightly above the detected apply button (detected at y ~2768), draw a faint rounded bar
shadow_box = [48, 2688, w-48, 2736]
try:
    draw.rounded_rectangle(shadow_box, radius=12, fill=muted_gray, outline=None)
except Exception:
    # fallback if rounded_rectangle not supported
    draw.rectangle(shadow_box, fill=muted_gray)

# Section card suggestion: draw faint separators (vertical margins) for content flow
# Left and right content margins as faint vertical guides
margin_x = 36
draw.line([(margin_x, toolbar_bottom+12), (margin_x, h-bottom_band_top)], fill=divider_subtle, width=1)
draw.line([(w-margin_x, toolbar_bottom+12), (w-margin_x, h-bottom_band_top)], fill=divider_subtle, width=1)

# Provide subtle group background behind the top categories region (light, not overlapping chips)
# This is a very faint rounded rectangle to hint grouping without duplicating chip shapes
cat_group_top = 340
cat_group_bottom = 1220
cat_group_box = [24, cat_group_top, w-24, cat_group_bottom]
try:
    draw.rounded_rectangle(cat_group_box, radius=22, fill=(255,255,255), outline=divider_subtle, width=1)
except Exception:
    draw.rectangle(cat_group_box, outline=divider_subtle, fill=(255,255,255))

# Small divider line under the "Show less categories" area to separate from event type
draw.line([(36, 1216), (w-36, 1216)], fill=divider_light, width=1)

# Subtle background for the event-type chips row (so chips pasted on top have a slight halo)
evt_group_top = 1420
evt_group_bottom = 1600
try:
    draw.rounded_rectangle([36, evt_group_top, w-36, evt_group_bottom], radius=18, fill=(255,255,255), outline=divider_subtle, width=1)
except Exception:
    draw.rectangle([36, evt_group_top, w-36, evt_group_bottom], fill=(255,255,255), outline=divider_subtle)

# Subtle background area for languages row
lang_group_top = 1850
lang_group_bottom = 2060
try:
    draw.rounded_rectangle([36, lang_group_top, w-36, lang_group_bottom], radius=18, fill=(255,255,255), outline=divider_subtle, width=1)
except Exception:
    draw.rectangle([36, lang_group_top, w-36, lang_group_bottom], fill=(255,255,255), outline=divider_subtle)

# Final top hairline above the very bottom band to separate content from the floating bar area
draw.line([(24, bottom_band_top), (w-24, bottom_band_top)], fill=divider_light, width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_10_2024_4_24_16_59_92c22920a83749c994864397a370a984-12/00_icon_Sports_Fitness.png
try:
    _c0 = get_crop(0, 378, 144)
    canvas.paste(_c0, (375, 510), _c0)
except Exception:
    pass
layout["Sports_&_Fitness"] = [375, 510, 753, 654]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_10_2024_4_24_16_59_92c22920a83749c994864397a370a984-12/01_icon_Health.png
try:
    _c1 = get_crop(1, 199, 144)
    canvas.paste(_c1, (777, 510), _c1)
except Exception:
    pass
layout["Health"] = [777, 510, 976, 654]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_10_2024_4_24_16_59_92c22920a83749c994864397a370a984-12/02_icon_Music.png
try:
    _c2 = get_crop(2, 187, 127)
    canvas.paste(_c2, (36, 383), _c2)
except Exception:
    pass
layout["Music"] = [36, 383, 223, 510]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_10_2024_4_24_16_59_92c22920a83749c994864397a370a984-12/03_icon_Food_Drink.png
try:
    _c3 = get_crop(3, 312, 144)
    canvas.paste(_c3, (512, 383), _c3)
except Exception:
    pass
layout["Food_&_Drink"] = [512, 383, 824, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_10_2024_4_24_16_59_92c22920a83749c994864397a370a984-12/04_icon_Government.png
try:
    _c4 = get_crop(4, 310, 144)
    canvas.paste(_c4, (734, 764), _c4)
except Exception:
    pass
layout["Government"] = [734, 764, 1044, 908]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_10_2024_4_24_16_59_92c22920a83749c994864397a370a984-12/05_icon_Community.png
try:
    _c5 = get_crop(5, 294, 144)
    canvas.paste(_c5, (848, 383), _c5)
except Exception:
    pass
layout["Community"] = [848, 383, 1142, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_10_2024_4_24_16_59_92c22920a83749c994864397a370a984-12/06_icon_Fashion.png
try:
    _c6 = get_crop(6, 220, 144)
    canvas.paste(_c6, (1068, 764), _c6)
except Exception:
    pass
layout["Fashion"] = [1068, 764, 1288, 908]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_10_2024_4_24_16_59_92c22920a83749c994864397a370a984-12/07_icon_Expo.png
try:
    _c7 = get_crop(7, 167, 144)
    canvas.paste(_c7, (614, 1464), _c7)
except Exception:
    pass
layout["Expo"] = [614, 1464, 781, 1608]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_10_2024_4_24_16_59_92c22920a83749c994864397a370a984-12/08_icon_Holiday.png
try:
    _c8 = get_crop(8, 218, 127)
    canvas.paste(_c8, (492, 764), _c8)
except Exception:
    pass
layout["Holiday"] = [492, 764, 710, 891]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_10_2024_4_24_16_59_92c22920a83749c994864397a370a984-12/09_icon_Business.png
try:
    _c9 = get_crop(9, 241, 144)
    canvas.paste(_c9, (247, 383), _c9)
except Exception:
    pass
layout["Business"] = [247, 383, 488, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_10_2024_4_24_16_59_92c22920a83749c994864397a370a984-12/10_icon_Spirituality.png
try:
    _c10 = get_crop(10, 282, 144)
    canvas.paste(_c10, (870, 637), _c10)
except Exception:
    pass
layout["Spirituality"] = [870, 637, 1152, 781]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_10_2024_4_24_16_59_92c22920a83749c994864397a370a984-12/11_icon_Apply_filters_1.png
try:
    _c11 = get_crop(11, 1344, 144)
    canvas.paste(_c11, (48, 2768), _c11)
except Exception:
    pass
layout["Apply_filters_(1)"] = [48, 2768, 1392, 2912]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_10_2024_4_24_16_59_92c22920a83749c994864397a370a984-12/12_icon_Spanish.png
try:
    _c12 = get_crop(12, 225, 144)
    canvas.paste(_c12, (519, 1910), _c12)
except Exception:
    pass
layout["Spanish"] = [519, 1910, 744, 2054]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_10_2024_4_24_16_59_92c22920a83749c994864397a370a984-12/13_icon_Arts.png
try:
    _c13 = get_crop(13, 152, 127)
    canvas.paste(_c13, (1166, 383), _c13)
except Exception:
    pass
layout["Arts"] = [1166, 383, 1318, 510]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_10_2024_4_24_16_59_92c22920a83749c994864397a370a984-12/14_icon_French.png
try:
    _c14 = get_crop(14, 205, 144)
    canvas.paste(_c14, (768, 1910), _c14)
except Exception:
    pass
layout["French"] = [768, 1910, 973, 2054]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_10_2024_4_24_16_59_92c22920a83749c994864397a370a984-12/15_icon_Auto_Boat_Air.png
try:
    _c15 = get_crop(15, 369, 144)
    canvas.paste(_c15, (449, 891), _c15)
except Exception:
    pass
layout["Auto,_Boat_&_Air"] = [449, 891, 818, 1035]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_10_2024_4_24_16_59_92c22920a83749c994864397a370a984-12/16_icon_Italian.png
try:
    _c16 = get_crop(16, 191, 144)
    canvas.paste(_c16, (997, 1910), _c16)
except Exception:
    pass
layout["Italian"] = [997, 1910, 1188, 2054]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_10_2024_4_24_16_59_92c22920a83749c994864397a370a984-12/17_icon_Seminar.png
try:
    _c17 = get_crop(17, 232, 144)
    canvas.paste(_c17, (358, 1464), _c17)
except Exception:
    pass
layout["Seminar"] = [358, 1464, 590, 1608]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_10_2024_4_24_16_59_92c22920a83749c994864397a370a984-12/18_icon_Family_Education.png
try:
    _c18 = get_crop(18, 432, 144)
    canvas.paste(_c18, (36, 764), _c18)
except Exception:
    pass
layout["Family_&_Education"] = [36, 764, 468, 908]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_10_2024_4_24_16_59_92c22920a83749c994864397a370a984-12/19_icon_Convention.png
try:
    _c19 = get_crop(19, 293, 144)
    canvas.paste(_c19, (805, 1464), _c19)
except Exception:
    pass
layout["Convention"] = [805, 1464, 1098, 1608]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_10_2024_4_24_16_59_92c22920a83749c994864397a370a984-12/20_icon_Charity.png
try:
    _c20 = get_crop(20, 397, 144)
    canvas.paste(_c20, (449, 637), _c20)
except Exception:
    pass
layout["Charity"] = [449, 637, 846, 781]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_10_2024_4_24_16_59_92c22920a83749c994864397a370a984-12/21_icon_Film_Media.png
try:
    _c21 = get_crop(21, 315, 127)
    canvas.paste(_c21, (36, 510), _c21)
except Exception:
    pass
layout["Film_&_Media"] = [36, 510, 351, 637]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_10_2024_4_24_16_59_92c22920a83749c994864397a370a984-12/22_icon_Science_Tech.png
try:
    _c22 = get_crop(22, 361, 144)
    canvas.paste(_c22, (1000, 510), _c22)
except Exception:
    pass
layout["Science_&_Tech"] = [1000, 510, 1361, 654]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_10_2024_4_24_16_59_92c22920a83749c994864397a370a984-12/23_icon_Festival.png
try:
    _c23 = get_crop(23, 219, 144)
    canvas.paste(_c23, (1122, 1464), _c23)
except Exception:
    pass
layout["Festival"] = [1122, 1464, 1341, 1608]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_10_2024_4_24_16_59_92c22920a83749c994864397a370a984-12/24_icon_Hobbies.png
try:
    _c24 = get_crop(24, 231, 144)
    canvas.paste(_c24, (842, 891), _c24)
except Exception:
    pass
layout["Hobbies"] = [842, 891, 1073, 1035]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_10_2024_4_24_16_59_92c22920a83749c994864397a370a984-12/25_icon_German.png
try:
    _c25 = get_crop(25, 225, 135)
    canvas.paste(_c25, (270, 1910), _c25)
except Exception:
    pass
layout["German"] = [270, 1910, 495, 2045]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_10_2024_4_24_16_59_92c22920a83749c994864397a370a984-12/26_icon_Home_Lifestyle.png
try:
    _c26 = get_crop(26, 389, 127)
    canvas.paste(_c26, (36, 891), _c26)
except Exception:
    pass
layout["Home_&_Lifestyle"] = [36, 891, 425, 1018]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_10_2024_4_24_16_59_92c22920a83749c994864397a370a984-12/27_icon_English.png
try:
    _c27 = get_crop(27, 210, 135)
    canvas.paste(_c27, (36, 1910), _c27)
except Exception:
    pass
layout["English"] = [36, 1910, 246, 2045]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_10_2024_4_24_16_59_92c22920a83749c994864397a370a984-12/28_icon_Travel_Outdoor.png
try:
    _c28 = get_crop(28, 389, 127)
    canvas.paste(_c28, (36, 637), _c28)
except Exception:
    pass
layout["Travel_&_Outdoor"] = [36, 637, 425, 764]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_10_2024_4_24_16_59_92c22920a83749c994864397a370a984-12/29_icon_Conference.png
try:
    _c29 = get_crop(29, 298, 135)
    canvas.paste(_c29, (36, 1464), _c29)
except Exception:
    pass
layout["Conference"] = [36, 1464, 334, 1599]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_10_2024_4_24_16_59_92c22920a83749c994864397a370a984-12/30_icon_School_Activities.png
try:
    _c30 = get_crop(30, 392, 135)
    canvas.paste(_c30, (36, 1018), _c30)
except Exception:
    pass
layout["School_Activities"] = [36, 1018, 428, 1153]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_10_2024_4_24_16_59_92c22920a83749c994864397a370a984-12/31_icon_Clear_all.png
try:
    _c31 = get_crop(31, 52, 67)
    canvas.paste(_c31, (1153, 1), _c31)
except Exception:
    pass
layout["Clear_all"] = [1153, 1, 1205, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_10_2024_4_24_16_59_92c22920a83749c994864397a370a984-12/32_icon_5.00.png
try:
    _c32 = get_crop(32, 144, 144)
    canvas.paste(_c32, (12, 72), _c32)
except Exception:
    pass
layout["5.00"] = [12, 72, 156, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_10_2024_4_24_16_59_92c22920a83749c994864397a370a984-12/33_icon_Clear_all.png
try:
    _c33 = get_crop(33, 99, 64)
    canvas.paste(_c33, (1211, 1), _c33)
except Exception:
    pass
layout["Clear_all"] = [1211, 1, 1310, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_10_2024_4_24_16_59_92c22920a83749c994864397a370a984-12/34_icon_5.00.png
try:
    _c34 = get_crop(34, 63, 64)
    canvas.paste(_c34, (112, 1), _c34)
except Exception:
    pass
layout["5.00"] = [112, 1, 175, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_10_2024_4_24_16_59_92c22920a83749c994864397a370a984-12/35_icon_Clear_all.png
try:
    _c35 = get_crop(35, 53, 63)
    canvas.paste(_c35, (1319, 1), _c35)
except Exception:
    pass
layout["Clear_all"] = [1319, 1, 1372, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_10_2024_4_24_16_59_92c22920a83749c994864397a370a984-12/36_icon_5.00.png
try:
    _c36 = get_crop(36, 60, 62)
    canvas.paste(_c36, (180, 1), _c36)
except Exception:
    pass
layout["5.00"] = [180, 1, 240, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_10_2024_4_24_16_59_92c22920a83749c994864397a370a984-12/37_icon_icon_37.png
try:
    _c37 = get_crop(37, 64, 61)
    canvas.paste(_c37, (308, 3), _c37)
except Exception:
    pass
layout["icon_37"] = [308, 3, 372, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_10_2024_4_24_16_59_92c22920a83749c994864397a370a984-12/38_icon_icon_38.png
try:
    _c38 = get_crop(38, 51, 61)
    canvas.paste(_c38, (249, 2), _c38)
except Exception:
    pass
layout["icon_38"] = [249, 2, 300, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_10_2024_4_24_16_59_92c22920a83749c994864397a370a984-12/39_icon_Clear_all.png
try:
    _c39 = get_crop(39, 178, 144)
    canvas.paste(_c39, (1214, 72), _c39)
except Exception:
    pass
layout["Clear_all"] = [1214, 72, 1392, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_10_2024_4_24_16_59_92c22920a83749c994864397a370a984-12/40_icon_clickable_35.png
try:
    _c40 = get_crop(40, 144, 144)
    canvas.paste(_c40, (1248, 2364), _c40)
except Exception:
    pass
layout["clickable_35"] = [1248, 2364, 1392, 2508]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_10_2024_4_24_16_59_92c22920a83749c994864397a370a984-12/41_text_5.00.png
try:
    _c41 = get_crop(41, 91, 45)
    canvas.paste(_c41, (20, 15), _c41)
except Exception:
    pass
layout["5.00"] = [20, 15, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_10_2024_4_24_16_59_92c22920a83749c994864397a370a984-12/42_text_Filters.png
try:
    _c42 = get_crop(42, 180, 66)
    canvas.paste(_c42, (631, 116), _c42)
except Exception:
    pass
layout["Filters"] = [631, 116, 811, 182]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_10_2024_4_24_16_59_92c22920a83749c994864397a370a984-12/43_text_Categories.png
try:
    _c43 = get_crop(43, 187, 127)
    canvas.paste(_c43, (36, 383), _c43)
except Exception:
    pass
layout["Categories"] = [36, 383, 223, 510]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_10_2024_4_24_16_59_92c22920a83749c994864397a370a984-12/44_text_Show_less_categories.png
try:
    _c44 = get_crop(44, 550, 144)
    canvas.paste(_c44, (0, 1153), _c44)
except Exception:
    pass
layout["Show_less_categories"] = [0, 1153, 550, 1297]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_10_2024_4_24_16_59_92c22920a83749c994864397a370a984-12/45_text_Event_type.png
try:
    _c45 = get_crop(45, 298, 135)
    canvas.paste(_c45, (36, 1464), _c45)
except Exception:
    pass
layout["Event_type"] = [36, 1464, 334, 1599]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_10_2024_4_24_16_59_92c22920a83749c994864397a370a984-12/46_text_Show_all_event_types.png
try:
    _c46 = get_crop(46, 535, 144)
    canvas.paste(_c46, (0, 1599), _c46)
except Exception:
    pass
layout["Show_all_event_types"] = [0, 1599, 535, 1743]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_10_2024_4_24_16_59_92c22920a83749c994864397a370a984-12/47_text_Languages.png
try:
    _c47 = get_crop(47, 210, 135)
    canvas.paste(_c47, (36, 1910), _c47)
except Exception:
    pass
layout["Languages"] = [36, 1910, 246, 2045]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_10_2024_4_24_16_59_92c22920a83749c994864397a370a984-12/48_text_Show_all_languages.png
try:
    _c48 = get_crop(48, 511, 144)
    canvas.paste(_c48, (0, 2045), _c48)
except Exception:
    pass
layout["Show_all_languages"] = [0, 2045, 511, 2189]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_10_2024_4_24_16_59_92c22920a83749c994864397a370a984-12/49_text_Price.png
try:
    _c49 = get_crop(49, 149, 63)
    canvas.paste(_c49, (45, 2249), _c49)
except Exception:
    pass
layout["Price"] = [45, 2249, 194, 2312]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_10_2024_4_24_16_59_92c22920a83749c994864397a370a984-12/50_text_Only_free_events.png
try:
    _c50 = get_crop(50, 511, 144)
    canvas.paste(_c50, (0, 2045), _c50)
except Exception:
    pass
layout["Only_free_events"] = [0, 2045, 511, 2189]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_10_2024_4_24_16_59_92c22920a83749c994864397a370a984-12/51_text_Sort_by.png
try:
    _c51 = get_crop(51, 206, 75)
    canvas.paste(_c51, (42, 2567), _c51)
except Exception:
    pass
layout["Sort_by"] = [42, 2567, 248, 2642]
