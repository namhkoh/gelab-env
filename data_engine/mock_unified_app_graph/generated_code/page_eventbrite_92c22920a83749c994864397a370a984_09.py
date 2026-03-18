# page_id: page_eventbrite_92c22920a83749c994864397a370a984_09
# screenshot: 2024_4_24_16_59_92c22920a83749c994864397a370a984-11.png
# step_index: 9/13
# task: Open Eventbrite. Set the city to "Chicago". Select the "Sports" category and view the recommended events. See the date of the first non-promoted event.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw UI background and structural elements for the filters page.
# Available variables:
# - canvas: PIL Image (1440x2960 RGB)
# - draw: PIL ImageDraw object
# - font_sm, font_md, font_lg, font_xl

w, h = canvas.size

# Colors (match screenshot tones)
bg_color = "#ffffff"           # page background (white)
status_bar_color = "#cfcfcf"   # top status bar light gray
toolbar_bg = "#ffffff"         # toolbar (keeps white)
divider_color = "#e6e7eb"      # subtle divider lines
section_bg = "#fbfdff"         # very light bluish/white section background
section_border = "#eef1f6"     # faint border for sections
bottom_area_bg = "#fbfbfd"     # subtle bottom area background

# Clear canvas (in case)
draw.rectangle([(0,0),(w,h)], fill=bg_color)

# Status bar (approx 56px tall)
status_h = 56
draw.rectangle([(0,0),(w,status_h)], fill=status_bar_color)

# Toolbar area below status bar (approx 110px tall)
toolbar_h = 110
toolbar_top = status_h
toolbar_bottom = toolbar_top + toolbar_h
draw.rectangle([(0, toolbar_top), (w, toolbar_bottom)], fill=toolbar_bg)
# Toolbar divider / subtle shadow
draw.line([(24, toolbar_bottom+0.5), (w-24, toolbar_bottom+0.5)], fill=divider_color, width=1)

# Define helper for section rounded rectangles
def draw_section(x0, y0, x1, y1, radius=28):
    # background
    draw.rounded_rectangle([(x0, y0), (x1, y1)], radius=radius, fill=section_bg, outline=section_border, width=1)

# Margins and section placements approximated from screenshot layout
left_margin = 36
right_margin = w - 36

# Categories section (covers the large chip area)
cats_top = toolbar_bottom + 36
cats_bottom = 1180
draw_section(left_margin, cats_top-8, right_margin, cats_bottom, radius=28)
# subtle separator below categories
draw.line([(24, cats_bottom+18), (w-24, cats_bottom+18)], fill=divider_color, width=1)

# Event type section (chips around y ~1464)
evt_top = 1408
evt_bottom = 1638
draw_section(left_margin, evt_top-8, right_margin, evt_bottom, radius=28)
draw.line([(24, evt_bottom+12), (w-24, evt_bottom+12)], fill=divider_color, width=1)

# Languages section (around y ~1910)
lang_top = 1850
lang_bottom = 1998
draw_section(left_margin, lang_top-8, right_margin, lang_bottom, radius=28)
draw.line([(24, lang_bottom+12), (w-24, lang_bottom+12)], fill=divider_color, width=1)

# Price section (includes "Only free events" toggle area)
price_top = 2188
price_bottom = 2360
draw_section(left_margin, price_top-8, right_margin, price_bottom, radius=20)
draw.line([(24, price_bottom+12), (w-24, price_bottom+12)], fill=divider_color, width=1)

# Sort by area (just above bottom controls)
sort_top = 2520
sort_bottom = 2700
draw_section(left_margin, sort_top-8, right_margin, sort_bottom, radius=20)
draw.line([(24, sort_bottom+12), (w-24, sort_bottom+12)], fill=divider_color, width=1)

# Bottom area background (below sort controls and behind the apply bar)
bottom_top = 2708
draw.rectangle([(0, bottom_top), (w, h)], fill=bottom_area_bg)
# subtle top divider for bottom area
draw.line([(24, bottom_top+2), (w-24, bottom_top+2)], fill=divider_color, width=1)

# Additional subtle horizontal separators between main logical blocks
sep_positions = [
    cats_bottom + 120,   # between categories and event type "Show less / Event type" area
    evt_bottom + 110,    # between event type and languages
    lang_bottom + 160,   # between languages and price
    price_bottom + 160,  # between price and sort
]
for y in sep_positions:
    if 0 < y < h:
        draw.line([(36, y), (w-36, y)], fill=divider_color, width=1)

# Small header accent line under centered toolbar title area (visual structure only)
accent_y = toolbar_top + toolbar_h//2 + 36
draw.line([(w*0.25, accent_y), (w*0.75, accent_y)], fill="#f2f2f7", width=2)

# Done drawing background and structural elements.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_09_2024_4_24_16_59_92c22920a83749c994864397a370a984-11/00_icon_Food_Drink.png
try:
    _c0 = get_crop(0, 312, 144)
    canvas.paste(_c0, (512, 383), _c0)
except Exception:
    pass
layout["Food_&_Drink"] = [512, 383, 824, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_09_2024_4_24_16_59_92c22920a83749c994864397a370a984-11/01_icon_Music.png
try:
    _c1 = get_crop(1, 187, 127)
    canvas.paste(_c1, (36, 383), _c1)
except Exception:
    pass
layout["Music"] = [36, 383, 223, 510]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_09_2024_4_24_16_59_92c22920a83749c994864397a370a984-11/02_icon_Health.png
try:
    _c2 = get_crop(2, 199, 144)
    canvas.paste(_c2, (777, 510), _c2)
except Exception:
    pass
layout["Health"] = [777, 510, 976, 654]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_09_2024_4_24_16_59_92c22920a83749c994864397a370a984-11/03_icon_Government.png
try:
    _c3 = get_crop(3, 310, 144)
    canvas.paste(_c3, (734, 764), _c3)
except Exception:
    pass
layout["Government"] = [734, 764, 1044, 908]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_09_2024_4_24_16_59_92c22920a83749c994864397a370a984-11/04_icon_Business.png
try:
    _c4 = get_crop(4, 241, 144)
    canvas.paste(_c4, (247, 383), _c4)
except Exception:
    pass
layout["Business"] = [247, 383, 488, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_09_2024_4_24_16_59_92c22920a83749c994864397a370a984-11/05_icon_Community.png
try:
    _c5 = get_crop(5, 294, 144)
    canvas.paste(_c5, (848, 383), _c5)
except Exception:
    pass
layout["Community"] = [848, 383, 1142, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_09_2024_4_24_16_59_92c22920a83749c994864397a370a984-11/06_icon_Expo.png
try:
    _c6 = get_crop(6, 167, 144)
    canvas.paste(_c6, (614, 1464), _c6)
except Exception:
    pass
layout["Expo"] = [614, 1464, 781, 1608]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_09_2024_4_24_16_59_92c22920a83749c994864397a370a984-11/07_icon_Auto_Boat_Air.png
try:
    _c7 = get_crop(7, 369, 144)
    canvas.paste(_c7, (449, 891), _c7)
except Exception:
    pass
layout["Auto,_Boat_&_Air"] = [449, 891, 818, 1035]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_09_2024_4_24_16_59_92c22920a83749c994864397a370a984-11/08_icon_Holiday.png
try:
    _c8 = get_crop(8, 218, 127)
    canvas.paste(_c8, (492, 764), _c8)
except Exception:
    pass
layout["Holiday"] = [492, 764, 710, 891]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_09_2024_4_24_16_59_92c22920a83749c994864397a370a984-11/09_icon_Spirituality.png
try:
    _c9 = get_crop(9, 282, 144)
    canvas.paste(_c9, (870, 637), _c9)
except Exception:
    pass
layout["Spirituality"] = [870, 637, 1152, 781]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_09_2024_4_24_16_59_92c22920a83749c994864397a370a984-11/10_icon_Spanish.png
try:
    _c10 = get_crop(10, 225, 144)
    canvas.paste(_c10, (519, 1910), _c10)
except Exception:
    pass
layout["Spanish"] = [519, 1910, 744, 2054]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_09_2024_4_24_16_59_92c22920a83749c994864397a370a984-11/11_icon_French.png
try:
    _c11 = get_crop(11, 205, 144)
    canvas.paste(_c11, (768, 1910), _c11)
except Exception:
    pass
layout["French"] = [768, 1910, 973, 2054]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_09_2024_4_24_16_59_92c22920a83749c994864397a370a984-11/12_icon_Fashion.png
try:
    _c12 = get_crop(12, 220, 144)
    canvas.paste(_c12, (1068, 764), _c12)
except Exception:
    pass
layout["Fashion"] = [1068, 764, 1288, 908]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_09_2024_4_24_16_59_92c22920a83749c994864397a370a984-11/13_icon_Arts.png
try:
    _c13 = get_crop(13, 152, 127)
    canvas.paste(_c13, (1166, 383), _c13)
except Exception:
    pass
layout["Arts"] = [1166, 383, 1318, 510]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_09_2024_4_24_16_59_92c22920a83749c994864397a370a984-11/14_icon_Seminar.png
try:
    _c14 = get_crop(14, 232, 144)
    canvas.paste(_c14, (358, 1464), _c14)
except Exception:
    pass
layout["Seminar"] = [358, 1464, 590, 1608]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_09_2024_4_24_16_59_92c22920a83749c994864397a370a984-11/15_icon_Italian.png
try:
    _c15 = get_crop(15, 191, 144)
    canvas.paste(_c15, (997, 1910), _c15)
except Exception:
    pass
layout["Italian"] = [997, 1910, 1188, 2054]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_09_2024_4_24_16_59_92c22920a83749c994864397a370a984-11/16_icon_Film_Media.png
try:
    _c16 = get_crop(16, 315, 127)
    canvas.paste(_c16, (36, 510), _c16)
except Exception:
    pass
layout["Film_&_Media"] = [36, 510, 351, 637]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_09_2024_4_24_16_59_92c22920a83749c994864397a370a984-11/17_icon_Family_Education.png
try:
    _c17 = get_crop(17, 432, 144)
    canvas.paste(_c17, (36, 764), _c17)
except Exception:
    pass
layout["Family_&_Education"] = [36, 764, 468, 908]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_09_2024_4_24_16_59_92c22920a83749c994864397a370a984-11/18_icon_Convention.png
try:
    _c18 = get_crop(18, 293, 144)
    canvas.paste(_c18, (805, 1464), _c18)
except Exception:
    pass
layout["Convention"] = [805, 1464, 1098, 1608]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_09_2024_4_24_16_59_92c22920a83749c994864397a370a984-11/19_icon_Science_Tech.png
try:
    _c19 = get_crop(19, 361, 144)
    canvas.paste(_c19, (1000, 510), _c19)
except Exception:
    pass
layout["Science_&_Tech"] = [1000, 510, 1361, 654]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_09_2024_4_24_16_59_92c22920a83749c994864397a370a984-11/20_icon_Sports_Fitness.png
try:
    _c20 = get_crop(20, 378, 144)
    canvas.paste(_c20, (375, 510), _c20)
except Exception:
    pass
layout["Sports_&_Fitness"] = [375, 510, 753, 654]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_09_2024_4_24_16_59_92c22920a83749c994864397a370a984-11/21_icon_Charity.png
try:
    _c21 = get_crop(21, 397, 144)
    canvas.paste(_c21, (449, 637), _c21)
except Exception:
    pass
layout["Charity"] = [449, 637, 846, 781]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_09_2024_4_24_16_59_92c22920a83749c994864397a370a984-11/22_icon_Home_Lifestyle.png
try:
    _c22 = get_crop(22, 389, 127)
    canvas.paste(_c22, (36, 891), _c22)
except Exception:
    pass
layout["Home_&_Lifestyle"] = [36, 891, 425, 1018]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_09_2024_4_24_16_59_92c22920a83749c994864397a370a984-11/23_icon_Festival.png
try:
    _c23 = get_crop(23, 219, 144)
    canvas.paste(_c23, (1122, 1464), _c23)
except Exception:
    pass
layout["Festival"] = [1122, 1464, 1341, 1608]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_09_2024_4_24_16_59_92c22920a83749c994864397a370a984-11/24_icon_Hobbies.png
try:
    _c24 = get_crop(24, 231, 144)
    canvas.paste(_c24, (842, 891), _c24)
except Exception:
    pass
layout["Hobbies"] = [842, 891, 1073, 1035]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_09_2024_4_24_16_59_92c22920a83749c994864397a370a984-11/25_icon_German.png
try:
    _c25 = get_crop(25, 225, 135)
    canvas.paste(_c25, (270, 1910), _c25)
except Exception:
    pass
layout["German"] = [270, 1910, 495, 2045]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_09_2024_4_24_16_59_92c22920a83749c994864397a370a984-11/26_icon_English.png
try:
    _c26 = get_crop(26, 210, 135)
    canvas.paste(_c26, (36, 1910), _c26)
except Exception:
    pass
layout["English"] = [36, 1910, 246, 2045]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_09_2024_4_24_16_59_92c22920a83749c994864397a370a984-11/27_icon_Travel_Outdoor.png
try:
    _c27 = get_crop(27, 389, 127)
    canvas.paste(_c27, (36, 637), _c27)
except Exception:
    pass
layout["Travel_&_Outdoor"] = [36, 637, 425, 764]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_09_2024_4_24_16_59_92c22920a83749c994864397a370a984-11/28_icon_Conference.png
try:
    _c28 = get_crop(28, 298, 135)
    canvas.paste(_c28, (36, 1464), _c28)
except Exception:
    pass
layout["Conference"] = [36, 1464, 334, 1599]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_09_2024_4_24_16_59_92c22920a83749c994864397a370a984-11/29_icon_School_Activities.png
try:
    _c29 = get_crop(29, 392, 135)
    canvas.paste(_c29, (36, 1018), _c29)
except Exception:
    pass
layout["School_Activities"] = [36, 1018, 428, 1153]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_09_2024_4_24_16_59_92c22920a83749c994864397a370a984-11/30_icon_Apply_filters.png
try:
    _c30 = get_crop(30, 1344, 144)
    canvas.paste(_c30, (48, 2768), _c30)
except Exception:
    pass
layout["Apply_filters"] = [48, 2768, 1392, 2912]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_09_2024_4_24_16_59_92c22920a83749c994864397a370a984-11/31_icon_Clear_all.png
try:
    _c31 = get_crop(31, 51, 68)
    canvas.paste(_c31, (1153, 1), _c31)
except Exception:
    pass
layout["Clear_all"] = [1153, 1, 1204, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_09_2024_4_24_16_59_92c22920a83749c994864397a370a984-11/32_icon_5.00.png
try:
    _c32 = get_crop(32, 144, 144)
    canvas.paste(_c32, (12, 72), _c32)
except Exception:
    pass
layout["5.00"] = [12, 72, 156, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_09_2024_4_24_16_59_92c22920a83749c994864397a370a984-11/33_icon_Clear_all.png
try:
    _c33 = get_crop(33, 100, 66)
    canvas.paste(_c33, (1211, 1), _c33)
except Exception:
    pass
layout["Clear_all"] = [1211, 1, 1311, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_09_2024_4_24_16_59_92c22920a83749c994864397a370a984-11/34_icon_5.00.png
try:
    _c34 = get_crop(34, 64, 64)
    canvas.paste(_c34, (112, 1), _c34)
except Exception:
    pass
layout["5.00"] = [112, 1, 176, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_09_2024_4_24_16_59_92c22920a83749c994864397a370a984-11/35_icon_Clear_all.png
try:
    _c35 = get_crop(35, 52, 64)
    canvas.paste(_c35, (1320, 1), _c35)
except Exception:
    pass
layout["Clear_all"] = [1320, 1, 1372, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_09_2024_4_24_16_59_92c22920a83749c994864397a370a984-11/36_icon_5.00.png
try:
    _c36 = get_crop(36, 60, 62)
    canvas.paste(_c36, (180, 1), _c36)
except Exception:
    pass
layout["5.00"] = [180, 1, 240, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_09_2024_4_24_16_59_92c22920a83749c994864397a370a984-11/37_icon_icon_37.png
try:
    _c37 = get_crop(37, 64, 61)
    canvas.paste(_c37, (308, 3), _c37)
except Exception:
    pass
layout["icon_37"] = [308, 3, 372, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_09_2024_4_24_16_59_92c22920a83749c994864397a370a984-11/38_icon_icon_38.png
try:
    _c38 = get_crop(38, 51, 61)
    canvas.paste(_c38, (249, 2), _c38)
except Exception:
    pass
layout["icon_38"] = [249, 2, 300, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_09_2024_4_24_16_59_92c22920a83749c994864397a370a984-11/39_icon_clickable_35.png
try:
    _c39 = get_crop(39, 144, 144)
    canvas.paste(_c39, (1248, 2364), _c39)
except Exception:
    pass
layout["clickable_35"] = [1248, 2364, 1392, 2508]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_09_2024_4_24_16_59_92c22920a83749c994864397a370a984-11/40_icon_Clear_all.png
try:
    _c40 = get_crop(40, 178, 144)
    canvas.paste(_c40, (1214, 72), _c40)
except Exception:
    pass
layout["Clear_all"] = [1214, 72, 1392, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_09_2024_4_24_16_59_92c22920a83749c994864397a370a984-11/41_text_5.00.png
try:
    _c41 = get_crop(41, 91, 45)
    canvas.paste(_c41, (20, 15), _c41)
except Exception:
    pass
layout["5.00"] = [20, 15, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_09_2024_4_24_16_59_92c22920a83749c994864397a370a984-11/42_text_Filters.png
try:
    _c42 = get_crop(42, 180, 66)
    canvas.paste(_c42, (631, 116), _c42)
except Exception:
    pass
layout["Filters"] = [631, 116, 811, 182]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_09_2024_4_24_16_59_92c22920a83749c994864397a370a984-11/43_text_Categories.png
try:
    _c43 = get_crop(43, 187, 127)
    canvas.paste(_c43, (36, 383), _c43)
except Exception:
    pass
layout["Categories"] = [36, 383, 223, 510]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_09_2024_4_24_16_59_92c22920a83749c994864397a370a984-11/44_text_Show_less_categories.png
try:
    _c44 = get_crop(44, 550, 144)
    canvas.paste(_c44, (0, 1153), _c44)
except Exception:
    pass
layout["Show_less_categories"] = [0, 1153, 550, 1297]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_09_2024_4_24_16_59_92c22920a83749c994864397a370a984-11/45_text_Event_type.png
try:
    _c45 = get_crop(45, 298, 135)
    canvas.paste(_c45, (36, 1464), _c45)
except Exception:
    pass
layout["Event_type"] = [36, 1464, 334, 1599]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_09_2024_4_24_16_59_92c22920a83749c994864397a370a984-11/46_text_Show_all_event_types.png
try:
    _c46 = get_crop(46, 535, 144)
    canvas.paste(_c46, (0, 1599), _c46)
except Exception:
    pass
layout["Show_all_event_types"] = [0, 1599, 535, 1743]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_09_2024_4_24_16_59_92c22920a83749c994864397a370a984-11/47_text_Languages.png
try:
    _c47 = get_crop(47, 210, 135)
    canvas.paste(_c47, (36, 1910), _c47)
except Exception:
    pass
layout["Languages"] = [36, 1910, 246, 2045]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_09_2024_4_24_16_59_92c22920a83749c994864397a370a984-11/48_text_Show_all_languages.png
try:
    _c48 = get_crop(48, 511, 144)
    canvas.paste(_c48, (0, 2045), _c48)
except Exception:
    pass
layout["Show_all_languages"] = [0, 2045, 511, 2189]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_09_2024_4_24_16_59_92c22920a83749c994864397a370a984-11/49_text_Price.png
try:
    _c49 = get_crop(49, 149, 63)
    canvas.paste(_c49, (45, 2249), _c49)
except Exception:
    pass
layout["Price"] = [45, 2249, 194, 2312]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_09_2024_4_24_16_59_92c22920a83749c994864397a370a984-11/50_text_Only_free_events.png
try:
    _c50 = get_crop(50, 511, 144)
    canvas.paste(_c50, (0, 2045), _c50)
except Exception:
    pass
layout["Only_free_events"] = [0, 2045, 511, 2189]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_09_2024_4_24_16_59_92c22920a83749c994864397a370a984-11/51_text_Sort_by.png
try:
    _c51 = get_crop(51, 206, 75)
    canvas.paste(_c51, (42, 2567), _c51)
except Exception:
    pass
layout["Sort_by"] = [42, 2567, 248, 2642]
