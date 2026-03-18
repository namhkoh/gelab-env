# page_id: page_eventbrite_31528ae16c264b1b98bd4e1f25c5d9e5_03
# screenshot: 2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-5.png
# step_index: 3/11
# task: Open Eventbrite. Search 'Fitness'. Filter for free events. Browse and select any 'Yoga' event. Note the location.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: fallback_compose
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw background and structural UI elements for Eventbrite-like page
# Variables available: canvas (PIL Image), draw (PIL ImageDraw), fonts: font_sm, font_md, font_lg, font_xl

W, H = canvas.size

# Colors
status_bar_color = "#bdbdbd"    # muted gray for status bar
header_divider = "#e6eefc"      # very light bluish divider (subtle)
section_divider = "#ececec"     # light divider between sections
card_shadow = "#efefef"         # subtle shadow behind cards
card_border = "#e6e6e6"         # card border
card_fill = "#ffffff"           # white card fill
bottom_nav_bg = "#fafafa"       # bottom navigation background
bottom_nav_border = "#e0e0e0"   # nav top border

# Status bar area (~top 50-88px)
status_h = 88
draw.rectangle([(0, 0), (W, status_h)], fill=status_bar_color)

# Header area background (keeps canvas white but add subtle bottom divider)
header_top = status_h
header_bottom = 176
draw.rectangle([(0, header_top), (W, header_bottom)], fill="#ffffff")
# subtle divider line under header
draw.line([(48, header_bottom), (W-48, header_bottom)], fill=header_divider, width=3)

# Popular / content separators
# divider under "Popular" area (keeps whitespace but add subtle rule)
popular_div_y = 360
draw.line([(48, popular_div_y), (W-48, popular_div_y)], fill=section_divider, width=1)

# divider above Events section
events_div_y = 980
draw.line([(48, events_div_y), (W-48, events_div_y)], fill=section_divider, width=1)

# Card areas for event list (as rounded white cards with subtle shadow and border)
card_rects = [
    (48, 1117, 1392, 1513),  # first event card area (backdrop)
    (48, 1513, 1392, 1909),  # second event card area
    (48, 1909, 1392, 2305),  # third event card area
    (48, 2305, 1392, 2701),  # fourth event card area
]

for (x1, y1, x2, y2) in card_rects:
    # shadow (slightly offset downwards)
    shadow_offset = 6
    draw.rounded_rectangle(
        [(x1 + 2, y1 + shadow_offset, x2 + 2, y2 + shadow_offset)],
        radius=16,
        fill=card_shadow,
        outline=None,
    )
    # main card
    draw.rounded_rectangle(
        [(x1, y1, x2, y2)],
        radius=16,
        fill=card_fill,
        outline=card_border,
        width=1,
    )

# Smaller subtle separators between the event cards (thin rules)
for (_, y1, _, y2) in card_rects:
    # Add a subtle rule at bottom of each card area (inside card border)
    sep_y = y2 + 10
    if sep_y < H - 200:
        draw.line([(64, sep_y), (W-64, sep_y)], fill=section_divider, width=1)

# Bottom navigation bar background and top border
nav_top = 2804
draw.rectangle([(0, nav_top), (W, H)], fill=bottom_nav_bg)
draw.line([(0, nav_top), (W, nav_top)], fill=bottom_nav_border, width=2)

# Left gutter subtle vertical guide (not part of detected elements, purely structural)
gutter_x = 48
draw.line([(gutter_x, header_bottom + 8), (gutter_x, nav_top - 8)], fill=section_divider, width=1)

# Right content margin guide (subtle)
right_margin_x = W - 48
draw.line([(right_margin_x, header_bottom + 8), (right_margin_x, nav_top - 8)], fill=section_divider, width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_03_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-5/00_icon_ycar_olds.png
try:
    _c0 = get_crop(0, 1344, 396)
    canvas.paste(_c0, (48, 1117), _c0)
except Exception:
    pass
layout["'ycar_olds"] = [48, 1117, 1392, 1513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_03_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-5/01_icon_Thrive.png
try:
    _c1 = get_crop(1, 1344, 396)
    canvas.paste(_c1, (48, 1513), _c1)
except Exception:
    pass
layout["Thrive"] = [48, 1513, 1392, 1909]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_03_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-5/02_icon_8_344_creator_followers.png
try:
    _c2 = get_crop(2, 1344, 396)
    canvas.paste(_c2, (48, 1117), _c2)
except Exception:
    pass
layout["8_344_creator_followers"] = [48, 1117, 1392, 1513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_03_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-5/03_icon_7.54.png
try:
    _c3 = get_crop(3, 126, 108)
    canvas.paste(_c3, (53, 115), _c3)
except Exception:
    pass
layout["7.54"] = [53, 115, 179, 223]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_03_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-5/04_icon_Fitness.png
try:
    _c4 = get_crop(4, 1344, 191)
    canvas.paste(_c4, (48, 72), _c4)
except Exception:
    pass
layout["Fitness]"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_03_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-5/05_icon_Fitness.png
try:
    _c5 = get_crop(5, 56, 58)
    canvas.paste(_c5, (313, 4), _c5)
except Exception:
    pass
layout["Fitness]"] = [313, 4, 369, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_03_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-5/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 43, 54)
    canvas.paste(_c6, (253, 6), _c6)
except Exception:
    pass
layout["icon_6"] = [253, 6, 296, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_03_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-5/07_icon_7.54.png
try:
    _c7 = get_crop(7, 53, 58)
    canvas.paste(_c7, (183, 4), _c7)
except Exception:
    pass
layout["7.54"] = [183, 4, 236, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_03_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-5/08_icon_7.54.png
try:
    _c8 = get_crop(8, 57, 60)
    canvas.paste(_c8, (115, 3), _c8)
except Exception:
    pass
layout["7.54"] = [115, 3, 172, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_03_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-5/09_icon_Thu_Apr_25.png
try:
    _c9 = get_crop(9, 1344, 396)
    canvas.paste(_c9, (48, 2305), _c9)
except Exception:
    pass
layout["Thu,_Apr_25"] = [48, 2305, 1392, 2701]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_03_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-5/10_icon_8_15_creator_followers.png
try:
    _c10 = get_crop(10, 1344, 396)
    canvas.paste(_c10, (48, 1909), _c10)
except Exception:
    pass
layout["8_15_creator_followers"] = [48, 1909, 1392, 2305]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_03_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-5/11_icon_Tickets.png
try:
    _c11 = get_crop(11, 288, 156)
    canvas.paste(_c11, (864, 2804), _c11)
except Exception:
    pass
layout["Tickets"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_03_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-5/12_icon_HIIT_Bodyweight_Family_Fitness_Weekly.png
try:
    _c12 = get_crop(12, 1344, 396)
    canvas.paste(_c12, (48, 1513), _c12)
except Exception:
    pass
layout["HIIT_Bodyweight_+_Family_"] = [48, 1513, 1392, 1909]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_03_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-5/13_icon_Sun.png
try:
    _c13 = get_crop(13, 288, 156)
    canvas.paste(_c13, (288, 2804), _c13)
except Exception:
    pass
layout["Sun,"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_03_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-5/14_icon_Cancel.png
try:
    _c14 = get_crop(14, 94, 63)
    canvas.paste(_c14, (1215, 0), _c14)
except Exception:
    pass
layout["Cancel"] = [1215, 0, 1309, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_03_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-5/15_icon_Cancel.png
try:
    _c15 = get_crop(15, 48, 60)
    canvas.paste(_c15, (1322, 2), _c15)
except Exception:
    pass
layout["Cancel"] = [1322, 2, 1370, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_03_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-5/16_icon_5_._IO_O0AM_PDT.png
try:
    _c16 = get_crop(16, 288, 156)
    canvas.paste(_c16, (576, 2804), _c16)
except Exception:
    pass
layout["5_._IO:O0AM_PDT"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_03_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-5/17_icon_Cancel.png
try:
    _c17 = get_crop(17, 144, 144)
    canvas.paste(_c17, (1099, 96), _c17)
except Exception:
    pass
layout["Cancel"] = [1099, 96, 1243, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_03_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-5/18_icon_7.54.png
try:
    _c18 = get_crop(18, 92, 60)
    canvas.paste(_c18, (15, 2), _c18)
except Exception:
    pass
layout["7.54"] = [15, 2, 107, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_03_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-5/19_icon_free_fitness.png
try:
    _c19 = get_crop(19, 1344, 120)
    canvas.paste(_c19, (48, 618), _c19)
except Exception:
    pass
layout["free_fitness"] = [48, 618, 1392, 738]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_03_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-5/20_icon_8_786_creator_followers.png
try:
    _c20 = get_crop(20, 1344, 396)
    canvas.paste(_c20, (48, 2305), _c20)
except Exception:
    pass
layout["8_786_creator_followers"] = [48, 2305, 1392, 2701]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_03_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-5/21_icon_Tiemeyer_Park.png
try:
    _c21 = get_crop(21, 234, 52)
    canvas.paste(_c21, (390, 1322), _c21)
except Exception:
    pass
layout["Tiemeyer_Park"] = [390, 1322, 624, 1374]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_03_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-5/22_icon_Strive_2_Fitness_Jump_into_Fitness.png
try:
    _c22 = get_crop(22, 1344, 396)
    canvas.paste(_c22, (48, 1117), _c22)
except Exception:
    pass
layout["Strive_2_Fitness_Jump_int"] = [48, 1117, 1392, 1513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_03_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-5/23_icon_Skin_Fitness_Lymphatic_Facial_Massage.png
try:
    _c23 = get_crop(23, 1344, 396)
    canvas.paste(_c23, (48, 2305), _c23)
except Exception:
    pass
layout["Skin_Fitness_Lymphatic_Fa"] = [48, 2305, 1392, 2701]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_03_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-5/24_icon_Home.png
try:
    _c24 = get_crop(24, 288, 156)
    canvas.paste(_c24, (0, 2804), _c24)
except Exception:
    pass
layout["Home"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_03_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-5/25_icon_Thrive.png
try:
    _c25 = get_crop(25, 181, 52)
    canvas.paste(_c25, (390, 1749), _c25)
except Exception:
    pass
layout["Thrive"] = [390, 1749, 571, 1801]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_03_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-5/26_icon_Cancel.png
try:
    _c26 = get_crop(26, 149, 144)
    canvas.paste(_c26, (1243, 97), _c26)
except Exception:
    pass
layout["Cancel"] = [1243, 97, 1392, 241]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_03_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-5/27_icon_fitness_classes.png
try:
    _c27 = get_crop(27, 1344, 120)
    canvas.paste(_c27, (48, 378), _c27)
except Exception:
    pass
layout["fitness_classes"] = [48, 378, 1392, 498]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_03_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-5/28_icon_More.png
try:
    _c28 = get_crop(28, 288, 156)
    canvas.paste(_c28, (1152, 2804), _c28)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_03_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-5/29_icon_Fitzgerald_Fitness_WARRIORS_WALK.png
try:
    _c29 = get_crop(29, 1344, 396)
    canvas.paste(_c29, (48, 1909), _c29)
except Exception:
    pass
layout["Fitzgerald_Fitness_WARRIO"] = [48, 1909, 1392, 2305]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_03_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-5/30_icon_icon_30.png
try:
    _c30 = get_crop(30, 45, 60)
    canvas.paste(_c30, (386, 4), _c30)
except Exception:
    pass
layout["icon_30"] = [386, 4, 431, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_03_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-5/31_text_Popular.png
try:
    _c31 = get_crop(31, 221, 78)
    canvas.paste(_c31, (44, 298), _c31)
except Exception:
    pass
layout["Popular"] = [44, 298, 265, 376]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_03_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-5/32_text_fitness_expo.png
try:
    _c32 = get_crop(32, 238, 52)
    canvas.paste(_c32, (159, 550), _c32)
except Exception:
    pass
layout["fitness_expo"] = [159, 550, 397, 602]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_03_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-5/33_text_sports_and_fitness.png
try:
    _c33 = get_crop(33, 1344, 120)
    canvas.paste(_c33, (48, 738), _c33)
except Exception:
    pass
layout["sports_and_fitness"] = [48, 738, 1392, 858]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_03_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-5/34_text_free_fitness_class_events.png
try:
    _c34 = get_crop(34, 1344, 144)
    canvas.paste(_c34, (48, 858), _c34)
except Exception:
    pass
layout["free_fitness_class_events"] = [48, 858, 1392, 1002]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_03_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-5/35_text_Events.png
try:
    _c35 = get_crop(35, 189, 57)
    canvas.paste(_c35, (46, 1029), _c35)
except Exception:
    pass
layout["Events"] = [46, 1029, 235, 1086]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_03_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-5/36_text_88274_creator_followers.png
try:
    _c36 = get_crop(36, 1344, 396)
    canvas.paste(_c36, (48, 1513), _c36)
except Exception:
    pass
layout["88274_creator_followers"] = [48, 1513, 1392, 1909]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_03_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-5/37_text_Sun.png
try:
    _c37 = get_crop(37, 88, 43)
    canvas.paste(_c37, (390, 2759), _c37)
except Exception:
    pass
layout["Sun,"] = [390, 2759, 478, 2802]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_03_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-5/38_text_5_._IO_O0AM_PDT.png
try:
    _c38 = get_crop(38, 288, 156)
    canvas.paste(_c38, (576, 2804), _c38)
except Exception:
    pass
layout["5_._IO:O0AM_PDT"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_03_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-5/39_clickable_fitness_expo.png
try:
    _c39 = get_crop(39, 1344, 120)
    canvas.paste(_c39, (48, 498), _c39)
except Exception:
    pass
layout["fitness_expo"] = [48, 498, 1392, 618]
