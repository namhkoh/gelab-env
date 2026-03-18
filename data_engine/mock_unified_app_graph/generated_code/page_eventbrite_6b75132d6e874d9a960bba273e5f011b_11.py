# page_id: page_eventbrite_6b75132d6e874d9a960bba273e5f011b_11
# screenshot: 2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-13.png
# step_index: 11/11
# task: Open Eventbrite. Set the city to 'San Francisco'. Search 'Outdoor'. Select an event starting after 5 PM. Check the ticket price.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Top status bar
draw.rectangle([(0, 0), (1440, 84)], fill="#d0d0d0")

# Header area (toolbar)
draw.rectangle([(0, 84), (1440, 220)], fill="#ffffff")
# header bottom divider
draw.line([(32, 220), (1408, 220)], fill="#ebe7ee", width=2)

# Light horizontal separators for sections
separator_color = "#efedf1"
draw.line([(32, 760), (1408, 760)], fill=separator_color, width=1)   # after about section
draw.line([(32, 1120), (1408, 1120)], fill=separator_color, width=1) # after location
draw.line([(32, 1960), (1408, 1960)], fill=separator_color, width=1) # below organizer
draw.line([(32, 2140), (1408, 2140)], fill=separator_color, width=1) # above related section

# Organizer card background (subtle off-white rounded rectangle)
org_x0, org_y0 = 120, 1440
org_x1, org_y1 = 1320, 1880
draw.rounded_rectangle([(org_x0, org_y0), (org_x1, org_y1)], radius=18, fill="#fbfafc", outline="#f0eef3", width=1)

# Subtle inner divider inside organizer area (to suggest spacing without drawing text/buttons)
draw.line([(org_x0+24, org_y0+120), (org_x1-24, org_y0+120)], fill="#f0edf2", width=1)

# Related-to-this-event container background (keeps space for ticket card that will be pasted)
rel_x0, rel_y0 = 48, 2040
rel_x1, rel_y1 = 1392, 2308  # stay above reserve area (reserve starts at y=2324)
draw.rounded_rectangle([(rel_x0, rel_y0), (rel_x1, rel_y1)], radius=16, fill="#ffffff", outline="#efecf5", width=2)

# Light shadow / subtle band above reserve area (separator)
draw.line([(0, 2324), (1440, 2324)], fill="#efe9ec", width=4)

# Safe content-area background band (beneath main content, above related)
draw.rectangle([(0, 2200), (1440, 2324)], fill="#ffffff")

# Small top content banner (pale lavender strip under header to give subtle visual structure)
draw.rectangle([(0, 220), (1440, 260)], fill="#faf8fb")

# Decorative subtle left/right gutters (very light) to match app feel
gutter_color = "#fbfbfc"
draw.rectangle([(0, 260), (32, 2200)], fill=gutter_color)
draw.rectangle([(1408, 260), (1440, 2200)], fill=gutter_color)

# Minor rounded corner accents at page edges for polished look
draw.rounded_rectangle([(18, 18), (1422, 72)], radius=10, outline="#d8d6da", width=0)

# Keep everything else blank/white; do not draw icons, text, buttons, or the reserve button area.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_11_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-13/00_icon_Follow.png
try:
    _c0 = get_crop(0, 384, 144)
    canvas.paste(_c0, (528, 1785), _c0)
except Exception:
    pass
layout["Follow"] = [528, 1785, 912, 1929]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_11_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-13/01_icon_More.png
try:
    _c1 = get_crop(1, 144, 144)
    canvas.paste(_c1, (1116, 108), _c1)
except Exception:
    pass
layout["More"] = [1116, 108, 1260, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_11_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-13/02_icon_Share.png
try:
    _c2 = get_crop(2, 144, 144)
    canvas.paste(_c2, (1260, 108), _c2)
except Exception:
    pass
layout["Share"] = [1260, 108, 1404, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_11_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-13/03_icon_Reserve_a_spot.png
try:
    _c3 = get_crop(3, 1440, 636)
    canvas.paste(_c3, (0, 2324), _c3)
except Exception:
    pass
layout["Reserve_a_spot"] = [0, 2324, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_11_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-13/04_icon_8.12.png
try:
    _c4 = get_crop(4, 144, 144)
    canvas.paste(_c4, (36, 108), _c4)
except Exception:
    pass
layout["8.12"] = [36, 108, 180, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_11_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-13/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 111, 105)
    canvas.paste(_c5, (988, 2440), _c5)
except Exception:
    pass
layout["icon_5"] = [988, 2440, 1099, 2545]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_11_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-13/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 106, 103)
    canvas.paste(_c6, (1216, 2442), _c6)
except Exception:
    pass
layout["icon_6"] = [1216, 2442, 1322, 2545]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_11_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-13/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 46, 66)
    canvas.paste(_c7, (1156, 1), _c7)
except Exception:
    pass
layout["icon_7"] = [1156, 1, 1202, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_11_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-13/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 90, 99)
    canvas.paste(_c8, (1109, 2444), _c8)
except Exception:
    pass
layout["icon_8"] = [1109, 2444, 1199, 2543]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_11_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-13/09_icon_Sports_Fitness.png
try:
    _c9 = get_crop(9, 234, 144)
    canvas.paste(_c9, (48, 736), _c9)
except Exception:
    pass
layout["Sports_&_Fitness"] = [48, 736, 282, 880]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_11_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-13/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 58, 59)
    canvas.paste(_c10, (312, 3), _c10)
except Exception:
    pass
layout["icon_10"] = [312, 3, 370, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_11_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-13/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 56, 58)
    canvas.paste(_c11, (183, 3), _c11)
except Exception:
    pass
layout["icon_11"] = [183, 3, 239, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_11_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-13/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 48, 57)
    canvas.paste(_c12, (250, 4), _c12)
except Exception:
    pass
layout["icon_12"] = [250, 4, 298, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_11_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-13/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 97, 64)
    canvas.paste(_c13, (1215, 0), _c13)
except Exception:
    pass
layout["icon_13"] = [1215, 0, 1312, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_11_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-13/14_icon_8.12.png
try:
    _c14 = get_crop(14, 55, 59)
    canvas.paste(_c14, (116, 3), _c14)
except Exception:
    pass
layout["8.12"] = [116, 3, 171, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_11_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-13/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 48, 60)
    canvas.paste(_c15, (1324, 3), _c15)
except Exception:
    pass
layout["icon_15"] = [1324, 3, 1372, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_11_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-13/16_icon_Outdoor_HIIT.png
try:
    _c16 = get_crop(16, 356, 79)
    canvas.paste(_c16, (241, 141), _c16)
except Exception:
    pass
layout["Outdoor_HIIT"] = [241, 141, 597, 220]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_11_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-13/17_icon_Show_map.png
try:
    _c17 = get_crop(17, 226, 144)
    canvas.paste(_c17, (1166, 954), _c17)
except Exception:
    pass
layout["Show_map"] = [1166, 954, 1392, 1098]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_11_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-13/18_icon_Free.png
try:
    _c18 = get_crop(18, 133, 106)
    canvas.paste(_c18, (102, 2574), _c18)
except Exception:
    pass
layout["Free"] = [102, 2574, 235, 2680]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_11_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-13/19_icon_Jackson_Playground.png
try:
    _c19 = get_crop(19, 427, 67)
    canvas.paste(_c19, (138, 1117), _c19)
except Exception:
    pass
layout["Jackson_Playground"] = [138, 1117, 565, 1184]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_11_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-13/20_icon_Free.png
try:
    _c20 = get_crop(20, 101, 114)
    canvas.paste(_c20, (233, 2573), _c20)
except Exception:
    pass
layout["Free"] = [233, 2573, 334, 2687]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_11_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-13/21_icon_Wayne_Squires.png
try:
    _c21 = get_crop(21, 403, 144)
    canvas.paste(_c21, (519, 1601), _c21)
except Exception:
    pass
layout["Wayne_Squires"] = [519, 1601, 922, 1745]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_11_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-13/22_icon_Read_more.png
try:
    _c22 = get_crop(22, 234, 144)
    canvas.paste(_c22, (48, 736), _c22)
except Exception:
    pass
layout["Read_more"] = [48, 736, 282, 880]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_11_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-13/23_icon_Jackson_Playground_Jackson_Playground_Sa.png
try:
    _c23 = get_crop(23, 226, 144)
    canvas.paste(_c23, (1166, 954), _c23)
except Exception:
    pass
layout["Jackson_Playground,_Jacks"] = [1166, 954, 1392, 1098]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_11_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-13/24_icon_icon_24.png
try:
    _c24 = get_crop(24, 47, 58)
    canvas.paste(_c24, (383, 4), _c24)
except Exception:
    pass
layout["icon_24"] = [383, 4, 430, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_11_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-13/25_text_8.12.png
try:
    _c25 = get_crop(25, 91, 43)
    canvas.paste(_c25, (20, 17), _c25)
except Exception:
    pass
layout["8.12"] = [20, 17, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_11_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-13/26_text_About_this_event.png
try:
    _c26 = get_crop(26, 144, 144)
    canvas.paste(_c26, (36, 108), _c26)
except Exception:
    pass
layout["About_this_event"] = [36, 108, 180, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_11_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-13/27_text_Get_ready_to_sweat_and_push_your_limits_.png
try:
    _c27 = get_crop(27, 234, 144)
    canvas.paste(_c27, (48, 736), _c27)
except Exception:
    pass
layout["Get_ready_to_sweat_and_pu"] = [48, 736, 282, 880]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_11_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-13/28_text_Location.png
try:
    _c28 = get_crop(28, 246, 63)
    canvas.paste(_c28, (41, 998), _c28)
except Exception:
    pass
layout["Location"] = [41, 998, 287, 1061]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_11_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-13/29_text_Organized_by.png
try:
    _c29 = get_crop(29, 403, 144)
    canvas.paste(_c29, (519, 1601), _c29)
except Exception:
    pass
layout["Organized_by"] = [519, 1601, 922, 1745]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_11_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-13/30_text_Related_to_this_event.png
try:
    _c30 = get_crop(30, 563, 63)
    canvas.paste(_c30, (43, 2182), _c30)
except Exception:
    pass
layout["Related_to_this_event"] = [43, 2182, 606, 2245]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_11_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-13/31_text_General_Admission.png
try:
    _c31 = get_crop(31, 415, 55)
    canvas.paste(_c31, (116, 2451), _c31)
except Exception:
    pass
layout["General_Admission"] = [116, 2451, 531, 2506]
