# page_id: page_eventbrite_31528ae16c264b1b98bd4e1f25c5d9e5_09
# screenshot: 2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-11.png
# step_index: 9/11
# task: Open Eventbrite. Search 'Fitness'. Filter for free events. Browse and select any 'Yoga' event. Note the location.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Top-level background
draw.rectangle((0, 0, 1440, 2960), fill="#fbfbfc")

# Status bar (approx 56px high)
status_h = 56
draw.rectangle((0, 0, 1440, status_h), fill="#bdbdbd")
# Slight darker top edge to mimic subtle device bezel
draw.line((0, 0, 1440, 0), fill="#b3b3b3", width=1)
# Slight lighter separator below status bar
draw.line((0, status_h, 1440, status_h), fill="#d6d6d6", width=1)

# Header / toolbar area (search bar background area)
header_top = status_h
header_bottom = 150
draw.rectangle((0, header_top, 1440, header_bottom), fill="#ffffff")
# Subtle divider under header
draw.line((48, header_bottom, 1392, header_bottom), fill="#e6e6e6", width=2)

# Filter / chips row area background (keeps page visually grouped)
chips_top = 259
chips_bottom = chips_top + 144
# leave chips themselves to be pasted; draw a faint background band behind them
draw.rectangle((0, chips_top - 8, 1440, chips_bottom + 8), fill="#ffffff")
draw.line((48, chips_bottom + 8, 1392, chips_bottom + 8), fill="#f0f0f0", width=1)

# Event card containers (rounded rects behind images + text areas)
card_x1 = 48
card_x2 = 1392
# Card 1: cover image (48,525,1392,1278) and its title area below
card1_top = 480
card1_bottom = 1320
draw.rounded_rectangle((card_x1, card1_top, card_x2, card1_bottom),
                       radius=16, fill="#ffffff", outline="#e6e9ee", width=1)
# Soft shadow effect under card1
draw.rectangle((card_x1+8, card1_bottom, card_x2-8, card1_bottom+6), fill="#f1f3f5")

# Card 2: cover image (48,1326,1392,2434) and title area below
card2_top = 1280
card2_bottom = 2440
draw.rounded_rectangle((card_x1, card2_top, card_x2, card2_bottom),
                       radius=16, fill="#ffffff", outline="#e6e9ee", width=1)
draw.rectangle((card_x1+8, card2_bottom, card_x2-8, card2_bottom+6), fill="#f1f3f5")

# Card 3: smaller/bottom event image (48,2482,1392,2816)
card3_top = 2440
card3_bottom = 2816
draw.rounded_rectangle((card_x1, card3_top, card_x2, card3_bottom),
                       radius=16, fill="#ffffff", outline="#e6e9ee", width=1)
draw.rectangle((card_x1+8, card3_bottom, card_x2-8, card3_bottom+6), fill="#f1f3f5")

# Separators between cards / sections
sep_x1 = 48
sep_x2 = 1392
draw.line((sep_x1, card1_bottom + 12, sep_x2, card1_bottom + 12), fill="#f0f0f0", width=1)
draw.line((sep_x1, card2_bottom + 12, sep_x2, card2_bottom + 12), fill="#f0f0f0", width=1)

# Subtle page-level horizontal divider above bottom navigation
bottom_nav_top = 2804
draw.line((0, bottom_nav_top, 1440, bottom_nav_top), fill="#e6e6e6", width=2)

# Bottom navigation background area
draw.rectangle((0, bottom_nav_top, 1440, 2960), fill="#ffffff")
# Soft top shadow for nav bar
draw.rectangle((0, bottom_nav_top, 1440, bottom_nav_top + 6), fill="#f2f4f6")

# Final subtle vertical padding lines at page sides to frame content
draw.line((48, 0, 48, 2960), fill="#fbfbfc", width=1)
draw.line((1392, 0, 1392, 2960), fill="#fbfbfc", width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_09_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-11/00_icon_Music.png
try:
    _c0 = get_crop(0, 198, 111)
    canvas.paste(_c0, (843, 406), _c0)
except Exception:
    pass
layout["Music"] = [843, 406, 1041, 517]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_09_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-11/01_icon_Anytime.png
try:
    _c1 = get_crop(1, 1344, 191)
    canvas.paste(_c1, (48, 72), _c1)
except Exception:
    pass
layout["Anytime"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_09_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-11/02_icon_Business.png
try:
    _c2 = get_crop(2, 251, 112)
    canvas.paste(_c2, (1042, 405), _c2)
except Exception:
    pass
layout["Business"] = [1042, 405, 1293, 517]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_09_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-11/03_icon_1_Filter.png
try:
    _c3 = get_crop(3, 536, 144)
    canvas.paste(_c3, (0, 259), _c3)
except Exception:
    pass
layout["1_Filter"] = [0, 259, 536, 403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_09_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-11/04_icon_Il.png
try:
    _c4 = get_crop(4, 144, 144)
    canvas.paste(_c4, (1092, 1842), _c4)
except Exception:
    pass
layout["Il"] = [1092, 1842, 1236, 1986]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_09_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-11/05_icon_Business.png
try:
    _c5 = get_crop(5, 144, 144)
    canvas.paste(_c5, (1092, 686), _c5)
except Exception:
    pass
layout["Business"] = [1092, 686, 1236, 830]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_09_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-11/06_icon_Fo.png
try:
    _c6 = get_crop(6, 137, 110)
    canvas.paste(_c6, (1296, 406), _c6)
except Exception:
    pass
layout["Fo("] = [1296, 406, 1433, 516]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_09_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-11/07_icon_Fo.png
try:
    _c7 = get_crop(7, 144, 144)
    canvas.paste(_c7, (1236, 686), _c7)
except Exception:
    pass
layout["Fo("] = [1236, 686, 1380, 830]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_09_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-11/08_icon_Il.png
try:
    _c8 = get_crop(8, 144, 144)
    canvas.paste(_c8, (1236, 1842), _c8)
except Exception:
    pass
layout["Il"] = [1236, 1842, 1380, 1986]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_09_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-11/09_icon_HIIT_Bodyweight_Family_Fitness_Weekly.png
try:
    _c9 = get_crop(9, 1344, 1108)
    canvas.paste(_c9, (48, 1326), _c9)
except Exception:
    pass
layout["HIIT_Bodyweight_+_Family_"] = [48, 1326, 1392, 2434]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_09_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-11/10_icon_7.55.png
try:
    _c10 = get_crop(10, 121, 112)
    canvas.paste(_c10, (55, 114), _c10)
except Exception:
    pass
layout["7.55"] = [55, 114, 176, 226]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_09_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-11/11_icon_Close_current_screen.png
try:
    _c11 = get_crop(11, 144, 144)
    canvas.paste(_c11, (1248, 96), _c11)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 96, 1392, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_09_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-11/12_icon_HIIT_Bodyweight_Family_Fitness_Weekly.png
try:
    _c12 = get_crop(12, 1344, 753)
    canvas.paste(_c12, (48, 525), _c12)
except Exception:
    pass
layout["HIIT_Bodyweight_+_Family_"] = [48, 525, 1392, 1278]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_09_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-11/13_icon_Fitness.png
try:
    _c13 = get_crop(13, 66, 64)
    canvas.paste(_c13, (308, 0), _c13)
except Exception:
    pass
layout["Fitness"] = [308, 0, 374, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_09_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-11/14_icon_7.55.png
try:
    _c14 = get_crop(14, 59, 65)
    canvas.paste(_c14, (181, 0), _c14)
except Exception:
    pass
layout["7.55"] = [181, 0, 240, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_09_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-11/15_icon_Fitness.png
try:
    _c15 = get_crop(15, 53, 65)
    canvas.paste(_c15, (247, 0), _c15)
except Exception:
    pass
layout["Fitness"] = [247, 0, 300, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_09_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-11/16_icon_7.55.png
try:
    _c16 = get_crop(16, 57, 65)
    canvas.paste(_c16, (116, 0), _c16)
except Exception:
    pass
layout["7.55"] = [116, 0, 173, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_09_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-11/17_icon_icon_17.png
try:
    _c17 = get_crop(17, 80, 61)
    canvas.paste(_c17, (1207, 0), _c17)
except Exception:
    pass
layout["icon_17"] = [1207, 0, 1287, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_09_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-11/18_icon_icon_18.png
try:
    _c18 = get_crop(18, 62, 61)
    canvas.paste(_c18, (1316, 0), _c18)
except Exception:
    pass
layout["icon_18"] = [1316, 0, 1378, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_09_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-11/19_icon_Tickets.png
try:
    _c19 = get_crop(19, 288, 156)
    canvas.paste(_c19, (864, 2804), _c19)
except Exception:
    pass
layout["Tickets"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_09_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-11/20_icon_Fitness.png
try:
    _c20 = get_crop(20, 1344, 191)
    canvas.paste(_c20, (48, 72), _c20)
except Exception:
    pass
layout["Fitness"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_09_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-11/21_icon_San_Francisco.png
try:
    _c21 = get_crop(21, 536, 144)
    canvas.paste(_c21, (0, 259), _c21)
except Exception:
    pass
layout["San_Francisco"] = [0, 259, 536, 403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_09_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-11/22_icon_icon_22.png
try:
    _c22 = get_crop(22, 49, 63)
    canvas.paste(_c22, (384, 1), _c22)
except Exception:
    pass
layout["icon_22"] = [384, 1, 433, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_09_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-11/23_icon_CLASSES.png
try:
    _c23 = get_crop(23, 288, 156)
    canvas.paste(_c23, (288, 2804), _c23)
except Exception:
    pass
layout["CLASSES"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_09_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-11/24_icon_More.png
try:
    _c24 = get_crop(24, 288, 156)
    canvas.paste(_c24, (1152, 2804), _c24)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_09_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-11/25_icon_HIIT_Bodyweight_Family_Fitness_Weekly.png
try:
    _c25 = get_crop(25, 1344, 1108)
    canvas.paste(_c25, (48, 1326), _c25)
except Exception:
    pass
layout["HIIT_Bodyweight_+_Family_"] = [48, 1326, 1392, 2434]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_09_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-11/26_icon_Anytime.png
try:
    _c26 = get_crop(26, 1344, 753)
    canvas.paste(_c26, (48, 525), _c26)
except Exception:
    pass
layout["Anytime"] = [48, 525, 1392, 1278]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_09_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-11/27_icon_icon_27.png
try:
    _c27 = get_crop(27, 42, 62)
    canvas.paste(_c27, (1273, 0), _c27)
except Exception:
    pass
layout["icon_27"] = [1273, 0, 1315, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_09_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-11/28_icon_Favorites.png
try:
    _c28 = get_crop(28, 288, 156)
    canvas.paste(_c28, (576, 2804), _c28)
except Exception:
    pass
layout["Favorites"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_09_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-11/29_icon_PRESENTED_BY.png
try:
    _c29 = get_crop(29, 288, 156)
    canvas.paste(_c29, (0, 2804), _c29)
except Exception:
    pass
layout["PRESENTED_BY"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_09_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-11/30_text_7.55.png
try:
    _c30 = get_crop(30, 92, 43)
    canvas.paste(_c30, (22, 17), _c30)
except Exception:
    pass
layout["7.55"] = [22, 17, 114, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_09_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-11/31_text_Sun.png
try:
    _c31 = get_crop(31, 101, 53)
    canvas.paste(_c31, (90, 1113), _c31)
except Exception:
    pass
layout["Sun,"] = [90, 1113, 191, 1166]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_09_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-11/32_text_19.png
try:
    _c32 = get_crop(32, 66, 48)
    canvas.paste(_c32, (275, 1112), _c32)
except Exception:
    pass
layout["19"] = [275, 1112, 341, 1160]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_09_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-11/33_text_10_00_AM_PDT.png
try:
    _c33 = get_crop(33, 280, 48)
    canvas.paste(_c33, (359, 1112), _c33)
except Exception:
    pass
layout["10:00_AM_PDT"] = [359, 1112, 639, 1160]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_09_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-11/34_text_Thrive.png
try:
    _c34 = get_crop(34, 121, 43)
    canvas.paste(_c34, (94, 1183), _c34)
except Exception:
    pass
layout["Thrive"] = [94, 1183, 215, 1226]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_09_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-11/35_text_WEEKLY.png
try:
    _c35 = get_crop(35, 311, 79)
    canvas.paste(_c35, (81, 1356), _c35)
except Exception:
    pass
layout["WEEKLY"] = [81, 1356, 392, 1435]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_09_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-11/36_text_Thrive.png
try:
    _c36 = get_crop(36, 121, 43)
    canvas.paste(_c36, (94, 2339), _c36)
except Exception:
    pass
layout["Thrive"] = [94, 2339, 215, 2382]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_09_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-11/37_text_WEEKLY.png
try:
    _c37 = get_crop(37, 308, 77)
    canvas.paste(_c37, (84, 2512), _c37)
except Exception:
    pass
layout["WEEKLY"] = [84, 2512, 392, 2589]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_09_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-11/38_text_CLASSES.png
try:
    _c38 = get_crop(38, 288, 156)
    canvas.paste(_c38, (0, 2804), _c38)
except Exception:
    pass
layout["CLASSES"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_09_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-11/39_text_PRESENTED_BY.png
try:
    _c39 = get_crop(39, 166, 30)
    canvas.paste(_c39, (88, 2742), _c39)
except Exception:
    pass
layout["PRESENTED_BY"] = [88, 2742, 254, 2772]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_09_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-11/40_clickable_Event_s_image.png
try:
    _c40 = get_crop(40, 1344, 334)
    canvas.paste(_c40, (48, 2482), _c40)
except Exception:
    pass
layout["Event's_image"] = [48, 2482, 1392, 2816]
