# page_id: page_eventbrite_f56b9f3bf9ef483cbd9847af47d34434_06
# screenshot: 2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-8.png
# step_index: 6/8
# task: Open Eventbrite. Look up "Gardening" events. Filter by events happening this week. Select the first event from the results. Follow the organizer and where is the event?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw background fill (dominant off-white/very light gray)
draw.rectangle([(0, 0), (1440, 2960)], fill="#f6f8fb")

# Status bar area at top (~96px)
status_h = 96
draw.rectangle([(0, 0), (1440, status_h)], fill="#cfcfcf")

# Header / toolbar area (white)
header_top = status_h
header_bottom = 220
draw.rectangle([(0, header_top), (1440, header_bottom)], fill="#ffffff")

# Thin divider under header
draw.line([(48, header_bottom), (1392, header_bottom)], fill="#e6e6e9", width=2)

# Subtle filter section divider (under the filter row)
filter_div_y = 380
draw.line([(48, filter_div_y), (1392, filter_div_y)], fill="#f0f2f5", width=1)

# Card style parameters
card_radius = 24
card_outline = "#e6e9ee"
card_fill = "#ffffff"
shadow_fill = "#e9eef5"

# First event card background (shadow + rounded white card)
card1_x0, card1_y0 = 48, 676
card1_x1, card1_y1 = card1_x0 + 1344, card1_y0 + 1108  # matches detected size
# shadow (subtle, offset)
draw.rounded_rectangle(
    [(card1_x0 + 6, card1_y0 + 8), (card1_x1 + 6, card1_y1 + 8)],
    radius=card_radius + 2,
    fill=shadow_fill,
    outline=None
)
# white card background
draw.rounded_rectangle(
    [(card1_x0, card1_y0), (card1_x1, card1_y1)],
    radius=card_radius,
    fill=card_fill,
    outline=card_outline,
    width=1
)

# Spacing separator below first card
sep_y1 = card1_y1 + 28
draw.line([(48, sep_y1), (1392, sep_y1)], fill="#f0f2f5", width=1)

# Second event card background (shadow + rounded white card)
card2_x0, card2_y0 = 48, 1832
card2_x1, card2_y1 = card2_x0 + 1344, card2_y0 + 984  # matches detected size
# shadow
draw.rounded_rectangle(
    [(card2_x0 + 6, card2_y0 + 8), (card2_x1 + 6, card2_y1 + 8)],
    radius=card_radius + 2,
    fill=shadow_fill,
    outline=None
)
# white card background
draw.rounded_rectangle(
    [(card2_x0, card2_y0), (card2_x1, card2_y1)],
    radius=card_radius,
    fill=card_fill,
    outline=card_outline,
    width=1
)

# Separator line between content list items (subtle)
middle_sep_y = card2_y0 - 24
draw.line([(48, middle_sep_y), (1392, middle_sep_y)], fill="#f0f2f5", width=1)

# Bottom navigation bar background
nav_top = 2804
draw.rectangle([(0, nav_top), (1440, 2960)], fill="#ffffff")
# nav top border
draw.line([(0, nav_top), (1440, nav_top)], fill="#e6e9ee", width=2)

# Small top and bottom page edge guidelines (very subtle)
draw.line([(48, 260), (1392, 260)], fill="#f3f5f7", width=1)
draw.line([(48, 2760), (1392, 2760)], fill="#f3f5f7", width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_06_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-8/00_icon_Music.png
try:
    _c0 = get_crop(0, 187, 103)
    canvas.paste(_c0, (882, 410), _c0)
except Exception:
    pass
layout["Music"] = [882, 410, 1069, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_06_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-8/01_icon_This_Week.png
try:
    _c1 = get_crop(1, 432, 103)
    canvas.paste(_c1, (438, 410), _c1)
except Exception:
    pass
layout["This_Week"] = [438, 410, 870, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_06_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-8/02_icon_Business.png
try:
    _c2 = get_crop(2, 241, 103)
    canvas.paste(_c2, (1081, 410), _c2)
except Exception:
    pass
layout["Business"] = [1081, 410, 1322, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_06_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-8/03_icon_1_Filter.png
try:
    _c3 = get_crop(3, 372, 103)
    canvas.paste(_c3, (54, 410), _c3)
except Exception:
    pass
layout["1_Filter"] = [54, 410, 426, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_06_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-8/04_icon_Vegetable.png
try:
    _c4 = get_crop(4, 144, 144)
    canvas.paste(_c4, (1092, 1192), _c4)
except Exception:
    pass
layout["Vegetable"] = [1092, 1192, 1236, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_06_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-8/05_icon_Vegetable.png
try:
    _c5 = get_crop(5, 144, 144)
    canvas.paste(_c5, (1236, 1192), _c5)
except Exception:
    pass
layout["Vegetable"] = [1236, 1192, 1380, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_06_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-8/06_icon_Favorite_button.png
try:
    _c6 = get_crop(6, 144, 144)
    canvas.paste(_c6, (1092, 2348), _c6)
except Exception:
    pass
layout["Favorite_button"] = [1092, 2348, 1236, 2492]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_06_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-8/07_icon_Business.png
try:
    _c7 = get_crop(7, 93, 110)
    canvas.paste(_c7, (1329, 407), _c7)
except Exception:
    pass
layout["Business"] = [1329, 407, 1422, 517]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_06_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-8/08_icon_Overflow_menu_button.png
try:
    _c8 = get_crop(8, 144, 144)
    canvas.paste(_c8, (1236, 2348), _c8)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 2348, 1380, 2492]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_06_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-8/09_icon_Backyard_Mindfulness_Seminar.png
try:
    _c9 = get_crop(9, 1344, 1108)
    canvas.paste(_c9, (48, 676), _c9)
except Exception:
    pass
layout["Backyard_Mindfulness_Semi"] = [48, 676, 1392, 1784]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_06_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-8/10_icon_Close_current_screen.png
try:
    _c10 = get_crop(10, 144, 144)
    canvas.paste(_c10, (1248, 96), _c10)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 96, 1392, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_06_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-8/11_icon_Gardening.png
try:
    _c11 = get_crop(11, 1344, 191)
    canvas.paste(_c11, (48, 72), _c11)
except Exception:
    pass
layout["Gardening"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_06_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-8/12_icon_Gardening.png
try:
    _c12 = get_crop(12, 64, 62)
    canvas.paste(_c12, (309, 1), _c12)
except Exception:
    pass
layout["Gardening"] = [309, 1, 373, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_06_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-8/13_icon_5.10.png
try:
    _c13 = get_crop(13, 53, 63)
    canvas.paste(_c13, (183, 1), _c13)
except Exception:
    pass
layout["5.10"] = [183, 1, 236, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_06_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-8/14_icon_5.10.png
try:
    _c14 = get_crop(14, 115, 110)
    canvas.paste(_c14, (58, 116), _c14)
except Exception:
    pass
layout["5.10"] = [58, 116, 173, 226]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_06_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-8/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 99, 62)
    canvas.paste(_c15, (1210, 0), _c15)
except Exception:
    pass
layout["icon_15"] = [1210, 0, 1309, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_06_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-8/16_icon_Wildlife_Gardening_for_Beginners.png
try:
    _c16 = get_crop(16, 1344, 984)
    canvas.paste(_c16, (48, 1832), _c16)
except Exception:
    pass
layout["Wildlife_Gardening_for_Be"] = [48, 1832, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_06_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-8/17_icon_icon_17.png
try:
    _c17 = get_crop(17, 45, 63)
    canvas.paste(_c17, (251, 0), _c17)
except Exception:
    pass
layout["icon_17"] = [251, 0, 296, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_06_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-8/18_icon_5.10.png
try:
    _c18 = get_crop(18, 56, 66)
    canvas.paste(_c18, (116, 0), _c18)
except Exception:
    pass
layout["5.10"] = [116, 0, 172, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_06_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-8/19_icon_icon_19.png
try:
    _c19 = get_crop(19, 56, 63)
    canvas.paste(_c19, (1317, 0), _c19)
except Exception:
    pass
layout["icon_19"] = [1317, 0, 1373, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_06_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-8/20_icon_IO_00_AM_GMT_OI_O0.png
try:
    _c20 = get_crop(20, 288, 156)
    canvas.paste(_c20, (288, 2804), _c20)
except Exception:
    pass
layout["IO:00_AM_GMT+OI:O0"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_06_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-8/21_icon_None.png
try:
    _c21 = get_crop(21, 353, 144)
    canvas.paste(_c21, (0, 259), _c21)
except Exception:
    pass
layout["None"] = [0, 259, 353, 403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_06_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-8/22_icon_Backyard_Mindfulness_Seminar.png
try:
    _c22 = get_crop(22, 1344, 1108)
    canvas.paste(_c22, (48, 676), _c22)
except Exception:
    pass
layout["Backyard_Mindfulness_Semi"] = [48, 676, 1392, 1784]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_06_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-8/23_icon_Gardening.png
try:
    _c23 = get_crop(23, 46, 62)
    canvas.paste(_c23, (384, 2), _c23)
except Exception:
    pass
layout["Gardening"] = [384, 2, 430, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_06_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-8/24_icon_IO_00_AM_GMT_OI_O0.png
try:
    _c24 = get_crop(24, 288, 156)
    canvas.paste(_c24, (576, 2804), _c24)
except Exception:
    pass
layout["IO:00_AM_GMT+OI:O0"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_06_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-8/25_icon_IO_00_AM_GMT_OI_O0.png
try:
    _c25 = get_crop(25, 288, 156)
    canvas.paste(_c25, (864, 2804), _c25)
except Exception:
    pass
layout["IO:00_AM_GMT+OI:O0"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_06_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-8/26_icon_More.png
try:
    _c26 = get_crop(26, 288, 156)
    canvas.paste(_c26, (1152, 2804), _c26)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_06_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-8/27_text_5.10.png
try:
    _c27 = get_crop(27, 89, 43)
    canvas.paste(_c27, (22, 17), _c27)
except Exception:
    pass
layout["5.10"] = [22, 17, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_06_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-8/28_text_1_576_events.png
try:
    _c28 = get_crop(28, 372, 103)
    canvas.paste(_c28, (54, 410), _c28)
except Exception:
    pass
layout["1,576_events"] = [54, 410, 426, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_06_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-8/29_text_430_Loudon_Rd_Concord_NH_USA.png
try:
    _c29 = get_crop(29, 642, 52)
    canvas.paste(_c29, (90, 1686), _c29)
except Exception:
    pass
layout["430_Loudon_Rd;_Concord,_N"] = [90, 1686, 732, 1738]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_06_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-8/30_clickable_Home.png
try:
    _c30 = get_crop(30, 288, 156)
    canvas.paste(_c30, (0, 2804), _c30)
except Exception:
    pass
layout["Home"] = [0, 2804, 288, 2960]
