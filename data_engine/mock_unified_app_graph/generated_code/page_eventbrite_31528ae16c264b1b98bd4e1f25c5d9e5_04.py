# page_id: page_eventbrite_31528ae16c264b1b98bd4e1f25c5d9e5_04
# screenshot: 2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-6.png
# step_index: 4/11
# task: Open Eventbrite. Search 'Fitness'. Filter for free events. Browse and select any 'Yoga' event. Note the location.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw overall background (dominant color is white - canvas already white, but ensure)
draw.rectangle([(0, 0), canvas.size], fill="#ffffff")

# Status bar area (top ~56px) - light grey to match screenshot status bar
status_bar_h = 56
draw.rectangle([(0, 0), (1440, status_bar_h)], fill="#cfcfcf")

# Header / toolbar area (below status bar)
header_top = status_bar_h
header_bottom = 148
draw.rectangle([(0, header_top), (1440, header_bottom)], fill="#ffffff")

# Thin divider under header
draw.line([(48, header_bottom), (1392, header_bottom)], fill="#e6e6e6", width=2)

# Secondary divider above filter chips row (to separate location/search area from chips)
chips_div_y = 360
draw.line([(48, chips_div_y), (1392, chips_div_y)], fill="#f0f0f0", width=1)

# Draw large content cards as rounded rectangles (leave space for pasted images/text)
card_x0 = 48
card_x1 = 1392
# Card 1 (first event)
card1_y0 = 500
card1_y1 = 980
# subtle shadow behind card 1
draw.rounded_rectangle([(card_x0+6, card1_y0+8), (card_x1+6, card1_y1+8)], radius=28, fill="#f4f4f4")
# card body
draw.rounded_rectangle([(card_x0, card1_y0), (card_x1, card1_y1)], radius=28, fill="#ffffff", outline="#e9e9e9", width=1)

# Image placeholder area inside card 1 (rounded)
img1_pad = 20
img1_top = card1_y0 + img1_pad
img1_bottom = img1_top + 360
draw.rounded_rectangle(
    [(card_x0 + img1_pad, img1_top), (card_x1 - img1_pad, img1_bottom)],
    radius=18,
    fill="#efefef",
    outline="#e6e6e6",
    width=1,
)

# Separator between card content and next card
sep_y = card1_y1 + 32
draw.line([(48, sep_y), (1392, sep_y)], fill="#f3f3f3", width=1)

# Card 2 (second event)
card2_y0 = sep_y + 24
card2_y1 = card2_y0 + 540
# subtle shadow behind card 2
draw.rounded_rectangle([(card_x0+6, card2_y0+8), (card_x1+6, card2_y1+8)], radius=28, fill="#f4f4f4")
# card body
draw.rounded_rectangle([(card_x0, card2_y0), (card_x1, card2_y1)], radius=28, fill="#ffffff", outline="#e9e9e9", width=1)

# Image placeholder area inside card 2 (darker to mimic poster/banner background)
img2_pad = 20
img2_top = card2_y0 + img2_pad
img2_bottom = img2_top + 360
draw.rounded_rectangle(
    [(card_x0 + img2_pad, img2_top), (card_x1 - img2_pad, img2_bottom)],
    radius=18,
    fill="#101010",
    outline="#e6e6e6",
    width=1,
)

# Small tag background area (for "Free" tag position) - subtle rounded rect (leave text for paste)
tag_w, tag_h = 88, 44
tag_x = card_x0 + 42
tag_y = img2_bottom + 18
draw.rounded_rectangle([(tag_x, tag_y), (tag_x + tag_w, tag_y + tag_h)], radius=8, fill="#eef6ee", outline="#dcefe0", width=1)

# Thin separator line between content and bottom area
bottom_sep_y = 2760
draw.line([(48, bottom_sep_y), (1392, bottom_sep_y)], fill="#e8e8e8", width=1)

# Bottom navigation bar background (bottom ~156px)
nav_top = 2804
draw.rectangle([(0, nav_top), (1440, 2960)], fill="#ffffff")
# nav top divider
draw.line([(48, nav_top), (1392, nav_top)], fill="#ececec", width=2)

# Subtle left edge guide line for the page (visual alignment helper)
draw.line([(48, status_bar_h), (48, 2760)], fill="#fafafa", width=1)

# Add a faint overall vertical rhythm lines (very subtle) to match clean whitespace feel
for y in (220, 420, 640, 860, 1080, 1300, 1520, 1740, 1960, 2180, 2400):
    draw.line([(60, y), (1380, y)], fill="#fbfbfb", width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_04_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-6/00_icon_Music.png
try:
    _c0 = get_crop(0, 187, 103)
    canvas.paste(_c0, (837, 410), _c0)
except Exception:
    pass
layout["Music"] = [837, 410, 1024, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_04_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-6/01_icon_Business.png
try:
    _c1 = get_crop(1, 241, 103)
    canvas.paste(_c1, (1036, 410), _c1)
except Exception:
    pass
layout["Business"] = [1036, 410, 1277, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_04_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-6/02_icon_Anytime.png
try:
    _c2 = get_crop(2, 400, 103)
    canvas.paste(_c2, (425, 410), _c2)
except Exception:
    pass
layout["Anytime"] = [425, 410, 825, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_04_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-6/03_icon_Filters.png
try:
    _c3 = get_crop(3, 359, 103)
    canvas.paste(_c3, (54, 410), _c3)
except Exception:
    pass
layout["Filters"] = [54, 410, 413, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_04_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-6/04_icon_Foo.png
try:
    _c4 = get_crop(4, 149, 110)
    canvas.paste(_c4, (1282, 406), _c4)
except Exception:
    pass
layout["Foo"] = [1282, 406, 1431, 516]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_04_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-6/05_icon_Favorite_button.png
try:
    _c5 = get_crop(5, 144, 144)
    canvas.paste(_c5, (1092, 1192), _c5)
except Exception:
    pass
layout["Favorite_button"] = [1092, 1192, 1236, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_04_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-6/06_icon_Overflow_menu_button.png
try:
    _c6 = get_crop(6, 144, 144)
    canvas.paste(_c6, (1236, 2252), _c6)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 2252, 1380, 2396]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_04_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-6/07_icon_APRIL_27_9_A.png
try:
    _c7 = get_crop(7, 144, 144)
    canvas.paste(_c7, (1092, 2252), _c7)
except Exception:
    pass
layout["APRIL_27_|_9_A"] = [1092, 2252, 1236, 2396]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_04_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-6/08_icon_Overflow_menu_button.png
try:
    _c8 = get_crop(8, 144, 144)
    canvas.paste(_c8, (1236, 1192), _c8)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 1192, 1380, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_04_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-6/09_icon_Foo.png
try:
    _c9 = get_crop(9, 144, 144)
    canvas.paste(_c9, (1248, 96), _c9)
except Exception:
    pass
layout["Foo"] = [1248, 96, 1392, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_04_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-6/10_icon_7.55.png
try:
    _c10 = get_crop(10, 124, 113)
    canvas.paste(_c10, (54, 114), _c10)
except Exception:
    pass
layout["7.55"] = [54, 114, 178, 227]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_04_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-6/11_icon_Fitness.png
try:
    _c11 = get_crop(11, 68, 64)
    canvas.paste(_c11, (308, 0), _c11)
except Exception:
    pass
layout["Fitness"] = [308, 0, 376, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_04_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-6/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 105, 61)
    canvas.paste(_c12, (1205, 0), _c12)
except Exception:
    pass
layout["icon_12"] = [1205, 0, 1310, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_04_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-6/13_icon_Fitness.png
try:
    _c13 = get_crop(13, 54, 64)
    canvas.paste(_c13, (246, 0), _c13)
except Exception:
    pass
layout["Fitness"] = [246, 0, 300, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_04_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-6/14_icon_7.55.png
try:
    _c14 = get_crop(14, 60, 63)
    canvas.paste(_c14, (181, 0), _c14)
except Exception:
    pass
layout["7.55"] = [181, 0, 241, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_04_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-6/15_icon_7.55.png
try:
    _c15 = get_crop(15, 61, 65)
    canvas.paste(_c15, (114, 0), _c15)
except Exception:
    pass
layout["7.55"] = [114, 0, 175, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_04_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-6/16_icon_San_Francisco.png
try:
    _c16 = get_crop(16, 536, 144)
    canvas.paste(_c16, (0, 259), _c16)
except Exception:
    pass
layout["San_Francisco"] = [0, 259, 536, 403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_04_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-6/17_icon_icon_17.png
try:
    _c17 = get_crop(17, 60, 61)
    canvas.paste(_c17, (1318, 0), _c17)
except Exception:
    pass
layout["icon_17"] = [1318, 0, 1378, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_04_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-6/18_icon_Fitness.png
try:
    _c18 = get_crop(18, 1344, 191)
    canvas.paste(_c18, (48, 72), _c18)
except Exception:
    pass
layout["Fitness"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_04_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-6/19_icon_icon_19.png
try:
    _c19 = get_crop(19, 51, 61)
    canvas.paste(_c19, (384, 3), _c19)
except Exception:
    pass
layout["icon_19"] = [384, 3, 435, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_04_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-6/20_icon_Wed_May_1_._I_0O_PM_EDT.png
try:
    _c20 = get_crop(20, 1344, 1012)
    canvas.paste(_c20, (48, 676), _c20)
except Exception:
    pass
layout["Wed,_May_1_._I:0O_PM_EDT"] = [48, 676, 1392, 1688]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_04_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-6/21_icon_Mastering_Work-Life_Harmony_From_Chaos_t.png
try:
    _c21 = get_crop(21, 288, 156)
    canvas.paste(_c21, (576, 2804), _c21)
except Exception:
    pass
layout["Mastering_Work-Life_Harmo"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_04_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-6/22_icon_Apr_27_9_00_AM_EDT.png
try:
    _c22 = get_crop(22, 288, 156)
    canvas.paste(_c22, (288, 2804), _c22)
except Exception:
    pass
layout["Apr_27_+_9:00_AM_EDT"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_04_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-6/23_icon_FROM_CHAOS_TO_CALM.png
try:
    _c23 = get_crop(23, 1344, 1080)
    canvas.paste(_c23, (48, 1736), _c23)
except Exception:
    pass
layout["FROM_CHAOS_TO_CALM"] = [48, 1736, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_04_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-6/24_icon_Mastering_Work-Life_Harmony_From_Chaos_t.png
try:
    _c24 = get_crop(24, 288, 156)
    canvas.paste(_c24, (1152, 2804), _c24)
except Exception:
    pass
layout["Mastering_Work-Life_Harmo"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_04_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-6/25_icon_Mastering_Work-Life_Harmony_From_Chaos_t.png
try:
    _c25 = get_crop(25, 288, 156)
    canvas.paste(_c25, (864, 2804), _c25)
except Exception:
    pass
layout["Mastering_Work-Life_Harmo"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_04_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-6/26_text_7.55.png
try:
    _c26 = get_crop(26, 92, 43)
    canvas.paste(_c26, (22, 17), _c26)
except Exception:
    pass
layout["7.55"] = [22, 17, 114, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_04_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-6/27_text_1_365_events.png
try:
    _c27 = get_crop(27, 359, 103)
    canvas.paste(_c27, (54, 410), _c27)
except Exception:
    pass
layout["1,365_events"] = [54, 410, 413, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_04_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-6/28_text_Free.png
try:
    _c28 = get_crop(28, 80, 36)
    canvas.paste(_c28, (117, 2452), _c28)
except Exception:
    pass
layout["Free"] = [117, 2452, 197, 2488]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_04_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-6/29_text_Mastering_Work-Life_Harmony_From_Chaos_t.png
try:
    _c29 = get_crop(29, 1344, 1080)
    canvas.paste(_c29, (48, 1736), _c29)
except Exception:
    pass
layout["Mastering_Work-Life_Harmo"] = [48, 1736, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_04_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-6/30_text_Calm.png
try:
    _c30 = get_crop(30, 151, 61)
    canvas.paste(_c30, (92, 2596), _c30)
except Exception:
    pass
layout["Calm"] = [92, 2596, 243, 2657]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_04_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-6/31_text_Apr_27_9_00_AM_EDT.png
try:
    _c31 = get_crop(31, 288, 156)
    canvas.paste(_c31, (288, 2804), _c31)
except Exception:
    pass
layout["Apr_27_+_9:00_AM_EDT"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_04_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-6/32_text_Online.png
try:
    _c32 = get_crop(32, 131, 50)
    canvas.paste(_c32, (90, 2745), _c32)
except Exception:
    pass
layout["Online"] = [90, 2745, 221, 2795]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_04_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-6/33_clickable_Home.png
try:
    _c33 = get_crop(33, 288, 156)
    canvas.paste(_c33, (0, 2804), _c33)
except Exception:
    pass
layout["Home"] = [0, 2804, 288, 2960]
