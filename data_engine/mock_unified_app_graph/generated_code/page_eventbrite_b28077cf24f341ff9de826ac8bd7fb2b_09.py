# page_id: page_eventbrite_b28077cf24f341ff9de826ac8bd7fb2b_09
# screenshot: 2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-11.png
# step_index: 9/16
# task: Open Eventbrite. Explore 'Wellness' events in Washington. Filter to only show free events. Add the first non-promoted event to favorite and follow the organizer.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Fill overall background (light off-white)
draw.rectangle((0, 0, 1440, 2960), fill=(249, 250, 252))

# Status bar area (top ~96px) - subtle gray
status_h = 96
draw.rectangle((0, 0, 1440, status_h), fill=(206, 208, 210))
# thin divider under status bar
draw.line((0, status_h, 1440, status_h), fill=(222, 223, 224), width=1)

# Header / toolbar area (search/title area) - keep mostly same as background but ensure a divider
header_top = status_h
header_bottom = 200
draw.rectangle((0, header_top, 1440, header_bottom), fill=(249, 250, 252))
# subtle bottom divider under header
draw.line((24, header_bottom, 1440-24, header_bottom), fill=(230, 231, 233), width=1)

# Separator under filters / location row
# (leave the actual filter pill shapes blank since they'll be pasted on top)
filters_row_top = 380
filters_row_bottom = 460
draw.rectangle((0, filters_row_bottom, 1440, filters_row_bottom+1), fill=(236, 237, 239))

# First event card background (rounded white card with subtle shadow)
card1_x0, card1_y0 = 48, 620
card1_x1, card1_y1 = 48 + 1344, 1860  # covers image + title region area
# shadow
shadow_offset = 10
draw.rounded_rectangle(
    (card1_x0 + shadow_offset, card1_y0 + shadow_offset, card1_x1 + shadow_offset, card1_y1 + shadow_offset),
    radius=28, fill=(235, 237, 240)
)
# card
draw.rounded_rectangle((card1_x0, card1_y0, card1_x1, card1_y1), radius=24, fill=(255, 255, 255), outline=(236, 237, 240))

# Divider between first and second card area (subtle)
divider_y = card1_y1 + 20
draw.line((48, divider_y, 1392, divider_y), fill=(242, 243, 244), width=1)

# Second event card background (rounded white card with subtle shadow)
card2_x0, card2_y0 = 48, 1700
card2_x1, card2_y1 = 48 + 1344, 2860
# shadow
draw.rounded_rectangle(
    (card2_x0 + shadow_offset, card2_y0 + shadow_offset, card2_x1 + shadow_offset, card2_y1 + shadow_offset),
    radius=28, fill=(235, 237, 240)
)
# card
draw.rounded_rectangle((card2_x0, card2_y0, card2_x1, card2_y1), radius=24, fill=(255, 255, 255), outline=(236, 237, 240))

# Thin separators / content dividers in content area
draw.line((48, 1560, 1392, 1560), fill=(245, 246, 247), width=1)
draw.line((48, 2100, 1392, 2100), fill=(245, 246, 247), width=1)

# Bottom navigation bar background and top border
nav_top = 2840
draw.rectangle((0, nav_top, 1440, 2960), fill=(255, 255, 255))
draw.line((0, nav_top, 1440, nav_top), fill=(230, 231, 233), width=1)

# Small subtle left/right margins guideline (non-intrusive, very light)
draw.line((24, header_bottom + 12, 24, nav_top - 12), fill=(249, 250, 251), width=1)
draw.line((1440-24, header_bottom + 12, 1440-24, nav_top - 12), fill=(249, 250, 251), width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_09_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-11/00_icon_Music.png
try:
    _c0 = get_crop(0, 187, 103)
    canvas.paste(_c0, (837, 410), _c0)
except Exception:
    pass
layout["Music"] = [837, 410, 1024, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_09_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-11/01_icon_Business.png
try:
    _c1 = get_crop(1, 241, 103)
    canvas.paste(_c1, (1036, 410), _c1)
except Exception:
    pass
layout["Business"] = [1036, 410, 1277, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_09_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-11/02_icon_Anytime.png
try:
    _c2 = get_crop(2, 400, 103)
    canvas.paste(_c2, (425, 410), _c2)
except Exception:
    pass
layout["Anytime"] = [425, 410, 825, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_09_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-11/03_icon_Filters.png
try:
    _c3 = get_crop(3, 359, 103)
    canvas.paste(_c3, (54, 410), _c3)
except Exception:
    pass
layout["Filters"] = [54, 410, 413, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_09_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-11/04_icon_ICDICMA.png
try:
    _c4 = get_crop(4, 144, 144)
    canvas.paste(_c4, (1092, 2252), _c4)
except Exception:
    pass
layout["ICDICMA"] = [1092, 2252, 1236, 2396]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_09_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-11/05_icon_Foo.png
try:
    _c5 = get_crop(5, 150, 110)
    canvas.paste(_c5, (1282, 406), _c5)
except Exception:
    pass
layout["Foo"] = [1282, 406, 1432, 516]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_09_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-11/06_icon_Favorite_button.png
try:
    _c6 = get_crop(6, 144, 144)
    canvas.paste(_c6, (1092, 1192), _c6)
except Exception:
    pass
layout["Favorite_button"] = [1092, 1192, 1236, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_09_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-11/07_icon_ICDICMA.png
try:
    _c7 = get_crop(7, 144, 144)
    canvas.paste(_c7, (1236, 2252), _c7)
except Exception:
    pass
layout["ICDICMA"] = [1236, 2252, 1380, 2396]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_09_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-11/08_icon_Overflow_menu_button.png
try:
    _c8 = get_crop(8, 144, 144)
    canvas.paste(_c8, (1236, 1192), _c8)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 1192, 1380, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_09_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-11/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 53, 66)
    canvas.paste(_c9, (1151, 0), _c9)
except Exception:
    pass
layout["icon_9"] = [1151, 0, 1204, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_09_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-11/10_icon_Close_current_screen.png
try:
    _c10 = get_crop(10, 144, 144)
    canvas.paste(_c10, (1248, 96), _c10)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 96, 1392, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_09_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-11/11_icon_4.44.png
try:
    _c11 = get_crop(11, 121, 113)
    canvas.paste(_c11, (56, 114), _c11)
except Exception:
    pass
layout["4.44"] = [56, 114, 177, 227]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_09_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-11/12_icon_Wellness.png
try:
    _c12 = get_crop(12, 65, 64)
    canvas.paste(_c12, (309, 0), _c12)
except Exception:
    pass
layout["Wellness"] = [309, 0, 374, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_09_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-11/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 100, 64)
    canvas.paste(_c13, (1212, 0), _c13)
except Exception:
    pass
layout["icon_13"] = [1212, 0, 1312, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_09_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-11/14_icon_KICKSTART_HEALTH_WELLNESS_EXPO_W.png
try:
    _c14 = get_crop(14, 1344, 1080)
    canvas.paste(_c14, (48, 1736), _c14)
except Exception:
    pass
layout["KICKSTART_HEALTH_&_WELLNE"] = [48, 1736, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_09_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-11/15_icon_4.44.png
try:
    _c15 = get_crop(15, 59, 64)
    canvas.paste(_c15, (114, 0), _c15)
except Exception:
    pass
layout["4.44"] = [114, 0, 173, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_09_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-11/16_icon_4.44.png
try:
    _c16 = get_crop(16, 58, 63)
    canvas.paste(_c16, (182, 0), _c16)
except Exception:
    pass
layout["4.44"] = [182, 0, 240, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_09_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-11/17_icon_icon_17.png
try:
    _c17 = get_crop(17, 50, 62)
    canvas.paste(_c17, (250, 1), _c17)
except Exception:
    pass
layout["icon_17"] = [250, 1, 300, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_09_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-11/18_icon_Wellness.png
try:
    _c18 = get_crop(18, 1344, 191)
    canvas.paste(_c18, (48, 72), _c18)
except Exception:
    pass
layout["Wellness"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_09_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-11/19_icon_icon_19.png
try:
    _c19 = get_crop(19, 55, 62)
    canvas.paste(_c19, (1318, 0), _c19)
except Exception:
    pass
layout["icon_19"] = [1318, 0, 1373, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_09_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-11/20_icon_Washington.png
try:
    _c20 = get_crop(20, 493, 144)
    canvas.paste(_c20, (0, 259), _c20)
except Exception:
    pass
layout["Washington"] = [0, 259, 493, 403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_09_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-11/21_icon_emrtnluthoa_King_Ja_Midliorial_Library_r.png
try:
    _c21 = get_crop(21, 1344, 1012)
    canvas.paste(_c21, (48, 676), _c21)
except Exception:
    pass
layout["emrtnluthoa_King_Ja_Midli"] = [48, 676, 1392, 1688]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_09_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-11/22_icon_9_O0_AM_EDT.png
try:
    _c22 = get_crop(22, 288, 156)
    canvas.paste(_c22, (576, 2804), _c22)
except Exception:
    pass
layout["9:O0_AM_EDT"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_09_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-11/23_icon_Wellness.png
try:
    _c23 = get_crop(23, 49, 62)
    canvas.paste(_c23, (384, 2), _c23)
except Exception:
    pass
layout["Wellness"] = [384, 2, 433, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_09_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-11/24_icon_KICKSTART_HEALTH_WELLNESS_EXPO_W.png
try:
    _c24 = get_crop(24, 1344, 1080)
    canvas.paste(_c24, (48, 1736), _c24)
except Exception:
    pass
layout["KICKSTART_HEALTH_&_WELLNE"] = [48, 1736, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_09_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-11/25_icon_4.44.png
try:
    _c25 = get_crop(25, 93, 63)
    canvas.paste(_c25, (12, 0), _c25)
except Exception:
    pass
layout["4.44"] = [12, 0, 105, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_09_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-11/26_icon_Rooftop_Terrace.png
try:
    _c26 = get_crop(26, 42, 61)
    canvas.paste(_c26, (285, 1583), _c26)
except Exception:
    pass
layout["Rooftop_Terrace"] = [285, 1583, 327, 1644]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_09_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-11/27_icon_ICDICMA.png
try:
    _c27 = get_crop(27, 288, 156)
    canvas.paste(_c27, (1152, 2804), _c27)
except Exception:
    pass
layout["ICDICMA"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_09_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-11/28_icon_KICKSTART_HEALTH_WELLNESS_EXPO_W.png
try:
    _c28 = get_crop(28, 288, 156)
    canvas.paste(_c28, (864, 2804), _c28)
except Exception:
    pass
layout["KICKSTART_HEALTH_&_WELLNE"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_09_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-11/29_icon_Capital_Turnaround.png
try:
    _c29 = get_crop(29, 42, 55)
    canvas.paste(_c29, (285, 2725), _c29)
except Exception:
    pass
layout["Capital_Turnaround"] = [285, 2725, 327, 2780]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_09_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-11/30_icon_Promoted.png
try:
    _c30 = get_crop(30, 248, 65)
    canvas.paste(_c30, (82, 1581), _c30)
except Exception:
    pass
layout["Promoted"] = [82, 1581, 330, 1646]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_09_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-11/31_icon_Capital_Turnaround.png
try:
    _c31 = get_crop(31, 288, 156)
    canvas.paste(_c31, (288, 2804), _c31)
except Exception:
    pass
layout["Capital_Turnaround"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_09_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-11/32_text_1_981_events.png
try:
    _c32 = get_crop(32, 359, 103)
    canvas.paste(_c32, (54, 410), _c32)
except Exception:
    pass
layout["1,981_events"] = [54, 410, 413, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_09_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-11/33_text_BLACK_GIRL_HEALTH.png
try:
    _c33 = get_crop(33, 375, 66)
    canvas.paste(_c33, (108, 1777), _c33)
except Exception:
    pass
layout["BLACK_GIRL_HEALTH"] = [108, 1777, 483, 1843]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_09_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-11/34_text_Capital_Turnaround.png
try:
    _c34 = get_crop(34, 288, 156)
    canvas.paste(_c34, (0, 2804), _c34)
except Exception:
    pass
layout["Capital_Turnaround"] = [0, 2804, 288, 2960]
