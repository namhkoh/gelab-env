# page_id: page_eventbrite_47f784058c8444bd8017b372f0857efe_02
# screenshot: 2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-4.png
# step_index: 2/11
# task: Open Eventbrite. Explore local events scheduled for this weekend. Select the first event from the 'Science' category. Read details of the event.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# background
draw.rectangle([(0, 0), (1440, 2960)], fill=(250, 250, 252))

# top status bar
status_h = 80
draw.rectangle([(0, 0), (1440, status_h)], fill=(200, 200, 200))

# subtle top hairline under status bar
draw.line([(0, status_h), (1440, status_h)], fill=(210, 210, 210), width=1)

# header / search area (rounded white card behind search input)
search_left, search_top, search_right, search_h = 48, 72, 1392, 120
search_bottom = search_top + search_h
draw.rounded_rectangle(
    [(search_left, search_top), (search_right, search_bottom)],
    radius=14,
    fill=(255, 255, 255),
    outline=(230, 230, 230),
    width=1
)

# subtle divider below header/search region
draw.line([(48, search_bottom + 18), (1392, search_bottom + 18)], fill=(230, 230, 230), width=1)

# thin separator above filters row area (guides filters area without drawing pills)
filters_sep_y = 420
draw.line([(48, filters_sep_y), (1392, filters_sep_y)], fill=(235, 235, 235), width=1)

# "10,000 events" area separator (just structural divider)
draw.line([(48, 600), (1392, 600)], fill=(240, 240, 240), width=1)

# First event card background (image container)
card1_left, card1_top = 48, 676
card1_w, card1_h = 1344, 1091
card1_right = card1_left + card1_w
card1_bottom = card1_top + card1_h

# shadow for card 1
draw.rounded_rectangle(
    [(card1_left + 8, card1_top + 10), (card1_right + 8, card1_bottom + 10)],
    radius=24,
    fill=(235, 235, 235)
)

# card 1 background
draw.rounded_rectangle(
    [(card1_left, card1_top), (card1_right, card1_bottom)],
    radius=20,
    fill=(255, 255, 255),
    outline=(220, 220, 220),
    width=2
)

# subtle divider below first card content area
draw.line([(48, card1_bottom + 20), (1392, card1_bottom + 20)], fill=(235, 235, 235), width=1)

# Small content background strip for metadata area beneath card1 (structural only)
meta_strip_top = card1_bottom + 30
draw.rectangle([(48, meta_strip_top), (1392, meta_strip_top + 120)], fill=(250, 250, 250))

# Second event card background (image container)
card2_left, card2_top = 48, 1815
card2_w, card2_h = 1344, 1001
card2_right = card2_left + card2_w
card2_bottom = card2_top + card2_h

# shadow for card 2
draw.rounded_rectangle(
    [(card2_left + 8, card2_top + 10), (card2_right + 8, card2_bottom + 10)],
    radius=24,
    fill=(235, 235, 235)
)

# card 2 background
draw.rounded_rectangle(
    [(card2_left, card2_top), (card2_right, card2_bottom)],
    radius=20,
    fill=(255, 255, 255),
    outline=(220, 220, 220),
    width=2
)

# subtle divider below second card content area
draw.line([(48, card2_bottom + 20), (1392, card2_bottom + 20)], fill=(235, 235, 235), width=1)

# bottom navigation bar background
nav_h = 140
nav_top = 2960 - nav_h
draw.rectangle([(0, nav_top), (1440, 2960)], fill=(255, 255, 255))

# top border for bottom nav
draw.line([(0, nav_top), (1440, nav_top)], fill=(230, 230, 230), width=1)

# small center divider above bottom nav to separate content
draw.line([(48, card2_bottom + 40), (1392, card2_bottom + 40)], fill=(240, 240, 240), width=1)

# a few subtle vertical guide separators to structure content columns (non-intrusive)
col_xs = [48 + 360, 48 + 720, 48 + 1080]
for x in col_xs:
    draw.line([(x, card1_bottom + 30), (x, nav_top - 20)], fill=(250, 250, 250), width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_02_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-4/00_icon_Music.png
try:
    _c0 = get_crop(0, 187, 103)
    canvas.paste(_c0, (837, 410), _c0)
except Exception:
    pass
layout["Music"] = [837, 410, 1024, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_02_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-4/01_icon_Business.png
try:
    _c1 = get_crop(1, 241, 103)
    canvas.paste(_c1, (1036, 410), _c1)
except Exception:
    pass
layout["Business"] = [1036, 410, 1277, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_02_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-4/02_icon_Anytime.png
try:
    _c2 = get_crop(2, 400, 103)
    canvas.paste(_c2, (425, 410), _c2)
except Exception:
    pass
layout["Anytime"] = [425, 410, 825, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_02_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-4/03_icon_Filters.png
try:
    _c3 = get_crop(3, 359, 103)
    canvas.paste(_c3, (54, 410), _c3)
except Exception:
    pass
layout["Filters"] = [54, 410, 413, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_02_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-4/04_icon_Foo.png
try:
    _c4 = get_crop(4, 151, 110)
    canvas.paste(_c4, (1282, 406), _c4)
except Exception:
    pass
layout["Foo"] = [1282, 406, 1433, 516]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_02_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-4/05_icon_Favorite_button.png
try:
    _c5 = get_crop(5, 144, 144)
    canvas.paste(_c5, (1092, 1192), _c5)
except Exception:
    pass
layout["Favorite_button"] = [1092, 1192, 1236, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_02_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-4/06_icon_Favorite_button.png
try:
    _c6 = get_crop(6, 144, 144)
    canvas.paste(_c6, (1092, 2331), _c6)
except Exception:
    pass
layout["Favorite_button"] = [1092, 2331, 1236, 2475]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_02_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-4/07_icon_BOTTLEROCK_Preferred_Shuttle_Bus_From_SA.png
try:
    _c7 = get_crop(7, 1344, 1091)
    canvas.paste(_c7, (48, 676), _c7)
except Exception:
    pass
layout["BOTTLEROCK_Preferred_Shut"] = [48, 676, 1392, 1767]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_02_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-4/08_icon_Overflow_menu_button.png
try:
    _c8 = get_crop(8, 144, 144)
    canvas.paste(_c8, (1236, 1192), _c8)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 1192, 1380, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_02_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-4/09_icon_Overflow_menu_button.png
try:
    _c9 = get_crop(9, 144, 144)
    canvas.paste(_c9, (1236, 2331), _c9)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 2331, 1380, 2475]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_02_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-4/10_icon_Juneteenth_Decades_Festival_Fillmorel_Mu.png
try:
    _c10 = get_crop(10, 1344, 1001)
    canvas.paste(_c10, (48, 1815), _c10)
except Exception:
    pass
layout["Juneteenth_Decades_Festiv"] = [48, 1815, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_02_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-4/11_icon_7.57.png
try:
    _c11 = get_crop(11, 128, 113)
    canvas.paste(_c11, (54, 116), _c11)
except Exception:
    pass
layout["7.57"] = [54, 116, 182, 229]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_02_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-4/12_icon_7.57.png
try:
    _c12 = get_crop(12, 63, 63)
    canvas.paste(_c12, (179, 0), _c12)
except Exception:
    pass
layout["7.57"] = [179, 0, 242, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_02_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-4/13_icon_Search_forae.png
try:
    _c13 = get_crop(13, 69, 63)
    canvas.paste(_c13, (307, 0), _c13)
except Exception:
    pass
layout["Search_forae"] = [307, 0, 376, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_02_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-4/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 55, 63)
    canvas.paste(_c14, (246, 0), _c14)
except Exception:
    pass
layout["icon_14"] = [246, 0, 301, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_02_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-4/15_icon_7.57.png
try:
    _c15 = get_crop(15, 62, 64)
    canvas.paste(_c15, (113, 0), _c15)
except Exception:
    pass
layout["7.57"] = [113, 0, 175, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_02_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-4/16_icon_Search_forae.png
try:
    _c16 = get_crop(16, 1344, 191)
    canvas.paste(_c16, (48, 72), _c16)
except Exception:
    pass
layout["Search_forae"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_02_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-4/17_icon_icon_17.png
try:
    _c17 = get_crop(17, 93, 59)
    canvas.paste(_c17, (1208, 0), _c17)
except Exception:
    pass
layout["icon_17"] = [1208, 0, 1301, 59]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_02_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-4/18_icon_icon_18.png
try:
    _c18 = get_crop(18, 66, 59)
    canvas.paste(_c18, (1314, 0), _c18)
except Exception:
    pass
layout["icon_18"] = [1314, 0, 1380, 59]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_02_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-4/19_icon_To_bottlerock_2024.png
try:
    _c19 = get_crop(19, 1344, 1091)
    canvas.paste(_c19, (48, 676), _c19)
except Exception:
    pass
layout["To_bottlerock_2024"] = [48, 676, 1392, 1767]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_02_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-4/20_icon_Juneteenth_Decades_Festival_Fillmorel_Mu.png
try:
    _c20 = get_crop(20, 288, 156)
    canvas.paste(_c20, (864, 2804), _c20)
except Exception:
    pass
layout["Juneteenth_Decades_Festiv"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_02_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-4/21_icon_San_Francisco.png
try:
    _c21 = get_crop(21, 536, 144)
    canvas.paste(_c21, (0, 259), _c21)
except Exception:
    pass
layout["San_Francisco"] = [0, 259, 536, 403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_02_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-4/22_icon_Search_forae.png
try:
    _c22 = get_crop(22, 53, 61)
    canvas.paste(_c22, (383, 2), _c22)
except Exception:
    pass
layout["Search_forae"] = [383, 2, 436, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_02_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-4/23_icon_TI_00AM_PDT.png
try:
    _c23 = get_crop(23, 288, 156)
    canvas.paste(_c23, (288, 2804), _c23)
except Exception:
    pass
layout["TI:00AM_PDT"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_02_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-4/24_icon_7.57.png
try:
    _c24 = get_crop(24, 101, 63)
    canvas.paste(_c24, (7, 0), _c24)
except Exception:
    pass
layout["7.57"] = [7, 0, 108, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_02_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-4/25_icon_Juneteenth_Decades_Festival_Fillmorel_Mu.png
try:
    _c25 = get_crop(25, 288, 156)
    canvas.paste(_c25, (576, 2804), _c25)
except Exception:
    pass
layout["Juneteenth_Decades_Festiv"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_02_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-4/26_icon_Promoted.png
try:
    _c26 = get_crop(26, 244, 67)
    canvas.paste(_c26, (84, 1659), _c26)
except Exception:
    pass
layout["Promoted"] = [84, 1659, 328, 1726]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_02_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-4/27_icon_Juneteenth_Decades_Festival_Fillmorel_Mu.png
try:
    _c27 = get_crop(27, 288, 156)
    canvas.paste(_c27, (1152, 2804), _c27)
except Exception:
    pass
layout["Juneteenth_Decades_Festiv"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_02_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-4/28_icon_Sat_Jun_15.png
try:
    _c28 = get_crop(28, 288, 156)
    canvas.paste(_c28, (0, 2804), _c28)
except Exception:
    pass
layout["Sat,_Jun_15"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_02_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-4/29_icon_icon_29.png
try:
    _c29 = get_crop(29, 39, 60)
    canvas.paste(_c29, (1275, 0), _c29)
except Exception:
    pass
layout["icon_29"] = [1275, 0, 1314, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_02_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-4/30_text_10_000_events.png
try:
    _c30 = get_crop(30, 359, 103)
    canvas.paste(_c30, (54, 410), _c30)
except Exception:
    pass
layout["10,000_events"] = [54, 410, 413, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_02_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-4/31_text_Westwood_Pickup_Location.png
try:
    _c31 = get_crop(31, 531, 56)
    canvas.paste(_c31, (92, 1600), _c31)
except Exception:
    pass
layout["Westwood_(Pickup_Location"] = [92, 1600, 623, 1656]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_02_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-4/32_text_JINETEEN.png
try:
    _c32 = get_crop(32, 423, 113)
    canvas.paste(_c32, (523, 1825), _c32)
except Exception:
    pass
layout["JINETEEN"] = [523, 1825, 946, 1938]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_02_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-4/33_text_JUNE_15.png
try:
    _c33 = get_crop(33, 237, 81)
    canvas.paste(_c33, (1131, 1830), _c33)
except Exception:
    pass
layout["JUNE_15"] = [1131, 1830, 1368, 1911]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_02_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-4/34_text_Iortc.png
try:
    _c34 = get_crop(34, 109, 49)
    canvas.paste(_c34, (1158, 742), _c34)
except Exception:
    pass
layout["Iortc"] = [1158, 742, 1267, 791]
