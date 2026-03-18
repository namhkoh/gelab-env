# page_id: page_eventbrite_eb32c51543d749539b68e6c61ff72fb8_07
# screenshot: 2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-9.png
# step_index: 7/19
# task: Open Eventbrite. Set the city to San Francisco. Filter for events occurring between May 1st and May 15th under the category 'Music'. Select the first event and check the pricing options available.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Background fill
draw.rectangle([(0, 0), (1440, 2960)], fill="#f6f7f8")

# Status bar area (top ~64px)
draw.rectangle([(0, 0), (1440, 64)], fill="#bfbfc1")

# Subtle separator under status bar
draw.line([(24, 64), (1416, 64)], fill="#d6d6d8", width=1)

# Search/header area background shadow and panel (behind the detected search content)
# shadow
draw.rounded_rectangle([(28, 72+6), (1412, 263+6)], radius=12, fill="#e9e9ea")
# white search header panel
draw.rounded_rectangle([(28, 72), (1412, 263)], radius=12, fill="#ffffff", outline="#e6e6e8", width=1)

# Thin divider under the header/search area
draw.line([(48, 264), (1392, 264)], fill="#dcdde0", width=2)

# Location row background (subtle)
draw.rectangle([(0, 280), (1440, 336)], fill="#f6f7f8")
draw.line([(48, 336), (1392, 336)], fill="#e8e8ea", width=1)

# Filter Pills area background band (a faint panel behind filter pills)
# draw a very light horizontal band to suggest area for filter chips without drawing chips themselves
draw.rectangle([(0, 360), (1440, 460)], fill="#fbfdfe")
draw.line([(48, 460), (1392, 460)], fill="#e9e9eb", width=1)

# First event card background (rounded rect with subtle shadow)
card1_outer = (36, 596, 1404, 1288)  # outer shadow rect
card1_inner = (40, 592, 1400, 1284)  # actual card
draw.rounded_rectangle(card1_outer, radius=20, fill="#e9eaec")
draw.rounded_rectangle(card1_inner, radius=18, fill="#ffffff", outline="#e6e6e8", width=2)

# Thin divider below first card title area (separator)
draw.line([(48, 1300), (1392, 1300)], fill="#ececed", width=1)

# Second event card background (rounded rect with subtle shadow)
card2_outer = (36, 1720, 1404, 2364)
card2_inner = (40, 1716, 1400, 2360)
draw.rounded_rectangle(card2_outer, radius=20, fill="#e9eaec")
draw.rounded_rectangle(card2_inner, radius=18, fill="#ffffff", outline="#e6e6e8", width=2)

# Small "Promoted"/tag background placeholder area behind where label chips appear (do not draw text)
# place a soft rounded rectangle at left under first card area to hint tag background
draw.rounded_rectangle([(72, 1600), (200, 1660)], radius=8, fill="#f1f6f8")

# Separator line between cards
draw.line([(48, 2364), (1392, 2364)], fill="#ececed", width=1)

# Content area backdrop band for the list (slight tint)
draw.rectangle([(0, 1284), (1440, 1716)], fill="#fafbfc")

# Bottom navigation bar background and top divider
nav_top = 2820
draw.line([(0, nav_top), (1440, nav_top)], fill="#e6e6e8", width=1)
draw.rectangle([(0, nav_top), (1440, 2960)], fill="#ffffff")

# subtle shadow above bottom nav
draw.rectangle([(0, nav_top-6), (1440, nav_top)], fill="#fafafa")

# final subtle page edge lines
draw.line([(24, 2960-36), (1416, 2960-36)], fill="#f1f2f3", width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_07_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-9/00_icon_Music.png
try:
    _c0 = get_crop(0, 187, 103)
    canvas.paste(_c0, (837, 410), _c0)
except Exception:
    pass
layout["Music"] = [837, 410, 1024, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_07_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-9/01_icon_Business.png
try:
    _c1 = get_crop(1, 241, 103)
    canvas.paste(_c1, (1036, 410), _c1)
except Exception:
    pass
layout["Business"] = [1036, 410, 1277, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_07_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-9/02_icon_Anytime.png
try:
    _c2 = get_crop(2, 400, 103)
    canvas.paste(_c2, (425, 410), _c2)
except Exception:
    pass
layout["Anytime"] = [425, 410, 825, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_07_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-9/03_icon_Filters.png
try:
    _c3 = get_crop(3, 359, 103)
    canvas.paste(_c3, (54, 410), _c3)
except Exception:
    pass
layout["Filters"] = [54, 410, 413, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_07_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-9/04_icon_Foo.png
try:
    _c4 = get_crop(4, 151, 110)
    canvas.paste(_c4, (1282, 406), _c4)
except Exception:
    pass
layout["Foo"] = [1282, 406, 1433, 516]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_07_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-9/05_icon_Favorite_button.png
try:
    _c5 = get_crop(5, 144, 144)
    canvas.paste(_c5, (1092, 1192), _c5)
except Exception:
    pass
layout["Favorite_button"] = [1092, 1192, 1236, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_07_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-9/06_icon_Favorite_button.png
try:
    _c6 = get_crop(6, 144, 144)
    canvas.paste(_c6, (1092, 2331), _c6)
except Exception:
    pass
layout["Favorite_button"] = [1092, 2331, 1236, 2475]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_07_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-9/07_icon_Overflow_menu_button.png
try:
    _c7 = get_crop(7, 144, 144)
    canvas.paste(_c7, (1236, 1192), _c7)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 1192, 1380, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_07_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-9/08_icon_Overflow_menu_button.png
try:
    _c8 = get_crop(8, 144, 144)
    canvas.paste(_c8, (1236, 2331), _c8)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 2331, 1380, 2475]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_07_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-9/09_icon_BOTTLEROCK_Preferred_Shuttle_Bus_From_SA.png
try:
    _c9 = get_crop(9, 1344, 1091)
    canvas.paste(_c9, (48, 676), _c9)
except Exception:
    pass
layout["BOTTLEROCK_Preferred_Shut"] = [48, 676, 1392, 1767]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_07_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-9/10_icon_Juneteenth_Decades_Festival_Fillmorel_Mu.png
try:
    _c10 = get_crop(10, 1344, 1001)
    canvas.paste(_c10, (48, 1815), _c10)
except Exception:
    pass
layout["Juneteenth_Decades_Festiv"] = [48, 1815, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_07_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-9/11_icon_7.47.png
try:
    _c11 = get_crop(11, 128, 113)
    canvas.paste(_c11, (54, 116), _c11)
except Exception:
    pass
layout["7.47"] = [54, 116, 182, 229]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_07_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-9/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 53, 64)
    canvas.paste(_c12, (1151, 0), _c12)
except Exception:
    pass
layout["icon_12"] = [1151, 0, 1204, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_07_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-9/13_icon_7.47.png
try:
    _c13 = get_crop(13, 62, 63)
    canvas.paste(_c13, (179, 0), _c13)
except Exception:
    pass
layout["7.47"] = [179, 0, 241, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_07_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-9/14_icon_Search_forae.png
try:
    _c14 = get_crop(14, 69, 63)
    canvas.paste(_c14, (307, 0), _c14)
except Exception:
    pass
layout["Search_forae"] = [307, 0, 376, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_07_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-9/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 95, 61)
    canvas.paste(_c15, (1212, 0), _c15)
except Exception:
    pass
layout["icon_15"] = [1212, 0, 1307, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_07_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-9/16_icon_icon_16.png
try:
    _c16 = get_crop(16, 54, 63)
    canvas.paste(_c16, (246, 0), _c16)
except Exception:
    pass
layout["icon_16"] = [246, 0, 300, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_07_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-9/17_icon_7.47.png
try:
    _c17 = get_crop(17, 62, 64)
    canvas.paste(_c17, (113, 0), _c17)
except Exception:
    pass
layout["7.47"] = [113, 0, 175, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_07_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-9/18_icon_Search_forae.png
try:
    _c18 = get_crop(18, 1344, 191)
    canvas.paste(_c18, (48, 72), _c18)
except Exception:
    pass
layout["Search_forae"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_07_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-9/19_icon_icon_19.png
try:
    _c19 = get_crop(19, 60, 59)
    canvas.paste(_c19, (1316, 0), _c19)
except Exception:
    pass
layout["icon_19"] = [1316, 0, 1376, 59]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_07_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-9/20_icon_To_bottlerock_2024.png
try:
    _c20 = get_crop(20, 1344, 1091)
    canvas.paste(_c20, (48, 676), _c20)
except Exception:
    pass
layout["To_bottlerock_2024"] = [48, 676, 1392, 1767]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_07_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-9/21_icon_Juneteenth_Decades_Festival_Fillmorel_Mu.png
try:
    _c21 = get_crop(21, 288, 156)
    canvas.paste(_c21, (864, 2804), _c21)
except Exception:
    pass
layout["Juneteenth_Decades_Festiv"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_07_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-9/22_icon_San_Francisco.png
try:
    _c22 = get_crop(22, 536, 144)
    canvas.paste(_c22, (0, 259), _c22)
except Exception:
    pass
layout["San_Francisco"] = [0, 259, 536, 403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_07_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-9/23_icon_TI_00AM_PDT.png
try:
    _c23 = get_crop(23, 288, 156)
    canvas.paste(_c23, (288, 2804), _c23)
except Exception:
    pass
layout["TI:00AM_PDT"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_07_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-9/24_icon_Search_forae.png
try:
    _c24 = get_crop(24, 53, 61)
    canvas.paste(_c24, (383, 2), _c24)
except Exception:
    pass
layout["Search_forae"] = [383, 2, 436, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_07_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-9/25_icon_Juneteenth_Decades_Festival_Fillmorel_Mu.png
try:
    _c25 = get_crop(25, 288, 156)
    canvas.paste(_c25, (576, 2804), _c25)
except Exception:
    pass
layout["Juneteenth_Decades_Festiv"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_07_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-9/26_icon_Promoted.png
try:
    _c26 = get_crop(26, 244, 67)
    canvas.paste(_c26, (84, 1659), _c26)
except Exception:
    pass
layout["Promoted"] = [84, 1659, 328, 1726]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_07_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-9/27_icon_7.47.png
try:
    _c27 = get_crop(27, 102, 63)
    canvas.paste(_c27, (7, 0), _c27)
except Exception:
    pass
layout["7.47"] = [7, 0, 109, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_07_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-9/28_icon_Juneteenth_Decades_Festival_Fillmorel_Mu.png
try:
    _c28 = get_crop(28, 288, 156)
    canvas.paste(_c28, (1152, 2804), _c28)
except Exception:
    pass
layout["Juneteenth_Decades_Festiv"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_07_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-9/29_icon_Sat_Jun_15.png
try:
    _c29 = get_crop(29, 288, 156)
    canvas.paste(_c29, (0, 2804), _c29)
except Exception:
    pass
layout["Sat,_Jun_15"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_07_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-9/30_text_10_000_events.png
try:
    _c30 = get_crop(30, 359, 103)
    canvas.paste(_c30, (54, 410), _c30)
except Exception:
    pass
layout["10,000_events"] = [54, 410, 413, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_07_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-9/31_text_Westwood_Pickup_Location.png
try:
    _c31 = get_crop(31, 531, 56)
    canvas.paste(_c31, (92, 1600), _c31)
except Exception:
    pass
layout["Westwood_(Pickup_Location"] = [92, 1600, 623, 1656]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_07_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-9/32_text_JINETEEN.png
try:
    _c32 = get_crop(32, 423, 113)
    canvas.paste(_c32, (523, 1825), _c32)
except Exception:
    pass
layout["JINETEEN"] = [523, 1825, 946, 1938]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_07_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-9/33_text_JUNE_15.png
try:
    _c33 = get_crop(33, 237, 81)
    canvas.paste(_c33, (1131, 1830), _c33)
except Exception:
    pass
layout["JUNE_15"] = [1131, 1830, 1368, 1911]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_07_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-9/34_text_Iortc.png
try:
    _c34 = get_crop(34, 109, 49)
    canvas.paste(_c34, (1158, 742), _c34)
except Exception:
    pass
layout["Iortc"] = [1158, 742, 1267, 791]
